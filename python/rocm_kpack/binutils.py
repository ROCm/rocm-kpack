import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile
from enum import Enum
from typing import Any

import msgpack


class BinaryType(Enum):
    """Type of bundled binary file."""

    STANDALONE = "standalone"  # .co files - directly in bundler format
    BUNDLED = "bundled"  # Executables/libraries with .hip_fatbin ELF section


class Toolchain:
    """Manages configuration of various toolchain locations.

    Tools are lazily found and cached on first access, so construction never fails.
    Only when a specific tool is accessed will it be searched for and validated.
    """

    def __init__(
        self,
        *,
        clang_offload_bundler: Path | None = None,
        objcopy: Path | None = None,
        readelf: Path | None = None,
    ):
        # Store explicit paths (may be None)
        self._clang_offload_bundler_path = clang_offload_bundler
        self._objcopy_path = objcopy
        self._readelf_path = readelf

        # Cached resolved paths
        self._clang_offload_bundler_cached: Path | None = None
        self._objcopy_cached: Path | None = None
        self._readelf_cached: Path | None = None

    @staticmethod
    def configure_argparse(p: argparse.ArgumentParser):
        p.add_argument(
            "--clang-offload-bundler", type=Path, help="Path to clang-offload-bundler"
        )

    @staticmethod
    def from_args(args: argparse.Namespace) -> "Toolchain":
        clang_offload_bundler: Path | None = args.clang_offload_bundler
        return Toolchain(clang_offload_bundler=clang_offload_bundler)

    def _validate_or_find(
        self, tool_file_name: str, explicit_path: Path | None
    ) -> Path:
        if explicit_path is None:
            found_path = shutil.which(tool_file_name)
            if found_path is None:
                raise OSError(f"Could not file tool '{tool_file_name}' on system path")
            explicit_path = Path(found_path)
        if not explicit_path.exists():
            raise OSError(
                f"Tool '{tool_file_name}' at path {explicit_path} does not exist"
            )
        return explicit_path

    @property
    def clang_offload_bundler(self) -> Path:
        """Get clang-offload-bundler path (lazy, cached)."""
        if self._clang_offload_bundler_cached is None:
            self._clang_offload_bundler_cached = self._validate_or_find(
                "clang-offload-bundler", self._clang_offload_bundler_path
            )
        return self._clang_offload_bundler_cached

    @property
    def objcopy(self) -> Path:
        """Get objcopy path (lazy, cached)."""
        if self._objcopy_cached is None:
            self._objcopy_cached = self._validate_or_find("objcopy", self._objcopy_path)
        return self._objcopy_cached

    @property
    def readelf(self) -> Path:
        """Get readelf path (lazy, cached)."""
        if self._readelf_cached is None:
            self._readelf_cached = self._validate_or_find("readelf", self._readelf_path)
        return self._readelf_cached

    def exec_capture_text(self, args: list[str | Path]):
        return subprocess.check_output(
            [str(a) for a in args], stderr=subprocess.STDOUT
        ).decode()

    def exec(self, args: list[str | Path]):
        # Use check_output to capture stderr in exceptions (discarding the output)
        subprocess.check_output([str(a) for a in args], stderr=subprocess.STDOUT)


class UnbundledContents:
    """Represents a directory of unbundled contents. This is a context manager that
    will optionally delete the contents on close.
    """

    def __init__(
        self,
        source_binary: "BundledBinary",
        dest_dir: Path,
        delete_on_close: bool,
        target_list: list[tuple[str, str]],
    ):
        self.source_binary = source_binary
        self.dest_dir = dest_dir
        self.delete_on_close = delete_on_close
        self.target_list = target_list

    def __enter__(self) -> "UnbundledContents":
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @property
    def file_names(self) -> list[str]:
        return [kv[1] for kv in self.target_list]

    def close(self):
        if self.delete_on_close and self.dest_dir.exists():
            shutil.rmtree(self.dest_dir)

    def __repr__(self):
        return f"UnbundledContents(dest_dir={self.dest_dir}, target_list={self.target_list})"


class BundledBinary:
    """Represents a bundled binary at some path.

    Supports two types of bundled binaries:
    1. STANDALONE - Files directly in clang-offload-bundler format (e.g., .co files)
    2. BUNDLED - ELF binaries containing .hip_fatbin section (executables, shared libraries)

    For BUNDLED binaries, the .hip_fatbin section is extracted and treated as
    a bundler-format input.
    """

    def __init__(self, file_path: Path, *, toolchain: Toolchain | None = None):
        # Initialize _temp_dir first to ensure cleanup works even if init fails
        self._temp_dir: Path | None = None  # For extracted .hip_fatbin sections
        self._has_concatenated_bundles: bool | None = None  # Cached detection result

        self.toolchain = toolchain or Toolchain()
        self.file_path = file_path
        self.binary_type = self._detect_binary_type()

    def unbundle(
        self, *, dest_dir: Path | None = None, delete_on_close: bool = True
    ) -> UnbundledContents:
        """Unbundles the binary, returning a context manager which can be used
        to hold the unbundled files open for as long as needed.
        """
        if dest_dir is None:
            dest_dir = Path(tempfile.TemporaryDirectory(delete=False).name)
        target_list = self._list_bundled_targets(self.file_path)
        contents = UnbundledContents(
            self, dest_dir, delete_on_close=delete_on_close, target_list=target_list
        )
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            self._unbundle(
                targets=[kv[0] for kv in target_list],
                outputs=[dest_dir / kv[1] for kv in target_list],
            )
        except:
            contents.close()
            raise
        return contents

    def _detect_binary_type(self) -> BinaryType:
        """Detect if this is a standalone bundler file or bundled ELF binary.

        Uses readelf to check for .hip_fatbin section to determine type.
        Files with .hip_fatbin section are BUNDLED (executables, libraries).
        Files without (or non-ELF files) are STANDALONE (.co files in bundler format).

        Returns:
            BinaryType indicating the file type

        Raises:
            RuntimeError: For unexpected errors during detection
        """
        try:
            result = subprocess.run(
                [str(self.toolchain.readelf), "-S", str(self.file_path.resolve())],
                capture_output=True,
                text=True,
                check=True,  # Raise CalledProcessError on non-zero exit
            )
            # readelf succeeded - this is an ELF file
            # Check for .hip_fatbin section
            if ".hip_fatbin" in result.stdout:
                return BinaryType.BUNDLED
            else:
                # ELF file without .hip_fatbin section
                return BinaryType.STANDALONE

        except subprocess.CalledProcessError:
            # readelf failed - likely not an ELF file
            # Assume STANDALONE (bundler format file like .co)
            return BinaryType.STANDALONE
        except Exception as e:
            # Unexpected error - fail fast
            raise RuntimeError(
                f"Unexpected error detecting binary type for {self.file_path}: {e}"
            )

    def _get_bundler_input(self) -> Path:
        """Get the file path to use as input to clang-offload-bundler.

        For STANDALONE files, returns the file path directly.
        For BUNDLED binaries, extracts the .hip_fatbin section to a temp file.

        Returns:
            Path to file in bundler format
        """
        if self.binary_type == BinaryType.STANDALONE:
            return self.file_path

        # Extract .hip_fatbin section from bundled binary
        if self._temp_dir is None:
            self._temp_dir = Path(tempfile.mkdtemp())

        fatbin_path = self._temp_dir / "fatbin.o"
        # Resolve to absolute paths for objcopy
        abs_file_path = self.file_path.resolve()
        abs_fatbin_path = fatbin_path.resolve()

        try:
            self.toolchain.exec(
                [
                    self.toolchain.objcopy,
                    "--dump-section",
                    f".hip_fatbin={abs_fatbin_path}",
                    abs_file_path,
                ]
            )
        except subprocess.CalledProcessError as e:
            # Include the actual stderr/stdout from objcopy
            error_output = e.output.decode() if e.output else "(no output)"
            raise RuntimeError(
                f"Failed to extract .hip_fatbin section from {self.file_path}. "
                f"objcopy exit code: {e.returncode}. Output: {error_output}"
            ) from e

        return fatbin_path

    def _detect_concatenated_bundles(self, fatbin_path: Path) -> bool:
        """Detect if the fatbin contains multiple concatenated bundles.

        rocRAND and other libraries using RDC (relocatable device code) have
        multiple __CLANG_OFFLOAD_BUNDLE__ blocks concatenated together.
        clang-offload-bundler only sees the first one, so we need to use our
        own parser for these cases.

        Args:
            fatbin_path: Path to extracted .hip_fatbin section

        Returns:
            True if multiple bundles detected, False otherwise
        """
        if self._has_concatenated_bundles is not None:
            return self._has_concatenated_bundles

        from rocm_kpack.ccob_parser import find_bundle_offsets

        data = fatbin_path.read_bytes()
        offsets = find_bundle_offsets(data)
        self._has_concatenated_bundles = len(offsets) > 1
        return self._has_concatenated_bundles

    def _list_bundled_targets(self, file_path: Path) -> list[tuple[str, str]]:
        """Returns a list of (target_name, file_name) for all bundles.

        For concatenated bundles (RDC libraries like rocRAND), this returns
        multiple entries per target, one for each bundle containing that target.
        The filenames are indexed (e.g., target_0.hsaco, target_1.hsaco).
        """
        bundler_input = self._get_bundler_input()

        # Check for concatenated bundles first - clang-offload-bundler can't handle these
        if self._detect_concatenated_bundles(bundler_input):
            return self._list_bundled_targets_concatenated(bundler_input)

        # Try clang-offload-bundler for single-bundle case
        try:
            lines = (
                self.toolchain.exec_capture_text(
                    [
                        self.toolchain.clang_offload_bundler,
                        "--list",
                        "--type=o",
                        f"--input={bundler_input}",
                    ]
                )
                .strip()
                .splitlines()
            )
        except subprocess.CalledProcessError as e:
            # Check if this is the known decompression bug with CCOB bundles
            # Issue: https://github.com/ROCm/llvm-project/issues/448
            # (stderr is merged into stdout by exec_capture_text)
            error_msg = (
                e.output.decode() if isinstance(e.output, bytes) else str(e.output)
            )

            if (
                "decompress" in error_msg.lower()
                or "src size is incorrect" in error_msg.lower()
            ):
                # Fall back to our CCOB parser
                return self._list_bundled_targets_with_ccob_parser(bundler_input)
            # Re-raise other errors
            raise

        def _map(target_name: str) -> str:
            if target_name.startswith("host"):
                return f"{target_name}.elf"
            elif target_name.startswith("hip"):
                return f"{target_name}.hsaco"
            raise ValueError(f"Unexpected unbundled target name {target_name}")

        return [
            (target_name, _map(target_name)) for target_name in lines if target_name
        ]

    def _list_bundled_targets_concatenated(
        self, fatbin_path: Path
    ) -> list[tuple[str, str]]:
        """List targets from concatenated bundles (RDC case).

        For each code object across all concatenated bundles, returns a
        (target_name, indexed_filename) tuple. This allows extracting all
        code objects, not just one per target.

        Args:
            fatbin_path: Path to extracted .hip_fatbin section

        Returns:
            List of (target_name, filename) tuples with indexed filenames
        """
        from rocm_kpack.ccob_parser import extract_all_code_objects

        data = fatbin_path.read_bytes()
        code_objects = extract_all_code_objects(data)

        result = []
        for i, obj in enumerate(code_objects):
            # Skip host entries
            if obj.target.startswith("host"):
                continue

            # Create indexed filename: hipv4-amdgcn-...-gfx1100_0.hsaco
            base_ext = ".hsaco" if obj.target.startswith("hip") else ".elf"
            filename = f"{obj.target}_{i}{base_ext}"
            result.append((obj.target, filename))

        return result

    def _list_bundled_targets_with_ccob_parser(
        self, fatbin_path: Path
    ) -> list[tuple[str, str]]:
        """Fallback: List targets using our CCOB parser when clang-offload-bundler fails.

        Args:
            fatbin_path: Path to extracted .hip_fatbin section or standalone bundle

        Returns:
            List of (target_name, filename) tuples
        """
        from rocm_kpack.ccob_parser import list_ccob_targets

        data = fatbin_path.read_bytes()
        targets = list_ccob_targets(data)

        def _map(target_name: str) -> str:
            if target_name.startswith("host"):
                return f"{target_name}.elf"
            elif target_name.startswith("hip"):
                return f"{target_name}.hsaco"
            raise ValueError(f"Unexpected unbundled target name {target_name}")

        return [
            (target_name, _map(target_name)) for target_name in targets if target_name
        ]

    def _unbundle(self, *, targets: list[str], outputs: list[Path]):
        """Unbundle targets from the binary.

        Args:
            targets: List of target names to unbundle
            outputs: List of output paths (must match length of targets)
        """
        if not targets:
            return

        bundler_input = self._get_bundler_input()

        # Check for concatenated bundles first - clang-offload-bundler can't handle these
        if self._detect_concatenated_bundles(bundler_input):
            self._unbundle_concatenated(bundler_input, outputs=outputs)
            return

        # Try clang-offload-bundler for single-bundle case
        try:
            args = [
                self.toolchain.clang_offload_bundler,
                "--unbundle",
                "--type=o",
                f"--input={bundler_input}",
                f"--targets={','.join(targets)}",
            ]
            for output in outputs:
                args.extend(["--output", output])
            self.toolchain.exec(args)
        except subprocess.CalledProcessError as e:
            # Check if this is the known decompression bug with CCOB bundles
            # Issue: https://github.com/ROCm/llvm-project/issues/448
            # (stderr is merged into stdout/output by exec)
            error_msg = (
                e.output.decode() if isinstance(e.output, bytes) else str(e.output)
            )

            if (
                "decompress" in error_msg.lower()
                or "src size is incorrect" in error_msg.lower()
            ):
                # Fall back to our CCOB parser
                self._unbundle_with_ccob_parser(
                    bundler_input, targets=targets, outputs=outputs
                )
            else:
                # Re-raise other errors
                raise

    def _unbundle_concatenated(self, fatbin_path: Path, *, outputs: list[Path]):
        """Unbundle all code objects from concatenated bundles (RDC case).

        Writes each code object to the corresponding output path. The outputs
        list should match the order returned by _list_bundled_targets_concatenated.

        Args:
            fatbin_path: Path to extracted .hip_fatbin section
            outputs: List of output paths for each code object
        """
        from rocm_kpack.ccob_parser import extract_all_code_objects

        data = fatbin_path.read_bytes()
        code_objects = extract_all_code_objects(data)

        # Filter out host entries to match the listing
        device_objects = [
            obj for obj in code_objects if not obj.target.startswith("host")
        ]

        if len(device_objects) != len(outputs):
            raise ValueError(
                f"Output count mismatch: {len(device_objects)} code objects "
                f"but {len(outputs)} output paths"
            )

        for obj, output_path in zip(device_objects, outputs):
            output_path.write_bytes(obj.data)

    def _unbundle_with_ccob_parser(
        self, fatbin_path: Path, *, targets: list[str], outputs: list[Path]
    ):
        """Fallback: Unbundle using our CCOB parser when clang-offload-bundler fails.

        Args:
            fatbin_path: Path to extracted .hip_fatbin section or standalone bundle
            targets: List of target names to unbundle
            outputs: List of output paths (must match length of targets)
        """
        from rocm_kpack.ccob_parser import parse_ccob_file

        if len(targets) != len(outputs):
            raise ValueError(
                f"targets and outputs must have same length: {len(targets)} != {len(outputs)}"
            )

        # Parse the CCOB bundle
        bundle = parse_ccob_file(fatbin_path)

        # Extract each requested target
        for target, output in zip(targets, outputs):
            code_obj = bundle.get_code_object(target)
            if code_obj is None:
                raise RuntimeError(f"Target {target} not found in bundle")

            output.write_bytes(code_obj)

    def list_bundles(self) -> list[str]:
        """List all architecture bundles in the binary.

        Returns:
            List of architecture strings (e.g., ['gfx1100', 'gfx1101'])
            Only returns GPU architectures, not host bundles.
        """
        target_list = self._list_bundled_targets(self.file_path)
        architectures = []
        for target_name, _ in target_list:
            # Extract architecture from target names like:
            # "hipv4-amdgcn-amd-amdhsa--gfx1100" -> "gfx1100"
            if target_name.startswith("hip"):
                parts = target_name.split("--")
                if len(parts) >= 2:
                    architectures.append(parts[-1])
        return architectures

    def create_host_only(self, output_path: Path) -> None:
        """Create a host-only version of the binary without GPU device code.

        Only supported for BUNDLED binaries (executables/libraries with .hip_fatbin).
        Removes the .hip_fatbin section and reclaims disk space using the ELF kpacker.

        Prerequisites:
            The input binary must have a `.rocm_kpack_ref` section added via
            add_kpack_ref_marker() before calling this method.

        Args:
            output_path: Path where host-only binary will be written

        Raises:
            RuntimeError: If operation fails or called on STANDALONE binary
        """
        if self.binary_type == BinaryType.STANDALONE:
            raise RuntimeError(
                f"create_host_only() not supported for STANDALONE binaries. "
                f"File {self.file_path} does not have a .hip_fatbin section to remove. "
                f"If you need to extract the host bundle from a .co file, use unbundle() instead."
            )

        # Use ELF offload kpacker (actually reclaims disk space)
        # Import here to avoid circular dependency
        from rocm_kpack.elf_offload_kpacker import kpack_offload_binary

        kpack_offload_binary(self.file_path, output_path, toolchain=self.toolchain)

    def remove_section_simple(self, output_path: Path, section_name: str) -> None:
        """Remove a section using objcopy (simple header removal, no space reclaimed).

        This is a simple section removal that only updates ELF headers without
        reclaiming disk space. Useful for benchmarking or comparison purposes.

        For production use with .hip_fatbin removal, use create_host_only() instead,
        which properly reclaims disk space.

        Args:
            output_path: Path where modified binary will be written
            section_name: Name of section to remove (e.g., ".hip_fatbin")

        Raises:
            RuntimeError: If objcopy fails
        """
        try:
            self.toolchain.exec(
                [
                    self.toolchain.objcopy,
                    "--remove-section",
                    section_name,
                    self.file_path,
                    output_path,
                ]
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to remove section {section_name} from {self.file_path}: {e}"
            )

    def cleanup(self) -> None:
        """Clean up temporary files created during operations."""
        if self._temp_dir and self._temp_dir.exists():
            shutil.rmtree(self._temp_dir)
            self._temp_dir = None

    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup()


def add_kpack_ref_marker(
    binary_path: Path,
    output_path: Path,
    kpack_search_paths: list[str],
    kernel_name: str,
    *,
    toolchain: Toolchain | None = None,
) -> None:
    """Add .rocm_kpack_ref marker section to a binary.

    The marker section contains a MessagePack structure pointing to kpack files
    and identifying the kernel name for TOC lookup.

    Args:
        binary_path: Path to input binary (ELF executable or shared library)
        output_path: Path where marked binary will be written
        kpack_search_paths: List of kpack file paths relative to binary location
        kernel_name: Kernel identifier for TOC lookup in kpack file
        toolchain: Toolchain instance (created if not provided)

    Raises:
        RuntimeError: If objcopy fails to add section
    """
    if toolchain is None:
        toolchain = Toolchain()

    # Create marker structure
    marker_data = {
        "kpack_search_paths": kpack_search_paths,
        "kernel_name": kernel_name,
    }

    # Serialize to MessagePack
    marker_bytes = msgpack.packb(marker_data, use_bin_type=True)

    # Write marker to temporary file and add section
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".marker") as f:
        f.write(marker_bytes)
        f.flush()

        # Add section to binary using objcopy
        # Use absolute paths to avoid objcopy path resolution issues
        abs_binary_path = binary_path.resolve()
        abs_output_path = output_path.resolve()
        abs_marker_file = Path(f.name).resolve()

        try:
            toolchain.exec(
                [
                    toolchain.objcopy,
                    "--add-section",
                    f".rocm_kpack_ref={abs_marker_file}",
                    abs_binary_path,
                    abs_output_path,
                ]
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to add .rocm_kpack_ref section to {binary_path}: {e}"
            )


def read_kpack_ref_marker(
    binary_path: Path,
    *,
    toolchain: Toolchain | None = None,
) -> dict[str, Any] | None:
    """Read .rocm_kpack_ref marker section from a binary.

    Args:
        binary_path: Path to binary with marker section
        toolchain: Toolchain instance (created if not provided)

    Returns:
        Marker data dictionary, or None if section doesn't exist

    Raises:
        RuntimeError: If readelf fails or section exists but cannot be read or parsed
    """
    if toolchain is None:
        toolchain = Toolchain()

    # Check if section exists using readelf
    result = subprocess.run(
        [str(toolchain.readelf), "-S", str(binary_path.resolve())],
        capture_output=True,
        text=True,
        check=True,
    )
    if ".rocm_kpack_ref" not in result.stdout:
        return None

    # Extract section to temporary file
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".marker") as f:
        abs_binary_path = binary_path.resolve()
        abs_marker_file = Path(f.name).resolve()

        try:
            toolchain.exec(
                [
                    toolchain.objcopy,
                    "--dump-section",
                    f".rocm_kpack_ref={abs_marker_file}",
                    abs_binary_path,
                ]
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to extract .rocm_kpack_ref section from {binary_path}: {e}"
            )

        # Read and parse MessagePack data
        try:
            with open(abs_marker_file, "rb") as marker_file:
                marker_bytes = marker_file.read()
                marker_data = msgpack.unpackb(marker_bytes, raw=False)
                return marker_data
        except Exception as e:
            raise RuntimeError(
                f"Failed to parse .rocm_kpack_ref marker data from {binary_path}: {e}"
            )


def get_section_vaddr(
    toolchain: Toolchain, binary_path: Path, section_name: str
) -> int | None:
    """
    Get the virtual address of a section in an ELF binary.

    Args:
        toolchain: Toolchain instance providing readelf
        binary_path: Path to ELF binary
        section_name: Name of section (e.g., ".custom_data", ".rocm_kpack_ref")

    Returns:
        Virtual address (sh_addr) of the section if it exists and has ALLOC flag,
        None otherwise.

    Note:
        Only returns addresses for sections with the ALLOC flag (A), which indicates
        they are mapped to memory at load time (part of a PT_LOAD segment).
    """
    try:
        # Run readelf to get section headers
        result = subprocess.run(
            [str(toolchain.readelf), "-S", str(binary_path)],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return None

    # Parse section headers
    # Format (two-line entries):
    # Line 1: [Nr] Name              Type             Address           Offset
    # Line 2:      Size              EntSize          Flags  Link  Info  Align
    lines = result.stdout.split("\n")
    for i, line in enumerate(lines):
        if section_name in line:
            parts = line.split()
            # Check if this is a section header line (starts with [Nr])
            if len(parts) >= 5 and parts[0].startswith("["):
                try:
                    # Address column is at index 3
                    vaddr = int(parts[3], 16)

                    # Check flags on the next line
                    if i + 1 < len(lines):
                        next_parts = lines[i + 1].split()
                        if len(next_parts) >= 3:
                            flags = next_parts[2]
                            # Only return address if section has ALLOC flag (A)
                            if "A" in flags:
                                return vaddr

                except (ValueError, IndexError):
                    continue

    return None


def get_section_size(
    toolchain: Toolchain, binary_path: Path, section_name: str
) -> int | None:
    """
    Get the size of a section in an ELF binary.

    Args:
        toolchain: Toolchain instance providing readelf
        binary_path: Path to ELF binary
        section_name: Name of section (e.g., ".hipFatBinSegment")

    Returns:
        Size (sh_size) of the section if found, None otherwise.
    """
    try:
        result = subprocess.run(
            [str(toolchain.readelf), "-S", str(binary_path)],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return None

    # Parse section headers
    # Format (two-line entries):
    # Line 1: [Nr] Name              Type             Address           Offset
    # Line 2:      Size              EntSize          Flags  Link  Info  Align
    lines = result.stdout.split("\n")
    for i, line in enumerate(lines):
        if section_name in line:
            parts = line.split()
            # Check if this is a section header line (starts with [Nr])
            if len(parts) >= 5 and parts[0].startswith("["):
                try:
                    # Size is on the next line, first column
                    if i + 1 < len(lines):
                        next_parts = lines[i + 1].split()
                        if len(next_parts) >= 1:
                            size = int(next_parts[0], 16)
                            return size
                except (ValueError, IndexError):
                    continue

    return None


def has_section(
    binary_path: Path,
    section_name: str,
    *,
    toolchain: Toolchain | None = None,
) -> bool:
    """Check if a binary has a specific section.

    Args:
        binary_path: Path to binary
        section_name: Name of section to check for (e.g., ".hip_fatbin", ".rocm_kpack_ref")
        toolchain: Toolchain instance (created if not provided)

    Returns:
        True if section exists, False otherwise

    Note:
        This function abstracts binary format tooling (readelf for ELF, etc.)
        to support cross-platform binary analysis.
    """
    if toolchain is None:
        toolchain = Toolchain()

    try:
        output = toolchain.exec_capture_text(
            [toolchain.readelf, "-S", str(binary_path)]
        )
        return section_name in output
    except Exception:
        return False


def get_section_type(
    binary_path: Path,
    section_name: str,
    *,
    toolchain: Toolchain | None = None,
) -> str | None:
    """Get the type of a section in a binary (e.g., PROGBITS, NOBITS).

    Args:
        binary_path: Path to binary
        section_name: Name of section (e.g., ".hip_fatbin")
        toolchain: Toolchain instance (created if not provided)

    Returns:
        Section type string (e.g., "PROGBITS", "NOBITS"), or None if section doesn't exist

    Note:
        This function abstracts binary format tooling (readelf for ELF, etc.)
        to support cross-platform binary analysis.
    """
    if toolchain is None:
        toolchain = Toolchain()

    try:
        output = toolchain.exec_capture_text(
            [toolchain.readelf, "-S", str(binary_path)]
        )

        # Parse section headers to find the type
        # Format: [Nr] Name              Type             Address           Offset
        for line in output.splitlines():
            if section_name in line:
                parts = line.split()
                # Check if this is a section header line (starts with [Nr])
                if len(parts) >= 3 and parts[0].startswith("["):
                    # Type is at index 2
                    return parts[2]

        return None

    except Exception:
        return None

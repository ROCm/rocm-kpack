"""Tests for ELF offload kpacker."""

import os
import subprocess
from pathlib import Path

from rocm_kpack.elf_offload_kpacker import kpack_offload_binary, ElfOffloadKpacker
from rocm_kpack import binutils


def test_kpack_fat_binary(
    tmp_path: Path, test_assets_dir: Path, toolchain: binutils.Toolchain
):
    """Test kpacking a fat binary with .hip_fatbin section."""
    # Use a multi-arch bundled binary
    input_binary = test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_multi.exe"
    marked_binary = tmp_path / "test_kernel_multi_marked.exe"
    output_binary = tmp_path / "test_kernel_multi_kpacked.exe"

    # Add .rocm_kpack_ref section first
    binutils.add_kpack_ref_marker(
        input_binary,
        marked_binary,
        kpack_search_paths=["test.kpack"],
        kernel_name="test_kernel",
        toolchain=toolchain,
    )

    # Get original size
    original_size = marked_binary.stat().st_size

    # Kpack
    result = kpack_offload_binary(
        marked_binary, output_binary, toolchain=toolchain, verbose=True
    )

    # Verify output exists
    assert output_binary.exists()

    # Verify size reduction
    new_size = output_binary.stat().st_size
    assert new_size < original_size
    assert result["removed"] > 0
    assert result["original_size"] == original_size
    assert result["new_size"] == new_size

    # Verify output is a valid ELF file
    readelf_result = subprocess.run(
        [toolchain.readelf, "-h", output_binary], capture_output=True, text=True
    )
    assert readelf_result.returncode == 0
    assert "ELF" in readelf_result.stdout

    # Verify .hip_fatbin section is gone (should be SHT_NULL or not listed)
    sections_result = subprocess.run(
        [toolchain.readelf, "-S", output_binary], capture_output=True, text=True
    )
    assert sections_result.returncode == 0
    # Either the section is marked as NULL or doesn't appear with size > 0
    if ".hip_fatbin" in sections_result.stdout:
        # If it appears, it should be NULL type
        assert "NULL" in sections_result.stdout

    # Verify executable permissions are preserved
    assert os.access(output_binary, os.X_OK), "Output binary should be executable"


def test_kpack_shared_library(
    tmp_path: Path, test_assets_dir: Path, toolchain: binutils.Toolchain
):
    """Test kpacking a shared library with .hip_fatbin section."""
    input_library = (
        test_assets_dir / "bundled_binaries/linux/cov5/libtest_kernel_single.so"
    )
    marked_library = tmp_path / "libtest_kernel_single_marked.so"
    output_library = tmp_path / "libtest_kernel_single_kpacked.so"

    # Add .rocm_kpack_ref section first
    binutils.add_kpack_ref_marker(
        input_library,
        marked_library,
        kpack_search_paths=["test.kpack"],
        kernel_name="test_kernel",
        toolchain=toolchain,
    )

    original_size = marked_library.stat().st_size

    result = kpack_offload_binary(
        marked_library, output_library, toolchain=toolchain, verbose=True
    )

    # Verify size reduction
    assert output_library.exists()
    assert output_library.stat().st_size < original_size
    assert result["removed"] > 0

    # Verify it's still a valid shared library
    readelf_result = subprocess.run(
        [toolchain.readelf, "-h", output_library], capture_output=True, text=True
    )
    assert readelf_result.returncode == 0
    assert "DYN (Shared object file)" in readelf_result.stdout


def test_kpack_host_only_binary(
    tmp_path: Path, test_assets_dir: Path, toolchain: binutils.Toolchain
):
    """Test kpacking a binary without .hip_fatbin (should just copy)."""
    input_binary = test_assets_dir / "bundled_binaries/linux/cov5/host_only.exe"
    marked_binary = tmp_path / "host_only_marked.exe"
    output_binary = tmp_path / "host_only_kpacked.exe"

    # Add .rocm_kpack_ref section first
    binutils.add_kpack_ref_marker(
        input_binary,
        marked_binary,
        kpack_search_paths=["test.kpack"],
        kernel_name="test_kernel",
        toolchain=toolchain,
    )

    original_size = marked_binary.stat().st_size

    result = kpack_offload_binary(
        marked_binary, output_binary, toolchain=toolchain, verbose=True
    )

    # Binary should be processed (mapping .rocm_kpack_ref adds overhead)
    assert output_binary.exists()
    # Mapping the section adds padding, so output may be larger
    assert (
        result["removed"] <= 0
    )  # No .hip_fatbin to remove, so "removed" is negative (size increased)
    assert result["original_size"] == original_size
    assert result["new_size"] == output_binary.stat().st_size


def test_kpacker_has_hip_fatbin(test_assets_dir: Path):
    """Test the has_hip_fatbin() method."""
    # Fat binary should have .hip_fatbin
    fat_binary = test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_multi.exe"
    kpacker = ElfOffloadKpacker(fat_binary)
    assert kpacker.has_hip_fatbin() is True

    # Host-only binary should not have .hip_fatbin
    host_only = test_assets_dir / "bundled_binaries/linux/cov5/host_only.exe"
    kpacker = ElfOffloadKpacker(host_only)
    assert kpacker.has_hip_fatbin() is False


def test_kpacker_calculate_removal_plan(test_assets_dir: Path):
    """Test the calculate_removal_plan() method."""
    fat_binary = test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_multi.exe"
    kpacker = ElfOffloadKpacker(fat_binary)

    plan = kpacker.calculate_removal_plan()

    # Verify plan structure
    assert "removal_size" in plan
    assert "removal_offset" in plan
    assert "removal_vaddr" in plan
    assert "sections_to_shift" in plan
    assert "phdrs_to_update" in plan

    # Verify plan makes sense
    assert plan["removal_size"] > 0
    assert plan["removal_offset"] > 0
    assert isinstance(plan["sections_to_shift"], list)
    assert isinstance(plan["phdrs_to_update"], list)


def test_integration_with_binutils_create_host_only(
    tmp_path: Path, test_assets_dir: Path, toolchain: binutils.Toolchain
):
    """Test that BundledBinary.create_host_only() uses the kpacker by default."""
    input_binary = (
        test_assets_dir / "bundled_binaries/linux/cov5/libtest_kernel_multi.so"
    )
    marked_binary = tmp_path / "libtest_kernel_multi_marked.so"
    output_binary = tmp_path / "libtest_kernel_multi_host_only.so"

    # Add .rocm_kpack_ref section first
    binutils.add_kpack_ref_marker(
        input_binary,
        marked_binary,
        kpack_search_paths=["test.kpack"],
        kernel_name="test_kernel",
        toolchain=toolchain,
    )

    original_size = marked_binary.stat().st_size

    # Create BundledBinary and call create_host_only (uses kpacker)
    bb = binutils.BundledBinary(marked_binary, toolchain=toolchain)
    bb.create_host_only(output_binary)

    # Verify size reduction (kpacker should actually reduce size)
    assert output_binary.exists()
    new_size = output_binary.stat().st_size
    assert new_size < original_size

    # Compare with simple objcopy path (should have minimal reduction)
    output_binary_objcopy = tmp_path / "libtest_kernel_multi_objcopy.so"
    bb.remove_section_simple(output_binary_objcopy, ".hip_fatbin")

    objcopy_size = output_binary_objcopy.stat().st_size
    # Kpacker should remove more bytes than simple objcopy
    kpacker_reduction = original_size - new_size
    objcopy_reduction = original_size - objcopy_size
    assert kpacker_reduction > objcopy_reduction


def test_kpacked_binary_compatible_with_objcopy(
    tmp_path: Path, test_assets_dir: Path, toolchain: binutils.Toolchain
):
    """Test that objcopy --add-section works on kpacked binaries."""
    input_binary = test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_multi.exe"
    marked_binary = tmp_path / "test_kernel_multi_marked.exe"
    output_binary = tmp_path / "test_kernel_multi_kpacked.exe"

    # Add .rocm_kpack_ref section first
    binutils.add_kpack_ref_marker(
        input_binary,
        marked_binary,
        kpack_search_paths=["test.kpack"],
        kernel_name="test_kernel",
        toolchain=toolchain,
    )

    # Neutralize the binary
    kpack_offload_binary(
        marked_binary, output_binary, toolchain=toolchain, verbose=True
    )

    # Verify kpacked binary exists
    assert output_binary.exists()

    # Create a dummy section content
    marker_file = tmp_path / "test_marker.bin"
    marker_file.write_text("test marker content")

    # Try to add a section using objcopy (this should work on valid ELF files)
    output_with_marker = tmp_path / "test_kernel_multi_with_marker.exe"
    try:
        subprocess.run(
            [
                toolchain.objcopy,
                "--add-section",
                f".test_marker={marker_file}",
                output_binary,
                output_with_marker,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise AssertionError(
            f"objcopy --add-section failed on kpacked binary. "
            f"This indicates the kpacker produced an invalid ELF file. "
            f"Error: {e.stderr}"
        )

    # Verify the marker was added
    readelf_result = subprocess.run(
        [toolchain.readelf, "-S", output_with_marker],
        capture_output=True,
        text=True,
        check=True,
    )
    assert (
        ".test_marker" in readelf_result.stdout
    ), "Marker section should be present in binary"

    # Verify the binary is still valid
    assert output_with_marker.exists()
    # Note: objcopy may re-layout the binary and remove padding, so size might not increase
    # The important thing is that it succeeded and the marker is present


def test_kpack_multi_wrapper_binary(
    tmp_path: Path, test_assets_dir: Path, toolchain: binutils.Toolchain
):
    """Test kpacking a binary with multiple __CudaFatBinaryWrapper structures.

    Binaries built with -fgpu-rdc (relocatable device code) have multiple
    fat binary wrappers in .hipFatBinSegment - one per translation unit.
    The kpacker must transform ALL wrappers, not just the first one.
    """
    input_library = (
        test_assets_dir / "bundled_binaries/linux/cov5/libtest_multi_wrapper.so"
    )
    marked_library = tmp_path / "libtest_multi_wrapper_marked.so"
    output_library = tmp_path / "libtest_multi_wrapper_kpacked.so"

    # Add .rocm_kpack_ref section first
    binutils.add_kpack_ref_marker(
        input_library,
        marked_library,
        kpack_search_paths=["test.kpack"],
        kernel_name="test_kernel",
        toolchain=toolchain,
    )

    original_size = marked_library.stat().st_size

    # Kpack the binary
    result = kpack_offload_binary(
        marked_library, output_library, toolchain=toolchain, verbose=True
    )

    # Verify output exists and size reduced
    assert output_library.exists()
    assert output_library.stat().st_size < original_size
    assert result["removed"] > 0

    # Verify .hipFatBinSegment has 2 wrappers (48 bytes = 2 * 24)
    sections_result = subprocess.run(
        [toolchain.readelf, "-S", output_library], capture_output=True, text=True
    )
    assert sections_result.returncode == 0

    # Parse section size to verify 2 wrappers
    # readelf output spans two lines per section, e.g.:
    #   [23] .hipFatBinSegment PROGBITS   0000000000008fa8  00005fa8
    #        0000000000000030  0000000000000000  WA       0     0     8
    lines = sections_result.stdout.split("\n")
    found_section = False
    for i, line in enumerate(lines):
        if ".hipFatBinSegment" in line:
            found_section = True
            # Size is on the next line (continuation)
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                # The size should be 0x30 = 48 bytes (2 wrappers * 24 bytes each)
                assert "0000000000000030" in next_line or "00000030" in next_line, (
                    f"Expected .hipFatBinSegment size 0x30 (48 bytes) for 2 wrappers, "
                    f"got continuation line: {next_line}"
                )
            break

    assert found_section, ".hipFatBinSegment section not found in output binary"

    # Verify BOTH wrappers have HIPK magic (not HIPF)
    # Read the .hipFatBinSegment section directly
    with open(output_library, "rb") as f:
        data = f.read()

    # Find .hipFatBinSegment offset by parsing readelf output
    # Format: [Nr] Name Type Addr Offset (on first line)
    #         Size EntSz Flags Link Info Align (on second line)
    section_offset = None
    for i, line in enumerate(lines):
        if ".hipFatBinSegment" in line:
            # Get hex values from header line (addr, offset)
            parts = line.split()
            hex_vals = [
                p
                for p in parts
                if all(c in "0123456789abcdef" for c in p.lower()) and len(p) >= 4
            ]
            if len(hex_vals) >= 2:
                section_offset = int(hex_vals[1], 16)  # Second hex is file offset
            break

    assert section_offset is not None, "Could not parse .hipFatBinSegment offset"

    # HIPK magic = 0x4B504948 = "HIPK" in little-endian
    HIPK_MAGIC = 0x4B504948
    import struct

    # Check wrapper 1 at offset 0
    wrapper1_magic = struct.unpack_from("<I", data, section_offset)[0]
    assert (
        wrapper1_magic == HIPK_MAGIC
    ), f"Wrapper 1 should have HIPK magic, got 0x{wrapper1_magic:08x}"

    # Check wrapper 2 at offset 24
    wrapper2_magic = struct.unpack_from("<I", data, section_offset + 24)[0]
    assert (
        wrapper2_magic == HIPK_MAGIC
    ), f"Wrapper 2 should have HIPK magic, got 0x{wrapper2_magic:08x}"

    # Verify it's still a valid shared library
    readelf_result = subprocess.run(
        [toolchain.readelf, "-h", output_library], capture_output=True, text=True
    )
    assert readelf_result.returncode == 0
    assert "DYN (Shared object file)" in readelf_result.stdout

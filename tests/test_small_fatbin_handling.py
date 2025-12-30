"""
Tests for handling binaries with small .hip_fatbin sections.

These tests verify the fix for https://github.com/ROCm/rocm-kpack/issues/1

The kpack tooling must gracefully handle edge cases where the .hip_fatbin
section is too small for efficient zero-paging.

Background:
-----------
The kpack artifact splitter uses "conservative zero-paging" to remove
device code from fat binaries. This works by:
1. Finding page-aligned regions within .hip_fatbin
2. Removing those pages from the file
3. Updating ELF structures to maintain validity

Edge cases occur when:
1. Section is < 4KB: No full pages exist to remove
2. Section is marginally > 4KB: Structural overhead exceeds savings

These cases occur in practice with test binaries and validation utilities
that have minimal GPU code.

Test Strategy:
--------------
We use patched binaries (via elf_test_utils.py) rather than mocking because:
1. Mocking would be fragile if we fix the underlying code
2. Real binaries test the full code path
3. Patching is reproducible and well-documented
"""

from pathlib import Path

import pytest

from rocm_kpack import binutils
from rocm_kpack.elf_offload_kpacker import kpack_offload_binary


class TestSmallHipFatbinSection:
    """
    Tests for binaries with .hip_fatbin sections too small to zero-page.

    Fixes: https://github.com/ROCm/rocm-kpack/issues/1

    When .hip_fatbin is < 4KB, there are no full pages to remove.
    The fix copies the file unchanged instead of failing.
    """

    def test_kpack_offload_small_section_should_not_fail(
        self,
        small_hip_fatbin_binary: Path,
        tmp_path: Path,
        toolchain: binutils.Toolchain,
    ):
        """
        Verify that kpack_offload_binary handles small sections gracefully.

        Current behavior (BROKEN):
            Raises RuntimeError("Zero-page optimization failed")

        Expected behavior (after fix):
            Completes successfully, skipping zero-page optimization.
            The binary will be slightly larger (marker overhead) but valid.

        This reproduces the CI failure:
            WARNING: Section too small or misaligned - no full pages to zero
              Section range: [0xf000, 0xfd6d)
            Error: Zero-page optimization failed
        """
        marked_binary = tmp_path / "marked.exe"
        output_binary = tmp_path / "output.exe"

        # Add the kpack reference marker
        binutils.add_kpack_ref_marker(
            small_hip_fatbin_binary,
            marked_binary,
            kpack_search_paths=["test/manifest.kpm"],
            kernel_name="test_kernel",
            toolchain=toolchain,
        )

        # This should NOT raise - it should handle small sections gracefully
        # Currently fails with: RuntimeError: Zero-page optimization failed
        result = kpack_offload_binary(
            marked_binary,
            output_binary,
            toolchain=toolchain,
            verbose=True,
        )

        # Verify output was created
        assert output_binary.exists(), "Output binary should be created"

        # The binary is valid even if it didn't shrink
        assert result is not None
        assert "kpack_ref_vaddr" in result


class TestMarginalHipFatbinSection:
    """
    Tests for binaries where kpack overhead exceeds zero-page savings.

    Fixes: https://github.com/ROCm/rocm-kpack/issues/1

    When .hip_fatbin is slightly > 4KB, we can zero one page (save 4KB),
    but structural changes (padding, PHDR relocation) may add more.
    The fix removes the strict size check that rejected this case.
    """

    def test_kpack_offload_marginal_section_size_growth(
        self,
        marginal_hip_fatbin_binary: Path,
        tmp_path: Path,
        toolchain: binutils.Toolchain,
    ):
        """
        Verify that kpack_offload_binary works even when size grows.

        Current behavior (BROKEN):
            In artifact_splitter.py, the check at line 584 fails:
            "Binary was not stripped or grew in size"

        Expected behavior (after fix):
            Completes successfully. The binary may be larger than original
            due to structural overhead, but this is acceptable for small
            binaries where the overhead exceeds zero-page savings.

        This reproduces the CI failure:
            Error: Binary was not stripped or grew in size: .../test_tuple
            Original: 764864 bytes, New: 766019 bytes
        """
        marked_binary = tmp_path / "marked.exe"
        output_binary = tmp_path / "output.exe"

        original_size = marginal_hip_fatbin_binary.stat().st_size

        # Add the kpack reference marker
        binutils.add_kpack_ref_marker(
            marginal_hip_fatbin_binary,
            marked_binary,
            kpack_search_paths=["test/manifest.kpm"],
            kernel_name="test_kernel",
            toolchain=toolchain,
        )

        # kpack_offload_binary itself succeeds...
        result = kpack_offload_binary(
            marked_binary,
            output_binary,
            toolchain=toolchain,
            verbose=True,
        )

        assert output_binary.exists()
        final_size = output_binary.stat().st_size

        # Document the size growth that causes the artifact_splitter check to fail
        # This is expected behavior for marginal cases - the fix is to relax the check
        print(
            f"\nSize change: {original_size} -> {final_size} ({final_size - original_size:+d} bytes)"
        )

        # The current strict check would fail here:
        # assert final_size < original_size  # This is what artifact_splitter checks

        # After the fix, we should accept this case:
        # The binary grew, but that's OK for marginal .hip_fatbin sections
        assert result is not None
        assert "kpack_ref_vaddr" in result

    def test_marginal_section_size_growth_is_acceptable(
        self,
        marginal_hip_fatbin_binary: Path,
        tmp_path: Path,
        toolchain: binutils.Toolchain,
    ):
        """
        Verify that size growth is acceptable for marginal .hip_fatbin sections.

        For small sections where structural overhead exceeds zero-page savings,
        the binary may grow slightly. This is acceptable - the important thing
        is that the transformation completes successfully and produces a valid
        binary with the kpack reference properly set up.

        This test documents the expected behavior: size growth is OK for edge cases.
        """
        marked_binary = tmp_path / "marked.exe"
        output_binary = tmp_path / "output.exe"

        original_size = marginal_hip_fatbin_binary.stat().st_size

        binutils.add_kpack_ref_marker(
            marginal_hip_fatbin_binary,
            marked_binary,
            kpack_search_paths=["test/manifest.kpm"],
            kernel_name="test_kernel",
            toolchain=toolchain,
        )

        # This should complete without error, even though size will grow
        result = kpack_offload_binary(
            marked_binary,
            output_binary,
            toolchain=toolchain,
        )

        final_size = output_binary.stat().st_size

        # Size growth is expected and acceptable for marginal sections
        assert final_size >= original_size, (
            f"Unexpected: size shrank for marginal section. "
            f"This test expects size growth due to structural overhead."
        )

        # The transformation should still succeed
        assert output_binary.exists()
        assert result is not None
        assert "kpack_ref_vaddr" in result

        # Document the overhead for future reference
        overhead = final_size - original_size
        print(
            f"\nMarginal section overhead: {overhead} bytes ({original_size} -> {final_size})"
        )

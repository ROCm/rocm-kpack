"""
Unit tests for the rocm_kpack.coff.verify module.

Tests CoffVerifier class for PE binary validation.
"""

import pytest
from pathlib import Path

from rocm_kpack.coff import (
    CoffSurgery,
    CoffVerifier,
    VerificationResult,
    verify_with_llvm_objdump,
    verify_all,
)


class TestCoffVerifier:
    """Tests for the CoffVerifier class."""

    def test_verify_valid_exe(self, test_assets_dir: Path):
        """Test verification of a valid executable passes."""
        binary = (
            test_assets_dir / "bundled_binaries/windows/cov5/test_kernel_single.exe"
        )
        result = CoffVerifier.verify(binary)
        assert result.passed, f"Verification failed: {result}"

    def test_verify_valid_dll(self, test_assets_dir: Path):
        """Test verification of a valid DLL passes."""
        binary = (
            test_assets_dir / "bundled_binaries/windows/cov5/test_kernel_single.dll"
        )
        result = CoffVerifier.verify(binary)
        assert result.passed, f"Verification failed: {result}"

    def test_verify_host_only_exe(self, test_assets_dir: Path):
        """Test verification of host-only executable passes."""
        binary = test_assets_dir / "bundled_binaries/windows/cov5/host_only.exe"
        result = CoffVerifier.verify(binary)
        assert result.passed, f"Verification failed: {result}"

    def test_verify_host_only_dll(self, test_assets_dir: Path):
        """Test verification of host-only DLL passes."""
        binary = test_assets_dir / "bundled_binaries/windows/cov5/host_only.dll"
        result = CoffVerifier.verify(binary)
        assert result.passed, f"Verification failed: {result}"

    def test_verify_multi_arch(self, test_assets_dir: Path):
        """Test verification of multi-arch binary passes."""
        binary = test_assets_dir / "bundled_binaries/windows/cov5/test_kernel_multi.exe"
        result = CoffVerifier.verify(binary)
        assert result.passed, f"Verification failed: {result}"

    def test_verify_multi_wrapper(self, test_assets_dir: Path):
        """Test verification of RDC multi-wrapper binary passes."""
        binary = (
            test_assets_dir / "bundled_binaries/windows/cov5/test_multi_wrapper.dll"
        )
        result = CoffVerifier.verify(binary)
        assert result.passed, f"Verification failed: {result}"


class TestVerificationResult:
    """Tests for VerificationResult class."""

    def test_empty_result_passes(self):
        """Empty result should pass by default."""
        result = VerificationResult()
        assert result.passed
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_add_error_fails(self):
        """Adding an error should fail the result."""
        result = VerificationResult()
        result.add_error("Test error")
        assert not result.passed
        assert len(result.errors) == 1
        assert "Test error" in result.errors[0]

    def test_add_warning_does_not_fail(self):
        """Adding a warning should not fail the result."""
        result = VerificationResult()
        result.add_warning("Test warning")
        assert result.passed
        assert len(result.warnings) == 1
        assert "Test warning" in result.warnings[0]

    def test_merge_results(self):
        """Test merging two results."""
        result1 = VerificationResult()
        result1.add_warning("Warning 1")

        result2 = VerificationResult()
        result2.add_error("Error 1")

        result1.merge(result2)
        assert not result1.passed
        assert len(result1.warnings) == 1
        assert len(result1.errors) == 1


class TestVerifyWithTools:
    """Tests for external tool verification (may skip if tools not available)."""

    def test_verify_with_objdump(self, test_assets_dir: Path, toolchain):
        """Test verification with objdump/llvm-objdump."""
        binary = (
            test_assets_dir / "bundled_binaries/windows/cov5/test_kernel_single.exe"
        )

        # Try to get objdump from toolchain (falls back to llvm-objdump)
        try:
            objdump_path = toolchain.objdump
        except OSError:
            pytest.skip("objdump/llvm-objdump not available")

        result = verify_with_llvm_objdump(binary, objdump_path)
        # Should pass or give "not found" warning, but not error
        if not result.passed:
            # Check it's just a "not found" warning
            for error in result.errors:
                assert "not found" in error.lower() or "warning" in error.lower()

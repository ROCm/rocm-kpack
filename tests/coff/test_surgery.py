"""
Unit tests for the rocm_kpack.coff.surgery module.

Tests CoffSurgery class for PE/COFF binary manipulation.
"""

import pytest
from pathlib import Path

from rocm_kpack.coff import (
    CoffSurgery,
    SectionInfo,
    IMAGE_SCN_MEM_READ,
    IMAGE_SCN_CNT_INITIALIZED_DATA,
)


class TestCoffSurgery:
    """Tests for the CoffSurgery class."""

    def test_load_executable(self, test_assets_dir: Path):
        """Test loading a valid PE executable."""
        binary = (
            test_assets_dir / "bundled_binaries/windows/cov5/test_kernel_single.exe"
        )
        surgery = CoffSurgery.load(binary)

        assert surgery.coff_header is not None
        assert surgery.optional_header is not None
        # Should be an executable (not DLL)
        assert not surgery.is_dll

    def test_load_dll(self, test_assets_dir: Path):
        """Test loading a valid DLL."""
        binary = (
            test_assets_dir / "bundled_binaries/windows/cov5/test_kernel_single.dll"
        )
        surgery = CoffSurgery.load(binary)

        assert surgery.coff_header is not None
        assert surgery.optional_header is not None
        # Should be a DLL
        assert surgery.is_dll

    def test_find_section(self, test_assets_dir: Path):
        """Test finding sections by name."""
        binary = (
            test_assets_dir / "bundled_binaries/windows/cov5/test_kernel_single.exe"
        )
        surgery = CoffSurgery.load(binary)

        # Should find .hip_fat (PE uses 8-char names)
        section = surgery.find_section(".hip_fat")
        assert section is not None
        assert section.name == ".hip_fat"
        assert section.raw_size > 0

        # Should not find nonexistent section
        assert surgery.find_section(".nonexistent") is None

    def test_find_hipfatb_segment(self, test_assets_dir: Path):
        """Test finding .hipFatB (wrapper) section."""
        binary = (
            test_assets_dir / "bundled_binaries/windows/cov5/test_kernel_single.exe"
        )
        surgery = CoffSurgery.load(binary)

        # Should find .hipFatB (wrapper section)
        section = surgery.find_section(".hipFatB")
        assert section is not None
        assert section.name == ".hipFatB"

    def test_iter_sections(self, test_assets_dir: Path):
        """Test iterating over all sections."""
        binary = (
            test_assets_dir / "bundled_binaries/windows/cov5/test_kernel_single.exe"
        )
        surgery = CoffSurgery.load(binary)

        sections = list(surgery.iter_sections())
        assert len(sections) > 0

        # Check we have expected sections
        names = [s.name for s in sections]
        assert ".text" in names
        assert ".hip_fat" in names
        assert ".hipFatB" in names

    def test_get_section_content(self, test_assets_dir: Path):
        """Test reading section content."""
        binary = (
            test_assets_dir / "bundled_binaries/windows/cov5/test_kernel_single.exe"
        )
        surgery = CoffSurgery.load(binary)

        section = surgery.find_section(".hip_fat")
        assert section is not None

        content = surgery.get_section_content(section)
        assert len(content) > 0

    def test_rva_to_file_offset(self, test_assets_dir: Path):
        """Test RVA to file offset conversion."""
        binary = (
            test_assets_dir / "bundled_binaries/windows/cov5/test_kernel_single.exe"
        )
        surgery = CoffSurgery.load(binary)

        section = surgery.find_section(".text")
        assert section is not None

        # RVA should map to file offset
        offset = surgery.rva_to_file_offset(section.rva)
        assert offset is not None
        assert offset == section.file_offset

    def test_add_section(self, test_assets_dir: Path, tmp_path: Path):
        """Test adding a new section."""
        binary = (
            test_assets_dir / "bundled_binaries/windows/cov5/test_kernel_single.exe"
        )
        surgery = CoffSurgery.load(binary)

        # Count original sections
        original_count = len(list(surgery.iter_sections()))

        # Add a new section
        test_data = b"Hello, test section!"
        result = surgery.add_section(
            name=".test",
            content=test_data,
            characteristics=IMAGE_SCN_MEM_READ | IMAGE_SCN_CNT_INITIALIZED_DATA,
        )

        assert result is not None
        assert result.section_index >= 0

        # Verify section exists
        new_section = surgery.find_section(".test")
        assert new_section is not None

        # Verify content can be read back (slice to virtual_size, not padded raw_size)
        content = surgery.get_section_content(new_section)
        content = content[: new_section.virtual_size]
        assert content == test_data

        # Save and reload
        output_path = tmp_path / "with_section.exe"
        surgery.save(output_path)

        # Reload and verify
        surgery2 = CoffSurgery.load(output_path)
        new_count = len(list(surgery2.iter_sections()))
        assert new_count == original_count + 1

        section = surgery2.find_section(".test")
        assert section is not None
        content = surgery2.get_section_content(section)
        content = content[: section.virtual_size]
        assert content == test_data


class TestHostOnlyBinaries:
    """Tests for host-only binaries (negative tests)."""

    def test_host_only_exe_no_hip_fat(self, test_assets_dir: Path):
        """Test that host-only exe has no .hip_fat section."""
        binary = test_assets_dir / "bundled_binaries/windows/cov5/host_only.exe"
        surgery = CoffSurgery.load(binary)

        # Should NOT have .hip_fat section
        assert surgery.find_section(".hip_fat") is None
        assert surgery.find_section(".hipFatB") is None

    def test_host_only_dll_no_hip_fat(self, test_assets_dir: Path):
        """Test that host-only DLL has no .hip_fat section."""
        binary = test_assets_dir / "bundled_binaries/windows/cov5/host_only.dll"
        surgery = CoffSurgery.load(binary)

        # Should NOT have .hip_fat section
        assert surgery.find_section(".hip_fat") is None
        assert surgery.find_section(".hipFatB") is None

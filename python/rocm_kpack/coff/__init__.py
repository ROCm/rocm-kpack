"""
PE/COFF binary manipulation package for rocm-kpack.

This package provides clean abstractions for PE/COFF surgery operations:
- types: Unified PE/COFF struct definitions
- surgery: High-level CoffSurgery class for modifications

The design mirrors the rocm_kpack.elf package, adapted for Windows PE/COFF.
"""

from .surgery import (
    CoffSurgery,
    SectionInfo,
    Modification,
    RelocationInfo,
)
from .types import (
    # Structs
    DosHeader,
    CoffHeader,
    OptionalHeader64,
    SectionHeader,
    DataDirectory,
    BaseRelocationBlock,
    BaseRelocationEntry,
    # Constants
    PAGE_SIZE,
    FILE_ALIGNMENT_DEFAULT,
    SECTION_ALIGNMENT_DEFAULT,
    DOS_MAGIC,
    PE_SIGNATURE,
    # Machine types
    IMAGE_FILE_MACHINE_AMD64,
    IMAGE_FILE_MACHINE_I386,
    # Optional header magic
    IMAGE_NT_OPTIONAL_HDR64_MAGIC,
    # Section characteristics
    IMAGE_SCN_CNT_CODE,
    IMAGE_SCN_CNT_INITIALIZED_DATA,
    IMAGE_SCN_CNT_UNINITIALIZED_DATA,
    IMAGE_SCN_MEM_READ,
    IMAGE_SCN_MEM_WRITE,
    IMAGE_SCN_MEM_EXECUTE,
    IMAGE_SCN_MEM_DISCARDABLE,
    # File characteristics
    IMAGE_FILE_DLL,
    IMAGE_FILE_EXECUTABLE_IMAGE,
    # DLL characteristics
    IMAGE_DLLCHARACTERISTICS_DYNAMIC_BASE,
    IMAGE_DLLCHARACTERISTICS_HIGH_ENTROPY_VA,
    IMAGE_DLLCHARACTERISTICS_NX_COMPAT,
    # Data directory indices
    IMAGE_DIRECTORY_ENTRY_EXPORT,
    IMAGE_DIRECTORY_ENTRY_IMPORT,
    IMAGE_DIRECTORY_ENTRY_BASERELOC,
    IMAGE_NUMBEROF_DIRECTORY_ENTRIES,
    # Base relocation types
    IMAGE_REL_BASED_ABSOLUTE,
    IMAGE_REL_BASED_DIR64,
    IMAGE_REL_BASED_HIGHLOW,
    # Structure sizes
    DOS_HEADER_SIZE,
    COFF_HEADER_SIZE,
    OPTIONAL_HEADER64_SIZE,
    SECTION_HEADER_SIZE,
    DATA_DIRECTORY_SIZE,
    # Helper functions
    round_up_to_alignment,
    round_down_to_alignment,
    round_up_to_page,
    round_down_to_page,
    section_name_to_bytes,
)

__all__ = [
    # High-level classes
    "CoffSurgery",
    "SectionInfo",
    "Modification",
    "RelocationInfo",
    # Structs
    "DosHeader",
    "CoffHeader",
    "OptionalHeader64",
    "SectionHeader",
    "DataDirectory",
    "BaseRelocationBlock",
    "BaseRelocationEntry",
    # Constants
    "PAGE_SIZE",
    "FILE_ALIGNMENT_DEFAULT",
    "SECTION_ALIGNMENT_DEFAULT",
    "DOS_MAGIC",
    "PE_SIGNATURE",
    # Machine types
    "IMAGE_FILE_MACHINE_AMD64",
    "IMAGE_FILE_MACHINE_I386",
    # Optional header magic
    "IMAGE_NT_OPTIONAL_HDR64_MAGIC",
    # Section characteristics
    "IMAGE_SCN_CNT_CODE",
    "IMAGE_SCN_CNT_INITIALIZED_DATA",
    "IMAGE_SCN_CNT_UNINITIALIZED_DATA",
    "IMAGE_SCN_MEM_READ",
    "IMAGE_SCN_MEM_WRITE",
    "IMAGE_SCN_MEM_EXECUTE",
    "IMAGE_SCN_MEM_DISCARDABLE",
    # File characteristics
    "IMAGE_FILE_DLL",
    "IMAGE_FILE_EXECUTABLE_IMAGE",
    # DLL characteristics
    "IMAGE_DLLCHARACTERISTICS_DYNAMIC_BASE",
    "IMAGE_DLLCHARACTERISTICS_HIGH_ENTROPY_VA",
    "IMAGE_DLLCHARACTERISTICS_NX_COMPAT",
    # Data directory indices
    "IMAGE_DIRECTORY_ENTRY_EXPORT",
    "IMAGE_DIRECTORY_ENTRY_IMPORT",
    "IMAGE_DIRECTORY_ENTRY_BASERELOC",
    "IMAGE_NUMBEROF_DIRECTORY_ENTRIES",
    # Base relocation types
    "IMAGE_REL_BASED_ABSOLUTE",
    "IMAGE_REL_BASED_DIR64",
    "IMAGE_REL_BASED_HIGHLOW",
    # Structure sizes
    "DOS_HEADER_SIZE",
    "COFF_HEADER_SIZE",
    "OPTIONAL_HEADER64_SIZE",
    "SECTION_HEADER_SIZE",
    "DATA_DIRECTORY_SIZE",
    # Helper functions
    "round_up_to_alignment",
    "round_down_to_alignment",
    "round_up_to_page",
    "round_down_to_page",
    "section_name_to_bytes",
]

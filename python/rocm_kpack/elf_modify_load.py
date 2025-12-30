#!/usr/bin/env python3
"""
Extended ELF manipulation tool for zero-paging and PT_LOAD mapping.

This tool provides zero-page optimization and PT_LOAD segment mapping for ELF binaries.
Designed for kpack runtime integration.
"""

import shutil
import struct
import sys
from pathlib import Path
from typing import NamedTuple, List, Optional, Tuple


PAGE_SIZE = 0x1000  # 4KB pages


class Elf64_Ehdr(NamedTuple):
    """ELF64 file header (minimal fields we need)"""

    e_phoff: int
    e_shoff: int
    e_phnum: int
    e_shnum: int
    e_shstrndx: int


class Elf64_Phdr(NamedTuple):
    """ELF64 program header"""

    p_type: int
    p_flags: int
    p_offset: int
    p_vaddr: int
    p_paddr: int
    p_filesz: int
    p_memsz: int
    p_align: int


class Elf64_Shdr(NamedTuple):
    """ELF64 section header"""

    sh_name: int
    sh_type: int
    sh_flags: int
    sh_addr: int
    sh_offset: int
    sh_size: int
    sh_link: int
    sh_info: int
    sh_addralign: int
    sh_entsize: int


# ELF constants
PT_LOAD = 1
PT_NOTE = 4
PT_PHDR = 6
PT_GNU_STACK = 0x6474E551
SHT_NOBITS = 8
SHT_RELA = 4
SHT_REL = 9
SHF_ALLOC = 0x2

# Relocation types
R_X86_64_RELATIVE = 8


def read_elf_header(data: bytes) -> Elf64_Ehdr:
    """Read minimal ELF header fields."""
    if data[:4] != b"\x7fELF":
        raise ValueError("Not an ELF file")

    e_phoff = struct.unpack_from("<Q", data, 32)[0]
    e_shoff = struct.unpack_from("<Q", data, 40)[0]
    e_phnum = struct.unpack_from("<H", data, 56)[0]
    e_shnum = struct.unpack_from("<H", data, 60)[0]
    e_shstrndx = struct.unpack_from("<H", data, 62)[0]

    return Elf64_Ehdr(e_phoff, e_shoff, e_phnum, e_shnum, e_shstrndx)


def read_program_header(data: bytes, offset: int) -> Elf64_Phdr:
    """Read a program header."""
    values = struct.unpack_from("<IIQQQQQQ", data, offset)
    return Elf64_Phdr(*values)


def write_program_header(data: bytearray, offset: int, phdr: Elf64_Phdr):
    """Write a program header."""
    struct.pack_into(
        "<IIQQQQQQ",
        data,
        offset,
        phdr.p_type,
        phdr.p_flags,
        phdr.p_offset,
        phdr.p_vaddr,
        phdr.p_paddr,
        phdr.p_filesz,
        phdr.p_memsz,
        phdr.p_align,
    )


def read_section_header(data: bytes, offset: int) -> Elf64_Shdr:
    """Read a section header."""
    values = struct.unpack_from("<IIQQQQIIQQ", data, offset)
    return Elf64_Shdr(*values)


def write_section_header(data: bytearray, offset: int, shdr: Elf64_Shdr):
    """Write a section header."""
    struct.pack_into(
        "<IIQQQQIIQQ",
        data,
        offset,
        shdr.sh_name,
        shdr.sh_type,
        shdr.sh_flags,
        shdr.sh_addr,
        shdr.sh_offset,
        shdr.sh_size,
        shdr.sh_link,
        shdr.sh_info,
        shdr.sh_addralign,
        shdr.sh_entsize,
    )


def get_section_name(data: bytes, shstrtab_offset: int, name_idx: int) -> str:
    """Get section name from string table."""
    name_offset = shstrtab_offset + name_idx
    end = data.find(b"\x00", name_offset)
    if end == -1:
        return ""
    return data[name_offset:end].decode("ascii", errors="ignore")


def round_up_to_page(addr: int) -> int:
    """Round address up to next page boundary."""
    return (addr + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1)


def round_down_to_page(addr: int) -> int:
    """Round address down to previous page boundary."""
    return addr & ~(PAGE_SIZE - 1)


def calculate_aligned_range(start: int, size: int) -> Tuple[int, int, int]:
    """
    Calculate the page-aligned range within a section.

    Returns: (aligned_start, aligned_size, saved_bytes)
    """
    end = start + size

    # Round up start to next page boundary
    aligned_start = round_up_to_page(start)

    # Round down end to previous page boundary
    aligned_end = round_down_to_page(end)

    if aligned_start >= aligned_end:
        # No full pages to zero
        return (0, 0, 0)

    aligned_size = aligned_end - aligned_start
    return (aligned_start, aligned_size, aligned_size)


def is_pie_or_shared_library(data: bytes) -> bool:
    """
    Check if binary is PIE or shared library (ET_DYN type).

    PIE executables and shared libraries require relocations for pointers
    to work correctly, as they are loaded at runtime-determined addresses.

    Args:
        data: ELF binary data

    Returns:
        True if binary is ET_DYN (PIE or shared library), False otherwise
    """
    # e_type is at offset 16, 2 bytes (in ELF64 header)
    ET_DYN = 3  # Shared object file (PIE or .so)
    e_type = struct.unpack_from("<H", data, 16)[0]
    return e_type == ET_DYN


def _get_phdr_capacity(data: bytes, ehdr: Elf64_Ehdr, min_content_offset: int) -> int:
    """Get allocated capacity of PHDR table (may be > e_phnum).

    Args:
        data: ELF binary data
        ehdr: ELF header
        min_content_offset: Minimum offset of content after PHDR

    Returns:
        Number of PHDR entries that can fit in available space
    """
    available_space = min_content_offset - ehdr.e_phoff
    return available_space // 56


def resize_phdr_table(
    data: bytearray,
    ehdr: Elf64_Ehdr,
    new_phdrs: List[Elf64_Phdr],
    min_content_offset: int,
    phdr_spare_slots: int = 16,
    verbose: bool = True,
) -> Tuple[bytearray, int]:
    """
    Resize program header table to accommodate new_phdrs.

    Args:
        data: ELF binary data
        ehdr: ELF header
        new_phdrs: New program headers to write
        min_content_offset: Minimum offset of content after PHDR (in data's coordinates)
        phdr_spare_slots: Number of spare slots when relocating (default: 16)
        verbose: Print progress

    Returns: (updated_data, e_phoff)

    If sufficient space exists at current location, writes in place.
    If already relocated with spare capacity, appends in place.
    Otherwise, relocates to end of file with over-allocation.
    """
    old_phdr_size = ehdr.e_phnum * 56
    new_phdr_size = len(new_phdrs) * 56

    available_space = min_content_offset - ehdr.e_phoff

    if new_phdr_size <= available_space:
        # Fits in place
        if verbose:
            print(f"  Writing {len(new_phdrs)} program headers in place")

        for i, phdr in enumerate(new_phdrs):
            write_program_header(data, ehdr.e_phoff + i * 56, phdr)

        struct.pack_into("<H", data, 56, len(new_phdrs))
        return (data, ehdr.e_phoff)

    # Check if already relocated with spare capacity
    current_capacity = _get_phdr_capacity(data, ehdr, min_content_offset)
    spare_slots = current_capacity - ehdr.e_phnum

    if spare_slots > 0 and len(new_phdrs) <= current_capacity:
        # Can append to existing relocated PHDR
        if verbose:
            print(f"  Using {spare_slots} spare PHDR slots (appending in place)")

        for i, phdr in enumerate(new_phdrs):
            write_program_header(data, ehdr.e_phoff + i * 56, phdr)

        struct.pack_into("<H", data, 56, len(new_phdrs))
        return (data, ehdr.e_phoff)

    # Need to relocate with over-allocation
    new_phoff = len(data)

    if verbose:
        print(f"  Relocating program headers to end of file:")
        print(
            f"    Need {new_phdr_size} bytes, have {available_space} bytes at 0x{ehdr.e_phoff:x}"
        )
        print(f"    New offset: 0x{new_phoff:x}")

    # Find max vaddr for placing relocated headers
    max_vaddr_end = 0
    for phdr in new_phdrs:
        if phdr.p_type == PT_LOAD:
            max_vaddr_end = max(max_vaddr_end, phdr.p_vaddr + phdr.p_memsz)

    phdr_vaddr = round_up_to_page(max_vaddr_end)

    # Align file offset for mmap
    vaddr_remainder = phdr_vaddr % PAGE_SIZE
    offset_remainder = new_phoff % PAGE_SIZE

    if offset_remainder != vaddr_remainder:
        padding = (vaddr_remainder - offset_remainder + PAGE_SIZE) % PAGE_SIZE
        data.extend(b"\x00" * padding)
        new_phoff = len(data)
        if verbose:
            print(f"    Added {padding} bytes padding for mmap alignment")

    # Calculate capacity with over-allocation
    final_phdr_count = len(new_phdrs) + 1  # +1 for PT_LOAD covering PHDR
    phdr_capacity = (
        (final_phdr_count + phdr_spare_slots - 1) // phdr_spare_slots
    ) * phdr_spare_slots
    phdr_allocated_size = phdr_capacity * 56

    if verbose:
        print(
            f"    Allocating {phdr_capacity} PHDR slots ({phdr_capacity - final_phdr_count} spare)"
        )

    # Update PT_PHDR if present
    for i, phdr in enumerate(new_phdrs):
        if phdr.p_type == PT_PHDR:
            new_phdrs[i] = Elf64_Phdr(
                PT_PHDR,
                phdr.p_flags,
                new_phoff,
                phdr_vaddr,
                phdr_vaddr,
                phdr_allocated_size,
                phdr_allocated_size,
                phdr.p_align,
            )
            break

    # Add PT_LOAD for relocated PHDR
    phdr_load = Elf64_Phdr(
        PT_LOAD,
        4,
        new_phoff,
        phdr_vaddr,
        phdr_vaddr,
        phdr_allocated_size,
        phdr_allocated_size,
        PAGE_SIZE,
    )
    new_phdrs.append(phdr_load)

    # Update PT_PHDR size to match
    for i, phdr in enumerate(new_phdrs):
        if phdr.p_type == PT_PHDR:
            new_phdrs[i] = Elf64_Phdr(
                PT_PHDR,
                phdr.p_flags,
                new_phoff,
                phdr_vaddr,
                phdr_vaddr,
                phdr_allocated_size,
                phdr_allocated_size,
                phdr.p_align,
            )
            break

    # Write all headers + zero padding
    for phdr in new_phdrs:
        data.extend(
            struct.pack(
                "<IIQQQQQQ",
                phdr.p_type,
                phdr.p_flags,
                phdr.p_offset,
                phdr.p_vaddr,
                phdr.p_paddr,
                phdr.p_filesz,
                phdr.p_memsz,
                phdr.p_align,
            )
        )

    # Zero-pad unused slots
    unused_slots = phdr_capacity - len(new_phdrs)
    data.extend(b"\x00" * (unused_slots * 56))

    # Update ELF header
    struct.pack_into("<Q", data, 32, new_phoff)
    struct.pack_into("<H", data, 56, len(new_phdrs))

    return (data, new_phoff)


def conservative_zero_page(
    input_path: Path,
    output_path: Path,
    section_name: str = ".hip_fatbin",
    verbose: bool = True,
    force_overflow: bool = False,
    phdr_spare_slots: int = 16,
) -> bool:
    """
    Apply conservative zero-page optimization to a section.

    Only zero-pages fully page-aligned regions. Preserves unaligned content.

    Args:
        input_path: Input ELF binary
        output_path: Output ELF binary
        section_name: Section to zero-page (default: .hip_fatbin)
        verbose: Print progress information
        force_overflow: Test mode - artificially reduce available space to force overflow
    """

    # Read input
    data = bytearray(input_path.read_bytes())
    original_size = len(data)

    try:
        ehdr = read_elf_header(data)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return False

    if verbose:
        print(f"Processing: {input_path}")
        print(f"  File size: {original_size:,} bytes")

    # Find target section
    shstrtab_shdr = read_section_header(data, ehdr.e_shoff + ehdr.e_shstrndx * 64)
    shstrtab_offset = shstrtab_shdr.sh_offset

    target_section_idx = None
    section_offset = None
    section_vaddr = None
    section_size = None

    for i in range(ehdr.e_shnum):
        shdr = read_section_header(data, ehdr.e_shoff + i * 64)
        name = get_section_name(data, shstrtab_offset, shdr.sh_name)

        if name == section_name:
            target_section_idx = i
            section_offset = shdr.sh_offset
            section_vaddr = shdr.sh_addr
            section_size = shdr.sh_size
            break

    if target_section_idx is None:
        print(f"ERROR: Section '{section_name}' not found", file=sys.stderr)
        return False

    if verbose:
        print(f"\nFound section '{section_name}':")
        print(f"  File offset: 0x{section_offset:x}")
        print(f"  Virtual addr: 0x{section_vaddr:x}")
        print(f"  Size: 0x{section_size:x} ({section_size:,} bytes)")

    # Calculate aligned range
    aligned_vaddr, aligned_size, saved_bytes = calculate_aligned_range(
        section_vaddr, section_size
    )

    if aligned_size == 0:
        # Section is too small (< 4KB) or misaligned such that no full pages
        # can be zero-paged. This is OK for small binaries with minimal GPU code.
        # Just copy the file unchanged and return success.
        if verbose:
            print(f"\n  Section too small or misaligned - no full pages to zero")
            print(
                f"  Section range: [0x{section_vaddr:x}, 0x{section_vaddr + section_size:x})"
            )
            print(f"  Copying file unchanged (no optimization possible)")
        shutil.copy2(input_path, output_path)
        return True

    aligned_offset = section_offset + (aligned_vaddr - section_vaddr)
    section_end_vaddr = section_vaddr + section_size
    aligned_end_vaddr = aligned_vaddr + aligned_size

    if verbose:
        print(f"\nConservative zero-page analysis:")
        print(f"  Total section: [0x{section_vaddr:x}, 0x{section_end_vaddr:x})")
        print(f"  Aligned range: [0x{aligned_vaddr:x}, 0x{aligned_end_vaddr:x})")
        print(f"  Aligned size: 0x{aligned_size:x} ({aligned_size:,} bytes)")
        print(f"  Pages to zero: {aligned_size // PAGE_SIZE}")

        if section_vaddr != aligned_vaddr:
            prefix_size = aligned_vaddr - section_vaddr
            print(f"  Prefix kept: 0x{prefix_size:x} ({prefix_size} bytes)")

        if section_end_vaddr != aligned_end_vaddr:
            suffix_size = section_end_vaddr - aligned_end_vaddr
            print(f"  Suffix kept: 0x{suffix_size:x} ({suffix_size} bytes)")

    # Find the PT_LOAD containing the section
    target_load_idx = None
    target_load = None

    for i in range(ehdr.e_phnum):
        phdr = read_program_header(data, ehdr.e_phoff + i * 56)
        if phdr.p_type == PT_LOAD:
            if phdr.p_vaddr <= section_vaddr < phdr.p_vaddr + phdr.p_memsz:
                target_load_idx = i
                target_load = phdr
                break

    if target_load is None:
        print("ERROR: Could not find PT_LOAD containing section", file=sys.stderr)
        return False

    if verbose:
        print(f"\nFound containing PT_LOAD at index {target_load_idx}:")
        print(f"  vaddr: 0x{target_load.p_vaddr:x}, memsz: 0x{target_load.p_memsz:x}")

    # Remove the aligned bytes from file
    new_data = bytearray(data[:aligned_offset])
    new_data.extend(data[aligned_offset + aligned_size :])

    if verbose:
        print(f"\nRemoving aligned region from file:")
        print(f"  Original size: {original_size:,} bytes")
        print(f"  New size: {len(new_data):,} bytes")
        print(f"  Saved: {saved_bytes:,} bytes ({100*saved_bytes/original_size:.1f}%)")

    # Now we need to split the PT_LOAD into up to 5 segments:
    # 1. Content before section (if any)
    # 2. Prefix of section before aligned region (if unaligned start)
    # 3. Aligned region (zero-page)
    # 4. Suffix of section after aligned region (if unaligned end)
    # 5. Content after section (if any)

    new_phdrs = []

    # Helper to create a PT_LOAD segment
    def make_load(vaddr, vsize, offset, fsize, flags=target_load.p_flags):
        return Elf64_Phdr(
            p_type=PT_LOAD,
            p_flags=flags,
            p_offset=offset,
            p_vaddr=vaddr,
            p_paddr=vaddr,
            p_filesz=fsize,
            p_memsz=vsize,
            p_align=target_load.p_align,
        )

    # Calculate all the pieces
    for i in range(ehdr.e_phnum):
        phdr = read_program_header(data, ehdr.e_phoff + i * 56)

        if i != target_load_idx:
            # Not the target load - adjust offset if after removed bytes
            if phdr.p_offset > aligned_offset:
                phdr = Elf64_Phdr(
                    phdr.p_type,
                    phdr.p_flags,
                    phdr.p_offset - aligned_size,
                    phdr.p_vaddr,
                    phdr.p_paddr,
                    phdr.p_filesz,
                    phdr.p_memsz,
                    phdr.p_align,
                )
            new_phdrs.append(phdr)
        else:
            # Split the target PT_LOAD
            # Piece 1: Before section (if any)
            if target_load.p_vaddr < section_vaddr:
                pre_vsize = section_vaddr - target_load.p_vaddr
                pre_fsize = min(pre_vsize, target_load.p_filesz)
                new_phdrs.append(
                    make_load(
                        target_load.p_vaddr, pre_vsize, target_load.p_offset, pre_fsize
                    )
                )

            # Piece 2: Section prefix (if unaligned start)
            if section_vaddr < aligned_vaddr:
                prefix_vsize = aligned_vaddr - section_vaddr
                prefix_fsize = prefix_vsize
                new_phdrs.append(
                    make_load(section_vaddr, prefix_vsize, section_offset, prefix_fsize)
                )

            # Piece 3: Aligned region (zero-page)
            new_phdrs.append(
                make_load(
                    aligned_vaddr,
                    aligned_size,
                    aligned_offset,
                    0,  # p_filesz=0 for zero-page
                )
            )

            # Piece 4: Section suffix (if unaligned end)
            if aligned_end_vaddr < section_end_vaddr:
                suffix_vsize = section_end_vaddr - aligned_end_vaddr
                suffix_fsize = suffix_vsize
                # File offset is now at aligned_offset (shifted down)
                new_phdrs.append(
                    make_load(
                        aligned_end_vaddr, suffix_vsize, aligned_offset, suffix_fsize
                    )
                )

            # Piece 5: After section (if any)
            target_load_end = target_load.p_vaddr + target_load.p_memsz
            if section_end_vaddr < target_load_end:
                post_vaddr = section_end_vaddr
                post_vsize = target_load_end - post_vaddr
                # Calculate file offset and size for post-section content
                section_file_end = section_offset + section_size
                post_file_offset = section_file_end - aligned_size  # Shifted down
                post_fsize = (
                    target_load.p_offset + target_load.p_filesz
                ) - section_file_end
                if post_fsize > 0:
                    new_phdrs.append(
                        make_load(post_vaddr, post_vsize, post_file_offset, post_fsize)
                    )

    if verbose:
        print(f"\nProgram headers: {ehdr.e_phnum} -> {len(new_phdrs)}")

    # Compute minimum content offset after PHDR (reading from original data)
    old_phdr_size = ehdr.e_phnum * 56
    min_content_offset = len(new_data)

    # Check section headers
    for i in range(ehdr.e_shnum):
        shdr = read_section_header(data, ehdr.e_shoff + i * 64)
        if shdr.sh_offset > ehdr.e_phoff + old_phdr_size:
            # Adjust offset for removed bytes
            adjusted_offset = (
                shdr.sh_offset - aligned_size
                if shdr.sh_offset > aligned_offset
                else shdr.sh_offset
            )
            min_content_offset = min(min_content_offset, adjusted_offset)

    # Also check program headers for data they reference (e.g., PT_INTERP)
    for i in range(ehdr.e_phnum):
        phdr = read_program_header(data, ehdr.e_phoff + i * 56)
        if phdr.p_offset > ehdr.e_phoff + old_phdr_size and phdr.p_filesz > 0:
            # Adjust offset for removed bytes
            adjusted_offset = (
                phdr.p_offset - aligned_size
                if phdr.p_offset > aligned_offset
                else phdr.p_offset
            )
            min_content_offset = min(min_content_offset, adjusted_offset)

    # Resize PHDR table (handles overflow, spare capacity, etc.)
    # Note: force_overflow handled by temporarily reducing space in test mode
    if force_overflow:
        # Artificially reduce space to trigger relocation for testing
        # Temporarily shrink available space by writing dummy PT_NOTE
        dummy_note = Elf64_Phdr(PT_NOTE, 0, ehdr.e_phoff + 56, 0, 0, 999999, 0, 1)
        new_phdrs.insert(0, dummy_note)
        if verbose:
            print(f"  [TEST MODE] Forcing overflow by adding dummy header")

    new_data, new_phoff = resize_phdr_table(
        new_data,
        ehdr,
        new_phdrs,
        min_content_offset,
        phdr_spare_slots=phdr_spare_slots,
        verbose=verbose,
    )

    if force_overflow:
        # Remove dummy PT_NOTE after relocation forced
        new_phdrs.pop(0)
        # Need to rewrite without dummy
        if new_phoff != ehdr.e_phoff:  # Was relocated
            # Rewrite headers without dummy
            write_offset = new_phoff
            for phdr in new_phdrs:
                struct.pack_into(
                    "<IIQQQQQQ",
                    new_data,
                    write_offset,
                    phdr.p_type,
                    phdr.p_flags,
                    phdr.p_offset,
                    phdr.p_vaddr,
                    phdr.p_paddr,
                    phdr.p_filesz,
                    phdr.p_memsz,
                    phdr.p_align,
                )
                write_offset += 56
            struct.pack_into("<H", new_data, 56, len(new_phdrs))

    # Update section header table offset
    new_shoff = (
        ehdr.e_shoff - aligned_size if ehdr.e_shoff > aligned_offset else ehdr.e_shoff
    )
    struct.pack_into("<Q", new_data, 40, new_shoff)

    # Update section headers
    for i in range(ehdr.e_shnum):
        shdr = read_section_header(data, ehdr.e_shoff + i * 64)

        if i == target_section_idx:
            # Mark as NOBITS
            shdr = Elf64_Shdr(
                shdr.sh_name,
                SHT_NOBITS,
                shdr.sh_flags,
                shdr.sh_addr,
                section_offset,
                shdr.sh_size,
                shdr.sh_link,
                shdr.sh_info,
                shdr.sh_addralign,
                shdr.sh_entsize,
            )
        elif shdr.sh_offset > aligned_offset:
            # Adjust offset
            shdr = Elf64_Shdr(
                shdr.sh_name,
                shdr.sh_type,
                shdr.sh_flags,
                shdr.sh_addr,
                shdr.sh_offset - aligned_size,
                shdr.sh_size,
                shdr.sh_link,
                shdr.sh_info,
                shdr.sh_addralign,
                shdr.sh_entsize,
            )

        write_section_header(new_data, new_shoff + i * 64, shdr)

    # Write output and preserve original permissions
    output_path.write_bytes(new_data)
    original_mode = input_path.stat().st_mode
    output_path.chmod(original_mode)

    if verbose:
        print(f"\n✅ Successfully wrote {output_path}")

    return True


class Elf64_Rela(NamedTuple):
    """ELF64 relocation with explicit addend"""

    r_offset: int
    r_info: int
    r_addend: int


class Elf64_Rel(NamedTuple):
    """ELF64 relocation with implicit addend"""

    r_offset: int
    r_info: int


def read_rela_entry(data: bytes, offset: int) -> Elf64_Rela:
    """Read a RELA relocation entry."""
    r_offset, r_info, r_addend = struct.unpack_from("<QQq", data, offset)
    return Elf64_Rela(r_offset, r_info, r_addend)


def write_rela_entry(data: bytearray, offset: int, rela: Elf64_Rela):
    """Write a RELA relocation entry."""
    struct.pack_into("<QQq", data, offset, rela.r_offset, rela.r_info, rela.r_addend)


def read_rel_entry(data: bytes, offset: int) -> Elf64_Rel:
    """Read a REL relocation entry."""
    r_offset, r_info = struct.unpack_from("<QQ", data, offset)
    return Elf64_Rel(r_offset, r_info)


def find_section_by_name(
    data: bytes, ehdr: Elf64_Ehdr, name: str
) -> Optional[Tuple[int, Elf64_Shdr]]:
    """Find a section by name. Returns (index, section_header) or None."""
    shstrtab_shdr = read_section_header(data, ehdr.e_shoff + ehdr.e_shstrndx * 64)
    shstrtab_offset = shstrtab_shdr.sh_offset

    for i in range(ehdr.e_shnum):
        shdr = read_section_header(data, ehdr.e_shoff + i * 64)
        section_name = get_section_name(data, shstrtab_offset, shdr.sh_name)
        if section_name == name:
            return (i, shdr)

    return None


def find_max_vaddr(phdrs: List[Elf64_Phdr]) -> int:
    """Find the maximum virtual address used by PT_LOAD segments."""
    max_vaddr = 0
    for phdr in phdrs:
        if phdr.p_type == PT_LOAD:
            vaddr_end = phdr.p_vaddr + phdr.p_memsz
            max_vaddr = max(max_vaddr, vaddr_end)
    return max_vaddr


def map_section_to_new_load(
    input_path: Path,
    output_path: Path,
    section_name: str,
    new_vaddr: Optional[int] = None,
    verbose: bool = True,
    phdr_spare_slots: int = 16,
) -> Optional[int]:
    """
    Map a section to a new PT_LOAD segment at a specified virtual address.

    The section must not already be part of a PT_LOAD segment (typically created
    with objcopy --add-section without ALLOC flag).

    Args:
        input_path: Input ELF binary
        output_path: Output ELF binary
        section_name: Section to map (e.g., ".rocm_kpack_ref")
        new_vaddr: Virtual address for new PT_LOAD (auto-allocate if None)
        verbose: Print progress information

    Returns:
        Virtual address where section was mapped, or None on error
    """
    # Read input
    data = bytearray(input_path.read_bytes())

    try:
        ehdr = read_elf_header(data)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return None

    if verbose:
        print(f"Mapping section '{section_name}' to new PT_LOAD:")
        print(f"  Input: {input_path}")

    # Find target section
    section_result = find_section_by_name(data, ehdr, section_name)
    if section_result is None:
        print(f"ERROR: Section '{section_name}' not found", file=sys.stderr)
        return None

    section_idx, section_shdr = section_result

    if verbose:
        print(f"  Section file offset: 0x{section_shdr.sh_offset:x}")
        print(
            f"  Section size: 0x{section_shdr.sh_size:x} ({section_shdr.sh_size} bytes)"
        )

    # Read all existing program headers
    phdrs = []
    for i in range(ehdr.e_phnum):
        phdr = read_program_header(data, ehdr.e_phoff + i * 56)
        phdrs.append(phdr)

    # Auto-allocate virtual address if not specified
    if new_vaddr is None:
        max_vaddr = find_max_vaddr(phdrs)
        new_vaddr = round_up_to_page(max_vaddr)
        if verbose:
            print(f"  Auto-allocated vaddr: 0x{new_vaddr:x}")
    else:
        if verbose:
            print(f"  Using specified vaddr: 0x{new_vaddr:x}")

    # Ensure file offset alignment matches vaddr alignment for mmap
    # Requirement: (p_offset % PAGE_SIZE) == (p_vaddr % PAGE_SIZE)
    section_file_offset = section_shdr.sh_offset
    vaddr_remainder = new_vaddr % PAGE_SIZE
    offset_remainder = section_file_offset % PAGE_SIZE

    new_section_offset = section_file_offset
    padding_needed = 0

    if offset_remainder != vaddr_remainder:
        # Need to relocate section data to have proper alignment
        # Append to end of file with correct alignment
        padding_needed = (
            vaddr_remainder - (len(data) % PAGE_SIZE) + PAGE_SIZE
        ) % PAGE_SIZE
        new_section_offset = len(data) + padding_needed

        if verbose:
            print(f"  Realigning section for mmap compatibility:")
            print(
                f"    Old offset: 0x{section_file_offset:x} (remainder 0x{offset_remainder:x})"
            )
            print(
                f"    New offset: 0x{new_section_offset:x} (remainder 0x{new_section_offset % PAGE_SIZE:x})"
            )
            print(f"    Padding: {padding_needed} bytes")

        # Add padding and copy section data to new location
        data.extend(b"\x00" * padding_needed)
        section_data = data[
            section_file_offset : section_file_offset + section_shdr.sh_size
        ]
        data.extend(section_data)

    # Create new PT_LOAD segment for the section
    new_load = Elf64_Phdr(
        p_type=PT_LOAD,
        p_flags=4,  # PF_R (read-only)
        p_offset=new_section_offset,
        p_vaddr=new_vaddr,
        p_paddr=new_vaddr,
        p_filesz=section_shdr.sh_size,
        p_memsz=section_shdr.sh_size,
        p_align=PAGE_SIZE,
    )

    phdrs.append(new_load)

    if verbose:
        print(f"  Created PT_LOAD:")
        print(f"    vaddr: 0x{new_vaddr:x}")
        print(f"    offset: 0x{new_section_offset:x}")
        print(f"    size: 0x{section_shdr.sh_size:x}")

    # Compute minimum content offset after PHDR
    old_phdr_size = ehdr.e_phnum * 56
    min_content_offset = len(data)

    # Check section headers
    for i in range(ehdr.e_shnum):
        shdr = read_section_header(data, ehdr.e_shoff + i * 64)
        if shdr.sh_offset > ehdr.e_phoff + old_phdr_size:
            min_content_offset = min(min_content_offset, shdr.sh_offset)

    # Also check program headers for data they reference (e.g., PT_INTERP)
    for i in range(ehdr.e_phnum):
        phdr = read_program_header(data, ehdr.e_phoff + i * 56)
        if phdr.p_offset > ehdr.e_phoff + old_phdr_size and phdr.p_filesz > 0:
            min_content_offset = min(min_content_offset, phdr.p_offset)

    # Resize program header table (handles in-place or relocation)
    data, new_phoff = resize_phdr_table(
        data,
        ehdr,
        phdrs,
        min_content_offset,
        phdr_spare_slots=phdr_spare_slots,
        verbose=verbose,
    )

    # Update e_phoff in ELF header if it changed
    if new_phoff != ehdr.e_phoff:
        struct.pack_into("<Q", data, 32, new_phoff)

    # Update e_phnum in ELF header
    struct.pack_into("<H", data, 56, len(phdrs))

    # Update section header for mapped section
    section_shdr = Elf64_Shdr(
        section_shdr.sh_name,
        section_shdr.sh_type,
        section_shdr.sh_flags | SHF_ALLOC,  # Mark as allocated
        new_vaddr,  # Update virtual address
        new_section_offset,  # Update file offset if relocated
        section_shdr.sh_size,
        section_shdr.sh_link,
        section_shdr.sh_info,
        section_shdr.sh_addralign,
        section_shdr.sh_entsize,
    )

    write_section_header(data, ehdr.e_shoff + section_idx * 64, section_shdr)

    # Write output
    output_path.write_bytes(data)
    original_mode = input_path.stat().st_mode
    output_path.chmod(original_mode)

    if verbose:
        print(f"✅ Successfully mapped section to 0x{new_vaddr:x}")

    return new_vaddr


def find_and_update_relocation(
    data: bytearray,
    ehdr: Elf64_Ehdr,
    vaddr: int,
    new_addend: int,
    old_addend: Optional[int] = None,
    verbose: bool = False,
) -> bool:
    """
    Find and update R_X86_64_RELATIVE relocation at given virtual address.

    Args:
        data: ELF binary data
        ehdr: ELF header
        vaddr: Virtual address of the location being relocated
        new_addend: New addend value to set
        old_addend: Optional expected old addend for validation (None = skip check)
        verbose: Print detailed information

    Returns:
        True if relocation found and updated, False otherwise
    """
    # Find all RELA sections
    rela_sections = []

    for i in range(ehdr.e_shnum):
        shdr = read_section_header(data, ehdr.e_shoff + i * 64)
        if shdr.sh_type == SHT_RELA:
            shstrtab_shdr = read_section_header(
                data, ehdr.e_shoff + ehdr.e_shstrndx * 64
            )
            name = get_section_name(data, shstrtab_shdr.sh_offset, shdr.sh_name)
            rela_sections.append((i, shdr, name))
            if verbose:
                print(f"  Found RELA section: {name}")

    if not rela_sections:
        if verbose:
            print("  No RELA sections found")
        return False

    # Search for the relocation
    for section_idx, rela_shdr, section_name in rela_sections:
        entry_count = rela_shdr.sh_size // 24  # sizeof(Elf64_Rela) = 24

        for entry_idx in range(entry_count):
            entry_offset = rela_shdr.sh_offset + entry_idx * 24
            rela = read_rela_entry(data, entry_offset)

            # Check if this is the relocation we're looking for
            if rela.r_offset == vaddr:
                # Extract relocation type
                reloc_type = rela.r_info & 0xFFFFFFFF

                if reloc_type != R_X86_64_RELATIVE:
                    if verbose:
                        print(
                            f"  WARNING: Found relocation at 0x{vaddr:x} "
                            f"but type is {reloc_type}, not R_X86_64_RELATIVE (8)",
                            file=sys.stderr,
                        )
                    continue

                if verbose:
                    print(f"  Found in {section_name}[{entry_idx}]:")
                    print(f"    r_offset: 0x{rela.r_offset:x}")
                    print(f"    r_info: 0x{rela.r_info:x} (type={reloc_type})")
                    print(f"    r_addend: 0x{rela.r_addend:x}")

                # Validate old addend if specified
                if old_addend is not None and rela.r_addend != old_addend:
                    if verbose:
                        print(
                            f"  WARNING: Expected addend 0x{old_addend:x}, "
                            f"found 0x{rela.r_addend:x}",
                            file=sys.stderr,
                        )
                        print(f"  Updating anyway...", file=sys.stderr)

                # Update addend
                new_rela = Elf64_Rela(rela.r_offset, rela.r_info, new_addend)
                write_rela_entry(data, entry_offset, new_rela)

                if verbose:
                    print(
                        f"  ✅ Updated addend: 0x{rela.r_addend:x} -> 0x{new_addend:x}"
                    )

                return True

    # Not found
    if verbose:
        print(f"  Relocation not found at vaddr 0x{vaddr:x}")
    return False


def update_relocation(
    input_path: Path,
    output_path: Path,
    relocation_vaddr: int,
    old_addend: int,
    new_addend: int,
    verbose: bool = True,
) -> bool:
    """
    Update a R_X86_64_RELATIVE relocation's addend.

    Finds the relocation entry where r_offset == relocation_vaddr and updates
    its r_addend from old_addend to new_addend.

    Args:
        input_path: Input ELF binary
        output_path: Output ELF binary
        relocation_vaddr: Virtual address of the location being relocated
        old_addend: Expected old addend value (for validation)
        new_addend: New addend value
        verbose: Print progress information

    Returns:
        True on success, False on error
    """
    # Read input
    data = bytearray(input_path.read_bytes())

    try:
        ehdr = read_elf_header(data)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return False

    if verbose:
        print(f"Updating relocation at vaddr 0x{relocation_vaddr:x}:")
        print(f"  Old addend: 0x{old_addend:x}")
        print(f"  New addend: 0x{new_addend:x}")

    # Use helper to find and update the relocation
    success = find_and_update_relocation(
        data,
        ehdr,
        relocation_vaddr,
        new_addend,
        old_addend=old_addend if old_addend != 0 else None,
        verbose=verbose,
    )

    if not success:
        print(
            f"ERROR: Relocation not found at vaddr 0x{relocation_vaddr:x}",
            file=sys.stderr,
        )
        return False

    # Write output
    output_path.write_bytes(data)
    original_mode = input_path.stat().st_mode
    output_path.chmod(original_mode)

    if verbose:
        print(f"✅ Successfully updated relocation")

    return True


def set_pointer(
    input_path: Path,
    output_path: Path,
    pointer_vaddr: int,
    target_vaddr: int,
    update_relocation: bool = True,
    verbose: bool = True,
) -> bool:
    """
    Write a pointer value and optionally update its relocation.

    This is a high-level operation that combines:
    1. Writing the pointer value directly to the file
    2. Updating any R_X86_64_RELATIVE relocation if it exists

    Args:
        input_path: Input ELF binary
        output_path: Output ELF binary
        pointer_vaddr: Virtual address of pointer location
        target_vaddr: Virtual address to point to
        update_relocation: If True, update R_X86_64_RELATIVE relocation (default: True)
        verbose: Print progress information

    Returns:
        True on success, False on failure

    Important:
        For PIE executables and shared libraries (ET_DYN type), the pointer location
        MUST have an existing R_X86_64_RELATIVE relocation. This function updates
        existing relocations but does not create new ones.

        To ensure a relocation exists, the pointer must be initialized in the source
        code with a relocatable address (e.g., &symbol) rather than a constant
        (e.g., 0x1000).

        Non-PIE executables (ET_EXEC type) do not require relocations, as they use
        absolute addressing.
    """
    # Read input
    data = bytearray(input_path.read_bytes())

    try:
        ehdr = read_elf_header(data)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return False

    # Check if binary is PIE or shared library (requires relocations)
    is_pie = is_pie_or_shared_library(data)

    if verbose:
        print(f"Setting pointer at 0x{pointer_vaddr:x} to 0x{target_vaddr:x}")

    # Find PT_LOAD containing pointer_vaddr
    pointer_load = None
    for i in range(ehdr.e_phnum):
        phdr = read_program_header(data, ehdr.e_phoff + i * 56)
        if phdr.p_type == PT_LOAD:
            if phdr.p_vaddr <= pointer_vaddr < phdr.p_vaddr + phdr.p_memsz:
                pointer_load = phdr
                break

    if pointer_load is None:
        print(f"ERROR: No PT_LOAD contains vaddr 0x{pointer_vaddr:x}", file=sys.stderr)
        return False

    # Calculate file offset
    offset_in_segment = pointer_vaddr - pointer_load.p_vaddr
    file_offset = pointer_load.p_offset + offset_in_segment

    if file_offset + 8 > len(data):
        print(
            f"ERROR: File offset 0x{file_offset:x} + 8 exceeds file size",
            file=sys.stderr,
        )
        return False

    if verbose:
        print(f"  File offset: 0x{file_offset:x}")

    # Write the pointer value (8 bytes, little-endian)
    struct.pack_into("<Q", data, file_offset, target_vaddr)

    if verbose:
        print(f"  ✅ Wrote pointer value to file")

    # Update relocation if requested
    if update_relocation:
        if verbose:
            print(f"  Checking for relocation at 0x{pointer_vaddr:x}...")

        reloc_updated = find_and_update_relocation(
            data,
            ehdr,
            pointer_vaddr,
            target_vaddr,
            old_addend=None,  # Don't validate old value
            verbose=verbose,
        )

        if reloc_updated:
            if verbose:
                print(f"  ✅ Updated relocation")
        else:
            # PIE/shared libraries REQUIRE relocations for pointers
            if is_pie:
                print(
                    f"ERROR: Binary is PIE/shared library but no relocation found at 0x{pointer_vaddr:x}",
                    file=sys.stderr,
                )
                print(
                    f"  File: {input_path}",
                    file=sys.stderr,
                )
                print(
                    f"  PIE binaries require R_X86_64_RELATIVE relocations for pointers to work correctly",
                    file=sys.stderr,
                )
                print(
                    f"  The pointer location must be initialized with a relocatable address (e.g., &symbol)",
                    file=sys.stderr,
                )
                print(
                    f"  Constant addresses (like 0x1000) do not generate relocations",
                    file=sys.stderr,
                )
                return False
            else:
                # Non-PIE executables: relocation is optional
                if verbose:
                    print(f"  ℹ️  No relocation found (OK for non-PIE binary)")

    # Write output
    output_path.write_bytes(data)
    original_mode = input_path.stat().st_mode
    output_path.chmod(original_mode)

    if verbose:
        print(f"✅ Successfully set pointer")

    return True


def main(argv: List[str]) -> int:
    """
    Main entry point for the tool.

    Args:
        argv: Command-line arguments (excluding program name)

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Extended ELF manipulation tool for zero-paging and PT_LOAD mapping"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # zero-page command
    zero_parser = subparsers.add_parser("zero-page", help="Zero-page a section")
    zero_parser.add_argument("input", type=Path, help="Input ELF binary")
    zero_parser.add_argument("output", type=Path, help="Output ELF binary")
    zero_parser.add_argument(
        "--section", default=".hip_fatbin", help="Section name to zero-page"
    )
    zero_parser.add_argument(
        "--phdr-spare-slots",
        type=int,
        default=16,
        help="Number of spare PHDR slots to allocate when relocating (default: 16)",
    )
    zero_parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress output"
    )

    # map-section command
    map_parser = subparsers.add_parser("map-section", help="Map section to new PT_LOAD")
    map_parser.add_argument("input", type=Path, help="Input ELF binary")
    map_parser.add_argument("output", type=Path, help="Output ELF binary")
    map_parser.add_argument(
        "--section", required=True, help="Section name to map (e.g., .rocm_kpack_ref)"
    )
    map_parser.add_argument(
        "--vaddr",
        type=lambda x: int(x, 0),
        help="Virtual address (hex, auto-allocate if not specified)",
    )
    map_parser.add_argument(
        "--phdr-spare-slots",
        type=int,
        default=16,
        help="Number of spare PHDR slots to allocate when relocating (default: 16)",
    )
    map_parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress output"
    )

    # update-relocation command
    reloc_parser = subparsers.add_parser(
        "update-relocation", help="Update a relocation addend"
    )
    reloc_parser.add_argument("input", type=Path, help="Input ELF binary")
    reloc_parser.add_argument("output", type=Path, help="Output ELF binary")
    reloc_parser.add_argument(
        "--vaddr",
        required=True,
        type=lambda x: int(x, 0),
        help="Virtual address of relocation",
    )
    reloc_parser.add_argument(
        "--old-addend",
        type=lambda x: int(x, 0),
        default=0,
        help="Expected old addend (for validation)",
    )
    reloc_parser.add_argument(
        "--new-addend", required=True, type=lambda x: int(x, 0), help="New addend value"
    )
    reloc_parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress output"
    )

    # set-pointer command
    pointer_parser = subparsers.add_parser(
        "set-pointer", help="Write pointer value and update relocation"
    )
    pointer_parser.add_argument("input", type=Path, help="Input ELF binary")
    pointer_parser.add_argument("output", type=Path, help="Output ELF binary")
    pointer_parser.add_argument(
        "--at",
        required=True,
        type=lambda x: int(x, 0),
        help="Virtual address of pointer location",
    )
    pointer_parser.add_argument(
        "--target",
        required=True,
        type=lambda x: int(x, 0),
        help="Virtual address to point to",
    )
    pointer_parser.add_argument(
        "--no-relocation",
        action="store_true",
        help="Skip relocation update (direct write only)",
    )
    pointer_parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress output"
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        return 1

    success = False

    if args.command == "zero-page":
        success = conservative_zero_page(
            args.input,
            args.output,
            section_name=args.section,
            verbose=not args.quiet,
            phdr_spare_slots=args.phdr_spare_slots,
        )
    elif args.command == "map-section":
        result = map_section_to_new_load(
            args.input,
            args.output,
            section_name=args.section,
            new_vaddr=args.vaddr,
            verbose=not args.quiet,
            phdr_spare_slots=args.phdr_spare_slots,
        )
        success = result is not None
    elif args.command == "update-relocation":
        success = update_relocation(
            args.input,
            args.output,
            relocation_vaddr=args.vaddr,
            old_addend=args.old_addend,
            new_addend=args.new_addend,
            verbose=not args.quiet,
        )
    elif args.command == "set-pointer":
        success = set_pointer(
            args.input,
            args.output,
            pointer_vaddr=args.at,
            target_vaddr=args.target,
            update_relocation=not args.no_relocation,
            verbose=not args.quiet,
        )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

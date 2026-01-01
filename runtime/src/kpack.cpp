// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <cstring>

#include "kpack_internal.h"

extern "C" {

// Archive opening/closing implemented in archive.cpp

kpack_error_t kpack_get_architecture_count(kpack_archive_t archive,
                                           size_t* count) {
  if (!archive || !count) {
    return KPACK_ERROR_INVALID_ARGUMENT;
  }

  *count = archive->gfx_arches.size();
  return KPACK_SUCCESS;
}

kpack_error_t kpack_get_architecture(kpack_archive_t archive, size_t index,
                                     const char** arch) {
  if (!archive || !arch) {
    return KPACK_ERROR_INVALID_ARGUMENT;
  }

  if (index >= archive->gfx_arches.size()) {
    return KPACK_ERROR_INVALID_ARGUMENT;
  }

  *arch = archive->gfx_arches[index].c_str();
  return KPACK_SUCCESS;
}

kpack_error_t kpack_get_binary_count(kpack_archive_t archive, size_t* count) {
  if (!archive || !count) {
    return KPACK_ERROR_INVALID_ARGUMENT;
  }

  *count = archive->binary_names.size();
  return KPACK_SUCCESS;
}

kpack_error_t kpack_get_binary(kpack_archive_t archive, size_t index,
                               const char** binary) {
  if (!archive || !binary) {
    return KPACK_ERROR_INVALID_ARGUMENT;
  }

  if (index >= archive->binary_names.size()) {
    return KPACK_ERROR_INVALID_ARGUMENT;
  }

  *binary = archive->binary_names[index].c_str();
  return KPACK_SUCCESS;
}

// Helper to find kernel metadata, supporting indexed keys for RDC binaries
// If binary_name not found, tries binary_name#0, binary_name#1, etc.
// Returns all matching kernels concatenated with length prefixes:
// [uint64_t size1][data1][uint64_t size2][data2]...
static kpack_error_t find_kernel_entries(
    kpack_archive_t archive, const char* binary_name, const char* arch,
    std::vector<const kpack::KernelMetadata*>& entries) {
  // First try exact match
  auto binary_it = archive->toc.find(binary_name);
  if (binary_it != archive->toc.end()) {
    auto arch_it = binary_it->second.find(arch);
    if (arch_it != binary_it->second.end()) {
      entries.push_back(&arch_it->second);
      return KPACK_SUCCESS;
    }
  }

  // Try indexed keys: binary_name#0, binary_name#1, ...
  // This handles RDC binaries with multiple code objects
  std::string base_name(binary_name);
  for (int idx = 0; idx < 100; ++idx) {  // Cap at 100 to prevent infinite loop
    std::string indexed_name = base_name + "#" + std::to_string(idx);
    auto it = archive->toc.find(indexed_name);
    if (it == archive->toc.end()) {
      break;  // No more indexed entries
    }
    auto arch_it = it->second.find(arch);
    if (arch_it != it->second.end()) {
      entries.push_back(&arch_it->second);
    }
  }

  return entries.empty() ? KPACK_ERROR_KERNEL_NOT_FOUND : KPACK_SUCCESS;
}

kpack_error_t kpack_get_kernel(kpack_archive_t archive, const char* binary_name,
                               const char* arch, void** kernel_data,
                               size_t* kernel_size) {
  if (!archive || !binary_name || !arch || !kernel_data || !kernel_size) {
    return KPACK_ERROR_INVALID_ARGUMENT;
  }

  // Find all matching kernel entries (handles indexed keys for RDC)
  std::vector<const kpack::KernelMetadata*> entries;
  kpack_error_t find_err =
      find_kernel_entries(archive, binary_name, arch, entries);
  if (find_err != KPACK_SUCCESS) {
    return find_err;
  }

  // Lock for kernel_cache access (decompression modifies shared buffer)
  std::lock_guard<std::mutex> lock(archive->kernel_mutex);

  // For single entry, use original simple path
  if (entries.size() == 1) {
    const kpack::KernelMetadata& km = *entries[0];

    // Decompress based on scheme
    kpack_error_t err;
    if (archive->compression_scheme == KPACK_COMPRESSION_NOOP) {
      err = kpack::decompress_noop(archive, km.ordinal, km.original_size);
    } else if (archive->compression_scheme ==
               KPACK_COMPRESSION_ZSTD_PER_KERNEL) {
      err = kpack::decompress_zstd(archive, km.ordinal, km.original_size);
    } else {
      return KPACK_ERROR_NOT_IMPLEMENTED;
    }

    if (err != KPACK_SUCCESS) {
      return err;
    }

    // Allocate copy for caller
    size_t size = archive->kernel_cache.size();
    void* copy = std::malloc(size);
    if (!copy) {
      return KPACK_ERROR_OUT_OF_MEMORY;
    }
    std::memcpy(copy, archive->kernel_cache.data(), size);

    *kernel_data = copy;
    *kernel_size = size;
    return KPACK_SUCCESS;
  }

  // Multiple entries (RDC case): concatenate with length prefixes
  // Format: [uint64_t count][uint64_t size1][data1][uint64_t size2][data2]...
  // First pass: decompress all and calculate total size
  std::vector<std::vector<uint8_t>> decompressed;
  decompressed.reserve(entries.size());
  size_t total_size = sizeof(uint64_t);  // count header

  for (const auto* km : entries) {
    kpack_error_t err;
    if (archive->compression_scheme == KPACK_COMPRESSION_NOOP) {
      err = kpack::decompress_noop(archive, km->ordinal, km->original_size);
    } else if (archive->compression_scheme ==
               KPACK_COMPRESSION_ZSTD_PER_KERNEL) {
      err = kpack::decompress_zstd(archive, km->ordinal, km->original_size);
    } else {
      return KPACK_ERROR_NOT_IMPLEMENTED;
    }

    if (err != KPACK_SUCCESS) {
      return err;
    }

    decompressed.emplace_back(archive->kernel_cache.begin(),
                              archive->kernel_cache.end());
    total_size += sizeof(uint64_t) + decompressed.back().size();
  }

  // Allocate and pack
  uint8_t* buffer = static_cast<uint8_t*>(std::malloc(total_size));
  if (!buffer) {
    return KPACK_ERROR_OUT_OF_MEMORY;
  }

  uint8_t* ptr = buffer;

  // Write count
  uint64_t count = entries.size();
  std::memcpy(ptr, &count, sizeof(count));
  ptr += sizeof(count);

  // Write each entry with size prefix
  for (const auto& data : decompressed) {
    uint64_t size = data.size();
    std::memcpy(ptr, &size, sizeof(size));
    ptr += sizeof(size);
    std::memcpy(ptr, data.data(), size);
    ptr += size;
  }

  *kernel_data = buffer;
  *kernel_size = total_size;
  return KPACK_SUCCESS;
}

void kpack_free_kernel(void* kernel_data) { std::free(kernel_data); }

}  // extern "C"

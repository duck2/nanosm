#pragma once

#include "compiler.hpp"

#include <array>
#include <cstddef>
#include <stdint.h>
#include <string>
#include <utility>

namespace nanosm {

inline constexpr std::size_t tma_descriptor_bytes = 64;

// The direction is the instruction's (tma.load / tma.store), not the
// descriptor's: one descriptor serves both directions of a tensor, as an
// NVIDIA tensor map does.
enum class TmaDirection : uint8_t {
    load = 0,
    store = 1,
};

enum class TmaSwizzle : uint8_t { linear = 0, xor64_a = 1, xor64_b = 2 };

// The descriptor describes the global tensor and the box only: where the box
// lands in shared memory and which mbar counts its bytes are operands of the
// instruction (TmaAddresses below), as in cp.async.bulk.tensor. Rank 1 or 2:
// the instruction's coordinates are two registers (x, y), because one RF read
// round reads at most one register per bank (trash/rfc14.md) and the third
// register carries the addresses. `box` is in the shared layout's frame
// (box[0] along u, box[1] along v; see tma_shmem_address); with transpose the
// global walk swaps them.
struct TmaDescriptor {
    uint8_t rank = 1;
    bool transpose = false;
    uint8_t element_bytes = 4;
    uint8_t swizzle = static_cast<uint8_t>(TmaSwizzle::linear);
    uint64_t gmem_base = 0;
    std::array<uint32_t, 2> dimensions{1, 1};
    uint32_t row_stride = 0; // elements from one global row to the next (rank 2)
    std::array<uint32_t, 2> box{1, 1};
};

// The 64-byte encoding. Bytes 4..7 and 36..63 are reserved and must be zero
// (kept free for an indirect row dimension, trash/rfc15.md addendum 2).
namespace tma_encoding {

inline constexpr std::size_t rank = 0;
inline constexpr std::size_t transpose = 1;
inline constexpr std::size_t element_bytes = 2;
inline constexpr std::size_t swizzle = 3;
inline constexpr std::size_t gmem_base = 8;
inline constexpr std::size_t dimensions = 16;
inline constexpr std::size_t row_stride = 24;
inline constexpr std::size_t box = 28;
inline constexpr std::size_t used_end = 36;

} // namespace tma_encoding

// The xor64 layouts: an operand object of `outer` rows (A: m, B: n), each row
// 16 bytes per K slice of E = 16 / element_bytes elements; slice s sits at
// 16 x outer x s. The outer extent is the object's own row count -- the same
// spacing the MMA reads an operand of that many rows with (src/sim.cpp
// mma_a_address / mma_b_address) -- so a box is always a compact object.
// Outer extents: the powers of two 4 .. 128 (a core block of rows up to a
// 128-row operand; tests/tma_test.cpp checks every one against the MMA's own
// operand addressing). K: a multiple of E, at most 64 elements.
inline constexpr uint32_t tma_xor64_min_outer = 4;
inline constexpr uint32_t tma_xor64_max_outer = 128;
inline constexpr uint32_t tma_xor64_max_k = 64;

inline bool is_xor64(const TmaDescriptor& descriptor) {
    return descriptor.swizzle != static_cast<uint8_t>(TmaSwizzle::linear);
}

// For an xor64 box: K along u for the A role, along v for the B role; the
// outer extent is the other one.
inline uint32_t tma_xor64_k_extent(const TmaDescriptor& descriptor) {
    return descriptor.swizzle == static_cast<uint8_t>(TmaSwizzle::xor64_a) ? descriptor.box[0]
                                                                          : descriptor.box[1];
}

inline uint32_t tma_xor64_outer_extent(const TmaDescriptor& descriptor) {
    return descriptor.swizzle == static_cast<uint8_t>(TmaSwizzle::xor64_a) ? descriptor.box[1]
                                                                          : descriptor.box[0];
}

inline void validate_tma_descriptor(const TmaDescriptor& descriptor) {
    if (descriptor.rank < 1 || descriptor.rank > 2) {
        throw SimError("tma descriptor rank must be 1 or 2");
    }
    if (descriptor.swizzle > static_cast<uint8_t>(TmaSwizzle::xor64_b)) {
        throw SimError("tma swizzle is invalid");
    }
    if (descriptor.element_bytes != 1 && descriptor.element_bytes != 2 &&
        descriptor.element_bytes != 4) {
        throw SimError("tma element size must be 1, 2 or 4 bytes");
    }
    for (uint32_t dimension = 0; dimension < 2; ++dimension) {
        if (dimension < descriptor.rank) {
            if (descriptor.dimensions[dimension] == 0 || descriptor.box[dimension] == 0) {
                throw SimError("tma active dimensions and box dimensions must be nonzero");
            }
        } else if (descriptor.dimensions[dimension] != 1 || descriptor.box[dimension] != 1) {
            throw SimError("tma inactive dimensions and box dimensions must be one");
        }
    }
    if (descriptor.rank == 1 && descriptor.row_stride != 0) {
        throw SimError("rank-1 tma descriptor has a nonzero row stride");
    }
    if (descriptor.rank == 2 && descriptor.row_stride == 0) {
        throw SimError("rank-2 tma descriptor needs a row stride");
    }
    if (descriptor.gmem_base % descriptor.element_bytes != 0) {
        throw SimError("tma global base is not element-aligned");
    }
    const bool swizzled = is_xor64(descriptor);
    if (descriptor.transpose && !swizzled) {
        throw SimError("linear tma transpose is unsupported by v0");
    }
    if (swizzled) {
        if (descriptor.rank != 2 ||
            (descriptor.element_bytes != 2 && descriptor.element_bytes != 4)) {
            throw SimError("xor64 tma requires rank 2 and 2- or 4-byte elements");
        }
        // The xor64 layout is a byte permutation within 16-byte rows
        // (xor64_byte_address); the element type only sets how many elements
        // share a row: E = 16 / element_bytes (8 fp16, 4 fp32). The K extent
        // spans K / E slices (a multiple of E, up to 64 elements); the outer
        // extent is the object's row count (a power of two, 4 .. 128), so
        // span = K x outer x element_bytes = slices x 16 x outer.
        const uint32_t row_elements = 16u / descriptor.element_bytes;
        const uint32_t k_extent = tma_xor64_k_extent(descriptor);
        const uint32_t outer = tma_xor64_outer_extent(descriptor);
        if (k_extent == 0 || k_extent > tma_xor64_max_k || k_extent % row_elements != 0) {
            throw SimError("xor64 tma K extent must be a multiple of " + std::to_string(row_elements) +
                " elements and at most " + std::to_string(tma_xor64_max_k));
        }
        if (outer < tma_xor64_min_outer || outer > tma_xor64_max_outer || (outer & (outer - 1)) != 0) {
            throw SimError("xor64 tma outer extent must be a power of two in " +
                std::to_string(tma_xor64_min_outer) + ".." + std::to_string(tma_xor64_max_outer) +
                " (it is " + std::to_string(outer) + ")");
        }
    }
}

// The tma.load / tma.store address operand: ONE register (the instruction's
// three registers fill the one RF read round, one per bank) holding
//     bits 15:0   the completion mbar's address in 8-byte units
//     bits 31:16  the box's shared address in 64-byte units
// so every mbar is 8-byte aligned and every box starts 64-byte aligned by
// construction (an xor64 object needs 64; NVIDIA asks 128 of a tensor copy's
// destination), and the operand reaches 512 KiB of mbars and 4 MiB of boxes.
// The mbar sits low so that one mov (a 22-bit signed immediate) loads the
// whole operand whenever the box starts below 2 KiB. A kernel with the box's
// shared address in rS and the mbar's in rM builds it as
//     shli rA, rS, 10          ; shared / 64 << 16 (rS 64-byte aligned)
//     shri rT, rM, 3           ; mbar / 8
//     or   rA, rA, rT
// The machine checks only ranges (the box and the mbar inside shmem).
inline constexpr uint32_t tma_mbar_unit_bytes = 8;
inline constexpr uint32_t tma_shared_unit_bytes = 64;

struct TmaAddresses {
    uint32_t shmem = 0; // byte address
    uint32_t mbar = 0;  // byte address
};

inline TmaAddresses unpack_tma_addresses(uint32_t value) {
    return {(value >> 16) * tma_shared_unit_bytes, (value & 0xffffu) * tma_mbar_unit_bytes};
}

inline uint32_t pack_tma_addresses(uint32_t shmem, uint32_t mbar) {
    if (shmem % tma_shared_unit_bytes != 0 || mbar % tma_mbar_unit_bytes != 0) {
        throw SimError("tma address operand: the shared address must be 64-byte aligned and the mbar "
            "8-byte aligned");
    }
    if (shmem / tma_shared_unit_bytes > 0xffffu || mbar / tma_mbar_unit_bytes > 0xffffu) {
        throw SimError("tma address operand: the shared address must be below 4 MiB and the mbar below 512 KiB");
    }
    return (shmem / tma_shared_unit_bytes) << 16 | mbar / tma_mbar_unit_bytes;
}

// The xor64 layout, defined on bytes and element-agnostic. A K slice of an
// object of `outer` rows holds one sixteen-byte row per outer index; `outer`
// selects the row and `byte_in_row` (0..15) the byte within it. Two
// address-bit permutations:
//   addr[2] ^= addr[6]   (word-in-row bit 0 ^= (outer >> 2) & 1)
//   addr[3] ^= phase>>1  (B operands, phase 2: word-in-row bit 1 ^= 1)
// `phase` is in word units (0 for A, 2 for B), as the MMA core has always
// passed it. This is the single source of truth for the layout: the MMA
// operand fetch, the TMA walker (both directions, any element size) and,
// written out in SIMT arithmetic, kernels that sts into such a tile.
inline uint32_t xor64_byte_address(
    uint32_t base, uint32_t outer, uint32_t byte_in_row, uint32_t phase) {
    return base + 16u * outer +
        (((byte_in_row & ~3u) ^ (4u * ((outer >> 2) & 1u)) ^ (4u * phase)) |
         (byte_in_row & 3u));
}

// fp16 operand element k (0..7) of row `outer`: byte_in_row = 2*k.
inline uint32_t xor64_operand_address(
    uint32_t base, uint32_t outer, uint32_t k, uint32_t phase) {
    return xor64_byte_address(base, outer, 2u * k, phase);
}

inline uint64_t tma_logical_bytes(const TmaDescriptor& descriptor) {
    validate_tma_descriptor(descriptor);
    uint64_t elements = 1;
    for (uint32_t dimension = 0; dimension < descriptor.rank; ++dimension) {
        if (elements > UINT64_MAX / descriptor.box[dimension]) {
            throw SimError("tma logical box size overflows");
        }
        elements *= descriptor.box[dimension];
    }
    if (elements > UINT64_MAX / descriptor.element_bytes) {
        throw SimError("tma logical byte count overflows");
    }
    return elements * descriptor.element_bytes;
}

// The shmem byte span a command's box occupies: [shmem, shmem + logical
// bytes). A linear box is dense in it; an xor64 box is a byte permutation of
// its K / E slices of 16 x outer bytes (xor64_byte_address never leaves a
// 16-byte row), so both layouts cover exactly this span. Load OOB elements
// are zero-filled inside it; store OOB elements skip their (in-span) source
// bytes. The machine's runtime "box fits in shmem" check is the same
// arithmetic.
inline std::pair<uint64_t, uint64_t> tma_shmem_span(const TmaDescriptor& descriptor, uint32_t shmem) {
    const auto bytes = tma_logical_bytes(descriptor);
    return {shmem, static_cast<uint64_t>(shmem) + bytes};
}

inline std::array<uint8_t, tma_descriptor_bytes> encode_tma_descriptor(
    const TmaDescriptor& descriptor) {
    validate_tma_descriptor(descriptor);
    std::array<uint8_t, tma_descriptor_bytes> bytes{};
    const auto put32 = [&](std::size_t offset, uint32_t value) {
        for (unsigned byte = 0; byte < 4; ++byte) {
            bytes[offset + byte] = static_cast<uint8_t>(value >> (byte * 8));
        }
    };
    const auto put64 = [&](std::size_t offset, uint64_t value) {
        for (unsigned byte = 0; byte < 8; ++byte) {
            bytes[offset + byte] = static_cast<uint8_t>(value >> (byte * 8));
        }
    };
    bytes[tma_encoding::rank] = descriptor.rank;
    bytes[tma_encoding::transpose] = descriptor.transpose;
    bytes[tma_encoding::element_bytes] = descriptor.element_bytes;
    bytes[tma_encoding::swizzle] = descriptor.swizzle;
    put64(tma_encoding::gmem_base, descriptor.gmem_base);
    for (std::size_t i = 0; i < descriptor.dimensions.size(); ++i) {
        put32(tma_encoding::dimensions + i * 4, descriptor.dimensions[i]);
    }
    put32(tma_encoding::row_stride, descriptor.row_stride);
    for (std::size_t i = 0; i < descriptor.box.size(); ++i) {
        put32(tma_encoding::box + i * 4, descriptor.box[i]);
    }
    return bytes;
}

inline TmaDescriptor decode_tma_descriptor(const uint8_t* bytes, std::size_t size) {
    if (bytes == nullptr || size != tma_descriptor_bytes) {
        throw SimError("tma descriptor must contain exactly 64 bytes");
    }
    const auto get32 = [&](std::size_t offset) {
        uint32_t value = 0;
        for (unsigned byte = 0; byte < 4; ++byte) {
            value |= static_cast<uint32_t>(bytes[offset + byte]) << (byte * 8);
        }
        return value;
    };
    const auto get64 = [&](std::size_t offset) {
        uint64_t value = 0;
        for (unsigned byte = 0; byte < 8; ++byte) {
            value |= static_cast<uint64_t>(bytes[offset + byte]) << (byte * 8);
        }
        return value;
    };
    for (std::size_t offset = 4; offset < 8; ++offset) {
        if (bytes[offset] != 0) throw SimError("tma descriptor reserved header bytes are nonzero");
    }
    for (std::size_t offset = tma_encoding::used_end; offset < tma_descriptor_bytes; ++offset) {
        if (bytes[offset] != 0) throw SimError("tma descriptor reserved tail bytes are nonzero");
    }
    if (bytes[tma_encoding::transpose] > 1) throw SimError("tma descriptor transpose byte must be 0 or 1");
    TmaDescriptor descriptor;
    descriptor.rank = bytes[tma_encoding::rank];
    descriptor.transpose = bytes[tma_encoding::transpose] != 0;
    descriptor.element_bytes = bytes[tma_encoding::element_bytes];
    descriptor.swizzle = bytes[tma_encoding::swizzle];
    descriptor.gmem_base = get64(tma_encoding::gmem_base);
    for (std::size_t i = 0; i < descriptor.dimensions.size(); ++i) {
        descriptor.dimensions[i] = get32(tma_encoding::dimensions + i * 4);
    }
    descriptor.row_stride = get32(tma_encoding::row_stride);
    for (std::size_t i = 0; i < descriptor.box.size(); ++i) {
        descriptor.box[i] = get32(tma_encoding::box + i * 4);
    }
    validate_tma_descriptor(descriptor);
    return descriptor;
}

} // namespace nanosm

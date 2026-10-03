#pragma once

#include "compiler.hpp"

#include <array>
#include <cstddef>
#include <stdint.h>
#include <utility>

namespace nanosm {

inline constexpr std::size_t tma_descriptor_bytes = 64;

enum class TmaDirection : uint8_t {
    load = 0,
    store = 1,
};

enum class TmaSwizzle : uint8_t { linear = 0, xor64_a = 1, xor64_b = 2 };

struct TmaDescriptor {
    uint8_t rank = 1;
    TmaDirection direction = TmaDirection::load;
    bool transpose = false;
    uint8_t element_bytes = 4;
    uint8_t swizzle = static_cast<uint8_t>(TmaSwizzle::linear);
    uint64_t gmem_base = 0;
    std::array<uint32_t, 3> dimensions{1, 1, 1};
    std::array<uint32_t, 2> strides{0, 0};
    std::array<uint32_t, 3> box{1, 1, 1};
    uint32_t shmem_base = 0;
    uint32_t completion_mbar = 0;
};

namespace tma_encoding {

inline constexpr std::size_t rank = 0;
inline constexpr std::size_t direction = 1;
inline constexpr std::size_t transpose = 2;
inline constexpr std::size_t element_bytes = 3;
inline constexpr std::size_t swizzle = 4;
inline constexpr std::size_t gmem_base = 8;
inline constexpr std::size_t dimensions = 16;
inline constexpr std::size_t strides = 28;
inline constexpr std::size_t box = 36;
inline constexpr std::size_t shmem_base = 48;
inline constexpr std::size_t completion_mbar = 52;

} // namespace tma_encoding

inline void validate_tma_descriptor(const TmaDescriptor& descriptor) {
    if (descriptor.rank < 1 || descriptor.rank > 3) {
        throw SimError("tma descriptor rank must be 1..3");
    }
    if (descriptor.direction != TmaDirection::load &&
        descriptor.direction != TmaDirection::store) {
        throw SimError("tma descriptor direction is invalid");
    }
    if (descriptor.swizzle > static_cast<uint8_t>(TmaSwizzle::xor64_b)) {
        throw SimError("tma swizzle is invalid");
    }
    if (descriptor.element_bytes != 1 && descriptor.element_bytes != 2 &&
        descriptor.element_bytes != 4) {
        throw SimError("tma element size must be 1, 2 or 4 bytes");
    }
    for (uint32_t dimension = 0; dimension < 3; ++dimension) {
        if (dimension < descriptor.rank) {
            if (descriptor.dimensions[dimension] == 0 || descriptor.box[dimension] == 0) {
                throw SimError("tma active dimensions and box dimensions must be nonzero");
            }
        } else if (descriptor.dimensions[dimension] != 1 || descriptor.box[dimension] != 1) {
            throw SimError("tma inactive dimensions and box dimensions must be one");
        }
    }
    if (descriptor.rank == 1 && descriptor.strides != std::array<uint32_t, 2>{0, 0}) {
        throw SimError("rank-1 tma descriptor has nonzero strides");
    }
    if (descriptor.rank == 2 &&
        (descriptor.strides[0] == 0 || descriptor.strides[1] != 0)) {
        throw SimError("rank-2 tma descriptor needs one row stride");
    }
    if (descriptor.rank == 3 &&
        (descriptor.strides[0] == 0 || descriptor.strides[1] == 0)) {
        throw SimError("rank-3 tma descriptor needs row and plane strides");
    }
    if (descriptor.shmem_base % descriptor.element_bytes != 0) {
        throw SimError("tma shmem base is not element-aligned");
    }
    if (descriptor.gmem_base % descriptor.element_bytes != 0) {
        throw SimError("tma global base is not element-aligned");
    }
    if (descriptor.completion_mbar % 8 != 0) {
        throw SimError("tma completion mbar is not 8-byte aligned");
    }
    const bool swizzled = descriptor.swizzle != static_cast<uint8_t>(TmaSwizzle::linear);
    if (descriptor.transpose && !swizzled) {
        throw SimError("linear tma transpose is unsupported by v0");
    }
    if (swizzled) {
        if (descriptor.rank != 2 ||
            (descriptor.element_bytes != 2 && descriptor.element_bytes != 4)) {
            throw SimError("xor64 tma requires rank 2 and 2- or 4-byte elements");
        }
        // The xor64 layout is a byte permutation of 1024-byte objects of 64
        // sixteen-byte rows (xor64_byte_address); the element type only sets
        // how many elements share a row: E = 16 / element_bytes (8 fp16, 4
        // fp32). Stage-granular boxes: the K axis spans K/E consecutive
        // objects (K extent a multiple of E, up to 64 elements: 1..8 fp16
        // objects, 1..16 fp32 objects); the outer axis stays the canonical
        // 64. Span = K*64*element_bytes = objects*1024, so the runtime span
        // check scales with the box through tma_logical_bytes.
        const uint32_t row_elements = 16u / descriptor.element_bytes;
        const bool is_a = descriptor.swizzle == static_cast<uint8_t>(TmaSwizzle::xor64_a);
        const uint32_t k_extent = is_a ? descriptor.box[0] : descriptor.box[1];
        const uint32_t outer_extent = is_a ? descriptor.box[1] : descriptor.box[0];
        if (k_extent == 0 || k_extent > 64 || k_extent % row_elements != 0 ||
            outer_extent != 64 || descriptor.box[2] != 1) {
            throw SimError("xor64 tma has the wrong logical box geometry");
        }
        if (descriptor.shmem_base % 64 != 0) {
            throw SimError("xor64 tma shmem base is not 64-byte aligned");
        }
    }
}

// The xor64 layout, defined on bytes and element-agnostic. One 1024-byte
// object holds 64 sixteen-byte rows; `outer` (0..63) selects the row and
// `byte_in_row` (0..15) the byte within it. Two address-bit permutations:
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

// The shmem byte span a command's box occupies: [shmem_base, shmem_base +
// logical bytes). A linear box is dense in it; an xor64 box is a byte
// permutation of K/E whole 1024-byte objects (xor64_byte_address never leaves
// its object), so both layouts cover exactly this span. Load OOB elements are
// zero-filled inside it; store OOB elements skip their (in-span) source bytes.
// The machine's runtime "box fits in shmem" check is the same arithmetic.
inline std::pair<uint64_t, uint64_t> tma_shmem_span(const TmaDescriptor& descriptor) {
    const auto bytes = tma_logical_bytes(descriptor);
    return {descriptor.shmem_base, static_cast<uint64_t>(descriptor.shmem_base) + bytes};
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
    bytes[tma_encoding::direction] = static_cast<uint8_t>(descriptor.direction);
    bytes[tma_encoding::transpose] = descriptor.transpose;
    bytes[tma_encoding::element_bytes] = descriptor.element_bytes;
    bytes[tma_encoding::swizzle] = descriptor.swizzle;
    put64(tma_encoding::gmem_base, descriptor.gmem_base);
    for (std::size_t i = 0; i < descriptor.dimensions.size(); ++i) {
        put32(tma_encoding::dimensions + i * 4, descriptor.dimensions[i]);
    }
    for (std::size_t i = 0; i < descriptor.strides.size(); ++i) {
        put32(tma_encoding::strides + i * 4, descriptor.strides[i]);
    }
    for (std::size_t i = 0; i < descriptor.box.size(); ++i) {
        put32(tma_encoding::box + i * 4, descriptor.box[i]);
    }
    put32(tma_encoding::shmem_base, descriptor.shmem_base);
    put32(tma_encoding::completion_mbar, descriptor.completion_mbar);
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
    for (std::size_t offset = 5; offset < 8; ++offset) {
        if (bytes[offset] != 0) throw SimError("tma descriptor reserved header bits are nonzero");
    }
    for (std::size_t offset = 56; offset < 64; ++offset) {
        if (bytes[offset] != 0) throw SimError("tma descriptor reserved tail bits are nonzero");
    }
    TmaDescriptor descriptor;
    descriptor.rank = bytes[tma_encoding::rank];
    descriptor.direction = static_cast<TmaDirection>(bytes[tma_encoding::direction]);
    descriptor.transpose = bytes[tma_encoding::transpose] != 0;
    descriptor.element_bytes = bytes[tma_encoding::element_bytes];
    descriptor.swizzle = bytes[tma_encoding::swizzle];
    descriptor.gmem_base = get64(tma_encoding::gmem_base);
    for (std::size_t i = 0; i < descriptor.dimensions.size(); ++i) {
        descriptor.dimensions[i] = get32(tma_encoding::dimensions + i * 4);
    }
    for (std::size_t i = 0; i < descriptor.strides.size(); ++i) {
        descriptor.strides[i] = get32(tma_encoding::strides + i * 4);
    }
    for (std::size_t i = 0; i < descriptor.box.size(); ++i) {
        descriptor.box[i] = get32(tma_encoding::box + i * 4);
    }
    descriptor.shmem_base = get32(tma_encoding::shmem_base);
    descriptor.completion_mbar = get32(tma_encoding::completion_mbar);
    validate_tma_descriptor(descriptor);
    return descriptor;
}

} // namespace nanosm

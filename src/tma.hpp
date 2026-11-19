#pragma once

#include "compiler.hpp"
#include "layout.hpp"

#include <array>
#include <cstddef>
#include <stdint.h>
#include <string>
#include <utility>

namespace nanosm {

// The TMA (trash/rfc17.md; trash/rfc17-slice-2026-11-17.md;
// trash/rfc17-layout-2026-11-18.md; trash/whole-rows-2026-11-18.md).
// PROVISIONAL: a first iteration, to be looked at through kernels and
// changed.
//
// A descriptor defines one memory movement of a tensor: what the tensor is
// (base, element size, four dimensions and their strides), how big a tile is
// and how the tile is laid out in shared memory. It is written once, in
// shared memory, and never touched while the kernel runs. The instruction
// says which tile (its coordinates), where it lands (the buffer) and who to
// tell (the barrier):
//     tma.load.Nd   [urS], [DESC, urC], [urM]
//     tma.store.Nd  [DESC, urC], [urS], [urM]
// so one descriptor serves both directions, every stage buffer and every
// tile of its tensor.
inline constexpr std::size_t tma_descriptor_bytes = 64;
inline constexpr uint32_t tma_max_coordinates = 4;

// The direction is the instruction's (tma.load / tma.store).
enum class TmaDirection : uint8_t {
    load = 0,
    store = 1,
};

// A tile is two-dimensional: `tile_rows` runs of `tile_length` elements along
// dimension 0 (the contiguous one); the rows step along `row_dimension`
// (1..3). The instruction's coordinates c0..c3 place it: element x of row y is
// at
//     base + (c0 + x + c1 s1 + c2 s2 + c3 s3 + y s_row) x element_bytes
// (s_row the row dimension's stride). The coordinates of the other two
// dimensions only choose which batch / head the tile is cut from. An element
// whose coordinate in any dimension is outside the tensor is zero-filled on
// a load and skipped on a store.
//
// In shared memory the tile is laid out in the descriptor's layout
// (src/layout.hpp), the same names and rules as the profile's mma.layout. The
// descriptor says nothing about who reads the tile: a kernel that feeds the
// tensor core picks the profile's layout, any other kernel whatever layout
// it wants.
//   dense                row after row: tile row y is stored row y.
//   swizzle64            the tile's rows are the stored rows, whole and
//                        packed: tile row y at buffer + y x (tile length x
//                        element bytes), shuffled inside its 64-byte lines
//                        by its address (src/layout.hpp), so a tile of a
//                        few rows landed inside a taller one is exactly
//                        those rows of it.
//   swizzle64 transposed the tile's positions along dimension 0 are the
//                        stored rows (tile_length of them) and the tile's row
//                        index runs along the stored row (tile_rows x element
//                        bytes each).
// (dense transposed is refused.)
struct TmaDescriptor {
    uint64_t base = 0;            // global byte address of element (0, 0, 0, 0)
    uint8_t element_bytes = 4;    // 1, 2 or 4 (8 and 16 are reserved)
    std::array<uint32_t, 4> dimensions{1, 1, 1, 1}; // d0..d3 in elements; an unused one is 1
    std::array<uint32_t, 3> strides{0, 0, 0};       // s1..s3: elements from one index of dimension 1..3 to the next
    uint16_t tile_length = 1;     // elements along dimension 0
    uint16_t tile_rows = 1;
    uint8_t row_dimension = 1;    // which of dimensions 1..3 the rows step along
    TileLayout layout = TileLayout::dense;
    bool transposed = false;      // swizzle64 only: dimension 0 positions become the stored rows
};

// The 64-byte encoding, little-endian. Bytes 44..63 are reserved and must
// be zero.
//     0       element bytes (1, 2, 4; 8 and 16 reserved)
//     1       row dimension (1..3)
//     2       layout (0 dense, 1 swizzle64)
//     3       transposed (0 no, 1 yes; 0 for a dense tile)
//     4..5    tile length (elements)
//     6..7    tile rows
//     8..15   base (global byte address)
//     16..31  dimensions d0..d3 (elements, 4 bytes each)
//     32..43  strides s1..s3 (elements, 4 bytes each)
namespace tma_encoding {

inline constexpr std::size_t element_bytes = 0;
inline constexpr std::size_t row_dimension = 1;
inline constexpr std::size_t layout = 2;
inline constexpr std::size_t transposed = 3;
inline constexpr std::size_t tile_length = 4;
inline constexpr std::size_t tile_rows = 6;
inline constexpr std::size_t base = 8;
inline constexpr std::size_t dimensions = 16;
inline constexpr std::size_t strides = 32;
inline constexpr std::size_t used_end = 44;

} // namespace tma_encoding

// The two address operands' low bits are not wired: the buffer is 64-byte
// aligned (a swizzle64 tile needs it; NVIDIA asks 128 of a tensor copy's
// destination) and the barrier 8-byte aligned. The machine ignores the low
// 6 and 3 bits; the sanitizer PANICs if they are not zero. A swizzle64
// tile's buffer is also a multiple of its row spacing when that is above 64
// (src/layout.hpp): the machine shuffles by address whatever the buffer, and
// the sanitizer PANICs on a buffer that is not.
inline constexpr uint32_t tma_buffer_alignment = 64;
inline constexpr uint32_t tma_barrier_alignment = 8;

// The descriptor's address is an instruction constant in 64-byte units (the
// 14-bit immediate beside three register fields, src/isa.hpp): 1 MiB of
// reach.

// A swizzle64 tile's stored rows: how many, and how many elements each holds
// -- the tile's rows and length, or (transposed) the other way round. The
// stored row's bytes are the row spacing (src/layout.hpp), the spacing a
// tensor-core instruction names to read the tile. Limits: 2- or 4-byte
// elements; a stored row of a power of two bytes, 16 or more; any number of
// rows, 1 included (the shuffle never moves a byte out of its row, and the
// row number is the address's, so the tile may be any rows of a taller one).
inline uint32_t tma_swizzle64_stored_rows(const TmaDescriptor& descriptor) {
    return descriptor.transposed ? descriptor.tile_length : descriptor.tile_rows;
}

inline uint32_t tma_swizzle64_row_bytes(const TmaDescriptor& descriptor) {
    return (descriptor.transposed ? descriptor.tile_rows : descriptor.tile_length) * descriptor.element_bytes;
}

// Elements from one index of `dimension` to the next (dimension 0: 1).
inline uint64_t tma_stride(const TmaDescriptor& descriptor, uint32_t dimension) {
    return dimension == 0 ? 1 : descriptor.strides.at(dimension - 1);
}

inline void validate_tma_descriptor(const TmaDescriptor& descriptor) {
    const auto bytes = descriptor.element_bytes;
    if (bytes == 8 || bytes == 16) {
        throw SimError("tma element size " + std::to_string(bytes) +
            " bytes is reserved (room for packed number formats, rfc17 section 7) and not supported yet");
    }
    if (bytes != 1 && bytes != 2 && bytes != 4) throw SimError("tma element size must be 1, 2 or 4 bytes");
    for (uint32_t dimension = 0; dimension < 4; ++dimension) {
        if (descriptor.dimensions[dimension] == 0) {
            throw SimError("tma dimension d" + std::to_string(dimension) + " is zero (an unused dimension is 1)");
        }
        if (dimension > 0 && descriptor.dimensions[dimension] > 1 && descriptor.strides[dimension - 1] == 0) {
            throw SimError("tma stride s" + std::to_string(dimension) + " is zero for a dimension of " +
                std::to_string(descriptor.dimensions[dimension]) + " elements");
        }
    }
    if (descriptor.tile_length == 0 || descriptor.tile_rows == 0) {
        throw SimError("tma tile length and rows must be nonzero");
    }
    if (descriptor.row_dimension < 1 || descriptor.row_dimension > 3) {
        throw SimError("tma row dimension must be 1, 2 or 3");
    }
    if (descriptor.layout > TileLayout::swizzle64) throw SimError("tma layout is invalid");
    if (descriptor.layout == TileLayout::dense && descriptor.transposed) {
        throw SimError("tma dense tiles cannot be transposed (only swizzle64 tiles can)");
    }
    if (descriptor.base % bytes != 0) throw SimError("tma global base is not element-aligned");
    if (descriptor.layout == TileLayout::swizzle64) {
        if (bytes != 2 && bytes != 4) throw SimError("tma swizzle64 tiles take 2- or 4-byte elements");
        const uint32_t row_bytes = tma_swizzle64_row_bytes(descriptor);
        if (!valid_swizzle64_spacing(row_bytes)) {
            throw SimError("tma swizzle64 stored row must be a power of two bytes, 16 or more (it is " +
                std::to_string(row_bytes) + ")");
        }
    }
}

// Where element x of tile row y lands in (or comes from) shared memory for a
// tile at `buffer`, in the descriptor's layout (src/layout.hpp: the functions
// the tensor core reads its operands with). dense: stored row y, position x.
// swizzle64: stored row y, position x; transposed: stored row x, position y;
// shuffled by the unshuffled address's own row number, so the buffer takes
// part. 64-bit: an address past 4 GiB is returned unshuffled for the
// caller's bounds check.
inline uint64_t tma_tile_address(const TmaDescriptor& descriptor, uint32_t buffer, uint32_t x, uint32_t y) {
    const uint64_t bytes = descriptor.element_bytes;
    if (descriptor.layout == TileLayout::dense) {
        return buffer + (uint64_t{y} * descriptor.tile_length + x) * bytes;
    }
    const uint32_t spacing = tma_swizzle64_row_bytes(descriptor);
    const uint64_t natural = buffer + (descriptor.transposed ? uint64_t{x} * spacing + y * bytes
                                                             : uint64_t{y} * spacing + x * bytes);
    if (natural > UINT32_MAX) return natural;
    return swizzle64_shuffle(static_cast<uint32_t>(natural), spacing);
}

// The tile's bytes (also what its completion subtracts from the barrier).
inline uint64_t tma_logical_bytes(const TmaDescriptor& descriptor) {
    validate_tma_descriptor(descriptor);
    return uint64_t{descriptor.tile_length} * descriptor.tile_rows * descriptor.element_bytes;
}

// The shmem byte span a command's tile occupies: [buffer, buffer + logical
// bytes). A dense tile fills it row after row; a swizzle64 tile is its rows
// packed, each shuffled inside its own bytes, so both cover exactly this
// span. Load OOB elements are
// zero-filled inside it; store OOB elements skip their (in-span) source
// bytes. The machine's runtime "tile fits in shmem" check is the same
// arithmetic.
inline std::pair<uint64_t, uint64_t> tma_shmem_span(const TmaDescriptor& descriptor, uint32_t buffer) {
    const auto bytes = tma_logical_bytes(descriptor);
    return {buffer, static_cast<uint64_t>(buffer) + bytes};
}

inline std::array<uint8_t, tma_descriptor_bytes> encode_tma_descriptor(
    const TmaDescriptor& descriptor) {
    validate_tma_descriptor(descriptor);
    std::array<uint8_t, tma_descriptor_bytes> bytes{};
    const auto put = [&](std::size_t offset, uint64_t value, unsigned size) {
        for (unsigned byte = 0; byte < size; ++byte) bytes[offset + byte] = static_cast<uint8_t>(value >> (byte * 8));
    };
    bytes[tma_encoding::element_bytes] = descriptor.element_bytes;
    bytes[tma_encoding::row_dimension] = descriptor.row_dimension;
    bytes[tma_encoding::layout] = static_cast<uint8_t>(descriptor.layout);
    bytes[tma_encoding::transposed] = descriptor.transposed ? 1 : 0;
    put(tma_encoding::tile_length, descriptor.tile_length, 2);
    put(tma_encoding::tile_rows, descriptor.tile_rows, 2);
    put(tma_encoding::base, descriptor.base, 8);
    for (std::size_t i = 0; i < 4; ++i) put(tma_encoding::dimensions + 4 * i, descriptor.dimensions[i], 4);
    for (std::size_t i = 0; i < 3; ++i) put(tma_encoding::strides + 4 * i, descriptor.strides[i], 4);
    return bytes;
}

inline TmaDescriptor decode_tma_descriptor(const uint8_t* bytes, std::size_t size) {
    if (bytes == nullptr || size != tma_descriptor_bytes) {
        throw SimError("tma descriptor must contain exactly 64 bytes");
    }
    const auto get = [&](std::size_t offset, unsigned width) {
        uint64_t value = 0;
        for (unsigned byte = 0; byte < width; ++byte) value |= static_cast<uint64_t>(bytes[offset + byte]) << (byte * 8);
        return value;
    };
    for (std::size_t offset = tma_encoding::used_end; offset < tma_descriptor_bytes; ++offset) {
        if (bytes[offset] != 0) throw SimError("tma descriptor reserved bytes 44..63 are nonzero");
    }
    if (bytes[tma_encoding::transposed] > 1) throw SimError("tma descriptor transposed byte must be 0 or 1");
    if (bytes[tma_encoding::layout] > static_cast<uint8_t>(TileLayout::swizzle64)) {
        throw SimError("tma descriptor layout must be 0 (dense) or 1 (swizzle64)");
    }
    TmaDescriptor descriptor;
    descriptor.element_bytes = bytes[tma_encoding::element_bytes];
    descriptor.row_dimension = bytes[tma_encoding::row_dimension];
    descriptor.layout = static_cast<TileLayout>(bytes[tma_encoding::layout]);
    descriptor.transposed = bytes[tma_encoding::transposed] != 0;
    descriptor.tile_length = static_cast<uint16_t>(get(tma_encoding::tile_length, 2));
    descriptor.tile_rows = static_cast<uint16_t>(get(tma_encoding::tile_rows, 2));
    descriptor.base = get(tma_encoding::base, 8);
    for (std::size_t i = 0; i < 4; ++i) {
        descriptor.dimensions[i] = static_cast<uint32_t>(get(tma_encoding::dimensions + 4 * i, 4));
    }
    for (std::size_t i = 0; i < 3; ++i) {
        descriptor.strides[i] = static_cast<uint32_t>(get(tma_encoding::strides + 4 * i, 4));
    }
    validate_tma_descriptor(descriptor);
    return descriptor;
}

} // namespace nanosm

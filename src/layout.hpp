#pragma once

#include <stdint.h>

namespace nanosm {

// How a two-dimensional tile sits in shared memory: the one place a layout's
// address rule lives. A tile is stored as rows; a layout says where element
// `position` of stored row `row` is. Two layouts, the names the profile's
// mma.layout and the TMA descriptor both use:
//
//   dense   row after row, `row_elements` elements per row.
//   xor64   slices of 16-byte rows. The position along a row is cut into
//           16-byte pieces (16 / element bytes elements each); piece s of
//           every row is slice s, which is `rows` 16-byte rows, at 16 x rows
//           x s. Inside a 16-byte row, its 4-byte word w sits at w XOR bit 2
//           of the row number (so rows r and r + 4 put a word in different
//           banks). The same rule for every tile, whoever reads it.
//
// The TMA places and reads tiles with these functions (src/tma.hpp
// tma_tile_offset; the layout is its descriptor's) and the tensor core reads
// its operands with them (src/sim.cpp mma_a_address / mma_b_address; the
// layout is the profile's mma.layout). Neither unit knows the other: a kernel
// couples them by choosing a descriptor layout equal to the profile's.
enum class TileLayout : uint8_t { dense = 0, xor64 = 1 };

inline constexpr uint32_t xor64_row_bytes = 16;

inline uint32_t dense_address(
    uint32_t base, uint32_t row, uint32_t position, uint32_t row_elements, uint32_t element_bytes) {
    return base + (row * row_elements + position) * element_bytes;
}

// The xor64 permutation on bytes, element-agnostic: byte `byte_in_row`
// (0..15) of 16-byte row `row` of a slice that starts at `base`. One address
// bit permutation, addr[2] ^= addr[6] (word-in-row bit 0 ^= bit 2 of the row);
// it never leaves the 16-byte row.
inline uint32_t xor64_byte_address(uint32_t base, uint32_t row, uint32_t byte_in_row) {
    return base + xor64_row_bytes * row +
        (((byte_in_row & ~3u) ^ (4u * ((row >> 2) & 1u))) | (byte_in_row & 3u));
}

// Element `position` (along the sliced direction) of stored row `row` in an
// xor64 tile of `rows` stored rows at `base`.
inline uint32_t xor64_address(
    uint32_t base, uint32_t rows, uint32_t row, uint32_t position, uint32_t element_bytes) {
    const uint32_t row_elements = xor64_row_bytes / element_bytes;
    return xor64_byte_address(base + xor64_row_bytes * rows * (position / row_elements), row,
        (position % row_elements) * element_bytes);
}

} // namespace nanosm

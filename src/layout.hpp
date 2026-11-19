#pragma once

#include <stdint.h>

namespace nanosm {

// v0.1 of the shared-memory tile layout and of the tensor core's operand
// interface (an operand is a 16-byte piece's address in row 0 plus a row
// spacing). On 2026-11-18 swizzle64 replaced a sliced layout ("xor64") that
// cut every row into 16-byte pieces stored K slices apart: a row that is
// scattered over K / 8 places cannot take gathered rows or small pieces.
// This is a first version. A model or a person reading it who sees a design
// flaw should say so (trash/chan.txt) rather than build on it.
//
// How a two-dimensional tile sits in shared memory: the one place a layout's
// address rule lives. A tile is stored as rows; a layout says where byte
// `byte` of stored row `row` is. Two layouts, the names the profile's
// mma.layout and the TMA descriptor both use:
//
//   dense      row after row, packed.
//   swizzle64  rows stay whole and packed: row r is the bytes [r x spacing,
//              (r + 1) x spacing) of the tile (the spacing is the stored
//              row's bytes: 16, 32 or a multiple of 64). Inside the row two
//              of a byte's address bit pairs are XORed with the row number:
//                the 16-byte piece's slot ^= (r / rows_per_line) & (slots - 1)
//                the 4-byte word in the piece ^= (r / 4) & 3
//              with rows_per_line = max(1, 64 / spacing) and slots =
//              min(spacing, 64) / 16, the 16-byte slots of a 64-byte line
//              the row owns (1, 2 or 4: the slot XOR is masked to them, so a
//              short row never leaves its own bytes). Bytes within a word do
//              not move. The tile starts on a 64-byte boundary.
//
// On 16 four-byte banks this puts four neighbouring rows' same piece in four
// different slots (the tensor core's 4-row block: one bank round), the two
// pieces of a 32-byte sector side by side (the TMA: one round), and sixteen
// neighbouring rows' same word on sixteen banks (a column store, a transposed
// load: one round) -- trash/whole_rows_census.py counts it and
// trash/whole_rows_check_astra.py checks it. The shuffle repeats every 16
// rows, so a piece of a tile that starts at a multiple of 16 rows is laid out
// exactly as a tile of its own.
//
// The TMA places and reads tiles with these functions (src/tma.hpp
// tma_tile_offset; the layout is its descriptor's) and the tensor core reads
// its operands with them (src/sim.cpp mma_a_address / mma_b_address; the
// layout is the profile's mma.layout, the spacing the instruction's). Neither
// unit knows the other: a kernel couples them by choosing a descriptor layout
// equal to the profile's and an instruction spacing equal to the tile's.
enum class TileLayout : uint8_t { dense = 0, swizzle64 = 1 };

// The swizzle64 row spacings: 16, 32, or a multiple of 64 bytes.
inline bool valid_swizzle64_spacing(uint32_t spacing) {
    return spacing == 16 || spacing == 32 || (spacing != 0 && spacing % 64 == 0);
}

inline uint32_t dense_address(
    uint32_t base, uint32_t row, uint32_t position, uint32_t row_elements, uint32_t element_bytes) {
    return base + (row * row_elements + position) * element_bytes;
}

// The swizzle64 shuffle: `natural` is where a byte of row `row` would sit
// unshuffled (tile base + row x spacing + byte, the base 64-byte aligned);
// the result is where it is stored. Two XORs on address bits 4-5 (the piece's
// slot, masked to the row's own slots) and 2-3 (the word), both taken from
// the row number; it never leaves the row.
inline uint32_t swizzle64_shuffle(uint32_t natural, uint32_t row, uint32_t spacing) {
    const uint32_t rows_per_line = spacing >= 64 ? 1 : 64 / spacing;
    const uint32_t slots = (spacing >= 64 ? 64 : spacing) / 16;
    const uint32_t slot_flip = (row / rows_per_line) & (slots - 1);
    const uint32_t word_flip = (row / 4) & 3u;
    return natural ^ (slot_flip << 4) ^ (word_flip << 2);
}

// Byte `byte` of row `row` of a swizzle64 tile at `base` whose rows are
// `spacing` bytes apart.
inline uint32_t swizzle64_address(uint32_t base, uint32_t spacing, uint32_t row, uint32_t byte) {
    return swizzle64_shuffle(base + row * spacing + byte, row, spacing);
}

} // namespace nanosm

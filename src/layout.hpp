#pragma once

#include <stdint.h>

namespace nanosm {

// v0.1 of the shared-memory tile layout and of the tensor core's operand
// interface (an operand is a 16-byte piece's address in row 0 plus a row
// spacing). On 2026-11-18 swizzle64 replaced a sliced layout ("xor64") that
// cut every row into 16-byte pieces stored K slices apart: a row that is
// scattered over K / 8 places cannot take gathered rows or small pieces. The
// same day the shuffle's row number became the byte's own address divided by
// the spacing instead of a count from the tile's row 0
// (trash/address-keyed-2026-11-18.md). This is a first version. A model or a
// person reading it who sees a design flaw should say so (trash/chan.txt)
// rather than build on it.
//
// How a two-dimensional tile sits in shared memory: the one place a layout's
// address rule lives. A tile is stored as rows; a layout says where byte
// `byte` of stored row `row` is. Two layouts, the names the profile's
// mma.layout and the TMA descriptor both use:
//
//   dense      row after row, packed.
//   swizzle64  rows stay whole and packed: row r of a tile at `base` is the
//              bytes [base + r x spacing, base + (r + 1) x spacing) (the
//              spacing is the stored row's bytes, a power of two: 16, 32, 64,
//              128, ...). A byte whose unshuffled address is a = base + r x
//              spacing + byte is stored at a with two of its bit pairs XORed
//              with the absolute row number R = a / spacing (a shift):
//                the 16-byte piece's slot ^= (R / rows_per_line) & (slots - 1)
//                the 4-byte word in the piece ^= (R / 4) & 3
//              with rows_per_line = max(1, 64 / spacing) and slots =
//              min(spacing, 64) / 16, the 16-byte slots of a 64-byte line
//              one row owns (1, 2 or 4: the slot XOR is masked to them).
//              Bytes within a word do not move, and no byte leaves its
//              64-byte line. A tile's base is a multiple of its spacing and
//              of 64, so every row keeps exactly its own bytes.
//
// Because the row number comes from the address, the stored place of a byte
// depends only on its unshuffled address and the spacing: every reader and
// writer agrees wherever a piece of a tile starts. A one-row TMA load landed
// at row 37 of a taller tile puts its bytes exactly where a load of the whole
// tile does, and a tensor-core operand may start at any row. A tile whose
// base is a multiple of 16 rows (the shuffle's period) is laid out as the
// rule counted from its own row 0 would lay it out.
//
// On 16 four-byte banks this puts any four neighbouring rows' same piece in
// four different slots (the tensor core's 4-row block: one bank round), the
// two pieces of a 32-byte sector side by side (the TMA: one round), and any
// sixteen neighbouring rows' same word on sixteen banks (a column store, a
// transposed load: one round) -- whatever row they start at:
// trash/address_keyed_census.py counts it. (One thing does depend on where a
// tile sits: on a 4x4x4 core, a step that reloads A reads half-pieces of A
// and B together, and whether they share a round depends on the two tiles'
// row numbers mod 16; the canonical 4x4x8 core reads whole pieces and does
// not care.)
//
// The TMA places and reads tiles with these functions (src/tma.hpp
// tma_tile_address, which a run's elements step through: src/sim.hpp TmaRun;
// the layout is its descriptor's) and the tensor core reads
// its operands with them (src/sim.cpp mma_a_address / mma_b_address; the
// layout is the profile's mma.layout, the spacing the instruction's). Neither
// unit knows the other: a kernel couples them by choosing a descriptor layout
// equal to the profile's and an instruction spacing equal to the tile's.
enum class TileLayout : uint8_t { dense = 0, swizzle64 = 1 };

// The swizzle64 row spacings: a power of two, 16 bytes or more.
inline bool valid_swizzle64_spacing(uint32_t spacing) {
    return spacing >= 16 && (spacing & (spacing - 1)) == 0;
}

// What a swizzle64 tile's base (and a tensor-core operand's row-0 line) must
// be a multiple of: the spacing, and 64.
inline uint32_t swizzle64_base_alignment(uint32_t spacing) {
    return spacing > 64 ? spacing : 64;
}

inline uint32_t dense_address(
    uint32_t base, uint32_t row, uint32_t position, uint32_t row_elements, uint32_t element_bytes) {
    return base + (row * row_elements + position) * element_bytes;
}

// The swizzle64 shuffle: `natural` is where a byte would sit unshuffled; the
// result is where it is stored. The row number is natural / spacing. Two
// XORs on address bits 4-5 (the piece's slot, masked to a row's own slots)
// and 2-3 (the word); the byte never leaves its 64-byte line.
inline uint32_t swizzle64_shuffle(uint32_t natural, uint32_t spacing) {
    const uint32_t row = natural / spacing;
    const uint32_t rows_per_line = spacing >= 64 ? 1 : 64 / spacing;
    const uint32_t slots = (spacing >= 64 ? 64 : spacing) / 16;
    const uint32_t slot_flip = (row / rows_per_line) & (slots - 1);
    const uint32_t word_flip = (row / 4) & 3u;
    return natural ^ (slot_flip << 4) ^ (word_flip << 2);
}

// Byte `byte` of row `row` of a swizzle64 tile at `base` whose rows are
// `spacing` bytes apart.
inline uint32_t swizzle64_address(uint32_t base, uint32_t spacing, uint32_t row, uint32_t byte) {
    return swizzle64_shuffle(base + row * spacing + byte, spacing);
}

// ---------------------------------------------------------------------------
// Tensor-core memory (tmem), v0.1 of 2026-11-19 (trash/acc-pointer-2026-11-19.md).
// An accumulator is no declared object: an instruction names a POINTER (a
// word address, held in a uniform register) and the words per row S of the
// region it treats as rows (an instruction constant; the matmul's result
// rows are its shape's n words). Word (row r, column c) of the region at
// pointer P has the LOGICAL address L = P + r x S + c. Where L is stored
// depends on L and S only (keyed by the address, as swizzle64 is): with the
// absolute row R = L / S and column C = L % S,
//
//   physical = ((R / 4) x (S / 4) + C / 4) x 16 + 4 x (R % 4) + ((C % 4) XOR ((R / 4) % 4))
//
// i.e. the region is cut into 4x4 macro tiles of 16 words, stored row-major
// by macro tile (S / 4 of them per four rows), and inside a macro tile word
// (R % 4, C % 4) sits at 4 x (R % 4) + (C % 4 XOR the band number R / 4 mod
// 4). Word w is in fine bank w % 16 (tmem.fine_banks is 16): one word per
// fine bank for
//   - any 4x4 macro tile (the tensor core's result block, and its C read):
//     R % 4 picks the bank's high bits, the XOR permutes the low ones;
//   - any 16 consecutive rows of one column (an acc.ld / acc.st beat, one
//     row per lane), wherever they start: each R % 4 occurs four times, in
//     four consecutive bands, whose XOR values differ.
// With S a power of two the rule is wiring: L's bits are [band | R % 4 |
// C / 4 | C % 4]; physical swaps the middle two fields to [band | C / 4 | R %
// 4] and XORs C % 4 with the band's low two bits. No word leaves its band of
// four rows (4 x S words). A region at a pointer that is a multiple of 16 x S
// is laid out as the per-accumulator rule of a5d0652 laid out an accumulator
// at that base.
inline bool valid_tmem_row_words(uint32_t row_words) {
    return row_words >= 4 && (row_words & (row_words - 1)) == 0;
}

inline uint64_t tmem_physical_word(uint64_t logical, uint32_t row_words) {
    const uint64_t row = logical / row_words;
    const uint64_t column = logical % row_words;
    return ((row / 4) * (row_words / 4) + column / 4) * 16 + 4 * (row % 4) + ((column % 4) ^ ((row / 4) % 4));
}

// The pointer rule (the sanitizer's; the machine addresses any pointer by the
// rule above): a pointer's row (pointer / S) is a multiple of 4, so a matmul's
// result blocks are whole macro tiles and a transfer's sixteen rows are four
// whole bands; and the columns an instruction touches stay inside the
// pointer's row (pointer % S + columns <= S: a matmul's pointer is at column
// 0, its rows being S = n words).
inline bool tmem_pointer_aligned(uint32_t pointer, uint32_t row_words, uint32_t columns) {
    return (pointer / row_words) % 4 == 0 && pointer % row_words + columns <= row_words;
}

} // namespace nanosm

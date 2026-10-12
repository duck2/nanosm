#pragma once

#include <stdint.h>

namespace nanosm {

// One mbar: two shmem words, read and written as one coherent 8-byte RMW.
// tx_bytes is the SIGNED transaction count (trash/rfc12.md section 8, as in
// PTX): expect_tx adds, complete_tx subtracts, in either order, so a
// complete_tx that runs ahead of its expect_tx leaves it negative. A phase
// completes when the pending arrivals (remaining) AND tx_bytes are both zero.
// Its range is +/-(2^20 - 1) (mbar_encoding::tx_limit); leaving it PANICs.
struct MbarState {
    bool phase = false;
    bool valid = false;
    uint16_t expected = 0;
    uint16_t remaining = 0;
    int32_t tx_bytes = 0;
};

namespace mbar_encoding {

inline constexpr unsigned phase_shift = 0;
inline constexpr unsigned valid_shift = 1;
inline constexpr unsigned expected_shift = 2;
inline constexpr unsigned remaining_shift = 17;
inline constexpr unsigned tx_bytes_shift = 32;   // bits 32-63: tx_bytes, 32-bit two's complement
inline constexpr uint64_t count_mask = 0x7fff;
inline constexpr uint64_t tx_bytes_mask = 0xffffffff;
inline constexpr uint16_t max_count = 0x7fff;
inline constexpr int32_t tx_limit = (1 << 20) - 1; // PTX's tx-count range: -tx_limit..tx_limit

} // namespace mbar_encoding

inline uint64_t encode_mbar_state(const MbarState& state) {
    return
        (static_cast<uint64_t>(state.phase) << mbar_encoding::phase_shift) |
        (static_cast<uint64_t>(state.valid) << mbar_encoding::valid_shift) |
        (static_cast<uint64_t>(state.expected) << mbar_encoding::expected_shift) |
        (static_cast<uint64_t>(state.remaining) << mbar_encoding::remaining_shift) |
        (static_cast<uint64_t>(static_cast<uint32_t>(state.tx_bytes)) << mbar_encoding::tx_bytes_shift);
}

inline MbarState decode_mbar_state(uint64_t word) {
    MbarState state;
    state.phase = ((word >> mbar_encoding::phase_shift) & 1) != 0;
    state.valid = ((word >> mbar_encoding::valid_shift) & 1) != 0;
    state.expected = static_cast<uint16_t>(
        (word >> mbar_encoding::expected_shift) & mbar_encoding::count_mask);
    state.remaining = static_cast<uint16_t>(
        (word >> mbar_encoding::remaining_shift) & mbar_encoding::count_mask);
    state.tx_bytes = static_cast<int32_t>(static_cast<uint32_t>(
        (word >> mbar_encoding::tx_bytes_shift) & mbar_encoding::tx_bytes_mask));
    return state;
}

} // namespace nanosm

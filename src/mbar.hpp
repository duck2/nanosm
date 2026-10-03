#pragma once

#include <stdint.h>

namespace nanosm {

struct MbarState {
    bool phase = false;
    bool valid = false;
    uint16_t expected = 0;
    uint16_t remaining = 0;
    uint32_t tx_bytes = 0;
};

namespace mbar_encoding {

inline constexpr unsigned phase_shift = 0;
inline constexpr unsigned valid_shift = 1;
inline constexpr unsigned expected_shift = 2;
inline constexpr unsigned remaining_shift = 17;
inline constexpr unsigned tx_bytes_shift = 32;
inline constexpr uint64_t count_mask = 0x7fff;
inline constexpr uint64_t tx_bytes_mask = 0xffffffff;
inline constexpr uint16_t max_count = 0x7fff;

} // namespace mbar_encoding

inline uint64_t encode_mbar_state(const MbarState& state) {
    return
        (static_cast<uint64_t>(state.phase) << mbar_encoding::phase_shift) |
        (static_cast<uint64_t>(state.valid) << mbar_encoding::valid_shift) |
        (static_cast<uint64_t>(state.expected) << mbar_encoding::expected_shift) |
        (static_cast<uint64_t>(state.remaining) << mbar_encoding::remaining_shift) |
        (static_cast<uint64_t>(state.tx_bytes) << mbar_encoding::tx_bytes_shift);
}

inline MbarState decode_mbar_state(uint64_t word) {
    MbarState state;
    state.phase = ((word >> mbar_encoding::phase_shift) & 1) != 0;
    state.valid = ((word >> mbar_encoding::valid_shift) & 1) != 0;
    state.expected = static_cast<uint16_t>(
        (word >> mbar_encoding::expected_shift) & mbar_encoding::count_mask);
    state.remaining = static_cast<uint16_t>(
        (word >> mbar_encoding::remaining_shift) & mbar_encoding::count_mask);
    state.tx_bytes = static_cast<uint32_t>(
        (word >> mbar_encoding::tx_bytes_shift) & mbar_encoding::tx_bytes_mask);
    return state;
}

} // namespace nanosm

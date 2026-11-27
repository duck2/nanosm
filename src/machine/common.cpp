// Helpers shared by the units: float bit conversions, round-robin grants, the lane mask and
// panics.
// (Machine is one class, src/machine/machine.hpp; the map of its files is at the top of
// src/machine/machine.cpp.)

#include "machine/machine.hpp"

#include <bit>

namespace nanosm {

uint32_t float_bits(float value) {
    return std::bit_cast<uint32_t>(value);
}

float bits_float(uint32_t value) {
    return std::bit_cast<float>(value);
}

uint32_t half_to_float_bits(uint16_t half) {
    const uint32_t sign = static_cast<uint32_t>(half & 0x8000) << 16;
    uint32_t exponent = (half >> 10) & 0x1f;
    uint32_t fraction = half & 0x3ff;
    if (exponent == 0) {
        if (fraction == 0) return sign;
        int shift = 0;
        while ((fraction & 0x400) == 0) { fraction <<= 1; ++shift; }
        fraction &= 0x3ff;
        return sign | static_cast<uint32_t>(127 - 14 - shift) << 23 | fraction << 13;
    }
    if (exponent == 31) return sign | 0x7f800000u | fraction << 13;
    return sign | (exponent + 112) << 23 | fraction << 13;
}

uint16_t float_to_half_bits(uint32_t bits) {
    const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000);
    const uint32_t exponent = (bits >> 23) & 0xff;
    const uint32_t fraction = bits & 0x7fffff;
    if (exponent == 0xff) {
        if (fraction == 0) return sign | 0x7c00; // infinity
        // NaN: keep the truncated payload so half NaNs round-trip; force the
        // quiet bit when truncation would make the payload vanish.
        const auto payload = static_cast<uint16_t>(fraction >> 13);
        return static_cast<uint16_t>(sign | 0x7c00 | (payload != 0 ? payload : 0x200));
    }
    const int32_t e = static_cast<int32_t>(exponent) - 127 + 15; // half-biased exponent
    if (e >= 31) return sign | 0x7c00; // overflow to infinity
    const uint32_t significand = fraction | (exponent != 0 ? 0x800000u : 0u);
    // Round to nearest even while dropping 13 bits (normal) or more
    // (subnormal destination). Everything below the round bit is sticky.
    const uint32_t shift = e <= 0 ? static_cast<uint32_t>(14 - e) : 13u;
    if (shift >= 25) return sign; // rounds to signed zero
    const uint32_t keep = significand >> shift;
    const uint32_t round = (significand >> (shift - 1)) & 1u;
    const uint32_t sticky = significand & ((1u << (shift - 1)) - 1u);
    const uint32_t rounded = keep + (round && (sticky || (keep & 1u)));
    // Reassemble arithmetically so a rounding carry ripples into the
    // exponent: subnormal->normal promotion and near-max overflow to
    // infinity both fall out of the addition.
    const uint32_t result = e <= 0 ? rounded
        : (static_cast<uint32_t>(e) << 10) + (rounded - 0x400u);
    return static_cast<uint16_t>(sign | result);
}

RoundRobinGrant round_robin_grant(const std::vector<uint8_t>& ready, uint32_t cursor, uint32_t ports) {
    RoundRobinGrant result;
    result.cursor_after = cursor;
    const auto holders = static_cast<uint32_t>(ready.size());
    for (uint32_t step = 0; step < holders && result.granted.size() < ports; ++step) {
        const auto holder = (cursor + step) % holders;
        if (ready[holder] == 0) continue;
        result.granted.push_back(holder);
        result.cursor_after = (holder + 1) % holders;
    }
    return result;
}

uint64_t Machine::full_mask() const {
    return config_.lanes >= 64 ? ~0ull : (1ull << config_.lanes) - 1;
}

[[noreturn]] void Machine::panic(
    const Warp& warp, const Instruction& inst, const std::string& reason) const {
    throw_issue_panic(site(warp, inst), reason);
}

[[noreturn]] void Machine::panic(const QueuedEntry& entry, const std::string& reason) const {
    throw_issue_panic(site(entry), reason);
}

[[noreturn]] void Machine::panic(const DispatchEntry& op, const std::string& reason) const {
    throw_issue_panic(site(op), reason);
}

[[noreturn]] void Machine::panic(const LsuCommand& command, const std::string& reason) const {
    throw_issue_panic(site(command), reason);
}

} // namespace nanosm

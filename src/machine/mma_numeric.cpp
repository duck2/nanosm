#include "machine/mma_numeric.hpp"

#include <algorithm>
#include <bit>
#include <limits>
#include <stdexcept>

// Arithmetic adapted from sf-tensor/tcgen05, commit
// d96b82180ac9041c09295f3a6a3c2c4e592ddce6, model.py's raw-window FP16 path.
// Copyright (c) 2026 San Francisco Tensor Company; MIT license, reproduced
// in third_party/tcgen05-LICENSE. No host floating-point arithmetic is used.
//
// The model, mma.numeric_model "fp16_raw_window25_rz_v1" (in the profile's
// fingerprint). One step of the core fuses its core_k fp16 products with the
// fp32 accumulator word C: the largest raw exponent among the nonzero terms
// (a product's is its operands' exponent sum) chooses the window; every term
// is aligned to a quantum of 2^(that exponent - 25), its magnitude truncated;
// the signed integer units are summed, and the sum is normalized and
// truncated once to fp32. Subnormal operands are kept; an exact or an
// underflowed zero is +0; a generated NaN is 0x7fffffff; an overflow is an
// infinity of its sign. Only fp32 survives between steps (the canonical
// 4x4x8 core fuses eight products and C in a step, not a whole mma.async).
// It takes tcgen05's measured arithmetic rules, not B200's fusion extent,
// and claims no whole-instruction B200 equivalence. There are no numerical
// tuning knobs.
namespace nanosm {
namespace {

constexpr uint32_t canonical_nan = 0x7fffffffu;
constexpr uint32_t infinity = 0x7f800000u;

struct Term {
    uint64_t magnitude;
    int value_exponent;
    int raw_exponent;
    bool negative;
    bool nan;
    bool infinite;
};

Term decode(uint32_t bits, unsigned fraction_bits, unsigned exponent_bits, int bias) {
    const uint32_t fraction = bits & ((1u << fraction_bits) - 1);
    const uint32_t exponent_mask = (1u << exponent_bits) - 1;
    const uint32_t exponent = (bits >> fraction_bits) & exponent_mask;
    const bool special = exponent == exponent_mask;
    const int raw_exponent = static_cast<int>(exponent == 0 ? 1 : exponent) - bias;
    return {
        fraction | (exponent != 0 && !special ? 1u << fraction_bits : 0u),
        raw_exponent - static_cast<int>(fraction_bits), raw_exponent,
        ((bits >> (fraction_bits + exponent_bits)) & 1u) != 0,
        special && fraction != 0, special && fraction == 0,
    };
}

Term product(uint16_t a_bits, uint16_t b_bits) {
    const auto a = decode(a_bits, 10, 5, 15);
    const auto b = decode(b_bits, 10, 5, 15);
    const bool a_zero = !a.nan && !a.infinite && a.magnitude == 0;
    const bool b_zero = !b.nan && !b.infinite && b.magnitude == 0;
    return {
        a.magnitude * b.magnitude, a.value_exponent + b.value_exponent,
        a.raw_exponent + b.raw_exponent, a.negative != b.negative,
        a.nan || b.nan || (a.infinite && b_zero) || (b.infinite && a_zero),
        a.infinite || b.infinite,
    };
}

int64_t align(const Term& term, int quantum) {
    if (term.magnitude == 0) return 0;
    const int shift = term.value_exponent - quantum;
    // The window is chosen from the maximum raw exponent. A product shifts
    // left by at most 5 bits; C by at most 2. Very distant terms vanish.
    const uint64_t units = shift >= 0 ? term.magnitude << shift
        : shift <= -64 ? 0 : term.magnitude >> -shift;
    return term.negative ? -static_cast<int64_t>(units) : static_cast<int64_t>(units);
}

uint32_t pack(int64_t sum, int quantum) {
    if (sum == 0) return 0; // exact cancellation and signed zeros become +0
    const uint32_t sign = sum < 0 ? 0x80000000u : 0;
    const uint64_t magnitude = static_cast<uint64_t>(sum < 0 ? -sum : sum);
    const int top_bit = static_cast<int>(std::bit_width(magnitude)) - 1;
    const int exponent = top_bit + quantum;
    if (exponent > 127) return sign | infinity;
    if (exponent >= -126) {
        const uint64_t significand = top_bit >= 23
            ? magnitude >> (top_bit - 23) : magnitude << (23 - top_bit);
        return sign | (static_cast<uint32_t>(exponent + 127) << 23)
            | (static_cast<uint32_t>(significand) & 0x7fffffu);
    }
    const int shift = quantum + 149;
    const uint64_t fraction = shift >= 0 ? magnitude << shift
        : shift <= -64 ? 0 : magnitude >> -shift;
    // Match the reference: even negative underflow-to-zero becomes +0.
    return fraction == 0 ? 0 : sign | static_cast<uint32_t>(fraction);
}

} // namespace

uint32_t mma_dot_f16(std::span<const uint16_t> a, std::span<const uint16_t> b, uint32_t c) {
    if (a.empty() || a.size() != b.size() || a.size() > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("MMA dot requires equal nonempty spans of at most UINT32_MAX terms");
    const auto accumulator = decode(c, 23, 8, 127);
    if (accumulator.nan) return canonical_nan;
    bool positive_inf = accumulator.infinite && !accumulator.negative;
    bool negative_inf = accumulator.infinite && accumulator.negative;
    int window = std::numeric_limits<int>::min();
    if (!accumulator.infinite && accumulator.magnitude != 0) window = accumulator.raw_exponent;
    for (std::size_t k = 0; k < a.size(); ++k) {
        const auto term = product(a[k], b[k]);
        if (term.nan) return canonical_nan;
        if (term.infinite) {
            positive_inf |= !term.negative;
            negative_inf |= term.negative;
        } else if (term.magnitude != 0) {
            window = std::max(window, term.raw_exponent);
        }
    }
    if (positive_inf && negative_inf) return canonical_nan;
    if (positive_inf || negative_inf) return infinity | (negative_inf ? 0x80000000u : 0u);
    if (window == std::numeric_limits<int>::min()) return 0;
    const int quantum = window - 25;
    int64_t sum = align(accumulator, quantum);
    // Each aligned product is < 2^27 units and C is < 2^26 units. Even
    // UINT32_MAX products fit in int64_t; no bigint or per-dot allocation.
    for (std::size_t k = 0; k < a.size(); ++k) sum += align(product(a[k], b[k]), quantum);
    return pack(sum, quantum);
}

} // namespace nanosm

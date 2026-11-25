// The arithmetic pipeline: ALU, FPU and the special-function unit.
// (Machine is one class, src/machine/machine.hpp; the map of its files is at the top of
// src/machine/machine.cpp.)

#include "machine/machine.hpp"
#include "machine/internal.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <utility>

namespace nanosm {

namespace {

// A NaN single-precision result is 0x7fc00000 (fsub, fmul, fma, fmax, ex2,
// rcp), so the host's NaN-payload propagation does not leak into the
// machine; fadd's is the exception: its NaN is the host's single-precision
// sum as it comes (the payload is the host's choice). cvt.pack's halves keep
// their own rule (float_to_half_bits).
uint32_t canonical_float(float value) {
    if (std::isnan(value)) return 0x7fc00000u;
    return float_bits(value);
}

// ex2 = 2^x computed in double, narrowed once with RNE. std::exp2(double) is
// the single libm call in the spec; tests pin exact result bits so platform
// drift breaks loudly instead of silently changing the machine.
uint32_t sfu_ex2(uint32_t bits) {
    const float x = bits_float(bits);
    if (std::isnan(x)) return 0x7fc00000u;
    return canonical_float(static_cast<float>(std::exp2(static_cast<double>(x))));
}

// rcp = 1/x as one double division, narrowed once with RNE. Hardware IEEE
// division, no libm: deterministic everywhere.
uint32_t sfu_rcp(uint32_t bits) {
    const float x = bits_float(bits);
    if (std::isnan(x)) return 0x7fc00000u;
    return canonical_float(static_cast<float>(1.0 / static_cast<double>(x)));
}

// IEEE maxNum: one NaN yields the other operand; +0 is preferred over -0.
uint32_t float_max_bits(uint32_t a_bits, uint32_t b_bits) {
    const float a = bits_float(a_bits);
    const float b = bits_float(b_bits);
    if (std::isnan(a) && std::isnan(b)) return 0x7fc00000u;
    if (std::isnan(a)) return b_bits;
    if (std::isnan(b)) return a_bits;
    if (a == 0.0f && b == 0.0f) {
        return (std::signbit(a) && std::signbit(b)) ? a_bits : float_bits(0.0f);
    }
    return a > b ? a_bits : b_bits;
}

// The vector floating point -- the FPU's fadd, fsub, fmul, fma, fmax and
// cvt.pack, the special-function unit's ex2 and rcp -- is this
// implementation's: the instruction set does not specify its rounding, its
// denormals or whether fma is fused, so a kernel's check carries a tolerance
// (README). This one rounds to nearest even, fuses fma, and has no
// subnormals. A subnormal word is zero of its sign; every single-precision
// operation of the two units takes its inputs and gives its result through
// this (denormals are zero in, flush to zero out). NaNs and infinities pass.
uint32_t no_subnormal(uint32_t bits) { return (bits & 0x7f800000u) == 0 ? bits & 0x80000000u : bits; }

__extension__ typedef unsigned __int128 Wide; // GCC's: 128 bits hold any exactly aligned sum below

// fma's a x b + c, exact, rounded once to single precision, nearest even --
// the simulator's own integer arithmetic, not the host's fmaf. The product of
// the two 24-bit significands is exact in 48 bits; it and c's significand
// go 64 bits up (guard bits), and the one with the smaller exponent is
// shifted down to the other's: exactly while the distance is at most 64;
// further, what falls off the bottom is kept as one sticky bit, which then
// sits more than 60 bits below where the sum is rounded, so the rounding
// sees the exact sum's. The sum is rounded at its 24th bit, or at 2^-149
// below the normal range (the IEEE result, subnormal included; the caller
// flushes that). NaN in, inf x 0, inf - inf: NaN. An exact zero sum of
// nonzero terms is +0; of two zeros, -0 only when both are.
uint32_t fused_multiply_add(uint32_t a, uint32_t b, uint32_t c) {
    constexpr uint32_t sign_bit = 0x80000000u;
    constexpr uint32_t infinity = 0x7f800000u;
    constexpr uint32_t nan = 0x7fc00000u;
    const auto is_nan = [](uint32_t x) { return (x & 0x7fffffffu) > infinity; };
    const auto is_inf = [](uint32_t x) { return (x & 0x7fffffffu) == infinity; };
    const auto is_zero = [](uint32_t x) { return (x & 0x7fffffffu) == 0; };
    if (is_nan(a) || is_nan(b) || is_nan(c)) return nan;
    const uint32_t product_sign = (a ^ b) & sign_bit;
    const uint32_t c_sign = c & sign_bit;
    if (is_inf(a) || is_inf(b)) {
        if (is_zero(a) || is_zero(b)) return nan;
        if (is_inf(c) && c_sign != product_sign) return nan;
        return product_sign | infinity;
    }
    if (is_inf(c)) return c;
    if (is_zero(a) || is_zero(b)) {
        if (!is_zero(c)) return c;
        return product_sign == c_sign ? product_sign : 0u;
    }
    // A finite word as significand x 2^exponent (a subnormal: no hidden bit).
    const auto significand = [](uint32_t x) -> uint64_t {
        const uint32_t field = (x >> 23) & 0xffu;
        return (x & 0x7fffffu) | (field != 0 ? 0x800000u : 0u);
    };
    const auto exponent = [](uint32_t x) -> int32_t {
        const auto field = static_cast<int32_t>((x >> 23) & 0xffu);
        return (field != 0 ? field : 1) - 150;
    };
    Wide product = static_cast<Wide>(significand(a) * significand(b)) << 64;
    int32_t product_exponent = exponent(a) + exponent(b) - 64;
    Wide addend = static_cast<Wide>(significand(c)) << 64;
    int32_t addend_exponent = is_zero(c) ? product_exponent : exponent(c) - 64;
    const auto shift_down = [](Wide x, int32_t distance) -> Wide {
        if (distance == 0) return x;
        if (distance >= 128) return x != 0 ? 1 : 0;
        const Wide lost = x & ((Wide{1} << distance) - 1);
        return (x >> distance) | (lost != 0 ? 1 : 0);
    };
    int32_t e = product_exponent;
    if (product_exponent >= addend_exponent) {
        addend = shift_down(addend, product_exponent - addend_exponent);
    } else {
        product = shift_down(product, addend_exponent - product_exponent);
        e = addend_exponent;
    }
    Wide magnitude = 0;
    uint32_t sign = product_sign;
    if (product_sign == c_sign) {
        magnitude = product + addend;
    } else if (product >= addend) {
        magnitude = product - addend;
    } else {
        magnitude = addend - product;
        sign = c_sign;
    }
    if (magnitude == 0) return 0u;
    const auto high = static_cast<uint64_t>(magnitude >> 64);
    const int32_t top = high != 0 ? 127 - std::countl_zero(high)
                                  : 63 - std::countl_zero(static_cast<uint64_t>(magnitude));
    // The exponent of the result's last bit: 24 bits from its top, at most
    // down to 2^-149.
    int32_t last = std::max(top + e - 23, -149);
    const int32_t below = last - e;
    Wide kept = 0;
    if (below <= 0) {
        kept = magnitude << -below;
    } else if (below < 128) {
        kept = magnitude >> below;
        const Wide rest = magnitude & ((Wide{1} << below) - 1);
        const Wide half = Wide{1} << (below - 1);
        if (rest > half || (rest == half && (kept & 1) != 0)) ++kept;
    }
    if (kept == 0) return sign;
    if (kept == Wide{1} << 24) {
        kept >>= 1;
        ++last;
    }
    const auto bits = static_cast<uint32_t>(kept);
    if (bits < 0x800000u) return sign | bits; // subnormal (2^-149 units)
    const int32_t biased = last + 150;
    if (biased >= 255) return sign | infinity;
    return sign | static_cast<uint32_t>(biased) << 23 | (bits & 0x7fffffu);
}
template <typename Operation, typename... Words>
uint32_t single_precision(Operation operation, Words... words) {
    return no_subnormal(operation(no_subnormal(words)...));
}

} // namespace

// ===========================================================================
// Arithmetic pipeline
// ===========================================================================

void Machine::execute_arithmetic(const ArithmeticRead& pending) {
    auto& warp = warps_.at(pending.warp);
    const auto& inst = pending.instruction;
    // A panic names the instruction's own pc (the warp's has moved on).
    const auto fail = [&](const std::string& reason) {
        throw_issue_panic(IssueSite{cycle_, pending.warp, pending.pc, inst, &config_.mma_shapes}, reason);
    };
    LaneWords result(config_.lanes, 0);
    // The round holds each source register once; duplicated operand slots
    // broadcast from the same captured latch.
    const auto operand = [&](uint8_t reg, uint32_t lane) {
        for (std::size_t i = 0; i < pending.reads.size(); ++i) {
            if (pending.reads[i] == reg) return pending.operands[i][lane];
        }
        throw SimError("internal arithmetic operand was not in its read round");
    };
    // The second source: a vector register's lane, or -- the U bit -- the
    // uniform register latched with the round (the round's one uniform read),
    // the same word in every lane.
    const auto source_b = [&](uint32_t lane) {
        return inst.src1_uniform ? pending.uniform_operands.at(0) : operand(inst.src1, lane);
    };
    switch (inst.opcode) {
    case Opcode::sread:
        // The per-lane special registers; WARP_ID / N_WARPS are usread's.
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            switch (static_cast<SpecialReg>(inst.imm)) {
            case SpecialReg::lane_id: result[lane] = lane; break;
            case SpecialReg::thread_id: result[lane] = warp.id * config_.lanes + lane; break;
            default: fail("invalid special register for sread (WARP_ID / N_WARPS are read with usread)");
            }
        }
        break;
    case Opcode::mov:
        std::fill(result.begin(), result.end(), static_cast<uint32_t>(inst.imm));
        break;
    case Opcode::ur2r:
        // The uniform operand latched at the sample, into every lane.
        std::fill(result.begin(), result.end(), pending.uniform_operands.at(0));
        break;
    case Opcode::add:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = operand(inst.src0, lane) + source_b(lane);
        }
        break;
    case Opcode::addi:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = operand(inst.src0, lane) + static_cast<uint32_t>(inst.imm);
        }
        break;
    case Opcode::shli:
        if (inst.imm < 0 || inst.imm > 31) fail("shift outside 0..31");
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = operand(inst.src0, lane) << inst.imm;
        }
        break;
    case Opcode::shri:
        if (inst.imm < 0 || inst.imm > 31) fail("shift outside 0..31");
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = operand(inst.src0, lane) >> inst.imm;
        }
        break;
    case Opcode::xori:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = operand(inst.src0, lane) ^ static_cast<uint32_t>(inst.imm);
        }
        break;
    case Opcode::andi:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = operand(inst.src0, lane) & static_cast<uint32_t>(inst.imm);
        }
        break;
    case Opcode::ori:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = operand(inst.src0, lane) | static_cast<uint32_t>(inst.imm);
        }
        break;
    case Opcode::and_:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = operand(inst.src0, lane) & source_b(lane);
        }
        break;
    case Opcode::or_:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = operand(inst.src0, lane) | source_b(lane);
        }
        break;
    case Opcode::xor_:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = operand(inst.src0, lane) ^ source_b(lane);
        }
        break;
    // Register shifts use the low 5 bits of the per-lane amount: the datapath
    // is a physical 5-bit shifter, so the masking IS the hardware, not a
    // silently repaired error.
    case Opcode::shl:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = operand(inst.src0, lane) << (source_b(lane) & 31);
        }
        break;
    case Opcode::shr:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = operand(inst.src0, lane) >> (source_b(lane) & 31);
        }
        break;
    case Opcode::fadd:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = single_precision(
                [](uint32_t a, uint32_t b) { return float_bits(bits_float(a) + bits_float(b)); },
                operand(inst.src0, lane), source_b(lane));
        }
        break;
    case Opcode::fmul:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = single_precision(
                [](uint32_t a, uint32_t b) { return canonical_float(bits_float(a) * bits_float(b)); },
                operand(inst.src0, lane), source_b(lane));
        }
        break;
    case Opcode::fsub:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = single_precision(
                [](uint32_t a, uint32_t b) { return canonical_float(bits_float(a) - bits_float(b)); },
                operand(inst.src0, lane), source_b(lane));
        }
        break;
    case Opcode::fmax:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = single_precision(float_max_bits, operand(inst.src0, lane), source_b(lane));
        }
        break;
    case Opcode::fma:
        // One rounding of the exact a x b + c (fused_multiply_add above).
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = single_precision(fused_multiply_add, operand(inst.src0, lane), source_b(lane),
                operand(inst.src2, lane));
        }
        break;
    case Opcode::cvt_pack:
        // Its inputs are single precision (a subnormal is zero: it would
        // round to a half's zero of its sign anyway); its results are halves,
        // whose subnormals are kept.
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = static_cast<uint32_t>(float_to_half_bits(no_subnormal(operand(inst.src0, lane)))) |
                static_cast<uint32_t>(float_to_half_bits(no_subnormal(source_b(lane)))) << 16;
        }
        break;
    case Opcode::ex2:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = single_precision(sfu_ex2, operand(inst.src0, lane));
        }
        break;
    case Opcode::rcp:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = single_precision(sfu_rcp, operand(inst.src0, lane));
        }
        break;
    case Opcode::setp_eq:
    case Opcode::setp_ne:
    case Opcode::setp_lt:
    case Opcode::setp_le:
    case Opcode::setp_gt:
    case Opcode::setp_ge:
    case Opcode::setpi_eq:
    case Opcode::setpi_ne:
    case Opcode::setpi_lt:
    case Opcode::setpi_le:
    case Opcode::setpi_gt:
    case Opcode::setpi_ge:
    case Opcode::setp_lt_s32:
    case Opcode::setp_le_s32:
    case Opcode::setp_gt_s32:
    case Opcode::setp_ge_s32:
    case Opcode::setpi_lt_s32:
    case Opcode::setpi_le_s32:
    case Opcode::setpi_gt_s32:
    case Opcode::setpi_ge_s32:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            const auto rhs = is_setp_immediate(inst.opcode)
                ? static_cast<uint32_t>(inst.imm)
                : source_b(lane);
            result[lane] = compare_setp(inst.opcode, operand(inst.src0, lane), rhs);
        }
        break;
    default: fail("non-arithmetic instruction in RF read pipeline");
    }
    // Into its unit's black box: in its output register E cycles from now,
    // at the write port it booked. Every lane computes; the issue cycle's
    // active mask AND the instruction's write mask (sampled with the
    // operands) is the write's lane enables (the port, its cycle and the
    // booking are the same whatever the lanes, none included).
    arithmetic_black_boxes_[static_cast<std::size_t>(arithmetic_unit(inst))].deliver(
        ArithmeticResult{warp.id, inst.dst, is_setp(inst.opcode), result, pending.active_mask}, cycle_,
        "execute_arithmetic");
}

// Which unit executes a fixed-latency instruction: the unit whose list
// (src/isa.cpp) has its opcode, asked in the order fixed_latency asks.
Machine::ArithmeticUnit Machine::arithmetic_unit(const Instruction& inst) {
    if (is_alu_operation(inst.opcode)) return ArithmeticUnit::alu;
    if (is_fpu_operation(inst.opcode)) return ArithmeticUnit::fpu;
    if (is_sfu_operation(inst.opcode)) return ArithmeticUnit::sfu;
    throw SimError("internal: an instruction with no fixed-latency unit in the arithmetic pipeline");
}

// ===========================================================================
// The three units as black boxes (src/machine/blackbox.hpp)
// ===========================================================================
//
// A unit takes an instruction when it executes (the arithmetic pipeline's
// last stage): its operands from the vector RF's read registers, the one
// uniform word it read (ur2r's, or a uniform second source's; the ALU and the
// FPU), the instruction, its warp and its lane enables from that stage. It computes the
// whole result then (execute_arithmetic, above: the inside of all three) and
// holds it E cycles, to its output register. What a real unit holds in its
// stages is not simulated.
void Machine::declare_arithmetic_black_boxes() {
    const auto& c = config_;
    const uint64_t lanes = c.lanes;
    const uint64_t lane_bits = lanes * 32;
    const auto operands = [&](uint64_t words) {
        return BlackBoxPort{std::to_string(words) + " vector operand" + (words == 1 ? "" : "s"), words * lane_bits,
            "the vector RF's read registers (the memory's)"};
    };
    const BlackBoxPort control{"instruction, warp, lane enables", 64 + bits_for(c.core_warps) + lanes,
        "the arithmetic pipeline's last stage (in the open)"};
    // A uniform word an instruction read in its round (ur2r's source, or the
    // second source of an instruction with its U bit): in the operand latches
    // when the URF's read delay is shorter than the RF's (it waits there),
    // else straight from the URF's read registers. One word, the same for
    // every lane.
    const auto uniform_operand = [&](const std::string& name) {
        return BlackBoxPort{name, 32,
            c.urf_read_latency_cycles < c.rf_read_latency_cycles
                ? "the arithmetic pipeline's operand latches (in the open)"
                : "the uniform register file's read registers (the memory's)"};
    };
    const auto declare = [&](ArithmeticUnit unit, BlackBoxDeclaration black_box, uint64_t result_bits) {
        auto& at = arithmetic_black_boxes_[static_cast<std::size_t>(unit)];
        black_box.unit = Unit::arithmetic;
        black_box.outputs = {BlackBoxPort{"result (warp, destination, lanes' words, lane enables)", result_bits + 1,
            at.name() + " (in the open)"}};
        at.build(black_box);
        black_boxes_.push_back(std::move(black_box));
    };

    // The ALU (mov, sread, ur2r, the integer and logic ops, shifts, setp).
    // The tag travels with a result because the write port needs its
    // destination: the write was booked ahead, but a booking does not say
    // where the result goes.
    {
        BlackBoxDeclaration alu("ALU");
        alu.latency = c.timing.alu;
        alu.latency_from = "core.alu_latency_cycles";
        alu.inputs = {operands(2),
            uniform_operand("uniform operand (ur2r; the second source of add, and, or, xor, shl, shr, setp)"),
            control};
        declare(ArithmeticUnit::alu, std::move(alu), bits_for(c.core_warps) + 8 + lane_bits + lanes + 1);
    }

    // The FPU (fadd, fsub, fmul, fma, fmax, cvt.pack), every lane each cycle.
    // With a uniform source an instruction's vector operands are one fewer.
    {
        BlackBoxDeclaration fpu("FPU");
        fpu.latency = c.timing.fadd;
        fpu.latency_from = "core.fadd_latency_cycles";
        fpu.inputs = {operands(3),
            uniform_operand("uniform operand (the second source of fadd, fsub, fmul, fmax, cvt.pack; fma's multiplicand b)"),
            control};
        declare(ArithmeticUnit::fpu, std::move(fpu), bits_for(c.core_warps) + 8 + lane_bits + lanes);
    }

    // The special-function unit (ex2, rcp). It serves sfu.lanes lanes a
    // cycle, so an instruction's G = ceil(lanes / sfu.lanes) lane groups start
    // one a cycle (issue keeps two instructions G apart: the interval) and
    // the last leaves the function pipeline sfu.latency_cycles (Ls) after it
    // starts: E = G - 1 + Ls.
    {
        const uint64_t groups = (lanes + c.sfu_lanes - 1) / c.sfu_lanes;
        BlackBoxDeclaration sfu("special-function unit");
        sfu.latency = c.timing.sfu;
        sfu.latency_from = "ceil(lanes / sfu.lanes) - 1 + sfu.latency_cycles";
        sfu.interval = static_cast<uint32_t>(groups);
        sfu.inputs = {operands(1), control};
        declare(ArithmeticUnit::sfu, std::move(sfu), bits_for(c.core_warps) + 8 + lane_bits + lanes);
    }
}

} // namespace nanosm

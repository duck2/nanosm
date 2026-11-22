// The arithmetic pipeline: ALU, FPU and the special-function unit.
// (Machine is one class, src/machine/machine.hpp; the map of its files is at the top of
// src/machine/machine.cpp.)

#include "machine/machine.hpp"
#include "machine/mma_numeric.hpp"
#include "machine/internal.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <set>
#include <tuple>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace nanosm {

namespace {

// Every new float opcode canonicalizes NaN outputs to 0x7fc00000: the sim is
// the spec, and platform NaN-payload propagation must not leak into it.
// (fadd predates this rule and keeps its legacy behavior.)
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

} // namespace

// ===========================================================================
// Arithmetic pipeline
// ===========================================================================

void Machine::execute_arithmetic(ArithmeticRead& pending, Wires& w) {
    auto& warp = warps_.at(pending.warp);
    const auto& inst = pending.instruction;
    // A panic names the instruction's own pc (the warp's has moved on).
    const auto fail = [&](const std::string& reason) {
        throw_issue_panic(IssueSite{cycle_, pending.warp, pending.pc, inst, &config_.mma_shapes}, reason);
    };
    std::vector<uint32_t> result(config_.lanes, 0);
    // The round holds each source register once; duplicated operand slots
    // broadcast from the same captured latch.
    const auto operand = [&](uint8_t reg, uint32_t lane) {
        for (std::size_t i = 0; i < pending.reads.size(); ++i) {
            if (pending.reads[i] == reg) return pending.operands[i][lane];
        }
        throw SimError("internal arithmetic operand was not in its read round");
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
            result[lane] = operand(inst.src0, lane) + operand(inst.src1, lane);
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
            result[lane] = operand(inst.src0, lane) & operand(inst.src1, lane);
        }
        break;
    case Opcode::or_:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = operand(inst.src0, lane) | operand(inst.src1, lane);
        }
        break;
    case Opcode::xor_:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = operand(inst.src0, lane) ^ operand(inst.src1, lane);
        }
        break;
    // Register shifts use the low 5 bits of the per-lane amount: the datapath
    // is a physical 5-bit shifter, so the masking IS the hardware, not a
    // silently repaired error.
    case Opcode::shl:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = operand(inst.src0, lane) << (operand(inst.src1, lane) & 31);
        }
        break;
    case Opcode::shr:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = operand(inst.src0, lane) >> (operand(inst.src1, lane) & 31);
        }
        break;
    case Opcode::fadd:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = float_bits(
                bits_float(operand(inst.src0, lane)) + bits_float(operand(inst.src1, lane)));
        }
        break;
    case Opcode::fmul:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = canonical_float(
                bits_float(operand(inst.src0, lane)) * bits_float(operand(inst.src1, lane)));
        }
        break;
    case Opcode::fsub:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = canonical_float(
                bits_float(operand(inst.src0, lane)) - bits_float(operand(inst.src1, lane)));
        }
        break;
    case Opcode::fmax:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = float_max_bits(operand(inst.src0, lane), operand(inst.src1, lane));
        }
        break;
    case Opcode::fma:
        // Single rounding via std::fma; NaN outputs canonicalized above.
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = canonical_float(std::fma(
                bits_float(operand(inst.src0, lane)),
                bits_float(operand(inst.src1, lane)),
                bits_float(operand(inst.src2, lane))));
        }
        break;
    case Opcode::cvt_pack:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = static_cast<uint32_t>(float_to_half_bits(operand(inst.src0, lane))) |
                static_cast<uint32_t>(float_to_half_bits(operand(inst.src1, lane))) << 16;
        }
        break;
    case Opcode::ex2:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = sfu_ex2(operand(inst.src0, lane));
        }
        break;
    case Opcode::rcp:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = sfu_rcp(operand(inst.src0, lane));
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
                : operand(inst.src1, lane);
            result[lane] = compare_setp(inst.opcode, operand(inst.src0, lane), rhs);
        }
        break;
    default: fail("non-arithmetic instruction in RF read pipeline");
    }
    // The execution pipeline's E cycles, then the write port it booked. Every
    // lane computes; the issue cycle's active mask AND the instruction's write
    // mask (sampled with the operands) is the write's lane enables (the port, its cycle and the
    // booking are the same whatever the lanes, none included).
    const auto port_cycle = pending.issue_cycle + issue_to_port(inst, config_);
    auto writeback = is_setp(inst.opcode)
        ? predicate_writeback(warp, inst.dst, port_cycle, std::move(result))
        : register_writeback(warp, inst.dst, port_cycle, std::move(result), opcode_name(inst.opcode));
    writeback.lane_enables = pending.active_mask;
    w.new_writebacks.push_back(std::move(writeback));
}

} // namespace nanosm

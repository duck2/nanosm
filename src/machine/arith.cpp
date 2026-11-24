// The arithmetic pipeline: ALU, FPU and the special-function unit.
// (Machine is one class, src/machine/machine.hpp; the map of its files is at the top of
// src/machine/machine.cpp.)

#include "machine/machine.hpp"
#include "machine/internal.hpp"

#include <algorithm>
#include <cmath>
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
    // Into its unit's black box: in its output register E cycles from now,
    // at the write port it booked. Every lane computes; the issue cycle's
    // active mask AND the instruction's write mask (sampled with the
    // operands) is the write's lane enables (the port, its cycle and the
    // booking are the same whatever the lanes, none included).
    arithmetic_black_boxes_[static_cast<std::size_t>(arithmetic_unit(inst))].deliver(
        ArithmeticResult{warp.id, inst.dst, is_setp(inst.opcode), result, pending.active_mask}, cycle_,
        "execute_arithmetic");
}

// BLACK BOX SHARED: arithmetic unit routing
// Which unit executes a fixed-latency instruction: the unit whose list
// (src/isa.cpp) has its opcode, asked in the order fixed_latency asks.
Machine::ArithmeticUnit Machine::arithmetic_unit(const Instruction& inst) {
    if (is_alu_operation(inst.opcode)) return ArithmeticUnit::alu;
    if (is_fpu_operation(inst.opcode)) return ArithmeticUnit::fpu;
    if (is_sfu_operation(inst.opcode)) return ArithmeticUnit::sfu;
    throw SimError("internal: an instruction with no fixed-latency unit in the arithmetic pipeline");
}
// END BLACK BOX SHARED: arithmetic unit routing

// ===========================================================================
// The three units as black boxes (src/machine/blackbox.hpp)
// ===========================================================================
//
// BLACK BOX SHARED: arithmetic units
// A unit takes an instruction when it executes (the arithmetic pipeline's
// last stage): its operands from the vector RF's read registers, the
// instruction, its warp and its lane enables from that stage. It computes the
// whole result then (execute_arithmetic, above: the inside of all three, not
// in their markers) and holds it E cycles, to its output register. What a
// real unit holds in its stages is its declared cost below; none of it is
// simulated. Each unit is between its own markers with its own sign-off
// line; this part, which all three use, is in each one's fingerprint.
void Machine::declare_arithmetic_black_boxes() {
    const auto& c = config_;
    const uint64_t lanes = c.lanes;
    const uint64_t lane_bits = lanes * 32;
    // A result's tag: warp, destination, lane enables and a valid bit (the
    // ALU's also says whether it is a setp's predicate).
    const uint64_t tag = bits_for(c.core_warps) + 8 + lanes + 1;
    const auto operands = [&](uint64_t words) {
        return BlackBoxPort{std::to_string(words) + " vector operand" + (words == 1 ? "" : "s"), words * lane_bits,
            "the vector RF's read registers (the memory's)"};
    };
    const BlackBoxPort control{"instruction, warp, lane enables", 64 + bits_for(c.core_warps) + lanes,
        "the arithmetic pipeline's last stage (in the open)"};
    const auto declare = [&](ArithmeticUnit unit, BlackBoxDeclaration black_box, uint64_t result_bits) {
        auto& at = arithmetic_black_boxes_[static_cast<std::size_t>(unit)];
        black_box.unit = Unit::arithmetic;
        black_box.outputs = {BlackBoxPort{"result (warp, destination, lanes' words, lane enables)", result_bits + 1,
            at.name() + " (in the open)"}};
        at.declare_output_bits(result_bits);
        at.build(black_box);
        black_boxes_.push_back(std::move(black_box));
    };
    const auto text = [](uint64_t value) { return std::to_string(value); };
    // END BLACK BOX SHARED: arithmetic units

    // BLACK BOX: ALU
    // BLACK BOX USES: arithmetic units
    // BLACK BOX USES: arithmetic unit order
    // BLACK BOX USES: arithmetic output registers
    // BLACK BOX USES: arithmetic unit routing
    // BLACK BOX USES: arithmetic unit latencies
    // BLACK BOX USES: ALU operations
    // BLACK BOX USES: ALU and FPU latencies
    // BLACK BOX USES: bit widths
    // The ALU (mov, sread, ur2r, the integer and logic ops, shifts, setp).
    // It would probably use, per lane, a 32-bit adder, the logic ops, a
    // barrel shifter and a comparator between the RF's read registers and its
    // output register, so at E = 1 nothing inside it is a register; each
    // further stage would hold every lane's result and the tag.
    {
        const uint64_t e = c.timing.alu;
        BlackBoxDeclaration alu("ALU", SignOff::not_signed); // sign-off: only the human changes this line
        alu.latency = c.timing.alu;
        alu.latency_from = "core.alu_latency_cycles";
        // ur2r's uniform word: in the operand latches when the URF's read
        // delay is shorter than the RF's (it waits there), else straight from
        // the URF's read registers.
        alu.inputs = {operands(2),
            BlackBoxPort{"uniform operand (ur2r)", 32,
                c.urf_read_latency_cycles < c.rf_read_latency_cycles
                    ? "the arithmetic pipeline's operand latches (in the open)"
                    : "the uniform register file's read registers (the memory's)"},
            control};
        alu.cost.flip_flops = (e - 1) * (lane_bits + tag + 1);
        alu.cost_function = "(E - 1) x (lanes x 32 + tag) flip-flops; E " + text(e) + ", lanes " + text(lanes) +
            ", tag " + text(tag + 1);
        declare(ArithmeticUnit::alu, std::move(alu), bits_for(c.core_warps) + 8 + lane_bits + lanes + 1);
    }
    // END BLACK BOX: ALU

    // BLACK BOX: FPU
    // BLACK BOX USES: arithmetic units
    // BLACK BOX USES: arithmetic unit order
    // BLACK BOX USES: arithmetic output registers
    // BLACK BOX USES: arithmetic unit routing
    // BLACK BOX USES: arithmetic unit latencies
    // BLACK BOX USES: FPU operations
    // BLACK BOX USES: ALU and FPU latencies
    // BLACK BOX USES: bit widths
    // The FPU (fadd, fsub, fmul, fma, fmax, cvt.pack), every lane each cycle.
    // It would probably use, per lane, a single-precision fused multiply-add
    // (fadd, fmul and fmax are its special cases): a 24 x 24 significand
    // product on two hard multipliers, and between its E - 1 inner stages
    // about 64 bits a lane (the 48-bit product and the addend aligned to it,
    // with the exponent) and the tag.
    {
        const uint64_t e = c.timing.fadd;
        BlackBoxDeclaration fpu("FPU", SignOff::not_signed); // sign-off: only the human changes this line
        fpu.latency = c.timing.fadd;
        fpu.latency_from = "core.fadd_latency_cycles";
        fpu.inputs = {operands(3), control};
        fpu.cost.flip_flops = (e - 1) * (lanes * 64 + tag);
        fpu.cost.multipliers = 2 * lanes;
        fpu.cost_function = "(E - 1) x (lanes x 64 + tag) flip-flops, 2 x lanes multipliers; E " + text(e) +
            ", lanes " + text(lanes) + ", tag " + text(tag);
        declare(ArithmeticUnit::fpu, std::move(fpu), bits_for(c.core_warps) + 8 + lane_bits + lanes);
    }
    // END BLACK BOX: FPU

    // BLACK BOX: special-function unit
    // BLACK BOX USES: arithmetic units
    // BLACK BOX USES: arithmetic unit order
    // BLACK BOX USES: arithmetic output registers
    // BLACK BOX USES: arithmetic unit routing
    // BLACK BOX USES: arithmetic unit latencies
    // BLACK BOX USES: special-function unit operations
    // BLACK BOX USES: special-function unit latency
    // BLACK BOX USES: bit widths
    // The special-function unit (ex2, rcp). It serves sfu.lanes lanes a
    // cycle, so an instruction's G = ceil(lanes / sfu.lanes) lane groups start
    // one a cycle (issue keeps two instructions G apart: the interval) and
    // the last leaves the function pipeline sfu.latency_cycles (Ls) after it
    // starts: E = G - 1 + Ls. It would probably use, per lane served, a table
    // of 128 entries of three polynomial coefficients (64 bits) for each of
    // its two functions and a quadratic in Horner form (two multipliers),
    // with about 64 bits a lane between stages. (The tables assume a
    // single-port ROM per lane; lanes sharing dual-port block RAMs in pairs
    // would halve their block RAMs, not their bits.) Its registers: the operands
    // of the G - 1 groups not yet started (the RF's read registers hold them
    // only in the cycle it takes them), the function pipeline's Ls - 1 inner
    // stages (their lanes, a valid bit and the group number), the results of
    // the G - 1 groups before the last (the last goes to the output
    // register), and the tags of the instructions inside, at most
    // ceil((E - 1) / G).
    {
        const uint64_t groups = (lanes + c.sfu_lanes - 1) / c.sfu_lanes;
        const uint64_t ls = c.sfu_latency_cycles;
        const uint64_t e = c.timing.sfu;
        const uint64_t inside = (e - 1 + groups - 1) / groups;
        BlackBoxDeclaration sfu("special-function unit", SignOff::not_signed); // sign-off: only the human changes this line
        sfu.latency = c.timing.sfu;
        sfu.latency_from = "ceil(lanes / sfu.lanes) - 1 + sfu.latency_cycles";
        sfu.interval = static_cast<uint32_t>(groups);
        sfu.inputs = {operands(1), control};
        sfu.cost.flip_flops = 2 * (groups - 1) * c.sfu_lanes * 32 + (ls - 1) * (c.sfu_lanes * 64 + 1 + bits_for(groups)) +
            inside * tag;
        sfu.cost.memory_bits = uint64_t{c.sfu_lanes} * 2 * 128 * 64;
        sfu.cost.multipliers = 2 * uint64_t{c.sfu_lanes};
        sfu.cost_function = "2 x (G - 1) x sfu.lanes x 32 + (Ls - 1) x (sfu.lanes x 64 + 1 + bits(G)) + "
            "ceil((E - 1) / G) x tag flip-flops, sfu.lanes x 2 x 128 x 64 table bits, 2 x sfu.lanes multipliers; G " +
            text(groups) + ", sfu.lanes " + text(c.sfu_lanes) + ", Ls " + text(ls) + ", E " + text(e) + ", tag " + text(tag);
        declare(ArithmeticUnit::sfu, std::move(sfu), bits_for(c.core_warps) + 8 + lane_bits + lanes);
    }
    // END BLACK BOX: special-function unit
}

} // namespace nanosm

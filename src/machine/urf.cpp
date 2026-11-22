// The uniform register file: its read rounds and its write pipeline.
// (Machine is one class, src/machine/machine.hpp; the map of its files is at the top of
// src/machine/machine.cpp.)

#include "machine/machine.hpp"
#include "machine/internal.hpp"

#include <algorithm>
#include <utility>

namespace nanosm {

// The round counter's instruction has rounds still to start after the one it
// samples now: an instruction accepted now would sample a uniform register
// in one of them.
bool Machine::urf_read_ports_held() const {
    const auto& rounds = urf_read_rounds_.cur();
    return rounds.has_value() && rounds->round + 1 < uniform_read_rounds(rounds->header.instruction).size();
}

uint32_t Machine::uniform_at_next_sample(uint32_t warp, uint8_t reg) const {
    const auto& landing = urf_pipe_.cur().back();
    if (landing.has_value() && landing->warp == warp && !is_usetp(landing->instruction.opcode) &&
        landing->instruction.dst == reg && landing->value.has_value()) {
        return *landing->value;
    }
    return warps_.at(warp).urf.at(reg);
}

// ===========================================================================
// The URF write pipeline (src/timing.hpp, UrfWrite in src/machine/machine.hpp)
// ===========================================================================

// From Q: the write in stage L - 1 is presented at the URF write port now
// (the trace's "UR write"); the one in the last stage is at the RAM now and
// lands at this cycle's clock edge, after every read of the cycle
// (land_urf_write). Each stage holds at most one write, so the one write
// port is never contended. A usetp in stage L - 1 presents its predicate at
// the predicate file instead -- the same bit in every lane, taken by the
// flip-flops at this cycle's edge (land_rf_writes) -- and leaves (step_urf).
void Machine::urf_ram_stage(Wires& w) {
    const auto& pipe = urf_pipe_.cur();
    if (const auto& presented = pipe.at(config_.urf_issue_to_port_cycles - 1); presented.has_value()) {
        if (!presented->value.has_value()) {
            throw SimError("internal: a uniform write reached the URF write port without its value");
        }
        if (is_usetp(presented->instruction.opcode)) {
            auto predicate = predicate_writeback(warps_.at(presented->warp), presented->instruction.dst, cycle_,
                std::vector<uint32_t>(config_.lanes, *presented->value));
            predicate.reason = "usetp";
            predicate.lane_enables = presented->active_mask;
            w.rf_landings.push_back(std::move(predicate));
        } else {
            trace_->urf_write(cycle_, presented->warp, presented->instruction.dst, *presented->value);
        }
    }
    if (pipe.back().has_value()) w.urf_landing = pipe.back();
}

// The pipeline's next state: every write moves one stage on. The one leaving
// stage 0 latches the operands it sampled now (step_rf_reads). A uniform-
// arithmetic write whose operands are at the URF output now (stage Lu)
// executes on the uniform ALU, and its result rides the remaining stages as
// the write's value (the ALU's Eu stages and the padding to L). r2ur's lane 0
// is at the RF output at stage Lr, which may be the write port's own stage
// (canonical: Lr 2 = L - 1, the RF output wired to the URF write port); the
// RF's read stages carry it until then, so the write carries it as its value
// from the sample on. Issue's accepted writer enters stage 0; the write in
// the last stage leaves (it lands now), and so does a usetp at the write
// port's stage (its predicate is presented now: urf_ram_stage).
void Machine::step_urf(Wires& w) {
    const auto& pipe = urf_pipe_.cur();
    UrfWritePipeline next(pipe.size());
    for (std::size_t stage = 0; stage + 1 < pipe.size(); ++stage) {
        if (!pipe[stage].has_value()) continue;
        if (is_usetp(pipe[stage]->instruction.opcode) && stage + 1 == config_.urf_issue_to_port_cycles) continue;
        auto write = *pipe[stage];
        if (stage == 0) write.operands = w.urf_sample.value();
        if (write.instruction.opcode == Opcode::r2ur) {
            if (stage == 0) write.value = write.operands.at(0);
        } else if (stage == config_.urf_read_latency_cycles) {
            write.value = execute_uniform(write);
        }
        next[stage + 1] = std::move(write);
    }
    next.front() = w.issue.urf_write;
    w.urf_pipe_next = std::move(next);
}

// The uniform ALU: one 32-bit result from the write's latched operands (its
// uniform sources in operand order) and its immediate. Register shifts use
// the low 5 bits, as the vector ALU's. A usetp's result is its predicate, 0
// or 1 (setp's comparisons: unsigned, the .s32 ones signed).
uint32_t Machine::execute_uniform(const UrfWrite& write) const {
    const auto& inst = write.instruction;
    const auto fail = [&](const std::string& reason) {
        throw_issue_panic(IssueSite{cycle_, write.warp, write.pc, inst, &config_.mma_shapes}, reason);
    };
    const auto a = [&] { return write.operands.at(0); };
    const auto b = [&] { return write.operands.at(1); };
    const auto imm = static_cast<uint32_t>(inst.imm);
    switch (inst.opcode) {
    case Opcode::umov: return a();
    case Opcode::umovi: return imm;
    case Opcode::uadd: return a() + b();
    case Opcode::uaddi: return a() + imm;
    case Opcode::usub: return a() - b();
    case Opcode::ushl: return a() << (b() & 31);
    case Opcode::ushr: return a() >> (b() & 31);
    case Opcode::ushli:
    case Opcode::ushri:
        if (inst.imm < 0 || inst.imm > 31) fail("shift outside 0..31");
        return inst.opcode == Opcode::ushli ? a() << inst.imm : a() >> inst.imm;
    case Opcode::uand: return a() & b();
    case Opcode::uor: return a() | b();
    case Opcode::uxor: return a() ^ b();
    case Opcode::uandi: return a() & imm;
    case Opcode::uori: return a() | imm;
    case Opcode::uxori: return a() ^ imm;
    case Opcode::usread:
        if (inst.imm == static_cast<int32_t>(SpecialReg::warp_id)) return write.warp;
        if (inst.imm == static_cast<int32_t>(SpecialReg::n_warps)) return static_cast<uint32_t>(warps_.size());
        fail("invalid special register for usread");
        break;
    default:
        if (is_usetp(inst.opcode)) {
            return compare_setp(inst.opcode, a(), is_usetp_immediate(inst.opcode) ? imm : b()) ? 1u : 0u;
        }
        break;
    }
    throw_issue_panic(IssueSite{cycle_, write.warp, write.pc, inst, &config_.mma_shapes}, "not uniform arithmetic");
}

// The clock edge's URF write (after registers_.commit()).
void Machine::land_urf_write(const Wires& w) {
    if (!w.urf_landing.has_value()) return;
    warps_.at(w.urf_landing->warp).urf.at(w.urf_landing->instruction.dst) = w.urf_landing->value.value();
}

// The URF write pipeline's next state (step_urf); held while it is empty.
void Machine::drive_urf_pipe(const Wires& w) {
    const auto busy = [](const UrfWritePipeline& pipe) {
        return std::any_of(pipe.begin(), pipe.end(), [](const auto& stage) { return stage.has_value(); });
    };
    if (!busy(urf_pipe_.cur()) && !busy(w.urf_pipe_next)) return;
    urf_pipe_.set_next(w.urf_pipe_next, "drive_urf_pipe");
}

} // namespace nanosm

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
    return rounds.has_value() && rounds->later_rounds != 0;
}

uint32_t Machine::uniform_at_next_sample(uint32_t warp, uint8_t reg) const {
    const auto& landing = urf_pipe_.cur().back();
    if (landing.has_value() && landing->warp == warp && !is_usetp(landing->instruction.opcode) &&
        landing->instruction.dst == reg && landing->value.has_value()) {
        return *landing->value;
    }
    return urf_.peek(0, urf_row(warp, reg))[0];
}

uint32_t Machine::read_urf(uint32_t warp, uint8_t reg, const char* tag) {
    if (reg >= config_.urf_registers) throw SimError("internal: uniform register " + std::to_string(reg) + " read");
    return urf_.read(0, urf_row(warp, reg), tag)[0];
}

// ===========================================================================
// The URF in the RF read stage (src/timing.hpp): the issue register's round
// samples its uniform registers now (READ_FIRST: a write landing at this
// cycle's edge is not seen), through the URF's two read ports; so do the URF
// write pipeline's stage 0 and the round counter's round. Issue refused every
// uniform reader that would sample with one of the counter's later rounds.
// ===========================================================================

// A round's uniform registers, in order, through the URF's read ports now:
// counted for check_urf_round_reads.
std::vector<uint32_t> Machine::read_urf_round(Wires& w, uint32_t warp, const std::vector<uint8_t>& regs, const char* tag) {
    w.round_reads.urf += static_cast<uint32_t>(regs.size());
    std::vector<uint32_t> values;
    for (const auto reg : regs) values.push_back(read_urf(warp, reg, tag));
    return values;
}

// The URF write pipeline's stage 0 (the issue register's instruction):
// uniform arithmetic samples its uniform sources (in operand order; a
// register named twice is one read), r2ur its vector source through the RF
// round's read port (lane 0 is what it keeps). The operands latch in
// step_urf.
void Machine::sample_urf_pipe(Wires& w) {
    const auto& write = urf_pipe_.cur().front();
    if (!write.has_value()) return;
    const auto& inst = write->instruction;
    if (inst.opcode == Opcode::r2ur) {
        w.urf_sample = std::vector<uint32_t>{read_rf_round(w, write->warp, inst.src0).at(0)};
        return;
    }
    const auto sources = uniform_sources(inst);
    w.round_reads.urf += static_cast<uint32_t>(rf_read_round(sources).size());
    std::vector<uint32_t> values;
    for (const auto reg : sources) values.push_back(read_urf(write->warp, reg, "URF read: uniform ALU sample"));
    w.urf_sample = std::move(values);
}

// The round counter's round now (src/timing.hpp uniform_read_rounds): its
// registers by operand position, in the counter's warp. A tma's pair and its
// tag enter the TMA operand read's URF read stages; an mma.async's later
// round (round 0 is its dispatch entry's own sample) joins its dispatch
// entry, the one issued at the counter's issue cycle, k stages on at round k.
void Machine::sample_urf_rounds(Wires& w) {
    const auto& rounds = urf_read_rounds_.cur();
    if (!rounds.has_value()) return;
    const auto& inst = rounds->header.instruction;
    const auto plan = uniform_read_rounds(inst);
    const auto& registers = plan.at(rounds->round);
    const auto warp = rounds->header.warp;
    if (is_tma_opcode(inst.opcode)) {
        TmaReadPair pair;
        pair.round = rounds->round;
        pair.last = rounds->later_rounds == 0;
        if (rounds->round == 0) pair.header = rounds->header;
        pair.words = read_urf_round(w, warp, registers, "URF read: a tma's round");
        w.tma_sampled = std::move(pair);
    } else if (rounds->round > 0) {
        std::size_t first = 0;
        for (uint32_t round = 0; round < rounds->round; ++round) first += plan[round].size();
        w.dispatch_samples.push_back(
            {rounds->round, first, warp, read_urf_round(w, warp, registers, "URF read: an mma.async's later round")});
    }
}

// After the read stage: the URF's samples this cycle -- these and the one
// memory-queue sample of the cycle (sample_newest) -- never exceed its read
// ports.
void Machine::check_urf_round_reads(const Wires& w) const {
    auto reads = w.round_reads.urf;
    if (w.queue_sample.has_value()) reads += static_cast<uint32_t>(w.queue_sample->uniform_operands.size());
    if (reads > urf_read_ports) {
        throw SimError("internal: the URF sampled " + std::to_string(reads) + " uniform registers at cycle " +
            std::to_string(cycle_) + " (it has " + std::to_string(urf_read_ports) + " read ports)");
    }
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
// flip-flops at this cycle's edge (drive_predicates) -- and leaves (step_urf).
void Machine::urf_ram_stage(Wires& w) {
    const auto& pipe = urf_pipe_.cur();
    if (const auto& presented = pipe.at(config_.urf_issue_to_port_cycles - 1); presented.has_value()) {
        if (!presented->value.has_value()) {
            throw SimError("internal: a uniform write reached the URF write port without its value");
        }
        if (is_usetp(presented->instruction.opcode)) {
            w.predicate_writes.push_back(PredicateWrite{presented->warp, presented->instruction.dst,
                *presented->value != 0 ? full_mask() : 0, presented->active_mask});
        } else {
            trace_->urf_write(cycle_, presented->warp, presented->instruction.dst, *presented->value);
        }
    }
    if (pipe.back().has_value()) w.urf_landing = pipe.back();
}

// The pipeline's work this cycle, by stage. The write leaving
// stage 0 latches the operands it sampled now (sample_urf_pipe). A uniform-
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
    for (std::size_t stage = 0; stage + 1 < pipe.size(); ++stage) {
        if (!pipe[stage].has_value()) continue;
        if (is_usetp(pipe[stage]->instruction.opcode) && stage + 1 == config_.urf_issue_to_port_cycles) {
            w.urf_stage_work.emplace_back(stage, std::nullopt);
            continue;
        }
        const bool r2ur = pipe[stage]->instruction.opcode == Opcode::r2ur;
        const bool latches = stage == 0;
        const bool executes = !r2ur && stage == config_.urf_read_latency_cycles;
        if (!latches && !executes) continue;
        auto write = *pipe[stage];
        if (latches) {
            write.operands = w.urf_sample.value();
            if (r2ur) write.value = write.operands.at(0);
        }
        if (executes) write.value = execute_uniform(write);
        w.urf_stage_work.emplace_back(stage, std::move(write));
    }
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

// The URF write at its RAM now, through the write port: it lands at the clock
// edge.
void Machine::land_urf_write(const Wires& w) {
    if (!w.urf_landing.has_value()) return;
    const auto& landing = *w.urf_landing;
    if (landing.instruction.dst >= config_.urf_registers) throw SimError("internal: a uniform write past the URF");
    const uint32_t value = landing.value.value();
    urf_.write(0, urf_row(landing.warp, landing.instruction.dst), std::span<const uint32_t>(&value, 1),
        Memory<uint32_t>::every_element, "URF write port");
}

// The URF write pipeline: each stage's work (step_urf) goes on with its
// write; issue's accepted writer enters stage 0.
void Machine::drive_urf_pipe(const Wires& w) {
    for (const auto& [stage, write] : w.urf_stage_work) urf_pipe_.replace(stage, write, "drive_urf_pipe: stage work");
    if (w.issue.urf_write.has_value()) urf_pipe_.push(*w.issue.urf_write, "drive_urf_pipe");
}

} // namespace nanosm

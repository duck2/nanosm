// Dispatch: the dispatch pipeline and its end, where each unit takes an issued instruction
// (a unit's own dispatch_* function is in the unit's file), and a dispatch entry's operands.
// (Machine is one class, src/machine/machine.hpp; the map of its files is at the top of
// src/machine/machine.cpp.)

#include "machine/machine.hpp"

#include <utility>

namespace nanosm {

// ===========================================================================
// Dispatch: the dispatch pipeline's end. The unit takes the instruction with
// its operands -- latched at its RF sample, at the RF output now.
// ===========================================================================

void Machine::dispatch_issued(Wires& w) {
    const auto& leaving = dispatch_pipe_.last();
    if (!leaving.has_value()) return;
    w.dispatched = true;
    hand_over(*leaving, w);
}

void Machine::hand_over(const DispatchEntry& op, Wires& w) {
    if (op.active_mask == 0) {
        complete_without_lanes(op, w);
        return;
    }
    if (is_mma_async(op.instruction.opcode)) {
        dispatch_mma(op, w);
        return;
    }
    switch (op.instruction.opcode) {
    case Opcode::mma_commit:
        dispatch_mma(op, w);
        break;
    case Opcode::acc_ld:
    case Opcode::acc_ld_v8:
    case Opcode::acc_st:
    case Opcode::acc_st_v8:
        dispatch_acc_transfer(op, w);
        break;
    case Opcode::ldg:
    case Opcode::stg:
    case Opcode::ldg_v8:
    case Opcode::stg_v8:
    case Opcode::ldg_uni:
        dispatch_lsu(op, w);
        break;
    default:
        panic(op, "unknown opcode");
    }
}

// A masked ldg / stg (or their v8 forms) whose write mask turned every active
// lane off. Issue could not know: it took its slot (the
// admission's claim), set its depbar and booked its read round like any.
// Here it completes with zero accesses: no LSU
// command, no buffer, no request; its depbar is decremented now (and a
// stg.v8's read depbar; the others' fell at their issue round's sample),
// so a waiting instruction sees it zero from the next cycle (issue +
// issue_to_dispatch + 1, canonical issue + 4). A load's destination keeps
// every lane's old value and is not in flight from then on. Only a
// write mask can leave no lane: the divergence stack's active mask is never
// empty, and every other dispatched instruction refuses to be masked. (A
// masked lds / sts never reaches dispatch: src/machine/shmem.cpp drop_queued.)
void Machine::complete_without_lanes(const DispatchEntry& op, Wires& w) {
    const auto& inst = op.instruction;
    if (!can_be_masked(inst.opcode)) {
        throw SimError("internal: " + opcode_name(inst.opcode) + " reached dispatch with no active lane");
    }
    record_depbar_completion(w, op.warp, inst.control.set_depbar,
        "depbar underflow on sb" + std::to_string(inst.control.set_depbar));
    if (const auto dst = destination_register(inst); dst.has_value()) {
        const uint32_t registers = inst.opcode == Opcode::ldg_v8 ? 8 : 1;
        for (uint32_t j = 0; j < registers; ++j) sanitizer_.record_reg_ready(op.warp, *dst + j, cycle_ + 1);
    }
    if (inst.opcode == Opcode::stg_v8) { // its data is never read: its read depbar is given back too
        for (uint32_t j = 0; j < 8; ++j) sanitizer_.record_async_reader_done(op.warp, inst.src1 + j, AsyncReader::stg_v8);
        lower_read_depbar(w, op.warp, inst);
    }
}

// The dispatch pipeline: the RF read stage's samples latch into their
// entries, which go on carrying them; the entry in the last stage dispatched
// now and leaves; issue's accepted entry enters stage 0.
void Machine::drive_dispatch_pipe(const Wires& w) {
    for (const auto& sample : w.dispatch_samples) {
        auto entry = dispatch_pipe_.stage(sample.stage).value();
        if (!sample.later_round) entry.operands = sample.operands;
        if (sample.write_mask_lanes.has_value()) entry.active_mask &= *sample.write_mask_lanes;
        for (std::size_t i = 0; i < sample.uniform_operands.size(); ++i) {
            entry.uniform_operands.at(sample.uniform_first + i) = sample.uniform_operands[i];
        }
        dispatch_pipe_.replace(sample.stage, std::move(entry), "drive_dispatch_pipe: the sample");
    }
    if (w.issue.dispatch.has_value()) dispatch_pipe_.push(*w.issue.dispatch, "drive_dispatch_pipe");
}

// A dispatched instruction's operand, as its RF read latched it at its
// sample (step_rf_reads). It is never read from the RF array here.
std::vector<uint32_t> Machine::capture(const DispatchEntry& op, uint8_t reg) const {
    for (std::size_t i = 0; i < op.reads.size(); ++i) {
        if (op.reads[i] != reg) continue;
        if (op.operands[i].size() != config_.lanes) {
            throw SimError("internal: r" + std::to_string(reg) + " used at dispatch before its RF sample");
        }
        return op.operands[i];
    }
    throw SimError("internal: r" + std::to_string(reg) + " used at dispatch but not read at issue");
}

// A uniform-register operand (ldg.uni bases, acc.ld / acc.st tmem pointers,
// MMA addresses and result pointer, mma.commit's mbar), as its round latched it
// at its sample (step_rf_reads). Uniform by construction: no lane check. The
// uniform operation's partial-mask rule stays (the sanitizer's PANIC).
uint32_t Machine::uniform_operand(const DispatchEntry& op, uint8_t ureg) const {
    sanitizer_.check_uniform_mask(site(op), op.active_mask, full_mask());
    for (std::size_t i = 0; i < op.uniform_reads.size(); ++i) {
        if (op.uniform_reads[i] == ureg) return op.uniform_operands[i];
    }
    throw SimError("internal: ur" + std::to_string(ureg) + " used at dispatch but not read at issue");
}

} // namespace nanosm

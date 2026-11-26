// Dispatch: the dispatch pipeline and its end, where the tensor core and the accumulator
// engines take an issued instruction (a unit's own dispatch_* function is in the unit's
// file), and a dispatch entry's uniform operands. (Shared and global memory instructions
// wait in their units' queues instead: src/machine/shmem.cpp, src/machine/lsu.cpp.)
// (Machine is one class, src/machine/machine.hpp; the map of its files is at the top of
// src/machine/machine.cpp.)

#include "machine/machine.hpp"

#include <utility>

namespace nanosm {

// ===========================================================================
// Dispatch: the dispatch pipeline's end. The unit takes the instruction with
// its uniform operands -- latched at their samples, at the URF output now.
// ===========================================================================

void Machine::dispatch_issued(Wires& w) {
    const auto& leaving = dispatch_pipe_.last();
    if (!leaving.has_value()) return;
    w.dispatched = true;
    hand_over(*leaving, w);
}

void Machine::hand_over(const DispatchEntry& op, Wires& w) {
    if (op.active_mask == 0) throw SimError("internal: an engine instruction dispatched with no active lane");
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
    default:
        panic(op, "unknown opcode");
    }
}

// The dispatch pipeline: the RF read stage's uniform samples latch into
// their entries, which go on carrying them; the entry in the last stage
// dispatched now and leaves; issue's accepted entry enters stage 0.
void Machine::drive_dispatch_pipe(const Wires& w) {
    for (const auto& sample : w.dispatch_samples) {
        auto entry = dispatch_pipe_.stage(sample.stage).value();
        for (std::size_t i = 0; i < sample.uniform_operands.size(); ++i) {
            entry.uniform_operands.at(sample.uniform_first + i) = sample.uniform_operands[i];
        }
        dispatch_pipe_.replace(sample.stage, std::move(entry), "drive_dispatch_pipe: the sample");
    }
    if (w.issue.dispatch.has_value()) dispatch_pipe_.push(*w.issue.dispatch, "drive_dispatch_pipe");
}

// A uniform-register operand (acc.ld / acc.st tmem pointers,
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

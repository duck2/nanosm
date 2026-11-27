// Dispatch: how an issued instruction reaches its unit. The dispatch pipeline and its end,
// where the tensor core and the accumulator engines take an issued instruction (a unit's own
// dispatch_* function is in the unit's file), and a dispatch entry's uniform operands. The
// memory queues, where shared and global memory instructions wait instead: the entry issue
// hands a queue, its issue + 1 sample, the drop of one left with no lane, the head leaving,
// and the queues' next values -- the same for shared memory's two sides and the load / store
// unit (what leaves a queue goes on in src/machine/shmem.cpp and src/machine/lsu.cpp).
// (Machine is one class, src/machine/machine.hpp; the map of its files is at the top of
// src/machine/machine.cpp.)

#include "machine/machine.hpp"
#include "machine/internal.hpp"

#include <utility>

namespace nanosm {

// ===========================================================================
// Dispatch: the dispatch pipeline's end. The unit takes the instruction with
// its uniform operands -- latched at their samples, at the URF output now.
// ===========================================================================

void Machine::dispatch_issued(Wires& w) {
    const auto& leaving = dispatch_pipe_.last();
    if (!leaving.has_value()) return;
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

// The dispatch pipeline in the RF read stage: the instruction in stage 0
// (accepted last cycle: the issue register's) samples its round's uniform
// registers through the URF's read ports (none of these reads a vector
// register or may be masked). An mma.async's later round is the round
// counter's (sample_urf_rounds).
void Machine::sample_dispatch_round(Wires& w) {
    const auto& first = dispatch_pipe_.cur().front();
    if (!first.has_value()) return;
    if (const auto rounds = uniform_read_rounds(first->instruction); !rounds.empty()) {
        w.dispatch_samples.push_back(
            {0, 0, first->warp, read_urf_round(w, first->warp, rounds.front(), "URF read: the issue register's round")});
    }
}

// The dispatch pipeline: the RF read stage's uniform samples latch into
// their entries, which go on carrying them (an mma.async's later round finds
// its entry k stages on when the round counter is at round k: both move one
// stage a cycle from issue + 1); the entry in the last stage dispatched now
// and leaves; issue's accepted entry enters stage 0.
void Machine::drive_dispatch_pipe(const Wires& w) {
    for (const auto& sample : w.dispatch_samples) {
        const auto& pipe = dispatch_pipe_.cur();
        if (sample.stage >= pipe.size() || !pipe[sample.stage].has_value() || pipe[sample.stage]->warp != sample.warp) {
            throw SimError("internal: a later uniform round lost its dispatch entry");
        }
        auto entry = *pipe[sample.stage];
        for (std::size_t i = 0; i < sample.uniform_operands.size(); ++i) {
            entry.uniform_operands.at(sample.uniform_first + i) = sample.uniform_operands[i];
        }
        dispatch_pipe_.replace(sample.stage, std::move(entry), "drive_dispatch_pipe: the sample");
    }
    if (w.issue.dispatch.has_value()) dispatch_pipe_.push(*w.issue.dispatch, "drive_dispatch_pipe");
}

// A uniform-register operand (acc.ld / acc.st tmem pointers,
// MMA addresses and result pointer, mma.commit's mbar), as its round latched it
// at its sample (sample_dispatch_round, sample_urf_rounds). Uniform by construction: no lane check. The
// uniform operation's partial-mask rule stays (the sanitizer's PANIC).
uint32_t Machine::uniform_operand(const DispatchEntry& op, uint8_t ureg) const {
    sanitizer_.check_uniform_mask(site(op), op.active_mask, full_mask());
    for (std::size_t i = 0; i < op.uniform_reads.size(); ++i) {
        if (op.uniform_reads[i] == ureg) return op.uniform_operands[i];
    }
    throw SimError("internal: ur" + std::to_string(ureg) + " used at dispatch but not read at issue");
}

// ===========================================================================
// The memory queues: shared memory's read and write sides' and the load /
// store unit's, each strictly first in first out for every warp.
//   issue       the instruction enters its queue at its issue's edge with
//               the active mask (queued_entry); issue asks only for the
//               queue's credit. Its depbar and read depbar rise then.
//   issue + 1   the newest entry samples its predicate (lanes = the active
//               mask AND the write mask) and its uniform registers
//               (sample_newest). One left with no lane leaves the queue then,
//               taking no register read and no access, and gives back its
//               entry, both depbars and what else it took at issue
//               (drop_queued).
//   head        the head leaves when its wait is over and its unit takes it:
//               a side's collect stage (advance_shmem_sides), a command slot
//               (advance_lsu_queue).
//   collect     there it reads its vector registers late, by one rule
//               (next_collect_register, collect_read); the words wait in
//               each unit's own state (a side's collect stage and its
//               countdown, the command slots' read pipeline).
// ===========================================================================

// The entry issue hands a queue: the instruction with its warp, pc and the
// issue cycle's active mask, the uniform registers its sample reads, and its
// wait. Its uniform operands must be at the URF output where they are used:
// as it leaves the queue for a try_wait's wait slot and an ldg.uni's plan,
// else when its collect stage starts, the cycle after.
Machine::QueuedEntry Machine::queued_entry(const Warp& warp, const Instruction& inst, const IssuePlan& plan) const {
    QueuedEntry entry;
    entry.warp = warp.id;
    entry.pc = head_pc(warp.id);
    entry.lanes = active_mask(warp.id);
    entry.instruction = inst;
    entry.uniform_reads = plan.uniform_reads;
    const auto lu = config_.urf_read_latency_cycles;
    const bool used_as_it_leaves = is_mbar_try_wait(inst.opcode) || inst.opcode == Opcode::ldg_uni;
    entry.wait = plan.uniform_reads.empty() ? 0 : static_cast<uint8_t>(used_as_it_leaves ? lu : lu - 1);
    return entry;
}

// The vector registers a queued instruction reads after issue, in operand
// order, each with its reader: an lds's or sts's address and data (its
// collect stage); an ldg's, stg's or ldg.v8's address and a stg's data (its
// command slot); a stg.v8's address (its command slot) and eight data
// registers (its data beats). A one-word store whose two registers are one
// reads it once. The sanitizer holds each read-owned from issue to its read.
std::vector<std::pair<uint8_t, AsyncReader>> Machine::late_vector_reads(const Instruction& inst) {
    const auto sources = vector_sources(inst);
    const bool lsu = is_lsu_opcode(inst.opcode);
    const bool v8 = register_span(inst.opcode) > 1;
    std::vector<std::pair<uint8_t, AsyncReader>> reads;
    for (std::size_t i = 0; i < sources.size(); ++i) {
        if (i == 1 && !v8 && sources[1] == sources[0]) continue;
        reads.emplace_back(sources[i], !lsu ? AsyncReader::shmem_collect
                : v8 && i > 0             ? AsyncReader::stg_v8
                                          : AsyncReader::lsu);
    }
    return reads;
}

// The lanes an entry's sample leaves it: the active mask AND the write mask,
// read from the predicate file.
uint64_t Machine::sampled_lanes(const QueuedEntry& entry) const {
    const auto& inst = entry.instruction;
    return has_write_mask(inst) ? entry.lanes & write_mask_lanes(warps_[entry.warp], inst) : entry.lanes;
}

// An entry with its sample taken.
Machine::QueuedEntry Machine::with_sample(QueuedEntry entry, const Wires::QueueSample& sample) {
    entry.lanes = sample.lanes;
    entry.uniform_operands = sample.uniform_operands;
    entry.sampled = true;
    return entry;
}

// The entry that entered `queue` at the last edge samples, in the issue
// register's cycle (its issue + 1): its predicate (sampled_lanes) and its
// uniform registers through the URF's read ports (check_urf_round_reads
// counts them with the cycle's other URF reads). The uniform operation's partial-mask
// rule is the sanitizer's PANIC here. An entry with no lane left leaves the
// queue now (drop_queued) and gives its credit back; a shared-memory access
// is otherwise recorded for the sanitizer now (record_queued_access).
void Machine::sample_newest(MemoryQueue queue, Wires& w) {
    const auto& entries = queues_[queue_index(queue)].cur();
    if (entries.empty() || entries.back().sampled) return;
    if (w.queue_sample.has_value()) throw SimError("internal: two queues' entries sampled in one cycle");
    const auto& entry = entries.back();
    Wires::QueueSample sample{queue, entry.warp, sampled_lanes(entry), {}, false};
    for (const auto reg : entry.uniform_reads) {
        sample.uniform_operands.push_back(read_urf(entry.warp, reg, "URF read: a queued instruction's sample"));
    }
    if (!entry.uniform_reads.empty()) sanitizer_.check_uniform_mask(site(entry), entry.lanes, full_mask());
    sample.dropped = sample.lanes == 0;
    if (sample.dropped) {
        drop_queued(entry, w);
        return_credit(w, queue_credit(queue));
    } else if (queue != MemoryQueue::lsu) {
        record_queued_access(with_sample(entry, sample));
    }
    w.queue_sample = std::move(sample);
}

// An lds, sts, ldg, stg, ldg.v8 or stg.v8 whose write mask turned every
// active lane off: no register read and no access. Its depbar and its read
// depbar are decremented now (a waiting instruction sees them zero from the
// next cycle, issue + 2); a load's destination keeps every lane's old value
// and is not in flight from then on; its source registers are no longer read;
// its queue entry (and a shared-memory access's warp counts,
// drive_shmem_counts) is given back at this edge. Only a write mask can leave
// no lane: the divergence stack's active mask is never empty, and every other
// queued instruction refuses to be masked.
void Machine::drop_queued(const QueuedEntry& entry, Wires& w) {
    const auto& inst = entry.instruction;
    if (!can_be_masked(inst.opcode)) {
        throw SimError("internal: " + opcode_name(inst.opcode) + " was queued with no active lane");
    }
    record_depbar_completion(w, entry.warp, inst.control.set_depbar,
        "depbar underflow on sb" + std::to_string(inst.control.set_depbar));
    lower_read_depbar(w, entry.warp, inst);
    if (const auto dst = vector_destination(inst); dst.has_value()) {
        for (uint32_t j = 0; j < register_span(inst.opcode); ++j) sanitizer_.record_reg_ready(entry.warp, *dst + j, cycle_ + 1);
    }
    for (const auto& [reg, reader] : late_vector_reads(inst)) sanitizer_.record_async_reader_done(entry.warp, reg, reader);
}

// A queue's head may leave now: its uniform operands will be where they are
// used (its wait is over) and its sample, if it takes it now, leaves it a
// lane.
bool Machine::head_may_leave(MemoryQueue queue) const {
    const auto& entries = queues_[queue_index(queue)].cur();
    if (entries.empty()) return false;
    const auto& head = entries.front();
    return head.wait == 0 && (head.sampled || sampled_lanes(head) != 0);
}

// The head as it leaves its queue now: with its issue + 1 sample if it takes
// it now (it is then the newest entry too).
Machine::QueuedEntry Machine::leaving_head(MemoryQueue queue, const Wires& w) const {
    const auto& head = queues_[queue_index(queue)].cur().front();
    if (head.sampled) return head;
    const auto& sample = w.queue_sample;
    if (!sample.has_value() || sample->queue != queue || sample->dropped) {
        throw SimError("internal: a queue entry left before its sample");
    }
    return with_sample(head, *sample);
}

// The head leaves its queue at this edge, giving its credit back.
void Machine::pop_queue(Wires& w, MemoryQueue queue) {
    w.queue_pops[queue_index(queue)] = true;
    return_credit(w, queue_credit(queue));
}

Credit Machine::queue_credit(MemoryQueue queue) {
    switch (queue) {
    case MemoryQueue::shmem_read: return Credit::read_queue;
    case MemoryQueue::shmem_write: return Credit::write_queue;
    case MemoryQueue::lsu: return Credit::lsu_queue;
    }
    throw SimError("internal: an unknown memory queue");
}

// What has left a queue reads its vector registers through the RF's async
// read ports (decide_rf_reads), the same way in a side's collect stage and a
// command slot: its address register first; a pair's second register (a
// one-word store's data) after it, or with it in one cycle (`data`: then it
// takes a port only if one is left, outside the round-robin); one read when
// the two are one register. The register asked for next (none once its
// reads are granted):
std::optional<uint8_t> Machine::next_collect_register(
    const DecodedInstruction& inst, bool pair, bool address_read, bool data_read, bool data) {
    if (data) return pair && !address_read && inst.src1 != inst.src0 ? std::optional<uint8_t>{inst.src1} : std::nullopt;
    if (!address_read) return inst.src0;
    if (pair && !data_read) return inst.src1;
    return std::nullopt;
}

// What a read granted now latches: the address, when it is the named
// reader's first read, else the data; both, when a pair's two registers are
// one. A pair's last read is its data, anything else's its address.
Machine::CollectRead Machine::collect_read(const DecodedInstruction& inst, bool pair, bool address_read, bool named) {
    const bool address = named && !address_read;
    return CollectRead{address, !address || (pair && inst.src1 == inst.src0)};
}

// Each queue, in one calculation: its head leaves (popped); its newest entry
// takes its issue + 1 sample or, left with no lane, leaves (dropped: never
// the head popped now); every other entry's `wait` counts down; issue's new
// entry enters at the tail.
void Machine::drive_queues(const Wires& w) {
    for (std::size_t index = 0; index < memory_queues; ++index) {
        const auto queue = static_cast<MemoryQueue>(index);
        auto& reg = queues_[index];
        const bool pop = w.queue_pops[index];
        const bool push = w.issue.queued.has_value() && w.issue.queued->queue == queue;
        const auto& sample = w.queue_sample;
        const bool sampled = sample.has_value() && sample->queue == queue;
        bool changed = pop || push || sampled;
        auto next = reg.cur();
        for (auto& entry : next) {
            if (entry.wait != 0) {
                --entry.wait;
                changed = true;
            }
        }
        if (sampled) {
            if (sample->dropped) {
                next.resize(next.size() - 1);
            } else {
                next.back() = with_sample(next.back(), *sample);
            }
        }
        if (pop) next.pop_front();
        if (push) next.push_back(w.issue.queued->entry);
        if (!changed) continue;
        if (next.size() > credit_capacity(queue_credit(queue))) {
            throw SimError("internal: a memory queue holds more entries than the profile gives it");
        }
        reg.set_next(std::move(next), "drive_queues");
    }
}

} // namespace nanosm

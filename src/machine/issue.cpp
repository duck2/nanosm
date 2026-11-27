// Issue: admission, the scheduler, wait masks and depbar counters, accept, and the checks
// made when an instruction issues.
// (Machine is one class, src/machine/machine.hpp; the map of its files is at the top of
// src/machine/machine.cpp.)

#include "machine/internal.hpp"
#include "machine/machine.hpp"

#include <algorithm>
#include <limits>
#include <utility>

namespace nanosm {

namespace {

// The first depbar in the instruction's wait mask that is still nonzero (Q).
std::optional<uint8_t> depbar_wait(uint8_t wait_mask, const std::array<uint32_t, 6>& depbar) {
    for (uint8_t bar = 0; bar < 6; ++bar) {
        if ((wait_mask & (1u << bar)) != 0 && depbar[bar] != 0) return bar;
    }
    return std::nullopt;
}

} // namespace

// ===========================================================================
// Admission: whether the warp's head instruction can be accepted now, from
// its plan (decoded when its packet landed: Machine::predecode) and
// registers only (Q), never an operand: a register's, a predicate's or
// memory's contents. The plan selects WHICH register (a bank's inhibit bit,
// the write calendar's tap, a credit, the warp's counts). Machine::try_issue
// asks it before it captures anything. The refusals are tested in a fixed
// order, which decides the reason reported when several apply:
//   1. a bank the round reads has its read inhibit bit set (the RF read
//      policy: set by an async reader denied last cycle, rf.cpp
//      drive_rf_read_policy, so a denial is seen one cycle late);
//   2. the round reads a uniform register while a tma's or an mma.async's
//      later rounds own the URF's two read ports (src/timing.hpp
//      uniform_read_rounds; a tma or an mma.async included);
//   3. what the kind needs: arithmetic the SFU's occupancy countdown, then
//      its write booking's calendar entry and its bank's write inhibit bit
//      (the RF write policy, a cycle late like the read's); fence.sh every
//      access of its warp at the RAM (its counts); a shared-memory access its
//      warp's order between the two sides (a read waits for the warp's
//      unserviced writes, a write for its unserviced reads; each side keeps a
//      warp's accesses in order by itself), then its queue's credit; the
//      tma, mma, acc.ld / acc.st and global-memory forms their credit. A
//      uniform-register writer (or usetp) needs nothing more: its write port
//      is its own pipeline stage (src/timing.hpp; usetp's predicate goes to
//      the flip-flops at that stage). Nor do control, nop and halt.
// A set depbar needs counter room too; no counter width is declared (S5),
// and a warp can hold far fewer async operations than any width would
// count, so that never refuses. Space a unit frees at t is seen from t + 1.
// ===========================================================================

std::optional<Refusal> Machine::admission_refusal(const Warp& warp, const IssuePlan& plan) const {
    for (const auto bank : plan.read_banks) {
        if (rf_read_policy_.cur()[bank].inhibit != 0) return Refusal::rf_read_inhibited;
    }
    if (!plan.uniform_reads.empty() && urf_read_ports_held()) return Refusal::urf_read_ports_held;
    const auto credit = [this](Credit which, Refusal refusal) {
        return credit_free(which) ? std::nullopt : std::optional{refusal};
    };
    const auto& counts = shmem_counts_.cur().at(warp.id);
    switch (plan.unit) {
    case IssueUnit::alu:
    case IssueUnit::fpu:
    case IssueUnit::sfu:
        if (plan.unit == IssueUnit::sfu && sfu_busy_.cur() != 0) return Refusal::sfu_busy;
        if (plan.write_reg.has_value()) {
            if (!write_slot_free(plan.write_bank, plan.write_delay)) return Refusal::rf_write_booked;
            if (rf_write_policy_.cur()[plan.write_bank].inhibit != 0) return Refusal::rf_write_inhibited;
        }
        return std::nullopt;
    case IssueUnit::tma: return credit(Credit::tma_queue, Refusal::tma_queue_full);
    case IssueUnit::mma: return credit(Credit::mma_queue, Refusal::mma_queue_full);
    case IssueUnit::acc_load: return credit(Credit::acc_ld, Refusal::acc_engine_busy);
    case IssueUnit::acc_store: return credit(Credit::acc_st, Refusal::acc_engine_busy);
    case IssueUnit::fence: return counts.not_at_ram == 0 ? std::nullopt : std::optional{Refusal::shmem_fence};
    case IssueUnit::shared_read:
        if (counts.writes != 0) return Refusal::shmem_ordered;
        return credit(Credit::read_queue, Refusal::read_queue_full);
    case IssueUnit::shared_write:
        if (counts.reads != 0) return Refusal::shmem_ordered;
        return credit(Credit::write_queue, Refusal::write_queue_full);
    case IssueUnit::lsu: return credit(Credit::lsu_queue, Refusal::lsu_queue_full);
    case IssueUnit::uniform:
    case IssueUnit::control:
    case IssueUnit::nop:
    case IssueUnit::halt: return std::nullopt;
    case IssueUnit::none: break;
    }
    throw SimError("internal: issue was given a word that is no instruction");
}

// ---- depbar: completions are recorded as they happen (the RF write
// pipeline's RAM stage, a store's last ack, an acc.st's last tmem write, an
// MMA's operands consumed); drive_depbar applies them and the admission.
// Issue reads the counters' Q: a completion presented at t is seen at t + 1.
// depbar_after_completions is the counter's D so far (underflow and trace). ----

void Machine::record_depbar_completion(
    Wires& w, uint32_t warp, uint8_t bar, const std::string& underflow) {
    const auto before = depbar_after_completions(w, warp, bar);
    if (before == 0) throw SimError(underflow);
    w.depbar_completions.push_back({warp, bar});
    trace_->depbar(cycle_, warp, bar, before - 1);
}

uint32_t Machine::depbar_after_completions(const Wires& w, uint32_t warp, uint8_t bar) const {
    uint32_t value = depbar_.cur().at(warp).at(bar);
    for (const auto& completion : w.depbar_completions) {
        if (completion.warp == warp && completion.bar == bar) --value;
    }
    return value;
}

// (An admission's trace value is the counter's next value, written where it
// is calculated, drive_depbar: a completion recorded later in the cycle, a
// dropped entry's at issue + 1, is in it.)
void Machine::admit_depbar(Wires& w, uint32_t warp, uint8_t bar) {
    w.issue.depbar_admission = DepbarEvent{warp, bar};
}

void Machine::admit_read_depbar(Wires& w, uint32_t warp, const Instruction& inst) {
    const auto bar = inst.control.read_depbar;
    if (bar >= 6) return;
    if (w.issue.depbar_admission.has_value() && w.issue.depbar_admission->bar == bar) {
        throw SimError("internal: an instruction's read depbar is its set depbar (kernel load refuses it)");
    }
    w.issue.read_depbar_admission = DepbarEvent{warp, bar};
}

void Machine::lower_read_depbar(Wires& w, uint32_t warp, const Instruction& inst) {
    const auto bar = inst.control.read_depbar;
    if (bar >= 6) return;
    record_depbar_completion(w, warp, bar, "depbar underflow on sb" + std::to_string(bar) + " (a read depbar)");
}

// ===========================================================================
// Issue: the scheduler, after the units and the dispatch pipeline's end
// ===========================================================================

// The scheduler: from the cursor, the first warp whose next instruction can
// be accepted issues (one per cycle). The heads are the FIFOs' Q, decoded as
// they landed: a packet landing now is a candidate from the next cycle.
//
// Every warp's trace state comes from its own registers whether or not the
// scheduler reached it: halted, its own waits (probe, wait slot, stall
// count), then instruction supply (branch unresolved, FIFO empty). A warp
// past all of those is either tried (issued, or the reason it was refused)
// or, once another warp has issued this cycle, "not examined" -- never shown
// as ready, because nothing checked its depbars or resources. The FIFO-empty
// count is a separate observation over every running warp: an empty FIFO is
// not by itself a lost issue (another warp may issue, or the warp may be
// waiting anyway). Both are trace bookkeeping, not modeled hardware.
void Machine::decide_issue(Wires& w) {
    struct State {
        WarpState state = WarpState::not_examined;
        std::string detail;
    };
    const auto& front_end = front_end_.cur();
    const auto& branch = issue_branch_.cur();
    std::vector<State> states(warps_.size());
    bool issued = false;
    uint64_t fifo_empty = 0;
    for (uint32_t warp = 0; warp < warps_.size(); ++warp) {
        if (!front_end[warp].halted && front_end[warp].buffer.empty()) ++fifo_empty;
    }
    trace_->instruction_fifo_empty(fifo_empty);
    // The scan keeps going past the winner (for the trace), so it walks from
    // the cursor as it was at the start of the cycle.
    const auto cursor = scheduler_cursor_.cur();
    for (uint32_t offset = 0; offset < warps_.size(); ++offset) {
        const auto index = (cursor + offset) % static_cast<uint32_t>(warps_.size());
        auto& warp = warps_[index];
        if (front_end[index].halted) {
            states[index].state = WarpState::halted;
            continue;
        }
        if (shmem_counts_.cur()[index].probe) {
            states[index] = {WarpState::mbar_wait, "mbar probe"};
            continue;
        }
        if (warp_waiting_in_slot(warp.id)) {
            states[index] = {WarpState::mbar_wait, "mbar phase"};
            continue;
        }
        if (stall_left_.cur()[index] != 0) {
            states[index] = {WarpState::stall_count, "stall count"};
            continue;
        }
        // Nothing after a branch or a sync issues until it resolves (no
        // prediction); after an ssy the warp goes on.
        if (branch.has_value() && branch->warp == index && holds_its_warp(*branch)) {
            states[index] = {WarpState::no_instruction, "branch unresolved"};
            continue;
        }
        if (front_end[index].buffer.empty()) {
            states[index] = {WarpState::no_instruction, "instruction FIFO empty"};
            continue;
        }
        if (issued) {
            states[index] = {WarpState::not_examined, {}};
            continue;
        }
        const auto attempt = try_issue(warp, w);
        if (attempt.issued) {
            states[index] = {WarpState::issued, {}};
            issued = true;
            continue;
        }
        states[index] = {attempt.state, attempt.reason};
    }

    if (!issued) {
        const auto draining = std::all_of(front_end.begin(), front_end.end(), [](const WarpFrontEnd& warp) {
            return warp.halted;
        });
        trace_->scheduler_idle(draining);
    }
    for (uint32_t warp = 0; warp < warps_.size(); ++warp) {
        trace_->warp_state(cycle_, warp, states[warp].state, std::move(states[warp].detail));
    }
}

// One warp's issue attempt, on its buffer's head entry (the instruction's
// fields and its plan -- its path, read round and write booking -- as the
// decoder left them when its packet landed):
//   1. metadata panics; the depbar wait, on the counters' Q (a completion
//      presented at t is seen from t + 1); register ranges and
//      the sanitizer's hazard checks against the cycles the round samples
//      at (at every attempt: contention must not hide a missing
//      stall);
//   2. admission: admission_refusal, on the plan and registers alone. A
//      refusal books nothing and reads nothing;
//   3. acceptance: the read round into the issue register, the write
//      booking (both in future cycles, past the issue register),
//      the metadata share of each unit's work and, for a memory, engine or
//      mbar instruction, its entry into its unit's queue or the dispatch
//      pipeline. Operand-value panics (alignment, bounds) happen in the
//      unit, where the addresses are added.
// What an acceptance changes goes into w.issue, which its drivers apply --
// the warp's stall countdown, the SFU's occupancy and the scheduler's cursor
// (drive_scheduler), the buffer's pop and the warp's pc (step_front_end)
// included.
Machine::IssueAttempt Machine::try_issue(Warp& warp, Wires& w) {
    const auto& head = head_entry(warp);
    const Instruction inst = head.instruction;
    const auto& plan = head.plan;
    panic_if_malformed(warp, inst);
    if (const auto bar = depbar_wait(inst.control.wait_mask, depbar_.cur().at(warp.id)); bar.has_value()) {
        return {false, WarpState::depbar, "sb" + std::to_string(*bar)};
    }
    check_issue_hazards(warp, inst);
    if (const auto refusal = admission_refusal(warp, plan); refusal.has_value()) {
        return refuse(*refusal);
    }
    accept(warp, inst, plan, w);
    return {true, WarpState::issued, {}};
}

// Panics on the instruction word alone: register encodings that wrap and
// special registers read by the wrong instruction. (A write mask on an
// instruction that cannot be masked was refused at kernel load:
// check_kernel.)
void Machine::panic_if_malformed(const Warp& warp, const Instruction& inst) const {
    if ((inst.opcode == Opcode::ldg_v8 && static_cast<unsigned>(inst.dst) + 8 > 256) ||
        (inst.opcode == Opcode::stg_v8 && static_cast<unsigned>(inst.src1) + 8 > 256))
        panic(warp, inst, "global vector register encoding wraps");
    if (is_acc_transfer(inst.opcode)) {
        // (The pointer is a uniform register: check_issue_hazards checks it
        // against the URF.)
        const auto base = is_acc_load(inst.opcode) ? inst.dst : inst.src1;
        if (base + register_span(inst.opcode) > 256) panic(warp, inst, "accumulator register encoding wraps");
    }
    if (inst.opcode == Opcode::sread && inst.imm != static_cast<int32_t>(SpecialReg::lane_id) &&
        inst.imm != static_cast<int32_t>(SpecialReg::thread_id)) {
        panic(warp, inst, "sread reads LANE_ID / THREAD_ID (WARP_ID / N_WARPS are read with usread)");
    }
    if (inst.opcode == Opcode::usread && inst.imm != static_cast<int32_t>(SpecialReg::warp_id) &&
        inst.imm != static_cast<int32_t>(SpecialReg::n_warps)) {
        panic(warp, inst, "usread reads WARP_ID / N_WARPS (LANE_ID / THREAD_ID are read with sread)");
    }
}

// Register and predicate ranges (machine panics: properties of the
// instruction), then the sanitizer's hazard checks, at the cycles the read
// rounds sample and its destination lands at (src/timing.hpp). The machine looks
// at no operand value; the sanitizer's transfer-versus-matmul check looks at
// the tmem pointer the round will sample (it has global knowledge).
void Machine::check_issue_hazards(const Warp& warp, const Instruction& inst) const {
    for (const auto reg : vector_sources(inst)) {
        if (reg >= kernel_.register_count) panic(warp, inst, "source register out of range");
    }
    const auto destination = vector_destination(inst);
    if (destination.has_value() && *destination >= kernel_.register_count) {
        panic(warp, inst, "destination register out of range");
    }
    // Uniform registers: in range, then RAW (each source's sample at issue +
    // 1) and WAW (the write lands L + Wu after issue) against the uniform
    // horizons, at every attempt.
    const auto uniform_sources_read = uniform_sources(inst);
    for (const auto reg : uniform_sources_read) {
        if (reg >= config_.urf_registers) panic(warp, inst, "uniform source register out of range");
    }
    const auto uniform_dst = uniform_destination(inst);
    if (uniform_dst.has_value() && *uniform_dst >= config_.urf_registers) {
        panic(warp, inst, "uniform destination register out of range");
    }
    // Each uniform register must be written by its own round's sample (a
    // tma's and an mma.async's later rounds sample later: src/timing.hpp
    // uniform_read_rounds).
    const auto rounds = uniform_read_rounds(inst);
    for (uint32_t round = 0; round < rounds.size(); ++round) {
        sanitizer_.check_uniform_raw(site(warp, inst), rounds[round], cycle_ + issue_to_uniform_round_sample(round));
    }
    if (uniform_dst.has_value()) {
        sanitizer_.check_uniform_waw(site(warp, inst), *uniform_dst, cycle_ + issue_to_urf_visible(config_));
    }
    if (const auto predicate = predicate_destination(inst); predicate.has_value()) {
        if (*predicate >= 7) panic(warp, inst, "predicate destination is not p0..p6");
        sanitizer_.check_pred_waw(site(warp, inst), *predicate);
    }
    // A branch's predicate and any other instruction's write mask are sampled in
    // the operand round (issue_to_predicate_sample = issue_to_sample).
    if (inst.control.predicate != 7) {
        sanitizer_.check_pred_raw(site(warp, inst), inst.control.predicate, cycle_ + issue_to_predicate_sample);
    }
    check_register_hazards(warp, inst);
    if (is_acc_transfer(inst.opcode)) {
        // Racing accepted MMA work is UB (cases 01/02): no ownership hardware
        // exists, but the simulator has global knowledge and diagnoses it, at
        // every attempt (contention must not hide it) and again at dispatch
        // (an MMA command issued just before is accepted by the MMA only
        // then). The words are the ones the pointer the round samples next
        // cycle names (a region outside tmem is the dispatch PANIC's).
        const auto pointer = uniform_at_next_sample(warp.id, inst.src0);
        if (const auto words = tmem_region_words(pointer, static_cast<uint32_t>(inst.imm), config_.lanes,
                register_span(inst.opcode));
            words.has_value()) {
            sanitizer_.check_acc_transfer_vs_mma(site(warp, inst), tmem_region(pointer, inst, *words));
        }
    }
}

// A refused attempt: the warp state and reason for the trace, and its counter.
Machine::IssueAttempt Machine::refuse(Refusal refusal) {
    switch (refusal) {
    case Refusal::rf_read_inhibited:
        trace_->rf_simt_read_inhibited();
        return {false, WarpState::structural, "RF read inhibited"};
    case Refusal::urf_read_ports_held:
        trace_->urf_read_ports_held();
        return {false, WarpState::structural, "uniform read ports held"};
    case Refusal::sfu_busy:
        trace_->sfu_unit_stall();
        return {false, WarpState::structural, "SFU unit busy"};
    case Refusal::rf_write_booked: return {false, WarpState::structural, "RF write reservation"};
    case Refusal::rf_write_inhibited:
        trace_->rf_simt_inhibited();
        return {false, WarpState::structural, "RF write inhibited"};
    case Refusal::tma_queue_full: return {false, WarpState::structural, "tma command queue full"};
    case Refusal::mma_queue_full: return {false, WarpState::structural, "MMA command queue full"};
    case Refusal::acc_engine_busy: return {false, WarpState::structural, "acc transfer engine busy"};
    case Refusal::shmem_ordered:
        trace_->shmem_order_wait();
        return {false, WarpState::shmem_order, "ordered behind own access"};
    // Implementer choice: the fence's wait is the same warp state (it is
    // ordered behind the warp's own accesses), told apart by its detail; it
    // is not counted in the summary's shmem "ordered=" (section 4 refusals).
    case Refusal::shmem_fence: return {false, WarpState::shmem_order, "fence.sh: own accesses not at the RAM"};
    case Refusal::read_queue_full:
        trace_->shmem_queue_full(false);
        return {false, WarpState::shmem_queue, "read queue full"};
    case Refusal::write_queue_full:
        trace_->shmem_queue_full(true);
        return {false, WarpState::shmem_queue, "write queue full"};
    case Refusal::lsu_queue_full:
        ++trace_->lsu_counters().lsu_queue_full_cycles;
        return {false, WarpState::structural, "LSU queue full"};
    }
    throw SimError("internal unknown issue refusal");
}

// An accepted instruction leaves its FIFO (the head pops, the warp's pc
// moves to the next word: step_front_end). Fixed arithmetic books its
// bundle; a branch, ssy or sync goes into the issue register, where it
// resolves next cycle (resolve_branch); a halt stops the warp's front end now
// (its FIFO is cleared, nothing more is fetched or issued; halting inside an
// arm, with stack entries left or a partial mask, is the sanitizer's PANIC);
// fence.sh and nop are complete now; a shared-memory instruction enters its
// side's queue, a global-memory one the load / store unit's; everything else
// (the engines' instructions) enters the dispatch pipeline.
void Machine::accept(Warp& warp, const Instruction& inst, const IssuePlan& plan, Wires& w) {
    w.issue.popped = warp.id;
    w.issue.credit = credit_for(plan.unit);
    switch (plan.unit) {
    case IssueUnit::alu:
    case IssueUnit::fpu:
    case IssueUnit::sfu:
        accept_arithmetic(warp, inst, plan, w);
        return;
    case IssueUnit::uniform:
        accept_uniform(warp, inst, plan, w);
        return;
    case IssueUnit::control:
        w.issue.branch = IssuedBranch{warp.id, head_pc(warp.id), inst};
        break;
    case IssueUnit::fence:
        break; // its admission was the whole of it (own_shmem_at_ram)
    case IssueUnit::nop:
        break; // its wait mask held it until now; its stall is set below
    case IssueUnit::halt: {
        // The sanitizer's view counts the warp's own control instruction
        // resolving this cycle: an ssy does not hold its warp, so the halt
        // right behind it issues in the cycle the ssy's push resolves, one
        // cycle before the registered depth shows it.
        const auto& divergence = divergence_.cur().at(warp.id);
        uint32_t depth = divergence.depth;
        uint64_t active = divergence.active;
        if (w.branch.has_value() && w.branch->warp == warp.id) {
            if (w.branch->push.has_value()) ++depth;
            if (w.branch->pop) --depth;
            if (w.branch->mask.has_value()) active = *w.branch->mask;
        }
        sanitizer_.check_halt_converged(site(warp, inst), depth, active, full_mask());
        w.issue.halt = true;
        break;
    }
    // A tma's and an mma.async's later uniform rounds are the round
    // counter's (src/timing.hpp uniform_read_rounds); a tma reaches its queue
    // through the TMA operand read, an mma.async through the dispatch
    // pipeline like every other engine instruction.
    case IssueUnit::tma:
        if (!plan.later_uniform_rounds) throw SimError("internal: a tma whose operands fit one uniform round");
        load_urf_rounds(warp, inst, w);
        break;
    case IssueUnit::mma:
    case IssueUnit::acc_load:
    case IssueUnit::acc_store:
        if (plan.later_uniform_rounds) load_urf_rounds(warp, inst, w);
        accept_for_dispatch(warp, inst, plan, w);
        break;
    case IssueUnit::shared_read:
    case IssueUnit::shared_write:
    case IssueUnit::lsu:
        accept_queued(warp, inst, plan, w);
        break;
    case IssueUnit::none:
        throw SimError("internal: issue accepted a word that is no instruction");
    }
    w.issue.stall = inst.control.stall;
    trace_->issue(cycle_, warp.id, head_pc(warp.id), inst);
}

// Fixed-latency work books its result write; its read round goes into the
// issue register (the next cycle's samples). step_arithmetic_reads captures the
// operands at the sample and executes the instruction when they are at the
// RF output. A booking of the result's bank is what sets that bank's
// async_owed bit; the read round's bank use goes into the issue register and,
// where it takes every read port of a bank, may set that bank's read inhibit.
void Machine::accept_arithmetic(Warp& warp, const Instruction& inst, const IssuePlan& plan, Wires& w) {
    if (plan.write_reg.has_value()) {
        reserve_write(plan.write_bank, plan.write_delay, w);
        w.issue.rf_write_bank = plan.write_bank;
    }
    w.issue.rf_read_banks.assign(config_.rf_banks, 0);
    for (const auto bank : plan.read_banks) ++w.issue.rf_read_banks[bank];
    if (plan.unit == IssueUnit::sfu) {
        w.issue.sfu = true;
        trace_->sfu_accept(sfu_occupancy());
    }
    for (const auto reg : plan.reads) {
        sanitizer_.record_reg_read_until(warp.id, reg, cycle_ + issue_to_sample + 1);
    }
    if (plan.write_reg.has_value()) {
        sanitizer_.record_reg_ready(warp.id, *plan.write_reg, cycle_ + issue_to_visible(inst, config_));
    } else {
        sanitizer_.record_pred_ready(warp.id, inst.dst, cycle_ + issue_to_predicate_visible(inst, config_));
    }
    ArithmeticRead pending;
    pending.warp = warp.id;
    pending.pc = head_pc(warp.id);
    pending.active_mask = active_mask(warp.id);
    pending.instruction = inst;
    pending.reads = plan.reads;
    pending.operands.resize(plan.reads.size());
    pending.uniform_reads = plan.uniform_reads;
    pending.uniform_operands.resize(plan.uniform_reads.size());
    w.issue.arithmetic = std::move(pending);
    w.issue.stall = inst.control.stall;
    trace_->issue(cycle_, warp.id, head_pc(warp.id), inst);
}

// Uniform arithmetic, r2ur and usetp: the write enters the URF write
// pipeline's stage 0 (the issue register's cycle, next cycle), where its
// operands are sampled; it presents at the URF write port at issue + L and is
// visible from issue + L + Wu + 1 (src/timing.hpp). usetp presents its predicate
// at the predicate file then instead, in the file from issue + L + 1. r2ur's
// vector source is part of the round: it owns its bank's read port in the
// issue register's cycle like any read. No write calendar and no structural
// check: one issue per cycle is one write per pipeline stage. r2ur reads lane
// 0, so like every lane-0 operation it is UB under a partial active mask (the
// sanitizer's PANIC); uniform arithmetic reads and writes no
// lanes (it runs once in every arm that issues it). usetp writes the
// predicate file, per lane: its active lanes only.
void Machine::accept_uniform(Warp& warp, const Instruction& inst, const IssuePlan& plan, Wires& w) {
    if (inst.opcode == Opcode::r2ur) sanitizer_.check_uniform_mask(site(warp, inst), active_mask(warp.id), full_mask());
    w.issue.rf_read_banks.assign(config_.rf_banks, 0);
    for (const auto bank : plan.read_banks) ++w.issue.rf_read_banks[bank];
    for (const auto reg : plan.reads) sanitizer_.record_reg_read_until(warp.id, reg, cycle_ + issue_to_sample + 1);
    if (is_usetp(inst.opcode)) {
        sanitizer_.record_pred_ready(warp.id, inst.dst, cycle_ + issue_to_uniform_predicate_visible(config_));
    } else {
        sanitizer_.record_uniform_ready(warp.id, inst.dst, cycle_ + issue_to_urf_visible(config_));
    }
    UrfWrite write;
    write.warp = warp.id;
    write.pc = head_pc(warp.id);
    write.active_mask = active_mask(warp.id);
    write.instruction = inst;
    w.issue.urf_write = std::move(write);
    w.issue.stall = inst.control.stall;
    trace_->issue(cycle_, warp.id, head_pc(warp.id), inst);
}

// An engine instruction (mma.async, mma.commit, acc.ld, acc.st and their v8
// forms) accepted now. What depends on metadata alone happens now: the
// depbar it sets (so the warp's next instruction sees it), the sanitizer's
// records for its destination (not ready until written) and an acc.st's
// data (read-owned until each beat). It reads no vector register at issue;
// its uniform registers are sampled at issue + 1 (an mma.async's urC a cycle
// later), and it enters the dispatch pipeline; at issue_to_dispatch its unit
// takes it with them (hand_over). Admission took that unit's credit.
void Machine::accept_for_dispatch(Warp& warp, const Instruction& inst, const IssuePlan& plan, Wires& w) {
    const auto opcode = inst.opcode;
    if (sets_depbar(opcode) && inst.control.set_depbar >= 6) {
        if (is_mma_async(opcode)) {
            panic(warp, inst, opcode_name(opcode, &config_.mma_shapes) + " has no operand-consumed depbar");
        }
        panic(warp, inst, "accumulator transfer has no depbar");
    }
    if (sets_depbar(opcode)) admit_depbar(w, warp.id, inst.control.set_depbar);
    if (!plan.reads.empty()) throw SimError("internal: an engine instruction with an issue-round vector read");
    const auto not_ready = std::numeric_limits<uint64_t>::max();
    if (const auto dst = vector_destination(inst); dst.has_value()) {
        for (uint32_t j = 0; j < register_span(opcode); ++j) {
            sanitizer_.record_reg_ready(warp.id, *dst + j, not_ready); // until written
        }
    }
    // WAR protection: a source read beat by beat stays read-owned until its beat.
    if (opcode == Opcode::acc_st || opcode == Opcode::acc_st_v8) {
        for (uint32_t j = 0; j < register_span(opcode); ++j) sanitizer_.record_async_reader_add(warp.id, inst.src1 + j);
    }
    DispatchEntry entry;
    entry.warp = warp.id;
    entry.pc = head_pc(warp.id);
    entry.active_mask = active_mask(warp.id);
    entry.instruction = inst;
    // Every round's uniform registers, in read order: the issue round's
    // (plan.uniform_reads) are latched at its sample, an mma.async's urC by
    // the round counter's round 1.
    for (const auto& round : uniform_read_rounds(inst)) {
        entry.uniform_reads.insert(entry.uniform_reads.end(), round.begin(), round.end());
    }
    entry.uniform_operands.resize(entry.uniform_reads.size());
    w.issue.dispatch = std::move(entry);
}

// A shared or global memory instruction accepted now enters its queue at
// this edge (src/machine/dispatch.cpp: shared memory's read or write side's,
// the load / store unit's) with the active mask; it samples its predicate
// and uniform registers next cycle, and reads no vector register at issue
// (late_vector_reads: its collect stage, command slot or data beats read
// them later). What depends on metadata alone happens now: the depbar it
// sets and its read depbar (lowered at its last vector source read), a
// shared access's warp counts (drive_shmem_counts), the sanitizer's records
// for its destination and a probe's predicate (not ready until written) and
// its vector sources (read-owned until read). Slot w of the mbar wait slots
// is warp w's: a try_wait by a warp past the last slot is a PANIC.
void Machine::accept_queued(Warp& warp, const Instruction& inst, const IssuePlan& plan, Wires& w) {
    const auto opcode = inst.opcode;
    const auto queue = plan.unit == IssueUnit::lsu ? MemoryQueue::lsu
        : plan.unit == IssueUnit::shared_write     ? MemoryQueue::shmem_write
                                                   : MemoryQueue::shmem_read;
    if (is_mbar_try_wait(opcode) && warp.id >= config_.mbar_wait_slots) {
        panic(warp, inst, "mbar.try_wait by warp " + std::to_string(warp.id) + ": the profile has " +
            std::to_string(config_.mbar_wait_slots) + " wait slots (sync.mbar_wait_slots), one for each of the first " +
            std::to_string(config_.mbar_wait_slots) + " warps");
    }
    if (sets_depbar(opcode) && inst.control.set_depbar >= 6) {
        panic(warp, inst, queue == MemoryQueue::lsu ? "global memory operation has no depbar" : opcode_name(opcode) + " has no depbar");
    }
    if (sets_depbar(opcode)) admit_depbar(w, warp.id, inst.control.set_depbar);
    admit_read_depbar(w, warp.id, inst);
    const auto not_ready = std::numeric_limits<uint64_t>::max();
    if (const auto dst = vector_destination(inst); dst.has_value()) {
        for (uint32_t j = 0; j < register_span(opcode); ++j) sanitizer_.record_reg_ready(warp.id, *dst + j, not_ready);
    }
    if (is_mbar_predicate_writer(opcode)) sanitizer_.record_pred_ready(warp.id, inst.dst, not_ready);
    for (const auto& [reg, reader] : late_vector_reads(inst)) sanitizer_.record_async_reader_add(warp.id, reg);
    w.issue.queued = IssueDecision::QueuePush{queue, queued_entry(warp, inst, plan)};
}

// A tma or an mma.async accepted now: it loads the
// URF's round counter, whose round 0 samples next cycle (the issue
// register's) and whose later rounds one per cycle after that. A tma holds
// its TMA queue credit from now on (accept); it reads no vector register and
// sets no depbar. Issue accepted it only with the counter free
// after this cycle (its last round, if any, is sampled now:
// urf_read_ports_held).
void Machine::load_urf_rounds(Warp& warp, const Instruction& inst, Wires& w) {
    UrfReadRounds rounds;
    rounds.header.warp = warp.id;
    rounds.header.pc = head_pc(warp.id);
    rounds.header.active_mask = active_mask(warp.id);
    rounds.header.instruction = inst;
    rounds.later_rounds = static_cast<uint32_t>(uniform_read_rounds(inst).size() - 1);
    w.issue.urf_rounds = std::move(rounds);
}

// The issue register's bank use: the round of the instruction accepted now
// samples in the next cycle; nothing accepted leaves it empty.
void Machine::drive_issue_read_banks(const Wires& w) {
    for (uint32_t bank = 0; bank < config_.rf_banks; ++bank) {
        const uint8_t next = w.issue.rf_read_banks.empty() ? 0 : w.issue.rf_read_banks[bank];
        if (next != issue_read_banks_.cur()[bank]) issue_read_banks_.set_next(bank, next, "drive_issue_read_banks");
    }
}

// The scheduler's registers. The cursor moves to the warp after the one
// accepted now. The accepted warp's stall countdown loads its instruction's
// stall count less one (0 for a stall of 0); every other count goes down to
// 0. The SFU's occupancy countdown loads G - 1 at an SFU op's accept and
// otherwise goes down to 0 (machine.hpp, beside the registers).
void Machine::drive_scheduler(const Wires& w) {
    if (w.issue.popped.has_value()) {
        const uint32_t next = (*w.issue.popped + 1) % static_cast<uint32_t>(warps_.size());
        if (next != scheduler_cursor_.cur()) scheduler_cursor_.set_next(next, "drive_scheduler");
    }
    const auto& stalls = stall_left_.cur();
    for (uint32_t warp = 0; warp < stalls.size(); ++warp) {
        const uint8_t left = w.issue.popped == warp ? w.issue.stall : stalls[warp];
        const uint8_t next = left == 0 ? 0 : static_cast<uint8_t>(left - 1);
        if (next != stalls[warp]) stall_left_.set_next(warp, next, "drive_scheduler");
    }
    const uint32_t busy = w.issue.sfu ? sfu_occupancy() : sfu_busy_.cur();
    const uint32_t next = busy == 0 ? 0 : busy - 1;
    if (next != sfu_busy_.cur()) sfu_busy_.set_next(next, "drive_scheduler");
}

uint32_t Machine::credit_capacity(Credit credit) const {
    switch (credit) {
    case Credit::tma_queue: return config_.tma_queue_commands;
    case Credit::mma_queue: return config_.mma_queue_commands;
    case Credit::acc_ld:
    case Credit::acc_st: return 1;
    case Credit::read_queue: return config_.shmem_read_queue_entries;
    case Credit::write_queue: return config_.shmem_write_queue_entries;
    case Credit::lsu_queue: return config_.lsu_queue_entries;
    }
    throw SimError("internal: an unknown credit");
}

// The credits: next = Q - the entries their units free now + issue's
// accept (a credit is never taken and given back in one cycle: an entry
// enters its resource at the accept's edge).
void Machine::drive_credits(const Wires& w) {
    const auto& cur = credits_.cur();
    for (std::size_t k = 0; k < credit_kinds; ++k) {
        const auto credit = static_cast<Credit>(k);
        const uint32_t returned = w.credit_returns[k];
        const uint32_t taken = w.issue.credit == credit ? 1 : 0;
        if (returned > cur[k]) throw SimError("internal: credit " + std::to_string(k) + " given back but not held");
        const uint32_t next = cur[k] - returned + taken;
        if (next > credit_capacity(credit)) throw SimError("internal: credit " + std::to_string(k) + " past its capacity");
        if (next != cur[k]) credits_.set_next(k, static_cast<uint8_t>(next), "drive_credits");
    }
}

// depbar: next = Q - this cycle's completions + issue's admissions (the
// depbar it sets and its read depbar: two different counters). (The
// completions were checked for underflow when recorded.)
void Machine::drive_depbar(const Wires& w) {
    const std::array<const std::optional<DepbarEvent>*, 2> admissions{
        &w.issue.depbar_admission, &w.issue.read_depbar_admission};
    if (w.depbar_completions.empty() && !admissions[0]->has_value() && !admissions[1]->has_value()) return;
    // (One calculation per warp: its counters' next values together.)
    std::vector<uint8_t> touched(depbar_.count(), 0);
    for (const auto& completion : w.depbar_completions) touched.at(completion.warp) = 1;
    for (const auto* admission : admissions) {
        if (admission->has_value()) touched.at((*admission)->warp) = 1;
    }
    for (uint32_t warp = 0; warp < touched.size(); ++warp) {
        if (touched[warp] == 0) continue;
        auto next = depbar_.cur()[warp];
        for (const auto& completion : w.depbar_completions) {
            if (completion.warp == warp) --next.at(completion.bar);
        }
        for (const auto* admission : admissions) {
            if (!admission->has_value() || (*admission)->warp != warp) continue;
            auto& counter = next.at((*admission)->bar);
            if (counter >= depbar_counter_max_) {
                throw SimError("internal: warp " + std::to_string(warp) + " depbar sb" +
                    std::to_string((*admission)->bar) + " would count past " + std::to_string(depbar_counter_max_) +
                    " (the counter's width) at cycle " + std::to_string(cycle_));
            }
            ++counter;
        }
        for (const auto* admission : admissions) {
            if (admission->has_value() && (*admission)->warp == warp) {
                trace_->depbar(cycle_, warp, (*admission)->bar, next.at((*admission)->bar));
            }
        }
        depbar_.set_next(warp, next, "drive_depbar");
    }
}

// No scoreboard: a hazard is a PANIC, diagnosed by the sanitizer at the
// cycles src/timing.hpp gives (each source's sample; the destination's earliest
// landing), the same numbers qtxc schedules with. Only the destination range
// check is a property of the instruction itself.
void Machine::check_register_hazards(const Warp& warp, const Instruction& inst) const {
    std::vector<std::pair<uint8_t, uint64_t>> samples;
    for (const auto reg : vector_sources(inst)) samples.emplace_back(reg, cycle_ + issue_to_sample);
    sanitizer_.check_rf_raw(site(warp, inst), samples);
    if (const auto dst = vector_destination(inst); dst.has_value()) {
        const auto width = register_span(inst.opcode);
        if (*dst + width > kernel_.register_count) panic(warp, inst, "RF destination range out of bounds");
        sanitizer_.check_rf_waw_war(site(warp, inst), *dst, width, cycle_ + issue_to_visible_min(inst, config_));
    }
}

} // namespace nanosm

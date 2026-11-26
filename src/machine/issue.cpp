// Issue: admission, the scheduler, wait masks and depbar counters, accept, and the checks
// made when an instruction issues.
// (Machine is one class, src/machine/machine.hpp; the map of its files is at the top of
// src/machine/machine.cpp.)

#include "machine/machine.hpp"

#include <algorithm>
#include <limits>
#include <utility>

namespace nanosm {

namespace {

bool is_sfu(Opcode opcode) {
    return opcode == Opcode::ex2 || opcode == Opcode::rcp;
}

bool is_lsu_opcode(Opcode opcode) {
    return opcode == Opcode::ldg || opcode == Opcode::stg || opcode == Opcode::ldg_v8 ||
        opcode == Opcode::stg_v8 || opcode == Opcode::ldg_uni;
}

bool is_mma_command(Opcode opcode) {
    return is_mma_async(opcode) || opcode == Opcode::mma_commit;
}

bool is_acc_transfer(Opcode opcode) {
    return is_acc_load(opcode) || opcode == Opcode::acc_st || opcode == Opcode::acc_st_v8;
}

// The shared-memory instructions, which wait in a queue (src/machine/shmem.cpp):
// the read side's (lds, lds.uni, the mbar probes) and the write side's (sts,
// the barrier changes, the atomics).
bool uses_shmem(Opcode opcode) {
    return is_shared_load(opcode) || opcode == Opcode::sts || is_mbar_rmw(opcode) ||
        is_mbar_predicate_writer(opcode) || is_atomic(opcode);
}

bool shmem_write_side(Opcode opcode) {
    return opcode == Opcode::sts || is_mbar_rmw(opcode) || is_atomic(opcode);
}

// A shared-memory instruction's admission, from registers: the warp's own
// order between the two sides -- a read waits for the warp's unserviced
// writes, a write for its unserviced reads (each side keeps a warp's
// accesses in order by itself) -- then a free entry in its queue.
std::optional<Refusal> shmem_refusal(Opcode opcode, const ResourceView& view) {
    const bool write = shmem_write_side(opcode);
    if (write ? view.unserviced_read : view.unserviced_write) return Refusal::shmem_ordered;
    if (!view.shmem_queue_room) return write ? Refusal::write_queue_full : Refusal::read_queue_full;
    return std::nullopt;
}

} // namespace

// ===========================================================================
// Admission: metadata and resource state only.
// These two functions see the instruction, its plan and a ResourceView. They
// are the only places issue decides whether to accept; Machine::try_issue
// calls them before it captures any operand.
// ===========================================================================

std::optional<uint8_t> depbar_wait(const Instruction& inst, const std::array<uint32_t, 6>& depbar) {
    for (uint8_t bar = 0; bar < 6; ++bar) {
        if ((inst.control.wait_mask & (1u << bar)) != 0 && depbar[bar] != 0) return bar;
    }
    return std::nullopt;
}

// In a fixed order, which decides the reason reported when several apply.
// A set depbar needs counter room too; no counter width is declared (S5),
// and a warp can hold far fewer async operations than any width would
// count, so that never refuses.
std::optional<Refusal> structural_refusal(const Instruction& inst, const IssuePlan& plan, const ResourceView& view) {
    const auto opcode = inst.opcode;
    if (view.read_inhibited) return Refusal::rf_read_inhibited;
    // A tma's or an mma.async's later operand rounds own the URF's two read
    // ports (src/timing.hpp uniform_read_rounds): an instruction whose read round
    // has a uniform register (a tma or an mma.async included) would sample in
    // one of them.
    if (view.urf_ports_held && !plan.uniform_reads.empty()) return Refusal::urf_read_ports_held;
    // A uniform-register writer (or usetp) needs nothing else: its write port
    // is its own pipeline stage (src/timing.hpp; usetp's predicate goes to the
    // flip-flops at that stage), its reads the issue register's round.
    if (uses_urf_pipeline(opcode)) return std::nullopt;
    if (uses_arithmetic_pipeline(inst)) {
        if (is_sfu(opcode) && !view.sfu_free) return Refusal::sfu_busy;
        if (plan.write_reg.has_value() && !view.write_slot_free) return Refusal::rf_write_booked;
        if (plan.write_reg.has_value() && view.write_inhibited) return Refusal::rf_write_inhibited;
        return std::nullopt;
    }
    if (is_tma_opcode(opcode) && !view.tma_queue_room) return Refusal::tma_queue_full;
    if (is_mma_command(opcode) && !view.mma_queue_room) return Refusal::mma_queue_full;
    if (is_acc_transfer(opcode) && !view.acc_engine_free) return Refusal::acc_engine_busy;
    if (opcode == Opcode::fence_sh && !view.own_shmem_at_ram) return Refusal::shmem_fence;
    if (uses_shmem(opcode)) {
        if (const auto refusal = shmem_refusal(opcode, view); refusal.has_value()) return refusal;
    }
    if (is_lsu_opcode(opcode) && !view.lsu_queue_room) return Refusal::lsu_queue_full;
    return std::nullopt;
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
// be accepted issues (one per cycle). The heads are the FIFOs' Q: a packet
// landing now is a candidate from the next cycle.
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
        if (probe_in_flight(warp.id)) {
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

// One warp's issue attempt:
//   1. metadata panics; the depbar wait, on the counters' Q (a completion
//      presented at t is seen from t + 1); the plan (its read round and
//      write booking, from metadata and the profile); register ranges and
//      the sanitizer's hazard checks against the cycles the round samples
//      at (at every attempt, as before: contention must not hide a missing
//      stall);
//   2. admission: structural_refusal, on metadata and resource state alone.
//      A refusal books nothing and reads nothing;
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
    const auto inst = head_instruction(warp);
    sanitizer_.record_issue_attempt(warp.id);
    panic_if_malformed(warp, inst);
    if (const auto bar = depbar_wait(inst, depbar_.cur().at(warp.id)); bar.has_value()) {
        return {false, WarpState::depbar, "sb" + std::to_string(*bar)};
    }
    const auto plan = issue_plan(inst);
    check_issue_hazards(warp, inst);
    if (const auto refusal = structural_refusal(inst, plan, resource_view(warp, inst, plan, w)); refusal.has_value()) {
        return refuse(*refusal);
    }
    sanitizer_.record_issue_accepted();
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
        const auto width = inst.opcode == Opcode::acc_ld_v8 || inst.opcode == Opcode::acc_st_v8 ? 8u : 1u;
        if (static_cast<unsigned>(base) + width > 256) panic(warp, inst, "accumulator register encoding wraps");
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
    for (const auto reg : source_registers(inst)) {
        if (reg >= kernel_.register_count) panic(warp, inst, "source register out of range");
    }
    const auto destination = destination_register(inst);
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
    if (is_setp(inst.opcode) || is_usetp(inst.opcode) || is_mbar_predicate_writer(inst.opcode)) {
        if (inst.dst >= 7) panic(warp, inst, "predicate destination is not p0..p6");
        sanitizer_.check_pred_waw(site(warp, inst), inst.dst);
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
        const uint32_t width = inst.opcode == Opcode::acc_ld_v8 || inst.opcode == Opcode::acc_st_v8 ? 8 : 1;
        const auto pointer = uniform_at_next_sample(warp.id, inst.src0);
        if (const auto words = tmem_region_words(pointer, static_cast<uint32_t>(inst.imm), config_.lanes, width);
            words.has_value()) {
            sanitizer_.check_acc_transfer_vs_mma(site(warp, inst), tmem_region(pointer, inst, *words));
        }
    }
}

// What an accepted instruction books, from metadata and the profile: its one
// read round (src/timing.hpp; kernel load refused any round that does not fit
// its banks' read ports), vector and uniform registers, and a fixed result's
// write. acc.st reads only its uniform coordinates at issue (its data beat by
// beat), and the shared and global memory forms no vector register (their
// collect stage or command slot reads them, later).
IssuePlan Machine::issue_plan(const Instruction& inst) const {
    IssuePlan plan;
    auto sources = source_registers(inst);
    if (inst.opcode == Opcode::acc_st || inst.opcode == Opcode::acc_st_v8 || inst.opcode == Opcode::lds ||
        inst.opcode == Opcode::sts || is_lsu_opcode(inst.opcode)) {
        sources.clear();
    }
    plan.reads = rf_read_round(sources);
    // The issue round's uniform registers: round 0 of src/timing.hpp's plan (a
    // tma's and an mma.async's later rounds are the round counter's).
    const auto rounds = uniform_read_rounds(inst);
    if (!rounds.empty()) plan.uniform_reads = rounds.front();
    const auto destination = destination_register(inst);
    if (uses_arithmetic_pipeline(inst) && destination.has_value()) {
        plan.write_reg = destination;
        plan.write_cycle = cycle_ + issue_to_port(inst, config_);
    }
    return plan;
}

// The resource state admission may see, as this cycle leaves it (c): the RF
// write calendar and this cycle's RF read / write decisions, unit occupancy,
// queue and slot space less what the dispatch pipeline's instructions will
// take (dispatch_claims). A shared or global memory instruction's terms are
// registers only: its queue's occupancy (and a shared-memory one's, the
// warp's counts). Instruction fields
// select WHICH resource (a register number's bank, an accumulator's tmem
// bank, the queue); no register content is read.
ResourceView Machine::resource_view(
    const Warp& warp, const Instruction& inst, const IssuePlan& plan, const Wires& w) const {
    const auto claims = dispatch_claims();
    ResourceView view;
    for (const auto reg : plan.reads) {
        if (w.rf_read.banks.at(reg % config_.rf_banks).inhibit_simt) view.read_inhibited = true;
    }
    view.sfu_free = sfu_busy_.cur() == 0;
    if (plan.write_reg.has_value()) {
        view.write_slot_free = write_slot_free(*plan.write_reg, plan.write_cycle);
        view.write_inhibited = w.rf_write.banks.at(*plan.write_reg % config_.rf_banks).inhibit_simt;
    }
    // A queue slot is counted as taken from a tma's accept: the queue after
    // this cycle, the command its operand read pushes now, the commands it
    // still holds. So the push always fits.
    view.tma_queue_room = w.tma_next.commands.size() + (w.tma_pushed.has_value() ? 1 : 0) +
        tma_commands_being_read() < config_.tma_queue_commands;
    view.urf_ports_held = urf_read_ports_held();
    view.mma_queue_room = w.mma_next.slots.size() + claims.mma_commands < config_.mma_queue_commands;
    if (is_acc_transfer(inst.opcode)) {
        const bool load = is_acc_load(inst.opcode);
        view.acc_engine_free = acc_engine_available(w, load) && claims.acc[load ? 0 : 1] == 0;
    }
    const auto& counts = shmem_counts_.cur().at(warp.id);
    view.unserviced_read = counts.reads != 0;
    view.unserviced_write = counts.writes != 0;
    view.own_shmem_at_ram = counts.not_at_ram == 0;
    view.shmem_queue_room = shmem_write_side(inst.opcode)
        ? write_queue_.cur().size() < config_.shmem_write_queue_entries
        : read_queue_.cur().size() < config_.shmem_read_queue_entries;
    view.lsu_queue_room = lsu_queue_.cur().size() < config_.lsu_queue_entries;
    return view;
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
    if (uses_arithmetic_pipeline(inst)) {
        accept_arithmetic(warp, inst, plan, w);
        return;
    }
    if (uses_urf_pipeline(inst.opcode)) {
        accept_uniform(warp, inst, plan, w);
        return;
    }
    switch (inst.opcode) {
    case Opcode::bra:
    case Opcode::bra_uni:
    case Opcode::ssy:
    case Opcode::sync:
        w.issue.branch = IssuedBranch{warp.id, head_pc(warp.id), inst};
        break;
    case Opcode::fence_sh:
        break; // its admission was the whole of it (own_shmem_at_ram)
    case Opcode::nop:
        break; // its wait mask held it until now; its stall is set below
    case Opcode::halt: {
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
    default:
        // A tma's and an mma.async's later uniform rounds are the round
        // counter's (src/timing.hpp uniform_read_rounds); a tma reaches its queue
        // through the TMA operand read, an mma.async through the dispatch
        // pipeline like every other unit's instruction.
        if (uniform_read_rounds(inst).size() > 1) load_urf_rounds(warp, inst, w);
        if (uses_shmem(inst.opcode)) {
            accept_shmem(warp, inst, plan, w);
        } else if (is_lsu_opcode(inst.opcode)) {
            accept_lsu(warp, inst, plan, w);
        } else if (!is_tma_opcode(inst.opcode)) {
            accept_for_dispatch(warp, inst, plan, w);
        }
        break;
    }
    check_issue_latency(warp, inst);
    w.issue.stall = inst.control.stall;
    trace_->issue(cycle_, warp.id, head_pc(warp.id), inst);
}

// Fixed-latency work books its result write; its read round goes into the
// issue register (the next cycle's samples). step_rf_reads captures the
// operands at the sample and executes the instruction when they are at the
// RF output. A booking of the result's bank is what sets that bank's
// async_owed bit; the read round sets its banks' read_owed bits.
void Machine::accept_arithmetic(Warp& warp, const Instruction& inst, const IssuePlan& plan, Wires& w) {
    if (plan.write_reg.has_value()) {
        reserve_write(*plan.write_reg, plan.write_cycle, w);
        w.issue.rf_write_bank = *plan.write_reg % config_.rf_banks;
    }
    w.issue.rf_read_banks.assign(config_.rf_banks, 0);
    for (const auto reg : plan.reads) ++w.issue.rf_read_banks[reg % config_.rf_banks];
    if (is_sfu(inst.opcode)) {
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
    for (const auto reg : plan.reads) {
        ++w.issue.rf_read_banks[reg % config_.rf_banks];
        sanitizer_.record_reg_read_until(warp.id, reg, cycle_ + issue_to_sample + 1);
    }
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
// takes it with them (hand_over). Admission secured that unit's space.
void Machine::accept_for_dispatch(Warp& warp, const Instruction& inst, const IssuePlan& plan, Wires& w) {
    const auto opcode = inst.opcode;
    const bool sets_depbar = is_acc_transfer(opcode) || is_mma_async(opcode);
    if (sets_depbar && inst.control.set_depbar >= 6) {
        if (is_mma_async(opcode)) {
            panic(warp, inst, opcode_name(opcode, &config_.mma_shapes) + " has no operand-consumed depbar");
        }
        panic(warp, inst, "accumulator transfer has no depbar");
    }
    if (sets_depbar) admit_depbar(w, warp.id, inst.control.set_depbar);
    if (!plan.reads.empty()) throw SimError("internal: an engine instruction with an issue-round vector read");
    const auto not_ready = std::numeric_limits<uint64_t>::max();
    if (const auto dst = destination_register(inst); dst.has_value()) {
        const auto width = opcode == Opcode::acc_ld_v8 ? 8u : 1u;
        for (uint32_t j = 0; j < width; ++j) sanitizer_.record_reg_ready(warp.id, *dst + j, not_ready); // until written
    }
    // WAR protection: a source read beat by beat stays read-owned until its beat.
    if (opcode == Opcode::acc_st || opcode == Opcode::acc_st_v8) {
        const auto width = opcode == Opcode::acc_st ? 1u : 8u;
        for (uint32_t j = 0; j < width; ++j) sanitizer_.record_async_reader_add(warp.id, inst.src1 + j);
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

// A shared-memory instruction accepted now enters its side's queue at this
// edge (src/machine/shmem.cpp), with the active mask; it samples its
// predicate and uniform registers next cycle, and reads no vector register
// at issue (its collect stage reads an lds's address and an sts's address
// and data later). What depends on metadata alone happens now: the depbar it
// sets and its read depbar, its warp's counts (drive_shmem_counts), the sanitizer's records for
// its destination (not ready until written) and its vector sources
// (read-owned until the collect stage reads them). Slot w of the mbar wait
// slots is warp w's: a try_wait by a warp past the last slot is a PANIC.
void Machine::accept_shmem(Warp& warp, const Instruction& inst, const IssuePlan& plan, Wires& w) {
    const auto opcode = inst.opcode;
    if (is_mbar_try_wait(opcode) && warp.id >= config_.mbar_wait_slots) {
        panic(warp, inst, "mbar.try_wait by warp " + std::to_string(warp.id) + ": the profile has " +
            std::to_string(config_.mbar_wait_slots) + " wait slots (sync.mbar_wait_slots), one for each of the first " +
            std::to_string(config_.mbar_wait_slots) + " warps");
    }
    const bool sets_depbar = is_shared_load(opcode) || opcode == Opcode::sts || is_atomic(opcode);
    if (sets_depbar && inst.control.set_depbar >= 6) panic(warp, inst, opcode_name(opcode) + " has no depbar");
    if (sets_depbar) admit_depbar(w, warp.id, inst.control.set_depbar);
    admit_read_depbar(w, warp.id, inst); // lds, sts: lowered when the collect stage has read their registers
    const auto not_ready = std::numeric_limits<uint64_t>::max();
    if (const auto dst = destination_register(inst); dst.has_value()) sanitizer_.record_reg_ready(warp.id, *dst, not_ready);
    if (is_mbar_predicate_writer(opcode)) sanitizer_.record_pred_ready(warp.id, inst.dst, not_ready); // until the probe returns
    if (opcode == Opcode::lds || opcode == Opcode::sts) sanitizer_.record_async_reader_add(warp.id, inst.src0);
    if (opcode == Opcode::sts && inst.src1 != inst.src0) sanitizer_.record_async_reader_add(warp.id, inst.src1);
    ShmemEntry entry;
    entry.warp = warp.id;
    entry.pc = head_pc(warp.id);
    entry.lanes = active_mask(warp.id);
    entry.instruction = inst;
    entry.uniform_reads = plan.uniform_reads;
    // A try_wait's uniform operands must be at the URF output as it leaves
    // the queue (its wait slot); anything else's when its collect stage
    // starts, the cycle after it leaves.
    const auto lu = config_.urf_read_latency_cycles;
    entry.wait = plan.uniform_reads.empty() ? 0 : static_cast<uint8_t>(is_mbar_try_wait(opcode) ? lu : lu - 1);
    w.issue.shmem = std::move(entry);
    w.issue.shmem_write_side = shmem_write_side(opcode);
}

// A global-memory instruction accepted now enters the load / store unit's
// queue at this edge (src/machine/lsu.cpp), with the active mask; it samples
// its predicate (and an ldg.uni its base) next cycle, and reads no vector
// register at issue (its command slot reads its address, and a scalar
// stg's data, later; a stg.v8's data in beats). What depends on metadata
// alone happens now: the depbar it sets and its read depbar, the
// sanitizer's records for its destination (not ready until written) and
// its vector sources (read-owned until its command slot or its data beats
// read them).
void Machine::accept_lsu(Warp& warp, const Instruction& inst, const IssuePlan&, Wires& w) {
    const auto opcode = inst.opcode;
    if (inst.control.set_depbar >= 6) panic(warp, inst, "global memory operation has no depbar");
    admit_depbar(w, warp.id, inst.control.set_depbar);
    admit_read_depbar(w, warp.id, inst); // ldg, stg, ldg.v8, stg.v8: lowered at their last vector source read
    const auto not_ready = std::numeric_limits<uint64_t>::max();
    if (const auto dst = destination_register(inst); dst.has_value()) {
        const auto width = opcode == Opcode::ldg_v8 ? 8u : 1u;
        for (uint32_t j = 0; j < width; ++j) sanitizer_.record_reg_ready(warp.id, *dst + j, not_ready); // until written
    }
    if (opcode != Opcode::ldg_uni) sanitizer_.record_async_reader_add(warp.id, inst.src0);
    if (opcode == Opcode::stg && inst.src1 != inst.src0) sanitizer_.record_async_reader_add(warp.id, inst.src1);
    if (opcode == Opcode::stg_v8) {
        for (uint32_t j = 0; j < 8; ++j) sanitizer_.record_async_reader_add(warp.id, inst.src1 + j);
    }
    LsuEntry entry;
    entry.warp = warp.id;
    entry.pc = head_pc(warp.id);
    entry.lanes = active_mask(warp.id);
    entry.instruction = inst;
    // An ldg.uni's base must be at the URF output as it leaves the queue
    // (it is planned then).
    entry.wait = opcode == Opcode::ldg_uni ? static_cast<uint8_t>(config_.urf_read_latency_cycles) : 0;
    w.issue.lsu = std::move(entry);
}

// A tma or an mma.async accepted now: it loads the
// URF's round counter, whose round 0 samples next cycle (the issue
// register's) and whose later rounds one per cycle after that. A tma from now
// on counts as a slot of the TMA queue (resource_view); it reads no vector
// register and sets no depbar. Issue accepted it only with the counter free
// after this cycle (its last round, if any, is sampled now:
// urf_read_ports_held).
void Machine::load_urf_rounds(Warp& warp, const Instruction& inst, Wires& w) {
    UrfReadRounds rounds;
    rounds.header.issue_cycle = cycle_;
    rounds.header.warp = warp.id;
    rounds.header.pc = head_pc(warp.id);
    rounds.header.active_mask = active_mask(warp.id);
    rounds.header.instruction = inst;
    w.issue.urf_rounds = std::move(rounds);
}

// What the dispatch pipeline's instructions will take when they reach their
// units: the claims every later admission subtracts from the space it sees.
Machine::DispatchClaims Machine::dispatch_claims() const {
    DispatchClaims claims;
    for (const auto& stage : dispatch_pipe_.cur()) {
        if (!stage.has_value()) continue;
        const auto opcode = stage->instruction.opcode;
        if (is_mma_command(opcode)) ++claims.mma_commands;
        if (is_acc_transfer(opcode)) ++claims.acc[is_acc_load(opcode) ? 0 : 1];
    }
    return claims;
}

// The ISA tables' invariant: outside the arithmetic pipeline only
// mbar.test_wait has a fixed (one-cycle predicate) latency, and only the
// listed opcodes may have none.
void Machine::check_issue_latency(const Warp& warp, const Instruction& inst) const {
    const auto latency = fixed_latency(inst, config_.timing);
    const auto opcode = inst.opcode;
    if (is_mbar_test_wait(opcode)) {
        if (latency != 1) panic(warp, inst, "mbar.test_wait must have one-cycle predicate latency");
    } else if (latency != 0) {
        panic(warp, inst, "fixed-latency opcode bypassed the arithmetic read pipeline");
    } else if (!is_shared_load(opcode) && opcode != Opcode::sts && !is_lsu_opcode(opcode) &&
               !is_tma_opcode(opcode) && !is_acc_transfer(opcode) && !is_mma_command(opcode) &&
               opcode != Opcode::bra && opcode != Opcode::bra_uni && opcode != Opcode::ssy &&
               opcode != Opcode::sync && opcode != Opcode::fence_sh &&
               !is_mbar_rmw(opcode) && !is_mbar_try_wait(opcode) && !is_atomic(opcode) &&
               opcode != Opcode::nop && opcode != Opcode::halt) {
        panic(warp, inst, "zero-latency RF writer is unsupported");
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

// The vector registers an instruction reads. A uniform second source (the U
// bit, src/isa.hpp takes_uniform_source) is the URF's, so the others are.
std::vector<uint8_t> Machine::source_registers(const Instruction& inst) const {
    if (is_mma_async(inst.opcode)) return {};
    if (inst.src1_uniform && takes_uniform_source(inst.opcode)) {
        if (inst.opcode == Opcode::fma) return {inst.src0, inst.src2};
        return {inst.src0};
    }
    switch (inst.opcode) {
    case Opcode::add:
    case Opcode::fadd:
    case Opcode::fmul:
    case Opcode::fsub:
    case Opcode::fmax:
    case Opcode::cvt_pack:
    case Opcode::setp_eq:
    case Opcode::setp_ne:
    case Opcode::setp_lt:
    case Opcode::setp_le:
    case Opcode::setp_gt:
    case Opcode::setp_ge:
    case Opcode::setp_lt_s32:
    case Opcode::setp_le_s32:
    case Opcode::setp_gt_s32:
    case Opcode::setp_ge_s32: return {inst.src0, inst.src1};
    case Opcode::fma: return {inst.src0, inst.src1, inst.src2};
    case Opcode::and_:
    case Opcode::or_:
    case Opcode::xor_:
    case Opcode::shl:
    case Opcode::shr: return {inst.src0, inst.src1};
    case Opcode::addi:
    case Opcode::shli:
    case Opcode::shri:
    case Opcode::xori:
    case Opcode::andi:
    case Opcode::ori:
    case Opcode::lds:
    case Opcode::ldg:
    case Opcode::ldg_v8:
    case Opcode::ex2:
    case Opcode::rcp:
    case Opcode::setpi_eq:
    case Opcode::setpi_ne:
    case Opcode::setpi_lt:
    case Opcode::setpi_le:
    case Opcode::setpi_gt:
    case Opcode::setpi_ge:
    case Opcode::setpi_lt_s32:
    case Opcode::setpi_le_s32:
    case Opcode::setpi_gt_s32:
    case Opcode::setpi_ge_s32:
    case Opcode::r2ur: return {inst.src0};
    case Opcode::sts:
    case Opcode::stg: return {inst.src0, inst.src1};
    case Opcode::stg_v8: {
        std::vector<uint8_t> result{inst.src0};
        for (uint16_t j = 0; j < 8; ++j) result.push_back(static_cast<uint8_t>(inst.src1 + j));
        return result;
    }
    case Opcode::acc_st: return {inst.src1};
    case Opcode::acc_st_v8: {
        std::vector<uint8_t> result;
        for (uint16_t j = 0; j < 8; ++j) result.push_back(static_cast<uint8_t>(inst.src1 + j));
        return result;
    }
    default: return {};
    }
}

std::optional<uint8_t> Machine::destination_register(const Instruction& inst) const {
    switch (inst.opcode) {
    case Opcode::sread:
    case Opcode::mov:
    case Opcode::add:
    case Opcode::addi:
    case Opcode::and_:
    case Opcode::andi:
    case Opcode::or_:
    case Opcode::ori:
    case Opcode::xor_:
    case Opcode::shl:
    case Opcode::shr:
    case Opcode::shli:
    case Opcode::shri:
    case Opcode::xori:
    case Opcode::fadd:
    case Opcode::fmul:
    case Opcode::fsub:
    case Opcode::fmax:
    case Opcode::fma:
    case Opcode::cvt_pack:
    case Opcode::ex2:
    case Opcode::rcp:
    case Opcode::ur2r:
    case Opcode::lds:
    case Opcode::lds_uni:
    case Opcode::ldg:
    case Opcode::ldg_v8:
    case Opcode::ldg_uni: return inst.dst;
    case Opcode::acc_ld:
    case Opcode::acc_ld_v8: return inst.dst;
    case Opcode::atom_add_uni:
    case Opcode::atom_and_uni:
    case Opcode::atom_xor_uni:
    case Opcode::atom_exch_uni: return inst.dst;
    default: return std::nullopt;
    }
}

// No scoreboard: a hazard is a PANIC, diagnosed by the sanitizer at the
// cycles src/timing.hpp gives (each source's sample; the destination's earliest
// landing), the same numbers qtxc schedules with. Only the destination range
// check is a property of the instruction itself.
void Machine::check_register_hazards(const Warp& warp, const Instruction& inst) const {
    std::vector<std::pair<uint8_t, uint64_t>> samples;
    for (const auto reg : source_registers(inst)) samples.emplace_back(reg, cycle_ + issue_to_sample);
    sanitizer_.check_rf_raw(site(warp, inst), samples);
    if (const auto dst = destination_register(inst); dst.has_value()) {
        const auto width = inst.opcode == Opcode::acc_ld_v8 || inst.opcode == Opcode::ldg_v8 ? 8u : 1u;
        if (static_cast<unsigned>(*dst) + width > kernel_.register_count)
            panic(warp, inst, "RF destination range out of bounds");
        sanitizer_.check_rf_waw_war(site(warp, inst), *dst, width, cycle_ + issue_to_visible_min(inst, config_));
    }
}

} // namespace nanosm

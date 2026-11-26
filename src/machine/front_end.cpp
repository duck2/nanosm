// The front end: fetch, the instruction memory's output and its decoder, packets, redirects.
// (Machine is one class, src/machine/machine.hpp; the map of its files is at the top of
// src/machine/machine.cpp.)

#include "machine/internal.hpp"
#include "machine/machine.hpp"

namespace nanosm {

// ===========================================================================
// The front end (two-word packets; the cycles in src/timing.hpp). Every
// decision here reads registers only (tick() (i)).
// ===========================================================================

// One instruction-memory read per cycle: one packet, the two words at an
// aligned 16-byte address. A warp asks for one when it has not halted, its
// next pc is known (no branch or sync of its in the issue register) and its instruction buffer has
// room: valid packets + its read in flight < the buffer's depth (a read in
// flight owns its entry until it lands or is dropped, so two reads never
// count on one free entry, and a landing always finds room).
// Round-robin over the askers (round_robin_grant, its own cursor). Whether
// the warp could issue does not matter: prefetch goes on through stall
// counts, depbars, probes, wait slots and refused heads.
Machine::FetchGrant Machine::decide_fetch() const {
    const auto& front_end = front_end_.cur();
    const auto& read = fetch_read_.cur();
    const auto& branch = issue_branch_.cur();
    std::vector<uint8_t> has_space(front_end.size(), 0);
    for (uint32_t warp = 0; warp < front_end.size(); ++warp) {
        const auto& slot = front_end[warp];
        const std::size_t in_flight = read.has_value() && read->warp == warp ? 1 : 0;
        const bool branch_unresolved = branch.has_value() && branch->warp == warp && holds_its_warp(*branch);
        has_space[warp] = !slot.halted && !branch_unresolved &&
            slot.buffer.size() + in_flight < config_.core_instruction_buffer_packets ? 1 : 0;
    }
    const auto choice = round_robin_grant(has_space, fetch_cursor_.cur(), 1);
    FetchGrant grant;
    grant.cursor_after = choice.cursor_after;
    if (!choice.granted.empty()) {
        grant.warp = choice.granted.front();
        grant.pc = front_end[*grant.warp].fetch_pc;
    }
    return grant;
}

// The packet at the instruction memory's output now, for the read in
// flight: the memory's read port, at the address its register sampled (an
// asynchronous read of the registered address): the kernel's two words
// there, each through the decoder (predecode). A word past the kernel's last
// is no instruction (the memory holds zero there): its entry stays empty, and
// it has no effect unless it becomes a warp's next instruction, which issue
// reports (head_entry). Whether the packet enters its warp's buffer is
// step_front_end's choice.
std::optional<Machine::FetchedPacket> Machine::instruction_memory_output() {
    const auto& read = fetch_read_.cur();
    if (!read.has_value()) return std::nullopt;
    FetchedPacket packet;
    packet.warp = read->warp;
    if (read->pc % 16 != 0) throw SimError("internal: a fetch of a packet at pc " + std::to_string(read->pc));
    const std::size_t row = read->pc / 16;
    if (row < instruction_memory_.rows()) {
        const auto words = instruction_memory_.read(0, row, "fetch: the instruction memory's output", 0);
        for (std::size_t half = 0; half < 2; ++half) {
            if (2 * row + half < kernel_.words.size()) packet.words[half] = predecode(words[half]);
        }
    }
    return packet;
}

// The decoder between the instruction memory's output and the buffers: the
// word's fields, and what issue will ask of the instruction (IssuePlan), from
// the word and the profile alone -- which of issue's paths it takes, its one
// read round (src/timing.hpp; kernel load refused any round that does not fit
// its banks' read ports) of vector and uniform registers, each vector
// register's bank, and a fixed result's write. acc.st reads only its uniform
// coordinates at issue (its data beat by beat), and the shared and global
// memory forms no vector register (their collect stage or command slot reads
// them, later).
Machine::BufferedInstruction Machine::predecode(uint64_t word) const {
    const Instruction inst = decode(word);
    const auto opcode = inst.opcode;
    BufferedInstruction entry;
    entry.instruction = inst;
    auto& plan = entry.plan;
    const bool lsu = is_lsu_opcode(opcode);
    const bool shmem_write = opcode == Opcode::sts || is_mbar_rmw(opcode) || is_atomic(opcode);
    const bool shmem_read = is_shared_load(opcode) || is_mbar_predicate_writer(opcode);
    if (uses_arithmetic_pipeline(inst)) {
        plan.kind = IssueKind::arithmetic;
    } else if (uses_urf_pipeline(opcode)) {
        plan.kind = IssueKind::uniform;
    } else if (opcode == Opcode::bra || opcode == Opcode::bra_uni || opcode == Opcode::ssy || opcode == Opcode::sync) {
        plan.kind = IssueKind::control;
    } else if (opcode == Opcode::fence_sh) {
        plan.kind = IssueKind::fence;
    } else if (opcode == Opcode::nop) {
        plan.kind = IssueKind::nop;
    } else if (opcode == Opcode::halt) {
        plan.kind = IssueKind::halt;
    } else if (is_tma_opcode(opcode)) {
        plan.kind = IssueKind::tma;
    } else if (is_mma_command(opcode)) {
        plan.kind = IssueKind::mma;
    } else if (is_acc_transfer(opcode)) {
        plan.kind = is_acc_load(opcode) ? IssueKind::acc_load : IssueKind::acc_store;
    } else if (shmem_write || shmem_read) {
        plan.kind = shmem_write ? IssueKind::shmem_write : IssueKind::shmem_read;
    } else if (lsu) {
        plan.kind = IssueKind::lsu;
    } else {
        throw SimError("internal: an instruction with no path through issue (kernel load checks every word)");
    }
    plan.sfu = opcode == Opcode::ex2 || opcode == Opcode::rcp;
    auto sources = source_registers(inst);
    if (opcode == Opcode::acc_st || opcode == Opcode::acc_st_v8 || opcode == Opcode::lds || opcode == Opcode::sts || lsu) {
        sources.clear();
    }
    plan.reads = rf_read_round(sources);
    for (const auto reg : plan.reads) plan.read_banks.push_back(static_cast<uint8_t>(reg % config_.rf_banks));
    // The issue round's uniform registers: round 0 of src/timing.hpp's plan (a
    // tma's and an mma.async's later rounds are the round counter's).
    const auto rounds = uniform_read_rounds(inst);
    if (!rounds.empty()) plan.uniform_reads = rounds.front();
    plan.later_uniform_rounds = rounds.size() > 1;
    const auto destination = destination_register(inst);
    if (plan.kind == IssueKind::arithmetic && destination.has_value()) {
        plan.write_reg = destination;
        plan.write_bank = static_cast<uint8_t>(*destination % config_.rf_banks);
        plan.write_delay = issue_to_port(inst, config_);
    }
    return entry;
}

// Every warp's front end after this cycle, in one calculation per warp
// (src/timing.hpp for the cycles), from Q, this cycle's front-end wires and
// issue's acceptance:
//   - issue: issue accepted the warp's next instruction (captured into the
//     issue register at this edge); pc moves to the next word, and the head
//     packet pops if that instruction was its second half;
//   - landing: the packet at the instruction memory's output, decoded, joins
//     the buffer behind the valid packets ([A, B] with A's last word accepted and C
//     landing -> [B, C]), unless a redirect or a halt clears the buffer now, or
//     the warp halted earlier (its read was granted in the halt's cycle);
//   - halt accepted now: the buffer is cleared and the warp stops;
//   - redirect (a taken branch resolves now): the buffer is cleared, the packet
//     landing now is dropped, pc takes the target and fetch pc the target's
//     packet (the target rounded down to 16: a target in a packet's second
//     half starts there). A not-taken branch changes nothing: its buffered
//     fall-through stays;
//   - fetch granted now: the fetch pc moves to the next packet.
// No fetch is granted to a warp whose branch is in the issue register, so a
// redirect and a grant never meet in one warp; a halt and a grant may (that
// read is dropped as it lands). The credit check guarantees the buffer never
// holds more than its depth, and while the warp runs its fetch pc is always
// pc rounded down to 16 + 16 x (valid packets + its read in flight); either
// failing is an internal error. The trace counts the fetch and the fetched
// words that never issue: those a redirect or a halt clears or drops, and the
// first half of a packet fetched for a second-half target, skipped as it
// lands.
void Machine::step_front_end(Wires& w) {
    const auto& cur = front_end_.cur();
    w.front_end_next = cur;
    // Fetched words not yet issued or counted: every valid packet's two, less
    // the head's first half when pc has moved past it (issued, or skipped and
    // counted as it landed).
    const auto unissued_words = [](const WarpFrontEnd& slot) -> uint64_t {
        if (slot.buffer.empty()) return 0;
        return 2 * slot.buffer.size() - (slot.pc % 16 == 8 ? 1 : 0);
    };
    uint64_t discarded = 0;
    for (uint32_t warp = 0; warp < cur.size(); ++warp) {
        auto& slot = w.front_end_next[warp];
        const bool issued = w.issue.popped == warp;
        const bool halting = issued && w.issue.halt;
        const bool redirect = w.branch.has_value() && w.branch->warp == warp && w.branch->taken;
        const bool landing = w.fetched.has_value() && w.fetched->warp == warp;
        if (issued) {
            if (slot.pc % 16 == 8) slot.buffer.pop_head();
            slot.pc += 8;
        }
        if (halting || redirect) {
            discarded += unissued_words(slot) + (landing ? 2 : 0);
            slot.buffer.flush();
        } else if (landing && cur[warp].halted) {
            discarded += 2;
        } else if (landing) {
            // Into an empty buffer with pc at a second half: the packet was
            // fetched for a second-half branch target; its first half never
            // issues.
            if (slot.buffer.empty() && slot.pc % 16 == 8) ++discarded;
            slot.buffer.append(w.fetched->words);
        }
        if (halting) slot.halted = true;
        if (redirect) {
            slot.pc = w.branch->target;
            slot.fetch_pc = w.branch->target / 16 * 16;
        }
        if (w.fetch.warp == warp) slot.fetch_pc += 16;
        if (slot.buffer.size() > config_.core_instruction_buffer_packets) {
            throw SimError("internal: warp " + std::to_string(warp) + "'s instruction FIFO overflowed at cycle " +
                std::to_string(cycle_));
        }
        if (!slot.halted) {
            const auto packets = static_cast<uint32_t>(slot.buffer.size()) + (w.fetch.warp == warp ? 1u : 0u);
            if (slot.fetch_pc != slot.pc / 16 * 16 + 16 * packets) {
                throw SimError("internal: warp " + std::to_string(warp) + "'s front end is inconsistent at cycle " +
                    std::to_string(cycle_) + ": fetch pc " + std::to_string(slot.fetch_pc) + ", pc " +
                    std::to_string(slot.pc) + ", " + std::to_string(packets) + " packets held or in flight");
            }
        }
    }
    if (w.fetch.warp.has_value()) trace_->fetch(cycle_, *w.fetch.warp, w.fetch.pc);
    if (discarded != 0) trace_->fetches_discarded(discarded);
}

// The front ends' next state (step_front_end).
void Machine::drive_front_end(const Wires& w) {
    const auto& cur = front_end_.cur();
    for (uint32_t warp = 0; warp < cur.size(); ++warp) {
        if (w.front_end_next[warp] != cur[warp]) front_end_.set_next(warp, w.front_end_next[warp], "drive_front_end");
    }
}

// The instruction memory's read: a granted fetch's address (and tag) is
// sampled at this edge; the read at the output now leaves. The fetch cursor
// moves past the granted warp, only on a grant.
void Machine::drive_fetch(const Wires& w) {
    std::optional<FetchRead> next;
    if (w.fetch.warp.has_value()) {
        next = FetchRead{*w.fetch.warp, w.fetch.pc};
        if (w.fetch.cursor_after != fetch_cursor_.cur()) fetch_cursor_.set_next(w.fetch.cursor_after, "drive_fetch");
    }
    if (next != fetch_read_.cur()) fetch_read_.set_next(next, "drive_fetch");
}

// The instruction pc names in the head packet: pc bit 3 selects its half.
const Machine::BufferedInstruction& Machine::head_entry(const Warp& warp) const {
    const auto& slot = front_end_.cur().at(warp.id);
    if (slot.pc % 8 != 0) throw SimError("unaligned PC in warp " + std::to_string(warp.id));
    if (slot.pc / 8 >= kernel_.words.size()) {
        throw SimError(
            "warp " + std::to_string(warp.id) + " fell off kernel at pc " +
            std::to_string(slot.pc));
    }
    return slot.buffer.head()[slot.pc / 8 % 2];
}

} // namespace nanosm

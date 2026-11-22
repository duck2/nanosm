// Branch resolution (bra, bra.uni, ssy, sync) and the divergence stack.
// (Machine is one class, src/machine/machine.hpp; the map of its files is at the top of
// src/machine/machine.cpp.)

#include "machine/machine.hpp"

#include <utility>

namespace nanosm {

// The control instruction in the issue register (accepted last cycle)
// resolves now (issue_to_branch_resolve),
// from the warp's divergence registers' Q (its active mask, stack depth and
// top copy) and the predicate file:
//   - bra.uni: @PT taken, @!PT not; otherwise its predicate is read like an
//     operand through the issue register, lane 0 (a declared-uniform operand;
//     the sanitizer checks the other lanes, and the active mask -- bra.uni is
//     a uniform operation in every form).
//   - bra (plain): every active lane goes; taken.
//   - bra with a predicate (the divergent branch): all its lanes are read;
//     taken = active AND p, fall = active AND NOT p. No taken lane: not
//     taken. No falling lane: taken, the mask unchanged. Both: (taken,
//     target) is pushed and the warp falls through under fall.
//   - ssy: (the active mask, its label) is pushed.
//   - sync: the top entry is popped (from the top copy, no memory read now)
//     and switched to, whoever pushed it: its mask becomes active and the
//     warp goes to its pc. If that pc is the sync's pc + 8 (one compare),
//     the words already fetched are the right ones: it continues like a
//     not-taken branch; any other pc is a redirect like a taken one.
// A push onto a full stack and a sync on an empty one are the machine's
// PANICs. Taken: the one target adder, the instruction's pc + 8 + its
// offset (an ssy's label too).
std::optional<Machine::BranchResolution> Machine::resolve_branch() const {
    const auto& branch = issue_branch_.cur();
    if (!branch.has_value()) return std::nullopt;
    const auto& inst = branch->instruction;
    const auto& warp = warps_.at(branch->warp);
    const auto& divergence = divergence_.cur();
    const auto& state = divergence.warps.at(branch->warp);
    const IssueSite here{cycle_, branch->warp, branch->pc, inst};
    if (divergence.refill.has_value() && divergence.refill->warp == branch->warp) {
        throw SimError("internal: warp " + std::to_string(branch->warp) +
            " resolves a control instruction while its stack top is being refilled, at cycle " +
            std::to_string(cycle_));
    }
    const auto target = [&]() {
        const auto address = static_cast<int64_t>(branch->pc) + 8 + inst.imm;
        if (inst.imm % 8 != 0 || address < 0 || static_cast<uint64_t>(address / 8) >= kernel_.words.size()) {
            throw_issue_panic(here, "branch target is outside the kernel");
        }
        return static_cast<uint32_t>(address);
    };
    BranchResolution resolution;
    resolution.warp = branch->warp;
    const auto push = [&](DivergenceEntry entry) {
        if (state.depth >= config_.core_divergence_stack_entries) {
            throw_issue_panic(here, "divergence stack overflow: warp " + std::to_string(branch->warp) +
                " already holds " + std::to_string(state.depth) + " entries (core.divergence_stack_entries)");
        }
        resolution.push = entry;
        resolution.push_slot = static_cast<std::size_t>(branch->warp) * config_.core_divergence_stack_entries +
            state.depth;
    };
    switch (inst.opcode) {
    case Opcode::ssy:
        push(DivergenceEntry{target(), state.active});
        return resolution;
    case Opcode::sync:
        if (state.depth == 0) throw_issue_panic(here, "sync with an empty divergence stack");
        resolution.pop = true;
        resolution.mask = state.top.mask;
        // One address compare: an entry whose pc is the next instruction
        // continues with the words already fetched; any other is a redirect.
        if (state.top.pc != branch->pc + 8) {
            resolution.taken = true;
            resolution.target = state.top.pc;
        }
        return resolution;
    case Opcode::bra:
        if (is_divergent_branch(inst)) {
            uint64_t predicate = 0;
            const auto lanes = inst.control.predicate == 7 ? std::vector<uint32_t>(config_.lanes, 1)
                                                          : capture_predicate(warp, inst.control.predicate);
            for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
                if ((lanes[lane] != 0) != inst.control.predicate_invert) predicate |= uint64_t{1} << lane;
            }
            const uint64_t taken = state.active & predicate;
            const uint64_t fall = state.active & ~predicate;
            if (taken == 0) return resolution;
            const auto destination = target();
            if (fall == 0) {
                resolution.taken = true;
                resolution.target = destination;
                return resolution;
            }
            push(DivergenceEntry{destination, taken});
            resolution.mask = fall;
            return resolution;
        }
        resolution.taken = true;
        resolution.target = target();
        return resolution;
    default:
        break;
    }
    sanitizer_.check_uniform_mask(here, state.active, full_mask()); // bra.uni
    bool take = !inst.control.predicate_invert; // @PT taken, @!PT not
    if (inst.control.predicate != 7) {
        const auto lanes = capture_predicate(warp, inst.control.predicate);
        sanitizer_.check_warp_uniform(here, lanes, "bra.uni predicate");
        take = (lanes[0] != 0) != inst.control.predicate_invert;
    }
    resolution.taken = take;
    if (take) resolution.target = target();
    return resolution;
}

// A pop presented the new top's address last cycle (DivergenceRefill): the
// entry is at the stack memory's output now.
std::optional<std::pair<uint32_t, Machine::DivergenceEntry>> Machine::divergence_memory_output() const {
    const auto& refill = divergence_.cur().refill;
    if (!refill.has_value()) return std::nullopt;
    return std::make_pair(refill->warp, divergence_stack_.at(
        static_cast<std::size_t>(refill->warp) * config_.core_divergence_stack_entries + refill->index));
}

bool Machine::holds_its_warp(const IssuedBranch& branch) {
    return branch.instruction.opcode != Opcode::ssy;
}

// The issue register's branch: the branch accepted now, else empty.
void Machine::drive_issue_branch(const Wires& w) {
    if (!w.issue.branch.has_value() && !issue_branch_.cur().has_value()) return;
    issue_branch_.set_next(w.issue.branch, "drive_issue_branch");
}

// The warps' divergence registers, in one
// calculation: the stack memory's output now lands in its warp's top copy (a
// pop's refill); the control instruction resolving now pushes (the top copy
// takes the entry the memory takes at this edge, the depth grows), pops (the
// depth shrinks and, if an entry is left, its address is presented: the
// memory's read in flight for the next cycle) and sets its warp's active
// mask. A refill and a resolution never touch one warp in one cycle
// (resolve_branch checks it).
void Machine::drive_divergence(const Wires& w) {
    const auto& cur = divergence_.cur();
    const bool stack_effect =
        w.branch.has_value() && (w.branch->mask.has_value() || w.branch->push.has_value() || w.branch->pop);
    if (!cur.refill.has_value() && !stack_effect) return;
    auto next = cur;
    next.refill.reset();
    if (w.divergence_refilled.has_value()) next.warps.at(w.divergence_refilled->first).top = w.divergence_refilled->second;
    if (stack_effect) {
        auto& warp = next.warps.at(w.branch->warp);
        if (w.branch->push.has_value()) {
            warp.top = *w.branch->push;
            ++warp.depth;
        }
        if (w.branch->pop) {
            --warp.depth;
            if (warp.depth > 0) next.refill = DivergenceRefill{w.branch->warp, warp.depth - 1};
        }
        if (w.branch->mask.has_value()) warp.active = *w.branch->mask;
    }
    divergence_.set_next(std::move(next), "drive_divergence");
}

std::vector<uint32_t> Machine::capture_predicate(const Warp& warp, uint8_t predicate) const {
    const auto& bits = warp.predicates.at(predicate);
    return std::vector<uint32_t>(bits.begin(), bits.end());
}

} // namespace nanosm

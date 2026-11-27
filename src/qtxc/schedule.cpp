// The qtx scheduler (src/qtxc/): turns a parsed qtx program into scheduled
// assembly by filling in each instruction's control bits: the stall counts
// its fixed-latency dependencies need (FixedTimeline, distances from
// src/timing.hpp), depbar set / wait for the asynchronous ones (with all six
// live, an instruction that needs a new one waits for the oldest and reuses
// it; every lds, sts and per-lane global load or store also names one shared
// read depbar, which a later writer of their vector sources waits on; the
// shared and global stores share one completion depbar and the acc.st's
// another, while one is free), a wait on a launch flag for an instruction that names a section of
// the launch block (flag_pending below), a full drain at
// every branch and sync, a loop body scheduled as a jump enters it (LoopEntry), and
// qtx's wait as a nop waiting for every live depbar (no drain).
// schedule_fixed_stalls is the same stall pass for hand assembly. It does not parse (parse.cpp), write the text (assemble.cpp) or
// encode.

#include "qtxc/internal.hpp"
#include "timing.hpp"

#include <algorithm>
#include <array>
#include <optional>

namespace nanosm {
namespace {

// The fixed-latency part of the virtual issue timeline (vtime: a warp's issue
// cycles, relative). Every distance comes from src/timing.hpp, the helper the
// simulator and the sanitizer use:
//   reg_visible: the vtime from which an RF sample sees a fixed result (async
//     results are covered by depbars instead);
//   read_until: the exclusive end of the fixed reads' samples;
//   predicate_ready: the vtime from which a predicate is in the file (a
//     conditional branch, and a masked instruction for its write mask, samples it
//     issue_to_predicate_sample after its issue; a predicate writer may issue
//     from it). A setp's comes from the
//     arithmetic pipeline, a usetp's from the URF write pipeline
//     (issue_to_uniform_predicate_visible);
//   ureg_visible: the vtime from which a URF sample sees a uniform register
//     (every uniform-register writer is fixed-latency: issue_to_urf_visible).
// A consumer is placed so its sample of each source is at or after that
// source's visibility; a writer so its earliest landing comes after the older
// write's (WAW) and at or after the older reads' samples (WAR). A uniform
// register needs no WAR rule: it is read at issue + 1 and written L + Wu
// later than its writer's issue.
struct FixedTimeline {
    std::array<int64_t, 256> reg_visible{};
    std::array<int64_t, 256> read_until{};
    std::array<int64_t, 7> predicate_ready{};
    std::array<int64_t, 256> ureg_visible{};

    // The earliest vtime `inst` may issue at, given the instructions before it.
    // Every source is sampled (or, read later beat by beat, read-owned) from
    // issue + issue_to_sample.
    [[nodiscard]] int64_t required(const Instruction& inst, const DeviceProfile& profile, int64_t vtime) const {
        int64_t at = vtime;
        for (const auto source : vector_sources(inst)) {
            at = std::max(at, reg_visible[source] - int64_t{issue_to_sample}); // RAW
        }
        if (const auto dst = vector_destination(inst); dst.has_value()) {
            const int64_t earliest = issue_to_visible_min(inst, profile);
            for (unsigned j = 0; j < register_span(inst.opcode); ++j) {
                at = std::max(at, reg_visible[*dst + j] - earliest + 1); // WAW: land after the older write
                at = std::max(at, read_until[*dst + j] - earliest);      // WAR: land at or after older samples
            }
        }
        // A branch's predicate and any other instruction's write mask are
        // sampled at the same point, the operand round. A masked
        // instruction's write is booked as a full write (record):
        // instructions issue in the order written, so the lanes it leaves
        // alone are the older write's, which WAW above has landed first.
        if (inst.control.predicate != 7) {
            at = std::max(at, predicate_ready[inst.control.predicate] - int64_t{issue_to_predicate_sample});
        }
        if (const auto dst = predicate_destination(inst); dst.has_value()) at = std::max(at, predicate_ready[*dst]);
        for (const auto source : uniform_sources(inst)) {
            at = std::max(at, ureg_visible[source] - int64_t{issue_to_urf_sample}); // uniform RAW
        }
        if (const auto dst = uniform_destination(inst); dst.has_value()) {
            // Uniform WAW: land after the older write (every writer has the same L).
            at = std::max(at, ureg_visible[*dst] - int64_t{issue_to_urf_visible(profile)} + 1);
        }
        return at;
    }

    // `inst` issues at vtime: its one read round and fixed results.
    void record(const Instruction& inst, const DeviceProfile& profile, int64_t vtime) {
        for (const auto reg : rf_read_round(issue_sources(inst))) {
            read_until[reg] = std::max<int64_t>(read_until[reg], vtime + issue_to_sample + 1);
        }
        if (uses_arithmetic_pipeline(inst)) {
            if (const auto dst = vector_destination(inst); dst.has_value()) {
                reg_visible[*dst] = vtime + issue_to_visible(inst, profile);
            }
            if (const auto dst = predicate_destination(inst); dst.has_value()) {
                predicate_ready[*dst] = vtime + issue_to_predicate_visible(inst, profile);
            }
        } else if (const auto dst = predicate_destination(inst);
                   dst.has_value() && fixed_latency(inst, profile.timing) != 0) {
            // mbar.test_wait: its warp issues nothing until the probe is back
            // and its predicate written, so its one-cycle latency always holds.
            predicate_ready[*dst] = vtime + fixed_latency(inst, profile.timing);
        }
        if (is_usetp(inst.opcode)) predicate_ready[inst.dst] = vtime + issue_to_uniform_predicate_visible(profile);
        if (const auto dst = uniform_destination(inst); dst.has_value()) {
            ureg_visible[*dst] = vtime + issue_to_urf_visible(profile);
        }
    }

    // The earliest vtime any next instruction may issue at once every fixed
    // result is visible, every older write landed after, every older sample
    // passed, every predicate written: a branch's block boundary.
    [[nodiscard]] int64_t drained(int64_t vtime, const DeviceProfile& profile) const {
        int64_t all_ready = vtime + 1;
        const int64_t earliest = earliest_rf_visible(profile);
        for (const auto visible : reg_visible) all_ready = std::max<int64_t>(all_ready, visible - issue_register_stages);
        for (const auto end : read_until) all_ready = std::max(all_ready, end - earliest);
        for (const auto ready : predicate_ready) all_ready = std::max(all_ready, ready);
        for (const auto visible : ureg_visible) {
            all_ready = std::max<int64_t>(all_ready, visible - int64_t{issue_to_urf_sample});
        }
        return all_ready;
    }

    void reset() {
        reg_visible.fill(0);
        read_until.fill(0);
        predicate_ready.fill(0);
        ureg_visible.fill(0);
    }
};

// Loop heads: the instructions a backward branch (one at or after its
// target) jumps to. A stall is a constant in the instruction, so a stall
// inside a loop is paid on every iteration, also when it is owed only by the
// code before the loop (an ex2 just before the head, say). So both passes
// schedule a loop body from a clean timeline -- the state every jump to the
// head arrives with, since every branch drains before it jumps -- and pay
// what the fall-through edge brings once, on the instruction before the head
// (LoopEntry). A divergent bra (@p bra) is a branch like any: a backward one
// makes a loop head. A sync's implicit edge -- to a divergent bra's target or
// an ssy's label, whichever entry it pops -- makes none: every jump into a
// label drains first, and a label that is not a loop head is scheduled from
// the fall-through's timeline, which is never too short (only, in a loop,
// slower).
std::vector<bool> loop_heads(const std::vector<Instruction>& instructions) {
    std::vector<bool> heads(instructions.size(), false);
    for (std::size_t index = 0; index < instructions.size(); ++index) {
        const auto& inst = instructions[index];
        if (!is_branch(inst.opcode)) continue;
        const auto target = static_cast<int64_t>(index) + 1 + inst.imm / 8;
        if (target >= 0 && target <= static_cast<int64_t>(index)) heads[static_cast<std::size_t>(target)] = true;
    }
    return heads;
}

// The fall-through edge into a loop head: the timeline the code before it
// left (`before_loop`, in that code's vtime) and the instruction before the
// head. The loop body's first stretch -- from the head up to its first
// branch, the next loop head or a halt, where the timeline is reset or
// nothing runs after -- is scheduled clean, in vtime from the head (0). Then
// the instruction before the head stalls `extra` cycles more than it does
// now: the fewest that make every instruction of the stretch legal against
// before_loop (required) and have before_loop drained by the time the
// stretch ends. Every path into the head is covered: a jump arrives clean,
// and the fall-through's leftovers are behind the stretch by its end, so
// what follows needs no more than the clean schedule gives it. (A register
// the body rewrites is still checked against before_loop's older write:
// an over-estimate, never too short.)
struct LoopEntry {
    std::size_t before = 0;     // the instruction before the head
    int64_t before_vtime = 0;   // its vtime and the head's, in before_loop's vtime
    int64_t head_vtime = 0;
    FixedTimeline before_loop;
    int64_t extra = 0;

    // A stretch instruction at body vtime `vtime`.
    void check(const Instruction& inst, const DeviceProfile& profile, int64_t vtime) {
        const auto at = head_vtime + vtime;
        extra = std::max(extra, before_loop.required(inst, profile, at) - at);
    }
    // The stretch ends; the next instruction would issue at body vtime `next`.
    void check_end(const DeviceProfile& profile, int64_t next) {
        const auto at = head_vtime + next;
        extra = std::max(extra, before_loop.drained(at - 1, profile) - at);
    }
    // The stall the instruction before the head needs, if more than it has.
    [[nodiscard]] std::optional<int64_t> stall() const {
        if (extra == 0) return std::nullopt;
        return head_vtime + extra - before_vtime;
    }
};

} // namespace

AssemblyProgram compile_qtx(std::string_view source, const DeviceProfile& profile) {
    validate_device_profile(profile);
    auto program = parse_program(source, false, profile);
    // qtx's registers are names; from here on they are numbers (registers.cpp).
    allocate_registers(program, profile);
    program.register_count = register_count(program);
    program.profile_name = profile.name;
    program.profile_fingerprint = device_profile_fingerprint(profile);
    std::array<bool, 6> live{};
    // Launch flags (docs/qtx.md "Launch"): a depbar the launch counted once in
    // every warp before the first instruction, decremented when the launch
    // copy has delivered its part of the launch block. flag_pending[b]: on
    // the path to this instruction nothing has waited on flag b yet, so its
    // count may still be there. An instruction that names a section waits on
    // the section's flag while it is pending; that wait proves it gone for
    // the straight-line code after. Any label may be entered from a path
    // that has not waited, and so may the instruction after a branch or a
    // sync: there every flag is pending again (a wait on a counter at zero
    // costs nothing). Flags are not the warp's own work: a branch, a halt or
    // qtx's wait do not wait for them, and the code that names no section
    // never does.
    std::array<bool, 6> launch_flag{};
    for (const auto& flag : program.flags) launch_flag[flag.bar] = true;
    std::array<bool, 6> flag_pending = launch_flag;
    std::vector<bool> labelled(program.instructions.size(), false);
    for (const auto& label : program.labels) {
        if (label.instruction_index < labelled.size()) labelled[label.instruction_index] = true;
    }
    std::array<std::optional<uint8_t>, 256> pending_register{};
    std::array<uint8_t, 256> pending_async_read_mask{};
    // tmem order between one warp's transfers (the
    // acc.ld and acc.st engines run side by side, nothing orders them): the
    // depbars of the warp's acc.st and acc.ld still in flight. A transfer
    // after an acc.st waits for it (RAW, WAW); an acc.st after an acc.ld
    // waits for that (WAR). One class for all of tmem: the region is a
    // pointer in a uniform register, so two transfers' words cannot be told
    // apart here.
    uint8_t acc_store_mask = 0;
    uint8_t acc_load_mask = 0;
    // The stores' completion depbars: one shared by the warp's sts, stg and
    // stg.v8 (store_barrier), one shared by its acc.st and acc.st.v8
    // (acc_store_barrier). Two, so that a writer of an acc.st's source
    // registers, which waits on the acc.st's counter, does not also wait for
    // an unrelated global or shared store (completion_barrier).
    std::optional<uint8_t> store_barrier;
    std::optional<uint8_t> acc_store_barrier;
    std::optional<uint8_t> mma_barrier;
    // The read depbar every lds, sts, ldg, stg, ldg.v8 and stg.v8 names (one
    // counter for all of them, as those stores share one completion counter):
    // lowered when each one has read its vector registers (a shared-memory
    // collect stage, the LSU's command slot, a stg.v8's last data beat). A
    // later writer of one of their vector sources waits on it, so it waits
    // for the reads of every one still queued.
    std::optional<uint8_t> read_barrier;

    // Virtual issue timeline. Consecutive same-warp issues are at least max(stall, 1)
    // apart, and real gaps only grow (scheduler contention, depbar waits, structural
    // stalls), so any spacing proven here holds on the machine. The fixed-latency
    // distances are FixedTimeline's (src/timing.hpp). A stall is paid only when a
    // consumer sits closer than that, and it lands on the instruction just before
    // the consumer.
    FixedTimeline timeline;
    int64_t vtime = 0;
    int64_t previous_vtime = 0;
    Instruction* previous = nullptr;

    // A wait on the depbars in `mask`: once it is satisfied their work is
    // done, so they are free and nothing pending on them needs a wait again.
    const auto release = [&](uint8_t mask) {
        for (auto& pending : pending_register) {
            if (pending.has_value() && (mask & (1u << *pending)) != 0) pending.reset();
        }
        for (auto& pending_mask : pending_async_read_mask) pending_mask &= static_cast<uint8_t>(~mask);
        acc_store_mask &= static_cast<uint8_t>(~mask);
        acc_load_mask &= static_cast<uint8_t>(~mask);
        for (uint8_t bar = 0; bar < 6; ++bar) {
            if ((mask & (1u << bar)) != 0) {
                live[bar] = false;
                flag_pending[bar] = false;
            }
        }
        if (store_barrier.has_value() && (mask & (1u << *store_barrier)) != 0) store_barrier.reset();
        if (acc_store_barrier.has_value() && (mask & (1u << *acc_store_barrier)) != 0) acc_store_barrier.reset();
        if (mma_barrier.has_value() && (mask & (1u << *mma_barrier)) != 0) mma_barrier.reset();
        if (read_barrier.has_value() && (mask & (1u << *read_barrier)) != 0) read_barrier.reset();
    };

    // The depbar `inst` sets: the lowest free one that is not a pending
    // launch flag; else the lowest free one that is (the counter then holds
    // the launch's count beside this instruction's: whoever waits on it
    // waits for both, which is only slower); with all six live, the oldest
    // live one (the earliest allocated): `inst` also waits for it, which
    // frees it, and then sets it. Never `exclude`: the counter `inst`'s other
    // field names (an lds's or sts's set and read depbars differ).
    std::array<uint64_t, 6> allocated_at{};
    uint64_t allocations = 0;
    // allocate's first choice alone: the lowest free depbar that is not a
    // pending launch flag, now live; none if there is no such one.
    const auto allocate_free = [&](std::optional<uint8_t> exclude = std::nullopt) -> std::optional<uint8_t> {
        for (uint8_t bar = 0; bar < 6; ++bar) {
            if (!live[bar] && !flag_pending[bar] && bar != exclude) {
                live[bar] = true;
                allocated_at[bar] = ++allocations;
                return bar;
            }
        }
        return std::nullopt;
    };
    const auto allocate = [&](Instruction& inst, std::optional<uint8_t> exclude = std::nullopt) -> uint8_t {
        if (const auto free = allocate_free(exclude); free.has_value()) return *free;
        std::optional<uint8_t> chosen;
        for (uint8_t bar = 0; bar < 6 && !chosen.has_value(); ++bar) {
            if (!live[bar] && bar != exclude) chosen = bar;
        }
        if (!chosen.has_value()) {
            for (uint8_t bar = 0; bar < 6; ++bar) {
                if (bar == exclude) continue;
                if (!chosen.has_value() || allocated_at[bar] < allocated_at[*chosen]) chosen = bar;
            }
            const auto mask = static_cast<uint8_t>(1u << *chosen);
            inst.control.wait_mask |= mask;
            release(mask);
        }
        live[*chosen] = true;
        allocated_at[*chosen] = ++allocations;
        return *chosen;
    };

    // The completion depbar a store sets: its kind's (`own`: store_barrier
    // for sts / stg / stg.v8, acc_store_barrier for acc.st / acc.st.v8) while
    // that is live. Else, with the other kind's live, a depbar of its own if
    // one is free and not a pending launch flag, and with none the other
    // kind's (whoever waits on it then waits for both stores, which is only
    // slower). Else allocate's.
    const auto completion_barrier = [&](Instruction& inst, std::optional<uint8_t>& own,
                                        const std::optional<uint8_t>& other) -> uint8_t {
        if (!own.has_value()) {
            if (!other.has_value()) own = allocate(inst);
            else if (const auto free = allocate_free(); free.has_value()) own = free;
            else own = other;
        }
        return *own;
    };

    // A loop head's fall-through edge, until the loop body's first stretch
    // ends (LoopEntry). Depbars are not part of it: the fall-through's view of
    // them carries into the loop as before, a sound over-estimate that costs
    // nothing on later iterations (a wait on a counter at zero is free).
    std::optional<LoopEntry> entry;
    const auto settle_entry = [&]() {
        if (const auto stall = entry->stall(); stall.has_value()) {
            auto& before = program.instructions[entry->before];
            if (*stall > static_cast<int64_t>(encoding::stall_mask)) {
                throw SimError(
                    "line " + std::to_string(before.line) +
                    ": the stall before a loop head exceeds the five-bit field");
            }
            before.control.stall = static_cast<uint8_t>(*stall);
        }
        entry.reset();
    };

    const auto heads = loop_heads(program.instructions);
    for (std::size_t index = 0; index < program.instructions.size(); ++index) {
        auto& inst = program.instructions[index];
        if (heads[index] && previous != nullptr) {
            // `previous` is the instruction just before the head. (After a
            // branch the timeline is clean already.)
            if (entry.has_value()) {
                entry->check_end(profile, vtime);
                settle_entry();
            }
            if (previous->opcode != Opcode::halt) {
                entry = LoopEntry{index - 1, previous_vtime, vtime, timeline};
            }
            timeline.reset();
            previous = nullptr;
            previous_vtime = 0;
            vtime = 0;
        }
        // A label: some path may arrive here without having waited on a flag.
        if (labelled[index]) flag_pending = launch_flag;
        check_instruction(inst, profile, "line " + std::to_string(inst.line));
        uint8_t wait_mask = 0;
        for (const auto source : vector_sources(inst)) {
            if (pending_register[source].has_value()) {
                wait_mask |= static_cast<uint8_t>(1u << *pending_register[source]);
            }
        }
        if (const auto dst = vector_destination(inst); dst.has_value()) {
            for (unsigned j = 0; j < register_span(inst.opcode); ++j) {
                if (pending_register[*dst + j].has_value())
                    wait_mask |= static_cast<uint8_t>(1u << *pending_register[*dst + j]);
                wait_mask |= pending_async_read_mask[*dst + j];
            }
        }
        if (inst.opcode == Opcode::acc_ld || inst.opcode == Opcode::acc_ld_v8 ||
            inst.opcode == Opcode::acc_st || inst.opcode == Opcode::acc_st_v8) {
            const bool load = inst.opcode == Opcode::acc_ld || inst.opcode == Opcode::acc_ld_v8;
            wait_mask |= acc_store_mask;
            if (!load) wait_mask |= acc_load_mask;
        }
        if (inst.opcode == Opcode::mbar_arrive) {
            // Pinned worker publication sequence: an ordinary arrival following
            // async stores is not allowed to publish until their counted
            // completion depbars (both kinds') reach zero.
            if (store_barrier.has_value()) wait_mask |= static_cast<uint8_t>(1u << *store_barrier);
            if (acc_store_barrier.has_value()) wait_mask |= static_cast<uint8_t>(1u << *acc_store_barrier);
        }
        // halt, every branch, sync and qtx's wait (parsed as a nop, the only
        // nop a qtx program has) wait for every live depbar. Unlike a branch,
        // the nop neither drains the fixed-latency timeline nor resets it
        // (below).
        if (inst.opcode == Opcode::halt || inst.opcode == Opcode::nop || transfers_control(inst.opcode)) {
            for (uint8_t bar = 0; bar < 6; ++bar) {
                if (live[bar]) wait_mask |= static_cast<uint8_t>(1u << bar);
            }
        }
        // The sections this instruction names: their launch flags, where
        // still pending on this path.
        for (uint8_t bar = 0; bar < 6; ++bar) {
            if ((program.named_flags[index] & (1u << bar)) != 0 && flag_pending[bar]) {
                wait_mask |= static_cast<uint8_t>(1u << bar);
            }
        }
        inst.control.wait_mask = wait_mask;

        if (wait_mask != 0) release(wait_mask);

        const auto required = timeline.required(inst, profile, vtime);
        if (required > vtime) {
            if (previous == nullptr) {
                throw SimError("internal: first instruction cannot need a stall");
            }
            const auto stall = required - previous_vtime;
            if (stall > static_cast<int64_t>(encoding::stall_mask)) {
                throw SimError(
                    "line " + std::to_string(inst.line) +
                    ": required stall exceeds the five-bit field (nop insertion not implemented)");
            }
            previous->control.stall = static_cast<uint8_t>(stall);
            vtime = required;
        }
        inst.control.stall = 0;
        if (entry.has_value()) {
            entry->check(inst, profile, vtime);
            if (inst.opcode == Opcode::halt) settle_entry();
        }
        timeline.record(inst, profile, vtime);
        if (inst.opcode == Opcode::lds || inst.opcode == Opcode::lds_uni || inst.opcode == Opcode::ldg ||
            inst.opcode == Opcode::ldg_v8 || inst.opcode == Opcode::ldg_uni || is_atomic(inst.opcode)) {
            const auto bar = allocate(inst);
            inst.control.set_depbar = bar;
            for (unsigned j = 0; j < register_span(inst.opcode); ++j) pending_register[inst.dst + j] = bar;
            // An lds's or ldg's address is read after issue (the collect
            // stage, the LSU's command slot): it names the read depbar, and
            // a later writer of the address waits on that.
            if (inst.opcode == Opcode::lds || inst.opcode == Opcode::ldg || inst.opcode == Opcode::ldg_v8) {
                if (!read_barrier.has_value()) read_barrier = allocate(inst, bar);
                inst.control.read_depbar = *read_barrier;
                pending_async_read_mask[inst.src0] |= static_cast<uint8_t>(1u << *read_barrier);
            }
        } else if (inst.opcode == Opcode::sts || inst.opcode == Opcode::stg || inst.opcode == Opcode::stg_v8) {
            inst.control.set_depbar = completion_barrier(inst, store_barrier, acc_store_barrier);
            // A store's address and data are read after issue (an sts's and
            // a stg's by their collect stage or command slot, a stg.v8's data
            // in beats, its read depbar falling at the last): it names the
            // read depbar, and a later writer of any of them waits on that.
            if (!read_barrier.has_value()) read_barrier = allocate(inst, *store_barrier);
            inst.control.read_depbar = *read_barrier;
            const auto bit = static_cast<uint8_t>(1u << *read_barrier);
            pending_async_read_mask[inst.src0] |= bit;
            for (unsigned j = 0; j < register_span(inst.opcode); ++j) pending_async_read_mask[inst.src1 + j] |= bit;
        } else if (inst.opcode == Opcode::acc_ld || inst.opcode == Opcode::acc_ld_v8) {
            const auto bar = allocate(inst);
            inst.control.set_depbar = bar;
            for (unsigned j = 0; j < register_span(inst.opcode); ++j) pending_register[inst.dst + j] = bar;
            acc_load_mask |= static_cast<uint8_t>(1u << bar);
        } else if (inst.opcode == Opcode::acc_st || inst.opcode == Opcode::acc_st_v8) {
            // Its sources are read in beats after issue and it has no read
            // depbar: a later writer of them waits on its completion depbar.
            const auto bar = completion_barrier(inst, acc_store_barrier, store_barrier);
            inst.control.set_depbar = bar;
            const auto bit = static_cast<uint8_t>(1u << bar);
            acc_store_mask |= bit;
            for (unsigned j = 0; j < register_span(inst.opcode); ++j) pending_async_read_mask[inst.src1 + j] |= bit;
        } else if (is_mma_async(inst.opcode)) {
            if (!mma_barrier.has_value()) mma_barrier = allocate(inst);
            inst.control.set_depbar = *mma_barrier;
        }

        if (transfers_control(inst.opcode)) {
            const auto all_ready = timeline.drained(vtime, profile);
            if (all_ready > vtime + 1) {
                const auto stall = all_ready - vtime;
                if (stall > static_cast<int64_t>(encoding::stall_mask)) {
                    throw SimError(
                        "line " + std::to_string(inst.line) +
                        ": control-flow drain exceeds the five-bit stall field");
                }
                inst.control.stall = static_cast<uint8_t>(stall);
            }
            if (entry.has_value()) {
                entry->check_end(profile, vtime + std::max<int64_t>(inst.control.stall, 1));
                settle_entry();
            }

            // Every branch and sync is a conservative block boundary for v0: its
            // wait mask drains async work, and its stall covers outstanding
            // fixed-latency writes. Every way on -- the target, the
            // fall-through, whatever pc a sync pops --
            // therefore begins with clean timing state.
            live.fill(false);
            flag_pending = launch_flag;
            pending_register.fill(std::nullopt);
            pending_async_read_mask.fill(0);
            acc_store_mask = 0;
            acc_load_mask = 0;
            store_barrier.reset();
            acc_store_barrier.reset();
            mma_barrier.reset();
            read_barrier.reset();
            timeline.reset();
            previous = nullptr;
            previous_vtime = 0;
            vtime = 0;
            continue;
        }

        previous = &inst;
        previous_vtime = vtime;
        vtime += 1;
    }
    if (entry.has_value()) settle_entry(); // the kernel ends inside the stretch
    return program;
}

AssemblyProgram compile_qtx_file(const std::string& path, const DeviceProfile& profile) {
    return compile_qtx(read_file(path), profile);
}

// Hand assembly's fixed-latency stalls: each instruction's stall becomes the
// larger of its own and what the fixed-latency register and predicate
// dependencies of the instructions after it need under `profile` -- the
// distances qtxc schedules with (FixedTimeline, src/timing.hpp) -- so the result
// never trips the sanitizer's RAW / WAW / WAR checks on a fixed result.
// Depbars, waits and every other field stay the author's. A branch drains
// like qtxc's block boundary, and the code after it starts clean. A loop body
// starts clean too, its fall-through edge paid before the head (LoopEntry).
Kernel schedule_fixed_stalls(Kernel kernel, const DeviceProfile& profile) {
    std::vector<Instruction> instructions;
    for (const auto word : kernel.words) instructions.push_back(decode(word));
    FixedTimeline timeline;
    int64_t vtime = 0;
    int64_t previous_vtime = 0;
    std::optional<std::size_t> previous;
    const auto set_stall = [](Instruction& inst, int64_t stall) {
        if (stall > static_cast<int64_t>(encoding::stall_mask)) {
            throw SimError("a fixed-latency dependency needs a stall beyond the five-bit field");
        }
        inst.control.stall = static_cast<uint8_t>(stall);
    };
    // A loop head's fall-through edge, as in compile_qtx (LoopEntry).
    std::optional<LoopEntry> entry;
    const auto settle_entry = [&]() {
        if (const auto stall = entry->stall(); stall.has_value()) set_stall(instructions[entry->before], *stall);
        entry.reset();
    };
    const auto heads = loop_heads(instructions);
    for (std::size_t index = 0; index < instructions.size(); ++index) {
        auto& inst = instructions[index];
        if (heads[index] && previous.has_value()) {
            if (entry.has_value()) {
                entry->check_end(profile, vtime);
                settle_entry();
            }
            if (instructions[*previous].opcode != Opcode::halt) {
                entry = LoopEntry{*previous, previous_vtime, vtime, timeline};
            }
            timeline.reset();
            previous.reset();
            previous_vtime = 0;
            vtime = 0;
        }
        const auto required = timeline.required(inst, profile, vtime);
        if (required > vtime) {
            if (!previous.has_value()) throw SimError("internal: first instruction cannot need a stall");
            set_stall(instructions[*previous], required - previous_vtime);
            vtime = required;
        }
        if (entry.has_value()) {
            entry->check(inst, profile, vtime);
            if (inst.opcode == Opcode::halt) settle_entry();
        }
        timeline.record(inst, profile, vtime);
        if (transfers_control(inst.opcode)) {
            const auto all_ready = timeline.drained(vtime, profile);
            if (all_ready - vtime > std::max<int64_t>(inst.control.stall, 1)) set_stall(inst, all_ready - vtime);
            if (entry.has_value()) {
                entry->check_end(profile, vtime + std::max<int64_t>(inst.control.stall, 1));
                settle_entry();
            }
            timeline.reset();
            previous.reset();
            previous_vtime = 0;
            vtime = 0;
            continue;
        }
        previous = index;
        previous_vtime = vtime;
        vtime += std::max<int64_t>(inst.control.stall, 1);
    }
    if (entry.has_value()) settle_entry();
    for (std::size_t index = 0; index < instructions.size(); ++index) kernel.words[index] = encode(instructions[index]);
    return kernel;
}

} // namespace nanosm

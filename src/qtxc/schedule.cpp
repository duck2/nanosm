// The qtx scheduler (src/qtxc/): turns a parsed qtx program into scheduled
// assembly by filling in each instruction's control bits: the stall counts
// its fixed-latency dependencies need (FixedTimeline, distances from
// src/rf.hpp), depbar set / wait for the asynchronous ones, and a full drain
// at every branch. schedule_fixed_stalls is the same stall pass for hand
// assembly. It does not parse (parse.cpp), write the text (assemble.cpp) or
// encode.

#include "qtxc/internal.hpp"
#include "rf.hpp"

#include <algorithm>
#include <array>
#include <optional>

namespace nanosm {
namespace {

// The fixed-latency part of the virtual issue timeline (vtime: a warp's issue
// cycles, relative). Every distance comes from src/rf.hpp, the helper the
// simulator and the sanitizer use:
//   reg_visible: the vtime from which an RF sample sees a fixed result (async
//     results are covered by depbars instead);
//   read_until: the exclusive end of the fixed reads' samples;
//   predicate_ready: the vtime from which a predicate is in the file (a
//     conditional branch samples it issue_to_predicate_sample after its issue;
//     a predicate writer may issue from it). A setp's comes from the
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
        for (const auto source : sources(inst)) {
            at = std::max(at, reg_visible[source] - int64_t{issue_to_sample}); // RAW
        }
        if (const auto dst = destination(inst); dst.has_value()) {
            const auto width = inst.opcode == Opcode::acc_ld_v8 || inst.opcode == Opcode::ldg_v8 ? 8u : 1u;
            const int64_t earliest = issue_to_visible_min(inst, profile);
            for (unsigned j = 0; j < width; ++j) {
                at = std::max(at, reg_visible[*dst + j] - earliest + 1); // WAW: land after the older write
                at = std::max(at, read_until[*dst + j] - earliest);      // WAR: land at or after older samples
            }
        }
        if (inst.opcode == Opcode::bra_uni && inst.control.predicate != 7) {
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
            if (const auto dst = destination(inst); dst.has_value()) {
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

} // namespace

AssemblyProgram compile_qtx(std::string_view source, const DeviceProfile& profile) {
    validate_device_profile(profile);
    auto program = parse_program(source, false, profile);
    program.register_count = register_count(program);
    program.profile_name = profile.name;
    program.profile_fingerprint = device_profile_fingerprint(profile);
    std::array<bool, 6> live{};
    std::array<std::optional<uint8_t>, 256> pending_register{};
    std::array<uint8_t, 256> pending_async_read_mask{};
    // tmem order between one warp's transfers (trash/rfc11.md section 0: the
    // acc.ld and acc.st engines run side by side, nothing orders them): per
    // accumulator, the depbars of its acc.st and acc.ld still in flight. A
    // transfer after an acc.st of the same accumulator waits for it (RAW,
    // WAW); an acc.st after an acc.ld waits for that (WAR). Whole-accumulator
    // granularity: the coordinates are registers.
    std::array<uint8_t, 256> acc_store_mask{};
    std::array<uint8_t, 256> acc_load_mask{};
    std::optional<uint8_t> store_barrier;
    std::optional<uint8_t> mma_barrier;

    // Virtual issue timeline. Consecutive same-warp issues are at least max(stall, 1)
    // apart, and real gaps only grow (scheduler contention, depbar waits, structural
    // stalls), so any spacing proven here holds on the machine. The fixed-latency
    // distances are FixedTimeline's (src/rf.hpp). A stall is paid only when a
    // consumer sits closer than that, and it lands on the instruction just before
    // the consumer.
    FixedTimeline timeline;
    int64_t vtime = 0;
    int64_t previous_vtime = 0;
    Instruction* previous = nullptr;

    const auto allocate = [&]() -> uint8_t {
        for (uint8_t bar = 0; bar < 6; ++bar) {
            if (!live[bar]) {
                live[bar] = true;
                return bar;
            }
        }
        throw SimError("qtx scheduler exhausted six live depbars");
    };

    for (auto& inst : program.instructions) {
        if (is_mbar_try_wait(inst.opcode) && profile.mbar_wait_slots == 0) {
            throw SimError(
                "line " + std::to_string(inst.line) +
                ": mbar.try_wait is unsupported by this device profile");
        }
        if (is_tma_opcode(inst.opcode) && profile.tma_queue_commands == 0) {
            throw SimError(
                "line " + std::to_string(inst.line) +
                ": tma.load / tma.store are unsupported by this device profile");
        }
        if ((is_mma_async(inst.opcode) || inst.opcode == Opcode::mma_commit) && profile.mma_queue_commands == 0) {
            throw SimError("line " + std::to_string(inst.line) + ": MMA is unsupported by this device profile");
        }
        if (inst.opcode == Opcode::bra &&
            (inst.control.predicate != 7 || inst.control.predicate_invert)) {
            throw SimError("line " + std::to_string(inst.line) + ": bra is unconditional");
        }
        if ((inst.control.predicate != 7 || inst.control.predicate_invert) &&
            inst.opcode != Opcode::bra_uni) {
            throw SimError(
                "line " + std::to_string(inst.line) +
                ": only bra.uni may be predicated before divergent execution exists");
        }
        uint8_t wait_mask = 0;
        for (const auto source : sources(inst)) {
            if (pending_register[source].has_value()) {
                wait_mask |= static_cast<uint8_t>(1u << *pending_register[source]);
            }
        }
        if (const auto dst = destination(inst); dst.has_value()) {
            const auto width = inst.opcode == Opcode::acc_ld_v8 || inst.opcode == Opcode::ldg_v8 ? 8u : 1u;
            for (unsigned j = 0; j < width; ++j) {
                if (pending_register[*dst + j].has_value())
                    wait_mask |= static_cast<uint8_t>(1u << *pending_register[*dst + j]);
                wait_mask |= pending_async_read_mask[*dst + j];
            }
        }
        if (inst.opcode == Opcode::acc_ld || inst.opcode == Opcode::acc_ld_v8 ||
            inst.opcode == Opcode::acc_st || inst.opcode == Opcode::acc_st_v8) {
            const bool load = inst.opcode == Opcode::acc_ld || inst.opcode == Opcode::acc_ld_v8;
            const auto accumulator = load ? inst.src0 : inst.dst;
            wait_mask |= acc_store_mask[accumulator];
            if (!load) wait_mask |= acc_load_mask[accumulator];
        }
        if (inst.opcode == Opcode::mbar_arrive && store_barrier.has_value()) {
            // Pinned worker publication sequence: an ordinary arrival following
            // async stores is not allowed to publish until their counted
            // completion depbar reaches zero.
            wait_mask |= static_cast<uint8_t>(1u << *store_barrier);
        }
        if (inst.opcode == Opcode::halt || is_branch(inst.opcode)) {
            for (uint8_t bar = 0; bar < 6; ++bar) {
                if (live[bar]) wait_mask |= static_cast<uint8_t>(1u << bar);
            }
        }
        inst.control.wait_mask = wait_mask;

        if (wait_mask != 0) {
            for (auto& pending : pending_register) {
                if (pending.has_value() && (wait_mask & (1u << *pending)) != 0) pending.reset();
            }
            for (auto& mask : pending_async_read_mask) mask &= static_cast<uint8_t>(~wait_mask);
            for (auto& mask : acc_store_mask) mask &= static_cast<uint8_t>(~wait_mask);
            for (auto& mask : acc_load_mask) mask &= static_cast<uint8_t>(~wait_mask);
            for (uint8_t bar = 0; bar < 6; ++bar) {
                if ((wait_mask & (1u << bar)) != 0) live[bar] = false;
            }
            if (store_barrier.has_value() && (wait_mask & (1u << *store_barrier)) != 0) {
                store_barrier.reset();
            }
            if (mma_barrier.has_value() && (wait_mask & (1u << *mma_barrier)) != 0)
                mma_barrier.reset();
        }

        check_rf_read_round(inst, profile, "line " + std::to_string(inst.line));
        check_mma_shape(inst, program.accumulators, profile, "line " + std::to_string(inst.line));
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
        timeline.record(inst, profile, vtime);
        if (inst.opcode == Opcode::lds || inst.opcode == Opcode::lds_uni || inst.opcode == Opcode::ldg ||
            inst.opcode == Opcode::ldg_v8 || inst.opcode == Opcode::ldg_uni || is_atomic(inst.opcode)) {
            const auto bar = allocate();
            inst.control.set_depbar = bar;
            const auto width = inst.opcode == Opcode::ldg_v8 ? 8u : 1u;
            for (unsigned j = 0; j < width; ++j) pending_register[inst.dst + j] = bar;
        } else if (inst.opcode == Opcode::sts || inst.opcode == Opcode::stg || inst.opcode == Opcode::stg_v8) {
            if (!store_barrier.has_value()) store_barrier = allocate();
            inst.control.set_depbar = *store_barrier;
            if (inst.opcode == Opcode::stg_v8) {
                for (unsigned j = 0; j < 8; ++j)
                    pending_async_read_mask[inst.src1 + j] |= static_cast<uint8_t>(1u << *store_barrier);
            }
        } else if (inst.opcode == Opcode::acc_ld || inst.opcode == Opcode::acc_ld_v8) {
            const auto bar = allocate();
            inst.control.set_depbar = bar;
            const auto width = inst.opcode == Opcode::acc_ld_v8 ? 8u : 1u;
            for (unsigned j = 0; j < width; ++j) pending_register[inst.dst + j] = bar;
            acc_load_mask[inst.src0] |= static_cast<uint8_t>(1u << bar);
        } else if (inst.opcode == Opcode::acc_st || inst.opcode == Opcode::acc_st_v8) {
            if (!store_barrier.has_value()) store_barrier = allocate();
            inst.control.set_depbar = *store_barrier;
            acc_store_mask[inst.dst] |= static_cast<uint8_t>(1u << *store_barrier);
            const auto width = inst.opcode == Opcode::acc_st_v8 ? 8u : 1u;
            for (unsigned j = 0; j < width; ++j)
                pending_async_read_mask[inst.src1 + j] |= static_cast<uint8_t>(1u << *store_barrier);
        } else if (is_mma_async(inst.opcode)) {
            if (!mma_barrier.has_value()) mma_barrier = allocate();
            inst.control.set_depbar = *mma_barrier;
        }

        if (is_branch(inst.opcode)) {
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

            // Every branch is a conservative block boundary for v0: its wait mask
            // drains async work, and its stall covers outstanding fixed-latency writes.
            // Both the target and fallthrough therefore begin with clean timing state.
            live.fill(false);
            pending_register.fill(std::nullopt);
            pending_async_read_mask.fill(0);
            acc_store_mask.fill(0);
            acc_load_mask.fill(0);
            store_barrier.reset();
            mma_barrier.reset();
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
    return program;
}

AssemblyProgram compile_qtx_file(const std::string& path, const DeviceProfile& profile) {
    return compile_qtx(read_file(path), profile);
}

// Hand assembly's fixed-latency stalls: each instruction's stall becomes the
// larger of its own and what the fixed-latency register and predicate
// dependencies of the instructions after it need under `profile` -- the
// distances qtxc schedules with (FixedTimeline, src/rf.hpp) -- so the result
// never trips the sanitizer's RAW / WAW / WAR checks on a fixed result.
// Depbars, waits and every other field stay the author's. A branch drains
// like qtxc's block boundary, and the code after it starts clean.
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
    for (std::size_t index = 0; index < instructions.size(); ++index) {
        auto& inst = instructions[index];
        const auto required = timeline.required(inst, profile, vtime);
        if (required > vtime) {
            if (!previous.has_value()) throw SimError("internal: first instruction cannot need a stall");
            set_stall(instructions[*previous], required - previous_vtime);
            vtime = required;
        }
        timeline.record(inst, profile, vtime);
        if (is_branch(inst.opcode)) {
            const auto all_ready = timeline.drained(vtime, profile);
            if (all_ready - vtime > std::max<int64_t>(inst.control.stall, 1)) set_stall(inst, all_ready - vtime);
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
    for (std::size_t index = 0; index < instructions.size(); ++index) kernel.words[index] = encode(instructions[index]);
    return kernel;
}

} // namespace nanosm

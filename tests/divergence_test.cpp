// Divergence: ssy, the divergent bra
// (@p bra), sync, the per-warp active mask and the SM's divergence stack.
// Every entry is a (mask, pc) pair; a sync always pops one and switches to
// it. Each arm ends in its own sync; the label after the last arm is an
// ordinary instruction. Smoke tests: per-lane if / else, an arm no lane
// takes, the timing facts the design rests on, and the refusals and PANICs.
// Then masked instructions (@p / @!p on the per-lane instructions) and every
// writer class under a partial active mask.
#include "test_support.hpp"

using namespace nanosm_test;

namespace {

// One warp's run, ticked by hand: the divergence stack's peak depth and the
// warp's active mask each time it changed (starting with the launch's).
struct DivergentRun {
    std::unique_ptr<nanosm::Machine> machine;
    uint32_t peak_depth = 0;
    std::vector<uint64_t> masks;
};

DivergentRun run_divergent(const nanosm::Kernel& kernel, const std::function<void(nanosm::Launch&)>& setup = {}) {
    auto launch = launch_warps(1);
    image_bytes(launch.gmem_image, 4096);
    image_bytes(launch.test_shmem, 8192);
    if (setup) setup(launch);
    DivergentRun run;
    run.machine = std::make_unique<nanosm::Machine>(canonical_profile(), kernel, std::move(launch),
        nanosm::TraceOptions{.timeline = true});
    run.masks.push_back(run.machine->active_mask(0));
    while (!run.machine->done()) {
        tick_capped(*run.machine);
        run.peak_depth = std::max(run.peak_depth, run.machine->divergence_depth(0));
        if (run.machine->active_mask(0) != run.masks.back()) run.masks.push_back(run.machine->active_mask(0));
    }
    return run;
}

DivergentRun run_divergent(const std::string& qtx, const std::function<void(nanosm::Launch&)>& setup = {}) {
    return run_divergent(compile(qtx), setup);
}

// Lane l's global words 0 and 256 hold 100 + l and 0xdead, its shared words
// 0 and 512 hold 200 + l and 0xbeef.
void fill_lane_words(nanosm::Launch& launch) {
    auto& gmem = launch.gmem_image.at(0).bytes;
    auto& shmem = launch.test_shmem.at(0).bytes;
    for (uint32_t lane = 0; lane < 16; ++lane) {
        write_u32(gmem, 4 * lane, 100 + lane);
        write_u32(gmem, 256 + 4 * lane, 0xdead);
        write_u32(shmem, 4 * lane, 200 + lane);
        write_u32(shmem, 512 + 4 * lane, 0xbeef);
    }
}

std::string describe_masks(const std::vector<uint64_t>& masks) {
    std::string text;
    for (const auto mask : masks) {
        std::ostringstream out;
        out << " 0x" << std::hex << mask;
        text += out.str();
    }
    return text;
}

// Issue cycles of every instruction named `name`, in order.
std::vector<uint64_t> issues_named(const nanosm::Machine& machine, const std::string& name) {
    std::vector<uint64_t> cycles;
    for (const auto& event : machine.trace().events()) {
        if (is_issue_event(event) && event.name == name) cycles.push_back(event.start_cycle);
    }
    return cycles;
}

std::string lane_name(uint32_t lane, bool on, const char* yes, const char* no) {
    return "lane " + std::to_string(lane) + (on ? yes : no);
}

} // namespace

// A per-lane if / else: even lanes write 11, odd lanes 22, into the same
// register; one stg after the join stores all sixteen. The fall-through arm
// (odd) runs first and ends in a sync that pops the even lanes (EVEN, the
// next instruction); theirs pops the ssy's entry (JOIN, the next instruction
// again). The stack holds two entries; the joined stg issues once, under the
// full mask. The timing: a divergent bra that pushes falls through like a
// not-taken branch, a sync that pops its own pc + 8 goes on like a not-taken
// one (both syncs here), an ssy holds nothing, each at least its own stall.
// With every lane falling, the odd arm's sync pops the ssy's entry, JOIN, past
// the empty even arm: a redirect like a taken branch.
TEST(Divergence, if_else_per_lane) {
    const auto if_else = [](const std::string& predicate) {
        return compile(R"(
        sread r0, LANE_ID
        andi r1, r0, 1
        )" + predicate + R"(
        shli r3, r0, 2
        ssy JOIN
        @p0 bra EVEN
        mov r2, 22
        sync
EVEN:   mov r2, 11
        sync
JOIN:   stg [r3 + 0], r2
        halt
    )");
    };
    // `pc` issues `interval` (or the stall of `from`, if longer) after `from`.
    const auto expect = [](const nanosm::Kernel& kernel, const nanosm::Machine& machine, const std::string& what,
                            uint32_t pc, uint32_t from, uint64_t interval) {
        const auto got = issued_at(machine, pc).at(0);
        const auto want = issued_at(machine, from).at(0) +
            std::max(interval, uint64_t{nanosm::decode(kernel.words.at(from / 8)).control.stall});
        require_eq(got, want, what + ": issue cycle", ":" + issue_log(machine));
    };

    const auto kernel = if_else("setp.eq.u32 p0, r1, 0");
    const auto run = run_divergent(kernel);
    const auto& machine = *run.machine;
    require_lanes(gmem_lanes(machine, 0), [](uint32_t lane) { return lane % 2 == 0 ? 11u : 22u; }, "each lane's word");
    require(run.masks == std::vector<uint64_t>{0xffff, 0xaaaa, 0x5555, 0xffff},
        "active masks:" + describe_masks(run.masks) + " (want full, odd, even, full)");
    require_eq(run.peak_depth, 2, "peak depth");
    require(issued_at(machine, 80).size() == 1, "the joined stg did not issue exactly once:" + issue_log(machine));
    require(issued_at(machine, 56).size() == 1 && issued_at(machine, 72).size() == 1, "each arm's sync issues once");
    expect(kernel, machine, "after ssy (no hold)", 40, 32, 1);
    expect(kernel, machine, "fall-through after a pushing bra", 48, 40, nanosm::branch_not_taken_interval);
    expect(kernel, machine, "the even arm after the odd arm's sync (next pc)", 64, 56, nanosm::branch_not_taken_interval);
    expect(kernel, machine, "the joined code after the even arm's sync (next pc)", 80, 72,
        nanosm::branch_not_taken_interval);

    const auto falling = if_else("setp.lt.u32 p0, r1, 0");
    const auto all_fall = run_divergent(falling);
    require_lanes(gmem_lanes(*all_fall.machine, 0), 22, "every lane falling");
    require(issued_at(*all_fall.machine, 64).empty(), "the empty even arm ran");
    expect(falling, *all_fall.machine, "the joined code after the odd arm's sync (past the even arm)", 80, 56,
        nanosm::branch_taken_interval);
}

// An arm no lane takes pushes nothing: every lane taken is a jump, no lane
// taken a fall-through; only the ssys' entries reach the stack. The arm
// that runs alone ends in a sync that pops the ssy's entry: past the other
// arm (a redirect) or, for the last arm, the next instruction.
TEST(Divergence, one_sided_branch_pushes_nothing) {
    const auto run = run_divergent(R"(
        sread r0, LANE_ID
        shli r3, r0, 2
        setp.lt.u32 p0, r0, 16
        mov r2, 1
        ssy J
        @p0 bra T
        mov r2, 2
        sync
T:      mov r2, 3
        sync
J:      ssy K
        @!p0 bra U
        addi r2, r2, 10
        sync
U:      mov r2, 99
        sync
K:      stg [r3 + 0], r2
        halt
    )");
    require_lanes(gmem_lanes(*run.machine, 0), 13, "lane result");
    require_eq(run.peak_depth, 1, "peak depth (1: the ssys' only)");
    require(run.masks == std::vector<uint64_t>{0xffff}, "the mask changed:" + describe_masks(run.masks));
    for (const uint32_t pc : {48u, 56u, 112u, 120u}) {
        require(issued_at(*run.machine, pc).empty(), "an arm no lane takes ran (pc " + std::to_string(pc) + ")");
    }
}

// A uniform operation in an arm is the sanitizer's PANIC (bra.uni included:
// it is a promise about the whole warp), and so is a halt inside an arm.
TEST(Divergence, uniform_operation_or_halt_in_an_arm_panics) {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    for (const auto& [body, needle] : std::vector<std::pair<std::string, std::string>>{
             {"umov ur1, 0\nlds.uni r1, [ur1]", "lds.uni is a uniform operation under a partial active mask"},
             {"bra.uni NEXT\nNEXT: mov r1, 1", "bra.uni is a uniform operation under a partial active mask"},
             {"halt", "halt inside a divergent region (2 divergence stack entries left, active mask 0xff)"}}) {
        auto launch = launch_warps(1);
        image_bytes(launch.test_shmem, 64);
        nanosm::Machine machine(canonical_profile(), compile(masked_region(0x00ff, body) + "halt\n"), std::move(launch));
        expect_panic(machine, {needle}, body);
    }
    // A halt directly behind an ssy issues in the cycle the ssy's push
    // resolves (an ssy does not hold its warp); the check counts that push.
    nanosm::Machine machine(canonical_profile(), compile("ssy JOIN\nhalt\nJOIN: halt\n"), launch_warps(1));
    expect_panic(machine, {"halt inside a divergent region (1 divergence stack entries left, active mask 0xffff)"},
        "halt directly behind an ssy");
}

// The machine's own PANICs: a push onto a full stack (two entries here) and
// a sync on an empty one.
TEST(Divergence, stack_overflow_and_empty_sync_panic) {
    auto small = canonical_profile();
    small.core_divergence_stack_entries = 2;
    nanosm::Machine overflow(small, compile("ssy A\nssy A\nssy A\nA: sync\nhalt\n", small), launch_warps(1));
    expect_panic(overflow, {"divergence stack overflow: warp 0 already holds 2 entries"}, "overflow");
    nanosm::Machine empty(canonical_profile(), compile("sync\nhalt\n"), launch_warps(1));
    expect_panic(empty, {"sync with an empty divergence stack"}, "empty sync");
}

// Masked instructions: the operation runs and its result is written with a
// mask, the predicate's lanes. p0 holds the odd lanes. Masked arithmetic,
// sread, the special-function unit, setp (its predicate bit kept where the
// write mask is off), ldg / stg and lds / sts write (or send) only the lanes
// their write mask names; inside a divergent region the lanes are the active
// mask AND the write mask. And the active mask alone is a write enable for
// every per-lane writer: register writers under 0x0f0f, memory under 0x00f0.
TEST(Divergence, write_masks_are_lane_write_enables) {
    {
        const auto run = run_divergent(R"(
            sread r0, LANE_ID
            andi r1, r0, 1
            shli r3, r0, 2
            setp.ne.u32 p0, r1, 0
            setp.eq.u32 p1, r0, r0
            mov r10, 0x77
            mov r11, 0x77
            mov r12, 0x77
            mov r13, 0x77
            mov r14, 0x77
            mov r15, 0
            mov r5, 0x55
            mov r20, 0
            mov r21, 0
            mov r22, 0
            @p0 addi r10, r0, 1000
            @!p0 sread r11, LANE_ID
            @p0 ex2 r12, r15
            @p0 setp.ne.u32 p1, r0, r0
            setp.eq.u32 p2, r1, 0
            @!p2 setp.eq.u32 p2, r0, r0
            @p2 mov r22, 1
            @p0 ldg r13, [r3 + 0]
            @p0 stg [r3 + 256], r5
            @!p0 lds r14, [r3 + 0]
            @!p0 sts [r3 + 512], r5
            @p1 mov r20, 1
        )" + masked_region(0x00ff, "@p0 mov r21, 1") + "halt\n", fill_lane_words);
        const auto& machine = *run.machine;
        for (uint32_t lane = 0; lane < 16; ++lane) {
            const bool odd = (lane & 1) != 0;
            const auto what = lane_name(lane, odd, " (odd)", " (even)");
            require(machine.read_reg(0, lane, 10) == (odd ? 1000 + lane : 0x77), what + ": @p0 addi");
            require(machine.read_reg(0, lane, 11) == (odd ? 0x77 : lane), what + ": @!p0 sread");
            require(machine.read_reg(0, lane, 12) == (odd ? 0x3f800000u : 0x77u), what + ": @p0 ex2");
            require(machine.read_reg(0, lane, 20) == (odd ? 0u : 1u), what + ": @p0 setp (even lanes keep p1 true)");
            require(machine.read_reg(0, lane, 22) == 1u,
                what + ": @!p2 setp p2 (an inverted write mask writing its own predicate)");
            require(machine.read_reg(0, lane, 13) == (odd ? 100 + lane : 0x77), what + ": @p0 ldg");
            require(machine.read_gmem_u32(256 + 4 * lane) == (odd ? 0x55u : 0xdeadu), what + ": @p0 stg");
            require(machine.read_reg(0, lane, 14) == (odd ? 0x77 : 200 + lane), what + ": @!p0 lds");
            require(machine.read_shmem_u32(512 + 4 * lane) == (odd ? 0xbeefu : 0x55u), what + ": @!p0 sts");
            require(machine.read_reg(0, lane, 21) == (odd && lane < 8 ? 1u : 0u), what + ": mask 0xff AND @p0");
        }
        require_eq(machine.trace().summary().lsu_sectors, 16, "gmem requests (8 + 8)");
    }
    {
        // An arithmetic write, a special register, ur2r, the special-function
        // unit, a conversion, a vector compare and a uniform compare under
        // 0x0f0f; the predicates come back out through divergent branches.
        constexpr uint64_t mask = 0x0f0f;
        const auto run = run_divergent(R"(
            sread r0, LANE_ID
            mov r10, 0x77
            mov r11, 0x77
            mov r12, 0x77
            mov r13, 0x77
            mov r14, 0x77
            mov r15, 0
            umov ur1, 0x55
            setp.eq.u32 p1, r0, r0
            usetp.eq.u32 p3, ur1, 0
        )" + masked_region(mask, R"(
            addi r10, r0, 1000
            sread r11, LANE_ID
            ur2r r12, ur1
            ex2 r13, r15
            cvt.pack.f16x2 r14, r15, r15
            setp.ne.u32 p1, r0, r0
            usetp.eq.u32 p3, ur1, 0x55
        )") + R"(
            mov r20, 0
            mov r21, 0
            ssy A
            @!p1 bra A1
            mov r20, 1
            sync
A1:         sync
A:          ssy B
            @!p3 bra B1
            mov r21, 1
            sync
B1:         sync
B:          halt
        )");
        const auto& machine = *run.machine;
        for (uint32_t lane = 0; lane < 16; ++lane) {
            const bool active = (mask >> lane & 1) != 0;
            const auto what = lane_name(lane, active, " (active)", " (inactive)");
            require(machine.read_reg(0, lane, 10) == (active ? 1000 + lane : 0x77), what + ": addi");
            require(machine.read_reg(0, lane, 11) == (active ? lane : 0x77), what + ": sread");
            require(machine.read_reg(0, lane, 12) == (active ? 0x55u : 0x77u), what + ": ur2r");
            require(machine.read_reg(0, lane, 13) == (active ? 0x3f800000u : 0x77u), what + ": ex2");
            require(machine.read_reg(0, lane, 14) == (active ? 0u : 0x77u), what + ": cvt.pack");
            require(machine.read_reg(0, lane, 20) == (active ? 0u : 1u), what + ": setp (p1 kept true when inactive)");
            require(machine.read_reg(0, lane, 21) == (active ? 1u : 0u), what + ": usetp (p3 set only when active)");
        }
        require(run.masks.back() == 0xffff && run.peak_depth == 2, "the region did not rejoin");
    }
    {
        // A scalar global load and store and a shared load and store under
        // 0x00f0: only the active lanes are requested (four sectors each way).
        constexpr uint64_t mask = 0x00f0;
        const auto run = run_divergent(R"(
            sread r0, LANE_ID
            shli r3, r0, 2
            mov r4, 0x77
            mov r6, 0x77
            mov r5, 0x55
        )" + masked_region(mask, R"(
            ldg r4, [r3 + 0]
            stg [r3 + 256], r5
            lds r6, [r3 + 0]
            sts [r3 + 512], r5
        )") + "halt\n", fill_lane_words);
        const auto& machine = *run.machine;
        for (uint32_t lane = 0; lane < 16; ++lane) {
            const bool active = (mask >> lane & 1) != 0;
            const auto what = lane_name(lane, active, " (active)", " (inactive)");
            require(machine.read_reg(0, lane, 4) == (active ? 100 + lane : 0x77), what + ": ldg");
            require(machine.read_gmem_u32(256 + 4 * lane) == (active ? 0x55u : 0xdeadu), what + ": stg");
            require(machine.read_reg(0, lane, 6) == (active ? 200 + lane : 0x77), what + ": lds");
            require(machine.read_shmem_u32(512 + 4 * lane) == (active ? 0x55u : 0xbeefu), what + ": sts");
        }
        const auto& summary = machine.trace().summary();
        require_eq(summary.lsu_sectors, 8, "gmem requests (4 for the load and 4 for the store)");
        require_eq(summary.lsu_useful_bytes, 32, "useful bytes");
    }
}

// Timing never depends on the write mask. The same program with p0 set in every
// lane and in none: every instruction issues at the same cycle (the
// special-function unit takes its full cycles, depbars are set and waited
// for alike). With no lane on, the ldg / lds / stg / sts make no request and
// leave memory and registers alone; each leaves its queue at its issue + 1
// (its predicate's sample), giving back its depbars, so the consumer waiting
// for the ldg's depbar issues 2 cycles after it. Nor on the active mask: two
// back-to-back SFU operations, and an arithmetic write and its consumer, are
// as far apart under one active lane as under sixteen.
TEST(Divergence, timing_ignores_the_write_mask) {
    std::vector<std::vector<uint64_t>> issues;
    for (const bool on : {true, false}) {
        const auto run = run_divergent(std::string(R"(
            sread r0, LANE_ID
            shli r3, r0, 2
            mov r4, 0x77
            mov r5, 0x77
            mov r15, 0
        )") + (on ? "setp.eq.u32 p0, r0, r0\n" : "setp.ne.u32 p0, r0, r0\n") + R"(
            @p0 ex2 r13, r15
            @p0 ex2 r14, r15
            @p0 addi r10, r13, 1
            @p0 ldg r4, [r3 + 0]
            addi r7, r4, 1
            @p0 lds r5, [r3 + 0]
            @p0 sts [r3 + 512], r5
            @p0 stg [r3 + 256], r7
            addi r8, r5, 1
            halt
        )", fill_lane_words);
        const auto& machine = *run.machine;
        std::vector<uint64_t> fixed;
        for (const auto* name : {"ex2", "addi"}) {
            for (const auto cycle : issues_named(machine, name)) fixed.push_back(cycle);
        }
        issues.push_back(fixed);
        if (on) continue;
        const auto& summary = machine.trace().summary();
        require(summary.lsu_sectors == 0 && summary.shmem_requests == 0,
            "a write mask with no lane on made requests: " + std::to_string(summary.lsu_sectors) + " gmem, " +
            std::to_string(summary.shmem_requests) + " shmem");
        for (uint32_t lane = 0; lane < 16; ++lane) {
            require(machine.read_reg(0, lane, 4) == 0x77 && machine.read_reg(0, lane, 5) == 0x77 &&
                    machine.read_reg(0, lane, 7) == 0x78 && machine.read_gmem_u32(256 + 4 * lane) == 0xdead &&
                    machine.read_shmem_u32(512 + 4 * lane) == 0xbeef,
                "lane " + std::to_string(lane) + ": a write mask with no lane on changed something");
        }
        const auto ldg = issues_named(machine, "ldg");
        const auto addi = issues_named(machine, "addi");
        require(ldg.size() == 1 && addi.size() == 3, "the ldg and the three addis did not issue once each");
        require(addi[1] == ldg[0] + 2,
            "the ldg's consumer issued " + std::to_string(addi[1] - ldg[0]) + " cycles after it (want 2)");
    }
    // The fixed instructions (ex2, ex2, addi, then the two consumers, which
    // wait for depbars) at the same cycles both ways: only the first three.
    require(issues[0].size() >= 3 && issues[1].size() >= 3 &&
            std::equal(issues[0].begin(), issues[0].begin() + 3, issues[1].begin()),
        "fixed issue cycles depend on the write mask");

    // A real store still in flight and an empty one on the same depbar (qtxc
    // gives every store one), then a wait. The empty store owes exactly one
    // decrement (an extra one would underflow, a missing one would never let
    // the wait go) and the real write still lands.
    const auto shared = run_divergent(R"(
        sread r0, LANE_ID
        shli r3, r0, 2
        mov r5, 0x55
        setp.ne.u32 p0, r0, r0
        stg [r3 + 256], r5
        @p0 stg [r3 + 512], r5
        wait
        ldg r7, [r3 + 256]
        halt
    )", [](nanosm::Launch& launch) {
        auto& gmem = launch.gmem_image.at(0).bytes;
        for (uint32_t lane = 0; lane < 16; ++lane) write_u32(gmem, 512 + 4 * lane, 0xdead);
    });
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(shared.machine->read_gmem_u32(256 + 4 * lane) == 0x55 &&
                shared.machine->read_reg(0, lane, 7) == 0x55 &&
                shared.machine->read_gmem_u32(512 + 4 * lane) == 0xdead,
            "lane " + std::to_string(lane) + ": a real and an empty store on one depbar");
    }

    // The active mask: one lane against sixteen.
    std::vector<std::vector<uint64_t>> distances;
    for (const uint64_t mask : {uint64_t{0x0001}, uint64_t{0xffff}}) {
        const auto run = run_divergent("mov r15, 0\n" + masked_region(mask, R"(
            ex2 r13, r15
            ex2 r14, r15
            addi r10, r13, 1
            addi r11, r10, 1
        )") + "halt\n");
        const auto ex2 = issues_named(*run.machine, "ex2");
        const auto addi = issues_named(*run.machine, "addi");
        require(ex2.size() == 2 && addi.size() == 2, "the region's instructions did not issue once each");
        distances.push_back({ex2[1] - ex2[0], addi[0] - ex2[1], addi[1] - addi[0]});
    }
    require_eq(distances[0], distances[1], "issue distances under one lane, against sixteen");
}

// What refuses to be masked: qtxc, the assembler and kernel load, by name.
// And mov rD, rS: the register form is addi rD, rS, 0, so it can be masked.
TEST(Divergence, write_mask_refusals) {
    for (const auto& masked : std::vector<std::pair<std::string, std::string>>{
             {"@p0 uadd ur1, ur1, ur1", "uadd"}, {"@p0 usetp.eq.u32 p1, ur1, 0", "usetp.eq.u32"},
             {"@p0 r2ur ur1, r1", "r2ur"}, {"@!p0 lds.uni r1, [ur1]", "lds.uni"},
             {"@p0 mbar.arrive [ur1]", "mbar.arrive"},
             {"@p0 halt", "halt"}, {"@p0 sync", "sync"}, {"@p0 ssy L", "ssy"}, {"@p0 fence.sh", "fence.sh"}}) {
        expect_error([&] { (void)compile(masked.first + "\nL: halt\n"); },
            {masked.second + " cannot be masked (@p)"}, masked.first);
    }
    const auto& profile = canonical_profile();
    expect_error([&] {
        (void)nanosm::assemble_asm(hand_asm(profile, "uadd ur1, ur1, ur1 {pred=!p2, stall=0, set=-, wait=-}\nhalt\n"),
            profile);
    }, {"uadd cannot be masked (@p)"}, "the assembler");
    auto kernel = compile("uadd ur1, ur1, ur1\nhalt\n");
    auto word = nanosm::decode(kernel.words.at(0));
    word.control.predicate = 2;
    kernel.words.at(0) = nanosm::encode(word);
    expect_error([&] { nanosm::Machine machine(profile, kernel, launch_warps(1)); },
        {"kernel pc 0: uadd cannot be masked (@p)"}, "kernel load");
    // mov rD, rS is addi rD, rS, 0 (and prints as that); mov rD, IMM stays mov.
    const auto moves = compile("mov r1, r2\nmov r1, 2\nhalt\n");
    const auto move = nanosm::decode(moves.words.at(0));
    require(move.opcode == nanosm::Opcode::addi && move.dst == 1 && move.src0 == 2 && move.imm == 0 &&
            nanosm::format_instruction(move, false) == "addi r1, r2, 0",
        "mov r1, r2: " + nanosm::format_instruction(move));
    require(nanosm::decode(moves.words.at(1)).opcode == nanosm::Opcode::mov, "mov r1, 2 is not mov");
    // A masked instruction needs no divergence stack: a profile without one
    // runs it (here the register form of mov, masked).
    auto zero = profile;
    zero.core_divergence_stack_entries = 0;
    nanosm::validate_device_profile(zero);
    nanosm::Machine machine(zero, compile(R"(
        sread r0, LANE_ID
        andi r1, r0, 1
        mov r2, 7
        mov r3, 9
        setp.ne.u32 p0, r1, 0
        @p0 mov r2, r3
        halt
    )", zero), launch_warps(1));
    machine.run();
    require_lanes(machine, 2, [](uint32_t lane) { return (lane & 1) != 0 ? 9u : 7u; },
        "a masked mov on a profile with no divergence stack");
    // What does need the divergence stack (ssy, a predicated bra, sync) is
    // refused on that profile by qtxc, the assembler and kernel load. The
    // kernel for the load is built under the canonical profile and stamped
    // with this one's fingerprint, the two profiles differing in nothing else.
    struct NeedsStack {
        std::string qtx;
        std::string assembly;
        std::string needle;
    };
    for (const auto& row : std::vector<NeedsStack>{
             {"ssy L\nL: halt\n", "ssy L\nL: halt\n", "ssy needs the divergence stack"},
             {"setp.eq.u32 p0, r0, 0\n@p0 bra L\nL: halt\n",
                 "setp.eq.u32 p0, r0, 0\nbra L {pred=p0, stall=0, set=-, wait=-}\nL: halt\n",
                 "a predicated bra needs the divergence stack"},
             {"sync\nhalt\n", "sync\nhalt\n", "sync needs the divergence stack"}}) {
        expect_error([&] { (void)compile(row.qtx, zero); }, {row.needle}, "qtxc: " + row.needle);
        expect_error([&] { (void)nanosm::assemble_asm(hand_asm(zero, row.assembly), zero); }, {row.needle},
            "the assembler: " + row.needle);
        auto restamped = compile(row.qtx);
        restamped.profile_fingerprint = nanosm::device_profile_fingerprint(zero);
        expect_error([&] { nanosm::Machine refused(zero, restamped, launch_warps(1)); },
            {"kernel pc", row.needle}, "kernel load: " + row.needle);
    }
}

// A write mask read before its predicate is in the file is the sanitizer's
// PANIC, as for a branch (setp's predicate is in the file 5 cycles after its
// issue; the write mask is sampled 1 cycle after the masked instruction's).
TEST(Divergence, early_write_mask_panics) {
    const auto& profile = canonical_profile();
    const auto machine = [&](uint32_t stall) {
        const auto program = "setp.eq.u32 p0, r0, r0 {pred=PT, stall=" + std::to_string(stall) +
            ", set=-, wait=-}\nmov r1, 5 {pred=p0, stall=0, set=-, wait=-}\nhalt\n";
        return std::make_unique<nanosm::Machine>(
            profile, nanosm::assemble_asm(hand_asm(profile, program), profile), launch_warps(1));
    };
    const auto at = machine(4);
    at->run(20000);
    require(at->read_reg(0, 3, 1) == 5, "at 4 cycles the write mask did not see the compare's predicate");
#if NANOSM_SANITIZER
    const auto early = machine(3);
    expect_panic(*early, {"predicate RAW hazard: missing stall for p0"}, "a masked instruction three cycles after its setp");
#endif
}

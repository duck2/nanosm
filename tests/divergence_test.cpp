// Divergence (trash/divergence-2026-11-21.md): ssy, the divergent bra
// (@p bra), sync, the per-warp active mask and the SM's divergence stack.
// Every entry is a (mask, pc) pair; a sync always pops one and switches to
// it. Each arm ends in its own sync; the label after the last arm is an
// ordinary instruction. Smoke tests: per-lane if / else, a nested if, an arm
// no lane takes, the loop that sheds one lane at a time (Astra's layout, two
// entries), every writer class under a partial mask, the timing facts the
// design rests on, and the refusals and PANICs.
#include "test_support.hpp"

namespace {

using namespace nanosm_test;

// One warp's run, ticked by hand: the divergence stack's peak depth and the
// warp's active mask each time it changed (starting with the launch's).
struct DivergentRun {
    std::unique_ptr<nanosm::Machine> machine;
    uint32_t peak_depth = 0;
    std::vector<uint64_t> masks;
};

DivergentRun run_divergent(const nanosm::Kernel& kernel, const std::function<void(nanosm::Launch&)>& setup = {},
    const nanosm::DeviceProfile& profile = canonical_profile()) {
    nanosm::Launch launch;
    launch.active_warps = 1;
    image_bytes(launch.gmem_image, 4096);
    image_bytes(launch.shmem_image, 8192);
    if (setup) setup(launch);
    DivergentRun run;
    run.machine = std::make_unique<nanosm::Machine>(profile, kernel, std::move(launch),
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

uint32_t gmem_word(const nanosm::Machine& machine, uint32_t address) {
    return machine.read_gmem_u32(address);
}

std::string hex(uint64_t value) {
    std::ostringstream out;
    out << "0x" << std::hex << value;
    return out.str();
}

std::string describe_masks(const std::vector<uint64_t>& masks) {
    std::string text;
    for (const auto mask : masks) text += " " + hex(mask);
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

// A per-lane if / else, Astra's first kernel: even lanes write 11, odd lanes
// 22, into the same register; one stg after the join stores all sixteen.
// The fall-through arm (odd) runs first and ends in a sync that pops the
// even lanes (EVEN, the next instruction); theirs pops the ssy's entry
// (JOIN, the next instruction again). The stack holds two entries; the
// joined stg issues once, under the full mask. Then the timing, from the
// issue cycles: a divergent bra that pushes falls through like a not-taken
// branch (+2), a sync that pops its own pc + 8 goes on like a not-taken one
// (+2; both syncs here), an ssy holds nothing (+1), each at least its own
// stall count.
void test_if_else_per_lane() {
    const auto kernel = compile(R"(
        sread r0, LANE_ID
        andi r1, r0, 1
        setp.eq.u32 p0, r1, 0
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
    const auto run = run_divergent(kernel);
    const auto& machine = *run.machine;
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(gmem_word(machine, 4 * lane) == (lane % 2 == 0 ? 11u : 22u), "lane " + std::to_string(lane) + "'s word");
    }
    require(run.masks == std::vector<uint64_t>{0xffff, 0xaaaa, 0x5555, 0xffff},
        "active masks:" + describe_masks(run.masks) + " (want full, odd, even, full)");
    require(run.peak_depth == 2, "peak depth " + std::to_string(run.peak_depth) + " (want 2)");
    require(issued_at(machine, 80).size() == 1, "the joined stg did not issue exactly once:" + issue_log(machine));
    require(issued_at(machine, 56).size() == 1 && issued_at(machine, 72).size() == 1, "each arm's sync issues once");

    const auto stall = [&](uint32_t pc) { return uint64_t{nanosm::decode(kernel.words.at(pc / 8)).control.stall}; };
    const auto at = [&](uint32_t pc, std::size_t nth = 0) { return issued_at(machine, pc).at(nth); };
    const auto expect = [&](const std::string& what, uint64_t got, uint64_t from, uint64_t interval, uint32_t pc) {
        const auto want = from + std::max(interval, stall(pc));
        require(got == want, what + ": issued at c" + std::to_string(got) + ", want c" + std::to_string(want) + ":" +
            issue_log(machine));
    };
    expect("after ssy (no hold)", at(40), at(32), 1, 32);
    expect("fall-through after a pushing bra", at(48), at(40), nanosm::branch_not_taken_interval, 40);
    expect("the even arm after the odd arm's sync (next pc)", at(64), at(56), nanosm::branch_not_taken_interval, 56);
    expect("the joined code after the even arm's sync (next pc)", at(80), at(72), nanosm::branch_not_taken_interval,
        72);
}

// The front end one cycle after a sync resolves. A split if / else: the odd
// arm's sync pops EVEN (64) and the even arm's pops JOIN (80), each its own
// pc + 8: no redirect, the FIFO keeps the words fetched before. The same
// if / else with every lane falling: the odd arm's sync pops the ssy's
// entry, JOIN (72), past the empty even arm: a redirect, the FIFO cleared,
// the even arm never issues.
void test_sync_to_the_next_pc_keeps_fetched_words() {
    struct Case {
        std::string predicate;                   // p0: the lanes that branch to EVEN
        std::vector<std::pair<uint32_t, bool>> syncs; // sync pc, continues
        uint32_t join;
    };
    for (const auto& [predicate, syncs, join] : std::vector<Case>{
             {"setp.eq.u32 p0, r1, 0", {{56, true}, {72, true}}, 80},
             {"setp.lt.u32 p0, r1, 0", {{56, false}}, 80}}) {
        nanosm::Launch launch;
        launch.active_warps = 1;
        image_bytes(launch.gmem_image, 4096);
        nanosm::Machine machine(canonical_profile(), compile(R"(
            sread r0, LANE_ID
            andi r1, r0, 1
            )" + predicate + R"(
            shli r3, r0, 2
            ssy JOIN
            @p0 bra EVEN
            mov r2, 22
            sync
EVEN:       mov r2, 11
            sync
JOIN:       stg [r3 + 0], r2
            halt
        )"), std::move(launch), nanosm::TraceOptions{.timeline = true});
        std::map<uint64_t, std::pair<std::size_t, uint32_t>> front_end; // Q after each tick: (FIFO packets, pc)
        while (!machine.done()) {
            const auto cycle = machine.cycle();
            tick_capped(machine);
            front_end[cycle] = {machine.instruction_fifo_packets(0), machine.warp_pc(0)};
        }
        for (const auto& [sync_pc, continues] : syncs) {
            const auto issues = issued_at(machine, sync_pc);
            require(issues.size() == 1, predicate + ": the sync at " + std::to_string(sync_pc) + " did not issue once");
            // Q after the resolve cycle (issue + 1).
            const auto [packets, pc] = front_end.at(issues[0] + nanosm::issue_to_branch_resolve);
            const auto want_pc = continues ? sync_pc + 8 : join;
            require(pc == want_pc && (continues ? packets > 0 : packets == 0), predicate + ": the sync at " +
                std::to_string(sync_pc) + ": " + std::to_string(packets) + " packets, pc " + std::to_string(pc) +
                (continues ? " (want the fetched words kept, pc " : " (want a cleared FIFO, pc ") +
                std::to_string(want_pc) + ")");
        }
        if (syncs.size() == 1) require(issued_at(machine, 64).empty(), "the empty even arm ran");
    }
}

// A nested if inside the even arm; the inner join must restore the even
// lanes, not every lane (odd lanes would get +100). The inner if has an
// empty second arm: its sync alone (SKIP). Peak: the outer ssy's entry, the
// odd lanes, the inner ssy's entry, the skipping lanes: four.
void test_nested_if() {
    const auto run = run_divergent(R"(
        sread r0, LANE_ID
        andi r1, r0, 1
        setp.eq.u32 p0, r1, 0
        andi r4, r0, 2
        setp.ne.u32 p1, r4, 0
        shli r3, r0, 2
        mov r2, 0
        ssy OUTER
        @!p0 bra ODD
        ssy INNER
        @!p1 bra SKIP
        mov r2, 5
        sync
SKIP:   sync
INNER:  addi r2, r2, 100
        sync
ODD:    mov r2, 7
        sync
OUTER:  stg [r3 + 0], r2
        halt
    )");
    for (uint32_t lane = 0; lane < 16; ++lane) {
        const uint32_t want = lane % 2 == 1 ? 7 : (lane & 2) != 0 ? 105 : 100;
        require(gmem_word(*run.machine, 4 * lane) == want, "lane " + std::to_string(lane) + ": " +
            std::to_string(gmem_word(*run.machine, 4 * lane)) + ", want " + std::to_string(want));
    }
    require(run.peak_depth == 4, "peak depth " + std::to_string(run.peak_depth) + " (want 4)");
    require(run.masks == std::vector<uint64_t>{0xffff, 0x5555, 0x4444, 0x1111, 0x5555, 0xaaaa, 0xffff},
        "active masks:" + describe_masks(run.masks));
}

// An arm no lane takes pushes nothing: every lane taken is a jump, no lane
// taken a fall-through; only the ssys' entries reach the stack. The arm
// that runs alone ends in a sync that pops the ssy's entry: past the other
// arm (a redirect) or, for the last arm, the next instruction.
void test_one_sided_branch_pushes_nothing() {
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
    for (uint32_t lane = 0; lane < 16; ++lane) require(gmem_word(*run.machine, 4 * lane) == 13, "lane result");
    require(run.peak_depth == 1, "peak depth " + std::to_string(run.peak_depth) + " (want 1: the ssys' only)");
    require(run.masks == std::vector<uint64_t>{0xffff}, "the mask changed:" + describe_masks(run.masks));
    for (const uint32_t pc : {48u, 56u, 112u, 120u}) {
        require(issued_at(*run.machine, pc).empty(), "an arm no lane takes ran (pc " + std::to_string(pc) + ")");
    }
}

// Astra's loop (trash/chan.txt 2026-11-21, trash/divergence-loop-astra.py):
// lane l runs the body l times (0 .. 15). The first pass goes to the test;
// on a mixed test the leaving lanes fall into the sync, which pops the
// staying lanes (BODY) straight back to the body, so the stack never holds
// more than the ssy's entry and one other. When no lane stays, the sync
// pops the ssy's entry: DONE, the next instruction. The joined code runs
// once, under the full mask; the sync is visited sixteen times (fifteen
// redirects to BODY, then DONE).
void test_loop_sheds_one_lane_at_a_time() {
    const auto run = run_divergent(R"(
        sread r0, LANE_ID
        shli r3, r0, 2
        mov r1, 0
        mov r5, 0
        ssy DONE
        bra TEST
BODY:   addi r1, r1, 1
TEST:   setp.lt.u32 p0, r1, r0
        @p0 bra BODY
        sync
DONE:   addi r5, r5, 1
        stg [r3 + 0], r1
        halt
    )");
    const auto& machine = *run.machine;
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(gmem_word(machine, 4 * lane) == lane && machine.read_reg(0, lane, 1) == lane,
            "lane " + std::to_string(lane) + " ran the body " + std::to_string(machine.read_reg(0, lane, 1)) + " times");
        require(machine.read_reg(0, lane, 5) == 1, "the joined addi did not reach lane " + std::to_string(lane));
    }
    require(issued_at(machine, 80).size() == 1 && issued_at(machine, 88).size() == 1,
        "the joined code did not run exactly once");
    require(issued_at(machine, 48).size() == 15, "the body issued " + std::to_string(issued_at(machine, 48).size()) +
        " times (want 15: once per count up to the last lane's)");
    require(issued_at(machine, 72).size() == 16, "the sync was visited " +
        std::to_string(issued_at(machine, 72).size()) + " times (want 16)");
    require(run.peak_depth == 2, "peak depth " + std::to_string(run.peak_depth) + " (want 2)");
    require(run.masks.back() == 0xffff, "the warp did not end converged");
}

// Every per-lane register and predicate writer honours the mask: lanes
// 0..3 and 8..11 run an arithmetic write, a special register, ur2r, the
// special-function unit, a conversion, a vector compare and a uniform
// compare; the other lanes keep their old values. The predicates come back
// out through divergent branches.
void test_register_writers_under_a_partial_mask() {
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
A1:     sync
A:      ssy B
        @!p3 bra B1
        mov r21, 1
        sync
B1:     sync
B:      halt
    )");
    const auto& machine = *run.machine;
    for (uint32_t lane = 0; lane < 16; ++lane) {
        const bool active = (mask >> lane & 1) != 0;
        const auto what = "lane " + std::to_string(lane) + (active ? " (active)" : " (inactive)");
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

// Timing does not depend on the mask where it is fixed: two back-to-back
// SFU operations are as far apart under one active lane as under sixteen
// (the unit still takes its full cycles), and so are an arithmetic write's
// port and its consumer.
void test_fixed_timing_ignores_the_mask() {
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
    require(distances[0] == distances[1], "issue distances under one lane differ from sixteen: " +
        std::to_string(distances[0][0]) + "/" + std::to_string(distances[0][1]) + "/" + std::to_string(distances[0][2]) +
        " vs " + std::to_string(distances[1][0]) + "/" + std::to_string(distances[1][1]) + "/" +
        std::to_string(distances[1][2]));
}

// The memory writers: a scalar global load and store and a shared load and
// store under lanes 4..7. Only the active lanes are requested (four sectors
// each way) and written; the others keep their old values.
void test_memory_writers_under_a_partial_mask() {
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
    )") + "halt\n", [](nanosm::Launch& launch) {
        auto& gmem = launch.gmem_image.at(0).bytes;
        auto& shmem = launch.shmem_image.at(0).bytes;
        for (uint32_t lane = 0; lane < 16; ++lane) {
            write_u32(gmem, 4 * lane, 100 + lane);
            write_u32(gmem, 256 + 4 * lane, 0xdead);
            write_u32(shmem, 4 * lane, 200 + lane);
            write_u32(shmem, 512 + 4 * lane, 0xbeef);
        }
    });
    const auto& machine = *run.machine;
    for (uint32_t lane = 0; lane < 16; ++lane) {
        const bool active = (mask >> lane & 1) != 0;
        const auto what = "lane " + std::to_string(lane) + (active ? " (active)" : " (inactive)");
        require(machine.read_reg(0, lane, 4) == (active ? 100 + lane : 0x77), what + ": ldg");
        require(machine.read_gmem_u32(256 + 4 * lane) == (active ? 0x55u : 0xdeadu), what + ": stg");
        require(machine.read_reg(0, lane, 6) == (active ? 200 + lane : 0x77), what + ": lds");
        require(machine.read_shmem_u32(512 + 4 * lane) == (active ? 0x55u : 0xbeefu), what + ": sts");
    }
    const auto& summary = machine.trace().summary();
    require(summary.lsu_sectors == 8, "gmem requests: " + std::to_string(summary.lsu_sectors) +
        " (want 4 for the load and 4 for the store)");
    require(summary.lsu_useful_bytes == 32, "useful bytes: " + std::to_string(summary.lsu_useful_bytes));
}

// A uniform operation in an arm is the sanitizer's PANIC (bra.uni included:
// it is a promise about the whole warp), and so is a halt inside an arm.
void test_uniform_operation_or_halt_in_an_arm_panics() {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    for (const auto& [body, needle] : std::vector<std::pair<std::string, std::string>>{
             {"umov ur1, 0\nlds.uni r1, [ur1]", "lds.uni is a uniform operation under a partial active mask"},
             {"bra.uni NEXT\nNEXT: mov r1, 1", "bra.uni is a uniform operation under a partial active mask"},
             {"halt", "halt inside a divergent region (2 divergence stack entries left, active mask 0xff)"}}) {
        nanosm::Launch launch;
        launch.active_warps = 1;
        image_bytes(launch.shmem_image, 64);
        nanosm::Machine machine(canonical_profile(), compile(masked_region(0x00ff, body) + "halt\n"), std::move(launch));
        expect_sim_panic(machine, needle, body);
    }
    // A halt directly behind an ssy issues in the cycle the ssy's push
    // resolves (an ssy does not hold its warp); the check counts that push.
    nanosm::Launch launch;
    launch.active_warps = 1;
    nanosm::Machine machine(canonical_profile(), compile("ssy JOIN\nhalt\nJOIN: halt\n"), std::move(launch));
    expect_sim_panic(machine, "halt inside a divergent region (1 divergence stack entries left, active mask 0xffff)",
        "halt directly behind an ssy");
}

// The machine's own PANICs: a push onto a full stack (two entries here) and
// a sync on an empty one.
void test_stack_overflow_and_empty_sync_panic() {
    auto small = canonical_profile();
    small.core_divergence_stack_entries = 2;
    {
        nanosm::Launch launch;
        launch.active_warps = 1;
        nanosm::Machine machine(small, compile("ssy A\nssy A\nssy A\nA: sync\nhalt\n", small), std::move(launch));
        expect_sim_panic(machine, "divergence stack overflow: warp 0 already holds 2 entries", "overflow");
    }
    {
        nanosm::Launch launch;
        launch.active_warps = 1;
        nanosm::Machine machine(canonical_profile(), compile("sync\nhalt\n"), std::move(launch));
        expect_sim_panic(machine, "sync with an empty divergence stack", "empty sync");
    }
}

// core.divergence_stack_entries: canonical 8, optional (absent: 8), 0..64,
// fingerprinted. Zero means no stack: ssy, sync and a predicated bra are
// refused by qtxc, the assembler and kernel load; anything else runs.
void test_depth_zero_profile_refuses_divergence() {
    std::ifstream input("profiles/nanosm-v0.json");
    nlohmann::json document;
    input >> document;
    const auto& profile = canonical_profile();
    require(profile.core_divergence_stack_entries == 8, "the canonical profile does not declare 8 entries");
    auto defaulted = document;
    defaulted["core"].erase("divergence_stack_entries");
    require(nanosm::device_profile_fingerprint(nanosm::parse_device_profile(defaulted.dump())) ==
            nanosm::device_profile_fingerprint(profile), "the default is not 8");
    for (const auto value : {65, -1}) {
        auto modified = document;
        modified["core"]["divergence_stack_entries"] = value;
        try {
            (void)nanosm::parse_device_profile(modified.dump());
            throw std::runtime_error(std::to_string(value) + " entries were accepted");
        } catch (const nanosm::ProfileError&) {}
    }
    auto zero_document = document;
    zero_document["core"]["divergence_stack_entries"] = 0;
    const auto zero = nanosm::parse_device_profile(zero_document.dump());
    require(nanosm::device_profile_fingerprint(zero) != nanosm::device_profile_fingerprint(profile),
        "divergence_stack_entries is not in the fingerprint");

    const std::string divergent = "sread r0, LANE_ID\nsetp.eq.u32 p0, r0, 0\nssy J\n@p0 bra J\nmov r1, 1\nJ: sync\nhalt\n";
    for (const auto& [what, program] : std::vector<std::pair<std::string, std::string>>{
             {"ssy", "ssy J\nJ: sync\nhalt\n"}, {"sync", "sync\nhalt\n"},
             {"a predicated bra", "setp.eq.u32 p0, r0, r0\n@p0 bra J\nJ: halt\n"}}) {
        try {
            (void)compile(program, zero);
            throw std::runtime_error("qtxc accepted " + what + " without a stack");
        } catch (const nanosm::SimError& error) {
            require_mentions(error.what(), {what + " needs the divergence stack"}, "qtxc: " + what);
        }
    }
    try {
        (void)nanosm::assemble_asm(i1_asm(zero, "ssy J\nJ: sync\nhalt\n"), zero);
        throw std::runtime_error("the assembler accepted ssy without a stack");
    } catch (const nanosm::SimError& error) {
        require_mentions(error.what(), {"ssy needs the divergence stack"}, "assembler");
    }
    auto kernel = compile(divergent);
    kernel.profile_fingerprint = nanosm::device_profile_fingerprint(zero);
    try {
        nanosm::Launch launch;
        launch.active_warps = 1;
        nanosm::Machine machine(zero, kernel, std::move(launch));
        throw std::runtime_error("kernel load accepted a divergent kernel without a stack");
    } catch (const nanosm::SimError& error) {
        require_mentions(error.what(), {"kernel pc 16: ssy needs the divergence stack"}, "kernel load");
    }
    nanosm::Launch launch;
    launch.active_warps = 1;
    nanosm::Machine plain(zero, compile("mov r0, 1\nbra.uni END\nEND: halt\n", zero), std::move(launch));
    plain.run();
}

// The toolchain: the three forms parse, encode and print; an ssy's label
// may name any instruction; bssy is not a mnemonic; guards on other
// instructions stay refused.
void test_toolchain_forms() {
    const auto kernel = compile("ssy J\n@!p3 bra J\nJ: sync\nhalt\n");
    const auto ssy = nanosm::decode(kernel.words.at(0));
    const auto bra = nanosm::decode(kernel.words.at(1));
    const auto sync = nanosm::decode(kernel.words.at(2));
    require(ssy.opcode == nanosm::Opcode::ssy && ssy.imm == 8 && nanosm::format_instruction(ssy, false) == "ssy +8",
        "ssy: " + nanosm::format_instruction(ssy));
    require(bra.opcode == nanosm::Opcode::bra && bra.control.predicate == 3 && bra.control.predicate_invert &&
            bra.imm == 0 && nanosm::is_divergent_branch(bra), "@!p3 bra: " + nanosm::format_instruction(bra));
    require(sync.opcode == nanosm::Opcode::sync && nanosm::format_instruction(sync, false) == "sync",
        "sync: " + nanosm::format_instruction(sync));
    require(static_cast<unsigned>(nanosm::Opcode::ssy) == 0x2e && static_cast<unsigned>(nanosm::Opcode::sync) == 0x2f,
        "opcodes");
    const auto any = compile("ssy J\nJ: mov r1, 1\nhalt\n");
    require(nanosm::decode(any.words.at(0)).opcode == nanosm::Opcode::ssy, "ssy naming a mov");
    for (const auto& [program, needle] : std::vector<std::pair<std::string, std::string>>{
             {"bssy J\nJ: halt\n", "unknown opcode 'bssy'"},
             {"setp.eq.u32 p0, r0, r0\n@p0 add r1, r0, r0\nhalt\n", "only bra and bra.uni may be predicated"}}) {
        try {
            (void)compile(program);
            throw std::runtime_error("accepted: " + program);
        } catch (const nanosm::SimError& error) {
            require_mentions(error.what(), {needle}, program);
        }
    }
}

} // namespace

// even / odd if-else: results, masks, depth 2, timing
TEST(Divergence, if_else_per_lane) {
    test_if_else_per_lane();
}

// a sync popping its own pc + 8 keeps the fetched words; another pc clears them
TEST(Divergence, sync_to_the_next_pc_keeps_fetched_words) {
    test_sync_to_the_next_pc_keeps_fetched_words();
}

// a nested if; the inner join restores the outer arm's lanes
TEST(Divergence, nested_if) {
    test_nested_if();
}

// all lanes one way: no push
TEST(Divergence, one_sided_branch_pushes_nothing) {
    test_one_sided_branch_pushes_nothing();
}

// lanes do 0..15 iterations; two entries; joined code once
TEST(Divergence, loop_sheds_one_lane_at_a_time) {
    test_loop_sheds_one_lane_at_a_time();
}

// arithmetic, sread, ur2r, SFU, cvt, setp, usetp under 0x0f0f
TEST(Divergence, register_writers_under_a_partial_mask) {
    test_register_writers_under_a_partial_mask();
}

// SFU and arithmetic issue distances do not depend on the mask
TEST(Divergence, fixed_timing_ignores_the_mask) {
    test_fixed_timing_ignores_the_mask();
}

// ldg / stg / lds / sts under 0x00f0
TEST(Divergence, memory_writers_under_a_partial_mask) {
    test_memory_writers_under_a_partial_mask();
}

// lds.uni, bra.uni or halt in an arm: sanitizer panics
TEST(Divergence, uniform_operation_or_halt_in_an_arm_panics) {
    test_uniform_operation_or_halt_in_an_arm_panics();
}

// overflow and an empty-stack sync: machine panics
TEST(Divergence, stack_overflow_and_empty_sync_panic) {
    test_stack_overflow_and_empty_sync_panic();
}

// the profile field; zero entries refuses ssy / sync / @p bra
TEST(Divergence, depth_zero_profile_refuses_divergence) {
    test_depth_zero_profile_refuses_divergence();
}

// parse, encode, print; ssy names any instruction; no bssy; no guards
TEST(Divergence, toolchain_forms) {
    test_toolchain_forms();
}

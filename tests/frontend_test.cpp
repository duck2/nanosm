// The front end: instruction FIFOs, fetch, branch redirect.
//
// Per warp an instruction FIFO of core.instruction_buffer_packets fetch
// packets (2 at canonical; a packet is the two words at an aligned 16-byte
// address); one instruction-memory read of one packet per cycle, fixed at
// one cycle; round-robin fetch over the warps with room, counting the read in
// flight, from registered state only; issuing a packet's second word pops
// it; a branch resolves in the issue register's cycle and a taken one clears
// its warp's FIFO and read in flight. The cycles are in src/timing.hpp.
#include "test_support.hpp"

namespace {

using namespace nanosm_test;

// The fetches granted (debug traces): (cycle, warp, pc).
std::vector<std::tuple<uint64_t, uint32_t, uint32_t>> fetch_records(const nanosm::Machine& machine) {
    std::vector<std::tuple<uint64_t, uint32_t, uint32_t>> fetches;
    for (const auto& event : machine.trace().events()) {
        if (event.category == nanosm::TraceCategory::warp && event.name == "fetch") {
            fetches.emplace_back(event.start_cycle, event.track - 100, static_cast<uint32_t>(event.id));
        }
    }
    return fetches;
}

// Hand assembly as written: every stall is the one written (no block-boundary
// drain on branches), so each scenario's cycles are its own.
std::unique_ptr<nanosm::Machine> machine_as_written(
    const nanosm::DeviceProfile& profile, const std::string& body, uint32_t warps = 1) {
    return debug_machine(profile, nanosm::assemble_asm(hand_asm(profile, body), profile), warps);
}

std::string independent_movs(uint32_t count) {
    std::string body;
    for (uint32_t i = 0; i < count; ++i) body += "mov r" + std::to_string(i % 16) + ", " + std::to_string(i) + "\n";
    return body;
}

} // namespace

// A lone warp of independent instructions from launch. Its first packet is
// fetched at c0 and its first word issues at fetch_to_issue. Each FIFO entry
// is a credit that goes round in fetch_to_issue + 2 cycles carrying two
// words: fetched at f, its words issued at f + 2 and f + 3 at the earliest,
// the packet popped at f + 3's edge and its entry seen by fetch at f + 4
// (fetch reads the registered valid packets, not this cycle's pop). So one
// entry issues two words in every four cycles, two entries (canonical) one
// word per cycle, three or four no faster. With C = min(N, 2) credits doing
// work, instruction k issues at fetch_to_issue + (k / 2C) x 4 + k mod 2C.
TEST(FrontEnd, lone_warp_issue_rate) {
    for (const uint32_t packets : {1u, 2u, 3u, 4u}) {
        auto profile = canonical_profile();
        profile.core_instruction_buffer_packets = packets;
        auto machine = machine_as_written(profile, independent_movs(12) + "halt\n");
        machine->run();
        const auto issues = issue_records(*machine);
        const uint64_t round_trip = nanosm::fetch_to_issue + 2;
        const uint64_t words_per_trip = 2 * std::min<uint64_t>(packets, round_trip / 2);
        require(issues.size() == 13, std::to_string(packets) + " packets: not 13 issues");
        for (uint64_t k = 0; k < issues.size(); ++k) {
            const auto expected = nanosm::fetch_to_issue + k / words_per_trip * round_trip + k % words_per_trip;
            require(issues[k].cycle == expected && issues[k].pc == 8 * k,
                std::to_string(packets) + " packets: instruction " + std::to_string(k) + " did not issue at " +
                    std::to_string(expected) + ":" + issue_log(*machine));
        }
        for (uint32_t reg = 0; reg < 12; ++reg) require(machine->read_reg(0, 3, reg) == reg, "mov data");
    }
}

// Several warps of independent instructions. Fetch serves the warps with room
// in turn and issue takes the heads in turn: warp w's k-th instruction issues
// at fetch_to_issue + w + W x k, one issue per cycle between the W warps
// (with sixteen, each warp needs a packet every 32 cycles: the fetch port has
// room to spare). From launch every warp has room for both its packet
// entries, so the first 2W fetches go round the W warps twice: fetch i at
// ci, warp i mod W, packet 16 x (i / W).
TEST(FrontEnd, warps_interleave) {
    for (const uint32_t warps : {2u, 16u}) {
        const auto& profile = canonical_profile();
        auto machine = machine_as_written(profile, independent_movs(8) + "halt\n", warps);
        machine->run();
        const auto issues = issue_records(*machine);
        require(issues.size() == 9u * warps, std::to_string(warps) + " warps: issue count");
        for (std::size_t i = 0; i < issues.size(); ++i) {
            const auto& [cycle, warp, pc, opcode] = issues[i];
            const uint64_t k = pc / 8;
            require(cycle == nanosm::fetch_to_issue + i && warp == i % warps &&
                    cycle == nanosm::fetch_to_issue + warp + warps * k,
                std::to_string(warps) + " warps: issue " + std::to_string(i) + " at c" + std::to_string(cycle) +
                    " warp " + std::to_string(warp) + " pc " + std::to_string(pc));
        }
        const auto fetches = fetch_records(*machine);
        for (std::size_t i = 0; i < 2u * warps; ++i) {
            const auto [cycle, warp, pc] = fetches.at(i);
            require(cycle == i && warp == i % warps && pc == 16 * (i / warps),
                std::to_string(warps) + " warps: fetch " + std::to_string(i) + " was not round-robin");
        }
        for (uint32_t warp = 0; warp < warps; ++warp) require(machine->read_reg(warp, 0, 7) == 7, "mov data");
    }
}

// Branches resolve in the issue register's cycle (the predicate is read one
// cycle after issue). Words: 0 setp, 1..5 movs, 6 bra.uni (packet 48's first
// half), 7 mov r6, 8 mov r7 and 9 the target mov r8 (packet 64's second
// half), 10 halt; the bra.uni issues at t. Taken: the FIFO and the packet
// landing at t + 1 are dropped -- the redirect wins over an old-path landing,
// although it holds the target -- so the target packet is read again at t + 2
// and the target issues at t + branch_taken_interval. Not taken: the landing
// is kept and the fall-through issues at t + branch_not_taken_interval with
// nothing fetched again. Nothing is fetched while the branch is unresolved.
//
// The taken run's supply state: "waiting for instruction" during launch
// (instruction FIFO empty), at t + 1 (branch unresolved) and at t + 2, t + 3
// (FIFO empty again): 5 warp-cycles; the FIFO-empty observation 4 (packet 48
// is cleared at t + 1's edge). Of the 8 fetches (16 words) 9 issue and 7 are
// discarded.
//
// The other branch forms resolve at the same point: bra and bra.uni @PT are
// taken, bra.uni @!PT is not.
TEST(FrontEnd, conditional_branch) {
    const auto& profile = canonical_profile();
    for (const bool taken : {true, false}) {
        const auto what = std::string(taken ? "taken: " : "not taken: ");
        auto machine = machine_as_written(profile, std::string("setp.eq.u32 p0, r0, ") + (taken ? "0" : "1") + R"(
            mov r1, 1
            mov r2, 2
            mov r3, 3
            mov r4, 4
            mov r5, 5
            bra.uni target {pred=p0, stall=0, set=-, wait=-}
            mov r6, 6
            mov r7, 7
        target: mov r8, 8
            halt
        )");
        constexpr uint32_t bra_pc = 6 * 8, target_pc = 9 * 8, target_packet = target_pc / 16 * 16;
        // The movs keep the warp busy, so the bra.uni issues six cycles after
        // the setp and samples p0 after the setp wrote it.
        const uint64_t t = nanosm::fetch_to_issue + 6;
        require(t + nanosm::issue_to_predicate_sample >=
                    nanosm::fetch_to_issue + nanosm::issue_to_predicate_visible(
                        nanosm::decode(nanosm::assemble_asm(hand_asm(profile, "setp.eq.u32 p0, r0, 0\nhalt"), profile).words[0]),
                        profile),
            "the scenario's bra.uni would sample p0 before the setp wrote it");
        machine->run();
        require(issued_at(*machine, bra_pc) == std::vector<uint64_t>{t}, what + "bra.uni issue:" + issue_log(*machine));
        const auto fetched = fetch_records(*machine);
        const auto fetches_of = [&](uint32_t packet) {
            std::vector<uint64_t> cycles;
            for (const auto& [cycle, warp, fetched_pc] : fetched) if (fetched_pc == packet) cycles.push_back(cycle);
            return cycles;
        };
        for (const auto& [cycle, warp, pc] : fetched) {
            require(cycle != t + nanosm::issue_to_branch_resolve, what + "a fetch was granted while the branch was unresolved");
        }
        if (taken) {
            require(issued_at(*machine, target_pc) == std::vector<uint64_t>{t + nanosm::branch_taken_interval} &&
                    fetches_of(target_packet) ==
                        std::vector<uint64_t>{t, t + nanosm::branch_taken_interval - nanosm::fetch_to_issue},
                what + "the target's packet was not dropped at t + 1 and read again at t + 2, its second half "
                       "issuing at t + 4:" + issue_log(*machine));
            require(issued_at(*machine, bra_pc + 8).empty() && issued_at(*machine, bra_pc + 16).empty() &&
                    machine->read_reg(0, 0, 6) == 0 && machine->read_reg(0, 0, 7) == 0,
                what + "an old-path word, or the first half of the target's packet, had an effect");

            std::vector<std::tuple<uint64_t, uint64_t, std::string>> waits;
            for (const auto& event : machine->trace().events()) {
                if (event.category == nanosm::TraceCategory::warp && event.kind == nanosm::TraceEventKind::slice &&
                    event.name == "waiting for instruction") {
                    waits.emplace_back(event.start_cycle, event.duration_cycles, event.detail);
                }
            }
            const std::vector<std::tuple<uint64_t, uint64_t, std::string>> expected_waits{
                {0, nanosm::fetch_to_issue, "instruction FIFO empty"},
                {t + 1, 1, "branch unresolved"},
                {t + 2, nanosm::fetch_to_issue, "instruction FIFO empty"}};
            require(waits == expected_waits, what + "the supply waits were not traced as launch, branch, refill");
            const auto& summary = machine->trace().summary();
            const auto supply = summary.warp_cycles[static_cast<std::size_t>(nanosm::WarpState::no_instruction)];
            require(supply == 5 && summary.instruction_fifo_empty == 4 && summary.fetches_discarded == 7 &&
                    summary.fetches == 8 && summary.instructions == 9,
                what + "the supply counters: waiting " + std::to_string(supply) + ", FIFO empty " +
                    std::to_string(summary.instruction_fifo_empty) + ", discarded " +
                    std::to_string(summary.fetches_discarded) + ", fetches " + std::to_string(summary.fetches) +
                    ", issued " + std::to_string(summary.instructions));
            const auto text = machine->trace().summary_text();
            require(text.find("  waiting for instruction    5\n") != std::string::npos &&
                    text.find("front end: fetches=8 discarded=7 waiting_for_instruction=5 fifo_empty=4\n") !=
                        std::string::npos,
                what + "the summary text lacks the supply state or the front-end line:\n" + text);
        } else {
            require(issued_at(*machine, bra_pc + 8) == std::vector<uint64_t>{t + nanosm::branch_not_taken_interval} &&
                    issued_at(*machine, bra_pc + 16) == std::vector<uint64_t>{t + nanosm::branch_not_taken_interval + 1},
                what + "the fall-through did not issue at t + 2:" + issue_log(*machine));
            require(fetches_of(bra_pc).size() == 1 && fetches_of(target_packet).size() == 1,
                what + "a kept fall-through packet was fetched again");
            require(machine->read_reg(0, 0, 6) == 6 && machine->read_reg(0, 0, 7) == 7, what + "fall-through data");
        }
        require(machine->read_reg(0, 0, 8) == 8, what + "target data");
    }

    for (const auto& [form, taken] : std::vector<std::pair<std::string, bool>>{
             {"bra target", true}, {"bra.uni target", true}, {"bra.uni target {pred=!PT, stall=0, set=-, wait=-}", false}}) {
        auto machine = machine_as_written(profile, R"(
            mov r1, 1
            mov r2, 2
            mov r3, 3
        )" + form + R"(
            mov r4, 4
        target: mov r5, 5
            halt
        )");
        machine->run();
        const uint64_t t = nanosm::fetch_to_issue + 3;
        require(issued_at(*machine, 3 * 8) == std::vector<uint64_t>{t}, form + ": the branch did not issue at c5");
        const uint32_t next_pc = taken ? 5 * 8 : 4 * 8;
        const auto interval = taken ? nanosm::branch_taken_interval : nanosm::branch_not_taken_interval;
        require(issued_at(*machine, next_pc) == std::vector<uint64_t>{t + interval},
            form + ": the next instruction did not issue at t + " + std::to_string(interval) + ":" + issue_log(*machine));
        require(machine->read_reg(0, 0, 4) == (taken ? 0u : 4u) && machine->read_reg(0, 0, 5) == 5, form + ": data");
    }
}

// Fetch runs ahead of issue: past a taken branch, past the kernel's last
// word. Such words have no effect unless one reaches issue. The kernel's
// last word is a taken branch: the words after it, past the end, are
// fetched (the memory reads zeros there) and cleared by its redirect; the
// halt after `bra skip` is fetched as its fall-through and cleared too (if
// it had issued, r2 would be 0). A warp that really runs past its last word
// is reported when that word reaches issue: "fell off kernel", nothing
// earlier.
TEST(FrontEnd, words_past_the_end) {
    const auto& profile = canonical_profile();
    auto machine = machine_as_written(profile, R"(
            mov r1, 7
            bra skip
        done: halt
        skip: mov r2, 5
            bra done
    )");
    machine->run();
    require(machine->read_reg(0, 0, 1) == 7 && machine->read_reg(0, 0, 2) == 5, "a cleared word had an effect");
    bool past_the_end = false;
    for (const auto& [cycle, warp, pc] : fetch_records(*machine)) past_the_end = past_the_end || pc >= 5 * 8;
    require(past_the_end, "no word past the kernel's end was fetched; the test proves little");
    require(issued_at(*machine, 2 * 8).size() == 1, "the halt did not issue exactly once (from the second branch)");
    require(machine->trace().summary().fetches_discarded >= 3, "the cleared words were not counted");

    auto falls = machine_as_written(profile, "mov r1, 7\nmov r2, 8\n");
    require_mentions(panic_of(*falls, "a warp running past its last word"), {"warp 0 fell off kernel at pc 16"},
        "the fall-off report");
    // The word past the end was fetched two cycles after the first issue and
    // would issue two cycles after that: the report comes then, not at its
    // fetch.
    require(falls->cycle() == nanosm::fetch_to_issue + 2, "the fall-off was not reported as the word reached issue");
}

// Prefetch goes on while the warp waits: through a stall count, a depbar
// wait and a refused head the FIFO fills, so when the wait ends the waiting
// instruction and the next two issue on consecutive cycles, and the warp is
// never "waiting for instruction" after launch.
TEST(FrontEnd, prefetch_through_waits) {
    const auto check = [](nanosm::Machine& machine, uint32_t waiting_pc, uint64_t release, const std::string& what) {
        machine.run();
        for (uint32_t i = 0; i < 3; ++i) {
            require(issued_at(machine, waiting_pc + 8 * i) == std::vector<uint64_t>{release + i},
                what + ": instruction " + std::to_string(i) + " after the wait did not issue at release + " +
                    std::to_string(i) + ":" + issue_log(machine));
        }
        require(machine.trace().summary().warp_cycles[static_cast<std::size_t>(nanosm::WarpState::no_instruction)] ==
                nanosm::fetch_to_issue,
            what + ": the warp waited for an instruction after launch");
    };
    const auto& profile = canonical_profile();
    {   // A stall count of 12 on the first mov.
        auto machine = machine_as_written(profile, "mov r1, 1 {pred=PT, stall=12, set=-, wait=-}\n" +
            independent_movs(4) + "halt\n");
        check(*machine, 8, nanosm::fetch_to_issue + 12, "stall count");
    }
    {   // A depbar: the lds.uni completes dispatch + 1 + D + Lb + 1 + Wd after
        // its issue (its RF write lands); the waiting mov sees the counter at
        // 0 the cycle after.
        auto machine = machine_as_written(profile, R"(
            lds.uni r20, [ur1] {pred=PT, stall=0, set=sb0, wait=-}
            mov r1, 1 {pred=PT, stall=0, set=-, wait=sb0}
        )" + independent_movs(3) + "halt\n");
        const auto release = nanosm::fetch_to_issue + issue_to_shmem_grant(profile) +
            profile.shmem_port_a_depth + profile.shmem_read_latency_cycles + 1 + profile.rf_write_depth_cycles + 1;
        check(*machine, 8, release, "depbar");
    }
    {   // A refused head: one read queue entry; the second lds.uni is refused
        // while the first holds it -- the first is in the queue the cycle
        // after its issue and leaves it then (its uniform operand at the URF
        // output as its collect stage starts), so the second issues two
        // cycles after the first.
        auto one = profile;
        one.shmem_read_queue_entries = 1;
        auto machine = machine_as_written(one, R"(
            lds.uni r20, [ur1] {pred=PT, stall=0, set=sb0, wait=-}
            lds.uni r22, [ur1 + 4] {pred=PT, stall=0, set=sb1, wait=-}
        )" + independent_movs(3) + "halt {pred=PT, stall=0, set=-, wait=sb0|sb1}\n");
        check(*machine, 8, nanosm::fetch_to_issue + 2, "refused head");
    }
}

// setp -> bra.uni (src/timing.hpp): a conditional branch samples its predicate
// issue_to_predicate_sample after its issue, so it may issue that much
// before the setp's predicate is in the file: issue_to_predicate_visible -
// issue_to_predicate_sample after the setp (4 at canonical). qtxc, the hand
// assembler's stall scheduling and the sanitizer all use that distance. At
// it, the branch sees the new predicate; one cycle less is a sanitizer PANIC
// (without the sanitizer the machine quietly reads the old predicate).
TEST(FrontEnd, setp_to_branch_distance) {
    const auto& profile = canonical_profile();
    const auto setp = nanosm::decode(nanosm::assemble_asm(hand_asm(profile, "setp.eq.u32 p0, r0, r0\nhalt"), profile).words[0]);
    const auto distance = nanosm::issue_to_predicate_visible(setp, profile) - nanosm::issue_to_predicate_sample;
    require(distance == 4, "the canonical setp -> bra.uni distance is not 4");
    const auto qtx = qtx_program("setp.eq.u32 p0, r0, r0\n@p0 bra.uni target\nmov r5, 1\ntarget: halt\n", profile);
    require(qtx.instructions[0].control.stall == distance, "qtxc did not place the bra.uni at the distance");
    const auto program = [&](uint32_t stall) {
        return "setp.eq.u32 p0, r0, r0 {pred=PT, stall=" + std::to_string(stall) + R"(, set=-, wait=-}
            bra.uni target {pred=p0, stall=0, set=-, wait=-}
            mov r5, 1
            halt
        target: mov r5, 2
            halt
        )";
    };
    const auto scheduled = hand_assemble(hand_asm(profile, program(0)), profile);
    require(nanosm::decode(scheduled.words[0]).control.stall == distance,
        "the hand assembler's stall scheduling did not raise the setp's stall to the distance");
    auto at = machine_as_written(profile, program(static_cast<uint32_t>(distance)));
    at->run();
    require(at->read_reg(0, 0, 5) == 2, "at the distance the bra.uni did not see the new predicate");
    auto early = machine_as_written(profile, program(static_cast<uint32_t>(distance - 1)));
#if NANOSM_SANITIZER
    expect_panic(*early, {"predicate RAW hazard: missing stall for p0"}, "one cycle early");
#else
    early->run();
    require(early->read_reg(0, 0, 5) == 1, "one cycle early, the machine did not read the old predicate");
#endif
}

// core.warps (16 at canonical) declares the SM's warp slots and is enforced
// at launch, before the RF rows: a one-register kernel launches 16 warps
// and is refused at 17 and 32 (the RF rows alone would accept both).
TEST(FrontEnd, warp_slots_and_profile_fields) {
    const auto& profile = canonical_profile();
    require(profile.core_warps == 16, "the canonical profile does not declare 16 warps");
    const auto kernel = compile("mov r0, 1\nhalt", profile);
    nanosm::Machine(profile, kernel, launch_warps(16)).run();
    for (const uint32_t warps : {17u, 32u}) {
        expect_error([&] { nanosm::Machine machine(profile, kernel, launch_warps(warps)); },
            {"launch of " + std::to_string(warps) + " warps exceeds the SM's 16 warp slots (core.warps)"},
            std::to_string(warps) + " warps on 16 warp slots");
    }
}

// A redirected warp refills behind the warps already asking. Sixteen warps
// start with 'bra target' (packet 0's first half; the target is packet 16's
// first half). Fetch round-robin gives warp w its packet 0 at cw; issue gives
// warp w its branch at fetch_to_issue + w, which resolves the cycle after.
// Warp 0's redirect makes it ask from the next cycle, but the cursor is past
// it: warps 4..15, still empty, get their packet 0 first, and the cursor
// comes back to warp 0 at c16 (its target packet), so its target issues at
// 16 + fetch_to_issue -- far more than branch_taken_interval after the
// resolve. No priority: just the round-robin.
TEST(FrontEnd, redirect_behind_many_fetchers) {
    const auto& profile = canonical_profile();
    auto machine = machine_as_written(profile, R"(
            bra target
            mov r1, 1
        target: mov r2, 2
            halt
    )", 16);
    machine->run();
    const auto fetches = fetch_records(*machine);
    for (uint32_t warp = 0; warp < 16; ++warp) {
        const auto [cycle, w, pc] = fetches.at(warp);
        require(cycle == warp && w == warp && pc == 0, "packet 0 fetches were not c0..c15, warps 0..15");
    }
    const auto [cycle16, warp16, pc16] = fetches.at(16);
    require(cycle16 == 16 && warp16 == 0 && pc16 == 16,
        "warp 0's target packet was not read when the round-robin came back to it at c16");
    require(warp_issue_cycles(*machine, 0, 16) == std::vector<uint64_t>{16 + nanosm::fetch_to_issue},
        "warp 0's target did not issue at c18:" + issue_log(*machine));
    for (uint32_t warp = 0; warp < 16; ++warp) {
        require(machine->read_reg(warp, 0, 1) == 0 && machine->read_reg(warp, 0, 2) == 2,
            "warp " + std::to_string(warp) + ": data");
    }
}

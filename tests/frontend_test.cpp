// The front end: instruction FIFOs, fetch, branch redirect.
// Each test's TEST wrapper is at the end.
#include "test_support.hpp"

namespace {

using namespace nanosm_test;

// ---------------------------------------------------------------------------
// The front end (two-word packets; the cycles in src/timing.hpp).
// Per warp an instruction FIFO of core.instruction_buffer_packets fetch
// packets (2 at canonical; a packet is the two words at an aligned 16-byte
// address); one instruction-memory read of one packet per cycle, fixed at
// one cycle; round-robin fetch over the warps with room, counting the read in
// flight, from registered state only; issuing a packet's second word pops
// it; a branch resolves in the issue register's cycle and a taken one clears
// its warp's FIFO and read in flight.
// ---------------------------------------------------------------------------

// Every issue of the run, in order: (cycle, warp, pc).
std::vector<std::tuple<uint64_t, uint32_t, uint32_t>> i6b2_issues(const nanosm::Machine& machine) {
    std::vector<std::tuple<uint64_t, uint32_t, uint32_t>> issues;
    for (const auto& event : machine.trace().events()) {
        if (!is_issue_event(event)) continue;
        const auto warp = std::stoul(event.detail.substr(5, event.detail.find(',') - 5));
        issues.emplace_back(event.start_cycle, static_cast<uint32_t>(warp), static_cast<uint32_t>(event.id));
    }
    return issues;
}

// The fetches granted (debug traces): (cycle, warp, pc).
std::vector<std::tuple<uint64_t, uint32_t, uint32_t>> i6b2_fetches(const nanosm::Machine& machine) {
    std::vector<std::tuple<uint64_t, uint32_t, uint32_t>> fetches;
    for (const auto& event : machine.trace().events()) {
        if (event.category == nanosm::TraceCategory::warp && event.name == "fetch") {
            fetches.emplace_back(event.start_cycle, event.track - 100, static_cast<uint32_t>(event.id));
        }
    }
    return fetches;
}

// Raw hand assembly by default: every stall is the one written (no
// block-boundary drain on branches), so each scenario's cycles are its own.
// `scheduled`: the fixed-latency stalls raised as qtxc would (hand_assemble).
std::unique_ptr<nanosm::Machine> i6b2_machine(
    const nanosm::DeviceProfile& profile, const std::string& body, uint32_t warps = 1, bool scheduled = false) {
    const auto text = i1_asm(profile, body);
    return i6a_machine(profile, scheduled ? hand_assemble(text, profile) : nanosm::assemble_asm(text, profile), warps);
}

std::string i6b2_independent_movs(uint32_t count) {
    std::string body;
    for (uint32_t i = 0; i < count; ++i) body += "mov r" + std::to_string(i % 16) + ", " + std::to_string(i) + "\n";
    return body;
}

// A lone warp of independent instructions from launch. Its first packet is
// fetched at c0 and its first word issues at fetch_to_issue (c2). Each FIFO
// entry is a credit that goes round in fetch_to_issue + 2 cycles carrying two
// words: fetched at f, its words issued at f + 2 and f + 3 at the earliest,
// the packet popped at f + 3's edge and its entry seen by fetch at f + 4
// (fetch reads the registered valid packets, not this cycle's pop). So N
// entries carry 2N words per 4 cycles: one entry issues two words in every
// four cycles, two entries (canonical) one word per cycle from c2, three or
// four no faster. With C = min(N, 2) credits doing work, instruction k issues
// at fetch_to_issue + (k / 2C) x 4 + k mod 2C.
void test_lone_warp_issue_rate() {
    for (const uint32_t packets : {1u, 2u, 3u, 4u}) {
        auto profile = canonical_profile();
        profile.core_instruction_buffer_packets = packets;
        auto machine = i6b2_machine(profile, i6b2_independent_movs(12) + "halt\n");
        machine->run();
        const auto issues = i6b2_issues(*machine);
        const uint64_t round_trip = nanosm::fetch_to_issue + 2;
        const uint64_t words_per_trip = 2 * std::min<uint64_t>(packets, round_trip / 2);
        require(issues.size() == 13, std::to_string(packets) + " packets: not 13 issues");
        for (uint64_t k = 0; k < issues.size(); ++k) {
            const auto expected = nanosm::fetch_to_issue + k / words_per_trip * round_trip + k % words_per_trip;
            require(std::get<0>(issues[k]) == expected && std::get<2>(issues[k]) == 8 * k,
                std::to_string(packets) + " packets: instruction " + std::to_string(k) + " did not issue at " +
                    std::to_string(expected) + ":" + issue_log(*machine));
        }
        for (uint32_t reg = 0; reg < 12; ++reg) require(machine->read_reg(0, 3, reg) == reg, "mov data");
    }
}

// Several warps of independent instructions. Fetch serves the warps with room
// in turn and issue takes the heads in turn: two warps alternate from c2
// (warp w's k-th instruction at fetch_to_issue + w + 2k), one issue per
// cycle between them. Sixteen warps: warp w's first packet is fetched at cw
// and its first word issues at fetch_to_issue + w; every warp then issues
// once in sixteen cycles, k-th at fetch_to_issue + w + 16k, the machine
// issuing every cycle (each warp needs a packet every 32 cycles: the fetch
// port has room to spare). From launch every warp has room for both its
// packet entries, so the first 2W fetches go round the W warps twice: fetch
// i at ci, warp i mod W, packet 16 x (i / W).
void test_warps_interleave() {
    for (const uint32_t warps : {2u, 16u}) {
        const auto& profile = canonical_profile();
        auto machine = i6b2_machine(profile, i6b2_independent_movs(8) + "halt\n", warps);
        machine->run();
        const auto issues = i6b2_issues(*machine);
        require(issues.size() == 9u * warps, std::to_string(warps) + " warps: issue count");
        for (std::size_t i = 0; i < issues.size(); ++i) {
            const auto [cycle, warp, pc] = issues[i];
            const uint64_t k = pc / 8;
            require(cycle == nanosm::fetch_to_issue + i && warp == i % warps &&
                    cycle == nanosm::fetch_to_issue + warp + warps * k,
                std::to_string(warps) + " warps: issue " + std::to_string(i) + " at c" + std::to_string(cycle) +
                    " warp " + std::to_string(warp) + " pc " + std::to_string(pc));
        }
        const auto fetches = i6b2_fetches(*machine);
        for (std::size_t i = 0; i < 2u * warps; ++i) {
            const auto [cycle, warp, pc] = fetches.at(i);
            require(cycle == i && warp == i % warps && pc == 16 * (i / warps),
                std::to_string(warps) + " warps: fetch " + std::to_string(i) + " was not round-robin");
        }
        for (uint32_t warp = 0; warp < warps; ++warp) require(machine->read_reg(warp, 0, 7) == 7, "mov data");
    }
}

// Fetch round-robin with a hot warp and three waiting warps (core.warps 16,
// launched 4). Warp 0 runs thirty independent movs; warps 1..3 sit on a
// stall count of 31 twice. Fetch ignores the waits: while the waiters' FIFOs
// have room the grants go round all four (the hot warp gets one fetch in
// four and issues at that rate); a waiter whose FIFO is full stops asking.
// Once all three are full the hot warp gets every fetch and, within
// fetch_to_issue of that, issues one per cycle until a waiter's stall ends.
// No priority policy: just the round-robin over warps with room.
// Two-word packets: before their branches the waiters' FIFOs are already
// full of their fall-through (prefetch runs through the setp -> bra.uni
// wait), so what counts is after the redirects: from the first waiter's
// first request (its branch's resolve + 1) to the first cycle after the
// last waiter's resolve in which all three are full. In that window each
// waiter is granted its two packet entries, with the hot warp's turns
// between: at least 2 x 3 + 1 grants, in rotation. (Since the warp id is read
// with usread + ur2r, the redirects fall where the waiters' first movs take
// issue slots from the hot warp, whose FIFO is then full for a few cycles:
// the rotation passes over it there, and only there -- a warp is passed over
// only while its packets and read in flight fill its FIFO.)
void test_fetch_round_robin_hot_and_waiting_warps() {
    const auto& profile = canonical_profile();
    const auto depth = profile.core_instruction_buffer_packets;
    auto machine = i6b2_machine(profile, R"(
        usread ur1, WARP_ID
        ur2r r29, ur1
        setp.ne.u32 p0, r29, 0
        bra.uni waiter {pred=p0, stall=0, set=-, wait=-}
    )" + i6b2_independent_movs(30) + R"(
        halt
    waiter: mov r20, 1 {pred=PT, stall=31, set=-, wait=-}
        mov r21, 2 {pred=PT, stall=31, set=-, wait=-}
        mov r22, 3
        halt
    )", 4, true);
    std::map<uint64_t, bool> waiters_full; // per cycle (Q): every waiter's FIFO full
    std::map<uint64_t, std::array<bool, 4>> room; // per cycle (Q): packets + read in flight below the depth
    while (!machine->done()) {
        waiters_full[machine->cycle()] = machine->instruction_fifo_packets(1) == depth &&
            machine->instruction_fifo_packets(2) == depth && machine->instruction_fifo_packets(3) == depth;
        for (uint32_t warp = 0; warp < 4; ++warp) {
            room[machine->cycle()][warp] = machine->instruction_fifo_packets(warp) +
                    (machine->fetch_in_flight() == std::optional<uint32_t>(warp) ? 1 : 0) < depth;
        }
        tick_capped(*machine);
    }
    const auto fetches = i6b2_fetches(*machine);
    const auto issues = i6b2_issues(*machine);
    // The waiters' branches (pc 24), their first stalled mov (pc 35 * 8:
    // after usread, ur2r, setp, bra.uni, thirty movs and the halt) and their
    // second.
    uint64_t first_branch = std::numeric_limits<uint64_t>::max();
    uint64_t last_branch = 0;
    uint64_t waiter_start = 0;
    uint64_t waiter_resume = std::numeric_limits<uint64_t>::max();
    for (const auto& [cycle, warp, pc] : issues) {
        if (warp != 0 && pc == 24) {
            first_branch = std::min(first_branch, cycle);
            last_branch = std::max(last_branch, cycle);
        }
        if (warp == 3 && pc == 35 * 8) waiter_start = cycle;
        if (warp != 0 && pc == 36 * 8) waiter_resume = std::min(waiter_resume, cycle);
    }
    require(waiter_start != 0 && waiter_resume != std::numeric_limits<uint64_t>::max(), "the waiters did not run");
    std::optional<uint64_t> full;
    for (const auto& [cycle, all] : waiters_full) {
        if (!full.has_value() && all && cycle > last_branch + nanosm::issue_to_branch_resolve) full = cycle;
    }
    require(full.has_value(), "the waiters' FIFOs never filled after their branches");
    // While the warps ask, the grants rotate w0, w1, w2, w3, passing over
    // only a warp with no room.
    std::vector<std::pair<uint64_t, uint32_t>> rotation; // (cycle, warp)
    for (const auto& [cycle, warp, pc] : fetches) {
        if (cycle > first_branch + nanosm::issue_to_branch_resolve && cycle < *full) rotation.emplace_back(cycle, warp);
    }
    require(rotation.size() >= 2 * 3 + 1, "the window where all four warps ask is too short to show the rotation");
    for (std::size_t i = 1; i < rotation.size(); ++i) {
        const auto [cycle, warp] = rotation[i];
        for (auto passed = (rotation[i - 1].second + 1) % 4; passed != warp; passed = (passed + 1) % 4) {
            require(!room.at(cycle)[passed], "fetch did not rotate over the warps with room: warp " +
                std::to_string(passed) + " was passed over at c" + std::to_string(cycle) + issue_log(*machine));
        }
    }
    // From then until the waiters ask again (their stall ends), every fetch
    // is the hot warp's, and it issues every cycle from full + 1.
    for (const auto& [cycle, warp, pc] : fetches) {
        if (cycle >= *full && cycle < waiter_resume) require(warp == 0, "a full waiter was granted a fetch");
    }
    std::vector<uint64_t> hot;
    for (const auto& [cycle, warp, pc] : issues) {
        if (warp == 0 && cycle > *full && cycle < waiter_resume) hot.push_back(cycle);
    }
    require(hot.size() >= 10 && hot.front() <= *full + nanosm::fetch_to_issue,
        "the hot warp did not issue within fetch_to_issue of the waiters' FIFOs filling");
    for (std::size_t i = 1; i < hot.size(); ++i) {
        require(hot[i] == hot[i - 1] + 1, "the hot warp lost a cycle while the waiters were full:" + issue_log(*machine));
    }
    for (uint32_t warp = 1; warp < 4; ++warp) require(machine->read_reg(warp, 0, 22) == 3, "waiter data");
    require(machine->read_reg(0, 0, 13) == 29, "hot warp data");
}

// Conditional branches (the predicate is read one cycle after
// issue), with two-word packets. Words: 0 setp, 1..5 movs, 6 bra.uni (the
// first half of packet 48), 7 mov r6 (its second half), 8 mov r7 and 9 the
// target mov r8 (packet 64: the target is a second half), 10 halt. A lone
// warp issues one word per cycle from c2 (two packet entries), so the
// bra.uni issues at t = c8. Its fetches: packet 0 at c0, 16 at c1, then each
// packet the cycle after the one two ahead pops (32 at c4, 48 at c6, 64 at
// c8 = t, granted from registered state as the bra.uni issues). So at t + 1
// (Q after t): packet 48 in the FIFO with pc at its second half (56),
// packet 64 in flight, fetch pc 80. The branch resolves at t + 1:
//   taken:      the FIFO (packet 48, whose second half was never issued) is
//               cleared and packet 64 landing at t + 1 is dropped -- a
//               redirect and an old-path landing in one cycle, the redirect
//               wins -- although it holds the target: pc takes the target
//               (72) and fetch pc its packet (64), which is read again at
//               t + 2 and its second half issues at t + 4
//               (branch_taken_interval). Its first half (mov r7) never
//               issues, nor does the cleared mov r6.
//   not taken:  the landing joins the kept packet 48; pc + 8 (its second
//               half) issues at t + 2 (branch_not_taken_interval) and packet
//               64's first half at t + 3; nothing is fetched again.
// Nothing of the warp issues or is fetched at t + 1.
void test_conditional_branch() {
    const auto& profile = canonical_profile();
    for (const bool taken : {true, false}) {
        const auto what = std::string(taken ? "taken: " : "not taken: ");
        auto machine = i6b2_machine(profile, std::string("setp.eq.u32 p0, r0, ") + (taken ? "0" : "1") + R"(
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
        // setp at c2 (its p0 in the file from c2 + 5); the movs keep the warp
        // busy, so the bra.uni issues at c8 and samples p0 at c9.
        const uint64_t t = nanosm::fetch_to_issue + 6;
        require(t + nanosm::issue_to_predicate_sample >=
                    nanosm::fetch_to_issue + nanosm::issue_to_predicate_visible(
                        nanosm::decode(nanosm::assemble_asm(i1_asm(profile, "setp.eq.u32 p0, r0, 0\nhalt"), profile).words[0]),
                        profile),
            "the scenario's bra.uni would sample p0 before the setp wrote it");
        std::map<uint64_t, std::tuple<std::size_t, std::optional<uint32_t>, uint32_t, uint32_t>> after; // Q after each tick
        while (!machine->done()) {
            const auto cycle = machine->cycle();
            tick_capped(*machine);
            after[cycle] = {machine->instruction_fifo_packets(0), machine->fetch_in_flight(), machine->warp_pc(0),
                machine->warp_fetch_pc(0)};
        }
        require(issued_at(*machine, bra_pc) == std::vector<uint64_t>{t}, what + "bra.uni issue:" + issue_log(*machine));
        // At t + 1 (Q after t): packet 48 in the FIFO, packet 64 in flight.
        require(after.at(t) == std::tuple<std::size_t, std::optional<uint32_t>, uint32_t, uint32_t>{
                    1, 0u, bra_pc + 8, bra_pc + 32},
            what + "the FIFO and the read in flight at the resolve");
        const auto fetched = i6b2_fetches(*machine);
        const auto fetches_of = [&](uint32_t packet) {
            std::vector<uint64_t> cycles;
            for (const auto& [cycle, warp, fetched_pc] : fetched) if (fetched_pc == packet) cycles.push_back(cycle);
            return cycles;
        };
        for (const auto& [cycle, warp, pc] : fetched) {
            require(cycle != t + nanosm::issue_to_branch_resolve, what + "a fetch was granted while the branch was unresolved");
        }
        if (taken) {
            require(after.at(t + 1) == std::tuple<std::size_t, std::optional<uint32_t>, uint32_t, uint32_t>{
                        0, std::nullopt, target_pc, target_packet},
                what + "the redirect did not clear the FIFO and the read in flight, or set the pcs");
            require(issued_at(*machine, target_pc) == std::vector<uint64_t>{t + nanosm::branch_taken_interval} &&
                    fetches_of(target_packet) ==
                        std::vector<uint64_t>{t, t + nanosm::branch_taken_interval - nanosm::fetch_to_issue},
                what + "the target's packet was not dropped at t + 1 and read again at t + 2, its second half "
                       "issuing at t + 4:" + issue_log(*machine));
            require(issued_at(*machine, bra_pc + 8).empty() && issued_at(*machine, bra_pc + 16).empty() &&
                    machine->read_reg(0, 0, 6) == 0 && machine->read_reg(0, 0, 7) == 0,
                what + "an old-path word, or the first half of the target's packet, had an effect");
        } else {
            require(after.at(t + 1) == std::tuple<std::size_t, std::optional<uint32_t>, uint32_t, uint32_t>{
                        2, std::nullopt, bra_pc + 8, bra_pc + 32},
                what + "the buffered fall-through was not kept (with the packet landing at the resolve)");
            require(issued_at(*machine, bra_pc + 8) == std::vector<uint64_t>{t + nanosm::branch_not_taken_interval} &&
                    issued_at(*machine, bra_pc + 16) == std::vector<uint64_t>{t + nanosm::branch_not_taken_interval + 1},
                what + "the fall-through did not issue at t + 2:" + issue_log(*machine));
            require(fetches_of(bra_pc).size() == 1 && fetches_of(target_packet).size() == 1,
                what + "a kept fall-through packet was fetched again");
            require(machine->read_reg(0, 0, 6) == 6 && machine->read_reg(0, 0, 7) == 7, what + "fall-through data");
        }
        require(machine->read_reg(0, 0, 8) == 8, what + "target data");
    }
}

// The other branch forms resolve at the same point (implementer choice: one
// rule for every branch, one target adder): bra and bra.uni @PT are taken
// (the next instruction at t + branch_taken_interval), bra.uni @!PT is not
// (t + branch_not_taken_interval).
void test_unconditional_branches() {
    const auto& profile = canonical_profile();
    for (const auto& [form, taken] : std::vector<std::pair<std::string, bool>>{
             {"bra target", true}, {"bra.uni target", true}, {"bra.uni target {pred=!PT, stall=0, set=-, wait=-}", false}}) {
        auto machine = i6b2_machine(profile, R"(
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
void test_words_past_the_end() {
    const auto& profile = canonical_profile();
    auto machine = i6b2_machine(profile, R"(
            mov r1, 7
            bra skip
        done: halt
        skip: mov r2, 5
            bra done
    )");
    machine->run();
    require(machine->read_reg(0, 0, 1) == 7 && machine->read_reg(0, 0, 2) == 5, "a cleared word had an effect");
    bool past_the_end = false;
    for (const auto& [cycle, warp, pc] : i6b2_fetches(*machine)) past_the_end = past_the_end || pc >= 5 * 8;
    require(past_the_end, "no word past the kernel's end was fetched; the test proves little");
    require(issued_at(*machine, 2 * 8).size() == 1, "the halt did not issue exactly once (from the second branch)");
    require(machine->trace().summary().fetches_discarded >= 3, "the cleared words were not counted");

    auto falls = i6b2_machine(profile, "mov r1, 7\nmov r2, 8\n");
    try {
        falls->run();
        throw std::runtime_error("a warp ran past its last word without a report");
    } catch (const nanosm::SimError& error) {
        require(std::string(error.what()).find("warp 0 fell off kernel at pc 16") != std::string::npos,
            std::string("wrong fall-off report: ") + error.what());
    }
    // The word past the end was fetched at c2 and would issue at c4: the
    // report comes then, not at its fetch.
    require(falls->cycle() == nanosm::fetch_to_issue + 2, "the fall-off was not reported as the word reached issue");
}

// Prefetch goes on while the warp waits: through a stall count, a depbar
// wait and a refused head, the FIFO fills (both packet entries, Q: the
// waiting word's packet and the next), so when the wait ends the waiting
// instruction and the next two issue on consecutive cycles, and the warp is
// never "waiting for instruction" after its first two cycles.
void test_prefetch_through_waits() {
    const auto check = [](nanosm::Machine& machine, uint32_t waiting_pc, uint64_t release, const std::string& what) {
        const auto depth = canonical_profile().core_instruction_buffer_packets;
        std::map<uint64_t, std::size_t> packets; // the FIFO's valid packets (Q) in each cycle
        while (!machine.done()) {
            packets[machine.cycle()] = machine.instruction_fifo_packets(0);
            tick_capped(machine);
        }
        require(packets.at(release) == depth, what + ": the FIFO was not full when the wait ended");
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
    {   // A stall count of 12 on the first mov (c2): the next issues at c14.
        auto machine = i6b2_machine(profile, "mov r1, 1 {pred=PT, stall=12, set=-, wait=-}\n" +
            i6b2_independent_movs(4) + "halt\n");
        check(*machine, 8, nanosm::fetch_to_issue + 12, "stall count");
    }
    {   // A depbar: the lds.uni at c2 completes at c2 + dispatch + 1 + D + Lb
        // + 1 + Wd (its RF write lands); the waiting mov sees the counter at 0
        // the cycle after.
        auto machine = i6b2_machine(profile, R"(
            lds.uni r20, [ur1] {pred=PT, stall=0, set=sb0, wait=-}
            mov r1, 1 {pred=PT, stall=0, set=-, wait=sb0}
        )" + i6b2_independent_movs(3) + "halt\n");
        const auto release = nanosm::fetch_to_issue + nanosm::issue_to_dispatch(profile) + 1 +
            profile.shmem_port_a_depth + profile.shmem_read_latency_cycles + 1 + profile.rf_write_depth_cycles + 1;
        check(*machine, 8, release, "depbar");
    }
    {   // A refused head: one load slot; the second lds.uni (c3) is refused
        // until the first's RF write frees the slot (the
        // cycle after its grant, 2 + i1_lds_shift cycles after the refusal
        // starts).
        auto one = profile;
        one.shmem_load_slots = 1;
        auto machine = i6b2_machine(one, R"(
            lds.uni r20, [ur1] {pred=PT, stall=0, set=sb0, wait=-}
            lds.uni r22, [ur1 + 4] {pred=PT, stall=0, set=sb1, wait=-}
        )" + i6b2_independent_movs(3) + "halt {pred=PT, stall=0, set=-, wait=sb0|sb1}\n");
        const auto release = nanosm::fetch_to_issue + 1 + 2 + i1_lds_shift(one);
        check(*machine, 8, release, "refused head");
        require(refusals(*machine, 0, "load slots full") ==
                std::vector<std::pair<uint64_t, uint64_t>>{{nanosm::fetch_to_issue + 1, 2 + i1_lds_shift(one)}},
            "refused head: the second lds.uni was not refused from c3 until the slot freed");
    }
}

// setp -> bra.uni (src/timing.hpp): a conditional branch samples its predicate
// issue_to_predicate_sample after its issue, so it may issue that much
// before the setp's predicate is in the file: issue_to_predicate_visible -
// issue_to_predicate_sample after the setp (4 at canonical; 5 before the
// front end). qtxc, the hand assembler's stall scheduling and the sanitizer
// all use that distance. At it, the branch sees the new predicate; one cycle
// less is a sanitizer PANIC (without the sanitizer the machine quietly reads
// the old predicate).
void test_setp_to_branch_distance() {
    const auto& profile = canonical_profile();
    const auto setp = nanosm::decode(nanosm::assemble_asm(i1_asm(profile, "setp.eq.u32 p0, r0, r0\nhalt"), profile).words[0]);
    const auto distance = nanosm::issue_to_predicate_visible(setp, profile) - nanosm::issue_to_predicate_sample;
    require(distance == 1 + profile.rf_read_latency_cycles + profile.timing.alu + 1 - 1 && distance == 4,
        "the canonical setp -> bra.uni distance is not 4");
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
    const auto scheduled = hand_assemble(i1_asm(profile, program(0)), profile);
    require(nanosm::decode(scheduled.words[0]).control.stall == distance,
        "the hand assembler's stall scheduling did not raise the setp's stall to the distance");
    auto at = i6b2_machine(profile, program(static_cast<uint32_t>(distance)));
    at->run();
    require(at->read_reg(0, 0, 5) == 2, "at the distance the bra.uni did not see the new predicate");
    auto early = i6b2_machine(profile, program(static_cast<uint32_t>(distance - 1)));
#if NANOSM_SANITIZER
    require_mentions(i4_panic(*early, "one cycle early"), {"predicate RAW hazard: missing stall for p0"},
        "one cycle early");
#else
    early->run();
    require(early->read_reg(0, 0, 5) == 1, "one cycle early, the machine did not read the old predicate");
#endif
}

// A loop head (src/qtxc/schedule.cpp LoopEntry): what the code before a
// loop leaves pending is paid once, by the instruction before the head, not
// by a stall inside the loop on every iteration (the branch at the top of
// this loop would otherwise carry the ex2's drain).
// Both passes. The loop body reads the ex2's result on the first pass, so
// the sanitizer checks the fall-through edge; later passes arrive drained.
void test_loop_head_pays_the_entry_once() {
    const auto& profile = canonical_profile();
    const std::string source = R"(
            umov ur1, 0
            ex2 r2, r1
        LOOP:
        @p3 bra.uni SKIP
            fadd r3, r2, r2
        SKIP:
            uaddi ur1, ur1, 1
            usetp.lt.u32 p2, ur1, 4
        @p2 bra.uni LOOP
            halt
    )";
    const auto qtx = qtx_program(source, profile);
    require(qtx.instructions[1].control.stall > 1, "qtxc did not pay the ex2 before the loop head");
    require(qtx.instructions[2].control.stall == 0, "qtxc's branch at the loop head still carries the ex2");
    const auto hand = hand_assemble(i1_asm(profile, R"(
            umov ur1, 0
            ex2 r2, r1
        loop: bra.uni skip {pred=p3, stall=0, set=-, wait=-}
            fadd r3, r2, r2
        skip: uaddi ur1, ur1, 1
            usetp.lt.u32 p2, ur1, 4
            bra.uni loop {pred=p2, stall=0, set=-, wait=-}
            halt
    )"), profile);
    require(nanosm::decode(hand.words[1]).control.stall == qtx.instructions[1].control.stall &&
            nanosm::decode(hand.words[2]).control.stall == 0,
        "the hand assembler's stall scheduling does not pay the ex2 before the loop head as qtxc does");
    for (const auto& kernel : {compile(source, profile), hand}) {
        auto machine = i6a_machine(profile, kernel);
        machine->run();
        require(machine->read_reg(0, 0, 3) == 0x40000000u, "the loop did not compute ex2(0) + ex2(0) = 2");
    }
}

// core.warps (16 at canonical) declares the SM's warp slots and is enforced
// at launch, before the RF rows: a one-register kernel launches 16 warps
// and is refused at 17 and 32 (the RF rows alone would accept both).
// core.instruction_buffer_packets (2) is the FIFO depth in
// two-word packets. Both are optional in the JSON (absent: 16, 2), checked
// (warps 1..64, packets 1..16) and part of the fingerprint.
void test_warp_slots_and_profile_fields() {
    const auto& profile = canonical_profile();
    require(profile.core_warps == 16 && profile.core_instruction_buffer_packets == 2,
        "the canonical profile does not declare 16 warps and 2 instruction packets");
    const auto kernel = compile("mov r0, 1\nhalt", profile);
    for (const uint32_t warps : {16u, 17u, 32u}) {
        nanosm::Launch launch;
        launch.active_warps = warps;
        try {
            nanosm::Machine machine(profile, kernel, std::move(launch));
            require(warps == 16, std::to_string(warps) + " warps were accepted on 16 warp slots");
            machine.run();
        } catch (const nanosm::SimError& error) {
            require(warps != 16 && std::string(error.what()).find("launch of " + std::to_string(warps) +
                    " warps exceeds the SM's 16 warp slots (core.warps)") != std::string::npos,
                std::string("warp slots: ") + error.what());
        }
    }
    std::ifstream input("profiles/nanosm-v0.json");
    nlohmann::json document;
    input >> document;
    for (const auto& [field, canonical, bad] : std::vector<std::tuple<std::string, uint32_t, std::vector<int>>>{
             {"warps", 16, {0, 65, -1}}, {"instruction_buffer_packets", 2, {0, 17, -1}}}) {
        auto defaulted = document;
        defaulted["core"].erase(field);
        require(nanosm::device_profile_fingerprint(nanosm::parse_device_profile(defaulted.dump())) ==
                nanosm::device_profile_fingerprint(profile),
            field + ": the default is not the canonical " + std::to_string(canonical));
        for (const auto value : bad) {
            auto modified = document;
            modified["core"][field] = value;
            try {
                (void)nanosm::parse_device_profile(modified.dump());
                throw std::runtime_error(field + " = " + std::to_string(value) + " was accepted");
            } catch (const nanosm::ProfileError&) {}
        }
        auto other = document;
        other["core"][field] = canonical + 1;
        require(nanosm::device_profile_fingerprint(nanosm::parse_device_profile(other.dump())) !=
                nanosm::device_profile_fingerprint(profile),
            field + " was omitted from the profile fingerprint");
    }
    auto zero = profile;
    zero.core_instruction_buffer_packets = 0;
    try {
        nanosm::validate_device_profile(zero);
        throw std::runtime_error("a zero-packet instruction FIFO passed validation");
    } catch (const nanosm::ProfileError&) {}
}

// The trace's instruction-supply state: "waiting for instruction" -- detail
// "instruction FIFO empty" or "branch unresolved" -- apart from the
// dependency and resource waits, counted in the summary's warp cycles and
// its front-end line (fetches, discarded words, those warp-cycles, and the
// FIFO-empty observation). A taken branch at t = c8 (the conditional-branch
// scenario): launch c0..c1 FIFO empty, t + 1 branch unresolved, t + 2 .. t + 3
// FIFO empty: 5 warp-cycles waiting. The FIFO itself is empty at c0, c1,
// t + 2, t + 3 but not at t + 1 (packet 48 is cleared at t + 1's edge): 4.
// Words discarded (the fetched words that never issue): at the redirect,
// packet 48's unissued second half (1) and the landing packet 64 (2); packet
// 64 read again for the target (its second half), its first half (mov r7)
// skipped as it lands at t + 3 (1); at the halt (the first half of packet 80,
// at t + 5) its unissued second half (1), and packet 96, granted in the
// halt's cycle and dropped as it lands (2): 7. The same from the totals: 8
// fetches (packets 0, 16, 32, 48, 64, 64, 80, 96) of 16 words, 9 issued.
void test_trace_supply_state() {
    const auto& profile = canonical_profile();
    auto machine = i6b2_machine(profile, R"(
        setp.eq.u32 p0, r0, 0
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
    machine->run();
    const uint64_t t = nanosm::fetch_to_issue + 6;
    std::vector<std::tuple<uint64_t, uint64_t, std::string>> waits;
    for (const auto& event : machine->trace().events()) {
        if (event.category == nanosm::TraceCategory::warp && event.kind == nanosm::TraceEventKind::slice &&
            event.name == "waiting for instruction") {
            waits.emplace_back(event.start_cycle, event.duration_cycles, event.detail);
        }
    }
    const std::vector<std::tuple<uint64_t, uint64_t, std::string>> expected{
        {0, nanosm::fetch_to_issue, "instruction FIFO empty"},
        {t + 1, 1, "branch unresolved"},
        {t + 2, nanosm::fetch_to_issue, "instruction FIFO empty"}};
    require(waits == expected, "the supply waits were not traced as launch, branch, refill");
    const auto& summary = machine->trace().summary();
    const auto supply = summary.warp_cycles[static_cast<std::size_t>(nanosm::WarpState::no_instruction)];
    require(supply == 5 && summary.instruction_fifo_empty == 4 && summary.fetches_discarded == 7 &&
            summary.fetches == 8 && summary.instructions == 9,
        "the supply counters: waiting " + std::to_string(supply) + ", FIFO empty " +
            std::to_string(summary.instruction_fifo_empty) + ", discarded " +
            std::to_string(summary.fetches_discarded) + ", fetches " + std::to_string(summary.fetches) +
            ", issued " + std::to_string(summary.instructions));
    require(nanosm::warp_state_name(nanosm::WarpState::no_instruction) == "waiting for instruction",
        "the state's name");
    const auto text = machine->trace().summary_text();
    require(text.find("  waiting for instruction    5\n") != std::string::npos &&
            text.find("front end: fetches=8 discarded=7 waiting_for_instruction=5 fifo_empty=4\n") !=
                std::string::npos,
        "the summary text lacks the supply state or the front-end line:\n" + text);
}

// Every warp's trace state comes from its own registers, however far the
// scheduler scanned (two warps, mov, mov, halt). Warp 0's
// first packet is fetched at c0 and warp 1's at c1; warp 0 issues at c2,
// warp 1 at c3, then they alternate. At c2 the scheduler stops at warp 0,
// the winner, but warp 1's FIFO is empty (its packet lands at c2's edge): it
// is "waiting for instruction" at c0, c1 and c2, warp 0 at c0 and c1 -- 5
// warp-cycles, and the FIFO-empty observation is 5 too. A warp past the
// winner with an instruction is "not examined (another warp issued)", never
// "ready": warp 0 at c3 and c5, warp 1 at c4 and c6 (warp 0's halt at c6,
// warp 1's at c7).
void test_trace_counts_every_warp() {
    const auto& profile = canonical_profile();
    auto machine = i6b2_machine(profile, "mov r0, 1\nmov r1, 2\nhalt\n", 2);
    machine->run();
    std::map<uint32_t, std::vector<std::pair<uint64_t, uint64_t>>> waits;
    std::map<uint32_t, std::vector<uint64_t>> not_examined;
    for (const auto& event : machine->trace().events()) {
        if (event.category != nanosm::TraceCategory::warp || event.kind != nanosm::TraceEventKind::slice) continue;
        const auto warp = static_cast<uint32_t>(event.track - 100);
        if (event.name == "waiting for instruction") waits[warp].emplace_back(event.start_cycle, event.duration_cycles);
        if (event.name == "not examined (another warp issued)") {
            for (uint64_t c = 0; c < event.duration_cycles; ++c) not_examined[warp].push_back(event.start_cycle + c);
        }
    }
    require(waits[0] == std::vector<std::pair<uint64_t, uint64_t>>{{0, 2}} &&
            waits[1] == std::vector<std::pair<uint64_t, uint64_t>>{{0, 3}},
        "warp 1's empty FIFO at c2 (past the winner) was not traced as waiting for instruction");
    require(not_examined[0] == std::vector<uint64_t>{3, 5} && not_examined[1] == std::vector<uint64_t>{4, 6},
        "the warps past the winner were not traced as not examined:" + issue_log(*machine));
    const auto& summary = machine->trace().summary();
    require(summary.warp_cycles[static_cast<std::size_t>(nanosm::WarpState::no_instruction)] == 5 &&
            summary.instruction_fifo_empty == 5 &&
            summary.warp_cycles[static_cast<std::size_t>(nanosm::WarpState::not_examined)] == 4,
        "the summary undercounts the warps past the winner");
    // Its summary line: the name is wider than the column, a space still
    // separates it from the count.
    const auto text = machine->trace().summary_text();
    require(text.find("\n  not examined (another warp issued) 4\n") != std::string::npos,
        "the summary's not-examined line has no space before its count:\n" + text);
    require(machine->read_reg(1, 0, 1) == 2, "data");
}

// Discarded words are the fetched words that never issue, including the
// first half of a packet fetched for a
// second-half branch target, skipped as it lands. `bra target`, `target:
// halt`: the target is word 1, packet 0's second half. Packet 0 is fetched at
// c0 and packet 16 at c1; the bra issues at t = c2 (packet 16 lands behind
// packet 0) and its redirect at t + 1 clears packet 0's second half and
// packet 16 (3). Packet 0 is read again at t + 2 and packet 16 at t + 3;
// packet 0 lands at t + 3 and its first half (the bra) is skipped (1); the
// halt issues at t + 4 and packet 16 is dropped as it lands (2): 6, as the
// totals say: 4 fetches of 8 words, 2 issued. With a mov before the target
// (word 2, packet 16's first half) nothing is skipped: the redirect clears 3,
// packet 16 is read again at t + 2 and packet 32 at t + 3, the halt issues
// at t + 4 with its packet's second half unissued (1) and packet 32 is
// dropped as it lands (2): 6 as well.
void test_skipped_first_half_is_discarded() {
    const auto& profile = canonical_profile();
    const uint64_t t = nanosm::fetch_to_issue;
    for (const auto& [body, target_packet, what] : std::vector<std::tuple<std::string, uint32_t, std::string>>{
             {"bra target\ntarget: halt\n", 0, "second-half target: "},
             {"bra target\nmov r0, 0\ntarget: halt\n", 16, "first-half target: "}}) {
        auto machine = i6b2_machine(profile, body);
        machine->run();
        const std::vector<std::tuple<uint64_t, uint32_t, uint32_t>> fetches{
            {0, 0, 0}, {1, 0, 16}, {t + 2, 0, target_packet}, {t + 3, 0, target_packet + 16}};
        require(i6b2_fetches(*machine) == fetches, what + "the fetches were not packets 0, 16, then the target's two");
        const auto issues = i6b2_issues(*machine);
        require(issues.size() == 2 && std::get<0>(issues[0]) == t &&
                std::get<0>(issues[1]) == t + nanosm::branch_taken_interval,
            what + "the bra did not issue at t and the halt at t + 4:" + issue_log(*machine));
        const auto& summary = machine->trace().summary();
        require(summary.fetches == 4 && summary.instructions == 2 && summary.fetches_discarded == 6 &&
                2 * summary.fetches == summary.instructions + summary.fetches_discarded,
            what + "fetches " + std::to_string(summary.fetches) + ", issued " +
                std::to_string(summary.instructions) + ", discarded " + std::to_string(summary.fetches_discarded));
        require(machine->trace().summary_text().find("front end: fetches=4 discarded=6 ") != std::string::npos,
            what + "the summary's front-end line");
    }
}

// Two warps, the same code, alternating one issue per cycle from c2 (warp w's
// k-th word at c(2 + w + 2k)). Words: 0..3 movs, 4 'bra target' (the first
// half of packet 32), 5 mov r5, 6 mov r6 (packet 48's first half), 7 the
// target mov r7 (its second half), 8 halt. Fetch (two entries each): packet 0
// at c0 (w0) and c1 (w1), 16 at c2 and c3, 32 at c5 (w0, after its packet 0
// popped at c4) and c6 (w1), 48 for w0 at c9 (its packet 16 popped at c8;
// w1 is full), 48 for w1 at c10 (popped at c9; w0 is full, its packet 48 in
// flight). The branches issue on consecutive cycles: warp 0's at c10, warp
// 1's at c11. So:
//   - warp 0 redirects at c11 while warp 1's packet 48 is at the memory
//     output: warp 1 keeps it (FIFO 2 packets after c11), warp 0's FIFO is
//     cleared, its pc the target 56, its fetch pc packet 48;
//   - warp 1 redirects at c12, the next cycle (consecutive redirects of
//     different warps: each clears only its own front end);
//   - fetch: warp 0's packet 48 again at c12 (warp 1's branch is still in the
//     issue register), warp 1's at c13; the targets issue at c14 and c15,
//     each branch's t + 4 (branch_taken_interval).
void test_redirects_across_warps() {
    const auto& profile = canonical_profile();
    auto machine = i6b2_machine(profile, R"(
            mov r1, 1
            mov r2, 2
            mov r3, 3
            mov r4, 4
            bra target
            mov r5, 5
            mov r6, 6
        target: mov r7, 7
            halt
    )", 2);
    std::map<uint64_t, std::tuple<std::optional<uint32_t>, std::size_t, std::size_t, uint32_t, uint32_t>> after;
    while (!machine->done()) {
        const auto cycle = machine->cycle();
        tick_capped(*machine);
        after[cycle] = {machine->fetch_in_flight(), machine->instruction_fifo_packets(0),
            machine->instruction_fifo_packets(1), machine->warp_pc(0), machine->warp_fetch_pc(0)};
    }
    constexpr uint32_t bra_pc = 4 * 8, target_pc = 7 * 8;
    const uint64_t t0 = nanosm::fetch_to_issue + 2 * 4, t1 = t0 + 1;
    const auto issues = i6b2_issues(*machine);
    const auto issued = [&](uint32_t warp, uint32_t pc) {
        std::vector<uint64_t> cycles;
        for (const auto& [cycle, w, issued_pc] : issues) if (w == warp && issued_pc == pc) cycles.push_back(cycle);
        return cycles;
    };
    require(issued(0, bra_pc) == std::vector<uint64_t>{t0} && issued(1, bra_pc) == std::vector<uint64_t>{t1},
        "the branches did not issue at c10 and c11:" + issue_log(*machine));
    require(std::get<0>(after.at(t0)) == std::optional<uint32_t>(1),
        "warp 1's packet was not at the memory output during warp 0's redirect");
    require(after.at(t1) == std::tuple<std::optional<uint32_t>, std::size_t, std::size_t, uint32_t, uint32_t>{
                std::nullopt, 0, 2, target_pc, target_pc / 16 * 16},
        "warp 0's redirect dropped warp 1's landing, or did not clear its own front end");
    const auto fetches = i6b2_fetches(*machine);
    const auto fetched = [&](uint32_t warp, uint32_t pc) {
        std::vector<uint64_t> cycles;
        for (const auto& [cycle, w, fetched_pc] : fetches) if (w == warp && fetched_pc == pc) cycles.push_back(cycle);
        return cycles;
    };
    require(fetched(0, 48) == std::vector<uint64_t>{t0 - 1, t1 + 1} && fetched(1, 48) == std::vector<uint64_t>{t0, t1 + 2},
        "the target packets were not read again at c12 (warp 0) and c13 (warp 1)");
    require(issued(0, target_pc) == std::vector<uint64_t>{t0 + nanosm::branch_taken_interval} &&
            issued(1, target_pc) == std::vector<uint64_t>{t1 + nanosm::branch_taken_interval},
        "the targets did not issue at each branch's t + 4:" + issue_log(*machine));
    for (uint32_t warp = 0; warp < 2; ++warp) {
        require(machine->read_reg(warp, 0, 4) == 4 && machine->read_reg(warp, 0, 5) == 0 &&
                machine->read_reg(warp, 0, 6) == 0 && machine->read_reg(warp, 0, 7) == 7,
            "warp " + std::to_string(warp) + ": an old-path word had an effect, or the target did not run");
    }
}

// A redirected warp refills behind the warps already asking. Sixteen warps
// start with 'bra target' (packet 0's first half; the target is packet 16's
// first half). Fetch round-robin gives warp w its packet 0 at cw; issue gives
// warp w its branch at c(2 + w) (each warp's head becomes valid just in time
// for its turn), which resolves at c(3 + w). Warp 0's redirect at c3 makes it
// ask from c4, but the cursor is past it: warps 4..15, still empty, get
// their packet 0 at c4..c15, and the cursor comes back to warp 0 at c16 (its
// target packet), so its target issues at c18 -- fifteen cycles after the
// resolve instead of branch_taken_interval's three. No priority: just the
// round-robin.
void test_redirect_behind_many_fetchers() {
    const auto& profile = canonical_profile();
    auto machine = i6b2_machine(profile, R"(
            bra target
            mov r1, 1
        target: mov r2, 2
            halt
    )", 16);
    machine->run();
    const auto fetches = i6b2_fetches(*machine);
    for (uint32_t warp = 0; warp < 16; ++warp) {
        const auto [cycle, w, pc] = fetches.at(warp);
        require(cycle == warp && w == warp && pc == 0, "packet 0 fetches were not c0..c15, warps 0..15");
    }
    const auto [cycle16, warp16, pc16] = fetches.at(16);
    require(cycle16 == 16 && warp16 == 0 && pc16 == 16,
        "warp 0's target packet was not read when the round-robin came back to it at c16");
    std::vector<uint64_t> target_issue;
    for (const auto& [cycle, warp, pc] : i6b2_issues(*machine)) if (warp == 0 && pc == 16) target_issue.push_back(cycle);
    require(target_issue == std::vector<uint64_t>{16 + nanosm::fetch_to_issue},
        "warp 0's target did not issue at c18:" + issue_log(*machine));
    for (uint32_t warp = 0; warp < 16; ++warp) {
        require(machine->read_reg(warp, 0, 1) == 0 && machine->read_reg(warp, 0, 2) == 2,
            "warp " + std::to_string(warp) + ": data");
    }
}

// The front end's consistency check: while a warp runs, its fetch pc is pc
// rounded down to 16 plus 16 per packet held or in flight. Breaking it
// between ticks is caught at the next tick, in both builds.
void test_front_end_consistency_check() {
    const auto& profile = canonical_profile();
    auto machine = i6b2_machine(profile, i6b2_independent_movs(8) + "halt\n");
    for (int i = 0; i < 3; ++i) tick_capped(*machine);
    machine->test_set_fetch_pc(0, machine->warp_fetch_pc(0) + 16);
    try {
        tick_capped(*machine);
        throw std::runtime_error("a broken fetch pc went unnoticed");
    } catch (const nanosm::SimError& error) {
        require(std::string(error.what()).find("warp 0's front end is inconsistent") != std::string::npos,
            std::string("wrong report: ") + error.what());
    }
}

} // namespace

// Lone warp issue rate at 1-4 packets
TEST(FrontEnd, lone_warp_issue_rate) {
    test_lone_warp_issue_rate();
}

// Warps interleave (2, 16)
TEST(FrontEnd, warps_interleave) {
    test_warps_interleave();
}

// Fetch round-robin, hot and waiting warps
TEST(FrontEnd, fetch_round_robin_hot_and_waiting_warps) {
    test_fetch_round_robin_hot_and_waiting_warps();
}

// Conditional branch taken / not taken
TEST(FrontEnd, conditional_branch) {
    test_conditional_branch();
}

// Unconditional branches
TEST(FrontEnd, unconditional_branches) {
    test_unconditional_branches();
}

// Words past the end and on a cleared path
TEST(FrontEnd, words_past_the_end) {
    test_words_past_the_end();
}

// Prefetch through waits
TEST(FrontEnd, prefetch_through_waits) {
    test_prefetch_through_waits();
}

// Setp -> bra.uni distance
TEST(FrontEnd, setp_to_branch_distance) {
    test_setp_to_branch_distance();
}

// Warp slots and profile fields
TEST(FrontEnd, loop_head_pays_the_entry_once) {
    test_loop_head_pays_the_entry_once();
}

TEST(FrontEnd, warp_slots_and_profile_fields) {
    test_warp_slots_and_profile_fields();
}

// Trace supply state
TEST(FrontEnd, trace_supply_state) {
    test_trace_supply_state();
}

// Trace counts every warp
TEST(FrontEnd, trace_counts_every_warp) {
    test_trace_counts_every_warp();
}

// A second-half target's skipped first half is discarded
TEST(FrontEnd, skipped_first_half_is_discarded) {
    test_skipped_first_half_is_discarded();
}

// Redirects across warps
TEST(FrontEnd, redirects_across_warps) {
    test_redirects_across_warps();
}

// Redirect behind many fetchers
TEST(FrontEnd, redirect_behind_many_fetchers) {
    test_redirect_behind_many_fetchers();
}

// Front-end consistency check
TEST(FrontEnd, front_end_consistency_check) {
    test_front_end_consistency_check();
}

// The register file: one-round reads, the RF write decision (I1), read / write depth (I6a), the read side and 16 warps (I6b-1).
// Moved from the original tests/tests.cpp (trash/test-setup-2026-10-22.md);
// each test's body is unchanged, its TEST wrapper is at the end.
#include "test_support.hpp"

namespace {

using namespace nanosm_test;

void check_read_capacity(const nanosm::Machine& machine, const nanosm::DeviceProfile& profile) {
    std::map<std::pair<uint64_t, uint32_t>, uint32_t> counts;
    require(!machine.trace().truncated(), "RF capacity check needs an untruncated trace");
    for (const auto& event : machine.trace().events()) {
        if (event.name != "RF read") continue;
        const auto bank = static_cast<uint32_t>(event.value);
        require(bank == event.id % profile.rf_banks, "read trace has the wrong bank");
        require(++counts[{event.start_cycle, bank}] <= profile.rf_read_ports_per_bank,
            "actual RF captures exceeded per-bank read capacity");
    }
}

// One round (src/rf.hpp): the fadd's two sources (r0 in bank 0, r1 in bank
// 1) are both sampled at issue + issue_to_sample; its operands are at the RF
// output Lr later, its result at the write port fadd later, landing at the
// edge Wd after that -- read_reg sees it after that cycle's tick. The
// source-free mov issues right behind it. setp p0, r0, r0 issues at c2 (its
// word fetched at c0: fetch_to_issue, src/rf.hpp) and reads r0 once, at c3;
// its predicate reaches the file at its port cycle c2 + 1 + Lr + alu (c6), at
// that edge, and done() waits for it: the run ends at c7.
void test_rf_one_round_fadd_and_halt() {
    const auto& profile = canonical_profile();
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& image = image_bytes(launch.shmem_image, 8);
    write_u32(image, 0, nanosm::float_bits(1.25f));
    write_u32(image, 4, nanosm::float_bits(2.5f));
    nanosm::Machine machine(profile, compile(R"(
        lds r0, [r2]
        lds r1, [r2+4]
        fadd r4, r0, r1
        mov r8, 9
        halt
    )", profile), std::move(launch), {.timeline = true, .debug = true});
    uint64_t changed = 0;
    while (!machine.done() && machine.cycle() < 1000) {
        const auto tick = machine.cycle();
        machine.tick();
        if (machine.read_reg(0, 0, 4) != 0 && changed == 0) changed = tick;
    }
    require(machine.done(), "the FADD pipeline did not drain");
    const auto issue = issued_at(machine, 16).at(0);
    require(changed == issue + nanosm::issue_to_sample + profile.rf_read_latency_cycles + profile.timing.fadd +
                profile.rf_write_depth_cycles,
        "accepted FADD latency varied");
    require(machine.read_reg(0, 0, 4) == nanosm::float_bits(3.75f), "the FADD lost its operands");
    std::vector<std::pair<uint64_t, uint64_t>> captures;
    for (const auto& event : machine.trace().events()) {
        if (event.name == "RF read" && event.start_cycle > issue && (event.id == 0 || event.id == 1))
            captures.emplace_back(event.start_cycle, event.id);
    }
    require(captures == std::vector<std::pair<uint64_t, uint64_t>>{{issue + 1, 0}, {issue + 1, 1}},
        "the FADD's sources were not both sampled at issue + 1");
    require(issued_at(machine, 24).at(0) == issue + 1, "the source-free MOV did not follow the FADD");
    check_read_capacity(machine, profile);

    nanosm::Launch predicate_launch;
    predicate_launch.active_warps = 1;
    nanosm::Machine predicate(profile, compile("setp.eq.u32 p0, r0, r0\nhalt", profile),
        std::move(predicate_launch), {.timeline = true, .debug = true});
    predicate.run();
    require(predicate.cycle() == nanosm::fetch_to_issue + nanosm::issue_to_sample + profile.rf_read_latency_cycles +
                profile.timing.alu + 1,
        "HALT dropped a pending predicate-only read pipeline");
    uint32_t duplicate_reads = 0;
    for (const auto& event : predicate.trace().events()) {
        if (event.name == "RF read") {
            require(event.start_cycle == nanosm::fetch_to_issue + nanosm::issue_to_sample,
                "the repeated operand was not sampled at c3");
            ++duplicate_reads;
        }
    }
    require(duplicate_reads == 1, "a repeated operand consumed several RF reads");
}

// One round ends a reader's samples at issue + issue_to_sample, before the
// earliest any later write of the warp can land (earliest_rf_visible: 7 at
// canonical). So qtxc buys no WAR stall, neither for a fixed writer nor for
// an async one, and the reader captures the old value. (With a read window
// of 8, before RFC14, the async writer needed one.) A counted loop in four
// warps still drains its fixed reads and predicates at every branch.
void test_rf_one_round_war_and_branch() {
    const auto& profile = canonical_profile();
    require(nanosm::issue_to_sample + 1 < nanosm::earliest_rf_visible(profile), "a younger write could land first");
    const auto fixed_writer = nanosm::compile_qtx("add r2, r0, r1\nmov r1, 7\nhalt", profile);
    require(fixed_writer.instructions[0].control.stall == 0, "qtxc bought a WAR stall a fixed writer cannot need");
    const auto source = "add r2, r0, r1\nlds r1, [r5]\nhalt";
    const auto async_writer = nanosm::compile_qtx(source, profile);
    require(async_writer.instructions[0].control.stall == 0, "qtxc bought a WAR stall an async writer cannot need");
    nanosm::Launch launch;
    launch.active_warps = 1;
    write_u32(image_bytes(launch.shmem_image, 64), 0, 7);
    nanosm::Machine war(profile, compile(source, profile), std::move(launch));
    war.run();
    require(war.read_reg(0, 0, 2) == 0 && war.read_reg(0, 0, 1) == 7,
        "the reader did not capture the old r1, or the lds did not land");

    nanosm::Launch loop_launch;
    loop_launch.active_warps = 4;
    nanosm::Machine loop(profile, compile(R"(
        mov r0, 0
        mov r4, 9
    loop:
        addi r0, r0, 1
        setp.lt.u32 p0, r0, r4
        @p0 bra.uni loop
        halt
    )", profile), std::move(loop_launch), {.timeline = true, .debug = true});
    loop.run();
    for (uint32_t warp = 0; warp < 4; ++warp)
        require(loop.read_reg(warp, 0, 0) == 9, "branch did not drain fixed read/predicate state");
    check_read_capacity(loop, profile);
}

// Seeded random adds and xoris over twelve registers, in one, four and
// sixteen warps and two or three banks: every add's sources are one
// register twice or two registers in different banks (one round, one read
// port per bank). Every warp ends with the sequential result, and no cycle
// samples a bank more than its ports allow.
void test_rf_seeded_competing_warps() {
    for (const uint32_t banks : {2u, 3u}) {
        for (const uint32_t warps : {1u, 4u, 16u}) {
            auto profile = canonical_profile();
            profile.rf_banks = banks;
            std::string source;
            std::array<uint32_t, 12> expected{};
            for (uint32_t r = 0; r < expected.size(); ++r) {
                expected[r] = r + 1;
                source += "mov r" + std::to_string(r) + ", " + std::to_string(r + 1) + "\n";
            }
            uint32_t seed = banks * 100 + warps;
            const auto random = [&]() { seed = seed * 1664525u + 1013904223u; return seed; };
            for (uint32_t i = 0; i < 400; ++i) {
                const auto dst = random() % expected.size();
                const auto a = random() % expected.size();
                std::vector<uint32_t> other_banks;
                for (uint32_t r = 0; r < expected.size(); ++r) {
                    if (r % banks != a % banks) other_banks.push_back(r);
                }
                const auto b = i % 3 == 0 ? a : other_banks[random() % other_banks.size()];
                if (i % 5 == 0) {
                    const auto immediate = random() % 1024;
                    source += "xori r" + std::to_string(dst) + ", r" + std::to_string(a) +
                        ", " + std::to_string(immediate) + "\n";
                    expected[dst] = expected[a] ^ immediate;
                } else {
                    source += "add r" + std::to_string(dst) + ", r" + std::to_string(a) +
                        ", r" + std::to_string(b) + "\n";
                    expected[dst] = expected[a] + expected[b];
                }
            }
            source += "halt\n";
            nanosm::Launch launch;
            launch.active_warps = warps;
            nanosm::Machine machine(profile, compile(source, profile), std::move(launch),
                {.timeline = true, .debug = true});
            machine.run(50000);
            for (uint32_t warp = 0; warp < warps; ++warp)
                for (uint32_t lane = 0; lane < profile.lanes; ++lane)
                    for (uint32_t r = 0; r < expected.size(); ++r)
                        require(machine.read_reg(warp, lane, r) == expected[r],
                            "seeded RF contention changed the sequential result");
            check_read_capacity(machine, profile);
        }
    }
}

// ---------------------------------------------------------------------------
// I1: the admission/execution split (trash/rfc10.md section 1), load slots
// (rfc12 section 3), the RF write decision (rfc10 section 3: async_owed plus
// round-robin among ready async holders) and the freezeable acc.ld pipeline.
// trash/i1-admission-c8-rf-2026-10-10.md. The timelines below are hand
// assembly with explicit controls, so every cycle is the program's own.
// ---------------------------------------------------------------------------

// I2 (rfc12 section 2): an lds's words come back D + Lb cycles after its
// grant -- the I1 timelines below were written for D = Lb = 0. They keep
// their RF scenarios by moving the fadd bookings and the instructions after
// each lds.uni by the same amount (fadd latency + X, the lds.uni's stall
// 1 + X), so every cycle after the lds.uni is the old one + X.
// I6a: the lds.uni also reaches its slot at dispatch, issue_to_dispatch after
// its issue, so X = D + Lb + issue_to_dispatch. A fadd's booking moves
// issue_to_dispatch (the issue register and Lr ahead of its execution) by
// itself, so its latency only adds D + Lb (i1_fadd_latency). And a result
// written at the RF port in cycle t lands at t + Wd: a test that watches the
// register sees it then. The scenarios' address register ur1 (a uniform
// register: lds.uni's base) is zero from launch: a `umov ur1, 0` feeding the
// lds.uni would need four cycles before it (six when this was a vector
// `mov r1, 0`), which would move the scenario, so the first instruction
// writes r61.
// I6b-2 (the front end): the first instruction issues at fetch_to_issue
// (c2: its word fetched at c0, src/rf.hpp), and the three-word FIFO then
// supplies one instruction per cycle -- through every stall and refusal, the
// FIFO fills behind them. So a scenario's cycles (the comments' c0, c1, ...)
// all move by i1_launch_shift; the tests add it to every absolute cycle.
constexpr uint64_t i1_launch_shift = nanosm::fetch_to_issue;

uint8_t i1_fadd_latency(const nanosm::DeviceProfile& profile, uint32_t base) {
    return static_cast<uint8_t>(base + profile.shmem_port_a_depth + profile.shmem_read_latency_cycles);
}

std::unique_ptr<nanosm::Machine> i1_machine(
    const nanosm::DeviceProfile& profile, const std::string& body, uint32_t warps = 1) {
    nanosm::Launch launch;
    launch.active_warps = warps;
    auto& shmem = image_bytes(launch.shmem_image, 4096);
    for (uint32_t word = 0; word < 1024; ++word) write_u32(shmem, 4 * word, 0x100 + word);
    return std::make_unique<nanosm::Machine>(profile, hand_assemble(i1_asm(profile, body), profile),
        std::move(launch), nanosm::TraceOptions{.timeline = true});
}

// C8, Fable's probe A (trash/causality-audit-fable-2026-10-08.md section 7):
// one instruction stream; shmem word 0 is the lane mask the plain lds's
// addresses are built from. Mask 15: sixteen lanes on sixteen banks, one
// round. Mask 7: lanes l and l + 8 share a bank, two rounds, and the result
// would meet the fadd's booked write of bank 0. Before I1 the lds issued at c7
// with mask 15 and was refused at c7 with mask 7 (its own addresses decided
// its issue). Now both issue in the same cycle; the rounds move only the
// completion.
// I6a: the lds at L reaches its slot at dispatch, its first round is granted
// one cycle later, its words come back D + Lb after its last grant and its
// gather register is complete one cycle after that: one round -> its RF write
// port cycle is L + one_round_port, two rounds -> one more. The fadd, issued
// shli_distance - 1 before the lds (the stall on it waits out shli -> lds),
// books bank 0's port at its issue + issue register + Lr + latency: the
// latency below puts it on the two-round result's port cycle, which is denied
// and written one cycle later. Every result lands Wd after its port.
void test_i1_lds_issue_ignores_its_rounds() {
    auto profile = canonical_profile();
    const auto dispatch = nanosm::issue_to_dispatch(profile);
    const auto one_round_port = dispatch + 1 + profile.shmem_port_a_depth + profile.shmem_read_latency_cycles + 1;
    const auto alu_distance = nanosm::issue_to_visible(nanosm::decode(compile("mov r0, 0\nhalt").words[0]), profile) -
        nanosm::issue_register_stages;
    const auto fadd_before_lds = alu_distance - 1; // shli, fadd, (stall), lds
    profile.timing.fadd = static_cast<uint8_t>(fadd_before_lds + one_round_port + 1 -
        nanosm::issue_register_stages - profile.rf_read_latency_cycles);
    const std::string body = R"(
        sread r0, LANE_ID
        umov ur4, 0
        lds.uni r2, [ur4] {pred=PT, stall=0, set=sb1, wait=-}
        and r0, r0, r2 {pred=PT, stall=0, set=-, wait=sb1}
        shli r0, r0, 2
        fadd r3, r1, r1
        lds r6, [r0] {pred=PT, stall=0, set=sb0, wait=-}
        halt {pred=PT, stall=0, set=-, wait=sb0}
    )";
    struct Outcome {
        uint64_t issue = 0;
        uint64_t written = 0;
        std::vector<uint32_t> rounds;
    };
    const auto run = [&](uint32_t mask) {
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.shmem_image, 256);
        write_u32(shmem, 0, mask);
        for (uint32_t word = 1; word < 64; ++word) write_u32(shmem, 4 * word, 0x100 + word);
        nanosm::Machine machine(profile, hand_assemble(i1_asm(profile, body), profile), std::move(launch),
            nanosm::TraceOptions{.timeline = true});
        const auto written = i1_run(machine, {6});
        for (uint32_t lane = 0; lane < 16; ++lane) {
            const auto word = lane & mask;
            require(machine.read_reg(0, lane, 6) == (word == 0 ? mask : 0x100 + word), "lds lane value");
        }
        const auto issue = issued_at(machine, 6 * 8);
        require(issue.size() == 1 && written.at(6).size() == 1, "the lds issued or wrote more than once");
        return Outcome{issue[0], written.at(6)[0], shmem_request_rounds(machine)};
    };
    const auto one_round = run(15);
    const auto two_rounds = run(7);
    require(one_round.rounds == std::vector<uint32_t>{1, 1} && two_rounds.rounds == std::vector<uint32_t>{1, 2},
        "the masks did not give one and two bank rounds");
    require(one_round.issue == two_rounds.issue,
        "the lds's issue cycle depends on its own addresses (C8): " + std::to_string(one_round.issue) + " vs " +
            std::to_string(two_rounds.issue));
    const auto wd = profile.rf_write_depth_cycles;
    require(one_round.written == one_round.issue + one_round_port + wd,
        "one round: the result did not land at issue + one_round_port + Wd");
    require(two_rounds.written == two_rounds.issue + one_round_port + 2 + wd,
        "two rounds: the result (ready when the fadd's booking holds bank 0) was not written one cycle later");
}

// The tripwire behind "a refused attempt reads nothing": a capture inside an
// attempt the admission has not accepted is an internal sanitizer PANIC.
void test_sanitizer_issue_capture_before_acceptance() {
    nanosm::Sanitizer sanitizer;
    sanitizer.record_cycle_start();
    sanitizer.record_issue_attempt(3);
    try {
        sanitizer.check_issue_capture(12, 3, 7, false);
        throw std::runtime_error("a capture before acceptance did not panic");
    } catch (const nanosm::SimError& error) {
        const std::string message = error.what();
        require(message.find("warp 3 read r7") != std::string::npos &&
                message.find("not accepted") != std::string::npos,
            "wrong tripwire message: " + message);
    }
    sanitizer.record_issue_accepted();
    sanitizer.check_issue_capture(12, 3, 7, false); // accepted: fine
    sanitizer.record_issue_attempt(4);              // the next attempt starts unaccepted again
    try {
        sanitizer.check_issue_capture(12, 4, 1, true);
        throw std::runtime_error("a predicate capture before acceptance did not panic");
    } catch (const nanosm::SimError& error) {
        require(std::string(error.what()).find("read p1") != std::string::npos, "wrong predicate tripwire message");
    }
}

// Refused attempts of every kind that carries operands complete without the
// tripwire firing (sanitizer ON), and the refusals really happened. Admission
// itself (depbar_wait, structural_refusal) takes no register at all.
void test_i1_refused_attempts_read_nothing() {
    std::set<std::string> reasons;
    const auto collect = [&](const nanosm::Machine& machine) {
        for (const auto& event : machine.trace().events()) {
            // I2: shmem refusals have their own warp states.
            const bool refused = event.name == "structural" || event.name == "no free slot" ||
                event.name == "ordered behind own access";
            if (event.category == nanosm::TraceCategory::warp && refused) reasons.insert(event.detail);
        }
    };
    {   // load slots full: both slots hold results the booked RF bank 0 denies
        auto profile = canonical_profile();
        profile.timing.fadd = 7;
        auto machine = i1_machine(profile, R"(
            umov ur1, 0
            fadd r30, r4, r4
            fadd r33, r4, r4
            fadd r36, r4, r4
            fadd r39, r4, r4
            fadd r42, r4, r4
            fadd r45, r4, r4
            fadd r48, r4, r4
            lds.uni r3, [ur1] {pred=PT, stall=0, set=sb0, wait=-}
            lds.uni r6, [ur1 + 4] {pred=PT, stall=0, set=sb1, wait=-}
            lds.uni r9, [ur1 + 8] {pred=PT, stall=0, set=sb2, wait=-}
            halt {pred=PT, stall=0, set=-, wait=sb0|sb1|sb2}
        )");
        machine->run();
        collect(*machine);
        require(machine->read_reg(0, 5, 9) == 0x102, "the refused lds.uni loaded the wrong word");
    }
    {   // RF write inhibited: a fadd while a load result is owed service
        auto profile = canonical_profile();
        const auto x = i1_lds_shift(profile);
        profile.timing.fadd = i1_fadd_latency(profile, 4);
        auto machine = i1_machine(profile, R"(
            mov r61, 0
            fadd r6, r4, r4
            fadd r9, r4, r4
            lds.uni r3, [ur1] {pred=PT, stall=)" + std::to_string(1 + x) + R"(, set=sb0, wait=-}
            fadd r12, r4, r4
            fadd r15, r4, r4
            halt {pred=PT, stall=0, set=-, wait=sb0}
        )");
        machine->run();
        collect(*machine);
    }
    {   // ordered behind own access (I2; was "port A busy"): an lds behind a two-round sts
        auto machine = i1_machine(canonical_profile(), R"(
            sread r0, LANE_ID
            shli r0, r0, 3
            mov r2, 9
            sts [r0], r2 {pred=PT, stall=0, set=sb0, wait=-}
            lds r5, [r0 + 4] {pred=PT, stall=0, set=sb1, wait=-}
            halt {pred=PT, stall=0, set=-, wait=sb0|sb1}
        )");
        machine->run();
        collect(*machine);
        require(machine->read_reg(0, 3, 5) == 0x100 + 7, "the refused lds loaded the wrong word");
    }
    {   // LSU command slot; acc transfer engine busy (two acc.st: one acc.st engine)
        auto profile = canonical_profile();
        profile.lsu_active_commands = 1;
        nanosm::Launch launch;
        launch.active_warps = 1;
        fill_pattern(image_bytes(launch.gmem_image, 4096), 0, 1024, 0);
        nanosm::Machine machine(profile, hand_assemble(i1_asm(profile, R"(
            sread r0, LANE_ID
            shli r0, r0, 2
            umov ur8, 0
            umov ur9, 256
            ldg r11, [r0] {pred=PT, stall=0, set=sb0, wait=-}
            ldg r12, [r0 + 64] {pred=PT, stall=0, set=sb1, wait=-}
            acc.st.v8 [ur8, 16], r0 {pred=PT, stall=0, set=sb2, wait=-}
            acc.st.v8 [ur9, 16], r0 {pred=PT, stall=0, set=sb3, wait=-}
            halt {pred=PT, stall=0, set=-, wait=sb0|sb1|sb2|sb3}
        )"), profile), std::move(launch), nanosm::TraceOptions{.timeline = true});
        machine.run();
        collect(machine);
        require(machine.read_reg(0, 4, 12) == pattern_word(20), "the refused ldg loaded the wrong word");
    }
    for (const std::string reason : {"load slots full", "RF write inhibited", "ordered behind own access",
             "LSU command slot", "acc transfer engine busy"}) {
        require(reasons.contains(reason), "no attempt was refused for '" + reason + "'; the test proves little");
    }
}

// (a) A ready lds result against dense arithmetic writing its bank (fadd
// latency 4). The fadds at c1, c2 book bank 0 for c5, c6 and set async_owed;
// the lds.uni (c3) has its word at c5: denied with service owed, so bank 0
// is inhibited and the fadd at c5 is refused (c5, c6); at c7 the old
// bookings have drained and the result is written. In that cycle the fadd
// books c11: owed is set again (a booking and a grant in one cycle: set wins).
void test_i1_rf_owed_inhibits_after_denial() {
    auto profile = canonical_profile();
    const auto x = i1_lds_shift(profile);
    const auto wd = profile.rf_write_depth_cycles;
    profile.timing.fadd = i1_fadd_latency(profile, 4);
    auto machine = i1_machine(profile, R"(
        mov r61, 0
        fadd r6, r4, r4
        fadd r9, r4, r4
        lds.uni r3, [ur1] {pred=PT, stall=)" + std::to_string(1 + x) + R"(, set=sb0, wait=-}
        fadd r12, r4, r4
        fadd r15, r4, r4
        halt {pred=PT, stall=0, set=-, wait=sb0}
    )");
    require(!machine->rf_write_owed(0), "async_owed did not start clear");
    const auto s = i1_launch_shift;
    std::map<uint64_t, std::pair<bool, uint32_t>> bank0; // cycle -> owed, cursor after it
    const auto written = i1_run(*machine, {3}, [&](uint64_t cycle) {
        bank0[cycle] = {machine->rf_write_owed(0), machine->rf_write_cursor(0)};
    });
    require(!bank0.at(s).first && bank0.at(s + 1).first, "a successful booking of bank 0 did not set async_owed");
    require(issued_at(*machine, 3 * 8) == std::vector<uint64_t>{s + 3}, "lds.uni issue");
    require(written.at(3) == std::vector<uint64_t>{s + 7 + x + wd},
        "the result was not written when the old bookings drained (c7 + x, landing Wd later)");
    require(refusals(*machine, 0, "RF write inhibited") == std::vector<std::pair<uint64_t, uint64_t>>{{s + 5 + x, 2}},
        "bank 0 was not inhibited exactly while the result was denied with service owed (c5 + x, c6 + x)");
    require(issued_at(*machine, 5 * 8) == std::vector<uint64_t>{s + 7 + x},
        "the inhibited fadd did not issue in the grant cycle");
    require(bank0.at(s + 7 + x).first, "a grant and a booking in one cycle cleared async_owed (set must win)");
    require(bank0.at(s + 6 + x).second == 0 && bank0.at(s + 7 + x).second == 1,
        "the cursor did not move past load slot 0 on its grant");
    const auto& summary = machine->trace().summary();
    require(summary.rf_async_waits == 2 && summary.rf_simt_inhibited == 2 && summary.rf_async_writes == 1,
        "rf counters");
}

// (a, cont.) async_owed = 0 permits one more booking; a refused candidate
// leaves the bit alone. fadd latency 5: the first lds.uni's result (c5) is
// granted on a free port and clears owed; the second's (c6) is denied with
// owed = 0, so the fadd at c6 still books (and sets owed); at c7 the result
// is denied again and the next fadd is inhibited; c8 it is written.
void test_i1_rf_owed_one_more_booking() {
    auto profile = canonical_profile();
    const auto x = i1_lds_shift(profile);
    const auto wd = profile.rf_write_depth_cycles;
    profile.timing.fadd = i1_fadd_latency(profile, 5);
    {
        auto machine = i1_machine(profile, R"(
            mov r61, 0
            fadd r6, r4, r4
            fadd r9, r4, r4
            lds.uni r3, [ur1] {pred=PT, stall=0, set=sb0, wait=-}
            lds.uni r12, [ur1] {pred=PT, stall=)" + std::to_string(1 + x) + R"(, set=sb1, wait=-}
            mov r2, 1
            fadd r15, r4, r4
            fadd r18, r4, r4
            halt {pred=PT, stall=0, set=-, wait=sb0|sb1}
        )");
        const auto s = i1_launch_shift;
        std::map<uint64_t, std::pair<bool, uint32_t>> bank0;
        const auto written = i1_run(*machine, {3, 12}, [&](uint64_t cycle) {
            bank0[cycle] = {machine->rf_write_owed(0), machine->rf_write_cursor(0)};
        });
        require(written.at(3) == std::vector<uint64_t>{s + 5 + x + wd} &&
                written.at(12) == std::vector<uint64_t>{s + 8 + x + wd},
            "write cycles");
        require(!bank0.at(s + 5 + x).first, "an async grant with no booking in its cycle did not clear async_owed");
        require(issued_at(*machine, 6 * 8) == std::vector<uint64_t>{s + 6 + x},
            "with async_owed = 0 a denied writer did not let one more booking through");
        require(bank0.at(s + 6 + x).first, "that booking did not set async_owed");
        require(refusals(*machine, 0, "RF write inhibited") ==
                std::vector<std::pair<uint64_t, uint64_t>>{{s + 7 + x, 1}},
            "the next booking was not inhibited (c7 + x)");
        require(bank0.at(s + 5 + x).second == 1 && bank0.at(s + 6 + x).second == 1 &&
                bank0.at(s + 7 + x).second == 1 && bank0.at(s + 8 + x).second == 2,
            "the cursor moved without a grant, or not on one");
    }
    {   // A refused booking (its port taken) does not set the bit. I6a: the add
        // after the lds.uni moved X with it, but its booking is only
        // issue_to_dispatch + alu ahead of its issue: the fadd latency that
        // puts their ports on one cycle is 5 + X here.
        auto refusing = profile;
        refusing.timing.fadd = static_cast<uint8_t>(5 + x);
        auto machine = i1_machine(refusing, R"(
            mov r61, 0
            fadd r6, r4, r4
            lds.uni r3, [ur1] {pred=PT, stall=)" + std::to_string(1 + x) + R"(, set=sb0, wait=-}
            mov r2, 1
            mov r5, 1
            add r9, r4, r4
            halt {pred=PT, stall=0, set=-, wait=sb0}
        )");
        const auto s = i1_launch_shift;
        std::map<uint64_t, bool> owed;
        (void)i1_run(*machine, {}, [&](uint64_t cycle) { owed[cycle] = machine->rf_write_owed(0); });
        require(refusals(*machine, 0, "RF write reservation") ==
                std::vector<std::pair<uint64_t, uint64_t>>{{s + 5 + x, 1}},
            "the add was not refused at c5 + x (its write cycle is the fadd's)");
        require(!owed.at(s + 4 + x) && !owed.at(s + 5 + x) && owed.at(s + 6 + x),
            "a refused booking changed async_owed");
    }
}

// (d) Inhibition refuses only fixed RF-write bookings to the inhibited bank:
// while bank 0 is inhibited (c5, c6) a mov writing bank 1 issues, and so does
// an lds.uni whose destination is in bank 0 (a destination is not a booking).
// (e) With no async writer ready, a set async_owed inhibits nothing.
void test_i1_rf_inhibition_is_bank_and_booking_local() {
    {
        auto profile = canonical_profile();
        const auto x = i1_lds_shift(profile);
        const auto wd = profile.rf_write_depth_cycles;
        profile.timing.fadd = i1_fadd_latency(profile, 4);
        auto machine = i1_machine(profile, R"(
            mov r61, 0
            fadd r6, r4, r4
            fadd r9, r4, r4
            lds.uni r3, [ur1] {pred=PT, stall=)" + std::to_string(1 + x) + R"(, set=sb0, wait=-}
            fadd r12, r4, r4
            mov r13, 1
            lds.uni r21, [ur1] {pred=PT, stall=0, set=sb1, wait=-}
            fadd r15, r4, r4
            halt {pred=PT, stall=0, set=-, wait=sb0|sb1}
        )");
        const auto s = i1_launch_shift;
        const auto written = i1_run(*machine, {3, 13, 21});
        require(written.at(3) == std::vector<uint64_t>{s + 7 + x + wd},
            "the first result was not written at c7 + x (+ Wd)");
        require(issued_at(*machine, 5 * 8) == std::vector<uint64_t>{s + 5 + x},
            "a bank-1 write was held by bank 0's inhibition");
        require(issued_at(*machine, 6 * 8) == std::vector<uint64_t>{s + 6 + x},
            "an lds.uni into bank 0 was held by bank 0's inhibition");
        // The second lds.uni's own result is x later too (it meets fadd r12's booking).
        require(written.at(21) == std::vector<uint64_t>{s + 9 + 2 * x + wd}, "the second result");
        require(machine->trace().summary().rf_simt_inhibited == 0, "nothing should have been inhibited");
    }
    {
        auto machine = i1_machine(canonical_profile(), R"(
            mov r1, 0
            fadd r6, r4, r4
            fadd r9, r4, r4
            fadd r12, r4, r4
            fadd r15, r4, r4
            fadd r18, r4, r4
            halt
        )");
        const auto s = i1_launch_shift;
        std::map<uint64_t, bool> owed;
        (void)i1_run(*machine, {}, [&](uint64_t cycle) { owed[cycle] = machine->rf_write_owed(0); });
        for (uint32_t pc = 1; pc <= 5; ++pc) {
            require(issued_at(*machine, pc * 8) == std::vector<uint64_t>{s + pc},
                "a booking was refused with no writer ready");
        }
        require(owed.at(s + 1) && owed.at(s + 5), "async_owed was not set by the bookings");
        require(machine->trace().summary().rf_simt_inhibited == 0, "inhibited with no async writer ready");
    }
}

// (c) Round-robin among ready holders. One RF bank (two read ports), so every
// holder targets it. I8: one acc.ld engine for the machine, so the holders
// are four load slots (four lds.uni results, shmem.load_slots = 4: holders
// 0-3) and the acc.ld pipeline (holder 6, after the two assembly registers
// 4-5). Twelve fadds book the bank's write
// port for twelve consecutive cycles from R, the cycle the first holder is
// ready (a calibration run with fadd latency 1 measures R and the first
// fadd's issue F0; the fadd latency that puts F0's booking on R is used), so
// every holder becomes ready inside the window and is denied with service
// owed: the mov r50 (placed by the acc.ld's stall so that its first attempt
// already wants a port inside the window: issue at R - 4) is held back (its
// own booked port, then the inhibition) and every holder waits. From W + 1 (W = R + 11, the window's last cycle)
// the holders drain in round-robin order from cursor 0: slots 0..3 (r40 ..
// r43), each grant moving the cursor past it; the first grant lifts the
// inhibition, so the mov is accepted at W + 1 and its write takes the port at
// W + 5 (issue register, Lr, alu): no async grant then, and the cursor stays
// at 4; then the eight beats r16 .. r23, each grant wrapping the cursor to 0
// and the scan skipping the empty slots and assemblies back to the
// pipeline. Every cycle from W + 1 writes once: no refill bubble (the
// pipeline was full while it held). Alternation between two persistently
// ready holders is the unit test below (the machine has one acc.ld engine).
void test_i1_rf_round_robin_among_holders() {
    auto profile = canonical_profile();
    profile.rf_banks = 1;
    profile.rf_read_ports_per_bank = 2;
    profile.shmem_load_slots = 4;
    const auto body_with = [](uint32_t acc_ld_stall) { return R"(
        sread r0, LANE_ID
        addi r0, r0, 1
        addi r1, r0, 100
        addi r2, r0, 200
        addi r3, r0, 300
        addi r4, r0, 400
        addi r5, r0, 500
        addi r6, r0, 600
        addi r7, r0, 700
        umov ur8, 0
        umov ur9, 0
        umov ur11, 64
        acc.st.v8 [ur8, 64], r0 {pred=PT, stall=0, set=sb0, wait=-}
        fadd r25, r10, r10
        fadd r26, r10, r10
        fadd r27, r10, r10
        fadd r28, r10, r10
        fadd r29, r10, r10
        fadd r30, r10, r10
        fadd r31, r10, r10
        fadd r32, r10, r10
        fadd r33, r10, r10
        fadd r34, r10, r10
        fadd r35, r10, r10
        fadd r36, r10, r10
        lds.uni r40, [ur11] {pred=PT, stall=0, set=sb1, wait=-}
        lds.uni r41, [ur11 + 4] {pred=PT, stall=0, set=sb2, wait=-}
        lds.uni r42, [ur11 + 8] {pred=PT, stall=0, set=sb3, wait=-}
        lds.uni r43, [ur11 + 12] {pred=PT, stall=0, set=sb4, wait=-}
        acc.ld.v8 r16, [ur8, 64] {pred=PT, stall=)" + std::to_string(acc_ld_stall) + R"(, set=sb5, wait=sb0}
        mov r50, 1
        halt {pred=PT, stall=0, set=-, wait=sb1|sb2|sb3|sb4|sb5}
    )"; };
    std::vector<uint32_t> regs{40, 41, 42, 43, 50};
    for (uint32_t reg = 16; reg < 24; ++reg) regs.push_back(reg);
    const auto wd = profile.rf_write_depth_cycles;
    const uint32_t first_fadd_pc = 13 * 8;
    const uint32_t acc_ld_pc = 29 * 8;
    uint64_t r = 0;
    uint64_t f0 = 0;
    uint64_t a = 0;
    {
        auto calibration_profile = profile;
        calibration_profile.timing.fadd = 1;
        // The mov far behind the holders, so its write cannot hide when the
        // first holder is ready.
        auto calibration = i1_machine(calibration_profile, body_with(30));
        const auto written = i1_run(*calibration, regs);
        r = UINT64_MAX;
        for (const auto reg : {40u, 41u, 42u, 43u, 16u}) r = std::min(r, written.at(reg).at(0) - wd);
        f0 = issued_at(*calibration, first_fadd_pc).at(0);
        a = issued_at(*calibration, acc_ld_pc).at(0);
    }
    // The mov first tries at R - 4 (its port R, the window's first cycle).
    const auto mov_stall = static_cast<uint32_t>(r - 4 - a);
    require(r >= a + 5, "the first holder is ready too soon after the acc.ld for this test's layout");
    profile.timing.fadd = static_cast<uint8_t>(r - f0 - nanosm::issue_register_stages - profile.rf_read_latency_cycles);
    nanosm::validate_device_profile(profile);
    auto machine = i1_machine(profile, body_with(mov_stall));
    std::map<uint64_t, uint32_t> cursor;
    const auto written = i1_run(*machine, regs, [&](uint64_t cycle) { cursor[cycle] = machine->rf_write_cursor(0); });
    require(issued_at(*machine, first_fadd_pc) == std::vector<uint64_t>{f0} &&
            issued_at(*machine, acc_ld_pc) == std::vector<uint64_t>{a},
        "the fadds or the acc.ld moved with the fadd latency; the calibration does not hold");
    std::vector<std::pair<uint64_t, uint32_t>> order; // (port cycle, register)
    for (const auto& [reg, cycles] : written) {
        require(cycles.size() == 1, "r" + std::to_string(reg) + " was not written exactly once");
        order.emplace_back(cycles[0] - wd, reg);
    }
    std::sort(order.begin(), order.end());
    const auto w = r + 11;
    const std::vector<uint32_t> expected{40, 41, 42, 43, 50, 16, 17, 18, 19, 20, 21, 22, 23};
    for (std::size_t i = 0; i < order.size(); ++i) {
        require(order[i].second == expected[i], "write " + std::to_string(i) + " was r" +
            std::to_string(order[i].second) + ", not r" + std::to_string(expected[i]) + " | issues" +
            issue_log(*machine));
        require(order[i].first == w + 1 + i, "write " + std::to_string(i) + " was not at W + 1 + " +
            std::to_string(i) + " (a refill bubble, or the window moved): port c" + std::to_string(order[i].first) +
            ", R = " + std::to_string(r) + ", W = " + std::to_string(w) + " | issues" + issue_log(*machine));
    }
    // The cursor moves past each holder at its grant: slot k -> k + 1, the
    // acc.ld pipeline (holder 6 of 7) -> 0; the mov's cycle grants nobody.
    for (uint32_t k = 0; k < 4; ++k) {
        require(cursor.at(w + 1 + k) == k + 1, "the cursor did not move past load slot " + std::to_string(k));
    }
    require(cursor.at(w + 5) == 4, "the cursor moved in the cycle no holder was granted");
    for (uint64_t cycle = w + 6; cycle <= w + 13; ++cycle) {
        require(cursor.at(cycle) == 0, "the cursor did not wrap past the acc.ld pipeline at its grant");
    }
    for (uint32_t lane = 0; lane < 16; ++lane) {
        for (uint32_t k = 0; k < 4; ++k) {
            require(machine->read_reg(0, lane, 40 + k) == 0x100 + 16 + k, "lds.uni data");
        }
        for (uint32_t beat = 0; beat < 8; ++beat) {
            require(machine->read_reg(0, lane, 16 + beat) == lane + 1 + 100 * beat, "acc.ld beat data");
        }
    }
}

// The round-robin choice itself (one bank): scan from the cursor, skip holders
// not ready, grant up to the free ports, move past the last one granted.
void test_i1_round_robin_grant() {
    using Grant = nanosm::RoundRobinGrant;
    const auto grant = [](std::vector<uint8_t> ready, uint32_t cursor, uint32_t ports) {
        return nanosm::round_robin_grant(ready, cursor, ports);
    };
    const auto same = [](const Grant& got, std::vector<uint32_t> granted, uint32_t cursor) {
        return got.granted == granted && got.cursor_after == cursor;
    };
    require(same(grant({1, 0, 1, 0}, 0, 1), {0}, 1), "from cursor 0");
    require(same(grant({1, 0, 1, 0}, 1, 1), {2}, 3), "skips a holder that is not ready");
    require(same(grant({1, 0, 1, 0}, 3, 1), {0}, 1), "wraps around");
    require(same(grant({1, 0, 1, 0}, 2, 0), {}, 2), "no free port: nobody granted, the cursor stays");
    require(same(grant({0, 0, 0, 0}, 2, 1), {}, 2), "nobody ready: the cursor stays");
    require(same(grant({1, 1, 1, 0}, 1, 2), {1, 2}, 3), "two ports: two grants, past the last");
    // Two holders that stay ready alternate.
    uint32_t cursor = 0;
    std::vector<uint32_t> turns;
    for (int i = 0; i < 6; ++i) {
        const auto result = grant({0, 1, 0, 1}, cursor, 1);
        turns.push_back(result.granted.at(0));
        cursor = result.cursor_after;
    }
    require(turns == std::vector<uint32_t>{1, 3, 1, 3, 1, 3}, "two persistently ready holders did not alternate");
}

// The acc.ld transfer pipeline freezes and resumes (rfc10 section 2). Six
// fadds (released when the acc.st's depbar clears, by an add that waits on
// it, so their bookings sit near P) book RF bank 1 for six consecutive cycles from P,
// the cycle beat 0 (r16, bank 1) is first the pipeline's output. Denied, the
// whole pipeline holds -- same destination and data, no new tmem read, depbar
// set, done() false -- while the add that wants bank 1 is held back first by
// its own booked port, then by the inhibition. At P + 6 the beat is written
// and the pipeline refills with no bubble; later beats hold one cycle behind
// each add's booking. The depbar clears as the last beat's write lands.
// I8: the acc.ld engine takes the transfer at its dispatch (A +
// issue_to_dispatch) and launches beat 0 the next cycle, L; beat k launches
// at L + k while the pipeline moves and samples tmem tmem.port_depth (Dt)
// later; beat 0 is the output tmem.read_latency_cycles (Lt) after its
// sample: P = L + Dt + Lt. So when the output holds at P, the beats at
// stages Dt - 1 .. Dt + Lt - 1 have sampled tmem (Lt + 1 of them: beat Lt's
// sample is at P itself) and nothing launches or samples while it holds. A
// calibration run (fadd latency 1, bookings long gone) measures A and the
// first fadd's issue F0; the fadd latency that puts F0's booking (F0 + issue
// register + Lr + latency) on P is then used. The add r46 first tries to
// issue at P - 3 (the acc.ld's stall puts it there: P - 3 - A cycles after
// the acc.ld) and books issue + 4 (issue register, Lr, alu): its ports P + 1 .. P + 5 are
// the fadds', then (P + 2 .. P + 5) bank 1 is inhibited while beat 0 is
// denied with service owed; it issues at P + 6 and books P + 10. The add
// r43 issues at P + 8 and books P + 12. The beats r16..r23 sit in banks 1,
// 2, 0, 1, 2, 0, 1, 2: beats 0-5 are written at P + 6 .. P + 11 (r46's
// bank-1 booking at P + 10 is not beat 4's bank), beat 6 (bank 1) meets
// r43's booking at P + 12 and goes at P + 13, beat 7 at P + 14. Every
// register changes Wd after its write.
void test_i1_acc_ld_pipeline_freeze() {
    auto profile = canonical_profile();
    const auto t = nanosm::issue_to_dispatch(profile);
    const auto dt = profile.tmem_port_depth;
    const auto lt = profile.tmem_read_latency_cycles;
    // P - A = t + 1 + Dt + Lt; the add r46 issues at P - 3.
    const auto r46_after = t + 1 + dt + lt - 3;
    const std::string body = R"(
        sread r0, LANE_ID
        addi r0, r0, 1
        addi r1, r0, 100
        addi r2, r0, 200
        addi r3, r0, 300
        addi r4, r0, 400
        addi r5, r0, 500
        addi r6, r0, 600
        addi r7, r0, 700
        umov ur8, 0
        umov ur9, 0
        acc.st.v8 [ur8, 64], r0 {pred=PT, stall=0, set=sb0, wait=-}
        add r44, r10, r10 {pred=PT, stall=0, set=-, wait=sb0}
        fadd r25, r10, r10
        fadd r28, r10, r10
        fadd r31, r10, r10
        fadd r34, r10, r10
        fadd r37, r10, r10
        fadd r40, r10, r10
        acc.ld.v8 r16, [ur8, 64] {pred=PT, stall=)" + std::to_string(r46_after) + R"(, set=sb1, wait=-}
        add r46, r10, r10 {pred=PT, stall=2, set=-, wait=-}
        add r43, r10, r10
        halt {pred=PT, stall=0, set=-, wait=sb1}
    )";
    profile.timing.fadd = 1;
    uint64_t acc_ld_issue = 0;
    uint64_t first_fadd = 0;
    {
        auto calibration = i1_machine(profile, body);
        calibration->run();
        acc_ld_issue = issued_at(*calibration, 19 * 8).at(0);
        first_fadd = issued_at(*calibration, 13 * 8).at(0);
    }
    const auto launch = acc_ld_issue + t + 1;
    const auto p = launch + dt + lt;
    profile.timing.fadd = static_cast<uint8_t>(p - first_fadd - nanosm::issue_register_stages -
        profile.rf_read_latency_cycles);
    auto machine = i1_machine(profile, body);
    const auto stages = machine->acc_ld_pipeline_stages();
    require(stages == dt + lt, "the acc.ld pipeline is not tmem port depth + read latency stages");
    struct Seen {
        std::optional<nanosm::Machine::AccLdStageView> output;
        uint64_t tmem_reads = 0;
        uint32_t sb1 = 0;
        bool done = false;
    };
    std::map<uint64_t, Seen> seen;
    std::vector<uint32_t> regs;
    for (uint32_t reg = 16; reg < 24; ++reg) regs.push_back(reg);
    const auto written = i1_run(*machine, regs, [&](uint64_t cycle) {
        seen[cycle] = {machine->acc_ld_stage(stages - 1), machine->trace().summary().tmem_load_beats,
            machine->depbar_value(0, 1), machine->done()};
    });
    require(issued_at(*machine, 19 * 8) == std::vector<uint64_t>{acc_ld_issue},
        "acc.ld issue moved with the fadd latency; the calibration does not hold");
    // Beats 0 .. Lt - 1 sampled tmem before P.
    require(seen.at(p - 1).tmem_reads == lt, "tmem samples before the hold");
    // Held: the output after the ticks of P - 1 .. P + 5 (Q of P .. P + 6) is beat 0, unchanged.
    for (uint64_t cycle = p - 1; cycle <= p + 5; ++cycle) {
        const auto& at = seen.at(cycle);
        require(at.output.has_value() && at.output->reg == 16 && !at.output->last,
            "the output stage did not hold beat 0 after c" + std::to_string(cycle));
        require(at.output->values == seen.at(p - 1).output->values, "the held beat's data changed");
        if (cycle >= p) {
            require(at.tmem_reads == lt + 1,
                "a tmem read launched or sampled while the output was held (c" + std::to_string(cycle) + ")");
        }
        require(at.sb1 == 1 && !at.done, "the depbar or done() moved while a beat was held");
    }
    const auto wd = profile.rf_write_depth_cycles;
    const std::vector<uint64_t> port{p + 6, p + 7, p + 8, p + 9, p + 10, p + 11, p + 13, p + 14};
    for (uint32_t beat = 0; beat < 8; ++beat) {
        const auto reg = 16 + beat;
        std::string seen_writes;
        for (const auto cycle : written.at(reg)) seen_writes += " " + std::to_string(cycle);
        require(written.at(reg) == std::vector<uint64_t>{port[beat] + wd},
            "beat " + std::to_string(beat) + " was not written exactly once at c" + std::to_string(port[beat]) +
                " (landing Wd later; P = " + std::to_string(p) + ", landed" + seen_writes + "; inhibited" +
                describe(refusals(*machine, 0, "RF write inhibited")) + "; booked" +
                describe(refusals(*machine, 0, "RF write reservation")) + "; issues" + issue_log(*machine) + ")");
        for (uint32_t lane = 0; lane < 16; ++lane) {
            require(machine->read_reg(0, lane, reg) == lane + 1 + 100 * beat, "beat " + std::to_string(beat) + " data");
        }
    }
    require(seen.at(p + 12).output.has_value() && seen.at(p + 12).output->reg == 22,
        "beat 6 did not hold through P + 12 after beats 0-5 were written");
    require(seen.at(port[7] + wd - 1).sb1 == 1 && seen.at(port[7] + wd).sb1 == 0,
        "the depbar did not clear exactly as the last beat's write landed");
    require(refusals(*machine, 0, "RF write inhibited") == std::vector<std::pair<uint64_t, uint64_t>>{{p + 2, 4}},
        "the add was not inhibited while beat 0 was denied with service owed");
    require(machine->trace().summary().tmem_load_beats == 8 && machine->trace().summary().rf_async_writes == 8,
        "beats launched or written more than once");
}

// Load slots (rfc12 section 3; shmem.load_slots = 2). Two lds.uni hold both
// slots, their results complete and waiting for the RF (eight fadds keep RF
// bank 0 booked over them); a third is refused (structural, counted) through
// the cycle the RF write decision grants slot 0 and issues the cycle after
// (issue counts the slots empty in Q: trash/slot-return-cut-2026-11-20.md).
// Meanwhile port A serves a store while both slots hold results.
// I6a re-derivation (was two warps, qtxc): one warp, hand assembly, so the
// fadd bookings are contiguous (a second warp's issues used to leave holes in
// them, and a hole under a ready result grants it). Relative to the first
// lds.uni at L0: the eight fadds issue at L0 - 8 .. L0 - 1 and book bank 0
// for L0 - 5 + latency .. L0 + 2 + latency (issue register, Lr, latency:
// src/rf.hpp); latency 13 covers L0 + 8 .. L0 + 15. Each lds.uni's result is
// at its RF port x + 2 after its issue (x = i1_lds_shift): L0 + 9 and
// L0 + 10, denied until the bookings end, so slot 0's write is granted at
// G0 = L0 + 16. The sts (a write) is ordered behind the two loads until the
// second one's round is granted (L0 + 5), issues then and is granted at
// L0 + 9 (dispatch, slot register), reaching the RAM at L0 + 11 while both
// results wait. The third lds.uni is ordered behind the sts until that grant
// (L0 + 6 .. L0 + 8; before I6a the sts was granted the cycle after its
// issue, so no order wait showed), then refused for load slots from L0 + 9
// through G0, and issues at G0 + 1. Results land Wd after their grants.
void test_i1_load_slots() {
    auto profile = canonical_profile();
    const auto x = i1_lds_shift(profile);
    const auto wd = profile.rf_write_depth_cycles;
    profile.timing.fadd = i1_fadd_latency(profile, 9);
    auto machine = i1_machine(profile, R"(
        sread r20, LANE_ID
        mov r21, 7
        shli r20, r20, 2
        fadd r30, r4, r4
        fadd r33, r4, r4
        fadd r36, r4, r4
        fadd r39, r4, r4
        fadd r42, r4, r4
        fadd r45, r4, r4
        fadd r48, r4, r4
        fadd r51, r4, r4
        lds.uni r3, [ur1] {pred=PT, stall=0, set=sb0, wait=-}
        lds.uni r6, [ur1 + 4] {pred=PT, stall=0, set=sb1, wait=-}
        sts [r20 + 1024], r21 {pred=PT, stall=0, set=sb2, wait=-}
        lds.uni r9, [ur1 + 8] {pred=PT, stall=0, set=sb3, wait=-}
        halt {pred=PT, stall=0, set=-, wait=sb0|sb1|sb2|sb3}
    )");
    // The slots as they stand during each cycle (Q: observed after the
    // previous tick), and the cycle the store's round lands.
    bool both_waiting = false;
    std::optional<uint64_t> store_cycle;
    bool store_while_both_waiting = false;
    const auto written = i1_run(*machine, {3, 6, 9}, [&](uint64_t cycle) {
        if (!store_cycle.has_value() && machine->read_shmem_u32(1024) == 7) {
            store_cycle = cycle;
            store_while_both_waiting = both_waiting;
        }
        both_waiting = machine->load_slot_complete(0) && machine->load_slot_complete(1);
    });
    const auto lds = [&](uint32_t n) { return issued_at(*machine, (11 + n) * 8 + (n == 2 ? 8 : 0)); };
    const auto at = [](const std::vector<uint64_t>& cycles) { return cycles.empty() ? std::string("-") : std::to_string(cycles.front()); };
    const auto observed = " lds " + at(lds(0)) + " " + at(lds(1)) + " " + at(lds(2)) + "; written " + at(written.at(3)) +
        " " + at(written.at(6)) + " " + at(written.at(9)) + "; refused" + describe(refusals(*machine, 0, "load slots full")) +
        "; store c" + (store_cycle.has_value() ? std::to_string(*store_cycle) : std::string("-")) +
        "; issues" + issue_log(*machine);
    require(lds(0).size() == 1 && lds(1) == std::vector<uint64_t>{lds(0)[0] + 1}, "the first two lds.uni:" + observed);
    const auto l0 = lds(0)[0];
    const auto g0 = l0 + 16;
    require(issued_at(*machine, 13 * 8) == std::vector<uint64_t>{l0 + 5},
        "the sts did not issue as the second load's round was granted:" + observed);
    // The sts behind the second load (L0 + 2 .. L0 + 4), the third load behind
    // the sts (L0 + 6 .. L0 + 8): issue_to_dispatch cycles each.
    const auto t = nanosm::issue_to_dispatch(profile);
    require(refusals(*machine, 0, "ordered behind own access") ==
            std::vector<std::pair<uint64_t, uint64_t>>{{l0 + 2, t}, {l0 + 6, t}},
        "the sts was not ordered behind the second load until its grant (L0 + 5), or the third lds.uni behind "
        "the sts until its grant (L0 + 9):" + observed + "; ordered" +
            describe(refusals(*machine, 0, "ordered behind own access")));
    const auto refused = refusals(*machine, 0, "load slots full");
    require(refused == std::vector<std::pair<uint64_t, uint64_t>>{{l0 + 9, g0 + 1 - l0 - 9}},
        "the third lds.uni was not refused for load slots from L0 + 9 through slot 0's write grant (G0):" + observed);
    require(machine->trace().summary().shmem_load_slot_waits == g0 + 1 - l0 - 9, "the refusals were not counted");
    require(lds(2) == std::vector<uint64_t>{g0 + 1} && written.at(3) == std::vector<uint64_t>{g0 + wd},
        "the third lds.uni did not take slot 0 the cycle after its write was granted:" + observed);
    require(written.at(6) == std::vector<uint64_t>{g0 + 1 + wd} &&
            written.at(9) == std::vector<uint64_t>{g0 + 1 + x + 2 + wd},
        "results:" + observed);
    require(machine->read_reg(0, 7, 9) == 0x102, "the third load's word");
    require(store_cycle == l0 + 11 && store_while_both_waiting,
        "port A did not serve the sts while both load slots held results waiting for the RF:" + observed);
    for (uint32_t lane = 0; lane < 16; ++lane) require(machine->read_shmem_u32(1024 + 4 * lane) == 7, "sts data");
}

// shmem.load_slots: optional in the profile JSON (default 2), an integer of
// at least 1, part of the fingerprint. With one slot, back-to-back lds.uni
// serialize on it: the second takes it the cycle after the first's write is
// granted (issue counts the slot empty in Q).
void test_i1_load_slots_profile_field() {
    std::ifstream input("profiles/nanosm-v0.json");
    nlohmann::json document;
    input >> document;
    require(nanosm::parse_device_profile(document.dump()).shmem_load_slots == 2,
        "the canonical profile does not declare two load slots");
    auto defaulted = document;
    defaulted["shmem"].erase("load_slots");
    require(nanosm::parse_device_profile(defaulted.dump()).shmem_load_slots == 2, "the default is not 2");
    for (const auto value : {nlohmann::json(0), nlohmann::json(-1), nlohmann::json(1.5)}) {
        auto modified = document;
        modified["shmem"]["load_slots"] = value;
        try {
            (void)nanosm::parse_device_profile(modified.dump());
            throw std::runtime_error("invalid load_slots " + value.dump() + " was accepted");
        } catch (const nanosm::ProfileError&) {}
    }
    auto zero = canonical_profile();
    zero.shmem_load_slots = 0;
    try {
        nanosm::validate_device_profile(zero);
        throw std::runtime_error("zero load slots passed validation");
    } catch (const nanosm::ProfileError&) {}
    auto three = canonical_profile();
    three.shmem_load_slots = 3;
    require(nanosm::device_profile_fingerprint(three) != nanosm::device_profile_fingerprint(canonical_profile()),
        "load slots were omitted from the profile fingerprint");
    auto one = canonical_profile();
    one.shmem_load_slots = 1;
    auto machine = i1_machine(one, R"(
        mov r61, 0
        lds.uni r3, [ur1] {pred=PT, stall=0, set=sb0, wait=-}
        lds.uni r6, [ur1 + 4] {pred=PT, stall=0, set=sb1, wait=-}
        halt {pred=PT, stall=0, set=-, wait=sb0|sb1}
    )");
    const auto written = i1_run(*machine, {3, 6});
    // I2: the slot is held while the first load's words come back (D + Lb).
    const auto refused = refusals(*machine, 0, "load slots full");
    require(refused == std::vector<std::pair<uint64_t, uint64_t>>{{i1_launch_shift + 2, 2 + i1_lds_shift(one)}},
        "the second lds.uni was not refused until the one slot freed:" + describe(refused));
    require(written.at(3).size() == 1 &&
            issued_at(*machine, 2 * 8) == std::vector<uint64_t>{written.at(3)[0] - one.rf_write_depth_cycles + 1},
        "the second lds.uni did not take the slot the cycle after it freed (the first's write grant, Wd before "
        "it lands)");
    require(machine->read_reg(0, 0, 6) == 0x101, "the second load's word");
}

// I1.1: the LSU has one RF write path (rfc09 section 3a: one lane-wide vector
// per cycle), so its output mux presents ONE assembly register per cycle to the
// RF write decision, round-robin from the LSU cursor. Warp 0 issues two ldg.uni:
// r9 (bank 0) then r10 (bank 1). Warp 1 keeps bank 0 booked with fadds, so the
// first load waits for its bank while the second completes: both assembly
// registers hold complete vectors at once, with bank 1 free. The LSU still
// writes only the register at its cursor (the bank-0 load, when bank 0 opens)
// and the other one afterwards -- never the second one past the first, and
// never two in one cycle.
void test_i1_lsu_one_rf_write_per_cycle() {
    auto config = canonical_profile();
    config.gmem_latency_dram_ticks = 64 * 25; // the loads return while warp 1's fadds book bank 0 every cycle
    config.gmem_bus_bytes_per_dram_tick = config.gmem_sector_bytes;
    config.dram_clock_hz = config.core_clock_hz * 64;
    config.lsu_pluck_words_per_cycle = 64; // a load completes as soon as its sector arrives
    nanosm::Launch launch;
    launch.active_warps = 2;
    auto& image = image_bytes(launch.gmem_image, 1024);
    fill_pattern(image, 0, 256, 0);
    nanosm::Machine machine(config, compile(R"(
        usread ur2, WARP_ID
        ur2r r2, ur2
        setp.ne.u32 p0, r2, 0
@p0     bra.uni ARITH
        umov ur0, 0
        umov ur1, 512
        ldg.uni r9, [ur0]
        ldg.uni r10, [ur1]
        halt
ARITH:
        mov r4, 1
        mov r5, 2
        fadd r3, r4, r5
        fadd r6, r4, r5
        fadd r24, r4, r5
        fadd r27, r4, r5
        fadd r30, r4, r5
        fadd r33, r4, r5
        fadd r36, r4, r5
        fadd r39, r4, r5
        fadd r3, r4, r5
        fadd r6, r4, r5
        fadd r24, r4, r5
        fadd r27, r4, r5
        fadd r30, r4, r5
        fadd r33, r4, r5
        fadd r36, r4, r5
        fadd r39, r4, r5
        fadd r3, r4, r5
        fadd r6, r4, r5
        fadd r24, r4, r5
        fadd r27, r4, r5
        fadd r30, r4, r5
        fadd r33, r4, r5
        fadd r36, r4, r5
        fadd r39, r4, r5
        fadd r3, r4, r5
        fadd r6, r4, r5
        fadd r24, r4, r5
        fadd r27, r4, r5
        fadd r30, r4, r5
        fadd r33, r4, r5
        fadd r36, r4, r5
        fadd r39, r4, r5
        fadd r3, r4, r5
        fadd r6, r4, r5
        fadd r24, r4, r5
        fadd r27, r4, r5
        fadd r30, r4, r5
        fadd r33, r4, r5
        fadd r36, r4, r5
        fadd r39, r4, r5
        fadd r3, r4, r5
        fadd r6, r4, r5
        fadd r24, r4, r5
        fadd r27, r4, r5
        fadd r30, r4, r5
        fadd r33, r4, r5
        fadd r36, r4, r5
        fadd r39, r4, r5
        fadd r3, r4, r5
        fadd r6, r4, r5
        fadd r24, r4, r5
        fadd r27, r4, r5
        fadd r30, r4, r5
        fadd r33, r4, r5
        fadd r36, r4, r5
        fadd r39, r4, r5
        fadd r3, r4, r5
        fadd r6, r4, r5
        fadd r24, r4, r5
        fadd r27, r4, r5
        fadd r30, r4, r5
        fadd r33, r4, r5
        fadd r36, r4, r5
        fadd r39, r4, r5
        halt
    )", config), std::move(launch));
    machine.enable_edge_probe();
    std::vector<std::tuple<uint64_t, uint64_t, uint64_t, uint32_t>> both_ready; // (cycle, owner 0, owner 1, cursor)
    while (!machine.done()) {
        const auto cycle = machine.cycle();
        if (machine.assembly_valid_count(0) == 16 && machine.assembly_valid_count(1) == 16 &&
            machine.assembly_owner(0).has_value() && machine.assembly_owner(1).has_value()) {
            both_ready.emplace_back(cycle, *machine.assembly_owner(0), *machine.assembly_owner(1),
                machine.rf_write_lsu_cursor());
        }
        tick_capped(machine);
    }
    std::map<uint64_t, std::vector<uint64_t>> writes_at; // cycle -> command ids written
    std::map<uint64_t, uint64_t> write_cycle;            // command id -> its RF write cycle
    for (const auto& event : machine.edge_events()) {
        if (event.edge != LsuEdge::rf_write_beat) continue;
        writes_at[event.cycle].push_back(event.id);
        write_cycle[event.id] = event.cycle;
    }
    for (const auto& [cycle, ids] : writes_at) {
        require(ids.size() == 1, "the LSU wrote " + std::to_string(ids.size()) + " RF beats in cycle " +
            std::to_string(cycle));
    }
    require(write_cycle.size() == 2, "two ldg.uni did not make two LSU RF writes");
    require(!both_ready.empty(), "the two assembly registers were never ready together (the test is vacuous)");
    for (const auto& [cycle, owner0, owner1, cursor] : both_ready) {
        const auto presented = cursor == 0 ? owner0 : owner1;
        const auto other = cursor == 0 ? owner1 : owner0;
        if (writes_at.count(cycle) == 1) {
            require(writes_at.at(cycle).front() == presented,
                "cycle " + std::to_string(cycle) + ": a register other than the one at the LSU cursor was written");
        }
        require(write_cycle.at(other) > write_cycle.at(presented),
            "the register behind the LSU cursor was written before the one at the cursor");
    }
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(machine.read_reg(0, lane, 9) == pattern_word(0), "the first ldg.uni's word");
        require(machine.read_reg(0, lane, 10) == pattern_word(128), "the second ldg.uni's word");
    }
}

// ---------------------------------------------------------------------------
// I6a: the RF's read latency Lr and write depth Wd, the issue register, and
// src/rf.hpp as the ONE timing helper of qtxc, the simulator and the
// sanitizer (trash/rfc10.md section 5, trash/i6a-rf-pipeline-2026-10-13.md).
// Port A's edge convention: a read SAMPLED at S has its operand at S + Lr; a
// write PRESENTED at the RF write port at W is at the RAM at W + Wd, lands at
// that edge and is visible to samples from W + Wd + 1 -- no forwarding.
// The debug trace shows the samples ("RF read") and the presentations ("RF
// write"); the landing is the cycle whose tick changed the register.
// ---------------------------------------------------------------------------

// r2ur reads its source through the vector RF, so the uniform writers' fixed
// path to the URF write port (urf.issue_to_port_cycles) must cover the issue
// register + Lr: it rises with a longer Lr (canonical 3 covers Lr 1 and 2).
nanosm::DeviceProfile i6a_profile(uint32_t lr, uint32_t wd) {
    auto profile = canonical_profile();
    profile.rf_read_latency_cycles = lr;
    profile.rf_write_depth_cycles = wd;
    profile.urf_issue_to_port_cycles = std::max(profile.urf_issue_to_port_cycles, 1 + lr);
    nanosm::validate_device_profile(profile);
    return profile;
}

const std::vector<std::pair<uint32_t, uint32_t>> i6a_depths = {{1, 1}, {2, 2}, {3, 1}};

std::string i6a_what(const nanosm::DeviceProfile& profile) {
    return "Lr/Wd " + std::to_string(profile.rf_read_latency_cycles) + "/" +
        std::to_string(profile.rf_write_depth_cycles) + ": ";
}

// The helper contract: for an add, an fadd, an SFU op, an lds, an sts and
// an acc.ld, the cycles the trace shows -- the samples, the port, the RAM
// (the register's landing edge), the visibility (a depbar waiter's issue) --
// are the helper's, relative to each instruction's accepted issue i, at
// Lr/Wd 1/1, 2/2 and 3/1 (t = issue_to_dispatch):
//   fixed:  samples i + issue_to_sample, port i + issue_to_port, RAM port +
//           Wd, visible i + issue_to_visible = RAM + 1;
//   lds:    address sampled at i + 1; A-read grant i + t + 1; the words at
//           port A's output D + Lb later; the RF port the cycle after; RAM
//           + Wd; its depbar clears at the visibility, when the waiting add
//           issues (the counter is 1 through the RAM cycle, 0 from RAM + 1);
//   sts:    address and data sampled at i + 1; A-write grant i + t + 1, the
//           word at the shmem RAM D later;
//   acc.ld: its tmem pointer (a uniform register) sampled from the URF at i + 1,
//           no RF read; tmem read (beat 0) at i + t + 1; RF port i + t + 2;
//           RAM + Wd; the waiting add at RAM + 1.
// Every RF write's visibility is at least earliest_rf_visible after its
// issue. The shli's stall lets every fixed result reach the RAM before the
// async section, and the acc.ld goes before the lds (its waiter first), so
// no write port or issue slot is contested: the uncontended pipeline.
void test_i6a_helper_contract() {
    for (const auto& [lr, wd] : i6a_depths) {
        const auto profile = i6a_profile(lr, wd);
        const auto what = i6a_what(profile);
        const auto kernel = hand_assemble(i1_asm(profile, R"(
            mov r1, 7
            mov r2, 0x3f8
            shli r2, r2, 20
            mov r11, 77
            umov ur8, 0
            umov ur9, 0
            sread r0, LANE_ID
            add r3, r0, r1
            fadd r4, r2, r2
            ex2 r5, r2
            shli r6, r0, 2 {pred=PT, stall=16, set=-, wait=-}
            acc.ld r11, [ur8, 64] {pred=PT, stall=0, set=sb2, wait=-}
            lds r7, [r6 + 256] {pred=PT, stall=0, set=sb0, wait=-}
            sts [r6 + 512], r4 {pred=PT, stall=0, set=sb1, wait=-}
            add r14, r11, r11 {pred=PT, stall=0, set=-, wait=sb2}
            add r12, r7, r7 {pred=PT, stall=0, set=-, wait=sb0}
            halt {pred=PT, stall=0, set=-, wait=sb1}
        )"), profile);
        auto machine = i6a_machine(profile, kernel);
        std::optional<uint64_t> stored; // the cycle whose edge put the sts's lane-0 word in the RAM
        std::map<uint64_t, uint32_t> sb0;  // depbar sb0 (Q) during each cycle
        const auto landed = i1_run(*machine, {3, 4, 5, 7, 11}, [&](uint64_t cycle) {
            if (!stored.has_value() && machine->read_shmem_u32(512) == nanosm::float_bits(2.0f)) stored = cycle;
            sb0[cycle + 1] = machine->depbar_value(0, 0);
        });
        const auto issue = [&](uint32_t index) {
            const auto cycles = issued_at(*machine, index * 8);
            require(cycles.size() == 1, what + "instruction " + std::to_string(index) + " did not issue once");
            return cycles[0];
        };
        const auto inst = [&](uint32_t index) { return nanosm::decode(kernel.words.at(index)); };
        const auto t = nanosm::issue_to_dispatch(profile);
        const auto floor = nanosm::earliest_rf_visible(profile);
        const auto log = " | issues" + issue_log(*machine);
        const auto sampled_once = [&](uint8_t reg, uint64_t cycle) {
            const auto samples = i6a_rf_events(*machine, "RF read", reg);
            return std::count(samples.begin(), samples.end(), cycle) == 1;
        };

        // Fixed results: the add (sources in banks 0 and 1), the fadd, the ex2.
        for (const auto& [index, dst, sources] : std::vector<std::tuple<uint32_t, uint8_t, std::vector<uint8_t>>>{
                 {7, 3, {0, 1}}, {8, 4, {2}}, {9, 5, {2}}}) {
            const auto i = issue(index);
            const auto op = inst(index);
            const auto name = what + nanosm::opcode_name(op.opcode) + ": ";
            for (const auto reg : nanosm::rf_read_round(sources)) {
                require(sampled_once(reg, i + nanosm::issue_to_sample),
                    name + "r" + std::to_string(reg) + " was not sampled at issue + issue_to_sample" + log);
            }
            const auto port = i + nanosm::issue_to_port(op, profile);
            require(i6a_rf_events(*machine, "RF write", dst) == std::vector<uint64_t>{port},
                name + "not presented at the RF write port at issue + issue_to_port" + log);
            require(landed.at(dst) == std::vector<uint64_t>{port + wd}, name + "did not land at port + Wd");
            require(nanosm::issue_to_visible(op, profile) == nanosm::issue_to_port(op, profile) + wd + 1 &&
                    nanosm::issue_to_visible(op, profile) >= floor,
                name + "visibility is not the landing + 1, or under the floor");
        }

        // lds r7, [r6 + 256]: one round; its depbar waiter is the add r12.
        {
            const auto i = issue(12);
            const auto name = what + "lds: ";
            require(sampled_once(6, i + nanosm::issue_register_stages),
                name + "the address was not sampled after the issue register");
            const auto reads = i2_grants(*machine, "A-read");
            require(reads.size() == 1 && reads[0].cycle == i + t + 1, name + "A-read grant:" + describe(reads));
            const auto port = reads[0].cycle + i2_return(profile) + 1;
            require(i6a_rf_events(*machine, "RF write", 7) == std::vector<uint64_t>{port},
                name + "not presented at the RF port the cycle after its words" + log);
            require(landed.at(7) == std::vector<uint64_t>{port + wd}, name + "did not land at port + Wd");
            require(sb0.at(port + wd) == 1 && sb0.at(port + wd + 1) == 0,
                name + "the depbar did not clear at the visibility (W + Wd + 1)");
            require(issue(15) == port + wd + 1, name + "the waiting add did not issue at the visibility" + log);
            require(port + wd + 1 - i >= floor, name + "visible before the floor");
        }

        // sts [r6 + 512], r4: one round.
        {
            const auto i = issue(13);
            const auto name = what + "sts: ";
            require(sampled_once(4, i + nanosm::issue_register_stages) && sampled_once(6, i + nanosm::issue_register_stages),
                name + "the address and data were not sampled after the issue register" + log);
            const auto writes = i2_grants(*machine, "A-write");
            require(writes.size() == 1 && writes[0].cycle == i + t + 1, name + "A-write grant:" + describe(writes));
            require(stored == writes[0].cycle + profile.shmem_port_a_depth, name + "the word did not reach the RAM D after the grant");
        }

        // acc.ld r11, [ur8, 64]: one beat; its depbar waiter is the add r14.
        {
            const auto i = issue(11);
            const auto name = what + "acc.ld: ";
            // Its coordinates are uniform registers: its read round samples
            // no RF bank (nothing else reads in that cycle).
            for (const auto& event : machine->trace().events()) {
                require(!(event.name == "RF read" && event.start_cycle == i + nanosm::issue_register_stages),
                    name + "the read round sampled the RF (r" + std::to_string(event.id) + ")" + log);
            }
            std::vector<uint64_t> beats;
            for (const auto& event : machine->trace().events()) {
                if (event.name == "acc.ld beat") beats.push_back(event.start_cycle);
            }
            // I8: the engine launches the read at dispatch + 1; it samples tmem
            // tmem.port_depth later and is at the pipeline's output (the RF
            // port) tmem.read_latency_cycles after that.
            const auto launch = i + t + 1;
            require(beats == std::vector<uint64_t>{launch + profile.tmem_port_depth},
                name + "the tmem sample was not at the launch (dispatch + 1) + tmem port depth");
            const auto port = launch + profile.tmem_port_depth + profile.tmem_read_latency_cycles;
            require(i6a_rf_events(*machine, "RF write", 11).back() == port, // after the mov's 77
                name + "not presented at the RF port after the pipeline stage" + log);
            require(landed.at(11).size() == 2 && landed.at(11).back() == port + wd,
                name + "did not land at port + Wd (after the mov's 77)");
            require(issue(14) == port + wd + 1, name + "the waiting add did not issue at the visibility" + log);
            require(port + wd + 1 - i >= floor, name + "visible before the floor");
        }
        require(machine->read_reg(0, 5, 3) == 12 && machine->read_reg(0, 5, 7) == 0x100 + 64 + 5 &&
                machine->read_reg(0, 5, 11) == 0 && machine->read_reg(0, 0, 5) == nanosm::float_bits(2.0f),
            what + "data");
    }
}

// The qtxc distance for a dependent add -> add is issue_to_visible -
// issue_register_stages (the consumer samples one cycle after its issue);
// hand assembly with exactly that stall runs and reads the new value. One
// cycle less puts the consumer's sample at the producer's RAM cycle W + Wd:
// the sanitizer PANICs (RF RAW); without it the machine is quiet and the
// sample sees the OLD word -- the write lands at that edge, after every read
// of the cycle (READ_FIRST, no forwarding).
void test_i6a_dependent_add_distance() {
    for (const auto& [lr, wd] : i6a_depths) {
        const auto profile = i6a_profile(lr, wd);
        const auto what = i6a_what(profile);
        const auto scheduled = nanosm::compile_qtx("add r1, r0, r0\nadd r2, r1, r1\nhalt\n", profile);
        const auto distance = nanosm::issue_to_visible(scheduled.instructions[0], profile) - nanosm::issue_register_stages;
        require(scheduled.instructions[0].control.stall == distance, what + "qtxc's stall is not the helper's distance");
        const auto run = [&](uint64_t stall) {
            const auto text = i1_asm(profile, "sread r0, LANE_ID {pred=PT, stall=" +
                std::to_string(alu_distance(profile)) + ", set=-, wait=-}\nadd r1, r0, r0 {pred=PT, stall=" +
                std::to_string(stall) + ", set=-, wait=-}\nadd r2, r1, r1\nhalt\n");
            return i6a_machine(profile, nanosm::assemble_asm(text, profile));
        };
        auto exact = run(distance);
        exact->run();
        require(issued_at(*exact, 16).at(0) - issued_at(*exact, 8).at(0) == distance, what + "the consumer moved");
        require(exact->read_reg(0, 5, 2) == 20, what + "at the qtxc distance the add did not read the new r1");
        auto early = run(distance - 1);
#if NANOSM_SANITIZER
        require_mentions(i4_panic(*early, what + "one cycle early"),
            {"RF RAW hazard: missing stall or depbar wait for r1", "inst='add r2, r1, r1"}, what + "one cycle early");
#else
        early->run();
        require(early->read_reg(0, 5, 1) == 10 && early->read_reg(0, 5, 2) == 0,
            what + "one cycle early, the sample at the landing edge did not see the OLD r1 (write-first forwarding?)");
#endif
    }
}

// WAW: an add after an fadd to the same register must land after it. qtxc
// stalls the fadd fadd - alu + 1 cycles (the younger's landing one past the
// older's); then r3 holds the add's value. Two cycles less the add lands
// first: the sanitizer PANICs (RF WAW); without it the older fadd's value
// lands last and stays. (One cycle less the two results meet at one write
// port in one cycle and the add is refused for the port: structurally
// safe.)
void test_i6a_waw_order() {
    for (const auto& [lr, wd] : i6a_depths) {
        const auto profile = i6a_profile(lr, wd);
        const auto what = i6a_what(profile);
        const auto scheduled = nanosm::compile_qtx("fadd r3, r1, r1\nadd r3, r2, r2\nhalt\n", profile);
        const uint64_t stall = profile.timing.fadd - profile.timing.alu + 1;
        require(scheduled.instructions[0].control.stall == stall, what + "qtxc's WAW stall");
        // r1 = 1.0f (0x3f8 << 20), r2 = 3; each mov / shli stalls its full
        // distance (assembled as written: the stalls under test are not raised).
        const auto run = [&](uint64_t fadd_stall) {
            const auto distance = std::to_string(alu_distance(profile));
            const auto text = i1_asm(profile, "mov r1, 0x3f8 {pred=PT, stall=" + distance + ", set=-, wait=-}\n"
                "shli r1, r1, 20\nmov r2, 3 {pred=PT, stall=" + distance + ", set=-, wait=-}\n"
                "fadd r3, r1, r1 {pred=PT, stall=" + std::to_string(fadd_stall) + ", set=-, wait=-}\n"
                "add r3, r2, r2\nhalt\n");
            return i6a_machine(profile, nanosm::assemble_asm(text, profile));
        };
        auto ordered = run(stall);
        ordered->run();
        require(ordered->read_reg(0, 0, 3) == 6, what + "the younger add's value did not land last");
        auto early = run(stall - 2);
#if NANOSM_SANITIZER
        require_mentions(i4_panic(*early, what + "WAW"), {"RF WAW hazard: missing stall or depbar wait for r3"},
            what + "WAW");
#else
        early->run();
        require(early->read_reg(0, 0, 3) == nanosm::float_bits(2.0f), what + "the add did not land before the fadd");
#endif
    }
}

// An async writer against a fixed booking: an lds.uni's result reaches the
// RF write port at P (issue + i1_lds_shift + 2) in bank 0, where an add
// accepted earlier (at P - issue_to_port) booked bank 0's one port. The
// booking wins; the lds's write is denied (one wait), granted at P + 1 and
// lands Wd later; the add's lands at P + Wd. (I6b-2: the lds.uni issues at
// fetch_to_issue; its stall places the add.)
void test_i6a_async_writer_vs_fixed_booking() {
    for (const auto& [lr, wd] : i6a_depths) {
        const auto profile = i6a_profile(lr, wd);
        const auto what = i6a_what(profile);
        nanosm::Instruction add;
        add.opcode = nanosm::Opcode::add;
        const auto lds_issue = nanosm::fetch_to_issue;
        const auto p = lds_issue + i1_lds_shift(profile) + 2;
        const auto add_issue = p - nanosm::issue_to_port(add, profile);
        auto machine = i6a_machine(profile, hand_assemble(i1_asm(profile,
            "lds.uni r3, [ur1] {pred=PT, stall=" + std::to_string(add_issue - lds_issue) + ", set=sb0, wait=-}\n"
            "add r6, r1, r1\n"
            "halt {pred=PT, stall=0, set=-, wait=sb0}\n"), profile));
        const auto landed = i1_run(*machine, {3});
        require(issued_at(*machine, 8) == std::vector<uint64_t>{add_issue}, what + "the add moved" + issue_log(*machine));
        require(i6a_rf_events(*machine, "RF write", 6) == std::vector<uint64_t>{p} &&
                i6a_rf_events(*machine, "RF write", 3) == std::vector<uint64_t>{p + 1},
            what + "the booked add did not keep the port at P, or the lds was not written at P + 1");
        require(landed.at(3) == std::vector<uint64_t>{p + 1 + wd} && machine->trace().summary().rf_async_waits == 1,
            what + "the lds's write did not land at P + 1 + Wd after one wait");
        require(machine->read_reg(0, 0, 3) == 0x100, what + "lds data");
    }
}

// Slots are claimed at admission and held while the operands travel (the
// coupled lds / sts slots stay reserved from accept; the unit takes the
// instruction issue_to_dispatch later). One load slot: lds.uni A accepted at
// a; lds.uni B, tried from a + 1, is refused "load slots full" although the
// slot register is still empty (A is in the dispatch pipeline). One store
// slot, two warps: warp 1's sts, tried the cycle after warp 0's, is refused
// "store slots full" while warp 0's sts travels and its slot is still empty.
void test_i6a_slots_claimed_while_operands_travel() {
    auto profile = canonical_profile();
    profile.shmem_load_slots = 1;
    profile.shmem_store_slots = 1;
    const auto t = nanosm::issue_to_dispatch(profile);
    {
        auto machine = i6a_machine(profile, hand_assemble(i1_asm(profile, R"(
            lds.uni r3, [ur1] {pred=PT, stall=0, set=sb0, wait=-}
            lds.uni r6, [ur1 + 4] {pred=PT, stall=0, set=sb1, wait=-}
            halt {pred=PT, stall=0, set=-, wait=sb0|sb1}
        )"), profile));
        std::map<uint64_t, bool> held; // the slot register (Q) during each cycle
        while (!machine->done()) {
            held[machine->cycle()] = machine->load_slot_held(0);
            tick_capped(*machine);
        }
        const auto a = issued_at(*machine, 0).at(0);
        const auto refused = refusals(*machine, 0, "load slots full");
        require(!refused.empty() && refused.front().first == a + 1, "B was not refused from a + 1:" + describe(refused));
        for (uint64_t cycle = a + 1; cycle <= a + t; ++cycle) {
            require(!held.at(cycle), "the slot register was not empty while A's operands travelled");
        }
        require(held.at(a + t + 1), "A did not take the slot at its dispatch");
    }
    {
        auto machine = i6a_machine(profile, hand_assemble(i1_asm(profile, R"(
            sread r0, LANE_ID
            usread ur1, WARP_ID
            ur2r r1, ur1
            shli r0, r0, 2
            shli r1, r1, 6
            add r0, r0, r1
            sts [r0 + 1024], r1 {pred=PT, stall=0, set=sb0, wait=-}
            halt {pred=PT, stall=0, set=-, wait=sb0}
        )"), profile), 2);
        std::map<uint64_t, bool> held;
        while (!machine->done()) {
            held[machine->cycle()] = machine->store_slot_held(0);
            tick_capped(*machine);
        }
        const auto issues = issued_at(*machine, 6 * 8);
        require(issues.size() == 2 && issues[1] > issues[0], "two sts:" + issue_log(*machine));
        const auto refused = refusals(*machine, 1, "store slots full");
        require(!refused.empty() && refused.front().first == issues[0] + 1 && !held.at(issues[0] + 1),
            "warp 1's sts was not refused for the slot warp 0's travelling sts claimed:" + describe(refused) +
                issue_log(*machine));
        for (uint32_t lane = 0; lane < 16; ++lane) {
            require(machine->read_shmem_u32(1024 + 4 * lane) == 0 && machine->read_shmem_u32(1024 + 64 + 4 * lane) == 64,
                "sts data");
        }
    }
}

// rf.read_latency_cycles (Lr) and rf.write_depth_cycles (Wd): optional in
// the JSON (absent: the canonical 2), nonzero register stages (1..8), part of
// the fingerprint. And the longest fixed dependency must fit the five-bit
// stall field (src/profile.cpp): Lr + E + Wd + 1 <= 31, E the longest fixed
// execution (canonical: the effective SFU latency, 11). (A longer Lr also
// needs the uniform writers' path urf.issue_to_port_cycles >= 1 + Lr: r2ur
// reads the vector RF. So Lr's checks are i21_check_optional_field's with
// that path raised for the other value, below.)
void check_rf_read_latency_field() {
    std::ifstream input("profiles/nanosm-v0.json");
    nlohmann::json document;
    input >> document;
    const auto parsed = nanosm::parse_device_profile(document.dump());
    require(parsed.rf_read_latency_cycles == 2 && document["rf"].contains("read_latency_cycles"),
        "rf.read_latency_cycles: the canonical profile does not declare 2");
    auto defaulted = document;
    defaulted["rf"].erase("read_latency_cycles");
    require(nanosm::device_profile_fingerprint(nanosm::parse_device_profile(defaulted.dump())) ==
            nanosm::device_profile_fingerprint(parsed), "rf.read_latency_cycles: the default is not 2");
    for (const auto value : {nlohmann::json(0), nlohmann::json(-1), nlohmann::json(1.5), nlohmann::json(65)}) {
        auto modified = document;
        modified["rf"]["read_latency_cycles"] = value;
        try {
            (void)nanosm::parse_device_profile(modified.dump());
            throw std::runtime_error("rf.read_latency_cycles = " + value.dump() + " was accepted");
        } catch (const nanosm::ProfileError&) {}
    }
    auto zero = canonical_profile();
    zero.rf_read_latency_cycles = 0;
    try {
        nanosm::validate_device_profile(zero);
        throw std::runtime_error("rf.read_latency_cycles = 0 passed validation");
    } catch (const nanosm::ProfileError&) {}
    // Lr 3 against the canonical Lr, both with the path Lr 3 needs.
    auto path = canonical_profile();
    path.urf_issue_to_port_cycles = 1 + 3;
    auto other = path;
    other.rf_read_latency_cycles = 3;
    require(nanosm::device_profile_fingerprint(other) != nanosm::device_profile_fingerprint(path),
        "rf.read_latency_cycles was omitted from the profile fingerprint");
}

void test_i6a_profile_fields() {
    check_rf_read_latency_field();
    i21_check_optional_field("rf", "write_depth_cycles", 2, &nanosm::DeviceProfile::rf_write_depth_cycles);
    auto profile = canonical_profile();
    profile.rf_read_latency_cycles = 8;
    profile.rf_write_depth_cycles = 8;
    profile.urf_issue_to_port_cycles = 1 + 8;
    profile.timing.sfu = static_cast<uint8_t>(31 - (8 + 8 + 1)); // exactly 31
    nanosm::validate_device_profile(profile);
    ++profile.timing.sfu; // 32
    try {
        nanosm::validate_device_profile(profile);
        throw std::runtime_error("a fixed dependency longer than the stall field passed validation");
    } catch (const nanosm::ProfileError& error) {
        require(std::string(error.what()).find("five-bit stall field") != std::string::npos,
            "the stall-field refusal has the wrong text");
    }
}

// ---------------------------------------------------------------------------
// I6b-1: the RF read side of trash/rfc14.md (sections 3-5) and 16 warps.
// Every instruction reads its operands in ONE round, sampled at issue +
// issue_to_sample; one instruction is accepted per cycle, so reads of
// different instructions never meet. The issue register's bank use owns the
// RF read ports in its sample cycle; acc.st beats and the LSU's store-data
// reads take the rest, with read_owed / inhibit making a hole for a denied
// reader. trash/rf-read-side-16-warps-2026-10-18.md.
// ---------------------------------------------------------------------------

// Every debug "RF read" sample of warp 0, by cycle: (register, bank).
std::map<uint64_t, std::vector<std::pair<uint32_t, uint32_t>>> i6b1_reads(const nanosm::Machine& machine) {
    std::map<uint64_t, std::vector<std::pair<uint32_t, uint32_t>>> reads;
    for (const auto& event : machine.trace().events()) {
        if (event.name == "RF read" && event.track == 100) {
            reads[event.start_cycle].emplace_back(static_cast<uint32_t>(event.id), static_cast<uint32_t>(event.value));
        }
    }
    return reads;
}

// Back-to-back same-bank reads. Nine independent instructions issue in
// consecutive cycles: eight read RF bank 0 (r0, r3, r6, r9: two-source adds,
// a repeated register, xori, addi, stg, lds, setp) and an lds.uni among them
// reads only its uniform base (ur0, no RF read). No refusal, no bubble; bank
// 0 is sampled in every cycle of the run but the lds.uni's, each
// instruction's sources at its own issue + issue_to_sample; the data are
// right. At Lr 1, 2 and 3 -- the RF's output latency changes when operands
// arrive, never when or whether an instruction is accepted. Then four warps
// run the arithmetic part of it: their issues interleave one per cycle with
// no gap, and bank 0 is sampled once per cycle.
void test_i6b1_back_to_back_same_bank_reads() {
    const std::string setup = R"(
        sread r9, LANE_ID
        usread ur4, WARP_ID
        ur2r r4, ur4
        shli r9, r9, 2
        shli r4, r4, 6
        add r9, r9, r4
        mov r0, 4
        umov ur0, 4
        mov r3, 8
        mov r6, 12
        mov r1, 100
        mov r2, 200
    )";
    // The run, from pc 96: (each instruction's distinct vector sources).
    const std::vector<std::vector<uint32_t>> mixed_sources{
        {0, 1}, {3, 2}, {6}, {9}, {}, {9, 1}, {6}, {9}, {9}};
    const std::string mixed = setup + R"(
        add r10, r0, r1
        add r11, r3, r2
        xori r13, r6, 5
        addi r14, r9, 1
        lds.uni r16, [ur0 + 512] {pred=PT, stall=0, set=sb0, wait=-}
        stg [r9 + 4096], r1 {pred=PT, stall=0, set=sb1, wait=-}
        add r17, r6, r6
        lds r19, [r9 + 2048] {pred=PT, stall=0, set=sb2, wait=-}
        setp.ne.u32 p0, r9, 0
        halt {pred=PT, stall=0, set=-, wait=sb0|sb1|sb2}
    )";
    const std::vector<std::vector<uint32_t>> arithmetic_sources{{0, 1}, {3, 2}, {6}, {9}, {6}, {9}};
    const std::string arithmetic = setup + R"(
        add r10, r0, r1
        add r11, r3, r2
        xori r13, r6, 5
        addi r14, r9, 1
        add r17, r6, r6
        setp.ne.u32 p0, r9, 0
        halt
    )";
    for (const uint32_t lr : {1u, 2u, 3u}) {
        const auto profile = i6a_profile(lr, 2);
        for (const auto& [warps, body, sources] : std::vector<std::tuple<uint32_t, std::string,
                 std::vector<std::vector<uint32_t>>>>{{1, mixed, mixed_sources}, {4, arithmetic, arithmetic_sources}}) {
            const auto what = i6a_what(profile) + std::to_string(warps) + " warp(s): ";
            auto machine = i6a_machine(profile, hand_assemble(i1_asm(profile, body), profile), warps);
            machine->run();
            const auto log = " | issues" + issue_log(*machine);
            require(machine->trace().summary().rf_simt_read_inhibited == 0, what + "a read was inhibited");
            std::set<uint64_t> run_issues;
            std::set<uint64_t> bank0_rounds;            // the read rounds of run instructions that read bank 0
            std::map<uint64_t, uint32_t> bank0_samples; // SIMT samples of bank 0 by cycle
            for (const auto& event : machine->trace().events()) {
                if (event.name == "RF read" && event.value == 0) ++bank0_samples[event.start_cycle];
                if (event.category == nanosm::TraceCategory::warp && event.detail.rfind("RF", 0) == 0) {
                    throw std::runtime_error(what + "an RF refusal: " + event.detail + log);
                }
            }
            for (uint32_t warp = 0; warp < warps; ++warp) {
                std::vector<uint64_t> issues;
                for (std::size_t index = 0; index < sources.size(); ++index) {
                    uint64_t cycle = 0;
                    uint32_t found = 0;
                    for (const auto& event : machine->trace().events()) {
                        if (is_issue_event(event) && event.id == 96 + 8 * index &&
                            event.detail.rfind("warp=" + std::to_string(warp) + ",", 0) == 0) {
                            cycle = event.start_cycle;
                            ++found;
                        }
                    }
                    require(found == 1, what + "run instruction " + std::to_string(index) + " did not issue once" + log);
                    issues.push_back(cycle);
                    run_issues.insert(cycle);
                    if (std::any_of(sources[index].begin(), sources[index].end(),
                            [&](uint32_t reg) { return reg % profile.rf_banks == 0; })) {
                        bank0_rounds.insert(cycle + 1);
                    }
                    // Its sources, sampled at its issue + 1, in their banks.
                    for (const auto reg : sources[index]) {
                        uint32_t samples = 0;
                        for (const auto& event : machine->trace().events()) {
                            if (event.name == "RF read" && event.id == reg && event.start_cycle == cycle + 1 &&
                                event.track == 100 + warp) {
                                ++samples;
                            }
                        }
                        require(samples == 1, what + "r" + std::to_string(reg) + " of run instruction " +
                            std::to_string(index) + " was not sampled at issue + 1" + log);
                    }
                }
                if (warps == 1) {
                    for (std::size_t i = 1; i < issues.size(); ++i) {
                        require(issues[i] == issues[0] + i, what + "the run did not issue back to back" + log);
                    }
                }
                for (uint32_t lane = 0; lane < profile.lanes; ++lane) {
                    const auto base = 4 * lane + 64 * warp;
                    require(machine->read_reg(warp, lane, 10) == 104 && machine->read_reg(warp, lane, 11) == 208 &&
                            machine->read_reg(warp, lane, 13) == (12u ^ 5u) &&
                            machine->read_reg(warp, lane, 14) == base + 1 && machine->read_reg(warp, lane, 17) == 24,
                        what + "arithmetic data");
                    if (warps == 1) {
                        require(machine->read_reg(warp, lane, 16) == 0x100 + 516 / 4 &&
                                machine->read_reg(warp, lane, 19) == 0x100 + (2048 + base) / 4 &&
                                machine->read_gmem_u32(4096 + base) == 100,
                            what + "memory data");
                    }
                }
            }
            // Full sampling rate: from the run's first issue to its last, an
            // instruction is accepted every cycle and bank 0 is sampled in
            // every following cycle, once -- but in the lds.uni's round
            // (uniform base only), where it is not sampled at all.
            const auto first = *run_issues.begin();
            const auto last = *run_issues.rbegin();
            require(run_issues.size() == last - first + 1, what + "the run's issues have a gap" + log);
            for (auto cycle = first + 1; cycle <= last + 1; ++cycle) {
                const auto expected = static_cast<uint32_t>(bank0_rounds.count(cycle));
                require(bank0_samples[cycle] == expected, what + "bank 0 was not sampled " +
                    std::to_string(expected) + " time(s) at c" + std::to_string(cycle) + log);
            }
            check_read_capacity(*machine, profile);
        }
    }
}

// Two DISTINCT source registers of one instruction in one RF bank are a
// malformed instruction (one read round, one read port per bank), not a
// stall: qtxc and the hand assembler reject the line, kernel load rejects
// the word -- in both builds, naming the instruction and the registers.
// Every operand form: the two- and three-source arithmetic, setp, sts's
// base and value, and stg. The kernel-load case carries words a looser
// profile (more read ports) assembled, restamped for the profile under
// test. The atomic, the mbar register forms and mma.async's two addresses
// take uniform registers, which are no RF reads: those forms cannot conflict
// (and an acc.st's uniform tmem pointer beside its vector data does not
// either).
void test_i6b1_same_bank_sources_are_malformed() {
    const auto& canonical = canonical_profile();
    struct Case {
        std::string line;
        std::string registers;
        const nanosm::DeviceProfile* profile;
    };
    const std::vector<Case> cases{
        {"add r2, r0, r3", "r0 and r3", &canonical},
        {"fadd r2, r1, r4", "r1 and r4", &canonical},
        {"fmul r2, r5, r8", "r5 and r8", &canonical},
        {"fsub r2, r0, r6", "r0 and r6", &canonical},
        {"fmax r1, r2, r8", "r2 and r8", &canonical},
        {"cvt.pack.f16x2 r2, r4, r7", "r4 and r7", &canonical},
        {"and r2, r0, r3", "r0 and r3", &canonical},
        {"or r2, r0, r3", "r0 and r3", &canonical},
        {"xor r2, r0, r3", "r0 and r3", &canonical},
        {"shl r2, r0, r3", "r0 and r3", &canonical},
        {"shr r2, r0, r3", "r0 and r3", &canonical},
        {"fma r1, r2, r5, r8", "r2, r5 and r8", &canonical},
        {"fma r1, r0, r3, r4", "r0 and r3", &canonical},
        {"setp.eq.u32 p0, r0, r3", "r0 and r3", &canonical},
        {"sts [r3 + 64], r6", "r3 and r6", &canonical},
        {"stg [r3], r6", "r3 and r6", &canonical},
    };
    const auto mentions = [](const std::string& message, const std::string& where, const Case& c) {
        return message.rfind(where, 0) == 0 && message.find(": malformed instruction '") != std::string::npos &&
            message.find("its distinct sources " + c.registers + " are in one RF bank") != std::string::npos;
    };
    for (const auto& c : cases) {
        const auto& profile = *c.profile;
        const auto source = c.line + "\nhalt\n";
        try {
            (void)nanosm::compile_qtx(source, profile);
            throw std::runtime_error("qtxc accepted '" + c.line + "'");
        } catch (const nanosm::SimError& error) {
            require(mentions(error.what(), "line 1", c), "qtxc's refusal of '" + c.line + "': " + error.what());
        }
        // The same line through a looser profile (one read port per source),
        // emitted as the profile under test's assembly.
        auto loose = profile;
        loose.rf_read_ports_per_bank = 3;
        auto program = nanosm::compile_qtx(source, loose);
        program.profile_fingerprint = nanosm::device_profile_fingerprint(profile);
        try {
            (void)nanosm::assemble_asm(nanosm::emit_asm(program), profile);
            throw std::runtime_error("the assembler accepted '" + c.line + "'");
        } catch (const nanosm::SimError& error) {
            require(mentions(error.what(), "line ", c), "the assembler's refusal of '" + c.line + "': " + error.what());
        }
        auto kernel = nanosm::assemble_asm(nanosm::emit_asm(nanosm::compile_qtx(source, loose)), loose);
        kernel.profile_fingerprint = nanosm::device_profile_fingerprint(profile);
        nanosm::Launch launch;
        launch.active_warps = 1;
        try {
            nanosm::Machine machine(profile, kernel, std::move(launch));
            throw std::runtime_error("kernel load accepted '" + c.line + "'");
        } catch (const nanosm::SimError& error) {
            require(mentions(error.what(), "kernel pc 0", c), "kernel load's refusal of '" + c.line + "': " + error.what());
        }
    }
    // Uniform operands are no RF reads: with register numbers that would
    // share an RF bank (3 and 6; ur0 beside r3), qtxc and the assembler
    // accept these forms.
    for (const std::string line : {"atom.add.uni r1, [ur3], ur6", "mbar.init [ur3], ur6",
             "mbar.arrive.expect_tx [ur3], ur6", "mbar.test_wait p0, [ur3], ur6", "mbar.try_wait p0, [ur3], ur6",
             "mma.async.m64n64k8 [ur0], [ur3], [ur6]", "acc.st [ur0, 64], r3"}) {
        try {
            const auto source = line + "\nhalt\n";
            (void)nanosm::assemble_asm(nanosm::emit_asm(nanosm::compile_qtx(source, canonical)), canonical);
        } catch (const nanosm::SimError& error) {
            throw std::runtime_error("'" + line + "' with uniform operands was refused: " + error.what());
        }
    }
    // The same register in several operand slots is one read: qtxc, the
    // assembler and kernel load accept it, and it computes.
    nanosm::Launch launch;
    launch.active_warps = 1;
    image_bytes(launch.shmem_image, 256);
    nanosm::Machine machine(canonical, compile(R"(
        sread r3, LANE_ID
        shli r3, r3, 2
        add r2, r3, r3
        mov r4, 0x3f8
        shli r4, r4, 20
        fma r5, r4, r4, r4
        setp.eq.u32 p0, r3, r3
        sts [r3 + 64], r3
        halt
    )", canonical), std::move(launch), {.timeline = true, .debug = true});
    machine.run();
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(machine.read_reg(0, lane, 2) == 8 * lane && machine.read_reg(0, lane, 5) == nanosm::float_bits(2.0f) &&
                machine.read_shmem_u32(64 + 4 * lane) == 4 * lane,
            "a repeated source register computed the wrong value");
    }
    check_read_capacity(machine, canonical);
}

// One async reader under continuous SIMT reads of its bank (RFC14 section
// 5). An acc.st's data beat (r0, bank 0) against addis that read r27 (bank
// 0) every cycle (its coordinates ur2, ur3 are uniform registers: its own
// read round takes no RF port). The acc.st, accepted at a, reaches its unit
// at a + t (issue_to_dispatch) and its beat wants bank 0 from a + t + 1 =
// a + 4.
// There the issue register owns bank 0 (the addi accepted at a + 3): the
// beat is denied, and read_owed (set by that acceptance) makes the denial an
// inhibition -- the next candidate, an add reading banks 0 AND 1, is refused
// at a + 4, all or none: nothing of it is read in a + 5, bank 1 included. At
// a + 5 the issue register is empty, the beat takes bank 0 and the add is
// accepted in the same cycle (read_owed: the acceptance sets it, the grant
// clears it, set wins; the cursor moves past the acc.st). Every accepted
// instruction samples at its issue + 1; the stream loses exactly one cycle.
void test_i6b1_async_reader_gets_the_hole() {
    const auto& profile = canonical_profile();
    auto machine = i6a_machine(profile, hand_assemble(i1_asm(profile, R"(
        umov ur2, 0
        umov ur3, 0
        mov r28, 7
        sread r0, LANE_ID
        mov r27, 5 {pred=PT, stall=8, set=-, wait=-}
        acc.st [ur2, 64], r0 {pred=PT, stall=0, set=sb0, wait=-}
        addi r1, r27, 1
        addi r2, r27, 2
        addi r5, r27, 3
        add r4, r27, r28
        addi r7, r27, 4
        acc.ld r10, [ur2, 64] {pred=PT, stall=0, set=sb1, wait=sb0}
        halt {pred=PT, stall=0, set=-, wait=sb1}
    )"), profile));
    std::map<uint64_t, bool> owed;       // Q during each cycle
    std::map<uint64_t, uint32_t> cursor;
    std::map<uint64_t, std::pair<uint32_t, uint32_t>> bank_use; // the issue register's banks 0 and 1
    (void)i1_run(*machine, {}, [&](uint64_t cycle) {
        owed[cycle + 1] = machine->rf_read_owed(0);
        cursor[cycle + 1] = machine->rf_read_cursor(0);
        bank_use[cycle + 1] = {machine->issue_read_bank_use(0), machine->issue_read_bank_use(1)};
    });
    const auto log = " | issues" + issue_log(*machine);
    const auto a = issued_at(*machine, 5 * 8).at(0);
    const std::vector<uint64_t> expected{a + 1, a + 2, a + 3, a + 5, a + 6};
    for (uint32_t i = 0; i < expected.size(); ++i) {
        require(issued_at(*machine, (6 + i) * 8) == std::vector<uint64_t>{expected[i]},
            "SIMT instruction " + std::to_string(i) + " did not issue at its cycle" + log);
    }
    require(refusals(*machine, 0, "RF read inhibited") == std::vector<std::pair<uint64_t, uint64_t>>{{a + 4, 1}},
        "the refusal:" + describe(refusals(*machine, 0, "RF read inhibited")) + log);
    const auto reads = i6b1_reads(*machine);
    require(reads.at(a + 5) == std::vector<std::pair<uint32_t, uint32_t>>{{0, 0}},
        "in the hole cycle the beat was not the only read (a refused instruction read a register?)");
    require(i6a_rf_events(*machine, "RF read", 27) ==
                std::vector<uint64_t>{a + 2, a + 3, a + 4, a + 6, a + 7} &&
            i6a_rf_events(*machine, "RF read", 28) == std::vector<uint64_t>{a + 6},
        "an accepted instruction was not sampled at its issue + 1");
    require(bank_use.at(a + 4) == std::pair<uint32_t, uint32_t>{1, 0} &&
            bank_use.at(a + 5) == std::pair<uint32_t, uint32_t>{0, 0} &&
            bank_use.at(a + 6) == std::pair<uint32_t, uint32_t>{1, 1},
        "the issue register's bank use");
    require(owed.at(a + 4) && owed.at(a + 5) && owed.at(a + 6),
        "read_owed was not held through the denial, or the grant beat the acceptance (set wins)");
    require(cursor.at(a + 5) == 0 && cursor.at(a + 6) == 1, "the cursor did not move past the acc.st at its grant");
    require(machine->trace().summary().rf_simt_read_inhibited == 1, "inhibited cycles");
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(machine->read_reg(0, lane, 4) == 12 && machine->read_reg(0, lane, 10) == lane,
            "the add or the acc.st -> acc.ld round trip");
    }
    check_read_capacity(*machine, profile);
}

// The two async readers of one bank take the holes in round-robin order:
// the acc.st engine's beat (holder 0) and stg.v8's first store-data beat (the
// LSU, holder 1), both wanting bank 0 (r0, r18) while addis read r27 (bank 0)
// every cycle they may. (I8: one acc.st engine for the machine, so there are
// two async readers, not one per tmem bank.) The acc.st (a) reaches its
// engine at a + 3 and wants its read from a + 4; the stg.v8 (a + 1) reaches
// the LSU at a + 4 and its first beat may read in that cycle. At a + 4 the
// issue register's addi samples bank 0: both are denied with service owed,
// an inhibition (refusal at a + 4); the hole at a + 5 goes to the acc.st
// (cursor 0), whose grant lifts the inhibition, so the addi is accepted at
// a + 5; at a + 6 the LSU is denied again (refusal), at a + 7 it takes the
// hole. The LSU's later beats in bank 0 (r21, r24) wait at most one cycle
// each. Data: tmem (read back) and gmem.
void test_i6b1_async_readers_round_robin() {
    const auto& profile = canonical_profile();
    std::string setup = "sread r0, LANE_ID\naddi r9, r0, 100\n";
    for (uint32_t j = 0; j < 8; ++j) setup += "addi r" + std::to_string(18 + j) + ", r0, " + std::to_string(200 + 10 * j) + "\n";
    auto machine = i6a_machine(profile, hand_assemble(i1_asm(profile, R"(
        umov ur2, 0
        umov ur3, 0
        mov r40, 0
    )" + setup + R"(
        mov r27, 5 {pred=PT, stall=8, set=-, wait=-}
        acc.st [ur2, 64], r0 {pred=PT, stall=0, set=sb0, wait=-}
        stg.v8 [r40], r18 {pred=PT, stall=0, set=sb2, wait=-}
        addi r1, r27, 1
        addi r2, r27, 2
        addi r4, r27, 3
        addi r5, r27, 4
        addi r7, r27, 5
        addi r8, r27, 6
        addi r10, r27, 7
        addi r11, r27, 8
        addi r13, r27, 9
        addi r14, r27, 10
        addi r16, r27, 11
        addi r17, r27, 12
        addi r1, r27, 1
        addi r2, r27, 2
        addi r4, r27, 3
        addi r5, r27, 4
        addi r7, r27, 5
        addi r8, r27, 6
        addi r10, r27, 7
        addi r11, r27, 8
        addi r13, r27, 9
        addi r14, r27, 10
        addi r16, r27, 11
        addi r17, r27, 12
        acc.ld r50, [ur2, 64] {pred=PT, stall=0, set=sb3, wait=sb0}
        halt {pred=PT, stall=0, set=-, wait=sb2|sb3}
    )"), profile));
    std::map<uint64_t, uint32_t> cursor;
    (void)i1_run(*machine, {}, [&](uint64_t cycle) { cursor[cycle + 1] = machine->rf_read_cursor(0); });
    const auto log = " | issues" + issue_log(*machine);
    const auto a = issued_at(*machine, 14 * 8).at(0);
    const auto refused = refusals(*machine, 0, "RF read inhibited");
    require(refused.size() >= 2 && refused[0] == std::pair<uint64_t, uint64_t>{a + 4, 1} &&
            refused[1] == std::pair<uint64_t, uint64_t>{a + 6, 1},
        "the first two refusals:" + describe(refused) + log);
    require(i6a_rf_events(*machine, "RF read", 0).back() == a + 5 && i6a_rf_events(*machine, "RF read", 18).back() == a + 7,
        "the holes did not go acc.st, LSU in turn" + log);
    require(cursor.at(a + 6) == 1 && cursor.at(a + 8) == 0,
        "the cursor did not move past each reader at its grant");
    // Every inhibited cycle's hole goes to a waiting reader the next cycle.
    const auto reads = i6b1_reads(*machine);
    for (const auto& [start, length] : refused) {
        require(length == 1, "a refusal lasted more than one cycle:" + describe(refused));
        bool hole_used = false;
        for (const auto& [reg, bank] : reads.at(start + 1)) hole_used = hole_used || (bank == 0 && reg != 27);
        require(hole_used, "the hole after the refusal at c" + std::to_string(start) + " went unused");
    }
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(machine->read_reg(0, lane, 50) == lane, "tmem data (acc.st -> acc.ld)");
        for (uint32_t j = 0; j < 8; ++j) {
            require(machine->read_gmem_u32(4 * (16 * j + lane)) == lane + 200 + 10 * j, "stg.v8 data");
        }
    }
    check_read_capacity(*machine, profile);
}

// Continuous SIMT pressure on bank 0 (60 addis reading r27) with the two
// persistent async readers (I8: one acc.st engine), whose beats come back to
// bank 0 every third register: acc.st.v8 C0 (r0..r7) and stg.v8 (r18..r25).
// No accepted read is ever stopped (each addi samples r27 at its issue + 1);
// every acc.st beat is granted within 2 x 2 - 1 = 3 cycles of wanting its
// bank (in bank 0 one hole per two cycles, the other reader at most ahead of
// it; in banks 1 and 2 only the readers compete); every inhibited cycle's
// hole goes to an async reader the next cycle; and the data (tmem read back,
// gmem) are right.
void test_i6b1_persistent_readers_under_pressure() {
    const auto& profile = canonical_profile();
    std::string setup = "sread r0, LANE_ID\n";
    for (uint32_t j = 1; j < 8; ++j) setup += "addi r" + std::to_string(j) + ", r0, " + std::to_string(10 * j) + "\n";
    for (uint32_t j = 0; j < 8; ++j) setup += "addi r" + std::to_string(9 + j) + ", r0, " + std::to_string(100 + 10 * j) + "\n";
    for (uint32_t j = 0; j < 8; ++j) setup += "addi r" + std::to_string(18 + j) + ", r0, " + std::to_string(200 + 10 * j) + "\n";
    std::string pressure;
    for (uint32_t i = 0; i < 60; ++i) pressure += "addi r41, r27, " + std::to_string(i) + "\n";
    const auto body = R"(
        umov ur2, 0
        umov ur3, 0
        mov r40, 0
    )" + setup + R"(
        mov r27, 5 {pred=PT, stall=8, set=-, wait=-}
        acc.st.v8 [ur2, 64], r0 {pred=PT, stall=0, set=sb0, wait=-}
        stg.v8 [r40], r18 {pred=PT, stall=0, set=sb2, wait=-}
    )" + pressure + R"(
        acc.ld.v8 r50, [ur2, 64] {pred=PT, stall=0, set=sb3, wait=sb0}
        halt {pred=PT, stall=0, set=-, wait=sb2|sb3}
    )";
    const auto kernel = hand_assemble(i1_asm(profile, body), profile);
    auto machine = i6a_machine(profile, kernel);
    machine->run();
    const auto log = " | issues" + issue_log(*machine);
    const uint32_t first_acc_st = 3 + 1 + 7 + 8 + 8 + 1; // index of acc.st.v8 C0 (after 28 setup instructions)
    const auto a = issued_at(*machine, first_acc_st * 8).at(0);
    const auto t = nanosm::issue_to_dispatch(profile);
    // Accepted reads are never stopped.
    std::set<uint64_t> expected_r27;
    for (uint32_t i = 0; i < 60; ++i) expected_r27.insert(issued_at(*machine, (first_acc_st + 2 + i) * 8).at(0) + 1);
    const auto r27 = i6a_rf_events(*machine, "RF read", 27);
    require(std::set<uint64_t>(r27.begin(), r27.end()) == expected_r27 && r27.size() == 60,
        "an accepted addi was not sampled at its issue + 1" + log);
    // Bounded waits of the acc.st beats: beat j wants its bank from its
    // transfer's dispatch + 1 (beat 0) or the cycle after beat j - 1's grant.
    uint32_t longest = 0;
    for (const auto& [base, issue] : std::vector<std::pair<uint32_t, uint64_t>>{{0, a}}) {
        auto ready = issue + t + 1;
        for (uint32_t j = 0; j < 8; ++j) {
            const auto grants = i6a_rf_events(*machine, "RF read", static_cast<uint8_t>(base + j));
            const auto grant = grants.back();
            require(grant >= ready, "a beat was read before it was wanted");
            longest = std::max(longest, static_cast<uint32_t>(grant - ready));
            ready = grant + 1;
        }
    }
    require(longest <= 3, "an acc.st beat waited " + std::to_string(longest) + " cycles" + log);
    require(machine->trace().summary().rf_simt_read_inhibited > 0, "the pressure never needed an inhibition");
    const auto reads = i6b1_reads(*machine);
    for (const auto& [start, length] : refusals(*machine, 0, "RF read inhibited")) {
        require(length == 1, "an inhibition lasted more than one cycle");
        bool hole_used = false;
        for (const auto& [reg, bank] : reads.at(start + 1)) hole_used = hole_used || (bank == 0 && reg != 27);
        require(hole_used, "the hole after the refusal at c" + std::to_string(start) + " went unused");
    }
    for (uint32_t lane = 0; lane < 16; ++lane) {
        for (uint32_t j = 0; j < 8; ++j) {
            require(machine->read_reg(0, lane, 50 + j) == lane + 10 * j, "tmem data (acc.st.v8 -> acc.ld.v8)");
            require(machine->read_gmem_u32(4 * (16 * j + lane)) == lane + 200 + 10 * j, "stg.v8 data");
        }
    }
    check_read_capacity(*machine, profile);
}

// Unrelated banks proceed. Three warps: warps 0 and 1 read bank 0 (r27)
// every cycle they issue, warp 2 reads only bank 1 (r28); warp 0 issues an
// acc.st of r0 (bank 0) amid its stream. Round-robin in steady state is
// w0, w1, w2: the beat wants bank 0 at a + 4, right after warp 0's addi at
// a + 3 -- denied, inhibited: warp 1's bank-0 addi is refused at a + 4, and
// warp 2's bank-1 addi is accepted in that same cycle. At a + 5 the beat
// takes the hole.
// I6b-2: warp 2 can take the cycle only if it has an instruction then. Its
// stream starts after a taken branch (an empty FIFO), and three warps
// issuing every cycle get one fetch per cycle between them: the words
// buffered stay as few as when the streams started (they grow only in
// cycles nothing issues), so warp 2's next word would arrive just in time
// for its own turn, not a cycle early. So the three warps first meet at an
// mbar barrier (arrive, try_wait) at the head of their streams: while they
// wait, fetch fills their FIFOs, and each stream starts with words in hand.
// (warp 0's bank-0 stream is its own copy: a branch to warp 1's would
// drain the acc.st's depbar first.)
void test_i6b1_unrelated_banks_proceed() {
    const auto& profile = canonical_profile();
    std::string pad;
    for (uint32_t i = 0; i < 12; ++i) pad += "addi r1, r27, 1\n";
    std::string bank0;
    for (uint32_t i = 0; i < 24; ++i) bank0 += "addi r2, r27, 2\n";
    std::string bank1;
    for (uint32_t i = 0; i < 48; ++i) bank1 += "addi r4, r28, 3\n";
    const auto meet = [](const std::string& label) {
        return "mbar.arrive [ur6]\n" + label + ": mbar.try_wait p3, [ur6], 0\n@!p3 bra.uni " + label + "\n";
    };
    nanosm::Launch launch;
    launch.active_warps = 3;
    nanosm::MbarState barrier;
    barrier.valid = true;
    barrier.expected = 3;
    barrier.remaining = 3;
    write_u64(image_bytes(launch.shmem_image, 8192), 4096, nanosm::encode_mbar_state(barrier));
    nanosm::Machine machine(profile, compile(R"(
        usread ur1, WARP_ID
        ur2r r50, ur1
        mov r27, 5
        mov r28, 7
        umov ur2, 0
        umov ur3, 0
        umov ur6, 4096
        sread r0, LANE_ID
        setp.eq.u32 p0, r50, 2
        @p0 bra.uni other
        setp.eq.u32 p1, r50, 1
        @p1 bra.uni plain
    )" + meet("meet0") + pad + R"(
        acc.st [ur2, 64], r0
    )" + bank0 + R"(
        halt
    plain:
    )" + meet("meet1") + bank0 + R"(
        halt
    other:
    )" + meet("meet2") + bank1 + R"(
        halt
    )", profile), std::move(launch), {.timeline = true, .debug = true});
    machine.run();
    const auto log = " | issues" + issue_log(machine);
    uint32_t acc_st_pc = 0;
    for (const auto& event : machine.trace().events()) {
        if (is_issue_event(event) && event.name == "acc.st") acc_st_pc = static_cast<uint32_t>(event.id);
    }
    const auto a = issued_at(machine, acc_st_pc).at(0);
    const auto warp_issued_at = [&](uint32_t warp, uint64_t cycle) {
        for (const auto& event : machine.trace().events()) {
            if (is_issue_event(event) && event.start_cycle == cycle &&
                event.detail.rfind("warp=" + std::to_string(warp) + ",", 0) == 0) {
                return true;
            }
        }
        return false;
    };
    require(warp_issued_at(0, a + 3) && warp_issued_at(1, a + 1) && warp_issued_at(2, a + 2),
        "the three warps were not in steady round-robin around the acc.st" + log);
    require(refusals(machine, 1, "RF read inhibited") == std::vector<std::pair<uint64_t, uint64_t>>{{a + 4, 1}},
        "warp 1's bank-0 addi was not refused exactly at a + 4:" + describe(refusals(machine, 1, "RF read inhibited")) + log);
    require(warp_issued_at(2, a + 4), "warp 2's bank-1 addi did not issue in the inhibited cycle" + log);
    require(refusals(machine, 2, "RF read inhibited").empty(), "warp 2 was inhibited for a bank it does not read");
    require(i6a_rf_events(machine, "RF read", 0).back() == a + 5, "the beat did not take the hole at a + 5" + log);
    for (uint32_t warp = 0; warp < 3; ++warp) {
        require(machine.read_reg(warp, 0, warp == 2 ? 4 : 2) == (warp == 2 ? 10u : 7u), "data");
    }
    check_read_capacity(machine, profile);
}

// 16 warps (HUMAN 10-17: "lets do 16 warps"). rf.rows_per_bank (352 at
// canonical, about as much RF as shmem) = 16 warps x 22 rows, so a
// 66-register kernel fits 16 warps and not 17. (I6b-2: core.warps, 16, now
// refuses a 17th warp first; with 32 warp slots declared, the RF rows still
// refuse it.) A 16-warp run in which every warp touches its own state:
// its RF rows and predicates, its uniform registers (WARP_ID through usread,
// the mbar address), WARP_ID / THREAD_ID, a depbar (lds), its shmem
// order (lds behind its own sts), the stall count, an mbar arrival and a
// try_wait on a barrier all 16 arrive at (more waiters than the 8 wait
// slots), and its trace track.
void test_i6b1_sixteen_warps() {
    const auto& profile = canonical_profile();
    auto more_slots = profile;
    more_slots.core_warps = 32;
    for (const auto* slots : std::vector<const nanosm::DeviceProfile*>{&profile, &more_slots}) {
        const auto wide = compile("mov r65, 1\nhalt", *slots);
        require(wide.register_count == 66, "the residency kernel is not 66 registers");
        for (const uint32_t warps : {16u, 17u}) {
            nanosm::Launch launch;
            launch.active_warps = warps;
            try {
                nanosm::Machine machine(*slots, wide, std::move(launch));
                require(warps == 16, "17 warps of 66 registers were resident in 352 rows");
            } catch (const nanosm::SimError& error) {
                const auto expected = slots->core_warps == 16 ? "launch of 17 warps exceeds the SM's 16 warp slots"
                                                               : "RF allocation cannot fit 17 active warps";
                require(warps == 17 && std::string(error.what()).find(expected) != std::string::npos,
                    std::string("residency: ") + error.what());
            }
        }
    }
    nanosm::Launch launch;
    launch.active_warps = 16;
    auto& shmem = image_bytes(launch.shmem_image, 8192);
    nanosm::MbarState initial;
    initial.valid = true;
    initial.expected = 16;
    initial.remaining = 16;
    write_u64(shmem, 4096, nanosm::encode_mbar_state(initial));
    nanosm::Machine machine(profile, compile(R"(
        usread ur0, WARP_ID
        ur2r r0, ur0
        sread r1, LANE_ID
        sread r10, THREAD_ID
        shli r2, r0, 6
        shli r4, r1, 2
        add r5, r2, r4
        addi r7, r0, 100
        sts [r5 + 1024], r7
        lds r8, [r5 + 1024]
        umov ur6, 4096
        mbar.arrive [ur6]
    wait:
        mbar.try_wait p0, [ur6], 0
        @!p0 bra.uni wait
        addi r65, r8, 1
        halt
    )", profile), std::move(launch), {.timeline = true});
    machine.run();
    std::set<uint32_t> tracks;
    for (const auto& event : machine.trace().events()) {
        if (event.category == nanosm::TraceCategory::warp) tracks.insert(event.track);
    }
    require(tracks.size() == 16 && *tracks.begin() == 100 && *tracks.rbegin() == 115, "16 warp tracks");
    for (uint32_t warp = 0; warp < 16; ++warp) {
        for (uint32_t lane = 0; lane < 16; ++lane) {
            require(machine.read_reg(warp, lane, 10) == 16 * warp + lane && machine.read_reg(warp, lane, 8) == warp + 100 &&
                    machine.read_reg(warp, lane, 65) == warp + 101 &&
                    machine.read_shmem_u32(1024 + 64 * warp + 4 * lane) == warp + 100,
                "warp " + std::to_string(warp) + " lane " + std::to_string(lane) + " data");
        }
        require(machine.read_ureg(warp, 0) == warp && machine.read_ureg(warp, 6) == 4096,
            "warp " + std::to_string(warp) + " uniform registers");
        for (uint8_t bar = 0; bar < 6; ++bar) require(machine.depbar_value(warp, bar) == 0, "a depbar left set");
    }
    const auto state = machine.read_mbar_state(4096);
    require(state.phase && state.remaining == 16, "the barrier did not complete one phase with 16 arrivals");
    const auto& summary = machine.trace().summary();
    uint64_t warp_cycles = 0;
    for (const auto cycles : summary.warp_cycles) warp_cycles += cycles;
    require(warp_cycles == summary.cycles * 16, "warp attribution does not cover 16 warps");
    require(summary.warp_cycles[static_cast<std::size_t>(nanosm::WarpState::mbar_wait_slot)] != 0,
        "16 waiters never found the 8 wait slots full");
}

} // namespace

// RF one-round FADD and HALT
TEST(Rf, rf_one_round_fadd_and_halt) {
    test_rf_one_round_fadd_and_halt();
}

// RF one-round WAR and branch
TEST(Rf, rf_one_round_war_and_branch) {
    test_rf_one_round_war_and_branch();
}

// RF seeded competing warps
TEST(RfSlow, rf_seeded_competing_warps) {
    test_rf_seeded_competing_warps();
}

// I1: lds issue ignores its own rounds (C8 probe A)
TEST(Rf, i1_lds_issue_ignores_its_rounds) {
    test_i1_lds_issue_ignores_its_rounds();
}

// sanitizer: issue capture before acceptance
TEST(Rf, sanitizer_issue_capture_before_acceptance) {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    test_sanitizer_issue_capture_before_acceptance();
}

// I1: refused attempts read nothing
TEST(Rf, i1_refused_attempts_read_nothing) {
    test_i1_refused_attempts_read_nothing();
}

// I1: RF owed inhibits after a denial
TEST(Rf, i1_rf_owed_inhibits_after_denial) {
    test_i1_rf_owed_inhibits_after_denial();
}

// I1: RF owed permits one more booking
TEST(Rf, i1_rf_owed_one_more_booking) {
    test_i1_rf_owed_one_more_booking();
}

// I1: RF inhibition is bank- and booking-local
TEST(Rf, i1_rf_inhibition_is_bank_and_booking_local) {
    test_i1_rf_inhibition_is_bank_and_booking_local();
}

// I1: RF round-robin among ready holders
TEST(Rf, i1_rf_round_robin_among_holders) {
    test_i1_rf_round_robin_among_holders();
}

// I1: round_robin_grant
TEST(Rf, i1_round_robin_grant) {
    test_i1_round_robin_grant();
}

// I1: acc.ld pipeline freeze
TEST(Rf, i1_acc_ld_pipeline_freeze) {
    test_i1_acc_ld_pipeline_freeze();
}

// I1: load slots
TEST(Rf, i1_load_slots) {
    test_i1_load_slots();
}

// I1: load slots profile field
TEST(Rf, i1_load_slots_profile_field) {
    test_i1_load_slots_profile_field();
}

// I1.1: one LSU RF write per cycle
TEST(Rf, i1_lsu_one_rf_write_per_cycle) {
    test_i1_lsu_one_rf_write_per_cycle();
}

// I6a: the RF timing helper is the trace's (Lr/Wd 1/1, 2/2, 3/1)
TEST(Rf, i6a_helper_contract) {
    test_i6a_helper_contract();
}

// I6a: dependent add at the qtxc distance; one less is a RAW
TEST(Rf, i6a_dependent_add_distance) {
    test_i6a_dependent_add_distance();
}

// I6a: WAW order
TEST(Rf, i6a_waw_order) {
    test_i6a_waw_order();
}

// I6a: async writer vs a fixed booking
TEST(Rf, i6a_async_writer_vs_fixed_booking) {
    test_i6a_async_writer_vs_fixed_booking();
}

// I6a: slots claimed while operands travel
TEST(Rf, i6a_slots_claimed_while_operands_travel) {
    test_i6a_slots_claimed_while_operands_travel();
}

// I6a: profile fields Lr, Wd; the stall-field bound
TEST(Rf, i6a_profile_fields) {
    test_i6a_profile_fields();
}

// I6b-1: back-to-back same-bank reads (Lr 1, 2, 3)
TEST(Rf, i6b1_back_to_back_same_bank_reads) {
    test_i6b1_back_to_back_same_bank_reads();
}

// I6b-1: same-bank distinct sources are malformed
TEST(Rf, i6b1_same_bank_sources_are_malformed) {
    test_i6b1_same_bank_sources_are_malformed();
}

// I6b-1: an async reader gets the hole
TEST(Rf, i6b1_async_reader_gets_the_hole) {
    test_i6b1_async_reader_gets_the_hole();
}

// I6b-1: async readers round-robin
TEST(Rf, i6b1_async_readers_round_robin) {
    test_i6b1_async_readers_round_robin();
}

// I6b-1: persistent readers under SIMT pressure
TEST(Rf, i6b1_persistent_readers_under_pressure) {
    test_i6b1_persistent_readers_under_pressure();
}

// I6b-1: unrelated banks proceed
TEST(Rf, i6b1_unrelated_banks_proceed) {
    test_i6b1_unrelated_banks_proceed();
}

// I6b-1: 16 warps
TEST(Rf, i6b1_sixteen_warps) {
    test_i6b1_sixteen_warps();
}

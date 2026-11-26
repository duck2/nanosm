// The register file: one-round reads, the RF write decision, read / write depth, the read side and 16 warps.
//
// Port A's edge convention: a read SAMPLED at S has its operand at S + Lr; a
// write PRESENTED at the RF write port at W is at the RAM at W + Wd, lands at
// that edge and is visible to samples from W + Wd + 1 -- no forwarding. The
// debug trace shows the samples ("RF read") and the presentations ("RF
// write"); the landing is the cycle whose tick changed the register.
#include "test_support.hpp"

namespace {

using namespace nanosm_test;

using Slices = std::vector<std::pair<uint64_t, uint64_t>>;

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

// The RF write-decision scenarios keep their cycles at any shmem depths:
// every cycle after an lds.uni moves by X = lds_shift (D + Lb + its issue to
// grant - 1), so the lds.uni's stall is 1 + X, and a fadd booking placed
// against its result needs latency base + D + Lb + its issue to grant -
// issue_to_dispatch - 1 (the fadd's booking already moves by
// issue_to_dispatch). The address register ur1 is zero from launch. The
// first instruction issues at fetch_to_issue.
uint8_t shifted_fadd_latency(const nanosm::DeviceProfile& profile, uint32_t base) {
    return static_cast<uint8_t>(base + profile.shmem_port_a_depth + profile.shmem_read_latency_cycles +
        issue_to_shmem_grant(profile) - nanosm::issue_to_dispatch(profile) - 1);
}

// The RF depths the timing tests run at. r2ur reads its source through the
// vector RF, so the uniform writers' fixed path to the URF write port
// (urf.issue_to_port_cycles) must cover the issue register + Lr.
nanosm::DeviceProfile rf_depth_profile(uint32_t lr, uint32_t wd) {
    auto profile = canonical_profile();
    profile.rf_read_latency_cycles = lr;
    profile.rf_write_depth_cycles = wd;
    profile.urf_issue_to_port_cycles = std::max(profile.urf_issue_to_port_cycles, 1 + lr);
    nanosm::validate_device_profile(profile);
    return profile;
}

const std::vector<std::pair<uint32_t, uint32_t>> rf_depths = {{1, 1}, {2, 2}, {3, 1}};

std::string depth_what(const nanosm::DeviceProfile& profile) {
    return "Lr/Wd " + std::to_string(profile.rf_read_latency_cycles) + "/" +
        std::to_string(profile.rf_write_depth_cycles) + ": ";
}

// Every debug "RF read" sample of warp 0, by cycle: (register, bank).
std::map<uint64_t, std::vector<std::pair<uint32_t, uint32_t>>> warp0_reads(const nanosm::Machine& machine) {
    std::map<uint64_t, std::vector<std::pair<uint32_t, uint32_t>>> reads;
    for (const auto& event : machine.trace().events()) {
        if (event.name == "RF read" && event.track == 100) {
            reads[event.start_cycle].emplace_back(static_cast<uint32_t>(event.id), static_cast<uint32_t>(event.value));
        }
    }
    return reads;
}

// The last cycle warp 0 sampled `reg`.
uint64_t last_rf_read(const nanosm::Machine& machine, uint8_t reg) {
    const auto reads = rf_events(machine, "RF read", reg);
    require(!reads.empty(), "r" + std::to_string(reg) + " was never sampled");
    return reads.back();
}

// Every inhibited cycle (a refusal of one cycle) leaves a hole that an async
// reader takes the next cycle: a bank-0 read of anything but the SIMT
// stream's r27.
void require_holes_used(const nanosm::Machine& machine, const Slices& refused) {
    const auto reads = warp0_reads(machine);
    for (const auto& [start, length] : refused) {
        require(length == 1, "an inhibition lasted more than one cycle:" + describe(refused));
        bool hole_used = false;
        if (const auto at = reads.find(start + 1); at != reads.end()) {
            for (const auto& [reg, bank] : at->second) hole_used = hole_used || (bank == 0 && reg != 27);
        }
        require(hole_used, "the hole after the refusal at c" + std::to_string(start) + " went unused");
    }
}

} // namespace

// One round ends a reader's samples at issue + issue_to_sample, before the
// earliest any later write of the warp can land (earliest_rf_visible: 7 at
// canonical). So qtxc buys no WAR stall, neither for a fixed writer nor for
// an async one, and the reader captures the old value. A counted loop in four
// warps still drains its fixed reads and predicates at every branch.
TEST(Rf, rf_one_round_war_and_branch) {
    const auto& profile = canonical_profile();
    require(nanosm::issue_to_sample + 1 < nanosm::earliest_rf_visible(profile), "a younger write could land first");
    const auto fixed_writer = qtx_program("add r2, r0, r1\nmov r1, 7\nhalt", profile);
    require(fixed_writer.instructions[0].control.stall == 0, "qtxc bought a WAR stall a fixed writer cannot need");
    const auto source = "add r2, r0, r1\nlds r1, [r5]\nhalt";
    const auto async_writer = qtx_program(source, profile);
    require(async_writer.instructions[0].control.stall == 0, "qtxc bought a WAR stall an async writer cannot need");
    auto launch = launch_warps(1);
    write_u32(image_bytes(launch.test_shmem, 64), 0, 7);
    nanosm::Machine war(profile, compile(source, profile), std::move(launch));
    war.run();
    require(war.read_reg(0, 0, 2) == 0 && war.read_reg(0, 0, 1) == 7,
        "the reader did not capture the old r1, or the lds did not land");

    nanosm::Machine loop(profile, compile(R"(
        mov r0, 0
        mov r4, 9
    loop:
        addi r0, r0, 1
        setp.lt.u32 p0, r0, r4
        @p0 bra.uni loop
        halt
    )", profile), launch_warps(4), {.timeline = true, .debug = true});
    loop.run();
    for (uint32_t warp = 0; warp < 4; ++warp)
        require(loop.read_reg(warp, 0, 0) == 9, "branch did not drain fixed read/predicate state");
    check_read_capacity(loop, profile);
}

// Seeded random adds and xoris over twelve registers in sixteen warps: every
// add's sources are one register twice or two registers in different banks
// (one round, one read port per bank). Every warp ends with the sequential
// result, and no cycle samples a bank more than its ports allow.
TEST(RfSlow, rf_seeded_competing_warps) {
    const auto& profile = canonical_profile();
    const uint32_t warps = 16;
    std::string source;
    std::array<uint32_t, 12> expected{};
    for (uint32_t r = 0; r < expected.size(); ++r) {
        expected[r] = r + 1;
        source += "mov r" + std::to_string(r) + ", " + std::to_string(r + 1) + "\n";
    }
    uint32_t seed = profile.rf_banks * 100 + warps;
    const auto random = [&]() { seed = seed * 1664525u + 1013904223u; return seed; };
    for (uint32_t i = 0; i < 400; ++i) {
        const auto dst = random() % expected.size();
        const auto a = random() % expected.size();
        std::vector<uint32_t> other_banks;
        for (uint32_t r = 0; r < expected.size(); ++r) {
            if (r % profile.rf_banks != a % profile.rf_banks) other_banks.push_back(r);
        }
        const auto b = i % 3 == 0 ? a : other_banks[random() % other_banks.size()];
        if (i % 5 == 0) {
            const auto immediate = random() % 1024;
            source += "xori r" + std::to_string(dst) + ", r" + std::to_string(a) + ", " + std::to_string(immediate) + "\n";
            expected[dst] = expected[a] ^ immediate;
        } else {
            source += "add r" + std::to_string(dst) + ", r" + std::to_string(a) + ", r" + std::to_string(b) + "\n";
            expected[dst] = expected[a] + expected[b];
        }
    }
    source += "halt\n";
    nanosm::Machine machine(profile, compile(source, profile), launch_warps(warps), {.timeline = true, .debug = true});
    machine.run(50000);
    for (uint32_t warp = 0; warp < warps; ++warp)
        for (uint32_t lane = 0; lane < profile.lanes; ++lane)
            for (uint32_t r = 0; r < expected.size(); ++r)
                require(machine.read_reg(warp, lane, r) == expected[r], "seeded RF contention changed the sequential result");
    check_read_capacity(machine, profile);
}

// An lds's issue does not depend on its own addresses. Shmem word 0 is the
// lane mask the plain lds's addresses are built from: mask 15 is one round
// (sixteen lanes on sixteen banks), mask 7 two (lanes l and l + 8 share a
// bank). Both issue in the same cycle; the rounds move only the completion.
// One round puts the result at the RF port at issue + one_round_port; the
// fadd's latency books bank 0's port on the two-round result's port cycle,
// so that result is denied and written one cycle later. Results land Wd
// after their port.
TEST(Rf, lds_issue_ignores_its_rounds) {
    auto profile = canonical_profile();
    const auto one_round_port =
        issue_to_shmem_grant(profile, true) + profile.shmem_port_a_depth + profile.shmem_read_latency_cycles + 1;
    const auto fadd_before_lds = alu_distance(profile) - 1; // shli, fadd, (stall), lds
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
        auto launch = launch_warps(1);
        auto& shmem = image_bytes(launch.test_shmem, 256);
        write_u32(shmem, 0, mask);
        for (uint32_t word = 1; word < 64; ++word) write_u32(shmem, 4 * word, 0x100 + word);
        nanosm::Machine machine(profile, hand_assemble(hand_asm(profile, body), profile), std::move(launch),
            nanosm::TraceOptions{.timeline = true});
        const auto written = watch_registers(machine, {6});
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
        "the lds's issue cycle depends on its own addresses: " + std::to_string(one_round.issue) + " vs " +
            std::to_string(two_rounds.issue));
    const auto wd = profile.rf_write_depth_cycles;
    require(one_round.written == one_round.issue + one_round_port + wd,
        "one round: the result did not land at issue + one_round_port + Wd");
    require(two_rounds.written == two_rounds.issue + one_round_port + 2 + wd,
        "two rounds: the result (ready when the fadd's booking holds bank 0) was not written one cycle later");
}

// The RF write decision against fixed bookings. A ready async result denied
// by a booking of its bank leaves service owed, and while it is owed the
// bank is inhibited: fixed bookings to it are refused until the result is
// written (the booking made in the grant cycle sets owed again). With
// nothing owed, a denied result lets one more booking through; a booking
// refused for its port changes nothing. Inhibition refuses only fixed
// bookings to the inhibited bank (a destination of an lds.uni is not a
// booking), and with no async writer ready nothing is inhibited. Cycles are
// relative to the first issue s; X = lds_shift.
TEST(Rf, rf_owed_inhibits_after_denial) {
    auto profile = canonical_profile();
    const auto x = lds_shift(profile);
    const auto wd = profile.rf_write_depth_cycles;
    const auto s = nanosm::fetch_to_issue;
    const auto stall = std::to_string(1 + x);
    using Counter = std::tuple<const char*, uint64_t nanosm::TraceSummary::*, uint64_t>;
    struct Case {
        std::string what;
        uint32_t fadd = 0;
        std::string body;
        std::vector<std::pair<uint32_t, std::vector<uint64_t>>> issues; // instruction index -> its issue cycles
        std::vector<std::pair<uint32_t, std::vector<uint64_t>>> writes; // register -> the cycles it landed
        std::optional<Slices> inhibited;                                // "RF write inhibited"
        std::optional<Slices> reserved;                                 // "RF write reservation"
        std::vector<Counter> counters;
    };
    const std::vector<Case> cases{
        {"owed inhibits after a denial", shifted_fadd_latency(profile, 4), R"(
            mov r61, 0
            fadd r6, r4, r4
            fadd r9, r4, r4
            lds.uni r3, [ur1] {pred=PT, stall=)" + stall + R"(, set=sb0, wait=-}
            fadd r12, r4, r4
            fadd r15, r4, r4
            halt {pred=PT, stall=0, set=-, wait=sb0}
        )",
            {{3, {s + 3}}, {5, {s + 7 + x}}}, {{3, {s + 7 + x + wd}}}, Slices{{s + 5 + x, 2}}, std::nullopt,
            {{"async waits", &nanosm::TraceSummary::rf_async_waits, 2},
                {"SIMT inhibited", &nanosm::TraceSummary::rf_simt_inhibited, 2},
                {"async writes", &nanosm::TraceSummary::rf_async_writes, 1}}},
        {"nothing owed lets one more booking through", shifted_fadd_latency(profile, 5), R"(
            mov r61, 0
            fadd r6, r4, r4
            fadd r9, r4, r4
            lds.uni r3, [ur1] {pred=PT, stall=0, set=sb0, wait=-}
            lds.uni r12, [ur1] {pred=PT, stall=)" + stall + R"(, set=sb1, wait=-}
            mov r2, 1
            fadd r15, r4, r4
            fadd r18, r4, r4
            halt {pred=PT, stall=0, set=-, wait=sb0|sb1}
        )",
            {{6, {s + 6 + x}}}, {{3, {s + 5 + x + wd}}, {12, {s + 8 + x + wd}}}, Slices{{s + 7 + x, 1}},
            std::nullopt, {}},
        // The add's booking is only issue_to_dispatch + alu ahead of its
        // issue: latency 5 + X puts the fadd's port on the add's.
        {"a booking refused for its port", static_cast<uint32_t>(5 + x), R"(
            mov r61, 0
            fadd r6, r4, r4
            lds.uni r3, [ur1] {pred=PT, stall=)" + stall + R"(, set=sb0, wait=-}
            mov r2, 1
            mov r5, 1
            add r9, r4, r4
            halt {pred=PT, stall=0, set=-, wait=sb0}
        )",
            {}, {}, std::nullopt, Slices{{s + 5 + x, 1}}, {}},
        // The second lds.uni's result meets fadd r12's booking: X later too.
        {"inhibition is bank- and booking-local", shifted_fadd_latency(profile, 4), R"(
            mov r61, 0
            fadd r6, r4, r4
            fadd r9, r4, r4
            lds.uni r3, [ur1] {pred=PT, stall=)" + stall + R"(, set=sb0, wait=-}
            fadd r12, r4, r4
            mov r13, 1
            lds.uni r21, [ur1] {pred=PT, stall=0, set=sb1, wait=-}
            fadd r15, r4, r4
            halt {pred=PT, stall=0, set=-, wait=sb0|sb1}
        )",
            {{5, {s + 5 + x}}, {6, {s + 6 + x}}}, {{3, {s + 7 + x + wd}}, {21, {s + 9 + 2 * x + wd}}},
            std::nullopt, std::nullopt, {{"SIMT inhibited", &nanosm::TraceSummary::rf_simt_inhibited, 0}}},
        {"no async writer ready", profile.timing.fadd, R"(
            mov r1, 0
            fadd r6, r4, r4
            fadd r9, r4, r4
            fadd r12, r4, r4
            fadd r15, r4, r4
            fadd r18, r4, r4
            halt
        )",
            {{1, {s + 1}}, {2, {s + 2}}, {3, {s + 3}}, {4, {s + 4}}, {5, {s + 5}}}, {}, std::nullopt, std::nullopt,
            {{"SIMT inhibited", &nanosm::TraceSummary::rf_simt_inhibited, 0}}},
    };
    for (const auto& c : cases) {
        auto config = profile;
        config.timing.fadd = static_cast<uint8_t>(c.fadd);
        auto machine = debug_machine(config, c.body);
        std::vector<uint32_t> regs;
        for (const auto& [reg, cycles] : c.writes) regs.push_back(reg);
        const auto written = watch_registers(*machine, regs);
        const auto where = c.what + ": ";
        const auto log = " | issues" + issue_log(*machine);
        for (const auto& [index, cycles] : c.issues) {
            require(issued_at(*machine, index * 8) == cycles,
                where + "instruction " + std::to_string(index) + " did not issue at its cycle" + log);
        }
        for (const auto& [reg, cycles] : c.writes) {
            const auto landed = written.find(reg);
            require(landed != written.end() && landed->second == cycles,
                where + "r" + std::to_string(reg) + " did not land at its cycle" + log);
        }
        if (c.inhibited) {
            require(refusals(*machine, 0, "RF write inhibited") == *c.inhibited,
                where + "inhibited" + describe(refusals(*machine, 0, "RF write inhibited")) + log);
        }
        if (c.reserved) {
            require(refusals(*machine, 0, "RF write reservation") == *c.reserved,
                where + "refused for the port" + describe(refusals(*machine, 0, "RF write reservation")) + log);
        }
        for (const auto& [name, member, value] : c.counters) {
            require(machine->trace().summary().*member == value, where + "the counter of " + name);
        }
    }
}

// The round-robin choice itself (one bank): scan from the cursor, skip holders
// not ready, grant up to the free ports, move past the last one granted.
TEST(Rf, round_robin_grant) {
    using RoundRobin = nanosm::RoundRobinGrant;
    const auto grant = [](std::vector<uint8_t> ready, uint32_t cursor, uint32_t ports) {
        return nanosm::round_robin_grant(ready, cursor, ports);
    };
    const auto same = [](const RoundRobin& got, std::vector<uint32_t> granted, uint32_t cursor) {
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

// The acc.ld transfer pipeline freezes and resumes. Beat 0 (r16, bank 1) is
// first at the pipeline's output at P = A + issue_to_dispatch + 1 + Dt + Lt
// (A the acc.ld's issue). Six fadds book bank 1 for P .. P + 5: the whole
// pipeline holds -- no tmem read launches or samples, the depbar stays set,
// done() is false -- while the add r46 that wants bank 1 is held, first by
// its own booked port, then (P + 2 .. P + 5) by the inhibition. Beats 0-5
// are written at P + 6 .. P + 11 with no refill bubble; beat 6 (bank 1)
// meets r43's booking at P + 12 and goes at P + 13, beat 7 at P + 14; the
// depbar clears as the last beat lands. A calibration run (fadd latency 1)
// measures A and the first fadd's issue, from which the fadd latency that
// puts the bookings on P follows.
TEST(Rf, acc_ld_pipeline_freeze) {
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
        auto calibration = debug_machine(profile, body);
        calibration->run();
        acc_ld_issue = issued_at(*calibration, 19 * 8).at(0);
        first_fadd = issued_at(*calibration, 13 * 8).at(0);
    }
    const auto p = acc_ld_issue + t + 1 + dt + lt;
    profile.timing.fadd = static_cast<uint8_t>(p - first_fadd - nanosm::issue_register_stages -
        profile.rf_read_latency_cycles);
    auto machine = debug_machine(profile, body);
    struct Seen {
        uint64_t tmem_reads = 0;
        uint32_t sb1 = 0;
        bool done = false;
    };
    std::map<uint64_t, Seen> seen; // after each cycle's tick
    std::vector<uint32_t> regs;
    for (uint32_t reg = 16; reg < 24; ++reg) regs.push_back(reg);
    const auto written = watch_registers(*machine, regs, [&](uint64_t cycle) {
        seen[cycle] = {machine->trace().summary().tmem_load_beats, machine->depbar_value(0, 1), machine->done()};
    });
    const auto log = " | issues" + issue_log(*machine);
    require(issued_at(*machine, 19 * 8) == std::vector<uint64_t>{acc_ld_issue},
        "acc.ld issue moved with the fadd latency; the calibration does not hold");
    // Beats 0 .. Lt - 1 sampled tmem before P; beat Lt samples at P itself.
    require(seen.at(p - 1).tmem_reads == lt, "tmem samples before the hold");
    for (uint64_t cycle = p - 1; cycle <= p + 5; ++cycle) {
        const auto& at = seen.at(cycle);
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
        require(written.at(reg) == std::vector<uint64_t>{port[beat] + wd},
            "beat " + std::to_string(beat) + " was not written exactly once at P + " + std::to_string(port[beat] - p) +
                " (landing Wd later)" + log);
        for (uint32_t lane = 0; lane < 16; ++lane) {
            require(machine->read_reg(0, lane, reg) == lane + 1 + 100 * beat, "beat " + std::to_string(beat) + " data");
        }
    }
    require(seen.at(port[7] + wd - 1).sb1 == 1 && seen.at(port[7] + wd).sb1 == 0,
        "the depbar did not clear exactly as the last beat's write landed");
    require(refusals(*machine, 0, "RF write inhibited") == Slices{{p + 2, 4}},
        "the add was not inhibited while beat 0 was denied with service owed" + log);
    require(machine->trace().summary().tmem_load_beats == 8 && machine->trace().summary().rf_async_writes == 8,
        "beats launched or written more than once");
}

// The write-back buffer (shmem.writeback_entries = 2). Two lds.uni hold both
// entries, their words back and waiting for the RF (eight fadds keep bank 0
// booked over them); a third lds.uni waits in the read side's serve stage --
// it is not an A-read requester -- until the cycle after the RF write
// decision grants entry 0 (an entry is free in Q from then). Meanwhile port
// A serves a store. Relative to the first lds.uni at L0 (canonical): its
// round is granted at L0 + 3 and its word written at the RF port at
// L0 + 8; the second's one cycle later. Entry 0's write is granted at G0,
// which the fadds hold off; the third load's round is granted at G0 + 1.
// The sts issues after the second load's round is granted (its warp's
// unserviced read) and reaches the RAM while both results wait; the third
// lds.uni issues after the sts's round is granted.
TEST(Rf, writeback_buffer) {
    auto profile = canonical_profile();
    const auto wd = profile.rf_write_depth_cycles;
    profile.timing.fadd = shifted_fadd_latency(profile, 9);
    auto machine = debug_machine(profile, R"(
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
    std::optional<uint64_t> store_cycle; // the cycle whose edge put the store's lane-0 word in the RAM
    const auto written = watch_registers(*machine, {3, 6, 9}, [&](uint64_t cycle) {
        if (!store_cycle.has_value() && machine->read_shmem_u32(1024) == 7) store_cycle = cycle;
    });
    const auto log = " | issues" + issue_log(*machine);
    const auto reads = grant_cycles(*machine, "A-read", "simt");
    require(reads.size() == 3, "three A-read rounds" + log);
    const auto g0 = written.at(3).at(0) - wd; // entry 0's RF write grant
    require(written.at(6).at(0) == g0 + 1 + wd, "the second load's write did not follow the first's" + log);
    require(reads[2] == g0 + 1, "the third load's round was not granted the cycle after entry 0's RF write grant" + log);
    require(store_cycle.has_value() && *store_cycle < g0, "port A did not serve the sts while both entries held words" + log);
    require(machine->read_reg(0, 7, 9) == 0x102, "the third load's word");
    for (uint32_t lane = 0; lane < 16; ++lane) require(machine->read_shmem_u32(1024 + 4 * lane) == 7, "sts data");
}


// The helper contract: for an add, an fadd, an SFU op, an lds, an sts and
// an acc.ld, the cycles the trace shows -- the samples, the port, the RAM
// (the register's landing edge), the visibility (a depbar waiter's issue) --
// are src/timing.hpp's, relative to each instruction's accepted issue i, at
// Lr/Wd 1/1, 2/2 and 3/1 (t = issue_to_dispatch; g = issue_to_shmem_grant
// of an lds / sts, 3 + Lr):
//   fixed:  samples i + issue_to_sample, port i + issue_to_port, RAM port +
//           Wd;
//   lds:    in its queue at i + 1, its collect stage at i + 2, which samples
//           its address; A-read grant i + g; the words at port A's output
//           D + Lb later; the RF port the cycle after; RAM + Wd; its depbar
//           clears at the visibility, when the waiting add issues (the
//           counter is 1 through the RAM cycle, 0 from RAM + 1);
//   sts:    address and data sampled by its collect stage at i + 2; A-write
//           grant i + g, the word at the shmem RAM D later;
//   acc.ld: its tmem pointer (a uniform register) sampled from the URF at i + 1,
//           no RF read; tmem read (beat 0) at i + t + 1 + Dt; RF port Lt
//           later; RAM + Wd; the waiting add at RAM + 1.
// An async write's visibility is at least earliest_rf_visible after its
// issue. The shli's stall lets every fixed result reach the RAM before the
// async section, and the acc.ld goes before the lds (its waiter first), so
// no write port or issue slot is contested: the uncontended pipeline.
// Then a predicate-only read: setp p0, r0, r0 reads r0 once, at its issue +
// issue_to_sample, and the halt behind it waits for the predicate to reach
// the file (its port cycle, issue + 1 + Lr + alu).
TEST(Rf, helper_contract) {
    for (const auto& [lr, wd] : rf_depths) {
        const auto profile = rf_depth_profile(lr, wd);
        const auto what = depth_what(profile);
        const auto kernel = hand_assemble(hand_asm(profile, R"(
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
        auto machine = debug_machine(profile, kernel);
        std::optional<uint64_t> stored; // the cycle whose edge put the sts's lane-0 word in the RAM
        std::map<uint64_t, uint32_t> sb0;  // depbar sb0 (Q) during each cycle
        const auto landed = watch_registers(*machine, {3, 4, 5, 7, 11}, [&](uint64_t cycle) {
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
            const auto samples = rf_events(*machine, "RF read", reg);
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
            require(rf_events(*machine, "RF write", dst) == std::vector<uint64_t>{port},
                name + "not presented at the RF write port at issue + issue_to_port" + log);
            require(landed.at(dst) == std::vector<uint64_t>{port + wd}, name + "did not land at port + Wd");
        }

        // lds r7, [r6 + 256]: one round; its depbar waiter is the add r12.
        {
            const auto i = issue(12);
            const auto name = what + "lds: ";
            require(sampled_once(6, i + 2), name + "the address was not sampled by the collect stage at i + 2" + log);
            const auto reads = grants_of(*machine, "A-read");
            require(reads.size() == 1 && reads[0].cycle == i + issue_to_shmem_grant(profile, true),
                name + "A-read grant:" + describe(reads));
            const auto port = reads[0].cycle + read_return(profile) + 1;
            require(rf_events(*machine, "RF write", 7) == std::vector<uint64_t>{port},
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
            require(sampled_once(4, i + 2) && sampled_once(6, i + 2),
                name + "the address and data were not sampled together by the collect stage at i + 2" + log);
            const auto writes = grants_of(*machine, "A-write");
            require(writes.size() == 1 && writes[0].cycle == i + issue_to_shmem_grant(profile, true),
                name + "A-write grant:" + describe(writes));
            require(stored == writes[0].cycle + profile.shmem_port_a_depth, name + "the word did not reach the RAM D after the grant");
        }

        // acc.ld r11, [ur8, 64]: one beat; its depbar waiter is the add r14.
        {
            const auto i = issue(11);
            const auto name = what + "acc.ld: ";
            // Its coordinates are uniform registers: its read round samples
            // no RF bank (nothing else reads in that cycle).
            for (const auto& event : machine->trace().events()) {
                if (event.name == "RF read" && event.start_cycle == i + nanosm::issue_register_stages) {
                    require(false, name + "the read round sampled the RF (r" + std::to_string(event.id) + ")" + log);
                }
            }
            std::vector<uint64_t> beats;
            for (const auto& event : events_named(*machine, "acc.ld beat")) beats.push_back(event.start_cycle);
            const auto launch = i + t + 1;
            require(beats == std::vector<uint64_t>{launch + profile.tmem_port_depth},
                name + "the tmem sample was not at the launch (dispatch + 1) + tmem port depth");
            const auto port = launch + profile.tmem_port_depth + profile.tmem_read_latency_cycles;
            const auto presented = rf_events(*machine, "RF write", 11); // the mov's 77, then the beat
            require(presented.size() == 2 && presented.back() == port,
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

    const auto& profile = canonical_profile();
    nanosm::Machine predicate(profile, compile("setp.eq.u32 p0, r0, r0\nhalt", profile), launch_warps(1),
        {.timeline = true, .debug = true});
    predicate.run();
    require(predicate.cycle() == nanosm::fetch_to_issue + nanosm::issue_to_sample + profile.rf_read_latency_cycles +
                profile.timing.alu + 1,
        "HALT dropped a pending predicate-only read pipeline");
    const auto reads = events_named(predicate, "RF read");
    require(reads.size() == 1 && reads[0].start_cycle == nanosm::fetch_to_issue + nanosm::issue_to_sample,
        "the repeated operand was not one RF read at issue + issue_to_sample");
}

// The qtxc distance for a dependent add -> add is issue_to_visible -
// issue_register_stages (the consumer samples one cycle after its issue);
// hand assembly with exactly that stall runs and reads the new value. One
// cycle less puts the consumer's sample at the producer's RAM cycle W + Wd:
// the sanitizer PANICs (RF RAW); without it the machine is quiet and the
// sample sees the OLD word -- the write lands at that edge, after every read
// of the cycle (READ_FIRST, no forwarding).
TEST(Rf, dependent_add_distance) {
    for (const auto& [lr, wd] : rf_depths) {
        const auto profile = rf_depth_profile(lr, wd);
        const auto what = depth_what(profile);
        const auto scheduled = qtx_program("add r1, r0, r0\nadd r2, r1, r1\nhalt\n", profile);
        const auto distance = nanosm::issue_to_visible(scheduled.instructions[0], profile) - nanosm::issue_register_stages;
        require(scheduled.instructions[0].control.stall == distance, what + "qtxc's stall is not the helper's distance");
        const auto run = [&](uint64_t stall) {
            const auto text = hand_asm(profile, "sread r0, LANE_ID {pred=PT, stall=" +
                std::to_string(alu_distance(profile)) + ", set=-, wait=-}\nadd r1, r0, r0 {pred=PT, stall=" +
                std::to_string(stall) + ", set=-, wait=-}\nadd r2, r1, r1\nhalt\n");
            return debug_machine(profile, nanosm::assemble_asm(text, profile));
        };
        auto exact = run(distance);
        exact->run();
        require(issued_at(*exact, 16).at(0) - issued_at(*exact, 8).at(0) == distance, what + "the consumer moved");
        require(exact->read_reg(0, 5, 2) == 20, what + "at the qtxc distance the add did not read the new r1");
        auto early = run(distance - 1);
#if NANOSM_SANITIZER
        expect_panic(*early, {"RF RAW hazard: missing stall or depbar wait for r1", "inst='add r2, r1, r1"},
            what + "one cycle early");
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
TEST(Rf, waw_order) {
    for (const auto& [lr, wd] : rf_depths) {
        const auto profile = rf_depth_profile(lr, wd);
        const auto what = depth_what(profile);
        const auto scheduled = qtx_program("fadd r3, r1, r1\nadd r3, r2, r2\nhalt\n", profile);
        const uint64_t stall = profile.timing.fadd - profile.timing.alu + 1;
        require(scheduled.instructions[0].control.stall == stall, what + "qtxc's WAW stall");
        // r1 = 1.0f (0x3f8 << 20), r2 = 3; each mov / shli stalls its full
        // distance (assembled as written: the stalls under test are not raised).
        const auto run = [&](uint64_t fadd_stall) {
            const auto distance = std::to_string(alu_distance(profile));
            const auto text = hand_asm(profile, "mov r1, 0x3f8 {pred=PT, stall=" + distance + ", set=-, wait=-}\n"
                "shli r1, r1, 20\nmov r2, 3 {pred=PT, stall=" + distance + ", set=-, wait=-}\n"
                "fadd r3, r1, r1 {pred=PT, stall=" + std::to_string(fadd_stall) + ", set=-, wait=-}\n"
                "add r3, r2, r2\nhalt\n");
            return debug_machine(profile, nanosm::assemble_asm(text, profile));
        };
        auto ordered = run(stall);
        ordered->run();
        require(ordered->read_reg(0, 0, 3) == 6, what + "the younger add's value did not land last");
        auto early = run(stall - 2);
#if NANOSM_SANITIZER
        expect_panic(*early, {"RF WAW hazard: missing stall or depbar wait for r3"}, what + "WAW");
#else
        early->run();
        require(early->read_reg(0, 0, 3) == nanosm::float_bits(2.0f), what + "the add did not land before the fadd");
#endif
    }
}

// An async writer against a fixed booking: an lds.uni's result reaches the
// RF write port at P (issue + lds_shift + 2) in bank 0, where an add
// accepted earlier (at P - issue_to_port) booked bank 0's one port. The
// booking wins; the lds's write is denied (one wait), granted at P + 1 and
// lands Wd later; the add's is presented at P. (The lds.uni issues at
// fetch_to_issue; its stall places the add.)
TEST(Rf, async_writer_vs_fixed_booking) {
    for (const auto& [lr, wd] : rf_depths) {
        const auto profile = rf_depth_profile(lr, wd);
        const auto what = depth_what(profile);
        nanosm::Instruction add;
        add.opcode = nanosm::Opcode::add;
        const auto lds_issue = nanosm::fetch_to_issue;
        const auto p = lds_issue + lds_shift(profile) + 2;
        const auto add_issue = p - nanosm::issue_to_port(add, profile);
        auto machine = debug_machine(profile,
            "lds.uni r3, [ur1] {pred=PT, stall=" + std::to_string(add_issue - lds_issue) + ", set=sb0, wait=-}\n"
            "add r6, r1, r1\n"
            "halt {pred=PT, stall=0, set=-, wait=sb0}\n");
        const auto landed = watch_registers(*machine, {3});
        require(issued_at(*machine, 8) == std::vector<uint64_t>{add_issue}, what + "the add moved" + issue_log(*machine));
        require(rf_events(*machine, "RF write", 6) == std::vector<uint64_t>{p} &&
                rf_events(*machine, "RF write", 3) == std::vector<uint64_t>{p + 1},
            what + "the booked add did not keep the port at P, or the lds was not written at P + 1");
        require(landed.at(3) == std::vector<uint64_t>{p + 1 + wd} && machine->trace().summary().rf_async_waits == 1,
            what + "the lds's write did not land at P + 1 + Wd after one wait");
        require(machine->read_reg(0, 0, 3) == 0x100, what + "lds data");
    }
}

// The longest fixed dependency must fit the five-bit stall field
// (src/profile.cpp): Lr + E + Wd + 1 <= 31, E the longest fixed execution.
// (A longer Lr also needs the uniform writers' path urf.issue_to_port_cycles
// >= 1 + Lr: r2ur reads the vector RF.)
TEST(Rf, profile_fields) {
    auto profile = canonical_profile();
    profile.rf_read_latency_cycles = 8;
    profile.rf_write_depth_cycles = 8;
    profile.urf_issue_to_port_cycles = 1 + 8;
    profile.timing.sfu = static_cast<uint8_t>(31 - (8 + 8 + 1)); // exactly 31
    nanosm::validate_device_profile(profile);
    ++profile.timing.sfu; // 32
    expect_error<nanosm::ProfileError>([&] { nanosm::validate_device_profile(profile); }, {"five-bit stall field"},
        "a fixed dependency longer than the stall field");
}

// Back-to-back same-bank reads. Nine independent instructions issue in
// consecutive cycles: seven read RF bank 0 in their issue round (r0, r3, r6,
// r9: two-source adds, a repeated register, xori, addi, stg, setp), an
// lds.uni among them reads only its uniform base (ur0, no RF read) and an
// lds reads its r9 later, in its collect stage (after the run). No refusal,
// no bubble; bank 0 is sampled in every cycle of the run but the lds.uni's
// and the lds's, each instruction's sources at its own issue +
// issue_to_sample; the data are right. Then four warps run the arithmetic
// part of it: their issues interleave one per cycle with no gap, and bank 0
// is sampled once per cycle.
TEST(Rf, back_to_back_same_bank_reads) {
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
        {0, 1}, {3, 2}, {6}, {9}, {}, {9, 1}, {6}, {}, {9}};
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
    const auto& profile = canonical_profile();
    for (const auto& [warps, body, sources] : std::vector<std::tuple<uint32_t, std::string,
             std::vector<std::vector<uint32_t>>>>{{1, mixed, mixed_sources}, {4, arithmetic, arithmetic_sources}}) {
        const auto what = std::to_string(warps) + " warp(s): ";
        auto machine = debug_machine(profile, body, warps);
        machine->run();
        const auto log = " | issues" + issue_log(*machine);
        require(machine->trace().summary().rf_simt_read_inhibited == 0, what + "a read was inhibited");
        std::set<uint64_t> run_issues;
        std::set<uint64_t> bank0_rounds;            // the read rounds of run instructions that read bank 0
        std::map<uint64_t, uint32_t> bank0_samples; // SIMT samples of bank 0 by cycle
        for (const auto& event : machine->trace().events()) {
            if (event.name == "RF read" && event.value == 0) ++bank0_samples[event.start_cycle];
            if (event.category == nanosm::TraceCategory::warp && event.detail.rfind("RF", 0) == 0) {
                require(false, what + "an RF refusal: " + event.detail + log);
            }
        }
        for (uint32_t warp = 0; warp < warps; ++warp) {
            std::vector<uint64_t> issues;
            for (std::size_t index = 0; index < sources.size(); ++index) {
                const auto cycles = warp_issue_cycles(*machine, warp, static_cast<uint32_t>(96 + 8 * index));
                require(cycles.size() == 1, what + "run instruction " + std::to_string(index) + " did not issue once" + log);
                const auto cycle = cycles[0];
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

// Two DISTINCT source registers of one instruction in one RF bank (one read
// round, one read port per bank): qtxc repairs it with a copy of one source
// into another bank and a warning; the hand assembler and kernel load refuse
// it, in both builds, naming the instruction and the registers. The rows
// cover two- and three-source arithmetic and setp. (An sts's base and value
// are read by its collect stage, one after the other when they share a bank:
// Shmem.sts_registers_in_one_bank.)
// The assembler and kernel-load cases carry words a looser profile (more
// read ports) produced, restamped for the profile under test. The atomic, the
// mbar register forms and mma.async's two addresses take uniform registers,
// which are no RF reads: those forms cannot conflict (nor an acc.st's
// uniform tmem pointer beside its vector data).
TEST(Rf, same_bank_sources_are_malformed) {
    const auto& profile = canonical_profile();
    const std::vector<std::pair<std::string, std::string>> cases{
        {"add r2, r0, r3", "r0 and r3"},
        {"fma r1, r2, r5, r8", "r2, r5 and r8"},
        {"setp.eq.u32 p0, r0, r3", "r0 and r3"},
    };
    const auto mentions = [](const std::string& message, const std::string& where, const std::string& registers) {
        return message.rfind(where, 0) == 0 && message.find(": malformed instruction '") != std::string::npos &&
            message.find("its distinct sources " + registers + " are in one RF bank") != std::string::npos;
    };
    for (const auto& [line, registers] : cases) {
        const auto source = line + "\nhalt\n";
        // qtxc's repair passes the assembler.
        const auto repaired = qtx_program(source, profile);
        require(repaired.copies >= 1 && repaired.warnings.size() == repaired.copies &&
                repaired.warnings.front().rfind("line 1: '", 0) == 0 &&
                repaired.warnings.front().find("from one register-file bank; qtxc inserted a copy of %r") !=
                    std::string::npos &&
                repaired.instructions.size() == 2 + repaired.copies &&
                repaired.instructions.front().opcode == nanosm::Opcode::addi,
            "qtxc's repair of '" + line + "': " +
                (repaired.warnings.empty() ? std::string("no warning") : repaired.warnings.front()));
        (void)nanosm::assemble_asm(nanosm::emit_asm(repaired), profile);
        // The same line through a looser profile (one read port per source).
        auto loose = profile;
        loose.rf_read_ports_per_bank = 3;
        auto program = qtx_program(source, loose);
        program.profile_fingerprint = nanosm::device_profile_fingerprint(profile);
        const auto assembler = error_of<nanosm::SimError>(
            [&] { (void)nanosm::assemble_asm(nanosm::emit_asm(program), profile); }, "the assembler on '" + line + "'");
        require(mentions(assembler, "line ", registers), "the assembler's refusal of '" + line + "': " + assembler);
        auto kernel = nanosm::assemble_asm(nanosm::emit_asm(qtx_program(source, loose)), loose);
        kernel.profile_fingerprint = nanosm::device_profile_fingerprint(profile);
        const auto load = error_of<nanosm::SimError>(
            [&] { nanosm::Machine machine(profile, kernel, launch_warps(1)); }, "kernel load of '" + line + "'");
        require(mentions(load, "kernel pc 0", registers), "kernel load's refusal of '" + line + "': " + load);
    }
    // Uniform operands are no RF reads: with register numbers that would
    // share an RF bank (3 and 6; ur0 beside r3), qtxc and the assembler
    // accept these forms.
    for (const std::string line : {"atom.add.uni r1, [ur3], ur6", "mbar.init [ur3], ur6",
             "mbar.arrive.expect_tx [ur3], ur6", "mbar.test_wait p0, [ur3], ur6", "mbar.try_wait p0, [ur3], ur6",
             "mma.async.m64n64k8 [ur0], [ur3], [ur6]", "acc.st [ur0, 64], r3"}) {
        (void)nanosm::assemble_asm(nanosm::emit_asm(qtx_program(line + "\nhalt\n", profile)), profile);
    }
    // The same register in several operand slots is one read: qtxc, the
    // assembler and kernel load accept it, and it computes.
    auto launch = launch_warps(1);
    image_bytes(launch.test_shmem, 256);
    nanosm::Machine machine(profile, compile(R"(
        sread r3, LANE_ID
        shli r3, r3, 2
        add r2, r3, r3
        mov r4, 0x3f8
        shli r4, r4, 20
        fma r5, r4, r4, r4
        setp.eq.u32 p0, r3, r3
        sts [r3 + 64], r3
        halt
    )", profile), std::move(launch), {.timeline = true, .debug = true});
    machine.run();
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(machine.read_reg(0, lane, 2) == 8 * lane && machine.read_reg(0, lane, 5) == nanosm::float_bits(2.0f) &&
                machine.read_shmem_u32(64 + 4 * lane) == 4 * lane,
            "a repeated source register computed the wrong value");
    }
    check_read_capacity(machine, profile);
}

// One async reader under continuous SIMT reads of its bank: an acc.st's data
// beat (r0, bank 0) against addis that read r27 (bank 0) every cycle (its
// tmem pointer is a uniform register: its own read round takes no RF port).
// The acc.st, accepted at a, reaches its unit at a + issue_to_dispatch and
// its beat wants bank 0 from a + 4, where the issue register (the addi
// accepted at a + 3) owns bank 0: the beat is denied with service owed, an
// inhibition -- the next candidate, an add reading banks 0 AND 1, is refused
// at a + 4, all or none: nothing of it is read in a + 5, bank 1 included. At
// a + 5 the beat takes bank 0 and the add is accepted in the same cycle.
// Every accepted instruction samples at its issue + 1; the stream loses
// exactly one cycle.
//
// Unrelated banks proceed: three warps, warps 0 and 1 reading bank 0 (r27)
// every cycle they issue, warp 2 only bank 1 (r28); warp 0 issues an acc.st
// of r0 amid its stream. In steady round-robin (w0, w1, w2) the beat wants
// bank 0 at a + 4, right after warp 0's addi at a + 3: warp 1's bank-0 addi
// is refused at a + 4, warp 2's bank-1 addi is accepted in that cycle, and
// the beat takes the hole at a + 5. The warps first meet at an mbar barrier
// so that every stream starts with its FIFO full (three warps issuing every
// cycle share one fetch a cycle, so a stream started behind a taken branch
// would get its words just in time, not early).
TEST(Rf, async_reader_gets_the_hole) {
    const auto& profile = canonical_profile();
    {
        auto machine = debug_machine(profile, R"(
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
        )");
        machine->run();
        const auto log = " | issues" + issue_log(*machine);
        const auto a = issued_at(*machine, 5 * 8).at(0);
        const std::vector<uint64_t> expected{a + 1, a + 2, a + 3, a + 5, a + 6};
        for (uint32_t i = 0; i < expected.size(); ++i) {
            require(issued_at(*machine, (6 + i) * 8) == std::vector<uint64_t>{expected[i]},
                "SIMT instruction " + std::to_string(i) + " did not issue at its cycle" + log);
        }
        require(refusals(*machine, 0, "RF read inhibited") == Slices{{a + 4, 1}},
            "the refusal:" + describe(refusals(*machine, 0, "RF read inhibited")) + log);
        const auto reads = warp0_reads(*machine);
        require(reads.contains(a + 5) && reads.at(a + 5) == std::vector<std::pair<uint32_t, uint32_t>>{{0, 0}},
            "in the hole cycle the beat was not the only read (a refused instruction read a register?)");
        require(rf_events(*machine, "RF read", 27) == std::vector<uint64_t>{a + 2, a + 3, a + 4, a + 6, a + 7} &&
                rf_events(*machine, "RF read", 28) == std::vector<uint64_t>{a + 6},
            "an accepted instruction was not sampled at its issue + 1");
        require(machine->trace().summary().rf_simt_read_inhibited == 1, "inhibited cycles");
        for (uint32_t lane = 0; lane < 16; ++lane) {
            require(machine->read_reg(0, lane, 4) == 12 && machine->read_reg(0, lane, 10) == lane,
                "the add or the acc.st -> acc.ld round trip");
        }
        check_read_capacity(*machine, profile);
    }
    {
        std::string pad;
        for (uint32_t i = 0; i < 12; ++i) pad += "addi r1, r27, 1\n";
        std::string bank0;
        for (uint32_t i = 0; i < 24; ++i) bank0 += "addi r2, r27, 2\n";
        std::string bank1;
        for (uint32_t i = 0; i < 48; ++i) bank1 += "addi r4, r28, 3\n";
        const auto meet = [](const std::string& label) {
            return "mbar.arrive [ur6]\n" + label + ": mbar.try_wait p3, [ur6], 0\n@!p3 bra.uni " + label + "\n";
        };
        auto launch = launch_warps(3);
        write_mbar(image_bytes(launch.test_shmem, 8192), 4096, mbar_state(3, 3));
        // Warp 0's bank-0 stream is its own copy: a branch to warp 1's would
        // drain the acc.st's depbar first.
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
        const auto issues = issue_records(machine);
        const auto acc_st = std::find_if(issues.begin(), issues.end(), [](const auto& i) { return i.opcode == "acc.st"; });
        require(acc_st != issues.end(), "the acc.st did not issue" + log);
        const auto a = acc_st->cycle;
        const auto warp_issued_at = [&](uint32_t warp, uint64_t cycle) {
            return std::any_of(issues.begin(), issues.end(),
                [&](const auto& i) { return i.warp == warp && i.cycle == cycle; });
        };
        require(warp_issued_at(0, a + 3) && warp_issued_at(1, a + 1) && warp_issued_at(2, a + 2),
            "the three warps were not in steady round-robin around the acc.st" + log);
        require(refusals(machine, 1, "RF read inhibited") == Slices{{a + 4, 1}},
            "warp 1's bank-0 addi was not refused exactly at a + 4:" + describe(refusals(machine, 1, "RF read inhibited")) + log);
        require(warp_issued_at(2, a + 4), "warp 2's bank-1 addi did not issue in the inhibited cycle" + log);
        require(refusals(machine, 2, "RF read inhibited").empty(), "warp 2 was inhibited for a bank it does not read");
        require(last_rf_read(machine, 0) == a + 5, "the beat did not take the hole at a + 5" + log);
        for (uint32_t warp = 0; warp < 3; ++warp) {
            require(machine.read_reg(warp, 0, warp == 2 ? 4 : 2) == (warp == 2 ? 10u : 7u), "data");
        }
        check_read_capacity(machine, profile);
    }
}

// The two async readers of one bank -- the acc.st engine's beat (holder 0)
// and an stg.v8's store-data beats (the LSU, holder 1) -- take the holes in
// a stream of addis reading r27 (bank 0) in round-robin order. First the
// order: the acc.st (a) wants r0 from a + 4, the stg.v8 (a + 1) its r18 from
// a + 4; the issue register's addi owns bank 0 there, so both are denied
// with service owed (refusal at a + 4); the hole at a + 5 goes to the acc.st,
// whose grant lifts the inhibition; at a + 6 the LSU is denied again
// (refusal) and at a + 7 it takes the hole. Then persistent pressure (60
// addis) against acc.st.v8 and stg.v8, both coming back to bank 0 every
// third register: no accepted read is ever stopped, every acc.st beat is
// granted within 2 x 2 - 1 = 3 cycles of wanting its bank (in bank 0 one hole
// per two cycles, the other reader at most ahead of it). In both, every
// inhibited cycle's hole goes to an async reader the next cycle, and the
// data (tmem read back, gmem) are right.
TEST(Rf, async_readers_round_robin) {
    const auto stg_data = [](const nanosm::Machine& machine) {
        for (uint32_t lane = 0; lane < 16; ++lane) {
            for (uint32_t j = 0; j < 8; ++j) {
                require(machine.read_gmem_u32(32 * lane + 4 * j) == lane + 200 + 10 * j, "stg.v8 data");
            }
        }
    };
    {
        // One lane group per stg.v8: it reads each of r18 .. r25 once.
        auto profile = canonical_profile();
        profile.lsu_store_group_lanes = 16;
        std::string setup = "sread r0, LANE_ID\naddi r9, r0, 100\n";
        for (uint32_t j = 0; j < 8; ++j) setup += "addi r" + std::to_string(18 + j) + ", r0, " + std::to_string(200 + 10 * j) + "\n";
        setup += "shli r40, r0, 5\n"; // the stg.v8's per-lane sectors: 32 x lane
        std::string stream;
        for (uint32_t pass = 0; pass < 2; ++pass) {
            for (const auto& [reg, imm] : std::vector<std::pair<uint32_t, uint32_t>>{{1, 1}, {2, 2}, {4, 3}, {5, 4},
                     {7, 5}, {8, 6}, {10, 7}, {11, 8}, {13, 9}, {14, 10}, {16, 11}, {17, 12}}) {
                stream += "addi r" + std::to_string(reg) + ", r27, " + std::to_string(imm) + "\n";
            }
        }
        auto machine = debug_machine(profile, R"(
            umov ur2, 0
            umov ur3, 0
        )" + setup + R"(
            mov r27, 5 {pred=PT, stall=8, set=-, wait=-}
            acc.st [ur2, 64], r0 {pred=PT, stall=0, set=sb0, wait=-}
            stg.v8 [r40], r18 {pred=PT, stall=0, set=sb2, wait=-}
        )" + stream + R"(
            acc.ld r50, [ur2, 64] {pred=PT, stall=0, set=sb3, wait=sb0}
            halt {pred=PT, stall=0, set=-, wait=sb2|sb3}
        )");
        machine->run();
        const auto log = " | issues" + issue_log(*machine);
        const auto a = issued_at(*machine, 14 * 8).at(0);
        const auto refused = refusals(*machine, 0, "RF read inhibited");
        require(refused.size() >= 2 && refused[0] == std::pair<uint64_t, uint64_t>{a + 4, 1} &&
                refused[1] == std::pair<uint64_t, uint64_t>{a + 6, 1},
            "the first two refusals:" + describe(refused) + log);
        require(last_rf_read(*machine, 0) == a + 5 && last_rf_read(*machine, 18) == a + 7,
            "the holes did not go acc.st, LSU in turn" + log);
        require_holes_used(*machine, refused);
        for (uint32_t lane = 0; lane < 16; ++lane) {
            require(machine->read_reg(0, lane, 50) == lane, "tmem data (acc.st -> acc.ld)");
        }
        stg_data(*machine);
        check_read_capacity(*machine, profile);
    }
    {
        const auto& profile = canonical_profile();
        std::string setup = "sread r0, LANE_ID\n";
        for (uint32_t j = 1; j < 8; ++j) setup += "addi r" + std::to_string(j) + ", r0, " + std::to_string(10 * j) + "\n";
        for (uint32_t j = 0; j < 8; ++j) setup += "addi r" + std::to_string(9 + j) + ", r0, " + std::to_string(100 + 10 * j) + "\n";
        for (uint32_t j = 0; j < 8; ++j) setup += "addi r" + std::to_string(18 + j) + ", r0, " + std::to_string(200 + 10 * j) + "\n";
        setup += "shli r40, r0, 5\n"; // the stg.v8's per-lane sectors: 32 x lane
        std::string pressure;
        for (uint32_t i = 0; i < 60; ++i) pressure += "addi r41, r27, " + std::to_string(i) + "\n";
        auto machine = debug_machine(profile, R"(
            umov ur2, 0
            umov ur3, 0
        )" + setup + R"(
            mov r27, 5 {pred=PT, stall=8, set=-, wait=-}
            acc.st.v8 [ur2, 64], r0 {pred=PT, stall=0, set=sb0, wait=-}
            stg.v8 [r40], r18 {pred=PT, stall=0, set=sb2, wait=-}
        )" + pressure + R"(
            acc.ld.v8 r50, [ur2, 64] {pred=PT, stall=0, set=sb3, wait=sb0}
            halt {pred=PT, stall=0, set=-, wait=sb2|sb3}
        )");
        machine->run();
        const auto log = " | issues" + issue_log(*machine);
        const uint32_t acc_st = 28; // after two umovs, 25 setup instructions and the mov r27
        const auto a = issued_at(*machine, acc_st * 8).at(0);
        const auto t = nanosm::issue_to_dispatch(profile);
        // Accepted reads are never stopped.
        std::set<uint64_t> expected_r27;
        for (uint32_t i = 0; i < 60; ++i) expected_r27.insert(issued_at(*machine, (acc_st + 2 + i) * 8).at(0) + 1);
        const auto r27 = rf_events(*machine, "RF read", 27);
        require(std::set<uint64_t>(r27.begin(), r27.end()) == expected_r27 && r27.size() == 60,
            "an accepted addi was not sampled at its issue + 1" + log);
        // Bounded waits of the acc.st beats: beat j wants its bank from its
        // transfer's dispatch + 1 (beat 0) or the cycle after beat j - 1's grant.
        uint32_t longest = 0;
        auto ready = a + t + 1;
        for (uint32_t j = 0; j < 8; ++j) {
            const auto grant = last_rf_read(*machine, static_cast<uint8_t>(j));
            require(grant >= ready, "a beat was read before it was wanted");
            longest = std::max(longest, static_cast<uint32_t>(grant - ready));
            ready = grant + 1;
        }
        require(longest <= 3, "an acc.st beat waited " + std::to_string(longest) + " cycles" + log);
        require(machine->trace().summary().rf_simt_read_inhibited > 0, "the pressure never needed an inhibition");
        require_holes_used(*machine, refusals(*machine, 0, "RF read inhibited"));
        for (uint32_t lane = 0; lane < 16; ++lane) {
            for (uint32_t j = 0; j < 8; ++j) {
                require(machine->read_reg(0, lane, 50 + j) == lane + 10 * j, "tmem data (acc.st.v8 -> acc.ld.v8)");
            }
        }
        stg_data(*machine);
        check_read_capacity(*machine, profile);
    }
}

// 16 warps, each touching its own state: its RF rows and predicates, its
// uniform registers (WARP_ID through usread, the mbar address), WARP_ID /
// THREAD_ID, a depbar (lds), its shmem order (lds behind its own sts), the
// stall count, an mbar arrival and a try_wait on a barrier all 16 arrive at
// (each in its own wait slot), and its trace track.
TEST(Rf, sixteen_warps) {
    const auto& profile = canonical_profile();
    auto launch = launch_warps(16);
    write_mbar(image_bytes(launch.test_shmem, 8192), 4096, mbar_state(16, 16));
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
}

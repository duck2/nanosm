// The RMW engine and store pause (I3), uniform atomics and D12 (I5).
// Moved from the original tests/tests.cpp (trash/test-setup-2026-10-22.md);
// each test's body is unchanged, its TEST wrapper is at the end.
#include "test_support.hpp"

namespace {

using namespace nanosm_test;

// Issue to dependent issue behind a uniform-register writer (umov, usread,
// r2ur): visible issue_to_urf_visible after its issue, sampled at the
// reader's issue + 1 (src/rf.hpp; 4 at canonical, where alu_distance is 6).
uint64_t urf_distance(const nanosm::DeviceProfile& profile) {
    return nanosm::issue_to_urf_visible(profile) - nanosm::issue_to_urf_sample;
}

// Every RMW, in order (the engine runs one at a time): start, read, write.
std::vector<I3Rmw> i3_rmws(const nanosm::Machine& machine) {
    std::vector<I3Rmw> rmws;
    for (const auto& [start, detail] : mbar_rmw_services(machine)) rmws.push_back({start, 0, 0, detail});
    std::size_t reads = 0;
    std::size_t writes = 0;
    for (const auto& grant : i2_grants(machine)) {
        if (grant.client != "rmw") continue;
        auto& next = grant.service == "A-read" ? reads : writes;
        require(next < rmws.size(), "an RMW grant without an RMW slice");
        (grant.service == "A-read" ? rmws[next].read : rmws[next].write) = grant.cycle;
        ++next;
    }
    return rmws;
}

// The cycles store transactions started (their first round's grant).
std::vector<uint64_t> i3_store_starts(const nanosm::Machine& machine) {
    std::vector<uint64_t> starts;
    for (const auto& event : machine.trace().events()) {
        if (event.category == nanosm::TraceCategory::shmem && event.kind == nanosm::TraceEventKind::slice &&
            event.name == "sts") {
            starts.push_back(event.start_cycle);
        }
    }
    std::sort(starts.begin(), starts.end());
    return starts;
}

// rfc12 section 6, the pause sequence with a store transaction under way
// when the RMW starts; at D = 2 and D = 3. All lanes on one bank: sts X is
// sixteen rounds, granted c4..c19. The host's event is held from c8: the RMW
// starts there (the store pause begins at that edge; X, under way, keeps its
// rounds); its read is granted at c20, strictly after X's last round (matched
// D: X's last write lands at c19 + D, before the RMW samples at c20 + D); its
// write at w = c20 + D + Lb + C. The younger sts Y, admitted at X's last
// grant, waits in its slot and is granted at w + 1 -- one cycle after the
// RMW's final write GRANT, not after its commit -- and still reaches the RAM
// after the RMW's write: the mbar lands at the edge of w + D, Y's first round
// at w + 1 + D. No store round between the RMW's read and its write.
// I6a: the same sequence relative to X's first grant g (X issues at s =
// 2 alu_distance + 1 -- it needs the mov's r5 -- and reaches its slot
// issue_to_dispatch later, t): X g .. g + 15, the event held from g + 4,
// the RMW read at g + 16; Y admitted at g + 15 waits in its slot from
// g + 16 + t to w (store_paused w - (g + 15 + t)). (I6b-2: the sread
// issues at fetch_to_issue, so s moves by that.)
void test_i3_store_pause_sequence() {
    for (const uint32_t d : {2u, 3u}) {
        auto profile = canonical_profile();
        profile.shmem_port_a_depth = d;
        const uint64_t lb = profile.shmem_read_latency_cycles;
        const uint64_t c = profile.shmem_rmw_compute_cycles;
        const auto what = "D=" + std::to_string(d) + ": ";
        auto machine = i2_machine(profile, R"(
            sread r0, LANE_ID
            shli r0, r0, 6
            mov r5, 0x5a
            sts [r0 + 8], r5 {pred=PT, stall=0, set=sb0, wait=-}
            sts [r0 + 12], r5 {pred=PT, stall=0, set=sb1, wait=-}
            halt {pred=PT, stall=0, set=-, wait=sb0|sb1}
        )");
        const auto t = nanosm::issue_to_dispatch(profile);
        const auto g = nanosm::fetch_to_issue + 2 * alu_distance(profile) + 1 + t + 1;
        i2_run_until(*machine, g + 4);
        host_rmw_event(*machine);
        std::optional<uint64_t> mbar_landed; // the cycle at whose clock edge the RMW's write landed
        std::optional<uint64_t> y_landed;    // ... Y's round 0 (lane 0's word)
        while (!machine->done()) {
            const auto cycle = machine->cycle();
            tick_capped(*machine);
            if (!mbar_landed && machine->read_mbar_state(i2_event_mbar).phase) mbar_landed = cycle;
            if (!y_landed && machine->read_shmem_u32(12) == 0x5a) y_landed = cycle;
        }
        const uint64_t w = g + 16 + d + lb + c;
        const auto rmws = i3_rmws(*machine);
        require(rmws.size() == 1 && rmws[0].start == g + 4 && rmws[0].read == g + 16 && rmws[0].write == w,
            what + "the RMW did not start at g + 4, read at g + 16 (X's last round + 1), write at g + 16 + D + Lb + C:" +
                describe(rmws) + " | issues" + issue_log(*machine));
        std::vector<uint64_t> stores;
        for (const auto& grant : i2_grants(*machine, "A-write")) {
            if (grant.client == "simt") stores.push_back(grant.cycle);
        }
        std::vector<uint64_t> expected;
        for (uint64_t round = 0; round < 16; ++round) expected.push_back(g + round);     // X
        for (uint64_t round = 0; round < 16; ++round) expected.push_back(w + 1 + round); // Y
        require(stores == expected, what + "store rounds (X g..g + 15, Y from w + 1):" +
            describe(i2_grants(*machine, "A-write")));
        require(issued_at(*machine, 4 * 8) == std::vector<uint64_t>{g + 15}, what + "Y was not admitted at X's last grant");
        require(mbar_landed == w + d && y_landed == w + 1 + d,
            what + "the RMW's write did not land at w + D, or Y's first round at w + 1 + D");
        require(machine->trace().summary().shmem_store_paused == w - (g + 15 + t),
            what + "store_paused should count g + 16 + t .. w (Y waiting in its slot under the pause)");
        for (uint32_t lane = 0; lane < 16; ++lane) {
            require(machine->read_shmem_u32(64 * lane + 8) == 0x5a && machine->read_shmem_u32(64 * lane + 12) == 0x5a,
                what + "store data");
        }
    }
}

// rfc12 section 6: reads continue during an RMW. The host's event is held
// from c5 and read at c5 (nothing waits on A); its write is granted at c5 +
// D + Lb + C = c11. In between A-read serves warp 0's lds.uni (c6), the
// TMA's descriptor read (c7) and warp 0's probe (c8).
// I6a: the three requests issue at c5, c6, c7 as before but reach A-read
// issue_to_dispatch later, so the event is placed at the lds.uni's dispatch
// e = 5 + issue_to_dispatch: the RMW reads at e, the lds.uni, the TMA and
// the probe are granted at e + 1, e + 2, e + 3, before its write at
// e + D + Lb + C. (I6b-2: the program starts at fetch_to_issue, so the
// lds.uni issues at fetch_to_issue + 5.)
// The tma reads its operands from uniform registers in two rounds (buffer
// and mbar at its issue + 1, the coordinate at + 2) and its command reaches
// the TMA at issue + 3, the old dispatch: its descriptor read stays at e + 2.
// The probe reads a uniform register, so issue refuses it in the cycle after
// the tma (its sample would meet the coordinate round): it issues one cycle
// later and is granted at e + 4 (was e + 3), still before the RMW's write.
void test_i3_reads_during_an_rmw() {
    const auto& profile = canonical_profile();
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.shmem_image, 8192);
    for (uint32_t word = 0; word < 512; ++word) write_u32(shmem, 4 * word, 0x100 + word);
    write_u64(shmem, i2_event_mbar, nanosm::encode_mbar_state(i21_mbar(1, 0, 64)));
    write_rank1_load(shmem, 5120, 0, 16, 4104);
    fill_pattern(image_bytes(launch.gmem_image, 64), 0, 16, 0);
    // The TMA's operands (buffer at 1024, completing into 4104, coordinate 0)
    // take the first three of the five issue slots before the lds.uni, so the
    // timeline is unchanged. The lds.uni's base ur1 is never written (0); the
    // probe's mbar is in ur12.
    nanosm::Machine machine(profile, hand_assemble(i1_asm(profile, i4_tma_operands(1024) + R"(
        umov ur12, 4104
        mov r25, 0
        lds.uni r3, [ur1 + 64] {pred=PT, stall=0, set=sb0, wait=-}
        tma.load.1d [ur10], [5120, ur8], [ur11]
        mbar.test_wait p0, [ur12], 0
        halt {pred=PT, stall=0, set=-, wait=sb0}
    )"), profile), std::move(launch), nanosm::TraceOptions{.timeline = true, .debug = true});
    const uint64_t e = nanosm::fetch_to_issue + 5 + nanosm::issue_to_dispatch(profile);
    i2_run_until(machine, e);
    host_rmw_event(machine);
    machine.run();
    const uint64_t w = e + i2_return(profile) + profile.shmem_rmw_compute_cycles;
    const auto rmws = i3_rmws(machine);
    require(rmws.size() == 2 && rmws[0].start == e && rmws[0].read == e && rmws[0].write == w,
        "the host event's RMW:" + describe(rmws));
    auto reads = i2_grants(machine, "A-read");
    reads.resize(std::min<std::size_t>(reads.size(), 4));
    require(reads == std::vector<I2Grant>{{e, "A-read", "rmw", 0}, {e + 1, "A-read", "simt", 0},
                         {e + 2, "A-read", "tma", 0}, {e + 4, "A-read", "probe", 0}} && reads.back().cycle < w,
        "A-read did not serve an lds, the TMA and a probe between the RMW's read and its write:" + describe(reads) +
            " | issues" + issue_log(machine));
    for (uint32_t lane = 0; lane < 16; ++lane) require(machine.read_reg(0, lane, 3) == 0x100 + 16, "lds.uni data");
    for (uint32_t i = 0; i < 16; ++i) require(machine.read_shmem_u32(1024 + 4 * i) == pattern_word(i), "TMA data");
}

// rfc12 sections 1 and 6: a read of the mbar that samples in the cycle the
// RMW's write reaches the RAM sees the OLD word (READ_FIRST). The host's
// event is read at c0; its write is granted at w = D + Lb + C and lands at
// w + D, completing the phase. Warp 0's test_wait, placed by its stall count,
// is granted at g and samples at g + D: it sees the new phase iff g > w --
// granted at w itself (the write-grant cycle) it reads the old phase.
// I6a: the test_wait needs the first mov's r4 (alu_distance) and reaches
// the probe register at its dispatch, so its probe is granted at
// alu_distance + issue_to_dispatch + 1 at the earliest (stalls 0..5), one
// cycle later per stall beyond. The host's event is placed (read at
// k = w - D - Lb - C) so that its write grant w is one cycle after that
// earliest grant: stalls 0..8 still cross w. (I6b-2: the first mov issues at
// fetch_to_issue.) URF: the mbar address is a umov's ur4, so the test_wait
// needs urf_distance (4) instead of alu_distance: its earliest grant is at
// stalls 0..3, and stalls 0..8 still cross w.
void test_i3_read_first_at_rmw_write() {
    const auto& profile = canonical_profile();
    const uint64_t rmw = i2_return(profile) + profile.shmem_rmw_compute_cycles;
    const uint64_t w = nanosm::fetch_to_issue + urf_distance(profile) + nanosm::issue_to_dispatch(profile) + 2;
    const uint64_t k = w - rmw;
    std::set<int> sides;
    for (uint32_t stall = 0; stall <= 8; ++stall) {
        auto machine = i2_machine(profile, R"(
            umov ur4, 4096 {pred=PT, stall=)" + std::to_string(stall) + R"(, set=-, wait=-}
            mov r9, 0
            mbar.test_wait p0, [ur4], 0
            bra.uni done {pred=!p0, stall=0, set=-, wait=-}
            mov r9, 1
        done: halt
        )");
        i2_run_until(*machine, k);
        host_rmw_event(*machine);
        machine->run();
        uint64_t probe = 0;
        for (const auto& grant : i2_grants(*machine, "A-read")) {
            if (grant.client == "probe") probe = grant.cycle;
        }
        const auto rmws = i3_rmws(*machine);
        const auto what = "probe granted c" + std::to_string(probe) + "," + describe(rmws) + ": ";
        require(probe != 0 && rmws.size() == 1 && rmws[0].read == k && rmws[0].write == w, what + "timeline");
        require(machine->read_reg(0, 0, 9) == (probe > w ? 1u : 0u),
            what + (probe > w ? "the probe missed the RMW's write" : "READ_FIRST: the probe did not read the old phase"));
        sides.insert(probe < w ? -1 : probe == w ? 0 : 1);
    }
    require(sides == std::set<int>{-1, 0, 1}, "the sweep did not cross the RMW's write-grant cycle");
}

// rfc12 section 5: the alternation bit between RMW starts and store
// transactions. Warp 0 issues eight mbar.arrive back to back (each waits at
// issue for the previous one's final write grant w, so the next RMW waits to
// start from w + 1); warp 1 issues eight one-round sts on the same words
// (each waits for the previous one's grant). Whenever both wait, the one
// that did not start last goes: the waiting store starts at w + 1 and the
// next RMW at w + 2 (its read at once: the store's one round was granted at
// w + 1). So the starts go R, S, R, S, ... -- back-to-back RMWs do not starve
// the stores -- and no store round is granted between an RMW's read and its
// write.
// I6a re-derivation: an arrive now reaches the SIMT RMW register
// issue_to_dispatch after its issue, so one warp's next arrive (issued at
// the previous one's write grant w) is there only at w + issue_to_dispatch
// + 1 -- after the waiting store already started alone at w + 1: the two
// never waited together and the alternation went unexercised. The arrives
// now come from two warps (0 and 2, four each): while one's RMW runs, the
// other's waits in the register, so at every w + 1 both classes wait, as in
// the one-warp scenario before I6a. Warp 1 issues the eight stores. The
// arrive warps' umov ur4 stalls 10 cycles (its result needs urf_distance,
// 4) so that warp 1's first store reaches its slot while the first RMW runs:
// it waits under that RMW's pause, as every later one does.
void test_i3_rmw_store_alternation() {
    const auto& profile = canonical_profile();
    std::string arrives;
    std::string stores;
    for (int i = 0; i < 8; ++i) {
        if (i < 4) arrives += "mbar.arrive [ur4]\n";
        stores += "sts [r20 + 2048], r21 {pred=PT, stall=0, set=sb0, wait=-}\n";
    }
    auto machine = i21_machine(profile, R"(
        usread ur9, WARP_ID
        ur2r r29, ur9
        setp.eq p6, r29, 1
        bra.uni stores {pred=p6, stall=0, set=-, wait=-}
        umov ur4, 4160 {pred=PT, stall=10, set=-, wait=-}
    )" + arrives + R"(
        halt
    stores: sread r20, LANE_ID
        shli r20, r20, 2
        mov r21, 0x33
    )" + stores + R"(
        halt {pred=PT, stall=0, set=-, wait=sb0}
    )", {{4160, i21_mbar(8, 8)}}, 3);
    machine->run();
    const auto rmws = i3_rmws(*machine);
    const auto store_starts = i3_store_starts(*machine);
    std::vector<std::pair<uint64_t, char>> starts;
    for (const auto& rmw : rmws) starts.emplace_back(rmw.start, 'R');
    for (const auto start : store_starts) starts.emplace_back(start, 'S');
    std::sort(starts.begin(), starts.end());
    std::string order;
    for (const auto& [cycle, kind] : starts) order += kind;
    require(order == "RSRSRSRSRSRSRSRS", "RMW (R) and store (S) starts did not alternate: " + order + describe(rmws));
    for (std::size_t k = 0; k < 8; ++k) {
        require(store_starts[k] == rmws[k].write + 1, "store " + std::to_string(k) +
            " did not start the cycle after RMW " + std::to_string(k) + "'s write grant:" + describe(rmws));
        if (k + 1 < 8) {
            require(rmws[k + 1].start == rmws[k].write + 2 && rmws[k + 1].read == rmws[k + 1].start,
                "RMW " + std::to_string(k + 1) + " did not start (and read) at w + 2:" + describe(rmws));
        }
    }
    for (const auto& grant : i2_grants(*machine, "A-write")) {
        for (const auto& rmw : rmws) {
            require(grant.client != "simt" || grant.cycle < rmw.read || grant.cycle > rmw.write,
                "a store round was granted between an RMW's read and its write, c" + std::to_string(grant.cycle) +
                    ":" + describe(rmws));
        }
    }
    require(machine->read_mbar_state(4160).phase, "the eight arrives did not complete the phase");
    for (uint32_t lane = 0; lane < 16; ++lane) require(machine->read_shmem_u32(2048 + 4 * lane) == 0x33, "store data");
}

// rfc12 section 6: a SIMT RMW takes the engine only when its warp has no
// unserviced lds / sts -- it waits for its own stores at issue, BEFORE the
// SIMT RMW register and the engine, so its store pause can never hold them
// (no deadlock). The host's event starts an RMW at c0 (read c0, write c6):
// its pause holds warp 0's sixteen-round sts, admitted at c4, until c7. The
// warp's mbar.arrive is ordered behind that sts (refused c5..c21), admitted
// at its last grant (c22), starts the next cycle with nothing to wait for
// (read c23) and is written D + Lb + C later.
// I6a: the sts issues at s (it needs the mov's r5: 2 alu_distance + 2) and
// reaches its slot at s + issue_to_dispatch (t). The host's event is placed
// as the sts issues (held, started and read at k = s + 1; written at w =
// k + D + Lb + C), so its pause holds the sts in its slot until w; the sts
// is granted w + 1 .. w + 16. The arrive is ordered behind it from s + 1,
// admitted at its last grant (w + 16), reaches the register at its dispatch
// and starts (reads) the cycle after, w + t + 17. (I6b-2: the sread issues
// at fetch_to_issue.)
void test_i3_simt_rmw_waits_for_own_stores() {
    const auto& profile = canonical_profile();
    const uint64_t rmw = i2_return(profile) + profile.shmem_rmw_compute_cycles; // read to write
    const uint64_t t = nanosm::issue_to_dispatch(profile);
    auto machine = i21_machine(profile, R"(
        sread r0, LANE_ID
        shli r0, r0, 6
        umov ur4, 4160
        mov r5, 0x5a
        sts [r0 + 8], r5 {pred=PT, stall=0, set=sb0, wait=-}
        mbar.arrive [ur4]
        halt {pred=PT, stall=0, set=-, wait=sb0}
    )", {{i2_event_mbar, i21_mbar(1, 0, 64)}, {4160, i21_mbar(2, 2)}});
    const uint64_t s = nanosm::fetch_to_issue + 2 * alu_distance(profile) + 2;
    const uint64_t k = s + 1;
    const uint64_t w = k + rmw;
    require(s + t < w, "the sts would not reach its slot under the pause");
    i2_run_until(*machine, k);
    host_rmw_event(*machine);
    machine->run();
    const auto rmws = i3_rmws(*machine);
    require(issued_at(*machine, 4 * 8) == std::vector<uint64_t>{s}, "the sts did not issue at s:" + issue_log(*machine));
    require(rmws.size() == 2 && rmws[0].start == k && rmws[0].read == k && rmws[0].write == w,
        "the host event's RMW:" + describe(rmws));
    std::vector<uint64_t> stores;
    for (const auto& grant : i2_grants(*machine, "A-write")) {
        if (grant.client == "simt") stores.push_back(grant.cycle);
    }
    std::vector<uint64_t> expected;
    for (uint64_t round = 0; round < 16; ++round) expected.push_back(w + 1 + round);
    require(stores == expected, "the sts did not resume at the event's write grant + 1:" + describe(i2_grants(*machine)));
    require(issued_at(*machine, 5 * 8) == std::vector<uint64_t>{w + 16} &&
            refusals(*machine, 0, "ordered behind own access") ==
                std::vector<std::pair<uint64_t, uint64_t>>{{s + 1, w + 16 - (s + 1)}},
        "the arrive was not ordered behind its own sts until the sts's last grant (s + 1 .. w + 15, admitted w + 16)");
    require(rmws[1].detail == "arrive, address=4160" && rmws[1].start == w + t + 17 && rmws[1].read == w + t + 17 &&
            rmws[1].write == w + t + 17 + rmw, "the arrive's RMW did not start and read at w + t + 17:" + describe(rmws));
    require(machine->read_mbar_state(4160).remaining == 1, "the arrive was lost");
}

// The RMW engine's admission (implementer choice, I3): the SIMT RMW register
// and the engines' event registers alternate by class. The host keeps the
// TMA's event register full (zero-byte complete_tx events on 4168, which
// change nothing) while warp 0 issues four mbar.arrive on 4160; each waits in
// the SIMT RMW register while the engine works. Whenever both classes hold a
// request, the class that did not start last goes, so the starts alternate
// TMA, SIMT, TMA, SIMT, ... one busy period apart: an endless supply of
// engine events cannot starve the warp. (Before I3 a held engine event
// refused a SIMT RMW at issue: engine events always went first.)
// I6a re-derivation: an arrive reaches the register issue_to_dispatch after
// its issue, and one warp issues its next arrive only at the previous one's
// write grant, so one warp's arrive would miss every other engine start
// (the TMA event then went alone, untested). Two warps issue two arrives
// each: while one's arrive runs, the other's waits in the register. Warp
// 0's first arrive issues at fetch_to_issue + alu_distance (its mov issues
// first, at fetch_to_issue; the arrive needs its r4) and is in the register
// from its dispatch + 1, a = that + issue_to_dispatch + 1; the host starts
// feeding one busy period before (f = a - busy), so the first TMA event
// ends as that arrive is held, and from then on both classes always hold a
// request. URF: the mbar address is a umov's ur4, so the first arrive issues
// at fetch_to_issue + urf_distance.
void test_i3_rmw_admission_fairness() {
    const auto& profile = canonical_profile();
    const auto busy = rmw_busy_cycles(profile);
    auto machine = i21_machine(profile, R"(
        umov ur4, 4160
        mbar.arrive [ur4]
        mbar.arrive [ur4]
        halt
    )", {{4160, i21_mbar(5, 5)}, {4168, i21_mbar(1, 1)}}, 2);
    const uint64_t f =
        nanosm::fetch_to_issue + urf_distance(profile) + nanosm::issue_to_dispatch(profile) + 1 - busy;
    while (!machine->done()) {
        const auto cycle = machine->cycle();
        if (cycle >= f && cycle < f + 9 * busy && !machine->tma_mbar_event_held()) {
            machine->complete_mbar_tx(4168, 0, 0);
        }
        tick_capped(*machine);
    }
    const auto services = mbar_rmw_services(*machine);
    std::vector<RmwService> expected;
    for (uint64_t k = 0; k < 9; ++k) {
        expected.emplace_back(f + k * busy, k % 2 == 0 ? "complete_tx (tma), address=4168" : "arrive, address=4160");
    }
    std::string seen;
    for (const auto& [cycle, detail] : services) seen += " c" + std::to_string(cycle) + ": " + detail + ";";
    require(services.size() >= 9 && std::vector<RmwService>(services.begin(), services.begin() + 9) == expected,
        "SIMT RMWs and engine events did not alternate:" + seen);
    const auto state = machine->read_mbar_state(4160);
    require(state.remaining == 1 && !state.phase, "an arrive was lost");
}

// rfc12 section 8: the signed tx count, in both orders ((pending arrivals,
// tx); the host's complete_tx and warp 0's arrive.expect_tx):
//   completion first:  (1, 0) -complete_tx 128-> (1, -128) -arrive.expect_tx 128-> (0, 0): complete
//   expectation first: (1, 0) -arrive.expect_tx 128-> (0, 128) -complete_tx 128-> (0, 0): complete
// At c18 the mbar holds the intermediate count; at the end the phase has
// flipped (arrivals reloaded, tx 0) and the warp's try_wait returned true.
// I6a: the observation moves to c20. Expectation first, the arrive issues at
// alu_distance (6: it needs the mov's r4), starts at its dispatch + 1 (10),
// is written D + Lb + C later (16) and lands D after that, at the edge of
// c18: seen from c19. Completion first, the arrive (behind the stall of 20)
// issues at c20, so nothing of it has landed by then.
// I6b-2: the program starts at fetch_to_issue (c2), so all of it -- and the
// observation -- moves by that: c22.
// URF: the address is a umov's ur4, so expectation first the arrive issues
// urf_distance (4) after it, two cycles earlier than before (seen from c19);
// completion first it still issues behind the stall, at c22.
void test_i3_signed_tx_both_orders() {
    const auto& profile = canonical_profile();
    for (const bool completion_first : {true, false}) {
        const auto what = std::string(completion_first ? "completion first: " : "expectation first: ");
        auto machine = i21_machine(profile, std::string(completion_first ? R"(
            umov ur4, 4160 {pred=PT, stall=20, set=-, wait=-}
            mbar.arrive.expect_tx [ur4], 128
        )" : R"(
            umov ur4, 4160
            mbar.arrive.expect_tx [ur4], 128
            mov r9, 0 {pred=PT, stall=20, set=-, wait=-}
        )") + R"(
            mov r9, 0
        wait: mbar.try_wait p0, [ur4], 0
            bra.uni wait {pred=!p0, stall=0, set=-, wait=-}
            mov r9, 1
            halt
        )", {{4160, i21_mbar(1, 1)}});
        if (completion_first) machine->complete_mbar_tx(4160, 128, 0);
        i2_run_until(*machine, nanosm::fetch_to_issue + 20);
        const auto middle = machine->read_mbar_state(4160);
        require(!middle.phase && middle.remaining == (completion_first ? 1 : 0) &&
                middle.tx_bytes == (completion_first ? -128 : 128),
            what + "intermediate (arrivals, tx) = (" + std::to_string(middle.remaining) + ", " +
                std::to_string(middle.tx_bytes) + ")");
        if (!completion_first) machine->complete_mbar_tx(4160, 128, 0);
        machine->run();
        const auto end = machine->read_mbar_state(4160);
        require(end.phase && end.remaining == 1 && end.tx_bytes == 0 && machine->read_reg(0, 0, 9) == 1,
            what + "the phase did not complete at (0, 0)");
    }
}

// rfc12 section 8: the TMA reads no mbar at its start (I2.1: after the
// descriptor, a one-round A-read of the completion mbar, which also checked
// that enough bytes were expected). A 64-byte TMA load completes into an mbar
// that expects one arrival and NO bytes yet -- the old check would have
// refused it -- and its complete_tx runs ahead: (1, 0) -> (1, -64). The
// warp's arrive.expect_tx of 64, long after, completes the phase. A-read
// serves the TMA exactly one round (its descriptor's), and shmem counts three
// requests (the descriptor read and two sector placements), one fewer than
// before.
void test_i3_tma_start_reads_no_mbar() {
    const auto& profile = canonical_profile();
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.shmem_image, 8192);
    write_rank1_load(shmem, 5120, 0, 16, 4104);
    write_u64(shmem, 4104, nanosm::encode_mbar_state(i21_mbar(1, 1)));
    fill_pattern(image_bytes(launch.gmem_image, 64), 0, 16, 0);
    std::string program = tma_operands_asm(10, 2048, 11, 4104, 8) + "umov ur12, 4104\ntma.load.1d [ur10], [5120, ur8], [ur11]\n";
    for (int i = 0; i < 10; ++i) program += "mov r9, 0 {pred=PT, stall=31, set=-, wait=-}\n";
    program += R"(
        mbar.arrive.expect_tx [ur12], 64
    wait: mbar.try_wait p0, [ur12], 0
        bra.uni wait {pred=!p0, stall=0, set=-, wait=-}
        mov r9, 1
        halt
    )";
    nanosm::Machine machine(profile, hand_assemble(i1_asm(profile, program), profile), std::move(launch),
        nanosm::TraceOptions{.timeline = true, .debug = true});
    std::vector<int32_t> counts{0}; // every tx count the mbar held, in order
    while (!machine.done()) {
        tick_capped(machine);
        const auto tx = machine.read_mbar_state(4104).tx_bytes;
        if (tx != counts.back()) counts.push_back(tx);
    }
    require(counts == std::vector<int32_t>{0, -64, 0} && machine.read_mbar_state(4104).phase &&
            machine.read_reg(0, 0, 9) == 1, "the TMA's complete_tx did not run ahead of the expect_tx (tx 0, -64, 0)");
    const auto& summary = machine.trace().summary();
    require(summary.shmem_a_read_tma == 1 && summary.shmem_requests == 3,
        "the TMA read more than its descriptor on A-read (a_read_tma=" + std::to_string(summary.shmem_a_read_tma) +
            ", requests=" + std::to_string(summary.shmem_requests) + ")");
    for (uint32_t i = 0; i < 16; ++i) require(machine.read_shmem_u32(2048 + 4 * i) == pattern_word(i), "TMA data");
}

// rfc12 section 8: tx arithmetic is checked (an atomic add wraps; this does
// not). Machine PANICs, both sanitizer builds: an operand above 2^20 - 1; a
// count rising above 2^20 - 1 (two arrive.expect_tx of 2^19); a count
// falling below -(2^20 - 1) (two host complete_tx of 2^20 - 1: the first
// reaches the limit exactly, which is legal; the second leaves it).
void test_i3_tx_range_panics() {
    const auto& profile = canonical_profile();
    {
        auto machine = i21_machine(profile, R"(
            umov ur4, 4160
            umov ur2, 524288
            mbar.arrive.expect_tx [ur4], ur2
            mbar.arrive.expect_tx [ur4], ur2
            halt
        )", {{4160, i21_mbar(2, 2)}});
        expect_sim_panic(*machine, "transaction byte count overflow", "a tx count above 2^20 - 1");
    }
    {
        auto machine = i21_machine(profile, R"(
            umov ur4, 4160
            umov ur2, 1048576
            mbar.arrive.expect_tx [ur4], ur2
            halt
        )", {{4160, i21_mbar(2, 2)}});
        expect_sim_panic(*machine, "transaction byte count operand exceeds 2^20 - 1", "a tx operand of 2^20");
    }
    {
        // The warp stays alive past the first event: kernel end would PANIC on tx != 0.
        auto machine = i21_machine(profile, "mov r9, 0 {pred=PT, stall=31, set=-, wait=-}\nhalt\n",
            {{4160, i21_mbar(1, 1)}});
        machine->complete_mbar_tx(4160, 1048575, 0);
        while (machine->read_mbar_state(4160).tx_bytes == 0) tick_capped(*machine);
        require(machine->read_mbar_state(4160).tx_bytes == -1048575, "the count did not reach -(2^20 - 1)");
        while (machine->tma_mbar_event_held()) tick_capped(*machine);
        machine->complete_mbar_tx(4160, 1048575, 0);
        expect_sim_panic(*machine, "transaction byte count underflow", "a tx count below -(2^20 - 1)");
    }
}

// SANITIZER (rfc12 section 8): the TMA's start-time mbar read is gone; the
// sanitizer peeks the completion mbar's phase when tma.load is accepted and
// compares it when the complete_tx reaches the mbar ("wrong phase"). Two ways
// a TMA's bytes land in a later phase:
//   plain arrive: the mbar expects one arrival and no bytes; the TMA is
//   accepted in phase 0, then a PLAIN mbar.arrive completes phase 0 at once
//   (tx still 0) -- rfc12's unsafe case -- and the TMA's complete_tx reaches
//   phase 1;
//   reuse: one arrive.expect_tx of 64 bytes but two 64-byte TMAs, both
//   accepted in phase 0; the first completes the phase, the second's
//   complete_tx reaches phase 1.
void test_i3_sanitizer_wrong_phase_complete_tx() {
    const auto& profile = canonical_profile();
    const auto machine_for = [&](const std::string& program) {
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.shmem_image, 8192);
        write_rank1_load(shmem, 5120, 0, 16, 4104);
        write_u64(shmem, 4104, nanosm::encode_mbar_state(i21_mbar(1, 1)));
        fill_pattern(image_bytes(launch.gmem_image, 64), 0, 16, 0);
        return std::make_unique<nanosm::Machine>(profile, hand_assemble(i1_asm(profile, program), profile),
            std::move(launch));
    };
    // Buffers at 2048 (ur10) and 2112 (ur13), both completing into 4104 (ur11).
    const auto operands = tma_operands_asm(10, 2048, 11, 4104, 8) + "umov ur13, 2112\n";
    auto plain = machine_for(operands + R"(
        umov ur12, 4104
        tma.load.1d [ur10], [5120, ur8], [ur11]
        mbar.arrive [ur12]
        halt
    )");
    expect_sim_panic(*plain, "wrong phase (tma #0 was accepted in phase 0; the mbar is in phase 1)",
        "a TMA completing after a plain arrive ended its phase");
    auto reuse = machine_for(operands + R"(
        umov ur12, 4104
        mbar.arrive.expect_tx [ur12], 64
        tma.load.1d [ur10], [5120, ur8], [ur11]
        tma.load.1d [ur13], [5120, ur8], [ur11]
        halt
    )");
    expect_sim_panic(*reuse, "wrong phase (tma #1 was accepted in phase 0; the mbar is in phase 1)",
        "a second TMA completing into the reused mbar's next phase");
}

// SANITIZER (rfc12 section 8): a tx count nonzero at kernel end -- a missing
// complete_tx (tx 64 left) or a missing expect_tx (tx -32 left) -- would hang
// a waiter: PANIC at the end of the run.
void test_i3_sanitizer_nonzero_tx_at_kernel_end() {
    const auto& profile = canonical_profile();
    {
        auto machine = i21_machine(profile, "umov ur4, 4160\nmbar.arrive.expect_tx [ur4], 64\nhalt\n", {{4160, i21_mbar(2, 2)}});
        expect_sim_panic(*machine, "transaction count is nonzero at kernel end (tx=64", "a missing complete_tx");
    }
    {
        auto machine = i21_machine(profile, "halt\n", {{4160, i21_mbar(1, 1)}});
        machine->complete_mbar_tx(4160, 32, 0);
        expect_sim_panic(*machine, "transaction count is nonzero at kernel end (tx=-32", "a missing expect_tx");
    }
}

std::vector<I5Atomic> i5_atomics(const nanosm::Machine& machine) {
    std::vector<I5Atomic> atomics;
    for (const auto& event : machine.trace().events()) {
        if (event.name != "atomic RMW") continue;
        atomics.push_back({event.start_cycle, event.start_cycle + event.duration_cycles, {}, {}, event.detail});
    }
    std::sort(atomics.begin(), atomics.end(), [](const auto& a, const auto& b) { return a.start < b.start; });
    for (const auto& grant : i2_grants(machine)) {
        if (grant.client != "rmw") continue;
        for (auto& atomic : atomics) {
            if (grant.cycle < atomic.start || grant.cycle >= atomic.end) continue;
            (grant.service == "A-read" ? atomic.reads : atomic.writes).push_back(grant.cycle);
        }
    }
    return atomics;
}

// The one atomic a run made.
I5Atomic i5_atomic(const nanosm::Machine& machine, const std::string& what) {
    const auto atomics = i5_atomics(machine);
    require(atomics.size() == 1, what + ": expected one atomic, found " + std::to_string(atomics.size()));
    return atomics.front();
}

// rfc13 section 6: one read at g, the old word at the port output at
// a = g + D + Lb, the write granted at w = a + C, the slice ending at w + 1.
// `read` is the expected g.
void i5_require_timeline(const I5Atomic& atomic, const nanosm::DeviceProfile& profile, uint64_t read,
    const std::string& what) {
    const uint64_t w = read + profile.shmem_port_a_depth + profile.shmem_read_latency_cycles +
        profile.shmem_rmw_compute_cycles;
    require(atomic.reads == std::vector<uint64_t>{read} && atomic.writes == std::vector<uint64_t>{w} &&
            atomic.end == w + 1,
        what + ": not read c" + std::to_string(read) + ", write c" + std::to_string(w) + ":" + describe(atomic));
}

// The atomic ALU, written independently of the machine's: 32-bit wrap through
// 64-bit arithmetic.
uint32_t i5_expected(nanosm::Opcode opcode, uint32_t old_word, uint32_t operand) {
    switch (opcode) {
    case nanosm::Opcode::atom_add_uni: return static_cast<uint32_t>((uint64_t{old_word} + operand) & 0xffffffffull);
    case nanosm::Opcode::atom_and_uni: return old_word & operand;
    case nanosm::Opcode::atom_xor_uni: return old_word ^ operand;
    default: return operand; // exch
    }
}

// An I2-style machine (shmem words 0..2047 hold 0x100 + word; gmem words
// 0..1023 hold 5 * word + 2); `setup` may change the launch (its image's
// shmem bytes are launch.shmem_image[0].bytes; its declared mbars).
std::unique_ptr<nanosm::Machine> i5_machine(const nanosm::DeviceProfile& profile, const std::string& program,
    uint32_t warps = 1, const std::function<void(nanosm::Launch&)>& setup = {}, bool hand_assembly = true) {
    nanosm::Launch launch;
    launch.active_warps = warps;
    auto& shmem = image_bytes(launch.shmem_image, 8192);
    for (uint32_t word = 0; word < 2048; ++word) write_u32(shmem, 4 * word, 0x100 + word);
    auto& gmem = image_bytes(launch.gmem_image, 4096);
    for (uint32_t word = 0; word < 1024; ++word) write_u32(gmem, 4 * word, 5 * word + 2);
    if (setup) setup(launch);
    const auto kernel = hand_assembly ? hand_assemble(i1_asm(profile, program), profile)
                                      : compile(program, profile);
    return std::make_unique<nanosm::Machine>(profile, kernel, std::move(launch),
        nanosm::TraceOptions{.timeline = true, .debug = true});
}

const std::array<std::pair<nanosm::Opcode, const char*>, 4> i5_ops = {{
    {nanosm::Opcode::atom_add_uni, "atom.add.uni"},
    {nanosm::Opcode::atom_and_uni, "atom.and.uni"},
    {nanosm::Opcode::atom_xor_uni, "atom.xor.uni"},
    {nanosm::Opcode::atom_exch_uni, "atom.exch.uni"},
}};

// The ISA side: four opcodes 0x39..0x3c, payload dst[8] base[8] operand[8]
// simm[14]; qtxc and the hand assembler parse "atom.op.uni rD, [urA + imm],
// urB" (base and operand are uniform registers); the offset is a signed
// 14-bit field; an atomic returns its old value through a depbar (qtxc
// allocates one, a consumer of rD waits on it; the hand assembler requires
// one); latency 0 (an async RF writer); hex round trip. The per-lane spelling
// "atom.add" is gone, and so is the vector-register base.
void test_i5_atomic_encoding_and_assembly() {
    const auto& profile = canonical_profile();
    uint32_t code = 0x39;
    for (const auto& [opcode, name] : i5_ops) {
        for (const int32_t offset : {-8192, 8191, 0, -4}) {
            nanosm::Instruction inst;
            inst.opcode = opcode;
            inst.dst = 200;
            inst.src0 = 129;
            inst.src1 = 255;
            inst.imm = offset;
            inst.control.set_depbar = 3;
            const auto word = nanosm::encode(inst);
            const auto decoded = nanosm::decode(word);
            require(word >> 56 == code && decoded.opcode == opcode && decoded.dst == 200 && decoded.src0 == 129 &&
                    decoded.src1 == 255 && decoded.imm == offset && decoded.control.set_depbar == 3,
                std::string(name) + " did not round-trip with offset " + std::to_string(offset));
            require(nanosm::fixed_latency(decoded, profile.timing) == 0, std::string(name) + " has a fixed latency");
        }
        nanosm::Instruction text;
        text.opcode = opcode;
        text.dst = 7;
        text.src0 = 1;
        text.src1 = 5;
        text.imm = 64;
        require(nanosm::format_instruction(text, false) == std::string(name) + " r7, [ur1 + 64], ur5",
            "format: " + nanosm::format_instruction(text, false));
        text.imm = 8192;
        try {
            (void)nanosm::encode(text);
            throw std::runtime_error(std::string(name) + " encoded a 15-bit offset");
        } catch (const nanosm::SimError& error) {
            require(std::string(error.what()).find("signed 14-bit") != std::string::npos, error.what());
        }
        ++code;
    }
    const auto program = nanosm::compile_qtx(
        "umov ur1, 0\numov ur2, 5\natom.xor.uni r3, [ur1 - 64], ur2\nadd r4, r3, r3\nhalt\n", profile);
    const auto& atom = program.instructions.at(2);
    const auto& consumer = program.instructions.at(3);
    require(atom.opcode == nanosm::Opcode::atom_xor_uni && atom.imm == -64 && atom.control.set_depbar < 6 &&
            (consumer.control.wait_mask >> atom.control.set_depbar & 1) != 0,
        "qtxc did not give the atomic a depbar its rD consumer waits on");
    const auto kernel = nanosm::assemble_asm(nanosm::emit_asm(program), profile);
    require(nanosm::load_hex(nanosm::emit_hex(kernel)).words == kernel.words, "atom did not survive hex");
    const auto rejects = [&](const std::string& source, bool hand, const std::string& needle) {
        try {
            if (hand) (void)hand_assemble(i1_asm(profile, source), profile);
            else (void)nanosm::compile_qtx(source, profile);
        } catch (const nanosm::SimError& error) {
            require(std::string(error.what()).find(needle) != std::string::npos,
                "'" + source + "': wrong error: " + error.what());
            return;
        }
        throw std::runtime_error("'" + source + "' was accepted");
    };
    rejects("atom.add.uni r3, [ur1 + 8192], ur2\nhalt\n", false, "atom offset must fit a signed 14-bit immediate");
    rejects("atom.add.uni r3, [ur1 - 8193], ur2\nhalt\n", false, "atom offset must fit a signed 14-bit immediate");
    rejects("atom.add.uni r3, [ur1]\nhalt\n", false, "expected 3 operands");
    rejects("atom.min.uni r3, [ur1], ur2\nhalt\n", false, "unknown opcode 'atom.min.uni'");
    rejects("atom.add r3, [r1], r2\nhalt\n", false, "unknown opcode 'atom.add'");
    rejects("atom.add.uni r3, [r1], r2\nhalt\n", false, "expected uniform register (ur0..ur15), got 'r1'");
    rejects("atom.exch.uni r3, [ur1], ur2\nhalt\n", true, "memory operation requires set=sb0..sb5");
}

// rfc13 sections 1 and 6: ONE word per warp instruction, the update applied
// ONCE (the operand is ur5 = 0x5a5: add adds it once, not per lane), every
// lane's rD gets the old word. One read, granted when the engine starts it
// (the cycle after the atomic's issue: nothing else holds A-read); its write
// D + Lb + C later; the slice ends at that write + 1. The old word reaches rD
// through the return slot at a + 1 (a = the read + D + Lb), BEFORE the write
// grant a + C; the warp's next shmem access (an lds.uni of the word) is
// admitted only at the write grant and reads the new word. Each op at the
// canonical D = Lb = C = 2 (7 cycles from start to the end of the pause);
// add also at D = 3, C = 3.
// I6a: the atomic reaches the SIMT RMW register at its dispatch, so it
// starts issue_to_dispatch + 1 after its issue; rD's RF write is granted at
// a + 1 (still before the word's write grant) and lands Wd later; the
// lds.uni, admitted at the write grant w, is granted at its dispatch + 1,
// w + issue_to_dispatch + 1.
void test_i5_one_word_timeline() {
    const std::string program_tail = R"( r7, [ur1 + 2048], ur5 {pred=PT, stall=0, set=sb0, wait=-}
        lds.uni r8, [ur1 + 2048] {pred=PT, stall=0, set=sb1, wait=-}
        halt {pred=PT, stall=0, set=-, wait=sb0|sb1}
    )";
    const std::string program_head = R"(
        umov ur1, 0
        umov ur5, 0x5a5
        )";
    struct Case {
        nanosm::Opcode opcode;
        const char* name;
        uint32_t d;
        uint32_t c;
    };
    std::vector<Case> cases;
    for (const auto& [opcode, name] : i5_ops) cases.push_back({opcode, name, 2, 2});
    cases.push_back({nanosm::Opcode::atom_add_uni, "atom.add.uni", 3, 3});
    for (const auto& test : cases) {
        auto profile = canonical_profile();
        profile.shmem_port_a_depth = test.d;
        profile.shmem_rmw_compute_cycles = test.c;
        const auto what = std::string(test.name) + " D=" + std::to_string(test.d) + " C=" + std::to_string(test.c);
        auto machine = i5_machine(profile, program_head + test.name + program_tail);
        const auto written = i1_run(*machine, {7});
        const auto atomic = i5_atomic(*machine, what);
        const auto t = nanosm::issue_to_dispatch(profile);
        const auto issue = issued_at(*machine, 16).at(0);
        require(atomic.start == issue + t + 1, what + ": did not start the cycle after its dispatch");
        i5_require_timeline(atomic, profile, atomic.start, what);
        const auto a = atomic.reads.at(0) + profile.shmem_port_a_depth + profile.shmem_read_latency_cycles;
        const auto w = atomic.writes.at(0);
        require(written.at(7) == std::vector<uint64_t>{a + 1 + profile.rf_write_depth_cycles} && a + 1 < w,
            what + ": rD's RF write was not granted at a + 1 (c" + std::to_string(a + 1) + "), before the write grant c" +
                std::to_string(w) + ", landing Wd later");
        require(issued_at(*machine, 24) == std::vector<uint64_t>{w}, what + ": the lds.uni was not admitted at the write grant");
        const auto lds = i2_grants(*machine, "A-read");
        require(lds.size() == 2 && lds[1] == I2Grant{w + t + 1, "A-read", "simt", lds[1].slot},
            what + ": the lds.uni's read was not granted at w + issue_to_dispatch + 1:" + describe(lds));
        const uint32_t updated = i5_expected(test.opcode, 0x300, 0x5a5);
        require(machine->read_shmem_u32(2048) == updated, what + ": the word was not updated once");
        for (uint32_t lane = 0; lane < 16; ++lane) {
            require(machine->read_reg(0, lane, 7) == 0x300 && machine->read_reg(0, lane, 8) == updated,
                what + ": lane " + std::to_string(lane) + " rD or the lds.uni");
        }
        require(atomic.detail == std::string(test.name) + ", warp=0", what + ": slice detail " + atomic.detail);
        if (test.d == 2 && test.c == 2) require(atomic.end - atomic.start == 7, what + ": not 7 cycles canonical");
    }
}

// int32 add WRAPS (rfc13 section 1: defined 32-bit two's-complement wrap,
// unlike the checked mbar tx count): one atomic each,
//   0x7fffffff + 1 = 0x80000000      0x80000000 + (-1) = 0x7fffffff
//   0xffffffff + 1 = 0               0x80000000 + 0x80000000 = 0
//   0x7fffffff + 0x7fffffff = 0xfffffffe
// and a chain across the boundary on one word: 0x7ffffffe + 1 + 1 + 1 + 1,
// each atomic reading the previous sum. Operands come from a table: lds.uni
// into vector registers, then r2ur into the uniform operand registers
// ur2..ur6 (the first r2ur waits for the loads).
void test_i5_add_wraps_at_the_int32_boundary() {
    const auto& profile = canonical_profile();
    const std::array<std::pair<uint32_t, uint32_t>, 5> cases = {{
        {0x7fffffffu, 1u}, {0x80000000u, 0xffffffffu}, {0xffffffffu, 1u}, {0x80000000u, 0x80000000u},
        {0x7fffffffu, 0x7fffffffu},
    }};
    auto machine = i5_machine(profile, R"(
        umov ur1, 0
        umov ur7, 1
        lds.uni r12, [ur1 + 7232] {pred=PT, stall=0, set=sb1, wait=-}
        lds.uni r14, [ur1 + 7236] {pred=PT, stall=0, set=sb1, wait=-}
        lds.uni r15, [ur1 + 7240] {pred=PT, stall=0, set=sb1, wait=-}
        lds.uni r17, [ur1 + 7244] {pred=PT, stall=0, set=sb1, wait=-}
        lds.uni r18, [ur1 + 7248] {pred=PT, stall=0, set=sb1, wait=-}
        r2ur ur2, r12 {pred=PT, stall=0, set=-, wait=sb1}
        r2ur ur3, r14
        r2ur ur4, r15
        r2ur ur5, r17
        r2ur ur6, r18
        atom.add.uni r7, [ur1 + 2048], ur2 {pred=PT, stall=0, set=sb0, wait=-}
        atom.add.uni r8, [ur1 + 2052], ur3 {pred=PT, stall=0, set=sb0, wait=-}
        atom.add.uni r9, [ur1 + 2056], ur4 {pred=PT, stall=0, set=sb0, wait=-}
        atom.add.uni r10, [ur1 + 2060], ur5 {pred=PT, stall=0, set=sb0, wait=-}
        atom.add.uni r11, [ur1 + 2064], ur6 {pred=PT, stall=0, set=sb0, wait=-}
        atom.add.uni r21, [ur1 + 2092], ur7 {pred=PT, stall=0, set=sb2, wait=-}
        atom.add.uni r22, [ur1 + 2092], ur7 {pred=PT, stall=0, set=sb2, wait=-}
        atom.add.uni r23, [ur1 + 2092], ur7 {pred=PT, stall=0, set=sb2, wait=-}
        atom.add.uni r24, [ur1 + 2092], ur7 {pred=PT, stall=0, set=sb2, wait=-}
        halt {pred=PT, stall=0, set=-, wait=sb0|sb2}
    )", 1, [&](nanosm::Launch& launch) {
        auto& shmem = launch.shmem_image.at(0).bytes;
        for (uint32_t k = 0; k < 5; ++k) {
            write_u32(shmem, 2048 + 4 * k, cases[k].first);
            write_u32(shmem, 7232 + 4 * k, cases[k].second);
        }
        write_u32(shmem, 2092, 0x7ffffffeu);
    });
    machine->run();
    for (uint32_t k = 0; k < 5; ++k) {
        const auto [old_word, operand] = cases[k];
        require(machine->read_shmem_u32(2048 + 4 * k) == static_cast<uint32_t>(uint64_t{old_word} + operand) &&
                machine->read_reg(0, 5, 7 + k) == old_word,
            "case " + std::to_string(k) + ": " + std::to_string(old_word) + " + " + std::to_string(operand));
    }
    for (uint32_t k = 0; k < 4; ++k) {
        require(machine->read_reg(0, 9, 21 + k) == 0x7ffffffeu + k, "chain: atomic " + std::to_string(k));
    }
    require(machine->read_shmem_u32(2092) == 0x80000002u, "chain: the sum did not wrap to 0x80000002");
}

// rfc13 section 4: an atomic reserves its RETURN load slot at admission; with
// none free it is not admitted (structural, counted as a load-slot wait), so
// it never takes the store pause with nowhere to put its old value. One load
// slot: the lds holds it from its grant G until its RF write at G + D + Lb +
// 1; the atomic (not ordered: the lds is serviced at its grant) is refused
// "load slots full" G .. G + D + Lb and admitted in the cycle the lds's RF
// write frees the slot (same-cycle reuse), then runs.
// I6a: the lds issues at l (it needs the shli's r1: 2 alu_distance) and is
// granted at G = l + issue_to_dispatch + 1; the atomic, first tried at l + 2
// (it needs the second mov's r3), is ordered behind the unserviced lds
// until G, then refused for its return slot exactly as before. (I6b-2: the
// sread issues at fetch_to_issue.) URF: the atomic's ur3 (umov, l - 4) is
// visible at l, so it is first tried at l + 1, the cycle after the lds, and
// ordered behind it l + 1 .. G - 1.
void test_i5_return_slot_admission() {
    auto profile = canonical_profile();
    profile.shmem_load_slots = 1;
    const uint64_t back = profile.shmem_port_a_depth + profile.shmem_read_latency_cycles;
    auto machine = i5_machine(profile, R"(
        sread r0, LANE_ID
        shli r1, r0, 2
        umov ur2, 0
        umov ur3, 5
        lds r6, [r1 + 256] {pred=PT, stall=0, set=sb1, wait=-}
        atom.add.uni r7, [ur2 + 1024], ur3 {pred=PT, stall=0, set=sb0, wait=-}
        halt {pred=PT, stall=0, set=-, wait=sb0|sb1}
    )");
    machine->run();
    const auto g = i2_grants(*machine, "A-read").at(0);
    const auto t = nanosm::issue_to_dispatch(profile);
    const auto l = nanosm::fetch_to_issue + 2 * alu_distance(profile);
    require(g.client == "simt" && issued_at(*machine, 32) == std::vector<uint64_t>{l} && g.cycle == l + t + 1,
        "the lds was not granted the cycle after its dispatch:" + issue_log(*machine));
    require(refusals(*machine, 0, "ordered behind own access") == std::vector<std::pair<uint64_t, uint64_t>>{{l + 1, t}},
        "the atomic was not ordered behind the lds until its grant:" +
            describe(refusals(*machine, 0, "ordered behind own access")));
    require(refusals(*machine, 0, "load slots full") == std::vector<std::pair<uint64_t, uint64_t>>{{g.cycle, back + 1}} &&
            issued_at(*machine, 40) == std::vector<uint64_t>{g.cycle + back + 1},
        "the atomic was not refused for its return slot c" + std::to_string(g.cycle) + " .. c" +
            std::to_string(g.cycle + back) + " and admitted at c" + std::to_string(g.cycle + back + 1) + ":" +
            describe(refusals(*machine, 0, "load slots full")));
    require(machine->trace().summary().shmem_load_slot_waits == back + 1, "the return-slot refusals were not counted");
    const auto atomic = i5_atomic(*machine, "after the refusal");
    i5_require_timeline(atomic, profile, atomic.start, "after the refusal");
    require(machine->read_shmem_u32(1024) == 0x205, "the atomic's word");
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(machine->read_reg(0, lane, 6) == 0x100 + 64 + lane && machine->read_reg(0, lane, 7) == 0x200,
            "lane " + std::to_string(lane) + ": the lds or the atomic returned the wrong word");
    }
}

// rfc13 section 4: the SIMT RMW register holds ONE operation machine-wide.
// Three warps, one atomic each on one word (operand 1): the first runs in the
// engine, the second waits in the register, the third is refused "SIMT RMW
// register busy" until the register empties. All three apply, one after
// another (each reads the previous sum), and the engine runs them back to
// back with no overlap.
void test_i5_simt_rmw_register_busy() {
    auto profile = canonical_profile();
    profile.shmem_load_slots = 3; // a return slot for each: the third waits only for the register
    auto machine = i5_machine(profile, R"(
        umov ur1, 0
        umov ur5, 1
        atom.add.uni r7, [ur1 + 2048], ur5 {pred=PT, stall=0, set=sb0, wait=-}
        halt {pred=PT, stall=0, set=-, wait=sb0}
    )", 3);
    machine->run();
    const auto atomics = i5_atomics(*machine);
    require(atomics.size() == 3, "expected three atomics");
    for (std::size_t k = 1; k < atomics.size(); ++k) {
        require(atomics[k].start >= atomics[k - 1].end, "two atomics overlapped in the engine:" + describe(atomics[k]));
    }
    uint32_t busy = 0;
    for (uint32_t warp = 0; warp < 3; ++warp) busy += static_cast<uint32_t>(refusals(*machine, warp, "SIMT RMW register busy").size());
    require(busy != 0, "no warp was refused for the busy SIMT RMW register");
    std::set<uint32_t> old_values;
    for (uint32_t warp = 0; warp < 3; ++warp) old_values.insert(machine->read_reg(warp, 0, 7));
    require(old_values == std::set<uint32_t>{0x300, 0x301, 0x302} && machine->read_shmem_u32(2048) == 0x303,
        "the three atomics did not apply one after another");
}

// rfc13 section 6 with rfc12 section 6: the global store pause covers the
// atomic from its start to its write grant w; stores start again at w + 1;
// reads continue. Warp 0: one atomic. Warp 1: an lds (granted while the
// atomic runs) and an sts on other words, admitted before w and granted
// exactly at w + 1; store_paused counts its wait.
// (Mutation: the pause ends at the atomic's read grant -> the sts goes in
// before w + 1 and this test fails.)
// I6a: every access now reaches its slot or register issue_to_dispatch after
// its issue. The stall on warp 0's mov r5 (11) re-places the atomic: it
// issues at c26 and starts at c30 (w = c36); warp 1's lds (issued c27) is
// granted at c31, during it; its sts, ordered behind the lds, issues at c31
// and is in its slot from c35, paused until w.
// Two-word packets: warp 0's mov r1 (c14) is the second half of its packet,
// so warp 0 asks for a fetch at c15, the cycle warp 1 first asks for its
// branch target's packet; the round-robin serves warp 0 first and warp 1's
// path runs a cycle later (its sread at c18, its lds at c30, granted c34;
// its sts at c34, in its slot from c38). The stall is 14 (was 11): the
// atomic issues at c29 and starts at c33 (w = c39), so the lds is granted
// during it and the sts waits in its slot c38 .. c39, as before.
// URF: the role dispatch is usread + ur2r (the branch issues at c16), and
// the packets fall differently. Warp 0's umov ur5 issues at c19 and, behind
// the same stall of 14, the atomic at c33, starting at c37 (w = c43); warp
// 1's sread issues at c21, its lds at c34 (granted c38, during the atomic),
// its sts at c38, waiting in its slot c42 .. c43.
void test_i5_store_pause_and_resume() {
    const auto& profile = canonical_profile();
    auto machine = i5_machine(profile, R"(
        usread ur9, WARP_ID
        ur2r r29, ur9
        setp.ne p6, r29, 0
        bra.uni other {pred=p6, stall=0, set=-, wait=-}
        umov ur1, 0
        umov ur5, 1 {pred=PT, stall=14, set=-, wait=-}
        atom.add.uni r7, [ur1 + 2048], ur5 {pred=PT, stall=0, set=sb0, wait=-}
        halt {pred=PT, stall=0, set=-, wait=sb0}
    other: sread r20, LANE_ID
        shli r20, r20, 2
        mov r21, 0x77
        lds r23, [r20 + 256] {pred=PT, stall=0, set=sb1, wait=-}
        sts [r20 + 3072], r21 {pred=PT, stall=0, set=sb2, wait=-}
        halt {pred=PT, stall=0, set=-, wait=sb1|sb2}
    )", 2);
    machine->run();
    const auto atomic = i5_atomic(*machine, "store pause");
    i5_require_timeline(atomic, profile, atomic.start, "store pause");
    std::vector<uint64_t> loads;
    std::vector<uint64_t> stores;
    for (const auto& grant : i2_grants(*machine)) {
        if (grant.client != "simt") continue;
        (grant.service == "A-read" ? loads : stores).push_back(grant.cycle);
    }
    const auto w = atomic.writes.at(0);
    const auto sts_issue = issued_at(*machine, 12 * 8);
    require(loads.size() == 1 && loads[0] > atomic.start && loads[0] < w,
        "warp 1's lds was not granted while the atomic ran (reads continue):" + (describe(i2_grants(*machine)) + " | issues" + issue_log(*machine)) +
            describe(atomic));
    require(sts_issue.size() == 1 && sts_issue[0] < w, "warp 1's sts was not admitted before the atomic's write");
    require(stores == std::vector<uint64_t>{w + 1},
        "the sts was not granted at the atomic's write + 1 (c" + std::to_string(w + 1) + "):" +
            describe(i2_grants(*machine, "A-write")) + describe(atomic));
    require(machine->trace().summary().shmem_store_paused == w - (sts_issue[0] + nanosm::issue_to_dispatch(profile)),
        "store_paused should count the sts's wait in its slot, its dispatch + 1 .. the atomic's write");
    require(machine->read_shmem_u32(2048) == 0x301, "the atomic's word");
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(machine->read_reg(0, lane, 7) == 0x300 && machine->read_reg(1, lane, 23) == 0x100 + 64 + lane &&
                machine->read_shmem_u32(3072 + 4 * lane) == 0x77,
            "lane " + std::to_string(lane) + ": wrong atomic, lds or sts data");
    }
}

// rfc12 section 6 step 2: an atomic that STARTS while another warp's store
// transaction is under way reads strictly after that store's last round
// (matched D then lands the store before the atomic's sample). Warp 1: one
// sts of sixteen rounds (every lane on bank 0); its LAST round writes the
// atomic's word. Warp 0's atomic starts during those rounds; its read is
// granted the cycle after the last round, and it reads the stored word.
// (Mutation: the read granted while the store is under way -> it is early
// and returns the old word.)
// I6a: warp 1's sts now reaches its slot issue_to_dispatch after its issue
// (rounds c32..c47) and the atomic starts issue_to_dispatch + 1 after its
// own; the stall on the last filler mov (14) re-places the atomic's start
// inside those rounds. URF: with usread + ur2r for the role dispatch the
// rounds are c38..c53 and the atomic starts at c41, still inside them.
void test_i5_read_after_the_store_under_way() {
    const auto& profile = canonical_profile();
    auto machine = i5_machine(profile, R"(
        usread ur9, WARP_ID
        ur2r r29, ur9
        setp.ne p6, r29, 0
        bra.uni other {pred=p6, stall=0, set=-, wait=-}
        umov ur1, 0
        umov ur5, 1
        mov r6, 0
        mov r6, 1
        mov r6, 2 {pred=PT, stall=14, set=-, wait=-}
        atom.add.uni r7, [ur1 + 960], ur5 {pred=PT, stall=0, set=sb0, wait=-}
        halt {pred=PT, stall=0, set=-, wait=sb0}
    other: sread r20, LANE_ID
        shli r20, r20, 6
        mov r21, 0x77
        sts [r20], r21 {pred=PT, stall=0, set=sb2, wait=-}
        halt {pred=PT, stall=0, set=-, wait=sb2}
    )", 2);
    machine->run();
    const auto atomic = i5_atomic(*machine, "read after the store");
    std::vector<uint64_t> rounds;
    for (const auto& grant : i2_grants(*machine, "A-write")) {
        if (grant.client == "simt") rounds.push_back(grant.cycle);
    }
    require(rounds.size() == 16, "the sts was not sixteen rounds");
    require(atomic.start > rounds.front() && atomic.start <= rounds.back(),
        "the atomic did not start while the sts was under way (the test needs that):" + describe(atomic) +
            " sts rounds c" + std::to_string(rounds.front()) + "..c" + std::to_string(rounds.back()));
    i5_require_timeline(atomic, profile, rounds.back() + 1, "read after the store");
    require(machine->read_reg(0, 0, 7) == 0x77 && machine->read_shmem_u32(960) == 0x78,
        "the atomic did not read the store's word (rD " + std::to_string(machine->read_reg(0, 0, 7)) + ")");
}

// rfc13 section 4 with rfc12 section 4: an atomic is a WRITE for its warp's
// order and counts as unserviced until its write grant.
//   sts B; atom B: the atomic waits for the sts's grant and reads its word;
//   atom B; lds B: the lds waits until the atomic's write grant w (admitted
//     at w) and reads the result in lane 0's word;
//   lds A; atom A: the atomic waits for the lds's grant; the lds reads the
//     old word (and so does the atomic: it applies after);
//   atom A; fence.sh: admitted at the atomic's write grant + D.
// Words B: 384 + l (sixteen banks, one round each); A: 256 + l. Three load
// slots, so atom A's return slot is free while both lds hold theirs: only
// the warp's order decides.
void test_i5_same_warp_order() {
    auto profile = canonical_profile();
    profile.shmem_load_slots = 3;
    auto machine = i5_machine(profile, R"(
        sread r0, LANE_ID
        shli r1, r0, 2
        umov ur2, 0
        umov ur3, 3
        mov r12, 0x1010
        sts [r1 + 1536], r12 {pred=PT, stall=0, set=sb1, wait=-}
        atom.add.uni r11, [ur2 + 1536], ur3 {pred=PT, stall=0, set=sb2, wait=-}
        lds r8, [r1 + 1536] {pred=PT, stall=0, set=sb3, wait=-}
        lds r6, [r1 + 1024] {pred=PT, stall=0, set=sb4, wait=-}
        atom.add.uni r7, [ur2 + 1024], ur3 {pred=PT, stall=0, set=sb0, wait=-}
        fence.sh
        halt {pred=PT, stall=0, set=-, wait=sb0|sb1|sb2|sb3|sb4}
    )");
    machine->run();
    const auto atomics = i5_atomics(*machine);
    require(atomics.size() == 2, "expected two atomics");
    std::vector<I2Grant> simt;
    for (const auto& grant : i2_grants(*machine)) {
        if (grant.client == "simt") simt.push_back(grant);
    }
    // sts B, lds B, lds A, in that order.
    require(simt.size() == 3 && simt[0].service == "A-write" && simt[1].service == "A-read" &&
            simt[2].service == "A-read",
        "the SIMT rounds:" + describe(simt));
    require(issued_at(*machine, 48) == std::vector<uint64_t>{simt[0].cycle} && atomics[0].reads.at(0) > simt[0].cycle,
        "atom B was not admitted at the sts's grant, reading after it");
    const auto w_b = atomics[0].writes.at(0);
    require(issued_at(*machine, 56) == std::vector<uint64_t>{w_b} && simt[1].cycle > w_b,
        "lds B was not admitted at atom B's write grant c" + std::to_string(w_b) + ":" + describe(atomics[0]));
    require(issued_at(*machine, 72) == std::vector<uint64_t>{simt[2].cycle} && atomics[1].reads.at(0) > simt[2].cycle,
        "atom A was not admitted at lds A's grant");
    require(issued_at(*machine, 80) == std::vector<uint64_t>{atomics[1].writes.at(0) + profile.shmem_port_a_depth},
        "fence.sh was not admitted at atom A's write grant + D");
    require(machine->read_shmem_u32(1536) == 0x1013 && machine->read_shmem_u32(1024) == 0x203,
        "the atomics' words");
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(machine->read_reg(0, lane, 11) == 0x1010 && machine->read_reg(0, lane, 7) == 0x200,
            "lane " + std::to_string(lane) + ": an atomic's old value");
        require(machine->read_reg(0, lane, 8) == (lane == 0 ? 0x1013u : 0x1010u) &&
                machine->read_reg(0, lane, 6) == 0x200 + lane,
            "lane " + std::to_string(lane) + ": an lds around the atomics");
    }
}

// Backlog D12 (HUMAN 2026-10-12): a uniform operation takes its uniform
// operands whatever the active mask -- the mask gates only its destination's
// lane write enables -- and a partial (or empty) mask is UB: the SANITIZER
// PANICs; the machine runs quietly, and what it does then is no contract
// (rfc13 section 2: the asserts below pin today's behavior so that it stays
// deterministic, nothing more). Lane 0 is inactive here (0xfffe):
//   atom.add.uni [ur1 + 2048] += ur5 (1): word 512 becomes 0x301; rD lanes
//     1..15 get 0x300, lane 0 keeps 0xdead;
//   lds.uni r3 from the same address: lanes 1..15 get 0x301;
//   mbar.arrive and mbar.test_wait on a count-1 mbar: the phase completes.
// An EMPTY mask still performs the atomic (word 512 = 0x301) and writes no
// lane of rD. (Before the URF these operands were vector registers read at
// lane 0, l-dependent in the other lanes; a uniform register has no lanes.)
void test_d12_uniform_ops_under_a_partial_mask() {
    const auto& profile = canonical_profile();
    const std::string program = R"(
        umov ur1, 0
        umov ur5, 1
        mov r7, 0xdead
        mov r3, 0xbeef
        umov ur4, 4160
        atom.add.uni r7, [ur1 + 2048], ur5 {pred=PT, stall=0, set=sb0, wait=-}
        lds.uni r3, [ur1 + 2048] {pred=PT, stall=0, set=sb1, wait=sb0}
        mbar.arrive [ur4]
        mbar.test_wait p0, [ur4], 0
        halt {pred=PT, stall=0, set=-, wait=sb0|sb1}
    )";
    const auto mbar = [](nanosm::Launch& launch) { write_count_mbar(launch.shmem_image.at(0).bytes, 4160, 1); };
    for (const uint64_t mask : {uint64_t{0xfffe}, uint64_t{0}}) {
        const auto what = "mask " + std::to_string(mask);
        auto machine = i5_machine(profile, program, 1, mbar);
        machine->test_set_active_mask(0, mask);
#if NANOSM_SANITIZER
        const auto message = i4_panic(*machine, what);
        require_mentions(message, {"atom.add.uni is a uniform operation under a partial active mask"}, what);
#else
        while (!machine->done()) tick_capped(*machine);
        require(machine->read_shmem_u32(2048) == 0x301, what + ": the atomic did not apply its uniform operand");
        require(machine->read_mbar_state(4160).phase, what + ": the arrive did not complete the phase");
        for (uint32_t lane = 0; lane < 16; ++lane) {
            const bool active = (mask >> lane & 1) != 0;
            require(machine->read_reg(0, lane, 7) == (active ? 0x300u : 0xdeadu) &&
                    machine->read_reg(0, lane, 3) == (active ? 0x301u : 0xbeefu),
                what + ": lane " + std::to_string(lane) + " rD (the mask gates only the write enables)");
        }
#endif
    }
#if NANOSM_SANITIZER
    for (const std::string uniform :
             {"lds.uni r3, [ur1] {pred=PT, stall=0, set=sb0, wait=-}", "mbar.arrive [ur2]", "mbar.test_wait p0, [ur2], 0"}) {
        auto machine = i5_machine(profile, "umov ur1, 0\numov ur2, 4160\n" + uniform + "\nhalt\n", 1, mbar);
        machine->test_set_active_mask(0, 0xfffe);
        require_mentions(i4_panic(*machine, uniform), {"uniform operation under a partial active mask"}, uniform);
    }
#endif
}

// Backlog D12 (HUMAN 2026-10-12, as on NVIDIA): two lanes of one sts naming
// one word is UB -- the SANITIZER PANICs at issue; the machine just writes
// every lane, in round order (each lane on the word is its own bank round),
// so the highest ACTIVE lane's word lands. Lane l stores l + 1 to word 0.
void test_d12_sts_two_lanes_on_one_word() {
    for (const uint64_t mask : {uint64_t{0xffff}, uint64_t{0x00f6}}) {
        nanosm::Launch launch;
        launch.active_warps = 1;
        image_bytes(launch.shmem_image, 4);
        nanosm::Machine machine(canonical_profile(), compile(R"(
            sread r1, LANE_ID
            addi r2, r1, 1
            mov r0, 0
            sts [r0], r2
            halt
        )"), std::move(launch));
        machine.test_set_active_mask(0, mask);
#if NANOSM_SANITIZER
        require_mentions(i4_panic(machine, "same-word sts"), {"multiple lanes write the same shmem word"},
            "same-word sts");
#else
        while (!machine.done()) tick_capped(machine);
        const uint32_t highest = mask == 0xffff ? 15 : 7;
        require(machine.read_shmem_u32(0) == highest + 1,
            "the highest active lane's word did not land (" + std::to_string(machine.read_shmem_u32(0)) + ")");
#endif
    }
}

// Backlog D12, gmem: two lanes of one stg naming one word is UB -- the
// SANITIZER PANICs at issue, before anything is written; the machine sends
// every lane's write, one request per lane in lane order, and with the
// placeholder controller's queue-front order the highest lane's lands last.
void test_d12_stg_two_lanes_on_one_word() {
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& gmem = image_bytes(launch.gmem_image, 4);
    write_u32(gmem, 0, 0x12345678);
    nanosm::Machine machine(canonical_profile(), compile(R"(
        mov   r0, 0
        sread r1, LANE_ID
        stg   [r0], r1
        halt
    )"), std::move(launch));
#if NANOSM_SANITIZER
    require_mentions(i4_panic(machine, "same-word stg"), {"multiple lanes write the same gmem word"}, "same-word stg");
    require(machine.read_gmem_u32(0) == 0x12345678, "the PANIC came after a write");
#else
    while (!machine.done()) tick_capped(machine);
    require(machine.read_gmem_u32(0) == 15, "lane 15's word did not land last (" +
        std::to_string(machine.read_gmem_u32(0)) + ")");
#endif
}

// Backlog D12: an sts over an mbar with a live waiter is UB (as on NVIDIA:
// an mbarrier's memory is used for nothing else until mbarrier.inval). The
// machine no longer checks it: the sts just writes the word. The SANITIZER's
// C10 PANICs (the mbar is declared by the launch, so it is live from launch).
// Warp 0 waits on the mbar (one try_wait, then halt); warp 1's sts writes 5
// into words 16 + l (lanes 0 and 1: the mbar's two words). Without the
// sanitizer the waiter's try_wait simply times out (no wake); the run is
// capped.
void test_d12_sts_over_an_mbar_with_a_live_waiter() {
    nanosm::Launch launch;
    launch.active_warps = 2;
    auto& shmem = image_bytes(launch.shmem_image, 128);
    write_count_mbar(shmem, 64, 1);
    launch.mbars = {64};
    nanosm::Machine machine(canonical_profile(), compile(R"(
        mov r1, 64
        umov ur1, 64
        usread ur0, WARP_ID
        ur2r r0, ur0
        setp.eq p0, r0, 0
        @p0 bra.uni waiter
        sread r3, LANE_ID
        shli r3, r3, 2
        add r3, r3, r1
        mov r2, 5
        sts [r3], r2
        halt
    waiter:
        mbar.try_wait p1, [ur1], 0
        halt
    )"), std::move(launch));
#if NANOSM_SANITIZER
    require_mentions(i4_panic(machine, "sts over a live mbar"), {"reason=live mbar opacity: sts writes bytes [64, 72) "
        "of the live mbar [64, 72) (declared by the launch image)"}, "sts over a live mbar");
#else
    while (!machine.done()) tick_capped(machine);
    require(machine.read_shmem_u32(64) == 5 && machine.read_shmem_u32(68) == 5, "the sts did not write the mbar's words");
#endif
}

// Backlog D12.1 (HUMAN 2026-10-12): EVERY uniform operation -- each
// instruction with a uniform-register or lane-0-sourced operand, and bra.uni
// in every form -- is UB under a partial (or empty) active mask: the
// SANITIZER PANICs naming it; the machine runs quietly (what it does then is
// no contract; a program with a partial mask uses the per-lane instruction).
// One program per uniform opcode family, each of which runs to completion
// under a full mask, run under 0xfffe (lane 0 inactive): the named opcode is
// the program's first uniform operation (the umov / usread that set up its
// uniform registers are uniform arithmetic, which the mask check does not
// cover).
void test_d12_1_every_uniform_op_under_a_partial_mask() {
    const auto& profile = canonical_profile();
    const auto setup = [](nanosm::Launch& launch) {
        auto& shmem = image_bytes(launch.shmem_image, 8192);
        write_count_mbar(shmem, 4096, 1);
        write_tma_descriptor(shmem, 5120, linear_tma(64));
        write_tx_mbar(shmem, 4104, 64);
        image_bytes(launch.gmem_image, 4096);
    };
    const auto tma_operands = i4_tma_operands(1024);
    const std::string mma_tail = "umov ur4, 4160\nmbar.init [ur4], 1\nmma.commit [ur4]\n"
                                 "wait:\nmbar.try_wait p0, [ur4], 0\n@!p0 bra.uni wait\nhalt\n";
    const std::vector<std::pair<std::string, std::string>> cases = {
        {"lds.uni", "umov ur0, 0\nlds.uni r1, [ur0]\nhalt\n"},
        {"ldg.uni", "umov ur0, 0\nldg.uni r1, [ur0]\nhalt\n"},
        {"ldg.v8", "mov r0, 0\nldg.v8 r8, [r0]\nhalt\n"},
        {"stg.v8", "mov r0, 0\nstg.v8 [r0], r8\nhalt\n"},
        {"mbar.init", "umov ur4, 4160\nmbar.init [ur4], 1\nhalt\n"},
        {"mbar.arrive", "umov ur4, 4096\nmbar.arrive [ur4]\nhalt\n"},
        {"mbar.arrive.expect_tx", "umov ur4, 4096\nmbar.arrive.expect_tx [ur4], 0\nhalt\n"},
        {"mbar.test_wait", "umov ur4, 4096\nmbar.test_wait p0, [ur4], 0\nhalt\n"},
        {"mbar.try_wait", "umov ur4, 4096\nmbar.try_wait p0, [ur4], 0\nhalt\n"},
        {"atom.add.uni", "umov ur0, 0\numov ur1, 1\natom.add.uni r2, [ur0], ur1\nhalt\n"},
        {"tma.load.1d", tma_operands + "umov ur4, 4104\ntma.load.1d [ur10], [5120, ur8], [ur11]\n"
                        "wait:\nmbar.try_wait p0, [ur4], 0\n@!p0 bra.uni wait\nhalt\n"},
        {"tma.store.1d", tma_operands + "umov ur4, 4104\ntma.store.1d [5120, ur8], [ur10], [ur11]\n"
                         "wait:\nmbar.try_wait p0, [ur4], 0\n@!p0 bra.uni wait\nhalt\n"},
        {"mma.async.init.m64n64k8",
         ".acc C, 0, 64, 64\numov ur0, 0\numov ur1, 1024\nmma.async.init.m64n64k8 C, [ur0], [ur1]\n" + mma_tail},
        {"mma.async.m64n64k8",
         ".acc C, 0, 64, 64\numov ur0, 0\numov ur1, 1024\nmma.async.m64n64k8 C, [ur0], [ur1]\n" + mma_tail},
        {"mma.async.init.m32n64k8",
         ".acc C, 0, 32, 64\numov ur0, 0\numov ur1, 1024\nmma.async.init.m32n64k8 C, [ur0], [ur1]\n" + mma_tail},
        {"mma.async.m32n64k8",
         ".acc C, 0, 32, 64\numov ur0, 0\numov ur1, 1024\nmma.async.m32n64k8 C, [ur0], [ur1]\n" + mma_tail},
        {"mma.commit", "umov ur4, 4096\nmma.commit [ur4]\nhalt\n"},
        {"acc.ld", ".acc C, 0, 64, 64\numov ur8, 0\numov ur9, 0\nacc.ld r10, C, ur8\nhalt\n"},
        {"acc.st.v8", ".acc C, 0, 64, 64\numov ur8, 0\numov ur9, 0\nmov r0, 1\nacc.st.v8 C, ur8, r0\nhalt\n"},
        {"bra.uni", "usread ur0, WARP_ID\nur2r r0, ur0\nsetp.eq p0, r0, 0\n@p0 bra.uni out\nmov r1, 1\nout:\nhalt\n"},
        {"bra.uni", "bra.uni out\nmov r1, 1\nout:\nhalt\n"}, // @PT: no operand, still uniform
    };
    for (const auto& [name, program] : cases) {
        for (const uint64_t mask : {uint64_t{0xffff}, uint64_t{0xfffe}}) {
            const auto what = name + " under mask " + std::to_string(mask) + ": " + program;
            nanosm::Launch launch;
            launch.active_warps = 1;
            setup(launch);
            nanosm::Machine machine(profile, compile(program, profile), std::move(launch));
            machine.test_set_active_mask(0, mask);
#if NANOSM_SANITIZER
            if (mask != 0xffff) {
                require_mentions(i4_panic(machine, what), {name + " is a uniform operation under a partial active mask"},
                    what);
                continue;
            }
#endif
            try {
                while (!machine.done()) tick_capped(machine);
            } catch (const nanosm::SimError& error) {
                throw std::runtime_error(what + ": the machine panicked: " + error.what());
            }
        }
    }
}

// Backlog D12.1: an mbar.init over an mbar with a live waiter is UB (as on
// NVIDIA: an mbarrier is reinitialized only after mbarrier.inval). The
// machine no longer checks it: the init just writes the new state and wakes
// nobody. The SANITIZER's C10 PANICs (the mbar is declared by the launch, so
// it is live from launch). Warp 0 parks in one try_wait on the count-1 mbar
// (nobody arrives); warp 1 re-initializes it with count 3 while warp 0 is
// parked. Without the sanitizer warp 0's try_wait times out (false: r9 stays
// 0) and both halt; the run is capped.
void test_d12_1_mbar_init_over_a_live_waiter() {
    nanosm::Launch launch;
    launch.active_warps = 2;
    auto& shmem = image_bytes(launch.shmem_image, 128);
    write_count_mbar(shmem, 64, 1);
    launch.mbars = {64};
    nanosm::Machine machine(canonical_profile(), compile(R"(
        umov ur1, 64
        usread ur0, WARP_ID
        ur2r r0, ur0
        setp.eq p0, r0, 0
        @p0 bra.uni waiter
        mbar.init [ur1], 3
        halt
    waiter:
        mov r9, 0
        mbar.try_wait p1, [ur1], 0
        @p1 bra.uni woken
        halt
    woken:
        mov r9, 7
        halt
    )"), std::move(launch));
#if NANOSM_SANITIZER
    require_mentions(i4_panic(machine, "mbar.init over a live waiter"),
        {"reason=mbar.init over the live mbar [64, 72) (declared by the launch image)", "inst='mbar.init [ur1], 3"},
        "mbar.init over a live waiter");
#else
    while (!machine.done()) tick_capped(machine);
    const auto state = machine.read_mbar_state(64);
    require(state.valid && state.expected == 3 && state.remaining == 3 && !state.phase,
        "the mbar.init did not write its state over the waited-on mbar");
    require(machine.read_reg(0, 0, 9) == 0, "the waiter was woken; a re-initialization wakes nobody");
#endif
}

// Backlog D12.1: a TMA load placing data over an mbar with a live waiter is
// UB (as on NVIDIA: an mbarrier's memory serves nothing else until
// mbarrier.inval). The machine no longer checks it: the placement just writes
// the bytes. The SANITIZER's C10 PANICs at the placement write (the mbar at
// 1032 is declared by the launch). Warp 0 spins in try_wait on that count-1
// mbar (nobody arrives); warp 1, after a delay that lets warp 0's first probe
// return and park, loads 64 bytes from gmem 0 over shmem [1024, 1088). The
// gmem bytes that land on the mbar are the same mbar with its phase flipped
// (only bit 0 differs, so any probe sees a valid mbar): without the
// sanitizer warp 0's next probe sees phase 1 and leaves, and warp 1 waits
// for the TMA's own mbar at 4104.
void test_d12_1_tma_load_over_a_live_waiter() {
    nanosm::Launch launch;
    launch.active_warps = 2;
    auto& shmem = image_bytes(launch.shmem_image, 8192);
    write_count_mbar(shmem, 1032, 1);
    write_tma_descriptor(shmem, 5120, linear_tma(64));
    write_tx_mbar(shmem, 4104, 64);
    launch.mbars = {1032};
    auto& gmem = image_bytes(launch.gmem_image, 64);
    for (uint32_t word = 0; word < 16; ++word) write_u32(gmem, 4 * word, 0x500 + word);
    nanosm::MbarState flipped;
    flipped.phase = true;
    flipped.valid = true;
    flipped.expected = 1;
    flipped.remaining = 1;
    write_u64(gmem, 8, nanosm::encode_mbar_state(flipped));
    nanosm::Machine machine(canonical_profile(), compile(R"(
        usread ur0, WARP_ID
        ur2r r0, ur0
        setp.eq p0, r0, 0
        @p0 bra.uni waiter
        mov r20, 0
        addi r20, r20, 1
        addi r20, r20, 1
        addi r20, r20, 1
        addi r20, r20, 1
        addi r20, r20, 1
        addi r20, r20, 1
        addi r20, r20, 1
        addi r20, r20, 1
    )" + i4_tma_operands(1024) + R"(
        umov ur4, 4104
        tma.load.1d [ur10], [5120, ur8], [ur11]
    loaded:
        mbar.try_wait p1, [ur4], 0
        @!p1 bra.uni loaded
        halt
    waiter:
        umov ur1, 1032
    spin:
        mbar.try_wait p2, [ur1], 0
        @!p2 bra.uni spin
        halt
    )"), std::move(launch));
#if NANOSM_SANITIZER
    require_mentions(i4_panic(machine, "TMA load over a live waiter"),
        {"reason=live mbar opacity: tma load #0 placement writes bytes [1032, 1040) of the live mbar [1032, 1040) "
         "(declared by the launch image)"},
        "TMA load over a live waiter");
#else
    while (!machine.done()) tick_capped(machine);
    const auto state = machine.read_mbar_state(1032);
    require(state.valid && state.phase && state.expected == 1 && state.remaining == 1,
        "the TMA did not write its bytes over the waited-on mbar");
    require(machine.read_shmem_u32(1024) == 0x500 && machine.read_shmem_u32(1084) == 0x50f,
        "the TMA's other bytes did not land");
#endif
}

// SANITIZER (rfc13 section 7, C10): an atomic writing a LIVE mbar PANICs at
// its write grant (using the RMW engine does not authorize it) -- an mbar
// made live by a SIMT mbar.init, and one the launch declared (live from
// launch). The launch-declared mbar is opaque to an sts and to an mbar.init
// as well, and stays usable by mbar operations. The atomic's other sanitizer
// checks: issued over a TMA load's destination it is a race; its word still
// travelling under a TMA store of it, without a fence.sh, a missing fence
// (with one: clean, and gmem gets its result).
void test_i5_atomic_sanitizer() {
    const auto& profile = canonical_profile();
    auto message = i4_panic(*i5_machine(profile, "umov ur4, 4160\nmbar.init [ur4], 1\numov ur11, 9\n"
        "atom.add.uni r7, [ur4], ur11\nhalt\n", 1, {}, false), "an atomic on a SIMT-initialized mbar");
    require_mentions(message, {"mbar address=4160 reason=live mbar opacity: atom.add.uni r7, [ur4], ur11 (warp 0 pc ",
        "writes bytes [4160, 4164) of the live mbar [4160, 4168) (mbar.init by warp 0 pc 8 at cycle ",
        "until kernel end only mbar operations may touch it"}, "an atomic on a SIMT-initialized mbar");

    const auto declared = [](nanosm::Launch& launch) {
        write_count_mbar(launch.shmem_image.at(0).bytes, 4160, 1);
        launch.mbars = {4160};
    };
    message = i4_panic(*i5_machine(profile, "umov ur4, 4164\numov ur11, 9\natom.exch.uni r7, [ur4], ur11\nhalt\n", 1,
        declared, false), "an atomic on a launch-declared mbar");
    require_mentions(message, {"mbar address=4160 reason=live mbar opacity: atom.exch.uni r7, [ur4], ur11 (warp 0 pc ",
        "writes bytes [4164, 4168) of the live mbar [4160, 4168) (declared by the launch image)"},
        "an atomic on a launch-declared mbar");
    message = i4_panic(*i5_machine(profile, "mov r4, 4160\nsread r20, LANE_ID\nshli r20, r20, 2\nadd r20, r20, r4\n"
        "mov r21, 9\nsts [r20], r21\nhalt\n", 1, declared, false), "an sts on a launch-declared mbar");
    require_mentions(message, {"reason=live mbar opacity: sts writes bytes [4160, 4168) of the live mbar [4160, 4168) "
        "(declared by the launch image)"}, "an sts on a launch-declared mbar");
    message = i4_panic(*i5_machine(profile, "umov ur4, 4160\nmbar.init [ur4], 1\nhalt\n", 1, declared, false),
        "an mbar.init over a launch-declared mbar");
    require_mentions(message, {"reason=mbar.init over the live mbar [4160, 4168) (declared by the launch image)"},
        "an mbar.init over a launch-declared mbar");
    auto legal = i5_machine(profile, "umov ur4, 4160\nmbar.arrive [ur4]\nwait:\nmbar.try_wait p0, [ur4], 0\n"
        "@!p0 bra.uni wait\nhalt\n", 1, declared, false);
    legal->run();
    require(legal->read_mbar_state(4160).phase, "an arrive on a launch-declared mbar did not complete its phase");

    const auto tma = [](nanosm::Launch& launch) {
        auto& shmem = launch.shmem_image.at(0).bytes;
        write_tma_descriptor(shmem, 5120, linear_tma(64));
        write_tx_mbar(shmem, 4104, 64);
    };
    const std::string tail = R"(
    wait:
        mbar.try_wait p0, [ur4], 0
        @!p0 bra.uni wait
        halt
    )";
    // The load's buffer is at 1024, the store's at 0 (ur12); both complete
    // into 4104 (ur4, also the wait's); the coordinate is ur13. The atomics
    // use ur10 and ur11.
    const auto prologue = [](uint32_t shared) { return tma_operands_asm(12, shared, 4, 4104, 13); };
    const std::string tma_load = "tma.load.1d [ur12], [5120, ur13], [ur4]\n";
    const std::string tma_store = "tma.store.1d [5120, ur13], [ur12], [ur4]\n";
    message = i4_panic(*i5_machine(profile, prologue(1024) + tma_load + "umov ur10, 1032\numov ur11, 1\n"
        "atom.add.uni r7, [ur10], ur11\n" + tail, 1, tma, false), "an atomic over a TMA load's destination");
    require_mentions(message, {"reason=shmem race: atom.add.uni writes 1 word, bytes [1032, 1036) owned by tma load #0"},
        "an atomic over a TMA load's destination");
    const std::string atomic = "umov ur10, 8\numov ur11, 0x1000\natom.add.uni r7, [ur10], ur11\n";
    message = i4_panic(*i5_machine(profile, prologue(0) + atomic + tma_store + tail, 1, tma, false),
        "a TMA store of an atomic's word without fence.sh");
    require_mentions(message, {"reason=missing fence.sh: tma store #0 source bytes [", "overlap this warp's earlier "
        "atom.add.uni r7, [ur10], ur11 (pc ", "whose write has not reached the RAM; issue fence.sh between them"},
        "a TMA store of an atomic's word without fence.sh");
    auto fenced = i5_machine(profile, prologue(0) + atomic + "fence.sh\n" + tma_store + tail, 1, tma, false);
    fenced->run();
    for (uint32_t i = 0; i < 16; ++i) {
        require(fenced->read_gmem_u32(4 * i) == (i == 2 ? 0x1102u : 0x100u + i),
            "the fenced TMA store did not carry the atomic's result");
    }
}

// C10 follow-up (both builds): a launch DECLARES the mbars the host
// initialized in its shmem image (Launch::mbars; the Python harness's
// shmem.mbar() declares, so FA's are declared). The launch formats carry the
// list ("mbars", absent: none, emitted only when there is one); the machine
// rejects a declaration that is not an 8-byte aligned shmem slot. C11 / C2:
// an "mbars" entry or "active_warps" that is not an unsigned 32-bit integer
// is rejected before it is narrowed (4294967296 is not address 0, 4160.9 is
// not 4160, -8 is not 0xfffffff8).
void test_i5_launch_declared_mbars_format() {
    nanosm::LaunchImage input;
    input.kernel = compile("halt");
    input.launch.active_warps = 1;
    input.launch.shmem_image.push_back({4160, std::vector<uint8_t>(8, 0)});
    const auto plain = nanosm::emit_launch_image(input);
    require(plain.find("mbars") == std::string::npos, "a launch with no declared mbar emitted an mbars field");
    input.launch.mbars = {4160, 4168};
    const auto declared = nanosm::emit_launch_image(input);
    const auto loaded = nanosm::load_launch_image(declared);
    require(loaded.launch.mbars == std::vector<uint32_t>{4160, 4168} && nanosm::emit_launch_image(loaded) == declared,
        "the launch image lost its declared mbars");
    const auto state_text = nanosm::emit_launch_state(input.launch);
    const auto state = nanosm::load_launch_state(state_text);
    require(state.mbars == input.launch.mbars, "the launch state lost its declared mbars");
    require(nanosm::load_launch_state(nanosm::emit_launch_state(nanosm::Launch{1, {}, {}, {}})).mbars.empty(),
        "an absent mbars field is not an empty list");
    for (const uint32_t bad : {4161u, 65536u}) {
        auto launch = input.launch;
        launch.mbars = {bad};
        try {
            nanosm::Machine machine(canonical_profile(), compile("halt"), std::move(launch));
            throw std::runtime_error("a declared mbar at " + std::to_string(bad) + " was accepted");
        } catch (const nanosm::SimError& error) {
            require(std::string(error.what()).find("launch declares an mbar at " + std::to_string(bad)) != std::string::npos,
                error.what());
        }
    }
    const auto rejected = [](const std::function<std::string()>& load, const std::string& needle, const std::string& what) {
        try {
            (void)load();
        } catch (const nanosm::SimError& error) {
            require(std::string(error.what()).find(needle) != std::string::npos, what + ": " + error.what());
            return;
        }
        throw std::runtime_error(what + " was accepted");
    };
    for (const auto& bad : {nlohmann::json(4294967296ull), nlohmann::json(4160.9), nlohmann::json(-8), nlohmann::json("64")}) {
        for (const bool image : {false, true}) {
            auto document = nlohmann::json::parse(image ? declared : state_text);
            document["mbars"] = {4160, bad};
            const auto text = document.dump();
            rejected([&] { return image ? nanosm::emit_launch_image(nanosm::load_launch_image(text))
                                        : nanosm::emit_launch_state(nanosm::load_launch_state(text)); },
                "launch: mbars[1] is not a shmem address", "mbars entry " + bad.dump());
        }
    }
    for (const auto& bad : {nlohmann::json(4294967297ull), nlohmann::json(1.5), nlohmann::json(-1)}) {
        for (const bool image : {false, true}) {
            auto document = nlohmann::json::parse(image ? declared : state_text);
            document["active_warps"] = bad;
            const auto text = document.dump();
            rejected([&] { return image ? nanosm::emit_launch_image(nanosm::load_launch_image(text))
                                        : nanosm::emit_launch_state(nanosm::load_launch_state(text)); },
                std::string(image ? "launch image" : "launch state") + ": active warp count is not an unsigned 32-bit integer",
                "active_warps " + bad.dump());
        }
    }
}

} // namespace

// I3: store pause sequence (D = 2, 3)
TEST(Rmw, i3_store_pause_sequence) {
    test_i3_store_pause_sequence();
}

// I3: reads during an RMW
TEST(Rmw, i3_reads_during_an_rmw) {
    test_i3_reads_during_an_rmw();
}

// I3: READ_FIRST at the RMW's write
TEST(Rmw, i3_read_first_at_rmw_write) {
    test_i3_read_first_at_rmw_write();
}

// I3: RMW / store alternation
TEST(Rmw, i3_rmw_store_alternation) {
    test_i3_rmw_store_alternation();
}

// I3: SIMT RMW waits for its own stores
TEST(Rmw, i3_simt_rmw_waits_for_own_stores) {
    test_i3_simt_rmw_waits_for_own_stores();
}

// I3: RMW admission fairness
TEST(Rmw, i3_rmw_admission_fairness) {
    test_i3_rmw_admission_fairness();
}

// I3: signed tx in both orders
TEST(Rmw, i3_signed_tx_both_orders) {
    test_i3_signed_tx_both_orders();
}

// I3: TMA start reads no mbar
TEST(Rmw, i3_tma_start_reads_no_mbar) {
    test_i3_tma_start_reads_no_mbar();
}

// I3: tx range PANICs
TEST(Rmw, i3_tx_range_panics) {
    test_i3_tx_range_panics();
}

// I3: sanitizer wrong-phase complete_tx
TEST(Rmw, i3_sanitizer_wrong_phase_complete_tx) {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    test_i3_sanitizer_wrong_phase_complete_tx();
}

// I3: sanitizer nonzero tx at kernel end
TEST(Rmw, i3_sanitizer_nonzero_tx_at_kernel_end) {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    test_i3_sanitizer_nonzero_tx_at_kernel_end();
}

// I5: atom.*.uni encoding, qtxc and hand assembly
TEST(Rmw, i5_atomic_encoding_and_assembly) {
    test_i5_atomic_encoding_and_assembly();
}

// I5: one word, applied once; the timeline (D, C = 2, 3)
TEST(Rmw, i5_one_word_timeline) {
    test_i5_one_word_timeline();
}

// I5: int32 add wraps at the boundary
TEST(Rmw, i5_add_wraps_at_the_int32_boundary) {
    test_i5_add_wraps_at_the_int32_boundary();
}

// I5: return slot admission
TEST(Rmw, i5_return_slot_admission) {
    test_i5_return_slot_admission();
}

// I5: SIMT RMW register busy
TEST(Rmw, i5_simt_rmw_register_busy) {
    test_i5_simt_rmw_register_busy();
}

// I5: store pause and resume
TEST(Rmw, i5_store_pause_and_resume) {
    test_i5_store_pause_and_resume();
}

// I5: read after the store under way
TEST(Rmw, i5_read_after_the_store_under_way) {
    test_i5_read_after_the_store_under_way();
}

// I5: same-warp order around an atomic
TEST(Rmw, i5_same_warp_order) {
    test_i5_same_warp_order();
}

// I5: launch-declared mbars (format; C11, C2)
TEST(Rmw, i5_launch_declared_mbars_format) {
    test_i5_launch_declared_mbars_format();
}

// I5: atomic sanitizer: C10, race, missing fence
TEST(Rmw, i5_atomic_sanitizer) {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    test_i5_atomic_sanitizer();
}

// D12: uniform ops under a partial mask
TEST(Rmw, d12_uniform_ops_under_a_partial_mask) {
    test_d12_uniform_ops_under_a_partial_mask();
}

// D12: sts with two lanes on one word
TEST(Rmw, d12_sts_two_lanes_on_one_word) {
    test_d12_sts_two_lanes_on_one_word();
}

// D12: stg with two lanes on one word
TEST(Rmw, d12_stg_two_lanes_on_one_word) {
    test_d12_stg_two_lanes_on_one_word();
}

// D12: sts over an mbar with a live waiter
TEST(Rmw, d12_sts_over_an_mbar_with_a_live_waiter) {
    test_d12_sts_over_an_mbar_with_a_live_waiter();
}

// D12.1: every uniform op under a partial mask
TEST(Rmw, d12_1_every_uniform_op_under_a_partial_mask) {
    test_d12_1_every_uniform_op_under_a_partial_mask();
}

// D12.1: mbar.init over an mbar with a live waiter
TEST(Rmw, d12_1_mbar_init_over_a_live_waiter) {
    test_d12_1_mbar_init_over_a_live_waiter();
}

// D12.1: TMA load over an mbar with a live waiter
TEST(Rmw, d12_1_tma_load_over_a_live_waiter) {
    test_d12_1_tma_load_over_a_live_waiter();
}

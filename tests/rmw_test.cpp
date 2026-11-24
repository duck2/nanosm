// The RMW engine: the store pause, its admission and alternation, the signed
// tx count, uniform atomics, and the sanitizer's checks of mbars, atomics
// and uniform operations under a partial mask.
#include "test_support.hpp"

namespace {

using namespace nanosm_test;

// Every mbar RMW, in order (the engine runs one at a time): start, read, write.
std::vector<RmwSlice> rmw_slices(const nanosm::Machine& machine) {
    std::vector<RmwSlice> rmws;
    for (const auto& [start, detail] : mbar_rmw_services(machine)) rmws.push_back({start, 0, 0, detail});
    std::size_t reads = 0;
    std::size_t writes = 0;
    for (const auto& grant : grants_of(machine)) {
        if (grant.client != "rmw") continue;
        auto& next = grant.service == "A-read" ? reads : writes;
        require(next < rmws.size(), "an RMW grant without an RMW slice");
        (grant.service == "A-read" ? rmws[next].read : rmws[next].write) = grant.cycle;
        ++next;
    }
    return rmws;
}

// The cycles store transactions started (their first round's grant).
std::vector<uint64_t> store_starts(const nanosm::Machine& machine) {
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

// Sixteen consecutive cycles from `first`: one sixteen-round transaction.
std::vector<uint64_t> sixteen_rounds(uint64_t first) {
    std::vector<uint64_t> cycles;
    for (uint64_t round = 0; round < 16; ++round) cycles.push_back(first + round);
    return cycles;
}

std::vector<AtomicSlice> atomics_of(const nanosm::Machine& machine) {
    std::vector<AtomicSlice> atomics;
    for (const auto& event : events_named(machine, "atomic RMW")) {
        atomics.push_back({event.start_cycle, event.start_cycle + event.duration_cycles, {}, {}, event.detail});
    }
    std::sort(atomics.begin(), atomics.end(), [](const auto& a, const auto& b) { return a.start < b.start; });
    for (const auto& grant : grants_of(machine)) {
        if (grant.client != "rmw") continue;
        for (auto& atomic : atomics) {
            if (grant.cycle < atomic.start || grant.cycle >= atomic.end) continue;
            (grant.service == "A-read" ? atomic.reads : atomic.writes).push_back(grant.cycle);
        }
    }
    return atomics;
}

// The one atomic a run made.
AtomicSlice one_atomic(const nanosm::Machine& machine, const std::string& what) {
    const auto atomics = atomics_of(machine);
    require(atomics.size() == 1, what + ": expected one atomic, found " + std::to_string(atomics.size()));
    return atomics.front();
}

// One read at g, the old word at the port output at a = g + D + Lb, the write
// granted at w = a + C, the slice ending at w + 1. `read` is the expected g.
void require_atomic_timeline(const AtomicSlice& atomic, const nanosm::DeviceProfile& profile, uint64_t read,
    const std::string& what) {
    const uint64_t w = read + profile.shmem_port_a_depth + profile.shmem_read_latency_cycles +
        profile.shmem_rmw_compute_cycles;
    require(atomic.reads == std::vector<uint64_t>{read} && atomic.writes == std::vector<uint64_t>{w} &&
            atomic.end == w + 1,
        what + ": not read c" + std::to_string(read) + ", write c" + std::to_string(w) + ":" + describe(atomic));
}

// The atomic ALU, written independently of the machine's: 32-bit wrap through
// 64-bit arithmetic.
uint32_t atomic_result(nanosm::Opcode opcode, uint32_t old_word, uint32_t operand) {
    switch (opcode) {
    case nanosm::Opcode::atom_add_uni: return static_cast<uint32_t>((uint64_t{old_word} + operand) & 0xffffffffull);
    case nanosm::Opcode::atom_and_uni: return old_word & operand;
    case nanosm::Opcode::atom_xor_uni: return old_word ^ operand;
    default: return operand; // exch
    }
}

const std::array<std::pair<nanosm::Opcode, const char*>, 4> atomic_ops = {{
    {nanosm::Opcode::atom_add_uni, "atom.add.uni"},
    {nanosm::Opcode::atom_and_uni, "atom.and.uni"},
    {nanosm::Opcode::atom_xor_uni, "atom.xor.uni"},
    {nanosm::Opcode::atom_exch_uni, "atom.exch.uni"},
}};

// A one-dimensional TMA load of 16 words (gmem 0, pattern_word) at
// descriptor 5120, completing into an mbar at 4104 that expects one arrival
// and no bytes yet; shmem otherwise zero.
nanosm::Launch tma_into_an_arrival_mbar() {
    auto launch = launch_warps(1);
    auto& shmem = image_bytes(launch.test_shmem, 8192);
    write_rank1_tma(shmem, 5120, 0, 16, 4104);
    write_mbar(shmem, 4104, mbar_state(1, 1));
    fill_pattern(image_bytes(launch.gmem_image, 64), 0, 16, 0);
    return launch;
}

} // namespace

// The global store pause. An RMW that STARTS while a store transaction is
// under way lets that store keep its rounds and reads the cycle after its
// last round (matched D lands the store's last write before the RMW's
// sample). From its start to its write grant w = read + D + Lb + C no new
// store round is granted; a paused store is granted at w + 1 -- one cycle
// after the RMW's final write GRANT, not after its commit -- and still reaches
// the RAM after the RMW's write (the mbar lands at the edge of w + D, the
// store's first round at w + 1 + D). Reads continue meanwhile. store_paused
// counts the cycles a store waits in its slot under the pause.
//   One warp, D = 2 and 3: sts X (sixteen rounds, one bank) granted from g,
//   the event held from g + 4; the younger sts Y, admitted at X's last grant,
//   waits in its slot from g + 16 + t to w.
//   Two warps, each a sixteen-round sts then an lds of other words: the event
//   is held from g + 2 (warp 0's first round g); warp 0's lds, ordered behind
//   its sts until its last round and then dispatched, is granted from read +
//   t, during the RMW; warp 1's sts waits in its slot from the start to w.
TEST(Rmw, store_pause_sequence) {
    for (const uint32_t d : {2u, 3u}) {
        auto profile = canonical_profile();
        profile.shmem_port_a_depth = d;
        const uint64_t lb = profile.shmem_read_latency_cycles;
        const uint64_t c = profile.shmem_rmw_compute_cycles;
        const auto what = "D=" + std::to_string(d) + ": ";
        auto machine = debug_machine(profile, R"(
            sread r0, LANE_ID
            shli r0, r0, 6
            mov r5, 0x5a
            sts [r0 + 8], r5 {pred=PT, stall=0, set=sb0, wait=-}
            sts [r0 + 12], r5 {pred=PT, stall=0, set=sb1, wait=-}
            halt {pred=PT, stall=0, set=-, wait=sb0|sb1}
        )", 1, event_mbars());
        const auto t = nanosm::issue_to_dispatch(profile);
        // X issues after sread -> shli and the mov's r5 (2 alu_distance + 1).
        const auto g = nanosm::fetch_to_issue + 2 * alu_distance(profile) + 1 + t + 1;
        run_until(*machine, g + 4);
        host_rmw_event(*machine);
        std::optional<uint64_t> mbar_landed; // the cycle at whose clock edge the RMW's write landed
        std::optional<uint64_t> y_landed;    // ... Y's round 0 (lane 0's word)
        while (!machine->done()) {
            const auto cycle = machine->cycle();
            tick_capped(*machine);
            if (!mbar_landed && machine->read_mbar_state(event_mbar).phase) mbar_landed = cycle;
            if (!y_landed && machine->read_shmem_u32(12) == 0x5a) y_landed = cycle;
        }
        const uint64_t w = g + 16 + d + lb + c;
        const auto rmws = rmw_slices(*machine);
        require(rmws.size() == 1 && rmws[0].start == g + 4 && rmws[0].read == g + 16 && rmws[0].write == w,
            what + "the RMW did not start at g + 4, read at g + 16 (X's last round + 1), write at g + 16 + D + Lb + C:" +
                describe(rmws) + " | issues" + issue_log(*machine));
        auto expected = sixteen_rounds(g);               // X
        for (const auto cycle : sixteen_rounds(w + 1)) expected.push_back(cycle); // Y
        require(grant_cycles(*machine, "A-write", "simt") == expected,
            what + "store rounds (X g..g + 15, Y from w + 1):" + describe(grants_of(*machine, "A-write")));
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

    const auto& profile = canonical_profile();
    auto machine = debug_machine(profile, R"(
        sread r0, LANE_ID
        shli r0, r0, 6
        mov r5, 0x5a
        sts [r0 + 4], r5 {pred=PT, stall=0, set=sb0, wait=-}
        lds r6, [r0 + 8] {pred=PT, stall=0, set=sb1, wait=-}
        halt {pred=PT, stall=0, set=-, wait=sb0|sb1}
    )", 2, event_mbars());
    // The host's event arrives while warp 0's sts is under way.
    while (grants_of(*machine, "A-write").empty()) tick_capped(*machine);
    tick_capped(*machine);
    host_rmw_event(*machine);
    const auto start = machine->cycle();
    machine->run();
    const auto grants = grants_of(*machine);
    const auto stores = grant_cycles(*machine, "A-write", "simt");
    const auto loads = grant_cycles(*machine, "A-read", "simt");
    const auto rmws = rmw_slices(*machine);
    require(stores.size() == 32 && loads.size() >= 16 && rmws.size() == 1, "two warps: the rounds:" + describe(grants));
    const auto g = stores.front();
    const auto read = rmws[0].read;
    const auto write = rmws[0].write;
    require(start == g + 2 && rmws[0].start == start, "two warps: the RMW did not start in the cycle the event was held (g + 2)");
    require(std::vector<uint64_t>(stores.begin(), stores.begin() + 16) == sixteen_rounds(g) && read == g + 16 &&
            write == read + read_return(profile) + profile.shmem_rmw_compute_cycles,
        "two warps: the RMW's read was not granted the cycle after the store's last round, or its write D + Lb + C "
        "later:" + describe(grants));
    require(std::vector<uint64_t>(loads.begin(), loads.begin() + 16) ==
            sixteen_rounds(read + nanosm::issue_to_dispatch(profile)),
        "two warps: reads did not continue during the RMW: warp 0's lds was not granted from read + "
        "issue_to_dispatch:" + describe(grants));
    require(std::vector<uint64_t>(stores.begin() + 16, stores.end()) == sixteen_rounds(write + 1),
        "two warps: warp 1's paused sts was not granted from the cycle after the RMW's write grant:" + describe(grants));
    require(machine->trace().summary().shmem_store_paused == write - start,
        "two warps: store_paused should count every pause cycle with warp 1's sts waiting");
    require(machine->read_mbar_state(event_mbar).phase, "two warps: the RMW's update was lost");
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(machine->read_shmem_u32(64 * lane + 4) == 0x5a, "two warps: sts data");
        require(machine->read_reg(0, lane, 6) == 0x100 + 16 * lane + 2, "two warps: lds data");
    }
}

// Reads continue during an RMW. The host's event is placed at the lds.uni's
// dispatch e (nothing waits on A: the RMW starts and reads at e); its write
// is granted at e + D + Lb + C. In between A-read serves warp 0's lds.uni (e +
// 1), the TMA's descriptor read (e + 2: the tma's command reaches the TMA at
// its issue + 3) and warp 0's probe (e + 4: it reads a uniform register, so
// issue refuses it in the cycle after the tma, whose coordinate round holds
// the URF read ports).
TEST(Rmw, reads_during_an_rmw) {
    const auto& profile = canonical_profile();
    auto launch = launch_warps(1);
    auto& shmem = image_bytes(launch.test_shmem, 8192);
    for (uint32_t word = 0; word < 512; ++word) write_u32(shmem, 4 * word, 0x100 + word);
    write_mbar(shmem, event_mbar, mbar_state(1, 0, 64));
    write_rank1_tma(shmem, 5120, 0, 16, 4104);
    fill_pattern(image_bytes(launch.gmem_image, 64), 0, 16, 0);
    // The TMA's operands: buffer at 1024, completing into 4104, coordinate 0.
    // The lds.uni's base ur1 is never written (0); the probe's mbar is in ur12.
    nanosm::Machine machine(profile, hand_assemble(hand_asm(profile, one_tma_operands(1024) + R"(
        umov ur12, 4104
        mov r25, 0
        lds.uni r3, [ur1 + 64] {pred=PT, stall=0, set=sb0, wait=-}
        tma.load.1d [ur10], [5120, ur8], [ur11]
        mbar.test_wait p0, [ur12], 0
        halt {pred=PT, stall=0, set=-, wait=sb0}
    )"), profile), std::move(launch), nanosm::TraceOptions{.timeline = true, .debug = true});
    // The lds.uni is the sixth instruction, issued at fetch_to_issue + 5.
    const uint64_t e = nanosm::fetch_to_issue + 5 + nanosm::issue_to_dispatch(profile);
    run_until(machine, e);
    host_rmw_event(machine);
    machine.run();
    const uint64_t w = e + read_return(profile) + profile.shmem_rmw_compute_cycles;
    const auto rmws = rmw_slices(machine);
    require(rmws.size() == 2 && rmws[0].start == e && rmws[0].read == e && rmws[0].write == w,
        "the host event's RMW:" + describe(rmws));
    auto reads = grants_of(machine, "A-read");
    reads.resize(std::min<std::size_t>(reads.size(), 4));
    require(reads == std::vector<Grant>{{e, "A-read", "rmw", 0}, {e + 1, "A-read", "simt", 0},
                         {e + 2, "A-read", "tma", 0}, {e + 4, "A-read", "probe", 0}} && reads.back().cycle < w,
        "A-read did not serve an lds, the TMA and a probe between the RMW's read and its write:" + describe(reads) +
            " | issues" + issue_log(machine));
    for (uint32_t lane = 0; lane < 16; ++lane) require(machine.read_reg(0, lane, 3) == 0x100 + 16, "lds.uni data");
    for (uint32_t i = 0; i < 16; ++i) require(machine.read_shmem_u32(1024 + 4 * i) == pattern_word(i), "TMA data");
}

// A read of the mbar that samples in the cycle the RMW's write reaches the
// RAM sees the OLD word (READ_FIRST). The host's event's RMW reads at k and
// has its write granted at w = k + D + Lb + C, landing at w + D. Warp 0's
// test_wait, placed by its stall count, is granted at g and samples at g + D:
// it sees the new phase iff g > w; granted at w itself it reads the old phase.
// Its earliest grant (it needs the umov's ur4 -- uniform_distance -- then its
// dispatch) is one cycle before w, so stalls 0..8 cross w.
TEST(Rmw, read_first_at_rmw_write) {
    const auto& profile = canonical_profile();
    const uint64_t rmw = read_return(profile) + profile.shmem_rmw_compute_cycles;
    const uint64_t w = nanosm::fetch_to_issue + uniform_distance(profile) + nanosm::issue_to_dispatch(profile) + 2;
    const uint64_t k = w - rmw;
    std::set<int> sides;
    for (uint32_t stall = 0; stall <= 8; ++stall) {
        auto machine = debug_machine(profile, R"(
            umov ur4, 4096 {pred=PT, stall=)" + std::to_string(stall) + R"(, set=-, wait=-}
            mov r9, 0
            mbar.test_wait p0, [ur4], 0
            bra.uni done {pred=!p0, stall=0, set=-, wait=-}
            mov r9, 1
        done: halt
        )", 1, event_mbars());
        run_until(*machine, k);
        host_rmw_event(*machine);
        machine->run();
        const auto probes = grant_cycles(*machine, "A-read", "probe");
        const uint64_t probe = probes.empty() ? 0 : probes.back();
        const auto rmws = rmw_slices(*machine);
        const auto what = "probe granted c" + std::to_string(probe) + "," + describe(rmws) + ": ";
        require(probe != 0 && rmws.size() == 1 && rmws[0].read == k && rmws[0].write == w, what + "timeline");
        require(machine->read_reg(0, 0, 9) == (probe > w ? 1u : 0u),
            what + (probe > w ? "the probe missed the RMW's write" : "READ_FIRST: the probe did not read the old phase"));
        sides.insert(probe < w ? -1 : probe == w ? 0 : 1);
    }
    require(sides == std::set<int>{-1, 0, 1}, "the sweep did not cross the RMW's write-grant cycle");
}

// The alternation between RMW starts and store transactions: whenever both
// wait, the one that did not start last goes. Warps 0 and 2 issue four
// mbar.arrive each (while one's RMW runs, the other's waits in the SIMT RMW
// register, so after every RMW both classes wait); warp 1 issues eight
// one-round sts on the same words (each waits for the previous one's grant).
// The waiting store starts at w + 1 (the RMW's write grant + 1) and the next
// RMW at w + 2, reading at once: the starts go R, S, R, S, ... -- back-to-back
// RMWs do not starve the stores -- and no store round is granted between an
// RMW's read and its write. The arrive warps' umov stalls 10 cycles so that
// warp 1's first store reaches its slot under the first RMW's pause.
TEST(Rmw, rmw_store_alternation) {
    const auto& profile = canonical_profile();
    std::string arrives;
    std::string stores;
    for (int i = 0; i < 8; ++i) {
        if (i < 4) arrives += "mbar.arrive [ur4]\n";
        stores += "sts [r20 + 2048], r21 {pred=PT, stall=0, set=sb0, wait=-}\n";
    }
    auto machine = debug_machine(profile, R"(
        usread ur9, WARP_ID
        ur2r r29, ur9
        setp.eq.u32 p6, r29, 1
        bra.uni stores {pred=p6, stall=0, set=-, wait=-}
        umov ur4, 4160 {pred=PT, stall=10, set=-, wait=-}
    )" + arrives + R"(
        halt
    stores: sread r20, LANE_ID
        shli r20, r20, 2
        mov r21, 0x33
    )" + stores + R"(
        halt {pred=PT, stall=0, set=-, wait=sb0}
    )", 3, {{4160, mbar_state(8, 8)}});
    machine->run();
    const auto rmws = rmw_slices(*machine);
    const auto starts_of_stores = store_starts(*machine);
    std::vector<std::pair<uint64_t, char>> starts;
    for (const auto& rmw : rmws) starts.emplace_back(rmw.start, 'R');
    for (const auto start : starts_of_stores) starts.emplace_back(start, 'S');
    std::sort(starts.begin(), starts.end());
    std::string order;
    for (const auto& [cycle, kind] : starts) order += kind;
    require(order == "RSRSRSRSRSRSRSRS", "RMW (R) and store (S) starts did not alternate: " + order + describe(rmws));
    for (std::size_t k = 0; k < 8; ++k) {
        require(starts_of_stores[k] == rmws[k].write + 1, "store " + std::to_string(k) +
            " did not start the cycle after RMW " + std::to_string(k) + "'s write grant:" + describe(rmws));
        if (k + 1 < 8) {
            require(rmws[k + 1].start == rmws[k].write + 2 && rmws[k + 1].read == rmws[k + 1].start,
                "RMW " + std::to_string(k + 1) + " did not start (and read) at w + 2:" + describe(rmws));
        }
    }
    for (const auto cycle : grant_cycles(*machine, "A-write", "simt")) {
        for (const auto& rmw : rmws) {
            require(cycle < rmw.read || cycle > rmw.write,
                "a store round was granted between an RMW's read and its write, c" + std::to_string(cycle) + ":" +
                    describe(rmws));
        }
    }
    require(machine->read_mbar_state(4160).phase, "the eight arrives did not complete the phase");
    for (uint32_t lane = 0; lane < 16; ++lane) require(machine->read_shmem_u32(2048 + 4 * lane) == 0x33, "store data");
}

// A SIMT RMW takes the engine only when its warp has no unserviced lds /
// sts: it waits at issue, BEFORE the SIMT RMW register and the engine, so its
// own store pause can never hold them (no deadlock). t = issue_to_dispatch.
//   Paused sts: the host's event is placed as the sts issues (at s; its RMW
//   reads at k = s + 1 and writes at w = k + D + Lb + C), and its pause holds
//   the sts in its slot until w: granted w + 1 .. w + 16. The arrive is
//   ordered behind it from s + 1, admitted at its last grant (w + 16), and
//   starts (reads) at its dispatch + 1, w + t + 17.
//   sts, lds, arrive, all lanes on one bank, at canonical and two
//   noncanonical depths: the sts's rounds s + t + 1 .. s + t + 16; the lds
//   over the same words, ordered behind it from s + 1 (t + 15 cycles), is
//   granted from s + 2t + 17 and reads the stored words; the arrive, ordered
//   behind the lds (again t + 15 cycles), is admitted at s + 2t + 32 and read
//   at s + 3t + 33.
TEST(Rmw, simt_rmw_waits_for_own_stores) {
    {
        const auto& profile = canonical_profile();
        const uint64_t rmw = read_return(profile) + profile.shmem_rmw_compute_cycles; // read to write
        const uint64_t t = nanosm::issue_to_dispatch(profile);
        auto machine = debug_machine(profile, R"(
            sread r0, LANE_ID
            shli r0, r0, 6
            umov ur4, 4160
            mov r5, 0x5a
            sts [r0 + 8], r5 {pred=PT, stall=0, set=sb0, wait=-}
            mbar.arrive [ur4]
            halt {pred=PT, stall=0, set=-, wait=sb0}
        )", 1, {{event_mbar, mbar_state(1, 0, 64)}, {4160, mbar_state(2, 2)}});
        // The sts needs the mov's r5: sread, shli, then 2 alu_distance + 2.
        const uint64_t s = nanosm::fetch_to_issue + 2 * alu_distance(profile) + 2;
        const uint64_t k = s + 1;
        const uint64_t w = k + rmw;
        require(s + t < w, "the sts would not reach its slot under the pause");
        run_until(*machine, k);
        host_rmw_event(*machine);
        machine->run();
        const auto rmws = rmw_slices(*machine);
        require(issued_at(*machine, 4 * 8) == std::vector<uint64_t>{s}, "the sts did not issue at s:" + issue_log(*machine));
        require(rmws.size() == 2 && rmws[0].start == k && rmws[0].read == k && rmws[0].write == w,
            "the host event's RMW:" + describe(rmws));
        require(grant_cycles(*machine, "A-write", "simt") == sixteen_rounds(w + 1),
            "the sts did not resume at the event's write grant + 1:" + describe(grants_of(*machine)));
        require(issued_at(*machine, 5 * 8) == std::vector<uint64_t>{w + 16} &&
                refusals(*machine, 0, "ordered behind own access") ==
                    std::vector<std::pair<uint64_t, uint64_t>>{{s + 1, w + 16 - (s + 1)}},
            "the arrive was not ordered behind its own sts until the sts's last grant (s + 1 .. w + 15, admitted w + 16)");
        require(rmws[1].detail == "arrive, address=4160" && rmws[1].start == w + t + 17 && rmws[1].read == w + t + 17 &&
                rmws[1].write == w + t + 17 + rmw,
            "the arrive's RMW did not start and read at w + t + 17:" + describe(rmws));
        require(machine->read_mbar_state(4160).remaining == 1, "the arrive was lost");
    }

    for (const auto& [d, lb, c] : std::vector<std::tuple<uint32_t, uint32_t, uint32_t>>{{2, 2, 2}, {3, 1, 1}, {1, 3, 3}}) {
        auto profile = canonical_profile();
        profile.shmem_port_a_depth = d;
        profile.shmem_read_latency_cycles = lb;
        profile.shmem_rmw_compute_cycles = c;
        const auto what = "D=" + std::to_string(d) + " Lb=" + std::to_string(lb) + " C=" + std::to_string(c) + ": ";
        auto machine = debug_machine(profile, R"(
            sread r0, LANE_ID
            shli r0, r0, 6
            umov ur4, 4160
            mov r5, 0x5a
            sts [r0 + 8], r5 {pred=PT, stall=0, set=sb0, wait=-}
            lds r6, [r0 + 8] {pred=PT, stall=0, set=sb1, wait=-}
            mbar.arrive [ur4]
            halt {pred=PT, stall=0, set=-, wait=sb0|sb1}
        )", 1, {{4160, mbar_state(2, 2)}});
        machine->run();
        const auto t = nanosm::issue_to_dispatch(profile);
        const auto s = nanosm::fetch_to_issue + 2 + 2 * alu_distance(profile); // sread, shli; umov ur4, mov r5; the sts
        const auto issues = " | issues" + issue_log(*machine);
        require(issued_at(*machine, 4 * 8) == std::vector<uint64_t>{s}, what + "the sts did not issue at s:" + issues);
        require(grant_cycles(*machine, "A-write", "simt") == sixteen_rounds(s + t + 1) &&
                grant_cycles(*machine, "A-read", "simt") == sixteen_rounds(s + 2 * t + 17),
            what + "sts rounds not from s + t + 1 or lds rounds not from s + 2t + 17:" + describe(grants_of(*machine)) +
                issues);
        require(issued_at(*machine, 5 * 8) == std::vector<uint64_t>{s + t + 16} &&
                issued_at(*machine, 6 * 8) == std::vector<uint64_t>{s + 2 * t + 32},
            what + "the lds was not admitted at the sts's last grant, or the arrive at the lds's");
        const auto ordered = refusals(*machine, 0, "ordered behind own access");
        require(ordered == std::vector<std::pair<uint64_t, uint64_t>>{{s + 1, t + 15}, {s + t + 17, t + 15}},
            what + "ordered refusals:" + describe(ordered));
        require(grant_cycles(*machine, "A-read", "rmw") == std::vector<uint64_t>{s + 3 * t + 33} &&
                grant_cycles(*machine, "A-write", "rmw") == std::vector<uint64_t>{s + 3 * t + 33 + d + lb + c},
            what + "the RMW was not read at s + 3t + 33 and written D + Lb + C later");
        for (uint32_t lane = 0; lane < 16; ++lane) {
            require(machine->read_reg(0, lane, 6) == 0x5a, what + "RAW: the lds missed the sts");
        }
        require(machine->read_mbar_state(4160).remaining == 1, what + "the arrive was lost");
    }
}

// The RMW engine's admission (implementer choice): the SIMT RMW register and
// the engines' event registers alternate by class -- the class that did not
// start last goes; at the first tie, SIMT. Admission reads registered state.
//   Fairness: the host keeps the TMA's event register full (zero-byte
//   complete_tx events on 4168, which change nothing) while warps 0 and 1
//   issue two mbar.arrive each on 4160 (while one's arrive runs, the other's
//   waits in the register). Feeding starts one busy period before the first
//   arrive is held (at its issue + issue_to_dispatch + 1), so the starts
//   alternate TMA, SIMT, ... one busy period apart: an endless supply of
//   engine events cannot starve the warps.
//   First tie: an mma.commit issued at m presents its arrive into the MMA's
//   event register at P = m + issue_to_dispatch + 1, the cycle the warp's
//   mbar.arrive (issued at m + 1) reaches the SIMT RMW register; both are held
//   at P + 1, the SIMT arrive is served first and the commit's a busy period
//   later.
TEST(Rmw, rmw_admission_fairness) {
    const auto& profile = canonical_profile();
    const auto busy = rmw_busy_cycles(profile);
    auto machine = debug_machine(profile, R"(
        umov ur4, 4160
        mbar.arrive [ur4]
        mbar.arrive [ur4]
        halt
    )", 2, {{4160, mbar_state(5, 5)}, {4168, mbar_state(1, 1)}});
    const uint64_t f =
        nanosm::fetch_to_issue + uniform_distance(profile) + nanosm::issue_to_dispatch(profile) + 1 - busy;
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

    auto tie = debug_machine(profile, R"(
        umov ur4, 4160
        umov ur5, 4168
        mma.commit [ur4]
        mbar.arrive [ur5]
        halt
    )", 1, {{4160, mbar_state(1, 1)}, {4168, mbar_state(2, 2)}});
    tie->run();
    // The commit needs the first umov's ur4 (the first umov issues at fetch_to_issue).
    const auto m = nanosm::fetch_to_issue + uniform_distance(profile);
    const auto p = m + nanosm::issue_to_dispatch(profile) + 1;
    require(issued_at(*tie, 2 * 8) == std::vector<uint64_t>{m} && issued_at(*tie, 3 * 8) == std::vector<uint64_t>{m + 1},
        "first tie: the commit did not issue at m or the arrive at m + 1 (dispatched in the cycle the commit was "
        "presented)");
    require(mbar_rmw_services(*tie) == std::vector<RmwService>{{p + 1, "arrive, address=4168"},
                                           {p + 1 + busy, "arrive (mma.commit), address=4160"}},
        "first tie: the SIMT arrive was not served at P + 1 and the commit after it");
    require(tie->read_mbar_state(4160).phase && tie->read_mbar_state(4168).remaining == 1, "first tie: an arrive was lost");
}

// The signed tx count, in both orders ((pending arrivals, tx); the host's
// complete_tx and warp 0's arrive.expect_tx):
//   completion first:  (1, 0) -complete_tx 128-> (1, -128) -arrive.expect_tx 128-> (0, 0): complete
//   expectation first: (1, 0) -arrive.expect_tx 128-> (0, 128) -complete_tx 128-> (0, 0): complete
// Observed at fetch_to_issue + 20 (expectation first the arrive has landed;
// completion first it is still behind its stall of 20); at the end the phase
// has flipped (arrivals reloaded, tx 0) and the warp's try_wait returned true.
// Completion first by a TMA: the TMA reads no mbar at its start (no check
// that enough bytes were expected), so a 64-byte load into an mbar expecting
// one arrival and no bytes runs its complete_tx ahead, (1, 0) -> (1, -64), and
// the warp's arrive.expect_tx of 64, long after, completes the phase. A-read
// serves the TMA one round (its descriptor's), and shmem counts three
// requests (the descriptor read and two sector placements).
TEST(Rmw, signed_tx_both_orders) {
    const auto& profile = canonical_profile();
    for (const bool completion_first : {true, false}) {
        const auto what = std::string(completion_first ? "completion first: " : "expectation first: ");
        auto machine = debug_machine(profile, std::string(completion_first ? R"(
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
        )", 1, {{4160, mbar_state(1, 1)}});
        if (completion_first) machine->complete_mbar_tx(4160, 128, 0);
        run_until(*machine, nanosm::fetch_to_issue + 20);
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

    std::string program = tma_operands_asm(10, 2048, 11, 4104, 8) + "umov ur12, 4104\ntma.load.1d [ur10], [5120, ur8], [ur11]\n";
    for (int i = 0; i < 10; ++i) program += "mov r9, 0 {pred=PT, stall=31, set=-, wait=-}\n";
    program += R"(
        mbar.arrive.expect_tx [ur12], 64
    wait: mbar.try_wait p0, [ur12], 0
        bra.uni wait {pred=!p0, stall=0, set=-, wait=-}
        mov r9, 1
        halt
    )";
    nanosm::Machine machine(profile, hand_assemble(hand_asm(profile, program), profile), tma_into_an_arrival_mbar(),
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

// An mbar expecting two arrivals: after one its phase has not completed (a
// test_wait for phase 0 answers false, one arrival remains); the second
// completes it (the test_wait answers true, the count reloads to two).
TEST(Rmw, two_arrival_phase) {
    const auto& profile = canonical_profile();
    for (const uint32_t arrivals : {1u, 2u}) {
        const auto what = std::to_string(arrivals) + " of 2 arrivals: ";
        std::string program = "umov ur4, 4160\nmbar.init [ur4], 2\n";
        for (uint32_t i = 0; i < arrivals; ++i) program += "mbar.arrive [ur4]\n";
        program += "mbar.test_wait p0, [ur4], 0\n@p0 bra.uni complete\nmov r1, 1\nhalt\ncomplete:\nmov r1, 2\nhalt\n";
        nanosm::Machine machine(profile, compile(program), launch_warps(1));
        machine.run();
        const bool complete = arrivals == 2;
        const auto state = machine.read_mbar_state(4160);
        require(state.valid && state.phase == complete && state.expected == 2 &&
                state.remaining == (complete ? 2 : 1),
            what + "(phase, expected, remaining) = (" + std::to_string(state.phase) + ", " +
                std::to_string(state.expected) + ", " + std::to_string(state.remaining) + ")");
        for (uint32_t lane = 0; lane < 16; ++lane) {
            require(machine.read_reg(0, lane, 1) == (complete ? 2u : 1u),
                what + "lane " + std::to_string(lane) + ": the test_wait did not answer " + (complete ? "true" : "false"));
        }
        require(machine.trace().summary().mbar_phase_changes == (complete ? 1u : 0u), what + "phase changes");
    }
}

// Tx arithmetic is checked (an atomic add wraps; this does not). Machine
// PANICs, both sanitizer builds: an operand above 2^20 - 1; a count rising
// above 2^20 - 1 (two arrive.expect_tx of 2^19); a count falling below
// -(2^20 - 1) (two host complete_tx of 2^20 - 1: the first reaches the limit
// exactly, which is legal; the second leaves it).
TEST(Rmw, tx_range_panics) {
    const auto& profile = canonical_profile();
    auto rising = debug_machine(profile, R"(
        umov ur4, 4160
        umov ur2, 524288
        mbar.arrive.expect_tx [ur4], ur2
        mbar.arrive.expect_tx [ur4], ur2
        halt
    )", 1, {{4160, mbar_state(2, 2)}});
    expect_panic(*rising, {"transaction byte count overflow"}, "a tx count above 2^20 - 1");
    auto operand = debug_machine(profile, R"(
        umov ur4, 4160
        umov ur2, 1048576
        mbar.arrive.expect_tx [ur4], ur2
        halt
    )", 1, {{4160, mbar_state(2, 2)}});
    expect_panic(*operand, {"transaction byte count operand exceeds 2^20 - 1"}, "a tx operand of 2^20");
    // The warp stays alive past the first event: kernel end would PANIC on tx != 0.
    auto falling = debug_machine(profile, "mov r9, 0 {pred=PT, stall=31, set=-, wait=-}\nhalt\n", 1,
        {{4160, mbar_state(1, 1)}});
    falling->complete_mbar_tx(4160, 1048575, 0);
    while (falling->read_mbar_state(4160).tx_bytes == 0) tick_capped(*falling);
    require(falling->read_mbar_state(4160).tx_bytes == -1048575, "the count did not reach -(2^20 - 1)");
    while (falling->tma_mbar_event_held()) tick_capped(*falling);
    falling->complete_mbar_tx(4160, 1048575, 0);
    expect_panic(*falling, {"transaction byte count underflow"}, "a tx count below -(2^20 - 1)");
}

// SANITIZER: the TMA reads no mbar at its start; the sanitizer peeks the
// completion mbar's phase when tma.load is accepted and compares it when the
// complete_tx reaches the mbar ("wrong phase"). Two ways a TMA's bytes land
// in a later phase:
//   plain arrive: the mbar expects one arrival and no bytes; the TMA is
//   accepted in phase 0, then a PLAIN mbar.arrive completes phase 0 at once
//   (tx still 0) -- the unsafe case -- and the TMA's complete_tx reaches
//   phase 1;
//   reuse: one arrive.expect_tx of 64 bytes but two 64-byte TMAs, both
//   accepted in phase 0; the first completes the phase, the second's
//   complete_tx reaches phase 1.
TEST(Rmw, sanitizer_wrong_phase_complete_tx) {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    const auto& profile = canonical_profile();
    const auto machine_for = [&](const std::string& program) {
        return std::make_unique<nanosm::Machine>(
            profile, hand_assemble(hand_asm(profile, program), profile), tma_into_an_arrival_mbar());
    };
    // Buffers at 2048 (ur10) and 2112 (ur13), both completing into 4104 (ur11).
    const auto operands = tma_operands_asm(10, 2048, 11, 4104, 8) + "umov ur13, 2112\n";
    auto plain = machine_for(operands + R"(
        umov ur12, 4104
        tma.load.1d [ur10], [5120, ur8], [ur11]
        mbar.arrive [ur12]
        halt
    )");
    expect_panic(*plain, {"wrong phase (tma #0 was accepted in phase 0; the mbar is in phase 1)"},
        "a TMA completing after a plain arrive ended its phase");
    auto reuse = machine_for(operands + R"(
        umov ur12, 4104
        mbar.arrive.expect_tx [ur12], 64
        tma.load.1d [ur10], [5120, ur8], [ur11]
        tma.load.1d [ur13], [5120, ur8], [ur11]
        halt
    )");
    expect_panic(*reuse, {"wrong phase (tma #1 was accepted in phase 0; the mbar is in phase 1)"},
        "a second TMA completing into the reused mbar's next phase");
}

// SANITIZER: a tx count nonzero at kernel end -- a missing complete_tx (tx 64
// left) or a missing expect_tx (tx -32 left) -- would hang a waiter: PANIC at
// the end of the run.
TEST(Rmw, sanitizer_nonzero_tx_at_kernel_end) {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    const auto& profile = canonical_profile();
    auto missing_complete = debug_machine(profile, "umov ur4, 4160\nmbar.arrive.expect_tx [ur4], 64\nhalt\n", 1,
        {{4160, mbar_state(2, 2)}});
    expect_panic(*missing_complete, {"transaction count is nonzero at kernel end (tx=64"}, "a missing complete_tx");
    auto missing_expect = debug_machine(profile, "halt\n", 1, {{4160, mbar_state(1, 1)}});
    missing_expect->complete_mbar_tx(4160, 32, 0);
    expect_panic(*missing_expect, {"transaction count is nonzero at kernel end (tx=-32"}, "a missing expect_tx");
}

// ONE word per warp instruction, the update applied ONCE (the operand is ur5
// = 0x5a5: add adds it once, not per lane), every lane's rD gets the old
// word. The atomic reaches the SIMT RMW register at its dispatch and starts
// the cycle after (nothing else holds A-read), reading then; its write is
// granted D + Lb + C later and its slice ends at that write + 1. The old word
// is at the port output at a = the read + D + Lb; rD's RF write is granted at
// a + 1, BEFORE the word's write grant a + C, and lands Wd later. The warp's
// next shmem access (an lds.uni of the word) is admitted only at the write
// grant w, granted at its dispatch + 1, and reads the new word. Each op at the
// canonical D = Lb = C = 2 (7 cycles from start to the end of the pause); add
// also at D = 3, C = 3.
TEST(Rmw, one_word_timeline) {
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
    for (const auto& [opcode, name] : atomic_ops) cases.push_back({opcode, name, 2, 2});
    cases.push_back({nanosm::Opcode::atom_add_uni, "atom.add.uni", 3, 3});
    for (const auto& test : cases) {
        auto profile = canonical_profile();
        profile.shmem_port_a_depth = test.d;
        profile.shmem_rmw_compute_cycles = test.c;
        const auto what = std::string(test.name) + " D=" + std::to_string(test.d) + " C=" + std::to_string(test.c);
        auto machine = gmem_debug_machine(profile, program_head + test.name + program_tail);
        const auto written = watch_registers(*machine, {7});
        const auto atomic = one_atomic(*machine, what);
        const auto t = nanosm::issue_to_dispatch(profile);
        const auto issue = issued_at(*machine, 16).at(0);
        require(atomic.start == issue + t + 1, what + ": did not start the cycle after its dispatch");
        require_atomic_timeline(atomic, profile, atomic.start, what);
        const auto a = atomic.reads.at(0) + profile.shmem_port_a_depth + profile.shmem_read_latency_cycles;
        const auto w = atomic.writes.at(0);
        require(written.at(7) == std::vector<uint64_t>{a + 1 + profile.rf_write_depth_cycles} && a + 1 < w,
            what + ": rD's RF write was not granted at a + 1 (c" + std::to_string(a + 1) + "), before the write grant c" +
                std::to_string(w) + ", landing Wd later");
        require(issued_at(*machine, 24) == std::vector<uint64_t>{w}, what + ": the lds.uni was not admitted at the write grant");
        const auto lds = grants_of(*machine, "A-read");
        require(lds.size() == 2 && lds[1] == Grant{w + t + 1, "A-read", "simt", lds[1].slot},
            what + ": the lds.uni's read was not granted at w + issue_to_dispatch + 1:" + describe(lds));
        const uint32_t updated = atomic_result(test.opcode, 0x300, 0x5a5);
        require(machine->read_shmem_u32(2048) == updated, what + ": the word was not updated once");
        for (uint32_t lane = 0; lane < 16; ++lane) {
            require(machine->read_reg(0, lane, 7) == 0x300 && machine->read_reg(0, lane, 8) == updated,
                what + ": lane " + std::to_string(lane) + " rD or the lds.uni");
        }
        require(atomic.detail == std::string(test.name) + ", warp=0", what + ": slice detail " + atomic.detail);
        if (test.d == 2 && test.c == 2) require(atomic.end - atomic.start == 7, what + ": not 7 cycles canonical");
    }
}

// int32 add WRAPS (defined 32-bit two's-complement wrap, unlike the checked
// mbar tx count): one atomic each,
//   0x7fffffff + 1 = 0x80000000      0x80000000 + (-1) = 0x7fffffff
//   0xffffffff + 1 = 0               0x80000000 + 0x80000000 = 0
//   0x7fffffff + 0x7fffffff = 0xfffffffe
// and a chain across the boundary on one word: 0x7ffffffe + 1 + 1 + 1 + 1,
// each atomic reading the previous sum. Operands come from a table: lds.uni
// into vector registers, then r2ur into the uniform operand registers
// ur2..ur6 (the first r2ur waits for the loads).
TEST(Rmw, add_wraps_at_the_int32_boundary) {
    const auto& profile = canonical_profile();
    const std::array<std::pair<uint32_t, uint32_t>, 5> cases = {{
        {0x7fffffffu, 1u}, {0x80000000u, 0xffffffffu}, {0xffffffffu, 1u}, {0x80000000u, 0x80000000u},
        {0x7fffffffu, 0x7fffffffu},
    }};
    auto machine = gmem_debug_machine(profile, R"(
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
    )", 1, [&](std::vector<uint8_t>& shmem) {
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

// An atomic reserves its RETURN load slot at admission; with none free it is
// not admitted (structural, counted as a load-slot wait), so it never takes
// the store pause with nowhere to put its old value. One load slot: the lds,
// issued at l (it needs the shli's r1: 2 alu_distance), is granted at G = l +
// issue_to_dispatch + 1 and holds the slot until its RF write is granted at
// G + D + Lb + 1. The atomic, first tried at l + 1 (its ur3 is visible then),
// is ordered behind the unserviced lds until G, then refused "load slots
// full" G .. G + D + Lb + 1 and admitted the cycle after, when issue sees the
// slot empty.
TEST(Rmw, return_slot_admission) {
    auto profile = canonical_profile();
    profile.shmem_load_slots = 1;
    const uint64_t back = profile.shmem_port_a_depth + profile.shmem_read_latency_cycles;
    auto machine = gmem_debug_machine(profile, R"(
        sread r0, LANE_ID
        shli r1, r0, 2
        umov ur2, 0
        umov ur3, 5
        lds r6, [r1 + 256] {pred=PT, stall=0, set=sb1, wait=-}
        atom.add.uni r7, [ur2 + 1024], ur3 {pred=PT, stall=0, set=sb0, wait=-}
        halt {pred=PT, stall=0, set=-, wait=sb0|sb1}
    )");
    machine->run();
    const auto g = grants_of(*machine, "A-read").at(0);
    const auto t = nanosm::issue_to_dispatch(profile);
    const auto l = nanosm::fetch_to_issue + 2 * alu_distance(profile);
    require(g.client == "simt" && issued_at(*machine, 32) == std::vector<uint64_t>{l} && g.cycle == l + t + 1,
        "the lds was not granted the cycle after its dispatch:" + issue_log(*machine));
    require(refusals(*machine, 0, "ordered behind own access") == std::vector<std::pair<uint64_t, uint64_t>>{{l + 1, t}},
        "the atomic was not ordered behind the lds until its grant:" +
            describe(refusals(*machine, 0, "ordered behind own access")));
    require(refusals(*machine, 0, "load slots full") == std::vector<std::pair<uint64_t, uint64_t>>{{g.cycle, back + 2}} &&
            issued_at(*machine, 40) == std::vector<uint64_t>{g.cycle + back + 2},
        "the atomic was not refused for its return slot c" + std::to_string(g.cycle) + " .. c" +
            std::to_string(g.cycle + back + 1) + " and admitted at c" + std::to_string(g.cycle + back + 2) + ":" +
            describe(refusals(*machine, 0, "load slots full")));
    require(machine->trace().summary().shmem_load_slot_waits == back + 2, "the return-slot refusals were not counted");
    const auto atomic = one_atomic(*machine, "after the refusal");
    require_atomic_timeline(atomic, profile, atomic.start, "after the refusal");
    require(machine->read_shmem_u32(1024) == 0x205, "the atomic's word");
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(machine->read_reg(0, lane, 6) == 0x100 + 64 + lane && machine->read_reg(0, lane, 7) == 0x200,
            "lane " + std::to_string(lane) + ": the lds or the atomic returned the wrong word");
    }
}

// The SIMT RMW register holds ONE operation machine-wide. Three warps, one
// atomic each on one word (operand 1): the first runs in the engine, the
// second waits in the register, the third is refused "SIMT RMW register
// busy" until the register empties. All three apply, one after another (each
// reads the previous sum), and the engine runs them back to back with no
// overlap.
TEST(Rmw, simt_rmw_register_busy) {
    auto profile = canonical_profile();
    profile.shmem_load_slots = 3; // a return slot for each: the third waits only for the register
    auto machine = gmem_debug_machine(profile, R"(
        umov ur1, 0
        umov ur5, 1
        atom.add.uni r7, [ur1 + 2048], ur5 {pred=PT, stall=0, set=sb0, wait=-}
        halt {pred=PT, stall=0, set=-, wait=sb0}
    )", 3);
    machine->run();
    const auto atomics = atomics_of(*machine);
    require(atomics.size() == 3, "expected three atomics");
    for (std::size_t k = 1; k < atomics.size(); ++k) {
        require(atomics[k].start >= atomics[k - 1].end, "two atomics overlapped in the engine:" + describe(atomics[k]));
    }
    uint32_t busy = 0;
    for (uint32_t warp = 0; warp < 3; ++warp) {
        busy += static_cast<uint32_t>(refusals(*machine, warp, "SIMT RMW register busy").size());
    }
    require(busy != 0, "no warp was refused for the busy SIMT RMW register");
    std::set<uint32_t> old_values;
    for (uint32_t warp = 0; warp < 3; ++warp) old_values.insert(machine->read_reg(warp, 0, 7));
    require(old_values == std::set<uint32_t>{0x300, 0x301, 0x302} && machine->read_shmem_u32(2048) == 0x303,
        "the three atomics did not apply one after another");
}

// An atomic is a WRITE for its warp's order and counts as unserviced until
// its write grant.
//   sts B; atom B: the atomic waits for the sts's grant and reads its word;
//   atom B; lds B: the lds waits until the atomic's write grant w (admitted
//     at w) and reads the result in lane 0's word;
//   lds A; atom A: the atomic waits for the lds's grant; the lds reads the
//     old word (and so does the atomic: it applies after);
//   atom A; fence.sh: admitted at the atomic's write grant + D.
// Words B: 384 + l (sixteen banks, one round each); A: 256 + l. Three load
// slots, so atom A's return slot is free while both lds hold theirs: only the
// warp's order decides.
TEST(Rmw, same_warp_order) {
    auto profile = canonical_profile();
    profile.shmem_load_slots = 3;
    auto machine = gmem_debug_machine(profile, R"(
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
    const auto atomics = atomics_of(*machine);
    require(atomics.size() == 2, "expected two atomics");
    std::vector<Grant> simt;
    for (const auto& grant : grants_of(*machine)) {
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
    require(machine->read_shmem_u32(1536) == 0x1013 && machine->read_shmem_u32(1024) == 0x203, "the atomics' words");
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(machine->read_reg(0, lane, 11) == 0x1010 && machine->read_reg(0, lane, 7) == 0x200,
            "lane " + std::to_string(lane) + ": an atomic's old value");
        require(machine->read_reg(0, lane, 8) == (lane == 0 ? 0x1013u : 0x1010u) &&
                machine->read_reg(0, lane, 6) == 0x200 + lane,
            "lane " + std::to_string(lane) + ": an lds around the atomics");
    }
}

// SANITIZER: two lanes of one sts naming one word is UB (as on NVIDIA): the
// sanitizer PANICs at issue, under a full and under a partial mask. Lane l
// stores l + 1 to word 0.
TEST(Rmw, sts_two_lanes_on_one_word) {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    for (const uint64_t mask : {uint64_t{0xffff}, uint64_t{0x00f6}}) {
        auto launch = launch_warps(1);
        image_bytes(launch.test_shmem, 4);
        nanosm::Machine machine(canonical_profile(), compile(R"(
            sread r1, LANE_ID
            addi r2, r1, 1
            mov r0, 0
        )" + masked_region(mask, "sts [r0], r2") + "halt\n"), std::move(launch));
        expect_panic(machine, {"multiple lanes write the same shmem word"}, "same-word sts, mask " + std::to_string(mask));
    }
}

// EVERY uniform operation -- each instruction with a uniform-register or
// lane-0-sourced operand, and bra.uni in every form -- is UB under a partial
// active mask: the SANITIZER PANICs naming it; the machine runs quietly (what
// it does then is no contract; a program with a partial mask uses the
// per-lane instruction). One program per uniform opcode family, each of which
// runs to completion under a full mask, run under 0xfffe (lane 0 inactive):
// the named opcode is the program's first uniform operation (the umov /
// usread that set up its uniform registers are uniform arithmetic, which the
// mask check does not cover).
TEST(Rmw, every_uniform_op_under_a_partial_mask) {
    const auto& profile = canonical_profile();
    const auto setup = [](nanosm::Launch& launch) {
        auto& shmem = image_bytes(launch.test_shmem, 8192);
        write_mbar(shmem, 4096, mbar_state(1, 1));
        write_tma_descriptor(shmem, 5120, plain_tma_1d(16));
        write_mbar(shmem, 4104, mbar_state(1, 0, 64));
        image_bytes(launch.gmem_image, 4096);
    };
    const auto tma_operands = one_tma_operands(1024);
    const std::string mma_tail = "umov ur4, 4160\nmbar.init [ur4], 1\nmma.commit [ur4]\n" + wait_on("ur4") + "halt\n";
    const std::vector<std::pair<std::string, std::string>> cases = {
        {"lds.uni", "umov ur0, 0\nlds.uni r1, [ur0]\nhalt\n"},
        {"ldg.uni", "umov ur0, 0\nldg.uni r1, [ur0]\nhalt\n"},
        {"mbar.init", "umov ur4, 4160\nmbar.init [ur4], 1\nhalt\n"},
        {"mbar.arrive", "umov ur4, 4096\nmbar.arrive [ur4]\nhalt\n"},
        {"mbar.arrive.expect_tx", "umov ur4, 4096\nmbar.arrive.expect_tx [ur4], 0\nhalt\n"},
        {"mbar.test_wait", "umov ur4, 4096\nmbar.test_wait p0, [ur4], 0\nhalt\n"},
        {"mbar.try_wait", "umov ur4, 4096\nmbar.try_wait p0, [ur4], 0\nhalt\n"},
        {"atom.add.uni", "umov ur0, 0\numov ur1, 1\natom.add.uni r2, [ur0], ur1\nhalt\n"},
        {"tma.load.1d", tma_operands + "umov ur4, 4104\ntma.load.1d [ur10], [5120, ur8], [ur11]\n" + wait_on("ur4") +
                            "halt\n"},
        {"tma.store.1d", tma_operands + "umov ur4, 4104\ntma.store.1d [5120, ur8], [ur10], [ur11]\n" +
                             wait_on("ur4") + "halt\n"},
        {"mma.async.init.m64n64k8",
         "umov ur0, 0\numov ur1, 1024\numov ur2, 0\nmma.async.init.m64n64k8 [ur2], [ur0], [ur1]\n" + mma_tail},
        {"mma.async.m64n64k8",
         "umov ur0, 0\numov ur1, 1024\numov ur2, 0\nmma.async.m64n64k8 [ur2], [ur0], [ur1]\n" + mma_tail},
        {"mma.async.init.m32n64k8",
         "umov ur0, 0\numov ur1, 1024\numov ur2, 0\nmma.async.init.m32n64k8 [ur2], [ur0], [ur1]\n" + mma_tail},
        {"mma.async.m32n64k8",
         "umov ur0, 0\numov ur1, 1024\numov ur2, 0\nmma.async.m32n64k8 [ur2], [ur0], [ur1]\n" + mma_tail},
        {"mma.commit", "umov ur4, 4096\nmma.commit [ur4]\nhalt\n"},
        {"acc.ld", "umov ur8, 0\nacc.ld r10, [ur8, 64]\nhalt\n"},
        {"acc.st.v8", "umov ur8, 0\nmov r0, 1\nacc.st.v8 [ur8, 64], r0\nhalt\n"},
        {"r2ur", "sread r0, LANE_ID\nr2ur ur1, r0\nhalt\n"}, // reads lane 0 of a vector register
        {"bra.uni", "usread ur0, WARP_ID\nur2r r0, ur0\nsetp.eq.u32 p0, r0, 0\n@p0 bra.uni out\nmov r1, 1\nout:\nhalt\n"},
        {"bra.uni", "bra.uni out\nmov r1, 1\nout:\nhalt\n"}, // @PT: no operand, still uniform
    };
    for (const auto& [name, program] : cases) {
        for (const uint64_t mask : {uint64_t{0xffff}, uint64_t{0xfffe}}) {
            const auto what = name + " under mask " + std::to_string(mask) + ": " + program;
            auto launch = launch_warps(1);
            setup(launch);
            // The partial mask: the program, less its final halt, runs inside
            // a divergent region (lane 0 waits on the stack); the warp halts
            // after the region (a halt inside it is the sanitizer's own PANIC).
            const auto body = program.substr(0, program.rfind("halt\n"));
            const auto text = mask == 0xffff ? program : masked_region(mask, body) + "halt\n";
            nanosm::Machine machine(profile, compile(text, profile), std::move(launch));
#if NANOSM_SANITIZER
            if (mask != 0xffff) {
                expect_panic(machine, {name + " is a uniform operation under a partial active mask"}, what);
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

// SANITIZER: an atomic writing a LIVE mbar PANICs at its write grant (using
// the RMW engine does not authorize it) -- an mbar made live by a SIMT
// mbar.init, and one the launch declared (live from launch). The
// launch-declared mbar is opaque to an sts and to an mbar.init as well, and
// stays usable by mbar operations. The atomic's other sanitizer checks:
// issued over a TMA load's destination it is a race; its word still
// travelling under a TMA store of it, without a fence.sh, a missing fence
// (with one: clean, and gmem gets its result).
TEST(Rmw, atomic_sanitizer) {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    const auto& profile = canonical_profile();
    const auto machine = [&](const std::string& program, const std::function<void(std::vector<uint8_t>&)>& setup = {},
                             const std::vector<uint32_t>& declared = {}) {
        return gmem_debug_machine(profile, compile(program, profile), 1, setup, declared);
    };
    expect_panic(*machine("umov ur4, 4160\nmbar.init [ur4], 1\numov ur11, 9\natom.add.uni r7, [ur4], ur11\nhalt\n"),
        {"mbar address=4160 reason=live mbar opacity: atom.add.uni r7, [ur4], ur11 (warp 0 pc ",
            "writes bytes [4160, 4164) of the live mbar [4160, 4168) (mbar.init by warp 0 pc 8 at cycle ",
            "until kernel end only mbar operations may touch it"},
        "an atomic on a SIMT-initialized mbar");

    const auto declared_mbar = [](std::vector<uint8_t>& shmem) { write_mbar(shmem, 4160, mbar_state(1, 1)); };
    const std::vector<uint32_t> declared = {4160};
    expect_panic(*machine("umov ur4, 4164\numov ur11, 9\natom.exch.uni r7, [ur4], ur11\nhalt\n", declared_mbar, declared),
        {"mbar address=4160 reason=live mbar opacity: atom.exch.uni r7, [ur4], ur11 (warp 0 pc ",
            "writes bytes [4164, 4168) of the live mbar [4160, 4168) (declared by the launch image)"},
        "an atomic on a launch-declared mbar");
    expect_panic(*machine("mov r4, 4160\nsread r20, LANE_ID\nshli r20, r20, 2\nadd r20, r20, r4\n"
                          "mov r21, 9\nsts [r20], r21\nhalt\n", declared_mbar, declared),
        {"reason=live mbar opacity: sts writes bytes [4160, 4168) of the live mbar [4160, 4168) "
         "(declared by the launch image)"},
        "an sts on a launch-declared mbar");
    expect_panic(*machine("umov ur4, 4160\nmbar.init [ur4], 1\nhalt\n", declared_mbar, declared),
        {"reason=mbar.init over the live mbar [4160, 4168) (declared by the launch image)"},
        "an mbar.init over a launch-declared mbar");
    auto legal = machine("umov ur4, 4160\nmbar.arrive [ur4]\n" + wait_on("ur4") + "halt\n", declared_mbar, declared);
    legal->run();
    require(legal->read_mbar_state(4160).phase, "an arrive on a launch-declared mbar did not complete its phase");

    const auto tma = [](std::vector<uint8_t>& shmem) {
        write_tma_descriptor(shmem, 5120, plain_tma_1d(16));
        write_mbar(shmem, 4104, mbar_state(1, 0, 64));
    };
    const std::string tail = wait_on("ur4") + "halt\n";
    // The load's buffer is at 1024, the store's at 0 (ur12); both complete
    // into 4104 (ur4, also the wait's); the coordinate is ur13. The atomics
    // use ur10 and ur11.
    const auto prologue = [](uint32_t shared) { return tma_operands_asm(12, shared, 4, 4104, 13); };
    const std::string tma_load = "tma.load.1d [ur12], [5120, ur13], [ur4]\n";
    const std::string tma_store = "tma.store.1d [5120, ur13], [ur12], [ur4]\n";
    expect_panic(*machine(prologue(1024) + tma_load + "umov ur10, 1032\numov ur11, 1\natom.add.uni r7, [ur10], ur11\n" +
                     tail, tma),
        {"reason=shmem race: atom.add.uni writes 1 word, bytes [1032, 1036) owned by tma load #0"},
        "an atomic over a TMA load's destination");
    const std::string atomic = "umov ur10, 8\numov ur11, 0x1000\natom.add.uni r7, [ur10], ur11\n";
    expect_panic(*machine(prologue(0) + atomic + tma_store + tail, tma),
        {"reason=missing fence.sh: tma store #0 source bytes [",
            "overlap this warp's earlier atom.add.uni r7, [ur10], ur11 (pc ",
            "whose write has not reached the RAM; issue fence.sh between them"},
        "a TMA store of an atomic's word without fence.sh");
    auto fenced = machine(prologue(0) + atomic + "fence.sh\n" + tma_store + tail, tma);
    fenced->run();
    for (uint32_t i = 0; i < 16; ++i) {
        require(fenced->read_gmem_u32(4 * i) == (i == 2 ? 0x1102u : 0x100u + i),
            "the fenced TMA store did not carry the atomic's result");
    }
}

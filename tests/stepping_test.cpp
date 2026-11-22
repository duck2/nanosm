// The stepping discipline: Reg<T>, the LSU assembly contract, simultaneous events, the driver shuffle.
// Moved from the original tests/tests.cpp (trash/test-setup-2026-10-22.md);
// each test's body is unchanged, its TEST wrapper is at the end.
#include "test_support.hpp"

namespace {

using namespace nanosm_test;

// ===========================================================================
// Stepping discipline (trash/stepping-discipline-2026-10-08.md)
// ===========================================================================

// ---- Reg<T> (src/reg.hpp) ----

void test_reg_hold_write_commit() {
    nanosm::Sanitizer sanitizer;
    uint64_t cycle = 7;
    nanosm::RegisterFile file(sanitizer, cycle);
    nanosm::Reg<int> reg(file, "test register", 5);
    sanitizer.record_cycle_start();
    file.commit();
    require(reg.cur() == 5, "a register nobody wrote did not hold its value");
    sanitizer.record_cycle_start();
    reg.set_next(9, "writer");
    require(reg.cur() == 5, "set_next() changed Q before the clock edge");
    file.commit();
    require(reg.cur() == 9, "commit did not publish D");
    sanitizer.record_cycle_start();
    file.commit();
    require(reg.cur() == 9, "the tick after a write did not hold");
}

// A second write in one tick is a PANIC naming the register, the cycle and
// both writers, even when the same writer writes twice.
void test_reg_duplicate_write_panics() {
    nanosm::Sanitizer sanitizer;
    uint64_t cycle = 7;
    nanosm::RegisterFile file(sanitizer, cycle);
    nanosm::Reg<int> reg(file, "test register", 0);
    const auto expect_panic = [&](const char* first, const char* second) {
        sanitizer.record_cycle_start();
        reg.set_next(1, first);
        try {
            reg.set_next(2, second);
        } catch (const nanosm::SimError& error) {
            const std::string message = error.what();
            for (const std::string needle : {std::string("register 'test register'"), std::string("cycle=7"),
                     std::string("first by '") + first + "'", std::string("then by '") + second + "'"}) {
                require(message.find(needle) != std::string::npos, "duplicate-write PANIC lacks '" + needle + "': " + message);
            }
            file.commit();
            return;
        }
        throw std::runtime_error("a second write in one tick did not panic");
    };
    expect_panic("drain", "fill");
    expect_panic("drain", "drain"); // the same writer twice
    sanitizer.record_cycle_start();
    reg.set_next(3, "next tick");    // a new tick: one write is fine again
    file.commit();
    require(reg.cur() == 3, "the register did not recover after a duplicate-write PANIC");
}

// ---- The load buffer's groups (2026-11-21: per-lane ldg.v8) ----

// A ldg.v8 whose every lane names the same sector (a uniform base): register
// first_reg + k holds that sector's word k in every lane.
void require_vector_data(const nanosm::Machine& machine, uint32_t warp, uint32_t first_reg, uint32_t first_word,
    const std::string& what) {
    for (uint32_t k = 0; k < 8; ++k)
        for (uint32_t lane = 0; lane < 16; ++lane)
            require(machine.read_reg(warp, lane, first_reg + k) == pattern_word(first_word + k),
                what + ": r" + std::to_string(first_reg + k) + " lane " + std::to_string(lane) +
                    " does not hold word " + std::to_string(k) + " of its sector");
}

// One ldg.v8 over sixteen per-lane sectors at the canonical profile (groups
// of four lanes): four groups, each written by eight RF writes (r8 .. r15,
// one a cycle at most), none before the group's last word is in the buffer,
// and no word of the next group in the buffer before the previous group's
// last write is granted (the buffer is reused in that cycle at the earliest).
void test_load_buffer_groups() {
    const auto config = canonical_profile();
    nanosm::Launch launch;
    launch.active_warps = 1;
    fill_pattern(image_bytes(launch.gmem_image, 4096), 0, 1024, 0);
    nanosm::Machine machine(config,
        compile("sread r1, LANE_ID\nshli r0, r1, 5\nldg.v8 r8, [r0]\nhalt\n", config), std::move(launch));
    machine.enable_edge_probe();
    machine.run();
    const uint32_t group_lanes = config.lsu_load_group_lanes;
    const uint32_t groups = 16 / group_lanes;
    std::vector<uint64_t> requests; // in arrival order = lane order
    std::map<uint64_t, uint64_t> first_word, last_word;
    std::vector<nanosm::LsuEdgeEvent> writes;
    for (const auto& event : machine.edge_events()) {
        if (event.edge == LsuEdge::assembly_write) {
            if (first_word.emplace(event.aux, event.cycle).second) requests.push_back(event.aux);
            last_word[event.aux] = event.cycle;
        }
        if (event.edge == LsuEdge::rf_write_beat) writes.push_back(event);
    }
    require(requests.size() == 16, "expected sixteen sectors, one per lane");
    require(writes.size() == 8 * groups, "expected eight RF writes per lane group");
    for (std::size_t k = 1; k < writes.size(); ++k)
        require(writes[k].cycle > writes[k - 1].cycle, "the LSU wrote two RF beats in one cycle");
    for (uint32_t group = 0; group < groups; ++group) {
        uint64_t complete = 0;
        for (uint32_t lane = group * group_lanes; lane < (group + 1) * group_lanes; ++lane)
            complete = std::max(complete, last_word.at(requests[lane]));
        for (uint32_t j = 0; j < 8; ++j) {
            const auto& write = writes[8 * group + j];
            require(write.aux == 8 + j, "a group's RF writes are not r8 .. r15 in order");
            require(write.cycle > complete, "a group was written to the RF before its last word was in the buffer");
        }
        if (group + 1 != groups) {
            require(first_word.at(requests[(group + 1) * group_lanes]) >= writes[8 * group + 7].cycle,
                "the next group's word entered the buffer before the previous group's last RF write");
        }
    }
    for (uint32_t lane = 0; lane < 16; ++lane)
        for (uint32_t j = 0; j < 8; ++j)
            require(machine.read_reg(0, lane, 8 + j) == pattern_word(8 * lane + j),
                "r" + std::to_string(8 + j) + " lane " + std::to_string(lane) + " is not word j of the lane's sector");
}

// ---- Same-cycle reuse after a load's final RF drain (staging = active = 1) ----

// One of the LSU's two buffers at the end of a cycle.
struct AssemblyAfter {
    std::optional<uint64_t> owner;
    std::vector<uint8_t> valid;
    uint32_t valid_count = 0;
};

struct ReuseRun {
    std::unique_ptr<nanosm::Machine> machine;
    uint64_t reuse_cycle = 0; // the second command's admission (dispatch)
    AssemblyAfter after;      // the load buffer after that cycle
    AssemblyAfter store_after; // the store buffer after that cycle
    std::map<uint64_t, AssemblyAfter> store_by_cycle; // the store buffer after every cycle
};

// `old_load` is LSU command 0, `second` (opcode `second_name`) command 1. With
// one command slot and one assembly register, the second waits for both: it
// issues in the cycle the load's final RF write releases them (look-ahead
// (c)) and the LSU admits it issue_to_dispatch later (I6a), into the register
// free since then. (Before I6a the admission was that same cycle, a reuse
// (d) of the register the drain was emptying; a claim now waits for the
// release, so issue cannot line the two up.)
// gmem holds pattern words; r5 = 4 * lane, r6 = 2048 + 4 * lane, r24..r31 =
// 101..108.
ReuseRun run_reuse_after_final_drain(const std::string& old_load, const std::string& second,
    const std::string& second_name) {
    auto config = fast_bus_profile(2);
    config.lsu_active_commands = 1;
    nanosm::Launch launch;
    launch.active_warps = 1;
    fill_pattern(image_bytes(launch.gmem_image, 4096), 0, 1024, 0);
    // r2: 2048 + 32 x lane, a stg.v8's per-lane sectors; r0 = 0 for every
    // lane, so the old ldg.v8 loads the sector at 0 into every lane.
    std::string text = "sread r4, LANE_ID\nshli r5, r4, 2\nori r6, r5, 2048\nmov r0, 0\nshli r2, r4, 5\naddi r2, r2, 2048\n";
    for (uint32_t j = 0; j < 8; ++j) text += "mov r" + std::to_string(24 + j) + ", " + std::to_string(101 + j) + "\n";
    text += old_load + "\n" + second + "\nhalt\n";
    ReuseRun run;
    run.machine = std::make_unique<nanosm::Machine>(
        config, compile(text, config), std::move(launch), nanosm::TraceOptions{.timeline = true});
    auto& machine = *run.machine;
    machine.enable_edge_probe();
    std::map<uint64_t, AssemblyAfter> after;       // the load buffer
    std::map<uint64_t, AssemblyAfter> store_after; // the store buffer
    while (!machine.done()) {
        const auto cycle = machine.cycle();
        tick_capped(machine);
        after[cycle] = {machine.assembly_owner(0), machine.assembly_valid(0), machine.assembly_valid_count(0)};
        store_after[cycle] = {machine.assembly_owner(1), machine.assembly_valid(1), machine.assembly_valid_count(1)};
    }
    uint64_t final_drain = 0;
    for (const auto& event : machine.edge_events())
        if (event.edge == LsuEdge::rf_write_beat && event.id == 0) final_drain = std::max(final_drain, event.cycle);
    std::optional<uint64_t> admitted;
    for (const auto& event : machine.trace().events())
        if (event.category == nanosm::TraceCategory::gmem && event.name == second_name && event.id == 1)
            admitted = event.start_cycle;
    const auto dispatch = final_drain + nanosm::issue_to_dispatch(config);
    require(admitted == dispatch,
        second_name + " was not admitted issue_to_dispatch after '" + old_load + "' made its final RF write");
    run.reuse_cycle = dispatch;
    run.after = after.at(dispatch);
    run.store_after = store_after.at(dispatch);
    run.store_by_cycle = std::move(store_after);
    return run;
}

void require_old_load_data(const nanosm::Machine& machine, const std::string& old_load) {
    if (old_load.starts_with("ldg.v8")) {
        require_vector_data(machine, 0, 8, 0, old_load);
        return;
    }
    for (uint32_t lane = 0; lane < 16; ++lane)
        require(machine.read_reg(0, lane, 8) == pattern_word(lane), old_load + ": lane " + std::to_string(lane));
}

// A scalar store admitted after a load's final drain: after its dispatch
// cycle it owns the STORE buffer with all sixteen lanes valid, and the load
// buffer is empty and unowned (2026-11-21: one buffer per direction, owned
// only while data is in it). The output alone cannot show it: scalar-store
// egress reads the words without valid_count.
void test_assembly_reuse_load_to_scalar_store() {
    for (const std::string old_load : {"ldg.v8 r8, [r0]", "ldg r8, [r5]"}) {
        const auto run = run_reuse_after_final_drain(old_load, "stg [r5 + 2048], r6", "stg");
        const auto where = " after the reuse cycle " + std::to_string(run.reuse_cycle) + " (" + old_load + ")";
        require(run.store_after.owner == 1u, "the store does not own the store buffer" + where);
        require(run.store_after.valid_count == 16 &&
                std::count(run.store_after.valid.begin(), run.store_after.valid.begin() + 16, 1) == 16,
            "the store's valid bits were lost" + where);
        require(!run.after.owner.has_value() && run.after.valid_count == 0,
            "the load buffer is still owned after the load's final drain" + where);
        for (uint32_t lane = 0; lane < 16; ++lane)
            require(run.machine->read_gmem_u32(2048 + 4 * lane) == 2048 + 4 * lane,
                "the scalar store wrote the wrong word for lane " + std::to_string(lane));
        require_old_load_data(*run.machine, old_load);
    }
}

// A stg.v8 admitted then: its first RF read beat is sampled in its admission
// cycle (bypass (e)) and fills the store buffer Lr later (I6a); the buffer has
// no owner until that data lands.
void test_assembly_reuse_load_to_vector_store() {
    const std::string old_load = "ldg.v8 r8, [r0]";
    const auto run = run_reuse_after_final_drain(old_load, "stg.v8 [r2], r24", "stg.v8");
    bool first_read_now = false;
    for (const auto& event : run.machine->edge_events())
        if (event.edge == LsuEdge::rf_read_beat && event.id == 1 && event.cycle == run.reuse_cycle) first_read_now = true;
    require(first_read_now, "the stg.v8's first RF read was not in the reuse cycle; the test proves little");
    require(!run.store_after.owner.has_value() && run.store_after.valid_count == 0,
        "the store buffer is owned before the stg.v8's first beat has landed");
    const auto filled = run.store_by_cycle.at(run.reuse_cycle + canonical_profile().rf_read_latency_cycles);
    // The first beat brings r24 for the first group's lanes: one word a lane.
    require(filled.owner == 1u && filled.valid_count == canonical_profile().lsu_store_group_lanes,
        "the stg.v8's first read beat is not in the store buffer Lr after the reuse cycle");
    for (uint32_t j = 0; j < 8; ++j)
        for (uint32_t lane = 0; lane < 16; ++lane)
            require(run.machine->read_gmem_u32(2048 + 32 * lane + 4 * j) == 101 + j,
                "the stg.v8 wrote the wrong word: r" + std::to_string(24 + j) + " lane " + std::to_string(lane));
    require_old_load_data(*run.machine, old_load);
}

// A new load admitted then: the load buffer is empty and unowned until its
// first word returns. (Assembly contract 5 covers ldg.v8 -> ldg.v8 through
// the data path.)
void test_assembly_reuse_load_to_load() {
    const std::string old_load = "ldg.v8 r8, [r0]";
    const auto run = run_reuse_after_final_drain(old_load, "ldg r16, [r5 + 1024]", "ldg");
    require(!run.after.owner.has_value(), "the load buffer is owned before the new load's first word returned");
    require(run.after.valid_count == 0,
        "the load buffer is not empty after the old load's final drain");
    for (uint32_t lane = 0; lane < 16; ++lane)
        require(run.machine->read_reg(0, lane, 16) == pattern_word(256 + lane),
            "the new load returned the wrong word for lane " + std::to_string(lane));
    require_old_load_data(*run.machine, old_load);
}

// The sanitizer's guard for the case above: a reader of an assembly register
// that takes a lane whose valid bit is clear panics.
void test_sanitizer_assembly_reader_takes_valid_lanes() {
    nanosm::Sanitizer sanitizer;
    std::vector<uint8_t> valid(16, 1);
    sanitizer.check_assembly_lanes_valid(5, 0, valid, 0, 16, "RF drain");
    valid[3] = 0;
    sanitizer.check_assembly_lanes_valid(5, 0, valid, 4, 12, "egress write port"); // lanes 4..15 only
    try {
        sanitizer.check_assembly_lanes_valid(5, 0, valid, 3, 1, "egress write port");
    } catch (const nanosm::SimError& error) {
        const std::string message = error.what();
        require(message.find("egress write port read lane 3 of assembly register 0") != std::string::npos,
            "the lane check's PANIC does not name the reader, lane and register: " + message);
        return;
    }
    throw std::runtime_error("a read of an invalid assembly lane did not panic");
}

// ---- The mbar event registers: both engines alike ----

// A TMA load and an MMA commit, far apart: each engine's event, presented in
// cycle p, is first claimable in p + 1, and the free RMW engine claims it
// then. (Before 2026-10-08 a TMA event could be claimed in p itself, an
// accident of the call order in tick().)
void test_mbar_engine_events_visible_next_cycle() {
    constexpr uint32_t commit_mbar = 4096;
    constexpr uint32_t tx_mbar = 4104;
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.test_shmem, 8192);
    auto& gmem = image_bytes(launch.gmem_image, 64);
    for (uint32_t i = 0; i < 16; ++i) write_u32(gmem, 4 * i, 0x100u + i);
    write_mbar(shmem, commit_mbar, 1, 0);
    write_mbar(shmem, tx_mbar, 0, 64);
    write_tma_descriptor(shmem, 6144, plain_tma_1d(16));
    std::string text = "umov ur2, 0\numov ur0, 0\numov ur1, 1024\numov ur4, 4096\n" +
        tma_operands_asm(10, 2048, 11, tx_mbar, 8) + "tma.load.1d [ur10], [6144, ur8], [ur11]\n";
    for (uint32_t i = 0; i < 300; ++i) text += "mov r" + std::to_string(12 + i % 8) + ", " + std::to_string(i) + "\n";
    text += "mma.async.init.m64n64k8 [ur2], [ur0], [ur1]\nmma.commit [ur4]\nhalt\n";
    nanosm::Machine machine(canonical_profile(), compile(text), std::move(launch), {.timeline = true});
    std::optional<uint64_t> tma_presented;
    std::optional<uint64_t> mma_presented;
    while (!machine.done()) {
        tick_capped(machine);
        if (!tma_presented && machine.tma_mbar_event_held()) tma_presented = machine.cycle() - 1;
        if (!mma_presented && machine.mma_mbar_event_held()) mma_presented = machine.cycle() - 1;
    }
    require(tma_presented && mma_presented, "an engine never presented its event");
    require(*tma_presented + 20 < *mma_presented, "the two events were meant to be far apart");
    require(mbar_rmw_services(machine) == std::vector<RmwService>{
                {*tma_presented + 1, "complete_tx (tma), address=4104"},
                {*mma_presented + 1, "arrive (mma.commit), address=4096"}},
        "an engine event was not claimed exactly one cycle after it was presented");
}

// ---- Simultaneous events combine ----

// A depbar completion and an admission on the same counter in one cycle: the
// lds on sb0 completes (its depbar clear lands at the RF write port) and the
// ldg that waits on sb0 and sets it again issues in that same cycle: issue
// sees the completion at once (bypass (b)), and the counter is one
// calculation, 1 - 1 + 1 = 1. Neither event is deferred.
// I6a: a completion reaches issue through the counter (Q), so an instruction
// WAITING on the depbar issues the cycle after its completion. The one
// calculation still combines a completion and an admission that land in one
// cycle: here the ldg sets sb0 without waiting on it, issued exactly when the
// lds.uni's RF write lands. The lds.uni at L (one round): dispatch L +
// issue_to_dispatch, its slot visible to A-read one cycle later, the RAM
// sample D after that grant, its word Lb later, the gather register one cycle
// later (the RF write decision's grant), the write lands Wd after it: its
// depbar completion is presented then. `pad` independent movs put the ldg
// there. (The umov issues at fetch_to_issue, its word fetched at c0; the
// three-word FIFO then supplies one instruction per cycle.) The lds.uni's
// base is the umov's uniform register; the ldg's vector base r0 is never
// written (0).
void test_simultaneous_depbar_completion_and_admission() {
    const auto& profile = canonical_profile();
    const uint64_t lds_issue = nanosm::fetch_to_issue + nanosm::issue_to_urf_visible(profile) -
        nanosm::issue_register_stages; // umov ur0 at c2, the lds.uni that reads it at the uniform distance
    const auto completion_cycle = lds_issue + nanosm::issue_to_dispatch(profile) + 1 + profile.shmem_port_a_depth +
        profile.shmem_read_latency_cycles + 1 + profile.rf_write_depth_cycles;
    const auto pad = completion_cycle - lds_issue - 1;
    std::string body = "umov ur0, 0\nlds.uni r1, [ur0] {pred=PT, stall=0, set=sb0, wait=-}\n";
    for (uint64_t i = 0; i < pad; ++i) body += "mov r5, 7\n";
    body += "ldg r3, [r0] {pred=PT, stall=0, set=sb0, wait=-}\nhalt {pred=PT, stall=0, set=-, wait=sb0}\n";
    nanosm::Launch launch;
    launch.active_warps = 1;
    write_u32(image_bytes(launch.test_shmem, 64), 0, 0xabcd);
    write_u32(image_bytes(launch.gmem_image, 64), 0, 0x1234);
    nanosm::Machine machine(profile, hand_assemble(i1_asm(profile, body), profile), std::move(launch), {.timeline = true});
    std::map<uint64_t, uint32_t> sb0_after; // cycle -> sb0 at the next cycle's start
    while (!machine.done()) {
        const auto cycle = machine.cycle();
        tick_capped(machine);
        sb0_after[cycle] = machine.depbar_value(0, 0);
    }
    std::vector<std::pair<uint64_t, std::string>> sb0_events;
    uint64_t ldg_issue = 0;
    for (const auto& event : machine.trace().events()) {
        if (event.name == "depbar sb0" && event.track == 100) sb0_events.emplace_back(event.start_cycle, event.detail);
        if (event.category == nanosm::TraceCategory::instruction && event.name == "ldg") ldg_issue = event.start_cycle;
    }
    require(ldg_issue == completion_cycle, "the ldg did not issue at the lds's completion; re-derive the test");
    const auto completion = std::find(sb0_events.begin(), sb0_events.end(),
        std::make_pair(ldg_issue, std::string("value=0")));
    require(completion != sb0_events.end(), "the lds did not complete in the ldg's issue cycle");
    require(completion + 1 != sb0_events.end() && *(completion + 1) == std::make_pair(ldg_issue, std::string("value=1")),
        "the admission did not follow the completion in the same cycle");
    require(sb0_after.at(ldg_issue) == 1, "sb0 is not 1 - 1 + 1 after the cycle");
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(machine.read_reg(0, lane, 1) == 0xabcd && machine.read_reg(0, lane, 3) == 0x1234,
            "a load's value was lost");
    }
}

// A full FIFO popped and pushed in one cycle. The MMA queue (two entries) is
// full; its head retires in the cycle a third mma.async issues: the pop and
// the push combine (depth' = 2 - 1 + 1), the issue is not deferred.
void test_simultaneous_full_queue_pop_and_push() {
    nanosm::Launch launch;
    launch.active_warps = 1;
    image_bytes(launch.test_shmem, 4096);
    nanosm::Machine machine(canonical_profile(), compile(R"(
        umov ur0, 0
        umov ur1, 1024
        umov ur2, 0
        mma.async.init.m64n64k8 [ur2], [ur0], [ur1]
        mma.async.m64n64k8 [ur2], [ur0], [ur1]
        mma.async.m64n64k8 [ur2], [ur0], [ur1]
        halt
    )"), std::move(launch), {.timeline = true});
    std::map<uint64_t, std::size_t> depth_after;
    bool waited_full = false;
    while (!machine.done()) {
        const auto cycle = machine.cycle();
        tick_capped(machine);
        depth_after[cycle] = machine.mma_queue_depth();
    }
    uint64_t third_issue = 0;
    uint64_t first_last_result = 0;
    for (const auto& event : machine.trace().events()) {
        if (event.category != nanosm::TraceCategory::mma && event.category != nanosm::TraceCategory::warp) continue;
        if (event.name == "mma.async" && event.id == 3) third_issue = event.start_cycle;
        if (event.name == "MMA tmem write" && event.id == 1) first_last_result = std::max(first_last_result, event.start_cycle);
        if (event.category == nanosm::TraceCategory::warp && event.detail == "MMA command queue full") waited_full = true;
    }
    // I6a: the "mma.async" event is the MMA's acceptance, at dispatch. The
    // third issues in the cycle the head retires (look-ahead (c): the queue
    // as that cycle leaves it) and reaches the queue issue_to_dispatch later:
    // the pop and the push no longer share a cycle (a claim waits for the
    // pop), so the depth is 2, then 1 until the dispatch, then 2.
    const auto dispatch = nanosm::issue_to_dispatch(canonical_profile());
    const auto third_accept = third_issue;
    require(waited_full, "the third mma.async never found the queue full");
    require(third_accept == first_last_result + dispatch,
        "the third mma.async did not issue in the cycle the head retired (accepted " +
            std::to_string(third_accept) + " vs retired " + std::to_string(first_last_result) + ")");
    require(depth_after.at(first_last_result - 1) == 2 && depth_after.at(first_last_result) == 1 &&
            depth_after.at(third_accept - 1) == 1 && depth_after.at(third_accept) == 2,
        "the queue's pop and the dispatched push did not land in their cycles");
}

// The ingress FIFO's declared full flag is registered (B2.1 implementer
// choice 3, kept): when the FIFO is full at the start of a cycle, the bus
// cannot write a beat in that cycle even if the read port frees room in it;
// the beat lands the next cycle. A FIFO that is not full is read and written
// in one cycle (count' = count + push - pop).
void test_simultaneous_ingress_push_and_pop() {
    auto config = fast_bus_profile(8); // one 32 B beat per pop: a pop frees a whole beat's room
    config.lsu_ingress_sectors = 2;
    // Warp 1's long bookings hold chunk 0's RF write for a while, so the
    // FIFO fills (I1: a mov storm no longer does; see booking_stream).
    config.gmem_latency_dram_ticks = 2 * booking_horizon;
    with_long_bookings(config);
    nanosm::Launch launch;
    launch.active_warps = 2;
    auto& gmem = image_bytes(launch.gmem_image, 4096);
    fill_pattern(gmem, 0, 1024, 0);
    std::string text = R"(
        usread ur0, WARP_ID
        ur2r r29, ur0
        setp.ne.u32 p0, r29, 0
        @p0 bra.uni other
        mov r30, 0
        ldg.v8 r8, [r30]
        halt
    other:
    )";
    text += booking_stream(2, 100); // bank 2: r8's
    text += "halt\n";
    nanosm::Machine machine(config, compile(text, config), std::move(launch));
    machine.enable_edge_probe();
    std::map<uint64_t, bool> held; // cycle -> a read beat was held (FIFO full at the cycle's start)
    uint64_t full = 0;
    while (!machine.done()) {
        const auto cycle = machine.cycle();
        tick_capped(machine);
        held[cycle] = machine.trace().summary().lsu_ingress_full_cycles > full;
        full = machine.trace().summary().lsu_ingress_full_cycles;
    }
    std::map<uint64_t, std::pair<uint32_t, uint32_t>> moves; // cycle -> (pushes, pops)
    for (const auto& event : machine.edge_events()) {
        if (event.edge == LsuEdge::bus_read_beat) ++moves[event.cycle].first;
        if (event.edge == LsuEdge::ingress_read) ++moves[event.cycle].second;
    }
    bool both = false;
    bool registered = false;
    for (const auto& [cycle, move] : moves) {
        if (move.first != 0 && move.second != 0) both = true;
        // Full at the start of the cycle (a beat was held the cycle before
        // and this one), the read port pops: the beat still waits a cycle.
        if (move.second != 0 && held.count(cycle - 1) && held.at(cycle - 1) && held.at(cycle)) {
            registered = true;
            require(move.first == 0, "a beat entered the full FIFO in the cycle it was popped");
            require(moves.count(cycle + 1) && moves.at(cycle + 1).first != 0,
                "the held beat did not land the cycle after the pop");
        }
    }
    require(both, "no cycle read and wrote the ingress FIFO together");
    require(registered, "the FIFO was never full when its read port popped; the test proves little");
    require_vector_data(machine, 0, 8, 0, "ingress push/pop");
}

// ---- The driver shuffle (regression guard; the launches run under ctest
// via scripts/driver_shuffle_check.py) ----

// Register drivers read only Q and the wires, so permuting their order (and
// the independent request generators') every cycle changes nothing: not a
// cycle, a counter, a trace byte or a memory byte. A guard against a future
// driver that reads another's D or shared state, not a proof of independence.
void test_driver_shuffle_unit_kernels() {
    struct Scenario {
        std::string name;
        std::function<std::unique_ptr<nanosm::Machine>()> make;
    };
    const auto backpressure = [] {
        auto config = canonical_profile();
        config.tma_queue_commands = 0;
        config.tma_sector_credits = 0;
        config.gmem_queue_sectors = 8;
        config.lsu_ingress_sectors = 2;
        with_long_bookings(config); // I1: a mov storm no longer holds the RF write port
        nanosm::Launch launch;
        launch.active_warps = 2;
        fill_pattern(image_bytes(launch.gmem_image, 4096), 0, 1024, 0);
        std::string text = "usread ur0, WARP_ID\nur2r r29, ur0\nsetp.ne.u32 p0, r29, 0\n@p0 bra.uni other\nmov r30, 0\nmov r31, 512\n"
            "mov r28, 2048\nsread r27, LANE_ID\nshli r27, r27, 5\naddi r27, r27, 2048\n"
            "ldg.v8 r0, [r30]\nldg.v8 r8, [r31]\nldg r16, [r28]\nstg.v8 [r27], r0\nhalt\nother:\n";
        text += booking_stream(0, 150);
        text += "halt\n";
        return std::make_unique<nanosm::Machine>(config, compile(text, config), std::move(launch),
            nanosm::TraceOptions{.timeline = true, .debug = true});
    };
    const auto tma_and_mma = [] {
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.test_shmem, 8192);
        auto& gmem = image_bytes(launch.gmem_image, 64);
        for (uint32_t i = 0; i < 16; ++i) write_u32(gmem, 4 * i, 0x100u + i);
        write_mbar(shmem, 4096, 1, 0);
        write_mbar(shmem, 4104, 0, 64);
        write_tma_descriptor(shmem, 6144, plain_tma_1d(16));
        return std::make_unique<nanosm::Machine>(canonical_profile(), compile(R"(
            umov ur0, 0
            umov ur1, 1024
            umov ur4, 4096
            umov ur2, 0
        )" + tma_operands_asm(10, 2048, 11, 4104, 8) + R"(
            mma.async.init.m64n64k8 [ur2], [ur0], [ur1]
            mma.commit [ur4]
            tma.load.1d [ur10], [6144, ur8], [ur11]
            mma.async.m64n64k8 [ur2], [ur0], [ur1]
            halt
        )"), std::move(launch), nanosm::TraceOptions{.timeline = true, .debug = true});
    };
    const auto acc_round_trip = [] {
        nanosm::Launch launch;
        launch.active_warps = 2;
        return std::make_unique<nanosm::Machine>(canonical_profile(), compile(R"(
            sread r0, THREAD_ID
            usread ur8, WARP_ID
            ushli ur8, ur8, 10
            umov ur9, 0
            acc.st.v8 [ur8, 64], r0
            acc.ld.v8 r16, [ur8, 64]
            fadd r24, r16, r17
            halt
        )"), std::move(launch), nanosm::TraceOptions{.timeline = true, .debug = true});
    };
    const auto scatter_and_tma_store = [] {
        nanosm::Launch launch;
        launch.active_warps = 2;
        auto& shmem = image_bytes(launch.test_shmem, 4096);
        fill_pattern(shmem, 1024, 64, 0);
        write_rank1_tma(shmem, 0, 2048, 64, 128);
        image_bytes(launch.gmem_image, 8192);
        return std::make_unique<nanosm::Machine>(canonical_profile(), compile(R"(
            usread ur0, WARP_ID
            ur2r r29, ur0
            setp.ne.u32 p0, r29, 0
            @p0 bra.uni other
        )" + tma_operands_asm(10, 1024, 3, 128, 8) + R"(
            tma.store.1d [0, ur8], [ur10], [ur3]
        wait:
            mbar.try_wait p1, [ur3], 0
            @!p1 bra.uni wait
            halt
        other:
            sread r4, LANE_ID
            shli r5, r4, 7
            stg [r5 + 4096], r4
            halt
        )"), std::move(launch), nanosm::TraceOptions{.timeline = true, .debug = true});
    };
    const std::vector<Scenario> scenarios = {
        {"vector loads under RF backpressure", backpressure},
        {"TMA load and MMA commit", tma_and_mma},
        {"acc.st / acc.ld round trip", acc_round_trip},
        {"scatter and TMA store", scatter_and_tma_store},
    };
    const auto fingerprint = [](nanosm::Machine& machine) {
        machine.run(20000);
        return std::make_tuple(machine.cycle(), machine.trace().summary_text(), machine.trace().perfetto_json(),
            machine.gmem(), machine.shmem());
    };
    for (const auto& scenario : scenarios) {
        auto reference_machine = scenario.make();
        const auto reference = fingerprint(*reference_machine);
        for (const uint64_t seed : {1ull, 2ull, 3ull, 0x9e3779b97f4a7c15ull}) {
            auto machine = scenario.make();
            machine->set_driver_shuffle_seed(seed);
            require(fingerprint(*machine) == reference,
                scenario.name + ": driver order changed the run (seed " + std::to_string(seed) + ")");
        }
    }
}

} // namespace

// reg: hold, write, commit
TEST(Stepping, reg_hold_write_commit) {
    test_reg_hold_write_commit();
}

// reg: duplicate write PANIC
TEST(Stepping, reg_duplicate_write_panics) {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    test_reg_duplicate_write_panics();
}

// the load buffer's lane groups: eight RF writes each, after the group's last word
TEST(Stepping, load_buffer_groups) {
    test_load_buffer_groups();
}

// assembly reuse: final load drain -> scalar store
TEST(Stepping, assembly_reuse_load_to_scalar_store) {
    test_assembly_reuse_load_to_scalar_store();
}

// assembly reuse: final load drain -> vector store
TEST(Stepping, assembly_reuse_load_to_vector_store) {
    test_assembly_reuse_load_to_vector_store();
}

// assembly reuse: final load drain -> load
TEST(Stepping, assembly_reuse_load_to_load) {
    test_assembly_reuse_load_to_load();
}

// sanitizer: assembly reader takes only valid lanes
TEST(Stepping, sanitizer_assembly_reader_takes_valid_lanes) {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    test_sanitizer_assembly_reader_takes_valid_lanes();
}

// mbar engines: events visible the next cycle
TEST(Stepping, mbar_engine_events_visible_next_cycle) {
    test_mbar_engine_events_visible_next_cycle();
}

// simultaneous: depbar completion and admission
TEST(Stepping, simultaneous_depbar_completion_and_admission) {
    test_simultaneous_depbar_completion_and_admission();
}

// simultaneous: full queue pop and push
TEST(Stepping, simultaneous_full_queue_pop_and_push) {
    test_simultaneous_full_queue_pop_and_push();
}

// simultaneous: ingress FIFO push and pop
TEST(Stepping, simultaneous_ingress_push_and_pop) {
    test_simultaneous_ingress_push_and_pop();
}

// driver shuffle: unit kernels unchanged
TEST(SteppingSlow, driver_shuffle_unit_kernels) {
    test_driver_shuffle_unit_kernels();
}

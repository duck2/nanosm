// The stepping discipline: Reg<T>, the LSU assembly contract, simultaneous events, the driver shuffle.
// Each test's TEST wrapper is at the end.
#include "test_support.hpp"

#include <regex>
#include <sstream>

namespace {

using namespace nanosm_test;

// ===========================================================================
// Stepping discipline
// ===========================================================================

// ---- Reg<T> (src/machine/reg.hpp) ----

void test_reg_hold_write_commit() {
    nanosm::Sanitizer sanitizer;
    uint64_t cycle = 7;
    nanosm::AllClocked all(sanitizer, cycle);
    nanosm::Reg<int> reg(all,"test register", 5);
    sanitizer.record_cycle_start();
    all.commit();
    require(reg.cur() == 5, "a register nobody wrote did not hold its value");
    sanitizer.record_cycle_start();
    reg.set_next(9, "writer");
    require(reg.cur() == 5, "set_next() changed Q before the clock edge");
    all.commit();
    require(reg.cur() == 9, "commit did not publish D");
    sanitizer.record_cycle_start();
    all.commit();
    require(reg.cur() == 9, "the tick after a write did not hold");
}

// A second write in one tick is a PANIC naming the register, the cycle and
// both writers, even when the same writer writes twice.
void test_reg_duplicate_write_panics() {
    nanosm::Sanitizer sanitizer;
    uint64_t cycle = 7;
    nanosm::AllClocked all(sanitizer, cycle);
    nanosm::Reg<int> reg(all,"test register", 0);
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
            all.commit();
            return;
        }
        throw std::runtime_error("a second write in one tick did not panic");
    };
    expect_panic("drain", "fill");
    expect_panic("drain", "drain"); // the same writer twice
    sanitizer.record_cycle_start();
    reg.set_next(3, "next tick");    // a new tick: one write is fine again
    all.commit();
    require(reg.cur() == 3, "the register did not recover after a duplicate-write PANIC");
}

// ---- The load buffer's groups (per-lane ldg.v8) ----

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
        if (event.edge == LsuEdge::buffer_write) {
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

// ---- Same-cycle reuse after a load's final RF drain (one command slot) ----

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
// one command slot, the second waits for it: it
// issues in the cycle the load's final RF write releases it (look-ahead
// (c)) and the LSU admits it issue_to_dispatch later, into the slot
// free since then. (A claim waits for the
// release, so issue cannot line the two up in one cycle.)
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
        after[cycle] = {machine.lsu_buffer_owner(0), machine.lsu_buffer_valid(0), machine.lsu_buffer_valid_count(0)};
        store_after[cycle] = {machine.lsu_buffer_owner(1), machine.lsu_buffer_valid(1), machine.lsu_buffer_valid_count(1)};
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
// buffer is empty and unowned (one buffer per direction, owned
// only while data is in it). The output alone cannot show it: scalar-store
// egress reads the words without valid_count.
void test_buffer_reuse_load_to_scalar_store() {
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
// cycle (bypass (e)) and fills the store buffer Lr later; the buffer has
// no owner until that data lands.
void test_buffer_reuse_load_to_vector_store() {
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
void test_buffer_reuse_load_to_load() {
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

// ---- Pipe<T> (src/machine/pipe.hpp) ----

// A value pushed in tick t is in stage n - 1 after n edges and gone after N;
// a hold keeps every stage; a replacement travels with its value; a value
// replaced by nothing leaves at that stage.
void test_pipe_advance_hold_replace() {
    nanosm::Sanitizer sanitizer;
    uint64_t cycle = 0;
    nanosm::AllClocked all(sanitizer, cycle);
    nanosm::Pipe<int> pipe(all,"test pipeline", 32);
    pipe.build(3);
    const auto tick = [&]() {
        all.commit();
        ++cycle;
        sanitizer.record_cycle_start();
    };
    const auto at = [&](std::size_t stage) { return pipe.stage(stage).value_or(-1); };
    sanitizer.record_cycle_start();
    require(pipe.stages() == 3 && pipe.empty(), "a new pipeline is not empty");
    pipe.push(10, "entry");
    require(pipe.empty(), "push() changed Q before the clock edge");
    tick();
    require(at(0) == 10 && at(1) == -1 && at(2) == -1, "a pushed value is not in stage 0 after one edge");
    pipe.push(11, "entry"); // consecutive: one behind
    tick();
    tick();                 // a gap behind 11
    require(at(0) == -1 && at(1) == 11 && at(2) == 10, "two values and a gap did not keep their distances");
    require(pipe.last() == std::optional<int>(10), "the output stage does not hold the first value after three edges");
    pipe.hold("stall");
    tick();
    require(at(0) == -1 && at(1) == 11 && at(2) == 10, "a held pipeline moved");
    pipe.replace(1, 21, "work at stage 1"); // the value at stage 1 carries 21 from the edge on
    tick();
    require(at(0) == -1 && at(1) == -1 && at(2) == 21, "a replacement did not travel with its value");
    pipe.push(12, "entry");
    tick();
    require(at(0) == 12 && at(2) == -1, "the output's value did not leave");
    pipe.replace(0, std::nullopt, "leaves early");
    tick();
    require(pipe.empty(), "a value replaced by nothing stayed in the pipeline");
    pipe.push(13, "entry");
    tick();
    pipe.hold("stall");
    pipe.replace(0, 23, "work under a hold"); // held: the value stays in its stage, with the replacement
    tick();
    require(at(0) == 23 && at(1) == -1, "a replacement under a hold did not stay in its stage");
    const auto size = pipe.size();
    require(std::string(size.kind) == "pipeline" && size.entries == 3 && size.bits_per_entry == std::optional<uint64_t>(33),
        "the pipeline does not report its stages and its declared bits (payload + valid)");
}

// Each write point is one net with one driver: a second push, a second
// replacement of one stage and a second hold in one tick are PANICs naming
// the pipeline and both writers.
void test_pipe_duplicate_write_panics() {
    nanosm::Sanitizer sanitizer;
    uint64_t cycle = 7;
    nanosm::AllClocked all(sanitizer, cycle);
    nanosm::Pipe<int> pipe(all,"test pipeline");
    pipe.build(2);
    const auto expect_panic = [&](const std::string& what, const std::function<void(const char*)>& write) {
        sanitizer.record_cycle_start();
        write("first");
        try {
            write("second");
        } catch (const nanosm::SimError& error) {
            const std::string message = error.what();
            for (const std::string needle : {std::string("test pipeline"), std::string("cycle=7"),
                     std::string("first by 'first'"), std::string("then by 'second'")}) {
                require(message.find(needle) != std::string::npos,
                    what + ": the duplicate-write PANIC lacks '" + needle + "': " + message);
            }
            return;
        }
        throw std::runtime_error(what + ": a second write in one tick did not panic");
    };
    expect_panic("push", [&](const char* tag) { pipe.push(1, tag); });
    expect_panic("replace", [&](const char* tag) { pipe.replace(0, 2, tag); });
    expect_panic("hold", [&](const char* tag) { pipe.hold(tag); });
    // Different points in one tick are different nets.
    nanosm::Pipe<int> other(all,"other pipeline");
    other.build(2);
    sanitizer.record_cycle_start();
    other.push(1, "entry");
    other.replace(0, std::nullopt, "stage 0");
    other.replace(1, std::nullopt, "stage 1");
}

// ---- Bounded<T, N> (src/machine/fixed.hpp) and the maxima (src/limits.hpp) ----

// A payload of at most N items: it counts what it holds, compares only the
// counted items, refuses one more than N, and a profile past an architectural
// maximum is refused by name before a machine is built.
void test_bounded_payload_and_limits() {
    nanosm::Bounded<uint32_t, 4> words;
    require(words.empty() && words.capacity() == 4, "a new payload is not empty");
    for (uint32_t value = 1; value <= 4; ++value) words.push_back(value);
    require(words.size() == 4 && words.front() == 1 && words.back() == 4, "a payload lost an item");
    try {
        words.push_back(5);
        throw std::runtime_error("a payload took more items than it has places");
    } catch (const nanosm::SimError&) {}
    words.pop_front();
    nanosm::Bounded<uint32_t, 4> same{2, 3, 4};
    require(words == same && words.size() == 3, "an ordered payload's head did not leave in order");
    words.clear();
    require(words == nanosm::Bounded<uint32_t, 4>(), "a cleared payload is not equal to an empty one");
    static_assert(std::is_trivially_copyable_v<nanosm::Bounded<uint32_t, 4>>);
    static_assert(nanosm::FixedPayload<std::optional<nanosm::Bounded<uint32_t, 4>>>);
    static_assert(!nanosm::FixedPayload<std::vector<uint32_t>>);

    auto profile = canonical_profile();
    profile.shmem_banks = static_cast<uint32_t>(nanosm::limits::shmem_banks) + 1;
    try {
        nanosm::validate_device_profile(profile);
        throw std::runtime_error("a profile past an architectural maximum was accepted");
    } catch (const nanosm::ProfileError& error) {
        require(std::string(error.what()).find("shmem banks") != std::string::npos,
            std::string("the refusal does not name the knob: ") + error.what());
    }
}

// ---- Regs<T> (src/machine/regs.hpp) ----

// A group is N registers under one name: a member written this tick takes its
// value at the edge, a member nobody writes holds, different members may be
// written in one tick, and one member written twice in a tick is a PANIC
// naming the group, the member and both writers.
void test_regs_members_hold_write_and_panic() {
    nanosm::Sanitizer sanitizer;
    uint64_t cycle = 7;
    nanosm::AllClocked all(sanitizer, cycle);
    nanosm::Regs<int> group(all,"test group", 5);
    group.build(4, 9);
    sanitizer.record_cycle_start();
    require(group.count() == 4 && group.cur() == std::vector<int>({9, 9, 9, 9}), "a new group lacks its initial values");
    group.set_next(1, 11, "writer of member 1");
    group[3].set_next(13, "writer of member 3"); // another member, other logic: its own net
    require(group.cur()[1] == 9 && group[3].cur() == 9, "set_next() changed Q before the clock edge");
    all.commit();
    require(group.cur() == std::vector<int>({9, 11, 9, 13}), "written members did not take D, or an unwritten one moved");
    sanitizer.record_cycle_start();
    all.commit();
    require(group.cur() == std::vector<int>({9, 11, 9, 13}), "a group nobody wrote did not hold");
    const auto size = group.size();
    require(std::string(size.kind) == "group" && size.entries == 4 && size.bits_per_entry == std::optional<uint64_t>(5),
        "the group does not report its members and their declared bits");
    sanitizer.record_cycle_start();
    group.set_next(2, 1, "first");
    try {
        group.set_next(2, 2, "second");
    } catch (const nanosm::SimError& error) {
        const std::string message = error.what();
        for (const std::string needle : {std::string("test group [2]"), std::string("cycle=7"),
                 std::string("first by 'first'"), std::string("then by 'second'")}) {
            require(message.find(needle) != std::string::npos,
                "the group's duplicate-write PANIC lacks '" + needle + "': " + message);
        }
        return;
    }
    throw std::runtime_error("a second write of one member in one tick did not panic");
}

// A FIFO: the next pop returns the oldest entry; an entry pushed at t is
// readable from t + its read latency; a pop and a push in one tick; its ports'
// widths; a push past its capacity is an internal error. A FIFO of single
// bits reads a bit by value and holds it across a tick; a zero-capacity FIFO
// takes a pop of nothing and holds.
void test_fifo_order_latency_and_widths() {
    nanosm::Sanitizer sanitizer;
    uint64_t cycle = 3;
    nanosm::AllClocked all(sanitizer, cycle);
    nanosm::Fifo<int> fifo(all,"test FIFO", 8);
    fifo.build(4, nanosm::FifoPorts{2, 2, 2});
    const auto tick = [&] {
        all.commit();
        sanitizer.record_cycle_start();
    };
    sanitizer.record_cycle_start();
    fifo.push(std::vector<int>{1, 2}, "writer");
    require(fifo.count() == 0, "a push changed Q before the clock edge");
    tick();
    require(fifo.count() == 2 && fifo.readable() == 0, "an entry was readable before its read latency");
    tick();
    require(fifo.readable() == 2 && fifo.front(0) == 1 && fifo.front(1) == 2, "the head is not the oldest entry");
    fifo.pop(1, "reader");
    fifo.push(3, "writer");
    tick();
    require(fifo.count() == 2 && fifo.readable() == 1 && fifo.front() == 2, "a pop and a push in one tick went wrong");
    tick();
    require(fifo.readable() == 2 && fifo.front(1) == 3, "a FIFO nobody wrote did not hold its order");
    const auto size = fifo.size();
    require(std::string(size.kind) == "FIFO" && size.entries == 4 && size.bits_per_entry == std::optional<uint64_t>(8),
        "the FIFO does not report its capacity and its entries' declared bits");
    bool refused = false;
    try {
        (void)fifo.front(2); // past the read port
    } catch (const nanosm::SimError&) {
        refused = true;
    }
    require(refused, "a read past the read port's width was allowed");
    fifo.push(std::vector<int>{4, 5}, "writer");
    tick();
    fifo.push(6, "writer");
    refused = false;
    try {
        tick();
    } catch (const nanosm::SimError& error) {
        refused = std::string(error.what()).find("holds more than its 4 entries") != std::string::npos;
    }
    require(refused, "a push past the capacity was not an internal error");
    // (A machine stops at an internal error; these two get an AllClocked of their own.)
    nanosm::Sanitizer quiet;
    nanosm::AllClocked other(quiet, cycle);
    nanosm::Fifo<bool> bits(other, "test bits FIFO", 1);
    bits.build(2);
    nanosm::Fifo<int> none(other, "zero-capacity FIFO");
    none.build(0);
    const auto other_tick = [&] {
        other.commit();
        quiet.record_cycle_start();
    };
    quiet.record_cycle_start();
    bits.push(true, "writer");
    none.pop(0, "reader");
    other_tick();
    none.pop(0, "reader");
    other_tick();
    const bool held = bits.front();
    require(held && bits.readable() == 1, "a one-bit FIFO did not hold its bit across a tick");
    require(none.empty() && none.room() == 0, "a zero-capacity FIFO did not hold after a pop of nothing");
}

// A second push into one FIFO in one tick PANICs, as a second write of a register.
void test_fifo_second_push_panics() {
    nanosm::Sanitizer sanitizer;
    uint64_t cycle = 3;
    nanosm::AllClocked all(sanitizer, cycle);
    nanosm::Fifo<int> second(all,"second FIFO");
    second.build(2);
    sanitizer.record_cycle_start();
    second.push(1, "first");
    try {
        second.push(2, "second");
    } catch (const nanosm::SimError& error) {
        const std::string message = error.what();
        for (const std::string needle : {std::string("second FIFO (push)"), std::string("first by 'first'"),
                 std::string("then by 'second'")}) {
            require(message.find(needle) != std::string::npos,
                "the FIFO's second-push PANIC lacks '" + needle + "': " + message);
        }
        return;
    }
    throw std::runtime_error("a second push in one tick did not panic");
}

// A black box's output register holds a value from its inputs' cycle +
// latency, for that one cycle; nothing passes through it in one cycle, and
// inputs closer than its interval are refused (a refused value is not a
// write: a good one may follow in the same tick). It refuses a value before
// it is built; one that keeps an inside is built with the inside's idle
// question, and is not empty while the inside holds work.
void test_black_box_latency_and_interval() {
    nanosm::Sanitizer sanitizer;
    uint64_t cycle = 0;
    nanosm::AllClocked all(sanitizer, cycle);
    nanosm::BlackBox<uint32_t> black_box(all, nanosm::Unit::none, "a black box");
    const auto refused = [&](uint32_t value, uint64_t taken_at) {
        try {
            black_box.deliver(value, taken_at, "test");
        } catch (const nanosm::SimError&) {
            return true;
        }
        return false;
    };
    require(refused(1, 0), "a black box not yet built took a value");
    nanosm::BlackBoxDeclaration declaration("a black box", nanosm::SignOff::not_signed);
    declaration.latency = 3;
    declaration.interval = 2;
    black_box.build(declaration);
    std::vector<std::optional<uint32_t>> outputs;
    for (; cycle < 8; ++cycle) {
        sanitizer.record_cycle_start();
        if (cycle == 0) black_box.deliver(10, 0, "test");
        if (cycle == 1) require(refused(11, 1), "an input one cycle after another was taken (interval 2)");
        if (cycle == 2) {
            require(refused(12, 1), "an input one cycle after another was taken (interval 2)");
            black_box.deliver(12, 2, "test"); // after a refusal in the same tick
        }
        if (cycle == 7) require(refused(13, 4), "a value due now was delivered (it would pass in one cycle)");
        outputs.push_back(black_box.output());
        all.commit();
    }
    const std::vector<std::optional<uint32_t>> expected{
        std::nullopt, std::nullopt, std::nullopt, 10u, std::nullopt, 12u, std::nullopt, std::nullopt};
    require(outputs == expected, "the black box's output register does not hold each value at its inputs' cycle + 3");
    require(black_box.empty(), "the black box is not empty after its outputs");

    nanosm::BlackBox<uint32_t, int> keeper(all, nanosm::Unit::none, "a black box with an inside");
    nanosm::BlackBoxDeclaration keeps("a black box with an inside", nanosm::SignOff::not_signed);
    bool unasked = false;
    try {
        keeper.build(keeps);
    } catch (const nanosm::SimError&) {
        unasked = true;
    }
    require(unasked, "a black box that keeps an inside was built without its idle question");
    keeper.build(keeps, [](const int& waiting) { return waiting == 0; });
    keeper.inside() = 1;
    require(!keeper.empty(), "a black box with work inside says it is empty");
    keeper.inside() = 0;
    require(keeper.empty(), "an idle black box says it is not empty");
}

// A memory: a read sees the word as it was at the start of the cycle and a
// write lands at the clock edge (read first); a write enables its elements one
// by one; two reads of one word through one port are one read; it reports its
// words, width and ports.
void test_memory_read_first_and_enables() {
    nanosm::Sanitizer sanitizer;
    uint64_t cycle = 0;
    nanosm::AllClocked all(sanitizer, cycle);
    nanosm::Memory<uint32_t> memory(all,"test memory");
    memory.build(nanosm::MemoryShape{2, 4, 2, 64, 1, 1, 2, nanosm::SameCycle::read_first});
    const auto tick = [&] {
        all.commit();
        sanitizer.record_cycle_start();
    };
    sanitizer.record_cycle_start();
    const std::array<uint32_t, 2> first{7, 8};
    memory.write(1, 3, first, nanosm::Memory<uint32_t>::every_element, "writer");
    require(memory.read(1, 3, "reader")[0] == 0, "a read saw a write before its clock edge");
    tick();
    const std::array<uint32_t, 2> second{9, 10};
    memory.write(1, 3, second, 0b10, "writer"); // element 1 only
    const auto word = memory.read(1, 3, "reader");
    require(word[0] == 7 && word[1] == 8, "a read and a write of one word in one cycle did not read first");
    (void)memory.read(1, 3, "a second reader of the same word");
    tick();
    require(memory.peek(1, 3)[0] == 7 && memory.peek(1, 3)[1] == 10, "a write did not keep its disabled element");
    require(memory.contents().at((3 * 2 + 1) * 2 + 1) == 10, "the storage is not in (row, bank, element) order");
    bool refused = false;
    try {
        memory.poke(0, 0, 2, 99); // element 2 of a two-element word: bank 1's word in the storage
    } catch (const nanosm::SimError&) {
        refused = true;
    }
    require(refused && memory.peek(1, 0)[0] == 0, "a poke past the word's elements reached the next word");
    nanosm::Memory<uint32_t> rom(all,"read-only memory");
    rom.build(nanosm::MemoryShape{1, 2, 1, 32, 1, 0, 1, nanosm::SameCycle::read_only});
    refused = false;
    try {
        const std::array<uint32_t, 1> value{1};
        rom.write(0, 0, value, nanosm::Memory<uint32_t>::every_element, "writer");
    } catch (const nanosm::SimError&) {
        refused = true;
    }
    require(refused, "a write into a memory with no write port was taken");
    const auto size = memory.size();
    require(std::string(size.kind) == "memory" && size.entries == 8 && size.banks == 2 &&
            size.bits_per_entry == std::optional<uint64_t>(64) && size.read_ports == 1 && size.write_ports == 1 &&
            size.read_delay == 2,
        "the memory does not report its words, width, banks and ports");
}

// A memory's port asked for a second word in one cycle, or a write with every
// write port taken, PANICs, as a second write of a register.
void test_memory_port_panics() {
    for (const bool write : {false, true}) {
        nanosm::Sanitizer sanitizer;
        uint64_t cycle = 5;
        nanosm::AllClocked all(sanitizer, cycle);
        nanosm::Memory<uint32_t> memory(all,"ported memory");
        memory.build(nanosm::MemoryShape{2, 4, 1, 32, 1, 1, 1, nanosm::SameCycle::read_first});
        sanitizer.record_cycle_start();
        const std::array<uint32_t, 1> value{1};
        try {
            if (write) {
                memory.write(0, 1, value, nanosm::Memory<uint32_t>::every_element, "first");
                memory.write(1, 1, value, nanosm::Memory<uint32_t>::every_element, "the other bank");
                memory.write(0, 2, value, nanosm::Memory<uint32_t>::every_element, "second");
            } else {
                (void)memory.read(0, 1, "first", 0);
                (void)memory.read(1, 2, "the other bank", 0);
                (void)memory.read(0, 2, "second", 0);
            }
        } catch (const nanosm::SimError& error) {
            const std::string message = error.what();
            const std::string needle = write ? "ported memory' bank 0: a write of row 2 with its 1 write port taken"
                                             : "ported memory' bank 0: read port 0 asked for row 2 after row 1";
            require(message.find("PANIC cycle=5") != std::string::npos && message.find(needle) != std::string::npos &&
                    message.find("'second'") != std::string::npos,
                "the memory's port PANIC does not say what happened: " + message);
            continue;
        }
        throw std::runtime_error(std::string("a second ") + (write ? "write" : "read") + " through one port did not panic");
    }
}

// The state report: every primitive has a unit and declared bits, except
// exactly the listed ones; each unit's lines add up to its total, the units'
// totals to the machine's, and that is the sum of the declared sizes.
void test_state_report_adds_up() {
    const auto config = canonical_profile();
    nanosm::Launch launch;
    launch.active_warps = 1;
    const nanosm::Machine machine(config, compile("halt\n", config), std::move(launch));
    const auto missing = machine.state_not_yet_declared();
    require(missing.empty(), "a primitive is listed as not yet declared");
    // What is counted in the open, and the memories' words (black boxes).
    uint64_t declared = 0;
    uint64_t memory_words = 0;
    for (const auto& size : machine.state_sizes()) {
        require(size.unit != nanosm::Unit::none, "'" + size.name + "' belongs to no unit");
        const bool listed = std::find(missing.begin(), missing.end(), size.name) != missing.end();
        require(size.bits().has_value() != listed,
            "'" + size.name + (listed ? "' is listed as not yet declared but declares bits"
                                      : "' declares no bits and is not listed"));
        (std::string(size.kind) == "memory" ? memory_words : declared) += size.bits().value_or(0);
    }
    nanosm::BlackBoxCost costs;
    uint64_t memory_black_boxes = 0;
    for (const auto& black_box : machine.black_boxes()) {
        costs.flip_flops += black_box.cost.flip_flops;
        costs.memory_bits += black_box.cost.memory_bits;
        costs.multipliers += black_box.cost.multipliers;
        for (const auto& size : machine.state_sizes()) {
            if (size.name == black_box.name && std::string(size.kind) == "memory") {
                memory_black_boxes += black_box.cost.memory_bits;
            }
        }
    }
    require(memory_black_boxes == memory_words, "the memories' black boxes do not declare the memories' words");
    const auto report = machine.state_report();
    const auto value = [](const std::string& digits) {
        std::string plain;
        for (const char c : digits) {
            if (c != ',') plain += c;
        }
        return std::stoull(plain);
    };
    const std::regex line(R"(^  (register|group|pipeline|FIFO) .* ([0-9,]+)  .*$)");
    const std::regex unit_total(R"(^  in the open: flip-flops ([0-9,]+), FIFO ([0-9,]+), total ([0-9,]+)$)");
    const std::regex black_box_line(R"(^  black box +.+ ([0-9,]+)  declared; (not signed off|signed off)$)");
    const std::regex unit_walls(R"(^  behind walls: flip-flops ([0-9,]+), memory ([0-9,]+), multipliers ([0-9,]+) \(declared\)$)");
    // The first total is the open state's (then the black boxes' declared one, then the sum).
    const std::regex grand_total(R"(^    total +([0-9,]+)$)");
    std::istringstream lines(report);
    std::string text;
    uint64_t unit_lines = 0;
    uint64_t black_box_lines = 0;
    uint64_t units = 0;
    uint64_t walls = 0;
    uint64_t unit_count = 0;
    std::optional<uint64_t> total;
    std::smatch match;
    while (std::getline(lines, text)) {
        if (std::regex_match(text, match, line)) {
            unit_lines += value(match[2]);
        } else if (std::regex_match(text, match, unit_total)) {
            const auto sum = value(match[1]) + value(match[2]);
            require(sum == value(match[3]) && sum == unit_lines, "a unit's total is not the sum of its lines: " + text);
            units += sum;
            unit_lines = 0;
            ++unit_count;
        } else if (std::regex_match(text, match, black_box_line)) {
            black_box_lines += value(match[1]);
        } else if (std::regex_match(text, match, unit_walls)) {
            require(value(match[1]) + value(match[2]) == black_box_lines, "a unit's black boxes do not add up: " + text);
            walls += black_box_lines;
            black_box_lines = 0;
        } else if (!total.has_value() && std::regex_match(text, match, grand_total)) {
            total = value(match[1]);
        }
    }
    require(unit_count == static_cast<uint64_t>(nanosm::last_unit), "the report does not list every unit");
    require(total.has_value() && *total == units && units == declared,
        "the report's open total is not the sum of its units and of the declared sizes");
    require(walls == costs.flip_flops + costs.memory_bits, "the units' black boxes do not add up to their costs");
    // The write calendar: 14 of its 15 stages can hold a booking, three bank
    // counts of one bit each, zero meaning free.
    bool calendar = false;
    for (const auto& size : machine.state_sizes()) {
        calendar = calendar || (size.name == "RF write calendar" && size.bits() == std::optional<uint64_t>(42));
    }
    require(calendar, "the RF write calendar is not declared as 42 flip-flops");
    // The canonical numbers: counted in the open, declared behind walls (the
    // memories' words with this one-word kernel's instruction memory, 128
    // bits; FA's is 22,144).
    require(report.find("  in the open (counted)\n    flip-flops       38,546\n    FIFO storage     6,690\n") !=
            std::string::npos,
        "the canonical profile's open flip-flops moved from 38,546 (or its FIFO storage from 6,690)");
    require(report.find("    flip-flops       18,083\n    memory bits      1,415,296\n    total            1,433,379\n"
                        "    multipliers      168\n") != std::string::npos,
        "the canonical profile's declared costs moved (18,083 flip-flops, 1,415,296 memory bits, 168 multipliers)");
    // Every black box prints the sign-off its definition gives, and the
    // totals count the signed ones as the definitions do.
    std::size_t signed_off = 0;
    for (const auto& black_box : machine.black_boxes()) {
        if (black_box.signed_off()) ++signed_off;
        require(report.find("  " + black_box.name + " (" + nanosm::unit_name(black_box.unit) + "): " +
                    (black_box.signed_off() ? "signed off\n" : "not signed off\n")) != std::string::npos,
            "the report does not print black box '" + black_box.name + "' with its definition's sign-off");
    }
    require(report.find("black boxes, " + std::to_string(signed_off) + " signed off)") != std::string::npos,
        "the report's totals do not count the black boxes signed off");
    // The MMA core: at a long result latency each further cycle only delays
    // the packed result (16 words, 512 bits); at a latency of exactly
    // tmem.port_depth + tmem.read_latency_cycles the report says plainly
    // that no clock edge is left after C.
    const auto mma_core = [](uint32_t latency) {
        auto profile = canonical_profile();
        profile.mma_result_latency_cycles = latency;
        nanosm::Launch one;
        one.active_warps = 1;
        const nanosm::Machine varied(profile, compile("halt\n", profile), std::move(one));
        for (const auto& black_box : varied.black_boxes()) {
            if (black_box.name == "MMA core") return std::pair{black_box.cost.flip_flops, varied.state_report()};
        }
        throw std::runtime_error("no MMA core black box");
    };
    const auto [at_3999, report_3999] = mma_core(3999);
    const auto [at_4000, report_4000] = mma_core(4000);
    require(at_4000 - at_3999 == 512, "one more cycle of MMA result latency costs " + std::to_string(at_4000 - at_3999) +
        " flip-flops, not the delayed result's 512");
    const auto [at_8, report_8] = mma_core(8);
    require(at_8 == 12'256, "the canonical MMA core's cost moved from 12,256 flip-flops: " + std::to_string(at_8));
    require(report_8.find("NOT A REGISTERED ONE") == std::string::npos, "the canonical MMA core is flagged");
    const auto [at_4, report_4] = mma_core(4);
    require(report_4.find("AT THIS PROFILE THE LATENCY IS NOT A REGISTERED ONE: mma.result_latency_cycles (4) equals "
                          "tmem.port_depth + tmem.read_latency_cycles (4)") != std::string::npos,
        "a result latency of port_depth + read_latency is not flagged on the MMA core's lines");
    // The operands the dispatch and arithmetic pipelines carry inside the
    // RF's and URF's read delays are the memories' (their lines give the
    // ports' bound: banks x read ports x word bits x delay); what is held
    // after a delay counts, as wide as the path's widest round there. The
    // dispatch entry is 175 bits a stage before its words, the arithmetic
    // one 132.
    struct Case {
        uint32_t rf_delay, urf_delay, rf_banks;
        uint64_t dispatch, arithmetic;
        const char* rf_note;
        const char* urf_note;
    };
    for (const auto& expected : {
             // canonical: uniform round 0 held one stage (two words dispatched, ur2r's one)
             Case{2, 1, 3, 3 * 175 + 64, 3 * 132 + 32, "at most 3,072 bits", "at most 64 bits"},
             // equal delays: the dispatch's vector holding stage, two words (sts, stg)
             Case{1, 1, 3, 3 * 175 + 1024 + 64, 2 * 132, "at most 1,536 bits", "at most 64 bits"},
             // a slow RF: uniform round 0 held two stages, an mma.async's urC one
             Case{3, 1, 3, 4 * 175 + 128 + 32, 4 * 132 + 64, "at most 4,608 bits", "at most 64 bits"},
             // one bank, one read port: the RF's bound is one word for two cycles
             Case{2, 1, 1, 3 * 175 + 64, 3 * 132 + 32, "at most 1,024 bits", "at most 64 bits"},
         }) {
        auto profile = canonical_profile();
        profile.rf_read_latency_cycles = expected.rf_delay;
        profile.urf_read_latency_cycles = expected.urf_delay;
        profile.rf_banks = expected.rf_banks;
        profile.urf_issue_to_port_cycles = std::max(profile.urf_issue_to_port_cycles, 1 + expected.rf_delay);
        nanosm::Launch one;
        one.active_warps = 1;
        const nanosm::Machine varied(profile, compile("halt\n", profile), std::move(one));
        const auto where = " (RF delay " + std::to_string(expected.rf_delay) + ", URF delay " +
            std::to_string(expected.urf_delay) + ", " + std::to_string(expected.rf_banks) + " RF banks)";
        const auto bits_of = [&](const std::string& name) {
            for (const auto& size : varied.state_sizes()) {
                if (size.name == name) return size.bits().value_or(0);
            }
            return uint64_t{0};
        };
        require(bits_of("dispatch pipeline") == expected.dispatch,
            "the dispatch pipeline declares " + std::to_string(bits_of("dispatch pipeline")) + " bits" + where);
        require(bits_of("arithmetic pipeline: operand latches") == expected.arithmetic,
            "the arithmetic pipeline declares " + std::to_string(bits_of("arithmetic pipeline: operand latches")) +
                " bits" + where);
        const auto text = varied.state_report();
        require(text.find("  vector register file (register file): not signed off\n    latency " +
                    std::to_string(expected.rf_delay) + " (rf.read_latency_cycles)\n") != std::string::npos &&
                text.find("read first\n    read data in flight: " + std::string(expected.rf_note)) !=
                    std::string::npos &&
                text.find("  uniform register file (uniform register file): not signed off\n    latency " +
                    std::to_string(expected.urf_delay) + " (urf.read_latency_cycles)\n") != std::string::npos &&
                text.find("2R 1W, read first\n    read data in flight: " + std::string(expected.urf_note)) !=
                    std::string::npos,
            "the register files' black boxes: their read delays and the read data in flight" + where);
        // The ALU's ur2r word: in the operand latches while the URF is the
        // faster, from the URF's read registers when the delays are equal.
        const std::string ur2r = expected.urf_delay < expected.rf_delay
            ? "uniform operand (ur2r): 32 bits, the arithmetic pipeline's operand latches (in the open)\n"
            : "uniform operand (ur2r): 32 bits, the uniform register file's read registers (the memory's)\n";
        require(text.find(ur2r) != std::string::npos, "the ALU's ur2r input names the wrong source" + where);
    }
}

// The host's view of shared memory is shmem.bytes long, though the memory's
// last row (4-byte words x banks) is whole.
void test_shmem_host_view_is_the_configured_bytes() {
    auto config = canonical_profile();
    config.shmem_bytes = 65532;
    nanosm::Launch launch;
    launch.active_warps = 1;
    const nanosm::Machine machine(config, compile("halt\n", config), std::move(launch));
    require(machine.shmem().size() == 65532, "the host sees more shared memory than shmem.bytes");
    (void)machine.read_shmem_u32(65528);
    bool refused = false;
    try {
        (void)machine.read_shmem_u32(65532);
    } catch (const nanosm::SimError&) {
        refused = true;
    }
    require(refused, "a host read past shmem.bytes succeeded");
}

// A group of single bits: a member's value is read by value (the bits have no
// address to refer to), holds across a tick nobody writes it, and takes its
// own write.
void test_regs_one_bit_members() {
    nanosm::Sanitizer sanitizer;
    uint64_t cycle = 0;
    nanosm::AllClocked all(sanitizer, cycle);
    nanosm::Regs<bool> bits(all,"test bits", 1);
    bits.build(3, false);
    const nanosm::Regs<bool>& view = bits;
    sanitizer.record_cycle_start();
    bits.set_next(1, true, "set");
    all.commit();
    const bool held = bits[1].cur();
    require(held && view[1].cur() && !bits[0].cur() && !view.at(2).cur(), "a one-bit member did not take its write");
    sanitizer.record_cycle_start();
    bits[2].set_next(true, "set");
    all.commit();
    require(bits[1].cur() && bits.at(2).cur() && !bits[0].cur(), "a one-bit member did not hold across a tick");
}

// Sixteen stores of one warp on one depbar, back to back, on a profile whose
// A-write pipeline is long: a store leaves its slot at its last grant and
// still counts on its depbar until that write reaches the RAM, so more
// stores than there are slots count on sb0 at once. The counter is wide
// enough for all of them (Machine::depbar_counter_max_) and every store
// lands.
void test_depbar_counts_stores_on_their_way_to_ram() {
    auto profile = canonical_profile();
    profile.shmem_port_a_depth = 64;
    profile.mbar_wait_slots = 1;
    profile.tma_queue_commands = 1;
    nanosm::validate_device_profile(profile);
    std::string body = "mov r1, 7\nsread r0, LANE_ID\nshli r0, r0, 2\n";
    for (uint32_t store = 0; store < 16; ++store) {
        body += "sts [r0 + " + std::to_string(64 * store) + "], r1 {pred=PT, stall=0, set=sb0, wait=-}\n";
    }
    body += "halt {pred=PT, stall=0, set=-, wait=sb0}\n";
    nanosm::Launch launch;
    launch.active_warps = 1;
    nanosm::Machine machine(profile, hand_assemble(i1_asm(profile, body), profile), std::move(launch));
    uint32_t most = 0;
    while (!machine.done()) {
        tick_capped(machine);
        most = std::max(most, machine.depbar_value(0, 0));
    }
    require(most > profile.shmem_store_slots, "the stores never outnumbered the store slots on sb0; re-derive the test");
    for (uint32_t store = 0; store < 16; ++store) {
        for (uint32_t lane = 0; lane < profile.lanes; ++lane) {
            uint32_t word = 0;
            std::memcpy(&word, machine.shmem().data() + 64 * store + 4 * lane, 4);
            require(word == 7,
                "store " + std::to_string(store) + " did not land in lane " + std::to_string(lane));
        }
    }
}

// The stg.v8 store-data read pipe (lsu_read_pipe_): a register-file read
// granted at t fills the store buffer at the edge of t + Lr, neither earlier
// nor later, so a gap between two grants reappears as the same gap at the
// buffer. Other instructions take the read ports while the store reads, so
// the grants come both back to back and with gaps. At two read latencies.
void test_store_data_read_pipe_keeps_gaps() {
    for (const uint32_t lr : {2u, 3u}) {
        auto config = canonical_profile();
        config.rf_read_latency_cycles = lr;
        config.urf_issue_to_port_cycles = std::max(config.urf_issue_to_port_cycles, 1 + lr);
        const uint32_t width = config.lsu_store_group_lanes;
        nanosm::Launch launch;
        launch.active_warps = 1;
        image_bytes(launch.gmem_image, 4096);
        // r2 = 2048 + 32 x lane; r(24 + j) = 1000 j + lane.
        std::string text = "sread r4, LANE_ID\nshli r2, r4, 5\naddi r2, r2, 2048\n";
        for (uint32_t j = 0; j < 8; ++j)
            text += "addi r" + std::to_string(24 + j) + ", r4, " + std::to_string(1000 * j) + "\n";
        text += "stg.v8 [r2], r24\n";
        // Issue reads r24 (bank 0) and r25 (bank 1) while the store wants them.
        for (uint32_t i = 0; i < 14; ++i) text += "add r40, r24, r25\n";
        text += "halt\n";
        nanosm::Machine machine(config, compile(text, config), std::move(launch), nanosm::TraceOptions{.timeline = true});
        machine.enable_edge_probe();
        struct After {
            std::optional<uint64_t> owner;
            uint32_t group = 0;
            std::vector<uint8_t> valid;
            std::vector<uint32_t> words;
        };
        std::map<uint64_t, After> after; // the store buffer after each cycle's edge
        while (!machine.done()) {
            const auto cycle = machine.cycle();
            tick_capped(machine);
            after[cycle] = {machine.lsu_buffer_owner(1), machine.lsu_buffer_group(1), machine.lsu_buffer_valid(1),
                machine.lsu_buffer_words(1)};
        }
        std::vector<std::pair<uint64_t, uint32_t>> grants; // (cycle, register)
        for (const auto& event : machine.edge_events())
            if (event.edge == LsuEdge::rf_read_beat) grants.emplace_back(event.cycle, event.aux);
        const auto where = " (Lr = " + std::to_string(lr) + ")";
        require(grants.size() == 8 * (16 / width), "the stg.v8 did not make eight reads per lane group" + where);
        bool back_to_back = false;
        bool gap_in_group = false;
        for (std::size_t i = 0; i < grants.size(); ++i) {
            const auto [cycle, reg] = grants[i];
            const uint32_t group = static_cast<uint32_t>(i / 8);
            const uint32_t beat = static_cast<uint32_t>(i % 8);
            require(reg == 24 + beat, "the reads are not rS .. rS + 7 in order" + where);
            if (beat != 0) {
                back_to_back = back_to_back || cycle == grants[i - 1].first + 1;
                gap_in_group = gap_in_group || cycle > grants[i - 1].first + 1;
            }
            const auto& landed = after.at(cycle + lr);
            const auto& before = after.at(cycle + lr - 1);
            const auto at = " at grant " + std::to_string(i) + " (cycle " + std::to_string(cycle) + ")" + where;
            require(landed.owner == 0u && landed.group == group, "the store buffer has the wrong owner or group" + at);
            for (uint32_t lane = group * width; lane < (group + 1) * width; ++lane) {
                const uint32_t slot = (lane - group * width) * 8 + beat;
                require(landed.valid.at(slot) == 1 && landed.words.at(slot) == 1000 * beat + lane,
                    "the beat is not in the store buffer Lr after its grant" + at);
                require(!(before.owner == 0u && before.group == group && before.valid.at(slot) == 1),
                    "the beat is in the store buffer earlier than Lr after its grant" + at);
            }
        }
        require(back_to_back && gap_in_group,
            "the run has no back-to-back pair or no gap between a group's reads; the test proves little" + where);
        // The machine reports the pipe with its stages.
        bool reported = false;
        for (const auto& size : machine.state_sizes()) {
            reported = reported || (size.name == "LSU store-data read pipeline" &&
                std::string(size.kind) == "pipeline" && size.entries == lr);
        }
        require(reported, "the state sizes do not list the store-data read pipe with Lr stages" + where);
        for (uint32_t lane = 0; lane < 16; ++lane)
            for (uint32_t j = 0; j < 8; ++j)
                require(machine.read_gmem_u32(2048 + 32 * lane + 4 * j) == 1000 * j + lane,
                    "the stg.v8 wrote the wrong word" + where);
    }
}

// The sanitizer's check for the case above: a reader of a buffer
// that takes a lane whose valid bit is clear panics.
void test_sanitizer_buffer_reader_takes_valid_lanes() {
    nanosm::Sanitizer sanitizer;
    std::vector<uint8_t> valid(16, 1);
    sanitizer.check_buffer_lanes_valid(5, 0, valid, 0, 16, "RF drain");
    valid[3] = 0;
    sanitizer.check_buffer_lanes_valid(5, 0, valid, 4, 12, "egress write port"); // lanes 4..15 only
    try {
        sanitizer.check_buffer_lanes_valid(5, 0, valid, 3, 1, "egress write port");
    } catch (const nanosm::SimError& error) {
        const std::string message = error.what();
        require(message.find("egress write port read lane 3 of LSU buffer 0") != std::string::npos,
            "the lane check's PANIC does not name the reader, lane and register: " + message);
        return;
    }
    throw std::runtime_error("a read of an invalid assembly lane did not panic");
}

// ---- The mbar event registers: both engines alike ----

// A TMA load and an MMA commit, far apart: each engine's event, presented in
// cycle p, is first claimable in p + 1, and the free RMW engine claims it
// then, never in p itself.
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
// A completion reaches issue through the counter (Q), so an instruction
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
    // The "mma.async" event is the MMA's acceptance, at dispatch. The
    // third issues in the cycle the head retires (look-ahead (c): the queue
    // as that cycle leaves it) and reaches the queue issue_to_dispatch later:
    // the pop and the push do not share a cycle (a claim waits for the
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

// The ingress FIFO's declared full flag is registered (an implementer
// choice): when the FIFO is full at the start of a cycle, the bus
// cannot write a beat in that cycle even if the read port frees room in it;
// the beat lands the next cycle. A FIFO that is not full is read and written
// in one cycle (count' = count + push - pop).
void test_simultaneous_ingress_push_and_pop() {
    auto config = fast_bus_profile(8); // one 32 B beat per pop: a pop frees a whole beat's room
    config.lsu_ingress_sectors = 2;
    // Warp 1's long bookings hold the first group's RF write for a while, so
    // the FIFO fills (a mov storm does not; see booking_stream).
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

// ---- The driver shuffle (regression check; the launches run under ctest
// via scripts/driver_shuffle_check.py) ----

// Register drivers read only Q and the wires, so permuting their order (and
// the independent request generators') every cycle changes nothing: not a
// cycle, a counter, a trace byte or a memory byte. A check against a future
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
        with_long_bookings(config); // a mov storm does not hold the RF write port
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

// pipe: advance, hold, replace, leave
TEST(Stepping, pipe_advance_hold_replace) {
    test_pipe_advance_hold_replace();
}

// pipe: a second write of one point PANICs
TEST(Stepping, pipe_duplicate_write_panics) {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    test_pipe_duplicate_write_panics();
}

// payloads: a maximum and a count; the profile is checked against the maxima
TEST(Stepping, bounded_payload_and_limits) {
    test_bounded_payload_and_limits();
}

// group: members hold or take their own write; one member written twice PANICs
TEST(Stepping, regs_members_hold_write_and_panic) {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    test_regs_members_hold_write_and_panic();
}

// FIFO: oldest first, read latency, port widths, capacity, one-bit entries, zero capacity
TEST(Stepping, fifo_order_latency_and_widths) {
    test_fifo_order_latency_and_widths();
}

// FIFO: a second push in one tick PANICs
TEST(Stepping, fifo_second_push_panics) {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    test_fifo_second_push_panics();
}

// memory: read first, element enables, one port one word, its size
TEST(Stepping, memory_read_first_and_enables) {
    test_memory_read_first_and_enables();
}

TEST(Stepping, black_box_latency_and_interval) {
    test_black_box_latency_and_interval();
}
// state report: declared bits by unit add up; exactly the listed four are not declared
TEST(Stepping, state_report_adds_up) {
    test_state_report_adds_up();
}

// shared memory: the host sees shmem.bytes, not the padded last row
TEST(Stepping, shmem_host_view_is_the_configured_bytes) {
    test_shmem_host_view_is_the_configured_bytes();
}

// memory: a port asked for a second word in one cycle PANICs
TEST(Stepping, memory_port_panics) {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    test_memory_port_panics();
}

// group: a one-bit member is read by value and holds
TEST(Stepping, regs_one_bit_members) {
    test_regs_one_bit_members();
}

// depbar: stores on their way to the RAM still count; the counter is wide enough
TEST(Stepping, depbar_counts_stores_on_their_way_to_ram) {
    test_depbar_counts_stores_on_their_way_to_ram();
}

// stg.v8's store-data read pipe: a grant at t lands at t + Lr; gaps are kept
TEST(Stepping, store_data_read_pipe_keeps_gaps) {
    test_store_data_read_pipe_keeps_gaps();
}

// the load buffer's lane groups: eight RF writes each, after the group's last word
TEST(Stepping, load_buffer_groups) {
    test_load_buffer_groups();
}

// assembly reuse: final load drain -> scalar store
TEST(Stepping, buffer_reuse_load_to_scalar_store) {
    test_buffer_reuse_load_to_scalar_store();
}

// assembly reuse: final load drain -> vector store
TEST(Stepping, buffer_reuse_load_to_vector_store) {
    test_buffer_reuse_load_to_vector_store();
}

// assembly reuse: final load drain -> load
TEST(Stepping, buffer_reuse_load_to_load) {
    test_buffer_reuse_load_to_load();
}

// sanitizer: assembly reader takes only valid lanes
TEST(Stepping, sanitizer_buffer_reader_takes_valid_lanes) {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    test_sanitizer_buffer_reader_takes_valid_lanes();
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

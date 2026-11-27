// The stepping discipline: the kinds of clocked state (Reg, Regs, Pipe, Fifo, Memory, BlackBox, Bounded payloads),
// the one-writer rule, the state report, LSU buffer reuse, events that land in one cycle.
#include "test_support.hpp"

#include <sstream>
#include <string>
#include <type_traits>

namespace {

using namespace nanosm_test;

// Runs `machine` to the end, tick by tick: what `read()` returns after each
// cycle's edge, by cycle.
template <typename Read>
std::map<uint64_t, std::invoke_result_t<Read&>> after_each_tick(nanosm::Machine& machine, Read read) {
    std::map<uint64_t, std::invoke_result_t<Read&>> after;
    while (!machine.done()) {
        const auto cycle = machine.cycle();
        tick_capped(machine);
        after[cycle] = read();
    }
    return after;
}

// One of the LSU's two buffers (0 load, 1 store) at the end of a cycle.
struct BufferAfter {
    std::optional<uint64_t> owner;
    std::vector<uint8_t> valid;
    uint32_t valid_count = 0;
};

BufferAfter buffer_after(const nanosm::Machine& machine, uint32_t buffer) {
    return {machine.lsu_buffer_owner(buffer), machine.lsu_buffer_valid(buffer), machine.lsu_buffer_valid_count(buffer)};
}

struct ReuseRun {
    std::unique_ptr<nanosm::Machine> machine;
    uint64_t reuse_cycle = 0;                                          // the second command takes the slot
    std::map<uint64_t, std::pair<BufferAfter, BufferAfter>> after;     // (load, store) buffer after each cycle
};

// `old_load` is LSU command 0, `second` (opcode `second_name`) command 1, with
// one command slot: the second waits at the LSU queue's head and takes the
// slot in the cycle the load's final RF write releases it (d). gmem holds
// pattern words; r5 = 4 x lane, r6 = 2048 + 4 x lane, r2 = 2048 + 32 x lane,
// r0 = 0, r24 .. r31 = 101 .. 108.
ReuseRun run_reuse_after_final_drain(const std::string& old_load, const std::string& second,
    const std::string& second_name) {
    auto config = fast_bus_profile(2);
    config.lsu_active_commands = 1;
    auto launch = launch_warps(1);
    fill_pattern(image_bytes(launch.gmem_image, 4096), 0, 1024, 0);
    std::string text = "sread r4, LANE_ID\nshli r5, r4, 2\nori r6, r5, 2048\nmov r0, 0\nshli r2, r4, 5\naddi r2, r2, 2048\n";
    for (uint32_t j = 0; j < 8; ++j) text += "mov r" + std::to_string(24 + j) + ", " + std::to_string(101 + j) + "\n";
    text += old_load + "\n" + second + "\nhalt\n";
    ReuseRun run;
    run.machine = std::make_unique<nanosm::Machine>(
        config, compile(text, config), std::move(launch), nanosm::TraceOptions{.timeline = true});
    auto& machine = *run.machine;
    machine.enable_edge_probe();
    run.after = after_each_tick(machine, [&] { return std::pair{buffer_after(machine, 0), buffer_after(machine, 1)}; });
    uint64_t final_drain = 0;
    for (const auto& event : machine.edge_events())
        if (event.edge == LsuEdge::rf_write_beat && event.id == 0) final_drain = std::max(final_drain, event.cycle);
    std::optional<uint64_t> admitted;
    for (const auto& event : events_named(machine, second_name))
        if (event.category == nanosm::TraceCategory::gmem && event.id == 1) admitted = event.start_cycle;
    run.reuse_cycle = final_drain;
    require(admitted == run.reuse_cycle,
        second_name + " did not take the slot in the cycle '" + old_load + "' made its final RF write");
    return run;
}

// The old load's data: an ldg.v8 of the sector at 0 into every lane (r8 + k
// holds word k), or an ldg of word `lane` into r8.
void require_old_load_data(const nanosm::Machine& machine, const std::string& old_load) {
    const bool vector = old_load.starts_with("ldg.v8");
    for (uint32_t k = 0; k < (vector ? 8u : 1u); ++k) {
        require_lanes(machine, 8 + k, [&](uint32_t lane) { return pattern_word(vector ? k : lane); }, old_load);
    }
}

} // namespace

TEST(Stepping, reg_hold_write_commit) {
    nanosm::Sanitizer sanitizer;
    uint64_t cycle = 7;
    nanosm::AllClocked all(sanitizer, cycle);
    nanosm::Reg<int> reg(all, "test register", 5);
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

// One net, one driver: a second write in one tick of a register (even by the
// same writer), a group's member, a pipeline's push, stage replacement or
// hold, a FIFO's push, or a memory's port is a PANIC naming the state, the
// cycle and both writers. A new tick takes one write again; different write
// points of one pipeline are different nets.
TEST(Stepping, reg_duplicate_write_panics) {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    using Writes = std::function<void(nanosm::AllClocked&, nanosm::Sanitizer&)>;
    struct Row {
        std::string what;
        Writes writes; // builds its state, starts the cycle, writes twice
        std::vector<std::string> needles;
    };
    const auto reg_twice = [](const char* first, const char* second) {
        return [=](nanosm::AllClocked& all, nanosm::Sanitizer& sanitizer) {
            nanosm::Reg<int> reg(all, "test register", 0);
            sanitizer.record_cycle_start();
            reg.set_next(1, first);
            reg.set_next(2, second);
        };
    };
    const auto pipe_twice = [](const std::function<void(nanosm::Pipe<int>&, const char*)>& write) {
        return [=](nanosm::AllClocked& all, nanosm::Sanitizer& sanitizer) {
            nanosm::Pipe<int> pipe(all, "test pipeline");
            pipe.build(2);
            sanitizer.record_cycle_start();
            write(pipe, "first");
            write(pipe, "second");
        };
    };
    const auto memory_twice = [](bool write) {
        return [=](nanosm::AllClocked& all, nanosm::Sanitizer& sanitizer) {
            nanosm::Memory<uint32_t> memory(all, "ported memory");
            memory.build(nanosm::MemoryShape{2, 4, 1, 32, 1, 1, 1, nanosm::SameCycle::read_first});
            sanitizer.record_cycle_start();
            const std::array<uint32_t, 1> value{1};
            const auto every = nanosm::Memory<uint32_t>::every_element;
            if (write) {
                memory.write(0, 1, value, every, "first");
                memory.write(1, 1, value, every, "the other bank");
                memory.write(0, 2, value, every, "second");
            } else {
                (void)memory.read(0, 1, "first", 0);
                (void)memory.read(1, 2, "the other bank", 0);
                (void)memory.read(0, 2, "second", 0);
            }
        };
    };
    const std::vector<std::string> pipe_needles = {"test pipeline", "cycle=7", "first by 'first'", "then by 'second'"};
    const std::vector<Row> rows = {
        {"register, two writers", reg_twice("drain", "fill"),
            {"register 'test register'", "cycle=7", "first by 'drain'", "then by 'fill'"}},
        {"register, the same writer twice", reg_twice("drain", "drain"),
            {"register 'test register'", "cycle=7", "first by 'drain'", "then by 'drain'"}},
        {"group member",
            [](nanosm::AllClocked& all, nanosm::Sanitizer& sanitizer) {
                nanosm::Regs<int> group(all, "test group");
                group.build(4, 9);
                sanitizer.record_cycle_start();
                group.set_next(2, 1, "first");
                group.set_next(2, 2, "second");
            },
            {"test group [2]", "cycle=7", "first by 'first'", "then by 'second'"}},
        {"pipeline push", pipe_twice([](nanosm::Pipe<int>& pipe, const char* tag) { pipe.push(1, tag); }), pipe_needles},
        {"pipeline replace", pipe_twice([](nanosm::Pipe<int>& pipe, const char* tag) { pipe.replace(0, 2, tag); }),
            pipe_needles},
        {"pipeline hold", pipe_twice([](nanosm::Pipe<int>& pipe, const char* tag) { pipe.hold(tag); }), pipe_needles},
        {"FIFO push",
            [](nanosm::AllClocked& all, nanosm::Sanitizer& sanitizer) {
                nanosm::Fifo<int> fifo(all, "second FIFO");
                fifo.build(2);
                sanitizer.record_cycle_start();
                fifo.push(1, "first");
                fifo.push(2, "second");
            },
            {"second FIFO (push)", "first by 'first'", "then by 'second'"}},
        {"memory read port", memory_twice(false),
            {"PANIC cycle=7", "ported memory' bank 0: read port 0 asked for row 2 after row 1", "'second'"}},
        {"memory write port", memory_twice(true),
            {"PANIC cycle=7", "ported memory' bank 0: a write of row 2 with its 1 write port taken", "'second'"}},
    };
    for (const auto& row : rows) {
        nanosm::Sanitizer sanitizer;
        uint64_t cycle = 7;
        nanosm::AllClocked all(sanitizer, cycle);
        const auto message = error_of<nanosm::SimError>([&] { row.writes(all, sanitizer); }, row.what + ": a second write");
        for (const auto& needle : row.needles)
            require(message.find(needle) != std::string::npos,
                row.what + ": the duplicate-write PANIC lacks '" + needle + "': " + message);
    }

    nanosm::Sanitizer sanitizer;
    uint64_t cycle = 7;
    nanosm::AllClocked all(sanitizer, cycle);
    nanosm::Reg<int> reg(all, "test register", 0);
    sanitizer.record_cycle_start();
    reg.set_next(1, "drain");
    (void)error_of<nanosm::SimError>([&] { reg.set_next(2, "fill"); }, "a second write of the register");
    all.commit();
    sanitizer.record_cycle_start();
    reg.set_next(3, "next tick");
    all.commit();
    require(reg.cur() == 3, "the register did not recover after a duplicate-write PANIC");
    nanosm::Pipe<int> other(all, "other pipeline");
    other.build(2);
    sanitizer.record_cycle_start();
    other.push(1, "entry");
    other.replace(0, std::nullopt, "stage 0");
    other.replace(1, std::nullopt, "stage 1");
}

// A value pushed in tick t is in stage n - 1 after n edges and gone after N;
// a hold keeps every stage; a replacement travels with its value; a value
// replaced by nothing leaves at that stage.
TEST(Stepping, pipe_advance_hold_replace) {
    nanosm::Sanitizer sanitizer;
    uint64_t cycle = 0;
    nanosm::AllClocked all(sanitizer, cycle);
    nanosm::Pipe<int> pipe(all, "test pipeline");
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
    require(std::string(size.kind) == "pipeline" && size.entries == 3, "the pipeline does not report its stages");
}

// A payload of at most N items: it counts what it holds, compares only the
// counted items, refuses one more than N, and a profile past an architectural
// maximum is refused by name before a machine is built. A standard container
// is not a fixed payload (the compiler refuses it as clocked state).
TEST(Stepping, bounded_payload_and_limits) {
    nanosm::Bounded<uint32_t, 4> words;
    require(words.empty() && words.capacity() == 4, "a new payload is not empty");
    for (uint32_t value = 1; value <= 4; ++value) words.push_back(value);
    require(words.size() == 4 && words.front() == 1 && words.back() == 4, "a payload lost an item");
    (void)error_of<nanosm::SimError>([&] { words.push_back(5); }, "a fifth item in a payload of four");
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
    expect_error<nanosm::ProfileError>([&] { nanosm::validate_device_profile(profile); }, {"shmem banks"},
        "a profile past an architectural maximum");
}

// A group is N registers under one name: a member written this tick takes its
// value at the edge, a member nobody writes holds, and different members may
// be written in one tick. A group of single bits reads a member by value (the
// bits have no address to refer to).
TEST(Stepping, regs_members_hold_and_write) {
    nanosm::Sanitizer sanitizer;
    uint64_t cycle = 7;
    nanosm::AllClocked all(sanitizer, cycle);
    nanosm::Regs<int> group(all, "test group");
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
    require(std::string(size.kind) == "group" && size.entries == 4, "the group does not report its members");

    nanosm::Regs<bool> bits(all, "test bits");
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

// A FIFO: the next pop returns the oldest entry; an entry pushed at t is
// readable from t + its read latency; a pop and a push in one tick; its ports'
// widths; a push past its capacity is an internal error. A FIFO of single
// bits reads a bit by value and holds it across a tick; a zero-capacity FIFO
// takes a pop of nothing and holds.
TEST(Stepping, fifo_order_latency_and_widths) {
    nanosm::Sanitizer sanitizer;
    uint64_t cycle = 3;
    nanosm::AllClocked all(sanitizer, cycle);
    nanosm::Fifo<int> fifo(all, "test FIFO");
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
    require(std::string(size.kind) == "FIFO" && size.entries == 4 && size.write_ports == 2 && size.read_ports == 2 &&
            size.read_delay == 2,
        "the FIFO does not report its capacity, its ports' widths and its read latency");
    (void)error_of<nanosm::SimError>([&] { (void)fifo.front(2); }, "a read past the read port's width");
    fifo.push(std::vector<int>{4, 5}, "writer");
    tick();
    fifo.push(6, "writer");
    expect_error<nanosm::SimError>(tick, {"holds more than its 4 entries"}, "a push past the capacity");
    // A machine stops at an internal error; these two get an AllClocked of their own.
    nanosm::Sanitizer quiet;
    nanosm::AllClocked other(quiet, cycle);
    nanosm::Fifo<bool> bits(other, "test bits FIFO");
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

// A memory: a read sees the word as it was at the start of the cycle and a
// write lands at the clock edge (read first); a write enables its elements one
// by one; two reads of one word through one port are one read; a memory with
// no write port refuses a write; it reports its words, width and ports.
TEST(Stepping, memory_read_first_and_enables) {
    nanosm::Sanitizer sanitizer;
    uint64_t cycle = 0;
    nanosm::AllClocked all(sanitizer, cycle);
    nanosm::Memory<uint32_t> memory(all, "test memory");
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
    // Element 2 of a two-element word would be bank 1's word in the storage.
    (void)error_of<nanosm::SimError>([&] { memory.poke(0, 0, 2, 99); }, "a poke past the word's elements");
    require(memory.peek(1, 0)[0] == 0, "a poke past the word's elements reached the next word");
    nanosm::Memory<uint32_t> rom(all, "read-only memory");
    rom.build(nanosm::MemoryShape{1, 2, 1, 32, 1, 0, 1, nanosm::SameCycle::read_only});
    const std::array<uint32_t, 1> value{1};
    (void)error_of<nanosm::SimError>(
        [&] { rom.write(0, 0, value, nanosm::Memory<uint32_t>::every_element, "writer"); },
        "a write into a memory with no write port");
    const auto size = memory.size();
    require(std::string(size.kind) == "memory" && size.entries == 8 && size.banks == 2 && size.word_bits == 64 &&
            size.read_ports == 1 && size.write_ports == 1 && size.read_delay == 2,
        "the memory does not report its words, width, banks and ports");
}

// A black box's output register holds a value from its inputs' cycle +
// latency, for that one cycle; nothing passes through it in one cycle, and
// inputs closer than its interval are refused (a refused value is not a
// write: a good one may follow in the same tick). It refuses a value before
// it is built; one that keeps an inside is built with the inside's idle
// question, and is not empty while the inside holds work.
TEST(Stepping, black_box_latency_and_interval) {
    nanosm::Sanitizer sanitizer;
    uint64_t cycle = 0;
    nanosm::AllClocked all(sanitizer, cycle);
    nanosm::BlackBox<uint32_t> black_box(all, nanosm::Unit::none, "a black box");
    const auto refuse = [&](uint32_t value, uint64_t taken_at, const std::string& what) {
        (void)error_of<nanosm::SimError>([&] { black_box.deliver(value, taken_at, "test"); }, what);
    };
    refuse(1, 0, "a value before the black box is built");
    nanosm::BlackBoxDeclaration declaration("a black box");
    declaration.latency = 3;
    declaration.interval = 2;
    black_box.build(declaration);
    std::vector<std::optional<uint32_t>> outputs;
    for (; cycle < 8; ++cycle) {
        sanitizer.record_cycle_start();
        if (cycle == 0) black_box.deliver(10, 0, "test");
        if (cycle == 1) refuse(11, 1, "an input one cycle after another (interval 2)");
        if (cycle == 2) {
            refuse(12, 1, "an input one cycle after another (interval 2)");
            black_box.deliver(12, 2, "test"); // after a refusal in the same tick
        }
        if (cycle == 7) refuse(13, 4, "a value due now (it would pass in one cycle)");
        outputs.push_back(black_box.output());
        all.commit();
    }
    const std::vector<std::optional<uint32_t>> expected{
        std::nullopt, std::nullopt, std::nullopt, 10u, std::nullopt, 12u, std::nullopt, std::nullopt};
    require(outputs == expected, "the black box's output register does not hold each value at its inputs' cycle + 3");
    require(black_box.empty(), "the black box is not empty after its outputs");

    nanosm::BlackBox<uint32_t, int> keeper(all, nanosm::Unit::none, "a black box with an inside");
    nanosm::BlackBoxDeclaration keeps("a black box with an inside");
    (void)error_of<nanosm::SimError>([&] { keeper.build(keeps); }, "a black box with an inside but no idle question");
    keeper.build(keeps, [](const int& waiting) { return waiting == 0; });
    keeper.inside() = 1;
    require(!keeper.empty(), "a black box with work inside says it is empty");
    keeper.inside() = 0;
    require(keeper.empty(), "an idle black box says it is not empty");
}

// The state report lists every piece of clocked state, each in a unit, and
// prints every black box with its unit and latency.
TEST(Stepping, state_report_inventory) {
    const nanosm::Machine machine(canonical_profile(), compile("halt\n"), launch_warps(1));
    const auto report = machine.state_report();
    for (const auto& size : machine.state_sizes()) {
        require(size.unit != nanosm::Unit::none, "'" + size.name + "' belongs to no unit");
        require(report.find(" " + size.name) != std::string::npos, "the report does not list '" + size.name + "'");
    }
    for (const auto& black_box : machine.black_boxes()) {
        require(report.find("  " + black_box.name + " (" + nanosm::unit_name(black_box.unit) + ")\n    latency " +
                    std::to_string(black_box.latency) + " (") != std::string::npos,
            "the report does not print black box '" + black_box.name + "' with its latency");
    }
}

// The machine's black boxes are these ten, by name. Only the names are
// checked: a change to what one takes in, gives out or how long it takes
// passes this test (and is still to be said plainly).
TEST(Stepping, black_boxes_are_the_listed_ten) {
    const nanosm::Machine machine(canonical_profile(), compile("halt\n"), launch_warps(1));
    std::vector<std::string> names;
    for (const auto& black_box : machine.black_boxes()) names.push_back(black_box.name);
    std::sort(names.begin(), names.end());
    const std::vector<std::string> expected{"ALU", "FPU", "MMA core", "divergence stack", "instruction memory",
        "shared memory", "special-function unit", "tensor memory", "uniform register file", "vector register file"};
    require(names == expected, "the machine's black boxes are not the listed ten");
}

// An LSU buffer is owned only while data is in it, one buffer per direction.
// After a load's final RF drain, with one command slot, the next command
// takes the slot in that cycle (run_reuse_after_final_drain), reads its
// registers from the next and finds the load buffer empty and unowned:
// - a scalar store's address and data are read in the cycle after (their
//   banks differ) and the data is in the store buffer, sixteen valid lanes,
//   Lr later; not before (its egress reads the words without valid_count,
//   so the output alone cannot show it);
// - a stg.v8 is planned when its address is at the RF output, its first RF
//   data read is sampled the cycle after and fills the store buffer Lr
//   later; the buffer has no owner until then;
// - a new load leaves the load buffer unowned until its first word returns.
TEST(Stepping, buffer_reuse_load_to_scalar_store) {
    using Check = std::function<void(const ReuseRun&, const std::string&)>;
    struct Row {
        std::string old_load;
        std::string second;
        std::string second_name;
        Check check;
    };
    const auto lr = canonical_profile().rf_read_latency_cycles;
    const Check scalar_store = [lr](const ReuseRun& run, const std::string& where) {
        require(!run.after.at(run.reuse_cycle + lr).second.owner.has_value(),
            "the store buffer is owned before the store's data read landed" + where);
        const auto& [load, store] = run.after.at(run.reuse_cycle + 1 + lr);
        require(store.owner == 1u, "the store does not own the store buffer" + where);
        require(store.valid_count == 16 && std::count(store.valid.begin(), store.valid.begin() + 16, 1) == 16,
            "the store's valid bits were lost" + where);
        require(!load.owner.has_value() && load.valid_count == 0,
            "the load buffer is still owned after the load's final drain" + where);
        require_lanes(gmem_lanes(*run.machine, 2048), [](uint32_t lane) { return 2048 + 4 * lane; },
            "the scalar store's word" + where);
    };
    const Check vector_store = [lr](const ReuseRun& run, const std::string& where) {
        // Its address read at reuse + 1, planned at reuse + 1 + Lr, its first
        // data beat the cycle after.
        const auto first_beat = run.reuse_cycle + 2 + lr;
        bool first_read_now = false;
        for (const auto& event : run.machine->edge_events())
            if (event.edge == LsuEdge::rf_read_beat && event.id == 1 && event.cycle == first_beat) first_read_now = true;
        require(first_read_now, "the stg.v8's first RF data read was not the cycle after its plan" + where);
        const auto& store = run.after.at(first_beat).second;
        require(!store.owner.has_value() && store.valid_count == 0,
            "the store buffer is owned before the stg.v8's first beat has landed" + where);
        // The first beat brings r24 for the first group's lanes: one word a lane.
        const auto& filled = run.after.at(first_beat + lr).second;
        require(filled.owner == 1u && filled.valid_count == canonical_profile().lsu_store_group_lanes,
            "the stg.v8's first read beat is not in the store buffer Lr after it" + where);
        for (uint32_t j = 0; j < 8; ++j) {
            require_lanes(gmem_lanes(*run.machine, 2048 + 4 * j, 32), 101 + j,
                "the stg.v8's word of r" + std::to_string(24 + j) + where);
        }
    };
    const Check load = [](const ReuseRun& run, const std::string& where) {
        const auto& buffer = run.after.at(run.reuse_cycle).first;
        require(!buffer.owner.has_value() && buffer.valid_count == 0,
            "the load buffer is owned or not empty before the new load's first word returned" + where);
        require_lanes(*run.machine, 16, [](uint32_t lane) { return pattern_word(256 + lane); },
            "the new load's word" + where);
    };
    const std::vector<Row> rows = {
        {"ldg.v8 r8, [r0]", "stg [r5 + 2048], r6", "stg", scalar_store},
        {"ldg r8, [r5]", "stg [r5 + 2048], r6", "stg", scalar_store},
        {"ldg.v8 r8, [r0]", "stg.v8 [r2], r24", "stg.v8", vector_store},
        {"ldg.v8 r8, [r0]", "ldg r16, [r5 + 1024]", "ldg", load},
    };
    for (const auto& row : rows) {
        const auto run = run_reuse_after_final_drain(row.old_load, row.second, row.second_name);
        row.check(run, " (" + row.old_load + " -> " + row.second + ", reuse cycle " + std::to_string(run.reuse_cycle) + ")");
        require_old_load_data(*run.machine, row.old_load);
    }
}

// A depbar completion and an admission on one counter in one cycle are one
// calculation, 1 - 1 + 1 = 1; neither is deferred. The lds.uni on sb0
// completes when its RF write lands: its A-read grant (issue +
// issue_to_shmem_grant: its queue, collect and serve stages), D + Lb to its
// word, one cycle to the RF write decision, Wd to the write. `pad` movs put
// the ldg, which sets sb0 without waiting on it, at that cycle. The lds.uni
// reads the umov's ur0 at the uniform distance; the ldg's base r0 is never
// written (0).
TEST(Stepping, simultaneous_depbar_completion_and_admission) {
    const auto& profile = canonical_profile();
    const uint64_t lds_issue = nanosm::fetch_to_issue + nanosm::issue_to_urf_visible(profile) -
        nanosm::issue_register_stages;
    const auto completion_cycle = lds_issue + issue_to_shmem_grant(profile) + profile.shmem_port_a_depth +
        profile.shmem_read_latency_cycles + 1 + profile.rf_write_depth_cycles;
    const auto pad = completion_cycle - lds_issue - 1;
    std::string body = "umov ur0, 0\nlds.uni r1, [ur0] {pred=PT, stall=0, set=sb0, wait=-}\n";
    for (uint64_t i = 0; i < pad; ++i) body += "mov r5, 7\n";
    body += "ldg r3, [r0] {pred=PT, stall=0, set=sb0, wait=-}\nhalt {pred=PT, stall=0, set=-, wait=sb0}\n";
    auto launch = launch_warps(1);
    write_u32(image_bytes(launch.test_shmem, 64), 0, 0xabcd);
    write_u32(image_bytes(launch.gmem_image, 64), 0, 0x1234);
    nanosm::Machine machine(profile, hand_assemble(hand_asm(profile, body), profile), std::move(launch), {.timeline = true});
    const auto sb0_after = after_each_tick(machine, [&] { return machine.depbar_value(0, 0); });
    std::vector<std::pair<uint64_t, std::string>> sb0_events;
    for (const auto& event : events_named(machine, "depbar sb0"))
        if (event.track == 100) sb0_events.emplace_back(event.start_cycle, event.detail);
    uint64_t ldg_issue = 0;
    for (const auto& event : events_named(machine, "ldg"))
        if (event.category == nanosm::TraceCategory::instruction) ldg_issue = event.start_cycle;
    require(ldg_issue == completion_cycle, "the ldg did not issue at the lds's completion; re-derive the test");
    const auto completion = std::find(sb0_events.begin(), sb0_events.end(),
        std::make_pair(ldg_issue, std::string("value=0")));
    require(completion != sb0_events.end(), "the lds did not complete in the ldg's issue cycle");
    require(completion + 1 != sb0_events.end() && *(completion + 1) == std::make_pair(ldg_issue, std::string("value=1")),
        "the admission did not follow the completion in the same cycle");
    require_eq(sb0_after.at(ldg_issue), 1, "sb0 after the cycle (1 - 1 + 1)");
    require_lanes(machine, 1, 0xabcd, "the first load's value");
    require_lanes(machine, 3, 0x1234, "the second load's value");
}

// A full MMA queue (two entries): the third mma.async issues the cycle after
// the head retires (issue reads the queue's register) and reaches the queue
// issue_to_dispatch later. The "mma.async" event is the queue's acceptance,
// at dispatch.
TEST(Stepping, full_queue_issue_after_the_retire_cycle) {
    auto launch = launch_warps(1);
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
    machine.run();
    uint64_t third_accept = 0;
    for (const auto& event : events_named(machine, "mma.async"))
        if (event.category == nanosm::TraceCategory::mma && event.id == 3) third_accept = event.start_cycle;
    uint64_t head_retire = 0; // the first command's last result write
    for (const auto& event : events_named(machine, "MMA tmem write"))
        if (event.category == nanosm::TraceCategory::mma && event.id == 1)
            head_retire = std::max(head_retire, event.start_cycle);
    require(!refusals(machine, 0, "MMA command queue full").empty(), "the third mma.async never found the queue full");
    require_eq(third_accept, head_retire + 1 + nanosm::issue_to_dispatch(canonical_profile()),
        "the third mma.async's acceptance (issued the cycle after the head retired at c" + std::to_string(head_retire) +
            ")");
}

// The same for the TMA queue at one entry (tma.queue_commands = 1): of two
// tma.load.1d back to back, the second issues the cycle after the first
// command leaves the queue -- the cycle the RMW engine claims its completion
// event ("complete_tx (tma)" starts), which is the end of its "tma command"
// slice less one.
TEST(Stepping, full_tma_queue_issue_after_the_retire_cycle) {
    auto profile = canonical_profile();
    profile.tma_queue_commands = 1;
    constexpr uint32_t bytes = 256;
    auto launch = launch_warps(1);
    auto& shmem = image_bytes(launch.test_shmem, 8192);
    write_tma_descriptor(shmem, 5120, plain_tma_1d(bytes / 4));
    write_mbar(shmem, 4104, mbar_state(1, 0, static_cast<int32_t>(2 * bytes))); // both commands' bytes
    fill_pattern(image_bytes(launch.gmem_image, bytes), 0, bytes / 4, 0);
    const auto program = one_tma_operands(0) + "umov ur12, 1024\n"
        "tma.load.1d [ur10], [5120, ur8], [ur11]\n"
        "tma.load.1d [ur12], [5120, ur8], [ur11]\n" + wait_on("ur11") + "halt\n";
    nanosm::Machine machine(profile, compile(program, profile), std::move(launch), {.timeline = true});
    machine.run(20000);
    std::vector<uint64_t> issues;
    for (const auto& event : machine.trace().events())
        if (is_issue_event(event) && event.name == "tma.load.1d") issues.push_back(event.start_cycle);
    std::vector<uint64_t> claims; // the TMA's completion events' RMW starts
    for (const auto& [start, detail] : mbar_rmw_services(machine))
        if (detail.rfind("complete_tx (tma)", 0) == 0) claims.push_back(start);
    std::vector<uint64_t> command_ends;
    for (const auto& event : events_named(machine, "tma command"))
        command_ends.push_back(event.start_cycle + event.duration_cycles);
    std::sort(command_ends.begin(), command_ends.end());
    require(issues.size() == 2 && claims.size() == 2 && command_ends.size() == 2, "two tma loads, two completions");
    require(!refusals(machine, 0, "tma command queue full").empty(), "the second tma.load never found the queue full");
    require(command_ends[0] == claims[0] + 1, "the first command did not leave the queue at its event's claim");
    require_eq(issues[1], claims[0] + 1,
        "the second tma.load's issue (the cycle after the first command left the queue at its claim)");
    for (uint32_t i = 0; i < bytes / 4; ++i)
        require(machine.read_shmem_u32(4 * i) == pattern_word(i) && machine.read_shmem_u32(1024 + 4 * i) == pattern_word(i),
            "a tma load's word");
}

// One acc.ld engine: two warps' acc.ld.v8, the second refused ("acc transfer
// engine busy") until the first transfer's last read has launched, at L
// (its "acc.ld beat", the tmem sample, at L + tmem.port_depth: Mma.tmem_depths).
// The engine is empty from L + 1, and the second transfer issues then.
TEST(Stepping, busy_acc_engine_issue_after_it_empties) {
    const auto& profile = canonical_profile();
    auto launch = launch_warps(2);
    nanosm::Machine machine(profile, compile(R"(
        usread ur8, WARP_ID
        ushli ur8, ur8, 10
        acc.ld.v8 r16, [ur8, 64]
        halt
    )", profile), std::move(launch), {.timeline = true, .debug = true});
    machine.run(20000);
    std::array<std::vector<uint64_t>, 2> beats;
    for (const auto& event : events_named(machine, "acc.ld beat"))
        beats.at(event.detail.rfind("warp=0,", 0) == 0 ? 0 : 1).push_back(event.start_cycle);
    std::array<std::vector<uint64_t>, 2> issues;
    for (const auto& event : machine.trace().events())
        if (is_issue_event(event) && event.name == "acc.ld.v8")
            issues.at(event.detail.rfind("warp=0,", 0) == 0 ? 0 : 1).push_back(event.start_cycle);
    require(beats[0].size() == 8 && beats[1].size() == 8 && issues[0].size() == 1 && issues[1].size() == 1,
        "two acc.ld.v8, eight beats each");
    const auto first = issues[0][0] < issues[1][0] ? 0u : 1u;
    const auto second = 1 - first;
    require(!refusals(machine, second, "acc transfer engine busy").empty(), "the second acc.ld never found the engine busy");
    const auto last_launch = beats[first].back() - profile.tmem_port_depth;
    require_eq(issues[second][0], last_launch + 1,
        "the second acc.ld's issue (the cycle after the first transfer's last read launched)");
}

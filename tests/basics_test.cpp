// Encoding, the compiler's controls, branches, mbar basics, launch images.
#include "test_support.hpp"

using namespace nanosm_test;

namespace {

// Hand assembly (controls default; fixed stalls raised).
nanosm::Kernel assemble(const std::string& body) {
    return hand_assemble(hand_asm(canonical_profile(), body), canonical_profile());
}

// The launch (docs/qtx.md "Launch"): the kernel's `.param` lines are its signature,
// and with its `.mbar` and `.data` they are the launch block, in declaration
// order at the top of shared memory. The host writes the block into global
// memory; the launch copy brings it into shared memory from the first cycle,
// and each launch flag clears when its part has landed. The kernel reads its
// parameters and its constants and stores them to the global address it was
// given; qtxc makes each instruction that names a section wait on its flag.
const char* const argument_kernel = R"(
.warps 1
.param ptr  OUT
.param u32  N
.param f32  X
.param desc D
.equ SEVEN, 7
.mbar BAR, expect 3
.data K, f32 1.5, -inf
.data T, u32 SEVEN, 9
        umov  ur1, OUT
        umov  ur2, N
        umov  ur3, X
        umov  ur4, K
        umov  ur5, T
        lds.uni r1, [ur1]
        lds.uni r2, [ur2]
        lds.uni r4, [ur3]
        lds.uni r5, [ur4]
        lds.uni r7, [ur5]
        sread r6, LANE_ID
        shli  r6, r6, 2
        add   r9, r1, r6
        stg   [r9], r2
        stg   [r9 + 64], r4
        stg   [r9 + 128], r5
        stg   [r9 + 192], r7
        halt
)";

nanosm::Launch argument_launch() {
    nanosm::TmaDescriptor descriptor;
    descriptor.base = 0x1122334455667788ull;
    descriptor.element_bytes = 4;
    descriptor.dimensions = {16, 1, 1, 1};
    descriptor.tile_length = 16;
    descriptor.tile_rows = 1;
    const auto encoded = nanosm::encode_tma_descriptor(descriptor);
    nanosm::Launch launch; // no warp count: the kernel says .warps 1
    launch.args = {
        {"X", 2.5},
        {"OUT", uint64_t{4096}},
        {"D", std::vector<uint8_t>(encoded.begin(), encoded.end())},
        {"N", uint64_t{41}},
    };
    return launch;
}

} // namespace

TEST(Basics, encoding_round_trip) {
    nanosm::Instruction inst;
    inst.opcode = nanosm::Opcode::add;
    inst.control.stall = 21; // needs the fifth stall bit
    inst.control.set_depbar = 4;
    inst.control.wait_mask = 0b100101;
    inst.dst = 200;
    inst.src0 = 129;
    inst.src1 = 255;
    const auto decoded = nanosm::decode(nanosm::encode(inst));
    require(decoded.opcode == inst.opcode, "opcode encoding mismatch");
    require(decoded.control.stall == 21, "stall encoding mismatch");
    require(decoded.control.set_depbar == 4, "set-depbar encoding mismatch");
    require(decoded.control.wait_mask == 0b100101, "wait-mask encoding mismatch");
    require(
        decoded.dst == 200 && decoded.src0 == 129 && decoded.src1 == 255,
        "eight-bit register encoding mismatch");

    inst.opcode = nanosm::Opcode::addi;
    inst.imm = -1234567;
    const auto immediate_decoded = nanosm::decode(nanosm::encode(inst));
    require(immediate_decoded.dst == 200 && immediate_decoded.src0 == 129, "RRI register mismatch");
    require(immediate_decoded.imm == -1234567, "signed 22-bit immediate mismatch");

    // A high register: the kernel's register count is the contiguous
    // allocation up to it, and it survives the hex text.
    const auto high = compile("mov r128, 7\nhalt");
    require(high.register_count == 129, "qtxas did not derive the contiguous RF allocation");
    require(nanosm::decode(high.words.front()).dst == 128, "r128 did not survive assembly");
    require(nanosm::load_hex(nanosm::emit_hex(high)).register_count == 129, "hex lost the kernel register count");

    // The divergence forms parse, encode and print; an ssy's label may name
    // any instruction; bssy is not a mnemonic.
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
    expect_error([] { (void)compile("bssy J\nJ: halt\n"); }, {"unknown opcode 'bssy'"}, "bssy");
}

TEST(Basics, fnv1a_64_vectors) {
    require(nanosm::fnv1a_64("") == 0xcbf29ce484222325ull, "FNV-1a empty vector mismatch");
    require(nanosm::fnv1a_64("hello") == 0xa430d84680aabd0bull, "FNV-1a hello vector mismatch");
}

TEST(Basics, mbar_word_encoding) {
    nanosm::MbarState state;
    state.phase = true;
    state.valid = true;
    state.expected = 0x1234;
    state.remaining = 0x2345;
    // The tx field is a signed 32-bit count (two's complement), bits 32-63:
    // 0x89abcdef is a negative count.
    state.tx_bytes = static_cast<int32_t>(0x89abcdefu);
    const auto word = nanosm::encode_mbar_state(state);
    require(
        word == 0x89abcdef468a48d3ull,
        "mbar word layout changed without updating its architectural encoding test");
    const auto decoded = nanosm::decode_mbar_state(word);
    require(
        decoded.phase == state.phase && decoded.valid == state.valid &&
            decoded.expected == state.expected && decoded.remaining == state.remaining &&
            decoded.tx_bytes == state.tx_bytes && decoded.tx_bytes < 0,
        "mbar word encoding did not round trip");
    state.tx_bytes = -128; // complete_tx ran 128 bytes ahead of its expect_tx
    require(nanosm::encode_mbar_state(state) >> 32 == 0xffffff80u &&
            nanosm::decode_mbar_state(nanosm::encode_mbar_state(state)).tx_bytes == -128,
        "a negative tx count did not round trip");
}

TEST(Basics, rf_capacity_determines_residency) {
    auto profile = canonical_profile();
    profile.rf_rows_per_bank = 1;
    const auto kernel = compile("mov r0, 1\nhalt", profile);
    expect_error<nanosm::SimError>([&] { nanosm::Machine machine(profile, kernel, launch_warps(2)); },
        {"RF allocation cannot fit"}, "two warps in one RF row per bank");
}

TEST(Basics, compiler_controls) {
    const auto& profile = canonical_profile();
    // src/timing.hpp: a fixed result is visible issue_to_visible after its
    // issue; its consumer samples it issue_register_stages after its own.
    const auto distance = [&](const nanosm::Instruction& producer) {
        return nanosm::issue_to_visible(producer, profile) - nanosm::issue_register_stages;
    };

    // The loads take sb0 and sb1, their consumer waits for both, the store
    // reuses a released depbar and the halt waits for it. A producer right
    // before its consumer stalls its full distance.
    const auto program = qtx_program(R"(
        mov  r0, 0
        lds  r1, [r0]
        lds  r2, [r0 + 64]
        fadd r4, r1, r2
        sts  [r0 + 128], r4
        halt
    )", profile);
    require(program.instructions[0].control.stall == distance(program.instructions[0]),
        "mov did not cover the lds right after it");
    require(program.instructions[1].control.set_depbar == 0, "first load did not allocate sb0");
    require(program.instructions[2].control.set_depbar == 1, "second load did not allocate sb1");
    require(program.instructions[3].control.wait_mask == 0b000011, "fadd did not wait for both loads");
    require(program.instructions[3].control.stall == distance(program.instructions[3]),
        "fadd did not cover its one-away consumer");
    require(program.instructions[4].control.set_depbar == 0, "store did not reuse a released depbar");
    require(program.instructions[5].control.wait_mask == 0b000001, "halt did not wait for store completion");

    // With all six depbars live, the instruction that needs a seventh waits
    // for the oldest live one and reuses it.
    const auto seven = qtx_program(R"(
        mov r0, 0
        lds r1, [r0]
        lds r2, [r0+4]
        lds r3, [r0+8]
        lds r4, [r0+12]
        lds r5, [r0+16]
        lds r6, [r0+20]
        lds r7, [r0+24]
        halt
    )", profile);
    for (unsigned load = 0; load < 6; ++load) {
        require(seven.instructions[1 + load].control.set_depbar == load, "the first six loads take sb0..sb5");
        require(seven.instructions[1 + load].control.wait_mask == 0, "the first six loads wait for nothing");
    }
    require(seven.instructions[7].control.wait_mask == 0b000001, "the seventh load did not wait for the oldest depbar");
    require(seven.instructions[7].control.set_depbar == 0, "the seventh load did not reuse the oldest depbar");
    require(seven.instructions[8].control.wait_mask == 0b111111, "halt did not wait for every live depbar");

    // A result consumed three instructions later: only what its distance
    // exceeds those issues is a stall, on the instruction just before the
    // consumer. The minimal schedule runs clean under the hazard sanitizer.
    const auto spaced = qtx_program(R"(
        mov  r0, 1
        fadd r1, r0, r0
        mov  r2, 5
        mov  r3, 6
        add  r4, r1, r2
        halt
    )", profile);
    require(spaced.instructions[0].control.stall == distance(spaced.instructions[0]),
        "mov did not cover the fadd right after it");
    require(spaced.instructions[1].control.stall == 0, "fadd stalled its own latency");
    require(spaced.instructions[2].control.stall == 0, "independent mov stalled");
    require(spaced.instructions[3].control.stall == distance(spaced.instructions[1]) - 2, "pre-consumer stall wrong");
    auto launch = launch_warps(1);
    image_bytes(launch.test_shmem, 4);
    nanosm::Machine machine(profile, nanosm::assemble_asm(nanosm::emit_asm(spaced), profile), std::move(launch));
    machine.run();
}

// A loop head (src/qtxc/schedule.cpp LoopEntry): what the code before a loop
// leaves pending is paid once, by the instruction before the head, not by a
// stall inside the loop on every iteration. The body reads the ex2's result
// on the first pass, so the sanitizer checks the fall-through edge.
TEST(Basics, loop_head_pays_the_entry_once) {
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
    const auto hand = hand_assemble(hand_asm(profile, R"(
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
        nanosm::Machine machine(profile, kernel, launch_warps(1));
        machine.run();
        require(machine.read_reg(0, 0, 3) == 0x40000000u, "the loop did not compute ex2(0) + ex2(0) = 2");
    }
}

TEST(Basics, counted_loop_and_label_relocation) {
    const auto program = qtx_program(R"(
        mov r0, 0
        mov r1, 4
    loop:
        addi r0, r0, 1
        setp.lt.u32 p0, r0, r1
        @p0 bra.uni loop
        halt
    )", canonical_profile());
    require(program.instructions[4].imm == -24, "backedge relocation is not relative to the next PC");
    require(program.instructions[4].control.predicate == 0, "branch lost its qtx predicate");
    const auto assembly = nanosm::emit_asm(program);
    require(assembly.find("loop:") != std::string::npos, "scheduled assembly lost its label");

    nanosm::Machine machine(canonical_profile(), nanosm::assemble_asm(assembly, canonical_profile()), launch_warps(1));
    machine.run();
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(machine.read_reg(0, lane, 0) == 4, "counted loop produced the wrong trip count");
    }
}

// bra.uni reads its predicate from lane 0 (src/machine/branch.cpp); one that
// differs across lanes is the sanitizer's PANIC. With the sanitizer compiled
// out, every lane follows lane 0, which takes the branch here.
TEST(Basics, nonuniform_bra_uni_panics) {
    nanosm::Machine machine(canonical_profile(), compile(
        "sread r0, LANE_ID\nsetp.lt.u32 p0, r0, 8\n@p0 bra.uni taken\nmov r5, 1\nhalt\ntaken:\nmov r5, 2\nhalt\n"),
        launch_warps(1));
#if NANOSM_SANITIZER
    expect_panic(machine, {"bra.uni predicate is not warp-uniform"}, "a non-uniform bra.uni predicate");
#else
    machine.run();
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(machine.read_reg(0, lane, 5) == 2, "bra.uni (lane 0 takes it): lane " + std::to_string(lane));
    }
#endif
}

TEST(Basics, branch_boundary_drains_async_work) {
    const auto program = qtx_program(R"(
        mov r0, 0
        ldg r1, [r0]
        bra done
    done:
        halt
    )", canonical_profile());
    require(program.instructions[2].control.wait_mask == 1, "branch did not drain its live load depbar");

    const auto fixed = qtx_program(R"(
        mov r0, 1
        fadd r1, r0, r0
        bra done
    done:
        halt
    )", canonical_profile());
    // The fadd issued one cycle before the branch; the block after it must
    // issue no earlier than one cycle before the fadd's result is visible (its
    // first sample is after the issue register): issue_to_visible - 1 cycles
    // after the fadd, issue_to_visible - 2 after the branch (src/timing.hpp).
    require(fixed.instructions[2].control.stall ==
            nanosm::issue_to_visible(fixed.instructions[1], canonical_profile()) -
                nanosm::issue_register_stages - 1,
        "branch did not drain a live fadd writeback");
}

// qtx's wait is a nop that waits for every live depbar (no drain).
TEST(Basics, wait_on_matmul_operands) {
    const auto& profile = canonical_profile();
    // Here the matmul's operand-consumed depbar: the sts after the wait
    // overwrites the A operand the matmul was reading, ordered by the wait, so
    // the sanitizer stays quiet and the store lands.
    const std::string source = R"(
        umov ur0, 0
        umov ur1, 1024
        umov ur4, 8192
        umov ur2, 0
        mma.async.init.m64n64k8 [ur2], [ur0], [ur1]
        mma.commit [ur4]
        wait
        sread r20, LANE_ID
        shli r20, r20, 2
        mov r21, 7
        sts [r20], r21
        halt
    )";
    const auto program = qtx_program(source, profile);
    const auto& matmul = program.instructions.at(4);
    const auto& wait = program.instructions.at(6);
    require(wait.opcode == nanosm::Opcode::nop && wait.text == "nop", "wait was not emitted as a nop");
    require(matmul.control.set_depbar < 6 && wait.control.wait_mask == 1u << matmul.control.set_depbar,
        "wait does not wait for the matmul's depbar");
    require(nanosm::emit_asm(program).find("        nop {pred=PT, stall=0, set=-, wait=sb") != std::string::npos,
        "the scheduled asm does not show the nop");
    auto launch = launch_warps(1);
    auto& shmem = image_bytes(launch.test_shmem, 8192 + 8);
    write_mbar(shmem, 8192, mbar_state(1, 1));
    nanosm::Machine machine(profile, compile(source, profile), std::move(launch));
    machine.run(20000);
    require(machine.read_shmem_u32(0) == 7, "the sts after the wait did not land");

    // A live load depbar: the load's register is used right after the wait
    // with no second wait, and a fixed-latency result still in flight (the
    // fadd) costs the nop no drain stall, as it would cost a branch.
    const std::string load_source = R"(
        mov r0, 0
        ldg r1, [r0]
        fadd r2, r0, r0
        wait
        add r3, r1, r1
        halt
    )";
    const auto loads = qtx_program(load_source, profile);
    const auto& load = loads.instructions.at(1);
    const auto& load_wait = loads.instructions.at(3);
    require(load_wait.opcode == nanosm::Opcode::nop && load_wait.control.wait_mask == 1u << load.control.set_depbar,
        "wait does not wait for the load's depbar");
    require(loads.instructions.at(4).control.wait_mask == 0, "the use after wait waited again");
    require(load_wait.control.stall == 0, "wait drained the fadd like a branch");
    auto load_launch = launch_warps(1);
    write_u32(image_bytes(load_launch.gmem_image, 64), 0, 21);
    nanosm::Machine load_machine(profile, compile(load_source, profile), std::move(load_launch));
    load_machine.run(20000);
    for (uint32_t lane = 0; lane < profile.lanes; ++lane) {
        require(load_machine.read_reg(0, lane, 3) == 42, "lane " + std::to_string(lane) + ": wrong sum after the wait");
    }
}

// nop in hand assembly: it carries a wait and a stall like any instruction
// and nothing else. Its word round-trips; a load's register is read after
// the nop's wait for it; qtx's wait and assembly's nop are each refused on
// the other side.
TEST(Basics, nop_in_hand_assembly) {
    const auto& profile = canonical_profile();
    const auto kernel = nanosm::assemble_asm(hand_asm(profile, R"(
        mov r0, 0
        ldg r1, [r0] {pred=PT, stall=0, set=sb2, wait=-}
        nop {pred=PT, stall=3, set=-, wait=sb2}
        add r3, r1, r1
        halt
    )"), profile);
    const auto nop = nanosm::decode(kernel.words.at(2));
    require(nop.opcode == nanosm::Opcode::nop && static_cast<uint8_t>(nop.opcode) == 0xfe, "nop is not opcode 0xfe");
    require(nanosm::format_instruction(nop) == "nop {pred=PT, stall=3, set=-, wait=sb2}",
        "nop formats as '" + nanosm::format_instruction(nop) + "'");
    require(nanosm::encode(nop) == kernel.words.at(2), "nop encode / decode");
    require(nanosm::load_hex(nanosm::emit_hex(kernel)).words == kernel.words, "nop hex round trip");

    auto launch = launch_warps(1);
    write_u32(image_bytes(launch.gmem_image, 64), 0, 5);
    nanosm::Machine machine(profile, nanosm::schedule_fixed_stalls(kernel, profile), std::move(launch));
    machine.run(20000);
    require(machine.read_reg(0, 0, 3) == 10, "wrong sum after the nop's wait");

    expect_error([&] { (void)nanosm::assemble_asm(hand_asm(profile, "nop r1\nhalt"), profile); },
        {"expected 0 operands"}, "nop with an operand");
    expect_error(
        [&] { (void)nanosm::assemble_asm(hand_asm(profile, "nop {pred=PT, stall=0, set=sb0, wait=-}\nhalt"), profile); },
        {"cannot set a depbar"}, "nop setting a depbar");
    expect_error([&] { (void)nanosm::assemble_asm(hand_asm(profile, "wait\nhalt"), profile); }, {"nop {wait="},
        "wait in assembly");
    expect_error([&] { (void)qtx_program("nop\nhalt\n", profile); }, {"qtx has no nop"}, "nop in qtx");
}

TEST(Basics, mbar_try_wait_sleeps_and_engine_completion_wakes) {
    auto launch = launch_warps(1);
    image_bytes(launch.test_shmem, 128);
    nanosm::Machine machine(canonical_profile(), compile(R"(
        umov ur0, 64
        mbar.init [ur0], 1
        mbar.arrive.expect_tx [ur0], 8
    wait:
        mbar.try_wait p0, [ur0], 0
        @!p0 bra.uni wait
        mov r1, 9
        halt
    )"), std::move(launch), {.timeline = true});

    // A bound only: the warp parks once its init, arrive and probe have
    // crossed issue, dispatch and the RMW engine.
    while (machine.trace().summary().mbar_sleeps == 0 && machine.cycle() < 200) machine.tick();
    require(machine.trace().summary().mbar_sleeps == 1, "mbar.try_wait never parked the warp");
    machine.complete_mbar_tx(64, 8, 0);
    machine.run();

    const auto state = machine.read_mbar_state(64);
    require(state.valid, "mbar lost its valid bit");
    require(state.phase, "complete_tx did not flip the mbar phase");
    require(state.expected == 1 && state.remaining == 1, "mbar did not reload its arrival count");
    require(state.tx_bytes == 0, "complete_tx did not consume transaction bytes");
    require(machine.read_reg(0, 0, 1) == 9, "woken warp did not resume after mbar.try_wait");
    const auto& summary = machine.trace().summary();
    require(summary.mbar_rmw_requests == 3, "wrong number of mbar RMW requests");
    require(summary.mbar_phase_changes == 1, "phase transition was not traced");
    require(summary.mbar_wakes == 1, "phase broadcast did not wake the warp");
    require(summary.mbar_timeouts == 0, "exact wake incorrectly timed out");
    const auto services = events_named(machine, "mbar RMW");
    require(services.size() == 3, "trace did not contain every mbar RMW service interval");
    for (const auto& event : services) {
        // Read grant to write grant, D + Lb + C + 1 cycles.
        require(event.duration_cycles == rmw_busy_cycles(), "mbar RMW did not hold A for D + Lb + C + 1 cycles");
    }
}

// A probe of the mbar at 64 (one arrival expected and pending, in phase
// `phase`): a test_wait or try_wait for phase 0 when phase 0 has completed
// answers true at once, in every lane; a try_wait on a phase that never
// completes times out false; two warps time out in their own wait slots;
// with one wait slot the second warp's try_wait is a PANIC (slot w is warp
// w's).
TEST(Basics, mbar_try_wait_immediate_success) {
    struct Row {
        std::string name;
        bool phase;
        uint32_t warps;
        uint32_t timeout;    // 0: the canonical profile's
        std::string probe;
        bool answer;
        int64_t probes, immediate, sleeps, timeouts, wakes; // the trace summary's counts; -1: not checked
    };
    const std::vector<Row> rows = {
        {"test_wait on a completed phase", true, 1, 0, "mbar.test_wait p0, [ur0], 0", true, -1, -1, -1, -1, -1},
        {"try_wait on a completed phase", true, 1, 0, "umov ur2, 0\nmbar.try_wait p0, [ur0], ur2", true, 1, 1, 0,
            -1, -1},
        {"try_wait timing out", false, 1, 3, "mbar.try_wait p0, [ur0], 0", false, -1, -1, -1, 1, 0},
        {"two warps, two wait slots", false, 2, 4, "mbar.try_wait p0, [ur0], 0", false, 2, -1, -1, 2, -1},
    };
    {
        auto profile = canonical_profile();
        profile.mbar_wait_slots = 1;
        auto launch = launch_warps(2);
        write_mbar(image_bytes(launch.test_shmem, 128), 64, mbar_state(1, 1, 0, false));
        nanosm::Machine machine(profile, compile("umov ur0, 64\nmbar.try_wait p0, [ur0], 0\nhalt\n", profile),
            std::move(launch));
        expect_panic(machine, {"mbar.try_wait by warp 1", "sync.mbar_wait_slots"}, "a try_wait past the last slot");
    }
    for (const auto& row : rows) {
        auto profile = canonical_profile();
        if (row.timeout != 0) profile.mbar_try_wait_timeout_cycles = row.timeout;
        auto launch = launch_warps(row.warps);
        write_mbar(image_bytes(launch.test_shmem, 128), 64, mbar_state(1, 1, 0, row.phase));
        nanosm::Machine machine(profile, compile("umov ur0, 64\n" + row.probe +
            "\n@p0 bra.uni complete\nmov r1, 1\nhalt\ncomplete:\nmov r1, 2\nhalt\n", profile), std::move(launch));
        machine.run();
        for (uint32_t warp = 0; warp < row.warps; ++warp) {
            for (uint32_t lane = 0; lane < 16; ++lane) {
                require(machine.read_reg(warp, lane, 1) == (row.answer ? 2u : 1u), row.name + ": warp " +
                    std::to_string(warp) + " lane " + std::to_string(lane) + " did not see " +
                    (row.answer ? "true" : "false"));
            }
        }
        const auto& summary = machine.trace().summary();
        const auto pin = [&](int64_t want, uint64_t got, const std::string& what) {
            require(want < 0 || got == static_cast<uint64_t>(want),
                row.name + ": " + what + " " + std::to_string(got) + ", want " + std::to_string(want));
        };
        pin(row.probes, summary.mbar_initial_probes, "initial probes");
        pin(row.immediate, summary.mbar_immediate_successes, "immediate successes");
        pin(row.sleeps, summary.mbar_sleeps, "sleeps");
        pin(row.timeouts, summary.mbar_timeouts, "timeouts");
        pin(row.wakes, summary.mbar_wakes, "exact wakes");
    }
}

// A phase flip that lands in the cycle of a try_wait's deadline wins: the
// probe answers true, an exact wake, not a timeout. The wake comes when the
// RMW's write reaches the RAM, 2D + Lb + C cycles after the event's claim,
// which is the cycle after the park; the timeout puts the deadline there.
TEST(Basics, mbar_phase_flip_wins_on_try_wait_deadline) {
    auto profile = canonical_profile();
    profile.mbar_try_wait_timeout_cycles = 2 * profile.shmem_port_a_depth + profile.shmem_read_latency_cycles +
        profile.shmem_rmw_compute_cycles + 1;
    auto launch = launch_warps(1);
    write_mbar(image_bytes(launch.test_shmem, 128), 64, mbar_state(1, 0, 8));
    nanosm::Machine machine(profile, compile(R"(
        umov ur0, 64
        mbar.try_wait p0, [ur0], 0
        @p0 bra.uni complete
        mov r1, 1
        halt
    complete:
        mov r1, 2
        halt
    )", profile), std::move(launch));

    while (machine.trace().summary().mbar_sleeps == 0 && machine.cycle() < 32) machine.tick();
    require(machine.trace().summary().mbar_sleeps == 1, "deadline test never parked");
    machine.complete_mbar_tx(64, 8, 0);
    machine.run();
    require(machine.read_reg(0, 0, 1) == 2, "phase flip on deadline did not return true");
    require(machine.trace().summary().mbar_wakes == 1, "deadline phase flip was not an exact wake");
    require(machine.trace().summary().mbar_timeouts == 0, "deadline phase flip lost to timeout");
}

TEST(Basics, mbar_arrival_underflow_panics) {
    auto launch = launch_warps(1);
    image_bytes(launch.test_shmem, 128);
    nanosm::Machine machine(canonical_profile(), compile(R"(
        umov ur0, 64
        mbar.init [ur0], 1
        mbar.arrive.expect_tx [ur0], 8
        mbar.arrive [ur0]
        halt
    )"), std::move(launch));
    expect_panic(machine, {"arrival count underflow"}, "mbar arrival underflow");
}

TEST(Basics, launch_arguments_reach_the_kernel) {
    const auto& profile = canonical_profile();
    const auto kernel = compile(argument_kernel);
    // Declaration order, each aligned to its own size: OUT 0, N 8, X 12, D 64,
    // BAR 128, K 136, T 144; 152 bytes, so the block starts at the top minus
    // 152, rounded down to 64.
    const uint32_t start = (profile.shmem_bytes - 152) / 64 * 64;
    const std::vector<uint32_t> offsets{0, 8, 12, 64, 128, 136, 144};
    require(kernel.warps == 1 && kernel.sections.size() == offsets.size(), "the kernel lost a section");
    for (std::size_t index = 0; index < offsets.size(); ++index) {
        require(kernel.sections[index].address == start + offsets[index],
            "section " + kernel.sections[index].name + " is not where the block rule puts it");
    }
    // Four launch flags (core.launch_flags): one each for OUT, N and X, the
    // last for everything after them.
    require(kernel.flags.size() == 4 && kernel.flags[0].bar == 5 && kernel.flags[0].bytes == 8 &&
            kernel.flags[1].bar == 4 && kernel.flags[1].bytes == 12 && kernel.flags[2].bar == 3 &&
            kernel.flags[2].bytes == 16 && kernel.flags[3].bar == 2 && kernel.flags[3].bytes == 152,
        "the launch flags are not the first policy's");
    require(nanosm::emit_hex(nanosm::load_hex(nanosm::emit_hex(kernel))) == nanosm::emit_hex(kernel) &&
            nanosm::load_hex(nanosm::emit_hex(kernel)).sections.size() == 7 &&
            nanosm::load_hex(nanosm::emit_hex(kernel)).flags.size() == 4,
        "the hex text lost what the kernel declares");

    nanosm::LaunchImage input{kernel, argument_launch()};
    const auto serialized = nanosm::emit_launch_image(input);
    auto loaded = nanosm::load_launch_image(serialized);
    require(nanosm::emit_launch_image(loaded) == serialized, "a launch image with arguments does not round-trip");
    nanosm::Machine machine(profile, std::move(loaded.kernel), std::move(loaded.launch));
    // Nothing of the block is in shared memory at the first cycle: it is in
    // global memory (by default at its top), and the copy has to bring it.
    require(machine.read_shmem_u32(start + 8) == 0 && machine.read_shmem_u32(start + 136) == 0,
        "launch bytes were in shared memory before the first cycle");
    const uint32_t global = (static_cast<uint32_t>(profile.gmem_bytes) - 152) / profile.gmem_sector_bytes *
        profile.gmem_sector_bytes;
    require(machine.read_gmem_u32(global + 8) == 41, "the host did not write the block into global memory");
    machine.run();
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(machine.read_gmem_u32(4096 + 4 * lane) == 41, "the u32 argument did not reach the kernel");
        require(machine.read_gmem_u32(4160 + 4 * lane) == nanosm::float_bits(2.5f),
            "the f32 argument did not reach the kernel");
        require(machine.read_gmem_u32(4224 + 4 * lane) == nanosm::float_bits(1.5f), "the kernel's .data did not reach it");
        require(machine.read_gmem_u32(4288 + 4 * lane) == 7, "a .data word named by an .equ is wrong");
    }
    require(machine.read_shmem_u32(start + 140) == 0xff800000u && machine.read_shmem_u32(start + 148) == 9,
        ".data words are not where the block puts them");
    const auto barrier = nanosm::decode_mbar_state(machine.read_shmem_u32(start + 128) |
        (static_cast<uint64_t>(machine.read_shmem_u32(start + 132)) << 32));
    require(barrier.valid && !barrier.phase && barrier.expected == 3 && barrier.remaining == 3,
        "the kernel's .mbar did not start in phase 0 expecting 3");
    // The copy pays global memory's latency, and the flags clear in order as
    // the block's sectors land (152 bytes: five sectors).
    const auto& summary = machine.trace().summary();
    const uint64_t latency = uint64_t{profile.gmem_latency_dram_ticks} * profile.core_clock_hz / profile.dram_clock_hz;
    require(summary.launch_block_bytes == 152 && summary.launch_flags.size() == 4, "the launch copy was not traced");
    require(summary.launch_flags[0].cycle > latency, "the first launch flag cleared before global memory could answer");
    require(summary.launch_flags[0].cycle == summary.launch_flags[2].cycle &&
            summary.launch_flags[3].cycle > summary.launch_flags[0].cycle,
        "the flags did not clear with their sectors: three in the first, the last with the fifth");
    require(machine.cycle() > summary.launch_flags[3].cycle, "the kernel ended before its launch block landed");

    // There is no untimed way for the block to reach shared memory: a profile
    // must reserve at least one launch flag.
    auto untimed = profile;
    untimed.core_launch_flags = 0;
    expect_error<nanosm::ProfileError>([&] { nanosm::validate_device_profile(untimed); },
        {"launch_flags must be in 1..6"}, "a profile with no launch flags");

    // The declared order is the delivery order, and it survives scheduled
    // assembly, the hex text and a saved image (so a replay has the same
    // timing): sections and flags come back the same.
    const auto same_block = [&](const nanosm::Kernel& other, const std::string& what) {
        require(other.sections.size() == kernel.sections.size() && other.flags.size() == kernel.flags.size(),
            what + " lost sections or flags");
        for (std::size_t index = 0; index < kernel.sections.size(); ++index) {
            require(other.sections[index].name == kernel.sections[index].name &&
                    other.sections[index].address == kernel.sections[index].address,
                what + " changed the order of the launch block at " + kernel.sections[index].name);
        }
        for (std::size_t index = 0; index < kernel.flags.size(); ++index) {
            require(other.flags[index].bar == kernel.flags[index].bar &&
                    other.flags[index].bytes == kernel.flags[index].bytes, what + " changed a launch flag");
        }
    };
    same_block(nanosm::assemble_asm(nanosm::emit_asm(qtx_program(argument_kernel, profile)), profile),
        "scheduled assembly");
    same_block(nanosm::load_hex(nanosm::emit_hex(kernel)), "the hex text");
    same_block(nanosm::load_launch_image(serialized).kernel, "a saved image");
}

TEST(Basics, launch_refusals) {
    const auto refused = [](const std::function<void()>& attempt, const std::string& needle) {
        expect_error(attempt, {needle}, "a refusal with '" + needle + "'");
    };
    const auto launch_with = [](const char* source, uint32_t warps, std::vector<nanosm::LaunchArgument> args) {
        return [=] {
            auto launch = launch_warps(warps);
            launch.args = args;
            nanosm::Machine machine(canonical_profile(), compile(source), std::move(launch));
        };
    };
    const char* const two = ".param ptr P\n.param u32 N\nhalt\n";
    refused(launch_with(two, 1, {{"N", uint64_t{2}}}), "'P' (ptr) is missing");
    refused(launch_with(two, 1, {{"P", uint64_t{0}}, {"N", uint64_t{2}}, {"Z", uint64_t{3}}}),
        "'Z' names no parameter of the kernel (its signature: ptr P, u32 N)");
    refused(launch_with(two, 1, {{"P", 1.5}, {"N", uint64_t{2}}}), "'P' (ptr) must be an unsigned integer");
    refused(launch_with(two, 1, {{"P", uint64_t{1} << 32}, {"N", uint64_t{2}}}), "does not fit 32 bits");
    refused(launch_with(two, 1, {{"P", uint64_t{0}}, {"N", uint64_t{1} << 32}}), "'N' (u32) must be");
    refused(launch_with(".param desc D\nhalt\n", 1, {{"D", std::vector<uint8_t>(8, 0)}}), "64 bytes");
    refused(launch_with(".warps 2\nhalt\n", 1, {}), "the kernel is written for 2");
    refused(launch_with("halt\n", 0, {}), "gives no warp count");
    // The launch block's global address: inside global memory, and not over
    // the launch's own global bytes.
    refused([] {
        auto launch = launch_warps(1);
        launch.args = {{"P", uint64_t{0}}, {"N", uint64_t{2}}};
        launch.block_address = 64;
        launch.gmem_image.push_back({64, {1, 2, 3, 4}});
        nanosm::Machine machine(canonical_profile(), compile(".param ptr P\n.param u32 N\nhalt\n"), std::move(launch));
    }, "overlaps the launch's own global memory");
    // What the kernel declares: named sections, each name once.
    refused([] { compile(".mbar 4, expect 1\nhalt\n"); }, "bad mbar name '4'");
    refused([] { compile(".equ B, 64\n.mbar B, expect 1\nhalt\n"); }, "duplicate constant 'B'");
    refused([] { compile(".param ptr A\n.param u32 A\nhalt\n"); }, "duplicate symbol 'A'");
    refused([] { compile(".param u64 A\nhalt\n"); }, "unknown parameter type 'u64'");
    refused([] { compile(".param ptr A\n.flag sb5, 8\nhalt\n"); }, "qtx has no .flag line");
    refused([] { compile(".warps 17\nhalt\n"); }, "warp slots");
    // Hand assembly states its own launch flags, and they must cover the block.
    refused([] { assemble(".param ptr A\nhalt\n"); }, "do not cover its launch block");
    refused([] { assemble(".param ptr A\n.param ptr B\n.flag sb5, 8\nhalt\n"); }, "do not cover its launch block");
    refused([] { assemble(".param ptr A\n.flag sb5, 8\n.flag sb5, 8\nhalt\n"); }, "given twice");
    // A profile with no TMA cannot make the copy.
    refused([] {
        auto profile = canonical_profile();
        profile.tma_queue_commands = 0;
        profile.tma_sector_credits = 0;
        nanosm::validate_device_profile(profile);
        nanosm::compile_and_assemble(".param ptr A\nhalt\n", profile);
    }, "no TMA to copy");

    // The launch formats. A launch declares the mbars a test initialized in
    // its shmem image (Launch::mbars): the "mbars" field, absent when there is
    // none; the machine refuses one that is not an 8-byte aligned shmem slot.
    nanosm::LaunchImage input;
    input.kernel = compile("halt");
    input.launch = launch_warps(1);
    input.launch.test_shmem.push_back({4160, std::vector<uint8_t>(8, 0)});
    require(nanosm::emit_launch_image(input).find("mbars") == std::string::npos,
        "a launch with no declared mbar emitted an mbars field");
    input.launch.mbars = {4160, 4168};
    const auto declared = nanosm::emit_launch_image(input);
    const auto state_text = nanosm::emit_launch_state(input.launch);
    require(nanosm::load_launch_image(declared).launch.mbars == input.launch.mbars &&
            nanosm::load_launch_state(state_text).mbars == input.launch.mbars,
        "the launch formats lost the declared mbars");
    for (const uint32_t bad : {4161u, 65536u}) {
        refused([&] {
            auto launch = input.launch;
            launch.mbars = {bad};
            nanosm::Machine machine(canonical_profile(), input.kernel, std::move(launch));
        }, "launch declares an mbar at " + std::to_string(bad));
    }
    // An integer that is not what its field holds is refused before it is
    // narrowed: 4294967296 is not address 0, 4160.9 not 4160, -8 not
    // 0xfffffff8.
    const auto reload = [](bool image, const std::string& text) {
        return [=] {
            if (image) {
                (void)nanosm::emit_launch_image(nanosm::load_launch_image(text));
            } else {
                (void)nanosm::emit_launch_state(nanosm::load_launch_state(text));
            }
        };
    };
    for (const auto& bad : {nlohmann::json(4294967296ull), nlohmann::json(4160.9), nlohmann::json(-8),
             nlohmann::json("64")}) {
        for (const bool image : {false, true}) {
            auto document = nlohmann::json::parse(image ? declared : state_text);
            document["mbars"] = {4160, bad};
            refused(reload(image, document.dump()), "launch: mbars[1] is not a shmem address");
        }
    }
    for (const auto& bad : {nlohmann::json(4294967297ull), nlohmann::json(1.5), nlohmann::json(-1)}) {
        for (const bool image : {false, true}) {
            auto document = nlohmann::json::parse(image ? declared : state_text);
            document["active_warps"] = bad;
            refused(reload(image, document.dump()), std::string(image ? "launch image" : "launch state") +
                ": active warp count is not an unsigned 32-bit integer");
        }
    }
    for (const auto& bad : {nlohmann::json(1.5), nlohmann::json(-1)}) {
        auto document = nlohmann::json::parse(declared);
        document["kernel"]["register_count"] = bad;
        refused(reload(true, document.dump()), "launch image: kernel.register_count is not an unsigned 32-bit integer");
        document = nlohmann::json::parse(declared);
        document["test_shmem"][0]["address"] = bad;
        refused(reload(true, document.dump()), "launch image: test_shmem[0].address is not an unsigned integer");
    }
    // A segment's bytes are hex.
    nanosm::LaunchImage segment;
    segment.kernel = compile("halt");
    segment.launch = launch_warps(1);
    segment.launch.test_shmem.push_back({0, {0xab}});
    auto serialized = nanosm::emit_launch_image(segment);
    const auto byte = serialized.find("ab", serialized.find("\"bytes\": \"ab\""));
    require(byte != std::string::npos, "launch test could not find its byte payload");
    serialized[byte] = 'z';
    refused([&] { (void)nanosm::load_launch_image(serialized); }, "non-hex");
}

// umov urD, imm takes any 32-bit word (dst[6] imm[32]): a float literal is its
// IEEE bits, with no memory behind the constant; it survives the encoding,
// scheduled assembly and the hex text.
TEST(Basics, umov_takes_a_32_bit_word) {
    const auto& profile = canonical_profile();
    const char* const source =
        "umov ur1, 0.180336878\numov ur2, -inf\numov ur3, 0xdeadbeef\numov ur4, -1\numov ur5, 4294967295\n"
        "umov ur31, 1e-3\numov ur6, 70000\numov ur7, 2147483648\nur2r r1, ur1\nhalt\n";
    const std::vector<std::pair<uint32_t, uint32_t>> expected{
        {1, nanosm::float_bits(0.180336878f)}, {2, 0xff800000u}, {3, 0xdeadbeefu}, {4, 0xffffffffu},
        {5, 0xffffffffu}, {31, nanosm::float_bits(1e-3f)}, {6, 70000u}, {7, 0x80000000u}};
    const auto check = [&](const nanosm::Kernel& kernel, const std::string& what) {
        nanosm::Machine machine(profile, kernel, launch_warps(1));
        machine.run();
        for (const auto& [reg, value] : expected) {
            require(machine.read_ureg(0, reg) == value, what + ": umov did not write its 32-bit word to ur" +
                std::to_string(reg));
        }
        require(machine.read_reg(0, 9, 1) == nanosm::float_bits(0.180336878f), what + ": ur2r lost the constant");
    };
    const auto kernel = compile(source);
    check(kernel, "qtx");
    const auto scheduled = nanosm::emit_asm(qtx_program(source, profile));
    check(nanosm::assemble_asm(scheduled, profile), "scheduled assembly");
    check(nanosm::load_hex(nanosm::emit_hex(kernel)), "the hex text");
    // A large word prints as its 32 bits in hex, a small one as a number.
    nanosm::Instruction inst;
    inst.opcode = nanosm::Opcode::umovi;
    inst.dst = 63;
    inst.imm = static_cast<int32_t>(0xff800000u);
    const auto decoded = nanosm::decode(nanosm::encode(inst));
    require(decoded.dst == 63 && static_cast<uint32_t>(decoded.imm) == 0xff800000u,
        "umov's dst[6] imm[32] payload does not round-trip");
    require(nanosm::format_instruction(decoded, false) == "umov ur63, 0xff800000",
        "a large umov word is not printed as hex: " + nanosm::format_instruction(decoded, false));
    inst.imm = -4;
    require(nanosm::format_instruction(nanosm::decode(nanosm::encode(inst)), false) == "umov ur63, -4",
        "a small umov word is not printed as a number");
    for (const char* bad : {"umov ur1, 4294967296\nhalt\n", "umov ur1, -2147483649\nhalt\n", "umov ur1, 1.5f\nhalt\n",
                            "umov ur1, 0x1p3\nhalt\n", "uaddi ur1, ur1, 1.5\nhalt\n", "mov r1, 4194304\nhalt\n"}) {
        expect_error([&] { (void)compile(bad); }, {"immediate"}, std::string("a bad immediate: ") + bad);
    }
}

// A test's untimed preload of shared memory (Launch::test_shmem) is reported
// in the run's summary and may not stand in for the launch block.
TEST(Basics, test_preload_is_reported_and_kept_off_the_block) {
    const auto& profile = canonical_profile();
    const auto preloaded_bytes = [&](bool preload) {
        auto launch = launch_warps(1);
        if (preload) launch.test_shmem.push_back({64, std::vector<uint8_t>(12, 0xa5)});
        nanosm::Machine machine(profile, compile("halt\n"), std::move(launch));
        machine.run();
        return machine.trace().summary().test_preload_bytes;
    };
    require(preloaded_bytes(true) == 12, "a test's preload is not in the run summary");
    require(preloaded_bytes(false) == 0, "a launch with no preload reports one");
    const auto kernel = compile(".param u32 A\nhalt\n");
    expect_error([&] {
        auto launch = launch_warps(1);
        launch.args = {{"A", uint64_t{7}}};
        launch.test_shmem.push_back({kernel.sections[0].address, {7, 0, 0, 0}});
        nanosm::Machine machine(profile, kernel, std::move(launch));
    }, {"touches the kernel's launch block"}, "a test preload over the launch block");
}

// A hand-written kernel that reads its parameter, or probes a barrier of the
// block, without waiting on the launch flag reads bytes that are still on
// their way: the sanitizer names the launch copy. Waiting on the flag first
// is what makes it legal.
TEST(Basics, early_launch_read_panics) {
    const auto probe = [](const std::string& wait) {
        return assemble(".mbar B, expect 1\n.flag sb5, 8\numov ur1, B " + wait +
            "\nmbar.test_wait p0, [ur1], 1\nhalt\n");
    };
    nanosm::Machine waiting(canonical_profile(), probe("{wait=sb5}"), launch_warps(1));
    waiting.run();
#if NANOSM_SANITIZER
    auto launch = launch_warps(1);
    launch.args = {{"OUT", uint64_t{4096}}};
    nanosm::Machine machine(canonical_profile(), assemble(R"(
        .param ptr OUT
        .flag sb5, 8
        umov ur1, OUT
        lds.uni r1, [ur1] {set=sb0}
        halt {wait=sb0}
    )"), std::move(launch));
    expect_panic(machine, {"the launch copy"}, "an early launch read");
    nanosm::Machine probing(canonical_profile(), probe(""), launch_warps(1));
    expect_panic(probing, {"the launch copy"}, "an early barrier probe");
#endif
}

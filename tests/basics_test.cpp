// Encoding, the compiler's controls, branches, mbar basics, launch images.
// Each test's TEST wrapper is at the end.
#include "test_support.hpp"

namespace {

using namespace nanosm_test;

void test_encoding_round_trip() {
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
}

void test_fnv1a_64_vectors() {
    require(nanosm::fnv1a_64("") == 0xcbf29ce484222325ull, "FNV-1a empty vector mismatch");
    require(nanosm::fnv1a_64("hello") == 0xa430d84680aabd0bull, "FNV-1a hello vector mismatch");
}

void test_mbar_word_encoding() {
    nanosm::MbarState state;
    state.phase = true;
    state.valid = true;
    state.expected = 0x1234;
    state.remaining = 0x2345;
    // The tx field is a signed 32-bit count (two's
    // complement), same bits 32-63: 0x89abcdef is a negative count.
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

void test_high_register_kernel_metadata() {
    const auto kernel = compile("mov r128, 7\nhalt");
    require(kernel.register_count == 129, "qtxas did not derive the contiguous RF allocation");
    require(nanosm::decode(kernel.words.front()).dst == 128, "r128 did not survive assembly");
    const auto loaded = nanosm::load_hex(nanosm::emit_hex(kernel));
    require(loaded.register_count == 129, "hex lost the kernel register count");
}

void test_rf_capacity_determines_residency() {
    auto profile = canonical_profile();
    profile.rf_rows_per_bank = 1;
    const auto kernel = compile("mov r0, 1\nhalt", profile);
    nanosm::Launch launch;
    launch.active_warps = 2;
    try {
        nanosm::Machine machine(profile, kernel, std::move(launch));
        throw std::runtime_error("launch ignored physical RF capacity");
    } catch (const nanosm::SimError& error) {
        require(
            std::string(error.what()).find("RF allocation cannot fit") != std::string::npos,
            "RF residency failure produced the wrong error");
    }
}

void test_compiler_controls() {
    const auto program = qtx_program(R"(
        mov  r0, 0
        lds  r1, [r0]
        lds  r2, [r0 + 64]
        fadd r4, r1, r2
        sts  [r0 + 128], r4
        halt
    )", canonical_profile());
    // src/timing.hpp: a fixed result is visible issue_to_visible after its
    // issue; its consumer samples issue_to_sample (1, the issue register)
    // after its own. The mov's result feeds the very next lds (7 - 1 = 6 at
    // canonical); the fadd's the sts after it (10 - 1 = 9).
    const auto& profile = canonical_profile();
    const auto distance = [&](const nanosm::Instruction& producer) {
        return nanosm::issue_to_visible(producer, profile) - nanosm::issue_register_stages;
    };
    require(program.instructions[0].control.stall == distance(program.instructions[0]),
        "mov did not cover the lds right after it");
    require(program.instructions[1].control.set_depbar == 0, "first load did not allocate sb0");
    require(program.instructions[2].control.set_depbar == 1, "second load did not allocate sb1");
    require(program.instructions[3].control.wait_mask == 0b000011, "fadd did not wait for both loads");
    require(program.instructions[3].control.stall == distance(program.instructions[3]),
        "fadd did not cover its one-away consumer");
    require(program.instructions[4].control.set_depbar == 0, "store did not reuse a released depbar");
    require(program.instructions[5].control.wait_mask == 0b000001, "halt did not wait for store completion");
}

// With all six depbars live, the instruction that needs a seventh waits for
// the oldest live one and reuses it.
void test_seventh_depbar_reuses_the_oldest() {
    const auto program = qtx_program(R"(
        mov r0, 0
        lds r1, [r0]
        lds r2, [r0+4]
        lds r3, [r0+8]
        lds r4, [r0+12]
        lds r5, [r0+16]
        lds r6, [r0+20]
        lds r7, [r0+24]
        halt
    )", canonical_profile());
    for (unsigned load = 0; load < 6; ++load) {
        require(program.instructions[1 + load].control.set_depbar == load, "the first six loads take sb0..sb5");
        require(program.instructions[1 + load].control.wait_mask == 0, "the first six loads wait for nothing");
    }
    require(program.instructions[7].control.wait_mask == 0b000001, "the seventh load did not wait for the oldest depbar");
    require(program.instructions[7].control.set_depbar == 0, "the seventh load did not reuse the oldest depbar");
    require(program.instructions[8].control.wait_mask == 0b111111, "halt did not wait for every live depbar");
}

void test_stalls_track_consumer_distance() {
    // fadd's result is consumed three instructions later, so only what its
    // distance (src/timing.hpp: issue_to_visible - the consumer's issue_to_sample)
    // exceeds those three issues is bought with a stall, and it lands just
    // before the consumer. The mov feeding the fadd stalls the full distance.
    const auto& profile = canonical_profile();
    const auto program = qtx_program(R"(
        mov  r0, 1
        fadd r1, r0, r0
        mov  r2, 5
        mov  r3, 6
        add  r4, r1, r2
        halt
    )", profile);
    const auto& fadd = program.instructions[1];
    const auto fadd_distance = nanosm::issue_to_visible(fadd, profile) - nanosm::issue_register_stages;
    require(program.instructions[0].control.stall ==
            nanosm::issue_to_visible(program.instructions[0], profile) - nanosm::issue_register_stages,
        "mov did not cover the fadd right after it");
    require(fadd.control.stall == 0, "fadd stalled its own latency");
    require(program.instructions[2].control.stall == 0, "independent mov stalled");
    require(program.instructions[3].control.stall == fadd_distance - 2, "pre-consumer stall wrong");
    nanosm::Launch launch;
    launch.active_warps = 1;
    image_bytes(launch.test_shmem, 4);
    nanosm::Machine machine(
        canonical_profile(),
        nanosm::assemble_asm(nanosm::emit_asm(program), canonical_profile()),
        std::move(launch));
    machine.run(); // the hazard sanitizer accepts the minimal schedule
}

void test_counted_loop_and_label_relocation() {
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

    nanosm::Launch launch;
    launch.active_warps = 1;
    nanosm::Machine machine(
        canonical_profile(),
        nanosm::assemble_asm(assembly, canonical_profile()),
        std::move(launch));
    machine.run();
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(machine.read_reg(0, lane, 0) == 4, "counted loop produced the wrong trip count");
    }
}

void test_uniform_role_dispatch() {
    nanosm::Launch launch;
    launch.active_warps = 2;
    nanosm::Machine machine(canonical_profile(), compile(R"(
        usread ur0, WARP_ID
        ur2r r0, ur0
        setp.eq.u32 p0, r0, 0
        @p0 bra.uni warp_zero
        mov r1, 11
        bra done
    warp_zero:
        mov r1, 22
    done:
        halt
    )"), std::move(launch));
    machine.run();
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(machine.read_reg(0, lane, 1) == 22, "warp-zero role branch was not taken");
        require(machine.read_reg(1, lane, 1) == 11, "nonzero warp did not take fallthrough");
    }
}

void test_nonuniform_bra_uni_panics() {
    nanosm::Launch launch;
    launch.active_warps = 1;
    nanosm::Machine machine(canonical_profile(), compile(R"(
        sread r0, LANE_ID
        setp.lt.u32 p0, r0, 8
        @p0 bra.uni split
        halt
    split:
        halt
    )"), std::move(launch));
    try {
        machine.run();
        throw std::runtime_error("nonuniform bra.uni executed without divergence support");
    } catch (const nanosm::SimError& error) {
        require(
            std::string(error.what()).find("bra.uni predicate is not warp-uniform") != std::string::npos,
            "nonuniform bra.uni produced the wrong PANIC");
    }
}

void test_branch_boundary_drains_async_work() {
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

// qtx's wait is a nop that waits
// for every live depbar, here the matmul's operand-consumed one. The sts
// after it overwrites the A operand the matmul was reading: ordered by the
// wait, so the sanitizer stays quiet and the store lands.
void test_wait_on_matmul_operands() {
    const auto& profile = canonical_profile();
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

    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.test_shmem, 8192 + 8);
    write_count_mbar(shmem, 8192, 1);
    nanosm::Machine machine(profile, compile(source, profile), std::move(launch));
    machine.run(20000);
    require(machine.read_shmem_u32(0) == 7, "the sts after the wait did not land");
}

// wait with a live load depbar: the load's register is used right after it
// with no second wait, and a fixed-latency result still in flight (the fadd)
// costs the nop no drain stall, as it would cost a branch.
void test_wait_on_load_then_use() {
    const auto& profile = canonical_profile();
    const std::string source = R"(
        mov r0, 0
        ldg r1, [r0]
        fadd r2, r0, r0
        wait
        add r3, r1, r1
        halt
    )";
    const auto program = qtx_program(source, profile);
    const auto& load = program.instructions.at(1);
    const auto& wait = program.instructions.at(3);
    const auto& use = program.instructions.at(4);
    require(wait.opcode == nanosm::Opcode::nop && wait.control.wait_mask == 1u << load.control.set_depbar,
        "wait does not wait for the load's depbar");
    require(use.control.wait_mask == 0, "the use after wait waited again");
    require(wait.control.stall == 0, "wait drained the fadd like a branch");

    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& gmem = image_bytes(launch.gmem_image, 64);
    write_u32(gmem, 0, 21);
    nanosm::Machine machine(profile, compile(source, profile), std::move(launch));
    machine.run(20000);
    for (uint32_t lane = 0; lane < profile.lanes; ++lane) {
        require(machine.read_reg(0, lane, 3) == 42, "lane " + std::to_string(lane) + ": wrong sum after the wait");
    }
}

// nop in hand assembly: it carries a wait and a stall like any instruction
// and nothing else. Its word round-trips; a load's register is read after
// the nop's wait for it; qtx's wait and assembly's nop are each refused on
// the other side.
void test_nop_in_hand_assembly() {
    const auto& profile = canonical_profile();
    const auto kernel = nanosm::assemble_asm(i1_asm(profile, R"(
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

    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& gmem = image_bytes(launch.gmem_image, 64);
    write_u32(gmem, 0, 5);
    nanosm::Machine machine(profile, nanosm::schedule_fixed_stalls(kernel, profile), std::move(launch));
    machine.run(20000);
    require(machine.read_reg(0, 0, 3) == 10, "wrong sum after the nop's wait");

    const auto refused = [](const std::function<void()>& build, const std::string& needle, const std::string& what) {
        try {
            build();
        } catch (const nanosm::SimError& error) {
            require(std::string(error.what()).find(needle) != std::string::npos,
                what + ": wrong error: " + error.what());
            return;
        }
        throw std::runtime_error(what + " was accepted");
    };
    refused([&] { nanosm::assemble_asm(i1_asm(profile, "nop r1\nhalt"), profile); }, "expected 0 operands",
        "nop with an operand");
    refused([&] { nanosm::assemble_asm(i1_asm(profile, "nop {pred=PT, stall=0, set=sb0, wait=-}\nhalt"), profile); },
        "cannot set a depbar", "nop setting a depbar");
    refused([&] { nanosm::assemble_asm(i1_asm(profile, "wait\nhalt"), profile); }, "nop {wait=", "wait in assembly");
    refused([&] { qtx_program("nop\nhalt\n", profile); }, "qtx has no nop", "nop in qtx");
}

void test_mbar_try_wait_sleeps_and_engine_completion_wakes() {
    nanosm::Launch launch;
    launch.active_warps = 1;
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
    require(machine.trace().summary().mbar_rmw_requests == 3, "wrong number of mbar RMW requests");
    require(machine.trace().summary().mbar_phase_changes == 1, "phase transition was not traced");
    require(machine.trace().summary().mbar_wakes == 1, "phase broadcast did not wake the warp");
    require(machine.trace().summary().mbar_timeouts == 0, "exact wake incorrectly timed out");
    std::size_t rmw_slices = 0;
    for (const auto& event : machine.trace().events()) {
        if (event.name != "mbar RMW") continue;
        ++rmw_slices;
        // Read grant to write grant, D + Lb + C + 1 cycles.
        require(event.duration_cycles == rmw_busy_cycles(), "mbar RMW did not hold A for D + Lb + C + 1 cycles");
    }
    require(rmw_slices == 3, "trace did not contain every mbar RMW service interval");
}

void test_mbar_test_wait_broadcasts_predicate() {
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.test_shmem, 128);
    nanosm::MbarState initial;
    initial.valid = true;
    initial.phase = true;
    initial.expected = 1;
    initial.remaining = 1;
    write_u64(shmem, 64, nanosm::encode_mbar_state(initial));
    nanosm::Machine machine(canonical_profile(), compile(R"(
        umov ur0, 64
        mbar.test_wait p0, [ur0], 0
        @p0 bra.uni complete
        mov r1, 1
        halt
    complete:
        mov r1, 2
        halt
    )"), std::move(launch));
    machine.run();
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(machine.read_reg(0, lane, 1) == 2, "mbar.test_wait did not broadcast true");
    }
}

void test_mbar_try_wait_immediate_success() {
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.test_shmem, 128);
    nanosm::MbarState initial;
    initial.valid = true;
    initial.phase = true;
    initial.expected = 1;
    initial.remaining = 1;
    write_u64(shmem, 64, nanosm::encode_mbar_state(initial));
    nanosm::Machine machine(canonical_profile(), compile(R"(
        umov ur0, 64
        umov ur2, 0
        mbar.try_wait p0, [ur0], ur2
        @p0 bra.uni complete
        mov r1, 1
        halt
    complete:
        mov r1, 2
        halt
    )"), std::move(launch));
    machine.run();
    require(machine.read_reg(0, 0, 1) == 2, "immediate try_wait did not write true");
    const auto& summary = machine.trace().summary();
    require(summary.mbar_initial_probes == 1, "try_wait initial probe was not traced");
    require(summary.mbar_immediate_successes == 1, "immediate try_wait was not counted");
    require(summary.mbar_sleeps == 0, "complete try_wait incorrectly parked");
}

void test_mbar_try_wait_times_out_false() {
    auto profile = canonical_profile();
    profile.mbar_try_wait_timeout_cycles = 3;
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.test_shmem, 128);
    nanosm::MbarState initial;
    initial.valid = true;
    initial.expected = 1;
    initial.remaining = 1;
    write_u64(shmem, 64, nanosm::encode_mbar_state(initial));
    nanosm::Machine machine(profile, compile(R"(
        umov ur0, 64
        mbar.try_wait p0, [ur0], 0
        @p0 bra.uni unexpected
        mov r1, 7
        halt
    unexpected:
        mov r1, 9
        halt
    )", profile), std::move(launch));
    machine.run();
    require(machine.read_reg(0, 0, 1) == 7, "timed-out try_wait did not write false");
    require(machine.trace().summary().mbar_timeouts == 1, "try_wait timeout was not traced");
    require(machine.trace().summary().mbar_wakes == 0, "timeout was counted as exact wake");
}

// The wake comes when the RMW's write reaches the RAM,
// 2D + Lb (+ C) cycles after the event's claim, which is the cycle after
// the park. The timeout puts the deadline in that cycle.
void test_mbar_phase_flip_wins_on_try_wait_deadline() {
    auto profile = canonical_profile();
    // The wake lands exactly on the deadline: the claim is the cycle after the
    // park, the wake the RMW write's commit, D + Lb + C + D later (+ C,
    // the RMW's ALU stages).
    profile.mbar_try_wait_timeout_cycles = 2 * profile.shmem_port_a_depth + profile.shmem_read_latency_cycles +
        profile.shmem_rmw_compute_cycles + 1;
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.test_shmem, 128);
    nanosm::MbarState initial;
    initial.valid = true;
    initial.expected = 1;
    initial.remaining = 0;
    initial.tx_bytes = 8;
    write_u64(shmem, 64, nanosm::encode_mbar_state(initial));
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

void test_mbar_try_wait_slots_limit_parked_warps() {
    auto profile = canonical_profile();
    profile.mbar_wait_slots = 1;
    profile.mbar_try_wait_timeout_cycles = 4;
    nanosm::Launch launch;
    launch.active_warps = 2;
    auto& shmem = image_bytes(launch.test_shmem, 128);
    nanosm::MbarState initial;
    initial.valid = true;
    initial.expected = 1;
    initial.remaining = 1;
    write_u64(shmem, 64, nanosm::encode_mbar_state(initial));
    nanosm::Machine machine(profile, compile(R"(
        umov ur0, 64
        mbar.try_wait p0, [ur0], 0
        halt
    )", profile), std::move(launch));
    machine.run();
    const auto& summary = machine.trace().summary();
    require(summary.mbar_initial_probes == 2, "not every warp eventually obtained a wait slot");
    require(summary.mbar_timeouts == 2, "parked warps did not release their wait slots");
    require(summary.mbar_slot_full_stalls != 0, "wait-slot exhaustion was not attributed");
    require(
        summary.warp_cycles[static_cast<std::size_t>(nanosm::WarpState::mbar_wait_slot)] != 0,
        "slot-full cycles were not a distinct warp state");
}

void test_staged_pipeline_kernel() {
    // The everything-so-far kernel: warp 0 stages gmem tiles into double-buffered shmem
    // through the RF, warp 1 consumes and stores back; full/empty barrier pairs, try_wait
    // retry loops with uxori parity toggling, counted loops, role dispatch.
    constexpr uint32_t tiles = 8;
    constexpr uint32_t elements = tiles * 16; // one lane-width tile per pass
    // The launch gives only the kernel's arguments (ptr A, ptr B, ptr C); its
    // two warps and its four barriers are the qtx's own (.warps, .mbar).
    nanosm::Launch launch;
    launch.args = {{"A", uint64_t{0}}, {"B", uint64_t{4096}}, {"C", uint64_t{8192}}};
    auto& gmem = image_bytes(launch.gmem_image, 8192 + elements * 4);
    for (uint32_t i = 0; i < elements; ++i) {
        write_u32(gmem, i * 4, nanosm::float_bits(static_cast<float>(i)));
        write_u32(gmem, 4096 + i * 4, nanosm::float_bits(static_cast<float>(2 * i)));
    }
    nanosm::LaunchImage input{
        nanosm::compile_and_assemble_file("kernels/pipeline.qtx", canonical_profile()),
        std::move(launch),
    };
    const auto serialized = nanosm::emit_launch_image(input);
    auto loaded = nanosm::load_launch_image(serialized);
    require(loaded.kernel.warps == 2 && loaded.launch.active_warps == 0,
        "launch image lost the kernel's warp count");
    // Three pointers and two pairs of barriers, each pair one array section.
    require(loaded.kernel.sections.size() == 5 && loaded.kernel.sections[3].count == 2 &&
            loaded.launch.args.size() == 3 && loaded.kernel.flags.size() == 4,
        "launch image lost the kernel's signature, its barriers, its launch flags or the arguments");
    require(loaded.launch.gmem_image.size() == 1, "launch image lost its gmem segment");
    require(
        nanosm::emit_launch_image(loaded) == serialized,
        "launch image serialization is not canonical");
    nanosm::Machine machine(
        canonical_profile(), std::move(loaded.kernel), std::move(loaded.launch));
    machine.run();
    for (uint32_t i = 0; i < elements; ++i) {
        require(
            nanosm::bits_float(machine.read_gmem_u32(8192 + i * 4)) == static_cast<float>(3 * i),
            "pipeline output mismatch at " + std::to_string(i));
    }
    const auto& summary = machine.trace().summary();
    require(summary.mbar_sleeps != 0, "pipeline never parked a warp");
    require(summary.mbar_wakes != 0, "pipeline never took an exact wake");
    // (The canonical timeout is long, a safety net: no wait here reaches it. The
    // timeout path has its own tests with short timeouts, above.)
    require(summary.gmem_commands == 3 * tiles, "wrong gmem command count for the pipeline");
    uint64_t warp_cycles = 0;
    for (const auto cycles : summary.warp_cycles) warp_cycles += cycles;
    require(warp_cycles == summary.cycles * 2, "pipeline warp attribution does not sum");
}

// The launch (README "Launch"): the kernel's `.param` lines are its signature,
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

// Hand assembly for the tests below (controls default; fixed stalls raised).
nanosm::Kernel assemble(const std::string& body) {
    return hand_assemble(i1_asm(canonical_profile(), body), canonical_profile());
}

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

void test_launch_arguments_reach_the_kernel() {
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

    const auto check = [&](nanosm::Machine& machine, const std::string& what) {
        for (uint32_t lane = 0; lane < 16; ++lane) {
            require(machine.read_gmem_u32(4096 + 4 * lane) == 41, what + ": the u32 argument did not reach the kernel");
            require(machine.read_gmem_u32(4160 + 4 * lane) == nanosm::float_bits(2.5f),
                what + ": the f32 argument did not reach the kernel");
            require(machine.read_gmem_u32(4224 + 4 * lane) == nanosm::float_bits(1.5f),
                what + ": the kernel's .data did not reach it");
            require(machine.read_gmem_u32(4288 + 4 * lane) == 7, what + ": a .data word named by an .equ is wrong");
        }
        require(machine.read_shmem_u32(start + 140) == 0xff800000u && machine.read_shmem_u32(start + 148) == 9,
            what + ": .data words are not where the block puts them");
        const auto barrier = nanosm::decode_mbar_state(machine.read_shmem_u32(start + 128) |
            (static_cast<uint64_t>(machine.read_shmem_u32(start + 132)) << 32));
        require(barrier.valid && !barrier.phase && barrier.expected == 3 && barrier.remaining == 3,
            what + ": the kernel's .mbar did not start in phase 0 expecting 3");
    };

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
    check(machine, "timed copy");
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
    try {
        nanosm::validate_device_profile(untimed);
        throw std::runtime_error("a profile with no launch flags was accepted");
    } catch (const nanosm::ProfileError& error) {
        require(std::string(error.what()).find("launch_flags must be in 1..6") != std::string::npos,
            std::string("no launch flags: wrong refusal: ") + error.what());
    }

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

// `.equ XX, X` is another name for section X: an instruction that names XX
// waits on X's launch flag like one that names X (with no wait it would read
// zeros when the sanitizer is off). A chain of such lines too.
void test_section_alias_waits_on_its_flag() {
    const auto run = [](const std::string& name) {
        nanosm::Launch launch;
        launch.active_warps = 1;
        launch.args = {{"X", uint64_t{0x12345678}}};
        nanosm::Machine machine(canonical_profile(), compile(
            ".param u32 X\n.equ XX, X\n.equ XXX, XX\n"
            "umov ur1, " + name + "\nlds.uni r1, [ur1]\nsread r2, LANE_ID\nshli r2, r2, 2\nsts [r2], r1\nhalt\n"),
            std::move(launch));
        machine.run();
        return machine.read_shmem_u32(0);
    };
    // Sanitizer on or off, the value is the argument.
    for (const std::string name : {"X", "XX", "XXX"}) {
        require(run(name) == 0x12345678u, "umov ur1, " + name + " read its parameter before it landed");
    }
    const auto scheduled = nanosm::emit_asm(qtx_program(
        ".param u32 X\n.equ XX, X\n.equ XXX, XX\numov ur1, XXX\nhalt\n", canonical_profile()));
    require(scheduled.find("wait=sb5") != std::string::npos, "an alias of a section got no launch-flag wait");
}

// `.mbar NAME[C]` is one section of C barriers, element i at NAME + 8 i and
// written NAME[i]; naming any of it waits for the whole array, and an index
// past its end is refused.
void test_mbar_array() {
    const auto& profile = canonical_profile();
    const char* const source = R"(
.warps 1
.param u32 A
.mbar PAIR[2], expect 3
.mbar ONE, expect 1
        umov ur1, PAIR
        umov ur2, PAIR[1]
        umov ur3, ONE
        mbar.arrive [ur2]
        halt
)";
    const auto kernel = compile(source);
    require(kernel.sections.size() == 3 && kernel.sections[1].count == 2 &&
            nanosm::section_bytes(kernel.sections[1]) == 16 &&
            kernel.sections[2].address == kernel.sections[1].address + 16,
        "an mbar array is not one section of 8 bytes per barrier");
    const auto text = nanosm::emit_hex(kernel);
    require(text.find(".mbar PAIR[2], ") != std::string::npos &&
            nanosm::load_hex(text).sections[1].count == 2, "the hex text lost the array");
    nanosm::LaunchImage image{kernel, {}};
    image.launch.args = {{"A", uint64_t{1}}};
    auto loaded = nanosm::load_launch_image(nanosm::emit_launch_image(image));
    require(loaded.kernel.sections[1].count == 2, "a saved image lost the array");
    nanosm::Machine machine(profile, std::move(loaded.kernel), std::move(loaded.launch));
    machine.run();
    const auto state = [&](uint32_t address) {
        return nanosm::decode_mbar_state(machine.read_shmem_u32(address) |
            (static_cast<uint64_t>(machine.read_shmem_u32(address + 4)) << 32));
    };
    const auto pair = kernel.sections[1].address;
    require(state(pair).valid && state(pair).expected == 3 && state(pair).remaining == 3,
        "element 0 of the array did not start expecting 3");
    require(state(pair + 8).valid && state(pair + 8).expected == 3 && state(pair + 8).remaining == 2,
        "element 1 of the array is not where NAME[1] says (the arrive went elsewhere)");
    for (const char* bad : {"umov ur1, PAIR[2]\nhalt\n", ".equ P, PAIR[7]\nhalt\n", "umov ur1, ONE[1]\nhalt\n"}) {
        try {
            compile(std::string(".mbar PAIR[2], expect 1\n.mbar ONE, expect 1\n") + bad);
            throw std::runtime_error(std::string("accepted an index past an mbar array's end: ") + bad);
        } catch (const nanosm::SimError& error) {
            require(std::string(error.what()).find("is past the end of .mbar") != std::string::npos,
                std::string("an index past an array's end: wrong refusal: ") + error.what());
        }
    }
    try {
        compile(".mbar PAIR[0], expect 1\nhalt\n");
        throw std::runtime_error("accepted an empty mbar array");
    } catch (const nanosm::SimError& error) {
        require(std::string(error.what()).find("an array has 1..") != std::string::npos,
            std::string("an empty mbar array: wrong refusal: ") + error.what());
    }
}

// umov urD, imm takes any 32-bit word (dst[6] imm[32]): a float literal is its
// IEEE bits, with no memory behind the constant; it survives the encoding,
// scheduled assembly and the hex text.
void test_umov_takes_a_32_bit_word() {
    const auto& profile = canonical_profile();
    const char* const source =
        "umov ur1, 0.180336878\numov ur2, -inf\numov ur3, 0xdeadbeef\numov ur4, -1\numov ur5, 4294967295\n"
        "umov ur31, 1e-3\numov ur6, 70000\numov ur7, 2147483648\nur2r r1, ur1\nhalt\n";
    const std::vector<std::pair<uint32_t, uint32_t>> expected{
        {1, nanosm::float_bits(0.180336878f)}, {2, 0xff800000u}, {3, 0xdeadbeefu}, {4, 0xffffffffu},
        {5, 0xffffffffu}, {31, nanosm::float_bits(1e-3f)}, {6, 70000u}, {7, 0x80000000u}};
    const auto check = [&](const nanosm::Kernel& kernel, const std::string& what) {
        nanosm::Launch launch;
        launch.active_warps = 1;
        nanosm::Machine machine(profile, kernel, std::move(launch));
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
    // A word outside the old 22-bit immediate prints as its 32 bits in hex; a
    // small one as a number.
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
        try {
            compile(bad);
            throw std::runtime_error(std::string("accepted: ") + bad);
        } catch (const nanosm::SimError& error) {
            require(std::string(error.what()).find("immediate") != std::string::npos,
                std::string("a bad immediate: wrong refusal: ") + error.what());
        }
    }
}

// A test's untimed preload of shared memory (Launch::test_shmem) is reported
// in the run's summary and may not stand in for the launch block.
void test_test_preload_is_reported_and_kept_off_the_block() {
    const auto& profile = canonical_profile();
    const auto preloaded_bytes = [&](bool preload) {
        nanosm::Launch launch;
        launch.active_warps = 1;
        if (preload) launch.test_shmem.push_back({64, std::vector<uint8_t>(12, 0xa5)});
        nanosm::Machine machine(profile, compile("halt\n"), std::move(launch));
        machine.run();
        return machine.trace().summary().test_preload_bytes;
    };
    require(preloaded_bytes(true) == 12, "a test's preload is not in the run summary");
    require(preloaded_bytes(false) == 0, "a launch with no preload reports one");
    const auto kernel = compile(".param u32 A\nhalt\n");
    try {
        nanosm::Launch launch;
        launch.active_warps = 1;
        launch.args = {{"A", uint64_t{7}}};
        launch.test_shmem.push_back({kernel.sections[0].address, {7, 0, 0, 0}});
        nanosm::Machine machine(profile, kernel, std::move(launch));
        throw std::runtime_error("a test preload over the launch block was accepted");
    } catch (const nanosm::SimError& error) {
        require(std::string(error.what()).find("touches the kernel's launch block") != std::string::npos,
            std::string("a preload over the launch block: wrong refusal: ") + error.what());
    }
}

// A hand-written kernel that reads its parameter without waiting on the
// launch flag reads bytes that are still on their way: the sanitizer names
// the launch copy.
void test_early_launch_read_panics() {
    const auto kernel = assemble(R"(
        .param ptr OUT
        .flag sb5, 8
        umov ur1, OUT
        lds.uni r1, [ur1] {set=sb0}
        halt {wait=sb0}
    )");
    nanosm::Launch launch;
    launch.active_warps = 1;
    launch.args = {{"OUT", uint64_t{4096}}};
    nanosm::Machine machine(canonical_profile(), kernel, std::move(launch));
    try {
        machine.run();
        throw std::runtime_error("a read of launch bytes that had not landed ran");
    } catch (const nanosm::SimError& error) {
        require(std::string(error.what()).find("the launch copy") != std::string::npos,
            std::string("an early launch read produced the wrong PANIC: ") + error.what());
    }
    // The same for a barrier of the block probed before it is there; waiting
    // on its launch flag first is what makes it legal.
    const auto probe = [](const char* wait) {
        nanosm::Launch one;
        one.active_warps = 1;
        nanosm::Machine probing(canonical_profile(), assemble(std::string(R"(
            .mbar B, expect 1
            .flag sb5, 8
            umov ur1, B )") + wait + R"(
            mbar.test_wait p0, [ur1], 1
            halt
        )"), std::move(one));
        probing.run();
    };
    try {
        probe("");
        throw std::runtime_error("a probe of a barrier that had not landed ran");
    } catch (const nanosm::SimError& error) {
        require(std::string(error.what()).find("the launch copy") != std::string::npos,
            std::string("an early barrier probe produced the wrong PANIC: ") + error.what());
    }
    probe("{wait=sb5}");
}

void test_launch_refusals() {
    const auto refused = [](const std::function<void()>& attempt, const std::string& needle) {
        try {
            attempt();
        } catch (const nanosm::SimError& error) {
            require(std::string(error.what()).find(needle) != std::string::npos,
                "wrong refusal (wanted '" + needle + "'): " + error.what());
            return;
        }
        throw std::runtime_error("accepted; wanted a refusal with '" + needle + "'");
    };
    const auto launch_with = [](const char* source, uint32_t warps, std::vector<nanosm::LaunchArgument> args) {
        return [=] {
            nanosm::Launch launch;
            launch.active_warps = warps;
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
        nanosm::Launch launch;
        launch.active_warps = 1;
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
}

void test_launch_image_rejects_bad_segment_hex() {
    nanosm::LaunchImage input;
    input.kernel = compile("halt");
    input.launch.active_warps = 1;
    input.launch.test_shmem.push_back({0, {0xab}});
    auto serialized = nanosm::emit_launch_image(input);
    const auto field = serialized.find("\"bytes\": \"ab\"");
    const auto byte = serialized.find("ab", field);
    require(byte != std::string::npos, "launch test could not find its byte payload");
    serialized[byte] = 'z';
    try {
        nanosm::load_launch_image(serialized);
        throw std::runtime_error("launch image accepted malformed segment hex");
    } catch (const nanosm::SimError& error) {
        require(
            std::string(error.what()).find("non-hex") != std::string::npos,
            "malformed segment produced the wrong launch-image error");
    }
}

void test_profile_can_disable_mbar_try_wait() {
    auto profile = canonical_profile();
    profile.mbar_wait_slots = 0;
    profile.mbar_try_wait_timeout_cycles = 0;
    nanosm::validate_device_profile(profile);
    try {
        qtx_program("umov ur0, 64\nmbar.try_wait p0, [ur0], 0\nhalt", profile);
        throw std::runtime_error("qtxc accepted try_wait with no wait slots");
    } catch (const nanosm::SimError& error) {
        require(
            std::string(error.what()).find("unsupported") != std::string::npos,
            "disabled try_wait produced the wrong error");
    }

    profile.mbar_wait_slots = 1;
    try {
        nanosm::validate_device_profile(profile);
        throw std::runtime_error("profile accepted wait slots without a timeout");
    } catch (const nanosm::ProfileError&) {
    }
}

void test_warp_handoff_through_mbar() {
    // The slice's acceptance kernel: producer writes a tile, arrives; consumer sleeps,
    // wakes on the producer's arrival, reads, transforms, writes back. The barrier is a
    // launch-time object (poked via the image) because runtime init has a use-before-init
    // race between warps -- exactly why design.md makes barriers launch-time.
    nanosm::Launch launch;
    launch.active_warps = 2;
    auto& shmem = image_bytes(launch.test_shmem, 512);
    nanosm::MbarState initial;
    initial.valid = true;
    initial.expected = 1;
    initial.remaining = 1;
    write_u64(shmem, 256, nanosm::encode_mbar_state(initial));
    nanosm::Machine machine(canonical_profile(), compile(R"(
        usread ur0, WARP_ID
        ur2r r0, ur0
        setp.eq.u32 p0, r0, 0
        @p0 bra.uni producer
        umov ur1, 256
    consume_wait:
        mbar.try_wait p1, [ur1], 0
        @!p1 bra.uni consume_wait
        sread r2, LANE_ID
        shli r2, r2, 2
        lds r3, [r2]
        addi r3, r3, 100
        sts [r2 + 320], r3
        halt
    producer:
        sread r2, LANE_ID
        shli r2, r2, 2
        sread r4, LANE_ID
        addi r4, r4, 7
        sts [r2], r4
        umov ur1, 256
        mbar.arrive [ur1]
        halt
    )"), std::move(launch));
    machine.run();
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(
            machine.read_shmem_u32(320 + lane * 4) == lane + 107,
            "handoff produced wrong data at lane " + std::to_string(lane));
    }
    const auto& summary = machine.trace().summary();
    require(summary.mbar_sleeps == 1, "consumer never slept on the barrier");
    require(summary.mbar_wakes == 1, "producer's arrival did not wake the consumer");
    require(
        summary.warp_cycles[static_cast<std::size_t>(nanosm::WarpState::mbar_wait)] != 0,
        "sleeping consumer cycles were not attributed to mbar wait");
    uint64_t warp_cycles = 0;
    for (const auto cycles : summary.warp_cycles) warp_cycles += cycles;
    require(
        warp_cycles == summary.cycles * 2,
        "warp attribution does not sum to total with a sleeping warp");
}

void test_mbar_arrival_underflow_panics() {
    nanosm::Launch launch;
    launch.active_warps = 1;
    image_bytes(launch.test_shmem, 128);
    nanosm::Machine machine(canonical_profile(), compile(R"(
        umov ur0, 64
        mbar.init [ur0], 1
        mbar.arrive.expect_tx [ur0], 8
        mbar.arrive [ur0]
        halt
    )"), std::move(launch));
    try {
        machine.run();
        throw std::runtime_error("mbar arrival underflow was accepted");
    } catch (const nanosm::SimError& error) {
        require(
            std::string(error.what()).find("arrival count underflow") != std::string::npos,
            "mbar arrival underflow produced the wrong PANIC");
    }
}

// Complete_tx carries no phase in hardware; the phase
// a sender meant (here the host's stated parity) is sanitizer metadata, so
// this is now a sanitizer PANIC (listed in expects_sanitizer_panic).
void test_mbar_complete_wrong_phase_panics() {
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.test_shmem, 128);
    nanosm::MbarState initial;
    initial.valid = true;
    initial.phase = true;
    initial.expected = 1;
    initial.remaining = 0;
    initial.tx_bytes = 8;
    write_u64(shmem, 64, nanosm::encode_mbar_state(initial));
    nanosm::Machine machine(canonical_profile(), compile("halt"), std::move(launch));
    machine.complete_mbar_tx(64, 8, 0);
    try {
        machine.run();
        throw std::runtime_error("complete_tx against the wrong phase was accepted");
    } catch (const nanosm::SimError& error) {
        require(
            std::string(error.what()).find("wrong phase") != std::string::npos,
            "wrong-phase complete_tx produced the wrong PANIC");
    }
}

} // namespace

// encoding round trip
TEST(Basics, encoding_round_trip) {
    test_encoding_round_trip();
}

// FNV-1a vectors
TEST(Basics, fnv1a_64_vectors) {
    test_fnv1a_64_vectors();
}

// mbar word encoding
TEST(Basics, mbar_word_encoding) {
    test_mbar_word_encoding();
}

// high register metadata
TEST(Basics, high_register_kernel_metadata) {
    test_high_register_kernel_metadata();
}

// RF-derived residency
TEST(Basics, rf_capacity_determines_residency) {
    test_rf_capacity_determines_residency();
}

// compiler controls
TEST(Basics, compiler_controls) {
    test_compiler_controls();
}

// consumer-distance stalls
TEST(Basics, stalls_track_consumer_distance) {
    test_stalls_track_consumer_distance();
}

// counted loop and relocation
TEST(Basics, counted_loop_and_label_relocation) {
    test_counted_loop_and_label_relocation();
}

// uniform role dispatch
TEST(Basics, uniform_role_dispatch) {
    test_uniform_role_dispatch();
}

// nonuniform bra.uni PANIC
TEST(Basics, nonuniform_bra_uni_panics) {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    test_nonuniform_bra_uni_panics();
}

// branch drains async work
TEST(Basics, branch_boundary_drains_async_work) {
    test_branch_boundary_drains_async_work();
}

// qtx wait / nop
TEST(Basics, wait_on_matmul_operands) {
    test_wait_on_matmul_operands();
}

TEST(Basics, wait_on_load_then_use) {
    test_wait_on_load_then_use();
}

TEST(Basics, nop_in_hand_assembly) {
    test_nop_in_hand_assembly();
}

// mbar sleep and wake
TEST(Basics, mbar_try_wait_sleeps_and_engine_completion_wakes) {
    test_mbar_try_wait_sleeps_and_engine_completion_wakes();
}

// mbar test_wait predicate
TEST(Basics, mbar_test_wait_broadcasts_predicate) {
    test_mbar_test_wait_broadcasts_predicate();
}

// mbar try_wait immediate
TEST(Basics, mbar_try_wait_immediate_success) {
    test_mbar_try_wait_immediate_success();
}

// mbar try_wait timeout
TEST(Basics, mbar_try_wait_times_out_false) {
    test_mbar_try_wait_times_out_false();
}

// mbar deadline phase priority
TEST(Basics, mbar_phase_flip_wins_on_try_wait_deadline) {
    test_mbar_phase_flip_wins_on_try_wait_deadline();
}

// mbar try_wait slots
TEST(Basics, mbar_try_wait_slots_limit_parked_warps) {
    test_mbar_try_wait_slots_limit_parked_warps();
}

// profile disables try_wait
TEST(Basics, profile_can_disable_mbar_try_wait) {
    test_profile_can_disable_mbar_try_wait();
}

// staged pipeline kernel
TEST(Basics, staged_pipeline_kernel) {
    test_staged_pipeline_kernel();
}

// launch image validation
TEST(Basics, launch_image_rejects_bad_segment_hex) {
    test_launch_image_rejects_bad_segment_hex();
}

// the launch passes arguments; the kernel declares the rest
TEST(Basics, launch_arguments_reach_the_kernel) {
    test_launch_arguments_reach_the_kernel();
}

TEST(Basics, launch_refusals) {
    test_launch_refusals();
}

TEST(Basics, section_alias_waits_on_its_flag) {
    test_section_alias_waits_on_its_flag();
}

TEST(Basics, mbar_array) {
    test_mbar_array();
}

TEST(Basics, umov_takes_a_32_bit_word) {
    test_umov_takes_a_32_bit_word();
}

TEST(Basics, test_preload_is_reported_and_kept_off_the_block) {
    test_test_preload_is_reported_and_kept_off_the_block();
}

TEST(Basics, early_launch_read_panics) {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    test_early_launch_read_panics();
}

// mbar warp handoff
TEST(Basics, seventh_depbar_reuses_the_oldest) {
    test_seventh_depbar_reuses_the_oldest();
}

TEST(Basics, warp_handoff_through_mbar) {
    test_warp_handoff_through_mbar();
}

// mbar arrival underflow
TEST(Basics, mbar_arrival_underflow_panics) {
    test_mbar_arrival_underflow_panics();
}

// mbar wrong-phase completion
TEST(Basics, mbar_complete_wrong_phase_panics) {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    test_mbar_complete_wrong_phase_panics();
}

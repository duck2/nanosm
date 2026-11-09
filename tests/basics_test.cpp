// Encoding, the compiler's controls, branches, mbar basics, launch images.
// Moved from the original tests/tests.cpp (trash/test-setup-2026-10-22.md);
// each test's body is unchanged, its TEST wrapper is at the end.
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
    // I3 (rfc12 section 8): the tx field is a signed 32-bit count (two's
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
    const auto program = nanosm::compile_qtx(R"(
        mov  r0, 0
        lds  r1, [r0]
        lds  r2, [r0 + 64]
        fadd r4, r1, r2
        sts  [r0 + 128], r4
        halt
    )", canonical_profile());
    // I6a (src/rf.hpp): a fixed result is visible issue_to_visible after its
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

void test_stalls_track_consumer_distance() {
    // fadd's result is consumed three instructions later, so only what its
    // distance (src/rf.hpp: issue_to_visible - the consumer's issue_to_sample)
    // exceeds those three issues is bought with a stall, and it lands just
    // before the consumer. The mov feeding the fadd stalls the full distance.
    const auto& profile = canonical_profile();
    const auto program = nanosm::compile_qtx(R"(
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
    image_bytes(launch.shmem_image, 4);
    nanosm::Machine machine(
        canonical_profile(),
        nanosm::assemble_asm(nanosm::emit_asm(program), canonical_profile()),
        std::move(launch));
    machine.run(); // the hazard sanitizer accepts the minimal schedule
}

void test_counted_loop_and_label_relocation() {
    const auto program = nanosm::compile_qtx(R"(
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
        setp.eq p0, r0, 0
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
        setp.lt p0, r0, 8
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
    const auto program = nanosm::compile_qtx(R"(
        mov r0, 0
        ldg r1, [r0]
        bra done
    done:
        halt
    )", canonical_profile());
    require(program.instructions[2].control.wait_mask == 1, "branch did not drain its live load depbar");

    const auto fixed = nanosm::compile_qtx(R"(
        mov r0, 1
        fadd r1, r0, r0
        bra done
    done:
        halt
    )", canonical_profile());
    // The fadd issued one cycle before the branch; the block after it must
    // issue no earlier than one cycle before the fadd's result is visible (its
    // first sample is after the issue register): issue_to_visible - 1 cycles
    // after the fadd, issue_to_visible - 2 after the branch (src/rf.hpp).
    require(fixed.instructions[2].control.stall ==
            nanosm::issue_to_visible(fixed.instructions[1], canonical_profile()) -
                nanosm::issue_register_stages - 1,
        "branch did not drain a live fadd writeback");
}

void test_mbar_try_wait_sleeps_and_engine_completion_wakes() {
    nanosm::Launch launch;
    launch.active_warps = 1;
    image_bytes(launch.shmem_image, 128);
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

    // A guard only: the warp parks once its init, arrive and probe have
    // crossed issue, dispatch (I6a) and the RMW engine.
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
        // I2: read grant to write grant, D + Lb + 1 cycles (was 2); I2.1: + C.
        require(event.duration_cycles == rmw_busy_cycles(), "mbar RMW did not hold A for D + Lb + C + 1 cycles");
    }
    require(rmw_slices == 3, "trace did not contain every mbar RMW service interval");
}

void test_mbar_test_wait_broadcasts_predicate() {
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.shmem_image, 128);
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
    auto& shmem = image_bytes(launch.shmem_image, 128);
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
    auto& shmem = image_bytes(launch.shmem_image, 128);
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

// I2: the wake comes when the RMW's write reaches the RAM (rfc12 section 6),
// 2D + Lb (I2.1: + C) cycles after the event's claim, which is the cycle after
// the park. The timeout puts the deadline in that cycle (was 2: claim, then
// write).
void test_mbar_phase_flip_wins_on_try_wait_deadline() {
    auto profile = canonical_profile();
    // The wake lands exactly on the deadline: the claim is the cycle after the
    // park, the wake the RMW write's commit, D + Lb + C + D later (I2.1: + C,
    // the RMW's ALU stages).
    profile.mbar_try_wait_timeout_cycles = 2 * profile.shmem_port_a_depth + profile.shmem_read_latency_cycles +
        profile.shmem_rmw_compute_cycles + 1;
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.shmem_image, 128);
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
    auto& shmem = image_bytes(launch.shmem_image, 128);
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
    nanosm::Launch launch;
    launch.active_warps = 2;
    auto& shmem = image_bytes(launch.shmem_image, 2048);
    nanosm::MbarState barrier;
    barrier.valid = true;
    barrier.expected = 1;
    barrier.remaining = 1;
    for (const uint32_t address : {1024u, 1032u, 1040u, 1048u}) {
        write_u64(shmem, address, nanosm::encode_mbar_state(barrier));
    }
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
    require(loaded.launch.active_warps == 2, "launch image lost its active warp count");
    require(loaded.launch.shmem_image.size() == 1, "launch image lost its shmem segment");
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
    require(summary.mbar_timeouts != 0, "gmem latency never outlasted the try_wait timeout");
    require(summary.gmem_commands == 3 * tiles, "wrong gmem command count for the pipeline");
    uint64_t warp_cycles = 0;
    for (const auto cycles : summary.warp_cycles) warp_cycles += cycles;
    require(warp_cycles == summary.cycles * 2, "pipeline warp attribution does not sum");
}

void test_launch_image_rejects_bad_segment_hex() {
    nanosm::LaunchImage input;
    input.kernel = compile("halt");
    input.launch.active_warps = 1;
    input.launch.shmem_image.push_back({0, {0xab}});
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
        nanosm::compile_qtx("umov ur0, 64\nmbar.try_wait p0, [ur0], 0\nhalt", profile);
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
    auto& shmem = image_bytes(launch.shmem_image, 512);
    nanosm::MbarState initial;
    initial.valid = true;
    initial.expected = 1;
    initial.remaining = 1;
    write_u64(shmem, 256, nanosm::encode_mbar_state(initial));
    nanosm::Machine machine(canonical_profile(), compile(R"(
        usread ur0, WARP_ID
        ur2r r0, ur0
        setp.eq p0, r0, 0
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
    image_bytes(launch.shmem_image, 128);
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

// I3 (rfc12 section 8): complete_tx carries no phase in hardware; the phase
// a sender meant (here the host's stated parity) is sanitizer metadata, so
// this is now a sanitizer PANIC (listed in expects_sanitizer_panic).
void test_mbar_complete_wrong_phase_panics() {
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.shmem_image, 128);
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

// mbar warp handoff
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

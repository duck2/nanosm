// The tensor core and tmem: numerics, the walker, mbar engine events, I7, I9, D15, I8.
// Moved from the original tests/tests.cpp (trash/test-setup-2026-10-22.md);
// each test's body is unchanged, its TEST wrapper is at the end.
#include "test_support.hpp"

namespace {

using namespace nanosm_test;

void test_tmem_profile_metadata_and_round_trip() {
    auto profile = canonical_profile();
    for (const auto field : {&nanosm::DeviceProfile::tmem_port_depth,
             &nanosm::DeviceProfile::tmem_read_latency_cycles}) {
        auto deeper = profile;
        ++(deeper.*field);
        require(nanosm::device_profile_fingerprint(profile) != nanosm::device_profile_fingerprint(deeper),
            "a tmem depth was omitted from the profile fingerprint");
    }

    const std::string source = R"(
        mov r0, 2044
        shli r0, r0, 20
        mov r1, 1
        mov r2, 2
        mov r3, 3
        mov r4, 4
        mov r5, 5
        mov r6, 6
        mov r7, 7
        mov r10, 10
        mov r11, 11
        usread ur8, WARP_ID
        ushli ur8, ur8, 10
        umov ur9, 0
        acc.st.v8 [ur8, 64], r0
        add r20, r10, r11
        acc.ld.v8 r40, [ur8, 64]
        halt
    )"; // warp w: rows 16 w .. 16 w + 15 of 64-word rows, pointer 1024 w
    const auto scheduled = nanosm::compile_qtx(source, profile);
    require(scheduled.instructions[15].opcode == nanosm::Opcode::add,
        "directed tmem test instruction index changed unexpectedly");
    require(scheduled.instructions[15].control.wait_mask == 0,
        "independent arithmetic was unnecessarily serialized behind acc.st");
    const auto protected_store = nanosm::compile_qtx(R"(
        mov r0, 1
        umov ur8, 0
        umov ur9, 0
        acc.st [ur8, 64], r0
        mov r0, 2
        halt
    )", profile);
    const auto store_bar = protected_store.instructions[3].control.set_depbar;
    require((protected_store.instructions[4].control.wait_mask & (1u << store_bar)) != 0,
        "qtxc did not protect the async acc.st source lifetime");
    const auto assembly = nanosm::emit_asm(scheduled);
    const auto kernel = nanosm::assemble_asm(assembly, profile);
    const auto loaded = nanosm::load_hex(nanosm::emit_hex(kernel));
    require(loaded.words == kernel.words && nanosm::emit_hex(kernel).find(".acc") == std::string::npos,
        "the hex round trip changed the kernel or carries an accumulator table");

    nanosm::Launch launch;
    launch.active_warps = 4;
    nanosm::Machine machine(profile, loaded, std::move(launch), {.timeline = true, .debug = true});
    machine.run(10000);
    require(machine.trace().summary().tmem_store_beats == 32 &&
            machine.trace().summary().tmem_load_beats == 32,
        "tmem trace did not expose every vector beat");
    for (uint32_t warp = 0; warp < 4; ++warp) {
        for (uint32_t lane = 0; lane < profile.lanes; ++lane) {
            require(machine.read_reg(warp, lane, 40) == 0x7fc00000u,
                "tmem transfer did not preserve raw NaN payload bits");
            for (uint32_t j = 1; j < 8; ++j) {
                require(machine.read_reg(warp, lane, 40 + j) == j,
                    "RF -> tmem -> RF vector round trip changed bits");
            }
            require(machine.read_reg(warp, lane, 20) == 21,
                "concurrent arithmetic made no forward progress");
        }
    }
}

void test_tmem_transfer_ordering_and_store_lifetime() {
    const auto& profile = canonical_profile();
    const std::string source = R"(
        mov r0, 17
        mov r1, 34
        umov ur8, 0
        umov ur9, 4096
        acc.st.v8 [ur8, 16], r0
        acc.st.v8 [ur9, 16], r0
        mov r0, 51
        acc.ld r2, [ur8, 16]
        acc.ld r3, [ur9, 16]
        halt
    )";
    const auto scheduled = nanosm::compile_qtx(source, profile);
    const auto& first_store = scheduled.instructions[4];
    const auto& second_store = scheduled.instructions[5];
    const auto& overwrite = scheduled.instructions[6];
    const auto& first_load = scheduled.instructions[7];
    require(first_store.control.set_depbar < 6 &&
            second_store.control.set_depbar == first_store.control.set_depbar,
        "concurrent accumulator stores did not share a counted completion depbar");
    require((overwrite.control.wait_mask & (1u << first_store.control.set_depbar)) != 0,
        "overwrite did not wait for every older accumulator-store reader");
    require(first_load.control.wait_mask == 0,
        "a wait already paid by the overwrite was redundantly retained on the load");

    const auto ordered = nanosm::compile_qtx(R"(
        mov r0, 17
        umov ur8, 0
        umov ur9, 0
        acc.st [ur8, 16], r0
        acc.ld r2, [ur8, 16]
        halt
    )", profile);
    nanosm::Launch ordered_launch;
    ordered_launch.active_warps = 1;
    nanosm::Machine ordered_machine(
        profile, nanosm::assemble_asm(nanosm::emit_asm(ordered), profile),
        std::move(ordered_launch), {.timeline = true, .debug = true});
    ordered_machine.run(1000);
    require(ordered_machine.read_reg(0, 0, 2) == 17,
        "qtxc's tmem order did not put an accumulator load after its store");
    uint64_t store_cycle = 0;
    uint64_t load_cycle = 0;
    for (const auto& event : ordered_machine.trace().events()) {
        if (event.name == "acc.st beat") store_cycle = event.start_cycle;
        if (event.name == "acc.ld beat") load_cycle = event.start_cycle;
    }
    require(store_cycle < load_cycle,
        "an accumulator load sampled tmem before the older store's final tmem write");

    nanosm::Launch launch;
    launch.active_warps = 1;
    nanosm::Machine machine(
        profile, nanosm::assemble_asm(nanosm::emit_asm(scheduled), profile),
        std::move(launch), {.timeline = true, .debug = true});
    machine.run(1000);
    for (uint32_t lane = 0; lane < profile.lanes; ++lane) {
        require(machine.read_reg(0, lane, 2) == 17 && machine.read_reg(0, lane, 3) == 17,
            "a younger overwrite changed a deferred accumulator-store source");
        require(machine.read_reg(0, lane, 0) == 51,
            "the source overwrite did not execute after both stores completed");
    }
    require(machine.trace().summary().tmem_store_beats == 16 &&
            machine.trace().summary().tmem_load_beats == 2,
        "scalar transfer completion did not account for each final tmem/RF beat");

    // Hand-written assembly is allowed to omit the compiler's wait, but the
    // simulator must diagnose the architectural WAR violation rather than
    // silently scoreboard or capture the source early.
    auto unsafe = nanosm::assemble_asm(nanosm::emit_asm(scheduled), profile);
    auto unsafe_overwrite = nanosm::decode(unsafe.words[6]);
    unsafe_overwrite.control.wait_mask = 0;
    unsafe.words[6] = nanosm::encode(unsafe_overwrite);
    nanosm::Launch unsafe_launch;
    unsafe_launch.active_warps = 1;
    try {
        nanosm::Machine unsafe_machine(
            profile, std::move(unsafe), std::move(unsafe_launch), {.timeline = true, .debug = true});
        unsafe_machine.run(1000);
        throw std::runtime_error("missing accumulator-store WAR wait did not panic");
    } catch (const nanosm::SimError& error) {
        require(std::string(error.what()).find("RF WAR hazard") != std::string::npos,
            "missing accumulator-store wait produced the wrong diagnostic");
    }
}

void test_mma_raw_window_numerics() {
    using nanosm::mma_dot_f16;
    const std::array<uint16_t, 4> zero{};
    const std::array<uint16_t, 4> one{0x3c00, 0, 0, 0};
    const auto check = [&](std::array<uint16_t, 4> a, std::array<uint16_t, 4> b,
                           uint32_t c, uint32_t expected) {
        require(mma_dot_f16(a, b, c) == expected, "MMA raw-window directed vector mismatch");
    };
    // Cancellation within a fusion group retains the unit that serial FMA loses.
    check({0x6c00, 0x3c00, 0xec00, 0}, {0x6c00, 0x3c00, 0x6c00, 0}, 0, 0x3f800000);
    check({0x6c00, 0xec00, 0x3c00, 0}, {0x6c00, 0x6c00, 0x3c00, 0}, 0, 0x3f800000);
    // But alignment can discard a small C before the large products cancel.
    check({0x6c00, 0xec00, 0, 0}, {0x6c00, 0x6c00, 0, 0}, 0x3e800000, 0);
    // Final truncation, not RNE: 1 + 1.5 * 2^-24 -> 1, for both signs.
    check({0x3c00, 0x3e00, 0, 0}, {0x3c00, 1, 0, 0}, 0, 0x3f800000);
    check({0xbc00, 0xbe00, 0, 0}, {0x3c00, 1, 0, 0}, 0, 0xbf800000);
    check(zero, zero, 1, 1); // FP32 C subnormals survive when there is no larger term
    check(zero, zero, 0x80000001, 0x80000001);
    check(zero, zero, 0x80000000, 0);
    check({0x7c00, 0, 0, 0}, one, 0, 0x7f800000);
    check({0xfc00, 0, 0, 0}, one, 0, 0xff800000);
    check({0x7c00, 0, 0, 0}, zero, 0, 0x7fffffff);
    check({0x7c00, 0, 0, 0}, one, 0xff800000, 0x7fffffff);
    check(zero, zero, 0x7f800001, 0x7fffffff);
    // All half encodings: identity, including denormals and signaling NaNs.
    for (uint32_t h = 0; h < 65536; ++h) {
        const uint32_t expected = (h & 0x7fff) == 0 ? 0
            : (h & 0x7c00) == 0x7c00 && (h & 0x3ff) != 0 ? 0x7fffffff
            : nanosm::half_to_float_bits(static_cast<uint16_t>(h));
        check({static_cast<uint16_t>(h), 0, 0, 0}, one, 0, expected);
    }
    // Frozen raw B200 FP16 K=16 vectors from SF Tensor validation.py at
    // d96b82180ac9041c09295f3a6a3c2c4e592ddce6 (seed 103). These validate
    // the arithmetic helper, not the emulator's smaller fusion boundary.
    const std::array<uint16_t, 16> a0{0xfaa3,0xb6dc,0x74cf,0xef28,0xb806,0xb24c,0xbc21,0xd179,
        0x3703,0xb067,0x13fa,0xe990,0x9a4d,0xbd38,0x1f36,0xa782};
    const std::array<uint16_t, 16> b0{0x61d9,0x760f,0xe3c3,0x568f,0xdf45,0xacdf,0x1e62,0x7d36,
        0x150d,0x21d3,0x38f4,0x954f,0xd3fd,0xf912,0x0654,0x0b47};
    const std::array<uint16_t, 16> a1{0xf638,0x1b73,0xb103,0x20ac,0x9b70,0x85ff,0x4b2e,0x2ef5,
        0xbef1,0x68d1,0xdc66,0xbbb2,0x94ac,0xb916,0xac3a,0x6a3f};
    const std::array<uint16_t, 16> b1{0x3a00,0xc940,0xa841,0x928f,0xc577,0xd3b3,0x9e19,0x3fb6,
        0x1608,0x5e7e,0x33db,0x9490,0x37f2,0xf226,0xb17e,0x9e4b};
    require(mma_dot_f16(a0, b0, 0x39a59521) == 0x7fffffff, "B200 NaN vector mismatch");
    require(mma_dot_f16(a1, b1, 0x82890b9b) == 0x49776bba, "B200 finite vector mismatch");

    // Compact offline regression against 20,000 pinned upstream oracle words.
    // xorshift32 order: four A, four B, C; FNV1a over little-endian result bytes.
    uint32_t seed = 0x12345678;
    const auto next = [&]() { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return seed; };
    uint64_t digest = 14695981039346656037ull;
    for (uint32_t i = 0; i < 20000; ++i) {
        std::array<uint16_t, 4> a, b;
        for (auto& h : a) h = static_cast<uint16_t>(next());
        for (auto& h : b) h = static_cast<uint16_t>(next());
        const uint32_t result = mma_dot_f16(a, b, next());
        for (unsigned shift = 0; shift < 32; shift += 8)
            digest = (digest ^ ((result >> shift) & 255)) * 1099511628211ull;
    }
    require(digest == 0xbc4eca1d7a740939ull, "MMA raw-window upstream corpus mismatch");

    std::ifstream input("profiles/nanosm-v0.json");
    const auto document = nlohmann::json::parse(input);
    require(document.at("mma").at("numeric_model") == nanosm::mma_numeric_model,
        "canonical profile does not name the implemented MMA arithmetic");
    // Old or invented arithmetic must not silently acquire the new semantics.
    for (const bool missing : {false, true}) {
        auto changed = document;
        if (missing) changed["mma"].erase("numeric_model");
        else changed["mma"]["numeric_model"] = "serial_fma_fp32_reference";
        bool rejected = false;
        try { (void)nanosm::parse_device_profile(changed.dump()); }
        catch (const nanosm::ProfileError&) { rejected = true; }
        require(rejected, "missing/unsupported MMA arithmetic accepted by profile parser");
    }
}

void test_mma_fusion_boundaries() {
    // Same eight products, different physical fusion extent. Four-wide must
    // round the first group to FP32 before cancellation in the second group.
    for (const uint32_t core_k : {4u, 8u}) {
        auto profile = canonical_profile();
        profile.mma_layout = nanosm::TileLayout::dense;
        profile.mma_core_k = core_k;
        const auto scheduled = nanosm::compile_qtx(
            "umov ur0, 0\numov ur1, 1024\numov ur4, 4096\numov ur8, 0\n"
            "mbar.init [ur4], 1\nmma.async.init.m64n64k8 [ur8], [ur0], [ur1]\nmma.commit [ur4]\n"
            "wait:\nmbar.try_wait p0, [ur4], 0\n@!p0 bra.uni wait\nacc.ld r10, [ur8, 64]\nhalt\n", profile);
        const auto kernel = nanosm::assemble_asm(nanosm::emit_asm(scheduled), profile);
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.test_shmem, 8192);
        const std::array<uint16_t, 8> a{0x6c00, 0x3c00, 0, 0, 0xec00, 0, 0, 0};
        const std::array<uint16_t, 8> b{0x6c00, 0x3c00, 0, 0, 0x6c00, 0, 0, 0};
        for (uint32_t row = 0; row < 64; ++row)
            for (uint32_t k = 0; k < 8; ++k) write_u16(shmem, 2 * (row * 8 + k), a[k]);
        for (uint32_t k = 0; k < 8; ++k)
            for (uint32_t col = 0; col < 64; ++col) write_u16(shmem, 1024 + 2 * (k * 64 + col), b[k]);
        nanosm::Machine machine(profile, kernel, std::move(launch));
        machine.run(20000);
        for (uint32_t lane = 0; lane < profile.lanes; ++lane)
            require(machine.read_reg(0, lane, 10) == (core_k == 4 ? 0u : 0x3f800000u),
                "MMA did not round at the physical core_k boundary");
    }
}

void test_autonomous_mma_gemm_and_commit() {
    auto profile = canonical_profile();
    auto dense = profile;
    dense.mma_layout = nanosm::TileLayout::dense;
    require(nanosm::device_profile_fingerprint(profile) != nanosm::device_profile_fingerprint(dense),
        "MMA operand layout was omitted from the profile fingerprint");

    std::ostringstream qtx;
    qtx << ""
        << "umov ur0, 0\numov ur1, 1024\numov ur4, 4096\numov ur8, 0\numov ur9, 0\n"
        << "mbar.init [ur4], 1\n"
        << "mma.async.init.m64n64k8 [ur8], [ur0], [ur1]\n";
    for (uint32_t chunk = 1; chunk < 64; ++chunk)
        qtx << "mma.async.m64n64k8 [ur8], [ur0], [ur1]\n";
    // The GEMM proof passes because of the protocol: the commit mbar is
    // explicitly observed before any transfer touches C (cases 01/13).
    qtx << "mma.commit [ur4]\n"
        << "wait_commit:\n"
        << "mbar.try_wait p0, [ur4], 0\n"
        << "@!p0 bra.uni wait_commit\n"
        << "acc.ld r10, [ur8, 64]\nhalt\n";
    const auto scheduled = nanosm::compile_qtx(qtx.str(), profile);
    require(scheduled.instructions[6].control.set_depbar < 6 &&
            scheduled.instructions[7].control.set_depbar < 6,
        "qtxc did not allocate MMA operand-consumed depbars");
    const auto kernel = nanosm::assemble_asm(nanosm::emit_asm(scheduled), profile);
    require(nanosm::decode(kernel.words[6]).opcode == nanosm::mma_opcode(0, true) && // m64n64k8, init
            nanosm::decode(kernel.words[70]).opcode == nanosm::Opcode::mma_commit,
        "MMA opcodes did not survive assembly encoding");

    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.test_shmem, 8192);
    auto a_address = [](uint32_t m, uint32_t k) {
        return 4 * (4 * m + ((k >> 1) ^ ((m >> 2) & 3))) + 2 * (k & 1);
    };
    auto b_address = [](uint32_t k, uint32_t n) { // B stored as n rows of k: A's rule
        return 1024 + 4 * (4 * n + ((k >> 1) ^ ((n >> 2) & 3))) + 2 * (k & 1);
    };
    for (uint32_t m = 0; m < 64; ++m)
        for (uint32_t k = 0; k < 8; ++k) write_u16(shmem, a_address(m, k), 0x3c00); // 1
    for (uint32_t k = 0; k < 8; ++k)
        for (uint32_t n = 0; n < 64; ++n) write_u16(shmem, b_address(k, n), 0x4000); // 2

    nanosm::Machine machine(profile, kernel, std::move(launch), {.timeline = true, .debug = true});
    machine.run(50000);
    for (uint32_t lane = 0; lane < profile.lanes; ++lane)
        require(machine.read_reg(0, lane, 10) == nanosm::float_bits(1024.0f),
            "64-command 64x512 MMA did not produce the ordered FP32 GEMM result");
    require(machine.read_mbar_state(4096).phase,
        "mma.commit did not publish the result-ready mbar after final tmem writes");
    const auto& summary = machine.trace().summary();
    // 4x4x8 core: 256 steps per command; each command reads its 16 A blocks
    // once, each with its first B block (two rounds), and 240 B-only steps
    // (one round each): 16 x 17 = 272 rounds per command.
    require(summary.mma_commands == 64 && summary.mma_input_steps == 64 * 256 &&
            summary.mma_result_steps == 64 * 256 && summary.mma_a_block_reads == 64 * 16 &&
            summary.mma_shmem_rounds == 64 * 272 && summary.mma_recurrence_stalls == 0,
        "canonical MMA trace did not show hold-A / stream-B steps at one round per B block");
    uint64_t second_admission = std::numeric_limits<uint64_t>::max();
    uint64_t first_last_write = 0;
    for (const auto& event : machine.trace().events()) {
        if (event.category != nanosm::TraceCategory::mma) continue;
        if (event.name == "mma.async" && event.id == 2) second_admission = event.start_cycle;
        if (event.name == "MMA tmem write" && event.id == 1)
            first_last_write = std::max(first_last_write, event.start_cycle);
    }
    require(second_admission < first_last_write,
        "adjacent MMA commands drained at the command boundary instead of overlapping");
}

void test_worker_acc_store_mbar_handoff() {
    const auto& profile = canonical_profile();
    const auto scheduled = nanosm::compile_qtx(R"(
        sread r0, THREAD_ID
        usread ur8, WARP_ID
        ushli ur8, ur8, 10
        umov ur9, 0
        umov ur12, 4096
        acc.st.v8 [ur8, 64], r0
        mbar.arrive [ur12]
        halt
    )", profile);
    const auto& store = scheduled.instructions[5];
    const auto& arrive = scheduled.instructions[6];
    require((arrive.control.wait_mask & (1u << store.control.set_depbar)) != 0,
        "worker mbar.arrive was not gated by its accumulator-store completion");
    nanosm::Launch launch;
    launch.active_warps = 2;
    auto& shmem = image_bytes(launch.test_shmem, 8192);
    nanosm::MbarState initial;
    initial.valid = true;
    initial.expected = 2;
    initial.remaining = 2;
    write_u64(shmem, 4096, nanosm::encode_mbar_state(initial));
    nanosm::Machine machine(profile,
        nanosm::assemble_asm(nanosm::emit_asm(scheduled), profile), std::move(launch));
    machine.run(2000);
    require(machine.read_mbar_state(4096).phase,
        "shared worker mbar did not receive exactly one post-store arrival per warp");
}

// A TMA complete_tx and an MMA commit become pending in the same cycle r, while
// the RMW engine is still busy (the write stage of an earlier TMA event). Both
// are served, alternating TMA, MMA, TMA, and the mma.async queued behind the
// commit is accepted in cycle r and starts fetching in r + 1, exactly as in a
// run with no TMA event at all: the commit presents into the MMA's own register.
// (The alternation here follows the last-served bit: T1 was served last, so
// the tie at r - 1 + busy goes to the MMA.) I2: services are busy = D + Lb + 1
// cycles apart (was 2; I2.1: D + Lb + C + 1); the commit, held from r + 1, waits busy - 2 cycles
// for T1 and T2 waits busy cycles for the commit (was 0 + 2).
// Before 2026-10-07 the two engines shared ONE event slot. In cycle r the
// commit found it holding the TMA event and stayed at the head of the MMA
// queue: the order was TMA, TMA, MMA, and the mma.async behind it (accepted
// at r into the entry the retired init command freed) could not start
// fetching until r + 2, one cycle late -- one cycle for every cycle the TMA
// event sat in the slot. (Measured on 29e320f: fetch 525 instead of 524.)
void test_mbar_engine_tma_event_does_not_stall_mma_queue() {
    constexpr uint32_t commit_mbar = 4096;
    constexpr uint32_t tx_mbar = 4104;
    const auto kernel = compile(R"(
        umov ur0, 0
        umov ur1, 1024
        umov ur4, 4096
        mma.async.init.m64n64k8 [ur8], [ur0], [ur1]   ; ur8, never written, is 0: the result at tmem 0
        mma.commit [ur4]
        mma.async.m64n64k8 [ur8], [ur0], [ur1]
        halt
    )");
    const auto make_launch = [&] {
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.test_shmem, 8192);
        write_mbar(shmem, commit_mbar, 1, 0);
        write_mbar(shmem, tx_mbar, 0, 128); // arrived; waits for 2 x 64 bytes
        return launch;
    };
    // The queued mma.async is command 3 (1 = the init command, 2 = the commit).
    const auto queued_accept_and_fetch = [](const nanosm::Machine& machine) {
        uint64_t accept = 0;
        uint64_t fetch = std::numeric_limits<uint64_t>::max();
        for (const auto& event : machine.trace().events()) {
            if (event.category != nanosm::TraceCategory::mma || event.id != 3) continue;
            if (event.name == "mma.async") accept = event.start_cycle;
            if (event.name == "MMA input step") fetch = std::min(fetch, event.start_cycle);
        }
        return std::pair{accept, fetch};
    };

    // Reference, no TMA event: the commit presents in cycle r, the cycle its
    // predecessor retires, and frees the queue slot the mma.async waits for:
    // the mma.async issues in r and reaches the MMA queue at its dispatch,
    // accept = r + issue_to_dispatch (I6a), fetching from the next cycle.
    nanosm::Machine reference(canonical_profile(), kernel, make_launch(), {.timeline = true});
    reference.run(5000);
    const auto [accept, fetch] = queued_accept_and_fetch(reference);
    const auto r = accept - nanosm::issue_to_dispatch(canonical_profile());
    require(r > 2 && fetch == accept + 1, "reference MMA queue timing changed; re-derive the test");
    require(mbar_rmw_services(reference) ==
            std::vector<RmwService>{{r + 1, "arrive (mma.commit), address=4096"}},
        "reference commit was not served at r + 1");

    nanosm::Machine machine(canonical_profile(), kernel, make_launch(), {.timeline = true});
    while (machine.cycle() + 1 < r) machine.tick();
    machine.complete_mbar_tx(tx_mbar, 64, 0); // T1, pending in r - 1: the engine is free
    machine.tick();
    require(!machine.tma_mbar_event_held(), "T1 did not take the free RMW engine");
    machine.complete_mbar_tx(tx_mbar, 64, 0); // T2, pending in r: the engine is in T1's write
    require(!machine.mma_mbar_event_held(), "the commit presented before cycle r");
    machine.tick(); // cycle r
    require(machine.tma_mbar_event_held() && machine.mma_mbar_event_held(),
        "the TMA event and the commit were not pending together in cycle r");
    machine.run(5000);

    require(queued_accept_and_fetch(machine) == std::pair{accept, fetch},
        "a held TMA event stalled the MMA queue behind the commit");
    const auto busy = rmw_busy_cycles();
    require(mbar_rmw_services(machine) == std::vector<RmwService>{
                {r - 1, "complete_tx (tma), address=4104"},
                {r - 1 + busy, "arrive (mma.commit), address=4096"},
                {r - 1 + 2 * busy, "complete_tx (tma), address=4104"}},
        "engine events were not served once each, alternating TMA, MMA, TMA");
    require(machine.trace().summary().mbar_engine_waits_on_other == 2 * busy - 2,
        "the commit should wait out T1's RMW from r + 1, and T2 the whole of the commit's");
    require(machine.read_mbar_state(commit_mbar).phase, "the commit's arrive was lost");
    const auto tx = machine.read_mbar_state(tx_mbar);
    require(tx.phase && tx.tx_bytes == 0, "a TMA completion was lost");
}

// Both engines hold an event at the first free point: the TMA goes first (the
// last-served bit starts at MMA). The TMA presents again before the next free
// point, so both are pending again: the MMA goes first. Then the second
// commit and that TMA event tie, and the TMA goes. complete_mbar_tx's
// occupied PANIC belongs to the TMA's register alone.
void test_mbar_engine_events_alternate_tma_first() {
    constexpr uint32_t commit0 = 4096;
    constexpr uint32_t tx_mbar = 4104;
    constexpr uint32_t commit1 = 4112;
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.test_shmem, 8192);
    write_mbar(shmem, commit0, 1, 0);
    write_mbar(shmem, commit1, 1, 0);
    write_mbar(shmem, tx_mbar, 0, 128);
    nanosm::Machine machine(canonical_profile(), compile(R"(
        umov ur4, 4096
        umov ur6, 4112
        mma.commit [ur4]
        mma.commit [ur6]
        halt
    )"), std::move(launch), {.timeline = true});

    while (!machine.mma_mbar_event_held()) tick_capped(machine); // the first commit presented
    require(!machine.tma_mbar_event_held() && mbar_rmw_services(machine).empty(),
        "an engine event was served before the first commit");
    const auto first = machine.cycle(); // the first free point with an engine event
    machine.complete_mbar_tx(tx_mbar, 64, 0); // accepted although the MMA register is full
    try {
        machine.complete_mbar_tx(tx_mbar, 64, 0);
        throw std::runtime_error("a second event entered the full TMA register");
    } catch (const nanosm::SimError& error) {
        require(std::string(error.what()).find("TMA mbar event register is occupied") !=
                std::string::npos, "wrong full-register PANIC");
    }
    machine.tick(); // first: both held
    require(!machine.tma_mbar_event_held() && machine.mma_mbar_event_held(),
        "the first tie did not serve the TMA");
    machine.tick(); // the TMA event's RMW is under way; the commit still waits
    machine.complete_mbar_tx(tx_mbar, 64, 0); // pending again from first + 2, before the next free point
    machine.run(1000);

    // I2: services are busy = D + Lb + 1 cycles apart (was 2); I2.1: + C.
    const auto busy = rmw_busy_cycles();
    require(mbar_rmw_services(machine) == std::vector<RmwService>{
                {first, "complete_tx (tma), address=4104"},
                {first + busy, "arrive (mma.commit), address=4096"},
                {first + 2 * busy, "complete_tx (tma), address=4104"},
                {first + 3 * busy, "arrive (mma.commit), address=4112"}},
        "tied engine events did not alternate TMA, MMA, TMA, MMA");
    // Each of the three ties holds the loser through all of the winner's RMW.
    require(machine.trace().summary().mbar_engine_waits_on_other == 3 * busy,
        "engine_waits_on_other does not count the winner's whole RMW per tie");
    require(machine.read_mbar_state(commit0).phase && machine.read_mbar_state(commit1).phase,
        "a commit's arrive was lost");
    const auto tx = machine.read_mbar_state(tx_mbar);
    require(tx.phase && tx.tx_bytes == 0, "a TMA completion was lost");
}

// The same-cycle case with a real TMA load. After the commit, umovs set the
// tma.load's uniform-register operands (buffer, barrier, coordinate), then
// filler movs set its issue delay: vector movs read no uniform register, so
// they issue one per cycle, and once they cover the umov -> tma distance each
// one moves the tma.load a cycle later. The sweep moves its completion across
// cycle r (the commit's presentation) in two-cycle steps (DRAM clock), and
// the test finds the delay at which both present in cycle r. Each event is
// visible to the RMW engine from the next cycle, so at r + 1 they tie: the
// TMA goes first (the last-served bit starts at MMA) and the commit follows
// once the TMA's RMW is done, at r + 1 + rmw_busy_cycles(). At every delay
// the queued mma.async is still accepted at r (the MMA queue never waits on
// the TMA; the tma.load's operand reads are long over by then) and the TMA
// command ends the cycle after its event is claimed.
// Two-cycle steps reach r only at one phase of r against the DRAM clock, and
// which phase the kernel lands on follows its issue distances: when the MMA
// addresses moved to uniform registers (2026-11-08; umov -> mma.async is 4
// cycles, a vector mov's was 6), r moved by one cycle and no delay aligned. So the sweep
// runs with no lead filler and, if no delay aligned, again with one filler mov
// at the start of the kernel, which moves r (and every later cycle) by one.
// Before 2026-10-08 (stepping discipline) the TMA's event could be claimed in
// the cycle it was presented, an artifact of tick()'s call order, while the
// MMA's waited a cycle: at the aligned delay the TMA was served at r and the
// commit at r + 2 (one cycle of waiting). Before 2026-10-07, at the aligned
// delay the commit filled the shared slot right behind the TMA's claimed
// event, and the TMA command could not end (nor free its queue entry) until
// that slot emptied (measured on 29e320f: 529 instead of 527).
void test_mbar_engine_real_tma_and_commit_same_cycle() {
    constexpr uint32_t commit_mbar = 4096;
    constexpr uint32_t tx_mbar = 4104;
    constexpr uint32_t tile = 2048;
    constexpr uint32_t descriptor_address = 6144;
    struct Observed {
        uint64_t accept = 0;  // the queued mma.async (command 3)
        uint64_t tma_end = 0; // end of the "tma command" slice
        std::vector<RmwService> services;
        uint64_t waits = 0;
    };
    const auto run = [&](uint32_t lead, std::optional<uint32_t> delay) {
        std::ostringstream qtx;
        // The TMA's operands (umov into ur10, ur11, ur8) are set after the
        // commit (with the fillers), so for one lead the commit's cycle r --
        // and its phase against the DRAM clock -- is the same in every run.
        qtx << "";
        for (uint32_t i = 0; i < lead; ++i) qtx << "mov r11, " << i << '\n';
        // ur20, never written, is 0: the matmuls' result at tmem 0.
        qtx << "umov ur0, 0\numov ur1, 1024\numov ur4, 4096\n"
            << "mma.async.init.m64n64k8 [ur20], [ur0], [ur1]\nmma.commit [ur4]\n";
        if (delay.has_value()) {
            qtx << tma_operands_asm(10, tile, 11, tx_mbar, 8);
            for (uint32_t i = 0; i < *delay; ++i) qtx << "mov r" << 12 + i % 8 << ", " << i << '\n';
            qtx << "tma.load.1d [ur10], [" << descriptor_address << ", ur8], [ur11]\n";
        }
        qtx << "mma.async.m64n64k8 [ur20], [ur0], [ur1]\nhalt\n";
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.test_shmem, 8192);
        auto& gmem = image_bytes(launch.gmem_image, 64);
        for (uint32_t i = 0; i < 16; ++i) write_u32(gmem, 4 * i, 0x100u + i);
        write_mbar(shmem, commit_mbar, 1, 0);
        write_mbar(shmem, tx_mbar, 0, 64);
        write_tma_descriptor(shmem, descriptor_address, plain_tma_1d(16));
        nanosm::Machine machine(canonical_profile(), compile(qtx.str()), std::move(launch),
            {.timeline = true});
        machine.run(5000);
        Observed observed;
        for (const auto& event : machine.trace().events()) {
            if (event.category == nanosm::TraceCategory::mma && event.name == "mma.async" &&
                event.id == 3) observed.accept = event.start_cycle;
            if (event.name == "tma command") observed.tma_end = event.start_cycle + event.duration_cycles;
        }
        observed.services = mbar_rmw_services(machine);
        observed.waits = machine.trace().summary().mbar_engine_waits_on_other;
        if (delay.has_value()) {
            require(machine.read_mbar_state(tx_mbar).phase &&
                    machine.read_shmem_u32(tile + 60) == 0x10fu, "the TMA load did not complete");
        }
        return observed;
    };
    const RmwService tma_service{0, "complete_tx (tma), address=4104"};
    const RmwService mma_service{0, "arrive (mma.commit), address=4096"};
    const auto at = [](RmwService service, uint64_t cycle) { service.first = cycle; return service; };

    bool aligned = false;
    for (uint32_t lead = 0; lead < 2 && !aligned; ++lead) {
        const auto reference = run(lead, std::nullopt); // no TMA
        // The commit presents in r; the mma.async behind it issues then and
        // reaches the queue at its dispatch (I6a).
        const auto r = reference.accept - nanosm::issue_to_dispatch(canonical_profile());
        require(reference.services == std::vector<RmwService>{at(mma_service, r + 1)},
            "reference commit was not served at r + 1");
        const auto early = run(lead, 0u); // the TMA completes long before r
        require(early.services.size() == 2 && early.services[0].second == tma_service.second &&
                early.services[0].first < r, "a delay-0 TMA did not complete before the commit");
        // The fillers issue one per cycle, so this delay lands the TMA near r.
        const auto estimate = static_cast<uint32_t>(r - early.services[0].first);
        for (uint32_t delay = estimate - 6; delay <= estimate + 6; ++delay) {
            const auto observed = run(lead, delay);
            const auto where = " (lead " + std::to_string(lead) + ", delay " + std::to_string(delay) + ")";
            require(observed.accept == reference.accept, "the TMA's completion delayed the MMA queue" + where);
            require(observed.services.size() == 2, "an engine event was lost" + where);
            const auto tma_claim = observed.services[0].second == tma_service.second
                ? observed.services[0].first : observed.services[1].first;
            require(observed.tma_end == tma_claim + 1,
                "the TMA command outlived its claimed event (held by the MMA's?)" + where);
            if (tma_claim != r + 1) continue; // the TMA presented in r too
            aligned = true;
            const auto busy = rmw_busy_cycles(); // I2: D + Lb + 1 (was 2); I2.1: + C
            require(observed.services ==
                    std::vector<RmwService>{at(tma_service, r + 1), at(mma_service, r + 1 + busy)},
                "same-cycle TMA completion and commit were not served TMA, then MMA" + where);
            require(observed.waits == busy, "the commit should wait out the TMA's RMW" + where);
        }
    }
    require(aligned, "no delay made the TMA completion and the commit present in one cycle");
}

void test_mma_dense_profile_recurrence_bubbles() {
    // D15: a shape with k > core_k must revisit each accumulator block late
    // enough to hide the accumulator latency, so a profile whose result
    // latency outruns one command's sweep is refused (m32n64k8 on a 4x4x4
    // core: 128 steps between visits). Recurrence bubbles remain for
    // back-to-back COMMANDS on one accumulator: the second command's first
    // step waits for the first command's result of the same block. Here
    // m32n64k8 on a 4x4x8 core (k = core_k: one visit per command), the
    // dense layout (B's k rows share a bank pair: about 8 rounds a step, so
    // one command's 128 steps take about 1100 cycles), a 4000-cycle result
    // latency, init then accumulate.
    auto too_slow = canonical_profile();
    too_slow.mma_shapes = {{32, 64, 8}};
    too_slow.mma_core_k = 4;
    too_slow.mma_layout = nanosm::TileLayout::dense;
    too_slow.mma_result_latency_cycles = 1000;
    try {
        nanosm::validate_device_profile(too_slow);
        throw std::runtime_error("a k > core_k shape that cannot hide its accumulator latency was accepted");
    } catch (const nanosm::ProfileError& error) {
        require_mentions(error.what(), {"m32n64k8", "revisited every 128 steps", "needs at least 1006"},
            "the latency rule's message");
    }
    auto profile = canonical_profile();
    profile.mma_shapes = {{32, 64, 8}};
    profile.mma_layout = nanosm::TileLayout::dense;
    profile.mma_result_latency_cycles = 4000;
    nanosm::validate_device_profile(profile);
    auto unsupported = profile;
    unsupported.mma_layout = nanosm::TileLayout::swizzle64;
    unsupported.mma_core_n = 2; // swizzle64 keeps a 4x4 block's rows apart: a 4x4x4 or 4x4x8 core only
    try {
        nanosm::validate_device_profile(unsupported);
        throw std::runtime_error("unsupported generalized XOR MMA layout was accepted");
    } catch (const nanosm::ProfileError&) {}
    const auto kernel = compile(R"(
        umov ur0, 0
        umov ur1, 1024
        umov ur4, 896
        mbar.init [ur4], 1
        umov ur8, 0
        umov ur9, 0
        mma.async.init.m32n64k8 [ur8], [ur0], [ur1]
        mma.async.m32n64k8 [ur8], [ur0], [ur1]
        mma.commit [ur4]
    wait_done:
        mbar.try_wait p0, [ur4], 0
        @!p0 bra.uni wait_done
        acc.ld r10, [ur8, 64]
        halt
    )", profile);
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.test_shmem, 2048);
    for (uint32_t m = 0; m < 32; ++m)
        for (uint32_t k = 0; k < 8; ++k) write_u16(shmem, 2 * (m * 8 + k), 0x3c00);
    for (uint32_t k = 0; k < 8; ++k)
        for (uint32_t n = 0; n < 64; ++n) write_u16(shmem, 1024 + 2 * (k * 64 + n), 0x3c00);
    nanosm::Machine machine(profile, kernel, std::move(launch), {.timeline = true});
    machine.run(40000);
    require(machine.read_reg(0, 0, 10) == nanosm::float_bits(16.0f),
        "dense diagnostic MMA profile produced the wrong result (8 + 8 ones)");
    require(machine.trace().summary().mma_recurrence_stalls > 0,
        "back-to-back commands on one accumulator did not expose their recurrence bubbles");
    require(machine.trace().summary().mma_shmem_rounds >
            machine.trace().summary().mma_input_steps,
        "dense diagnostic footprint did not charge its shmem bank conflicts");
}

void test_mma_bare_transfer_race_panics() {
    // The FAIL.asm shape (cases 01/02): three commands, a commit, and an
    // acc.ld with no wait on the commit mbar. Racing accepted MMA work is UB
    // and the simulator diagnoses it instead of returning stale columns.
    // (Result latency 4: the shortest that covers the C read, tmem port depth
    // 2 + read latency 2.)
    const auto profile = dense_one_bank_profile(4);
    const auto kernel = compile(R"(
        umov ur4, 14000
        mbar.init [ur4], 1
        umov ur0, 0
        umov ur1, 2048
        mma.async.init.m32n64k8 [ur8], [ur0], [ur1]   ; ur8 is 0 until written below: tmem 0
        umov ur0, 4096
        umov ur1, 6144
        mma.async.m32n64k8 [ur8], [ur0], [ur1]
        umov ur0, 8192
        umov ur1, 10240
        mma.async.m32n64k8 [ur8], [ur0], [ur1]
        mma.commit [ur4]
        umov ur8, 0
        umov ur9, 0
        acc.ld r10, [ur8, 64]
        halt
    )", profile);
    nanosm::Launch launch;
    launch.active_warps = 1;
    nanosm::Machine machine(profile, kernel, std::move(launch));
    try {
        machine.run(10000);
        throw std::runtime_error("bare commit -> acc.ld race was not diagnosed");
    } catch (const nanosm::SimError& error) {
        require(std::string(error.what()).find("races accepted MMA work") != std::string::npos,
            "the transfer/MMA race produced the wrong diagnostic");
    }
}

void test_mma_multi_round_fetch_timing() {
    // Case 03/05/06 (I9: the m32n64k8 shape): one bank makes every port-B
    // round one word. Steps run k block, then the 8 m blocks, then the 16 n
    // blocks; a step at n block 0 reloads A (eight A words + eight B words =
    // 16 rounds), every other step reads B only (8 rounds). A step granted its
    // first round at t is granted its last at t + rounds - 1; that round
    // reaches the RAM D later and the port output D + Lb later, where the step
    // enters the core (the port-B read pipeline has port A's D and Lb); its
    // result leaves the core at entry + latency and lands in tmem
    // tmem.port_depth later (I8); the next step's first round is granted at
    // t + rounds; and the operand-consumed depbar clears at the LAST PHYSICAL
    // READ of the final step: its last round's RAM sample.
    const auto profile = dense_one_bank_profile(12);
    const std::string source = R"(
        umov ur0, 0
        umov ur1, 1024
        umov ur4, 896
        mbar.init [ur4], 1
        umov ur8, 0
        umov ur9, 0
        mma.async.init.m32n64k8 [ur8], [ur0], [ur1]
        mma.commit [ur4]
    wait_done:
        mbar.try_wait p0, [ur4], 0
        @!p0 bra.uni wait_done
        acc.ld r10, [ur8, 64]
        halt
    )";
    const auto scheduled = nanosm::compile_qtx(source, profile);
    require(scheduled.instructions[6].opcode == nanosm::mma_opcode(0, true), // m32n64k8 (its only shape), init
        "multi-round timing test instruction index changed unexpectedly");
    const auto bar = scheduled.instructions[6].control.set_depbar;
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.test_shmem, 2048);
    for (uint32_t m = 0; m < 32; ++m)
        for (uint32_t k = 0; k < 8; ++k) write_u16(shmem, 2 * (m * 8 + k), 0x3c00);
    for (uint32_t k = 0; k < 8; ++k)
        for (uint32_t n = 0; n < 64; ++n) write_u16(shmem, 1024 + 2 * (k * 64 + n), 0x3c00);
    nanosm::Machine machine(profile,
        nanosm::assemble_asm(nanosm::emit_asm(scheduled), profile),
        std::move(launch), {.timeline = true, .debug = true});
    machine.run(8000);
    std::vector<uint64_t> inputs;
    uint64_t first_result = 0;
    uint64_t depbar_cleared = 0;
    for (const auto& event : machine.trace().events()) {
        if (event.name == "MMA input step") inputs.push_back(event.start_cycle);
        if (event.name == "MMA tmem write" && first_result == 0) first_result = event.start_cycle;
        if (depbar_cleared == 0 && event.name == "depbar sb" + std::to_string(bar) &&
            event.detail == "value=0") {
            depbar_cleared = event.start_cycle;
        }
    }
    // 8 m blocks x 16 n blocks x 2 k blocks; a step's rounds: 16 at n block 0.
    const auto rounds = [](std::size_t step) -> uint64_t { return step % 16 == 0 ? 16 : 8; };
    require(inputs.size() == 256, "one dense m32n64k8 command on a 4x4x4 core is 256 core steps");
    for (std::size_t i = 1; i < inputs.size(); ++i)
        require(inputs[i] == inputs[i - 1] + rounds(i - 1),
            "step " + std::to_string(i) + ": the fetches did not serialize the port at one round per cycle");
    const auto d = profile.shmem_port_a_depth;
    const auto lb = profile.shmem_read_latency_cycles;
    require(first_result == inputs[0] + 15 + d + lb + 12 + profile.tmem_port_depth,
        "the result did not land at core entry (last round + D + Lb) plus the pipeline latency plus the "
        "tmem port depth");
    require(depbar_cleared == inputs[255] + rounds(255) - 1 + d,
        "operand-consumed depbar did not clear at the final physical read (the last round's sample)");
    require(machine.trace().summary().mma_recurrence_stalls == 0,
        "a tile is revisited 128 steps (over a thousand rounds) later; no bubbles expected");
    for (uint32_t lane = 0; lane < profile.lanes; ++lane)
        require(machine.read_reg(0, lane, 10) == nanosm::float_bits(8.0f),
            "per-round operand capture changed the numerical result");
}

void test_mma_varying_round_service() {
    // Unswizzled dense 64x64x8 on 16 banks has steps of different conflict
    // costs (the 2026-09-04 audit counted four- and five-round steps). Each
    // fetch must occupy the port for exactly its own rounds: the span of all
    // input steps equals the sum of served rounds plus counted stalls.
    auto profile = canonical_profile();
    profile.mma_layout = nanosm::TileLayout::dense;
    nanosm::validate_device_profile(profile);
    const auto kernel = compile(R"(
        umov ur0, 0
        umov ur1, 2048
        umov ur4, 8192
        mbar.init [ur4], 1
        umov ur8, 0
        umov ur9, 0
        mma.async.init.m64n64k8 [ur8], [ur0], [ur1]
        mma.commit [ur4]
    wait_done:
        mbar.try_wait p0, [ur4], 0
        @!p0 bra.uni wait_done
        acc.ld r10, [ur8, 64]
        halt
    )", profile);
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.test_shmem, 4096);
    for (uint32_t i = 0; i < 512; ++i) write_u16(shmem, 2 * i, 0x3c00);
    for (uint32_t i = 0; i < 512; ++i) write_u16(shmem, 2048 + 2 * i, 0x3c00);
    nanosm::Machine machine(profile, kernel, std::move(launch), {.timeline = true});
    machine.run(8000);
    std::vector<std::pair<uint64_t, uint64_t>> inputs; // cycle, rounds
    for (const auto& event : machine.trace().events()) {
        if (event.name != "MMA input step") continue;
        inputs.emplace_back(event.start_cycle, std::stoull(event.detail.substr(7)));
    }
    require(inputs.size() == 256, "one canonical-shape command is 256 core steps (4x4x8 core)");
    uint64_t served = 0;
    std::set<uint64_t> distinct;
    for (std::size_t i = 0; i + 1 < inputs.size(); ++i) {
        require(inputs[i + 1].first - inputs[i].first >= inputs[i].second,
            "a fetch began before the previous step's rounds freed the port");
        served += inputs[i].second;
        distinct.insert(inputs[i].second);
    }
    distinct.insert(inputs.back().second);
    require(distinct.size() >= 2, "the dense walk was expected to vary its conflict rounds");
    require(inputs.back().first - inputs.front().first ==
            served + machine.trace().summary().mma_recurrence_stalls,
        "input-step spacing did not equal served rounds plus counted stalls");
    for (uint32_t lane = 0; lane < profile.lanes; ++lane)
        require(machine.read_reg(0, lane, 10) == nanosm::float_bits(8.0f),
            "varying-round service changed the numerical result");
}

// ---- I7: tensor core 4x4x8, hold A / stream B (trash/rfc11.md section 0) ----

// Canonical swizzle64 operand addresses at the default row spacing (16 bytes:
// a K = 8 tile), restated independently of the production helper: A element
// (m, k) and B element (k, n) of 64x64x8 tiles at 0 and 1024 (one rule: B is
// stored as n rows of k, as A is m rows of k).
uint32_t i7_a_address(uint32_t m, uint32_t k) {
    return 4 * (4 * m + ((k >> 1) ^ ((m >> 2) & 3))) + 2 * (k & 1);
}

uint32_t i7_b_address(uint32_t k, uint32_t n) {
    return 1024 + 4 * (4 * n + ((k >> 1) ^ ((n >> 2) & 3))) + 2 * (k & 1);
}

// One 64x64x8 tile pair (A = a_k on every row, B = b_k on every column) and
// `commands` MMA commands into one accumulator, then commit, wait, acc.ld.
std::unique_ptr<nanosm::Machine> i7_machine(const std::array<uint16_t, 8>& a,
    const std::array<uint16_t, 8>& b, uint32_t commands) {
    const auto& profile = canonical_profile();
    std::ostringstream qtx;
    qtx << "umov ur0, 0\numov ur1, 1024\numov ur4, 4096\numov ur8, 0\numov ur9, 0\n"
        << "mbar.init [ur4], 1\nmma.async.init.m64n64k8 [ur8], [ur0], [ur1]\n";
    for (uint32_t command = 1; command < commands; ++command) qtx << "mma.async.m64n64k8 [ur8], [ur0], [ur1]\n";
    qtx << "mma.commit [ur4]\nwait:\nmbar.try_wait p0, [ur4], 0\n@!p0 bra.uni wait\n"
        << "acc.ld r10, [ur8, 64]\nhalt\n";
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.test_shmem, 8192);
    for (uint32_t m = 0; m < 64; ++m)
        for (uint32_t k = 0; k < 8; ++k) write_u16(shmem, i7_a_address(m, k), a[k]);
    for (uint32_t k = 0; k < 8; ++k)
        for (uint32_t n = 0; n < 64; ++n) write_u16(shmem, i7_b_address(k, n), b[k]);
    return std::make_unique<nanosm::Machine>(profile, compile(qtx.str(), profile), std::move(launch),
        nanosm::TraceOptions{.timeline = true, .debug = true});
}

// A 4x4x8 step is ONE fused group of eight products plus C (rfc11 section 0),
// not two successive four-product groups. These eight products tell them
// apart: 4096 * 4096 + 1 * 1 is 2^24 + 1, which a four-product group
// truncates to 2^24 in FP32 before the second group's -2^24 cancels it (0);
// one eight-product group keeps the unit (1.0).
void test_i7_dot8_group_boundary() {
    const std::array<uint16_t, 8> a{0x6c00, 0x3c00, 0, 0, 0xec00, 0, 0, 0}; // 4096, 1, 0, 0, -4096, 0...
    const std::array<uint16_t, 8> b{0x6c00, 0x3c00, 0, 0, 0x6c00, 0, 0, 0}; // 4096, 1, 0, 0, 4096, 0...
    const auto eight = nanosm::mma_dot_f16(a, b, 0);
    const auto first_four = nanosm::mma_dot_f16(std::span(a).first(4), std::span(b).first(4), 0);
    const auto two_fours = nanosm::mma_dot_f16(std::span(a).last(4), std::span(b).last(4), first_four);
    require(eight == 0x3f800000u && first_four == 0x4b800000u && two_fours == 0,
        "the eight-product group and two four-product groups do not differ as derived");
    require(canonical_profile().mma_core_k == 8, "the canonical MMA core is not 4x4x8");
    auto machine = i7_machine(a, b, 1);
    machine->run(20000);
    for (uint32_t lane = 0; lane < 16; ++lane)
        require(machine->read_reg(0, lane, 10) == 0x3f800000u,
            "the canonical 4x4x8 core did not compute one eight-product group");
}

// One canonical command, step by step. Hold A / stream B: for each of the
// 16 A blocks (m0 = 4i), every B block (n0 = 4j) in order; the step with
// j = 0 reads its A block and its B block (2 bank rounds), the other 15 read
// one B block (1 round) and use the held A block. Port B grants one round
// per cycle and a step its first round the cycle after the previous step's
// last, so step s + 1 starts rounds(s) cycles after step s. A step whose
// last round is granted at g enters the core at g + D + Lb; its result
// leaves the core result_latency later and is written at the tmem RAM
// tmem.port_depth after that (I8).
void test_i7_hold_a_stream_b_walker() {
    const auto& profile = canonical_profile();
    std::array<uint16_t, 8> a{}, b{};
    a.fill(0x3c00); // 1
    b.fill(0x4000); // 2
    auto machine = i7_machine(a, b, 1);
    machine->run(20000);
    for (uint32_t lane = 0; lane < 16; ++lane)
        require(machine->read_reg(0, lane, 10) == nanosm::float_bits(16.0f),
            "hold-A walker computed the wrong value");
    std::vector<const nanosm::TimelineEvent*> inputs, writes;
    for (const auto& event : machine->trace().events()) {
        if (event.name == "MMA input step") inputs.push_back(&event);
        if (event.name == "MMA tmem write") writes.push_back(&event);
    }
    require(inputs.size() == 256 && writes.size() == 256, "one 64x64x8 command is 256 4x4x8 steps");
    const auto d = profile.shmem_port_a_depth;
    const auto lb = profile.shmem_read_latency_cycles;
    for (uint32_t s = 0; s < 256; ++s) {
        const bool reads_a = s % 16 == 0;
        const uint32_t rounds = reads_a ? 2 : 1;
        require(static_cast<uint32_t>(inputs[s]->value) == s &&
                inputs[s]->detail == (reads_a ? "rounds=2, reads A" : "rounds=1"),
            "step " + std::to_string(s) + " read its A block at the wrong step: " + inputs[s]->detail);
        if (s + 1 < 256)
            require(inputs[s + 1]->start_cycle == inputs[s]->start_cycle + rounds,
                "step " + std::to_string(s + 1) + " did not start the cycle after the previous step's last round");
        const uint32_t tile = 64 * (4 * (s / 16)) + 4 * (s % 16); // m0 * 64 + n0, n fastest
        require(static_cast<uint32_t>(writes[s]->value) == tile,
            "step " + std::to_string(s) + " wrote tile " + std::to_string(writes[s]->value) +
                ", not the hold-A / stream-B order's " + std::to_string(tile));
        require(writes[s]->start_cycle == inputs[s]->start_cycle + (rounds - 1) + d + lb +
                profile.mma_result_latency_cycles + profile.tmem_port_depth,
            "step " + std::to_string(s) + ": tmem write is not last round + D + Lb + result latency + "
            "tmem port depth");
    }
    const auto& summary = machine->trace().summary();
    require(summary.mma_a_block_reads == 16 && summary.mma_shmem_rounds == 16 * 17,
        "one command did not read 16 A blocks in 16 x 17 bank rounds");
}

// Back-to-back commands into one accumulator: the next command's first step
// is granted the cycle after the previous command's last round, so N
// commands take 272 N rounds = 272 N cycles of port B from the first input
// step: 256 steps per 272 cycles, 16/17 of one step per cycle (the rfc11
// section 0 ceiling, here measured, with no recurrence or transfer stalls).
void test_i7_back_to_back_throughput() {
    constexpr uint32_t commands = 8;
    std::array<uint16_t, 8> a{}, b{};
    a.fill(0x3c00);
    b.fill(0x3c00);
    auto machine = i7_machine(a, b, commands);
    machine->run(100000);
    for (uint32_t lane = 0; lane < 16; ++lane)
        require(machine->read_reg(0, lane, 10) == nanosm::float_bits(8.0f * commands),
            "back-to-back commands accumulated the wrong value");
    uint64_t first = std::numeric_limits<uint64_t>::max(), last = 0, steps = 0;
    for (const auto& event : machine->trace().events()) {
        if (event.name != "MMA input step") continue;
        first = std::min(first, event.start_cycle);
        last = std::max(last, event.start_cycle);
        ++steps;
    }
    require(steps == 256 * commands, "back-to-back commands lost steps");
    // The last step reads one round, so port B is busy through `last`.
    require(last - first + 1 == 272 * commands,
        "back-to-back commands did not keep port B busy one round per cycle: " +
            std::to_string(last - first + 1) + " cycles for " + std::to_string(272 * commands) + " rounds");
    require(machine->trace().summary().mma_recurrence_stalls == 0, "back-to-back commands hit recurrence stalls");
}

// ---- I9: MMA shapes as opcodes, the profile's shape list, the 32x64 tile ----
// (trash/chan.txt 2026-10-19; trash/i9-32x64-tile-proposal-fable-2026-10-19.md)

// Independent restatements (not the production helpers): swizzle64 operand
// element addresses at the default 16-byte row spacing for objects at any
// 64-byte-aligned base (the word XOR is the address's own row number / 4,
// base / 16 + m: rows past 31 simply continue, so a 64-row object is two
// stacked 32-row ones), the tmem word of an accumulator element (16 fine banks, 4x4
// macro-tiles row-major), and small exact FP16 integers.
uint32_t i9_a_address(uint32_t base, uint32_t m, uint32_t k) {
    return base + 4 * (4 * m + ((k >> 1) ^ (((base / 16 + m) >> 2) & 3))) + 2 * (k & 1);
}

uint32_t i9_b_address(uint32_t base, uint32_t k, uint32_t n) {
    return base + 4 * (4 * n + ((k >> 1) ^ (((base / 16 + n) >> 2) & 3))) + 2 * (k & 1);
}

// Element (row, col) of the tmem region at `pointer` whose rows are row_words
// words (src/layout.hpp, restated): keyed by the logical address's absolute
// row R and column C -- 4x4 macro tiles row-major, inside one the word at 4 x
// (R % 4) + (C % 4 XOR the band R / 4 mod 4).
uint32_t i9_tmem_word(uint32_t pointer, uint32_t row_words, uint32_t row, uint32_t col) {
    const uint32_t logical = pointer + row * row_words + col;
    const uint32_t r = logical / row_words;
    const uint32_t c = logical % row_words;
    return ((r / 4) * (row_words / 4) + c / 4) * 16 + 4 * (r % 4) + ((c % 4) ^ ((r / 4) % 4));
}

uint16_t i9_fp16(uint32_t value) {
    static constexpr std::array<uint16_t, 5> bits{0x0000, 0x3c00, 0x4000, 0x4200, 0x4400}; // 0..4
    return bits.at(value);
}

// A(m, k) and B(k, n): small integers, so every dot product is exact in FP32.
uint32_t i9_a(uint32_t m, uint32_t k) { return (m + 3 * k) % 5; }

uint32_t i9_b(uint32_t k, uint32_t n) { return (2 * k + n) % 3; }

float i9_reference(uint32_t m, uint32_t n) {
    float sum = 0.0f;
    for (uint32_t k = 0; k < 8; ++k) sum += static_cast<float>(i9_a(m, k) * i9_b(k, n));
    return sum;
}

// A 64-row A object at 0 and a B object at 2048, both from i9_a / i9_b.
void i9_write_operands(nanosm::Launch& launch) {
    auto& shmem = image_bytes(launch.test_shmem, 4096);
    for (uint32_t m = 0; m < 64; ++m)
        for (uint32_t k = 0; k < 8; ++k) write_u16(shmem, i9_a_address(0, m, k), i9_fp16(i9_a(m, k)));
    for (uint32_t k = 0; k < 8; ++k)
        for (uint32_t n = 0; n < 64; ++n) write_u16(shmem, i9_b_address(2048, k, n), i9_fp16(i9_b(k, n)));
}

// D15 (trash/chan.txt 2026-10-22): the mma.async block. Shape i of the
// profile's list is opcode 0x70 + 2i (accumulate) and 0x70 + 2i + 1 (init);
// names follow the profile, and without one an opcode is named by its place
// in the block. A lossless encoding; the text qtxc and the hand assembler
// parse; shape names parsed strictly; the shape-less spellings are not
// instructions.
void test_i9_mma_opcodes() {
    const auto& profile = canonical_profile();
    struct Expected {
        uint32_t index;
        bool init;
        uint8_t number;
        const char* name;
        nanosm::MmaShape shape;
    };
    const std::array<Expected, 4> expected{{
        {0, false, 0x70, "mma.async.m64n64k8", {64, 64, 8}},
        {0, true, 0x71, "mma.async.init.m64n64k8", {64, 64, 8}},
        {1, false, 0x72, "mma.async.m32n64k8", {32, 64, 8}},
        {1, true, 0x73, "mma.async.init.m32n64k8", {32, 64, 8}},
    }};
    for (const auto& e : expected) {
        const auto opcode = nanosm::mma_opcode(e.index, e.init);
        const auto slot = std::string(e.init ? "mma.async.init.slot" : "mma.async.slot") + std::to_string(e.index);
        require(static_cast<uint8_t>(opcode) == e.number && nanosm::is_mma_async(opcode) &&
                nanosm::is_mma_init(opcode) == e.init && nanosm::mma_shape_index(opcode) == e.index &&
                nanosm::mma_shape_of(profile, opcode) == e.shape &&
                nanosm::opcode_name(opcode, &profile.mma_shapes) == e.name && nanosm::opcode_name(opcode) == slot,
            std::string(e.name) + ": opcode, init, shape or name");
        nanosm::Instruction inst;
        inst.opcode = opcode;
        inst.src2 = 3; // urC, the result's tmem pointer
        inst.src0 = 7;
        inst.src1 = 11;
        inst.imm = nanosm::mma_spacing_immediate(16, 16);
        inst.control.set_depbar = 2;
        const auto decoded = nanosm::decode(nanosm::encode(inst));
        require(decoded.opcode == opcode && decoded.src2 == 3 && decoded.src0 == 7 && decoded.src1 == 11 &&
                nanosm::uniform_sources(decoded) == std::vector<uint8_t>{7, 11, 3} &&
                nanosm::format_instruction(decoded, false, &profile.mma_shapes) ==
                    std::string(e.name) + " [ur3], [ur7], [ur11]",
            std::string(e.name) + ": encode / decode / text: " +
                nanosm::format_instruction(decoded, false, &profile.mma_shapes));
        // The row spacings (16-byte units in the 14 spare bits): printed when
        // not the default 16, and the largest one fits.
        inst.imm = nanosm::mma_spacing_immediate(128, nanosm::mma_max_spacing);
        const auto spaced = nanosm::decode(nanosm::encode(inst));
        require(nanosm::mma_row_spacing(spaced, true) == 128 &&
                nanosm::mma_row_spacing(spaced, false) == nanosm::mma_max_spacing &&
                nanosm::format_instruction(spaced, false, &profile.mma_shapes) ==
                    std::string(e.name) + " [ur3], [ur7, 128], [ur11, 2032]",
            std::string(e.name) + ": row spacings did not survive encode / decode / text: " +
                nanosm::format_instruction(spaced, false, &profile.mma_shapes));
    }
    // The assembly operand: [urA] is spacing 16; [urA, spacing] names it. A
    // spacing that is not a multiple of 16 or past the field is refused at
    // assembly; one the layout does not take (or shorter than the row a
    // shape reads) at assembly and kernel load (check_mma_shape).
    const auto parsed = nanosm::compile_qtx("mma.async.m64n64k8 [ur8], [ur0, 128], [ur1]\nhalt\n",
        profile);
    require(nanosm::mma_row_spacing(parsed.instructions.at(0), true) == 128 &&
            nanosm::mma_row_spacing(parsed.instructions.at(0), false) == 16,
        "[ur0, 128], [ur1] did not parse to spacings 128 and 16");
    for (const auto& [operand, needle] : {std::pair{"[ur0, 24]", "multiple of 16"},
             std::pair{"[ur0, 2048]", "multiple of 16"}, std::pair{"[ur0, 48]", "a power of two"},
             std::pair{"[ur0, 192]", "a power of two"}, std::pair{"[ur0, 1, 2]", "bad mma operand"}}) {
        try {
            (void)nanosm::compile_qtx(std::string("mma.async.m64n64k8 [ur8], ") + operand +
                ", [ur1]\nhalt\n", profile);
            throw std::runtime_error(std::string("the mma operand ") + operand + " was accepted");
        } catch (const nanosm::SimError& error) {
            require_mentions(error.what(), {needle}, std::string("the mma operand ") + operand);
        }
    }
    require(nanosm::is_mma_async(static_cast<nanosm::Opcode>(0x70)) &&
            nanosm::is_mma_async(static_cast<nanosm::Opcode>(0x7f)) &&
            !nanosm::is_mma_async(static_cast<nanosm::Opcode>(0x6f)) &&
            !nanosm::is_mma_async(static_cast<nanosm::Opcode>(0x80)) &&
            !nanosm::is_mma_async(nanosm::Opcode::mma_commit),
        "the mma.async block is 0x70..0x7f, and mma.commit is not in it");
    require(!nanosm::mma_shape_of(profile, nanosm::mma_opcode(2, false)).has_value() &&
            !nanosm::mma_shape_of(profile, nanosm::Opcode::mma_commit).has_value(),
        "an opcode past the profile's list, or outside the block, has a shape");
    require(nanosm::parse_mma_shape_name("m128n32k16") == nanosm::MmaShape{128, 32, 16} &&
            !nanosm::parse_mma_shape_name("m64n64").has_value() &&
            !nanosm::parse_mma_shape_name("m0n64k8").has_value() &&
            !nanosm::parse_mma_shape_name("m64n64k8x").has_value() &&
            !nanosm::parse_mma_shape_name("M64N64K8").has_value() &&
            !nanosm::parse_mma_shape_name("m-4n64k8").has_value(),
        "shape names are parsed strictly");
    for (const auto& e : expected) {
        const auto rows = e.shape.m;
        const auto source = "umov ur0, 0\numov ur1, 2048\numov ur2, " + std::to_string(64 * (64 - rows)) + "\n" +
            std::string(e.name) + " [ur2], [ur0], [ur1]\nhalt\n";
        const auto scheduled = nanosm::compile_qtx(source, profile);
        const auto kernel = nanosm::assemble_asm(nanosm::emit_asm(scheduled), profile);
        require(nanosm::decode(kernel.words.at(3)).opcode == nanosm::mma_opcode(e.index, e.init),
            std::string(e.name) + ": qtxc and the hand assembler did not produce the opcode");
        require(nanosm::emit_hex(kernel, &profile.mma_shapes).find(std::string(e.name) + " [ur2], [ur0], [ur1]") !=
                std::string::npos,
            std::string(e.name) + ": the hex comment does not name the shape and the operands");
    }
    for (const auto* old : {"mma.async C, [ur0], [ur1]", "mma.async.init C, [ur0], [ur1]"}) {
        try {
            (void)nanosm::compile_qtx(std::string("umov ur0, 0\numov ur1, 2048\n") + old + "\nhalt\n",
                profile);
            throw std::runtime_error(std::string("the shape-less '") + old + "' was accepted");
        } catch (const nanosm::SimError&) {}
    }
}

// The profile lists its MMA shapes by name (D15: any shape that follows the
// rules, each once): m, n, k whole multiples of the core; an accumulator of
// at least `lanes` rows that fits in tmem; k > core_k only with enough
// blocks per K block to hide the accumulator latency; at most eight shapes
// (the opcode block). The list is fingerprinted, so its ORDER (which assigns
// the opcodes) is too. A kernel using a shape or an opcode the profile does
// not define is refused by qtxc, by the hand assembler and at kernel load;
// the result operand is a tmem pointer alone, [urC].
void test_i9_profile_shapes_and_refusals() {
    const auto& profile = canonical_profile();
    require(profile.mma_shapes == std::vector<nanosm::MmaShape>{{64, 64, 8}, {32, 64, 8}},
        "the canonical profile does not list m64n64k8 and m32n64k8");
    std::ifstream input("profiles/nanosm-v0.json");
    nlohmann::json document;
    input >> document;
    const auto refused = [&](const nlohmann::json& shapes, const std::string& what) {
        auto changed = document;
        if (shapes.is_null()) changed["mma"].erase("shapes");
        else changed["mma"]["shapes"] = shapes;
        try {
            (void)nanosm::parse_device_profile(changed.dump());
            throw std::runtime_error(what + " was accepted");
        } catch (const nanosm::ProfileError&) {}
    };
    refused(nullptr, "a profile without mma.shapes");
    refused(nlohmann::json::array(), "an empty shape list");
    refused("m64n64k8", "a shape name that is not a list");
    refused(nlohmann::json::array({"m64n64k8", "m64n64k8"}), "a shape listed twice");
    refused(nlohmann::json::array({64}), "a shape that is not a name");
    refused(nlohmann::json::array({"m64n64"}), "a malformed shape name");
    const auto rule = [&](const char* shape, const std::string& expected, const std::string& what) {
        auto changed = document;
        changed["mma"]["shapes"] = nlohmann::json::array({"m64n64k8", shape});
        try {
            (void)nanosm::parse_device_profile(changed.dump());
            throw std::runtime_error(what + " was accepted");
        } catch (const nanosm::ProfileError& error) {
            require_mentions(error.what(), {std::string("mma shape ") + shape, expected}, what);
        }
    };
    // Every dimension is a power of two (HUMAN 2026-10-22), checked first.
    rule("m6n64k8", "m, n and k must be powers of two (m is 6)", "an m that is not a power of two");
    rule("m48n64k8", "m, n and k must be powers of two (m is 48)", "m = 48");
    rule("m64n12k8", "m, n and k must be powers of two (n is 12)", "n = 12");
    rule("m64n64k12", "m, n and k must be powers of two (k is 12)", "k = 12");
    // Powers of two that are not whole core blocks.
    rule("m64n2k8", "multiples of the core's 4x4x8", "an n smaller than core_n");
    rule("m64n64k4", "multiples of the core's 4x4x8", "a k smaller than core_k");
    rule("m8n64k8", "an accumulator needs at least 16", "an accumulator smaller than a lane each");
    rule("m128n128k8", "does not fit in tmem", "an accumulator larger than tmem");
    // The latency rule at canonical timing: D 2 + Lb 2 + result latency 8 +
    // tmem port depth 2 = 14 steps between visits of one accumulator block.
    require(nanosm::mma_min_blocks_per_k(profile) == 14, "the canonical latency bound is 2 + 2 + 8 + 2");
    rule("m16n8k16", "revisited every 8 steps", "a k > core_k shape revisiting its blocks too soon");
    // (With every dimension a power of two, the blocks per K block come in
    // powers of two too: 8 is short of 14, 16 is enough.)
    rule("m16n12k16", "m, n and k must be powers of two (n is 12)", "a k > core_k shape with n = 12");
    auto nine = document;
    nine["mma"]["shapes"] = nlohmann::json::array();
    for (const char* shape : {"m16n4k8", "m16n8k8", "m16n16k8", "m16n32k8", "m16n64k8", "m16n128k8",
             "m32n4k8", "m32n8k8", "m32n16k8"})
        nine["mma"]["shapes"].push_back(shape);
    try {
        (void)nanosm::parse_device_profile(nine.dump());
        throw std::runtime_error("nine shapes were accepted");
    } catch (const nanosm::ProfileError& error) {
        require_mentions(error.what(), {"lists 9 shapes", "holds 8"}, "nine shapes");
    }
    auto wider = document;
    wider["mma"]["shapes"] = nlohmann::json::array({"m64n64k8", "m16n16k16", "m16n64k32", "m128n64k8"});
    const auto accepted = nanosm::parse_device_profile(wider.dump()); // 16 blocks per K block >= 14
    require(accepted.mma_shapes.size() == 4 && accepted.mma_shapes[1] == nanosm::MmaShape{16, 16, 16},
        "legal shapes (k > core_k with 16 and 64 blocks per K block, a 128-row shape) were refused");
    auto only_m64 = document;
    only_m64["mma"]["shapes"] = nlohmann::json::array({"m64n64k8"});
    const auto restricted = nanosm::parse_device_profile(only_m64.dump());
    require(restricted.mma_shapes == std::vector<nanosm::MmaShape>{{64, 64, 8}} &&
            nanosm::device_profile_fingerprint(restricted) != nanosm::device_profile_fingerprint(profile),
        "the shape list is not parsed or not fingerprinted");
    auto odd_core = profile;
    odd_core.mma_layout = nanosm::TileLayout::dense;
    odd_core.mma_core_m = 3; // 32 and 64 rows are not whole 3-row blocks
    try {
        nanosm::validate_device_profile(odd_core);
        throw std::runtime_error("a shape that is not whole core blocks was accepted");
    } catch (const nanosm::ProfileError&) {}

    const std::string m32 =
        "umov ur0, 0\numov ur1, 2048\numov ur2, 4096\nmma.async.init.m32n64k8 [ur2], [ur0], [ur1]\nhalt\n";
    // qtxc: the profile does not list m32n64k8.
    try {
        (void)nanosm::compile_qtx(m32, restricted);
        throw std::runtime_error("qtxc accepted an unsupported shape");
    } catch (const nanosm::SimError& error) {
        require_mentions(error.what(), {"line 4", "mma.async.init.m32n64k8: shape m32n64k8 is not one of this "
            "device profile's mma shapes"}, "qtxc unsupported shape");
    }
    // The hand assembler: an m64 kernel assembled for the restricted profile,
    // its mnemonic changed to m32 in the text.
    auto text = nanosm::emit_asm(nanosm::compile_qtx(
        "umov ur0, 0\numov ur1, 2048\nmma.async.init.m64n64k8 [ur8], [ur0], [ur1]\nhalt\n", restricted));
    text.replace(text.find("mma.async.init.m64n64k8"), 23, "mma.async.init.m32n64k8");
    try {
        (void)nanosm::assemble_asm(text, restricted);
        throw std::runtime_error("the hand assembler accepted an unsupported shape");
    } catch (const nanosm::SimError& error) {
        require_mentions(error.what(), {"shape m32n64k8 is not one of this device profile's mma shapes"},
            "hand assembler unsupported shape");
    }
    // Kernel load: a canonical m32 kernel relabelled for the restricted profile.
    auto relabelled = compile(m32, profile);
    relabelled.profile_fingerprint = nanosm::device_profile_fingerprint(restricted);
    try {
        nanosm::Launch launch;
        launch.active_warps = 1;
        nanosm::Machine machine(restricted, relabelled, std::move(launch));
        throw std::runtime_error("kernel load accepted an unsupported shape");
    } catch (const nanosm::SimError& error) {
        require_mentions(error.what(), {"kernel pc 24",
            "mma.async.init.slot1: this device profile defines no mma shape for it (its list has 1 shapes)"},
            "kernel load: an opcode past the profile's shape list");
    }
    // The shape list's order assigns the opcodes and is fingerprinted: a
    // kernel built for one order does not load on a profile with another.
    auto reordered = document;
    reordered["mma"]["shapes"] = nlohmann::json::array({"m32n64k8", "m64n64k8"});
    const auto swapped = nanosm::parse_device_profile(reordered.dump());
    require(nanosm::device_profile_fingerprint(swapped) != nanosm::device_profile_fingerprint(profile),
        "the shape list's order is not fingerprinted");
    try {
        nanosm::Launch launch;
        launch.active_warps = 1;
        nanosm::Machine machine(swapped, compile(m32, profile), std::move(launch));
        throw std::runtime_error("a kernel built for one shape order loaded on another");
    } catch (const nanosm::SimError& error) {
        require_mentions(error.what(), {"does not match simulator"}, "reordered shape list at load");
    }
    // The result operand is a pointer alone (its rows are the shape's n
    // words): a row length, a vector register or a name is refused. There is
    // no accumulator to compare a shape with (2026-11-19).
    for (const auto& [result, needle] : {std::pair{"[ur2, 64]", "without a row length"},
             std::pair{"[r2]", "a uniform register holding its tmem pointer"},
             std::pair{"T", "a uniform register holding its tmem pointer"}}) {
        try {
            (void)nanosm::compile_qtx(std::string("mma.async.init.m32n64k8 ") + result + ", [ur0], [ur1]\nhalt\n",
                profile);
            throw std::runtime_error(std::string("the mma result ") + result + " was accepted");
        } catch (const nanosm::SimError& error) {
            require_mentions(error.what(), {needle}, std::string("the mma result ") + result);
        }
    }
}

// The full bank census for both shapes, from the restated layout: at every
// 64-byte-aligned A base in the first 4 KiB, a step's A block (4 rows x 8 k)
// and B block (8 k x 4 columns) are each 16 distinct words in 16 distinct
// banks, together two rounds; so a step that reloads A is 2 rounds and every
// other step 1, for m32 exactly as for m64 (no layout change needed).
void test_i9_bank_census() {
    for (const uint32_t rows : {32u, 64u}) {
        for (uint32_t a_base = 0; a_base < 4096; a_base += 64) {
            for (uint32_t m0 = 0; m0 < rows; m0 += 4) {
                for (uint32_t n0 = 0; n0 < 64; n0 += 4) {
                    std::map<uint32_t, std::set<uint32_t>> words_by_bank;
                    const bool reload = n0 == 0;
                    if (reload) {
                        for (uint32_t m = m0; m < m0 + 4; ++m)
                            for (uint32_t k = 0; k < 8; ++k) {
                                const auto word = i9_a_address(a_base, m, k) / 4;
                                words_by_bank[word % 16].insert(word);
                            }
                    }
                    for (uint32_t k = 0; k < 8; ++k)
                        for (uint32_t n = n0; n < n0 + 4; ++n) {
                            const auto word = i9_b_address(8192, k, n) / 4;
                            words_by_bank[word % 16].insert(word);
                        }
                    std::size_t rounds = 0;
                    for (const auto& [bank, words] : words_by_bank) rounds = std::max(rounds, words.size());
                    require(words_by_bank.size() == 16 && rounds == (reload ? 2u : 1u),
                        "census: m" + std::to_string(rows) + " A base " + std::to_string(a_base) + " m0 " +
                            std::to_string(m0) + " n0 " + std::to_string(n0) + " takes " +
                            std::to_string(rounds) + " rounds");
                }
            }
        }
    }
}

// One m32n64k8 command on the BOTTOM half of a 64-row A object (base 512):
// 8 A blocks x 16 B blocks = 128 steps, the step at n block 0 reading A and B
// (2 rounds), the other 15 B only (1 round); 8 A reloads and 136 port-B
// rounds; tiles in hold-A order; each result written at last round + D + Lb
// + result latency + tmem port depth; every element equal to the logical
// reference of rows 32..63.
void test_i9_m32_walker() {
    const auto& profile = canonical_profile();
    nanosm::Launch launch;
    launch.active_warps = 1;
    i9_write_operands(launch);
    nanosm::Machine machine(profile, compile(R"(
        umov ur0, 512
        umov ur1, 2048
        umov ur4, 8192
        umov ur2, 4096
        mbar.init [ur4], 1
        mma.async.init.m32n64k8 [ur2], [ur0], [ur1]
        mma.commit [ur4]
    wait:
        mbar.try_wait p0, [ur4], 0
        @!p0 bra.uni wait
        halt
    )", profile), std::move(launch), nanosm::TraceOptions{.timeline = true, .debug = true});
    machine.run(20000);
    std::vector<const nanosm::TimelineEvent*> inputs, writes;
    for (const auto& event : machine.trace().events()) {
        if (event.name == "MMA input step") inputs.push_back(&event);
        if (event.name == "MMA tmem write") writes.push_back(&event);
    }
    require(inputs.size() == 128 && writes.size() == 128, "one m32n64k8 command is 128 4x4x8 steps");
    const auto d = profile.shmem_port_a_depth;
    const auto lb = profile.shmem_read_latency_cycles;
    for (uint32_t s = 0; s < 128; ++s) {
        const bool reads_a = s % 16 == 0;
        const uint32_t rounds = reads_a ? 2 : 1;
        require(static_cast<uint32_t>(inputs[s]->value) == s &&
                inputs[s]->detail == (reads_a ? "rounds=2, reads A" : "rounds=1"),
            "m32 step " + std::to_string(s) + ": " + inputs[s]->detail);
        if (s + 1 < 128)
            require(inputs[s + 1]->start_cycle == inputs[s]->start_cycle + rounds,
                "m32 step " + std::to_string(s + 1) + " did not start the cycle after the previous step's last round");
        const uint32_t tile = 4096 + 64 * (4 * (s / 16)) + 4 * (s % 16); // the block's address: 4096 + m0 * 64 + n0
        require(static_cast<uint32_t>(writes[s]->value) == tile &&
                writes[s]->start_cycle == inputs[s]->start_cycle + (rounds - 1) + d + lb +
                    profile.mma_result_latency_cycles + profile.tmem_port_depth,
            "m32 step " + std::to_string(s) + ": tile or tmem write cycle");
    }
    const auto& summary = machine.trace().summary();
    require(summary.mma_a_block_reads == 8 && summary.mma_shmem_rounds == 136,
        "one m32 command did not read 8 A blocks in 136 bank rounds");
    for (uint32_t row = 0; row < 32; ++row)
        for (uint32_t col = 0; col < 64; ++col)
            require(nanosm::bits_float(machine.tmem_word(i9_tmem_word(4096, 64, row, col))) ==
                    i9_reference(32 + row, col),
                "m32 on the bottom half: element (" + std::to_string(row) + ", " + std::to_string(col) + ")");
}

// The same 64-row A object as one m64 command and as two m32 commands on its
// halves (bases 0 and 512): the halves equal the m64 result bit for bit, and
// both equal the logical reference. Then two m32 commands back to back keep
// port B busy exactly 2 x 136 = 272 cycles from the first input step.
void test_i9_two_m32_halves_equal_one_m64() {
    const auto& profile = canonical_profile();
    nanosm::Launch launch;
    launch.active_warps = 1;
    i9_write_operands(launch);
    nanosm::Machine machine(profile, compile(R"(
        umov ur0, 0
        umov ur2, 512
        umov ur1, 2048
        umov ur4, 8192
        umov ur5, 0
        umov ur6, 4096
        umov ur7, 6144
        mbar.init [ur4], 1
        mma.async.init.m64n64k8 [ur5], [ur0], [ur1]
        mma.async.init.m32n64k8 [ur6], [ur0], [ur1]
        mma.async.init.m32n64k8 [ur7], [ur2], [ur1]
        mma.commit [ur4]
    wait:
        mbar.try_wait p0, [ur4], 0
        @!p0 bra.uni wait
        halt
    )", profile), std::move(launch), nanosm::TraceOptions{.timeline = true, .debug = true});
    machine.run(40000);
    for (uint32_t row = 0; row < 64; ++row) {
        for (uint32_t col = 0; col < 64; ++col) {
            const auto full = machine.tmem_word(i9_tmem_word(0, 64, row, col));
            const auto half = machine.tmem_word(i9_tmem_word(row < 32 ? 4096 : 6144, 64, row % 32, col));
            require(full == half && nanosm::bits_float(full) == i9_reference(row, col),
                "row " + std::to_string(row) + " col " + std::to_string(col) + ": m64 " +
                    std::to_string(nanosm::bits_float(full)) + ", m32 half " + std::to_string(nanosm::bits_float(half)));
        }
    }
    std::vector<uint64_t> starts;
    for (const auto& event : machine.trace().events())
        if (event.name == "MMA input step") starts.push_back(event.start_cycle);
    const auto& summary = machine.trace().summary();
    require(starts.size() == 256 + 128 + 128 && summary.mma_shmem_rounds == 272 + 136 + 136 &&
            summary.mma_a_block_reads == 16 + 8 + 8,
        "m64 + two m32 commands: steps, rounds or A reloads");
    // The two m32 commands (steps 256..511) run back to back: the last step
    // reads one round, so port B is busy through its start.
    require(starts.back() - starts.at(256) + 1 == 272,
        "two m32 commands did not keep port B busy 272 cycles: " + std::to_string(starts.back() - starts.at(256) + 1));
    require(summary.mma_recurrence_stalls == 0, "independent accumulators hit recurrence stalls");
}

// A mixed queue: m64 init into F, m32 init into T (top half), m64 adding into
// F, m32 init into L (bottom half). F holds twice the m64 product, T and L
// the halves once; 768 steps, 816 rounds.
void test_i9_mixed_queue() {
    const auto& profile = canonical_profile();
    nanosm::Launch launch;
    launch.active_warps = 1;
    i9_write_operands(launch);
    nanosm::Machine machine(profile, compile(R"(
        umov ur0, 0
        umov ur2, 512
        umov ur1, 2048
        umov ur4, 8192
        umov ur5, 0
        umov ur6, 4096
        umov ur7, 6144
        mbar.init [ur4], 1
        mma.async.init.m64n64k8 [ur5], [ur0], [ur1]
        mma.async.init.m32n64k8 [ur6], [ur0], [ur1]
        mma.async.m64n64k8 [ur5], [ur0], [ur1]
        mma.async.init.m32n64k8 [ur7], [ur2], [ur1]
        mma.commit [ur4]
    wait:
        mbar.try_wait p0, [ur4], 0
        @!p0 bra.uni wait
        halt
    )", profile), std::move(launch), nanosm::TraceOptions{.timeline = true});
    machine.run(40000);
    for (uint32_t row = 0; row < 64; ++row) {
        for (uint32_t col = 0; col < 64; ++col) {
            require(nanosm::bits_float(machine.tmem_word(i9_tmem_word(0, 64, row, col))) == 2.0f * i9_reference(row, col),
                "mixed queue: F row " + std::to_string(row) + " col " + std::to_string(col));
            require(nanosm::bits_float(machine.tmem_word(i9_tmem_word(row < 32 ? 4096 : 6144, 64, row % 32, col))) ==
                    i9_reference(row, col),
                "mixed queue: half row " + std::to_string(row) + " col " + std::to_string(col));
        }
    }
    const auto& summary = machine.trace().summary();
    require(summary.mma_input_steps == 768 && summary.mma_shmem_rounds == 816 && summary.mma_a_block_reads == 48,
        "mixed queue: steps, rounds or A reloads");
}

// Four 8 KiB (32x64) accumulators fill tmem (4 x 2048 words). acc.st.v8 and
// acc.ld.v8 at row bases 0 and 16 of each: the kernel chooses the row base,
// nothing in hardware ties rows to warps. Every stored word lands at its
// restated tmem word and loads back.
void test_i9_four_8k_accumulators() {
    const auto& profile = canonical_profile();
    require(profile.tmem_fine_banks * profile.tmem_words_per_fine_bank == 4 * 2048,
        "tmem is not four 32x64 FP32 accumulators");
    std::ostringstream qtx;
    const std::array<const char*, 4> names{"S0", "S1", "O0", "O1"};
    // Region r at tmem 2048 r, rows of 64 words: row base b is pointer 2048 r + 64 b.
    qtx << "sread r30, LANE_ID\nshli r32, r30, 3\nshli r33, r30, 1\nadd r31, r32, r33\numov ur9, 0\n"; // r31 = 10 * lane
    for (uint32_t region = 0; region < 4; ++region) {
        for (const uint32_t row_base : {0u, 16u}) {
            const uint32_t base_value = 10000 * region + 1000 * (row_base / 16);
            qtx << "umov ur8, " << 2048 * region + 64 * row_base << "\n";
            for (uint32_t j = 0; j < 8; ++j) qtx << "addi r" << 16 + j << ", r31, " << base_value + j << "\n";
            qtx << "acc.st.v8 [ur8, 64], r16\n";
        }
    }
    for (uint32_t region = 0; region < 4; ++region)
        qtx << "umov ur8, " << 2048 * region + 1024 << "\nacc.ld.v8 r" << 40 + 8 * region << ", [ur8, 64]\n";
    qtx << "halt\n";
    nanosm::Launch launch;
    launch.active_warps = 1;
    nanosm::Machine machine(profile, compile(qtx.str(), profile), std::move(launch));
    machine.run(20000);
    for (uint32_t region = 0; region < 4; ++region) {
        for (const uint32_t row_base : {0u, 16u}) {
            for (uint32_t lane = 0; lane < 16; ++lane) {
                for (uint32_t j = 0; j < 8; ++j) {
                    const uint32_t value = 10000 * region + 1000 * (row_base / 16) + 10 * lane + j;
                    require(machine.tmem_word(i9_tmem_word(2048 * region, 64, row_base + lane, j)) == value,
                        std::string(names[region]) + " row " + std::to_string(row_base + lane) + " col " +
                            std::to_string(j));
                    if (row_base == 16) {
                        require(machine.read_reg(0, lane, static_cast<uint8_t>(40 + 8 * region + j)) == value,
                            std::string(names[region]) + ": acc.ld.v8 at row base 16, lane " + std::to_string(lane));
                    }
                }
            }
        }
    }
}

// SANITIZER. The sanitizer's MMA operand spans come from the command's shape: an m32
// command on A at 0 read-owns [0, 512), so an sts into the other half of a
// 64-row object ([512, 1024)) is quiet while one inside its span PANICs.
void test_i9_operand_span_follows_the_shape() {
    const auto& profile = canonical_profile();
    const auto program = [](uint32_t sts_base) {
        return "umov ur0, 0\numov ur1, 2048\numov ur4, 8192\numov ur2, 4096\nmbar.init [ur4], 1\n"
               "sread r20, LANE_ID\nshli r20, r20, 2\nmov r21, 0\n"
               "mma.async.init.m32n64k8 [ur2], [ur0], [ur1]\n"
               "sts [r20 + " + std::to_string(sts_base) + "], r21\n"
               "mma.commit [ur4]\nwait:\nmbar.try_wait p0, [ur4], 0\n@!p0 bra.uni wait\nhalt\n";
    };
    nanosm::Launch quiet_launch;
    quiet_launch.active_warps = 1;
    i9_write_operands(quiet_launch);
    nanosm::Machine quiet(profile, compile(program(512), profile), std::move(quiet_launch));
    quiet.run(20000);
    nanosm::Launch race_launch;
    race_launch.active_warps = 1;
    i9_write_operands(race_launch);
    const auto message = run_expecting_panic(profile, compile(program(256), profile), std::move(race_launch),
        "an sts into an m32 command's A rows");
    require_mentions(message, {"mma.async #1", "A operand [0, 512)"}, "m32 operand span");
}

// ---- D15: MMA shapes configurable from the profile (trash/chan.txt 2026-10-22) ----

// The operand layout for any k: rows of k fp16, whole, `d15_spacing(k)`
// bytes apart (the smallest swizzle64 spacing that holds them; the command
// names it), so a shape with k > 8 reads consecutive 16-byte pieces of the
// same rows.
uint32_t d15_spacing(uint32_t k) {
    return std::max(16u, 2 * k);
}

uint32_t d15_a_address(uint32_t base, uint32_t k_extent, uint32_t m, uint32_t k) {
    return reference_swizzle64(base, d15_spacing(k_extent), m, 2 * k);
}

uint32_t d15_b_address(uint32_t base, uint32_t k_extent, uint32_t k, uint32_t n) {
    return reference_swizzle64(base, d15_spacing(k_extent), n, 2 * k);
}

float d15_reference(uint32_t m, uint32_t n, uint32_t depth) {
    float sum = 0.0f;
    for (uint32_t k = 0; k < depth; ++k) sum += static_cast<float>(i9_a(m, k) * i9_b(k, n));
    return sum;
}

// One mma.async.init of `shape` on `profile` (which lists it): A at 0, B at
// the next 1 KiB boundary, the accumulator at tmem word 0. Every element is
// checked against the logical reference; the machine is returned for its
// trace.
std::unique_ptr<nanosm::Machine> d15_run_shape(const nanosm::DeviceProfile& profile, const nanosm::MmaShape& shape) {
    const auto name = nanosm::mma_shape_name(shape);
    const uint32_t spacing = d15_spacing(shape.k);
    const uint32_t a_bytes = shape.m * spacing;
    const uint32_t b_base = (a_bytes + 1023) / 1024 * 1024;
    const uint32_t b_bytes = shape.n * spacing;
    const uint32_t mbar = 16384;
    std::ostringstream source;
    source << "umov ur0, 0\numov ur1, " << b_base << "\numov ur4, " << mbar << "\numov ur2, 0\nmbar.init [ur4], 1\n"
           << "mma.async.init." << name << " [ur2], [ur0, " << spacing << "], [ur1, " << spacing << "]\n"
           << "mma.commit [ur4]\nwait:\nmbar.try_wait p0, [ur4], 0\n@!p0 bra.uni wait\nhalt\n";
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.test_shmem, b_base + b_bytes);
    for (uint32_t m = 0; m < shape.m; ++m)
        for (uint32_t k = 0; k < shape.k; ++k)
            write_u16(shmem, d15_a_address(0, shape.k, m, k), i9_fp16(i9_a(m, k)));
    for (uint32_t k = 0; k < shape.k; ++k)
        for (uint32_t n = 0; n < shape.n; ++n)
            write_u16(shmem, d15_b_address(b_base, shape.k, k, n), i9_fp16(i9_b(k, n)));
    auto machine = std::make_unique<nanosm::Machine>(profile, compile(source.str(), profile), std::move(launch),
        nanosm::TraceOptions{.timeline = true, .debug = true});
    machine->run(200000);
    for (uint32_t row = 0; row < shape.m; ++row)
        for (uint32_t col = 0; col < shape.n; ++col)
            require(nanosm::bits_float(machine->tmem_word(i9_tmem_word(0, shape.n, row, col))) ==
                    d15_reference(row, col, shape.k),
                name + ": element (" + std::to_string(row) + ", " + std::to_string(col) + ")");
    return machine;
}

nanosm::DeviceProfile d15_profile(std::vector<nanosm::MmaShape> shapes, uint32_t result_latency = 8) {
    auto profile = canonical_profile();
    profile.mma_shapes = std::move(shapes);
    profile.mma_result_latency_cycles = result_latency;
    nanosm::validate_device_profile(profile);
    return profile;
}

// Legal shapes beyond the canonical two run correctly against the logical
// reference: a k > core_k shape over four K blocks, a short 16-row shape and
// a 128-row shape that fills tmem. Steps and port-B rounds follow the walk:
// per K block every (m, n) block, A read once per m block (two rounds), and
// no step of one command ever waits on its own accumulator.
void test_d15_more_shapes_run() {
    const auto profile = d15_profile({{64, 64, 8}, {16, 64, 32}, {16, 64, 8}, {128, 64, 8}});
    for (const nanosm::MmaShape shape : {nanosm::MmaShape{16, 64, 32}, nanosm::MmaShape{16, 64, 8},
             nanosm::MmaShape{128, 64, 8}}) {
        const auto machine = d15_run_shape(profile, shape);
        const auto& summary = machine->trace().summary();
        const uint32_t blocks_m = shape.m / 4;
        const uint32_t steps = blocks_m * (shape.n / 4) * (shape.k / 8);
        const uint32_t a_reads = blocks_m * (shape.k / 8);
        require(summary.mma_input_steps == steps && summary.mma_a_block_reads == a_reads &&
                summary.mma_shmem_rounds == steps + a_reads && summary.mma_recurrence_stalls == 0,
            nanosm::mma_shape_name(shape) + ": steps " + std::to_string(summary.mma_input_steps) + ", A reads " +
                std::to_string(summary.mma_a_block_reads) + ", rounds " + std::to_string(summary.mma_shmem_rounds) +
                ", recurrence stalls " + std::to_string(summary.mma_recurrence_stalls));
    }
}

// The latency bound at equality. With result latency 10 the bound is
// D 2 + Lb 2 + 10 + Dt 2 = 16, and m16n16k16 revisits each block every
// (16 / 4) x (16 / 4) = 16 steps: accepted, and it never stalls. The step
// that revisits a block starts its operand rounds no earlier than the cycle
// the block's previous result lands. That slack is exactly what the
// derivation says: the 16 steps between the visits take 16 + 4 rounds
// (each m block's A reload is one more), so a visit that is itself a reload
// (2 rounds) leaves 4 - 1 = 3 cycles and any other leaves 4: the validated
// bound counts one round per step and is short of exact by the reloads.
// With result latency 11 the bound is 17 and the shape is refused.
void test_d15_latency_bound() {
    const nanosm::MmaShape shape{16, 16, 16};
    const auto profile = d15_profile({{64, 64, 8}, shape}, 10);
    require(nanosm::mma_min_blocks_per_k(profile) == 16 && nanosm::mma_blocks_per_k(profile, shape) == 16,
        "the bound or the blocks per K block");
    const auto machine = d15_run_shape(profile, shape);
    std::vector<const nanosm::TimelineEvent*> inputs, writes;
    for (const auto& event : machine->trace().events()) {
        if (event.name == "MMA input step") inputs.push_back(&event);
        if (event.name == "MMA tmem write") writes.push_back(&event);
    }
    require(inputs.size() == 32 && writes.size() == 32 && machine->trace().summary().mma_recurrence_stalls == 0,
        "m16n16k16 is 32 steps with no recurrence stall");
    int64_t least = std::numeric_limits<int64_t>::max();
    for (uint32_t s = 0; s < 16; ++s) {
        const auto slack = static_cast<int64_t>(inputs[s + 16]->start_cycle) -
            static_cast<int64_t>(writes[s]->start_cycle);
        const bool reload = s % 4 == 0;
        require(slack == (reload ? 3 : 4), "step " + std::to_string(s + 16) + " revisits its block with slack " +
            std::to_string(slack) + " (expected " + std::to_string(reload ? 3 : 4) + ")");
        least = std::min(least, slack);
    }
    require(least == 3, "the least slack is the reloads' extra rounds less one");
    auto slower = profile;
    slower.mma_result_latency_cycles = 11;
    try {
        nanosm::validate_device_profile(slower);
        throw std::runtime_error("m16n16k16 was accepted with a bound of 17");
    } catch (const nanosm::ProfileError& error) {
        require_mentions(error.what(), {"m16n16k16", "revisited every 16 steps", "needs at least 17"},
            "the bound one past equality");
    }
}

// D15 follow-ups (trash/chan.txt 2026-10-22). On a 4x4x4 core, m16n16k4 is
// legal (every dimension a power of two and whole core blocks) but reads
// only 4 of the 8 fp16 in each 16-byte swizzle64 operand row: a warning, not
// an error. Admission counts the bytes the walk actually reads: at the
// default 16-byte spacing rows 12..15 keep their four fp16 in the second
// half of their row (word XOR 3), so A's last byte read is 256 B past its
// base, not 15 x 16 + 8. Codex's reproducer puts A at 65536 - 128: refused at
// admission. At 65536 - 256 it runs. k = 12 is not a power of two: refused
// at load.
void test_d15_partial_k_block_footprint_and_warning() {
    auto profile = canonical_profile();
    profile.mma_core_k = 4;
    profile.mma_shapes = {{16, 16, 4}};
    nanosm::validate_device_profile(profile);
    const auto warnings = nanosm::device_profile_warnings(profile);
    require(warnings.size() == 1, "one partial-K-block warning for m16n16k4");
    require_mentions(warnings[0], {"mma shape m16n16k4", "reads only 4 of the 8 fp16", "50%"},
        "the partial-K-block warning");
    require(nanosm::device_profile_warnings(canonical_profile()).empty(), "the canonical profile warns");
    auto twelve = profile;
    twelve.mma_shapes = {{16, 16, 12}};
    try {
        nanosm::validate_device_profile(twelve);
        throw std::runtime_error("m16n16k12 was accepted");
    } catch (const nanosm::ProfileError& error) {
        require_mentions(error.what(), {"m16n16k12", "powers of two (k is 12)"}, "k = 12 at load");
    }
    const auto run = [&](uint32_t a_base) {
        nanosm::Launch launch;
        launch.active_warps = 1;
        const auto source = "umov ur0, " + std::to_string(a_base) +
            "\numov ur1, 0\nmma.async.init.m16n16k4 [ur8], [ur0], [ur1]\nhalt\n";
        nanosm::Machine machine(profile, compile(source, profile), std::move(launch));
        machine.run(10000);
    };
    try {
        run(65536 - 128);
        throw std::runtime_error("an A operand whose storage runs past shmem was admitted");
    } catch (const nanosm::SimError& error) {
        require_mentions(error.what(), {"MMA operand alignment or footprint is outside shmem"},
            "the partly filled A object at 65408");
    }
    run(65536 - 256); // the whole 256-byte object fits: admitted and run
}

// ---------------------------------------------------------------------------
// I8 (trash/rfc11.md section 0): tmem is 16 fine banks, each with two read
// and two write services per cycle -- the MMA's accumulator read and write,
// the acc.ld engine's read, the acc.st engine's write. Nobody arbitrates for
// them: the MMA never waits for a transfer, a transfer never waits for the
// MMA (racing them is the sanitizer's business), and one acc.ld and one
// acc.st engine serve the whole machine (a second transfer of a kind waits
// at admission). Depths: a request leaving its producer reaches the RAM
// tmem.port_depth (Dt) later (a read's sample, a write's edge); a read's
// words are at the output tmem.read_latency_cycles (Lt) after its sample.
// ---------------------------------------------------------------------------

// A machine with I7's operand patterns (A all `a`, B all `b`) and `body` as
// qtx, for `warps` warps.
std::unique_ptr<nanosm::Machine> i8_machine(const nanosm::DeviceProfile& profile, const std::string& body,
    uint32_t warps, uint16_t a, uint16_t b) {
    nanosm::Launch launch;
    launch.active_warps = warps;
    auto& shmem = image_bytes(launch.test_shmem, 8192);
    for (uint32_t m = 0; m < 64; ++m)
        for (uint32_t k = 0; k < 8; ++k) write_u16(shmem, i7_a_address(m, k), a);
    for (uint32_t k = 0; k < 8; ++k)
        for (uint32_t n = 0; n < 64; ++n) write_u16(shmem, i7_b_address(k, n), b);
    return std::make_unique<nanosm::Machine>(profile, compile(body, profile), std::move(launch),
        nanosm::TraceOptions{.timeline = true, .debug = true});
}

// Warp 0 runs `commands` MMA commands into S; warp 1 meanwhile streams
// acc.st.v8 / acc.ld.v8 round trips through O (another accumulator). The MMA
// takes its accumulator read and write every step on its own services: every
// step starts the cycle after the previous step's last round and writes tmem
// exactly (rounds - 1) + D + Lb + result latency + Dt after its start, the
// I7 walker's timing with nothing else running, and back to back the
// commands still keep port B busy one round per cycle (16/17 of a step per
// cycle). The transfers really run alongside (beats of both kinds inside the
// MMA's span), and every value is right.
void test_i8_mma_beside_transfers() {
    const auto& profile = canonical_profile();
    constexpr uint32_t commands = 4;
    std::string mma = "mma.async.init.m64n64k8 [ur2], [ur0], [ur1]\n"; // S: 64x64 at tmem 0
    for (uint32_t command = 1; command < commands; ++command) mma += "mma.async.m64n64k8 [ur2], [ur0], [ur1]\n";
    const std::string body = R"(
        usread ur12, WARP_ID
        ur2r r20, ur12
        setp.eq.u32 p0, r20, 0
        @p0 bra.uni MMA_WARP
        sread r0, LANE_ID
        addi r1, r0, 100
        addi r2, r0, 200
        addi r3, r0, 300
        addi r4, r0, 400
        addi r5, r0, 500
        addi r6, r0, 600
        addi r7, r0, 700
        umov ur8, 4096
        umov ur9, 0
        mov r30, 0
    LOOP:
        acc.st.v8 [ur8, 64], r0
        acc.ld.v8 r16, [ur8, 64]
        addi r30, r30, 1
        setp.lt.u32 p1, r30, 12
        @p1 bra.uni LOOP
        halt
    MMA_WARP:
        umov ur0, 0
        umov ur1, 1024
        umov ur4, 4096
        umov ur2, 0
        mbar.init [ur4], 1
    )" + mma + R"(
        mma.commit [ur4]
    WAIT:
        mbar.try_wait p0, [ur4], 0
        @!p0 bra.uni WAIT
        umov ur8, 0
        umov ur9, 0
        acc.ld r10, [ur8, 64]
        halt
    )";
    auto machine = i8_machine(profile, body, 2, 0x3c00, 0x4000); // 1 x 2
    machine->run(100000);
    std::vector<const nanosm::TimelineEvent*> inputs, writes;
    std::vector<uint64_t> ld_beats, st_beats;
    for (const auto& event : machine->trace().events()) {
        if (event.name == "MMA input step") inputs.push_back(&event);
        if (event.name == "MMA tmem write") writes.push_back(&event);
        if (event.name == "acc.ld beat") ld_beats.push_back(event.start_cycle);
        if (event.name == "acc.st beat") st_beats.push_back(event.start_cycle);
    }
    require(inputs.size() == 256 * commands && writes.size() == 256 * commands, "MMA steps lost");
    const auto d = profile.shmem_port_a_depth;
    const auto lb = profile.shmem_read_latency_cycles;
    for (std::size_t s = 0; s < inputs.size(); ++s) {
        const uint32_t rounds = s % 16 == 0 ? 2 : 1;
        if (s + 1 < inputs.size())
            require(inputs[s + 1]->start_cycle == inputs[s]->start_cycle + rounds,
                "step " + std::to_string(s + 1) + " waited (the transfers stalled the MMA?)");
        require(writes[s]->start_cycle == inputs[s]->start_cycle + (rounds - 1) + d + lb +
                profile.mma_result_latency_cycles + profile.tmem_port_depth,
            "step " + std::to_string(s) + "'s tmem write moved (the transfers delayed the MMA?)");
    }
    require(inputs.back()->start_cycle - inputs.front()->start_cycle + 1 == 272 * commands,
        "back to back with transfers alongside, the MMA did not keep port B busy one round per cycle");
    const auto first = inputs.front()->start_cycle;
    const auto last = writes.back()->start_cycle;
    const auto inside = [&](const std::vector<uint64_t>& beats) {
        return std::count_if(beats.begin(), beats.end(), [&](uint64_t cycle) { return cycle >= first && cycle <= last; });
    };
    require(inside(ld_beats) >= 16 && inside(st_beats) >= 16,
        "the transfers did not run alongside the MMA (" + std::to_string(inside(ld_beats)) + " acc.ld and " +
            std::to_string(inside(st_beats)) + " acc.st beats inside it)");
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(machine->read_reg(0, lane, 10) == nanosm::float_bits(16.0f * commands), "MMA result");
        for (uint32_t j = 0; j < 8; ++j) {
            require(machine->read_reg(1, lane, 16 + j) == lane + 100 * j, "acc.st -> acc.ld round trip");
        }
    }
}

// One acc.ld engine and one acc.st engine for the machine. Two warps each
// store eight columns of their own rows and load them back. At most one
// acc.ld beat (a tmem sample) and one acc.st beat (a tmem write) happen per
// cycle; the second warp's transfer of a kind waits at admission ("acc
// transfer engine busy") while the first warp's runs; an acc.ld and an
// acc.st do run in the same cycles; every value is right.
void test_i8_one_engine_of_each_kind() {
    const auto& profile = canonical_profile();
    const std::string body = R"(
        sread r0, LANE_ID
        usread ur2, WARP_ID
        ur2r r20, ur2
        ushli ur8, ur2, 10
        shli r22, r20, 10
        add r0, r0, r22
        addi r1, r0, 100
        addi r2, r0, 200
        addi r3, r0, 300
        addi r4, r0, 400
        addi r5, r0, 500
        addi r6, r0, 600
        addi r7, r0, 700
        umov ur9, 0
        acc.st.v8 [ur8, 64], r0
        acc.ld.v8 r16, [ur8, 64]
        halt
    )";
    auto machine = i8_machine(profile, body, 2, 0, 0);
    machine->run(20000);
    std::map<uint64_t, uint32_t> ld_per_cycle, st_per_cycle;
    for (const auto& event : machine->trace().events()) {
        if (event.name == "acc.ld beat") ++ld_per_cycle[event.start_cycle];
        if (event.name == "acc.st beat") ++st_per_cycle[event.start_cycle];
    }
    require(ld_per_cycle.size() == 16 && st_per_cycle.size() == 16, "two warps x eight beats of each kind");
    for (const auto& [cycle, count] : ld_per_cycle) require(count == 1, "two acc.ld beats in one cycle");
    for (const auto& [cycle, count] : st_per_cycle) require(count == 1, "two acc.st beats in one cycle");
    bool overlap = false;
    for (const auto& [cycle, count] : ld_per_cycle) overlap = overlap || st_per_cycle.contains(cycle);
    require(overlap, "an acc.ld and an acc.st never ran in the same cycle");
    bool engine_busy = false;
    for (uint32_t warp = 0; warp < 2; ++warp) engine_busy = engine_busy || !refusals(*machine, warp, "acc transfer engine busy").empty();
    require(engine_busy, "no transfer ever waited for its engine");
    for (uint32_t warp = 0; warp < 2; ++warp)
        for (uint32_t lane = 0; lane < 16; ++lane)
            for (uint32_t j = 0; j < 8; ++j)
                require(machine->read_reg(warp, lane, 16 + j) == 1024 * warp + lane + 100 * j, "round trip data");

    // Two acc.ld.v8 at once (no stores first): one waits at admission until
    // the other's last beat has launched, so the two transfers' beats do not
    // interleave.
    auto loads = i8_machine(profile, R"(
        usread ur8, WARP_ID
        ushli ur8, ur8, 10
        umov ur9, 0
        acc.ld.v8 r16, [ur8, 64]
        halt
    )", 2, 0, 0);
    loads->run(20000);
    std::vector<std::string> owners;
    for (const auto& event : loads->trace().events()) {
        if (event.name == "acc.ld beat") owners.push_back(event.detail.substr(0, event.detail.find(',')));
    }
    require(owners.size() == 16, "two acc.ld.v8 are sixteen beats");
    for (uint32_t i = 0; i < 16; ++i) {
        require(owners[i] == owners[i < 8 ? 0 : 8], "the two acc.ld transfers' beats interleaved");
    }
    require(owners[0] != owners[8], "both transfers' beats came from one warp");
    bool load_waited = false;
    for (uint32_t warp = 0; warp < 2; ++warp) load_waited = load_waited || !refusals(*loads, warp, "acc transfer engine busy").empty();
    require(load_waited, "the second acc.ld did not wait for the acc.ld engine");
}

// The tmem depths, at (Dt, Lt) = (1, 1), (2, 2), (3, 1), from the debug trace
// (hand assembly, one warp): an acc.st beat whose RF read is granted at g
// reaches the tmem RAM ("acc.st beat") at g + Lr + Dt; an acc.ld launched at
// L = its dispatch + 1 samples tmem ("acc.ld beat") at L + Dt and is at the RF
// port ("RF write") at L + Dt + Lt; an MMA step writes tmem ("MMA tmem
// write") (rounds - 1) + D + Lb + result latency + Dt after its start.
void test_i8_tmem_depths() {
    for (const auto& [dt, lt] : std::vector<std::pair<uint32_t, uint32_t>>{{1, 1}, {2, 2}, {3, 1}}) {
        auto profile = canonical_profile();
        profile.tmem_port_depth = dt;
        profile.tmem_read_latency_cycles = lt;
        nanosm::validate_device_profile(profile);
        const auto what = "Dt/Lt " + std::to_string(dt) + "/" + std::to_string(lt) + ": ";
        auto machine = i6a_machine(profile, hand_assemble(i1_asm(profile, R"(
            sread r0, LANE_ID
            umov ur8, 0
            umov ur9, 0
            acc.st [ur8, 64], r0 {pred=PT, stall=0, set=sb0, wait=-}
            acc.ld r10, [ur8, 64] {pred=PT, stall=0, set=sb1, wait=sb0}
            halt {pred=PT, stall=0, set=-, wait=sb1}
        )"), profile));
        machine->run();
        const auto st_issue = issued_at(*machine, 3 * 8).at(0);
        const auto ld_issue = issued_at(*machine, 4 * 8).at(0);
        const auto t = nanosm::issue_to_dispatch(profile);
        uint64_t st_beat = 0, ld_beat = 0;
        for (const auto& event : machine->trace().events()) {
            if (event.name == "acc.st beat") st_beat = event.start_cycle;
            if (event.name == "acc.ld beat") ld_beat = event.start_cycle;
        }
        const auto g = i6a_rf_events(*machine, "RF read", 0).back(); // the acc.st's data read
        require(g >= st_issue + t + 1, what + "the acc.st's RF read before its engine had it");
        require(st_beat == g + profile.rf_read_latency_cycles + dt, what + "acc.st did not land at g + Lr + Dt");
        const auto launch = ld_issue + t + 1;
        require(ld_beat == launch + dt, what + "acc.ld did not sample at its launch + Dt");
        require(i6a_rf_events(*machine, "RF write", 10).back() == launch + dt + lt,
            what + "acc.ld was not at the RF port at its launch + Dt + Lt");
        for (uint32_t lane = 0; lane < 16; ++lane)
            require(machine->read_reg(0, lane, 10) == lane, what + "acc.st -> acc.ld data");

        // The MMA's result write.
        const std::string mma_body = R"(
            umov ur0, 0
            umov ur1, 1024
            umov ur4, 4096
            mbar.init [ur4], 1
            mma.async.init.m64n64k8 [ur8], [ur0], [ur1]   ; ur8 is 0 until written below: tmem 0
            mma.commit [ur4]
        WAIT:
            mbar.try_wait p0, [ur4], 0
            @!p0 bra.uni WAIT
            umov ur8, 0
            umov ur9, 0
            acc.ld r10, [ur8, 64]
            halt
        )";
        auto mma = i8_machine(profile, mma_body, 1, 0x3c00, 0x4000);
        mma->run(20000);
        const nanosm::TimelineEvent* input = nullptr;
        const nanosm::TimelineEvent* write = nullptr;
        for (const auto& event : mma->trace().events()) {
            if (event.name == "MMA input step" && input == nullptr) input = &event;
            if (event.name == "MMA tmem write" && write == nullptr) write = &event;
        }
        require(input != nullptr && write != nullptr &&
                write->start_cycle == input->start_cycle + 1 + profile.shmem_port_a_depth +
                    profile.shmem_read_latency_cycles + profile.mma_result_latency_cycles + dt,
            what + "the first MMA step's tmem write is not its start + 1 + D + Lb + result latency + Dt");
        for (uint32_t lane = 0; lane < 16; ++lane)
            require(mma->read_reg(0, lane, 10) == nanosm::float_bits(16.0f), what + "MMA data");
    }
}

// Same word, same cycle (READ_FIRST, as shmem): an acc.st issued at I and an
// acc.ld of the same column `gap` cycles later (hand assembly, no wait: a
// race). The store's beat is granted its RF read at I + t + 1 (its engine has
// it from then; its data r1 is the only vector register either reads: the
// tmem pointer ur8 is uniform) and reaches the tmem RAM at
// I + t + 1 + Lr + Dt; the load launches at I + gap + t + 1 and samples at
// I + gap + t + 1 + Dt: the same cycle when gap = Lr (2). The sanitizer
// PANICs at the load's dispatch (it reads words a store still writes);
// without it the machine is quiet and the sample sees the OLD words (the
// write lands at that cycle's edge, after every read of the cycle). One
// cycle later (gap 3), it sees the new ones.
void test_i8_read_first_same_word() {
    const auto& profile = canonical_profile();
    require(profile.rf_read_latency_cycles == 2, "the derivation needs Lr = 2");
    for (const uint32_t gap : {2u, 3u}) {
        auto machine = i6a_machine(profile, hand_assemble(i1_asm(profile, R"(
            sread r1, LANE_ID
            addi r1, r1, 7
            umov ur8, 0
            umov ur9, 0
            acc.st [ur8, 64], r1 {pred=PT, stall=)" + std::to_string(gap) + R"(, set=sb0, wait=-}
            acc.ld r10, [ur8, 64] {pred=PT, stall=0, set=sb1, wait=-}
            halt {pred=PT, stall=0, set=-, wait=sb0|sb1}
        )"), profile));
#if NANOSM_SANITIZER
        require_mentions(i4_panic(*machine, "acc.ld over an in-flight acc.st"),
            {"tmem race: acc.ld reads tmem 0 (64 words per row) rows [0, 16) cols [0, 1)", "overlapping acc.st"}, "same-word acc.ld after acc.st");
#else
        while (!machine->done()) tick_capped(*machine);
        uint64_t st_beat = 0, ld_beat = 0;
        for (const auto& event : machine->trace().events()) {
            if (event.name == "acc.st beat") st_beat = event.start_cycle;
            if (event.name == "acc.ld beat") ld_beat = event.start_cycle;
        }
        require(ld_beat == st_beat + gap - 2, "the load's sample is not " + std::to_string(gap - 2) +
            " cycles after the store's write");
        const uint32_t expected_lane0 = gap == 2 ? 0u : 7u; // old (zero) tmem word, or the stored lane + 7
        require(machine->read_reg(0, 0, 10) == expected_lane0,
            gap == 2 ? "a same-cycle sample saw the landing write (READ_FIRST)"
                     : "a sample one cycle after the write did not see it");
#endif
    }
}

// Racing accepted MMA work is UB: the sanitizer PANICs; without it the
// machine runs quietly to the end (no ownership hardware stops anything).
void test_i8_transfer_racing_mma_is_quiet_without_the_sanitizer() {
    const auto& profile = canonical_profile();
    for (const bool load : {true, false}) {
        const std::string transfer = load ? "acc.ld.v8 r24, [ur8, 64]\n" : "acc.st.v8 [ur8, 64], r16\n";
        auto machine = i8_machine(profile, R"(
            umov ur0, 0
            umov ur1, 1024
            umov ur8, 0
            umov ur9, 0
            mma.async.init.m64n64k8 [ur8], [ur0], [ur1]
        )" + transfer + "halt\n", 1, 0x3c00, 0x3c00);
#if NANOSM_SANITIZER
        require_mentions(i4_panic(*machine, "transfer racing an MMA"),
            {"races accepted MMA work: mma.async #1"}, load ? "acc.ld vs MMA" : "acc.st vs MMA");
#else
        while (!machine->done()) tick_capped(*machine);
#endif
    }
}

// ---- an accumulator is a tmem pointer (trash/acc-pointer-2026-11-19.md) ----

// The layout (src/layout.hpp tmem_physical_word) against an independent
// statement of the rule as bit fields: with S = 2^s words per row, logical
// word L = [band | row in band (2) | column / 4 (s - 2) | column % 4 (2)] is
// stored at [band | column / 4 | row in band | column % 4 XOR band % 4]. Then
// the properties the fine banks need, for every S: each band of four rows is
// a permutation of itself; every 4x4 macro tile, and every 16 consecutive
// rows of one column wherever they start, hit 16 different fine banks (word
// % 16). And a region at a multiple of 16 rows is laid out as a5d0652 laid
// out an accumulator at that base.
void test_tmem_layout_matches_the_rule() {
    for (uint32_t s = 2; s <= 10; ++s) {
        const uint32_t S = 1u << s;
        const auto what = "S " + std::to_string(S) + ": ";
        for (uint32_t L = 0; L < 8192; ++L) {
            const uint32_t column = L & 3;
            const uint32_t quad = (L >> 2) & (S / 4 - 1);
            const uint32_t row = (L >> s) & 3;
            const uint32_t band = L >> (s + 2);
            const uint32_t expected = (band << (s + 2)) | (quad << 4) | (row << 2) | (column ^ (band & 3));
            require(nanosm::tmem_physical_word(L, S) == expected, what + "word " + std::to_string(L));
        }
        for (uint32_t band = 0; band < 8; ++band) {
            std::set<uint64_t> words;
            for (uint32_t L = band * 4 * S; L < (band + 1) * 4 * S; ++L) words.insert(nanosm::tmem_physical_word(L, S));
            require(words.size() == 4 * S && *words.begin() == band * 4 * S && *words.rbegin() == (band + 1) * 4 * S - 1,
                what + "band " + std::to_string(band) + " is not a permutation of itself");
        }
        for (uint32_t r0 = 0; r0 < 32; r0 += 4) {
            for (uint32_t c0 = 0; c0 < S; c0 += 4) {
                std::set<uint64_t> banks;
                for (uint32_t r = 0; r < 4; ++r)
                    for (uint32_t c = 0; c < 4; ++c) banks.insert(nanosm::tmem_physical_word((r0 + r) * S + c0 + c, S) % 16);
                require(banks.size() == 16, what + "the macro tile at row " + std::to_string(r0) + " column " +
                    std::to_string(c0) + " shares a fine bank");
            }
        }
        for (uint32_t r0 = 0; r0 < 24; ++r0) {
            for (uint32_t c = 0; c < S; ++c) {
                std::set<uint64_t> banks;
                for (uint32_t lane = 0; lane < 16; ++lane) banks.insert(nanosm::tmem_physical_word((r0 + lane) * S + c, S) % 16);
                require(banks.size() == 16, what + "the column beat at row " + std::to_string(r0) + " column " +
                    std::to_string(c) + " shares a fine bank");
            }
        }
    }
    const auto old_rule = [](uint32_t base, uint32_t cols, uint32_t row, uint32_t col) {
        return base + ((row / 4) * (cols / 4) + col / 4) * 16 + 4 * (row % 4) + ((col % 4) ^ ((row / 4) % 4));
    };
    for (const auto& [base, cols] : {std::pair{0u, 64u}, std::pair{2048u, 64u}, std::pair{4096u, 64u},
             std::pair{6144u, 64u}, std::pair{4096u, 16u}, std::pair{256u, 16u}}) {
        for (uint32_t row = 0; row < 32; ++row)
            for (uint32_t col = 0; col < cols; ++col)
                require(nanosm::tmem_physical_word(base + row * cols + col, cols) == old_rule(base, cols, row, col),
                    "a region at " + std::to_string(base) + " moved from the per-accumulator layout");
    }
    require(nanosm::valid_tmem_row_words(4) && nanosm::valid_tmem_row_words(64) && !nanosm::valid_tmem_row_words(2) &&
            !nanosm::valid_tmem_row_words(48) && !nanosm::valid_tmem_row_words(0),
        "words per row: powers of two from 4");
    require(nanosm::tmem_pointer_aligned(1304, 64, 8) && nanosm::tmem_pointer_aligned(3, 8, 1) &&
            !nanosm::tmem_pointer_aligned(2 * 64, 64, 1) && !nanosm::tmem_pointer_aligned(60, 64, 8) &&
            nanosm::tmem_pointer_aligned(2048, 64, 64) && !nanosm::tmem_pointer_aligned(2048 + 8, 64, 64),
        "the pointer rule: row a multiple of 4, the columns inside the row");
}

// acc.st / acc.ld through pointers at a non-zero row and column, and the
// address keys the layout: acc.st.v8 at row 20, column 24 of a region of
// 64-word rows (pointer 1304) puts word (lane, j) where the rule says; an
// acc.ld.v8 at row 16, column 24 (1048) reads it back four lanes down (rows
// 16..19 were never written), and an acc.ld at row 32, column 31 (2079)
// sees the last four rows' column 7.
void test_acc_pointer_at_a_row_and_column() {
    const auto& profile = canonical_profile();
    std::string source = "sread r0, LANE_ID\naddi r0, r0, 1\n";
    for (uint32_t j = 0; j < 8; ++j) source += "addi r" + std::to_string(16 + j) + ", r0, " + std::to_string(1000 * j) + "\n";
    source += R"(
        umov ur8, 1304
        acc.st.v8 [ur8, 64], r16
        umov ur9, 1048
        acc.ld.v8 r40, [ur9, 64]
        umov ur10, 2079
        acc.ld r50, [ur10, 64]
        halt
    )";
    nanosm::Launch launch;
    launch.active_warps = 1;
    nanosm::Machine machine(profile, compile(source, profile), std::move(launch));
    machine.run(20000);
    for (uint32_t lane = 0; lane < 16; ++lane) {
        for (uint32_t j = 0; j < 8; ++j) {
            const uint32_t stored = lane + 1 + 1000 * j;
            require(machine.tmem_word(i9_tmem_word(1304, 64, lane, j)) == stored &&
                    machine.tmem_word(nanosm::tmem_physical_word(1304 + 64 * lane + j, 64)) == stored,
                "row " + std::to_string(20 + lane) + " column " + std::to_string(24 + j) + " is not where the rule puts it");
            require(machine.read_reg(0, lane, static_cast<uint8_t>(40 + j)) == (lane < 4 ? 0 : stored - 4),
                "the acc.ld.v8 at row 16, column 24, lane " + std::to_string(lane));
        }
        require(machine.read_reg(0, lane, 50) == (lane < 4 ? 12 + lane + 1 + 7000 : 0),
            "the acc.ld at row 32, column 31, lane " + std::to_string(lane));
    }
}

// The case that needed a branch (trash/fa32-2026-11-19.md "what hurt" 1): two
// m32n64k8 matmuls write rows 0..31 (pointer 0, A rows 0..31) and rows
// 32..63 (pointer 2048, A rows 32..63) of one region of 64-word rows. Four
// warps run the same code with no branch on the half: warp w points at rows
// 16 w.. (pointer 1024 w, from its warp id) and reads columns 0..7 and, eight
// columns at a time, steps on to 56..63. Warp 0 also reads rows 24..39,
// columns 8..15 through one pointer (1544): a view across both results.
void test_acc_view_over_two_matmul_results() {
    const auto& profile = canonical_profile();
    nanosm::Launch launch;
    launch.active_warps = 4;
    i9_write_operands(launch);
    launch.test_shmem.push_back({8192, std::vector<uint8_t>(8, 0)}); // the commit's barrier, expecting 1
    write_count_mbar(launch.test_shmem.back().bytes, 0, 1);
    nanosm::Machine machine(profile, compile(R"(
        umov ur4, 8192
        usread ur10, WARP_ID
        usetp.ne.u32 p0, ur10, 0
        @p0 bra.uni WAIT
        umov ur0, 0
        umov ur1, 2048
        umov ur5, 0
        mma.async.init.m32n64k8 [ur5], [ur0], [ur1]
        umov ur0, 512
        umov ur5, 2048
        mma.async.init.m32n64k8 [ur5], [ur0], [ur1]
        mma.commit [ur4]
    WAIT:
        mbar.try_wait p1, [ur4], 0
        @!p1 bra.uni WAIT
        ushli ur8, ur10, 10
        acc.ld.v8 r16, [ur8, 64]
        uaddi ur8, ur8, 56
        acc.ld.v8 r24, [ur8, 64]
        umov ur9, 1544
        acc.ld.v8 r32, [ur9, 64]
        halt
    )", profile), std::move(launch));
    machine.run(40000);
    for (uint32_t warp = 0; warp < 4; ++warp) {
        for (uint32_t lane = 0; lane < 16; ++lane) {
            for (uint32_t j = 0; j < 8; ++j) {
                const auto where = "warp " + std::to_string(warp) + " lane " + std::to_string(lane) + " j " + std::to_string(j);
                require(nanosm::bits_float(machine.read_reg(warp, lane, static_cast<uint8_t>(16 + j))) ==
                        i9_reference(16 * warp + lane, j), where + ": columns 0..7");
                require(nanosm::bits_float(machine.read_reg(warp, lane, static_cast<uint8_t>(24 + j))) ==
                        i9_reference(16 * warp + lane, 56 + j), where + ": columns 56..63");
                require(nanosm::bits_float(machine.read_reg(warp, lane, static_cast<uint8_t>(32 + j))) ==
                        i9_reference(24 + lane, 8 + j), where + ": rows 24..39 across both results");
            }
        }
    }
}

// An mma.async reads three uniform registers: urA and urB in its issue round
// (issue + 1), urC in a second round at issue + 2 (the TMA's round counter,
// src/rf.hpp uniform_read_rounds). At the canonical depths it still reaches
// the MMA at issue + 3 (issue_to_dispatch). A uniform reader right behind it
// is refused once ("uniform read ports held") and issues at issue + 2; so is
// a second matmul. A vector instruction is not refused, nor is a uniform
// write that reads nothing -- even one that rewrites urC at issue + 1: it
// lands at issue + 5, after the second round read the old pointer.
void test_mma_three_uniform_reads() {
    const auto& profile = canonical_profile();
    nanosm::Instruction mma;
    mma.opcode = nanosm::mma_opcode(0, true);
    mma.src0 = 1;
    mma.src1 = 2;
    mma.src2 = 3;
    require(nanosm::uniform_read_rounds(mma) == std::vector<std::vector<uint8_t>>{{1, 2}, {3}},
        "an mma.async reads urA, urB, then urC");
    require(nanosm::issue_to_dispatch(profile) == 3, "canonical: an mma.async reaches the MMA at issue + 3");
    const auto launch = [] {
        nanosm::Launch result;
        result.active_warps = 1;
        i9_write_operands(result);
        return result;
    };
    const std::string setup = R"(
        umov ur0, 0
        umov ur1, 2048
        umov ur2, 0
        umov ur3, 5
        umov ur4, 8192
        mbar.init [ur4], 1
    )";
    const std::string tail = R"(
        mma.commit [ur4]
    wait:
        mbar.try_wait p0, [ur4], 0
        @!p0 bra.uni wait
        halt
    )";
    // pcs: the setup is 6 instructions, so the matmul is pc 48.
    nanosm::Machine refused(profile, compile(setup + R"(
        mma.async.init.m64n64k8 [ur2], [ur0], [ur1]
        uaddi ur5, ur3, 1
        mma.async.init.m64n64k8 [ur2], [ur0], [ur1]
    )" + tail, profile), launch(), {.timeline = true});
    refused.run(40000);
    const auto first = issued_at(refused, 48).at(0);
    require(issued_at(refused, 56).at(0) == first + 2, "the uniform reader behind the matmul was not refused once");
    require(issued_at(refused, 64).at(0) == first + 3, "the second matmul did not issue the cycle after the reader");
    std::vector<uint64_t> accepts;
    for (const auto& event : refused.trace().events()) {
        if (event.category == nanosm::TraceCategory::mma && event.name.starts_with("mma.async")) {
            accepts.push_back(event.start_cycle);
        }
    }
    require(accepts == std::vector<uint64_t>{first + 3, first + 6}, "the matmuls did not reach the MMA at issue + 3");
    // Refused for the ports: the reader at T + 1 and the commit at T + 4 (the
    // second matmul's later round); the commit then waits for MMA queue room.
    require(refused.trace().summary().urf_read_ports_held == 2 && refused.read_ureg(0, 5) == 6,
        "two refused warp-cycles, and the reader's result");
    require(nanosm::bits_float(refused.tmem_word(i9_tmem_word(0, 64, 5, 7))) == i9_reference(5, 7),
        "the matmuls' result");

    nanosm::Machine free_ports(profile, compile(setup + R"(
        mma.async.init.m64n64k8 [ur2], [ur0], [ur1]
        umov ur2, 4096
        mov r1, 7
    )" + tail, profile), launch(), {.timeline = true});
    free_ports.run(40000);
    const auto matmul = issued_at(free_ports, 48).at(0);
    require(issued_at(free_ports, 56).at(0) == matmul + 1 && issued_at(free_ports, 64).at(0) == matmul + 2,
        "the uniform write that reads nothing or the vector mov waited");
    require(free_ports.trace().summary().urf_read_ports_held == 0, "nothing was refused");
    for (uint32_t row = 0; row < 64; row += 9) {
        require(nanosm::bits_float(free_ports.tmem_word(i9_tmem_word(0, 64, row, 3))) == i9_reference(row, 3) &&
                free_ports.tmem_word(i9_tmem_word(4096, 64, row, 3)) == 0,
            "the matmul did not read the old urC (0) in its second round");
    }
}

// Where a pointer may point (src/layout.hpp). The machine addresses any
// pointer by the layout rule; a region with a word outside tmem is its PANIC
// (both builds), as an operand outside shmem is.
void test_tmem_region_outside_tmem_panics() {
    const auto& profile = canonical_profile();
    const auto one_warp = [] {
        nanosm::Launch launch;
        launch.active_warps = 1;
        return launch;
    };
    nanosm::Machine load(profile, compile("umov ur8, 7232\nacc.ld r1, [ur8, 64]\nhalt\n", profile), one_warp());
    expect_sim_panic(load, "accumulator transfer region outside tmem", "an acc.ld whose rows 113.. leave tmem");
    nanosm::Machine matmul(profile, compile(
        "umov ur0, 0\numov ur1, 2048\numov ur2, 6144\nmma.async.init.m64n64k8 [ur2], [ur0], [ur1]\nhalt\n", profile),
        one_warp());
    expect_sim_panic(matmul, "MMA result region is outside tmem", "a 64-row result at row 96");
}

// The pointer rule is the sanitizer's PANIC: a pointer whose row is not a
// multiple of 4, an acc.ld.v8 whose eight columns leave the row, a matmul
// result not at column 0.
void test_tmem_pointer_rule_panics() {
    const auto& profile = canonical_profile();
    const auto one_warp = [] {
        nanosm::Launch launch;
        launch.active_warps = 1;
        return launch;
    };
    nanosm::Machine row(profile, compile("umov ur8, 128\nacc.ld.v8 r16, [ur8, 64]\nhalt\n", profile), one_warp());
    expect_sim_panic(row, "acc.ld.v8: tmem pointer 128 with 64 words per row is row 2 column 0", "a pointer at row 2");
    nanosm::Machine columns(profile, compile("umov ur8, 60\nacc.st.v8 [ur8, 64], r16\nhalt\n", profile), one_warp());
    expect_sim_panic(columns, "the 8 columns it touches stay inside its row", "eight columns from column 60");
    nanosm::Machine result(profile, compile(
        "umov ur0, 0\numov ur1, 2048\numov ur2, 32\nmma.async.init.m32n64k8 [ur2], [ur0], [ur1]\nhalt\n", profile),
        one_warp());
    expect_sim_panic(result, "tmem pointer 32 with 64 words per row is row 0 column 32", "a matmul result at column 32");
}

} // namespace

// tmem metadata and round trip
TEST(Mma, tmem_profile_metadata_and_round_trip) {
    test_tmem_profile_metadata_and_round_trip();
}

// tmem ordering and store lifetime
TEST(Mma, tmem_transfer_ordering_and_store_lifetime) {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    test_tmem_transfer_ordering_and_store_lifetime();
}

// MMA raw-window numerics
TEST(Mma, mma_raw_window_numerics) {
    test_mma_raw_window_numerics();
}

// MMA fusion boundaries
TEST(Mma, mma_fusion_boundaries) {
    test_mma_fusion_boundaries();
}

// autonomous MMA GEMM and commit
TEST(MmaSlow, autonomous_mma_gemm_and_commit) {
    test_autonomous_mma_gemm_and_commit();
}

// worker acc.store mbar handoff
TEST(Mma, worker_acc_store_mbar_handoff) {
    test_worker_acc_store_mbar_handoff();
}

// mbar engines: TMA event does not stall MMA queue
TEST(Mma, mbar_engine_tma_event_does_not_stall_mma_queue) {
    test_mbar_engine_tma_event_does_not_stall_mma_queue();
}

// mbar engines: tie alternates, TMA first
TEST(Mma, mbar_engine_events_alternate_tma_first) {
    test_mbar_engine_events_alternate_tma_first();
}

// mbar engines: real TMA and commit in one cycle
TEST(MmaSlow, mbar_engine_real_tma_and_commit_same_cycle) {
    test_mbar_engine_real_tma_and_commit_same_cycle();
}

// MMA dense profile recurrence bubbles
TEST(MmaSlow, mma_dense_profile_recurrence_bubbles) {
    test_mma_dense_profile_recurrence_bubbles();
}

// MMA bare transfer race panics
TEST(Mma, mma_bare_transfer_race_panics) {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    test_mma_bare_transfer_race_panics();
}

// MMA multi-round fetch timing
TEST(Mma, mma_multi_round_fetch_timing) {
    test_mma_multi_round_fetch_timing();
}

// MMA varying round service
TEST(Mma, mma_varying_round_service) {
    test_mma_varying_round_service();
}

// I7: one eight-product group at the 4x4x8 boundary
TEST(Mma, i7_dot8_group_boundary) {
    test_i7_dot8_group_boundary();
}

// I7: hold A / stream B walker
TEST(Mma, i7_hold_a_stream_b_walker) {
    test_i7_hold_a_stream_b_walker();
}

// I7: back-to-back MMA throughput
TEST(Mma, i7_back_to_back_throughput) {
    test_i7_back_to_back_throughput();
}

// I8: MMA beside acc transfers
TEST(Mma, i8_mma_beside_transfers) {
    test_i8_mma_beside_transfers();
}

// I8: one acc.ld and one acc.st engine
TEST(Mma, i8_one_engine_of_each_kind) {
    test_i8_one_engine_of_each_kind();
}

// I8: tmem depths (Dt/Lt 1/1, 2/2, 3/1)
TEST(Mma, i8_tmem_depths) {
    test_i8_tmem_depths();
}

// I8: same word, same cycle (READ_FIRST)
TEST(Mma, i8_read_first_same_word) {
    test_i8_read_first_same_word();
}

// I8: a transfer racing an MMA is quiet without the sanitizer
TEST(Mma, i8_transfer_racing_mma_is_quiet_without_the_sanitizer) {
    test_i8_transfer_racing_mma_is_quiet_without_the_sanitizer();
}

// I9: MMA opcodes per shape and init
TEST(Mma, i9_mma_opcodes) {
    test_i9_mma_opcodes();
}

// I9: profile shape list; unsupported shape and accumulator mismatch refused
TEST(Mma, i9_profile_shapes_and_refusals) {
    test_i9_profile_shapes_and_refusals();
}

// I9: bank census for both shapes
TEST(Mma, i9_bank_census) {
    test_i9_bank_census();
}

// I9: m32n64k8 walker
TEST(Mma, i9_m32_walker) {
    test_i9_m32_walker();
}

// I9: two m32 halves equal one m64
TEST(Mma, i9_two_m32_halves_equal_one_m64) {
    test_i9_two_m32_halves_equal_one_m64();
}

// I9: mixed m64 / m32 queue
TEST(Mma, i9_mixed_queue) {
    test_i9_mixed_queue();
}

// I9: four 8 KiB accumulators, row bases 0 and 16
TEST(Mma, i9_four_8k_accumulators) {
    test_i9_four_8k_accumulators();
}

// I9: sanitizer operand span follows the shape
TEST(Mma, i9_operand_span_follows_the_shape) {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    test_i9_operand_span_follows_the_shape();
}

// D15: more MMA shapes run (k > core_k, 16 and 128 rows)
TEST(Mma, d15_more_shapes_run) {
    test_d15_more_shapes_run();
}

// D15: the accumulator latency bound at equality
TEST(Mma, d15_latency_bound) {
    test_d15_latency_bound();
}

// D15: a partly filled K block warns and counts its whole storage
TEST(Mma, d15_partial_k_block_footprint_and_warning) {
    test_d15_partial_k_block_footprint_and_warning();
}

// acc pointer: the tmem layout against an independent statement of the rule
TEST(Mma, tmem_layout_matches_the_rule) {
    test_tmem_layout_matches_the_rule();
}

// acc pointer: acc.st / acc.ld through pointers at a non-zero row and column
TEST(Mma, acc_pointer_at_a_row_and_column) {
    test_acc_pointer_at_a_row_and_column();
}

// acc pointer: one worker code reads a 64-row view over two m32 results
TEST(Mma, acc_view_over_two_matmul_results) {
    test_acc_view_over_two_matmul_results();
}

// acc pointer: an mma.async reads three uniform registers in two rounds
TEST(Mma, mma_three_uniform_reads) {
    test_mma_three_uniform_reads();
}

// acc pointer: a region outside tmem is the machine's PANIC
TEST(Mma, tmem_region_outside_tmem_panics) {
    test_tmem_region_outside_tmem_panics();
}

// acc pointer: the pointer rule is the sanitizer's PANIC
TEST(Mma, tmem_pointer_rule_panics) {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    test_tmem_pointer_rule_panics();
}
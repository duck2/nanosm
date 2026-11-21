// The LSU (rfc09): vector and scalar gmem access, the staging pool, the handoff edges.
// Moved from the original tests/tests.cpp (trash/test-setup-2026-10-22.md);
// each test's body is unchanged, its TEST wrapper is at the end.
#include "test_support.hpp"

namespace {

using namespace nanosm_test;

struct LsuEventSpan {
    std::string name;
    std::string detail;
    uint64_t start = 0;
    uint64_t end = 0;
};

std::vector<LsuEventSpan> trace_spans(
    const nanosm::Machine& machine, nanosm::TraceCategory category, const std::string& name) {
    std::vector<LsuEventSpan> spans;
    for (const auto& event : machine.trace().events()) {
        if (event.kind != nanosm::TraceEventKind::slice || event.category != category) continue;
        if (!name.empty() && event.name != name) continue;
        spans.push_back({event.name, event.detail, event.start_cycle, event.start_cycle + event.duration_cycles});
    }
    std::sort(spans.begin(), spans.end(), [](const auto& a, const auto& b) { return a.start < b.start; });
    return spans;
}

void test_vadd() {
    // kernels/vadd.qtx: 8192 fp32 per vector, four warps, ldg.v8/fadd/stg.v8.
    constexpr uint32_t n = 8192;
    constexpr uint32_t b = 32768;
    constexpr uint32_t c = 65536;
    auto config = canonical_profile();
    nanosm::Launch launch;
    launch.active_warps = 4;
    auto& gmem = image_bytes(launch.gmem_image, c + n * 4 + 64);
    for (uint32_t i = 0; i < n; ++i) {
        write_u32(gmem, i * 4, nanosm::float_bits(static_cast<float>(i % 97)));
        write_u32(gmem, b + i * 4, nanosm::float_bits(static_cast<float>(2 * (i % 89))));
    }
    std::fill(gmem.begin() + c + n * 4, gmem.end(), 0xA5);
    nanosm::Machine machine(
        config, nanosm::compile_and_assemble_file("kernels/vadd.qtx", config), std::move(launch));
    machine.run();
    for (uint32_t i = 0; i < n; ++i) {
        require(
            nanosm::bits_float(machine.read_gmem_u32(c + i * 4)) ==
                static_cast<float>(i % 97 + 2 * (i % 89)),
            "vadd output mismatch at " + std::to_string(i));
    }
    for (uint32_t byte = c + n * 4; byte < c + n * 4 + 64; ++byte)
        require(machine.gmem()[byte] == 0xA5, "vadd wrote past C");
    const auto& summary = machine.trace().summary();
    require(summary.lsu_vector_loads == 128 && summary.lsu_vector_stores == 64 &&
            summary.lsu_sectors == 3 * n * 4 / 32 && summary.lsu_bytes == 3 * n * 4 &&
            summary.lsu_useful_bytes == summary.lsu_bytes && summary.gmem_bytes == 3 * n * 4,
        "vadd traffic is not exactly 3 x 32 KiB of whole sectors");
}

void test_sparse_launch_images() {
    nanosm::Launch launch;
    launch.active_warps = 1;
    launch.shmem_image.push_back({64, std::vector<uint8_t>(4, 0)});
    launch.gmem_image.push_back({128, std::vector<uint8_t>(4, 0)});
    write_u32(launch.shmem_image.back().bytes, 0, 0x11223344u);
    write_u32(launch.gmem_image.back().bytes, 0, 0x55667788u);
    nanosm::Machine machine(canonical_profile(), compile("halt"), std::move(launch));
    machine.run();
    require(machine.read_shmem_u32(64) == 0x11223344u, "sparse shmem image landed incorrectly");
    require(machine.read_gmem_u32(128) == 0x55667788u, "sparse gmem image landed incorrectly");
    require(machine.read_shmem_u32(0) == 0, "unspecified shmem was not zero");
    require(machine.read_gmem_u32(0) == 0, "unspecified gmem was not zero");
}

// Replaces "gmem coalescing" (16 contiguous lanes -> 2 sectors): there is no
// coalescer, so the same access is sixteen requests.
void test_lsu_scalar_ldg_is_one_request_per_lane() {
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& gmem = image_bytes(launch.gmem_image, 64);
    fill_pattern(gmem, 0, 16, 0);
    nanosm::Machine machine(canonical_profile(), compile(R"(
        sread r0, LANE_ID
        shli  r0, r0, 2
        ldg   r1, [r0]
        halt
    )"), std::move(launch));
    machine.run();
    for (uint32_t lane = 0; lane < 16; ++lane)
        require(machine.read_reg(0, lane, 1) == pattern_word(lane), "scalar ldg lane value mismatch");
    const auto& summary = machine.trace().summary();
    require(summary.gmem_sectors == 16 && summary.lsu_sectors == 16 && summary.lsu_scalar_loads == 1,
        "16 contiguous lanes were merged instead of issuing 16 sector requests");
    require(summary.lsu_useful_bytes == 64 && summary.lsu_bytes == 512,
        "scalar ldg useful/sector byte accounting is wrong");
}

// (a) Replaces "gmem v8 roundtrip" (row-per-lane): register rD+j, lane l
// <-> byte rA + 4*(16*j + l), both directions, with guard bytes.
void test_lsu_v8_interleaved_mapping() {
    constexpr uint32_t source = 1024;
    constexpr uint32_t target = 4096;
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& gmem = image_bytes(launch.gmem_image, 8192);
    fill_pattern(gmem, source, 128, 0);
    std::fill(gmem.begin() + target - 64, gmem.begin() + target + 512 + 64, 0xA5);
    std::string source_text = R"(
        mov r30, 1024
        ldg.v8 r8, [r30]
        sread r1, LANE_ID
        shli r2, r1, 8
    )";
    for (uint32_t j = 0; j < 8; ++j)
        source_text += "addi r" + std::to_string(16 + j) + ", r2, " + std::to_string(0x100000 + j * 0x10000) + "\n";
    source_text += "mov r31, 4096\nstg.v8 [r31], r16\nhalt\n";
    nanosm::Machine machine(canonical_profile(), compile(source_text), std::move(launch));
    machine.run();
    for (uint32_t j = 0; j < 8; ++j) {
        for (uint32_t lane = 0; lane < 16; ++lane) {
            // Written-out contract, independent of the implementation.
            const uint32_t word = 16 * j + lane;
            require(machine.read_reg(0, lane, 8 + j) == pattern_word(word),
                "ldg.v8 r" + std::to_string(8 + j) + " lane " + std::to_string(lane) +
                    " is not word 16*j + l of the block");
            require(machine.read_gmem_u32(target + 4 * word) == lane * 256 + 0x100000 + j * 0x10000,
                "stg.v8 r" + std::to_string(16 + j) + " lane " + std::to_string(lane) +
                    " did not land at word 16*j + l");
        }
    }
    for (uint32_t byte = target - 64; byte < target; ++byte)
        require(machine.gmem()[byte] == 0xA5, "stg.v8 damaged the guard before its block");
    for (uint32_t byte = target + 512; byte < target + 576; ++byte)
        require(machine.gmem()[byte] == 0xA5, "stg.v8 damaged the guard after its block");
    const auto& summary = machine.trace().summary();
    require(summary.lsu_vector_loads == 1 && summary.lsu_vector_stores == 1 &&
            summary.lsu_sectors == 32 && summary.gmem_sectors == 32 && summary.lsu_useful_bytes == 1024,
        "a v8 command is not exactly sixteen whole sectors");
}

// Replaces the old "gmem v8 source lifetime" with a uniform base.
void test_lsu_v8_store_source_lifetime() {
    const std::string text = R"(
        mov r0, 0
        mov r8, 17
        mov r9, 18
        mov r10, 19
        mov r11, 20
        mov r12, 21
        mov r13, 22
        mov r14, 23
        mov r15, 24
        stg.v8 [r0], r8
        mov r8, 99
        halt
    )";
    const auto scheduled = nanosm::compile_qtx(text, canonical_profile());
    require(scheduled.instructions[10].control.wait_mask != 0,
        "qtxc did not protect the deferred stg.v8 source tuple");
    nanosm::Launch launch;
    launch.active_warps = 1;
    image_bytes(launch.gmem_image, 512);
    nanosm::Machine machine(canonical_profile(), compile(text), std::move(launch));
    machine.run();
    for (uint32_t j = 0; j < 8; ++j)
        for (uint32_t lane = 0; lane < 16; ++lane)
            require(machine.read_gmem_u32(4 * (16 * j + lane)) == 17 + j,
                "younger overwrite changed a deferred stg.v8 source");
    require(machine.read_reg(0, 3, 8) == 99, "the younger overwrite was lost");
}

void test_gmem_v8_predication_is_rejected() {
    try {
        (void)compile("@p0 stg.v8 [r0], r8\nhalt\n");
        throw std::runtime_error("predicated stg.v8 was accepted (a warp-wide store cannot be masked)");
    } catch (const nanosm::SimError& error) {
        require(std::string(error.what()).find("stg.v8 cannot be masked (@p)") != std::string::npos,
            "predicated stg.v8 produced the wrong rejection: " + std::string(error.what()));
    }
}

// Replaces "gmem v8 shared return": LSU RF writes are full-width beats through
// the reservations, at most one per cycle machine-wide, never single lanes.
void test_lsu_rf_writes_are_full_width_beats() {
    auto config = canonical_profile();
    config.gmem_latency_dram_ticks = 0;
    config.gmem_bus_bytes_per_dram_tick = config.gmem_sector_bytes;
    config.dram_clock_hz = config.core_clock_hz * 64;
    nanosm::Launch launch;
    launch.active_warps = 2;
    auto& image = image_bytes(launch.gmem_image, 1024);
    fill_pattern(image, 0, 256, 0);
    nanosm::Machine machine(config, compile(R"(
        usread ur0, WARP_ID
        ur2r r0, ur0
        shli r0, r0, 9
        ldg.v8 r8, [r0]
        halt
    )", config), std::move(launch));
    std::vector<uint32_t> previous(2 * 8 * 16, 0);
    uint32_t writes = 0;
    while (!machine.done()) {
        tick_capped(machine);
        uint32_t registers_changed = 0;
        for (uint32_t warp = 0; warp < 2; ++warp) {
            for (uint32_t j = 0; j < 8; ++j) {
                uint32_t lanes_changed = 0;
                for (uint32_t lane = 0; lane < 16; ++lane) {
                    auto& old = previous[(warp * 8 + j) * 16 + lane];
                    const auto value = machine.read_reg(warp, lane, 8 + j);
                    if (value != old) ++lanes_changed;
                    old = value;
                }
                require(lanes_changed == 0 || lanes_changed == 16, "an LSU RF write touched a subset of lanes");
                if (lanes_changed == 16) ++registers_changed;
            }
        }
        require(registers_changed <= 1, "the LSU wrote more than one RF beat in a cycle");
        writes += registers_changed;
    }
    require(writes == 16, "two ldg.v8 commands did not make sixteen RF write beats");
    for (uint32_t warp = 0; warp < 2; ++warp)
        for (uint32_t j = 0; j < 8; ++j)
            for (uint32_t lane = 0; lane < 16; ++lane)
                require(machine.read_reg(warp, lane, 8 + j) == pattern_word(warp * 128 + 16 * j + lane),
                    "overlapping ldg.v8 commands lost or swapped a word");
}

// (b) Replaces "gmem v8 invalid footprints": the v8/uni/scalar contracts.
void test_lsu_contract_panics() {
    {
        auto profile = canonical_profile();
        profile.timing.alu = 4;
        const auto scheduled = nanosm::compile_qtx("mov r0, 0\nldg.v8 r8, [r0]\nhalt\n", profile);
        require(scheduled.instructions.front().control.stall != 0,
            "qtxc did not schedule the ldg.v8 address RAW dependency");
    }
    const auto run = [](const std::string& text, uint64_t gmem_bytes, const std::string& needle,
                        const std::string& what) {
        auto config = canonical_profile();
        config.gmem_bytes = gmem_bytes;
        nanosm::Launch launch;
        launch.active_warps = 1;
        nanosm::Machine machine(config, compile(text, config), std::move(launch));
        expect_sim_panic(machine, needle, what);
    };
    // (The non-uniform ldg.v8 / stg.v8 bases moved to "uniformity: hardware
    // uses lane 0": since I1 they are sanitizer panics, and this test runs in
    // both builds. ldg.uni's base is a uniform register.)
    run("mov r0, 32\nldg.v8 r8, [r0]\nhalt\n", 4096,
        "ldg.v8 address 32 is not 64-byte aligned", "rA = 32 mod 64 ldg.v8");
    run("mov r0, 64\nstg.v8 [r0 + 32], r8\nhalt\n", 4096,
        "stg.v8 address 96 is not 64-byte aligned", "rA + imm = 32 mod 64 stg.v8");
    run("sread r0, LANE_ID\nshli r0, r0, 2\nldg r1, [r0 + 2]\nhalt\n", 4096,
        "unaligned 32-bit gmem access", "2-byte aligned scalar ldg");
    run("umov ur0, 6\nldg.uni r1, [ur0]\nhalt\n", 4096,
        "unaligned 32-bit gmem access", "2-byte aligned ldg.uni");
    run("mov r0, 576\nldg.v8 r8, [r0]\nhalt\n", 1024, "out of bounds", "out-of-bounds ldg.v8 block");
}

// Replaces "gmem command capacity" (a footprint larger than the controller
// queue used to PANIC): commands now stream through any queue >= 1 entry.
void test_lsu_commands_stream_through_small_queue() {
    auto config = canonical_profile();
    config.tma_queue_commands = 0;
    config.tma_sector_credits = 0;
    config.gmem_queue_sectors = 2;
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& gmem = image_bytes(launch.gmem_image, 2048);
    fill_pattern(gmem, 0, 128, 0);
    nanosm::Machine machine(config, compile(R"(
        mov r0, 0
        ldg.v8 r8, [r0]
        stg.v8 [r0 + 1024], r8
        sread r1, LANE_ID
        shli r1, r1, 2
        ldg r16, [r1]
        halt
    )", config), std::move(launch));
    machine.run();
    for (uint32_t word = 0; word < 128; ++word)
        require(machine.read_gmem_u32(1024 + 4 * word) == pattern_word(word), "streamed v8 copy mismatch");
    for (uint32_t lane = 0; lane < 16; ++lane)
        require(machine.read_reg(0, lane, 16) == pattern_word(lane), "streamed scalar ldg mismatch");
    const auto& summary = machine.trace().summary();
    require(summary.lsu_request_wait_cycles != 0, "a two-entry queue never lowered request-ready");
    require(summary.gmem_max_queue_depth <= 2, "the controller queue exceeded its capacity");
}

// Replaces "gmem queue pressure": the LSU command slot is the structural
// wait point at issue.
void test_lsu_command_slot_structurally_stalls() {
    auto config = canonical_profile();
    config.lsu_active_commands = 1;
    nanosm::Launch launch;
    launch.active_warps = 2;
    image_bytes(launch.gmem_image, 128);
    nanosm::Machine machine(config, compile(R"(
        sread r0, THREAD_ID
        shli  r0, r0, 2
        ldg   r1, [r0]
        halt
    )", config), std::move(launch));
    machine.run();
    const auto& summary = machine.trace().summary();
    require(summary.lsu_no_slot_cycles != 0 && summary.lsu_no_staging_cycles == 0,
        "one command slot never made the second ldg ineligible");
    require(summary.warp_cycles[static_cast<std::size_t>(nanosm::WarpState::structural)] >=
            summary.lsu_no_slot_cycles,
        "LSU ineligibility was not a structural warp state");
}

// (c) Gathers: one request per lane, no merging, any lane-address relation;
// ldg.uni is the declared broadcast (one request).
void test_lsu_gather_requests() {
    const auto gather = [](const std::string& address_code, uint32_t expected_sectors,
                           const std::function<uint32_t(uint32_t)>& word_of_lane,
                           nanosm::ControllerTestOrder order = nanosm::ControllerTestOrder::fifo) {
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& gmem = image_bytes(launch.gmem_image, 16384);
        fill_pattern(gmem, 0, 4096, 0);
        nanosm::Machine machine(canonical_profile(),
            compile("sread r1, LANE_ID\n" + address_code + "halt\n"), std::move(launch),
            {.timeline = true});
        machine.set_controller_test_order(order);
        machine.run();
        for (uint32_t lane = 0; lane < 16; ++lane)
            require(machine.read_reg(0, lane, 10) == pattern_word(word_of_lane(lane)),
                "gathered word mismatch in lane " + std::to_string(lane));
        require(machine.trace().summary().lsu_sectors == expected_sectors,
            "gather issued " + std::to_string(machine.trace().summary().lsu_sectors) +
                " sector requests, expected " + std::to_string(expected_sectors));
        std::vector<uint64_t> served;
        for (const auto& span : trace_spans(machine, nanosm::TraceCategory::gmem, "read sector")) {
            const auto at = span.detail.find("address=");
            served.push_back(std::stoull(span.detail.substr(at + 8)));
        }
        return served;
    };
    // Sixteen distinct sectors: lane*96 + 4*(lane & 7).
    const std::string distinct = R"(
        shli r2, r1, 5
        shli r3, r1, 6
        add r4, r2, r3
        andi r5, r1, 7
        shli r5, r5, 2
        add r6, r4, r5
        ldg r10, [r6]
    )";
    const auto distinct_word = [](uint32_t lane) { return (lane * 96 + 4 * (lane & 7)) / 4; };
    const auto in_order = gather(distinct, 16, distinct_word);
    require(std::is_sorted(in_order.begin(), in_order.end()),
        "the placeholder FIFO controller reordered gather returns");
    // Same gather with the legal free-ID reorder: returns come back out of
    // request order and every lane still gets its own word.
    const auto reordered = gather(distinct, 16, distinct_word, nanosm::ControllerTestOrder::reverse_free_ids);
    require(!std::is_sorted(reordered.begin(), reordered.end()),
        "the reverse_free_ids controller did not reorder gather returns");
    // Two lanes per sector (lanes 2k, 2k+1): still sixteen requests.
    gather(R"(
        shli r7, r1, 2
        shri r8, r1, 1
        shli r8, r8, 8
        add r9, r7, r8
        ldg r10, [r9 + 4096]
    )", 16, [](uint32_t lane) { return (4096 + 4 * lane + 256 * (lane >> 1)) / 4; });
    // All sixteen lanes in one sector: sixteen requests.
    gather(R"(
        andi r12, r1, 7
        shli r12, r12, 2
        ldg r10, [r12 + 8192]
    )", 16, [](uint32_t lane) { return (8192 + 4 * (lane & 7)) / 4; });
    // The same word for every lane, declared: one request, fanned out.
    gather("umov ur0, 0\nldg.uni r10, [ur0 + 8200]\n", 1, [](uint32_t) { return 8200 / 4; });
}

// (d) Scatters: sixteen masked sector writes; byte enables keep every
// unselected byte, including other lanes' words in a shared sector.
void test_lsu_scatter_byte_enables() {
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& gmem = image_bytes(launch.gmem_image, 2048);
    std::fill(gmem.begin(), gmem.end(), 0xA5);
    nanosm::Machine machine(canonical_profile(), compile(R"(
        sread r1, LANE_ID
        shli r6, r1, 6
        add r5, r1, r1
        add r5, r5, r1
        andi r5, r5, 7
        shli r5, r5, 2
        add r2, r6, r5
        addi r3, r1, 4096
        stg [r2], r3
        shli r7, r1, 2
        addi r9, r1, 8192
        stg [r7 + 1024], r9
        halt
    )"), std::move(launch));
    machine.run();
    std::vector<bool> written(2048, false);
    for (uint32_t lane = 0; lane < 16; ++lane) {
        const uint32_t own = lane * 64 + 4 * ((lane * 3) & 7);
        require(machine.read_gmem_u32(own) == 4096 + lane, "scatter lane word mismatch");
        require(machine.read_gmem_u32(1024 + 4 * lane) == 8192 + lane, "shared-sector scatter word mismatch");
        for (uint32_t byte = 0; byte < 4; ++byte) {
            written[own + byte] = true;
            written[1024 + 4 * lane + byte] = true;
        }
    }
    for (uint32_t byte = 0; byte < 2048; ++byte)
        if (!written[byte]) require(machine.gmem()[byte] == 0xA5, "a masked write damaged byte " + std::to_string(byte));
    const auto& summary = machine.trace().summary();
    require(summary.lsu_scalar_stores == 2 && summary.lsu_sectors == 32,
        "each scatter is not sixteen masked sector writes");
}

// (e) The staging pool is owned at admission. With one assembly register the
// second warp's ldg.v8 is ineligible until the first command's eighth RF
// write commits; with two it is admitted at once.
void test_lsu_staging_pool_admission() {
    const auto run = [](uint32_t staging) {
        auto config = canonical_profile();
        config.lsu_staging_vectors = staging;
        nanosm::Launch launch;
        launch.active_warps = 2;
        auto& gmem = image_bytes(launch.gmem_image, 1024);
        fill_pattern(gmem, 0, 256, 0);
        nanosm::Machine machine(config, compile(R"(
            usread ur0, WARP_ID
            ur2r r0, ur0
            shli r0, r0, 9
            ldg.v8 r8, [r0]
            halt
        )", config), std::move(launch), {.timeline = true});
        machine.run();
        for (uint32_t warp = 0; warp < 2; ++warp)
            for (uint32_t j = 0; j < 8; ++j)
                for (uint32_t lane = 0; lane < 16; ++lane)
                    require(machine.read_reg(warp, lane, 8 + j) == pattern_word(warp * 128 + 16 * j + lane),
                        "pool admission corrupted a vector load");
        const auto spans = trace_spans(machine, nanosm::TraceCategory::gmem, "ldg.v8");
        require(spans.size() == 2, "expected two ldg.v8 command events");
        return std::make_tuple(spans[0], spans[1], machine.trace().summary().lsu_no_staging_cycles,
            machine.trace().summary().lsu_no_slot_cycles);
    };
    // I6a: the second ldg.v8 issues in the cycle the first command's eighth RF
    // write releases the register (its grant: first.end) and reaches the LSU
    // issue_to_dispatch later (the command slice starts at dispatch).
    const auto dispatch = nanosm::issue_to_dispatch(canonical_profile());
    const auto [first, second, no_staging, no_slot] = run(1);
    require(second.start == first.end + dispatch,
        "with one assembly register the second ldg.v8 was admitted at cycle " + std::to_string(second.start) +
            ", not at the first command's eighth write (" + std::to_string(first.end) + ") + dispatch");
    // Warp 1 is tried every cycle from its ldg.v8 until admission.
    require(no_staging + 4 >= second.start - first.start,
        "no_staging did not count the ineligible cycles");
    require(no_staging != 0 && no_slot == 0, "the wait was not attributed to the staging pool");
    const auto [first2, second2, no_staging2, no_slot2] = run(2);
    require(second2.start <= first2.start + 2 && no_staging2 == 0 && no_slot2 == 0,
        "with two assembly registers the second ldg.v8 was not admitted at once");
}

// (f) The backpressure chain: RF write port blocked -> assembly register full
// -> (B2.1: no vector sector register) ingress FIFO head blocked -> FIFO full
// -> return-ready low ->
// the controller holds beats -> queue full -> request-ready low -> both
// command slots held -> the next ldg is ineligible. Nothing is lost.
void test_lsu_backpressure_chain() {
    const auto run = [](bool storm) {
        auto config = canonical_profile();
        config.tma_queue_commands = 0;
        config.tma_sector_credits = 0;
        config.gmem_queue_sectors = 8;
        config.lsu_ingress_sectors = 2;
        with_long_bookings(config);
        nanosm::Launch launch;
        launch.active_warps = 2;
        auto& gmem = image_bytes(launch.gmem_image, 4096);
        fill_pattern(gmem, 0, 1024, 0);
        // Warp 1 books RF bank 0 every cycle, booking_horizon cycles ahead
        // (I1: a ready assembly then waits out up to that many old bookings
        // per denial; a mov storm no longer starves it).
        std::string text = R"(
            usread ur0, WARP_ID
            ur2r r29, ur0
            setp.ne.u32 p0, r29, 0
            @p0 bra.uni other
            mov r30, 0
            mov r31, 512
            mov r28, 2048
            ldg.v8 r0, [r30]
            ldg.v8 r8, [r31]
            ldg r16, [r28]
            halt
        other:
        )";
        if (storm) text += booking_stream(0, 500);
        text += "halt\n";
        nanosm::Machine machine(config, compile(text, config), std::move(launch), {.timeline = true});
        bool linked = false;
        uint64_t ingress = 0, request = 0, slot = 0;
        while (!machine.done()) {
            tick_capped(machine);
            const auto& s = machine.trace().summary();
            if (s.lsu_ingress_full_cycles > ingress && s.lsu_request_wait_cycles > request &&
                s.lsu_no_slot_cycles > slot) linked = true;
            ingress = s.lsu_ingress_full_cycles;
            request = s.lsu_request_wait_cycles;
            slot = s.lsu_no_slot_cycles;
        }
        for (uint32_t j = 0; j < 16; ++j)
            for (uint32_t lane = 0; lane < 16; ++lane)
                require(machine.read_reg(0, lane, j) == pattern_word(16 * j + lane),
                    "a byte was lost or corrupted under backpressure");
        for (uint32_t lane = 0; lane < 16; ++lane)
            require(machine.read_reg(0, lane, 16) == pattern_word(512), "the stalled scalar ldg read the wrong word");
        return std::make_tuple(linked, machine.trace().summary());
    };
    const auto [linked, summary] = run(true);
    // I1: each of the six chunks bound for bank 0 (r0, r3, r6, r9, r12, r15)
    // waits out at most the bookings already made when it was denied (the
    // owed bit then inhibits warp 1): blocked, but never starved.
    require(summary.lsu_rf_write_wait_cycles >= 5 * booking_horizon, "the RF write port was never the blocked link");
    require(summary.lsu_rf_write_wait_cycles <= 6 * (booking_horizon + 1), "an assembly waited past the booking horizon");
    require(summary.lsu_ingress_full_cycles != 0, "return-ready never dropped (ingress FIFO never full)");
    require(summary.lsu_request_wait_cycles != 0, "request-ready never dropped");
    require(summary.lsu_no_slot_cycles != 0, "the next ldg was never ineligible");
    require(linked, "no cycle showed ingress full, request-ready low and the slot wait together");
    const auto [control_linked, control] = run(false);
    require(!control_linked && control.lsu_ingress_full_cycles == 0 && control.lsu_rf_write_wait_cycles == 0,
        "the control run (free RF write port) still filled the ingress FIFO");
}

// (g) TMA priority on port A: a returned TMA sector waits only for the SIMT
// access already in port A, never for one granted after it became ready.
// (h) Read streams: TMA (ID 0) and vector (ID 1) sectors interleave in the
// controller's global order at sector granularity; each lands correctly.
void test_lsu_tma_and_vector_streams_interleave() {
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.shmem_image, 4096);
    write_rank1_load(shmem, 0, 0, 128, 128);
    auto& gmem = image_bytes(launch.gmem_image, 1024);
    fill_pattern(gmem, 0, 256, 0);
    nanosm::Machine machine(canonical_profile(), compile(tma_operands_asm(10, 1024, 3, 128, 8) + R"(
        mov r30, 512
        tma.load.1d [ur10], [0, ur8], [ur3]
        ldg.v8 r8, [r30]
    wait:
        mbar.try_wait p1, [ur3], 0
        @!p1 bra.uni wait
        halt
    )"), std::move(launch), {.timeline = true});
    machine.run();
    for (uint32_t i = 0; i < 128; ++i)
        require(machine.read_shmem_u32(1024 + 4 * i) == pattern_word(i), "TMA stream data mismatch");
    for (uint32_t j = 0; j < 8; ++j)
        for (uint32_t lane = 0; lane < 16; ++lane)
            require(machine.read_reg(0, lane, 8 + j) == pattern_word(128 + 16 * j + lane),
                "vector stream data mismatch with TMA returns interleaved");
    uint32_t switches = 0;
    std::string previous;
    for (const auto& span : trace_spans(machine, nanosm::TraceCategory::gmem, "read sector")) {
        const auto owner = span.detail.substr(span.detail.find("owner="));
        if (!previous.empty() && owner != previous) ++switches;
        previous = owner;
    }
    require(switches >= 8, "TMA and vector returns did not interleave at sector granularity (" +
        std::to_string(switches) + " owner switches)");
}

// The boundary tripwire: a controller that returns an ordered ID out of
// request order is a contract violation the SM refuses, by PANIC.
void test_lsu_controller_contract_tripwire() {
    nanosm::Launch launch;
    launch.active_warps = 1;
    image_bytes(launch.gmem_image, 512);
    nanosm::Machine machine(canonical_profile(), compile("mov r0, 0\nldg.v8 r8, [r0]\nhalt\n"), std::move(launch));
    machine.set_controller_test_order(nanosm::ControllerTestOrder::illegal_ordered_reorder);
    expect_sim_panic(machine, "contract violation: read stream 1 (vector) returned request",
        "an out-of-order vector stream");
}

void test_lsu_profile_and_encoding() {
    const auto profile = canonical_profile();
    require(profile.lsu_active_commands == 2 && profile.lsu_staging_vectors == 2 &&
            profile.lsu_read_streams == 2 && profile.lsu_pluck_words_per_cycle == 2 &&
            profile.lsu_ingress_sectors == 8 && profile.lsu_egress_sectors == 8 &&
            profile.lsu_sectors_in_flight == 16 && profile.lsu_ingress_latency_cycles == 1 &&
            profile.lsu_egress_latency_cycles == 1,
        "canonical lsu profile values changed");
    auto other = profile;
    other.lsu_staging_vectors = 1;
    require(nanosm::device_profile_fingerprint(profile) != nanosm::device_profile_fingerprint(other),
        "lsu sizing was omitted from the profile fingerprint");
    for (const auto field : {&nanosm::DeviceProfile::lsu_ingress_latency_cycles,
                             &nanosm::DeviceProfile::lsu_egress_latency_cycles}) {
        auto latency = profile;
        latency.*field = 3;
        require(nanosm::device_profile_fingerprint(profile) != nanosm::device_profile_fingerprint(latency),
            "an lsu FIFO latency was omitted from the profile fingerprint");
    }
    std::ifstream input("profiles/nanosm-v0.json");
    const auto document = nlohmann::json::parse(input);
    const auto rejects = [&](const std::function<void(nlohmann::json&)>& edit, const std::string& needle) {
        auto changed = document;
        edit(changed);
        try {
            (void)nanosm::parse_device_profile(changed.dump());
        } catch (const nanosm::ProfileError& error) {
            require(std::string(error.what()).find(needle) != std::string::npos,
                std::string("wrong lsu profile rejection: ") + error.what());
            return;
        }
        throw std::runtime_error("invalid lsu profile accepted: " + needle);
    };
    rejects([](auto& j) { j.erase("lsu"); }, "missing 'lsu'");
    // B2.1: read_streams generalized (stream 0 TMA, 1..n-1 vector), >= 2.
    rejects([](auto& j) { j["lsu"]["read_streams"] = 1; }, "read_streams must be in 2..64");
    rejects([](auto& j) { j["lsu"]["ingress_sectors"] = 1; }, "at least two sectors");
    rejects([](auto& j) { j["lsu"]["egress_sectors"] = 1; }, "at least two sectors");
    rejects([](auto& j) { j["lsu"]["staging_vectors"] = 0; }, "must be nonzero");
    // B2.1: a BRAM FIFO is never fall-through; latency 0 is rejected.
    rejects([](auto& j) { j["lsu"]["ingress_latency_cycles"] = 0; }, "latency must be in 1..64");
    rejects([](auto& j) { j["lsu"]["egress_latency_cycles"] = 0; }, "latency must be in 1..64");
    rejects([](auto& j) { j["lsu"].erase("egress_latency_cycles"); }, "egress_latency_cycles");
    // active_commands > staging_vectors is a legal profile: the pool makes
    // the extra slot ineligible (no_staging), the sanitizer asserts the bound.
    auto loose = document;
    loose["lsu"]["staging_vectors"] = 1;
    (void)nanosm::parse_device_profile(loose.dump());
    auto streams = document;
    streams["lsu"]["read_streams"] = 3;
    require(nanosm::parse_device_profile(streams.dump()).lsu_read_streams == 3, "read_streams = 3 rejected");

    const auto kernel = compile("umov ur3, 0\nldg.uni r5, [ur3 + 8]\nhalt\n");
    const auto decoded = nanosm::decode(kernel.words[1]);
    require(decoded.opcode == nanosm::Opcode::ldg_uni && decoded.dst == 5 && decoded.src0 == 3 &&
            decoded.imm == 8 && decoded.control.set_depbar < 6,
        "ldg.uni did not survive qtxc + qtxas");
    require(nanosm::format_instruction(decoded, false) == "ldg.uni r5, [ur3 + 8]", "ldg.uni formatting changed");
    require(nanosm::decode(nanosm::encode(decoded)).opcode == nanosm::Opcode::ldg_uni, "ldg.uni encoding round trip");
}

// One TMA command at the origin (its tile at shmem `shared` in ur10,
// completing into the mbar at `mbar_address` in ur3, its coordinate ur8),
// then a wait for that mbar.
std::string tma_and_wait(bool store, uint32_t descriptor_address, uint32_t shared, uint32_t mbar_address) {
    const auto descriptor = std::to_string(descriptor_address);
    return tma_operands_asm(10, shared, 3, mbar_address, 8) +
        (store ? "tma.store.1d [" + descriptor + ", ur8], [ur10], [ur3]\n"
               : "tma.load.1d [ur10], [" + descriptor + ", ur8], [ur3]\n") +
        "wait:\nmbar.try_wait p1, [ur3], 0\n@!p1 bra.uni wait\nhalt\n";
}

struct EdgeLoad {
    uint32_t bytes = 0;
    uint32_t events = 0;
};

std::map<std::pair<uint64_t, LsuEdge>, EdgeLoad> edge_load(const nanosm::Machine& machine) {
    std::map<std::pair<uint64_t, LsuEdge>, EdgeLoad> load;
    for (const auto& event : machine.edge_events()) {
        auto& entry = load[{event.cycle, event.edge}];
        entry.bytes += event.bytes;
        ++entry.events;
    }
    return load;
}

uint32_t max_bytes_per_cycle(const nanosm::Machine& machine, LsuEdge edge) {
    uint32_t most = 0;
    for (const auto& [key, entry] : edge_load(machine))
        if (key.second == edge) most = std::max(most, entry.bytes);
    return most;
}

// Every edge of the 3a table, every cycle of the run: never more than its
// declared width.
void check_edge_widths(const nanosm::Machine& machine, const nanosm::DeviceProfile& config,
                       const std::string& what) {
    const uint32_t pluck = config.lsu_pluck_words_per_cycle * 4;
    std::map<uint64_t, uint32_t> beats_per_tick;
    std::map<uint64_t, uint64_t> slot_client;
    for (const auto& event : machine.edge_events()) {
        if (event.edge == LsuEdge::bus_read_beat || event.edge == LsuEdge::bus_write_beat) {
            require(event.bytes <= config.gmem_bus_bytes_per_dram_tick, what + ": a beat wider than the bus");
            require(++beats_per_tick[event.dram_tick] == 1,
                what + ": two bus beats in DRAM tick " + std::to_string(event.dram_tick));
        }
        if (event.edge == LsuEdge::egress_write) {
            const auto [it, fresh] = slot_client.emplace(event.id, event.aux);
            require(it->second == event.aux, what + ": an egress slot written by two clients");
        }
    }
    const auto load = edge_load(machine);
    for (const auto& [key, entry] : load) {
        const auto [cycle, edge] = key;
        const auto at = what + ", cycle " + std::to_string(cycle) + ": ";
        switch (edge) {
        case LsuEdge::ingress_read:
            require(entry.bytes <= pluck, at + "ingress FIFO read " + std::to_string(entry.bytes) +
                " B, over lsu.pluck_words_per_cycle");
            break;
        case LsuEdge::tma_reg_write:
            require(entry.bytes <= pluck, at + "TMA sector register filled with " +
                std::to_string(entry.bytes) + " B, over the pluck width");
            break;
        case LsuEdge::assembly_write:
            require(entry.bytes <= pluck, at + "assembly register filled with " +
                std::to_string(entry.bytes) + " B, over the pluck width");
            break;
        case LsuEdge::egress_write:
            require(entry.bytes <= pluck && entry.events == 1,
                at + "egress write port over its width or shared by two clients");
            break;
        case LsuEdge::rf_write_beat:
        case LsuEdge::rf_read_beat:
            require(entry.events <= 1, at + "more than one full-width LSU RF beat");
            break;
        case LsuEdge::tma_place_round:
        case LsuEdge::tma_store_round: {
            const auto other = edge == LsuEdge::tma_place_round ? LsuEdge::tma_store_round : LsuEdge::tma_place_round;
            require(entry.events <= 1 && load.count({cycle, other}) == 0,
                at + "more than one TMA port-A round");
            break;
        }
        default:
            break;
        }
    }
}

// E1: per-edge widths. The vector path at pluck 1 and 2, gathers, scatters
// and TMA (load and store) at pluck 1, all under a bus 8-32x faster than the
// pluck, so that the declared SM widths are the binding ones (and the test
// requires they are reached, so it is not vacuous).
void test_lsu_edge_widths() {
    for (const uint32_t pluck : {1u, 2u}) {
        const auto config = fast_bus_profile(pluck);
        nanosm::Launch launch;
        launch.active_warps = 2;
        auto& gmem = image_bytes(launch.gmem_image, 4096);
        fill_pattern(gmem, 0, 256, 0);
        nanosm::Machine machine(config, compile(R"(
            usread ur0, WARP_ID
            ur2r r0, ur0
            shli r0, r0, 9
            ldg.v8 r8, [r0]
            stg.v8 [r0 + 2048], r8
            halt
        )", config), std::move(launch));
        machine.enable_edge_probe();
        machine.run();
        for (uint32_t word = 0; word < 256; ++word)
            require(machine.read_gmem_u32(2048 + 4 * word) == pattern_word(word), "vector copy mismatch");
        const auto what = "vector path, pluck " + std::to_string(pluck);
        check_edge_widths(machine, config, what);
        require(max_bytes_per_cycle(machine, LsuEdge::assembly_write) == pluck * 4 &&
                max_bytes_per_cycle(machine, LsuEdge::ingress_read) == pluck * 4 &&
                max_bytes_per_cycle(machine, LsuEdge::egress_write) == pluck * 4,
            what + ": the pluck width was never reached; the test proves nothing");
        require(machine.cycle() >= 1024 / (4 * pluck), what + ": faster than the ingress read port allows");
    }
    { // gathers and scatters at pluck 1
        const auto config = fast_bus_profile(1);
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& gmem = image_bytes(launch.gmem_image, 8192);
        fill_pattern(gmem, 0, 1024, 0);
        nanosm::Machine machine(config, compile(R"(
            sread r1, LANE_ID
            shli r2, r1, 7
            ldg r10, [r2 + 4]
            stg [r2 + 4096], r10
            halt
        )", config), std::move(launch));
        machine.enable_edge_probe();
        machine.run();
        for (uint32_t lane = 0; lane < 16; ++lane)
            require(machine.read_gmem_u32(4096 + 128 * lane) == pattern_word(32 * lane + 1), "gather/scatter mismatch");
        check_edge_widths(machine, config, "gather/scatter, pluck 1");
        // A whole 32 B sector is read through a 4 B port per gathered word.
        require(max_bytes_per_cycle(machine, LsuEdge::ingress_read) == 4, "gather never reached the pluck width");
    }
    for (const bool store : {false, true}) { // TMA at pluck 1: 2 KiB
        const auto config = fast_bus_profile(1);
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.shmem_image, 4096);
        write_rank1_tma(shmem, 0, 0, 512, 128);
        auto& gmem = image_bytes(launch.gmem_image, 4096);
        if (store) fill_pattern(shmem, 1024, 512, 0); else fill_pattern(gmem, 0, 512, 0);
        nanosm::Machine machine(config, compile(tma_and_wait(store, 0, 1024, 128), config), std::move(launch));
        machine.enable_edge_probe();
        machine.run();
        for (uint32_t i = 0; i < 512; ++i)
            require((store ? machine.read_gmem_u32(4 * i) : machine.read_shmem_u32(1024 + 4 * i)) == pattern_word(i),
                "TMA data mismatch at pluck 1");
        const std::string what = store ? "TMA store, pluck 1" : "TMA load, pluck 1";
        check_edge_widths(machine, config, what);
        require(max_bytes_per_cycle(machine, store ? LsuEdge::egress_write : LsuEdge::tma_reg_write) == 4,
            what + ": never reached the pluck width");
        require(machine.cycle() >= 2048 / 4, what + ": faster than one word per cycle");
    }
}

// Codex's B2 probe (chan 2026-10-04, finding 3): a 16 KiB one-dimensional
// plain TMA store, DRAM = core clock, 32 B per tick, latency 0, egress_sectors 2. B2
// took 520 cycles at pluck 1, 2 and 8 (the store bypassed egress). Now every
// TMA store sector crosses the shared egress write port, so the run is
// pluck-bound: 16384 / (4 x pluck) cycles plus a short tail.
void test_tma_store_egress_is_pluck_bound() {
    for (const uint32_t pluck : {1u, 2u, 8u}) {
        auto config = fast_bus_profile(pluck);
        config.lsu_egress_sectors = 2;
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.shmem_image, 16384 + 256);
        fill_pattern(shmem, 0, 4096, 0);
        write_rank1_tma(shmem, 16384, 0, 4096, 16384 + 128);
        image_bytes(launch.gmem_image, 16384);
        nanosm::Machine machine(config, compile(tma_and_wait(true, 16384, 0, 16384 + 128), config), std::move(launch));
        machine.run();
        for (uint32_t i = 0; i < 4096; ++i)
            require(machine.read_gmem_u32(4 * i) == pattern_word(i), "16 KiB TMA store data mismatch");
        // I2 (rfc12 section 2): a sector's source read reaches the one TMA
        // store sector register D + Lb cycles after its grant, and the
        // register holds the sector until its last word is in the egress
        // FIFO, so each of the 512 sectors costs D + Lb plus its 8 / pluck
        // drain cycles (before I2: the drain alone, 16384 / (4 * pluck)).
        const uint64_t bound = 512 * (config.shmem_port_a_depth + config.shmem_read_latency_cycles + 8 / pluck);
        require(machine.cycle() >= bound && machine.cycle() <= bound + 100,
            "16 KiB TMA store at pluck " + std::to_string(pluck) + " took " + std::to_string(machine.cycle()) +
                " cycles; the store register's round trip bounds it at " + std::to_string(bound));
    }
}

// Timelines of single data under an uncontended profile: DRAM = core clock,
// 8 B beats, gmem latency 4 ticks, pluck 2 (= the bus rate), FIFO latencies
// as given. Each entry is (stage-to-stage) cycle deltas; E2 asserts them,
// E5 compares two latency settings.
struct Timelines {
    std::vector<int64_t> vector_bus_to_assembly;   // per beat
    std::vector<int64_t> vector_assembly_to_rf;    // per chunk
    std::vector<uint64_t> vector_rf_writes;
    std::vector<int64_t> tma_bus_to_register;      // per beat
    int64_t tma_register_to_shmem = 0;
    uint64_t tma_place = 0;
    int64_t uni_bus_to_assembly = 0;
    int64_t uni_assembly_to_rf = 0;
    uint64_t uni_rf_write = 0;
    std::vector<int64_t> gather_bus_to_read;       // per beat
    int64_t gather_assembly_to_rf = 0;
    uint64_t gather_rf_write = 0;
    std::vector<int64_t> store_rf_to_egress;       // per chunk
    std::vector<int64_t> store_drain_to_next_rf;   // per chunk boundary
    std::vector<int64_t> store_slot_to_request;    // per slot
    std::vector<int64_t> store_request_to_bus;     // per slot
    std::vector<int64_t> store_ack_after_last_beat; // per slot, DRAM ticks
    uint64_t store_first_beat = 0;
    std::array<uint64_t, 5> total_cycles{};
};

Timelines lsu_timelines(uint32_t ingress_latency, uint32_t egress_latency) {
    auto config = canonical_profile();
    config.dram_clock_hz = config.core_clock_hz;
    config.gmem_bus_bytes_per_dram_tick = 8;
    config.gmem_latency_dram_ticks = 4;
    config.lsu_pluck_words_per_cycle = 2;
    config.lsu_ingress_latency_cycles = ingress_latency;
    config.lsu_egress_latency_cycles = egress_latency;
    Timelines t;
    const auto run = [&](const std::string& text, bool tma, std::size_t slot) {
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& gmem = image_bytes(launch.gmem_image, 8192);
        fill_pattern(gmem, 0, 2048, 0);
        if (tma) write_rank1_tma(image_bytes(launch.shmem_image, 2048), 0, 0, 8, 128);
        auto machine = std::make_unique<nanosm::Machine>(config, compile(text, config), std::move(launch));
        machine->enable_edge_probe();
        machine->run();
        t.total_cycles[slot] = machine->cycle();
        return machine;
    };
    const auto of = [](const nanosm::Machine& machine, LsuEdge edge) {
        std::vector<nanosm::LsuEdgeEvent> events;
        for (const auto& event : machine.edge_events()) if (event.edge == edge) events.push_back(event);
        return events;
    };
    const auto read_cycle = [&](const nanosm::Machine& machine, uint64_t request, uint64_t offset) {
        for (const auto& event : of(machine, LsuEdge::ingress_read))
            if (event.id == request && event.aux == offset) return event.cycle;
        throw std::runtime_error("a returned beat was never read from the ingress FIFO");
    };
    { // vector load: bus -> ingress FIFO -> assembly -> RF
        const auto machine = run("mov r0, 0\nldg.v8 r8, [r0]\nhalt\n", false, 0);
        std::map<uint64_t, uint64_t> last_word; // request -> cycle of its last assembly write
        for (const auto& event : of(*machine, LsuEdge::assembly_write)) last_word[event.aux] = event.cycle;
        for (const auto& beat : of(*machine, LsuEdge::bus_read_beat)) {
            const auto read = read_cycle(*machine, beat.id, beat.aux);
            bool assembled = false;
            for (const auto& event : of(*machine, LsuEdge::assembly_write))
                assembled |= event.aux == beat.id && event.cycle == read;
            require(assembled, "a vector beat read from the FIFO did not land in the assembly register that cycle");
            t.vector_bus_to_assembly.push_back(static_cast<int64_t>(read) - static_cast<int64_t>(beat.cycle));
        }
        std::vector<uint64_t> requests;
        for (const auto& [request, cycle] : last_word) requests.push_back(request);
        const auto writes = of(*machine, LsuEdge::rf_write_beat);
        require(writes.size() == 8 && requests.size() == 16, "vector load: expected 8 RF beats from 16 sectors");
        for (uint32_t j = 0; j < 8; ++j) {
            const auto chunk_done = std::max(last_word[requests[2 * j]], last_word[requests[2 * j + 1]]);
            t.vector_assembly_to_rf.push_back(static_cast<int64_t>(writes[j].cycle) - static_cast<int64_t>(chunk_done));
            t.vector_rf_writes.push_back(writes[j].cycle);
        }
    }
    { // TMA load, one sector: bus -> ingress FIFO -> TMA sector register -> shmem
        const auto machine = run(tma_and_wait(false, 0, 1024, 128), true, 1);
        uint64_t last = 0;
        for (const auto& beat : of(*machine, LsuEdge::bus_read_beat)) {
            for (const auto& event : of(*machine, LsuEdge::tma_reg_write)) {
                if (event.id != beat.id || event.aux != beat.aux) continue;
                t.tma_bus_to_register.push_back(static_cast<int64_t>(event.cycle) - static_cast<int64_t>(beat.cycle));
                last = std::max(last, event.cycle);
            }
        }
        const auto rounds = of(*machine, LsuEdge::tma_place_round);
        require(rounds.size() == 1 && t.tma_bus_to_register.size() == 4, "TMA load: expected 4 beats, 1 round");
        t.tma_register_to_shmem = static_cast<int64_t>(rounds[0].cycle) - static_cast<int64_t>(last);
        t.tma_place = rounds[0].cycle;
        for (uint32_t i = 0; i < 8; ++i)
            require(machine->read_shmem_u32(1024 + 4 * i) == pattern_word(i), "TMA timeline data mismatch");
    }
    { // ldg.uni of word 5 (byte 20, in the beat at offset 16)
        const auto machine = run("umov ur0, 0\nldg.uni r1, [ur0 + 20]\nhalt\n", false, 2);
        const auto words = of(*machine, LsuEdge::assembly_write);
        const auto writes = of(*machine, LsuEdge::rf_write_beat);
        require(words.size() == 1 && writes.size() == 1, "ldg.uni: one word, one RF beat");
        for (const auto& beat : of(*machine, LsuEdge::bus_read_beat))
            if (beat.aux == 16) t.uni_bus_to_assembly = static_cast<int64_t>(words[0].cycle) - static_cast<int64_t>(beat.cycle);
        t.uni_assembly_to_rf = static_cast<int64_t>(writes[0].cycle) - static_cast<int64_t>(words[0].cycle);
        t.uni_rf_write = writes[0].cycle;
        require(machine->read_reg(0, 7, 1) == pattern_word(5), "ldg.uni timeline data mismatch");
    }
    { // gather: 16 lanes, 16 sectors
        const auto machine = run("sread r1, LANE_ID\nshli r2, r1, 7\nldg r10, [r2 + 8]\nhalt\n", false, 3);
        for (const auto& beat : of(*machine, LsuEdge::bus_read_beat))
            t.gather_bus_to_read.push_back(
                static_cast<int64_t>(read_cycle(*machine, beat.id, beat.aux)) - static_cast<int64_t>(beat.cycle));
        uint64_t last = 0;
        for (const auto& event : of(*machine, LsuEdge::assembly_write)) last = std::max(last, event.cycle);
        const auto writes = of(*machine, LsuEdge::rf_write_beat);
        require(writes.size() == 1, "gather: one RF beat");
        t.gather_assembly_to_rf = static_cast<int64_t>(writes[0].cycle) - static_cast<int64_t>(last);
        t.gather_rf_write = writes[0].cycle;
        for (uint32_t lane = 0; lane < 16; ++lane)
            require(machine->read_reg(0, lane, 10) == pattern_word(32 * lane + 2), "gather timeline data mismatch");
    }
    { // stg.v8: RF -> assembly -> egress FIFO -> bus -> acknowledgement
        std::string text = "mov r0, 4096\n";
        for (uint32_t j = 0; j < 8; ++j) text += "mov r" + std::to_string(8 + j) + ", " + std::to_string(100 + j) + "\n";
        text += "stg.v8 [r0], r8\nhalt\n";
        const auto machine = run(text, false, 4);
        const auto reads = of(*machine, LsuEdge::rf_read_beat);
        require(reads.size() == 8, "stg.v8: eight RF read beats");
        std::map<uint64_t, std::pair<uint64_t, uint64_t>> slot_span; // slot -> first, last egress write
        for (const auto& event : of(*machine, LsuEdge::egress_write)) {
            auto [it, fresh] = slot_span.emplace(event.id, std::make_pair(event.cycle, event.cycle));
            it->second.second = event.cycle;
        }
        std::vector<std::pair<uint64_t, uint64_t>> slots;
        for (const auto& [slot, span] : slot_span) slots.push_back(span);
        require(slots.size() == 16, "stg.v8: sixteen egress slots");
        for (uint32_t j = 0; j < 8; ++j) {
            t.store_rf_to_egress.push_back(static_cast<int64_t>(slots[2 * j].first) - static_cast<int64_t>(reads[j].cycle));
            if (j != 0)
                t.store_drain_to_next_rf.push_back(
                    static_cast<int64_t>(reads[j].cycle) - static_cast<int64_t>(slots[2 * j - 1].second));
        }
        std::map<uint64_t, uint64_t> request_of_slot, request_cycle, first_beat, last_beat_tick, ack_tick;
        for (const auto& event : of(*machine, LsuEdge::write_request)) {
            request_of_slot[event.aux] = event.id;
            request_cycle[event.id] = event.cycle;
        }
        for (const auto& event : of(*machine, LsuEdge::bus_write_beat)) {
            first_beat.emplace(event.id, event.cycle);
            last_beat_tick[event.id] = event.dram_tick;
        }
        for (const auto& event : of(*machine, LsuEdge::write_ack)) ack_tick[event.id] = event.dram_tick;
        uint32_t index = 0;
        for (const auto& [slot, span] : slot_span) {
            const auto request = request_of_slot.at(slot);
            t.store_slot_to_request.push_back(
                static_cast<int64_t>(request_cycle.at(request)) - static_cast<int64_t>(span.second));
            t.store_request_to_bus.push_back(
                static_cast<int64_t>(first_beat.at(request)) - static_cast<int64_t>(request_cycle.at(request)));
            t.store_ack_after_last_beat.push_back(
                static_cast<int64_t>(ack_tick.at(request)) - static_cast<int64_t>(last_beat_tick.at(request)));
            if (index++ == 0) t.store_first_beat = first_beat.at(request);
        }
        for (uint32_t word = 0; word < 128; ++word)
            require(machine->read_gmem_u32(4096 + 4 * word) == 100 + word / 16, "stg.v8 timeline data mismatch");
    }
    return t;
}

bool all_equal(const std::vector<int64_t>& values, int64_t expected) {
    return !values.empty() && std::all_of(values.begin(), values.end(), [&](int64_t v) { return v == expected; });
}

// E2: no fall-through. Each datum spends exactly one cycle per register
// stage plus the declared FIFO latency, uncontended.
void test_lsu_no_fall_through_timelines() {
    const auto t = lsu_timelines(1, 1);
    require(all_equal(t.vector_bus_to_assembly, 1),
        "vector: bus beat -> assembly register is not exactly the ingress FIFO latency (1)");
    require(all_equal(t.vector_assembly_to_rf, 1),
        "vector: the RF write is not the cycle after the 16th word lands");
    require(all_equal(t.tma_bus_to_register, 1), "TMA: bus beat -> TMA sector register is not 1 cycle");
    require(t.tma_register_to_shmem == 1, "TMA: sector register -> shmem round is not the next cycle");
    require(t.uni_bus_to_assembly == 1 && t.uni_assembly_to_rf == 1, "ldg.uni: bus -> assembly -> RF is not 1 + 1");
    require(all_equal(t.gather_bus_to_read, 1) && t.gather_assembly_to_rf == 1,
        "gather: bus -> FIFO read -> RF is not 1 + 1");
    // I6a: the beat sampled at t is at the RF output, filling the assembly
    // register, at t + Lr; the egress port reads it (Q) one cycle later.
    require(all_equal(t.store_rf_to_egress, canonical_profile().rf_read_latency_cycles + 1),
        "stg.v8: RF read beat -> first egress word is not Lr + 1 cycles");
    // The assembly register is refilled in the cycle its last word drained
    // (its consumer runs first in the tick): a handoff, not a read-through.
    require(all_equal(t.store_drain_to_next_rf, 0), "stg.v8: drained assembly register not refilled at once");
    require(all_equal(t.store_slot_to_request, 1), "stg.v8: complete slot -> write request is not the egress latency (1)");
    require(all_equal(t.store_request_to_bus, 0), "stg.v8: an accepted write did not start on the idle bus");
    require(all_equal(t.store_ack_after_last_beat, 4), "stg.v8: the ack is not latency_dram_ticks after the last beat");
}

// E5: the two FIFO latencies are real stages: raising both from 1 to 3
// shifts every timeline that crosses one FIFO by exactly 2 cycles.
void test_lsu_fifo_latency_fields() {
    const auto one = lsu_timelines(1, 1);
    const auto three = lsu_timelines(3, 3);
    require(all_equal(three.vector_bus_to_assembly, 3) && all_equal(three.tma_bus_to_register, 3) &&
            three.uni_bus_to_assembly == 3 && all_equal(three.gather_bus_to_read, 3),
        "ingress latency 3: bus -> plucked is not 3 cycles");
    require(all_equal(three.store_slot_to_request, 3), "egress latency 3: slot -> write request is not 3 cycles");
    for (std::size_t j = 0; j < 8; ++j)
        require(three.vector_rf_writes[j] == one.vector_rf_writes[j] + 2, "vector RF write did not shift by exactly 2");
    require(three.tma_place == one.tma_place + 2 && three.uni_rf_write == one.uni_rf_write + 2 &&
            three.gather_rf_write == one.gather_rf_write + 2 && three.store_first_beat == one.store_first_beat + 2,
        "a FIFO-crossing timeline did not shift by exactly 2");
    for (std::size_t k = 0; k < one.total_cycles.size(); ++k)
        require(three.total_cycles[k] == one.total_cycles[k] + 2,
            "scenario " + std::to_string(k) + " ran " + std::to_string(three.total_cycles[k]) + " vs " +
                std::to_string(one.total_cycles[k]) + " cycles: not exactly +2");
}

// E3: one write contract. A TMA store and an stg.v8 of the same 512 B under
// the same (canonical) profile: every write is acknowledged exactly
// gmem.latency_dram_ticks after its last bus beat; TMA credits are released
// in the ack's cycle; the LSU slot at the commit of the ack (the next cycle,
// with the depbar), so both are reusable from the cycle after the ack.
void test_lsu_one_write_contract() {
    const auto config = canonical_profile();
    const auto ack_lags = [](const nanosm::Machine& machine) {
        std::map<uint64_t, uint64_t> last_beat;
        std::vector<int64_t> lags;
        for (const auto& event : machine.edge_events()) {
            if (event.edge == LsuEdge::bus_write_beat) last_beat[event.id] = event.dram_tick;
            if (event.edge == LsuEdge::write_ack)
                lags.push_back(static_cast<int64_t>(event.dram_tick) - static_cast<int64_t>(last_beat.at(event.id)));
        }
        return lags;
    };
    std::vector<int64_t> tma_lags, lsu_lags;
    { // TMA store, 128 fp32
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.shmem_image, 2048);
        write_rank1_tma(shmem, 0, 1024, 128, 128);
        fill_pattern(shmem, 1024, 128, 0);
        image_bytes(launch.gmem_image, 2048);
        nanosm::Machine machine(config, compile(tma_and_wait(true, 0, 1024, 128)), std::move(launch));
        machine.enable_edge_probe();
        std::multiset<uint64_t> releases;
        uint32_t credits = 0;
        while (!machine.done()) {
            tick_capped(machine);
            const auto now = machine.tma_sector_credits_used();
            for (uint32_t k = now; k < credits; ++k) releases.insert(machine.cycle() - 1);
            credits = now;
        }
        std::multiset<uint64_t> acks;
        for (const auto& event : machine.edge_events()) if (event.edge == LsuEdge::write_ack) acks.insert(event.cycle);
        require(acks.size() == 16 && releases == acks, "TMA store credits were not released exactly at the acks");
        tma_lags = ack_lags(machine);
        for (uint32_t i = 0; i < 128; ++i) require(machine.read_gmem_u32(1024 + 4 * i) == pattern_word(i), "TMA store data");
    }
    { // stg.v8, 128 words
        nanosm::Launch launch;
        launch.active_warps = 1;
        image_bytes(launch.gmem_image, 2048);
        std::string text = "mov r0, 1024\nsread r1, LANE_ID\n";
        for (uint32_t j = 0; j < 8; ++j) text += "addi r" + std::to_string(8 + j) + ", r1, " + std::to_string(16 * j) + "\n";
        text += "stg.v8 [r0], r8\nhalt\n";
        nanosm::Machine machine(config, compile(text), std::move(launch));
        machine.enable_edge_probe();
        uint64_t released = 0;
        std::size_t in_flight = 0;
        while (!machine.done()) {
            tick_capped(machine);
            const auto now = machine.lsu_commands_in_flight();
            if (now < in_flight) released = machine.cycle() - 1;
            in_flight = now;
        }
        uint64_t last_ack = 0;
        for (const auto& event : machine.edge_events()) if (event.edge == LsuEdge::write_ack) last_ack = event.cycle;
        require(released == last_ack + 1, "the stg.v8 slot was not released at the commit of its last ack");
        lsu_lags = ack_lags(machine);
        for (uint32_t i = 0; i < 128; ++i) require(machine.read_gmem_u32(1024 + 4 * i) == i, "stg.v8 data");
    }
    require(tma_lags.size() == 16 && lsu_lags.size() == 16, "expected sixteen writes per client");
    require(all_equal(tma_lags, config.gmem_latency_dram_ticks) && tma_lags == lsu_lags,
        "TMA store and stg.v8 acks are not both gmem.latency_dram_ticks after the last beat");
}

// E4: lsu.read_streams = 3 (TMA + two vector streams). Two ldg.v8 commands
// land on streams 1 and 2 (round-robin); the legal interleave order
// reorders their sectors ACROSS the streams, never within one; data is
// correct either way. With two streams (both commands on stream 1) the same
// hook cannot interleave them, and the illegal reorder still panics.
void test_lsu_read_streams_generalized() {
    const auto run = [](uint32_t streams, nanosm::ControllerTestOrder order) {
        auto config = canonical_profile();
        config.lsu_read_streams = streams;
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& gmem = image_bytes(launch.gmem_image, 1024);
        fill_pattern(gmem, 0, 256, 0);
        nanosm::Machine machine(config, compile(R"(
            mov r0, 0
            mov r1, 512
            ldg.v8 r8, [r0]
            ldg.v8 r16, [r1]
            halt
        )", config), std::move(launch), {.timeline = true});
        machine.set_controller_test_order(order);
        machine.run();
        for (uint32_t j = 0; j < 8; ++j)
            for (uint32_t lane = 0; lane < 16; ++lane) {
                require(machine.read_reg(0, lane, 8 + j) == pattern_word(16 * j + lane), "stream A data mismatch");
                require(machine.read_reg(0, lane, 16 + j) == pattern_word(128 + 16 * j + lane), "stream B data mismatch");
            }
        std::vector<uint64_t> served;
        for (const auto& span : trace_spans(machine, nanosm::TraceCategory::gmem, "read sector"))
            served.push_back(std::stoull(span.detail.substr(span.detail.find("address=") + 8)));
        return served;
    };
    const auto switches = [](const std::vector<uint64_t>& served) {
        uint32_t count = 0;
        for (std::size_t k = 1; k < served.size(); ++k) count += (served[k] < 512) != (served[k - 1] < 512);
        return count;
    };
    const auto in_order_within = [](const std::vector<uint64_t>& served) {
        uint64_t a = 0, b = 512;
        for (const auto address : served) {
            auto& next = address < 512 ? a : b;
            if (address != next) return false;
            next += 32;
        }
        return true;
    };
    const auto fifo = run(3, nanosm::ControllerTestOrder::fifo);
    require(switches(fifo) == 1 && in_order_within(fifo), "the FIFO controller reordered vector sectors");
    const auto interleaved = run(3, nanosm::ControllerTestOrder::interleave_ordered_streams);
    require(switches(interleaved) >= 16, "read_streams 3: the commands' sectors did not interleave across streams (" +
        std::to_string(switches(interleaved)) + " switches)");
    require(in_order_within(interleaved), "read_streams 3: a stream was reordered within its ID");
    const auto same_stream = run(2, nanosm::ControllerTestOrder::interleave_ordered_streams);
    require(switches(same_stream) == 1 && in_order_within(same_stream),
        "read_streams 2: the legal hook reordered within one ID");
    auto config = canonical_profile();
    config.lsu_read_streams = 3;
    nanosm::Launch launch;
    launch.active_warps = 1;
    image_bytes(launch.gmem_image, 1024);
    nanosm::Machine machine(config, compile("mov r0, 0\nmov r1, 512\nldg.v8 r8, [r0]\nldg.v8 r16, [r1]\nhalt\n", config),
        std::move(launch));
    machine.set_controller_test_order(nanosm::ControllerTestOrder::illegal_ordered_reorder);
    expect_sim_panic(machine, "contract violation: read stream 1 (vector) returned request",
        "an out-of-order vector stream with three read streams");
}

} // namespace

// vadd
TEST(LsuSlow, vadd) {
    test_vadd();
}

// sparse launch images
TEST(Lsu, sparse_launch_images) {
    test_sparse_launch_images();
}

// lsu scalar ldg: one request per lane
TEST(Lsu, lsu_scalar_ldg_is_one_request_per_lane) {
    test_lsu_scalar_ldg_is_one_request_per_lane();
}

// lsu v8 interleaved mapping
TEST(Lsu, lsu_v8_interleaved_mapping) {
    test_lsu_v8_interleaved_mapping();
}

// lsu v8 store source lifetime
TEST(Lsu, lsu_v8_store_source_lifetime) {
    test_lsu_v8_store_source_lifetime();
}

// gmem v8 predication rejection
TEST(Lsu, gmem_v8_predication_is_rejected) {
    test_gmem_v8_predication_is_rejected();
}

// lsu RF writes are full-width beats
TEST(Lsu, lsu_rf_writes_are_full_width_beats) {
    test_lsu_rf_writes_are_full_width_beats();
}

// lsu contract panics
TEST(Lsu, lsu_contract_panics) {
    test_lsu_contract_panics();
}

// lsu commands stream through a small queue
TEST(Lsu, lsu_commands_stream_through_small_queue) {
    test_lsu_commands_stream_through_small_queue();
}

// lsu command slot stalls issue
TEST(Lsu, lsu_command_slot_structurally_stalls) {
    test_lsu_command_slot_structurally_stalls();
}

// lsu gather requests
TEST(Lsu, lsu_gather_requests) {
    test_lsu_gather_requests();
}

// lsu scatter byte enables
TEST(Lsu, lsu_scatter_byte_enables) {
    test_lsu_scatter_byte_enables();
}

// lsu staging pool admission
TEST(Lsu, lsu_staging_pool_admission) {
    test_lsu_staging_pool_admission();
}

// lsu backpressure chain
TEST(Lsu, lsu_backpressure_chain) {
    test_lsu_backpressure_chain();
}

// lsu TMA/vector stream interleave
TEST(Lsu, lsu_tma_and_vector_streams_interleave) {
    test_lsu_tma_and_vector_streams_interleave();
}

// lsu controller contract tripwire
TEST(Lsu, lsu_controller_contract_tripwire) {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    test_lsu_controller_contract_tripwire();
}

// lsu profile and ldg.uni encoding
TEST(Lsu, lsu_profile_and_encoding) {
    test_lsu_profile_and_encoding();
}

// lsu B2.1: per-edge widths
TEST(Lsu, lsu_edge_widths) {
    test_lsu_edge_widths();
}

// lsu B2.1: TMA store egress is pluck-bound
TEST(LsuSlow, tma_store_egress_is_pluck_bound) {
    test_tma_store_egress_is_pluck_bound();
}

// lsu B2.1: no fall-through timelines
TEST(Lsu, lsu_no_fall_through_timelines) {
    test_lsu_no_fall_through_timelines();
}

// lsu B2.1: FIFO latency fields
TEST(Lsu, lsu_fifo_latency_fields) {
    test_lsu_fifo_latency_fields();
}

// lsu B2.1: one write contract
TEST(Lsu, lsu_one_write_contract) {
    test_lsu_one_write_contract();
}

// lsu B2.1: read_streams generalized
TEST(Lsu, lsu_read_streams_generalized) {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    test_lsu_read_streams_generalized();
}

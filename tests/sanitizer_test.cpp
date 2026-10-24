// The race sanitizer's ownership checks.
// Moved from the original tests/tests.cpp (trash/test-setup-2026-10-22.md);
// each test's body is unchanged, its TEST wrapper is at the end.
#include "test_support.hpp"

namespace {

using namespace nanosm_test;

void test_sanitizer_sts_into_tma_store_source() {
    // tma-out break (1): the issuer fires the store before the workers finished
    // staging. The store read-owns its source from accept until its last source
    // read, so the late sts panics whatever the port timing.
    const auto& profile = canonical_profile();
    const auto launch_with_store = [] {
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.shmem_image, 8192);
        image_bytes(launch.gmem_image, 2048);
        for (uint32_t i = 0; i < 256; ++i) write_u32(shmem, 4 * i, 3 * i + 1);
        write_tma_descriptor(shmem, 4096, linear_tma(true, 0, 1024, 4160));
        write_tx_mbar(shmem, 4160, 1024);
        return launch;
    };
    const std::string prologue = R"(
        mov r0, 0
        mov r1, 0
        mov r2, 0
        mov r4, 4160
        tma.async [4096], r0
    )";
    const std::string stage = R"(
        sread r20, LANE_ID
        shli r20, r20, 2
        mov r21, 7
        sts [r20], r21
        halt
    )";
    const auto message = run_expecting_panic(profile, compile(prologue + stage),
        launch_with_store(), "sts into a live TMA store source");
    require_mentions(message, {"shmem race: sts writes 16 words, bytes [0, 64)",
        "tma store #0", "read-owner of its source [0, 1024) until its last source read",
        "wait the TMA completion mbar [4160]", "inst='sts"}, "TMA store source race");

    // Control: waiting the store's completion mbar orders the overwrite.
    nanosm::Machine machine(profile, compile(prologue + R"(
    wait:
        mbar.try_wait p0, [r4], 0
        @!p0 bra.uni wait
    )" + stage), launch_with_store());
    machine.run(20000);
    for (uint32_t i = 0; i < 256; ++i)
        require(machine.read_gmem_u32(4 * i) == 3 * i + 1, "the waited store captured a late sts");
    require(machine.read_shmem_u32(0) == 7, "the waited sts did not execute");
}

void test_sanitizer_sts_into_mma_operand_stage() {
    // tma-out break (2): a write into a stage that a queued or active MMA will
    // still read. Queue depth 4 lets a command wait behind an unfired commit.
    auto profile = canonical_profile();
    profile.mma_queue_commands = 4;
    nanosm::validate_device_profile(profile);
    for (const bool queued : {true, false}) {
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.shmem_image, 8192 + 8);
        write_count_mbar(shmem, 8192, 1);
        const auto kernel = compile(std::string(R"(
            .acc C, 0, 64, 64
            mov r0, 0
            mov r1, 1024
            mov r2, 2048
            mov r3, 3072
            mov r4, 8192
            mma.async.init.m64n64k8 C, [r0], [r1]
            mma.commit [r4]
            mma.async.m64n64k8 C, [r2], [r3]
            sread r20, LANE_ID
            shli r20, r20, 2
            addi r20, r20, )") + (queued ? "2048" : "0") + R"(
            mov r21, 0
            sts [r20], r21
            halt
        )", profile);
        const auto message = run_expecting_panic(profile, kernel, std::move(launch),
            queued ? "sts into a queued MMA's stage" : "sts into an active MMA's stage");
        if (queued) {
            require_mentions(message, {"sts writes 16 words, bytes [2048, 2112)",
                "mma.async #3 [queued]", "read-owner of its A operand [2048, 3072)",
                "until its final operand read", "wait the operand-consumed depbar sb",
                "/ commit mbar"}, "queued-MMA stage race");
        } else {
            require_mentions(message, {"sts writes 16 words, bytes [0, 64)",
                "mma.async #1 [fetching]", "read-owner of its A operand [0, 1024)",
                "wait the operand-consumed depbar sb"}, "active-MMA stage race");
        }
    }
}

void test_sanitizer_lds_from_tma_load_destination() {
    const auto& profile = canonical_profile();
    const auto launch_with_load = [] {
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.shmem_image, 8192);
        auto& gmem = image_bytes(launch.gmem_image, 1024);
        for (uint32_t i = 0; i < 256; ++i) write_u32(gmem, 4 * i, 5 * i + 2);
        write_tma_descriptor(shmem, 4096, linear_tma(false, 0, 1024, 4160));
        write_tx_mbar(shmem, 4160, 1024);
        return launch;
    };
    const std::string prologue = R"(
        mov r0, 0
        mov r1, 0
        mov r2, 0
        mov r4, 4160
        sread r20, LANE_ID
        shli r20, r20, 2
        tma.async [4096], r0
    )";
    const auto message = run_expecting_panic(profile, compile(prologue + R"(
        lds r21, [r20]
        halt
    )"), launch_with_load(), "lds from a pending TMA load destination");
    require_mentions(message, {"shmem race: lds reads 16 words, bytes [0, 64)",
        "tma load #0", "write-owner of its destination [0, 1024) until its last shmem write commits",
        "wait the TMA completion mbar [4160]"}, "TMA load destination read race");

    nanosm::Machine machine(profile, compile(prologue + R"(
    wait:
        mbar.try_wait p0, [r4], 0
        @!p0 bra.uni wait
        lds r21, [r20]
        halt
    )"), launch_with_load());
    machine.run(20000);
    for (uint32_t lane = 0; lane < profile.lanes; ++lane)
        require(machine.read_reg(0, lane, 21) == 5 * lane + 2, "the waited lds read the wrong value");
}

void test_sanitizer_mbar_rmw_into_owned_range() {
    const auto& profile = canonical_profile();
    {
        // A SIMT arrive whose mbar sits inside a pending TMA load destination.
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.shmem_image, 8192);
        image_bytes(launch.gmem_image, 1024);
        write_tma_descriptor(shmem, 4096, linear_tma(false, 0, 1024, 4160));
        write_tx_mbar(shmem, 4160, 1024);
        write_count_mbar(shmem, 512, 2);
        const auto message = run_expecting_panic(profile, compile(R"(
            mov r0, 0
            mov r1, 0
            mov r2, 0
            mov r4, 512
            tma.async [4096], r0
            mbar.arrive [r4]
            halt
        )"), std::move(launch), "mbar.arrive inside a TMA load destination");
        require_mentions(message, {"shmem race: mbar.arrive writes mbar bytes [512, 520)",
            "tma load #0", "wait the TMA completion mbar [4160]", "warp=0"},
            "SIMT RMW ownership race");
    }
    {
        // An engine RMW (TMA complete_tx) whose completion mbar sits inside an
        // operand stage two queued MMAs still read.
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.shmem_image, 8192);
        image_bytes(launch.gmem_image, 1024);
        write_tma_descriptor(shmem, 4096, linear_tma(false, 2048, 1024, 512));
        write_tx_mbar(shmem, 512, 1024);
        const auto message = run_expecting_panic(profile, compile(R"(
            .acc C, 0, 64, 64
            mov r0, 0
            mov r1, 1024
            mov r8, 0
            mov r9, 0
            mov r10, 0
            mma.async.init.m64n64k8 C, [r0], [r1]
            mma.async.m64n64k8 C, [r0], [r1]
            tma.async [4096], r8
            halt
        )"), std::move(launch), "engine complete_tx into an MMA operand stage");
        require_mentions(message, {"engine complete_tx writes mbar bytes [512, 520)",
            "read-owner of its A operand [0, 1024)", "source=engine"},
            "engine RMW ownership race");
    }
}

void test_sanitizer_tma_and_mma_accept_conflicts() {
    const auto& profile = canonical_profile();
    const auto launch_with = [](bool second_store) {
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.shmem_image, 8192);
        image_bytes(launch.gmem_image, 1024);
        write_tma_descriptor(shmem, 4096, linear_tma(false, 0, 1024, 4160));
        write_tma_descriptor(shmem, 4224, linear_tma(second_store, 1024, 1024, 4168));
        write_tma_descriptor(shmem, 4288, linear_tma(true, 0, 1024, 4168));
        write_tx_mbar(shmem, 4160, 1024);
        write_tx_mbar(shmem, 4168, 1024);
        return launch;
    };
    // TMA load accepted over a stage an active MMA reads.
    auto message = run_expecting_panic(profile, compile(R"(
        .acc C, 0, 64, 64
        mov r0, 0
        mov r1, 1024
        mov r8, 0
        mov r9, 0
        mov r10, 0
        mma.async.init.m64n64k8 C, [r0], [r1]
        tma.async [4096], r8
        halt
    )"), launch_with(false), "TMA load over an active MMA operand");
    require_mentions(message, {"shmem race: tma load #0 destination bytes [0, 1024)",
        "owned by mma.async #1 [fetching]", "A operand [0, 1024)",
        "wait the operand-consumed depbar sb", "inst='tma.async"}, "TMA load vs MMA");
    // MMA accepted over a pending TMA load destination (the other direction).
    message = run_expecting_panic(profile, compile(R"(
        .acc C, 0, 64, 64
        mov r0, 0
        mov r1, 1024
        mov r8, 0
        mov r9, 0
        mov r10, 0
        tma.async [4224], r8
        mma.async.init.m64n64k8 C, [r0], [r1]
        halt
    )"), launch_with(false), "MMA over a pending TMA load destination");
    require_mentions(message, {"shmem race: mma.async #1 B operand bytes [1024, 2048)",
        "owned by tma load #0", "wait the TMA completion mbar [4168]"}, "MMA vs TMA load");
    // TMA store accepted whose source a pending TMA load still writes.
    message = run_expecting_panic(profile, compile(R"(
        mov r8, 0
        mov r9, 0
        mov r10, 0
        tma.async [4096], r8
        tma.async [4288], r8
        halt
    )"), launch_with(false), "TMA store of a pending TMA load destination");
    require_mentions(message, {"shmem race: tma store #1 source bytes [0, 1024)",
        "owned by tma load #0", "wait the TMA completion mbar [4160]"}, "TMA store vs TMA load");
    // Two readers never conflict: a TMA store of an MMA operand stage is legal.
    nanosm::Machine readers(profile, compile(R"(
        .acc C, 0, 64, 64
        mov r0, 0
        mov r1, 1024
        mov r4, 4168
        mov r8, 0
        mov r9, 0
        mov r10, 0
        mma.async.init.m64n64k8 C, [r0], [r1]
        tma.async [4224], r8
    wait:
        mbar.try_wait p0, [r4], 0
        @!p0 bra.uni wait
        halt
    )"), launch_with(true));
    readers.run(20000);
}

void test_sanitizer_tmem_ownership_both_directions() {
    const auto& profile = canonical_profile();
    const auto one_warp = [] {
        nanosm::Launch launch;
        launch.active_warps = 1;
        return launch;
    };
    const std::string header = R"(
        .acc C, 0, 64, 64
        mov r0, 0
        mov r1, 1024
        mov r8, 0
        mov r9, 0
        mov r10, 16
        mov r11, 0
    )";
    // C4, new direction: an MMA accepted while an acc.st on its accumulator
    // is still writing tmem.
    auto message = run_expecting_panic(profile, compile(header + R"(
        acc.st.v8 C, r8, r16
        mma.async.init.m64n64k8 C, [r0], [r1]
        halt
    )"), one_warp(), "MMA over an in-flight acc.st");
    require_mentions(message, {"tmem race: mma.async #1 on 'C'",
        "acc.st #0", "write-owner of 'C' rows [0, 16) cols [0, 8) until its final tmem write",
        "wait that transfer's depbar sb"}, "MMA vs acc.st");
    // ... and while an acc.ld still reads it (WAR on tmem). I2.1 (F6): the
    // read-ownership ends at the final tmem read (I8: the last beat's tmem
    // sample), not the final RF write.
    message = run_expecting_panic(profile, compile(header + R"(
        acc.ld.v8 r24, C, r8
        mma.async.init.m64n64k8 C, [r0], [r1]
        halt
    )"), one_warp(), "MMA over an in-flight acc.ld");
    require_mentions(message, {"tmem race: mma.async #1 on 'C'", "acc.ld #0",
        "read-owner of 'C' rows [0, 16) cols [0, 8) until its final tmem read"}, "MMA vs acc.ld");
    // Existing direction, same text plus the owner.
    message = run_expecting_panic(profile, compile(header + R"(
        mma.async.init.m64n64k8 C, [r0], [r1]
        acc.ld.v8 r24, C, r8
        halt
    )"), one_warp(), "acc.ld over accepted MMA work");
    require_mentions(message, {"accumulator transfer races accepted MMA work on 'C' owned by "
        "mma.async #1", "write-owner of 'C' rows [0, 64) cols [0, 64) until its last result retires",
        ": wait on the mma.commit mbar phase first"}, "acc.ld vs MMA");
    // Transfer vs transfer (I8): the acc.ld and acc.st engines run side by
    // side and nothing orders them, so an overlapping pair is a race. qtxc
    // orders one warp's transfers of an accumulator by depbar; hand assembly
    // without that wait shows the sanitizer's diagnosis.
    const std::string hand_header = R"(
        .acc C, 0, 64, 64
        mov r8, 0
        mov r9, 0
        mov r10, 16
        mov r11, 0
    )";
    message = run_expecting_panic(profile, hand_assemble(i1_asm(profile, hand_header + R"(
        acc.st.v8 C, r8, r16 {pred=PT, stall=0, set=sb0, wait=-}
        acc.ld.v8 r24, C, r8 {pred=PT, stall=0, set=sb1, wait=-}
        halt {pred=PT, stall=0, set=-, wait=sb0|sb1}
    )"), profile), one_warp(), "acc.ld over an in-flight acc.st");
    require_mentions(message, {"tmem race: acc.ld reads 'C' rows [0, 16) cols [0, 8)",
        "overlapping acc.st #0", "wait that transfer's depbar sb"}, "acc.ld vs acc.st");
    message = run_expecting_panic(profile, hand_assemble(i1_asm(profile, hand_header + R"(
        acc.ld.v8 r24, C, r8 {pred=PT, stall=0, set=sb1, wait=-}
        acc.st.v8 C, r8, r16 {pred=PT, stall=0, set=sb0, wait=-}
        halt {pred=PT, stall=0, set=-, wait=sb0|sb1}
    )"), profile), one_warp(), "acc.st over an in-flight acc.ld");
    require_mentions(message, {"tmem race: acc.st writes 'C' rows [0, 16) cols [0, 8)",
        "overlapping acc.ld #0"}, "acc.st vs acc.ld");
    // Disjoint rows: concurrent, legal, and exact.
    nanosm::Machine disjoint(profile, hand_assemble(i1_asm(profile, hand_header + R"(
        mov r16, 41
        acc.st.v8 C, r10, r16 {pred=PT, stall=0, set=sb0, wait=-}
        acc.ld.v8 r24, C, r8 {pred=PT, stall=0, set=sb1, wait=-}
        halt {pred=PT, stall=0, set=-, wait=sb0|sb1}
    )"), profile), one_warp());
    disjoint.run(2000);
    require(disjoint.read_reg(0, 0, 24) == 0, "disjoint acc.ld read the concurrent store's rows");
}

void test_sanitizer_double_buffer_and_p_tile_are_clean() {
    const auto& profile = canonical_profile();
    {
        // TMA fills stage 1 while MMA reads stage 0, then the roles swap after
        // the stage mbar. Disjoint ranges: no panic, exact result.
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.shmem_image, 8192 + 24);
        auto& gmem = image_bytes(launch.gmem_image, 1024);
        for (uint32_t i = 0; i < 512; ++i) write_u16(gmem, 2 * i, 0x3c00);
        for (uint32_t stage = 0; stage < 4; ++stage) {
            nanosm::TmaDescriptor descriptor;
            descriptor.element_bytes = 2;
            descriptor.dimensions = {512, 1, 1};
            descriptor.box = {512, 1, 1};
            descriptor.shmem_base = 1024 * stage;
            descriptor.completion_mbar = stage < 2 ? 8192 : 8200;
            write_tma_descriptor(shmem, 4096 + 64 * stage, descriptor);
        }
        for (uint32_t mbar : {8192u, 8200u, 8208u}) write_count_mbar(shmem, mbar, 1);
        nanosm::Machine machine(profile, compile(R"(
            .acc C, 0, 64, 64
            mov r8, 0
            mov r9, 0
            mov r10, 0
            mov r2, 8192
            mov r3, 8200
            mov r4, 8208
            mbar.arrive.expect_tx [r2], 2048
            tma.async [4096], r8
            tma.async [4160], r8
        L0:
            mbar.try_wait p1, [r2], 0
            @!p1 bra.uni L0
            mov r0, 0
            mov r1, 1024
            mma.async.init.m64n64k8 C, [r0], [r1]
            mbar.arrive.expect_tx [r3], 2048
            tma.async [4224], r8
            tma.async [4288], r8
        L1:
            mbar.try_wait p1, [r3], 0
            @!p1 bra.uni L1
            mov r0, 2048
            mov r1, 3072
            mma.async.m64n64k8 C, [r0], [r1]
            mbar.arrive.expect_tx [r2], 2048
            tma.async [4096], r8
            tma.async [4160], r8
        L2:
            mbar.try_wait p1, [r2], 1
            @!p1 bra.uni L2
            mov r0, 0
            mov r1, 1024
            mma.async.m64n64k8 C, [r0], [r1]
            mma.commit [r4]
        L3:
            mbar.try_wait p1, [r4], 0
            @!p1 bra.uni L3
            mov r12, 0
            mov r13, 0
            acc.ld.v8 r16, C, r12
            halt
        )"), std::move(launch), {.timeline = true});
        machine.run(50000);
        for (uint32_t lane = 0; lane < profile.lanes; ++lane)
            for (uint32_t j = 0; j < 8; ++j)
                require(machine.read_reg(0, lane, 16 + j) == nanosm::float_bits(24.0f),
                    "double-buffered MMA chain produced the wrong value");
        // The overlap really happened: stage 1's fill began before MMA #1 (on
        // stage 0) finished its operand reads, and stage 0's refill began
        // before MMA #2 (on stage 1) finished.
        std::map<uint64_t, uint64_t> tma_begin;
        std::map<uint64_t, uint64_t> mma_last_input;
        for (const auto& event : machine.trace().events()) {
            if (event.name == "tma command") tma_begin[event.id] = event.start_cycle;
            if (event.name == "MMA input step")
                mma_last_input[event.id] = std::max(mma_last_input[event.id], event.start_cycle);
        }
        require(tma_begin.size() == 6 && mma_last_input.size() == 3,
            "double-buffer trace lost its commands");
        require(tma_begin.at(2) < mma_last_input.at(1) && tma_begin.at(4) < mma_last_input.at(2),
            "double-buffer test did not overlap a TMA fill with an MMA read");
    }
    {
        // FA's P-tile handoff, twice: worker sts P -> mbar -> MMA reads P ->
        // commit mbar -> worker overwrites P -> mbar -> MMA reads it again.
        nanosm::Launch launch;
        launch.active_warps = 2;
        auto& shmem = image_bytes(launch.shmem_image, 8192 + 16);
        for (uint32_t i = 0; i < 512; ++i) write_u16(shmem, 1024 + 2 * i, 0x3c00);
        write_count_mbar(shmem, 8192, 1);
        write_count_mbar(shmem, 8200, 1);
        nanosm::Machine machine(profile, compile(R"(
            .acc C, 0, 64, 64
            mov r10, 8192
            mov r11, 8200
            sread r0, WARP_ID
            setp.eq p0, r0, 0
            @p0 bra.uni ISSUER
            sread r20, LANE_ID
            shli r20, r20, 2
            mov r21, 15360
            shli r22, r21, 16
            or r21, r21, r22
            mov r25, 0
        FILL1:
            add r26, r20, r25
            sts [r26], r21
            addi r25, r25, 64
            setp.lt p1, r25, 1024
            @p1 bra.uni FILL1
            mbar.arrive [r10]
        DONE0:
            mbar.try_wait p1, [r11], 0
            @!p1 bra.uni DONE0
            mov r21, 16384
            shli r22, r21, 16
            or r21, r21, r22
            mov r25, 0
        FILL2:
            add r26, r20, r25
            sts [r26], r21
            addi r25, r25, 64
            setp.lt p1, r25, 1024
            @p1 bra.uni FILL2
            mbar.arrive [r10]
            halt
        ISSUER:
            mov r0, 0
            mov r1, 1024
        READY0:
            mbar.try_wait p1, [r10], 0
            @!p1 bra.uni READY0
            mma.async.init.m64n64k8 C, [r0], [r1]
            mma.commit [r11]
        WAIT0:
            mbar.try_wait p1, [r11], 0
            @!p1 bra.uni WAIT0
        READY1:
            mbar.try_wait p1, [r10], 1
            @!p1 bra.uni READY1
            mma.async.m64n64k8 C, [r0], [r1]
            mma.commit [r11]
        WAIT1:
            mbar.try_wait p1, [r11], 1
            @!p1 bra.uni WAIT1
            mov r12, 0
            mov r13, 0
            acc.ld.v8 r16, C, r12
            halt
        )"), std::move(launch));
        machine.run(50000);
        for (uint32_t lane = 0; lane < profile.lanes; ++lane)
            for (uint32_t j = 0; j < 8; ++j)
                require(machine.read_reg(0, lane, 16 + j) == nanosm::float_bits(24.0f),
                    "P-tile handoff produced the wrong value");
    }
}

} // namespace

// sanitizer: sts into TMA store source
TEST(Sanitizer, sanitizer_sts_into_tma_store_source) {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    test_sanitizer_sts_into_tma_store_source();
}

// sanitizer: sts into MMA operand stage
TEST(Sanitizer, sanitizer_sts_into_mma_operand_stage) {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    test_sanitizer_sts_into_mma_operand_stage();
}

// sanitizer: lds from TMA load destination
TEST(Sanitizer, sanitizer_lds_from_tma_load_destination) {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    test_sanitizer_lds_from_tma_load_destination();
}

// sanitizer: mbar RMW into owned range
TEST(Sanitizer, sanitizer_mbar_rmw_into_owned_range) {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    test_sanitizer_mbar_rmw_into_owned_range();
}

// sanitizer: TMA/MMA accept conflicts
TEST(Sanitizer, sanitizer_tma_and_mma_accept_conflicts) {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    test_sanitizer_tma_and_mma_accept_conflicts();
}

// sanitizer: tmem ownership both directions
TEST(Sanitizer, sanitizer_tmem_ownership_both_directions) {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    test_sanitizer_tmem_ownership_both_directions();
}

// sanitizer: double buffer and P tile clean
TEST(Sanitizer, sanitizer_double_buffer_and_p_tile_are_clean) {
    test_sanitizer_double_buffer_and_p_tile_are_clean();
}

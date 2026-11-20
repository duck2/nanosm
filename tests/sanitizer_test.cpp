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
        write_tma_descriptor(shmem, 4096, linear_tma(1024));
        write_tx_mbar(shmem, 4160, 1024);
        return launch;
    };
    const std::string prologue = tma_operands_asm(10, 0, 4, 4160, 8) + R"(
        tma.store.1d [4096, ur8], [ur10], [ur4]
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
        mbar.try_wait p0, [ur4], 0
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
            umov ur0, 0
            umov ur1, 1024
            umov ur2, 2048
            umov ur3, 3072
            umov ur4, 8192
            umov ur5, 0
            mma.async.init.m64n64k8 [ur5], [ur0], [ur1]
            mma.commit [ur4]
            mma.async.m64n64k8 [ur5], [ur2], [ur3]
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
        write_tma_descriptor(shmem, 4096, linear_tma(1024));
        write_tx_mbar(shmem, 4160, 1024);
        return launch;
    };
    const std::string prologue = tma_operands_asm(10, 0, 4, 4160, 8) + R"(
        sread r20, LANE_ID
        shli r20, r20, 2
        tma.load.1d [ur10], [4096, ur8], [ur4]
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
        mbar.try_wait p0, [ur4], 0
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
        write_tma_descriptor(shmem, 4096, linear_tma(1024));
        write_tx_mbar(shmem, 4160, 1024);
        write_count_mbar(shmem, 512, 2);
        const auto message = run_expecting_panic(profile, compile(tma_operands_asm(10, 0, 11, 4160, 8) + R"(
            umov ur4, 512
            tma.load.1d [ur10], [4096, ur8], [ur11]
            mbar.arrive [ur4]
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
        write_tma_descriptor(shmem, 4096, linear_tma(1024));
        write_tx_mbar(shmem, 512, 1024);
        const auto message = run_expecting_panic(profile, compile(R"(
            umov ur0, 0
            umov ur1, 1024
            umov ur2, 0
        )" + tma_operands_asm(10, 2048, 11, 512, 8) + R"(
            mma.async.init.m64n64k8 [ur2], [ur0], [ur1]
            mma.async.m64n64k8 [ur2], [ur0], [ur1]
            tma.load.1d [ur10], [4096, ur8], [ur11]
            halt
        )"), std::move(launch), "engine complete_tx into an MMA operand stage");
        require_mentions(message, {"engine complete_tx writes mbar bytes [512, 520)",
            "read-owner of its A operand [0, 1024)", "source=engine"},
            "engine RMW ownership race");
    }
}

void test_sanitizer_tma_and_mma_accept_conflicts() {
    const auto& profile = canonical_profile();
    // One descriptor (1 KiB at gmem 0) serves every command below: each one's
    // direction, shmem side and completion mbar are the instruction's.
    const auto launch_with_descriptor = [] {
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.shmem_image, 8192);
        image_bytes(launch.gmem_image, 1024);
        write_tma_descriptor(shmem, 4096, linear_tma(1024));
        write_tx_mbar(shmem, 4160, 1024);
        write_tx_mbar(shmem, 4168, 1024);
        return launch;
    };
    // TMA load accepted over a stage an active MMA reads.
    auto message = run_expecting_panic(profile, compile(R"(
        umov ur0, 0
        umov ur1, 1024
        umov ur2, 0
    )" + tma_operands_asm(10, 0, 11, 4160, 8) + R"(
        mma.async.init.m64n64k8 [ur2], [ur0], [ur1]
        tma.load.1d [ur10], [4096, ur8], [ur11]
        halt
    )"), launch_with_descriptor(), "TMA load over an active MMA operand");
    require_mentions(message, {"shmem race: tma load #0 destination bytes [0, 1024)",
        "owned by mma.async #1 [fetching]", "A operand [0, 1024)",
        "wait the operand-consumed depbar sb", "inst='tma.load.1d [ur10], [4096, ur8], [ur11]"}, "TMA load vs MMA");
    // MMA accepted over a pending TMA load destination (the other direction).
    message = run_expecting_panic(profile, compile(R"(
        umov ur0, 0
        umov ur1, 1024
        umov ur2, 0
    )" + tma_operands_asm(10, 1024, 11, 4168, 8) + R"(
        tma.load.1d [ur10], [4096, ur8], [ur11]
        mma.async.init.m64n64k8 [ur2], [ur0], [ur1]
        halt
    )"), launch_with_descriptor(), "MMA over a pending TMA load destination");
    require_mentions(message, {"shmem race: mma.async #1 B operand bytes [1024, 2048)",
        "owned by tma load #0", "wait the TMA completion mbar [4168]"}, "MMA vs TMA load");
    // TMA store accepted whose source a pending TMA load still writes (both
    // commands' buffer and coordinate registers are ur10 and ur8).
    message = run_expecting_panic(profile, compile(tma_operands_asm(10, 0, 11, 4160, 8) + R"(
        umov ur12, 4168
        tma.load.1d [ur10], [4096, ur8], [ur11]
        tma.store.1d [4096, ur8], [ur10], [ur12]
        halt
    )"), launch_with_descriptor(), "TMA store of a pending TMA load destination");
    require_mentions(message, {"shmem race: tma store #1 source bytes [0, 1024)",
        "owned by tma load #0", "wait the TMA completion mbar [4160]"}, "TMA store vs TMA load");
    // Two readers never conflict: a TMA store of an MMA operand stage is legal.
    nanosm::Machine readers(profile, compile(R"(
        umov ur0, 0
        umov ur1, 1024
        umov ur2, 0
    )" + tma_operands_asm(10, 1024, 4, 4168, 8) + R"(
        mma.async.init.m64n64k8 [ur2], [ur0], [ur1]
        tma.store.1d [4096, ur8], [ur10], [ur4]
    wait:
        mbar.try_wait p0, [ur4], 0
        @!p0 bra.uni wait
        halt
    )"), launch_with_descriptor());
    readers.run(20000);
}

void test_sanitizer_tmem_ownership_both_directions() {
    const auto& profile = canonical_profile();
    const auto one_warp = [] {
        nanosm::Launch launch;
        launch.active_warps = 1;
        return launch;
    };
    // tmem pointers (src/layout.hpp): ur8 rows 0..15 and ur10 rows 16..31 of a
    // region of 64-word rows at 0, ur9 the matmul's result (64x64 at 0).
    // Ownership is by the words touched.
    const std::string header = R"(
        umov ur0, 0
        umov ur1, 1024
        umov ur8, 0
        umov ur9, 0
        umov ur10, 1024
        umov ur11, 0
    )";
    // C4, new direction: an MMA accepted while an acc.st on its result is
    // still writing tmem.
    auto message = run_expecting_panic(profile, compile(header + R"(
        acc.st.v8 [ur8, 64], r16
        mma.async.init.m64n64k8 [ur9], [ur0], [ur1]
        halt
    )"), one_warp(), "MMA over an in-flight acc.st");
    require_mentions(message, {"tmem race: mma.async #1 on tmem 0 (64 words per row) rows [0, 64) cols [0, 64)",
        "acc.st #0", "write-owner of tmem 0 (64 words per row) rows [0, 16) cols [0, 8) until its final tmem write",
        "wait that transfer's depbar sb"}, "MMA vs acc.st");
    // ... and while an acc.ld still reads it (WAR on tmem). I2.1 (F6): the
    // read-ownership ends at the final tmem read (I8: the last beat's tmem
    // sample), not the final RF write.
    message = run_expecting_panic(profile, compile(header + R"(
        acc.ld.v8 r24, [ur8, 64]
        mma.async.init.m64n64k8 [ur9], [ur0], [ur1]
        halt
    )"), one_warp(), "MMA over an in-flight acc.ld");
    require_mentions(message, {"tmem race: mma.async #1 on tmem 0", "acc.ld #0",
        "read-owner of tmem 0 (64 words per row) rows [0, 16) cols [0, 8) until its final tmem read"},
        "MMA vs acc.ld");
    // Existing direction, same text plus the owner.
    message = run_expecting_panic(profile, compile(header + R"(
        mma.async.init.m64n64k8 [ur9], [ur0], [ur1]
        acc.ld.v8 r24, [ur8, 64]
        halt
    )"), one_warp(), "acc.ld over accepted MMA work");
    require_mentions(message, {"accumulator transfer of tmem 0 (64 words per row) rows [0, 16) cols [0, 8) "
        "races accepted MMA work: mma.async #1", "write-owner of tmem 0 (64 words per row) rows [0, 64) cols [0, 64) "
        "until its last result retires", "wait on the mma.commit mbar phase first"}, "acc.ld vs MMA");
    // Transfer vs transfer (I8): the acc.ld and acc.st engines run side by
    // side and nothing orders them, so an overlapping pair is a race. qtxc
    // orders one warp's transfers by depbar; hand assembly without that wait
    // shows the sanitizer's diagnosis.
    const std::string hand_header = R"(
        umov ur8, 0
        umov ur9, 0
        umov ur10, 1024
        umov ur11, 0
    )";
    message = run_expecting_panic(profile, hand_assemble(i1_asm(profile, hand_header + R"(
        acc.st.v8 [ur8, 64], r16 {pred=PT, stall=0, set=sb0, wait=-}
        acc.ld.v8 r24, [ur8, 64] {pred=PT, stall=0, set=sb1, wait=-}
        halt {pred=PT, stall=0, set=-, wait=sb0|sb1}
    )"), profile), one_warp(), "acc.ld over an in-flight acc.st");
    require_mentions(message, {"tmem race: acc.ld reads tmem 0 (64 words per row) rows [0, 16) cols [0, 8)",
        "overlapping acc.st #0", "wait that transfer's depbar sb"}, "acc.ld vs acc.st");
    message = run_expecting_panic(profile, hand_assemble(i1_asm(profile, hand_header + R"(
        acc.ld.v8 r24, [ur8, 64] {pred=PT, stall=0, set=sb1, wait=-}
        acc.st.v8 [ur8, 64], r16 {pred=PT, stall=0, set=sb0, wait=-}
        halt {pred=PT, stall=0, set=-, wait=sb0|sb1}
    )"), profile), one_warp(), "acc.st over an in-flight acc.ld");
    require_mentions(message, {"tmem race: acc.st writes tmem 0 (64 words per row) rows [0, 16) cols [0, 8)",
        "overlapping acc.ld #0"}, "acc.st vs acc.ld");
    // Disjoint rows: concurrent, legal, and exact.
    nanosm::Machine disjoint(profile, hand_assemble(i1_asm(profile, hand_header + R"(
        mov r16, 41
        acc.st.v8 [ur10, 64], r16 {pred=PT, stall=0, set=sb0, wait=-}
        acc.ld.v8 r24, [ur8, 64] {pred=PT, stall=0, set=sb1, wait=-}
        halt {pred=PT, stall=0, set=-, wait=sb0|sb1}
    )"), profile), one_warp());
    disjoint.run(2000);
    require(disjoint.read_reg(0, 0, 24) == 0, "disjoint acc.ld read the concurrent store's rows");
}

// Pointers alias (trash/acc-pointer-2026-11-19.md): an m32n64 matmul writes
// rows 32..63 of a 64-row view at 0 (its pointer 2048). A transfer through a
// DIFFERENT pointer that names some of those words races it -- acc.ld at row
// 48 (3072) while the matmul runs PANICs; one at rows 0..15 (pointer 0) and
// one at rows 16..31 (1024) touch none of its words and pass, concurrently,
// reading what the matmul does not write.
void test_sanitizer_tmem_race_by_words() {
    const auto& profile = canonical_profile();
    const auto launch = [] {
        nanosm::Launch result;
        result.active_warps = 1;
        image_bytes(result.shmem_image, 4096);
        return result;
    };
    const std::string header = R"(
        umov ur0, 0
        umov ur1, 1024
        umov ur9, 2048
        sread r16, LANE_ID
        umov ur10, 0
        acc.st [ur10, 64], r16
        umov ur10, 1024
        acc.st [ur10, 64], r16
        mma.async.init.m32n64k8 [ur9], [ur0], [ur1]
    )";
    auto message = run_expecting_panic(profile, compile(header + R"(
        umov ur12, 3072
        acc.ld.v8 r24, [ur12, 64]
        halt
    )"), launch(), "acc.ld at row 48 of a 64-row view while an m32 matmul writes rows 32..63");
    require_mentions(message, {"accumulator transfer of tmem 3072 (64 words per row) rows [0, 16) cols [0, 8) "
        "races accepted MMA work: mma.async #1", "write-owner of tmem 2048 (64 words per row) rows [0, 32) cols [0, 64)",
        "wait on the mma.commit mbar phase first"}, "a transfer over a matmul's words through another pointer");
    nanosm::Machine clean(profile, compile(header + R"(
        umov ur12, 0
        acc.ld r24, [ur12, 64]
        umov ur12, 1024
        acc.ld r25, [ur12, 64]
        halt
    )"), launch());
    clean.run(20000);
    require(clean.done(), "the disjoint transfers did not finish");
    for (uint32_t lane = 0; lane < profile.lanes; ++lane) {
        require(clean.read_reg(0, lane, 24) == lane && clean.read_reg(0, lane, 25) == lane,
            "a transfer beside a running matmul did not read the words it named");
    }
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
        // One descriptor (512 fp16 at gmem 0); each stage's command names its
        // destination (ur5..ur8) and stage mbar (ur2, ur3) in uniform
        // registers, its coordinate ur9.
        nanosm::TmaDescriptor descriptor;
        descriptor.element_bytes = 2;
        descriptor.dimensions = {512, 1, 1, 1};
        descriptor.tile_length = 512;
        descriptor.tile_rows = 1;
        write_tma_descriptor(shmem, 4096, descriptor);
        for (uint32_t mbar : {8192u, 8200u, 8208u}) write_count_mbar(shmem, mbar, 1);
        nanosm::Machine machine(profile, compile(R"(
            umov ur14, 0
            umov ur5, 0
            umov ur6, 1024
            umov ur7, 2048
            umov ur8, 3072
            umov ur9, 0
            umov ur2, 8192
            umov ur3, 8200
            umov ur4, 8208
            mbar.arrive.expect_tx [ur2], 2048
            tma.load.1d [ur5], [4096, ur9], [ur2]
            tma.load.1d [ur6], [4096, ur9], [ur2]
        L0:
            mbar.try_wait p1, [ur2], 0
            @!p1 bra.uni L0
            umov ur0, 0
            umov ur1, 1024
            mma.async.init.m64n64k8 [ur14], [ur0], [ur1]
            mbar.arrive.expect_tx [ur3], 2048
            tma.load.1d [ur7], [4096, ur9], [ur3]
            tma.load.1d [ur8], [4096, ur9], [ur3]
        L1:
            mbar.try_wait p1, [ur3], 0
            @!p1 bra.uni L1
            umov ur0, 2048
            umov ur1, 3072
            mma.async.m64n64k8 [ur14], [ur0], [ur1]
            mbar.arrive.expect_tx [ur2], 2048
            tma.load.1d [ur5], [4096, ur9], [ur2]
            tma.load.1d [ur6], [4096, ur9], [ur2]
        L2:
            mbar.try_wait p1, [ur2], 1
            @!p1 bra.uni L2
            umov ur0, 0
            umov ur1, 1024
            mma.async.m64n64k8 [ur14], [ur0], [ur1]
            mma.commit [ur4]
        L3:
            mbar.try_wait p1, [ur4], 0
            @!p1 bra.uni L3
            umov ur12, 0
            acc.ld.v8 r16, [ur12, 64]
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
            umov ur14, 0
            umov ur10, 8192
            umov ur11, 8200
            usread ur0, WARP_ID
            ur2r r0, ur0
            setp.eq.u32 p0, r0, 0
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
            setp.lt.u32 p1, r25, 1024
            @p1 bra.uni FILL1
            mbar.arrive [ur10]
        DONE0:
            mbar.try_wait p1, [ur11], 0
            @!p1 bra.uni DONE0
            mov r21, 16384
            shli r22, r21, 16
            or r21, r21, r22
            mov r25, 0
        FILL2:
            add r26, r20, r25
            sts [r26], r21
            addi r25, r25, 64
            setp.lt.u32 p1, r25, 1024
            @p1 bra.uni FILL2
            mbar.arrive [ur10]
            halt
        ISSUER:
            umov ur0, 0
            umov ur1, 1024
        READY0:
            mbar.try_wait p1, [ur10], 0
            @!p1 bra.uni READY0
            mma.async.init.m64n64k8 [ur14], [ur0], [ur1]
            mma.commit [ur11]
        WAIT0:
            mbar.try_wait p1, [ur11], 0
            @!p1 bra.uni WAIT0
        READY1:
            mbar.try_wait p1, [ur10], 1
            @!p1 bra.uni READY1
            mma.async.m64n64k8 [ur14], [ur0], [ur1]
            mma.commit [ur11]
        WAIT1:
            mbar.try_wait p1, [ur11], 1
            @!p1 bra.uni WAIT1
            umov ur12, 0
            acc.ld.v8 r16, [ur12, 64]
            halt
        )"), std::move(launch));
        machine.run(50000);
        for (uint32_t lane = 0; lane < profile.lanes; ++lane)
            for (uint32_t j = 0; j < 8; ++j)
                require(machine.read_reg(0, lane, 16 + j) == nanosm::float_bits(24.0f),
                    "P-tile handoff produced the wrong value");
    }
}

// Row spacing (src/sanitizer.hpp record_mma_row_spacing): a reader states the
// row spacing its region's writer used. Shared memory: a TMA load lands a
// swizzle64 tile of 64 rows of 128 bytes at 0; a matmul reading it as A with
// [ur0, 64] PANICs, with [ur0, 128] runs clean; sts over the words it reads
// makes the tile hand-written, and [ur0, 64] is clean again.
void test_sanitizer_row_spacing_shmem() {
    const auto& profile = canonical_profile();
    const auto launch = [] {
        nanosm::Launch result;
        result.active_warps = 1;
        auto& shmem = image_bytes(result.shmem_image, 9344);
        auto& gmem = image_bytes(result.gmem_image, 8192);
        for (uint32_t i = 0; i < 4096; ++i) write_u16(gmem, 2 * i, 0x3c00);
        nanosm::TmaDescriptor descriptor;
        descriptor.element_bytes = 2;
        descriptor.dimensions = {64, 64, 1, 1};
        descriptor.strides = {64, 0, 0};
        descriptor.tile_length = 64;
        descriptor.tile_rows = 64;
        descriptor.layout = nanosm::TileLayout::swizzle64;
        write_tma_descriptor(shmem, 9280, descriptor);
        write_tx_mbar(shmem, 9216, 8192);
        write_count_mbar(shmem, 9224, 1);
        return result;
    };
    const std::string load = tma_operands_asm(10, 0, 4, 9216, 8, {0, 0}) + R"(
        tma.load.2d [ur10], [9280, ur8], [ur4]
    LANDED:
        mbar.try_wait p0, [ur4], 0
        @!p0 bra.uni LANDED
    )";
    const auto matmul = [](uint32_t spacing) {
        return R"(
            umov ur0, 0
            umov ur1, 8192
            umov ur5, 0
            umov ur6, 9224
            mma.async.init.m64n64k8 [ur5], [ur0, )" + std::to_string(spacing) + R"(], [ur1]
            mma.commit [ur6]
        DONE:
            mbar.try_wait p0, [ur6], 0
            @!p0 bra.uni DONE
            halt
        )";
    };
    const auto message = run_expecting_panic(profile, compile(load + matmul(64)), launch(),
        "a matmul reading a 128-byte-row TMA tile with rows 64 bytes apart");
    require_mentions(message, {"row spacing mismatch: mma.async #1 A operand reads shmem byte 0",
        "with rows 64 bytes apart, but a TMA load into a swizzle64 tile wrote it with rows 128 bytes apart"},
        "shmem row spacing mismatch");
    for (const auto& clean : {load + matmul(128), load + R"(
            sread r20, LANE_ID
            shli r20, r20, 2
            mov r21, 0
        HAND:
            sts [r20], r21
            addi r20, r20, 64
            setp.lt.u32 p1, r20, 4096
            @p1 bra.uni HAND
            fence.sh
        )" + matmul(64)}) {
        nanosm::Machine machine(profile, compile(clean), launch());
        machine.run(50000);
        require(machine.done(), "a matmul reading its tile with the writer's spacing (or a hand-written one) did not finish");
    }
}

// Tensor-core memory: acc.st with rows 64 words apart, acc.ld of the same
// pointer with 32 PANICs; with 64 it reads back what was stored.
void test_sanitizer_row_spacing_tmem() {
    const auto& profile = canonical_profile();
    const auto one_warp = [] {
        nanosm::Launch launch;
        launch.active_warps = 1;
        return launch;
    };
    const auto program = [](uint32_t spacing) {
        return R"(
            umov ur8, 0
            sread r16, LANE_ID
            acc.st [ur8, 64], r16
            acc.ld r24, [ur8, )" + std::to_string(spacing) + R"(]
            halt
        )";
    };
    const auto message = run_expecting_panic(profile, compile(program(32)), one_warp(),
        "acc.ld with rows 32 words apart of what acc.st wrote with 64");
    require_mentions(message, {"row spacing mismatch: acc.ld reads tmem word",
        "with rows 32 words apart, but an acc.st wrote it with rows 64 words apart"}, "tmem row spacing mismatch");
    nanosm::Machine machine(profile, compile(program(64)), one_warp());
    machine.run(20000);
    for (uint32_t lane = 0; lane < profile.lanes; ++lane)
        require(machine.read_reg(0, lane, 24) == lane, "acc.ld with the acc.st's spacing did not read it back");
}

// A swizzle64 TMA store checks the row spacing of the words it READS, at its
// source rounds' grants -- not its whole tile (Astra's false alarm,
// trash/chan.txt 2026-11-20; trash/tma_clipped_spacing_astra.launch.py). A
// 2-row fp32 tile of 64-byte stored rows at shmem 1024 is stored to a
// tensor one row tall; a swizzle64 load with 32-byte rows wrote `load_at`
// (64 bytes) before it. Store coordinate y = 0 reads row 0 (1024..1087)
// only; y = 1 reads nothing.
nanosm::Launch tma_store_spacing_launch() {
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.shmem_image, 1152);
    auto& gmem = image_bytes(launch.gmem_image, 384);
    std::fill(shmem.begin() + 1024, shmem.end(), 0x61);
    std::fill(gmem.begin(), gmem.begin() + 64, 0x34);
    std::fill(gmem.begin() + 256, gmem.end(), 0xcc);
    nanosm::TmaDescriptor spaced; // 2 rows of 8 fp32: 32-byte stored rows
    spaced.dimensions = {8, 2, 1, 1};
    spaced.strides = {8, 0, 0};
    spaced.tile_length = 8;
    spaced.tile_rows = 2;
    spaced.layout = nanosm::TileLayout::swizzle64;
    write_tma_descriptor(shmem, 0, spaced);
    nanosm::TmaDescriptor clipped; // a 2-row tile of 16 fp32 (64-byte rows) of a 1-row tensor
    clipped.base = 256;
    clipped.dimensions = {16, 1, 1, 1};
    clipped.strides = {16, 0, 0};
    clipped.tile_length = 16;
    clipped.tile_rows = 2;
    clipped.layout = nanosm::TileLayout::swizzle64;
    write_tma_descriptor(shmem, 64, clipped);
    write_tx_mbar(shmem, 128, 64);
    write_tx_mbar(shmem, 136, 128);
    return launch;
}

std::string tma_store_spacing_program(uint32_t load_at, int32_t store_y) {
    return tma_operands_asm(1, load_at, 3, 128, 8, {0, 0}) + R"(
        tma.load.2d [ur1], [0, ur8], [ur3]
    LOADED:
        mbar.try_wait p0, [ur3], 0
        @!p0 bra.uni LOADED
    )" + tma_operands_asm(2, 1024, 4, 136, 10, {0, store_y}) + R"(
        tma.store.2d [64, ur10], [ur2], [ur4]
    STORED:
        mbar.try_wait p0, [ur4], 0
        @!p0 bra.uni STORED
        halt
    )";
}

// The rows a store does not read are not checked: the clipped store (row 1,
// 1088..1151, has the load's 32-byte spacing) and a wholly out-of-range one
// run clean and write what they should.
void test_sanitizer_tma_store_spacing_unread_rows() {
    const auto& profile = canonical_profile();
    for (const int32_t y : {0, 1}) {
        nanosm::Machine machine(profile, compile(tma_store_spacing_program(1088, y)), tma_store_spacing_launch());
        machine.run(20000);
        require(machine.done(), "a store over rows it does not read did not finish (y = " + std::to_string(y) + ")");
        for (uint32_t byte = 256; byte < 384; ++byte) {
            const uint8_t expected = y == 0 && byte < 320 ? 0x61 : 0xcc;
            require(machine.gmem()[byte] == expected, "clipped store (y = " + std::to_string(y) + "): gmem byte " +
                std::to_string(byte));
        }
    }
}

// A mismatch in the row the store reads still PANICs: the 32-byte-row load
// lands at 1024, row 0 of the store.
void test_sanitizer_tma_store_spacing_read_row_panics() {
    const auto message = run_expecting_panic(canonical_profile(), compile(tma_store_spacing_program(1024, 0)),
        tma_store_spacing_launch(), "a swizzle64 store reading a row written with another spacing");
    require_mentions(message, {"row spacing mismatch: tma store #", "(swizzle64 source) reads shmem byte 10",
        "with rows 64 bytes apart, but a TMA load into a swizzle64 tile wrote it with rows 32 bytes apart",
        "tma.store.2d"}, "tma store row spacing mismatch");
}

} // namespace

// a swizzle64 TMA store is not checked on rows it does not read (clipped, or wholly out of range)
TEST(Sanitizer, sanitizer_tma_store_spacing_unread_rows) {
    test_sanitizer_tma_store_spacing_unread_rows();
}

// a swizzle64 TMA store reading a word written with another row spacing PANICs
TEST(Sanitizer, sanitizer_tma_store_spacing_read_row_panics) {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    test_sanitizer_tma_store_spacing_read_row_panics();
}

TEST(Sanitizer, sanitizer_row_spacing_shmem) {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    test_sanitizer_row_spacing_shmem();
}

TEST(Sanitizer, sanitizer_row_spacing_tmem) {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    test_sanitizer_row_spacing_tmem();
}

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

// sanitizer: a transfer races a matmul on shared words through another
// pointer; disjoint words do not
TEST(Sanitizer, sanitizer_tmem_race_by_words) {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    test_sanitizer_tmem_race_by_words();
}

// sanitizer: double buffer and P tile clean
TEST(Sanitizer, sanitizer_double_buffer_and_p_tile_are_clean) {
    test_sanitizer_double_buffer_and_p_tile_are_clean();
}

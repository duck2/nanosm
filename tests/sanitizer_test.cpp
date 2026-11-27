// The race sanitizer's ownership checks: shared memory and tmem owned by TMA
// commands, MMA commands and accumulator transfers; row spacing.
#include "test_support.hpp"

using namespace nanosm_test;

namespace {

// A TMA command's setup: a descriptor at 4096 for 1 KiB at gmem 0 (words
// 5 i + 2), completion mbars at 4160 and 4168 expecting it, shmem words
// 0..255 holding 3 i + 1, a commit mbar at 8192, and `mbars` written over them.
nanosm::Launch tma_launch(const std::vector<std::pair<uint32_t, nanosm::MbarState>>& mbars = {}) {
    auto launch = launch_warps(1);
    auto& shmem = image_bytes(launch.test_shmem, 8200);
    auto& gmem = image_bytes(launch.gmem_image, 2048);
    for (uint32_t i = 0; i < 256; ++i) {
        write_u32(shmem, 4 * i, 3 * i + 1);
        write_u32(gmem, 4 * i, 5 * i + 2);
    }
    write_tma_descriptor(shmem, 4096, plain_tma_1d(256));
    write_mbar(shmem, 4160, mbar_state(1, 0, 1024));
    write_mbar(shmem, 4168, mbar_state(1, 0, 1024));
    write_mbar(shmem, 8192, mbar_state(1, 1));
    for (const auto& [address, state] : mbars) write_mbar(shmem, address, state);
    return launch;
}

// A swizzle64 TMA store checks the row spacing of the words it READS, at its
// source rounds' grants, not its whole tile (a clipped store reads only part
// of it). A 2-row fp32 tile of 64-byte stored rows at shmem 1024 is stored to
// a tensor one row tall; a swizzle64 load with 32-byte rows wrote `load_at`
// (64 bytes) before it. Store coordinate y = 0 reads row 0 (1024..1087)
// only; y = 1 reads nothing.
nanosm::Launch tma_store_spacing_launch() {
    auto launch = launch_warps(1);
    auto& shmem = image_bytes(launch.test_shmem, 1152);
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
    write_mbar(shmem, 128, mbar_state(1, 0, 64));
    write_mbar(shmem, 136, mbar_state(1, 0, 128));
    return launch;
}

std::string tma_store_spacing_program(uint32_t load_at, int32_t store_y) {
    return tma_operands_asm(1, load_at, 3, 128, 8, {0, 0}) + "tma.load.2d [ur1], [0, ur8], [ur3]\n" +
        wait_on("ur3", "LOADED") + tma_operands_asm(2, 1024, 4, 136, 10, {0, store_y}) +
        "tma.store.2d [64, ur10], [ur2], [ur4]\n" + wait_on("ur4", "STORED") + "halt\n";
}

} // namespace

// Who owns a shared-memory range from its acceptance, and who may not touch
// it meanwhile: a TMA store read-owns its source until its last source read;
// a TMA load write-owns its destination until its last shmem write commits;
// an MMA command, queued or fetching, read-owns its operands until its final
// operand read. A write into a read-owned range, or any access to a
// write-owned one, PANICs when the accessor is accepted, whatever the port
// timing; two readers never conflict; waiting on the owner's completion
// orders the access.
TEST(Sanitizer, sanitizer_tma_and_mma_accept_conflicts) {
    const auto& profile = canonical_profile();
    const std::string store_source = tma_operands_asm(10, 0, 4, 4160, 8) + "tma.store.1d [4096, ur8], [ur10], [ur4]\n";
    const std::string stage = "sread r20, LANE_ID\nshli r20, r20, 2\nmov r21, 7\nsts [r20], r21\nhalt\n";
    const std::string load_destination = tma_operands_asm(10, 0, 4, 4160, 8) +
        "sread r20, LANE_ID\nshli r20, r20, 2\ntma.load.1d [ur10], [4096, ur8], [ur4]\n";
    const std::string mma_header = "umov ur0, 0\numov ur1, 1024\numov ur2, 0\n";
#if NANOSM_SANITIZER
    // Queue depth 4 lets a command wait behind an unfired commit.
    auto deep_queue = canonical_profile();
    deep_queue.mma_queue_commands = 4;
    nanosm::validate_device_profile(deep_queue);
    const auto sts_into_stage = [](uint32_t address) {
        return "umov ur0, 0\numov ur1, 1024\numov ur2, 2048\numov ur3, 3072\numov ur4, 8192\numov ur5, 0\n"
               "mma.async.init.m64n64k8 [ur5], [ur0], [ur1]\nmma.commit [ur4]\nmma.async.m64n64k8 [ur5], [ur2], [ur3]\n"
               "sread r20, LANE_ID\nshli r20, r20, 2\naddi r20, r20, " + std::to_string(address) +
            "\nmov r21, 0\nsts [r20], r21\nhalt\n";
    };
    struct Race {
        std::string what;
        const nanosm::DeviceProfile* profile;
        std::string kernel;
        std::vector<std::string> needles;
    };
    const std::vector<Race> races{
        {"sts into a live TMA store source", &profile, store_source + stage,
            {"shmem race: sts writes 16 words, bytes [0, 64)", "tma store #0",
                "read-owner of its source [0, 1024) until its last source read", "wait the TMA completion mbar [4160]",
                "inst='sts"}},
        {"sts into a queued MMA's stage", &deep_queue, sts_into_stage(2048),
            {"sts writes 16 words, bytes [2048, 2112)", "mma.async #3 [queued]", "read-owner of its A operand [2048, 3072)",
                "until its final operand read", "wait the operand-consumed depbar sb", "/ commit mbar"}},
        {"sts into an active MMA's stage", &deep_queue, sts_into_stage(0),
            {"sts writes 16 words, bytes [0, 64)", "mma.async #1 [fetching]", "read-owner of its A operand [0, 1024)",
                "wait the operand-consumed depbar sb"}},
        {"lds from a pending TMA load destination", &profile, load_destination + "lds r21, [r20]\nhalt\n",
            {"shmem race: lds reads 16 words, bytes [0, 64)", "tma load #0",
                "write-owner of its destination [0, 1024) until its last shmem write commits",
                "wait the TMA completion mbar [4160]"}},
        {"a TMA load over an active MMA operand", &profile,
            mma_header + tma_operands_asm(10, 0, 11, 4160, 8) +
                "mma.async.init.m64n64k8 [ur2], [ur0], [ur1]\ntma.load.1d [ur10], [4096, ur8], [ur11]\nhalt\n",
            {"shmem race: tma load #0 destination bytes [0, 1024)", "owned by mma.async #1 [fetching]",
                "A operand [0, 1024)", "wait the operand-consumed depbar sb",
                "inst='tma.load.1d [ur10], [4096, ur8], [ur11]"}},
        {"an MMA over a pending TMA load destination", &profile,
            mma_header + tma_operands_asm(10, 1024, 11, 4168, 8) +
                "tma.load.1d [ur10], [4096, ur8], [ur11]\nmma.async.init.m64n64k8 [ur2], [ur0], [ur1]\nhalt\n",
            {"shmem race: mma.async #1 B operand bytes [1024, 2048)", "owned by tma load #0",
                "wait the TMA completion mbar [4168]"}},
        // Both commands' buffer and coordinate registers are ur10 and ur8.
        {"a TMA store of a pending TMA load destination", &profile,
            tma_operands_asm(10, 0, 11, 4160, 8) +
                "umov ur12, 4168\ntma.load.1d [ur10], [4096, ur8], [ur11]\ntma.store.1d [4096, ur8], [ur10], [ur12]\nhalt\n",
            {"shmem race: tma store #1 source bytes [0, 1024)", "owned by tma load #0",
                "wait the TMA completion mbar [4160]"}},
    };
    for (const auto& race : races) {
        const auto message = panic_of(*race.profile, compile(race.kernel, *race.profile), tma_launch(), race.what);
        for (const auto& needle : race.needles)
            require(message.find(needle) != std::string::npos, race.what + ": diagnostic lacks '" + needle + "': " + message);
    }
#endif

    nanosm::Machine waited_store(profile, compile(store_source + wait_on("ur4") + stage), tma_launch());
    waited_store.run(20000);
    for (uint32_t i = 0; i < 256; ++i)
        require(waited_store.read_gmem_u32(4 * i) == 3 * i + 1, "the waited store captured a late sts");
    require(waited_store.read_shmem_u32(0) == 7, "the waited sts did not execute");

    nanosm::Machine waited_load(profile, compile(load_destination + wait_on("ur4") + "lds r21, [r20]\nhalt\n"), tma_launch());
    waited_load.run(20000);
    require_lanes(waited_load, 21, [](uint32_t lane) { return 5 * lane + 2; }, "the waited lds");

    // Two readers: a TMA store of an MMA operand stage is legal.
    nanosm::Machine readers(profile, compile(mma_header + tma_operands_asm(10, 1024, 4, 4168, 8) +
        "mma.async.init.m64n64k8 [ur2], [ur0], [ur1]\ntma.store.1d [4096, ur8], [ur10], [ur4]\n" + wait_on("ur4") +
        "halt\n"), tma_launch());
    readers.run(20000);
}

// An mbar RMW inside an owned range is an access too: a SIMT arrive whose
// mbar sits in a pending TMA load's destination, and an engine RMW (a TMA's
// complete_tx) whose mbar sits in an operand stage two queued MMAs still read.
TEST(Sanitizer, sanitizer_mbar_rmw_into_owned_range) {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    const auto& profile = canonical_profile();
    auto message = panic_of(profile, compile(tma_operands_asm(10, 0, 11, 4160, 8) + R"(
        umov ur4, 512
        tma.load.1d [ur10], [4096, ur8], [ur11]
        mbar.arrive [ur4]
        halt
    )"), tma_launch({{512, mbar_state(2, 2)}}), "mbar.arrive inside a TMA load destination");
    require_mentions(message, {"shmem race: mbar.arrive writes mbar bytes [512, 520)", "tma load #0",
        "wait the TMA completion mbar [4160]", "warp=0"}, "SIMT RMW ownership race");
    message = panic_of(profile, compile(R"(
        umov ur0, 0
        umov ur1, 1024
        umov ur2, 0
    )" + tma_operands_asm(10, 2048, 11, 512, 8) + R"(
        mma.async.init.m64n64k8 [ur2], [ur0], [ur1]
        mma.async.m64n64k8 [ur2], [ur0], [ur1]
        tma.load.1d [ur10], [4096, ur8], [ur11]
        halt
    )"), tma_launch({{512, mbar_state(1, 0, 1024)}}), "engine complete_tx into an MMA operand stage");
    require_mentions(message, {"engine complete_tx writes mbar bytes [512, 520)",
        "read-owner of its A operand [0, 1024)", "source=engine"}, "engine RMW ownership race");
}

// tmem ownership is by the words touched (pointers, src/layout.hpp: ur8 rows
// 0..15 and ur10 rows 16..31 of a region of 64-word rows at 0, ur9 the
// matmul's result, 64x64 at 0). An MMA accepted while an acc.st on its result
// still writes tmem, or while an acc.ld still reads it (until its final tmem
// read, the last beat's sample, not its final RF write), races it; so does a
// transfer over accepted MMA work. The acc.ld and acc.st engines run side by
// side and nothing orders them, so an overlapping pair is a race too (qtxc
// orders one warp's transfers by depbar; hand assembly without the wait shows
// the diagnosis); disjoint rows run concurrently and exactly.
TEST(Sanitizer, sanitizer_tmem_ownership_both_directions) {
    const auto& profile = canonical_profile();
    const std::string hand_header = R"(
        umov ur8, 0
        umov ur9, 0
        umov ur10, 1024
        umov ur11, 0
    )";
#if NANOSM_SANITIZER
    const std::string header = "umov ur0, 0\numov ur1, 1024\n" + hand_header;
    auto message = panic_of(profile, compile(header + R"(
        acc.st.v8 [ur8, 64], r16
        mma.async.init.m64n64k8 [ur9], [ur0], [ur1]
        halt
    )"), launch_warps(1), "MMA over an in-flight acc.st");
    require_mentions(message, {"tmem race: mma.async #1 on tmem 0 (64 words per row) rows [0, 64) cols [0, 64)",
        "acc.st #0", "write-owner of tmem 0 (64 words per row) rows [0, 16) cols [0, 8) until its final tmem write",
        "wait that transfer's depbar sb"}, "MMA vs acc.st");
    message = panic_of(profile, compile(header + R"(
        acc.ld.v8 r24, [ur8, 64]
        mma.async.init.m64n64k8 [ur9], [ur0], [ur1]
        halt
    )"), launch_warps(1), "MMA over an in-flight acc.ld");
    require_mentions(message, {"tmem race: mma.async #1 on tmem 0", "acc.ld #0",
        "read-owner of tmem 0 (64 words per row) rows [0, 16) cols [0, 8) until its final tmem read"}, "MMA vs acc.ld");
    message = panic_of(profile, compile(header + R"(
        mma.async.init.m64n64k8 [ur9], [ur0], [ur1]
        acc.ld.v8 r24, [ur8, 64]
        halt
    )"), launch_warps(1), "acc.ld over accepted MMA work");
    require_mentions(message, {"accumulator transfer of tmem 0 (64 words per row) rows [0, 16) cols [0, 8) "
        "races accepted MMA work: mma.async #1", "write-owner of tmem 0 (64 words per row) rows [0, 64) cols [0, 64) "
        "until its last result retires", "wait on the mma.commit mbar phase first"}, "acc.ld vs MMA");
    message = panic_of(profile, hand_assemble(hand_asm(profile, hand_header + R"(
        acc.st.v8 [ur8, 64], r16 {pred=PT, stall=0, set=sb0, wait=-}
        acc.ld.v8 r24, [ur8, 64] {pred=PT, stall=0, set=sb1, wait=-}
        halt {pred=PT, stall=0, set=-, wait=sb0|sb1}
    )"), profile), launch_warps(1), "acc.ld over an in-flight acc.st");
    require_mentions(message, {"tmem race: acc.ld reads tmem 0 (64 words per row) rows [0, 16) cols [0, 8)",
        "overlapping acc.st #0", "wait that transfer's depbar sb"}, "acc.ld vs acc.st");
    message = panic_of(profile, hand_assemble(hand_asm(profile, hand_header + R"(
        acc.ld.v8 r24, [ur8, 64] {pred=PT, stall=0, set=sb1, wait=-}
        acc.st.v8 [ur8, 64], r16 {pred=PT, stall=0, set=sb0, wait=-}
        halt {pred=PT, stall=0, set=-, wait=sb0|sb1}
    )"), profile), launch_warps(1), "acc.st over an in-flight acc.ld");
    require_mentions(message, {"tmem race: acc.st writes tmem 0 (64 words per row) rows [0, 16) cols [0, 8)",
        "overlapping acc.ld #0"}, "acc.st vs acc.ld");
#endif
    // Disjoint rows: the store writes rows 16..31 while the load reads the
    // untouched rows 0..15.
    nanosm::Machine disjoint(profile, hand_assemble(hand_asm(profile, hand_header + R"(
        mov r16, 41
        acc.st.v8 [ur10, 64], r16 {pred=PT, stall=0, set=sb0, wait=-}
        acc.ld.v8 r24, [ur8, 64] {pred=PT, stall=0, set=sb1, wait=-}
        halt {pred=PT, stall=0, set=-, wait=sb0|sb1}
    )"), profile), launch_warps(1));
    disjoint.run(2000);
    for (uint32_t j = 0; j < 8; ++j) {
        require_lanes(disjoint, 24 + j, 0, "the disjoint acc.ld (the concurrent store's rows are not its own)");
    }
    require_lanes([&](uint32_t lane) {
        return disjoint.tmem_word(static_cast<uint32_t>(nanosm::tmem_physical_word(1024 + 64 * lane, 64)));
    }, 41, "the concurrent acc.st's row 16 + lane");
}

// Pointers alias: an m32n64 matmul writes rows 32..63 of a 64-row view at 0
// (its pointer 2048). A transfer through a DIFFERENT pointer that names some
// of those words races it (an acc.ld at row 48, pointer 3072); one at rows
// 0..15 (pointer 0) and one at rows 16..31 (1024) touch none of its words
// and run concurrently, reading what the matmul does not write.
TEST(Sanitizer, sanitizer_tmem_race_by_words) {
    const auto& profile = canonical_profile();
    const auto launch = [] {
        auto result = launch_warps(1);
        image_bytes(result.test_shmem, 4096);
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
#if NANOSM_SANITIZER
    const auto message = panic_of(profile, compile(header + R"(
        umov ur12, 3072
        acc.ld.v8 r24, [ur12, 64]
        halt
    )"), launch(), "acc.ld at row 48 of a 64-row view while an m32 matmul writes rows 32..63");
    require_mentions(message, {"accumulator transfer of tmem 3072 (64 words per row) rows [0, 16) cols [0, 8) "
        "races accepted MMA work: mma.async #1", "write-owner of tmem 2048 (64 words per row) rows [0, 32) cols [0, 64)",
        "wait on the mma.commit mbar phase first"}, "a transfer over a matmul's words through another pointer");
#endif
    nanosm::Machine clean(profile, compile(header + R"(
        umov ur12, 0
        acc.ld r24, [ur12, 64]
        umov ur12, 1024
        acc.ld r25, [ur12, 64]
        halt
    )"), launch());
    clean.run(20000);
    for (const uint32_t reg : {24u, 25u}) {
        require_lanes(clean, reg, [](uint32_t lane) { return lane; }, "a transfer beside a running matmul");
    }
}

// Row spacing (src/machine/sanitizer.hpp record_mma_row_spacing): a reader
// states the row spacing its region's writer used. Shared memory: a TMA load
// lands a swizzle64 tile of 64 rows of 128 bytes at 0; a matmul reading it as
// A with [ur0, 64] PANICs, with [ur0, 128] runs clean; sts over the words it
// reads makes the tile hand-written, and [ur0, 64] is clean again.
TEST(Sanitizer, sanitizer_row_spacing_shmem) {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    const auto& profile = canonical_profile();
    const auto launch = [] {
        auto result = launch_warps(1);
        auto& shmem = image_bytes(result.test_shmem, 9344);
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
        write_mbar(shmem, 9216, mbar_state(1, 0, 8192));
        return result;
    };
    const std::string load = tma_operands_asm(10, 0, 4, 9216, 8, {0, 0}) + "tma.load.2d [ur10], [9280, ur8], [ur4]\n" +
        wait_on("ur4", "LANDED");
    const auto matmul = [](uint32_t spacing) {
        return mma_kernel({.b = 8192, .a_spacing = spacing, .mbar = 9224});
    };
    const auto message = panic_of(profile, compile(load + matmul(64)), launch(),
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
    }
}

// Tensor-core memory: acc.st with rows 64 words apart, acc.ld of the same
// pointer with 32 PANICs; with 64 it reads back what was stored.
TEST(Sanitizer, sanitizer_row_spacing_tmem) {
    const auto& profile = canonical_profile();
    const auto program = [](uint32_t spacing) {
        return "umov ur8, 0\nsread r16, LANE_ID\nacc.st [ur8, 64], r16\nacc.ld r24, [ur8, " + std::to_string(spacing) +
            "]\nhalt\n";
    };
#if NANOSM_SANITIZER
    const auto message = panic_of(profile, compile(program(32)), launch_warps(1),
        "acc.ld with rows 32 words apart of what acc.st wrote with 64");
    require_mentions(message, {"row spacing mismatch: acc.ld reads tmem word",
        "with rows 32 words apart, but an acc.st wrote it with rows 64 words apart"}, "tmem row spacing mismatch");
#endif
    nanosm::Machine machine(profile, compile(program(64)), launch_warps(1));
    machine.run(20000);
    require_lanes(machine, 24, [](uint32_t lane) { return lane; }, "acc.ld with the acc.st's spacing");
}

// A swizzle64 TMA store is not checked on rows it does not read: the clipped
// store (row 1, 1088..1151, has the load's 32-byte spacing) and a wholly
// out-of-range one run clean and write what they should. A mismatch in the
// row the store reads still PANICs: the 32-byte-row load lands at 1024, row 0
// of the store.
TEST(Sanitizer, sanitizer_tma_store_spacing_unread_rows) {
    const auto& profile = canonical_profile();
    for (const int32_t y : {0, 1}) {
        nanosm::Machine machine(profile, compile(tma_store_spacing_program(1088, y)), tma_store_spacing_launch());
        machine.run(20000);
        for (uint32_t byte = 256; byte < 384; ++byte) {
            const uint8_t expected = y == 0 && byte < 320 ? 0x61 : 0xcc;
            require(machine.gmem()[byte] == expected,
                "clipped store (y = " + std::to_string(y) + "): gmem byte " + std::to_string(byte));
        }
    }
#if NANOSM_SANITIZER
    const auto message = panic_of(profile, compile(tma_store_spacing_program(1024, 0)), tma_store_spacing_launch(),
        "a swizzle64 store reading a row written with another spacing");
    require_mentions(message, {"row spacing mismatch: tma store #", "(swizzle64 source) reads shmem byte 10",
        "with rows 64 bytes apart, but a TMA load into a swizzle64 tile wrote it with rows 32 bytes apart",
        "tma.store.2d"}, "tma store row spacing mismatch");
#endif
}

// The sanitizer's MMA operand spans come from the command's shape: an m32
// command on A at 0 read-owns [0, 512), so an sts into the other half of a
// 64-row object ([512, 1024)) is quiet while one inside its span PANICs.
TEST(Sanitizer, operand_span_follows_the_shape) {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    const auto& profile = canonical_profile();
    const auto program = [](uint32_t sts_base) {
        return "umov ur0, 0\numov ur1, 2048\numov ur4, 8192\numov ur2, 4096\nmbar.init [ur4], 1\n"
               "sread r20, LANE_ID\nshli r20, r20, 2\nmov r21, 0\n"
               "mma.async.init.m32n64k8 [ur2], [ur0], [ur1]\n"
               "sts [r20 + " + std::to_string(sts_base) + "], r21\n"
               "mma.commit [ur4]\n" + wait_on("ur4") + "halt\n";
    };
    nanosm::Machine quiet(profile, compile(program(512), profile), launch_warps(1));
    quiet.run(20000);
    const auto message = panic_of(profile, compile(program(256), profile), launch_warps(1),
        "an sts into an m32 command's A rows");
    require_mentions(message, {"mma.async #1", "A operand [0, 512)"}, "m32 operand span");
}

// The tmem pointer rule (src/layout.hpp) is the sanitizer's PANIC: a pointer
// whose row is not a multiple of 4, an acc.ld.v8 whose eight columns leave the
// row, a matmul result not at column 0.
TEST(Sanitizer, tmem_pointer_rule_panics) {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    const auto& profile = canonical_profile();
    nanosm::Machine row(profile, compile("umov ur8, 128\nacc.ld.v8 r16, [ur8, 64]\nhalt\n", profile), launch_warps(1));
    expect_panic(row, {"acc.ld.v8: tmem pointer 128 with 64 words per row is row 2 column 0"}, "a pointer at row 2");
    nanosm::Machine columns(profile, compile("umov ur8, 60\nacc.st.v8 [ur8, 64], r16\nhalt\n", profile), launch_warps(1));
    expect_panic(columns, {"the 8 columns it touches stay inside its row"}, "eight columns from column 60");
    nanosm::Machine result(profile, compile(
        "umov ur0, 0\numov ur1, 2048\numov ur2, 32\nmma.async.init.m32n64k8 [ur2], [ur0], [ur1]\nhalt\n", profile),
        launch_warps(1));
    expect_panic(result, {"tmem pointer 32 with 64 words per row is row 0 column 32"}, "a matmul result at column 32");
}

// An acc.ld's tmem read-ownership ends at its FINAL TMEM READ (its values are
// then in its pipeline), while its destination registers stay not-ready
// until each beat's RF write. Here the acc.ld.v8's last beat (r23, bank 2) is
// denied the RF write port by four fadds booked on bank 2, so it is written a
// few cycles after its tmem read. An mma.async.init on the accumulator
// accepted (at its dispatch, issue_to_dispatch after its issue) after the last
// tmem read but before that RF write is legal, and the acc.ld still returns
// the old values; one accepted before the last tmem read is a tmem race.
TEST(Sanitizer, acc_ld_read_ownership) {
    auto profile = canonical_profile();
    profile.timing.fadd = 16;
    const auto program = [](uint32_t gap) {
        std::string body = R"(
            sread r0, LANE_ID
            umov ur8, 0
            umov ur9, 0
            umov ur1, 0
            umov ur2, 1024
        )"; // ur8: the acc.ld's tmem pointer, ur9 the matmul's (both 0)
        body += "mov r23, 77\n"; // nonzero: the acc.ld's write of the old (zero) tmem value shows
        for (uint32_t i = 0; i < 6; ++i) body += "setp.eq.u32 p" + std::to_string(5 + i % 2) + ", r0, 0\n";
        body += R"(
            fadd r26, r10, r10
            fadd r29, r10, r10
            fadd r32, r10, r10
            fadd r35, r10, r10
            setp.eq.u32 p5, r0, 0
            setp.eq.u32 p6, r0, 0
            setp.eq.u32 p5, r0, 0
            acc.ld.v8 r16, [ur8, 64] {pred=PT, stall=0, set=sb0, wait=-}
        )";
        // One filler per cycle: they rotate over six predicates (a setp
        // waits until the previous write of its predicate is in the file,
        // issue_to_predicate_visible after it, so two alternating predicates
        // would issue in pairs five cycles apart).
        for (uint32_t i = 0; i < gap; ++i) body += "setp.eq.u32 p" + std::to_string(1 + i % 6) + ", r0, 0\n";
        body += R"(
            mma.async.init.m64n64k8 [ur9], [ur1], [ur2] {pred=PT, stall=0, set=sb1, wait=-}
            halt {pred=PT, stall=0, set=-, wait=sb0|sb1}
        )";
        return body;
    };
    const auto machine_for = [&](uint32_t gap) {
        auto launch = launch_warps(1);
        auto& shmem = image_bytes(launch.test_shmem, 4096);
        for (uint32_t i = 0; i < 1024; ++i) write_u16(shmem, 2 * i, 0x3c00); // A and B: 1.0
        return std::make_unique<nanosm::Machine>(profile, hand_assemble(hand_asm(profile, program(gap)), profile),
            std::move(launch), nanosm::TraceOptions{.timeline = true});
    };
    constexpr uint32_t acc_ld_pc = 19 * 8; // the acc.ld is the 20th instruction
    // Reference: the MMA long after; the final tmem read and final RF write.
    auto reference = machine_for(24);
    const auto written = watch_registers(*reference, {23});
    const auto beats = events_named(*reference, "acc.ld beat");
    require(!beats.empty() && written.contains(23), "the reference acc.ld did not run");
    uint64_t final_read = 0;
    for (const auto& beat : beats) final_read = std::max(final_read, beat.start_cycle);
    const auto final_write = written.at(23).back();
    require(issued_at(*reference, acc_ld_pc).size() == 1 && final_write >= final_read + 2,
        "the last beat's RF write is not delayed past its tmem read; the test proves nothing (read c" +
            std::to_string(final_read) + ", write c" + std::to_string(final_write) + ")");
    const auto acc_ld_issue = issued_at(*reference, acc_ld_pc).front();
    uint32_t legal_between = 0;
    [[maybe_unused]] uint32_t races = 0;
    const auto dispatch = nanosm::issue_to_dispatch(profile);
    for (uint32_t gap = 0; acc_ld_issue + 1 + gap + dispatch < final_write; ++gap) {
        auto machine = machine_for(gap);
        const auto mma_issue = acc_ld_issue + 1 + gap;
        const auto mma_accept = mma_issue + dispatch;
        const auto where = "mma.async.init issued c" + std::to_string(mma_issue) + ", accepted c" +
            std::to_string(mma_accept) + " (last tmem read c" + std::to_string(final_read) + ", last RF write c" +
            std::to_string(final_write) + "): ";
        if (mma_accept < final_read) {
#if NANOSM_SANITIZER
            require_mentions(panic_of(*machine, where + "accepted while the acc.ld still reads its region", 5000),
                {"tmem race: mma.async", "acc.ld #0", "until its final tmem read"}, where);
            ++races;
#endif
            continue;
        }
        machine->run(5000); // no PANIC
        require(issued_at(*machine, (19 + 1 + gap) * 8) == std::vector<uint64_t>{mma_issue}, where + "issue cycle moved");
        for (uint32_t reg = 16; reg < 24; ++reg)
            require(machine->read_reg(0, 0, reg) == 0, where + "the acc.ld returned the MMA's result, not the old values");
        ++legal_between;
    }
    require(legal_between >= 2, "the sweep did not reach past the last tmem read");
#if NANOSM_SANITIZER
    require(races != 0, "the sweep did not reach before the last tmem read");
#endif
}

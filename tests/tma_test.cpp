// The TMA's commands: the direction, swizzle64 placement and its census
// against the MMA's operand addressing, the limits of swizzle64 tiles, the
// rows-stay-whole smoke tests,
// out-of-bounds handling; and the profile stamp and the DRAM clock. The
// descriptor (64 bytes in shared memory, src/tma_descriptor.hpp) holds the global
// tensor, the tile's shape and its shared-memory layout (dense, or swizzle64
// and whether it is transposed; the rules are src/layout.hpp's, which the MMA
// reads its operands with). The
// instruction holds the descriptor's shared
// address as a constant and everything else in uniform registers: the
// buffer's shared address, the tile's coordinates (one register per
// dimension, from urC on) and the completion barrier's shared address:
//     tma.load.Nd  [urS], [DESC, urC], [urM]
//     tma.store.Nd [DESC, urC], [urS], [urM]
#include "test_support.hpp"

namespace {

using namespace nanosm_test;

std::string encode_error(const nanosm::TmaDescriptor& descriptor) {
    try {
        (void)nanosm::encode_tma_descriptor(descriptor);
        return {};
    } catch (const nanosm::SimError& error) {
        return error.what();
    }
}

// The direction is the opcode, and one descriptor (16 fp32 at gmem 0) serves
// both: a tma.load into [1024, 1088) completing into the mbar at 128, then a
// tma.store from [2048, 2112) completing into 136. Each command's bytes go to
// its own mbar, and the two opcodes encode, decode and print apart.
void test_tma_direction_is_the_opcode() {
    const auto& profile = canonical_profile();
    for (const std::string text : {"tma.load.1d [ur4], [64, ur8], [ur5]", "tma.store.1d [64, ur8], [ur4], [ur5]"}) {
        const auto program = qtx_program(text + "\nhalt\n", profile);
        require(nanosm::format_instruction(nanosm::decode(nanosm::encode(program.instructions.at(0))), false) == text,
            "'" + text + "' did not survive encode / decode / print");
    }
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.test_shmem, 4096);
    auto& gmem = image_bytes(launch.gmem_image, 64);
    for (uint32_t i = 0; i < 16; ++i) {
        write_u32(gmem, 4 * i, 0x100 + i);
        write_u32(shmem, 2048 + 4 * i, 0x200 + i);
    }
    write_tma_descriptor(shmem, 0, linear_tma(64));
    write_tx_mbar(shmem, 128, 64);
    write_tx_mbar(shmem, 136, 64);
    nanosm::Machine machine(profile, compile(R"(
        umov ur1, 1024
        umov ur2, 2048
        umov ur3, 128
        umov ur4, 136
        umov ur8, 0
        tma.load.1d [ur1], [0, ur8], [ur3]
    loaded:
        mbar.try_wait p0, [ur3], 0
        @!p0 bra.uni loaded
        tma.store.1d [0, ur8], [ur2], [ur4]
    stored:
        mbar.try_wait p0, [ur4], 0
        @!p0 bra.uni stored
        halt
    )"), std::move(launch));
    machine.run(20000);
    for (uint32_t i = 0; i < 16; ++i) {
        require(machine.read_shmem_u32(1024 + 4 * i) == 0x100 + i, "tma.load did not load");
        require(machine.read_gmem_u32(4 * i) == 0x200 + i, "tma.store did not store");
    }
    for (const uint32_t mbar : {128u, 136u}) {
        const auto state = machine.read_mbar_state(mbar);
        require(state.phase && state.tx_bytes == 0, "a command's bytes missed its own mbar " + std::to_string(mbar));
    }
}

// One swizzle64 descriptor, transposed or not, used by a load and then a
// store: an fp16 tile of 64 stored rows of 8 (16-byte rows) at (3, 2) of a
// tensor with row stride 83. Element x of tile row y is position x of stored
// row y, or (transposed) position y of stored row x.
void test_tma_swizzle64_transposed_load_store() {
    constexpr uint32_t mbar_address = 128;
    constexpr uint32_t tile_address = 1024;
    constexpr uint32_t stride = 83;
    const std::string operands = tma_operands_asm(4, tile_address, 3, mbar_address, 8, {3, 2});
    const std::string wait = "wait:\nmbar.try_wait p0, [ur3], 0\n@!p0 bra.uni wait\nhalt\n";
    const std::string load = operands + "tma.load.2d [ur4], [0, ur8], [ur3]\n" + wait;
    const std::string store = operands + "tma.store.2d [0, ur8], [ur4], [ur3]\n" + wait;
    for (bool transposed : {false, true}) {
        const uint32_t tile_length = transposed ? 64 : 8;
        const uint32_t tile_rows = transposed ? 8 : 64;
        const uint32_t dim_x = 3 + tile_length + 5;
        const uint32_t dim_y = 2 + tile_rows + 3;
        nanosm::TmaDescriptor descriptor;
        descriptor.element_bytes = 2;
        descriptor.layout = nanosm::TileLayout::swizzle64;
        descriptor.transposed = transposed;
        descriptor.dimensions = {dim_x, dim_y, 1, 1};
        descriptor.strides = {stride, 0, 0};
        descriptor.tile_length = static_cast<uint16_t>(tile_length);
        descriptor.tile_rows = static_cast<uint16_t>(tile_rows);
        const auto placement = [&](uint32_t x, uint32_t y) {
            return transposed ? reference_swizzle64(tile_address, 16, x, 2 * y)
                              : reference_swizzle64(tile_address, 16, y, 2 * x);
        };

        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.test_shmem, tile_address + 1088);
        auto& gmem = image_bytes(launch.gmem_image, stride * dim_y * 2);
        for (uint32_t y = 0; y < dim_y; ++y)
            for (uint32_t x = 0; x < dim_x; ++x)
                write_u16(gmem, 2 * (y * stride + x),
                    static_cast<uint16_t>(0x100 + 3 * y + 5 * x));
        write_tma_descriptor(shmem, 0, descriptor);
        nanosm::MbarState mbar{false, true, 1, 0, 1024};
        write_u64(shmem, mbar_address, nanosm::encode_mbar_state(mbar));
        nanosm::Machine machine(canonical_profile(), compile(load), std::move(launch));
        machine.run(20000);
        std::set<uint32_t> addresses;
        for (uint32_t y = 0; y < tile_rows; ++y) for (uint32_t x = 0; x < tile_length; ++x) {
            const uint32_t address = placement(x, y);
            require(addresses.insert(address).second, "swizzle64 map is not bijective");
            const uint32_t gx = 3 + x;
            const uint32_t gy = 2 + y;
            const uint16_t actual = static_cast<uint16_t>(
                machine.read_shmem_u32(address & ~3u) >> (8 * (address & 3u)));
            require(actual ==
                static_cast<uint16_t>(0x100 + 3 * gy + 5 * gx),
                "swizzle64 load mapped the wrong element");
        }
        require(addresses.size() == 512, "swizzle64 map did not cover the operand");

        nanosm::Launch store_launch;
        store_launch.active_warps = 1;
        auto& store_shmem = image_bytes(store_launch.test_shmem, tile_address + 1088);
        auto& store_gmem = image_bytes(store_launch.gmem_image, stride * dim_y * 2);
        std::fill(store_gmem.begin(), store_gmem.end(), 0x5a);
        for (uint32_t y = 0; y < tile_rows; ++y) for (uint32_t x = 0; x < tile_length; ++x)
            write_u16(store_shmem, placement(x, y), static_cast<uint16_t>(0x400 + 7 * y + 11 * x));
        write_tma_descriptor(store_shmem, 0, descriptor);
        write_u64(store_shmem, mbar_address, nanosm::encode_mbar_state(mbar));
        nanosm::Machine store_machine(canonical_profile(), compile(store), std::move(store_launch));
        store_machine.run(20000);
        for (uint32_t y = 0; y < tile_rows; ++y) for (uint32_t x = 0; x < tile_length; ++x) {
            const uint32_t gx = 3 + x;
            const uint32_t gy = 2 + y;
            uint16_t actual = 0;
            const uint32_t address = 2 * (gy * stride + gx);
            std::memcpy(&actual, store_machine.gmem().data() + address, sizeof(actual));
            require(actual ==
                static_cast<uint16_t>(0x400 + 7 * y + 11 * x),
                "swizzle64 store mapped the wrong element");
        }
    }
}

void test_tma_swizzle64_feeds_mma_orientations() {
    // Period 7: coprime to every origin/box offset in play (8, 24, 56), so a
    // misplaced origin cannot alias to the correct values the way a power-of-two
    // period would. All values are exact halfs with exact fp32 products/sums.
    const auto half_bits = [](uint32_t selector) {
        constexpr std::array<uint16_t, 7> bits{
            0x3800, 0x3c00, 0x4000, 0xbc00, 0x3e00, 0xb800, 0x4200};
        return bits[selector % 7];
    };
    const auto half_value = [](uint32_t selector) {
        constexpr std::array<float, 7> values{0.5f, 1.0f, 2.0f, -1.0f, 1.5f, -0.5f, 3.0f};
        return values[selector % 7];
    };
    for (bool pv : {false, true}) {
        constexpr uint32_t stride_a = 21;
        constexpr uint32_t stride_b_qk = 23;
        constexpr uint32_t stride_b_pv = 71;
        constexpr uint64_t base_a = 0;
        constexpr uint64_t base_b = 8192;
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.test_shmem, 7168);
        auto& gmem = image_bytes(launch.gmem_image, 20000);
        for (uint32_t m = 0; m < 64; ++m)
            for (uint32_t k = 0; k < 16; ++k)
                write_u16(gmem, base_a + 2 * (m * stride_a + k), half_bits(m + 2 * k));
        if (!pv) {
            for (uint32_t n = 0; n < 64; ++n)
                for (uint32_t k = 0; k < 16; ++k)
                    write_u16(gmem, base_b + 2 * (n * stride_b_qk + k), half_bits(3 * n + k + 1));
        } else {
            for (uint32_t k = 0; k < 16; ++k)
                for (uint32_t n = 0; n < 64; ++n)
                    write_u16(gmem, base_b + 2 * (k * stride_b_pv + n), half_bits(3 * n + k + 1));
        }
        // One swizzle64 descriptor per tensor (A at 0, B at 64), each tile 64
        // stored rows of K = 8 (16-byte rows, the default spacing the MMA
        // reads with); each chunk's command names its buffer and mbar: A
        // chunk c at 1024 + 2048c (mbar 512 + 16c), B chunk c at 2048 + 2048c
        // (mbar 520 + 16c). The MMA reads both operands as rows of K (A: m,
        // B: n). A's K runs along its rows; B's runs along the row under
        // QK^T (K's rows are keys) and across the rows under PV (V's rows
        // are keys: transposed).
        nanosm::TmaDescriptor a;
        a.element_bytes = 2; a.layout = nanosm::TileLayout::swizzle64;
        a.base = base_a; a.dimensions = {16, 64, 1, 1};
        a.strides = {stride_a, 0, 0}; a.tile_length = 8; a.tile_rows = 64;
        nanosm::TmaDescriptor b;
        b.element_bytes = 2; b.layout = nanosm::TileLayout::swizzle64; b.base = base_b;
        if (!pv) {
            b.dimensions = {16, 64, 1, 1}; b.strides = {stride_b_qk, 0, 0};
            b.tile_length = 8; b.tile_rows = 64;
        } else {
            b.transposed = true; b.dimensions = {64, 16, 1, 1}; b.strides = {stride_b_pv, 0, 0};
            b.tile_length = 64; b.tile_rows = 8;
        }
        write_tma_descriptor(shmem, 0, a);
        write_tma_descriptor(shmem, 64, b);
        nanosm::MbarState ready{false, true, 1, 0, 1024};
        for (const uint32_t mbar : {512u, 520u, 528u, 536u}) write_u64(shmem, mbar, nanosm::encode_mbar_state(ready));
        nanosm::MbarState commit{false, true, 1, 1, 0};
        write_u64(shmem, 544, nanosm::encode_mbar_state(commit));
        // Chunk 1's coordinates: A's K advances along
        // dimension 0 in both orientations, (8, 0); B's along dimension 0
        // under QK^T, (8, 0), but along dimension 1 under PV, (0, 8).
        const std::string chunk1_b = pv
            ? "umov ur8, 0\numov ur9, 8\ntma.load.2d [ur14], [64, ur8], [ur15]\n"
            : "tma.load.2d [ur14], [64, ur8], [ur15]\n";
        const auto kernel = compile("umov ur20, 0\n" + tma_operands_asm(1, 1024, 2, 512, 8, {0, 0}) +
            tma_operands_asm(3, 2048, 5, 520, 8, {}) + tma_operands_asm(6, 3072, 7, 528, 8, {}) +
            tma_operands_asm(14, 4096, 15, 536, 8, {}) + R"(
            tma.load.2d [ur1], [0, ur8], [ur2]
            tma.load.2d [ur3], [64, ur8], [ur5]
            umov ur8, 8
            tma.load.2d [ur6], [0, ur8], [ur7]
)" + chunk1_b + R"(
            umov ur4, 512
        wait0:
            mbar.try_wait p0, [ur4], 0
            @!p0 bra.uni wait0
            umov ur4, 520
        wait1:
            mbar.try_wait p0, [ur4], 0
            @!p0 bra.uni wait1
            umov ur4, 528
        wait2:
            mbar.try_wait p0, [ur4], 0
            @!p0 bra.uni wait2
            umov ur4, 536
        wait3:
            mbar.try_wait p0, [ur4], 0
            @!p0 bra.uni wait3
            umov ur10, 1024
            umov ur11, 2048
            mma.async.init.m64n64k8 [ur20], [ur10], [ur11]
            umov ur10, 3072
            umov ur11, 4096
            mma.async.m64n64k8 [ur20], [ur10], [ur11]
            umov ur4, 544
            mma.commit [ur4]
        wait_commit:
            mbar.try_wait p0, [ur4], 0
            @!p0 bra.uni wait_commit
            umov ur8, 0
            acc.ld.v8 r20, [ur8, 64]
            umov ur12, 3128            ; row 48, column 56
            acc.ld.v8 r28, [ur12, 64]
            halt
        )");
        nanosm::Machine machine(canonical_profile(), kernel, std::move(launch), {.timeline = true});
        machine.run(50000);
        const auto expected_c = [&](uint32_t m, uint32_t n) {
            float sum = 0;
            for (uint32_t k = 0; k < 16; ++k)
                sum += half_value(m + 2 * k) * half_value(3 * n + k + 1);
            return sum;
        };
        for (uint32_t lane = 0; lane < 16; ++lane) {
            for (uint32_t j = 0; j < 8; ++j) {
                require(machine.read_reg(0, lane, 20 + j) ==
                        nanosm::float_bits(expected_c(lane, j)),
                    pv ? "TMA-fed PV orientation produced the wrong MMA value"
                       : "TMA-fed QK^T orientation produced the wrong MMA value");
                require(machine.read_reg(0, lane, 28 + j) ==
                        nanosm::float_bits(expected_c(48 + lane, 56 + j)),
                    "TMA-fed MMA far-corner block is wrong");
            }
        }
        const auto& summary = machine.trace().summary();
        require(summary.tma_commands == 4 && summary.tma_command_completions == 4,
            "TMA-fed MMA proof did not complete all operand commands");
        // Hold A / stream B on a 4x4x8 core: the step that reads an A block
        // reads A and B (two bank rounds), the other fifteen read one B block
        // (one round), and each of the 16 A blocks is read once per command.
        require(summary.mma_commands == 2 && summary.mma_input_steps == 2 * 256 &&
                summary.mma_a_block_reads == 2 * 16 &&
                summary.mma_shmem_rounds == summary.mma_input_steps + summary.mma_a_block_reads,
            "TMA-authored operands did not keep one bank round per B block and per A block");
    }
}

// The census, for whole rows: for
// stored-row counts R (4 .. 128) and fp16 K extents 8, 16, 32, 64 (rows of
// 16, 32, 64, 128 bytes), a TMA load of a swizzle64 R x K tile writes exactly
// what an MMA with m = R (as A) or n = R (as B), k = K and row spacing 2K
// reads: every element where the MMA's own operand addressing
// (nanosm::mma_operand_address) looks for it, and no byte outside the tile's
// R x K x 2 bytes. A's global is m rows of K; B's is K rows of n
// (transposed). The buffer is a multiple of every spacing here (128) but not
// of 16 rows of any of them, so the tile's row 0 is not row 0 of the shuffle.
void test_tma_swizzle64_census_against_mma() {
    const auto& profile = canonical_profile();
    constexpr uint32_t base = 1152;
    constexpr uint32_t mbar_address = 64;
    constexpr uint32_t shmem_bytes = base + 128 * 64 * 2 + 1024;
    constexpr uint8_t poison = 0xa5;
    const auto tag = [](uint32_t outer, uint32_t k) { return static_cast<uint16_t>(0x8000u | outer << 6 | k); };
    const auto kernel = compile(tma_operands_asm(4, base, 3, mbar_address, 8, {0, 0}) + R"(
        tma.load.2d [ur4], [0, ur8], [ur3]
    wait:
        mbar.try_wait p0, [ur3], 0
        @!p0 bra.uni wait
        halt
    )");
    uint32_t cases = 0;
    for (const bool a_role : {true, false}) {
        for (uint32_t rows = 4; rows <= 128; rows *= 2) {
            for (const uint32_t k_extent : {8u, 16u, 32u, 64u}) {
                const auto what = std::string(a_role ? "A" : "B") + ", " + std::to_string(rows) + " rows, K " +
                    std::to_string(k_extent);
                // Row-major global, the tile the whole tensor: A is rows x K
                // (K along the row), B is K x n (K across the rows).
                const uint32_t length = a_role ? k_extent : rows;
                const uint32_t height = a_role ? rows : k_extent;
                nanosm::TmaDescriptor descriptor;
                descriptor.element_bytes = 2;
                descriptor.layout = nanosm::TileLayout::swizzle64;
                descriptor.transposed = !a_role;
                descriptor.dimensions = {length, height, 1, 1};
                descriptor.strides = {length, 0, 0};
                descriptor.tile_length = static_cast<uint16_t>(length);
                descriptor.tile_rows = static_cast<uint16_t>(height);
                nanosm::Launch launch;
                launch.active_warps = 1;
                auto& shmem = image_bytes(launch.test_shmem, shmem_bytes);
                std::fill(shmem.begin(), shmem.end(), poison);
                std::fill(shmem.begin(), shmem.begin() + 128, 0);
                write_tma_descriptor(shmem, 0, descriptor);
                write_tx_mbar(shmem, mbar_address, rows * k_extent * 2);
                auto& gmem = image_bytes(launch.gmem_image, rows * k_extent * 2);
                for (uint32_t outer = 0; outer < rows; ++outer) {
                    for (uint32_t k = 0; k < k_extent; ++k) {
                        write_u16(gmem, 2 * (a_role ? outer * k_extent + k : k * rows + outer), tag(outer, k));
                    }
                }
                nanosm::Machine machine(profile, kernel, std::move(launch));
                machine.run(100000);
                require(machine.read_mbar_state(mbar_address).phase, what + ": the load did not complete");
                const nanosm::MmaShape shape = a_role ? nanosm::MmaShape{rows, 16, k_extent}
                                                      : nanosm::MmaShape{16, rows, k_extent};
                const auto& bytes = machine.shmem();
                std::set<uint32_t> addresses;
                for (uint32_t outer = 0; outer < rows; ++outer) {
                    for (uint32_t k = 0; k < k_extent; ++k) {
                        const auto address =
                            nanosm::mma_operand_address(profile, shape, base, 2 * k_extent, a_role, outer, k);
                        require(address >= base && address + 2 <= base + rows * k_extent * 2,
                            what + ": the MMA reads outside the tile's compact object");
                        require(addresses.insert(address).second, what + ": the MMA reads one address twice");
                        uint16_t value = 0;
                        std::memcpy(&value, bytes.data() + address, sizeof(value));
                        require(value == tag(outer, k), what + ": the MMA's element (" + std::to_string(outer) + ", " +
                            std::to_string(k) + ") is not where the TMA put it");
                    }
                }
                for (uint32_t address = 128; address < shmem_bytes; ++address) {
                    if (address >= base && address < base + rows * k_extent * 2) continue;
                    require(bytes[address] == poison, what + ": the TMA wrote byte " + std::to_string(address) +
                        " outside the object");
                }
                ++cases;
            }
        }
    }
    require(cases == 2 * 6 * 4, "the census did not cover every stored-row count and K extent");
}

// The TMA does not know who reads a tile: a TMA-only kernel with explicit
// swizzle64 descriptors (one tile not transposed, one transposed) leaves
// every shared byte identical when only the profile's mma.layout changes
// (swizzle64 -> dense).
void test_tma_swizzle64_tile_ignores_the_mma_layout() {
    constexpr uint32_t mbar_address = 128;
    constexpr uint32_t stride = 83;
    auto dense = canonical_profile();
    dense.mma_layout = nanosm::TileLayout::dense;
    nanosm::validate_device_profile(dense);
    const auto run = [&](const nanosm::DeviceProfile& profile) {
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.test_shmem, 8192);
        auto& gmem = image_bytes(launch.gmem_image, 2 * stride * 70);
        for (uint32_t y = 0; y < 70; ++y)
            for (uint32_t x = 0; x < stride; ++x)
                write_u16(gmem, 2 * (y * stride + x), static_cast<uint16_t>(0x100 + 3 * y + 5 * x));
        for (const bool transposed : {false, true}) {
            nanosm::TmaDescriptor descriptor;
            descriptor.element_bytes = 2;
            descriptor.layout = nanosm::TileLayout::swizzle64;
            descriptor.transposed = transposed;
            descriptor.dimensions = {stride, 70, 1, 1};
            descriptor.strides = {stride, 0, 0};
            descriptor.tile_length = transposed ? 64 : 16;
            descriptor.tile_rows = transposed ? 16 : 64;
            write_tma_descriptor(shmem, transposed ? 64 : 0, descriptor);
        }
        write_tx_mbar(shmem, mbar_address, 2 * 2048);
        nanosm::Machine machine(profile, compile(tma_operands_asm(4, 1024, 3, mbar_address, 8, {3, 2}) + R"(
            tma.load.2d [ur4], [0, ur8], [ur3]
            umov ur5, 4096
            tma.load.2d [ur5], [64, ur8], [ur3]
        wait:
            mbar.try_wait p0, [ur3], 0
            @!p0 bra.uni wait
            halt
        )", profile), std::move(launch));
        machine.run(50000);
        require(machine.read_mbar_state(mbar_address).phase, "the loads did not complete");
        return machine.shmem();
    };
    const auto with_swizzle64 = run(canonical_profile());
    const auto with_dense = run(dense);
    require(std::any_of(with_swizzle64.begin() + 1024, with_swizzle64.begin() + 6144,
                [](uint8_t b) { return b != 0; }),
        "the tiles were not loaded");
    require(with_swizzle64 == with_dense, "a swizzle64 TMA tile changed with the profile's mma.layout");
}

void test_tma_swizzle64_validation() {
    // The stored row (the tile length, or the rows when transposed, times
    // the element bytes): a power of two bytes, 16 or more. Any number of
    // stored rows (the other extent), 1 included. A dense tile cannot be
    // transposed.
    nanosm::TmaDescriptor descriptor;
    descriptor.element_bytes = 2;
    descriptor.layout = nanosm::TileLayout::swizzle64;
    descriptor.dimensions = {128, 128, 1, 1};
    descriptor.strides = {128, 0, 0};
    const auto rejection = [&](bool transposed, uint32_t tile_length, uint32_t tile_rows) {
        descriptor.transposed = transposed;
        descriptor.tile_length = static_cast<uint16_t>(tile_length);
        descriptor.tile_rows = static_cast<uint16_t>(tile_rows);
        return encode_error(descriptor);
    };
    constexpr bool along = false;
    constexpr bool across = true;
    for (uint32_t elements = 1; elements <= 256; ++elements) {
        const uint32_t bytes = 2 * elements;
        const bool allowed = bytes == 16 || bytes == 32 || bytes == 64 || bytes == 128 || bytes == 256 ||
            bytes == 512;
        require(rejection(along, elements, 64).empty() == allowed &&
                rejection(across, 64, elements).empty() == allowed,
            "a swizzle64 stored row of " + std::to_string(bytes) + " bytes was " +
                (allowed ? "rejected" : "accepted"));
        if (!allowed) {
            require(rejection(along, elements, 64).find("stored row must be") != std::string::npos,
                "a swizzle64 stored row of " + std::to_string(bytes) + " bytes rejected with the wrong diagnostic");
        }
    }
    for (const uint32_t rows : {1u, 3u, 5u, 17u, 200u}) {
        require(rejection(along, 8, rows).empty() && rejection(across, rows, 16).empty(),
            "a swizzle64 tile of " + std::to_string(rows) + " stored rows was rejected");
    }
    auto dense = descriptor;
    dense.layout = nanosm::TileLayout::dense;
    dense.transposed = true;
    dense.tile_length = 16;
    dense.tile_rows = 64;
    require(encode_error(dense).find("dense tiles cannot be transposed") != std::string::npos,
        "a transposed dense tile was accepted");

    // The span is the packed tile, 64 x 64 x 2 = 8192 bytes: at 61440 it
    // ends past the 64 KiB shmem.
    nanosm::TmaDescriptor big;
    big.element_bytes = 2;
    big.layout = nanosm::TileLayout::swizzle64;
    big.dimensions = {128, 128, 1, 1};
    big.strides = {128, 0, 0};
    big.tile_length = 64;
    big.tile_rows = 64;
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.test_shmem, 33032);
    image_bytes(launch.gmem_image, 65536);
    write_tma_descriptor(shmem, 0, big);
    write_u64(shmem, 33024, nanosm::encode_mbar_state({false, true, 1, 0, 8192}));
    nanosm::Machine machine(canonical_profile(),
        compile(tma_operands_asm(4, 61440, 3, 33024, 8, {0, 0}) + "tma.load.2d [ur4], [0, ur8], [ur3]\nhalt\n"),
        std::move(launch));
    try {
        machine.run(20000);
        throw std::runtime_error("oversized swizzle64 span was accepted");
    } catch (const nanosm::SimError& error) {
        require(std::string(error.what()).find("does not fit in shmem") != std::string::npos,
            "oversized swizzle64 span rejected with the wrong diagnostic");
    }
}

// `commands` loads of 8192 / commands bytes each, one descriptor, the tile's
// coordinates in ur8, ur9 (set by the caller, moved on by `step` between
// commands): command i's buffer (ur4) starts at tile + (8192 / commands) x
// i. Every command completes into the mbar in ur3, which the wait reads.
std::string tile_piece_loads(uint32_t commands, uint32_t tile, uint32_t mbar, const std::string& step) {
    std::string source = "umov ur3, " + std::to_string(mbar) + "\n";
    for (uint32_t i = 0; i < commands; ++i) {
        if (i != 0) source += step;
        source += "umov ur4, " + std::to_string(tile + 8192 / commands * i) +
            "\ntma.load.2d [ur4], [0, ur8], [ur3]\n";
    }
    return source + "wait:\nmbar.try_wait p0, [ur3], 0\n@!p0 bra.uni wait\nhalt\n";
}

// The shuffle's row number is the address's, so pieces of a tile land
// anywhere: 16 loads of 4 stored rows, or 64 loads of ONE stored row, each
// landed right after the last, build exactly the 64-row tile one load makes:
// fp16, 128-byte stored rows, not transposed (the pieces step along the
// global rows) and transposed (they step along dimension 0, whose positions
// are the stored rows). The tile is at 0x4080, so its row 0 is row 1 of the
// shuffle (the one-row gather case): its
// first logical byte is stored at 0x4090 and its logical byte 16 at 0x4080,
// which a reader or writer still counting rows from the tile would miss.
void test_tma_swizzle64_pieces_land_anywhere() {
    constexpr uint32_t mbar_address = 576;
    constexpr uint32_t tile_address = 0x4080;
    constexpr uint32_t stride = 83;
    constexpr uint32_t dim_x = 80;
    constexpr uint32_t dim_y = 70;
    for (const bool transposed : {false, true}) {
        const auto run_span = [&](uint32_t commands) {
            nanosm::Launch launch;
            launch.active_warps = 1;
            auto& shmem = image_bytes(launch.test_shmem, tile_address + 8192);
            auto& gmem = image_bytes(launch.gmem_image, 2 * stride * dim_y + 64);
            for (uint32_t y = 0; y < dim_y; ++y)
                for (uint32_t x = 0; x < dim_x; ++x)
                    write_u16(gmem, 2 * (y * stride + x),
                        static_cast<uint16_t>(0x100 + 3 * y + 5 * x));
            const uint32_t stored_rows = 64 / commands;
            nanosm::TmaDescriptor descriptor;
            descriptor.element_bytes = 2;
            descriptor.layout = nanosm::TileLayout::swizzle64;
            descriptor.transposed = transposed;
            descriptor.dimensions = {dim_x, dim_y, 1, 1};
            descriptor.strides = {stride, 0, 0};
            descriptor.tile_length = static_cast<uint16_t>(transposed ? stored_rows : 64);
            descriptor.tile_rows = static_cast<uint16_t>(transposed ? 64 : stored_rows);
            write_tma_descriptor(shmem, 0, descriptor);
            write_u64(shmem, mbar_address,
                nanosm::encode_mbar_state({false, true, 1, 0, 8192}));
            const std::string step = "uaddi ur" + std::string(transposed ? "8" : "9") + ", ur" +
                (transposed ? "8" : "9") + ", " + std::to_string(stored_rows) + "\n";
            const std::string source = "umov ur8, 3\numov ur9, 2\n" +
                tile_piece_loads(commands, tile_address, mbar_address, step);
            nanosm::Machine machine(canonical_profile(), compile(source), std::move(launch));
            machine.run(200000);
            std::vector<uint32_t> words;
            words.reserve(2048);
            for (uint32_t offset = 0; offset < 8192; offset += 4)
                words.push_back(machine.read_shmem_u32(tile_address + offset));
            return words;
        };
        const auto whole = run_span(1);
        require(whole == run_span(16), "one 64-row swizzle64 load disagreed with sixteen 4-row loads");
        require(whole == run_span(64), "one 64-row swizzle64 load disagreed with 64 one-row loads");
        // Stored row 0's logical bytes 0 and 16, at their physical addresses
        // written out by hand: element (0, 0), and (8, 0) or transposed (0, 8).
        const auto halfword = [&](uint32_t address) {
            return static_cast<uint16_t>(whole[(address - tile_address) / 4] >> (8 * (address & 3u)));
        };
        const auto value = [](uint32_t x, uint32_t y) { return static_cast<uint16_t>(0x100 + 3 * (2 + y) + 5 * (3 + x)); };
        require(halfword(0x4090) == value(0, 0) && halfword(0x4080) == (transposed ? value(0, 8) : value(8, 0)),
            "stored row 0's logical bytes 0 and 16 are not at 0x4090 and 0x4080");
        std::set<uint32_t> addresses;
        for (uint32_t y = 0; y < 64; ++y) {
            for (uint32_t x = 0; x < 64; ++x) {
                const uint32_t position = transposed ? y : x;
                const uint32_t row = transposed ? x : y;
                const auto address = reference_swizzle64(tile_address, 128, row, 2 * position);
                require(addresses.insert(address).second, "swizzle64 map is not bijective");
                const uint32_t gx = 3 + x;
                const uint32_t gy = 2 + y;
                const auto word = whole[(address - tile_address) / 4];
                const auto actual = static_cast<uint16_t>(word >> (8 * (address & 3u)));
                require(actual == static_cast<uint16_t>(0x100 + 3 * gy + 5 * gx),
                    "swizzle64 load mapped the wrong element");
            }
        }
        require(addresses.size() == 4096, "swizzle64 map did not cover the tile");
    }
}

void test_tma_swizzle64_sector_economics() {
    constexpr uint32_t mbar_address = 576;
    constexpr uint32_t tile_address = 1024;
    // Natural head-innermost K: 64 keys x 64 fp16, dense 128-byte rows.
    const auto run_summary = [&](uint32_t commands) {
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.test_shmem, tile_address + 8192);
        auto& gmem = image_bytes(launch.gmem_image, 8192);
        for (uint32_t i = 0; i < 4096; ++i)
            write_u16(gmem, 2 * i, static_cast<uint16_t>((i * 2654435761u) >> 16));
        const uint32_t k_extent = 64 / commands;
        nanosm::TmaDescriptor descriptor;
        descriptor.element_bytes = 2;
        // Not transposed: the QK^T orientation of K's natural layout. The
        // narrow commands cut K into 16-byte rows, each its own tile.
        descriptor.layout = nanosm::TileLayout::swizzle64;
        descriptor.dimensions = {64, 64, 1, 1};
        descriptor.strides = {64, 0, 0};
        descriptor.tile_length = static_cast<uint16_t>(k_extent);
        descriptor.tile_rows = 64;
        write_tma_descriptor(shmem, 0, descriptor);
        write_u64(shmem, mbar_address, nanosm::encode_mbar_state({false, true, 1, 0, 8192}));
        const std::string source = "umov ur8, 0\numov ur9, 0\n" +
            tile_piece_loads(commands, tile_address, mbar_address, "uaddi ur8, ur8, 8\n");
        nanosm::Machine machine(canonical_profile(), compile(source), std::move(launch));
        machine.run(200000);
        return machine.trace().summary();
    };
    const auto wide = run_summary(1);
    require(wide.tma_command_completions == 1 && wide.tma_useful_bytes == 8192 &&
            wide.tma_sector_bytes == 8192,
        "wide natural-layout load was not 1.00x sector traffic");
    const auto narrow = run_summary(8);
    require(narrow.tma_command_completions == 8 && narrow.tma_useful_bytes == 8192 &&
            narrow.tma_sector_bytes == 16384,
        "narrow chunked load did not show the 2x half-sector refetch");
}

// The layout function (src/layout.hpp) against the rule written out
// (reference_swizzle64) for spacings 16 ..
// 512, at base 0 and at a legal base 17 x max(spacing, 64) that is not a
// multiple of 16 rows: the same address for every byte, every row keeps
// exactly its own bytes [base + row x spacing, base + (row + 1) x spacing),
// and the shuffle is a permutation. Keyed by address: a piece of the tile
// that starts at row j (base + j x spacing, any j) puts every byte where the
// tile does. The tensor core's operand pointer is a logical position:
// logical byte 16 of row 1 at spacing 128 is stored at 128 (its piece slot
// XORed with 1), not 16 past where byte 0 is. The spacings are the powers
// of two from 16.
void test_swizzle64_layout_matches_the_rule() {
    for (const uint32_t spacing : {16u, 32u, 64u, 128u, 256u, 512u}) {
        for (const uint32_t base : {0u, 17 * std::max(spacing, 64u)}) {
            std::set<uint32_t> addresses;
            for (uint32_t row = 0; row < 64; ++row) {
                for (uint32_t byte = 0; byte < spacing; ++byte) {
                    const auto address = nanosm::swizzle64_address(base, spacing, row, byte);
                    const auto what = "spacing " + std::to_string(spacing) + " base " + std::to_string(base) +
                        " row " + std::to_string(row) + " byte " + std::to_string(byte);
                    require(address == reference_swizzle64(base, spacing, row, byte), what + ": not the rule");
                    require(address >= base + row * spacing && address < base + (row + 1) * spacing,
                        what + ": left its row");
                    require(addresses.insert(address).second, what + ": two bytes in one place");
                    for (const uint32_t first : {1u, 5u, 37u}) {
                        if (row < first) continue;
                        require(nanosm::swizzle64_address(base + first * spacing, spacing, row - first, byte) ==
                                address,
                            what + ": a piece starting at row " + std::to_string(first) + " puts it elsewhere");
                    }
                }
            }
        }
    }
    require(nanosm::swizzle64_address(0, 128, 1, 16) == 128 && nanosm::swizzle64_address(0, 128, 1, 0) == 144,
        "logical byte 16 of row 1 is not where the rule puts it");
    for (uint32_t spacing = 0; spacing <= 4096; ++spacing) {
        const bool power = spacing >= 16 && (spacing & (spacing - 1)) == 0;
        require(nanosm::valid_swizzle64_spacing(spacing) == power,
            "spacing " + std::to_string(spacing) + (power ? " was refused" : " was accepted"));
    }
}

void test_tma_fp32_swizzle64_validation() {
    // The stored row runs along the tile's rows (a) or, transposed, across
    // them (b): a power of two bytes, 16 or more.
    const auto rejection = [](uint8_t element_bytes, bool transposed, uint32_t tile_length, uint32_t tile_rows) {
        nanosm::TmaDescriptor descriptor;
        descriptor.element_bytes = element_bytes;
        descriptor.layout = nanosm::TileLayout::swizzle64;
        descriptor.transposed = transposed;
        descriptor.dimensions = {128, 128, 1, 1};
        descriptor.strides = {128, 0, 0};
        descriptor.tile_length = static_cast<uint16_t>(tile_length);
        descriptor.tile_rows = static_cast<uint16_t>(tile_rows);
        return encode_error(descriptor);
    };
    constexpr bool a = false;
    constexpr bool b = true;
    // fp32: 4 (16 B), 8 (32 B), 16, 32, 64 ... elements per stored row.
    require(rejection(4, a, 64, 64).empty(), "fp32 swizzle64 64x64 tile was rejected");
    require(rejection(4, b, 64, 64).empty(), "fp32 transposed swizzle64 64x64 tile was rejected");
    require(rejection(4, a, 4, 64).empty() && rejection(4, a, 8, 64).empty() && rejection(4, a, 128, 64).empty(),
        "fp32 swizzle64 rows of 16, 32 and 512 bytes were rejected");
    require(rejection(4, a, 64, 48).empty() && rejection(4, b, 32, 8).empty() && rejection(4, a, 64, 1).empty(),
        "fp32 swizzle64 tiles of other row counts were rejected");
    for (const uint32_t k : {2u, 6u, 12u, 20u, 48u}) {
        require(rejection(4, a, k, 64).find("stored row must be") != std::string::npos &&
                rejection(4, b, 64, k).find("stored row must be") != std::string::npos,
            "fp32 swizzle64 accepted a stored row of " + std::to_string(4 * k) + " bytes");
    }
    require(!rejection(8, a, 64, 64).empty(), "8-byte swizzle64 elements were accepted");
    require(rejection(1, a, 64, 64).find("2- or 4-byte elements") != std::string::npos,
        "1-byte swizzle64 elements were accepted");
    // fp16: 12, 4 and 40 elements are 24, 8 and 80 bytes.
    for (const uint32_t k : {12u, 4u, 40u}) {
        require(rejection(2, a, k, 64).find("stored row must be") != std::string::npos,
            "fp16 swizzle64 accepted a stored row of " + std::to_string(2 * k) + " bytes");
    }
}

// Independent fp32 swizzle64 address: a 64 x 64 fp32 tile's row is 256
// bytes.
uint32_t reference_fp32_swizzle64(uint32_t base, uint32_t row, uint32_t col) {
    return reference_swizzle64(base, 256, row, 4 * col);
}

float fp32_tile_value(uint32_t row, uint32_t col) {
    // Periods 31 in both axes: coprime to 4, 8, 16, 64. Small integers, exact.
    return static_cast<float>(static_cast<int>((row * 17 + col * 11) % 31) - 15);
}

void test_tma_fp32_swizzle64_store_and_load() {
    constexpr uint32_t mbar_address = 128;
    constexpr uint32_t tile_address = 1024;
    constexpr uint32_t tile_bytes = 16384;
    constexpr uint64_t gmem_base = 512;
    constexpr uint32_t gmem_size = 512 + tile_bytes + 512;

    std::set<uint32_t> addresses;
    for (uint32_t row = 0; row < 64; ++row) {
        for (uint32_t col = 0; col < 64; ++col) {
            const auto address = reference_fp32_swizzle64(tile_address, row, col);
            require(address % 4 == 0 && address >= tile_address + 256 * row &&
                    address < tile_address + 256 * (row + 1),
                "fp32 swizzle64 address left its row");
            require(addresses.insert(address).second, "fp32 swizzle64 map is not bijective");
        }
    }
    require(addresses.size() == 4096, "fp32 swizzle64 map did not cover the tile");

    // A 64 x 64 fp32 tile, the whole tensor: swizzle64 (not transposed) or dense.
    const auto make_descriptor = [&](nanosm::TileLayout layout) {
        nanosm::TmaDescriptor descriptor;
        descriptor.element_bytes = 4;
        descriptor.layout = layout;
        descriptor.base = gmem_base;
        descriptor.dimensions = {64, 64, 1, 1};
        descriptor.strides = {64, 0, 0};
        descriptor.tile_length = 64;
        descriptor.tile_rows = 64;
        return descriptor;
    };
    const auto program = [&](bool store) {
        return tma_operands_asm(4, tile_address, 3, mbar_address, 8, {0, 0}) +
            (store ? "tma.store.2d [0, ur8], [ur4], [ur3]\n" : "tma.load.2d [ur4], [0, ur8], [ur3]\n") +
            "wait:\nmbar.try_wait p0, [ur3], 0\n@!p0 bra.uni wait\nhalt\n";
    };
    const nanosm::MbarState mbar{false, true, 1, 0, tile_bytes};

    // STORE: shmem tile authored at the independent address -> row-major gmem.
    const auto run_store = [&](nanosm::TileLayout layout) {
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.test_shmem, tile_address + tile_bytes);
        auto& gmem = image_bytes(launch.gmem_image, gmem_size);
        std::fill(gmem.begin(), gmem.end(), 0x5a);
        for (uint32_t row = 0; row < 64; ++row)
            for (uint32_t col = 0; col < 64; ++col)
                write_u32(shmem, layout == nanosm::TileLayout::swizzle64
                        ? reference_fp32_swizzle64(tile_address, row, col)
                        : tile_address + 4 * (64 * row + col),
                    nanosm::float_bits(fp32_tile_value(row, col)));
        write_tma_descriptor(shmem, 0, make_descriptor(layout));
        write_u64(shmem, mbar_address, nanosm::encode_mbar_state(mbar));
        nanosm::Machine machine(canonical_profile(), compile(program(true)), std::move(launch));
        machine.run(200000);
        for (uint32_t row = 0; row < 64; ++row) {
            for (uint32_t col = 0; col < 64; ++col) {
                uint32_t actual = 0;
                std::memcpy(&actual, machine.gmem().data() + gmem_base + 4 * (64 * row + col),
                    sizeof(actual));
                require(actual == nanosm::float_bits(fp32_tile_value(row, col)),
                    "fp32 swizzle64 TMA store mapped the wrong element at row " +
                        std::to_string(row) + " col " + std::to_string(col));
            }
        }
        for (uint32_t offset = 0; offset < gmem_size; ++offset) {
            if (offset >= gmem_base && offset < gmem_base + tile_bytes) continue;
            require(machine.gmem()[offset] == 0x5a, "fp32 TMA store wrote outside its tile");
        }
        return machine.trace().summary();
    };
    const auto swizzled_store = run_store(nanosm::TileLayout::swizzle64);
    // A 32-byte sector is eight consecutive fp32 of one row: two 16-byte
    // pieces of one 64-byte line, which the shuffle keeps in two different
    // slots (eight different banks): one port-A round per sector, as linear
    // placement (eight adjacent words). The sliced layout took two.
    require(swizzled_store.tma_command_completions == 1 &&
            swizzled_store.tma_useful_bytes == tile_bytes &&
            swizzled_store.tma_sector_bytes == tile_bytes,
        "fp32 swizzle64 TMA store was not 1.00x sector traffic");
    require(swizzled_store.tma_port_requests == 512 && swizzled_store.tma_port_rounds == 512,
        "fp32 swizzle64 TMA store did not read one port-A round per sector");
    const auto linear_store = run_store(nanosm::TileLayout::dense);
    require(linear_store.tma_port_requests == 512 && linear_store.tma_port_rounds == 512,
        "linear fp32 TMA store did not read one port-A round per sector");

    // LOAD: row-major gmem -> shmem; every element checked at the independent address.
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.test_shmem, tile_address + tile_bytes);
    auto& gmem = image_bytes(launch.gmem_image, gmem_size);
    std::fill(shmem.begin() + tile_address, shmem.end(), 0x7f);
    for (uint32_t row = 0; row < 64; ++row)
        for (uint32_t col = 0; col < 64; ++col)
            write_u32(gmem, static_cast<uint32_t>(gmem_base) + 4 * (64 * row + col),
                nanosm::float_bits(fp32_tile_value(row, col)));
    write_tma_descriptor(shmem, 0, make_descriptor(nanosm::TileLayout::swizzle64));
    write_u64(shmem, mbar_address, nanosm::encode_mbar_state(mbar));
    nanosm::Machine machine(canonical_profile(), compile(program(false)), std::move(launch));
    machine.run(200000);
    for (uint32_t row = 0; row < 64; ++row) {
        for (uint32_t col = 0; col < 64; ++col) {
            require(machine.read_shmem_u32(reference_fp32_swizzle64(tile_address, row, col)) ==
                    nanosm::float_bits(fp32_tile_value(row, col)),
                "fp32 swizzle64 TMA load mapped the wrong element at row " +
                    std::to_string(row) + " col " + std::to_string(col));
        }
    }
    const auto& load_summary = machine.trace().summary();
    require(load_summary.tma_port_requests == 512 && load_summary.tma_port_rounds == 512,
        "fp32 swizzle64 TMA load did not write one port-A round per sector");
}

// ---- rows stay whole: smoke tests ----

uint16_t whole_rows_fp16(uint32_t value) {
    static constexpr std::array<uint16_t, 5> bits{0x0000, 0x3c00, 0x4000, 0x4200, 0x4400}; // 0..4
    return bits.at(value);
}

// A(m, k) and B(k, n): small integers (exact sums), different in every
// 16-byte piece of a row (k and k + 8 differ by 24 = 4 mod 5 and 16 = 1 mod
// 3), so a command that reads the wrong piece is caught.
uint32_t whole_rows_a(uint32_t m, uint32_t k) { return (m + 3 * k) % 5; }

uint32_t whole_rows_b(uint32_t k, uint32_t n) { return (2 * k + n) % 3; }

// The tensor core reads TMA tiles of 128-byte rows. A (64 x 64 fp16) and B
// (64 rows of 64 fp16: B's n rows of k, as a K cache) land whole, one load
// each. Eight m64n64k8 commands step both operand pointers by one 16-byte
// piece with row spacing 128 (FA's QK^T); eight m32n64k8 commands read A's
// bottom half, the pointer at row 32 (+ 32 x 128), and eight more read rows
// 21..52 (+ 21 x 128: an operand may start at any row, the shuffle keyed by
// address). A at 0x4080 starts at row 1 of the shuffle and B at 0x180 at
// row 3 (multiples of 128, not of 2048): a reader counting rows from the
// tile would read the wrong pieces. Bank rounds as with 16-byte rows: 17
// per A block and B sweep, and one TMA round per sector.
void test_swizzle64_mma_reads_a_tma_tile() {
    constexpr uint32_t a_tile = 0x4080;
    constexpr uint32_t b_tile = 0x180;
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.test_shmem, a_tile + 8192);
    auto& gmem = image_bytes(launch.gmem_image, 16384);
    for (uint32_t row = 0; row < 64; ++row) {
        for (uint32_t k = 0; k < 64; ++k) {
            write_u16(gmem, 2 * (64 * row + k), whole_rows_fp16(whole_rows_a(row, k)));
            write_u16(gmem, 8192 + 2 * (64 * row + k), whole_rows_fp16(whole_rows_b(k, row)));
        }
    }
    for (const uint32_t descriptor_address : {0u, 64u}) {
        nanosm::TmaDescriptor descriptor;
        descriptor.element_bytes = 2;
        descriptor.layout = nanosm::TileLayout::swizzle64;
        descriptor.base = descriptor_address == 0 ? 0 : 8192;
        descriptor.dimensions = {64, 64, 1, 1};
        descriptor.strides = {64, 0, 0};
        descriptor.tile_length = 64;
        descriptor.tile_rows = 64;
        write_tma_descriptor(shmem, descriptor_address, descriptor);
    }
    write_u64(shmem, 128, nanosm::encode_mbar_state({false, true, 1, 0, 16384}));
    write_u64(shmem, 136, nanosm::encode_mbar_state({false, true, 1, 1, 0}));
    // Three results in tmem: C (64x64) at 0, H and G (32x64) at 4096 and 6144.
    std::string source = "umov ur20, 0\numov ur21, 4096\numov ur22, 6144\n" +
        tma_operands_asm(1, a_tile, 2, 128, 8, {0, 0}) + "umov ur3, " + std::to_string(b_tile) + "\n" +
        "tma.load.2d [ur1], [0, ur8], [ur2]\ntma.load.2d [ur3], [64, ur8], [ur2]\n"
        "loaded:\nmbar.try_wait p0, [ur2], 0\n@!p0 bra.uni loaded\n";
    for (const auto& [accumulator, shape, a_start] :
         {std::tuple{"[ur20]", "m64n64k8", a_tile}, std::tuple{"[ur21]", "m32n64k8", a_tile + 32 * 128},
          std::tuple{"[ur22]", "m32n64k8", a_tile + 21 * 128}}) {
        source += "umov ur10, " + std::to_string(a_start) + "\numov ur11, " + std::to_string(b_tile) + "\n";
        for (uint32_t piece = 0; piece < 8; ++piece) {
            if (piece != 0) source += "uaddi ur10, ur10, 16\nuaddi ur11, ur11, 16\n";
            source += std::string("mma.async.") + (piece == 0 ? "init." : "") + shape + " " + accumulator +
                ", [ur10, 128], [ur11, 128]\n";
        }
    }
    source += "umov ur4, 136\nmma.commit [ur4]\ncommitted:\nmbar.try_wait p0, [ur4], 0\n@!p0 bra.uni committed\n"
        // C row 0 / row 48 column 56; H and G row 16 column 8 (4096 + 1024 + 8, 6144 + 1024 + 8).
        "acc.ld.v8 r20, [ur20, 64]\numov ur12, 3128\nacc.ld.v8 r28, [ur12, 64]\n"
        "umov ur14, 5128\numov ur15, 7176\nacc.ld.v8 r36, [ur14, 64]\nacc.ld.v8 r44, [ur15, 64]\nhalt\n";
    nanosm::Machine machine(canonical_profile(), compile(source), std::move(launch), {.timeline = true});
    machine.run(100000);
    const auto expected = [](uint32_t m, uint32_t n) {
        float sum = 0;
        for (uint32_t k = 0; k < 64; ++k) sum += static_cast<float>(whole_rows_a(m, k) * whole_rows_b(k, n));
        return nanosm::float_bits(sum);
    };
    for (uint32_t lane = 0; lane < 16; ++lane) {
        for (uint32_t j = 0; j < 8; ++j) {
            require(machine.read_reg(0, lane, 20 + j) == expected(lane, j),
                "C rows 0..15 are wrong: an operand read the wrong piece or row");
            require(machine.read_reg(0, lane, 28 + j) == expected(48 + lane, 56 + j), "C's far corner is wrong");
            require(machine.read_reg(0, lane, 36 + j) == expected(48 + lane, 8 + j),
                "the m32 command at A's row 32 did not read rows 32..63");
            require(machine.read_reg(0, lane, 44 + j) == expected(37 + lane, 8 + j),
                "the m32 command at A's row 21 did not read rows 21..52");
        }
    }
    const auto& summary = machine.trace().summary();
    require(summary.mma_commands == 24 && summary.mma_a_block_reads == 8 * 16 + 2 * 8 * 8 &&
            summary.mma_shmem_rounds == 8 * 272 + 2 * 8 * 136,
        "128-byte rows did not keep one bank round per B block and per A block: " +
            std::to_string(summary.mma_shmem_rounds) + " rounds");
    require(summary.tma_port_requests == 512 && summary.tma_port_rounds == 512,
        "the TMA did not place one sector per port-A round");
}

// The base rule (src/layout.hpp): a swizzle64 tile's buffer is a multiple of
// its spacing and of 64, and a tensor-core operand's row 0 lies in one row
// whose line is. The machine shuffles by address and checks neither; the
// sanitizer PANICs: a one-row tile of 128-byte rows at 1088 (a multiple of
// 64, not of 128), and an m64n64k8 A operand of 16-byte rows at 1040 (row
// 1 of a tile at 1024: the 64 rule).
void test_swizzle64_base_rule_panics() {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    {
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.test_shmem, 4096);
        image_bytes(launch.gmem_image, 128);
        nanosm::TmaDescriptor descriptor;
        descriptor.element_bytes = 2;
        descriptor.layout = nanosm::TileLayout::swizzle64;
        descriptor.dimensions = {64, 1, 1, 1};
        descriptor.strides = {64, 0, 0};
        descriptor.tile_length = 64;
        descriptor.tile_rows = 1;
        write_tma_descriptor(shmem, 0, descriptor);
        write_tx_mbar(shmem, 128, 128);
        nanosm::Machine machine(canonical_profile(), compile(tma_operands_asm(4, 1088, 3, 128, 8, {0, 0}) +
            "tma.load.2d [ur4], [0, ur8], [ur3]\nwait:\nmbar.try_wait p0, [ur3], 0\n@!p0 bra.uni wait\nhalt\n"),
            std::move(launch));
        expect_sim_panic(machine, "swizzle64 buffer 1088 is not a multiple of 128", "a 128-byte-row tile at 1088");
    }
    {
        nanosm::Launch launch;
        launch.active_warps = 1;
        image_bytes(launch.test_shmem, 8192);
        nanosm::Machine machine(canonical_profile(), compile(
            "umov ur0, 1040\numov ur1, 4096\numov ur2, 0\nmma.async.init.m64n64k8 [ur2], [ur0], [ur1]\nhalt\n"),
            std::move(launch));
        expect_sim_panic(machine, "mma operand A at 1040 reads 16 bytes per row at row spacing 16",
            "an operand of 16-byte rows at 1040");
    }
}

// One transposed load with 128-byte stored rows (V-like: the global rows are
// keys of 64 fp16; the stored rows are the 64 positions along dimension 0,
// each holding the 64 keys): element (x, y) at position y of stored row x.
// A sector's sixteen fp16 go to sixteen neighbouring stored rows, which the
// shuffle spreads over sixteen banks: one placement round per sector.
void test_swizzle64_transposed_load() {
    constexpr uint32_t tile_address = 1024;
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.test_shmem, tile_address + 8192);
    auto& gmem = image_bytes(launch.gmem_image, 8192);
    const auto value = [](uint32_t x, uint32_t y) { return static_cast<uint16_t>(0x100 + 3 * y + 5 * x); };
    for (uint32_t y = 0; y < 64; ++y)
        for (uint32_t x = 0; x < 64; ++x) write_u16(gmem, 2 * (64 * y + x), value(x, y));
    nanosm::TmaDescriptor descriptor;
    descriptor.element_bytes = 2;
    descriptor.layout = nanosm::TileLayout::swizzle64;
    descriptor.transposed = true;
    descriptor.dimensions = {64, 64, 1, 1};
    descriptor.strides = {64, 0, 0};
    descriptor.tile_length = 64;
    descriptor.tile_rows = 64;
    write_tma_descriptor(shmem, 0, descriptor);
    write_tx_mbar(shmem, 128, 8192);
    nanosm::Machine machine(canonical_profile(), compile(tma_operands_asm(4, tile_address, 3, 128, 8, {0, 0}) +
        "tma.load.2d [ur4], [0, ur8], [ur3]\nwait:\nmbar.try_wait p0, [ur3], 0\n@!p0 bra.uni wait\nhalt\n"),
        std::move(launch));
    machine.run(50000);
    require(machine.read_mbar_state(128).phase, "the transposed load did not complete");
    for (uint32_t y = 0; y < 64; ++y) {
        for (uint32_t x = 0; x < 64; ++x) {
            const auto address = reference_swizzle64(tile_address, 128, x, 2 * y);
            uint16_t actual = 0;
            std::memcpy(&actual, machine.shmem().data() + address, sizeof(actual));
            require(actual == value(x, y), "transposed element (" + std::to_string(x) + ", " +
                std::to_string(y) + ") is not at position y of stored row x");
        }
    }
    const auto& summary = machine.trace().summary();
    require(summary.tma_port_requests == 256 && summary.tma_port_rounds == 256,
        "the transposed load did not place one sector per port-A round: " +
            std::to_string(summary.tma_port_rounds) + " rounds for " + std::to_string(summary.tma_port_requests));
}

// A one-dimensional tensor of 4 fp32 and a tile of 8 at coordinate -4: the
// first four elements are outside the tensor (zero-filled), the last four are
// the tensor.
void test_tma_rank1_load_and_oob_zero_fill() {
    constexpr uint32_t descriptor_address = 0;
    constexpr uint32_t mbar_address = 128;
    constexpr uint32_t tile_address = 256;
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.test_shmem, tile_address + 32);
    auto& gmem = image_bytes(launch.gmem_image, 32);
    for (uint32_t i = 0; i < 4; ++i) write_u32(gmem, i * 4, 0x100u + i);

    nanosm::TmaDescriptor descriptor;
    descriptor.base = 0;
    descriptor.dimensions = {4, 1, 1, 1};
    descriptor.tile_length = 8;
    write_tma_descriptor(shmem, descriptor_address, descriptor);
    nanosm::MbarState mbar;
    mbar.valid = true;
    mbar.expected = 1;
    mbar.tx_bytes = 32;
    write_u64(shmem, mbar_address, nanosm::encode_mbar_state(mbar));

    nanosm::Machine machine(canonical_profile(), compile(tma_operands_asm(4, tile_address, 3, mbar_address, 8, {-4}) + R"(
        tma.load.1d [ur4], [0, ur8], [ur3]
    wait:
        mbar.try_wait p0, [ur3], 0
        @!p0 bra.uni wait
        halt
    )"), std::move(launch));
    machine.run();
    for (uint32_t i = 0; i < 4; ++i) {
        require(machine.read_shmem_u32(tile_address + i * 4) == 0, "tma OOB load was not zero");
        require(
            machine.read_shmem_u32(tile_address + (i + 4) * 4) == 0x100u + i,
            "tma rank-1 load mapped the wrong global element");
    }
    const auto completed = machine.read_mbar_state(mbar_address);
    require(completed.phase && completed.tx_bytes == 0, "tma load completed the mbar incorrectly");
    const auto& summary = machine.trace().summary();
    require(
        summary.tma_commands == 1 && summary.tma_command_completions == 1,
        "tma load command lifetime was not counted");
    require(
        summary.tma_logical_bytes == 32 && summary.tma_useful_bytes == 16 &&
            summary.tma_sector_bytes == 32,
        "tma load logical/useful/physical byte accounting is wrong");
}

void test_tma_store_preserves_other_sector_bytes() {
    constexpr uint32_t descriptor_address = 64;
    constexpr uint32_t mbar_address = 128;
    constexpr uint32_t tile_address = 256;
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.test_shmem, tile_address + 16);
    auto& gmem = image_bytes(launch.gmem_image, 64);
    std::fill(gmem.begin(), gmem.end(), 0x5a);
    for (uint32_t i = 0; i < 4; ++i) write_u32(shmem, tile_address + i * 4, 0x200u + i);

    write_tma_descriptor(shmem, descriptor_address, plain_tma_1d(4, 12));
    nanosm::MbarState mbar;
    mbar.valid = true;
    mbar.expected = 1;
    mbar.tx_bytes = 16;
    write_u64(shmem, mbar_address, nanosm::encode_mbar_state(mbar));

    nanosm::Machine machine(canonical_profile(), compile(tma_operands_asm(4, tile_address, 3, mbar_address, 8) + R"(
        tma.store.1d [64, ur8], [ur4], [ur3]
    wait:
        mbar.try_wait p0, [ur3], 0
        @!p0 bra.uni wait
        halt
    )"), std::move(launch));
    machine.run();
    for (uint32_t address = 0; address < 12; ++address) {
        require(machine.gmem()[address] == 0x5a, "tma store damaged the sector prefix");
    }
    for (uint32_t i = 0; i < 4; ++i) {
        require(machine.read_gmem_u32(12 + i * 4) == 0x200u + i, "tma store value mismatch");
    }
    for (uint32_t address = 28; address < 64; ++address) {
        require(machine.gmem()[address] == 0x5a, "tma store damaged the sector suffix");
    }
    const auto& summary = machine.trace().summary();
    require(
        summary.tma_logical_bytes == 16 && summary.tma_useful_bytes == 16 &&
            summary.tma_sector_bytes == 32,
        "tma store logical/useful/physical byte accounting is wrong");
}

// Two dimensions with a row stride and an origin, and a third dimension
// walked in software (once the rank-3 strides test). A 3 x 4 x 2 fp32 tensor
// with row stride 5 and plane stride 30 = 6 rows is described as
// two-dimensional (3 x 10, row stride 5): plane z is rows 6z .. 6z + 3. A loop
// issues one 2 x 2 tile per plane at (1, 1 + 6z), each into its own buffer 64
// bytes on, both completing into one mbar. (A three-dimensional descriptor,
// plane stride 30, with a .3d command would walk the planes in the TMA.)
void test_tma_rank2_strides_planes_in_software() {
    constexpr uint32_t descriptor_address = 0;
    constexpr uint32_t mbar_address = 128;
    constexpr uint32_t tile_address = 256;
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.test_shmem, tile_address + 128);
    auto& gmem = image_bytes(launch.gmem_image, 64 * 4);
    for (uint32_t i = 0; i < 64; ++i) write_u32(gmem, i * 4, 1000 + i);

    nanosm::TmaDescriptor descriptor;
    descriptor.dimensions = {3, 10, 1, 1};
    descriptor.strides = {5, 0, 0};
    descriptor.tile_length = 2;
    descriptor.tile_rows = 2;
    write_tma_descriptor(shmem, descriptor_address, descriptor);
    write_tx_mbar(shmem, mbar_address, 32);

    nanosm::Machine machine(canonical_profile(), compile(tma_operands_asm(4, tile_address, 3, mbar_address, 8, {1, 1}) + R"(
        mov r4, 0
    plane:
        tma.load.2d [ur4], [0, ur8], [ur3]
        uaddi ur9, ur9, 6
        uaddi ur4, ur4, 64
        addi r4, r4, 1
        setp.lt.u32 p0, r4, 2
        @p0 bra.uni plane
    wait:
        mbar.try_wait p0, [ur3], 0
        @!p0 bra.uni wait
        halt
    )"), std::move(launch));
    machine.run();
    const std::array<std::array<uint32_t, 4>, 2> expected{{{1006, 1007, 1011, 1012}, {1036, 1037, 1041, 1042}}};
    for (uint32_t plane = 0; plane < 2; ++plane) {
        for (uint32_t i = 0; i < 4; ++i) {
            require(machine.read_shmem_u32(tile_address + 64 * plane + i * 4) == expected[plane][i],
                "tma rank-2 stride / origin mapping is wrong in plane " + std::to_string(plane));
        }
    }
    require(machine.trace().summary().tma_command_completions == 2, "one command per plane did not complete");
}

void test_profile_stamp_rejects_mismatch() {
    const auto profile = nanosm::load_device_profile("profiles/nanosm-v0.json");
    const auto program = qtx_program("mov r0, 1\nhalt", profile);
    const auto assembly = nanosm::emit_asm(program);
    const auto kernel = nanosm::assemble_asm(assembly, profile);
    require(kernel.profile_name == profile.name, "profile name was not stamped into kernel");
    require(
        kernel.profile_fingerprint == nanosm::device_profile_fingerprint(profile),
        "profile fingerprint was not stamped into kernel");

    auto other = profile;
    ++other.timing.fadd;
    try {
        (void)nanosm::assemble_asm(assembly, other);
        throw std::runtime_error("qtxas accepted assembly scheduled for another profile");
    } catch (const nanosm::SimError& error) {
        require(
            std::string(error.what()).find("does not match") != std::string::npos,
            "profile mismatch produced the wrong error");
    }

    nanosm::Launch launch;
    launch.active_warps = 1;
    try {
        nanosm::Machine machine(other, kernel, std::move(launch));
        throw std::runtime_error("simulator accepted a kernel for another profile");
    } catch (const nanosm::SimError& error) {
        require(
            std::string(error.what()).find("does not match simulator") != std::string::npos,
            "simulator profile mismatch produced the wrong error");
    }
}

void test_dram_uses_its_own_clock() {
    auto config = canonical_profile();
    config.core_clock_hz = 4;
    config.dram_clock_hz = 1;
    config.gmem_latency_dram_ticks = 0;
    config.gmem_bus_bytes_per_dram_tick = 16;
    nanosm::Launch launch;
    launch.active_warps = 1;
    image_bytes(launch.gmem_image, 64);
    // ldg.uni: one declared sector request, two 16-byte beats. (This test
    // used a 16-lane scalar ldg when the coalescer made that two sectors.)
    nanosm::Machine machine(config, compile(R"(
        umov ur0, 0
        ldg.uni r1, [ur0]
        halt
    )", config), std::move(launch), {.timeline = true, .debug = true});
    machine.run();
    std::vector<uint64_t> beat_cycles;
    for (const auto& event : machine.trace().events()) {
        if (event.category == nanosm::TraceCategory::gmem && event.name == "bus beat") {
            beat_cycles.push_back(event.start_cycle);
        }
    }
    require(beat_cycles.size() == 2, "one sector did not take two 16-byte DRAM beats");
    require(beat_cycles[1] - beat_cycles[0] == 4, "DRAM beats advanced at the core clock");
}

// ---------------------------------------------------------------------------
// The walker's runs: one run, zero
// fill or skipped stretch per cycle, never past a row's end.

const std::string tma_wait = "wait:\nmbar.try_wait p0, [ur3], 0\n@!p0 bra.uni wait\nhalt\n";

// The cycle a command's completion flips the mbar at `mbar` (Q, between ticks).
uint64_t tick_until_phase(nanosm::Machine& machine, uint32_t mbar) {
    while (!machine.read_mbar_state(mbar).phase) {
        require(!machine.done() && machine.cycle() < 50000, "the tma command never completed its barrier");
        machine.tick();
    }
    return machine.cycle();
}

// A store whose tile is entirely outside its tensor (its rows, its columns,
// or a dimension it does not walk) writes nothing and completes its barrier
// with the tile's bytes; the walker skips one row per cycle, so a tile of 33
// rows completes exactly 32 cycles after a tile of 1.
void test_tma_store_entirely_out_of_range() {
    constexpr uint32_t mbar_address = 128;
    constexpr uint32_t tile_address = 1024;
    struct Case { std::string what; std::vector<int32_t> coordinates; };
    for (const auto& c : {Case{"rows", {0, 100, 0}}, Case{"columns", {-64, 0, 0}}, Case{"third dimension", {0, 0, 5}}}) {
        std::vector<uint64_t> completion;
        for (const uint32_t rows : {1u, 33u}) {
            nanosm::Launch launch;
            launch.active_warps = 1;
            auto& shmem = image_bytes(launch.test_shmem, tile_address + 64 * 33);
            auto& gmem = image_bytes(launch.gmem_image, 8192);
            std::fill(gmem.begin(), gmem.end(), 0x5a);
            for (uint32_t i = 0; i < 16 * rows; ++i) write_u32(shmem, tile_address + 4 * i, 0x1000 + i);
            nanosm::TmaDescriptor descriptor;
            descriptor.dimensions = {16, 40, 3, 1};
            descriptor.strides = {16, 640, 0};
            descriptor.tile_length = 16;
            descriptor.tile_rows = static_cast<uint16_t>(rows);
            write_tma_descriptor(shmem, 0, descriptor);
            write_tx_mbar(shmem, mbar_address, 64 * rows);
            nanosm::Machine machine(canonical_profile(),
                compile(tma_operands_asm(4, tile_address, 3, mbar_address, 8, c.coordinates) +
                    "tma.store.3d [0, ur8], [ur4], [ur3]\n" + tma_wait),
                std::move(launch));
            completion.push_back(tick_until_phase(machine, mbar_address));
            machine.run(50000);
            require(std::all_of(machine.gmem().begin(), machine.gmem().begin() + 8192, [](uint8_t b) { return b == 0x5a; }),
                "an out-of-range store (" + c.what + ") wrote gmem");
            const auto state = machine.read_mbar_state(mbar_address);
            require(state.phase && state.tx_bytes == 0, "an out-of-range store (" + c.what + ") left its barrier's bytes");
            const auto& summary = machine.trace().summary();
            require(summary.tma_command_completions == 1 && summary.tma_useful_bytes == 0 &&
                    summary.tma_sector_bytes == 0 && summary.tma_port_requests == 0,
                "an out-of-range store (" + c.what + ") moved data");
        }
        require(completion[1] == completion[0] + 32, "an out-of-range store (" + c.what + ") of 33 rows completed " +
            std::to_string(completion[1] - completion[0]) + " cycles after one of 1 row, not 32 (one skip per row)");
    }
}

// A load and a store of a 16 x 10 fp32 tile at (-4, -2) of an 8 x 6 tensor:
// four out-of-range columns on each side and two out-of-range rows above and
// below. The load zero-fills them (one fill per row-side band, at most a
// sector each: 4 rows x 2 + 6 rows x 2 = 20 fills, 6 sectors); the store
// leaves every byte outside the tensor alone.
void test_tma_out_of_range_bands() {
    constexpr uint32_t mbar_address = 128;
    constexpr uint32_t tile_address = 1024;
    constexpr uint32_t length = 16;
    constexpr uint32_t rows = 10;
    const auto inside = [](uint32_t x, uint32_t y) { return x >= 4 && x < 12 && y >= 2 && y < 8; };
    nanosm::TmaDescriptor descriptor;
    descriptor.dimensions = {8, 6, 1, 1};
    descriptor.strides = {8, 0, 0};
    descriptor.tile_length = length;
    descriptor.tile_rows = rows;
    const std::string operands = tma_operands_asm(4, tile_address, 3, mbar_address, 8, {-4, -2});
    {
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.test_shmem, tile_address + 4 * length * rows + 64);
        auto& gmem = image_bytes(launch.gmem_image, 256);
        for (uint32_t i = 0; i < 48; ++i) write_u32(gmem, 4 * i, 0x100 + i);
        std::fill(shmem.begin() + tile_address, shmem.end(), 0xa5);
        write_tma_descriptor(shmem, 0, descriptor);
        write_tx_mbar(shmem, mbar_address, 4 * length * rows);
        nanosm::Machine machine(canonical_profile(), compile(operands + "tma.load.2d [ur4], [0, ur8], [ur3]\n" + tma_wait),
            std::move(launch));
        machine.run(50000);
        for (uint32_t y = 0; y < rows; ++y) for (uint32_t x = 0; x < length; ++x) {
            const uint32_t expected = inside(x, y) ? 0x100 + 8 * (y - 2) + (x - 4) : 0;
            require(machine.read_shmem_u32(tile_address + 4 * (length * y + x)) == expected,
                "out-of-range band load: element (" + std::to_string(x) + ", " + std::to_string(y) + ")");
        }
        require(machine.read_shmem_u32(tile_address + 4 * length * rows) == 0xa5a5a5a5u, "the load wrote past its tile");
        const auto& summary = machine.trace().summary();
        require(summary.tma_port_requests == 26 && summary.tma_sector_bytes == 192 && summary.tma_useful_bytes == 192,
            "out-of-range band load: " + std::to_string(summary.tma_port_requests) + " placements, " +
                std::to_string(summary.tma_sector_bytes) + " sector bytes (expected 20 fills + 6 sectors, 192)");
    }
    {
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.test_shmem, tile_address + 4 * length * rows);
        auto& gmem = image_bytes(launch.gmem_image, 256);
        std::fill(gmem.begin(), gmem.end(), 0x5a);
        for (uint32_t i = 0; i < length * rows; ++i) write_u32(shmem, tile_address + 4 * i, 0x2000 + i);
        write_tma_descriptor(shmem, 0, descriptor);
        write_tx_mbar(shmem, mbar_address, 4 * length * rows);
        nanosm::Machine machine(canonical_profile(), compile(operands + "tma.store.2d [0, ur8], [ur4], [ur3]\n" + tma_wait),
            std::move(launch));
        machine.run(50000);
        for (uint32_t i = 0; i < 64; ++i) {
            const uint32_t expected = i < 48 ? 0x2000 + length * (i / 8 + 2) + (i % 8 + 4) : 0x5a5a5a5au;
            require(machine.read_gmem_u32(4 * i) == expected, "out-of-range band store: gmem word " + std::to_string(i));
        }
        require(machine.read_mbar_state(mbar_address).phase, "out-of-range band store did not complete");
    }
}

// The placement tickets take one entry a cycle. A 64 x 32 fp32 tile of an
// 8-column tensor: every row is one sector read and seven zero fills, so the
// walker zero-fills while earlier rows' sectors come back, and on this
// profile some fill falls in a cycle the ingress completes a sector. That
// sector's ticket enters and the zero fill waits a cycle; every element
// still lands.
void test_tma_zero_fill_waits_for_a_returning_sector() {
    constexpr uint32_t mbar_address = 128;
    constexpr uint32_t tile_address = 1024;
    constexpr uint32_t length = 64;
    constexpr uint32_t rows = 32;
    nanosm::TmaDescriptor descriptor;
    descriptor.dimensions = {8, rows, 1, 1};
    descriptor.strides = {8, 0, 0};
    descriptor.tile_length = length;
    descriptor.tile_rows = rows;
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.test_shmem, tile_address + 4 * length * rows);
    auto& gmem = image_bytes(launch.gmem_image, 4 * 8 * rows);
    for (uint32_t i = 0; i < 8 * rows; ++i) write_u32(gmem, 4 * i, 0x100 + i);
    std::fill(shmem.begin() + tile_address, shmem.end(), 0xa5);
    write_tma_descriptor(shmem, 0, descriptor);
    write_tx_mbar(shmem, mbar_address, 4 * length * rows);
    nanosm::Machine machine(canonical_profile(),
        compile(tma_operands_asm(4, tile_address, 3, mbar_address, 8, {0, 0}) + "tma.load.2d [ur4], [0, ur8], [ur3]\n" +
            tma_wait),
        std::move(launch));
    machine.enable_edge_probe();
    machine.run(50000);
    const auto& events = machine.edge_events();
    require(std::any_of(events.begin(), events.end(),
                [](const auto& event) { return event.edge == nanosm::LsuEdge::tma_zero_fill_wait; }),
        "no zero fill met a returning sector's ticket in its cycle; the test proves little");
    for (uint32_t y = 0; y < rows; ++y) for (uint32_t x = 0; x < length; ++x) {
        const uint32_t expected = x < 8 ? 0x100 + 8 * y + x : 0;
        require(machine.read_shmem_u32(tile_address + 4 * (length * y + x)) == expected,
            "zero fills between returning sectors: element (" + std::to_string(x) + ", " + std::to_string(y) + ")");
    }
    require(machine.trace().summary().tma_port_requests == 8 * rows, "expected a sector and seven zero fills per row");
}

// A transposed swizzle64 fp16 tile (64 stored rows of 16 positions, 32-byte
// rows) at (-8, -2) of a 40 x 20 tensor: positions 8..47 along dimension 0
// and tile rows 2..15 are inside. The load places element (x, y) at
// position y of stored row x (zero outside); the store writes back exactly
// the inside.
void test_tma_transposed_out_of_range() {
    constexpr uint32_t mbar_address = 128;
    constexpr uint32_t tile_address = 2048;
    constexpr uint32_t length = 64;
    constexpr uint32_t rows = 16;
    constexpr uint32_t dim_x = 40;
    constexpr uint32_t dim_y = 20;
    const auto value = [](uint32_t gx, uint32_t gy) { return static_cast<uint16_t>(0x100 + 7 * gy + gx); };
    const auto inside = [](uint32_t x, uint32_t y) { return x >= 8 && x < 48 && y >= 2; };
    nanosm::TmaDescriptor descriptor;
    descriptor.element_bytes = 2;
    descriptor.layout = nanosm::TileLayout::swizzle64;
    descriptor.transposed = true;
    descriptor.dimensions = {dim_x, dim_y, 1, 1};
    descriptor.strides = {dim_x, 0, 0};
    descriptor.tile_length = length;
    descriptor.tile_rows = rows;
    const auto placement = [&](uint32_t x, uint32_t y) { return reference_swizzle64(tile_address, 32, x, 2 * y); };
    const auto read_u16 = [](const std::vector<uint8_t>& memory, uint32_t address) {
        return static_cast<uint16_t>(memory.at(address) | memory.at(address + 1) << 8);
    };
    const std::string operands = tma_operands_asm(4, tile_address, 3, mbar_address, 8, {-8, -2});
    {
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.test_shmem, tile_address + 2 * length * rows);
        auto& gmem = image_bytes(launch.gmem_image, 2 * dim_x * dim_y);
        for (uint32_t y = 0; y < dim_y; ++y) for (uint32_t x = 0; x < dim_x; ++x) write_u16(gmem, 2 * (dim_x * y + x), value(x, y));
        std::fill(shmem.begin() + tile_address, shmem.end(), 0xa5);
        write_tma_descriptor(shmem, 0, descriptor);
        write_tx_mbar(shmem, mbar_address, 2 * length * rows);
        nanosm::Machine machine(canonical_profile(), compile(operands + "tma.load.2d [ur4], [0, ur8], [ur3]\n" + tma_wait),
            std::move(launch));
        machine.run(50000);
        for (uint32_t y = 0; y < rows; ++y) for (uint32_t x = 0; x < length; ++x) {
            const uint16_t expected = inside(x, y) ? value(x - 8, y - 2) : 0;
            require(read_u16(machine.shmem(), placement(x, y)) == expected,
                "transposed out-of-range load: element (" + std::to_string(x) + ", " + std::to_string(y) + ")");
        }
    }
    {
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.test_shmem, tile_address + 2 * length * rows);
        auto& gmem = image_bytes(launch.gmem_image, 2 * dim_x * dim_y + 64);
        std::fill(gmem.begin(), gmem.end(), 0x5a);
        for (uint32_t y = 0; y < rows; ++y) for (uint32_t x = 0; x < length; ++x) {
            write_u16(shmem, placement(x, y), static_cast<uint16_t>(0x4000 + 64 * y + x));
        }
        write_tma_descriptor(shmem, 0, descriptor);
        write_tx_mbar(shmem, mbar_address, 2 * length * rows);
        nanosm::Machine machine(canonical_profile(), compile(operands + "tma.store.2d [0, ur8], [ur4], [ur3]\n" + tma_wait),
            std::move(launch));
        machine.run(50000);
        for (uint32_t gy = 0; gy < dim_y; ++gy) for (uint32_t gx = 0; gx < dim_x; ++gx) {
            const uint32_t x = gx + 8;
            const uint32_t y = gy + 2;
            const uint16_t expected = y < rows ? static_cast<uint16_t>(0x4000 + 64 * y + x) : 0x5a5a;
            require(read_u16(machine.gmem(), 2 * (dim_x * gy + gx)) == expected,
                "transposed out-of-range store: tensor element (" + std::to_string(gx) + ", " + std::to_string(gy) + ")");
        }
        for (uint32_t byte = 2 * dim_x * dim_y; byte < 2 * dim_x * dim_y + 64; ++byte) {
            require(machine.gmem()[byte] == 0x5a, "transposed out-of-range store wrote past its tensor");
        }
    }
}

// A dense tile of 1-byte elements: the longest runs (32 elements, a whole
// sector) and crooked ones. 3 rows of 96 at (5, 1) of a 200-byte-row tensor:
// rows start at 205, 405, 605, so each row is 4 runs (19 + 32 + 32 + 13,
// 11 + 32 + 32 + 21, 3 + 32 + 32 + 29) and every placement, its shared and
// global offsets differing mod 4, takes at most two rounds.
void test_tma_byte_tile_longest_run() {
    constexpr uint32_t mbar_address = 128;
    constexpr uint32_t tile_address = 1024;
    constexpr uint32_t length = 96;
    constexpr uint32_t rows = 3;
    nanosm::TmaDescriptor descriptor;
    descriptor.element_bytes = 1;
    descriptor.dimensions = {200, 4, 1, 1};
    descriptor.strides = {200, 0, 0};
    descriptor.tile_length = length;
    descriptor.tile_rows = rows;
    const auto value = [](uint32_t address) { return static_cast<uint8_t>(address * 7 + 3); };
    const std::string operands = tma_operands_asm(4, tile_address, 3, mbar_address, 8, {5, 1});
    {
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.test_shmem, tile_address + length * rows);
        auto& gmem = image_bytes(launch.gmem_image, 800);
        for (uint32_t i = 0; i < 800; ++i) gmem[i] = value(i);
        write_tma_descriptor(shmem, 0, descriptor);
        write_tx_mbar(shmem, mbar_address, length * rows);
        nanosm::Machine machine(canonical_profile(), compile(operands + "tma.load.2d [ur4], [0, ur8], [ur3]\n" + tma_wait),
            std::move(launch));
        machine.run(50000);
        for (uint32_t y = 0; y < rows; ++y) for (uint32_t x = 0; x < length; ++x) {
            require(machine.shmem()[tile_address + length * y + x] == value(200 * (y + 1) + 5 + x),
                "byte tile load: element (" + std::to_string(x) + ", " + std::to_string(y) + ")");
        }
        const auto& summary = machine.trace().summary();
        require(summary.tma_port_requests == 12 && summary.tma_sector_bytes == 384 &&
                summary.tma_port_rounds > summary.tma_port_requests && summary.tma_port_rounds <= 2 * summary.tma_port_requests,
            "byte tile load: " + std::to_string(summary.tma_port_requests) + " placements of " +
                std::to_string(summary.tma_port_rounds) + " rounds (expected 12 sectors, at most two rounds each)");
    }
    {
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.test_shmem, tile_address + length * rows);
        auto& gmem = image_bytes(launch.gmem_image, 800);
        std::fill(gmem.begin(), gmem.end(), 0x5a);
        for (uint32_t i = 0; i < length * rows; ++i) shmem[tile_address + i] = value(i);
        write_tma_descriptor(shmem, 0, descriptor);
        write_tx_mbar(shmem, mbar_address, length * rows);
        nanosm::Machine machine(canonical_profile(), compile(operands + "tma.store.2d [0, ur8], [ur4], [ur3]\n" + tma_wait),
            std::move(launch));
        machine.run(50000);
        for (uint32_t address = 0; address < 800; ++address) {
            const uint32_t row = address / 200;
            const uint32_t column = address % 200;
            const bool tile = row >= 1 && row <= rows && column >= 5 && column < 5 + length;
            const uint8_t expected = tile ? value(length * (row - 1) + column - 5) : 0x5a;
            require(machine.gmem()[address] == expected, "byte tile store: gmem byte " + std::to_string(address));
        }
        require(machine.trace().summary().tma_sector_bytes == 384, "byte tile store: not 12 sectors");
    }
}

// The walker's start stages (tma.walker_start_cycles): the canonical
// profile declares 2, absent is 2, 0..8 are
// legal (0: no stages), anything else is refused, and the value is in the
// fingerprint. A lone 2 x 16 fp32 load at (-4, 0) (zero fills and sectors)
// makes its first placement exactly one cycle later per stage, with the same
// bytes.
void test_tma_walker_start_stages() {
    std::ifstream input("profiles/nanosm-v0.json");
    nlohmann::json document;
    input >> document;
    const auto parsed = nanosm::parse_device_profile(document.dump());
    require(parsed.tma_walker_start_cycles == 2 && document["tma"].contains("walker_start_cycles"),
        "tma.walker_start_cycles: the canonical profile does not declare 2");
    auto absent = document;
    absent["tma"].erase("walker_start_cycles");
    require(nanosm::device_profile_fingerprint(nanosm::parse_device_profile(absent.dump())) ==
            nanosm::device_profile_fingerprint(parsed), "tma.walker_start_cycles: the default is not 2");
    for (const auto value : {nlohmann::json(-1), nlohmann::json(1.5), nlohmann::json(9)}) {
        auto modified = document;
        modified["tma"]["walker_start_cycles"] = value;
        try {
            (void)nanosm::parse_device_profile(modified.dump());
            throw std::runtime_error("tma.walker_start_cycles = " + value.dump() + " was accepted");
        } catch (const nanosm::ProfileError&) {}
    }
    auto zero = document;
    zero["tma"]["walker_start_cycles"] = 0;
    require(nanosm::parse_device_profile(zero.dump()).tma_walker_start_cycles == 0, "tma.walker_start_cycles = 0 refused");
    auto nine = canonical_profile();
    nine.tma_walker_start_cycles = 9;
    try {
        nanosm::validate_device_profile(nine);
        throw std::runtime_error("tma.walker_start_cycles = 9 passed validation");
    } catch (const nanosm::ProfileError&) {}
    for (const uint32_t stages : {0u, 3u}) {
        auto other = canonical_profile();
        other.tma_walker_start_cycles = stages;
        require(nanosm::device_profile_fingerprint(other) != nanosm::device_profile_fingerprint(canonical_profile()),
            "tma.walker_start_cycles was omitted from the profile fingerprint");
    }

    constexpr uint32_t mbar_address = 128;
    constexpr uint32_t tile_address = 1024;
    nanosm::TmaDescriptor descriptor;
    descriptor.dimensions = {12, 2, 1, 1};
    descriptor.strides = {12, 0, 0};
    descriptor.tile_length = 16;
    descriptor.tile_rows = 2;
    std::vector<uint64_t> completion;
    std::vector<std::vector<uint8_t>> landed;
    for (const uint32_t stages : {0u, 1u, 2u, 5u}) {
        auto profile = canonical_profile();
        profile.tma_walker_start_cycles = stages;
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.test_shmem, tile_address + 128);
        auto& gmem = image_bytes(launch.gmem_image, 96);
        for (uint32_t i = 0; i < 24; ++i) write_u32(gmem, 4 * i, 0x300 + i);
        std::fill(shmem.begin() + tile_address, shmem.end(), 0xa5);
        write_tma_descriptor(shmem, 0, descriptor);
        write_tx_mbar(shmem, mbar_address, 128);
        nanosm::Machine machine(profile, compile(tma_operands_asm(4, tile_address, 3, mbar_address, 8, {-4, 0}) +
            "tma.load.2d [ur4], [0, ur8], [ur3]\n" + tma_wait, profile), std::move(launch));
        machine.enable_edge_probe();
        machine.run(50000);
        // The first placement round: the zero fill of x = 0..3, the walk's
        // first step (core cycles only; the end-to-end time also moves by
        // DRAM-tick alignment).
        const auto& events = machine.edge_events();
        const auto first = std::find_if(events.begin(), events.end(),
            [](const auto& event) { return event.edge == LsuEdge::tma_place_round; });
        require(first != events.end(), "no tma placement round");
        completion.push_back(first->cycle);
        landed.emplace_back(machine.shmem().begin() + tile_address, machine.shmem().begin() + tile_address + 128);
        for (uint32_t i = 0; i < 32; ++i) {
            const uint32_t x = i % 16;
            const uint32_t expected = x < 4 ? 0 : 0x300 + 12 * (i / 16) + x - 4;
            require(machine.read_shmem_u32(tile_address + 4 * i) == expected,
                std::to_string(stages) + " start stages: tile word " + std::to_string(i));
        }
    }
    const uint64_t added[] = {0, 1, 2, 5};
    for (std::size_t k = 1; k < completion.size(); ++k) {
        require(completion[k] == completion[0] + added[k], std::to_string(added[k]) + " start stages: the first " +
            "placement came " + std::to_string(completion[k] - completion[0]) + " cycles after none's, not " +
            std::to_string(added[k]));
        require(landed[k] == landed[0], "the start stages changed the landed bytes");
    }
}

} // namespace

// a store entirely out of range: nothing written, one skip per row, barrier completed
TEST(Tma, tma_store_entirely_out_of_range) {
    test_tma_store_entirely_out_of_range();
}

// out-of-range bands on every side: zero-filled by a load, untouched by a store
TEST(Tma, tma_out_of_range_bands) {
    test_tma_out_of_range_bands();
}

// zero fills: one placement ticket a cycle; a returning sector's goes first
TEST(Tma, tma_zero_fill_waits_for_a_returning_sector) {
    test_tma_zero_fill_waits_for_a_returning_sector();
}

// a transposed swizzle64 tile with out-of-range parts, loaded and stored
TEST(Tma, tma_transposed_out_of_range) {
    test_tma_transposed_out_of_range();
}

// 1-byte elements: sector-long runs and crooked ones, loaded and stored
TEST(Tma, tma_byte_tile_longest_run) {
    test_tma_byte_tile_longest_run();
}

// tma.walker_start_cycles: parsed, fingerprinted, one cycle per stage, same bytes
TEST(Tma, tma_walker_start_stages) {
    test_tma_walker_start_stages();
}

// tma direction is the opcode; one descriptor serves both
TEST(Tma, tma_direction_is_the_opcode) {
    test_tma_direction_is_the_opcode();
}

// tma swizzle64 load/store, transposed or not
TEST(Tma, tma_swizzle64_transposed_load_store) {
    test_tma_swizzle64_transposed_load_store();
}

// a swizzle64 TMA tile is the same whatever the profile's mma.layout
TEST(Tma, tma_swizzle64_tile_ignores_the_mma_layout) {
    test_tma_swizzle64_tile_ignores_the_mma_layout();
}

// tma swizzle64 feeds MMA
TEST(TmaSlow, tma_swizzle64_feeds_mma_orientations) {
    test_tma_swizzle64_feeds_mma_orientations();
}

// tma swizzle64 census: row counts and K extents vs the MMA's addressing
TEST(TmaSlow, tma_swizzle64_census_against_mma) {
    test_tma_swizzle64_census_against_mma();
}

// tma swizzle64 validation
TEST(Tma, tma_swizzle64_validation) {
    test_tma_swizzle64_validation();
}

// sixteen 4-row and 64 one-row swizzle64 loads build the 64-row tile
TEST(TmaSlow, tma_swizzle64_pieces_land_anywhere) {
    test_tma_swizzle64_pieces_land_anywhere();
}

// a swizzle64 tile or operand off its base rule: the sanitizer's PANIC
TEST(Tma, swizzle64_base_rule_panics) {
    test_swizzle64_base_rule_panics();
}

// tma swizzle64 sectors
TEST(TmaSlow, tma_swizzle64_sector_economics) {
    test_tma_swizzle64_sector_economics();
}

// the swizzle64 layout function == the rule written out
TEST(Tma, swizzle64_layout_matches_the_rule) {
    test_swizzle64_layout_matches_the_rule();
}

// tma fp32 swizzle64 validation
TEST(Tma, tma_fp32_swizzle64_validation) {
    test_tma_fp32_swizzle64_validation();
}

// tma fp32 swizzle64 store/load
TEST(TmaSlow, tma_fp32_swizzle64_store_and_load) {
    test_tma_fp32_swizzle64_store_and_load();
}

// the tensor core reads TMA tiles of 128-byte rows, and m32 operands at rows 32 and 21
TEST(TmaSlow, swizzle64_mma_reads_a_tma_tile) {
    test_swizzle64_mma_reads_a_tma_tile();
}

// one transposed swizzle64 load with 128-byte stored rows
TEST(Tma, swizzle64_transposed_load) {
    test_swizzle64_transposed_load();
}

// tma rank-1 load and OOB
TEST(Tma, tma_rank1_load_and_oob_zero_fill) {
    test_tma_rank1_load_and_oob_zero_fill();
}

// tma store masked sector
TEST(Tma, tma_store_preserves_other_sector_bytes) {
    test_tma_store_preserves_other_sector_bytes();
}

// tma rank-2 strides, planes walked in software
TEST(Tma, tma_rank2_strides_planes_in_software) {
    test_tma_rank2_strides_planes_in_software();
}

// profile stamp
TEST(Tma, profile_stamp_rejects_mismatch) {
    test_profile_stamp_rejects_mismatch();
}

// DRAM clock domain
TEST(Tma, dram_uses_its_own_clock) {
    test_dram_uses_its_own_clock();
}

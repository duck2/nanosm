// The TMA's commands: the direction, xor64 placement and its census against
// the MMA's operand addressing, the limits of xor64 tiles, a dense tile read
// by a dense-profile MMA, out-of-bounds handling; and the profile stamp and
// the DRAM clock. Moved from the original tests/tests.cpp
// (trash/test-setup-2026-10-22.md). The descriptor (64 bytes in shared
// memory, src/tma.hpp) holds the global tensor, the tile's shape and its
// shared-memory layout (dense, or xor64 and whether it is transposed; the
// rules are src/layout.hpp's, which the MMA reads its operands with). The
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
        const auto program = nanosm::compile_qtx(text + "\nhalt\n", profile);
        require(nanosm::format_instruction(nanosm::decode(nanosm::encode(program.instructions.at(0))), false) == text,
            "'" + text + "' did not survive encode / decode / print");
    }
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.shmem_image, 4096);
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

// The fp16 xor64 address of element `position` (< 8) of stored row `row`,
// written out.
uint32_t reference_xor64_address(uint32_t base, uint32_t row, uint32_t position) {
    return base + 4 * (4 * row + ((position >> 1) ^ ((row >> 2) & 1))) + 2 * (position & 1);
}

// One xor64 descriptor, transposed or not, used by a load and then a store:
// an fp16 tile of 64 stored rows by 8 sliced at (3, 2) of a tensor with row
// stride 83. Element x of tile row y is position x of stored row y, or
// (transposed) position y of stored row x.
void test_tma_xor64_transposed_load_store() {
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
        descriptor.layout = nanosm::TileLayout::xor64;
        descriptor.transposed = transposed;
        descriptor.dimensions = {dim_x, dim_y, 1, 1};
        descriptor.strides = {stride, 0, 0};
        descriptor.tile_length = static_cast<uint16_t>(tile_length);
        descriptor.tile_rows = static_cast<uint16_t>(tile_rows);
        const auto placement = [&](uint32_t x, uint32_t y) {
            return transposed ? reference_xor64_address(tile_address, x, y)
                              : reference_xor64_address(tile_address, y, x);
        };

        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.shmem_image, tile_address + 1088);
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
            require(addresses.insert(address).second, "xor64 map is not bijective");
            const uint32_t gx = 3 + x;
            const uint32_t gy = 2 + y;
            const uint16_t actual = static_cast<uint16_t>(
                machine.read_shmem_u32(address & ~3u) >> (8 * (address & 3u)));
            require(actual ==
                static_cast<uint16_t>(0x100 + 3 * gy + 5 * gx),
                "xor64 load mapped the wrong element");
        }
        require(addresses.size() == 512, "xor64 map did not cover the operand");

        nanosm::Launch store_launch;
        store_launch.active_warps = 1;
        auto& store_shmem = image_bytes(store_launch.shmem_image, tile_address + 1088);
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
                "xor64 store mapped the wrong element");
        }
    }
}

void test_tma_xor64_feeds_mma_orientations() {
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
        auto& shmem = image_bytes(launch.shmem_image, 7168);
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
        // One xor64 descriptor per tensor (A at 0, B at 64), each tile 64
        // stored rows by K = 8; each chunk's command names its buffer and
        // mbar: A chunk c at 1024 + 2048c (mbar 512 + 16c), B chunk c at 2048
        // + 2048c (mbar 520 + 16c). The MMA stores both operands as rows of K
        // (A: m, B: n). A's K runs along its rows; B's runs along the row
        // under QK^T (K's rows are keys) and across the rows under PV (V's
        // rows are keys: transposed).
        nanosm::TmaDescriptor a;
        a.element_bytes = 2; a.layout = nanosm::TileLayout::xor64;
        a.base = base_a; a.dimensions = {16, 64, 1, 1};
        a.strides = {stride_a, 0, 0}; a.tile_length = 8; a.tile_rows = 64;
        nanosm::TmaDescriptor b;
        b.element_bytes = 2; b.layout = nanosm::TileLayout::xor64; b.base = base_b;
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
        // Chunk 1's coordinates per rfc07's table: A's K advances along
        // dimension 0 in both orientations, (8, 0); B's along dimension 0
        // under QK^T, (8, 0), but along dimension 1 under PV, (0, 8).
        const std::string chunk1_b = pv
            ? "umov ur8, 0\numov ur9, 8\ntma.load.2d [ur14], [64, ur8], [ur15]\n"
            : "tma.load.2d [ur14], [64, ur8], [ur15]\n";
        const auto kernel = compile(".acc C, 0, 64, 64\n" + tma_operands_asm(1, 1024, 2, 512, 8, {0, 0}) +
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
            mma.async.init.m64n64k8 C, [ur10], [ur11]
            umov ur10, 3072
            umov ur11, 4096
            mma.async.m64n64k8 C, [ur10], [ur11]
            umov ur4, 544
            mma.commit [ur4]
        wait_commit:
            mbar.try_wait p0, [ur4], 0
            @!p0 bra.uni wait_commit
            umov ur8, 0
            umov ur9, 0
            acc.ld.v8 r20, C, ur8
            umov ur12, 48
            umov ur13, 56
            acc.ld.v8 r28, C, ur12
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

// The census (trash/tma_generalize_proto.py check 3): for every allowed
// stored-row count R (the powers of two 4 .. 128) and every fp16 K extent (8
// .. 64), a TMA load of an xor64 R x K tile writes exactly the compact tile an
// MMA with m = R (as A) or n = R (as B) and k = K reads: every element where
// the MMA's own operand addressing (nanosm::mma_operand_address) looks for
// it, and no byte outside the tile's R x K x 2 bytes. A's global is m rows of
// K; B's is K rows of n (transposed). The buffer starts on a 64-byte boundary
// that is not a 1024-byte one.
void test_tma_xor64_census_against_mma() {
    const auto& profile = canonical_profile();
    constexpr uint32_t base = 1088;
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
        for (uint32_t rows = nanosm::tma_xor64_min_rows; rows <= nanosm::tma_xor64_max_rows; rows *= 2) {
            for (uint32_t k_extent = 8; k_extent <= nanosm::tma_xor64_max_sliced; k_extent += 8) {
                const auto what = std::string(a_role ? "A" : "B") + ", " + std::to_string(rows) + " rows, K " +
                    std::to_string(k_extent);
                // Row-major global, the tile the whole tensor: A is rows x K
                // (K along the row), B is K x n (K across the rows).
                const uint32_t length = a_role ? k_extent : rows;
                const uint32_t height = a_role ? rows : k_extent;
                nanosm::TmaDescriptor descriptor;
                descriptor.element_bytes = 2;
                descriptor.layout = nanosm::TileLayout::xor64;
                descriptor.transposed = !a_role;
                descriptor.dimensions = {length, height, 1, 1};
                descriptor.strides = {length, 0, 0};
                descriptor.tile_length = static_cast<uint16_t>(length);
                descriptor.tile_rows = static_cast<uint16_t>(height);
                nanosm::Launch launch;
                launch.active_warps = 1;
                auto& shmem = image_bytes(launch.shmem_image, shmem_bytes);
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
                        const auto address = nanosm::mma_operand_address(profile, shape, base, a_role, outer, k);
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
    require(cases == 2 * 6 * 8, "the census did not cover every stored-row count and K extent");
}

// The TMA does not know who reads a tile: a TMA-only kernel with explicit
// xor64 descriptors (one tile not transposed, one transposed) leaves every
// shared byte identical when only the profile's mma.layout changes (xor64 ->
// dense).
void test_tma_xor64_tile_ignores_the_mma_layout() {
    constexpr uint32_t mbar_address = 128;
    constexpr uint32_t stride = 83;
    auto dense = canonical_profile();
    dense.mma_layout = nanosm::TileLayout::dense;
    nanosm::validate_device_profile(dense);
    const auto run = [&](const nanosm::DeviceProfile& profile) {
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.shmem_image, 8192);
        auto& gmem = image_bytes(launch.gmem_image, 2 * stride * 70);
        for (uint32_t y = 0; y < 70; ++y)
            for (uint32_t x = 0; x < stride; ++x)
                write_u16(gmem, 2 * (y * stride + x), static_cast<uint16_t>(0x100 + 3 * y + 5 * x));
        for (const bool transposed : {false, true}) {
            nanosm::TmaDescriptor descriptor;
            descriptor.element_bytes = 2;
            descriptor.layout = nanosm::TileLayout::xor64;
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
    const auto with_xor64 = run(canonical_profile());
    const auto with_dense = run(dense);
    require(std::any_of(with_xor64.begin() + 1024, with_xor64.begin() + 6144, [](uint8_t b) { return b != 0; }),
        "the tiles were not loaded");
    require(with_xor64 == with_dense, "an xor64 TMA tile changed with the profile's mma.layout");
}

void test_tma_wide_xor64_validation() {
    // The sliced extent (the tile length, or the rows when transposed): a
    // multiple of 8 fp16, at most 64. The stored-row count (the other
    // extent): a power of two 4 .. 128. A dense tile cannot be transposed.
    nanosm::TmaDescriptor descriptor;
    descriptor.element_bytes = 2;
    descriptor.layout = nanosm::TileLayout::xor64;
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
    for (uint32_t k = 8; k <= 64; k += 8) {
        require(rejection(along, k, 64).empty() && rejection(across, 64, k).empty(),
            "an xor64 sliced extent was rejected");
    }
    for (const uint32_t k : {4u, 12u, 60u, 72u}) {
        require(rejection(along, k, 64).find("sliced extent") != std::string::npos &&
                rejection(across, 64, k).find("sliced extent") != std::string::npos,
            "xor64 sliced extent " + std::to_string(k) + " was accepted");
    }
    for (uint32_t rows = 1; rows <= 256; ++rows) {
        const bool allowed = rows >= 4 && rows <= 128 && (rows & (rows - 1)) == 0;
        require(rejection(along, 16, rows).empty() == allowed && rejection(across, rows, 16).empty() == allowed,
            "xor64 stored-row count " + std::to_string(rows) + " was " + (allowed ? "rejected" : "accepted"));
        if (!allowed) {
            require(rejection(along, 16, rows).find("stored-row count") != std::string::npos,
                "xor64 stored-row count " + std::to_string(rows) + " rejected with the wrong diagnostic");
        }
    }
    auto dense = descriptor;
    dense.layout = nanosm::TileLayout::dense;
    dense.transposed = true;
    dense.tile_length = 16;
    dense.tile_rows = 64;
    require(encode_error(dense).find("dense tiles cannot be transposed") != std::string::npos,
        "a transposed dense tile was accepted");

    // The span is the compact tile, 64 x 64 x 2 = 8192 bytes: at 61440 it
    // ends past the 64 KiB shmem.
    nanosm::TmaDescriptor big;
    big.element_bytes = 2;
    big.layout = nanosm::TileLayout::xor64;
    big.dimensions = {128, 128, 1, 1};
    big.strides = {128, 0, 0};
    big.tile_length = 64;
    big.tile_rows = 64;
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.shmem_image, 33032);
    image_bytes(launch.gmem_image, 65536);
    write_tma_descriptor(shmem, 0, big);
    write_u64(shmem, 33024, nanosm::encode_mbar_state({false, true, 1, 0, 8192}));
    nanosm::Machine machine(canonical_profile(),
        compile(tma_operands_asm(4, 61440, 3, 33024, 8, {0, 0}) + "tma.load.2d [ur4], [0, ur8], [ur3]\nhalt\n"),
        std::move(launch));
    try {
        machine.run(20000);
        throw std::runtime_error("oversized xor64 span was accepted");
    } catch (const nanosm::SimError& error) {
        require(std::string(error.what()).find("does not fit in shmem") != std::string::npos,
            "oversized xor64 span rejected with the wrong diagnostic");
    }
}

// Independent restatement of the 64-row map: the sliced position k picks the
// 1024-byte slice (16 bytes x 64 rows) and the position inside the row.
// Never calls the production helper.
uint32_t reference_wide_xor64(uint32_t base, uint32_t row, uint32_t k) {
    return base + 1024 * (k / 8) +
        4 * (4 * row + (((k % 8) >> 1) ^ ((row >> 2) & 1))) +
        2 * (k % 2);
}

// `commands` loads of a K / commands slab each, one descriptor, the tile's
// coordinates in ur8, ur9 (set by the caller, moved on by `step` between
// commands): command i's buffer (ur4) starts at tile + 1024 x (its K / 8) x
// i. Every command completes into the mbar in ur3, which the wait reads.
std::string xor64_slab_loads(uint32_t commands, uint32_t tile, uint32_t mbar, const std::string& step) {
    const uint32_t k_extent = 64 / commands;
    std::string source = "umov ur3, " + std::to_string(mbar) + "\n";
    for (uint32_t i = 0; i < commands; ++i) {
        if (i != 0) source += step;
        source += "umov ur4, " + std::to_string(tile + 1024 * (k_extent / 8) * i) +
            "\ntma.load.2d [ur4], [0, ur8], [ur3]\n";
    }
    return source + "wait:\nmbar.try_wait p0, [ur3], 0\n@!p0 bra.uni wait\nhalt\n";
}

void test_tma_wide_xor64_matches_narrow() {
    constexpr uint32_t mbar_address = 576;
    constexpr uint32_t tile_address = 1024;
    constexpr uint32_t stride = 83;
    constexpr uint32_t dim_x = 80;
    constexpr uint32_t dim_y = 70;
    // Transposed: the sliced origin advances along y per k8 chunk (else along x).
    for (const bool transposed : {false, true}) {
        const auto run_span = [&](uint32_t commands) {
            nanosm::Launch launch;
            launch.active_warps = 1;
            auto& shmem = image_bytes(launch.shmem_image, tile_address + 8192);
            auto& gmem = image_bytes(launch.gmem_image, 2 * stride * dim_y + 64);
            for (uint32_t y = 0; y < dim_y; ++y)
                for (uint32_t x = 0; x < dim_x; ++x)
                    write_u16(gmem, 2 * (y * stride + x),
                        static_cast<uint16_t>(0x100 + 3 * y + 5 * x));
            const uint32_t k_extent = 64 / commands;
            nanosm::TmaDescriptor descriptor;
            descriptor.element_bytes = 2;
            descriptor.layout = nanosm::TileLayout::xor64;
            descriptor.transposed = transposed;
            descriptor.dimensions = {dim_x, dim_y, 1, 1};
            descriptor.strides = {stride, 0, 0};
            descriptor.tile_length = static_cast<uint16_t>(transposed ? 64 : k_extent);
            descriptor.tile_rows = static_cast<uint16_t>(transposed ? k_extent : 64);
            write_tma_descriptor(shmem, 0, descriptor);
            write_u64(shmem, mbar_address,
                nanosm::encode_mbar_state({false, true, 1, 0, 8192}));
            const std::string source = "umov ur8, 3\numov ur9, 2\n" + xor64_slab_loads(commands, tile_address,
                mbar_address, transposed ? "uaddi ur9, ur9, 8\n" : "uaddi ur8, ur8, 8\n");
            nanosm::Machine machine(canonical_profile(), compile(source), std::move(launch));
            machine.run(200000);
            std::vector<uint32_t> words;
            words.reserve(2048);
            for (uint32_t offset = 0; offset < 8192; offset += 4)
                words.push_back(machine.read_shmem_u32(tile_address + offset));
            return words;
        };
        const auto wide = run_span(1);
        const auto narrow = run_span(8);
        require(wide == narrow, "wide xor64 command disagreed with eight narrow commands");
        std::set<uint32_t> addresses;
        for (uint32_t y = 0; y < 64; ++y) {
            for (uint32_t x = 0; x < 64; ++x) {
                const uint32_t k = transposed ? y : x;
                const uint32_t row = transposed ? x : y;
                const auto address = reference_wide_xor64(tile_address, row, k);
                require(address >= tile_address && address < tile_address + 8192,
                    "wide xor64 address left the span");
                require(addresses.insert(address).second, "wide xor64 map is not bijective");
                const uint32_t gx = 3 + x;
                const uint32_t gy = 2 + y;
                const auto word = wide[(address - tile_address) / 4];
                const auto actual = static_cast<uint16_t>(word >> (8 * (address & 3u)));
                require(actual == static_cast<uint16_t>(0x100 + 3 * gy + 5 * gx),
                    "wide xor64 load mapped the wrong element");
            }
        }
        require(addresses.size() == 4096, "wide xor64 map did not cover the span");
    }
}

void test_tma_wide_xor64_sector_economics() {
    constexpr uint32_t mbar_address = 576;
    constexpr uint32_t tile_address = 1024;
    // Natural head-innermost K: 64 keys x 64 fp16, dense 128-byte rows.
    const auto run_summary = [&](uint32_t commands) {
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.shmem_image, tile_address + 8192);
        auto& gmem = image_bytes(launch.gmem_image, 8192);
        for (uint32_t i = 0; i < 4096; ++i)
            write_u16(gmem, 2 * i, static_cast<uint16_t>((i * 2654435761u) >> 16));
        const uint32_t k_extent = 64 / commands;
        nanosm::TmaDescriptor descriptor;
        descriptor.element_bytes = 2;
        descriptor.layout = nanosm::TileLayout::xor64; // not transposed: the QK^T orientation of K's natural layout
        descriptor.dimensions = {64, 64, 1, 1};
        descriptor.strides = {64, 0, 0};
        descriptor.tile_length = static_cast<uint16_t>(k_extent);
        descriptor.tile_rows = 64;
        write_tma_descriptor(shmem, 0, descriptor);
        write_u64(shmem, mbar_address, nanosm::encode_mbar_state({false, true, 1, 0, 8192}));
        const std::string source = "umov ur8, 0\numov ur9, 0\n" +
            xor64_slab_loads(commands, tile_address, mbar_address, "uaddi ur8, ur8, 8\n");
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

// The pre-2026-10-03 fp16 operand formula (A's: one rule for every xor64
// tile since 2026-11-18), written out literally. The byte primitive and the
// element function (src/layout.hpp) must reproduce it for every fp16 address
// in play; the element function's slices are 16 x rows bytes apart.
void test_xor64_byte_primitive_matches_fp16_formula() {
    for (uint32_t base : {0u, 1024u, 40960u, 65472u}) {
        for (uint32_t row = 0; row < 64; ++row) {
            for (uint32_t k = 0; k < 8; ++k) {
                const uint32_t old = base + 4 * (4 * row + ((k >> 1) ^ ((row >> 2) & 1))) + 2 * (k & 1);
                require(nanosm::xor64_byte_address(base, row, 2 * k) == old &&
                        nanosm::xor64_byte_address(base, row, 2 * k + 1) == old + 1,
                    "xor64 byte primitive diverged from the fp16 formula");
                for (uint32_t slice = 0; slice < 2; ++slice) {
                    require(nanosm::xor64_address(base, 64, row, 8 * slice + k, 2) == old + 1024 * slice,
                        "xor64 element address diverged from the fp16 formula");
                }
            }
        }
    }
}

void test_tma_fp32_xor64_validation() {
    // The sliced extent runs along the row (a) or, transposed, across the rows (b).
    const auto rejection = [](uint8_t element_bytes, bool transposed, uint32_t tile_length, uint32_t tile_rows) {
        nanosm::TmaDescriptor descriptor;
        descriptor.element_bytes = element_bytes;
        descriptor.layout = nanosm::TileLayout::xor64;
        descriptor.transposed = transposed;
        descriptor.dimensions = {128, 128, 1, 1};
        descriptor.strides = {128, 0, 0};
        descriptor.tile_length = static_cast<uint16_t>(tile_length);
        descriptor.tile_rows = static_cast<uint16_t>(tile_rows);
        return encode_error(descriptor);
    };
    constexpr bool a = false;
    constexpr bool b = true;
    // fp32: four elements per 16-byte row; sliced extent a multiple of 4, up to 64.
    require(rejection(4, a, 64, 64).empty(), "fp32 xor64 64x64 tile was rejected");
    require(rejection(4, b, 64, 64).empty(), "fp32 transposed xor64 64x64 tile was rejected");
    require(rejection(4, a, 4, 64).empty() && rejection(4, a, 12, 64).empty(),
        "fp32 xor64 single/odd slice counts were rejected");
    require(rejection(4, a, 64, 32).empty() && rejection(4, b, 32, 8).empty(),
        "fp32 xor64 tiles of other row counts were rejected");
    for (const uint32_t k : {6u, 2u, 68u}) {
        require(rejection(4, a, k, 64).find("sliced extent") != std::string::npos,
            "fp32 xor64 accepted a sliced extent that is not whole slices");
    }
    require(rejection(4, a, 64, 48).find("stored-row count") != std::string::npos,
        "fp32 xor64 accepted 48 rows");
    require(rejection(4, b, 64, 6).find("sliced extent") != std::string::npos,
        "fp32 transposed xor64 accepted a sliced extent that is not whole slices");
    require(!rejection(8, a, 64, 64).empty(), "8-byte xor64 elements were accepted");
    require(rejection(1, a, 64, 64).find("2- or 4-byte elements") != std::string::npos,
        "1-byte xor64 elements were accepted");
    // fp16 keeps its eight-element rows: 12 is whole fp32 slices, not fp16.
    for (const uint32_t k : {12u, 4u, 72u}) {
        require(rejection(2, a, k, 64).find("sliced extent") != std::string::npos,
            "fp16 xor64 wrong sliced extent was accepted after the fp32 generalization");
    }
}

// Independent fp32 xor64 address: slice = col / 4 (four floats per 16-byte
// row; 1024 bytes for 64 rows), row = the 16-byte row, word-in-row = (col %
// 4) XOR bit 2 of the row. Never calls the production primitive.
uint32_t reference_fp32_xor64(uint32_t base, uint32_t row, uint32_t col) {
    return base + 1024 * (col / 4) + 16 * row + 4 * ((col % 4) ^ ((row / 4) % 2));
}

float fp32_tile_value(uint32_t row, uint32_t col) {
    // Periods 31 in both axes: coprime to 4, 8, 16, 64. Small integers, exact.
    return static_cast<float>(static_cast<int>((row * 17 + col * 11) % 31) - 15);
}

void test_tma_fp32_xor64_store_and_load() {
    constexpr uint32_t mbar_address = 128;
    constexpr uint32_t tile_address = 1024;
    constexpr uint32_t tile_bytes = 16384;
    constexpr uint64_t gmem_base = 512;
    constexpr uint32_t gmem_size = 512 + tile_bytes + 512;

    std::set<uint32_t> addresses;
    for (uint32_t row = 0; row < 64; ++row) {
        for (uint32_t col = 0; col < 64; ++col) {
            const auto address = reference_fp32_xor64(tile_address, row, col);
            require(address % 4 == 0 && address >= tile_address &&
                    address < tile_address + tile_bytes,
                "fp32 xor64 address left the 16-slice span");
            require(addresses.insert(address).second, "fp32 xor64 map is not bijective");
        }
    }
    require(addresses.size() == 4096, "fp32 xor64 map did not cover the 16 slices");

    // A 64 x 64 fp32 tile, the whole tensor: xor64 (not transposed) or dense.
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
        auto& shmem = image_bytes(launch.shmem_image, tile_address + tile_bytes);
        auto& gmem = image_bytes(launch.gmem_image, gmem_size);
        std::fill(gmem.begin(), gmem.end(), 0x5a);
        for (uint32_t row = 0; row < 64; ++row)
            for (uint32_t col = 0; col < 64; ++col)
                write_u32(shmem, layout == nanosm::TileLayout::xor64
                        ? reference_fp32_xor64(tile_address, row, col)
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
                    "fp32 xor64 TMA store mapped the wrong element at row " +
                        std::to_string(row) + " col " + std::to_string(col));
            }
        }
        for (uint32_t offset = 0; offset < gmem_size; ++offset) {
            if (offset >= gmem_base && offset < gmem_base + tile_bytes) continue;
            require(machine.gmem()[offset] == 0x5a, "fp32 TMA store wrote outside its tile");
        }
        return machine.trace().summary();
    };
    const auto swizzled_store = run_store(nanosm::TileLayout::xor64);
    // A 32-byte sector is eight consecutive fp32 of one row: four in slice
    // col/4 and four in the next. Both halves sit in the same row of their
    // slices, so on the same four banks (1024 B = 256 words = 0 mod 16) at
    // different words: two port-A rounds per sector. Linear placement of the
    // same sector is eight adjacent words: one round.
    require(swizzled_store.tma_command_completions == 1 &&
            swizzled_store.tma_useful_bytes == tile_bytes &&
            swizzled_store.tma_sector_bytes == tile_bytes,
        "fp32 xor64 TMA store was not 1.00x sector traffic");
    require(swizzled_store.tma_port_requests == 512 && swizzled_store.tma_port_rounds == 1024,
        "fp32 xor64 TMA store did not read two port-A rounds per sector");
    const auto linear_store = run_store(nanosm::TileLayout::dense);
    require(linear_store.tma_port_requests == 512 && linear_store.tma_port_rounds == 512,
        "linear fp32 TMA store did not read one port-A round per sector");

    // LOAD: row-major gmem -> shmem; every element checked at the independent address.
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.shmem_image, tile_address + tile_bytes);
    auto& gmem = image_bytes(launch.gmem_image, gmem_size);
    std::fill(shmem.begin() + tile_address, shmem.end(), 0x7f);
    for (uint32_t row = 0; row < 64; ++row)
        for (uint32_t col = 0; col < 64; ++col)
            write_u32(gmem, static_cast<uint32_t>(gmem_base) + 4 * (64 * row + col),
                nanosm::float_bits(fp32_tile_value(row, col)));
    write_tma_descriptor(shmem, 0, make_descriptor(nanosm::TileLayout::xor64));
    write_u64(shmem, mbar_address, nanosm::encode_mbar_state(mbar));
    nanosm::Machine machine(canonical_profile(), compile(program(false)), std::move(launch));
    machine.run(200000);
    for (uint32_t row = 0; row < 64; ++row) {
        for (uint32_t col = 0; col < 64; ++col) {
            require(machine.read_shmem_u32(reference_fp32_xor64(tile_address, row, col)) ==
                    nanosm::float_bits(fp32_tile_value(row, col)),
                "fp32 xor64 TMA load mapped the wrong element at row " +
                    std::to_string(row) + " col " + std::to_string(col));
        }
    }
    const auto& load_summary = machine.trace().summary();
    require(load_summary.tma_port_requests == 512 && load_summary.tma_port_rounds == 1024,
        "fp32 xor64 TMA load did not write two port-A rounds per sector");
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
    auto& shmem = image_bytes(launch.shmem_image, tile_address + 32);
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
    auto& shmem = image_bytes(launch.shmem_image, tile_address + 16);
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
    auto& shmem = image_bytes(launch.shmem_image, tile_address + 128);
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
        setp.lt p0, r4, 2
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
    const auto program = nanosm::compile_qtx("mov r0, 1\nhalt", profile);
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

} // namespace

// tma direction is the opcode; one descriptor serves both
TEST(Tma, tma_direction_is_the_opcode) {
    test_tma_direction_is_the_opcode();
}

// tma xor64 load/store, transposed or not
TEST(Tma, tma_xor64_transposed_load_store) {
    test_tma_xor64_transposed_load_store();
}

// an xor64 TMA tile is the same whatever the profile's mma.layout
TEST(Tma, tma_xor64_tile_ignores_the_mma_layout) {
    test_tma_xor64_tile_ignores_the_mma_layout();
}

// tma xor64 feeds MMA
TEST(TmaSlow, tma_xor64_feeds_mma_orientations) {
    test_tma_xor64_feeds_mma_orientations();
}

// tma xor64 census: every outer extent and K extent vs the MMA's addressing
TEST(TmaSlow, tma_xor64_census_against_mma) {
    test_tma_xor64_census_against_mma();
}

// tma wide xor64 validation
TEST(Tma, tma_wide_xor64_validation) {
    test_tma_wide_xor64_validation();
}

// tma wide xor64 vs narrow
TEST(TmaSlow, tma_wide_xor64_matches_narrow) {
    test_tma_wide_xor64_matches_narrow();
}

// tma wide xor64 sectors
TEST(TmaSlow, tma_wide_xor64_sector_economics) {
    test_tma_wide_xor64_sector_economics();
}

// xor64 byte primitive == fp16 formula
TEST(Tma, xor64_byte_primitive_matches_fp16_formula) {
    test_xor64_byte_primitive_matches_fp16_formula();
}

// tma fp32 xor64 validation
TEST(Tma, tma_fp32_xor64_validation) {
    test_tma_fp32_xor64_validation();
}

// tma fp32 xor64 store/load
TEST(TmaSlow, tma_fp32_xor64_store_and_load) {
    test_tma_fp32_xor64_store_and_load();
}

// tma rank-1 load and OOB
TEST(Tma, tma_rank1_load_and_oob_zero_fill) {
    test_tma_rank1_load_and_oob_zero_fill();
}

// tma store masked sector
TEST(Tma, tma_store_preserves_other_sector_bytes) {
    test_tma_store_preserves_other_sector_bytes();
}

// tma rank-2 strides, planes walked in software (was rank-3 strides)
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

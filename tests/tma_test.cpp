// TMA descriptors and the tma.load / tma.store operands, xor64 placement and
// its census against the MMA's operand addressing, out-of-bounds handling,
// the profile stamp. Moved from the original tests/tests.cpp
// (trash/test-setup-2026-10-22.md); converted to the NVIDIA-style split
// (trash/tma-nvidia-alignment-2026-11-06.md): the descriptor holds the global
// tensor and the box, the instruction its direction, shared address and mbar.
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

void test_tma_descriptor_encoding() {
    nanosm::TmaDescriptor descriptor;
    descriptor.rank = 2;
    descriptor.gmem_base = 0x1122334455667788ull;
    descriptor.dimensions = {17, 19};
    descriptor.row_stride = 32;
    descriptor.box = {4, 5};
    descriptor.element_bytes = 2;
    const auto bytes = nanosm::encode_tma_descriptor(descriptor);
    require(bytes[0] == 2 && bytes[1] == 0 && bytes[2] == 2 && bytes[3] == 0, "tma header bytes changed");
    require(bytes[8] == 0x88 && bytes[15] == 0x11, "tma gmem base is not little-endian");
    require(bytes[16] == 17 && bytes[20] == 19 && bytes[24] == 32 && bytes[28] == 4 && bytes[32] == 5,
        "tma dimension / row stride / box offsets changed");
    for (std::size_t offset = 0; offset < bytes.size(); ++offset) {
        if ((offset >= 4 && offset < 8) || offset >= 36) require(bytes[offset] == 0, "a reserved tma byte is not zero");
    }
    const auto decoded = nanosm::decode_tma_descriptor(bytes.data(), bytes.size());
    require(
        decoded.rank == descriptor.rank && decoded.transpose == descriptor.transpose &&
            decoded.gmem_base == descriptor.gmem_base && decoded.dimensions == descriptor.dimensions &&
            decoded.row_stride == descriptor.row_stride && decoded.box == descriptor.box &&
            decoded.element_bytes == descriptor.element_bytes && decoded.swizzle == descriptor.swizzle,
        "tma descriptor did not round trip");

    for (const std::size_t offset : {4u, 7u, 36u, 63u}) {
        auto corrupt = bytes;
        corrupt[offset] = 1;
        try {
            (void)nanosm::decode_tma_descriptor(corrupt.data(), corrupt.size());
            throw std::runtime_error("tma decoder accepted a nonzero reserved byte");
        } catch (const nanosm::SimError& error) {
            require(std::string(error.what()).find("reserved") != std::string::npos,
                "reserved tma byte produced the wrong panic");
        }
    }

    // Rank 3 is gone: the instruction carries two coordinates (x, y).
    auto rank3 = bytes;
    rank3[0] = 3;
    try {
        (void)nanosm::decode_tma_descriptor(rank3.data(), rank3.size());
        throw std::runtime_error("tma decoder accepted rank 3");
    } catch (const nanosm::SimError& error) {
        require(std::string(error.what()).find("rank must be 1 or 2") != std::string::npos,
            "rank 3 produced the wrong panic");
    }

    descriptor.transpose = true;
    require(encode_error(descriptor).find("transpose") != std::string::npos,
        "a linear tma descriptor accepted transpose");
    descriptor.transpose = false;
    descriptor.swizzle = 3;
    require(encode_error(descriptor).find("swizzle") != std::string::npos, "an unknown tma swizzle was accepted");

    for (uint8_t layout : {uint8_t{1}, uint8_t{2}}) for (bool transpose : {false, true}) {
        nanosm::TmaDescriptor operand;
        operand.rank = 2;
        operand.transpose = transpose;
        operand.element_bytes = 2;
        operand.swizzle = layout;
        operand.dimensions = {96, 96};
        operand.row_stride = 101;
        operand.box = layout == 1 ? std::array<uint32_t, 2>{8, 64} : std::array<uint32_t, 2>{64, 8};
        const auto encoded = nanosm::encode_tma_descriptor(operand);
        const auto round_trip = nanosm::decode_tma_descriptor(encoded.data(), encoded.size());
        require(round_trip.swizzle == layout && round_trip.transpose == transpose &&
                round_trip.box == operand.box,
            "xor64 descriptor layout/transpose did not round trip");
    }

    nanosm::TmaDescriptor invalid;
    invalid.rank = 2; invalid.element_bytes = 2; invalid.swizzle = 1;
    invalid.dimensions = {64, 64}; invalid.row_stride = 64;
    invalid.box = {8, 63};
    require(encode_error(invalid).find("outer extent") != std::string::npos,
        "an xor64 box of 63 rows was accepted");
    invalid.box = {12, 64};
    require(encode_error(invalid).find("K extent") != std::string::npos,
        "an xor64 box with K = 12 fp16 was accepted");
}

// The tma.load / tma.store address operand (src/tma.hpp): the completion
// mbar / 8 in bits 15:0, the box's shared address / 64 in bits 31:16. Its
// round trip and reach (past the 64 KiB canonical shmem: 512 KiB of mbars,
// 4 MiB of boxes), the addresses it cannot hold, the one-mov range (a box
// below 2 KiB), and on the machine the two range checks: a box that does not
// fit shmem, an mbar outside it.
void test_tma_address_operand() {
    require(nanosm::pack_tma_addresses(0, 0) == 0 && nanosm::pack_tma_addresses(64, 8) == 0x00010001u &&
            nanosm::pack_tma_addresses(1024, 4104) == (16u << 16 | 513u),
        "the tma address operand's fields moved");
    const auto far = nanosm::pack_tma_addresses(4 * 1024 * 1024 - 64, 512 * 1024 - 8);
    const auto unpacked = nanosm::unpack_tma_addresses(far);
    require(far == 0xffffffffu && unpacked.shmem == 4 * 1024 * 1024 - 64 && unpacked.mbar == 512 * 1024 - 8,
        "the tma address operand does not reach 4 MiB of boxes and 512 KiB of mbars");
    for (const uint32_t shared : {0u, 64u, 1984u, 2048u, 40960u, 65472u}) {
        for (const uint32_t mbar : {0u, 8u, 4104u, 49184u, 65528u}) {
            const auto back = nanosm::unpack_tma_addresses(nanosm::pack_tma_addresses(shared, mbar));
            require(back.shmem == shared && back.mbar == mbar, "the tma address operand did not round trip");
        }
    }
    // One mov (a 22-bit signed immediate) holds the operand of any box below 2 KiB.
    require(nanosm::pack_tma_addresses(1984, 512 * 1024 - 8) == static_cast<uint32_t>(nanosm::encoding::immediate_max) &&
            nanosm::pack_tma_addresses(2048, 0) == static_cast<uint32_t>(nanosm::encoding::immediate_max) + 1,
        "the one-mov range of the tma address operand moved");
    const auto pack_error = [](uint32_t shared, uint32_t mbar) {
        try {
            (void)nanosm::pack_tma_addresses(shared, mbar);
            return std::string();
        } catch (const nanosm::SimError& error) {
            return std::string(error.what());
        }
    };
    require(pack_error(32, 8).find("64-byte aligned") != std::string::npos &&
            pack_error(64, 4).find("8-byte aligned") != std::string::npos &&
            pack_error(4 * 1024 * 1024, 0).find("below 4 MiB") != std::string::npos &&
            pack_error(0, 512 * 1024).find("below 512 KiB") != std::string::npos,
        "an address the operand cannot hold was packed");

    // A `bytes` box at `shared` completing into `mbar`, on the canonical 64 KiB.
    const auto run = [](uint32_t shared, uint32_t mbar, uint32_t bytes) {
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.shmem_image, 65536);
        image_bytes(launch.gmem_image, bytes);
        write_tma_descriptor(shmem, 0, linear_tma(bytes));
        if (mbar + 8 <= shmem.size()) write_tx_mbar(shmem, mbar, bytes);
        return std::make_unique<nanosm::Machine>(canonical_profile(),
            compile("mov r0, 0\nmov r1, 0\n" + tma_operand_asm(2, shared, mbar) + "umov ur3, " + std::to_string(mbar) +
                "\ntma.load [0], r0, r2\nwait:\nmbar.try_wait p0, [ur3], 0\n@!p0 bra.uni wait\nhalt\n"),
            std::move(launch));
    };
    // The last box and the last mbar of shmem are in range.
    auto last_box = run(65536 - 64, 128, 64);
    last_box->run(20000);
    auto last_mbar = run(1024, 65536 - 8, 64);
    last_mbar->run(20000);
    require(last_box->read_mbar_state(128).phase && last_mbar->read_mbar_state(65536 - 8).phase,
        "a box or an mbar at the end of shmem did not complete");
    expect_sim_panic(*run(65536 - 64, 128, 128), "tma logical box does not fit in shmem", "a box ending past shmem");
    expect_sim_panic(*run(65536, 128, 64), "tma logical box does not fit in shmem", "a box starting past shmem");
    expect_sim_panic(*run(1024, 65536, 64), "tma completion mbar is outside shmem", "an mbar past shmem");
}

// The direction is the opcode, and one descriptor (16 fp32 at gmem 0) serves
// both: a tma.load into [1024, 1088) completing into the mbar at 128, then a
// tma.store from [2048, 2112) completing into 136. Each command's bytes go to
// its own mbar, and the two opcodes encode, decode and print apart.
void test_tma_direction_is_the_opcode() {
    const auto& profile = canonical_profile();
    for (const std::string text : {"tma.load [64], r3, r5", "tma.store [64], r3, r5"}) {
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
    nanosm::Machine machine(profile, compile("mov r0, 0\nmov r1, 0\n" + tma_operand_asm(2, 1024, 128) +
        tma_operand_asm(5, 2048, 136) + R"(
        umov ur3, 128
        umov ur4, 136
        tma.load [0], r0, r2
    loaded:
        mbar.try_wait p0, [ur3], 0
        @!p0 bra.uni loaded
        tma.store [0], r0, r5
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

uint32_t reference_xor64_address(
    uint32_t base, uint8_t layout, uint32_t u, uint32_t v) {
    const uint32_t outer = layout == 1 ? v : u;
    const uint32_t k = layout == 1 ? u : v;
    const uint32_t phase = layout == 1 ? 0 : 2;
    return base + 4 * (4 * outer + ((k >> 1) ^ ((outer >> 2) & 1) ^ phase)) +
        2 * (k & 1);
}

// One descriptor per layout and transpose, used by a load and then a store.
void test_tma_xor64_transpose_load_store() {
    constexpr uint32_t mbar_address = 128;
    constexpr uint32_t tile_address = 1024;
    constexpr uint32_t stride = 83;
    const auto program = [&](const std::string& opcode) {
        return "mov r0, 3\nmov r1, 2\nmov r2, " + std::to_string(nanosm::pack_tma_addresses(tile_address, mbar_address)) +
            "\numov ur3, 128\n" + opcode + " [0], r0, r2\nwait:\nmbar.try_wait p0, [ur3], 0\n@!p0 bra.uni wait\nhalt\n";
    };
    for (uint8_t layout : {uint8_t{1}, uint8_t{2}}) {
        const uint32_t box_x = layout == 1 ? 8 : 64;
        const uint32_t box_y = layout == 1 ? 64 : 8;
        for (bool transpose : {false, true}) {
            const uint32_t footprint_x = transpose ? box_y : box_x;
            const uint32_t footprint_y = transpose ? box_x : box_y;
            const uint32_t dim_x = 3 + footprint_x + 5;
            const uint32_t dim_y = 2 + footprint_y + 3;
            nanosm::TmaDescriptor descriptor;
            descriptor.rank = 2;
            descriptor.transpose = transpose;
            descriptor.element_bytes = 2;
            descriptor.swizzle = layout;
            descriptor.dimensions = {dim_x, dim_y};
            descriptor.row_stride = stride;
            descriptor.box = {box_x, box_y};

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
            nanosm::Machine machine(canonical_profile(), compile(program("tma.load")), std::move(launch));
            machine.run(20000);
            std::set<uint32_t> addresses;
            for (uint32_t v = 0; v < box_y; ++v) for (uint32_t u = 0; u < box_x; ++u) {
                const uint32_t address = reference_xor64_address(tile_address, layout, u, v);
                require(addresses.insert(address).second, "xor64 map is not bijective");
                const uint32_t gx = 3 + (transpose ? v : u);
                const uint32_t gy = 2 + (transpose ? u : v);
                const uint16_t actual = static_cast<uint16_t>(
                    machine.read_shmem_u32(address & ~3u) >> (8 * (address & 3u)));
                require(actual ==
                    static_cast<uint16_t>(0x100 + 3 * gy + 5 * gx),
                    "xor64 transpose load mapped the wrong element");
            }
            require(addresses.size() == 512, "xor64 map did not cover the operand");

            nanosm::Launch store_launch;
            store_launch.active_warps = 1;
            auto& store_shmem = image_bytes(store_launch.shmem_image, tile_address + 1088);
            auto& store_gmem = image_bytes(store_launch.gmem_image, stride * dim_y * 2);
            std::fill(store_gmem.begin(), store_gmem.end(), 0x5a);
            for (uint32_t v = 0; v < box_y; ++v) for (uint32_t u = 0; u < box_x; ++u)
                write_u16(store_shmem, reference_xor64_address(tile_address, layout, u, v),
                    static_cast<uint16_t>(0x400 + 7 * v + 11 * u));
            write_tma_descriptor(store_shmem, 0, descriptor);
            write_u64(store_shmem, mbar_address, nanosm::encode_mbar_state(mbar));
            nanosm::Machine store_machine(canonical_profile(), compile(program("tma.store")), std::move(store_launch));
            store_machine.run(20000);
            for (uint32_t v = 0; v < box_y; ++v) for (uint32_t u = 0; u < box_x; ++u) {
                const uint32_t gx = 3 + (transpose ? v : u);
                const uint32_t gy = 2 + (transpose ? u : v);
                uint16_t actual = 0;
                const uint32_t address = 2 * (gy * stride + gx);
                std::memcpy(&actual, store_machine.gmem().data() + address, sizeof(actual));
                require(actual ==
                    static_cast<uint16_t>(0x400 + 7 * v + 11 * u),
                    "xor64 transpose store mapped the wrong element");
            }
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
        // One descriptor per tensor (A at 0, B at 64); each chunk's command
        // names its stage and mbar: A chunk c at 1024 + 2048c (mbar 512 + 16c),
        // B chunk c at 2048 + 2048c (mbar 520 + 16c).
        nanosm::TmaDescriptor a;
        a.rank = 2; a.element_bytes = 2; a.swizzle = 1;
        a.gmem_base = base_a; a.dimensions = {16, 64};
        a.row_stride = stride_a; a.box = {8, 64};
        nanosm::TmaDescriptor b;
        b.rank = 2; b.element_bytes = 2; b.swizzle = 2; b.gmem_base = base_b;
        b.box = {64, 8};
        if (!pv) {
            b.transpose = true; b.dimensions = {16, 64}; b.row_stride = stride_b_qk;
        } else {
            b.dimensions = {64, 16}; b.row_stride = stride_b_pv;
        }
        write_tma_descriptor(shmem, 0, a);
        write_tma_descriptor(shmem, 64, b);
        nanosm::MbarState ready{false, true, 1, 0, 1024};
        for (const uint32_t mbar : {512u, 520u, 528u, 536u}) write_u64(shmem, mbar, nanosm::encode_mbar_state(ready));
        nanosm::MbarState commit{false, true, 1, 1, 0};
        write_u64(shmem, 544, nanosm::encode_mbar_state(commit));
        // Chunk 1 origins per rfc07's table: A advances k in global x for both
        // orientations; B advances k in x under QK^T (transpose=1) but in y
        // under PV (transpose=0) -- V's rows are keys.
        const std::string chunk1_b = pv
            ? "mov r0, 0\nmov r1, 8\ntma.load [64], r0, r17\n"
            : "tma.load [64], r0, r17\n";
        const auto kernel = compile(".acc C, 0, 64, 64\n" + tma_operand_asm(2, 1024, 512) +
            tma_operand_asm(5, 2048, 520) + tma_operand_asm(14, 3072, 528) + tma_operand_asm(17, 4096, 536) + R"(
            mov r0, 0
            mov r1, 0
            tma.load [0], r0, r2
            tma.load [64], r0, r5
            mov r0, 8
            tma.load [0], r0, r14
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
                       : "TMA-fed QK transpose produced the wrong MMA value");
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

// The census (trash/tma_generalize_proto.py check 3): for every allowed outer
// extent R (the powers of two 4 .. 128) and every fp16 K extent (8 .. 64),
// both roles, a TMA load of an R x K box writes exactly the compact object an
// MMA with m = R (A role) or n = R (B role) and k = K reads: every element
// where the MMA's own operand addressing (nanosm::mma_operand_address) looks
// for it, and no byte outside the object's R x K x 2 bytes. The box starts
// on a 64-byte boundary that is not a 1024-byte one.
void test_tma_xor64_census_against_mma() {
    const auto& profile = canonical_profile();
    constexpr uint32_t base = 1088;
    constexpr uint32_t mbar_address = 64;
    constexpr uint32_t shmem_bytes = base + 128 * 64 * 2 + 1024;
    constexpr uint8_t poison = 0xa5;
    const auto tag = [](uint32_t outer, uint32_t k) { return static_cast<uint16_t>(0x8000u | outer << 6 | k); };
    const auto kernel = compile("mov r0, 0\nmov r1, 0\n" + tma_operand_asm(2, base, mbar_address) + R"(
        umov ur3, 64
        tma.load [0], r0, r2
    wait:
        mbar.try_wait p0, [ur3], 0
        @!p0 bra.uni wait
        halt
    )");
    uint32_t cases = 0;
    for (const bool a_role : {true, false}) {
        for (uint32_t rows = nanosm::tma_xor64_min_outer; rows <= nanosm::tma_xor64_max_outer; rows *= 2) {
            for (uint32_t k_extent = 8; k_extent <= nanosm::tma_xor64_max_k; k_extent += 8) {
                const auto what = std::string(a_role ? "A" : "B") + " role, " + std::to_string(rows) + " rows, K " +
                    std::to_string(k_extent);
                // Row-major global: A is rows x K (x = k), B is K x n (x = n).
                nanosm::TmaDescriptor descriptor;
                descriptor.rank = 2;
                descriptor.element_bytes = 2;
                descriptor.swizzle = a_role ? 1 : 2;
                descriptor.dimensions = a_role ? std::array<uint32_t, 2>{k_extent, rows}
                                               : std::array<uint32_t, 2>{rows, k_extent};
                descriptor.row_stride = descriptor.dimensions[0];
                descriptor.box = descriptor.dimensions;
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
                            what + ": the MMA reads outside the box's compact object");
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
    require(cases == 2 * 6 * 8, "the census did not cover every outer extent and K extent");
}

void test_tma_wide_xor64_validation() {
    // K (u for A, v for B): a multiple of 8 fp16, at most 64. Outer (v for A,
    // u for B): a power of two 4 .. 128.
    nanosm::TmaDescriptor descriptor;
    descriptor.rank = 2;
    descriptor.element_bytes = 2;
    descriptor.dimensions = {128, 128};
    descriptor.row_stride = 128;
    const auto rejection = [&](uint8_t swizzle, std::array<uint32_t, 2> box) {
        descriptor.swizzle = swizzle;
        descriptor.box = box;
        return encode_error(descriptor);
    };
    for (uint32_t k = 8; k <= 64; k += 8) {
        require(rejection(1, {k, 64}).empty() && rejection(2, {64, k}).empty(), "an xor64 K extent was rejected");
    }
    for (const uint32_t k : {4u, 12u, 60u, 72u}) {
        require(rejection(1, {k, 64}).find("K extent") != std::string::npos &&
                rejection(2, {64, k}).find("K extent") != std::string::npos,
            "xor64 K extent " + std::to_string(k) + " was accepted");
    }
    for (uint32_t outer = 1; outer <= 256; ++outer) {
        const bool allowed = outer >= 4 && outer <= 128 && (outer & (outer - 1)) == 0;
        require(rejection(1, {16, outer}).empty() == allowed && rejection(2, {outer, 16}).empty() == allowed,
            "xor64 outer extent " + std::to_string(outer) + " was " + (allowed ? "rejected" : "accepted"));
        if (!allowed) {
            require(rejection(1, {16, outer}).find("outer extent") != std::string::npos,
                "xor64 outer extent " + std::to_string(outer) + " rejected with the wrong diagnostic");
        }
    }

    // The span is the compact object, 64 x 64 x 2 = 8192 bytes: at 61440 it
    // ends past the 64 KiB shmem.
    nanosm::TmaDescriptor big;
    big.rank = 2;
    big.element_bytes = 2;
    big.swizzle = 1;
    big.dimensions = {128, 128};
    big.row_stride = 128;
    big.box = {64, 64};
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.shmem_image, 33032);
    image_bytes(launch.gmem_image, 65536);
    write_tma_descriptor(shmem, 0, big);
    write_u64(shmem, 33024, nanosm::encode_mbar_state({false, true, 1, 0, 8192}));
    nanosm::Machine machine(canonical_profile(),
        compile("mov r0, 0\nmov r1, 0\n" + tma_operand_asm(2, 61440, 33024) + "tma.load [0], r0, r2\nhalt\n"),
        std::move(launch));
    try {
        machine.run(20000);
        throw std::runtime_error("oversized xor64 span was accepted");
    } catch (const nanosm::SimError& error) {
        require(std::string(error.what()).find("does not fit in shmem") != std::string::npos,
            "oversized xor64 span rejected with the wrong diagnostic");
    }
}

// Independent restatement of the 64-row map: logical K picks the 1024-byte
// slice (16 bytes x 64 rows) and the position inside it. Never calls the
// production helper.
uint32_t reference_wide_xor64(uint32_t base, uint8_t layout, uint32_t u, uint32_t v) {
    const uint32_t k = layout == 1 ? u : v;
    const uint32_t outer = layout == 1 ? v : u;
    const uint32_t phase = layout == 1 ? 0u : 2u;
    return base + 1024 * (k / 8) +
        4 * (4 * outer + (((k % 8) >> 1) ^ ((outer >> 2) & 1) ^ phase)) +
        2 * (k % 2);
}

// `commands` loads of a K / commands slab each, one descriptor, their
// address operands built in turn in r2: command i's box starts at
// tile + 1024 x (its K / 8) x i. The wait's mbar address is in ur3.
std::string xor64_slab_loads(uint32_t commands, uint32_t tile, uint32_t mbar, const std::string& step) {
    const uint32_t k_extent = 64 / commands;
    std::string source = "umov ur3, " + std::to_string(mbar) + "\n";
    for (uint32_t i = 0; i < commands; ++i) {
        if (i != 0) source += step;
        source += tma_operand_asm(2, tile + 1024 * (k_extent / 8) * i, mbar) + "tma.load [0], r0, r2\n";
    }
    return source + "wait:\nmbar.try_wait p0, [ur3], 0\n@!p0 bra.uni wait\nhalt\n";
}

void test_tma_wide_xor64_matches_narrow() {
    constexpr uint32_t mbar_address = 576;
    constexpr uint32_t tile_address = 1024;
    constexpr uint32_t stride = 83;
    constexpr uint32_t dim_x = 80;
    constexpr uint32_t dim_y = 70;
    struct Case {
        uint8_t layout;
        bool transpose;
        char step_axis; // global axis the K origin advances along, per k8 chunk
    };
    for (const auto& test_case : {Case{1, false, 'x'}, Case{2, false, 'y'}, Case{2, true, 'x'}}) {
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
            descriptor.rank = 2;
            descriptor.transpose = test_case.transpose;
            descriptor.element_bytes = 2;
            descriptor.swizzle = test_case.layout;
            descriptor.dimensions = {dim_x, dim_y};
            descriptor.row_stride = stride;
            descriptor.box = test_case.layout == 1
                ? std::array<uint32_t, 2>{k_extent, 64}
                : std::array<uint32_t, 2>{64, k_extent};
            write_tma_descriptor(shmem, 0, descriptor);
            write_u64(shmem, mbar_address,
                nanosm::encode_mbar_state({false, true, 1, 0, 8192}));
            const std::string source = "mov r0, 3\nmov r1, 2\n" + xor64_slab_loads(commands, tile_address, mbar_address,
                test_case.step_axis == 'x' ? "addi r0, r0, 8\n" : "addi r1, r1, 8\n");
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
        for (uint32_t v = 0; v < 64; ++v) {
            for (uint32_t u = 0; u < 64; ++u) {
                const auto address =
                    reference_wide_xor64(tile_address, test_case.layout, u, v);
                require(address >= tile_address && address < tile_address + 8192,
                    "wide xor64 address left the span");
                require(addresses.insert(address).second, "wide xor64 map is not bijective");
                const uint32_t gx = 3 + (test_case.transpose ? v : u);
                const uint32_t gy = 2 + (test_case.transpose ? u : v);
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
        descriptor.rank = 2;
        descriptor.transpose = true; // the QK^T orientation on the natural layout
        descriptor.element_bytes = 2;
        descriptor.swizzle = 2;
        descriptor.dimensions = {64, 64};
        descriptor.row_stride = 64;
        descriptor.box = {64, k_extent};
        write_tma_descriptor(shmem, 0, descriptor);
        write_u64(shmem, mbar_address, nanosm::encode_mbar_state({false, true, 1, 0, 8192}));
        const std::string source = "mov r0, 0\nmov r1, 0\n" +
            xor64_slab_loads(commands, tile_address, mbar_address, "addi r0, r0, 8\n");
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

// The pre-2026-10-03 fp16 operand formula, written out literally. The byte
// primitive must reproduce it for every MMA/TMA fp16 address in play.
void test_xor64_byte_primitive_matches_fp16_formula() {
    for (uint32_t base : {0u, 1024u, 40960u, 65472u}) {
        for (uint32_t phase : {0u, 2u}) {
            for (uint32_t outer = 0; outer < 64; ++outer) {
                for (uint32_t k = 0; k < 8; ++k) {
                    const uint32_t old = base +
                        4 * (4 * outer + ((k >> 1) ^ ((outer >> 2) & 1) ^ phase)) +
                        2 * (k & 1);
                    require(nanosm::xor64_operand_address(base, outer, k, phase) == old,
                        "xor64 operand wrapper diverged from the fp16 formula");
                    require(nanosm::xor64_byte_address(base, outer, 2 * k, phase) == old &&
                            nanosm::xor64_byte_address(base, outer, 2 * k + 1, phase) == old + 1,
                        "xor64 byte primitive diverged from the fp16 formula");
                }
            }
        }
    }
}

void test_tma_fp32_xor64_validation() {
    const auto rejection = [](uint8_t element_bytes, uint8_t swizzle, std::array<uint32_t, 2> box) {
        nanosm::TmaDescriptor descriptor;
        descriptor.rank = 2;
        descriptor.element_bytes = element_bytes;
        descriptor.swizzle = swizzle;
        descriptor.dimensions = {128, 128};
        descriptor.row_stride = 128;
        descriptor.box = box;
        return encode_error(descriptor);
    };
    // fp32: four elements per 16-byte row; K extent a multiple of 4, up to 64.
    require(rejection(4, 1, {64, 64}).empty(), "fp32 xor64_a 64x64 tile was rejected");
    require(rejection(4, 2, {64, 64}).empty(), "fp32 xor64_b 64x64 tile was rejected");
    require(rejection(4, 1, {4, 64}).empty() && rejection(4, 1, {12, 64}).empty(),
        "fp32 xor64 single/odd slice counts were rejected");
    require(rejection(4, 1, {64, 32}).empty() && rejection(4, 2, {32, 8}).empty(),
        "fp32 xor64 objects of other row counts were rejected");
    for (std::array<uint32_t, 2> box : {std::array<uint32_t, 2>{6, 64},
                                        std::array<uint32_t, 2>{2, 64},
                                        std::array<uint32_t, 2>{68, 64}}) {
        require(rejection(4, 1, box).find("K extent") != std::string::npos,
            "fp32 xor64 accepted a K extent that is not whole slices");
    }
    require(rejection(4, 1, {64, 48}).find("outer extent") != std::string::npos,
        "fp32 xor64 accepted 48 rows");
    require(rejection(4, 2, {64, 6}).find("K extent") != std::string::npos,
        "fp32 xor64_b accepted a K extent that is not whole slices");
    require(!rejection(8, 1, {64, 64}).empty(), "8-byte xor64 elements were accepted");
    require(rejection(1, 1, {64, 64}).find("requires rank 2") != std::string::npos,
        "1-byte xor64 elements were accepted");
    // fp16 keeps its eight-element rows: 12 is whole fp32 slices, not fp16.
    for (std::array<uint32_t, 2> box : {std::array<uint32_t, 2>{12, 64},
                                        std::array<uint32_t, 2>{4, 64},
                                        std::array<uint32_t, 2>{72, 64}}) {
        require(rejection(2, 1, box).find("K extent") != std::string::npos,
            "fp16 xor64 wrong K extent was accepted after the fp32 generalization");
    }
}

// Independent fp32 xor64_a address: slice = col / 4 (four floats per
// 16-byte row; 1024 bytes for 64 rows), row = the 16-byte row, word-in-row =
// (col % 4) XOR bit 2 of the row. Never calls the production primitive.
uint32_t reference_fp32_xor64_a(uint32_t base, uint32_t row, uint32_t col) {
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
            const auto address = reference_fp32_xor64_a(tile_address, row, col);
            require(address % 4 == 0 && address >= tile_address &&
                    address < tile_address + tile_bytes,
                "fp32 xor64 address left the 16-slice span");
            require(addresses.insert(address).second, "fp32 xor64 map is not bijective");
        }
    }
    require(addresses.size() == 4096, "fp32 xor64 map did not cover the 16 slices");

    const auto make_descriptor = [&](uint8_t swizzle) {
        nanosm::TmaDescriptor descriptor;
        descriptor.rank = 2;
        descriptor.element_bytes = 4;
        descriptor.swizzle = swizzle;
        descriptor.gmem_base = gmem_base;
        descriptor.dimensions = {64, 64};
        descriptor.row_stride = 64;
        descriptor.box = {64, 64};
        return descriptor;
    };
    const auto program = [&](const std::string& opcode) {
        return "mov r0, 0\nmov r1, 0\nmov r2, " + std::to_string(nanosm::pack_tma_addresses(tile_address, mbar_address)) +
            "\numov ur3, 128\n" + opcode + " [0], r0, r2\nwait:\nmbar.try_wait p0, [ur3], 0\n@!p0 bra.uni wait\nhalt\n";
    };
    const nanosm::MbarState mbar{false, true, 1, 0, tile_bytes};

    // STORE: shmem tile authored at the independent address -> row-major gmem.
    const auto run_store = [&](uint8_t swizzle) {
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.shmem_image, tile_address + tile_bytes);
        auto& gmem = image_bytes(launch.gmem_image, gmem_size);
        std::fill(gmem.begin(), gmem.end(), 0x5a);
        for (uint32_t row = 0; row < 64; ++row)
            for (uint32_t col = 0; col < 64; ++col)
                write_u32(shmem, swizzle == 1
                        ? reference_fp32_xor64_a(tile_address, row, col)
                        : tile_address + 4 * (64 * row + col),
                    nanosm::float_bits(fp32_tile_value(row, col)));
        write_tma_descriptor(shmem, 0, make_descriptor(swizzle));
        write_u64(shmem, mbar_address, nanosm::encode_mbar_state(mbar));
        nanosm::Machine machine(canonical_profile(), compile(program("tma.store")), std::move(launch));
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
    const auto swizzled_store = run_store(1);
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
    const auto linear_store = run_store(0);
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
    write_tma_descriptor(shmem, 0, make_descriptor(1));
    write_u64(shmem, mbar_address, nanosm::encode_mbar_state(mbar));
    nanosm::Machine machine(canonical_profile(), compile(program("tma.load")), std::move(launch));
    machine.run(200000);
    for (uint32_t row = 0; row < 64; ++row) {
        for (uint32_t col = 0; col < 64; ++col) {
            require(machine.read_shmem_u32(reference_fp32_xor64_a(tile_address, row, col)) ==
                    nanosm::float_bits(fp32_tile_value(row, col)),
                "fp32 xor64 TMA load mapped the wrong element at row " +
                    std::to_string(row) + " col " + std::to_string(col));
        }
    }
    const auto& load_summary = machine.trace().summary();
    require(load_summary.tma_port_requests == 512 && load_summary.tma_port_rounds == 1024,
        "fp32 xor64 TMA load did not write two port-A rounds per sector");
}

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
    descriptor.gmem_base = 0;
    descriptor.dimensions = {4, 1};
    descriptor.box = {8, 1};
    write_tma_descriptor(shmem, descriptor_address, descriptor);
    nanosm::MbarState mbar;
    mbar.valid = true;
    mbar.expected = 1;
    mbar.tx_bytes = 32;
    write_u64(shmem, mbar_address, nanosm::encode_mbar_state(mbar));

    nanosm::Machine machine(canonical_profile(), compile(R"(
        mov r0, -4
        mov r1, 0
        mov r2, )" + std::to_string(nanosm::pack_tma_addresses(tile_address, mbar_address)) + R"(
        umov ur3, 128
        tma.load [0], r0, r2
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

    nanosm::TmaDescriptor descriptor;
    descriptor.gmem_base = 12;
    descriptor.dimensions = {4, 1};
    descriptor.box = {4, 1};
    write_tma_descriptor(shmem, descriptor_address, descriptor);
    nanosm::MbarState mbar;
    mbar.valid = true;
    mbar.expected = 1;
    mbar.tx_bytes = 16;
    write_u64(shmem, mbar_address, nanosm::encode_mbar_state(mbar));

    nanosm::Machine machine(canonical_profile(), compile(R"(
        mov r0, 0
        mov r1, 0
        mov r2, )" + std::to_string(nanosm::pack_tma_addresses(tile_address, mbar_address)) + R"(
        umov ur3, 128
        tma.store [64], r0, r2
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

// Rank 2 with a row stride and an origin, and a third dimension walked in
// software (this was the rank-3 strides test; a z coordinate no longer fits
// the instruction's registers). A 3 x 4 x 2 fp32 tensor with row stride 5
// and plane stride 30 = 6 rows is viewed as rank 2 (3 x 10, row stride 5):
// plane z is rows 6z .. 6z + 3. A loop issues one 2 x 2 box per plane at
// (1, 1 + 6z), each into its own box 64 bytes on (the address operand's
// shared field + 1), both completing into one mbar.
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
    descriptor.rank = 2;
    descriptor.dimensions = {3, 10};
    descriptor.row_stride = 5;
    descriptor.box = {2, 2};
    write_tma_descriptor(shmem, descriptor_address, descriptor);
    write_tx_mbar(shmem, mbar_address, 32);

    nanosm::Machine machine(canonical_profile(), compile(R"(
        mov r0, 1
        mov r1, 1
        mov r2, )" + std::to_string(nanosm::pack_tma_addresses(tile_address, mbar_address)) + R"(
        umov ur3, 128
        mov r4, 0
    plane:
        tma.load [0], r0, r2
        addi r1, r1, 6
        addi r2, r2, 65536
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

// tma descriptor encoding
TEST(Tma, tma_descriptor_encoding) {
    test_tma_descriptor_encoding();
}

// tma address operand: packing, reach, range checks
TEST(Tma, tma_address_operand) {
    test_tma_address_operand();
}

// tma direction is the opcode; one descriptor serves both
TEST(Tma, tma_direction_is_the_opcode) {
    test_tma_direction_is_the_opcode();
}

// tma xor64 transpose load/store
TEST(Tma, tma_xor64_transpose_load_store) {
    test_tma_xor64_transpose_load_store();
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

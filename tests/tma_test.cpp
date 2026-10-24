// TMA descriptors, xor64 placement, ranks, out-of-bounds handling, the profile stamp.
// Moved from the original tests/tests.cpp (trash/test-setup-2026-10-22.md);
// each test's body is unchanged, its TEST wrapper is at the end.
#include "test_support.hpp"

namespace {

using namespace nanosm_test;

void test_tma_descriptor_encoding() {
    nanosm::TmaDescriptor descriptor;
    descriptor.rank = 3;
    descriptor.direction = nanosm::TmaDirection::store;
    descriptor.gmem_base = 0x1122334455667788ull;
    descriptor.dimensions = {17, 19, 23};
    descriptor.strides = {32, 1024};
    descriptor.box = {4, 5, 6};
    descriptor.element_bytes = 2;
    descriptor.shmem_base = 0x200;
    descriptor.completion_mbar = 0x180;
    const auto bytes = nanosm::encode_tma_descriptor(descriptor);
    require(bytes[0] == 3 && bytes[1] == 1 && bytes[3] == 2, "tma header bytes changed");
    require(bytes[8] == 0x88 && bytes[15] == 0x11, "tma gmem base is not little-endian");
    const auto decoded = nanosm::decode_tma_descriptor(bytes.data(), bytes.size());
    require(
        decoded.rank == descriptor.rank && decoded.direction == descriptor.direction &&
            decoded.gmem_base == descriptor.gmem_base &&
            decoded.dimensions == descriptor.dimensions && decoded.strides == descriptor.strides &&
            decoded.box == descriptor.box && decoded.element_bytes == descriptor.element_bytes &&
            decoded.shmem_base == descriptor.shmem_base &&
            decoded.completion_mbar == descriptor.completion_mbar,
        "tma descriptor did not round trip");

    auto corrupt = bytes;
    corrupt[56] = 1;
    try {
        (void)nanosm::decode_tma_descriptor(corrupt.data(), corrupt.size());
        throw std::runtime_error("tma decoder accepted a nonzero reserved byte");
    } catch (const nanosm::SimError& error) {
        require(
            std::string(error.what()).find("reserved tail") != std::string::npos,
            "reserved tma byte produced the wrong panic");
    }

    descriptor.transpose = true;
    try {
        (void)nanosm::encode_tma_descriptor(descriptor);
        throw std::runtime_error("v0 tma descriptor accepted transpose");
    } catch (const nanosm::SimError& error) {
        require(
            std::string(error.what()).find("transpose") != std::string::npos,
            "unsupported tma transpose produced the wrong panic");
    }
    descriptor.transpose = false;
    descriptor.swizzle = 3;
    try {
        (void)nanosm::encode_tma_descriptor(descriptor);
        throw std::runtime_error("v0 tma descriptor accepted swizzle");
    } catch (const nanosm::SimError& error) {
        require(
            std::string(error.what()).find("swizzle") != std::string::npos,
            "unsupported tma swizzle produced the wrong panic");
    }

    for (uint8_t layout : {uint8_t{1}, uint8_t{2}}) for (bool transpose : {false, true}) {
        nanosm::TmaDescriptor operand;
        operand.rank = 2;
        operand.transpose = transpose;
        operand.element_bytes = 2;
        operand.swizzle = layout;
        operand.dimensions = {96, 96, 1};
        operand.strides = {101, 0};
        operand.box = layout == 1
            ? std::array<uint32_t, 3>{8, 64, 1}
            : std::array<uint32_t, 3>{64, 8, 1};
        operand.shmem_base = 1024;
        operand.completion_mbar = 128;
        const auto encoded = nanosm::encode_tma_descriptor(operand);
        const auto round_trip = nanosm::decode_tma_descriptor(encoded.data(), encoded.size());
        require(round_trip.swizzle == layout && round_trip.transpose == transpose &&
                round_trip.box == operand.box,
            "xor64 descriptor layout/transpose did not round trip");
    }

    nanosm::TmaDescriptor invalid;
    invalid.rank = 2; invalid.element_bytes = 2; invalid.swizzle = 1;
    invalid.dimensions = {64, 64, 1}; invalid.strides = {64, 0};
    invalid.box = {8, 63, 1}; invalid.shmem_base = 1024;
    try {
        (void)nanosm::encode_tma_descriptor(invalid);
        throw std::runtime_error("xor64 descriptor accepted the wrong geometry");
    } catch (const nanosm::SimError& error) {
        require(std::string(error.what()).find("geometry") != std::string::npos,
            "wrong xor64 geometry produced the wrong panic");
    }
    invalid.box = {8, 64, 1}; invalid.shmem_base = 1026;
    try {
        (void)nanosm::encode_tma_descriptor(invalid);
        throw std::runtime_error("xor64 descriptor accepted an unaligned object");
    } catch (const nanosm::SimError& error) {
        require(std::string(error.what()).find("64-byte") != std::string::npos,
            "unaligned xor64 object produced the wrong panic");
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

void test_tma_xor64_transpose_load_store() {
    constexpr uint32_t mbar_address = 128;
    constexpr uint32_t tile_address = 1024;
    constexpr uint32_t stride = 83;
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
            descriptor.dimensions = {dim_x, dim_y, 1};
            descriptor.strides = {stride, 0};
            descriptor.box = {box_x, box_y, 1};
            descriptor.shmem_base = tile_address;
            descriptor.completion_mbar = mbar_address;

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
            nanosm::Machine machine(canonical_profile(), compile(R"(
                mov r0, 3
                mov r1, 2
                mov r2, 0
                mov r3, 128
                tma.async [0], r0
            wait:
                mbar.try_wait p0, [r3], 0
                @!p0 bra.uni wait
                halt
            )"), std::move(launch));
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

            descriptor.direction = nanosm::TmaDirection::store;
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
            nanosm::Machine store_machine(canonical_profile(), compile(R"(
                mov r0, 3
                mov r1, 2
                mov r2, 0
                mov r3, 128
                tma.async [0], r0
            wait:
                mbar.try_wait p0, [r3], 0
                @!p0 bra.uni wait
                halt
            )"), std::move(store_launch));
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
        for (uint32_t chunk = 0; chunk < 2; ++chunk) {
            nanosm::TmaDescriptor a;
            a.rank = 2; a.element_bytes = 2; a.swizzle = 1;
            a.gmem_base = base_a; a.dimensions = {16, 64, 1};
            a.strides = {stride_a, 0}; a.box = {8, 64, 1};
            a.shmem_base = 1024 + 2048 * chunk; a.completion_mbar = 512 + 16 * chunk;
            nanosm::TmaDescriptor b;
            b.rank = 2; b.element_bytes = 2; b.swizzle = 2; b.gmem_base = base_b;
            b.box = {64, 8, 1}; b.shmem_base = 2048 + 2048 * chunk;
            b.completion_mbar = 520 + 16 * chunk;
            if (!pv) {
                b.transpose = true; b.dimensions = {16, 64, 1}; b.strides = {stride_b_qk, 0};
            } else {
                b.dimensions = {64, 16, 1}; b.strides = {stride_b_pv, 0};
            }
            write_tma_descriptor(shmem, 64 * (2 * chunk), a);
            write_tma_descriptor(shmem, 64 * (2 * chunk + 1), b);
            nanosm::MbarState ready{false, true, 1, 0, 1024};
            write_u64(shmem, a.completion_mbar, nanosm::encode_mbar_state(ready));
            write_u64(shmem, b.completion_mbar, nanosm::encode_mbar_state(ready));
        }
        nanosm::MbarState commit{false, true, 1, 1, 0};
        write_u64(shmem, 544, nanosm::encode_mbar_state(commit));
        // Chunk 1 origins per rfc07's table: A advances k in global x for both
        // orientations; B advances k in x under QK^T (transpose=1) but in y
        // under PV (transpose=0) ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â V's rows are keys.
        const std::string chunk1_b = pv
            ? "mov r0, 0\nmov r1, 8\ntma.async [192], r0\n"
            : "tma.async [192], r0\n";
        const auto kernel = compile(std::string(R"(
            .acc C, 0, 64, 64
            mov r0, 0
            mov r1, 0
            mov r2, 0
            tma.async [0], r0
            tma.async [64], r0
            mov r0, 8
            tma.async [128], r0
)") + chunk1_b + R"(
            mov r4, 512
        wait0:
            mbar.try_wait p0, [r4], 0
            @!p0 bra.uni wait0
            mov r4, 520
        wait1:
            mbar.try_wait p0, [r4], 0
            @!p0 bra.uni wait1
            mov r4, 528
        wait2:
            mbar.try_wait p0, [r4], 0
            @!p0 bra.uni wait2
            mov r4, 536
        wait3:
            mbar.try_wait p0, [r4], 0
            @!p0 bra.uni wait3
            mov r10, 1024
            mov r11, 2048
            mma.async.init.m64n64k8 C, [r10], [r11]
            mov r10, 3072
            mov r11, 4096
            mma.async.m64n64k8 C, [r10], [r11]
            mov r4, 544
            mma.commit [r4]
        wait_commit:
            mbar.try_wait p0, [r4], 0
            @!p0 bra.uni wait_commit
            mov r8, 0
            mov r9, 0
            acc.ld.v8 r20, C, r8
            mov r12, 48
            mov r13, 56
            acc.ld.v8 r28, C, r12
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

// Independent restatement of the stage-granular map: logical K picks the
// 1024-byte operand object and the position inside it. Never calls the
// production helper.
uint32_t reference_wide_xor64(uint32_t base, uint8_t layout, uint32_t u, uint32_t v) {
    const uint32_t k = layout == 1 ? u : v;
    const uint32_t outer = layout == 1 ? v : u;
    const uint32_t phase = layout == 1 ? 0u : 2u;
    return base + 1024 * (k / 8) +
        4 * (4 * outer + (((k % 8) >> 1) ^ ((outer >> 2) & 1) ^ phase)) +
        2 * (k % 2);
}

void test_tma_wide_xor64_validation() {
    nanosm::TmaDescriptor descriptor;
    descriptor.rank = 2;
    descriptor.element_bytes = 2;
    descriptor.dimensions = {128, 128, 1};
    descriptor.strides = {128, 0};
    descriptor.shmem_base = 1024;
    const auto expect = [&](uint8_t swizzle, std::array<uint32_t, 3> box, bool ok) {
        descriptor.swizzle = swizzle;
        descriptor.box = box;
        bool accepted = true;
        try {
            (void)nanosm::encode_tma_descriptor(descriptor);
        } catch (const nanosm::SimError&) {
            accepted = false;
        }
        require(accepted == ok, "wide xor64 box validation disagreed with the contract");
    };
    expect(1, {16, 64, 1}, true);
    expect(1, {64, 64, 1}, true);
    expect(2, {64, 16, 1}, true);
    expect(2, {64, 64, 1}, true);
    expect(1, {12, 64, 1}, false);
    expect(1, {72, 64, 1}, false);
    expect(1, {16, 32, 1}, false);
    expect(2, {32, 16, 1}, false);

    nanosm::TmaDescriptor big;
    big.rank = 2;
    big.element_bytes = 2;
    big.swizzle = 1;
    big.dimensions = {128, 128, 1};
    big.strides = {128, 0};
    big.box = {64, 64, 1};
    big.shmem_base = 61440; // 8192-byte span ends past the 64 KiB shmem
    big.completion_mbar = 33024;
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.shmem_image, 33032);
    image_bytes(launch.gmem_image, 65536);
    write_tma_descriptor(shmem, 0, big);
    write_u64(shmem, 33024, nanosm::encode_mbar_state({false, true, 1, 0, 8192}));
    nanosm::Machine machine(canonical_profile(),
        compile("mov r0, 0\nmov r1, 0\nmov r2, 0\ntma.async [0], r0\nhalt\n"),
        std::move(launch));
    try {
        machine.run(20000);
        throw std::runtime_error("oversized xor64 span was accepted");
    } catch (const nanosm::SimError& error) {
        require(std::string(error.what()).find("does not fit in shmem") != std::string::npos,
            "oversized xor64 span rejected with the wrong diagnostic");
    }
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
            for (uint32_t i = 0; i < commands; ++i) {
                nanosm::TmaDescriptor descriptor;
                descriptor.rank = 2;
                descriptor.transpose = test_case.transpose;
                descriptor.element_bytes = 2;
                descriptor.swizzle = test_case.layout;
                descriptor.dimensions = {dim_x, dim_y, 1};
                descriptor.strides = {stride, 0};
                descriptor.box = test_case.layout == 1
                    ? std::array<uint32_t, 3>{k_extent, 64, 1}
                    : std::array<uint32_t, 3>{64, k_extent, 1};
                descriptor.shmem_base = tile_address + 1024 * (k_extent / 8) * i;
                descriptor.completion_mbar = mbar_address;
                write_tma_descriptor(shmem, 64 * i, descriptor);
            }
            write_u64(shmem, mbar_address,
                nanosm::encode_mbar_state({false, true, 1, 0, 8192}));
            std::string source = "mov r0, 3\nmov r1, 2\nmov r2, 0\nmov r3, " +
                std::to_string(mbar_address) + "\n";
            for (uint32_t i = 0; i < commands; ++i) {
                if (i != 0) {
                    source += test_case.step_axis == 'x'
                        ? "addi r0, r0, 8\n" : "addi r1, r1, 8\n";
                }
                source += "tma.async [" + std::to_string(64 * i) + "], r0\n";
            }
            source += "wait:\nmbar.try_wait p0, [r3], 0\n@!p0 bra.uni wait\nhalt\n";
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
        for (uint32_t i = 0; i < commands; ++i) {
            nanosm::TmaDescriptor descriptor;
            descriptor.rank = 2;
            descriptor.transpose = true; // the QK^T orientation on the natural layout
            descriptor.element_bytes = 2;
            descriptor.swizzle = 2;
            descriptor.dimensions = {64, 64, 1};
            descriptor.strides = {64, 0};
            descriptor.box = {64, k_extent, 1};
            descriptor.shmem_base = tile_address + 1024 * (k_extent / 8) * i;
            descriptor.completion_mbar = mbar_address;
            write_tma_descriptor(shmem, 64 * i, descriptor);
        }
        write_u64(shmem, mbar_address, nanosm::encode_mbar_state({false, true, 1, 0, 8192}));
        std::string source = "mov r0, 0\nmov r1, 0\nmov r2, 0\nmov r3, " +
            std::to_string(mbar_address) + "\n";
        for (uint32_t i = 0; i < commands; ++i) {
            if (i != 0) source += "addi r0, r0, 8\n";
            source += "tma.async [" + std::to_string(64 * i) + "], r0\n";
        }
        source += "wait:\nmbar.try_wait p0, [r3], 0\n@!p0 bra.uni wait\nhalt\n";
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
    const auto accepted = [](uint8_t element_bytes, uint8_t swizzle,
                             std::array<uint32_t, 3> box, nanosm::TmaDirection direction) {
        nanosm::TmaDescriptor descriptor;
        descriptor.rank = 2;
        descriptor.direction = direction;
        descriptor.element_bytes = element_bytes;
        descriptor.swizzle = swizzle;
        descriptor.dimensions = {128, 128, 1};
        descriptor.strides = {128, 0};
        descriptor.box = box;
        descriptor.shmem_base = 1024;
        descriptor.completion_mbar = 128;
        try {
            (void)nanosm::encode_tma_descriptor(descriptor);
            return std::string();
        } catch (const nanosm::SimError& error) {
            return std::string(error.what());
        }
    };
    const auto load = nanosm::TmaDirection::load;
    const auto store = nanosm::TmaDirection::store;
    // fp32: four elements per 16-byte row; K extent a multiple of 4, up to 64.
    for (auto direction : {load, store}) {
        require(accepted(4, 1, {64, 64, 1}, direction).empty(),
            "fp32 xor64_a 64x64 tile was rejected");
        require(accepted(4, 2, {64, 64, 1}, direction).empty(),
            "fp32 xor64_b 64x64 tile was rejected");
        require(accepted(4, 1, {4, 64, 1}, direction).empty() &&
                accepted(4, 1, {12, 64, 1}, direction).empty(),
            "fp32 xor64 single/odd object counts were rejected");
    }
    for (std::array<uint32_t, 3> box : {std::array<uint32_t, 3>{6, 64, 1},
                                        std::array<uint32_t, 3>{2, 64, 1},
                                        std::array<uint32_t, 3>{68, 64, 1},
                                        std::array<uint32_t, 3>{64, 32, 1}}) {
        require(accepted(4, 1, box, store).find("geometry") != std::string::npos,
            "fp32 xor64 accepted a K extent that is not whole objects");
    }
    require(accepted(4, 2, {64, 6, 1}, load).find("geometry") != std::string::npos,
        "fp32 xor64_b accepted a K extent that is not whole objects");
    require(!accepted(8, 1, {64, 64, 1}, store).empty(), "8-byte xor64 elements were accepted");
    require(accepted(1, 1, {64, 64, 1}, store).find("requires rank 2") != std::string::npos,
        "1-byte xor64 elements were accepted");
    // fp16 keeps its eight-element rows: 12 is whole fp32 objects, not fp16.
    for (std::array<uint32_t, 3> box : {std::array<uint32_t, 3>{12, 64, 1},
                                        std::array<uint32_t, 3>{4, 64, 1},
                                        std::array<uint32_t, 3>{72, 64, 1},
                                        std::array<uint32_t, 3>{16, 32, 1}}) {
        require(accepted(2, 1, box, load).find("geometry") != std::string::npos,
            "fp16 xor64 wrong geometry was accepted after the fp32 generalization");
    }
}

// Independent fp32 xor64_a address: object = col / 4 (four floats per
// 16-byte row), row = the 16-byte row, word-in-row = (col % 4) XOR bit 2 of
// the row. Never calls the production primitive.
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
                "fp32 xor64 address left the 16-object span");
            require(addresses.insert(address).second, "fp32 xor64 map is not bijective");
        }
    }
    require(addresses.size() == 4096, "fp32 xor64 map did not cover the 16 objects");

    const auto make_descriptor = [&](nanosm::TmaDirection direction, uint8_t swizzle) {
        nanosm::TmaDescriptor descriptor;
        descriptor.rank = 2;
        descriptor.direction = direction;
        descriptor.element_bytes = 4;
        descriptor.swizzle = swizzle;
        descriptor.gmem_base = gmem_base;
        descriptor.dimensions = {64, 64, 1};
        descriptor.strides = {64, 0};
        descriptor.box = {64, 64, 1};
        descriptor.shmem_base = tile_address;
        descriptor.completion_mbar = mbar_address;
        return descriptor;
    };
    const char* program = R"(
        mov r0, 0
        mov r1, 0
        mov r2, 0
        mov r3, 128
        tma.async [0], r0
    wait:
        mbar.try_wait p0, [r3], 0
        @!p0 bra.uni wait
        halt
    )";
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
        write_tma_descriptor(shmem, 0, make_descriptor(nanosm::TmaDirection::store, swizzle));
        write_u64(shmem, mbar_address, nanosm::encode_mbar_state(mbar));
        nanosm::Machine machine(canonical_profile(), compile(program), std::move(launch));
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
    // A 32-byte sector is eight consecutive fp32 of one row: four in object
    // col/4 and four in the next. Both halves sit in the same row of their
    // objects, so on the same four banks (1024 B = 256 words = 0 mod 16) at
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
    write_tma_descriptor(shmem, 0, make_descriptor(nanosm::TmaDirection::load, 1));
    write_u64(shmem, mbar_address, nanosm::encode_mbar_state(mbar));
    nanosm::Machine machine(canonical_profile(), compile(program), std::move(launch));
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
    descriptor.dimensions = {4, 1, 1};
    descriptor.box = {8, 1, 1};
    descriptor.shmem_base = tile_address;
    descriptor.completion_mbar = mbar_address;
    write_tma_descriptor(shmem, descriptor_address, descriptor);
    nanosm::MbarState mbar;
    mbar.valid = true;
    mbar.expected = 1;
    mbar.tx_bytes = 32;
    write_u64(shmem, mbar_address, nanosm::encode_mbar_state(mbar));

    nanosm::Machine machine(canonical_profile(), compile(R"(
        mov r0, -4
        mov r1, 0
        mov r2, 0
        mov r3, 128
        tma.async [0], r0
    wait:
        mbar.try_wait p0, [r3], 0
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
    descriptor.direction = nanosm::TmaDirection::store;
    descriptor.gmem_base = 12;
    descriptor.dimensions = {4, 1, 1};
    descriptor.box = {4, 1, 1};
    descriptor.shmem_base = tile_address;
    descriptor.completion_mbar = mbar_address;
    write_tma_descriptor(shmem, descriptor_address, descriptor);
    nanosm::MbarState mbar;
    mbar.valid = true;
    mbar.expected = 1;
    mbar.tx_bytes = 16;
    write_u64(shmem, mbar_address, nanosm::encode_mbar_state(mbar));

    nanosm::Machine machine(canonical_profile(), compile(R"(
        mov r0, 0
        mov r1, 0
        mov r2, 0
        mov r3, 128
        tma.async [64], r0
    wait:
        mbar.try_wait p0, [r3], 0
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

void test_tma_rank3_strides() {
    constexpr uint32_t descriptor_address = 0;
    constexpr uint32_t mbar_address = 128;
    constexpr uint32_t tile_address = 256;
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.shmem_image, tile_address + 16);
    auto& gmem = image_bytes(launch.gmem_image, 64 * 4);
    for (uint32_t i = 0; i < 64; ++i) write_u32(gmem, i * 4, 1000 + i);

    nanosm::TmaDescriptor descriptor;
    descriptor.rank = 3;
    descriptor.dimensions = {3, 4, 2};
    descriptor.strides = {5, 30};
    descriptor.box = {2, 2, 1};
    descriptor.shmem_base = tile_address;
    descriptor.completion_mbar = mbar_address;
    write_tma_descriptor(shmem, descriptor_address, descriptor);
    nanosm::MbarState mbar;
    mbar.valid = true;
    mbar.expected = 1;
    mbar.tx_bytes = 16;
    write_u64(shmem, mbar_address, nanosm::encode_mbar_state(mbar));

    nanosm::Machine machine(canonical_profile(), compile(R"(
        mov r0, 1
        mov r1, 1
        mov r2, 1
        mov r3, 128
        tma.async [0], r0
    wait:
        mbar.try_wait p0, [r3], 0
        @!p0 bra.uni wait
        halt
    )"), std::move(launch));
    machine.run();
    const std::array<uint32_t, 4> expected{1036, 1037, 1041, 1042};
    for (uint32_t i = 0; i < expected.size(); ++i) {
        require(
            machine.read_shmem_u32(tile_address + i * 4) == expected[i],
            "tma rank-3 stride mapping is wrong");
    }
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
        mov r0, 0
        ldg.uni r1, [r0]
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

// tma xor64 transpose load/store
TEST(Tma, tma_xor64_transpose_load_store) {
    test_tma_xor64_transpose_load_store();
}

// tma xor64 feeds MMA
TEST(TmaSlow, tma_xor64_feeds_mma_orientations) {
    test_tma_xor64_feeds_mma_orientations();
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

// tma rank-3 strides
TEST(Tma, tma_rank3_strides) {
    test_tma_rank3_strides();
}

// profile stamp
TEST(Tma, profile_stamp_rejects_mismatch) {
    test_profile_stamp_rejects_mismatch();
}

// DRAM clock domain
TEST(Tma, dram_uses_its_own_clock) {
    test_dram_uses_its_own_clock();
}

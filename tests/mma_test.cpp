// The tensor core and tmem: numerics, the walker, the mbar engines' events,
// shapes from the profile, tmem transfers beside matmuls, tmem pointers.
#include "test_support.hpp"

using namespace nanosm_test;

namespace {

// The byte address of element (row, k) of a swizzle64 operand object at shmem
// `base` whose rows are `spacing` bytes apart (reference_swizzle64): A is m
// rows of k, B is n rows of k.
uint32_t operand_address(uint32_t base, uint32_t row, uint32_t k, uint32_t spacing = 16) {
    return reference_swizzle64(base, spacing, row, 2 * k);
}

using ShmemFill = std::function<void(std::vector<uint8_t>&)>;

// A machine under a debug trace running `kernel` on `warps` warps, its 8 KiB
// shmem image written by `fill` (zero without one).
std::unique_ptr<nanosm::Machine> mma_machine(
    const nanosm::DeviceProfile& profile, const nanosm::Kernel& kernel, uint32_t warps, const ShmemFill& fill) {
    auto launch = launch_warps(warps);
    auto& shmem = image_bytes(launch.test_shmem, 8192);
    if (fill) fill(shmem);
    return std::make_unique<nanosm::Machine>(profile, kernel, std::move(launch),
        nanosm::TraceOptions{.timeline = true, .debug = true});
}

std::array<uint16_t, 8> filled(uint16_t value) {
    std::array<uint16_t, 8> values{};
    values.fill(value);
    return values;
}

// A 64x64x8 swizzle64 tile pair: A at 0 with a[k] in every row, B at 1024
// with b[k] in every column.
ShmemFill tiles(const std::array<uint16_t, 8>& a, const std::array<uint16_t, 8>& b) {
    return [=](std::vector<uint8_t>& shmem) {
        for (uint32_t row = 0; row < 64; ++row) {
            for (uint32_t k = 0; k < 8; ++k) {
                write_u16(shmem, operand_address(0, row, k), a[k]);
                write_u16(shmem, operand_address(1024, row, k), b[k]);
            }
        }
    };
}

// The same in the dense layout (row-major, compact in the shape): A, m rows
// of 8 at 0; B, 8 rows of n at 1024.
ShmemFill dense_tiles(uint32_t m, uint32_t n, const std::array<uint16_t, 8>& a, const std::array<uint16_t, 8>& b) {
    return [=](std::vector<uint8_t>& shmem) {
        for (uint32_t k = 0; k < 8; ++k) {
            for (uint32_t row = 0; row < m; ++row) write_u16(shmem, 2 * (row * 8 + k), a[k]);
            for (uint32_t col = 0; col < n; ++col) write_u16(shmem, 1024 + 2 * (k * n + col), b[k]);
        }
    };
}

// A(m, k) and B(k, n) as small integers, so every dot product is exact in FP32.
uint32_t small_a(uint32_t m, uint32_t k) { return (m + 3 * k) % 5; }

uint32_t small_b(uint32_t k, uint32_t n) { return (2 * k + n) % 3; }

uint16_t small_fp16(uint32_t value) {
    static constexpr std::array<uint16_t, 5> bits{0x0000, 0x3c00, 0x4000, 0x4200, 0x4400}; // 0..4
    return bits.at(value);
}

float product_reference(uint32_t m, uint32_t n, uint32_t depth = 8) {
    float sum = 0.0f;
    for (uint32_t k = 0; k < depth; ++k) sum += static_cast<float>(small_a(m, k) * small_b(k, n));
    return sum;
}

// A 64-row A object at 0 and a 64-column B object at 2048, from small_a and
// small_b.
void write_small_operands(std::vector<uint8_t>& shmem) {
    for (uint32_t row = 0; row < 64; ++row) {
        for (uint32_t k = 0; k < 8; ++k) {
            write_u16(shmem, operand_address(0, row, k), small_fp16(small_a(row, k)));
            write_u16(shmem, operand_address(2048, row, k), small_fp16(small_b(k, row)));
        }
    }
}

// Element (row, col) of the tmem region at `pointer` whose rows are
// `row_words` words (src/layout.hpp; tmem_layout_matches_the_rule holds the
// layout against the rule).
uint32_t result_word(const nanosm::Machine& machine, uint32_t pointer, uint32_t row_words, uint32_t row, uint32_t col) {
    return machine.tmem_word(static_cast<uint32_t>(nanosm::tmem_physical_word(pointer + row * row_words + col, row_words)));
}

float result_at(const nanosm::Machine& machine, uint32_t pointer, uint32_t row_words, uint32_t row, uint32_t col) {
    return nanosm::bits_float(result_word(machine, pointer, row_words, row, col));
}

// The operand rows of a shape with k fp16 are whole and max(16, 2k) bytes
// apart (the smallest swizzle64 spacing that holds them; the command names
// it), so a shape with k > 8 reads consecutive 16-byte pieces of the same rows.
uint32_t row_spacing(uint32_t k) {
    return std::max(16u, 2 * k);
}

// One mma.async.init of `shape` on `profile` (which lists it): A at 0, B at
// the next 1 KiB boundary, the result at tmem 0, every element checked
// against the reference. The machine is returned for its trace.
std::unique_ptr<nanosm::Machine> run_shape(const nanosm::DeviceProfile& profile, const nanosm::MmaShape& shape) {
    const auto name = nanosm::mma_shape_name(shape);
    const uint32_t spacing = row_spacing(shape.k);
    const uint32_t b_base = (shape.m * spacing + 1023) / 1024 * 1024;
    auto machine = mma_machine(profile,
        compile(mma_kernel({.shape = name, .b = b_base, .a_spacing = spacing, .b_spacing = spacing}), profile), 1,
        [&](std::vector<uint8_t>& shmem) {
            for (uint32_t k = 0; k < shape.k; ++k) {
                for (uint32_t m = 0; m < shape.m; ++m)
                    write_u16(shmem, operand_address(0, m, k, spacing), small_fp16(small_a(m, k)));
                for (uint32_t n = 0; n < shape.n; ++n)
                    write_u16(shmem, operand_address(b_base, n, k, spacing), small_fp16(small_b(k, n)));
            }
        });
    machine->run(200000);
    for (uint32_t row = 0; row < shape.m; ++row)
        for (uint32_t col = 0; col < shape.n; ++col)
            require(result_at(*machine, 0, shape.n, row, col) == product_reference(row, col, shape.k),
                name + ": element (" + std::to_string(row) + ", " + std::to_string(col) + ")");
    return machine;
}

nanosm::DeviceProfile shape_profile(std::vector<nanosm::MmaShape> shapes, uint32_t result_latency = 8) {
    auto profile = canonical_profile();
    profile.mma_shapes = std::move(shapes);
    profile.mma_result_latency_cycles = result_latency;
    nanosm::validate_device_profile(profile);
    return profile;
}

const std::string acc_ld_r10 = "acc.ld r10, [ur2, 64]\n";

} // namespace

// acc.st.v8 / acc.ld.v8 through qtxc, the hex file and four warps: qtxc does
// not hold independent arithmetic behind an acc.st; the hex carries no
// accumulator table; every warp's eight registers come back bit for bit, a
// NaN's payload too.
TEST(Mma, tmem_profile_metadata_and_round_trip) {
    const auto& profile = canonical_profile();
    const auto scheduled = qtx_program(R"(
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
    )", profile); // warp w: rows 16 w .. 16 w + 15 of 64-word rows, pointer 1024 w
    const auto add = std::find_if(scheduled.instructions.begin(), scheduled.instructions.end(),
        [](const auto& instruction) { return instruction.opcode == nanosm::Opcode::add; });
    require(add != scheduled.instructions.end() && add->control.wait_mask == 0,
        "independent arithmetic was serialized behind acc.st");
    const auto kernel = nanosm::assemble_asm(nanosm::emit_asm(scheduled), profile);
    const auto hex = nanosm::emit_hex(kernel);
    const auto loaded = nanosm::load_hex(hex);
    require(loaded.words == kernel.words && hex.find(".acc") == std::string::npos,
        "the hex round trip changed the kernel or carries an accumulator table");

    nanosm::Machine machine(profile, loaded, launch_warps(4), {.timeline = true, .debug = true});
    machine.run(10000);
    require(machine.trace().summary().tmem_store_beats == 32 && machine.trace().summary().tmem_load_beats == 32,
        "tmem trace did not expose every vector beat");
    for (uint32_t warp = 0; warp < 4; ++warp) {
        for (uint32_t lane = 0; lane < profile.lanes; ++lane) {
            require(machine.read_reg(warp, lane, 40) == 0x7fc00000u, "tmem transfer did not preserve raw NaN payload bits");
            for (uint32_t j = 1; j < 8; ++j)
                require(machine.read_reg(warp, lane, 40 + j) == j, "RF -> tmem -> RF vector round trip changed bits");
            require(machine.read_reg(warp, lane, 20) == 21, "concurrent arithmetic made no forward progress");
        }
    }
}

// qtxc orders one warp's tmem transfers by depbar: two acc.st share one
// counted depbar, an overwrite of their source waits for both, and a load of
// stored words samples after the store's final tmem write. Without the
// overwrite's wait (hand-edited) it is an RF WAR hazard, a sanitizer PANIC.
TEST(Mma, tmem_transfer_ordering_and_store_lifetime) {
    const auto& profile = canonical_profile();
    const auto scheduled = qtx_program(R"(
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
    )", profile);
    const auto& first_store = scheduled.instructions[4];
    const auto& second_store = scheduled.instructions[5];
    const auto& overwrite = scheduled.instructions[6];
    const auto& first_load = scheduled.instructions[7];
    require(first_store.control.set_depbar < 6 && second_store.control.set_depbar == first_store.control.set_depbar,
        "concurrent accumulator stores did not share a counted completion depbar");
    require((overwrite.control.wait_mask & (1u << first_store.control.set_depbar)) != 0,
        "overwrite did not wait for every older accumulator-store reader");
    require(first_load.control.wait_mask == 0, "a wait already paid by the overwrite was retained on the load");

    const auto ordered = qtx_program(R"(
        mov r0, 17
        umov ur8, 0
        umov ur9, 0
        acc.st [ur8, 16], r0
        acc.ld r2, [ur8, 16]
        halt
    )", profile);
    nanosm::Machine ordered_machine(profile, nanosm::assemble_asm(nanosm::emit_asm(ordered), profile), launch_warps(1),
        {.timeline = true, .debug = true});
    ordered_machine.run(1000);
    require(ordered_machine.read_reg(0, 0, 2) == 17, "qtxc's tmem order did not put an accumulator load after its store");
    const auto store_beats = events_named(ordered_machine, "acc.st beat");
    const auto load_beats = events_named(ordered_machine, "acc.ld beat");
    require(!store_beats.empty() && !load_beats.empty() &&
            store_beats.back().start_cycle < load_beats.back().start_cycle,
        "an accumulator load sampled tmem before the older store's final tmem write");

    nanosm::Machine machine(profile, nanosm::assemble_asm(nanosm::emit_asm(scheduled), profile), launch_warps(1),
        {.timeline = true, .debug = true});
    machine.run(1000);
    for (uint32_t lane = 0; lane < profile.lanes; ++lane) {
        require(machine.read_reg(0, lane, 2) == 17 && machine.read_reg(0, lane, 3) == 17,
            "a younger overwrite changed a deferred accumulator-store source");
        require(machine.read_reg(0, lane, 0) == 51, "the source overwrite did not execute after both stores completed");
    }
    require(machine.trace().summary().tmem_store_beats == 16 && machine.trace().summary().tmem_load_beats == 2,
        "scalar transfer completion did not account for each final tmem/RF beat");

#if NANOSM_SANITIZER
    auto unsafe = nanosm::assemble_asm(nanosm::emit_asm(scheduled), profile);
    auto unsafe_overwrite = nanosm::decode(unsafe.words[6]); // the overwrite
    unsafe_overwrite.control.wait_mask = 0;
    unsafe.words[6] = nanosm::encode(unsafe_overwrite);
    nanosm::Machine unsafe_machine(profile, std::move(unsafe), launch_warps(1));
    expect_panic(unsafe_machine, {"RF WAR hazard"}, "an acc.st source overwritten without its wait");
#endif
}

TEST(Mma, mma_raw_window_numerics) {
    using nanosm::mma_dot_f16;
    const std::array<uint16_t, 4> zero{};
    const std::array<uint16_t, 4> one{0x3c00, 0, 0, 0};
    const auto check = [&](std::array<uint16_t, 4> a, std::array<uint16_t, 4> b, uint32_t c, uint32_t expected) {
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
    // Frozen raw B200 FP16 K=16 vectors from an upstream tensor-core validation
    // script (seed 103). These validate the arithmetic helper, not the
    // machine's smaller fusion boundary.
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

    // The profile names the arithmetic; a missing or unknown name is refused.
    std::ifstream input("profiles/nanosm-v0.json");
    const auto document = nlohmann::json::parse(input);
    require(document.at("mma").at("numeric_model") == nanosm::mma_numeric_model,
        "canonical profile does not name the implemented MMA arithmetic");
    for (const bool missing : {false, true}) {
        auto changed = document;
        if (missing) changed["mma"].erase("numeric_model");
        else changed["mma"]["numeric_model"] = "serial_fma_fp32_reference";
        (void)error_of<nanosm::ProfileError>([&] { (void)nanosm::parse_device_profile(changed.dump()); },
            missing ? "a profile without mma.numeric_model" : "an unsupported mma.numeric_model");
    }
}

// A step is one fused group of core_k products plus C. These eight products
// tell the groups apart: 4096 x 4096 + 1 x 1 is 2^24 + 1, which a group of
// four rounds to 2^24 in FP32 before the second group's -2^24 cancels it (0);
// one group of eight keeps the unit (1.0). In either operand layout.
TEST(Mma, mma_fusion_boundaries) {
    const std::array<uint16_t, 8> a{0x6c00, 0x3c00, 0, 0, 0xec00, 0, 0, 0}; // 4096, 1, 0, 0, -4096, 0...
    const std::array<uint16_t, 8> b{0x6c00, 0x3c00, 0, 0, 0x6c00, 0, 0, 0}; // 4096, 1, 0, 0, 4096, 0...
    const auto eight = nanosm::mma_dot_f16(a, b, 0);
    const auto first_four = nanosm::mma_dot_f16(std::span(a).first(4), std::span(b).first(4), 0);
    const auto two_fours = nanosm::mma_dot_f16(std::span(a).last(4), std::span(b).last(4), first_four);
    require(eight == 0x3f800000u && first_four == 0x4b800000u && two_fours == 0,
        "the eight-product group and two four-product groups do not differ as derived");
    require(canonical_profile().mma_core_k == 8, "the canonical MMA core is not 4x4x8");
    struct Row {
        const char* what;
        nanosm::TileLayout layout;
        uint32_t core_k;
        uint32_t expected;
    };
    for (const auto& row : {Row{"canonical (swizzle64, 4x4x8)", nanosm::TileLayout::swizzle64, 8, 0x3f800000u},
             Row{"dense 4x4x8", nanosm::TileLayout::dense, 8, 0x3f800000u},
             Row{"dense 4x4x4", nanosm::TileLayout::dense, 4, 0u}}) {
        auto profile = canonical_profile();
        profile.mma_layout = row.layout;
        profile.mma_core_k = row.core_k;
        const auto fill = row.layout == nanosm::TileLayout::dense ? dense_tiles(64, 64, a, b) : tiles(a, b);
        auto machine = mma_machine(profile, compile(mma_kernel({.tail = acc_ld_r10}), profile), 1, fill);
        machine->run(20000);
        for (uint32_t lane = 0; lane < profile.lanes; ++lane)
            require(machine->read_reg(0, lane, 10) == row.expected,
                std::string(row.what) + ": the MMA did not round at the core_k boundary");
    }
}

TEST(Mma, worker_acc_store_mbar_handoff) {
    const auto& profile = canonical_profile();
    const auto scheduled = qtx_program(R"(
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
    auto launch = launch_warps(2);
    write_mbar(image_bytes(launch.test_shmem, 8192), 4096, mbar_state(2, 2));
    nanosm::Machine machine(profile, nanosm::assemble_asm(nanosm::emit_asm(scheduled), profile), std::move(launch));
    machine.run(2000);
    require(machine.read_mbar_state(4096).phase,
        "shared worker mbar did not receive exactly one post-store arrival per warp");
}

// Each mbar engine event, the TMA's complete_tx and the MMA's commit, has its
// own register: a TMA event pending in the cycle r the commit presents does
// not hold the commit at the MMA queue's head, so the mma.async queued behind
// it is accepted and fetches exactly as with no TMA event. The RMW engine
// serves the events rmw_busy_cycles() apart, alternating TMA, MMA, TMA (T1
// was served last, so the commit wins the tie at r - 1 + busy).
TEST(Mma, mbar_engine_tma_event_does_not_stall_mma_queue) {
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
        auto launch = launch_warps(1);
        auto& shmem = image_bytes(launch.test_shmem, 8192);
        write_mbar(shmem, commit_mbar, mbar_state(1, 1));
        write_mbar(shmem, tx_mbar, mbar_state(1, 0, 128)); // arrived; waits for 2 x 64 bytes
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

    // Reference, no TMA event: the commit presents in r, the cycle its
    // predecessor retires; the mma.async behind it issues the cycle after
    // (issue reads the queue's register) and is accepted at its dispatch,
    // fetching from the next cycle.
    nanosm::Machine reference(canonical_profile(), kernel, make_launch(), {.timeline = true});
    reference.run(5000);
    const auto [accept, fetch] = queued_accept_and_fetch(reference);
    const auto r = accept - nanosm::issue_to_dispatch(canonical_profile()) - 1;
    require(r > 2 && fetch == accept + 1, "reference: the queued mma.async did not fetch the cycle after its acceptance");
    require(mbar_rmw_services(reference) == std::vector<RmwService>{{r + 1, "arrive (mma.commit), address=4096"}},
        "reference commit was not served at r + 1");

    nanosm::Machine machine(canonical_profile(), kernel, make_launch(), {.timeline = true});
    run_until(machine, r - 1);
    machine.complete_mbar_tx(tx_mbar, 64, 0); // T1, pending in r - 1: the engine is free
    machine.tick();
    machine.complete_mbar_tx(tx_mbar, 64, 0); // T2, pending in r with the commit: the engine is in T1's write
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
// last-served bit starts at MMA). Every later tie goes to the engine served
// less recently: TMA, MMA, TMA, MMA, rmw_busy_cycles() apart, each loser
// waiting out the winner's whole RMW. complete_mbar_tx into the occupied TMA
// register PANICs, whatever the MMA's register holds.
TEST(Mma, mbar_engine_events_alternate_tma_first) {
    constexpr uint32_t commit0 = 4096;
    constexpr uint32_t tx_mbar = 4104;
    constexpr uint32_t commit1 = 4112;
    auto launch = launch_warps(1);
    auto& shmem = image_bytes(launch.test_shmem, 8192);
    write_mbar(shmem, commit0, mbar_state(1, 1));
    write_mbar(shmem, commit1, mbar_state(1, 1));
    write_mbar(shmem, tx_mbar, mbar_state(1, 0, 128));
    nanosm::Machine machine(canonical_profile(), compile(R"(
        umov ur4, 4096
        umov ur6, 4112
        mma.commit [ur4]
        mma.commit [ur6]
        halt
    )"), std::move(launch), {.timeline = true});

    while (!machine.mma_mbar_event_held()) tick_capped(machine); // the first commit presented
    const auto first = machine.cycle(); // the first free point with an engine event
    machine.complete_mbar_tx(tx_mbar, 64, 0); // accepted although the MMA register is full
    expect_error<nanosm::SimError>([&] { machine.complete_mbar_tx(tx_mbar, 64, 0); },
        {"TMA mbar event register is occupied"}, "a second event into the full TMA register");
    machine.tick(); // the tie: the TMA's RMW starts
    machine.tick();
    machine.complete_mbar_tx(tx_mbar, 64, 0); // pending again from first + 2, before the next free point
    machine.run(1000);

    const auto busy = rmw_busy_cycles();
    require(mbar_rmw_services(machine) == std::vector<RmwService>{
                {first, "complete_tx (tma), address=4104"},
                {first + busy, "arrive (mma.commit), address=4096"},
                {first + 2 * busy, "complete_tx (tma), address=4104"},
                {first + 3 * busy, "arrive (mma.commit), address=4112"}},
        "tied engine events did not alternate TMA, MMA, TMA, MMA");
    require(machine.trace().summary().mbar_engine_waits_on_other == 3 * busy,
        "engine_waits_on_other does not count the winner's whole RMW per tie");
    require(machine.read_mbar_state(commit0).phase && machine.read_mbar_state(commit1).phase,
        "a commit's arrive was lost");
    const auto tx = machine.read_mbar_state(tx_mbar);
    require(tx.phase && tx.tx_bytes == 0, "a TMA completion was lost");
}

// Back-to-back commands on one accumulator wait for each other's results
// when one command's sweep is shorter than a result's way to tmem
// (recurrence bubbles, counted): m16n4k8 on the dense layout is four steps of
// two rounds, eight cycles, against 15 from a step's start to its tmem write.
TEST(MmaSlow, mma_dense_profile_recurrence_bubbles) {
    auto profile = canonical_profile();
    profile.mma_shapes = {{16, 4, 8}};
    profile.mma_layout = nanosm::TileLayout::dense;
    nanosm::validate_device_profile(profile);
    auto machine = mma_machine(profile, compile(mma_kernel({.shape = "m16n4k8", .commands = 2}), profile), 1,
        dense_tiles(16, 4, filled(0x3c00), filled(0x3c00)));
    machine->run(20000);
    for (uint32_t row = 0; row < 16; ++row)
        for (uint32_t col = 0; col < 4; ++col)
            require(result_at(*machine, 0, 4, row, col) == 16.0f, "two commands of eight ones did not make 16");
    require(machine->trace().summary().mma_recurrence_stalls > 0,
        "back-to-back commands on one accumulator did not expose their recurrence bubbles");
}

// The dense layout on one shmem bank makes every port-B round one word: a
// step that reloads A (n block 0) is 16 rounds (eight A words, eight B
// words), every other step 8. Steps go back to back, one round a cycle; a
// step enters the core when its last round reaches the port output, D + Lb
// after its grant; its result lands in tmem result latency + Dt later; the
// operand-consumed depbar clears at the final step's last round's RAM sample.
TEST(Mma, mma_multi_round_fetch_timing) {
    const auto profile = dense_one_bank_profile(12);
    const auto scheduled = qtx_program(mma_kernel({.shape = "m32n64k8", .tail = acc_ld_r10}), profile);
    const auto init = std::find_if(scheduled.instructions.begin(), scheduled.instructions.end(),
        [](const auto& instruction) { return instruction.opcode == nanosm::mma_opcode(0, true); });
    require(init != scheduled.instructions.end(), "the kernel has no mma.async.init");
    const auto bar = init->control.set_depbar;
    auto machine = mma_machine(profile, nanosm::assemble_asm(nanosm::emit_asm(scheduled), profile), 1,
        dense_tiles(32, 64, filled(0x3c00), filled(0x3c00)));
    machine->run(8000);
    std::vector<uint64_t> inputs;
    for (const auto& event : events_named(*machine, "MMA input step")) inputs.push_back(event.start_cycle);
    const auto writes = events_named(*machine, "MMA tmem write");
    uint64_t depbar_cleared = 0;
    for (const auto& event : events_named(*machine, "depbar sb" + std::to_string(bar))) {
        if (event.detail == "value=0") {
            depbar_cleared = event.start_cycle;
            break;
        }
    }
    // 8 m blocks x 16 n blocks x 2 k blocks.
    const auto rounds = [](std::size_t step) -> uint64_t { return step % 16 == 0 ? 16 : 8; };
    require(inputs.size() == 256 && !writes.empty(), "one dense m32n64k8 command on a 4x4x4 core is 256 core steps");
    for (std::size_t i = 1; i < inputs.size(); ++i)
        require(inputs[i] == inputs[i - 1] + rounds(i - 1),
            "step " + std::to_string(i) + ": the fetches did not serialize the port at one round per cycle");
    const auto d = profile.shmem_port_a_depth;
    const auto lb = profile.shmem_read_latency_cycles;
    require(writes.front().start_cycle == inputs[0] + 15 + d + lb + 12 + profile.tmem_port_depth,
        "the result did not land at core entry (last round + D + Lb) plus the result latency plus Dt");
    require(depbar_cleared == inputs[255] + rounds(255) - 1 + d,
        "operand-consumed depbar did not clear at the final physical read (the last round's sample)");
    require(machine->trace().summary().mma_recurrence_stalls == 0,
        "a tile is revisited 128 steps (over a thousand rounds) later; no bubbles expected");
    for (uint32_t lane = 0; lane < profile.lanes; ++lane)
        require(machine->read_reg(0, lane, 10) == nanosm::float_bits(8.0f),
            "per-round operand capture changed the numerical result");
}

// One command, step by step (hold A / stream B): for each A block (m0 = 4i),
// every B block (n0 = 4j) in order; the step with j = 0 reads its A block and
// its B block (2 bank rounds), the other 15 one B block (1 round) with the
// held A block. Step s + 1 starts rounds(s) cycles after step s; a step whose
// last round is granted at g writes tile C + 64 m0 + n0 at tmem at
// g + D + Lb + result latency + Dt. m64 on a 64-row A object, and m32 on its
// bottom half (base 512).
TEST(Mma, hold_a_stream_b_walker) {
    const auto& profile = canonical_profile();
    struct Row {
        const char* shape;
        uint32_t a, acc, rows, a_reads, rounds;
    };
    for (const auto& row : {Row{"m64n64k8", 0, 0, 64, 16, 272}, Row{"m32n64k8", 512, 4096, 32, 8, 136}}) {
        const auto what = std::string(row.shape) + " ";
        const uint32_t steps = row.rows / 4 * 16;
        auto machine = mma_machine(profile,
            compile(mma_kernel({.shape = row.shape, .a = row.a, .b = 2048, .acc = row.acc}), profile), 1,
            write_small_operands);
        machine->run(20000);
        const auto inputs = events_named(*machine, "MMA input step");
        const auto writes = events_named(*machine, "MMA tmem write");
        require(inputs.size() == steps && writes.size() == steps, what + "is not " + std::to_string(steps) + " 4x4x8 steps");
        const auto d = profile.shmem_port_a_depth;
        const auto lb = profile.shmem_read_latency_cycles;
        for (uint32_t s = 0; s < steps; ++s) {
            const auto step = what + "step " + std::to_string(s);
            const bool reads_a = s % 16 == 0;
            const uint32_t rounds = reads_a ? 2 : 1;
            require(static_cast<uint32_t>(inputs[s].value) == s &&
                    inputs[s].detail == (reads_a ? "rounds=2, reads A" : "rounds=1"),
                step + " read its A block at the wrong step: " + inputs[s].detail);
            if (s + 1 < steps)
                require(inputs[s + 1].start_cycle == inputs[s].start_cycle + rounds,
                    step + ": the next step did not start the cycle after this step's last round");
            const uint32_t tile = row.acc + 64 * (4 * (s / 16)) + 4 * (s % 16); // C + m0 * 64 + n0, n fastest
            require(static_cast<uint32_t>(writes[s].value) == tile,
                step + " wrote tile " + std::to_string(writes[s].value) + ", not " + std::to_string(tile));
            require(writes[s].start_cycle == inputs[s].start_cycle + (rounds - 1) + d + lb +
                        profile.mma_result_latency_cycles + profile.tmem_port_depth,
                step + ": tmem write is not last round + D + Lb + result latency + Dt");
        }
        const auto& summary = machine->trace().summary();
        require(summary.mma_a_block_reads == row.a_reads && summary.mma_shmem_rounds == row.rounds,
            what + "did not read " + std::to_string(row.a_reads) + " A blocks in " + std::to_string(row.rounds) +
                " bank rounds");
        for (uint32_t r = 0; r < row.rows; ++r)
            for (uint32_t col = 0; col < 64; ++col)
                require(result_at(*machine, row.acc, 64, r, col) == product_reference(row.a / 16 + r, col),
                    what + "element (" + std::to_string(r) + ", " + std::to_string(col) + ")");
    }
}

// Back-to-back commands into one accumulator: the next command's first step
// is granted the cycle after the previous command's last round, so N commands
// keep port B busy 272 N cycles from the first input step (16/17 of a step a
// cycle), with no recurrence stall.
TEST(Mma, back_to_back_throughput) {
    constexpr uint32_t commands = 8;
    auto machine = mma_machine(canonical_profile(), compile(mma_kernel({.commands = commands, .tail = acc_ld_r10})), 1,
        tiles(filled(0x3c00), filled(0x3c00)));
    machine->run(100000);
    for (uint32_t lane = 0; lane < 16; ++lane)
        require(machine->read_reg(0, lane, 10) == nanosm::float_bits(8.0f * commands),
            "back-to-back commands accumulated the wrong value");
    const auto inputs = events_named(*machine, "MMA input step");
    require(inputs.size() == 256 * commands, "back-to-back commands lost steps");
    uint64_t first = std::numeric_limits<uint64_t>::max(), last = 0;
    for (const auto& event : inputs) {
        first = std::min(first, event.start_cycle);
        last = std::max(last, event.start_cycle);
    }
    // The last step reads one round, so port B is busy through `last`.
    require(last - first + 1 == 272 * commands,
        "back-to-back commands did not keep port B busy one round per cycle: " + std::to_string(last - first + 1) +
            " cycles for " + std::to_string(272 * commands) + " rounds");
    require(machine->trace().summary().mma_recurrence_stalls == 0, "back-to-back commands hit recurrence stalls");
}

// Tmem is 16 fine banks, each with two read and two write services a cycle:
// the MMA's accumulator read and write, the acc.ld engine's read, the acc.st
// engine's write. Nobody arbitrates for them. Warp 0 runs `commands` MMA
// commands into S; warp 1 meanwhile streams acc.st.v8 / acc.ld.v8 round trips
// through O (another region): every MMA step keeps the walker's timing
// (hold_a_stream_b_walker) and the commands keep port B busy one round a
// cycle, while beats of both transfer kinds run inside the MMA's span.
TEST(Mma, mma_beside_transfers) {
    const auto& profile = canonical_profile();
    constexpr uint32_t commands = 4;
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
    )" + mma_kernel({.commands = commands, .tail = acc_ld_r10});
    auto machine = mma_machine(profile, compile(body, profile), 2, tiles(filled(0x3c00), filled(0x4000))); // 1 x 2
    machine->run(100000);
    const auto inputs = events_named(*machine, "MMA input step");
    const auto writes = events_named(*machine, "MMA tmem write");
    require(inputs.size() == 256 * commands && writes.size() == 256 * commands, "MMA steps lost");
    const auto d = profile.shmem_port_a_depth;
    const auto lb = profile.shmem_read_latency_cycles;
    for (std::size_t s = 0; s < inputs.size(); ++s) {
        const uint32_t rounds = s % 16 == 0 ? 2 : 1;
        if (s + 1 < inputs.size())
            require(inputs[s + 1].start_cycle == inputs[s].start_cycle + rounds,
                "step " + std::to_string(s + 1) + " waited (the transfers stalled the MMA?)");
        require(writes[s].start_cycle == inputs[s].start_cycle + (rounds - 1) + d + lb +
                    profile.mma_result_latency_cycles + profile.tmem_port_depth,
            "step " + std::to_string(s) + "'s tmem write moved (the transfers delayed the MMA?)");
    }
    require(inputs.back().start_cycle - inputs.front().start_cycle + 1 == 272 * commands,
        "back to back with transfers alongside, the MMA did not keep port B busy one round per cycle");
    const auto first = inputs.front().start_cycle;
    const auto last = writes.back().start_cycle;
    const auto inside = [&](const std::string& name) {
        const auto beats = events_named(*machine, name);
        return std::count_if(beats.begin(), beats.end(),
            [&](const auto& beat) { return beat.start_cycle >= first && beat.start_cycle <= last; });
    };
    require(inside("acc.ld beat") >= 16 && inside("acc.st beat") >= 16,
        "the transfers did not run alongside the MMA (" + std::to_string(inside("acc.ld beat")) + " acc.ld and " +
            std::to_string(inside("acc.st beat")) + " acc.st beats inside it)");
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(machine->read_reg(0, lane, 10) == nanosm::float_bits(16.0f * commands), "MMA result");
        for (uint32_t j = 0; j < 8; ++j)
            require(machine->read_reg(1, lane, 16 + j) == lane + 100 * j, "acc.st -> acc.ld round trip");
    }
}

// One acc.ld engine and one acc.st engine for the machine. Two warps each
// store eight columns of their own rows and load them back: at most one
// acc.ld beat (a tmem sample) and one acc.st beat (a tmem write) a cycle; the
// second warp's transfer of a kind waits at admission ("acc transfer engine
// busy"); an acc.ld and an acc.st do run in the same cycles. Two acc.ld.v8 at
// once: one waits until the other's last beat has launched, so their beats do
// not interleave.
TEST(Mma, one_engine_of_each_kind) {
    const auto& profile = canonical_profile();
    auto machine = mma_machine(profile, compile(R"(
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
    )", profile), 2, {});
    machine->run(20000);
    std::map<uint64_t, uint32_t> ld_per_cycle, st_per_cycle;
    for (const auto& event : events_named(*machine, "acc.ld beat")) ++ld_per_cycle[event.start_cycle];
    for (const auto& event : events_named(*machine, "acc.st beat")) ++st_per_cycle[event.start_cycle];
    require(ld_per_cycle.size() == 16 && st_per_cycle.size() == 16, "two warps x eight beats of each kind");
    for (const auto& [cycle, count] : ld_per_cycle) require(count == 1, "two acc.ld beats in one cycle");
    for (const auto& [cycle, count] : st_per_cycle) require(count == 1, "two acc.st beats in one cycle");
    bool overlap = false;
    for (const auto& [cycle, count] : ld_per_cycle) overlap = overlap || st_per_cycle.contains(cycle);
    require(overlap, "an acc.ld and an acc.st never ran in the same cycle");
    const auto engine_waited = [](const nanosm::Machine& m) {
        return !refusals(m, 0, "acc transfer engine busy").empty() || !refusals(m, 1, "acc transfer engine busy").empty();
    };
    require(engine_waited(*machine), "no transfer ever waited for its engine");
    for (uint32_t warp = 0; warp < 2; ++warp)
        for (uint32_t lane = 0; lane < 16; ++lane)
            for (uint32_t j = 0; j < 8; ++j)
                require(machine->read_reg(warp, lane, 16 + j) == 1024 * warp + lane + 100 * j, "round trip data");

    auto loads = mma_machine(profile, compile(R"(
        usread ur8, WARP_ID
        ushli ur8, ur8, 10
        umov ur9, 0
        acc.ld.v8 r16, [ur8, 64]
        halt
    )", profile), 2, {});
    loads->run(20000);
    std::vector<std::string> owners;
    for (const auto& event : events_named(*loads, "acc.ld beat")) owners.push_back(event.detail.substr(0, event.detail.find(',')));
    require(owners.size() == 16, "two acc.ld.v8 are sixteen beats");
    for (uint32_t i = 0; i < 16; ++i)
        require(owners[i] == owners[i < 8 ? 0 : 8], "the two acc.ld transfers' beats interleaved");
    require(owners[0] != owners[8], "both transfers' beats came from one warp");
    require(engine_waited(*loads), "the second acc.ld did not wait for the acc.ld engine");
}

// The tmem depths at (Dt, Lt) = (1, 1), (2, 2), (3, 1), from the debug trace
// (hand assembly, one warp): an acc.st beat whose RF read is granted at g
// reaches the tmem RAM ("acc.st beat") at g + Lr + Dt; an acc.ld launched at
// L = its dispatch + 1 samples tmem ("acc.ld beat") at L + Dt and is at the RF
// port ("RF write") at L + Dt + Lt; an MMA step writes tmem ("MMA tmem
// write") (rounds - 1) + D + Lb + result latency + Dt after its start.
TEST(Mma, tmem_depths) {
    for (const auto& [dt, lt] : std::vector<std::pair<uint32_t, uint32_t>>{{1, 1}, {2, 2}, {3, 1}}) {
        auto profile = canonical_profile();
        profile.tmem_port_depth = dt;
        profile.tmem_read_latency_cycles = lt;
        nanosm::validate_device_profile(profile);
        const auto what = "Dt/Lt " + std::to_string(dt) + "/" + std::to_string(lt) + ": ";
        auto machine = debug_machine(profile, R"(
            sread r0, LANE_ID
            umov ur8, 0
            umov ur9, 0
            acc.st [ur8, 64], r0 {pred=PT, stall=0, set=sb0, wait=-}
            acc.ld r10, [ur8, 64] {pred=PT, stall=0, set=sb1, wait=sb0}
            halt {pred=PT, stall=0, set=-, wait=sb1}
        )");
        machine->run();
        const auto st_issue = issued_at(*machine, 3 * 8).at(0);
        const auto ld_issue = issued_at(*machine, 4 * 8).at(0);
        const auto t = nanosm::issue_to_dispatch(profile);
        const auto st_beats = events_named(*machine, "acc.st beat");
        const auto ld_beats = events_named(*machine, "acc.ld beat");
        const auto reads = rf_events(*machine, "RF read", 0);
        const auto writes = rf_events(*machine, "RF write", 10);
        require(!st_beats.empty() && !ld_beats.empty() && !reads.empty() && !writes.empty(),
            what + "a beat or an RF event is missing from the trace");
        const auto g = reads.back(); // the acc.st's data read
        require(g >= st_issue + t + 1, what + "the acc.st's RF read before its engine had it");
        require(st_beats.back().start_cycle == g + profile.rf_read_latency_cycles + dt,
            what + "acc.st did not land at g + Lr + Dt");
        const auto launch = ld_issue + t + 1;
        require(ld_beats.back().start_cycle == launch + dt, what + "acc.ld did not sample at its launch + Dt");
        require(writes.back() == launch + dt + lt, what + "acc.ld was not at the RF port at its launch + Dt + Lt");
        for (uint32_t lane = 0; lane < 16; ++lane)
            require(machine->read_reg(0, lane, 10) == lane, what + "acc.st -> acc.ld data");

        auto mma = mma_machine(profile, compile(mma_kernel({.tail = acc_ld_r10}), profile), 1,
            tiles(filled(0x3c00), filled(0x4000)));
        mma->run(20000);
        const auto inputs = events_named(*mma, "MMA input step");
        const auto results = events_named(*mma, "MMA tmem write");
        require(!inputs.empty() && !results.empty() &&
                results.front().start_cycle == inputs.front().start_cycle + 1 + profile.shmem_port_a_depth +
                    profile.shmem_read_latency_cycles + profile.mma_result_latency_cycles + dt,
            what + "the first MMA step's tmem write is not its start + 1 + D + Lb + result latency + Dt");
        for (uint32_t lane = 0; lane < 16; ++lane)
            require(mma->read_reg(0, lane, 10) == nanosm::float_bits(16.0f), what + "MMA data");
    }
}

// Same word, same cycle (READ_FIRST, as shmem): an acc.st issued at I and an
// acc.ld of the same column `gap` cycles later (hand assembly, no wait: a
// race). The store's beat reaches the tmem RAM at I + t + 1 + Lr + Dt; the
// load samples at I + gap + t + 1 + Dt, the same cycle when gap = Lr (2), and
// then sees the OLD word (the write lands at that cycle's edge, after every
// read of the cycle); at gap 3 it sees the new one. The sanitizer PANICs at
// the load's dispatch, before the sample, so the timing is checked only with
// the sanitizer compiled out.
TEST(Mma, read_first_same_word) {
    const auto& profile = canonical_profile();
    require(profile.rf_read_latency_cycles == 2, "the derivation needs Lr = 2");
    for (const uint32_t gap : {2u, 3u}) {
        auto machine = debug_machine(profile, R"(
            sread r1, LANE_ID
            addi r1, r1, 7
            umov ur8, 0
            umov ur9, 0
            acc.st [ur8, 64], r1 {pred=PT, stall=)" + std::to_string(gap) + R"(, set=sb0, wait=-}
            acc.ld r10, [ur8, 64] {pred=PT, stall=0, set=sb1, wait=-}
            halt {pred=PT, stall=0, set=-, wait=sb0|sb1}
        )");
#if NANOSM_SANITIZER
        require_mentions(panic_of(*machine, "acc.ld over an in-flight acc.st"),
            {"tmem race: acc.ld reads tmem 0 (64 words per row) rows [0, 16) cols [0, 1)", "overlapping acc.st"},
            "same-word acc.ld after acc.st");
#else
        while (!machine->done()) tick_capped(*machine);
        const auto st_beats = events_named(*machine, "acc.st beat");
        const auto ld_beats = events_named(*machine, "acc.ld beat");
        require(!st_beats.empty() && !ld_beats.empty() &&
                ld_beats.back().start_cycle == st_beats.back().start_cycle + gap - 2,
            "the load's sample is not " + std::to_string(gap - 2) + " cycles after the store's write");
        const uint32_t expected_lane0 = gap == 2 ? 0u : 7u; // the old (zero) tmem word, or the stored lane + 7
        require(machine->read_reg(0, 0, 10) == expected_lane0,
            gap == 2 ? "a same-cycle sample saw the landing write (READ_FIRST)"
                     : "a sample one cycle after the write did not see it");
#endif
    }
}

// The mma.async block. Shape i of the profile's list is opcode 0x70 + 2i
// (accumulate) and 0x70 + 2i + 1 (init); names follow the profile, and
// without one an opcode is named by its place in the block. A lossless
// encoding; the text qtxc and the hand assembler parse; shape names parsed
// strictly; the shape-less spellings are not instructions.
TEST(Mma, mma_opcodes) {
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
    const auto parsed = qtx_program("mma.async.m64n64k8 [ur8], [ur0, 128], [ur1]\nhalt\n", profile);
    require(nanosm::mma_row_spacing(parsed.instructions.at(0), true) == 128 &&
            nanosm::mma_row_spacing(parsed.instructions.at(0), false) == 16,
        "[ur0, 128], [ur1] did not parse to spacings 128 and 16");
    for (const auto& [operand, needle] : {std::pair{"[ur0, 24]", "multiple of 16"},
             std::pair{"[ur0, 2048]", "multiple of 16"}, std::pair{"[ur0, 48]", "a power of two"},
             std::pair{"[ur0, 192]", "a power of two"}, std::pair{"[ur0, 1, 2]", "bad mma operand"}}) {
        const auto source = std::string("mma.async.m64n64k8 [ur8], ") + operand + ", [ur1]\nhalt\n";
        expect_error<nanosm::SimError>([&] { (void)qtx_program(source, profile); }, {needle},
            std::string("the mma operand ") + operand);
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
        const auto scheduled = qtx_program(source, profile);
        const auto kernel = nanosm::assemble_asm(nanosm::emit_asm(scheduled), profile);
        require(nanosm::decode(kernel.words.at(3)).opcode == nanosm::mma_opcode(e.index, e.init),
            std::string(e.name) + ": qtxc and the hand assembler did not produce the opcode");
        require(nanosm::emit_hex(kernel, &profile.mma_shapes).find(std::string(e.name) + " [ur2], [ur0], [ur1]") !=
                std::string::npos,
            std::string(e.name) + ": the hex comment does not name the shape and the operands");
    }
    for (const auto* shapeless : {"mma.async C, [ur0], [ur1]", "mma.async.init C, [ur0], [ur1]"}) {
        (void)error_of<nanosm::SimError>(
            [&] { (void)qtx_program(std::string("umov ur0, 0\numov ur1, 2048\n") + shapeless + "\nhalt\n", profile); },
            std::string("the shape-less '") + shapeless + "'");
    }
}

// The profile lists its MMA shapes by name (any shape that follows the
// rules, each once): m, n, k powers of two and whole multiples of the core; an
// accumulator of at least `lanes` rows that fits in tmem; k > core_k only
// with enough blocks per K block to hide the accumulator latency; at most
// eight shapes (the opcode block). The list is fingerprinted, so its ORDER
// (which assigns the opcodes) is too. The core and tmem geometry: tmem gives
// each of its 16 fine banks one word per port a cycle, so an acc.ld / acc.st
// beat of more than 16 lanes or a core result block with two rows in one fine
// bank is refused; the result latency must be more than the accumulator read.
// A kernel using a shape or an opcode the profile does not define is refused
// by qtxc, by the hand assembler and at kernel load; the result operand is a
// tmem pointer alone, [urC].
TEST(Mma, profile_shapes_and_refusals) {
    const auto& profile = canonical_profile();
    require(profile.mma_shapes == std::vector<nanosm::MmaShape>{{64, 64, 8}, {32, 64, 8}},
        "the canonical profile does not list m64n64k8 and m32n64k8");
    std::ifstream input("profiles/nanosm-v0.json");
    nlohmann::json document;
    input >> document;
    // The latency rule at canonical timing: D 2 + Lb 2 + result latency 8 +
    // tmem port depth 2 = 14 steps between visits of one accumulator block.
    require(nanosm::mma_min_blocks_per_k(profile) == 14, "the canonical latency bound is 2 + 2 + 8 + 2");

    struct ListRow {
        nlohmann::json shapes; // null: no mma.shapes at all
        std::vector<std::string> needles;
        std::string what;
    };
    const auto rule = [](const std::string& shape, const std::string& expected, const std::string& what) {
        return ListRow{nlohmann::json::array({"m64n64k8", shape}), {"mma shape " + shape, expected}, what};
    };
    auto nine = nlohmann::json::array();
    for (const char* shape : {"m16n4k8", "m16n8k8", "m16n16k8", "m16n32k8", "m16n64k8", "m16n128k8",
             "m32n4k8", "m32n8k8", "m32n16k8"})
        nine.push_back(shape);
    const std::vector<ListRow> lists{
        {nullptr, {}, "a profile without mma.shapes"},
        {nlohmann::json::array(), {}, "an empty shape list"},
        {"m64n64k8", {}, "a shape name that is not a list"},
        {nlohmann::json::array({"m64n64k8", "m64n64k8"}), {}, "a shape listed twice"},
        {nlohmann::json::array({64}), {}, "a shape that is not a name"},
        {nlohmann::json::array({"m64n64"}), {}, "a malformed shape name"},
        // Every dimension is a power of two, checked first.
        rule("m6n64k8", "m, n and k must be powers of two (m is 6)", "an m that is not a power of two"),
        rule("m48n64k8", "m, n and k must be powers of two (m is 48)", "m = 48"),
        rule("m64n12k8", "m, n and k must be powers of two (n is 12)", "n = 12"),
        rule("m64n64k12", "m, n and k must be powers of two (k is 12)", "k = 12"),
        // Powers of two that are not whole core blocks.
        rule("m64n2k8", "multiples of the core's 4x4x8", "an n smaller than core_n"),
        rule("m64n64k4", "multiples of the core's 4x4x8", "a k smaller than core_k"),
        rule("m8n64k8", "an accumulator needs at least 16", "an accumulator smaller than a lane each"),
        rule("m128n128k8", "does not fit in tmem", "an accumulator larger than tmem"),
        rule("m16n8k16", "revisited every 8 steps", "a k > core_k shape revisiting its blocks too soon"),
        // With every dimension a power of two, the blocks per K block come in
        // powers of two too: 8 is short of 14, 16 is enough.
        rule("m16n12k16", "m, n and k must be powers of two (n is 12)", "a k > core_k shape with n = 12"),
        {nine, {"lists 9 shapes", "holds 8"}, "nine shapes"},
    };
    const auto mentions = [](const std::string& message, const std::vector<std::string>& needles, const std::string& what) {
        for (const auto& needle : needles)
            require(message.find(needle) != std::string::npos, what + ": diagnostic lacks '" + needle + "': " + message);
    };
    for (const auto& row : lists) {
        auto changed = document;
        if (row.shapes.is_null()) changed["mma"].erase("shapes");
        else changed["mma"]["shapes"] = row.shapes;
        mentions(error_of<nanosm::ProfileError>([&] { (void)nanosm::parse_device_profile(changed.dump()); }, row.what),
            row.needles, row.what);
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

    const auto changed = [&](const std::function<void(nanosm::DeviceProfile&)>& change) {
        auto result = profile;
        change(result);
        return result;
    };
    const auto dense = changed([](auto& p) { p.mma_layout = nanosm::TileLayout::dense; });
    require(nanosm::device_profile_fingerprint(profile) != nanosm::device_profile_fingerprint(dense),
        "MMA operand layout was omitted from the profile fingerprint");
    struct ProfileRow {
        std::string what;
        std::function<void()> validate;
        std::vector<std::string> needles;
    };
    const std::vector<ProfileRow> refused{
        {"32 lanes on 16 tmem fine banks",
            [&] { nanosm::validate_device_profile(changed([](auto& p) { p.lanes = 32; })); },
            {"lanes 32", "tmem fine_banks (16)"}},
        {"a dense 8x2 core",
            [&] { nanosm::validate_device_profile(changed([](auto& p) {
                p.mma_layout = nanosm::TileLayout::dense;
                p.mma_core_m = 8;
                p.mma_core_n = 2;
            })); },
            {"core_m x core_n 8x2", "one tmem fine bank"}},
        {"a dense 3-row core (32 and 64 rows are not whole blocks)",
            [&] { nanosm::validate_device_profile(changed([](auto& p) {
                p.mma_layout = nanosm::TileLayout::dense;
                p.mma_core_m = 3;
            })); },
            {"multiples of the core's 3x4x8"}},
        {"swizzle64 on a 4x2 core",
            [&] { nanosm::validate_device_profile(changed([](auto& p) { p.mma_core_n = 2; })); },
            {"swizzle64 MMA layout needs a 4x4x4 or 4x4x8 core"}},
        {"a result latency equal to the accumulator read (Dt 2 + Lt 2)", [] { (void)dense_one_bank_profile(4); },
            {"mma result_latency_cycles 4", "tmem port_depth + read_latency_cycles = 4", "at least 5"}},
    };
    for (const auto& row : refused) mentions(error_of<nanosm::ProfileError>(row.validate, row.what), row.needles, row.what);
    // The controls: the dense 4x4x8 core, one column of eight rows (eight
    // fine banks), the shortest result latency past the accumulator read.
    nanosm::validate_device_profile(dense);
    nanosm::validate_device_profile(changed([](auto& p) {
        p.mma_layout = nanosm::TileLayout::dense;
        p.mma_core_m = 8;
        p.mma_core_n = 1;
    }));
    (void)dense_one_bank_profile(5);

    const std::string m32 =
        "umov ur0, 0\numov ur1, 2048\numov ur2, 4096\nmma.async.init.m32n64k8 [ur2], [ur0], [ur1]\nhalt\n";
    expect_error<nanosm::SimError>([&] { (void)qtx_program(m32, restricted); },
        {"line 4", "mma.async.init.m32n64k8: shape m32n64k8 is not one of this device profile's mma shapes"},
        "qtxc: a shape the profile does not list");
    // The hand assembler: an m64 kernel assembled for the restricted profile,
    // its mnemonic changed to m32 in the text.
    auto text = nanosm::emit_asm(qtx_program(
        "umov ur0, 0\numov ur1, 2048\nmma.async.init.m64n64k8 [ur8], [ur0], [ur1]\nhalt\n", restricted));
    text.replace(text.find("mma.async.init.m64n64k8"), 23, "mma.async.init.m32n64k8");
    expect_error<nanosm::SimError>([&] { (void)nanosm::assemble_asm(text, restricted); },
        {"shape m32n64k8 is not one of this device profile's mma shapes"}, "the hand assembler: an unsupported shape");
    // Kernel load: a canonical m32 kernel relabelled for the restricted profile.
    auto relabelled = compile(m32, profile);
    relabelled.profile_fingerprint = nanosm::device_profile_fingerprint(restricted);
    expect_error<nanosm::SimError>([&] { nanosm::Machine machine(restricted, relabelled, launch_warps(1)); },
        {"kernel pc 24", "mma.async.init.slot1: this device profile defines no mma shape for it (its list has 1 shapes)"},
        "kernel load: an opcode past the profile's shape list");
    // The shape list's order assigns the opcodes and is fingerprinted: a
    // kernel built for one order does not load on a profile with another.
    auto reordered = document;
    reordered["mma"]["shapes"] = nlohmann::json::array({"m32n64k8", "m64n64k8"});
    const auto swapped = nanosm::parse_device_profile(reordered.dump());
    require(nanosm::device_profile_fingerprint(swapped) != nanosm::device_profile_fingerprint(profile),
        "the shape list's order is not fingerprinted");
    expect_error<nanosm::SimError>([&] { nanosm::Machine machine(swapped, compile(m32, profile), launch_warps(1)); },
        {"does not match simulator"}, "a kernel built for one shape order, loaded on another");
    // The result operand is a pointer alone (its rows are the shape's n
    // words): a row length, a vector register or a name is refused.
    for (const auto& [result, needle] : {std::pair{"[ur2, 64]", "without a row length"},
             std::pair{"[r2]", "%r2 is used here as a uniform register"},
             std::pair{"T", "a uniform register holding its tmem pointer"}}) {
        const auto source = std::string("mma.async.init.m32n64k8 ") + result + ", [ur0], [ur1]\nhalt\n";
        expect_error<nanosm::SimError>([&] { (void)qtx_program(source, profile); }, {needle},
            std::string("the mma result ") + result);
    }
}

// One object as both operands (mma.async C, A, A): B(k, n) is the object's
// row n, so the result is A x A^T. Where a step's B rows are its A rows, the
// two blocks are the same sixteen words: one bank round fills both from the
// operand latch's A and B positions.
TEST(Mma, one_object_as_both_operands) {
    const auto& profile = canonical_profile();
    auto machine = mma_machine(profile,
        compile(mma_kernel({.shape = "m32n64k8", .a = 0, .b = 0, .acc = 4096}), profile), 1, write_small_operands);
    machine->run(20000);
    const auto inputs = events_named(*machine, "MMA input step");
    require(inputs.size() == 128 && inputs[0].detail == "rounds=1, reads A" && inputs[16].detail == "rounds=2, reads A",
        "a step whose A and B blocks are the same words did not take one round");
    for (uint32_t row = 0; row < 32; ++row) {
        for (uint32_t col = 0; col < 64; ++col) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < 8; ++k) sum += static_cast<float>(small_a(row, k) * small_a(col, k));
            require(result_at(*machine, 4096, 64, row, col) == sum,
                "A x A^T: element (" + std::to_string(row) + ", " + std::to_string(col) + ")");
        }
    }
}

// The same 64-row A object as one m64 command (F) and as two m32 commands on
// its halves (T at base 0, L at 512): the halves equal the m64 result bit for
// bit and the reference, and the two m32 commands back to back keep port B
// busy exactly 2 x 136 = 272 cycles from their first input step. With an m64
// command adding into F between them (a mixed queue), F holds twice the
// product and the halves once: 768 steps, 816 rounds.
TEST(Mma, two_m32_halves_equal_one_m64) {
    const auto& profile = canonical_profile();
    struct Row {
        const char* what;
        const char* third; // between the two m32 commands
        uint32_t f_factor, steps, rounds, a_reads;
    };
    for (const auto& row : {Row{"two m32 halves", "", 1, 512, 544, 32},
             Row{"mixed queue", "mma.async.m64n64k8 [ur5], [ur0], [ur1]\n", 2, 768, 816, 48}}) {
        const std::string what = std::string(row.what) + ": ";
        auto machine = mma_machine(profile, compile(std::string(R"(
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
        )") + row.third + "mma.async.init.m32n64k8 [ur7], [ur2], [ur1]\nmma.commit [ur4]\n" + wait_on("ur4") + "halt\n",
            profile), 1, write_small_operands);
        machine->run(40000);
        for (uint32_t r = 0; r < 64; ++r) {
            for (uint32_t col = 0; col < 64; ++col) {
                const auto full = result_word(*machine, 0, 64, r, col);
                const auto half = result_word(*machine, r < 32 ? 4096 : 6144, 64, r % 32, col);
                const auto where = what + "row " + std::to_string(r) + " col " + std::to_string(col);
                require(nanosm::bits_float(full) == static_cast<float>(row.f_factor) * product_reference(r, col),
                    where + ": F " + std::to_string(nanosm::bits_float(full)));
                require(nanosm::bits_float(half) == product_reference(r, col) && (row.f_factor != 1 || full == half),
                    where + ": the m32 half " + std::to_string(nanosm::bits_float(half)));
            }
        }
        const auto& summary = machine->trace().summary();
        require(summary.mma_input_steps == row.steps && summary.mma_shmem_rounds == row.rounds &&
                summary.mma_a_block_reads == row.a_reads,
            what + "steps, rounds or A reloads");
        if (row.f_factor != 1) continue;
        // The two m32 commands (steps 256..511) run back to back: the last
        // step reads one round, so port B is busy through its start.
        const auto inputs = events_named(*machine, "MMA input step");
        require(inputs.back().start_cycle - inputs.at(256).start_cycle + 1 == 272,
            what + "the two m32 commands did not keep port B busy 272 cycles: " +
                std::to_string(inputs.back().start_cycle - inputs.at(256).start_cycle + 1));
        require(summary.mma_recurrence_stalls == 0, what + "independent accumulators hit recurrence stalls");
    }
}

// Legal shapes beyond the canonical two run correctly against the reference:
// a k > core_k shape over four K blocks, a short 16-row shape and a 128-row
// shape that fills tmem. Steps and port-B rounds follow the walk: per K
// block every (m, n) block, A read once per m block (two rounds), and no step
// of one command waits on its own accumulator. On a 4x4x4 core m16n16k4 is
// legal but reads only 4 of the 8 fp16 in each 16-byte operand row: a
// warning, not an error; admission counts the bytes the walk reads (rows
// 12..15 keep their four fp16 in the second half of their row, word XOR 3,
// so A's last byte read is 256 bytes past its base): A at 65536 - 128 is
// refused at admission, at 65536 - 256 it runs.
TEST(Mma, more_shapes_run) {
    const auto profile = shape_profile({{64, 64, 8}, {16, 64, 32}, {16, 64, 8}, {128, 64, 8}});
    for (const nanosm::MmaShape shape : {nanosm::MmaShape{16, 64, 32}, nanosm::MmaShape{16, 64, 8},
             nanosm::MmaShape{128, 64, 8}}) {
        const auto machine = run_shape(profile, shape);
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

    auto partial = canonical_profile();
    partial.mma_core_k = 4;
    partial.mma_shapes = {{16, 16, 4}};
    nanosm::validate_device_profile(partial);
    const auto warnings = nanosm::device_profile_warnings(partial);
    require(warnings.size() == 1, "one partial-K-block warning for m16n16k4");
    require_mentions(warnings[0], {"mma shape m16n16k4", "reads only 4 of the 8 fp16", "50%"},
        "the partial-K-block warning");
    require(nanosm::device_profile_warnings(canonical_profile()).empty(), "the canonical profile warns");
    const auto run_partial = [&](uint32_t a_base) {
        nanosm::Machine machine(partial, compile("umov ur0, " + std::to_string(a_base) +
            "\numov ur1, 0\nmma.async.init.m16n16k4 [ur8], [ur0], [ur1]\nhalt\n", partial), launch_warps(1));
        machine.run(10000);
    };
    expect_error<nanosm::SimError>([&] { run_partial(65536 - 128); },
        {"MMA operand alignment or footprint is outside shmem"}, "an A operand whose last byte read is past shmem");
    run_partial(65536 - 256); // the whole 256-byte object fits: admitted and run
}

// The latency bound at equality. With result latency 10 the bound is
// D 2 + Lb 2 + 10 + Dt 2 = 16, and m16n16k16 revisits each block every
// (16 / 4) x (16 / 4) = 16 steps: accepted, and it never stalls. The step
// that revisits a block starts its operand rounds no earlier than the cycle
// the block's previous result lands. The 16 steps between the visits take
// 16 + 4 rounds (each m block's A reload is one more), so a visit that is
// itself a reload (2 rounds) leaves 4 - 1 = 3 cycles and any other leaves 4:
// the validated bound counts one round a step and is short of exact by the
// reloads. With result latency 11 the bound is 17 and the shape is refused.
TEST(Mma, latency_bound) {
    const nanosm::MmaShape shape{16, 16, 16};
    const auto profile = shape_profile({{64, 64, 8}, shape}, 10);
    require(nanosm::mma_min_blocks_per_k(profile) == 16 && nanosm::mma_blocks_per_k(profile, shape) == 16,
        "the bound or the blocks per K block");
    const auto machine = run_shape(profile, shape);
    const auto inputs = events_named(*machine, "MMA input step");
    const auto writes = events_named(*machine, "MMA tmem write");
    require(inputs.size() == 32 && writes.size() == 32 && machine->trace().summary().mma_recurrence_stalls == 0,
        "m16n16k16 is 32 steps with no recurrence stall");
    int64_t least = std::numeric_limits<int64_t>::max();
    for (uint32_t s = 0; s < 16; ++s) {
        const auto slack = static_cast<int64_t>(inputs[s + 16].start_cycle) - static_cast<int64_t>(writes[s].start_cycle);
        const bool reload = s % 4 == 0;
        require(slack == (reload ? 3 : 4), "step " + std::to_string(s + 16) + " revisits its block with slack " +
            std::to_string(slack) + " (expected " + std::to_string(reload ? 3 : 4) + ")");
        least = std::min(least, slack);
    }
    require(least == 3, "the least slack is the reloads' extra rounds less one");
    auto slower = profile;
    slower.mma_result_latency_cycles = 11;
    expect_error<nanosm::ProfileError>([&] { nanosm::validate_device_profile(slower); },
        {"m16n16k16", "revisited every 16 steps", "needs at least 17"}, "the bound one past equality");
}

// The layout (src/layout.hpp tmem_physical_word) against an independent
// statement of the rule as bit fields: with S = 2^s words per row, logical
// word L = [band | row in band (2) | column / 4 (s - 2) | column % 4 (2)] is
// stored at [band | column / 4 | row in band | column % 4 XOR band % 4]. Then
// the properties the fine banks need, for every S: each band of four rows is
// a permutation of itself; every 4x4 macro tile, and every 16 consecutive
// rows of one column wherever they start, hit 16 different fine banks (word
// % 16). And a region at a multiple of 16 rows is the macro-tile layout
// relative to its base.
TEST(Mma, tmem_layout_matches_the_rule) {
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
    // A region of `cols`-word rows at `base`: 4x4 macro tiles row-major from
    // the base, inside one the word at 4 x (row % 4) + (col % 4 XOR row / 4 % 4).
    const auto macro_tile_rule = [](uint32_t base, uint32_t cols, uint32_t row, uint32_t col) {
        return base + ((row / 4) * (cols / 4) + col / 4) * 16 + 4 * (row % 4) + ((col % 4) ^ ((row / 4) % 4));
    };
    for (const auto& [base, cols] : {std::pair{0u, 64u}, std::pair{2048u, 64u}, std::pair{4096u, 64u},
             std::pair{6144u, 64u}, std::pair{4096u, 16u}, std::pair{256u, 16u}}) {
        for (uint32_t row = 0; row < 32; ++row)
            for (uint32_t col = 0; col < cols; ++col)
                require(nanosm::tmem_physical_word(base + row * cols + col, cols) == macro_tile_rule(base, cols, row, col),
                    "a region at " + std::to_string(base) + " is not the macro-tile layout from its base");
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
TEST(Mma, acc_pointer_at_a_row_and_column) {
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
    nanosm::Machine machine(profile, compile(source, profile), launch_warps(1));
    machine.run(20000);
    for (uint32_t lane = 0; lane < 16; ++lane) {
        for (uint32_t j = 0; j < 8; ++j) {
            const uint32_t stored = lane + 1 + 1000 * j;
            require(result_word(machine, 1304, 64, lane, j) == stored,
                "row " + std::to_string(20 + lane) + " column " + std::to_string(24 + j) + " is not where the rule puts it");
            require(machine.read_reg(0, lane, static_cast<uint8_t>(40 + j)) == (lane < 4 ? 0 : stored - 4),
                "the acc.ld.v8 at row 16, column 24, lane " + std::to_string(lane));
        }
        require(machine.read_reg(0, lane, 50) == (lane < 4 ? 12 + lane + 1 + 7000 : 0),
            "the acc.ld at row 32, column 31, lane " + std::to_string(lane));
    }
}

// An mma.async reads three uniform registers: urA and urB in its issue round
// (issue + 1), urC in a second round at issue + 2 (src/timing.hpp
// uniform_read_rounds). At the canonical depths it still reaches the MMA at
// issue + 3 (issue_to_dispatch). A uniform reader right behind it is refused
// once ("uniform read ports held") and issues at issue + 2; so is a second
// matmul. A vector instruction is not refused, nor is a uniform write that
// reads nothing -- even one that rewrites urC at issue + 1: it lands at
// issue + 5, after the second round read the old pointer.
TEST(Mma, mma_three_uniform_reads) {
    const auto& profile = canonical_profile();
    nanosm::Instruction mma;
    mma.opcode = nanosm::mma_opcode(0, true);
    mma.src0 = 1;
    mma.src1 = 2;
    mma.src2 = 3;
    require(nanosm::uniform_read_rounds(mma) == std::vector<std::vector<uint8_t>>{{1, 2}, {3}},
        "an mma.async reads urA, urB, then urC");
    require(nanosm::issue_to_dispatch(profile) == 3, "canonical: an mma.async reaches the MMA at issue + 3");
    const std::string setup = R"(
        umov ur0, 0
        umov ur1, 2048
        umov ur2, 0
        umov ur3, 5
        umov ur4, 8192
        mbar.init [ur4], 1
    )";
    const std::string tail = "mma.commit [ur4]\n" + wait_on("ur4") + "halt\n";
    const auto machine_for = [&](const std::string& body) {
        auto launch = launch_warps(1);
        write_small_operands(image_bytes(launch.test_shmem, 4096));
        return std::make_unique<nanosm::Machine>(profile, compile(setup + body + tail, profile), std::move(launch),
            nanosm::TraceOptions{.timeline = true});
    };
    // pcs: the setup is 6 instructions, so the matmul is pc 48.
    auto refused = machine_for(R"(
        mma.async.init.m64n64k8 [ur2], [ur0], [ur1]
        uaddi ur5, ur3, 1
        mma.async.init.m64n64k8 [ur2], [ur0], [ur1]
    )");
    refused->run(40000);
    const auto first = issued_at(*refused, 48).at(0);
    require(issued_at(*refused, 56).at(0) == first + 2, "the uniform reader behind the matmul was not refused once");
    require(issued_at(*refused, 64).at(0) == first + 3, "the second matmul did not issue the cycle after the reader");
    std::vector<uint64_t> accepts;
    for (const auto& event : refused->trace().events()) {
        if (event.category == nanosm::TraceCategory::mma && event.name.starts_with("mma.async"))
            accepts.push_back(event.start_cycle);
    }
    require(accepts == std::vector<uint64_t>{first + 3, first + 6}, "the matmuls did not reach the MMA at issue + 3");
    // Refused for the ports: the reader at T + 1 and the commit at T + 4 (the
    // second matmul's later round); the commit then waits for MMA queue room.
    require(refused->trace().summary().urf_read_ports_held == 2 && refused->read_ureg(0, 5) == 6,
        "two refused warp-cycles, and the reader's result");
    require(result_at(*refused, 0, 64, 5, 7) == product_reference(5, 7), "the matmuls' result");

    auto free_ports = machine_for(R"(
        mma.async.init.m64n64k8 [ur2], [ur0], [ur1]
        umov ur2, 4096
        mov r1, 7
    )");
    free_ports->run(40000);
    const auto matmul = issued_at(*free_ports, 48).at(0);
    require(issued_at(*free_ports, 56).at(0) == matmul + 1 && issued_at(*free_ports, 64).at(0) == matmul + 2,
        "the uniform write that reads nothing or the vector mov waited");
    require(free_ports->trace().summary().urf_read_ports_held == 0, "nothing was refused");
    for (uint32_t row = 0; row < 64; row += 9) {
        require(result_at(*free_ports, 0, 64, row, 3) == product_reference(row, 3) &&
                result_word(*free_ports, 4096, 64, row, 3) == 0,
            "the matmul did not read the old urC (0) in its second round");
    }
}

// Where a pointer may point (src/layout.hpp). The machine addresses any
// pointer by the layout rule; a region with a word outside tmem is its PANIC
// (both builds), as an operand outside shmem is.
TEST(Mma, tmem_region_outside_tmem_panics) {
    const auto& profile = canonical_profile();
    nanosm::Machine load(profile, compile("umov ur8, 7232\nacc.ld r1, [ur8, 64]\nhalt\n", profile), launch_warps(1));
    expect_panic(load, {"accumulator transfer region outside tmem"}, "an acc.ld whose rows 113.. leave tmem");
    nanosm::Machine matmul(profile, compile(
        "umov ur0, 0\numov ur1, 2048\numov ur2, 6144\nmma.async.init.m64n64k8 [ur2], [ur0], [ur1]\nhalt\n", profile),
        launch_warps(1));
    expect_panic(matmul, {"MMA result region is outside tmem"}, "a 64-row result at row 96");
}

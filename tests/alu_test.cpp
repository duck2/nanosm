// Integer and float ALU, SFU, conversions.
#include "test_support.hpp"

using namespace nanosm_test;

namespace {

// The FPU / SFU menu on one warp: lane l reads a[l], b[l], c[l] from shmem 0,
// 64 and 128 and writes fmul r5, fadd r6, fsub r13, fmax r7, fma r8,
// cvt.pack r14, ex2 r10, rcp r11.
const char* const fpu_menu = R"(
    sread r0, LANE_ID
    shli r0, r0, 2
    lds r1, [r0]
    lds r2, [r0+64]
    lds r3, [r0+128]
    fmul r5, r1, r2
    fadd r6, r1, r2
    fsub r13, r1, r2
    fmax r7, r1, r2
    fma r8, r1, r2, r3
    cvt.pack.f16x2 r14, r1, r2
    ex2 r10, r1
    rcp r11, r1
    halt
)";

std::unique_ptr<nanosm::Machine> run_fpu_menu(
    const std::array<uint32_t, 16>& a, const std::array<uint32_t, 16>& b, const std::array<uint32_t, 16>& c) {
    auto launch = launch_warps(1);
    auto& shmem = image_bytes(launch.test_shmem, 256);
    for (uint32_t lane = 0; lane < 16; ++lane) {
        write_u32(shmem, lane * 4, a[lane]);
        write_u32(shmem, 64 + lane * 4, b[lane]);
        write_u32(shmem, 128 + lane * 4, c[lane]);
    }
    auto machine = std::make_unique<nanosm::Machine>(canonical_profile(), compile(fpu_menu), std::move(launch));
    machine->run(10000);
    return machine;
}

} // namespace

TEST(Alu, integer_alu_menu) {
    // Toolchain round trip for the eight integer ALU ops.
    const auto program = qtx_program(R"(
        sread r0, LANE_ID
        lds r1, [r0]
        lds r2, [r0+64]
        and r4, r1, r2
        or  r5, r1, r2
        xor r7, r1, r2
        shl r8, r1, r2
        shr r10, r1, r2
        andi r11, r1, 255
        ori r13, r1, 4080
        shri r14, r1, 7
        halt
    )", canonical_profile());
    const auto assembled = nanosm::assemble_asm(nanosm::emit_asm(program), canonical_profile());
    const auto reloaded = nanosm::load_hex(nanosm::emit_hex(assembled));
    require(assembled.words == reloaded.words, "asm/hex round trip changed integer ALU words");
    std::set<nanosm::Opcode> seen;
    for (const auto word : assembled.words) seen.insert(nanosm::decode(word).opcode);
    for (const auto opcode : {nanosm::Opcode::and_, nanosm::Opcode::or_, nanosm::Opcode::xor_,
             nanosm::Opcode::shl, nanosm::Opcode::shr, nanosm::Opcode::andi,
             nanosm::Opcode::ori, nanosm::Opcode::shri}) {
        require(seen.count(opcode) == 1, "an integer ALU opcode did not survive the toolchain");
    }

    // Per-lane numeric vectors. B holds register shift amounts including 32
    // and 45: the 5-bit shifter uses amount mod 32, so 32 acts as 0 and 45 as 13.
    const std::array<uint32_t, 16> a{
        0xffffffffu, 0x80000000u, 0xdeadbeefu, 0x00000001u,
        0x12345678u, 0xa5a5a5a5u, 0x0f0f0f0fu, 0xfffffffeu,
        0x00000000u, 0x7fffffffu, 0xcafebabeu, 0x55555555u,
        0x89abcdefu, 0x40000000u, 0x00010000u, 0xffff0000u};
    const std::array<uint32_t, 16> b{
        0, 31, 1, 31,
        4, 8, 16, 1,
        13, 30, 32, 45,   // 32 -> 0, 45 -> 13 under the 5-bit shifter
        2, 62, 33, 95};   // 62 -> 30, 33 -> 1, 95 -> 31
    auto launch = launch_warps(1);
    auto& image = image_bytes(launch.test_shmem, 128);
    for (uint32_t lane = 0; lane < 16; ++lane) {
        write_u32(image, lane * 4, a[lane]);
        write_u32(image, 64 + lane * 4, b[lane]);
    }
    nanosm::Machine machine(canonical_profile(), compile(R"(
        sread r0, LANE_ID
        shli r0, r0, 2
        lds r1, [r0]
        lds r2, [r0+64]
        and r4, r1, r2
        or  r5, r1, r2
        xor r7, r1, r2
        shl r8, r1, r2
        shr r10, r1, r2
        andi r11, r1, 255
        ori r13, r1, 4080
        shri r14, r1, 5
        shri r16, r1, 0
        shri r17, r1, 31
        halt
    )"), std::move(launch));
    machine.run(20000);
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(machine.read_reg(0, lane, 4) == (a[lane] & b[lane]), "and lane mismatch");
        require(machine.read_reg(0, lane, 5) == (a[lane] | b[lane]), "or lane mismatch");
        require(machine.read_reg(0, lane, 7) == (a[lane] ^ b[lane]), "xor lane mismatch");
        require(machine.read_reg(0, lane, 8) == (a[lane] << (b[lane] & 31)), "shl lane mismatch (5-bit amount rule)");
        require(machine.read_reg(0, lane, 10) == (a[lane] >> (b[lane] & 31)), "shr lane mismatch (5-bit amount rule)");
        require(machine.read_reg(0, lane, 11) == (a[lane] & 255u), "andi lane mismatch");
        require(machine.read_reg(0, lane, 13) == (a[lane] | 4080u), "ori lane mismatch");
        require(machine.read_reg(0, lane, 14) == (a[lane] >> 5), "shri lane mismatch");
        require(machine.read_reg(0, lane, 16) == a[lane], "shri by 0 changed the value");
        require(machine.read_reg(0, lane, 17) == (a[lane] >> 31), "shri by 31 mismatch");
    }

    // An immediate shift outside 0..31 compiles and assembles, and PANICs
    // when it runs.
    nanosm::Machine bad(canonical_profile(), compile("mov r0, 1\nshri r2, r0, 32\nhalt"), launch_warps(1));
    expect_panic(bad, {"shift outside 0..31"}, "an out-of-range shri immediate");
}

TEST(Alu, half_conversion_and_pack_vectors) {
    // Every half value round-trips through fp32 exactly: finite and infinite
    // by value, NaNs by preserved payload (<<13 then >>13).
    for (uint32_t h = 0; h < 0x10000; ++h) {
        const auto half = static_cast<uint16_t>(h);
        const auto back = nanosm::float_to_half_bits(nanosm::half_to_float_bits(half));
        require(back == half, "half -> float -> half round trip failed");
    }
    // Directed fp32 -> fp16 pins: RNE ties, overflow, subnormals, NaN, zeros.
    const std::array<std::pair<uint32_t, uint16_t>, 12> pins{{
        {0x3f800000u, 0x3c00}, // 1.0
        {0x3f801000u, 0x3c00}, // 1 + 2^-11: tie to even (down)
        {0x3f801800u, 0x3c01}, // 1 + 3*2^-12: rounds up
        {0x477fe000u, 0x7bff}, // 65504, half max
        {0x477ff000u, 0x7c00}, // 65520: tie rounds to infinity
        {0x33000000u, 0x0000}, // 2^-25: tie to even zero
        {0x33400000u, 0x0001}, // 1.5 * 2^-25 rounds to smallest subnormal
        {0xb3800000u, 0x8001}, // -2^-24
        {0x7f800000u, 0x7c00}, // +inf
        {0x7fc00000u, 0x7e00}, // canonical qNaN
        {0x7f800001u, 0x7e00}, // signaling payload truncates to zero: quiet bit forced
        {0x80000000u, 0x8000}, // -0
    }};
    for (const auto& [bits, expected] : pins) {
        require(nanosm::float_to_half_bits(bits) == expected, "fp32 -> fp16 pin mismatch for " + std::to_string(bits));
    }
}

// The FPU and SFU menu: encodings, the toolchain round trip, and two vector
// sets through one kernel. The first set recomputes the spec algorithms
// locally (its marked lanes also pin literal result bits, so a platform libm
// drift in exp2 breaks loudly). The second is the subnormal rule: no
// subnormals in the FPU and the SFU, a subnormal input is read as zero of its
// sign and a subnormal result delivered as zero of its sign; fma rounds once
// (its product is not flushed); normal values, zeros and NaNs pass; cvt.pack
// keeps a half's subnormal.
TEST(Alu, fa_menu_numeric_vectors) {
    nanosm::Instruction fma;
    fma.opcode = nanosm::Opcode::fma;
    fma.control.stall = 19;
    fma.dst = 8;
    fma.src0 = 250;
    fma.src1 = 13;
    fma.src2 = 201;
    const auto decoded = nanosm::decode(nanosm::encode(fma));
    require(decoded.opcode == nanosm::Opcode::fma && decoded.dst == 8 && decoded.src0 == 250 && decoded.src1 == 13 &&
            decoded.src2 == 201,
        "fma four-register encoding mismatch");
    nanosm::Instruction unary;
    unary.opcode = nanosm::Opcode::ex2;
    unary.dst = 100;
    unary.src0 = 55;
    const auto unary_decoded = nanosm::decode(nanosm::encode(unary));
    require(unary_decoded.opcode == nanosm::Opcode::ex2 && unary_decoded.dst == 100 && unary_decoded.src0 == 55,
        "ex2 encoding mismatch");
    const auto program = qtx_program(fpu_menu, canonical_profile());
    const auto assembled = nanosm::assemble_asm(nanosm::emit_asm(program), canonical_profile());
    const auto reloaded = nanosm::load_hex(nanosm::emit_hex(assembled));
    require(assembled.words == reloaded.words, "asm/hex round trip changed the FA menu words");
    std::set<nanosm::Opcode> seen;
    for (const auto word : assembled.words) seen.insert(nanosm::decode(word).opcode);
    for (const auto opcode : {nanosm::Opcode::fmul, nanosm::Opcode::fadd, nanosm::Opcode::fsub, nanosm::Opcode::fmax,
             nanosm::Opcode::fma, nanosm::Opcode::cvt_pack, nanosm::Opcode::ex2, nanosm::Opcode::rcp}) {
        require(seen.count(opcode) == 1, "an FA menu opcode did not survive the toolchain");
    }

    constexpr uint32_t nan = 0x7fc00000u;
    {
        const std::array<uint32_t, 16> a{
            0x3f800000u, 0xbf800000u, 0x3f000000u, 0xbf000000u, // 1, -1, 0.5, -0.5
            0x40400000u, 0x477fe000u, 0x33800000u, 0x00000000u, // 3, 65504, 2^-24, +0
            0x80000000u, 0x7f800000u, 0xff800000u, nan,          // -0, +inf, -inf, NaN
            0x40200000u, 0xc30c0000u, 0xc3160000u, 0x3f800001u}; // 2.5, -140, -150, 1+2^-23
        const std::array<uint32_t, 16> b{
            0x40000000u, 0xc0400000u, 0x3f000000u, 0xbf000000u,
            0x3fc00000u, 0x3f800000u, 0x42000000u, 0x80000000u, // ... fmax(+0, -0)
            0x00000000u, nan, 0xc0000000u, 0x40a00000u,          // fmax(NaN, 5) = 5
            0xbf000000u, 0x3f800000u, 0x3f800000u, 0x3f800001u};
        const std::array<uint32_t, 16> c{
            0x3f800000u, 0x3f800000u, 0x3f800000u, 0x3f800000u,
            0x3f800000u, 0x3f800000u, 0x3f800000u, 0x3f800000u,
            0x3f800000u, 0x3f800000u, 0x3f800000u, 0x3f800000u,
            0x3f800000u, 0x3f800000u, 0x3f800000u, 0xbf800002u}; // fma single-rounding case
        const auto machine = run_fpu_menu(a, b, c);
        // NaN canonical; a subnormal result flushed to zero of its sign (no
        // input here is subnormal).
        const auto canonical = [](float value) {
            const uint32_t bits = nanosm::float_bits(value);
            return std::isnan(value) ? nan : (bits & 0x7f800000u) == 0 ? bits & 0x80000000u : bits;
        };
        const auto as_float = nanosm::bits_float;
        for (uint32_t lane = 0; lane < 16; ++lane) {
            const float fa = as_float(a[lane]);
            const float fb = as_float(b[lane]);
            const float fc = as_float(c[lane]);
            require(machine->read_reg(0, lane, 5) == canonical(fa * fb), "fmul vector mismatch");
            require(machine->read_reg(0, lane, 13) == canonical(fa - fb), "fsub vector mismatch");
            const uint32_t expected_max = (std::isnan(fa) && std::isnan(fb)) ? nan
                : std::isnan(fa) ? b[lane]
                : std::isnan(fb) ? a[lane]
                : (fa == 0.0f && fb == 0.0f)
                    ? ((std::signbit(fa) && std::signbit(fb)) ? a[lane] : 0u)
                    : (fa > fb ? a[lane] : b[lane]);
            require(machine->read_reg(0, lane, 7) == expected_max, "fmax vector mismatch");
            require(machine->read_reg(0, lane, 8) == canonical(std::fma(fa, fb, fc)), "fma vector mismatch");
            const uint32_t expected_pack = static_cast<uint32_t>(nanosm::float_to_half_bits(a[lane])) |
                static_cast<uint32_t>(nanosm::float_to_half_bits(b[lane])) << 16;
            require(machine->read_reg(0, lane, 14) == expected_pack, "cvt.pack vector mismatch");
            const uint32_t expected_ex2 = std::isnan(fa) ? nan
                : canonical(static_cast<float>(std::exp2(static_cast<double>(fa))));
            require(machine->read_reg(0, lane, 10) == expected_ex2, "ex2 vector mismatch");
            const uint32_t expected_rcp = std::isnan(fa) ? nan
                : canonical(static_cast<float>(1.0 / static_cast<double>(fa)));
            require(machine->read_reg(0, lane, 11) == expected_rcp, "rcp vector mismatch");
        }
        // Literal pins, independent of this test's own reference arithmetic.
        require(machine->read_reg(0, 0, 10) == 0x40000000u, "ex2(1) != 2");
        require(machine->read_reg(0, 2, 10) == 0x3fb504f3u, "ex2(0.5) != sqrt(2) bits");
        require(machine->read_reg(0, 3, 10) == 0x3f3504f3u, "ex2(-0.5) != 1/sqrt(2) bits");
        require(machine->read_reg(0, 13, 10) == 0x00000000u, "ex2(-140), a subnormal 2^-140, was not flushed to +0");
        require(machine->read_reg(0, 14, 10) == 0x00000000u, "ex2(-150) did not flush the tie to zero");
        require(machine->read_reg(0, 10, 10) == 0x00000000u, "ex2(-inf) != +0");
        require(machine->read_reg(0, 7, 11) == 0x7f800000u, "rcp(+0) != +inf");
        require(machine->read_reg(0, 8, 11) == 0xff800000u, "rcp(-0) != -inf");
        require(machine->read_reg(0, 9, 11) == 0x00000000u, "rcp(+inf) != +0");
        require(machine->read_reg(0, 7, 7) == 0x00000000u, "fmax(+0, -0) != +0");
        require(machine->read_reg(0, 11, 7) == 0x40a00000u, "fmax(NaN, 5) != 5");
        require(machine->read_reg(0, 15, 8) == 0x28800000u,
            "fma lost its single rounding: (1+2^-23)^2 - (1+2^-22) != 2^-46");
    }

    const std::array<uint32_t, 16> a{
        0x00400000u, 0x80400000u, 0x0d800000u, 0x8d800000u, // 2^-127, -2^-127, 2^-100, -2^-100
        0x00800001u, 0x00800001u, 0x1c800000u, 0x1c800000u, // 2^-126 (1 + 2^-23) twice, 2^-70 twice
        0x3fc00000u, 0x80000000u, 0x00800000u, 0xc3020000u, // 1.5, -0, 2^-126, -130
        0x7f000000u, 0x00000001u, 0x7fc00000u, 0x80000001u}; // 2^127, 2^-149, NaN, -2^-149
    const std::array<uint32_t, 16> b{
        0x00000000u, 0x00000000u, 0x30800000u, 0x30800000u, // +0, +0, 2^-30, 2^-30
        0x80800000u, 0x00800000u, 0x1c800000u, 0x1c800000u, // -2^-126, 2^-126, 2^-70, 2^-70
        0x40000000u, 0x80000000u, 0x3f800000u, 0x3f800000u, // 2, -0, 1, 1
        0x3f800000u, 0x7f000000u, 0x3f800000u, 0x35800000u}; // 1, 2^127, 1, 2^-20
    std::array<uint32_t, 16> c{};
    c[6] = 0x00800000u; // 2^-70 x 2^-70 + 2^-126
    c[8] = 0x3f800000u; // 1.5 x 2 + 1
    const auto machine = run_fpu_menu(a, b, c);
    struct Pin {
        uint32_t lane, reg, bits;
        const char* what;
    };
    for (const auto& pin : {
             Pin{0, 6, 0x00000000u, "fadd(2^-127, +0): a subnormal input is zero"},
             Pin{0, 7, 0x00000000u, "fmax(2^-127, +0): a subnormal input is zero"},
             Pin{0, 10, 0x3f800000u, "ex2(2^-127) = ex2(0)"},
             Pin{0, 11, 0x7f800000u, "rcp(2^-127) = rcp(+0) = +inf"},
             Pin{1, 11, 0xff800000u, "rcp(-2^-127) = rcp(-0) = -inf"},
             Pin{2, 5, 0x00000000u, "fmul 2^-100 x 2^-30: a subnormal result is +0"},
             Pin{2, 8, 0x00000000u, "fma 2^-100 x 2^-30 + 0: a subnormal result is +0"},
             Pin{3, 5, 0x80000000u, "fmul -2^-100 x 2^-30: a subnormal result is -0"},
             Pin{4, 6, 0x00000000u, "fadd cancelling to 2^-149 is +0"},
             Pin{5, 13, 0x00000000u, "fsub cancelling to 2^-149 is +0"},
             Pin{6, 8, 0x00800200u, "fma 2^-70 x 2^-70 + 2^-126: its product is not flushed"},
             Pin{7, 8, 0x00000000u, "fma 2^-70 x 2^-70 + 0: a subnormal result is +0"},
             Pin{8, 5, 0x40400000u, "fmul 1.5 x 2"},
             Pin{8, 6, 0x40600000u, "fadd 1.5 + 2"},
             Pin{8, 8, 0x40800000u, "fma 1.5 x 2 + 1"},
             Pin{9, 6, 0x80000000u, "fadd -0 + -0 = -0"},
             Pin{9, 5, 0x00000000u, "fmul -0 x -0 = +0"},
             Pin{9, 7, 0x80000000u, "fmax(-0, -0) = -0"},
             Pin{10, 5, 0x00800000u, "fmul 2^-126 x 1: the smallest normal passes"},
             Pin{11, 10, 0x00000000u, "ex2(-130): a subnormal result is +0"},
             Pin{12, 11, 0x00000000u, "rcp(2^127): a subnormal result is +0"},
             Pin{13, 5, 0x00000000u, "fmul 2^-149 x 2^127: a subnormal input is zero (not 2^-22)"},
             Pin{14, 5, 0x7fc00000u, "fmul NaN x 1 is the canonical NaN"},
             Pin{15, 14, 0x00108000u, "cvt.pack(-2^-149, 2^-20): -0 and a half's subnormal"},
         }) {
        const uint32_t got = machine->read_reg(0, pin.lane, pin.reg);
        require(got == pin.bits, std::string(pin.what) + ": got " + std::to_string(got));
    }
}

// The SFU: 4 lanes, so an ex2 or rcp of 16 lanes takes 4 passes and the unit
// takes a new op every 4 cycles; qtxc schedules its effective latency
// (occupancy 3 + latency 8 = 11). Two warps' four ex2 are admitted exactly 4
// cycles apart, while an independent ALU op issues beside the busy unit.
TEST(Alu, sfu_unit_timing) {
    const auto& profile = canonical_profile();
    require(profile.timing.sfu == 11, "canonical effective SFU latency is not 3 + 8");
    const auto scheduled = qtx_program(R"(
        sread r1, LANE_ID
        ex2 r2, r1
        fadd r4, r2, r2
        halt
    )", profile);
    // The ex2's result feeds the very next fadd: its whole distance
    // (src/timing.hpp), the effective SFU latency inside it.
    require(scheduled.instructions[1].control.stall ==
            nanosm::issue_to_visible(scheduled.instructions[1], profile) - nanosm::issue_register_stages,
        "qtxc did not schedule the effective SFU latency");

    nanosm::Machine machine(profile, compile(R"(
        sread r1, LANE_ID
        ex2 r2, r1
        add r5, r1, r1
        ex2 r3, r1
        halt
    )"), launch_warps(2), {.timeline = true});
    machine.run(10000);
    const auto first = issued_at(machine, 8);
    const auto second = issued_at(machine, 24);
    require(first.size() == 2 && second.size() == 2, "expected four SFU issues");
    std::vector<uint64_t> sfu_issues{first[0], first[1], second[0], second[1]};
    std::sort(sfu_issues.begin(), sfu_issues.end());
    for (std::size_t i = 1; i < sfu_issues.size(); ++i) {
        require(sfu_issues[i] - sfu_issues[i - 1] == 4,
            "SFU admissions were not 4 cycles apart: " + std::to_string(sfu_issues[i] - sfu_issues[i - 1]));
    }
    const auto adds = issued_at(machine, 16);
    require(adds.size() == 2, "expected one add per warp");
    for (uint32_t warp = 0; warp < 2; ++warp) {
        require(adds[warp] > first[warp] && adds[warp] < first[warp] + 4, "an independent ALU op waited on the busy SFU");
    }
    const auto& summary = machine.trace().summary();
    require(summary.sfu_ops == 4 && summary.sfu_busy_cycles == 16, "SFU counters did not record four quarter-rate ops");
    require(summary.sfu_unit_stalls > 0, "SFU contention produced no counted stalls");
    for (uint32_t warp = 0; warp < 2; ++warp)
        for (uint32_t lane = 0; lane < 16; ++lane)
            require(machine.read_reg(warp, lane, 5) == 2 * lane, "the interleaved ALU op computed the wrong value");
}

TEST(Alu, sfu_profile_validation) {
    std::ifstream input("profiles/nanosm-v0.json");
    nlohmann::json document;
    input >> document;
    require(nanosm::parse_device_profile(document.dump()).timing.sfu == 11,
        "canonical sfu occupancy + latency did not derive 11");
    const auto refused = [&](const std::string& field, const nlohmann::json& value) {
        auto modified = document;
        modified["sfu"][field] = value;
        return error_of<nanosm::ProfileError>([&] { (void)nanosm::parse_device_profile(modified.dump()); },
            "sfu." + field + " = " + value.dump());
    };
    for (const auto& value : {nlohmann::json(0), nlohmann::json(17), nlohmann::json(2.5), nlohmann::json(-1)})
        (void)refused("lanes", value);
    for (const auto& value : {nlohmann::json(0), nlohmann::json(25), nlohmann::json(1.5)})
        (void)refused("latency_cycles", value);
    // A required integer out of its type's range (or negative, or a
    // fraction) is refused by name, never narrowed (alu_latency_cycles is 8
    // bits: 257 would read as 1).
    for (const auto& [section, name, value] : std::vector<std::tuple<std::string, std::string, nlohmann::json>>{
             {"core", "alu_latency_cycles", 257}, {"lsu", "active_commands", 4294967297ull},
             {"shmem", "banks", -16}, {"mma", "queue_commands", 2.5}}) {
        auto modified = document;
        modified[section][name] = value;
        const auto what = name + " = " + value.dump();
        expect_error<nanosm::ProfileError>([&] { (void)nanosm::parse_device_profile(modified.dump()); },
            {"device profile field '" + name + "' must be an integer in 0.."}, what);
    }
    auto faster = document;
    faster["sfu"]["lanes"] = 16;
    require(nanosm::device_profile_fingerprint(nanosm::parse_device_profile(faster.dump())) !=
            nanosm::device_profile_fingerprint(nanosm::parse_device_profile(document.dump())),
        "sfu throughput was omitted from the profile fingerprint");
}

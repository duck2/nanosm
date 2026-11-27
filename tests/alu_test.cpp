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
    require_lanes(machine, 4, [&](uint32_t lane) { return a[lane] & b[lane]; }, "and");
    require_lanes(machine, 5, [&](uint32_t lane) { return a[lane] | b[lane]; }, "or");
    require_lanes(machine, 7, [&](uint32_t lane) { return a[lane] ^ b[lane]; }, "xor");
    require_lanes(machine, 8, [&](uint32_t lane) { return a[lane] << (b[lane] & 31); }, "shl (5-bit amount rule)");
    require_lanes(machine, 10, [&](uint32_t lane) { return a[lane] >> (b[lane] & 31); }, "shr (5-bit amount rule)");
    require_lanes(machine, 11, [&](uint32_t lane) { return a[lane] & 255u; }, "andi");
    require_lanes(machine, 13, [&](uint32_t lane) { return a[lane] | 4080u; }, "ori");
    require_lanes(machine, 14, [&](uint32_t lane) { return a[lane] >> 5; }, "shri");
    require_lanes(machine, 16, [&](uint32_t lane) { return a[lane]; }, "shri by 0 changed the value");
    require_lanes(machine, 17, [&](uint32_t lane) { return a[lane] >> 31; }, "shri by 31");

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
            require(machine->read_reg(0, lane, 5) == canonical(fa * fb), "fmul vector mismatch");
            require(machine->read_reg(0, lane, 13) == canonical(fa - fb), "fsub vector mismatch");
            const uint32_t expected_max = (std::isnan(fa) && std::isnan(fb)) ? nan
                : std::isnan(fa) ? b[lane]
                : std::isnan(fb) ? a[lane]
                : (fa == 0.0f && fb == 0.0f)
                    ? ((std::signbit(fa) && std::signbit(fb)) ? a[lane] : 0u)
                    : (fa > fb ? a[lane] : b[lane]);
            require(machine->read_reg(0, lane, 7) == expected_max, "fmax vector mismatch");
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
        require_eq(machine->read_reg(0, pin.lane, pin.reg), pin.bits, pin.what);
    }
}

// fma is a x b + c rounded once, nearest even, by the simulator's own
// arithmetic. The expected bits were derived independently: the exact value
// with Python's fractions.Fraction, rounded to binary32 by hand (gradual
// underflow), then the machine's rule (a subnormal result is zero of its
// sign). The last three rows are IEEE's special cases.
TEST(Alu, fma_rounds_once) {
    struct Row {
        uint32_t a, b, c, expected;
        const char* what;
    };
    const std::vector<Row> rows = {
        {0x40200000u, 0x3fedb6dbu, 0x40955555u, 0x4114f3cfu, "2.5 x 1.8571428 + 4.6666665 (a host fmaf's miss)"},
        {0x3f800001u, 0x3f800001u, 0xbf800002u, 0x28800000u, "(1+2^-23)^2 - (1+2^-22) = 2^-46: massive cancellation"},
        {0x3f8ccccdu, 0x3f8ccccdu, 0xbf9ae148u, 0x3275c290u, "1.1f^2 - round(1.1f^2): the product's rounding error"},
        {0x3f800800u, 0x3f800800u, 0x00000000u, 0x3f801000u, "(1+2^-12)^2: a tie, to even (down)"},
        {0x3f800800u, 0x3f800800u, 0x34000000u, 0x3f801002u, "(1+2^-12)^2 + 2^-23: a tie, to even (up)"},
        {0x7f7fffffu, 0x3f800000u, 0x73000000u, 0x7f800000u, "max + half its ulp: a tie, to even, overflows to inf"},
        {0x7f000000u, 0x40000000u, 0x00000000u, 0x7f800000u, "2^127 x 2 overflows to inf"},
        {0xff000000u, 0x40000000u, 0x3f800000u, 0xff800000u, "-2^127 x 2 + 1 overflows to -inf"},
        {0x1c800000u, 0x1c800000u, 0x00000000u, 0x00000000u, "2^-70 x 2^-70 = 2^-140, subnormal: +0"},
        {0x9c800000u, 0x1c800000u, 0x00000000u, 0x80000000u, "-2^-70 x 2^-70: -0"},
        {0x1a000000u, 0x9a000000u, 0x00800000u, 0x00800000u, "2^-126 - 2^-150: a tie in the subnormal range rounds to 2^-126"},
        {0x3f800000u, 0x3f800000u, 0xbf800000u, 0x00000000u, "1 x 1 - 1: an exact zero is +0"},
        {0x80000000u, 0x3f800000u, 0x80000000u, 0x80000000u, "-0 x 1 + -0 = -0"},
        {0x40400000u, 0x3eaaaaabu, 0xbf800000u, 0x33000000u, "3 x round(1/3) - 1 = 2^-25 exactly"},
        {0xbfd113bbu, 0xc031cf30u, 0xc018099eu, 0x400a6617u, "fused and unfused differ"},
        {0x3ffe2f42u, 0xbf6eed33u, 0x402c8b88u, 0x3f57b739u, "fused and unfused differ"},
        {0xc0214564u, 0xc07aae5fu, 0xc0604c04u, 0x40cbb137u, "fused and unfused differ"},
        {0xbf80a054u, 0x3fca332au, 0x3f90f726u, 0xbee8e523u, "fused and unfused differ"},
        {0xc02f1ddbu, 0xc01ab1dcu, 0xbf356714u, 0x40bcf65du, "fused and unfused differ"},
        {0xc0436550u, 0xc07b8f25u, 0x3ffb51acu, 0x415f6bcdu, "fused and unfused differ"},
        {0x3f800800u, 0x3f800800u, 0x17800000u, 0x3f801001u, "just above a tie: a double fma then a narrowing would round down"},
        {0x7f800000u, 0x00000000u, 0x3f800000u, 0x7fc00000u, "inf x 0 + 1 is NaN"},
        {0x7f800000u, 0x3f800000u, 0xff800000u, 0x7fc00000u, "inf x 1 - inf is NaN"},
        {0x3f800000u, 0x7fc00000u, 0x3f800000u, 0x7fc00000u, "1 x NaN + 1 is the canonical NaN"},
    };
    for (std::size_t first = 0; first < rows.size(); first += 16) {
        std::array<uint32_t, 16> a{}, b{}, c{};
        for (std::size_t i = first; i < std::min(rows.size(), first + 16); ++i) {
            a[i - first] = rows[i].a;
            b[i - first] = rows[i].b;
            c[i - first] = rows[i].c;
        }
        const auto machine = run_fpu_menu(a, b, c);
        for (std::size_t i = first; i < std::min(rows.size(), first + 16); ++i) {
            require_eq(machine->read_reg(0, static_cast<uint32_t>(i - first), 8), rows[i].expected, rows[i].what);
        }
    }
}

// One source from a uniform register (src/isa.hpp takes_uniform_source):
// the U bit round-trips through encode, decode, the text and hex; the
// values of fadd, fsub, fmul, fma (the multiplicand written first or
// second), shl, setp and cvt.pack with a uniform second source, masked and
// not.
TEST(Alu, uniform_second_source) {
    const auto& profile = canonical_profile();
    nanosm::Instruction fma;
    fma.opcode = nanosm::Opcode::fma;
    fma.dst = 8;
    fma.src0 = 250;
    fma.src1 = 31;
    fma.src2 = 201;
    fma.src1_uniform = true;
    const auto decoded = nanosm::decode(nanosm::encode(fma));
    require(decoded.src1_uniform && decoded.src0 == 250 && decoded.src1 == 31 && decoded.src2 == 201,
        "fma's uniform multiplicand did not survive encode / decode");
    require(nanosm::format_instruction(decoded, false) == "fma r8, r250, ur31, r201", "fma's uniform text");
    nanosm::Instruction immediate;
    immediate.opcode = nanosm::Opcode::addi;
    immediate.src1_uniform = true;
    expect_error([&] { (void)nanosm::encode(immediate); }, {"addi has no uniform source"}, "a U bit on addi");

    const std::string source = R"(
        sread r0, LANE_ID
        shli r9, r0, 2
        lds r1, [r9]
        lds r2, [r9+64]
        umov ur1, 1.5
        umov ur2, 3
        fadd r3, r1, ur1
        fsub r4, r1, ur1
        fmul r5, r1, ur1
        fma r6, ur1, r1, r2
        fma r7, r1, ur1, r2
        shl r8, r0, ur2
        setp.lt.u32 p0, r0, ur2
        mov r10, 100
        mov r11, 0
        @p0 add r10, r0, ur2
        @!p0 fmul r11, r1, ur1
        cvt.pack.f16x2 r12, r1, ur1
        halt
    )";
    const auto kernel = compile(source, profile);
    const std::vector<std::string> expected = {"fadd r3, r1, ur1", "fsub r4, r1, ur1", "fmul r5, r1, ur1",
        "fma r6, r1, ur1, r2", "fma r7, r1, ur1, r2", "shl r8, r0, ur2", "setp.lt.u32 p0, r0, ur2",
        "add r10, r0, ur2", "fmul r11, r1, ur1", "cvt.pack.f16x2 r12, r1, ur1"};
    std::vector<std::string> texts;
    for (const auto word : kernel.words) {
        const auto inst = nanosm::decode(word);
        if (inst.src1_uniform) texts.push_back(nanosm::format_instruction(inst, false));
        require(nanosm::encode(inst) == word, "encode / decode changed a word");
    }
    require(texts == expected, "the uniform sources did not reach the words as written");
    require(nanosm::load_hex(nanosm::emit_hex(kernel)).words == kernel.words, "hex round trip");

    auto launch = launch_warps(1);
    auto& shmem = image_bytes(launch.test_shmem, 128);
    for (uint32_t lane = 0; lane < 16; ++lane) {
        write_u32(shmem, lane * 4, nanosm::float_bits(static_cast<float>(lane) + 0.25f));
        write_u32(shmem, 64 + lane * 4, nanosm::float_bits(2.0f * static_cast<float>(lane)));
    }
    nanosm::Machine machine(profile, kernel, std::move(launch));
    machine.run(20000);
    for (uint32_t lane = 0; lane < 16; ++lane) {
        const float a = static_cast<float>(lane) + 0.25f;
        const float c = 2.0f * static_cast<float>(lane);
        const auto at = [&](uint32_t reg) { return machine.read_reg(0, lane, reg); };
        const std::string where = "lane " + std::to_string(lane) + ": ";
        require(at(3) == nanosm::float_bits(a + 1.5f), where + "fadd x, U");
        require(at(4) == nanosm::float_bits(a - 1.5f), where + "fsub x, U is x - U");
        require(at(5) == nanosm::float_bits(a * 1.5f), where + "fmul x, U");
        // a x 1.5 + c is exact in single precision for these values.
        const auto product_sum = nanosm::float_bits(static_cast<float>(static_cast<double>(a) * 1.5 + c));
        require(at(6) == product_sum && at(7) == product_sum, where + "fma with a uniform multiplicand, either place");
        require(at(8) == lane << 3, where + "shl by a uniform count");
        require(at(10) == (lane < 3 ? lane + 3 : 100u), where + "setp against U, then add under @p0");
        require(at(11) == (lane < 3 ? 0u : nanosm::float_bits(a * 1.5f)), where + "fmul under @!p0");
        require(at(12) == (static_cast<uint32_t>(nanosm::float_to_half_bits(nanosm::float_bits(a))) |
                           static_cast<uint32_t>(nanosm::float_to_half_bits(nanosm::float_bits(1.5f))) << 16),
            where + "cvt.pack with a uniform high half");
    }
}

// The float compare (setp.f32, on the FPU), fmin, mul24 and sub in one
// kernel. Each setp.f32 is followed at once by a mov under its predicate, so
// the reader waits for the FPU's latency, not the ALU's. Then an FPU setp and
// an ALU setp three issues later present their predicates in the same cycle
// (checked from the issue cycles), each landing in its own predicate. The
// expected compare bits and fmin words are written out by hand.
TEST(Alu, float_compare_fmin_mul24_sub) {
    const auto& profile = canonical_profile();
    const std::array<uint32_t, 16> a{
        0x3f800000u, 0x40000000u, 0x3fc00000u, 0x80000000u, // 1, 2, 1.5, -0
        0x7fc00001u, 0x3f800000u, 0xffc00000u, 0x00000001u, // NaN, 1, NaN, 2^-149
        0x00400000u, 0x80400000u, 0x7f800000u, 0xff800000u, // 2^-127, -2^-127, +inf, -inf
        0xc0000000u, 0x7f7fffffu, 0x3f800000u, 0xbf800000u}; // -2, max, 1, -1
    const std::array<uint32_t, 16> b{
        0x40000000u, 0x3f800000u, 0x3fc00000u, 0x00000000u, // 2, 1, 1.5, +0
        0x3f800000u, 0x7fc00000u, 0x7f800001u, 0x00000000u, // 1, NaN, NaN, +0
        0x00000001u, 0x00000000u, 0x7f7fffffu, 0xbf800000u, // 2^-149, +0, max, -1
        0xc0400000u, 0x7f800000u, 0x3f800001u, 0x3f800000u}; // -3, +inf, 1 + 2^-23, 1
    // Per lane: eq ne lt le gt ge (a against b), then lt against U = 1.0.
    const std::array<const char*, 16> compares{
        "011100" "0", "010011" "0", "100101" "0", "100101" "1",
        "010000" "0", "010000" "0", "010000" "0", "100101" "1",
        "100101" "1", "100101" "1", "010011" "0", "011100" "1",
        "010011" "1", "011100" "0", "011100" "0", "011100" "1"};
    const std::array<uint32_t, 16> minimum{
        0x3f800000u, 0x3f800000u, 0x3fc00000u, 0x80000000u, 0x3f800000u, 0x3f800000u, 0x7fc00000u, 0x00000000u,
        0x00000000u, 0x80000000u, 0x7f7fffffu, 0xff800000u, 0xc0400000u, 0x7f7fffffu, 0x3f800000u, 0xbf800000u};
    const std::string source = R"(
        sread r0, LANE_ID
        shli r9, r0, 2
        lds r1, [r9]
        lds r2, [r9+64]
        umov ur1, 1.0
        umov ur2, 5
        mov r10, 0
        mov r11, 0
        mov r12, 0
        mov r13, 0
        mov r14, 0
        mov r15, 0
        mov r16, 0
        mov r21, 0
        mov r22, 0
        setp.eq.f32 p0, r1, r2
        @p0 mov r10, 1
        setp.ne.f32 p1, r1, r2
        @p1 mov r11, 1
        setp.lt.f32 p2, r1, r2
        @p2 mov r12, 1
        setp.le.f32 p3, r1, r2
        @p3 mov r13, 1
        setp.gt.f32 p4, r1, r2
        @p4 mov r14, 1
        setp.ge.f32 p5, r1, r2
        @p5 mov r15, 1
        setp.lt.f32 p6, r1, ur1
        @p6 mov r16, 1
        setp.gt.f32 p0, r2, r1
        sub r17, r1, r2
        sub r18, r1, ur2
        setp.lt.u32 p1, r0, ur2
        @p0 mov r21, 1
        @p1 mov r22, 1
        fmin r20, r1, r2
        mul24.u32 r19, r1, r2
        halt
    )";
    const auto kernel = compile(source, profile);
    require(nanosm::load_hex(nanosm::emit_hex(kernel)).words == kernel.words, "hex round trip");
    auto launch = launch_warps(1);
    auto& shmem = image_bytes(launch.test_shmem, 128);
    for (uint32_t lane = 0; lane < 16; ++lane) {
        write_u32(shmem, lane * 4, a[lane]);
        write_u32(shmem, 64 + lane * 4, b[lane]);
    }
    nanosm::Machine machine(profile, kernel, std::move(launch), nanosm::TraceOptions{.timeline = true});
    machine.run(20000);
    for (uint32_t lane = 0; lane < 16; ++lane) {
        const auto at = [&](uint32_t reg) { return machine.read_reg(0, lane, reg); };
        const std::string where = "lane " + std::to_string(lane) + ": ";
        for (uint32_t column = 0; column < 7; ++column) {
            require_eq(at(10 + column), static_cast<uint32_t>(compares[lane][column] - '0'),
                where + "setp." + std::array<const char*, 7>{"eq", "ne", "lt", "le", "gt", "ge", "lt (against U)"}[column] +
                    ".f32");
        }
        require_eq(at(21), static_cast<uint32_t>(compares[lane][2] - '0'), where + "setp.gt.f32 b, a");
        require_eq(at(22), lane < 5 ? 1u : 0u, where + "the ALU setp presented beside the FPU's");
        require_eq(at(20), minimum[lane], where + "fmin");
        require_eq(at(17), a[lane] - b[lane], where + "sub");
        require_eq(at(18), a[lane] - 5u, where + "sub x, U");
        require_eq(at(19), static_cast<uint32_t>(uint64_t{a[lane] & 0xffffffu} * (b[lane] & 0xffffffu)), where + "mul24");
    }
    require_eq(machine.read_reg(0, 0, 17), 0xff800000u, "sub 0x3f800000 - 0x40000000 wraps");
    require_eq(machine.read_reg(0, 7, 18), 0xfffffffcu, "sub 1 - 5 wraps");
    require_eq(machine.read_reg(0, 13, 19), 0xff800000u, "mul24 0x7fffff x 0x800000: the low word of 46 bits");
    require_eq(machine.read_reg(0, 15, 19), 0u, "mul24 0x800000 x 0x800000 = 2^46: its low word is 0");

    // The FPU's setp.gt.f32 p0 and the ALU's setp.lt.u32 p1 reached the
    // predicate file's input in one cycle.
    std::optional<uint64_t> fpu_port, alu_port;
    for (std::size_t index = 0; index < kernel.words.size(); ++index) {
        const auto inst = nanosm::decode(kernel.words[index]);
        const bool fpu = inst.opcode == nanosm::Opcode::setp_gt_f32 && inst.dst == 0;
        const bool alu = inst.opcode == nanosm::Opcode::setp_lt && inst.dst == 1;
        if (!fpu && !alu) continue;
        for (const auto& event : machine.trace().events()) {
            if (event.category != nanosm::TraceCategory::instruction || event.id != 8 * index) continue;
            (fpu ? fpu_port : alu_port) = event.start_cycle + nanosm::issue_to_port(inst, profile);
        }
    }
    require(fpu_port.has_value() && alu_port.has_value(), "the two setps' issues are not in the trace");
    require_eq(*fpu_port, *alu_port, "the FPU's and the ALU's setp presented in one cycle");

    // No immediate float compare; no uniform one.
    expect_error([&] { (void)compile("sread r0, LANE_ID\nsetp.lt.f32 p0, r0, 1\nhalt\n", profile); },
        {"compares two registers"}, "setp.f32 with an immediate");
    expect_error([&] { (void)compile("umov ur1, 1\nusetp.lt.f32 p0, ur1, ur1\nhalt\n", profile); },
        {"the uniform ALU has no float compare"}, "usetp.f32");
}

// The SFU: 4 lanes, so an ex2 or rcp of 16 lanes takes 4 passes and the unit
// takes a new op every 4 cycles; qtxc schedules its effective latency
// (occupancy 3 + latency 8 = 11). Two warps' four ex2 are admitted exactly 4
// cycles apart, while an independent ALU op issues beside the busy unit.
TEST(Alu, sfu_unit_timing) {
    const auto& profile = canonical_profile();
    require_eq(profile.timing.sfu, 11, "canonical effective SFU latency (3 + 8)");
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
        require_eq(sfu_issues[i] - sfu_issues[i - 1], 4, "cycles between SFU admissions");
    }
    const auto adds = issued_at(machine, 16);
    require(adds.size() == 2, "expected one add per warp");
    for (uint32_t warp = 0; warp < 2; ++warp) {
        require(adds[warp] > first[warp] && adds[warp] < first[warp] + 4, "an independent ALU op waited on the busy SFU");
    }
    const auto& summary = machine.trace().summary();
    require(summary.sfu_ops == 4 && summary.sfu_busy_cycles == 16, "SFU counters did not record four quarter-rate ops");
    require(summary.sfu_unit_stalls > 0, "SFU contention produced no counted stalls");
    for (uint32_t warp = 0; warp < 2; ++warp) {
        require_lanes(machine, 5, [](uint32_t lane) { return 2 * lane; }, "the interleaved ALU op", warp);
    }
}

TEST(Alu, sfu_profile_validation) {
    std::ifstream input("profiles/nanosm-v0.json");
    nlohmann::json document;
    input >> document;
    require_eq(nanosm::parse_device_profile(document.dump()).timing.sfu, 11, "canonical sfu occupancy + latency");
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

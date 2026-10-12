#pragma once

#include <stdint.h>
#include <string>
#include <vector>

namespace nanosm {

enum class Opcode : uint8_t {
    sread = 0x01,
    mov   = 0x02,
    add   = 0x03,
    addi  = 0x04,
    shli  = 0x05,
    fadd  = 0x06,
    xori  = 0x07,
    setp_eq = 0x08,
    setp_ne = 0x09,
    setp_lt = 0x0a,
    setp_le = 0x0b,
    setp_gt = 0x0c,
    setp_ge = 0x0d,
    fmul  = 0x0e,
    fsub  = 0x0f,
    lds   = 0x10,
    sts   = 0x11,
    ldg   = 0x12,
    stg   = 0x13,
    fmax  = 0x14,
    fma   = 0x15,
    cvt_pack = 0x16, // cvt.pack.f16x2: rd = half(ra) | half(rb) << 16
    ex2   = 0x17,    // SFU: 2^x
    rcp   = 0x18,    // SFU: 1/x
    and_  = 0x19,    // trailing underscores dodge the C++ operator tokens
    or_   = 0x1a,
    xor_  = 0x1b,
    shl   = 0x1c,    // register shifts use the low 5 bits: a 5-bit shifter
    shr   = 0x1d,    // logical
    andi  = 0x1e,
    ori   = 0x1f,
    bra   = 0x20,
    bra_uni = 0x21,
    shri  = 0x22,    // logical; 0x19..0x1f were full
    // Interleaved vector (rfc09): rA warp-uniform, 64 B aligned, 512 contiguous
    // bytes; register rD+j, lane l <-> word 16*j + l.
    ldg_v8 = 0x23,
    stg_v8 = 0x24,
    ldg_uni = 0x25,  // rA warp-uniform: one sector request, the word fans out to every lane
    lds_uni = 0x26,  // rA warp-uniform: one shmem bank access, the word fans out to every lane
    // No operands (trash/rfc12.md section 10): admitted only once every earlier
    // shmem access of the warp has reached the RAM (stores written, loads
    // sampled), so nothing after it issues before that. Never auto-inserted.
    fence_sh = 0x27,

    setpi_eq = 0x28,
    setpi_ne = 0x29,
    setpi_lt = 0x2a,
    setpi_le = 0x2b,
    setpi_gt = 0x2c,
    setpi_ge = 0x2d,
    mbar_init_i = 0x30,
    mbar_init_r = 0x31,
    mbar_arrive = 0x32,
    mbar_arrive_expect_tx_i = 0x33,
    mbar_arrive_expect_tx_r = 0x34,
    mbar_test_wait_i = 0x35,
    mbar_test_wait_r = 0x36,
    mbar_try_wait_i = 0x37,
    mbar_try_wait_r = 0x38,
    // Uniform shared-memory atomics on signed int32 (trash/rfc13.md):
    // atom.op.uni rD, [rA + imm], rB. ONE 32-bit word per warp instruction:
    // the address (rA + imm) and the operand (rB) come from lane 0, the update
    // is applied once, and every active lane's rD receives the word's old
    // value. add wraps (32-bit two's complement); exch writes the operand. An
    // mbar-engine operation: the global store pause covers its one read, the
    // ALU's C stages and its one write (D + Lb + C + 1 cycles canonical: 7).
    atom_add_uni = 0x39,
    atom_and_uni = 0x3a,
    atom_xor_uni = 0x3b,
    atom_exch_uni = 0x3c,
    tma_async = 0x40,
    acc_ld = 0x50,
    acc_ld_v8 = 0x51,
    acc_st = 0x52,
    acc_st_v8 = 0x53,
    mma_async = 0x60,
    mma_async_init = 0x61,
    mma_commit = 0x62,
    halt  = 0xff,
};

enum class SpecialReg : uint8_t {
    lane_id = 0,
    warp_id = 1,
    thread_id = 2,
    n_warps = 3,
};

struct Timing {
    uint8_t alu = 0;
    uint8_t fadd = 0;
    // Effective SFU result latency from execution start: the unit serves
    // sfu.lanes lanes per cycle, so this is (occupancy - 1) + sfu latency.
    // Derived from the profile's sfu section at parse time.
    uint8_t sfu = 0;
};

struct Control {
    // Predicate id 7 is PT. Bit 3 of the encoded predicate field is inversion.
    uint8_t predicate = 7;
    bool predicate_invert = false;
    uint8_t stall = 0;
    uint8_t set_depbar = 7; // 0..5 are real depbars, 7 means none
    uint8_t wait_mask = 0;   // bits 0..5
};

struct Instruction {
    Opcode opcode{};
    Control control{};
    uint8_t dst = 0;
    uint8_t src0 = 0;
    uint8_t src1 = 0;
    uint8_t src2 = 0; // third source; only fma uses it (RRR reserved bits)
    int32_t imm = 0;
    uint32_t line = 0;
    std::string text;
};

struct Kernel {
    struct Accumulator {
        std::string name;
        uint8_t bank = 0;
        uint32_t base_word = 0;
        uint16_t rows = 0;
        uint16_t cols = 0;
    };
    std::string profile_name;
    uint64_t profile_fingerprint = 0;
    uint16_t register_count = 0;
    std::vector<Accumulator> accumulators;
    std::vector<uint64_t> words;
};

namespace encoding {

// 63                            38 37                                      0
// +-------------------------------+-----------------------------------------+
// | opcode/predicate/stall/depbars | opcode-selected 38-bit payload          |
// +-------------------------------+-----------------------------------------+
// RRR payload: dst[8], src0[8], src1[8], reserved[14]
// RRI payload: dst[8], src0[8], simm[22]
// store payload: base[8], value[8], simm[22]
// RI payload: dst[8], reserved[8], simm[22]
// atomic payload: dst[8], base[8], operand[8], simm[14] (the offset is 14 bits:
// the third register takes the immediate's high bits)
// setp uses dst as p0..p6; immediate and register RHS forms have distinct opcodes.
// Branch immediates are signed byte displacements relative to the following PC.
// mbar addresses are held in a register; mbar immediates and register operands
// have distinct opcodes, as do setp immediates and register operands.
// tma.async uses src0 as a three-register coordinate-group base and its
// immediate as a static 64-byte-aligned shmem descriptor address.
// stall is 5 bits so a slow FPGA-priced SFU (20+ cycles) can still be a stall count.
inline constexpr unsigned opcode_shift = 56;
inline constexpr unsigned predicate_shift = 52;
inline constexpr unsigned stall_shift = 47;
inline constexpr unsigned set_depbar_shift = 44;
inline constexpr unsigned wait_mask_shift = 38;
inline constexpr unsigned immediate_shift = 0;

inline constexpr uint64_t opcode_mask = 0xff;
inline constexpr uint64_t predicate_mask = 0x0f;
inline constexpr uint64_t stall_mask = 0x1f;
inline constexpr uint64_t set_depbar_mask = 0x07;
inline constexpr uint64_t wait_mask_mask = 0x3f;
inline constexpr uint64_t register_mask = 0xff;
inline constexpr uint64_t immediate_mask = 0x3fffff;
inline constexpr int32_t immediate_min = -(1 << 21);
inline constexpr int32_t immediate_max = (1 << 21) - 1;
inline constexpr uint64_t atomic_immediate_mask = 0x3fff;
inline constexpr int32_t atomic_immediate_min = -(1 << 13);
inline constexpr int32_t atomic_immediate_max = (1 << 13) - 1;

inline constexpr unsigned reg0_shift = 30;
inline constexpr unsigned reg1_shift = 22;
inline constexpr unsigned reg2_shift = 14;
inline constexpr unsigned reg3_shift = 6; // fma's rc, in RRR reserved bits

} // namespace encoding

// The uniform shared atomics (trash/rfc13.md): atom.add/and/xor/exch.uni.
inline bool is_atomic(Opcode opcode) {
    return opcode == Opcode::atom_add_uni || opcode == Opcode::atom_and_uni ||
        opcode == Opcode::atom_xor_uni || opcode == Opcode::atom_exch_uni;
}

uint64_t encode(const Instruction& instruction);
Instruction decode(uint64_t word);
std::string format_instruction(const Instruction& instruction, bool include_control = true);
std::string format_control(const Control& control);
std::string opcode_name(Opcode opcode);
uint8_t fixed_latency(const Instruction& instruction, const Timing& timing);

} // namespace nanosm

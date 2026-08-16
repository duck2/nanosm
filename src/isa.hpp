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
    setp_eq = 0x08,
    setp_ne = 0x09,
    setp_lt = 0x0a,
    setp_le = 0x0b,
    setp_gt = 0x0c,
    setp_ge = 0x0d,
    lds   = 0x10,
    sts   = 0x11,
    ldg   = 0x12,
    stg   = 0x13,
    bra   = 0x20,
    bra_uni = 0x21,
    setpi_eq = 0x28,
    setpi_ne = 0x29,
    setpi_lt = 0x2a,
    setpi_le = 0x2b,
    setpi_gt = 0x2c,
    setpi_ge = 0x2d,
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
    int32_t imm = 0;
    uint32_t line = 0;
    std::string text;
};

struct Kernel {
    std::string profile_name;
    uint64_t profile_fingerprint = 0;
    uint16_t register_count = 0;
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
// setp uses dst as p0..p6; immediate and register RHS forms have distinct opcodes.
// Branch immediates are signed byte displacements relative to the following PC.
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

inline constexpr unsigned reg0_shift = 30;
inline constexpr unsigned reg1_shift = 22;
inline constexpr unsigned reg2_shift = 14;

} // namespace encoding

uint64_t encode(const Instruction& instruction);
Instruction decode(uint64_t word);
std::string format_instruction(const Instruction& instruction, bool include_control = true);
std::string format_control(const Control& control);
std::string opcode_name(Opcode opcode);
uint8_t fixed_latency(const Instruction& instruction, const Timing& timing);

} // namespace nanosm

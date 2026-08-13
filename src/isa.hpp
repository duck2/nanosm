#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace nanosm {

enum class Opcode : std::uint8_t {
    sread = 0x01,
    mov   = 0x02,
    add   = 0x03,
    addi  = 0x04,
    shli  = 0x05,
    fadd  = 0x06,
    lds   = 0x10,
    sts   = 0x11,
    halt  = 0xff,
};

enum class SpecialReg : std::uint8_t {
    lane_id = 0,
    warp_id = 1,
    thread_id = 2,
    n_warps = 3,
};

struct Timing {
    std::uint8_t alu{1};
    std::uint8_t fadd{4};
};

struct Control {
    // Predicate id 7 is PT. Bit 3 of the encoded predicate field is inversion.
    std::uint8_t predicate{7};
    bool predicate_invert{};
    std::uint8_t stall{};
    std::uint8_t set_depbar{7}; // 0..5 are real depbars, 7 means none
    std::uint8_t wait_mask{};   // bits 0..5
};

struct Instruction {
    Opcode opcode{};
    Control control{};
    std::uint8_t dst{};
    std::uint8_t src0{};
    std::uint8_t src1{};
    std::int32_t imm{};
    std::uint32_t line{};
    std::string text;
};

struct Kernel {
    std::vector<std::uint64_t> words;
};

namespace encoding {

//  63       56 55    52 51     47 46  44 43      38 37   32 31   26 25   20 19       0
// +-----------+--------+---------+------+----------+-------+-------+-------+----------+
// | opcode[8] | pred[4]|stall[5] |set[3]| wait[6]  |dst[6] |src0[6]|src1[6]| simm[20] |
// +-----------+--------+---------+------+----------+-------+-------+-------+----------+
// stall is 5 bits so a slow FPGA-priced SFU (20+ cycles) can still be a stall count.
inline constexpr unsigned opcode_shift = 56;
inline constexpr unsigned predicate_shift = 52;
inline constexpr unsigned stall_shift = 47;
inline constexpr unsigned set_depbar_shift = 44;
inline constexpr unsigned wait_mask_shift = 38;
inline constexpr unsigned dst_shift = 32;
inline constexpr unsigned src0_shift = 26;
inline constexpr unsigned src1_shift = 20;
inline constexpr unsigned immediate_shift = 0;

inline constexpr std::uint64_t opcode_mask = 0xff;
inline constexpr std::uint64_t predicate_mask = 0x0f;
inline constexpr std::uint64_t stall_mask = 0x1f;
inline constexpr std::uint64_t set_depbar_mask = 0x07;
inline constexpr std::uint64_t wait_mask_mask = 0x3f;
inline constexpr std::uint64_t register_mask = 0x3f;
inline constexpr std::uint64_t immediate_mask = 0xfffff;
inline constexpr std::int32_t immediate_min = -(1 << 19);
inline constexpr std::int32_t immediate_max = (1 << 19) - 1;

} // namespace encoding

std::uint64_t encode(const Instruction& instruction);
Instruction decode(std::uint64_t word);
std::string format_instruction(const Instruction& instruction, bool include_control = true);
std::string format_control(const Control& control);
std::string opcode_name(Opcode opcode);
std::uint8_t fixed_latency(const Instruction& instruction, const Timing& timing);

} // namespace nanosm

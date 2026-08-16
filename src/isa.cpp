#include "isa.hpp"

#include "compiler.hpp"

#include <iomanip>
#include <sstream>

namespace nanosm {
namespace {

std::string reg(uint8_t value) {
    return "r" + std::to_string(value);
}

std::string special_name(int32_t value) {
    switch (static_cast<SpecialReg>(value)) {
    case SpecialReg::lane_id: return "LANE_ID";
    case SpecialReg::warp_id: return "WARP_ID";
    case SpecialReg::thread_id: return "THREAD_ID";
    case SpecialReg::n_warps: return "N_WARPS";
    }
    return "SPECIAL(" + std::to_string(value) + ")";
}

std::string address(uint8_t base, int32_t offset) {
    if (offset == 0) {
        return "[" + reg(base) + "]";
    }
    return "[" + reg(base) + (offset > 0 ? " + " : " - ") +
        std::to_string(offset > 0 ? offset : -offset) + "]";
}

} // namespace

uint64_t encode(const Instruction& inst) {
    if (inst.control.predicate > 7 || inst.control.stall > encoding::stall_mask ||
        inst.control.set_depbar > 7 || inst.control.wait_mask > encoding::wait_mask_mask ||
        inst.imm < encoding::immediate_min || inst.imm > encoding::immediate_max) {
        throw SimError("instruction field does not fit 64-bit encoding: " + format_instruction(inst));
    }
    const auto predicate = static_cast<uint64_t>(
        inst.control.predicate | (inst.control.predicate_invert ? 8 : 0));
    uint64_t word =
        (static_cast<uint64_t>(inst.opcode) << encoding::opcode_shift) |
        (predicate << encoding::predicate_shift) |
        (static_cast<uint64_t>(inst.control.stall) << encoding::stall_shift) |
        (static_cast<uint64_t>(inst.control.set_depbar) << encoding::set_depbar_shift) |
        (static_cast<uint64_t>(inst.control.wait_mask) << encoding::wait_mask_shift);
    const auto immediate = static_cast<uint32_t>(inst.imm) & encoding::immediate_mask;
    switch (inst.opcode) {
    case Opcode::sread:
    case Opcode::mov:
        word |= static_cast<uint64_t>(inst.dst) << encoding::reg0_shift;
        word |= immediate;
        break;
    case Opcode::add:
    case Opcode::fadd:
    case Opcode::setp_eq:
    case Opcode::setp_ne:
    case Opcode::setp_lt:
    case Opcode::setp_le:
    case Opcode::setp_gt:
    case Opcode::setp_ge:
        word |= static_cast<uint64_t>(inst.dst) << encoding::reg0_shift;
        word |= static_cast<uint64_t>(inst.src0) << encoding::reg1_shift;
        word |= static_cast<uint64_t>(inst.src1) << encoding::reg2_shift;
        break;
    case Opcode::addi:
    case Opcode::shli:
    case Opcode::lds:
    case Opcode::ldg:
    case Opcode::setpi_eq:
    case Opcode::setpi_ne:
    case Opcode::setpi_lt:
    case Opcode::setpi_le:
    case Opcode::setpi_gt:
    case Opcode::setpi_ge:
        word |= static_cast<uint64_t>(inst.dst) << encoding::reg0_shift;
        word |= static_cast<uint64_t>(inst.src0) << encoding::reg1_shift;
        word |= immediate;
        break;
    case Opcode::sts:
    case Opcode::stg:
        word |= static_cast<uint64_t>(inst.src0) << encoding::reg0_shift;
        word |= static_cast<uint64_t>(inst.src1) << encoding::reg1_shift;
        word |= immediate;
        break;
    case Opcode::bra:
    case Opcode::bra_uni:
        word |= immediate;
        break;
    case Opcode::halt:
        break;
    }
    return word;
}

Instruction decode(uint64_t word) {
    Instruction inst;
    inst.opcode = static_cast<Opcode>((word >> encoding::opcode_shift) & encoding::opcode_mask);
    const auto predicate = static_cast<uint8_t>(
        (word >> encoding::predicate_shift) & encoding::predicate_mask);
    inst.control.predicate = predicate & 7;
    inst.control.predicate_invert = (predicate & 8) != 0;
    inst.control.stall = static_cast<uint8_t>(
        (word >> encoding::stall_shift) & encoding::stall_mask);
    inst.control.set_depbar = static_cast<uint8_t>(
        (word >> encoding::set_depbar_shift) & encoding::set_depbar_mask);
    inst.control.wait_mask = static_cast<uint8_t>(
        (word >> encoding::wait_mask_shift) & encoding::wait_mask_mask);
    const auto reg0 = static_cast<uint8_t>((word >> encoding::reg0_shift) & encoding::register_mask);
    const auto reg1 = static_cast<uint8_t>((word >> encoding::reg1_shift) & encoding::register_mask);
    const auto reg2 = static_cast<uint8_t>((word >> encoding::reg2_shift) & encoding::register_mask);
    auto immediate = static_cast<uint32_t>(word & encoding::immediate_mask);
    if ((immediate & (1u << 21)) != 0) {
        immediate |= ~static_cast<uint32_t>(encoding::immediate_mask);
    }
    switch (inst.opcode) {
    case Opcode::sread:
    case Opcode::mov:
        inst.dst = reg0;
        inst.imm = static_cast<int32_t>(immediate);
        break;
    case Opcode::add:
    case Opcode::fadd:
    case Opcode::setp_eq:
    case Opcode::setp_ne:
    case Opcode::setp_lt:
    case Opcode::setp_le:
    case Opcode::setp_gt:
    case Opcode::setp_ge:
        inst.dst = reg0;
        inst.src0 = reg1;
        inst.src1 = reg2;
        break;
    case Opcode::addi:
    case Opcode::shli:
    case Opcode::lds:
    case Opcode::ldg:
    case Opcode::setpi_eq:
    case Opcode::setpi_ne:
    case Opcode::setpi_lt:
    case Opcode::setpi_le:
    case Opcode::setpi_gt:
    case Opcode::setpi_ge:
        inst.dst = reg0;
        inst.src0 = reg1;
        inst.imm = static_cast<int32_t>(immediate);
        break;
    case Opcode::sts:
    case Opcode::stg:
        inst.src0 = reg0;
        inst.src1 = reg1;
        inst.imm = static_cast<int32_t>(immediate);
        break;
    case Opcode::bra:
    case Opcode::bra_uni:
        inst.imm = static_cast<int32_t>(immediate);
        break;
    case Opcode::halt:
        break;
    }
    return inst;
}

std::string opcode_name(Opcode opcode) {
    switch (opcode) {
    case Opcode::sread: return "sread";
    case Opcode::mov: return "mov";
    case Opcode::add: return "add";
    case Opcode::addi: return "addi";
    case Opcode::shli: return "shli";
    case Opcode::fadd: return "fadd";
    case Opcode::setp_eq: return "setp.eq";
    case Opcode::setp_ne: return "setp.ne";
    case Opcode::setp_lt: return "setp.lt";
    case Opcode::setp_le: return "setp.le";
    case Opcode::setp_gt: return "setp.gt";
    case Opcode::setp_ge: return "setp.ge";
    case Opcode::setpi_eq: return "setp.eq";
    case Opcode::setpi_ne: return "setp.ne";
    case Opcode::setpi_lt: return "setp.lt";
    case Opcode::setpi_le: return "setp.le";
    case Opcode::setpi_gt: return "setp.gt";
    case Opcode::setpi_ge: return "setp.ge";
    case Opcode::lds: return "lds";
    case Opcode::sts: return "sts";
    case Opcode::ldg: return "ldg";
    case Opcode::stg: return "stg";
    case Opcode::bra: return "bra";
    case Opcode::bra_uni: return "bra.uni";
    case Opcode::halt: return "halt";
    }
    std::ostringstream out;
    out << "opcode_0x" << std::hex << std::setw(2) << std::setfill('0')
        << static_cast<unsigned>(opcode);
    return out.str();
}

std::string format_control(const Control& control) {
    std::ostringstream out;
    out << "{pred=";
    if (control.predicate_invert) out << '!';
    if (control.predicate == 7) out << "PT";
    else out << 'p' << static_cast<unsigned>(control.predicate);
    out << ", stall=" << static_cast<unsigned>(control.stall) << ", set=";
    if (control.set_depbar == 7) out << '-';
    else out << "sb" << static_cast<unsigned>(control.set_depbar);
    out << ", wait=";
    if (control.wait_mask == 0) {
        out << '-';
    } else {
        bool first = true;
        for (unsigned bar = 0; bar < 6; ++bar) {
            if ((control.wait_mask & (1u << bar)) == 0) continue;
            if (!first) out << '|';
            out << "sb" << bar;
            first = false;
        }
    }
    out << '}';
    return out.str();
}

std::string format_instruction(const Instruction& inst, bool include_control) {
    std::ostringstream out;
    out << opcode_name(inst.opcode);
    switch (inst.opcode) {
    case Opcode::sread:
        out << ' ' << reg(inst.dst) << ", " << special_name(inst.imm);
        break;
    case Opcode::mov:
        out << ' ' << reg(inst.dst) << ", " << inst.imm;
        break;
    case Opcode::add:
    case Opcode::fadd:
        out << ' ' << reg(inst.dst) << ", " << reg(inst.src0) << ", " << reg(inst.src1);
        break;
    case Opcode::setp_eq:
    case Opcode::setp_ne:
    case Opcode::setp_lt:
    case Opcode::setp_le:
    case Opcode::setp_gt:
    case Opcode::setp_ge:
        out << " p" << static_cast<unsigned>(inst.dst) << ", " << reg(inst.src0)
            << ", " << reg(inst.src1);
        break;
    case Opcode::setpi_eq:
    case Opcode::setpi_ne:
    case Opcode::setpi_lt:
    case Opcode::setpi_le:
    case Opcode::setpi_gt:
    case Opcode::setpi_ge:
        out << " p" << static_cast<unsigned>(inst.dst) << ", " << reg(inst.src0)
            << ", " << inst.imm;
        break;
    case Opcode::addi:
    case Opcode::shli:
        out << ' ' << reg(inst.dst) << ", " << reg(inst.src0) << ", " << inst.imm;
        break;
    case Opcode::lds:
    case Opcode::ldg:
        out << ' ' << reg(inst.dst) << ", " << address(inst.src0, inst.imm);
        break;
    case Opcode::sts:
    case Opcode::stg:
        out << ' ' << address(inst.src0, inst.imm) << ", " << reg(inst.src1);
        break;
    case Opcode::bra:
    case Opcode::bra_uni:
        out << ' ' << (inst.imm >= 0 ? "+" : "") << inst.imm;
        break;
    case Opcode::halt:
        break;
    }
    if (include_control) {
        out << ' ' << format_control(inst.control);
    }
    return out.str();
}

uint8_t fixed_latency(const Instruction& inst, const Timing& timing) {
    switch (inst.opcode) {
    case Opcode::sread:
    case Opcode::mov:
    case Opcode::add:
    case Opcode::addi:
    case Opcode::shli:
        return timing.alu;
    case Opcode::fadd:
        return timing.fadd;
    case Opcode::setp_eq:
    case Opcode::setp_ne:
    case Opcode::setp_lt:
    case Opcode::setp_le:
    case Opcode::setp_gt:
    case Opcode::setp_ge:
    case Opcode::setpi_eq:
    case Opcode::setpi_ne:
    case Opcode::setpi_lt:
    case Opcode::setpi_le:
    case Opcode::setpi_gt:
    case Opcode::setpi_ge:
        return timing.alu;
    case Opcode::lds:
    case Opcode::sts:
    case Opcode::ldg:
    case Opcode::stg:
    case Opcode::bra:
    case Opcode::bra_uni:
    case Opcode::halt:
        return 0;
    }
    throw SimError("latency requested for unknown opcode");
}

} // namespace nanosm

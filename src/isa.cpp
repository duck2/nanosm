// The instruction set's functions, each a loop over the form and opcode
// tables (src/isa.hpp): encode, decode, an instruction's registers, its text.

#include "isa.hpp"

#include "qtxc/qtxc.hpp"

#include <bit>
#include <iomanip>
#include <sstream>

namespace nanosm {
namespace {

std::string special_name(int32_t value) {
    switch (static_cast<SpecialReg>(value)) {
    case SpecialReg::lane_id: return "LANE_ID";
    case SpecialReg::warp_id: return "WARP_ID";
    case SpecialReg::thread_id: return "THREAD_ID";
    case SpecialReg::n_warps: return "N_WARPS";
    }
    return "SPECIAL(" + std::to_string(value) + ")";
}

std::string address(const std::string& base, int32_t offset) {
    if (offset == 0) {
        return "[" + base + "]";
    }
    return "[" + base + (offset > 0 ? " + " : " - ") +
        std::to_string(offset > 0 ? offset : -offset) + "]";
}

// An Instruction's register fields in the forms' order: dst, src0, src1, src2.
std::array<const uint8_t*, 4> registers_of(const Instruction& inst) {
    return {&inst.dst, &inst.src0, &inst.src1, &inst.src2};
}

std::array<uint8_t*, 4> registers_of(Instruction& inst) {
    return {&inst.dst, &inst.src0, &inst.src1, &inst.src2};
}

// Each operand field's place in the word: register fields 0..3, then umov's.
constexpr std::array<unsigned, 5> field_shift = {encoding::reg0_shift, encoding::reg1_shift, encoding::reg2_shift,
    encoding::reg3_shift, encoding::umovi_dst_shift};
constexpr std::array<uint64_t, 5> field_mask = {encoding::register_mask, encoding::register_mask,
    encoding::register_mask, encoding::register_mask, encoding::umovi_dst_mask};

// The operand's file as this instruction uses it (an X2 operand by its U bit).
RegisterFile file_of(const Operand& operand, const Instruction& inst) {
    if (operand.file != RegisterFile::either) return operand.file;
    return inst.src1_uniform ? RegisterFile::uniform : RegisterFile::vector;
}

uint32_t width_of(const Operand& operand, Opcode opcode) {
    switch (operand.width) {
    case OperandWidth::one: return 1;
    case OperandWidth::span: return register_span(opcode);
    case OperandWidth::coordinates: return tma_coordinate_count(opcode);
    }
    return 1;
}

std::string register_text(RegisterFile file, uint8_t value) {
    const char* prefix = file == RegisterFile::uniform ? "ur" : file == RegisterFile::predicate ? "p" : "r";
    return prefix + std::to_string(value);
}

int32_t sign_extended(uint64_t word, unsigned bits) {
    const auto field = static_cast<uint32_t>(word & ((uint64_t{1} << bits) - 1));
    const auto sign = uint32_t{1} << (bits - 1);
    return static_cast<int32_t>((field ^ sign) - sign);
}

// The registers of the operands in `file` the instruction reads (src0, src1,
// src2), each register of a wide operand; a tma's coordinates after the rest.
std::vector<uint8_t> sources_in(const Instruction& inst, RegisterFile file) {
    const auto& form = form_info(inst.opcode);
    const auto registers = registers_of(inst);
    std::vector<uint8_t> result;
    for (const bool coordinates : {false, true}) {
        for (std::size_t slot = 1; slot < 4; ++slot) {
            const auto& operand = form.operands[slot];
            if (file_of(operand, inst) != file || (operand.width == OperandWidth::coordinates) != coordinates) continue;
            for (uint32_t j = 0; j < width_of(operand, inst.opcode); ++j) {
                result.push_back(static_cast<uint8_t>(*registers[slot] + j));
            }
        }
    }
    return result;
}

std::optional<uint8_t> destination_in(const Instruction& inst, RegisterFile file) {
    if (form_info(inst.opcode).operands[0].file == file) return inst.dst;
    return std::nullopt;
}

} // namespace

uint64_t encode(const Instruction& inst) {
    const auto& form = form_info(inst.opcode);
    const auto kind = form.immediate;
    if (inst.control.predicate > 7 || inst.control.stall > encoding::stall_mask ||
        inst.control.set_depbar > 7 || inst.control.wait_mask > encoding::wait_mask_mask ||
        (kind != ImmediateKind::word && (inst.imm < encoding::immediate_min || inst.imm > encoding::immediate_max))) {
        throw SimError("instruction field does not fit 64-bit encoding: " + format_instruction(inst));
    }
    if (kind == ImmediateKind::s14 &&
        (inst.imm < encoding::atomic_immediate_min || inst.imm > encoding::atomic_immediate_max)) {
        throw SimError("atomic offset does not fit its signed 14-bit immediate: " + format_instruction(inst));
    }
    if (kind == ImmediateKind::descriptor &&
        (inst.imm < 0 || inst.imm % encoding::tma_descriptor_field_unit != 0 ||
         static_cast<uint64_t>(inst.imm) / encoding::tma_descriptor_field_unit > encoding::tma_descriptor_field_mask)) {
        throw SimError("tma descriptor address must be 64-byte aligned and below 1 MiB: " + format_instruction(inst));
    }
    if (inst.src1_uniform && !takes_uniform_source(inst.opcode)) {
        throw SimError(opcode_name(inst.opcode) + " has no uniform source: " + format_instruction(inst));
    }
    const bool vector_memory = kind == ImmediateKind::memory;
    if (inst.control.read_depbar > 7 || (!vector_memory && inst.control.read_depbar != 7)) {
        throw SimError(opcode_name(inst.opcode) + " has no read depbar field: " + format_instruction(inst));
    }
    if (vector_memory &&
        (inst.imm < encoding::vector_memory_immediate_min || inst.imm > encoding::vector_memory_immediate_max)) {
        throw SimError("offset does not fit the signed 19-bit immediate: " + format_instruction(inst));
    }
    if (kind == ImmediateKind::spacings && (inst.imm < 0 || inst.imm >= (1 << (2 * mma_spacing_field_bits)))) {
        throw SimError("mma.async row spacings do not fit their two 7-bit fields: " + format_instruction(inst));
    }
    if (form.operands[0].field == 4 && inst.dst > encoding::umovi_dst_mask) {
        throw SimError("umov's uniform register does not fit its 6-bit field: " + format_instruction(inst));
    }
    const auto predicate = static_cast<uint64_t>(
        inst.control.predicate | (inst.control.predicate_invert ? 8 : 0));
    uint64_t word =
        (static_cast<uint64_t>(inst.opcode) << encoding::opcode_shift) |
        (predicate << encoding::predicate_shift) |
        (static_cast<uint64_t>(inst.control.stall) << encoding::stall_shift) |
        (static_cast<uint64_t>(inst.control.set_depbar) << encoding::set_depbar_shift) |
        (static_cast<uint64_t>(inst.control.wait_mask) << encoding::wait_mask_shift);
    const auto registers = registers_of(inst);
    for (std::size_t slot = 0; slot < 4; ++slot) {
        const auto& operand = form.operands[slot];
        if (operand.file != RegisterFile::none) word |= uint64_t{*registers[slot]} << field_shift[operand.field];
    }
    if (takes_uniform_source(inst.opcode) && inst.src1_uniform) word |= uint64_t{1} << encoding::uniform_source_shift;
    const auto imm = static_cast<uint32_t>(inst.imm);
    switch (kind) {
    case ImmediateKind::none: break;
    case ImmediateKind::s22: word |= imm & encoding::immediate_mask; break;
    case ImmediateKind::memory:
        word |= uint64_t{inst.control.read_depbar} << encoding::read_depbar_shift |
            (imm & encoding::vector_memory_immediate_mask);
        break;
    case ImmediateKind::s14: word |= imm & encoding::atomic_immediate_mask; break;
    case ImmediateKind::descriptor: word |= imm / encoding::tma_descriptor_field_unit; break;
    case ImmediateKind::spacings:
    case ImmediateKind::word: word |= imm; break;
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
    const auto& form = form_info(inst.opcode);
    const auto registers = registers_of(inst);
    for (std::size_t slot = 0; slot < 4; ++slot) {
        const auto& operand = form.operands[slot];
        if (operand.file == RegisterFile::none) continue;
        *registers[slot] = static_cast<uint8_t>((word >> field_shift[operand.field]) & field_mask[operand.field]);
    }
    if (takes_uniform_source(inst.opcode)) inst.src1_uniform = ((word >> encoding::uniform_source_shift) & 1) != 0;
    switch (form.immediate) {
    case ImmediateKind::none: break;
    case ImmediateKind::s22: inst.imm = sign_extended(word, 22); break;
    case ImmediateKind::memory:
        inst.control.read_depbar = static_cast<uint8_t>((word >> encoding::read_depbar_shift) & encoding::read_depbar_mask);
        inst.imm = sign_extended(word, 19);
        break;
    case ImmediateKind::s14: inst.imm = sign_extended(word, 14); break;
    case ImmediateKind::descriptor:
        inst.imm = static_cast<int32_t>((word & encoding::tma_descriptor_field_mask) * encoding::tma_descriptor_field_unit);
        break;
    case ImmediateKind::spacings: inst.imm = static_cast<int32_t>(word & ((1u << (2 * mma_spacing_field_bits)) - 1)); break;
    case ImmediateKind::word: inst.imm = static_cast<int32_t>(static_cast<uint32_t>(word)); break;
    }
    return inst;
}

std::optional<unsigned> reserved_bit(uint64_t word) {
    const auto stray = word ^ encode(decode(word));
    if (stray == 0) return std::nullopt;
    return static_cast<unsigned>(std::countr_zero(stray));
}

std::vector<uint8_t> uniform_sources(const Instruction& inst) {
    return sources_in(inst, RegisterFile::uniform);
}

std::optional<uint8_t> uniform_destination(const Instruction& inst) {
    return destination_in(inst, RegisterFile::uniform);
}

std::vector<uint8_t> vector_sources(const Instruction& inst) {
    return sources_in(inst, RegisterFile::vector);
}

std::vector<uint8_t> issue_sources(const Instruction& inst) {
    if (form_info(inst.opcode).reads == VectorReads::after_issue) return {};
    return vector_sources(inst);
}

std::optional<uint8_t> vector_destination(const Instruction& inst) {
    return destination_in(inst, RegisterFile::vector);
}

std::optional<uint8_t> predicate_destination(const Instruction& inst) {
    return destination_in(inst, RegisterFile::predicate);
}

std::string mma_shape_name(const MmaShape& shape) {
    return "m" + std::to_string(shape.m) + "n" + std::to_string(shape.n) + "k" + std::to_string(shape.k);
}

std::optional<MmaShape> parse_mma_shape_name(const std::string& name) {
    MmaShape shape;
    std::size_t at = 0;
    for (const auto& [letter, field] : {std::pair{'m', &shape.m}, std::pair{'n', &shape.n}, std::pair{'k', &shape.k}}) {
        if (at >= name.size() || name[at] != letter) return std::nullopt;
        ++at;
        const auto digits = at;
        uint64_t value = 0;
        while (at < name.size() && name[at] >= '0' && name[at] <= '9' && at - digits < 6) {
            value = value * 10 + static_cast<uint64_t>(name[at] - '0');
            ++at;
        }
        if (at == digits || value == 0 || (at < name.size() && name[at] >= '0' && name[at] <= '9')) {
            return std::nullopt;
        }
        *field = static_cast<uint32_t>(value);
    }
    if (at != name.size()) return std::nullopt;
    return shape;
}

std::string opcode_name(Opcode opcode, const std::vector<MmaShape>* mma_shapes) {
    if (is_mma_async(opcode)) {
        const auto index = mma_shape_index(opcode);
        const std::string family = is_mma_init(opcode) ? "mma.async.init." : "mma.async.";
        if (mma_shapes != nullptr && index < mma_shapes->size()) {
            return family + mma_shape_name((*mma_shapes)[index]);
        }
        return family + "slot" + std::to_string(index);
    }
    if (const auto* mnemonic = opcode_info(opcode).mnemonic; mnemonic != nullptr) return mnemonic;
    std::ostringstream out;
    out << "opcode_0x" << std::hex << std::setw(2) << std::setfill('0')
        << static_cast<unsigned>(opcode);
    return out.str();
}

std::optional<Opcode> opcode_for_mnemonic(std::string_view mnemonic) {
    for (uint32_t value = 0; value < isa_tables::opcodes.size(); ++value) {
        const auto* name = isa_tables::opcodes[value].mnemonic;
        if (name != nullptr && mnemonic == name) return static_cast<Opcode>(value);
    }
    return std::nullopt;
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
    // The read depbar, where one is named (none is not printed: only the six
    // vector memory forms have the field).
    if (control.read_depbar != 7) out << ", read=sb" << static_cast<unsigned>(control.read_depbar);
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

// The mnemonic, then the form's operand text with each letter replaced
// (src/isa.hpp, the forms).
std::string format_instruction(
    const Instruction& inst, bool include_control, const std::vector<MmaShape>* mma_shapes) {
    std::ostringstream out;
    out << opcode_name(inst.opcode, mma_shapes);
    const auto& form = form_info(inst.opcode);
    const auto registers = registers_of(inst);
    const auto operand_text = [&](std::size_t slot) {
        return register_text(file_of(form.operands[slot], inst), *registers[slot]);
    };
    // An mma.async operand: its spacing is printed when it is not the default.
    const auto mma_operand = [&](uint8_t base, bool a_operand) {
        const auto spacing = mma_row_spacing(inst, a_operand);
        return "[" + register_text(RegisterFile::uniform, base) +
            (spacing == mma_default_spacing ? std::string() : ", " + std::to_string(spacing)) + "]";
    };
    if (*form.text != '\0') out << ' ';
    for (const char* letter = form.text; *letter != '\0'; ++letter) {
        switch (*letter) {
        case 'd': out << operand_text(0); break;
        case 'a': out << operand_text(1); break;
        case 'b': out << operand_text(2); break;
        case 'c': out << operand_text(3); break;
        case 'i': out << inst.imm; break;
        case 'm': out << address(operand_text(1), inst.imm); break;
        case 'j': out << (inst.imm >= 0 ? "+" : "") << inst.imm; break;
        case 's': out << special_name(inst.imm); break;
        case 'A': out << mma_operand(inst.src0, true); break;
        case 'B': out << mma_operand(inst.src1, false); break;
        case 'x':
            // A small value as a number (addresses, counts); any other word as
            // its 32 bits in hex (a float constant, a mask).
            if (inst.imm >= encoding::immediate_min && inst.imm <= encoding::immediate_max) {
                out << inst.imm;
            } else {
                out << "0x" << std::hex << std::setw(8) << std::setfill('0') << static_cast<uint32_t>(inst.imm)
                    << std::dec << std::setfill(' ');
            }
            break;
        default: out << *letter; break;
        }
    }
    if (include_control) {
        out << ' ' << format_control(inst.control);
    }
    return out.str();
}

uint8_t fixed_latency(const Instruction& inst, const Timing& timing) {
    switch (opcode_info(inst.opcode).unit) {
    case IssueUnit::alu: return timing.alu;
    case IssueUnit::fpu: return timing.fadd;
    case IssueUnit::sfu: return timing.sfu;
    case IssueUnit::none: throw SimError("latency requested for unknown opcode");
    default: return is_mbar_test_wait(inst.opcode) ? 1 : 0;
    }
}

} // namespace nanosm

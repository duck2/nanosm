#include "isa.hpp"

#include "compiler.hpp"

#include <iomanip>
#include <sstream>

namespace nanosm {
namespace {

std::string reg(uint8_t value) {
    return "r" + std::to_string(value);
}

std::string ureg(uint8_t value) {
    return "ur" + std::to_string(value);
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

std::string address(const std::string& base, int32_t offset) {
    if (offset == 0) {
        return "[" + base + "]";
    }
    return "[" + base + (offset > 0 ? " + " : " - ") +
        std::to_string(offset > 0 ? offset : -offset) + "]";
}

std::string address(uint8_t base, int32_t offset) {
    return address(reg(base), offset);
}

} // namespace

uint64_t encode(const Instruction& inst) {
    if (inst.control.predicate > 7 || inst.control.stall > encoding::stall_mask ||
        inst.control.set_depbar > 7 || inst.control.wait_mask > encoding::wait_mask_mask ||
        inst.imm < encoding::immediate_min || inst.imm > encoding::immediate_max) {
        throw SimError("instruction field does not fit 64-bit encoding: " + format_instruction(inst));
    }
    if (is_atomic(inst.opcode) &&
        (inst.imm < encoding::atomic_immediate_min || inst.imm > encoding::atomic_immediate_max)) {
        throw SimError("atomic offset does not fit its signed 14-bit immediate: " + format_instruction(inst));
    }
    if (is_tma_opcode(inst.opcode) &&
        (inst.imm < 0 || inst.imm % encoding::tma_descriptor_field_unit != 0 ||
         static_cast<uint64_t>(inst.imm) / encoding::tma_descriptor_field_unit > encoding::tma_descriptor_field_mask)) {
        throw SimError("tma descriptor address must be 64-byte aligned and below 1 MiB: " + format_instruction(inst));
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
    if (is_mma_async(inst.opcode)) {
        // urC (the result's tmem pointer), A and B base registers, A's and
        // B's row spacings (the block's opcodes share one form).
        if (inst.imm < 0 || inst.imm >= (1 << (2 * mma_spacing_field_bits))) {
            throw SimError("mma.async row spacings do not fit their two 7-bit fields: " + format_instruction(inst));
        }
        word |= static_cast<uint64_t>(inst.src2) << encoding::reg0_shift;
        word |= static_cast<uint64_t>(inst.src0) << encoding::reg1_shift;
        word |= static_cast<uint64_t>(inst.src1) << encoding::reg2_shift;
        word |= static_cast<uint64_t>(inst.imm);
        return word;
    }
    switch (inst.opcode) {
    case Opcode::sread:
    case Opcode::mov:
        word |= static_cast<uint64_t>(inst.dst) << encoding::reg0_shift;
        word |= immediate;
        break;
    case Opcode::add:
    case Opcode::and_:
    case Opcode::or_:
    case Opcode::xor_:
    case Opcode::shl:
    case Opcode::shr:
    case Opcode::fadd:
    case Opcode::fmul:
    case Opcode::fsub:
    case Opcode::fmax:
    case Opcode::cvt_pack:
    case Opcode::setp_eq:
    case Opcode::setp_ne:
    case Opcode::setp_lt:
    case Opcode::setp_le:
    case Opcode::setp_gt:
    case Opcode::setp_ge:
    case Opcode::setp_lt_s32:
    case Opcode::setp_le_s32:
    case Opcode::setp_gt_s32:
    case Opcode::setp_ge_s32:
        word |= static_cast<uint64_t>(inst.dst) << encoding::reg0_shift;
        word |= static_cast<uint64_t>(inst.src0) << encoding::reg1_shift;
        word |= static_cast<uint64_t>(inst.src1) << encoding::reg2_shift;
        break;
    case Opcode::fma:
        word |= static_cast<uint64_t>(inst.dst) << encoding::reg0_shift;
        word |= static_cast<uint64_t>(inst.src0) << encoding::reg1_shift;
        word |= static_cast<uint64_t>(inst.src1) << encoding::reg2_shift;
        word |= static_cast<uint64_t>(inst.src2) << encoding::reg3_shift;
        break;
    case Opcode::ex2:
    case Opcode::rcp:
        word |= static_cast<uint64_t>(inst.dst) << encoding::reg0_shift;
        word |= static_cast<uint64_t>(inst.src0) << encoding::reg1_shift;
        break;
    case Opcode::addi:
    case Opcode::shli:
    case Opcode::shri:
    case Opcode::xori:
    case Opcode::andi:
    case Opcode::ori:
    case Opcode::lds:
    case Opcode::ldg:
    case Opcode::ldg_v8:
    case Opcode::ldg_uni:
    case Opcode::lds_uni:
    case Opcode::acc_ld:
    case Opcode::acc_ld_v8:
    case Opcode::setpi_eq:
    case Opcode::setpi_ne:
    case Opcode::setpi_lt:
    case Opcode::setpi_le:
    case Opcode::setpi_gt:
    case Opcode::setpi_ge:
    case Opcode::setpi_lt_s32:
    case Opcode::setpi_le_s32:
    case Opcode::setpi_gt_s32:
    case Opcode::setpi_ge_s32:
        word |= static_cast<uint64_t>(inst.dst) << encoding::reg0_shift;
        word |= static_cast<uint64_t>(inst.src0) << encoding::reg1_shift;
        word |= immediate;
        break;
    case Opcode::mbar_init_i:
    case Opcode::mbar_arrive_expect_tx_i:
        word |= static_cast<uint64_t>(inst.src0) << encoding::reg0_shift;
        word |= immediate;
        break;
    case Opcode::mbar_init_r:
    case Opcode::mbar_arrive:
    case Opcode::mbar_arrive_expect_tx_r:
        word |= static_cast<uint64_t>(inst.src0) << encoding::reg0_shift;
        word |= static_cast<uint64_t>(inst.src1) << encoding::reg1_shift;
        break;
    case Opcode::mbar_test_wait_i:
    case Opcode::mbar_try_wait_i:
        word |= static_cast<uint64_t>(inst.dst) << encoding::reg0_shift;
        word |= static_cast<uint64_t>(inst.src0) << encoding::reg1_shift;
        word |= immediate;
        break;
    case Opcode::mbar_test_wait_r:
    case Opcode::mbar_try_wait_r:
        word |= static_cast<uint64_t>(inst.dst) << encoding::reg0_shift;
        word |= static_cast<uint64_t>(inst.src0) << encoding::reg1_shift;
        word |= static_cast<uint64_t>(inst.src1) << encoding::reg2_shift;
        break;
    case Opcode::tma_load_1d:
    case Opcode::tma_load_2d:
    case Opcode::tma_load_3d:
    case Opcode::tma_load_4d:
    case Opcode::tma_store_1d:
    case Opcode::tma_store_2d:
    case Opcode::tma_store_3d:
    case Opcode::tma_store_4d:
        word |= static_cast<uint64_t>(inst.src0) << encoding::reg0_shift;
        word |= static_cast<uint64_t>(inst.src1) << encoding::reg1_shift;
        word |= static_cast<uint64_t>(inst.src2) << encoding::reg2_shift;
        word |= static_cast<uint64_t>(inst.imm) / encoding::tma_descriptor_field_unit;
        break;
    case Opcode::mma_commit:
        word |= static_cast<uint64_t>(inst.src0) << encoding::reg0_shift;
        break;
    case Opcode::sts:
    case Opcode::stg:
    case Opcode::stg_v8:
    case Opcode::acc_st:
    case Opcode::acc_st_v8:
        word |= static_cast<uint64_t>(inst.src0) << encoding::reg0_shift;
        word |= static_cast<uint64_t>(inst.src1) << encoding::reg1_shift;
        word |= immediate;
        break;
    case Opcode::atom_add_uni:
    case Opcode::atom_and_uni:
    case Opcode::atom_xor_uni:
    case Opcode::atom_exch_uni:
        word |= static_cast<uint64_t>(inst.dst) << encoding::reg0_shift;
        word |= static_cast<uint64_t>(inst.src0) << encoding::reg1_shift;
        word |= static_cast<uint64_t>(inst.src1) << encoding::reg2_shift;
        word |= static_cast<uint32_t>(inst.imm) & encoding::atomic_immediate_mask;
        break;
    case Opcode::bra:
    case Opcode::bra_uni:
    case Opcode::ssy:
        word |= immediate;
        break;
    case Opcode::umov:
    case Opcode::r2ur:
    case Opcode::ur2r:
        word |= static_cast<uint64_t>(inst.dst) << encoding::reg0_shift;
        word |= static_cast<uint64_t>(inst.src0) << encoding::reg1_shift;
        break;
    case Opcode::umovi:
    case Opcode::usread:
        word |= static_cast<uint64_t>(inst.dst) << encoding::reg0_shift;
        word |= immediate;
        break;
    case Opcode::uadd:
    case Opcode::usub:
    case Opcode::ushl:
    case Opcode::ushr:
    case Opcode::uand:
    case Opcode::uor:
    case Opcode::uxor:
    case Opcode::usetp_eq:
    case Opcode::usetp_ne:
    case Opcode::usetp_lt:
    case Opcode::usetp_le:
    case Opcode::usetp_gt:
    case Opcode::usetp_ge:
    case Opcode::usetp_lt_s32:
    case Opcode::usetp_le_s32:
    case Opcode::usetp_gt_s32:
    case Opcode::usetp_ge_s32:
        word |= static_cast<uint64_t>(inst.dst) << encoding::reg0_shift;
        word |= static_cast<uint64_t>(inst.src0) << encoding::reg1_shift;
        word |= static_cast<uint64_t>(inst.src1) << encoding::reg2_shift;
        break;
    case Opcode::uaddi:
    case Opcode::ushli:
    case Opcode::ushri:
    case Opcode::uandi:
    case Opcode::uori:
    case Opcode::uxori:
    case Opcode::usetpi_eq:
    case Opcode::usetpi_ne:
    case Opcode::usetpi_lt:
    case Opcode::usetpi_le:
    case Opcode::usetpi_gt:
    case Opcode::usetpi_ge:
    case Opcode::usetpi_lt_s32:
    case Opcode::usetpi_le_s32:
    case Opcode::usetpi_gt_s32:
    case Opcode::usetpi_ge_s32:
        word |= static_cast<uint64_t>(inst.dst) << encoding::reg0_shift;
        word |= static_cast<uint64_t>(inst.src0) << encoding::reg1_shift;
        word |= immediate;
        break;
    case Opcode::fence_sh:
    case Opcode::sync:
    case Opcode::nop:
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
    if (is_mma_async(inst.opcode)) {
        inst.src2 = reg0;
        inst.src0 = reg1;
        inst.src1 = reg2;
        inst.imm = static_cast<int32_t>(word & ((1u << (2 * mma_spacing_field_bits)) - 1));
        return inst;
    }
    switch (inst.opcode) {
    case Opcode::sread:
    case Opcode::mov:
        inst.dst = reg0;
        inst.imm = static_cast<int32_t>(immediate);
        break;
    case Opcode::add:
    case Opcode::and_:
    case Opcode::or_:
    case Opcode::xor_:
    case Opcode::shl:
    case Opcode::shr:
    case Opcode::fadd:
    case Opcode::fmul:
    case Opcode::fsub:
    case Opcode::fmax:
    case Opcode::cvt_pack:
    case Opcode::setp_eq:
    case Opcode::setp_ne:
    case Opcode::setp_lt:
    case Opcode::setp_le:
    case Opcode::setp_gt:
    case Opcode::setp_ge:
    case Opcode::setp_lt_s32:
    case Opcode::setp_le_s32:
    case Opcode::setp_gt_s32:
    case Opcode::setp_ge_s32:
        inst.dst = reg0;
        inst.src0 = reg1;
        inst.src1 = reg2;
        break;
    case Opcode::fma:
        inst.dst = reg0;
        inst.src0 = reg1;
        inst.src1 = reg2;
        inst.src2 = static_cast<uint8_t>((word >> encoding::reg3_shift) & encoding::register_mask);
        break;
    case Opcode::ex2:
    case Opcode::rcp:
        inst.dst = reg0;
        inst.src0 = reg1;
        break;
    case Opcode::addi:
    case Opcode::shli:
    case Opcode::shri:
    case Opcode::xori:
    case Opcode::andi:
    case Opcode::ori:
    case Opcode::lds:
    case Opcode::ldg:
    case Opcode::ldg_v8:
    case Opcode::ldg_uni:
    case Opcode::lds_uni:
    case Opcode::acc_ld:
    case Opcode::acc_ld_v8:
    case Opcode::setpi_eq:
    case Opcode::setpi_ne:
    case Opcode::setpi_lt:
    case Opcode::setpi_le:
    case Opcode::setpi_gt:
    case Opcode::setpi_ge:
    case Opcode::setpi_lt_s32:
    case Opcode::setpi_le_s32:
    case Opcode::setpi_gt_s32:
    case Opcode::setpi_ge_s32:
        inst.dst = reg0;
        inst.src0 = reg1;
        inst.imm = static_cast<int32_t>(immediate);
        break;
    case Opcode::mbar_init_i:
    case Opcode::mbar_arrive_expect_tx_i:
        inst.src0 = reg0;
        inst.imm = static_cast<int32_t>(immediate);
        break;
    case Opcode::mbar_init_r:
    case Opcode::mbar_arrive:
    case Opcode::mbar_arrive_expect_tx_r:
        inst.src0 = reg0;
        inst.src1 = reg1;
        break;
    case Opcode::mbar_test_wait_i:
    case Opcode::mbar_try_wait_i:
        inst.dst = reg0;
        inst.src0 = reg1;
        inst.imm = static_cast<int32_t>(immediate);
        break;
    case Opcode::mbar_test_wait_r:
    case Opcode::mbar_try_wait_r:
        inst.dst = reg0;
        inst.src0 = reg1;
        inst.src1 = reg2;
        break;
    case Opcode::tma_load_1d:
    case Opcode::tma_load_2d:
    case Opcode::tma_load_3d:
    case Opcode::tma_load_4d:
    case Opcode::tma_store_1d:
    case Opcode::tma_store_2d:
    case Opcode::tma_store_3d:
    case Opcode::tma_store_4d:
        inst.src0 = reg0;
        inst.src1 = reg1;
        inst.src2 = reg2;
        inst.imm = static_cast<int32_t>((word & encoding::tma_descriptor_field_mask) * encoding::tma_descriptor_field_unit);
        break;
    case Opcode::mma_commit:
        inst.src0 = reg0;
        break;
    case Opcode::sts:
    case Opcode::stg:
    case Opcode::stg_v8:
    case Opcode::acc_st:
    case Opcode::acc_st_v8:
        inst.src0 = reg0;
        inst.src1 = reg1;
        inst.imm = static_cast<int32_t>(immediate);
        break;
    case Opcode::atom_add_uni:
    case Opcode::atom_and_uni:
    case Opcode::atom_xor_uni:
    case Opcode::atom_exch_uni: {
        inst.dst = reg0;
        inst.src0 = reg1;
        inst.src1 = reg2;
        auto offset = static_cast<uint32_t>(word & encoding::atomic_immediate_mask);
        if ((offset & (1u << 13)) != 0) offset |= ~static_cast<uint32_t>(encoding::atomic_immediate_mask);
        inst.imm = static_cast<int32_t>(offset);
        break;
    }
    case Opcode::bra:
    case Opcode::bra_uni:
    case Opcode::ssy:
        inst.imm = static_cast<int32_t>(immediate);
        break;
    case Opcode::umov:
    case Opcode::r2ur:
    case Opcode::ur2r:
        inst.dst = reg0;
        inst.src0 = reg1;
        break;
    case Opcode::umovi:
    case Opcode::usread:
        inst.dst = reg0;
        inst.imm = static_cast<int32_t>(immediate);
        break;
    case Opcode::uadd:
    case Opcode::usub:
    case Opcode::ushl:
    case Opcode::ushr:
    case Opcode::uand:
    case Opcode::uor:
    case Opcode::uxor:
    case Opcode::usetp_eq:
    case Opcode::usetp_ne:
    case Opcode::usetp_lt:
    case Opcode::usetp_le:
    case Opcode::usetp_gt:
    case Opcode::usetp_ge:
    case Opcode::usetp_lt_s32:
    case Opcode::usetp_le_s32:
    case Opcode::usetp_gt_s32:
    case Opcode::usetp_ge_s32:
        inst.dst = reg0;
        inst.src0 = reg1;
        inst.src1 = reg2;
        break;
    case Opcode::uaddi:
    case Opcode::ushli:
    case Opcode::ushri:
    case Opcode::uandi:
    case Opcode::uori:
    case Opcode::uxori:
    case Opcode::usetpi_eq:
    case Opcode::usetpi_ne:
    case Opcode::usetpi_lt:
    case Opcode::usetpi_le:
    case Opcode::usetpi_gt:
    case Opcode::usetpi_ge:
    case Opcode::usetpi_lt_s32:
    case Opcode::usetpi_le_s32:
    case Opcode::usetpi_gt_s32:
    case Opcode::usetpi_ge_s32:
        inst.dst = reg0;
        inst.src0 = reg1;
        inst.imm = static_cast<int32_t>(immediate);
        break;
    case Opcode::fence_sh:
    case Opcode::sync:
    case Opcode::nop:
    case Opcode::halt:
        break;
    }
    return inst;
}

std::vector<uint8_t> uniform_sources(const Instruction& inst) {
    if (is_mma_async(inst.opcode)) return {inst.src0, inst.src1, inst.src2};
    if (is_tma_opcode(inst.opcode)) {
        std::vector<uint8_t> result{inst.src0, inst.src2};
        for (uint32_t i = 0; i < tma_coordinate_count(inst.opcode); ++i) {
            result.push_back(static_cast<uint8_t>(inst.src1 + i));
        }
        return result;
    }
    switch (inst.opcode) {
    case Opcode::umov:
    case Opcode::uaddi:
    case Opcode::ushli:
    case Opcode::ushri:
    case Opcode::uandi:
    case Opcode::uori:
    case Opcode::uxori:
    case Opcode::usetpi_eq:
    case Opcode::usetpi_ne:
    case Opcode::usetpi_lt:
    case Opcode::usetpi_le:
    case Opcode::usetpi_gt:
    case Opcode::usetpi_ge:
    case Opcode::usetpi_lt_s32:
    case Opcode::usetpi_le_s32:
    case Opcode::usetpi_gt_s32:
    case Opcode::usetpi_ge_s32:
    case Opcode::ur2r:
    case Opcode::lds_uni:
    case Opcode::ldg_uni:
    case Opcode::mbar_init_i:
    case Opcode::mbar_arrive:
    case Opcode::mbar_arrive_expect_tx_i:
    case Opcode::mbar_test_wait_i:
    case Opcode::mbar_try_wait_i:
    case Opcode::mma_commit:
    case Opcode::acc_ld:
    case Opcode::acc_ld_v8:
    case Opcode::acc_st:
    case Opcode::acc_st_v8: return {inst.src0};
    case Opcode::uadd:
    case Opcode::usub:
    case Opcode::ushl:
    case Opcode::ushr:
    case Opcode::uand:
    case Opcode::uor:
    case Opcode::uxor:
    case Opcode::usetp_eq:
    case Opcode::usetp_ne:
    case Opcode::usetp_lt:
    case Opcode::usetp_le:
    case Opcode::usetp_gt:
    case Opcode::usetp_ge:
    case Opcode::usetp_lt_s32:
    case Opcode::usetp_le_s32:
    case Opcode::usetp_gt_s32:
    case Opcode::usetp_ge_s32:
    case Opcode::atom_add_uni:
    case Opcode::atom_and_uni:
    case Opcode::atom_xor_uni:
    case Opcode::atom_exch_uni:
    case Opcode::mbar_init_r:
    case Opcode::mbar_arrive_expect_tx_r:
    case Opcode::mbar_test_wait_r:
    case Opcode::mbar_try_wait_r: return {inst.src0, inst.src1};
    default: return {};
    }
}

std::optional<uint8_t> uniform_destination(const Instruction& inst) {
    if (writes_uniform_register(inst.opcode)) return inst.dst;
    return std::nullopt;
}

std::string mma_shape_name(const MmaShape& shape) {
    return "m" + std::to_string(shape.m) + "n" + std::to_string(shape.n) + "k" + std::to_string(shape.k);
}

std::optional<MmaShape> parse_mma_shape_name(const std::string& name) {
    MmaShape shape;
    std::size_t at = 0;
    for (const auto [letter, field] : {std::pair{'m', &shape.m}, std::pair{'n', &shape.n}, std::pair{'k', &shape.k}}) {
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
    switch (opcode) {
    case Opcode::sread: return "sread";
    case Opcode::mov: return "mov";
    case Opcode::add: return "add";
    case Opcode::addi: return "addi";
    case Opcode::and_: return "and";
    case Opcode::andi: return "andi";
    case Opcode::or_: return "or";
    case Opcode::ori: return "ori";
    case Opcode::xor_: return "xor";
    case Opcode::shl: return "shl";
    case Opcode::shr: return "shr";
    case Opcode::shli: return "shli";
    case Opcode::shri: return "shri";
    case Opcode::xori: return "xori";
    case Opcode::fadd: return "fadd";
    case Opcode::fmul: return "fmul";
    case Opcode::fsub: return "fsub";
    case Opcode::fmax: return "fmax";
    case Opcode::fma: return "fma";
    case Opcode::cvt_pack: return "cvt.pack.f16x2";
    case Opcode::ex2: return "ex2";
    case Opcode::rcp: return "rcp";
    case Opcode::setp_eq:
    case Opcode::setpi_eq: return "setp.eq.u32";
    case Opcode::setp_ne:
    case Opcode::setpi_ne: return "setp.ne.u32";
    case Opcode::setp_lt:
    case Opcode::setpi_lt: return "setp.lt.u32";
    case Opcode::setp_le:
    case Opcode::setpi_le: return "setp.le.u32";
    case Opcode::setp_gt:
    case Opcode::setpi_gt: return "setp.gt.u32";
    case Opcode::setp_ge:
    case Opcode::setpi_ge: return "setp.ge.u32";
    case Opcode::setp_lt_s32:
    case Opcode::setpi_lt_s32: return "setp.lt.s32";
    case Opcode::setp_le_s32:
    case Opcode::setpi_le_s32: return "setp.le.s32";
    case Opcode::setp_gt_s32:
    case Opcode::setpi_gt_s32: return "setp.gt.s32";
    case Opcode::setp_ge_s32:
    case Opcode::setpi_ge_s32: return "setp.ge.s32";
    case Opcode::mbar_init_i:
    case Opcode::mbar_init_r: return "mbar.init";
    case Opcode::mbar_arrive: return "mbar.arrive";
    case Opcode::mbar_arrive_expect_tx_i:
    case Opcode::mbar_arrive_expect_tx_r: return "mbar.arrive.expect_tx";
    case Opcode::mbar_test_wait_i:
    case Opcode::mbar_test_wait_r: return "mbar.test_wait";
    case Opcode::mbar_try_wait_i:
    case Opcode::mbar_try_wait_r: return "mbar.try_wait";
    case Opcode::tma_load_1d: return "tma.load.1d";
    case Opcode::tma_load_2d: return "tma.load.2d";
    case Opcode::tma_load_3d: return "tma.load.3d";
    case Opcode::tma_load_4d: return "tma.load.4d";
    case Opcode::tma_store_1d: return "tma.store.1d";
    case Opcode::tma_store_2d: return "tma.store.2d";
    case Opcode::tma_store_3d: return "tma.store.3d";
    case Opcode::tma_store_4d: return "tma.store.4d";
    case Opcode::acc_ld: return "acc.ld";
    case Opcode::acc_ld_v8: return "acc.ld.v8";
    case Opcode::acc_st: return "acc.st";
    case Opcode::acc_st_v8: return "acc.st.v8";
    case Opcode::mma_commit: return "mma.commit";
    case Opcode::lds: return "lds";
    case Opcode::sts: return "sts";
    case Opcode::ldg: return "ldg";
    case Opcode::stg: return "stg";
    case Opcode::ldg_v8: return "ldg.v8";
    case Opcode::stg_v8: return "stg.v8";
    case Opcode::ldg_uni: return "ldg.uni";
    case Opcode::lds_uni: return "lds.uni";
    case Opcode::fence_sh: return "fence.sh";
    case Opcode::nop: return "nop";
    case Opcode::atom_add_uni: return "atom.add.uni";
    case Opcode::atom_and_uni: return "atom.and.uni";
    case Opcode::atom_xor_uni: return "atom.xor.uni";
    case Opcode::atom_exch_uni: return "atom.exch.uni";
    case Opcode::bra: return "bra";
    case Opcode::bra_uni: return "bra.uni";
    case Opcode::ssy: return "ssy";
    case Opcode::sync: return "sync";
    case Opcode::umov:
    case Opcode::umovi: return "umov";
    case Opcode::uadd: return "uadd";
    case Opcode::uaddi: return "uaddi";
    case Opcode::usub: return "usub";
    case Opcode::ushl: return "ushl";
    case Opcode::ushr: return "ushr";
    case Opcode::ushli: return "ushli";
    case Opcode::ushri: return "ushri";
    case Opcode::uand: return "uand";
    case Opcode::uor: return "uor";
    case Opcode::uxor: return "uxor";
    case Opcode::uandi: return "uandi";
    case Opcode::uori: return "uori";
    case Opcode::uxori: return "uxori";
    case Opcode::usread: return "usread";
    case Opcode::r2ur: return "r2ur";
    case Opcode::ur2r: return "ur2r";
    case Opcode::usetp_eq:
    case Opcode::usetpi_eq: return "usetp.eq.u32";
    case Opcode::usetp_ne:
    case Opcode::usetpi_ne: return "usetp.ne.u32";
    case Opcode::usetp_lt:
    case Opcode::usetpi_lt: return "usetp.lt.u32";
    case Opcode::usetp_le:
    case Opcode::usetpi_le: return "usetp.le.u32";
    case Opcode::usetp_gt:
    case Opcode::usetpi_gt: return "usetp.gt.u32";
    case Opcode::usetp_ge:
    case Opcode::usetpi_ge: return "usetp.ge.u32";
    case Opcode::usetp_lt_s32:
    case Opcode::usetpi_lt_s32: return "usetp.lt.s32";
    case Opcode::usetp_le_s32:
    case Opcode::usetpi_le_s32: return "usetp.le.s32";
    case Opcode::usetp_gt_s32:
    case Opcode::usetpi_gt_s32: return "usetp.gt.s32";
    case Opcode::usetp_ge_s32:
    case Opcode::usetpi_ge_s32: return "usetp.ge.s32";
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

std::string format_instruction(
    const Instruction& inst, bool include_control, const std::vector<MmaShape>* mma_shapes) {
    std::ostringstream out;
    out << opcode_name(inst.opcode, mma_shapes);
    if (is_mma_async(inst.opcode)) {
        // The spacing is printed when it is not the default (as it is written).
        const auto operand = [&](uint8_t base, bool a_operand) {
            const auto spacing = mma_row_spacing(inst, a_operand);
            return "[" + ureg(base) +
                (spacing == mma_default_spacing ? std::string() : ", " + std::to_string(spacing)) + "]";
        };
        out << " [" << ureg(inst.src2) << "], " << operand(inst.src0, true) << ", "
            << operand(inst.src1, false);
        if (include_control) out << ' ' << format_control(inst.control);
        return out.str();
    }
    switch (inst.opcode) {
    case Opcode::sread:
        out << ' ' << reg(inst.dst) << ", " << special_name(inst.imm);
        break;
    case Opcode::mov:
        out << ' ' << reg(inst.dst) << ", " << inst.imm;
        break;
    case Opcode::add:
    case Opcode::and_:
    case Opcode::or_:
    case Opcode::xor_:
    case Opcode::shl:
    case Opcode::shr:
    case Opcode::fadd:
    case Opcode::fmul:
    case Opcode::fsub:
    case Opcode::fmax:
    case Opcode::cvt_pack:
        out << ' ' << reg(inst.dst) << ", " << reg(inst.src0) << ", " << reg(inst.src1);
        break;
    case Opcode::fma:
        out << ' ' << reg(inst.dst) << ", " << reg(inst.src0) << ", " << reg(inst.src1)
            << ", " << reg(inst.src2);
        break;
    case Opcode::ex2:
    case Opcode::rcp:
        out << ' ' << reg(inst.dst) << ", " << reg(inst.src0);
        break;
    case Opcode::setp_eq:
    case Opcode::setp_ne:
    case Opcode::setp_lt:
    case Opcode::setp_le:
    case Opcode::setp_gt:
    case Opcode::setp_ge:
    case Opcode::setp_lt_s32:
    case Opcode::setp_le_s32:
    case Opcode::setp_gt_s32:
    case Opcode::setp_ge_s32:
        out << " p" << static_cast<unsigned>(inst.dst) << ", " << reg(inst.src0)
            << ", " << reg(inst.src1);
        break;
    case Opcode::setpi_eq:
    case Opcode::setpi_ne:
    case Opcode::setpi_lt:
    case Opcode::setpi_le:
    case Opcode::setpi_gt:
    case Opcode::setpi_ge:
    case Opcode::setpi_lt_s32:
    case Opcode::setpi_le_s32:
    case Opcode::setpi_gt_s32:
    case Opcode::setpi_ge_s32:
        out << " p" << static_cast<unsigned>(inst.dst) << ", " << reg(inst.src0)
            << ", " << inst.imm;
        break;
    case Opcode::mbar_init_i:
        out << " [" << ureg(inst.src0) << "], " << inst.imm;
        break;
    case Opcode::mbar_init_r:
        out << " [" << ureg(inst.src0) << "], " << ureg(inst.src1);
        break;
    case Opcode::mbar_arrive:
        out << " [" << ureg(inst.src0) << ']';
        break;
    case Opcode::mbar_arrive_expect_tx_i:
        out << " [" << ureg(inst.src0) << "], " << inst.imm;
        break;
    case Opcode::mbar_arrive_expect_tx_r:
        out << " [" << ureg(inst.src0) << "], " << ureg(inst.src1);
        break;
    case Opcode::mbar_test_wait_i:
    case Opcode::mbar_try_wait_i:
        out << " p" << static_cast<unsigned>(inst.dst) << ", [" << ureg(inst.src0)
            << "], " << inst.imm;
        break;
    case Opcode::mbar_test_wait_r:
    case Opcode::mbar_try_wait_r:
        out << " p" << static_cast<unsigned>(inst.dst) << ", [" << ureg(inst.src0)
            << "], " << ureg(inst.src1);
        break;
    case Opcode::tma_load_1d:
    case Opcode::tma_load_2d:
    case Opcode::tma_load_3d:
    case Opcode::tma_load_4d:
        out << " [" << ureg(inst.src0) << "], [" << inst.imm << ", " << ureg(inst.src1) << "], ["
            << ureg(inst.src2) << ']';
        break;
    case Opcode::tma_store_1d:
    case Opcode::tma_store_2d:
    case Opcode::tma_store_3d:
    case Opcode::tma_store_4d:
        out << " [" << inst.imm << ", " << ureg(inst.src1) << "], [" << ureg(inst.src0) << "], ["
            << ureg(inst.src2) << ']';
        break;
    case Opcode::acc_ld:
    case Opcode::acc_ld_v8:
        out << ' ' << reg(inst.dst) << ", [" << ureg(inst.src0) << ", " << inst.imm << ']';
        break;
    case Opcode::acc_st:
    case Opcode::acc_st_v8:
        out << " [" << ureg(inst.src0) << ", " << inst.imm << "], " << reg(inst.src1);
        break;
    case Opcode::mma_commit:
        out << " [" << ureg(inst.src0) << ']';
        break;
    case Opcode::umov:
    case Opcode::r2ur:
        out << ' ' << ureg(inst.dst) << ", " << (inst.opcode == Opcode::umov ? ureg(inst.src0) : reg(inst.src0));
        break;
    case Opcode::ur2r:
        out << ' ' << reg(inst.dst) << ", " << ureg(inst.src0);
        break;
    case Opcode::umovi:
        out << ' ' << ureg(inst.dst) << ", " << inst.imm;
        break;
    case Opcode::usread:
        out << ' ' << ureg(inst.dst) << ", " << special_name(inst.imm);
        break;
    case Opcode::uadd:
    case Opcode::usub:
    case Opcode::ushl:
    case Opcode::ushr:
    case Opcode::uand:
    case Opcode::uor:
    case Opcode::uxor:
        out << ' ' << ureg(inst.dst) << ", " << ureg(inst.src0) << ", " << ureg(inst.src1);
        break;
    case Opcode::uaddi:
    case Opcode::ushli:
    case Opcode::ushri:
    case Opcode::uandi:
    case Opcode::uori:
    case Opcode::uxori:
        out << ' ' << ureg(inst.dst) << ", " << ureg(inst.src0) << ", " << inst.imm;
        break;
    case Opcode::usetp_eq:
    case Opcode::usetp_ne:
    case Opcode::usetp_lt:
    case Opcode::usetp_le:
    case Opcode::usetp_gt:
    case Opcode::usetp_ge:
    case Opcode::usetp_lt_s32:
    case Opcode::usetp_le_s32:
    case Opcode::usetp_gt_s32:
    case Opcode::usetp_ge_s32:
        out << " p" << static_cast<unsigned>(inst.dst) << ", " << ureg(inst.src0) << ", " << ureg(inst.src1);
        break;
    case Opcode::usetpi_eq:
    case Opcode::usetpi_ne:
    case Opcode::usetpi_lt:
    case Opcode::usetpi_le:
    case Opcode::usetpi_gt:
    case Opcode::usetpi_ge:
    case Opcode::usetpi_lt_s32:
    case Opcode::usetpi_le_s32:
    case Opcode::usetpi_gt_s32:
    case Opcode::usetpi_ge_s32:
        out << " p" << static_cast<unsigned>(inst.dst) << ", " << ureg(inst.src0) << ", " << inst.imm;
        break;
    case Opcode::addi:
    case Opcode::shli:
    case Opcode::shri:
    case Opcode::xori:
    case Opcode::andi:
    case Opcode::ori:
        out << ' ' << reg(inst.dst) << ", " << reg(inst.src0) << ", " << inst.imm;
        break;
    case Opcode::lds:
    case Opcode::ldg:
    case Opcode::ldg_v8:
        out << ' ' << reg(inst.dst) << ", " << address(inst.src0, inst.imm);
        break;
    case Opcode::ldg_uni:
    case Opcode::lds_uni:
        out << ' ' << reg(inst.dst) << ", " << address(ureg(inst.src0), inst.imm);
        break;
    case Opcode::sts:
    case Opcode::stg:
    case Opcode::stg_v8:
        out << ' ' << address(inst.src0, inst.imm) << ", " << reg(inst.src1);
        break;
    case Opcode::atom_add_uni:
    case Opcode::atom_and_uni:
    case Opcode::atom_xor_uni:
    case Opcode::atom_exch_uni:
        out << ' ' << reg(inst.dst) << ", " << address(ureg(inst.src0), inst.imm) << ", " << ureg(inst.src1);
        break;
    case Opcode::bra:
    case Opcode::bra_uni:
    case Opcode::ssy:
        out << ' ' << (inst.imm >= 0 ? "+" : "") << inst.imm;
        break;
    case Opcode::fence_sh:
    case Opcode::sync:
    case Opcode::nop:
    case Opcode::halt:
        break;
    }
    if (include_control) {
        out << ' ' << format_control(inst.control);
    }
    return out.str();
}

uint8_t fixed_latency(const Instruction& inst, const Timing& timing) {
    if (is_mma_async(inst.opcode)) return 0; // asynchronous: its depbar clears at the last operand read
    switch (inst.opcode) {
    case Opcode::sread:
    case Opcode::mov:
    case Opcode::add:
    case Opcode::addi:
    case Opcode::and_:
    case Opcode::andi:
    case Opcode::or_:
    case Opcode::ori:
    case Opcode::xor_:
    case Opcode::shl:
    case Opcode::shr:
    case Opcode::shli:
    case Opcode::shri:
    case Opcode::xori:
    // ur2r: its uniform operand, sampled with the round at issue + 1, rides
    // the vector arithmetic pipeline to the RF write port like mov.
    case Opcode::ur2r:
        return timing.alu;
    case Opcode::fadd:
    // The new float ops ride the FPU pipeline: fsub is fadd with a negation,
    // and fmul/fma/fmax/cvt are full-width float datapaths, not 1-cycle ALU.
    case Opcode::fmul:
    case Opcode::fsub:
    case Opcode::fmax:
    case Opcode::fma:
    case Opcode::cvt_pack:
        return timing.fadd;
    case Opcode::ex2:
    case Opcode::rcp:
        return timing.sfu;
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
    case Opcode::setp_lt_s32:
    case Opcode::setp_le_s32:
    case Opcode::setp_gt_s32:
    case Opcode::setp_ge_s32:
    case Opcode::setpi_lt_s32:
    case Opcode::setpi_le_s32:
    case Opcode::setpi_gt_s32:
    case Opcode::setpi_ge_s32:
        return timing.alu;
    case Opcode::mbar_test_wait_i:
    case Opcode::mbar_test_wait_r:
        return 1;
    case Opcode::lds:
    case Opcode::sts:
    case Opcode::ldg:
    case Opcode::stg:
    case Opcode::ldg_v8:
    case Opcode::stg_v8:
    case Opcode::ldg_uni:
    case Opcode::lds_uni:
    case Opcode::bra:
    case Opcode::bra_uni:
    case Opcode::ssy:
    case Opcode::sync:
    case Opcode::mbar_init_i:
    case Opcode::mbar_init_r:
    case Opcode::mbar_arrive:
    case Opcode::mbar_arrive_expect_tx_i:
    case Opcode::mbar_arrive_expect_tx_r:
    case Opcode::mbar_try_wait_i:
    case Opcode::mbar_try_wait_r:
    case Opcode::tma_load_1d:
    case Opcode::tma_load_2d:
    case Opcode::tma_load_3d:
    case Opcode::tma_load_4d:
    case Opcode::tma_store_1d:
    case Opcode::tma_store_2d:
    case Opcode::tma_store_3d:
    case Opcode::tma_store_4d:
    case Opcode::acc_ld:
    case Opcode::acc_ld_v8:
    case Opcode::acc_st:
    case Opcode::acc_st_v8:
    case Opcode::mma_commit:
    case Opcode::fence_sh:
    case Opcode::atom_add_uni:
    case Opcode::atom_and_uni:
    case Opcode::atom_xor_uni:
    case Opcode::atom_exch_uni:
    case Opcode::nop:
    case Opcode::halt:
        return 0;
    // The uniform register writers are not vector arithmetic: they present at
    // the URF write port urf.issue_to_port_cycles after issue (src/rf.hpp).
    // usetp rides the same pipeline and presents its predicate then.
    case Opcode::usetp_eq:
    case Opcode::usetp_ne:
    case Opcode::usetp_lt:
    case Opcode::usetp_le:
    case Opcode::usetp_gt:
    case Opcode::usetp_ge:
    case Opcode::usetpi_eq:
    case Opcode::usetpi_ne:
    case Opcode::usetpi_lt:
    case Opcode::usetpi_le:
    case Opcode::usetpi_gt:
    case Opcode::usetpi_ge:
    case Opcode::usetp_lt_s32:
    case Opcode::usetp_le_s32:
    case Opcode::usetp_gt_s32:
    case Opcode::usetp_ge_s32:
    case Opcode::usetpi_lt_s32:
    case Opcode::usetpi_le_s32:
    case Opcode::usetpi_gt_s32:
    case Opcode::usetpi_ge_s32:
    case Opcode::umov:
    case Opcode::umovi:
    case Opcode::uadd:
    case Opcode::uaddi:
    case Opcode::usub:
    case Opcode::ushl:
    case Opcode::ushr:
    case Opcode::ushli:
    case Opcode::ushri:
    case Opcode::uand:
    case Opcode::uor:
    case Opcode::uxor:
    case Opcode::uandi:
    case Opcode::uori:
    case Opcode::uxori:
    case Opcode::usread:
    case Opcode::r2ur:
        return 0;
    }
    throw SimError("latency requested for unknown opcode");
}

} // namespace nanosm

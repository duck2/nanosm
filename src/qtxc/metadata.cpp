// Instruction metadata for the toolchain (src/qtxc/): which vector registers
// an instruction reads (sources, issue_sources) and writes (destination),
// which predicate it writes, the opcode groups the scheduler, the assembler
// and the checks test for, and a program's register count. The uniform side
// (uniform_sources, uniform_destination) is src/isa.hpp's, the timing of a
// read or a write src/rf.hpp's. Nothing here parses, schedules or refuses.

#include "qtxc/internal.hpp"

#include <algorithm>

namespace nanosm {
namespace qtxc_internal {

// bra and bra.uni: a target (a displacement) and a control transfer.
bool is_branch(Opcode opcode) {
    return opcode == Opcode::bra || opcode == Opcode::bra_uni;
}

// Where control may go somewhere other than the next instruction: the
// branches, and sync (trash/divergence-2026-11-21.md), whose implicit edge
// goes to the pc of the entry it pops: a divergent bra's target or an ssy's
// label (the instruction after the sync when that is the pc). The
// scheduler drains at each of them.
bool transfers_control(Opcode opcode) {
    return is_branch(opcode) || opcode == Opcode::sync;
}

// The instructions whose immediate is a displacement to a label: the
// branches and ssy (its label).
bool has_displacement(Opcode opcode) {
    return is_branch(opcode) || opcode == Opcode::ssy;
}

bool is_mbar_test_wait(Opcode opcode) {
    return opcode == Opcode::mbar_test_wait_i || opcode == Opcode::mbar_test_wait_r;
}

bool is_mbar_try_wait(Opcode opcode) {
    return opcode == Opcode::mbar_try_wait_i || opcode == Opcode::mbar_try_wait_r;
}

bool is_mbar_predicate_writer(Opcode opcode) {
    return is_mbar_test_wait(opcode) || is_mbar_try_wait(opcode);
}

// The VECTOR registers an instruction reads (its uniform registers:
// uniform_sources, src/isa.hpp). acc.st and stg.v8 read their data later,
// beat by beat (issue_sources: only what the issue round reads).
std::vector<uint8_t> sources(const Instruction& inst) {
    if (is_mma_async(inst.opcode)) return {};
    switch (inst.opcode) {
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
    case Opcode::setp_ge_s32: return {inst.src0, inst.src1};
    case Opcode::fma: return {inst.src0, inst.src1, inst.src2};
    case Opcode::addi:
    case Opcode::shli:
    case Opcode::shri:
    case Opcode::xori:
    case Opcode::andi:
    case Opcode::ori:
    case Opcode::lds:
    case Opcode::ldg:
    case Opcode::ldg_v8:
    case Opcode::ex2:
    case Opcode::rcp:
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
    case Opcode::r2ur: return {inst.src0};
    case Opcode::sts:
    case Opcode::stg: return {inst.src0, inst.src1};
    case Opcode::stg_v8: {
        std::vector<uint8_t> result{inst.src0};
        for (uint16_t j = 0; j < 8; ++j) result.push_back(static_cast<uint8_t>(inst.src1 + j));
        return result;
    }
    case Opcode::acc_st: return {inst.src1};
    case Opcode::acc_st_v8: {
        std::vector<uint8_t> result;
        for (uint16_t j = 0; j < 8; ++j) result.push_back(static_cast<uint8_t>(inst.src1 + j));
        return result;
    }
    default: return {};
    }
}

std::optional<uint8_t> predicate_destination(const Instruction& inst) {
    if (is_setp(inst.opcode) || is_usetp(inst.opcode) || is_mbar_predicate_writer(inst.opcode)) return inst.dst;
    return std::nullopt;
}

std::optional<uint8_t> destination(const Instruction& inst) {
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
    case Opcode::fadd:
    case Opcode::fmul:
    case Opcode::fsub:
    case Opcode::fmax:
    case Opcode::fma:
    case Opcode::cvt_pack:
    case Opcode::ex2:
    case Opcode::rcp:
    case Opcode::ur2r:
    case Opcode::lds:
    case Opcode::lds_uni:
    case Opcode::ldg: return inst.dst;
    case Opcode::ldg_v8: return inst.dst;
    case Opcode::ldg_uni: return inst.dst;
    case Opcode::acc_ld:
    case Opcode::acc_ld_v8: return inst.dst;
    case Opcode::atom_add_uni:
    case Opcode::atom_and_uni:
    case Opcode::atom_xor_uni:
    case Opcode::atom_exch_uni: return inst.dst;
    default: return std::nullopt;
    }
}

uint16_t register_count(const AssemblyProgram& program) {
    uint16_t count = 0;
    for (const auto& inst : program.instructions) {
        for (const auto reg : sources(inst)) {
            count = std::max<uint16_t>(count, static_cast<uint16_t>(reg) + 1);
        }
        if (const auto dst = destination(inst); dst.has_value()) {
            const auto width = inst.opcode == Opcode::acc_ld_v8 || inst.opcode == Opcode::ldg_v8 ? 8 : 1;
            count = std::max<uint16_t>(count, static_cast<uint16_t>(*dst) + width);
        }
    }
    return count;
}

// The vector registers an instruction's issue reads (acc.st reads its data
// beat by beat and its coordinates from uniform registers: nothing here;
// stg.v8 only its base).
std::vector<uint8_t> issue_sources(const Instruction& inst) {
    if (inst.opcode == Opcode::acc_st || inst.opcode == Opcode::acc_st_v8) return {};
    if (inst.opcode == Opcode::stg_v8) return {inst.src0};
    return sources(inst);
}

} // namespace qtxc_internal
} // namespace nanosm

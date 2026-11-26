// Small pure functions that more than one of the machine's files uses. Included by
// src/machine/*.cpp only; each file gets its own copy (an anonymous namespace). The opcode
// classes not below are src/isa.hpp's.
#pragma once

#include "machine/machine.hpp"

#include <bit>
#include <cmath>
#include <optional>

namespace nanosm {

namespace {

// setp's and usetp's comparisons: unsigned 32-bit, the .s32 ones signed. An
// immediate arrives sign-extended from its 22-bit field either way.
inline bool compare_setp(Opcode opcode, uint32_t lhs, uint32_t rhs) {
    const auto signed_lhs = static_cast<int32_t>(lhs);
    const auto signed_rhs = static_cast<int32_t>(rhs);
    switch (opcode) {
    case Opcode::setp_eq:
    case Opcode::setpi_eq:
    case Opcode::usetp_eq:
    case Opcode::usetpi_eq: return lhs == rhs;
    case Opcode::setp_ne:
    case Opcode::setpi_ne:
    case Opcode::usetp_ne:
    case Opcode::usetpi_ne: return lhs != rhs;
    case Opcode::setp_lt:
    case Opcode::setpi_lt:
    case Opcode::usetp_lt:
    case Opcode::usetpi_lt: return lhs < rhs;
    case Opcode::setp_le:
    case Opcode::setpi_le:
    case Opcode::usetp_le:
    case Opcode::usetpi_le: return lhs <= rhs;
    case Opcode::setp_gt:
    case Opcode::setpi_gt:
    case Opcode::usetp_gt:
    case Opcode::usetpi_gt: return lhs > rhs;
    case Opcode::setp_ge:
    case Opcode::setpi_ge:
    case Opcode::usetp_ge:
    case Opcode::usetpi_ge: return lhs >= rhs;
    case Opcode::setp_lt_s32:
    case Opcode::setpi_lt_s32:
    case Opcode::usetp_lt_s32:
    case Opcode::usetpi_lt_s32: return signed_lhs < signed_rhs;
    case Opcode::setp_le_s32:
    case Opcode::setpi_le_s32:
    case Opcode::usetp_le_s32:
    case Opcode::usetpi_le_s32: return signed_lhs <= signed_rhs;
    case Opcode::setp_gt_s32:
    case Opcode::setpi_gt_s32:
    case Opcode::usetp_gt_s32:
    case Opcode::usetpi_gt_s32: return signed_lhs > signed_rhs;
    case Opcode::setp_ge_s32:
    case Opcode::setpi_ge_s32:
    case Opcode::usetp_ge_s32:
    case Opcode::usetpi_ge_s32: return signed_lhs >= signed_rhs;
    default: throw SimError("internal non-setp comparison");
    }
}

// Opcode classes by the unit that takes the instruction.
inline bool is_lsu_opcode(Opcode opcode) {
    return opcode == Opcode::ldg || opcode == Opcode::stg || opcode == Opcode::ldg_v8 ||
        opcode == Opcode::stg_v8 || opcode == Opcode::ldg_uni;
}

inline bool is_mma_command(Opcode opcode) {
    return is_mma_async(opcode) || opcode == Opcode::mma_commit;
}

inline bool is_acc_transfer(Opcode opcode) {
    return is_acc_load(opcode) || opcode == Opcode::acc_st || opcode == Opcode::acc_st_v8;
}

inline uint32_t beats_per_sector(const Config& config) {
    return (config.gmem_sector_bytes + config.gmem_bus_bytes_per_dram_tick - 1) /
        config.gmem_bus_bytes_per_dram_tick;
}

} // namespace

} // namespace nanosm

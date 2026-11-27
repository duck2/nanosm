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

// setp's and usetp's integer comparisons: unsigned 32-bit, the .s32 ones
// signed. An immediate arrives sign-extended from its 22-bit field either
// way. (setp.f32's are the FPU's: src/machine/arith.cpp float_compare.)
inline bool compare_setp(Opcode opcode, uint32_t lhs, uint32_t rhs) {
    const auto signed_lhs = static_cast<int32_t>(lhs);
    const auto signed_rhs = static_cast<int32_t>(rhs);
    switch (opcode_info(opcode).compare) {
    case Comparison::eq: return lhs == rhs;
    case Comparison::ne: return lhs != rhs;
    case Comparison::lt: return lhs < rhs;
    case Comparison::le: return lhs <= rhs;
    case Comparison::gt: return lhs > rhs;
    case Comparison::ge: return lhs >= rhs;
    case Comparison::lt_s: return signed_lhs < signed_rhs;
    case Comparison::le_s: return signed_lhs <= signed_rhs;
    case Comparison::gt_s: return signed_lhs > signed_rhs;
    case Comparison::ge_s: return signed_lhs >= signed_rhs;
    case Comparison::eq_f:
    case Comparison::ne_f:
    case Comparison::lt_f:
    case Comparison::le_f:
    case Comparison::gt_f:
    case Comparison::ge_f: throw SimError("internal: a float comparison off the FPU");
    case Comparison::none: break;
    }
    throw SimError("internal non-setp comparison");
}

// Opcode classes by the unit that takes the instruction.
inline bool is_lsu_opcode(Opcode opcode) {
    return opcode_info(opcode).unit == IssueUnit::lsu;
}

inline bool is_mma_command(Opcode opcode) {
    return opcode_info(opcode).unit == IssueUnit::mma;
}

inline bool is_acc_transfer(Opcode opcode) {
    const auto unit = opcode_info(opcode).unit;
    return unit == IssueUnit::acc_load || unit == IssueUnit::acc_store;
}

inline uint32_t beats_per_sector(const Config& config) {
    return (config.gmem_sector_bytes + config.gmem_bus_bytes_per_dram_tick - 1) /
        config.gmem_bus_bytes_per_dram_tick;
}

} // namespace

} // namespace nanosm

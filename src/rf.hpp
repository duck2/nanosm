#pragma once

#include "profile.hpp"

#include <algorithm>
#include <vector>

// The fixed RF read window.
//
// Arithmetic operands are not all read in the issue cycle. An accepted
// instruction owns a window of rf.arithmetic_read_cycles cycles starting at
// issue; its operand captures sit at fixed offsets inside that window, and
// execution begins at the window's final cycle. rf_read_plan computes that
// placement. It depends only on the instruction and the profile -- never on
// contention -- so the compiler and the simulator derive the identical
// schedule: structural conflicts delay admission, they never move an
// accepted read. The canonical window is one cycle, which reproduces the
// original capture-at-issue machine exactly.

namespace nanosm {

// One scheduled capture: read `reg` exactly `offset` cycles after issue.
struct RfRead {
    uint8_t reg;
    uint32_t offset;
};

// Fixed-latency ALU/FPU/SFU work (MOV/SREAD and setp included) runs through
// the arithmetic read pipeline. Everything else -- memory, TMA, mbar engines,
// and mbar.test_wait's predicate broadcast -- captures all operands in the
// issue cycle. The Timing{1, 1, 1} probe only asks "is this opcode in the
// fixed-latency class at all"; any nonzero dummy latencies would do.
inline bool uses_arithmetic_pipeline(const Instruction& inst) {
    if (inst.opcode == Opcode::mbar_test_wait_i) return false;
    if (inst.opcode == Opcode::mbar_test_wait_r) return false;
    return fixed_latency(inst, Timing{1, 1, 1}) != 0;
}

// Result latency from accepted issue: the reads spend window - 1 cycles
// before execution can begin, then the execution latency is unchanged.
inline uint32_t scheduled_fixed_latency(const Instruction& inst, const DeviceProfile& profile) {
    const auto window = uses_arithmetic_pipeline(inst) ? profile.rf_arithmetic_read_cycles : 1;
    return (window - 1) + fixed_latency(inst, profile.timing);
}

// Place this instruction's RF reads inside its fixed window.
//
//   - A register is read once, however many operand slots name it; the value
//     is broadcast from its latch.
//   - A bank serves rf_read_ports_per_bank distinct registers per cycle, so a
//     bank holding more operands than that needs several rounds.
//   - Banks are independent: reads in different banks may share a cycle.
//   - Each bank's rounds are pushed as late as possible ("right-aligned"), so
//     every instruction's last capture lands on the window's final cycle --
//     exactly when execution starts. That shared alignment is what gives all
//     arithmetic one uniform result latency.
//   - Operands that cannot fit the window make the instruction illegal for
//     this profile: qtxc rejects it at compile time, and the same throw fires
//     at issue for hand-written asm.
//
// Example, canonical 3 banks / 1 port, window 3, add r4, r0, r3:
// r0 and r3 both live in bank 0, so bank 0 needs two rounds, right-aligned
// to offsets 1 and 2; bank-independent operands would have joined at offset 2.
inline std::vector<RfRead> rf_read_plan(
    const Instruction& inst, const DeviceProfile& profile, const std::vector<uint8_t>& sources) {
    const auto window = uses_arithmetic_pipeline(inst) ? profile.rf_arithmetic_read_cycles : 1;

    // Distinct source registers, grouped by their RF bank (reg mod banks).
    std::vector<std::vector<uint8_t>> banks(profile.rf_banks);
    for (const auto reg : sources) {
        auto& bank = banks[reg % profile.rf_banks];
        if (std::find(bank.begin(), bank.end(), reg) == bank.end()) bank.push_back(reg);
    }

    std::vector<RfRead> plan;
    for (const auto& bank : banks) {
        if (bank.empty()) continue;
        const auto ports = profile.rf_read_ports_per_bank;
        const auto rounds = static_cast<uint32_t>(1 + (bank.size() - 1) / ports); // ceil(size / ports)
        if (rounds > window) {
            throw ProfileError(
                "RF operands exceed the fixed read window in '" + format_instruction(inst) + "'");
        }
        const auto first_round = window - rounds; // right-aligned
        for (std::size_t i = 0; i < bank.size(); ++i) {
            plan.push_back({bank[i], first_round + static_cast<uint32_t>(i / ports)});
        }
    }
    return plan;
}

} // namespace nanosm

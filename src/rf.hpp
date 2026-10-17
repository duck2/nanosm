#pragma once

#include "profile.hpp"

#include <algorithm>
#include <vector>

// The RF's timing: the one helper the compiler (qtxc's stall distances), the
// simulator (its bookings and pipelines) and the sanitizer (its ready /
// read-until horizons) all derive their numbers from (trash/rfc10.md
// section 5, slice I6a).
//
// The cycle convention is shmem port A's (rfc12 section 2): a DEPTH counts
// the register stages from a request's presentation up to and including the
// RAM.
//   - Issue register (fixed, one cycle): an instruction accepted at c0
//     presents its register numbers at the RF address inputs at c0 + 1.
//   - A read whose address is SAMPLED at S has its operand at the RF output
//     at S + Lr (rf.read_latency_cycles).
//   - A write PRESENTED at the RF write port at W is at the RAM at W + Wd
//     (rf.write_depth_cycles) and lands at that cycle's clock edge: a read
//     sampled at W + Wd still sees the old word (READ_FIRST), one sampled
//     from W + Wd + 1 sees the new one. W + Wd + 1 is the write's VISIBILITY.
//     There is no same-edge forwarding.
//   - The predicate file is flip-flops, not an SRAM: a predicate presented at
//     W is in the file from W + 1.
//
// Fixed arithmetic (the "arithmetic pipeline"): an accepted instruction owns
// a read window of rf.arithmetic_read_cycles rounds; its reads are
// right-aligned in it, so the last one is sampled at c0 + window; all its
// operands are at the RF output at c0 + window + Lr, when execution starts;
// its result reaches the write port E cycles later:
//
//     accept c0, sample c0 + 1 + offset, operands c0 + window + Lr,
//     port W = c0 + window + Lr + E, RAM W + Wd, visible W + Wd + 1
//
// Canonical (window 1, Lr 2, E 1, Wd 2): an add accepted at c0 samples at
// c1, has its operands at c3, presents its result at c4, writes the RAM at
// c6, and a dependent add sampling from c7 sees it -- it issues at c6, six
// cycles later.
//
// Everything else (memory, engines, mbar operations) also reads its operands
// through the issue register and the RF (one round, sampled at c0 + 1) and
// reaches its unit at DISPATCH, c0 + 1 + Lr, when its operands exist.

namespace nanosm {

// The issue register: one cycle from accepted issue to the RF address inputs.
inline constexpr uint32_t issue_register_stages = 1;

// One scheduled capture: read `reg` `offset` rounds into the read window,
// i.e. sample it issue_register_stages + offset cycles after issue.
struct RfRead {
    uint8_t reg;
    uint32_t offset;
};

// Fixed-latency ALU/FPU/SFU work (MOV/SREAD and setp included) runs through
// the arithmetic read pipeline. Everything else -- memory, TMA, mbar engines,
// and mbar.test_wait's predicate broadcast -- reads its operands in one round
// and is dispatched to its unit. The Timing{1, 1, 1} probe only asks "is this
// opcode in the fixed-latency class at all"; any nonzero dummy latencies
// would do.
inline bool uses_arithmetic_pipeline(const Instruction& inst) {
    if (inst.opcode == Opcode::mbar_test_wait_i) return false;
    if (inst.opcode == Opcode::mbar_test_wait_r) return false;
    return fixed_latency(inst, Timing{1, 1, 1}) != 0;
}

inline uint32_t read_window(const Instruction& inst, const DeviceProfile& profile) {
    return uses_arithmetic_pipeline(inst) ? profile.rf_arithmetic_read_cycles : 1;
}

// Accepted issue -> this read's address sample.
inline uint32_t issue_to_sample(const RfRead& read) {
    return issue_register_stages + read.offset;
}

// Accepted issue -> every operand at the RF output (execution starts, or the
// unit takes the instruction).
inline uint32_t issue_to_operands(const Instruction& inst, const DeviceProfile& profile) {
    return issue_register_stages + read_window(inst, profile) - 1 + profile.rf_read_latency_cycles;
}

// Accepted issue -> a memory / engine instruction reaches its unit.
inline uint32_t issue_to_dispatch(const DeviceProfile& profile) {
    return issue_register_stages + profile.rf_read_latency_cycles;
}

// Accepted issue -> a fixed result's presentation at the RF write port (or
// the predicate file's input).
inline uint32_t issue_to_port(const Instruction& inst, const DeviceProfile& profile) {
    return issue_to_operands(inst, profile) + fixed_latency(inst, profile.timing);
}

// Accepted issue -> the first cycle a read sample sees a fixed RF result.
inline uint32_t issue_to_visible(const Instruction& inst, const DeviceProfile& profile) {
    return issue_to_port(inst, profile) + profile.rf_write_depth_cycles + 1;
}

// Accepted issue -> the first cycle a fixed predicate result is in the file.
inline uint32_t issue_to_predicate_visible(const Instruction& inst, const DeviceProfile& profile) {
    return issue_to_port(inst, profile) + 1;
}

// The floor under every RF write's visibility, fixed or not: a destination
// is written from something the instruction read through the issue register
// and the RF (Lr), held in at least one register before the write port, then
// written through Wd. The fastest fixed result (window 1, E 1) meets it
// exactly; every asynchronous writer (an lds's gather, an atomic's return
// slot, an LSU assembly register, an acc.ld pipeline) is slower.
inline uint32_t earliest_rf_visible(const DeviceProfile& profile) {
    return issue_register_stages + profile.rf_read_latency_cycles + 1 + profile.rf_write_depth_cycles + 1;
}

// Accepted issue -> the earliest cycle this instruction's RF destination can
// become visible: exact for fixed results, the floor for the others. WAW and
// WAR are judged on it: a younger write lands at its visibility - 1.
inline uint32_t issue_to_visible_min(const Instruction& inst, const DeviceProfile& profile) {
    return uses_arithmetic_pipeline(inst) ? issue_to_visible(inst, profile) : earliest_rf_visible(profile);
}

// Place this instruction's RF reads inside its fixed window.
//
//   - A register is read once, however many operand slots name it; the value
//     is broadcast from its latch.
//   - A bank serves rf_read_ports_per_bank distinct registers per cycle, so a
//     bank holding more operands than that needs several rounds.
//   - Banks are independent: reads in different banks may share a cycle.
//   - Each bank's rounds are pushed as late as possible ("right-aligned"), so
//     every instruction's last capture lands on the window's final round --
//     all its operands reach the RF output together, Lr later. That shared
//     alignment is what gives all arithmetic one uniform result latency.
//   - Operands that cannot fit the window make the instruction illegal for
//     this profile: qtxc rejects it at compile time, and the same throw fires
//     at issue for hand-written asm.
//
// Example, canonical 3 banks / 1 port, window 3, add r4, r0, r3:
// r0 and r3 both live in bank 0, so bank 0 needs two rounds, right-aligned
// to offsets 1 and 2; bank-independent operands would have joined at offset 2.
inline std::vector<RfRead> rf_read_plan(
    const Instruction& inst, const DeviceProfile& profile, const std::vector<uint8_t>& sources) {
    const auto window = read_window(inst, profile);

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

// Accepted issue -> this instruction's sample of `reg`. A source the plan
// does not read at issue (acc.st / stg.v8 data, read beat by beat later)
// counts as sampled as early as possible, after the issue register: the
// conservative answer for a RAW check.
inline uint32_t issue_to_sample_of(const std::vector<RfRead>& plan, uint8_t reg) {
    for (const auto& read : plan) {
        if (read.reg == reg) return issue_to_sample(read);
    }
    return issue_register_stages;
}

} // namespace nanosm

#pragma once

#include "profile.hpp"

#include <algorithm>
#include <optional>
#include <string>
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
// RF reads (trash/rfc14.md sections 3 and 4): every instruction reads all
// its operands in ONE round, sampled at the same offset after its accepted
// issue -- the issue register's cycle, c0 + 1. At most one instruction is
// accepted per cycle, so the reads of two instructions never meet in one
// cycle: the accepted instruction's bank use, carried by the issue
// register, owns the RF read ports in its sample cycle, and the asynchronous
// readers (acc.st beats, the LSU's store data) take the ports it leaves free.
// No future read is booked anywhere. An instruction whose distinct source
// registers need more of one bank's read ports than the bank has is
// MALFORMED for this machine (rf_read_port_conflict): qtxc, the hand
// assembler and kernel load reject it; nothing serializes it.
//
// Fixed arithmetic (the "arithmetic pipeline"): all its operands are at the
// RF output Lr after the sample, when execution starts; its result reaches
// the write port E cycles later:
//
//     accept c0, sample c0 + 1, operands c0 + 1 + Lr,
//     port W = c0 + 1 + Lr + E, RAM W + Wd, visible W + Wd + 1
//
// Canonical (Lr 2, E 1, Wd 2): an add accepted at c0 samples at c1, has its
// operands at c3, presents its result at c4, writes the RAM at c6, and a
// dependent add sampling from c7 sees it -- it issues at c6, six cycles
// later.
//
// Everything else (memory, engines, mbar operations) reads its operands the
// same way and reaches its unit at DISPATCH, c0 + 1 + Lr, when its operands
// exist.
//
// The front end (trash/front-end-2026-10-18.md; two-word packets since
// trash/front-end-packets-2026-10-18.md): every warp issues from its
// instruction FIFO's head packet, the half its pc names. One instruction-
// memory read per cycle, one packet (the two words at an aligned 16-byte
// address), fixed at one cycle (HUMAN 10-18): a fetch granted at f (its
// address sampled at f's edge) has its packet at the memory output at f + 1,
// written into the warp's FIFO at that edge, and its words can issue from
// f + 2. Fetch decides from registered state only (the FIFO's valid packets
// and the fetches in flight), so a packet popped by the issue of its second
// word at t is refilled by a fetch at t + 1 at the earliest. A packet is a
// credit that goes round in fetch_to_issue + 2 cycles carrying two words, so
// a lone warp of independent instructions issues min(1, packets / 2) per
// cycle: one packet entry, two issues in four cycles; two or more, one per
// cycle.
//
// Branches (HUMAN 10-18: "it can be one cycle later ... long latency is
// ok"): nothing after a branch issues until it resolves, and it resolves in
// the issue register's cycle, c0 + 1, where a conditional branch reads its
// predicate like any operand (bra and the PT / !PT forms resolve there too).
// Not taken: the buffered fall-through issues from c0 + 2. Taken: the FIFO is
// cleared, the target is fetched from c0 + 2 and issues from c0 + 4 (at the
// soonest: the target's fetch competes with the other warps' fetches).
//
// qtxc models none of this (issuing later is always safe); only the
// predicate's sample point enters its distances (issue_to_predicate_sample).

namespace nanosm {

// The issue register: one cycle from accepted issue to the RF address inputs.
inline constexpr uint32_t issue_register_stages = 1;

// The instruction memory's read (fixed, not a profile field): an address
// sampled at f has its word at the memory output at f + 1.
inline constexpr uint32_t instruction_read_cycles = 1;

// A fetch granted at f -> its word can issue (it is in the FIFO's Q) from f + 2.
inline constexpr uint32_t fetch_to_issue = instruction_read_cycles + 1;

// Accepted issue -> the branch resolves: its predicate is sampled, its next
// PC is chosen. The issue register's cycle.
inline constexpr uint32_t issue_to_branch_resolve = issue_register_stages;

// Accepted issue -> a conditional branch's predicate sample.
inline constexpr uint32_t issue_to_predicate_sample = issue_to_branch_resolve;

// A branch accepted at c0: the warp's next instruction issues from c0 + this,
// not taken (the fall-through already buffered), and taken (the target
// fetched the cycle after the resolve, uncontended).
inline constexpr uint32_t branch_not_taken_interval = issue_to_branch_resolve + 1;
inline constexpr uint32_t branch_taken_interval = issue_to_branch_resolve + 1 + fetch_to_issue;

// Accepted issue -> the sample of every register the instruction reads (its
// one round). A source read later, beat by beat (acc.st / stg.v8 data), is
// read-owned from here too: the conservative answer for a RAW check.
inline constexpr uint32_t issue_to_sample = issue_register_stages;

// Fixed-latency ALU/FPU/SFU work (MOV/SREAD and setp included) runs through
// the arithmetic read pipeline. Everything else -- memory, TMA, mbar engines,
// and mbar.test_wait's predicate broadcast -- is dispatched to its unit. The
// Timing{1, 1, 1} probe only asks "is this opcode in the fixed-latency class
// at all"; any nonzero dummy latencies would do.
inline bool uses_arithmetic_pipeline(const Instruction& inst) {
    if (inst.opcode == Opcode::mbar_test_wait_i) return false;
    if (inst.opcode == Opcode::mbar_test_wait_r) return false;
    return fixed_latency(inst, Timing{1, 1, 1}) != 0;
}

// Accepted issue -> every operand at the RF output (execution starts, or the
// unit takes the instruction).
inline uint32_t issue_to_operands(const DeviceProfile& profile) {
    return issue_to_sample + profile.rf_read_latency_cycles;
}

// Accepted issue -> a memory / engine instruction reaches its unit.
inline uint32_t issue_to_dispatch(const DeviceProfile& profile) {
    return issue_to_operands(profile);
}

// Accepted issue -> a fixed result's presentation at the RF write port (or
// the predicate file's input).
inline uint32_t issue_to_port(const Instruction& inst, const DeviceProfile& profile) {
    return issue_to_operands(profile) + fixed_latency(inst, profile.timing);
}

// Accepted issue -> the first cycle a read sample sees a fixed RF result.
inline uint32_t issue_to_visible(const Instruction& inst, const DeviceProfile& profile) {
    return issue_to_port(inst, profile) + profile.rf_write_depth_cycles + 1;
}

// Accepted issue -> the first cycle a fixed predicate result is in the file.
// A conditional branch reading it may issue issue_to_predicate_sample earlier:
// a setp accepted at c0 (canonical, alu 1) is in the file from c5, so a
// bra.uni on it issues from c4 and samples it at c5.
inline uint32_t issue_to_predicate_visible(const Instruction& inst, const DeviceProfile& profile) {
    return issue_to_port(inst, profile) + 1;
}

// The floor under every RF write's visibility, fixed or not: a destination
// is written from something the instruction read through the issue register
// and the RF (Lr), held in at least one register before the write port, then
// written through Wd. The fastest fixed result (E 1) meets it exactly; every
// asynchronous writer (an lds's gather, an atomic's return slot, an LSU
// assembly register, an acc.ld pipeline) is slower.
inline uint32_t earliest_rf_visible(const DeviceProfile& profile) {
    return issue_to_sample + profile.rf_read_latency_cycles + 1 + profile.rf_write_depth_cycles + 1;
}

// Accepted issue -> the earliest cycle this instruction's RF destination can
// become visible: exact for fixed results, the floor for the others. WAW and
// WAR are judged on it: a younger write lands at its visibility - 1.
inline uint32_t issue_to_visible_min(const Instruction& inst, const DeviceProfile& profile) {
    return uses_arithmetic_pipeline(inst) ? issue_to_visible(inst, profile) : earliest_rf_visible(profile);
}

// The registers an instruction reads in its one round, from the source
// registers its issue reads: each distinct register once. A register named
// by several operand slots is read once; the value is broadcast from its
// latch.
inline std::vector<uint8_t> rf_read_round(const std::vector<uint8_t>& sources) {
    std::vector<uint8_t> round;
    for (const auto reg : sources) {
        if (std::find(round.begin(), round.end(), reg) == round.end()) round.push_back(reg);
    }
    return round;
}

// A bank (reg mod rf.banks) serves rf.read_ports_per_bank distinct
// registers per cycle -- one at the canonical profile. An instruction whose
// round needs more of one bank is malformed: the text names the instruction,
// the registers and their bank (callers add the line or pc). None when every
// bank fits.
//
// Example, canonical 3 banks / 1 port: add r4, r0, r3 reads r0 and r3, both
// in bank 0 -- malformed. add r4, r0, r0 reads r0 once -- fine.
inline std::optional<std::string> rf_read_port_conflict(
    const Instruction& inst, const DeviceProfile& profile, const std::vector<uint8_t>& sources) {
    std::vector<std::vector<uint8_t>> banks(profile.rf_banks);
    for (const auto reg : rf_read_round(sources)) banks[reg % profile.rf_banks].push_back(reg);
    for (uint32_t bank = 0; bank < banks.size(); ++bank) {
        if (banks[bank].size() <= profile.rf_read_ports_per_bank) continue;
        std::string registers;
        for (std::size_t i = 0; i < banks[bank].size(); ++i) {
            registers += (i == 0 ? "" : i + 1 == banks[bank].size() ? " and " : ", ") +
                ("r" + std::to_string(banks[bank][i]));
        }
        const auto ports = profile.rf_read_ports_per_bank;
        return "malformed instruction '" + format_instruction(inst, false, &profile.mma_shapes) +
            "': its distinct sources " +
            registers + " are in one RF bank (" + std::to_string(bank) + "), which has " +
            std::to_string(ports) + (ports == 1 ? " read port" : " read ports");
    }
    return std::nullopt;
}

} // namespace nanosm

#pragma once

#include "profile.hpp"

#include <algorithm>
#include <optional>
#include <string>
#include <vector>

// The RF's timing: the one helper the compiler (qtxc's stall distances), the
// simulator (its bookings and pipelines) and the sanitizer (its ready /
// read-until horizons) all derive their numbers from.
//
// The cycle convention is shmem port A's: a DEPTH counts
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
// RF reads: every instruction reads all
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
// same way and reaches its unit at DISPATCH, when its operands exist:
// c0 + 1 + Lr, unless the profile's URF is as slow as its RF (then one later:
// issue_to_dispatch).
//
// The front end: every warp issues from its
// instruction FIFO's head packet, the half its pc names. One instruction-
// memory read per cycle, one packet (the two words at an aligned 16-byte
// address), fixed at one cycle: a fetch granted at f (its
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
// Branches (a branch may cost a cycle more: a long branch latency is
// acceptable): nothing after a branch issues until it resolves, and it resolves in
// the issue register's cycle, c0 + 1, where a conditional branch reads its
// predicate like any operand (bra and the PT / !PT forms resolve there too).
// Not taken: the buffered fall-through issues from c0 + 2. Taken: the FIFO is
// cleared, the target is fetched from c0 + 2 and issues from c0 + 4 (at the
// soonest: the target's fetch competes with the other warps' fetches).
// Divergence uses the same stage: a
// divergent bra reads every lane of its predicate at c0 + 1 and either
// pushes the taken lanes and falls through (c0 + 2), or moves the whole mask
// (c0 + 2 or c0 + 4); a sync pops at c0 + 1 from a registered copy of the
// stack's top and switches to it -- a popped pc equal to the sync's pc + 8
// goes on with the words already fetched (c0 + 2), any other pc is a
// redirect (c0 + 4); an ssy pushes at c0 + 1 and does not hold its warp
// (c0 + 1). A
// pop's refill of the top copy is a one-cycle stack-memory read the warp's
// next control instruction (c0 + 3 at the soonest) never waits for.
//
// qtxc models none of this (issuing later is always safe); only the
// predicate's sample point enters its distances (issue_to_predicate_sample).
//
// The uniform register file:
// per warp urf.registers 32-bit registers, TWO read ports and ONE write port,
// the same edge convention as the RF with its own depths Lu
// (urf.read_latency_cycles) and Wu (urf.write_depth_cycles):
//   - Reads: an instruction's uniform registers (at most two, so the two
//     ports always suffice) are sampled in its one read round, at the issue
//     register's cycle c0 + 1, beside its vector registers; their values are
//     at the URF output Lu later (no later than the vector operands: Lu <=
//     Lr, validated). The exceptions are a tma,
//     which reads N + 2 registers over R rounds, two per round, and an
//     mma.async, which reads three over two rounds (uniform_read_rounds
//     below); each holds the two ports for its later rounds.
//   - Writes: the only writers are uniform arithmetic (the SM's one uniform
//     ALU, Eu = urf.alu_latency_cycles) and r2ur (lane 0 of a vector
//     register). Each presents its result at the URF write port at the SAME
//     fixed L = urf.issue_to_port_cycles after its issue -- the shorter path
//     padded to the longer:
//         uniform ALU  accept c0, sample c0 + 1, operands c0 + 1 + Lu, result c0 + 1 + Lu + Eu
//         r2ur         accept c0, RF sample c0 + 1, lane 0 at c0 + 1 + Lr
//     so L >= 1 + max(Lu + Eu, Lr). At most one instruction is accepted per
//     cycle, so two writes never present in one cycle: one write port, no
//     write calendar, no arbitration. A write presented at W is at the RAM
//     at W + Wu and lands at that edge: a sample at W + Wu still sees the old
//     value, one from W + Wu + 1 the new one.
//   - Canonical (Lu 1, Eu 1, Wu 1, Lr 2, L 3): a writer accepted at c0
//     presents at c3, lands at c4, is visible from c5; a dependent uniform
//     reader (sampling at its issue + 1) issues from c4, four cycles later.
//   - Reuse of a uniform register while it is still being read cannot
//     happen: every read is at issue + 1 except a tma's and an mma.async's
//     later rounds, the last at issue + R <= issue + 3; a write issued after the tma (at issue
//     + 1 at the earliest) lands at issue + 1 + L + Wu >= issue + 5 (L >= 3,
//     Wu >= 1, validated), after the tma's last read. So a tma followed at
//     once by a rewrite of its operand registers is correct (no rule for
//     software).
//   - ur2r rD, urS (a uniform register into every lane of a vector
//     register) is a fixed-latency VECTOR result: its uniform operand is
//     sampled at c0 + 1 and it rides the arithmetic pipeline like mov, its
//     write booked on the RF write calendar at issue_to_port.
//   - usetp pD, urA, urB | imm (the uniform compare) is uniform arithmetic
//     whose result
//     is one bit for the predicate file: it rides the URF write pipeline like
//     every uniform-ALU instruction (operands sampled at c0 + 1, the ALU at
//     c0 + 1 + Lu) and at the write port's cycle, c0 + L, presents pD -- the
//     same value in every lane -- at the predicate file's input instead of
//     the URF write port, and leaves the pipeline. The predicate file is
//     flip-flops: pD is in it from c0 + L + 1 (canonical c0 + 4), so a
//     bra.uni on it issues from c0 + L (canonical 3 cycles later; a vector
//     setp's is 4). Its write follows the predicate file's one rule (no port,
//     no calendar): a predicate writer issues only once the older write to
//     the same predicate is in the file, so one write per predicate is ever in
//     flight, and writes to different predicates are different flip-flops.

namespace nanosm {

// The issue register: one cycle from accepted issue to the RF address inputs.
inline constexpr uint32_t issue_register_stages = 1;

// BLACK BOX SHARED: instruction memory read
// The instruction memory's read (fixed, not a profile field): an address
// sampled at f has its word at the memory output at f + 1.
inline constexpr uint32_t instruction_read_cycles = 1;
// END BLACK BOX SHARED: instruction memory read

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

// The most uniform read rounds an instruction that travels the dispatch
// pipeline takes: the mma.async's two (uniform_read_rounds below; its third
// register, urC, is sampled at issue + 2). The tma's later rounds feed its own
// holding register, not the dispatch pipeline.
inline constexpr uint32_t dispatch_uniform_rounds = 2;

// Accepted issue -> a memory / engine instruction reaches its unit, with
// every operand: its vector operands at the RF output (issue + 1 + Lr), its
// uniform operands at the URF output (round r's at issue + 1 + r + Lu). The
// dispatch pipeline is one fixed depth for every instruction (it keeps issue
// order), so it covers the matmul's second round: 1 + max(Lr, Lu + 1).
// Canonical (Lr 2, Lu 1): issue + 3, whatever the round; only a profile
// with Lu = Lr dispatches one cycle later than its vector operands arrive.
inline uint32_t issue_to_dispatch(const DeviceProfile& profile) {
    return issue_to_sample + std::max(profile.rf_read_latency_cycles,
        profile.urf_read_latency_cycles + dispatch_uniform_rounds - 1);
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
// asynchronous writer (an lds's gather, an atomic's return slot, the load
// buffer, an acc.ld pipeline) is slower.
inline uint32_t earliest_rf_visible(const DeviceProfile& profile) {
    return issue_to_sample + profile.rf_read_latency_cycles + 1 + profile.rf_write_depth_cycles + 1;
}

// Accepted issue -> the earliest cycle this instruction's RF destination can
// become visible: exact for fixed results, the floor for the others. WAW and
// WAR are judged on it: a younger write lands at its visibility - 1.
inline uint32_t issue_to_visible_min(const Instruction& inst, const DeviceProfile& profile) {
    return uses_arithmetic_pipeline(inst) ? issue_to_visible(inst, profile) : earliest_rf_visible(profile);
}

// BLACK BOX SHARED: uniform register file ports
// The URF's ports (fixed: two read ports, one write port).
inline constexpr uint32_t urf_read_ports = 2;
inline constexpr uint32_t urf_write_ports = 1;
// END BLACK BOX SHARED: uniform register file ports

// Accepted issue -> the sample of every uniform register the instruction
// reads (the same round as its vector registers).
inline constexpr uint32_t issue_to_urf_sample = issue_register_stages;

// Accepted issue -> its uniform operands at the URF output.
inline uint32_t issue_to_urf_operands(const DeviceProfile& profile) {
    return issue_to_urf_sample + profile.urf_read_latency_cycles;
}

// The longest uniform-register writer's path to the URF write port (the
// least L the profile may declare): the issue register, then the uniform
// ALU's URF read and execution or r2ur's vector RF read.
inline uint32_t urf_longest_writer_path(const DeviceProfile& profile) {
    return issue_register_stages + std::max(profile.urf_read_latency_cycles + profile.urf_alu_latency_cycles,
        profile.rf_read_latency_cycles);
}

// Accepted issue -> a uniform-register writer's presentation at the URF
// write port (L, the same for every writer).
inline uint32_t issue_to_urf_port(const DeviceProfile& profile) {
    return profile.urf_issue_to_port_cycles;
}

// Accepted issue -> the first cycle a sample sees the written uniform
// register (canonical 5).
inline uint32_t issue_to_urf_visible(const DeviceProfile& profile) {
    return issue_to_urf_port(profile) + profile.urf_write_depth_cycles + 1;
}

// Accepted issue -> the first cycle usetp's predicate is in the file: it is
// presented at the write port's cycle (L) and the flip-flops take it at that
// edge. A bra.uni reading it may issue issue_to_predicate_sample earlier: a
// usetp accepted at c0 (canonical L 3) is in the file from c4, so a bra.uni
// on it issues from c3 and samples it at c4.
inline uint32_t issue_to_uniform_predicate_visible(const DeviceProfile& profile) {
    return issue_to_urf_port(profile) + 1;
}

// The tma's operand read. The URF's two read
// ports serve the whole SM. A tma.Nd reads N + 2 registers by operand
// position (a register named twice is read twice), two per round: round 0
// samples urS and urM at the issue register's cycle like any read, then one
// round per cycle samples the coordinates in pairs. R = 1 + ceil(N / 2): two
// rounds for N = 1, 2 and three for N = 3, 4. tma.load.4d accepted at T
// (canonical depths, Lu 1):
//     T      accepted; one slot of the TMA queue is counted as taken
//     T+1    round 0 samples urS, urM
//     T+2    round 1 samples c0, c1    urS, urM arrive in the holding register
//     T+3    round 2 samples c2, c3    c0, c1 arrive
//     T+4                              c2, c3 arrive; the command is pushed into the queue
//     T+5    the engine may start on it (its descriptor read)
// While the command still has rounds to start (T+1 .. T+R-1) issue refuses
// every instruction that reads a uniform register: its sample, a cycle after
// its issue, would meet a later round. Uniform writers that read nothing and
// vector-only instructions are never refused. So commands are accepted at
// most every R cycles. qtxc does not model the rounds: it keeps every operand
// register's writer the usual uniform distance from the tma (all N + 2
// visible by its first read).
inline constexpr uint32_t tma_operand_rounds(uint32_t coordinates) {
    return 1 + (coordinates + 1) / 2;
}

// The registers round `round` of a tma samples, in operand order.
inline std::vector<uint8_t> tma_operand_round(const Instruction& inst, uint32_t round) {
    if (round == 0) return {inst.src0, inst.src2};
    std::vector<uint8_t> registers;
    const uint32_t coordinates = tma_coordinate_count(inst.opcode);
    for (uint32_t i = 2 * (round - 1); i < std::min(coordinates, 2 * round); ++i) {
        registers.push_back(static_cast<uint8_t>(inst.src1 + i));
    }
    return registers;
}

// Accepted issue -> the sample of uniform read round `round` (round 0 is the
// issue register's cycle, like every read).
inline constexpr uint32_t issue_to_uniform_round_sample(uint32_t round) {
    return issue_to_urf_sample + round;
}

// Accepted issue -> the sample of round `round` of a tma.
inline constexpr uint32_t issue_to_tma_round_sample(uint32_t round) {
    return issue_to_uniform_round_sample(round);
}

// Accepted issue -> the cycle a tma's last words reach the holding register
// and the command is pushed into the TMA queue (the engine sees it from the
// next cycle): canonical T + 3 for N = 1, 2 and T + 4 for N = 3, 4.
inline uint32_t issue_to_tma_command(uint32_t coordinates, const DeviceProfile& profile) {
    return issue_to_tma_round_sample(tma_operand_rounds(coordinates) - 1) + profile.urf_read_latency_cycles;
}

// Every instruction's uniform reads as rounds of at most two registers (the
// URF's read ports), round r sampled at issue + 1 + r. One round for every
// instruction that reads at most two (each distinct register once,
// rf_read_round below), none for one that reads none. Two instructions read
// more, by operand position (a register named twice is read twice), with the
// TMA's round counter (src/machine/machine.hpp UrfReadRounds):
//   tma.Nd     urS, urM | c0, c1 | c2, c3     (tma_operand_round)
//   mma.async  urA, urB | urC                 (urC, the
//              result's tmem pointer, joins its dispatch entry from the URF
//              output at issue + 2 + Lu, by its dispatch: issue_to_dispatch)
// While an instruction still has rounds to start, issue refuses every
// instruction that reads a uniform register (its sample would meet a later
// round); uniform writers that read nothing and vector-only instructions are
// never refused. qtxc does not model the rounds: every operand register's
// writer is kept the usual distance from the instruction's issue, so all
// are visible by its first round.
inline std::vector<uint8_t> rf_read_round(const std::vector<uint8_t>& sources);

inline std::vector<std::vector<uint8_t>> uniform_read_rounds(const Instruction& inst) {
    if (is_tma_opcode(inst.opcode)) {
        std::vector<std::vector<uint8_t>> rounds;
        for (uint32_t round = 0; round < tma_operand_rounds(tma_coordinate_count(inst.opcode)); ++round) {
            rounds.push_back(tma_operand_round(inst, round));
        }
        return rounds;
    }
    if (is_mma_async(inst.opcode)) return {{inst.src0, inst.src1}, {inst.src2}};
    const auto round = rf_read_round(uniform_sources(inst));
    if (round.empty()) return {};
    return {round};
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

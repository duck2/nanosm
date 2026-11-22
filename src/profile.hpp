#pragma once

#include "isa.hpp"
#include "layout.hpp"

#include <stdint.h>
#include <stdexcept>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace nanosm {

class ProfileError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// One profile is consumed by qtxc, qtxas and the simulator. Core quantities are
// measured in core cycles; controller quantities are measured in DRAM ticks.
struct DeviceProfile {
    std::string name;
    uint32_t lanes = 0;
    uint32_t rf_banks = 0;
    // The SM's warp slots (core.warps): the scheduler's and the front end's
    // per-warp state is sized by it, and a launch may not ask for more warps
    // (the RF rows must fit too). Canonical 16. 1..64.
    uint32_t core_warps = 16;
    // Instruction FIFO entries per warp (core.instruction_buffer_packets): each
    // entry is one fetch packet, two 64-bit instruction words read together
    // from an aligned 16-byte address.
    // Canonical 2, configurable down to one. 1..16.
    uint32_t core_instruction_buffer_packets = 2;
    // Entries of each warp's divergence stack
    // (core.divergence_stack_entries): (mask, pc) entries of 48 bits, one
    // memory of core.warps x entries for the SM. Canonical 8: the default
    // machine has a divergence stack, and an implementation may leave it out. 0
    // is legal and means no stack: a kernel with ssy, sync or a predicated bra
    // is refused by qtxc, the assembler and kernel load. 0..64.
    uint32_t core_divergence_stack_entries = 8;
    // Launch flags (core.launch_flags; README "Launch"): how many of each
    // warp's six depbars,
    // from sb5 downward, the launch may count once in every warp before the
    // first instruction; each is decremented in every warp when the launch
    // copy has delivered its part of the kernel's launch block (.param,
    // .mbar, .data) from global memory into shared memory. Canonical 4. 1..6:
    // there is no untimed way for the block to be in shared memory, so a
    // kernel with sections needs at least one flag to wait on.
    uint32_t core_launch_flags = 4;
    // One row of one bank holds one lane-wide register (lanes x 4 B). A
    // launch fits when active warps x the kernel's rows in every bank fit
    // (and the warps fit core.warps). Canonical 352: 3 x 352 x 64 B = 66 KiB,
    // about the size of shmem; 16 warps x 22 rows (66 registers each).
    uint32_t rf_rows_per_bank = 0;
    // Distinct registers one bank serves per cycle. Every instruction reads
    // its operands in one round (src/timing.hpp), so its distinct sources in a
    // bank may not exceed this (one at the canonical profile).
    uint32_t rf_read_ports_per_bank = 0;
    uint32_t rf_write_ports_per_bank = 0;
    // Lr: register stages of the RF itself
    // from a read's address sample to its operand at the RF output. >= 1.
    uint32_t rf_read_latency_cycles = 2;
    // Wd: register stages from a write's presentation at the RF write port to
    // the RAM, inclusive (shared memory port A's convention): a write
    // presented at W is written at the clock edge ending W + Wd; a read
    // sampled at W + Wd still sees the old word (READ_FIRST), one sampled from
    // W + Wd + 1 the new one. >= 1.
    uint32_t rf_write_depth_cycles = 2;
    // The uniform register file:
    // urf.registers 32-bit registers per warp (canonical 32, LUTRAM), two read
    // ports and one write port (src/timing.hpp). Its depths follow the RF's
    // convention: Lu (urf.read_latency_cycles) register stages from a read's
    // sample to its value at the output, Wu (urf.write_depth_cycles) from a
    // write's presentation at the write port to the RAM (canonical 1 each,
    // >= 1). Lu may not exceed rf.read_latency_cycles: an instruction's
    // uniform operands must reach its unit with its vector ones.
    uint32_t urf_registers = 32;
    uint32_t urf_read_latency_cycles = 1;
    uint32_t urf_write_depth_cycles = 1;
    // Eu: the uniform ALU's register stages (canonical 1, >= 1).
    uint32_t urf_alu_latency_cycles = 1;
    // L: every uniform-register writer (uniform arithmetic, r2ur) presents its
    // result at the URF write port this many cycles after its issue, so one
    // issue per cycle is one URF write per cycle. It must cover the longest
    // writer: issue register + max(Lu + Eu, rf.read_latency_cycles); a
    // shorter writer is padded to it. Canonical 3 (1 + max(1 + 1, 2)).
    uint32_t urf_issue_to_port_cycles = 3;
    uint32_t shmem_bytes = 0;
    uint32_t shmem_banks = 0;
    // Load slots: an admitted lds / lds.uni holds one from
    // admission until the RF write path accepts its gathered vector. >= 1.
    uint32_t shmem_load_slots = 2;
    // Store slots: an admitted sts holds one from admission
    // until its last write round is granted. >= 1.
    uint32_t shmem_store_slots = 2;
    // D: register stages from a shmem grant to the RAM --
    // the read's sample, the write's edge. The same for A-read, A-write and the
    // port-B write, so accesses reach the RAM in grant order. >= 1.
    uint32_t shmem_port_a_depth = 2;
    // Lb: register stages from a read's RAM sample to its words at the port
    // output. >= 1.
    uint32_t shmem_read_latency_cycles = 2;
    // C: register stages of the RMW engine's ALU, from a
    // read's words at the port output to the update ready for its write. The
    // mbar update and the uniform atomics use it. >= 1.
    uint32_t shmem_rmw_compute_cycles = 2;
    // tmem: fine_banks word-wide banks of
    // words_per_fine_bank 32-bit words each. Every fine bank has two read and
    // two write services per cycle: one read and one write for the MMA's
    // accumulator, one read for the acc.ld engine and one write for the
    // acc.st engine. Nobody arbitrates for them.
    uint32_t tmem_fine_banks = 0;
    uint32_t tmem_words_per_fine_bank = 0;
    uint32_t tmem_max_vector_width = 0;
    // The tmem RAM's own depths: register stages
    // from a request leaving its producer to the RAM -- a read's sample, a
    // write's edge, the same for reads and writes -- and from a read's sample
    // to its words at the output. Nothing waits after a request leaves its
    // producer. >= 1.
    uint32_t tmem_port_depth = 2;
    uint32_t tmem_read_latency_cycles = 2;
    // The logical MMA shapes this device runs (configurable, not hardcoded).
    // Shape i is the mma.async block's opcode
    // pair i (src/isa.hpp: mma_block_first + 2i, + 2i + 1 for init); a
    // kernel using an opcode or a shape the list lacks is refused at assembly
    // and at load. validate_device_profile states the rules a shape follows.
    std::vector<MmaShape> mma_shapes;
    uint32_t mma_core_m = 0;
    uint32_t mma_core_n = 0;
    uint32_t mma_core_k = 0;
    uint32_t mma_queue_commands = 0;
    // The physical core IS the rate: one core step enters per cycle once its
    // operands are fetched. Cheaper throughput is declared by smaller core
    // dimensions, not by a stall parameter.
    uint32_t mma_result_latency_cycles = 0;
    // mma.layout in the profile: the layout the tensor core reads both
    // operands in (src/layout.hpp), fixed for a device, not selected by
    // individual MMA instructions. swizzle64 is the only supported swizzle
    // (each operand's row spacing is the instruction's); dense is an
    // unswizzled diagnostic baseline, compact in the command's shape.
    TileLayout mma_layout = TileLayout::dense;
    // SFU: one separate unit whose throughput is independent of lane count.
    // It serves sfu_lanes lanes per cycle, so one warp op occupies it for
    // ceil(lanes / sfu_lanes) cycles. timing.sfu is derived from these.
    uint32_t sfu_lanes = 0;
    uint32_t sfu_latency_cycles = 0;
    uint32_t mbar_wait_slots = 0;
    uint32_t mbar_try_wait_timeout_cycles = 0;
    uint32_t tma_queue_commands = 0;
    uint32_t tma_sector_credits = 0;
    // TMA store sector registers (32 B flops each): a store sector's source
    // read reserves one at its A-read grant; it frees when its last word has
    // gone into the egress FIFO. One register turns over every D + Lb +
    // ceil(sector words / pluck) cycles. >= 1 (canonical 1).
    uint32_t tma_store_sector_registers = 1;
    // The register stages of the walker's start: between the cycle a TMA
    // command's descriptor words are back and the cycle its walker registers
    // are loaded (the three coordinate x stride products and the add chain
    // into row 0's address; src/machine/machine.hpp TmaWalkerStart). 0..8; canonical 2
    // ("some configurable delay to those multipliers"). 0 is
    // the timing before the stages existed: products in the return cycle.
    uint32_t tma_walker_start_cycles = 2;
    uint64_t gmem_bytes = 0;

    Timing timing;
    uint64_t core_clock_hz = 0;
    uint64_t dram_clock_hz = 0;
    uint32_t gmem_sector_bytes = 0;
    uint32_t gmem_bus_bytes_per_dram_tick = 0;
    uint32_t gmem_latency_dram_ticks = 0;
    // The data bus is one bidirectional bus (a DRAM channel's): the minimum
    // number of idle DRAM ticks between the last data beat in one direction
    // and the first beat in the other. Ticks the bus was idle anyway count.
    // 0: a change of direction is free.
    uint32_t gmem_turnaround_dram_ticks = 0;
    uint32_t gmem_queue_sectors = 0;

    // The SM's load/store unit. All config,
    // none canonical by ruling.
    // Machine-wide outstanding ldg/stg commands (vector, scalar, uniform).
    uint32_t lsu_active_commands = 0;
    // The LSU's two buffers: one load buffer, one store buffer, each held
    // only while data is in it (machine.hpp AssemblyRegister). A v8 instruction's
    // lanes pass through them in fixed groups of this many lanes: the load
    // buffer is 8 x load_group_lanes words (256 bits per lane), written to
    // the RF as eight writes with the group's lanes enabled; the store
    // buffer 8 x store_group_lanes words, filled by eight RF read beats.
    // Powers of two dividing the lanes; the store buffer also holds a scalar
    // store's lanes, so store_group_lanes x 8 >= lanes.
    uint32_t lsu_load_group_lanes = 0;
    uint32_t lsu_store_group_lanes = 0;
    // Ordered read IDs: 0 = TMA (one 32 B sector register),
    // 1 = the LSU's one ordered read stream (every ldg / ldg.uni / ldg.v8:
    // data returns in request order and passes through the one load buffer).
    // >= 2; streams above 1 are not used. Scatter writes take free IDs above
    // these.
    uint32_t lsu_read_streams = 0;
    // Ingress FIFO read port and the shared egress FIFO write port, 32-bit
    // words per cycle.
    uint32_t lsu_pluck_words_per_cycle = 0;
    // Ingress / egress FIFO capacity in 32 B sector slots.
    uint32_t lsu_ingress_sectors = 0;
    uint32_t lsu_egress_sectors = 0;
    // BRAM FIFO write-to-read latency in core cycles: a beat (ingress) or a
    // completed sector slot (egress) written in cycle t is readable at
    // t + latency. Start 1 (registered-output simple dual port); never 0.
    uint32_t lsu_ingress_latency_cycles = 0;
    uint32_t lsu_egress_latency_cycles = 0;
    // Outstanding scalar (gather, scatter, uniform) sectors = free-ID tags.
    uint32_t lsu_sectors_in_flight = 0;
};

using Config = DeviceProfile;

DeviceProfile load_device_profile(const std::string& path);
DeviceProfile parse_device_profile(std::string_view source);
uint64_t device_profile_fingerprint(const DeviceProfile& profile);
std::string format_profile_fingerprint(uint64_t fingerprint);
void validate_device_profile(const DeviceProfile& profile);

// The shape an mma.async opcode means on this profile; nullopt for any other
// opcode, or for an opcode of the block past the end of the shape list.
std::optional<MmaShape> mma_shape_of(const DeviceProfile& profile, Opcode opcode);

// Steps of one K block a command of `shape` walks before it visits the same
// accumulator block again: (m / core_m) x (n / core_n).
uint32_t mma_blocks_per_k(const DeviceProfile& profile, const MmaShape& shape);
// The fewest such steps that keep a shape with k > core_k from stalling on
// its own accumulator (validate_device_profile explains the bound):
// shmem port_a_depth + read_latency + mma result_latency + tmem port_depth.
uint32_t mma_min_blocks_per_k(const DeviceProfile& profile);

// Legal but wasteful profile choices (e.g. an MMA shape that reads only part
// of the last 16-byte piece of each swizzle64 operand row). Warnings, not
// errors.
std::vector<std::string> device_profile_warnings(const DeviceProfile& profile);
// Prints each distinct warning to stderr once per process; parse_device_profile
// calls it after validation.
void report_profile_warnings(const DeviceProfile& profile);

} // namespace nanosm

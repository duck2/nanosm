#pragma once

#include "isa.hpp"

#include <stdint.h>
#include <stdexcept>
#include <string>
#include <string_view>

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
    // One row of one bank holds one lane-wide register (lanes x 4 B). A
    // launch fits when active warps x the kernel's rows in every bank fit
    // (no separate warp-count field). Canonical 352: 3 x 352 x 64 B = 66 KiB,
    // about the size of shmem; 16 warps x 22 rows (66 registers each).
    uint32_t rf_rows_per_bank = 0;
    // Distinct registers one bank serves per cycle. Every instruction reads
    // its operands in one round (src/rf.hpp), so its distinct sources in a
    // bank may not exceed this (one at the canonical profile).
    uint32_t rf_read_ports_per_bank = 0;
    uint32_t rf_write_ports_per_bank = 0;
    // Lr (trash/rfc10.md section 5, I6a): register stages of the RF itself
    // from a read's address sample to its operand at the RF output. >= 1.
    uint32_t rf_read_latency_cycles = 2;
    // Wd: register stages from a write's presentation at the RF write port to
    // the RAM, inclusive (the port-A convention, rfc12 section 2): a write
    // presented at W is written at the clock edge ending W + Wd; a read
    // sampled at W + Wd still sees the old word (READ_FIRST), one sampled from
    // W + Wd + 1 the new one. >= 1.
    uint32_t rf_write_depth_cycles = 2;
    uint32_t shmem_bytes = 0;
    uint32_t shmem_banks = 0;
    // Load slots (rfc12 section 3): an admitted lds / lds.uni holds one from
    // admission until the RF write path accepts its gathered vector. >= 1.
    uint32_t shmem_load_slots = 2;
    // Store slots (rfc12 section 3): an admitted sts holds one from admission
    // until its last write round is granted. >= 1.
    uint32_t shmem_store_slots = 2;
    // D (rfc12 section 2): register stages from a shmem grant to the RAM --
    // the read's sample, the write's edge. The same for A-read, A-write and the
    // port-B write, so accesses reach the RAM in grant order. >= 1.
    uint32_t shmem_port_a_depth = 2;
    // Lb: register stages from a read's RAM sample to its words at the port
    // output. >= 1.
    uint32_t shmem_read_latency_cycles = 2;
    // C (rfc12 section 2): register stages of the RMW engine's ALU, from a
    // read's words at the port output to the update ready for its write. The
    // mbar update and the uniform atomics (trash/rfc13.md) use it. >= 1.
    uint32_t shmem_rmw_compute_cycles = 2;
    uint32_t tmem_banks = 0;
    uint32_t tmem_words_per_bank = 0;
    uint32_t tmem_fine_banks = 0;
    uint32_t tmem_max_vector_width = 0;
    // False: one whole SIMT transfer owns a coarse bank. True: one load and
    // one store may independently use its read/write sides.
    bool tmem_simt_read_write_duplex = false;
    uint32_t mma_m = 0;
    uint32_t mma_n = 0;
    uint32_t mma_k = 0;
    uint32_t mma_core_m = 0;
    uint32_t mma_core_n = 0;
    uint32_t mma_core_k = 0;
    uint32_t mma_queue_commands = 0;
    // The physical core IS the rate: one core step enters per cycle once its
    // operands are fetched. Cheaper throughput is declared by smaller core
    // dimensions, not by a stall parameter.
    uint32_t mma_result_latency_cycles = 0;
    // false is the dense diagnostic baseline; true is RFC05's 64x64x8 XOR layout.
    bool mma_xor_layout = false;
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
    // ceil(sector words / pluck) cycles. >= 1 (canonical 1, HUMAN).
    uint32_t tma_store_sector_registers = 1;
    uint64_t gmem_bytes = 0;

    Timing timing;
    uint64_t core_clock_hz = 0;
    uint64_t dram_clock_hz = 0;
    uint32_t gmem_sector_bytes = 0;
    uint32_t gmem_bus_bytes_per_dram_tick = 0;
    uint32_t gmem_latency_dram_ticks = 0;
    uint32_t gmem_queue_sectors = 0;

    // The SM's load/store unit (trash/rfc09.md section 3, B2). All config,
    // none canonical by ruling.
    // Machine-wide outstanding ldg/stg commands (vector, scalar, uniform).
    uint32_t lsu_active_commands = 0;
    // 64 B (lane-wide) assembly registers. A command owns one from admission
    // to its end, so admitted commands never exceed this number.
    uint32_t lsu_staging_vectors = 0;
    // Ordered read IDs (rfc09 3a): 0 = TMA (one 32 B sector register),
    // 1..n-1 = vector streams assigned round-robin to vector loads (they
    // pluck straight into the command's assembly register). >= 2. Scalar
    // gathers and ldg.uni ride free IDs above these.
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

} // namespace nanosm

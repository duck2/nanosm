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
    uint32_t rf_rows_per_bank = 0;
    uint32_t rf_read_ports_per_bank = 0;
    uint32_t rf_write_ports_per_bank = 0;
    // Fixed arithmetic read window; execution latency starts at its final cycle.
    uint32_t rf_arithmetic_read_cycles = 1;
    uint32_t shmem_bytes = 0;
    uint32_t shmem_banks = 0;
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
    uint64_t gmem_bytes = 0;

    Timing timing;
    uint64_t core_clock_hz = 0;
    uint64_t dram_clock_hz = 0;
    uint32_t gmem_sector_bytes = 0;
    uint32_t gmem_bus_bytes_per_dram_tick = 0;
    uint32_t gmem_latency_dram_ticks = 0;
    uint32_t gmem_queue_sectors = 0;
};

using Config = DeviceProfile;

DeviceProfile load_device_profile(const std::string& path);
DeviceProfile parse_device_profile(std::string_view source);
uint64_t device_profile_fingerprint(const DeviceProfile& profile);
std::string format_profile_fingerprint(uint64_t fingerprint);
void validate_device_profile(const DeviceProfile& profile);

} // namespace nanosm

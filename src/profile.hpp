#pragma once

#include "isa.hpp"

#include <stdint.h>
#include <stdexcept>
#include <string>

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
    uint32_t shmem_bytes = 0;
    uint32_t shmem_banks = 0;
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
uint64_t device_profile_fingerprint(const DeviceProfile& profile);
std::string format_profile_fingerprint(uint64_t fingerprint);
void validate_device_profile(const DeviceProfile& profile);

} // namespace nanosm

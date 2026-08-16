#include "profile.hpp"
#include "hash.hpp"

#include <nlohmann/json.hpp>

#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

namespace nanosm {
namespace {

using json = nlohmann::json;

json profile_json(const DeviceProfile& p) {
    return {
        {"name", p.name},
        {"lanes", p.lanes},
        {"rf", {
            {"banks", p.rf_banks},
            {"rows_per_bank", p.rf_rows_per_bank},
            {"read_ports_per_bank", p.rf_read_ports_per_bank},
            {"write_ports_per_bank", p.rf_write_ports_per_bank},
        }},
        {"shmem", {
            {"bytes", p.shmem_bytes},
            {"banks", p.shmem_banks},
        }},
        {"core", {
            {"clock_hz", p.core_clock_hz},
            {"alu_latency_cycles", p.timing.alu},
            {"fadd_latency_cycles", p.timing.fadd},
        }},
        {"gmem", {
            {"bytes", p.gmem_bytes},
            {"dram_clock_hz", p.dram_clock_hz},
            {"sector_bytes", p.gmem_sector_bytes},
            {"bus_bytes_per_dram_tick", p.gmem_bus_bytes_per_dram_tick},
            {"latency_dram_ticks", p.gmem_latency_dram_ticks},
            {"queue_sectors", p.gmem_queue_sectors},
        }},
    };
}

template<typename T>
T field(const json& object, const char* name) {
    try {
        return object.at(name).get<T>();
    } catch (const json::exception& error) {
        throw ProfileError(std::string("device profile field '") + name + "': " + error.what());
    }
}

} // namespace

void validate_device_profile(const DeviceProfile& p) {
    if (p.name.empty()) throw ProfileError("device profile: name is empty");
    if (p.name.find_first_of(" \t,;") != std::string::npos) {
        throw ProfileError("device profile: name may not contain whitespace, comma or semicolon");
    }
    if (p.lanes == 0 || p.rf_banks == 0 || p.rf_rows_per_bank == 0 ||
        p.rf_read_ports_per_bank == 0 || p.rf_write_ports_per_bank == 0 ||
        p.shmem_bytes == 0 || p.shmem_banks == 0 || p.gmem_bytes == 0 ||
        p.timing.alu == 0 || p.timing.fadd == 0 || p.core_clock_hz == 0 ||
        p.dram_clock_hz == 0 || p.gmem_sector_bytes == 0 ||
        p.gmem_sector_bytes % 4 != 0 || p.gmem_bus_bytes_per_dram_tick == 0 ||
        p.gmem_queue_sectors == 0) {
        throw ProfileError("device profile: zero, range or alignment constraint failed");
    }
}

DeviceProfile load_device_profile(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw ProfileError("cannot open device profile '" + path + "'");

    json document;
    try {
        input >> document;
    } catch (const json::exception& error) {
        throw ProfileError("cannot parse device profile '" + path + "': " + error.what());
    }

    DeviceProfile p;
    p.name = field<std::string>(document, "name");
    p.lanes = field<uint32_t>(document, "lanes");
    const auto& rf = document.at("rf");
    p.rf_banks = field<uint32_t>(rf, "banks");
    p.rf_rows_per_bank = field<uint32_t>(rf, "rows_per_bank");
    p.rf_read_ports_per_bank = field<uint32_t>(rf, "read_ports_per_bank");
    p.rf_write_ports_per_bank = field<uint32_t>(rf, "write_ports_per_bank");
    const auto& shmem = document.at("shmem");
    p.shmem_bytes = field<uint32_t>(shmem, "bytes");
    p.shmem_banks = field<uint32_t>(shmem, "banks");
    const auto& core = document.at("core");
    p.core_clock_hz = field<uint64_t>(core, "clock_hz");
    p.timing.alu = field<uint8_t>(core, "alu_latency_cycles");
    p.timing.fadd = field<uint8_t>(core, "fadd_latency_cycles");
    const auto& gmem = document.at("gmem");
    p.gmem_bytes = field<uint64_t>(gmem, "bytes");
    p.dram_clock_hz = field<uint64_t>(gmem, "dram_clock_hz");
    p.gmem_sector_bytes = field<uint32_t>(gmem, "sector_bytes");
    p.gmem_bus_bytes_per_dram_tick = field<uint32_t>(gmem, "bus_bytes_per_dram_tick");
    p.gmem_latency_dram_ticks = field<uint32_t>(gmem, "latency_dram_ticks");
    p.gmem_queue_sectors = field<uint32_t>(gmem, "queue_sectors");
    validate_device_profile(p);
    return p;
}

uint64_t device_profile_fingerprint(const DeviceProfile& p) {
    validate_device_profile(p);
    return fnv1a_64(profile_json(p).dump());
}

std::string format_profile_fingerprint(uint64_t fingerprint) {
    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << fingerprint;
    return out.str();
}

} // namespace nanosm

#include "profile.hpp"
#include "hash.hpp"
#include "mma_numeric.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
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
            {"read_latency_cycles", p.rf_read_latency_cycles},
            {"write_depth_cycles", p.rf_write_depth_cycles},
        }},
        {"shmem", {
            {"bytes", p.shmem_bytes},
            {"banks", p.shmem_banks},
            {"load_slots", p.shmem_load_slots},
            {"store_slots", p.shmem_store_slots},
            {"port_a_depth", p.shmem_port_a_depth},
            {"read_latency_cycles", p.shmem_read_latency_cycles},
            {"rmw_compute_cycles", p.shmem_rmw_compute_cycles},
        }},
        {"tmem", {
            {"banks", p.tmem_banks},
            {"words_per_bank", p.tmem_words_per_bank},
            {"fine_banks", p.tmem_fine_banks},
            {"max_vector_width", p.tmem_max_vector_width},
            {"simt_read_write_duplex", p.tmem_simt_read_write_duplex},
        }},
        {"mma", {
            {"m", p.mma_m}, {"n", p.mma_n}, {"k", p.mma_k},
            {"core_m", p.mma_core_m}, {"core_n", p.mma_core_n}, {"core_k", p.mma_core_k},
            {"queue_commands", p.mma_queue_commands},
            {"result_latency_cycles", p.mma_result_latency_cycles},
            {"layout", p.mma_xor_layout ? "xor64" : "dense"},
            {"numeric_model", mma_numeric_model},
        }},
        {"sync", {
            {"mbar_wait_slots", p.mbar_wait_slots},
            {"mbar_try_wait_timeout_cycles", p.mbar_try_wait_timeout_cycles},
        }},
        {"tma", {
            {"queue_commands", p.tma_queue_commands},
            {"sector_credits", p.tma_sector_credits},
            {"store_sector_registers", p.tma_store_sector_registers},
        }},
        {"core", {
            {"clock_hz", p.core_clock_hz},
            {"alu_latency_cycles", p.timing.alu},
            {"fadd_latency_cycles", p.timing.fadd},
            {"warps", p.core_warps},
            {"instruction_buffer_packets", p.core_instruction_buffer_packets},
        }},
        {"sfu", {
            {"lanes", p.sfu_lanes},
            {"latency_cycles", p.sfu_latency_cycles},
        }},
        {"gmem", {
            {"bytes", p.gmem_bytes},
            {"dram_clock_hz", p.dram_clock_hz},
            {"sector_bytes", p.gmem_sector_bytes},
            {"bus_bytes_per_dram_tick", p.gmem_bus_bytes_per_dram_tick},
            {"latency_dram_ticks", p.gmem_latency_dram_ticks},
            {"queue_sectors", p.gmem_queue_sectors},
        }},
        {"lsu", {
            {"active_commands", p.lsu_active_commands},
            {"staging_vectors", p.lsu_staging_vectors},
            {"read_streams", p.lsu_read_streams},
            {"pluck_words_per_cycle", p.lsu_pluck_words_per_cycle},
            {"ingress_sectors", p.lsu_ingress_sectors},
            {"egress_sectors", p.lsu_egress_sectors},
            {"sectors_in_flight", p.lsu_sectors_in_flight},
            {"ingress_latency_cycles", p.lsu_ingress_latency_cycles},
            {"egress_latency_cycles", p.lsu_egress_latency_cycles},
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
    // The RF's own depths (trash/rfc10.md section 5): nonzero register stages.
    if (p.rf_read_latency_cycles == 0 || p.rf_read_latency_cycles > 8 ||
        p.rf_write_depth_cycles == 0 || p.rf_write_depth_cycles > 8) {
        throw ProfileError("device profile: rf read_latency_cycles and write_depth_cycles must be in 1..8 "
            "register stages");
    }
    // The longest fixed-latency dependency must fit the five-bit stall field:
    // a consumer issued d cycles after its producer samples at d + 1 (the
    // issue register), the producer's result is visible from 1 + Lr + E + Wd
    // + 1 (src/rf.hpp), so d = Lr + E + Wd + 1.
    const uint32_t longest_execution = std::max({p.timing.alu, p.timing.fadd, p.timing.sfu});
    if (p.rf_read_latency_cycles + longest_execution + p.rf_write_depth_cycles + 1 > encoding::stall_mask) {
        throw ProfileError("device profile: the longest fixed-latency dependency (RF depths and execution) "
            "must fit the five-bit stall field");
    }
    if (p.core_warps == 0 || p.core_warps > 64) {
        throw ProfileError("device profile: core warps must be in 1..64");
    }
    if (p.core_instruction_buffer_packets == 0 || p.core_instruction_buffer_packets > 16) {
        throw ProfileError("device profile: core instruction_buffer_packets must be in 1..16");
    }
    if (p.shmem_load_slots == 0 || p.shmem_store_slots == 0) {
        throw ProfileError("device profile: shmem load and store slots must be at least 1");
    }
    // Every latency is a nonzero number of register stages (rfc12 section 2).
    if (p.shmem_port_a_depth == 0 || p.shmem_port_a_depth > 64 ||
        p.shmem_read_latency_cycles == 0 || p.shmem_read_latency_cycles > 64 ||
        p.shmem_rmw_compute_cycles == 0 || p.shmem_rmw_compute_cycles > 64) {
        throw ProfileError("device profile: shmem port_a_depth, read_latency_cycles and rmw_compute_cycles "
            "must be in 1..64 register stages");
    }
    if (p.tma_store_sector_registers == 0 || p.tma_store_sector_registers > 64) {
        throw ProfileError("device profile: tma store_sector_registers must be in 1..64");
    }
    if (p.sfu_lanes == 0 || p.sfu_lanes > p.lanes) {
        throw ProfileError("device profile: sfu lanes must be in 1..lanes");
    }
    if (p.sfu_latency_cycles == 0 || p.sfu_latency_cycles > 24) {
        throw ProfileError("device profile: sfu latency must be in 1..24 cycles");
    }
    // timing.sfu = occupancy - 1 + latency must leave stall counts encodable.
    if (p.timing.sfu == 0 || p.timing.sfu < p.sfu_latency_cycles || p.timing.sfu > 30) {
        throw ProfileError("device profile: effective sfu latency must fit the stall field");
    }
    if (p.name.empty()) throw ProfileError("device profile: name is empty");
    if (p.name.find_first_of(" \t,;") != std::string::npos) {
        throw ProfileError("device profile: name may not contain whitespace, comma or semicolon");
    }
    if (p.lanes == 0 || p.rf_banks == 0 || p.rf_rows_per_bank == 0 ||
        p.rf_read_ports_per_bank == 0 || p.rf_write_ports_per_bank == 0 ||
        p.shmem_bytes == 0 || p.shmem_banks == 0 || p.tmem_banks != 2 ||
        p.tmem_words_per_bank == 0 || p.tmem_fine_banks != 16 ||
        p.tmem_max_vector_width == 0 || p.tmem_max_vector_width > 8 || p.gmem_bytes == 0 ||
        p.timing.alu == 0 || p.timing.fadd == 0 || p.core_clock_hz == 0 ||
        p.dram_clock_hz == 0 || p.gmem_sector_bytes == 0 ||
        p.gmem_sector_bytes % 4 != 0 || p.gmem_bus_bytes_per_dram_tick == 0 ||
        p.gmem_queue_sectors == 0) {
        throw ProfileError("device profile: zero, range or alignment constraint failed");
    }
    if ((p.mbar_wait_slots == 0) != (p.mbar_try_wait_timeout_cycles == 0)) {
        throw ProfileError(
            "device profile: mbar wait slots and try-wait timeout must both be zero or nonzero");
    }
    if ((p.tma_queue_commands == 0) != (p.tma_sector_credits == 0)) {
        throw ProfileError(
            "device profile: tma command queue and sector credits must both be zero or nonzero");
    }
    if (p.mma_m == 0 || p.mma_n == 0 || p.mma_k == 0 || p.mma_core_m == 0 ||
        p.mma_core_n == 0 || p.mma_core_k == 0 || p.mma_queue_commands == 0 ||
        p.mma_result_latency_cycles == 0 ||
        p.mma_m % p.mma_core_m != 0 || p.mma_n % p.mma_core_n != 0 ||
        p.mma_k % p.mma_core_k != 0) {
        throw ProfileError("device profile: invalid MMA shape, timing, or queue capacity");
    }
    // The xor64 layout keeps a 4-row block's 16-byte rows in distinct banks,
    // which holds for a 4x4x4 or a 4x4x8 core (rfc11 section 0: 4x4x8).
    if (p.mma_xor_layout &&
        (p.mma_m != 64 || p.mma_n != 64 || p.mma_k != 8 || p.mma_core_m != 4 || p.mma_core_n != 4 ||
         (p.mma_core_k != 4 && p.mma_core_k != 8))) {
        throw ProfileError("device profile: xor64 MMA layout only supports 64x64x8 with a 4x4x4 or 4x4x8 core");
    }
    if (p.tma_sector_credits >= p.gmem_queue_sectors) {
        throw ProfileError("device profile: tma sector credits must be smaller than gmem queue");
    }
    // LSU. A lane-wide register (lanes x 4 B) must be whole sectors, a sector
    // whole words, and a bus beat whole words, so every plucked word has one
    // lane and never straddles a beat.
    if (p.gmem_sector_bytes > 64 || (p.lanes * 4) % p.gmem_sector_bytes != 0 ||
        p.gmem_bus_bytes_per_dram_tick % 4 != 0 || p.lanes > 64) {
        throw ProfileError(
            "device profile: lsu needs sectors <= 64 B dividing a lane-wide register and a "
            "bus beat of whole words");
    }
    if (p.lsu_active_commands == 0 || p.lsu_staging_vectors == 0 ||
        p.lsu_pluck_words_per_cycle == 0 || p.lsu_sectors_in_flight == 0) {
        throw ProfileError(
            "device profile: lsu active_commands, staging_vectors, pluck_words_per_cycle and "
            "sectors_in_flight must be nonzero");
    }
    if (p.lsu_read_streams < 2 || p.lsu_read_streams > 64) {
        throw ProfileError(
            "device profile: lsu read_streams must be in 2..64 (stream 0 TMA, 1..n-1 vector)");
    }
    if (p.lsu_ingress_sectors < 2 || p.lsu_egress_sectors < 2) {
        throw ProfileError("device profile: lsu ingress and egress FIFOs must hold at least two sectors");
    }
    if (p.lsu_ingress_latency_cycles == 0 || p.lsu_egress_latency_cycles == 0 ||
        p.lsu_ingress_latency_cycles > 64 || p.lsu_egress_latency_cycles > 64) {
        throw ProfileError(
            "device profile: lsu ingress/egress FIFO latency must be in 1..64 cycles (a BRAM FIFO "
            "is never fall-through)");
    }
}

DeviceProfile parse_device_profile(std::string_view source) {
    json document;
    try {
        document = json::parse(source);
    } catch (const json::exception& error) {
        throw ProfileError(std::string("cannot parse device profile: ") + error.what());
    }

    DeviceProfile p;
    p.name = field<std::string>(document, "name");
    p.lanes = field<uint32_t>(document, "lanes");
    const auto& rf = document.at("rf");
    p.rf_banks = field<uint32_t>(rf, "banks");
    p.rf_rows_per_bank = field<uint32_t>(rf, "rows_per_bank");
    p.rf_read_ports_per_bank = field<uint32_t>(rf, "read_ports_per_bank");
    p.rf_write_ports_per_bank = field<uint32_t>(rf, "write_ports_per_bank");
    // Removed (trash/rfc14.md section 3): every instruction reads its operands
    // in one round, so there is no read window to size. A profile that still
    // names one is refused rather than silently read differently.
    if (rf.contains("arithmetic_read_cycles")) {
        throw ProfileError("device profile: rf.arithmetic_read_cycles was removed (every instruction reads "
            "its operands in one round); delete the field");
    }
    // Optional (absent: the canonical 2); present: an integer in 1..8.
    const auto optional_rf = [&](const char* name, uint32_t& out) {
        if (!rf.contains(name)) return;
        const auto& value = rf.at(name);
        if (!value.is_number_unsigned() || value < 1 || value > 8) {
            throw ProfileError(std::string("device profile: rf ") + name + " must be an integer in 1..8");
        }
        out = value.get<uint32_t>();
    };
    optional_rf("read_latency_cycles", p.rf_read_latency_cycles);
    optional_rf("write_depth_cycles", p.rf_write_depth_cycles);
    const auto& shmem = document.at("shmem");
    p.shmem_bytes = field<uint32_t>(shmem, "bytes");
    p.shmem_banks = field<uint32_t>(shmem, "banks");
    // Optional (absent: the struct's canonical default); present: an integer
    // in [low, high]. A 0 is rejected here and again by validation.
    const auto optional_shmem = [&](const char* name, uint32_t low, uint32_t high, uint32_t& out) {
        if (!shmem.contains(name)) return;
        const auto& value = shmem.at(name);
        if (!value.is_number_unsigned() || value < low || value > high) {
            throw ProfileError(std::string("device profile: shmem ") + name + " must be an integer in " +
                std::to_string(low) + ".." + std::to_string(high));
        }
        out = value.get<uint32_t>();
    };
    optional_shmem("load_slots", 1, 1024, p.shmem_load_slots);
    optional_shmem("store_slots", 1, 1024, p.shmem_store_slots);
    optional_shmem("port_a_depth", 1, 64, p.shmem_port_a_depth);
    optional_shmem("read_latency_cycles", 1, 64, p.shmem_read_latency_cycles);
    optional_shmem("rmw_compute_cycles", 1, 64, p.shmem_rmw_compute_cycles);
    const auto& tmem = document.at("tmem");
    p.tmem_banks = field<uint32_t>(tmem, "banks");
    p.tmem_words_per_bank = field<uint32_t>(tmem, "words_per_bank");
    p.tmem_fine_banks = field<uint32_t>(tmem, "fine_banks");
    p.tmem_max_vector_width = field<uint32_t>(tmem, "max_vector_width");
    p.tmem_simt_read_write_duplex = field<bool>(tmem, "simt_read_write_duplex");
    const auto& mma = document.at("mma");
    p.mma_m = field<uint32_t>(mma, "m");
    p.mma_n = field<uint32_t>(mma, "n");
    p.mma_k = field<uint32_t>(mma, "k");
    p.mma_core_m = field<uint32_t>(mma, "core_m");
    p.mma_core_n = field<uint32_t>(mma, "core_n");
    p.mma_core_k = field<uint32_t>(mma, "core_k");
    p.mma_queue_commands = field<uint32_t>(mma, "queue_commands");
    p.mma_result_latency_cycles = field<uint32_t>(mma, "result_latency_cycles");
    const auto layout = field<std::string>(mma, "layout");
    if (field<std::string>(mma, "numeric_model") != mma_numeric_model)
        throw ProfileError("device profile: unsupported MMA numeric_model (expected fp16_raw_window25_rz_v1)");
    if (layout != "dense" && layout != "xor64")
        throw ProfileError("device profile: MMA layout must be 'dense' or 'xor64'");
    p.mma_xor_layout = layout == "xor64";
    const auto& sync = document.at("sync");
    p.mbar_wait_slots = field<uint32_t>(sync, "mbar_wait_slots");
    p.mbar_try_wait_timeout_cycles = field<uint32_t>(sync, "mbar_try_wait_timeout_cycles");
    const auto& tma = document.at("tma");
    p.tma_queue_commands = field<uint32_t>(tma, "queue_commands");
    p.tma_sector_credits = field<uint32_t>(tma, "sector_credits");
    if (tma.contains("store_sector_registers")) { // optional (absent: 1); present: an integer in 1..64
        const auto& value = tma.at("store_sector_registers");
        if (!value.is_number_unsigned() || value < 1 || value > 64) {
            throw ProfileError("device profile: tma store_sector_registers must be an integer in 1..64");
        }
        p.tma_store_sector_registers = value.get<uint32_t>();
    }
    const auto& core = document.at("core");
    p.core_clock_hz = field<uint64_t>(core, "clock_hz");
    p.timing.alu = field<uint8_t>(core, "alu_latency_cycles");
    p.timing.fadd = field<uint8_t>(core, "fadd_latency_cycles");
    // Optional (absent: the canonical 16 warps, 2 packets); present: an integer
    // in [low, high].
    const auto optional_core = [&](const char* name, uint32_t low, uint32_t high, uint32_t& out) {
        if (!core.contains(name)) return;
        const auto& value = core.at(name);
        if (!value.is_number_unsigned() || value < low || value > high) {
            throw ProfileError(std::string("device profile: core ") + name + " must be an integer in " +
                std::to_string(low) + ".." + std::to_string(high));
        }
        out = value.get<uint32_t>();
    };
    optional_core("warps", 1, 64, p.core_warps);
    optional_core("instruction_buffer_packets", 1, 16, p.core_instruction_buffer_packets);
    const auto& sfu = document.at("sfu");
    const auto checked_sfu = [&](const char* name, uint32_t low, uint32_t high) {
        const auto& value = sfu.at(name);
        if (!value.is_number_integer() || value < low || value > high) {
            throw ProfileError(std::string("device profile: sfu ") + name +
                " must be an integer in " + std::to_string(low) + ".." + std::to_string(high));
        }
        return value.get<uint32_t>();
    };
    p.sfu_lanes = checked_sfu("lanes", 1, p.lanes);
    p.sfu_latency_cycles = checked_sfu("latency_cycles", 1, 24);
    const auto sfu_occupancy = (p.lanes + p.sfu_lanes - 1) / p.sfu_lanes;
    p.timing.sfu = static_cast<uint8_t>(sfu_occupancy - 1 + p.sfu_latency_cycles);
    const auto& gmem = document.at("gmem");
    p.gmem_bytes = field<uint64_t>(gmem, "bytes");
    p.dram_clock_hz = field<uint64_t>(gmem, "dram_clock_hz");
    p.gmem_sector_bytes = field<uint32_t>(gmem, "sector_bytes");
    p.gmem_bus_bytes_per_dram_tick = field<uint32_t>(gmem, "bus_bytes_per_dram_tick");
    p.gmem_latency_dram_ticks = field<uint32_t>(gmem, "latency_dram_ticks");
    p.gmem_queue_sectors = field<uint32_t>(gmem, "queue_sectors");
    if (!document.contains("lsu")) throw ProfileError("device profile: missing 'lsu' section");
    const auto& lsu = document.at("lsu");
    p.lsu_active_commands = field<uint32_t>(lsu, "active_commands");
    p.lsu_staging_vectors = field<uint32_t>(lsu, "staging_vectors");
    p.lsu_read_streams = field<uint32_t>(lsu, "read_streams");
    p.lsu_pluck_words_per_cycle = field<uint32_t>(lsu, "pluck_words_per_cycle");
    p.lsu_ingress_sectors = field<uint32_t>(lsu, "ingress_sectors");
    p.lsu_egress_sectors = field<uint32_t>(lsu, "egress_sectors");
    p.lsu_sectors_in_flight = field<uint32_t>(lsu, "sectors_in_flight");
    p.lsu_ingress_latency_cycles = field<uint32_t>(lsu, "ingress_latency_cycles");
    p.lsu_egress_latency_cycles = field<uint32_t>(lsu, "egress_latency_cycles");
    validate_device_profile(p);
    return p;
}

DeviceProfile load_device_profile(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw ProfileError("cannot open device profile '" + path + "'");
    std::ostringstream contents;
    contents << input.rdbuf();
    return parse_device_profile(contents.str());
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

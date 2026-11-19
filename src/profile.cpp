#include "profile.hpp"
#include "hash.hpp"
#include "rf.hpp"
#include "mma_numeric.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <bit>
#include <fstream>
#include <iostream>
#include <mutex>
#include <set>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>

namespace nanosm {
namespace {

using json = nlohmann::json;

std::vector<std::string> mma_shape_names(const std::vector<MmaShape>& shapes) {
    std::vector<std::string> names;
    for (const auto& shape : shapes) names.push_back(mma_shape_name(shape));
    return names;
}

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
        {"urf", {
            {"registers", p.urf_registers},
            {"read_latency_cycles", p.urf_read_latency_cycles},
            {"write_depth_cycles", p.urf_write_depth_cycles},
            {"alu_latency_cycles", p.urf_alu_latency_cycles},
            {"issue_to_port_cycles", p.urf_issue_to_port_cycles},
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
            {"fine_banks", p.tmem_fine_banks},
            {"words_per_fine_bank", p.tmem_words_per_fine_bank},
            {"max_vector_width", p.tmem_max_vector_width},
            {"port_depth", p.tmem_port_depth},
            {"read_latency_cycles", p.tmem_read_latency_cycles},
        }},
        {"mma", {
            {"shapes", mma_shape_names(p.mma_shapes)},
            {"core_m", p.mma_core_m}, {"core_n", p.mma_core_n}, {"core_k", p.mma_core_k},
            {"queue_commands", p.mma_queue_commands},
            {"result_latency_cycles", p.mma_result_latency_cycles},
            {"layout", p.mma_layout == TileLayout::swizzle64 ? "swizzle64" : "dense"},
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
    if (p.mma_layout != TileLayout::dense && p.mma_layout != TileLayout::swizzle64) {
        throw ProfileError("device profile: unsupported MMA layout");
    }
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
    // The uniform register file (src/rf.hpp). Its depths are nonzero register
    // stages; a uniform operand must reach its unit no later than the vector
    // operands read in the same round (dispatch is issue + 1 + Lr); every
    // writer presents at issue + L, so L covers the longest writer (r2ur
    // through the vector RF, or the uniform ALU through the URF) and a
    // dependent uniform reader, d = L + Wu cycles later, fits the stall field.
    if (p.urf_registers == 0 || p.urf_registers > 64) {
        throw ProfileError("device profile: urf registers must be in 1..64");
    }
    if (p.urf_read_latency_cycles == 0 || p.urf_read_latency_cycles > 8 ||
        p.urf_write_depth_cycles == 0 || p.urf_write_depth_cycles > 8 ||
        p.urf_alu_latency_cycles == 0 || p.urf_alu_latency_cycles > 8) {
        throw ProfileError("device profile: urf read_latency_cycles, write_depth_cycles and alu_latency_cycles "
            "must be in 1..8 register stages");
    }
    if (p.urf_read_latency_cycles > p.rf_read_latency_cycles) {
        throw ProfileError("device profile: urf read_latency_cycles (" + std::to_string(p.urf_read_latency_cycles) +
            ") may not exceed rf read_latency_cycles (" + std::to_string(p.rf_read_latency_cycles) +
            "): an instruction's uniform operands must reach its unit with its vector operands");
    }
    const uint32_t longest_writer = urf_longest_writer_path(p);
    if (p.urf_issue_to_port_cycles < longest_writer) {
        throw ProfileError("device profile: urf issue_to_port_cycles (" + std::to_string(p.urf_issue_to_port_cycles) +
            ") must cover the longest uniform-register writer, " + std::to_string(longest_writer) +
            " (the issue register + the larger of urf read_latency_cycles + alu_latency_cycles and rf "
            "read_latency_cycles)");
    }
    if (p.urf_issue_to_port_cycles + p.urf_write_depth_cycles > encoding::stall_mask) {
        throw ProfileError("device profile: a dependent uniform-register read (urf issue_to_port_cycles + "
            "write_depth_cycles) must fit the five-bit stall field");
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
        p.shmem_bytes == 0 || p.shmem_banks == 0 ||
        p.tmem_words_per_fine_bank == 0 || p.tmem_fine_banks != 16 ||
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
    if (p.mma_core_m == 0 || p.mma_core_n == 0 || p.mma_core_k == 0 || p.mma_queue_commands == 0 ||
        p.mma_result_latency_cycles == 0) {
        throw ProfileError("device profile: invalid MMA core, timing, or queue capacity");
    }
    // The MMA shapes (trash/chan.txt 2026-10-22, HUMAN: "m,n,k need to be a
    // multiple of core m,n,k and if mma k is more than core k, mma m/n needs
    // to be large enough to hide the accumulator latency"). Each listed shape
    // takes one opcode pair of the mma.async block, in list order.
    if (p.mma_shapes.empty()) throw ProfileError("device profile: mma shapes must list at least one shape");
    if (p.mma_shapes.size() > mma_block_shapes) {
        throw ProfileError("device profile: mma shapes lists " + std::to_string(p.mma_shapes.size()) +
            " shapes; the mma.async opcode block holds " + std::to_string(mma_block_shapes));
    }
    for (std::size_t i = 0; i < p.mma_shapes.size(); ++i) {
        const auto& shape = p.mma_shapes[i];
        const auto name = "device profile: mma shape " + mma_shape_name(shape);
        if (std::find(p.mma_shapes.begin(), p.mma_shapes.begin() + static_cast<std::ptrdiff_t>(i), shape) !=
            p.mma_shapes.begin() + static_cast<std::ptrdiff_t>(i)) {
            throw ProfileError(name + " is listed twice");
        }
        // Every dimension is a power of two (HUMAN 2026-10-22: "K also needs to
        // be a power of 2"; "m and n can get that rule too ... it needs to
        // somehow line up with lanes and shmem width").
        for (const auto& [letter, size] :
             std::initializer_list<std::pair<char, uint32_t>>{{'m', shape.m}, {'n', shape.n}, {'k', shape.k}}) {
            if (!std::has_single_bit(size)) {
                throw ProfileError(name + ": m, n and k must be powers of two (" + letter + " is " +
                    std::to_string(size) + ")");
            }
        }
        // The walker covers m in core_m blocks, n in core_n blocks and k in
        // core_k steps, each step one fused group of core_k products plus C.
        if (shape.m % p.mma_core_m != 0 || shape.n % p.mma_core_n != 0 || shape.k % p.mma_core_k != 0) {
            throw ProfileError(name + ": m, n and k must be multiples of the core's " +
                std::to_string(p.mma_core_m) + "x" + std::to_string(p.mma_core_n) + "x" +
                std::to_string(p.mma_core_k));
        }
        // The accumulator: one row per lane in each acc.ld / acc.st beat, so
        // an accumulator holds at least `lanes` rows (the .acc rule), and it
        // fits in tmem.
        if (shape.m < p.lanes) {
            throw ProfileError(name + ": its accumulator has " + std::to_string(shape.m) +
                " rows; acc.ld / acc.st move one row per lane, so an accumulator needs at least " +
                std::to_string(p.lanes));
        }
        if (static_cast<uint64_t>(shape.m) * shape.n >
            static_cast<uint64_t>(p.tmem_fine_banks) * p.tmem_words_per_fine_bank) {
            throw ProfileError(name + ": its accumulator (" + std::to_string(uint64_t{shape.m} * shape.n) +
                " words) does not fit in tmem");
        }
        // k > core_k: the command visits each accumulator block once per K
        // block, mma_blocks_per_k steps apart. A step's operand rounds start at
        // f and end at f + r - 1; it enters the core D + Lb later, its C is
        // sampled tmem.port_depth (Dt) after entry, and its result lands
        // result_latency (R) + Dt after entry (written at that cycle's edge;
        // READ_FIRST). The walker starts a step only when no earlier step of
        // the same block is still in the read pipeline or the core, i.e. from
        // that landing cycle on. Steps of one command start back to back, one
        // per operand round (at least one each), so the next visit starts at
        // f + r + (blocks - 1) or later: no stall iff
        //   blocks >= D + Lb + R + Dt   (mma_min_blocks_per_k).
        // The bound holds for any layout; under swizzle64 each A reload takes
        // one more round, which only adds slack.
        if (shape.k > p.mma_core_k && mma_blocks_per_k(p, shape) < mma_min_blocks_per_k(p)) {
            throw ProfileError(name + ": k " + std::to_string(shape.k) + " is more than the core's " +
                std::to_string(p.mma_core_k) + ", so each accumulator block is revisited every " +
                std::to_string(mma_blocks_per_k(p, shape)) + " steps (m / core_m x n / core_n); hiding the " +
                "accumulator latency needs at least " + std::to_string(mma_min_blocks_per_k(p)) +
                " (shmem port_a_depth + read_latency_cycles + mma result_latency_cycles + tmem port_depth)");
        }
    }
    // tmem's own depths: nonzero register stages.
    if (p.tmem_port_depth == 0 || p.tmem_port_depth > 64 ||
        p.tmem_read_latency_cycles == 0 || p.tmem_read_latency_cycles > 64) {
        throw ProfileError("device profile: tmem port_depth and read_latency_cycles must be in 1..64 "
            "register stages");
    }
    // A step's C is read from tmem as the step enters the core and is added in
    // the core's last stage, so the core's result latency must cover that read.
    if (p.mma_result_latency_cycles < p.tmem_port_depth + p.tmem_read_latency_cycles) {
        throw ProfileError("device profile: mma result_latency_cycles must cover the accumulator read "
            "(tmem port_depth + read_latency_cycles)");
    }
    // The swizzle64 layout puts four neighbouring rows' same 16-byte piece in
    // four different slots of 16 banks, which serves a core block of 4 rows
    // by 8 or 4 fp16 of k: a 4x4x4 or a 4x4x8 core (rfc11 section 0: 4x4x8).
    // It depends on the row only (any four neighbouring rows, wherever the
    // tile sits), so any m and n in whole core blocks read one bank round per
    // block (src/layout.hpp; src/sim.cpp mma_a_address). On a 4x4x4 core a
    // step that reloads A reads half-pieces of A and B together; whether they
    // share a round depends on the two tiles' row numbers mod 16
    // (trash/address_keyed_census.py).
    if (p.mma_layout == TileLayout::swizzle64 && (p.mma_core_m != 4 || p.mma_core_n != 4 ||
                             (p.mma_core_k != 4 && p.mma_core_k != 8))) {
        throw ProfileError("device profile: swizzle64 MMA layout needs a 4x4x4 or 4x4x8 core");
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
    // The uniform register file: an optional section, each field optional
    // (absent: the canonical values); present: an integer in [low, high].
    if (document.contains("urf")) {
        const auto& urf = document.at("urf");
        const auto optional_urf = [&](const char* name, uint32_t low, uint32_t high, uint32_t& out) {
            if (!urf.contains(name)) return;
            const auto& value = urf.at(name);
            if (!value.is_number_unsigned() || value < low || value > high) {
                throw ProfileError(std::string("device profile: urf ") + name + " must be an integer in " +
                    std::to_string(low) + ".." + std::to_string(high));
            }
            out = value.get<uint32_t>();
        };
        optional_urf("registers", 1, 64, p.urf_registers);
        optional_urf("read_latency_cycles", 1, 8, p.urf_read_latency_cycles);
        optional_urf("write_depth_cycles", 1, 8, p.urf_write_depth_cycles);
        optional_urf("alu_latency_cycles", 1, 8, p.urf_alu_latency_cycles);
        optional_urf("issue_to_port_cycles", 1, 30, p.urf_issue_to_port_cycles);
    }
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
    p.tmem_fine_banks = field<uint32_t>(tmem, "fine_banks");
    p.tmem_words_per_fine_bank = field<uint32_t>(tmem, "words_per_fine_bank");
    p.tmem_max_vector_width = field<uint32_t>(tmem, "max_vector_width");
    // Optional (absent: the canonical 2); present: an integer in 1..64.
    const auto optional_tmem = [&](const char* name, uint32_t& out) {
        if (!tmem.contains(name)) return;
        const auto& value = tmem.at(name);
        if (!value.is_number_unsigned() || value < 1 || value > 64) {
            throw ProfileError(std::string("device profile: tmem ") + name + " must be an integer in 1..64");
        }
        out = value.get<uint32_t>();
    };
    optional_tmem("port_depth", p.tmem_port_depth);
    optional_tmem("read_latency_cycles", p.tmem_read_latency_cycles);
    const auto& mma = document.at("mma");
    if (!mma.contains("shapes") || !mma.at("shapes").is_array()) {
        throw ProfileError("device profile: mma shapes must be an array of shape names");
    }
    for (const auto& name : mma.at("shapes")) {
        const auto shape = name.is_string() ? parse_mma_shape_name(name.get<std::string>()) : std::nullopt;
        if (!shape.has_value()) {
            throw ProfileError("device profile: mma shape " + name.dump() +
                " is not written m<rows>n<columns>k<depth> (e.g. \"m64n64k8\")");
        }
        p.mma_shapes.push_back(*shape);
    }
    p.mma_core_m = field<uint32_t>(mma, "core_m");
    p.mma_core_n = field<uint32_t>(mma, "core_n");
    p.mma_core_k = field<uint32_t>(mma, "core_k");
    p.mma_queue_commands = field<uint32_t>(mma, "queue_commands");
    p.mma_result_latency_cycles = field<uint32_t>(mma, "result_latency_cycles");
    const auto layout = field<std::string>(mma, "layout");
    if (field<std::string>(mma, "numeric_model") != mma_numeric_model)
        throw ProfileError("device profile: unsupported MMA numeric_model (expected fp16_raw_window25_rz_v1)");
    if (layout != "dense" && layout != "swizzle64")
        throw ProfileError("device profile: MMA layout must be 'dense' or 'swizzle64'");
    p.mma_layout = layout == "swizzle64" ? TileLayout::swizzle64 : TileLayout::dense;
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
    report_profile_warnings(p);
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

std::optional<MmaShape> mma_shape_of(const DeviceProfile& profile, Opcode opcode) {
    if (!is_mma_async(opcode)) return std::nullopt;
    const auto index = mma_shape_index(opcode);
    if (index >= profile.mma_shapes.size()) return std::nullopt;
    return profile.mma_shapes[index];
}

uint32_t mma_blocks_per_k(const DeviceProfile& profile, const MmaShape& shape) {
    return (shape.m / profile.mma_core_m) * (shape.n / profile.mma_core_n);
}

uint32_t mma_min_blocks_per_k(const DeviceProfile& profile) {
    return profile.shmem_port_a_depth + profile.shmem_read_latency_cycles +
        profile.mma_result_latency_cycles + profile.tmem_port_depth;
}

std::vector<std::string> device_profile_warnings(const DeviceProfile& profile) {
    std::vector<std::string> warnings;
    // A swizzle64 operand row is read in 16-byte pieces of 8 fp16, so a shape
    // whose k is not a multiple of 8 reads only part of its last piece in each
    // row (HUMAN 2026-10-22: "it should warn you if there is a partially
    // occupied storage block"). The row spacing still counts whole pieces.
    if (profile.mma_layout == TileLayout::swizzle64) {
        for (const auto& shape : profile.mma_shapes) {
            const uint32_t used = shape.k % 8;
            if (used == 0) continue;
            const uint32_t blocks = shape.k / 8 + 1;
            const uint32_t unused_percent = 100 * (8 - used) / (8 * blocks);
            warnings.push_back("device profile: mma shape " + mma_shape_name(shape) + " reads only " +
                std::to_string(used) + " of the 8 fp16 in the last 16-byte piece of each swizzle64 operand row; " +
                std::to_string(unused_percent) + "% of the pieces it reads is unused");
        }
    }
    return warnings;
}

void report_profile_warnings(const DeviceProfile& profile) {
    // Once per distinct warning per process: tests and tools load profiles
    // many times.
    static std::mutex mutex;
    static std::set<std::string> reported;
    for (const auto& warning : device_profile_warnings(profile)) {
        const std::lock_guard lock(mutex);
        if (reported.insert(warning).second) std::cerr << "warning: " << warning << '\n';
    }
}

} // namespace nanosm

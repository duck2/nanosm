#include "trace.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace nanosm {
namespace {

using json = nlohmann::json;

constexpr uint32_t scheduler_track = 1;
constexpr uint32_t warp_track_base = 100;
constexpr uint32_t shmem_a_read_track = 1000;
constexpr uint32_t gmem_bus_track = 1001;
constexpr uint32_t gmem_command_track = 1002;
constexpr uint32_t gmem_queue_track = 1003;
constexpr uint32_t tma_command_track = 1010;
constexpr uint32_t tma_command_depth_track = 1011;
constexpr uint32_t tma_credit_depth_track = 1012;
constexpr uint32_t tma_return_depth_track = 1013;
constexpr uint32_t shmem_a_write_track = 1004;
constexpr uint32_t shmem_b_write_track = 1005;
constexpr uint32_t mbar_rmw_track = 1006;
constexpr uint32_t acc_ld_engine_track = 1020;
constexpr uint32_t acc_st_engine_track = 1021;

uint32_t shmem_track(ShmemService service) {
    switch (service) {
    case ShmemService::a_read: return shmem_a_read_track;
    case ShmemService::a_write: return shmem_a_write_track;
    case ShmemService::b_write: return shmem_b_write_track;
    }
    return shmem_a_read_track;
}

std::string category_name(TraceCategory category) {
    switch (category) {
    case TraceCategory::instruction: return "instruction";
    case TraceCategory::warp: return "warp";
    case TraceCategory::shmem: return "shmem";
    case TraceCategory::gmem: return "gmem";
    case TraceCategory::sync: return "sync";
    case TraceCategory::tma: return "tma";
    case TraceCategory::tmem: return "tmem";
    case TraceCategory::mma: return "mma";
    }
    return "unknown";
}

double percent(uint64_t part, uint64_t whole) {
    return whole == 0 ? 0.0 : 100.0 * static_cast<double>(part) / static_cast<double>(whole);
}

} // namespace

std::string warp_state_name(WarpState state) {
    switch (state) {
    case WarpState::not_examined: return "not examined (another warp issued)";
    case WarpState::issued: return "issued";
    case WarpState::stall_count: return "stall count";
    case WarpState::depbar: return "depbar";
    case WarpState::mbar_wait: return "mbar wait";
    case WarpState::mbar_wait_slot: return "mbar wait slot";
    case WarpState::structural: return "structural";
    case WarpState::shmem_slot: return "no free slot";
    case WarpState::shmem_order: return "ordered behind own access";
    case WarpState::no_instruction: return "waiting for instruction";
    case WarpState::halted: return "halted";
    case WarpState::count: break;
    }
    return "unknown";
}

TraceRecorder::TraceRecorder(
    uint32_t active_warps,
    uint64_t core_clock_hz,
    uint64_t dram_clock_hz,
    TraceOptions options)
    : active_warps_(active_warps),
      core_clock_hz_(core_clock_hz),
      dram_clock_hz_(dram_clock_hz),
      options_(options),
      warp_intervals_(active_warps) {
    if (options_.timeline) events_.reserve(std::min<std::size_t>(options_.max_events, 4096));
}

void TraceRecorder::add(TimelineEvent event) {
    if (!options_.timeline) return;
    if (events_.size() == options_.max_events) {
        ++summary_.timeline_events_dropped;
        return;
    }
    if (events_.size() > options_.max_events) {
        ++summary_.timeline_events_dropped;
        return;
    }
    events_.push_back(std::move(event));
}

void TraceRecorder::issue(uint64_t cycle, uint32_t warp, uint32_t pc, const Instruction& instruction) {
    ++summary_.instructions;
    ++summary_.scheduler_issued_cycles;
    add({
        TraceEventKind::instant,
        TraceCategory::instruction,
        opcode_name(instruction.opcode, &mma_shapes_),
        "warp=" + std::to_string(warp) + ", " + format_instruction(instruction, true, &mma_shapes_),
        scheduler_track,
        cycle,
        0,
        0,
        pc,
    });
}

void TraceRecorder::fetch(uint64_t cycle, uint32_t warp, uint32_t pc) {
    ++summary_.fetches;
    if (!options_.debug) return;
    add({
        TraceEventKind::instant,
        TraceCategory::warp,
        "fetch",
        "pc=" + std::to_string(pc),
        warp_track_base + warp,
        cycle,
        0,
        0,
        pc,
    });
}

void TraceRecorder::rf_write(uint64_t cycle, uint32_t warp, uint8_t reg, uint32_t bank) {
    if (!options_.debug) return;
    add({
        TraceEventKind::instant,
        TraceCategory::instruction,
        "RF write",
        "r" + std::to_string(reg) + ", bank=" + std::to_string(bank),
        warp_track_base + warp,
        cycle,
        0,
        static_cast<int64_t>(bank),
        reg,
    });
}

void TraceRecorder::urf_write(uint64_t cycle, uint32_t warp, uint8_t reg, uint32_t value) {
    ++summary_.urf_writes;
    add({
        TraceEventKind::instant,
        TraceCategory::instruction,
        "UR write",
        "ur" + std::to_string(reg) + " = " + std::to_string(value),
        warp_track_base + warp,
        cycle,
        0,
        static_cast<int64_t>(value),
        reg,
    });
}

void TraceRecorder::rf_read(uint64_t cycle, uint32_t warp, uint8_t reg, uint32_t bank) {
    if (!options_.debug) return;
    add({
        TraceEventKind::instant,
        TraceCategory::instruction,
        "RF read",
        "r" + std::to_string(reg) + ", bank=" + std::to_string(bank),
        warp_track_base + warp,
        cycle,
        0,
        static_cast<int64_t>(bank),
        reg,
    });
}

void TraceRecorder::scheduler_idle(bool draining) {
    if (draining) ++summary_.scheduler_draining_cycles;
    else ++summary_.scheduler_idle_cycles;
}

void TraceRecorder::close_warp_interval(uint32_t warp, uint64_t end_cycle) {
    auto& interval = warp_intervals_.at(warp);
    if (!interval.active || end_cycle <= interval.start_cycle) return;
    add({
        TraceEventKind::slice,
        TraceCategory::warp,
        warp_state_name(interval.state),
        interval.detail,
        warp_track_base + warp,
        interval.start_cycle,
        end_cycle - interval.start_cycle,
    });
}

void TraceRecorder::warp_state(uint64_t cycle, uint32_t warp, WarpState state, std::string detail) {
    ++summary_.warp_cycles.at(static_cast<std::size_t>(state));
    if (state == WarpState::mbar_wait_slot) ++summary_.mbar_slot_full_stalls;
    auto& interval = warp_intervals_.at(warp);
    if (interval.active && interval.state == state && interval.detail == detail) return;
    close_warp_interval(warp, cycle);
    interval.active = true;
    interval.state = state;
    interval.detail = std::move(detail);
    interval.start_cycle = cycle;
}

void TraceRecorder::depbar(uint64_t cycle, uint32_t warp, uint8_t bar, uint32_t value) {
    add({
        TraceEventKind::instant,
        TraceCategory::sync,
        "depbar sb" + std::to_string(bar),
        "value=" + std::to_string(value),
        warp_track_base + warp,
        cycle,
    });
}

void TraceRecorder::tmem_beat(uint64_t cycle, uint32_t warp, bool load, uint32_t address) {
    if (load) ++summary_.tmem_load_beats;
    else ++summary_.tmem_store_beats;
    add({TraceEventKind::instant, TraceCategory::tmem, load ? "acc.ld beat" : "acc.st beat",
        "warp=" + std::to_string(warp) + ", address=" + std::to_string(address),
        load ? acc_ld_engine_track : acc_st_engine_track, cycle, 0, static_cast<int64_t>(address)});
}

void TraceRecorder::mma_command(uint64_t cycle, uint64_t command, uint32_t warp, bool init) {
    ++summary_.mma_commands;
    add({TraceEventKind::instant, TraceCategory::mma, init ? "mma.async.init" : "mma.async",
        "warp=" + std::to_string(warp), 1040, cycle, 0, 0, command});
}

void TraceRecorder::mma_input(uint64_t cycle, uint64_t command, uint32_t rounds, uint32_t step, bool reads_a) {
    ++summary_.mma_input_steps;
    summary_.mma_shmem_rounds += rounds;
    if (reads_a) ++summary_.mma_a_block_reads;
    add({TraceEventKind::instant, TraceCategory::mma, "MMA input step",
        "rounds=" + std::to_string(rounds) + (reads_a ? ", reads A" : ""), 1040, cycle, 0, step, command});
}

void TraceRecorder::mma_result(uint64_t cycle, uint64_t command, uint32_t address) {
    ++summary_.mma_result_steps;
    add({TraceEventKind::instant, TraceCategory::mma, "MMA tmem write",
        "address=" + std::to_string(address), 1041, cycle, 0, address, command});
}

void TraceRecorder::mma_recurrence_stall() { ++summary_.mma_recurrence_stalls; }

void TraceRecorder::sfu_accept(uint32_t occupancy_cycles) {
    ++summary_.sfu_ops;
    summary_.sfu_busy_cycles += occupancy_cycles;
}

void TraceRecorder::sfu_unit_stall() { ++summary_.sfu_unit_stalls; }

void TraceRecorder::shmem_begin(ShmemService service, uint64_t cycle, std::string operation,
    std::string detail, TraceCategory category, uint64_t id) {
    ++summary_.shmem_requests;
    shmem_slices_[static_cast<std::size_t>(service)] = OpenSlice{
        cycle, shmem_track(service), category, std::move(operation), std::move(detail), id};
}

void TraceRecorder::shmem_grant(uint64_t cycle, ShmemService service, ShmemClient client, uint32_t slot) {
    static const char* const service_names[] = {"A-read", "A-write", "B-write"};
    static const char* const client_names[] = {"simt", "probe", "rmw", "tma"};
    if (client == ShmemClient::simt || client == ShmemClient::tma) ++summary_.shmem_rounds;
    switch (service) {
    case ShmemService::a_read:
        ++(client == ShmemClient::simt ? summary_.shmem_a_read_simt
            : client == ShmemClient::probe ? summary_.shmem_a_read_probe
            : client == ShmemClient::rmw ? summary_.shmem_a_read_rmw : summary_.shmem_a_read_tma);
        break;
    case ShmemService::a_write:
        ++(client == ShmemClient::rmw ? summary_.shmem_a_write_rmw : summary_.shmem_a_write_simt);
        break;
    case ShmemService::b_write:
        ++summary_.shmem_b_write_tma;
        break;
    }
    if (!options_.debug) return;
    add({TraceEventKind::instant, TraceCategory::shmem,
        std::string(service_names[static_cast<int>(service)]) + " grant",
        client_names[static_cast<int>(client)], shmem_track(service), cycle, 0, slot});
}

void TraceRecorder::close_slice(OpenSlice slice, uint64_t end_cycle) {
    add({
        TraceEventKind::slice,
        slice.category,
        std::move(slice.name),
        std::move(slice.detail),
        slice.track,
        slice.start_cycle,
        std::max<uint64_t>(1, end_cycle - slice.start_cycle),
        0,
        slice.id,
        slice.async,
    });
}

void TraceRecorder::shmem_end(ShmemService service, uint64_t end_cycle) {
    auto& slice = shmem_slices_[static_cast<std::size_t>(service)];
    if (!slice.has_value()) throw std::runtime_error("trace shmem end without begin");
    close_slice(std::move(*slice), end_cycle);
    slice.reset();
}

void TraceRecorder::mbar_rmw_begin(
    uint64_t cycle, uint32_t byte_address, std::string operation) {
    ++summary_.mbar_rmw_requests;
    mbar_rmw_slice_ = OpenSlice{
        cycle,
        mbar_rmw_track,
        TraceCategory::sync,
        "mbar RMW",
        std::move(operation) + ", address=" + std::to_string(byte_address),
    };
}

void TraceRecorder::mbar_rmw_end(uint64_t end_cycle, bool phase_changed) {
    if (!mbar_rmw_slice_.has_value()) throw std::runtime_error("trace mbar RMW end without begin");
    if (phase_changed) ++summary_.mbar_phase_changes;
    close_slice(std::move(*mbar_rmw_slice_), end_cycle);
    mbar_rmw_slice_.reset();
}

// The engine runs one operation at a time, so an atomic's slice shares the
// mbar RMW's track and open-slice slot.
void TraceRecorder::atomic_rmw_begin(uint64_t cycle, std::string detail) {
    mbar_rmw_slice_ = OpenSlice{cycle, mbar_rmw_track, TraceCategory::sync, "atomic RMW", std::move(detail)};
}

void TraceRecorder::atomic_rmw_end(uint64_t end_cycle) {
    if (!mbar_rmw_slice_.has_value()) throw std::runtime_error("trace atomic RMW end without begin");
    close_slice(std::move(*mbar_rmw_slice_), end_cycle);
    mbar_rmw_slice_.reset();
}

void TraceRecorder::mbar_probe(
    uint64_t cycle, uint32_t warp, uint32_t byte_address, uint8_t parity, bool complete) {
    ++summary_.mbar_initial_probes;
    if (complete) ++summary_.mbar_immediate_successes;
    add({
        TraceEventKind::instant,
        TraceCategory::sync,
        "mbar try-wait probe",
        "address=" + std::to_string(byte_address) + ", parity=" + std::to_string(parity) +
            ", complete=" + (complete ? "true" : "false"),
        warp_track_base + warp,
        cycle,
    });
}

void TraceRecorder::mbar_sleep(
    uint64_t cycle, uint32_t warp, uint32_t byte_address, uint8_t parity) {
    ++summary_.mbar_sleeps;
    add({
        TraceEventKind::instant,
        TraceCategory::sync,
        "mbar sleep",
        "address=" + std::to_string(byte_address) + ", parity=" + std::to_string(parity),
        warp_track_base + warp,
        cycle,
    });
}

void TraceRecorder::mbar_wake(
    uint64_t cycle, uint32_t warp, uint32_t byte_address, uint8_t phase) {
    ++summary_.mbar_wakes;
    add({
        TraceEventKind::instant,
        TraceCategory::sync,
        "mbar wake",
        "address=" + std::to_string(byte_address) + ", phase=" + std::to_string(phase),
        warp_track_base + warp,
        cycle,
    });
}

void TraceRecorder::mbar_timeout(
    uint64_t cycle, uint32_t warp, uint32_t byte_address, uint8_t parity) {
    ++summary_.mbar_timeouts;
    add({
        TraceEventKind::instant,
        TraceCategory::sync,
        "mbar try-wait timeout",
        "address=" + std::to_string(byte_address) + ", parity=" + std::to_string(parity),
        warp_track_base + warp,
        cycle,
    });
}

void TraceRecorder::lsu_command_begin(
    uint64_t cycle, uint64_t command, uint32_t warp, std::string name) {
    ++summary_.gmem_commands;
    gmem_commands_.emplace(command, OpenSlice{
        cycle,
        gmem_command_track,
        TraceCategory::gmem,
        std::move(name),
        "warp=" + std::to_string(warp),
        command,
        true, // commands from different warps overlap in flight
    });
}

void TraceRecorder::lsu_command_end(
    uint64_t end_cycle, uint64_t command, const std::string& phases) {
    const auto found = gmem_commands_.find(command);
    if (found == gmem_commands_.end()) throw std::runtime_error("trace lsu command end without begin");
    summary_.gmem_command_latency_cycles += end_cycle - found->second.start_cycle;
    found->second.detail += ", " + phases;
    close_slice(std::move(found->second), end_cycle);
    gmem_commands_.erase(found);
}

void TraceRecorder::gmem_sector_enqueue() {
    ++summary_.gmem_sectors;
}

void TraceRecorder::gmem_bus_begin(
    uint64_t cycle, uint64_t request, uint64_t sector_address, bool load, std::string owner) {
    gmem_bus_requests_.emplace(request, OpenSlice{
        cycle,
        gmem_bus_track,
        TraceCategory::gmem,
        load ? "read sector" : "write sector",
        "address=" + std::to_string(sector_address) + ", owner=" + std::move(owner),
        request,
    });
}

void TraceRecorder::tma_command_begin(
    uint64_t cycle, uint64_t command, uint32_t warp, uint32_t descriptor_address,
    std::size_t command_depth) {
    ++summary_.tma_commands;
    tma_commands_.emplace(command, OpenSlice{
        cycle,
        tma_command_track,
        TraceCategory::tma,
        "tma command",
        "warp=" + std::to_string(warp) + ", descriptor=" +
            std::to_string(descriptor_address),
        command,
        true,
    });
    tma_depths(cycle, command_depth, last_tma_credit_depth_.value_or(0),
        last_tma_return_depth_.value_or(0));
}

void TraceRecorder::tma_command_end(
    uint64_t end_cycle, uint64_t command, uint64_t logical_bytes, uint64_t useful_bytes) {
    const auto found = tma_commands_.find(command);
    if (found == tma_commands_.end()) {
        throw std::runtime_error("trace tma command end without begin");
    }
    ++summary_.tma_command_completions;
    summary_.tma_logical_bytes += logical_bytes;
    summary_.tma_useful_bytes += useful_bytes;
    close_slice(std::move(found->second), end_cycle);
    tma_commands_.erase(found);
}

void TraceRecorder::tma_sector(uint32_t sector_bytes) {
    summary_.tma_sector_bytes += sector_bytes;
}

void TraceRecorder::tma_data_port(uint32_t rounds) {
    ++summary_.tma_port_requests;
    summary_.tma_port_rounds += rounds;
}

void TraceRecorder::tma_depths(
    uint64_t cycle, std::size_t command_depth, std::size_t credit_depth,
    std::size_t return_depth) {
    summary_.tma_max_command_depth =
        std::max<uint64_t>(summary_.tma_max_command_depth, command_depth);
    summary_.tma_max_credit_depth =
        std::max<uint64_t>(summary_.tma_max_credit_depth, credit_depth);
    summary_.tma_max_return_depth =
        std::max<uint64_t>(summary_.tma_max_return_depth, return_depth);
    const auto counter = [&](std::optional<std::size_t>& previous, std::size_t value,
                             uint32_t track, const char* name) {
        if (previous == value) return;
        previous = value;
        add({TraceEventKind::counter, TraceCategory::tma, name, {}, track, cycle, 0,
            static_cast<int64_t>(value)});
    };
    counter(last_tma_command_depth_, command_depth, tma_command_depth_track, "command depth");
    counter(last_tma_credit_depth_, credit_depth, tma_credit_depth_track, "sector credits");
    counter(last_tma_return_depth_, return_depth, tma_return_depth_track, "return depth");
}

void TraceRecorder::gmem_bus_beat(uint64_t cycle, uint64_t dram_tick, uint64_t request) {
    ++summary_.gmem_bus_beats;
    if (!options_.debug) return;
    add({
        TraceEventKind::instant,
        TraceCategory::gmem,
        "bus beat",
        "dram_tick=" + std::to_string(dram_tick),
        gmem_bus_track,
        cycle,
        0,
        0,
        request,
    });
}

void TraceRecorder::gmem_bus_end(uint64_t end_cycle, uint64_t request, uint32_t sector_bytes) {
    const auto found = gmem_bus_requests_.find(request);
    if (found == gmem_bus_requests_.end()) throw std::runtime_error("trace gmem bus end without begin");
    summary_.gmem_bytes += sector_bytes;
    close_slice(std::move(found->second), end_cycle);
    gmem_bus_requests_.erase(found);
}

void TraceRecorder::gmem_queue_depth(uint64_t cycle, std::size_t depth) {
    summary_.gmem_max_queue_depth = std::max<uint64_t>(summary_.gmem_max_queue_depth, depth);
    if (last_gmem_queue_depth_ == depth) return;
    last_gmem_queue_depth_ = depth;
    add({
        TraceEventKind::counter,
        TraceCategory::gmem,
        "controller queue depth",
        {},
        gmem_queue_track,
        cycle,
        0,
        static_cast<int64_t>(depth),
    });
}

void TraceRecorder::finish(uint64_t cycle) {
    summary_.cycles = cycle;
    for (uint32_t warp = 0; warp < active_warps_; ++warp) close_warp_interval(warp, cycle);
}

std::string TraceRecorder::summary_text() const {
    const auto warp_total = summary_.cycles * active_warps_;
    std::ostringstream out;
    out << "cycles                         " << summary_.cycles << '\n';
    out << "instructions issued            " << summary_.instructions << '\n';
    out << "issue utilization              " << std::fixed << std::setprecision(1)
        << percent(summary_.scheduler_issued_cycles, summary_.cycles) << "%\n";
    out << "scheduler cycles:\n";
    out << "  issued                       " << summary_.scheduler_issued_cycles << '\n';
    out << "  no eligible warp             " << summary_.scheduler_idle_cycles << '\n';
    out << "  draining                     " << summary_.scheduler_draining_cycles << '\n';
    out << "warp cycles (total " << warp_total << "):\n";
    // At least one space after a state's name, however long ("not examined
    // (another warp issued)" is wider than the column).
    for (std::size_t i = 0; i < static_cast<std::size_t>(WarpState::count); ++i) {
        const auto state = static_cast<WarpState>(i);
        out << "  " << std::left << std::setw(26) << warp_state_name(state) << ' ' << std::right
            << summary_.warp_cycles[i] << '\n';
    }
    out << "shmem: requests=" << summary_.shmem_requests
        << " rounds=" << summary_.shmem_rounds
        << " load_slot_full=" << summary_.shmem_load_slot_waits
        << " store_slot_full=" << summary_.shmem_store_slot_waits
        << " ordered=" << summary_.shmem_order_waits
        << " store_paused=" << summary_.shmem_store_paused << '\n';
    out << "shmem grants: a_read_simt=" << summary_.shmem_a_read_simt
        << " a_read_probe=" << summary_.shmem_a_read_probe
        << " a_read_rmw=" << summary_.shmem_a_read_rmw
        << " a_read_tma=" << summary_.shmem_a_read_tma
        << " a_write_simt=" << summary_.shmem_a_write_simt
        << " a_write_rmw=" << summary_.shmem_a_write_rmw
        << " b_write_tma=" << summary_.shmem_b_write_tma << '\n';
    out << "rf: async_writes=" << summary_.rf_async_writes
        << " async_waits=" << summary_.rf_async_waits
        << " simt_inhibited=" << summary_.rf_simt_inhibited
        << " simt_read_inhibited=" << summary_.rf_simt_read_inhibited << '\n';
    out << "urf: writes=" << summary_.urf_writes << " read_ports_held=" << summary_.urf_read_ports_held << '\n';
    out << "mbar: rmw=" << summary_.mbar_rmw_requests
        << " phase_changes=" << summary_.mbar_phase_changes
        << " probes=" << summary_.mbar_initial_probes
        << " immediate=" << summary_.mbar_immediate_successes
        << " slot_full=" << summary_.mbar_slot_full_stalls
        << " sleeps=" << summary_.mbar_sleeps
        << " wakes=" << summary_.mbar_wakes
        << " timeouts=" << summary_.mbar_timeouts
        << " engine_waits_on_other=" << summary_.mbar_engine_waits_on_other << '\n';
    out << "gmem: commands=" << summary_.gmem_commands
        << " sectors=" << summary_.gmem_sectors
        << " beats=" << summary_.gmem_bus_beats
        << " bytes=" << summary_.gmem_bytes
        << " max_queue=" << summary_.gmem_max_queue_depth << '\n';
    // The achieved-vs-roofline number: beats are DRAM bus ticks, so occupancy is
    // beats over the run's total DRAM ticks.
    if (summary_.cycles != 0 && core_clock_hz_ != 0 && dram_clock_hz_ != 0) {
        const auto dram_ticks = summary_.cycles * dram_clock_hz_ / core_clock_hz_;
        if (dram_ticks != 0) {
            out << "gmem bus occupancy             " << std::fixed << std::setprecision(1)
                << percent(summary_.gmem_bus_beats, dram_ticks) << "% ("
                << summary_.gmem_bus_beats << " of " << dram_ticks << " dram ticks)\n";
        }
    }
    if (summary_.gmem_commands != 0) {
        out << "gmem avg command latency       "
            << summary_.gmem_command_latency_cycles / summary_.gmem_commands << " cycles\n";
    }
    out << "lsu: vector_ld=" << summary_.lsu_vector_loads
        << " vector_st=" << summary_.lsu_vector_stores
        << " scalar_ld=" << summary_.lsu_scalar_loads
        << " scalar_st=" << summary_.lsu_scalar_stores
        << " uni=" << summary_.lsu_uniform_loads
        << " sectors=" << summary_.lsu_sectors
        << " bytes=" << summary_.lsu_bytes
        << " useful_bytes=" << summary_.lsu_useful_bytes
        << " no_slot=" << summary_.lsu_no_slot_cycles
        << " no_staging=" << summary_.lsu_no_staging_cycles
        << " request_wait=" << summary_.lsu_request_wait_cycles
        << " ingress_full=" << summary_.lsu_ingress_full_cycles
        << " rf_write_wait=" << summary_.lsu_rf_write_wait_cycles
        << " rf_read_wait=" << summary_.lsu_rf_read_wait_cycles
        << " egress_wait=" << summary_.lsu_egress_wait_cycles << '\n';
    out << "tma: commands=" << summary_.tma_commands
        << " completed=" << summary_.tma_command_completions
        << " logical_bytes=" << summary_.tma_logical_bytes
        << " useful_bytes=" << summary_.tma_useful_bytes
        << " sector_bytes=" << summary_.tma_sector_bytes
        << " port_requests=" << summary_.tma_port_requests
        << " port_rounds=" << summary_.tma_port_rounds
        << " max_commands=" << summary_.tma_max_command_depth
        << " max_credits=" << summary_.tma_max_credit_depth
        << " max_returns=" << summary_.tma_max_return_depth << '\n';
    out << "tmem: load_beats=" << summary_.tmem_load_beats
        << " store_beats=" << summary_.tmem_store_beats << '\n';
    out << "mma: commands=" << summary_.mma_commands
        << " input_steps=" << summary_.mma_input_steps
        << " result_steps=" << summary_.mma_result_steps
        << " shmem_rounds=" << summary_.mma_shmem_rounds
        << " a_block_reads=" << summary_.mma_a_block_reads
        << " recurrence_stalls=" << summary_.mma_recurrence_stalls << '\n';
    out << "sfu: ops=" << summary_.sfu_ops
        << " busy_cycles=" << summary_.sfu_busy_cycles
        << " unit_stalls=" << summary_.sfu_unit_stalls << '\n';
    out << "front end: fetches=" << summary_.fetches
        << " discarded=" << summary_.fetches_discarded
        << " waiting_for_instruction="
        << summary_.warp_cycles[static_cast<std::size_t>(WarpState::no_instruction)]
        << " fifo_empty=" << summary_.instruction_fifo_empty << '\n';
    if (truncated()) out << "timeline events dropped        " << summary_.timeline_events_dropped << '\n';
    return out.str();
}

std::string TraceRecorder::perfetto_json() const {
    std::ostringstream output;
    output << R"({"traceEvents":[)";
    bool first = true;
    const auto write = [&](const json& item) {
        if (!first) output << ',';
        first = false;
        output << item.dump();
    };
    const auto metadata = [&](uint32_t track, const std::string& name) {
        write({
            {"name", "thread_name"}, {"ph", "M"}, {"pid", 1}, {"tid", track},
            {"args", {{"name", name}}},
        });
    };
    metadata(scheduler_track, "SM scheduler");
    for (uint32_t warp = 0; warp < active_warps_; ++warp) {
        metadata(warp_track_base + warp, "warp " + std::to_string(warp));
    }
    metadata(shmem_a_read_track, "shmem A-read");
    metadata(shmem_a_write_track, "shmem A-write");
    metadata(shmem_b_write_track, "shmem B-write");
    metadata(mbar_rmw_track, "mbar RMW engine");
    metadata(acc_ld_engine_track, "acc.ld engine");
    metadata(acc_st_engine_track, "acc.st engine");
    metadata(gmem_bus_track, "gmem bus");
    metadata(gmem_command_track, "gmem commands");
    metadata(gmem_queue_track, "gmem queue");
    metadata(tma_command_track, "tma commands");
    metadata(tma_command_depth_track, "tma command depth");
    metadata(tma_credit_depth_track, "tma sector credits");
    metadata(tma_return_depth_track, "tma returns");

    const auto timestamp = [&](uint64_t cycle) {
        return static_cast<double>(cycle) * 1'000'000.0 / static_cast<double>(core_clock_hz_);
    };
    for (const auto& event : events_) {
        json item = {
            {"name", event.name},
            {"cat", category_name(event.category)},
            {"pid", 1},
            {"tid", event.track},
            {"ts", timestamp(event.start_cycle)},
            {"args", {
                {"detail", event.detail},
                {"start_cycle", event.start_cycle},
                {"id", event.id},
            }},
        };
        switch (event.kind) {
        case TraceEventKind::instant:
            item["ph"] = "i";
            item["s"] = "t";
            break;
        case TraceEventKind::slice:
            if (event.async) {
                // Overlapping lifetimes become async begin/end pairs keyed by id, so the
                // viewer lays out concurrent spans instead of guessing how they nest.
                std::ostringstream id_text;
                id_text << "0x" << std::hex << event.id;
                item["id"] = id_text.str();
                item["args"]["duration_cycles"] = event.duration_cycles;
                item["ph"] = "b";
                write(item);
                item["ph"] = "e";
                item["ts"] = timestamp(event.start_cycle + event.duration_cycles);
                write(item);
                continue;
            }
            item["ph"] = "X";
            item["dur"] = timestamp(event.duration_cycles);
            item["args"]["duration_cycles"] = event.duration_cycles;
            break;
        case TraceEventKind::counter:
            item["ph"] = "C";
            // Counter args are the plotted series, one per key — anything extra
            // becomes a second curve on the track, not metadata.
            item["args"] = {{"value", event.value}};
            break;
        }
        write(item);
    }
    if (truncated()) {
        write({
            {"name", "trace truncated"}, {"cat", "trace"}, {"ph", "i"}, {"s", "g"},
            {"pid", 1}, {"tid", scheduler_track}, {"ts", timestamp(summary_.cycles)},
            {"args", {{"events_dropped", summary_.timeline_events_dropped}}},
        });
    }
    output << R"(],"displayTimeUnit":"ns"})";
    return output.str();
}

void TraceRecorder::write_perfetto(const std::string& path) const {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot create trace '" + path + "'");
    output << perfetto_json();
}

} // namespace nanosm

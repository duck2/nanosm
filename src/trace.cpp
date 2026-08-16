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
constexpr uint32_t shmem_track = 1000;
constexpr uint32_t gmem_bus_track = 1001;
constexpr uint32_t gmem_command_track = 1002;
constexpr uint32_t gmem_queue_track = 1003;

std::string category_name(TraceCategory category) {
    switch (category) {
    case TraceCategory::instruction: return "instruction";
    case TraceCategory::warp: return "warp";
    case TraceCategory::shmem: return "shmem";
    case TraceCategory::gmem: return "gmem";
    case TraceCategory::sync: return "sync";
    }
    return "unknown";
}

double percent(uint64_t part, uint64_t whole) {
    return whole == 0 ? 0.0 : 100.0 * static_cast<double>(part) / static_cast<double>(whole);
}

} // namespace

std::string warp_state_name(WarpState state) {
    switch (state) {
    case WarpState::not_selected: return "ready / not selected";
    case WarpState::issued: return "issued";
    case WarpState::stall_count: return "stall count";
    case WarpState::depbar: return "depbar";
    case WarpState::structural: return "structural";
    case WarpState::halted: return "halted";
    case WarpState::count: break;
    }
    return "unknown";
}

TraceRecorder::TraceRecorder(uint32_t active_warps, uint64_t core_clock_hz, TraceOptions options)
    : active_warps_(active_warps),
      core_clock_hz_(core_clock_hz),
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
        opcode_name(instruction.opcode),
        "warp=" + std::to_string(warp) + ", " + format_instruction(instruction),
        scheduler_track,
        cycle,
        0,
        0,
        pc,
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

void TraceRecorder::shmem_begin(uint64_t cycle, uint32_t warp, bool load, uint32_t rounds) {
    ++summary_.shmem_requests;
    shmem_slice_ = OpenSlice{
        cycle,
        shmem_track,
        TraceCategory::shmem,
        load ? "lds" : "sts",
        "warp=" + std::to_string(warp) + ", rounds=" + std::to_string(rounds),
    };
}

void TraceRecorder::shmem_round() {
    ++summary_.shmem_rounds;
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

void TraceRecorder::shmem_end(uint64_t end_cycle) {
    if (!shmem_slice_.has_value()) throw std::runtime_error("trace shmem end without begin");
    close_slice(std::move(*shmem_slice_), end_cycle);
    shmem_slice_.reset();
}

void TraceRecorder::gmem_command_begin(uint64_t cycle, uint64_t command, uint32_t warp, bool load) {
    ++summary_.gmem_commands;
    gmem_commands_.emplace(command, OpenSlice{
        cycle,
        gmem_command_track,
        TraceCategory::gmem,
        load ? "ldg command" : "stg command",
        "warp=" + std::to_string(warp),
        command,
        true, // commands from different warps overlap in flight
    });
}

void TraceRecorder::gmem_command_end(uint64_t end_cycle, uint64_t command) {
    const auto found = gmem_commands_.find(command);
    if (found == gmem_commands_.end()) throw std::runtime_error("trace gmem command end without begin");
    summary_.gmem_command_latency_cycles += end_cycle - found->second.start_cycle;
    close_slice(std::move(found->second), end_cycle);
    gmem_commands_.erase(found);
}

void TraceRecorder::gmem_sector_enqueue() {
    ++summary_.gmem_sectors;
}

void TraceRecorder::gmem_bus_begin(
    uint64_t cycle, uint64_t request, uint64_t sector_address, bool load) {
    gmem_bus_requests_.emplace(request, OpenSlice{
        cycle,
        gmem_bus_track,
        TraceCategory::gmem,
        load ? "read sector" : "write sector",
        "address=" + std::to_string(sector_address),
        request,
    });
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
    for (std::size_t i = 0; i < static_cast<std::size_t>(WarpState::count); ++i) {
        const auto state = static_cast<WarpState>(i);
        out << "  " << std::left << std::setw(27) << warp_state_name(state) << std::right
            << summary_.warp_cycles[i] << '\n';
    }
    out << "shmem: requests=" << summary_.shmem_requests
        << " rounds=" << summary_.shmem_rounds << '\n';
    out << "gmem: commands=" << summary_.gmem_commands
        << " sectors=" << summary_.gmem_sectors
        << " beats=" << summary_.gmem_bus_beats
        << " bytes=" << summary_.gmem_bytes
        << " max_queue=" << summary_.gmem_max_queue_depth << '\n';
    if (summary_.gmem_commands != 0) {
        out << "gmem avg command latency       "
            << summary_.gmem_command_latency_cycles / summary_.gmem_commands << " cycles\n";
    }
    if (truncated()) out << "timeline events dropped        " << summary_.timeline_events_dropped << '\n';
    return out.str();
}

void TraceRecorder::write_perfetto(const std::string& path) const {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot create trace '" + path + "'");

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
    metadata(shmem_track, "shmem port A");
    metadata(gmem_bus_track, "gmem bus");
    metadata(gmem_command_track, "gmem commands");
    metadata(gmem_queue_track, "gmem queue");

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
}

} // namespace nanosm

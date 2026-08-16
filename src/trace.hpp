#pragma once

#include "isa.hpp"

#include <array>
#include <cstddef>
#include <stdint.h>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace nanosm {

enum class WarpState : uint8_t {
    not_selected,
    issued,
    stall_count,
    depbar,
    structural,
    halted,
    count,
};

enum class TraceEventKind : uint8_t {
    instant,
    slice,
    counter,
};

enum class TraceCategory : uint8_t {
    instruction,
    warp,
    shmem,
    gmem,
    sync,
};

struct TraceOptions {
    bool timeline = false;
    bool debug = false;
    std::size_t max_events = 200'000;
};

struct TimelineEvent {
    TraceEventKind kind = TraceEventKind::instant;
    TraceCategory category = TraceCategory::instruction;
    std::string name;
    std::string detail;
    uint32_t track = 0;
    uint64_t start_cycle = 0;
    uint64_t duration_cycles = 0;
    int64_t value = 0;
    uint64_t id = 0;
    // Lifetimes that may partially overlap others on their track (gmem commands from
    // different warps in flight together). The exporter must not render these as
    // complete 'X' slices: overlapping completes on one track are ambiguous to nest.
    bool async = false;
};

struct TraceSummary {
    uint64_t cycles = 0;
    uint64_t instructions = 0;
    uint64_t scheduler_issued_cycles = 0;
    uint64_t scheduler_idle_cycles = 0;
    uint64_t scheduler_draining_cycles = 0;
    std::array<uint64_t, static_cast<std::size_t>(WarpState::count)> warp_cycles{};
    uint64_t shmem_requests = 0;
    uint64_t shmem_rounds = 0;
    uint64_t gmem_commands = 0;
    uint64_t gmem_sectors = 0;
    uint64_t gmem_bus_beats = 0;
    uint64_t gmem_bytes = 0;
    uint64_t gmem_command_latency_cycles = 0;
    uint64_t gmem_max_queue_depth = 0;
    uint64_t timeline_events_dropped = 0;
};

class TraceRecorder {
public:
    TraceRecorder(uint32_t active_warps, uint64_t core_clock_hz, TraceOptions options = {});

    void issue(uint64_t cycle, uint32_t warp, uint32_t pc, const Instruction& instruction);
    void scheduler_idle(bool draining);
    void warp_state(uint64_t cycle, uint32_t warp, WarpState state, std::string detail = {});

    void depbar(uint64_t cycle, uint32_t warp, uint8_t bar, uint32_t value);
    void shmem_begin(uint64_t cycle, uint32_t warp, bool load, uint32_t rounds);
    void shmem_round();
    void shmem_end(uint64_t end_cycle);

    void gmem_command_begin(uint64_t cycle, uint64_t command, uint32_t warp, bool load);
    void gmem_command_end(uint64_t end_cycle, uint64_t command);
    void gmem_sector_enqueue();
    void gmem_bus_begin(uint64_t cycle, uint64_t request, uint64_t sector_address, bool load);
    void gmem_bus_beat(uint64_t cycle, uint64_t dram_tick, uint64_t request);
    void gmem_bus_end(uint64_t end_cycle, uint64_t request, uint32_t sector_bytes);
    void gmem_queue_depth(uint64_t cycle, std::size_t depth);

    void finish(uint64_t cycle);
    [[nodiscard]] const TraceSummary& summary() const { return summary_; }
    [[nodiscard]] const std::vector<TimelineEvent>& events() const { return events_; }
    [[nodiscard]] bool truncated() const { return summary_.timeline_events_dropped != 0; }
    [[nodiscard]] std::string summary_text() const;
    void write_perfetto(const std::string& path) const;

private:
    struct WarpInterval {
        bool active = false;
        WarpState state = WarpState::not_selected;
        std::string detail;
        uint64_t start_cycle = 0;
    };

    struct OpenSlice {
        uint64_t start_cycle = 0;
        uint32_t track = 0;
        TraceCategory category = TraceCategory::gmem;
        std::string name;
        std::string detail;
        uint64_t id = 0;
        bool async = false;
    };

    uint32_t active_warps_ = 0;
    uint64_t core_clock_hz_ = 0;
    TraceOptions options_;
    TraceSummary summary_;
    std::vector<TimelineEvent> events_;
    std::vector<WarpInterval> warp_intervals_;
    std::optional<OpenSlice> shmem_slice_;
    std::map<uint64_t, OpenSlice> gmem_commands_;
    std::map<uint64_t, OpenSlice> gmem_bus_requests_;
    std::optional<std::size_t> last_gmem_queue_depth_;

    void add(TimelineEvent event);
    void close_warp_interval(uint32_t warp, uint64_t end_cycle);
    void close_slice(OpenSlice slice, uint64_t end_cycle);
};

std::string warp_state_name(WarpState state);

} // namespace nanosm

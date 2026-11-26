#pragma once

#include "isa.hpp"

#include <array>
#include <cstddef>
#include <stdint.h>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace nanosm {

enum class WarpState : uint8_t {
    // Has an instruction and no wait of its own, but another warp issued
    // first this cycle, so its depbars and resources were not checked.
    not_examined,
    issued,
    stall_count,
    depbar,
    mbar_wait,
    structural,
    shmem_queue,  // a shared-memory instruction whose queue is full
    shmem_order,  // a shmem access ordered behind the warp's own unserviced access
    // The warp has no instruction to offer issue: its instruction FIFO is
    // empty, or its branch has not resolved.
    // Instruction supply, not a dependency or resource wait.
    no_instruction,
    halted,
    count,
};

// The shmem services that the trace attributes grants to. The MMA's port-B
// reads are counted by the MMA.
enum class ShmemService : uint8_t { a_read, a_write, b_write };
enum class ShmemClient : uint8_t { simt, probe, rmw, tma };

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
    tma,
    tmem,
    mma,
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
    // SIMT lds / lds.uni / sts accesses and TMA shmem transactions (data and
    // descriptor), and their bank rounds; probes and RMW steps are counted by
    // the mbar line.
    uint64_t shmem_requests = 0;
    uint64_t shmem_rounds = 0;
    // Warp-cycles a read-side (write-side) instruction was refused because
    // its queue was full, and warp-cycles a shmem access was ordered behind
    // the warp's own unserviced access.
    uint64_t shmem_read_queue_full = 0;
    uint64_t shmem_write_queue_full = 0;
    uint64_t shmem_order_waits = 0;
    // Per side: cycles the queue's head could leave but the collect stage was
    // busy, and cycles the collect stage asked for an RF bank and was denied.
    uint64_t shmem_read_head_waits = 0;
    uint64_t shmem_write_head_waits = 0;
    uint64_t shmem_read_bank_waits = 0;
    uint64_t shmem_write_bank_waits = 0;
    // Cycles a store waited to start (the write side's serve holds it, not
    // started) while an RMW's store pause held A-write.
    uint64_t shmem_store_paused = 0;
    // Rounds granted, per service and client.
    uint64_t shmem_a_read_simt = 0;
    uint64_t shmem_a_read_probe = 0;
    uint64_t shmem_a_read_rmw = 0;
    uint64_t shmem_a_read_tma = 0;
    uint64_t shmem_a_write_simt = 0;
    uint64_t shmem_a_write_rmw = 0;
    uint64_t shmem_b_write_tma = 0;
    // The RF write decision: vector beats written by async
    // holders (load slots, the load buffer, acc.ld pipelines); holder-cycles a
    // ready holder was not granted; warp-cycles a SIMT instruction was refused
    // because its result bank was inhibited (async service owed); and the
    // mirrored read rule's: warp-cycles refused because a bank its reads
    // book was inhibited (async read service owed).
    uint64_t rf_async_writes = 0;
    uint64_t rf_async_waits = 0;
    uint64_t rf_simt_inhibited = 0;
    uint64_t rf_simt_read_inhibited = 0;
    // Uniform-register writes presented at the URF write port; warp-cycles a
    // uniform reader was refused because a tma's later operand rounds held
    // the URF read ports (src/timing.hpp tma_operand_rounds).
    uint64_t urf_writes = 0;
    uint64_t urf_read_ports_held = 0;
    uint64_t mbar_rmw_requests = 0;
    uint64_t mbar_phase_changes = 0;
    // Cycles an engine's held mbar event (TMA complete_tx / MMA commit arrive)
    // waited while the RMW engine served the OTHER engine's event.
    uint64_t mbar_engine_waits_on_other = 0;
    uint64_t mbar_initial_probes = 0;
    uint64_t mbar_immediate_successes = 0;
    uint64_t mbar_sleeps = 0;
    uint64_t mbar_wakes = 0;
    uint64_t mbar_timeouts = 0;
    uint64_t gmem_commands = 0;
    uint64_t gmem_sectors = 0;
    uint64_t gmem_bus_beats = 0;
    uint64_t gmem_bytes = 0;
    uint64_t gmem_command_latency_cycles = 0;
    uint64_t gmem_max_queue_depth = 0;
    // LSU.
    uint64_t lsu_vector_loads = 0;
    uint64_t lsu_vector_stores = 0;
    uint64_t lsu_scalar_loads = 0;
    uint64_t lsu_scalar_stores = 0;
    uint64_t lsu_uniform_loads = 0;
    uint64_t lsu_sectors = 0;           // controller requests accepted from the LSU
    uint64_t lsu_bytes = 0;             // their sector bytes on the bus
    uint64_t lsu_useful_bytes = 0;      // bytes the instructions named
    uint64_t lsu_queue_full_cycles = 0; // warp-cycles: an LSU op refused at issue, its queue full
    uint64_t lsu_no_slot_cycles = 0;    // cycles the queue's head waited for a command slot
    uint64_t lsu_store_buffer_wait_cycles = 0; // cycles the queue's head (a scalar stg) waited for the store buffer
    uint64_t lsu_request_wait_cycles = 0; // LSU request ready, controller queue full
    uint64_t lsu_ingress_full_cycles = 0; // cycles a read beat was held: ingress FIFO full
    uint64_t lsu_rf_write_wait_cycles = 0;
    uint64_t lsu_rf_read_wait_cycles = 0;
    uint64_t lsu_egress_wait_cycles = 0;
    uint64_t gmem_bus_turns = 0;        // changes of direction on the data bus
    uint64_t gmem_turnaround_ticks = 0; // DRAM ticks a ready entry waited for gmem.turnaround_dram_ticks
    // The launch copy (0 bytes: none): where the block came from and went,
    // and when each launch flag cleared (depbar, block bytes, cycle).
    // Bytes a TEST placed in shared memory before the first cycle, untimed
    // (Launch::test_shmem). 0 for a launch that used no such access.
    uint64_t test_preload_bytes = 0;
    uint64_t launch_block_bytes = 0;
    uint64_t launch_global_address = 0;
    uint32_t launch_shared_address = 0;
    struct LaunchFlagCleared {
        uint8_t bar = 0;
        uint32_t bytes = 0;
        uint64_t cycle = 0;
    };
    std::vector<LaunchFlagCleared> launch_flags;
    uint64_t tma_commands = 0;
    uint64_t tma_command_completions = 0;
    uint64_t tma_logical_bytes = 0;
    uint64_t tma_useful_bytes = 0;
    uint64_t tma_sector_bytes = 0;
    // Data placement (port-B write) and readback (A-read): one request per TMA
    // load-write (incl. synthetic OOB zero fills) or store-read, and the bank
    // rounds those requests consume. Descriptor reads are excluded.
    uint64_t tma_port_requests = 0;
    uint64_t tma_port_rounds = 0;
    uint64_t tma_max_command_depth = 0;
    uint64_t tma_max_credit_depth = 0;
    uint64_t tma_max_return_depth = 0;
    uint64_t tmem_load_beats = 0;
    uint64_t tmem_store_beats = 0;
    uint64_t mma_commands = 0;
    uint64_t mma_input_steps = 0;
    uint64_t mma_result_steps = 0;
    uint64_t mma_recurrence_stalls = 0;
    uint64_t mma_shmem_rounds = 0;
    uint64_t mma_a_block_reads = 0;   // steps that read their A block (the rest use the held one)
    uint64_t sfu_ops = 0;
    uint64_t sfu_busy_cycles = 0;
    uint64_t sfu_unit_stalls = 0;
    // The front end: instruction-memory reads (one packet each, one per cycle
    // at most), and fetched words that never issued -- dropped as they came
    // out of the memory, or cleared from a FIFO, by a taken branch or a halt,
    // or the first half of a packet fetched for a second-half branch target,
    // skipped as it landed. Once the run is done (every warp halted, no read
    // in flight): 2 x fetches = instructions + fetches_discarded.
    // The warp-cycles with no instruction to offer are
    // warp_cycles[no_instruction] (the warp's state: its own waits come
    // first). instruction_fifo_empty counts every running warp's cycles with
    // an empty FIFO whatever else it waits on: a supply observation, not a
    // count of lost issues.
    uint64_t fetches = 0;
    uint64_t fetches_discarded = 0;
    uint64_t instruction_fifo_empty = 0;
    uint64_t timeline_events_dropped = 0;
};

class TraceRecorder {
public:
    TraceRecorder(
        uint32_t active_warps,
        uint64_t core_clock_hz,
        uint64_t dram_clock_hz,
        TraceOptions options = {});

    // The profile's MMA shape list, so an issued mma.async is named by its
    // shape (src/isa.hpp opcode_name).
    void set_mma_shapes(std::vector<MmaShape> shapes) { mma_shapes_ = std::move(shapes); }
    void issue(uint64_t cycle, uint32_t warp, uint32_t pc, const Instruction& instruction);
    // A fetch granted (debug traces: an instant on the warp's track; its id
    // is the pc read), and fetched words discarded without issuing.
    void fetch(uint64_t cycle, uint32_t warp, uint32_t pc);
    void fetches_discarded(uint64_t words) { summary_.fetches_discarded += words; }
    // This cycle's running warps with an empty instruction FIFO.
    void instruction_fifo_empty(uint64_t warps) { summary_.instruction_fifo_empty += warps; }
    void rf_read(uint64_t cycle, uint32_t warp, uint8_t reg, uint32_t bank);
    // Debug only: a write presented at the RF write port (src/timing.hpp: it is
    // at the RAM Wd cycles later and visible from the cycle after that).
    void rf_write(uint64_t cycle, uint32_t warp, uint8_t reg, uint32_t bank);
    // A uniform-register write presented at the URF write port (src/timing.hpp:
    // visible Wu + 1 later): an instant on the warp's track with the register
    // and its value, in every timeline trace (one per uniform writer).
    void urf_write(uint64_t cycle, uint32_t warp, uint8_t reg, uint32_t value);
    void sfu_accept(uint32_t occupancy_cycles);
    void sfu_unit_stall();
    void scheduler_idle(bool draining);
    void warp_state(uint64_t cycle, uint32_t warp, WarpState state, std::string detail = {});

    void depbar(uint64_t cycle, uint32_t warp, uint8_t bar, uint32_t value);
    // A SIMT access's or TMA transaction's slice on its service's track, from
    // its first grant to its last; one shmem request.
    void shmem_begin(ShmemService service, uint64_t cycle, std::string operation, std::string detail,
        TraceCategory category, uint64_t id = 0);
    void shmem_end(ShmemService service, uint64_t end_cycle);
    // One bank round granted (an instant on the service's track in debug
    // traces; its value is the requester's slot index).
    void shmem_grant(uint64_t cycle, ShmemService service, ShmemClient client, uint32_t slot = 0);
    void mbar_rmw_begin(uint64_t cycle, uint32_t byte_address, std::string operation);
    void mbar_rmw_end(uint64_t end_cycle, bool phase_changed);
    // A uniform atomic's "atomic RMW" slice on the RMW engine's track, from
    // its start to its write grant + 1 (not an mbar RMW: not counted in the
    // mbar line; its read and write are a_read_rmw / a_write_rmw grants).
    void atomic_rmw_begin(uint64_t cycle, std::string detail);
    void atomic_rmw_end(uint64_t end_cycle);
    void mbar_engine_wait_on_other() { ++summary_.mbar_engine_waits_on_other; }
    void mbar_probe(
        uint64_t cycle, uint32_t warp, uint32_t byte_address, uint8_t parity, bool complete);
    void mbar_sleep(uint64_t cycle, uint32_t warp, uint32_t byte_address, uint8_t parity);
    void mbar_wake(uint64_t cycle, uint32_t warp, uint32_t byte_address, uint8_t phase);
    void mbar_timeout(uint64_t cycle, uint32_t warp, uint32_t byte_address, uint8_t parity);

    void shmem_queue_full(bool write_side) { ++(write_side ? summary_.shmem_write_queue_full : summary_.shmem_read_queue_full); }
    void shmem_head_wait(bool write_side) { ++(write_side ? summary_.shmem_write_head_waits : summary_.shmem_read_head_waits); }
    void shmem_bank_wait(bool write_side) { ++(write_side ? summary_.shmem_write_bank_waits : summary_.shmem_read_bank_waits); }
    void shmem_order_wait() { ++summary_.shmem_order_waits; }
    void shmem_store_pause_wait() { ++summary_.shmem_store_paused; }
    void rf_async_write() { ++summary_.rf_async_writes; }
    void rf_async_wait() { ++summary_.rf_async_waits; }
    void rf_simt_inhibited() { ++summary_.rf_simt_inhibited; }
    void rf_simt_read_inhibited() { ++summary_.rf_simt_read_inhibited; }
    void urf_read_ports_held() { ++summary_.urf_read_ports_held; }
    // One event per LSU command (gmem commands track): admit .. completion,
    // its phases and stall reasons in the detail. No per-word events.
    void lsu_command_begin(uint64_t cycle, uint64_t command, uint32_t warp, std::string name);
    void lsu_command_end(uint64_t end_cycle, uint64_t command, const std::string& phases);
    [[nodiscard]] TraceSummary& lsu_counters() { return summary_; }
    void gmem_sector_enqueue();
    void gmem_bus_begin(
        uint64_t cycle, uint64_t request, uint64_t sector_address, bool load,
        std::string owner = "lsu");
    void gmem_bus_beat(uint64_t cycle, uint64_t dram_tick, uint64_t request);
    void gmem_bus_end(uint64_t end_cycle, uint64_t request, uint32_t sector_bytes);
    void gmem_queue_depth(uint64_t cycle, std::size_t depth);

    void tma_command_begin(
        uint64_t cycle, uint64_t command, uint32_t warp, uint32_t descriptor_address,
        std::size_t command_depth);
    void tma_command_end(
        uint64_t end_cycle, uint64_t command, uint64_t logical_bytes, uint64_t useful_bytes);
    // The launch copy: the TMA command the launch record starts (it counts in
    // the tma and gmem numbers like any load), and each launch flag's
    // decrement (the cycle from which a waiting instruction can issue).
    void launch_copy_begin(
        uint64_t cycle, uint64_t command, uint64_t global_address, uint32_t shared_address, uint32_t bytes);
    void launch_flag(uint64_t cycle, uint8_t bar, uint32_t bytes);
    void test_preload(uint64_t bytes);
    void tma_sector(uint32_t sector_bytes);
    void tma_data_port(uint32_t rounds);
    void tma_depths(
        uint64_t cycle, std::size_t command_depth, std::size_t credit_depth,
        std::size_t return_depth);
    // One beat of the acc.ld engine (a tmem read) or the acc.st engine (a tmem
    // write), on that engine's track; `address` is lane 0's logical tmem word.
    void tmem_beat(uint64_t cycle, uint32_t warp, bool load, uint32_t address);
    void mma_command(uint64_t cycle, uint64_t command, uint32_t warp, bool init);
    void mma_input(uint64_t cycle, uint64_t command, uint32_t rounds, uint32_t step, bool reads_a);
    // A result block's tmem write; `address` is its first logical tmem word.
    void mma_result(uint64_t cycle, uint64_t command, uint32_t address);
    void mma_recurrence_stall();

    void finish(uint64_t cycle);
    [[nodiscard]] const TraceSummary& summary() const { return summary_; }
    [[nodiscard]] const std::vector<TimelineEvent>& events() const { return events_; }
    [[nodiscard]] bool truncated() const { return summary_.timeline_events_dropped != 0; }
    [[nodiscard]] std::string summary_text() const;
    [[nodiscard]] std::string perfetto_json() const;
    void write_perfetto(const std::string& path) const;

private:
    struct WarpInterval {
        bool active = false;
        WarpState state = WarpState::not_examined;
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
    uint64_t dram_clock_hz_ = 0;
    TraceOptions options_;
    TraceSummary summary_;
    std::vector<MmaShape> mma_shapes_;
    std::vector<TimelineEvent> events_;
    std::vector<WarpInterval> warp_intervals_;
    std::array<std::optional<OpenSlice>, 3> shmem_slices_; // per ShmemService
    std::optional<OpenSlice> mbar_rmw_slice_;
    std::map<uint64_t, OpenSlice> gmem_commands_;
    std::map<uint64_t, OpenSlice> gmem_bus_requests_;
    std::map<uint64_t, OpenSlice> tma_commands_;
    std::optional<std::size_t> last_gmem_queue_depth_;
    std::optional<std::size_t> last_tma_command_depth_;
    std::optional<std::size_t> last_tma_credit_depth_;
    std::optional<std::size_t> last_tma_return_depth_;

    void add(TimelineEvent event);
    void close_warp_interval(uint32_t warp, uint64_t end_cycle);
    void close_slice(OpenSlice slice, uint64_t end_cycle);
};

std::string warp_state_name(WarpState state);

} // namespace nanosm

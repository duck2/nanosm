#pragma once

#include "compiler.hpp"
#include "image.hpp"
#include "mbar.hpp"
#include "profile.hpp"
#include "reg.hpp"
#include "rf.hpp"
#include "sanitizer.hpp"
#include "trace.hpp"
#include "tma.hpp"

#include <array>
#include <cstddef>
#include <stdint.h>
#include <deque>
#include <map>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace nanosm {

// Test seam for the placeholder gmem controller's return order. The default
// (fifo) serves the queue strictly in acceptance order, which is a legal
// special case of the boundary contract (in order within a read ID, free
// across IDs). reverse_free_ids is also legal: when the queue front is a
// free-ID (scalar) read it serves the youngest ready free-ID read of that
// leading run first, provoking out-of-order gather returns.
// illegal_ordered_reorder breaks the contract on purpose (serves the second
// entry of an ordered stream before the first) so the SM's tripwire can be
// tested. Never used by kernels or launches.
// interleave_ordered_streams is legal too: when the front is an ordered-
// stream read and the last read served was of the same stream, it serves the
// oldest read of a DIFFERENT ordered stream from the leading run of reads,
// so returns interleave across stream IDs while every ID stays in order.
enum class ControllerTestOrder : uint8_t {
    fifo,
    reverse_free_ids,
    illegal_ordered_reorder,
    interleave_ordered_streams,
};

// Test hook (B2.1): one event per datum movement across a declared edge of
// the rfc09 3a handoff table. Recorded only when enabled; never consulted by
// the machine. Field meaning per edge is listed with each enumerator.
enum class LsuEdge : uint8_t {
    bus_read_beat,     // bus -> ingress FIFO. id = request, aux = byte offset in sector
    ingress_read,      // ingress FIFO read port. id = request, aux = byte offset
    tma_reg_write,     // ingress -> TMA sector register. id = request, aux = byte offset
    assembly_write,    // ingress -> assembly register. id = command, aux = request
    rf_write_beat,     // assembly -> RF. id = command, aux = register
    tma_place_round,   // TMA sector register -> shmem (port A round). id = request, aux = round
    rf_read_beat,      // RF -> assembly (stores). id = command, aux = register
    tma_store_round,   // shmem -> TMA store sector register (port A round). id = tma command, aux = round
    egress_write,      // assembly / TMA store register -> egress FIFO. id = slot, aux = 0 TMA / 1 LSU
    write_request,     // a write's request accepted by the controller. id = request, aux = slot
    bus_write_beat,    // egress FIFO -> bus. id = request, aux = byte offset in sector
    write_ack,         // the write's acknowledgement. id = request
};

struct LsuEdgeEvent {
    uint64_t cycle = 0;
    uint64_t dram_tick = 0;
    LsuEdge edge = LsuEdge::bus_read_beat;
    uint64_t id = 0;
    uint64_t aux = 0;
    uint32_t bytes = 0;
};

class Machine {
public:
    Machine(Config config, Kernel kernel, Launch launch, TraceOptions trace_options = {});
    // The registers are registered by address (src/reg.hpp): never copied or moved.
    Machine(const Machine&) = delete;
    Machine& operator=(const Machine&) = delete;

    void set_controller_test_order(ControllerTestOrder order) { controller_test_order_ = order; }
    void enable_edge_probe() { edge_probe_enabled_ = true; }
    // Test seam (the driver-shuffle regression guard): run the register
    // drivers, and the independent request generators, in a random order every
    // cycle. Every result, trace byte and counter must be unchanged.
    void set_driver_shuffle_seed(uint64_t seed);
    [[nodiscard]] const std::vector<LsuEdgeEvent>& edge_events() const { return edge_events_; }
    [[nodiscard]] uint32_t tma_sector_credits_used() const { return tma_.cur().credits_used; }
    [[nodiscard]] std::size_t lsu_commands_in_flight() const { return lsu_commands_.cur().size(); }
    // Test seams: whether each engine's one-entry mbar event register is full.
    [[nodiscard]] bool tma_mbar_event_held() const { return tma_mbar_event_.cur().has_value(); }
    [[nodiscard]] bool mma_mbar_event_held() const { return mma_mbar_event_.cur().has_value(); }
    // Test seams (Q, between ticks): a warp's depbar counter; the MMA queue's depth.
    [[nodiscard]] uint32_t depbar_value(uint32_t warp, uint8_t bar) const { return depbar_.cur().at(warp).at(bar); }
    [[nodiscard]] std::size_t mma_queue_depth() const { return mma_.cur().queue.size(); }
    // Test seams (Q, between ticks): an assembly register's owner (an LSU
    // command id), its lane valid bits and their count.
    [[nodiscard]] std::optional<uint64_t> assembly_owner(uint32_t index) const {
        return staging_.at(index).cur().owner;
    }
    [[nodiscard]] const std::vector<uint8_t>& assembly_valid(uint32_t index) const {
        return staging_.at(index).cur().valid;
    }
    [[nodiscard]] uint32_t assembly_valid_count(uint32_t index) const {
        return staging_.at(index).cur().valid_count;
    }

    void tick();
    void run(uint64_t cycle_limit = 100000);

    [[nodiscard]] bool done() const;
    [[nodiscard]] uint64_t cycle() const { return cycle_; }
    [[nodiscard]] const TraceRecorder& trace() const { return *trace_; }
    [[nodiscard]] const std::vector<uint8_t>& shmem() const { return shmem_; }
    [[nodiscard]] const std::vector<uint8_t>& gmem() const { return gmem_; }

    [[nodiscard]] uint32_t read_shmem_u32(uint32_t byte_address) const;
    [[nodiscard]] uint32_t read_gmem_u32(uint64_t byte_address) const;
    [[nodiscard]] MbarState read_mbar_state(uint32_t byte_address) const;
    [[nodiscard]] uint32_t read_reg(
        uint32_t warp, uint32_t lane, uint32_t reg) const;
    // Host/test seam for a DMA complete_tx: it is written into the TMA's event
    // register between ticks (throws if that register is full; the MMA's is
    // independent).
    void complete_mbar_tx(uint32_t byte_address, uint32_t bytes, uint8_t parity);

private:
    // =======================================================================
    // Machine structures (unchanged meaning; see trash/lsu-b21-2026-10-04.md
    // and trash/harness-and-mbar-events-2026-10-07.md)
    // =======================================================================

    struct MbarWaitSlot {
        uint32_t warp = 0;
        uint32_t byte_address = 0;
        uint8_t parity = 0;
        uint8_t predicate_dst = 0;
        bool parked = false;
        uint64_t deadline_cycle = 0;
    };

    // Issue-internal warp state (only the scheduler reads or writes it), plus
    // the warp's RF and predicate memories. The depbar counters and the mbar
    // wait slots, which other units complete, are registers (below).
    struct Warp {
        uint32_t id = 0;
        uint32_t pc = 0; // byte PC; every decoded instruction still occupies one 64-bit slot
        bool halted = false;
        uint64_t stall_until = 0;
        std::vector<uint32_t> rf;
        // RAW/WAW/WAR horizons (no scoreboard exists) live in the Sanitizer.
        std::array<std::vector<uint8_t>, 7> predicates;
    };

    struct Writeback {
        uint64_t ready_cycle = 0;
        uint32_t warp = 0;
        std::optional<uint8_t> dst;
        std::optional<uint8_t> predicate_dst;
        std::vector<uint32_t> values;
        std::optional<uint8_t> clear_depbar;
        // The final LSU RF write / last-ack completion ends its command: the
        // command slot and its assembly register are released at this commit.
        std::optional<uint64_t> release_lsu_command;
        std::string reason;
    };

    struct ArithmeticRead {
        uint64_t issue_cycle = 0;
        uint32_t warp = 0;
        Instruction instruction;
        std::vector<RfRead> reads;
        std::vector<std::vector<uint32_t>> operands;
    };

    struct LaneAccess {
        uint32_t lane = 0;
        uint8_t word = 0;
        uint64_t byte_address = 0;
        uint32_t value = 0;
    };

    struct ShmemRequest {
        bool load = false;
        bool fan_out = false; // lds.uni: its one word goes to every lane
        uint32_t warp = 0;
        uint8_t dst = 0;
        uint8_t depbar = 0;
        std::vector<std::vector<LaneAccess>> rounds;
        std::size_t next_round = 0;
        std::vector<uint32_t> load_values;
    };

    // The sanitizer keys its tmem owner by the slot [bank][direction].
    struct AccTransfer {
        bool load = false;
        uint32_t warp = 0;
        uint8_t accumulator = 0;
        uint8_t rf_base = 0;
        uint8_t depbar = 0;
        uint32_t row_base = 0;
        uint32_t col_base = 0;
        uint32_t width = 1;
        uint32_t next_column = 0;
    };

    struct MmaCommand {
        uint64_t id = 0;
        bool commit = false;
        bool init = false;
        uint32_t warp = 0;
        uint8_t accumulator = 0;
        uint8_t depbar = 0;
        uint32_t a_base = 0;
        uint32_t b_base = 0;
        uint32_t mbar_address = 0;
        uint32_t next_step = 0;
        uint32_t outstanding_results = 0;
        bool operands_consumed = false;
    };

    struct MmaResult {
        uint64_t ready_cycle = 0;
        uint64_t command = 0;
        uint8_t accumulator = 0;
        uint32_t m0 = 0;
        uint32_t n0 = 0;
        std::vector<uint32_t> values;
    };

    // One in-flight operand fetch. The port serves one bank round per cycle
    // and values are captured in the cycle their round actually reads them;
    // fetch completion is core entry, one event with three names (case 05).
    struct MmaFetch {
        uint64_t command = 0;
        uint8_t accumulator = 0;
        bool last_step = false;
        bool init_step = false;
        uint32_t m0 = 0;
        uint32_t n0 = 0;
        uint32_t k0 = 0;
        uint32_t a_base = 0;
        uint32_t b_base = 0;
        uint32_t next_round = 0;
        std::vector<std::vector<uint32_t>> round_words;
        std::vector<std::vector<uint32_t>> round_elements;
        std::map<uint32_t, uint16_t> captured;
    };

    enum class MbarRmwOperation : uint8_t {
        init,
        arrive,
        arrive_expect_tx,
        complete_tx,
    };

    struct MbarRmwRequest {
        MbarRmwOperation operation = MbarRmwOperation::arrive;
        uint32_t byte_address = 0;
        uint32_t value = 0;
        uint8_t parity = 0;
        std::optional<uint32_t> warp;
        uint32_t pc = 0;
        Instruction instruction;
    };

    // The two engines that present mbar events to the RMW engine.
    enum class MbarEngine : uint8_t { tma, mma };

    struct ActiveMbarRmw {
        MbarRmwRequest request;
        MbarState state;
        bool write_stage = false;
        // (T) Whose engine event this is (none: SIMT). Read only for the
        // trace label and the engine_waits_on_other counter.
        std::optional<MbarEngine> engine;
    };

    struct MbarReadRequest {
        bool test = false;
        uint32_t warp = 0;
        uint8_t predicate_dst = 0;
        std::optional<uint32_t> wait_slot;
        uint32_t byte_address = 0;
        uint8_t parity = 0;
        uint32_t pc = 0;
        Instruction instruction;
    };

    // ---- gmem controller (placeholder) + LSU: trash/rfc09.md section 3a,
    // trash/lsu-b21-2026-10-04.md ----
    // Two clients share one request channel: the TMA and the LSU. When both
    // have a request in the same core cycle they alternate at sector
    // boundaries; a client that is not ready never consumes the other's turn.
    // A write's request is issued by its client when its sector is complete
    // and visible at the egress FIFO's request pointer.
    enum class GmemClient : uint8_t { tma, lsu };

    // Read IDs. Stream 0 is the TMA (ordered, one 32 B sector register);
    // 1..read_streams-1 are ordered vector streams (round-robin over vector
    // loads; no sector register, beats pluck straight into the command's
    // assembly register). Free IDs (gathers, ldg.uni, scatters) are tags in
    // [read_streams, read_streams + lsu.sectors_in_flight).
    static constexpr uint32_t tma_stream = 0;

    // One entry of the controller queue (capacity gmem.queue_sectors, the
    // entry in bus service included). Address/metadata only: read data lives
    // in DRAM until its beats go on the bus; write data lives in the SM's
    // egress FIFO until its beats go on the bus. Reads hold their entry until
    // their last beat is in the ingress FIFO; writes until their last beat is
    // on the bus.
    struct GmemSectorRequest {
        enum class Owner : uint8_t {
            lsu,
            tma_load,
            tma_store,
        };

        uint64_t id = 0;
        uint64_t command = 0;       // tma command id or lsu command id
        uint32_t stream = 0;        // read ID; writes: free-ID tag stream for scatters
        uint64_t sector_address = 0;
        uint64_t ready_dram_tick = 0;
        bool load = false;
        Owner owner = Owner::lsu;
        struct TmaFragment {
            uint64_t gmem_address = 0;
            uint32_t shmem_address = 0;
            uint8_t size = 0;
            std::array<uint8_t, 4> data{}; // descriptor/phase reads and OOB zero fills only
        };
        std::vector<uint8_t> data;  // reads: the sector as DRAM holds it when its burst starts
        uint64_t byte_enable = 0;   // writes: bit b enables sector byte b (AXI strobes)
        uint64_t egress_slot = 0;   // writes: the egress FIFO slot holding the data
        uint32_t beats = 0;
        uint32_t next_beat = 0;
    };

    // What the SM remembers about an accepted read until its sector has been
    // consumed: where each returning byte goes. This is request metadata (an
    // ID-indexed table), not payload storage.
    enum class ReadKind : uint8_t { tma, vector, gather, uniform };
    struct ReadRecord {
        ReadKind kind = ReadKind::tma;
        uint32_t stream = 0;
        uint64_t command = 0;
        uint64_t sector_address = 0;
        uint32_t sector_index = 0;  // vector: sector within the 512 B block
        uint32_t lane = 0;          // gather: the lane that asked
        uint32_t word_offset = 0;   // gather/uniform: the word's byte offset in the sector
        uint32_t tag = 0;           // gather/uniform: free-ID tag
        uint32_t received = 0;      // bytes plucked at the SM
        std::vector<GmemSectorRequest::TmaFragment> fragments; // tma placement (shmem addresses)
    };

    // One bus beat in the SM's ingress FIFO (a BRAM, simple dual port).
    // Capacity lsu.ingress_sectors x sector bytes. Written by the bus (one
    // beat per DRAM tick); readable from visible_cycle = write cycle +
    // lsu.ingress_latency_cycles; read lsu.pluck_words_per_cycle words per
    // cycle. Never fall-through. (Room freed by a read is a separate
    // contract: reusable from the next cycle, see tick_gmem_controller.)
    struct IngressBeat {
        uint64_t request = 0;
        uint32_t offset = 0;        // byte offset of this beat in its sector
        std::vector<uint8_t> bytes;
        uint32_t consumed = 0;      // bytes already plucked
        uint64_t visible_cycle = 0;
    };

    // The TMA stream's 32 B sector register (flops). Filled from the ingress
    // FIFO at the pluck rate; when complete, placed into shmem by port A in
    // its bank rounds, which read the register's Q directly. Free after the
    // last round (and refillable in that cycle).
    struct TmaSectorRegister {
        std::optional<uint64_t> request;
        uint64_t sector_address = 0;
        std::vector<uint8_t> data;
        uint32_t filled = 0;        // bytes plucked in
        bool complete = false;      // all bytes in; its ticket is in the TMA queue's returns
    };

    // The TMA store sector register (32 B flops). Allocated when the walker
    // starts a sector's port-A source read; written by those bank rounds;
    // when complete, moved into one egress slot at the pluck rate through
    // the shared egress write port (TMA priority). Free after its last word.
    struct TmaStoreRegister {
        bool busy = false;
        uint64_t command = 0;
        uint64_t sector_address = 0;
        std::vector<uint8_t> data;
        uint64_t byte_enable = 0;
        bool complete = false;
        uint32_t words_out = 0;     // words already written into the egress FIFO
    };

    // A 64 B (lane-wide) assembly register from the lsu.staging_vectors pool.
    // Owner: one LSU command from admission to its end (vector load: its
    // eighth RF write; gather/uniform: its single RF write; store: its last
    // acknowledgement). Never recycled between the chunks of a command.
    // Loads: filled from the ingress FIFO at the pluck rate (vector: lane =
    // word offset in the 64 B chunk; gather: the selected word; uni: one word
    // to every lane), dumped by one full-width RF write. Stores: filled by
    // one full-width RF read beat (vector) or the issue read (scalar), drained
    // into the egress FIFO at the pluck rate.
    struct AssemblyRegister {
        std::optional<uint64_t> owner;
        std::vector<uint32_t> words;     // one per lane
        std::vector<uint8_t> valid;      // lane enables
        uint32_t valid_count = 0;
    };

    enum class LsuKind : uint8_t { vector_load, vector_store, scalar_load, scalar_store, uniform_load };

    // One controller request an LSU command will emit (scalar/uniform plan;
    // vector requests are computed from the block base).
    struct LsuSector {
        uint64_t sector_address = 0;
        uint32_t lane = 0;
        uint32_t word_offset = 0;
    };

    // An admitted ldg/stg/ldg.v8/stg.v8/ldg.uni. Owner of one command slot
    // (lsu.active_commands) and one assembly register from admission until
    // its completion commits.
    struct LsuCommand {
        uint64_t id = 0;
        LsuKind kind = LsuKind::vector_load;
        uint32_t warp = 0;
        uint32_t pc = 0;
        Instruction instruction;
        uint8_t reg = 0;            // rD (loads) / rS (vector store)
        uint8_t depbar = 0;
        uint32_t assembly = 0;      // index into staging_
        uint64_t base = 0;          // vector: 64 B-aligned block address
        std::vector<LsuSector> sectors; // scalar / uniform
        uint32_t stream = 0;        // vector load: its ordered read stream (1..read_streams-1)
        uint32_t requests_total = 0;
        uint32_t requests_emitted = 0;
        uint32_t chunks_done = 0;   // vector load: RF writes; vector store: RF reads
        uint32_t chunks_plucked = 0; // vector store: chunks moved into the egress FIFO
        uint32_t words_plucked = 0; // stores: words of the current chunk (vector) / lanes (scalar) in egress
        uint32_t acks = 0;
    };

    // (T) Trace phases and per-command stall reasons of an LSU command. Read
    // only for its trace slice; updated in place by the decisions that see
    // the stall (increments and set-once fields, so order cannot matter).
    struct LsuCommandPhases {
        uint64_t admit_cycle = 0;
        std::optional<uint64_t> requests_done_cycle;
        std::optional<uint64_t> first_return_cycle;
        uint64_t request_wait = 0;
        uint64_t rf_wait = 0;
        uint64_t egress_wait = 0;
    };

    // One 32 B slot of the egress FIFO (a BRAM, lsu.egress_sectors slots).
    // Every store client writes here through one shared write port of
    // lsu.pluck_words_per_cycle words, one client per cycle, granted per
    // slot (a started slot is never preempted; TMA wins each new grant).
    // A slot's write request is offered only when all of its words are in
    // and visible (complete_cycle + lsu.egress_latency_cycles); its beats
    // leave at one per DRAM tick; the slot is free after its last beat.
    // A scalar stg lane is one slot holding one word and 4 byte enables.
    struct EgressSlot {
        uint64_t id = 0;            // running slot number (trace / probe)
        GmemClient client = GmemClient::lsu;
        uint64_t command = 0;
        uint32_t index = 0;         // LSU: the command's request index this slot carries
        uint64_t sector_address = 0;
        std::vector<uint8_t> data;
        uint64_t byte_enable = 0;
        uint32_t words_total = 0;   // words the client writes into this slot
        uint32_t words_written = 0;
        uint64_t complete_cycle = 0;
        std::optional<uint64_t> request; // write request accepted by the controller
    };

    struct WriteAck {
        uint64_t dram_tick = 0;
        uint64_t request = 0;
        GmemClient client = GmemClient::lsu;
        uint64_t command = 0;
        std::optional<uint32_t> tag; // scalar scatter: the free-ID tag to return
    };

    enum class TmaCommandState : uint8_t {
        descriptor,
        phase,
        walking,
        data_complete,
        event_presented,
    };

    struct TmaCommand {
        uint64_t id = 0;
        uint32_t warp = 0;
        uint32_t pc = 0;
        Instruction instruction;
        uint32_t descriptor_address = 0;
        std::array<int32_t, 3> origin{};
        TmaCommandState state = TmaCommandState::descriptor;
        TmaDescriptor descriptor;
        uint8_t mbar_phase = 0;
        uint64_t logical_bytes = 0;
        uint64_t useful_bytes = 0;
        uint32_t x = 0;
        uint32_t y = 0;
        uint32_t z = 0;
        uint32_t pending_load_writes = 0;
        uint32_t pending_store_captures = 0;
        bool walker_done = false;
    };

    enum class TmaPortKind : uint8_t {
        descriptor,
        phase,
        load_write,
        store_read,
    };

    // One port-A transaction of the TMA. Load placements read the TMA sector
    // register (or write zeros for an OOB fill); store reads write the TMA
    // store sector register. Fragments carry addresses; only the descriptor
    // and mbar-phase reads hold their data here (TMA frontend latch).
    struct TmaPortRequest {
        struct Piece {
            uint32_t fragment = 0;
            uint8_t offset = 0;
            uint8_t size = 0;
        };
        TmaPortKind kind = TmaPortKind::descriptor;
        uint64_t command = 0;
        uint64_t sector_request = 0;
        bool from_sector_register = false; // load_write: the data is in tma_sector_
        uint32_t next_round = 0;
        std::vector<std::vector<Piece>> rounds;
        std::vector<GmemSectorRequest::TmaFragment> fragments;
    };

    // A TMA sector ready for port-A placement: either the TMA stream's sector
    // register (complete; it stays occupied until its last round) or a
    // synthetic OOB zero fill (no bus data, placement metadata only).
    struct TmaReturn {
        uint64_t command = 0;
        uint64_t sector_request = 0;
        std::vector<GmemSectorRequest::TmaFragment> fragments;
        bool from_sector_register = false;
    };

    struct IssueAttempt {
        bool issued = false;
        WarpState state = WarpState::structural;
        std::string reason;
    };

    // =======================================================================
    // Register contents that change together (one Reg each, below)
    // =======================================================================

    // The RMW engine: the operation in flight and the round-robin bit between
    // the two engines' events (they change together, at an engine claim).
    struct RmwEngine {
        std::optional<ActiveMbarRmw> active;
        // The engine whose event was claimed last (set on every engine claim,
        // contested or not). On a tie the other engine is served. Starts at
        // mma, so the first contested free point serves the TMA.
        MbarEngine last_served = MbarEngine::mma;
    };

    using WaitSlots = std::vector<std::optional<MbarWaitSlot>>;
    using DepbarCounters = std::vector<std::array<uint32_t, 6>>;             // [warp][sb]
    using AccSlots = std::vector<std::array<std::optional<AccTransfer>, 2>>; // [tmem bank][0 load, 1 store]

    // The MMA engine's queue and pipeline. One group: a command's next_step
    // and operands_consumed track the fetch, its outstanding_results the
    // result list. (Its event register is separate.)
    struct MmaPipeline {
        std::deque<MmaCommand> queue;   // mma.queue_commands; issue pushes at the back
        std::optional<MmaFetch> fetch;  // the operand fetch in port-B service
        std::deque<MmaResult> results;  // computed steps waiting for their tmem write
    };

    // The TMA engine's command queue and walker bookkeeping. One group:
    // credits, tickets and each command's pending counts move together. (Its
    // event register, port-A transaction and the two sector registers are
    // separate registers.)
    struct TmaQueue {
        std::map<uint64_t, TmaCommand> commands; // tma.queue_commands; issue inserts
        std::optional<uint64_t> frontend;        // the command the descriptor/phase/walker frontend serves
        std::optional<uint64_t> event_command;   // whose completion event is in the TMA's event register
        uint32_t credits_used = 0;               // of tma.sector_credits
        std::deque<TmaReturn> returns;           // sectors ready for port-A placement
    };

    struct IngressFifo {
        std::deque<IngressBeat> beats;
        uint32_t bytes = 0;
    };

    // =======================================================================
    // Wires: this cycle's decisions. Each is computed once in tick(), in the
    // dependency order written there, from register Q values, memories and
    // earlier wires. The register drivers read only Q and these.
    // =======================================================================

    // A port-A write. shmem takes it at the clock edge: every read of shmem
    // this cycle (port A's own, port B's) sees the start-of-cycle contents.
    // (Port A serves one user per cycle, so it never reads what it writes in
    // the same cycle; a same-word port-A write and port-B read is case 08's
    // collision PANIC.)
    struct ShmemWrite {
        uint32_t address = 0;
        std::vector<uint8_t> bytes;
    };

    struct DepbarEvent {
        uint32_t warp = 0;
        uint8_t bar = 0;
    };

    // LSU: assembly register -> RF (the RF write port at this cycle).
    struct LsuRfDrain {
        std::optional<uint64_t> command;  // the load drained this cycle
        uint32_t assembly = 0;
        uint8_t reg = 0;
        bool last = false;                // its final write: the command completes
    };

    // LSU / TMA -> egress FIFO (the shared egress write port).
    struct EgressWrite {
        std::optional<GmemClient> client;              // who writes the egress FIFO this cycle
        std::optional<EgressSlot> new_slot;            // a new grant opens this slot
        std::vector<std::pair<uint32_t, uint32_t>> words; // (byte offset in the slot, word)
        bool slot_complete = false;                    // the open slot's last word lands now
        uint64_t lsu_command = 0;
        uint32_t lsu_assembly = 0;
        bool lsu_chunk_drained = false;                // a vector store's assembly register empties
        bool tma_store_drained = false;                // the TMA store register's last word leaves
    };

    // The TMA walker's next sector, planned from Q.
    struct TmaWalkerPlan {
        enum class Sector : uint8_t { none, zero_fill, load, store };
        Sector sector = Sector::none;
        TmaCommand command_after;   // the command after this cycle's walk
        std::vector<GmemSectorRequest::TmaFragment> fragments;
        uint64_t sector_address = 0;
        uint64_t store_byte_enable = 0;
    };

    // What the TMA wants this cycle, from Q (independent of the LSU's wants).
    struct TmaRequests {
        std::optional<uint64_t> present;           // the command whose completion event it presents
        std::optional<MbarRmwRequest> present_event; // that event (complete_tx)
        bool port_a_idle = false;                  // port A free for a new TMA transaction
        std::optional<TmaPortRequest> port_a;      // the transaction it starts on idle port A
        bool placement = false;                    // port_a places the front ticket
        std::optional<uint64_t> frontend;          // the frontend command (selection persists)
        bool frontend_reset = false;
        std::optional<TmaWalkerPlan> walker;
        bool write_request = false;                // a TMA store sector waits at the egress request pointer
    };

    // What the LSU wants on the request channel, from Q.
    struct LsuRequestWant {
        std::optional<uint64_t> command;           // its oldest command with requests left, if ready
    };

    // The gmem request channel: one accepted request per core cycle.
    struct RequestChannel {
        std::optional<GmemClient> granted;
        uint64_t lsu_command = 0;                  // the LSU command granted
        bool tma_write = false;                    // the TMA's grant is its store write (else its walker's read)
        std::optional<uint64_t> tma_write_slot;    // the egress slot whose write request is accepted now
        std::optional<uint64_t> lsu_write_slot;
        bool walker_proceeds = false;              // the walker emits its sector this cycle
        // Request ids, assigned in the fixed order TMA write, walker sector, LSU.
        uint64_t tma_write_id = 0;
        uint64_t walker_id = 0;
        uint64_t lsu_id = 0;
        std::optional<uint32_t> lsu_tag;           // the free-ID tag an LSU scalar request takes
        std::optional<GmemSectorRequest> accepted; // the request the controller queue takes
        std::vector<std::pair<uint64_t, ReadRecord>> new_read_records; // reads accepted now
    };

    // shmem port A: the one user this cycle, chosen by select_port_a().
    struct PortAGrant {
        enum class User : uint8_t {
            idle,
            tma_transaction,   // a started TMA transaction: its next round
            simt_access,       // a started SIMT lds/sts: its next round
            mbar_read,         // a SIMT mbar test/try-wait read
            rmw,               // the RMW engine's next stage
            engine_event,      // a held engine event claims the free RMW engine (read stage now)
            tma_start,         // the TMA starts a new transaction (round 0 now)
        };
        User user = User::idle;
        MbarEngine engine = MbarEngine::tma; // engine_event: whose event
    };

    // What port A did this cycle (its data path).
    struct PortAService {
        PortAGrant grant;
        std::vector<ShmemWrite> writes;                   // applied at the clock edge
        std::optional<TmaPortRequest> tma_after;          // the TMA transaction after this round (none: done)
        std::optional<TmaPortRequest> tma_completed;      // the TMA transaction that finished, with its data
        std::vector<std::pair<uint32_t, uint8_t>> store_bytes; // TMA store read: (register offset, byte)
        std::optional<ShmemRequest> simt_after;           // the SIMT access after this round (none: done)
        std::optional<ActiveMbarRmw> rmw_after;           // the RMW engine's operation after this cycle
        bool tma_event_claimed = false;
        bool mma_event_claimed = false;
        std::optional<std::pair<uint32_t, uint64_t>> park; // try-wait: (slot, deadline)
    };

    // LSU: the ingress FIFO read port this cycle.
    struct IngressPluck {
        uint32_t bytes = 0;                               // read from the FIFO head
        struct LaneWrite { uint32_t assembly = 0; uint32_t lane = 0; uint32_t word = 0; };
        std::vector<LaneWrite> lanes;
        std::optional<TmaSectorRegister> tma_sector_after; // the TMA sector register after its fill
        std::optional<TmaReturn> tma_ticket;               // the sector completed: placement ticket
        std::map<uint64_t, std::optional<ReadRecord>> records; // updated (none: erased)
        std::vector<uint32_t> freed_tags;
    };

    // The placeholder controller's DRAM ticks this core cycle.
    struct ControllerTick {
        std::vector<IngressBeat> ingress_beats;          // bus -> ingress FIFO
        uint32_t egress_slots_freed = 0;                 // last beats left the FIFO head
        uint32_t tma_credit_releases = 0;                // TMA write acks
        std::vector<uint32_t> freed_tags;                // scatter write acks
        std::map<uint64_t, uint32_t> lsu_acks;           // command -> write acks
    };

    // LSU: RF -> assembly register (vector stores), below issue in priority.
    struct LsuRfRead {
        std::optional<uint64_t> command;
        uint32_t assembly = 0;
        std::vector<uint32_t> values;
    };

    // Everything an issued instruction hands to other units.
    struct IssueDecision {
        std::optional<DepbarEvent> depbar_admission;
        std::optional<ShmemRequest> port_a;              // lds / sts
        std::optional<ActiveMbarRmw> rmw;                // SIMT mbar RMW: the engine was free
        std::optional<MbarReadRequest> mbar_read;
        std::optional<std::pair<uint32_t, MbarWaitSlot>> wait_slot; // try-wait: (slot, record)
        std::optional<MmaCommand> mma;
        struct AccAdmission { uint8_t bank = 0; bool load = false; AccTransfer transfer; };
        std::optional<AccAdmission> acc;
        struct LsuAdmission {
            LsuCommand command;
            std::optional<std::vector<uint32_t>> store_data; // scalar stg: the issue-cycle RF read
        };
        std::optional<LsuAdmission> lsu;
        std::optional<TmaCommand> tma;
    };

    // All of a cycle's wires, in the order tick() computes them.
    struct Wires {
        // The RF write port (write-first).
        LsuRfDrain lsu_drain;
        std::vector<uint64_t> released_lsu_commands;  // their last write / ack committed now
        // Collected as they happen, from several steps below.
        std::vector<DepbarEvent> depbar_completions;  // in the order they happened
        std::vector<uint32_t> completed_waits;        // try-wait slots completed this cycle
        std::vector<Writeback> new_writebacks;        // RF writes booked now, in wire order
        // The store path's last stage; request generation; the two selects.
        EgressWrite egress;
        TmaRequests tma;
        LsuRequestWant lsu_want;
        RequestChannel channel;
        PortAService port_a;
        // Next state of the units issue looks ahead at (committed by their
        // drivers, with issue's additions).
        MmaPipeline mma_next;
        std::optional<MbarRmwRequest> mma_present;    // a commit presented into the MMA's event register
        AccSlots acc_next;
        IngressPluck ingress;
        ControllerTick controller;
        TmaQueue tma_next;
        bool tma_store_allocated = false;             // the walker took the TMA store register
        // Issue, then what ranks below it.
        IssueDecision issue;
        LsuRfRead lsu_rf_read;
    };

    // =======================================================================
    // State
    // =======================================================================

    Config config_;
    Kernel kernel_;
    uint64_t cycle_ = 0;
    // Shadow state that only decides whether to PANIC. Write-only from here:
    // the rule is at the top of sanitizer.hpp.
    Sanitizer sanitizer_;
    // Every Reg below registers itself here; tick() ends with registers_.commit().
    RegisterFile registers_{sanitizer_, cycle_};

    // ---- Memories (not registers: written through their ports, see tick()) ----
    std::vector<Warp> warps_;                     // + RF and predicate files (write-first)
    std::vector<uint8_t> shmem_;                  // port-A writes land at the clock edge
    std::vector<uint8_t> gmem_;                   // the controller's
    std::vector<std::vector<uint32_t>> tmem_;     // MMA results / acc beats, banks exclusive

    // ---- Registers: the declared storage boundaries ----
    // Scheduler-visible state other units complete.
    Reg<DepbarCounters> depbar_{registers_, "depbar counters"};
    Reg<WaitSlots> mbar_wait_slots_{registers_, "mbar wait slots"};
    // shmem port A, the RMW engine and the engines' event registers. Each
    // engine holds at most one pending event in its own one-entry register
    // (design.md Mbar) and stalls only on its own register being full:
    //   TMA: complete_tx {address, bytes, parity} (TMA data complete, or the
    //        host's complete_mbar_tx).
    //   MMA: a commit's arrive {address}.
    // A presented event is visible to the RMW engine from the next cycle.
    Reg<std::optional<ShmemRequest>> port_a_{registers_, "SIMT port-A access"};
    Reg<std::optional<MbarReadRequest>> mbar_read_{registers_, "SIMT mbar read"};
    Reg<RmwEngine> rmw_{registers_, "RMW engine"};
    Reg<std::optional<MbarRmwRequest>> tma_mbar_event_{registers_, "TMA mbar event register"};
    Reg<std::optional<MbarRmwRequest>> mma_mbar_event_{registers_, "MMA mbar event register"};
    // MMA and accumulator transfers.
    Reg<MmaPipeline> mma_{registers_, "MMA queue and pipeline"};
    Reg<AccSlots> acc_transfers_{registers_, "acc transfer slots"};
    // TMA.
    Reg<TmaQueue> tma_{registers_, "TMA command queue"};
    // A TMA port-A transaction's rounds after its first. Round 0 is served in
    // the grant cycle straight from tma_requests (bypass (h) in tick()).
    Reg<std::optional<TmaPortRequest>> tma_port_a_{registers_, "TMA port-A transaction"};
    Reg<TmaSectorRegister> tma_sector_{registers_, "TMA sector register"};
    Reg<TmaStoreRegister> tma_store_sector_{registers_, "TMA store sector register"};
    // LSU.
    Reg<std::deque<LsuCommand>> lsu_commands_{registers_, "LSU command slots"}; // admission order
    std::deque<Reg<AssemblyRegister>> staging_;   // the lsu.staging_vectors pool, one register each
    Reg<IngressFifo> ingress_{registers_, "ingress FIFO"};
    Reg<std::deque<EgressSlot>> egress_fifo_{registers_, "egress FIFO"};
    Reg<std::map<uint64_t, ReadRecord>> read_records_{registers_, "read ID table"};
    Reg<std::vector<uint8_t>> free_id_tags_{registers_, "free-ID tags"}; // 1 = in flight
    // The RF write port's schedule: bookings that execute at their cycle.
    Reg<std::deque<Writeback>> writebacks_{registers_, "RF writeback schedule"};

    // ---- Unit-internal state (only its own unit's code touches it) ----
    // Issue: scheduler cursor, SFU occupancy, the arithmetic operand latches.
    uint32_t scheduler_cursor_ = 0;
    // First cycle the SFU is free again. The unit is occupied from an SFU
    // op's execution start (window end) for ceil(lanes / sfu_lanes) cycles;
    // a later SFU op stalls structurally until its own execution start fits.
    uint64_t sfu_busy_until_ = 0;
    std::deque<ArithmeticRead> arithmetic_reads_;
    // The RF ports' reservation calendar (the arbiter's schedule). Booked only
    // by the decisions in tick(), in the order of rf_requester (below).
    std::map<uint64_t, std::vector<uint32_t>> write_reservations_;
    // Ring horizon and maximum live operand-latch records equal the fixed window.
    std::vector<std::vector<uint32_t>> read_reservations_;
    // gmem controller (outside the SM): its queue, bus and B channel.
    std::deque<GmemSectorRequest> gmem_queue_;
    std::optional<GmemSectorRequest> gmem_bus_;
    std::deque<WriteAck> pending_write_acks_;
    ControllerTestOrder controller_test_order_ = ControllerTestOrder::fifo;
    // Request channel: TMA/LSU alternate (the client granted last yields a tie).
    GmemClient request_channel_last_ = GmemClient::lsu;
    std::optional<uint32_t> last_served_read_stream_; // interleave_ordered_streams test order
    uint64_t dram_tick_ = 0;
    uint64_t dram_phase_ = 0;
    // Id counters, advanced only by the decisions that allocate (fixed order).
    uint32_t next_vector_stream_ = 0;     // round-robin over 1..read_streams-1
    uint64_t next_egress_slot_id_ = 0;
    uint64_t next_lsu_command_id_ = 0;
    uint64_t next_gmem_request_id_ = 0;
    uint64_t next_tma_command_id_ = 0;
    uint64_t next_mma_command_id_ = 1;
    // (T) trace-only per-command LSU phases.
    std::map<uint64_t, LsuCommandPhases> lsu_phases_;
    bool edge_probe_enabled_ = false;
    std::vector<LsuEdgeEvent> edge_events_;
    std::optional<TraceRecorder> trace_;

    // The driver-shuffle test seam.
    std::optional<std::mt19937_64> shuffle_;

    // =======================================================================
    // tick(): decisions (wires) in dependency order, then drivers, then commit
    // =======================================================================

    // ---- RF write port, write-first ----
    LsuRfDrain decide_lsu_rf_drain();
    void execute_rf_writes(Wires& w);
    // ---- store path ----
    EgressWrite decide_egress_write();
    // ---- request generation (independent; any order) ----
    [[nodiscard]] TmaRequests tma_requests(const EgressWrite& egress) const;
    [[nodiscard]] LsuRequestWant lsu_request_want() const;
    // ---- shared resources: one select each ----
    RequestChannel select_request_channel(const TmaRequests& tma, const LsuRequestWant& lsu);
    void assign_channel_requests(RequestChannel& channel, const TmaRequests& tma);
    void accept_tma_read(RequestChannel& channel, const TmaWalkerPlan& plan);
    void accept_lsu_request(RequestChannel& channel);
    [[nodiscard]] PortAGrant select_port_a(const TmaRequests& tma) const;
    PortAService serve_port_a(const PortAGrant& grant, const TmaRequests& tma, Wires& w);
    // ---- units whose next state issue looks ahead at ----
    void step_mma(Wires& w);
    void step_acc_transfers(Wires& w);
    void decide_mbar_timeouts(Wires& w);
    IngressPluck decide_ingress(const Wires& w);
    void step_gmem_controller(Wires& w);
    void step_tma(Wires& w);
    // ---- issue (the scheduler) ----
    void decide_issue(Wires& w);
    IssueAttempt try_issue(Warp& warp, Wires& w);
    // ---- below issue ----
    LsuRfRead decide_lsu_rf_read(const Wires& w);
    void step_arithmetic_reads(Wires& w);

    // ---- register drivers: next state from Q + wires; any order ----
    using Driver = void (Machine::*)(const Wires&);
    void drive_depbar(const Wires& w);
    void drive_wait_slots(const Wires& w);
    void drive_port_a_access(const Wires& w);
    void drive_mbar_read(const Wires& w);
    void drive_rmw_engine(const Wires& w);
    void drive_tma_event(const Wires& w);
    void drive_mma_event(const Wires& w);
    void drive_mma(const Wires& w);
    void drive_acc_transfers(const Wires& w);
    void drive_tma_queue(const Wires& w);
    void drive_tma_transaction(const Wires& w);
    void drive_tma_sector(const Wires& w);
    void drive_tma_store_sector(const Wires& w);
    void drive_lsu_commands(const Wires& w);
    void drive_assembly_pool(const Wires& w);
    void drive_ingress(const Wires& w);
    void drive_egress(const Wires& w);
    void drive_read_records(const Wires& w);
    void drive_free_id_tags(const Wires& w);
    void drive_writebacks(const Wires& w);
    void run_drivers(const Wires& w);

    // ---- helpers ----
    // Sanitizer argument builders: pure reads of hardware state whose results
    // go only to sanitizer_.
    [[nodiscard]] IssueSite site(const Warp& warp, const Instruction& inst) const;
    [[nodiscard]] static std::vector<uint32_t> access_addresses(
        const std::vector<std::vector<LaneAccess>>& rounds);
    [[nodiscard]] std::vector<uint32_t> lane_values(const Warp& warp, uint8_t reg) const;
    [[nodiscard]] std::optional<std::pair<uint32_t, uint32_t>> mma_operand_span(
        uint32_t base, bool a_operand) const;
    [[nodiscard]] std::array<uint8_t, tma_descriptor_bytes> pending_tma_descriptor_bytes(
        uint32_t address, const Wires& w) const;

    // RF port reservation calendar. Same-cycle requests for the same bank and
    // target cycle are decided by booking order, which is the declared
    // priority: LSU assembly drain, acc transfers, issue, LSU store read.
    enum class RfRequester : uint8_t { lsu_drain, acc_transfer, issue, lsu_store_read };
    [[nodiscard]] bool can_reserve_reads(const std::vector<RfRead>& reads) const;
    void reserve_reads(const std::vector<RfRead>& reads, RfRequester requester);
    [[nodiscard]] bool reserve_write(uint8_t reg, uint64_t at_cycle, RfRequester requester);
    void note_rf_requester(RfRequester requester);
    RfRequester rf_requester_ = RfRequester::lsu_drain;

    // depbar: completions recorded as they happen (issue sees cur - completions).
    void record_depbar_completion(Wires& w, uint32_t warp, uint8_t bar, const std::string& underflow);
    [[nodiscard]] uint32_t depbar_after_completions(const Wires& w, uint32_t warp, uint8_t bar) const;
    void admit_depbar(Wires& w, uint32_t warp, uint8_t bar);
    // mbar wait slots, as issue sees them after this cycle's completions.
    [[nodiscard]] bool wait_slot_completed(const Wires& w, uint32_t slot) const;
    [[nodiscard]] std::optional<uint32_t> waiting_slot_of(const Wires& w, uint32_t warp) const;
    [[nodiscard]] std::optional<uint32_t> free_mbar_wait_slot(const Wires& w) const;
    void complete_mbar_try_wait(Wires& w, uint32_t slot, bool result);

    // Port-A users.
    [[nodiscard]] bool port_a_idle() const;
    [[nodiscard]] bool port_a_busy_next_cycle(const Wires& w) const;
    [[nodiscard]] bool tma_wants_port_a_next_cycle(const Wires& w) const;
    void serve_tma_round(TmaPortRequest request, PortAService& service);
    void serve_simt_round(ShmemRequest request, PortAService& service, Wires& w);
    void serve_mbar_read(PortAService& service, Wires& w);
    void serve_rmw_stage(ActiveMbarRmw active, PortAService& service, Wires& w);
    [[nodiscard]] MbarEngine engine_to_claim(bool tma_held, bool mma_held) const;
    [[nodiscard]] static std::string rmw_label(const ActiveMbarRmw& active);
    void wake_mbar_waiters(Wires& w, uint32_t byte_address, uint8_t new_phase);

    // MMA, acc.
    [[nodiscard]] bool mma_bank_busy(const MmaPipeline& mma, uint8_t bank) const;
    [[nodiscard]] bool acc_slot_available(const Wires& w, uint8_t bank, bool load) const;
    void advance_mma(MmaPipeline& mma, Wires& w);
    void serve_mma_fetch_round(MmaPipeline& mma);
    void complete_mma_fetch(MmaPipeline& mma, Wires& w);
    [[nodiscard]] std::size_t tmem_word_index(
        const Kernel::Accumulator& acc, uint32_t row, uint32_t col) const;
    void execute_arithmetic(ArithmeticRead& pending, Wires& w);

    // TMA.
    void apply_tma_requests(TmaQueue& tma, Wires& w);
    void apply_tma_walker(TmaQueue& tma, Wires& w);
    void complete_tma_port_request(TmaQueue& tma, const TmaPortRequest& request, const Wires& w);
    void mark_tma_data_complete(TmaCommand& command);
    void release_tma_credit(TmaQueue& tma);
    void trace_tma_depths(const TmaQueue& tma, uint64_t at_cycle);
    [[nodiscard]] std::optional<uint64_t> next_tma_frontend_command(const TmaQueue& tma) const;
    [[nodiscard]] std::optional<uint64_t> next_data_complete_tma_command(const TmaQueue& tma) const;
    [[nodiscard]] std::vector<std::vector<TmaPortRequest::Piece>> tma_port_rounds(
        const std::vector<GmemSectorRequest::TmaFragment>& fragments) const;
    [[nodiscard]] TmaWalkerPlan plan_tma_walker(const TmaCommand& command, uint32_t credits_used,
        bool store_register_free) const;
    [[nodiscard]] static bool tma_advance_coordinate(TmaCommand& command);
    [[nodiscard]] bool tma_coordinate_in_bounds(const TmaCommand& command) const;
    [[nodiscard]] uint64_t tma_gmem_address(const TmaCommand& command) const;
    [[nodiscard]] uint32_t tma_shmem_address(const TmaCommand& command) const;
    void validate_tma_runtime_descriptor(const TmaCommand& command) const;

    // Issue.
    [[nodiscard]] Instruction instruction(const Warp& warp) const;
    [[nodiscard]] std::vector<uint8_t> source_registers(const Instruction& inst) const;
    [[nodiscard]] std::optional<uint8_t> destination_register(const Instruction& inst) const;
    void check_register_hazards(const Warp& warp, const Instruction& inst) const;
    [[nodiscard]] uint32_t uniform_reg(
        const Warp& warp, const Instruction& inst, uint8_t reg, const std::string& name) const;
    [[nodiscard]] MbarState load_mbar(uint32_t byte_address) const;
    [[nodiscard]] uint32_t& rf(Warp& warp, uint32_t lane, uint32_t reg);
    [[nodiscard]] uint32_t rf(const Warp& warp, uint32_t lane, uint32_t reg) const;
    // SIMT shmem bank rounds (lds/sts by bank decode; lds.uni: one access).
    [[nodiscard]] std::vector<std::vector<LaneAccess>> build_shmem_rounds(
        const Warp& warp, const Instruction& inst, bool load, const Wires& w) const;
    [[nodiscard]] std::vector<std::vector<LaneAccess>> build_uniform_shmem_round(
        const Warp& warp, const Instruction& inst) const;
    [[nodiscard]] uint32_t shmem_lane_address(const Warp& warp, const Instruction& inst, uint32_t lane) const;
    void panic_if_store_hits_live_mbar(
        const Warp& warp, const Instruction& inst, uint32_t byte_address, const Wires& w) const;
    void panic_if_word_already_stored(const Warp& warp, const Instruction& inst,
        const std::vector<LaneAccess>& bank_lanes, uint32_t byte_address) const;
    void apply_image(
        std::vector<uint8_t>& memory,
        const std::vector<ImageSegment>& image,
        const std::string& name);
    Writeback register_writeback(
        Warp& warp, uint8_t dst, uint64_t ready_cycle, std::vector<uint32_t> values, std::string reason);
    Writeback predicate_writeback(
        Warp& warp, uint8_t dst, uint64_t ready_cycle, std::vector<uint32_t> values);
    [[noreturn]] void panic(const Warp& warp, const Instruction& inst, const std::string& reason) const;
    [[noreturn]] void mbar_rmw_panic(
        const MbarRmwRequest& request, const std::string& reason) const;

    // ---- gmem controller + LSU (src/lsu.cpp) ----
    [[nodiscard]] std::size_t gmem_outstanding_sectors() const;
    [[nodiscard]] std::optional<std::size_t> select_controller_entry() const;
    void tick_gmem_controller(Wires& w, bool& return_ready_low);
    [[nodiscard]] std::optional<uint64_t> egress_slot_request(const EgressSlot& slot, const Wires& w) const;
    void finish_gmem_write(GmemSectorRequest request);
    void deliver_write_acks(Wires& w);
    [[nodiscard]] static bool is_lsu(Opcode opcode);
    [[nodiscard]] static bool is_store(LsuKind kind);
    LsuCommand plan_lsu_command(const Warp& warp, const Instruction& inst) const;
    [[nodiscard]] std::optional<uint32_t> free_assembly_register(const Wires& w) const;
    [[nodiscard]] const LsuCommand* find_lsu_command(uint64_t id) const;
    [[nodiscard]] std::optional<uint32_t> free_id_tag() const;
    [[nodiscard]] const EgressSlot* egress_request_slot(std::optional<uint64_t> requested_now) const;
    [[nodiscard]] bool egress_slot_visible(const EgressSlot& slot) const;
    [[nodiscard]] const LsuCommand* lsu_request_candidate(std::optional<uint64_t> requested_now) const;
    GmemSectorRequest write_request(const EgressSlot& slot, GmemClient client, uint64_t id,
        std::optional<uint32_t> tag);
    GmemSectorRequest lsu_read_request(const LsuCommand& command, uint64_t id, std::optional<uint32_t> tag);
    void pluck_tma(const IngressBeat& beat, uint32_t consumed, uint32_t take, TmaSectorRegister& reg,
        ReadRecord& record, IngressPluck& pluck);
    [[nodiscard]] bool pluck_vector(ReadRecord& record, const IngressBeat& beat, uint32_t consumed,
        uint32_t take, IngressPluck& pluck, const Wires& w);
    void pick_scalar_word(ReadRecord& record, const IngressBeat& beat, uint32_t consumed, uint32_t take,
        IngressPluck& pluck);
    void note_first_return(uint64_t command);
    void probe(LsuEdge edge, uint64_t id, uint64_t aux, uint32_t bytes);
    [[nodiscard]] std::string lsu_phase_detail(const LsuCommand& command) const;
};

uint32_t float_bits(float value);
float bits_float(uint32_t value);
// fp16 <-> fp32 bit conversions. MMA now retains raw halves and uses its
// integer decoder in mma_numeric.cpp. The encode direction (RNE, overflow to inf, subnormal support,
// NaN keeps its truncated payload or the quiet bit) is cvt.pack.f16x2's
// spec. Finite and infinite halves round-trip exactly.
uint32_t half_to_float_bits(uint16_t half);
uint16_t float_to_half_bits(uint32_t bits);

} // namespace nanosm

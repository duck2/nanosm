#pragma once

#include "compiler.hpp"
#include "image.hpp"
#include "mbar.hpp"
#include "profile.hpp"
#include "rf.hpp"
#include "trace.hpp"
#include "tma.hpp"

#include <array>
#include <cstddef>
#include <stdint.h>
#include <deque>
#include <map>
#include <optional>
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
    void set_controller_test_order(ControllerTestOrder order) { controller_test_order_ = order; }
    void enable_edge_probe() { edge_probe_enabled_ = true; }
    [[nodiscard]] const std::vector<LsuEdgeEvent>& edge_events() const { return edge_events_; }
    [[nodiscard]] uint32_t tma_sector_credits_used() const { return tma_sector_credits_used_; }
    [[nodiscard]] std::size_t lsu_commands_in_flight() const { return lsu_commands_.size(); }

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
    void complete_mbar_tx(uint32_t byte_address, uint32_t bytes, uint8_t parity);

private:
    struct MbarWaitSlot {
        uint32_t warp = 0;
        uint32_t byte_address = 0;
        uint8_t parity = 0;
        uint8_t predicate_dst = 0;
        bool parked = false;
        uint64_t deadline_cycle = 0;
    };

    struct Warp {
        uint32_t id = 0;
        uint32_t pc = 0; // byte PC; every decoded instruction still occupies one 64-bit slot
        bool halted = false;
        uint64_t stall_until = 0;
        std::array<uint32_t, 6> depbar{};
        std::vector<uint32_t> rf;
        std::vector<uint64_t> ready_cycle;
        // Sanitizer metadata, not a runtime dependency scoreboard. Exclusive end.
        std::vector<uint64_t> read_until;
        // Async acc.st readers; a younger write is illegal until all stores finish.
        std::vector<uint32_t> async_readers;
        std::array<std::vector<uint8_t>, 7> predicates;
        std::array<uint64_t, 7> predicate_ready_cycle{};
        std::optional<uint32_t> mbar_wait_slot;
    };

    struct Writeback {
        uint64_t ready_cycle = 0;
        uint32_t warp = 0;
        std::optional<uint8_t> dst;
        std::optional<uint8_t> predicate_dst;
        std::vector<uint32_t> values;
        std::optional<uint8_t> clear_depbar;
        // Sanitizer only: the final acc.ld beat releases its tmem read-ownership.
        std::optional<uint64_t> release_acc_transfer;
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
        uint32_t warp = 0;
        uint8_t dst = 0;
        uint8_t depbar = 0;
        std::vector<std::vector<LaneAccess>> rounds;
        std::size_t next_round = 0;
        std::vector<uint32_t> load_values;
    };

    struct AccTransfer {
        uint64_t id = 0; // sanitizer ownership key only
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
        bool event_presented = false;
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

    struct ActiveMbarRmw {
        MbarRmwRequest request;
        MbarState state;
        bool write_stage = false;
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
    // cycle. Never fall-through.
    struct IngressBeat {
        uint64_t request = 0;
        uint32_t offset = 0;        // byte offset of this beat in its sector
        std::vector<uint8_t> bytes;
        uint32_t consumed = 0;      // bytes already plucked
        uint64_t visible_cycle = 0;
    };

    // The TMA stream's 32 B sector register (flops). Filled from the ingress
    // FIFO at the pluck rate; when complete, placed into shmem by port A in
    // its bank rounds, which read the register directly. Free after the last
    // round. last_write_cycle is the fall-through tripwire: a consumer may
    // only read it in a later cycle.
    struct TmaSectorRegister {
        std::optional<uint64_t> request;
        uint64_t sector_address = 0;
        std::vector<uint8_t> data;
        uint32_t filled = 0;        // bytes plucked in
        bool complete = false;      // all bytes in; its ticket is in tma_returns_
        uint64_t last_write_cycle = 0;
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
        uint64_t last_write_cycle = 0;
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
        uint64_t last_write_cycle = 0;   // fall-through tripwire
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
        bool finishing = false;     // completion writeback scheduled
        // Trace phases and per-command stall reasons.
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

    // ---- Race sanitizer (pure bookkeeping; trash/shmem-ownership-sanitizer-2026-10-03.md) ----
    // There is no ownership hardware. An outstanding engine command is modeled
    // as owning shmem byte ranges from its accepted issue until an existing
    // completion event; a conflicting access PANICS naming the owner and the
    // wait that would have ordered it. Nothing here is consulted by admission,
    // port arbitration or data movement.
    enum class ShmemOwnerKind : uint8_t {
        tma_load,       // write-owner of the destination box until its last shmem write commits
        tma_store,      // read-owner of the source box until its last source read
        tma_descriptor, // read-owner of the 64-byte descriptor until the engine reads it
        mma_a,          // read-owner of the A operand until the final operand read
        mma_b,          // read-owner of the B operand until the final operand read
    };

    struct ShmemOwner {
        ShmemOwnerKind kind = ShmemOwnerKind::tma_load;
        uint64_t command = 0; // tma or mma command id
        bool write = false;
        uint32_t begin = 0;   // byte range [begin, end)
        uint32_t end = 0;
        uint32_t warp = 0;
        uint32_t pc = 0;
        uint64_t issue_cycle = 0;
        std::optional<uint32_t> completion_mbar; // tma; unknown if the descriptor is invalid
        uint8_t depbar = 0;                      // mma operand-consumed depbar
    };

    enum class TmemOwnerKind : uint8_t {
        mma,    // write-owner (read-modify-write) of the whole accumulator until its last result retires
        acc_ld, // read-owner of its rows x columns until its final RF write commits
        acc_st, // write-owner of its rows x columns until its final tmem write
    };

    struct TmemOwner {
        TmemOwnerKind kind = TmemOwnerKind::mma;
        uint64_t id = 0; // mma command id or acc transfer id
        uint8_t accumulator = 0;
        uint32_t row_begin = 0;
        uint32_t row_end = 0;
        uint32_t col_begin = 0;
        uint32_t col_end = 0;
        uint32_t warp = 0;
        uint32_t pc = 0;
        uint64_t issue_cycle = 0;
        uint8_t depbar = 0;
    };

    Config config_;
    Kernel kernel_;
    std::vector<Warp> warps_;
    std::vector<uint8_t> shmem_;
    std::vector<uint8_t> gmem_;
    std::vector<std::vector<uint32_t>> tmem_;
    // Slots are [coarse bank][0=load, 1=store]. In whole-transfer mode either
    // slot blocks the other; duplex mode permits one of each.
    std::vector<std::array<std::optional<AccTransfer>, 2>> acc_transfers_;
    std::deque<MmaCommand> mma_commands_;
    std::deque<MmaResult> mma_results_;
    std::optional<MmaFetch> mma_fetch_;
    uint64_t next_mma_command_id_ = 1;
    // First cycle the SFU is free again. The unit is occupied from an SFU
    // op's execution start (window end) for ceil(lanes / sfu_lanes) cycles;
    // a later SFU op stalls structurally until its own execution start fits.
    uint64_t sfu_busy_until_ = 0;
    // Shmem words written through port A this cycle (sts rounds, TMA load
    // returns). A same-cycle same-word MMA port-B read is UB and panics.
    std::vector<uint32_t> port_a_written_words_;
    std::optional<ShmemRequest> port_a_;
    std::optional<TmaPortRequest> tma_port_a_;
    // No RMW input FIFO: the engine serves one operation, SIMT requesters stall
    // structurally at issue, and each engine holds at most one pending event.
    std::optional<MbarRmwRequest> engine_mbar_event_;
    std::optional<ActiveMbarRmw> mbar_rmw_;
    std::optional<MbarReadRequest> mbar_read_;
    std::vector<std::optional<MbarWaitSlot>> mbar_wait_slots_;
    std::deque<GmemSectorRequest> gmem_queue_;
    std::optional<GmemSectorRequest> gmem_bus_;
    ControllerTestOrder controller_test_order_ = ControllerTestOrder::fifo;
    // Request channel: one accepted request per core cycle, TMA/LSU alternate.
    bool request_channel_used_ = false;
    GmemClient request_channel_last_ = GmemClient::lsu;
    std::deque<WriteAck> pending_write_acks_;
    // SM side of the boundary (LSU state).
    std::map<uint64_t, ReadRecord> read_records_;
    std::vector<std::deque<uint64_t>> ordered_stream_requests_; // request order per ordered ID
    uint32_t next_vector_stream_ = 0;     // round-robin over 1..read_streams-1
    std::optional<uint32_t> last_served_read_stream_; // interleave_ordered_streams test order
    std::deque<IngressBeat> ingress_fifo_;
    uint32_t ingress_bytes_ = 0;
    // Bytes read out of the ingress FIFO this cycle: room they free is seen by
    // the write side only next cycle (the full flag comes from the pointers
    // as registered at the start of the cycle).
    uint32_t ingress_freed_this_cycle_ = 0;
    bool return_ready_low_this_cycle_ = false;
    TmaSectorRegister tma_sector_;
    TmaStoreRegister tma_store_sector_;
    std::vector<AssemblyRegister> staging_;
    std::deque<LsuCommand> lsu_commands_; // admission order
    std::deque<EgressSlot> egress_fifo_;
    uint64_t next_egress_slot_id_ = 0;
    std::vector<uint8_t> free_id_tags_;   // lsu.sectors_in_flight; 1 = in flight
    uint64_t next_lsu_command_id_ = 0;
    bool edge_probe_enabled_ = false;
    std::vector<LsuEdgeEvent> edge_events_;
    std::map<uint64_t, TmaCommand> tma_commands_;
    std::deque<TmaReturn> tma_returns_;
    std::optional<uint64_t> tma_frontend_command_;
    std::optional<uint64_t> tma_event_command_;
    uint32_t tma_sector_credits_used_ = 0;
    std::deque<Writeback> writebacks_;
    std::map<uint64_t, std::vector<uint32_t>> write_reservations_;
    // Ring horizon and maximum live operand-latch records equal the fixed window.
    std::vector<std::vector<uint32_t>> read_reservations_;
    std::deque<ArithmeticRead> arithmetic_reads_;
    std::optional<TraceRecorder> trace_;
    uint64_t cycle_ = 0;
    uint64_t dram_tick_ = 0;
    uint64_t dram_phase_ = 0;
    uint32_t scheduler_cursor_ = 0;
    uint64_t next_gmem_request_id_ = 0;
    uint64_t next_tma_command_id_ = 0;
    // Sanitizer state. n is tiny (a few outstanding commands); checks are a
    // linear scan, chosen for clarity over speed.
    std::vector<ShmemOwner> shmem_owners_;
    std::vector<TmemOwner> tmem_owners_;
    uint64_t next_acc_transfer_id_ = 0;

    // Sanitizer helpers. Every check either returns or panics; none feeds a
    // timing or admission decision.
    [[nodiscard]] const ShmemOwner* shmem_owner_overlapping(
        uint64_t begin, uint64_t end, bool writers_only) const;
    [[nodiscard]] std::string describe_shmem_owner(const ShmemOwner& owner) const;
    [[nodiscard]] std::string shmem_race(
        const std::string& access, uint64_t begin, uint64_t end, const ShmemOwner& owner) const;
    void check_simt_shmem_access(
        const Warp& warp, const Instruction& inst,
        const std::vector<std::vector<LaneAccess>>& rounds, bool load) const;
    [[nodiscard]] std::optional<std::pair<uint32_t, uint32_t>> mma_operand_span(
        uint32_t base, bool a_operand) const;
    [[nodiscard]] std::optional<TmaDescriptor> peek_tma_descriptor(uint32_t address) const;
    void sanitize_tma_accept(const Warp& warp, const Instruction& inst, const TmaCommand& command);
    void sanitize_mma_accept(const Warp& warp, const Instruction& inst, const MmaCommand& command);
    void verify_tma_descriptor_ownership(const TmaCommand& command);
    void mark_tma_data_complete(TmaCommand& command);
    void release_mma_operands(uint64_t command);
    [[nodiscard]] static bool tmem_overlap(const TmemOwner& a, const TmemOwner& b);
    [[nodiscard]] std::string describe_tmem_owner(const TmemOwner& owner) const;
    void release_tmem_owner(TmemOwnerKind kind, uint64_t id);

    void commit_writebacks();
    void advance_rf_reads();
    void advance_mma();
    void serve_mma_fetch_round();
    void complete_mma_fetch();
    void record_port_a_shmem_write(uint64_t byte_address, uint32_t size);
    [[nodiscard]] bool mma_bank_busy(uint8_t bank) const;
    void execute_arithmetic(ArithmeticRead& pending);
    void advance_tma();
    void advance_shmem();
    void advance_acc_transfers();
    void advance_mbar_rmw();
    void advance_mbar_read();
    void advance_mbar_wait_timeouts();
    void advance_dram_clock();
    void tick_gmem_controller();
    void issue_one();
    IssueAttempt try_issue(Warp& warp);

    [[nodiscard]] Instruction instruction(const Warp& warp) const;
    [[nodiscard]] std::vector<uint8_t> source_registers(const Instruction& inst) const;
    [[nodiscard]] std::optional<uint8_t> destination_register(const Instruction& inst) const;
    void check_register_hazards(const Warp& warp, const Instruction& inst) const;
    [[nodiscard]] bool can_reserve_reads(const std::vector<RfRead>& reads) const;
    void reserve_reads(const std::vector<RfRead>& reads);
    [[nodiscard]] bool reserve_write(uint8_t reg, uint64_t at_cycle);
    [[nodiscard]] bool port_a_busy() const;
    [[nodiscard]] bool acc_slot_available(uint8_t bank, bool load) const;
    [[nodiscard]] std::size_t tmem_word_index(
        const Kernel::Accumulator& acc, uint32_t row, uint32_t col) const;
    [[nodiscard]] uint32_t uniform_reg(
        const Warp& warp, const Instruction& inst, uint8_t reg, const std::string& name) const;
    [[nodiscard]] MbarState load_mbar(uint32_t byte_address) const;
    void store_mbar(uint32_t byte_address, const MbarState& state);
    void wake_mbar_waiters(uint32_t byte_address, uint8_t new_phase);
    [[nodiscard]] std::optional<uint32_t> free_mbar_wait_slot() const;
    void complete_mbar_try_wait(uint32_t slot, bool result);

    [[nodiscard]] uint32_t& rf(Warp& warp, uint32_t lane, uint32_t reg);
    [[nodiscard]] uint32_t rf(const Warp& warp, uint32_t lane, uint32_t reg) const;
    [[nodiscard]] std::vector<std::vector<LaneAccess>> build_shmem_rounds(
        const Warp& warp, const Instruction& inst, bool load) const;
    [[nodiscard]] std::size_t gmem_outstanding_sectors() const;
    void complete_tma_port_request(TmaPortRequest request);

    // ---- gmem controller + LSU (src/lsu.cpp) ----
    [[nodiscard]] bool controller_request_ready() const; // queue has room
    [[nodiscard]] bool request_channel_grants(GmemClient client) const;
    void accept_gmem_request(GmemSectorRequest request, GmemClient client);
    [[nodiscard]] std::optional<std::size_t> select_controller_entry() const;
    void finish_gmem_write(GmemSectorRequest request);
    void deliver_write_acks();
    [[nodiscard]] bool tma_wants_port_a() const;
    [[nodiscard]] static bool is_lsu(Opcode opcode);
    LsuCommand plan_lsu_command(const Warp& warp, const Instruction& inst) const;
    [[nodiscard]] std::optional<uint32_t> free_assembly_register() const;
    void admit_lsu_command(Warp& warp, const Instruction& inst, LsuCommand command, uint32_t assembly);
    [[nodiscard]] const LsuCommand* lsu_request_candidate() const;
    [[nodiscard]] LsuCommand& lsu_command(uint64_t id);
    [[nodiscard]] std::optional<uint32_t> free_id_tag() const;
    [[nodiscard]] const EgressSlot* egress_request_slot() const;
    [[nodiscard]] bool egress_slot_visible(const EgressSlot& slot) const;
    void accept_write_request(EgressSlot& slot, GmemClient client, std::optional<uint32_t> tag);
    void advance_tma_write_requests();
    void advance_lsu_requests();
    void advance_ingress();
    void check_ordered_beat(uint32_t stream, const IngressBeat& beat);
    [[nodiscard]] bool pluck_tma(IngressBeat& beat, uint32_t take);
    [[nodiscard]] bool pluck_vector(ReadRecord& record, IngressBeat& beat, uint32_t take);
    void pick_scalar_word(ReadRecord& record, const IngressBeat& beat, uint32_t take);
    void note_first_return(uint64_t command);
    void advance_lsu_rf_writes();
    void advance_egress_writes();
    void advance_lsu_rf_reads();
    void probe(LsuEdge edge, uint64_t id, uint64_t aux, uint32_t bytes);
    void finish_lsu_command(LsuCommand& command);
    void release_lsu_command(uint64_t id);
    [[nodiscard]] std::string lsu_phase_detail(const LsuCommand& command) const;
    [[noreturn]] void controller_contract_panic(const std::string& reason) const;
    void advance_tma_frontend();
    void present_tma_mbar_event();
    [[nodiscard]] std::optional<uint64_t> next_tma_frontend_command() const;
    [[nodiscard]] std::optional<uint64_t> next_data_complete_tma_command() const;
    [[nodiscard]] std::vector<std::vector<TmaPortRequest::Piece>> tma_port_rounds(
        const std::vector<GmemSectorRequest::TmaFragment>& fragments) const;
    void serve_tma_port_round(TmaPortRequest& request);
    [[nodiscard]] bool tma_walker_step(TmaCommand& command);
    [[nodiscard]] bool tma_advance_coordinate(TmaCommand& command);
    [[nodiscard]] bool tma_coordinate_in_bounds(const TmaCommand& command) const;
    [[nodiscard]] uint64_t tma_gmem_address(const TmaCommand& command) const;
    [[nodiscard]] uint32_t tma_shmem_address(const TmaCommand& command) const;
    void validate_tma_runtime_descriptor(const TmaCommand& command) const;
    void release_tma_credit();
    void apply_image(
        std::vector<uint8_t>& memory,
        const std::vector<ImageSegment>& image,
        const std::string& name);

    void schedule_register_write(
        Warp& warp,
        uint8_t dst,
        uint64_t ready_cycle,
        std::vector<uint32_t> values,
        std::string reason);
    void schedule_predicate_write(
        Warp& warp,
        uint8_t dst,
        uint64_t ready_cycle,
        std::vector<uint32_t> values);
    [[noreturn]] void panic(const Warp& warp, const Instruction& inst, const std::string& reason) const;
    [[noreturn]] void mbar_rmw_panic(
        const MbarRmwRequest& request, const std::string& reason) const;
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

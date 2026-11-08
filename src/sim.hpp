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

#include <algorithm>
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
    tma_place_round,   // TMA sector register -> shmem (port-B write round, at its grant). id = request, aux = round
    rf_read_beat,      // RF -> assembly (stores). id = command, aux = register
    tma_store_round,   // shmem -> TMA store sector register (A-read round, at its return). id = tma command, aux = round
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

// ===========================================================================
// Admission (trash/rfc10.md section 1). Whether an instruction issues this
// cycle is decided from its metadata and resource state only -- never from
// register, predicate or memory contents. depbar_wait and structural_refusal
// are free functions of the decoded instruction, its IssuePlan (metadata and
// profile) and a ResourceView (flags); no argument can hold an operand value.
// Operands are captured after acceptance (Machine::accept).
// ===========================================================================

// What an instruction books if it is accepted, from metadata and the profile.
struct IssuePlan {
    std::vector<uint8_t> reads;         // its one RF read round: each register sampled at issue + issue_to_sample (src/rf.hpp)
    std::optional<uint8_t> write_reg;   // a fixed RF-write booking (arithmetic with an RF destination)
    uint64_t write_cycle = 0;           // the cycle that result enters its bank's write port
};

// Resource state as issue sees it this cycle: Q, plus what this cycle's
// decisions free (look-ahead (c) in Machine::tick).
struct ResourceView {
    bool read_inhibited = false;     // the RF read decision inhibits new SIMT reads of a bank the round uses
    bool sfu_free = false;           // the SFU is free by this op's execution start
    bool write_slot_free = false;    // the write booking's (bank, cycle) has a free port
    bool write_inhibited = false;    // the RF write decision inhibits new bookings to that bank
    bool wait_slot_free = false;     // an mbar wait slot
    bool tma_queue_room = false;
    bool mma_queue_room = false;
    bool acc_engine_free = false;    // the acc.ld (load) or acc.st (store) engine takes this transfer
    // shmem (trash/rfc12.md sections 3, 4 and 6), as this cycle's grants leave it:
    bool no_active_lanes = false;    // the warp's active mask is empty: its lds / sts is absent (no slot)
    bool unserviced_read = false;    // the warp holds a load slot with rounds not yet granted
    bool unserviced_write = false;   // ... a store slot, or a SIMT RMW (mbar op or atomic) whose write is not yet granted
    bool own_shmem_at_ram = false;   // fence.sh: every shmem access of the warp has reached the RAM by now
    bool load_slot_free = false;     // an lds's slot, or an atomic's return slot
    bool store_slot_free = false;
    bool probe_register_free = false;
    bool simt_rmw_register_free = false; // the SIMT RMW register is empty after this cycle
    bool lsu_slot_free = false;
    bool assembly_free = false;
};

enum class Refusal : uint8_t {
    rf_read_inhibited,
    sfu_busy,
    rf_write_booked,
    rf_write_inhibited,
    wait_slots_full,
    tma_queue_full,
    mma_queue_full,
    acc_engine_busy,
    shmem_ordered,
    shmem_fence,
    load_slots_full,
    store_slots_full,
    probe_register_busy,
    simt_rmw_register_busy,
    lsu_slot,
    lsu_assembly,
};

// The first depbar in the instruction's wait mask that is still nonzero.
std::optional<uint8_t> depbar_wait(const Instruction& inst, const std::array<uint32_t, 6>& depbar);
// The first structural reason the instruction cannot be accepted now.
std::optional<Refusal> structural_refusal(const Instruction& inst, const IssuePlan& plan, const ResourceView& view);

// One RF bank's choice among ready async writers (rfc10 section 3 (b)):
// scan from `cursor`, skip holders not ready for this bank, grant up to
// `ports`; the cursor moves past the last holder granted, and only then.
struct RoundRobinGrant {
    std::vector<uint32_t> granted;  // holder indices, in grant order
    uint32_t cursor_after = 0;
};
RoundRobinGrant round_robin_grant(const std::vector<uint8_t>& ready, uint32_t cursor, uint32_t ports);

// The shmem byte address the MMA reads operand element (outer, k) of a
// command of `shape` from, its operand starting at `base` (outer: m for A, n
// for B): the operand fetch's own addressing (src/sim.cpp mma_a_address /
// mma_b_address), named here so tests can hold a TMA box against it.
uint32_t mma_operand_address(const Config& config, const MmaShape& shape, uint32_t base, bool a_operand,
    uint32_t outer, uint32_t k);

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
    // Test seams (Q, between ticks): whether load slot `index` is held, and
    // whether its gathered vector is complete (a ready async RF writer);
    // whether store slot `index` is held.
    [[nodiscard]] bool load_slot_held(uint32_t index) const { return load_slots_.at(index).cur().has_value(); }
    [[nodiscard]] bool load_slot_complete(uint32_t index) const {
        const auto& slot = load_slots_.at(index).cur();
        return slot.has_value() && slot->complete();
    }
    [[nodiscard]] bool store_slot_held(uint32_t index) const { return store_slots_.at(index).cur().has_value(); }
    // SHMEM-TEST-ONLY seam (between ticks): warp `warp`'s active-lane mask.
    // It exists so the shmem tests can build divergent accesses before
    // divergence exists. Read ONLY by the shmem path: lds / sts build their
    // bank rounds from the active lanes (an lds writes only those lanes),
    // and lds.uni / atom.*.uni write only their active destination lanes.
    // Every uniform operation under a partial mask is the SANITIZER's PANIC
    // (backlog D12.1). Arithmetic, global memory, TMA, MMA, acc transfers,
    // branches and everything else ignore it and run all lanes (backlog S10:
    // making every instruction honor it is divergence work). It is NOT divergence:
    // real divergence (design.md's pre-Volta reconvergence stack, reserved) is
    // future work, and this mask is only a stand-in for that stack's top
    // entry. Never used by launches or kernels.
    void test_set_active_mask(uint32_t warp, uint64_t mask);
    // Test seams (Q, between ticks): a warp's front end -- its FIFO's valid
    // packets, its pc (the next instruction's) and fetch pc (the next packet
    // to read) -- and the warp whose packet is at the instruction memory's
    // output now (a read in flight), if any.
    [[nodiscard]] std::size_t instruction_fifo_packets(uint32_t warp) const { return front_end_.cur().at(warp).fifo.size(); }
    [[nodiscard]] uint32_t warp_pc(uint32_t warp) const { return front_end_.cur().at(warp).pc; }
    [[nodiscard]] uint32_t warp_fetch_pc(uint32_t warp) const { return front_end_.cur().at(warp).fetch_pc; }
    [[nodiscard]] std::optional<uint32_t> fetch_in_flight() const {
        const auto& read = fetch_read_.cur();
        return read.has_value() ? std::optional<uint32_t>(read->warp) : std::nullopt;
    }
    // Test seam (between ticks): overwrite a warp's fetch pc, to show that the
    // front end's consistency check (step_front_end) catches a broken state.
    void test_set_fetch_pc(uint32_t warp, uint32_t pc);
    // Test seams (Q, between ticks): the RF write decision's state per bank.
    [[nodiscard]] bool rf_write_owed(uint32_t bank) const { return rf_write_policy_.cur().owed.at(bank) != 0; }
    [[nodiscard]] uint32_t rf_write_cursor(uint32_t bank) const { return rf_write_policy_.cur().cursor.at(bank); }
    [[nodiscard]] uint32_t rf_write_lsu_cursor() const { return rf_write_policy_.cur().lsu_cursor; }
    // Test seams (Q, between ticks): the RF read decision's state per bank,
    // and the issue register's bank use (reads its round samples this cycle).
    [[nodiscard]] bool rf_read_owed(uint32_t bank) const { return rf_read_policy_.cur().owed.at(bank) != 0; }
    [[nodiscard]] uint32_t rf_read_cursor(uint32_t bank) const { return rf_read_policy_.cur().cursor.at(bank); }
    [[nodiscard]] uint32_t issue_read_bank_use(uint32_t bank) const { return issue_read_banks_.cur().at(bank); }
    // Test seam (Q, between ticks): stage `stage` of the acc.ld engine's
    // pipeline (tmem.port_depth stages to the tmem sample, then
    // tmem.read_latency_cycles to the output, its last stage): (destination
    // register, lane values -- empty until the beat's tmem sample -- last
    // beat), or none when the stage is empty.
    struct AccLdStageView {
        uint8_t reg = 0;
        std::vector<uint32_t> values;
        bool last = false;
    };
    [[nodiscard]] std::optional<AccLdStageView> acc_ld_stage(uint32_t stage) const {
        const auto& beat = acc_ld_pipe_.cur().at(stage);
        if (!beat.has_value()) return std::nullopt;
        return AccLdStageView{beat->reg, beat->values, beat->last};
    }
    [[nodiscard]] uint32_t acc_ld_pipeline_stages() const {
        return config_.tmem_port_depth + config_.tmem_read_latency_cycles;
    }
    // Test seam (between ticks): tmem word `word` (word w is in fine bank
    // w % tmem.fine_banks).
    [[nodiscard]] uint32_t tmem_word(uint32_t word) const { return tmem_.at(word); }

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
    // wait slots, which other units complete, are registers (below), and so
    // is the warp's front end (its PCs, instruction FIFO and halted bit,
    // WarpFrontEnd), which fetch reads.
    struct Warp {
        uint32_t id = 0;
        uint64_t active_mask = 0; // all lanes; shmem-test-only (see test_set_active_mask)
        // The stall count's end: the warp issues again from this cycle. Only
        // the scheduler reads it (before it accepts anything) and writes it
        // (at its acceptance, for the warp it accepted), so the in-place write
        // is the register's next value; nothing else -- fetch included --
        // reads it.
        uint64_t stall_until = 0;
        std::vector<uint32_t> rf;
        // RAW/WAW/WAR horizons (no scoreboard exists) live in the Sanitizer.
        std::array<std::vector<uint8_t>, 7> predicates;
    };

    // One warp slot's front end (trash/front-end-packets-2026-10-18.md), one
    // of core.warps. Its instruction FIFO is a shift FIFO of
    // core.instruction_buffer_packets fetch packets: a packet is the two
    // 64-bit words at an aligned 16-byte address, read together. Only valid
    // entries are held, head first (the valid bits are a thermometer,
    // fifo.size() is the occupancy). The warp's next instruction is the head
    // packet's word that pc names: pc bit 3 selects the half, so no other
    // per-warp state says which half is next (a branch into a packet's second
    // half starts there; its first half is never used). Issuing the head's
    // second half pops the packet; a landing fetch appends; both in one
    // next-state update, with the accepted word captured by the issue
    // register at that same edge. Consecutive entries are consecutive
    // packets: fetch is sequential and a taken branch clears the FIFO. Byte
    // PCs (32 bits; the instruction memory's address width in hardware).
    // While the warp runs, fetch_pc = pc rounded down to 16 + 16 x (valid
    // packets + its read in flight): step_front_end checks it every cycle.
    using FetchPacket = std::array<uint64_t, 2>;
    struct WarpFrontEnd {
        std::vector<FetchPacket> fifo; // fetched packets, head first
        uint32_t pc = 0;               // the warp's next instruction (in the head packet when it is valid)
        uint32_t fetch_pc = 0;         // the next packet to read (a multiple of 16)
        bool halted = false;           // a halt was accepted (or the slot was not launched): no fetch, no issue
        bool operator==(const WarpFrontEnd&) const = default;
    };

    // The instruction memory's read in flight (instruction_read_cycles = 1,
    // src/rf.hpp): the packet address it sampled at the grant's edge, and the
    // tag beside it -- the warp the packet is for (valid + 4 bits at 16
    // warps). The packet is at the memory output (128 bits) in the cycle this
    // register holds it. No stage stalls: the credit check guarantees its
    // FIFO has room.
    struct FetchRead {
        uint32_t warp = 0;
        uint32_t pc = 0; // the address sampled (32-bit byte pc)
        bool operator==(const FetchRead&) const = default;
    };

    // The issue register's control part: a branch accepted last cycle. It
    // resolves now (src/rf.hpp issue_to_branch_resolve): a conditional
    // branch's predicate is read, and the one target adder adds the branch's
    // pc + 8 and its offset. Its warp neither issues nor fetches meanwhile.
    struct IssuedBranch {
        uint32_t warp = 0;
        uint32_t pc = 0;         // the branch's own pc (32 bits)
        Instruction instruction; // opcode, predicate, invert, offset
    };

    // One write through the RF write port (src/rf.hpp): PRESENTED at
    // port_cycle -- a fixed result's booked cycle, an async holder's grant --
    // and at the RAM at land_cycle, where it lands at the clock edge: port +
    // Wd for an RF register, the port cycle itself for the predicate file
    // (flip-flops). A completion with no destination (a store's last ack)
    // happens at its port_cycle == land_cycle.
    struct Writeback {
        uint64_t port_cycle = 0;
        uint64_t land_cycle = 0;
        uint32_t warp = 0;
        std::optional<uint8_t> dst;
        std::optional<uint8_t> predicate_dst;
        std::vector<uint32_t> values;
        std::optional<uint64_t> lane_enables; // an lds under a partial mask writes its active lanes only
        // Cleared as the write lands: the waiting warp's counter is 0 from
        // its visibility (land + 1) on.
        std::optional<uint8_t> clear_depbar;
        // The final LSU RF write / last-ack completion ends its command: the
        // command slot and its assembly register are released at the port
        // (from the grant on, the write pipeline owns the data).
        std::optional<uint64_t> release_lsu_command;
        // An async result's registers: the sanitizer marks them visible as
        // the write lands (a fixed result's visibility is known at issue).
        std::vector<uint8_t> visible_regs;
        std::string reason;
    };

    // A fixed-latency instruction in the arithmetic pipeline, from accepted
    // issue until its operands are all at the RF output (issue +
    // issue_to_operands), when it executes: its operand latches.
    struct ArithmeticRead {
        uint64_t issue_cycle = 0;
        uint32_t warp = 0;
        uint32_t pc = 0;                             // (S) PANIC text: the instruction's own pc
        Instruction instruction;
        std::vector<uint8_t> reads;                  // the round's registers
        std::vector<std::vector<uint32_t>> operands; // by read, latched at the sample
    };

    // A memory / engine / mbar instruction between accepted issue and its
    // unit (trash/rfc10.md section 2, slice I6a): the issue register and the
    // RF read pipeline carry it. Its operands are latched at its RF sample
    // (issue + 1) and reach the unit at DISPATCH (issue + 1 + Lr), where the
    // unit takes it (hand_over) exactly as issue used to. Admission secured
    // the unit's space when it accepted the instruction: every later
    // admission counts the entries here against that space (a credit counter
    // per resource in hardware; dispatch_claims). The pc and the active mask
    // are the issue cycle's.
    struct DispatchEntry {
        uint64_t issue_cycle = 0;
        uint32_t warp = 0;
        uint32_t pc = 0;
        uint64_t active_mask = 0;
        Instruction instruction;
        std::vector<uint8_t> reads;                  // the round's registers
        std::vector<std::vector<uint32_t>> operands; // by read, latched at the sample
    };

    struct LaneAccess {
        uint32_t lane = 0;
        uint8_t word = 0;
        uint64_t byte_address = 0;
        uint32_t value = 0;
    };

    // A SIMT shmem access's bank rounds (trash/rfc12.md section 3): its ACTIVE
    // lanes only, by bank decode (round r serves every bank's r-th active
    // lane, in lane order), and how many rounds A-read / A-write has granted.
    // The lanes' addresses (and a store's data) are the slot's address
    // registers and 64 B data register.
    struct SlotRounds {
        std::vector<std::vector<LaneAccess>> rounds;
        std::size_t granted = 0;
        [[nodiscard]] bool started() const { return granted != 0; }
        [[nodiscard]] bool serviced() const { return granted == rounds.size(); }
    };

    // A load slot (rfc12 section 3): one lds / lds.uni from admission until
    // the RF write decision grants its write. Secured at admission (a claim
    // in the dispatch pipeline, I6a), installed at dispatch with the captured
    // operands, so it is ready for A-read from the next cycle.
    // Serviced when its last round is granted; complete when every round's
    // words are back in `words`, the 64 B gather register -- then it is a
    // ready async RF writer.
    // Or a uniform atomic's RETURN slot (trash/rfc13.md section 5), reserved
    // at the atomic's admission: no rounds of its own (never an A-read
    // requester, never an unserviced read); the RMW engine fills it once, with
    // the word's old value for every lane, and then it waits for the RF like
    // any lds.uni.
    struct LoadSlot {
        uint32_t warp = 0;
        bool uniform = false;      // lds.uni / an atomic: its one word goes to every lane
        uint64_t mask = 0;         // the active lanes: its RF write's lane enables
        SlotRounds access;
        uint8_t dst = 0;
        uint8_t depbar = 0;
        std::vector<uint32_t> words;
        std::size_t returned = 0;  // rounds whose words are in `words`
        bool engine_filled = false; // an atomic's return slot: one fill from the RMW engine
        [[nodiscard]] bool complete() const { return returned == (engine_filled ? 1 : access.rounds.size()); }
    };

    // A store slot (rfc12 section 3): one sts from admission until its last
    // round is granted. From each grant the A-write pipeline owns that round's
    // addresses, data and (last round) depbar completion for D cycles.
    struct StoreSlot {
        uint32_t warp = 0;
        SlotRounds access;
        uint8_t depbar = 0;
    };

    // One beat in the acc.ld engine's pipeline: one column of the transfer
    // (sixteen rows, one word per lane, sixteen distinct fine banks), with its
    // destination and completion context. Its tmem words are sampled when it
    // reaches the RAM stage (stage tmem.port_depth - 1), once.
    struct AccLdBeat {
        uint32_t warp = 0;
        uint8_t reg = 0;
        uint8_t accumulator = 0;
        uint32_t row_base = 0;
        uint32_t column = 0;          // the accumulator column this beat reads
        bool sampled = false;
        std::vector<uint32_t> values; // the words, from the tmem sample on
        bool last = false;    // the transfer's final beat: its sample ends the read, its RF write clears the depbar
        uint8_t depbar = 0;
        bool operator==(const AccLdBeat&) const = default;
    };
    // The acc.ld engine's pipeline: tmem.port_depth stages to the tmem sample,
    // then tmem.read_latency_cycles stages to the output, which asks the RF
    // write decision for service. Stage 0 takes the launched read.
    using AccLdPipeline = std::vector<std::optional<AccLdBeat>>;

    // One acc.st beat between its RF read and its tmem write: read (sampled)
    // at its grant, at the RF output Lr cycles later (where it leaves for
    // tmem), at the tmem RAM tmem.port_depth after that (land_cycle: written
    // at that cycle's edge).
    struct AccStBeat {
        uint32_t column = 0;
        uint64_t land_cycle = 0;
        std::vector<uint32_t> values;
    };

    // The transfer an engine is running. The sanitizer keys its tmem owner by
    // the engine (acc.ld or acc.st).
    struct AccTransfer {
        bool load = false;
        uint32_t warp = 0;
        uint8_t accumulator = 0;
        uint8_t rf_base = 0;
        uint8_t depbar = 0;
        uint32_t row_base = 0;
        uint32_t col_base = 0;
        uint32_t width = 1;
        uint32_t next_column = 0;   // beats launched (acc.ld: tmem reads; acc.st: RF reads)
        // acc.st: its beats in flight to tmem, oldest first, and how many
        // landed. The acc.st engine frees when the last one lands.
        std::deque<AccStBeat> in_flight;
        uint32_t landed = 0;
    };

    struct MmaCommand {
        uint64_t id = 0;
        bool commit = false;
        bool init = false;
        MmaShape shape;          // from the opcode (m 32 or 64, n 64, k 8)
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

    // One step in the core, from its entry to its result's tmem write. C (its
    // tile's sixteen accumulator words, one round of the MMA's tmem read
    // service) is requested as the step enters and sampled tmem.port_depth
    // later; the fused dot-plus-C result leaves the core result_latency after
    // entry and reaches the tmem RAM tmem.port_depth after that (land_cycle:
    // written at that cycle's edge). The command's result retires there.
    struct MmaResult {
        uint64_t c_sample_cycle = 0;
        uint64_t land_cycle = 0;
        uint64_t command = 0;
        uint8_t accumulator = 0;
        uint32_t m0 = 0;
        uint32_t n0 = 0;
        bool init_step = false;          // C = 0 instead of the tmem words
        std::vector<uint16_t> a;         // the step's A block, core_m x core_k fp16
        std::vector<uint16_t> b;         // its B block, core_n x core_k fp16 (transposed snapshot)
        bool c_sampled = false;
        std::vector<uint32_t> values;    // the result, computed at the C sample
    };

    // One core step: which tile, which K block, and whether it reloads the
    // held A block (rfc11 section 0: hold the 4x8 A block, stream the 8x4 B
    // blocks across N; A is read again only when the walker moves to the
    // next A block). It travels with the step's rounds through port B.
    struct MmaStep {
        uint64_t command = 0;
        uint8_t accumulator = 0;
        bool last_step = false;  // the command's last step: its final sample frees the operands
        bool init_step = false;  // C = 0 instead of the tmem value
        bool reload_a = false;   // this step reads its A block (n block 0); otherwise the held one
        uint32_t m0 = 0;
        uint32_t n0 = 0;
        uint32_t k0 = 0;
        uint32_t a_base = 0;
        uint32_t b_base = 0;
        MmaShape shape;          // the command's: the dense layout's strides follow it
    };

    // The step whose port-B rounds are being granted: one round per cycle,
    // from its first round to its last (no preemption).
    struct MmaFetch {
        MmaStep step;
        uint32_t next_round = 0;
        std::vector<std::vector<uint32_t>> round_elements; // fp16 element byte addresses, per round
    };

    // One granted port-B read round, in the read pipeline: D stages to the
    // RAM sample, then Lb stages to the port output (the same D and Lb as
    // port A, rfc10 section 5). Its element values are read at the sample.
    struct MmaRead {
        MmaStep step;
        bool last_round = false;
        std::vector<uint32_t> elements;                       // fp16 element byte addresses
        std::vector<std::pair<uint32_t, uint16_t>> values;    // (address, bits), read at the RAM sample
    };

    // What the RMW engine does with the word(s) it reads: an mbar operation
    // (both mbar words), or a uniform atomic (trash/rfc13.md section 3: one
    // 4-byte word, old value back through its return slot).
    enum class MbarRmwOperation : uint8_t {
        init,
        arrive,
        arrive_expect_tx,
        complete_tx,
        atom_add,
        atom_and,
        atom_xor,
        atom_exch,
    };

    // Field classes as in trash/sanitizer-split-2026-10-07.md: unmarked
    // fields are hardware (routing, completion context); (S) marks a field
    // read only for a PANIC decision or its text; (T) only for the trace.
    // Neither is pipeline width.
    // complete_tx carries no phase in hardware (rfc12 section 8): it changes
    // the mbar's CURRENT phase. Which phase the sender meant is sanitizer
    // metadata only: the TMA command id (its completion mbar's phase was
    // peeked at accept) or the host's stated parity.
    // A uniform atomic is a SIMT request like an mbar operation: its address,
    // its 32-bit operand in `value`, and the load slot its old value returns
    // through.
    struct MbarRmwRequest {
        MbarRmwOperation operation = MbarRmwOperation::arrive;
        uint32_t byte_address = 0;
        uint32_t value = 0;
        std::optional<uint32_t> warp;   // a SIMT operation's warp (its own shmem order); none: an engine event
        std::optional<uint32_t> return_slot; // an atomic: its return load slot
        uint8_t parity = 0;             // (S) the host's complete_tx (complete_mbar_tx): the phase it says it completes
        std::optional<uint64_t> tma_command; // (S) a TMA's complete_tx: its command (the accept-time phase peek)
        uint32_t pc = 0;                // (S) PANIC text
        Instruction instruction;        // (S) PANIC text
    };

    // The two engines that present mbar events to the RMW engine.
    enum class MbarEngine : uint8_t { tma, mma };

    // The RMW engine's one operation (rfc12 section 6), from its START to its
    // final write grant. Its requester waits in a register (the SIMT RMW
    // register, or an engine's event register) until the RMW starts: the
    // engine takes it and the global STORE PAUSE begins (no new store
    // transaction starts on A-write). Its read is granted on A-read (first
    // among new requesters) strictly after the store transaction under way, if
    // any, has had its last round granted; its write D + Lb + C cycles after
    // the read (the words reach the port output, then the update crosses the C
    // ALU stages). The write grant ends the operation and the pause.
    struct ActiveMbarRmw {
        MbarRmwRequest request;
        bool read_granted = false;
        // (T) Whose engine event this is (none: SIMT). Read only for the
        // trace label and the engine_waits_on_other counter.
        std::optional<MbarEngine> engine;
    };

    // An RMW start: the requester that enters the idle RMW engine now.
    struct RmwStart {
        std::optional<MbarEngine> engine;   // that engine's event register; none: the SIMT RMW register
    };

    // A probe (test_wait / try_wait): where its result goes and what it compares.
    struct MbarReadRequest {
        bool test = false;
        uint32_t warp = 0;
        uint8_t predicate_dst = 0;
        std::optional<uint32_t> wait_slot;
        uint32_t byte_address = 0;
        uint8_t parity = 0;
        uint32_t pc = 0;                // (S) PANIC text
        Instruction instruction;        // (S) PANIC text
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
            std::array<uint8_t, 4> data{}; // descriptor reads and OOB zero fills only
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
    // FIFO at the pluck rate; when complete, placed into shmem by the port-B
    // write in its bank rounds, which read the register's Q at their grants.
    // Free after the last round's grant (and refillable in that cycle).
    struct TmaSectorRegister {
        std::optional<uint64_t> request;
        uint64_t sector_address = 0;
        std::vector<uint8_t> data;
        uint32_t filled = 0;        // bytes plucked in
        bool complete = false;      // all bytes in; its ticket is in the TMA queue's returns
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

    // I3 (rfc12 section 8): no mbar phase read between descriptor and walking.
    enum class TmaCommandState : uint8_t {
        descriptor,
        walking,
        data_complete,
        event_presented,
    };

    struct TmaCommand {
        uint64_t id = 0;
        uint32_t warp = 0;
        uint32_t pc = 0;
        Instruction instruction;
        // The instruction's operands: the opcode's direction, the descriptor
        // (an immediate), the origin (x, y) and the address register's shared
        // address and completion mbar (src/tma.hpp TmaAddresses).
        TmaDirection direction = TmaDirection::load;
        uint32_t descriptor_address = 0;
        std::array<int32_t, 2> origin{};
        uint32_t shmem_address = 0;
        uint32_t mbar_address = 0;
        TmaCommandState state = TmaCommandState::descriptor;
        TmaDescriptor descriptor;
        uint64_t logical_bytes = 0;
        uint64_t useful_bytes = 0;
        uint32_t x = 0;
        uint32_t y = 0;
        uint32_t pending_load_writes = 0;
        uint32_t pending_store_captures = 0;
        bool walker_done = false;
    };

    enum class TmaPortKind : uint8_t {
        descriptor,
        load_write,
        store_read,
    };

    // One shmem transaction of the TMA. Load placements (port-B write) read
    // the TMA sector register (or write zeros for an OOB fill); store reads
    // (A-read) fill the TMA store sector register. Fragments carry addresses;
    // only the descriptor read (A-read) holds its returned data here (the TMA
    // frontend latch). next_round: rounds granted.
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

    // A TMA sector ready for placement (port-B write): either the TMA stream's sector
    // register (complete; it stays occupied until its last round) or a
    // synthetic OOB zero fill (no bus data, placement metadata only).
    struct TmaReturn {
        uint64_t command = 0;
        uint64_t sector_request = 0;
        std::vector<GmemSectorRequest::TmaFragment> fragments;
        bool from_sector_register = false;
    };

    // A TMA store sector register (32 B flops; tma.store_sector_registers of
    // them, a ring in reservation order). The store-sector path, one one-round
    // sector, L = D + Lb, d = ceil(sector words / lsu.pluck_words_per_cycle):
    //   g          the walker's source read is granted on A-read (bypass (h):
    //              planned from Q and this cycle's egress drain). The register
    //              at the ring's tail is reserved now; the read's rounds carry
    //              its index (the return context). A-read is free again after
    //              the read's last round's grant.
    //   g + L      the words reach the port output and are written into the
    //              register at the edge (its `source` says where each byte goes)
    //   g + L + 1  complete in Q: the oldest complete register (the ring's
    //              head) gets the egress write port (TMA wins a new egress
    //              slot) and drains pluck words per cycle
    //   g + L + d  its last word leaves: the register is free, and a source
    //              read granted in this same cycle may reserve it (d)
    // One register therefore turns over every L + d cycles; N registers and a
    // free egress port sustain min(N x 32 / (L + d), 32 / d) bytes per cycle.
    struct TmaStoreRegister {
        uint64_t command = 0;
        uint64_t sector_address = 0;
        std::vector<uint8_t> data;
        uint64_t byte_enable = 0;
        TmaPortRequest source;      // the source read's rounds and fragments: each returned byte's place in `data`
        uint32_t returned = 0;      // source rounds whose words are in `data`
        uint32_t words_out = 0;     // words already written into the egress FIFO
        [[nodiscard]] bool complete() const { return returned == source.rounds.size(); }
    };
    struct TmaStoreRegisters {
        std::vector<std::optional<TmaStoreRegister>> ring;
        uint32_t head = 0;          // the oldest held register: the one the egress drains
        [[nodiscard]] uint32_t held() const {
            return static_cast<uint32_t>(std::count_if(ring.begin(), ring.end(),
                [](const auto& reg) { return reg.has_value(); }));
        }
    };

    struct IssueAttempt {
        bool issued = false;
        WarpState state = WarpState::structural;
        std::string reason;
    };

    // =======================================================================
    // Register contents that change together (one Reg each, below)
    // =======================================================================

    // A shmem write: its bytes land at the clock edge of the cycle it reaches
    // the RAM; every read of shmem in that cycle sees the old word (READ_FIRST).
    struct ShmemWrite {
        uint32_t address = 0;
        std::vector<uint8_t> bytes;
    };

    struct DepbarEvent {
        uint32_t warp = 0;
        uint8_t bar = 0;
    };

    // One granted A-read round in the read pipeline: D stages to the RAM
    // sample, then Lb stages to the port output, where its requester takes the
    // words. Its return context says where they go: a load slot or a TMA
    // store register (by index), the TMA frontend latch (the TMA read
    // transaction, held until its words are back), the RMW engine's ALU, or a
    // probe -- whose register frees at its grant, so the request rides along.
    enum class ReadClient : uint8_t { load_slot, probe, rmw, tma_frontend, tma_store };
    struct ReadBeat {
        ReadClient client = ReadClient::load_slot;
        uint32_t slot = 0;                    // load_slot / tma_store: which load slot / store register
        std::size_t round = 0;                // load_slot / tma_*: the transaction's round
        std::vector<uint32_t> addresses;      // word-aligned byte addresses, at most one per bank
        std::vector<uint32_t> words;          // their contents, read at the RAM sample
        std::optional<MbarReadRequest> probe; // probe: the request
    };

    struct MbarWake {
        uint32_t address = 0;
        uint8_t phase = 0;
    };

    // A TMA placement's last round reaches the RAM: its command's load write
    // ends (pending count, credit release, data complete).
    struct PlacementEnd {
        uint64_t command = 0;
        std::vector<GmemSectorRequest::TmaFragment> fragments; // (S) the live-mbar PANIC
    };

    // One granted write round in a write pipeline (A-write, port-B write): D
    // stages to the RAM; at the last its bytes land at the clock edge and its
    // completion happens in that cycle.
    struct WriteBeat {
        std::vector<ShmemWrite> writes;
        std::optional<DepbarEvent> sts_done;   // an sts's last round: its depbar clears
        std::optional<MbarWake> wake;          // an RMW write that flips the phase wakes the waiters
        std::optional<PlacementEnd> placed;    // a TMA placement's last round: its load write ends
    };
    using ReadPipeline = std::vector<std::optional<ReadBeat>>;   // D + Lb stages
    using WritePipeline = std::vector<std::optional<WriteBeat>>; // D stages

    // The RMW engine: the operation in flight (its presence IS the store
    // PAUSE: store_pause()), its ALU stages and the admission state between
    // its requesters (they change together).
    struct RmwEngine {
        std::optional<ActiveMbarRmw> active;
        // The ALU (shmem.rmw_compute_cycles = C register stages, rfc12 section
        // 2): the update computed from the read's words at the port output
        // enters stage 0 at that cycle's edge; it is ready in the last stage C
        // cycles after the port output, and its write is granted then.
        std::vector<std::optional<WriteBeat>> alu;
        // The engine whose event was started last (set on every engine-event
        // start, contested or not). When both engines hold an event, the other
        // engine's goes. Starts at mma, so the first such tie serves the TMA.
        MbarEngine last_served = MbarEngine::mma;
        // Fairness between the two classes of requester (implementer choice,
        // I3): when the SIMT RMW register and an engine event register both
        // hold a request at a start, the class that did not start last goes.
        // Set on every start. Starts false (an engine event went last), so the
        // first such tie serves the SIMT RMW.
        bool simt_started_last = false;
    };

    // The TMA's A-read transaction. A descriptor read (the frontend latch)
    // holds it from its first grant until its last round's words are back; a
    // store-source read only while it has rounds left to be granted (A-read
    // is released at its last round's grant; each round carries its store
    // register's index back). The TMA starts no other read while it is held.
    struct TmaReadTransaction {
        TmaPortRequest request;       // next_round: rounds granted; descriptor fragments latch the words
        uint32_t returned = 0;        // descriptor: rounds whose words are back
        uint32_t store_register = 0;  // store-source: the TMA store register its rounds land in
    };

    // The A-read and A-write round-robin pointers (rfc12 section 5). A-read
    // requesters in order: load slots 0..L-1, the probe register, the TMA;
    // A-write: store slots 0..S-1. A pointer moves past the requester it
    // grants, only on a grant. And A-write's alternation bit between RMW
    // starts and store transactions.
    struct ShmemArbiters {
        uint32_t a_read = 0;
        uint32_t a_write = 0;
        // The last start on A-write was an RMW (true) or a store transaction
        // (false). When an RMW and a store both wait to start, the one that did
        // not start last goes: one store transaction between consecutive RMWs.
        // Consulted only to choose which STARTS next -- never between an RMW's
        // read and its write (the PAUSE holds every new store then). Starts
        // false: the first contested start goes to the RMW (implementer choice).
        bool rmw_started_last = false;
        bool operator==(const ShmemArbiters&) const = default;
    };

    // The RF read decision's state (rfc10 section 3, "RF reads: the same rule,
    // mirrored"), per RF bank: the read_owed bit and the round-robin cursor
    // over the async readers.
    struct RfReadPolicy {
        std::vector<uint8_t> owed;
        std::vector<uint32_t> cursor;
        bool operator==(const RfReadPolicy&) const = default;
    };

    // The RF write decision's state (rfc10 section 3), per RF bank: the
    // async_owed bit and the round-robin cursor over the async holders. Both
    // change in the same calculation.
    struct RfWritePolicy {
        std::vector<uint8_t> owed;
        std::vector<uint32_t> cursor;
        // The LSU has ONE RF write path (rfc09 section 3a: one lane-wide vector
        // per cycle). Its output mux presents at most one assembly register per
        // cycle, round-robin from here; it moves past the register written.
        uint32_t lsu_cursor = 0;
        bool operator==(const RfWritePolicy&) const = default;
    };

    using WaitSlots = std::vector<std::optional<MbarWaitSlot>>;
    using DepbarCounters = std::vector<std::array<uint32_t, 6>>;             // [warp][sb]
    using AccEngines = std::array<std::optional<AccTransfer>, 2>; // [0 the acc.ld engine, 1 the acc.st engine]

    // The MMA engine's queue and pipeline. One group: a command's next_step
    // and operands_consumed track the fetch, its outstanding_results the
    // result list. (Its event register is separate.)
    struct MmaPipeline {
        std::deque<MmaCommand> queue;   // mma.queue_commands; issue pushes at the back
        std::optional<MmaFetch> fetch;  // the step whose rounds port B is granting
        std::vector<std::optional<MmaRead>> reads; // the port-B read pipeline: D + Lb stages
        // The core's operand latch: the values of the step whose rounds are
        // arriving at the port output (A + B of one step: at most 128 B).
        std::vector<std::pair<uint32_t, uint16_t>> arriving;
        // The held A block: core_m x core_k fp16 (64 B at 4x8), written when a
        // reloading step enters the core, used by the steps after it.
        std::vector<uint16_t> a_held;
        std::deque<MmaResult> results;  // steps in the core and in the tmem write pipeline
    };

    // The TMA engine's command queue and walker bookkeeping. One group:
    // credits, tickets and each command's pending counts move together. (Its
    // event register, shmem transactions and the two sector registers are
    // separate registers.)
    struct TmaQueue {
        std::map<uint64_t, TmaCommand> commands; // tma.queue_commands; issue inserts
        std::optional<uint64_t> frontend;        // the command the descriptor / walker frontend serves
        std::optional<uint64_t> event_command;   // whose completion event is in the TMA's event register
        uint32_t credits_used = 0;               // of tma.sector_credits
        std::deque<TmaReturn> returns;           // sectors ready for placement (port-B write)
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

    // The async RF writers (rfc10 section 3): holders of a complete result
    // vector that waits for its bank's write port. Round-robin order: load
    // slots, then assembly registers, then the acc.ld engine's pipeline output.
    enum class AsyncWriter : uint8_t { load_slot, assembly, acc_ld };
    struct AsyncWrite {
        AsyncWriter writer = AsyncWriter::load_slot;
        uint32_t index = 0;   // load slot / assembly register / 0 (the acc.ld engine)
        uint32_t warp = 0;
        uint8_t reg = 0;
    };

    // The RF write decision of this cycle, computed from Q before issue.
    struct RfWriteDecision {
        std::vector<std::optional<AsyncWrite>> holders; // every holder in round-robin order; set when ready
        std::vector<uint8_t> granted;                    // per holder: written into the RF this cycle
        struct Bank {
            bool ready = false;          // async_ready
            bool granted = false;        // async_grant
            bool inhibit_simt = false;   // async_denied AND async_owed
            uint32_t cursor_after = 0;
        };
        std::vector<Bank> banks;
        std::optional<uint32_t> lsu_presented;           // the assembly register on the LSU's one write path
        [[nodiscard]] bool granted_to(AsyncWriter writer, uint32_t index) const;
    };

    // The async RF readers (rfc10 section 3, mirrored): each wants a read
    // port of one bank in the CURRENT cycle, where the issue register's
    // instruction owns the ports it samples (trash/rfc14.md section 4).
    // Round-robin order: the acc.st engine's next beat, then the LSU's
    // store-data read (vector stores).
    enum class RfReader : uint8_t { acc_st, lsu_store };
    struct RfReadRequest {
        RfReader reader = RfReader::acc_st;
        uint32_t index = 0;          // 0 (the acc.st engine)
        uint32_t warp = 0;
        uint8_t reg = 0;
        uint64_t lsu_command = 0;    // lsu_store
        uint32_t assembly = 0;       // lsu_store
    };

    // The RF read decision of this cycle, from Q before issue:
    //   read_ready   = an async reader wants this bank
    //   read_grant   = read_ready AND the issue register's instruction leaves
    //                  one of the bank's read ports free now
    //   inhibit_simt = read_ready AND NOT read_grant AND read_owed
    struct RfReadDecision {
        std::vector<std::optional<RfReadRequest>> holders;
        std::vector<uint8_t> granted;
        struct Bank {
            bool ready = false;
            bool granted = false;
            bool inhibit_simt = false;
            uint32_t cursor_after = 0;
        };
        std::vector<Bank> banks;
        [[nodiscard]] bool granted_to(RfReader reader, uint32_t index) const;
    };

    // A-read's port output -> a load slot's gather register (one round's words).
    struct LoadFill {
        uint32_t slot = 0;
        std::vector<std::pair<uint32_t, uint32_t>> lanes; // (lane, word)
    };

    // A-read's port output -> a TMA store register (one source round's words).
    struct TmaStoreFill {
        uint32_t store_register = 0;
        uint64_t command = 0;
        std::vector<std::pair<uint32_t, uint8_t>> bytes;  // (byte offset in the sector, byte)
        bool last = false;                                // the source read's last round: the register is complete
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
        bool tma_store_drained = false;                // the head TMA store register's last word leaves
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
        std::optional<TmaPortRequest> placement;   // the front ticket's placement, to start on the port-B write
        std::optional<TmaPortRequest> a_read;      // a descriptor / store-source read to start on A-read
        std::optional<uint64_t> frontend;          // the frontend command (selection persists)
        bool frontend_reset = false;
        std::optional<TmaWalkerPlan> walker;
        std::optional<uint32_t> store_register;    // a store sector: the TMA store register its source read will reserve
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

    // The shmem grants of this cycle: one select per service (rfc12 section 5).
    enum class AReadUser : uint8_t { none, load_slot, probe, rmw, tma };
    struct AReadGrant {
        AReadUser user = AReadUser::none;
        uint32_t slot = 0;                // load_slot: which
        bool tma_start = false;           // tma: a new transaction, round 0 now (bypass (h))
        std::optional<uint32_t> pointer_after; // a round-robin grant: the A-read pointer's next value
        static AReadGrant to(AReadUser user, uint32_t slot = 0) {
            AReadGrant grant;
            grant.user = user;
            grant.slot = slot;
            return grant;
        }
    };
    enum class AWriteUser : uint8_t { none, store_slot, rmw };
    struct AWriteGrant {
        AWriteUser user = AWriteUser::none;
        uint32_t slot = 0;                // store_slot: which
        std::optional<uint32_t> pointer_after; // a round-robin grant (a new store transaction): the A-write pointer's next value
        bool rmw_start = false;           // A-write's alternation lets an RMW start now: its store PAUSE begins
    };
    enum class BWriteUser : uint8_t { none, tma_continue, tma_start };

    // shmem this cycle (step_shmem), in order: the RAM stage (from the
    // pipelines' Q), the grants, their data paths.
    struct ShmemCycle {
        std::optional<ReadBeat> sampled;             // the read round at the RAM now, with its words
        std::optional<ReadBeat> delivered;           // the read round at the port output now
        std::vector<ShmemWrite> writes;              // the write rounds at the RAM now: land at the clock edge
        AWriteGrant a_write;
        std::optional<RmwStart> rmw_start;           // the requester the RMW engine takes now (its PAUSE begins)
        AReadGrant a_read;
        BWriteUser b_write = BWriteUser::none;
        std::optional<ReadBeat> a_read_beat;         // the rounds granted now, entering their pipelines
        std::optional<WriteBeat> a_write_beat;
        std::optional<WriteBeat> b_write_beat;
        ShmemArbiters arbiters;                      // the round-robin pointers after the grants
        // The warp whose own access A-read / A-write grants a round of now: an
        // lds / sts round, a probe, its SIMT RMW's read / write (none: idle, a
        // TMA read, an engine event's RMW). It reaches the RAM D cycles later.
        std::optional<uint32_t> a_read_warp;
        std::optional<uint32_t> a_write_warp;
        // What the grants and returns do to their requesters.
        std::optional<LoadFill> load_fill;           // a load round's words for its gather register
        std::optional<WriteBeat> rmw_update;         // the RMW's update, from its read's words: enters ALU stage 0
        std::optional<ActiveMbarRmw> rmw_started;    // the operation the RMW engine takes now (from rmw_start's register)
        bool simt_rmw_started = false;               // rmw_start empties the SIMT RMW register
        bool tma_event_claimed = false;              // ... the TMA's event register
        bool mma_event_claimed = false;              // ... the MMA's event register
        bool tma_read_changed = false;               // tma_read_after is the TMA read transaction's next value
        std::optional<TmaReadTransaction> tma_read_after;
        std::optional<TmaPortRequest> tma_read_done; // a descriptor read's last words came back now, with its data
        std::optional<uint32_t> tma_store_reserved;  // a store-source read granted now reserves this TMA store register
        std::optional<TmaStoreFill> tma_store_fill;  // a store-source round's words for its TMA store register
        bool tma_write_changed = false;              // tma_write_after is the placement transaction's next value
        std::optional<TmaPortRequest> tma_write_after;
        bool tma_sector_released = false;            // a placement's last round read the TMA sector register now
        std::optional<PlacementEnd> tma_placed;      // a placement's last write reached the RAM now
        std::optional<std::pair<uint32_t, uint64_t>> park; // try-wait probe back incomplete: (slot, deadline)
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
    // The LSU's store-data RF read (vector stores): sampled at its grant, at
    // the RF output Lr cycles later (arrive_cycle), when it fills the
    // command's assembly register.
    struct LsuRfRead {
        std::optional<uint64_t> command;
        uint32_t assembly = 0;
        std::vector<uint32_t> values;
        uint64_t arrive_cycle = 0;
    };

    // The front end this cycle, all from Q: the fetch grant (round-robin over
    // the warps with FIFO space, counting the read in flight; no input from
    // this cycle's issue), the word at the instruction memory's output, and
    // the branch in the issue register resolved.
    struct FetchGrant {
        std::optional<uint32_t> warp; // granted now: the memory samples its fetch_pc at this edge
        uint32_t pc = 0;              // the packet's address (a multiple of 16)
        uint32_t cursor_after = 0;
    };
    struct FetchedPacket {
        uint32_t warp = 0;
        FetchPacket words{};          // the memory output (zero past the kernel's words)
    };
    struct BranchResolution {
        uint32_t warp = 0;
        bool taken = false;
        uint32_t target = 0;          // taken: the warp's pc and fetch pc from the next cycle
    };

    // What an accepted instruction changes outside the issue unit now.
    struct IssueDecision {
        std::optional<uint32_t> popped;                  // the warp whose FIFO head was accepted
        bool halt = false;                               // ... a halt: the warp's front end stops
        std::optional<IssuedBranch> branch;              // ... a branch: into the issue register
        std::optional<DepbarEvent> depbar_admission;
        std::optional<uint32_t> rf_write_bank;           // a fixed RF-write booking was made to this bank
        std::vector<uint8_t> rf_read_banks;              // per bank: its round's reads (into the issue register; read_owed)
        std::optional<DispatchEntry> dispatch;           // a memory / engine instruction enters the dispatch pipeline
    };

    // Everything a dispatched instruction hands to its unit (the unit takes it now).
    struct DispatchDecision {
        std::optional<std::pair<uint32_t, LoadSlot>> load_slot;   // lds: (slot, its new contents)
        std::optional<std::pair<uint32_t, StoreSlot>> store_slot; // sts: (slot, its new contents)
        std::optional<MbarRmwRequest> rmw;               // SIMT mbar RMW: into the SIMT RMW register
        std::optional<MbarReadRequest> mbar_probe;
        std::optional<std::pair<uint32_t, MbarWaitSlot>> wait_slot; // try-wait: (slot, record)
        std::optional<MmaCommand> mma;
        struct AccAdmission { bool load = false; AccTransfer transfer; };
        std::optional<AccAdmission> acc;
        struct LsuAdmission {
            LsuCommand command;
            std::optional<std::vector<uint32_t>> store_data; // scalar stg: its issue round's read, latched
        };
        std::optional<LsuAdmission> lsu;
        std::optional<TmaCommand> tma;
    };

    // The space the instructions in the dispatch pipeline will take when
    // they reach their units (admission secured it at their issue).
    struct DispatchClaims {
        uint32_t load_slots = 0;
        uint32_t store_slots = 0;
        uint32_t simt_rmw = 0;
        uint32_t probes = 0;
        uint32_t wait_slots = 0;
        uint32_t lsu_commands = 0;
        uint32_t tma_commands = 0;
        uint32_t mma_commands = 0;
        std::array<uint32_t, 2> acc{}; // [0 acc.ld, 1 acc.st]
    };

    // All of a cycle's wires, in the order tick() computes them.
    struct Wires {
        // The front end, from Q only.
        FetchGrant fetch;
        std::optional<FetchedPacket> fetched;
        std::optional<BranchResolution> branch;
        // The RF write port: async grants and SIMT inhibition.
        RfWriteDecision rf_write;
        std::vector<uint64_t> released_lsu_commands;  // their last write / ack at the port now
        // The RF and predicate writes at the RAM now: they land at this
        // cycle's clock edge (after every read of this cycle).
        std::vector<Writeback> rf_landings;
        // The tmem writes at the RAM now (MMA results, acc.st beats): they
        // land at this cycle's clock edge, after every tmem read of this
        // cycle (READ_FIRST, as shmem). (word address, value)
        std::vector<std::pair<uint32_t, uint32_t>> tmem_writes;
        // Collected as they happen, from several steps below.
        std::vector<DepbarEvent> depbar_completions;  // in the order they happened
        std::vector<uint32_t> completed_waits;        // try-wait slots completed this cycle
        std::vector<Writeback> new_writebacks;        // RF writes booked now, in wire order
        // The store path's last stage; request generation; the two selects.
        EgressWrite egress;
        TmaRequests tma;
        LsuRequestWant lsu_want;
        RequestChannel channel;
        ShmemCycle shmem;
        // Next state of the units issue looks ahead at (committed by their
        // drivers, with issue's additions).
        MmaPipeline mma_next;
        std::optional<MbarRmwRequest> mma_present;    // a commit presented into the MMA's event register
        AccEngines acc_next;
        AccLdPipeline acc_ld_next;
        IngressPluck ingress;
        ControllerTick controller;
        TmaQueue tma_next;
        // The dispatch pipeline's end: its unit takes an instruction now.
        bool dispatched = false;
        DispatchDecision dispatch;
        // The async RF readers (the read ports the issue register leaves free now), before issue.
        RfReadDecision rf_read;
        std::optional<AccStBeat> acc_st_read;               // the acc.st engine's beat read now
        LsuRfRead lsu_rf_read;                              // the LSU store-data beat read now
        // Issue.
        IssueDecision issue;
        // The front ends after this cycle (step_front_end, after issue).
        std::vector<WarpFrontEnd> front_end_next;
        // The RF read stage: dispatch entries' operands latched now
        // (index in the pipeline's Q, operands by read).
        std::vector<std::pair<std::size_t, std::vector<std::vector<uint32_t>>>> dispatch_samples;
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
    std::vector<uint8_t> shmem_;                  // writes land at the clock edge of the cycle they reach the RAM
    std::vector<uint8_t> gmem_;                   // the controller's
    // tmem: fine_banks x words_per_fine_bank words; word w is in fine bank
    // w % fine_banks. Each fine bank's two read and two write services belong
    // to the MMA (C read, result write) and to the acc.ld and acc.st engines.
    std::vector<uint32_t> tmem_;

    // ---- Registers: the declared storage boundaries ----
    // The front end (trash/front-end-2026-10-18.md): per warp slot its FIFO,
    // PCs and halted bit; the instruction memory's read in flight; the fetch
    // round-robin cursor (4 bits at 16 warps); the issue register's branch.
    Reg<std::vector<WarpFrontEnd>> front_end_{registers_, "instruction FIFOs and PCs"};
    Reg<std::optional<FetchRead>> fetch_read_{registers_, "instruction memory read (address, tag)"};
    Reg<uint32_t> fetch_cursor_{registers_, "fetch round-robin cursor"};
    Reg<std::optional<IssuedBranch>> issue_branch_{registers_, "issue register: branch"};
    // Scheduler-visible state other units complete.
    Reg<DepbarCounters> depbar_{registers_, "depbar counters"};
    Reg<WaitSlots> mbar_wait_slots_{registers_, "mbar wait slots"};
    // shmem (rfc12): the SIMT service slots, the probe register, the RMW
    // engine and its three requester registers, the services' pipelines and
    // arbiters. Each requester holds at most one pending RMW in its own
    // one-entry register:
    //   SIMT: one mbar.init / arrive / arrive.expect_tx or one uniform
    //         atomic, accepted at issue (machine-wide; a warp waits at issue
    //         while it is held).
    //   TMA:  complete_tx {address, bytes} (TMA data complete, or the host's
    //         complete_mbar_tx). An engine stalls only on its own register.
    //   MMA:  a commit's arrive {address}.
    // A request written at t is visible to the RMW engine from t + 1.
    std::deque<Reg<std::optional<LoadSlot>>> load_slots_;   // shmem.load_slots, one register each
    std::deque<Reg<std::optional<StoreSlot>>> store_slots_; // shmem.store_slots, one register each
    Reg<std::optional<MbarReadRequest>> mbar_probe_{registers_, "SIMT mbar probe"};
    Reg<std::optional<MbarRmwRequest>> simt_rmw_{registers_, "SIMT RMW register"};
    Reg<RmwEngine> rmw_{registers_, "RMW engine"};
    Reg<ReadPipeline> a_read_pipe_{registers_, "A-read pipeline"};
    Reg<WritePipeline> a_write_pipe_{registers_, "A-write pipeline"};
    Reg<WritePipeline> b_write_pipe_{registers_, "port-B write pipeline"};
    Reg<ShmemArbiters> shmem_arbiters_{registers_, "shmem A-read / A-write arbiters"};
    // fence.sh (rfc12 section 10), per warp: how many cycles from now until
    // the warp's latest granted shmem round reaches the RAM (a write's edge,
    // a read's sample: grant + D). A grant at g sets it to D - 1 at g + 1; it
    // counts down to 0, the cycle that round is at the RAM. 0..D-1:
    // ceil(log2(D)) bits per warp.
    Reg<std::vector<uint32_t>> shmem_to_ram_{registers_, "per-warp shmem cycles to the RAM"};
    Reg<std::optional<MbarRmwRequest>> tma_mbar_event_{registers_, "TMA mbar event register"};
    Reg<std::optional<MbarRmwRequest>> mma_mbar_event_{registers_, "MMA mbar event register"};
    // MMA and accumulator transfers.
    Reg<MmaPipeline> mma_{registers_, "MMA queue and pipeline"};
    Reg<AccEngines> acc_engines_{registers_, "acc transfer engines"};
    Reg<AccLdPipeline> acc_ld_pipe_{registers_, "acc.ld pipeline"};
    // TMA.
    Reg<TmaQueue> tma_{registers_, "TMA command queue"};
    // The TMA's shmem transactions. Round 0 of each is granted in the cycle
    // tma_requests builds it (bypass (h) in tick()). The placement (port-B
    // write) register holds the rounds after the first; the A-read one holds
    // a descriptor read until its last round's words are back, a
    // store-source read until its last round's grant.
    Reg<std::optional<TmaPortRequest>> tma_write_txn_{registers_, "TMA placement transaction"};
    Reg<std::optional<TmaReadTransaction>> tma_read_txn_{registers_, "TMA read transaction"};
    Reg<TmaSectorRegister> tma_sector_{registers_, "TMA sector register"};
    Reg<TmaStoreRegisters> tma_store_registers_{registers_, "TMA store sector registers"};
    // LSU.
    Reg<std::deque<LsuCommand>> lsu_commands_{registers_, "LSU command slots"}; // admission order
    std::deque<Reg<AssemblyRegister>> staging_;   // the lsu.staging_vectors pool, one register each
    Reg<IngressFifo> ingress_{registers_, "ingress FIFO"};
    Reg<std::deque<EgressSlot>> egress_fifo_{registers_, "egress FIFO"};
    Reg<std::map<uint64_t, ReadRecord>> read_records_{registers_, "read ID table"};
    Reg<std::vector<uint8_t>> free_id_tags_{registers_, "free-ID tags"}; // 1 = in flight
    // The RF write path: fixed results waiting for their booked port cycle,
    // and every write from its port cycle until it lands (src/rf.hpp).
    Reg<std::deque<Writeback>> writebacks_{registers_, "RF write pipeline"};
    Reg<RfWritePolicy> rf_write_policy_{registers_, "RF write policy (async_owed, cursors)"};
    Reg<RfReadPolicy> rf_read_policy_{registers_, "RF read policy (read_owed, cursors)"};
    // The issue register's RF bank use (trash/rfc14.md section 4): per bank,
    // how many registers the instruction accepted last cycle samples now (at
    // most rf.read_ports_per_bank: one bit per bank at the canonical one
    // port; zero when nothing was accepted). It owns those read ports this
    // cycle; the async readers take the rest (decide_rf_reads).
    Reg<std::vector<uint8_t>> issue_read_banks_{registers_, "issue register: RF read bank use"};
    // Issue -> units: the issue register and the RF read pipeline
    // (issue_to_dispatch stages), carrying memory / engine instructions.
    Reg<std::deque<DispatchEntry>> dispatch_pipe_{registers_, "dispatch pipeline"};
    // The LSU's store-data RF reads travelling to the assembly registers (Lr).
    Reg<std::deque<LsuRfRead>> lsu_read_pipe_{registers_, "LSU store-data read pipeline"};

    // ---- Unit-internal state (only its own unit's code touches it) ----
    // Issue: scheduler cursor, SFU occupancy, the arithmetic operand latches.
    // (Read only by the scheduler before it accepts, written at acceptance:
    // fetch never reads them.)
    uint32_t scheduler_cursor_ = 0;
    // First cycle the SFU is free again. The unit is occupied from an SFU
    // op's execution start (issue + issue_to_operands) for ceil(lanes /
    // sfu_lanes) cycles; a later SFU op stalls structurally until its own
    // execution start fits.
    uint64_t sfu_busy_until_ = 0;
    std::deque<ArithmeticRead> arithmetic_reads_;
    // The RF write port's reservation calendar (the arbiter's schedule):
    // fixed issue books a FUTURE cycle only, its write at issue +
    // issue_to_port; the async writers use the current write port (rfc10
    // section 3). A map by absolute cycle; in hardware a shift register per
    // bank of issue_to_port(slowest fixed op) bits. (There is no read
    // calendar: reads are one round at issue + 1, owned by the issue
    // register -- issue_read_banks_.)
    std::map<uint64_t, std::vector<uint32_t>> write_reservations_;
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

    // ---- the front end, from Q only: fetch, the memory output, the branch ----
    [[nodiscard]] FetchGrant decide_fetch() const;
    [[nodiscard]] std::optional<FetchedPacket> instruction_memory_output() const;
    [[nodiscard]] std::optional<BranchResolution> resolve_branch() const;
    // After issue: every warp's front end after this cycle (pop, landing,
    // halt, redirect, fetch pc), and its trace.
    void step_front_end(Wires& w);
    // ---- RF write path: the one async decision, the port, the RAM stage ----
    [[nodiscard]] RfWriteDecision decide_rf_writes() const;
    [[nodiscard]] std::vector<std::optional<AsyncWrite>> ready_async_writers(std::optional<uint32_t> lsu_presented) const;
    [[nodiscard]] std::optional<AsyncWrite> assembly_rf_write(uint32_t index) const;
    [[nodiscard]] std::optional<uint32_t> lsu_rf_presented() const;
    [[nodiscard]] uint32_t writes_booked_now(uint32_t bank) const;
    void rf_write_port(Wires& w);
    void rf_ram_stage(Wires& w);
    void land_rf_writes(const Wires& w);
    void release_lsu_command(Wires& w, uint64_t id);
    Writeback async_writeback(const AsyncWrite& write);
    void count_rf_write_waits(const RfWriteDecision& decision);
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
    // shmem (rfc12): the RAM stage, one select per service, their data paths.
    void step_shmem(Wires& w);
    void shmem_ram_stage(ShmemCycle& shmem) const;
    void commit_shmem_writes(Wires& w);
    void deliver_shmem_read(Wires& w);
    [[nodiscard]] AWriteGrant select_a_write(const Wires& w) const;
    [[nodiscard]] std::optional<RmwStart> select_rmw_admission(const Wires& w) const;
    [[nodiscard]] AReadGrant select_a_read(const Wires& w) const;
    [[nodiscard]] BWriteUser select_b_write(const Wires& w) const;
    void serve_rmw_start(Wires& w);
    void serve_a_read(Wires& w);
    void serve_a_write(Wires& w);
    void serve_b_write(Wires& w);
    // ---- units whose next state issue looks ahead at ----
    void step_mma(Wires& w);
    void step_acc_transfers(Wires& w);
    void step_acc_ld(std::optional<AccTransfer>& engine, AccLdPipeline& pipe, const Wires& w);
    std::optional<AccLdBeat> launch_acc_ld_beat(std::optional<AccTransfer>& engine);
    void sample_acc_ld_beat(AccLdBeat& beat);
    void step_acc_st(std::optional<AccTransfer>& engine, Wires& w);
    void decide_mbar_timeouts(Wires& w);
    IngressPluck decide_ingress(const Wires& w);
    void step_gmem_controller(Wires& w);
    void step_tma(Wires& w);
    // ---- dispatch: the RF read pipeline's end, the unit takes it ----
    void dispatch_issued(Wires& w);
    void hand_over(const DispatchEntry& op, Wires& w);
    void dispatch_mma(const DispatchEntry& op, Wires& w);
    void dispatch_acc_transfer(const DispatchEntry& op, Wires& w);
    void dispatch_shared_load(const DispatchEntry& op, Wires& w);
    void dispatch_shared_store(const DispatchEntry& op, Wires& w);
    void dispatch_lsu(const DispatchEntry& op, Wires& w);
    void dispatch_tma(const DispatchEntry& op, Wires& w);
    void dispatch_mbar_rmw(const DispatchEntry& op, Wires& w);
    void dispatch_atomic(const DispatchEntry& op, Wires& w);
    void dispatch_mbar_wait(const DispatchEntry& op, Wires& w);
    [[nodiscard]] DispatchClaims dispatch_claims() const;
    // ---- the async RF readers (the read ports free now), before issue ----
    RfReadDecision decide_rf_reads(Wires& w);
    [[nodiscard]] std::optional<RfReadRequest> lsu_store_read_request(const Wires& w) const;
    // ---- issue (the scheduler): admission, then acceptance ----
    void decide_issue(Wires& w);
    IssueAttempt try_issue(Warp& warp, Wires& w);
    [[nodiscard]] IssuePlan issue_plan(const Instruction& inst) const;
    [[nodiscard]] ResourceView resource_view(
        const Warp& warp, const Instruction& inst, const IssuePlan& plan, const Wires& w) const;
    IssueAttempt refuse(Refusal refusal);
    void accept(Warp& warp, const Instruction& inst, const IssuePlan& plan, Wires& w);
    void accept_arithmetic(Warp& warp, const Instruction& inst, const IssuePlan& plan, Wires& w);
    // An accepted memory / engine / mbar instruction: its metadata work now
    // (depbar, the sanitizer's destination and reader records), its read
    // round's bank use, its entry into the dispatch pipeline.
    void accept_for_dispatch(Warp& warp, const Instruction& inst, const IssuePlan& plan, Wires& w);
    [[nodiscard]] static bool needs_dispatch(const Instruction& inst, uint64_t active_mask);
    void panic_if_malformed(const Warp& warp, const Instruction& inst) const;
    void check_issue_hazards(const Warp& warp, const Instruction& inst) const;
    void check_issue_latency(const Warp& warp, const Instruction& inst) const;
    // ---- the RF read stage: this cycle's fixed samples ----
    void step_rf_reads(Wires& w);

    // ---- register drivers: next state from Q + wires; any order ----
    using Driver = void (Machine::*)(const Wires&);
    void drive_front_end(const Wires& w);
    void drive_fetch(const Wires& w);
    void drive_issue_branch(const Wires& w);
    void drive_depbar(const Wires& w);
    void drive_wait_slots(const Wires& w);
    void drive_store_slots(const Wires& w);
    void drive_mbar_probe(const Wires& w);
    void drive_simt_rmw(const Wires& w);
    void drive_rmw_engine(const Wires& w);
    void drive_a_read_pipe(const Wires& w);
    void drive_write_pipes(const Wires& w);
    void drive_shmem_arbiters(const Wires& w);
    void drive_shmem_to_ram(const Wires& w);
    void drive_tma_event(const Wires& w);
    void drive_mma_event(const Wires& w);
    void drive_mma(const Wires& w);
    void drive_acc_transfers(const Wires& w);
    void drive_tma_queue(const Wires& w);
    void drive_tma_transactions(const Wires& w);
    void drive_tma_sector(const Wires& w);
    void drive_tma_store_registers(const Wires& w);
    void drive_lsu_commands(const Wires& w);
    void drive_assembly_pool(const Wires& w);
    void drive_ingress(const Wires& w);
    void drive_egress(const Wires& w);
    void drive_read_records(const Wires& w);
    void drive_free_id_tags(const Wires& w);
    void drive_writebacks(const Wires& w);
    void drive_load_slots(const Wires& w);
    void drive_acc_ld_pipe(const Wires& w);
    void drive_rf_write_policy(const Wires& w);
    void drive_rf_read_policy(const Wires& w);
    void drive_issue_read_banks(const Wires& w);
    void drive_dispatch_pipe(const Wires& w);
    void drive_lsu_read_pipe(const Wires& w);
    void run_drivers(const Wires& w);

    // ---- helpers ----
    // Sanitizer argument builders: pure reads of hardware state whose results
    // go only to sanitizer_.
    [[nodiscard]] IssueSite site(const Warp& warp, const Instruction& inst) const;
    [[nodiscard]] IssueSite site(const DispatchEntry& op) const;
    [[nodiscard]] static std::vector<uint32_t> access_addresses(
        const std::vector<std::vector<LaneAccess>>& rounds);
    [[nodiscard]] static std::vector<std::vector<uint32_t>> round_addresses(
        const std::vector<std::vector<LaneAccess>>& rounds);
    [[nodiscard]] std::vector<uint32_t> lane_values(const Warp& warp, uint8_t reg) const;
    [[nodiscard]] std::optional<std::pair<uint32_t, uint32_t>> mma_operand_span(
        uint32_t base, bool a_operand, const MmaShape& shape) const;
    [[nodiscard]] std::vector<uint8_t> shmem_after_edge(uint32_t address, uint32_t size, const Wires& w) const;
    [[nodiscard]] std::array<uint8_t, tma_descriptor_bytes> tma_descriptor_at_accept(
        uint32_t address, const Wires& w) const;
    [[nodiscard]] std::optional<uint64_t> pending_tma_completion_mbar(uint32_t mbar, const Wires& w) const;

    // The RF write calendar. Issue books its fixed write (at issue +
    // issue_to_port), a future cycle; the async writers take the current
    // write port's free ports (decide_rf_writes).
    [[nodiscard]] bool write_slot_free(uint8_t reg, uint64_t at_cycle) const;
    void reserve_write(uint8_t reg, uint64_t at_cycle);

    // depbar: completions recorded as they happen; issue sees the counters'
    // Q only (a completion presented at t is visible to issue at t + 1).
    void record_depbar_completion(Wires& w, uint32_t warp, uint8_t bar, const std::string& underflow);
    [[nodiscard]] uint32_t depbar_after_completions(const Wires& w, uint32_t warp, uint8_t bar) const;
    void admit_depbar(Wires& w, uint32_t warp, uint8_t bar);
    // mbar wait slots: a warp parked in one (Q); the space issue may take.
    [[nodiscard]] bool wait_slot_completed(const Wires& w, uint32_t slot) const;
    [[nodiscard]] std::optional<uint32_t> waiting_slot_of(const Wires& w, uint32_t warp) const;
    [[nodiscard]] std::optional<uint32_t> free_mbar_wait_slot(const Wires& w) const;
    void complete_mbar_try_wait(Wires& w, uint32_t slot, bool result);

    // Service slots and the warp's shmem order, as issue sees them after this
    // cycle's grants (look-ahead (c), reuse (d)).
    [[nodiscard]] std::optional<uint32_t> free_load_slot(const Wires& w) const;
    [[nodiscard]] std::optional<uint32_t> free_store_slot(const Wires& w) const;
    [[nodiscard]] uint32_t free_load_slot_count(const Wires& w) const;
    [[nodiscard]] uint32_t free_store_slot_count(const Wires& w) const;
    [[nodiscard]] uint32_t free_wait_slot_count(const Wires& w) const;
    // The warp has an instruction in the dispatch pipeline that will read /
    // write shmem / probe an mbar.
    [[nodiscard]] bool warp_dispatching(uint32_t warp, bool reads, bool writes, bool probes) const;
    [[nodiscard]] bool load_rounds_left(const Wires& w, uint32_t slot) const;
    [[nodiscard]] bool store_slot_released(const Wires& w, uint32_t slot) const;
    [[nodiscard]] bool warp_unserviced_read(const Wires& w, uint32_t warp) const;
    [[nodiscard]] bool warp_unserviced_write(const Wires& w, uint32_t warp) const;
    [[nodiscard]] bool own_shmem_at_ram(const Wires& w, uint32_t warp) const;
    [[nodiscard]] bool probe_in_flight(uint32_t warp) const;
    [[nodiscard]] uint64_t full_mask() const;

    // shmem requesters and their data paths.
    [[nodiscard]] bool store_pause() const;
    [[nodiscard]] bool rmw_requester_held() const;
    [[nodiscard]] bool rmw_awaits_read(const Wires& w) const;
    [[nodiscard]] std::optional<uint32_t> load_slot_in_service() const;
    [[nodiscard]] std::optional<uint32_t> store_slot_in_service() const;
    [[nodiscard]] bool store_slot_waiting() const;
    [[nodiscard]] bool tma_read_in_service() const;
    [[nodiscard]] AReadGrant a_read_round_robin(const Wires& w) const;
    ReadBeat load_round(uint32_t slot);
    ReadBeat tma_read_round(const TmaPortRequest& request, bool start, uint32_t store_register);
    WriteBeat store_round(uint32_t slot);
    WriteBeat rmw_update(const ReadBeat& read);
    [[nodiscard]] static bool is_atomic_operation(MbarRmwOperation operation);
    [[nodiscard]] WriteBeat atomic_update(const MbarRmwRequest& request, uint32_t old_word) const;
    WriteBeat placement_round(const TmaPortRequest& request, bool start);
    void advance_tma_read(Wires& w);
    void fill_tma_store_register(const ReadBeat& beat, Wires& w);
    [[nodiscard]] std::optional<uint32_t> tma_store_register_free(const EgressWrite& egress) const;
    void deliver_probe(const ReadBeat& beat, Wires& w);
    void count_engine_wait_on_other(const Wires& w);
    [[nodiscard]] MbarEngine engine_to_claim(bool tma_held, bool mma_held) const;
    [[nodiscard]] static std::string rmw_label(const ActiveMbarRmw& active);
    [[nodiscard]] std::vector<uint32_t> tma_round_addresses(const TmaPortRequest& request, std::size_t round) const;
    void wake_mbar_waiters(Wires& w, uint32_t byte_address, uint8_t new_phase);

    // MMA, acc.
    [[nodiscard]] bool acc_engine_available(const Wires& w, bool load) const;
    void advance_mma(MmaPipeline& mma, Wires& w);
    [[nodiscard]] std::optional<MmaFetch> next_mma_step(MmaPipeline& mma, Wires& w);
    [[nodiscard]] MmaRead grant_mma_round(MmaPipeline& mma);
    void sample_mma_read(MmaPipeline& mma, MmaRead& read, Wires& w);
    void enter_mma_core(MmaPipeline& mma, const MmaStep& step);
    void sample_mma_c(MmaResult& result);
    [[nodiscard]] std::size_t tmem_word_index(
        const Kernel::Accumulator& acc, uint32_t row, uint32_t col) const;
    void execute_arithmetic(ArithmeticRead& pending, Wires& w);

    // TMA.
    void apply_tma_requests(TmaQueue& tma, Wires& w);
    void apply_tma_walker(TmaQueue& tma, Wires& w);
    void complete_tma_frontend_read(TmaQueue& tma, const TmaPortRequest& request);
    void complete_tma_load_write(TmaQueue& tma, const PlacementEnd& placed);
    void complete_tma_store_capture(TmaQueue& tma, uint64_t command);
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
    // The warp's FIFO head (Q), decoded; "fell off kernel" if its pc is past
    // the kernel's words (a word fetched there is reported only here).
    [[nodiscard]] Instruction head_instruction(const Warp& warp) const;
    [[nodiscard]] uint32_t head_pc(uint32_t warp) const { return front_end_.cur().at(warp).pc; }
    [[nodiscard]] std::vector<uint8_t> source_registers(const Instruction& inst) const;
    [[nodiscard]] std::optional<uint8_t> destination_register(const Instruction& inst) const;
    void check_register_hazards(const Warp& warp, const Instruction& inst) const;
    // A dispatched instruction's operand: every lane of one register, as its
    // RF read latched it at its sample.
    [[nodiscard]] std::vector<uint32_t> capture(const DispatchEntry& op, uint8_t reg) const;
    // A conditional branch's predicate, every lane, read from the predicate
    // file in the issue register's cycle (resolve_branch).
    [[nodiscard]] std::vector<uint32_t> capture_predicate(const Warp& warp, uint8_t predicate) const;
    // A declared-uniform operand: lane 0's value (the sanitizer checks the rest).
    [[nodiscard]] uint32_t uniform_reg(const DispatchEntry& op, uint8_t reg, const std::string& name) const;
    [[nodiscard]] uint32_t mbar_address_operand(const DispatchEntry& op) const;
    [[nodiscard]] MbarState load_mbar(uint32_t byte_address) const;
    [[nodiscard]] uint32_t& rf(Warp& warp, uint32_t lane, uint32_t reg);
    [[nodiscard]] uint32_t rf(const Warp& warp, uint32_t lane, uint32_t reg) const;
    // SIMT shmem bank rounds (lds/sts by bank decode; lds.uni: one access),
    // from the captured operands.
    [[nodiscard]] std::vector<std::vector<LaneAccess>> build_shmem_rounds(const DispatchEntry& op, bool load) const;
    [[nodiscard]] std::vector<std::vector<LaneAccess>> build_uniform_shmem_round(const DispatchEntry& op) const;
    [[nodiscard]] uint32_t shmem_address(const DispatchEntry& op, uint32_t base) const;
    void apply_image(
        std::vector<uint8_t>& memory,
        const std::vector<ImageSegment>& image,
        const std::string& name);
    Writeback register_writeback(
        const Warp& warp, uint8_t dst, uint64_t port_cycle, std::vector<uint32_t> values, std::string reason) const;
    Writeback predicate_writeback(
        const Warp& warp, uint8_t dst, uint64_t port_cycle, std::vector<uint32_t> values) const;
    [[noreturn]] void panic(const Warp& warp, const Instruction& inst, const std::string& reason) const;
    [[noreturn]] void panic(const DispatchEntry& op, const std::string& reason) const;
    [[noreturn]] void mbar_rmw_panic(
        const MbarRmwRequest& request, const std::string& reason) const;

    // ---- gmem controller + LSU (src/lsu.cpp) ----
    [[nodiscard]] std::size_t gmem_outstanding_sectors() const;
    [[nodiscard]] std::optional<std::size_t> select_controller_entry() const;
    void tick_gmem_controller(Wires& w, bool& return_ready_low);
    [[nodiscard]] std::optional<uint64_t> egress_slot_request(const EgressSlot& slot, const Wires& w) const;
    void finish_gmem_write(GmemSectorRequest request);
    void deliver_write_acks(Wires& w);
    [[nodiscard]] static bool is_store(LsuKind kind);
    LsuCommand plan_lsu_command(const DispatchEntry& op) const;
    [[nodiscard]] std::optional<uint32_t> free_assembly_register(const Wires& w) const;
    [[nodiscard]] uint32_t free_assembly_register_count(const Wires& w) const;
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

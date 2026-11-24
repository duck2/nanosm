#pragma once

#include "qtxc/qtxc.hpp"
#include "image.hpp"
#include "limits.hpp"
#include "mbar.hpp"
#include "profile.hpp"
#include "machine/blackbox.hpp"
#include "machine/fifo.hpp"
#include "machine/fixed.hpp"
#include "machine/memory.hpp"
#include "machine/pipe.hpp"
#include "machine/reg.hpp"
#include "machine/regs.hpp"
#include "timing.hpp"
#include "machine/sanitizer.hpp"
#include "machine/state.hpp"
#include "machine/trace.hpp"
#include "tma_descriptor.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdint.h>
#include <deque>
#include <map>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

// class Machine: the whole simulated SM, one class whose member functions are
// in src/machine/*.cpp (the file map is at the top of machine.cpp). This
// header, in order:
//   - test seams and hooks (ControllerTestOrder, LsuEdge)
//   - admission: IssuePlan, ResourceView, Refusal (free functions of an
//     instruction's metadata and resource flags)
//   - class Machine, public: construction, tick / run, the host's and the
//     tests' read accessors
//   - "Machine structures": what each unit's state is made of (wait slots,
//     front end, divergence, the dispatch entry, load and store slots, the
//     tensor core, the memory controller and the load / store unit, the TMA)
//   - "Register contents that change together": the structs one Reg holds
//   - "Wires": this cycle's decisions, one struct per decision
//   - "State": the memories, then every Reg, then unit-internal state
//   - "tick()": the decision functions in dependency order, the register
//     drivers, the helpers
namespace nanosm {

// Test seam for the placeholder gmem controller's return order. The default
// (fifo) serves the queue strictly in acceptance order, which is a legal
// special case of the boundary contract (in order within a read ID, free
// across IDs). Every load is on the one ordered LSU stream and no read has a
// free ID, so there is no order to try for reads on free IDs.
// illegal_ordered_reorder breaks the contract on purpose (serves the second
// entry of an ordered stream before the first) so the SM's tripwire can be
// tested. Never used by kernels or launches.
// interleave_ordered_streams is legal too: when the front is an ordered-
// stream read and the last read served was of the same stream, it serves the
// oldest read of a DIFFERENT ordered stream from the leading run of reads,
// so returns interleave across stream IDs while every ID stays in order.
enum class ControllerTestOrder : uint8_t {
    fifo,
    illegal_ordered_reorder,
    interleave_ordered_streams,
};

// Test hook: one event per datum movement across a declared edge between
// the bus, the FIFOs, the buffers and the register file. Recorded only when
// enabled; never consulted by
// the machine. Field meaning per edge is listed with each enumerator.
enum class LsuEdge : uint8_t {
    bus_read_beat,     // bus -> ingress FIFO. id = request, aux = byte offset in sector
    ingress_read,      // ingress FIFO read port. id = request, aux = byte offset
    tma_reg_write,     // ingress -> TMA sector register. id = request, aux = byte offset
    buffer_write,      // ingress -> the load buffer. id = command, aux = request
    rf_write_beat,     // the load buffer -> RF. id = command, aux = register
    tma_place_round,   // TMA sector register -> shmem (port-B write round, at its grant). id = request, aux = round
    rf_read_beat,      // RF -> the store buffer. id = command, aux = register
    tma_store_round,   // shmem -> TMA store sector register (A-read round, at its return). id = tma command, aux = round
    egress_write,      // the store buffer / TMA store register -> egress FIFO. id = slot, aux = 0 TMA / 1 LSU
    write_request,     // a write's request accepted by the controller. id = request, aux = slot
    bus_write_beat,    // egress FIFO -> bus. id = request, aux = byte offset in sector
    write_ack,         // the write's acknowledgement. id = request
    // Not a movement: a zero fill found the placement tickets' write port
    // taken by a sector the ingress completed, and waits. id = tma command
    tma_zero_fill_wait,
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
// Admission. Whether an instruction issues this
// cycle is decided from its metadata and resource state only -- never from
// register, predicate or memory contents. depbar_wait and structural_refusal
// are free functions of the decoded instruction, its IssuePlan (metadata and
// profile) and a ResourceView (flags); no argument can hold an operand value.
// Operands are captured after acceptance (Machine::accept).
// ===========================================================================

// What an instruction books if it is accepted, from metadata and the profile.
struct IssuePlan {
    std::vector<uint8_t> reads;         // its one RF read round: each register sampled at issue + issue_to_sample (src/timing.hpp)
    std::vector<uint8_t> uniform_reads; // the same round's uniform registers (distinct; at most the URF's two read ports)
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
    bool tma_queue_room = false;     // counting the commands the TMA operand read holds
    bool urf_ports_held = false;     // a tma's or an mma.async's uniform read has rounds still to start after this cycle
    bool mma_queue_room = false;
    bool acc_engine_free = false;    // the acc.ld (load) or acc.st (store) engine takes this transfer
    // shmem, as this cycle's grants leave it:
    bool unserviced_read = false;    // the warp holds a load slot with rounds not yet granted
    bool unserviced_write = false;   // ... a store slot, or a SIMT RMW (mbar op or atomic) whose write is not yet granted
    bool own_shmem_at_ram = false;   // fence.sh: every shmem access of the warp has reached the RAM by now
    bool load_slot_free = false;     // an lds's slot, or an atomic's return slot
    bool store_slot_free = false;
    bool probe_register_free = false;
    bool simt_rmw_register_free = false; // the SIMT RMW register is empty after this cycle
    bool lsu_slot_free = false;
    bool store_buffer_free = false;  // a scalar stg: no store still has data to move into the egress FIFO
};

enum class Refusal : uint8_t {
    rf_read_inhibited,
    urf_read_ports_held,
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
    lsu_store_buffer,
};

// The first depbar in the instruction's wait mask that is still nonzero.
std::optional<uint8_t> depbar_wait(const Instruction& inst, const std::array<uint32_t, 6>& depbar);
// The first structural reason the instruction cannot be accepted now.
std::optional<Refusal> structural_refusal(const Instruction& inst, const IssuePlan& plan, const ResourceView& view);

// One RF bank's choice among ready async writers:
// scan from `cursor`, skip holders not ready for this bank, grant up to
// `ports`; the cursor moves past the last holder granted, and only then.
struct RoundRobinGrant {
    std::vector<uint32_t> granted;  // holder indices, in grant order
    uint32_t cursor_after = 0;
};
RoundRobinGrant round_robin_grant(const std::vector<uint8_t>& ready, uint32_t cursor, uint32_t ports);

// The shmem byte address the MMA reads operand element (outer, k) of a
// command of `shape` from, its operand at `base` with row spacing `spacing`
// (outer: m for A, n for B): the operand fetch's own addressing (src/machine/tensor_core.cpp
// mma_a_address / mma_b_address, in the profile's layout), named here so
// tests can hold a TMA tile against it.
uint32_t mma_operand_address(const Config& config, const MmaShape& shape, uint32_t base, uint32_t spacing,
    bool a_operand, uint32_t outer, uint32_t k);

class Machine {
public:
    Machine(Config config, Kernel kernel, Launch launch, TraceOptions trace_options = {});
    // The registers are registered by address (src/machine/reg.hpp): never copied or moved.
    Machine(const Machine&) = delete;
    Machine& operator=(const Machine&) = delete;

    void set_controller_test_order(ControllerTestOrder order) { controller_test_order_ = order; }
    void enable_edge_probe() { edge_probe_enabled_ = true; }
    // Test seam (the driver-shuffle regression check): run the register
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
    [[nodiscard]] std::size_t mma_queue_depth() const { return mma_.cur().slots.size(); }
    // Test seams (Q, between ticks): the owner of the load buffer (0) or the
    // store buffer (1) (an LSU command id: the buffer holds its slot), its
    // valid bits and their count.
    [[nodiscard]] std::optional<uint64_t> lsu_buffer_owner(uint32_t index) const {
        const auto& owner = lsu_buffers_.at(index).cur().owner;
        if (!owner.has_value()) return std::nullopt;
        return lsu_commands_.cur().in_slot(*owner).id;
    }
    [[nodiscard]] std::vector<uint8_t> lsu_buffer_valid(uint32_t index) const {
        return lsu_buffers_.at(index).cur().valid;
    }
    [[nodiscard]] uint32_t lsu_buffer_valid_count(uint32_t index) const {
        return lsu_buffers_.at(index).cur().valid_count;
    }
    [[nodiscard]] uint32_t lsu_buffer_group(uint32_t index) const { return lsu_buffers_.at(index).cur().group; }
    [[nodiscard]] std::vector<uint32_t> lsu_buffer_words(uint32_t index) const {
        return lsu_buffers_.at(index).cur().words;
    }
    // Every clocked register's and memory's declared size, in declaration order.
    [[nodiscard]] std::vector<StateSize> state_sizes() const { return clocked_.sizes(); }
    // The state report (src/machine/state_report.cpp): declared bits by unit,
    // flip-flops, memories and FIFO storage, the black boxes, and the totals.
    [[nodiscard]] std::string state_report() const;
    // Every black box's contract and declared cost (src/machine/blackbox.hpp).
    [[nodiscard]] const std::vector<BlackBoxDeclaration>& black_boxes() const { return black_boxes_; }    // The primitives that cannot declare their bits yet, by name, each with
    // what is still in it.
    [[nodiscard]] std::vector<std::string> state_not_yet_declared() const {
        std::vector<std::string> names;
        for (const auto& entry : not_yet_declared()) names.push_back(entry.first->name());
        return names;
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
    // Test seams (Q, between ticks): a warp's active mask and its divergence
    // stack's depth. The mask is set only by
    // the machine (ssy / divergent bra / sync); there is no seam to set it.
    [[nodiscard]] uint64_t active_mask(uint32_t warp) const { return divergence_.cur().at(warp).active; }
    [[nodiscard]] uint32_t divergence_depth(uint32_t warp) const { return divergence_.cur().at(warp).depth; }
    // Test seams (Q, between ticks): a warp's front end -- its buffer's valid
    // packets, its pc (the next instruction's) and fetch pc (the next packet
    // to read) -- and the warp whose packet is at the instruction memory's
    // output now (a read in flight), if any.
    [[nodiscard]] std::size_t instruction_fifo_packets(uint32_t warp) const { return front_end_.cur().at(warp).buffer.size(); }
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
    [[nodiscard]] bool rf_write_owed(uint32_t bank) const { return rf_write_policy_.cur().at(bank).owed != 0; }
    [[nodiscard]] uint32_t rf_write_cursor(uint32_t bank) const { return rf_write_policy_.cur().at(bank).cursor; }
    // Test seams (Q, between ticks): the RF read decision's state per bank,
    // and the issue register's bank use (reads its round samples this cycle).
    [[nodiscard]] bool rf_read_owed(uint32_t bank) const { return rf_read_policy_.cur().at(bank).owed != 0; }
    [[nodiscard]] uint32_t rf_read_cursor(uint32_t bank) const { return rf_read_policy_.cur().at(bank).cursor; }
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
    [[nodiscard]] uint32_t tmem_word(uint32_t word) const {
        return tmem_.peek(word % config_.tmem_fine_banks, word / config_.tmem_fine_banks)[0];
    }
    // Test seam (between ticks): uniform register `reg` of warp `warp`.
    [[nodiscard]] uint32_t read_ureg(uint32_t warp, uint32_t reg) const;

    void tick();
    void run(uint64_t cycle_limit = 100000);

    [[nodiscard]] bool done() const;
    [[nodiscard]] uint64_t cycle() const { return cycle_; }
    [[nodiscard]] const TraceRecorder& trace() const { return *trace_; }
    // Shared memory's shmem.bytes bytes in address order (an observer's copy;
    // the memory's last row may hold more, which no address reaches).
    [[nodiscard]] std::vector<uint8_t> shmem() const {
        return std::vector<uint8_t>(shmem_.contents().begin(),
            shmem_.contents().begin() + static_cast<std::ptrdiff_t>(shmem_bytes()));
    }
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
    // Machine structures: what each unit's state is made of
    // =======================================================================

    // One register's words, a word per lane (core.lanes of them).
    using LaneWords = Bounded<uint32_t, limits::lanes>;

    struct MbarWaitSlot {
        uint32_t warp = 0;
        uint32_t byte_address = 0;
        uint8_t parity = 0;
        uint8_t predicate_dst = 0;
        bool parked = false;
        uint64_t deadline_cycle = 0;
    };

    // Issue-internal warp state (only the scheduler reads or writes it). The
    // depbar counters and the mbar wait slots, which other units complete,
    // are registers (below), and so are the warp's front end (its PCs,
    // instruction buffer and halted bit, WarpFrontEnd), which fetch reads,
    // and its predicate file (predicates_). Its vector and uniform registers
    // are in the register files' memories (rf_, urf_).
    struct Warp {
        uint32_t id = 0;
        // (The warp's active mask is a register: divergence_, below.)
        // The stall count's end: the warp issues again from this cycle. Only
        // the scheduler reads it (before it accepts anything) and writes it
        // (at its acceptance, for the warp it accepted), so the in-place write
        // is the register's next value; nothing else -- fetch included --
        // reads it.
        uint64_t stall_until = 0;
        // RAW/WAW/WAR horizons (no scoreboard exists) live in the Sanitizer.
    };

    // One warp slot's front end, one
    // of core.warps. Its instruction buffer holds up to
    // core.instruction_buffer_packets fetch packets: a packet is the two
    // 64-bit words at an aligned 16-byte address, read together. Only valid
    // entries are held, head first (the valid bits are a thermometer,
    // buffer.size() is the occupancy). The warp's next instruction is the head
    // packet's word that pc names: pc bit 3 selects the half, so no other
    // per-warp state says which half is next (a branch into a packet's second
    // half starts there; its first half is never used). Issuing the head's
    // second half pops the packet; a landing fetch appends; both in one
    // next-state update, with the accepted word captured by the issue
    // register at that same edge. Consecutive entries are consecutive
    // packets: fetch is sequential and a taken branch clears the buffer. Byte
    // PCs (32 bits; the instruction memory's address width in hardware).
    // While the warp runs, fetch_pc = pc rounded down to 16 + 16 x (valid
    // packets + its read in flight): step_front_end checks it every cycle.
    using FetchPacket = std::array<uint64_t, 2>;
    // The instruction buffer: a few registers in a row (two packets at the
    // canonical profile), not a memory. Its operations are the front end's:
    // the head packet is read by issue, popped when its second half issues;
    // a landing fetch appends behind the valid packets; a redirect or a halt
    // flushes it. All of them in one next-state calculation per warp
    // (step_front_end).
    struct InstructionBuffer {
        Bounded<FetchPacket, limits::instruction_buffer_packets> packets; // head first
        [[nodiscard]] std::size_t size() const { return packets.size(); }
        [[nodiscard]] bool empty() const { return packets.empty(); }
        [[nodiscard]] const FetchPacket& head() const { return packets.front(); }
        void pop_head() { packets.pop_front(); }
        void append(const FetchPacket& packet) { packets.push_back(packet); }
        void flush() { packets.clear(); }
        bool operator==(const InstructionBuffer&) const = default;
    };
    struct WarpFrontEnd {
        InstructionBuffer buffer;      // fetched packets, head first
        uint32_t pc = 0;               // the warp's next instruction (in the head packet when it is valid)
        uint32_t fetch_pc = 0;         // the next packet to read (a multiple of 16)
        bool halted = false;           // a halt was accepted (or the slot was not launched): no fetch, no issue
        bool operator==(const WarpFrontEnd&) const = default;
    };

    // The instruction memory's read in flight (instruction_read_cycles = 1,
    // src/timing.hpp): the packet address it sampled at the grant's edge, and the
    // tag beside it -- the warp the packet is for (valid + 4 bits at 16
    // warps). The packet is at the memory output (128 bits) in the cycle this
    // register holds it. No stage stalls: the credit check guarantees its
    // FIFO has room.
    struct FetchRead {
        uint32_t warp = 0;
        uint32_t pc = 0; // the address sampled (32-bit byte pc)
        bool operator==(const FetchRead&) const = default;
    };

    // The issue register's control part: a control instruction accepted last
    // cycle -- bra, bra.uni, ssy or sync. It resolves now (src/timing.hpp
    // issue_to_branch_resolve): a conditional branch's predicate is read (every
    // lane of it for a divergent bra), the one target adder adds the
    // instruction's pc + 8 and its offset, and the divergence stack is pushed
    // or popped. A branch's or a sync's warp neither issues nor fetches
    // meanwhile; an ssy's does (it changes neither pc nor mask).
    struct IssuedBranch {
        uint32_t warp = 0;
        uint32_t pc = 0;         // the instruction's own pc (32 bits)
        DecodedInstruction instruction; // opcode, predicate, invert, offset
    };

    // The divergence stack. Every entry is
    // the same thing, a (mask, pc) pair, whoever pushed it (an ssy: the
    // active mask and its label; a divergent bra whose lanes went both ways:
    // the taken lanes and its target); a sync pops one and switches to it.
    // 48 bits as held: pc (a 32-bit byte pc; its low three bits are always
    // zero) and mask (core.lanes bits, 16).
    struct DivergenceEntry {
        uint32_t pc = 0;    // where the warp goes when it is popped
        uint64_t mask = 0;  // the lanes it makes active when popped
        bool operator==(const DivergenceEntry&) const = default;
    };
    // Per warp slot (core.warps): the active mask (core.lanes bits; every
    // per-lane writer's lane enables), the stack's depth (entries held,
    // 0..core.divergence_stack_entries: 4 bits at 8) and a registered copy of
    // the top entry (48 bits; meaningless at depth 0), so a sync resolves in
    // the issue register's cycle without reading the stack memory.
    struct WarpDivergence {
        uint64_t active = 0;
        uint32_t depth = 0;
        DivergenceEntry top;
        bool operator==(const WarpDivergence&) const = default;
    };
    // The stack memory's read in flight: a pop presented the address of the
    // new top (its warp, entry depth - 1) at the resolve cycle's edge; the
    // entry is at the memory output the next cycle and the top copy takes it
    // at that edge (valid + 4-bit warp; the address sits in the memory's own
    // address register). The warp's next stack operation resolves two cycles
    // after the pop at the soonest, so it always sees the refilled copy.
    struct DivergenceRefill {
        uint32_t warp = 0;
        uint32_t index = 0;
        bool operator==(const DivergenceRefill&) const = default;
    };

    // The fixed-latency units (src/timing.hpp): an instruction executes at
    // issue + issue_to_operands and its result is at its unit's output E
    // cycles later, the cycle of the RF write port it booked (a setp's: the
    // predicate file's input). Each unit has one E -- the ALU
    // core.alu_latency_cycles (setp, mov, sread and ur2r included), the FPU
    // core.fadd_latency_cycles, the special-function unit its occupancy - 1 +
    // sfu.latency_cycles -- so each is a black box of latency E
    // (src/machine/blackbox.hpp; its contract and cost are in arith.cpp): the
    // result of an instruction it takes when it executes is in the black box's
    // output register E cycles later, at its port.
    // BLACK BOX SHARED: arithmetic unit order
    // (The units' black boxes are arithmetic_black_boxes_ in this order.)
    enum class ArithmeticUnit : uint8_t { alu, fpu, sfu };
    static constexpr std::size_t arithmetic_units = 3;
    // END BLACK BOX SHARED: arithmetic unit order
    struct ArithmeticResult {
        uint32_t warp = 0;
        uint8_t dst = 0;           // its register, or a setp's predicate
        bool predicate = false;    // a setp: the predicate file takes it; no RF write port
        LaneWords values;
        uint64_t lane_enables = 0; // the issue cycle's active mask AND the write mask
    };

    // One write in an RF write port's pipeline (one pipeline per write port:
    // rf.banks x rf.write_ports_per_bank of them, rf.write_depth_cycles
    // stages). It enters when it is PRESENTED at the port -- a fixed result
    // at the cycle it booked, an async holder at its grant -- and is at the
    // RAM in the last stage, where it lands at the clock edge (src/timing.hpp).
    // A load's last write counts its depbar down there (the waiting warp sees
    // it from the next cycle, the write's visibility); an async result's
    // registers become visible to the sanitizer then too (a fixed result's
    // visibility is known at its issue).
    struct RfWrite {
        uint32_t warp = 0;
        uint8_t reg = 0;
        LaneWords values;
        uint64_t lane_enables = 0;
        std::optional<uint8_t> clear_depbar;
        Bounded<uint8_t, limits::v8_registers> visible_regs; // (S)
    };

    // A predicate at the predicate file's input now (a setp at its port
    // cycle, a usetp at the URF write port's, a probe's or a try-wait's
    // result): the file is flip-flops and has it from the next cycle, in its
    // enabled lanes.
    struct PredicateWrite {
        uint32_t warp = 0;
        uint8_t predicate = 0;
        uint64_t bits = 0;
        uint64_t lane_enables = 0;
    };

    // A fixed-latency instruction in the arithmetic pipeline, from accepted
    // issue until its operands are all at the RF output (issue +
    // issue_to_operands: the last stage of arithmetic_reads_), when it
    // executes: its operand latches.
    struct ArithmeticRead {
        uint32_t warp = 0;
        uint32_t pc = 0;                             // (S) PANIC text: the instruction's own pc
        // Its result's lane enables (16 bits): the issue cycle's active mask,
        // ANDed with the instruction's write mask at the sample.
        uint64_t active_mask = 0;
        DecodedInstruction instruction;
        Bounded<uint8_t, limits::round_reads> reads;                  // the round's registers
        Bounded<LaneWords, limits::round_reads> operands; // by read, latched at the sample
        Bounded<uint8_t, limits::round_uniform_reads> uniform_reads;          // ur2r: the round's uniform register
        Bounded<uint32_t, limits::round_uniform_reads> uniform_operands;      // by uniform read, latched at the sample
    };

    // A memory / engine / mbar instruction between accepted issue and its
    // unit: the issue register and the
    // RF read pipeline carry it. Its operands are latched at its RF sample
    // (issue + 1) and reach the unit at DISPATCH (issue + 1 + Lr), where the
    // unit takes it (hand_over). Admission secured
    // the unit's space when it accepted the instruction: every later
    // admission counts the entries here against that space (a credit counter
    // per resource in hardware; dispatch_claims). The pc and the active mask
    // are the issue cycle's; a masked lds / sts / ldg / stg's write mask is
    // ANDed into the active mask at its sample, so from issue + 2 the field
    // holds its lane enables. It is in stage k of dispatch_pipe_ in cycle
    // issue + 1 + k: stage 0 is its sample, the last stage its dispatch.
    struct DispatchEntry {
        uint32_t warp = 0;
        uint32_t pc = 0;
        uint64_t active_mask = 0;
        DecodedInstruction instruction;
        Bounded<uint8_t, limits::round_reads> reads;                  // the round's registers
        Bounded<LaneWords, limits::round_reads> operands; // by read, latched at the sample
        Bounded<uint8_t, limits::round_uniform_reads> uniform_reads;          // the round's uniform registers
        Bounded<uint32_t, limits::round_uniform_reads> uniform_operands;      // by uniform read, latched at the sample
    };

    // One uniform-register write in the URF write pipeline (src/timing.hpp),
    // from its writer's accepted issue at t
    // to its landing. The pipeline has L + Wu stages (urf.issue_to_port_cycles
    // + urf.write_depth_cycles); stage k holds the write in cycle t + 1 + k:
    //   stage 0          the issue register's cycle: its operands are sampled
    //                    (uniform sources from the URF; r2ur's vector source
    //                    from the RF, in the instruction's one read round)
    //   stage Lu         uniform arithmetic: its operands are at the URF
    //                    output and the uniform ALU takes them (its Eu
    //                    stages, then padding to L)
    //   stage Lr         r2ur: lane 0 is at the RF output (padding to L; at
    //                    the canonical depths this is the write port's own
    //                    stage). The RF's read stages hold it until then, so
    //                    the write carries it from the sample on.
    //   stage L - 1      presented at the URF write port (a usetp: its
    //                    predicate at the predicate file; it leaves here)
    //   stage L + Wu - 1 at the RAM: it lands at this cycle's edge
    // One instruction is accepted per cycle, so each stage holds at most one
    // write and the write port sees at most one per cycle: no calendar.
    // Widths: valid, warp (4 bits at 16 warps), the ALU's control (opcode,
    // destination register: 5 bits at 32, a 22-bit immediate), two 32-bit
    // operand latches, the 32-bit value, and the issue cycle's active mask
    // (16 bits: a usetp's lane enables at the predicate file; a URF write
    // ignores it).
    struct UrfWrite {
        uint32_t warp = 0;
        uint32_t pc = 0;                  // (S) PANIC text: the instruction's own pc
        uint64_t active_mask = 0;
        DecodedInstruction instruction;          // the opcode, its registers and immediate
        Bounded<uint32_t, limits::round_uniform_reads> operands;   // from the sample: the uniform sources in operand order, or r2ur's lane 0
        std::optional<uint32_t> value;    // the result, from its execution on
    };

    // A SIMT shmem access's lanes and its bank rounds, in the slot's
    // registers: the lanes that access (the ACTIVE lanes of an lds / sts; an
    // lds.uni's one access is lane 0's), each one's byte address and the round
    // it is in, set once when the slot is filled (round r serves, in every
    // bank, that bank's r-th accessing lane in ascending lane order: the
    // rounds are the most lanes on any one bank), the number of rounds and how
    // many A-read / A-write has granted. A round's lanes are on different
    // banks, so a round is at most one word per bank.
    struct SlotRounds {
        uint64_t lanes = 0;
        std::array<uint32_t, limits::lanes> address{};
        std::array<uint8_t, limits::lanes> round{};
        uint32_t rounds = 0;
        uint32_t granted = 0;
        [[nodiscard]] bool started() const { return granted != 0; }
        [[nodiscard]] bool serviced() const { return granted == rounds; }
        [[nodiscard]] bool in_round(uint32_t lane, uint32_t r) const { return (lanes >> lane & 1) != 0 && round[lane] == r; }
    };

    // A load slot: one lds / lds.uni from admission until
    // the RF write decision grants its write. Secured at admission (a claim
    // in the dispatch pipeline), installed at dispatch with the captured
    // operands, so it is ready for A-read from the next cycle.
    // Serviced when its last round is granted; complete when every round's
    // words are back in `words`, the 64 B gather register -- then it is a
    // ready async RF writer.
    // Or a uniform atomic's RETURN slot, reserved
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
        LaneWords words;
        uint32_t returned = 0;     // rounds whose words are in `words`
        bool engine_filled = false; // an atomic's return slot: one fill from the RMW engine
        [[nodiscard]] bool complete() const { return returned == (engine_filled ? 1 : access.rounds); }
    };

    // A store slot: one sts from admission until its last
    // round is granted, with each accessing lane's word (its 64 B data
    // register). From each grant the A-write pipeline owns that round's
    // addresses, data and (last round) depbar completion for D cycles.
    struct StoreSlot {
        uint32_t warp = 0;
        SlotRounds access;
        std::array<uint32_t, limits::lanes> data{};
        uint8_t depbar = 0;
    };

    // One beat in the acc.ld engine's pipeline: one column of the transfer
    // (sixteen rows, one word per lane, sixteen distinct fine banks), with its
    // destination and completion context. Its tmem words are sampled when it
    // reaches the RAM stage (stage tmem.port_depth - 1), once.
    struct AccLdBeat {
        uint32_t warp = 0;
        uint8_t reg = 0;
        uint32_t address = 0;         // lane 0's logical tmem word (the pointer + the beat's column)
        uint32_t row_words = 0;       // the region's words per row: lane l reads address + l x row_words
        bool sampled = false;
        LaneWords values; // the words, from the tmem sample on
        bool last = false;    // the transfer's final beat: its sample ends the read, its RF write clears the depbar
        uint8_t depbar = 0;
        bool operator==(const AccLdBeat&) const = default;
    };
    // The acc.ld engine's pipeline: tmem.port_depth stages to the tmem sample,
    // then tmem.read_latency_cycles stages to the output, which asks the RF
    // write decision for service. Stage 0 takes the launched read.

    // One acc.st beat between its RF read and its tmem write: read (sampled)
    // at its grant, at the RF output Lr cycles later (where it leaves for
    // tmem), at the tmem RAM tmem.port_depth after that (the last stage of
    // acc_st_pipe_: written at that cycle's edge).
    struct AccStBeat {
        uint32_t column = 0;
        LaneWords values;
    };

    // The transfer an engine is running: the region at `pointer` (a logical
    // tmem word, src/layout.hpp) whose rows are row_words words; lane l moves
    // row l, beat j column j. The sanitizer keys its tmem owner by the engine
    // (acc.ld or acc.st).
    struct AccTransfer {
        bool load = false;
        uint32_t warp = 0;
        uint8_t rf_base = 0;
        uint8_t depbar = 0;
        uint32_t pointer = 0;
        uint32_t row_words = 0;
        uint32_t width = 1;
        uint32_t next_column = 0;   // beats launched (acc.ld: tmem reads; acc.st: RF reads)
        // acc.st: how many of its beats (in flight in acc_st_pipe_) landed.
        // The acc.st engine frees when the last one lands.
        uint32_t landed = 0;
    };

    struct MmaCommand {
        uint64_t id = 0;
        bool commit = false;
        bool init = false;
        MmaShape shape;          // from the opcode (m 32 or 64, n 64, k 8)
        uint32_t warp = 0;
        uint32_t c_pointer = 0;  // urC as read: the result's logical tmem word, rows of n words
        uint8_t depbar = 0;
        uint32_t a_base = 0;     // the operand's first 16-byte piece in its row 0
        uint32_t b_base = 0;
        uint32_t a_spacing = 0;  // bytes from one operand row to the next (the instruction's)
        uint32_t b_spacing = 0;
        uint32_t mbar_address = 0;
        uint32_t next_step = 0;
        uint32_t outstanding_results = 0;
        bool operands_consumed = false;
    };

    // A step from its entry into the core to its result's tmem write. The
    // core's arithmetic is a black box (mma_core_; its contract and cost are
    // in tensor_core.cpp): it takes the step's A and B blocks as the step
    // enters and its C (the tile's sixteen accumulator words, one round of the
    // MMA's tmem read service) at tmem's output, and has the result in its
    // output register result_latency after entry. Beside it, in the open, the
    // step's tag rides a pipeline of result_latency stages: its C read is at
    // the tmem RAM in stage port_depth - 1, and from the last stage the tag
    // and the result enter the write pipeline, tmem.port_depth stages to the
    // tmem RAM (its last stage: written at that cycle's edge), where the
    // command's result retires.
    // The tag: the step's command, its result block (the command's pointer,
    // the block's m0 and n0, rows of n words: mma_block_words) and whether C
    // is zero.
    struct MmaTag {
        uint64_t command = 0;
        uint32_t c_pointer = 0;
        uint32_t m0 = 0;
        uint32_t n0 = 0;
        uint32_t row_words = 0; // the shape's n
        bool init_step = false; // C = 0 instead of the tmem words
    };
    // A result block: core_m x core_n words, row-major.
    using MmaBlock = Bounded<uint32_t, limits::mma_block>;
    struct MmaWrite {
        MmaTag tag;
        MmaBlock values;
    };
    // Inside the core's black box: a step's A and B blocks, from its entry until
    // its C comes.
    struct MmaCoreStep {
        uint64_t entered = 0;
        Bounded<uint16_t, limits::mma_block> a; // core_m x core_k fp16
        Bounded<uint16_t, limits::mma_block> b; // core_n x core_k fp16 (B transposed)
    };

    // One core step: which tile, which K block, and whether it reloads the
    // held A block (hold the 4x8 A block, stream the 8x4 B
    // blocks across N; A is read again only when the walker moves to the
    // next A block). It travels with the step's rounds through port B.
    struct MmaStep {
        uint64_t command = 0;
        uint32_t c_pointer = 0;  // the command's result pointer (rows of shape.n words)
        bool last_step = false;  // the command's last step: its final sample frees the operands
        bool init_step = false;  // C = 0 instead of the tmem value
        bool reload_a = false;   // this step reads its A block (n block 0); otherwise the held one
        uint32_t m0 = 0;
        uint32_t n0 = 0;
        uint32_t k0 = 0;
        uint32_t a_base = 0;
        uint32_t b_base = 0;
        uint32_t a_spacing = 0;
        uint32_t b_spacing = 0;
        MmaShape shape;          // the command's: the dense layout's strides follow it
    };

    // A step's operand elements by position: its A block (core_m x core_k,
    // row-major; read only by a step that reloads A), then its B block (core_k
    // x core_n, k-major). Each one's shared-memory address follows from the
    // step (mma_element_address).
    static constexpr std::size_t mma_step_elements = 2 * limits::mma_block;

    // The step whose port-B rounds are being granted: one round per cycle,
    // from its first round to its last (no preemption). Its bank rounds are
    // set once when it starts: round r reads every bank's r-th distinct 4-byte
    // word (ascending), and each element is in its word's round (the two
    // halves of a word, and A and B elements at one address, share it).
    struct MmaFetch {
        MmaStep step;
        uint32_t rounds = 0;
        uint32_t next_round = 0;
        std::array<uint8_t, mma_step_elements> element_round{};
    };

    // One granted port-B read round, in the read pipeline: D stages to the
    // RAM sample, then Lb stages to the port output (the same D and Lb as
    // port A). Its elements' values are read at the sample.
    struct MmaRead {
        MmaStep step;
        bool last_round = false;
        std::array<bool, mma_step_elements> in_round{};    // the step's elements this round reads
        std::array<uint16_t, mma_step_elements> values{};  // their bits, from the RAM sample on
    };

    // What the RMW engine does with the word(s) it reads: an mbar operation
    // (both mbar words), or a uniform atomic (one
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

    // Field classes: unmarked
    // fields are hardware (routing, completion context); (S) marks a field
    // read only for a PANIC decision or its text; (T) only for the trace.
    // Neither is pipeline width.
    // complete_tx carries no phase in hardware: it changes
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
        DecodedInstruction instruction;        // (S) PANIC text
    };

    // The two engines that present mbar events to the RMW engine.
    enum class MbarEngine : uint8_t { tma, mma };

    // The RMW engine's one operation, from its START to its
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
        DecodedInstruction instruction;        // (S) PANIC text
    };

    // ---- gmem controller (placeholder) + LSU ----
    // Two clients share one request channel: the TMA and the LSU. When both
    // have a request in the same core cycle they alternate at sector
    // boundaries; a client that is not ready never consumes the other's turn.
    // A write's request is issued by its client when its sector is complete
    // and visible at the egress FIFO's request pointer.
    enum class GmemClient : uint8_t { tma, lsu };

    // IDs. Stream 0 is the TMA's reads (ordered, one 32 B sector register);
    // stream 1 is every LSU read -- v8 loads, gathers, ldg.uni -- ordered, no
    // sector register, beats pluck straight into the load buffer. lsu.
    // read_streams is those two. Free IDs are for scatter WRITES only: tags
    // in [read_streams, read_streams + lsu.sectors_in_flight), so that their
    // acknowledgements may return in any order.
    static constexpr uint32_t tma_stream = 0;
    static constexpr uint32_t lsu_stream = 1;   // every LSU read, in request order

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

        uint64_t id = 0;            // simulator request number (trace, probe); the wire is stream + tag
        uint64_t command = 0;       // (T) tma command id or lsu command id
        uint8_t lsu_slot = 0;       // an LSU write: its command's slot (the acknowledgement's owner)
        uint32_t stream = 0;        // read ID; writes: free-ID tag stream for scatters
        uint64_t sector_address = 0;
        uint64_t ready_dram_tick = 0;
        bool load = false;
        Owner owner = Owner::lsu;
        // Reads: the sector as DRAM holds it when its burst starts (filled
        // by the placeholder controller at that moment; gmem.sector_bytes,
        // checked there). The SM never holds it: its beats go on the bus.
        std::vector<uint8_t> data;
        uint64_t byte_enable = 0;   // writes: bit b enables sector byte b (AXI strobes)
        uint64_t egress_slot = 0;   // (T) writes: the running number of the egress slot holding the data
        uint32_t beats = 0;
        uint32_t next_beat = 0;
    };

    // One sector's work for the TMA: a RUN of elements along ONE tile row,
    // contiguous in global memory and inside one gmem sector -- or, for a
    // load's out-of-range elements, a zero fill of at most a sector's bytes of
    // that row. The walker makes it; it is all that travels with the sector
    // (walker plan -> read record -> placement ticket -> placement; or walker
    // plan -> store register). Element i (0 <= i < elements) is at
    //     global   global_address + i x element_bytes
    //     shared   layout(shared_address + i x step): step = element_bytes,
    //              or the stored-row spacing for a transposed tile; layout()
    //              is the identity for a dense tile and swizzle64_shuffle(.,
    //              spacing) for a swizzle64 one (src/layout.hpp)
    // so the run holds its first element's tile coordinate in the form each
    // memory needs (its global and its unshuffled shared address) rather than
    // (x, y): from (x, y) a dense tile's placement would need a y x tile-length
    // multiply. Every byte's place, the placement rounds, the store's byte
    // enables and the useful bytes follow from these fields.
    // Width at the canonical profile (4 commands, 64 KiB shmem, 32 B
    // sectors): command slot 2 + global address 64 + shared address 16 +
    // elements 6 (1..32) + element size 2 + zero fill 1 + layout 1 +
    // transposed 1 + spacing 5 (its log2) = 98 bits.
    // A store's command may leave its slot once its last sector is captured,
    // while its store registers and egress slots still hold its runs; from
    // then on nothing looks its slot up (the egress path needs no owner).
    struct TmaRun {
        uint8_t slot = 0;             // the command's queue slot (2 bits at 4 commands)
        uint64_t command = 0;         // (T) the command's number, for the trace and the sanitizer
        uint64_t global_address = 0;  // element 0's global byte address (a zero fill: 0, unused)
        uint32_t shared_address = 0;  // element 0's unshuffled shared byte address
        uint8_t elements = 0;         // 1 .. sector bytes / element bytes
        uint8_t element_bytes = 0;    // 1, 2 or 4
        bool zero_fill = false;       // a load's out-of-range elements: zeros, no gmem request
        TileLayout layout = TileLayout::dense;
        bool transposed = false;      // swizzle64: element i is in stored row i (step = spacing)
        uint32_t spacing = 0;         // swizzle64: the stored row's bytes, a power of two (16 ..)
    };

    // What the SM remembers about an accepted read until its sector has been
    // consumed: where each returning byte goes. Request metadata, not payload
    // storage. Both read streams return in request order (the controller's
    // contract: in order within a read ID) and each sector's bytes are read
    // out of the ingress FIFO in order, so per stream an entry is added when
    // its read is accepted, only the oldest is ever looked at, and it leaves
    // when its sector's last byte is consumed: two FIFOs, one per stream
    // (tma_reads_, lsu_reads_). A returning word says which stream it is on,
    // which picks the head; one counter per stream says how many bytes of the
    // head's sector have been consumed (the next byte's place in it).
    //   TMA stream  the sector's run (TmaRun: 98 bits; its sector address
    //               follows from it): at most tma.sector_credits entries (a
    //               credit is held from the walk to the placement's end).
    //   LSU stream  LsuRead below: requests run ahead of the load buffer, so
    //               each command slot may have one sector per lane in flight,
    //               plus the tail of a retired gather or ldg.uni (see
    //               drive_read_records): lsu.active_commands x lanes + 1
    //               entries (33 at the canonical profile).
    enum class LsuReadKind : uint8_t { vector, gather, uniform };
    struct LsuRead {
        LsuReadKind kind = LsuReadKind::vector; // 2 bits
        uint8_t slot = 0;           // the command's slot (1 bit at 2 commands)
        uint8_t lane = 0;           // the lane that asked (4 bits)
        uint8_t word_offset = 0;    // gather / uniform: the word's byte offset in the sector (3 bits of word)
        // (T) The command's number. A gather or ldg.uni retires (and frees its
        // slot) once its word is in, while the rest of its sector is still
        // read and dropped; meanwhile its slot may hold a new command. The
        // tail's bytes look nothing up in the slot; only the trace's first
        // return is stamped, by this number, and finds nothing.
        uint64_t command = 0;
    };

    // One bus beat as the controller delivers it this cycle (a wire): bus ->
    // ingress FIFO.
    struct IngressBeat {
        uint64_t request = 0;       // (T) the simulator's request number
        uint8_t stream = 0;         // the read stream it returns on (tma_stream or lsu_stream)
        uint32_t offset = 0;        // byte offset of this beat in its sector
        std::vector<uint8_t> bytes; // gmem.bus_bytes_per_dram_tick at most (8)
    };

    // One word of the SM's ingress FIFO (a BRAM, simple dual port), as the
    // read port sees it. The FIFO is written a bus beat wide (one beat per
    // DRAM tick: up to ceil(DRAM clock / core clock) beats per core cycle) and
    // read lsu.pluck_words_per_cycle words wide, so its read pointer moves by
    // words and a beat may be read over several cycles. Capacity
    // lsu.ingress_sectors x sector words (64 words, 256 B, at the canonical
    // profile). An entry is readable lsu.ingress_latency_cycles after it is
    // written (never fall-through). Room freed by a read is a separate
    // contract: reusable from the next cycle (tick_gmem_controller).
    // Hardware: the word and its stream, 33 bits.
    struct IngressWord {
        uint32_t word = 0;
        uint8_t stream = 0;         // tma_stream or lsu_stream (1 bit)
        uint64_t request = 0;       // (S, T) the request number: the sanitizer's order check, the edge probe
        uint32_t offset = 0;        // (S) its byte offset in its sector, as the bus sent it
    };

    // A run of words the read port takes this cycle for one consumer: one
    // stream's words inside one bus beat of one sector (a wire).
    struct IngressPiece {
        uint8_t stream = 0;
        uint32_t offset = 0;        // its first byte's place in the sector: the stream head's bytes consumed
        Bounded<uint32_t, limits::sector_bytes / 4> words;
        uint64_t request = 0;       // (S, T) the first word's request number
        uint32_t sent_offset = 0;   // (S) the first word's offset as the bus sent it
        [[nodiscard]] uint32_t bytes() const { return static_cast<uint32_t>(4 * words.size()); }
    };

    // The TMA stream's 32 B sector register (flops). Filled from the ingress
    // FIFO at the pluck rate; when complete, placed into shmem by the port-B
    // write in its bank rounds, which read the register's Q at their grants.
    // Free after the last round's grant (and refillable in that cycle).
    struct TmaSectorRegister {
        std::optional<uint64_t> request;
        uint64_t sector_address = 0;
        Bounded<uint8_t, limits::sector_bytes> data;  // gmem.sector_bytes, sized at construction (32 B)
        uint32_t filled = 0;        // bytes plucked in
        bool complete = false;      // all bytes in; its ticket is in the TMA queue's returns
    };

    // The LSU's two buffers, one on each pipe; no register serves both
    // directions (a store can then never hold the buffer returning data needs).
    //   lsu_buffers_[load_buffer]   ingress FIFO -> RF write port
    //   lsu_buffers_[store_buffer]  RF read port (or the issue read) -> egress FIFO
    // A buffer has an owner only while data is in it:
    //   load   from the first word plucked for a group (a v8 load's lane
    //          group; a gather's or ldg.uni's words) until that group's last
    //          RF write is granted. Filled at the pluck rate.
    //   store  a scalar store from its dispatch (the issue read's data), a
    //          vector store from each RF read beat's landing, until the last
    //          word of that data is pushed into the egress FIFO (at the pluck
    //          rate). NOT until the write acknowledgement.
    // The command slot holds the command (addresses, counters, depbar) to its
    // end: a load's last RF write, a store's last acknowledgement.
    // Why returning data always has somewhere to go: only loads use the load
    // buffer, every LSU read returns on one ordered stream (request order =
    // command order), and the buffer's only exit is the RF write port, which
    // the async-owed rule always opens. So the ingress FIFO always drains
    // and a read never holds the bus for good. A store waiting for egress
    // room (the FIFO full, a TMA store ahead of it) waits only for the bus.
    //
    // What a buffer holds is one GROUP of a command: a fixed set of lanes, so
    // a lane's place in the buffer is wiring (lsu_group_width):
    //   ldg.v8 / stg.v8  lsu.load_group_lanes / lsu.store_group_lanes lanes,
    //                    eight words each: word w of lane l at buffer word
    //                    (l - the group's first lane) x 8 + w. A load group is
    //                    written by EIGHT RF writes (rD + j, the group's
    //                    active lanes enabled, one a cycle); a store group is
    //                    filled by EIGHT RF read beats (rS + j, one a cycle).
    //   ldg (gather)     as many lanes as the load buffer has words (all of
    //                    them when it has sixteen or more), one word each;
    //                    one RF write with those lanes enabled.
    //   ldg.uni          one word, fanned out by its one RF write.
    //   stg (scatter)    every lane's word, from the issue read at dispatch.
    // Sizes: load buffer 8 x lsu.load_group_lanes words (256 bits per lane of
    // the group); store buffer 8 x lsu.store_group_lanes words, at least one
    // word per lane (a scatter's data). Per buffer also: owner (a command
    // slot number), group (4 bits), the group's lane enables (lanes bits),
    // word valid bits, and a beat counter (3 bits).
    struct LsuBuffer {
        std::optional<uint8_t> owner;    // the owning command's slot
        uint32_t group = 0;              // which lane group of the owner
        uint64_t lanes = 0;              // the group's active lanes (a load's RF write enables)
        Bounded<uint32_t, limits::v8_registers * limits::lanes> words;     // the buffer's words (sized at construction)
        Bounded<uint8_t, limits::v8_registers * limits::lanes> valid;      // one bit per word
        uint32_t valid_count = 0;
        uint32_t expected = 0;           // words this group brings; complete when valid_count == expected
        uint32_t beats = 0;              // v8 load: RF writes granted; v8 store: RF read beats landed
        uint32_t beats_total = 0;        // v8: 8; a gather / uniform load: 1
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
    // (lsu.active_commands) from admission until its completion commits; it
    // holds the load or the store buffer only while its data is in it.
    struct LsuCommand {
        uint64_t id = 0;            // (T) the command's number: the trace, the sanitizer, the wires' names
        uint8_t slot = 0;           // the command slot it holds (chosen at its dispatch)
        LsuKind kind = LsuKind::vector_load;
        uint32_t warp = 0;
        uint32_t pc = 0;
        DecodedInstruction instruction;
        uint8_t reg = 0;            // rD (loads) / rS (vector store)
        uint8_t depbar = 0;
        uint32_t buffer = 0;      // load_buffer or store_buffer (by kind; kept for the trace and probes)
        // The instruction's lanes (16 bits; the active mask AND its write mask,
        // as its operands were sampled): an inactive lane makes no request, a
        // load writes only these lanes (a uniform load: all lanes).
        uint64_t lanes = 0;
        // One per ACTIVE lane, in lane order (ldg / stg: the lane's word; ldg.v8
        // / stg.v8: the lane's sector, word offset 0); uniform: one. Hardware:
        // the lanes' addresses latched at admission (16 x 64 bits as held
        // here; a sector address and a word offset each).
        Bounded<LsuSector, limits::lanes> sectors;
        uint32_t stream = 0;        // a load: the ordered LSU read stream (lsu_stream)
        uint32_t requests_total = 0;
        uint32_t requests_emitted = 0;
        uint32_t lanes_written = 0;    // loads: active lanes whose data is written to the RF
        uint32_t group_beats_read = 0; // v8 store: RF read beats of the current group (0..8)
        uint32_t sectors_pushed = 0; // v8 store: lanes (sectors) moved into the egress FIFO
        uint32_t words_pushed = 0; // stores: words of the current sector (v8) / sectors (scalar) in egress
        uint32_t acks = 0;
    };

    // The load / store unit's command slots: lsu.active_commands slots (2),
    // each holding one command from its dispatch until its last
    // acknowledgement or register write, and which is older. A command takes
    // the lowest free slot (as its dispatch cycle's releases leave them).
    // Requests, store data and register reads are chosen oldest first; a
    // command leaves from either slot. State that refers to a command (a
    // buffer's owner, a read record, an egress slot, a read in the store-data
    // pipeline) holds its slot number; this cycle's wires name it by its
    // number (LsuCommand::id), found here by comparing the slots.
    struct LsuCommandSlots {
        std::array<std::optional<LsuCommand>, limits::lsu_active_commands> slots{};
        Bounded<uint8_t, limits::lsu_active_commands> by_age; // occupied slots, oldest first
        [[nodiscard]] std::size_t size() const { return by_age.size(); }
        [[nodiscard]] bool empty() const { return by_age.empty(); }
        // The command in slot `slot`; it must be there.
        [[nodiscard]] const LsuCommand& in_slot(uint8_t slot) const {
            if (slot >= slots.size() || !slots[slot].has_value()) {
                throw SimError("internal reference to the empty LSU command slot " + std::to_string(slot));
            }
            return *slots[slot];
        }
        // The lowest slot of the profile's `capacity` free once the commands
        // `leaving` (by number) have left.
        [[nodiscard]] std::optional<uint8_t> lowest_free(std::size_t capacity, const std::vector<uint64_t>& leaving) const {
            for (std::size_t slot = 0; slot < capacity && slot < slots.size(); ++slot) {
                if (!slots[slot].has_value() ||
                    std::find(leaving.begin(), leaving.end(), slots[slot]->id) != leaving.end()) {
                    return static_cast<uint8_t>(slot);
                }
            }
            return std::nullopt;
        }
        [[nodiscard]] const LsuCommand* find(uint64_t id) const {
            for (const auto slot : by_age) {
                if (slots[slot]->id == id) return &*slots[slot];
            }
            return nullptr;
        }
        [[nodiscard]] LsuCommand* find(uint64_t id) {
            return const_cast<LsuCommand*>(std::as_const(*this).find(id));
        }
        // The commands, oldest first.
        [[nodiscard]] Bounded<const LsuCommand*, limits::lsu_active_commands> oldest_first() const {
            Bounded<const LsuCommand*, limits::lsu_active_commands> commands;
            for (const auto slot : by_age) commands.push_back(&*slots[slot]);
            return commands;
        }
        // A new command takes the slot its dispatch chose (the lowest free one).
        void take(const LsuCommand& command, std::size_t capacity) {
            if (command.slot >= capacity || command.slot >= slots.size() || slots[command.slot].has_value() ||
                lowest_free(capacity, {}) != command.slot) {
                throw SimError("internal: the LSU holds more than lsu.active_commands commands");
            }
            slots[command.slot] = command;
            by_age.push_back(command.slot);
        }
        void release(uint64_t id) {
            for (std::size_t index = 0; index < by_age.size(); ++index) {
                const auto slot = by_age[index];
                if (slots[slot]->id != id) continue;
                slots[slot].reset();
                by_age.erase_at(index);
                return;
            }
            throw SimError("internal: LSU command #" + std::to_string(id) + " left a slot it is not in");
        }
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

    // The egress FIFO: lsu.egress_sectors slots of 32 B (8 at the canonical
    // profile), a BRAM. Every store client writes it through one shared write
    // port of lsu.pluck_words_per_cycle words, one client per cycle, granted
    // per slot (a started slot is never preempted; TMA wins each new grant).
    // A scalar stg lane is one slot holding one word and 4 byte enables.
    // A slot passes three points, each the head of its own queue:
    //   open      the write port fills it (EgressOpenSlot, the port's
    //             register); when its last word is written it is pushed,
    //             complete, into both FIFOs below at once;
    //   request   the request FIFO's head is the request pointer: the oldest
    //             complete slot whose write request the controller has not
    //             accepted. It is offered lsu.egress_latency_cycles after
    //             the push (the RAM's write-to-read latency) and leaves when
    //             accepted;
    //   data      the data FIFO's head is the slot the bus reads, one beat
    //             per DRAM tick; it leaves after its last beat.
    // Writes leave in FIFO order, so the bus's head is always the oldest
    // accepted request. Slots in use: the data FIFO's entries and the open
    // one; a new grant needs a free slot (drive_egress checks it).
    struct EgressOpenSlot {
        GmemClient client = GmemClient::lsu;
        uint8_t owner = 0;          // LSU: its command's slot
        uint32_t index = 0;         // LSU: the command's request index this slot carries
        uint64_t sector_address = 0;
        uint64_t byte_enable = 0;
        uint32_t words_total = 0;   // words the client writes into this slot
        uint32_t words_written = 0;
        // The words written so far. In hardware they are in the RAM at the
        // write pointer's slot already; the write pointer moves on (the push)
        // when the last one is written.
        std::array<uint8_t, limits::sector_bytes> data{};
        uint64_t id = 0;            // (T) running slot number (trace / probe)
        uint64_t command = 0;       // (T) its command's number
    };
    // The request FIFO's entry: what the write request needs (and the LSU's
    // check that the request pointer is its command's next sector).
    struct EgressRequest {
        GmemClient client = GmemClient::lsu;
        uint8_t owner = 0;          // LSU: its command's slot
        uint32_t index = 0;         // LSU: the command's request index
        uint64_t sector_address = 0;
        uint64_t byte_enable = 0;
        uint64_t id = 0;            // (T) running slot number
        uint64_t command = 0;       // (T) its command's number
    };
    // The data FIFO's entry: the slot's sector (gmem.sector_bytes of it).
    struct EgressSector {
        std::array<uint8_t, limits::sector_bytes> data{};
        uint64_t id = 0;            // (T) running slot number: the controller's check that it is at the head
    };

    struct WriteAck {
        uint64_t dram_tick = 0;
        uint64_t request = 0;
        GmemClient client = GmemClient::lsu;
        uint8_t lsu_slot = 0;        // LSU: the command's slot
        uint64_t command = 0;        // (T)
        std::optional<uint32_t> tag; // scalar scatter: the free-ID tag to return
    };

    // No mbar phase read between descriptor and walking.
    enum class TmaCommandState : uint8_t {
        descriptor,
        starting, // its walker's start is in the start pipeline (TmaWalkerStart)
        walking,
        data_complete,
        event_presented,
        launch_done, // the launch copy: its last sector has landed; it leaves the queue next cycle
    };

    // One TMA command, from its push into the queue to its completion
    // event's claim (tma.queue_commands of them: 4 at the canonical profile).
    // The decoded descriptor is not kept here: it goes into the walker's
    // registers when the descriptor read returns (TmaWalker).
    struct TmaCommand {
        uint64_t id = 0;            // (T) the command's number: the trace, the sanitizer
        uint8_t slot = 0;           // the queue slot it holds (set when it takes one)
        uint32_t warp = 0;
        uint32_t pc = 0;
        DecodedInstruction instruction;
        // The instruction's operands: the opcode's direction, the descriptor
        // (a constant), the tile's coordinates c0..c3 (zero past the opcode's
        // N), the buffer's and the barrier's shared addresses (their low bits
        // dropped: not wired, src/tma_descriptor.hpp).
        TmaDirection direction = TmaDirection::load;
        uint32_t descriptor_address = 0;
        std::array<int32_t, tma_max_coordinates> coordinates{};
        uint32_t shmem_address = 0;
        uint32_t mbar_address = 0;
        TmaCommandState state = TmaCommandState::descriptor;
        uint64_t logical_bytes = 0;     // the tile's bytes: what its completion subtracts (32 bits)
        uint64_t useful_bytes = 0;      // (T) in-range bytes moved, for the trace
        uint32_t pending_load_writes = 0;    // sectors walked whose placement has not reached the RAM
        uint32_t pending_store_captures = 0; // store sectors whose source read has not returned
        bool walker_done = false;
        // The launch copy (start_launch_copy): no instruction, no descriptor,
        // no barrier. launch_landed: the bytes of the launch block whose
        // placement has reached the RAM (17 bits), in order from its start.
        bool launch = false;
        uint32_t launch_landed = 0;
    };

    // The TMA's walker: ONE set of registers, serving the frontend command
    // from its descriptor's return to its last step (only one command walks at
    // a time). It is loaded tma.walker_start_cycles after the descriptor's last
    // words come back (tma_walker_start: three coordinate x stride products,
    // through TmaWalkerStart below) and steps
    // once per cycle (plan_tma_walker). A step is one of:
    //   - an in-range run: from (x, y) to the first of the sector's end, the
    //     row's end and the tensor's edge along dimension 0 -- a load sector
    //     (needs a credit and the request channel) or a store sector (a
    //     credit, a free store register and A-read);
    //   - a load's out-of-range stretch, at most a sector's bytes of it per
    //     step: a zero fill (a credit, and the placement tickets' write port,
    //     which a sector the ingress completes in that cycle has first);
    //   - a store's out-of-range stretch of one row, skipped whole (needs
    //     nothing; one step, one cycle: an entirely out-of-range store takes
    //     one cycle per row).
    // A stretch is out of range when its row is (y outside [y_low, y_high)) or
    // it lies left of x_low or right of x_high. Every boundary is a compare;
    // addresses move by adders: x x element_bytes and x x spacing are shifts,
    // a row is one add of the row steps.
    // Widths at the canonical profile: row_global 64, row_shared 16, x 16,
    // y 16, x_low 16, x_high 16, y_low 16, y_high 16, row_step_global 34,
    // row_step_shared 18, tile_length 16, tile_rows 16, element size 2,
    // layout 1, transposed 1, spacing 5 (log2) = 269 bits.
    struct TmaWalker {
        // Computed at the descriptor's decode, loaded from the start pipeline, fixed for the command.
        uint64_t row_step_global = 0;   // s_row x element bytes: one row to the next in gmem (a shift)
        uint32_t row_step_shared = 0;   // unshuffled shmem from one row to the next: the stored row
                                        // (tile length x element bytes) or, transposed, the element bytes
        uint16_t x_low = 0;             // the tile positions inside the tensor along dimension 0: [x_low, x_high)
        uint16_t x_high = 0;
        uint16_t y_low = 0;             // the tile rows inside the tensor (row dimension, and the other two
        uint16_t y_high = 0;            // dimensions' coordinates; empty if those are outside): [y_low, y_high)
        uint16_t tile_length = 0;
        uint16_t tile_rows = 0;
        uint8_t element_bytes = 0;
        TileLayout layout = TileLayout::dense;
        bool transposed = false;
        uint32_t spacing = 0;           // swizzle64: the stored row's bytes
        // Stepped.
        uint64_t row_global = 0;        // global address of row y's element 0, modulo 2^64: exact wherever
                                        // the row is in range (it may start "negative")
        uint32_t row_shared = 0;        // unshuffled shared address of row y's element 0
        uint16_t x = 0;                 // the next element along the row
        uint16_t y = 0;                 // the row
    };

    // The walker's start pipeline: tma.walker_start_cycles register stages
    // (canonical 2) between the cycle a command's descriptor words are back
    // and the cycle its walker registers are loaded. The three coordinate x
    // stride products and the add chain into row 0's address get those
    // stages. A command enters stage 0 at its descriptor's return (in state
    // `starting`), moves one stage per cycle, and from the last stage loads
    // the walker (state `walking`): its first step is walker_start_cycles
    // cycles later than with no stages. Only the frontend command is in it,
    // so at most one stage is occupied. Each stage carries the finished
    // walker registers (the multipliers' partial products are not modelled
    // bit by bit) and the command's queue slot; 0 stages: no pipeline, the
    // walker is loaded at the descriptor's return, the timing before.
    struct TmaWalkerStart {
        uint8_t slot = 0;           // the command's queue slot
        uint64_t command = 0;       // (T) its number
        TmaWalker walker;
    };

    enum class TmaPortKind : uint8_t {
        descriptor,
        load_write,
        store_read,
    };

    // One shmem transaction of the TMA: its run and how many bank rounds it
    // has been granted. Load placements (port-B write) read the TMA sector
    // register (or write zeros for an OOB fill); store reads (A-read) fill a
    // TMA store sector register; the descriptor read (A-read) fills the
    // frontend latch (TmaReadTransaction) -- it is the run of the
    // descriptor's sixteen words, dense. A round's pieces are a function of
    // the run and the round number (tma_round_plan), so nothing per piece is
    // held. At most tma_max_rounds() rounds (2 at the canonical profile).
    struct TmaPortRequest {
        // One element of the run, served in one round: its bytes lie in one
        // shared word (and, from the sector register, one sector word).
        struct Piece {
            uint8_t element = 0;
        };
        TmaPortKind kind = TmaPortKind::descriptor;
        uint64_t sector_request = 0; // (T) a placement's sector request (the edge probe)
        TmaRun run;                  // run.command: the command
        uint32_t rounds = 0;         // bank rounds (tma_round_plan)
        uint32_t next_round = 0;     // rounds granted
    };

    // A TMA sector ready for placement (port-B write): either the TMA
    // stream's sector register (complete; it stays occupied until its last
    // round) or a synthetic OOB zero fill (no bus data: run.zero_fill).
    // The tickets wait in tma_returns_, a FIFO of runs (98 bits each): every
    // entry holds a sector credit (from its walk to its placement's end), so
    // at most tma.sector_credits entries (16): a 16 x 98-bit LUTRAM FIFO.
    struct TmaReturn {
        TmaRun run;
        uint64_t sector_request = 0; // (T) the request id (trace, probe)
    };

    // A TMA store sector register (32 B flops; tma.store_sector_registers of
    // them, a group used as a ring in reservation order: a source read
    // reserves the register after the held ones, its rounds fill it by its
    // index -- several may be filling while an older one drains -- and the
    // egress drains the oldest, the head; tma_store_head_ is that pointer).
    // Not a FIFO: a register is written after it is taken. The store-sector path, one one-round
    // sector, L = D + Lb, d = ceil(sector words / lsu.pluck_words_per_cycle):
    //   g          the walker's source read is granted on A-read (bypass (h):
    //              planned from Q and this cycle's egress drain). The register
    //              at the ring's tail is reserved now; the read's rounds carry
    //              its index (the return context). A-read is free again after
    //              the read's last round's grant.
    //   g + L      the words reach the port output and are written into the
    //              register at the edge (its run says where each byte goes)
    //   g + L + 1  complete in Q: the oldest complete register (the ring's
    //              head) gets the egress write port (TMA wins a new egress
    //              slot) and drains pluck words per cycle
    //   g + L + d  its last word leaves: the register is free, and a source
    //              read granted in this same cycle may reserve it (d)
    // One register therefore turns over every L + d cycles; N registers and a
    // free egress port sustain min(N x 32 / (L + d), 32 / d) bytes per cycle.
    // Its sector address and byte enables are the run's (run_sector_address,
    // run_byte_enable). Width: data 256 + run 98 + rounds 2 + returned 2 +
    // words_out 4 bits.
    struct TmaStoreRegister {
        TmaRun run;                 // its sector's run: each returned byte's place in `data`, the byte enables
        uint32_t rounds = 0;        // the source read's rounds
        std::array<uint8_t, limits::sector_bytes> data{}; // gmem.sector_bytes of it (32 B)
        uint32_t returned = 0;      // source rounds whose words are in `data`
        uint32_t words_out = 0;     // words already written into the egress FIFO
        [[nodiscard]] bool complete() const { return returned == rounds; }
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
    // (At most 8 bytes: an mbar's two words. A store's lane and an atomic
    // write a word; a placement piece 1, 2 or 4 bytes.)
    struct ShmemWrite {
        uint32_t address = 0;
        Bounded<uint8_t, 8> bytes;
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
        Bounded<uint32_t, limits::shmem_banks> addresses;      // word-aligned byte addresses, at most one per bank
        Bounded<uint32_t, limits::shmem_banks> words;          // their contents, read at the RAM sample
        std::optional<MbarReadRequest> probe; // probe: the request
    };

    struct MbarWake {
        uint32_t address = 0;
        uint8_t phase = 0;
    };

    // A TMA placement's last round reaches the RAM: its command's load write
    // ends (pending count, credit release, data complete).
    struct PlacementEnd {
        uint8_t slot = 0;        // the command's queue slot
        uint64_t command = 0;    // (T) its number
        uint32_t shared_end = 0; // the end of the run's bytes in shared memory (the launch copy's mark)
    };

    // One granted write round in a write pipeline (A-write, port-B write): D
    // stages to the RAM; at the last its bytes land at the clock edge and its
    // completion happens in that cycle.
    // A round writes at most a word per bank, as pieces: 4 x shmem.banks
    // one-byte pieces at the most. (Hardware holds a word and byte enables
    // per bank; here the pieces are kept as they were made.)
    struct WriteBeat {
        Bounded<ShmemWrite, 4 * limits::shmem_banks> writes;
        std::optional<DepbarEvent> sts_done;   // an sts's last round: its depbar clears
        std::optional<MbarWake> wake;          // an RMW write that flips the phase wakes the waiters
        std::optional<PlacementEnd> placed;    // a TMA placement's last round: its load write ends
    };

    // What the RMW engine's ALU computed from the word(s) it read, in its ALU
    // stages: the word to write back (an mbar's eight bytes, an atomic's four
    // in the low half) and, for an mbar operation that flips the phase, the
    // new phase (the waiters wake when the write reaches the RAM). Its address
    // and whether it is an atomic are the operation's, in the engine's register
    // until the write's grant, where the write round is made (rmw_write_round).
    struct RmwUpdate {
        uint64_t word = 0;
        std::optional<uint8_t> wake_phase;
    };

    // The RMW engine: the operation in flight (its presence IS the store
    // PAUSE: store_pause()), its ALU stages and the admission state between
    // its requesters (they change together).
    struct RmwEngine {
        std::optional<ActiveMbarRmw> active;
        // The engine whose event was started last (set on every engine-event
        // start, contested or not). When both engines hold an event, the other
        // engine's goes. Starts at mma, so the first such tie serves the TMA.
        MbarEngine last_served = MbarEngine::mma;
        // Fairness between the two classes of requester (implementer
        // choice): when the SIMT RMW register and an engine event register both
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
        TmaPortRequest request;       // next_round: rounds granted
        uint32_t returned = 0;        // descriptor: rounds whose words are back
        uint32_t store_register = 0;  // store-source: the TMA store register its rounds land in
        // The frontend latch: the descriptor's 64 bytes as its rounds come
        // back (512 bits of flops).
        std::array<uint8_t, tma_descriptor_bytes> latch{};
    };

    // The A-read and A-write round-robin pointers. A-read
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

    // The RF read decision's state (the write decision's rule, mirrored),
    // per RF bank: the read_owed bit and the round-robin cursor
    // over the async readers.
    struct RfReadPolicy {
        uint8_t owed = 0;
        uint32_t cursor = 0;
        bool operator==(const RfReadPolicy&) const = default;
    };

    // The RF write decision's state, per RF bank: the
    // async_owed bit and the round-robin cursor over the async holders. Both
    // change in the same calculation.
    struct RfWritePolicy {
        uint8_t owed = 0;
        uint32_t cursor = 0;
        // (The LSU has ONE RF write path: the load buffer,
        // one lane-wide vector per cycle. No cursor: there is one buffer.)
        bool operator==(const RfWritePolicy&) const = default;
    };

    // One warp's depbar counters, sb0..sb5: how many of the warp's
    // instructions that count on the depbar are admitted and not yet
    // decremented. The counter's width follows from the places such an
    // instruction can be between its admission and its decrement
    // (Machine::depbar_counter_max_, where each place is listed: 23 at the
    // canonical profile, a 5-bit counter). An admission past it is an
    // internal error.
    using DepbarCounters = std::array<uint32_t, 6>;
    using AccEngines = std::array<std::optional<AccTransfer>, 2>; // [0 the acc.ld engine, 1 the acc.st engine]

    // The MMA engine's queue and pipeline. One group: a command's next_step
    // and operands_consumed track the fetch, its outstanding_results the
    // result list. (Its event register is separate.)
    struct MmaPipeline {
        // The command slots: mma.queue_commands commands in order, oldest
        // first (a pair at the canonical profile). Issue's command takes the
        // place behind the last; commands retire in order, from the front.
        // The second can issue steps while the first still waits for its
        // results, and a result finds its command by the command's number,
        // so this is a row of slots with an order, not a queue read only at
        // its head.
        Bounded<MmaCommand, limits::mma_queue_commands> slots;
        std::optional<MmaFetch> fetch;  // the step whose rounds port B is granting
        // The core's operand latch, by element position: the values of the
        // step whose rounds are arriving at the port output (A + B of one
        // step: 128 B at the canonical profile). Its last round enters the
        // core with them.
        std::array<uint16_t, mma_step_elements> latch{};
        // The held A block: core_m x core_k fp16 (64 B at 4x8), written when a
        // reloading step enters the core, used by the steps after it.
        Bounded<uint16_t, limits::mma_block> a_held;
    };

    // The URF's multi-round operand read (the rounds are src/timing.hpp
    // uniform_read_rounds). Two instructions read more
    // uniform registers than the two read ports give one round: a tma (N + 2,
    // R rounds) and an mma.async (urA, urB | urC). One
    // declared register serves both:
    //   - the round counter (UrfReadRounds): the instruction whose rounds are
    //     still to start, loaded at accept. Round 0 samples in the issue
    //     register's cycle (a tma's urS / urM; an mma.async's urA / urB, in
    //     its dispatch entry like any read); round k >= 1 one cycle later
    //     each (a tma's coordinates 2(k-1) and 2(k-1) + 1; an mma.async's
    //     urC, which joins its dispatch entry from the URF output, Lu later,
    //     before its dispatch: src/timing.hpp issue_to_dispatch); the counter
    //     empties after the last round. While it has rounds left issue
    //     refuses every uniform reader (urf_read_ports_held). In hardware:
    //     the warp (4 bits at 16 warps), the round (2 bits) and the next
    //     register (5 bits); round 0's fields are the issue register's. Here
    //     it keeps the whole instruction.
    // The TMA's operand read is that counter and two registers of its own:
    //   - the URF read stages (tma_read_pairs_, Lu stages): each pair sampled
    //     for a tma travels to the URF output with its tag -- which fields it
    //     is (the round) and whether it is the last; round 0's also carries
    //     the instruction's own fields. Without the tag, back-to-back
    //     commands would put one command's words into the next's.
    //   - the holding register (TmaReadHolding): the command being read --
    //     the two addresses, four coordinates, the descriptor address, the
    //     direction (about 180 bits) -- filled as the pairs arrive. When the
    //     last pair arrives the command is pushed into the TMA queue. One is
    //     enough: a command's last words arrive at T + R + Lu, the next
    //     command's first at T' + 1 + Lu, and T' >= T + R.
    struct UrfReadHeader {
        uint64_t issue_cycle = 0;
        uint32_t warp = 0;
        uint32_t pc = 0;            // (S) PANIC text: the instruction's own pc
        uint64_t active_mask = 0;   // (S) the uniform operation's partial-mask PANIC
        DecodedInstruction instruction;    // the opcode and its registers (a tma: the descriptor address, the direction and N)
    };
    struct UrfReadRounds {
        UrfReadHeader header;
        uint32_t round = 0;         // the round sampled next
    };
    struct TmaReadPair {
        uint32_t round = 0;                  // 0: the buffer and the barrier; k >= 1: coordinates 2(k-1) ..
        bool last = false;                   // the command's last round
        std::optional<UrfReadHeader> header; // round 0: the instruction's fields
        Bounded<uint32_t, limits::round_uniform_reads> words;         // the sampled registers' values, in operand order
    };
    struct TmaReadHolding {
        UrfReadHeader header;
        uint32_t buffer = 0;        // urS as read (its low bits are the sanitizer's)
        uint32_t barrier = 0;       // urM as read
        std::array<int32_t, tma_max_coordinates> coordinates{}; // zero until supplied
    };

    // The TMA engine's command queue and walker bookkeeping. One group:
    // credits, the walker and each command's pending counts move together.
    // (Its event register, shmem transactions, the sector registers and the
    // placement tickets, tma_returns_, are separate.) Sizes, checked in
    // drive_tma_queue:
    //   commands  tma.queue_commands entries (4); the oldest walking or
    //             descriptor-state one is the frontend (4 slots of about 120
    //             bits plus an age order).
    // The TMA's command slots: tma.queue_commands slots (4), each holding one
    // command from the cycle its operand read pushes it until it leaves, and
    // the slots' age order. A command takes the lowest free slot; the
    // frontend and the event presentation choose the oldest command in the
    // state they want; a command leaves from any slot (a store can be done
    // before an older load's placements have landed). A run (TmaRun), a
    // placement's end and a walker start name their command by its slot; the
    // frontend and the event register's command are still named by number,
    // found here by comparing the slots.
    struct TmaCommandSlots {
        std::array<std::optional<TmaCommand>, limits::tma_queue_commands> slots{};
        Bounded<uint8_t, limits::tma_queue_commands> by_age; // occupied slots, oldest first
        [[nodiscard]] std::size_t size() const { return by_age.size(); }
        [[nodiscard]] bool empty() const { return by_age.empty(); }
        [[nodiscard]] const TmaCommand* find(uint64_t id) const {
            for (const auto slot : by_age) {
                if (slots[slot]->id == id) return &*slots[slot];
            }
            return nullptr;
        }
        [[nodiscard]] TmaCommand* find(uint64_t id) {
            return const_cast<TmaCommand*>(std::as_const(*this).find(id));
        }
        [[nodiscard]] const TmaCommand& at(uint64_t id) const {
            const auto* command = find(id);
            if (command == nullptr) throw SimError("internal: TMA command #" + std::to_string(id) + " is in no slot");
            return *command;
        }
        [[nodiscard]] TmaCommand& at(uint64_t id) { return const_cast<TmaCommand&>(std::as_const(*this).at(id)); }
        // The command in slot `slot`; `id` is the number the referring state
        // carries beside the slot (a check that the slot was not reused).
        [[nodiscard]] const TmaCommand& in_slot(uint8_t slot, uint64_t id) const {
            if (slot >= slots.size() || !slots[slot].has_value() || slots[slot]->id != id) {
                throw SimError("internal: TMA command #" + std::to_string(id) + " is not in slot " + std::to_string(slot));
            }
            return *slots[slot];
        }
        [[nodiscard]] TmaCommand& in_slot(uint8_t slot, uint64_t id) {
            return const_cast<TmaCommand&>(std::as_const(*this).in_slot(slot, id));
        }
        [[nodiscard]] const TmaCommand* find_in_slot(uint8_t slot, uint64_t id) const {
            if (slot >= slots.size() || !slots[slot].has_value() || slots[slot]->id != id) return nullptr;
            return &*slots[slot];
        }
        [[nodiscard]] TmaCommand* find_in_slot(uint8_t slot, uint64_t id) {
            return const_cast<TmaCommand*>(std::as_const(*this).find_in_slot(slot, id));
        }
        // The commands, oldest first.
        [[nodiscard]] Bounded<const TmaCommand*, limits::tma_queue_commands> oldest_first() const {
            Bounded<const TmaCommand*, limits::tma_queue_commands> commands;
            for (const auto slot : by_age) commands.push_back(&*slots[slot]);
            return commands;
        }
        // A new command takes the lowest free slot of the profile's `capacity`.
        void take(const TmaCommand& command, std::size_t capacity) {
            for (std::size_t slot = 0; slot < capacity && slot < slots.size(); ++slot) {
                if (slots[slot].has_value()) continue;
                slots[slot] = command;
                slots[slot]->slot = static_cast<uint8_t>(slot);
                by_age.push_back(static_cast<uint8_t>(slot));
                return;
            }
            throw SimError("internal: the TMA holds more than tma.queue_commands commands");
        }
        void release(uint64_t id) {
            for (std::size_t index = 0; index < by_age.size(); ++index) {
                const auto slot = by_age[index];
                if (slots[slot]->id != id) continue;
                slots[slot].reset();
                by_age.erase_at(index);
                return;
            }
            throw SimError("internal: TMA command #" + std::to_string(id) + " left a slot it is not in");
        }
    };
    struct TmaQueue {
        TmaCommandSlots commands;                // tma.queue_commands slots; issue's command takes one
        std::optional<uint64_t> frontend;        // the command the descriptor / walker frontend serves
        std::optional<uint64_t> event_command;   // whose completion event is in the TMA's event register
        uint32_t credits_used = 0;               // of tma.sector_credits (5 bits)
        TmaWalker walker;                        // the frontend command's walk
    };

    // =======================================================================
    // Wires: this cycle's decisions. Each is computed once in tick(), in the
    // dependency order written there, from register Q values, memories and
    // earlier wires. The register drivers read only Q and these.
    // =======================================================================

    // The async RF writers: holders of a complete result
    // vector that waits for its bank's write port. Round-robin order: load
    // slots, then the load buffer, then the acc.ld engine's pipeline output.
    enum class AsyncWriter : uint8_t { load_slot, lsu_buffer, acc_ld };
    struct AsyncWrite {
        AsyncWriter writer = AsyncWriter::load_slot;
        uint32_t index = 0;   // load slot / load_buffer / 0 (the acc.ld engine)
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
        std::optional<uint32_t> lsu_presented;           // the load buffer, on the LSU's one write path
        [[nodiscard]] bool granted_to(AsyncWriter writer, uint32_t index) const;
    };

    // The async RF readers: each wants a read
    // port of one bank in the CURRENT cycle, where the issue register's
    // instruction owns the ports it samples.
    // Round-robin order: the acc.st engine's next beat, then the LSU's
    // store-data read (vector stores).
    enum class RfReader : uint8_t { acc_st, lsu_store };
    struct RfReadRequest {
        RfReader reader = RfReader::acc_st;
        uint32_t index = 0;          // 0 (the acc.st engine)
        uint32_t warp = 0;
        uint8_t reg = 0;
        uint64_t lsu_command = 0;    // lsu_store
        uint32_t buffer = 0;       // lsu_store
        uint8_t lsu_slot = 0;        // lsu_store: the command's slot
        uint32_t group = 0;          // lsu_store: the lane group this beat is for
        uint32_t beat = 0;           // lsu_store: which of rS .. rS+7
        uint64_t lanes = 0;          // lsu_store: the group's active lanes
        bool last_group = false;     // lsu_store: the command reads this register for the last time
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
        uint8_t slot = 0;          // the command's queue slot
        uint64_t command = 0;      // (T) its number
        std::vector<std::pair<uint32_t, uint8_t>> bytes;  // (byte offset in the sector, byte): at most a sector's
        bool last = false;                                // the source read's last round: the register is complete
    };

    // LSU / TMA -> egress FIFO (the shared egress write port).
    struct EgressWrite {
        std::optional<GmemClient> client;              // who writes the egress FIFO this cycle
        std::optional<EgressOpenSlot> new_slot;        // a new grant opens this slot
        std::vector<std::pair<uint32_t, uint32_t>> words; // (byte offset in the slot, word)
        bool slot_complete = false;                    // the open slot's last word lands now
        uint64_t lsu_command = 0;
        bool lsu_group_drained = false;                // a vector store's group has left the store buffer
        bool lsu_store_drained = false;                // a scalar store's last word has left the store buffer
        bool tma_store_drained = false;                // the head TMA store register's last word leaves
    };

    // The TMA walker's step this cycle, planned from Q. none: nothing to do or
    // blocked (no credit; a store without a free store register or A-read);
    // skip: a store's out-of-range stretch passes (no sector). The others
    // emit `run` when their resource grants it.
    struct TmaWalkerPlan {
        enum class Sector : uint8_t { none, skip, zero_fill, load, store };
        Sector sector = Sector::none;
        TmaWalker walker_after;     // the walker after this step
        bool done_after = false;    // this step ends the tile
        TmaRun run;
        uint64_t sector_address = 0; // load / store: the run's sector
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
        bool egress_requested = false;             // the egress request FIFO's head is accepted now (a write)
        bool walker_proceeds = false;              // the walker emits its sector this cycle
        // Request ids, assigned in the fixed order TMA write, walker sector, LSU
        // (a zero fill takes its id after them, in step_tma).
        uint64_t tma_write_id = 0;
        uint64_t walker_id = 0;                    // the walker's read sector
        uint64_t lsu_id = 0;
        std::optional<uint32_t> lsu_tag;           // the free-ID tag an LSU scalar request takes
        std::optional<GmemSectorRequest> accepted; // the request the controller queue takes
        std::optional<TmaRun> new_tma_read;        // a TMA read accepted now: its read record
        std::optional<LsuRead> new_lsu_read;       // an LSU read accepted now: its read record
    };

    // The shmem grants of this cycle: one select per service.
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
        std::size_t port_a_writes = 0;               // the first this many are A-write's (port A), the rest port B's
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
        std::optional<RmwUpdate> rmw_update;         // the RMW's update, from its read's words: enters ALU stage 0
        std::optional<ActiveMbarRmw> rmw_started;    // the operation the RMW engine takes now (from rmw_start's register)
        bool simt_rmw_started = false;               // rmw_start empties the SIMT RMW register
        bool tma_event_claimed = false;              // ... the TMA's event register
        bool mma_event_claimed = false;              // ... the MMA's event register
        bool tma_read_changed = false;               // tma_read_after is the TMA read transaction's next value
        std::optional<TmaReadTransaction> tma_read_after;
        std::optional<TmaReadTransaction> tma_read_done; // a descriptor read's last words came back now: its latch
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
        uint32_t words = 0;                               // read from the FIFO head (they leave it)
        struct LaneWrite { uint32_t buffer = 0; uint32_t lane = 0; uint32_t word = 0; }; // lane: the buffer word
        std::vector<LaneWrite> lanes;                     // words into the load buffer
        // Each stream's read records: the heads whose sector's last byte was
        // consumed now (they leave), and the bytes consumed of the head after.
        struct StreamHead {
            uint32_t finished = 0;
            uint32_t received = 0;
            bool changed = false;
        };
        StreamHead tma_reads;
        StreamHead lsu_reads;
        // The group whose first word claims the free load buffer now.
        struct LoadClaim {
            uint8_t slot = 0;         // its command's slot
            uint32_t group = 0;
            uint64_t lanes = 0;       // its active lanes
            uint32_t expected = 0;    // the words it brings
            uint32_t beats_total = 0; // the RF writes it needs
        };
        std::optional<LoadClaim> load_claim;
        std::optional<TmaSectorRegister> tma_sector_after; // the TMA sector register after its fill
        std::optional<TmaReturn> tma_ticket;               // the sector completed: placement ticket
    };

    // The placeholder controller's DRAM ticks this core cycle.
    struct ControllerTick {
        std::vector<IngressBeat> ingress_beats;          // bus -> ingress FIFO
        uint32_t egress_slots_freed = 0;                 // last beats left the FIFO head
        uint32_t tma_credit_releases = 0;                // TMA write acks
        std::vector<uint32_t> freed_tags;                // scatter write acks
        std::map<uint64_t, uint32_t> lsu_acks;           // command -> write acks
    };

    // LSU: RF -> the store buffer (vector stores), below issue in priority.
    // The LSU's store-data RF read (vector stores): sampled at its grant, at
    // the RF output Lr cycles later (the last stage of lsu_read_pipe_), when
    // it fills the store buffer.
    struct LsuRfRead {
        std::optional<uint8_t> command;  // the store's command slot
        uint32_t buffer = 0;
        LaneWords values;   // the register, every lane
        uint32_t group = 0;             // the lane group it is for
        uint32_t beat = 0;              // which of rS .. rS+7
        uint64_t lanes = 0;             // the group's active lanes: the words kept
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
        bool taken = false;           // a redirect: a taken branch, or a sync popping a pc other than its own + 8
        uint32_t target = 0;          // taken: the warp's pc and fetch pc from the next cycle
        // The divergence stack (drive_divergence): the warp's active mask from
        // the next cycle; an entry written into the stack memory at entry
        // `depth` at this edge (and into the top copy); the top entry leaving.
        std::optional<uint64_t> mask;
        std::optional<DivergenceEntry> push;
        std::size_t push_slot = 0;    // its memory word: warp x entries + depth
        bool pop = false;
    };

    // What an accepted instruction changes outside the issue unit now.
    struct IssueDecision {
        std::optional<uint32_t> popped;                  // the warp whose FIFO head was accepted
        bool halt = false;                               // ... a halt: the warp's front end stops
        std::optional<IssuedBranch> branch;              // ... a branch: into the issue register
        std::optional<DepbarEvent> depbar_admission;
        std::optional<uint32_t> rf_write_bank;           // a fixed RF-write booking was made to this bank
        std::optional<std::size_t> rf_write_stage;       // ... at this stage of the write calendar
        std::optional<ArithmeticRead> arithmetic;        // a fixed-latency instruction enters the arithmetic pipeline
        std::vector<uint8_t> rf_read_banks;              // per bank: its round's reads (into the issue register; read_owed)
        std::optional<DispatchEntry> dispatch;           // a memory / engine instruction enters the dispatch pipeline
        std::optional<UrfWrite> urf_write;               // a uniform-register writer enters the URF write pipeline
        std::optional<UrfReadRounds> urf_rounds;         // a tma or an mma.async loads the URF's round counter
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
        uint32_t lsu_stores = 0;     // stg / stg.v8: their data is still to move through the store buffer
        uint32_t mma_commands = 0;
        std::array<uint32_t, 2> acc{}; // [0 acc.ld, 1 acc.st]
    };

    // All of a cycle's wires, in the order tick() computes them.
    struct Wires {
        // The front end, from Q only.
        FetchGrant fetch;
        std::optional<FetchedPacket> fetched;
        std::optional<BranchResolution> branch;
        // The divergence stack memory's output now: a pop's read of the new
        // top, for its warp's top copy (warp, entry).
        std::optional<std::pair<uint32_t, DivergenceEntry>> divergence_refilled;
        // The RF write port: async grants and SIMT inhibition.
        RfWriteDecision rf_write;
        std::vector<uint64_t> released_lsu_commands;  // their last RF write granted now, or their store completed
        // Per RF write port (bank x rf.write_ports_per_bank + port): the
        // write presented now, entering that port's write pipeline.
        std::vector<std::optional<RfWrite>> rf_port_writes;
        // The predicate writes now: the file has them from the next cycle.
        std::vector<PredicateWrite> predicate_writes;
        // The tmem writes at the RAM now, through each fine bank's two write
        // ports (MMA results, acc.st beats): they land at this cycle's clock
        // edge, after every tmem read of this cycle (READ_FIRST, as shmem).
        // (physical word, value)
        std::vector<std::pair<uint32_t, uint32_t>> mma_tmem_writes;
        std::vector<std::pair<uint32_t, uint32_t>> acc_st_tmem_writes;
        // Collected as they happen, from several steps below.
        std::vector<DepbarEvent> depbar_completions;  // in the order they happened
        std::vector<uint32_t> completed_waits;        // try-wait slots completed this cycle
        // LSU command slots whose store had its last acknowledgement now.
        std::vector<uint8_t> store_completions;
        // The store path's last stage; request generation; the two selects.
        EgressWrite egress;
        TmaRequests tma;
        LsuRequestWant lsu_want;
        RequestChannel channel;
        ShmemCycle shmem;
        // Next state of the units issue looks ahead at (committed by their
        // drivers, with issue's additions).
        MmaPipeline mma_next;
        // The MMA's pipelines this cycle: the round at the shmem RAM with the
        // elements it sampled, and the round port B granted (it enters the
        // read pipeline); the tag of the step entering the core (it enters
        // the tag pipeline), and the result leaving the core with its tag
        // (it enters the write pipeline).
        std::optional<MmaRead> mma_read_sampled;
        std::optional<MmaRead> mma_read_granted;
        std::optional<MmaTag> mma_tag_entering;
        std::optional<MmaWrite> mma_write_entering;
        std::optional<MbarRmwRequest> mma_present;    // a commit presented into the MMA's event register
        AccEngines acc_next;
        // The acc.ld pipeline this cycle: the beat at the tmem RAM stage with
        // the words it sampled; the engine's next read entering stage 0; or
        // a hold (its output is valid and was denied the RF write).
        std::optional<AccLdBeat> acc_ld_sampled;
        std::optional<AccLdBeat> acc_ld_launch;
        bool acc_ld_hold = false;
        IngressPluck ingress;
        ControllerTick controller;
        TmaQueue tma_next;
        // The placement ticket entering tma_returns_ now through its one write
        // port (step_tma); whether its head leaves now (its placement starts).
        std::optional<TmaReturn> tma_returns_pushed;
        bool tma_return_taken = false;
        // A walker start entering the start pipeline's stage 0 now (its
        // descriptor came back this cycle; tma.walker_start_cycles > 0).
        std::optional<TmaWalkerStart> tma_walker_starting;
        // The TMA operand read's end: the pair at the URF output now goes
        // into the holding register; with the last one the command is pushed
        // into the TMA queue (tma_pushed; drive_tma_queue adds it).
        bool tma_holding_changed = false;
        std::optional<TmaReadHolding> tma_holding_after;
        std::optional<TmaCommand> tma_pushed;
        // The dispatch pipeline's end: its unit takes an instruction now.
        bool dispatched = false;
        // The arithmetic pipeline's stage 0: its instruction with the round
        // it sampled now.
        std::optional<ArithmeticRead> arithmetic_sampled;
        DispatchDecision dispatch;
        // The async RF readers (the read ports the issue register leaves free now), before issue.
        RfReadDecision rf_read;
        std::optional<AccStBeat> acc_st_read;               // the acc.st engine's beat read now
        LsuRfRead lsu_rf_read;                              // the LSU store-data beat read now
        // Issue.
        IssueDecision issue;
        // The front ends after this cycle (step_front_end, after issue).
        std::vector<WarpFrontEnd> front_end_next;
        // The RF read stage: dispatch entries' operands latched now (the
        // entry's stage, operands by read, uniform operands by uniform
        // read from position uniform_first). A later round of an mma.async
        // (the round counter's) latches only its uniform operands.
        struct DispatchSample {
            std::size_t stage = 0;
            bool later_round = false;
            Bounded<LaneWords, limits::round_reads> operands;
            std::size_t uniform_first = 0;
            std::vector<uint32_t> uniform_operands;
            // A masked entry's write mask as lanes, sampled from the
            // predicate file now: ANDed into its active mask.
            std::optional<uint64_t> write_mask_lanes;
        };
        std::vector<DispatchSample> dispatch_samples;
        // The URF write pipeline: the write at the RAM now (it lands at this
        // cycle's edge), the operands its stage-0 write samples now (the read
        // stage), and the pipeline's work by stage (step_urf).
        std::optional<UrfWrite> urf_landing;
        std::optional<std::vector<uint32_t>> urf_sample;
        // The URF write pipeline's work this cycle, by stage: the write at
        // that stage as it goes on (operands latched, value computed), or
        // nothing for a usetp leaving at the write port's stage.
        std::vector<std::pair<std::size_t, std::optional<UrfWrite>>> urf_stage_work;
        // The round counter's tma round sampled now (the URF's read ports).
        std::optional<TmaReadPair> tma_sampled;
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
    // The roll call of everything clocked below: tick() ends with
    // clocked_.commit(), and the state report walks it.
    AllClocked clocked_{sanitizer_, cycle_};

    // ---- Memories (src/machine/memory.hpp: read and written through their ports) ----
    // Each memory is a black box (src/machine/blackbox.hpp): between its
    // markers are its description, its shape (built in the constructor) and
    // its declaration as a black box with its sign-off line (read by
    // declare_black_boxes); memory_blackbox, their shared part, gives the
    // cost.
    std::vector<Warp> warps_;                     // + the predicate files
    // BLACK BOX: vector register file
    // BLACK BOX USES: memories
    // BLACK BOX USES: memory shape
    // BLACK BOX USES: memory notes
    // BLACK BOX USES: bit widths
    // The vector register file: rf.banks memories (one per bank) of
    // rf.rows_per_bank rows, a row one lane-wide register (lanes x 32 bits,
    // a write enables lanes one by one); rf.read_ports_per_bank read ports
    // and rf.write_ports_per_bank write ports per bank. A read sampled at S
    // has its words at the output S + Lr later (the pipelines that carry
    // them: the arithmetic and dispatch pipelines, the acc.st and store-data
    // read pipelines); a write lands at the edge of its RAM cycle (read
    // first, src/timing.hpp). Warp w's register r is at rf_address(w, r).
    Memory<uint32_t> rf_{clocked_,Unit::register_file, "vector register file"};
    [[nodiscard]] MemoryShape rf_shape() const {
        return MemoryShape{config_.rf_banks, config_.rf_rows_per_bank, config_.lanes, uint64_t{32} * config_.lanes,
            config_.rf_read_ports_per_bank, config_.rf_write_ports_per_bank, config_.rf_read_latency_cycles,
            SameCycle::read_first};
    }
    [[nodiscard]] BlackBoxDeclaration rf_black_box() const {
        return memory_blackbox(rf_,
            SignOff::not_signed, // sign-off: only the human changes this line
            "rf.read_latency_cycles", "the issue register's round, the acc.st and store-data readers (in the open)",
            "the RF write pipelines' last stages (in the open)");
    }
    // END BLACK BOX: vector register file
    // A warp's rows in each bank: the kernel's register count over the banks,
    // rounded up (rf_address).
    uint32_t rf_rows_per_warp_ = 0;
    // BLACK BOX: uniform register file
    // BLACK BOX USES: memories
    // BLACK BOX USES: memory shape
    // BLACK BOX USES: memory notes
    // BLACK BOX USES: bit widths
    // BLACK BOX USES: uniform register file ports
    // The uniform register file: one memory of core.warps x urf.registers
    // 32-bit words (LUTRAM), warp w's register r at row w x urf.registers +
    // r. The SM's two read ports (urf_read_ports, src/timing.hpp: an
    // instruction's round, the TMA's and the mma.async's later rounds, the
    // uniform ALU's sample) and one write port (the URF write pipeline's RAM
    // stage). A read sampled at S has its word at the output S + Lu later;
    // a write lands at the edge of its RAM cycle (read first).
    Memory<uint32_t> urf_{clocked_,Unit::uniform_file, "uniform register file"};
    [[nodiscard]] MemoryShape urf_shape() const {
        return MemoryShape{1, std::size_t{config_.core_warps} * config_.urf_registers, 1, 32, urf_read_ports,
            urf_write_ports, config_.urf_read_latency_cycles, SameCycle::read_first};
    }
    [[nodiscard]] BlackBoxDeclaration urf_black_box() const {
        return memory_blackbox(urf_,
            SignOff::not_signed, // sign-off: only the human changes this line
            "urf.read_latency_cycles",
            "the issue register's round, the round counter, the URF write pipeline's sample (in the open)",
            "the URF write pipeline's last stage (in the open)");
    }
    // END BLACK BOX: uniform register file
    // BLACK BOX: shared memory
    // BLACK BOX USES: memories
    // BLACK BOX USES: memory shape
    // BLACK BOX USES: memory notes
    // BLACK BOX USES: bit widths
    // Shared memory: shmem.banks memories (banks) of 4-byte words, word w of
    // the byte address space in bank w mod banks, row w / banks (shmem.bytes
    // over 4 x banks rows: 1024 at the canonical profile); a write enables
    // bytes one by one. (An mbar's two words are read and written in one
    // round, so on a one-bank profile the words are 8 bytes: shmem_word_bytes_.
    // The bank rounds still count 4-byte words.) Two read and two write ports
    // per bank: port A (the
    // A-read service's sample; the A-write service: stores, RMW writes) and
    // port B (the tensor core's operand fetch; the TMA's placement). A
    // round reaches the RAM D (shmem.port_a_depth) cycles after its grant
    // (the services' pipelines), and a read's words are at the port output Lb
    // (shmem.read_latency_cycles) after its sample; a write lands at the
    // edge of its RAM cycle (read first).
    Memory<uint8_t> shmem_{clocked_,Unit::shared_memory, "shared memory"};
    [[nodiscard]] MemoryShape shmem_shape() const {
        // An mbar's two words are read and written in one round (a probe, an
        // RMW's read and write), so a one-bank shared memory is 8 bytes wide.
        const uint32_t word_bytes = config_.shmem_banks == 1 ? 8 : 4;
        const std::size_t line_bytes = std::size_t{word_bytes} * config_.shmem_banks;
        return MemoryShape{config_.shmem_banks, (config_.shmem_bytes + line_bytes - 1) / line_bytes, word_bytes,
            uint64_t{8} * word_bytes, 2, 2, config_.shmem_read_latency_cycles, SameCycle::read_first};
    }
    [[nodiscard]] BlackBoxDeclaration shmem_black_box() const {
        return memory_blackbox(shmem_,
            SignOff::not_signed, // sign-off: only the human changes this line
            "shmem.read_latency_cycles", "the A-read and port-B read pipelines' RAM stage (in the open)",
            "the A-write and port-B write pipelines' last stages (in the open)");
    }
    // END BLACK BOX: shared memory
    static constexpr uint32_t shmem_port_a = 0;
    static constexpr uint32_t shmem_port_b = 1;
    uint32_t shmem_word_bytes_ = 4; // its word's bytes (shmem_shape)
    std::vector<uint8_t> gmem_;                   // the controller's (outside the SM)
    // BLACK BOX: tensor memory
    // BLACK BOX USES: memories
    // BLACK BOX USES: memory shape
    // BLACK BOX USES: memory notes
    // BLACK BOX USES: bit widths
    // tmem: tmem.fine_banks memories (fine banks) of tmem.words_per_fine_bank
    // 32-bit words; physical word w (src/layout.hpp) is in fine bank
    // w % fine_banks, row w / fine_banks. Each fine bank's two read and two
    // write ports belong to the MMA (read port 0: its C read; write port 0:
    // its results) and to the accumulator engines (read port 1: acc.ld; write
    // port 1: acc.st). A request reaches the RAM tmem.port_depth after it
    // leaves its producer and a read's words are at the output
    // tmem.read_latency_cycles after the sample (the MMA core's tags and write
    // pipeline, the acc.ld and acc.st pipelines); a write lands at the edge of its RAM
    // cycle (read first).
    Memory<uint32_t> tmem_{clocked_,Unit::tensor_core, "tensor memory"};
    [[nodiscard]] MemoryShape tmem_shape() const {
        return MemoryShape{config_.tmem_fine_banks, config_.tmem_words_per_fine_bank, 1, 32, 2, 2,
            config_.tmem_read_latency_cycles, SameCycle::read_first};
    }
    [[nodiscard]] BlackBoxDeclaration tmem_black_box() const {
        return memory_blackbox(tmem_,
            SignOff::not_signed, // sign-off: only the human changes this line
            "tmem.read_latency_cycles",
            "the MMA core's step tags and the acc.ld pipeline at their RAM stage (in the open)",
            "the MMA result write and acc.st pipelines' last stages (in the open)");
    }
    // END BLACK BOX: tensor memory
    static constexpr uint32_t tmem_port_mma = 0;
    static constexpr uint32_t tmem_port_acc = 1;
    [[nodiscard]] uint32_t read_tmem(uint32_t word, uint32_t port, const char* tag) {
        return tmem_.read(word % config_.tmem_fine_banks, word / config_.tmem_fine_banks, tag, port)[0];
    }
    [[nodiscard]] std::size_t tmem_words() const {
        return std::size_t{config_.tmem_fine_banks} * config_.tmem_words_per_fine_bank;
    }
    // BLACK BOX: divergence stack
    // BLACK BOX USES: memories
    // BLACK BOX USES: memory shape
    // BLACK BOX USES: bit widths
    // The divergence stack memory: core.warps x core.divergence_stack_entries
    // entries of 32 + core.lanes bits (DivergenceEntry: 48 canonical), warp
    // w's entry i at row w x entries + i (16 x 8 x 48 = 6144 bits canonical).
    // One write port and one read port: the one control instruction resolving
    // in the issue register writes a push, which lands at the clock edge of
    // its resolve cycle, or presents a pop's address (divergence_refill_, the
    // address register), whose entry is at the read port's output the next
    // cycle (an asynchronous read of the registered address: read delay 1).
    // A read and a write in one cycle are two warps' entries; the read would
    // see the old entry (read first).
    Memory<DivergenceEntry> divergence_stack_{clocked_,Unit::branches, "divergence stack"};
    [[nodiscard]] MemoryShape divergence_stack_shape() const {
        return MemoryShape{1, std::size_t{config_.core_warps} * config_.core_divergence_stack_entries, 1,
            uint64_t{32} + config_.lanes, 1, 1, 1, SameCycle::read_first};
    }
    [[nodiscard]] BlackBoxDeclaration divergence_stack_black_box() const {
        return memory_blackbox(divergence_stack_,
            SignOff::not_signed, // sign-off: only the human changes this line
            "a pop's new top the next cycle", "the divergence stack memory read register (in the open)",
            "the issue register's branch (a push; in the open)");
    }
    // END BLACK BOX: divergence stack
    // BLACK BOX: instruction memory
    // BLACK BOX USES: memories
    // BLACK BOX USES: memory shape
    // BLACK BOX USES: memory notes
    // BLACK BOX USES: bit widths
    // BLACK BOX USES: instruction memory read
    // The instruction memory: the kernel's words, two to a row (a fetch
    // packet: the two 64-bit words at an aligned 16-byte address), loaded by
    // the host before the first cycle and never written. One read port: a
    // fetch's address is sampled at its grant's edge (fetch_read_, the
    // address register) and the packet is at the output the next cycle
    // (instruction_read_cycles, src/timing.hpp). Its size is the kernel's: the
    // profile has none.
    Memory<uint64_t> instruction_memory_{clocked_,Unit::front_end, "instruction memory"};
    [[nodiscard]] MemoryShape instruction_memory_shape() const {
        return MemoryShape{1, (kernel_.words.size() + 1) / 2, 2, 128, 1, 0, instruction_read_cycles,
            SameCycle::read_only};
    }
    [[nodiscard]] BlackBoxDeclaration instruction_memory_black_box() const {
        return memory_blackbox(instruction_memory_,
            SignOff::not_signed, // sign-off: only the human changes this line
            "a fetch's packet the next cycle", "the instruction memory read register (in the open)", "");
    }
    // END BLACK BOX: instruction memory

    // ---- Registers: the declared storage boundaries ----
    // The front end: per warp slot its FIFO,
    // PCs and halted bit; the instruction memory's read in flight; the fetch
    // round-robin cursor (4 bits at 16 warps); the issue register's branch.
    Regs<WarpFrontEnd> front_end_{clocked_,Unit::front_end, "instruction buffers and PCs"}; // per warp slot
    Reg<std::optional<FetchRead>> fetch_read_{clocked_,Unit::front_end, "instruction memory read (address, tag)"};
    Reg<uint32_t> fetch_cursor_{clocked_,Unit::front_end, "fetch round-robin cursor"};
    Reg<std::optional<IssuedBranch>> issue_branch_{clocked_,Unit::branches, "issue register: branch"};
    // Per warp slot (core.warps) its active mask, stack depth and top copy;
    // and the stack memory's one read in flight.
    Regs<WarpDivergence> divergence_{clocked_,Unit::branches, "active masks, divergence stack depths and tops"};
    Reg<std::optional<DivergenceRefill>> divergence_refill_{clocked_,Unit::branches, "divergence stack memory read"};
    // Scheduler-visible state other units complete.
    // (Per warp slot, core.warps; sync.mbar_wait_slots slots.)
    Regs<DepbarCounters> depbar_{clocked_,Unit::issue, "depbar counters"};
    uint32_t depbar_counter_max_ = 0;
    Regs<std::optional<MbarWaitSlot>> mbar_wait_slots_{clocked_,Unit::barriers, "mbar wait slots"};
    // shmem: the SIMT service slots, the probe register, the RMW
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
    Regs<std::optional<LoadSlot>> load_slots_{clocked_,Unit::shared_memory, "load slots"};    // shmem.load_slots
    Regs<std::optional<StoreSlot>> store_slots_{clocked_,Unit::shared_memory, "store slots"}; // shmem.store_slots
    Reg<std::optional<MbarReadRequest>> mbar_probe_{clocked_,Unit::barriers, "SIMT mbar probe"};
    Reg<std::optional<MbarRmwRequest>> simt_rmw_{clocked_,Unit::barriers, "SIMT RMW register"};
    Reg<RmwEngine> rmw_{clocked_,Unit::barriers, "RMW engine"};
    // A-read: D stages to the RAM, where the round's words are sampled (stage
    // D - 1), then Lb to the port output (the last stage). A-write and port-B
    // write: D stages to the RAM (the last stage), where the bytes land.
    Pipe<ReadBeat> a_read_pipe_{clocked_,Unit::shared_memory, "A-read pipeline"};
    Pipe<WriteBeat> a_write_pipe_{clocked_,Unit::shared_memory, "A-write pipeline"};
    Pipe<WriteBeat> b_write_pipe_{clocked_,Unit::shared_memory, "port-B write pipeline"};
    // The RMW engine's ALU (shmem.rmw_compute_cycles = C stages): the update
    // computed from the read's words at the port output enters stage 0 at that
    // cycle's edge; it is ready in the last stage C cycles after the port
    // output, and its write is granted then.
    Pipe<RmwUpdate> rmw_alu_{clocked_,Unit::barriers, "RMW engine: ALU stages"};
    Reg<ShmemArbiters> shmem_arbiters_{clocked_,Unit::shared_memory, "shmem A-read / A-write arbiters"};
    // fence.sh, per warp: how many cycles from now until
    // the warp's latest granted shmem round reaches the RAM (a write's edge,
    // a read's sample: grant + D). A grant at g sets it to D - 1 at g + 1; it
    // counts down to 0, the cycle that round is at the RAM. 0..D-1:
    // ceil(log2(D)) bits per warp slot (core.warps).
    Regs<uint32_t> shmem_to_ram_{clocked_,Unit::shared_memory, "per-warp shmem cycles to the RAM"};
    Reg<std::optional<MbarRmwRequest>> tma_mbar_event_{clocked_,Unit::barriers, "TMA mbar event register"};
    Reg<std::optional<MbarRmwRequest>> mma_mbar_event_{clocked_,Unit::barriers, "MMA mbar event register"};
    // MMA and accumulator transfers.
    Reg<MmaPipeline> mma_{clocked_,Unit::tensor_core, "MMA queue and pipeline"};
    // The port-B read pipeline: D stages to the shmem RAM (a round's elements
    // are sampled at stage D - 1), then Lb to the port output, the last stage.
    Pipe<MmaRead> mma_reads_{clocked_,Unit::tensor_core, "MMA port-B read pipeline"};
    // The steps in the core and on their way to tmem (MmaTag above): the
    // core's black box and its output register, the tags beside it (result_latency
    // stages; the C read at the RAM in stage port_depth - 1), the results'
    // write pipeline (tmem.port_depth stages to the tmem RAM, the last).
    // BLACK BOX SHARED: MMA core output register
    BlackBox<MmaBlock, std::deque<MmaCoreStep>> mma_core_{clocked_,Unit::tensor_core, "MMA core"};
    // END BLACK BOX SHARED: MMA core output register
    Pipe<MmaTag> mma_tags_{clocked_,Unit::tensor_core, "MMA core: step tags"};
    Pipe<MmaWrite> mma_writes_{clocked_,Unit::tensor_core, "MMA result write pipeline"};
    Reg<AccEngines> acc_engines_{clocked_,Unit::tensor_core, "acc transfer engines"};
    // The acc.ld engine's pipeline: tmem.port_depth stages to the tmem RAM
    // (sampled at stage port_depth - 1), then tmem.read_latency_cycles to the
    // output, the last stage, which holds the whole pipeline while the RF
    // write decision denies it.
    Pipe<AccLdBeat> acc_ld_pipe_{clocked_,Unit::tensor_core, "acc.ld pipeline"};
    // The acc.st engine's beats on their way to tmem: the RF's Lr read stages,
    // then tmem.port_depth stages to the RAM, the last stage.
    Pipe<AccStBeat> acc_st_pipe_{clocked_,Unit::tensor_core, "acc.st pipeline"};
    // TMA.
    Reg<TmaQueue> tma_{clocked_,Unit::tma, "TMA command queue"};
    Pipe<TmaWalkerStart> tma_walker_start_pipe_{clocked_,Unit::tma, "TMA walker start pipeline"};
    // The TMA's shmem transactions. Round 0 of each is granted in the cycle
    // tma_requests builds it (bypass (h) in tick()). The placement (port-B
    // write) register holds the rounds after the first; the A-read one holds
    // a descriptor read until its last round's words are back, a
    // store-source read until its last round's grant.
    Reg<std::optional<TmaPortRequest>> tma_write_txn_{clocked_,Unit::tma, "TMA placement transaction"};
    Reg<std::optional<TmaReadTransaction>> tma_read_txn_{clocked_,Unit::tma, "TMA read transaction"};
    Reg<TmaSectorRegister> tma_sector_{clocked_,Unit::tma, "TMA sector register"};
    // The placement tickets (TmaReturn): tma.sector_credits entries, one
    // write port (muxed between the ingress's completed sector and the
    // walker's zero fill; the sector has it first).
    Fifo<TmaReturn> tma_returns_{clocked_,Unit::tma, "TMA placement tickets"};
    // tma.store_sector_registers registers, and the ring's head (the oldest
    // held one, which the egress drains).
    Regs<std::optional<TmaStoreRegister>> tma_store_registers_{clocked_,Unit::tma, "TMA store sector registers"};
    Reg<uint32_t> tma_store_head_{clocked_,Unit::tma, "TMA store sector registers: head"};
    // The URF's round counter (UrfReadRounds above), then the TMA operand
    // read's URF read stages with their tags (Lu stages) and holding register.
    Reg<std::optional<UrfReadRounds>> urf_read_rounds_{clocked_,Unit::uniform_file, "URF operand read: round counter"};
    Pipe<TmaReadPair> tma_read_pairs_{clocked_,Unit::tma, "TMA operand read: URF read stages and tags"};
    Reg<std::optional<TmaReadHolding>> tma_holding_{clocked_,Unit::tma, "TMA operand read: holding register"};
    // LSU.
    // Sizes: command slots lsu.active_commands (2; checked in
    // drive_lsu_commands); one load buffer and one store buffer (LsuBuffer);
    // the ingress FIFO (IngressWord), the egress FIFO (EgressOpenSlot and its
    // two FIFOs) and the read records (LsuRead, TmaRun) as their comments
    // say, each built from the profile (a push past one is an internal
    // error); free-ID tags lsu.sectors_in_flight bits (16, fixed).
    Reg<LsuCommandSlots> lsu_commands_{clocked_,Unit::load_store, "LSU command slots"};
    Regs<LsuBuffer> lsu_buffers_{clocked_,Unit::load_store, "LSU load and store buffers"}; // [load_buffer], [store_buffer]
    static constexpr uint32_t load_buffer = 0;
    static constexpr uint32_t store_buffer = 1;
    Fifo<IngressWord> ingress_{clocked_,Unit::load_store, "ingress FIFO"};
    Reg<std::optional<EgressOpenSlot>> egress_open_{clocked_,Unit::load_store, "egress FIFO: the write port's open slot"};
    Fifo<EgressRequest> egress_requests_{clocked_,Unit::load_store, "egress FIFO: write requests"};
    Fifo<EgressSector> egress_data_{clocked_,Unit::load_store, "egress FIFO: sectors"};
    // The read records, one FIFO per read stream, and the bytes consumed of
    // each head's sector.
    Fifo<TmaRun> tma_reads_{clocked_,Unit::tma, "read records: TMA stream"};
    Fifo<LsuRead> lsu_reads_{clocked_,Unit::load_store, "read records: LSU stream"};
    Reg<uint32_t> tma_read_received_{clocked_,Unit::tma, "read records: TMA stream head's bytes consumed"};
    Reg<uint32_t> lsu_read_received_{clocked_,Unit::load_store, "read records: LSU stream head's bytes consumed"};
    Regs<uint8_t> free_id_tags_{clocked_,Unit::load_store, "free-ID tags", 1}; // lsu.sectors_in_flight bits; 1 = in flight
    // A store whose last acknowledgement came at t completes at t + 1: its
    // command slot frees at the RF write port and its depbar counts down at
    // the RAM stage (the memory controller is decided after both). One bit per
    // LSU command slot (lsu.active_commands): the slot still holds the
    // command's warp and depbar.
    Regs<uint8_t> store_completions_{clocked_,Unit::load_store, "store completions", 1};
    // The RF write path (src/timing.hpp): the fixed units (black boxes, each
    // with its output register), then one write pipeline per RF write port
    // (bank x rf.write_ports_per_bank + port), from the port to the RAM.
    // BLACK BOX SHARED: arithmetic output registers
    std::array<BlackBox<ArithmeticResult>, arithmetic_units> arithmetic_black_boxes_{{
        {clocked_,Unit::arithmetic, "ALU"},
        {clocked_,Unit::arithmetic, "FPU"},
        {clocked_,Unit::arithmetic, "special-function unit"},
    }};
    // END BLACK BOX SHARED: arithmetic output registers
    // The units slowest first: at one port cycle their results are presented
    // in the order they executed.
    std::array<ArithmeticUnit, arithmetic_units> units_slowest_first_{};
    std::deque<Pipe<RfWrite>> rf_write_pipes_;
    // The predicate files: per warp slot (core.warps) seven predicates of
    // core.lanes bits, flip-flops, not a memory (src/timing.hpp): a predicate
    // presented at W is in the file from W + 1. Predicate p of warp w is
    // member w x 7 + p (predicate_index); bit l is lane l.
    Regs<uint64_t> predicates_{clocked_,Unit::register_file, "predicate files"};
    static constexpr uint32_t predicates_per_warp = 7;
    [[nodiscard]] static std::size_t predicate_index(uint32_t warp, uint32_t predicate) {
        return std::size_t{warp} * predicates_per_warp + predicate;
    }
    Regs<RfWritePolicy> rf_write_policy_{clocked_,Unit::register_file, "RF write policy (async_owed, cursor), per bank"};
    Regs<RfReadPolicy> rf_read_policy_{clocked_,Unit::register_file, "RF read policy (read_owed, cursor), per bank"};
    // The issue register's RF bank use: per bank,
    // how many registers the instruction accepted last cycle samples now (at
    // most rf.read_ports_per_bank: one bit per bank at the canonical one
    // port; zero when nothing was accepted). It owns those read ports this
    // cycle; the async readers take the rest (decide_rf_reads).
    Regs<uint8_t> issue_read_banks_{clocked_,Unit::issue, "issue register: RF read bank use"}; // per bank
    // Issue -> units: the issue register and the RF read pipeline
    // (issue_to_dispatch stages), carrying memory / engine instructions.
    // issue_to_dispatch stages: an instruction accepted at t is in stage k at
    // t + 1 + k and dispatches from the last stage.
    Pipe<DispatchEntry> dispatch_pipe_{clocked_,Unit::dispatch, "dispatch pipeline"};
    // The LSU's store-data RF reads travelling to the store buffer: the RF's
    // Lr read stages. A read granted at t is in stage 0 at t + 1 and at the RF
    // output, the last stage, at t + Lr.
    Pipe<LsuRfRead> lsu_read_pipe_{clocked_,Unit::load_store, "LSU store-data read pipeline"};
    // The URF write pipeline (UrfWrite): L + Wu stages, the uniform ALU's and
    // r2ur's path from the issue register to the URF RAM.
    Pipe<UrfWrite> urf_pipe_{clocked_,Unit::uniform_file, "URF write pipeline"};

    // ---- Unit-internal state (only its own unit's code touches it) ----
    // Issue: scheduler cursor, SFU occupancy. (Read only by the scheduler
    // before it accepts, written at acceptance: fetch never reads them.)
    uint32_t scheduler_cursor_ = 0;
    // First cycle the SFU is free again. The unit is occupied from an SFU
    // op's execution start (issue + issue_to_operands) for ceil(lanes /
    // sfu_lanes) cycles; a later SFU op stalls structurally until its own
    // execution start fits.
    uint64_t sfu_busy_until_ = 0;
    // The arithmetic pipeline's operand latches: issue_to_operands stages. An
    // instruction accepted at t samples its round in stage 0 (t + 1) and
    // executes from the last stage (t + issue_to_operands), its operands at
    // the RF output.
    Pipe<ArithmeticRead> arithmetic_reads_{clocked_,Unit::arithmetic, "arithmetic pipeline: operand latches"};
    // The RF write port's reservation calendar (the arbiter's schedule): a
    // shift register per bank, one count of booked ports per stage (one bit
    // per bank at one write port). The last stage is this cycle's port, the
    // stage before it the next cycle's, and so on: issue_to_port(slowest fixed
    // op) + 1 stages. Fixed issue books a FUTURE cycle only, its write at
    // issue + issue_to_port: the booking enters at the stage that many edges
    // from the last one; the async writers use the current write port. (There
    // is no read calendar: reads are one round at issue + 1, owned by the
    // issue register -- issue_read_banks_.)
    Pipe<Bounded<uint32_t, limits::rf_banks>> write_calendar_{clocked_,Unit::register_file, "RF write calendar"};
    // gmem controller (outside the SM): its queue, bus and B channel. The
    // queue and the bus entry together hold at most gmem.queue_sectors (64;
    // checked in step_gmem_controller). The B channel holds an ack per write
    // whose last beat has left and whose ack is not yet due: at most one per
    // write in flight -- the TMA's (each holds a credit: tma.sector_credits)
    // and the LSU's (lsu.active_commands commands of at most max(lanes, 8 x
    // lanes x 4 / sector bytes) requests): 16 + 2 x 16 = 48 at the canonical
    // profile; checked in finish_gmem_write.
    std::deque<GmemSectorRequest> gmem_queue_;
    std::optional<GmemSectorRequest> gmem_bus_;
    // The bus's last data beat: its DRAM tick and direction (true: a read).
    // The controller's own state, for gmem.turnaround_dram_ticks.
    std::optional<uint64_t> gmem_last_beat_tick_;
    bool gmem_last_beat_load_ = false;
    std::deque<WriteAck> pending_write_acks_;
    ControllerTestOrder controller_test_order_ = ControllerTestOrder::fifo;
    // Request channel: TMA/LSU alternate (the client granted last yields a tie).
    GmemClient request_channel_last_ = GmemClient::lsu;
    std::optional<uint32_t> last_served_read_stream_; // interleave_ordered_streams test order
    uint64_t dram_tick_ = 0;
    uint64_t dram_phase_ = 0;
    // Id counters, advanced only by the decisions that allocate (fixed order).
    uint64_t next_egress_slot_id_ = 0;
    uint64_t next_lsu_command_id_ = 0;
    uint64_t next_gmem_request_id_ = 0;
    uint64_t next_tma_command_id_ = 0;
    // The launch record (src/image.hpp): the registers the host writes before
    // the first cycle, read-only after. None: nothing to copy.
    struct LaunchRegisters {
        uint64_t global_address = 0;
        uint32_t shared_address = 0;
        uint32_t bytes = 0;
        Bounded<LaunchFlag, 6> flags; // at most core.launch_flags, one per depbar
    };
    Reg<std::optional<LaunchRegisters>> launch_record_{clocked_,Unit::launch, "launch record"};
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
    [[nodiscard]] std::optional<FetchedPacket> instruction_memory_output();
    [[nodiscard]] std::optional<BranchResolution> resolve_branch() const;
    [[nodiscard]] std::optional<std::pair<uint32_t, DivergenceEntry>> divergence_memory_output();
    void land_divergence_push(const Wires& w);
    // Whether the control instruction in the issue register holds its warp
    // (no issue, no fetch) until it resolves: every one but ssy.
    [[nodiscard]] static bool holds_its_warp(const IssuedBranch& branch);
    // After issue: every warp's front end after this cycle (pop, landing,
    // halt, redirect, fetch pc), and its trace.
    void step_front_end(Wires& w);
    // ---- RF write path: the one async decision, the port, the RAM stage ----
    [[nodiscard]] RfWriteDecision decide_rf_writes() const;
    [[nodiscard]] std::vector<std::optional<AsyncWrite>> ready_async_writers(std::optional<uint32_t> lsu_presented) const;
    [[nodiscard]] std::optional<AsyncWrite> lsu_buffer_rf_write(uint32_t index) const;
    [[nodiscard]] std::optional<uint32_t> lsu_rf_presented() const;
    [[nodiscard]] uint32_t writes_booked_now(uint32_t bank) const;
    void rf_write_port(Wires& w);
    void rf_ram_stage(Wires& w);
    void land_rf_writes(const Wires& w);
    void release_lsu_command(Wires& w, uint64_t id);
    // A granted async holder's write, and the LSU command its write ends
    // (the load buffer's last write of a command).
    RfWrite async_rf_write(const AsyncWrite& write, std::optional<uint64_t>& ends_command);
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
    // shmem: the RAM stage, one select per service, their data paths.
    void step_shmem(Wires& w);
    void shmem_ram_stage(ShmemCycle& shmem);
    struct ShmemWord {
        uint32_t bank = 0;
        uint32_t row = 0;
        uint32_t byte = 0; // the byte's place in the word
    };
    [[nodiscard]] ShmemWord shmem_word(uint32_t byte_address) const;
    [[nodiscard]] uint32_t read_shmem_word(uint32_t byte_address, uint32_t port, const char* tag);
    void land_shmem_writes(const Wires& w);
    [[nodiscard]] std::size_t shmem_bytes() const { return config_.shmem_bytes; }
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
    void step_acc_ld(std::optional<AccTransfer>& engine, Wires& w);
    std::optional<AccLdBeat> launch_acc_ld_beat(std::optional<AccTransfer>& engine);
    void sample_acc_ld_beat(AccLdBeat& beat);
    void step_acc_st(std::optional<AccTransfer>& engine, Wires& w);
    void decide_mbar_timeouts(Wires& w);
    IngressPluck decide_ingress(const Wires& w);
    void step_gmem_controller(Wires& w);
    void step_tma(Wires& w);
    // ---- the TMA operand read's end: the pair at the URF output now, the push ----
    void arrive_tma_operands(Wires& w);
    [[nodiscard]] TmaCommand tma_command_from(const TmaReadHolding& holding, Wires& w);
    [[nodiscard]] uint32_t tma_commands_being_read() const;
    [[nodiscard]] bool urf_read_ports_held() const;
    // The value uniform register `reg` of `warp` will have at a sample next
    // cycle (the issue register's): its Q, or the write landing at this
    // cycle's edge. The sanitizer's issue-time look at a tmem pointer.
    [[nodiscard]] uint32_t uniform_at_next_sample(uint32_t warp, uint8_t reg) const;
    // ---- dispatch: the RF read pipeline's end, the unit takes it ----
    void dispatch_issued(Wires& w);
    void hand_over(const DispatchEntry& op, Wires& w);
    void complete_without_lanes(const DispatchEntry& op, Wires& w);
    void dispatch_mma(const DispatchEntry& op, Wires& w);
    void dispatch_acc_transfer(const DispatchEntry& op, Wires& w);
    void dispatch_shared_load(const DispatchEntry& op, Wires& w);
    void dispatch_shared_store(const DispatchEntry& op, Wires& w);
    void dispatch_lsu(const DispatchEntry& op, Wires& w);
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
    // An accepted uniform-register writer (uniform arithmetic, r2ur): its
    // write enters the URF write pipeline; r2ur's vector read joins the round.
    void accept_uniform(Warp& warp, const Instruction& inst, const IssuePlan& plan, Wires& w);
    // An accepted memory / engine / mbar instruction: its metadata work now
    // (depbar, the sanitizer's destination and reader records), its read
    // round's bank use, its entry into the dispatch pipeline.
    void accept_for_dispatch(Warp& warp, const Instruction& inst, const IssuePlan& plan, Wires& w);
    // An accepted tma or mma.async: it loads the URF's round counter.
    void load_urf_rounds(Warp& warp, const Instruction& inst, Wires& w);
    void panic_if_malformed(const Warp& warp, const Instruction& inst) const;
    void check_issue_hazards(const Warp& warp, const Instruction& inst) const;
    void check_issue_latency(const Warp& warp, const Instruction& inst) const;
    // ---- the RF read stage: this cycle's fixed samples ----
    void step_rf_reads(Wires& w);
    // ---- the URF write pipeline: its write port and RAM stage (from Q),
    // and its next state after the read stage (the ALU, r2ur, issue's entry) ----
    void urf_ram_stage(Wires& w);
    void step_urf(Wires& w);
    [[nodiscard]] uint32_t execute_uniform(const UrfWrite& write) const;
    void land_urf_write(const Wires& w);

    // ---- register drivers: next state from Q + wires; any order ----
    using Driver = void (Machine::*)(const Wires&);
    void drive_front_end(const Wires& w);
    void drive_fetch(const Wires& w);
    void drive_issue_branch(const Wires& w);
    void drive_divergence(const Wires& w);
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
    void drive_tma_walker_start(const Wires& w);
    void drive_tma_transactions(const Wires& w);
    void drive_tma_sector(const Wires& w);
    void drive_tma_store_registers(const Wires& w);
    void drive_tma_returns(const Wires& w);
    void drive_tma_operand_read(const Wires& w);
    void drive_lsu_commands(const Wires& w);
    void drive_lsu_buffers(const Wires& w);
    void drive_ingress(const Wires& w);
    void drive_egress(const Wires& w);
    void drive_read_records(const Wires& w);
    void drive_free_id_tags(const Wires& w);
    void drive_rf_write_pipes(const Wires& w);
    void drive_store_completions(const Wires& w);
    void drive_predicates(const Wires& w);
    void drive_load_slots(const Wires& w);
    void drive_acc_ld_pipe(const Wires& w);
    void drive_rf_write_policy(const Wires& w);
    void drive_rf_read_policy(const Wires& w);
    void drive_issue_read_banks(const Wires& w);
    void drive_dispatch_pipe(const Wires& w);
    void drive_arithmetic_reads(const Wires& w);
    void drive_write_calendar(const Wires& w);
    void drive_lsu_read_pipe(const Wires& w);
    void drive_urf_pipe(const Wires& w);
    void run_drivers(const Wires& w);

    // ---- helpers ----
    // Sanitizer argument builders: pure reads of hardware state whose results
    // go only to sanitizer_.
    [[nodiscard]] IssueSite site(const Warp& warp, const Instruction& inst) const;
    [[nodiscard]] IssueSite site(const DispatchEntry& op) const;
    [[nodiscard]] std::vector<uint32_t> access_addresses(const SlotRounds& access) const;
    [[nodiscard]] std::vector<std::vector<uint32_t>> round_addresses(const SlotRounds& access) const;
    [[nodiscard]] std::optional<std::pair<uint32_t, uint32_t>> mma_operand_span(
        uint32_t base, uint32_t spacing, bool a_operand, const MmaShape& shape) const;
    [[nodiscard]] std::vector<uint8_t> shmem_after_edge(uint32_t address, uint32_t size, const Wires& w) const;
    [[nodiscard]] std::array<uint8_t, tma_descriptor_bytes> tma_descriptor_at_accept(
        uint32_t address, const Wires& w) const;
    [[nodiscard]] std::optional<uint64_t> pending_tma_completion_mbar(uint32_t mbar, const Wires& w) const;

    // The RF write calendar. Issue books its fixed write (at issue +
    // issue_to_port), a future cycle; the async writers take the current
    // write port's free ports (decide_rf_writes).
    [[nodiscard]] bool write_slot_free(uint8_t reg, uint64_t at_cycle) const;
    void reserve_write(uint8_t reg, uint64_t at_cycle, Wires& w);
    [[nodiscard]] std::size_t write_calendar_stage(uint64_t at_cycle) const;

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
    // cycle's grants (look-ahead (c), reuse (d)) -- except the load slots'
    // count, which is Q only (a slot whose RF write is granted at t counts
    // for issue from t + 1).
    [[nodiscard]] std::optional<uint32_t> free_load_slot(const Wires& w) const;
    [[nodiscard]] std::optional<uint32_t> free_store_slot(const Wires& w) const;
    [[nodiscard]] uint32_t free_load_slot_count() const;
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
    RmwUpdate rmw_update(const ReadBeat& read);
    [[nodiscard]] static bool is_atomic_operation(MbarRmwOperation operation);
    [[nodiscard]] static RmwUpdate atomic_update(const MbarRmwRequest& request, uint32_t old_word);
    // The RMW's write round: the update's word at the operation's address.
    [[nodiscard]] WriteBeat rmw_write_round(const RmwUpdate& update) const;
    WriteBeat placement_round(const TmaPortRequest& request, bool start);
    void advance_tma_read(Wires& w);
    void fill_tma_store_register(const ReadBeat& beat, Wires& w);
    [[nodiscard]] std::optional<uint32_t> tma_store_register_free(const EgressWrite& egress) const;
    void deliver_probe(const ReadBeat& beat, Wires& w);
    void count_engine_wait_on_other(const Wires& w);
    [[nodiscard]] MbarEngine engine_to_claim(bool tma_held, bool mma_held) const;
    [[nodiscard]] static std::string rmw_label(const ActiveMbarRmw& active);
    [[nodiscard]] std::vector<uint32_t> tma_round_addresses(const TmaPortRequest& request,
        const std::vector<TmaPortRequest::Piece>& pieces) const;
    void wake_mbar_waiters(Wires& w, uint32_t byte_address, uint8_t new_phase);

    // MMA, acc.
    [[nodiscard]] bool acc_engine_available(const Wires& w, bool load) const;
    void advance_mma(MmaPipeline& mma, Wires& w);
    [[nodiscard]] std::optional<MmaFetch> next_mma_step(MmaPipeline& mma, Wires& w);
    [[nodiscard]] MmaRead grant_mma_round(MmaPipeline& mma);
    [[nodiscard]] std::optional<uint32_t> mma_element_address(const MmaStep& step, std::size_t element) const;
    [[nodiscard]] std::size_t mma_elements() const;
    void sample_mma_read(MmaPipeline& mma, MmaRead& read, Wires& w);
    void enter_mma_core(MmaPipeline& mma, const MmaStep& step, Wires& w);
    void read_mma_c(const MmaTag& tag);
    void finish_mma_step(const MmaBlock& c);
    void declare_mma_core_black_box();
    [[nodiscard]] static MmaTag mma_tag(const MmaStep& step);
    void land_tmem_writes(const Wires& w);

    // The state report's declarations (src/machine/state_report.cpp), and
    // the black boxes' contracts, each beside its black box.
    void declare_state_bits();
    void declare_black_boxes();
    std::vector<BlackBoxDeclaration> black_boxes_;
    [[nodiscard]] std::vector<std::pair<const Clocked*, std::string>> not_yet_declared() const;
    // The physical tmem word of logical word `logical` in a region whose rows
    // are row_words words (src/layout.hpp); dispatch panicked on any word
    // outside tmem, so past that this is an internal error.
    [[nodiscard]] std::size_t tmem_index(uint64_t logical, uint32_t row_words) const;
    // The physical words of `rows` rows x `columns` columns of the region at
    // `pointer` (row_words words per row), row-major; nullopt when one lies
    // outside tmem (the dispatch PANIC).
    [[nodiscard]] std::optional<std::vector<uint32_t>> tmem_region_words(
        uint32_t pointer, uint32_t row_words, uint32_t rows, uint32_t columns) const;
    [[nodiscard]] std::vector<uint32_t> mma_block_words(const MmaTag& tag) const;
    // An acc.ld / acc.st's region for the sanitizer: its pointer, the
    // instruction's words per row and width, its words (sorted here).
    [[nodiscard]] TmemRegion tmem_region(uint32_t pointer, const Instruction& inst, std::vector<uint32_t> words) const;
    void execute_arithmetic(const ArithmeticRead& pending);
    [[nodiscard]] static ArithmeticUnit arithmetic_unit(const Instruction& inst);
    void declare_arithmetic_black_boxes();

    // TMA.
    void apply_tma_requests(TmaQueue& tma, Wires& w);
    void apply_tma_walker(TmaQueue& tma, Wires& w);
    void complete_tma_frontend_read(TmaQueue& tma, const TmaReadTransaction& read, Wires& w);
    void complete_tma_load_write(TmaQueue& tma, const PlacementEnd& placed);
    void complete_tma_store_capture(TmaQueue& tma, uint8_t slot, uint64_t command);
    void mark_tma_data_complete(TmaCommand& command);
    // The launch copy (README "Launch"): the TMA's first command and walker
    // and the warps' launch flags, from the launch record; each landed
    // sector's bytes and the flags they clear.
    void start_launch_copy(const LaunchRecord& record);
    void land_launch_bytes(TmaQueue& tma, const PlacementEnd& placed, Wires& w);
    void release_tma_credit(TmaQueue& tma);
    void trace_tma_depths(const TmaQueue& tma, uint64_t at_cycle);
    // (T) The placement tickets as step_tma has left them so far this cycle:
    // what trace_tma_depths reports.
    std::size_t tma_returns_traced_ = 0;
    [[nodiscard]] std::size_t tma_store_registers_held() const;
    [[nodiscard]] std::optional<uint64_t> next_tma_frontend_command(const TmaQueue& tma) const;
    [[nodiscard]] std::optional<uint64_t> next_data_complete_tma_command(const TmaQueue& tma) const;
    // A TMA transaction's bank rounds, each a list of pieces (a wire: the
    // run's combinational decode), and the bounds they are checked against.
    using TmaRoundPlan = std::vector<std::vector<TmaPortRequest::Piece>>;
    [[nodiscard]] TmaRoundPlan tma_round_plan(const TmaPortRequest& request) const;
    [[nodiscard]] uint32_t tma_max_rounds() const;
    [[nodiscard]] uint32_t tma_max_pieces() const;
    [[nodiscard]] TmaPortRequest tma_port_request(TmaPortKind kind, const TmaRun& run, uint64_t sector_request = 0) const;
    [[nodiscard]] uint32_t tma_piece_address(const TmaRun& run, const TmaPortRequest::Piece& piece) const;
    [[nodiscard]] uint64_t run_sector_address(const TmaRun& run) const;
    [[nodiscard]] uint32_t run_sector_offset(const TmaRun& run) const;
    [[nodiscard]] uint64_t run_byte_enable(const TmaRun& run) const;
    [[nodiscard]] TmaWalkerPlan plan_tma_walker(const TmaCommand& command, const TmaWalker& walker,
        uint32_t credits_used, bool store_register_free) const;
    void validate_tma_runtime_descriptor(const TmaCommand& command, const TmaDescriptor& descriptor) const;
    [[nodiscard]] TmaWalker tma_walker_start(const TmaCommand& command, const TmaDescriptor& descriptor) const;

    // Issue.
    // The warp's FIFO head (Q), decoded; "fell off kernel" if its pc is past
    // the kernel's words (a word fetched there is reported only here).
    [[nodiscard]] Instruction head_instruction(const Warp& warp) const;
    [[nodiscard]] uint32_t head_pc(uint32_t warp) const { return front_end_.cur().at(warp).pc; }
    // The VECTOR registers an instruction reads and writes (its uniform ones:
    // uniform_sources / uniform_destination, src/isa.hpp).
    [[nodiscard]] std::vector<uint8_t> source_registers(const Instruction& inst) const;
    [[nodiscard]] std::optional<uint8_t> destination_register(const Instruction& inst) const;
    void check_register_hazards(const Warp& warp, const Instruction& inst) const;
    // A dispatched instruction's operand: every lane of one register, as its
    // RF read latched it at its sample.
    [[nodiscard]] std::vector<uint32_t> capture(const DispatchEntry& op, uint8_t reg) const;
    // A conditional branch's predicate, every lane, read from the predicate
    // file in the issue register's cycle (resolve_branch).
    [[nodiscard]] std::vector<uint32_t> capture_predicate(const Warp& warp, uint8_t predicate) const;
    // A masked instruction's write mask as lanes (bit l = lane l writes), read
    // from the predicate file in its operand round (step_rf_reads).
    [[nodiscard]] uint64_t write_mask_lanes(const Warp& warp, const Instruction& inst) const;
    // A uniform-register operand, as the round latched it at its sample:
    // uniform by construction (the partial-mask rule still applies).
    [[nodiscard]] uint32_t uniform_operand(const DispatchEntry& op, uint8_t ureg) const;
    [[nodiscard]] uint32_t mbar_address_operand(const DispatchEntry& op) const;
    [[nodiscard]] MbarState load_mbar(uint32_t byte_address) const;
    // Where a warp's vector register is in the register file's three
    // memories (rf_address, src/machine/rf.cpp), and a read of every lane of
    // it through one of its bank's read ports.
    struct RfAddress {
        uint32_t bank = 0;
        uint32_t row = 0;
    };
    [[nodiscard]] RfAddress rf_address(uint32_t warp, uint32_t reg) const;
    [[nodiscard]] std::vector<uint32_t> read_rf(uint32_t warp, uint8_t reg, const char* tag);
    // A uniform register through one of the URF's read ports now.
    [[nodiscard]] uint32_t read_urf(uint32_t warp, uint8_t reg, const char* tag);
    [[nodiscard]] std::size_t urf_row(uint32_t warp, uint32_t reg) const {
        return std::size_t{warp} * config_.urf_registers + reg;
    }
    // SIMT shmem bank rounds (lds/sts by bank decode; lds.uni: one access),
    // from the captured operands.
    [[nodiscard]] SlotRounds build_shmem_rounds(const DispatchEntry& op) const;
    [[nodiscard]] SlotRounds build_uniform_shmem_round(const DispatchEntry& op) const;
    // The lanes in round `round` of an access, in bank order (one per bank
    // at most): the round's words at the RAM.
    [[nodiscard]] Bounded<uint8_t, limits::shmem_banks> round_lanes(const SlotRounds& access, uint32_t round) const;
    [[nodiscard]] uint32_t shmem_address(const DispatchEntry& op, uint32_t base) const;
    void apply_image(
        std::vector<uint8_t>& memory,
        const std::vector<ImageSegment>& image,
        const std::string& name);
    [[noreturn]] void panic(const Warp& warp, const Instruction& inst, const std::string& reason) const;
    [[noreturn]] void panic(const DispatchEntry& op, const std::string& reason) const;
    [[noreturn]] void mbar_rmw_panic(
        const MbarRmwRequest& request, const std::string& reason) const;

    // ---- gmem controller + LSU (src/machine/gmem.cpp, src/machine/lsu.cpp) ----
    [[nodiscard]] std::size_t gmem_outstanding_sectors() const;
    [[nodiscard]] std::optional<std::size_t> select_controller_entry() const;
    void tick_gmem_controller(Wires& w, bool& return_ready_low);
    void finish_gmem_write(GmemSectorRequest request);
    void deliver_write_acks(Wires& w);
    [[nodiscard]] static bool is_store(LsuKind kind);
    LsuCommand plan_lsu_command(const DispatchEntry& op) const;
    [[nodiscard]] static bool store_has_data_left(const LsuCommand& command);
    [[nodiscard]] bool store_buffer_free() const;
    [[nodiscard]] uint32_t lsu_group_width(const LsuCommand& command) const;
    [[nodiscard]] uint64_t lsu_group_lanes(const LsuCommand& command, uint32_t group) const;
    [[nodiscard]] bool load_buffer_released(const Wires& w) const;
    bool load_buffer_takes(const LsuCommand& command, uint32_t group, IngressPluck& pluck, const Wires& w) const;
    [[nodiscard]] const LsuCommand* find_lsu_command(uint64_t id) const;
    [[nodiscard]] const LsuCommand& lsu_read_command(const LsuRead& record) const;
    [[nodiscard]] std::optional<uint32_t> free_id_tag() const;
    [[nodiscard]] const EgressRequest* egress_request_slot(bool head_requested_now) const;
    [[nodiscard]] const LsuCommand* lsu_request_candidate(bool head_requested_now) const;
    GmemSectorRequest write_request(const EgressRequest& slot, GmemClient client, uint64_t id,
        std::optional<uint32_t> tag);
    GmemSectorRequest lsu_read_request(const LsuCommand& command, uint64_t id);
    void pluck_tma(const IngressPiece& piece, const TmaRun& run, TmaSectorRegister& reg, IngressPluck& pluck);
    [[nodiscard]] bool pluck_vector(const LsuRead& record, const IngressPiece& piece, IngressPluck& pluck,
        const Wires& w);
    [[nodiscard]] bool pick_scalar_word(const LsuRead& record, const IngressPiece& piece, IngressPluck& pluck,
        const Wires& w);
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

// ---- Clocked state that is not yet a fixed bundle of bits ----
// Every register, group, pipeline, FIFO and memory of the machine holds a
// payload of a fixed size (machine/state.hpp checks it when this compiles).
// A payload that cannot be one yet is named here, one line each, as
//   template <> struct NotYetFixedPayload<T> : std::true_type {}; // what it still holds
// with what it would become. There is none.

} // namespace nanosm

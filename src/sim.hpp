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

class Machine {
public:
    Machine(Config config, Kernel kernel, Launch launch, TraceOptions trace_options = {});

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
        bool outstanding_ldg = false;
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
        bool clear_outstanding_ldg = false;
        // Sanitizer only: the final acc.ld beat releases its tmem read-ownership.
        std::optional<uint64_t> release_acc_transfer;
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
        std::vector<float> values;
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
        std::map<uint32_t, float> captured;
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

    struct GmemSectorRequest {
        enum class Owner : uint8_t {
            simt,
            tma_load,
            tma_store,
        };

        uint64_t id = 0;
        uint64_t command = 0;
        uint64_t sector_address = 0;
        uint64_t ready_dram_tick = 0;
        bool load = false;
        bool ready = true;
        Owner owner = Owner::simt;
        std::vector<LaneAccess> accesses;
        struct TmaFragment {
            uint64_t gmem_address = 0;
            uint32_t shmem_address = 0;
            uint8_t size = 0;
            std::array<uint8_t, 4> data{};
        };
        std::vector<TmaFragment> tma_fragments;
        uint32_t beats_remaining = 0;
    };

    struct GmemCommand {
        bool load = false;
        uint32_t warp = 0;
        uint8_t dst = 0;
        uint8_t depbar = 0;
        uint32_t remaining_sectors = 0;
        uint8_t width = 1;
        uint8_t next_store_capture = 0;
        std::vector<uint32_t> load_values;
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

    struct TmaPortRequest {
        struct Piece {
            uint32_t fragment = 0;
            uint8_t offset = 0;
            uint8_t size = 0;
        };
        TmaPortKind kind = TmaPortKind::descriptor;
        uint64_t command = 0;
        uint64_t sector_request = 0;
        uint32_t next_round = 0;
        std::vector<std::vector<Piece>> rounds;
        std::vector<GmemSectorRequest::TmaFragment> fragments;
    };

    struct TmaReturn {
        uint64_t command = 0;
        uint64_t sector_request = 0;
        std::vector<GmemSectorRequest::TmaFragment> fragments;
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
    std::map<uint64_t, GmemCommand> gmem_commands_;
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
    uint64_t next_gmem_command_id_ = 0;
    // One shared lane-wide SIMT global-load return path. This is the first
    // core cycle not yet assigned to a scalar/vector return beat.
    uint64_t gmem_return_available_cycle_ = 0;
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
    void advance_gmem_store_captures();
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
    [[nodiscard]] std::vector<std::vector<LaneAccess>> build_gmem_sectors(
        const Warp& warp, const Instruction& inst, bool load) const;
    [[nodiscard]] std::size_t gmem_outstanding_sectors() const;
    void complete_gmem_sector(GmemSectorRequest request);
    void complete_tma_port_request(TmaPortRequest request);
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
    [[nodiscard]] GmemSectorRequest* queued_gmem_request(uint64_t request);
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
// fp16 <-> fp32 bit conversions. The decode direction is the MMA operand
// decoder; the encode direction (RNE, overflow to inf, subnormal support,
// NaN keeps its truncated payload or the quiet bit) is cvt.pack.f16x2's
// spec. Finite and infinite halves round-trip exactly.
uint32_t half_to_float_bits(uint16_t half);
uint16_t float_to_half_bits(uint32_t bits);

} // namespace nanosm

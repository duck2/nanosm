#pragma once

// =============================================================================
// THE SANITIZER RULE
//
// The Sanitizer holds every piece of shadow state the simulator keeps only to
// decide whether to PANIC (ownership tables, hazard horizons, fall-through
// and port-collision tripwires, the ordered-ID tripwire). None of it is
// hardware. The machine's behavior -- every cycle, counter, result byte and
// trace event -- must never depend on it.
//
//   - The machine talks to it in one direction only. record_*() tells it what
//     happened; check_*() asks whether something is legal. Both return
//     nothing. Either may throw a PANIC (SimError); neither has any other
//     effect on the machine.
//   - There is no accessor, no bool, no "enabled" flag. Do not add one. If the
//     machine needs a fact, that fact is hardware state and belongs in the
//     machine's own (declared) structures, not here.
//   - Arguments are computed from hardware state; nothing flows back.
//
// Mechanical proof: NANOSM_SANITIZER=0 compiles this class to no-ops. The
// default build also builds that twin and ctest runs scripts/
// sanitizer_off_check.py, which requires every launch's stdout and Perfetto
// trace and every non-sanitizer-panic unit test to be byte-identical between
// the two. A change that makes computation depend on sanitizer state fails it.
// Classification of all machine state: trash/sanitizer-split-2026-10-07.md.
// =============================================================================

#include "compiler.hpp"
#include "isa.hpp"
#include "profile.hpp"
#include "tma.hpp"

#include <array>
#include <cstddef>
#include <deque>
#include <optional>
#include <stdint.h>
#include <string>
#include <utility>
#include <vector>

#ifndef NANOSM_SANITIZER
#define NANOSM_SANITIZER 1
#endif

namespace nanosm {

// The issuing warp and instruction a check is about (PANIC text only).
struct IssueSite {
    uint64_t cycle = 0;
    uint32_t warp = 0;
    uint32_t pc = 0;
    const Instruction& inst;
};

// PANIC text shared by Machine and Sanitizer, so a moved check keeps its
// message byte for byte. Always compiled.
[[noreturn]] void throw_issue_panic(const IssueSite& site, const std::string& reason);
// requester == nullptr: an engine's request ("source=engine").
[[noreturn]] void throw_mbar_rmw_panic(
    uint64_t cycle, uint32_t address, const std::string& reason, const IssueSite* requester);

// Which async reader of a register finished (selects the internal-underflow text).
enum class AsyncReader : uint8_t { acc_st, stg_v8 };

#if NANOSM_SANITIZER

class Sanitizer {
public:
    Sanitizer() = default;

    // ---- lifetime ----
    void record_launch(const Config& config, const Kernel& kernel, uint32_t warps);
    void record_cycle_start();

    // ---- RF / predicate hazards (there is no scoreboard; a violation PANICS) ----
    void record_reg_ready(uint32_t warp, uint32_t reg, uint64_t cycle);
    void record_reg_read_until(uint32_t warp, uint32_t reg, uint64_t exclusive_end);
    void record_async_reader_add(uint32_t warp, uint32_t reg);
    void record_async_reader_done(uint32_t warp, uint32_t reg, AsyncReader reader);
    void record_pred_ready(uint32_t warp, uint32_t pred, uint64_t cycle);
    void check_pred_waw(const IssueSite& site, uint8_t pred) const;
    void check_pred_raw(const IssueSite& site, uint8_t pred) const;
    void check_rf_raw(const IssueSite& site, const std::vector<uint8_t>& sources) const;
    void check_rf_waw_war(const IssueSite& site, uint8_t dst, unsigned width) const;

    // ---- shmem ownership (trash/shmem-ownership-sanitizer-2026-10-03.md) ----
    // word_addresses: every lane access of every round, duplicates included.
    void check_simt_shmem_access(
        const IssueSite& site, const std::vector<uint32_t>& word_addresses, bool load) const;
    void check_simt_mbar_rmw(const IssueSite& site, uint32_t address) const;
    void check_engine_mbar_rmw(uint64_t cycle, uint32_t address, bool complete_tx) const;
    // Panics on a conflict, else records the descriptor (and box) owners.
    // descriptor_bytes: what the engine's later port-A read will see.
    void record_tma_accept(
        const IssueSite& site, uint64_t command, uint32_t descriptor_address,
        const std::array<uint8_t, tma_descriptor_bytes>& descriptor_bytes);
    void record_tma_descriptor_read(uint64_t command, const TmaDescriptor& decoded);
    void record_tma_data_complete(uint64_t command);
    // Panics on a conflict, else records the A/B operand and accumulator owners.
    void record_mma_accept(
        const IssueSite& site, uint64_t command, uint8_t accumulator, uint8_t depbar,
        std::optional<std::pair<uint32_t, uint32_t>> a_span,
        std::optional<std::pair<uint32_t, uint32_t>> b_span);
    void record_mma_fetch_start(uint64_t command);
    void record_mma_operands_consumed(uint64_t command);
    void record_mma_retired(uint64_t command);

    // ---- tmem ownership ----
    void check_acc_transfer_vs_mma(const IssueSite& site, uint8_t accumulator) const;
    void check_acc_transfer_overlap(
        const IssueSite& site, uint8_t accumulator, bool load,
        uint32_t row, uint32_t col, uint32_t width) const;
    void record_acc_transfer_accept(
        const IssueSite& site, uint8_t bank, bool load, uint8_t accumulator,
        uint32_t row, uint32_t col, uint32_t width, uint8_t depbar);
    // The transfer in slot [bank][load ? 0 : 1] made its final beat. A store's
    // ownership ends now; a load's when its final RF write commits.
    void record_acc_transfer_final_beat(uint8_t bank, bool load, uint64_t commit_cycle);
    void record_writebacks_committed(uint64_t cycle);

    // ---- C5: port-A write vs same-cycle MMA port-B read of one word ----
    void record_port_a_write(uint64_t byte_address, uint32_t size);
    void check_port_b_round(const std::vector<uint32_t>& words) const;

    // ---- fall-through tripwires: a register stage is read only after the
    // cycle it was written ----
    void record_tma_sector_write(uint64_t cycle);
    void check_tma_sector_read(uint64_t cycle) const;
    void record_tma_store_sector_write(uint64_t cycle);
    void check_tma_store_sector_read(uint64_t cycle) const;
    void record_assembly_write(uint32_t index, uint64_t cycle);
    void check_assembly_dump(uint32_t index, uint64_t cycle) const;
    void check_store_assembly_read(uint32_t index, uint64_t cycle) const;

    // ---- boundary tripwire: an ordered read ID returns in request order ----
    void record_ordered_request(uint32_t stream, uint64_t request);
    void check_ordered_beat(uint64_t cycle, uint32_t stream, uint64_t request, uint32_t byte_offset) const;
    void record_ordered_pluck(uint32_t stream, uint32_t bytes);

    // ---- rfc09 3: admitted LSU commands never exceed the assembly pool ----
    void check_lsu_admitted(std::size_t admitted) const;

private:
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
        bool fetching = false;                   // mma: its first operand fetch has started
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

    struct WarpHazards {
        std::vector<uint64_t> ready_cycle;   // RAW/WAW: the scheduled write's commit cycle
        std::vector<uint64_t> read_until;    // WAR: last scheduled read, exclusive end
        std::vector<uint32_t> async_readers; // WAR: acc.st / stg.v8 sources not yet read
        std::array<uint64_t, 7> predicate_ready_cycle{};
    };

    struct OrderedRequest {
        uint64_t request = 0;
        uint32_t received = 0; // bytes plucked
    };

    struct PendingAccLoadRelease {
        uint64_t id = 0;
        uint64_t commit_cycle = 0;
    };

    [[nodiscard]] const ShmemOwner* shmem_owner_overlapping(
        uint64_t begin, uint64_t end, bool writers_only) const;
    [[nodiscard]] std::string describe_shmem_owner(const ShmemOwner& owner) const;
    [[nodiscard]] std::string shmem_race(
        const std::string& access, uint64_t begin, uint64_t end, const ShmemOwner& owner) const;
    [[nodiscard]] static bool tmem_overlap(const TmemOwner& a, const TmemOwner& b);
    [[nodiscard]] std::string describe_tmem_owner(const TmemOwner& owner) const;
    void release_tmem_owner(TmemOwnerKind kind, uint64_t id);
    [[nodiscard]] std::string stream_name(uint32_t stream) const;

    // Copied from the launch (no reference into Machine, which may move).
    uint32_t lanes_ = 0;
    uint32_t lsu_read_streams_ = 0;
    uint32_t gmem_sector_bytes_ = 0;
    uint32_t lsu_staging_vectors_ = 0;
    uint32_t lsu_active_commands_ = 0;
    std::vector<std::string> accumulator_names_;
    std::vector<std::pair<uint32_t, uint32_t>> accumulator_shapes_; // rows, cols

    // n is tiny (a few outstanding commands); checks are linear scans.
    std::vector<ShmemOwner> shmem_owners_;
    std::vector<TmemOwner> tmem_owners_;
    uint64_t next_acc_transfer_id_ = 0;
    std::vector<std::array<std::optional<uint64_t>, 2>> acc_transfer_in_slot_; // [bank][0 load, 1 store]
    std::vector<PendingAccLoadRelease> pending_acc_ld_releases_;
    std::vector<WarpHazards> warps_;
    std::vector<uint32_t> port_a_written_words_; // this cycle
    uint64_t tma_sector_last_write_ = 0;
    uint64_t tma_store_sector_last_write_ = 0;
    std::vector<uint64_t> assembly_last_write_;
    std::vector<std::deque<OrderedRequest>> ordered_requests_; // per ordered read ID
};

#else // NANOSM_SANITIZER == 0: every entry point is a no-op.

class Sanitizer {
public:
#define NANOSM_SANITIZER_NOOP(name) \
    template <typename... Args> void name(Args&&...) const {}
    NANOSM_SANITIZER_NOOP(record_launch)
    NANOSM_SANITIZER_NOOP(record_cycle_start)
    NANOSM_SANITIZER_NOOP(record_reg_ready)
    NANOSM_SANITIZER_NOOP(record_reg_read_until)
    NANOSM_SANITIZER_NOOP(record_async_reader_add)
    NANOSM_SANITIZER_NOOP(record_async_reader_done)
    NANOSM_SANITIZER_NOOP(record_pred_ready)
    NANOSM_SANITIZER_NOOP(check_pred_waw)
    NANOSM_SANITIZER_NOOP(check_pred_raw)
    NANOSM_SANITIZER_NOOP(check_rf_raw)
    NANOSM_SANITIZER_NOOP(check_rf_waw_war)
    NANOSM_SANITIZER_NOOP(check_simt_shmem_access)
    NANOSM_SANITIZER_NOOP(check_simt_mbar_rmw)
    NANOSM_SANITIZER_NOOP(check_engine_mbar_rmw)
    NANOSM_SANITIZER_NOOP(record_tma_accept)
    NANOSM_SANITIZER_NOOP(record_tma_descriptor_read)
    NANOSM_SANITIZER_NOOP(record_tma_data_complete)
    NANOSM_SANITIZER_NOOP(record_mma_accept)
    NANOSM_SANITIZER_NOOP(record_mma_fetch_start)
    NANOSM_SANITIZER_NOOP(record_mma_operands_consumed)
    NANOSM_SANITIZER_NOOP(record_mma_retired)
    NANOSM_SANITIZER_NOOP(check_acc_transfer_vs_mma)
    NANOSM_SANITIZER_NOOP(check_acc_transfer_overlap)
    NANOSM_SANITIZER_NOOP(record_acc_transfer_accept)
    NANOSM_SANITIZER_NOOP(record_acc_transfer_final_beat)
    NANOSM_SANITIZER_NOOP(record_writebacks_committed)
    NANOSM_SANITIZER_NOOP(record_port_a_write)
    NANOSM_SANITIZER_NOOP(check_port_b_round)
    NANOSM_SANITIZER_NOOP(record_tma_sector_write)
    NANOSM_SANITIZER_NOOP(check_tma_sector_read)
    NANOSM_SANITIZER_NOOP(record_tma_store_sector_write)
    NANOSM_SANITIZER_NOOP(check_tma_store_sector_read)
    NANOSM_SANITIZER_NOOP(record_assembly_write)
    NANOSM_SANITIZER_NOOP(check_assembly_dump)
    NANOSM_SANITIZER_NOOP(check_store_assembly_read)
    NANOSM_SANITIZER_NOOP(record_ordered_request)
    NANOSM_SANITIZER_NOOP(check_ordered_beat)
    NANOSM_SANITIZER_NOOP(record_ordered_pluck)
    NANOSM_SANITIZER_NOOP(check_lsu_admitted)
#undef NANOSM_SANITIZER_NOOP
};

#endif

} // namespace nanosm

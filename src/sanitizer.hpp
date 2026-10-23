#pragma once

// =============================================================================
// THE SANITIZER RULE
//
// The Sanitizer holds every piece of shadow state the simulator keeps only to
// decide whether to PANIC (ownership tables, hazard horizons, fall-through
// and port-collision tripwires, the ordered-ID tripwire, the shadow of SIMT
// shmem accesses still travelling to the RAM, the live-mbar list). None of
// it is hardware. The machine's behavior -- every cycle, counter, result byte and
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
#include <map>
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
// mma_shapes, the profile's shape list, names an mma.async by its shape.
struct IssueSite {
    uint64_t cycle = 0;
    uint32_t warp = 0;
    uint32_t pc = 0;
    const Instruction& inst;
    const std::vector<MmaShape>* mma_shapes = nullptr;
};

// PANIC text shared by Machine and Sanitizer, so a moved check keeps its
// message byte for byte. Always compiled.
[[noreturn]] void throw_issue_panic(const IssueSite& site, const std::string& reason);
// requester == nullptr: an engine's request ("source=engine").
[[noreturn]] void throw_mbar_rmw_panic(
    uint64_t cycle, uint32_t address, const std::string& reason, const IssueSite* requester);

// Which async reader of a register finished (selects the internal-underflow text).
enum class AsyncReader : uint8_t { acc_st, stg_v8 };

// Where a SIMT shmem access waits for its grants (rfc12 sections 3, 6, 9):
// a load slot, a store slot (by slot index), the probe register or the SIMT
// RMW register / engine (by warp: a warp has at most one probe and one SIMT
// RMW outstanding).
enum class SimtShmemHolder : uint8_t { load_slot, store_slot, probe, rmw };

#if NANOSM_SANITIZER

class Sanitizer {
public:
    Sanitizer() = default;

    // ---- lifetime ----
    void record_launch(const Config& config, const Kernel& kernel, uint32_t warps);
    void record_cycle_start();

    // ---- RF / predicate hazards (there is no scoreboard; a violation PANICS) ----
    // Cycles follow src/rf.hpp. An RF register's READY cycle is its latest
    // write's VISIBILITY, the first cycle an RF sample sees it (a pending
    // asynchronous write: never, until it lands). READ_UNTIL is the exclusive
    // end of its pending fixed samples. A predicate's ready cycle is the first
    // cycle it is in the file: a conditional branch's sample (issue +
    // issue_to_predicate_sample, src/rf.hpp) must be at or after it, and a
    // predicate writer may not issue before it.
    void record_reg_ready(uint32_t warp, uint32_t reg, uint64_t visible_cycle);
    void record_reg_read_until(uint32_t warp, uint32_t reg, uint64_t exclusive_end);
    void record_async_reader_add(uint32_t warp, uint32_t reg);
    void record_async_reader_done(uint32_t warp, uint32_t reg, AsyncReader reader);
    void record_pred_ready(uint32_t warp, uint32_t pred, uint64_t cycle);
    void check_pred_waw(const IssueSite& site, uint8_t pred) const;
    void check_pred_raw(const IssueSite& site, uint8_t pred, uint64_t sample_cycle) const;
    // RAW: each source register with the cycle this instruction samples it
    // must be visible by then.
    void check_rf_raw(const IssueSite& site, const std::vector<std::pair<uint8_t, uint64_t>>& samples) const;
    // WAW / WAR for a write that cannot become visible before
    // earliest_visible: it must land (earliest_visible - 1) after the older
    // write lands and at or after the older samples, and no asynchronous
    // reader may still own the register.
    void check_rf_waw_war(const IssueSite& site, uint8_t dst, unsigned width, uint64_t earliest_visible) const;

    // ---- admission reads no operand (rfc10 section 1) ----
    // Every issue attempt opens with record_issue_attempt; record_issue_accepted
    // marks the moment admission has decided. A capture in an attempt not
    // (yet) accepted is an internal PANIC (a refused attempt must consume
    // nothing). No machine path reads an operand at issue today: RF operands
    // (I6a) and, since the front end (I6b-2), a conditional branch's
    // predicate are sampled through the issue register, a cycle after
    // acceptance. The tripwire stays for any future issue-time read.
    void record_issue_attempt(uint32_t warp);
    void record_issue_accepted();
    void check_issue_capture(uint64_t cycle, uint32_t warp, uint8_t reg, bool predicate) const;

    // ---- declared uniformity (rfc09 pin 3) ----
    // A declared-uniform operand (lds.uni / ldg.uni / ldg.v8 / stg.v8 bases,
    // engine and mbar operands, accumulator coordinates, the bra.uni
    // predicate) must hold one value in every lane. Hardware reads lane 0
    // only; anything else is UB, diagnosed here. lane_values: the operand, by lane.
    void check_warp_uniform(
        const IssueSite& site, const std::vector<uint32_t>& lane_values, const std::string& what) const;
    // A uniform operation -- every instruction with a declared-uniform
    // operand (the list above; the machine calls this from uniform_reg) and
    // bra.uni -- under a partial or empty active mask. Hardware reads lane 0
    // whatever the mask, so this is UB, diagnosed here (HUMAN 2026-10-12,
    // backlog D12 / D12.1, rfc13 section 2); what the machine does then is
    // no contract.
    void check_uniform_mask(const IssueSite& site, uint64_t active_mask, uint64_t full_mask) const;

    // ---- same-word stores (HUMAN 2026-10-12, backlog D12) ----
    // Two active lanes of one sts / stg naming one 4-byte word: UB, as on
    // NVIDIA (one write lands; which is undefined). The machine lets one
    // land quietly (the highest active lane); no program may rely on which.
    // word_addresses: the active lanes' word byte addresses, in lane order;
    // space: "shmem" or "gmem" (the PANIC text).
    void check_store_word_collision(
        const IssueSite& site, const std::vector<uint64_t>& word_addresses, const std::string& space) const;

    // ---- shmem ownership (trash/shmem-ownership-sanitizer-2026-10-03.md) ----
    // word_addresses: every lane access of every round, duplicates included.
    void check_simt_shmem_access(
        const IssueSite& site, const std::vector<uint32_t>& word_addresses, bool load) const;
    void check_simt_mbar_rmw(const IssueSite& site, uint32_t address) const;
    void check_engine_mbar_rmw(uint64_t cycle, uint32_t address, bool complete_tx) const;

    // ---- missing fence.sh / race (trash/rfc12.md sections 10 and 11) ----
    // A shadow of every SIMT shmem access until it REACHES THE RAM (a write's
    // edge, a read's sample: its round's grant + D). An access is recorded
    // when issue accepts it -- an lds / lds.uni / sts with its bank rounds'
    // word addresses (access_bytes 4), a probe or a SIMT mbar RMW as one
    // 8-byte round, a uniform atomic as one 4-byte round (a write: it reads
    // and writes the word) -- and each round's grant then records when it
    // reaches the RAM (ram_cycle). A SIMT RMW (an mbar operation or an atomic)
    // counts at its write grant. The
    // engine-command accepts (record_tma_accept, record_mma_accept) intersect
    // their footprint with the rounds that have not reached the RAM by the
    // accept cycle (ungranted, or ram_cycle later than it): an overlap where
    // at least one side writes PANICs -- with the issuing warp's own access
    // "missing fence.sh", with another warp's "shmem race". Read/read overlap
    // never PANICs. This diagnoses the PHYSICAL hazard present at accept; it
    // does not track whether a fence.sh was issued (backlog D10), so an
    // unfenced command whose warp's accesses already reached the RAM passes.
    void record_simt_shmem_issue(const IssueSite& site, SimtShmemHolder holder, uint32_t index,
        const std::vector<std::vector<uint32_t>>& rounds, uint32_t access_bytes, bool write);
    // The next round of the access in (holder, index) is granted now (cycle);
    // it reaches the RAM at ram_cycle. An sts round is also checked against
    // the live mbars (C10, below).
    void record_simt_shmem_grant(uint64_t cycle, SimtShmemHolder holder, uint32_t index, uint64_t ram_cycle);

    // ---- C10 live-mbar opacity (trash/rfc12.md section 11) ----
    // An accepted SIMT mbar.init makes [address, address + 8) a LIVE mbar
    // until kernel end (v0 retirement); so does the launch's declaration of an
    // mbar the host initialized in the shmem image (Launch::mbars, live from
    // launch; I5). From then on only mbar operations (init, arrive,
    // expect_tx, the probes, the engines' complete_tx and commit arrive) may
    // touch it: an sts round (checked at its grant), a TMA placement round
    // (port-B write, check_live_mbar_write) or a uniform atomic's write
    // (check_live_mbar_write) writing any of its bytes PANIC naming the
    // mbar. Implementer choice: an mbar.init of an already-live mbar PANICs
    // too (no mbar.inval in v0: an mbar is initialized once per kernel, and a
    // declared one already was, by the host).
    void record_launch_mbars(const std::vector<uint32_t>& addresses);
    void record_mbar_init(const IssueSite& site, uint32_t address);
    // A non-mbar write granted now (a TMA placement round, a uniform
    // atomic's write): `writer` names it, byte_ranges are its [begin, end).
    void check_live_mbar_write(uint64_t cycle, const std::string& writer,
        const std::vector<std::pair<uint32_t, uint32_t>>& byte_ranges) const;

    // Panics on a conflict, else records the descriptor (and box) owners.
    // First the missing-fence / race check: the descriptor's 64 bytes, then
    // the box (a load's destination is written, a store's source read).
    // descriptor_bytes: what the engine's later A-read will see -- memory as
    // of this cycle's clock edge (anything still in flight PANICked above).
    // completion_mbar_word: the descriptor's completion mbar as it stands at
    // accept (none: invalid descriptor). Its phase is recorded for the
    // command's complete_tx (rfc12 section 8: the hardware reads no mbar at
    // TMA start; this peek is sanitizer-only metadata).
    void record_tma_accept(
        const IssueSite& site, uint64_t command, uint32_t descriptor_address,
        const std::array<uint8_t, tma_descriptor_bytes>& descriptor_bytes,
        std::optional<uint64_t> completion_mbar_word);
    // A complete_tx reaches its mbar, whose current phase is `phase`: it must
    // be the phase its sender meant -- a TMA command's accept-time peek
    // (tma_command), else the host's stated parity. A mismatch PANICs ("wrong
    // phase"): the transaction bytes landed in a later phase (e.g. a plain
    // final arrive completed the phase before the TMA's bytes came back).
    void check_complete_tx_phase(uint64_t cycle, uint32_t address, std::optional<uint64_t> tma_command,
        uint8_t stated_parity, bool phase);
    // Every mbar RMW's resulting tx count; at kernel end each touched mbar's
    // count must be zero (a missing expect_tx or complete_tx hangs a waiter).
    void record_mbar_tx(uint32_t address, int32_t tx);
    void check_kernel_end(uint64_t cycle) const;
    void record_tma_descriptor_read(uint64_t command, const TmaDescriptor& decoded);
    void record_tma_data_complete(uint64_t command);
    // Panics on a conflict (first the missing-fence / race check of both
    // operand spans, port-B reads), else records the A/B operand and
    // accumulator owners.
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
        const IssueSite& site, bool load, uint8_t accumulator,
        uint32_t row, uint32_t col, uint32_t width, uint8_t depbar);
    // The transfer in the acc.ld (load) or acc.st engine made its final tmem access:
    // its tmem ownership ends now -- a store's at its final tmem write, a
    // load's at its final tmem READ (hardware truth: the values are in its
    // pipeline). A load's destination registers stay not-ready until each
    // beat's RF write (record_reg_ready), a separate lifetime.
    void record_acc_transfer_final_beat(bool load);

    // ---- clocked registers (src/reg.hpp): at most one write per tick ----
    // The B2.1 fall-through tripwires (TMA sector, TMA store sector, assembly
    // registers) are retired: those registers are Reg<T>, read through cur().
    void record_register_write(uint32_t id, const char* tag);
    void check_register_rewrite(
        uint32_t id, const std::string& name, uint64_t cycle, const char* tag) const;

    // ---- boundary tripwire: an ordered read ID returns in request order ----
    void record_ordered_request(uint32_t stream, uint64_t request);
    void check_ordered_beat(uint64_t cycle, uint32_t stream, uint64_t request, uint32_t byte_offset) const;
    void record_ordered_pluck(uint32_t stream, uint32_t bytes);

    // ---- rfc09 3: admitted LSU commands never exceed the assembly pool ----
    void check_lsu_admitted(std::size_t admitted) const;
    // A reader of an assembly register (the RF drain, the egress write port)
    // takes only lanes marked valid: lanes [first_lane, first_lane + lanes)
    // of `valid`, the register's Q. A lost valid bit panics here.
    void check_assembly_lanes_valid(uint64_t cycle, uint32_t assembly, const std::vector<uint8_t>& valid,
        uint32_t first_lane, uint32_t lanes, const char* reader) const;

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
        acc_ld, // read-owner of its rows x columns until its final tmem read
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
        std::vector<uint64_t> ready_cycle;   // RAW/WAW: the latest write's visibility (src/rf.hpp)
        std::vector<uint64_t> read_until;    // WAR: last scheduled sample, exclusive end
        std::vector<uint32_t> async_readers; // WAR: acc.st / stg.v8 sources not yet read
        std::array<uint64_t, 7> predicate_ready_cycle{};
    };

    struct OrderedRequest {
        uint64_t request = 0;
        uint32_t received = 0; // bytes plucked
    };

    // One SIMT shmem access until all its rounds have reached the RAM.
    struct SimtShmemRound {
        std::vector<uint32_t> addresses;   // byte address of each access in the round
        std::optional<uint64_t> ram_cycle; // granted: the cycle it reaches the RAM (grant + D)
    };
    struct SimtShmemAccess {
        SimtShmemHolder holder = SimtShmemHolder::load_slot;
        uint32_t index = 0;                // slot index, or the warp (probe, RMW)
        uint32_t warp = 0;
        uint32_t pc = 0;
        uint64_t issue_cycle = 0;
        Instruction inst;
        bool write = false;
        uint32_t access_bytes = 4;         // 4 (lds / sts word) or 8 (mbar)
        std::vector<SimtShmemRound> rounds;
        std::size_t granted = 0;
    };

    struct LiveMbar {
        uint32_t address = 0;
        bool declared_by_launch = false; // live from launch (else: a SIMT mbar.init, below)
        uint32_t warp = 0;
        uint32_t pc = 0;
        uint64_t init_cycle = 0;
    };
    struct LiveMbarHit {
        const LiveMbar* mbar = nullptr;
        uint64_t first_byte = 0; // the mbar bytes the write round writes: [first_byte, last_byte)
        uint64_t last_byte = 0;
    };

    // The missing-fence / race check of one footprint range of an engine
    // command accepted at site (label: e.g. "tma load #3 destination").
    void check_engine_footprint(
        const IssueSite& site, const std::string& label, uint64_t begin, uint64_t end, bool write) const;
    [[nodiscard]] const LiveMbar* live_mbar_overlapping(uint64_t begin, uint64_t end) const;
    [[nodiscard]] std::optional<LiveMbarHit> live_mbar_hit(
        const std::vector<std::pair<uint32_t, uint32_t>>& byte_ranges) const;
    [[nodiscard]] static std::string describe_live_mbar(const LiveMbar& mbar);

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
    // SIMT shmem accesses that may not have reached the RAM yet (pruned at
    // each issue: bounded by the slots, the probe / RMW registers and D).
    std::vector<SimtShmemAccess> simt_shmem_;
    // C10: mbars the launch declared or a SIMT mbar.init made live, until kernel end.
    std::vector<LiveMbar> live_mbars_;
    // TMA command -> its completion mbar's phase at accept, until its complete_tx.
    std::vector<std::pair<uint64_t, bool>> tma_completion_phases_;
    // mbar address -> its tx count as the last RMW left it.
    std::map<uint32_t, int32_t> mbar_tx_;
    std::vector<TmemOwner> tmem_owners_;
    uint64_t next_acc_transfer_id_ = 0;
    // The tmem owner ids of the transfers each engine has accepted and not yet
    // finished, oldest first ([0] acc.ld, [1] acc.st). An acc.ld engine takes
    // its next transfer at its previous one's last launch, before that one's
    // last tmem sample, so its finishes come in acceptance order.
    std::array<std::deque<uint64_t>, 2> acc_transfer_in_engine_;
    std::vector<WarpHazards> warps_;
    // The issue attempt in progress (record_issue_attempt) and whether
    // admission has accepted it.
    std::optional<uint32_t> attempt_warp_;
    bool attempt_accepted_ = false;
    // Registers written this tick and their first writer's tag.
    std::vector<std::pair<uint32_t, const char*>> register_writes_;
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
    NANOSM_SANITIZER_NOOP(record_issue_attempt)
    NANOSM_SANITIZER_NOOP(record_issue_accepted)
    NANOSM_SANITIZER_NOOP(check_issue_capture)
    NANOSM_SANITIZER_NOOP(check_warp_uniform)
    NANOSM_SANITIZER_NOOP(check_uniform_mask)
    NANOSM_SANITIZER_NOOP(check_store_word_collision)
    NANOSM_SANITIZER_NOOP(check_simt_shmem_access)
    NANOSM_SANITIZER_NOOP(check_simt_mbar_rmw)
    NANOSM_SANITIZER_NOOP(check_engine_mbar_rmw)
    NANOSM_SANITIZER_NOOP(record_simt_shmem_issue)
    NANOSM_SANITIZER_NOOP(record_simt_shmem_grant)
    NANOSM_SANITIZER_NOOP(record_launch_mbars)
    NANOSM_SANITIZER_NOOP(record_mbar_init)
    NANOSM_SANITIZER_NOOP(check_live_mbar_write)
    NANOSM_SANITIZER_NOOP(record_tma_accept)
    NANOSM_SANITIZER_NOOP(check_complete_tx_phase)
    NANOSM_SANITIZER_NOOP(record_mbar_tx)
    NANOSM_SANITIZER_NOOP(check_kernel_end)
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
    NANOSM_SANITIZER_NOOP(record_register_write)
    NANOSM_SANITIZER_NOOP(check_register_rewrite)
    NANOSM_SANITIZER_NOOP(record_ordered_request)
    NANOSM_SANITIZER_NOOP(check_ordered_beat)
    NANOSM_SANITIZER_NOOP(record_ordered_pluck)
    NANOSM_SANITIZER_NOOP(check_lsu_admitted)
    NANOSM_SANITIZER_NOOP(check_assembly_lanes_valid)
#undef NANOSM_SANITIZER_NOOP
};

#endif

} // namespace nanosm

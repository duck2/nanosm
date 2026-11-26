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
// =============================================================================

#include "qtxc/qtxc.hpp"
#include "isa.hpp"
#include "profile.hpp"
#include "tma_descriptor.hpp"

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
    DecodedInstruction inst;
    const std::vector<MmaShape>* mma_shapes = nullptr;
};

// PANIC text shared by Machine and Sanitizer, so a moved check keeps its
// message byte for byte. Always compiled.
[[noreturn]] void throw_issue_panic(const IssueSite& site, const std::string& reason);
// requester == nullptr: an engine's request ("source=engine").
[[noreturn]] void throw_mbar_rmw_panic(
    uint64_t cycle, uint32_t address, const std::string& reason, const IssueSite* requester);

// Which async reader of a register finished (selects the internal-underflow text).
enum class AsyncReader : uint8_t { acc_st, stg_v8, shmem_collect };

// Where a SIMT shmem access is granted: shared memory's read side (lds,
// lds.uni, probes) or write side (sts, barrier changes, atomics: an RMW at
// its final write). Each side serves its accesses one at a time, in the
// order they were recorded.
enum class SimtShmemHolder : uint8_t { read_side, write_side };

// A tmem region an instruction touches (src/layout.hpp): `rows` x `columns`
// of the region at `pointer` (a logical word) whose rows are row_words words,
// and the physical words they are, sorted. Ownership and races are decided on
// `words`; the rest is PANIC text. A matmul's is its m x n result (rows of
// n words), a transfer's its sixteen rows x one or eight columns.
struct TmemRegion {
    uint32_t pointer = 0;
    uint32_t row_words = 0;
    uint32_t rows = 0;
    uint32_t columns = 0;
    std::vector<uint32_t> words;
};

#if NANOSM_SANITIZER

class Sanitizer {
public:
    Sanitizer() = default;

    // ---- lifetime ----
    void record_launch(const Config& config, const Kernel& kernel, uint32_t warps);
    void record_cycle_start();

    // ---- RF / predicate hazards (there is no scoreboard; a violation PANICS) ----
    // Cycles follow src/timing.hpp. An RF register's READY cycle is its latest
    // write's VISIBILITY, the first cycle an RF sample sees it (a pending
    // asynchronous write: never, until it lands). READ_UNTIL is the exclusive
    // end of its pending fixed samples. A predicate's ready cycle is the first
    // cycle it is in the file: a conditional branch's sample (issue +
    // issue_to_predicate_sample, src/timing.hpp) must be at or after it, and a
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

    // ---- uniform-register hazards (src/timing.hpp) ----
    // A uniform register's READY cycle is its latest write's visibility (every
    // writer is fixed-latency: issue + L + Wu + 1). RAW: each uniform source,
    // sampled at sample_cycle, must be visible by then (a tma's registers:
    // each at its own round's sample, src/timing.hpp tma_operand_round). WAW: the
    // younger write (visible from visible_cycle) must land after the older
    // one. No WAR: a uniform register is read at issue + 1 (a tma's last round
    // at issue + 3 at most), written L + Wu >= 4 after its writer's issue.
    // Values from uniform registers are uniform by construction: no lane
    // check applies to them.
    void record_uniform_ready(uint32_t warp, uint32_t reg, uint64_t visible_cycle);
    void check_uniform_raw(const IssueSite& site, const std::vector<uint8_t>& sources, uint64_t sample_cycle) const;
    void check_uniform_waw(const IssueSite& site, uint8_t dst, uint64_t visible_cycle) const;

    // ---- admission reads no operand ----
    // Every issue attempt opens with record_issue_attempt; record_issue_accepted
    // marks the moment admission has decided. A capture in an attempt not
    // (yet) accepted is an internal PANIC (a refused attempt must consume
    // nothing). No machine path reads an operand at issue: RF operands and a
    // conditional branch's
    // predicate are sampled through the issue register, a cycle after
    // acceptance. The tripwire stays for any future issue-time read.
    void record_issue_attempt(uint32_t warp);
    void record_issue_accepted();
    void check_issue_capture(uint64_t cycle, uint32_t warp, uint8_t reg, bool predicate) const;

    // ---- declared uniformity ----
    // A declared-uniform operand taken from a per-lane file (the bra.uni
    // predicate; every other uniform operand is a uniform register) must
    // hold one value in every lane. Hardware reads lane 0 only; anything else
    // is UB, diagnosed here. lane_values: the operand, by lane.
    void check_warp_uniform(
        const IssueSite& site, const std::vector<uint32_t>& lane_values, const std::string& what) const;
    // A uniform operation -- every instruction with a declared-uniform
    // operand (the list above and every uniform-register operand of a ".uni"
    // consumer; the machine calls this from uniform_operand)
    // and bra.uni -- under a partial active mask (inside a divergent arm; a
    // mask is never empty). Hardware reads lane 0
    // whatever the mask, so this is UB, diagnosed here: a program with a
    // partial mask uses the per-lane instruction. What the machine does then
    // is no contract.
    void check_uniform_mask(const IssueSite& site, uint64_t active_mask, uint64_t full_mask) const;
    // ---- divergence ----
    // A halt must find its warp converged: an empty divergence stack and
    // every lane active. Halting inside an arm drops the other lanes' work;
    // the machine just halts the warp.
    void check_halt_converged(const IssueSite& site, uint32_t stack_depth, uint64_t active_mask,
        uint64_t full_mask) const;
    // A tma's buffer address must be 64-byte aligned and its barrier address
    // 8-byte aligned (src/tma_descriptor.hpp). The machine does not wire the low bits
    // (it ignores them); a program that sets them is wrong, diagnosed here.
    void check_tma_operand_alignment(const IssueSite& site, uint32_t buffer, uint32_t barrier) const;
    // A swizzle64 tensor-core operand (src/layout.hpp): `pointer` is a
    // 16-byte piece of its row 0 and the shape reads `row_bytes` from it in
    // every row. Those bytes must lie in one row whose line (the pointer
    // rounded down to the spacing) is a multiple of the spacing and of 64.
    // The machine reads whatever the address-keyed rule names and checks
    // none of this; a pointer that breaks it is wrong, diagnosed here. (A
    // swizzle64 TMA tile's buffer is checked in record_tma_accept, where the
    // descriptor is known.)
    void check_swizzle64_operand(const IssueSite& site, const std::string& operand, uint32_t pointer,
        uint32_t spacing, uint32_t row_bytes) const;
    // A tmem pointer (src/layout.hpp tmem_pointer_aligned): its row (pointer
    // / row_words) is a multiple of 4 and the `columns` it touches from it
    // stay inside its row. The machine addresses any pointer by the layout
    // rule and checks only that the words are in tmem; a pointer that breaks
    // this is wrong, diagnosed here. `what` names the operand.
    void check_tmem_pointer(const IssueSite& site, const std::string& what, uint32_t pointer,
        uint32_t row_words, uint32_t columns) const;

    // ---- same-word stores ----
    // Two active lanes of one sts / stg naming one 4-byte word: UB, as on
    // NVIDIA (one write lands; which is undefined). The machine lets one
    // land quietly (the highest active lane); no program may rely on which.
    // word_addresses: the active lanes' word byte addresses, in lane order;
    // space: "shmem" or "gmem" (the PANIC text).
    void check_store_word_collision(
        const IssueSite& site, const std::vector<uint64_t>& word_addresses, const std::string& space) const;

    // ---- shmem ownership ----
    // word_addresses: every lane access of every round, duplicates included.
    void check_simt_shmem_access(
        const IssueSite& site, const std::vector<uint32_t>& word_addresses, bool load) const;
    void check_simt_mbar_rmw(const IssueSite& site, uint32_t address) const;
    // A probe (mbar.test_wait / try_wait) reads the mbar's two words: it
    // conflicts with a write-owner (a barrier of the launch block that has
    // not landed yet).
    void check_simt_mbar_probe(const IssueSite& site, uint32_t address) const;
    void check_engine_mbar_rmw(uint64_t cycle, uint32_t address, bool complete_tx) const;

    // ---- missing fence.sh / race ----
    // A shadow of every SIMT shmem access until it REACHES THE RAM (a write's
    // edge, a read's sample: its round's grant + D). An access is recorded
    // at its issue + 1, while it waits in its queue (its words are the ones
    // its registers name then) -- an lds / lds.uni / sts with its bank rounds'
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
    // does not track whether a fence.sh was issued, so an
    // unfenced command whose warp's accesses already reached the RAM passes.
    void record_simt_shmem_issue(const IssueSite& site, SimtShmemHolder holder, uint32_t index,
        const std::vector<std::vector<uint32_t>>& rounds, uint32_t access_bytes, bool write);
    // The next round of the oldest access in (holder, index) with rounds left
    // is granted now (cycle);
    // it reaches the RAM at ram_cycle. An sts round is also checked against
    // the live mbars (below).
    void record_simt_shmem_grant(uint64_t cycle, SimtShmemHolder holder, uint32_t index, uint64_t ram_cycle);

    // ---- live mbars are opaque ----
    // An accepted SIMT mbar.init makes [address, address + 8) a LIVE mbar
    // until kernel end (v0 retirement); so does the launch's declaration of an
    // mbar the host initialized in the shmem image (Launch::mbars, live from
    // launch). From then on only mbar operations (init, arrive,
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
    // load / shmem_address / completion_mbar: the instruction's direction, its
    // buffer and its barrier (low bits dropped). completion_mbar_word: that
    // mbar as it stands at accept (none: outside shmem). Its phase is
    // recorded for the command's complete_tx (the hardware
    // reads no mbar at TMA start; this peek is sanitizer-only metadata).
    void record_tma_accept(
        const IssueSite& site, uint64_t command, uint32_t descriptor_address,
        const std::array<uint8_t, tma_descriptor_bytes>& descriptor_bytes, bool load,
        uint32_t shmem_address, uint32_t completion_mbar, std::optional<uint64_t> completion_mbar_word);
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
    void record_tma_descriptor_read(
        uint64_t command, const TmaDescriptor& decoded, uint32_t shmem_address, bool load);
    void record_tma_data_complete(uint64_t command);
    // ---- the launch copy (docs/qtx.md "Launch") ----
    // The launch block's bytes [begin, end) in shared memory are on their way
    // from global memory: until they land they are write-owned by the launch
    // copy, so an lds, an sts, an mbar operation or a TMA or tensor-core
    // command touching them PANICs like any shmem race (the fix: name the
    // section, so that qtxc waits on its launch flag). record_launch_landed:
    // everything below landed_end has landed.
    void record_launch_copy(uint32_t begin, uint32_t end);
    void record_launch_landed(uint32_t landed_end);
    // Panics on a conflict (first the missing-fence / race check of both
    // operand spans, port-B reads; then its result words against every
    // transfer that owns one), else records the A/B operand owners and the
    // result's tmem owner.
    void record_mma_accept(
        const IssueSite& site, uint64_t command, const TmemRegion& result, uint8_t depbar,
        std::optional<std::pair<uint32_t, uint32_t>> a_span,
        std::optional<std::pair<uint32_t, uint32_t>> b_span);
    // ---- row spacing: a reader states its writer's (src/layout.hpp) ----
    // A swizzle64 tile and a tmem region are addressed by a logical address
    // and a row spacing each instruction states for itself; a reader whose
    // spacing differs from its writer's quietly reads other words. The
    // sanitizer remembers, per word (shmem: byte address / 4; tmem: the
    // physical word), the spacing its last spacing-aware writer used -- a
    // TMA load into a swizzle64 tile (the stored row's bytes), a matmul's
    // result (n words), an acc.st (S words) -- and PANICs when a
    // spacing-aware reader -- a matmul's A / B operands and accumulator, an
    // acc.ld, a TMA store from a swizzle64 tile -- reads one of those words
    // with another spacing. A write that states no spacing (an sts, any
    // other SIMT shmem write, a dense TMA load) forgets its words, so a tile
    // written by hand is never accused. Writers record and readers check at
    // accept, in issue order: the order every correct kernel already keeps
    // between a writer and its reader (the race checks above see to it).
    // A TMA store is the exception: it checks the words it actually reads,
    // at each of its source rounds' grants (check_tma_store_source), so the
    // rows and stretches of its tile outside the global tensor, which it
    // never reads, are not checked. Its accept records its spacing; its box
    // is read-owned from accept on, so no writer can change those words'
    // shadow in between without a race PANIC.
    // The TMA, SIMT writes and acc transfers are handled in
    // record_tma_accept, record_simt_shmem_issue and
    // record_acc_transfer_accept; a matmul here. a_words / b_words: the shmem
    // words its operands read (src/machine/tensor_core.cpp mma_operand_address, / 4). Its
    // result's words are its accumulator (checked unless init) and then
    // written with n.
    void record_mma_row_spacing(const IssueSite& site, uint64_t command, bool init, const TmemRegion& result,
        const std::vector<uint32_t>& a_words, uint32_t a_spacing, const std::vector<uint32_t>& b_words,
        uint32_t b_spacing);
    // A swizzle64 TMA store's source round granted now (A-read): the shared
    // words it reads (byte addresses of whole words) against the spacing
    // their writer used. A store with no swizzle64 source (or none recorded)
    // is not checked. PANIC text: the store's instruction, this cycle.
    void check_tma_store_source(uint64_t cycle, uint64_t command, const std::vector<uint32_t>& word_addresses) const;
    void record_mma_fetch_start(uint64_t command);
    void record_mma_operands_consumed(uint64_t command);
    void record_mma_retired(uint64_t command);

    // ---- tmem ownership, by the physical words touched ----
    // A transfer whose words meet an accepted matmul's result (queued or
    // running, until its last result retires).
    void check_acc_transfer_vs_mma(const IssueSite& site, const TmemRegion& region) const;
    // A transfer whose words meet a transfer of the other engine still in
    // flight (a load over a store, or a store over a load or a store).
    void check_acc_transfer_overlap(const IssueSite& site, const TmemRegion& region, bool load) const;
    void record_acc_transfer_accept(const IssueSite& site, bool load, const TmemRegion& region, uint8_t depbar);
    // The transfer in the acc.ld (load) or acc.st engine made its final tmem access:
    // its tmem ownership ends now -- a store's at its final tmem write, a
    // load's at its final tmem READ (hardware truth: the values are in its
    // pipeline). A load's destination registers stay not-ready until each
    // beat's RF write (record_reg_ready), a separate lifetime.
    void record_acc_transfer_final_beat(bool load);

    // ---- clocked registers (src/machine/reg.hpp): at most one write per tick ----
    // (The TMA sector register, the TMA store register and the load / store
    // unit's buffers need no fall-through tripwire: they are Reg<T>, read
    // through cur().)
    void record_register_write(uint32_t id, const char* tag);
    void check_register_rewrite(
        uint32_t id, const std::string& name, uint64_t cycle, const char* tag) const;
    // ---- memories (src/machine/memory.hpp): a port takes one word per cycle ----
    // A memory used more ports in one cycle than it has (what is wrong, in
    // words): the same kind of PANIC as a second write of a register.
    void panic_memory_ports(const std::string& name, const std::string& what, uint64_t cycle, const char* tag) const;

    // ---- boundary tripwire: an ordered read ID returns in request order ----
    void record_ordered_request(uint32_t stream, uint64_t request);
    void check_ordered_beat(uint64_t cycle, uint32_t stream, uint64_t request, uint32_t byte_offset) const;
    void record_ordered_pluck(uint32_t stream, uint32_t bytes);

    // ---- admitted LSU commands never exceed the command slots ----
    void check_lsu_admitted(std::size_t admitted) const;
    // A reader of the load or store buffer (the RF drain, the egress write port)
    // takes only lanes marked valid: lanes [first_lane, first_lane + lanes)
    // of `valid`, the register's Q. A lost valid bit panics here.
    void check_buffer_lanes_valid(uint64_t cycle, uint32_t buffer, const std::vector<uint8_t>& valid,
        uint32_t first_lane, uint32_t lanes, const char* reader) const;

private:
    enum class ShmemOwnerKind : uint8_t {
        tma_load,       // write-owner of the destination box until its last shmem write commits
        tma_store,      // read-owner of the source box until its last source read
        tma_descriptor, // read-owner of the 64-byte descriptor until the engine reads it
        mma_a,          // read-owner of the A operand until the final operand read
        mma_b,          // read-owner of the B operand until the final operand read
        launch_copy,    // write-owner of the launch block's bytes that have not landed yet
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
        std::optional<uint32_t> completion_mbar; // tma: the instruction's completion mbar
        uint8_t depbar = 0;                      // mma operand-consumed depbar
        bool fetching = false;                   // mma: its first operand fetch has started
    };

    enum class TmemOwnerKind : uint8_t {
        mma,    // write-owner (read-modify-write) of its whole result until its last result retires
        acc_ld, // read-owner of its rows x columns until its final tmem read
        acc_st, // write-owner of its rows x columns until its final tmem write
    };

    struct TmemOwner {
        TmemOwnerKind kind = TmemOwnerKind::mma;
        uint64_t id = 0; // mma command id or acc transfer id
        TmemRegion region;
        uint32_t warp = 0;
        uint32_t pc = 0;
        uint64_t issue_cycle = 0;
        uint8_t depbar = 0;
    };

    struct WarpHazards {
        std::vector<uint64_t> ready_cycle;   // RAW/WAW: the latest write's visibility (src/timing.hpp)
        std::vector<uint64_t> read_until;    // WAR: last scheduled sample, exclusive end
        std::vector<uint32_t> async_readers; // WAR: acc.st / stg.v8 sources not yet read
        std::array<uint64_t, 7> predicate_ready_cycle{};
        std::vector<uint64_t> uniform_ready_cycle; // uniform RAW/WAW: the latest write's visibility
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
        SimtShmemHolder holder = SimtShmemHolder::read_side;
        uint32_t index = 0;                // 0 (one access at a time per side)
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
    [[nodiscard]] static bool tmem_overlap(const TmemRegion& a, const TmemRegion& b);
    [[nodiscard]] static std::string describe_tmem_region(const TmemRegion& region);
    [[nodiscard]] std::string describe_tmem_owner(const TmemOwner& owner) const;
    void release_tmem_owner(TmemOwnerKind kind, uint64_t id);

    // Row spacing shadow (record_mma_row_spacing): who last wrote a word with
    // a spacing, and the spacing. One byte a word: the writer in the top two
    // bits (none = 0) and log2(spacing) below (every spacing is a power of
    // two). Allocated at launch for all of shmem and tmem, never grown: a
    // quarter of the bytes of the memory it shadows.
    enum class SpacingWriter : uint8_t { none = 0, tma_load = 1, mma = 2, acc_st = 3 };
    void check_row_spacing(const IssueSite& site, const std::string& reader, bool tmem,
        const std::vector<uint32_t>& words, uint32_t spacing) const;
    void record_row_spacing(bool tmem, const std::vector<uint32_t>& words, uint32_t spacing, SpacingWriter writer);
    void forget_row_spacing(uint64_t begin_byte, uint64_t end_byte); // shmem bytes [begin, end)
    std::vector<uint8_t> shmem_row_spacing_;
    std::vector<uint8_t> tmem_row_spacing_;
    // An accepted swizzle64 TMA store, until its data completes: the spacing
    // its source rounds are checked with, and its instruction (PANIC text).
    struct TmaStoreSpacing {
        uint64_t command = 0;
        uint32_t spacing = 0;
        uint32_t warp = 0;
        uint32_t pc = 0;
        Instruction inst;
        const std::vector<MmaShape>* mma_shapes = nullptr;
    };
    std::vector<TmaStoreSpacing> tma_store_spacings_;
    [[nodiscard]] std::string stream_name(uint32_t stream) const;

    // Copied from the launch (no reference into Machine, which may move).
    uint32_t lanes_ = 0;
    uint32_t lsu_read_streams_ = 0;
    uint32_t gmem_sector_bytes_ = 0;
    uint32_t lsu_active_commands_ = 0;

    // n is tiny (a few outstanding commands); checks are linear scans.
    std::vector<ShmemOwner> shmem_owners_;
    // SIMT shmem accesses that may not have reached the RAM yet (pruned at
    // each issue: bounded by the slots, the probe / RMW registers and D).
    std::vector<SimtShmemAccess> simt_shmem_;
    // Live mbars: those the launch declared or a SIMT mbar.init made live, until kernel end.
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
    NANOSM_SANITIZER_NOOP(record_uniform_ready)
    NANOSM_SANITIZER_NOOP(check_uniform_raw)
    NANOSM_SANITIZER_NOOP(check_uniform_waw)
    NANOSM_SANITIZER_NOOP(record_issue_attempt)
    NANOSM_SANITIZER_NOOP(record_issue_accepted)
    NANOSM_SANITIZER_NOOP(check_issue_capture)
    NANOSM_SANITIZER_NOOP(check_warp_uniform)
    NANOSM_SANITIZER_NOOP(check_uniform_mask)
    NANOSM_SANITIZER_NOOP(check_halt_converged)
    NANOSM_SANITIZER_NOOP(check_tma_operand_alignment)
    NANOSM_SANITIZER_NOOP(check_swizzle64_operand)
    NANOSM_SANITIZER_NOOP(check_tmem_pointer)
    NANOSM_SANITIZER_NOOP(check_store_word_collision)
    NANOSM_SANITIZER_NOOP(check_simt_shmem_access)
    NANOSM_SANITIZER_NOOP(check_simt_mbar_rmw)
    NANOSM_SANITIZER_NOOP(check_simt_mbar_probe)
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
    NANOSM_SANITIZER_NOOP(record_launch_copy)
    NANOSM_SANITIZER_NOOP(record_launch_landed)
    NANOSM_SANITIZER_NOOP(record_mma_accept)
    NANOSM_SANITIZER_NOOP(record_mma_row_spacing)
    NANOSM_SANITIZER_NOOP(check_tma_store_source)
    NANOSM_SANITIZER_NOOP(record_mma_fetch_start)
    NANOSM_SANITIZER_NOOP(record_mma_operands_consumed)
    NANOSM_SANITIZER_NOOP(record_mma_retired)
    NANOSM_SANITIZER_NOOP(check_acc_transfer_vs_mma)
    NANOSM_SANITIZER_NOOP(check_acc_transfer_overlap)
    NANOSM_SANITIZER_NOOP(record_acc_transfer_accept)
    NANOSM_SANITIZER_NOOP(record_acc_transfer_final_beat)
    NANOSM_SANITIZER_NOOP(record_register_write)
    NANOSM_SANITIZER_NOOP(check_register_rewrite)
    NANOSM_SANITIZER_NOOP(panic_memory_ports)
    NANOSM_SANITIZER_NOOP(record_ordered_request)
    NANOSM_SANITIZER_NOOP(check_ordered_beat)
    NANOSM_SANITIZER_NOOP(record_ordered_pluck)
    NANOSM_SANITIZER_NOOP(check_lsu_admitted)
    NANOSM_SANITIZER_NOOP(check_buffer_lanes_valid)
#undef NANOSM_SANITIZER_NOOP
};

#endif

} // namespace nanosm

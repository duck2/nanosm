// The Sanitizer: shadow state that only decides whether to PANIC. Read the
// rule at the top of sanitizer.hpp before adding anything here.

#include "sanitizer.hpp"

#include "mbar.hpp"

#include <algorithm>
#include <bit>
#include <limits>
#include <set>
#include <sstream>

namespace nanosm {

[[noreturn]] void throw_issue_panic(const IssueSite& site, const std::string& reason) {
    throw SimError(
        "PANIC cycle=" + std::to_string(site.cycle) + " warp=" + std::to_string(site.warp) +
        " pc=" + std::to_string(site.pc) + " reason=" + reason + " inst='" +
        format_instruction(site.inst, true, site.mma_shapes) + "'");
}

[[noreturn]] void throw_mbar_rmw_panic(
    uint64_t cycle, uint32_t address, const std::string& reason, const IssueSite* requester) {
    std::string message = "PANIC cycle=" + std::to_string(cycle) + " mbar_rmw address=" +
        std::to_string(address) + " reason=" + reason;
    if (requester != nullptr) {
        message += " warp=" + std::to_string(requester->warp) +
            " pc=" + std::to_string(requester->pc) +
            " inst='" + format_instruction(requester->inst, true, requester->mma_shapes) + "'";
    } else {
        message += " source=engine";
    }
    throw SimError(message);
}

#if NANOSM_SANITIZER

namespace {

// Sanitizer diagnostics: a half-open range "[begin, end)".
std::string half_open(uint64_t begin, uint64_t end) {
    return "[" + std::to_string(begin) + ", " + std::to_string(end) + ")";
}

} // namespace

// ---------------------------------------------------------------------------
// Lifetime
// ---------------------------------------------------------------------------

void Sanitizer::record_launch(const Config& config, const Kernel& kernel, uint32_t warps) {
    lanes_ = config.lanes;
    lsu_read_streams_ = config.lsu_read_streams;
    gmem_sector_bytes_ = config.gmem_sector_bytes;
    lsu_staging_vectors_ = config.lsu_staging_vectors;
    lsu_active_commands_ = config.lsu_active_commands;
    ordered_requests_.resize(config.lsu_read_streams);
    warps_.resize(warps);
    for (auto& warp : warps_) {
        warp.ready_cycle.resize(kernel.register_count, 0);
        warp.read_until.resize(kernel.register_count, 0);
        warp.async_readers.resize(kernel.register_count, 0);
        warp.uniform_ready_cycle.resize(config.urf_registers, 0);
    }
    shmem_row_spacing_.assign(config.shmem_bytes / 4, 0);
    tmem_row_spacing_.assign(static_cast<std::size_t>(config.tmem_fine_banks) * config.tmem_words_per_fine_bank, 0);
}

void Sanitizer::record_cycle_start() {
    register_writes_.clear();
    attempt_warp_.reset();
    attempt_accepted_ = false;
}

// ---------------------------------------------------------------------------
// Admission reads no operand (rfc10 section 1). Issue decides from metadata
// and resource state, then captures operands; the machine's admission
// functions cannot see the RF at all, and this tripwire catches a capture
// placed before the decision.
// ---------------------------------------------------------------------------

void Sanitizer::record_issue_attempt(uint32_t warp) {
    attempt_warp_ = warp;
    attempt_accepted_ = false;
}

void Sanitizer::record_issue_accepted() {
    if (!attempt_warp_.has_value()) throw SimError("internal sanitizer: issue accepted without an attempt");
    attempt_accepted_ = true;
}

void Sanitizer::check_issue_capture(uint64_t cycle, uint32_t warp, uint8_t reg, bool predicate) const {
    if (attempt_warp_ == warp && attempt_accepted_) return;
    throw SimError("internal sanitizer: cycle " + std::to_string(cycle) + ": warp " + std::to_string(warp) +
        " read " + (predicate ? "p" : "r") + std::to_string(reg) +
        " in an issue attempt that admission had not accepted (a refused attempt must read nothing)");
}

// ---------------------------------------------------------------------------
// RF / predicate hazards
// ---------------------------------------------------------------------------

void Sanitizer::record_reg_ready(uint32_t warp, uint32_t reg, uint64_t visible_cycle) {
    warps_.at(warp).ready_cycle[reg] = visible_cycle;
}

void Sanitizer::record_reg_read_until(uint32_t warp, uint32_t reg, uint64_t exclusive_end) {
    auto& until = warps_.at(warp).read_until[reg];
    until = std::max(until, exclusive_end);
}

void Sanitizer::record_async_reader_add(uint32_t warp, uint32_t reg) {
    ++warps_.at(warp).async_readers[reg];
}

void Sanitizer::record_async_reader_done(uint32_t warp, uint32_t reg, AsyncReader reader) {
    auto& readers = warps_.at(warp).async_readers[reg];
    if (readers == 0) {
        throw SimError(reader == AsyncReader::acc_st
            ? "internal acc.st reader underflow" : "internal stg.v8 reader underflow");
    }
    --readers;
}

void Sanitizer::record_pred_ready(uint32_t warp, uint32_t pred, uint64_t cycle) {
    warps_.at(warp).predicate_ready_cycle[pred] = cycle;
}

void Sanitizer::check_pred_waw(const IssueSite& site, uint8_t pred) const {
    if (warps_.at(site.warp).predicate_ready_cycle[pred] > site.cycle) {
        throw_issue_panic(site, "predicate WAW hazard: missing stall for p" + std::to_string(pred));
    }
}

void Sanitizer::check_pred_raw(const IssueSite& site, uint8_t pred, uint64_t sample_cycle) const {
    if (warps_.at(site.warp).predicate_ready_cycle[pred] > sample_cycle) {
        throw_issue_panic(site, "predicate RAW hazard: missing stall for p" + std::to_string(pred));
    }
}

void Sanitizer::check_rf_raw(
    const IssueSite& site, const std::vector<std::pair<uint8_t, uint64_t>>& samples) const {
    const auto& warp = warps_.at(site.warp);
    for (const auto& [reg, sample] : samples) {
        if (warp.ready_cycle.at(reg) > sample) {
            throw_issue_panic(site, "RF RAW hazard: missing stall or depbar wait for r" + std::to_string(reg));
        }
    }
}

// The younger write lands at earliest_visible - 1 or later; the older write
// landed (or lands) at ready_cycle - 1; the older samples end before
// read_until.
void Sanitizer::check_rf_waw_war(
    const IssueSite& site, uint8_t dst, unsigned width, uint64_t earliest_visible) const {
    const auto& warp = warps_.at(site.warp);
    for (unsigned j = 0; j < width; ++j) {
        const auto reg = dst + j;
        if (warp.ready_cycle[reg] >= earliest_visible)
            throw_issue_panic(site, "RF WAW hazard: missing stall or depbar wait for r" + std::to_string(reg));
        if (warp.read_until[reg] > earliest_visible || warp.async_readers[reg] != 0)
            throw_issue_panic(site, "RF WAR hazard: pending reader of r" + std::to_string(reg));
    }
}

void Sanitizer::record_uniform_ready(uint32_t warp, uint32_t reg, uint64_t visible_cycle) {
    warps_.at(warp).uniform_ready_cycle.at(reg) = visible_cycle;
}

void Sanitizer::check_uniform_raw(
    const IssueSite& site, const std::vector<uint8_t>& sources, uint64_t sample_cycle) const {
    const auto& warp = warps_.at(site.warp);
    for (const auto reg : sources) {
        if (warp.uniform_ready_cycle.at(reg) > sample_cycle) {
            throw_issue_panic(site, "URF RAW hazard: missing stall for ur" + std::to_string(reg));
        }
    }
}

// The younger write lands at visible_cycle - 1, the older at its ready
// cycle - 1: the younger must be later.
void Sanitizer::check_uniform_waw(const IssueSite& site, uint8_t dst, uint64_t visible_cycle) const {
    if (warps_.at(site.warp).uniform_ready_cycle.at(dst) >= visible_cycle) {
        throw_issue_panic(site, "URF WAW hazard: missing stall for ur" + std::to_string(dst));
    }
}

// ---------------------------------------------------------------------------
// Declared uniformity (rfc09 pin 3): hardware never compares lanes; it reads
// a declared-uniform operand from lane 0. One that differs across lanes is
// UB, diagnosed here.
// ---------------------------------------------------------------------------

void Sanitizer::check_warp_uniform(
    const IssueSite& site, const std::vector<uint32_t>& lane_values, const std::string& what) const {
    for (const auto value : lane_values) {
        if (value != lane_values.front()) throw_issue_panic(site, what + " is not warp-uniform");
    }
}

void Sanitizer::check_uniform_mask(const IssueSite& site, uint64_t active_mask, uint64_t full_mask) const {
    if (active_mask != full_mask) {
        throw_issue_panic(site, opcode_name(site.inst.opcode, site.mma_shapes) +
            " is a uniform operation under a partial active mask");
    }
}

void Sanitizer::check_halt_converged(const IssueSite& site, uint32_t stack_depth, uint64_t active_mask,
    uint64_t full_mask) const {
    if (stack_depth != 0 || active_mask != full_mask) {
        throw_issue_panic(site, "halt inside a divergent region (" + std::to_string(stack_depth) +
            " divergence stack entries left, active mask 0x" + [&] {
                std::ostringstream out;
                out << std::hex << active_mask;
                return out.str();
            }() + ")");
    }
}

void Sanitizer::check_tma_operand_alignment(const IssueSite& site, uint32_t buffer, uint32_t barrier) const {
    if (buffer % tma_buffer_alignment != 0) {
        throw_issue_panic(site, "tma buffer address " + std::to_string(buffer) +
            " is not 64-byte aligned (the machine does not wire its low 6 bits)");
    }
    if (barrier % tma_barrier_alignment != 0) {
        throw_issue_panic(site, "tma barrier address " + std::to_string(barrier) +
            " is not 8-byte aligned (the machine does not wire its low 3 bits)");
    }
}

void Sanitizer::check_swizzle64_operand(const IssueSite& site, const std::string& operand, uint32_t pointer,
    uint32_t spacing, uint32_t row_bytes) const {
    const uint32_t line = pointer - pointer % spacing;
    if (line % swizzle64_base_alignment(spacing) != 0 || pointer % spacing + row_bytes > spacing) {
        throw_issue_panic(site, "mma operand " + operand + " at " + std::to_string(pointer) + " reads " +
            std::to_string(row_bytes) + " bytes per row at row spacing " + std::to_string(spacing) +
            ": its row 0 must lie in one row starting at a multiple of " +
            std::to_string(swizzle64_base_alignment(spacing)) + " (src/layout.hpp)");
    }
}

// ---------------------------------------------------------------------------
// Same-word stores (backlog D12): which lane's word lands is not part of the
// contract, so two lanes of one store naming one word is UB, diagnosed here.
// ---------------------------------------------------------------------------

void Sanitizer::check_store_word_collision(
    const IssueSite& site, const std::vector<uint64_t>& word_addresses, const std::string& space) const {
    std::set<uint64_t> seen;
    for (const auto address : word_addresses) {
        if (!seen.insert(address).second) throw_issue_panic(site, "multiple lanes write the same " + space + " word");
    }
}

// ---------------------------------------------------------------------------
// Shmem ownership. An outstanding engine command is modeled as owning shmem
// byte ranges from its accepted issue until an existing completion event; a
// conflicting access PANICS naming the owner and the wait that would have
// ordered it. See trash/shmem-ownership-sanitizer-2026-10-03.md.
// ---------------------------------------------------------------------------

const Sanitizer::ShmemOwner* Sanitizer::shmem_owner_overlapping(
    uint64_t begin, uint64_t end, bool writers_only) const {
    for (const auto& owner : shmem_owners_) {
        if (writers_only && !owner.write) continue;
        if (begin < owner.end && owner.begin < end) return &owner;
    }
    return nullptr;
}

std::string Sanitizer::describe_shmem_owner(const ShmemOwner& owner) const {
    std::string text;
    std::string role;
    switch (owner.kind) {
    case ShmemOwnerKind::tma_load:
        text = "tma load #" + std::to_string(owner.command);
        role = "write-owner of its destination " + half_open(owner.begin, owner.end) +
            " until its last shmem write commits";
        break;
    case ShmemOwnerKind::tma_store:
        text = "tma store #" + std::to_string(owner.command);
        role = "read-owner of its source " + half_open(owner.begin, owner.end) +
            " until its last source read";
        break;
    case ShmemOwnerKind::tma_descriptor:
        text = "tma #" + std::to_string(owner.command);
        role = "read-owner of its descriptor " + half_open(owner.begin, owner.end) +
            " until the engine reads it";
        break;
    case ShmemOwnerKind::mma_a:
    case ShmemOwnerKind::mma_b:
        // An operand owner lives from accept (command queued) to its final
        // operand read, so its command is always in the MMA queue.
        text = "mma.async #" + std::to_string(owner.command);
        text += owner.fetching ? " [fetching]" : " [queued]";
        role = std::string("read-owner of its ") +
            (owner.kind == ShmemOwnerKind::mma_a ? "A" : "B") + " operand " +
            half_open(owner.begin, owner.end) + " until its final operand read";
        break;
    case ShmemOwnerKind::launch_copy:
        return "the launch copy (write-owner of the launch block's bytes " + half_open(owner.begin, owner.end) +
            " until they land from global memory)";
    }
    return text + " (warp " + std::to_string(owner.warp) + " pc " + std::to_string(owner.pc) +
        ", accepted cycle " + std::to_string(owner.issue_cycle) + "; " + role + ")";
}

std::string Sanitizer::shmem_race(
    const std::string& access, uint64_t begin, uint64_t end, const ShmemOwner& owner) const {
    const auto overlap_begin = std::max<uint64_t>(begin, owner.begin);
    const auto overlap_end = std::min<uint64_t>(end, owner.end);
    std::string fix;
    switch (owner.kind) {
    case ShmemOwnerKind::tma_load:
    case ShmemOwnerKind::tma_store:
    case ShmemOwnerKind::tma_descriptor:
        fix = "wait the TMA completion mbar" +
            (owner.completion_mbar.has_value()
                ? " [" + std::to_string(*owner.completion_mbar) + "]" : std::string()) +
            " first";
        break;
    case ShmemOwnerKind::mma_a:
    case ShmemOwnerKind::mma_b:
        fix = "wait the operand-consumed depbar sb" + std::to_string(owner.depbar) +
            " of warp " + std::to_string(owner.warp) + " / commit mbar first";
        break;
    case ShmemOwnerKind::launch_copy:
        fix = "wait on the section's launch flag first (qtxc does when an instruction names the section)";
        break;
    }
    return "shmem race: " + access + " bytes " + half_open(overlap_begin, overlap_end) +
        " owned by " + describe_shmem_owner(owner) + "; " + fix;
}

void Sanitizer::check_simt_shmem_access(
    const IssueSite& site, const std::vector<uint32_t>& word_addresses, bool load) const {
    // Issue-time check over every word of every round. An sts still conflicts
    // with read-owners (it would change bytes an engine has yet to read); an
    // lds only with write-owners.
    for (const auto& owner : shmem_owners_) {
        if (load && !owner.write) continue;
        uint64_t first = std::numeric_limits<uint64_t>::max();
        uint64_t last = 0;
        uint32_t words = 0;
        for (const auto address : word_addresses) {
            if (address + 4 <= owner.begin || address >= owner.end) continue;
            first = std::min<uint64_t>(first, address);
            last = std::max<uint64_t>(last, static_cast<uint64_t>(address) + 4);
            ++words;
        }
        if (words == 0) continue;
        const auto access = is_atomic(site.inst.opcode) ? opcode_name(site.inst.opcode) + " writes "
            : load ? std::string("lds reads ") : std::string("sts writes ");
        throw_issue_panic(site, shmem_race(
            access + std::to_string(words) + (words == 1 ? " word," : " words,"), first, last, owner));
    }
}

void Sanitizer::check_simt_mbar_rmw(const IssueSite& site, uint32_t address) const {
    // The RMW writes both mbar words; checked at admission like sts.
    if (const auto* owner = shmem_owner_overlapping(address, address + 8ull, false)) {
        throw_issue_panic(site, shmem_race(
            opcode_name(site.inst.opcode) + " writes mbar", address, address + 8ull, *owner));
    }
}

void Sanitizer::check_simt_mbar_probe(const IssueSite& site, uint32_t address) const {
    if (const auto* owner = shmem_owner_overlapping(address, address + 8ull, true)) {
        throw_issue_panic(site, shmem_race(
            opcode_name(site.inst.opcode) + " reads mbar", address, address + 8ull, *owner));
    }
}

void Sanitizer::check_engine_mbar_rmw(uint64_t cycle, uint32_t address, bool complete_tx) const {
    // An engine's RMW is admitted when it takes the free engine (SIMT ones at issue).
    if (const auto* owner = shmem_owner_overlapping(address, address + 8ull, false)) {
        throw_mbar_rmw_panic(cycle, address, shmem_race(
            std::string(complete_tx ? "engine complete_tx" : "engine mma.commit arrive") +
                " writes mbar",
            address, address + 8ull, *owner), nullptr);
    }
}

// ---------------------------------------------------------------------------
// Missing fence.sh / race (trash/rfc12.md sections 10 and 11). The shadow of
// the SIMT shmem accesses that may still be travelling to the RAM, and the
// check an engine command's footprint makes against it at accept.
// ---------------------------------------------------------------------------

void Sanitizer::record_simt_shmem_issue(const IssueSite& site, SimtShmemHolder holder, uint32_t index,
    const std::vector<std::vector<uint32_t>>& rounds, uint32_t access_bytes, bool write) {
    // Accesses whose every round has reached the RAM are done.
    std::erase_if(simt_shmem_, [&](const SimtShmemAccess& access) {
        return access.granted == access.rounds.size() && std::all_of(access.rounds.begin(), access.rounds.end(),
            [&](const SimtShmemRound& round) { return *round.ram_cycle <= site.cycle; });
    });
    SimtShmemAccess access;
    access.holder = holder;
    access.index = index;
    access.warp = site.warp;
    access.pc = site.pc;
    access.issue_cycle = site.cycle;
    access.inst = site.inst;
    access.write = write;
    access.access_bytes = access_bytes;
    for (const auto& addresses : rounds) access.rounds.push_back({addresses, std::nullopt});
    simt_shmem_.push_back(std::move(access));
    // A write that states no row spacing (record_mma_row_spacing).
    if (write) {
        for (const auto& addresses : rounds) {
            for (const auto address : addresses) forget_row_spacing(address, uint64_t{address} + access_bytes);
        }
    }
}

void Sanitizer::record_simt_shmem_grant(uint64_t cycle, SimtShmemHolder holder, uint32_t index, uint64_t ram_cycle) {
    // The holder serves one access at a time: the one with rounds left.
    const auto access = std::find_if(simt_shmem_.begin(), simt_shmem_.end(), [&](const SimtShmemAccess& candidate) {
        return candidate.holder == holder && candidate.index == index && candidate.granted < candidate.rounds.size();
    });
    if (access == simt_shmem_.end()) throw SimError("internal sanitizer: a SIMT shmem grant has no recorded access");
    auto& round = access->rounds.at(access->granted++);
    round.ram_cycle = ram_cycle;
    if (holder != SimtShmemHolder::store_slot) return;
    // C10: an sts round writing a live mbar, at its grant.
    std::vector<std::pair<uint32_t, uint32_t>> written;
    for (const auto address : round.addresses) written.emplace_back(address, address + 4);
    if (const auto hit = live_mbar_hit(written); hit.has_value()) {
        throw_issue_panic(IssueSite{cycle, access->warp, access->pc, access->inst},
            "live mbar opacity: sts writes bytes " + half_open(hit->first_byte, hit->last_byte) + " of the live mbar " +
                describe_live_mbar(*hit->mbar));
    }
}

void Sanitizer::check_engine_footprint(
    const IssueSite& site, const std::string& label, uint64_t begin, uint64_t end, bool write) const {
    // The issuing warp's own accesses first ("missing fence.sh"), then the
    // other warps' ("shmem race"); the first conflicting access is reported.
    for (const bool own : {true, false}) {
        for (const auto& access : simt_shmem_) {
            if ((access.warp == site.warp) != own) continue;
            if (!write && !access.write) continue; // read / read never conflicts
            uint64_t first = std::numeric_limits<uint64_t>::max();
            uint64_t last = 0;
            for (const auto& round : access.rounds) {
                if (round.ram_cycle.has_value() && *round.ram_cycle <= site.cycle) continue; // reached the RAM
                for (const auto address : round.addresses) {
                    const uint64_t a = address;
                    const uint64_t b = a + access.access_bytes;
                    if (b <= begin || a >= end) continue;
                    first = std::min(first, std::max(a, begin));
                    last = std::max(last, std::min(b, end));
                }
            }
            if (last <= first) continue;
            const auto what = format_instruction(access.inst, false) + " (pc " + std::to_string(access.pc) +
                ", issued cycle " + std::to_string(access.issue_cycle) + "), whose " +
                (access.write ? "write has not reached the RAM" : "read has not sampled the RAM");
            if (own) {
                throw_issue_panic(site, "missing fence.sh: " + label + " bytes " + half_open(first, last) +
                    " overlap this warp's earlier " + what + "; issue fence.sh between them");
            }
            throw_issue_panic(site, "shmem race: " + label + " bytes " + half_open(first, last) + " overlap warp " +
                std::to_string(access.warp) + "'s " + what + "; order them with an mbar");
        }
    }
}

// ---------------------------------------------------------------------------
// C10 live-mbar opacity (trash/rfc12.md section 11).
// ---------------------------------------------------------------------------

const Sanitizer::LiveMbar* Sanitizer::live_mbar_overlapping(uint64_t begin, uint64_t end) const {
    for (const auto& mbar : live_mbars_) {
        if (begin < mbar.address + 8ull && mbar.address < end) return &mbar;
    }
    return nullptr;
}

// The first live mbar one write round touches, and the bytes of it the round writes.
std::optional<Sanitizer::LiveMbarHit> Sanitizer::live_mbar_hit(
    const std::vector<std::pair<uint32_t, uint32_t>>& byte_ranges) const {
    for (const auto& [begin, end] : byte_ranges) {
        const auto* mbar = live_mbar_overlapping(begin, end);
        if (mbar == nullptr) continue;
        LiveMbarHit hit{mbar, std::numeric_limits<uint64_t>::max(), 0};
        for (const auto& [b, e] : byte_ranges) {
            if (e <= mbar->address || b >= mbar->address + 8ull) continue;
            hit.first_byte = std::min<uint64_t>(hit.first_byte, std::max<uint64_t>(b, mbar->address));
            hit.last_byte = std::max<uint64_t>(hit.last_byte, std::min<uint64_t>(e, mbar->address + 8ull));
        }
        return hit;
    }
    return std::nullopt;
}

std::string Sanitizer::describe_live_mbar(const LiveMbar& mbar) {
    const auto origin = mbar.declared_by_launch ? std::string("declared by the launch image")
        : "mbar.init by warp " + std::to_string(mbar.warp) + " pc " + std::to_string(mbar.pc) + " at cycle " +
            std::to_string(mbar.init_cycle);
    return half_open(mbar.address, mbar.address + 8ull) + " (" + origin +
        "); until kernel end only mbar operations may touch it";
}

// The launch's declared mbars (host-initialized in the shmem image): live
// from launch. A repeated declaration is one mbar.
void Sanitizer::record_launch_mbars(const std::vector<uint32_t>& addresses) {
    for (const auto address : addresses) {
        if (live_mbar_overlapping(address, address + 8ull) != nullptr) continue;
        LiveMbar mbar;
        mbar.address = address;
        mbar.declared_by_launch = true;
        live_mbars_.push_back(mbar);
    }
}

void Sanitizer::record_mbar_init(const IssueSite& site, uint32_t address) {
    if (const auto* mbar = live_mbar_overlapping(address, address + 8ull)) {
        throw_issue_panic(site, "mbar.init over the live mbar " + describe_live_mbar(*mbar) +
            " (v0 has no mbar.inval: an mbar is initialized once per kernel)");
    }
    LiveMbar mbar;
    mbar.address = address;
    mbar.warp = site.warp;
    mbar.pc = site.pc;
    mbar.init_cycle = site.cycle;
    live_mbars_.push_back(mbar);
}

void Sanitizer::check_live_mbar_write(uint64_t cycle, const std::string& writer,
    const std::vector<std::pair<uint32_t, uint32_t>>& byte_ranges) const {
    const auto hit = live_mbar_hit(byte_ranges);
    if (!hit.has_value()) return;
    throw SimError("PANIC cycle=" + std::to_string(cycle) + " mbar address=" + std::to_string(hit->mbar->address) +
        " reason=live mbar opacity: " + writer + " writes bytes " + half_open(hit->first_byte, hit->last_byte) +
        " of the live mbar " + describe_live_mbar(*hit->mbar));
}

void Sanitizer::check_complete_tx_phase(uint64_t cycle, uint32_t address, std::optional<uint64_t> tma_command,
    uint8_t stated_parity, bool phase) {
    std::optional<bool> meant = stated_parity != 0;
    std::string who = "the host completes phase " + std::to_string(stated_parity);
    if (tma_command.has_value()) {
        // A TMA's: the phase its accept-time peek recorded (none: the
        // descriptor was invalid at accept, and its own read panicked).
        meant.reset();
        const auto record = std::find_if(tma_completion_phases_.begin(), tma_completion_phases_.end(),
            [&](const auto& entry) { return entry.first == *tma_command; });
        if (record != tma_completion_phases_.end()) {
            meant = record->second;
            who = "tma #" + std::to_string(*tma_command) + " was accepted in phase " +
                std::to_string(record->second ? 1 : 0);
            tma_completion_phases_.erase(record);
        }
    }
    if (meant.has_value() && *meant != phase) {
        throw_mbar_rmw_panic(cycle, address, "completing transaction bytes against the wrong phase (" + who +
            "; the mbar is in phase " + std::to_string(phase ? 1 : 0) + ")", nullptr);
    }
}

void Sanitizer::record_mbar_tx(uint32_t address, int32_t tx) {
    mbar_tx_[address] = tx;
}

void Sanitizer::check_kernel_end(uint64_t cycle) const {
    for (const auto& [address, tx] : mbar_tx_) {
        if (tx == 0) continue;
        throw SimError("PANIC cycle=" + std::to_string(cycle) + " mbar address=" + std::to_string(address) +
            " reason=transaction count is nonzero at kernel end (tx=" + std::to_string(tx) +
            "; a missing expect_tx or complete_tx would hang its waiters)");
    }
}

void Sanitizer::record_tma_accept(
    const IssueSite& site, uint64_t command, uint32_t descriptor_address,
    const std::array<uint8_t, tma_descriptor_bytes>& descriptor_bytes, bool load,
    uint32_t shmem_address, uint32_t completion_mbar, std::optional<uint64_t> completion_mbar_word) {
    ShmemOwner descriptor_owner;
    descriptor_owner.kind = ShmemOwnerKind::tma_descriptor;
    descriptor_owner.command = command;
    descriptor_owner.begin = descriptor_address;
    descriptor_owner.end = descriptor_address + static_cast<uint32_t>(tma_descriptor_bytes);
    descriptor_owner.warp = site.warp;
    descriptor_owner.pc = site.pc;
    descriptor_owner.issue_cycle = site.cycle;

    // rfc12 section 11: nothing the engine will read may still be travelling
    // to the RAM -- first the descriptor itself (an sts that wrote it needs a
    // fence.sh), so the bytes decoded below are the ones the engine reads.
    check_engine_footprint(site, "tma #" + std::to_string(command) + " descriptor", descriptor_owner.begin,
        descriptor_owner.end, false);

    // The descriptor the engine will read later. An invalid one has no box
    // owner: the engine's own descriptor read will panic.
    std::optional<TmaDescriptor> descriptor;
    try {
        descriptor = decode_tma_descriptor(descriptor_bytes.data(), descriptor_bytes.size());
        (void)tma_shmem_span(*descriptor, shmem_address);
    } catch (const SimError&) {
        descriptor.reset();
    }

    // A swizzle64 tile's buffer is a multiple of its row spacing and of 64
    // (src/layout.hpp). The machine shuffles by address whatever the buffer.
    if (descriptor.has_value() && descriptor->layout == TileLayout::swizzle64) {
        const auto alignment = swizzle64_base_alignment(tma_swizzle64_row_bytes(*descriptor));
        if (shmem_address % alignment != 0) {
            throw_issue_panic(site, "tma #" + std::to_string(command) + " swizzle64 buffer " +
                std::to_string(shmem_address) + " is not a multiple of " + std::to_string(alignment) +
                " (its stored row's bytes; src/layout.hpp)");
        }
    }

    std::optional<ShmemOwner> data_owner;
    descriptor_owner.completion_mbar = completion_mbar;
    if (descriptor.has_value()) {
        const auto [begin, end] = tma_shmem_span(*descriptor, shmem_address);
        const auto label = std::string(load ? "tma load #" : "tma store #") +
            std::to_string(command) + (load ? " destination" : " source");
        // The box against SIMT accesses still travelling to the RAM: a load's
        // placements write it, a store's source reads read it.
        check_engine_footprint(site, label, begin, end, load);
        // A load writes its box: any owner conflicts. A store reads it: only
        // a pending writer conflicts.
        if (const auto* owner = shmem_owner_overlapping(begin, end, !load))
            throw_issue_panic(site, shmem_race(label, begin, end, *owner));
        if (end <= std::numeric_limits<uint32_t>::max()) {
            ShmemOwner owner = descriptor_owner;
            owner.kind = load ? ShmemOwnerKind::tma_load : ShmemOwnerKind::tma_store;
            owner.write = load;
            owner.begin = static_cast<uint32_t>(begin);
            owner.end = static_cast<uint32_t>(end);
            data_owner = owner;
        }
    }
    if (const auto* owner = shmem_owner_overlapping(
            descriptor_owner.begin, descriptor_owner.end, true)) {
        throw_issue_panic(site, shmem_race(
            "tma #" + std::to_string(command) + " descriptor", descriptor_owner.begin,
            descriptor_owner.end, *owner));
    }
    shmem_owners_.push_back(descriptor_owner);
    if (data_owner.has_value()) shmem_owners_.push_back(*data_owner);
    // Row spacing (record_mma_row_spacing). A swizzle64 tile is its span's
    // words, each row's bytes shuffled inside the row: a load writes all of
    // them with the stored row's bytes (its out-of-range elements are zero
    // fills, written too). A store reads only its in-range elements' words:
    // its spacing is recorded here and its source rounds are checked as they
    // are granted (check_tma_store_source). A dense load states no spacing.
    if (data_owner.has_value()) {
        const auto [begin, end] = tma_shmem_span(*descriptor, shmem_address);
        if (descriptor->layout == TileLayout::swizzle64) {
            const auto spacing = tma_swizzle64_row_bytes(*descriptor);
            if (load) {
                std::vector<uint32_t> words;
                for (uint64_t byte = begin; byte < end; byte += 4) words.push_back(static_cast<uint32_t>(byte / 4));
                record_row_spacing(false, words, spacing, SpacingWriter::tma_load);
            } else {
                tma_store_spacings_.push_back({command, spacing, site.warp, site.pc, site.inst, site.mma_shapes});
            }
        } else if (load) {
            forget_row_spacing(begin, end);
        }
    }
    if (descriptor.has_value() && completion_mbar_word.has_value()) {
        tma_completion_phases_.emplace_back(command, decode_mbar_state(*completion_mbar_word).phase);
    }
}

void Sanitizer::record_tma_descriptor_read(
    uint64_t command, const TmaDescriptor& decoded, uint32_t shmem_address, bool load) {
    // Nothing was travelling to the descriptor at accept (else the missing-
    // fence / race check panicked) and it was read-owned since, so the engine
    // must decode exactly what the sanitizer peeked. A mismatch is a
    // sanitizer bug.
    std::erase_if(shmem_owners_, [&](const ShmemOwner& owner) {
        return owner.kind == ShmemOwnerKind::tma_descriptor && owner.command == command;
    });
    const auto [begin, end] = tma_shmem_span(decoded, shmem_address);
    const auto owner = std::find_if(shmem_owners_.begin(), shmem_owners_.end(),
        [&](const ShmemOwner& candidate) {
            return candidate.command == command &&
                (candidate.kind == ShmemOwnerKind::tma_load ||
                 candidate.kind == ShmemOwnerKind::tma_store);
        });
    if (owner == shmem_owners_.end() || owner->begin != begin || owner->end != end ||
        owner->write != load) {
        throw SimError("internal sanitizer: tma #" + std::to_string(command) +
            " decoded a descriptor different from the one peeked at accept (a write in flight "
            "at accept escaped the missing-fence / race check; the owner record cannot be trusted)");
    }
}

void Sanitizer::record_launch_copy(uint32_t begin, uint32_t end) {
    ShmemOwner owner;
    owner.kind = ShmemOwnerKind::launch_copy;
    owner.write = true;
    owner.begin = begin;
    owner.end = end;
    shmem_owners_.push_back(owner);
}

void Sanitizer::record_launch_landed(uint32_t landed_end) {
    for (auto& owner : shmem_owners_) {
        if (owner.kind == ShmemOwnerKind::launch_copy) owner.begin = std::max(owner.begin, landed_end);
    }
    std::erase_if(shmem_owners_, [](const ShmemOwner& owner) {
        return owner.kind == ShmemOwnerKind::launch_copy && owner.begin >= owner.end;
    });
}

void Sanitizer::record_tma_data_complete(uint64_t command) {
    std::erase_if(shmem_owners_, [&](const ShmemOwner& owner) {
        return owner.command == command &&
            (owner.kind == ShmemOwnerKind::tma_load || owner.kind == ShmemOwnerKind::tma_store);
    });
    std::erase_if(tma_store_spacings_, [&](const TmaStoreSpacing& store) { return store.command == command; });
}

void Sanitizer::check_tma_store_source(
    uint64_t cycle, uint64_t command, const std::vector<uint32_t>& word_addresses) const {
    const auto store = std::find_if(tma_store_spacings_.begin(), tma_store_spacings_.end(),
        [&](const TmaStoreSpacing& candidate) { return candidate.command == command; });
    if (store == tma_store_spacings_.end()) return; // a dense source, or an invalid descriptor (the machine PANICs)
    std::vector<uint32_t> words;
    words.reserve(word_addresses.size());
    for (const auto address : word_addresses) words.push_back(address / 4);
    check_row_spacing(IssueSite{cycle, store->warp, store->pc, store->inst, store->mma_shapes},
        "tma store #" + std::to_string(command) + " (swizzle64 source)", false, words, store->spacing);
}

void Sanitizer::record_mma_accept(
    const IssueSite& site, uint64_t command, const TmemRegion& result, uint8_t depbar,
    std::optional<std::pair<uint32_t, uint32_t>> a_span,
    std::optional<std::pair<uint32_t, uint32_t>> b_span) {
    for (const bool a_operand : {true, false}) {
        const auto& span = a_operand ? a_span : b_span;
        if (!span.has_value()) continue; // the machine's footprint check already panicked
        const auto label = "mma.async #" + std::to_string(command) +
            (a_operand ? " A operand" : " B operand");
        // rfc12 section 11: the port-B reads against SIMT writes still
        // travelling to the RAM (an sts into the tile needs a fence.sh).
        check_engine_footprint(site, label, span->first, span->second, false);
        if (const auto* owner = shmem_owner_overlapping(span->first, span->second, true))
            throw_issue_panic(site, shmem_race(label, span->first, span->second, *owner));
        ShmemOwner owner;
        owner.kind = a_operand ? ShmemOwnerKind::mma_a : ShmemOwnerKind::mma_b;
        owner.command = command;
        owner.begin = span->first;
        owner.end = span->second;
        owner.warp = site.warp;
        owner.pc = site.pc;
        owner.issue_cycle = site.cycle;
        owner.depbar = depbar;
        shmem_owners_.push_back(owner);
    }

    TmemOwner mma;
    mma.kind = TmemOwnerKind::mma;
    mma.id = command;
    mma.region = result;
    mma.warp = site.warp;
    mma.pc = site.pc;
    mma.issue_cycle = site.cycle;
    mma.depbar = depbar;
    // MMA after MMA is ordered by the engine queue (a later command's step
    // waits for an earlier one's result on the same words); only SIMT
    // transfers race it.
    for (const auto& owner : tmem_owners_) {
        if (owner.kind == TmemOwnerKind::mma || !tmem_overlap(owner.region, mma.region)) continue;
        throw_issue_panic(site, "tmem race: mma.async #" + std::to_string(command) + " on " +
            describe_tmem_region(result) + " (read-modify-write of its whole result) overlaps " +
            describe_tmem_owner(owner) + "; wait that transfer's depbar sb" +
            std::to_string(owner.depbar) + " of warp " + std::to_string(owner.warp) +
            " (or an mbar its warp arrives on after it) first");
    }
    tmem_owners_.push_back(mma);
}

void Sanitizer::record_mma_row_spacing(const IssueSite& site, uint64_t command, bool init,
    const TmemRegion& result, const std::vector<uint32_t>& a_words, uint32_t a_spacing,
    const std::vector<uint32_t>& b_words, uint32_t b_spacing) {
    const auto name = "mma.async #" + std::to_string(command);
    check_row_spacing(site, name + " A operand", false, a_words, a_spacing);
    check_row_spacing(site, name + " B operand", false, b_words, b_spacing);
    if (!init) check_row_spacing(site, name + " accumulator", true, result.words, result.row_words);
    record_row_spacing(true, result.words, result.row_words, SpacingWriter::mma);
}

void Sanitizer::check_row_spacing(const IssueSite& site, const std::string& reader, bool tmem,
    const std::vector<uint32_t>& words, uint32_t spacing) const {
    const auto& shadow = tmem ? tmem_row_spacing_ : shmem_row_spacing_;
    const auto unit = std::string(tmem ? " words" : " bytes");
    for (const auto word : words) {
        if (word >= shadow.size() || shadow[word] == 0) continue; // outside: the machine's PANIC
        const uint32_t written = 1u << (shadow[word] & 63u);
        if (written == spacing) continue;
        static const char* const writers[] = {"", "a TMA load into a swizzle64 tile", "a matmul's result", "an acc.st"};
        const auto where = tmem ? "tmem word " + std::to_string(word) : "shmem byte " + std::to_string(4 * word);
        throw_issue_panic(site, "row spacing mismatch: " + reader + " reads " + where + " with rows " +
            std::to_string(spacing) + unit + " apart, but " + writers[shadow[word] >> 6] + " wrote it with rows " +
            std::to_string(written) + unit + " apart (a reader states its writer's row spacing; src/layout.hpp)");
    }
}

void Sanitizer::record_row_spacing(
    bool tmem, const std::vector<uint32_t>& words, uint32_t spacing, SpacingWriter writer) {
    auto& shadow = tmem ? tmem_row_spacing_ : shmem_row_spacing_;
    const auto code = static_cast<uint8_t>((static_cast<uint32_t>(writer) << 6) | std::countr_zero(spacing));
    for (const auto word : words) {
        if (word < shadow.size()) shadow[word] = code;
    }
}

void Sanitizer::forget_row_spacing(uint64_t begin_byte, uint64_t end_byte) {
    const auto end_word = std::min<uint64_t>((end_byte + 3) / 4, shmem_row_spacing_.size());
    for (uint64_t word = begin_byte / 4; word < end_word; ++word) shmem_row_spacing_[word] = 0;
}

void Sanitizer::record_mma_fetch_start(uint64_t command) {
    for (auto& owner : shmem_owners_) {
        if (owner.command == command &&
            (owner.kind == ShmemOwnerKind::mma_a || owner.kind == ShmemOwnerKind::mma_b)) {
            owner.fetching = true;
        }
    }
}

void Sanitizer::record_mma_operands_consumed(uint64_t command) {
    std::erase_if(shmem_owners_, [&](const ShmemOwner& owner) {
        return owner.command == command &&
            (owner.kind == ShmemOwnerKind::mma_a || owner.kind == ShmemOwnerKind::mma_b);
    });
}

void Sanitizer::record_mma_retired(uint64_t command) {
    release_tmem_owner(TmemOwnerKind::mma, command); // last result retired
}

// ---------------------------------------------------------------------------
// Tmem ownership
// ---------------------------------------------------------------------------

// Two regions overlap when they share a physical tmem word: pointers are
// runtime values and may alias (a 64-row view over two 32-row results), so
// nothing but the words decides. Both word lists are sorted.
bool Sanitizer::tmem_overlap(const TmemRegion& a, const TmemRegion& b) {
    auto i = a.words.begin();
    auto j = b.words.begin();
    while (i != a.words.end() && j != b.words.end()) {
        if (*i == *j) return true;
        if (*i < *j) ++i;
        else ++j;
    }
    return false;
}

// "tmem 2048 (64 words per row) rows [0, 16) cols [8, 16)": the region as the
// instruction named it, relative to its pointer.
std::string Sanitizer::describe_tmem_region(const TmemRegion& region) {
    return "tmem " + std::to_string(region.pointer) + " (" + std::to_string(region.row_words) +
        " words per row) rows " + half_open(0, region.rows) + " cols " + half_open(0, region.columns);
}

void Sanitizer::check_tmem_pointer(const IssueSite& site, const std::string& what, uint32_t pointer,
    uint32_t row_words, uint32_t columns) const {
    if (tmem_pointer_aligned(pointer, row_words, columns)) return;
    throw_issue_panic(site, what + ": tmem pointer " + std::to_string(pointer) + " with " +
        std::to_string(row_words) + " words per row is row " + std::to_string(pointer / row_words) + " column " +
        std::to_string(pointer % row_words) + "; a pointer's row is a multiple of 4 and the " +
        std::to_string(columns) + " columns it touches stay inside its row (src/layout.hpp)");
}

std::string Sanitizer::describe_tmem_owner(const TmemOwner& owner) const {
    std::string text;
    std::string mode;
    std::string until;
    switch (owner.kind) {
    case TmemOwnerKind::mma:
        text = "mma.async #" + std::to_string(owner.id);
        mode = "write-owner";
        until = "until its last result retires";
        break;
    case TmemOwnerKind::acc_ld:
        text = "acc.ld #" + std::to_string(owner.id);
        mode = "read-owner";
        until = "until its final tmem read";
        break;
    case TmemOwnerKind::acc_st:
        text = "acc.st #" + std::to_string(owner.id);
        mode = "write-owner";
        until = "until its final tmem write";
        break;
    }
    return text + " (warp " + std::to_string(owner.warp) + " pc " + std::to_string(owner.pc) +
        ", accepted cycle " + std::to_string(owner.issue_cycle) + "; " + mode + " of " +
        describe_tmem_region(owner.region) + " " + until + ")";
}

void Sanitizer::release_tmem_owner(TmemOwnerKind kind, uint64_t id) {
    std::erase_if(tmem_owners_, [&](const TmemOwner& owner) {
        return owner.kind == kind && owner.id == id;
    });
}

void Sanitizer::check_acc_transfer_vs_mma(const IssueSite& site, const TmemRegion& region) const {
    // Racing accepted MMA work is UB (cases 01/02): no ownership hardware
    // exists, but the simulator has global knowledge and diagnoses it. An
    // accepted MMA (queued or not) write-owns every word of its result until
    // its last result retires; a transfer touching none of them is no race.
    for (const auto& owner : tmem_owners_) {
        if (owner.kind == TmemOwnerKind::mma && tmem_overlap(owner.region, region)) {
            throw_issue_panic(site,
                "accumulator transfer of " + describe_tmem_region(region) + " races accepted MMA work: " +
                describe_tmem_owner(owner) + " shares its words; wait on the mma.commit mbar phase first");
        }
    }
}

void Sanitizer::check_acc_transfer_overlap(const IssueSite& site, const TmemRegion& region, bool load) const {
    // Transfer vs transfer: an acc.ld and an acc.st run in their own engines
    // at the same time, so a load over a store still in flight (or the other
    // way round) is a race. Two stores never overlap in time (one acc.st
    // engine); two loads only read.
    for (const auto& owner : tmem_owners_) {
        if (owner.kind == TmemOwnerKind::mma) continue;
        if (load && owner.kind == TmemOwnerKind::acc_ld) continue; // two readers
        if (!tmem_overlap(owner.region, region)) continue;
        throw_issue_panic(site, std::string("tmem race: ") + (load ? "acc.ld reads " : "acc.st writes ") +
            describe_tmem_region(region) + " overlapping " +
            describe_tmem_owner(owner) + "; wait that transfer's depbar sb" +
            std::to_string(owner.depbar) + " of warp " + std::to_string(owner.warp) + " first");
    }
}

void Sanitizer::record_acc_transfer_accept(const IssueSite& site, bool load, const TmemRegion& region, uint8_t depbar) {
    // Conflicts were checked at admission (check_acc_transfer_*).
    TmemOwner owner;
    owner.kind = load ? TmemOwnerKind::acc_ld : TmemOwnerKind::acc_st;
    owner.id = next_acc_transfer_id_++;
    owner.region = region;
    owner.warp = site.warp;
    owner.pc = site.pc;
    owner.issue_cycle = site.cycle;
    owner.depbar = depbar;
    tmem_owners_.push_back(owner);
    acc_transfer_in_engine_[load ? 0 : 1].push_back(owner.id);
    // Row spacing (record_mma_row_spacing): S words per row.
    if (load) {
        check_row_spacing(site, opcode_name(site.inst.opcode), true, region.words, region.row_words);
    } else {
        record_row_spacing(true, region.words, region.row_words, SpacingWriter::acc_st);
    }
}

void Sanitizer::record_acc_transfer_final_beat(bool load) {
    auto& engine = acc_transfer_in_engine_[load ? 0 : 1];
    if (engine.empty()) throw SimError("internal sanitizer: acc transfer engine has no owner");
    const auto id = engine.front();
    engine.pop_front();
    // An acc.ld's final tmem read / an acc.st's final tmem write.
    release_tmem_owner(load ? TmemOwnerKind::acc_ld : TmemOwnerKind::acc_st, id);
}

// ---------------------------------------------------------------------------
// Clocked registers: at most one write per register per tick (src/reg.hpp).
// Only the tags live here; whether D was written is the register's own
// mechanism.
// ---------------------------------------------------------------------------

void Sanitizer::record_register_write(uint32_t id, const char* tag) {
    register_writes_.emplace_back(id, tag);
}

void Sanitizer::check_register_rewrite(
    uint32_t id, const std::string& name, uint64_t cycle, const char* tag) const {
    const char* first = "?";
    for (const auto& [written, first_tag] : register_writes_) {
        if (written == id) first = first_tag;
    }
    throw SimError(
        "PANIC cycle=" + std::to_string(cycle) + " reason=register '" + name +
        "' written twice in one tick: first by '" + first + "', then by '" + tag + "'");
}

// ---------------------------------------------------------------------------
// Boundary tripwire: an ordered ID's beats belong to its oldest outstanding
// request, contiguous and in order. The placeholder controller cannot
// violate this; a B4 controller that does panics here instead of corrupting
// a register.
// ---------------------------------------------------------------------------

std::string Sanitizer::stream_name(uint32_t stream) const {
    if (stream == 0) return "0 (TMA)";
    if (stream < lsu_read_streams_) return std::to_string(stream) + " (vector)";
    return std::to_string(stream) + " (free)";
}

void Sanitizer::record_ordered_request(uint32_t stream, uint64_t request) {
    ordered_requests_.at(stream).push_back({request, 0});
}

void Sanitizer::check_ordered_beat(
    uint64_t cycle, uint32_t stream, uint64_t request, uint32_t byte_offset) const {
    const auto contract_panic = [&](const std::string& reason) {
        throw SimError(
            "PANIC cycle=" + std::to_string(cycle) + " reason=gmem controller contract violation: " +
            reason);
    };
    const auto& order = ordered_requests_.at(stream);
    if (order.empty() || order.front().request != request) {
        if (!order.empty() && order.front().received != 0) {
            contract_panic(
                "read stream " + stream_name(stream) + " delivered a beat of request #" +
                std::to_string(request) + " while request #" + std::to_string(order.front().request) +
                " is mid-sector (" + std::to_string(order.front().received) + "/" +
                std::to_string(gmem_sector_bytes_) + " bytes)");
        }
        contract_panic(
            "read stream " + stream_name(stream) + " returned request #" +
            std::to_string(request) + " out of request order (expected #" +
            (order.empty() ? std::string("none") : std::to_string(order.front().request)) + ")");
    }
    if (byte_offset != order.front().received) {
        contract_panic(
            "read stream " + stream_name(stream) + " delivered request #" +
            std::to_string(request) + " bytes out of order");
    }
}

void Sanitizer::record_ordered_pluck(uint32_t stream, uint32_t bytes) {
    auto& order = ordered_requests_.at(stream);
    if (order.empty()) throw SimError("internal sanitizer: pluck on an empty ordered stream");
    order.front().received += bytes;
    if (order.front().received == gmem_sector_bytes_) order.pop_front();
}

// ---------------------------------------------------------------------------
// rfc09 3: every admitted command owns one assembly register for its whole
// life, so the admitted set can never exceed the pool.
// ---------------------------------------------------------------------------

void Sanitizer::check_lsu_admitted(std::size_t admitted) const {
    if (admitted > lsu_staging_vectors_ || admitted > lsu_active_commands_) {
        throw SimError(
            "internal LSU sanitizer: " + std::to_string(admitted) +
            " admitted commands exceed lsu.staging_vectors/active_commands");
    }
}

void Sanitizer::check_assembly_lanes_valid(uint64_t cycle, uint32_t assembly, const std::vector<uint8_t>& valid,
    uint32_t first_lane, uint32_t lanes, const char* reader) const {
    for (uint32_t lane = first_lane; lane < first_lane + lanes; ++lane) {
        if (valid.at(lane) != 0) continue;
        throw SimError(
            "internal LSU sanitizer: cycle " + std::to_string(cycle) + ": the " + reader +
            " read lane " + std::to_string(lane) + " of assembly register " + std::to_string(assembly) +
            ", which is not valid");
    }
}

#endif // NANOSM_SANITIZER

} // namespace nanosm

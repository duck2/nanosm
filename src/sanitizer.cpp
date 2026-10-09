// The Sanitizer: shadow state that only decides whether to PANIC. Read the
// rule at the top of sanitizer.hpp before adding anything here.

#include "sanitizer.hpp"

#include <algorithm>
#include <limits>

namespace nanosm {

[[noreturn]] void throw_issue_panic(const IssueSite& site, const std::string& reason) {
    throw SimError(
        "PANIC cycle=" + std::to_string(site.cycle) + " warp=" + std::to_string(site.warp) +
        " pc=" + std::to_string(site.pc) + " reason=" + reason + " inst='" +
        format_instruction(site.inst) + "'");
}

[[noreturn]] void throw_mbar_rmw_panic(
    uint64_t cycle, uint32_t address, const std::string& reason, const IssueSite* requester) {
    std::string message = "PANIC cycle=" + std::to_string(cycle) + " mbar_rmw address=" +
        std::to_string(address) + " reason=" + reason;
    if (requester != nullptr) {
        message += " warp=" + std::to_string(requester->warp) +
            " pc=" + std::to_string(requester->pc) +
            " inst='" + format_instruction(requester->inst) + "'";
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
    for (const auto& acc : kernel.accumulators) {
        accumulator_names_.push_back(acc.name);
        accumulator_shapes_.emplace_back(acc.rows, acc.cols);
    }
    acc_transfer_in_slot_.resize(config.tmem_banks);
    ordered_requests_.resize(config.lsu_read_streams);
    warps_.resize(warps);
    for (auto& warp : warps_) {
        warp.ready_cycle.resize(kernel.register_count, 0);
        warp.read_until.resize(kernel.register_count, 0);
        warp.async_readers.resize(kernel.register_count, 0);
    }
}

void Sanitizer::record_cycle_start() {
    port_a_written_words_.clear();
    register_writes_.clear();
}

// ---------------------------------------------------------------------------
// RF / predicate hazards
// ---------------------------------------------------------------------------

void Sanitizer::record_reg_ready(uint32_t warp, uint32_t reg, uint64_t cycle) {
    warps_.at(warp).ready_cycle[reg] = cycle;
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

void Sanitizer::check_pred_raw(const IssueSite& site, uint8_t pred) const {
    if (warps_.at(site.warp).predicate_ready_cycle[pred] > site.cycle) {
        throw_issue_panic(site, "predicate RAW hazard: missing stall for p" + std::to_string(pred));
    }
}

void Sanitizer::check_rf_raw(const IssueSite& site, const std::vector<uint8_t>& sources) const {
    const auto& warp = warps_.at(site.warp);
    for (const auto reg : sources) {
        if (warp.ready_cycle.at(reg) > site.cycle) {
            throw_issue_panic(site, "RF RAW hazard: missing stall or depbar wait for r" + std::to_string(reg));
        }
    }
}

void Sanitizer::check_rf_waw_war(const IssueSite& site, uint8_t dst, unsigned width) const {
    const auto& warp = warps_.at(site.warp);
    for (unsigned j = 0; j < width; ++j) {
        const auto reg = dst + j;
        if (warp.ready_cycle[reg] > site.cycle)
            throw_issue_panic(site, "RF WAW hazard: missing stall or depbar wait for r" + std::to_string(reg));
        if (warp.read_until[reg] > site.cycle || warp.async_readers[reg] != 0)
            throw_issue_panic(site, "RF WAR hazard: pending reader of r" + std::to_string(reg));
    }
}

// ---------------------------------------------------------------------------
// Declared uniformity (rfc09 pin 3): hardware never compares lanes'
// addresses; an lds.uni address that differs across lanes is UB, diagnosed here.
// ---------------------------------------------------------------------------

void Sanitizer::check_warp_uniform(
    const IssueSite& site, const std::vector<uint32_t>& lane_values, const std::string& what) const {
    for (const auto value : lane_values) {
        if (value != lane_values.front()) throw_issue_panic(site, what + " is not warp-uniform");
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
        throw_issue_panic(site, shmem_race(
            std::string(load ? "lds reads " : "sts writes ") + std::to_string(words) +
                (words == 1 ? " word," : " words,"),
            first, last, owner));
    }
}

void Sanitizer::check_simt_mbar_rmw(const IssueSite& site, uint32_t address) const {
    // The RMW writes both mbar words; checked at admission like sts (its
    // same-cycle port-B collision is the C5 record).
    if (const auto* owner = shmem_owner_overlapping(address, address + 8ull, false)) {
        throw_issue_panic(site, shmem_race(
            opcode_name(site.inst.opcode) + " writes mbar", address, address + 8ull, *owner));
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

void Sanitizer::record_tma_accept(
    const IssueSite& site, uint64_t command, uint32_t descriptor_address,
    const std::array<uint8_t, tma_descriptor_bytes>& descriptor_bytes) {
    ShmemOwner descriptor_owner;
    descriptor_owner.kind = ShmemOwnerKind::tma_descriptor;
    descriptor_owner.command = command;
    descriptor_owner.begin = descriptor_address;
    descriptor_owner.end = descriptor_address + static_cast<uint32_t>(tma_descriptor_bytes);
    descriptor_owner.warp = site.warp;
    descriptor_owner.pc = site.pc;
    descriptor_owner.issue_cycle = site.cycle;

    // The descriptor the engine will read later. An invalid one has no box
    // owner: the engine's own descriptor read will panic.
    std::optional<TmaDescriptor> descriptor;
    try {
        descriptor = decode_tma_descriptor(descriptor_bytes.data(), descriptor_bytes.size());
        (void)tma_shmem_span(*descriptor);
    } catch (const SimError&) {
        descriptor.reset();
    }

    std::optional<ShmemOwner> data_owner;
    if (descriptor.has_value()) {
        const bool load = descriptor->direction == TmaDirection::load;
        const auto [begin, end] = tma_shmem_span(*descriptor);
        descriptor_owner.completion_mbar = descriptor->completion_mbar;
        const auto label = std::string(load ? "tma load #" : "tma store #") +
            std::to_string(command) + (load ? " destination" : " source");
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
}

void Sanitizer::record_tma_descriptor_read(uint64_t command, const TmaDescriptor& decoded) {
    // The descriptor was read-owned since accept, so the engine must decode
    // exactly what the sanitizer peeked. A mismatch is a sanitizer bug.
    std::erase_if(shmem_owners_, [&](const ShmemOwner& owner) {
        return owner.kind == ShmemOwnerKind::tma_descriptor && owner.command == command;
    });
    const auto [begin, end] = tma_shmem_span(decoded);
    const bool load = decoded.direction == TmaDirection::load;
    const auto owner = std::find_if(shmem_owners_.begin(), shmem_owners_.end(),
        [&](const ShmemOwner& candidate) {
            return candidate.command == command &&
                (candidate.kind == ShmemOwnerKind::tma_load ||
                 candidate.kind == ShmemOwnerKind::tma_store);
        });
    if (owner == shmem_owners_.end() || owner->begin != begin || owner->end != end ||
        owner->write != load) {
        throw SimError("internal sanitizer: tma #" + std::to_string(command) +
            " decoded a descriptor different from the one peeked at accept (a write still in "
            "flight at accept changed it; the owner record cannot be trusted)");
    }
}

void Sanitizer::record_tma_data_complete(uint64_t command) {
    std::erase_if(shmem_owners_, [&](const ShmemOwner& owner) {
        return owner.command == command &&
            (owner.kind == ShmemOwnerKind::tma_load || owner.kind == ShmemOwnerKind::tma_store);
    });
}

void Sanitizer::record_mma_accept(
    const IssueSite& site, uint64_t command, uint8_t accumulator, uint8_t depbar,
    std::optional<std::pair<uint32_t, uint32_t>> a_span,
    std::optional<std::pair<uint32_t, uint32_t>> b_span) {
    for (const bool a_operand : {true, false}) {
        const auto& span = a_operand ? a_span : b_span;
        if (!span.has_value()) continue; // the machine's footprint check already panicked
        const auto label = "mma.async #" + std::to_string(command) +
            (a_operand ? " A operand" : " B operand");
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
    mma.accumulator = accumulator;
    mma.row_end = accumulator_shapes_.at(accumulator).first;
    mma.col_end = accumulator_shapes_.at(accumulator).second;
    mma.warp = site.warp;
    mma.pc = site.pc;
    mma.issue_cycle = site.cycle;
    mma.depbar = depbar;
    // MMA after MMA on one accumulator is ordered by the engine queue; only
    // SIMT transfers race it.
    for (const auto& owner : tmem_owners_) {
        if (owner.kind == TmemOwnerKind::mma || !tmem_overlap(owner, mma)) continue;
        throw_issue_panic(site, "tmem race: mma.async #" + std::to_string(command) + " on '" +
            accumulator_names_.at(accumulator) +
            "' (read-modify-write of the whole accumulator) overlaps " +
            describe_tmem_owner(owner) + "; wait that transfer's depbar sb" +
            std::to_string(owner.depbar) + " of warp " + std::to_string(owner.warp) +
            " (or an mbar its warp arrives on after it) first");
    }
    tmem_owners_.push_back(mma);
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

bool Sanitizer::tmem_overlap(const TmemOwner& a, const TmemOwner& b) {
    // The toolchain rejects overlapping accumulator allocations, so regions
    // of different accumulators never alias.
    return a.accumulator == b.accumulator &&
        a.row_begin < b.row_end && b.row_begin < a.row_end &&
        a.col_begin < b.col_end && b.col_begin < a.col_end;
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
        until = "until its final RF write";
        break;
    case TmemOwnerKind::acc_st:
        text = "acc.st #" + std::to_string(owner.id);
        mode = "write-owner";
        until = "until its final tmem write";
        break;
    }
    return text + " (warp " + std::to_string(owner.warp) + " pc " + std::to_string(owner.pc) +
        ", accepted cycle " + std::to_string(owner.issue_cycle) + "; " + mode + " of '" +
        accumulator_names_.at(owner.accumulator) + "' rows " +
        half_open(owner.row_begin, owner.row_end) + " cols " +
        half_open(owner.col_begin, owner.col_end) + " " + until + ")";
}

void Sanitizer::release_tmem_owner(TmemOwnerKind kind, uint64_t id) {
    std::erase_if(tmem_owners_, [&](const TmemOwner& owner) {
        return owner.kind == kind && owner.id == id;
    });
}

void Sanitizer::check_acc_transfer_vs_mma(const IssueSite& site, uint8_t accumulator) const {
    // Racing accepted MMA work is UB (cases 01/02): no ownership hardware
    // exists, but the simulator has global knowledge and diagnoses it. An
    // accepted MMA (queued or not) write-owns its whole accumulator until its
    // last result retires.
    for (const auto& owner : tmem_owners_) {
        if (owner.kind == TmemOwnerKind::mma && owner.accumulator == accumulator) {
            throw_issue_panic(site,
                "accumulator transfer races accepted MMA work on '" +
                accumulator_names_.at(accumulator) + "' owned by " +
                describe_tmem_owner(owner) + ": wait on the mma.commit mbar phase first");
        }
    }
}

void Sanitizer::check_acc_transfer_overlap(
    const IssueSite& site, uint8_t accumulator, bool load,
    uint32_t row, uint32_t col, uint32_t width) const {
    // Transfer vs transfer (store vs load, store vs store), checked after the
    // structural slot stall: whole-transfer (non-duplex) slots structurally
    // order same-bank transfers, so only a duplex profile can admit an
    // overlapping pair.
    TmemOwner region;
    region.accumulator = accumulator;
    region.row_begin = row;
    region.col_begin = col;
    region.row_end = row + lanes_;
    region.col_end = col + width;
    for (const auto& owner : tmem_owners_) {
        if (owner.kind == TmemOwnerKind::mma) continue;
        if (load && owner.kind == TmemOwnerKind::acc_ld) continue; // two readers
        if (!tmem_overlap(owner, region)) continue;
        throw_issue_panic(site, std::string("tmem race: ") + (load ? "acc.ld reads" : "acc.st writes") +
            " '" + accumulator_names_.at(accumulator) + "' rows " +
            half_open(region.row_begin, region.row_end) + " cols " +
            half_open(region.col_begin, region.col_end) + " overlapping " +
            describe_tmem_owner(owner) + "; wait that transfer's depbar sb" +
            std::to_string(owner.depbar) + " of warp " + std::to_string(owner.warp) + " first");
    }
}

void Sanitizer::record_acc_transfer_accept(
    const IssueSite& site, uint8_t bank, bool load, uint8_t accumulator,
    uint32_t row, uint32_t col, uint32_t width, uint8_t depbar) {
    // Conflicts were checked at admission (check_acc_transfer_*).
    TmemOwner owner;
    owner.kind = load ? TmemOwnerKind::acc_ld : TmemOwnerKind::acc_st;
    owner.id = next_acc_transfer_id_++;
    owner.accumulator = accumulator;
    owner.row_begin = row;
    owner.row_end = row + lanes_;
    owner.col_begin = col;
    owner.col_end = col + width;
    owner.warp = site.warp;
    owner.pc = site.pc;
    owner.issue_cycle = site.cycle;
    owner.depbar = depbar;
    tmem_owners_.push_back(owner);
    acc_transfer_in_slot_.at(bank)[load ? 0 : 1] = owner.id;
}

void Sanitizer::record_acc_transfer_final_beat(uint8_t bank, bool load, uint64_t commit_cycle) {
    auto& slot = acc_transfer_in_slot_.at(bank)[load ? 0 : 1];
    if (!slot.has_value()) throw SimError("internal sanitizer: acc transfer slot has no owner");
    const auto id = *slot;
    slot.reset();
    if (load) {
        pending_acc_ld_releases_.push_back({id, commit_cycle}); // final RF write commits then
    } else {
        release_tmem_owner(TmemOwnerKind::acc_st, id); // final tmem write
    }
}

void Sanitizer::record_writebacks_committed(uint64_t cycle) {
    std::erase_if(pending_acc_ld_releases_, [&](const PendingAccLoadRelease& pending) {
        if (pending.commit_cycle > cycle) return false;
        release_tmem_owner(TmemOwnerKind::acc_ld, pending.id);
        return true;
    });
}

// ---------------------------------------------------------------------------
// C5: port-A writes vs a same-cycle MMA port-B round
// ---------------------------------------------------------------------------

void Sanitizer::record_port_a_write(uint64_t byte_address, uint32_t size) {
    const auto first = static_cast<uint32_t>(byte_address / 4);
    const auto last = static_cast<uint32_t>((byte_address + size - 1) / 4);
    for (uint32_t word = first; word <= last; ++word) port_a_written_words_.push_back(word);
}

void Sanitizer::check_port_b_round(const std::vector<uint32_t>& words) const {
    for (const auto word : words) {
        if (std::find(port_a_written_words_.begin(), port_a_written_words_.end(), word) !=
            port_a_written_words_.end()) {
            throw SimError(
                "MMA port-B round and a port-A write touch shmem word " +
                std::to_string(word) + " in the same cycle");
        }
    }
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

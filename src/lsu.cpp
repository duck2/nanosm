// The memory front end of rfc09 (slice B2, corrected by B2.1 to the section
// 3a handoff table). Read top to bottom:
//
//   1. controller: one request channel (TMA and LSU alternate), a FIFO queue
//      of gmem.queue_sectors entries, one bus beat per DRAM tick (reads into
//      the ingress FIFO, writes out of the egress FIFO), one write contract
//   2. LSU admission (at issue) and request emission (one per core cycle;
//      reads from the command, writes from the egress FIFO's request pointer)
//   3. ingress: ingress FIFO -> TMA sector register (TMA stream) or straight
//      into the owning command's assembly register (vector, gather, uni)
//   4. RF stages: assembly -> RF (loads), RF -> assembly (vector stores)
//   5. egress write port: assembly (stores) / TMA store sector register ->
//      egress FIFO, one client per cycle, TMA priority
//
// Every named storage element is a register stage (see Machine::tick for the
// consumer-before-producer order that makes fall-through impossible; the
// tripwires that would catch one live in the Sanitizer). Every
// edge moves at most its declared width per cycle. No coalescer, no address
// comparison: every lane of a scalar access is its own request.
// Structures, owners and lifetimes: sim.hpp and trash/lsu-b21-2026-10-04.md.

#include "sim.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <set>
#include <sstream>

namespace nanosm {
namespace {

uint32_t beats_per_sector(const Config& config) {
    return (config.gmem_sector_bytes + config.gmem_bus_bytes_per_dram_tick - 1) /
        config.gmem_bus_bytes_per_dram_tick;
}

uint64_t full_byte_enable(uint32_t sector_bytes) {
    return sector_bytes == 64 ? ~0ull : (1ull << sector_bytes) - 1;
}

} // namespace

void Machine::probe(LsuEdge edge, uint64_t id, uint64_t aux, uint32_t bytes) {
    if (!edge_probe_enabled_) return;
    edge_events_.push_back({cycle_, dram_tick_, edge, id, aux, bytes});
}

// ---------------------------------------------------------------------------
// 1. The placeholder controller
// ---------------------------------------------------------------------------

// Request-ready: the controller queue (the entry in bus service included) has
// room. A read keeps its entry until its last beat is in the SM's ingress
// FIFO, so a full ingress FIFO eventually lowers this.
bool Machine::controller_request_ready() const {
    return gmem_outstanding_sectors() < config_.gmem_queue_sectors;
}

// One request per core cycle on the shared channel. When both clients have
// one, the client not granted last goes; a client with nothing to send never
// holds the other back. The TMA asks first in the cycle, so it is the one
// that must yield on the LSU's turn.
bool Machine::request_channel_grants(GmemClient client) const {
    if (request_channel_used_) return false;
    if (client == GmemClient::tma && request_channel_last_ == GmemClient::tma &&
        lsu_request_candidate() != nullptr) {
        return false;
    }
    return true;
}

void Machine::accept_gmem_request(GmemSectorRequest request, GmemClient client) {
    if (!controller_request_ready() || request_channel_used_) {
        throw SimError("internal gmem request accepted without request-ready");
    }
    request_channel_used_ = true;
    request_channel_last_ = client;
    gmem_queue_.push_back(std::move(request));
    trace_->gmem_sector_enqueue();
    trace_->gmem_queue_depth(cycle_, gmem_outstanding_sectors());
}

// Which queue entry goes on the bus next. The placeholder serves the queue
// front (global acceptance order), which honors the boundary contract: in
// order and non-interleaved within a read ID, free across IDs. Selection is
// by entry, and an entry's beats are contiguous, so beats of different IDs
// never interleave within a sector. Writes are never reordered (their data
// waits in egress FIFO order). The test orders exist to exercise the SM side.
std::optional<std::size_t> Machine::select_controller_entry() const {
    if (gmem_queue_.empty()) return std::nullopt;
    const auto servable = [&](const GmemSectorRequest& entry) {
        return entry.ready_dram_tick <= dram_tick_;
    };
    const auto free_id_read = [&](const GmemSectorRequest& entry) {
        return entry.load && entry.stream >= config_.lsu_read_streams;
    };
    const auto ordered_read = [&](const GmemSectorRequest& entry) {
        return entry.load && entry.stream < config_.lsu_read_streams;
    };
    const auto& front = gmem_queue_.front();
    switch (controller_test_order_) {
    case ControllerTestOrder::fifo:
        break;
    case ControllerTestOrder::reverse_free_ids:
        if (free_id_read(front)) {
            std::size_t run = 0;
            while (run < gmem_queue_.size() && free_id_read(gmem_queue_[run])) ++run;
            for (std::size_t index = run; index-- > 0;) {
                if (servable(gmem_queue_[index])) return index;
            }
            return std::nullopt;
        }
        break;
    case ControllerTestOrder::illegal_ordered_reorder:
        if (ordered_read(front)) {
            for (std::size_t index = 1; index < gmem_queue_.size(); ++index) {
                const auto& entry = gmem_queue_[index];
                if (entry.load && entry.stream == front.stream && servable(entry)) return index;
            }
        }
        break;
    case ControllerTestOrder::interleave_ordered_streams:
        // Legal: within the leading run of reads, serve the oldest read of an
        // ordered stream other than the last one served, if it is that
        // stream's oldest queued entry. Never reorders within an ID.
        if (ordered_read(front) && last_served_read_stream_ == front.stream) {
            std::set<uint32_t> seen;
            for (std::size_t index = 0; index < gmem_queue_.size() && gmem_queue_[index].load; ++index) {
                const auto& entry = gmem_queue_[index];
                const bool first_of_stream = seen.insert(entry.stream).second;
                if (index != 0 && first_of_stream && ordered_read(entry) &&
                    entry.stream != front.stream && servable(entry)) {
                    return index;
                }
            }
        }
        break;
    }
    if (servable(front)) return 0;
    return std::nullopt;
}

// One DRAM tick: deliver due write acks, start the next entry on an idle bus,
// move one beat. A read beat needs return-ready (room in the ingress FIFO as
// of the start of the core cycle); without it the controller holds the beat
// in its entry and the bus waits. A write beat comes from the egress FIFO's
// head slot, whose data has been complete and visible since its request.
void Machine::tick_gmem_controller() {
    deliver_write_acks();
    if (!gmem_bus_.has_value()) {
        const auto selected = select_controller_entry();
        if (selected.has_value()) {
            const auto entry = gmem_queue_.begin() + static_cast<std::ptrdiff_t>(*selected);
            gmem_bus_ = std::move(*entry);
            gmem_queue_.erase(entry);
            auto& request = *gmem_bus_;
            if (request.load) { // the sector as DRAM holds it when its burst starts
                last_served_read_stream_ = request.stream;
                request.data.assign(config_.gmem_sector_bytes, 0);
                const auto available = std::min<uint64_t>(
                    config_.gmem_sector_bytes, gmem_.size() - request.sector_address);
                std::memcpy(request.data.data(),
                    gmem_.data() + static_cast<std::size_t>(request.sector_address),
                    static_cast<std::size_t>(available));
            } else if (egress_fifo_.empty() || egress_fifo_.front().request != request.id) {
                throw SimError("internal write request is not at the egress FIFO head");
            }
            std::string owner = "lsu";
            if (request.owner == GmemSectorRequest::Owner::tma_load) owner = "tma load";
            if (request.owner == GmemSectorRequest::Owner::tma_store) owner = "tma store";
            trace_->gmem_bus_begin(
                cycle_, request.id, request.sector_address, request.load, std::move(owner));
            trace_->gmem_queue_depth(cycle_, gmem_outstanding_sectors());
        }
    }
    if (!gmem_bus_.has_value()) return;

    auto& request = *gmem_bus_;
    if (request.next_beat >= request.beats) throw SimError("gmem request has no bus beats left");
    const uint32_t offset = request.next_beat * config_.gmem_bus_bytes_per_dram_tick;
    const uint32_t size = std::min(
        config_.gmem_bus_bytes_per_dram_tick, config_.gmem_sector_bytes - offset);
    if (request.load) {
        const uint64_t capacity =
            static_cast<uint64_t>(config_.lsu_ingress_sectors) * config_.gmem_sector_bytes;
        // Room freed by this cycle's pluck is not seen until the next cycle.
        if (ingress_bytes_ + ingress_freed_this_cycle_ + size > capacity) {
            if (!return_ready_low_this_cycle_) { // return-ready low: hold the beat
                return_ready_low_this_cycle_ = true;
                ++trace_->lsu_counters().lsu_ingress_full_cycles;
            }
            return;
        }
        IngressBeat beat;
        beat.request = request.id;
        beat.offset = offset;
        beat.bytes.assign(request.data.begin() + offset, request.data.begin() + offset + size);
        beat.visible_cycle = cycle_ + config_.lsu_ingress_latency_cycles;
        ingress_fifo_.push_back(std::move(beat));
        ingress_bytes_ += size;
        probe(LsuEdge::bus_read_beat, request.id, offset, size);
    } else {
        auto& slot = egress_fifo_.front();
        if (!egress_slot_visible(slot)) {
            throw SimError("internal fall-through: egress slot on the bus before it is visible");
        }
        // AXI write strobes: the controller never writes an unselected byte.
        for (uint32_t byte = offset; byte < offset + size; ++byte) {
            if (((request.byte_enable >> byte) & 1u) == 0) continue;
            gmem_[static_cast<std::size_t>(request.sector_address + byte)] = slot.data[byte];
        }
        probe(LsuEdge::bus_write_beat, request.id, offset, size);
    }
    ++request.next_beat;
    trace_->gmem_bus_beat(cycle_, dram_tick_, request.id);
    if (request.next_beat != request.beats) return;

    auto complete = std::move(request);
    gmem_bus_.reset();
    trace_->gmem_bus_end(cycle_ + 1, complete.id, config_.gmem_sector_bytes);
    trace_->gmem_queue_depth(cycle_, gmem_outstanding_sectors());
    if (!complete.load) {
        egress_fifo_.pop_front(); // its last beat has left: the slot is free
        finish_gmem_write(std::move(complete));
    }
}

// One write contract for every client (rfc09 3a): the data went on the bus
// beat by beat; the acknowledgement comes gmem.latency_dram_ticks after the
// last beat, and only then are the TMA credit or the LSU slot released.
void Machine::finish_gmem_write(GmemSectorRequest request) {
    WriteAck ack;
    ack.dram_tick = dram_tick_ + config_.gmem_latency_dram_ticks;
    ack.request = request.id;
    ack.command = request.command;
    ack.client = request.owner == GmemSectorRequest::Owner::tma_store ? GmemClient::tma : GmemClient::lsu;
    if (ack.client == GmemClient::lsu && request.stream >= config_.lsu_read_streams)
        ack.tag = request.stream - config_.lsu_read_streams;
    pending_write_acks_.push_back(ack);
}

void Machine::deliver_write_acks() {
    while (!pending_write_acks_.empty() && pending_write_acks_.front().dram_tick <= dram_tick_) {
        const auto ack = pending_write_acks_.front();
        pending_write_acks_.pop_front();
        probe(LsuEdge::write_ack, ack.request, 0, 0);
        if (ack.client == GmemClient::tma) {
            release_tma_credit();
            continue;
        }
        if (ack.tag.has_value()) free_id_tags_.at(*ack.tag) = 0;
        auto& command = lsu_command(ack.command);
        ++command.acks;
        if (command.acks == command.requests_total) finish_lsu_command(command);
    }
}

// Port-A arbiter, TMA side: does the TMA have a port-A request it would start
// at the next free grant? (A complete sector or a zero fill to place, a
// descriptor or mbar-phase read, or a store walker that can start a sector's
// source read: a credit is free and the TMA store sector register is empty.)
bool Machine::tma_wants_port_a() const {
    if (!tma_returns_.empty()) return true;
    const auto id = tma_frontend_command_.has_value()
        ? tma_frontend_command_
        : next_tma_frontend_command();
    if (!id.has_value()) return false;
    const auto found = tma_commands_.find(*id);
    if (found == tma_commands_.end()) return false;
    const auto& command = found->second;
    if (command.state == TmaCommandState::descriptor || command.state == TmaCommandState::phase) {
        return true;
    }
    return command.state == TmaCommandState::walking && !command.walker_done &&
        command.descriptor.direction == TmaDirection::store &&
        tma_sector_credits_used_ < config_.tma_sector_credits && !tma_store_sector_.busy;
}

// ---------------------------------------------------------------------------
// 2. LSU admission and request emission
// ---------------------------------------------------------------------------

bool Machine::is_lsu(Opcode opcode) {
    return opcode == Opcode::ldg || opcode == Opcode::stg || opcode == Opcode::ldg_v8 ||
        opcode == Opcode::stg_v8 || opcode == Opcode::ldg_uni;
}

// Decode and check an LSU instruction's addresses at issue. Contract
// violations panic here: a non-uniform or misaligned vector/uniform base,
// a misaligned or out-of-bounds lane, two lanes storing one word.
Machine::LsuCommand Machine::plan_lsu_command(const Warp& warp, const Instruction& inst) const {
    LsuCommand command;
    command.warp = warp.id;
    command.pc = warp.pc;
    command.instruction = inst;
    const uint64_t lane_bytes = static_cast<uint64_t>(config_.lanes) * 4;
    const auto checked_address = [&](uint32_t base) {
        const auto address = static_cast<int64_t>(base) + inst.imm;
        if (address < 0) panic(warp, inst, "gmem address arithmetic underflow");
        return static_cast<uint64_t>(address);
    };
    switch (inst.opcode) {
    case Opcode::ldg_v8:
    case Opcode::stg_v8: {
        const bool load = inst.opcode == Opcode::ldg_v8;
        command.kind = load ? LsuKind::vector_load : LsuKind::vector_store;
        command.reg = load ? inst.dst : inst.src1;
        const auto address = checked_address(
            uniform_reg(warp, inst, inst.src0, opcode_name(inst.opcode) + " base address"));
        if (address % lane_bytes != 0) {
            panic(warp, inst, opcode_name(inst.opcode) + " address " + std::to_string(address) +
                " is not " + std::to_string(lane_bytes) + "-byte aligned");
        }
        const uint64_t block = 8 * lane_bytes;
        if (address > gmem_.size() || block > gmem_.size() - address) {
            panic(warp, inst, "gmem access out of bounds");
        }
        command.base = address;
        command.requests_total = static_cast<uint32_t>(block / config_.gmem_sector_bytes);
        break;
    }
    case Opcode::ldg:
    case Opcode::stg: {
        const bool load = inst.opcode == Opcode::ldg;
        command.kind = load ? LsuKind::scalar_load : LsuKind::scalar_store;
        command.reg = load ? inst.dst : inst.src1;
        std::set<uint64_t> stored_words;
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            const auto address = checked_address(rf(warp, lane, inst.src0));
            if (address % 4 != 0) panic(warp, inst, "unaligned 32-bit gmem access");
            if (address > gmem_.size() || gmem_.size() - address < 4) {
                panic(warp, inst, "gmem access out of bounds");
            }
            if (!load && !stored_words.insert(address).second) {
                panic(warp, inst, "multiple lanes write the same gmem word");
            }
            const uint64_t sector = address / config_.gmem_sector_bytes * config_.gmem_sector_bytes;
            command.sectors.push_back({sector, lane, static_cast<uint32_t>(address - sector)});
        }
        command.requests_total = config_.lanes;
        break;
    }
    case Opcode::ldg_uni: {
        command.kind = LsuKind::uniform_load;
        command.reg = inst.dst;
        const auto address = checked_address(
            uniform_reg(warp, inst, inst.src0, "ldg.uni base address"));
        if (address % 4 != 0) panic(warp, inst, "unaligned 32-bit gmem access");
        if (address > gmem_.size() || gmem_.size() - address < 4) {
            panic(warp, inst, "gmem access out of bounds");
        }
        const uint64_t sector = address / config_.gmem_sector_bytes * config_.gmem_sector_bytes;
        command.sectors.push_back({sector, 0, static_cast<uint32_t>(address - sector)});
        command.requests_total = 1;
        break;
    }
    default:
        throw SimError("internal non-LSU opcode planned as an LSU command");
    }
    return command;
}

std::optional<uint32_t> Machine::free_assembly_register() const {
    for (uint32_t index = 0; index < staging_.size(); ++index) {
        if (!staging_[index].owner.has_value()) return index;
    }
    return std::nullopt;
}

std::optional<uint32_t> Machine::free_id_tag() const {
    for (uint32_t tag = 0; tag < free_id_tags_.size(); ++tag) {
        if (free_id_tags_[tag] == 0) return tag;
    }
    return std::nullopt;
}

Machine::LsuCommand& Machine::lsu_command(uint64_t id) {
    for (auto& command : lsu_commands_) {
        if (command.id == id) return command;
    }
    throw SimError("internal reference to a retired LSU command #" + std::to_string(id));
}

void Machine::admit_lsu_command(
    Warp& warp, const Instruction& inst, LsuCommand command, uint32_t assembly) {
    auto& counters = trace_->lsu_counters();
    command.id = next_lsu_command_id_++;
    command.depbar = inst.control.set_depbar;
    command.assembly = assembly;
    command.admit_cycle = cycle_;
    auto& staging = staging_.at(assembly);
    if (staging.owner.has_value()) throw SimError("internal LSU admission into an owned assembly register");
    staging.owner = command.id;
    std::fill(staging.valid.begin(), staging.valid.end(), 0);
    staging.valid_count = 0;
    switch (command.kind) {
    case LsuKind::vector_load:
        for (uint32_t j = 0; j < 8; ++j)
            sanitizer_.record_reg_ready(warp.id, command.reg + j, std::numeric_limits<uint64_t>::max());
        // Ordered vector streams 1..read_streams-1, round-robin.
        command.stream = 1 + next_vector_stream_++ % (config_.lsu_read_streams - 1);
        ++counters.lsu_vector_loads;
        counters.lsu_useful_bytes += 32ull * config_.lanes;
        break;
    case LsuKind::vector_store:
        // WAR protection: each source stays read-owned until its RF read beat.
        for (uint32_t j = 0; j < 8; ++j) sanitizer_.record_async_reader_add(warp.id, command.reg + j);
        ++counters.lsu_vector_stores;
        counters.lsu_useful_bytes += 32ull * config_.lanes;
        break;
    case LsuKind::scalar_load:
        sanitizer_.record_reg_ready(warp.id, command.reg, std::numeric_limits<uint64_t>::max());
        ++counters.lsu_scalar_loads;
        counters.lsu_useful_bytes += 4ull * config_.lanes;
        break;
    case LsuKind::scalar_store:
        // The issue-cycle RF read beat (planned with the address operand)
        // lands the store data in the command's assembly register.
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            staging.words[lane] = rf(warp, lane, inst.src1);
            staging.valid[lane] = 1;
        }
        staging.valid_count = config_.lanes;
        sanitizer_.record_assembly_write(assembly, cycle_);
        probe(LsuEdge::rf_read_beat, command.id, inst.src1, config_.lanes * 4);
        ++counters.lsu_scalar_stores;
        counters.lsu_useful_bytes += 4ull * config_.lanes;
        break;
    case LsuKind::uniform_load:
        sanitizer_.record_reg_ready(warp.id, command.reg, std::numeric_limits<uint64_t>::max());
        ++counters.lsu_uniform_loads;
        counters.lsu_useful_bytes += 4;
        break;
    }
    ++warp.depbar[command.depbar];
    trace_->depbar(cycle_, warp.id, command.depbar, warp.depbar[command.depbar]);
    trace_->lsu_command_begin(cycle_, command.id, warp.id, opcode_name(inst.opcode));
    lsu_commands_.push_back(std::move(command));
    // Sanitizer: every admitted command owns one assembly register for its
    // whole life, so the admitted set can never exceed the pool.
    sanitizer_.check_lsu_admitted(lsu_commands_.size());
}

// The egress FIFO's request pointer: the oldest slot whose write request has
// not been issued yet.
const Machine::EgressSlot* Machine::egress_request_slot() const {
    for (const auto& slot : egress_fifo_) {
        if (!slot.request.has_value()) return &slot;
    }
    return nullptr;
}

// A slot may be offered to the controller (and its beats leave) only when
// all its words are in AND the BRAM's write-to-read latency has passed.
bool Machine::egress_slot_visible(const EgressSlot& slot) const {
    return slot.words_written == slot.words_total &&
        slot.complete_cycle + config_.lsu_egress_latency_cycles <= cycle_;
}

void Machine::accept_write_request(EgressSlot& slot, GmemClient client, std::optional<uint32_t> tag) {
    GmemSectorRequest request;
    request.id = next_gmem_request_id_++;
    request.command = slot.command;
    request.owner = client == GmemClient::tma
        ? GmemSectorRequest::Owner::tma_store : GmemSectorRequest::Owner::lsu;
    request.load = false;
    request.sector_address = slot.sector_address;
    request.byte_enable = slot.byte_enable;
    request.egress_slot = slot.id;
    request.beats = beats_per_sector(config_);
    request.ready_dram_tick = dram_tick_; // the data is present: no latency before the bus
    if (tag.has_value()) request.stream = config_.lsu_read_streams + *tag;
    slot.request = request.id;
    probe(LsuEdge::write_request, request.id, slot.id, 0);
    accept_gmem_request(std::move(request), client);
}

// TMA store sectors: the write request of the slot at the egress request
// pointer, once its data is complete and visible. Before the TMA walker's
// read requests in the same cycle (the data is waiting and holds a credit).
void Machine::advance_tma_write_requests() {
    const auto* slot = egress_request_slot();
    if (slot == nullptr || slot->client != GmemClient::tma || !egress_slot_visible(*slot)) return;
    if (!controller_request_ready() || !request_channel_grants(GmemClient::tma)) return;
    accept_write_request(const_cast<EgressSlot&>(*slot), GmemClient::tma, std::nullopt);
}

// The request channel serves LSU commands in admission order: the oldest
// command with requests left is the only candidate, so one SM's gmem
// accesses reach the controller in program order. Its next request is ready
// when: vector load - always; store - the egress request pointer is its next
// slot, complete and visible (scatter: and a free-ID tag); gather/uniform -
// a free-ID tag is available.
const Machine::LsuCommand* Machine::lsu_request_candidate() const {
    for (const auto& command : lsu_commands_) {
        if (command.requests_emitted == command.requests_total) continue;
        switch (command.kind) {
        case LsuKind::vector_load:
            return &command;
        case LsuKind::vector_store:
        case LsuKind::scalar_store: {
            const auto* slot = egress_request_slot();
            if (slot != nullptr && slot->client == GmemClient::lsu && slot->command == command.id &&
                egress_slot_visible(*slot) &&
                (command.kind == LsuKind::vector_store || free_id_tag().has_value())) {
                return &command;
            }
            return nullptr;
        }
        default:
            return free_id_tag().has_value() ? &command : nullptr;
        }
    }
    return nullptr;
}

void Machine::advance_lsu_requests() {
    const auto* candidate = lsu_request_candidate();
    if (candidate == nullptr) return;
    auto& command = lsu_command(candidate->id);
    auto& counters = trace_->lsu_counters();
    if (!controller_request_ready()) { // request-ready low
        ++counters.lsu_request_wait_cycles;
        ++command.request_wait;
        return;
    }
    if (!request_channel_grants(GmemClient::lsu)) return; // the TMA had this cycle's turn

    const auto index = command.requests_emitted;
    if (command.kind == LsuKind::vector_store || command.kind == LsuKind::scalar_store) {
        auto& slot = const_cast<EgressSlot&>(*egress_request_slot());
        if (slot.index != index) throw SimError("internal egress slot out of request order");
        std::optional<uint32_t> tag;
        if (command.kind == LsuKind::scalar_store) {
            tag = free_id_tag();
            free_id_tags_.at(*tag) = 1;
        }
        accept_write_request(slot, GmemClient::lsu, tag);
    } else {
        GmemSectorRequest request;
        request.id = next_gmem_request_id_++;
        request.command = command.id;
        request.owner = GmemSectorRequest::Owner::lsu;
        request.load = true;
        request.beats = beats_per_sector(config_);
        request.ready_dram_tick = dram_tick_ + config_.gmem_latency_dram_ticks;
        ReadRecord record;
        record.command = command.id;
        if (command.kind == LsuKind::vector_load) {
            request.stream = command.stream;
            request.sector_address = command.base + static_cast<uint64_t>(index) * config_.gmem_sector_bytes;
            record.kind = ReadKind::vector;
            record.sector_index = index;
            sanitizer_.record_ordered_request(command.stream, request.id);
        } else {
            const auto tag = free_id_tag();
            if (!tag.has_value()) throw SimError("internal scalar LSU request without a free-ID tag");
            free_id_tags_[*tag] = 1;
            const auto& sector = command.sectors.at(index);
            request.stream = config_.lsu_read_streams + *tag;
            request.sector_address = sector.sector_address;
            record.kind = command.kind == LsuKind::uniform_load ? ReadKind::uniform : ReadKind::gather;
            record.lane = sector.lane;
            record.word_offset = sector.word_offset;
            record.tag = *tag;
        }
        record.stream = request.stream;
        record.sector_address = request.sector_address;
        read_records_.emplace(request.id, std::move(record));
        accept_gmem_request(std::move(request), GmemClient::lsu);
    }
    ++counters.lsu_sectors;
    counters.lsu_bytes += config_.gmem_sector_bytes;
    if (++command.requests_emitted == command.requests_total) command.requests_done_cycle = cycle_;
}

// ---------------------------------------------------------------------------
// 3. Ingress
// ---------------------------------------------------------------------------

// The ingress FIFO read port: lsu.pluck_words_per_cycle words per cycle from
// the head, only from beats whose BRAM latency has passed, each word to its
// stream's consumer. A consumer that cannot take its word (the TMA sector
// register still holds an unplaced sector; a vector command's assembly
// register still holds the previous chunk) stalls the head: shared-FIFO
// head-of-line blocking. Gathers read the whole sector at this rate and keep
// one word.
void Machine::advance_ingress() {
    uint32_t budget = config_.lsu_pluck_words_per_cycle * 4;
    while (budget > 0 && !ingress_fifo_.empty()) {
        auto& beat = ingress_fifo_.front();
        if (beat.visible_cycle > cycle_) break; // not yet readable from the BRAM
        const auto found = read_records_.find(beat.request);
        if (found == read_records_.end()) throw SimError("internal ingress beat without a read record");
        auto& record = found->second;
        const auto kind = record.kind;
        const auto command = record.command;
        const uint32_t offset = beat.offset + beat.consumed;
        const uint32_t take = std::min<uint32_t>(
            budget, static_cast<uint32_t>(beat.bytes.size()) - beat.consumed);
        if (kind == ReadKind::tma) {
            if (!pluck_tma(beat, take)) break;
        } else if (kind == ReadKind::vector) {
            if (!pluck_vector(record, beat, take)) break;
            note_first_return(command);
        } else {
            note_first_return(command);
            pick_scalar_word(record, beat, take);
        }
        probe(LsuEdge::ingress_read, beat.request, offset, take);
        budget -= take;
        beat.consumed += take;
        ingress_bytes_ -= take;
        ingress_freed_this_cycle_ += take;
        if (beat.consumed == beat.bytes.size()) ingress_fifo_.pop_front();
    }
}

// Ingress -> the TMA stream's 32 B sector register at the pluck rate. When
// the sector is complete it becomes a placement ticket; the register stays
// occupied until the placement's last port-A round has read it.
bool Machine::pluck_tma(IngressBeat& beat, uint32_t take) {
    auto& reg = tma_sector_;
    if (reg.request.has_value() && reg.complete) return false; // unplaced sector: the head waits
    sanitizer_.check_ordered_beat(cycle_, tma_stream, beat.request, beat.offset + beat.consumed);
    auto& record = read_records_.at(beat.request);
    if (!reg.request.has_value()) {
        reg.request = beat.request;
        reg.sector_address = record.sector_address;
        reg.filled = 0;
        reg.complete = false;
    }
    std::memcpy(reg.data.data() + reg.filled, beat.bytes.data() + beat.consumed, take);
    probe(LsuEdge::tma_reg_write, beat.request, reg.filled, take);
    reg.filled += take;
    sanitizer_.record_tma_sector_write(cycle_);
    record.received += take;
    sanitizer_.record_ordered_pluck(tma_stream, take);
    if (reg.filled != config_.gmem_sector_bytes) return true;

    TmaReturn returned;
    returned.command = record.command;
    returned.sector_request = beat.request;
    returned.fragments = std::move(record.fragments);
    returned.from_sector_register = true;
    reg.complete = true;
    read_records_.erase(beat.request);
    tma_returns_.push_back(std::move(returned));
    trace_->tma_depths(cycle_, tma_commands_.size(), tma_sector_credits_used_, tma_returns_.size());
    return true;
}

// Ingress -> the owning vector load's assembly register at the pluck rate,
// lane = word offset within the 64 B chunk. No sector register. Blocked
// while the register still holds the previous complete chunk (its RF write
// has not happened yet).
bool Machine::pluck_vector(ReadRecord& record, IngressBeat& beat, uint32_t take) {
    auto& command = lsu_command(record.command);
    auto& assembly = staging_.at(command.assembly);
    if (assembly.valid_count == config_.lanes) return false;
    sanitizer_.check_ordered_beat(cycle_, record.stream, beat.request, beat.offset + beat.consumed);
    const uint32_t sectors_per_chunk = config_.lanes * 4 / config_.gmem_sector_bytes;
    if (record.sector_index / sectors_per_chunk != command.chunks_done) {
        throw SimError("internal vector sector reached the wrong chunk");
    }
    const uint32_t chunk_offset = (record.sector_index % sectors_per_chunk) * config_.gmem_sector_bytes;
    for (uint32_t byte = 0; byte < take; byte += 4) {
        const uint32_t lane = (chunk_offset + beat.offset + beat.consumed + byte) / 4;
        if (assembly.valid[lane] != 0) throw SimError("internal vector word overwrote a valid lane");
        std::memcpy(&assembly.words[lane], beat.bytes.data() + beat.consumed + byte, 4);
        assembly.valid[lane] = 1;
        ++assembly.valid_count;
    }
    sanitizer_.record_assembly_write(command.assembly, cycle_);
    probe(LsuEdge::assembly_write, command.id, beat.request, take);
    record.received += take;
    sanitizer_.record_ordered_pluck(record.stream, take);
    if (record.received == config_.gmem_sector_bytes) {
        read_records_.erase(beat.request); // record is dangling from here on
    }
    return true;
}

// Gather / ldg.uni: no sector register. The lane's word is taken off the
// FIFO read port into its lane (uniform: every lane) of the command's
// assembly register; the rest of the sector is read at the same rate and
// dropped.
void Machine::pick_scalar_word(ReadRecord& record, const IngressBeat& beat, uint32_t take) {
    const uint32_t begin = beat.offset + beat.consumed;
    if (record.word_offset >= begin && record.word_offset + 4 <= begin + take) {
        uint32_t word = 0;
        std::memcpy(&word, beat.bytes.data() + (record.word_offset - beat.offset), sizeof(word));
        auto& command = lsu_command(record.command);
        auto& assembly = staging_.at(command.assembly);
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            if (record.kind == ReadKind::gather && lane != record.lane) continue;
            assembly.words[lane] = word;
            if (assembly.valid[lane] == 0) {
                assembly.valid[lane] = 1;
                ++assembly.valid_count;
            }
        }
        sanitizer_.record_assembly_write(command.assembly, cycle_);
        probe(LsuEdge::assembly_write, command.id, beat.request, 4);
    }
    record.received += take;
    if (record.received == config_.gmem_sector_bytes) {
        free_id_tags_.at(record.tag) = 0;
        read_records_.erase(beat.request);
    }
}

// A gather or ldg.uni may already have retired while the tail of its last
// sector is still being read out of the FIFO and dropped (its free-ID tag is
// held until the sector's last byte), so this lookup tolerates retirement.
void Machine::note_first_return(uint64_t command) {
    for (auto& entry : lsu_commands_) {
        if (entry.id != command) continue;
        if (!entry.first_return_cycle.has_value()) entry.first_return_cycle = cycle_;
        return;
    }
}

// ---------------------------------------------------------------------------
// 4. RF stages
// ---------------------------------------------------------------------------

// Assembly -> RF, at the START of the tick (before this cycle's pluck and
// before commit): one full-width write beat in this cycle for the oldest
// load whose assembly register was completed in an earlier cycle. Bank
// (rD+j) mod banks; arithmetic booked at earlier issues keeps its port.
void Machine::advance_lsu_rf_writes() {
    auto& counters = trace_->lsu_counters();
    for (auto& command : lsu_commands_) {
        if (command.finishing || command.kind == LsuKind::vector_store ||
            command.kind == LsuKind::scalar_store) continue;
        auto& assembly = staging_.at(command.assembly);
        if (assembly.valid_count != config_.lanes) continue;
        sanitizer_.check_assembly_dump(command.assembly, cycle_);
        const bool vector = command.kind == LsuKind::vector_load;
        const auto reg = static_cast<uint8_t>(command.reg + (vector ? command.chunks_done : 0));
        if (!reserve_write(reg, cycle_)) {
            ++counters.lsu_rf_write_wait_cycles;
            ++command.rf_wait;
            break;
        }
        auto& warp = warps_.at(command.warp);
        Writeback writeback;
        writeback.ready_cycle = cycle_;
        writeback.warp = command.warp;
        writeback.dst = reg;
        writeback.values = assembly.words;
        writeback.reason = opcode_name(command.instruction.opcode);
        std::fill(assembly.valid.begin(), assembly.valid.end(), 0);
        assembly.valid_count = 0;
        ++command.chunks_done;
        if (!vector || command.chunks_done == 8) {
            // rD..rD+7 count as in flight until the last write.
            const uint32_t width = vector ? 8 : 1;
            for (uint32_t j = 0; j < width; ++j) sanitizer_.record_reg_ready(warp.id, command.reg + j, cycle_);
            writeback.clear_depbar = command.depbar;
            writeback.release_lsu_command = command.id;
            command.finishing = true;
        }
        probe(LsuEdge::rf_write_beat, command.id, reg, config_.lanes * 4);
        writebacks_.push_back(std::move(writeback));
        break;
    }
}

// RF -> assembly (vector stores), after issue: one full-width read beat per
// cycle, the oldest vector store whose assembly register has drained into
// the egress FIFO reads its next source register.
void Machine::advance_lsu_rf_reads() {
    auto& counters = trace_->lsu_counters();
    for (auto& command : lsu_commands_) {
        if (command.kind != LsuKind::vector_store || command.chunks_done == 8) continue;
        auto& assembly = staging_.at(command.assembly);
        if (assembly.valid_count != 0) continue;
        const auto reg = static_cast<uint8_t>(command.reg + command.chunks_done);
        const std::vector<RfRead> reads{{reg, 0}};
        if (!can_reserve_reads(reads)) {
            ++counters.lsu_rf_read_wait_cycles;
            ++command.rf_wait;
            break;
        }
        reserve_reads(reads);
        auto& warp = warps_.at(command.warp);
        trace_->rf_read(cycle_, warp.id, reg, reg % config_.rf_banks);
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            assembly.words[lane] = rf(warp, lane, reg);
            assembly.valid[lane] = 1;
        }
        assembly.valid_count = config_.lanes;
        sanitizer_.record_assembly_write(command.assembly, cycle_);
        probe(LsuEdge::rf_read_beat, command.id, reg, config_.lanes * 4);
        sanitizer_.record_async_reader_done(warp.id, reg, AsyncReader::stg_v8);
        ++command.chunks_done;
        break;
    }
}

// ---------------------------------------------------------------------------
// 5. Egress write port
// ---------------------------------------------------------------------------

// One shared write port of lsu.pluck_words_per_cycle words into the egress
// FIFO, one client per cycle, granted per 32 B slot: a started slot is
// finished by its client (never preempted); at each new grant the TMA store
// sector register wins over the LSU. The LSU's only candidate is its oldest
// store with data left (egress is in command order). A slot holds one
// sector: a TMA sector (8 words, its byte enables), half a vector chunk
// (8 words), or one scatter lane (1 word, 4 byte enables, a whole slot).
// Sources were written in an earlier cycle (tripwire); this stage runs
// before their producers in the tick.
void Machine::advance_egress_writes() {
    auto& counters = trace_->lsu_counters();
    const uint32_t sector_words = config_.gmem_sector_bytes / 4;
    const uint32_t sectors_per_chunk = config_.lanes * 4 / config_.gmem_sector_bytes;
    EgressSlot* open = nullptr;
    if (!egress_fifo_.empty() && egress_fifo_.back().words_written < egress_fifo_.back().words_total) {
        open = &egress_fifo_.back();
    }
    if (open == nullptr) {
        const bool tma_ready = tma_store_sector_.busy && tma_store_sector_.complete;
        LsuCommand* lsu = nullptr;
        for (auto& command : lsu_commands_) {
            if (command.kind == LsuKind::vector_store && command.chunks_plucked < 8) {
                lsu = &command;
                break;
            }
            if (command.kind == LsuKind::scalar_store && command.words_plucked < config_.lanes) {
                lsu = &command;
                break;
            }
        }
        if (lsu != nullptr && lsu->kind == LsuKind::vector_store &&
            staging_.at(lsu->assembly).valid_count != config_.lanes) {
            lsu = nullptr; // its next RF read beat has not landed yet
        }
        if (!tma_ready && lsu == nullptr) return;
        if (egress_fifo_.size() == config_.lsu_egress_sectors) {
            if (!tma_ready) {
                ++counters.lsu_egress_wait_cycles;
                ++lsu->egress_wait;
            }
            return;
        }
        EgressSlot slot;
        slot.id = next_egress_slot_id_++;
        slot.data.assign(config_.gmem_sector_bytes, 0);
        if (tma_ready) {
            slot.client = GmemClient::tma;
            slot.command = tma_store_sector_.command;
            slot.sector_address = tma_store_sector_.sector_address;
            slot.byte_enable = tma_store_sector_.byte_enable;
            slot.words_total = sector_words;
        } else if (lsu->kind == LsuKind::vector_store) {
            slot.client = GmemClient::lsu;
            slot.command = lsu->id;
            slot.index = lsu->chunks_plucked * sectors_per_chunk + lsu->words_plucked / sector_words;
            slot.sector_address = lsu->base + static_cast<uint64_t>(slot.index) * config_.gmem_sector_bytes;
            slot.byte_enable = full_byte_enable(config_.gmem_sector_bytes);
            slot.words_total = sector_words;
        } else {
            const auto& sector = lsu->sectors.at(lsu->words_plucked);
            slot.client = GmemClient::lsu;
            slot.command = lsu->id;
            slot.index = lsu->words_plucked;
            slot.sector_address = sector.sector_address;
            slot.byte_enable = 0xfull << sector.word_offset; // just this lane's word
            slot.words_total = 1;
        }
        egress_fifo_.push_back(std::move(slot));
        open = &egress_fifo_.back();
    }

    const uint32_t words = std::min(config_.lsu_pluck_words_per_cycle, open->words_total - open->words_written);
    if (open->client == GmemClient::tma) {
        auto& reg = tma_store_sector_;
        if (!reg.busy || !reg.complete) {
            throw SimError("internal fall-through: TMA store sector register read before it was complete");
        }
        sanitizer_.check_tma_store_sector_read(cycle_);
        std::memcpy(open->data.data() + reg.words_out * 4, reg.data.data() + reg.words_out * 4, words * 4);
        reg.words_out += words;
        if (reg.words_out == sector_words) { // drained: the walker may start the next sector
            reg.busy = false;
            reg.complete = false;
        }
    } else {
        auto& command = lsu_command(open->command);
        auto& assembly = staging_.at(command.assembly);
        sanitizer_.check_store_assembly_read(command.assembly, cycle_);
        for (uint32_t word = 0; word < words; ++word) {
            if (command.kind == LsuKind::vector_store) {
                const uint32_t byte = command.words_plucked * 4 % config_.gmem_sector_bytes;
                std::memcpy(open->data.data() + byte, &assembly.words[command.words_plucked], 4);
                if (++command.words_plucked == config_.lanes) { // chunk drained
                    std::fill(assembly.valid.begin(), assembly.valid.end(), 0);
                    assembly.valid_count = 0;
                    command.words_plucked = 0;
                    ++command.chunks_plucked;
                }
            } else {
                const auto& sector = command.sectors.at(command.words_plucked);
                std::memcpy(open->data.data() + sector.word_offset,
                    &assembly.words[command.words_plucked], 4);
                ++command.words_plucked;
            }
        }
    }
    open->words_written += words;
    if (open->words_written == open->words_total) open->complete_cycle = cycle_;
    probe(LsuEdge::egress_write, open->id, open->client == GmemClient::tma ? 0 : 1, words * 4);
}

// Store completion: the last acknowledgement. The depbar clears and the
// command's slot and assembly register are released at the next commit.
void Machine::finish_lsu_command(LsuCommand& command) {
    Writeback completion;
    completion.ready_cycle = cycle_ + 1;
    completion.warp = command.warp;
    completion.clear_depbar = command.depbar;
    completion.release_lsu_command = command.id;
    completion.reason = opcode_name(command.instruction.opcode);
    command.finishing = true;
    writebacks_.push_back(std::move(completion));
}

void Machine::release_lsu_command(uint64_t id) {
    const auto found = std::find_if(lsu_commands_.begin(), lsu_commands_.end(),
        [&](const LsuCommand& command) { return command.id == id; });
    if (found == lsu_commands_.end()) throw SimError("internal release of an unknown LSU command");
    auto& assembly = staging_.at(found->assembly);
    assembly.owner.reset();
    std::fill(assembly.valid.begin(), assembly.valid.end(), 0);
    assembly.valid_count = 0;
    trace_->lsu_command_end(cycle_, id, lsu_phase_detail(*found));
    lsu_commands_.erase(found);
}

std::string Machine::lsu_phase_detail(const LsuCommand& command) const {
    std::ostringstream out;
    out << "admit=" << command.admit_cycle
        << " requests_done=" << (command.requests_done_cycle ? std::to_string(*command.requests_done_cycle) : "-")
        << " first_return=" << (command.first_return_cycle ? std::to_string(*command.first_return_cycle) : "-")
        << (command.kind == LsuKind::vector_store || command.kind == LsuKind::scalar_store
            ? " last_ack=" : " last_rf_write=") << cycle_
        << " request_wait=" << command.request_wait
        << " rf_wait=" << command.rf_wait
        << " egress_wait=" << command.egress_wait;
    return out.str();
}

} // namespace nanosm

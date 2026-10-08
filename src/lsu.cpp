// The memory front end of rfc09 (slice B2, corrected by B2.1 to the section
// 3a handoff table; stepped as decisions + register drivers since 2026-10-08).
// Read top to bottom:
//
//   1. controller: one request channel (TMA and LSU alternate; one select), a
//      FIFO queue of gmem.queue_sectors entries, one bus beat per DRAM tick
//      (reads into the ingress FIFO, writes out of the egress FIFO), one
//      write contract
//   2. LSU admission (at issue) and request generation (one per core cycle;
//      reads from the command, writes from the egress FIFO's request pointer)
//   3. ingress: ingress FIFO -> TMA sector register (TMA stream) or straight
//      into the owning command's assembly register (vector, gather, uni)
//   4. RF stages: assembly -> RF (loads, in sim.cpp's RF write port),
//      RF -> assembly (vector stores)
//   5. egress write port: assembly (stores) / TMA store sector register ->
//      egress FIFO, one client per cycle, TMA priority
//
// Every named storage element is a register (Reg<T>, read through cur()), so
// a datum written in cycle t is visible to the next stage at t+1; the two
// BRAM FIFOs add their declared latency on top. Every edge moves at most its
// declared width per cycle. No coalescer, no address comparison: every lane
// of a scalar access is its own request. Structures, owners and lifetimes:
// sim.hpp and trash/lsu-b21-2026-10-04.md.

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
// 1. The placeholder controller and the request channel
// ---------------------------------------------------------------------------

std::size_t Machine::gmem_outstanding_sectors() const {
    return gmem_queue_.size() + (gmem_bus_.has_value() ? 1 : 0);
}

// The gmem request channel (its one select). One request per core cycle,
// accepted only while the controller queue has room (request-ready; the
// entry in bus service counts). The TMA asks first unless it was granted
// last and the LSU has a request too: they alternate, and a client with
// nothing to send never holds the other back. Within the TMA, a store sector
// at the egress request pointer goes before the walker's next read (its data
// is waiting and holds a credit). The LSU then sees the channel as the TMA
// left it (the request pointer past a TMA write granted now, one entry less
// of queue room). Request ids are assigned in the fixed order TMA write,
// walker sector, LSU.
Machine::RequestChannel Machine::select_request_channel(const TmaRequests& tma, const LsuRequestWant& lsu) {
    using Sector = TmaWalkerPlan::Sector;
    RequestChannel channel;
    const auto outstanding = gmem_outstanding_sectors();
    const bool ready = outstanding < config_.gmem_queue_sectors;
    const bool tma_turn = !(request_channel_last_ == GmemClient::tma && lsu.command.has_value());
    const bool walker_reads = tma.walker.has_value() && tma.walker->sector == Sector::load;
    const EgressSlot* tma_slot = tma.write_request ? egress_request_slot(std::nullopt) : nullptr;
    if (ready && tma_turn && tma.write_request) {
        channel.granted = GmemClient::tma;
        channel.tma_write = true;
    } else if (ready && tma_turn && walker_reads) {
        channel.granted = GmemClient::tma;
        channel.walker_proceeds = true;
    }
    if (tma.walker.has_value() &&
        (tma.walker->sector == Sector::zero_fill || tma.walker->sector == Sector::store)) {
        channel.walker_proceeds = true; // needs no request channel
    }

    const auto requested_now = channel.tma_write ? std::optional<uint64_t>(tma_slot->id) : std::nullopt;
    const auto* lsu_command = lsu_request_candidate(requested_now);
    const bool ready_after = outstanding + (channel.granted.has_value() ? 1 : 0) < config_.gmem_queue_sectors;
    if (lsu_command != nullptr) {
        if (!ready_after) { // request-ready low
            ++trace_->lsu_counters().lsu_request_wait_cycles;
            ++lsu_phases_[lsu_command->id].request_wait;
        } else if (!channel.granted.has_value()) {
            channel.granted = GmemClient::lsu;
            channel.lsu_command = lsu_command->id;
        }
    }

    assign_channel_requests(channel, tma);
    return channel;
}

// The requests the channel decision lets through this cycle, with their ids
// assigned in the fixed order TMA write, walker sector, LSU (ids appear in
// traces, so their order must not depend on anything else).
void Machine::assign_channel_requests(RequestChannel& channel, const TmaRequests& tma) {
    using Sector = TmaWalkerPlan::Sector;
    if (channel.tma_write) {
        const auto* slot = egress_request_slot(std::nullopt);
        channel.tma_write_id = next_gmem_request_id_++;
        channel.tma_write_slot = slot->id;
        channel.accepted = write_request(*slot, GmemClient::tma, channel.tma_write_id, std::nullopt);
    }
    // A walker sector takes an id whether it reads (a channel request) or
    // zero-fills (a placement ticket); a store sector has none yet.
    if (channel.walker_proceeds && tma.walker->sector != Sector::store) {
        channel.walker_id = next_gmem_request_id_++;
        if (tma.walker->sector == Sector::load) accept_tma_read(channel, *tma.walker);
    }
    if (channel.granted == GmemClient::lsu) accept_lsu_request(channel);
}

// The TMA walker's sector read. Its placement description rides in the SM's
// ID table; the data returns through the ingress FIFO and the TMA sector
// register.
void Machine::accept_tma_read(RequestChannel& channel, const TmaWalkerPlan& plan) {
    GmemSectorRequest request;
    request.id = channel.walker_id;
    request.command = plan.command_after.id;
    request.stream = tma_stream;
    request.sector_address = plan.sector_address;
    request.ready_dram_tick = dram_tick_ + config_.gmem_latency_dram_ticks;
    request.load = true;
    request.owner = GmemSectorRequest::Owner::tma_load;
    request.beats = beats_per_sector(config_);
    ReadRecord record;
    record.kind = ReadKind::tma;
    record.stream = tma_stream;
    record.command = plan.command_after.id;
    record.sector_address = plan.sector_address;
    record.fragments = plan.fragments;
    channel.new_read_records.emplace_back(request.id, std::move(record));
    sanitizer_.record_ordered_request(tma_stream, request.id);
    channel.accepted = std::move(request);
}

// The LSU command's next request: a store's write from the egress request
// pointer (a scatter takes a free-ID tag), or a read (gather/uniform take a
// tag) with its ID-table entry.
void Machine::accept_lsu_request(RequestChannel& channel) {
    const auto& command = *find_lsu_command(channel.lsu_command);
    channel.lsu_id = next_gmem_request_id_++;
    if (is_store(command.kind)) {
        const auto* slot = egress_request_slot(std::nullopt);
        if (slot->index != command.requests_emitted) throw SimError("internal egress slot out of request order");
        std::optional<uint32_t> tag;
        if (command.kind == LsuKind::scalar_store) tag = free_id_tag();
        channel.lsu_tag = tag;
        channel.lsu_write_slot = slot->id;
        channel.accepted = write_request(*slot, GmemClient::lsu, channel.lsu_id, tag);
    } else {
        std::optional<uint32_t> tag;
        if (command.kind != LsuKind::vector_load) {
            tag = free_id_tag();
            if (!tag.has_value()) throw SimError("internal scalar LSU request without a free-ID tag");
            channel.lsu_tag = tag;
        }
        channel.accepted = lsu_read_request(command, channel.lsu_id, tag);
        ReadRecord record;
        record.command = command.id;
        record.stream = channel.accepted->stream;
        record.sector_address = channel.accepted->sector_address;
        if (command.kind == LsuKind::vector_load) {
            record.kind = ReadKind::vector;
            record.sector_index = command.requests_emitted;
            sanitizer_.record_ordered_request(command.stream, channel.lsu_id);
        } else {
            const auto& sector = command.sectors.at(command.requests_emitted);
            record.kind = command.kind == LsuKind::uniform_load ? ReadKind::uniform : ReadKind::gather;
            record.lane = sector.lane;
            record.word_offset = sector.word_offset;
            record.tag = *tag;
        }
        channel.new_read_records.emplace_back(channel.lsu_id, std::move(record));
    }
    auto& counters = trace_->lsu_counters();
    ++counters.lsu_sectors;
    counters.lsu_bytes += config_.gmem_sector_bytes;
    if (command.requests_emitted + 1 == command.requests_total) {
        lsu_phases_[command.id].requests_done_cycle = cycle_;
    }
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

// The controller's core cycle (it is outside the SM; its queue, bus and B
// channel are its own state). The request the channel accepted now enters
// its queue and may go on an idle bus in this same cycle (bypass (g)). Then
// its DRAM ticks: acks, bus start, one beat each. Its effects on the SM are
// wires: beats into the ingress FIFO, slots out of the egress FIFO, write
// acks (TMA credits, free-ID tags, LSU completions).
void Machine::step_gmem_controller(Wires& w) {
    if (w.channel.accepted.has_value()) {
        if (gmem_outstanding_sectors() >= config_.gmem_queue_sectors) {
            throw SimError("internal gmem request accepted without request-ready");
        }
        request_channel_last_ = *w.channel.granted;
        gmem_queue_.push_back(*w.channel.accepted);
        trace_->gmem_sector_enqueue();
        trace_->gmem_queue_depth(cycle_, gmem_outstanding_sectors());
    }
    bool return_ready_low = false;
    const auto whole_ticks = config_.dram_clock_hz / config_.core_clock_hz;
    const auto remainder = config_.dram_clock_hz % config_.core_clock_hz;
    if (whole_ticks > 1024) throw SimError("DRAM/core clock ratio exceeds simulator limit");
    for (uint64_t i = 0; i < whole_ticks; ++i) {
        ++dram_tick_;
        tick_gmem_controller(w, return_ready_low);
    }
    if (remainder == 0) return;
    if (dram_phase_ >= config_.core_clock_hz - remainder) {
        dram_phase_ -= config_.core_clock_hz - remainder;
        ++dram_tick_;
        tick_gmem_controller(w, return_ready_low);
    } else {
        dram_phase_ += remainder;
    }
}

// The write request an egress slot carries: accepted in an earlier cycle (in
// Q), or by the channel now.
std::optional<uint64_t> Machine::egress_slot_request(const EgressSlot& slot, const Wires& w) const {
    if (slot.request.has_value()) return slot.request;
    if (w.channel.tma_write_slot == slot.id) return w.channel.tma_write_id;
    if (w.channel.lsu_write_slot == slot.id) return w.channel.lsu_id;
    return std::nullopt;
}

// One DRAM tick: deliver due write acks, start the next entry on an idle bus,
// move one beat. A read beat needs return-ready: room in the ingress FIFO,
// counting its start-of-cycle occupancy (Q) plus the beats the bus already
// wrote this cycle, and not subtracting this cycle's pops. Room a pop frees
// is reusable from the next cycle whatever lsu.ingress_latency_cycles is
// (that field sets only when a written beat becomes readable): a fixed,
// conservative capacity convention. Without return-ready the controller
// holds the beat in its entry and the bus waits.
// A write beat comes from the egress FIFO's head slot, whose data has been
// complete and visible since its request.
void Machine::tick_gmem_controller(Wires& w, bool& return_ready_low) {
    deliver_write_acks(w);
    const auto& egress = egress_fifo_.cur();
    const auto head = w.controller.egress_slots_freed; // the FIFO head, after this cycle's pops
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
            } else if (head >= egress.size() || egress_slot_request(egress[head], w) != request.id) {
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
        uint64_t occupied = ingress_.cur().bytes;
        for (const auto& beat : w.controller.ingress_beats) occupied += beat.bytes.size();
        if (occupied + size > capacity) {
            if (!return_ready_low) { // return-ready low: hold the beat
                return_ready_low = true;
                ++trace_->lsu_counters().lsu_ingress_full_cycles;
            }
            return;
        }
        IngressBeat beat;
        beat.request = request.id;
        beat.offset = offset;
        beat.bytes.assign(request.data.begin() + offset, request.data.begin() + offset + size);
        beat.visible_cycle = cycle_ + config_.lsu_ingress_latency_cycles;
        w.controller.ingress_beats.push_back(std::move(beat));
        probe(LsuEdge::bus_read_beat, request.id, offset, size);
    } else {
        const auto& slot = egress.at(head);
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
        ++w.controller.egress_slots_freed; // its last beat has left: the slot is free
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

// Acks due now. A TMA ack returns its sector credit; an LSU ack its scatter
// tag, and the command's last ack completes it: the depbar clears and its
// slot and assembly register are released at the next cycle's RF write port.
void Machine::deliver_write_acks(Wires& w) {
    while (!pending_write_acks_.empty() && pending_write_acks_.front().dram_tick <= dram_tick_) {
        const auto ack = pending_write_acks_.front();
        pending_write_acks_.pop_front();
        probe(LsuEdge::write_ack, ack.request, 0, 0);
        if (ack.client == GmemClient::tma) {
            ++w.controller.tma_credit_releases;
            continue;
        }
        if (ack.tag.has_value()) w.controller.freed_tags.push_back(*ack.tag);
        const auto* command = find_lsu_command(ack.command);
        if (command == nullptr) throw SimError("internal reference to a retired LSU command #" + std::to_string(ack.command));
        const auto acks = command->acks + ++w.controller.lsu_acks[ack.command];
        if (acks != command->requests_total) continue;
        Writeback completion;
        completion.ready_cycle = cycle_ + 1;
        completion.warp = command->warp;
        completion.clear_depbar = command->depbar;
        completion.release_lsu_command = command->id;
        completion.reason = opcode_name(command->instruction.opcode);
        w.new_writebacks.push_back(std::move(completion));
    }
}

// ---------------------------------------------------------------------------
// 2. LSU admission and request generation
// ---------------------------------------------------------------------------

bool Machine::is_lsu(Opcode opcode) {
    return opcode == Opcode::ldg || opcode == Opcode::stg || opcode == Opcode::ldg_v8 ||
        opcode == Opcode::stg_v8 || opcode == Opcode::ldg_uni;
}

bool Machine::is_store(LsuKind kind) {
    return kind == LsuKind::vector_store || kind == LsuKind::scalar_store;
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

// The lowest-numbered assembly register free after this cycle's releases (d).
std::optional<uint32_t> Machine::free_assembly_register(const Wires& w) const {
    for (uint32_t index = 0; index < staging_.size(); ++index) {
        const auto& owner = staging_[index].cur().owner;
        if (!owner.has_value()) return index;
        if (std::find(w.released_lsu_commands.begin(), w.released_lsu_commands.end(), *owner) !=
            w.released_lsu_commands.end()) {
            return index;
        }
    }
    return std::nullopt;
}

std::optional<uint32_t> Machine::free_id_tag() const {
    const auto& tags = free_id_tags_.cur();
    for (uint32_t tag = 0; tag < tags.size(); ++tag) {
        if (tags[tag] == 0) return tag;
    }
    return std::nullopt;
}

const Machine::LsuCommand* Machine::find_lsu_command(uint64_t id) const {
    for (const auto& command : lsu_commands_.cur()) {
        if (command.id == id) return &command;
    }
    return nullptr;
}

// The egress FIFO's request pointer: the oldest slot whose write request has
// not been issued (requested_now: a slot whose request the channel accepts
// in this cycle).
const Machine::EgressSlot* Machine::egress_request_slot(std::optional<uint64_t> requested_now) const {
    for (const auto& slot : egress_fifo_.cur()) {
        if (!slot.request.has_value() && slot.id != requested_now) return &slot;
    }
    return nullptr;
}

// A slot may be offered to the controller (and its beats leave) only when
// all its words are in AND the BRAM's write-to-read latency has passed.
bool Machine::egress_slot_visible(const EgressSlot& slot) const {
    return slot.words_written == slot.words_total &&
        slot.complete_cycle + config_.lsu_egress_latency_cycles <= cycle_;
}

// The request channel serves LSU commands in admission order: the oldest
// command with requests left is the only candidate, so one SM's gmem
// accesses reach the controller in program order. Its next request is ready
// when: vector load - always; store - the egress request pointer is its next
// slot, complete and visible (scatter: and a free-ID tag); gather/uniform -
// a free-ID tag is available.
const Machine::LsuCommand* Machine::lsu_request_candidate(std::optional<uint64_t> requested_now) const {
    for (const auto& command : lsu_commands_.cur()) {
        if (command.requests_emitted == command.requests_total) continue;
        switch (command.kind) {
        case LsuKind::vector_load:
            return &command;
        case LsuKind::vector_store:
        case LsuKind::scalar_store: {
            const auto* slot = egress_request_slot(requested_now);
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

Machine::LsuRequestWant Machine::lsu_request_want() const {
    LsuRequestWant want;
    if (const auto* command = lsu_request_candidate(std::nullopt); command != nullptr) want.command = command->id;
    return want;
}

Machine::GmemSectorRequest Machine::write_request(
    const EgressSlot& slot, GmemClient client, uint64_t id, std::optional<uint32_t> tag) {
    GmemSectorRequest request;
    request.id = id;
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
    probe(LsuEdge::write_request, request.id, slot.id, 0);
    return request;
}

Machine::GmemSectorRequest Machine::lsu_read_request(
    const LsuCommand& command, uint64_t id, std::optional<uint32_t> tag) {
    GmemSectorRequest request;
    request.id = id;
    request.command = command.id;
    request.owner = GmemSectorRequest::Owner::lsu;
    request.load = true;
    request.beats = beats_per_sector(config_);
    request.ready_dram_tick = dram_tick_ + config_.gmem_latency_dram_ticks;
    const auto index = command.requests_emitted;
    if (command.kind == LsuKind::vector_load) {
        request.stream = command.stream;
        request.sector_address = command.base + static_cast<uint64_t>(index) * config_.gmem_sector_bytes;
    } else {
        request.stream = config_.lsu_read_streams + *tag;
        request.sector_address = command.sectors.at(index).sector_address;
    }
    return request;
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
// one word. The destinations are seen as this cycle's drains leave them
// (same-cycle space reuse (d)): a register drained now takes new words now,
// visible next cycle; no word passes through.
Machine::IngressPluck Machine::decide_ingress(const Wires& w) {
    IngressPluck pluck;
    TmaSectorRegister tma_reg = tma_sector_.cur();
    const auto& completed = w.port_a.tma_completed;
    if (completed.has_value() && completed->kind == TmaPortKind::load_write && completed->from_sector_register) {
        tma_reg.request.reset(); // its placement's last round read it now
        tma_reg.filled = 0;
        tma_reg.complete = false;
    }
    bool tma_filled = false;
    std::map<uint64_t, ReadRecord> touched; // working copies of the read records plucked into
    const auto record_of = [&](uint64_t request) -> ReadRecord& {
        if (const auto found = touched.find(request); found != touched.end()) return found->second;
        const auto& records = read_records_.cur();
        const auto source = records.find(request);
        if (source == records.end()) throw SimError("internal ingress beat without a read record");
        return touched.emplace(request, source->second).first->second;
    };

    uint32_t budget = config_.lsu_pluck_words_per_cycle * 4;
    const auto& beats = ingress_.cur().beats;
    std::size_t index = 0;
    uint32_t consumed = beats.empty() ? 0 : beats.front().consumed;
    while (budget > 0 && index < beats.size()) {
        const auto& beat = beats[index];
        if (beat.visible_cycle > cycle_) break; // not yet readable from the BRAM
        auto& record = record_of(beat.request);
        const auto kind = record.kind;
        const auto command = record.command;
        const uint32_t offset = beat.offset + consumed;
        const uint32_t take = std::min<uint32_t>(
            budget, static_cast<uint32_t>(beat.bytes.size()) - consumed);
        if (kind == ReadKind::tma) {
            if (tma_reg.request.has_value() && tma_reg.complete) break; // unplaced sector: the head waits
            pluck_tma(beat, consumed, take, tma_reg, record, pluck);
            tma_filled = true;
        } else if (kind == ReadKind::vector) {
            if (!pluck_vector(record, beat, consumed, take, pluck, w)) break;
            note_first_return(command);
        } else {
            note_first_return(command);
            pick_scalar_word(record, beat, consumed, take, pluck);
        }
        probe(LsuEdge::ingress_read, beat.request, offset, take);
        budget -= take;
        consumed += take;
        pluck.bytes += take;
        if (consumed == beat.bytes.size()) {
            ++index;
            consumed = 0;
        }
    }
    for (auto& [id, record] : touched) {
        if (record.received == config_.gmem_sector_bytes) pluck.records[id] = std::nullopt; // fully read
        else pluck.records[id] = std::move(record);
    }
    if (tma_filled) pluck.tma_sector_after = std::move(tma_reg);
    return pluck;
}

// Ingress -> the TMA stream's 32 B sector register at the pluck rate. When
// the sector is complete it becomes a placement ticket; the register stays
// occupied until the placement's last port-A round has read it.
void Machine::pluck_tma(const IngressBeat& beat, uint32_t consumed, uint32_t take,
    TmaSectorRegister& reg, ReadRecord& record, IngressPluck& pluck) {
    sanitizer_.check_ordered_beat(cycle_, tma_stream, beat.request, beat.offset + consumed);
    if (!reg.request.has_value()) {
        reg.request = beat.request;
        reg.sector_address = record.sector_address;
        reg.filled = 0;
        reg.complete = false;
    }
    std::memcpy(reg.data.data() + reg.filled, beat.bytes.data() + consumed, take);
    probe(LsuEdge::tma_reg_write, beat.request, reg.filled, take);
    reg.filled += take;
    record.received += take;
    sanitizer_.record_ordered_pluck(tma_stream, take);
    if (reg.filled != config_.gmem_sector_bytes) return;

    TmaReturn returned;
    returned.command = record.command;
    returned.sector_request = beat.request;
    returned.fragments = record.fragments;
    returned.from_sector_register = true;
    reg.complete = true;
    pluck.tma_ticket = std::move(returned);
}

// Ingress -> the owning vector load's assembly register at the pluck rate,
// lane = word offset within the 64 B chunk. No sector register. Blocked
// while the register still holds a complete chunk. The chunk being filled is
// the command's next one, counting a chunk drained to the RF this cycle:
// filling_chunk = chunks_done + drained now.
bool Machine::pluck_vector(ReadRecord& record, const IngressBeat& beat, uint32_t consumed,
    uint32_t take, IngressPluck& pluck, const Wires& w) {
    const auto* command = find_lsu_command(record.command);
    if (command == nullptr) throw SimError("internal reference to a retired LSU command #" + std::to_string(record.command));
    const auto& start = staging_.at(command->assembly).cur();
    const bool drained = w.lsu_drain.command == command->id;
    const auto lane_valid = [&](uint32_t lane) {
        if (!drained && start.valid[lane] != 0) return true;
        return std::any_of(pluck.lanes.begin(), pluck.lanes.end(), [&](const IngressPluck::LaneWrite& write) {
            return write.assembly == command->assembly && write.lane == lane;
        });
    };
    uint32_t valid_count = drained ? 0 : start.valid_count;
    for (const auto& write : pluck.lanes) {
        if (write.assembly == command->assembly) ++valid_count;
    }
    if (valid_count == config_.lanes) return false;
    sanitizer_.check_ordered_beat(cycle_, record.stream, beat.request, beat.offset + consumed);
    const uint32_t sectors_per_chunk = config_.lanes * 4 / config_.gmem_sector_bytes;
    const uint32_t filling_chunk = command->chunks_done + (drained ? 1 : 0);
    if (record.sector_index / sectors_per_chunk != filling_chunk) {
        throw SimError("internal vector sector reached the wrong chunk");
    }
    const uint32_t chunk_offset = (record.sector_index % sectors_per_chunk) * config_.gmem_sector_bytes;
    for (uint32_t byte = 0; byte < take; byte += 4) {
        const uint32_t lane = (chunk_offset + beat.offset + consumed + byte) / 4;
        if (lane_valid(lane)) throw SimError("internal vector word overwrote a valid lane");
        uint32_t word = 0;
        std::memcpy(&word, beat.bytes.data() + consumed + byte, 4);
        pluck.lanes.push_back({command->assembly, lane, word});
    }
    probe(LsuEdge::assembly_write, command->id, beat.request, take);
    record.received += take;
    sanitizer_.record_ordered_pluck(record.stream, take);
    return true;
}

// Gather / ldg.uni: no sector register. The lane's word is taken off the
// FIFO read port into its lane (uniform: every lane) of the command's
// assembly register; the rest of the sector is read at the same rate and
// dropped. The free-ID tag returns with the sector's last byte.
void Machine::pick_scalar_word(ReadRecord& record, const IngressBeat& beat, uint32_t consumed,
    uint32_t take, IngressPluck& pluck) {
    const uint32_t begin = beat.offset + consumed;
    if (record.word_offset >= begin && record.word_offset + 4 <= begin + take) {
        uint32_t word = 0;
        std::memcpy(&word, beat.bytes.data() + (record.word_offset - beat.offset), sizeof(word));
        const auto* command = find_lsu_command(record.command);
        if (command == nullptr) throw SimError("internal reference to a retired LSU command #" + std::to_string(record.command));
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            if (record.kind == ReadKind::gather && lane != record.lane) continue;
            pluck.lanes.push_back({command->assembly, lane, word});
        }
        probe(LsuEdge::assembly_write, command->id, beat.request, 4);
    }
    record.received += take;
    if (record.received == config_.gmem_sector_bytes) pluck.freed_tags.push_back(record.tag);
}

// (T) A gather or ldg.uni may already have retired while the tail of its
// last sector is still being read out of the FIFO and dropped (its free-ID
// tag is held until the sector's last byte), so this tolerates retirement.
void Machine::note_first_return(uint64_t command) {
    const auto found = lsu_phases_.find(command);
    if (found == lsu_phases_.end()) return;
    if (!found->second.first_return_cycle.has_value()) found->second.first_return_cycle = cycle_;
}

// ---------------------------------------------------------------------------
// 4. RF read stage (vector stores). The RF write stage (assembly -> RF) is
// the RF write port's first booking: decide_lsu_rf_drain in sim.cpp.
// ---------------------------------------------------------------------------

// RF -> assembly (vector stores), after issue in RF read priority: one
// full-width read beat per cycle. The oldest vector store whose assembly
// register is empty -- in Q, or drained into the egress FIFO now (d) --
// reads its next source register. A stg.v8 issued this cycle takes part
// (its first beat in its issue cycle: bypass (e)).
Machine::LsuRfRead Machine::decide_lsu_rf_read(const Wires& w) {
    LsuRfRead read;
    std::vector<const LsuCommand*> commands;
    for (const auto& command : lsu_commands_.cur()) commands.push_back(&command);
    if (w.issue.lsu.has_value()) commands.push_back(&w.issue.lsu->command);
    for (const auto* command : commands) {
        if (command->kind != LsuKind::vector_store || command->chunks_done == 8) continue;
        const bool admitted_now = w.issue.lsu.has_value() && command == &w.issue.lsu->command;
        const bool drained_now = w.egress.lsu_chunk_drained && w.egress.lsu_command == command->id;
        const bool empty = admitted_now || drained_now || staging_.at(command->assembly).cur().valid_count == 0;
        if (!empty) continue;
        const auto reg = static_cast<uint8_t>(command->reg + command->chunks_done);
        const std::vector<RfRead> reads{{reg, 0}};
        if (!can_reserve_reads(reads)) {
            ++trace_->lsu_counters().lsu_rf_read_wait_cycles;
            ++lsu_phases_[command->id].rf_wait;
            break;
        }
        reserve_reads(reads, RfRequester::lsu_store_read);
        const auto& warp = warps_.at(command->warp);
        trace_->rf_read(cycle_, warp.id, reg, reg % config_.rf_banks);
        read.command = command->id;
        read.assembly = command->assembly;
        read.values.resize(config_.lanes);
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) read.values[lane] = rf(warp, lane, reg);
        probe(LsuEdge::rf_read_beat, command->id, reg, config_.lanes * 4);
        sanitizer_.record_async_reader_done(warp.id, reg, AsyncReader::stg_v8);
        break;
    }
    return read;
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
// Sources are read through Q: written in an earlier cycle.
Machine::EgressWrite Machine::decide_egress_write() {
    EgressWrite write;
    const auto& fifo = egress_fifo_.cur();
    const uint32_t sector_words = config_.gmem_sector_bytes / 4;
    const uint32_t sectors_per_chunk = config_.lanes * 4 / config_.gmem_sector_bytes;
    const EgressSlot* open = nullptr;
    if (!fifo.empty() && fifo.back().words_written < fifo.back().words_total) open = &fifo.back();
    if (open == nullptr) {
        const auto& tma_reg = tma_store_sector_.cur();
        const bool tma_ready = tma_reg.busy && tma_reg.complete;
        const LsuCommand* lsu = nullptr;
        for (const auto& command : lsu_commands_.cur()) {
            if ((command.kind == LsuKind::vector_store && command.chunks_plucked < 8) ||
                (command.kind == LsuKind::scalar_store && command.words_plucked < config_.lanes)) {
                lsu = &command;
                break;
            }
        }
        if (lsu != nullptr && lsu->kind == LsuKind::vector_store &&
            staging_.at(lsu->assembly).cur().valid_count != config_.lanes) {
            lsu = nullptr; // its next RF read beat has not landed yet
        }
        if (!tma_ready && lsu == nullptr) return write;
        if (fifo.size() == config_.lsu_egress_sectors) {
            if (!tma_ready) {
                ++trace_->lsu_counters().lsu_egress_wait_cycles;
                ++lsu_phases_[lsu->id].egress_wait;
            }
            return write;
        }
        EgressSlot slot;
        slot.id = next_egress_slot_id_++;
        slot.data.assign(config_.gmem_sector_bytes, 0);
        if (tma_ready) {
            slot.client = GmemClient::tma;
            slot.command = tma_reg.command;
            slot.sector_address = tma_reg.sector_address;
            slot.byte_enable = tma_reg.byte_enable;
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
        write.new_slot = std::move(slot);
        open = &*write.new_slot;
    }

    const uint32_t words = std::min(config_.lsu_pluck_words_per_cycle, open->words_total - open->words_written);
    write.client = open->client;
    if (open->client == GmemClient::tma) {
        const auto& reg = tma_store_sector_.cur();
        if (!reg.busy || !reg.complete) {
            throw SimError("internal: TMA store sector register drained before it was complete");
        }
        for (uint32_t word = 0; word < words; ++word) {
            const uint32_t byte = (reg.words_out + word) * 4;
            uint32_t value = 0;
            std::memcpy(&value, reg.data.data() + byte, 4);
            write.words.emplace_back(byte, value);
        }
        write.tma_store_drained = reg.words_out + words == sector_words; // the walker may reuse it now
    } else {
        const auto* command = find_lsu_command(open->command);
        if (command == nullptr) throw SimError("internal reference to a retired LSU command #" + std::to_string(open->command));
        const auto& assembly = staging_.at(command->assembly).cur();
        sanitizer_.check_assembly_lanes_valid(
            cycle_, command->assembly, assembly.valid, command->words_plucked, words, "egress write port");
        write.lsu_command = command->id;
        write.lsu_assembly = command->assembly;
        for (uint32_t word = 0; word < words; ++word) {
            const uint32_t lane = command->words_plucked + word;
            if (command->kind == LsuKind::vector_store) {
                write.words.emplace_back(lane * 4 % config_.gmem_sector_bytes, assembly.words[lane]);
                if (lane + 1 == config_.lanes) write.lsu_chunk_drained = true; // refillable now (d)
            } else {
                write.words.emplace_back(command->sectors.at(lane).word_offset, assembly.words[lane]);
            }
        }
    }
    write.slot_complete = open->words_written + words == open->words_total;
    probe(LsuEdge::egress_write, open->id, open->client == GmemClient::tma ? 0 : 1, words * 4);
    return write;
}

std::string Machine::lsu_phase_detail(const LsuCommand& command) const {
    LsuCommandPhases phases;
    if (const auto found = lsu_phases_.find(command.id); found != lsu_phases_.end()) phases = found->second;
    std::ostringstream out;
    out << "admit=" << phases.admit_cycle
        << " requests_done=" << (phases.requests_done_cycle ? std::to_string(*phases.requests_done_cycle) : "-")
        << " first_return=" << (phases.first_return_cycle ? std::to_string(*phases.first_return_cycle) : "-")
        << (is_store(command.kind) ? " last_ack=" : " last_rf_write=") << cycle_
        << " request_wait=" << phases.request_wait
        << " rf_wait=" << phases.rf_wait
        << " egress_wait=" << phases.egress_wait;
    return out.str();
}

} // namespace nanosm

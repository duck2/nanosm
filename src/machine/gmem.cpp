// Global memory as the machine sees it: the memory controller. One request channel (the TMA
// and the load / store unit share it), the controller's queue, the bus (one beat per DRAM
// tick, with its turnaround) and the write acknowledgements. This is part 1 of the memory
// front end described at the top of src/machine/lsu.cpp.
// (Machine is one class, src/machine/machine.hpp; the map of its files is at the top of
// src/machine/machine.cpp.)

#include "machine/machine.hpp"
#include "machine/internal.hpp"

#include <algorithm>
#include <cstring>
#include <set>
#include <utility>

namespace nanosm {

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
    if (ready && tma_turn && tma.write_request) {
        channel.granted = GmemClient::tma;
        channel.tma_write = true;
    } else if (ready && tma_turn && walker_reads) {
        channel.granted = GmemClient::tma;
        channel.walker_proceeds = true;
    }
    // (A zero fill needs no request channel: it goes when the placement
    // tickets' write port is free, step_tma; a store sector goes on its
    // A-read grant.)

    const auto* lsu_command = lsu_request_candidate(channel.tma_write);
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
// traces, so their order must not depend on anything else; a zero fill takes
// its id after these, when it goes in step_tma).
void Machine::assign_channel_requests(RequestChannel& channel, const TmaRequests& tma) {
    if (channel.tma_write) {
        const auto* slot = egress_request_slot(false);
        channel.tma_write_id = next_gmem_request_id_++;
        channel.egress_requested = true;
        channel.accepted = write_request(*slot, GmemClient::tma, channel.tma_write_id, std::nullopt);
    }
    // The walker's read sector (a store sector has no id yet).
    if (channel.walker_proceeds) {
        channel.walker_id = next_gmem_request_id_++;
        accept_tma_read(channel, *tma.walker);
    }
    if (channel.granted == GmemClient::lsu) accept_lsu_request(channel);
}

// The TMA walker's sector read. Its run is its read record, at the TMA
// stream's tail; the data returns through the ingress FIFO and the TMA sector
// register.
void Machine::accept_tma_read(RequestChannel& channel, const TmaWalkerPlan& plan) {
    GmemSectorRequest request;
    request.id = channel.walker_id;
    request.command = plan.run.command;
    request.stream = tma_stream;
    request.sector_address = plan.sector_address;
    request.ready_dram_tick = dram_tick_ + config_.gmem_latency_dram_ticks;
    request.load = true;
    request.owner = GmemSectorRequest::Owner::tma_load;
    request.beats = beats_per_sector(config_);
    channel.new_tma_read = plan.run;
    sanitizer_.record_ordered_request(tma_stream, request.id);
    channel.accepted = std::move(request);
}

// The LSU command's next request: a store's write from the egress request
// pointer (a scatter takes a free-ID tag), or a read on the ordered LSU
// stream with its read record.
void Machine::accept_lsu_request(RequestChannel& channel) {
    const auto& command = *find_lsu_command(channel.lsu_command);
    channel.lsu_id = next_gmem_request_id_++;
    if (is_store(command.kind)) {
        const auto* slot = egress_request_slot(false);
        if (slot->index != command.requests_emitted) throw SimError("internal egress slot out of request order");
        std::optional<uint32_t> tag;
        if (command.kind == LsuKind::scalar_store) tag = free_id_tag();
        channel.lsu_tag = tag;
        channel.egress_requested = true;
        channel.accepted = write_request(*slot, GmemClient::lsu, channel.lsu_id, tag);
    } else {
        // Every LSU read rides the one ordered LSU read stream: its data
        // returns in request order, which is command admission order, so the
        // one load buffer passes from a command to the next.
        channel.accepted = lsu_read_request(command, channel.lsu_id);
        sanitizer_.record_ordered_request(command.stream, channel.lsu_id);
        const auto& sector = command.sectors.at(command.requests_emitted);
        LsuRead record;
        record.kind = command.kind == LsuKind::vector_load ? LsuReadKind::vector
            : command.kind == LsuKind::uniform_load ? LsuReadKind::uniform : LsuReadKind::gather;
        record.slot = command.slot;
        record.lane = static_cast<uint8_t>(sector.lane);
        record.word_offset = static_cast<uint8_t>(sector.word_offset);
        record.command = command.id;
        channel.new_lsu_read = record;
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
    const auto ordered_read = [&](const GmemSectorRequest& entry) {
        return entry.load && entry.stream < config_.lsu_read_streams;
    };
    const auto& front = gmem_queue_.front();
    switch (controller_test_order_) {
    case ControllerTestOrder::fifo:
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

// One DRAM tick: deliver due write acks, start the next entry on an idle bus,
// move one beat. A read beat needs return-ready: room in the ingress FIFO,
// counting its start-of-cycle occupancy (Q) plus the beats the bus already
// wrote this cycle, and not subtracting this cycle's pops. Room a pop frees
// is reusable from the next cycle whatever lsu.ingress_latency_cycles is
// (that field sets only when a written beat becomes readable): a fixed,
// conservative capacity convention. Without return-ready the controller
// holds the beat in its entry and the bus waits.
// A write beat comes from the egress data FIFO's head slot, whose data has
// been complete and readable since its request.
void Machine::tick_gmem_controller(Wires& w, bool& return_ready_low) {
    deliver_write_acks(w);
    const auto head = w.controller.egress_slots_freed; // the data FIFO's head, after this cycle's pops
    // Writes leave in FIFO order: the head is the slot of the oldest accepted
    // request (the running numbers are the simulator's check of it).
    const auto head_slot = [&]() -> const EgressSector& {
        if (head >= egress_data_.readable()) {
            throw SimError("internal fall-through: egress slot on the bus before it is visible");
        }
        return egress_data_.front(head);
    };
    if (!gmem_bus_.has_value()) {
        auto selected = select_controller_entry();
        // The data bus is one bidirectional bus: between the last data beat
        // one way and the first beat the other way it stays idle for at least
        // gmem.turnaround_dram_ticks ticks (ticks it was idle anyway count).
        // The entry simply waits at the head of the controller's queue.
        if (selected.has_value() && gmem_last_beat_tick_.has_value() &&
            gmem_queue_[*selected].load != gmem_last_beat_load_ &&
            dram_tick_ - *gmem_last_beat_tick_ <= config_.gmem_turnaround_dram_ticks) {
            ++trace_->lsu_counters().gmem_turnaround_ticks;
            selected.reset();
        }
        if (selected.has_value()) {
            const auto entry = gmem_queue_.begin() + static_cast<std::ptrdiff_t>(*selected);
            // A turn is counted when the entry of the other direction starts
            // on the bus, whether it waited for the turnaround or the bus had
            // been idle long enough anyway.
            if (gmem_last_beat_tick_.has_value() && entry->load != gmem_last_beat_load_) {
                ++trace_->lsu_counters().gmem_bus_turns;
            }
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
            } else if (head >= egress_data_.readable() || egress_data_.front(head).id != request.egress_slot) {
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
        uint64_t occupied = 4 * ingress_.count();
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
        beat.stream = static_cast<uint8_t>(request.stream);
        beat.offset = offset;
        beat.bytes.assign(request.data.begin() + offset, request.data.begin() + offset + size);
        w.controller.ingress_beats.push_back(std::move(beat));
        probe(LsuEdge::bus_read_beat, request.id, offset, size);
    } else {
        const auto& slot = head_slot();
        // AXI write strobes: the controller never writes an unselected byte.
        for (uint32_t byte = offset; byte < offset + size; ++byte) {
            if (((request.byte_enable >> byte) & 1u) == 0) continue;
            gmem_[static_cast<std::size_t>(request.sector_address + byte)] = slot.data[byte];
        }
        probe(LsuEdge::bus_write_beat, request.id, offset, size);
    }
    ++request.next_beat;
    gmem_last_beat_tick_ = dram_tick_;
    gmem_last_beat_load_ = request.load;
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

// One write contract for every client: the data went on the bus
// beat by beat; the acknowledgement comes gmem.latency_dram_ticks after the
// last beat, and only then are the TMA credit or the LSU slot released.
void Machine::finish_gmem_write(GmemSectorRequest request) {
    WriteAck ack;
    ack.dram_tick = dram_tick_ + config_.gmem_latency_dram_ticks;
    ack.request = request.id;
    ack.command = request.command;
    ack.lsu_slot = request.lsu_slot;
    ack.client = request.owner == GmemSectorRequest::Owner::tma_store ? GmemClient::tma : GmemClient::lsu;
    if (ack.client == GmemClient::lsu && request.stream >= config_.lsu_read_streams)
        ack.tag = request.stream - config_.lsu_read_streams;
    pending_write_acks_.push_back(ack);
    // The B channel's size (machine.hpp, gmem_queue_): one ack per write in flight.
    const std::size_t lsu_requests = std::max<std::size_t>(
        config_.lanes, std::size_t{8} * config_.lanes * 4 / config_.gmem_sector_bytes);
    if (pending_write_acks_.size() > config_.tma_sector_credits + config_.lsu_active_commands * lsu_requests) {
        throw SimError("internal: more write acknowledgements pending than writes can be in flight");
    }
}

// Acks due now. A TMA ack returns its sector credit; an LSU ack its scatter
// tag, and the command's last ack completes it in the next cycle
// (store_completions_): its slot is released at that cycle's RF write port
// and its depbar counts down at its RAM stage.
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
        // A store holds its slot until its last acknowledgement.
        const auto* command = &lsu_commands_.cur().in_slot(ack.lsu_slot);
        if (command->id != ack.command) {
            throw SimError("internal: LSU command #" + std::to_string(ack.command) + "'s write was acknowledged to #" +
                std::to_string(command->id));
        }
        const auto acks = command->acks + ++w.controller.lsu_acks[ack.command];
        if (acks != command->requests_total) continue;
        w.store_completions.push_back(ack.lsu_slot);
    }
}

} // namespace nanosm

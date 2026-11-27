// The load / store unit: the register file's path to global memory. The
// memory front end, read top to bottom:
//
//   1. controller (src/machine/gmem.cpp): one request channel (TMA and LSU
//      alternate; one select), a FIFO queue of gmem.queue_sectors entries, one
//      bus beat per DRAM tick (reads into the ingress FIFO, writes out of the
//      egress FIFO), one write contract
//   2. the LSU's front end: its queue, its command slots reading their
//      registers and planning their requests, and request generation (one
//      per core cycle; reads from the command, writes from the egress FIFO's
//      request pointer)
//   3. ingress: ingress FIFO -> TMA sector register (TMA stream) or the load
//      buffer (ldg.v8, gather, ldg.uni)
//   4. RF stages: load buffer -> RF (in src/machine/rf.cpp's RF write port),
//      RF -> store buffer (stg.v8)
//   5. egress write port: store buffer / TMA store sector registers -> egress
//      FIFO, one client per cycle, TMA priority
//
// Every named storage element is a register (Reg<T>, read through cur()), so
// a datum written in cycle t is visible to the next stage at t+1; the two
// BRAM FIFOs add their declared latency on top. Every edge moves at most its
// declared width per cycle. No coalescer, no address comparison: every lane
// of a scalar access is its own request. One load buffer and one store buffer,
// each held only while data is in it: a store can never hold the buffer that
// returning data needs. Structures, owners and lifetimes: machine.hpp.
//
// The front end, in order (one queue, strictly first in first out, for
// every warp):
//   issue       an ldg / stg / ldg.v8 / stg.v8 / ldg.uni enters the queue at
//               its issue's edge with the active mask; issue asks only for a
//               free queue entry (Q). Its depbar and read depbar rise then.
//   issue + 1   the newest entry samples its predicate (lanes = the active
//               mask AND the write mask) and an ldg.uni its base register.
//               An entry left with no lane leaves the queue then and gives
//               back its entry and both depbars; it makes no request.
//   head        the queue's head leaves for a command slot, in order, when a
//               slot is free (as this cycle's releases leave them (d)) and,
//               for a scalar stg, the store buffer is free (Q). An ldg.uni
//               adds its immediate and is planned as it leaves (its base at
//               the URF output then).
//   slot        an unplanned command reads its vector registers through the
//               RF's async read ports (decide_rf_reads, the owed-bit rule),
//               the oldest first: its address; a scalar stg's address and
//               data in one cycle when both banks are granted, else the
//               address first. The grant of the last one lowers the read
//               depbar (a stg.v8's falls at its last data beat).
//   plan        when the address is at the RF output (Lr after its grant)
//               the command adds the immediate, checks every lane and plans
//               its requests; a scalar stg's data at the RF output fills the
//               store buffer. From the next cycle it is a command as below.
// Requests leave in command order: an unplanned oldest command holds the
// request channel's LSU candidate, so a warp's stg then ldg of one word
// reach the controller in that order and the load reads the store.
//
// In this file: parts 2 to 5, then the front end's queue and command slots
// (step_lsu_queue) and the unit's register drivers.
// (Machine is one class, src/machine/machine.hpp; the map of its files is at the top of
// src/machine/machine.cpp.)

#include "machine/machine.hpp"
#include "machine/internal.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <sstream>
#include <utility>

namespace nanosm {

namespace {

uint64_t full_byte_enable(uint32_t sector_bytes) {
    return sector_bytes == 64 ? ~0ull : (1ull << sector_bytes) - 1;
}

} // namespace

void Machine::probe(LsuEdge edge, uint64_t id, uint64_t aux, uint32_t bytes) {
    if (!edge_probe_enabled_) return;
    edge_events_.push_back({cycle_, dram_tick_, edge, id, aux, bytes});
}

// ---------------------------------------------------------------------------
// 2. The commands' plans and request generation (the queue and the command
//    slots' reads: "The front end", below)
// ---------------------------------------------------------------------------

bool Machine::is_store(LsuKind kind) {
    return kind == LsuKind::vector_store || kind == LsuKind::scalar_store;
}

// A command's plan: its lanes' addresses (base + immediate, the plan's
// adders) and checks. `bases` is its address register as read, every lane
// (an ldg.uni's: its uniform base in lane 0). Contract violations panic
// here: a misaligned or out-of-bounds lane (a v8 lane must be
// sector-aligned) or uniform base (and, in the sanitizer, two lanes of a
// stg.v8 naming one sector, or two lanes of a scalar store naming one word:
// UB, as on NVIDIA -- the machine sends every lane's write, one
// request per lane in lane order, and with the placeholder controller's
// queue-front order the highest lane's lands last; a reordering controller
// may land another, which is why no program may rely on it).
Machine::LsuFrontEnd::Plan Machine::plan_lsu_command(const LsuCommand& command, const LaneWords& bases) const {
    const auto& inst = command.instruction;
    LsuFrontEnd::Plan plan;
    plan.command = command.id;
    const auto checked_address = [&](uint32_t base) {
        const auto address = static_cast<int64_t>(base) + inst.imm;
        if (address < 0) panic(command, "gmem address arithmetic underflow");
        return static_cast<uint64_t>(address);
    };
    switch (command.kind) {
    case LsuKind::vector_load:
    case LsuKind::vector_store: {
        // One sector per ACTIVE lane, at that lane's own address: the
        // sector's eight words are rD..rD+7 (rS..rS+7) at that lane. An
        // inactive lane's address is never checked and it makes no request.
        if (config_.gmem_sector_bytes != 32) { // refused at kernel load (check_kernel)
            throw SimError("internal: " + opcode_name(inst.opcode) + " planned on a profile without 32-byte sectors");
        }
        std::vector<uint64_t> named;
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            if ((command.lanes >> lane & 1) == 0) continue;
            const auto address = checked_address(bases[lane]);
            if (address % config_.gmem_sector_bytes != 0) {
                panic(command, opcode_name(inst.opcode) + " address " + std::to_string(address) +
                    " is not " + std::to_string(config_.gmem_sector_bytes) + "-byte aligned");
            }
            if (address > gmem_.size() || gmem_.size() - address < config_.gmem_sector_bytes) {
                panic(command, "gmem access out of bounds");
            }
            named.push_back(address);
            plan.sectors.push_back({address, lane, 0});
        }
        if (command.kind == LsuKind::vector_store) {
            sanitizer_.check_store_word_collision(site(command), named, "gmem sector");
        }
        break;
    }
    case LsuKind::scalar_load:
    case LsuKind::scalar_store: {
        // Only the active lanes: an inactive lane's address is never
        // checked, it makes no request, and a load leaves its lane unwritten.
        std::vector<uint64_t> words;
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            if ((command.lanes >> lane & 1) == 0) continue;
            const auto address = checked_address(bases[lane]);
            if (address % 4 != 0) panic(command, "unaligned 32-bit gmem access");
            if (address > gmem_.size() || gmem_.size() - address < 4) {
                panic(command, "gmem access out of bounds");
            }
            words.push_back(address);
            const uint64_t sector = address / config_.gmem_sector_bytes * config_.gmem_sector_bytes;
            plan.sectors.push_back({sector, lane, static_cast<uint32_t>(address - sector)});
        }
        if (command.kind == LsuKind::scalar_store) sanitizer_.check_store_word_collision(site(command), words, "gmem");
        break;
    }
    case LsuKind::uniform_load: {
        const auto address = checked_address(bases.at(0));
        if (address % 4 != 0) panic(command, "unaligned 32-bit gmem access");
        if (address > gmem_.size() || gmem_.size() - address < 4) {
            panic(command, "gmem access out of bounds");
        }
        const uint64_t sector = address / config_.gmem_sector_bytes * config_.gmem_sector_bytes;
        plan.sectors.push_back({sector, 0, static_cast<uint32_t>(address - sector)});
        break;
    }
    }
    plan.requests_total = static_cast<uint32_t>(plan.sectors.size());
    return plan;
}

// The command an entry leaving the queue becomes, with `lanes` (its
// sample's): unplanned, its number and slot still to be given.
Machine::LsuCommand Machine::lsu_command_from(const LsuEntry& entry, uint64_t lanes) const {
    const auto& inst = entry.instruction;
    LsuCommand command;
    command.warp = entry.warp;
    command.pc = entry.pc;
    command.instruction = inst;
    command.depbar = inst.control.set_depbar;
    // A stg.v8 reads its data in beats once planned: its read depbar rides
    // with the command to its last data read. Every other form's is lowered
    // when its command slot's read of its last register is granted.
    command.read_depbar = inst.control.read_depbar;
    command.lanes = lanes;
    switch (inst.opcode) {
    case Opcode::ldg_v8:
        command.kind = LsuKind::vector_load;
        command.reg = inst.dst;
        break;
    case Opcode::stg_v8:
        command.kind = LsuKind::vector_store;
        command.reg = inst.src1;
        break;
    case Opcode::ldg:
        command.kind = LsuKind::scalar_load;
        command.reg = inst.dst;
        break;
    case Opcode::stg:
        command.kind = LsuKind::scalar_store;
        command.reg = inst.src1;
        break;
    case Opcode::ldg_uni:
        command.kind = LsuKind::uniform_load;
        command.reg = inst.dst;
        command.lanes = full_mask();
        command.address_read = true; // its base is a uniform register, sampled in the queue
        break;
    default:
        throw SimError("internal non-LSU opcode in the LSU's queue");
    }
    return command;
}

// A store whose data has not all been moved into the egress FIFO (in Q). Such
// a store holds the store buffer or is waiting for it; stores take the buffer
// in command order. A store not yet planned has all its data left.
bool Machine::store_has_data_left(const LsuCommand& command) {
    if (!command.planned) return is_store(command.kind);
    return (command.kind == LsuKind::vector_store && command.sectors_pushed < command.requests_total) ||
        (command.kind == LsuKind::scalar_store && command.words_pushed < command.sectors.size());
}

// The lanes a command's buffer group holds (fixed lane groups, so a lane's
// place in the buffer is wiring): a v8 load's lsu.load_group_lanes, a v8
// store's lsu.store_group_lanes, a gather's as many lanes as the load buffer
// has words (all sixteen when it has that many); a uniform load is one word,
// a scalar store brings every lane with its data read.
uint32_t Machine::lsu_group_width(const LsuCommand& command) const {
    switch (command.kind) {
    case LsuKind::vector_load: return config_.lsu_load_group_lanes;
    case LsuKind::vector_store: return config_.lsu_store_group_lanes;
    case LsuKind::scalar_load: return std::min(config_.lanes, 8 * config_.lsu_load_group_lanes);
    default: return config_.lanes;
    }
}

// The ACTIVE lanes of a command's group `group`.
uint64_t Machine::lsu_group_lanes(const LsuCommand& command, uint32_t group) const {
    const auto width = lsu_group_width(command);
    uint64_t mask = 0;
    for (uint32_t lane = group * width; lane < std::min(config_.lanes, (group + 1) * width); ++lane) {
        mask |= uint64_t{1} << lane;
    }
    return mask & command.lanes;
}

// The store buffer is free for a scalar store leaving the queue: no admitted
// store still has data to move (a released one has none). No store can come
// between it and its data's landing: stores take the buffer in command order.
bool Machine::store_buffer_free() const {
    const auto commands = lsu_commands_.cur().oldest_first();
    return std::none_of(commands.begin(), commands.end(),
        [](const LsuCommand* command) { return store_has_data_left(*command); });
}

std::optional<uint32_t> Machine::free_id_tag() const {
    const auto& tags = free_id_tags_.cur();
    for (uint32_t tag = 0; tag < tags.size(); ++tag) {
        if (tags[tag] == 0) return tag;
    }
    return std::nullopt;
}

const Machine::LsuCommand* Machine::find_lsu_command(uint64_t id) const {
    return lsu_commands_.cur().find(id);
}

// The egress FIFO's request pointer: the request FIFO's head, the oldest
// complete slot whose write request has not been accepted. A slot is offered
// to the controller only when it is readable there: all its words are in AND
// the RAM's write-to-read latency has passed. head_requested_now: the
// channel accepts the head's request in this cycle, and the pointer is seen
// as that leaves it (the next entry). Only the LSU's request-wait count looks
// there: the channel takes one request per cycle.
const Machine::EgressRequest* Machine::egress_request_slot(bool head_requested_now) const {
    const std::size_t index = head_requested_now ? 1 : 0;
    if (egress_requests_.readable() <= index) return nullptr;
    return &egress_requests_.front(index);
}

// The request channel serves LSU commands in admission order: the oldest
// command with requests left is the only candidate, so one SM's gmem
// accesses reach the controller in program order. A command not yet planned
// has all its requests left, so it holds every younger one back. Its next
// request is ready when: load - always (requests run ahead of the load
// buffer: the data waits in DRAM, in the controller and in the ingress
// FIFO); store - the egress request pointer is its next slot, complete and
// visible (scatter: and a free-ID tag).
const Machine::LsuCommand* Machine::lsu_request_candidate(bool head_requested_now) const {
    for (const auto* oldest : lsu_commands_.cur().oldest_first()) {
        const auto& command = *oldest;
        if (!command.planned) return nullptr;
        if (command.requests_emitted == command.requests_total) continue;
        switch (command.kind) {
        case LsuKind::vector_load:
            return &command;
        case LsuKind::vector_store:
        case LsuKind::scalar_store: {
            const auto* slot = egress_request_slot(head_requested_now);
            if (slot != nullptr && slot->client == GmemClient::lsu && slot->owner == command.slot &&
                (command.kind == LsuKind::vector_store || free_id_tag().has_value())) {
                return &command;
            }
            return nullptr;
        }
        default:
            return &command;
        }
    }
    return nullptr;
}

Machine::LsuRequestWant Machine::lsu_request_want() const {
    LsuRequestWant want;
    if (const auto* command = lsu_request_candidate(false); command != nullptr) want.command = command->id;
    return want;
}

Machine::GmemSectorRequest Machine::write_request(
    const EgressRequest& slot, GmemClient client, uint64_t id, std::optional<uint32_t> tag) {
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

Machine::GmemSectorRequest Machine::lsu_read_request(const LsuCommand& command, uint64_t id) {
    GmemSectorRequest request;
    request.id = id;
    request.command = command.id;
    request.owner = GmemSectorRequest::Owner::lsu;
    request.load = true;
    request.beats = beats_per_sector(config_);
    request.ready_dram_tick = dram_tick_ + config_.gmem_latency_dram_ticks;
    const auto index = command.requests_emitted;
    request.stream = lsu_stream;
    request.sector_address = command.sectors.at(index).sector_address;
    return request;
}

// ---------------------------------------------------------------------------
// 3. Ingress
// ---------------------------------------------------------------------------

// The ingress FIFO read port: lsu.pluck_words_per_cycle words per cycle from
// the head, only readable words (their RAM latency has passed), each word to
// its stream's consumer. A word says which read stream it returns on; that
// stream's oldest read record says where it goes, and the stream's count of
// bytes consumed of that record's sector is the word's place in the sector
// (each stream returns in request order and a sector's beats in order: the
// controller's contract, which the sanitizer checks word by word). The words
// go to their consumer in pieces: one stream's words inside one bus beat of
// one sector. A consumer that cannot take its piece (the TMA sector register
// still holds an unplaced sector; the load buffer still holds a group
// waiting for the RF write port, the previous group of this load or an older
// load's) stalls the head: shared-FIFO head-of-line blocking. Gathers read
// the whole sector at this rate and keep one word. The destinations are seen
// as this cycle's drains leave them (same-cycle space reuse (d)): a register
// drained now takes new words now, visible next cycle; no word passes
// through.
Machine::IngressPluck Machine::decide_ingress(const Wires& w) {
    IngressPluck pluck;
    TmaSectorRegister tma_reg = tma_sector_.cur();
    if (w.shmem.tma_sector_released) {
        tma_reg.request.reset(); // its placement's last round read it now
        tma_reg.filled = 0;
        tma_reg.complete = false;
    }
    bool tma_filled = false;
    // Each stream's head as the read moves through it this cycle: its record
    // (counted from the record FIFO's head) and the bytes consumed of its sector.
    struct Head {
        std::size_t entry = 0;
        uint32_t received = 0;
    };
    Head tma{0, tma_read_received_.cur()};
    Head lsu{0, lsu_read_received_.cur()};
    const uint32_t sector_bytes = config_.gmem_sector_bytes;
    const uint32_t beat_bytes = config_.gmem_bus_bytes_per_dram_tick;
    const std::size_t width = std::min<std::size_t>(ingress_.readable(), config_.lsu_pluck_words_per_cycle);
    std::size_t index = 0;
    while (index < width) {
        const auto& first = ingress_.front(index);
        const bool on_tma = first.stream == tma_stream;
        auto& head = on_tma ? tma : lsu;
        IngressPiece piece;
        piece.stream = first.stream;
        piece.offset = head.received;
        piece.request = first.request;
        piece.sent_offset = first.offset;
        // To the end of the read port's width, of the bus beat, or of the sector.
        while (true) {
            piece.words.push_back(ingress_.front(index + piece.words.size()).word);
            const uint32_t end = piece.offset + piece.bytes();
            if (index + piece.words.size() == width || end % beat_bytes == 0 || end == sector_bytes ||
                ingress_.front(index + piece.words.size()).stream != first.stream) {
                break;
            }
        }
        if (on_tma) {
            if (tma_reg.request.has_value() && tma_reg.complete) break; // unplaced sector: the head waits
            if (tma.entry >= tma_reads_.readable()) throw SimError("internal ingress word without a read record");
            pluck_tma(piece, tma_reads_.front(tma.entry), tma_reg, pluck);
            tma_filled = true;
        } else {
            if (lsu.entry >= lsu_reads_.readable()) throw SimError("internal ingress word without a read record");
            const auto& record = lsu_reads_.front(lsu.entry);
            const bool taken = record.kind == LsuReadKind::vector ? pluck_vector(record, piece, pluck, w)
                : pick_scalar_word(record, piece, pluck, w);
            if (!taken) break;
            note_first_return(record.command);
        }
        probe(LsuEdge::ingress_read, piece.request, piece.sent_offset, piece.bytes());
        head.received += piece.bytes();
        if (head.received == sector_bytes) { // the record's sector is consumed: it leaves
            ++head.entry;
            head.received = 0;
        }
        index += piece.words.size();
    }
    pluck.words = static_cast<uint32_t>(index);
    const auto heads = [](const Head& head, uint32_t received_before) {
        return IngressPluck::StreamHead{static_cast<uint32_t>(head.entry), head.received,
            head.entry != 0 || head.received != received_before};
    };
    pluck.tma_reads = heads(tma, tma_read_received_.cur());
    pluck.lsu_reads = heads(lsu, lsu_read_received_.cur());
    if (tma_filled) pluck.tma_sector_after = std::move(tma_reg);
    return pluck;
}

// Ingress -> the TMA stream's 32 B sector register at the pluck rate. When
// the sector is complete it becomes a placement ticket; the register stays
// occupied until the placement's last port-B write round has read it.
void Machine::pluck_tma(const IngressPiece& piece, const TmaRun& run, TmaSectorRegister& reg, IngressPluck& pluck) {
    sanitizer_.check_ordered_beat(cycle_, tma_stream, piece.request, piece.sent_offset);
    if (!reg.request.has_value()) {
        reg.request = piece.request;
        reg.sector_address = run_sector_address(run);
        reg.filled = 0;
        reg.complete = false;
    }
    std::memcpy(reg.data.data() + reg.filled, piece.words.data(), piece.bytes());
    probe(LsuEdge::tma_reg_write, piece.request, reg.filled, piece.bytes());
    reg.filled += piece.bytes();
    sanitizer_.record_ordered_pluck(tma_stream, piece.bytes());
    if (reg.filled != config_.gmem_sector_bytes) return;

    TmaReturn returned;
    returned.run = run;
    returned.sector_request = piece.request;
    reg.complete = true;
    pluck.tma_ticket = std::move(returned);
}

// The load buffer's group whose LAST RF write is granted now: the buffer is
// free from now on (same-cycle reuse (d)).
bool Machine::load_buffer_released(const Wires& w) const {
    const auto& start = lsu_buffers_.at(load_buffer).cur();
    return w.rf_write.granted_to(AsyncWriter::lsu_buffer, load_buffer) && start.beats + 1 == start.beats_total;
}

// The load buffer as a word arriving for group `group` of `command` sees it
// this cycle: free (never owned, or its group's last RF write is granted
// now), or owned -- in Q or by a claim earlier in this cycle's pluck. A word
// may enter when the buffer is free (it claims it for its group) or already
// holds this group, unfinished. It waits at the FIFO head while the buffer
// holds another group (this load's previous one or an older load's), or this
// group complete. Returns whether the word may enter.
bool Machine::load_buffer_takes(const LsuCommand& command, uint32_t group, IngressPluck& pluck,
    const Wires& w) const {
    const auto& start = lsu_buffers_.at(load_buffer).cur();
    const bool released = load_buffer_released(w);
    if (pluck.load_claim.has_value()) {
        const auto& claim = *pluck.load_claim;
        return claim.slot == command.slot && claim.group == group && pluck.lanes.size() != claim.expected;
    }
    if (!start.owner.has_value() || released) {
        IngressPluck::LoadClaim claim;
        claim.slot = command.slot;
        claim.group = group;
        if (command.kind == LsuKind::uniform_load) {
            claim.lanes = full_mask();
            claim.expected = 1;
            claim.beats_total = 1;
        } else {
            claim.lanes = lsu_group_lanes(command, group);
            const auto lanes = static_cast<uint32_t>(std::popcount(claim.lanes));
            claim.expected = command.kind == LsuKind::vector_load ? 8 * lanes : lanes;
            claim.beats_total = command.kind == LsuKind::vector_load ? 8 : 1;
        }
        pluck.load_claim = claim;
        return true;
    }
    if (*start.owner != command.slot || start.group != group) return false;
    return start.valid_count + pluck.lanes.size() != start.expected;
}

// The live command a read record names: the one in its slot, which must be
// the one that made the read (a retired gather's tail never asks).
const Machine::LsuCommand& Machine::lsu_read_command(const LsuRead& record) const {
    const auto& command = lsu_commands_.cur().in_slot(record.slot);
    if (command.id != record.command) {
        throw SimError("internal: LSU command #" + std::to_string(record.command) + "'s read record found #" +
            std::to_string(command.id) + " in its slot");
    }
    return command;
}

// Ingress -> the load buffer at the pluck rate: a v8 load's sector is one
// lane's eight words, word w of lane l at buffer word (l - first lane of the
// group) x 8 + w. No sector register.
bool Machine::pluck_vector(const LsuRead& record, const IngressPiece& piece, IngressPluck& pluck, const Wires& w) {
    const auto& command = lsu_read_command(record);
    const auto width = lsu_group_width(command);
    const auto group = record.lane / width;
    // The order of the FIFO head is checked before asking whether the buffer
    // takes it: a sector returned out of request order may belong to a later
    // group, which the buffer would refuse for good while it waits for the
    // sector that was skipped (the check reads nothing forward: the count of
    // bytes consumed moves only when they are consumed).
    sanitizer_.check_ordered_beat(cycle_, lsu_stream, piece.request, piece.sent_offset);
    if (!load_buffer_takes(command, group, pluck, w)) return false;
    for (std::size_t k = 0; k < piece.words.size(); ++k) {
        const uint32_t index = (record.lane - group * width) * 8 + piece.offset / 4 + static_cast<uint32_t>(k);
        pluck.lanes.push_back({load_buffer, index, piece.words[k]});
    }
    probe(LsuEdge::buffer_write, command.id, piece.request, piece.bytes());
    sanitizer_.record_ordered_pluck(lsu_stream, piece.bytes());
    return true;
}

// Gather / ldg.uni: no sector register. The lane's word is taken off the
// FIFO read port into the load buffer (a gather: word l - first lane of its
// group; uniform: the one word, fanned out to every lane by the RF write);
// the rest of the sector is read at the same rate and dropped (those bytes
// need no buffer). The word waits at the FIFO head while the buffer holds
// another group. Returns false when the head must wait.
bool Machine::pick_scalar_word(const LsuRead& record, const IngressPiece& piece, IngressPluck& pluck,
    const Wires& w) {
    const uint32_t begin = piece.offset;
    // As in pluck_vector: the head's order first, then the buffer.
    sanitizer_.check_ordered_beat(cycle_, lsu_stream, piece.request, piece.sent_offset);
    if (record.word_offset >= begin && record.word_offset + 4u <= begin + piece.bytes()) {
        const auto& command = lsu_read_command(record);
        const bool uniform = record.kind == LsuReadKind::uniform;
        const auto width = lsu_group_width(command);
        const auto group = uniform ? 0 : record.lane / width;
        if (!load_buffer_takes(command, group, pluck, w)) return false;
        const uint32_t word = piece.words[(record.word_offset - begin) / 4];
        pluck.lanes.push_back({load_buffer, uniform ? 0 : record.lane - group * width, word});
        probe(LsuEdge::buffer_write, command.id, piece.request, 4);
    }
    sanitizer_.record_ordered_pluck(lsu_stream, piece.bytes());
    return true;
}

// (T) A gather or ldg.uni may already have retired while the tail of its
// last sector is still being read out of the FIFO and dropped (its read
// record lives until the sector's last byte: drive_read_records), so this
// tolerates retirement. It is given the command's number, not its slot: the
// slot may already hold the next command.
void Machine::note_first_return(uint64_t command) {
    const auto found = lsu_phases_.find(command);
    if (found == lsu_phases_.end()) return;
    if (!found->second.first_return_cycle.has_value()) found->second.first_return_cycle = cycle_;
}

// ---------------------------------------------------------------------------
// 4. RF read stage (vector stores). The RF write stage (load buffer -> RF) is a
// client of the RF write decision: decide_rf_writes in machine/rf.cpp.
// ---------------------------------------------------------------------------

// RF -> the store buffer (v8 stores): the LSU's async RF reader, one
// full-width read beat per cycle at most (decide_rf_reads grants it a read
// port the issue register leaves free). Stores use the one store buffer in
// command order, so the only candidate is the OLDEST store that still has
// data to move (in Q), when it is a v8 store. Its lanes go in fixed groups of
// lsu.store_group_lanes: for a group it reads rS .. rS+7 once, one beat a
// cycle (the beats follow each other through the RF read stages), and keeps
// the group's active lanes -- beat j's lane l lands in buffer word (l - first
// lane of the group) x 8 + j. The group's first beat needs the buffer free:
// empty in Q, or its previous data's last word pushed into the egress FIFO
// now (d). A stg.v8 reads its data once it is planned (in Q).
// `egress`: this cycle's egress decision, for that drain.
std::optional<Machine::RfReadRequest> Machine::lsu_store_read_request(const EgressWrite& egress) const {
    for (const auto* command : lsu_commands_.cur().oldest_first()) {
        if (!store_has_data_left(*command)) continue;
        if (command->kind != LsuKind::vector_store || !command->planned) return std::nullopt;
        const bool drained_now = egress.lsu_group_drained && egress.lsu_command == command->id;
        const bool buffer_drained_now = egress.lsu_store_drained;
        const uint32_t sector = command->sectors_pushed + (drained_now ? 1 : 0);
        if (sector == command->requests_total) return std::nullopt; // its last group has just left
        const uint32_t beat = drained_now ? 0 : command->group_beats_read;
        if (beat == 8) return std::nullopt; // the group's eight beats are read; it is landing or draining
        if (beat == 0 && !drained_now && !buffer_drained_now &&
            lsu_buffers_.at(store_buffer).cur().owner.has_value()) {
            return std::nullopt;
        }
        const auto width = config_.lsu_store_group_lanes;
        RfReadRequest request{RfReader::lsu_store, 0, command->warp,
            static_cast<uint8_t>(command->reg + beat), command->id, store_buffer, command->slot};
        request.group = command->sectors.at(sector).lane / width;
        request.beat = beat;
        request.lanes = lsu_group_lanes(*command, request.group);
        // The command's last group: no active lane above it.
        request.last_group = (request.group + 1) * width >= config_.lanes ||
            (command->lanes >> ((request.group + 1) * width)) == 0;
        request.read_depbar = command->read_depbar;
        return request;
    }
    return std::nullopt;
}

// ---------------------------------------------------------------------------
// 5. Egress write port
// ---------------------------------------------------------------------------

// One shared write port of lsu.pluck_words_per_cycle words into the egress
// FIFO, one client per cycle, granted per 32 B slot: a started slot is
// finished by its client (never preempted); at each new grant the oldest TMA
// store sector register (the ring's head), once complete, wins over the LSU.
// The LSU's only candidate is its oldest store with data left (egress is in
// command order). A slot holds one
// sector: a TMA sector (8 words, its byte enables), one lane of a stg.v8
// (8 words), or one scatter lane (1 word, 4 byte enables, a whole slot).
// Sources are read through Q: written in an earlier cycle.
Machine::EgressWrite Machine::decide_egress_write() {
    EgressWrite write;
    const uint32_t sector_words = config_.gmem_sector_bytes / 4;
    const EgressOpenSlot* open = egress_open_.cur().has_value() ? &*egress_open_.cur() : nullptr;
    const auto& tma_head = tma_store_registers_.at(tma_store_head_.cur()).cur();
    if (open == nullptr) {
        const bool tma_ready = tma_head.has_value() && tma_head->complete();
        const LsuCommand* lsu = nullptr;
        for (const auto* command : lsu_commands_.cur().oldest_first()) {
            if (store_has_data_left(*command)) {
                lsu = command;
                break;
            }
        }
        if (lsu != nullptr) {
            const auto& buffer = lsu_buffers_.at(store_buffer).cur();
            if (!lsu->planned || buffer.owner != lsu->slot || buffer.valid_count != buffer.expected) {
                // Not planned yet, or its data not all in: a v8 group's eight
                // RF read beats, a scalar store's one data read.
                lsu = nullptr;
            }
        }
        if (!tma_ready && lsu == nullptr) return write;
        // Every slot in use (no slot is open now): the data FIFO is full.
        if (egress_data_.count() == config_.lsu_egress_sectors) {
            if (!tma_ready) {
                ++trace_->lsu_counters().lsu_egress_wait_cycles;
                ++lsu_phases_[lsu->id].egress_wait;
            }
            return write;
        }
        EgressOpenSlot slot;
        slot.id = next_egress_slot_id_++;
        if (tma_ready) { // the register's run gives its sector and byte enables
            slot.client = GmemClient::tma;
            slot.command = tma_head->run.command;
            slot.sector_address = run_sector_address(tma_head->run);
            slot.byte_enable = run_byte_enable(tma_head->run);
            slot.words_total = sector_words;
        } else if (lsu->kind == LsuKind::vector_store) {
            // The next active lane's sector, in lane order.
            slot.client = GmemClient::lsu;
            slot.owner = lsu->slot;
            slot.command = lsu->id;
            slot.index = lsu->sectors_pushed;
            slot.sector_address = lsu->sectors.at(lsu->sectors_pushed).sector_address;
            slot.byte_enable = full_byte_enable(config_.gmem_sector_bytes);
            slot.words_total = sector_words;
        } else {
            const auto& sector = lsu->sectors.at(lsu->words_pushed);
            slot.client = GmemClient::lsu;
            slot.owner = lsu->slot;
            slot.command = lsu->id;
            slot.index = lsu->words_pushed;
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
        // The open TMA slot is the head register's: it opened with it and is
        // never preempted, and the head moves only when its last word leaves.
        if (!tma_head.has_value() || !tma_head->complete()) {
            throw SimError("internal: TMA store sector register drained before it was complete");
        }
        const auto& reg = *tma_head;
        for (uint32_t word = 0; word < words; ++word) {
            const uint32_t byte = (reg.words_out + word) * 4;
            uint32_t value = 0;
            std::memcpy(&value, reg.data.data() + byte, 4);
            write.words.emplace_back(byte, value);
        }
        write.tma_store_drained = reg.words_out + words == sector_words; // a source read may reserve it now (d)
    } else {
        const auto* command = &lsu_commands_.cur().in_slot(open->owner);
        const auto& buffer = lsu_buffers_.at(store_buffer).cur();
        if (buffer.owner != command->slot) throw SimError("internal: the egress port read another store's buffer");
        write.lsu_command = command->id;
        for (uint32_t word = 0; word < words; ++word) {
            const uint32_t index = command->words_pushed + word;
            if (command->kind == LsuKind::vector_store) {
                // Word `index` of the current lane's sector.
                const auto width = config_.lsu_store_group_lanes;
                const auto& sector = command->sectors.at(command->sectors_pushed);
                const uint32_t at = (sector.lane - buffer.group * width) * 8 + index;
                sanitizer_.check_buffer_lanes_valid(cycle_, store_buffer, buffer.valid, at, 1,
                    "egress write port");
                write.words.emplace_back(index * 4, buffer.words[at]);
                if (index + 1 == sector_words) {
                    // The lane's sector is complete; the group has left the
                    // buffer when this was its last active lane (d).
                    const auto next = command->sectors_pushed + 1;
                    if (next == command->requests_total ||
                        command->sectors.at(next).lane / width != buffer.group) {
                        write.lsu_group_drained = true;
                    }
                }
            } else {
                // A scatter's sectors are its active lanes', in lane order.
                const auto& sector = command->sectors.at(index);
                sanitizer_.check_buffer_lanes_valid(cycle_, store_buffer, buffer.valid, sector.lane, 1,
                    "egress write port");
                write.words.emplace_back(sector.word_offset, buffer.words[sector.lane]);
                // Its last word leaves: the store buffer is free from the
                // next cycle (the command still has its requests to send).
                if (index + 1 == command->sectors.size()) write.lsu_store_drained = true;
            }
        }
    }
    write.slot_complete = open->words_written + words == open->words_total;
    probe(LsuEdge::egress_write, open->id, open->client == GmemClient::tma ? 0 : 1, words * 4);
    return write;
}

std::string Machine::lsu_phase_detail(uint64_t command, bool store) const {
    LsuCommandPhases phases;
    if (const auto found = lsu_phases_.find(command); found != lsu_phases_.end()) phases = found->second;
    std::ostringstream out;
    out << "admit=" << phases.admit_cycle
        << " requests_done=" << (phases.requests_done_cycle ? std::to_string(*phases.requests_done_cycle) : "-")
        << " first_return=" << (phases.first_return_cycle ? std::to_string(*phases.first_return_cycle) : "-")
        << (store ? " last_ack=" : " last_rf_write=") << cycle_
        << " request_wait=" << phases.request_wait
        << " rf_wait=" << phases.rf_wait
        << " egress_wait=" << phases.egress_wait;
    return out.str();
}

// An LSU command leaves its slot, which is free for a new owner this same
// cycle (d). A load ends here (its last register write is granted). A store
// only hands over: all its sector requests are accepted, and it ends when its
// record's last acknowledgement completes it (complete_stores).
void Machine::release_lsu_command(Wires& w, uint64_t id) {
    const auto* command = find_lsu_command(id);
    if (command == nullptr) throw SimError("internal release of an unknown LSU command");
    if (!is_store(command->kind)) {
        trace_->lsu_command_end(cycle_, id, lsu_phase_detail(id, false));
        lsu_phases_.erase(id);
    }
    w.released_lsu_commands.push_back(id);
}

// Stores whose last sector request was accepted last cycle (in Q: every
// request emitted) give up their command slots now. From Q only, before issue.
void Machine::release_requested_stores(Wires& w) {
    for (const auto* command : lsu_commands_.cur().oldest_first()) {
        if (is_store(command->kind) && command->planned && command->requests_emitted == command->requests_total) {
            release_lsu_command(w, command->id);
        }
    }
}

// Stores whose last acknowledgement came last cycle (in Q: a record with none
// owed) complete now: the depbar counts down at the RF's RAM stage.
void Machine::complete_stores(Wires& w) {
    for (const auto& record : store_acks_.cur()) {
        if (!record.valid || record.acks_owed != 0) continue;
        record_depbar_completion(w, record.warp, record.depbar, "depbar underflow on sb" + std::to_string(record.depbar));
        trace_->lsu_command_end(cycle_, record.command, lsu_phase_detail(record.command, true));
        lsu_phases_.erase(record.command);
    }
}

// The acknowledgement record a store planned now takes: the lowest one free
// in Q. There is always one (store_acks_ is sized for every place a store
// can be; Machine::Machine).
uint32_t Machine::free_store_ack_record() const {
    const auto& records = store_acks_.cur();
    for (uint32_t index = 0; index < records.size(); ++index) {
        if (!records[index].valid) return index;
    }
    throw SimError("internal: a store planned with every acknowledgement record in use (" +
        std::to_string(records.size()) + ") at cycle " + std::to_string(cycle_));
}

// The acknowledgement records, each in one calculation: a store that
// completed now leaves; this cycle's acknowledgements count down; a store
// planned now takes the record its plan chose (free in Q).
void Machine::drive_store_acks(const Wires& w) {
    const auto& records = store_acks_.cur();
    const LsuCommand* planned = nullptr;
    if (w.lsu.plan.has_value()) {
        const auto& command = lsu_commands_.cur().at(w.lsu.plan->command);
        if (is_store(command.kind)) planned = &command;
    }
    for (uint32_t index = 0; index < records.size(); ++index) {
        const auto& cur = records[index];
        StoreAckRecord next = cur;
        if (cur.valid && cur.acks_owed == 0) {
            next = StoreAckRecord{};
        } else if (const auto acks = w.controller.store_acks.find(index); acks != w.controller.store_acks.end()) {
            next.acks_owed -= acks->second; // (checked when each was delivered)
        }
        if (planned != nullptr && w.lsu.plan->ack_record == index) {
            if (cur.valid) throw SimError("internal: a store took an acknowledgement record in use");
            next.valid = true;
            next.warp = planned->warp;
            next.depbar = planned->depbar;
            next.acks_owed = w.lsu.plan->requests_total;
            next.command = planned->id;
        }
        if (next != cur) store_acks_.set_next(index, next, "drive_store_acks");
    }
}

// ---------------------------------------------------------------------------
// The front end: the queue and the command slots' reads and plans
// ---------------------------------------------------------------------------

// The queue this cycle, from Q (after the RF write port: a command slot its
// grant releases now is free for the queue's head now (d)): the newest
// entry's issue + 1 sample, the head's move into a command slot, and the plan
// of the command whose address is at the RF output now.
void Machine::step_lsu_queue(Wires& w) {
    sample_lsu_entry(w);
    advance_lsu_queue(w);
    plan_arriving_address(w);
}

// The entry that entered the queue at the last edge samples, in the issue
// register's cycle (its issue + 1): its predicate, read from the predicate
// file (lanes = the active mask AND the write mask), and an ldg.uni its base
// through a URF read port (step_rf_reads counts it with the cycle's other
// URF reads). The uniform operation's partial-mask rule is the sanitizer's
// PANIC here. An entry with no lane left leaves the queue now (drop_lsu_entry).
void Machine::sample_lsu_entry(Wires& w) {
    const auto& queue = lsu_queue_.cur();
    if (queue.empty() || queue.back().sampled) return;
    const auto& entry = queue.back();
    const auto& inst = entry.instruction;
    LsuFrontEnd::Sample sample;
    sample.lanes = entry.lanes;
    if (has_write_mask(inst)) sample.lanes &= write_mask_lanes(warps_[entry.warp], inst);
    if (inst.opcode == Opcode::ldg_uni) {
        sanitizer_.check_uniform_mask(site(entry), entry.lanes, full_mask());
        sample.base = read_urf(entry.warp, inst.src0, "URF read: a queued ldg.uni's sample");
        sample.uniform_read = true;
    }
    sample.dropped = sample.lanes == 0;
    if (sample.dropped) drop_lsu_entry(entry, w);
    w.lsu.sample = sample;
}

// An ldg / stg / ldg.v8 / stg.v8 whose write mask turned every active lane
// off: no register read, no command slot, no request. Its depbar and its
// read depbar are decremented now (a waiting instruction sees them zero from
// the next cycle, issue + 2); a load's destination keeps every lane's old
// value and is not in flight from then on; its source registers are no
// longer read; its queue entry is given back at this edge (drive_lsu_queue).
// Only a write mask can leave no lane: the divergence stack's active mask is
// never empty, and an ldg.uni refuses to be masked.
void Machine::drop_lsu_entry(const LsuEntry& entry, Wires& w) {
    const auto& inst = entry.instruction;
    if (!can_be_masked(inst.opcode)) {
        throw SimError("internal: " + opcode_name(inst.opcode) + " was queued with no active lane");
    }
    record_depbar_completion(w, entry.warp, inst.control.set_depbar,
        "depbar underflow on sb" + std::to_string(inst.control.set_depbar));
    lower_read_depbar(w, entry.warp, inst);
    if (const auto dst = destination_register(inst); dst.has_value()) {
        const uint32_t registers = inst.opcode == Opcode::ldg_v8 ? 8 : 1;
        for (uint32_t j = 0; j < registers; ++j) sanitizer_.record_reg_ready(entry.warp, *dst + j, cycle_ + 1);
    }
    sanitizer_.record_async_reader_done(entry.warp, inst.src0, AsyncReader::lsu);
    if (inst.opcode == Opcode::stg && inst.src1 != inst.src0) {
        sanitizer_.record_async_reader_done(entry.warp, inst.src1, AsyncReader::lsu);
    }
    if (inst.opcode == Opcode::stg_v8) {
        for (uint32_t j = 0; j < 8; ++j) sanitizer_.record_async_reader_done(entry.warp, inst.src1 + j, AsyncReader::stg_v8);
    }
}

// The queue's head leaves for a command slot when it may: sampled (or
// sampled now and not dropped), its wait over (an ldg.uni's base at the URF
// output), a command slot free as this cycle's releases leave them, and for
// a scalar stg the store buffer free (Q). It takes the lowest free slot
// (drive_lsu_commands puts it there). An ldg.uni is planned as it leaves:
// its one address needs no register read.
void Machine::advance_lsu_queue(Wires& w) {
    const auto& queue = lsu_queue_.cur();
    if (queue.empty()) return;
    const auto& head = queue.front();
    uint64_t lanes = head.lanes;
    uint32_t base = head.base;
    if (!head.sampled) { // the head is the newest entry: its sample is now
        const auto& sample = w.lsu.sample.value();
        if (sample.dropped) return;
        lanes = sample.lanes;
        base = sample.base;
    }
    if (head.wait != 0) return;
    auto& counters = trace_->lsu_counters();
    const auto slot = lsu_commands_.cur().lowest_free(config_.lsu_active_commands, w.released_lsu_commands);
    if (!slot.has_value()) {
        ++counters.lsu_no_slot_cycles;
        return;
    }
    const auto& inst = head.instruction;
    if (inst.opcode == Opcode::stg && !store_buffer_free()) {
        ++counters.lsu_store_buffer_wait_cycles;
        return;
    }
    auto command = lsu_command_from(head, lanes);
    command.id = lsu_command_number_.cur();
    command.slot = *slot;
    if (command.kind == LsuKind::uniform_load) {
        LaneWords bases(1, base);
        const auto plan = plan_lsu_command(command, bases);
        command.sectors = plan.sectors;
        command.requests_total = plan.requests_total;
        command.planned = true;
        ++counters.lsu_uniform_loads;
        counters.lsu_useful_bytes += 4;
    }
    trace_->lsu_command_begin(cycle_, command.id, command.warp, opcode_name(inst.opcode));
    lsu_phases_[command.id].admit_cycle = cycle_;
    // Sanitizer: the admitted set never exceeds the command slots.
    sanitizer_.check_lsu_admitted(lsu_commands_.cur().size() - w.released_lsu_commands.size() + 1);
    w.lsu.queue_pop = true;
    w.lsu.entering = std::move(command);
}

// The command slots' reads at the RF output now (the read stages' last
// stage, Q): an address is planned now (its command is planned from the next
// cycle); a scalar store's data goes into the store buffer.
void Machine::plan_arriving_address(Wires& w) {
    const auto& arriving = lsu_collect_pipe_.last();
    if (!arriving.has_value()) return;
    const auto& command = lsu_commands_.cur().in_slot(arriving->slot, arriving->command);
    if (arriving->address.has_value()) {
        auto plan = plan_lsu_command(command, *arriving->address);
        if (is_store(command.kind)) plan.ack_record = free_store_ack_record();
        auto& counters = trace_->lsu_counters();
        switch (command.kind) {
        case LsuKind::vector_load:
            ++counters.lsu_vector_loads;
            counters.lsu_useful_bytes += 32ull * plan.sectors.size(); // its active lanes' sectors
            break;
        case LsuKind::vector_store:
            ++counters.lsu_vector_stores;
            counters.lsu_useful_bytes += 32ull * plan.sectors.size();
            break;
        case LsuKind::scalar_load:
            ++counters.lsu_scalar_loads;
            counters.lsu_useful_bytes += 4ull * plan.sectors.size(); // its active lanes
            break;
        case LsuKind::scalar_store:
            ++counters.lsu_scalar_stores;
            counters.lsu_useful_bytes += 4ull * plan.sectors.size(); // its active lanes
            break;
        case LsuKind::uniform_load:
            throw SimError("internal: an ldg.uni's address read through the RF");
        }
        w.lsu.plan = std::move(plan);
    }
    if (arriving->data.has_value()) w.lsu.store_data = std::make_pair(arriving->slot, *arriving->data);
}

// The command slots' next RF read, from Q: the oldest command with a register
// still unread (its address; a scalar store's address and data, the data
// read with the address in one cycle when its bank has a port left, else
// next as this requester's own read; one register when they are the same).
// `data`: that pair's second register.
std::optional<Machine::RfReadRequest> Machine::lsu_collect_read_request(bool data) const {
    for (const auto* command : lsu_commands_.cur().oldest_first()) {
        const bool pair = command->kind == LsuKind::scalar_store;
        const bool data_unread = pair && !command->data_read;
        if (command->address_read && !data_unread) continue; // its reads are granted
        const auto& inst = command->instruction;
        RfReadRequest request;
        request.reader = data ? RfReader::lsu_collect_data : RfReader::lsu_collect;
        request.warp = command->warp;
        request.lsu_command = command->id;
        request.lsu_slot = command->slot;
        if (data) {
            if (!pair || command->address_read || inst.src1 == inst.src0) return std::nullopt;
            request.reg = inst.src1;
            return request;
        }
        request.reg = command->address_read ? inst.src1 : inst.src0;
        return request;
    }
    return std::nullopt;
}

// The LSU's queue, in one calculation: its head leaves for a command slot
// (popped) or, the newest entry, it has no lane left (dropped); the newest
// entry takes its issue + 1 sample; every other entry's `wait` counts down;
// issue's new entry enters at the tail.
void Machine::drive_lsu_queue(const Wires& w) {
    const bool push = w.issue.lsu.has_value();
    const auto& sample = w.lsu.sample;
    bool changed = w.lsu.queue_pop || push || sample.has_value();
    auto next = lsu_queue_.cur();
    for (auto& entry : next) {
        if (entry.wait != 0) {
            --entry.wait;
            changed = true;
        }
    }
    if (sample.has_value()) {
        if (sample->dropped) {
            next.resize(next.size() - 1); // the newest entry; also the head when it alone is queued
        } else {
            auto& newest = next.back();
            newest.lanes = sample->lanes;
            newest.base = sample->base;
            newest.sampled = true;
        }
    }
    if (w.lsu.queue_pop) next.pop_front();
    if (push) next.push_back(*w.issue.lsu);
    if (!changed) return;
    if (next.size() > config_.lsu_queue_entries) throw SimError("internal: the LSU queue holds more than lsu.queue_entries");
    lsu_queue_.set_next(std::move(next), "drive_lsu_queue");
}

// The command slots' reads in flight: the one sampled now enters the RF's
// read stages; the one at the RF output leaves at this edge (it was planned
// or filled the store buffer now).
void Machine::drive_lsu_collect_pipe(const Wires& w) {
    if (w.lsu.read.has_value()) lsu_collect_pipe_.push(*w.lsu.read, "drive_lsu_collect_pipe");
}

// The LSU command slots (admission order): every stage's progress, the
// releases, and the queue's head entering, in one calculation. A command
// released now frees its slot for the head now (d).
void Machine::drive_lsu_commands(const Wires& w) {
    // A load group whose last RF write is granted now: its lanes are done.
    std::optional<std::pair<uint64_t, uint32_t>> drained; // (command, lanes)
    if (load_buffer_released(w)) {
        const auto& buffer = lsu_buffers_[load_buffer].cur();
        const auto& owner = lsu_commands_.cur().in_slot(*buffer.owner);
        drained = std::make_pair(owner.id, owner.kind == LsuKind::uniform_load
            ? 1u : static_cast<uint32_t>(std::popcount(buffer.lanes)));
    }
    const bool changed = drained.has_value() || w.egress.client == GmemClient::lsu ||
        w.channel.granted == GmemClient::lsu || w.lsu_rf_read.command.has_value() ||
        !w.released_lsu_commands.empty() || w.lsu.entering.has_value() || w.lsu.plan.has_value() ||
        w.lsu.read.has_value();
    if (!changed) return;
    auto next = lsu_commands_.cur();
    const auto command = [&](uint64_t id) -> LsuCommand& {
        if (auto* found = next.find(id); found != nullptr) return *found;
        throw SimError("internal reference to a retired LSU command #" + std::to_string(id));
    };
    if (drained.has_value()) command(drained->first).lanes_written += drained->second;
    if (w.egress.client == GmemClient::lsu) {
        auto& store = command(w.egress.lsu_command);
        store.words_pushed += static_cast<uint32_t>(w.egress.words.size());
        if (store.kind == LsuKind::vector_store && w.egress.slot_complete) {
            store.words_pushed = 0; // the lane's sector is in the egress FIFO
            ++store.sectors_pushed;
        }
        // The group has left the store buffer: the next group's RF reads start.
        if (w.egress.lsu_group_drained) store.group_beats_read = 0;
    }
    if (w.channel.granted == GmemClient::lsu) ++command(w.channel.lsu_command).requests_emitted;
    // A command planned now: its requests; a command slot's read granted now.
    if (w.lsu.plan.has_value()) {
        auto& planned = command(w.lsu.plan->command);
        planned.sectors = w.lsu.plan->sectors;
        planned.requests_total = w.lsu.plan->requests_total;
        planned.ack_record = w.lsu.plan->ack_record;
        planned.planned = true;
    }
    if (const auto& read = w.lsu.read; read.has_value()) {
        auto& reading = next.in_slot(read->slot, read->command);
        if (read->address.has_value()) reading.address_read = true;
        if (read->data.has_value()) reading.data_read = true;
    }
    for (const auto released : w.released_lsu_commands) next.release(released);
    // The queue's head takes the slot it chose: the lowest free one.
    if (w.lsu.entering.has_value() &&
        next.take(*w.lsu.entering, config_.lsu_active_commands) != w.lsu.entering->slot) {
        throw SimError("internal: the LSU holds more than lsu.active_commands commands");
    }
    if (w.lsu.entering.has_value()) lsu_command_number_.set_next(w.lsu.entering->id + 1, "drive_lsu_commands");
    if (w.lsu_rf_read.command.has_value()) {
        auto& reading = next.slots.at(*w.lsu_rf_read.command);
        if (!reading.has_value()) throw SimError("internal: a store-data read for an empty LSU command slot");
        ++reading->group_beats_read;
    }
    if (next.size() > config_.lsu_active_commands) {
        throw SimError("internal: the LSU holds more than lsu.active_commands commands");
    }
    lsu_commands_.set_next(std::move(next), "drive_lsu_commands");
}

// The LSU's two buffers, each in one calculation. A buffer has an owner only
// while data is in it (machine.hpp, LsuBuffer).
//   load buffer   1. one of its group's RF writes is granted now: the beat
//                    counter moves; the group's LAST write empties it, no
//                    owner;
//                 2. the first word plucked for a group claims it;
//                 3. this cycle's plucked words.
//   store buffer  1. its data's last word is pushed into the egress FIFO now
//                    (a v8 store's lane group, a scalar store's vector):
//                    empty, no owner;
//                 2. a scalar store's data at the RF output now (its
//                    command slot's read);
//                 3. a v8 store's RF read beat at the RF output now: the
//                    group's lanes of that register.
// The emptying comes first, so a buffer reused in the cycle it empties (d)
// keeps everything the new data put in. A fill is the next value; it is read
// from the next cycle on.
void Machine::drive_lsu_buffers(const Wires& w) {
    const auto clear = [&](LsuBuffer& reg) {
        std::fill(reg.valid.begin(), reg.valid.end(), 0);
        reg.valid_count = 0;
        reg.expected = 0;
        reg.beats = 0;
        reg.beats_total = 0;
        reg.lanes = 0;
        reg.group = 0;
        reg.owner.reset();
    };
    const auto put = [&](LsuBuffer& reg, uint32_t at, uint32_t word) {
        reg.words.at(at) = word;
        if (reg.valid.at(at) != 0) throw SimError("internal: an LSU buffer word was written twice");
        reg.valid[at] = 1;
        ++reg.valid_count;
    };
    {
        const auto& cur = lsu_buffers_[load_buffer].cur();
        std::optional<LsuBuffer> next;
        const auto edit = [&]() -> LsuBuffer& {
            if (!next.has_value()) next = cur;
            return *next;
        };
        if (w.rf_write.granted_to(AsyncWriter::lsu_buffer, load_buffer)) {
            if (load_buffer_released(w)) clear(edit());
            else ++edit().beats;
        }
        if (w.ingress.load_claim.has_value()) {
            auto& reg = edit();
            if (reg.owner.has_value()) throw SimError("internal: a load claimed the owned load buffer");
            const auto& claim = *w.ingress.load_claim;
            clear(reg);
            reg.owner = claim.slot;
            reg.group = claim.group;
            reg.lanes = claim.lanes;
            reg.expected = claim.expected;
            reg.beats_total = claim.beats_total;
        }
        for (const auto& write : w.ingress.lanes) {
            auto& reg = edit();
            if (!reg.owner.has_value()) throw SimError("internal: a word plucked into the unowned load buffer");
            put(reg, write.lane, write.word);
        }
        if (next.has_value()) lsu_buffers_[load_buffer].set_next(std::move(*next), "drive_lsu_buffers");
    }
    {
        const auto& cur = lsu_buffers_[store_buffer].cur();
        std::optional<LsuBuffer> next;
        const auto edit = [&]() -> LsuBuffer& {
            if (!next.has_value()) next = cur;
            return *next;
        };
        if (w.egress.lsu_group_drained || w.egress.lsu_store_drained) clear(edit());
        if (w.lsu.store_data.has_value()) {
            // A scatter: every lane's word (the inactive ones are never sent).
            const auto& [slot, data] = *w.lsu.store_data;
            auto& reg = edit();
            if (reg.owner.has_value()) throw SimError("internal: store data into the owned store buffer");
            reg.owner = slot;
            reg.lanes = lsu_commands_.cur().in_slot(slot).lanes;
            for (uint32_t lane = 0; lane < config_.lanes; ++lane) put(reg, lane, data[lane]);
            reg.expected = config_.lanes;
        }
        if (const auto& arriving = lsu_read_pipe_.last(); arriving.has_value()) {
            // The store-data beat at the RF output now (sampled Lr ago): the
            // group's active lanes of register rS + beat.
            const auto& beat = *arriving;
            auto& reg = edit();
            const auto lanes = static_cast<uint32_t>(std::popcount(beat.lanes));
            if (!reg.owner.has_value()) {
                reg.owner = *beat.command;
                reg.group = beat.group;
                reg.lanes = beat.lanes;
                reg.expected = 8 * lanes;
                reg.beats_total = 8;
            } else if (reg.owner != beat.command || reg.group != beat.group) {
                throw SimError("internal: a store beat landed in another group's store buffer");
            }
            for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
                if ((beat.lanes >> lane & 1) == 0) continue;
                put(reg, (lane - beat.group * config_.lsu_store_group_lanes) * 8 + beat.beat, beat.values[lane]);
            }
            ++reg.beats;
        }
        if (next.has_value()) lsu_buffers_[store_buffer].set_next(std::move(*next), "drive_lsu_buffers");
    }
}

// The ingress FIFO: the read port's words leave the head; the bus's beats
// enter at the tail, word by word, readable lsu.ingress_latency_cycles on.
void Machine::drive_ingress(const Wires& w) {
    if (w.ingress.words != 0) ingress_.pop(w.ingress.words, "drive_ingress");
    if (w.controller.ingress_beats.empty()) return;
    auto& words = ingress_.write_port("drive_ingress");
    for (const auto& beat : w.controller.ingress_beats) {
        for (uint32_t byte = 0; byte < beat.bytes.size(); byte += 4) {
            IngressWord word;
            std::memcpy(&word.word, beat.bytes.data() + byte, 4);
            word.stream = beat.stream;
            word.request = beat.request;
            word.offset = beat.offset + byte;
            words.push_back(word);
        }
    }
}

// The egress FIFO: the request FIFO's head leaves when the channel accepts
// its write request, the data FIFO's head when its last beat has left; the
// write port's words go into the open slot (a new grant opens it), and its
// last word pushes it, complete, into both FIFOs (the write pointer moves on).
void Machine::drive_egress(const Wires& w) {
    if (w.channel.egress_requested) egress_requests_.pop(1, "drive_egress");
    if (w.controller.egress_slots_freed != 0) egress_data_.pop(w.controller.egress_slots_freed, "drive_egress");
    if (!w.egress.client.has_value()) return;
    auto open = w.egress.new_slot.has_value() ? *w.egress.new_slot : egress_open_.cur().value();
    for (const auto& [offset, word] : w.egress.words) std::memcpy(open.data.data() + offset, &word, 4);
    open.words_written += static_cast<uint32_t>(w.egress.words.size());
    if (!w.egress.slot_complete) {
        egress_open_.set_next(open, "drive_egress");
        return;
    }
    egress_requests_.push(static_cast<const EgressRequest&>(open), "drive_egress");
    EgressSector sector;
    sector.data = open.data;
    sector.id = open.id;
    egress_data_.push(sector, "drive_egress");
    if (egress_open_.cur().has_value()) egress_open_.set_next(std::nullopt, "drive_egress");
}

// The read records, per stream: a read accepted now enters its stream's FIFO
// (it cannot return now); the heads whose sector's last byte was consumed now
// leave, and the count of the bytes consumed of the head moves on.
// The LSU stream's size: requests run ahead of the load buffer, so each
// command slot may have all its sectors in flight (one per lane). One more
// entry is the tail of a retired gather or ldg.uni: such a load writes the
// RF, and frees its command slot, once its last word is in, while the rest of
// that word's sector is still coming off the bus and being read out of the
// ingress FIFO and dropped; its entry lives until the sector's last byte.
// Only one such tail can exist: the stream is ordered and the FIFO is read in
// order, so the next load to retire has its last word behind that tail and
// retires after it is gone. Hardware: lsu.active_commands x lanes + 1
// entries (the FIFO's capacity; a push past it is an internal error).
void Machine::drive_read_records(const Wires& w) {
    if (w.channel.new_tma_read.has_value()) tma_reads_.push(*w.channel.new_tma_read, "drive_read_records");
    if (w.channel.new_lsu_read.has_value()) lsu_reads_.push(*w.channel.new_lsu_read, "drive_read_records");
    const auto& tma = w.ingress.tma_reads;
    if (tma.finished != 0) tma_reads_.pop(tma.finished, "drive_read_records");
    if (tma.changed) tma_read_received_.set_next(tma.received, "drive_read_records");
    const auto& lsu = w.ingress.lsu_reads;
    if (lsu.finished != 0) lsu_reads_.pop(lsu.finished, "drive_read_records");
    if (lsu.changed) lsu_read_received_.set_next(lsu.received, "drive_read_records");
}

// Free-ID tags (scatter writes only): returned when a scatter's write is
// acknowledged; then taken by a scatter's request now (a tag free in Q, so
// today never one returned now).
void Machine::drive_free_id_tags(const Wires& w) {
    if (!w.channel.lsu_tag.has_value() && w.controller.freed_tags.empty()) return;
    // (A tag is taken only when free in Q, so no tag is both freed and taken now.)
    for (const auto tag : w.controller.freed_tags) free_id_tags_.set_next(tag, 0, "drive_free_id_tags");
    if (w.channel.lsu_tag.has_value()) free_id_tags_.set_next(*w.channel.lsu_tag, 1, "drive_free_id_tags");
}

// The LSU's store-data reads in flight: the one sampled now enters the RF's
// read stages; the one at the RF output leaves at this edge (it fills the
// store buffer, drive_lsu_buffers).
void Machine::drive_lsu_read_pipe(const Wires& w) {
    if (w.lsu_rf_read.command.has_value()) lsu_read_pipe_.push(w.lsu_rf_read, "drive_lsu_read_pipe");
}

} // namespace nanosm

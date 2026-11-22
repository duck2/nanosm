// The load / store unit: the register file's path to global memory. The
// memory front end, read top to bottom:
//
//   1. controller (src/machine/gmem.cpp): one request channel (TMA and LSU
//      alternate; one select), a FIFO queue of gmem.queue_sectors entries, one
//      bus beat per DRAM tick (reads into the ingress FIFO, writes out of the
//      egress FIFO), one write contract
//   2. LSU admission (at issue) and request generation (one per core cycle;
//      reads from the command, writes from the egress FIFO's request pointer)
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
// In this file: parts 2 to 5, then how a command enters the unit
// (dispatch_lsu) and the unit's register drivers.
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
// 2. LSU admission and request generation
// ---------------------------------------------------------------------------

bool Machine::is_store(LsuKind kind) {
    return kind == LsuKind::vector_store || kind == LsuKind::scalar_store;
}

// Capture and check an accepted LSU instruction's addresses. Contract
// violations panic here: a misaligned or out-of-bounds lane (a v8 lane must
// be sector-aligned) or uniform base (and, in the sanitizer, two lanes of a
// stg.v8 naming one sector, or two lanes of a scalar store naming one word:
// UB, as on NVIDIA -- the machine sends every lane's write, one
// request per lane in lane order, and with the placeholder controller's
// queue-front order the highest lane's lands last; a reordering controller
// may land another, which is why no program may rely on it).
Machine::LsuCommand Machine::plan_lsu_command(const DispatchEntry& op) const {
    const auto& inst = op.instruction;
    LsuCommand command;
    command.warp = op.warp;
    command.pc = op.pc;
    command.instruction = inst;
    const auto checked_address = [&](uint32_t base) {
        const auto address = static_cast<int64_t>(base) + inst.imm;
        if (address < 0) panic(op, "gmem address arithmetic underflow");
        return static_cast<uint64_t>(address);
    };
    switch (inst.opcode) {
    case Opcode::ldg_v8:
    case Opcode::stg_v8: {
        // One sector per ACTIVE lane, at that lane's own address: the
        // sector's eight words are rD..rD+7 (rS..rS+7) at that lane. An
        // inactive lane's address is never checked and it makes no request.
        const bool load = inst.opcode == Opcode::ldg_v8;
        command.kind = load ? LsuKind::vector_load : LsuKind::vector_store;
        command.reg = load ? inst.dst : inst.src1;
        command.lanes = op.active_mask;
        if (config_.gmem_sector_bytes != 32) { // refused at kernel load (check_v8_sectors)
            throw SimError("internal: " + opcode_name(inst.opcode) + " dispatched on a profile without 32-byte sectors");
        }
        std::vector<uint64_t> named;
        const auto bases = capture(op, inst.src0);
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            if ((op.active_mask >> lane & 1) == 0) continue;
            const auto address = checked_address(bases[lane]);
            if (address % config_.gmem_sector_bytes != 0) {
                panic(op, opcode_name(inst.opcode) + " address " + std::to_string(address) +
                    " is not " + std::to_string(config_.gmem_sector_bytes) + "-byte aligned");
            }
            if (address > gmem_.size() || gmem_.size() - address < config_.gmem_sector_bytes) {
                panic(op, "gmem access out of bounds");
            }
            named.push_back(address);
            command.sectors.push_back({address, lane, 0});
        }
        if (!load) sanitizer_.check_store_word_collision(site(op), named, "gmem sector");
        command.requests_total = static_cast<uint32_t>(command.sectors.size());
        break;
    }
    case Opcode::ldg:
    case Opcode::stg: {
        // Only the active lanes: an inactive lane's address is never
        // checked, it makes no request, and a load leaves its lane unwritten.
        const bool load = inst.opcode == Opcode::ldg;
        command.kind = load ? LsuKind::scalar_load : LsuKind::scalar_store;
        command.reg = load ? inst.dst : inst.src1;
        command.lanes = op.active_mask;
        std::vector<uint64_t> words;
        const auto bases = capture(op, inst.src0);
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            if ((op.active_mask >> lane & 1) == 0) continue;
            const auto address = checked_address(bases[lane]);
            if (address % 4 != 0) panic(op, "unaligned 32-bit gmem access");
            if (address > gmem_.size() || gmem_.size() - address < 4) {
                panic(op, "gmem access out of bounds");
            }
            words.push_back(address);
            const uint64_t sector = address / config_.gmem_sector_bytes * config_.gmem_sector_bytes;
            command.sectors.push_back({sector, lane, static_cast<uint32_t>(address - sector)});
        }
        if (!load) sanitizer_.check_store_word_collision(site(op), words, "gmem");
        command.requests_total = static_cast<uint32_t>(command.sectors.size());
        break;
    }
    case Opcode::ldg_uni: {
        command.kind = LsuKind::uniform_load;
        command.reg = inst.dst;
        command.lanes = full_mask();
        const auto address = checked_address(uniform_operand(op, inst.src0));
        if (address % 4 != 0) panic(op, "unaligned 32-bit gmem access");
        if (address > gmem_.size() || gmem_.size() - address < 4) {
            panic(op, "gmem access out of bounds");
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

// A store whose data has not all been moved into the egress FIFO (in Q). Such
// a store holds the store buffer or is waiting for it; stores take the buffer
// in command order.
bool Machine::store_has_data_left(const LsuCommand& command) {
    return (command.kind == LsuKind::vector_store && command.sectors_pushed < command.requests_total) ||
        (command.kind == LsuKind::scalar_store && command.words_pushed < command.sectors.size());
}

// The lanes a command's buffer group holds (fixed lane groups, so a lane's
// place in the buffer is wiring): a v8 load's lsu.load_group_lanes, a v8
// store's lsu.store_group_lanes, a gather's as many lanes as the load buffer
// has words (all sixteen when it has that many); a uniform load is one word,
// a scalar store brings every lane at its dispatch.
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

// The store buffer is free for a scalar store's data at its dispatch: no
// admitted store still has data to move (a released one has none).
bool Machine::store_buffer_free() const {
    const auto& commands = lsu_commands_.cur();
    return std::none_of(commands.begin(), commands.end(), store_has_data_left);
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
// when: load - always (requests run ahead of the load buffer: the data waits
// in DRAM, in the controller and in the ingress FIFO); store - the egress
// request pointer is its next slot, complete and visible (scatter: and a
// free-ID tag).
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
            return &command;
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

Machine::GmemSectorRequest Machine::lsu_read_request(const LsuCommand& command, uint64_t id) {
    GmemSectorRequest request;
    request.id = id;
    request.command = command.id;
    request.owner = GmemSectorRequest::Owner::lsu;
    request.load = true;
    request.beats = beats_per_sector(config_);
    request.ready_dram_tick = dram_tick_ + config_.gmem_latency_dram_ticks;
    const auto index = command.requests_emitted;
    request.stream = command.stream;
    request.sector_address = command.sectors.at(index).sector_address;
    return request;
}

// ---------------------------------------------------------------------------
// 3. Ingress
// ---------------------------------------------------------------------------

// The ingress FIFO read port: lsu.pluck_words_per_cycle words per cycle from
// the head, only from beats whose BRAM latency has passed, each word to its
// stream's consumer. A consumer that cannot take its word (the TMA sector
// register still holds an unplaced sector; the load buffer still holds a
// group waiting for the RF write port, the previous group of this load or an
// older load's) stalls the head: shared-FIFO head-of-line blocking. Gathers
// read the whole sector at this rate and keep one word. The destinations are seen as this cycle's drains leave them
// (same-cycle space reuse (d)): a register drained now takes new words now,
// visible next cycle; no word passes through.
Machine::IngressPluck Machine::decide_ingress(const Wires& w) {
    IngressPluck pluck;
    TmaSectorRegister tma_reg = tma_sector_.cur();
    if (w.shmem.tma_sector_released) {
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
            if (!pick_scalar_word(record, beat, consumed, take, pluck, w)) break;
            note_first_return(command);
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
// occupied until the placement's last port-B write round has read it.
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
    returned.run = record.run;
    returned.sector_request = beat.request;
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
        return claim.command == command.id && claim.group == group && pluck.lanes.size() != claim.expected;
    }
    if (!start.owner.has_value() || released) {
        IngressPluck::LoadClaim claim;
        claim.command = command.id;
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
    if (*start.owner != command.id || start.group != group) return false;
    return start.valid_count + pluck.lanes.size() != start.expected;
}

// Ingress -> the load buffer at the pluck rate: a v8 load's sector is one
// lane's eight words, word w of lane l at buffer word (l - first lane of the
// group) x 8 + w. No sector register.
bool Machine::pluck_vector(ReadRecord& record, const IngressBeat& beat, uint32_t consumed,
    uint32_t take, IngressPluck& pluck, const Wires& w) {
    const auto* command = find_lsu_command(record.command);
    if (command == nullptr) throw SimError("internal reference to a retired LSU command #" + std::to_string(record.command));
    const auto width = lsu_group_width(*command);
    const auto group = record.lane / width;
    // The order of the FIFO head is checked before asking whether the buffer
    // takes it: a sector returned out of request order may belong to a later
    // group, which the buffer would refuse for good while it waits for the
    // sector that was skipped (the check reads nothing forward: the count of
    // bytes received moves only when they are consumed, below).
    sanitizer_.check_ordered_beat(cycle_, record.stream, beat.request, beat.offset + consumed);
    if (!load_buffer_takes(*command, group, pluck, w)) return false;
    for (uint32_t byte = 0; byte < take; byte += 4) {
        const uint32_t index = (record.lane - group * width) * 8 + (beat.offset + consumed + byte) / 4;
        uint32_t word = 0;
        std::memcpy(&word, beat.bytes.data() + consumed + byte, 4);
        pluck.lanes.push_back({load_buffer, index, word});
    }
    probe(LsuEdge::buffer_write, command->id, beat.request, take);
    record.received += take;
    sanitizer_.record_ordered_pluck(record.stream, take);
    return true;
}

// Gather / ldg.uni: no sector register. The lane's word is taken off the
// FIFO read port into the load buffer (a gather: word l - first lane of its
// group; uniform: the one word, fanned out to every lane by the RF write);
// the rest of the sector is read at the same rate and dropped (those bytes
// need no buffer). The word waits at the FIFO head while the buffer holds
// another group. Returns false when the head must wait.
bool Machine::pick_scalar_word(ReadRecord& record, const IngressBeat& beat, uint32_t consumed,
    uint32_t take, IngressPluck& pluck, const Wires& w) {
    const uint32_t begin = beat.offset + consumed;
    // As in pluck_vector: the head's order first, then the buffer.
    sanitizer_.check_ordered_beat(cycle_, record.stream, beat.request, beat.offset + consumed);
    if (record.word_offset >= begin && record.word_offset + 4 <= begin + take) {
        const auto* command = find_lsu_command(record.command);
        if (command == nullptr) throw SimError("internal reference to a retired LSU command #" + std::to_string(record.command));
        const bool uniform = record.kind == ReadKind::uniform;
        const auto width = lsu_group_width(*command);
        const auto group = uniform ? 0 : record.lane / width;
        if (!load_buffer_takes(*command, group, pluck, w)) return false;
        uint32_t word = 0;
        std::memcpy(&word, beat.bytes.data() + (record.word_offset - beat.offset), sizeof(word));
        pluck.lanes.push_back({load_buffer, uniform ? 0 : record.lane - group * width, word});
        probe(LsuEdge::buffer_write, command->id, beat.request, 4);
    }
    record.received += take;
    sanitizer_.record_ordered_pluck(record.stream, take);
    return true;
}

// (T) A gather or ldg.uni may already have retired while the tail of its
// last sector is still being read out of the FIFO and dropped (its read ID
// table entry lives until the sector's last byte: drive_read_records), so
// this tolerates retirement.
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
// now (d). A stg.v8 dispatched this cycle takes part (bypass (e)).
std::optional<Machine::RfReadRequest> Machine::lsu_store_read_request(const Wires& w) const {
    std::vector<const LsuCommand*> commands;
    for (const auto& command : lsu_commands_.cur()) commands.push_back(&command);
    if (w.dispatch.lsu.has_value()) commands.push_back(&w.dispatch.lsu->command);
    for (const auto* command : commands) {
        if (!store_has_data_left(*command)) continue;
        if (command->kind != LsuKind::vector_store) return std::nullopt;
        const bool drained_now = w.egress.lsu_group_drained && w.egress.lsu_command == command->id;
        const uint32_t sector = command->sectors_pushed + (drained_now ? 1 : 0);
        if (sector == command->requests_total) return std::nullopt; // its last group has just left
        const uint32_t beat = drained_now ? 0 : command->group_beats_read;
        if (beat == 8) return std::nullopt; // the group's eight beats are read; it is landing or draining
        if (beat == 0 && !drained_now && !w.egress.lsu_store_drained &&
            lsu_buffers_.at(store_buffer).cur().owner.has_value()) {
            return std::nullopt;
        }
        const auto width = config_.lsu_store_group_lanes;
        RfReadRequest request{RfReader::lsu_store, 0, command->warp,
            static_cast<uint8_t>(command->reg + beat), command->id, store_buffer};
        request.group = command->sectors.at(sector).lane / width;
        request.beat = beat;
        request.lanes = lsu_group_lanes(*command, request.group);
        // The command's last group: no active lane above it.
        request.last_group = (request.group + 1) * width >= config_.lanes ||
            (command->lanes >> ((request.group + 1) * width)) == 0;
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
    const auto& fifo = egress_fifo_.cur();
    const uint32_t sector_words = config_.gmem_sector_bytes / 4;
    const EgressSlot* open = nullptr;
    if (!fifo.empty() && fifo.back().words_written < fifo.back().words_total) open = &fifo.back();
    const auto& tma_stores = tma_store_registers_.cur();
    const auto& tma_head = tma_stores.ring.at(tma_stores.head);
    if (open == nullptr) {
        const bool tma_ready = tma_head.has_value() && tma_head->complete();
        const LsuCommand* lsu = nullptr;
        for (const auto& command : lsu_commands_.cur()) {
            if (store_has_data_left(command)) {
                lsu = &command;
                break;
            }
        }
        if (lsu != nullptr && lsu->kind == LsuKind::vector_store) {
            const auto& buffer = lsu_buffers_.at(store_buffer).cur();
            if (buffer.owner != lsu->id || buffer.valid_count != buffer.expected) {
                lsu = nullptr; // its group's eight RF read beats have not all landed yet
            }
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
        if (tma_ready) { // the register's run gives its sector and byte enables
            slot.client = GmemClient::tma;
            slot.command = tma_head->run.command;
            slot.sector_address = run_sector_address(tma_head->run);
            slot.byte_enable = run_byte_enable(tma_head->run);
            slot.words_total = sector_words;
        } else if (lsu->kind == LsuKind::vector_store) {
            // The next active lane's sector, in lane order.
            slot.client = GmemClient::lsu;
            slot.command = lsu->id;
            slot.index = lsu->sectors_pushed;
            slot.sector_address = lsu->sectors.at(lsu->sectors_pushed).sector_address;
            slot.byte_enable = full_byte_enable(config_.gmem_sector_bytes);
            slot.words_total = sector_words;
        } else {
            const auto& sector = lsu->sectors.at(lsu->words_pushed);
            slot.client = GmemClient::lsu;
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
        const auto* command = find_lsu_command(open->command);
        if (command == nullptr) throw SimError("internal reference to a retired LSU command #" + std::to_string(open->command));
        const auto& buffer = lsu_buffers_.at(store_buffer).cur();
        if (buffer.owner != command->id) throw SimError("internal: the egress port read another store's buffer");
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
                // next cycle (the command still waits for its acks).
                if (index + 1 == command->sectors.size()) write.lsu_store_drained = true;
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

// An LSU command ends: its slot is free for a new owner this same cycle (d).
void Machine::release_lsu_command(Wires& w, uint64_t id) {
    const auto* command = find_lsu_command(id);
    if (command == nullptr) throw SimError("internal release of an unknown LSU command");
    trace_->lsu_command_end(cycle_, id, lsu_phase_detail(*command));
    lsu_phases_.erase(id);
    w.released_lsu_commands.push_back(id);
}

// ldg / stg / ldg.v8 / stg.v8 / ldg.uni: the addresses arrive and are
// checked now; the command takes the slot admission secured.
void Machine::dispatch_lsu(const DispatchEntry& op, Wires& w) {
    const auto& inst = op.instruction;
    auto& counters = trace_->lsu_counters();
    auto command = plan_lsu_command(op);
    command.id = next_lsu_command_id_++;
    command.depbar = inst.control.set_depbar;
    command.buffer = is_store(command.kind) ? store_buffer : load_buffer;
    // Every load rides the one ordered LSU read stream (accept_lsu_request).
    command.stream = lsu_stream;
    DispatchDecision::LsuAdmission admission;
    switch (command.kind) {
    case LsuKind::vector_load:
        ++counters.lsu_vector_loads;
        counters.lsu_useful_bytes += 32ull * command.sectors.size(); // its active lanes' sectors
        break;
    case LsuKind::vector_store:
        ++counters.lsu_vector_stores;
        counters.lsu_useful_bytes += 32ull * command.sectors.size();
        break;
    case LsuKind::scalar_load:
        ++counters.lsu_scalar_loads;
        counters.lsu_useful_bytes += 4ull * command.sectors.size(); // its active lanes
        break;
    case LsuKind::scalar_store:
        // Its RF read beat (booked with the address operand) lands the store
        // data in the store buffer.
        admission.store_data = capture(op, inst.src1);
        probe(LsuEdge::rf_read_beat, command.id, inst.src1, config_.lanes * 4);
        ++counters.lsu_scalar_stores;
        counters.lsu_useful_bytes += 4ull * command.sectors.size(); // its active lanes
        break;
    case LsuKind::uniform_load:
        ++counters.lsu_uniform_loads;
        counters.lsu_useful_bytes += 4;
        break;
    }
    trace_->lsu_command_begin(cycle_, command.id, op.warp, opcode_name(inst.opcode));
    lsu_phases_[command.id].admit_cycle = cycle_;
    // Sanitizer: the admitted set never exceeds the command slots.
    sanitizer_.check_lsu_admitted(lsu_commands_.cur().size() - w.released_lsu_commands.size() + 1);
    admission.command = std::move(command);
    w.dispatch.lsu = std::move(admission);
}

// The LSU command slots (admission order): every stage's progress, the
// releases, and issue's admission, in one calculation. A command released
// now frees its slot for an admission now (d).
void Machine::drive_lsu_commands(const Wires& w) {
    // A load group whose last RF write is granted now: its lanes are done.
    std::optional<std::pair<uint64_t, uint32_t>> drained; // (command, lanes)
    if (load_buffer_released(w)) {
        const auto& buffer = lsu_buffers_[load_buffer].cur();
        const auto* owner = find_lsu_command(*buffer.owner);
        drained = std::make_pair(*buffer.owner, owner->kind == LsuKind::uniform_load
            ? 1u : static_cast<uint32_t>(std::popcount(buffer.lanes)));
    }
    const bool changed = drained.has_value() || w.egress.client == GmemClient::lsu ||
        w.channel.granted == GmemClient::lsu || !w.controller.lsu_acks.empty() ||
        w.lsu_rf_read.command.has_value() || !w.released_lsu_commands.empty() || w.dispatch.lsu.has_value();
    if (!changed) return;
    auto next = lsu_commands_.cur();
    const auto command = [&](uint64_t id) -> LsuCommand& {
        for (auto& candidate : next) {
            if (candidate.id == id) return candidate;
        }
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
    for (const auto& [id, acks] : w.controller.lsu_acks) command(id).acks += acks;
    std::erase_if(next, [&](const LsuCommand& candidate) {
        return std::find(w.released_lsu_commands.begin(), w.released_lsu_commands.end(), candidate.id) !=
            w.released_lsu_commands.end();
    });
    if (w.dispatch.lsu.has_value()) next.push_back(w.dispatch.lsu->command);
    if (w.lsu_rf_read.command.has_value()) ++command(*w.lsu_rf_read.command).group_beats_read;
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
//                 2. a scalar store dispatched now brings its data;
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
            reg.owner = claim.command;
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
        if (w.dispatch.lsu.has_value() && w.dispatch.lsu->store_data.has_value()) {
            // A scatter: every lane's word (the inactive ones are never sent).
            auto& reg = edit();
            if (reg.owner.has_value()) throw SimError("internal: store data into the owned store buffer");
            reg.owner = w.dispatch.lsu->command.id;
            reg.lanes = w.dispatch.lsu->command.lanes;
            for (uint32_t lane = 0; lane < config_.lanes; ++lane) put(reg, lane, (*w.dispatch.lsu->store_data)[lane]);
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

// The ingress FIFO: the read port consumed bytes from the head; the bus
// wrote beats at the tail (readable from their visible cycle on).
void Machine::drive_ingress(const Wires& w) {
    if (w.ingress.bytes == 0 && w.controller.ingress_beats.empty()) return;
    auto next = ingress_.cur();
    uint32_t left = w.ingress.bytes;
    while (left > 0) {
        auto& head = next.beats.front();
        const auto take = std::min<uint32_t>(left, static_cast<uint32_t>(head.bytes.size()) - head.consumed);
        head.consumed += take;
        left -= take;
        if (head.consumed == head.bytes.size()) next.beats.pop_front();
    }
    next.bytes -= w.ingress.bytes;
    for (const auto& beat : w.controller.ingress_beats) {
        next.bytes += static_cast<uint32_t>(beat.bytes.size());
        next.beats.push_back(beat);
    }
    // The BRAM's size (IngressFifo): bytes, and the beats of that many sectors.
    const uint32_t beats_per_sector =
        (config_.gmem_sector_bytes + config_.gmem_bus_bytes_per_dram_tick - 1) / config_.gmem_bus_bytes_per_dram_tick;
    if (next.bytes > config_.lsu_ingress_sectors * config_.gmem_sector_bytes ||
        next.beats.size() > std::size_t{config_.lsu_ingress_sectors} * beats_per_sector) {
        throw SimError("internal: the ingress FIFO holds more than lsu.ingress_sectors sectors");
    }
    ingress_.set_next(std::move(next), "drive_ingress");
}

// The egress FIFO: write requests accepted for its slots, slots whose last
// beat left, a new slot opened by the write port, and the words written.
void Machine::drive_egress(const Wires& w) {
    const bool changed = w.channel.tma_write_slot.has_value() || w.channel.lsu_write_slot.has_value() ||
        w.controller.egress_slots_freed != 0 || w.egress.client.has_value();
    if (!changed) return;
    auto next = egress_fifo_.cur();
    const auto mark = [&](std::optional<uint64_t> slot_id, uint64_t request) {
        if (!slot_id.has_value()) return;
        for (auto& slot : next) {
            if (slot.id == *slot_id) slot.request = request;
        }
    };
    mark(w.channel.tma_write_slot, w.channel.tma_write_id);
    mark(w.channel.lsu_write_slot, w.channel.lsu_id);
    for (uint32_t freed = 0; freed < w.controller.egress_slots_freed; ++freed) next.pop_front();
    if (w.egress.new_slot.has_value()) next.push_back(*w.egress.new_slot);
    if (w.egress.client.has_value()) {
        auto& open = next.back();
        for (const auto& [offset, word] : w.egress.words) std::memcpy(open.data.data() + offset, &word, 4);
        open.words_written += static_cast<uint32_t>(w.egress.words.size());
        if (w.egress.slot_complete) open.complete_cycle = cycle_;
    }
    if (next.size() > config_.lsu_egress_sectors) {
        throw SimError("internal: the egress FIFO holds more than lsu.egress_sectors slots");
    }
    egress_fifo_.set_next(std::move(next), "drive_egress");
}

// The read ID table: entries for reads accepted now; the pluck's updates.
// (A read accepted now cannot return now: no entry gets both.)
void Machine::drive_read_records(const Wires& w) {
    if (w.channel.new_read_records.empty() && w.ingress.records.empty()) return;
    auto next = read_records_.cur();
    for (const auto& [id, record] : w.channel.new_read_records) next.emplace(id, record);
    for (const auto& [id, record] : w.ingress.records) {
        if (record.has_value()) next[id] = *record;
        else next.erase(id);
    }
    // The table's size (ReadRecord): per kind of reader.
    std::size_t tma = 0;
    std::size_t vector = 0;
    std::size_t free_ids = 0;
    for (const auto& [id, record] : next) {
        (void)id;
        if (record.kind == ReadKind::tma) ++tma;
        else if (record.kind == ReadKind::vector) ++vector;
        else ++free_ids;
    }
    // Every LSU read is on the ordered LSU stream; requests run ahead of the
    // load buffer, so each command slot may have all its sectors in flight
    // (one per lane). One more entry is the tail of a retired gather or
    // ldg.uni: such a load writes the RF, and frees its command slot, once its
    // last word is in, while the rest of that word's sector is still coming
    // off the bus and being read out of the ingress FIFO and dropped; its
    // entry lives until the sector's last byte. Only one such tail can exist:
    // the stream is ordered and the FIFO is read in order, so the next load
    // to retire has its last word behind that tail and retires after it is
    // gone. Hardware: lsu.active_commands x lanes + 1 entries.
    const std::size_t command_sectors = config_.lanes;
    const std::size_t retired_tail = 1;
    if (tma > config_.tma_sector_credits ||
        vector + free_ids > config_.lsu_active_commands * command_sectors + retired_tail) {
        throw SimError("internal: the read ID table holds more reads than its readers can have in flight");
    }
    read_records_.set_next(std::move(next), "drive_read_records");
}

// Free-ID tags (scatter writes only): returned when a scatter's write is
// acknowledged; then taken by a scatter's request now (a tag free in Q, so
// today never one returned now).
void Machine::drive_free_id_tags(const Wires& w) {
    if (!w.channel.lsu_tag.has_value() && w.controller.freed_tags.empty()) return;
    auto next = free_id_tags_.cur();
    for (const auto tag : w.controller.freed_tags) next.at(tag) = 0;
    if (w.channel.lsu_tag.has_value()) next.at(*w.channel.lsu_tag) = 1;
    free_id_tags_.set_next(std::move(next), "drive_free_id_tags");
}

// The LSU's store-data reads in flight: the one sampled now enters the RF's
// read stages; the one at the RF output leaves at this edge (it fills the
// store buffer, drive_lsu_buffers).
void Machine::drive_lsu_read_pipe(const Wires& w) {
    if (w.lsu_rf_read.command.has_value()) lsu_read_pipe_.push(w.lsu_rf_read, "drive_lsu_read_pipe");
}

} // namespace nanosm

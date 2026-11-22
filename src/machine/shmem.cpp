// Shared memory: ports A and B, the selects, load and store slots, bank rounds, and what
// issue sees of them.
// (Machine is one class, src/machine/machine.hpp; the map of its files is at the top of
// src/machine/machine.cpp.)

#include "machine/machine.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

namespace nanosm {

// ===========================================================================
// shmem. Four services, each granting at most one bank round per core cycle:
//   A-read   SIMT lds / lds.uni (load slots), mbar probes, RMW reads, TMA reads
//   A-write  SIMT sts (store slots), RMW writes
//   B-read   the MMA's operand fetch (step_mma)
//   B-write  TMA placement (gmem -> shmem)
// A transaction -- one SIMT access, one TMA read or placement, one RMW step --
// keeps its service from its first round to its last (no preemption). Every
// wait happens before a grant. A granted round reaches the RAM
// shmem.port_a_depth (D) cycles later -- the same D on all four services, so
// the RAM sees rounds in grant order -- and a read's words reach the port
// output shmem.read_latency_cycles (Lb) cycles after its sample. An RMW's
// update then crosses the RMW engine's shmem.rmw_compute_cycles (C) ALU
// stages before its write is granted. A read and a write of one word at the
// RAM in one cycle: READ_FIRST (the write lands at the clock edge). The MMA's
// port-B read pipeline lives in the MMA engine's register (advance_mma,
// src/machine/tensor_core.cpp); the RMW engine's own timeline is at the top of
// src/machine/barriers.cpp, the TMA's rounds are src/machine/tma_ports.cpp.
//
// fence.sh is admitted in the cycle its warp's last
// earlier shmem access REACHES THE RAM (a store's write edge, a load's
// sample: its round's grant + D), so the instruction after it issues once
// that edge has passed. Its declared state is one count per warp
// (shmem_to_ram_: cycles until the warp's latest granted round is at the
// RAM); own_shmem_at_ram is the admission term. Whether an engine command
// needed one is the sanitizer's question at the command's accept (missing
// fence.sh / race).
// ===========================================================================

// shmem this cycle, in wire order:
//   1. the RAM stage, from the pipelines' Q: the write rounds reaching the RAM
//      now and the completions they carry (an sts's depbar, an RMW's wake, a
//      placement's end); the read round sampled now; the read round at the
//      port output, whose requester takes its words now;
//   2. the selects, one per shared resource, in this order: A-write (an RMW's
//      write, a store round, and which transaction STARTS next: an RMW or a
//      store), the RMW engine's admission (which requester an RMW start
//      takes), A-read (sees that start: an RMW's read may be granted in its
//      start cycle), B-write;
//   3. their data paths: the RMW start, the granted rounds enter the
//      pipelines, the requesters advance.
void Machine::step_shmem(Wires& w) {
    w.shmem.arbiters = shmem_arbiters_.cur();
    shmem_ram_stage(w.shmem);
    commit_shmem_writes(w);
    deliver_shmem_read(w);
    w.shmem.a_write = select_a_write(w);
    w.shmem.rmw_start = select_rmw_admission(w);
    w.shmem.a_read = select_a_read(w);
    w.shmem.b_write = select_b_write(w);
    serve_rmw_start(w);
    serve_a_read(w);
    serve_a_write(w);
    serve_b_write(w);
    advance_tma_read(w);
    count_engine_wait_on_other(w);
}

// The RAM stage. The sample reads shmem as it is now, before this cycle's
// writes land at the clock edge (READ_FIRST).
void Machine::shmem_ram_stage(ShmemCycle& shmem) const {
    const auto& reads = a_read_pipe_.cur();
    shmem.delivered = reads.back();
    if (const auto& at_ram = reads.at(config_.shmem_port_a_depth - 1); at_ram.has_value()) {
        shmem.sampled = *at_ram;
        for (const auto address : at_ram->addresses) shmem.sampled->words.push_back(read_shmem_u32(address));
    }
    for (const auto* pipe : {&a_write_pipe_.cur(), &b_write_pipe_.cur()}) {
        if (!pipe->back().has_value()) continue;
        const auto& writes = pipe->back()->writes;
        shmem.writes.insert(shmem.writes.end(), writes.begin(), writes.end());
    }
}

// What a write round carries to the RAM happens there: an sts's last round
// clears its depbar; an RMW write that flipped the phase wakes the waiters
// (at the commit, never at the grant); a placement's last
// round ends the TMA's load write (step_tma).
void Machine::commit_shmem_writes(Wires& w) {
    if (const auto& beat = a_write_pipe_.cur().back(); beat.has_value()) {
        if (const auto& done = beat->sts_done; done.has_value()) {
            record_depbar_completion(w, done->warp, done->bar, "depbar underflow on sb" + std::to_string(done->bar));
        }
        if (beat->wake.has_value()) wake_mbar_waiters(w, beat->wake->address, beat->wake->phase);
    }
    if (const auto& beat = b_write_pipe_.cur().back(); beat.has_value() && beat->placed.has_value()) {
        w.shmem.tma_placed = beat->placed;
    }
}

// The read round at the port output goes to its requester: a load round to
// its slot's gather register, a probe to its warp, an RMW's words through its
// update into the ALU (an atomic's old word also to every lane of its return
// slot), a store-source round to its TMA store register. (A descriptor round:
// the frontend latch, advance_tma_read.)
void Machine::deliver_shmem_read(Wires& w) {
    const auto& beat = w.shmem.delivered;
    if (!beat.has_value()) return;
    switch (beat->client) {
    case ReadClient::probe:
        deliver_probe(*beat, w);
        return;
    case ReadClient::rmw:
        w.shmem.rmw_update = rmw_update(*beat);
        if (const auto slot = rmw_.cur().active->request.return_slot; slot.has_value()) {
            LoadFill fill{*slot, {}};
            for (uint32_t lane = 0; lane < config_.lanes; ++lane) fill.lanes.emplace_back(lane, beat->words.at(0));
            w.shmem.load_fill = std::move(fill);
        }
        return;
    case ReadClient::tma_store:
        fill_tma_store_register(*beat, w);
        return;
    case ReadClient::tma_frontend:
        return;
    case ReadClient::load_slot:
        break;
    }
    const auto& slot = *load_slots_.at(beat->slot).cur();
    const auto& lanes = slot.access.rounds.at(beat->round);
    LoadFill fill{beat->slot, {}};
    for (std::size_t access = 0; access < lanes.size(); ++access) {
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            if (slot.uniform || lane == lanes[access].lane) fill.lanes.emplace_back(lane, beat->words[access]);
        }
    }
    w.shmem.load_fill = std::move(fill);
}

// ---- The requesters, from Q ----

// The global store PAUSE is a register: the RMW engine's
// operation register, held from the RMW's start to its final write grant.
// While it is set no new store transaction starts on A-write. It clears at
// the edge of the final write grant w, so stores may start at w + 1 (matched
// D still lands them after the RMW's write; nothing waits for its commit).
bool Machine::store_pause() const {
    return rmw_.cur().active.has_value();
}

// The load slot whose transaction is under way on A-read (started, rounds left).
std::optional<uint32_t> Machine::load_slot_in_service() const {
    for (uint32_t index = 0; index < load_slots_.size(); ++index) {
        const auto& slot = load_slots_[index].cur();
        if (slot.has_value() && slot->access.started() && !slot->access.serviced()) return index;
    }
    return std::nullopt;
}

// The store slot whose transaction is under way on A-write (started; a
// serviced store slot is already free).
std::optional<uint32_t> Machine::store_slot_in_service() const {
    for (uint32_t index = 0; index < store_slots_.size(); ++index) {
        const auto& slot = store_slots_[index].cur();
        if (slot.has_value() && slot->access.started()) return index;
    }
    return std::nullopt;
}

// A store transaction waits to start: a store slot held and not started.
bool Machine::store_slot_waiting() const {
    return std::any_of(store_slots_.begin(), store_slots_.end(), [](const auto& slot) {
        return slot.cur().has_value() && !slot.cur()->access.started();
    });
}

// ---- The selects: one per shared resource ----

// A-write, in order:
//   1. the RMW's write, when its update is in the ALU's last stage (Q):
//      granted now, D + Lb + C after its read. It is the RMW's final write:
//      the PAUSE ends at this cycle's edge;
//   2. a store transaction under way takes its next round -- also once an
//      RMW has started: that RMW's read waits for this store's last round
//      (select_a_read);
//   3. PAUSE: no new store transaction starts;
//   4. which transaction STARTS next. An RMW waiting to start (the engine
//      idle, a requester held) and a store waiting to start (a store slot
//      not started) alternate: when both wait, the one that did not start
//      last goes (the alternation bit, ShmemArbiters::rmw_started_last), so
//      one store transaction goes between consecutive RMWs; when one waits,
//      it goes. An RMW start (rmw_start) begins the PAUSE; the engine's
//      admission picks the requester. It may share its cycle with step 2's
//      round. A store start is round 0 of the store slot the round-robin
//      pointer picks.
// The alternation bit is read only here, to choose a START: from an RMW's
// start to its final write grant the PAUSE holds every new store, so no store
// is ever granted between an RMW's read and its write.
Machine::AWriteGrant Machine::select_a_write(const Wires&) const {
    AWriteGrant grant;
    const bool rmw_waiting = !store_pause() && rmw_requester_held();
    const bool stores_turn = store_slot_waiting() && (!rmw_waiting || shmem_arbiters_.cur().rmw_started_last);
    grant.rmw_start = rmw_waiting && !stores_turn;
    if (rmw_alu_.last().has_value()) { // (the PAUSE is set: no RMW starts)
        grant.user = AWriteUser::rmw;
        return grant;
    }
    if (const auto slot = store_slot_in_service(); slot.has_value()) {
        grant.user = AWriteUser::store_slot;
        grant.slot = *slot;
        return grant;
    }
    if (store_pause() || !stores_turn) return grant;
    std::vector<uint8_t> ready(store_slots_.size(), 0);
    for (std::size_t index = 0; index < ready.size(); ++index) ready[index] = store_slots_[index].cur().has_value() ? 1 : 0;
    const auto choice = round_robin_grant(ready, shmem_arbiters_.cur().a_write, 1);
    grant.user = AWriteUser::store_slot;
    grant.slot = choice.granted.front();
    grant.pointer_after = choice.cursor_after;
    return grant;
}

// A-read, in order:
//   1. a transaction under way takes its next round (no preemption);
//   2. a started RMW waiting for its read (started earlier, or starting now)
//      goes before every new requester. It is granted strictly after the
//      last round of the store transaction under way on A-write, if any:
//      matched D then lands that store before the RMW's sample. Until then
//      nothing new starts on A-read (implementer choice: with no preemption a
//      new multi-round read could only delay the RMW's read and so lengthen
//      the store pause);
//   3. ONE round-robin over the new requesters: load slots not yet started
//      (an atomic's return slot is not a requester: the engine reads for
//      it), the probe register, the TMA's new read.
// Reads are never paused: between an RMW's read and its write, step 3 serves
// loads, probes and TMA reads as usual.
Machine::AReadGrant Machine::select_a_read(const Wires& w) const {
    if (tma_read_in_service()) return AReadGrant::to(AReadUser::tma);
    if (const auto slot = load_slot_in_service(); slot.has_value()) return AReadGrant::to(AReadUser::load_slot, *slot);
    if (rmw_awaits_read(w)) return store_slot_in_service().has_value() ? AReadGrant{} : AReadGrant::to(AReadUser::rmw);
    return a_read_round_robin(w);
}

Machine::AReadGrant Machine::a_read_round_robin(const Wires& w) const {
    const auto slots = static_cast<uint32_t>(load_slots_.size());
    std::vector<uint8_t> ready(slots + 2, 0);
    for (uint32_t index = 0; index < slots; ++index) {
        const auto& slot = load_slots_[index].cur();
        ready[index] = slot.has_value() && !slot->engine_filled && !slot->access.started() ? 1 : 0;
    }
    ready[slots] = mbar_probe_.cur().has_value() ? 1 : 0;
    ready[slots + 1] = w.tma.a_read.has_value() ? 1 : 0;
    const auto choice = round_robin_grant(ready, shmem_arbiters_.cur().a_read, 1);
    if (choice.granted.empty()) return {};
    const auto requester = choice.granted.front();
    AReadGrant grant;
    if (requester < slots) {
        grant = AReadGrant::to(AReadUser::load_slot, requester);
    } else if (requester == slots) {
        grant = AReadGrant::to(AReadUser::probe);
    } else {
        grant = AReadGrant::to(AReadUser::tma);
        grant.tma_start = true;
    }
    grant.pointer_after = choice.cursor_after; // the one place the pointer's next value is computed
    return grant;
}

// B-write has one client: the TMA's placement under way, else a new one.
Machine::BWriteUser Machine::select_b_write(const Wires& w) const {
    if (tma_write_txn_.cur().has_value()) return BWriteUser::tma_continue;
    return w.tma.placement.has_value() ? BWriteUser::tma_start : BWriteUser::none;
}

// ---- The data paths ----

// The granted A-read round enters the read pipeline. A round-robin grant
// moves the pointer past its requester (pointer_after, from the select). A
// store-source read starting now reserves its TMA store register. A warp's
// own round (a_read_warp) reaches the RAM -- its sample -- D cycles from now.
void Machine::serve_a_read(Wires& w) {
    auto& shmem = w.shmem;
    const auto& grant = shmem.a_read;
    if (grant.pointer_after.has_value()) shmem.arbiters.a_read = *grant.pointer_after;
    switch (grant.user) {
    case AReadUser::none:
        return;
    case AReadUser::load_slot:
        shmem.a_read_warp = load_slots_.at(grant.slot).cur()->warp;
        shmem.a_read_beat = load_round(grant.slot);
        return;
    case AReadUser::probe: {
        ReadBeat beat;
        beat.client = ReadClient::probe;
        beat.probe = *mbar_probe_.cur();
        beat.addresses = {beat.probe->byte_address, beat.probe->byte_address + 4};
        trace_->shmem_grant(cycle_, ShmemService::a_read, ShmemClient::probe);
        sanitizer_.record_simt_shmem_grant(cycle_, SimtShmemHolder::probe, beat.probe->warp,
            cycle_ + config_.shmem_port_a_depth);
        shmem.a_read_warp = beat.probe->warp;
        shmem.a_read_beat = std::move(beat);
        return;
    }
    case AReadUser::rmw: { // the engine's operation, or the one starting now: one round
        const auto& active = rmw_.cur().active.has_value() ? *rmw_.cur().active : *shmem.rmw_started;
        trace_->shmem_grant(cycle_, ShmemService::a_read, ShmemClient::rmw);
        ReadBeat beat;
        beat.client = ReadClient::rmw;
        const auto address = active.request.byte_address;
        // An mbar operation reads both mbar words; an atomic its one word.
        beat.addresses = is_atomic_operation(active.request.operation) ? std::vector<uint32_t>{address}
                                                                       : std::vector<uint32_t>{address, address + 4};
        shmem.a_read_warp = active.request.warp; // none: an engine event's RMW
        shmem.a_read_beat = std::move(beat);
        return;
    }
    case AReadUser::tma: {
        const auto& request = grant.tma_start ? *w.tma.a_read : tma_read_txn_.cur()->request;
        uint32_t store_register = 0;
        if (request.kind == TmaPortKind::store_read) {
            store_register = grant.tma_start ? *w.tma.store_register : tma_read_txn_.cur()->store_register;
            if (grant.tma_start) shmem.tma_store_reserved = store_register;
        }
        shmem.a_read_beat = tma_read_round(request, grant.tma_start, store_register);
        return;
    }
    }
}

// A load slot's next round: its lanes' word addresses enter the read pipeline.
Machine::ReadBeat Machine::load_round(uint32_t index) {
    const auto& slot = *load_slots_.at(index).cur();
    const auto round = slot.access.granted;
    const auto rounds = slot.access.rounds.size();
    if (round == 0) {
        trace_->shmem_begin(ShmemService::a_read, cycle_, slot.uniform ? "lds.uni" : "lds",
            "warp=" + std::to_string(slot.warp) + ", rounds=" + std::to_string(rounds), TraceCategory::shmem);
    }
    trace_->shmem_grant(cycle_, ShmemService::a_read, ShmemClient::simt, index);
    if (round + 1 == rounds) trace_->shmem_end(ShmemService::a_read, cycle_ + 1);
    sanitizer_.record_simt_shmem_grant(cycle_, SimtShmemHolder::load_slot, index, cycle_ + config_.shmem_port_a_depth);
    ReadBeat beat;
    beat.client = ReadClient::load_slot;
    beat.slot = index;
    beat.round = round;
    for (const auto& lane : slot.access.rounds[round]) beat.addresses.push_back(static_cast<uint32_t>(lane.byte_address));
    return beat;
}

// The granted A-write round enters the A-write pipeline. A new store
// transaction moves the round-robin pointer past its slot (pointer_after,
// from the select); every start sets the alternation bit (an RMW start: true;
// a store start: false). An RMW's write takes its update from the ALU's last
// stage; a phase flip wakes the waiters when the write reaches the RAM.
void Machine::serve_a_write(Wires& w) {
    auto& shmem = w.shmem;
    const auto& grant = shmem.a_write;
    if (grant.pointer_after.has_value()) shmem.arbiters.a_write = *grant.pointer_after;
    if (grant.rmw_start) {
        shmem.arbiters.rmw_started_last = true;
    } else if (grant.pointer_after.has_value()) {
        shmem.arbiters.rmw_started_last = false;
    }
    if (store_pause() && store_slot_waiting()) trace_->shmem_store_pause_wait();
    switch (shmem.a_write.user) {
    case AWriteUser::none:
        return;
    case AWriteUser::store_slot:
        shmem.a_write_warp = store_slots_.at(shmem.a_write.slot).cur()->warp;
        shmem.a_write_beat = store_round(shmem.a_write.slot);
        return;
    case AWriteUser::rmw: {
        const auto& update = *rmw_alu_.last();
        const auto& request = rmw_.cur().active->request;
        trace_->shmem_grant(cycle_, ShmemService::a_write, ShmemClient::rmw);
        if (is_atomic_operation(request.operation)) {
            trace_->atomic_rmw_end(cycle_ + 1);
            // The sanitizer's live-mbar check: an ordinary atomic may not write a live mbar --
            // using the RMW engine does not authorize it.
            sanitizer_.check_live_mbar_write(cycle_, format_instruction(request.instruction, false) + " (warp " +
                std::to_string(*request.warp) + " pc " + std::to_string(request.pc) + ")",
                std::vector<std::pair<uint32_t, uint32_t>>{{request.byte_address, request.byte_address + 4}});
        } else {
            trace_->mbar_rmw_end(cycle_ + 1, update.wake.has_value());
        }
        // A SIMT RMW's final write: its warp's access reaches the RAM D cycles from now.
        if (const auto warp = request.warp; warp.has_value()) {
            sanitizer_.record_simt_shmem_grant(cycle_, SimtShmemHolder::rmw, *warp, cycle_ + config_.shmem_port_a_depth);
            shmem.a_write_warp = warp;
        }
        shmem.a_write_beat = update;
        return;
    }
    }
}

// A store slot's next round: its lanes' words enter the A-write pipeline; the
// last round carries the depbar completion to the RAM.
Machine::WriteBeat Machine::store_round(uint32_t index) {
    const auto& slot = *store_slots_.at(index).cur();
    const auto round = slot.access.granted;
    const auto rounds = slot.access.rounds.size();
    if (round == 0) {
        trace_->shmem_begin(ShmemService::a_write, cycle_, "sts",
            "warp=" + std::to_string(slot.warp) + ", rounds=" + std::to_string(rounds), TraceCategory::shmem);
    }
    trace_->shmem_grant(cycle_, ShmemService::a_write, ShmemClient::simt, index);
    // (The sanitizer checks here too that the round does not write a live mbar.)
    sanitizer_.record_simt_shmem_grant(cycle_, SimtShmemHolder::store_slot, index, cycle_ + config_.shmem_port_a_depth);
    WriteBeat beat;
    for (const auto& access : slot.access.rounds[round]) {
        std::vector<uint8_t> bytes(4);
        std::memcpy(bytes.data(), &access.value, sizeof(access.value));
        beat.writes.push_back({static_cast<uint32_t>(access.byte_address), std::move(bytes)});
    }
    if (round + 1 == rounds) {
        beat.sts_done = DepbarEvent{slot.warp, slot.depbar};
        trace_->shmem_end(ShmemService::a_write, cycle_ + 1);
    }
    return beat;
}

// The placement round granted on the port-B write enters its pipeline. Its
// last round frees the TMA sector register (the pipeline owns the bytes now).
void Machine::serve_b_write(Wires& w) {
    auto& shmem = w.shmem;
    if (shmem.b_write == BWriteUser::none) return;
    const bool start = shmem.b_write == BWriteUser::tma_start;
    auto request = start ? *w.tma.placement : *tma_write_txn_.cur();
    shmem.b_write_beat = placement_round(request, start);
    ++request.next_round;
    shmem.tma_write_changed = true;
    if (request.next_round != request.rounds) {
        shmem.tma_write_after = std::move(request);
        return;
    }
    shmem.tma_sector_released = !request.run.zero_fill;
    shmem.tma_write_after.reset();
}

// ---- What issue sees of shmem (look-ahead (c), reuse (d)): the service
// slots and the warp's own order ----

// Dispatch's slot: an empty one, or one whose RF write is granted now (d).
std::optional<uint32_t> Machine::free_load_slot(const Wires& w) const {
    for (uint32_t index = 0; index < load_slots_.size(); ++index) {
        if (!load_slots_[index].cur().has_value()) return index;
        if (w.rf_write.granted_to(AsyncWriter::load_slot, index)) return index;
    }
    return std::nullopt;
}

// Issue's count: the slots empty in Q only. A slot whose RF write is granted
// at t counts from t + 1 (the slot register's next value); this cycle's grant
// does not reach the warp selection (RF winner -> slot space -> winning warp
// would be the longest path through issue).
uint32_t Machine::free_load_slot_count() const {
    uint32_t count = 0;
    for (uint32_t index = 0; index < load_slots_.size(); ++index) {
        if (!load_slots_[index].cur().has_value()) ++count;
    }
    return count;
}

// A store slot frees when its last round is granted.
std::optional<uint32_t> Machine::free_store_slot(const Wires& w) const {
    for (uint32_t index = 0; index < store_slots_.size(); ++index) {
        if (!store_slots_[index].cur().has_value() || store_slot_released(w, index)) return index;
    }
    return std::nullopt;
}

uint32_t Machine::free_store_slot_count(const Wires& w) const {
    uint32_t count = 0;
    for (uint32_t index = 0; index < store_slots_.size(); ++index) {
        if (!store_slots_[index].cur().has_value() || store_slot_released(w, index)) ++count;
    }
    return count;
}

// The warp's instructions still in the dispatch pipeline (Q): accesses it
// has issued that have no slot yet. They count as unserviced from their
// issue (a reserved slot is unserviced immediately).
bool Machine::warp_dispatching(uint32_t warp, bool reads, bool writes, bool probes) const {
    for (const auto& stage : dispatch_pipe_.cur()) {
        if (!stage.has_value() || stage->warp != warp) continue;
        const auto opcode = stage->instruction.opcode;
        if (reads && is_shared_load(opcode)) return true;
        if (writes && (opcode == Opcode::sts || is_mbar_rmw(opcode) || is_atomic(opcode))) return true;
        if (probes && is_mbar_predicate_writer(opcode)) return true;
    }
    return false;
}

bool Machine::store_slot_released(const Wires& w, uint32_t index) const {
    const auto& slot = store_slots_[index].cur();
    return slot.has_value() && w.shmem.a_write.user == AWriteUser::store_slot && w.shmem.a_write.slot == index &&
        slot->access.granted + 1 == slot->access.rounds.size();
}

// Load slot `index` still has a round to be granted after this cycle.
bool Machine::load_rounds_left(const Wires& w, uint32_t index) const {
    const auto& slot = *load_slots_[index].cur();
    const bool granted_now = w.shmem.a_read.user == AReadUser::load_slot && w.shmem.a_read.slot == index;
    return slot.access.granted + (granted_now ? 1 : 0) < slot.access.rounds.size();
}

// An unserviced read: the warp has an lds in the dispatch pipeline, or holds
// a load slot with a round not yet granted.
bool Machine::warp_unserviced_read(const Wires& w, uint32_t warp) const {
    if (warp_dispatching(warp, true, false, false)) return true;
    for (uint32_t index = 0; index < load_slots_.size(); ++index) {
        const auto& slot = load_slots_[index].cur();
        if (slot.has_value() && slot->warp == warp && load_rounds_left(w, index)) return true;
    }
    return false;
}

// An unserviced write: the warp holds a store slot, or its SIMT RMW waits in
// the SIMT RMW register or runs in the engine with its final write not
// granted by the end of this cycle (a SIMT RMW counts as
// unserviced until its FINAL write grant).
bool Machine::warp_unserviced_write(const Wires& w, uint32_t warp) const {
    if (warp_dispatching(warp, false, true, false)) return true;
    for (uint32_t index = 0; index < store_slots_.size(); ++index) {
        const auto& slot = store_slots_[index].cur();
        if (slot.has_value() && slot->warp == warp && !store_slot_released(w, index)) return true;
    }
    if (simt_rmw_.cur().has_value() && simt_rmw_.cur()->warp == warp) return true;
    const auto& active = rmw_.cur().active;
    return active.has_value() && active->request.warp == warp && w.shmem.a_write.user != AWriteUser::rmw;
}

// fence.sh: every earlier shmem access of the warp has
// REACHED THE RAM by now (stores written, loads sampled) -- none unserviced
// after this cycle's grants (in the dispatch pipeline or in a slot), none
// granted now (that round reaches the RAM D cycles from now), and the latest
// round granted before is at the RAM now or was earlier (its count is 0). A
// probe needs no term: its warp issues nothing until the probe returns, D +
// Lb after its grant.
bool Machine::own_shmem_at_ram(const Wires& w, uint32_t warp) const {
    if (warp_unserviced_read(w, warp) || warp_unserviced_write(w, warp)) return false;
    if (w.shmem.a_read_warp == warp || w.shmem.a_write_warp == warp) return false;
    return shmem_to_ram_.cur().at(warp) == 0;
}

// lds / lds.uni: the address
// arrives now and the bank rounds come from it (active lanes only). The load
// takes the load slot admission secured; A-read serves it from the next
// cycle; its gathered vector waits in the slot for the RF write decision. No
// RF booking: the depbar completes it. lds.uni ignores the mask (its base is
// a uniform register; the mask gates only its RF write's lanes): a partial
// mask is the sanitizer's PANIC (in uniform_operand).
void Machine::dispatch_shared_load(const DispatchEntry& op, Wires& w) {
    const auto& inst = op.instruction;
    const bool uniform = inst.opcode == Opcode::lds_uni;
    auto rounds = uniform ? build_uniform_shmem_round(op) : build_shmem_rounds(op, true);
    const auto index = *free_load_slot(w);
    sanitizer_.check_simt_shmem_access(site(op), access_addresses(rounds), true);
    sanitizer_.record_simt_shmem_issue(site(op), SimtShmemHolder::load_slot, index, round_addresses(rounds),
        4, false);
    LoadSlot slot;
    slot.warp = op.warp;
    slot.uniform = uniform;
    slot.mask = op.active_mask;
    slot.access.rounds = std::move(rounds);
    slot.dst = inst.dst;
    slot.depbar = inst.control.set_depbar;
    slot.words.assign(config_.lanes, 0);
    w.dispatch.load_slot = std::make_pair(index, std::move(slot));
}

// sts: address and data arrive now (its RF reads were
// booked with the issue) into the store slot admission secured; A-write
// serves it from the next cycle. Two lanes on one word is UB (the
// sanitizer's PANIC); the machine just writes both, in round
// order, so the highest active lane's word lands.
void Machine::dispatch_shared_store(const DispatchEntry& op, Wires& w) {
    auto rounds = build_shmem_rounds(op, false);
    const auto index = *free_store_slot(w);
    const auto words = access_addresses(rounds);
    sanitizer_.check_store_word_collision(site(op), std::vector<uint64_t>(words.begin(), words.end()), "shmem");
    sanitizer_.check_simt_shmem_access(site(op), words, false);
    sanitizer_.record_simt_shmem_issue(site(op), SimtShmemHolder::store_slot, index, round_addresses(rounds),
        4, true);
    StoreSlot slot;
    slot.warp = op.warp;
    slot.access.rounds = std::move(rounds);
    slot.depbar = op.instruction.control.set_depbar;
    w.dispatch.store_slot = std::make_pair(index, std::move(slot));
}

// Each load slot, in one calculation: the old owner leaves when the RF write
// decision grants its write; dispatch installs a new owner (into a slot
// freed now, (d)); A-read grants its next round; a round's words come back
// into the gather register.
void Machine::drive_load_slots(const Wires& w) {
    for (uint32_t index = 0; index < load_slots_.size(); ++index) {
        const bool written = w.rf_write.granted_to(AsyncWriter::load_slot, index);
        const bool admitted = w.dispatch.load_slot.has_value() && w.dispatch.load_slot->first == index;
        const bool granted = w.shmem.a_read.user == AReadUser::load_slot && w.shmem.a_read.slot == index;
        const bool filled = w.shmem.load_fill.has_value() && w.shmem.load_fill->slot == index;
        if (!written && !admitted && !granted && !filled) continue;
        auto next = written ? std::nullopt : load_slots_[index].cur();
        if (admitted) {
            if (next.has_value()) throw SimError("internal lds admitted into a held load slot");
            next = w.dispatch.load_slot->second;
        }
        if ((granted || filled) && !next.has_value()) throw SimError("internal shmem round for an empty load slot");
        if (granted) ++next->access.granted;
        if (filled) {
            for (const auto& [lane, word] : w.shmem.load_fill->lanes) next->words.at(lane) = word;
            ++next->returned;
        }
        load_slots_[index].set_next(std::move(next), "drive_load_slots");
    }
}

// Each store slot: A-write grants its next round, and its last grant frees
// it -- for a dispatch in this same cycle (d).
void Machine::drive_store_slots(const Wires& w) {
    for (uint32_t index = 0; index < store_slots_.size(); ++index) {
        const bool granted = w.shmem.a_write.user == AWriteUser::store_slot && w.shmem.a_write.slot == index;
        const bool admitted = w.dispatch.store_slot.has_value() && w.dispatch.store_slot->first == index;
        if (!granted && !admitted) continue;
        auto next = store_slots_[index].cur();
        if (granted) {
            if (!next.has_value()) throw SimError("internal shmem round for an empty store slot");
            ++next->access.granted;
            if (next->access.serviced()) next.reset();
        }
        if (admitted) {
            if (next.has_value()) throw SimError("internal sts admitted into a held store slot");
            next = w.dispatch.store_slot->second;
        }
        store_slots_[index].set_next(std::move(next), "drive_store_slots");
    }
}

// The read pipeline: the round granted now enters stage 0; the round at the
// RAM now (stage D - 1) goes on carrying the words it sampled; the round at
// the output leaves.
void Machine::drive_a_read_pipe(const Wires& w) {
    if (w.shmem.a_read_beat.has_value()) a_read_pipe_.push(*w.shmem.a_read_beat, "drive_a_read_pipe");
    if (w.shmem.sampled.has_value()) {
        a_read_pipe_.replace(config_.shmem_port_a_depth - 1, w.shmem.sampled, "drive_a_read_pipe: the RAM's words");
    }
}

// The A-write and port-B write pipelines: the round granted now enters stage
// 0; the round at the last stage reached the RAM now.
void Machine::drive_write_pipes(const Wires& w) {
    if (w.shmem.a_write_beat.has_value()) a_write_pipe_.push(*w.shmem.a_write_beat, "drive_write_pipes: A-write");
    if (w.shmem.b_write_beat.has_value()) b_write_pipe_.push(*w.shmem.b_write_beat, "drive_write_pipes: port-B write");
}

void Machine::drive_shmem_arbiters(const Wires& w) {
    if (w.shmem.arbiters == shmem_arbiters_.cur()) return;
    shmem_arbiters_.set_next(w.shmem.arbiters, "drive_shmem_arbiters");
}

// fence.sh's per-warp count: a warp whose own round A-read or A-write grants
// now starts again at D - 1 (that round is at the RAM D cycles from now);
// every other count moves one cycle closer, down to 0.
void Machine::drive_shmem_to_ram(const Wires& w) {
    auto next = shmem_to_ram_.cur();
    for (uint32_t warp = 0; warp < next.size(); ++warp) {
        const bool granted = w.shmem.a_read_warp == warp || w.shmem.a_write_warp == warp;
        next[warp] = granted ? config_.shmem_port_a_depth - 1 : (next[warp] == 0 ? 0 : next[warp] - 1);
    }
    if (next == shmem_to_ram_.cur()) return;
    shmem_to_ram_.set_next(std::move(next), "drive_shmem_to_ram");
}

// ---------------------------------------------------------------------------
// SIMT shmem bank rounds (uniformity is declared, never
// detected). Every ACTIVE lane of an lds/sts is one access to bank
// (byte_address / 4) % shmem_banks, even when lanes name the same word; an
// inactive lane is no access at all. Round r serves, in
// every bank, that bank's r-th active lane in ascending lane order, so rounds
// = the most active lanes on any one bank. lds.uni is the declared broadcast.
// An sts with two lanes on one word (UB: the sanitizer's PANIC) writes both,
// in round order, so the higher lane's word lands last; one over a live mbar
// (UB: the sanitizer's live-mbar check) simply writes it.
// ---------------------------------------------------------------------------

std::vector<std::vector<Machine::LaneAccess>> Machine::build_shmem_rounds(const DispatchEntry& op, bool load) const {
    const auto& inst = op.instruction;
    const auto bases = capture(op, inst.src0);
    const auto data = load ? std::vector<uint32_t>(config_.lanes, 0) : capture(op, inst.src1);
    std::vector<std::vector<LaneAccess>> lanes_by_bank(config_.shmem_banks);
    for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
        if ((op.active_mask >> lane & 1) == 0) continue;
        const auto byte_address = shmem_address(op, bases[lane]);
        auto& bank_lanes = lanes_by_bank[(byte_address / 4) % config_.shmem_banks];
        bank_lanes.push_back({.lane = lane, .byte_address = byte_address, .value = data[lane]});
    }
    std::size_t round_count = 0;
    for (const auto& bank_lanes : lanes_by_bank) round_count = std::max(round_count, bank_lanes.size());
    std::vector<std::vector<LaneAccess>> rounds(round_count);
    for (const auto& bank_lanes : lanes_by_bank) {
        for (std::size_t round = 0; round < bank_lanes.size(); ++round) {
            rounds[round].push_back(bank_lanes[round]);
        }
    }
    return rounds;
}

// lds.uni: the address comes from a uniform register (as ldg.uni's does) and
// makes one access; serve_simt_round fans the word out to every lane.
std::vector<std::vector<Machine::LaneAccess>> Machine::build_uniform_shmem_round(const DispatchEntry& op) const {
    const auto base = uniform_operand(op, op.instruction.src0);
    return {{LaneAccess{.lane = 0, .byte_address = shmem_address(op, base)}}};
}

// One lane's address from its captured base register value and the immediate.
uint32_t Machine::shmem_address(const DispatchEntry& op, uint32_t base) const {
    const auto address = static_cast<int64_t>(base) + op.instruction.imm;
    if (address < 0 || address > std::numeric_limits<uint32_t>::max()) {
        panic(op, "shmem address arithmetic overflow");
    }
    const auto byte_address = static_cast<uint32_t>(address);
    if (byte_address % 4 != 0) panic(op, "unaligned 32-bit shmem access");
    if (static_cast<uint64_t>(byte_address) + 4 > shmem_.size()) {
        panic(op, "shmem access out of bounds");
    }
    return byte_address;
}

} // namespace nanosm

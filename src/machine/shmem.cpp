// Shared memory: ports A and B, the two sides (queues, collect and serve stages), the
// write-back buffer, the selects, bank rounds, and what issue sees of them. At the register
// file's async ports (src/machine/rf.cpp) the collect stages present reads, the write-back
// entries and the atomic's old-word register present writes.
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
//   A-read   the read side (lds, lds.uni, mbar probes), RMW reads, TMA reads
//   A-write  the write side's stores (sts), RMW writes
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
// The warps' instructions reach port A through two sides that share
// nothing, each strictly in order for all warps:
//
//   READ   lds, lds.uni, mbar.test_wait, mbar.try_wait
//          read queue -> collect -> serve (A-read, one access at a time)
//          -> a load's words into the write-back buffer -> the RF; a probe's
//          result to its predicate / wait slot
//   WRITE  sts, mbar.init, mbar.arrive, mbar.arrive.expect_tx, the atomics
//          write queue -> collect -> serve: a store's rounds on A-write, a
//          barrier change or atomic to the RMW engine (an atomic's old word
//          into the old-word register -> the RF)
//
// The queues are the memory queues of src/machine/dispatch.cpp: an
// instruction enters at its issue's edge, samples its predicate and uniform
// registers at issue + 1, and an lds / sts left with no lane leaves then
// (giving back its warp's counts too). The queue's head moves into the
// collect stage when that stage is free or moves on now; a try_wait installs
// its wait slot as it leaves. The collect stage reads the access's vector
// registers (an lds its address, an sts its address and data: the collect
// rule of src/machine/dispatch.cpp) through the RF's async read ports: it
// presents its next read (collect_rf_read) and takes the grant
// (take_collect_reads; the read of the last one lowers the instruction's read
// depbar); when they are at the RF output, it adds the immediate and plans
// the bank rounds, and the access moves into serve when serve is free or
// frees now. Issue asks only for its queue's credit and the warp's order
// (its counts, Q).
//
// The order of one warp's accesses: each side keeps it by construction;
// between the sides, issue holds a read while the warp has an unserviced
// write and a write while it has an unserviced read (an RMW is unserviced
// until its final write grant). fence.sh is admitted once the warp's count
// of accesses not at the RAM (a read's last sample, a write's last landing)
// is zero. Whether an engine command needed one is the sanitizer's question
// at the command's accept (missing fence.sh / race).
// ===========================================================================

// shmem this cycle, in wire order:
//   1. the RAM stage, from the pipelines' Q: the write rounds reaching the RAM
//      now and the completions they carry (an sts's depbar, an RMW's wake, a
//      placement's end, a warp's access at the RAM); the read round sampled
//      now; the read round at the port output, whose requester takes its
//      words now;
//   2. the selects, one per shared resource, in this order: A-write (an RMW's
//      write, a store round, and which transaction STARTS next: an RMW or a
//      store), the RMW engine's admission (which requester an RMW start
//      takes), A-read (sees that start: an RMW's read may be granted in its
//      start cycle), B-write;
//   3. their data paths: the RMW start, the granted rounds enter the
//      pipelines, the requesters advance;
//   4. the two sides' stages move (advance_shmem_sides).
void Machine::step_shmem(Wires& w) {
    w.shmem.arbiters = shmem_arbiters_.cur();
    shmem_ram_stage(w.shmem);
    commit_shmem_writes(w);
    deliver_shmem_read(w);
    w.shmem.a_write = select_a_write();
    w.shmem.rmw_start = select_rmw_admission(w);
    w.shmem.a_read = select_a_read(w);
    w.shmem.b_write = select_b_write(w);
    serve_rmw_start(w);
    serve_a_read(w);
    serve_a_write(w);
    serve_b_write(w);
    advance_tma_read(w);
    count_engine_wait_on_other(w);
    advance_shmem_sides(w);
}

// The RAM stage. The sample reads shmem through port A as it is now, before
// this cycle's writes land at the clock edge (READ_FIRST).
void Machine::shmem_ram_stage(ShmemCycle& shmem) {
    const auto& reads = a_read_pipe_.cur();
    shmem.delivered = reads.back();
    if (const auto& at_ram = reads.at(config_.shmem_port_a_depth - 1); at_ram.has_value()) {
        shmem.sampled = *at_ram;
        for (const auto address : at_ram->addresses) {
            shmem.sampled->words.push_back(read_shmem_word(address, shmem_port_a, "A-read sample"));
        }
        shmem.read_at_ram = at_ram->last_of_warp;
    }
    for (const auto* pipe : {&a_write_pipe_.cur(), &b_write_pipe_.cur()}) {
        if (pipe->back().has_value()) {
            const auto& writes = pipe->back()->writes;
            shmem.writes.insert(shmem.writes.end(), writes.begin(), writes.end());
        }
        if (pipe == &a_write_pipe_.cur()) shmem.port_a_writes = shmem.writes.size();
    }
}

// The memory word holding byte `byte_address`: its bank and row (shmem_).
Machine::ShmemWord Machine::shmem_word(uint32_t byte_address) const {
    const uint32_t word = byte_address / shmem_word_bytes_;
    return ShmemWord{word % config_.shmem_banks, word / config_.shmem_banks, byte_address % shmem_word_bytes_};
}

// The 4-byte word at `byte_address` (4-aligned) through read port `port` of
// its bank now.
uint32_t Machine::read_shmem_word(uint32_t byte_address, uint32_t port, const char* tag) {
    if (byte_address % 4 != 0 || uint64_t{byte_address} + 4 > shmem_bytes()) {
        throw SimError("internal: a shmem word read at " + std::to_string(byte_address));
    }
    const auto at = shmem_word(byte_address);
    const auto bytes = shmem_.read(at.bank, at.row, tag, port);
    uint32_t value = 0;
    std::memcpy(&value, bytes.data() + at.byte, sizeof(value));
    return value;
}

// The write rounds at the RAM now, through their ports (A-write's on port
// A, the port-B write's on port B): each piece's bytes enter their words with
// those bytes enabled, and land at the clock edge, port B's after port A's.
void Machine::land_shmem_writes(const Wires& w) {
    for (std::size_t index = 0; index < w.shmem.writes.size(); ++index) {
        const auto& write = w.shmem.writes[index];
        const uint32_t port = index < w.shmem.port_a_writes ? shmem_port_a : shmem_port_b;
        const char* tag = port == shmem_port_a ? "A-write" : "port-B write";
        if (uint64_t{write.address} + write.bytes.size() > shmem_bytes()) {
            throw SimError("internal: a shmem write past the end at " + std::to_string(write.address));
        }
        std::size_t byte = 0;
        while (byte < write.bytes.size()) {
            const auto at = shmem_word(write.address + static_cast<uint32_t>(byte));
            std::array<uint8_t, 8> values{};
            uint64_t enables = 0;
            for (uint32_t element = at.byte; element < shmem_word_bytes_ && byte < write.bytes.size();
                 ++element, ++byte) {
                values[element] = write.bytes[byte];
                enables |= uint64_t{1} << element;
            }
            shmem_.write(at.bank, at.row, std::span<const uint8_t>(values.data(), shmem_word_bytes_), enables, tag,
                port);
        }
    }
}

// What a write round carries to the RAM happens there: an sts's last round
// clears its depbar; a warp's write (an sts's last round, a SIMT RMW's write)
// has reached the RAM; an RMW write that flipped the phase wakes the waiters
// (at the commit, never at the grant); a placement's last round ends the
// TMA's load write (step_tma).
void Machine::commit_shmem_writes(Wires& w) {
    if (const auto& beat = a_write_pipe_.cur().back(); beat.has_value()) {
        if (const auto& done = beat->sts_done; done.has_value()) {
            record_depbar_completion(w, done->warp, done->bar, "depbar underflow on sb" + std::to_string(done->bar));
        }
        w.shmem.write_at_ram = beat->last_of_warp;
        if (beat->wake.has_value()) wake_mbar_waiters(w, beat->wake->address, beat->wake->phase);
    }
    if (const auto& beat = b_write_pipe_.cur().back(); beat.has_value() && beat->placed.has_value()) {
        w.shmem.tma_placed = beat->placed;
    }
}

// The read round at the port output goes to its requester: a load round to
// its write-back entry (each word to its lane; an lds.uni's one word to
// every lane), a probe to its warp, an RMW's words through its update into
// the ALU (an atomic's old word also into the old-word register), a
// store-source round to its TMA store register. (A descriptor round: the
// frontend latch, advance_tma_read.)
void Machine::deliver_shmem_read(Wires& w) {
    const auto& beat = w.shmem.delivered;
    if (!beat.has_value()) return;
    switch (beat->client) {
    case ReadClient::probe:
        w.shmem.probe_returned = beat->probe->warp;
        deliver_probe(*beat, w);
        return;
    case ReadClient::rmw:
        w.shmem.rmw_update = rmw_update(*beat);
        if (const auto& request = rmw_.cur().active->request;
            is_atomic_operation(request.operation) && request.warp.has_value()) {
            w.shmem.atomic_fill = beat->words.at(0);
        }
        return;
    case ReadClient::tma_store:
        fill_tma_store_register(*beat, w);
        return;
    case ReadClient::tma_frontend:
        return;
    case ReadClient::load:
        break;
    }
    WritebackFill fill{beat->slot, {}};
    if (beat->uniform) {
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) fill.lanes.emplace_back(lane, beat->words.at(0));
    } else {
        if (beat->lanes.size() != beat->words.size()) throw SimError("internal: a load round came back with the wrong words");
        for (std::size_t at = 0; at < beat->lanes.size(); ++at) fill.lanes.emplace_back(beat->lanes[at], beat->words[at]);
    }
    w.shmem.writeback_fill = std::move(fill);
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

// The read side's access is under way on A-read (started, rounds left).
bool Machine::read_side_in_service() const {
    const auto& serve = read_serve_.cur();
    return serve.has_value() && serve->access.started() && !serve->access.serviced();
}

// The write side's store is under way on A-write (started; a store whose
// last round is granted has left serve).
bool Machine::store_in_service() const {
    const auto& serve = write_serve_.cur();
    return serve.has_value() && !serve->rmw.has_value() && serve->access.started();
}

// A store transaction waits to start: the write side's serve holds a store
// not started.
bool Machine::store_waiting() const {
    const auto& serve = write_serve_.cur();
    return serve.has_value() && !serve->rmw.has_value() && !serve->access.started();
}

// The read side asks A-read for a new transaction: its serve holds an access
// not started -- a probe, or a load once a write-back entry is free (Q).
bool Machine::read_serve_requests() const {
    const auto& serve = read_serve_.cur();
    if (!serve.has_value() || serve->access.started()) return false;
    return serve->probe.has_value() || free_writeback_entry().has_value();
}

// The write-back entry a load's first grant takes: the first one empty in
// Q (an entry whose RF write is granted at t is free from t + 1).
std::optional<uint32_t> Machine::free_writeback_entry() const {
    for (uint32_t index = 0; index < writeback_.count(); ++index) {
        if (!writeback_[index].cur().has_value()) return index;
    }
    return std::nullopt;
}

// The write side's barrier change or atomic may start in the RMW engine: an
// atomic only with the old-word register empty (Q).
bool Machine::write_serve_rmw_ready() const {
    const auto& serve = write_serve_.cur();
    if (!serve.has_value() || !serve->rmw.has_value()) return false;
    return !is_atomic_operation(serve->rmw->operation) || !atomic_return_.cur().has_value();
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
//      idle, a requester held) and a store waiting to start (the write
//      side's store) alternate: when both wait, the one that did not start
//      last goes (the alternation bit, ShmemArbiters::rmw_started_last), so
//      one store transaction goes between consecutive RMWs; when one waits,
//      it goes. An RMW start (rmw_start) begins the PAUSE; the engine's
//      admission picks the requester. It may share its cycle with step 2's
//      round. A store start is the store's round 0.
// The alternation bit is read only here, to choose a START: from an RMW's
// start to its final write grant the PAUSE holds every new store, so no store
// is ever granted between an RMW's read and its write.
Machine::AWriteGrant Machine::select_a_write() const {
    AWriteGrant grant;
    const bool rmw_waiting = !store_pause() && rmw_requester_held();
    const bool stores_turn = store_waiting() && (!rmw_waiting || shmem_arbiters_.cur().rmw_started_last);
    grant.rmw_start = rmw_waiting && !stores_turn;
    if (rmw_alu_.last().has_value()) { // (the PAUSE is set: no RMW starts)
        grant.user = AWriteUser::rmw;
        return grant;
    }
    if (store_in_service()) {
        grant.user = AWriteUser::store;
        return grant;
    }
    if (store_pause() || !stores_turn) return grant;
    grant.user = AWriteUser::store;
    grant.store_start = true;
    return grant;
}

// A-read, in order:
//   1. a transaction under way takes its next round (no preemption);
//   2. a started RMW waiting for its read (started earlier, or starting now)
//      goes before every new requester. It is granted strictly after the
//      last round of the store transaction under way on A-write, if any:
//      matched D then lands that store before the RMW's sample. Until then
//      nothing new starts on A-read (with no preemption a new multi-round
//      read could only delay the RMW's read and so lengthen the store
//      pause);
//   3. ONE round-robin over the new requesters: the read side's access (a
//      load with a free write-back entry, or a probe), the TMA's new read.
// Reads are never paused: between an RMW's read and its write, step 3 serves
// loads, probes and TMA reads as usual.
Machine::AReadGrant Machine::select_a_read(const Wires& w) const {
    if (tma_read_in_service()) return AReadGrant::to(AReadUser::tma);
    if (read_side_in_service()) return AReadGrant::to(AReadUser::read_side);
    if (rmw_awaits_read(w)) return store_in_service() ? AReadGrant{} : AReadGrant::to(AReadUser::rmw);
    return a_read_round_robin(w);
}

Machine::AReadGrant Machine::a_read_round_robin(const Wires& w) const {
    const std::vector<uint8_t> ready{static_cast<uint8_t>(read_serve_requests() ? 1 : 0),
        static_cast<uint8_t>(w.tma.a_read.has_value() ? 1 : 0)};
    const auto choice = round_robin_grant(ready, shmem_arbiters_.cur().a_read, 1);
    if (choice.granted.empty()) return {};
    AReadGrant grant;
    if (choice.granted.front() == 0) {
        grant = AReadGrant::to(AReadUser::read_side);
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
// store-source read starting now reserves its TMA store register.
void Machine::serve_a_read(Wires& w) {
    auto& shmem = w.shmem;
    const auto& grant = shmem.a_read;
    if (grant.pointer_after.has_value()) shmem.arbiters.a_read = *grant.pointer_after;
    switch (grant.user) {
    case AReadUser::none:
        return;
    case AReadUser::read_side:
        shmem.a_read_beat = read_side_round(w);
        return;
    case AReadUser::rmw: { // the engine's operation, or the one starting now: one round
        const auto& active = rmw_.cur().active.has_value() ? *rmw_.cur().active : *shmem.rmw_started;
        trace_->shmem_grant(cycle_, ShmemService::a_read, ShmemClient::rmw);
        ReadBeat beat;
        beat.client = ReadClient::rmw;
        const auto address = active.request.byte_address;
        // An mbar operation reads both mbar words; an atomic its one word.
        beat.addresses = is_atomic_operation(active.request.operation) ? std::vector<uint32_t>{address}
                                                                       : std::vector<uint32_t>{address, address + 4};
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

// The read side's next round: a load round's word addresses (with their
// lanes and its write-back entry: the first round reserves one), or the
// probe's two mbar words. The last round services the warp's read; at the
// RAM, D cycles from now, it has reached the RAM.
Machine::ReadBeat Machine::read_side_round(Wires& w) {
    const auto& serve = *read_serve_.cur();
    const auto round = serve.access.granted;
    const auto rounds = serve.access.rounds;
    ReadBeat beat;
    const bool last = round + 1 == rounds;
    if (serve.probe.has_value()) {
        beat.client = ReadClient::probe;
        beat.probe = serve.probe;
        beat.addresses = {serve.probe->byte_address, serve.probe->byte_address + 4};
        trace_->shmem_grant(cycle_, ShmemService::a_read, ShmemClient::probe);
    } else {
        if (round == 0) {
            trace_->shmem_begin(ShmemService::a_read, cycle_, serve.uniform ? "lds.uni" : "lds",
                "warp=" + std::to_string(serve.warp) + ", rounds=" + std::to_string(rounds), TraceCategory::shmem);
            w.shmem.writeback_reserved = free_writeback_entry().value();
        }
        trace_->shmem_grant(cycle_, ShmemService::a_read, ShmemClient::simt);
        if (last) trace_->shmem_end(ShmemService::a_read, cycle_ + 1);
        beat.client = ReadClient::load;
        beat.slot = round == 0 ? *w.shmem.writeback_reserved : serve.writeback.value();
        beat.uniform = serve.uniform;
        beat.lanes = round_lanes(serve.access, round);
        for (const auto lane : beat.lanes) beat.addresses.push_back(serve.access.address[lane]);
    }
    sanitizer_.record_simt_shmem_grant(cycle_, SimtShmemHolder::read_side, 0, cycle_ + config_.shmem_port_a_depth);
    if (last) {
        w.shmem.read_serviced = serve.warp;
        beat.last_of_warp = serve.warp;
    }
    return beat;
}

// The granted A-write round enters the A-write pipeline. Every start sets the
// alternation bit (an RMW start: true; a store start: false). An RMW's write
// takes its update from the ALU's last stage; a phase flip wakes the waiters
// when the write reaches the RAM.
void Machine::serve_a_write(Wires& w) {
    auto& shmem = w.shmem;
    const auto& grant = shmem.a_write;
    if (grant.rmw_start) {
        shmem.arbiters.rmw_started_last = true;
    } else if (grant.store_start) {
        shmem.arbiters.rmw_started_last = false;
    }
    if (store_pause() && store_waiting()) trace_->shmem_store_pause_wait();
    switch (shmem.a_write.user) {
    case AWriteUser::none:
        return;
    case AWriteUser::store:
        shmem.a_write_beat = store_round();
        if (shmem.a_write_beat->sts_done.has_value()) shmem.write_serviced = write_serve_.cur()->warp;
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
            trace_->mbar_rmw_end(cycle_ + 1, update.wake_phase.has_value());
        }
        shmem.a_write_beat = rmw_write_round(update);
        // A SIMT RMW's final write: its warp's write is serviced now and
        // reaches the RAM D cycles from now.
        if (const auto warp = request.warp; warp.has_value()) {
            sanitizer_.record_simt_shmem_grant(cycle_, SimtShmemHolder::write_side, 0, cycle_ + config_.shmem_port_a_depth);
            shmem.write_serviced = warp;
            shmem.a_write_beat->last_of_warp = warp;
        }
        return;
    }
    }
}

// The store's next round: its lanes' words enter the A-write pipeline; the
// last round carries the depbar completion and its warp to the RAM.
Machine::WriteBeat Machine::store_round() {
    const auto& serve = *write_serve_.cur();
    const auto round = serve.access.granted;
    const auto rounds = serve.access.rounds;
    if (round == 0) {
        trace_->shmem_begin(ShmemService::a_write, cycle_, "sts",
            "warp=" + std::to_string(serve.warp) + ", rounds=" + std::to_string(rounds), TraceCategory::shmem);
    }
    trace_->shmem_grant(cycle_, ShmemService::a_write, ShmemClient::simt);
    // (The sanitizer checks here too that the round does not write a live mbar.)
    sanitizer_.record_simt_shmem_grant(cycle_, SimtShmemHolder::write_side, 0, cycle_ + config_.shmem_port_a_depth);
    WriteBeat beat;
    for (const auto lane : round_lanes(serve.access, round)) {
        ShmemWrite write{serve.access.address[lane], {}};
        for (uint32_t byte = 0; byte < 4; ++byte) write.bytes.push_back(static_cast<uint8_t>(serve.data[lane] >> (8 * byte)));
        beat.writes.push_back(write);
    }
    if (round + 1 == rounds) {
        beat.sts_done = DepbarEvent{serve.warp, serve.depbar};
        beat.last_of_warp = serve.warp;
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

// ===========================================================================
// The two sides' stages
// ===========================================================================

// The two sides after this cycle's grants. On each side, in order: serve
// frees (a read's last round granted now; a store's last round granted now,
// or a barrier change / atomic taken by the RMW engine now); the collect
// stage's access moves into serve when its registers are at the RF output
// and serve is empty or frees now (the immediate added and the rounds
// planned on the way: read_serve_from / write_serve_from); the queue's head
// moves into the collect stage when it may leave (head_may_leave: a head its
// sample, later this cycle, leaves with no lane does not) and that stage is
// empty or moves on now. Each step uses only Q and this cycle's grants.
void Machine::advance_shmem_sides(Wires& w) {
    auto& shmem = w.shmem;
    const auto& read = read_serve_.cur();
    const auto& write = write_serve_.cur();
    const std::array<bool, shmem_sides> serve_frees{
        !read.has_value() ||
            (shmem.a_read.user == AReadUser::read_side && read->access.granted + 1 == read->access.rounds),
        !write.has_value() || shmem.simt_rmw_started ||
            (shmem.a_write.user == AWriteUser::store && write->access.granted + 1 == write->access.rounds)};
    for (const auto queue : shmem_side_queues) {
        const auto side = queue_index(queue);
        const bool write_side = queue == MemoryQueue::shmem_write;
        const auto& collect = collects_[side].cur();
        if (collect.has_value() && collect_ready(*collect) && serve_frees[side]) {
            if (write_side) {
                shmem.write_serve_in = write_serve_from(*collect);
            } else {
                shmem.read_serve_in = read_serve_from(*collect);
            }
            shmem.collect_out[side] = true;
        }
        if (!head_may_leave(queue)) continue;
        if (collect.has_value() && !shmem.collect_out[side]) {
            trace_->shmem_head_wait(write_side);
            continue;
        }
        pop_queue(w, queue);
        const auto& head = queues_[side].cur().front();
        if (is_mbar_try_wait(head.instruction.opcode)) {
            shmem.wait_slot = IntoSlot<MbarWaitSlot>{head.warp, wait_slot_for(head)};
        }
    }
}

// A collect stage's access has its registers at the RF output now: an lds
// its address, an sts its address and its data (no RF read: at once).
bool Machine::collect_ready(const ShmemCollect& collect) const {
    const auto opcode = collect.entry.instruction.opcode;
    if (opcode == Opcode::lds) return collect.address_read && collect.wait == 0;
    if (opcode == Opcode::sts) return collect.address_read && collect.data_read && collect.wait == 0;
    return true;
}

// The read side's access entering serve: an lds's lanes' addresses (base +
// immediate) and bank rounds, an lds.uni's one access, or a probe.
Machine::ReadServe Machine::read_serve_from(const ShmemCollect& collect) const {
    const auto& entry = collect.entry;
    const auto& inst = entry.instruction;
    ReadServe serve;
    serve.warp = entry.warp;
    serve.pc = entry.pc;
    serve.instruction = inst;
    if (is_mbar_predicate_writer(inst.opcode)) {
        MbarReadRequest request;
        request.test = is_mbar_test_wait(inst.opcode);
        request.warp = entry.warp;
        request.predicate_dst = inst.dst;
        request.byte_address = entry_mbar_address(entry);
        request.parity = entry_parity(entry);
        request.pc = entry.pc;
        request.instruction = inst;
        serve.probe = request;
        serve.access.rounds = 1;
        return serve;
    }
    serve.uniform = inst.opcode == Opcode::lds_uni;
    serve.access = serve.uniform ? *uniform_shmem_round(entry, entry_uniform(entry, inst.src0), true)
                                 : *shmem_rounds(entry, collect.address, true);
    serve.dst = inst.dst;
    serve.mask = entry.lanes;
    serve.depbar = inst.control.set_depbar;
    return serve;
}

// The write side's access entering serve: an sts's addresses, rounds and
// each accessing lane's word; or a barrier change / atomic's request.
Machine::WriteServe Machine::write_serve_from(const ShmemCollect& collect) const {
    const auto& entry = collect.entry;
    const auto& inst = entry.instruction;
    WriteServe serve;
    serve.warp = entry.warp;
    serve.pc = entry.pc;
    serve.instruction = inst;
    serve.depbar = inst.control.set_depbar;
    if (inst.opcode != Opcode::sts) {
        serve.rmw = entry_rmw_request(entry);
        serve.dst = inst.dst;
        serve.mask = entry.lanes;
        return serve;
    }
    // Two lanes on one word is UB (the sanitizer's PANIC); the machine just
    // writes both, in round order, so the highest active lane's word lands.
    serve.access = *shmem_rounds(entry, collect.address, true);
    for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
        if ((serve.access.lanes >> lane & 1) != 0) serve.data[lane] = collect.data[lane];
    }
    return serve;
}

// A try_wait's wait slot (its warp's), installed as it leaves the queue: a
// wake that comes while its probe travels finds it.
Machine::MbarWaitSlot Machine::wait_slot_for(const QueuedEntry& entry) const {
    MbarWaitSlot slot;
    slot.byte_address = entry_mbar_address(entry);
    slot.parity = entry_parity(entry);
    slot.predicate_dst = entry.instruction.dst;
    return slot;
}

// ---- The sides' and the write-back buffer's RF ports ----

// A collect stage's RF read request, from Q (next_collect_register): the
// read side's lds address; the write side's sts address and data, a pair
// (its data wanted with its address in one cycle).
std::optional<Machine::RfReadRequest> Machine::collect_rf_read(MemoryQueue side) const {
    const bool write_side = side == MemoryQueue::shmem_write;
    const auto& collect = collects_[queue_index(side)].cur();
    if (!collect.has_value()) return std::nullopt;
    const auto& inst = collect->entry.instruction;
    if (inst.opcode != (write_side ? Opcode::sts : Opcode::lds)) return std::nullopt;
    const auto next = [&](bool with_address) {
        return next_collect_register(inst, write_side, collect->address_read, collect->data_read, with_address);
    };
    const auto reg = next(false);
    if (!reg.has_value()) return std::nullopt;
    return RfReadRequest{collect->entry.warp, *reg, next(true)};
}

// Each collect stage takes its RF read grant: a denied request is a bank
// wait (T); a granted register is read-owned no longer (the sanitizer) and
// its words latch as the collect rule says (collect_read: the address, or
// the data, or both for one register); the read of its last vector register
// -- an sts's data, an lds's address -- lowers the instruction's read
// depbar. The read side first, then the write side.
void Machine::take_collect_reads(Wires& w) {
    for (const auto side : shmem_side_queues) {
        const auto request = collect_rf_read(side);
        if (!request.has_value()) continue;
        const auto& grant = w.rf_read.granted[collect_reader(side)];
        if (!grant.words.has_value()) {
            trace_->shmem_bank_wait(side == MemoryQueue::shmem_write);
            continue;
        }
        const auto& stage = *collects_[queue_index(side)].cur();
        const bool write_side = side == MemoryQueue::shmem_write;
        auto& sampled = w.collect_sampled[queue_index(side)];
        const auto latch = [&](uint8_t reg, const LaneWords& words, bool named) {
            sanitizer_.record_async_reader_done(stage.entry.warp, reg, AsyncReader::shmem_collect);
            const auto got = collect_read(stage.entry.instruction, write_side, stage.address_read, named);
            if (got.address) sampled.address = words;
            if (got.data) sampled.data = words;
        };
        latch(request->reg, *grant.words, true);
        if (grant.pair_words.has_value()) latch(*request->pair, *grant.pair_words, false);
    }
    for (const auto side : shmem_side_queues) {
        const auto& sampled = w.collect_sampled[queue_index(side)];
        if (!(side == MemoryQueue::shmem_write ? sampled.data : sampled.address).has_value()) continue;
        const auto& entry = collects_[queue_index(side)].cur()->entry;
        lower_read_depbar(w, entry.warp, entry.instruction);
    }
}

// The write-back entries' RF write requests, one per entry: a complete
// entry's load result, its lanes enabled, its depbar cleared as it lands.
std::vector<std::optional<Machine::RfWrite>> Machine::writeback_rf_writes() const {
    std::vector<std::optional<RfWrite>> writes(writeback_.count());
    for (uint32_t index = 0; index < writeback_.count(); ++index) {
        const auto& entry = writeback_[index].cur();
        if (!entry.has_value() || !entry->complete()) continue;
        writes[index] = RfWrite{entry->warp, entry->dst, entry->words, entry->mask, entry->depbar, {entry->dst}};
    }
    return writes;
}

// The old-word register's RF write request, once its word is in: the word to
// every lane of the atomic's lanes, its depbar cleared as it lands.
std::optional<Machine::RfWrite> Machine::atomic_return_rf_write() const {
    const auto& old = atomic_return_.cur();
    if (!old.has_value() || !old->word.has_value()) return std::nullopt;
    return RfWrite{old->warp, old->dst, LaneWords(config_.lanes, *old->word), old->mask, old->depbar, {old->dst}};
}

// The sanitizer's record of a queued access at its issue + 1, with the
// words it will touch: what the collect stage reads later is the register as
// it stands now (its RAW was checked against this sample, and a write landing
// before the collect stage's read is a WAR PANIC), and the uniform operands
// were just sampled. An address the machine will refuse at its collect stage
// (overflow, alignment, bounds: the machine's PANIC there) is not recorded.
// Pure reads: nothing here feeds the machine.
void Machine::record_queued_access(const QueuedEntry& entry) {
    const auto& inst = entry.instruction;
    const auto at = site(entry);
    const auto operand = [&](uint8_t reg) { return entry_uniform(entry, reg); };
    switch (inst.opcode) {
    case Opcode::lds:
    case Opcode::lds_uni:
    case Opcode::sts: {
        const auto access = inst.opcode == Opcode::lds_uni
            ? uniform_shmem_round(entry, operand(inst.src0), false)
            : shmem_rounds(entry, rf_peek(entry.warp, inst.src0), false);
        if (!access.has_value()) return;
        const auto words = access_addresses(*access);
        const bool store = inst.opcode == Opcode::sts;
        if (store) sanitizer_.check_store_word_collision(at, std::vector<uint64_t>(words.begin(), words.end()), "shmem");
        sanitizer_.check_simt_shmem_access(at, words, !store);
        sanitizer_.record_simt_shmem_issue(at, store ? SimtShmemHolder::write_side : SimtShmemHolder::read_side, 0,
            round_addresses(*access), 4, store);
        return;
    }
    default:
        break;
    }
    const uint32_t base = operand(inst.src0);
    if (is_atomic(inst.opcode)) {
        const auto address = shmem_address(entry, base, false);
        if (!address.has_value()) return;
        sanitizer_.check_simt_shmem_access(at, std::vector<uint32_t>{*address}, false);
        // The shadow of the warp's access: one 4-byte write round, reaching
        // the RAM at its write grant + D.
        sanitizer_.record_simt_shmem_issue(at, SimtShmemHolder::write_side, 0,
            std::vector<std::vector<uint32_t>>{{*address}}, 4, true);
        return;
    }
    if (base % 8 != 0 || static_cast<uint64_t>(base) + 8 > shmem_bytes()) return;
    if (is_mbar_predicate_writer(inst.opcode)) {
        sanitizer_.check_simt_mbar_probe(at, base);
        sanitizer_.record_simt_shmem_issue(at, SimtShmemHolder::read_side, 0,
            std::vector<std::vector<uint32_t>>{{base}}, 8, false);
        return;
    }
    // (An mbar.init makes its mbar live at its RMW start, in the write side's
    // order: serve_rmw_start.)
    sanitizer_.check_simt_mbar_rmw(at, base);
    sanitizer_.record_simt_shmem_issue(at, SimtShmemHolder::write_side, 0,
        std::vector<std::vector<uint32_t>>{{base}}, 8, true);
}

// ---- The registers' next values ----

// Each collect stage, in one calculation: its access leaves for serve; the
// queue's head enters (with its issue + 1 sample if it takes it now); the RF
// samples granted now latch (at the RF output Lr cycles later: `wait`); the
// wait counts down.
void Machine::drive_shmem_collects(const Wires& w) {
    for (const auto queue : shmem_side_queues) {
        const auto side = queue_index(queue);
        auto& reg = collects_[side];
        const bool out = w.shmem.collect_out[side];
        const bool pop = w.queue_pops[side];
        const auto& [address, data] = w.collect_sampled[side];
        const bool sampled = address.has_value() || data.has_value();
        const bool counting = reg.cur().has_value() && reg.cur()->wait != 0;
        if (!out && !pop && !sampled && !counting) continue;
        auto next = reg.cur();
        if (next.has_value() && next->wait != 0) --next->wait;
        if (out) next.reset();
        if (pop) {
            if (next.has_value()) throw SimError("internal: a queue's head entered a busy collect stage");
            next = ShmemCollect{leaving_head(queue, w), false, false, 0, {}, {}};
        }
        if (sampled) {
            if (!next.has_value() || out || pop) throw SimError("internal: an RF sample for a collect stage that moved");
            if (address.has_value()) {
                next->address = *address;
                next->address_read = true;
            }
            if (data.has_value()) {
                next->data = *data;
                next->data_read = true;
            }
            next->wait = static_cast<uint8_t>(config_.rf_read_latency_cycles - 1);
        }
        reg.set_next(std::move(next), "drive_shmem_collects");
    }
}

// Each serve stage: A-read / A-write grants its next round (the read side's
// first also names its write-back entry), the last frees it; the write
// side's barrier change or atomic leaves when the RMW engine takes it; the
// collect stage's access enters (into a stage freed now).
void Machine::drive_shmem_serves(const Wires& w) {
    {
        const bool granted = w.shmem.a_read.user == AReadUser::read_side;
        const auto& in = w.shmem.read_serve_in;
        if (granted || in.has_value()) {
            auto next = read_serve_.cur();
            if (granted) {
                if (!next.has_value()) throw SimError("internal: an A-read round for an empty read side");
                if (!next->access.started() && !next->probe.has_value()) next->writeback = w.shmem.writeback_reserved;
                ++next->access.granted;
                if (next->access.serviced()) next.reset();
            }
            if (in.has_value()) {
                if (next.has_value()) throw SimError("internal: an access entered a busy read serve stage");
                next = in;
            }
            read_serve_.set_next(std::move(next), "drive_shmem_serves: read");
        }
    }
    const bool granted = w.shmem.a_write.user == AWriteUser::store;
    const bool started = w.shmem.simt_rmw_started;
    const auto& in = w.shmem.write_serve_in;
    if (!granted && !started && !in.has_value()) return;
    auto next = write_serve_.cur();
    if (granted) {
        if (!next.has_value()) throw SimError("internal: an A-write round for an empty write side");
        ++next->access.granted;
        if (next->access.serviced()) next.reset();
    }
    if (started) next.reset();
    if (in.has_value()) {
        if (next.has_value()) throw SimError("internal: an access entered a busy write serve stage");
        next = in;
    }
    write_serve_.set_next(std::move(next), "drive_shmem_serves: write");
}

// Each write-back entry, in one calculation: the RF write decision grants
// its write (it leaves); a load's first grant reserves it; a round's words
// come back into it.
void Machine::drive_writeback(const Wires& w) {
    for (uint32_t index = 0; index < writeback_.count(); ++index) {
        const bool written = w.rf_write.granted.writeback[index] != 0;
        const bool reserved = w.shmem.writeback_reserved == index;
        const bool filled = w.shmem.writeback_fill.has_value() && w.shmem.writeback_fill->entry == index;
        if (!written && !reserved && !filled) continue;
        auto next = written ? std::nullopt : writeback_[index].cur();
        if (reserved) {
            if (next.has_value()) throw SimError("internal: a held write-back entry reserved again");
            const auto& serve = *read_serve_.cur();
            WritebackEntry entry;
            entry.warp = serve.warp;
            entry.dst = serve.dst;
            entry.mask = serve.mask;
            entry.depbar = serve.depbar;
            entry.rounds = serve.access.rounds;
            entry.words.assign(config_.lanes, 0);
            next = std::move(entry);
        }
        if (filled) {
            if (!next.has_value()) throw SimError("internal: a load round for an empty write-back entry");
            for (const auto& [lane, word] : w.shmem.writeback_fill->lanes) next->words.at(lane) = word;
            ++next->returned;
        }
        writeback_[index].set_next(std::move(next), "drive_writeback");
    }
}

// The atomic's old-word register: its RF write granted (it leaves), an
// atomic's RMW start reserving it, its read's old word.
void Machine::drive_atomic_return(const Wires& w) {
    const bool written = w.rf_write.granted.atomic_return != 0;
    if (!written && !w.shmem.atomic_reserved.has_value() && !w.shmem.atomic_fill.has_value()) return;
    auto next = written ? std::nullopt : atomic_return_.cur();
    if (w.shmem.atomic_reserved.has_value()) {
        if (next.has_value()) throw SimError("internal: the old-word register reserved while held");
        next = w.shmem.atomic_reserved;
    }
    if (w.shmem.atomic_fill.has_value()) {
        if (!next.has_value()) throw SimError("internal: an old word for an empty old-word register");
        next->word = w.shmem.atomic_fill;
    }
    atomic_return_.set_next(std::move(next), "drive_atomic_return");
}

// Each warp's counts: issue's access adds one to its side and to the count
// not at the RAM; a last round's grant (an RMW's final write) services one;
// a last round at the RAM takes one off that count; a dropped entry gives
// back both.
void Machine::drive_shmem_counts(const Wires& w) {
    const auto& cur = shmem_counts_.cur();
    for (uint32_t warp = 0; warp < cur.size(); ++warp) {
        auto next = cur[warp];
        if (w.shmem.probe_returned == warp) {
            if (!next.probe) throw SimError("internal: warp " + std::to_string(warp) + "'s probe returned but was not out");
            next.probe = false;
        }
        if (const auto& issued = w.issue.queued;
            issued.has_value() && issued->queue != MemoryQueue::lsu && issued->entry.warp == warp) {
            ++(issued->queue == MemoryQueue::shmem_write ? next.writes : next.reads);
            ++next.not_at_ram;
            if (is_mbar_predicate_writer(issued->entry.instruction.opcode)) next.probe = true;
        }
        const auto down = [&](uint32_t& count) {
            if (count == 0) throw SimError("internal: warp " + std::to_string(warp) + "'s shmem count underflow");
            --count;
        };
        if (w.shmem.read_serviced == warp) down(next.reads);
        if (w.shmem.write_serviced == warp) down(next.writes);
        if (w.shmem.read_at_ram == warp) down(next.not_at_ram);
        if (w.shmem.write_at_ram == warp) down(next.not_at_ram);
        if (const auto& sample = w.queue_sample; sample.has_value() && sample->queue != MemoryQueue::lsu &&
            sample->dropped && sample->warp == warp) {
            down(sample->queue == MemoryQueue::shmem_write ? next.writes : next.reads);
            down(next.not_at_ram);
        }
        if (next != cur[warp]) shmem_counts_.set_next(warp, next, "drive_shmem_counts");
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

// ---------------------------------------------------------------------------
// SIMT shmem bank rounds (uniformity is declared, never
// detected). Every lane of an lds/sts (the active lanes AND its write mask)
// is one access to bank (byte_address / 4) % shmem_banks, even when lanes
// name the same word; a lane that is off is no access at all. Round r
// serves, in every bank, that bank's r-th accessing lane in ascending lane
// order, so rounds = the most accessing lanes on any one bank. lds.uni is the
// declared broadcast. An sts with two lanes on one word (UB: the sanitizer's
// PANIC) writes both, in round order, so the higher lane's word lands last;
// one over a live mbar (UB: the sanitizer's live-mbar check) simply writes
// it.
// ---------------------------------------------------------------------------

std::optional<Machine::SlotRounds> Machine::shmem_rounds(
    const QueuedEntry& entry, const LaneWords& bases, bool checked) const {
    SlotRounds access;
    std::array<uint32_t, limits::shmem_banks> lanes_on_bank{};
    for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
        if ((entry.lanes >> lane & 1) == 0) continue;
        const auto byte_address = shmem_address(entry, bases[lane], checked);
        if (!byte_address.has_value()) return std::nullopt;
        auto& on_bank = lanes_on_bank[(*byte_address / 4) % config_.shmem_banks];
        access.lanes |= uint64_t{1} << lane;
        access.address[lane] = *byte_address;
        access.round[lane] = static_cast<uint8_t>(on_bank++);
        access.rounds = std::max(access.rounds, on_bank);
    }
    return access;
}

// lds.uni: the address comes from a uniform register (as ldg.uni's does) and
// makes one access, lane 0's; its word goes to every lane (deliver_shmem_read).
std::optional<Machine::SlotRounds> Machine::uniform_shmem_round(
    const QueuedEntry& entry, uint32_t base, bool checked) const {
    const auto address = shmem_address(entry, base, checked);
    if (!address.has_value()) return std::nullopt;
    SlotRounds access;
    access.lanes = 1;
    access.address[0] = *address;
    access.rounds = 1;
    return access;
}

// A round's lanes, one per bank at most, in bank order: bank b's lane is the
// one of its lanes whose round it is.
Bounded<uint8_t, limits::shmem_banks> Machine::round_lanes(const SlotRounds& access, uint32_t round) const {
    std::array<int, limits::shmem_banks> lane_on_bank;
    lane_on_bank.fill(-1);
    for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
        if (access.in_round(lane, round)) lane_on_bank[(access.address[lane] / 4) % config_.shmem_banks] = static_cast<int>(lane);
    }
    Bounded<uint8_t, limits::shmem_banks> lanes;
    for (uint32_t bank = 0; bank < config_.shmem_banks; ++bank) {
        if (lane_on_bank[bank] >= 0) lanes.push_back(static_cast<uint8_t>(lane_on_bank[bank]));
    }
    return lanes;
}

// One lane's address from its base and the immediate (the collect stage's
// adder). Checked, a bad address is the machine's PANIC; unchecked, none.
std::optional<uint32_t> Machine::shmem_address(const QueuedEntry& entry, uint32_t base, bool checked) const {
    const auto address = static_cast<int64_t>(base) + entry.instruction.imm;
    const char* problem = nullptr;
    if (address < 0 || address > std::numeric_limits<uint32_t>::max()) {
        problem = "shmem address arithmetic overflow";
    } else if (address % 4 != 0) {
        problem = "unaligned 32-bit shmem access";
    } else if (static_cast<uint64_t>(address) + 4 > shmem_bytes()) {
        problem = "shmem access out of bounds";
    }
    if (problem == nullptr) return static_cast<uint32_t>(address);
    if (checked) panic(entry, problem);
    return std::nullopt;
}

} // namespace nanosm

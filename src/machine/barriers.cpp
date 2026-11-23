// Barriers (mbar): probes, the read-modify-write engine, atomics, and the try-wait slots.
// (Machine is one class, src/machine/machine.hpp; the map of its files is at the top of
// src/machine/machine.cpp.)

#include "machine/machine.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

namespace nanosm {

// ===========================================================================
// The RMW engine: one operation at a time, each one
// coherent 8-byte mbar RMW (both words in one read round, one write round)
// or one uniform atomic (one 4-byte word, read and written
// the same way). Its requester waits in a register (the SIMT RMW register,
// the TMA's or the MMA's event register). One RMW:
//   s          START. A-write's alternation lets an RMW begin
//              (select_a_write step 4) and the engine takes a requester
//              (select_rmw_admission). The global store PAUSE -- the engine's
//              operation register -- is set from s + 1: no new store
//              transaction starts. A store transaction under way keeps its
//              rounds.
//   r >= s     READ, granted on A-read before any new requester, strictly
//              after that store transaction's last round (r = s when no store
//              and no A-read transaction is under way)
//   r + D      the RAM sample
//   r + D + Lb the words at the port output; the ALU computes the update (an
//              atomic's old word also goes to its return load slot now)
//   w = r + D + Lb + C
//              WRITE (the final write) granted on A-write; the PAUSE ends at
//              this cycle's edge
//   w + 1      stores may start again: one granted now reaches the RAM at
//              w + 1 + D, after the RMW's write (matched D, no drain)
//   w + D      the write reaches the RAM; a phase flip wakes the waiters now
// Reads are never paused: between r and w A-read serves loads, probes and
// TMA reads, and a read of the mbar sampling at w + D sees the old word
// (READ_FIRST). An atomic's old value waits in its return slot for the RF
// write decision, so rD may be written before w; the warp's next shmem
// access still waits for w (its unserviced write), fence.sh for w + D.
// (D, Lb and C are shared memory's depths: the top of src/machine/shmem.cpp.)
// ===========================================================================

// A probe's words at the port output. test_wait writes its predicate now
// (write-first (a)); its warp issues again now (b). try_wait completes its
// wait slot or parks it. The probing warp was blocked while the probe
// travelled, so if its slot is gone, completed now (commit_shmem_writes ran
// first) or another warp's, a wake ended the wait first: the stale probe does
// nothing.
void Machine::deliver_probe(const ReadBeat& beat, Wires& w) {
    const auto& request = *beat.probe;
    auto& warp = warps_.at(request.warp);
    const auto state = decode_mbar_state(static_cast<uint64_t>(beat.words.at(0)) |
        static_cast<uint64_t>(beat.words.at(1)) << 32);
    if (!state.valid || state.expected == 0 || state.remaining > state.expected) {
        throw SimError(
            "PANIC cycle=" + std::to_string(cycle_) + " warp=" + std::to_string(warp.id) +
            " pc=" + std::to_string(request.pc) +
            " reason=wait/test on an uninitialized or corrupt mbar inst='" +
            format_instruction(request.instruction) + "'");
    }
    const bool complete = state.phase != (request.parity != 0);
    if (request.test) {
        // The predicate file (flip-flops) takes it at this cycle's edge; the
        // warp issues from the next cycle (probe_in_flight).
        Writeback predicate;
        predicate.port_cycle = cycle_;
        predicate.land_cycle = cycle_;
        predicate.warp = warp.id;
        predicate.predicate_dst = request.predicate_dst;
        predicate.values.assign(config_.lanes, complete ? 1 : 0);
        w.rf_landings.push_back(std::move(predicate));
        sanitizer_.record_pred_ready(warp.id, request.predicate_dst, cycle_ + 1);
        return;
    }
    const auto index = request.wait_slot.value();
    const auto& slot = mbar_wait_slots_.cur().at(index);
    if (!slot.has_value() || slot->warp != warp.id || wait_slot_completed(w, index)) return; // stale
    if (slot->byte_address != request.byte_address ||
        slot->parity != request.parity || slot->predicate_dst != request.predicate_dst) {
        throw SimError("internal mbar try-wait record mismatch");
    }
    trace_->mbar_probe(cycle_, warp.id, request.byte_address, request.parity, complete);
    if (complete) {
        complete_mbar_try_wait(w, index, true);
        return;
    }
    const auto deadline = cycle_ > std::numeric_limits<uint64_t>::max() -
            config_.mbar_try_wait_timeout_cycles
        ? std::numeric_limits<uint64_t>::max()
        : cycle_ + config_.mbar_try_wait_timeout_cycles;
    w.shmem.park = std::make_pair(index, deadline);
    trace_->mbar_sleep(cycle_, warp.id, request.byte_address, request.parity);
}

// An RMW waits to start in one of the engine's three requester registers.
bool Machine::rmw_requester_held() const {
    return simt_rmw_.cur().has_value() || tma_mbar_event_.cur().has_value() || mma_mbar_event_.cur().has_value();
}

// A started RMW waits for its read: the engine's operation (Q) whose read is
// not granted yet, or the RMW starting now (A-write's alternation and the
// engine's admission, earlier in wire order).
bool Machine::rmw_awaits_read(const Wires& w) const {
    const auto& active = rmw_.cur().active;
    return (active.has_value() && !active->read_granted) || w.shmem.rmw_start.has_value();
}

// The RMW engine's admission: when A-write lets an RMW
// start, which requester the idle engine takes. Implementer choice, one
// simple fairness rule: the SIMT RMW register and the engines' event
// registers alternate BY CLASS -- when both classes hold a request, the class
// that did not start last goes (RmwEngine::simt_started_last); within the
// engines the existing TMA / MMA alternation decides (engine_to_claim).
std::optional<Machine::RmwStart> Machine::select_rmw_admission(const Wires& w) const {
    if (!w.shmem.a_write.rmw_start) return std::nullopt;
    const bool tma = tma_mbar_event_.cur().has_value();
    const bool mma = mma_mbar_event_.cur().has_value();
    const bool simt = simt_rmw_.cur().has_value();
    if (simt && (!(tma || mma) || !rmw_.cur().simt_started_last)) return RmwStart{};
    return RmwStart{engine_to_claim(tma, mma)};
}

// An RMW starts: the engine takes the request out of its register (an
// engine's event: its command ends now, step_tma; the sanitizer admits an
// engine's RMW here, a SIMT one at issue). The PAUSE is set from the next
// cycle; the trace's "mbar RMW" (an atomic: "atomic RMW") slice runs from
// now to the write grant.
void Machine::serve_rmw_start(Wires& w) {
    auto& shmem = w.shmem;
    if (!shmem.rmw_start.has_value()) return;
    ActiveMbarRmw active;
    active.engine = shmem.rmw_start->engine;
    if (!active.engine.has_value()) {
        shmem.simt_rmw_started = true;
        active.request = *simt_rmw_.cur();
    } else {
        const bool tma = *active.engine == MbarEngine::tma;
        (tma ? shmem.tma_event_claimed : shmem.mma_event_claimed) = true;
        active.request = *(tma ? tma_mbar_event_ : mma_mbar_event_).cur();
        sanitizer_.check_engine_mbar_rmw(cycle_, active.request.byte_address,
            active.request.operation == MbarRmwOperation::complete_tx);
    }
    if (is_atomic_operation(active.request.operation)) {
        trace_->atomic_rmw_begin(cycle_, rmw_label(active) + ", warp=" + std::to_string(*active.request.warp));
    } else {
        trace_->mbar_rmw_begin(cycle_, active.request.byte_address, rmw_label(active));
    }
    shmem.rmw_started = std::move(active);
}

// The RMW's update: its read's words are at the port output now. The ALU
// computes the new mbar word (and whether the phase flips) from them; the
// result enters ALU stage 0 at this cycle's edge and its write is granted C
// cycles from now (select_a_write).
//
// The tx count is signed: arrive.expect_tx adds,
// complete_tx subtracts, in either order --
//     completion first:  (1, 0) -complete_tx 128-> (1, -128) -arrive.expect_tx 128-> (0, 0) done
//     expectation first: (1, 0) -arrive.expect_tx 128-> (0, 128) -complete_tx 128-> (0, 0) done
// (pending arrivals, tx). arrive.expect_tx changes both, THEN completion is
// tested; the phase completes when both are zero. tx arithmetic is checked: an
// operand above 2^20 - 1 or a count leaving -(2^20 - 1)..2^20 - 1 PANICs (an
// atomic add wraps; this does not). complete_tx carries no phase: it changes
// the current one. Whether that is the phase its sender meant is the
// sanitizer's question (the TMA's accept-time peek; the host's parity).
// An atomic's word is ordinary data, not an mbar: it goes to atomic_update
// before any mbar decode (no live-phase validation, no tx check).
Machine::WriteBeat Machine::rmw_update(const ReadBeat& read) {
    const auto& request = rmw_.cur().active->request;
    if (is_atomic_operation(request.operation)) return atomic_update(request, read.words.at(0));
    auto state = decode_mbar_state(static_cast<uint64_t>(read.words.at(0)) | static_cast<uint64_t>(read.words.at(1)) << 32);

    const auto validate_live = [&] {
        if (!state.valid) mbar_rmw_panic(request, "operation on an uninitialized mbar");
        if (state.expected == 0 || state.remaining > state.expected) {
            mbar_rmw_panic(request, "corrupt mbar state");
        }
    };
    const auto tx_operand = [&]() -> int64_t {
        if (request.value > static_cast<uint32_t>(mbar_encoding::tx_limit)) {
            mbar_rmw_panic(request, "transaction byte count operand exceeds 2^20 - 1");
        }
        return request.value;
    };
    const auto add_tx = [&](int64_t delta) {
        const int64_t count = static_cast<int64_t>(state.tx_bytes) + delta;
        if (count > mbar_encoding::tx_limit) mbar_rmw_panic(request, "transaction byte count overflow (above 2^20 - 1)");
        if (count < -mbar_encoding::tx_limit) mbar_rmw_panic(request, "transaction byte count underflow (below -(2^20 - 1))");
        state.tx_bytes = static_cast<int32_t>(count);
    };

    switch (request.operation) {
    case MbarRmwOperation::init:
        // An init over an mbar with a live waiter is UB (as on NVIDIA: an
        // mbarrier is reinitialized only after mbarrier.inval); the
        // sanitizer PANICs (mbar.init over a live mbar). The machine
        // just writes the new state: a waiter is not woken by it.
        if (request.value == 0 || request.value > mbar_encoding::max_count) {
            mbar_rmw_panic(request, "expected arrival count is outside 1..32767");
        }
        state = {};
        state.valid = true;
        state.expected = static_cast<uint16_t>(request.value);
        state.remaining = static_cast<uint16_t>(request.value);
        break;
    case MbarRmwOperation::arrive:
        validate_live();
        if (state.remaining == 0) mbar_rmw_panic(request, "arrival count underflow");
        --state.remaining;
        break;
    case MbarRmwOperation::arrive_expect_tx:
        validate_live();
        if (state.remaining == 0) mbar_rmw_panic(request, "arrival count underflow");
        --state.remaining;
        add_tx(tx_operand());
        break;
    case MbarRmwOperation::complete_tx:
        validate_live();
        sanitizer_.check_complete_tx_phase(cycle_, request.byte_address, request.tma_command, request.parity,
            state.phase);
        add_tx(-tx_operand());
        break;
    case MbarRmwOperation::atom_add:
    case MbarRmwOperation::atom_and:
    case MbarRmwOperation::atom_xor:
    case MbarRmwOperation::atom_exch:
        throw SimError("internal: an atomic reached the mbar update");
    }

    WriteBeat beat;
    if (request.operation != MbarRmwOperation::init && state.remaining == 0 && state.tx_bytes == 0) {
        state.phase = !state.phase;
        state.remaining = state.expected;
        beat.wake = MbarWake{request.byte_address, static_cast<uint8_t>(state.phase)};
    }
    sanitizer_.record_mbar_tx(request.byte_address, state.tx_bytes);
    const auto word = encode_mbar_state(state);
    std::vector<uint8_t> bytes(8);
    std::memcpy(bytes.data(), &word, sizeof(word));
    beat.writes.push_back({request.byte_address, std::move(bytes)});
    return beat;
}

bool Machine::is_atomic_operation(MbarRmwOperation operation) {
    return operation == MbarRmwOperation::atom_add || operation == MbarRmwOperation::atom_and ||
        operation == MbarRmwOperation::atom_xor || operation == MbarRmwOperation::atom_exch;
}

// A uniform atomic's update, on signed int32: the
// new word from the old one and the operand, written back as one 4-byte
// word. add wraps: the sum is taken in unsigned 32-bit arithmetic, i.e.
// modulo 2^32 -- defined 32-bit two's-complement wrap, never host signed
// overflow. (For add, and, xor and exchange the signed and unsigned bit
// results agree.) No wake: an atomic is not an mbar operation.
Machine::WriteBeat Machine::atomic_update(const MbarRmwRequest& request, uint32_t old_word) const {
    uint32_t word = 0;
    switch (request.operation) {
    case MbarRmwOperation::atom_add: word = old_word + request.value; break;
    case MbarRmwOperation::atom_and: word = old_word & request.value; break;
    case MbarRmwOperation::atom_xor: word = old_word ^ request.value; break;
    case MbarRmwOperation::atom_exch: word = request.value; break;
    default: throw SimError("internal: an mbar operation in the atomic ALU");
    }
    WriteBeat beat;
    std::vector<uint8_t> bytes(4);
    std::memcpy(bytes.data(), &word, sizeof(word));
    beat.writes.push_back({request.byte_address, std::move(bytes)});
    return beat;
}

// (T) An engine's held event waits while the RMW engine works on the OTHER
// engine's event (from its start to its write grant). An event presented this
// cycle is held from the next one.
void Machine::count_engine_wait_on_other(const Wires& w) {
    std::optional<MbarEngine> serving;
    if (w.shmem.rmw_start.has_value()) serving = w.shmem.rmw_start->engine;
    else if (rmw_.cur().active.has_value()) serving = rmw_.cur().active->engine;
    if (!serving.has_value()) return;
    if ((*serving == MbarEngine::tma ? mma_mbar_event_ : tma_mbar_event_).cur().has_value()) {
        trace_->mbar_engine_wait_on_other();
    }
}

// The RMW engine's choice between the two engines' held events: one holder
// is started; when both hold one, the engine not started last.
Machine::MbarEngine Machine::engine_to_claim(bool tma_held, bool mma_held) const {
    if (!mma_held) return MbarEngine::tma;
    if (!tma_held) return MbarEngine::mma;
    return rmw_.cur().last_served == MbarEngine::tma ? MbarEngine::mma : MbarEngine::tma;
}

std::string Machine::rmw_label(const ActiveMbarRmw& active) {
    std::string operation;
    switch (active.request.operation) {
    case MbarRmwOperation::init: operation = "init"; break;
    case MbarRmwOperation::arrive: operation = "arrive"; break;
    case MbarRmwOperation::arrive_expect_tx: operation = "arrive.expect_tx"; break;
    case MbarRmwOperation::complete_tx: operation = "complete_tx"; break;
    case MbarRmwOperation::atom_add: operation = "atom.add.uni"; break;
    case MbarRmwOperation::atom_and: operation = "atom.and.uni"; break;
    case MbarRmwOperation::atom_xor: operation = "atom.xor.uni"; break;
    case MbarRmwOperation::atom_exch: operation = "atom.exch.uni"; break;
    }
    if (active.engine.has_value()) {
        operation += *active.engine == MbarEngine::tma ? " (tma)" : " (mma.commit)";
    }
    return operation;
}

// The warp's probe has not come back: it travels the dispatch pipeline, waits
// in the probe register, or travels the read pipeline up to the port output
// (one at the output returns now: its predicate lands at this cycle's edge
// and the warp issues from the next cycle). The scan stands for a per-warp
// "probe outstanding" bit (set at issue, cleared when the probe returns), the
// hardware form; both give the same answer every cycle.
bool Machine::probe_in_flight(uint32_t warp) const {
    if (warp_dispatching(warp, false, false, true)) return true;
    const auto& waiting = mbar_probe_.cur();
    if (waiting.has_value() && waiting->warp == warp) return true;
    const auto& pipe = a_read_pipe_.cur();
    for (std::size_t stage = 0; stage < pipe.size(); ++stage) {
        const auto& beat = pipe[stage];
        if (beat.has_value() && beat->client == ReadClient::probe && beat->probe->warp == warp) return true;
    }
    return false;
}

// ---- mbar wait slots (the scheduler-visible try-wait state) ----

bool Machine::wait_slot_completed(const Wires& w, uint32_t slot) const {
    return std::find(w.completed_waits.begin(), w.completed_waits.end(), slot) != w.completed_waits.end();
}

// The slot a warp is parked in (Q): a slot completed now frees at this
// cycle's edge, and its warp issues from the next cycle.
std::optional<uint32_t> Machine::waiting_slot_of(const Wires&, uint32_t warp) const {
    const auto& slots = mbar_wait_slots_.cur();
    for (uint32_t index = 0; index < slots.size(); ++index) {
        if (slots[index].has_value() && slots[index]->warp == warp) return index;
    }
    return std::nullopt;
}

// A wait slot for a try-wait's dispatch now: free in Q, or completed now (d).
std::optional<uint32_t> Machine::free_mbar_wait_slot(const Wires& w) const {
    const auto& slots = mbar_wait_slots_.cur();
    for (uint32_t index = 0; index < slots.size(); ++index) {
        if (!slots[index].has_value() || wait_slot_completed(w, index)) return index;
    }
    return std::nullopt;
}

uint32_t Machine::free_wait_slot_count(const Wires& w) const {
    const auto& slots = mbar_wait_slots_.cur();
    uint32_t count = 0;
    for (uint32_t index = 0; index < slots.size(); ++index) {
        if (!slots[index].has_value() || wait_slot_completed(w, index)) ++count;
    }
    return count;
}

void Machine::wake_mbar_waiters(Wires& w, uint32_t byte_address, uint8_t new_phase) {
    const auto& slots = mbar_wait_slots_.cur();
    for (uint32_t index = 0; index < slots.size(); ++index) {
        const auto& slot = slots[index];
        if (!slot.has_value() || slot->byte_address != byte_address ||
            slot->parity == new_phase) {
            continue;
        }
        trace_->mbar_wake(cycle_, slot->warp, byte_address, new_phase);
        complete_mbar_try_wait(w, index, true);
    }
}

// A try-wait ends: its predicate lands in the file at this cycle's edge and
// its slot frees (drive_wait_slots clears it): the warp issues from the next
// cycle.
void Machine::complete_mbar_try_wait(Wires& w, uint32_t index, bool result) {
    const auto& slots = mbar_wait_slots_.cur();
    if (index >= slots.size() || !slots[index].has_value() || wait_slot_completed(w, index)) {
        throw SimError("internal completion of an empty mbar try-wait slot");
    }
    const auto& slot = *slots[index];
    Writeback predicate;
    predicate.port_cycle = cycle_;
    predicate.land_cycle = cycle_;
    predicate.warp = slot.warp;
    predicate.predicate_dst = slot.predicate_dst;
    predicate.values.assign(config_.lanes, result ? 1 : 0);
    w.rf_landings.push_back(std::move(predicate));
    sanitizer_.record_pred_ready(slot.warp, slot.predicate_dst, cycle_ + 1);
    w.completed_waits.push_back(index);
}

void Machine::decide_mbar_timeouts(Wires& w) {
    const auto& slots = mbar_wait_slots_.cur();
    for (uint32_t index = 0; index < slots.size(); ++index) {
        const auto& slot = slots[index];
        if (!slot.has_value() || !slot->parked || cycle_ < slot->deadline_cycle) continue;
        if (wait_slot_completed(w, index)) continue; // woken this cycle: the phase flip wins
        trace_->mbar_timeout(cycle_, slot->warp, slot->byte_address, slot->parity);
        complete_mbar_try_wait(w, index, false);
    }
}

// mbar.init / arrive / arrive.expect_tx: a uniform operation, its operands
// uniform registers whatever the active mask (a partial mask is the
// sanitizer's PANIC, in uniform_operand).
void Machine::dispatch_mbar_rmw(const DispatchEntry& op, Wires& w) {
    const auto& inst = op.instruction;
    MbarRmwRequest request;
    request.byte_address = mbar_address_operand(op);
    request.warp = op.warp;
    request.pc = op.pc;
    request.instruction = inst;
    if (inst.opcode == Opcode::mbar_init_i || inst.opcode == Opcode::mbar_init_r) {
        request.operation = MbarRmwOperation::init;
        request.value = inst.opcode == Opcode::mbar_init_i
            ? static_cast<uint32_t>(inst.imm)
            : uniform_operand(op, inst.src1);
        if (request.value == 0 || request.value > mbar_encoding::max_count) {
            panic(op, "mbar expected count is outside 1..32767");
        }
    } else if (inst.opcode == Opcode::mbar_arrive) {
        request.operation = MbarRmwOperation::arrive;
    } else {
        request.operation = MbarRmwOperation::arrive_expect_tx;
        if (inst.opcode == Opcode::mbar_arrive_expect_tx_i && inst.imm < 0) {
            panic(op, "mbar tx byte count is negative");
        }
        request.value = inst.opcode == Opcode::mbar_arrive_expect_tx_i
            ? static_cast<uint32_t>(inst.imm)
            : uniform_operand(op, inst.src1);
    }
    sanitizer_.check_simt_mbar_rmw(site(op), request.byte_address);
    if (request.operation == MbarRmwOperation::init) sanitizer_.record_mbar_init(site(op), request.byte_address);
    sanitizer_.record_simt_shmem_issue(site(op), SimtShmemHolder::rmw, op.warp,
        std::vector<std::vector<uint32_t>>{{request.byte_address}}, 8, true);
    // The SIMT RMW register is empty after this cycle (admission secured
    // it). The operation waits there until the RMW engine starts it
    // (A-write's alternation, then the engine's admission); its warp's
    // younger shmem accesses wait until its final write grant
    // (warp_unserviced_write).
    w.dispatch.rmw = std::move(request);
}

// atom.op.uni rD, [urA + imm], urB. Admission found the warp
// with no unserviced read or write and secured a load slot and the SIMT RMW
// register. Now the address (4-byte aligned, in bounds: PANIC otherwise) and
// the operand, uniform registers, arrive -- whatever the active mask; a
// partial mask is the sanitizer's PANIC -- the RETURN load slot is installed
// (rD's lanes, the depbar, the old word), and the atomic enters the SIMT RMW
// register like an mbar
// operation: the RMW engine reads the word, the ALU computes the new one,
// and its write is granted D + Lb + C after the read. The warp's younger
// shmem accesses wait for that write grant (warp_unserviced_write); rD waits
// for the depbar, cleared as the return slot's RF write lands -- possibly
// before the write grant.
void Machine::dispatch_atomic(const DispatchEntry& op, Wires& w) {
    const auto& inst = op.instruction;
    const auto address = shmem_address(op, uniform_operand(op, inst.src0));
    const auto operand = uniform_operand(op, inst.src1);
    const auto slot_index = *free_load_slot(w);
    sanitizer_.check_simt_shmem_access(site(op), std::vector<uint32_t>{address}, false);
    // The shadow of the warp's access: one 4-byte write round, reaching the
    // RAM at its write grant + D.
    sanitizer_.record_simt_shmem_issue(site(op), SimtShmemHolder::rmw, op.warp,
        std::vector<std::vector<uint32_t>>{{address}}, 4, true);
    LoadSlot slot;
    slot.warp = op.warp;
    slot.uniform = true;
    slot.mask = op.active_mask;
    slot.dst = inst.dst;
    slot.depbar = inst.control.set_depbar;
    slot.words.assign(config_.lanes, 0);
    slot.engine_filled = true;
    w.dispatch.load_slot = std::make_pair(slot_index, std::move(slot));
    MbarRmwRequest request;
    switch (inst.opcode) {
    case Opcode::atom_add_uni: request.operation = MbarRmwOperation::atom_add; break;
    case Opcode::atom_and_uni: request.operation = MbarRmwOperation::atom_and; break;
    case Opcode::atom_xor_uni: request.operation = MbarRmwOperation::atom_xor; break;
    default: request.operation = MbarRmwOperation::atom_exch; break;
    }
    request.byte_address = address;
    request.value = operand;
    request.warp = op.warp;
    request.return_slot = slot_index;
    request.pc = op.pc;
    request.instruction = inst;
    w.dispatch.rmw = std::move(request);
}

// mbar.test_wait / try_wait: the probe goes into the probe register; A-read
// serves it from the next cycle and its warp waits until it comes back --
// from its issue on, since it is in the dispatch pipeline first
// (probe_in_flight). try_wait installs its wait slot first,
// so a wake that comes while the probe travels finds it.
void Machine::dispatch_mbar_wait(const DispatchEntry& op, Wires& w) {
    const auto& inst = op.instruction;
    const bool test = is_mbar_test_wait(inst.opcode);
    const bool immediate = inst.opcode == Opcode::mbar_test_wait_i ||
        inst.opcode == Opcode::mbar_try_wait_i;
    const auto parity_value = immediate
        ? static_cast<uint32_t>(inst.imm)
        : uniform_operand(op, inst.src1);
    if (parity_value > 1) panic(op, "mbar parity is not 0 or 1");
    MbarReadRequest request;
    request.test = test;
    request.warp = op.warp;
    request.predicate_dst = inst.dst;
    request.byte_address = mbar_address_operand(op);
    request.parity = static_cast<uint8_t>(parity_value);
    request.pc = op.pc;
    request.instruction = inst;
    if (!test) {
        const auto slot_index = free_mbar_wait_slot(w);
        if (!slot_index.has_value()) {
            throw SimError("internal mbar try-wait slot disappeared before dispatch");
        }
        MbarWaitSlot slot;
        slot.warp = op.warp;
        slot.byte_address = request.byte_address;
        slot.parity = request.parity;
        slot.predicate_dst = request.predicate_dst;
        w.dispatch.wait_slot = std::make_pair(*slot_index, slot);
        request.wait_slot = *slot_index;
    }
    sanitizer_.check_simt_mbar_probe(site(op), request.byte_address);
    sanitizer_.record_simt_shmem_issue(site(op), SimtShmemHolder::probe, op.warp,
        std::vector<std::vector<uint32_t>>{{request.byte_address}}, 8, false);
    w.dispatch.mbar_probe = std::move(request);
}

// mbar wait slots: completed slots free; a try-wait that found its phase
// incomplete parks; issue may take a slot freed this cycle (d).
void Machine::drive_wait_slots(const Wires& w) {
    if (w.completed_waits.empty() && !w.shmem.park.has_value() && !w.dispatch.wait_slot.has_value()) return;
    // (One calculation per slot touched now: free, then park, then take.)
    std::vector<uint32_t> touched(w.completed_waits.begin(), w.completed_waits.end());
    if (w.shmem.park.has_value()) touched.push_back(w.shmem.park->first);
    if (w.dispatch.wait_slot.has_value()) touched.push_back(w.dispatch.wait_slot->first);
    std::sort(touched.begin(), touched.end());
    touched.erase(std::unique(touched.begin(), touched.end()), touched.end());
    for (const auto index : touched) {
        auto next = mbar_wait_slots_.cur().at(index);
        if (std::find(w.completed_waits.begin(), w.completed_waits.end(), index) != w.completed_waits.end()) {
            next.reset();
        }
        if (const auto& park = w.shmem.park; park.has_value() && park->first == index) {
            next->parked = true;
            next->deadline_cycle = park->second;
        }
        if (const auto& taken = w.dispatch.wait_slot; taken.has_value() && taken->first == index) {
            if (next.has_value()) throw SimError("internal mbar wait slot allocated twice");
            next = taken->second;
        }
        mbar_wait_slots_.set_next(index, std::move(next), "drive_wait_slots");
    }
}

// The probe register: freed by its A-read grant, refilled by dispatch (d).
void Machine::drive_mbar_probe(const Wires& w) {
    const bool granted = w.shmem.a_read.user == AReadUser::probe;
    if (w.dispatch.mbar_probe.has_value()) {
        if (mbar_probe_.cur().has_value() && !granted) throw SimError("internal mbar probe accepted over another");
        mbar_probe_.set_next(w.dispatch.mbar_probe, "dispatch: new mbar probe");
    } else if (granted) {
        mbar_probe_.set_next(std::nullopt, "A-read: mbar probe granted");
    }
}

// The SIMT RMW register: emptied by its RMW's start, refilled by dispatch in
// the same cycle (d).
void Machine::drive_simt_rmw(const Wires& w) {
    const bool started = w.shmem.simt_rmw_started;
    if (w.dispatch.rmw.has_value()) {
        if (simt_rmw_.cur().has_value() && !started) throw SimError("internal SIMT RMW accepted over another");
        simt_rmw_.set_next(w.dispatch.rmw, "dispatch: SIMT RMW");
    } else if (started) {
        simt_rmw_.set_next(std::nullopt, "RMW engine: SIMT RMW started");
    }
}

// The RMW engine, in one calculation: the update computed from the words at
// the port output now enters its ALU's stage 0 (the last stage's leaves with
// the write granted now); the write granted now ends the
// operation and the PAUSE; an RMW starting now installs its operation (and
// the admission state: which class, which engine started last); the read
// granted now marks it.
void Machine::drive_rmw_engine(const Wires& w) {
    const auto& cur = rmw_.cur();
    const bool started = w.shmem.rmw_start.has_value();
    const bool read_granted = w.shmem.a_read.user == AReadUser::rmw;
    const bool write_granted = w.shmem.a_write.user == AWriteUser::rmw;
    if (w.shmem.rmw_update.has_value()) rmw_alu_.push(*w.shmem.rmw_update, "drive_rmw_engine: ALU");
    if (!started && !read_granted && !write_granted) return;
    auto next = cur;
    if (write_granted) next.active.reset();
    if (started) {
        if (next.active.has_value()) throw SimError("internal RMW started over a busy RMW engine");
        next.active = w.shmem.rmw_started;
        const auto engine = w.shmem.rmw_start->engine;
        next.simt_started_last = !engine.has_value();
        if (engine.has_value()) next.last_served = *engine;
    }
    if (read_granted) next.active->read_granted = true;
    rmw_.set_next(std::move(next), "drive_rmw_engine");
}

uint32_t Machine::mbar_address_operand(const DispatchEntry& op) const {
    const auto address = uniform_operand(op, op.instruction.src0);
    if (address % 8 != 0 || static_cast<uint64_t>(address) + 8 > shmem_.size()) {
        panic(op, "mbar address is not 8-byte aligned and in shmem");
    }
    return address;
}

MbarState Machine::load_mbar(uint32_t byte_address) const {
    if (byte_address % 8 != 0 || static_cast<uint64_t>(byte_address) + 8 > shmem_.size()) {
        throw SimError("invalid 64-bit mbar shmem access");
    }
    const auto low = static_cast<uint64_t>(read_shmem_u32(byte_address));
    const auto high = static_cast<uint64_t>(read_shmem_u32(byte_address + 4));
    return decode_mbar_state(low | (high << 32));
}

[[noreturn]] void Machine::mbar_rmw_panic(
    const MbarRmwRequest& request, const std::string& reason) const {
    if (request.warp.has_value()) {
        const IssueSite requester{cycle_, *request.warp, request.pc, request.instruction};
        throw_mbar_rmw_panic(cycle_, request.byte_address, reason, &requester);
    }
    throw_mbar_rmw_panic(cycle_, request.byte_address, reason, nullptr);
}

} // namespace nanosm

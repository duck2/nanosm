// The vector register file: the write calendar, the read and write port decisions, the
// RAM stages, and writebacks.
// (Machine is one class, src/machine/machine.hpp; the map of its files is at the top of
// src/machine/machine.cpp.)

#include "machine/machine.hpp"

#include <algorithm>
#include <bit>
#include <utility>

namespace nanosm {

// ===========================================================================
// The RF ports: the reservation calendars, the RF read and write decisions,
// the write port and the write pipeline (every cycle number follows
// src/timing.hpp)
// ===========================================================================

// Fixed issue never contests the current cycle's RF ports (the issue
// register): its read round samples at issue + 1, its fixed write is booked
// in the write calendar at issue + issue_to_port. What the current cycle
// leaves goes to the asynchronous clients, decided from Q before issue:
//   READS   decide_rf_reads: the issue
//           register's instruction owns the read ports its round samples
//           now; acc.st beats and the LSU's store-data read share the rest
//           by round-robin. A denied reader whose bank's read_owed bit is
//           set inhibits the acceptance of new SIMT reads of that bank, so
//           the bank's ports are free one cycle later (the mirrored rule).
//   WRITES  decide_rf_writes: load slots, the LSU's load buffer and the
//           acc.ld pipelines share the current write port's unbooked ports;
//           a denied writer whose bank's async_owed bit is set inhibits new
//           fixed write bookings to that bank.
// A read SAMPLES the RF at its cycle and has its operand Lr cycles later; a
// write PRESENTED at the port lands at the clock edge Wd cycles later and is
// visible to samples from the next cycle (no same-edge forwarding).

// The calendar's stage that holds the bookings for `at_cycle`: the last stage
// is this cycle, each stage before it one cycle later.
std::size_t Machine::write_calendar_stage(uint64_t at_cycle) const {
    const auto ahead = at_cycle - cycle_;
    if (at_cycle < cycle_ || ahead >= write_calendar_.stages()) {
        throw SimError("internal RF write booking beyond the write calendar");
    }
    return write_calendar_.stages() - 1 - static_cast<std::size_t>(ahead);
}

bool Machine::write_slot_free(uint8_t reg, uint64_t at_cycle) const {
    const auto& booked = write_calendar_.stage(write_calendar_stage(at_cycle));
    return !booked.has_value() || booked->at(reg % config_.rf_banks) < config_.rf_write_ports_per_bank;
}

// The booking enters the calendar at this edge (drive_write_calendar), at the
// stage of its cycle.
void Machine::reserve_write(uint8_t reg, uint64_t at_cycle, Wires& w) {
    if (at_cycle <= cycle_) throw SimError("internal RF write booking for the current cycle");
    if (!write_slot_free(reg, at_cycle)) throw SimError("internal RF write booking over a full port");
    w.issue.rf_write_stage = write_calendar_stage(at_cycle);
}

// The calendar's last stage: a bank's writes booked (in earlier cycles) for
// this cycle.
uint32_t Machine::writes_booked_now(uint32_t bank) const {
    const auto& now = write_calendar_.last();
    return now.has_value() ? now->at(bank) : 0;
}

bool Machine::RfWriteDecision::granted_to(AsyncWriter writer, uint32_t index) const {
    for (std::size_t holder = 0; holder < holders.size(); ++holder) {
        if (granted[holder] != 0 && holders[holder]->writer == writer && holders[holder]->index == index) return true;
    }
    return false;
}

bool Machine::RfReadDecision::granted_to(RfReader reader, uint32_t index) const {
    for (std::size_t holder = 0; holder < holders.size(); ++holder) {
        if (granted[holder] != 0 && holders[holder]->reader == reader && holders[holder]->index == index) return true;
    }
    return false;
}

// The load buffer as an async RF holder: its next write when it holds a
// complete group (in Q), else none (the store buffer never writes the RF). A
// v8 group is eight writes, rD + j one after another.
std::optional<Machine::AsyncWrite> Machine::lsu_buffer_rf_write(uint32_t index) const {
    if (index != load_buffer) return std::nullopt;
    const auto& buffer = lsu_buffers_.at(index).cur();
    if (!buffer.owner.has_value() || buffer.valid_count != buffer.expected) return std::nullopt;
    const auto* command = &lsu_commands_.cur().in_slot(*buffer.owner);
    if (is_store(command->kind)) throw SimError("internal: the load buffer's owner is not a load");
    const bool vector = command->kind == LsuKind::vector_load;
    const auto reg = static_cast<uint8_t>(command->reg + (vector ? buffer.beats : 0));
    return AsyncWrite{AsyncWriter::lsu_buffer, index, command->warp, reg};
}

// The LSU's one RF write path: the load buffer, when it holds a complete
// group (one write a cycle).
std::optional<uint32_t> Machine::lsu_rf_presented() const {
    if (lsu_buffer_rf_write(load_buffer).has_value()) return load_buffer;
    return std::nullopt;
}

// Every async RF holder, in round-robin order, with its write when its vector
// is complete in Q: the load slots, the load buffer on the LSU's write
// path, then each tmem bank's acc.ld pipeline output.
std::vector<std::optional<Machine::AsyncWrite>> Machine::ready_async_writers(
    std::optional<uint32_t> lsu_presented) const {
    std::vector<std::optional<AsyncWrite>> holders;
    for (uint32_t index = 0; index < load_slots_.count(); ++index) {
        const auto& slot = load_slots_[index].cur();
        if (slot.has_value() && slot->complete()) {
            holders.push_back(AsyncWrite{AsyncWriter::load_slot, index, slot->warp, slot->dst});
        } else {
            holders.emplace_back();
        }
    }
    for (uint32_t index = 0; index < lsu_buffers_.count(); ++index) {
        holders.push_back(lsu_presented == index ? lsu_buffer_rf_write(index) : std::nullopt);
    }
    if (const auto& output = acc_ld_pipe_.cur().back(); output.has_value()) {
        holders.push_back(AsyncWrite{AsyncWriter::acc_ld, 0, output->warp, output->reg});
    } else {
        holders.emplace_back();
    }
    return holders;
}

// The RF write decision (make an opening, then choose a writer): one per RF
// bank, from Q only, before issue.
//   async_ready  = a ready async holder targets the bank
//   async_grant  = async_ready AND one of this cycle's write ports is unbooked
//   inhibit_simt = async_ready AND NOT async_grant AND async_owed
// The holders granted (round-robin from the bank's cursor) enter the write
// pipeline now; the owed bit and the cursor take their next values in
// drive_rf_write_policy. Old bookings always execute; issue's outcome cannot
// change these grants.
Machine::RfWriteDecision Machine::decide_rf_writes() const {
    RfWriteDecision decision;
    decision.lsu_presented = lsu_rf_presented();
    decision.holders = ready_async_writers(decision.lsu_presented);
    decision.granted.assign(decision.holders.size(), 0);
    decision.banks.resize(config_.rf_banks);
    const auto& policy = rf_write_policy_.cur();
    for (uint32_t bank = 0; bank < config_.rf_banks; ++bank) {
        std::vector<uint8_t> ready(decision.holders.size(), 0);
        for (std::size_t holder = 0; holder < ready.size(); ++holder) {
            const auto& write = decision.holders[holder];
            ready[holder] = write.has_value() && write->reg % config_.rf_banks == bank ? 1 : 0;
        }
        const auto free_ports = config_.rf_write_ports_per_bank - writes_booked_now(bank);
        const auto choice = round_robin_grant(ready, policy.at(bank).cursor, free_ports);
        for (const auto holder : choice.granted) decision.granted[holder] = 1;
        auto& state = decision.banks[bank];
        state.ready = std::find(ready.begin(), ready.end(), 1) != ready.end();
        state.granted = !choice.granted.empty();
        state.inhibit_simt = state.ready && !state.granted && policy.at(bank).owed != 0;
        state.cursor_after = choice.cursor_after;
    }
    return decision;
}

// The payload and completion context a granted async holder hands to the RF
// write path: presented at the port now, landing Wd cycles later. Its
// depbar (if it ends its operation) clears as it lands; the sanitizer marks
// its registers visible then.
Machine::Writeback Machine::async_writeback(const AsyncWrite& write) {
    Writeback writeback;
    writeback.port_cycle = cycle_;
    writeback.land_cycle = cycle_ + config_.rf_write_depth_cycles;
    writeback.warp = write.warp;
    writeback.dst = write.reg;
    trace_->rf_async_write();
    switch (write.writer) {
    case AsyncWriter::load_slot: {
        const auto& slot = *load_slots_.at(write.index).cur();
        writeback.values = slot.words;
        writeback.lane_enables = slot.mask;
        writeback.clear_depbar = slot.depbar;
        writeback.visible_regs = {write.reg};
        return writeback;
    }
    case AsyncWriter::lsu_buffer: {
        // One write of the load buffer's group: a v8 group's register rD + j
        // with the group's active lanes enabled (lane l's word j); a gather
        // group's active lanes; a uniform load's one word to every lane.
        const auto& buffer = lsu_buffers_.at(write.index).cur();
        const auto& command = lsu_commands_.cur().in_slot(*buffer.owner);
        const auto width = lsu_group_width(command);
        const bool vector = command.kind == LsuKind::vector_load;
        const bool uniform = command.kind == LsuKind::uniform_load;
        writeback.values.assign(config_.lanes, 0);
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            if (uniform) {
                writeback.values[lane] = buffer.words[0];
                continue;
            }
            if ((buffer.lanes >> lane & 1) == 0) continue;
            const uint32_t at = vector ? (lane - buffer.group * width) * 8 + buffer.beats
                                       : lane - buffer.group * width;
            sanitizer_.check_buffer_lanes_valid(cycle_, write.index, buffer.valid, at, 1, "RF drain");
            writeback.values[lane] = buffer.words[at];
        }
        if (!uniform) writeback.lane_enables = buffer.lanes;
        const bool group_done = buffer.beats + 1 == buffer.beats_total;
        const uint32_t lanes_done = uniform ? 1 : static_cast<uint32_t>(std::popcount(buffer.lanes));
        if (group_done && command.lanes_written + lanes_done == command.requests_total) {
            // The command's last write: rD..rD+7 count as in flight until it lands.
            const uint32_t registers = vector ? 8 : 1;
            for (uint32_t j = 0; j < registers; ++j) writeback.visible_regs.push_back(static_cast<uint8_t>(command.reg + j));
            writeback.clear_depbar = command.depbar;
            writeback.release_lsu_command = command.id;
        }
        // The bytes written: the group's enabled lanes (every lane for ldg.uni).
        probe(LsuEdge::rf_write_beat, command.id, write.reg,
            static_cast<uint32_t>(std::popcount(buffer.lanes)) * 4);
        return writeback;
    }
    case AsyncWriter::acc_ld: {
        const auto& beat = *acc_ld_pipe_.cur().back();
        writeback.values = beat.values;
        writeback.visible_regs = {write.reg};
        // (Its tmem read-ownership ended at the transfer's last tmem sample,
        // sample_acc_ld_beat; the destination stays not-ready until this
        // write lands.)
        if (beat.last) writeback.clear_depbar = beat.depbar;
        return writeback;
    }
    }
    throw SimError("internal unknown async RF writer");
}

// (T) Ready holders not written this cycle.
void Machine::count_rf_write_waits(const RfWriteDecision& decision) {
    for (std::size_t holder = 0; holder < decision.holders.size(); ++holder) {
        const auto& write = decision.holders[holder];
        if (!write.has_value() || write->writer == AsyncWriter::lsu_buffer || decision.granted[holder] != 0) continue;
        trace_->rf_async_wait();
    }
    for (uint32_t index = 0; index < lsu_buffers_.count(); ++index) {
        if (!lsu_buffer_rf_write(index).has_value() || decision.granted_to(AsyncWriter::lsu_buffer, index)) continue;
        trace_->rf_async_wait();
        ++trace_->lsu_counters().lsu_rf_write_wait_cycles;
        ++lsu_phases_[lsu_commands_.cur().in_slot(*lsu_buffers_.at(index).cur().owner).id].rf_wait;
    }
}

// The RF write port at this cycle: every fixed result booked for it, then
// the async holders the RF write decision granted (never on a booked port),
// enter the write pipeline. A write that ends an LSU command releases it now
// (the write pipeline owns its data from here); a store's last-ack
// completion due now does too. A load slot granted now is free for dispatch
// now (d).
void Machine::rf_write_port(Wires& w) {
    std::vector<uint32_t> port_uses(config_.rf_banks, 0);
    const auto use_port = [&](const Writeback& writeback) {
        if (!writeback.dst.has_value()) return;
        const auto bank = *writeback.dst % config_.rf_banks;
        if (++port_uses[bank] > config_.rf_write_ports_per_bank) {
            throw SimError("internal: more RF writes than write ports in bank " +
                std::to_string(bank) + " at cycle " + std::to_string(cycle_));
        }
        trace_->rf_write(cycle_, writeback.warp, *writeback.dst, bank);
    };
    for (const auto& writeback : writebacks_.cur()) {
        if (writeback.port_cycle != cycle_) continue;
        use_port(writeback);
        if (writeback.release_lsu_command.has_value()) release_lsu_command(w, *writeback.release_lsu_command);
    }
    const auto& decision = w.rf_write;
    uint32_t lsu_writes = 0;
    for (std::size_t holder = 0; holder < decision.holders.size(); ++holder) {
        if (decision.granted[holder] == 0) continue;
        if (decision.holders[holder]->writer == AsyncWriter::lsu_buffer && ++lsu_writes > 1) {
            throw SimError("internal: more than one LSU RF write at cycle " + std::to_string(cycle_));
        }
        auto writeback = async_writeback(*decision.holders[holder]);
        use_port(writeback);
        if (writeback.release_lsu_command.has_value()) release_lsu_command(w, *writeback.release_lsu_command);
        w.new_writebacks.push_back(std::move(writeback));
    }
    count_rf_write_waits(decision);
}

// The write pipeline's RAM stage: the writes at the RAM now. They land at
// this cycle's clock edge (land_rf_writes, after every read of the cycle).
// Their completions are presented now: a cleared depbar counter is 0 in Q
// from the next cycle -- the write's visibility -- so the waiting warp issues
// then at the earliest; an async result is visible to the sanitizer from the
// next cycle.
void Machine::rf_ram_stage(Wires& w) {
    for (const auto& writeback : writebacks_.cur()) {
        if (writeback.land_cycle != cycle_) continue;
        if (writeback.clear_depbar.has_value()) {
            record_depbar_completion(w, writeback.warp, *writeback.clear_depbar,
                "depbar underflow on sb" + std::to_string(*writeback.clear_depbar));
        }
        for (const auto reg : writeback.visible_regs) sanitizer_.record_reg_ready(writeback.warp, reg, cycle_ + 1);
        if (writeback.dst.has_value() || writeback.predicate_dst.has_value()) w.rf_landings.push_back(writeback);
    }
}

// The RF writes at their RAM now: each goes through a write port of its bank,
// its lanes enabled one by one, and lands at the clock edge.
void Machine::land_rf_writes(const Wires& w) {
    for (const auto& writeback : w.rf_landings) {
        if (!writeback.dst.has_value()) continue;
        if (writeback.values.size() != config_.lanes) throw SimError("internal writeback lane count mismatch");
        const auto address = rf_address(writeback.warp, *writeback.dst);
        rf_.write(address.bank, address.row, writeback.values, writeback.lane_enables.value_or(full_mask()),
            "RF write port");
    }
}

// The predicate files (flip-flops): a predicate presented now (a setp's at
// its port cycle, a usetp's at the URF write port's) is in the file from the
// next cycle, in its enabled lanes. One write per predicate is ever in flight
// (src/timing.hpp), so each register takes at most one.
void Machine::drive_predicates(const Wires& w) {
    for (const auto& writeback : w.rf_landings) {
        if (!writeback.predicate_dst.has_value()) continue;
        if (writeback.values.size() != config_.lanes || *writeback.predicate_dst >= predicates_per_warp) {
            throw SimError("internal: a predicate write of the wrong shape");
        }
        const auto index = predicate_index(writeback.warp, *writeback.predicate_dst);
        const uint64_t enables = writeback.lane_enables.value_or(full_mask());
        uint64_t bits = 0;
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            if (writeback.values[lane] != 0) bits |= uint64_t{1} << lane;
        }
        const uint64_t next = (predicates_.cur()[index] & ~enables) | (bits & enables);
        predicates_.set_next(index, next, "drive_predicates");
    }
}

// The async RF readers' decision (the write decision's rule, mirrored), per
// RF bank, from Q before issue:
//   read_ready   = an async reader wants the bank now
//   read_grant   = read_ready AND the issue register's round leaves one of
//                  the bank's read ports free (simt_read_now: its bank use)
//   inhibit_simt = read_ready AND NOT read_grant AND read_owed
// A candidate instruction whose round uses an inhibited bank is not accepted
// this cycle (resource_view), so the issue register leaves that bank free
// next cycle: the hole the denied reader is owed. Reads already accepted are
// never stopped. The readers granted (round-robin from the bank's cursor)
// sample the RF now: an acc.st beat is at the RF output Lr later and at the
// tmem RAM tmem.port_depth after that; the LSU's beat fills the store
// buffer Lr later. A denied
// reader holds (an acc.st keeps its beat, the LSU its candidate; a source
// stays read-owned). The owed bit and cursor change in drive_rf_read_policy.
Machine::RfReadDecision Machine::decide_rf_reads(Wires& w) {
    RfReadDecision decision;
    const auto& store_engine = acc_engines_.cur()[1];
    if (store_engine.has_value() && store_engine->next_column < store_engine->width) {
        decision.holders.push_back(RfReadRequest{RfReader::acc_st, 0, store_engine->warp,
            static_cast<uint8_t>(store_engine->rf_base + store_engine->next_column), 0, 0});
    } else {
        decision.holders.emplace_back();
    }
    decision.holders.push_back(lsu_store_read_request(w));
    decision.granted.assign(decision.holders.size(), 0);
    decision.banks.resize(config_.rf_banks);
    const auto& policy = rf_read_policy_.cur();
    const auto& simt_read_now = issue_read_banks_.cur();
    for (uint32_t bank = 0; bank < config_.rf_banks; ++bank) {
        std::vector<uint8_t> ready(decision.holders.size(), 0);
        for (std::size_t holder = 0; holder < ready.size(); ++holder) {
            const auto& read = decision.holders[holder];
            ready[holder] = read.has_value() && read->reg % config_.rf_banks == bank ? 1 : 0;
        }
        const auto free_ports = config_.rf_read_ports_per_bank - simt_read_now.at(bank);
        const auto choice = round_robin_grant(ready, policy.at(bank).cursor, free_ports);
        for (const auto holder : choice.granted) decision.granted[holder] = 1;
        auto& state = decision.banks[bank];
        state.ready = std::find(ready.begin(), ready.end(), 1) != ready.end();
        state.granted = !choice.granted.empty();
        state.inhibit_simt = state.ready && !state.granted && policy.at(bank).owed != 0;
        state.cursor_after = choice.cursor_after;
    }
    for (std::size_t holder = 0; holder < decision.holders.size(); ++holder) {
        const auto& read = decision.holders[holder];
        if (!read.has_value()) continue;
        if (decision.granted[holder] == 0) {
            if (read->reader == RfReader::lsu_store) {
                ++trace_->lsu_counters().lsu_rf_read_wait_cycles;
                ++lsu_phases_[read->lsu_command].rf_wait;
            }
            continue;
        }
        const auto& warp = warps_.at(read->warp);
        trace_->rf_read(cycle_, warp.id, read->reg, read->reg % config_.rf_banks);
        auto values = read_rf(warp.id, read->reg,
            read->reader == RfReader::acc_st ? "RF read: acc.st beat" : "RF read: store data");
        if (read->reader == RfReader::acc_st) {
            sanitizer_.record_async_reader_done(warp.id, read->reg, AsyncReader::acc_st);
            w.acc_st_read = AccStBeat{store_engine->next_column, std::move(values)};
        } else {
            // A stg.v8 reads rS + j once per lane group; the register is free
            // to be overwritten after the last group's read.
            if (read->last_group) sanitizer_.record_async_reader_done(warp.id, read->reg, AsyncReader::stg_v8);
            probe(LsuEdge::rf_read_beat, read->lsu_command, read->reg, config_.lanes * 4);
            w.lsu_rf_read.command = read->lsu_slot;
            w.lsu_rf_read.buffer = read->buffer;
            w.lsu_rf_read.group = read->group;
            w.lsu_rf_read.beat = read->beat;
            w.lsu_rf_read.lanes = read->lanes;
            w.lsu_rf_read.values = std::move(values);
        }
    }
    return decision;
}

// The RF read stage (src/timing.hpp): the round of the instruction in the issue
// register (accepted last cycle) samples the RF now -- before this cycle's
// writes land at its edge (READ_FIRST). An arithmetic instruction whose
// operands are all at the RF output now (issue + issue_to_operands)
// executes, and its result is on its way to the write port it booked. A
// dispatch entry's samples are latched into the entry (the RF output
// pipeline carries them to its dispatch). Each bank's samples -- the round's
// and the async readers' (decide_rf_reads) -- never exceed its read ports,
// and the round's are exactly the issue register's bank use. The round's
// uniform registers are sampled from the URF now too (READ_FIRST: a write
// landing at this cycle's edge is not seen); one instruction's round reads
// at most two, the URF's read ports. The URF write pipeline's stage-0 write
// samples here as well (its operands latch in step_urf), and so does the TMA
// operand read's round (issue refused every other uniform reader that would
// sample with one of its later rounds).
void Machine::step_rf_reads(Wires& w) {
    std::vector<uint32_t> simt_reads(config_.rf_banks, 0);
    uint32_t uniform_reads = 0;
    const auto sample = [&](uint32_t warp, uint8_t reg) {
        ++simt_reads[reg % config_.rf_banks];
        trace_->rf_read(cycle_, warp, reg, reg % config_.rf_banks);
        return read_rf(warp, reg, "RF read: the issue register's round");
    };
    const auto sample_uniform = [&](uint32_t warp, const std::vector<uint8_t>& regs) {
        uniform_reads += static_cast<uint32_t>(regs.size());
        std::vector<uint32_t> values;
        for (const auto reg : regs) values.push_back(read_urf(warp, reg, "URF read: the issue register's round"));
        return values;
    };
    // The arithmetic pipeline: the instruction in its last stage has its
    // operands at the RF output and executes; the one in stage 0 (accepted
    // last cycle) samples its round.
    if (const auto& ready = arithmetic_reads_.last(); ready.has_value()) execute_arithmetic(*ready, w);
    if (const auto& first = arithmetic_reads_.stage(0); first.has_value()) {
        auto sampled = *first;
        for (std::size_t i = 0; i < sampled.reads.size(); ++i) {
            sampled.operands[i] = sample(sampled.warp, sampled.reads[i]);
        }
        sampled.uniform_operands = sample_uniform(sampled.warp, sampled.uniform_reads);
        // The write mask joins the round: its lanes AND the issue cycle's
        // active mask are the result's lane enables.
        if (has_write_mask(sampled.instruction)) {
            sampled.active_mask &= write_mask_lanes(warps_[sampled.warp], sampled.instruction);
        }
        w.arithmetic_sampled = std::move(sampled);
    }
    // The dispatch pipeline's stage 0: the instruction accepted last cycle
    // samples its round.
    const auto& pipe = dispatch_pipe_.cur();
    if (const auto& first = pipe.front(); first.has_value()) {
        const auto& op = *first;
        const bool masked = has_write_mask(op.instruction);
        if (!(op.reads.empty() && op.uniform_reads.empty() && !masked)) {
            auto operands = op.operands;
            for (std::size_t i = 0; i < op.reads.size(); ++i) operands[i] = sample(op.warp, op.reads[i]);
            const auto rounds = uniform_read_rounds(op.instruction);
            w.dispatch_samples.push_back({0, false, std::move(operands), 0,
                sample_uniform(op.warp, rounds.empty() ? std::vector<uint8_t>{} : rounds.front()),
                masked ? std::optional<uint64_t>{write_mask_lanes(warps_[op.warp], op.instruction)} : std::nullopt});
        }
    }
    // The URF write pipeline's stage 0: uniform arithmetic samples its
    // uniform sources (in operand order; a register named twice is one read),
    // r2ur its vector source (lane 0 is what it keeps).
    if (const auto& write = urf_pipe_.cur().front(); write.has_value()) {
        const auto& inst = write->instruction;
        if (inst.opcode == Opcode::r2ur) {
            w.urf_sample = std::vector<uint32_t>{sample(write->warp, inst.src0).at(0)};
        } else {
            const auto sources = uniform_sources(inst);
            uniform_reads += static_cast<uint32_t>(rf_read_round(sources).size());
            std::vector<uint32_t> values;
            for (const auto reg : sources) values.push_back(read_urf(write->warp, reg, "URF read: uniform ALU sample"));
            w.urf_sample = std::move(values);
        }
    }
    // The round counter's round now (src/timing.hpp uniform_read_rounds): its
    // registers by operand position, in the counter's warp. A tma's pair and
    // its tag enter the TMA operand read's URF read stages; an mma.async's
    // later round (round 0 is its dispatch entry's own sample, above) joins
    // its dispatch entry, the one issued at the counter's issue cycle.
    if (const auto& rounds = urf_read_rounds_.cur(); rounds.has_value()) {
        const auto& inst = rounds->header.instruction;
        const auto plan = uniform_read_rounds(inst);
        const auto& registers = plan.at(rounds->round);
        if (is_tma_opcode(inst.opcode)) {
            TmaReadPair pair;
            pair.round = rounds->round;
            pair.last = rounds->round + 1 == plan.size();
            if (rounds->round == 0) pair.header = rounds->header;
            for (const auto reg : registers) {
                pair.words.push_back(read_urf(rounds->header.warp, reg, "URF read: a tma's round"));
            }
            uniform_reads += static_cast<uint32_t>(registers.size());
            w.tma_sampled = std::move(pair);
        } else if (rounds->round > 0) {
            // Its dispatch entry, accepted at the counter's issue cycle, is in
            // stage (now - issue - 1).
            const std::size_t stage = static_cast<std::size_t>(cycle_ - rounds->header.issue_cycle - issue_to_sample);
            if (stage >= pipe.size() || !pipe[stage].has_value() || pipe[stage]->warp != rounds->header.warp) {
                throw SimError("internal: a later uniform round lost its dispatch entry");
            }
            std::size_t first = 0;
            for (uint32_t round = 0; round < rounds->round; ++round) first += plan[round].size();
            std::vector<uint32_t> values;
            for (const auto reg : registers) {
                values.push_back(read_urf(rounds->header.warp, reg, "URF read: an mma.async's later round"));
            }
            uniform_reads += static_cast<uint32_t>(registers.size());
            w.dispatch_samples.push_back({stage, true, {}, first, std::move(values), std::nullopt});
        }
    }
    if (uniform_reads > urf_read_ports) {
        throw SimError("internal: the URF sampled " + std::to_string(uniform_reads) + " uniform registers at cycle " +
            std::to_string(cycle_) + " (it has " + std::to_string(urf_read_ports) + " read ports)");
    }
    const auto& owned = issue_read_banks_.cur();
    for (uint32_t bank = 0; bank < config_.rf_banks; ++bank) {
        uint32_t async_reads = 0;
        for (std::size_t holder = 0; holder < w.rf_read.holders.size(); ++holder) {
            const auto& read = w.rf_read.holders[holder];
            if (w.rf_read.granted[holder] != 0 && read->reg % config_.rf_banks == bank) ++async_reads;
        }
        if (simt_reads[bank] != owned.at(bank) || simt_reads[bank] + async_reads > config_.rf_read_ports_per_bank) {
            throw SimError("internal: RF bank " + std::to_string(bank) + " sampled " +
                std::to_string(simt_reads[bank]) + " issued and " + std::to_string(async_reads) +
                " async reads (issue register: " + std::to_string(owned.at(bank)) + ") at cycle " +
                std::to_string(cycle_));
        }
    }
}

// The RF write decision's state, per bank:
//   async_owed_next = simt_booked_this_cycle OR (async_owed AND NOT async_grant)
// (a successful SIMT booking sets it, an actual async grant clears it, set
// wins); the cursor moves past the holder granted last, only on a grant.
void Machine::drive_rf_write_policy(const Wires& w) {
    for (uint32_t bank = 0; bank < config_.rf_banks; ++bank) {
        const auto& cur = rf_write_policy_.cur()[bank];
        const auto& decision = w.rf_write.banks[bank];
        const bool booked = w.issue.rf_write_bank == bank;
        RfWritePolicy next;
        next.owed = booked || (cur.owed != 0 && !decision.granted) ? 1 : 0;
        next.cursor = decision.cursor_after;
        if (next != cur) rf_write_policy_.set_next(bank, next, "drive_rf_write_policy");
    }
}

// The RF read decision's state, per bank (the write decision's, mirrored):
//   read_owed_next = simt_accepted_using_bank OR (read_owed AND NOT read_grant)
// (an accepted instruction whose round uses the bank sets it, an actual async
// read grant clears it, set wins); the cursor moves past the reader granted
// last, only on a grant.
void Machine::drive_rf_read_policy(const Wires& w) {
    for (uint32_t bank = 0; bank < config_.rf_banks; ++bank) {
        const auto& cur = rf_read_policy_.cur()[bank];
        const auto& decision = w.rf_read.banks[bank];
        const bool accepted = !w.issue.rf_read_banks.empty() && w.issue.rf_read_banks[bank] != 0;
        RfReadPolicy next;
        next.owed = accepted || (cur.owed != 0 && !decision.granted) ? 1 : 0;
        next.cursor = decision.cursor_after;
        if (next != cur) rf_read_policy_.set_next(bank, next, "drive_rf_read_policy");
    }
}

// The arithmetic pipeline's operand latches: the instruction in stage 0 goes
// on with the round it sampled now; the one in the last stage executed now
// and leaves; issue's accepted instruction enters stage 0.
void Machine::drive_arithmetic_reads(const Wires& w) {
    if (w.arithmetic_sampled.has_value()) {
        arithmetic_reads_.replace(0, w.arithmetic_sampled, "drive_arithmetic_reads: the sample");
    }
    if (w.issue.arithmetic.has_value()) arithmetic_reads_.push(*w.issue.arithmetic, "drive_arithmetic_reads");
}

// The RF write calendar: this cycle's bookings leave (the last stage); a
// booking made now enters at the stage of its cycle, added to the bank's
// count there.
void Machine::drive_write_calendar(const Wires& w) {
    if (!w.issue.rf_write_stage.has_value()) return;
    const auto stage = *w.issue.rf_write_stage;
    auto booked = write_calendar_.stage(stage).value_or(std::vector<uint32_t>(config_.rf_banks, 0));
    ++booked.at(w.issue.rf_write_bank.value());
    write_calendar_.replace(stage, std::move(booked), "drive_write_calendar");
}

// The issue register's bank use: the round of the instruction accepted now
// samples in the next cycle; nothing accepted leaves it empty.
void Machine::drive_issue_read_banks(const Wires& w) {
    for (uint32_t bank = 0; bank < config_.rf_banks; ++bank) {
        const uint8_t next = w.issue.rf_read_banks.empty() ? 0 : w.issue.rf_read_banks[bank];
        if (next != issue_read_banks_.cur()[bank]) issue_read_banks_.set_next(bank, next, "drive_issue_read_banks");
    }
}

// The RF write schedule: the writes executed now leave it; this cycle's new
// bookings join it in wire order.
// The RF write pipeline: the writes landed now leave; the results booked
// now (executed arithmetic) and the async writes granted now join.
void Machine::drive_writebacks(const Wires& w) {
    const auto& cur = writebacks_.cur();
    const bool landed = std::any_of(cur.begin(), cur.end(),
        [&](const Writeback& writeback) { return writeback.land_cycle <= cycle_; });
    if (!landed && w.new_writebacks.empty()) return;
    std::deque<Writeback> next;
    for (const auto& writeback : cur) {
        if (writeback.land_cycle > cycle_) next.push_back(writeback);
    }
    for (const auto& writeback : w.new_writebacks) next.push_back(writeback);
    writebacks_.set_next(std::move(next), "drive_writebacks");
}

// A masked instruction's write mask as lanes, read from the predicate
// file in the instruction's operand round, as a branch reads its predicate:
// the lanes whose predicate bit differs from the invert bit (PT is set in
// every lane, so @!PT is no lane). Sixteen XORs, beside the round's reads.
uint64_t Machine::write_mask_lanes(const Warp& warp, const Instruction& inst) const {
    const uint64_t bits =
        inst.control.predicate == 7 ? full_mask() : predicates_.cur().at(predicate_index(warp.id, inst.control.predicate));
    return (inst.control.predicate_invert ? ~bits : bits) & full_mask();
}

// Where warp `warp`'s register `reg` is: bank reg mod rf.banks, row
// warp x rows per warp + reg / rf.banks. Rows per warp is the kernel's
// register count over the banks, rounded up (bank 0 holds the most of a
// warp's registers), the same in every bank; the launch fits when active
// warps x that many rows fit rf.rows_per_bank (checked when the machine is
// built). So the kernel's register count sets each warp's stride, and the
// warps sit one after another from row 0.
Machine::RfAddress Machine::rf_address(uint32_t warp, uint32_t reg) const {
    return RfAddress{reg % config_.rf_banks, warp * rf_rows_per_warp_ + reg / config_.rf_banks};
}

// Every lane of one register, read through a read port of its bank now.
std::vector<uint32_t> Machine::read_rf(uint32_t warp, uint8_t reg, const char* tag) {
    const auto address = rf_address(warp, reg);
    const auto word = rf_.read(address.bank, address.row, tag);
    return std::vector<uint32_t>(word.begin(), word.end());
}

// A fixed result on its way to the RF write port it booked (port_cycle),
// landing Wd later. Its visibility was recorded at issue.
Machine::Writeback Machine::register_writeback(
    const Warp& warp,
    uint8_t dst,
    uint64_t port_cycle,
    std::vector<uint32_t> values) const {
    Writeback writeback;
    writeback.port_cycle = port_cycle;
    writeback.land_cycle = port_cycle + config_.rf_write_depth_cycles;
    writeback.warp = warp.id;
    writeback.dst = dst;
    writeback.values = std::move(values);
    return writeback;
}

// A setp result: the predicate file (flip-flops) takes it at the edge of its
// port cycle.
Machine::Writeback Machine::predicate_writeback(
    const Warp& warp,
    uint8_t dst,
    uint64_t port_cycle,
    std::vector<uint32_t> values) const {
    Writeback writeback;
    writeback.port_cycle = port_cycle;
    writeback.land_cycle = port_cycle;
    writeback.warp = warp.id;
    writeback.predicate_dst = dst;
    writeback.values = std::move(values);
    return writeback;
}

} // namespace nanosm

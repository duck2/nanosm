// The vector register file: the write calendar, the read and write port decisions, the
// write ports' pipelines, the RAM stage, the issue register's read port and the predicate
// files.
// (Machine is one class, src/machine/machine.hpp; the map of its files is at the top of
// src/machine/machine.cpp.)

#include "machine/machine.hpp"

#include <algorithm>
#include <utility>

namespace nanosm {

// ===========================================================================
// The RF ports: the write calendar, the RF read and write decisions, the
// write port and the write pipelines (every cycle number follows
// src/timing.hpp)
// ===========================================================================

// Fixed issue never contests the current cycle's RF ports: the issue
// register's read round samples at issue + 1 through the ports it owns (each
// unit's stage 0 reads its operands through read_rf_round), its fixed write
// is booked in the write calendar at issue + issue_to_port and presented by
// its arithmetic unit at that cycle. What the current cycle
// leaves goes to the async clients, by one protocol:
//   PRESENT  each async writer and reader presents at most one request a
//            cycle, from its own Q, built in its own file: a writer the write
//            it would make (RfWrite), a reader a register (and a pair's
//            second register). tick() wires them to the RF.
//   GRANT    per bank, round-robin from the bank's cursor over the requests
//            in their fixed order, as many as the bank has ports the issue
//            register's round (reads) or the calendar's bookings (writes)
//            leave free. A granted write is presented at the write port now;
//            a granted read samples the RF now and its words go back with the
//            grant. Each client takes its grant in its own file.
//   INHIBIT  a writer denied with its bank's async_owed bit set, or a reader
//            denied in a cycle whose accepted instruction takes every read
//            port of the bank too, sets the bank's inhibit bit: from the next
//            cycle issue makes no booking to the bank / accepts no read of
//            it, so a port is free for the denied client.
// Each inhibit bit is a register: issue sees a denial one cycle after it.
// A read SAMPLES the RF at its cycle and has its operand Lr cycles later; a
// write PRESENTED at the port lands at the clock edge Wd cycles later and is
// visible to samples from the next cycle (no same-edge forwarding).

// The calendar's stage that holds the bookings for the cycle `delay` cycles
// from now: the last stage is this cycle, each stage before it one cycle
// later. An instruction's delay is fixed by its opcode and the profile, so
// its stage is the same at every attempt.
std::size_t Machine::write_calendar_stage(uint32_t delay) const {
    if (delay >= write_calendar_.stages()) throw SimError("internal RF write booking beyond the write calendar");
    return write_calendar_.stages() - 1 - delay;
}

bool Machine::write_slot_free(uint32_t bank, uint32_t delay) const {
    const auto& booked = write_calendar_.stage(write_calendar_stage(delay));
    return !booked.has_value() || booked->at(bank) < config_.rf_write_ports_per_bank;
}

// The booking enters the calendar at this edge (drive_write_calendar), at the
// stage of its cycle.
void Machine::reserve_write(uint32_t bank, uint32_t delay, Wires& w) {
    if (delay == 0) throw SimError("internal RF write booking for the current cycle");
    if (!write_slot_free(bank, delay)) throw SimError("internal RF write booking over a full port");
    w.issue.rf_write_stage = write_calendar_stage(delay);
}

// The calendar's last stage: a bank's writes booked (in earlier cycles) for
// this cycle.
uint32_t Machine::writes_booked_now(uint32_t bank) const {
    const auto& now = write_calendar_.last();
    return now.has_value() ? now->at(bank) : 0;
}

// The RF write decision (make an opening, then choose a writer): one per RF
// bank, from the writers' requests (Q).
//   async_ready  = a request targets the bank
//   async_grant  = async_ready AND one of this cycle's write ports is unbooked
// The requests granted (round-robin from the bank's cursor, in the writers'
// line order) are presented at the write port now; the owed bit, the inhibit
// bit and the cursor take their next values in drive_rf_write_policy. Old
// bookings always execute; issue's outcome cannot change these grants. A
// bank that is ready, not granted and owed refuses issue's bookings from the
// next cycle (its inhibit bit).
Machine::RfWriteDecision Machine::decide_rf_writes(const RfWriteRequests& requests) const {
    RfWriteDecision decision;
    decision.granted.writeback.assign(requests.writeback.size(), 0);
    const auto& policy = rf_write_policy_.cur();
    for (uint32_t bank = 0; bank < config_.rf_banks; ++bank) {
        std::vector<uint8_t> ready(requests.size(), 0);
        for (std::size_t line = 0; line < ready.size(); ++line) {
            const auto& write = requests[line];
            ready[line] = write.has_value() && write->reg % config_.rf_banks == bank ? 1 : 0;
        }
        auto& state = decision.banks[bank];
        state.fixed = writes_booked_now(bank);
        const auto choice = round_robin_grant(ready, policy.at(bank).cursor, config_.rf_write_ports_per_bank - state.fixed);
        for (const auto line : choice.granted) decision.granted[line] = 1;
        state.ready = std::find(ready.begin(), ready.end(), 1) != ready.end();
        state.granted = !choice.granted.empty();
        state.ports = static_cast<uint32_t>(choice.granted.size());
        state.cursor_after = choice.cursor_after;
    }
    return decision;
}

// The RF write port at this cycle: every fixed result booked for it (`fixed`:
// at its unit's output now), then the async writes the RF write decision
// granted (never on a booked port, in the writers' line order), are
// presented, each on the next free write port of its bank, and enter that
// port's write pipeline. From its grant on the write pipeline owns a granted
// write's data.
void Machine::rf_write_port(const FixedRfWrites& fixed, const RfWriteRequests& requests, Wires& w) {
    w.rf_port_writes.assign(rf_write_pipes_.size(), std::nullopt);
    std::vector<uint32_t> port_uses(config_.rf_banks, 0);
    const auto present = [&](RfWrite write) {
        const auto bank = write.reg % config_.rf_banks;
        if (port_uses[bank] == config_.rf_write_ports_per_bank) {
            throw SimError("internal: more RF writes than write ports in bank " +
                std::to_string(bank) + " at cycle " + std::to_string(cycle_));
        }
        trace_->rf_write(cycle_, write.warp, write.reg, bank);
        w.rf_port_writes[std::size_t{bank} * config_.rf_write_ports_per_bank + port_uses[bank]++] = std::move(write);
    };
    for (const auto& write : fixed) present(write);
    for (uint32_t bank = 0; bank < config_.rf_banks; ++bank) {
        if (port_uses[bank] != w.rf_write.banks[bank].fixed) {
            throw SimError("internal: RF bank " + std::to_string(bank) + " was booked for " +
                std::to_string(w.rf_write.banks[bank].fixed) + " fixed writes and got " + std::to_string(port_uses[bank]) +
                " at cycle " + std::to_string(cycle_));
        }
    }
    // (T) The trace counts the async writes and the ready writers not written.
    for (std::size_t line = 0; line < requests.size(); ++line) {
        if (!requests[line].has_value()) continue;
        if (w.rf_write.granted[line] == 0) {
            trace_->rf_async_wait();
            continue;
        }
        trace_->rf_async_write();
        present(*requests[line]);
    }
}

// The RAM stage: each write port's last stage holds the write at the RAM now.
// It lands at this cycle's clock edge (land_rf_writes, after every read of
// the cycle). Its completion is presented now: a cleared depbar counter is 0
// in Q from the next cycle -- the write's visibility -- so the waiting warp
// issues then at the earliest; an async result is visible to the sanitizer
// from the next cycle.
void Machine::rf_ram_stage(Wires& w) {
    for (const auto& pipe : rf_write_pipes_) {
        const auto& write = pipe.last();
        if (!write.has_value()) continue;
        if (write->clear_depbar.has_value()) {
            record_depbar_completion(w, write->warp, *write->clear_depbar,
                "depbar underflow on sb" + std::to_string(*write->clear_depbar));
        }
        for (const auto reg : write->visible_regs) sanitizer_.record_reg_ready(write->warp, reg, cycle_ + 1);
    }
}

// The RF writes at their RAM now (each write port's last stage): each goes
// through its own port, its lanes enabled one by one, and lands at the clock
// edge.
void Machine::land_rf_writes() {
    for (std::size_t port = 0; port < rf_write_pipes_.size(); ++port) {
        const auto& write = rf_write_pipes_[port].last();
        if (!write.has_value()) continue;
        if (write->values.size() != config_.lanes) throw SimError("internal: an RF write of the wrong lane count");
        const auto address = rf_address(write->warp, write->reg);
        rf_.write(address.bank, address.row, write->values, write->lane_enables, "RF write port",
            static_cast<uint32_t>(port % config_.rf_write_ports_per_bank));
    }
}

// The predicate files (flip-flops): a predicate presented now (a setp's at
// its port cycle, a usetp's at the URF write port's) is in the file from the
// next cycle, in its enabled lanes. One write per predicate is ever in flight
// (src/timing.hpp), so each register takes at most one.
void Machine::drive_predicates(const Wires& w) {
    for (const auto& write : w.predicate_writes) {
        if (write.predicate >= predicates_per_warp) throw SimError("internal: a write of predicate p" +
            std::to_string(write.predicate));
        const auto index = predicate_index(write.warp, write.predicate);
        const uint64_t next = (predicates_.cur()[index] & ~write.lane_enables) | (write.bits & write.lane_enables);
        predicates_.set_next(index, next, "drive_predicates");
    }
}

// The async RF readers' decision, per RF bank, from their requests and the
// issue register's bank use (`round_banks`: the ports its round samples now):
//   read_ready   = a reader's request is for the bank
//   read_grant   = read_ready AND the issue register's round leaves one of
//                  the bank's read ports free
// A bank that is ready and not granted, in a cycle whose accepted
// instruction takes every read port of it, sets its read inhibit bit
// (drive_rf_read_policy): the next cycle issue accepts no instruction whose
// round uses the bank, so the issue register leaves it free the cycle after
// -- the hole the denied reader is owed. (If this cycle's accepted
// instruction leaves a port of the bank free, the issue register leaves it
// free next cycle anyway, and no bubble is needed.) Reads already accepted
// are never stopped. The readers granted (round-robin from the bank's
// cursor, in RfReader order) sample the RF now, and their words go back in
// the grant (at the RF output Lr later). The second register of a pair is
// outside the round-robin: it takes a port of its bank only if one is left
// after it (in RfReader order: the write side's pair, then the command
// slot's), so a pair never waits for two holes; if not granted now it is
// asked for later on its own. A denied reader holds its request. The
// inhibit bit and the cursor change in drive_rf_read_policy.
Machine::RfReadDecision Machine::decide_rf_reads(const RfReadRequests& requests,
    const std::vector<uint8_t>& round_banks) {
    static constexpr std::array<const char*, rf_readers> tags{"RF read: acc.st beat", "RF read: store data",
        "RF read: a shared-memory collect stage", "RF read: a shared-memory collect stage",
        "RF read: a load / store unit command slot"};
    RfReadDecision decision;
    std::array<bool, rf_readers> granted{};
    std::array<bool, rf_readers> pair_granted{};
    const auto& policy = rf_read_policy_.cur();
    std::vector<uint32_t> ports_used(config_.rf_banks, 0);
    for (uint32_t bank = 0; bank < config_.rf_banks; ++bank) {
        std::vector<uint8_t> ready(rf_readers, 0);
        for (std::size_t line = 0; line < ready.size(); ++line) {
            const auto& read = requests[line];
            ready[line] = read.has_value() && read->reg % config_.rf_banks == bank ? 1 : 0;
        }
        auto& state = decision.banks[bank];
        state.fixed = round_banks.at(bank);
        const auto choice = round_robin_grant(ready, policy.at(bank).cursor, config_.rf_read_ports_per_bank - state.fixed);
        for (const auto line : choice.granted) granted[line] = true;
        ports_used[bank] = state.fixed + static_cast<uint32_t>(choice.granted.size());
        state.ready = std::find(ready.begin(), ready.end(), 1) != ready.end();
        state.granted = !choice.granted.empty();
        state.cursor_after = choice.cursor_after;
    }
    for (std::size_t line = 0; line < rf_readers; ++line) {
        const auto& read = requests[line];
        if (!granted[line] || !read->pair.has_value()) continue;
        const auto bank = *read->pair % config_.rf_banks;
        if (ports_used[bank] < config_.rf_read_ports_per_bank) {
            pair_granted[line] = true;
            decision.banks[bank].granted = true;
            ++ports_used[bank];
        }
    }
    for (uint32_t bank = 0; bank < config_.rf_banks; ++bank) decision.banks[bank].ports = ports_used[bank] - decision.banks[bank].fixed;
    // The granted reads sample the RF now: every named read, then the pairs.
    const auto sample = [&](uint32_t warp, uint8_t reg, std::size_t line) {
        trace_->rf_read(cycle_, warp, reg, reg % config_.rf_banks);
        return LaneWords(read_rf(warp, reg, tags[line]));
    };
    for (std::size_t line = 0; line < rf_readers; ++line) {
        if (granted[line]) decision.granted[line].words = sample(requests[line]->warp, requests[line]->reg, line);
    }
    for (std::size_t line = 0; line < rf_readers; ++line) {
        if (pair_granted[line]) decision.granted[line].pair_words = sample(requests[line]->warp, *requests[line]->pair, line);
    }
    return decision;
}

// The RF read stage (src/timing.hpp): the round of the instruction in the
// issue register (accepted last cycle) samples the RF now, before this
// cycle's writes land at its edge (READ_FIRST), through the read ports the
// round owns (the issue register's bank use, which the async readers' decision
// left it). Each unit's stage 0 samples its own operands through this port
// (tick()'s read stage); the samples are counted for check_round_reads.
std::vector<uint32_t> Machine::read_rf_round(Wires& w, uint32_t warp, uint8_t reg) {
    ++w.round_reads.rf.at(reg % config_.rf_banks);
    trace_->rf_read(cycle_, warp, reg, reg % config_.rf_banks);
    return read_rf(warp, reg, "RF read: the issue register's round");
}

// After the read stage: each bank's samples -- the round's and the async
// readers' -- never exceed its read ports, and the round's are exactly the
// issue register's bank use.
void Machine::check_round_reads(const Wires& w) const {
    for (uint32_t bank = 0; bank < config_.rf_banks; ++bank) {
        const auto round = w.round_reads.rf[bank];
        const auto& decision = w.rf_read.banks[bank];
        if (round != decision.fixed || round + decision.ports > config_.rf_read_ports_per_bank) {
            throw SimError("internal: RF bank " + std::to_string(bank) + " sampled " + std::to_string(round) +
                " issued and " + std::to_string(decision.ports) + " async reads (issue register: " +
                std::to_string(decision.fixed) + ") at cycle " + std::to_string(cycle_));
        }
    }
}

// The RF write decision's state, per bank:
//   async_owed_next = simt_booked_this_cycle OR (async_owed AND NOT async_grant)
// (a successful SIMT booking sets it, an actual async grant clears it, set
// wins);
//   inhibit_next    = async_ready AND NOT async_grant AND async_owed
// (issue refuses every booking to the bank while it is set: from the cycle
// after a denial with service owed through the cycle of the grant); the cursor
// moves past the writer granted last, only on a grant.
void Machine::drive_rf_write_policy(const Wires& w) {
    for (uint32_t bank = 0; bank < config_.rf_banks; ++bank) {
        const auto& cur = rf_write_policy_.cur()[bank];
        const auto& decision = w.rf_write.banks[bank];
        const bool booked = w.issue.rf_write_bank == bank;
        RfWritePolicy next;
        next.owed = booked || (cur.owed != 0 && !decision.granted) ? 1 : 0;
        next.inhibit = decision.ready && !decision.granted && cur.owed != 0 ? 1 : 0;
        next.cursor = decision.cursor_after;
        if (next != cur) rf_write_policy_.set_next(bank, next, "drive_rf_write_policy");
    }
}

// The RF read decision's state, per bank:
//   inhibit_next = read_ready AND NOT read_grant
//                  AND simt_accepted_takes_every_port
// (issue refuses every instruction whose round uses the bank while it is
// set). A reader is denied only when the issue register's round holds every
// read port of the bank. The last term: the round accepted now, the issue
// register's next cycle, takes every port of the bank too; if it leaves one
// free, the reader gets it next cycle without a bubble. The cursor moves past
// the reader granted last, only on a grant. (The read side needs no owed bit
// like the write side's: a reader is denied only by the round accepted the
// cycle before, so service is always owed at a denial.)
void Machine::drive_rf_read_policy(const Wires& w) {
    for (uint32_t bank = 0; bank < config_.rf_banks; ++bank) {
        const auto& cur = rf_read_policy_.cur()[bank];
        const auto& decision = w.rf_read.banks[bank];
        const bool takes_every_port =
            !w.issue.rf_read_banks.empty() && w.issue.rf_read_banks[bank] >= config_.rf_read_ports_per_bank;
        RfReadPolicy next;
        next.inhibit = decision.ready && !decision.granted && takes_every_port ? 1 : 0;
        next.cursor = decision.cursor_after;
        if (next != cur) rf_read_policy_.set_next(bank, next, "drive_rf_read_policy");
    }
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

// The write ports' pipelines: each takes the write presented on it now.
void Machine::drive_rf_write_pipes(const Wires& w) {
    for (std::size_t port = 0; port < w.rf_port_writes.size(); ++port) {
        if (w.rf_port_writes[port].has_value()) rf_write_pipes_[port].push(*w.rf_port_writes[port], "drive_rf_write_pipes");
    }
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

} // namespace nanosm

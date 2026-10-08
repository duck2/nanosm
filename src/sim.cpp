#include "sim.hpp"
#include "mma_numeric.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace nanosm {

uint32_t float_bits(float value) {
    return std::bit_cast<uint32_t>(value);
}

float bits_float(uint32_t value) {
    return std::bit_cast<float>(value);
}

uint32_t half_to_float_bits(uint16_t half) {
    const uint32_t sign = static_cast<uint32_t>(half & 0x8000) << 16;
    uint32_t exponent = (half >> 10) & 0x1f;
    uint32_t fraction = half & 0x3ff;
    if (exponent == 0) {
        if (fraction == 0) return sign;
        int shift = 0;
        while ((fraction & 0x400) == 0) { fraction <<= 1; ++shift; }
        fraction &= 0x3ff;
        return sign | static_cast<uint32_t>(127 - 14 - shift) << 23 | fraction << 13;
    }
    if (exponent == 31) return sign | 0x7f800000u | fraction << 13;
    return sign | (exponent + 112) << 23 | fraction << 13;
}

uint16_t float_to_half_bits(uint32_t bits) {
    const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000);
    const uint32_t exponent = (bits >> 23) & 0xff;
    const uint32_t fraction = bits & 0x7fffff;
    if (exponent == 0xff) {
        if (fraction == 0) return sign | 0x7c00; // infinity
        // NaN: keep the truncated payload so half NaNs round-trip; force the
        // quiet bit when truncation would make the payload vanish.
        const auto payload = static_cast<uint16_t>(fraction >> 13);
        return static_cast<uint16_t>(sign | 0x7c00 | (payload != 0 ? payload : 0x200));
    }
    const int32_t e = static_cast<int32_t>(exponent) - 127 + 15; // half-biased exponent
    if (e >= 31) return sign | 0x7c00; // overflow to infinity
    const uint32_t significand = fraction | (exponent != 0 ? 0x800000u : 0u);
    // Round to nearest even while dropping 13 bits (normal) or more
    // (subnormal destination). Everything below the round bit is sticky.
    const uint32_t shift = e <= 0 ? static_cast<uint32_t>(14 - e) : 13u;
    if (shift >= 25) return sign; // rounds to signed zero
    const uint32_t keep = significand >> shift;
    const uint32_t round = (significand >> (shift - 1)) & 1u;
    const uint32_t sticky = significand & ((1u << (shift - 1)) - 1u);
    const uint32_t rounded = keep + (round && (sticky || (keep & 1u)));
    // Reassemble arithmetically so a rounding carry ripples into the
    // exponent: subnormal->normal promotion and near-max overflow to
    // infinity both fall out of the addition.
    const uint32_t result = e <= 0 ? rounded
        : (static_cast<uint32_t>(e) << 10) + (rounded - 0x400u);
    return static_cast<uint16_t>(sign | result);
}

namespace {

bool is_setp(Opcode opcode) {
    return (opcode >= Opcode::setp_eq && opcode <= Opcode::setp_ge) ||
        (opcode >= Opcode::setpi_eq && opcode <= Opcode::setpi_ge);
}

bool is_setp_immediate(Opcode opcode) {
    return opcode >= Opcode::setpi_eq && opcode <= Opcode::setpi_ge;
}

bool is_mbar_test_wait(Opcode opcode) {
    return opcode == Opcode::mbar_test_wait_i || opcode == Opcode::mbar_test_wait_r;
}

bool is_mbar_try_wait(Opcode opcode) {
    return opcode == Opcode::mbar_try_wait_i || opcode == Opcode::mbar_try_wait_r;
}

bool is_mbar_predicate_writer(Opcode opcode) {
    return is_mbar_test_wait(opcode) || is_mbar_try_wait(opcode);
}

bool is_mbar_rmw(Opcode opcode) {
    return opcode == Opcode::mbar_init_i || opcode == Opcode::mbar_init_r ||
        opcode == Opcode::mbar_arrive ||
        opcode == Opcode::mbar_arrive_expect_tx_i ||
        opcode == Opcode::mbar_arrive_expect_tx_r;
}

bool is_tma(Opcode opcode) {
    return opcode == Opcode::tma_async;
}

bool is_sfu(Opcode opcode) {
    return opcode == Opcode::ex2 || opcode == Opcode::rcp;
}

// Every new float opcode canonicalizes NaN outputs to 0x7fc00000: the sim is
// the spec, and platform NaN-payload propagation must not leak into it.
// (fadd predates this rule and keeps its legacy behavior.)
uint32_t canonical_float(float value) {
    if (std::isnan(value)) return 0x7fc00000u;
    return float_bits(value);
}

// ex2 = 2^x computed in double, narrowed once with RNE. std::exp2(double) is
// the single libm call in the spec; tests pin exact result bits so platform
// drift breaks loudly instead of silently changing the machine.
uint32_t sfu_ex2(uint32_t bits) {
    const float x = bits_float(bits);
    if (std::isnan(x)) return 0x7fc00000u;
    return canonical_float(static_cast<float>(std::exp2(static_cast<double>(x))));
}

// rcp = 1/x as one double division, narrowed once with RNE. Hardware IEEE
// division, no libm: deterministic everywhere.
uint32_t sfu_rcp(uint32_t bits) {
    const float x = bits_float(bits);
    if (std::isnan(x)) return 0x7fc00000u;
    return canonical_float(static_cast<float>(1.0 / static_cast<double>(x)));
}

// IEEE maxNum: one NaN yields the other operand; +0 is preferred over -0.
uint32_t float_max_bits(uint32_t a_bits, uint32_t b_bits) {
    const float a = bits_float(a_bits);
    const float b = bits_float(b_bits);
    if (std::isnan(a) && std::isnan(b)) return 0x7fc00000u;
    if (std::isnan(a)) return b_bits;
    if (std::isnan(b)) return a_bits;
    if (a == 0.0f && b == 0.0f) {
        return (std::signbit(a) && std::signbit(b)) ? a_bits : float_bits(0.0f);
    }
    return a > b ? a_bits : b_bits;
}

bool compare_setp(Opcode opcode, uint32_t lhs, uint32_t rhs) {
    switch (opcode) {
    case Opcode::setp_eq:
    case Opcode::setpi_eq: return lhs == rhs;
    case Opcode::setp_ne:
    case Opcode::setpi_ne: return lhs != rhs;
    case Opcode::setp_lt:
    case Opcode::setpi_lt: return lhs < rhs;
    case Opcode::setp_le:
    case Opcode::setpi_le: return lhs <= rhs;
    case Opcode::setp_gt:
    case Opcode::setpi_gt: return lhs > rhs;
    case Opcode::setp_ge:
    case Opcode::setpi_ge: return lhs >= rhs;
    default: throw SimError("internal non-setp comparison");
    }
}

} // namespace

Machine::Machine(Config config, Kernel kernel, Launch launch, TraceOptions trace_options)
    : config_(config),
      kernel_(std::move(kernel)) {
    try {
        validate_device_profile(config_);
    } catch (const ProfileError& error) {
        throw SimError(error.what());
    }
    if (config_.lanes == 0 || config_.rf_banks == 0 || config_.shmem_banks == 0 ||
        config_.gmem_bytes == 0 || kernel_.register_count > 256 ||
        config_.gmem_bytes > std::numeric_limits<std::size_t>::max() ||
        config_.gmem_sector_bytes == 0 || config_.gmem_bus_bytes_per_dram_tick == 0 ||
        config_.gmem_sector_bytes % 4 != 0 || config_.gmem_queue_sectors == 0 ||
        kernel_.words.empty()) {
        throw SimError("invalid zero-sized or unencodable machine resource");
    }
    const auto expected_profile = device_profile_fingerprint(config_);
    if (kernel_.profile_name != config_.name || kernel_.profile_fingerprint != expected_profile) {
        throw SimError(
            "kernel profile '" + kernel_.profile_name + "@" +
            format_profile_fingerprint(kernel_.profile_fingerprint) + "' does not match simulator '" +
            config_.name + "@" + format_profile_fingerprint(expected_profile) + "'");
    }
    if (config_.mbar_wait_slots == 0) {
        for (const auto word : kernel_.words) {
            if (is_mbar_try_wait(decode(word).opcode)) {
                throw SimError("mbar.try_wait is unsupported by this device profile");
            }
        }
    }
    if (config_.tma_queue_commands == 0) {
        for (const auto word : kernel_.words) {
            if (is_tma(decode(word).opcode)) {
                throw SimError("tma.async is unsupported by this device profile");
            }
        }
    }
    if (launch.active_warps == 0) throw SimError("active warp count is zero");
    for (const auto& acc : kernel_.accumulators) {
        const uint64_t words = static_cast<uint64_t>(acc.rows) * acc.cols;
        if (acc.bank >= config_.tmem_banks || acc.base_word % config_.tmem_fine_banks != 0 ||
            acc.rows < config_.lanes || acc.rows % 4 != 0 || acc.cols % 4 != 0 ||
            static_cast<uint64_t>(acc.base_word) + words > config_.tmem_words_per_bank) {
            throw SimError("invalid accumulator metadata for '" + acc.name + "'");
        }
    }
    for (uint32_t bank = 0; bank < config_.rf_banks; ++bank) {
        const auto rows_per_warp = kernel_.register_count <= bank
            ? 0
            : (kernel_.register_count - 1 - bank) / config_.rf_banks + 1;
        if (static_cast<uint64_t>(rows_per_warp) * launch.active_warps > config_.rf_rows_per_bank) {
            throw SimError(
                "kernel RF allocation cannot fit " + std::to_string(launch.active_warps) +
                " active warps (bank " + std::to_string(bank) + ")");
        }
    }
    try {
        shmem_.assign(config_.shmem_bytes, 0);
        gmem_.assign(static_cast<std::size_t>(config_.gmem_bytes), 0);
        tmem_.assign(config_.tmem_banks, std::vector<uint32_t>(config_.tmem_words_per_bank, 0));
        acc_transfers_.poke(AccSlots(config_.tmem_banks));
        mbar_wait_slots_.poke(WaitSlots(config_.mbar_wait_slots));
        read_reservations_.assign(
            config_.rf_arithmetic_read_cycles, std::vector<uint32_t>(config_.rf_banks, 0));
        for (uint32_t index = 0; index < config_.lsu_staging_vectors; ++index) {
            AssemblyRegister empty;
            empty.words.assign(config_.lanes, 0);
            empty.valid.assign(config_.lanes, 0);
            staging_.emplace_back(registers_, "assembly register " + std::to_string(index), std::move(empty));
        }
        free_id_tags_.poke(std::vector<uint8_t>(config_.lsu_sectors_in_flight, 0));
        TmaSectorRegister sector;
        sector.data.assign(config_.gmem_sector_bytes, 0);
        tma_sector_.poke(std::move(sector));
        TmaStoreRegister store;
        store.data.assign(config_.gmem_sector_bytes, 0);
        tma_store_sector_.poke(std::move(store));
        warps_.reserve(launch.active_warps);
        for (uint32_t id = 0; id < launch.active_warps; ++id) {
            Warp warp;
            warp.id = id;
            warp.rf.resize(static_cast<std::size_t>(config_.lanes) * kernel_.register_count, 0);
            for (auto& predicate : warp.predicates) predicate.resize(config_.lanes, 0);
            warps_.push_back(std::move(warp));
        }
        depbar_.poke(DepbarCounters(launch.active_warps));
        sanitizer_.record_launch(config_, kernel_, launch.active_warps);
    } catch (const std::bad_alloc&) {
        throw SimError("machine state does not fit in host memory");
    } catch (const std::length_error&) {
        throw SimError("machine state exceeds host container limits");
    }
    trace_.emplace(
        launch.active_warps, config_.core_clock_hz, config_.dram_clock_hz, trace_options);
    apply_image(shmem_, launch.shmem_image, "shmem");
    apply_image(gmem_, launch.gmem_image, "gmem");
}

void Machine::set_driver_shuffle_seed(uint64_t seed) {
    shuffle_.emplace(seed);
}

// ===========================================================================
// One core cycle (trash/stepping-discipline-2026-10-08.md).
//
// 1. WIRES. Every decision of the cycle is computed once, in the order below.
//    A decision reads register Q values (cur()), memories, and the wires
//    above it: requests flow toward the shared resource, the resource's one
//    select function grants, data flows downstream. No register's next value
//    is visible here. Same-cycle visibility exists only where a wire carries
//    it. The declared ones (each is today's behavior, kept):
//      (a) RF / predicate write-first: everything due at t lands before any
//          read at t (booked writebacks, the LSU's assembly drain, mbar
//          test/try-wait results).
//      (b) Completions reach issue in their own cycle: depbar clears, mbar
//          wait slots completed (wake, immediate success, timeout).
//      (c) Issue looks ahead: an instruction accepted at t occupies its
//          resources from t+1, so issue sees the space this cycle frees
//          (queue entries, slots, port A, assembly registers) and the TMA's
//          port-A request as it will stand at t+1 (TMA priority on the next
//          grant).
//      (d) Same-cycle space reuse: a register drained at t may be refilled at
//          t (the assembly register, the TMA sector / store registers, the
//          MMA event register, an LSU slot or acc slot reallocated by issue).
//          The refill's data is visible at t+1; nothing passes through.
//      (e) An issued stg.v8 makes its first RF read beat in its issue cycle.
//      (f) Inside the MMA: a result's tmem write precedes the same cycle's
//          accumulator read (write-first).
//      (g) The placeholder controller may put a request accepted at t on the
//          bus at t (outside the SM; the timeline test pins request -> beat 0).
//      (h) A port-A grant serves its first round in the grant cycle. For the
//          TMA's new transaction this is a declared combinational path:
//          latched TMA state (its queue and sector registers, Q; for a store
//          sector also this cycle's egress drain of the store register (d))
//          -> tma_requests builds the request -> select_port_a grants it ->
//          serve_port_a serves round 0, all in the grant cycle. No register
//          holds the transaction before round 0; the TMA port-A transaction
//          register latches only the remaining rounds. A held engine event's
//          claim likewise serves its read stage at once. A SIMT access
//          accepted at issue starts the next cycle. A new MMA fetch serves
//          its first port-B round in the cycle it starts.
// 2. DRIVERS. Each register's next value, from Q and the wires. They run in
//    any order (set_driver_shuffle_seed() permutes them every cycle).
// 3. CLOCK EDGE. Every register takes D; shmem takes port A's writes.
// ===========================================================================
void Machine::tick() {
    if (done()) return;
    sanitizer_.record_cycle_start();
    rf_requester_ = RfRequester::lsu_drain;
    Wires w;

    // RF write port, write-first (a). The LSU's assembly drain books this
    // cycle's write port first: other writes due now were booked earlier.
    w.lsu_drain = decide_lsu_rf_drain();
    execute_rf_writes(w);

    // The store path's last stage frees the TMA store register / a store
    // assembly register for a refill this same cycle (d).
    w.egress = decide_egress_write();

    // Request generation: independent of each other.
    if (shuffle_.has_value() && ((*shuffle_)() & 1u) != 0) {
        w.lsu_want = lsu_request_want();
        w.tma = tma_requests(w.egress);
    } else {
        w.tma = tma_requests(w.egress);
        w.lsu_want = lsu_request_want();
    }

    // The shared resources, one select each.
    w.channel = select_request_channel(w.tma, w.lsu_want);
    w.port_a = serve_port_a(select_port_a(w.tma), w.tma, w);

    // Units whose next state issue looks ahead at (c). The MMA's commit may
    // present into the event register the RMW engine claimed just now (d).
    step_mma(w);
    step_acc_transfers(w);        // RF ports: after the LSU drain, before issue
    decide_mbar_timeouts(w);      // a wake this cycle wins over the deadline
    w.ingress = decide_ingress(w);
    step_gmem_controller(w);
    step_tma(w);

    // The scheduler: the last decision of the cycle.
    decide_issue(w);

    // Below issue in RF read priority.
    w.lsu_rf_read = decide_lsu_rf_read(w);
    step_arithmetic_reads(w);

    run_drivers(w);

    // Clock edge.
    registers_.commit();
    for (const auto& write : w.port_a.writes) {
        std::memcpy(shmem_.data() + write.address, write.bytes.data(), write.bytes.size());
    }
    ++cycle_;
}

void Machine::run(uint64_t cycle_limit) {
    while (!done() && cycle_ < cycle_limit) tick();
    if (!done()) throw SimError("cycle cap at cycle " + std::to_string(cycle_));
    trace_->finish(cycle_);
}

bool Machine::done() const {
    const bool warps_halted = std::all_of(warps_.begin(), warps_.end(), [](const Warp& warp) {
        return warp.halted;
    });
    const auto& waits = mbar_wait_slots_.cur();
    const bool wait_slots_empty = std::all_of(
        waits.begin(), waits.end(), [](const auto& slot) { return !slot.has_value(); });
    const auto& tma = tma_.cur();
    const auto& mma = mma_.cur();
    const auto& acc = acc_transfers_.cur();
    return warps_halted && wait_slots_empty &&
        !port_a_.cur().has_value() && !tma_port_a_.cur().has_value() &&
        !tma_mbar_event_.cur().has_value() && !mma_mbar_event_.cur().has_value() &&
        !rmw_.cur().active.has_value() && !mbar_read_.cur().has_value() && gmem_queue_.empty() &&
        !gmem_bus_.has_value() && pending_write_acks_.empty() && lsu_commands_.cur().empty() &&
        read_records_.cur().empty() && ingress_.cur().beats.empty() && egress_fifo_.cur().empty() &&
        !tma_sector_.cur().request.has_value() && !tma_store_sector_.cur().busy &&
        tma.commands.empty() && tma.returns.empty() && !tma.event_command.has_value() &&
        tma.credits_used == 0 && writebacks_.cur().empty() && arithmetic_reads_.empty() &&
        mma.queue.empty() && mma.results.empty() && !mma.fetch.has_value() &&
        std::all_of(acc.begin(), acc.end(), [](const auto& slots) {
            return !slots[0].has_value() && !slots[1].has_value();
        });
}

uint32_t Machine::read_shmem_u32(uint32_t byte_address) const {
    if (byte_address % 4 != 0 || static_cast<uint64_t>(byte_address) + 4 > shmem_.size()) {
        throw SimError("invalid host shmem u32 read");
    }
    uint32_t value = 0;
    std::memcpy(&value, shmem_.data() + byte_address, sizeof(value));
    return value;
}

uint32_t Machine::read_gmem_u32(uint64_t byte_address) const {
    if (byte_address % 4 != 0 || byte_address + 4 < byte_address || byte_address + 4 > gmem_.size()) {
        throw SimError("invalid host gmem u32 read");
    }
    uint32_t value = 0;
    std::memcpy(&value, gmem_.data() + static_cast<std::size_t>(byte_address), sizeof(value));
    return value;
}

MbarState Machine::read_mbar_state(uint32_t byte_address) const {
    return load_mbar(byte_address);
}

void Machine::complete_mbar_tx(uint32_t byte_address, uint32_t bytes, uint8_t parity) {
    if (byte_address % 8 != 0 || static_cast<uint64_t>(byte_address) + 8 > shmem_.size()) {
        throw SimError("invalid engine mbar address");
    }
    if (parity > 1) throw SimError("invalid engine mbar parity");
    // A DMA complete_tx: it belongs in the TMA's event register.
    if (tma_mbar_event_.cur().has_value()) {
        throw SimError("TMA mbar event register is occupied; an engine holds one event at a time");
    }
    MbarRmwRequest request;
    request.operation = MbarRmwOperation::complete_tx;
    request.byte_address = byte_address;
    request.value = bytes;
    request.parity = parity;
    tma_mbar_event_.poke(std::move(request)); // between ticks: held from the next cycle on
}

uint32_t Machine::read_reg(
    uint32_t warp, uint32_t lane, uint32_t reg) const {
    if (warp >= warps_.size() || lane >= config_.lanes || reg >= kernel_.register_count) {
        throw SimError("invalid host RF read");
    }
    return rf(warps_[warp], lane, reg);
}

// ===========================================================================
// The RF ports: a reservation calendar, and the write port's execution
// ===========================================================================

// The RF ports are a reservation schedule. A booking made in an earlier cycle
// keeps its promise; executing a booked write is not arbitration. Only
// requests made in the SAME cycle for the SAME bank and target cycle contend,
// and booking order decides them. That order is the declared priority, the
// order in which tick() makes the decisions that book:
//   1. the LSU assembly drain     write at t            decide_lsu_rf_drain
//   2. acc transfers (tmem bank   acc.ld write at t+1,  step_acc_transfers
//      order, load then store)    acc.st read at t
//   3. issue (one instruction,    reads at t.., write   decide_issue
//      all-or-nothing)            at its result cycle
//   4. the LSU store read         read at t             decide_lsu_rf_read
// (The contests that can actually occur: acc.ld vs a latency-1 result at
// t+1; acc.st vs issue vs the LSU store read at t.) note_rf_requester()
// enforces that bookings happen in this order.
void Machine::note_rf_requester(RfRequester requester) {
    if (requester < rf_requester_) {
        throw SimError("internal RF arbiter: a booking was made out of the declared priority order");
    }
    rf_requester_ = requester;
}

// All-or-none admission test: does every read in the plan still fit its
// bank's ports in its cycle? Counting runs on a scratch copy of the ring so
// a rejected attempt books nothing. The count must be cumulative because one
// plan can put several reads in the same (cycle, bank) slot when a bank has
// more than one read port.
bool Machine::can_reserve_reads(const std::vector<RfRead>& reads) const {
    auto proposed = read_reservations_;
    for (const auto& read : reads) {
        auto& count = proposed[(cycle_ + read.offset) % proposed.size()][read.reg % config_.rf_banks];
        if (++count > config_.rf_read_ports_per_bank) return false;
    }
    return true;
}

void Machine::reserve_reads(const std::vector<RfRead>& reads, RfRequester requester) {
    note_rf_requester(requester);
    for (const auto& read : reads) {
        ++read_reservations_[(cycle_ + read.offset) % read_reservations_.size()][read.reg % config_.rf_banks];
    }
}

bool Machine::reserve_write(uint8_t reg, uint64_t at_cycle, RfRequester requester) {
    note_rf_requester(requester);
    auto& slots = write_reservations_[at_cycle];
    if (slots.empty()) slots.resize(config_.rf_banks, 0);
    const auto bank = reg % config_.rf_banks;
    if (slots[bank] == config_.rf_write_ports_per_bank) return false;
    ++slots[bank];
    return true;
}

// LSU: assembly register -> RF (rfc09 3a). The oldest load whose assembly
// register is full in Q (its sixteenth word landed in an earlier cycle)
// writes it to the RF in one full-width beat this cycle, if its bank's write
// port is free at this cycle. A write booked by an earlier issue keeps its
// port and the LSU waits.
Machine::LsuRfDrain Machine::decide_lsu_rf_drain() {
    LsuRfDrain drain;
    for (const auto& command : lsu_commands_.cur()) {
        if (is_store(command.kind)) continue;
        const auto& assembly = staging_.at(command.assembly).cur();
        if (assembly.valid_count != config_.lanes) continue;
        const bool vector = command.kind == LsuKind::vector_load;
        const auto reg = static_cast<uint8_t>(command.reg + (vector ? command.chunks_done : 0));
        if (!reserve_write(reg, cycle_, RfRequester::lsu_drain)) {
            ++trace_->lsu_counters().lsu_rf_write_wait_cycles;
            ++lsu_phases_[command.id].rf_wait;
            return drain;
        }
        drain.command = command.id;
        drain.assembly = command.assembly;
        drain.reg = reg;
        drain.last = !vector || command.chunks_done + 1 == 8;
        return drain;
    }
    return drain;
}

// The RF write port at this cycle, write-first (bypass (a)): every booked
// writeback due now lands, then the LSU's assembly drain, before anything
// reads the RF or the predicates. A write that ends an async operation
// clears its depbar (seen by this cycle's issue, bypass (b)) and may end its
// LSU command, whose slot and assembly register issue may reallocate now (d).
void Machine::execute_rf_writes(Wires& w) {
    const auto execute = [&](const Writeback& writeback) {
        auto& warp = warps_.at(writeback.warp);
        if (writeback.dst.has_value()) {
            if (writeback.values.size() != config_.lanes) {
                throw SimError("internal writeback lane count mismatch");
            }
            for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
                rf(warp, lane, *writeback.dst) = writeback.values[lane];
            }
        }
        if (writeback.predicate_dst.has_value()) {
            if (writeback.values.size() != config_.lanes) {
                throw SimError("internal predicate writeback lane count mismatch");
            }
            for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
                warp.predicates[*writeback.predicate_dst][lane] = writeback.values[lane] != 0;
            }
        }
        if (writeback.clear_depbar.has_value()) {
            record_depbar_completion(w, warp.id, *writeback.clear_depbar,
                "depbar underflow on sb" + std::to_string(*writeback.clear_depbar));
        }
        if (writeback.release_lsu_command.has_value()) {
            const auto id = *writeback.release_lsu_command;
            const auto* command = find_lsu_command(id);
            if (command == nullptr) throw SimError("internal release of an unknown LSU command");
            trace_->lsu_command_end(cycle_, id, lsu_phase_detail(*command));
            lsu_phases_.erase(id);
            w.released_lsu_commands.push_back(id);
        }
    };

    std::optional<Writeback> drain;
    if (w.lsu_drain.command.has_value()) {
        const auto& command = *find_lsu_command(*w.lsu_drain.command);
        const auto& assembly = staging_.at(w.lsu_drain.assembly).cur();
        sanitizer_.check_assembly_lanes_valid(
            cycle_, w.lsu_drain.assembly, assembly.valid, 0, config_.lanes, "RF drain");
        Writeback writeback;
        writeback.ready_cycle = cycle_;
        writeback.warp = command.warp;
        writeback.dst = w.lsu_drain.reg;
        writeback.values = assembly.words;
        writeback.reason = opcode_name(command.instruction.opcode);
        if (w.lsu_drain.last) {
            // rD..rD+7 count as in flight until the last write.
            const uint32_t width = command.kind == LsuKind::vector_load ? 8 : 1;
            for (uint32_t j = 0; j < width; ++j) sanitizer_.record_reg_ready(command.warp, command.reg + j, cycle_);
            writeback.clear_depbar = command.depbar;
            writeback.release_lsu_command = command.id;
        }
        probe(LsuEdge::rf_write_beat, command.id, w.lsu_drain.reg, config_.lanes * 4);
        drain = std::move(writeback);
    }
    for (const auto& writeback : writebacks_.cur()) {
        if (writeback.ready_cycle <= cycle_) execute(writeback);
    }
    if (drain.has_value()) execute(*drain); // booked this cycle, so it lands last
    sanitizer_.record_writebacks_committed(cycle_); // acc.ld final RF writes end read-ownership
    write_reservations_.erase(cycle_);
}

// ---- depbar: completions are recorded as they happen; issue sees
// cur - completions (bypass (b)); drive_depbar applies them and the
// admission. ----

void Machine::record_depbar_completion(
    Wires& w, uint32_t warp, uint8_t bar, const std::string& underflow) {
    const auto before = depbar_after_completions(w, warp, bar);
    if (before == 0) throw SimError(underflow);
    w.depbar_completions.push_back({warp, bar});
    trace_->depbar(cycle_, warp, bar, before - 1);
}

uint32_t Machine::depbar_after_completions(const Wires& w, uint32_t warp, uint8_t bar) const {
    uint32_t value = depbar_.cur().at(warp).at(bar);
    for (const auto& completion : w.depbar_completions) {
        if (completion.warp == warp && completion.bar == bar) --value;
    }
    return value;
}

void Machine::admit_depbar(Wires& w, uint32_t warp, uint8_t bar) {
    w.issue.depbar_admission = DepbarEvent{warp, bar};
    trace_->depbar(cycle_, warp, bar, depbar_after_completions(w, warp, bar) + 1);
}

// ===========================================================================
// shmem port A: one user per cycle
// ===========================================================================

bool Machine::port_a_idle() const {
    return !tma_port_a_.cur().has_value() && !port_a_.cur().has_value() &&
        !mbar_read_.cur().has_value() && !rmw_.cur().active.has_value() &&
        !tma_mbar_event_.cur().has_value() && !mma_mbar_event_.cur().has_value();
}

// The RMW engine's choice between the two engines' held events: one holder
// is served; when both hold one, the engine not served last.
Machine::MbarEngine Machine::engine_to_claim(bool tma_held, bool mma_held) const {
    if (!mma_held) return MbarEngine::tma;
    if (!tma_held) return MbarEngine::mma;
    return rmw_.cur().last_served == MbarEngine::tma ? MbarEngine::mma : MbarEngine::tma;
}

// Port A's arbiter. A started access (a TMA transaction, a SIMT access, the
// RMW engine) is never preempted. On a free port, held engine events go
// first (they claim the RMW engine), then the TMA's new transaction. New SIMT
// work is not a requester here: issue accepts it for the NEXT cycle, only if
// this decision leaves port A free then and the TMA will not want it
// (port_a_busy_next_cycle, tma_wants_port_a_next_cycle).
Machine::PortAGrant Machine::select_port_a(const TmaRequests& tma) const {
    using User = PortAGrant::User;
    if (tma_port_a_.cur().has_value()) return {User::tma_transaction};
    if (port_a_.cur().has_value()) return {User::simt_access};
    if (mbar_read_.cur().has_value()) return {User::mbar_read};
    if (rmw_.cur().active.has_value()) return {User::rmw};
    const bool tma_held = tma_mbar_event_.cur().has_value();
    const bool mma_held = mma_mbar_event_.cur().has_value();
    if (tma_held || mma_held) return {User::engine_event, engine_to_claim(tma_held, mma_held)};
    if (tma.port_a.has_value()) return {User::tma_start};
    return {};
}

// Port A's data path this cycle: the granted user's round, read stage or
// write stage. Reads see shmem as it was at the start of the cycle; writes
// are collected and land at the clock edge.
Machine::PortAService Machine::serve_port_a(const PortAGrant& grant, const TmaRequests& tma, Wires& w) {
    using User = PortAGrant::User;
    PortAService service;
    service.grant = grant;
    std::optional<MbarEngine> serving; // (T) counter attribution only
    switch (grant.user) {
    case User::idle:
        break;
    case User::tma_start: {
        const auto& request = *tma.port_a;
        static const char* const labels[] = {
            "tma descriptor read", "tma mbar phase read", "tma load write", "tma store read"};
        trace_->tma_port_begin(cycle_, request.command, labels[static_cast<int>(request.kind)],
            static_cast<uint32_t>(request.rounds.size()));
        if (request.kind == TmaPortKind::load_write || request.kind == TmaPortKind::store_read) {
            trace_->tma_data_port(static_cast<uint32_t>(request.rounds.size()));
        }
        serve_tma_round(request, service);
        break;
    }
    case User::tma_transaction:
        serve_tma_round(*tma_port_a_.cur(), service);
        break;
    case User::simt_access:
        serve_simt_round(*port_a_.cur(), service, w);
        break;
    case User::mbar_read:
        serve_mbar_read(service, w);
        break;
    case User::rmw:
        serving = rmw_.cur().active->engine;
        serve_rmw_stage(*rmw_.cur().active, service, w);
        break;
    case User::engine_event: {
        // The held event claims the free RMW engine; its read stage is now.
        const bool tma_event = grant.engine == MbarEngine::tma;
        ActiveMbarRmw active;
        active.request = tma_event ? *tma_mbar_event_.cur() : *mma_mbar_event_.cur();
        active.engine = grant.engine;
        (tma_event ? service.tma_event_claimed : service.mma_event_claimed) = true;
        // Sanitizer: an engine's RMW is admitted here (SIMT ones at issue).
        // Engine events never carry a requesting warp.
        sanitizer_.check_engine_mbar_rmw(cycle_, active.request.byte_address,
            active.request.operation == MbarRmwOperation::complete_tx);
        serving = grant.engine;
        serve_rmw_stage(std::move(active), service, w);
        break;
    }
    }
    // An event held through a cycle the RMW engine spends on the OTHER
    // engine's event (the claim cycle or its write stage). An event presented
    // this cycle is held from the next one.
    if (serving.has_value() &&
        (*serving == MbarEngine::tma ? mma_mbar_event_ : tma_mbar_event_).cur().has_value()) {
        trace_->mbar_engine_wait_on_other();
    }
    return service;
}

// One bank round of a TMA port-A transaction. A load placement reads the TMA
// sector register's Q (filled in earlier cycles) or writes zeros (OOB fill);
// a store read captures shmem bytes for the TMA store sector register; a
// descriptor or mbar-phase read captures them into the transaction's latch.
void Machine::serve_tma_round(TmaPortRequest request, PortAService& service) {
    const bool write = request.kind == TmaPortKind::load_write;
    const bool store_read = request.kind == TmaPortKind::store_read;
    if (request.rounds.empty() || request.next_round >= request.rounds.size()) {
        throw SimError("internal zero-round tma port-A request");
    }
    const auto& sector = tma_sector_.cur();
    if (write && request.from_sector_register && !sector.complete) {
        throw SimError("internal: TMA placement started before its sector register was complete");
    }
    uint32_t bytes = 0;
    for (const auto& piece : request.rounds.at(request.next_round)) {
        auto& fragment = request.fragments.at(piece.fragment);
        const uint32_t address = fragment.shmem_address + piece.offset;
        if (static_cast<uint64_t>(address) + piece.size > shmem_.size()) {
            throw SimError("tma port service is outside shmem");
        }
        bytes += piece.size;
        if (write) {
            const uint8_t* source = fragment.data.data() + piece.offset; // OOB zero fill
            if (request.from_sector_register) {
                source = sector.data.data() + (fragment.gmem_address - sector.sector_address) + piece.offset;
            }
            service.writes.push_back({address, std::vector<uint8_t>(source, source + piece.size)});
            sanitizer_.record_port_a_write(address, piece.size);
        } else if (store_read) {
            const auto offset = static_cast<uint32_t>(fragment.gmem_address % config_.gmem_sector_bytes);
            for (uint32_t byte = 0; byte < piece.size; ++byte) {
                service.store_bytes.emplace_back(offset + piece.offset + byte, shmem_[address + byte]);
            }
        } else {
            std::memcpy(fragment.data.data() + piece.offset, shmem_.data() + address, piece.size);
        }
    }
    if (write) probe(LsuEdge::tma_place_round, request.sector_request, request.next_round, bytes);
    if (store_read) probe(LsuEdge::tma_store_round, request.command, request.next_round, bytes);
    ++request.next_round;
    trace_->shmem_round();
    if (request.next_round != request.rounds.size()) {
        service.tma_after = std::move(request);
        return;
    }
    trace_->shmem_end(cycle_ + 1);
    service.tma_completed = std::move(request);
}

// One bank round of a SIMT lds/sts. The last round schedules the completion:
// lds values and the depbar clear land at the next cycle's RF write port.
void Machine::serve_simt_round(ShmemRequest request, PortAService& service, Wires& w) {
    const auto& round = request.rounds.at(request.next_round);
    for (const auto& access : round) {
        if (request.load) {
            request.load_values[access.lane] = read_shmem_u32(static_cast<uint32_t>(access.byte_address));
        } else {
            std::vector<uint8_t> bytes(4);
            std::memcpy(bytes.data(), &access.value, sizeof(access.value));
            service.writes.push_back({static_cast<uint32_t>(access.byte_address), std::move(bytes)});
            sanitizer_.record_port_a_write(access.byte_address, sizeof(access.value));
        }
    }
    trace_->shmem_round();
    ++request.next_round;
    if (request.next_round != request.rounds.size()) {
        service.simt_after = std::move(request);
        return;
    }
    Writeback completion;
    completion.ready_cycle = cycle_ + 1;
    completion.warp = request.warp;
    completion.clear_depbar = request.depbar;
    if (request.load) {
        completion.dst = request.dst;
        completion.values = std::move(request.load_values);
        completion.reason = "lds";
    } else {
        completion.reason = "sts";
    }
    w.new_writebacks.push_back(std::move(completion));
    trace_->shmem_end(cycle_ + 1);
}

// A SIMT mbar.test_wait / mbar.try_wait read, the cycle after issue. Its
// predicate (and a try-wait's completion) reach this cycle's issue (a), (b).
void Machine::serve_mbar_read(PortAService& service, Wires& w) {
    const auto& request = *mbar_read_.cur();
    auto& warp = warps_.at(request.warp);
    const auto state = load_mbar(request.byte_address);
    if (!state.valid || state.expected == 0 || state.remaining > state.expected) {
        throw SimError(
            "PANIC cycle=" + std::to_string(cycle_) + " warp=" + std::to_string(warp.id) +
            " pc=" + std::to_string(request.pc) +
            " reason=wait/test on an uninitialized or corrupt mbar inst='" +
            format_instruction(request.instruction) + "'");
    }
    const bool complete = state.phase != (request.parity != 0);
    if (request.test) {
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            warp.predicates[request.predicate_dst][lane] = complete;
        }
        sanitizer_.record_pred_ready(warp.id, request.predicate_dst, cycle_);
        return;
    }
    const auto& slots = mbar_wait_slots_.cur();
    if (!request.wait_slot.has_value() || *request.wait_slot >= slots.size() ||
        !slots[*request.wait_slot].has_value() || slots[*request.wait_slot]->warp != warp.id) {
        throw SimError("internal mbar try-wait slot mismatch");
    }
    const auto& slot = *slots[*request.wait_slot];
    if (slot.byte_address != request.byte_address ||
        slot.parity != request.parity || slot.predicate_dst != request.predicate_dst) {
        throw SimError("internal mbar try-wait record mismatch");
    }
    trace_->mbar_probe(cycle_, warp.id, request.byte_address, request.parity, complete);
    if (complete) {
        complete_mbar_try_wait(w, *request.wait_slot, true);
        return;
    }
    const auto deadline = cycle_ > std::numeric_limits<uint64_t>::max() -
            config_.mbar_try_wait_timeout_cycles
        ? std::numeric_limits<uint64_t>::max()
        : cycle_ + config_.mbar_try_wait_timeout_cycles;
    service.park = std::make_pair(*request.wait_slot, deadline);
    trace_->mbar_sleep(cycle_, warp.id, request.byte_address, request.parity);
}

std::string Machine::rmw_label(const ActiveMbarRmw& active) {
    std::string operation;
    switch (active.request.operation) {
    case MbarRmwOperation::init: operation = "init"; break;
    case MbarRmwOperation::arrive: operation = "arrive"; break;
    case MbarRmwOperation::arrive_expect_tx: operation = "arrive.expect_tx"; break;
    case MbarRmwOperation::complete_tx: operation = "complete_tx"; break;
    }
    if (active.engine.has_value()) {
        operation += *active.engine == MbarEngine::tma ? " (tma)" : " (mma.commit)";
    }
    return operation;
}

// The RMW engine's two stages: read the mbar (first cycle), then write it
// back updated (second cycle). A phase flip wakes the waiters in the write
// cycle; they may issue in that cycle (b).
void Machine::serve_rmw_stage(ActiveMbarRmw active, PortAService& service, Wires& w) {
    if (!active.write_stage) {
        active.state = load_mbar(active.request.byte_address);
        active.write_stage = true;
        trace_->mbar_rmw_begin(cycle_, active.request.byte_address, rmw_label(active));
        service.rmw_after = std::move(active); // read cycle
        return;
    }

    auto& state = active.state;
    const auto& request = active.request;

    const auto validate_live = [&] {
        if (!state.valid) mbar_rmw_panic(request, "operation on an uninitialized mbar");
        if (state.expected == 0 || state.remaining > state.expected) {
            mbar_rmw_panic(request, "corrupt mbar state");
        }
    };

    bool phase_changed = false;
    switch (request.operation) {
    case MbarRmwOperation::init:
        for (const auto& wait_slot : mbar_wait_slots_.cur()) {
            if (wait_slot.has_value() &&
                wait_slot->byte_address == request.byte_address) {
                mbar_rmw_panic(request, "reinitializing an mbar with a live waiter");
            }
        }
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
        if (request.value > std::numeric_limits<uint32_t>::max() - state.tx_bytes) {
            mbar_rmw_panic(request, "transaction byte count overflow");
        }
        --state.remaining;
        state.tx_bytes += request.value;
        break;
    case MbarRmwOperation::complete_tx:
        validate_live();
        if (state.phase != (request.parity != 0)) {
            mbar_rmw_panic(request, "completing transaction bytes against the wrong phase");
        }
        if (request.value > state.tx_bytes) {
            mbar_rmw_panic(request, "transaction byte count underflow");
        }
        state.tx_bytes -= request.value;
        break;
    }

    if (request.operation != MbarRmwOperation::init &&
        state.remaining == 0 && state.tx_bytes == 0) {
        state.phase = !state.phase;
        state.remaining = state.expected;
        phase_changed = true;
    }
    const auto word = encode_mbar_state(state);
    std::vector<uint8_t> bytes(8);
    std::memcpy(bytes.data(), &word, sizeof(word));
    service.writes.push_back({request.byte_address, std::move(bytes)});
    // C5: the RMW write stage is a port-A write like any other; a same-cycle
    // same-word MMA port-B round must see it.
    sanitizer_.record_port_a_write(request.byte_address, 8);
    if (phase_changed) {
        wake_mbar_waiters(w, request.byte_address, static_cast<uint8_t>(state.phase));
    }
    trace_->mbar_rmw_end(cycle_ + 1, phase_changed);
}

// ---- mbar wait slots (the scheduler-visible try-wait state) ----

bool Machine::wait_slot_completed(const Wires& w, uint32_t slot) const {
    return std::find(w.completed_waits.begin(), w.completed_waits.end(), slot) != w.completed_waits.end();
}

// The slot a warp is parked in, as issue sees it after this cycle's
// completions (b).
std::optional<uint32_t> Machine::waiting_slot_of(const Wires& w, uint32_t warp) const {
    const auto& slots = mbar_wait_slots_.cur();
    for (uint32_t index = 0; index < slots.size(); ++index) {
        if (slots[index].has_value() && slots[index]->warp == warp && !wait_slot_completed(w, index)) {
            return index;
        }
    }
    return std::nullopt;
}

std::optional<uint32_t> Machine::free_mbar_wait_slot(const Wires& w) const {
    const auto& slots = mbar_wait_slots_.cur();
    for (uint32_t index = 0; index < slots.size(); ++index) {
        if (!slots[index].has_value() || wait_slot_completed(w, index)) return index;
    }
    return std::nullopt;
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

// A try-wait ends: its predicate is written now (write-first, (a)) and its
// slot frees; drive_wait_slots clears it.
void Machine::complete_mbar_try_wait(Wires& w, uint32_t index, bool result) {
    const auto& slots = mbar_wait_slots_.cur();
    if (index >= slots.size() || !slots[index].has_value() || wait_slot_completed(w, index)) {
        throw SimError("internal completion of an empty mbar try-wait slot");
    }
    const auto& slot = *slots[index];
    auto& warp = warps_.at(slot.warp);
    for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
        warp.predicates[slot.predicate_dst][lane] = result;
    }
    sanitizer_.record_pred_ready(warp.id, slot.predicate_dst, cycle_);
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

// Port A at the next cycle, as this cycle's decisions leave it (look-ahead
// (c)): a continuing TMA transaction, SIMT access or RMW, or an engine event
// held then (still unclaimed, or presented now).
bool Machine::port_a_busy_next_cycle(const Wires& w) const {
    const auto& service = w.port_a;
    if (service.tma_after.has_value() || service.simt_after.has_value() || service.rmw_after.has_value()) {
        return true;
    }
    const bool tma_event = (tma_mbar_event_.cur().has_value() && !service.tma_event_claimed) ||
        w.tma.present.has_value();
    const bool mma_event = (mma_mbar_event_.cur().has_value() && !service.mma_event_claimed) ||
        w.mma_present.has_value();
    return tma_event || mma_event;
}

// ===========================================================================
// MMA engine (port B) and accumulator transfers
// ===========================================================================

namespace {

uint32_t mma_a_address(const Config& config, uint32_t base, uint32_t m, uint32_t k) {
    if (!config.mma_xor_layout) return base + 2 * (m * config.mma_k + k);
    return xor64_operand_address(base, m, k, 0);
}

uint32_t mma_b_address(const Config& config, uint32_t base, uint32_t k, uint32_t n) {
    if (!config.mma_xor_layout) return base + 2 * (k * config.mma_n + n);
    return xor64_operand_address(base, n, k, 2);
}

} // namespace

bool Machine::acc_slot_available(const Wires& w, uint8_t bank, bool load) const {
    // As this cycle leaves the MMA and the transfer slots (look-ahead (c)).
    if (mma_bank_busy(w.mma_next, bank)) return false;
    const auto& slots = w.acc_next.at(bank);
    if (slots[load ? 0 : 1].has_value()) return false;
    return config_.tmem_simt_read_write_duplex ||
        (!slots[0].has_value() && !slots[1].has_value());
}

bool Machine::mma_bank_busy(const MmaPipeline& mma, uint8_t bank) const {
    for (const auto& command : mma.queue) {
        if (!command.commit && (command.next_step != 0 || command.operands_consumed) &&
            kernel_.accumulators.at(command.accumulator).bank == bank) return true;
    }
    for (const auto& result : mma.results)
        if (kernel_.accumulators.at(result.accumulator).bank == bank) return true;
    return false;
}

// The MMA engine's cycle, computed on a copy of its queue and pipeline:
// issue looks ahead at the queue entries and tmem banks it frees (c), so the
// next state is a wire; drive_mma commits it with issue's new command.
// tmem is the MMA's memory here (its banks are exclusive with acc transfers):
// results retire into it before this cycle's accumulator read (f).
void Machine::step_mma(Wires& w) {
    auto mma = mma_.cur();
    advance_mma(mma, w);
    w.mma_next = std::move(mma);
}

// One cycle of the MMA engine, applied to `mma` (a copy of its Q): results
// due now write tmem, retired commands leave the queue head, then one port-B
// round of the fetch in service -- or a new fetch, or the head commit's
// presentation.
void Machine::advance_mma(MmaPipeline& mma, Wires& w) {
    for (auto it = mma.results.begin(); it != mma.results.end();) {
        if (it->ready_cycle > cycle_) { ++it; continue; }
        const auto& acc = kernel_.accumulators.at(it->accumulator);
        for (uint32_t mi = 0; mi < config_.mma_core_m; ++mi)
            for (uint32_t ni = 0; ni < config_.mma_core_n; ++ni)
                tmem_[acc.bank][tmem_word_index(acc, it->m0 + mi, it->n0 + ni)] =
                    it->values[mi * config_.mma_core_n + ni];
        trace_->mma_result(cycle_, it->command, it->m0 * config_.mma_n + it->n0);
        const auto id = it->command;
        auto command = std::find_if(mma.queue.begin(), mma.queue.end(),
            [id](const auto& candidate) { return candidate.id == id; });
        if (command == mma.queue.end() || command->outstanding_results == 0)
            throw SimError("internal MMA result lost its command");
        --command->outstanding_results;
        it = mma.results.erase(it);
    }
    while (!mma.queue.empty() && !mma.queue.front().commit &&
           mma.queue.front().operands_consumed && mma.queue.front().outstanding_results == 0) {
        sanitizer_.record_mma_retired(mma.queue.front().id); // last result retired
        mma.queue.pop_front();
    }
    // Port service: one bank round per cycle for the in-flight fetch. A new
    // fetch may start only in a cycle with no fetch in service, so the next
    // step begins the cycle after the port frees; one-round layouts sustain
    // one core step per cycle.
    if (mma.fetch.has_value()) {
        serve_mma_fetch_round(mma);
        if (mma.fetch->next_round == mma.fetch->round_words.size()) complete_mma_fetch(mma, w);
        return;
    }
    if (mma.queue.empty()) return;
    auto command_it = std::find_if(mma.queue.begin(), mma.queue.end(),
        [](const auto& command) { return !command.operands_consumed; });
    if (command_it == mma.queue.end()) return;
    auto& command = *command_it;
    if (command.commit) {
        // A commit at the queue head presents into the MMA's own event
        // register: free if empty in Q, or claimed by the RMW engine now (d).
        const bool register_free = !mma_mbar_event_.cur().has_value() || w.port_a.mma_event_claimed;
        if (command_it != mma.queue.begin() || !register_free) return;
        MbarRmwRequest request;
        request.operation = MbarRmwOperation::arrive;
        request.byte_address = command.mbar_address;
        w.mma_present = std::move(request);
        mma.queue.pop_front();
        return;
    }
    const auto& acc = kernel_.accumulators.at(command.accumulator);
    const auto& transfers = acc_transfers_.cur().at(acc.bank);
    if (transfers[0].has_value() || transfers[1].has_value()) return;
    const uint32_t blocks_m = config_.mma_m / config_.mma_core_m;
    const uint32_t blocks_n = config_.mma_n / config_.mma_core_n;
    const uint32_t blocks_per_k = blocks_m * blocks_n;
    const uint32_t k_block = command.next_step / blocks_per_k;
    const uint32_t block = command.next_step % blocks_per_k;
    const uint32_t n_block = block / blocks_m;
    const uint32_t m_block = block % blocks_m;
    const uint32_t m0 = m_block * config_.mma_core_m;
    const uint32_t n0 = n_block * config_.mma_core_n;
    const uint32_t k0 = k_block * config_.mma_core_k;
    for (const auto& pending : mma.results) {
        if (pending.accumulator == command.accumulator && pending.m0 == m0 && pending.n0 == n0)
        {
            trace_->mma_recurrence_stall();
            return; // explicit recurrence bubble for a profile whose sweep is too short
        }
    }
    std::vector<std::set<uint32_t>> words_by_bank(config_.shmem_banks);
    const auto insert_word = [&](uint32_t address) {
        const auto word = address / 4;
        words_by_bank[word % config_.shmem_banks].insert(word);
    };
    for (uint32_t mi = 0; mi < config_.mma_core_m; ++mi)
        for (uint32_t kk = 0; kk < config_.mma_core_k; ++kk)
            insert_word(mma_a_address(config_, command.a_base, m0 + mi, k0 + kk));
    for (uint32_t kk = 0; kk < config_.mma_core_k; ++kk)
        for (uint32_t ni = 0; ni < config_.mma_core_n; ++ni)
            insert_word(mma_b_address(config_, command.b_base, k0 + kk, n0 + ni));
    uint32_t rounds = 1;
    std::map<uint32_t, uint32_t> word_round;
    for (const auto& words : words_by_bank) {
        rounds = std::max<uint32_t>(rounds, static_cast<uint32_t>(words.size()));
        uint32_t index = 0;
        for (const auto word : words) word_round[word] = index++;
    }
    MmaFetch fetch;
    fetch.command = command.id;
    fetch.accumulator = command.accumulator;
    fetch.init_step = command.init && k0 == 0;
    fetch.m0 = m0;
    fetch.n0 = n0;
    fetch.k0 = k0;
    fetch.a_base = command.a_base;
    fetch.b_base = command.b_base;
    fetch.round_words.resize(rounds);
    fetch.round_elements.resize(rounds);
    for (const auto& [word, round] : word_round) fetch.round_words[round].push_back(word);
    const auto place_element = [&](uint32_t address) {
        fetch.round_elements[word_round.at(address / 4)].push_back(address);
    };
    for (uint32_t mi = 0; mi < config_.mma_core_m; ++mi)
        for (uint32_t kk = 0; kk < config_.mma_core_k; ++kk)
            place_element(mma_a_address(config_, command.a_base, m0 + mi, k0 + kk));
    for (uint32_t kk = 0; kk < config_.mma_core_k; ++kk)
        for (uint32_t ni = 0; ni < config_.mma_core_n; ++ni)
            place_element(mma_b_address(config_, command.b_base, k0 + kk, n0 + ni));
    const uint32_t total_steps = blocks_per_k * (config_.mma_k / config_.mma_core_k);
    fetch.last_step = command.next_step + 1 == total_steps;
    ++command.next_step;
    sanitizer_.record_mma_fetch_start(command.id);
    trace_->mma_input(cycle_, command.id, rounds, command.next_step - 1);
    mma.fetch = std::move(fetch);
    serve_mma_fetch_round(mma);
    if (mma.fetch->next_round == mma.fetch->round_words.size()) complete_mma_fetch(mma, w);
}

// One port-B bank round: captures the operand halves this round reads, from
// shmem as it was at the start of the cycle (a same-word port-A write in this
// cycle is case 08's collision PANIC: C5).
void Machine::serve_mma_fetch_round(MmaPipeline& mma) {
    auto& fetch = *mma.fetch;
    sanitizer_.check_port_b_round(fetch.round_words.at(fetch.next_round));
    for (const auto address : fetch.round_elements.at(fetch.next_round)) {
        if (static_cast<uint64_t>(address) + 2 > shmem_.size())
            throw SimError("MMA operand footprint is outside shmem");
        uint16_t bits = 0;
        std::memcpy(&bits, shmem_.data() + address, 2);
        fetch.captured[address] = bits;
    }
    ++fetch.next_round;
}

// Fetch completion is core entry: the accumulator input is read, the step's
// result is scheduled one pipeline latency later, and the last step's
// operand-consumed depbar clears here, at the final physical read.
void Machine::complete_mma_fetch(MmaPipeline& mma, Wires& w) {
    const auto fetch = std::move(*mma.fetch);
    mma.fetch.reset();
    const auto id = fetch.command;
    auto command_it = std::find_if(mma.queue.begin(), mma.queue.end(),
        [id](const auto& candidate) { return candidate.id == id; });
    if (command_it == mma.queue.end()) throw SimError("internal MMA fetch lost its command");
    auto& command = *command_it;
    const auto& acc = kernel_.accumulators.at(fetch.accumulator);
    const auto captured = [&](uint32_t address) {
        const auto value = fetch.captured.find(address);
        if (value == fetch.captured.end())
            throw SimError("internal MMA operand was not captured by its round");
        return value->second;
    };
    MmaResult result;
    result.ready_cycle = cycle_ + config_.mma_result_latency_cycles;
    result.command = command.id;
    result.accumulator = fetch.accumulator;
    result.m0 = fetch.m0;
    result.n0 = fetch.n0;
    result.values.resize(config_.mma_core_m * config_.mma_core_n);
    // Gather raw operands once per step, not via map lookups per product.
    // B is transposed here only in the simulator's local operand snapshot.
    std::vector<uint16_t> a(config_.mma_core_m * config_.mma_core_k);
    std::vector<uint16_t> b(config_.mma_core_n * config_.mma_core_k);
    for (uint32_t mi = 0; mi < config_.mma_core_m; ++mi)
        for (uint32_t kk = 0; kk < config_.mma_core_k; ++kk)
            a[mi * config_.mma_core_k + kk] = captured(
                mma_a_address(config_, fetch.a_base, fetch.m0 + mi, fetch.k0 + kk));
    for (uint32_t ni = 0; ni < config_.mma_core_n; ++ni)
        for (uint32_t kk = 0; kk < config_.mma_core_k; ++kk)
            b[ni * config_.mma_core_k + kk] = captured(
                mma_b_address(config_, fetch.b_base, fetch.k0 + kk, fetch.n0 + ni));
    for (uint32_t mi = 0; mi < config_.mma_core_m; ++mi) {
        for (uint32_t ni = 0; ni < config_.mma_core_n; ++ni) {
            const uint32_t c = fetch.init_step ? 0u : tmem_[acc.bank][
                tmem_word_index(acc, fetch.m0 + mi, fetch.n0 + ni)];
            result.values[mi * config_.mma_core_n + ni] = mma_dot_f16(
                std::span<const uint16_t>(a).subspan(mi * config_.mma_core_k, config_.mma_core_k),
                std::span<const uint16_t>(b).subspan(ni * config_.mma_core_k, config_.mma_core_k), c);
        }
    }
    mma.results.push_back(std::move(result));
    ++command.outstanding_results;
    if (fetch.last_step) {
        command.operands_consumed = true;
        sanitizer_.record_mma_operands_consumed(command.id); // final operand read
        record_depbar_completion(w, command.warp, command.depbar, "internal MMA depbar underflow");
    }
}

std::size_t Machine::tmem_word_index(
    const Kernel::Accumulator& acc, uint32_t row, uint32_t col) const {
    // Candidate rfc05 fine layout. The row-major 4x4 macro-tile stride is
    // dimension-aware; chan.txt's 16*(row/4)+(col/4) is only the N=64 case.
    const uint32_t fine_bank = 4 * (row % 4) + ((col % 4) ^ ((row / 4) % 4));
    const uint32_t macro_columns = (acc.cols + 3) / 4;
    const uint64_t bank_address = static_cast<uint64_t>(row / 4) * macro_columns + col / 4;
    const uint64_t offset = bank_address * config_.tmem_fine_banks + fine_bank;
    const uint64_t index = static_cast<uint64_t>(acc.base_word) + offset;
    if (index >= tmem_.at(acc.bank).size()) throw SimError("internal tmem layout overflow");
    return static_cast<std::size_t>(index);
}

// One beat per transfer slot per cycle. acc.ld books the RF write port for
// the next cycle and reads tmem now; acc.st books an RF read port now and
// writes tmem now. Slots run in tmem-bank order, load before store (their
// place in the RF booking order). The next slots are a wire: issue may
// reallocate a slot freed now (d).
void Machine::step_acc_transfers(Wires& w) {
    auto slots = acc_transfers_.cur();
    for (uint32_t bank = 0; bank < slots.size(); ++bank) {
        for (uint32_t direction = 0; direction < 2; ++direction) {
            auto& slot = slots[bank][direction];
            if (!slot.has_value()) continue;
            auto& transfer = *slot;
            auto& warp = warps_.at(transfer.warp);
            const auto& acc = kernel_.accumulators.at(transfer.accumulator);
            const auto reg = static_cast<uint8_t>(transfer.rf_base + transfer.next_column);
            if (transfer.load) {
                if (!reserve_write(reg, cycle_ + 1, RfRequester::acc_transfer)) continue;
                std::vector<uint32_t> values(config_.lanes);
                for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
                    values[lane] = tmem_[bank][tmem_word_index(
                        acc, transfer.row_base + lane, transfer.col_base + transfer.next_column)];
                }
                Writeback writeback;
                writeback.ready_cycle = cycle_ + 1;
                writeback.warp = transfer.warp;
                writeback.dst = reg;
                writeback.values = std::move(values);
                writeback.reason = "acc.ld";
                if (transfer.next_column + 1 == transfer.width) {
                    writeback.clear_depbar = transfer.depbar;
                    // tmem read-ownership ends when this final RF write commits
                    sanitizer_.record_acc_transfer_final_beat(
                        static_cast<uint8_t>(bank), true, writeback.ready_cycle);
                }
                w.new_writebacks.push_back(std::move(writeback));
                sanitizer_.record_reg_ready(warp.id, reg, cycle_ + 1);
            } else {
                const std::vector<RfRead> reads{{reg, 0}};
                if (!can_reserve_reads(reads)) continue;
                reserve_reads(reads, RfRequester::acc_transfer);
                trace_->rf_read(cycle_, warp.id, reg, reg % config_.rf_banks);
                for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
                    tmem_[bank][tmem_word_index(
                        acc, transfer.row_base + lane, transfer.col_base + transfer.next_column)] =
                        rf(warp, lane, reg);
                }
            }
            ++transfer.next_column;
            trace_->tmem_beat(cycle_, warp.id, bank, transfer.load, transfer.next_column - 1);
            if (transfer.next_column != transfer.width) continue;
            if (!transfer.load) {
                for (uint32_t j = 0; j < transfer.width; ++j) {
                    sanitizer_.record_async_reader_done(
                        warp.id, transfer.rf_base + j, AsyncReader::acc_st);
                }
                record_depbar_completion(w, warp.id, transfer.depbar, "internal acc.st depbar underflow");
                // final tmem write: write-ownership ends
                sanitizer_.record_acc_transfer_final_beat(static_cast<uint8_t>(bank), false, cycle_);
            }
            slot.reset();
        }
    }
    w.acc_next = std::move(slots);
}

// ===========================================================================
// TMA engine
// ===========================================================================

// What the TMA wants this cycle, from Q alone (no other unit's wants).
//   1. Present its completion event: into its own register, when that is
//      empty and the previous command has ended. The event is visible to the
//      RMW engine from the next cycle (like the MMA's).
//   2. A TMA store sector waiting, complete and visible, at the egress FIFO's
//      request pointer wants the request channel.
//   3. Port A: a new transaction only on an idle port, and not in the cycle
//      it presents an event (the event takes the next free grant). Then, in
//      order: place the front ticket; or serve the frontend command (its
//      descriptor read, its mbar-phase read, or the walker's next sector).
//      The walker runs only on those same cycles, even for loads.
Machine::TmaRequests Machine::tma_requests(const EgressWrite& egress) const {
    TmaRequests want;
    const auto& tma = tma_.cur();
    if (!tma_mbar_event_.cur().has_value() && !tma.event_command.has_value()) {
        want.present = next_data_complete_tma_command(tma);
        if (want.present.has_value()) {
            const auto& command = tma.commands.at(*want.present);
            if (command.logical_bytes > std::numeric_limits<uint32_t>::max()) {
                throw SimError("tma completion byte count exceeds the mbar encoding");
            }
            MbarRmwRequest event;
            event.operation = MbarRmwOperation::complete_tx;
            event.byte_address = command.descriptor.completion_mbar;
            event.value = static_cast<uint32_t>(command.logical_bytes);
            event.parity = command.mbar_phase;
            want.present_event = std::move(event);
        }
    }
    const auto* slot = egress_request_slot(std::nullopt);
    want.write_request = slot != nullptr && slot->client == GmemClient::tma && egress_slot_visible(*slot);

    want.port_a_idle = port_a_idle() && !want.present.has_value();
    if (!want.port_a_idle) return want;
    if (!tma.returns.empty()) {
        const auto& ticket = tma.returns.front();
        TmaPortRequest request;
        request.kind = TmaPortKind::load_write;
        request.command = ticket.command;
        request.sector_request = ticket.sector_request;
        request.from_sector_register = ticket.from_sector_register;
        request.fragments = ticket.fragments;
        request.rounds = tma_port_rounds(request.fragments);
        want.port_a = std::move(request);
        want.placement = true;
        return want;
    }

    want.frontend = tma.frontend.has_value() ? tma.frontend : next_tma_frontend_command(tma);
    if (!want.frontend.has_value()) return want;
    const auto& command = tma.commands.at(*want.frontend);
    if (command.state == TmaCommandState::descriptor || command.state == TmaCommandState::phase) {
        const bool descriptor = command.state == TmaCommandState::descriptor;
        TmaPortRequest request;
        request.kind = descriptor ? TmaPortKind::descriptor : TmaPortKind::phase;
        request.command = command.id;
        const uint32_t base = descriptor ? command.descriptor_address : command.descriptor.completion_mbar;
        const uint32_t size = descriptor ? static_cast<uint32_t>(tma_descriptor_bytes) : 8u;
        for (uint32_t offset = 0; offset < size; offset += 4) {
            GmemSectorRequest::TmaFragment fragment;
            fragment.shmem_address = base + offset;
            fragment.size = 4;
            request.fragments.push_back(fragment);
        }
        request.rounds = tma_port_rounds(request.fragments);
        want.port_a = std::move(request);
        return want;
    }
    if (command.state != TmaCommandState::walking) {
        want.frontend_reset = true;
        return want;
    }
    // A store sector's source read needs the TMA store register: free in Q,
    // or drained into the egress FIFO this cycle (d).
    const bool store_register_free = !tma_store_sector_.cur().busy || egress.tma_store_drained;
    want.walker = plan_tma_walker(command, tma.credits_used, store_register_free);
    if (want.walker->sector == TmaWalkerPlan::Sector::store) {
        TmaPortRequest request;
        request.kind = TmaPortKind::store_read;
        request.command = command.id;
        request.fragments = want.walker->fragments;
        request.rounds = tma_port_rounds(request.fragments);
        want.port_a = std::move(request);
    }
    return want;
}

// The walker's next sector, planned from Q. A sector is a run of elements in
// one gmem sector (contiguous) or, for a load's out-of-bounds elements, a
// zero fill of at most one sector of shmem. Store OOB elements are skipped
// (no transaction at all), even in a cycle where the sector is then blocked.
// Blocked: no credit; a store with the TMA store register busy. A load
// sector also needs the request channel, which select_request_channel decides.
Machine::TmaWalkerPlan Machine::plan_tma_walker(
    const TmaCommand& start, uint32_t credits_used, bool store_register_free) const {
    TmaWalkerPlan plan;
    plan.command_after = start;
    auto& command = plan.command_after;
    if (command.walker_done) return plan;
    const bool load = command.descriptor.direction == TmaDirection::load;

    while (!load && !command.walker_done && !tma_coordinate_in_bounds(command)) {
        (void)tma_advance_coordinate(command);
    }
    if (command.walker_done) return plan;
    if (credits_used == config_.tma_sector_credits) return plan;
    if (!load && !store_register_free) return plan;

    const bool synthetic_zero = load && !tma_coordinate_in_bounds(command);
    uint64_t sector_address = 0;
    uint64_t expected_next_gmem = 0;
    uint32_t bytes = 0;
    while (!command.walker_done) {
        const bool in_bounds = tma_coordinate_in_bounds(command);
        if (load && synthetic_zero != !in_bounds) break;
        if (!load && !in_bounds) break;

        const auto shmem_address = tma_shmem_address(command);
        uint64_t gmem_address = 0;
        if (in_bounds) {
            gmem_address = tma_gmem_address(command);
            const auto element_sector =
                (gmem_address / config_.gmem_sector_bytes) * config_.gmem_sector_bytes;
            if (!plan.fragments.empty() &&
                (element_sector != sector_address || gmem_address != expected_next_gmem)) {
                break;
            }
            sector_address = element_sector;
            expected_next_gmem = gmem_address + command.descriptor.element_bytes;
        } else if (bytes + command.descriptor.element_bytes > config_.gmem_sector_bytes) {
            break;
        }

        GmemSectorRequest::TmaFragment fragment;
        fragment.gmem_address = gmem_address;
        fragment.shmem_address = shmem_address;
        fragment.size = command.descriptor.element_bytes;
        plan.fragments.push_back(fragment);
        bytes += fragment.size;
        if (in_bounds) command.useful_bytes += fragment.size;
        (void)tma_advance_coordinate(command);
    }
    if (plan.fragments.empty()) throw SimError("tma walker failed to emit an element");
    plan.sector_address = sector_address;
    plan.sector = synthetic_zero ? TmaWalkerPlan::Sector::zero_fill
        : load ? TmaWalkerPlan::Sector::load : TmaWalkerPlan::Sector::store;
    if (!load) {
        for (const auto& fragment : plan.fragments) {
            const auto offset = fragment.gmem_address - sector_address;
            for (uint32_t byte = 0; byte < fragment.size; ++byte)
                plan.store_byte_enable |= 1ull << (offset + byte);
        }
    }
    return plan;
}

// The TMA's cycle, computed on a copy of its queue: the command end (the old
// event's command leaves before a presentation names a new one), its own
// requests, port A's completion, the ingress ticket, the write acks. Issue
// looks ahead at the result (queue entries, the TMA's next port-A request
// (c)); drive_tma_queue commits it with issue's new command.
void Machine::step_tma(Wires& w) {
    auto tma = tma_.cur();
    // The command whose event the register holds ends in the cycle the RMW
    // engine claims that event. (Today the TMA never presents in that cycle:
    // it presents only when event_command is empty in Q.)
    if (tma.event_command.has_value() && w.port_a.tma_event_claimed) {
        const auto& command = tma.commands.at(*tma.event_command);
        trace_->tma_command_end(cycle_ + 1, command.id, command.logical_bytes, command.useful_bytes);
        tma.commands.erase(*tma.event_command);
        tma.event_command.reset();
        trace_tma_depths(tma, cycle_ + 1);
    }
    apply_tma_requests(tma, w);
    if (w.port_a.tma_completed.has_value()) complete_tma_port_request(tma, *w.port_a.tma_completed, w);
    if (w.ingress.tma_ticket.has_value()) {
        tma.returns.push_back(*w.ingress.tma_ticket);
        trace_tma_depths(tma, cycle_);
    }
    for (uint32_t release = 0; release < w.controller.tma_credit_releases; ++release) release_tma_credit(tma);
    w.tma_next = std::move(tma);
}

void Machine::apply_tma_requests(TmaQueue& tma, Wires& w) {
    if (w.tma.present.has_value()) {
        tma.commands.at(*w.tma.present).state = TmaCommandState::event_presented;
        tma.event_command = w.tma.present;
    }
    if (!w.tma.port_a_idle) return;
    if (w.tma.placement) {
        tma.returns.pop_front(); // its rounds read the sector register until the last one
        trace_tma_depths(tma, cycle_);
        return;
    }
    tma.frontend = w.tma.frontend;
    if (!tma.frontend.has_value()) return;
    if (w.tma.frontend_reset) {
        tma.frontend.reset();
        return;
    }
    if (w.tma.walker.has_value()) apply_tma_walker(tma, w);
}

void Machine::apply_tma_walker(TmaQueue& tma, Wires& w) {
    using Sector = TmaWalkerPlan::Sector;
    const auto& plan = *w.tma.walker;
    auto& command = tma.commands.at(*tma.frontend);
    const bool emits = plan.sector != Sector::none && w.channel.walker_proceeds;
    if (plan.sector == Sector::none || emits) command = plan.command_after;
    if (emits) {
        ++tma.credits_used;
        trace_tma_depths(tma, cycle_);
        switch (plan.sector) {
        case Sector::zero_fill: {
            TmaReturn returned;
            returned.command = command.id;
            returned.sector_request = w.channel.walker_id;
            returned.fragments = plan.fragments;
            ++command.pending_load_writes;
            tma.returns.push_back(std::move(returned));
            trace_tma_depths(tma, cycle_);
            break;
        }
        case Sector::load:
            trace_->tma_sector(config_.gmem_sector_bytes);
            ++command.pending_load_writes;
            break;
        case Sector::store:
            trace_->tma_sector(config_.gmem_sector_bytes);
            ++command.pending_store_captures;
            w.tma_store_allocated = true;
            break;
        case Sector::none:
            break;
        }
    }
    if (!command.walker_done) return; // credit or channel pressure: retry next cycle
    const bool load = command.descriptor.direction == TmaDirection::load;
    if ((load && command.pending_load_writes == 0) || (!load && command.pending_store_captures == 0)) {
        mark_tma_data_complete(command);
    }
    tma.frontend.reset();
}

// The last round of a TMA port-A transaction has run (its data is in
// `request`). The sector registers' own changes are their drivers'.
void Machine::complete_tma_port_request(TmaQueue& tma, const TmaPortRequest& request, const Wires& w) {
    auto command_it = tma.commands.find(request.command);
    if (command_it == tma.commands.end()) throw SimError("orphaned tma port-A request");
    auto& command = command_it->second;

    switch (request.kind) {
    case TmaPortKind::descriptor: {
        if (command.state != TmaCommandState::descriptor) {
            throw SimError("tma descriptor read completed in the wrong command state");
        }
        std::array<uint8_t, tma_descriptor_bytes> bytes{};
        for (const auto& fragment : request.fragments) {
            const uint32_t offset = fragment.shmem_address - command.descriptor_address;
            if (offset + fragment.size > bytes.size()) {
                throw SimError("tma descriptor fragment is outside the descriptor");
            }
            std::memcpy(bytes.data() + offset, fragment.data.data(), fragment.size);
        }
        command.descriptor = decode_tma_descriptor(bytes.data(), bytes.size());
        command.logical_bytes = tma_logical_bytes(command.descriptor);
        validate_tma_runtime_descriptor(command);
        // releases the descriptor read-ownership; verifies the accept-time peek
        sanitizer_.record_tma_descriptor_read(command.id, command.descriptor);
        command.state = TmaCommandState::phase;
        return;
    }
    case TmaPortKind::phase: {
        if (command.state != TmaCommandState::phase) {
            throw SimError("tma mbar phase read completed in the wrong command state");
        }
        uint64_t word = 0;
        for (const auto& fragment : request.fragments) {
            const uint32_t offset = fragment.shmem_address - command.descriptor.completion_mbar;
            if (offset + fragment.size > sizeof(word)) {
                throw SimError("tma mbar fragment is outside the mbar");
            }
            std::memcpy(reinterpret_cast<uint8_t*>(&word) + offset,
                fragment.data.data(), fragment.size);
        }
        const auto state = decode_mbar_state(word);
        if (!state.valid || state.expected == 0 || state.remaining > state.expected) {
            throw SimError("tma completion mbar is uninitialized or corrupt");
        }
        if (state.tx_bytes < command.logical_bytes) {
            throw SimError("tma completion mbar does not expect enough transaction bytes");
        }
        command.mbar_phase = static_cast<uint8_t>(state.phase);
        command.state = TmaCommandState::walking;
        return;
    }
    case TmaPortKind::load_write:
        if (command.descriptor.direction != TmaDirection::load ||
            command.pending_load_writes == 0) {
            throw SimError("unexpected tma load return");
        }
        for (const auto& fragment : request.fragments) {
            if (static_cast<uint64_t>(fragment.shmem_address) + fragment.size > shmem_.size()) {
                throw SimError("tma load return is outside shmem");
            }
            for (const auto& wait_slot : mbar_wait_slots_.cur()) {
                if (!wait_slot.has_value()) continue;
                const auto fragment_end = static_cast<uint64_t>(fragment.shmem_address) + fragment.size;
                const auto mbar_end = static_cast<uint64_t>(wait_slot->byte_address) + 8;
                if (fragment.shmem_address < mbar_end && wait_slot->byte_address < fragment_end) {
                    throw SimError("tma load repurposes an mbar with a live waiter");
                }
            }
        }
        --command.pending_load_writes;
        release_tma_credit(tma);
        if (command.walker_done && command.pending_load_writes == 0) {
            mark_tma_data_complete(command);
        }
        return;
    case TmaPortKind::store_read: {
        // The TMA store register holds this sector: allocated in an earlier
        // cycle, or by the walker this cycle (a one-round source read).
        const auto& store = tma_store_sector_.cur();
        const bool held = (store.busy && !store.complete && store.command == command.id) ||
            (w.tma_store_allocated && w.tma.frontend == command.id);
        if (command.descriptor.direction != TmaDirection::store ||
            command.pending_store_captures == 0 || !held) {
            throw SimError("unexpected tma store capture");
        }
        // The source is captured: the TMA store sector register is complete
        // and goes to the egress FIFO from the next cycle on.
        --command.pending_store_captures;
        if (command.walker_done && command.pending_store_captures == 0) {
            mark_tma_data_complete(command);
        }
        return;
    }
    }
    throw SimError("unknown tma port-A request");
}

std::optional<uint64_t> Machine::next_tma_frontend_command(const TmaQueue& tma) const {
    for (const auto& [id, command] : tma.commands) {
        if (command.state == TmaCommandState::descriptor ||
            command.state == TmaCommandState::phase ||
            (command.state == TmaCommandState::walking && !command.walker_done)) {
            return id;
        }
    }
    return std::nullopt;
}

std::optional<uint64_t> Machine::next_data_complete_tma_command(const TmaQueue& tma) const {
    for (const auto& [id, command] : tma.commands) {
        if (command.state == TmaCommandState::data_complete) return id;
    }
    return std::nullopt;
}

// Port-A arbiter, TMA side, looking ahead (c): does the TMA have a port-A
// request it would start at the next cycle's free grant, as this cycle
// leaves it? (A complete sector or a zero fill to place, a descriptor or
// mbar-phase read, or a store walker that can start a sector's source read:
// a credit is free and the TMA store sector register is empty.)
bool Machine::tma_wants_port_a_next_cycle(const Wires& w) const {
    const auto& tma = w.tma_next;
    if (!tma.returns.empty()) return true;
    const auto id = tma.frontend.has_value() ? tma.frontend : next_tma_frontend_command(tma);
    if (!id.has_value()) return false;
    const auto found = tma.commands.find(*id);
    if (found == tma.commands.end()) return false;
    const auto& command = found->second;
    if (command.state == TmaCommandState::descriptor || command.state == TmaCommandState::phase) {
        return true;
    }
    const bool store_busy_next = (tma_store_sector_.cur().busy && !w.egress.tma_store_drained) ||
        w.tma_store_allocated;
    return command.state == TmaCommandState::walking && !command.walker_done &&
        command.descriptor.direction == TmaDirection::store &&
        tma.credits_used < config_.tma_sector_credits && !store_busy_next;
}

std::vector<std::vector<Machine::TmaPortRequest::Piece>> Machine::tma_port_rounds(
    const std::vector<GmemSectorRequest::TmaFragment>& fragments) const {
    using Piece = TmaPortRequest::Piece;
    std::vector<std::map<uint32_t, std::vector<Piece>>> words(config_.shmem_banks);
    std::set<std::pair<uint32_t, uint8_t>> occupied;
    for (uint32_t index = 0; index < fragments.size(); ++index) {
        const auto& fragment = fragments[index];
        if (fragment.size == 0 || fragment.size > fragment.data.size()) {
            throw SimError("invalid tma shmem fragment size");
        }
        uint8_t offset = 0;
        while (offset < fragment.size) {
            const uint32_t address = fragment.shmem_address + offset;
            const uint32_t word = address / 4;
            const uint8_t within = static_cast<uint8_t>(address % 4);
            const uint8_t size = static_cast<uint8_t>(
                std::min<uint32_t>(fragment.size - offset, 4 - within));
            for (uint8_t byte = 0; byte < size; ++byte) {
                if (!occupied.emplace(word, uint8_t(within + byte)).second) {
                    throw SimError("overlapping tma fragments target the same shmem byte");
                }
            }
            words[word % config_.shmem_banks][word].push_back(Piece{index, offset, size});
            offset = static_cast<uint8_t>(offset + size);
        }
    }
    std::size_t rounds = 0;
    for (const auto& bank : words) rounds = std::max(rounds, bank.size());
    if (rounds == 0 || rounds > std::numeric_limits<uint32_t>::max()) {
        throw SimError("invalid tma shmem request round count");
    }
    std::vector<std::vector<Piece>> result(rounds);
    for (const auto& bank : words) {
        std::size_t round = 0;
        for (const auto& [word, pieces] : bank) {
            (void)word;
            result[round].insert(result[round].end(), pieces.begin(), pieces.end());
            ++round;
        }
    }
    return result;
}

bool Machine::tma_advance_coordinate(TmaCommand& command) {
    const uint32_t source_x = command.descriptor.transpose
        ? command.descriptor.box[1] : command.descriptor.box[0];
    const uint32_t source_y = command.descriptor.transpose
        ? command.descriptor.box[0] : command.descriptor.box[1];
    if (++command.x < source_x) return true;
    command.x = 0;
    if (++command.y < source_y) return true;
    command.y = 0;
    if (++command.z < command.descriptor.box[2]) return true;
    command.z = 0;
    command.walker_done = true;
    return false;
}

bool Machine::tma_coordinate_in_bounds(const TmaCommand& command) const {
    const std::array<uint32_t, 3> local{command.x, command.y, command.z};
    for (uint32_t dimension = 0; dimension < command.descriptor.rank; ++dimension) {
        const auto coordinate = static_cast<int64_t>(command.origin[dimension]) + local[dimension];
        if (coordinate < 0 ||
            static_cast<uint64_t>(coordinate) >= command.descriptor.dimensions[dimension]) {
            return false;
        }
    }
    return true;
}

uint64_t Machine::tma_gmem_address(const TmaCommand& command) const {
    const auto x = static_cast<uint64_t>(
        static_cast<int64_t>(command.origin[0]) + command.x);
    const auto y = static_cast<uint64_t>(
        static_cast<int64_t>(command.origin[1]) + command.y);
    const auto z = static_cast<uint64_t>(
        static_cast<int64_t>(command.origin[2]) + command.z);
    uint64_t element = x;
    const auto add_product = [&](uint64_t coordinate, uint64_t stride) {
        if (coordinate != 0 && stride > UINT64_MAX / coordinate) {
            throw SimError("tma global address arithmetic overflow");
        }
        const auto term = coordinate * stride;
        if (element > UINT64_MAX - term) {
            throw SimError("tma global address arithmetic overflow");
        }
        element += term;
    };
    if (command.descriptor.rank >= 2) add_product(y, command.descriptor.strides[0]);
    if (command.descriptor.rank >= 3) add_product(z, command.descriptor.strides[1]);
    if (element > UINT64_MAX / command.descriptor.element_bytes) {
        throw SimError("tma global byte address arithmetic overflow");
    }
    const auto byte_offset = element * command.descriptor.element_bytes;
    if (command.descriptor.gmem_base > UINT64_MAX - byte_offset) {
        throw SimError("tma global byte address arithmetic overflow");
    }
    const auto address = command.descriptor.gmem_base + byte_offset;
    if (address > gmem_.size() || command.descriptor.element_bytes > gmem_.size() - address) {
        throw SimError("tma global access is outside gmem");
    }
    return address;
}

uint32_t Machine::tma_shmem_address(const TmaCommand& command) const {
    const uint32_t u = command.descriptor.transpose ? command.y : command.x;
    const uint32_t v = command.descriptor.transpose ? command.x : command.y;
    // Stage-granular boxes: with E = 16 / element_bytes elements per 16-byte
    // row (8 fp16, 4 fp32), logical K selects the 1024-byte object (k / E)
    // and the byte within the row ((k % E) * element_bytes); the outer
    // coordinate is the row. For fp16 this is exactly the former k >> 3 /
    // k & 7 / 2*k map, and a K-extent-8 box keeps the single-object behavior.
    const uint32_t element_bytes = command.descriptor.element_bytes;
    const uint32_t row_elements = 16u / element_bytes;
    if (command.descriptor.swizzle == static_cast<uint8_t>(TmaSwizzle::xor64_a)) {
        return xor64_byte_address(
            command.descriptor.shmem_base + 1024 * (u / row_elements), v,
            (u % row_elements) * element_bytes, 0);
    }
    if (command.descriptor.swizzle == static_cast<uint8_t>(TmaSwizzle::xor64_b)) {
        return xor64_byte_address(
            command.descriptor.shmem_base + 1024 * (v / row_elements), u,
            (v % row_elements) * element_bytes, 2);
    }
    uint64_t element = u;
    element += static_cast<uint64_t>(v) * command.descriptor.box[0];
    element += static_cast<uint64_t>(command.z) * command.descriptor.box[0] *
        command.descriptor.box[1];
    const auto offset = element * command.descriptor.element_bytes;
    const auto address = static_cast<uint64_t>(command.descriptor.shmem_base) + offset;
    if (address > shmem_.size() || command.descriptor.element_bytes > shmem_.size() - address) {
        throw SimError("tma shmem access is outside shmem");
    }
    return static_cast<uint32_t>(address);
}

void Machine::validate_tma_runtime_descriptor(const TmaCommand& command) const {
    if (command.logical_bytes == 0 ||
        command.logical_bytes > std::numeric_limits<uint32_t>::max()) {
        throw SimError("tma logical byte count is outside the mbar encoding");
    }
    if (command.descriptor.shmem_base > shmem_.size() ||
        command.logical_bytes > shmem_.size() - command.descriptor.shmem_base) {
        throw SimError("tma logical box does not fit in shmem");
    }
    if (static_cast<uint64_t>(command.descriptor.completion_mbar) + 8 > shmem_.size()) {
        throw SimError("tma completion mbar is outside shmem");
    }
    if (command.descriptor.gmem_base > gmem_.size()) {
        throw SimError("tma global base is outside gmem");
    }
}

void Machine::release_tma_credit(TmaQueue& tma) {
    if (tma.credits_used == 0) throw SimError("tma sector credit underflow");
    --tma.credits_used;
    trace_tma_depths(tma, cycle_);
}

void Machine::trace_tma_depths(const TmaQueue& tma, uint64_t at_cycle) {
    trace_->tma_depths(at_cycle, tma.commands.size(), tma.credits_used, tma.returns.size());
}

// Hardware: the command's data movement is done. Sanitizer: its box
// ownership ends here.
void Machine::mark_tma_data_complete(TmaCommand& command) {
    command.state = TmaCommandState::data_complete;
    sanitizer_.record_tma_data_complete(command.id);
}

// ===========================================================================
// Issue: the scheduler, the last decision of the cycle
// ===========================================================================

// The scheduler: from the cursor, the first warp whose instruction can be
// accepted issues (one per cycle); every warp's state is traced.
void Machine::decide_issue(Wires& w) {
    struct State {
        WarpState state = WarpState::not_selected;
        std::string detail;
    };
    std::vector<State> states(warps_.size());
    bool issued = false;
    for (uint32_t offset = 0; offset < warps_.size(); ++offset) {
        const auto index = (scheduler_cursor_ + offset) % static_cast<uint32_t>(warps_.size());
        auto& warp = warps_[index];
        if (warp.halted) {
            states[index].state = WarpState::halted;
            continue;
        }
        if (waiting_slot_of(w, warp.id).has_value()) {
            states[index] = {WarpState::mbar_wait, "mbar phase"};
            continue;
        }
        if (cycle_ < warp.stall_until) {
            states[index] = {WarpState::stall_count, "stall count"};
            continue;
        }
        const auto attempt = try_issue(warp, w);
        if (attempt.issued) {
            states[index] = {WarpState::issued, {}};
            scheduler_cursor_ = (index + 1) % static_cast<uint32_t>(warps_.size());
            issued = true;
            break;
        }
        states[index] = {attempt.state, attempt.reason};
    }

    if (!issued) {
        const auto draining = std::all_of(warps_.begin(), warps_.end(), [](const Warp& warp) {
            return warp.halted;
        });
        trace_->scheduler_idle(draining);
    }
    for (uint32_t warp = 0; warp < warps_.size(); ++warp) {
        if (states[warp].state == WarpState::not_selected && warps_[warp].halted) {
            states[warp].state = WarpState::halted;
        }
        trace_->warp_state(cycle_, warp, states[warp].state, std::move(states[warp].detail));
    }
}

// One warp's issue attempt. All-or-nothing: every structural check comes
// before anything is booked or handed over. The warp's own state (pc, stall
// count, halted, the SFU and the arithmetic operand latches) is the issue
// unit's and changes in place; everything for another unit goes into
// w.issue, which that unit's driver applies.
Machine::IssueAttempt Machine::try_issue(Warp& warp, Wires& w) {
    const auto inst = instruction(warp);
    if ((inst.opcode == Opcode::ldg_v8 && static_cast<unsigned>(inst.dst) + 8 > 256) ||
        (inst.opcode == Opcode::stg_v8 && static_cast<unsigned>(inst.src1) + 8 > 256))
        panic(warp, inst, "global vector register encoding wraps");
    if (inst.opcode == Opcode::acc_ld || inst.opcode == Opcode::acc_ld_v8 ||
        inst.opcode == Opcode::acc_st || inst.opcode == Opcode::acc_st_v8) {
        const bool load = inst.opcode == Opcode::acc_ld || inst.opcode == Opcode::acc_ld_v8;
        const auto coord = load ? inst.src1 : inst.src0;
        const auto base = load ? inst.dst : inst.src1;
        const auto width = inst.opcode == Opcode::acc_ld_v8 || inst.opcode == Opcode::acc_st_v8 ? 8u : 1u;
        if (coord > 254 || static_cast<unsigned>(base) + width > 256)
            panic(warp, inst, "accumulator register encoding wraps");
    }
    if (inst.opcode == Opcode::bra &&
        (inst.control.predicate != 7 || inst.control.predicate_invert)) {
        panic(warp, inst, "bra is unconditional");
    }
    if ((inst.control.predicate != 7 || inst.control.predicate_invert) &&
        inst.opcode != Opcode::bra_uni) {
        panic(warp, inst, "only bra.uni may be predicated before divergent execution exists");
    }
    // depbar waits see this cycle's completions (bypass (b)).
    for (uint8_t bar = 0; bar < 6; ++bar) {
        if ((inst.control.wait_mask & (1u << bar)) != 0 && depbar_after_completions(w, warp.id, bar) != 0) {
            return {false, WarpState::depbar, "sb" + std::to_string(bar)};
        }
    }

    const auto sources = source_registers(inst);
    const auto destination = destination_register(inst);
    for (const auto reg : sources) {
        if (reg >= kernel_.register_count) panic(warp, inst, "source register out of range");
    }
    if (destination.has_value() && *destination >= kernel_.register_count) {
        panic(warp, inst, "destination register out of range");
    }
    if (is_setp(inst.opcode) || is_mbar_predicate_writer(inst.opcode)) {
        if (inst.dst >= 7) panic(warp, inst, "predicate destination is not p0..p6");
        sanitizer_.check_pred_waw(site(warp, inst), inst.dst);
    }
    if (inst.opcode == Opcode::bra_uni && inst.control.predicate != 7) {
        sanitizer_.check_pred_raw(site(warp, inst), inst.control.predicate);
    }
    check_register_hazards(warp, inst);
    auto issue_sources = sources;
    if (inst.opcode == Opcode::acc_st || inst.opcode == Opcode::acc_st_v8) {
        issue_sources = {inst.src0, static_cast<uint8_t>(inst.src0 + 1)};
    } else if (inst.opcode == Opcode::stg_v8) {
        issue_sources = {inst.src0};
    }
    const auto reads = rf_read_plan(inst, config_, issue_sources);
    if (!can_reserve_reads(reads)) {
        return {false, WarpState::structural, "RF read reservation"};
    }
    if (uses_arithmetic_pipeline(inst)) {
        // All fixed operations (including MOV/SREAD) enter execution at the same
        // offset. Single accepted issue/cycle therefore reserves execution as well.
        // Check everything before mutating either RF calendar or warp state.
        if (arithmetic_reads_.size() >= config_.rf_arithmetic_read_cycles) {
            throw SimError("internal RF read pipeline exceeded its bounded window");
        }
        // The single SFU must be free by this op's execution start (the
        // window's final cycle). Check-only: a rejected admission books nothing.
        if (is_sfu(inst.opcode) &&
            cycle_ + config_.rf_arithmetic_read_cycles - 1 < sfu_busy_until_) {
            trace_->sfu_unit_stall();
            return {false, WarpState::structural, "SFU unit busy"};
        }
        const auto result_cycle = cycle_ + scheduled_fixed_latency(inst, config_);
        if (destination.has_value() && !reserve_write(*destination, result_cycle, RfRequester::issue)) {
            return {false, WarpState::structural, "RF write reservation"};
        }
        reserve_reads(reads, RfRequester::issue);
        if (is_sfu(inst.opcode)) {
            const auto occupancy = (config_.lanes + config_.sfu_lanes - 1) / config_.sfu_lanes;
            sfu_busy_until_ = cycle_ + config_.rf_arithmetic_read_cycles - 1 + occupancy;
            trace_->sfu_accept(occupancy);
        }
        for (const auto& read : reads) {
            sanitizer_.record_reg_read_until(warp.id, read.reg, cycle_ + read.offset + 1);
        }
        if (destination.has_value()) sanitizer_.record_reg_ready(warp.id, *destination, result_cycle);
        else sanitizer_.record_pred_ready(warp.id, inst.dst, result_cycle);
        ArithmeticRead pending;
        pending.issue_cycle = cycle_;
        pending.warp = warp.id;
        pending.instruction = inst;
        pending.reads = reads;
        pending.operands.resize(reads.size());
        arithmetic_reads_.push_back(std::move(pending));
        warp.stall_until = cycle_ + inst.control.stall;
        const auto old_pc = warp.pc;
        warp.pc += 8;
        trace_->issue(cycle_, warp.id, old_pc, inst);
        return {true, WarpState::issued, {}};
    }
    // Structural space, as this cycle leaves it (look-ahead (c)).
    if (is_mbar_try_wait(inst.opcode) && !free_mbar_wait_slot(w).has_value()) {
        return {false, WarpState::mbar_wait_slot, "mbar wait slots full"};
    }
    if (is_tma(inst.opcode) && w.tma_next.commands.size() == config_.tma_queue_commands) {
        return {false, WarpState::structural, "tma command queue full"};
    }
    if ((inst.opcode == Opcode::mma_async || inst.opcode == Opcode::mma_async_init ||
         inst.opcode == Opcode::mma_commit) && w.mma_next.queue.size() >= config_.mma_queue_commands) {
        return {false, WarpState::structural, "MMA command queue full"};
    }
    const bool acc_load = inst.opcode == Opcode::acc_ld || inst.opcode == Opcode::acc_ld_v8;
    const bool acc_store = inst.opcode == Opcode::acc_st || inst.opcode == Opcode::acc_st_v8;
    if (acc_load || acc_store) {
        const auto acc_id = acc_load ? inst.src0 : inst.dst;
        if (acc_id >= kernel_.accumulators.size()) panic(warp, inst, "accumulator id out of range");
        // Racing accepted MMA work is UB (cases 01/02): no ownership hardware
        // exists, but the simulator has global knowledge and diagnoses it.
        sanitizer_.check_acc_transfer_vs_mma(site(warp, inst), acc_id);
        if (!acc_slot_available(w, kernel_.accumulators[acc_id].bank, acc_load)) {
            return {false, WarpState::structural, "tmem bank transfer busy"};
        }
        // Admitted. Transfer vs transfer is checked after the structural slot
        // stall. The coordinates are read here only to name the region (the
        // same uniform reads, with the same panics, recur at execution below).
        const uint8_t coord = acc_load ? inst.src1 : inst.src0;
        const auto row = uniform_reg(warp, inst, coord, "accumulator row coordinate");
        const auto col = uniform_reg(
            warp, inst, static_cast<uint8_t>(coord + 1), "accumulator column coordinate");
        sanitizer_.check_acc_transfer_overlap(site(warp, inst), acc_id, acc_load, row, col,
            inst.opcode == Opcode::acc_ld_v8 || inst.opcode == Opcode::acc_st_v8 ? 8u : 1u);
    }
    // Port-A arbiter, SIMT side. A SIMT access accepted now starts next
    // cycle, so it needs port A free next cycle; and SIMT may not take that
    // free grant while the TMA will want it (TMA priority, counted per
    // warp-cycle). Both as this cycle leaves them (look-ahead (c)).
    if (inst.opcode == Opcode::lds || inst.opcode == Opcode::sts ||
        is_mbar_rmw(inst.opcode) ||
        is_mbar_test_wait(inst.opcode) || is_mbar_try_wait(inst.opcode)) {
        if (port_a_busy_next_cycle(w)) return {false, WarpState::structural, "port A busy"};
        if (tma_wants_port_a_next_cycle(w)) {
            trace_->shmem_tma_priority_wait();
            return {false, WarpState::structural, "port A: TMA priority"};
        }
    }

    // LSU admission: a command slot AND an assembly register, or the warp is
    // ineligible this cycle. Addresses are checked first: a malformed access
    // panics as soon as the warp tries it. Slots and registers a command
    // released this cycle are free again (d).
    std::optional<LsuCommand> lsu_plan;
    std::optional<uint32_t> lsu_assembly;
    if (is_lsu(inst.opcode)) {
        lsu_plan = plan_lsu_command(warp, inst);
        const auto in_flight = lsu_commands_.cur().size() - w.released_lsu_commands.size();
        if (in_flight >= config_.lsu_active_commands) {
            ++trace_->lsu_counters().lsu_no_slot_cycles;
            return {false, WarpState::structural, "LSU command slot"};
        }
        lsu_assembly = free_assembly_register(w);
        if (!lsu_assembly.has_value()) {
            ++trace_->lsu_counters().lsu_no_staging_cycles;
            return {false, WarpState::structural, "LSU assembly register"};
        }
    }

    const auto latency = fixed_latency(inst, config_.timing);
    const auto result_cycle = cycle_ + latency;
    uint32_t next_pc = warp.pc + 8;
    const auto mbar_address = [&] {
        const auto address = uniform_reg(warp, inst, inst.src0, "mbar address");
        if (address % 8 != 0 || static_cast<uint64_t>(address) + 8 > shmem_.size()) {
            panic(warp, inst, "mbar address is not 8-byte aligned and in shmem");
        }
        return address;
    };

    switch (inst.opcode) {
    case Opcode::mma_async:
    case Opcode::mma_async_init: {
        if (inst.dst >= kernel_.accumulators.size()) panic(warp, inst, "accumulator id out of range");
        const auto& acc = kernel_.accumulators[inst.dst];
        if (acc.rows != config_.mma_m || acc.cols != config_.mma_n)
            panic(warp, inst, "MMA accumulator shape does not match the device profile");
        const auto a = uniform_reg(warp, inst, inst.src0, "MMA A address");
        const auto b = uniform_reg(warp, inst, inst.src1, "MMA B address");
        const uint64_t a_bytes = static_cast<uint64_t>(config_.mma_m) * config_.mma_k * 2;
        const uint64_t b_bytes = static_cast<uint64_t>(config_.mma_k) * config_.mma_n * 2;
        if ((config_.mma_xor_layout && ((a | b) & 63u) != 0) ||
            static_cast<uint64_t>(a) + a_bytes > shmem_.size() ||
            static_cast<uint64_t>(b) + b_bytes > shmem_.size())
            panic(warp, inst, "MMA operand alignment or footprint is outside shmem");
        if (inst.control.set_depbar >= 6) panic(warp, inst, "mma.async has no operand-consumed depbar");
        admit_depbar(w, warp.id, inst.control.set_depbar);
        MmaCommand command;
        command.id = next_mma_command_id_++;
        command.init = inst.opcode == Opcode::mma_async_init;
        command.warp = warp.id;
        command.accumulator = inst.dst;
        command.depbar = inst.control.set_depbar;
        command.a_base = a;
        command.b_base = b;
        sanitizer_.record_mma_accept(site(warp, inst), command.id, command.accumulator,
            command.depbar, mma_operand_span(a, true), mma_operand_span(b, false));
        trace_->mma_command(cycle_, command.id, warp.id, inst.opcode == Opcode::mma_async_init);
        w.issue.mma = std::move(command);
        break;
    }
    case Opcode::mma_commit: {
        MmaCommand command;
        command.id = next_mma_command_id_++;
        command.commit = true;
        command.warp = warp.id;
        command.mbar_address = mbar_address();
        w.issue.mma = std::move(command);
        break;
    }
    case Opcode::acc_ld:
    case Opcode::acc_ld_v8:
    case Opcode::acc_st:
    case Opcode::acc_st_v8: {
        const bool load = inst.opcode == Opcode::acc_ld || inst.opcode == Opcode::acc_ld_v8;
        const uint32_t width = inst.opcode == Opcode::acc_ld_v8 || inst.opcode == Opcode::acc_st_v8 ? 8 : 1;
        const uint8_t acc_id = load ? inst.src0 : inst.dst;
        const uint8_t coord = load ? inst.src1 : inst.src0;
        const uint8_t rf_base = load ? inst.dst : inst.src1;
        const auto& acc = kernel_.accumulators[acc_id];
        const auto row = uniform_reg(warp, inst, coord, "accumulator row coordinate");
        const auto col = uniform_reg(warp, inst, static_cast<uint8_t>(coord + 1), "accumulator column coordinate");
        if (static_cast<uint64_t>(row) + config_.lanes > acc.rows ||
            static_cast<uint64_t>(col) + width > acc.cols ||
            static_cast<unsigned>(rf_base) + width > kernel_.register_count) {
            panic(warp, inst, "accumulator coordinate or RF range out of bounds");
        }
        if (inst.control.set_depbar >= 6) panic(warp, inst, "accumulator transfer has no depbar");
        admit_depbar(w, warp.id, inst.control.set_depbar);
        AccTransfer transfer;
        transfer.load = load;
        transfer.warp = warp.id;
        transfer.accumulator = acc_id;
        transfer.rf_base = rf_base;
        transfer.depbar = inst.control.set_depbar;
        transfer.row_base = row;
        transfer.col_base = col;
        transfer.width = width;
        if (load) {
            for (uint32_t j = 0; j < width; ++j)
                sanitizer_.record_reg_ready(warp.id, rf_base + j, std::numeric_limits<uint64_t>::max());
        } else {
            for (uint32_t j = 0; j < width; ++j) sanitizer_.record_async_reader_add(warp.id, rf_base + j);
        }
        // conflicts were checked at admission above
        sanitizer_.record_acc_transfer_accept(site(warp, inst), acc.bank, load, acc_id,
            row, col, width, inst.control.set_depbar);
        w.issue.acc = IssueDecision::AccAdmission{acc.bank, load, std::move(transfer)};
        break;
    }
    case Opcode::lds: {
        if (inst.control.set_depbar >= 6) panic(warp, inst, "lds has no depbar");
        auto rounds = build_shmem_rounds(warp, inst, true, w);
        const auto complete_cycle = cycle_ + rounds.size() + 1;
        if (!reserve_write(inst.dst, complete_cycle, RfRequester::issue)) {
            return {false, WarpState::structural, "RF write reservation"};
        }
        // admitted: sanitizer check
        sanitizer_.check_simt_shmem_access(site(warp, inst), access_addresses(rounds), true);
        sanitizer_.record_reg_ready(warp.id, inst.dst, complete_cycle);
        admit_depbar(w, warp.id, inst.control.set_depbar);
        ShmemRequest request;
        request.load = true;
        request.warp = warp.id;
        request.dst = inst.dst;
        request.depbar = inst.control.set_depbar;
        request.rounds = std::move(rounds);
        request.load_values.resize(config_.lanes, 0);
        trace_->shmem_begin(
            cycle_, warp.id, true, static_cast<uint32_t>(request.rounds.size()));
        w.issue.port_a = std::move(request);
        break;
    }
    case Opcode::sts: {
        if (inst.control.set_depbar >= 6) panic(warp, inst, "sts has no depbar");
        auto rounds = build_shmem_rounds(warp, inst, false, w);
        sanitizer_.check_simt_shmem_access(site(warp, inst), access_addresses(rounds), false);
        admit_depbar(w, warp.id, inst.control.set_depbar);
        ShmemRequest request;
        request.load = false;
        request.warp = warp.id;
        request.depbar = inst.control.set_depbar;
        request.rounds = std::move(rounds);
        trace_->shmem_begin(
            cycle_, warp.id, false, static_cast<uint32_t>(request.rounds.size()));
        w.issue.port_a = std::move(request);
        break;
    }
    case Opcode::ldg:
    case Opcode::stg:
    case Opcode::ldg_v8:
    case Opcode::stg_v8:
    case Opcode::ldg_uni: {
        if (inst.control.set_depbar >= 6) panic(warp, inst, "global memory operation has no depbar");
        auto& counters = trace_->lsu_counters();
        auto command = std::move(*lsu_plan);
        command.id = next_lsu_command_id_++;
        command.depbar = inst.control.set_depbar;
        command.assembly = *lsu_assembly;
        IssueDecision::LsuAdmission admission;
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
        case LsuKind::scalar_store: {
            // The issue-cycle RF read beat (planned with the address operand)
            // lands the store data in the command's assembly register.
            std::vector<uint32_t> data(config_.lanes);
            for (uint32_t lane = 0; lane < config_.lanes; ++lane) data[lane] = rf(warp, lane, inst.src1);
            admission.store_data = std::move(data);
            probe(LsuEdge::rf_read_beat, command.id, inst.src1, config_.lanes * 4);
            ++counters.lsu_scalar_stores;
            counters.lsu_useful_bytes += 4ull * config_.lanes;
            break;
        }
        case LsuKind::uniform_load:
            sanitizer_.record_reg_ready(warp.id, command.reg, std::numeric_limits<uint64_t>::max());
            ++counters.lsu_uniform_loads;
            counters.lsu_useful_bytes += 4;
            break;
        }
        admit_depbar(w, warp.id, command.depbar);
        trace_->lsu_command_begin(cycle_, command.id, warp.id, opcode_name(inst.opcode));
        lsu_phases_[command.id].admit_cycle = cycle_;
        // Sanitizer: every admitted command owns one assembly register for its
        // whole life, so the admitted set can never exceed the pool.
        sanitizer_.check_lsu_admitted(lsu_commands_.cur().size() - w.released_lsu_commands.size() + 1);
        admission.command = std::move(command);
        w.issue.lsu = std::move(admission);
        break;
    }
    case Opcode::tma_async: {
        if (inst.imm < 0 || inst.imm % 64 != 0 ||
            static_cast<uint64_t>(inst.imm) + tma_descriptor_bytes > shmem_.size()) {
            panic(warp, inst, "tma descriptor is not 64-byte aligned and in shmem");
        }
        if (inst.src0 > 253) panic(warp, inst, "tma coordinate register triplet is invalid");
        const auto command_id = next_tma_command_id_++;
        TmaCommand command;
        command.id = command_id;
        command.warp = warp.id;
        command.pc = warp.pc;
        command.instruction = inst;
        command.descriptor_address = static_cast<uint32_t>(inst.imm);
        command.origin = {
            static_cast<int32_t>(uniform_reg(warp, inst, inst.src0, "tma x coordinate")),
            static_cast<int32_t>(uniform_reg(warp, inst, inst.src0 + 1, "tma y coordinate")),
            static_cast<int32_t>(uniform_reg(warp, inst, inst.src0 + 2, "tma z coordinate")),
        };
        sanitizer_.record_tma_accept(site(warp, inst), command_id, command.descriptor_address,
            pending_tma_descriptor_bytes(command.descriptor_address, w));
        trace_->tma_command_begin(cycle_, command_id, warp.id, static_cast<uint32_t>(inst.imm),
            w.tma_next.commands.size() + 1);
        w.issue.tma = std::move(command);
        break;
    }
    case Opcode::mbar_init_i:
    case Opcode::mbar_init_r:
    case Opcode::mbar_arrive:
    case Opcode::mbar_arrive_expect_tx_i:
    case Opcode::mbar_arrive_expect_tx_r: {
        MbarRmwRequest request;
        request.byte_address = mbar_address();
        request.warp = warp.id;
        request.pc = warp.pc;
        request.instruction = inst;
        if (inst.opcode == Opcode::mbar_init_i || inst.opcode == Opcode::mbar_init_r) {
            request.operation = MbarRmwOperation::init;
            request.value = inst.opcode == Opcode::mbar_init_i
                ? static_cast<uint32_t>(inst.imm)
                : uniform_reg(warp, inst, inst.src1, "mbar expected count");
            if (request.value == 0 || request.value > mbar_encoding::max_count) {
                panic(warp, inst, "mbar expected count is outside 1..32767");
            }
        } else if (inst.opcode == Opcode::mbar_arrive) {
            request.operation = MbarRmwOperation::arrive;
        } else {
            request.operation = MbarRmwOperation::arrive_expect_tx;
            if (inst.opcode == Opcode::mbar_arrive_expect_tx_i && inst.imm < 0) {
                panic(warp, inst, "mbar tx byte count is negative");
            }
            request.value = inst.opcode == Opcode::mbar_arrive_expect_tx_i
                ? static_cast<uint32_t>(inst.imm)
                : uniform_reg(warp, inst, inst.src1, "mbar tx byte count");
        }
        sanitizer_.check_simt_mbar_rmw(site(warp, inst), request.byte_address);
        // The engine is free next cycle (port A busy is a structural stall
        // above), so the accepted request takes it; its read cycle is next.
        ActiveMbarRmw active;
        active.request = std::move(request);
        w.issue.rmw = std::move(active);
        break;
    }
    case Opcode::mbar_test_wait_i:
    case Opcode::mbar_test_wait_r:
    case Opcode::mbar_try_wait_i:
    case Opcode::mbar_try_wait_r: {
        const bool test = is_mbar_test_wait(inst.opcode);
        const bool immediate = inst.opcode == Opcode::mbar_test_wait_i ||
            inst.opcode == Opcode::mbar_try_wait_i;
        const auto parity_value = immediate
            ? static_cast<uint32_t>(inst.imm)
            : uniform_reg(warp, inst, inst.src1, "mbar parity");
        if (parity_value > 1) panic(warp, inst, "mbar parity is not 0 or 1");
        MbarReadRequest request;
        request.test = test;
        request.warp = warp.id;
        request.predicate_dst = inst.dst;
        request.byte_address = mbar_address();
        request.parity = static_cast<uint8_t>(parity_value);
        request.pc = warp.pc;
        request.instruction = inst;
        if (test) {
            sanitizer_.record_pred_ready(warp.id, inst.dst, result_cycle);
        } else {
            const auto slot_index = free_mbar_wait_slot(w);
            if (!slot_index.has_value()) {
                throw SimError("internal mbar try-wait slot disappeared during issue");
            }
            MbarWaitSlot slot;
            slot.warp = warp.id;
            slot.byte_address = request.byte_address;
            slot.parity = request.parity;
            slot.predicate_dst = request.predicate_dst;
            w.issue.wait_slot = std::make_pair(*slot_index, slot);
            sanitizer_.record_pred_ready(warp.id, inst.dst, std::numeric_limits<uint64_t>::max());
            request.wait_slot = *slot_index;
        }
        w.issue.mbar_read = std::move(request);
        break;
    }
    case Opcode::bra:
    case Opcode::bra_uni: {
        bool take = true;
        if (inst.opcode == Opcode::bra_uni && inst.control.predicate != 7) {
            take = warp.predicates[inst.control.predicate][0] != 0;
            if (inst.control.predicate_invert) take = !take;
            for (uint32_t lane = 1; lane < config_.lanes; ++lane) {
                bool lane_take = warp.predicates[inst.control.predicate][lane] != 0;
                if (inst.control.predicate_invert) lane_take = !lane_take;
                if (lane_take != take) {
                    panic(warp, inst, "bra.uni predicate is not warp-uniform");
                }
            }
        } else if (inst.opcode == Opcode::bra_uni && inst.control.predicate_invert) {
            take = false; // @!PT
        }
        if (take) {
            const auto target = static_cast<int64_t>(warp.pc) + 8 + inst.imm;
            if (inst.imm % 8 != 0 || target < 0 ||
                static_cast<uint64_t>(target / 8) >= kernel_.words.size()) {
                panic(warp, inst, "branch target is outside the kernel");
            }
            next_pc = static_cast<uint32_t>(target);
        }
        break;
    }
    case Opcode::halt:
        warp.halted = true;
        break;
    default:
        panic(warp, inst, "unknown opcode");
    }

    if (is_mbar_test_wait(inst.opcode)) {
        if (latency != 1) panic(warp, inst, "mbar.test_wait must have one-cycle predicate latency");
    } else if (latency != 0) {
        panic(warp, inst, "fixed-latency opcode bypassed the arithmetic read pipeline");
    } else if (inst.opcode != Opcode::lds && inst.opcode != Opcode::sts &&
               inst.opcode != Opcode::ldg && inst.opcode != Opcode::stg &&
               inst.opcode != Opcode::ldg_v8 && inst.opcode != Opcode::stg_v8 &&
               inst.opcode != Opcode::ldg_uni && inst.opcode != Opcode::tma_async &&
               inst.opcode != Opcode::acc_ld && inst.opcode != Opcode::acc_ld_v8 &&
               inst.opcode != Opcode::acc_st && inst.opcode != Opcode::acc_st_v8 &&
               inst.opcode != Opcode::mma_async && inst.opcode != Opcode::mma_async_init &&
               inst.opcode != Opcode::mma_commit &&
               inst.opcode != Opcode::bra && inst.opcode != Opcode::bra_uni &&
               !is_mbar_rmw(inst.opcode) && !is_mbar_try_wait(inst.opcode) &&
               inst.opcode != Opcode::halt) {
        panic(warp, inst, "zero-latency RF writer is unsupported");
    }

    reserve_reads(reads, RfRequester::issue);
    for (const auto& read : reads) trace_->rf_read(cycle_, warp.id, read.reg, read.reg % config_.rf_banks);
    warp.stall_until = cycle_ + inst.control.stall;
    const auto old_pc = warp.pc;
    warp.pc = next_pc;
    trace_->issue(cycle_, warp.id, old_pc, inst);
    return {true, WarpState::issued, {}};
}

// ===========================================================================
// Arithmetic pipeline
// ===========================================================================

void Machine::execute_arithmetic(ArithmeticRead& pending, Wires& w) {
    auto& warp = warps_.at(pending.warp);
    const auto& inst = pending.instruction;
    std::vector<uint32_t> result(config_.lanes, 0);
    // The plan holds each source register once; duplicated operand slots
    // broadcast from the same captured latch.
    const auto operand = [&](uint8_t reg, uint32_t lane) {
        for (std::size_t i = 0; i < pending.reads.size(); ++i) {
            if (pending.reads[i].reg == reg) return pending.operands[i][lane];
        }
        throw SimError("internal arithmetic operand was not reserved");
    };
    switch (inst.opcode) {
    case Opcode::sread:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            switch (static_cast<SpecialReg>(inst.imm)) {
            case SpecialReg::lane_id: result[lane] = lane; break;
            case SpecialReg::warp_id: result[lane] = warp.id; break;
            case SpecialReg::thread_id: result[lane] = warp.id * config_.lanes + lane; break;
            case SpecialReg::n_warps: result[lane] = static_cast<uint32_t>(warps_.size()); break;
            default: panic(warp, inst, "invalid special register");
            }
        }
        break;
    case Opcode::mov:
        std::fill(result.begin(), result.end(), static_cast<uint32_t>(inst.imm));
        break;
    case Opcode::add:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = operand(inst.src0, lane) + operand(inst.src1, lane);
        }
        break;
    case Opcode::addi:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = operand(inst.src0, lane) + static_cast<uint32_t>(inst.imm);
        }
        break;
    case Opcode::shli:
        if (inst.imm < 0 || inst.imm > 31) panic(warp, inst, "shift outside 0..31");
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = operand(inst.src0, lane) << inst.imm;
        }
        break;
    case Opcode::shri:
        if (inst.imm < 0 || inst.imm > 31) panic(warp, inst, "shift outside 0..31");
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = operand(inst.src0, lane) >> inst.imm;
        }
        break;
    case Opcode::xori:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = operand(inst.src0, lane) ^ static_cast<uint32_t>(inst.imm);
        }
        break;
    case Opcode::andi:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = operand(inst.src0, lane) & static_cast<uint32_t>(inst.imm);
        }
        break;
    case Opcode::ori:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = operand(inst.src0, lane) | static_cast<uint32_t>(inst.imm);
        }
        break;
    case Opcode::and_:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = operand(inst.src0, lane) & operand(inst.src1, lane);
        }
        break;
    case Opcode::or_:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = operand(inst.src0, lane) | operand(inst.src1, lane);
        }
        break;
    case Opcode::xor_:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = operand(inst.src0, lane) ^ operand(inst.src1, lane);
        }
        break;
    // Register shifts use the low 5 bits of the per-lane amount: the datapath
    // is a physical 5-bit shifter, so the masking IS the hardware, not a
    // silently repaired error.
    case Opcode::shl:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = operand(inst.src0, lane) << (operand(inst.src1, lane) & 31);
        }
        break;
    case Opcode::shr:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = operand(inst.src0, lane) >> (operand(inst.src1, lane) & 31);
        }
        break;
    case Opcode::fadd:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = float_bits(
                bits_float(operand(inst.src0, lane)) + bits_float(operand(inst.src1, lane)));
        }
        break;
    case Opcode::fmul:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = canonical_float(
                bits_float(operand(inst.src0, lane)) * bits_float(operand(inst.src1, lane)));
        }
        break;
    case Opcode::fsub:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = canonical_float(
                bits_float(operand(inst.src0, lane)) - bits_float(operand(inst.src1, lane)));
        }
        break;
    case Opcode::fmax:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = float_max_bits(operand(inst.src0, lane), operand(inst.src1, lane));
        }
        break;
    case Opcode::fma:
        // Single rounding via std::fma; NaN outputs canonicalized above.
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = canonical_float(std::fma(
                bits_float(operand(inst.src0, lane)),
                bits_float(operand(inst.src1, lane)),
                bits_float(operand(inst.src2, lane))));
        }
        break;
    case Opcode::cvt_pack:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = static_cast<uint32_t>(float_to_half_bits(operand(inst.src0, lane))) |
                static_cast<uint32_t>(float_to_half_bits(operand(inst.src1, lane))) << 16;
        }
        break;
    case Opcode::ex2:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = sfu_ex2(operand(inst.src0, lane));
        }
        break;
    case Opcode::rcp:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = sfu_rcp(operand(inst.src0, lane));
        }
        break;
    case Opcode::setp_eq:
    case Opcode::setp_ne:
    case Opcode::setp_lt:
    case Opcode::setp_le:
    case Opcode::setp_gt:
    case Opcode::setp_ge:
    case Opcode::setpi_eq:
    case Opcode::setpi_ne:
    case Opcode::setpi_lt:
    case Opcode::setpi_le:
    case Opcode::setpi_gt:
    case Opcode::setpi_ge:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            const auto rhs = is_setp_immediate(inst.opcode)
                ? static_cast<uint32_t>(inst.imm)
                : operand(inst.src1, lane);
            result[lane] = compare_setp(inst.opcode, operand(inst.src0, lane), rhs);
        }
        break;
    default: panic(warp, inst, "non-arithmetic instruction in RF read pipeline");
    }
    const auto result_cycle = pending.issue_cycle + scheduled_fixed_latency(inst, config_);
    if (is_setp(inst.opcode)) {
        w.new_writebacks.push_back(predicate_writeback(warp, inst.dst, result_cycle, std::move(result)));
    } else {
        w.new_writebacks.push_back(
            register_writeback(warp, inst.dst, result_cycle, std::move(result), opcode_name(inst.opcode)));
    }
}

// Arithmetic operand captures: every capture the plans scheduled for this
// cycle reads the RF now (after this cycle's writes landed, bypass (a)),
// including the offset-0 reads of the instruction issued this cycle. At a
// window's final cycle the instruction executes; its result is booked.
void Machine::step_arithmetic_reads(Wires& w) {
    for (auto it = arithmetic_reads_.begin(); it != arithmetic_reads_.end();) {
        for (std::size_t i = 0; i < it->reads.size(); ++i) {
            if (it->issue_cycle + it->reads[i].offset != cycle_) continue;
            auto& values = it->operands[i];
            values.resize(config_.lanes);
            for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
                values[lane] = rf(warps_[it->warp], lane, it->reads[i].reg);
            }
            trace_->rf_read(cycle_, it->warp, it->reads[i].reg, it->reads[i].reg % config_.rf_banks);
        }
        // The final window cycle: all operands are latched, execution begins.
        if (it->issue_cycle + config_.rf_arithmetic_read_cycles - 1 == cycle_) {
            execute_arithmetic(*it, w);
            it = arithmetic_reads_.erase(it);
        } else {
            ++it;
        }
    }
    // This cycle's ring slot is spent; cleared, it becomes the slot for
    // cycle + window, the farthest cycle the next issue may reserve.
    auto& expired = read_reservations_[cycle_ % read_reservations_.size()];
    std::fill(expired.begin(), expired.end(), 0);
}

// ===========================================================================
// Register drivers. Each computes its register's next value from Q and the
// wires, and writes it once (or not at all: hold). They read no other
// register's D and no state another driver writes, so they run in any order;
// set_driver_shuffle_seed() permutes them every cycle as a regression guard.
// Simultaneous events combine in one calculation (a release and a
// reallocation, a pop and a push, completions and an admission); the
// one-write rule is never a reason to defer an event. Inside a calculation
// the old owner's drains, clears and release come before the new owner's
// install and fills, so a same-cycle reuse (d) cannot erase the new state.
// ===========================================================================

void Machine::run_drivers(const Wires& w) {
    static constexpr std::array<Driver, 20> canonical = {
        &Machine::drive_depbar,
        &Machine::drive_wait_slots,
        &Machine::drive_port_a_access,
        &Machine::drive_mbar_read,
        &Machine::drive_rmw_engine,
        &Machine::drive_tma_event,
        &Machine::drive_mma_event,
        &Machine::drive_mma,
        &Machine::drive_acc_transfers,
        &Machine::drive_tma_queue,
        &Machine::drive_tma_transaction,
        &Machine::drive_tma_sector,
        &Machine::drive_tma_store_sector,
        &Machine::drive_lsu_commands,
        &Machine::drive_assembly_pool,
        &Machine::drive_ingress,
        &Machine::drive_egress,
        &Machine::drive_read_records,
        &Machine::drive_free_id_tags,
        &Machine::drive_writebacks,
    };
    auto order = canonical;
    if (shuffle_.has_value()) std::shuffle(order.begin(), order.end(), *shuffle_);
    for (const auto driver : order) (this->*driver)(w);
}

// depbar: next = Q - this cycle's completions + issue's admission. (The
// completions were checked for underflow when recorded.)
void Machine::drive_depbar(const Wires& w) {
    if (w.depbar_completions.empty() && !w.issue.depbar_admission.has_value()) return;
    auto next = depbar_.cur();
    for (const auto& completion : w.depbar_completions) --next.at(completion.warp).at(completion.bar);
    if (const auto& admission = w.issue.depbar_admission; admission.has_value()) {
        ++next.at(admission->warp).at(admission->bar);
    }
    depbar_.set_next(std::move(next), "drive_depbar");
}

// mbar wait slots: completed slots free; a try-wait that found its phase
// incomplete parks; issue may take a slot freed this cycle (d).
void Machine::drive_wait_slots(const Wires& w) {
    if (w.completed_waits.empty() && !w.port_a.park.has_value() && !w.issue.wait_slot.has_value()) return;
    auto next = mbar_wait_slots_.cur();
    for (const auto slot : w.completed_waits) next.at(slot).reset();
    if (const auto& park = w.port_a.park; park.has_value()) {
        auto& slot = next.at(park->first);
        slot->parked = true;
        slot->deadline_cycle = park->second;
    }
    if (const auto& taken = w.issue.wait_slot; taken.has_value()) {
        auto& slot = next.at(taken->first);
        if (slot.has_value()) throw SimError("internal mbar wait slot allocated twice");
        slot = taken->second;
    }
    mbar_wait_slots_.set_next(std::move(next), "drive_wait_slots");
}

// The SIMT port-A access: its next round, or a new access issue accepted
// (only when this one ends now).
void Machine::drive_port_a_access(const Wires& w) {
    const bool served = w.port_a.grant.user == PortAGrant::User::simt_access;
    if (w.issue.port_a.has_value()) {
        if (w.port_a.simt_after.has_value()) throw SimError("internal SIMT port-A access accepted over another");
        port_a_.set_next(w.issue.port_a, "issue: new SIMT access");
    } else if (served) {
        port_a_.set_next(w.port_a.simt_after, "port A: SIMT round");
    }
}

void Machine::drive_mbar_read(const Wires& w) {
    const bool served = w.port_a.grant.user == PortAGrant::User::mbar_read;
    if (w.issue.mbar_read.has_value()) {
        if (mbar_read_.cur().has_value() && !served) throw SimError("internal mbar read accepted over another");
        mbar_read_.set_next(w.issue.mbar_read, "issue: new mbar read");
    } else if (served) {
        mbar_read_.set_next(std::nullopt, "port A: mbar read served");
    }
}

// The RMW engine: its stage advanced (or a claimed engine event, now past
// its read stage), the round-robin bit at an engine claim, a SIMT RMW issue
// accepted for the next cycle.
void Machine::drive_rmw_engine(const Wires& w) {
    using User = PortAGrant::User;
    const auto user = w.port_a.grant.user;
    const bool engine_ran = user == User::rmw || user == User::engine_event;
    if (!engine_ran && !w.issue.rmw.has_value()) return;
    auto next = rmw_.cur();
    if (engine_ran) next.active = w.port_a.rmw_after;
    if (user == User::engine_event) next.last_served = w.port_a.grant.engine;
    if (w.issue.rmw.has_value()) {
        if (next.active.has_value()) throw SimError("internal SIMT RMW accepted over a busy RMW engine");
        next.active = w.issue.rmw;
    }
    rmw_.set_next(std::move(next), "drive_rmw_engine");
}

// The TMA's event register: emptied by the RMW engine's claim, filled by the
// TMA's presentation (never both in one cycle: the TMA presents only into an
// empty register).
void Machine::drive_tma_event(const Wires& w) {
    if (!w.port_a.tma_event_claimed && !w.tma.present_event.has_value()) return;
    auto next = w.port_a.tma_event_claimed ? std::nullopt : tma_mbar_event_.cur();
    if (w.tma.present_event.has_value()) {
        if (next.has_value()) throw SimError("internal TMA event presented into a full register");
        next = w.tma.present_event;
    }
    tma_mbar_event_.set_next(std::move(next), "drive_tma_event");
}

// The MMA's event register: a claim and a new commit's presentation may
// happen in one cycle (d).
void Machine::drive_mma_event(const Wires& w) {
    if (!w.port_a.mma_event_claimed && !w.mma_present.has_value()) return;
    auto next = w.port_a.mma_event_claimed ? std::nullopt : mma_mbar_event_.cur();
    if (w.mma_present.has_value()) {
        if (next.has_value()) throw SimError("internal MMA event presented into a full register");
        next = w.mma_present;
    }
    mma_mbar_event_.set_next(std::move(next), "drive_mma_event");
}

// The MMA queue and pipeline: step_mma's result, plus issue's new command.
void Machine::drive_mma(const Wires& w) {
    auto next = w.mma_next;
    if (w.issue.mma.has_value()) next.queue.push_back(*w.issue.mma);
    mma_.set_next(std::move(next), "drive_mma");
}

// The acc transfer slots: step_acc_transfers' result, plus issue's new
// transfer (into a slot freed this cycle, possibly).
void Machine::drive_acc_transfers(const Wires& w) {
    auto next = w.acc_next;
    if (const auto& admission = w.issue.acc; admission.has_value()) {
        auto& slot = next.at(admission->bank)[admission->load ? 0 : 1];
        if (slot.has_value()) throw SimError("internal acc transfer slot allocated twice");
        slot = admission->transfer;
    }
    acc_transfers_.set_next(std::move(next), "drive_acc_transfers");
}

// The TMA command queue: step_tma's result, plus issue's new command.
void Machine::drive_tma_queue(const Wires& w) {
    auto next = w.tma_next;
    if (w.issue.tma.has_value()) next.commands.emplace(w.issue.tma->id, *w.issue.tma);
    tma_.set_next(std::move(next), "drive_tma_queue");
}

// The TMA's port-A transaction: after this cycle's round (none when it ended).
void Machine::drive_tma_transaction(const Wires& w) {
    using User = PortAGrant::User;
    const auto user = w.port_a.grant.user;
    if (user != User::tma_transaction && user != User::tma_start) return;
    tma_port_a_.set_next(w.port_a.tma_after, "port A: TMA round");
}

// The TMA sector register: freed by its placement's last round, refilled by
// the ingress FIFO (the pluck's copy already starts from the freed register,
// so a same-cycle refill is one calculation (d)).
void Machine::drive_tma_sector(const Wires& w) {
    const auto& completed = w.port_a.tma_completed;
    const bool placed = completed.has_value() && completed->kind == TmaPortKind::load_write &&
        completed->from_sector_register;
    if (w.ingress.tma_sector_after.has_value()) {
        tma_sector_.set_next(*w.ingress.tma_sector_after, "ingress: TMA sector fill");
    } else if (placed) {
        auto next = tma_sector_.cur();
        next.request.reset();
        next.filled = 0;
        next.complete = false;
        tma_sector_.set_next(std::move(next), "port A: TMA sector placed");
    }
}

// The TMA store sector register: words drain into the egress FIFO; once
// empty the walker may allocate it for the next sector in the same cycle (d);
// port A's source-read rounds fill it; the last round completes it.
void Machine::drive_tma_store_sector(const Wires& w) {
    const uint32_t words_out = w.egress.client == GmemClient::tma ? static_cast<uint32_t>(w.egress.words.size()) : 0;
    const auto& completed = w.port_a.tma_completed;
    const bool captured = completed.has_value() && completed->kind == TmaPortKind::store_read;
    if (words_out == 0 && !w.tma_store_allocated && w.port_a.store_bytes.empty() && !captured) return;
    auto next = tma_store_sector_.cur();
    next.words_out += words_out;
    if (w.egress.tma_store_drained) {
        next.busy = false;
        next.complete = false;
    }
    if (w.tma_store_allocated) {
        const auto& plan = *w.tma.walker;
        next.busy = true;
        next.command = *w.tma.frontend;
        next.sector_address = plan.sector_address;
        std::fill(next.data.begin(), next.data.end(), 0);
        next.byte_enable = plan.store_byte_enable;
        next.complete = false;
        next.words_out = 0;
    }
    for (const auto& [offset, byte] : w.port_a.store_bytes) next.data.at(offset) = byte;
    if (captured) next.complete = true;
    tma_store_sector_.set_next(std::move(next), "drive_tma_store_sector");
}

// The LSU command slots (admission order): every stage's progress, the
// releases, and issue's admission, in one calculation. A command released
// now frees its slot for an admission now (d).
void Machine::drive_lsu_commands(const Wires& w) {
    const bool changed = w.lsu_drain.command.has_value() || w.egress.client == GmemClient::lsu ||
        w.channel.granted == GmemClient::lsu || !w.controller.lsu_acks.empty() ||
        w.lsu_rf_read.command.has_value() || !w.released_lsu_commands.empty() || w.issue.lsu.has_value();
    if (!changed) return;
    auto next = lsu_commands_.cur();
    const auto command = [&](uint64_t id) -> LsuCommand& {
        for (auto& candidate : next) {
            if (candidate.id == id) return candidate;
        }
        throw SimError("internal reference to a retired LSU command #" + std::to_string(id));
    };
    if (w.lsu_drain.command.has_value()) ++command(*w.lsu_drain.command).chunks_done;
    if (w.egress.client == GmemClient::lsu) {
        auto& store = command(w.egress.lsu_command);
        if (w.egress.lsu_chunk_drained) {
            store.words_plucked = 0;
            ++store.chunks_plucked;
        } else {
            store.words_plucked += static_cast<uint32_t>(w.egress.words.size());
        }
    }
    if (w.channel.granted == GmemClient::lsu) ++command(w.channel.lsu_command).requests_emitted;
    for (const auto& [id, acks] : w.controller.lsu_acks) command(id).acks += acks;
    std::erase_if(next, [&](const LsuCommand& candidate) {
        return std::find(w.released_lsu_commands.begin(), w.released_lsu_commands.end(), candidate.id) !=
            w.released_lsu_commands.end();
    });
    if (w.issue.lsu.has_value()) next.push_back(w.issue.lsu->command);
    if (w.lsu_rf_read.command.has_value()) ++command(*w.lsu_rf_read.command).chunks_done;
    lsu_commands_.set_next(std::move(next), "drive_lsu_commands");
}

// Each assembly register of the pool, in one calculation, in three steps:
//   1. the current owner empties it: a chunk drained to the RF (load) or into
//      the egress FIFO (vector store), and the owner's release;
//   2. issue's admission installs a new owner (with a scalar store's data);
//   3. this cycle's fills: the ingress pluck, the store RF read.
// The old owner's steps come first, so a register reused in its old owner's
// final cycle (d) keeps everything the new owner put in. A fill is the next
// value; it is read from the next cycle on.
void Machine::drive_assembly_pool(const Wires& w) {
    for (uint32_t index = 0; index < staging_.size(); ++index) {
        const auto& cur = staging_[index].cur();
        std::optional<AssemblyRegister> next;
        const auto edit = [&]() -> AssemblyRegister& {
            if (!next.has_value()) next = cur;
            return *next;
        };
        const auto clear = [&] {
            auto& reg = edit();
            std::fill(reg.valid.begin(), reg.valid.end(), 0);
            reg.valid_count = 0;
        };

        // 1. The current owner empties the register.
        const bool rf_drained = w.lsu_drain.command.has_value() && w.lsu_drain.assembly == index;
        const bool egress_drained = w.egress.lsu_chunk_drained && w.egress.lsu_assembly == index;
        const bool released = cur.owner.has_value() &&
            std::find(w.released_lsu_commands.begin(), w.released_lsu_commands.end(), *cur.owner) !=
                w.released_lsu_commands.end();
        if (rf_drained || egress_drained) clear();
        if (released) {
            edit().owner.reset();
            clear();
        }

        // 2. Issue's admission installs the new owner.
        if (w.issue.lsu.has_value() && w.issue.lsu->command.assembly == index) {
            if (edit().owner.has_value()) throw SimError("internal LSU admission into an owned assembly register");
            next->owner = w.issue.lsu->command.id;
            clear();
            if (const auto& data = w.issue.lsu->store_data; data.has_value()) {
                next->words = *data;
                std::fill(next->valid.begin(), next->valid.end(), 1);
                next->valid_count = config_.lanes;
            }
        }

        // 3. This cycle's fills.
        for (const auto& write : w.ingress.lanes) {
            if (write.assembly != index) continue;
            auto& reg = edit();
            reg.words[write.lane] = write.word;
            if (reg.valid[write.lane] == 0) {
                reg.valid[write.lane] = 1;
                ++reg.valid_count;
            }
        }
        if (w.lsu_rf_read.command.has_value() && w.lsu_rf_read.assembly == index) {
            auto& reg = edit();
            reg.words = w.lsu_rf_read.values;
            std::fill(reg.valid.begin(), reg.valid.end(), 1);
            reg.valid_count = config_.lanes;
        }
        if (next.has_value()) staging_[index].set_next(std::move(*next), "drive_assembly_pool");
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
    read_records_.set_next(std::move(next), "drive_read_records");
}

// Free-ID tags: returned when a gather's sector is fully read or a scatter's
// write is acknowledged; then taken by a scalar request now (a tag free in
// Q, so today never one returned now).
void Machine::drive_free_id_tags(const Wires& w) {
    if (!w.channel.lsu_tag.has_value() && w.ingress.freed_tags.empty() && w.controller.freed_tags.empty()) {
        return;
    }
    auto next = free_id_tags_.cur();
    for (const auto tag : w.ingress.freed_tags) next.at(tag) = 0;
    for (const auto tag : w.controller.freed_tags) next.at(tag) = 0;
    if (w.channel.lsu_tag.has_value()) next.at(*w.channel.lsu_tag) = 1;
    free_id_tags_.set_next(std::move(next), "drive_free_id_tags");
}

// The RF write schedule: the writes executed now leave it; this cycle's new
// bookings join it in wire order.
void Machine::drive_writebacks(const Wires& w) {
    const auto& cur = writebacks_.cur();
    const bool executed = std::any_of(cur.begin(), cur.end(),
        [&](const Writeback& writeback) { return writeback.ready_cycle <= cycle_; });
    if (!executed && w.new_writebacks.empty()) return;
    std::deque<Writeback> next;
    for (const auto& writeback : cur) {
        if (writeback.ready_cycle > cycle_) next.push_back(writeback);
    }
    for (const auto& writeback : w.new_writebacks) next.push_back(writeback);
    writebacks_.set_next(std::move(next), "drive_writebacks");
}

// ===========================================================================
// Helpers
// ===========================================================================

Instruction Machine::instruction(const Warp& warp) const {
    if (warp.pc % 8 != 0) throw SimError("unaligned PC in warp " + std::to_string(warp.id));
    const auto index = warp.pc / 8;
    if (index >= kernel_.words.size()) {
        throw SimError(
            "warp " + std::to_string(warp.id) + " fell off kernel at pc " +
            std::to_string(warp.pc));
    }
    return decode(kernel_.words[index]);
}

std::vector<uint8_t> Machine::source_registers(const Instruction& inst) const {
    switch (inst.opcode) {
    case Opcode::add:
    case Opcode::fadd:
    case Opcode::fmul:
    case Opcode::fsub:
    case Opcode::fmax:
    case Opcode::cvt_pack:
    case Opcode::setp_eq:
    case Opcode::setp_ne:
    case Opcode::setp_lt:
    case Opcode::setp_le:
    case Opcode::setp_gt:
    case Opcode::setp_ge: return {inst.src0, inst.src1};
    case Opcode::fma: return {inst.src0, inst.src1, inst.src2};
    case Opcode::and_:
    case Opcode::or_:
    case Opcode::xor_:
    case Opcode::shl:
    case Opcode::shr: return {inst.src0, inst.src1};
    case Opcode::addi:
    case Opcode::shli:
    case Opcode::shri:
    case Opcode::xori:
    case Opcode::andi:
    case Opcode::ori:
    case Opcode::lds:
    case Opcode::ldg:
    case Opcode::ldg_v8:
    case Opcode::ldg_uni:
    case Opcode::ex2:
    case Opcode::rcp:
    case Opcode::setpi_eq:
    case Opcode::setpi_ne:
    case Opcode::setpi_lt:
    case Opcode::setpi_le:
    case Opcode::setpi_gt:
    case Opcode::setpi_ge: return {inst.src0};
    case Opcode::mbar_init_i:
    case Opcode::mbar_arrive:
    case Opcode::mbar_arrive_expect_tx_i:
    case Opcode::mbar_test_wait_i:
    case Opcode::mbar_try_wait_i: return {inst.src0};
    case Opcode::mbar_init_r:
    case Opcode::mbar_arrive_expect_tx_r:
    case Opcode::mbar_test_wait_r:
    case Opcode::mbar_try_wait_r: return {inst.src0, inst.src1};
    case Opcode::tma_async: return {
        inst.src0,
        static_cast<uint8_t>(inst.src0 + 1),
        static_cast<uint8_t>(inst.src0 + 2),
    };
    case Opcode::sts:
    case Opcode::stg: return {inst.src0, inst.src1};
    case Opcode::stg_v8: {
        std::vector<uint8_t> result{inst.src0};
        for (uint16_t j = 0; j < 8; ++j) result.push_back(static_cast<uint8_t>(inst.src1 + j));
        return result;
    }
    case Opcode::acc_ld:
    case Opcode::acc_ld_v8: return {inst.src1, static_cast<uint8_t>(inst.src1 + 1)};
    case Opcode::acc_st: return {inst.src0, static_cast<uint8_t>(inst.src0 + 1), inst.src1};
    case Opcode::acc_st_v8: {
        std::vector<uint8_t> result{inst.src0, static_cast<uint8_t>(inst.src0 + 1)};
        for (uint16_t j = 0; j < 8; ++j) result.push_back(static_cast<uint8_t>(inst.src1 + j));
        return result;
    }
    case Opcode::mma_async:
    case Opcode::mma_async_init: return {inst.src0, inst.src1};
    case Opcode::mma_commit: return {inst.src0};
    default: return {};
    }
}

std::optional<uint8_t> Machine::destination_register(const Instruction& inst) const {
    switch (inst.opcode) {
    case Opcode::sread:
    case Opcode::mov:
    case Opcode::add:
    case Opcode::addi:
    case Opcode::and_:
    case Opcode::andi:
    case Opcode::or_:
    case Opcode::ori:
    case Opcode::xor_:
    case Opcode::shl:
    case Opcode::shr:
    case Opcode::shli:
    case Opcode::shri:
    case Opcode::xori:
    case Opcode::fadd:
    case Opcode::fmul:
    case Opcode::fsub:
    case Opcode::fmax:
    case Opcode::fma:
    case Opcode::cvt_pack:
    case Opcode::ex2:
    case Opcode::rcp:
    case Opcode::lds:
    case Opcode::ldg:
    case Opcode::ldg_v8:
    case Opcode::ldg_uni: return inst.dst;
    case Opcode::acc_ld:
    case Opcode::acc_ld_v8: return inst.dst;
    default: return std::nullopt;
    }
}

// No scoreboard: a hazard is a PANIC, diagnosed by the sanitizer. Only the
// destination range check is a property of the instruction itself.
void Machine::check_register_hazards(const Warp& warp, const Instruction& inst) const {
    sanitizer_.check_rf_raw(site(warp, inst), source_registers(inst));
    if (const auto dst = destination_register(inst); dst.has_value()) {
        const auto width = inst.opcode == Opcode::acc_ld_v8 || inst.opcode == Opcode::ldg_v8 ? 8u : 1u;
        if (static_cast<unsigned>(*dst) + width > kernel_.register_count)
            panic(warp, inst, "RF destination range out of bounds");
        sanitizer_.check_rf_waw_war(site(warp, inst), *dst, width);
    }
}

uint32_t Machine::uniform_reg(
    const Warp& warp, const Instruction& inst, uint8_t reg, const std::string& name) const {
    const auto value = rf(warp, 0, reg);
    for (uint32_t lane = 1; lane < config_.lanes; ++lane) {
        if (rf(warp, lane, reg) != value) {
            panic(warp, inst, name + " is not warp-uniform");
        }
    }
    return value;
}

MbarState Machine::load_mbar(uint32_t byte_address) const {
    if (byte_address % 8 != 0 || static_cast<uint64_t>(byte_address) + 8 > shmem_.size()) {
        throw SimError("invalid 64-bit mbar shmem access");
    }
    const auto low = static_cast<uint64_t>(read_shmem_u32(byte_address));
    const auto high = static_cast<uint64_t>(read_shmem_u32(byte_address + 4));
    return decode_mbar_state(low | (high << 32));
}

uint32_t& Machine::rf(Warp& warp, uint32_t lane, uint32_t reg) {
    return warp.rf.at(static_cast<std::size_t>(lane) * kernel_.register_count + reg);
}

uint32_t Machine::rf(const Warp& warp, uint32_t lane, uint32_t reg) const {
    return warp.rf.at(static_cast<std::size_t>(lane) * kernel_.register_count + reg);
}

std::vector<std::vector<Machine::LaneAccess>> Machine::build_shmem_rounds(
    const Warp& warp, const Instruction& inst, bool load, const Wires& w) const {
    struct Group {
        uint32_t address = 0;
        std::vector<LaneAccess> accesses;
    };
    std::vector<std::vector<Group>> groups(config_.shmem_banks);
    for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
        const auto base = static_cast<int64_t>(rf(warp, lane, inst.src0));
        const auto address = base + inst.imm;
        if (address < 0 || address > std::numeric_limits<uint32_t>::max()) {
            panic(warp, inst, "shmem address arithmetic overflow");
        }
        const auto byte_address = static_cast<uint32_t>(address);
        if (byte_address % 4 != 0) panic(warp, inst, "unaligned 32-bit shmem access");
        if (static_cast<uint64_t>(byte_address) + 4 > shmem_.size()) {
            panic(warp, inst, "shmem access out of bounds");
        }
        if (!load) {
            const auto& slots = mbar_wait_slots_.cur();
            for (uint32_t index = 0; index < slots.size(); ++index) {
                const auto& wait_slot = slots[index];
                if (wait_slot.has_value() && !wait_slot_completed(w, index) &&
                    byte_address >= wait_slot->byte_address &&
                    byte_address < wait_slot->byte_address + 8) {
                    panic(warp, inst, "repurposing an mbar with a live waiter");
                }
            }
        }
        const auto bank = (byte_address / 4) % config_.shmem_banks;
        auto& bank_groups = groups[bank];
        auto found = std::find_if(bank_groups.begin(), bank_groups.end(), [&](const Group& group) {
            return group.address == byte_address;
        });
        if (found != bank_groups.end()) {
            if (!load) panic(warp, inst, "multiple lanes write the same shmem word");
            found->accesses.push_back({lane, 0, byte_address, 0});
            continue;
        }
        Group group;
        group.address = byte_address;
        group.accesses.push_back({lane, 0, byte_address, load ? 0 : rf(warp, lane, inst.src1)});
        bank_groups.push_back(std::move(group));
    }

    std::size_t round_count = 0;
    for (auto& bank_groups : groups) {
        std::sort(bank_groups.begin(), bank_groups.end(), [](const Group& a, const Group& b) {
            return a.address < b.address;
        });
        round_count = std::max(round_count, bank_groups.size());
    }
    if (round_count == 0) panic(warp, inst, "empty shmem request");
    std::vector<std::vector<LaneAccess>> rounds(round_count);
    for (const auto& bank_groups : groups) {
        for (std::size_t round = 0; round < bank_groups.size(); ++round) {
            rounds[round].insert(
                rounds[round].end(), bank_groups[round].accesses.begin(), bank_groups[round].accesses.end());
        }
    }
    return rounds;
}

Machine::Writeback Machine::register_writeback(
    Warp& warp,
    uint8_t dst,
    uint64_t ready_cycle,
    std::vector<uint32_t> values,
    std::string reason) {
    sanitizer_.record_reg_ready(warp.id, dst, ready_cycle);
    Writeback writeback;
    writeback.ready_cycle = ready_cycle;
    writeback.warp = warp.id;
    writeback.dst = dst;
    writeback.values = std::move(values);
    writeback.reason = std::move(reason);
    return writeback;
}

Machine::Writeback Machine::predicate_writeback(
    Warp& warp,
    uint8_t dst,
    uint64_t ready_cycle,
    std::vector<uint32_t> values) {
    sanitizer_.record_pred_ready(warp.id, dst, ready_cycle);
    Writeback writeback;
    writeback.ready_cycle = ready_cycle;
    writeback.warp = warp.id;
    writeback.predicate_dst = dst;
    writeback.values = std::move(values);
    writeback.reason = "setp";
    return writeback;
}

void Machine::apply_image(
    std::vector<uint8_t>& memory,
    const std::vector<ImageSegment>& image,
    const std::string& name) {
    for (const auto& segment : image) {
        if (segment.address > memory.size() || segment.bytes.size() > memory.size() - segment.address) {
            throw SimError("launch " + name + " image segment does not fit");
        }
        std::copy(
            segment.bytes.begin(),
            segment.bytes.end(),
            memory.begin() + static_cast<std::size_t>(segment.address));
    }
}

// ---------------------------------------------------------------------------
// Sanitizer argument builders. Pure reads of hardware state; their results go
// only to sanitizer_ (src/sanitizer.hpp). Nothing here feeds the machine.
// ---------------------------------------------------------------------------

IssueSite Machine::site(const Warp& warp, const Instruction& inst) const {
    return IssueSite{cycle_, warp.id, warp.pc, inst};
}

std::vector<uint32_t> Machine::access_addresses(
    const std::vector<std::vector<LaneAccess>>& rounds) {
    std::vector<uint32_t> addresses;
    for (const auto& round : rounds) {
        for (const auto& access : round) addresses.push_back(static_cast<uint32_t>(access.byte_address));
    }
    return addresses;
}

std::optional<std::pair<uint32_t, uint32_t>> Machine::mma_operand_span(
    uint32_t base, bool a_operand) const {
    // Ask the operand geometry itself (the same functions the fetch uses):
    // the hull of every element address of the profile-static shape.
    uint64_t first = std::numeric_limits<uint64_t>::max();
    uint64_t last = 0;
    const uint32_t outer = a_operand ? config_.mma_m : config_.mma_n;
    for (uint32_t i = 0; i < outer; ++i) {
        for (uint32_t k = 0; k < config_.mma_k; ++k) {
            const uint64_t address = a_operand
                ? mma_a_address(config_, base, i, k)
                : mma_b_address(config_, base, k, i);
            first = std::min(first, address);
            last = std::max(last, address + 2);
        }
    }
    if (last > std::numeric_limits<uint32_t>::max()) return std::nullopt;
    return std::pair<uint32_t, uint32_t>{static_cast<uint32_t>(first), static_cast<uint32_t>(last)};
}

std::array<uint8_t, tma_descriptor_bytes> Machine::pending_tma_descriptor_bytes(
    uint32_t address, const Wires& w) const {
    // The descriptor the engine will read later. The engine's descriptor read
    // is a port-A request, so port-A writes already accepted are ordered
    // before it (one path): this cycle's port-A writes (they land at the
    // clock edge), then a SIMT sts's rounds still to come. Writes accepted
    // after this point would conflict with the descriptor's read-ownership
    // and panic.
    std::array<uint8_t, tma_descriptor_bytes> bytes{};
    std::memcpy(bytes.data(), shmem_.data() + address, bytes.size());
    const auto overlay = [&](uint64_t at, const uint8_t* data, std::size_t size) {
        for (std::size_t byte = 0; byte < size; ++byte) {
            if (at + byte >= address && at + byte < address + tma_descriptor_bytes) {
                bytes[at + byte - address] = data[byte];
            }
        }
    };
    for (const auto& write : w.port_a.writes) overlay(write.address, write.bytes.data(), write.bytes.size());
    if (const auto& access = w.port_a.simt_after; access.has_value() && !access->load) {
        for (std::size_t round = access->next_round; round < access->rounds.size(); ++round) {
            for (const auto& lane : access->rounds[round]) {
                overlay(lane.byte_address, reinterpret_cast<const uint8_t*>(&lane.value), sizeof(lane.value));
            }
        }
    }
    return bytes;
}

[[noreturn]] void Machine::panic(
    const Warp& warp, const Instruction& inst, const std::string& reason) const {
    throw_issue_panic(site(warp, inst), reason);
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

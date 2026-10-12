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

bool is_shared_load(Opcode opcode) {
    return opcode == Opcode::lds || opcode == Opcode::lds_uni;
}

bool is_lsu_opcode(Opcode opcode) {
    return opcode == Opcode::ldg || opcode == Opcode::stg || opcode == Opcode::ldg_v8 ||
        opcode == Opcode::stg_v8 || opcode == Opcode::ldg_uni;
}

bool is_mma_command(Opcode opcode) {
    return opcode == Opcode::mma_async || opcode == Opcode::mma_async_init || opcode == Opcode::mma_commit;
}

bool is_acc_load(Opcode opcode) {
    return opcode == Opcode::acc_ld || opcode == Opcode::acc_ld_v8;
}

bool is_acc_transfer(Opcode opcode) {
    return is_acc_load(opcode) || opcode == Opcode::acc_st || opcode == Opcode::acc_st_v8;
}

// The SIMT shmem users: accesses (load / store slots), mbar probes (the probe
// register) and mbar RMWs (the RMW engine).
bool uses_shmem(Opcode opcode) {
    return is_shared_load(opcode) || opcode == Opcode::sts || is_mbar_rmw(opcode) ||
        is_mbar_predicate_writer(opcode);
}

// trash/rfc12.md sections 3, 4 and 6: first the warp's own order -- a read
// waits for the warp's unserviced writes, a write (sts, an RMW) for its
// unserviced reads and writes; reads may pass reads -- then the resource that
// holds the operation. So a SIMT RMW enters the SIMT RMW register (and later
// the engine, whose store pause would block them) only once its warp's own
// lds / sts are serviced. An lds / sts with no active lane is absent: no
// slot, no order.
std::optional<Refusal> shmem_refusal(Opcode opcode, const ResourceView& view) {
    if ((opcode == Opcode::lds || opcode == Opcode::sts) && view.no_active_lanes) return std::nullopt;
    const bool write = opcode == Opcode::sts || is_mbar_rmw(opcode);
    if (view.unserviced_write || (write && view.unserviced_read)) return Refusal::shmem_ordered;
    if (is_shared_load(opcode) && !view.load_slot_free) return Refusal::load_slots_full;
    if (opcode == Opcode::sts && !view.store_slot_free) return Refusal::store_slots_full;
    if (is_mbar_predicate_writer(opcode) && !view.probe_register_free) return Refusal::probe_register_busy;
    if (is_mbar_rmw(opcode) && !view.simt_rmw_register_free) return Refusal::simt_rmw_register_busy;
    return std::nullopt;
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

// ===========================================================================
// Admission (trash/rfc10.md section 1): metadata and resource state only.
// These two functions see the instruction, its plan and a ResourceView. They
// are the only places issue decides whether to accept; Machine::try_issue
// calls them before it captures any operand.
// ===========================================================================

std::optional<uint8_t> depbar_wait(const Instruction& inst, const std::array<uint32_t, 6>& depbar) {
    for (uint8_t bar = 0; bar < 6; ++bar) {
        if ((inst.control.wait_mask & (1u << bar)) != 0 && depbar[bar] != 0) return bar;
    }
    return std::nullopt;
}

// In a fixed order, which decides the reason reported when several apply.
// A set depbar needs counter room too; no counter width is declared (S5),
// and a warp can hold far fewer async operations than any width would
// count, so that never refuses.
std::optional<Refusal> structural_refusal(const Instruction& inst, const IssuePlan& plan, const ResourceView& view) {
    const auto opcode = inst.opcode;
    if (!view.reads_fit) return Refusal::rf_read_booked;
    if (uses_arithmetic_pipeline(inst)) {
        if (is_sfu(opcode) && !view.sfu_free) return Refusal::sfu_busy;
        if (plan.write_reg.has_value() && !view.write_slot_free) return Refusal::rf_write_booked;
        if (plan.write_reg.has_value() && view.write_inhibited) return Refusal::rf_write_inhibited;
        return std::nullopt;
    }
    if (is_mbar_try_wait(opcode) && !view.wait_slot_free) return Refusal::wait_slots_full;
    if (is_tma(opcode) && !view.tma_queue_room) return Refusal::tma_queue_full;
    if (is_mma_command(opcode) && !view.mma_queue_room) return Refusal::mma_queue_full;
    if (is_acc_transfer(opcode) && !view.acc_slot_free) return Refusal::tmem_bank_busy;
    if (opcode == Opcode::fence_sh && !view.own_shmem_at_ram) return Refusal::shmem_fence;
    if (uses_shmem(opcode)) {
        if (const auto refusal = shmem_refusal(opcode, view); refusal.has_value()) return refusal;
    }
    if (is_lsu_opcode(opcode) && !view.lsu_slot_free) return Refusal::lsu_slot;
    if (is_lsu_opcode(opcode) && !view.assembly_free) return Refusal::lsu_assembly;
    return std::nullopt;
}

RoundRobinGrant round_robin_grant(const std::vector<uint8_t>& ready, uint32_t cursor, uint32_t ports) {
    RoundRobinGrant result;
    result.cursor_after = cursor;
    const auto holders = static_cast<uint32_t>(ready.size());
    for (uint32_t step = 0; step < holders && result.granted.size() < ports; ++step) {
        const auto holder = (cursor + step) % holders;
        if (ready[holder] == 0) continue;
        result.granted.push_back(holder);
        result.cursor_after = (holder + 1) % holders;
    }
    return result;
}

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
        acc_ld_pipes_.poke(AccLdPipelines(
            config_.tmem_banks, std::vector<std::optional<AccLdBeat>>(acc_ld_read_stages)));
        for (uint32_t index = 0; index < config_.shmem_load_slots; ++index) {
            load_slots_.emplace_back(registers_, "load slot " + std::to_string(index));
        }
        for (uint32_t index = 0; index < config_.shmem_store_slots; ++index) {
            store_slots_.emplace_back(registers_, "store slot " + std::to_string(index));
        }
        a_read_pipe_.poke(ReadPipeline(config_.shmem_port_a_depth + config_.shmem_read_latency_cycles));
        a_write_pipe_.poke(WritePipeline(config_.shmem_port_a_depth));
        b_write_pipe_.poke(WritePipeline(config_.shmem_port_a_depth));
        RmwEngine engine;
        engine.alu.resize(config_.shmem_rmw_compute_cycles);
        rmw_.poke(std::move(engine));
        RfWritePolicy policy;
        policy.owed.assign(config_.rf_banks, 0);
        policy.cursor.assign(config_.rf_banks, 0);
        rf_write_policy_.poke(std::move(policy));
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
        TmaStoreRegisters stores;
        stores.ring.resize(config_.tma_store_sector_registers);
        tma_store_registers_.poke(std::move(stores));
        warps_.reserve(launch.active_warps);
        for (uint32_t id = 0; id < launch.active_warps; ++id) {
            Warp warp;
            warp.id = id;
            warp.active_mask = full_mask();
            warp.rf.resize(static_cast<std::size_t>(config_.lanes) * kernel_.register_count, 0);
            for (auto& predicate : warp.predicates) predicate.resize(config_.lanes, 0);
            warps_.push_back(std::move(warp));
        }
        depbar_.poke(DepbarCounters(launch.active_warps));
        shmem_to_ram_.poke(std::vector<uint32_t>(launch.active_warps, 0));
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
//          read at t (booked writebacks, the async writes the RF write
//          decision grants at t, mbar test/try-wait results).
//      (b) Completions reach issue in their own cycle: depbar clears, mbar
//          wait slots completed (wake, immediate success, timeout).
//      (c) Issue looks ahead: an instruction accepted at t occupies its
//          resources from t+1, so issue sees the space this cycle frees
//          (queue entries, slots freed by this cycle's grants, the SIMT RMW
//          register emptied by an RMW start, assembly registers) and the
//          warp's shmem accesses as this cycle's grants leave them (rfc12
//          section 4).
//      (d) Same-cycle space reuse: a register drained at t may be refilled at
//          t (the assembly register, a load slot whose RF write is granted
//          at t, the TMA sector / store registers, the MMA event register,
//          an LSU slot or acc slot reallocated by issue). The refill's data
//          is visible at t+1; nothing passes through.
//      (e) An issued stg.v8 makes its first RF read beat in its issue cycle.
//      (f) Inside the MMA: a result's tmem write precedes the same cycle's
//          accumulator read (write-first).
//      (g) The placeholder controller may put a request accepted at t on the
//          bus at t (outside the SM; the timeline test pins request -> beat 0).
//      (h) A TMA shmem transaction is granted its first round in the cycle
//          it is requested: a declared combinational path from latched TMA
//          state (its queue and sector registers, Q; for a store sector also
//          this cycle's egress drain of a store register (d)) ->
//          tma_requests builds the request -> select_a_read / select_b_write
//          grants it -> round 0 enters its pipeline, all in one cycle. No
//          register holds the transaction before round 0. Likewise an RMW's
//          start (A-write's alternation lets it begin, the engine takes the
//          request from its register) and its read grant may share a cycle:
//          select_a_read sees the start (shmem wire order: A-write, RMW
//          admission, A-read). A SIMT access or RMW accepted at issue sits in
//          its slot / register and is served from the next cycle. A new MMA
//          fetch serves its first port-B round in the cycle it starts.
// 2. DRIVERS. Each register's next value, from Q and the wires. They run in
//    any order (set_driver_shuffle_seed() permutes them every cycle).
// 3. CLOCK EDGE. Every register takes D; shmem takes the write rounds that
//    reached the RAM this cycle (A-write, then port-B write).
// ===========================================================================
void Machine::tick() {
    if (done()) return;
    sanitizer_.record_cycle_start();
    rf_requester_ = RfRequester::acc_transfer;
    Wires w;

    // RF write port, write-first (a). The one RF write decision, from Q and
    // before any SIMT candidate: the ready async holders share this cycle's
    // unbooked write ports; banks whose service is owed inhibit new SIMT
    // bookings. Then everything due now lands.
    w.rf_write = decide_rf_writes();
    execute_rf_writes(w);

    // The store path's last stage frees a TMA store register / a store
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

    // The shared resources, one select each: the request channel; shmem's
    // A-read, A-write and port-B write (after the RAM stage: the write rounds
    // reaching the RAM and their completions, the read round at the port
    // output and its requester).
    w.channel = select_request_channel(w.tma, w.lsu_want);
    step_shmem(w);

    // Units whose next state issue looks ahead at (c). The MMA's commit may
    // present into the event register the RMW engine claimed just now (d).
    step_mma(w);
    step_acc_transfers(w);        // acc.ld advances on its RF grant; acc.st books its RF read before issue
    decide_mbar_timeouts(w);      // a wake this cycle wins over the deadline
    w.ingress = decide_ingress(w);
    step_gmem_controller(w);
    step_tma(w);

    // The scheduler: the last decision of the cycle. Its fixed RF-write
    // booking (if any) only updates the owed bits' next state.
    decide_issue(w);

    // Below issue in RF read priority.
    w.lsu_rf_read = decide_lsu_rf_read(w);
    step_arithmetic_reads(w);

    run_drivers(w);

    // Clock edge.
    registers_.commit();
    for (const auto& write : w.shmem.writes) {
        std::memcpy(shmem_.data() + write.address, write.bytes.data(), write.bytes.size());
    }
    ++cycle_;
    // Kernel end: the sanitizer's end-of-kernel checks (a nonzero mbar tx count).
    if (done()) sanitizer_.check_kernel_end(cycle_);
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
    const auto empty = [](const auto& reg) { return !reg.cur().has_value(); };
    const auto drained = [](const auto& pipe) {
        return std::all_of(pipe.begin(), pipe.end(), [](const auto& stage) { return !stage.has_value(); });
    };
    // rfc12 section 13: the slots in every state, the probe register, the RMW
    // engine (its SIMT RMW register, operation and ALU stages) and the
    // services' pipelines (a store whose slot is free may still be travelling
    // to its write edge).
    const bool shmem_idle = std::all_of(load_slots_.begin(), load_slots_.end(), empty) &&
        std::all_of(store_slots_.begin(), store_slots_.end(), empty) && empty(mbar_probe_) && empty(simt_rmw_) &&
        !rmw_.cur().active.has_value() && drained(rmw_.cur().alu) && drained(a_read_pipe_.cur()) &&
        drained(a_write_pipe_.cur()) && drained(b_write_pipe_.cur()) && empty(tma_read_txn_) &&
        empty(tma_write_txn_);
    const auto& pipes = acc_ld_pipes_.cur();
    const bool acc_ld_pipes_empty = std::all_of(pipes.begin(), pipes.end(), [](const auto& pipe) {
        return std::all_of(pipe.begin(), pipe.end(), [](const auto& stage) { return !stage.has_value(); });
    });
    return warps_halted && wait_slots_empty && shmem_idle && acc_ld_pipes_empty &&
        !tma_mbar_event_.cur().has_value() && !mma_mbar_event_.cur().has_value() && gmem_queue_.empty() &&
        !gmem_bus_.has_value() && pending_write_acks_.empty() && lsu_commands_.cur().empty() &&
        read_records_.cur().empty() && ingress_.cur().beats.empty() && egress_fifo_.cur().empty() &&
        !tma_sector_.cur().request.has_value() && tma_store_registers_.cur().held() == 0 &&
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
    // A DMA complete_tx: it belongs in the TMA's event register. It changes the
    // mbar's current phase; `parity` (the phase the host means) is sanitizer
    // metadata only (rfc12 section 8).
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

// Shmem-test-only (sim.hpp): nothing but the shmem path reads the mask.
void Machine::test_set_active_mask(uint32_t warp, uint64_t mask) {
    if (warp >= warps_.size() || (mask & ~full_mask()) != 0) throw SimError("invalid active mask");
    warps_[warp].active_mask = mask;
}

uint32_t Machine::read_reg(
    uint32_t warp, uint32_t lane, uint32_t reg) const {
    if (warp >= warps_.size() || lane >= config_.lanes || reg >= kernel_.register_count) {
        throw SimError("invalid host RF read");
    }
    return rf(warps_[warp], lane, reg);
}

// ===========================================================================
// The RF ports: the reservation calendars, the RF write decision, and the
// write port's execution
// ===========================================================================

// RF READS are a reservation schedule (unchanged in slice I1; S2 mirrors the
// write rule below). A booking made in an earlier cycle keeps its promise.
// Only requests made in the SAME cycle for the SAME bank and target cycle
// contend, and booking order decides them, in the order tick() makes them:
//   1. acc.st beats (tmem bank order)   read at t        step_acc_transfers
//   2. issue (all-or-nothing)            reads at t..     decide_issue
//   3. the LSU store read                read at t        decide_lsu_rf_read
// note_rf_requester() enforces that order.
//
// RF WRITES: the calendar holds fixed SIMT bookings only -- issue books the
// cycle an accepted instruction's result enters its bank's write port, always
// a future cycle (latency >= 1). Writers that cannot promise a cycle (load
// slots, LSU assemblies, acc.ld pipelines) never book: decide_rf_writes
// grants them this cycle's unbooked ports and decides which banks inhibit new
// SIMT bookings (trash/rfc10.md section 3).
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

bool Machine::write_slot_free(uint8_t reg, uint64_t at_cycle) const {
    const auto found = write_reservations_.find(at_cycle);
    return found == write_reservations_.end() ||
        found->second.at(reg % config_.rf_banks) < config_.rf_write_ports_per_bank;
}

void Machine::reserve_write(uint8_t reg, uint64_t at_cycle) {
    if (at_cycle <= cycle_) throw SimError("internal RF write booking for the current cycle");
    if (!write_slot_free(reg, at_cycle)) throw SimError("internal RF write booking over a full port");
    auto& slots = write_reservations_[at_cycle];
    if (slots.empty()) slots.resize(config_.rf_banks, 0);
    ++slots.at(reg % config_.rf_banks);
}

// calendar[0] of a bank: its writes booked (in earlier cycles) for this cycle.
uint32_t Machine::writes_booked_now(uint32_t bank) const {
    const auto found = write_reservations_.find(cycle_);
    return found == write_reservations_.end() ? 0 : found->second.at(bank);
}

bool Machine::RfWriteDecision::granted_to(AsyncWriter writer, uint32_t index) const {
    for (std::size_t holder = 0; holder < holders.size(); ++holder) {
        if (granted[holder] != 0 && holders[holder]->writer == writer && holders[holder]->index == index) return true;
    }
    return false;
}

// Assembly register `index` as an async RF holder: its write when it holds a
// complete load vector (in Q), else none.
std::optional<Machine::AsyncWrite> Machine::assembly_rf_write(uint32_t index) const {
    const auto& assembly = staging_.at(index).cur();
    const auto* command = assembly.owner.has_value() ? find_lsu_command(*assembly.owner) : nullptr;
    if (command == nullptr || is_store(command->kind) || assembly.valid_count != config_.lanes) return std::nullopt;
    const bool vector = command->kind == LsuKind::vector_load;
    const auto reg = static_cast<uint8_t>(command->reg + (vector ? command->chunks_done : 0));
    return AsyncWrite{AsyncWriter::assembly, index, command->warp, reg};
}

// The LSU's output mux: the first assembly register, from the LSU cursor, that
// holds a complete load vector. Only it reaches the RF write decision this
// cycle (one LSU write path); the others wait for the mux.
std::optional<uint32_t> Machine::lsu_rf_presented() const {
    const auto registers = static_cast<uint32_t>(staging_.size());
    const auto cursor = rf_write_policy_.cur().lsu_cursor;
    for (uint32_t step = 0; step < registers; ++step) {
        const auto index = (cursor + step) % registers;
        if (assembly_rf_write(index).has_value()) return index;
    }
    return std::nullopt;
}

// Every async RF holder, in round-robin order, with its write when its vector
// is complete in Q: the load slots, the assembly register on the LSU's write
// path, then each tmem bank's acc.ld pipeline output.
std::vector<std::optional<Machine::AsyncWrite>> Machine::ready_async_writers(
    std::optional<uint32_t> lsu_presented) const {
    std::vector<std::optional<AsyncWrite>> holders;
    for (uint32_t index = 0; index < load_slots_.size(); ++index) {
        const auto& slot = load_slots_[index].cur();
        if (slot.has_value() && slot->complete()) {
            holders.push_back(AsyncWrite{AsyncWriter::load_slot, index, slot->warp, slot->dst});
        } else {
            holders.emplace_back();
        }
    }
    for (uint32_t index = 0; index < staging_.size(); ++index) {
        holders.push_back(lsu_presented == index ? assembly_rf_write(index) : std::nullopt);
    }
    const auto& pipes = acc_ld_pipes_.cur();
    for (uint32_t bank = 0; bank < pipes.size(); ++bank) {
        const auto& output = pipes[bank].back();
        if (output.has_value()) {
            holders.push_back(AsyncWrite{AsyncWriter::acc_ld, bank, output->warp, output->reg});
        } else {
            holders.emplace_back();
        }
    }
    return holders;
}

// The RF write decision (trash/rfc10.md section 3, "make an opening, then
// choose a writer"): one per RF bank, from Q only, before issue.
//   async_ready  = a ready async holder targets the bank
//   async_grant  = async_ready AND one of this cycle's write ports is unbooked
//   inhibit_simt = async_ready AND NOT async_grant AND async_owed
// The holders granted (round-robin from the bank's cursor) write now; the
// owed bit and the cursor take their next values in drive_rf_write_policy.
// Old bookings always execute; issue's outcome cannot change these grants.
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
        const auto choice = round_robin_grant(ready, policy.cursor.at(bank), free_ports);
        for (const auto holder : choice.granted) decision.granted[holder] = 1;
        auto& state = decision.banks[bank];
        state.ready = std::find(ready.begin(), ready.end(), 1) != ready.end();
        state.granted = !choice.granted.empty();
        state.inhibit_simt = state.ready && !state.granted && policy.owed.at(bank) != 0;
        state.cursor_after = choice.cursor_after;
    }
    return decision;
}

// The payload and completion context a granted async holder hands to the RF
// write path. With no RF write latency the write lands now (write-first).
Machine::Writeback Machine::async_writeback(const AsyncWrite& write) {
    Writeback writeback;
    writeback.ready_cycle = cycle_;
    writeback.warp = write.warp;
    writeback.dst = write.reg;
    trace_->rf_async_write();
    switch (write.writer) {
    case AsyncWriter::load_slot: {
        const auto& slot = *load_slots_.at(write.index).cur();
        writeback.values = slot.words;
        writeback.lane_enables = slot.mask;
        writeback.clear_depbar = slot.depbar;
        writeback.reason = "lds";
        sanitizer_.record_reg_ready(write.warp, write.reg, cycle_);
        return writeback;
    }
    case AsyncWriter::assembly: {
        const auto& assembly = staging_.at(write.index).cur();
        const auto& command = *find_lsu_command(*assembly.owner);
        sanitizer_.check_assembly_lanes_valid(cycle_, write.index, assembly.valid, 0, config_.lanes, "RF drain");
        writeback.values = assembly.words;
        writeback.reason = opcode_name(command.instruction.opcode);
        const bool vector = command.kind == LsuKind::vector_load;
        if (!vector || command.chunks_done + 1 == 8) {
            // rD..rD+7 count as in flight until the last write.
            const uint32_t width = vector ? 8 : 1;
            for (uint32_t j = 0; j < width; ++j) sanitizer_.record_reg_ready(command.warp, command.reg + j, cycle_);
            writeback.clear_depbar = command.depbar;
            writeback.release_lsu_command = command.id;
        }
        probe(LsuEdge::rf_write_beat, command.id, write.reg, config_.lanes * 4);
        return writeback;
    }
    case AsyncWriter::acc_ld: {
        const auto& beat = *acc_ld_pipes_.cur().at(write.index).back();
        writeback.values = beat.values;
        writeback.reason = "acc.ld";
        sanitizer_.record_reg_ready(write.warp, write.reg, cycle_);
        // (Its tmem read-ownership ended at the transfer's last tmem read,
        // launch_acc_ld_beat; the destination stays not-ready until this write.)
        if (beat.last) writeback.clear_depbar = beat.depbar;
        return writeback;
    }
    }
    throw SimError("internal unknown async RF writer");
}

// (T) Ready holders not written this cycle, including assembly registers the
// LSU's output mux did not present.
void Machine::count_rf_write_waits(const RfWriteDecision& decision) {
    for (std::size_t holder = 0; holder < decision.holders.size(); ++holder) {
        const auto& write = decision.holders[holder];
        if (!write.has_value() || write->writer == AsyncWriter::assembly || decision.granted[holder] != 0) continue;
        trace_->rf_async_wait();
    }
    for (uint32_t index = 0; index < staging_.size(); ++index) {
        if (!assembly_rf_write(index).has_value() || decision.granted_to(AsyncWriter::assembly, index)) continue;
        trace_->rf_async_wait();
        ++trace_->lsu_counters().lsu_rf_write_wait_cycles;
        ++lsu_phases_[*staging_.at(index).cur().owner].rf_wait;
    }
}

// The RF write port at this cycle, write-first (bypass (a)): every booked
// writeback due now lands, then the granted async writes (never on a booked
// port), before anything reads the RF or the predicates. A write that ends an
// async operation clears its depbar (seen by this cycle's issue, bypass (b))
// and may end its LSU command, whose slot and assembly register issue may
// reallocate now (d); a load slot granted now is free for issue now too.
void Machine::execute_rf_writes(Wires& w) {
    std::vector<uint32_t> port_uses(config_.rf_banks, 0);
    const auto execute = [&](const Writeback& writeback) {
        auto& warp = warps_.at(writeback.warp);
        if (writeback.dst.has_value()) {
            if (writeback.values.size() != config_.lanes) {
                throw SimError("internal writeback lane count mismatch");
            }
            if (++port_uses[*writeback.dst % config_.rf_banks] > config_.rf_write_ports_per_bank) {
                throw SimError("internal: more RF writes than write ports in bank " +
                    std::to_string(*writeback.dst % config_.rf_banks) + " at cycle " + std::to_string(cycle_));
            }
            for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
                if (writeback.lane_enables.has_value() && (*writeback.lane_enables >> lane & 1) == 0) continue;
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

    for (const auto& writeback : writebacks_.cur()) {
        if (writeback.ready_cycle <= cycle_) execute(writeback);
    }
    const auto& decision = w.rf_write;
    uint32_t lsu_writes = 0;
    for (std::size_t holder = 0; holder < decision.holders.size(); ++holder) {
        if (decision.granted[holder] == 0) continue;
        if (decision.holders[holder]->writer == AsyncWriter::assembly && ++lsu_writes > 1) {
            throw SimError("internal: more than one LSU RF write at cycle " + std::to_string(cycle_));
        }
        execute(async_writeback(*decision.holders[holder]));
    }
    count_rf_write_waits(decision);
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
// shmem (trash/rfc12.md sections 1-5, rfc11 section 0). Four services, each
// granting at most one bank round per core cycle:
//   A-read   SIMT lds / lds.uni (load slots), mbar probes, RMW reads, TMA reads
//   A-write  SIMT sts (store slots), RMW writes
//   B-read   the MMA's operand fetch (step_mma)
//   B-write  TMA placement (gmem -> shmem)
// A transaction -- one SIMT access, one TMA read or placement, one RMW step --
// keeps its service from its first round to its last (no preemption). Every
// wait happens before a grant. A granted round reaches the RAM
// shmem.port_a_depth (D) cycles later -- the same D on A-read, A-write and
// B-write, so the RAM sees rounds in grant order -- and a read's words reach
// the port output shmem.read_latency_cycles (Lb) cycles after its sample. An
// RMW's update then crosses the RMW engine's shmem.rmw_compute_cycles (C) ALU
// stages before its write is granted. A read and a write of one word at the
// RAM in one cycle: READ_FIRST (the write lands at the clock edge). The MMA's
// port-B read is not pipelined: it reads the RAM in its fetch cycle.
//
// The RMW engine (rfc12 section 6): one operation at a time, each one
// coherent 8-byte mbar RMW (both words in one read round, one write round).
// Its requester waits in a register (the SIMT RMW register, the TMA's or the
// MMA's event register). One RMW, with D, Lb and C as above:
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
//   r + D + Lb the words at the port output; the ALU computes the update
//   w = r + D + Lb + C
//              WRITE (the final write) granted on A-write; the PAUSE ends at
//              this cycle's edge
//   w + 1      stores may start again: one granted now reaches the RAM at
//              w + 1 + D, after the RMW's write (matched D, no drain)
//   w + D      the write reaches the RAM; a phase flip wakes the waiters now
// Reads are never paused: between r and w A-read serves loads, probes and
// TMA reads, and a read of the mbar sampling at w + D sees the old word
// (READ_FIRST).
//
// fence.sh (rfc12 section 10) is admitted in the cycle its warp's last
// earlier shmem access REACHES THE RAM (a store's write edge, a load's
// sample: its round's grant + D), so the instruction after it issues once
// that edge has passed. Its declared state is one count per warp
// (shmem_to_ram_: cycles until the warp's latest granted round is at the
// RAM); own_shmem_at_ram is the admission term. Whether an engine command
// needed one is the sanitizer's question at the command's accept (missing
// fence.sh / race, rfc12 section 11).
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
// (rfc12 section 6: at the commit, never at the grant); a placement's last
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
// update into the ALU, a store-source round to its TMA store register. (A
// descriptor round: the frontend latch, advance_tma_read.)
void Machine::deliver_shmem_read(Wires& w) {
    const auto& beat = w.shmem.delivered;
    if (!beat.has_value()) return;
    switch (beat->client) {
    case ReadClient::probe:
        deliver_probe(*beat, w);
        return;
    case ReadClient::rmw:
        w.shmem.rmw_update = rmw_update(*beat);
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

// A probe's words at the port output. test_wait writes its predicate now
// (write-first (a)); its warp issues again now (b). try_wait completes its
// wait slot or parks it. The probing warp was blocked while the probe
// travelled, so if its slot is gone, completed now (commit_shmem_writes ran
// first) or another warp's, a wake ended the wait first: the stale probe does
// nothing (rfc12 section 9).
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
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            warp.predicates[request.predicate_dst][lane] = complete;
        }
        sanitizer_.record_pred_ready(warp.id, request.predicate_dst, cycle_);
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

// ---- The requesters, from Q ----

// The global store PAUSE (rfc12 section 6) is a register: the RMW engine's
// operation register, held from the RMW's start to its final write grant.
// While it is set no new store transaction starts on A-write. It clears at
// the edge of the final write grant w, so stores may start at w + 1 (matched
// D still lands them after the RMW's write; nothing waits for its commit).
bool Machine::store_pause() const {
    return rmw_.cur().active.has_value();
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

bool Machine::tma_read_in_service() const {
    const auto& transaction = tma_read_txn_.cur();
    return transaction.has_value() && transaction->request.next_round < transaction->request.rounds.size();
}

// ---- The selects: one per shared resource (rfc12 sections 5 and 6) ----

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
    if (rmw_.cur().alu.back().has_value()) { // (the PAUSE is set: no RMW starts)
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

// The RMW engine's admission (rfc12 section 6): when A-write lets an RMW
// start, which requester the idle engine takes. Implementer choice (I3), one
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

// A-read, in order:
//   1. a transaction under way takes its next round (no preemption);
//   2. a started RMW waiting for its read (started earlier, or starting now)
//      goes before every new requester. It is granted strictly after the
//      last round of the store transaction under way on A-write, if any:
//      matched D then lands that store before the RMW's sample. Until then
//      nothing new starts on A-read (implementer choice: with no preemption a
//      new multi-round read could only delay the RMW's read and so lengthen
//      the store pause);
//   3. ONE round-robin over the new requesters: load slots not yet started,
//      the probe register, the TMA's new read.
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
        ready[index] = slot.has_value() && !slot->access.started() ? 1 : 0;
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

// An RMW starts: the engine takes the request out of its register (an
// engine's event: its command ends now, step_tma; the sanitizer admits an
// engine's RMW here, a SIMT one at issue). The PAUSE is set from the next
// cycle; the trace's "mbar RMW" slice runs from now to the write grant.
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
    trace_->mbar_rmw_begin(cycle_, active.request.byte_address, rmw_label(active));
    shmem.rmw_started = std::move(active);
}

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
    case AReadUser::rmw: { // the engine's operation, or the one starting now: both mbar words, one round
        const auto& active = rmw_.cur().active.has_value() ? *rmw_.cur().active : *shmem.rmw_started;
        trace_->shmem_grant(cycle_, ShmemService::a_read, ShmemClient::rmw);
        ReadBeat beat;
        beat.client = ReadClient::rmw;
        beat.addresses = {active.request.byte_address, active.request.byte_address + 4};
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

namespace {
const char* tma_port_label(int kind) {
    static const char* const labels[] = {"tma descriptor read", "tma load write", "tma store read"};
    return labels[kind];
}
} // namespace

// A TMA read's next round (round 0 when it starts now). A store-source round
// carries its store register's index; descriptor rounds return to the
// frontend latch.
Machine::ReadBeat Machine::tma_read_round(const TmaPortRequest& request, bool start, uint32_t store_register) {
    const auto rounds = static_cast<uint32_t>(request.rounds.size());
    if (start) {
        trace_->shmem_begin(ShmemService::a_read, cycle_, tma_port_label(static_cast<int>(request.kind)),
            "command=" + std::to_string(request.command) + ", rounds=" + std::to_string(rounds),
            TraceCategory::tma, request.command);
        if (request.kind == TmaPortKind::store_read) trace_->tma_data_port(rounds);
    }
    trace_->shmem_grant(cycle_, ShmemService::a_read, ShmemClient::tma);
    if (request.next_round + 1 == rounds) trace_->shmem_end(ShmemService::a_read, cycle_ + 1);
    ReadBeat beat;
    const bool store = request.kind == TmaPortKind::store_read;
    beat.client = store ? ReadClient::tma_store : ReadClient::tma_frontend;
    beat.slot = store ? store_register : 0;
    beat.round = request.next_round;
    beat.addresses = tma_round_addresses(request, request.next_round);
    return beat;
}

// The distinct words one TMA round touches (at most one per bank).
std::vector<uint32_t> Machine::tma_round_addresses(const TmaPortRequest& request, std::size_t round) const {
    std::vector<uint32_t> addresses;
    for (const auto& piece : request.rounds.at(round)) {
        const auto& fragment = request.fragments.at(piece.fragment);
        const uint32_t address = fragment.shmem_address + piece.offset;
        if (static_cast<uint64_t>(address) + piece.size > shmem_.size()) {
            throw SimError("tma port service is outside shmem");
        }
        const uint32_t word = address / 4 * 4;
        if (std::find(addresses.begin(), addresses.end(), word) == addresses.end()) addresses.push_back(word);
    }
    return addresses;
}

// The TMA read transaction after this cycle: a round granted (a new
// transaction starts with round 0), a descriptor round's words back into the
// frontend latch. A descriptor read ends when its last round's words are
// back, and the TMA takes its data (step_tma); a store-source read ends at
// its last round's grant (its words go to its store register:
// fill_tma_store_register).
void Machine::advance_tma_read(Wires& w) {
    auto& shmem = w.shmem;
    const bool granted = shmem.a_read.user == AReadUser::tma;
    const bool returned = shmem.delivered.has_value() && shmem.delivered->client == ReadClient::tma_frontend;
    if (!granted && !returned) return;
    auto transaction = shmem.a_read.tma_start
        ? TmaReadTransaction{*w.tma.a_read, 0, shmem.tma_store_reserved.value_or(0)}
        : *tma_read_txn_.cur();
    auto& request = transaction.request;
    shmem.tma_read_changed = true;
    if (granted) ++request.next_round;
    if (request.kind == TmaPortKind::store_read) {
        if (request.next_round == request.rounds.size()) {
            shmem.tma_read_after.reset(); // A-read is released at the last round's grant
        } else {
            shmem.tma_read_after = std::move(transaction);
        }
        return;
    }
    if (returned) {
        const auto& beat = *shmem.delivered;
        for (const auto& piece : request.rounds.at(beat.round)) {
            auto& fragment = request.fragments.at(piece.fragment);
            const uint32_t address = fragment.shmem_address + piece.offset;
            const auto at = std::find(beat.addresses.begin(), beat.addresses.end(), address / 4 * 4);
            const auto word = beat.words.at(static_cast<std::size_t>(at - beat.addresses.begin()));
            for (uint32_t byte = 0; byte < piece.size; ++byte) {
                fragment.data.at(piece.offset + byte) = static_cast<uint8_t>(word >> (8 * (address % 4 + byte)));
            }
        }
        ++transaction.returned;
    }
    if (transaction.returned == request.rounds.size()) {
        shmem.tma_read_done = std::move(request);
        shmem.tma_read_after.reset();
        return;
    }
    shmem.tma_read_after = std::move(transaction);
}

// A store-source round's words at the port output, for the TMA store
// register the round names (its return context). The register's own source
// map says where each byte goes in the sector; the last round completes it.
void Machine::fill_tma_store_register(const ReadBeat& beat, Wires& w) {
    const auto& reg = tma_store_registers_.cur().ring.at(beat.slot);
    if (!reg.has_value() || reg->complete()) throw SimError("internal: a TMA store round returned to a free register");
    TmaStoreFill fill;
    fill.store_register = beat.slot;
    fill.command = reg->command;
    for (const auto& piece : reg->source.rounds.at(beat.round)) {
        const auto& fragment = reg->source.fragments.at(piece.fragment);
        const uint32_t address = fragment.shmem_address + piece.offset;
        const auto at = std::find(beat.addresses.begin(), beat.addresses.end(), address / 4 * 4);
        const auto word = beat.words.at(static_cast<std::size_t>(at - beat.addresses.begin()));
        const auto offset = static_cast<uint32_t>(fragment.gmem_address - reg->sector_address);
        for (uint32_t byte = 0; byte < piece.size; ++byte) {
            fill.bytes.emplace_back(offset + piece.offset + byte, static_cast<uint8_t>(word >> (8 * (address % 4 + byte))));
        }
    }
    fill.last = reg->returned + 1 == reg->source.rounds.size();
    probe(LsuEdge::tma_store_round, reg->command, beat.round, static_cast<uint32_t>(4 * beat.addresses.size()));
    w.shmem.tma_store_fill = std::move(fill);
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
        const auto& update = *rmw_.cur().alu.back();
        trace_->shmem_grant(cycle_, ShmemService::a_write, ShmemClient::rmw);
        trace_->mbar_rmw_end(cycle_ + 1, update.wake.has_value());
        // A SIMT RMW's final write: its warp's access reaches the RAM D cycles from now.
        if (const auto warp = rmw_.cur().active->request.warp; warp.has_value()) {
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
    // (The sanitizer also checks C10 here: the round must not write a live mbar.)
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

// The RMW's update: its read's words are at the port output now. The ALU
// computes the new mbar word (and whether the phase flips) from them; the
// result enters ALU stage 0 at this cycle's edge and its write is granted C
// cycles from now (select_a_write).
//
// The tx count is signed (rfc12 section 8, as in PTX): arrive.expect_tx adds,
// complete_tx subtracts, in either order --
//     completion first:  (1, 0) -complete_tx 128-> (1, -128) -arrive.expect_tx 128-> (0, 0) done
//     expectation first: (1, 0) -arrive.expect_tx 128-> (0, 128) -complete_tx 128-> (0, 0) done
// (pending arrivals, tx). arrive.expect_tx changes both, THEN completion is
// tested; the phase completes when both are zero. tx arithmetic is checked: an
// operand above 2^20 - 1 or a count leaving -(2^20 - 1)..2^20 - 1 PANICs (an
// atomic add wraps; this does not). complete_tx carries no phase: it changes
// the current one. Whether that is the phase its sender meant is the
// sanitizer's question (the TMA's accept-time peek; the host's parity).
Machine::WriteBeat Machine::rmw_update(const ReadBeat& read) {
    const auto& request = rmw_.cur().active->request;
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
        for (const auto& wait_slot : mbar_wait_slots_.cur()) {
            if (wait_slot.has_value() && wait_slot->byte_address == request.byte_address) {
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
        --state.remaining;
        add_tx(tx_operand());
        break;
    case MbarRmwOperation::complete_tx:
        validate_live();
        sanitizer_.check_complete_tx_phase(cycle_, request.byte_address, request.tma_command, request.parity,
            state.phase);
        add_tx(-tx_operand());
        break;
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
    if (request.next_round != request.rounds.size()) {
        shmem.tma_write_after = std::move(request);
        return;
    }
    shmem.tma_sector_released = request.from_sector_register;
    shmem.tma_write_after.reset();
}

// One placement round: its bytes from the TMA sector register's Q (filled in
// earlier cycles), or zeros for an OOB fill. The last round carries the
// placement to the RAM, where its load write ends.
Machine::WriteBeat Machine::placement_round(const TmaPortRequest& request, bool start) {
    const auto rounds = static_cast<uint32_t>(request.rounds.size());
    if (request.rounds.empty() || request.next_round >= rounds) {
        throw SimError("internal zero-round tma placement");
    }
    if (start) {
        trace_->shmem_begin(ShmemService::b_write, cycle_, tma_port_label(static_cast<int>(request.kind)),
            "command=" + std::to_string(request.command) + ", rounds=" + std::to_string(rounds),
            TraceCategory::tma, request.command);
        trace_->tma_data_port(rounds);
    }
    const auto& sector = tma_sector_.cur();
    if (request.from_sector_register && !sector.complete) {
        throw SimError("internal: TMA placement started before its sector register was complete");
    }
    WriteBeat beat;
    uint32_t bytes = 0;
    for (const auto& piece : request.rounds[request.next_round]) {
        const auto& fragment = request.fragments.at(piece.fragment);
        const uint32_t address = fragment.shmem_address + piece.offset;
        if (static_cast<uint64_t>(address) + piece.size > shmem_.size()) {
            throw SimError("tma port service is outside shmem");
        }
        const uint8_t* source = fragment.data.data() + piece.offset; // OOB zero fill
        if (request.from_sector_register) {
            source = sector.data.data() + (fragment.gmem_address - sector.sector_address) + piece.offset;
        }
        beat.writes.push_back({address, std::vector<uint8_t>(source, source + piece.size)});
        bytes += piece.size;
    }
    // C10 (sanitizer): a placement round must not write a live mbar.
    std::vector<std::pair<uint32_t, uint32_t>> written;
    for (const auto& write : beat.writes) {
        written.emplace_back(write.address, write.address + static_cast<uint32_t>(write.bytes.size()));
    }
    sanitizer_.check_live_mbar_write(cycle_, "tma load #" + std::to_string(request.command) + " placement", written);
    probe(LsuEdge::tma_place_round, request.sector_request, request.next_round, bytes);
    trace_->shmem_grant(cycle_, ShmemService::b_write, ShmemClient::tma);
    if (request.next_round + 1 == rounds) {
        beat.placed = PlacementEnd{request.command, request.fragments};
        trace_->shmem_end(ShmemService::b_write, cycle_ + 1);
    }
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
    }
    if (active.engine.has_value()) {
        operation += *active.engine == MbarEngine::tma ? " (tma)" : " (mma.commit)";
    }
    return operation;
}

// ---- What issue sees of shmem (look-ahead (c), reuse (d)): the service
// slots and the warp's own order (rfc12 sections 3, 4 and 6) ----

std::optional<uint32_t> Machine::free_load_slot(const Wires& w) const {
    for (uint32_t index = 0; index < load_slots_.size(); ++index) {
        if (!load_slots_[index].cur().has_value()) return index;
        if (w.rf_write.granted_to(AsyncWriter::load_slot, index)) return index;
    }
    return std::nullopt;
}

// A store slot frees when its last round is granted.
std::optional<uint32_t> Machine::free_store_slot(const Wires& w) const {
    for (uint32_t index = 0; index < store_slots_.size(); ++index) {
        if (!store_slots_[index].cur().has_value() || store_slot_released(w, index)) return index;
    }
    return std::nullopt;
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

// An unserviced read: the warp holds a load slot with a round not yet granted.
bool Machine::warp_unserviced_read(const Wires& w, uint32_t warp) const {
    for (uint32_t index = 0; index < load_slots_.size(); ++index) {
        const auto& slot = load_slots_[index].cur();
        if (slot.has_value() && slot->warp == warp && load_rounds_left(w, index)) return true;
    }
    return false;
}

// An unserviced write: the warp holds a store slot, or its SIMT RMW waits in
// the SIMT RMW register or runs in the engine with its final write not
// granted by the end of this cycle (rfc12 section 6: a SIMT RMW counts as
// unserviced until its FINAL write grant).
bool Machine::warp_unserviced_write(const Wires& w, uint32_t warp) const {
    for (uint32_t index = 0; index < store_slots_.size(); ++index) {
        const auto& slot = store_slots_[index].cur();
        if (slot.has_value() && slot->warp == warp && !store_slot_released(w, index)) return true;
    }
    if (simt_rmw_.cur().has_value() && simt_rmw_.cur()->warp == warp) return true;
    const auto& active = rmw_.cur().active;
    return active.has_value() && active->request.warp == warp && w.shmem.a_write.user != AWriteUser::rmw;
}

// fence.sh (rfc12 section 10): every earlier shmem access of the warp has
// REACHED THE RAM by now (stores written, loads sampled) -- none unserviced
// after this cycle's grants, none granted now (that round reaches the RAM D
// cycles from now), and the latest round granted before is at the RAM now or
// was earlier (its count is 0). A probe needs no term: its warp issues
// nothing until the probe returns, D + Lb after its grant.
bool Machine::own_shmem_at_ram(const Wires& w, uint32_t warp) const {
    if (warp_unserviced_read(w, warp) || warp_unserviced_write(w, warp)) return false;
    if (w.shmem.a_read_warp == warp || w.shmem.a_write_warp == warp) return false;
    return shmem_to_ram_.cur().at(warp) == 0;
}

// The warp's probe has not come back: it waits in the probe register or
// travels the read pipeline short of the port output (one at the output
// returns now (b)). The scan stands for a per-warp "probe outstanding" bit
// (set at issue, cleared when the probe returns), the hardware form; both
// give the same answer every cycle.
bool Machine::probe_in_flight(uint32_t warp) const {
    const auto& waiting = mbar_probe_.cur();
    if (waiting.has_value() && waiting->warp == warp) return true;
    const auto& pipe = a_read_pipe_.cur();
    for (std::size_t stage = 0; stage + 1 < pipe.size(); ++stage) {
        const auto& beat = pipe[stage];
        if (beat.has_value() && beat->client == ReadClient::probe && beat->probe->warp == warp) return true;
    }
    return false;
}

uint64_t Machine::full_mask() const {
    return config_.lanes >= 64 ? ~0ull : (1ull << config_.lanes) - 1;
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
        const bool register_free = !mma_mbar_event_.cur().has_value() || w.shmem.mma_event_claimed;
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
// shmem as it was at the start of the cycle (a same-word write reaching the
// RAM in this cycle lands at the clock edge: READ_FIRST, rfc12 section 1).
void Machine::serve_mma_fetch_round(MmaPipeline& mma) {
    auto& fetch = *mma.fetch;
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

// The accumulator transfers of every tmem bank, computed on copies of their
// registers: issue looks ahead at the slots (c) and may reallocate one freed
// now (d). Banks in order, the acc.ld pipeline before the acc.st beat.
void Machine::step_acc_transfers(Wires& w) {
    auto slots = acc_transfers_.cur();
    auto pipes = acc_ld_pipes_.cur();
    for (uint32_t bank = 0; bank < slots.size(); ++bank) {
        step_acc_ld(bank, slots[bank][0], pipes[bank], w);
        step_acc_st(bank, slots[bank][1], w);
    }
    w.acc_next = std::move(slots);
    w.acc_ld_next = std::move(pipes);
}

// One tmem bank's acc.ld transfer pipeline (trash/rfc10.md section 2,
// "Accumulator transfers"). Its output asks the RF write decision for
// service; this cycle's grant was decided at the top of the cycle.
//   output empty, or granted -> advance: every stage moves one toward the
//                               output; stage 0 takes a new tmem read if the
//                               transfer has a beat left, else stays empty
//   output valid and denied  -> hold: no stage moves, nothing launches
// A granted beat belongs to the RF write path from its grant on.
void Machine::step_acc_ld(uint32_t bank, std::optional<AccTransfer>& slot,
    std::vector<std::optional<AccLdBeat>>& pipe, const Wires& w) {
    const bool output_valid = pipe.back().has_value();
    if (output_valid && !w.rf_write.granted_to(AsyncWriter::acc_ld, bank)) return;
    for (std::size_t stage = pipe.size() - 1; stage > 0; --stage) pipe[stage] = std::move(pipe[stage - 1]);
    pipe.front() = launch_acc_ld_beat(bank, slot);
}

// The tmem read of the transfer's next column, all lanes, now. The transfer
// slot -- the bank's coarse ownership -- frees with its last read, as before
// the pipeline existed; each beat carries its own destination and context.
std::optional<Machine::AccLdBeat> Machine::launch_acc_ld_beat(uint32_t bank, std::optional<AccTransfer>& slot) {
    if (!slot.has_value()) return std::nullopt;
    auto& transfer = *slot;
    const auto& acc = kernel_.accumulators.at(transfer.accumulator);
    AccLdBeat beat;
    beat.warp = transfer.warp;
    beat.reg = static_cast<uint8_t>(transfer.rf_base + transfer.next_column);
    beat.values.resize(config_.lanes);
    for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
        beat.values[lane] = tmem_[bank][tmem_word_index(
            acc, transfer.row_base + lane, transfer.col_base + transfer.next_column)];
    }
    beat.last = transfer.next_column + 1 == transfer.width;
    beat.depbar = transfer.depbar;
    trace_->tmem_beat(cycle_, transfer.warp, bank, true, transfer.next_column);
    ++transfer.next_column;
    if (beat.last) {
        sanitizer_.record_acc_transfer_final_beat(static_cast<uint8_t>(bank), true);
        slot.reset();
    }
    return beat;
}

// One acc.st beat per cycle, unchanged by I1: an RF read booked and taken
// now (its place in the RF read booking order), the tmem write now.
void Machine::step_acc_st(uint32_t bank, std::optional<AccTransfer>& slot, Wires& w) {
    if (!slot.has_value()) return;
    auto& transfer = *slot;
    auto& warp = warps_.at(transfer.warp);
    const auto& acc = kernel_.accumulators.at(transfer.accumulator);
    const auto reg = static_cast<uint8_t>(transfer.rf_base + transfer.next_column);
    const std::vector<RfRead> reads{{reg, 0}};
    if (!can_reserve_reads(reads)) return;
    reserve_reads(reads, RfRequester::acc_transfer);
    trace_->rf_read(cycle_, warp.id, reg, reg % config_.rf_banks);
    for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
        tmem_[bank][tmem_word_index(acc, transfer.row_base + lane, transfer.col_base + transfer.next_column)] =
            rf(warp, lane, reg);
    }
    trace_->tmem_beat(cycle_, warp.id, bank, false, transfer.next_column);
    ++transfer.next_column;
    if (transfer.next_column != transfer.width) return;
    for (uint32_t j = 0; j < transfer.width; ++j) {
        sanitizer_.record_async_reader_done(warp.id, transfer.rf_base + j, AsyncReader::acc_st);
    }
    record_depbar_completion(w, warp.id, transfer.depbar, "internal acc.st depbar underflow");
    // final tmem write: write-ownership ends
    sanitizer_.record_acc_transfer_final_beat(static_cast<uint8_t>(bank), false);
    slot.reset();
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
//   3. Port-B write: the front ticket's placement, when no placement is
//      under way.
//   4. The frontend command: its descriptor read (A-read), or the walker's
//      next sector -- a load sector wants the request channel, a store sector
//      the A-read of its source (and a free TMA store register for its
//      words). One TMA read in service at a time: none while a descriptor
//      read's words are still coming back, or while a store-source read has
//      rounds left to be granted. There is no mbar read at the start (rfc12
//      section 8): the walk begins once the descriptor is decoded.
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
            event.tma_command = command.id; // (S) the sanitizer's accept-time phase peek
            want.present_event = std::move(event);
        }
    }
    const auto* slot = egress_request_slot(std::nullopt);
    want.write_request = slot != nullptr && slot->client == GmemClient::tma && egress_slot_visible(*slot);

    if (!tma_write_txn_.cur().has_value() && !tma.returns.empty()) {
        const auto& ticket = tma.returns.front();
        TmaPortRequest request;
        request.kind = TmaPortKind::load_write;
        request.command = ticket.command;
        request.sector_request = ticket.sector_request;
        request.from_sector_register = ticket.from_sector_register;
        request.fragments = ticket.fragments;
        request.rounds = tma_port_rounds(request.fragments);
        want.placement = std::move(request);
    }

    want.frontend = tma.frontend.has_value() ? tma.frontend : next_tma_frontend_command(tma);
    if (!want.frontend.has_value()) return want;
    const auto& command = tma.commands.at(*want.frontend);
    const bool read_free = !tma_read_txn_.cur().has_value();
    if (command.state == TmaCommandState::descriptor) {
        if (!read_free) return want;
        TmaPortRequest request;
        request.kind = TmaPortKind::descriptor;
        request.command = command.id;
        for (uint32_t offset = 0; offset < tma_descriptor_bytes; offset += 4) {
            GmemSectorRequest::TmaFragment fragment;
            fragment.shmem_address = command.descriptor_address + offset;
            fragment.size = 4;
            request.fragments.push_back(fragment);
        }
        request.rounds = tma_port_rounds(request.fragments);
        want.a_read = std::move(request);
        return want;
    }
    if (command.state != TmaCommandState::walking) {
        want.frontend_reset = true;
        return want;
    }
    // A store sector's source read needs a TMA store register (free in Q, or
    // the head one drained into the egress FIFO this cycle (d)) and no TMA
    // read in service.
    const auto store_register = read_free ? tma_store_register_free(egress) : std::nullopt;
    want.walker = plan_tma_walker(command, tma.credits_used, store_register.has_value());
    if (want.walker->sector == TmaWalkerPlan::Sector::store) {
        TmaPortRequest request;
        request.kind = TmaPortKind::store_read;
        request.command = command.id;
        request.fragments = want.walker->fragments;
        request.rounds = tma_port_rounds(request.fragments);
        want.a_read = std::move(request);
        want.store_register = store_register;
    }
    return want;
}

// The TMA store register a source read granted now would reserve: the ring's
// tail, when a register is free in Q or the head one drains this cycle (d).
std::optional<uint32_t> Machine::tma_store_register_free(const EgressWrite& egress) const {
    const auto& stores = tma_store_registers_.cur();
    const auto size = static_cast<uint32_t>(stores.ring.size());
    const auto held = stores.held();
    if (held == size && !egress.tma_store_drained) return std::nullopt;
    return (stores.head + held) % size; // full and draining: the head itself
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
// requests, its shmem transactions' ends, the ingress ticket, the write acks.
// Issue looks ahead at the result (queue entries (c)); drive_tma_queue
// commits it with issue's new command.
void Machine::step_tma(Wires& w) {
    auto tma = tma_.cur();
    // The command whose event the register holds ends in the cycle the RMW
    // engine claims that event (the event's RMW starts: serve_rmw_start).
    // (Today the TMA never presents in that cycle:
    // it presents only when event_command is empty in Q.)
    if (tma.event_command.has_value() && w.shmem.tma_event_claimed) {
        const auto& command = tma.commands.at(*tma.event_command);
        trace_->tma_command_end(cycle_ + 1, command.id, command.logical_bytes, command.useful_bytes);
        tma.commands.erase(*tma.event_command);
        tma.event_command.reset();
        trace_tma_depths(tma, cycle_ + 1);
    }
    apply_tma_requests(tma, w);
    if (w.shmem.tma_read_done.has_value()) complete_tma_frontend_read(tma, *w.shmem.tma_read_done);
    if (w.shmem.tma_store_fill.has_value() && w.shmem.tma_store_fill->last) {
        complete_tma_store_capture(tma, w.shmem.tma_store_fill->command);
    }
    if (w.shmem.tma_placed.has_value()) complete_tma_load_write(tma, *w.shmem.tma_placed);
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
    if (w.shmem.b_write == BWriteUser::tma_start) {
        tma.returns.pop_front(); // its rounds read the sector register until the last one
        trace_tma_depths(tma, cycle_);
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
    // A store sector goes when A-read grants its source read; the others
    // when the request channel lets them through.
    const bool store_read_granted = w.shmem.a_read.user == AReadUser::tma && w.shmem.a_read.tma_start;
    const bool emits = plan.sector == Sector::store ? store_read_granted
                                                    : plan.sector != Sector::none && w.channel.walker_proceeds;
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
        case Sector::store: // its source read reserved a TMA store register (serve_a_read)
            trace_->tma_sector(config_.gmem_sector_bytes);
            ++command.pending_store_captures;
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

// A descriptor read's last round is back: its data is in `request` (the
// frontend latch). The command walks next (I3: no mbar phase read and no
// "enough transaction bytes" check -- the tx count is signed, rfc12 section 8).
void Machine::complete_tma_frontend_read(TmaQueue& tma, const TmaPortRequest& request) {
    auto command_it = tma.commands.find(request.command);
    if (command_it == tma.commands.end()) throw SimError("orphaned tma shmem transaction");
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
        command.state = TmaCommandState::walking;
        return;
    }
    case TmaPortKind::load_write:
    case TmaPortKind::store_read:
        break;
    }
    throw SimError("unexpected tma frontend read");
}

// A placement's last write reached the RAM: its command's load write ends.
void Machine::complete_tma_load_write(TmaQueue& tma, const PlacementEnd& placed) {
    auto command_it = tma.commands.find(placed.command);
    if (command_it == tma.commands.end()) throw SimError("orphaned tma shmem transaction");
    auto& command = command_it->second;
    if (command.descriptor.direction != TmaDirection::load || command.pending_load_writes == 0) {
        throw SimError("unexpected tma load return");
    }
    for (const auto& fragment : placed.fragments) {
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
}

// A store sector's source is captured: its TMA store register is complete
// (from the next cycle it waits for the egress write port).
void Machine::complete_tma_store_capture(TmaQueue& tma, uint64_t command_id) {
    auto command_it = tma.commands.find(command_id);
    if (command_it == tma.commands.end()) throw SimError("orphaned tma shmem transaction");
    auto& command = command_it->second;
    if (command.descriptor.direction != TmaDirection::store || command.pending_store_captures == 0) {
        throw SimError("unexpected tma store capture");
    }
    --command.pending_store_captures;
    if (command.walker_done && command.pending_store_captures == 0) {
        mark_tma_data_complete(command);
    }
}

std::optional<uint64_t> Machine::next_tma_frontend_command(const TmaQueue& tma) const {
    for (const auto& [id, command] : tma.commands) {
        if (command.state == TmaCommandState::descriptor ||
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
        if (probe_in_flight(warp.id)) {
            states[index] = {WarpState::mbar_wait, "mbar probe"};
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

// One warp's issue attempt (trash/rfc10.md section 1):
//   1. metadata panics; the depbar wait; register ranges and the sanitizer's
//      hazard checks (at every attempt, as before: contention must not hide
//      a missing stall);
//   2. admission: structural_refusal, on metadata and resource state alone.
//      A refusal books nothing and reads nothing;
//   3. acceptance: bookings, operand captures, each unit's share of the work.
//      Operand-value panics (alignment, bounds) happen there, where the
//      operands are used.
// The warp's own state (pc, stall count, halted, the SFU and the arithmetic
// operand latches) is the issue unit's and changes in place; everything for
// another unit goes into w.issue, which that unit's driver applies.
Machine::IssueAttempt Machine::try_issue(Warp& warp, Wires& w) {
    const auto inst = instruction(warp);
    sanitizer_.record_issue_attempt(warp.id);
    panic_if_malformed(warp, inst);
    std::array<uint32_t, 6> depbar{};
    for (uint8_t bar = 0; bar < 6; ++bar) depbar[bar] = depbar_after_completions(w, warp.id, bar); // (b)
    if (const auto bar = depbar_wait(inst, depbar); bar.has_value()) {
        return {false, WarpState::depbar, "sb" + std::to_string(*bar)};
    }
    check_issue_hazards(warp, inst);
    const auto plan = issue_plan(inst);
    if (const auto refusal = structural_refusal(inst, plan, resource_view(warp, inst, plan, w)); refusal.has_value()) {
        return refuse(*refusal);
    }
    sanitizer_.record_issue_accepted();
    accept(warp, inst, plan, w);
    return {true, WarpState::issued, {}};
}

// Panics on the instruction word alone: register encodings that wrap, and
// predication outside bra.uni.
void Machine::panic_if_malformed(const Warp& warp, const Instruction& inst) const {
    if ((inst.opcode == Opcode::ldg_v8 && static_cast<unsigned>(inst.dst) + 8 > 256) ||
        (inst.opcode == Opcode::stg_v8 && static_cast<unsigned>(inst.src1) + 8 > 256))
        panic(warp, inst, "global vector register encoding wraps");
    if (is_acc_transfer(inst.opcode)) {
        const bool load = is_acc_load(inst.opcode);
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
}

// Register, predicate and accumulator-id ranges (machine panics: properties
// of the instruction), then the sanitizer's hazard checks. Neither looks at
// an operand value.
void Machine::check_issue_hazards(const Warp& warp, const Instruction& inst) const {
    for (const auto reg : source_registers(inst)) {
        if (reg >= kernel_.register_count) panic(warp, inst, "source register out of range");
    }
    const auto destination = destination_register(inst);
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
    if (is_acc_transfer(inst.opcode)) {
        const auto accumulator = is_acc_load(inst.opcode) ? inst.src0 : inst.dst;
        if (accumulator >= kernel_.accumulators.size()) panic(warp, inst, "accumulator id out of range");
        // Racing accepted MMA work is UB (cases 01/02): no ownership hardware
        // exists, but the simulator has global knowledge and diagnoses it.
        sanitizer_.check_acc_transfer_vs_mma(site(warp, inst), accumulator);
    }
}

// What an accepted instruction books, from metadata and the profile. acc.st
// reads only its coordinates at issue (its data beat by beat) and stg.v8
// only its base (its data in the LSU's RF read stage).
IssuePlan Machine::issue_plan(const Instruction& inst) const {
    IssuePlan plan;
    auto sources = source_registers(inst);
    if (inst.opcode == Opcode::acc_st || inst.opcode == Opcode::acc_st_v8) {
        sources = {inst.src0, static_cast<uint8_t>(inst.src0 + 1)};
    } else if (inst.opcode == Opcode::stg_v8) {
        sources = {inst.src0};
    }
    plan.reads = rf_read_plan(inst, config_, sources);
    const auto destination = destination_register(inst);
    if (uses_arithmetic_pipeline(inst) && destination.has_value()) {
        plan.write_reg = destination;
        plan.write_cycle = cycle_ + scheduled_fixed_latency(inst, config_);
    }
    return plan;
}

// The resource state admission may see, as this cycle leaves it (c): the RF
// calendars and this cycle's RF write decision, unit occupancy, queue and
// slot space. Instruction fields select WHICH resource (a register number's
// bank, an accumulator's tmem bank); no register content is read.
ResourceView Machine::resource_view(
    const Warp& warp, const Instruction& inst, const IssuePlan& plan, const Wires& w) const {
    ResourceView view;
    view.reads_fit = can_reserve_reads(plan.reads);
    view.sfu_free = cycle_ + config_.rf_arithmetic_read_cycles - 1 >= sfu_busy_until_;
    if (plan.write_reg.has_value()) {
        view.write_slot_free = write_slot_free(*plan.write_reg, plan.write_cycle);
        view.write_inhibited = w.rf_write.banks.at(*plan.write_reg % config_.rf_banks).inhibit_simt;
    }
    view.wait_slot_free = free_mbar_wait_slot(w).has_value();
    view.tma_queue_room = w.tma_next.commands.size() < config_.tma_queue_commands;
    view.mma_queue_room = w.mma_next.queue.size() < config_.mma_queue_commands;
    if (is_acc_transfer(inst.opcode)) {
        const bool load = is_acc_load(inst.opcode);
        const auto& accumulator = kernel_.accumulators.at(load ? inst.src0 : inst.dst);
        view.acc_slot_free = acc_slot_available(w, accumulator.bank, load);
    }
    view.no_active_lanes = warp.active_mask == 0;
    view.unserviced_read = warp_unserviced_read(w, warp.id);
    view.unserviced_write = warp_unserviced_write(w, warp.id);
    if (inst.opcode == Opcode::fence_sh) view.own_shmem_at_ram = own_shmem_at_ram(w, warp.id);
    view.load_slot_free = free_load_slot(w).has_value();
    view.store_slot_free = free_store_slot(w).has_value();
    view.probe_register_free = !mbar_probe_.cur().has_value() || w.shmem.a_read.user == AReadUser::probe;
    view.simt_rmw_register_free = !simt_rmw_.cur().has_value() || w.shmem.simt_rmw_started;
    view.lsu_slot_free =
        lsu_commands_.cur().size() - w.released_lsu_commands.size() < config_.lsu_active_commands;
    view.assembly_free = free_assembly_register(w).has_value();
    return view;
}

// A refused attempt: the warp state and reason for the trace, and its counter.
Machine::IssueAttempt Machine::refuse(Refusal refusal) {
    switch (refusal) {
    case Refusal::rf_read_booked: return {false, WarpState::structural, "RF read reservation"};
    case Refusal::sfu_busy:
        trace_->sfu_unit_stall();
        return {false, WarpState::structural, "SFU unit busy"};
    case Refusal::rf_write_booked: return {false, WarpState::structural, "RF write reservation"};
    case Refusal::rf_write_inhibited:
        trace_->rf_simt_inhibited();
        return {false, WarpState::structural, "RF write inhibited"};
    case Refusal::wait_slots_full: return {false, WarpState::mbar_wait_slot, "mbar wait slots full"};
    case Refusal::tma_queue_full: return {false, WarpState::structural, "tma command queue full"};
    case Refusal::mma_queue_full: return {false, WarpState::structural, "MMA command queue full"};
    case Refusal::tmem_bank_busy: return {false, WarpState::structural, "tmem bank transfer busy"};
    case Refusal::shmem_ordered:
        trace_->shmem_order_wait();
        return {false, WarpState::shmem_order, "ordered behind own access"};
    // Implementer choice (I4): the fence's wait is the same warp state (it is
    // ordered behind the warp's own accesses), told apart by its detail; it
    // is not counted in the summary's shmem "ordered=" (section 4 refusals).
    case Refusal::shmem_fence: return {false, WarpState::shmem_order, "fence.sh: own accesses not at the RAM"};
    case Refusal::load_slots_full:
        trace_->shmem_load_slot_wait();
        return {false, WarpState::shmem_slot, "load slots full"};
    case Refusal::store_slots_full:
        trace_->shmem_store_slot_wait();
        return {false, WarpState::shmem_slot, "store slots full"};
    case Refusal::probe_register_busy: return {false, WarpState::structural, "mbar probe register busy"};
    case Refusal::simt_rmw_register_busy: return {false, WarpState::structural, "SIMT RMW register busy"};
    case Refusal::lsu_slot:
        ++trace_->lsu_counters().lsu_no_slot_cycles;
        return {false, WarpState::structural, "LSU command slot"};
    case Refusal::lsu_assembly:
        ++trace_->lsu_counters().lsu_no_staging_cycles;
        return {false, WarpState::structural, "LSU assembly register"};
    }
    throw SimError("internal unknown issue refusal");
}

// An accepted instruction: its bookings, its operand captures, its units' work.
void Machine::accept(Warp& warp, const Instruction& inst, const IssuePlan& plan, Wires& w) {
    if (uses_arithmetic_pipeline(inst)) {
        accept_arithmetic(warp, inst, plan, w);
        return;
    }
    // The units' captures (hand_over) come before the read booking in code
    // only: admission already proved the reads fit (reads_fit), and both
    // happen in this one accepted cycle.
    const auto next_pc = hand_over(warp, inst, w);
    check_issue_latency(warp, inst);
    reserve_reads(plan.reads, RfRequester::issue);
    for (const auto& read : plan.reads) trace_->rf_read(cycle_, warp.id, read.reg, read.reg % config_.rf_banks);
    warp.stall_until = cycle_ + inst.control.stall;
    const auto old_pc = warp.pc;
    warp.pc = next_pc;
    trace_->issue(cycle_, warp.id, old_pc, inst);
}

// Fixed-latency work books its read window and its result write as one
// bundle; step_arithmetic_reads captures the operands at their offsets. A
// booking of the result's bank is what sets that bank's async_owed bit.
void Machine::accept_arithmetic(Warp& warp, const Instruction& inst, const IssuePlan& plan, Wires& w) {
    // All fixed operations (including MOV/SREAD) enter execution at the same
    // offset, so one accepted issue per cycle reserves execution as well.
    if (arithmetic_reads_.size() >= config_.rf_arithmetic_read_cycles) {
        throw SimError("internal RF read pipeline exceeded its bounded window");
    }
    if (plan.write_reg.has_value()) {
        reserve_write(*plan.write_reg, plan.write_cycle);
        w.issue.rf_write_bank = *plan.write_reg % config_.rf_banks;
    }
    reserve_reads(plan.reads, RfRequester::issue);
    if (is_sfu(inst.opcode)) {
        const auto occupancy = (config_.lanes + config_.sfu_lanes - 1) / config_.sfu_lanes;
        sfu_busy_until_ = cycle_ + config_.rf_arithmetic_read_cycles - 1 + occupancy;
        trace_->sfu_accept(occupancy);
    }
    const auto result_cycle = cycle_ + scheduled_fixed_latency(inst, config_);
    for (const auto& read : plan.reads) {
        sanitizer_.record_reg_read_until(warp.id, read.reg, cycle_ + read.offset + 1);
    }
    if (plan.write_reg.has_value()) sanitizer_.record_reg_ready(warp.id, *plan.write_reg, result_cycle);
    else sanitizer_.record_pred_ready(warp.id, inst.dst, result_cycle);
    ArithmeticRead pending;
    pending.issue_cycle = cycle_;
    pending.warp = warp.id;
    pending.instruction = inst;
    pending.reads = plan.reads;
    pending.operands.resize(plan.reads.size());
    arithmetic_reads_.push_back(std::move(pending));
    warp.stall_until = cycle_ + inst.control.stall;
    const auto old_pc = warp.pc;
    warp.pc += 8;
    trace_->issue(cycle_, warp.id, old_pc, inst);
}

uint32_t Machine::hand_over(Warp& warp, const Instruction& inst, Wires& w) {
    switch (inst.opcode) {
    case Opcode::mma_async:
    case Opcode::mma_async_init:
    case Opcode::mma_commit:
        issue_mma(warp, inst, w);
        break;
    case Opcode::acc_ld:
    case Opcode::acc_ld_v8:
    case Opcode::acc_st:
    case Opcode::acc_st_v8:
        issue_acc_transfer(warp, inst, w);
        break;
    case Opcode::lds:
    case Opcode::lds_uni:
        issue_shared_load(warp, inst, w);
        break;
    case Opcode::sts:
        issue_shared_store(warp, inst, w);
        break;
    case Opcode::ldg:
    case Opcode::stg:
    case Opcode::ldg_v8:
    case Opcode::stg_v8:
    case Opcode::ldg_uni:
        issue_lsu(warp, inst, w);
        break;
    case Opcode::tma_async:
        issue_tma(warp, inst, w);
        break;
    case Opcode::mbar_init_i:
    case Opcode::mbar_init_r:
    case Opcode::mbar_arrive:
    case Opcode::mbar_arrive_expect_tx_i:
    case Opcode::mbar_arrive_expect_tx_r:
        issue_mbar_rmw(warp, inst, w);
        break;
    case Opcode::mbar_test_wait_i:
    case Opcode::mbar_test_wait_r:
    case Opcode::mbar_try_wait_i:
    case Opcode::mbar_try_wait_r:
        issue_mbar_wait(warp, inst, w);
        break;
    case Opcode::bra:
    case Opcode::bra_uni:
        return branch_target(warp, inst);
    case Opcode::fence_sh:
        break; // its admission was the whole of it (own_shmem_at_ram)
    case Opcode::halt:
        warp.halted = true;
        break;
    default:
        panic(warp, inst, "unknown opcode");
    }
    return warp.pc + 8;
}

void Machine::issue_mma(const Warp& warp, const Instruction& inst, Wires& w) {
    if (inst.opcode == Opcode::mma_commit) {
        MmaCommand command;
        command.id = next_mma_command_id_++;
        command.commit = true;
        command.warp = warp.id;
        command.mbar_address = mbar_address_operand(warp, inst);
        w.issue.mma = std::move(command);
        return;
    }
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
}

// acc.ld / acc.st: the coordinates are captured now; the transfer runs one
// beat per column from its slot (step_acc_transfers).
void Machine::issue_acc_transfer(const Warp& warp, const Instruction& inst, Wires& w) {
    const bool load = is_acc_load(inst.opcode);
    const uint32_t width = inst.opcode == Opcode::acc_ld_v8 || inst.opcode == Opcode::acc_st_v8 ? 8 : 1;
    const uint8_t acc_id = load ? inst.src0 : inst.dst;
    const uint8_t coord = load ? inst.src1 : inst.src0;
    const uint8_t rf_base = load ? inst.dst : inst.src1;
    const auto& acc = kernel_.accumulators[acc_id];
    const auto row = uniform_reg(warp, inst, coord, "accumulator row coordinate");
    const auto col = uniform_reg(warp, inst, static_cast<uint8_t>(coord + 1), "accumulator column coordinate");
    sanitizer_.check_acc_transfer_overlap(site(warp, inst), acc_id, load, row, col, width);
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
    sanitizer_.record_acc_transfer_accept(site(warp, inst), acc.bank, load, acc_id,
        row, col, width, inst.control.set_depbar);
    w.issue.acc = IssueDecision::AccAdmission{acc.bank, load, std::move(transfer)};
}

// lds / lds.uni (trash/rfc10.md section 2, rfc12 section 3): the address is
// captured now and the bank rounds come from it (active lanes only). The
// load takes its load slot now; A-read serves it from the next cycle; its
// gathered vector waits in the slot for the RF write decision. No RF booking:
// the depbar completes it. With no active lane a plain lds is absent: no
// slot, no depbar, no RF write.
void Machine::issue_shared_load(const Warp& warp, const Instruction& inst, Wires& w) {
    if (inst.control.set_depbar >= 6) panic(warp, inst, opcode_name(inst.opcode) + " has no depbar");
    const bool uniform = inst.opcode == Opcode::lds_uni;
    if (uniform) panic_if_partial_mask(warp, inst);
    if (warp.active_mask == 0) return;
    auto rounds = uniform ? build_uniform_shmem_round(warp, inst) : build_shmem_rounds(warp, inst, true, w);
    const auto index = *free_load_slot(w);
    sanitizer_.check_simt_shmem_access(site(warp, inst), access_addresses(rounds), true);
    sanitizer_.record_simt_shmem_issue(site(warp, inst), SimtShmemHolder::load_slot, index, round_addresses(rounds),
        4, false);
    sanitizer_.record_reg_ready(warp.id, inst.dst, std::numeric_limits<uint64_t>::max()); // until its RF write
    admit_depbar(w, warp.id, inst.control.set_depbar);
    LoadSlot slot;
    slot.warp = warp.id;
    slot.uniform = uniform;
    slot.mask = warp.active_mask;
    slot.access.rounds = std::move(rounds);
    slot.dst = inst.dst;
    slot.depbar = inst.control.set_depbar;
    slot.words.assign(config_.lanes, 0);
    w.issue.load_slot = std::make_pair(index, std::move(slot));
}

// sts (rfc12 section 3): address and data captured now (its RF reads were
// booked with the issue), into a store slot that A-write serves from the
// next cycle. With no active lane it is absent.
void Machine::issue_shared_store(const Warp& warp, const Instruction& inst, Wires& w) {
    if (inst.control.set_depbar >= 6) panic(warp, inst, "sts has no depbar");
    if (warp.active_mask == 0) return;
    auto rounds = build_shmem_rounds(warp, inst, false, w);
    const auto index = *free_store_slot(w);
    sanitizer_.check_simt_shmem_access(site(warp, inst), access_addresses(rounds), false);
    sanitizer_.record_simt_shmem_issue(site(warp, inst), SimtShmemHolder::store_slot, index, round_addresses(rounds),
        4, true);
    admit_depbar(w, warp.id, inst.control.set_depbar);
    StoreSlot slot;
    slot.warp = warp.id;
    slot.access.rounds = std::move(rounds);
    slot.depbar = inst.control.set_depbar;
    w.issue.store_slot = std::make_pair(index, std::move(slot));
}

// ldg / stg / ldg.v8 / stg.v8 / ldg.uni: the addresses are captured and
// checked now; the command takes its slot and assembly register.
void Machine::issue_lsu(const Warp& warp, const Instruction& inst, Wires& w) {
    if (inst.control.set_depbar >= 6) panic(warp, inst, "global memory operation has no depbar");
    auto& counters = trace_->lsu_counters();
    auto command = plan_lsu_command(warp, inst);
    command.id = next_lsu_command_id_++;
    command.depbar = inst.control.set_depbar;
    command.assembly = *free_assembly_register(w);
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
    case LsuKind::scalar_store:
        // The issue-cycle RF read beat (planned with the address operand)
        // lands the store data in the command's assembly register.
        admission.store_data = capture(warp, inst.src1);
        probe(LsuEdge::rf_read_beat, command.id, inst.src1, config_.lanes * 4);
        ++counters.lsu_scalar_stores;
        counters.lsu_useful_bytes += 4ull * config_.lanes;
        break;
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
}

void Machine::issue_tma(const Warp& warp, const Instruction& inst, Wires& w) {
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
    // Sanitizer: the descriptor the engine will read -- memory as of this
    // cycle's edge; anything still travelling to it PANICs in
    // record_tma_accept (missing fence.sh / race, rfc12 section 11) -- and the
    // phase of its completion mbar now, the accept-time peek a TMA's
    // complete_tx is checked against (rfc12 section 8; the hardware reads no
    // mbar).
    const auto descriptor_bytes = tma_descriptor_at_accept(command.descriptor_address, w);
    sanitizer_.record_tma_accept(site(warp, inst), command_id, command.descriptor_address, descriptor_bytes,
        pending_tma_completion_mbar(descriptor_bytes, w));
    trace_->tma_command_begin(cycle_, command_id, warp.id, static_cast<uint32_t>(inst.imm),
        w.tma_next.commands.size() + 1);
    w.issue.tma = std::move(command);
}

void Machine::issue_mbar_rmw(const Warp& warp, const Instruction& inst, Wires& w) {
    panic_if_partial_mask(warp, inst);
    MbarRmwRequest request;
    request.byte_address = mbar_address_operand(warp, inst);
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
    if (request.operation == MbarRmwOperation::init) sanitizer_.record_mbar_init(site(warp, inst), request.byte_address);
    sanitizer_.record_simt_shmem_issue(site(warp, inst), SimtShmemHolder::rmw, warp.id,
        std::vector<std::vector<uint32_t>>{{request.byte_address}}, 8, true);
    // The SIMT RMW register is empty after this cycle (admission). The
    // operation waits there until the RMW engine starts it (A-write's
    // alternation, then the engine's admission); its warp's younger shmem
    // accesses wait until its final write grant (warp_unserviced_write).
    w.issue.rmw = std::move(request);
}

// mbar.test_wait / try_wait: the probe goes into the probe register; A-read
// serves it from the next cycle and its warp waits until it comes back
// (rfc12 section 9). try_wait installs its wait slot first, so a wake that
// comes while the probe travels finds it.
void Machine::issue_mbar_wait(const Warp& warp, const Instruction& inst, Wires& w) {
    panic_if_partial_mask(warp, inst);
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
    request.byte_address = mbar_address_operand(warp, inst);
    request.parity = static_cast<uint8_t>(parity_value);
    request.pc = warp.pc;
    request.instruction = inst;
    if (test) {
        sanitizer_.record_pred_ready(warp.id, inst.dst, std::numeric_limits<uint64_t>::max()); // until the probe returns
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
    sanitizer_.record_simt_shmem_issue(site(warp, inst), SimtShmemHolder::probe, warp.id,
        std::vector<std::vector<uint32_t>>{{request.byte_address}}, 8, false);
    w.issue.mbar_probe = std::move(request);
}

// bra / bra.uni. bra.uni's predicate is a declared-uniform operand: hardware
// reads lane 0 (the sanitizer checks the other lanes).
uint32_t Machine::branch_target(const Warp& warp, const Instruction& inst) const {
    bool take = true;
    if (inst.opcode == Opcode::bra_uni && inst.control.predicate != 7) {
        const auto lanes = capture_predicate(warp, inst.control.predicate);
        sanitizer_.check_warp_uniform(site(warp, inst), lanes, "bra.uni predicate");
        take = (lanes[0] != 0) != inst.control.predicate_invert;
    } else if (inst.opcode == Opcode::bra_uni && inst.control.predicate_invert) {
        take = false; // @!PT
    }
    if (!take) return warp.pc + 8;
    const auto target = static_cast<int64_t>(warp.pc) + 8 + inst.imm;
    if (inst.imm % 8 != 0 || target < 0 ||
        static_cast<uint64_t>(target / 8) >= kernel_.words.size()) {
        panic(warp, inst, "branch target is outside the kernel");
    }
    return static_cast<uint32_t>(target);
}

// The ISA tables' invariant: outside the arithmetic pipeline only
// mbar.test_wait has a fixed (one-cycle predicate) latency, and only the
// listed opcodes may have none.
void Machine::check_issue_latency(const Warp& warp, const Instruction& inst) const {
    const auto latency = fixed_latency(inst, config_.timing);
    const auto opcode = inst.opcode;
    if (is_mbar_test_wait(opcode)) {
        if (latency != 1) panic(warp, inst, "mbar.test_wait must have one-cycle predicate latency");
    } else if (latency != 0) {
        panic(warp, inst, "fixed-latency opcode bypassed the arithmetic read pipeline");
    } else if (!is_shared_load(opcode) && opcode != Opcode::sts && !is_lsu_opcode(opcode) &&
               !is_tma(opcode) && !is_acc_transfer(opcode) && !is_mma_command(opcode) &&
               opcode != Opcode::bra && opcode != Opcode::bra_uni && opcode != Opcode::fence_sh &&
               !is_mbar_rmw(opcode) && !is_mbar_try_wait(opcode) && opcode != Opcode::halt) {
        panic(warp, inst, "zero-latency RF writer is unsupported");
    }
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
    static constexpr std::array<Driver, 28> canonical = {
        &Machine::drive_depbar,
        &Machine::drive_wait_slots,
        &Machine::drive_load_slots,
        &Machine::drive_store_slots,
        &Machine::drive_mbar_probe,
        &Machine::drive_simt_rmw,
        &Machine::drive_rmw_engine,
        &Machine::drive_a_read_pipe,
        &Machine::drive_write_pipes,
        &Machine::drive_shmem_arbiters,
        &Machine::drive_shmem_to_ram,
        &Machine::drive_tma_event,
        &Machine::drive_mma_event,
        &Machine::drive_mma,
        &Machine::drive_acc_transfers,
        &Machine::drive_acc_ld_pipes,
        &Machine::drive_tma_queue,
        &Machine::drive_tma_transactions,
        &Machine::drive_tma_sector,
        &Machine::drive_tma_store_registers,
        &Machine::drive_lsu_commands,
        &Machine::drive_assembly_pool,
        &Machine::drive_ingress,
        &Machine::drive_egress,
        &Machine::drive_read_records,
        &Machine::drive_free_id_tags,
        &Machine::drive_writebacks,
        &Machine::drive_rf_write_policy,
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
    if (w.completed_waits.empty() && !w.shmem.park.has_value() && !w.issue.wait_slot.has_value()) return;
    auto next = mbar_wait_slots_.cur();
    for (const auto slot : w.completed_waits) next.at(slot).reset();
    if (const auto& park = w.shmem.park; park.has_value()) {
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

// Each load slot, in one calculation: the old owner leaves when the RF write
// decision grants its write; issue's admission installs a new owner (into a
// slot freed now, (d)); A-read grants its next round; a round's words come
// back into the gather register.
void Machine::drive_load_slots(const Wires& w) {
    for (uint32_t index = 0; index < load_slots_.size(); ++index) {
        const bool written = w.rf_write.granted_to(AsyncWriter::load_slot, index);
        const bool admitted = w.issue.load_slot.has_value() && w.issue.load_slot->first == index;
        const bool granted = w.shmem.a_read.user == AReadUser::load_slot && w.shmem.a_read.slot == index;
        const bool filled = w.shmem.load_fill.has_value() && w.shmem.load_fill->slot == index;
        if (!written && !admitted && !granted && !filled) continue;
        auto next = written ? std::nullopt : load_slots_[index].cur();
        if (admitted) {
            if (next.has_value()) throw SimError("internal lds admitted into a held load slot");
            next = w.issue.load_slot->second;
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
// it -- for issue's admission in this same cycle (d).
void Machine::drive_store_slots(const Wires& w) {
    for (uint32_t index = 0; index < store_slots_.size(); ++index) {
        const bool granted = w.shmem.a_write.user == AWriteUser::store_slot && w.shmem.a_write.slot == index;
        const bool admitted = w.issue.store_slot.has_value() && w.issue.store_slot->first == index;
        if (!granted && !admitted) continue;
        auto next = store_slots_[index].cur();
        if (granted) {
            ++next->access.granted;
            if (next->access.serviced()) next.reset();
        }
        if (admitted) {
            if (next.has_value()) throw SimError("internal sts admitted into a held store slot");
            next = w.issue.store_slot->second;
        }
        store_slots_[index].set_next(std::move(next), "drive_store_slots");
    }
}

// The probe register: freed by its A-read grant, refilled by issue (d).
void Machine::drive_mbar_probe(const Wires& w) {
    const bool granted = w.shmem.a_read.user == AReadUser::probe;
    if (w.issue.mbar_probe.has_value()) {
        if (mbar_probe_.cur().has_value() && !granted) throw SimError("internal mbar probe accepted over another");
        mbar_probe_.set_next(w.issue.mbar_probe, "issue: new mbar probe");
    } else if (granted) {
        mbar_probe_.set_next(std::nullopt, "A-read: mbar probe granted");
    }
}

// The SIMT RMW register: emptied by its RMW's start, refilled by issue in the
// same cycle (d).
void Machine::drive_simt_rmw(const Wires& w) {
    const bool started = w.shmem.simt_rmw_started;
    if (w.issue.rmw.has_value()) {
        if (simt_rmw_.cur().has_value() && !started) throw SimError("internal SIMT RMW accepted over another");
        simt_rmw_.set_next(w.issue.rmw, "issue: SIMT RMW");
    } else if (started) {
        simt_rmw_.set_next(std::nullopt, "RMW engine: SIMT RMW started");
    }
}

// The RMW engine, in one calculation: its ALU moves one stage (the update
// computed from the words at the port output now enters stage 0; the last
// stage's leaves with the write granted now); the write granted now ends the
// operation and the PAUSE; an RMW starting now installs its operation (and
// the admission state: which class, which engine started last); the read
// granted now marks it.
void Machine::drive_rmw_engine(const Wires& w) {
    const auto& cur = rmw_.cur();
    const bool started = w.shmem.rmw_start.has_value();
    const bool read_granted = w.shmem.a_read.user == AReadUser::rmw;
    const bool write_granted = w.shmem.a_write.user == AWriteUser::rmw;
    const bool computing = w.shmem.rmw_update.has_value() ||
        std::any_of(cur.alu.begin(), cur.alu.end(), [](const auto& stage) { return stage.has_value(); });
    if (!started && !read_granted && !write_granted && !computing) return;
    auto next = cur;
    for (std::size_t stage = next.alu.size() - 1; stage > 0; --stage) next.alu[stage] = cur.alu[stage - 1];
    next.alu.front() = w.shmem.rmw_update;
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

// The read pipeline moves one stage: the round granted now enters stage 0,
// the round sampled now carries its words into stage D, the round at the
// output leaves.
void Machine::drive_a_read_pipe(const Wires& w) {
    const auto& cur = a_read_pipe_.cur();
    const bool idle = !w.shmem.a_read_beat.has_value() &&
        std::none_of(cur.begin(), cur.end(), [](const auto& stage) { return stage.has_value(); });
    if (idle) return;
    ReadPipeline next(cur.size());
    next.front() = w.shmem.a_read_beat;
    for (std::size_t stage = 1; stage < cur.size(); ++stage) next[stage] = cur[stage - 1];
    // The shift put cur[D - 1] (the round at the RAM now) into next[D]; this
    // replaces it with the same round carrying the words it sampled.
    next.at(config_.shmem_port_a_depth) = w.shmem.sampled;
    a_read_pipe_.set_next(std::move(next), "drive_a_read_pipe");
}

// The A-write and port-B write pipelines move one stage: the round granted
// now enters stage 0; the round at the last stage reached the RAM now.
void Machine::drive_write_pipes(const Wires& w) {
    const auto shift = [](Reg<WritePipeline>& pipe, const std::optional<WriteBeat>& entering, const char* tag) {
        const auto& cur = pipe.cur();
        const bool idle = !entering.has_value() &&
            std::none_of(cur.begin(), cur.end(), [](const auto& stage) { return stage.has_value(); });
        if (idle) return;
        WritePipeline next(cur.size());
        next.front() = entering;
        for (std::size_t stage = 1; stage < cur.size(); ++stage) next[stage] = cur[stage - 1];
        pipe.set_next(std::move(next), tag);
    };
    shift(a_write_pipe_, w.shmem.a_write_beat, "drive_write_pipes: A-write");
    shift(b_write_pipe_, w.shmem.b_write_beat, "drive_write_pipes: port-B write");
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

// The TMA's event register: emptied by the RMW engine's claim, filled by the
// TMA's presentation (never both in one cycle: the TMA presents only into an
// empty register).
void Machine::drive_tma_event(const Wires& w) {
    if (!w.shmem.tma_event_claimed && !w.tma.present_event.has_value()) return;
    auto next = w.shmem.tma_event_claimed ? std::nullopt : tma_mbar_event_.cur();
    if (w.tma.present_event.has_value()) {
        if (next.has_value()) throw SimError("internal TMA event presented into a full register");
        next = w.tma.present_event;
    }
    tma_mbar_event_.set_next(std::move(next), "drive_tma_event");
}

// The MMA's event register: a claim and a new commit's presentation may
// happen in one cycle (d).
void Machine::drive_mma_event(const Wires& w) {
    if (!w.shmem.mma_event_claimed && !w.mma_present.has_value()) return;
    auto next = w.shmem.mma_event_claimed ? std::nullopt : mma_mbar_event_.cur();
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

// The acc.ld pipelines: step_acc_transfers' advance or hold, per tmem bank.
void Machine::drive_acc_ld_pipes(const Wires& w) {
    if (w.acc_ld_next == acc_ld_pipes_.cur()) return;
    acc_ld_pipes_.set_next(w.acc_ld_next, "drive_acc_ld_pipes");
}

// The RF write decision's state, per bank (trash/rfc10.md section 3):
//   async_owed_next = simt_booked_this_cycle OR (async_owed AND NOT async_grant)
// (a successful SIMT booking sets it, an actual async grant clears it, set
// wins); the cursor moves past the holder granted last, only on a grant.
void Machine::drive_rf_write_policy(const Wires& w) {
    auto next = rf_write_policy_.cur();
    for (uint32_t bank = 0; bank < config_.rf_banks; ++bank) {
        const auto& decision = w.rf_write.banks[bank];
        const bool booked = w.issue.rf_write_bank == bank;
        next.owed[bank] = booked || (next.owed[bank] != 0 && !decision.granted) ? 1 : 0;
        next.cursor[bank] = decision.cursor_after;
    }
    const auto presented = w.rf_write.lsu_presented;
    if (presented.has_value() && w.rf_write.granted_to(AsyncWriter::assembly, *presented)) {
        next.lsu_cursor = (*presented + 1) % static_cast<uint32_t>(staging_.size());
    }
    if (next == rf_write_policy_.cur()) return;
    rf_write_policy_.set_next(std::move(next), "drive_rf_write_policy");
}

// The TMA command queue: step_tma's result, plus issue's new command.
void Machine::drive_tma_queue(const Wires& w) {
    auto next = w.tma_next;
    if (w.issue.tma.has_value()) next.commands.emplace(w.issue.tma->id, *w.issue.tma);
    tma_.set_next(std::move(next), "drive_tma_queue");
}

// The TMA's shmem transactions after this cycle's grants and returns.
void Machine::drive_tma_transactions(const Wires& w) {
    if (w.shmem.tma_read_changed) tma_read_txn_.set_next(w.shmem.tma_read_after, "shmem: TMA read");
    if (w.shmem.tma_write_changed) tma_write_txn_.set_next(w.shmem.tma_write_after, "shmem: TMA placement");
}

// The TMA sector register: freed by its placement's last round, refilled by
// the ingress FIFO (the pluck's copy already starts from the freed register,
// so a same-cycle refill is one calculation (d)).
void Machine::drive_tma_sector(const Wires& w) {
    const bool placed = w.shmem.tma_sector_released;
    if (w.ingress.tma_sector_after.has_value()) {
        tma_sector_.set_next(*w.ingress.tma_sector_after, "ingress: TMA sector fill");
    } else if (placed) {
        auto next = tma_sector_.cur();
        next.request.reset();
        next.filled = 0;
        next.complete = false;
        tma_sector_.set_next(std::move(next), "port-B write: TMA sector placed");
    }
}

// The TMA store sector registers, in one calculation: the head register's
// words drain into the egress FIFO and its last word frees it (the head
// moves on); a source read granted now reserves the tail register -- the
// one just freed, when the ring was full (d); a source round's words land in
// the register the round names.
void Machine::drive_tma_store_registers(const Wires& w) {
    const bool draining = w.egress.client == GmemClient::tma;
    const auto& reserved = w.shmem.tma_store_reserved;
    const auto& fill = w.shmem.tma_store_fill;
    if (!draining && !reserved.has_value() && !fill.has_value()) return;
    auto next = tma_store_registers_.cur();
    const auto size = static_cast<uint32_t>(next.ring.size());
    if (draining) {
        auto& head = next.ring.at(next.head);
        head->words_out += static_cast<uint32_t>(w.egress.words.size());
        if (w.egress.tma_store_drained) {
            head.reset();
            next.head = (next.head + 1) % size;
        }
    }
    if (reserved.has_value()) {
        auto& reg = next.ring.at(*reserved);
        if (reg.has_value()) throw SimError("internal: a TMA store register reserved twice");
        const auto& plan = *w.tma.walker;
        TmaStoreRegister fresh;
        fresh.command = *w.tma.frontend;
        fresh.sector_address = plan.sector_address;
        fresh.data.assign(config_.gmem_sector_bytes, 0);
        fresh.byte_enable = plan.store_byte_enable;
        fresh.source = *w.tma.a_read;
        reg = std::move(fresh);
    }
    if (fill.has_value()) {
        auto& reg = next.ring.at(fill->store_register);
        for (const auto& [offset, byte] : fill->bytes) reg->data.at(offset) = byte;
        ++reg->returned;
    }
    tma_store_registers_.set_next(std::move(next), "drive_tma_store_registers");
}

// The LSU command slots (admission order): every stage's progress, the
// releases, and issue's admission, in one calculation. A command released
// now frees its slot for an admission now (d).
void Machine::drive_lsu_commands(const Wires& w) {
    std::vector<uint64_t> drained; // loads whose assembly register's RF write was granted now
    for (uint32_t index = 0; index < staging_.size(); ++index) {
        if (w.rf_write.granted_to(AsyncWriter::assembly, index)) drained.push_back(*staging_[index].cur().owner);
    }
    const bool changed = !drained.empty() || w.egress.client == GmemClient::lsu ||
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
    for (const auto id : drained) ++command(id).chunks_done;
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
        const bool rf_drained = w.rf_write.granted_to(AsyncWriter::assembly, index);
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
    case Opcode::lds_uni:
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
    case Opcode::lds_uni:
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

std::vector<uint32_t> Machine::capture(const Warp& warp, uint8_t reg) const {
    sanitizer_.check_issue_capture(cycle_, warp.id, reg, false);
    return lane_values(warp, reg);
}

std::vector<uint32_t> Machine::capture_predicate(const Warp& warp, uint8_t predicate) const {
    sanitizer_.check_issue_capture(cycle_, warp.id, predicate, true);
    const auto& bits = warp.predicates.at(predicate);
    return std::vector<uint32_t>(bits.begin(), bits.end());
}

// A declared-uniform operand (rfc09 pin 3): hardware reads lane 0. A value
// that differs across lanes is UB; the sanitizer panics on it with the
// operand's name ("... is not warp-uniform").
uint32_t Machine::uniform_reg(
    const Warp& warp, const Instruction& inst, uint8_t reg, const std::string& name) const {
    const auto lanes = capture(warp, reg);
    sanitizer_.check_warp_uniform(site(warp, inst), lanes, name);
    return lanes[0];
}

// rfc12 section 3: lds.uni, probes and mbar operations are uniform operations;
// a partial active mask is a PANIC.
void Machine::panic_if_partial_mask(const Warp& warp, const Instruction& inst) const {
    if (warp.active_mask != full_mask()) {
        panic(warp, inst, opcode_name(inst.opcode) + " is a uniform operation under a partial active mask");
    }
}

uint32_t Machine::mbar_address_operand(const Warp& warp, const Instruction& inst) const {
    const auto address = uniform_reg(warp, inst, inst.src0, "mbar address");
    if (address % 8 != 0 || static_cast<uint64_t>(address) + 8 > shmem_.size()) {
        panic(warp, inst, "mbar address is not 8-byte aligned and in shmem");
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

uint32_t& Machine::rf(Warp& warp, uint32_t lane, uint32_t reg) {
    return warp.rf.at(static_cast<std::size_t>(lane) * kernel_.register_count + reg);
}

uint32_t Machine::rf(const Warp& warp, uint32_t lane, uint32_t reg) const {
    return warp.rf.at(static_cast<std::size_t>(lane) * kernel_.register_count + reg);
}

// ---------------------------------------------------------------------------
// SIMT shmem bank rounds (rfc09 pin 3: uniformity is declared, never
// detected). Every ACTIVE lane of an lds/sts is one access to bank
// (byte_address / 4) % shmem_banks, even when lanes name the same word; an
// inactive lane is no access at all (rfc12 section 3). Round r serves, in
// every bank, that bank's r-th active lane in ascending lane order, so rounds
// = the most active lanes on any one bank. lds.uni is the declared broadcast.
// ---------------------------------------------------------------------------

std::vector<std::vector<Machine::LaneAccess>> Machine::build_shmem_rounds(
    const Warp& warp, const Instruction& inst, bool load, const Wires& w) const {
    const auto bases = capture(warp, inst.src0);
    const auto data = load ? std::vector<uint32_t>(config_.lanes, 0) : capture(warp, inst.src1);
    std::vector<std::vector<LaneAccess>> lanes_by_bank(config_.shmem_banks);
    for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
        if ((warp.active_mask >> lane & 1) == 0) continue;
        const auto byte_address = shmem_address(warp, inst, bases[lane]);
        auto& bank_lanes = lanes_by_bank[(byte_address / 4) % config_.shmem_banks];
        if (!load) {
            panic_if_store_hits_live_mbar(warp, inst, byte_address, w);
            panic_if_word_already_stored(warp, inst, bank_lanes, byte_address);
        }
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

// lds.uni: hardware takes the address from lane 0 alone (as ldg.uni does) and
// makes one access; serve_simt_round fans the word out to every lane.
std::vector<std::vector<Machine::LaneAccess>> Machine::build_uniform_shmem_round(
    const Warp& warp, const Instruction& inst) const {
    const auto base = uniform_reg(warp, inst, inst.src0, "lds.uni base address");
    return {{LaneAccess{.lane = 0, .byte_address = shmem_address(warp, inst, base)}}};
}

// One lane's address from its captured base register value and the immediate.
uint32_t Machine::shmem_address(const Warp& warp, const Instruction& inst, uint32_t base) const {
    const auto address = static_cast<int64_t>(base) + inst.imm;
    if (address < 0 || address > std::numeric_limits<uint32_t>::max()) {
        panic(warp, inst, "shmem address arithmetic overflow");
    }
    const auto byte_address = static_cast<uint32_t>(address);
    if (byte_address % 4 != 0) panic(warp, inst, "unaligned 32-bit shmem access");
    if (static_cast<uint64_t>(byte_address) + 4 > shmem_.size()) {
        panic(warp, inst, "shmem access out of bounds");
    }
    return byte_address;
}

void Machine::panic_if_store_hits_live_mbar(
    const Warp& warp, const Instruction& inst, uint32_t byte_address, const Wires& w) const {
    const auto& slots = mbar_wait_slots_.cur();
    for (uint32_t index = 0; index < slots.size(); ++index) {
        const auto& wait_slot = slots[index];
        if (wait_slot.has_value() && !wait_slot_completed(w, index) &&
            byte_address >= wait_slot->byte_address && byte_address < wait_slot->byte_address + 8) {
            panic(warp, inst, "repurposing an mbar with a live waiter");
        }
    }
}

// Two lanes storing one word is UB. This is a stateless diagnostic, not
// hardware (the bank would simply take both, in lane order); it fires in both
// sanitizer builds, like its gmem twin in plan_lsu_command.
void Machine::panic_if_word_already_stored(const Warp& warp, const Instruction& inst,
    const std::vector<LaneAccess>& bank_lanes, uint32_t byte_address) const {
    for (const auto& earlier : bank_lanes) {
        if (earlier.byte_address == byte_address) panic(warp, inst, "multiple lanes write the same shmem word");
    }
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

std::vector<uint32_t> Machine::lane_values(const Warp& warp, uint8_t reg) const {
    std::vector<uint32_t> values(config_.lanes);
    for (uint32_t lane = 0; lane < config_.lanes; ++lane) values[lane] = rf(warp, lane, reg);
    return values;
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

// Each bank round's word addresses (the sanitizer's shadow of an access).
std::vector<std::vector<uint32_t>> Machine::round_addresses(const std::vector<std::vector<LaneAccess>>& rounds) {
    std::vector<std::vector<uint32_t>> addresses;
    for (const auto& round : rounds) {
        auto& words = addresses.emplace_back();
        for (const auto& access : round) words.push_back(static_cast<uint32_t>(access.byte_address));
    }
    return addresses;
}

// shmem bytes [address, address + size) as they stand after this cycle's
// clock edge: memory, plus the write rounds at the RAM now (they land at the
// edge). An access at the RAM in a command's accept cycle has REACHED it
// (rfc12 section 10), so this is what an engine command accepted now finds.
std::vector<uint8_t> Machine::shmem_after_edge(uint32_t address, uint32_t size, const Wires& w) const {
    std::vector<uint8_t> bytes(shmem_.begin() + address, shmem_.begin() + address + size);
    for (const auto& write : w.shmem.writes) {
        for (std::size_t byte = 0; byte < write.bytes.size(); ++byte) {
            const uint64_t at = static_cast<uint64_t>(write.address) + byte;
            if (at >= address && at < static_cast<uint64_t>(address) + size) bytes[at - address] = write.bytes[byte];
        }
    }
    return bytes;
}

// The descriptor the engine will read: memory as of this cycle's edge. I4
// (rfc12 section 11) replaced the overlay of in-flight stores this peek had:
// a write still travelling to the descriptor at accept is now a missing-fence
// / race PANIC (record_tma_accept), and the descriptor's read-ownership
// covers every write after it, so the engine reads exactly these bytes.
std::array<uint8_t, tma_descriptor_bytes> Machine::tma_descriptor_at_accept(uint32_t address, const Wires& w) const {
    const auto memory = shmem_after_edge(address, static_cast<uint32_t>(tma_descriptor_bytes), w);
    std::array<uint8_t, tma_descriptor_bytes> bytes{};
    std::copy(memory.begin(), memory.end(), bytes.begin());
    return bytes;
}

// The accept-time peek of a TMA's completion mbar (rfc12 section 8): its word
// as the RMW engine will leave it -- memory as of this cycle's edge, then the
// writes already granted and still travelling (the write pipelines' earlier
// stages, oldest first, then this cycle's grants: an RMW's write among them),
// then the RMW's computed update (its ALU stages and this cycle's update:
// its write is granted later). None when the peeked descriptor is invalid or
// its mbar is outside shmem (the engine's own descriptor read panics then).
// Sanitizer-only: the hardware reads no mbar at TMA start.
std::optional<uint64_t> Machine::pending_tma_completion_mbar(
    const std::array<uint8_t, tma_descriptor_bytes>& descriptor_bytes, const Wires& w) const {
    uint32_t mbar = 0;
    try {
        mbar = decode_tma_descriptor(descriptor_bytes.data(), descriptor_bytes.size()).completion_mbar;
    } catch (const SimError&) {
        return std::nullopt;
    }
    if (mbar % 8 != 0 || static_cast<uint64_t>(mbar) + 8 > shmem_.size()) return std::nullopt;
    auto bytes = shmem_after_edge(mbar, 8, w);
    const auto overlay_beat = [&](const std::optional<WriteBeat>& beat) {
        if (!beat.has_value()) return;
        for (const auto& write : beat->writes) {
            for (std::size_t byte = 0; byte < write.bytes.size(); ++byte) {
                const uint64_t at = static_cast<uint64_t>(write.address) + byte;
                if (at >= mbar && at < mbar + 8ull) bytes[at - mbar] = write.bytes[byte];
            }
        }
    };
    const auto& a_writes = a_write_pipe_.cur();
    const auto& b_writes = b_write_pipe_.cur();
    for (std::size_t stage = a_writes.size() - 1; stage-- > 0;) { // the last stage is at the RAM now
        overlay_beat(a_writes[stage]);
        overlay_beat(b_writes[stage]);
    }
    overlay_beat(w.shmem.a_write_beat);
    overlay_beat(w.shmem.b_write_beat);
    const auto& alu = rmw_.cur().alu;
    for (std::size_t stage = alu.size(); stage-- > 0;) overlay_beat(alu[stage]);
    overlay_beat(w.shmem.rmw_update);
    uint64_t word = 0;
    std::memcpy(&word, bytes.data(), sizeof(word));
    return word;
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

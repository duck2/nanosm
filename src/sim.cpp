#include "sim.hpp"
#include "mma_numeric.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <set>
#include <tuple>
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
    return is_tma_opcode(opcode);
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
    return is_mma_async(opcode) || opcode == Opcode::mma_commit;
}

bool is_acc_load(Opcode opcode) {
    return opcode == Opcode::acc_ld || opcode == Opcode::acc_ld_v8;
}

bool is_acc_transfer(Opcode opcode) {
    return is_acc_load(opcode) || opcode == Opcode::acc_st || opcode == Opcode::acc_st_v8;
}

// The SIMT shmem users: accesses (load / store slots), mbar probes (the probe
// register), mbar RMWs and uniform atomics (the RMW engine; an atomic also
// holds a load slot for its old value).
bool uses_shmem(Opcode opcode) {
    return is_shared_load(opcode) || opcode == Opcode::sts || is_mbar_rmw(opcode) ||
        is_mbar_predicate_writer(opcode) || is_atomic(opcode);
}

// trash/rfc12.md sections 3, 4 and 6: first the warp's own order -- a read
// waits for the warp's unserviced writes, a write (sts, an RMW, an atomic)
// for its unserviced reads and writes; reads may pass reads -- then the
// resources that hold the operation. So a SIMT RMW enters the SIMT RMW
// register (and later the engine, whose store pause would block them) only
// once its warp's own lds / sts are serviced. A uniform atomic (rfc13
// section 4) needs its RETURN load slot as well as the SIMT RMW register: it
// is admitted with both or not at all, so it never takes the store pause with
// nowhere to put its old value. An lds / sts with no active lane is absent:
// no slot, no order. (A uniform operation ignores the mask: it is never
// absent.)
std::optional<Refusal> shmem_refusal(Opcode opcode, const ResourceView& view) {
    if ((opcode == Opcode::lds || opcode == Opcode::sts) && view.no_active_lanes) return std::nullopt;
    const bool write = opcode == Opcode::sts || is_mbar_rmw(opcode) || is_atomic(opcode);
    if (view.unserviced_write || (write && view.unserviced_read)) return Refusal::shmem_ordered;
    if ((is_shared_load(opcode) || is_atomic(opcode)) && !view.load_slot_free) return Refusal::load_slots_full;
    if (opcode == Opcode::sts && !view.store_slot_free) return Refusal::store_slots_full;
    if (is_mbar_predicate_writer(opcode) && !view.probe_register_free) return Refusal::probe_register_busy;
    if ((is_mbar_rmw(opcode) || is_atomic(opcode)) && !view.simt_rmw_register_free) {
        return Refusal::simt_rmw_register_busy;
    }
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

// setp's and usetp's comparisons: unsigned 32-bit, the .s32 ones signed. An
// immediate arrives sign-extended from its 22-bit field either way.
bool compare_setp(Opcode opcode, uint32_t lhs, uint32_t rhs) {
    const auto signed_lhs = static_cast<int32_t>(lhs);
    const auto signed_rhs = static_cast<int32_t>(rhs);
    switch (opcode) {
    case Opcode::setp_eq:
    case Opcode::setpi_eq:
    case Opcode::usetp_eq:
    case Opcode::usetpi_eq: return lhs == rhs;
    case Opcode::setp_ne:
    case Opcode::setpi_ne:
    case Opcode::usetp_ne:
    case Opcode::usetpi_ne: return lhs != rhs;
    case Opcode::setp_lt:
    case Opcode::setpi_lt:
    case Opcode::usetp_lt:
    case Opcode::usetpi_lt: return lhs < rhs;
    case Opcode::setp_le:
    case Opcode::setpi_le:
    case Opcode::usetp_le:
    case Opcode::usetpi_le: return lhs <= rhs;
    case Opcode::setp_gt:
    case Opcode::setpi_gt:
    case Opcode::usetp_gt:
    case Opcode::usetpi_gt: return lhs > rhs;
    case Opcode::setp_ge:
    case Opcode::setpi_ge:
    case Opcode::usetp_ge:
    case Opcode::usetpi_ge: return lhs >= rhs;
    case Opcode::setp_lt_s32:
    case Opcode::setpi_lt_s32:
    case Opcode::usetp_lt_s32:
    case Opcode::usetpi_lt_s32: return signed_lhs < signed_rhs;
    case Opcode::setp_le_s32:
    case Opcode::setpi_le_s32:
    case Opcode::usetp_le_s32:
    case Opcode::usetpi_le_s32: return signed_lhs <= signed_rhs;
    case Opcode::setp_gt_s32:
    case Opcode::setpi_gt_s32:
    case Opcode::usetp_gt_s32:
    case Opcode::usetpi_gt_s32: return signed_lhs > signed_rhs;
    case Opcode::setp_ge_s32:
    case Opcode::setpi_ge_s32:
    case Opcode::usetp_ge_s32:
    case Opcode::usetpi_ge_s32: return signed_lhs >= signed_rhs;
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
    if (view.read_inhibited) return Refusal::rf_read_inhibited;
    // A tma's or an mma.async's later operand rounds own the URF's two read
    // ports (src/rf.hpp uniform_read_rounds): an instruction whose read round
    // has a uniform register (a tma or an mma.async included) would sample in
    // one of them.
    if (view.urf_ports_held && !plan.uniform_reads.empty()) return Refusal::urf_read_ports_held;
    // A uniform-register writer (or usetp) needs nothing else: its write port
    // is its own pipeline stage (src/rf.hpp; usetp's predicate goes to the
    // flip-flops at that stage), its reads the issue register's round.
    if (uses_urf_pipeline(opcode)) return std::nullopt;
    if (uses_arithmetic_pipeline(inst)) {
        if (is_sfu(opcode) && !view.sfu_free) return Refusal::sfu_busy;
        if (plan.write_reg.has_value() && !view.write_slot_free) return Refusal::rf_write_booked;
        if (plan.write_reg.has_value() && view.write_inhibited) return Refusal::rf_write_inhibited;
        return std::nullopt;
    }
    if (is_mbar_try_wait(opcode) && !view.wait_slot_free) return Refusal::wait_slots_full;
    if (is_tma(opcode) && !view.tma_queue_room) return Refusal::tma_queue_full;
    if (is_mma_command(opcode) && !view.mma_queue_room) return Refusal::mma_queue_full;
    if (is_acc_transfer(opcode) && !view.acc_engine_free) return Refusal::acc_engine_busy;
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
                throw SimError("tma.load / tma.store are unsupported by this device profile");
            }
        }
    }
    // Distinct sources beyond a bank's read ports are a property of the
    // instruction (src/rf.hpp): refused here, at kernel load, in both builds.
    check_rf_read_rounds(kernel_, config_);
    // An mma.async's shape is one the profile lists; an acc.ld / acc.st's
    // words per row are legal (src/layout.hpp).
    check_mma_shapes(kernel_, config_);
    // Every uniform register named is one the profile's URF has.
    check_uniform_registers(kernel_, config_);
    if (launch.active_warps == 0) throw SimError("active warp count is zero");
    // The SM's warp slots (core.warps): the scheduler's and the front end's
    // per-warp state exists for that many warps, whatever the RF could hold.
    if (launch.active_warps > config_.core_warps) {
        throw SimError("launch of " + std::to_string(launch.active_warps) + " warps exceeds the SM's " +
            std::to_string(config_.core_warps) + " warp slots (core.warps)");
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
        tmem_.assign(static_cast<std::size_t>(config_.tmem_fine_banks) * config_.tmem_words_per_fine_bank, 0);
        acc_engines_.poke(AccEngines{});
        acc_ld_pipe_.poke(AccLdPipeline(config_.tmem_port_depth + config_.tmem_read_latency_cycles));
        for (uint32_t index = 0; index < config_.shmem_load_slots; ++index) {
            load_slots_.emplace_back(registers_, "load slot " + std::to_string(index));
        }
        for (uint32_t index = 0; index < config_.shmem_store_slots; ++index) {
            store_slots_.emplace_back(registers_, "store slot " + std::to_string(index));
        }
        a_read_pipe_.poke(ReadPipeline(config_.shmem_port_a_depth + config_.shmem_read_latency_cycles));
        a_write_pipe_.poke(WritePipeline(config_.shmem_port_a_depth));
        b_write_pipe_.poke(WritePipeline(config_.shmem_port_a_depth));
        MmaPipeline mma;
        mma.reads.resize(config_.shmem_port_a_depth + config_.shmem_read_latency_cycles);
        mma_.poke(std::move(mma));
        RmwEngine engine;
        engine.alu.resize(config_.shmem_rmw_compute_cycles);
        rmw_.poke(std::move(engine));
        RfWritePolicy policy;
        policy.owed.assign(config_.rf_banks, 0);
        policy.cursor.assign(config_.rf_banks, 0);
        rf_write_policy_.poke(std::move(policy));
        mbar_wait_slots_.poke(WaitSlots(config_.mbar_wait_slots));
        RfReadPolicy read_policy;
        read_policy.owed.assign(config_.rf_banks, 0);
        read_policy.cursor.assign(config_.rf_banks, 0);
        rf_read_policy_.poke(std::move(read_policy));
        issue_read_banks_.poke(std::vector<uint8_t>(config_.rf_banks, 0));
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
        tma_walker_start_pipe_.poke(TmaWalkerStartPipeline(config_.tma_walker_start_cycles));
        warps_.reserve(launch.active_warps);
        for (uint32_t id = 0; id < launch.active_warps; ++id) {
            Warp warp;
            warp.id = id;
            warp.active_mask = full_mask();
            warp.rf.resize(static_cast<std::size_t>(config_.lanes) * kernel_.register_count, 0);
            for (auto& predicate : warp.predicates) predicate.resize(config_.lanes, 0);
            warp.urf.resize(config_.urf_registers, 0);
            warps_.push_back(std::move(warp));
        }
        urf_pipe_.poke(UrfWritePipeline(config_.urf_issue_to_port_cycles + config_.urf_write_depth_cycles));
        tma_read_pairs_.poke(TmaReadPairs(config_.urf_read_latency_cycles));
        // Every slot's FIFO empty and nothing in flight: launched warps fetch
        // from pc 0; a slot not launched stays halted.
        std::vector<WarpFrontEnd> front_end(config_.core_warps);
        for (uint32_t slot = launch.active_warps; slot < config_.core_warps; ++slot) front_end[slot].halted = true;
        front_end_.poke(std::move(front_end));
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
    trace_->set_mma_shapes(config_.mma_shapes);
    apply_image(shmem_, launch.shmem_image, "shmem");
    apply_image(gmem_, launch.gmem_image, "gmem");
    // The mbars the launch declares (initialized by the host in the shmem
    // image, e.g. FA's): live from launch for the sanitizer's C10 opacity,
    // like a SIMT mbar.init's (rfc12 section 11). A declaration is metadata;
    // the machine only checks that it names an mbar slot.
    for (const auto address : launch.mbars) {
        if (address % 8 != 0 || static_cast<uint64_t>(address) + 8 > shmem_.size()) {
            throw SimError("launch declares an mbar at " + std::to_string(address) +
                ", which is not 8-byte aligned in shmem");
        }
    }
    sanitizer_.record_launch_mbars(launch.mbars);
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
//    it. The declared ones:
//      (a) None through the RF, the URF or the predicate file (slice I6a, src/rf.hpp):
//          a write lands at the clock edge of its RAM cycle, after every
//          read of that cycle (READ_FIRST); no same-edge forwarding.
//      (b) None from a completion to issue: a depbar clear, an mbar wait
//          slot's completion (wake, immediate success, timeout), a probe's
//          return reach issue through their registers, the next cycle.
//      (c) Issue looks ahead: an instruction accepted at t occupies its
//          resources from t+1, so issue sees the space this cycle frees
//          (queue entries, slots freed by this cycle's grants, the SIMT RMW
//          register emptied by an RMW start, assembly registers) and the
//          warp's shmem accesses as this cycle's grants leave them (rfc12
//          section 4) -- less the space the dispatch pipeline's
//          instructions will take (dispatch_claims). Not the load slots
//          (an lds's or an atomic's return slot): issue counts the slots
//          empty in Q, so a slot whose RF write is granted at t is free for
//          a new admission from t + 1 (HUMAN 11-20, trash/slot-return-cut-
//          2026-11-20.md). The RF write decision's fairness inhibit, the
//          LSU command slot and assembly register its grant releases, and
//          the acc.ld engine its grant lets advance are still seen at t.
//      (d) Same-cycle space reuse: a register drained at t may be refilled at
//          t (the assembly register, a load slot whose RF write is granted
//          at t -- by dispatch, which issue admitted earlier against the
//          slots' Q -- the TMA sector / store registers, the MMA event register,
//          an LSU slot or acc slot reallocated by dispatch). The refill's
//          data is visible at t+1; nothing passes through.
//      (e) A dispatched stg.v8 may make its first RF read beat in its
//          dispatch cycle.
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
//      (i) None from issue into fetch (HUMAN 10-18, trash/front-end-
//          2026-10-18.md): the fetch decision reads registers only (the
//          FIFOs' valid words, the read in flight, halted, the issue
//          register's branch). A slot an issue frees at t is seen by fetch
//          at t + 1; a branch accepted at t blocks its warp's fetch from
//          t + 1. Issue sees the FIFO heads' Q: a word landing at t issues
//          from t + 1.
// 2. DRIVERS. Each register's next value, from Q and the wires. They run in
//    any order (set_driver_shuffle_seed() permutes them every cycle).
// 3. CLOCK EDGE. Every register takes D; shmem takes the write rounds that
//    reached the RAM this cycle (A-write, then port-B write).
// ===========================================================================
void Machine::tick() {
    if (done()) return;
    sanitizer_.record_cycle_start();
    Wires w;

    // The front end, from Q only (i): the fetch grant, the word at the
    // instruction memory's output, the branch in the issue register resolved.
    w.fetch = decide_fetch();
    w.fetched = instruction_memory_output();
    w.branch = resolve_branch();

    // The RF write path, from Q and before any SIMT candidate: the one RF
    // write decision (the ready async holders share this cycle's unbooked
    // write ports; banks whose service is owed inhibit new SIMT bookings),
    // the write port (booked results and granted holders enter the write
    // pipeline) and its RAM stage (the writes landing at this cycle's edge,
    // and their completions).
    w.rf_write = decide_rf_writes();
    rf_write_port(w);
    rf_ram_stage(w);
    // The URF write pipeline's write port and RAM stage, from Q: nothing
    // contends for them (one writer per stage by construction).
    urf_ram_stage(w);

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

    // The TMA operand read's end: a tma's last words at the URF output now
    // push its command into the TMA queue. Then the dispatch pipeline's end:
    // a memory / engine / mbar instruction whose operands arrive now goes to
    // its unit.
    arrive_tma_operands(w);
    dispatch_issued(w);
    // The async RF readers (acc.st beats, the LSU's store-data read) take the
    // read ports the issue register's instruction leaves free now, from Q,
    // before issue.
    w.rf_read = decide_rf_reads(w);

    // The scheduler. Its write booking is in a future cycle and its read
    // round is the next cycle's (the issue register); it updates the owed
    // bits' next state.
    decide_issue(w);

    // The front ends' next state: issue's pop (and halt), the landing word,
    // the redirect, the fetch grant -- one calculation per warp.
    step_front_end(w);

    // The RF read stage: the issue register's round samples now (vector and
    // uniform registers). Then the URF write pipeline's next state: its
    // sampled write, the uniform ALU and r2ur's results, issue's new write.
    step_rf_reads(w);
    step_urf(w);

    run_drivers(w);

    // Clock edge: the registers, then the memories' writes at their RAM now.
    registers_.commit();
    for (const auto& write : w.shmem.writes) {
        std::memcpy(shmem_.data() + write.address, write.bytes.data(), write.bytes.size());
    }
    for (const auto& [word, value] : w.tmem_writes) tmem_[word] = value;
    land_rf_writes(w);
    land_urf_write(w);
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
    // Every warp halted and no instruction-memory read in flight (its word
    // would be dropped, but the read is still under way).
    const auto& front_end = front_end_.cur();
    const bool warps_halted = std::all_of(front_end.begin(), front_end.end(), [](const WarpFrontEnd& warp) {
        return warp.halted;
    }) && !fetch_read_.cur().has_value();
    const auto& waits = mbar_wait_slots_.cur();
    const bool wait_slots_empty = std::all_of(
        waits.begin(), waits.end(), [](const auto& slot) { return !slot.has_value(); });
    const auto& tma = tma_.cur();
    const auto& mma = mma_.cur();
    const auto& acc = acc_engines_.cur();
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
    return warps_halted && wait_slots_empty && shmem_idle && drained(acc_ld_pipe_.cur()) &&
        !tma_mbar_event_.cur().has_value() && !mma_mbar_event_.cur().has_value() && gmem_queue_.empty() &&
        !gmem_bus_.has_value() && pending_write_acks_.empty() && lsu_commands_.cur().empty() &&
        read_records_.cur().empty() && ingress_.cur().beats.empty() && egress_fifo_.cur().empty() &&
        !tma_sector_.cur().request.has_value() && tma_store_registers_.cur().held() == 0 &&
        tma.commands.empty() && tma.returns.empty() && !tma.event_command.has_value() &&
        tma.credits_used == 0 && drained(tma_walker_start_pipe_.cur()) && empty(urf_read_rounds_) &&
        drained(tma_read_pairs_.cur()) &&
        empty(tma_holding_) && writebacks_.cur().empty() && arithmetic_reads_.empty() &&
        dispatch_pipe_.cur().empty() && lsu_read_pipe_.cur().empty() && drained(urf_pipe_.cur()) &&
        mma.queue.empty() && mma.results.empty() && !mma.fetch.has_value() && drained(mma.reads) &&
        mma.arriving.empty() &&
        !acc[0].has_value() && !acc[1].has_value();
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

void Machine::test_set_fetch_pc(uint32_t warp, uint32_t pc) {
    auto front_end = front_end_.cur();
    front_end.at(warp).fetch_pc = pc;
    front_end_.poke(std::move(front_end));
}

uint32_t Machine::read_reg(
    uint32_t warp, uint32_t lane, uint32_t reg) const {
    if (warp >= warps_.size() || lane >= config_.lanes || reg >= kernel_.register_count) {
        throw SimError("invalid host RF read");
    }
    return rf(warps_[warp], lane, reg);
}

uint32_t Machine::read_ureg(uint32_t warp, uint32_t reg) const {
    if (warp >= warps_.size() || reg >= config_.urf_registers) throw SimError("invalid host URF read");
    return warps_[warp].urf[reg];
}

// ===========================================================================
// The front end (trash/front-end-2026-10-18.md, two-word packets since
// trash/front-end-packets-2026-10-18.md; the cycles in src/rf.hpp). Every
// decision here reads registers only (tick() (i)).
// ===========================================================================

// One instruction-memory read per cycle: one packet, the two words at an
// aligned 16-byte address. A warp asks for one when it has not halted, its
// next pc is known (no branch of its in the issue register) and its FIFO has
// room: valid packets + its read in flight < the FIFO's depth (a read in
// flight owns its entry until it lands or is dropped, so two reads never
// count on one free entry, and a landing always finds room).
// Round-robin over the askers (round_robin_grant, its own cursor). Whether
// the warp could issue does not matter: prefetch goes on through stall
// counts, depbars, probes, wait slots and refused heads.
Machine::FetchGrant Machine::decide_fetch() const {
    const auto& front_end = front_end_.cur();
    const auto& read = fetch_read_.cur();
    const auto& branch = issue_branch_.cur();
    std::vector<uint8_t> has_space(front_end.size(), 0);
    for (uint32_t warp = 0; warp < front_end.size(); ++warp) {
        const auto& slot = front_end[warp];
        const std::size_t in_flight = read.has_value() && read->warp == warp ? 1 : 0;
        const bool branch_unresolved = branch.has_value() && branch->warp == warp;
        has_space[warp] = !slot.halted && !branch_unresolved &&
            slot.fifo.size() + in_flight < config_.core_instruction_buffer_packets ? 1 : 0;
    }
    const auto choice = round_robin_grant(has_space, fetch_cursor_.cur(), 1);
    FetchGrant grant;
    grant.cursor_after = choice.cursor_after;
    if (!choice.granted.empty()) {
        grant.warp = choice.granted.front();
        grant.pc = front_end[*grant.warp].fetch_pc;
    }
    return grant;
}

// The packet at the instruction memory's output now, for the read in
// flight: the kernel's two words at its address, zero past the kernel's
// words (whatever the memory holds there, no such word may have an effect
// unless it reaches issue, which reports it: head_instruction). Whether it
// enters its warp's FIFO is step_front_end's choice.
std::optional<Machine::FetchedPacket> Machine::instruction_memory_output() const {
    const auto& read = fetch_read_.cur();
    if (!read.has_value()) return std::nullopt;
    FetchedPacket packet;
    packet.warp = read->warp;
    for (uint32_t half = 0; half < 2; ++half) {
        const auto index = read->pc / 8 + half;
        packet.words[half] = index < kernel_.words.size() ? kernel_.words[index] : 0;
    }
    return packet;
}

// The branch in the issue register (accepted last cycle) resolves now
// (issue_to_branch_resolve). bra and bra.uni @PT are taken and bra.uni @!PT
// is not; any other bra.uni reads its predicate from the file now, like an
// operand through the issue register: lane 0 (a declared-uniform operand;
// the sanitizer checks the other lanes, and the active mask -- bra.uni is a
// uniform operation in every form, backlog D12.1). Taken: the one target
// adder, the branch's pc + 8 + its offset.
std::optional<Machine::BranchResolution> Machine::resolve_branch() const {
    const auto& branch = issue_branch_.cur();
    if (!branch.has_value()) return std::nullopt;
    const auto& inst = branch->instruction;
    const auto& warp = warps_.at(branch->warp);
    const IssueSite here{cycle_, branch->warp, branch->pc, inst};
    if (inst.opcode == Opcode::bra_uni) sanitizer_.check_uniform_mask(here, warp.active_mask, full_mask());
    bool take = true;
    if (inst.opcode == Opcode::bra_uni && inst.control.predicate != 7) {
        const auto lanes = capture_predicate(warp, inst.control.predicate);
        sanitizer_.check_warp_uniform(here, lanes, "bra.uni predicate");
        take = (lanes[0] != 0) != inst.control.predicate_invert;
    } else if (inst.opcode == Opcode::bra_uni && inst.control.predicate_invert) {
        take = false; // @!PT
    }
    BranchResolution resolution;
    resolution.warp = branch->warp;
    resolution.taken = take;
    if (!take) return resolution;
    const auto target = static_cast<int64_t>(branch->pc) + 8 + inst.imm;
    if (inst.imm % 8 != 0 || target < 0 || static_cast<uint64_t>(target / 8) >= kernel_.words.size()) {
        throw_issue_panic(here, "branch target is outside the kernel");
    }
    resolution.target = static_cast<uint32_t>(target);
    return resolution;
}

// Every warp's front end after this cycle, in one calculation per warp
// (src/rf.hpp for the cycles), from Q, this cycle's front-end wires and
// issue's acceptance:
//   - issue: issue accepted the warp's next instruction (captured into the
//     issue register at this edge); pc moves to the next word, and the head
//     packet pops if that instruction was its second half;
//   - landing: the packet at the instruction memory's output joins the FIFO
//     behind the valid packets ([A, B] with A's last word accepted and C
//     landing -> [B, C]), unless a redirect or a halt clears the FIFO now, or
//     the warp halted earlier (its read was granted in the halt's cycle);
//   - halt accepted now: the FIFO is cleared and the warp stops;
//   - redirect (a taken branch resolves now): the FIFO is cleared, the packet
//     landing now is dropped, pc takes the target and fetch pc the target's
//     packet (the target rounded down to 16: a target in a packet's second
//     half starts there). A not-taken branch changes nothing: its buffered
//     fall-through stays;
//   - fetch granted now: the fetch pc moves to the next packet.
// No fetch is granted to a warp whose branch is in the issue register, so a
// redirect and a grant never meet in one warp; a halt and a grant may (that
// read is dropped as it lands). The credit check guarantees the FIFO never
// holds more than its depth, and while the warp runs its fetch pc is always
// pc rounded down to 16 + 16 x (valid packets + its read in flight); either
// failing is an internal error. The trace counts the fetch and the fetched
// words that never issue: those a redirect or a halt clears or drops, and the
// first half of a packet fetched for a second-half target, skipped as it
// lands.
void Machine::step_front_end(Wires& w) {
    const auto& cur = front_end_.cur();
    w.front_end_next = cur;
    // Fetched words not yet issued or counted: every valid packet's two, less
    // the head's first half when pc has moved past it (issued, or skipped and
    // counted as it landed).
    const auto unissued_words = [](const WarpFrontEnd& slot) -> uint64_t {
        if (slot.fifo.empty()) return 0;
        return 2 * slot.fifo.size() - (slot.pc % 16 == 8 ? 1 : 0);
    };
    uint64_t discarded = 0;
    for (uint32_t warp = 0; warp < cur.size(); ++warp) {
        auto& slot = w.front_end_next[warp];
        const bool issued = w.issue.popped == warp;
        const bool halting = issued && w.issue.halt;
        const bool redirect = w.branch.has_value() && w.branch->warp == warp && w.branch->taken;
        const bool landing = w.fetched.has_value() && w.fetched->warp == warp;
        if (issued) {
            if (slot.pc % 16 == 8) slot.fifo.erase(slot.fifo.begin());
            slot.pc += 8;
        }
        if (halting || redirect) {
            discarded += unissued_words(slot) + (landing ? 2 : 0);
            slot.fifo.clear();
        } else if (landing && cur[warp].halted) {
            discarded += 2;
        } else if (landing) {
            // Into an empty FIFO with pc at a second half: the packet was
            // fetched for a second-half branch target; its first half never
            // issues.
            if (slot.fifo.empty() && slot.pc % 16 == 8) ++discarded;
            slot.fifo.push_back(w.fetched->words);
        }
        if (halting) slot.halted = true;
        if (redirect) {
            slot.pc = w.branch->target;
            slot.fetch_pc = w.branch->target / 16 * 16;
        }
        if (w.fetch.warp == warp) slot.fetch_pc += 16;
        if (slot.fifo.size() > config_.core_instruction_buffer_packets) {
            throw SimError("internal: warp " + std::to_string(warp) + "'s instruction FIFO overflowed at cycle " +
                std::to_string(cycle_));
        }
        if (!slot.halted) {
            const auto packets = static_cast<uint32_t>(slot.fifo.size()) + (w.fetch.warp == warp ? 1u : 0u);
            if (slot.fetch_pc != slot.pc / 16 * 16 + 16 * packets) {
                throw SimError("internal: warp " + std::to_string(warp) + "'s front end is inconsistent at cycle " +
                    std::to_string(cycle_) + ": fetch pc " + std::to_string(slot.fetch_pc) + ", pc " +
                    std::to_string(slot.pc) + ", " + std::to_string(packets) + " packets held or in flight");
            }
        }
    }
    if (w.fetch.warp.has_value()) trace_->fetch(cycle_, *w.fetch.warp, w.fetch.pc);
    if (discarded != 0) trace_->fetches_discarded(discarded);
}

// ===========================================================================
// The RF ports: the reservation calendars, the RF read and write decisions,
// the write port and the write pipeline (trash/rfc10.md sections 3 and 5;
// every cycle number follows src/rf.hpp)
// ===========================================================================

// Fixed issue never contests the current cycle's RF ports (the issue
// register): its read round samples at issue + 1, its fixed write is booked
// in the write calendar at issue + issue_to_port. What the current cycle
// leaves goes to the asynchronous clients, decided from Q before issue:
//   READS   decide_rf_reads (trash/rfc14.md sections 4 and 5): the issue
//           register's instruction owns the read ports its round samples
//           now; acc.st beats and the LSU's store-data read share the rest
//           by round-robin. A denied reader whose bank's read_owed bit is
//           set inhibits the acceptance of new SIMT reads of that bank, so
//           the bank's ports are free one cycle later (the mirrored rule).
//   WRITES  decide_rf_writes: load slots, the LSU's assembly register and the
//           acc.ld pipelines share the current write port's unbooked ports;
//           a denied writer whose bank's async_owed bit is set inhibits new
//           fixed write bookings to that bank.
// A read SAMPLES the RF at its cycle and has its operand Lr cycles later; a
// write PRESENTED at the port lands at the clock edge Wd cycles later and is
// visible to samples from the next cycle (no same-edge forwarding).

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

bool Machine::RfReadDecision::granted_to(RfReader reader, uint32_t index) const {
    for (std::size_t holder = 0; holder < holders.size(); ++holder) {
        if (granted[holder] != 0 && holders[holder]->reader == reader && holders[holder]->index == index) return true;
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
    if (const auto& output = acc_ld_pipe_.cur().back(); output.has_value()) {
        holders.push_back(AsyncWrite{AsyncWriter::acc_ld, 0, output->warp, output->reg});
    } else {
        holders.emplace_back();
    }
    return holders;
}

// The RF write decision (trash/rfc10.md section 3, "make an opening, then
// choose a writer"): one per RF bank, from Q only, before issue.
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
        writeback.reason = slot.engine_filled ? "atom" : "lds"; // an atomic's old value, or an lds
        writeback.visible_regs = {write.reg};
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
            // rD..rD+7 count as in flight until the last write lands.
            const uint32_t width = vector ? 8 : 1;
            for (uint32_t j = 0; j < width; ++j) writeback.visible_regs.push_back(static_cast<uint8_t>(command.reg + j));
            writeback.clear_depbar = command.depbar;
            writeback.release_lsu_command = command.id;
        }
        probe(LsuEdge::rf_write_beat, command.id, write.reg, config_.lanes * 4);
        return writeback;
    }
    case AsyncWriter::acc_ld: {
        const auto& beat = *acc_ld_pipe_.cur().back();
        writeback.values = beat.values;
        writeback.reason = "acc.ld";
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

// An LSU command ends: its slot and assembly register are free for a new
// owner this same cycle (d).
void Machine::release_lsu_command(Wires& w, uint64_t id) {
    const auto* command = find_lsu_command(id);
    if (command == nullptr) throw SimError("internal release of an unknown LSU command");
    trace_->lsu_command_end(cycle_, id, lsu_phase_detail(*command));
    lsu_phases_.erase(id);
    w.released_lsu_commands.push_back(id);
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
        if (decision.holders[holder]->writer == AsyncWriter::assembly && ++lsu_writes > 1) {
            throw SimError("internal: more than one LSU RF write at cycle " + std::to_string(cycle_));
        }
        auto writeback = async_writeback(*decision.holders[holder]);
        use_port(writeback);
        if (writeback.release_lsu_command.has_value()) release_lsu_command(w, *writeback.release_lsu_command);
        w.new_writebacks.push_back(std::move(writeback));
    }
    count_rf_write_waits(decision);
    write_reservations_.erase(cycle_);
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

// The clock edge's RF and predicate writes (after registers_.commit()).
void Machine::land_rf_writes(const Wires& w) {
    for (const auto& writeback : w.rf_landings) {
        auto& warp = warps_.at(writeback.warp);
        if (writeback.values.size() != config_.lanes) throw SimError("internal writeback lane count mismatch");
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            if (writeback.lane_enables.has_value() && (*writeback.lane_enables >> lane & 1) == 0) continue;
            if (writeback.dst.has_value()) rf(warp, lane, *writeback.dst) = writeback.values[lane];
            if (writeback.predicate_dst.has_value()) {
                warp.predicates[*writeback.predicate_dst][lane] = writeback.values[lane] != 0;
            }
        }
    }
}

// The async RF readers' decision (trash/rfc10.md section 3, "RF reads: the
// same rule, mirrored"; trash/rfc14.md section 5), per RF bank, from Q
// before issue:
//   read_ready   = an async reader wants the bank now
//   read_grant   = read_ready AND the issue register's round leaves one of
//                  the bank's read ports free (simt_read_now: its bank use)
//   inhibit_simt = read_ready AND NOT read_grant AND read_owed
// A candidate instruction whose round uses an inhibited bank is not accepted
// this cycle (resource_view), so the issue register leaves that bank free
// next cycle: the hole the denied reader is owed. Reads already accepted are
// never stopped. The readers granted (round-robin from the bank's cursor)
// sample the RF now: an acc.st beat is at the RF output Lr later and at the
// tmem RAM tmem.port_depth after that; the LSU's beat fills its assembly
// register Lr later. A denied
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
        const auto choice = round_robin_grant(ready, policy.cursor.at(bank), free_ports);
        for (const auto holder : choice.granted) decision.granted[holder] = 1;
        auto& state = decision.banks[bank];
        state.ready = std::find(ready.begin(), ready.end(), 1) != ready.end();
        state.granted = !choice.granted.empty();
        state.inhibit_simt = state.ready && !state.granted && policy.owed.at(bank) != 0;
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
        auto values = lane_values(warp, read->reg);
        if (read->reader == RfReader::acc_st) {
            sanitizer_.record_async_reader_done(warp.id, read->reg, AsyncReader::acc_st);
            w.acc_st_read = AccStBeat{store_engine->next_column,
                cycle_ + config_.rf_read_latency_cycles + config_.tmem_port_depth, std::move(values)};
        } else {
            sanitizer_.record_async_reader_done(warp.id, read->reg, AsyncReader::stg_v8);
            probe(LsuEdge::rf_read_beat, read->lsu_command, read->reg, config_.lanes * 4);
            w.lsu_rf_read.command = read->lsu_command;
            w.lsu_rf_read.assembly = read->assembly;
            w.lsu_rf_read.values = std::move(values);
            w.lsu_rf_read.arrive_cycle = cycle_ + config_.rf_read_latency_cycles;
        }
    }
    return decision;
}

// ---- depbar: completions are recorded as they happen (the RF write
// pipeline's RAM stage, a store's last ack, an acc.st's last tmem write, an
// MMA's operands consumed); drive_depbar applies them and the admission.
// Issue reads the counters' Q: a completion presented at t is seen at t + 1.
// depbar_after_completions is the counter's D so far (underflow and trace). ----

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
// shmem.port_a_depth (D) cycles later -- the same D on all four services, so
// the RAM sees rounds in grant order -- and a read's words reach the port
// output shmem.read_latency_cycles (Lb) cycles after its sample. An RMW's
// update then crosses the RMW engine's shmem.rmw_compute_cycles (C) ALU
// stages before its write is granted. A read and a write of one word at the
// RAM in one cycle: READ_FIRST (the write lands at the clock edge). The MMA's
// port-B read pipeline lives in the MMA engine's register (advance_mma).
//
// The RMW engine (rfc12 section 6): one operation at a time, each one
// coherent 8-byte mbar RMW (both words in one read round, one write round)
// or one uniform atomic (trash/rfc13.md: one 4-byte word, read and written
// the same way). Its requester waits in a register (the SIMT RMW register,
// the TMA's or the MMA's event register). One RMW, with D, Lb and C as above:
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
        // The predicate file (flip-flops) takes it at this cycle's edge; the
        // warp issues from the next cycle (probe_in_flight).
        Writeback predicate;
        predicate.port_cycle = cycle_;
        predicate.land_cycle = cycle_;
        predicate.warp = warp.id;
        predicate.predicate_dst = request.predicate_dst;
        predicate.values.assign(config_.lanes, complete ? 1 : 0);
        predicate.reason = "mbar.test_wait";
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
    return transaction.has_value() && transaction->request.next_round < transaction->request.rounds;
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
    const auto rounds = request.rounds;
    const auto command = request.run.command;
    if (start) {
        trace_->shmem_begin(ShmemService::a_read, cycle_, tma_port_label(static_cast<int>(request.kind)),
            "command=" + std::to_string(command) + ", rounds=" + std::to_string(rounds),
            TraceCategory::tma, command);
        if (request.kind == TmaPortKind::store_read) trace_->tma_data_port(rounds);
    }
    trace_->shmem_grant(cycle_, ShmemService::a_read, ShmemClient::tma);
    if (request.next_round + 1 == rounds) trace_->shmem_end(ShmemService::a_read, cycle_ + 1);
    ReadBeat beat;
    const bool store = request.kind == TmaPortKind::store_read;
    beat.client = store ? ReadClient::tma_store : ReadClient::tma_frontend;
    beat.slot = store ? store_register : 0;
    beat.round = request.next_round;
    beat.addresses = tma_round_addresses(request, tma_round_plan(request).at(request.next_round));
    // (S) the words a swizzle64 store reads, against their writer's row spacing
    if (store) sanitizer_.check_tma_store_source(cycle_, command, beat.addresses);
    return beat;
}

// The distinct words one TMA round touches: at most one per bank (checked).
std::vector<uint32_t> Machine::tma_round_addresses(
    const TmaPortRequest& request, const std::vector<TmaPortRequest::Piece>& pieces) const {
    std::vector<uint32_t> addresses;
    for (const auto& piece : pieces) {
        const uint32_t address = tma_piece_address(request.run, piece);
        if (static_cast<uint64_t>(address) + request.run.element_bytes > shmem_.size()) {
            throw SimError("tma port service is outside shmem");
        }
        const uint32_t word = address / 4 * 4;
        if (std::find(addresses.begin(), addresses.end(), word) == addresses.end()) addresses.push_back(word);
    }
    if (addresses.size() > config_.shmem_banks) throw SimError("internal: a tma round reads more words than banks");
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
        ? TmaReadTransaction{*w.tma.a_read, 0, shmem.tma_store_reserved.value_or(0), {}}
        : *tma_read_txn_.cur();
    auto& request = transaction.request;
    shmem.tma_read_changed = true;
    if (granted) ++request.next_round;
    if (request.kind == TmaPortKind::store_read) {
        if (request.next_round == request.rounds) {
            shmem.tma_read_after.reset(); // A-read is released at the last round's grant
        } else {
            shmem.tma_read_after = std::move(transaction);
        }
        return;
    }
    if (returned) {
        // The descriptor's words into the frontend latch: element k is bytes
        // 4k .. 4k + 3.
        const auto& beat = *shmem.delivered;
        const auto plan = tma_round_plan(request);
        for (const auto& piece : plan.at(beat.round)) {
            const uint32_t address = tma_piece_address(request.run, piece);
            const auto at = std::find(beat.addresses.begin(), beat.addresses.end(), address / 4 * 4);
            const auto word = beat.words.at(static_cast<std::size_t>(at - beat.addresses.begin()));
            std::memcpy(transaction.latch.data() + 4 * piece.element, &word, 4);
        }
        ++transaction.returned;
    }
    if (transaction.returned == request.rounds) {
        shmem.tma_read_done = std::move(transaction);
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
    fill.command = reg->run.command;
    // The store-source read's request as its rounds were planned: the
    // register's run (a store read: whole words, no rotation rule).
    TmaPortRequest source;
    source.kind = TmaPortKind::store_read;
    source.run = reg->run;
    const uint32_t bytes = reg->run.element_bytes;
    const uint32_t first = run_sector_offset(reg->run);
    const auto plan = tma_round_plan(source);
    for (const auto& piece : plan.at(beat.round)) {
        const uint32_t address = tma_piece_address(reg->run, piece);
        const auto at = std::find(beat.addresses.begin(), beat.addresses.end(), address / 4 * 4);
        const auto word = beat.words.at(static_cast<std::size_t>(at - beat.addresses.begin()));
        const uint32_t offset = first + uint32_t{piece.element} * bytes;
        for (uint32_t byte = 0; byte < bytes; ++byte) {
            fill.bytes.emplace_back(offset + byte, static_cast<uint8_t>(word >> (8 * (address % 4 + byte))));
        }
    }
    if (fill.bytes.size() > config_.gmem_sector_bytes) throw SimError("internal: a tma store round fills more than a sector");
    fill.last = reg->returned + 1 == reg->rounds;
    probe(LsuEdge::tma_store_round, reg->run.command, beat.round, static_cast<uint32_t>(4 * beat.addresses.size()));
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
        const auto& request = rmw_.cur().active->request;
        trace_->shmem_grant(cycle_, ShmemService::a_write, ShmemClient::rmw);
        if (is_atomic_operation(request.operation)) {
            trace_->atomic_rmw_end(cycle_ + 1);
            // C10 (sanitizer): an ordinary atomic may not write a live mbar --
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
        // sanitizer's C10 PANICs (mbar.init over a live mbar). The machine
        // just writes the new state: a waiter is not woken by it (backlog
        // D12.1).
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

// A uniform atomic's update (trash/rfc13.md section 3), on signed int32: the
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

// One placement round: its bytes from the TMA sector register's Q (filled in
// earlier cycles), or zeros for an OOB fill. The last round carries the
// placement to the RAM, where its load write ends.
Machine::WriteBeat Machine::placement_round(const TmaPortRequest& request, bool start) {
    const auto rounds = request.rounds;
    const auto& run = request.run;
    if (rounds == 0 || request.next_round >= rounds) {
        throw SimError("internal zero-round tma placement");
    }
    if (start) {
        trace_->shmem_begin(ShmemService::b_write, cycle_, tma_port_label(static_cast<int>(request.kind)),
            "command=" + std::to_string(run.command) + ", rounds=" + std::to_string(rounds),
            TraceCategory::tma, run.command);
        trace_->tma_data_port(rounds);
    }
    const auto& sector = tma_sector_.cur();
    if (!run.zero_fill && (!sector.complete || sector.sector_address != run_sector_address(run))) {
        throw SimError("internal: TMA placement started before its sector register held its sector");
    }
    // Each piece: its element's bytes from the sector register (the run's
    // offset in the sector plus element x size), or zeros.
    WriteBeat beat;
    uint32_t bytes = 0;
    const uint32_t size = run.element_bytes;
    const uint32_t first = run.zero_fill ? 0 : run_sector_offset(run);
    const auto plan = tma_round_plan(request);
    for (const auto& piece : plan.at(request.next_round)) {
        const uint32_t address = tma_piece_address(run, piece);
        if (static_cast<uint64_t>(address) + size > shmem_.size()) {
            throw SimError("tma port service is outside shmem");
        }
        std::vector<uint8_t> data(size, 0);
        if (!run.zero_fill) {
            const uint8_t* source = sector.data.data() + first + uint32_t{piece.element} * size;
            std::copy(source, source + size, data.begin());
        }
        beat.writes.push_back({address, std::move(data)});
        bytes += size;
    }
    if (bytes > 4 * config_.shmem_banks) throw SimError("internal: a tma placement round writes more than its banks");
    // C10 (sanitizer): a placement round must not write a live mbar.
    std::vector<std::pair<uint32_t, uint32_t>> written;
    for (const auto& write : beat.writes) {
        written.emplace_back(write.address, write.address + static_cast<uint32_t>(write.bytes.size()));
    }
    sanitizer_.check_live_mbar_write(cycle_, "tma load #" + std::to_string(run.command) + " placement", written);
    probe(LsuEdge::tma_place_round, request.sector_request, request.next_round, bytes);
    trace_->shmem_grant(cycle_, ShmemService::b_write, ShmemClient::tma);
    if (request.next_round + 1 == rounds) {
        beat.placed = PlacementEnd{run.command};
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

// ---- What issue sees of shmem (look-ahead (c), reuse (d)): the service
// slots and the warp's own order (rfc12 sections 3, 4 and 6) ----

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
// does not reach the warp selection (trash/issue-depth-2026-11-20/findings.md:
// RF winner -> slot space -> winning warp was the longest issue path).
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
// issue (rfc12 section 3: a reserved slot is unserviced immediately).
bool Machine::warp_dispatching(uint32_t warp, bool reads, bool writes, bool probes) const {
    for (const auto& op : dispatch_pipe_.cur()) {
        if (op.warp != warp) continue;
        const auto opcode = op.instruction.opcode;
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
// granted by the end of this cycle (rfc12 section 6: a SIMT RMW counts as
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

// fence.sh (rfc12 section 10): every earlier shmem access of the warp has
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

uint64_t Machine::full_mask() const {
    return config_.lanes >= 64 ? ~0ull : (1ull << config_.lanes) - 1;
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
    predicate.reason = "mbar.try_wait";
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

// ===========================================================================
// MMA engine (port B) and accumulator transfers
// ===========================================================================

namespace {

// An operand element's byte address, in the profile's layout through the
// layout rule the TMA places tiles with (src/layout.hpp); the same rule for
// A and B. Under swizzle64 both operands are rows of k -- A m rows, B n rows
// -- `spacing` bytes apart (the instruction's), and `base` is the operand's
// first 16-byte piece in its row 0 (tile + 16 x piece: a logical position,
// not a shuffled one). Element (row, k) is at the logical address base + row
// x spacing + 2k, shuffled by that address's own row number (address /
// spacing, src/layout.hpp): a K = 8 command reads one piece of every row; a
// shape with k > 8 reads k / 8 consecutive pieces of the same rows. The
// address generator walks the rows in order and adds the spacing once per
// row (an adder, no multiplier); `row * spacing` here is the simulator's
// shorthand for that running sum, and the row number is the running
// address's bits above the spacing (wires, no counter). An operand may start
// at any row of a taller tile. The dense diagnostic layout is row-major and
// compact in the command's shape: A m rows of k, B k rows of n (no spacing).
uint32_t mma_a_address(const Config& config, const MmaShape& shape, uint32_t base, uint32_t spacing,
    uint32_t m, uint32_t k) {
    if (config.mma_layout == TileLayout::dense) return dense_address(base, m, k, shape.k, 2);
    return swizzle64_shuffle(base + m * spacing + 2 * k, spacing);
}

uint32_t mma_b_address(const Config& config, const MmaShape& shape, uint32_t base, uint32_t spacing,
    uint32_t k, uint32_t n) {
    if (config.mma_layout == TileLayout::dense) return dense_address(base, k, n, shape.n, 2);
    return swizzle64_shuffle(base + n * spacing + 2 * k, spacing);
}

uint32_t mma_operand_element(const Config& config, const MmaShape& shape, uint32_t base, uint32_t spacing,
    bool a_operand, uint32_t outer, uint32_t k) {
    return a_operand ? mma_a_address(config, shape, base, spacing, outer, k)
                     : mma_b_address(config, shape, base, spacing, k, outer);
}

// The bytes an operand of `shape` reads, as [first, end): the hull of every
// element it reads (under swizzle64 a later row's piece can sit below or
// above its logical place, so the ends come from the addresses themselves).
std::pair<uint64_t, uint64_t> mma_operand_hull(const Config& config, const MmaShape& shape, uint32_t base,
    uint32_t spacing, bool a_operand) {
    uint64_t first = std::numeric_limits<uint64_t>::max();
    uint64_t end = 0;
    const uint32_t outer = a_operand ? shape.m : shape.n;
    for (uint32_t i = 0; i < outer; ++i) {
        for (uint32_t k = 0; k < shape.k; ++k) {
            const uint64_t address = mma_operand_element(config, shape, base, spacing, a_operand, i, k);
            first = std::min(first, address);
            end = std::max(end, address + 2);
        }
    }
    return {first, end};
}

uint32_t mma_total_steps(const Config& config, const MmaShape& shape) {
    return (shape.m / config.mma_core_m) * (shape.n / config.mma_core_n) * (shape.k / config.mma_core_k);
}

} // namespace

uint32_t mma_operand_address(const Config& config, const MmaShape& shape, uint32_t base, uint32_t spacing,
    bool a_operand, uint32_t outer, uint32_t k) {
    return mma_operand_element(config, shape, base, spacing, a_operand, outer, k);
}

// The acc.ld (load) or acc.st engine, as this cycle leaves it (look-ahead
// (c)): free if it runs no transfer. The MMA never matters here -- it has
// tmem services of its own (trash/rfc11.md section 0); a transfer racing MMA
// work is the sanitizer's business.
bool Machine::acc_engine_available(const Wires& w, bool load) const {
    return !w.acc_next[load ? 0 : 1].has_value();
}

// The MMA engine's cycle, computed on a copy of its queue and pipeline:
// issue looks ahead at the queue entries it frees (c), so the next state is a
// wire; drive_mma commits it with issue's new command. Its tmem reads (C) and
// writes (results) use its own tmem services; the writes land at this
// cycle's edge (w.tmem_writes).
void Machine::step_mma(Wires& w) {
    auto mma = mma_.cur();
    advance_mma(mma, w);
    w.mma_next = std::move(mma);
}

// One cycle of the MMA engine, applied to `mma` (a copy of its Q):
//   1. the steps in the core: one whose C is at the tmem RAM now samples it
//      and computes its result; one whose result is at the tmem RAM now
//      writes it (at this cycle's edge) and retires; retired commands leave
//      the queue head;
//   2. the port-B read pipeline: the round at the RAM samples its elements;
//      the round at the port output adds them to the core's operand latch,
//      and a step's last round enters the core (C is read from tmem then);
//   3. port B grants one round: the next round of the step in service, else
//      a new step's first round (or the head commit's presentation);
//   4. the read pipeline moves one stage; the granted round enters stage 0.
// A step granted its last round at g samples it at g + D and enters the core
// at g + D + Lb (the same D and Lb as port A, rfc10 section 5).
void Machine::advance_mma(MmaPipeline& mma, Wires& w) {
    for (auto it = mma.results.begin(); it != mma.results.end();) {
        if (!it->c_sampled && it->c_sample_cycle == cycle_) sample_mma_c(*it);
        if (it->land_cycle != cycle_) { ++it; continue; }
        if (!it->c_sampled) throw SimError("internal MMA result lands before its C sample");
        for (std::size_t i = 0; i < it->words.size(); ++i) w.tmem_writes.emplace_back(it->words[i], it->values[i]);
        trace_->mma_result(cycle_, it->command, it->address);
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
    if (auto& at_ram = mma.reads.at(config_.shmem_port_a_depth - 1); at_ram.has_value())
        sample_mma_read(mma, *at_ram, w);
    if (const auto& output = mma.reads.back(); output.has_value()) {
        mma.arriving.insert(mma.arriving.end(), output->values.begin(), output->values.end());
        if (output->last_round) {
            enter_mma_core(mma, output->step);
            mma.arriving.clear();
        }
    }
    std::optional<MmaRead> granted;
    if (!mma.fetch.has_value()) mma.fetch = next_mma_step(mma, w);
    if (mma.fetch.has_value()) {
        granted = grant_mma_round(mma);
        if (mma.fetch->next_round == mma.fetch->round_elements.size()) mma.fetch.reset();
    }
    for (std::size_t stage = mma.reads.size() - 1; stage > 0; --stage)
        mma.reads[stage] = std::move(mma.reads[stage - 1]);
    mma.reads.front() = std::move(granted);
}

// The next step port B may start now, if any; or the head commit presents
// into the MMA's event register. Steps walk hold-A / stream-B (rfc11 section
// 0): for each K block, for each A block (m), every B block (n) in order; the
// first step of each A block reads A and B, the rest read B only and use the
// held A block. A command's steps may start once the previous command's last
// step has been granted (commands overlap in the pipelines).
std::optional<Machine::MmaFetch> Machine::next_mma_step(MmaPipeline& mma, Wires& w) {
    auto command_it = std::find_if(mma.queue.begin(), mma.queue.end(), [this](const auto& command) {
        return command.commit || command.next_step < mma_total_steps(config_, command.shape);
    });
    if (command_it == mma.queue.end()) return std::nullopt;
    auto& command = *command_it;
    if (command.commit) {
        // A commit at the queue head presents into the MMA's own event
        // register: free if empty in Q, or claimed by the RMW engine now (d).
        const bool register_free = !mma_mbar_event_.cur().has_value() || w.shmem.mma_event_claimed;
        if (command_it != mma.queue.begin() || !register_free) return std::nullopt;
        MbarRmwRequest request;
        request.operation = MbarRmwOperation::arrive;
        request.byte_address = command.mbar_address;
        w.mma_present = std::move(request);
        mma.queue.pop_front();
        return std::nullopt;
    }
    // The command's shape sets the walk: rows in core_m blocks (8 for m32,
    // 16 for m64), the n span in core_n blocks, k in core_k steps.
    const uint32_t blocks_n = command.shape.n / config_.mma_core_n;
    const uint32_t blocks_per_k = (command.shape.m / config_.mma_core_m) * blocks_n;
    const uint32_t total_steps = mma_total_steps(config_, command.shape);
    MmaStep step;
    const uint32_t k_block = command.next_step / blocks_per_k;
    const uint32_t block = command.next_step % blocks_per_k;
    const uint32_t m_block = block / blocks_n;
    const uint32_t n_block = block % blocks_n;
    step.command = command.id;
    step.c_pointer = command.c_pointer;
    step.m0 = m_block * config_.mma_core_m;
    step.n0 = n_block * config_.mma_core_n;
    step.k0 = k_block * config_.mma_core_k;
    step.a_base = command.a_base;
    step.b_base = command.b_base;
    step.a_spacing = command.a_spacing;
    step.b_spacing = command.b_spacing;
    step.shape = command.shape;
    step.init_step = command.init && step.k0 == 0;
    step.reload_a = n_block == 0;
    step.last_step = command.next_step + 1 == total_steps;
    // A step whose tile is still in the core or on its way to tmem (a short
    // sweep) waits: its C must be the earlier step's result. Tiles are told
    // apart by the tmem words they write (any shared word: the same tile, or
    // two commands' overlapping results), so pointers may alias freely.
    const auto words = mma_block_words(step);
    const auto shares_words = [&](const std::vector<uint32_t>& other) {
        return std::any_of(other.begin(), other.end(), [&](uint32_t word) {
            return std::find(words.begin(), words.end(), word) != words.end();
        });
    };
    const bool tile_in_flight =
        std::any_of(mma.results.begin(), mma.results.end(), [&](const MmaResult& result) {
            return shares_words(result.words);
        }) ||
        std::any_of(mma.reads.begin(), mma.reads.end(), [&](const std::optional<MmaRead>& read) {
            return read.has_value() && shares_words(mma_block_words(read->step));
        });
    if (tile_in_flight) {
        trace_->mma_recurrence_stall();
        return std::nullopt; // explicit recurrence bubble for a profile whose sweep is too short
    }
    std::vector<uint32_t> elements;
    if (step.reload_a) {
        for (uint32_t mi = 0; mi < config_.mma_core_m; ++mi)
            for (uint32_t kk = 0; kk < config_.mma_core_k; ++kk)
                elements.push_back(mma_a_address(config_, step.shape, step.a_base, step.a_spacing,
                    step.m0 + mi, step.k0 + kk));
    }
    for (uint32_t kk = 0; kk < config_.mma_core_k; ++kk)
        for (uint32_t ni = 0; ni < config_.mma_core_n; ++ni)
            elements.push_back(mma_b_address(config_, step.shape, step.b_base, step.b_spacing,
                step.k0 + kk, step.n0 + ni));
    // Bank rounds: round r reads every bank's r-th distinct word.
    std::vector<std::set<uint32_t>> words_by_bank(config_.shmem_banks);
    for (const auto address : elements) {
        const auto word = address / 4;
        words_by_bank[word % config_.shmem_banks].insert(word);
    }
    uint32_t rounds = 1;
    std::map<uint32_t, uint32_t> word_round;
    for (const auto& words : words_by_bank) {
        rounds = std::max<uint32_t>(rounds, static_cast<uint32_t>(words.size()));
        uint32_t index = 0;
        for (const auto word : words) word_round[word] = index++;
    }
    MmaFetch fetch;
    fetch.step = step;
    fetch.round_elements.resize(rounds);
    for (const auto address : elements) fetch.round_elements[word_round.at(address / 4)].push_back(address);
    ++command.next_step;
    ++command.outstanding_results; // retires at its tmem write
    sanitizer_.record_mma_fetch_start(command.id);
    trace_->mma_input(cycle_, command.id, rounds, command.next_step - 1, step.reload_a);
    return fetch;
}

// Port B grants the step in service its next round.
Machine::MmaRead Machine::grant_mma_round(MmaPipeline& mma) {
    auto& fetch = *mma.fetch;
    MmaRead read;
    read.step = fetch.step;
    read.elements = fetch.round_elements.at(fetch.next_round);
    ++fetch.next_round;
    read.last_round = fetch.next_round == fetch.round_elements.size();
    return read;
}

// A round at the RAM reads its elements from shmem as it is now, before this
// cycle's writes land at the clock edge (READ_FIRST, rfc12 section 1). The
// command's last step's last round is its final physical operand read: the
// operands are consumed and its depbar clears here.
void Machine::sample_mma_read(MmaPipeline& mma, MmaRead& read, Wires& w) {
    for (const auto address : read.elements) {
        if (static_cast<uint64_t>(address) + 2 > shmem_.size())
            throw SimError("MMA operand footprint is outside shmem");
        uint16_t bits = 0;
        std::memcpy(&bits, shmem_.data() + address, 2);
        read.values.emplace_back(address, bits);
    }
    if (!read.last_round || !read.step.last_step) return;
    const auto id = read.step.command;
    auto command = std::find_if(mma.queue.begin(), mma.queue.end(),
        [id](const auto& candidate) { return candidate.id == id; });
    if (command == mma.queue.end()) throw SimError("internal MMA read lost its command");
    command->operands_consumed = true;
    sanitizer_.record_mma_operands_consumed(command->id); // final operand read
    record_depbar_completion(w, command->warp, command->depbar, "internal MMA depbar underflow");
}

// A step's operands are all at the port output: it enters the core. A
// reloading step's A block replaces the held one; the others use it. Its C
// read leaves for tmem now (sampled tmem.port_depth later, sample_mma_c);
// its result leaves the core result_latency after entry and lands in tmem
// tmem.port_depth after that.
void Machine::enter_mma_core(MmaPipeline& mma, const MmaStep& step) {
    std::map<uint32_t, uint16_t> latch(mma.arriving.begin(), mma.arriving.end());
    const auto value = [&](uint32_t address) {
        const auto found = latch.find(address);
        if (found == latch.end()) throw SimError("internal MMA operand did not arrive with its step");
        return found->second;
    };
    if (step.reload_a) {
        mma.a_held.assign(config_.mma_core_m * config_.mma_core_k, 0);
        for (uint32_t mi = 0; mi < config_.mma_core_m; ++mi)
            for (uint32_t kk = 0; kk < config_.mma_core_k; ++kk)
                mma.a_held[mi * config_.mma_core_k + kk] = value(mma_a_address(config_, step.shape,
                    step.a_base, step.a_spacing, step.m0 + mi, step.k0 + kk));
    }
    if (mma.a_held.size() != config_.mma_core_m * config_.mma_core_k)
        throw SimError("internal MMA step found no held A block");
    // B is transposed here only in the simulator's local operand snapshot.
    std::vector<uint16_t> b(config_.mma_core_n * config_.mma_core_k);
    for (uint32_t ni = 0; ni < config_.mma_core_n; ++ni)
        for (uint32_t kk = 0; kk < config_.mma_core_k; ++kk)
            b[ni * config_.mma_core_k + kk] = value(mma_b_address(config_, step.shape,
                step.b_base, step.b_spacing, step.k0 + kk, step.n0 + ni));
    MmaResult result;
    result.c_sample_cycle = cycle_ + config_.tmem_port_depth;
    result.land_cycle = cycle_ + config_.mma_result_latency_cycles + config_.tmem_port_depth;
    result.command = step.command;
    result.address = step.c_pointer + step.m0 * step.shape.n + step.n0;
    result.words = mma_block_words(step);
    result.init_step = step.init_step;
    result.a = mma.a_held;
    result.b = std::move(b);
    mma.results.push_back(std::move(result));
}

// A step's C is at the tmem RAM now: its sixteen words are sampled (one
// round of the MMA's tmem read service; READ_FIRST, before this cycle's
// writes land) and the fused dot-plus-C result is computed. It arrives in the
// core tmem.read_latency_cycles later, before the core's last stage (profile
// validation: result_latency >= port_depth + read_latency).
void Machine::sample_mma_c(MmaResult& result) {
    result.values.resize(config_.mma_core_m * config_.mma_core_n);
    for (uint32_t mi = 0; mi < config_.mma_core_m; ++mi) {
        for (uint32_t ni = 0; ni < config_.mma_core_n; ++ni) {
            const uint32_t c = result.init_step ? 0u : tmem_.at(result.words.at(mi * config_.mma_core_n + ni));
            result.values[mi * config_.mma_core_n + ni] = mma_dot_f16(
                std::span<const uint16_t>(result.a).subspan(mi * config_.mma_core_k, config_.mma_core_k),
                std::span<const uint16_t>(result.b).subspan(ni * config_.mma_core_k, config_.mma_core_k), c);
        }
    }
    result.c_sampled = true;
}

std::size_t Machine::tmem_index(uint64_t logical, uint32_t row_words) const {
    const uint64_t index = tmem_physical_word(logical, row_words);
    if (index >= tmem_.size()) throw SimError("internal tmem layout overflow");
    return static_cast<std::size_t>(index);
}

std::optional<std::vector<uint32_t>> Machine::tmem_region_words(
    uint32_t pointer, uint32_t row_words, uint32_t rows, uint32_t columns) const {
    std::vector<uint32_t> words;
    words.reserve(static_cast<std::size_t>(rows) * columns);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t column = 0; column < columns; ++column) {
            const uint64_t physical = tmem_physical_word(
                uint64_t{pointer} + uint64_t{row} * row_words + column, row_words);
            if (physical >= tmem_.size()) return std::nullopt;
            words.push_back(static_cast<uint32_t>(physical));
        }
    }
    return words;
}

// The tmem words of a matmul step's core_m x core_n result block, row-major:
// rows m0.. and columns n0.. of the command's result region (rows of n words).
std::vector<uint32_t> Machine::mma_block_words(const MmaStep& step) const {
    std::vector<uint32_t> words;
    words.reserve(config_.mma_core_m * config_.mma_core_n);
    for (uint32_t mi = 0; mi < config_.mma_core_m; ++mi)
        for (uint32_t ni = 0; ni < config_.mma_core_n; ++ni)
            words.push_back(static_cast<uint32_t>(tmem_index(
                uint64_t{step.c_pointer} + uint64_t{step.m0 + mi} * step.shape.n + step.n0 + ni, step.shape.n)));
    return words;
}

// The two accumulator transfer engines (trash/rfc11.md section 0: one acc.ld
// engine, one acc.st engine for the whole machine, each with its own tmem
// service in every fine bank), computed on copies of their registers: issue
// looks ahead at them (c) and may give a freed engine its next transfer now
// (d). The acc.ld pipeline before the acc.st beat.
void Machine::step_acc_transfers(Wires& w) {
    auto engines = acc_engines_.cur();
    auto pipe = acc_ld_pipe_.cur();
    step_acc_ld(engines[0], pipe, w);
    step_acc_st(engines[1], w);
    w.acc_next = std::move(engines);
    w.acc_ld_next = std::move(pipe);
}

// The acc.ld engine's pipeline (trash/rfc10.md section 2, "Accumulator
// transfers"). The beat at the tmem RAM stage samples its words now (once).
// Its output asks the RF write decision for service; this cycle's grant was
// decided at the top of the cycle.
//   output empty, or granted -> advance: every stage moves one toward the
//                               output; stage 0 takes the engine's next read
//                               if its transfer has a beat left, else stays
//                               empty
//   output valid and denied  -> hold: no stage moves, nothing launches
// A granted beat belongs to the RF write path from its grant on.
void Machine::step_acc_ld(std::optional<AccTransfer>& engine, AccLdPipeline& pipe, const Wires& w) {
    if (auto& at_ram = pipe.at(config_.tmem_port_depth - 1); at_ram.has_value() && !at_ram->sampled) {
        sample_acc_ld_beat(*at_ram);
    }
    const bool output_valid = pipe.back().has_value();
    if (output_valid && !w.rf_write.granted_to(AsyncWriter::acc_ld, 0)) return;
    for (std::size_t stage = pipe.size() - 1; stage > 0; --stage) pipe[stage] = std::move(pipe[stage - 1]);
    pipe.front() = launch_acc_ld_beat(engine);
}

// The engine's next read leaves for tmem now: the transfer's next column, all
// lanes. The engine frees with its last read's launch; each beat carries its
// own destination and context down the pipeline.
std::optional<Machine::AccLdBeat> Machine::launch_acc_ld_beat(std::optional<AccTransfer>& engine) {
    if (!engine.has_value()) return std::nullopt;
    auto& transfer = *engine;
    AccLdBeat beat;
    beat.warp = transfer.warp;
    beat.reg = static_cast<uint8_t>(transfer.rf_base + transfer.next_column);
    beat.address = transfer.pointer + transfer.next_column;
    beat.row_words = transfer.row_words;
    beat.last = transfer.next_column + 1 == transfer.width;
    beat.depbar = transfer.depbar;
    ++transfer.next_column;
    if (beat.last) engine.reset();
    return beat;
}

// A beat at the tmem RAM: its sixteen words (sixteen distinct fine banks) are
// read now, as tmem is before this cycle's writes land (READ_FIRST). The
// transfer's last sample ends its tmem read-ownership.
void Machine::sample_acc_ld_beat(AccLdBeat& beat) {
    beat.values.resize(config_.lanes);
    for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
        beat.values[lane] = tmem_[tmem_index(uint64_t{beat.address} + uint64_t{lane} * beat.row_words, beat.row_words)];
    }
    beat.sampled = true;
    trace_->tmem_beat(cycle_, beat.warp, true, beat.address);
    if (beat.last) sanitizer_.record_acc_transfer_final_beat(true);
}

// The acc.st engine's transfer: its beats are read from the RF one per cycle
// when the RF read decision grants them (decide_rf_reads: sampled at the
// grant, at the RF output Lr later, at the tmem RAM tmem.port_depth after
// that); a beat at the tmem RAM now is written at this cycle's edge. The last
// beat's write ends the transfer: its depbar clears, its tmem write-ownership
// ends and the engine frees.
void Machine::step_acc_st(std::optional<AccTransfer>& engine, Wires& w) {
    if (!engine.has_value()) return;
    auto& transfer = *engine;
    while (!transfer.in_flight.empty() && transfer.in_flight.front().land_cycle == cycle_) {
        const auto& beat = transfer.in_flight.front();
        const uint64_t address = uint64_t{transfer.pointer} + beat.column;
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            w.tmem_writes.emplace_back(
                static_cast<uint32_t>(tmem_index(address + uint64_t{lane} * transfer.row_words, transfer.row_words)),
                beat.values[lane]);
        }
        trace_->tmem_beat(cycle_, transfer.warp, false, static_cast<uint32_t>(address));
        transfer.in_flight.pop_front();
        ++transfer.landed;
    }
    if (transfer.landed != transfer.width) return;
    record_depbar_completion(w, transfer.warp, transfer.depbar, "internal acc.st depbar underflow");
    // final tmem write: write-ownership ends
    sanitizer_.record_acc_transfer_final_beat(false);
    engine.reset();
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
//      step (plan_tma_walker) -- a load sector wants the request channel, a
//      store sector the A-read of its source (and a free TMA store register
//      for its words); a zero fill or a skipped store stretch wants no shared
//      resource. One TMA read in service at a time: none while a descriptor
//      read's words are still coming back, or while a store-source read has
//      rounds left to be granted. There is no mbar read at the start (rfc12
//      section 8): the walk begins once the descriptor is decoded and the
//      walker's start has left its pipeline (nothing is wanted meanwhile).
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
            event.byte_address = command.mbar_address;
            event.value = static_cast<uint32_t>(command.logical_bytes);
            event.tma_command = command.id; // (S) the sanitizer's accept-time phase peek
            want.present_event = std::move(event);
        }
    }
    const auto* slot = egress_request_slot(std::nullopt);
    want.write_request = slot != nullptr && slot->client == GmemClient::tma && egress_slot_visible(*slot);

    if (!tma_write_txn_.cur().has_value() && !tma.returns.empty()) {
        const auto& ticket = tma.returns.front();
        want.placement = tma_port_request(TmaPortKind::load_write, ticket.run, ticket.sector_request);
    }

    want.frontend = tma.frontend.has_value() ? tma.frontend : next_tma_frontend_command(tma);
    if (!want.frontend.has_value()) return want;
    const auto& command = tma.commands.at(*want.frontend);
    const bool read_free = !tma_read_txn_.cur().has_value();
    if (command.state == TmaCommandState::descriptor) {
        if (!read_free) return want;
        // The descriptor read: the run of its sixteen words (dense).
        TmaRun run;
        run.command = command.id;
        run.shared_address = command.descriptor_address;
        run.elements = static_cast<uint8_t>(tma_descriptor_bytes / 4);
        run.element_bytes = 4;
        want.a_read = tma_port_request(TmaPortKind::descriptor, run);
        return want;
    }
    // The walker's start is in its pipeline: the frontend waits for it.
    if (command.state == TmaCommandState::starting) return want;
    if (command.state != TmaCommandState::walking) {
        want.frontend_reset = true;
        return want;
    }
    // A store sector's source read needs a TMA store register (free in Q, or
    // the head one drained into the egress FIFO this cycle (d)) and no TMA
    // read in service.
    const auto store_register = read_free ? tma_store_register_free(egress) : std::nullopt;
    want.walker = plan_tma_walker(command, tma.walker, tma.credits_used, store_register.has_value());
    if (want.walker->sector == TmaWalkerPlan::Sector::store) {
        want.a_read = tma_port_request(TmaPortKind::store_read, want.walker->run);
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

// The walker's step this cycle, planned from Q (TmaWalker, sim.hpp): ONE
// run, one zero fill or one skipped stretch, found in closed form from the
// walker's position and its bounds -- never a loop over elements. From (x, y):
//   in range   a run to the first of the sector's end ((sector bytes -
//              address mod sector bytes) / element bytes elements on) and
//              x_high (the tensor's edge, or the row's end): a load or store
//              sector;
//   a load's   a zero fill to the first of x_low (the in-range part) or the
//   stretch    row's end, and a sector's bytes;
//   a store's  a skip to x_low or the row's end: no sector, no credit, one
//   stretch    cycle.
// A run never leaves its row: a tile whose rows are contiguous in gmem and
// shorter than a sector takes a sector request per row.
// Blocked (Sector::none; nothing changes): no credit; a store sector without
// a free store register (and A-read). A load sector also needs the request
// channel, which select_request_channel decides.
Machine::TmaWalkerPlan Machine::plan_tma_walker(const TmaCommand& command, const TmaWalker& walker,
    uint32_t credits_used, bool store_register_free) const {
    using Sector = TmaWalkerPlan::Sector;
    TmaWalkerPlan plan;
    if (command.walker_done) return plan;
    const bool load = command.direction == TmaDirection::load;
    const uint32_t bytes = walker.element_bytes;
    const uint32_t sector_bytes = config_.gmem_sector_bytes;
    const uint32_t x = walker.x;
    const bool row_in_range = walker.y >= walker.y_low && walker.y < walker.y_high;
    const bool in_range = row_in_range && x >= walker.x_low && x < walker.x_high;
    const uint64_t global = walker.row_global + uint64_t{x} * bytes; // x x bytes: a shift
    uint32_t end = 0;
    if (in_range) {
        const uint32_t to_sector_end = (sector_bytes - static_cast<uint32_t>(global % sector_bytes)) / bytes;
        end = std::min<uint32_t>(walker.x_high, x + to_sector_end);
        plan.sector = load ? Sector::load : Sector::store;
    } else {
        end = row_in_range && x < walker.x_low ? walker.x_low : walker.tile_length;
        if (load) end = std::min<uint32_t>(end, x + sector_bytes / bytes);
        plan.sector = load ? Sector::zero_fill : Sector::skip;
    }
    if (end <= x || end > walker.tile_length) throw SimError("internal: the tma walker made no progress");
    if (plan.sector != Sector::skip) {
        if (credits_used >= config_.tma_sector_credits ||
            (plan.sector == Sector::store && !store_register_free)) {
            plan.sector = Sector::none;
            return plan;
        }
        auto& run = plan.run;
        run.command = command.id;
        run.elements = static_cast<uint8_t>(end - x);
        run.element_bytes = static_cast<uint8_t>(bytes);
        run.zero_fill = !in_range;
        run.layout = walker.layout;
        run.transposed = walker.transposed;
        run.spacing = walker.spacing;
        // Element x's unshuffled shared address: x x bytes, or (transposed) x x
        // the stored row's bytes -- a shift either way.
        run.shared_address = walker.row_shared + x * (walker.transposed ? walker.spacing : bytes);
        const uint64_t step = walker.transposed ? walker.spacing : bytes;
        if (uint64_t{run.shared_address} + (run.elements - 1) * step + bytes > shmem_.size()) {
            throw SimError("internal: a tma run is outside shmem (the tile was checked at its decode)");
        }
        if (in_range) {
            run.global_address = global;
            const uint64_t run_bytes = uint64_t{run.elements} * bytes;
            if (global > gmem_.size() || run_bytes > gmem_.size() - global) {
                throw SimError("tma global access is outside gmem");
            }
            plan.sector_address = run_sector_address(run);
        }
    }
    // The walker after the step: along the row, or to the next row (one add
    // of each row step).
    auto& next = plan.walker_after;
    next = walker;
    next.x = static_cast<uint16_t>(end);
    if (end == walker.tile_length) {
        next.x = 0;
        next.y = static_cast<uint16_t>(walker.y + 1);
        next.row_global += walker.row_step_global;
        next.row_shared += walker.row_step_shared;
        plan.done_after = next.y == walker.tile_rows;
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
    // The walker's start at the start pipeline's last stage loads the walker:
    // the command walks from the next cycle.
    const auto& start_pipe = tma_walker_start_pipe_.cur();
    if (!start_pipe.empty() && start_pipe.back().has_value()) {
        const auto& start = *start_pipe.back();
        auto& command = tma.commands.at(start.command);
        if (command.state != TmaCommandState::starting || tma.frontend != start.command) {
            throw SimError("internal: a tma walker start left its pipeline for a command not starting");
        }
        tma.walker = start.walker;
        command.state = TmaCommandState::walking;
    }
    if (w.shmem.tma_read_done.has_value()) complete_tma_frontend_read(tma, *w.shmem.tma_read_done, w);
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
    // when the request channel lets them through; a skip always goes.
    const bool store_read_granted = w.shmem.a_read.user == AReadUser::tma && w.shmem.a_read.tma_start;
    const bool emits = plan.sector == Sector::store ? store_read_granted
        : (plan.sector == Sector::load || plan.sector == Sector::zero_fill) && w.channel.walker_proceeds;
    if (plan.sector == Sector::skip || emits) {
        tma.walker = plan.walker_after;
        if (plan.done_after) command.walker_done = true;
    }
    if (emits) {
        ++tma.credits_used;
        if (tma.credits_used > config_.tma_sector_credits) throw SimError("internal: tma sector credits overrun");
        trace_tma_depths(tma, cycle_);
        if (!plan.run.zero_fill) command.useful_bytes += uint64_t{plan.run.elements} * plan.run.element_bytes;
        switch (plan.sector) {
        case Sector::zero_fill: {
            TmaReturn returned;
            returned.run = plan.run;
            returned.sector_request = w.channel.walker_id;
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
        case Sector::skip:
            break;
        }
    }
    if (!command.walker_done) return; // credit or channel pressure: retry next cycle
    const bool load = command.direction == TmaDirection::load;
    if ((load && command.pending_load_writes == 0) || (!load && command.pending_store_captures == 0)) {
        mark_tma_data_complete(command);
    }
    tma.frontend.reset();
}

// A descriptor read's last round is back: its data is in `request` (the
// frontend latch). The walker's start enters its pipeline (or, with no
// stages, loads the walker now); the command walks after it (I3: no mbar
// phase read and no "enough transaction bytes" check -- the tx count is
// signed, rfc12 section 8).
void Machine::complete_tma_frontend_read(TmaQueue& tma, const TmaReadTransaction& read, Wires& w) {
    const auto& request = read.request;
    auto command_it = tma.commands.find(request.run.command);
    if (command_it == tma.commands.end()) throw SimError("orphaned tma shmem transaction");
    auto& command = command_it->second;

    switch (request.kind) {
    case TmaPortKind::descriptor: {
        if (command.state != TmaCommandState::descriptor) {
            throw SimError("tma descriptor read completed in the wrong command state");
        }
        if (tma.frontend != command.id) throw SimError("internal: a descriptor returned for a command not walked next");
        const auto descriptor = decode_tma_descriptor(read.latch.data(), read.latch.size());
        command.logical_bytes = tma_logical_bytes(descriptor);
        validate_tma_runtime_descriptor(command, descriptor);
        auto walker = tma_walker_start(command, descriptor);
        // releases the descriptor read-ownership; verifies the accept-time peek
        sanitizer_.record_tma_descriptor_read(command.id, descriptor, command.shmem_address,
            command.direction == TmaDirection::load);
        if (config_.tma_walker_start_cycles == 0) {
            tma.walker = std::move(walker);
            command.state = TmaCommandState::walking;
        } else {
            w.tma_walker_starting = TmaWalkerStart{command.id, std::move(walker)};
            command.state = TmaCommandState::starting;
        }
        return;
    }
    case TmaPortKind::load_write:
    case TmaPortKind::store_read:
        break;
    }
    throw SimError("unexpected tma frontend read");
}

// A placement's last write reached the RAM: its command's load write ends.
// A placement over an mbar with a live waiter is UB (as on NVIDIA: an
// mbarrier's memory serves nothing else until mbarrier.inval); the
// sanitizer's C10 PANICs at the placement write. The machine just wrote the
// bytes (backlog D12.1).
void Machine::complete_tma_load_write(TmaQueue& tma, const PlacementEnd& placed) {
    auto command_it = tma.commands.find(placed.command);
    if (command_it == tma.commands.end()) throw SimError("orphaned tma shmem transaction");
    auto& command = command_it->second;
    if (command.direction != TmaDirection::load || command.pending_load_writes == 0) {
        throw SimError("unexpected tma load return");
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
    if (command.direction != TmaDirection::store || command.pending_store_captures == 0) {
        throw SimError("unexpected tma store capture");
    }
    --command.pending_store_captures;
    if (command.walker_done && command.pending_store_captures == 0) {
        mark_tma_data_complete(command);
    }
}

std::optional<uint64_t> Machine::next_tma_frontend_command(const TmaQueue& tma) const {
    for (const auto& [id, command] : tma.commands) {
        if (command.state == TmaCommandState::descriptor || command.state == TmaCommandState::starting ||
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

// The bank rounds of one TMA shmem transaction, from its run: each round,
// every bank serves one word. A piece is one element: its bytes lie in one
// shared word and (a sector run) one sector word, because both addresses are
// multiples of the element size (the descriptor's base is; the strides count
// elements; the buffer is 64-byte aligned) and the shuffle moves whole words.
//
// A load's placement from the sector register has one more rule (HUMAN
// 2026-11-18): in a round a bank takes its bytes from ONE 4-byte word of the
// sector register, rotated by one byte amount, and writes them with byte
// enables. That is all the wiring the path has per bank: a word select, a
// four-way byte rotate, four byte enables. A shared word whose bytes come
// from two sector words (a row that starts off a 4-byte boundary) is
// therefore two partial writes in two rounds; a word that straddles two
// sectors gets its halves from the two sectors' placements. Crooked data is
// slower, never refused. Reads (the descriptor, a store's source) and zero
// fills take whole words and have no such rule. In each bank the sources
// take rounds in (shared word, sector word, rotation) order.
//
// Worst case per sector (tma_max_rounds, tma_max_pieces; checked here):
//   pieces  one per element: at most sector bytes / element bytes (32 at the
//           canonical profile, a run of 1-byte elements; the descriptor
//           read: 16);
//   rounds  with 16 or more banks, a power of two (the canonical 16): a
//           row's run covers at most sector bytes / 4 + 1 consecutive
//           unshuffled words (9), shuffled by one row number, so at most
//           ceil(9 / 16) = 1 per bank, each with at most two sources: 2
//           rounds; a transposed run's elements sit in consecutive stored
//           rows, any 16 of which put the same word on 16 banks: 1 round;
//           the descriptor's 16 aligned words: 1 round. So 2 x ceil((sector
//           bytes / 4 + 1) / 16) = 2 rounds at the canonical profile. Any
//           other bank count: at most one round per piece.
uint32_t Machine::tma_max_pieces() const {
    return std::max<uint32_t>(config_.gmem_sector_bytes, tma_descriptor_bytes / 4);
}

uint32_t Machine::tma_max_rounds() const {
    const uint32_t banks = config_.shmem_banks;
    if (banks >= 16 && (banks & (banks - 1)) == 0) return 2 * ((config_.gmem_sector_bytes / 4 + 1 + 15) / 16);
    return tma_max_pieces();
}

// Where a piece (element) starts in shared memory: the run's unshuffled
// address plus the element's step (a shift), through the layout's shuffle.
uint32_t Machine::tma_piece_address(const TmaRun& run, const TmaPortRequest::Piece& piece) const {
    const uint32_t step = run.transposed ? run.spacing : run.element_bytes;
    const uint32_t natural = run.shared_address + uint32_t{piece.element} * step;
    return run.layout == TileLayout::swizzle64 ? swizzle64_shuffle(natural, run.spacing) : natural;
}

Machine::TmaRoundPlan Machine::tma_round_plan(const TmaPortRequest& request) const {
    using Piece = TmaPortRequest::Piece;
    const auto& run = request.run;
    const bool from_sector_register = request.kind == TmaPortKind::load_write && !run.zero_fill;
    const uint32_t bytes = run.element_bytes;
    if (run.elements == 0 || run.elements > tma_max_pieces() || (bytes != 1 && bytes != 2 && bytes != 4)) {
        throw SimError("internal: a tma run of " + std::to_string(run.elements) + " elements of " +
            std::to_string(bytes) + " bytes");
    }
    // Per bank: (shared word, sector word, byte rotation) -> its pieces.
    using Source = std::tuple<uint32_t, uint64_t, uint8_t>;
    std::vector<std::map<Source, std::vector<Piece>>> banks(config_.shmem_banks);
    for (uint32_t element = 0; element < run.elements; ++element) {
        const Piece piece{static_cast<uint8_t>(element)};
        const uint32_t address = tma_piece_address(run, piece);
        if (address % bytes != 0) throw SimError("internal: a tma element is not aligned in shmem");
        Source source{address / 4, 0, 0};
        if (from_sector_register) {
            const uint64_t from = run.global_address + uint64_t{element} * bytes;
            if (from % bytes != 0) throw SimError("internal: a tma element is not aligned in gmem");
            source = {address / 4, from / 4, static_cast<uint8_t>((address % 4 + 4 - from % 4) % 4)};
        }
        banks[(address / 4) % config_.shmem_banks][source].push_back(piece);
    }
    std::size_t rounds = 0;
    for (const auto& bank : banks) rounds = std::max(rounds, bank.size());
    if (rounds == 0 || rounds > tma_max_rounds()) {
        throw SimError("internal: a tma transaction of " + std::to_string(rounds) + " bank rounds (the bound is " +
            std::to_string(tma_max_rounds()) + ")");
    }
    TmaRoundPlan plan(rounds);
    for (const auto& bank : banks) {
        std::size_t round = 0;
        for (const auto& [source, pieces] : bank) {
            (void)source;
            plan[round].insert(plan[round].end(), pieces.begin(), pieces.end());
            ++round;
        }
    }
    return plan;
}

Machine::TmaPortRequest Machine::tma_port_request(TmaPortKind kind, const TmaRun& run, uint64_t sector_request) const {
    TmaPortRequest request;
    request.kind = kind;
    request.sector_request = sector_request;
    request.run = run;
    request.rounds = static_cast<uint32_t>(tma_round_plan(request).size());
    return request;
}

// The run's sector, its first byte's offset in it, and the byte enables of
// its bytes (a store's AXI strobes): a decoder of (offset, length).
uint64_t Machine::run_sector_address(const TmaRun& run) const {
    return run.global_address - run.global_address % config_.gmem_sector_bytes;
}

uint32_t Machine::run_sector_offset(const TmaRun& run) const {
    return static_cast<uint32_t>(run.global_address % config_.gmem_sector_bytes);
}

uint64_t Machine::run_byte_enable(const TmaRun& run) const {
    const uint32_t length = uint32_t{run.elements} * run.element_bytes;
    const uint64_t mask = length >= 64 ? ~0ull : (1ull << length) - 1;
    return mask << run_sector_offset(run);
}

// The walker's registers for the command whose descriptor has just come
// back (TmaWalker, sim.hpp), computed in that cycle from the decoded
// descriptor and the command's coordinates; they travel through the walker
// start pipeline's tma.walker_start_cycles stages (TmaWalkerStart, canonical
// 2) before the walker is loaded, so the walk starts that many cycles after
// the next one (0 stages: in the next one, the timing before the stages).
// Three products, side by side, each a 32-bit coordinate x a 32-bit stride:
// the row dimension's (c_row x s_row) and the two other dimensions' (c_k x
// s_k; their in-range test is two compares each); the stages are theirs and
// the add chain's (trash/walker-followups-2026-11-20.md).
// Everything else is subtracts, compares and shifts:
//   x_low  = clamp(-c0, 0, length)        x_high = clamp(d0 - c0, 0, length)
//   y_low  = clamp(-c_row, 0, rows)       y_high = clamp(d_row - c_row, 0, rows)
//            (both 0 when another dimension's coordinate is outside)
//   row_global (row 0) = base + (c0 + c_row s_row + c_a s_a + c_b s_b) x element bytes,
//            modulo 2^64: exact wherever a row is in range.
void Machine::validate_tma_runtime_descriptor(const TmaCommand& command, const TmaDescriptor& descriptor) const {
    if (command.logical_bytes == 0 ||
        command.logical_bytes > std::numeric_limits<uint32_t>::max()) {
        throw SimError("tma logical byte count is outside the mbar encoding");
    }
    // The buffer address is 64-byte aligned (its low bits are not wired,
    // src/tma.hpp); the whole tile must lie in shmem.
    if (command.shmem_address > shmem_.size() ||
        command.logical_bytes > shmem_.size() - command.shmem_address) {
        throw SimError("tma tile does not fit in shmem");
    }
    if (descriptor.base > gmem_.size()) {
        throw SimError("tma global base is outside gmem");
    }
}

Machine::TmaWalker Machine::tma_walker_start(const TmaCommand& command, const TmaDescriptor& descriptor) const {
    TmaWalker walker;
    const uint32_t bytes = descriptor.element_bytes;
    const uint32_t row = descriptor.row_dimension;
    const int64_t length = descriptor.tile_length;
    const int64_t rows = descriptor.tile_rows;
    const int64_t c0 = command.coordinates[0];
    const int64_t c_row = command.coordinates[row];
    const auto clamp = [](int64_t value, int64_t high) {
        return static_cast<uint16_t>(std::clamp<int64_t>(value, 0, high));
    };
    walker.tile_length = descriptor.tile_length;
    walker.tile_rows = descriptor.tile_rows;
    walker.element_bytes = static_cast<uint8_t>(bytes);
    walker.layout = descriptor.layout;
    walker.transposed = descriptor.transposed;
    walker.spacing = descriptor.layout == TileLayout::swizzle64 ? tma_swizzle64_row_bytes(descriptor) : 0;
    walker.x_low = clamp(-c0, length);
    walker.x_high = clamp(int64_t{descriptor.dimensions[0]} - c0, length);
    // The two dimensions the tile does not walk: inside, and their offset.
    bool others_in_range = true;
    uint64_t others = 0; // < 2 x 2^31 x 2^32: fits
    for (uint32_t dimension = 1; dimension < tma_max_coordinates; ++dimension) {
        if (dimension == row) continue;
        const int64_t coordinate = command.coordinates[dimension];
        if (coordinate < 0 || static_cast<uint64_t>(coordinate) >= descriptor.dimensions[dimension]) {
            others_in_range = false;
            continue;
        }
        others += static_cast<uint64_t>(coordinate) * tma_stride(descriptor, dimension);
    }
    if (others_in_range) {
        walker.y_low = clamp(-c_row, rows);
        walker.y_high = clamp(int64_t{descriptor.dimensions[row]} - c_row, rows);
    }
    const uint64_t s_row = tma_stride(descriptor, row);
    // Element (0, 0)'s index and address, modulo 2^64 (unsigned wrap): |c_row x
    // s_row| < 2^63.
    const uint64_t element0 = static_cast<uint64_t>(c0) + static_cast<uint64_t>(c_row * static_cast<int64_t>(s_row)) +
        others;
    walker.row_global = descriptor.base + element0 * bytes;
    walker.row_step_global = s_row * bytes;
    walker.row_shared = command.shmem_address;
    walker.row_step_shared = descriptor.transposed ? bytes : uint32_t{descriptor.tile_length} * bytes;
    // Simulator check, not machine behaviour: the highest in-range element's
    // address must not leave 64 bits (each in-range address is then exact).
    if (walker.x_low < walker.x_high && walker.y_low < walker.y_high) {
        const auto overflow = [] { throw SimError("tma global address arithmetic overflow"); };
        const uint64_t last_x = static_cast<uint64_t>(c0 + walker.x_high - 1);
        const uint64_t last_row = static_cast<uint64_t>(c_row + walker.y_high - 1);
        if (last_row != 0 && s_row > UINT64_MAX / last_row) overflow();
        uint64_t element = last_row * s_row;
        if (element > UINT64_MAX - last_x) overflow();
        element += last_x;
        if (element > UINT64_MAX - others) overflow();
        element += others;
        if (element >= UINT64_MAX / bytes) overflow();
        if (descriptor.base > UINT64_MAX - (element + 1) * bytes) overflow();
    }
    return walker;
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
// Issue: the scheduler, after the units and the dispatch pipeline's end
// ===========================================================================

// The scheduler: from the cursor, the first warp whose next instruction can
// be accepted issues (one per cycle). The heads are the FIFOs' Q: a packet
// landing now is a candidate from the next cycle.
//
// Every warp's trace state comes from its own registers whether or not the
// scheduler reached it: halted, its own waits (probe, wait slot, stall
// count), then instruction supply (branch unresolved, FIFO empty). A warp
// past all of those is either tried (issued, or the reason it was refused)
// or, once another warp has issued this cycle, "not examined" -- never shown
// as ready, because nothing checked its depbars or resources. The FIFO-empty
// count is a separate observation over every running warp: an empty FIFO is
// not by itself a lost issue (another warp may issue, or the warp may be
// waiting anyway). Both are trace bookkeeping, not modeled hardware.
void Machine::decide_issue(Wires& w) {
    struct State {
        WarpState state = WarpState::not_examined;
        std::string detail;
    };
    const auto& front_end = front_end_.cur();
    const auto& branch = issue_branch_.cur();
    std::vector<State> states(warps_.size());
    bool issued = false;
    uint64_t fifo_empty = 0;
    for (uint32_t warp = 0; warp < warps_.size(); ++warp) {
        if (!front_end[warp].halted && front_end[warp].fifo.empty()) ++fifo_empty;
    }
    trace_->instruction_fifo_empty(fifo_empty);
    // The scan keeps going past the winner (for the trace), so it walks from
    // the cursor as it was at the start of the cycle.
    const auto cursor = scheduler_cursor_;
    for (uint32_t offset = 0; offset < warps_.size(); ++offset) {
        const auto index = (cursor + offset) % static_cast<uint32_t>(warps_.size());
        auto& warp = warps_[index];
        if (front_end[index].halted) {
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
        // Nothing after a branch issues until it resolves (no prediction).
        if (branch.has_value() && branch->warp == index) {
            states[index] = {WarpState::no_instruction, "branch unresolved"};
            continue;
        }
        if (front_end[index].fifo.empty()) {
            states[index] = {WarpState::no_instruction, "instruction FIFO empty"};
            continue;
        }
        if (issued) {
            states[index] = {WarpState::not_examined, {}};
            continue;
        }
        const auto attempt = try_issue(warp, w);
        if (attempt.issued) {
            states[index] = {WarpState::issued, {}};
            scheduler_cursor_ = (index + 1) % static_cast<uint32_t>(warps_.size());
            issued = true;
            continue;
        }
        states[index] = {attempt.state, attempt.reason};
    }

    if (!issued) {
        const auto draining = std::all_of(front_end.begin(), front_end.end(), [](const WarpFrontEnd& warp) {
            return warp.halted;
        });
        trace_->scheduler_idle(draining);
    }
    for (uint32_t warp = 0; warp < warps_.size(); ++warp) {
        trace_->warp_state(cycle_, warp, states[warp].state, std::move(states[warp].detail));
    }
}

// One warp's issue attempt (trash/rfc10.md section 1):
//   1. metadata panics; the depbar wait, on the counters' Q (a completion
//      presented at t is seen from t + 1); the plan (its read round and
//      write booking, from metadata and the profile); register ranges and
//      the sanitizer's hazard checks against the cycles the round samples
//      at (at every attempt, as before: contention must not hide a missing
//      stall);
//   2. admission: structural_refusal, on metadata and resource state alone.
//      A refusal books nothing and reads nothing;
//   3. acceptance: the read round into the issue register, the write
//      booking (both in future cycles, past the issue register),
//      the metadata share of each unit's work and, for a memory / engine /
//      mbar instruction, its entry into the dispatch pipeline, which carries
//      it to its unit as its operands arrive. Operand-value panics
//      (alignment, bounds) happen there, at dispatch.
// The issue unit's own state (stall count, the SFU and the arithmetic operand
// latches) changes in place; everything else goes into w.issue, which its
// drivers apply -- the FIFO's pop and the warp's pc included
// (step_front_end).
Machine::IssueAttempt Machine::try_issue(Warp& warp, Wires& w) {
    const auto inst = head_instruction(warp);
    sanitizer_.record_issue_attempt(warp.id);
    panic_if_malformed(warp, inst);
    if (const auto bar = depbar_wait(inst, depbar_.cur().at(warp.id)); bar.has_value()) {
        return {false, WarpState::depbar, "sb" + std::to_string(*bar)};
    }
    const auto plan = issue_plan(inst);
    check_issue_hazards(warp, inst);
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
        // (The pointer is a uniform register: check_issue_hazards checks it
        // against the URF.)
        const auto base = is_acc_load(inst.opcode) ? inst.dst : inst.src1;
        const auto width = inst.opcode == Opcode::acc_ld_v8 || inst.opcode == Opcode::acc_st_v8 ? 8u : 1u;
        if (static_cast<unsigned>(base) + width > 256) panic(warp, inst, "accumulator register encoding wraps");
    }
    if (inst.opcode == Opcode::bra &&
        (inst.control.predicate != 7 || inst.control.predicate_invert)) {
        panic(warp, inst, "bra is unconditional");
    }
    if ((inst.control.predicate != 7 || inst.control.predicate_invert) &&
        inst.opcode != Opcode::bra_uni) {
        panic(warp, inst, "only bra.uni may be predicated before divergent execution exists");
    }
    if (inst.opcode == Opcode::sread && inst.imm != static_cast<int32_t>(SpecialReg::lane_id) &&
        inst.imm != static_cast<int32_t>(SpecialReg::thread_id)) {
        panic(warp, inst, "sread reads LANE_ID / THREAD_ID (WARP_ID / N_WARPS are read with usread)");
    }
    if (inst.opcode == Opcode::usread && inst.imm != static_cast<int32_t>(SpecialReg::warp_id) &&
        inst.imm != static_cast<int32_t>(SpecialReg::n_warps)) {
        panic(warp, inst, "usread reads WARP_ID / N_WARPS (LANE_ID / THREAD_ID are read with sread)");
    }
}

// Register and predicate ranges (machine panics: properties of the
// instruction), then the sanitizer's hazard checks, at the cycles the read
// rounds sample and its destination lands at (src/rf.hpp). The machine looks
// at no operand value; the sanitizer's transfer-versus-matmul check looks at
// the tmem pointer the round will sample (it has global knowledge).
void Machine::check_issue_hazards(const Warp& warp, const Instruction& inst) const {
    for (const auto reg : source_registers(inst)) {
        if (reg >= kernel_.register_count) panic(warp, inst, "source register out of range");
    }
    const auto destination = destination_register(inst);
    if (destination.has_value() && *destination >= kernel_.register_count) {
        panic(warp, inst, "destination register out of range");
    }
    // Uniform registers: in range, then RAW (each source's sample at issue +
    // 1) and WAW (the write lands L + Wu after issue) against the uniform
    // horizons, at every attempt.
    const auto uniform_sources_read = uniform_sources(inst);
    for (const auto reg : uniform_sources_read) {
        if (reg >= config_.urf_registers) panic(warp, inst, "uniform source register out of range");
    }
    const auto uniform_dst = uniform_destination(inst);
    if (uniform_dst.has_value() && *uniform_dst >= config_.urf_registers) {
        panic(warp, inst, "uniform destination register out of range");
    }
    // Each uniform register must be written by its own round's sample (a
    // tma's and an mma.async's later rounds sample later: src/rf.hpp
    // uniform_read_rounds).
    const auto rounds = uniform_read_rounds(inst);
    for (uint32_t round = 0; round < rounds.size(); ++round) {
        sanitizer_.check_uniform_raw(site(warp, inst), rounds[round], cycle_ + issue_to_uniform_round_sample(round));
    }
    if (uniform_dst.has_value()) {
        sanitizer_.check_uniform_waw(site(warp, inst), *uniform_dst, cycle_ + issue_to_urf_visible(config_));
    }
    if (is_setp(inst.opcode) || is_usetp(inst.opcode) || is_mbar_predicate_writer(inst.opcode)) {
        if (inst.dst >= 7) panic(warp, inst, "predicate destination is not p0..p6");
        sanitizer_.check_pred_waw(site(warp, inst), inst.dst);
    }
    if (inst.opcode == Opcode::bra_uni && inst.control.predicate != 7) {
        sanitizer_.check_pred_raw(site(warp, inst), inst.control.predicate, cycle_ + issue_to_predicate_sample);
    }
    check_register_hazards(warp, inst);
    if (is_acc_transfer(inst.opcode)) {
        // Racing accepted MMA work is UB (cases 01/02): no ownership hardware
        // exists, but the simulator has global knowledge and diagnoses it, at
        // every attempt (contention must not hide it) and again at dispatch
        // (an MMA command issued just before is accepted by the MMA only
        // then). The words are the ones the pointer the round samples next
        // cycle names (a region outside tmem is the dispatch PANIC's).
        const uint32_t width = inst.opcode == Opcode::acc_ld_v8 || inst.opcode == Opcode::acc_st_v8 ? 8 : 1;
        const auto pointer = uniform_at_next_sample(warp.id, inst.src0);
        if (const auto words = tmem_region_words(pointer, static_cast<uint32_t>(inst.imm), config_.lanes, width);
            words.has_value()) {
            sanitizer_.check_acc_transfer_vs_mma(site(warp, inst), tmem_region(pointer, inst, *words));
        }
    }
}

// What a transfer touches, for the sanitizer: the region and its tmem words.
TmemRegion Machine::tmem_region(uint32_t pointer, const Instruction& inst, std::vector<uint32_t> words) const {
    TmemRegion region;
    region.pointer = pointer;
    region.row_words = static_cast<uint32_t>(inst.imm);
    region.rows = config_.lanes;
    region.columns = inst.opcode == Opcode::acc_ld_v8 || inst.opcode == Opcode::acc_st_v8 ? 8 : 1;
    std::sort(words.begin(), words.end());
    region.words = std::move(words);
    return region;
}

// What an accepted instruction books, from metadata and the profile: its one
// read round (src/rf.hpp; kernel load refused any round that does not fit
// its banks' read ports), vector and uniform registers, and a fixed result's
// write. acc.st reads only its uniform coordinates at issue (its data beat by
// beat) and stg.v8 only its base (its data in the LSU's RF read stage).
IssuePlan Machine::issue_plan(const Instruction& inst) const {
    IssuePlan plan;
    auto sources = source_registers(inst);
    if (inst.opcode == Opcode::acc_st || inst.opcode == Opcode::acc_st_v8) {
        sources.clear();
    } else if (inst.opcode == Opcode::stg_v8) {
        sources = {inst.src0};
    }
    plan.reads = rf_read_round(sources);
    // The issue round's uniform registers: round 0 of src/rf.hpp's plan (a
    // tma's and an mma.async's later rounds are the round counter's).
    const auto rounds = uniform_read_rounds(inst);
    if (!rounds.empty()) plan.uniform_reads = rounds.front();
    const auto destination = destination_register(inst);
    if (uses_arithmetic_pipeline(inst) && destination.has_value()) {
        plan.write_reg = destination;
        plan.write_cycle = cycle_ + issue_to_port(inst, config_);
    }
    return plan;
}

// The resource state admission may see, as this cycle leaves it (c): the RF
// write calendar and this cycle's RF read / write decisions, unit occupancy,
// queue and slot space less what the dispatch pipeline's instructions will
// take (dispatch_claims). Instruction fields select WHICH resource (a
// register number's bank, an accumulator's tmem bank); no register content
// is read.
ResourceView Machine::resource_view(
    const Warp& warp, const Instruction& inst, const IssuePlan& plan, const Wires& w) const {
    const auto claims = dispatch_claims();
    ResourceView view;
    for (const auto reg : plan.reads) {
        if (w.rf_read.banks.at(reg % config_.rf_banks).inhibit_simt) view.read_inhibited = true;
    }
    view.sfu_free = cycle_ + issue_to_operands(config_) >= sfu_busy_until_;
    if (plan.write_reg.has_value()) {
        view.write_slot_free = write_slot_free(*plan.write_reg, plan.write_cycle);
        view.write_inhibited = w.rf_write.banks.at(*plan.write_reg % config_.rf_banks).inhibit_simt;
    }
    view.wait_slot_free = free_wait_slot_count(w) > claims.wait_slots;
    // A queue slot is counted as taken from a tma's accept: the queue after
    // this cycle, the command its operand read pushes now, the commands it
    // still holds. So the push always fits.
    view.tma_queue_room = w.tma_next.commands.size() + (w.tma_pushed.has_value() ? 1 : 0) +
        tma_commands_being_read() < config_.tma_queue_commands;
    view.urf_ports_held = urf_read_ports_held();
    view.mma_queue_room = w.mma_next.queue.size() + claims.mma_commands < config_.mma_queue_commands;
    if (is_acc_transfer(inst.opcode)) {
        const bool load = is_acc_load(inst.opcode);
        view.acc_engine_free = acc_engine_available(w, load) && claims.acc[load ? 0 : 1] == 0;
    }
    view.no_active_lanes = warp.active_mask == 0;
    view.unserviced_read = warp_unserviced_read(w, warp.id);
    view.unserviced_write = warp_unserviced_write(w, warp.id);
    if (inst.opcode == Opcode::fence_sh) view.own_shmem_at_ram = own_shmem_at_ram(w, warp.id);
    view.load_slot_free = free_load_slot_count() > claims.load_slots;
    view.store_slot_free = free_store_slot_count(w) > claims.store_slots;
    view.probe_register_free =
        (!mbar_probe_.cur().has_value() || w.shmem.a_read.user == AReadUser::probe) && claims.probes == 0;
    view.simt_rmw_register_free =
        (!simt_rmw_.cur().has_value() || w.shmem.simt_rmw_started) && claims.simt_rmw == 0;
    view.lsu_slot_free = lsu_commands_.cur().size() - w.released_lsu_commands.size() + claims.lsu_commands <
        config_.lsu_active_commands;
    view.assembly_free = free_assembly_register_count(w) > claims.lsu_commands;
    return view;
}

// A refused attempt: the warp state and reason for the trace, and its counter.
Machine::IssueAttempt Machine::refuse(Refusal refusal) {
    switch (refusal) {
    case Refusal::rf_read_inhibited:
        trace_->rf_simt_read_inhibited();
        return {false, WarpState::structural, "RF read inhibited"};
    case Refusal::urf_read_ports_held:
        trace_->urf_read_ports_held();
        return {false, WarpState::structural, "uniform read ports held"};
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
    case Refusal::acc_engine_busy: return {false, WarpState::structural, "acc transfer engine busy"};
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

// An accepted instruction leaves its FIFO (the head pops, the warp's pc
// moves to the next word: step_front_end). Fixed arithmetic books its
// bundle; a branch goes into the issue register, where it resolves next
// cycle (resolve_branch); a halt stops the warp's front end now (its FIFO is
// cleared, nothing more is fetched or issued); fence.sh and nop are complete
// now; everything else books its operand reads and enters the dispatch
// pipeline.
void Machine::accept(Warp& warp, const Instruction& inst, const IssuePlan& plan, Wires& w) {
    w.issue.popped = warp.id;
    if (uses_arithmetic_pipeline(inst)) {
        accept_arithmetic(warp, inst, plan, w);
        return;
    }
    if (uses_urf_pipeline(inst.opcode)) {
        accept_uniform(warp, inst, plan, w);
        return;
    }
    switch (inst.opcode) {
    case Opcode::bra:
    case Opcode::bra_uni:
        w.issue.branch = IssuedBranch{warp.id, head_pc(warp.id), inst};
        break;
    case Opcode::fence_sh:
        break; // its admission was the whole of it (own_shmem_at_ram)
    case Opcode::nop:
        break; // its wait mask held it until now; its stall is set below
    case Opcode::halt:
        w.issue.halt = true;
        break;
    default:
        // A tma's and an mma.async's later uniform rounds are the round
        // counter's (src/rf.hpp uniform_read_rounds); a tma reaches its queue
        // through the TMA operand read, an mma.async through the dispatch
        // pipeline like every other unit's instruction.
        if (uniform_read_rounds(inst).size() > 1) load_urf_rounds(warp, inst, w);
        if (!is_tma(inst.opcode)) accept_for_dispatch(warp, inst, plan, w);
        break;
    }
    check_issue_latency(warp, inst);
    warp.stall_until = cycle_ + inst.control.stall;
    trace_->issue(cycle_, warp.id, head_pc(warp.id), inst);
}

// Fixed-latency work books its result write; its read round goes into the
// issue register (the next cycle's samples). step_rf_reads captures the
// operands at the sample and executes the instruction when they are at the
// RF output. A booking of the result's bank is what sets that bank's
// async_owed bit; the read round sets its banks' read_owed bits.
void Machine::accept_arithmetic(Warp& warp, const Instruction& inst, const IssuePlan& plan, Wires& w) {
    // One accepted issue per cycle, each living issue_to_operands + 1 cycles.
    if (arithmetic_reads_.size() > issue_to_operands(config_)) {
        throw SimError("internal RF read pipeline exceeded its bounded depth");
    }
    if (plan.write_reg.has_value()) {
        reserve_write(*plan.write_reg, plan.write_cycle);
        w.issue.rf_write_bank = *plan.write_reg % config_.rf_banks;
    }
    w.issue.rf_read_banks.assign(config_.rf_banks, 0);
    for (const auto reg : plan.reads) ++w.issue.rf_read_banks[reg % config_.rf_banks];
    if (is_sfu(inst.opcode)) {
        const auto occupancy = (config_.lanes + config_.sfu_lanes - 1) / config_.sfu_lanes;
        sfu_busy_until_ = cycle_ + issue_to_operands(config_) + occupancy;
        trace_->sfu_accept(occupancy);
    }
    for (const auto reg : plan.reads) {
        sanitizer_.record_reg_read_until(warp.id, reg, cycle_ + issue_to_sample + 1);
    }
    if (plan.write_reg.has_value()) {
        sanitizer_.record_reg_ready(warp.id, *plan.write_reg, cycle_ + issue_to_visible(inst, config_));
    } else {
        sanitizer_.record_pred_ready(warp.id, inst.dst, cycle_ + issue_to_predicate_visible(inst, config_));
    }
    ArithmeticRead pending;
    pending.issue_cycle = cycle_;
    pending.warp = warp.id;
    pending.pc = head_pc(warp.id);
    pending.instruction = inst;
    pending.reads = plan.reads;
    pending.operands.resize(plan.reads.size());
    pending.uniform_reads = plan.uniform_reads;
    pending.uniform_operands.resize(plan.uniform_reads.size());
    arithmetic_reads_.push_back(std::move(pending));
    warp.stall_until = cycle_ + inst.control.stall;
    trace_->issue(cycle_, warp.id, head_pc(warp.id), inst);
}

// Uniform arithmetic, r2ur and usetp: the write enters the URF write
// pipeline's stage 0 (the issue register's cycle, next cycle), where its
// operands are sampled; it presents at the URF write port at issue + L and is
// visible from issue + L + Wu + 1 (src/rf.hpp). usetp presents its predicate
// at the predicate file then instead, in the file from issue + L + 1. r2ur's
// vector source is part of the round: it owns its bank's read port in the
// issue register's cycle like any read. No write calendar and no structural
// check: one issue per cycle is one write per pipeline stage. r2ur reads lane
// 0, so like every lane-0 operation it is UB under a partial or empty active
// mask (the sanitizer's PANIC, backlog D12.1); uniform arithmetic (usetp
// included) reads no lanes.
void Machine::accept_uniform(Warp& warp, const Instruction& inst, const IssuePlan& plan, Wires& w) {
    if (inst.opcode == Opcode::r2ur) sanitizer_.check_uniform_mask(site(warp, inst), warp.active_mask, full_mask());
    w.issue.rf_read_banks.assign(config_.rf_banks, 0);
    for (const auto reg : plan.reads) {
        ++w.issue.rf_read_banks[reg % config_.rf_banks];
        sanitizer_.record_reg_read_until(warp.id, reg, cycle_ + issue_to_sample + 1);
    }
    if (is_usetp(inst.opcode)) {
        sanitizer_.record_pred_ready(warp.id, inst.dst, cycle_ + issue_to_uniform_predicate_visible(config_));
    } else {
        sanitizer_.record_uniform_ready(warp.id, inst.dst, cycle_ + issue_to_urf_visible(config_));
    }
    UrfWrite write;
    write.warp = warp.id;
    write.pc = head_pc(warp.id);
    write.instruction = inst;
    w.issue.urf_write = std::move(write);
    warp.stall_until = cycle_ + inst.control.stall;
    trace_->issue(cycle_, warp.id, head_pc(warp.id), inst);
}

// An lds / sts with no active lane is absent (rfc12 section 3): no slot, no
// depbar, no order, no dispatch. Everything else that reaches here goes to
// its unit.
bool Machine::needs_dispatch(const Instruction& inst, uint64_t active_mask) {
    return !((inst.opcode == Opcode::lds || inst.opcode == Opcode::sts) && active_mask == 0);
}

// A memory / engine / mbar instruction accepted now. What depends on
// metadata alone happens now: the depbar it sets (so the warp's next
// instruction sees it), the sanitizer's records for its destination (not
// ready until written) and its asynchronously read sources (read-owned
// until each beat). Its read round goes into the issue register (sampled at
// issue + 1), and it enters the dispatch pipeline; at issue + 1 + Lr its
// unit takes it with its operands (hand_over). Admission secured that unit's
// space.
void Machine::accept_for_dispatch(Warp& warp, const Instruction& inst, const IssuePlan& plan, Wires& w) {
    const auto opcode = inst.opcode;
    const bool sets_depbar = is_shared_load(opcode) || opcode == Opcode::sts || is_lsu_opcode(opcode) ||
        is_acc_transfer(opcode) || is_mma_async(opcode) || is_atomic(opcode);
    if (sets_depbar && inst.control.set_depbar >= 6) {
        if (is_mma_async(opcode)) {
            panic(warp, inst, opcode_name(opcode, &config_.mma_shapes) + " has no operand-consumed depbar");
        }
        if (is_acc_transfer(opcode)) panic(warp, inst, "accumulator transfer has no depbar");
        if (is_lsu_opcode(opcode)) panic(warp, inst, "global memory operation has no depbar");
        panic(warp, inst, opcode_name(opcode) + " has no depbar");
    }
    if (!needs_dispatch(inst, warp.active_mask)) return;
    if (sets_depbar) admit_depbar(w, warp.id, inst.control.set_depbar);
    const auto not_ready = std::numeric_limits<uint64_t>::max();
    if (const auto dst = destination_register(inst); dst.has_value()) {
        const auto width = opcode == Opcode::acc_ld_v8 || opcode == Opcode::ldg_v8 ? 8u : 1u;
        for (uint32_t j = 0; j < width; ++j) sanitizer_.record_reg_ready(warp.id, *dst + j, not_ready); // until written
    }
    if (is_mbar_predicate_writer(opcode)) sanitizer_.record_pred_ready(warp.id, inst.dst, not_ready); // until the probe returns
    // WAR protection: a source read beat by beat stays read-owned until its beat.
    if (opcode == Opcode::acc_st || opcode == Opcode::acc_st_v8 || opcode == Opcode::stg_v8) {
        const auto width = opcode == Opcode::acc_st ? 1u : 8u;
        for (uint32_t j = 0; j < width; ++j) sanitizer_.record_async_reader_add(warp.id, inst.src1 + j);
    }
    w.issue.rf_read_banks.assign(config_.rf_banks, 0);
    for (const auto reg : plan.reads) {
        ++w.issue.rf_read_banks[reg % config_.rf_banks];
        sanitizer_.record_reg_read_until(warp.id, reg, cycle_ + issue_to_sample + 1);
    }
    DispatchEntry entry;
    entry.issue_cycle = cycle_;
    entry.warp = warp.id;
    entry.pc = head_pc(warp.id);
    entry.active_mask = warp.active_mask;
    entry.instruction = inst;
    entry.reads = plan.reads;
    entry.operands.resize(plan.reads.size());
    // Every round's uniform registers, in read order: the issue round's
    // (plan.uniform_reads) are latched at its sample, an mma.async's urC by
    // the round counter's round 1.
    for (const auto& round : uniform_read_rounds(inst)) {
        entry.uniform_reads.insert(entry.uniform_reads.end(), round.begin(), round.end());
    }
    entry.uniform_operands.resize(entry.uniform_reads.size());
    w.issue.dispatch = std::move(entry);
}

// A tma or an mma.async accepted now (trash/rfc17.md section 4): it loads the
// URF's round counter, whose round 0 samples next cycle (the issue
// register's) and whose later rounds one per cycle after that. A tma from now
// on counts as a slot of the TMA queue (resource_view); it reads no vector
// register and sets no depbar. Issue accepted it only with the counter free
// after this cycle (its last round, if any, is sampled now:
// urf_read_ports_held).
void Machine::load_urf_rounds(Warp& warp, const Instruction& inst, Wires& w) {
    UrfReadRounds rounds;
    rounds.header.issue_cycle = cycle_;
    rounds.header.warp = warp.id;
    rounds.header.pc = head_pc(warp.id);
    rounds.header.active_mask = warp.active_mask;
    rounds.header.instruction = inst;
    w.issue.urf_rounds = std::move(rounds);
}

// What the dispatch pipeline's instructions will take when they reach their
// units: the claims every later admission subtracts from the space it sees.
Machine::DispatchClaims Machine::dispatch_claims() const {
    DispatchClaims claims;
    for (const auto& op : dispatch_pipe_.cur()) {
        const auto opcode = op.instruction.opcode;
        if (is_shared_load(opcode) || is_atomic(opcode)) ++claims.load_slots;
        if (opcode == Opcode::sts) ++claims.store_slots;
        if (is_mbar_rmw(opcode) || is_atomic(opcode)) ++claims.simt_rmw;
        if (is_mbar_predicate_writer(opcode)) ++claims.probes;
        if (is_mbar_try_wait(opcode)) ++claims.wait_slots;
        if (is_lsu_opcode(opcode)) ++claims.lsu_commands;
        if (is_mma_command(opcode)) ++claims.mma_commands;
        if (is_acc_transfer(opcode)) ++claims.acc[is_acc_load(opcode) ? 0 : 1];
    }
    return claims;
}

// ===========================================================================
// Dispatch: the dispatch pipeline's end. The unit takes the instruction with
// its operands -- latched at its RF sample, at the RF output now -- exactly
// as issue handed it over before I6a.
// ===========================================================================

void Machine::dispatch_issued(Wires& w) {
    const auto& pipe = dispatch_pipe_.cur();
    if (pipe.empty() || pipe.front().issue_cycle + issue_to_dispatch(config_) != cycle_) return;
    w.dispatched = true;
    hand_over(pipe.front(), w);
}

void Machine::hand_over(const DispatchEntry& op, Wires& w) {
    if (is_mma_async(op.instruction.opcode)) {
        dispatch_mma(op, w);
        return;
    }
    switch (op.instruction.opcode) {
    case Opcode::mma_commit:
        dispatch_mma(op, w);
        break;
    case Opcode::acc_ld:
    case Opcode::acc_ld_v8:
    case Opcode::acc_st:
    case Opcode::acc_st_v8:
        dispatch_acc_transfer(op, w);
        break;
    case Opcode::lds:
    case Opcode::lds_uni:
        dispatch_shared_load(op, w);
        break;
    case Opcode::sts:
        dispatch_shared_store(op, w);
        break;
    case Opcode::ldg:
    case Opcode::stg:
    case Opcode::ldg_v8:
    case Opcode::stg_v8:
    case Opcode::ldg_uni:
        dispatch_lsu(op, w);
        break;
    case Opcode::mbar_init_i:
    case Opcode::mbar_init_r:
    case Opcode::mbar_arrive:
    case Opcode::mbar_arrive_expect_tx_i:
    case Opcode::mbar_arrive_expect_tx_r:
        dispatch_mbar_rmw(op, w);
        break;
    case Opcode::mbar_test_wait_i:
    case Opcode::mbar_test_wait_r:
    case Opcode::mbar_try_wait_i:
    case Opcode::mbar_try_wait_r:
        dispatch_mbar_wait(op, w);
        break;
    case Opcode::atom_add_uni:
    case Opcode::atom_and_uni:
    case Opcode::atom_xor_uni:
    case Opcode::atom_exch_uni:
        dispatch_atomic(op, w);
        break;
    default:
        panic(op, "unknown opcode");
    }
}

void Machine::dispatch_mma(const DispatchEntry& op, Wires& w) {
    const auto& inst = op.instruction;
    if (inst.opcode == Opcode::mma_commit) {
        MmaCommand command;
        command.id = next_mma_command_id_++;
        command.commit = true;
        command.warp = op.warp;
        command.mbar_address = mbar_address_operand(op);
        w.dispatch.mma = std::move(command);
        return;
    }
    // The shape is the profile's (its list gives each opcode pair a shape);
    // its existence was checked at kernel load (check_mma_shapes).
    const auto shape = *mma_shape_of(config_, inst.opcode);
    const auto a = uniform_operand(op, inst.src0);
    const auto b = uniform_operand(op, inst.src1);
    // urC, from the round counter's round 1 (src/rf.hpp uniform_read_rounds):
    // the result's tmem pointer, its rows the shape's n words. Every word it
    // writes must be in tmem (the machine's PANIC, as an operand outside
    // shmem); a pointer that breaks the alignment rule is the sanitizer's
    // (src/layout.hpp tmem_pointer_aligned). The machine writes whatever
    // words the layout names.
    const auto c = uniform_operand(op, inst.src2);
    auto c_words = tmem_region_words(c, shape.n, shape.m, shape.n);
    if (!c_words.has_value()) panic(op, "MMA result region is outside tmem");
    sanitizer_.check_tmem_pointer(site(op), opcode_name(inst.opcode, &config_.mma_shapes) + " result", c,
        shape.n, shape.n);
    // The spacings are instruction constants, checked at kernel load
    // (check_mma_shapes): under swizzle64 a power of two, 16 or more.
    const auto a_spacing = mma_row_spacing(inst, true);
    const auto b_spacing = mma_row_spacing(inst, false);
    // A swizzle64 operand starts at a 16-byte piece. Every byte it reads must
    // be in shmem: the hull of the addresses the walk reads (under swizzle64
    // a row's piece moves inside its row, so not simply base + rows x spacing).
    const auto inside = [&](uint32_t base, uint32_t spacing, bool a_operand) {
        return base < shmem_.size() &&
            mma_operand_hull(config_, shape, base, spacing, a_operand).second <= shmem_.size();
    };
    if ((config_.mma_layout == TileLayout::swizzle64 && ((a | b) & 15u) != 0) ||
        !inside(a, a_spacing, true) || !inside(b, b_spacing, false))
        panic(op, "MMA operand alignment or footprint is outside shmem");
    // The machine reads whatever the address-keyed rule names; an operand
    // whose row-0 bytes are not inside one legally placed row is the
    // sanitizer's PANIC (src/layout.hpp: a row-0 line on a multiple of the
    // spacing and of 64).
    if (config_.mma_layout == TileLayout::swizzle64) {
        sanitizer_.check_swizzle64_operand(site(op), "A", a, a_spacing, 2 * shape.k);
        sanitizer_.check_swizzle64_operand(site(op), "B", b, b_spacing, 2 * shape.k);
    }
    MmaCommand command;
    command.id = next_mma_command_id_++;
    command.init = is_mma_init(inst.opcode);
    command.shape = shape;
    command.warp = op.warp;
    command.c_pointer = c;
    command.depbar = inst.control.set_depbar;
    command.a_base = a;
    command.b_base = b;
    command.a_spacing = a_spacing;
    command.b_spacing = b_spacing;
    TmemRegion result;
    result.pointer = c;
    result.row_words = shape.n;
    result.rows = shape.m;
    result.columns = shape.n;
    std::sort(c_words->begin(), c_words->end());
    result.words = std::move(*c_words);
    if constexpr (NANOSM_SANITIZER != 0) {
        // The shmem words each operand reads, for the row spacing check
        // (computed for the sanitizer only). The dense diagnostic layout
        // reads no spacing: none.
        const auto operand_words = [&](uint32_t base, uint32_t spacing, bool a_operand) {
            std::vector<uint32_t> words;
            if (config_.mma_layout != TileLayout::swizzle64) return words;
            for (uint32_t outer = 0; outer < (a_operand ? shape.m : shape.n); ++outer) {
                for (uint32_t k = 0; k < shape.k; ++k) {
                    words.push_back(mma_operand_address(config_, shape, base, spacing, a_operand, outer, k) / 4);
                }
            }
            std::sort(words.begin(), words.end());
            words.erase(std::unique(words.begin(), words.end()), words.end());
            return words;
        };
        sanitizer_.record_mma_row_spacing(site(op), command.id, command.init, result,
            operand_words(a, a_spacing, true), a_spacing, operand_words(b, b_spacing, false), b_spacing);
    }
    sanitizer_.record_mma_accept(site(op), command.id, result, command.depbar,
        mma_operand_span(a, a_spacing, true, shape), mma_operand_span(b, b_spacing, false, shape));
    trace_->mma_command(cycle_, command.id, op.warp, command.init);
    w.dispatch.mma = std::move(command);
}

// acc.ld / acc.st: the pointer arrives now; the transfer runs one beat per
// column in its engine (step_acc_transfers, and decide_rf_reads for acc.st):
// lane l moves row l of the region at the pointer (rows of S words, the
// instruction's), beat j column j. Every word it touches must be in tmem
// (the machine's PANIC); the alignment rule is the sanitizer's
// (src/layout.hpp tmem_pointer_aligned).
void Machine::dispatch_acc_transfer(const DispatchEntry& op, Wires& w) {
    const auto& inst = op.instruction;
    const bool load = is_acc_load(inst.opcode);
    const uint32_t width = inst.opcode == Opcode::acc_ld_v8 || inst.opcode == Opcode::acc_st_v8 ? 8 : 1;
    const uint8_t rf_base = load ? inst.dst : inst.src1;
    const auto row_words = static_cast<uint32_t>(inst.imm);
    const auto pointer = uniform_operand(op, inst.src0);
    auto words = tmem_region_words(pointer, row_words, config_.lanes, width);
    if (!words.has_value() || static_cast<unsigned>(rf_base) + width > kernel_.register_count) {
        panic(op, "accumulator transfer region outside tmem or RF range out of bounds");
    }
    const auto region = tmem_region(pointer, inst, std::move(*words));
    sanitizer_.check_tmem_pointer(site(op), opcode_name(inst.opcode), pointer, row_words, width);
    // Racing accepted MMA work (check_issue_hazards): an MMA command issued
    // before this transfer was accepted by the MMA before this dispatch
    // (dispatch keeps issue order).
    sanitizer_.check_acc_transfer_vs_mma(site(op), region);
    sanitizer_.check_acc_transfer_overlap(site(op), region, load);
    AccTransfer transfer;
    transfer.load = load;
    transfer.warp = op.warp;
    transfer.rf_base = rf_base;
    transfer.depbar = inst.control.set_depbar;
    transfer.pointer = pointer;
    transfer.row_words = row_words;
    transfer.width = width;
    sanitizer_.record_acc_transfer_accept(site(op), load, region, inst.control.set_depbar);
    w.dispatch.acc = DispatchDecision::AccAdmission{load, std::move(transfer)};
}

// lds / lds.uni (trash/rfc10.md section 2, rfc12 section 3): the address
// arrives now and the bank rounds come from it (active lanes only). The load
// takes the load slot admission secured; A-read serves it from the next
// cycle; its gathered vector waits in the slot for the RF write decision. No
// RF booking: the depbar completes it. lds.uni ignores the mask (its base is
// a uniform register; the mask gates only its RF write's lanes): a partial or
// empty mask is the sanitizer's PANIC (backlog D12, in uniform_operand).
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

// sts (rfc12 section 3): address and data arrive now (its RF reads were
// booked with the issue) into the store slot admission secured; A-write
// serves it from the next cycle. Two lanes on one word is UB (the
// sanitizer's PANIC, backlog D12); the machine just writes both, in round
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

// ldg / stg / ldg.v8 / stg.v8 / ldg.uni: the addresses arrive and are
// checked now; the command takes the slot and assembly register admission
// secured.
void Machine::dispatch_lsu(const DispatchEntry& op, Wires& w) {
    const auto& inst = op.instruction;
    auto& counters = trace_->lsu_counters();
    auto command = plan_lsu_command(op);
    command.id = next_lsu_command_id_++;
    command.depbar = inst.control.set_depbar;
    command.assembly = *free_assembly_register(w);
    DispatchDecision::LsuAdmission admission;
    switch (command.kind) {
    case LsuKind::vector_load:
        // Ordered vector streams 1..read_streams-1, round-robin.
        command.stream = 1 + next_vector_stream_++ % (config_.lsu_read_streams - 1);
        ++counters.lsu_vector_loads;
        counters.lsu_useful_bytes += 32ull * config_.lanes;
        break;
    case LsuKind::vector_store:
        ++counters.lsu_vector_stores;
        counters.lsu_useful_bytes += 32ull * config_.lanes;
        break;
    case LsuKind::scalar_load:
        ++counters.lsu_scalar_loads;
        counters.lsu_useful_bytes += 4ull * config_.lanes;
        break;
    case LsuKind::scalar_store:
        // Its RF read beat (booked with the address operand) lands the store
        // data in the command's assembly register.
        admission.store_data = capture(op, inst.src1);
        probe(LsuEdge::rf_read_beat, command.id, inst.src1, config_.lanes * 4);
        ++counters.lsu_scalar_stores;
        counters.lsu_useful_bytes += 4ull * config_.lanes;
        break;
    case LsuKind::uniform_load:
        ++counters.lsu_uniform_loads;
        counters.lsu_useful_bytes += 4;
        break;
    }
    trace_->lsu_command_begin(cycle_, command.id, op.warp, opcode_name(inst.opcode));
    lsu_phases_[command.id].admit_cycle = cycle_;
    // Sanitizer: every admitted command owns one assembly register for its
    // whole life, so the admitted set can never exceed the pool.
    sanitizer_.check_lsu_admitted(lsu_commands_.cur().size() - w.released_lsu_commands.size() + 1);
    admission.command = std::move(command);
    w.dispatch.lsu = std::move(admission);
}

// ===========================================================================
// The TMA operand read's end (UrfReadRounds in src/sim.hpp; the timing is
// src/rf.hpp tma_operand_rounds). The pair in the URF read stages' last stage
// is at the URF output now: it goes into the holding register (round 0 starts
// the command: its instruction's fields, the buffer and the barrier, the
// coordinates zero until supplied). With the command's last pair the command
// is pushed into the TMA queue now; the engine sees it from the next cycle.
// ===========================================================================

void Machine::arrive_tma_operands(Wires& w) {
    const auto& pairs = tma_read_pairs_.cur();
    if (pairs.empty() || !pairs.back().has_value()) return;
    const auto& pair = *pairs.back();
    auto holding = tma_holding_.cur();
    if (pair.round == 0) {
        if (holding.has_value()) {
            throw SimError("internal: a tma's operands arrived while the holding register held another command");
        }
        holding = TmaReadHolding{};
        holding->header = pair.header.value();
        holding->buffer = pair.words.at(0);
        holding->barrier = pair.words.at(1);
    } else {
        if (!holding.has_value()) throw SimError("internal: a tma's coordinates arrived without their command");
        for (std::size_t i = 0; i < pair.words.size(); ++i) {
            holding->coordinates.at(2 * (pair.round - 1) + i) = static_cast<int32_t>(pair.words[i]);
        }
    }
    if (pair.last) {
        w.tma_pushed = tma_command_from(*holding, w);
        holding.reset();
    }
    w.tma_holding_changed = true;
    w.tma_holding_after = std::move(holding);
}

// The holding register's command, complete now, as the TMA queue takes it
// (what dispatch did before rfc17): the machine's panics on its operands, and
// the sanitizer's checks -- the uniform operation's partial-mask rule, the
// address operands' low bits, the descriptor's and the tile's footprint.
Machine::TmaCommand Machine::tma_command_from(const TmaReadHolding& holding, Wires& w) {
    const auto& header = holding.header;
    const auto& inst = header.instruction;
    const IssueSite where{cycle_, header.warp, header.pc, inst, &config_.mma_shapes};
    if (static_cast<uint64_t>(inst.imm) + tma_descriptor_bytes > shmem_.size()) {
        throw_issue_panic(where, "tma descriptor is not in shmem");
    }
    sanitizer_.check_uniform_mask(where, header.active_mask, full_mask());
    sanitizer_.check_tma_operand_alignment(where, holding.buffer, holding.barrier);
    TmaCommand command;
    command.id = next_tma_command_id_++;
    command.warp = header.warp;
    command.pc = header.pc;
    command.instruction = inst;
    command.direction = is_tma_store(inst.opcode) ? TmaDirection::store : TmaDirection::load;
    command.descriptor_address = static_cast<uint32_t>(inst.imm);
    command.coordinates = holding.coordinates;
    // The two addresses' low bits are not wired (src/tma.hpp).
    command.shmem_address = holding.buffer & ~(tma_buffer_alignment - 1);
    command.mbar_address = holding.barrier & ~(tma_barrier_alignment - 1);
    if (static_cast<uint64_t>(command.mbar_address) + 8 > shmem_.size()) {
        throw_issue_panic(where, "tma completion mbar is outside shmem");
    }
    // Sanitizer: the descriptor the engine will read -- memory as of this
    // cycle's edge; anything still travelling to it PANICs in
    // record_tma_accept (missing fence.sh / race, rfc12 section 11) -- and the
    // phase of its completion mbar now, the accept-time peek a TMA's
    // complete_tx is checked against (rfc12 section 8; the hardware reads no
    // mbar). The engine accepts the command now, at its push.
    const auto descriptor_bytes = tma_descriptor_at_accept(command.descriptor_address, w);
    sanitizer_.record_tma_accept(where, command.id, command.descriptor_address, descriptor_bytes,
        command.direction == TmaDirection::load, command.shmem_address, command.mbar_address,
        pending_tma_completion_mbar(command.mbar_address, w));
    trace_->tma_command_begin(cycle_, command.id, header.warp, command.descriptor_address,
        w.tma_next.commands.size() + 1);
    return command;
}

// The commands the TMA operand read holds after this cycle, each counted as
// a slot of the TMA queue since its accept: the one in the round counter, and
// each whose last pair is still in the URF read stages (the pair in the last
// stage arrives now: tma_pushed).
uint32_t Machine::tma_commands_being_read() const {
    const auto& rounds = urf_read_rounds_.cur();
    uint32_t commands = rounds.has_value() && is_tma(rounds->header.instruction.opcode) ? 1 : 0;
    const auto& pairs = tma_read_pairs_.cur();
    for (std::size_t stage = 0; stage + 1 < pairs.size(); ++stage) {
        if (pairs[stage].has_value() && pairs[stage]->last) ++commands;
    }
    return commands;
}

// The round counter's instruction has rounds still to start after the one it
// samples now: an instruction accepted now would sample a uniform register
// in one of them.
bool Machine::urf_read_ports_held() const {
    const auto& rounds = urf_read_rounds_.cur();
    return rounds.has_value() && rounds->round + 1 < uniform_read_rounds(rounds->header.instruction).size();
}

uint32_t Machine::uniform_at_next_sample(uint32_t warp, uint8_t reg) const {
    const auto& landing = urf_pipe_.cur().back();
    if (landing.has_value() && landing->warp == warp && !is_usetp(landing->instruction.opcode) &&
        landing->instruction.dst == reg && landing->value.has_value()) {
        return *landing->value;
    }
    return warps_.at(warp).urf.at(reg);
}

// mbar.init / arrive / arrive.expect_tx: a uniform operation, its operands
// uniform registers whatever the active mask (a partial mask is the
// sanitizer's PANIC, backlog D12, in uniform_operand).
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

// atom.op.uni rD, [urA + imm], urB (trash/rfc13.md). Admission found the warp
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
// serves it from the next cycle and its warp waits until it comes back
// (rfc12 section 9) -- from its issue on, since it was in the dispatch
// pipeline before (probe_in_flight). try_wait installs its wait slot first,
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
    sanitizer_.record_simt_shmem_issue(site(op), SimtShmemHolder::probe, op.warp,
        std::vector<std::vector<uint32_t>>{{request.byte_address}}, 8, false);
    w.dispatch.mbar_probe = std::move(request);
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
               !is_mbar_rmw(opcode) && !is_mbar_try_wait(opcode) && !is_atomic(opcode) &&
               opcode != Opcode::nop && opcode != Opcode::halt) {
        panic(warp, inst, "zero-latency RF writer is unsupported");
    }
}

// ===========================================================================
// Arithmetic pipeline
// ===========================================================================

void Machine::execute_arithmetic(ArithmeticRead& pending, Wires& w) {
    auto& warp = warps_.at(pending.warp);
    const auto& inst = pending.instruction;
    // A panic names the instruction's own pc (the warp's has moved on).
    const auto fail = [&](const std::string& reason) {
        throw_issue_panic(IssueSite{cycle_, pending.warp, pending.pc, inst, &config_.mma_shapes}, reason);
    };
    std::vector<uint32_t> result(config_.lanes, 0);
    // The round holds each source register once; duplicated operand slots
    // broadcast from the same captured latch.
    const auto operand = [&](uint8_t reg, uint32_t lane) {
        for (std::size_t i = 0; i < pending.reads.size(); ++i) {
            if (pending.reads[i] == reg) return pending.operands[i][lane];
        }
        throw SimError("internal arithmetic operand was not in its read round");
    };
    switch (inst.opcode) {
    case Opcode::sread:
        // The per-lane special registers; WARP_ID / N_WARPS are usread's.
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            switch (static_cast<SpecialReg>(inst.imm)) {
            case SpecialReg::lane_id: result[lane] = lane; break;
            case SpecialReg::thread_id: result[lane] = warp.id * config_.lanes + lane; break;
            default: fail("invalid special register for sread (WARP_ID / N_WARPS are read with usread)");
            }
        }
        break;
    case Opcode::mov:
        std::fill(result.begin(), result.end(), static_cast<uint32_t>(inst.imm));
        break;
    case Opcode::ur2r:
        // The uniform operand latched at the sample, into every lane.
        std::fill(result.begin(), result.end(), pending.uniform_operands.at(0));
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
        if (inst.imm < 0 || inst.imm > 31) fail("shift outside 0..31");
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = operand(inst.src0, lane) << inst.imm;
        }
        break;
    case Opcode::shri:
        if (inst.imm < 0 || inst.imm > 31) fail("shift outside 0..31");
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
    case Opcode::setp_lt_s32:
    case Opcode::setp_le_s32:
    case Opcode::setp_gt_s32:
    case Opcode::setp_ge_s32:
    case Opcode::setpi_lt_s32:
    case Opcode::setpi_le_s32:
    case Opcode::setpi_gt_s32:
    case Opcode::setpi_ge_s32:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            const auto rhs = is_setp_immediate(inst.opcode)
                ? static_cast<uint32_t>(inst.imm)
                : operand(inst.src1, lane);
            result[lane] = compare_setp(inst.opcode, operand(inst.src0, lane), rhs);
        }
        break;
    default: fail("non-arithmetic instruction in RF read pipeline");
    }
    // The execution pipeline's E cycles, then the write port it booked.
    const auto port_cycle = pending.issue_cycle + issue_to_port(inst, config_);
    if (is_setp(inst.opcode)) {
        w.new_writebacks.push_back(predicate_writeback(warp, inst.dst, port_cycle, std::move(result)));
    } else {
        w.new_writebacks.push_back(
            register_writeback(warp, inst.dst, port_cycle, std::move(result), opcode_name(inst.opcode)));
    }
}

// The RF read stage (src/rf.hpp): the round of the instruction in the issue
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
        return lane_values(warps_[warp], reg);
    };
    const auto sample_uniform = [&](uint32_t warp, const std::vector<uint8_t>& regs) {
        uniform_reads += static_cast<uint32_t>(regs.size());
        std::vector<uint32_t> values;
        for (const auto reg : regs) values.push_back(warps_[warp].urf.at(reg));
        return values;
    };
    for (auto it = arithmetic_reads_.begin(); it != arithmetic_reads_.end();) {
        if (it->issue_cycle + issue_to_sample == cycle_) {
            for (std::size_t i = 0; i < it->reads.size(); ++i) it->operands[i] = sample(it->warp, it->reads[i]);
            it->uniform_operands = sample_uniform(it->warp, it->uniform_reads);
        }
        if (it->issue_cycle + issue_to_operands(config_) == cycle_) {
            execute_arithmetic(*it, w);
            it = arithmetic_reads_.erase(it);
        } else {
            ++it;
        }
    }
    const auto& pipe = dispatch_pipe_.cur();
    for (std::size_t index = 0; index < pipe.size(); ++index) {
        const auto& op = pipe[index];
        if (op.issue_cycle + issue_to_sample != cycle_ || (op.reads.empty() && op.uniform_reads.empty())) continue;
        auto operands = op.operands;
        for (std::size_t i = 0; i < op.reads.size(); ++i) operands[i] = sample(op.warp, op.reads[i]);
        const auto rounds = uniform_read_rounds(op.instruction);
        w.dispatch_samples.push_back({index, false, std::move(operands), 0,
            sample_uniform(op.warp, rounds.empty() ? std::vector<uint8_t>{} : rounds.front())});
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
            for (const auto reg : sources) values.push_back(warps_[write->warp].urf.at(reg));
            w.urf_sample = std::move(values);
        }
    }
    // The round counter's round now (src/rf.hpp uniform_read_rounds): its
    // registers by operand position, in the counter's warp. A tma's pair and
    // its tag enter the TMA operand read's URF read stages; an mma.async's
    // later round (round 0 is its dispatch entry's own sample, above) joins
    // its dispatch entry, the one issued at the counter's issue cycle.
    if (const auto& rounds = urf_read_rounds_.cur(); rounds.has_value()) {
        const auto& inst = rounds->header.instruction;
        const auto plan = uniform_read_rounds(inst);
        const auto& registers = plan.at(rounds->round);
        if (is_tma(inst.opcode)) {
            TmaReadPair pair;
            pair.round = rounds->round;
            pair.last = rounds->round + 1 == plan.size();
            if (rounds->round == 0) pair.header = rounds->header;
            for (const auto reg : registers) pair.words.push_back(warps_[rounds->header.warp].urf.at(reg));
            uniform_reads += static_cast<uint32_t>(registers.size());
            w.tma_sampled = std::move(pair);
        } else if (rounds->round > 0) {
            const auto entry = std::find_if(pipe.begin(), pipe.end(), [&](const DispatchEntry& op) {
                return op.issue_cycle == rounds->header.issue_cycle && op.warp == rounds->header.warp;
            });
            if (entry == pipe.end()) throw SimError("internal: a later uniform round lost its dispatch entry");
            std::size_t first = 0;
            for (uint32_t round = 0; round < rounds->round; ++round) first += plan[round].size();
            std::vector<uint32_t> values;
            for (const auto reg : registers) values.push_back(warps_[rounds->header.warp].urf.at(reg));
            uniform_reads += static_cast<uint32_t>(registers.size());
            w.dispatch_samples.push_back({static_cast<std::size_t>(entry - pipe.begin()), true, {}, first,
                std::move(values)});
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

// ===========================================================================
// The URF write pipeline (src/rf.hpp, UrfWrite in src/sim.hpp)
// ===========================================================================

// From Q: the write in stage L - 1 is presented at the URF write port now
// (the trace's "UR write"); the one in the last stage is at the RAM now and
// lands at this cycle's clock edge, after every read of the cycle
// (land_urf_write). Each stage holds at most one write, so the one write
// port is never contended. A usetp in stage L - 1 presents its predicate at
// the predicate file instead -- the same bit in every lane, taken by the
// flip-flops at this cycle's edge (land_rf_writes) -- and leaves (step_urf).
void Machine::urf_ram_stage(Wires& w) {
    const auto& pipe = urf_pipe_.cur();
    if (const auto& presented = pipe.at(config_.urf_issue_to_port_cycles - 1); presented.has_value()) {
        if (!presented->value.has_value()) {
            throw SimError("internal: a uniform write reached the URF write port without its value");
        }
        if (is_usetp(presented->instruction.opcode)) {
            auto predicate = predicate_writeback(warps_.at(presented->warp), presented->instruction.dst, cycle_,
                std::vector<uint32_t>(config_.lanes, *presented->value));
            predicate.reason = "usetp";
            w.rf_landings.push_back(std::move(predicate));
        } else {
            trace_->urf_write(cycle_, presented->warp, presented->instruction.dst, *presented->value);
        }
    }
    if (pipe.back().has_value()) w.urf_landing = pipe.back();
}

// The pipeline's next state: every write moves one stage on. The one leaving
// stage 0 latches the operands it sampled now (step_rf_reads). A uniform-
// arithmetic write whose operands are at the URF output now (stage Lu)
// executes on the uniform ALU, and its result rides the remaining stages as
// the write's value (the ALU's Eu stages and the padding to L). r2ur's lane 0
// is at the RF output at stage Lr, which may be the write port's own stage
// (canonical: Lr 2 = L - 1, the RF output wired to the URF write port); the
// RF's read stages carry it until then, so the write carries it as its value
// from the sample on. Issue's accepted writer enters stage 0; the write in
// the last stage leaves (it lands now), and so does a usetp at the write
// port's stage (its predicate is presented now: urf_ram_stage).
void Machine::step_urf(Wires& w) {
    const auto& pipe = urf_pipe_.cur();
    UrfWritePipeline next(pipe.size());
    for (std::size_t stage = 0; stage + 1 < pipe.size(); ++stage) {
        if (!pipe[stage].has_value()) continue;
        if (is_usetp(pipe[stage]->instruction.opcode) && stage + 1 == config_.urf_issue_to_port_cycles) continue;
        auto write = *pipe[stage];
        if (stage == 0) write.operands = w.urf_sample.value();
        if (write.instruction.opcode == Opcode::r2ur) {
            if (stage == 0) write.value = write.operands.at(0);
        } else if (stage == config_.urf_read_latency_cycles) {
            write.value = execute_uniform(write);
        }
        next[stage + 1] = std::move(write);
    }
    next.front() = w.issue.urf_write;
    w.urf_pipe_next = std::move(next);
}

// The uniform ALU: one 32-bit result from the write's latched operands (its
// uniform sources in operand order) and its immediate. Register shifts use
// the low 5 bits, as the vector ALU's. A usetp's result is its predicate, 0
// or 1 (setp's comparisons: unsigned, the .s32 ones signed).
uint32_t Machine::execute_uniform(const UrfWrite& write) const {
    const auto& inst = write.instruction;
    const auto fail = [&](const std::string& reason) {
        throw_issue_panic(IssueSite{cycle_, write.warp, write.pc, inst, &config_.mma_shapes}, reason);
    };
    const auto a = [&] { return write.operands.at(0); };
    const auto b = [&] { return write.operands.at(1); };
    const auto imm = static_cast<uint32_t>(inst.imm);
    switch (inst.opcode) {
    case Opcode::umov: return a();
    case Opcode::umovi: return imm;
    case Opcode::uadd: return a() + b();
    case Opcode::uaddi: return a() + imm;
    case Opcode::usub: return a() - b();
    case Opcode::ushl: return a() << (b() & 31);
    case Opcode::ushr: return a() >> (b() & 31);
    case Opcode::ushli:
    case Opcode::ushri:
        if (inst.imm < 0 || inst.imm > 31) fail("shift outside 0..31");
        return inst.opcode == Opcode::ushli ? a() << inst.imm : a() >> inst.imm;
    case Opcode::uand: return a() & b();
    case Opcode::uor: return a() | b();
    case Opcode::uxor: return a() ^ b();
    case Opcode::uandi: return a() & imm;
    case Opcode::uori: return a() | imm;
    case Opcode::uxori: return a() ^ imm;
    case Opcode::usread:
        if (inst.imm == static_cast<int32_t>(SpecialReg::warp_id)) return write.warp;
        if (inst.imm == static_cast<int32_t>(SpecialReg::n_warps)) return static_cast<uint32_t>(warps_.size());
        fail("invalid special register for usread");
        break;
    default:
        if (is_usetp(inst.opcode)) {
            return compare_setp(inst.opcode, a(), is_usetp_immediate(inst.opcode) ? imm : b()) ? 1u : 0u;
        }
        break;
    }
    throw_issue_panic(IssueSite{cycle_, write.warp, write.pc, inst, &config_.mma_shapes}, "not uniform arithmetic");
}

// The clock edge's URF write (after registers_.commit()).
void Machine::land_urf_write(const Wires& w) {
    if (!w.urf_landing.has_value()) return;
    warps_.at(w.urf_landing->warp).urf.at(w.urf_landing->instruction.dst) = w.urf_landing->value.value();
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
    static constexpr std::array<Driver, 38> canonical = {
        &Machine::drive_front_end,
        &Machine::drive_fetch,
        &Machine::drive_issue_branch,
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
        &Machine::drive_acc_ld_pipe,
        &Machine::drive_tma_queue,
        &Machine::drive_tma_walker_start,
        &Machine::drive_tma_transactions,
        &Machine::drive_tma_sector,
        &Machine::drive_tma_store_registers,
        &Machine::drive_tma_operand_read,
        &Machine::drive_lsu_commands,
        &Machine::drive_assembly_pool,
        &Machine::drive_ingress,
        &Machine::drive_egress,
        &Machine::drive_read_records,
        &Machine::drive_free_id_tags,
        &Machine::drive_writebacks,
        &Machine::drive_rf_write_policy,
        &Machine::drive_rf_read_policy,
        &Machine::drive_issue_read_banks,
        &Machine::drive_dispatch_pipe,
        &Machine::drive_lsu_read_pipe,
        &Machine::drive_urf_pipe,
    };
    auto order = canonical;
    if (shuffle_.has_value()) std::shuffle(order.begin(), order.end(), *shuffle_);
    for (const auto driver : order) (this->*driver)(w);
}

// The front ends' next state (step_front_end).
void Machine::drive_front_end(const Wires& w) {
    if (w.front_end_next == front_end_.cur()) return;
    front_end_.set_next(w.front_end_next, "drive_front_end");
}

// The instruction memory's read: a granted fetch's address (and tag) is
// sampled at this edge; the read at the output now leaves. The fetch cursor
// moves past the granted warp, only on a grant.
void Machine::drive_fetch(const Wires& w) {
    std::optional<FetchRead> next;
    if (w.fetch.warp.has_value()) {
        next = FetchRead{*w.fetch.warp, w.fetch.pc};
        if (w.fetch.cursor_after != fetch_cursor_.cur()) fetch_cursor_.set_next(w.fetch.cursor_after, "drive_fetch");
    }
    if (next != fetch_read_.cur()) fetch_read_.set_next(next, "drive_fetch");
}

// The issue register's branch: the branch accepted now, else empty.
void Machine::drive_issue_branch(const Wires& w) {
    if (!w.issue.branch.has_value() && !issue_branch_.cur().has_value()) return;
    issue_branch_.set_next(w.issue.branch, "drive_issue_branch");
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
    if (w.completed_waits.empty() && !w.shmem.park.has_value() && !w.dispatch.wait_slot.has_value()) return;
    auto next = mbar_wait_slots_.cur();
    for (const auto slot : w.completed_waits) next.at(slot).reset();
    if (const auto& park = w.shmem.park; park.has_value()) {
        auto& slot = next.at(park->first);
        slot->parked = true;
        slot->deadline_cycle = park->second;
    }
    if (const auto& taken = w.dispatch.wait_slot; taken.has_value()) {
        auto& slot = next.at(taken->first);
        if (slot.has_value()) throw SimError("internal mbar wait slot allocated twice");
        slot = taken->second;
    }
    mbar_wait_slots_.set_next(std::move(next), "drive_wait_slots");
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
    if (w.dispatch.mma.has_value()) next.queue.push_back(*w.dispatch.mma);
    mma_.set_next(std::move(next), "drive_mma");
}

// The two acc transfer engines: step_acc_transfers' result (acc.ld launches,
// acc.st landings), plus this cycle's acc.st RF read (the beat joins the
// acc.st transfer's in-flight list) and dispatch's new transfer (into an
// engine freed this cycle, possibly).
void Machine::drive_acc_transfers(const Wires& w) {
    auto next = w.acc_next;
    if (const auto& beat = w.acc_st_read; beat.has_value()) {
        auto& engine = next[1];
        if (!engine.has_value()) throw SimError("internal acc.st beat read with no acc.st transfer");
        engine->in_flight.push_back(*beat);
        ++engine->next_column;
    }
    if (const auto& admission = w.dispatch.acc; admission.has_value()) {
        auto& engine = next[admission->load ? 0 : 1];
        if (engine.has_value()) throw SimError("internal acc transfer engine given two transfers");
        engine = admission->transfer;
    }
    acc_engines_.set_next(std::move(next), "drive_acc_transfers");
}

// The acc.ld pipeline: step_acc_transfers' advance or hold.
void Machine::drive_acc_ld_pipe(const Wires& w) {
    if (w.acc_ld_next == acc_ld_pipe_.cur()) return;
    acc_ld_pipe_.set_next(w.acc_ld_next, "drive_acc_ld_pipe");
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

// The RF read decision's state, per bank (trash/rfc10.md section 3,
// mirrored; trash/rfc14.md section 5):
//   read_owed_next = simt_accepted_using_bank OR (read_owed AND NOT read_grant)
// (an accepted instruction whose round uses the bank sets it, an actual async
// read grant clears it, set wins); the cursor moves past the reader granted
// last, only on a grant.
void Machine::drive_rf_read_policy(const Wires& w) {
    auto next = rf_read_policy_.cur();
    for (uint32_t bank = 0; bank < config_.rf_banks; ++bank) {
        const auto& decision = w.rf_read.banks[bank];
        const bool accepted = !w.issue.rf_read_banks.empty() && w.issue.rf_read_banks[bank] != 0;
        next.owed[bank] = accepted || (next.owed[bank] != 0 && !decision.granted) ? 1 : 0;
        next.cursor[bank] = decision.cursor_after;
    }
    if (next == rf_read_policy_.cur()) return;
    rf_read_policy_.set_next(std::move(next), "drive_rf_read_policy");
}

// The issue register's bank use: the round of the instruction accepted now
// samples in the next cycle; nothing accepted leaves it empty.
void Machine::drive_issue_read_banks(const Wires& w) {
    auto next = w.issue.rf_read_banks;
    if (next.empty()) next.assign(config_.rf_banks, 0);
    if (next == issue_read_banks_.cur()) return;
    issue_read_banks_.set_next(std::move(next), "drive_issue_read_banks");
}

// The TMA command queue: step_tma's result, plus issue's new command.
void Machine::drive_tma_queue(const Wires& w) {
    auto next = w.tma_next;
    if (w.tma_pushed.has_value()) next.commands.emplace(w.tma_pushed->id, *w.tma_pushed);
    // The hardware sizes (TmaQueue, sim.hpp): issue's admission and the
    // credits keep them; exceeding one is a machine bug.
    if (next.commands.size() > config_.tma_queue_commands) {
        throw SimError("internal: the TMA queue holds more than tma.queue_commands commands");
    }
    if (next.credits_used > config_.tma_sector_credits || next.returns.size() > next.credits_used) {
        throw SimError("internal: TMA placement tickets exceed the sector credits");
    }
    tma_.set_next(std::move(next), "drive_tma_queue");
}

// The walker start pipeline moves one stage: a start whose descriptor came
// back now enters stage 0; the one in the last stage loaded the walker now
// (step_tma) and leaves. Only the frontend command starts, so at most one
// stage is occupied (checked).
void Machine::drive_tma_walker_start(const Wires& w) {
    const auto& cur = tma_walker_start_pipe_.cur();
    if (cur.empty()) {
        if (w.tma_walker_starting.has_value()) throw SimError("internal: a tma walker start with no start stages");
        return;
    }
    const auto occupied = [](const TmaWalkerStartPipeline& pipe) {
        return std::count_if(pipe.begin(), pipe.end(), [](const auto& stage) { return stage.has_value(); });
    };
    if (!w.tma_walker_starting.has_value() && occupied(cur) == 0) return;
    TmaWalkerStartPipeline next(cur.size());
    next.front() = w.tma_walker_starting;
    for (std::size_t stage = 1; stage < cur.size(); ++stage) next[stage] = cur[stage - 1];
    if (occupied(next) > 1) throw SimError("internal: two tma walker starts in the start pipeline");
    tma_walker_start_pipe_.set_next(std::move(next), "drive_tma_walker_start");
}

// The URF's round counter moves to its next round (it empties after the
// last) or takes issue's new instruction; the TMA operand read's sampled
// pairs move one URF read stage on (the one in the last stage arrived now);
// its holding register takes this cycle's arrival.
void Machine::drive_tma_operand_read(const Wires& w) {
    const auto& rounds = urf_read_rounds_.cur();
    std::optional<UrfReadRounds> next_rounds;
    if (rounds.has_value() && urf_read_ports_held()) {
        UrfReadRounds advanced = *rounds;
        ++advanced.round;
        next_rounds = std::move(advanced);
    }
    if (w.issue.urf_rounds.has_value()) {
        if (next_rounds.has_value()) {
            throw SimError("internal: a multi-round uniform reader was accepted while the counter had rounds left");
        }
        next_rounds = w.issue.urf_rounds;
    }
    if (rounds.has_value() || next_rounds.has_value()) {
        urf_read_rounds_.set_next(std::move(next_rounds), "drive_tma_operand_read: URF round counter");
    }
    const auto& pairs = tma_read_pairs_.cur();
    const bool in_flight = std::any_of(pairs.begin(), pairs.end(), [](const auto& stage) { return stage.has_value(); });
    if (in_flight || w.tma_sampled.has_value()) {
        TmaReadPairs next(pairs.size());
        next.front() = w.tma_sampled;
        for (std::size_t stage = 0; stage + 1 < pairs.size(); ++stage) next[stage + 1] = pairs[stage];
        tma_read_pairs_.set_next(std::move(next), "drive_tma_operand_read: URF read stages");
    }
    if (w.tma_holding_changed) tma_holding_.set_next(w.tma_holding_after, "drive_tma_operand_read: holding");
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
        TmaStoreRegister fresh;
        fresh.run = w.tma.walker->run;
        fresh.rounds = w.tma.a_read->rounds;
        fresh.data.assign(config_.gmem_sector_bytes, 0);
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
        w.lsu_rf_read.command.has_value() || !w.released_lsu_commands.empty() || w.dispatch.lsu.has_value();
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
    if (w.dispatch.lsu.has_value()) next.push_back(w.dispatch.lsu->command);
    if (w.lsu_rf_read.command.has_value()) ++command(*w.lsu_rf_read.command).chunks_done;
    if (next.size() > config_.lsu_active_commands) {
        throw SimError("internal: the LSU holds more than lsu.active_commands commands");
    }
    lsu_commands_.set_next(std::move(next), "drive_lsu_commands");
}

// Each assembly register of the pool, in one calculation, in three steps:
//   1. the current owner empties it: a chunk drained to the RF (load) or into
//      the egress FIFO (vector store), and the owner's release;
//   2. dispatch installs a new owner (with a scalar store's data);
//   3. this cycle's fills: the ingress pluck, the store-data beat at the RF
//      output now.
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

        // 2. Dispatch installs the new owner.
        if (w.dispatch.lsu.has_value() && w.dispatch.lsu->command.assembly == index) {
            if (edit().owner.has_value()) throw SimError("internal LSU admission into an owned assembly register");
            next->owner = w.dispatch.lsu->command.id;
            clear();
            if (const auto& data = w.dispatch.lsu->store_data; data.has_value()) {
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
        const auto& reads = lsu_read_pipe_.cur();
        if (!reads.empty() && reads.front().arrive_cycle == cycle_ && reads.front().assembly == index) {
            // The store-data beat at the RF output now (sampled Lr ago).
            auto& reg = edit();
            reg.words = reads.front().values;
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
    const std::size_t vector_sectors = std::size_t{8} * config_.lanes * 4 / config_.gmem_sector_bytes;
    if (tma > config_.tma_sector_credits || vector > config_.lsu_active_commands * vector_sectors ||
        free_ids > config_.lsu_sectors_in_flight) {
        throw SimError("internal: the read ID table holds more reads than its readers can have in flight");
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

// The dispatch pipeline: the RF read stage's samples latch into their
// entries; the entry dispatched now leaves; issue's accepted entry joins.
void Machine::drive_dispatch_pipe(const Wires& w) {
    if (!w.dispatched && w.dispatch_samples.empty() && !w.issue.dispatch.has_value()) return;
    auto next = dispatch_pipe_.cur();
    for (const auto& sample : w.dispatch_samples) {
        auto& entry = next.at(sample.index);
        if (!sample.later_round) entry.operands = sample.operands;
        for (std::size_t i = 0; i < sample.uniform_operands.size(); ++i) {
            entry.uniform_operands.at(sample.uniform_first + i) = sample.uniform_operands[i];
        }
    }
    if (w.dispatched) next.pop_front();
    if (w.issue.dispatch.has_value()) next.push_back(*w.issue.dispatch);
    dispatch_pipe_.set_next(std::move(next), "drive_dispatch_pipe");
}

// The URF write pipeline's next state (step_urf); held while it is empty.
void Machine::drive_urf_pipe(const Wires& w) {
    const auto busy = [](const UrfWritePipeline& pipe) {
        return std::any_of(pipe.begin(), pipe.end(), [](const auto& stage) { return stage.has_value(); });
    };
    if (!busy(urf_pipe_.cur()) && !busy(w.urf_pipe_next)) return;
    urf_pipe_.set_next(w.urf_pipe_next, "drive_urf_pipe");
}

// The LSU's store-data reads in flight: the one at the RF output now leaves
// (it fills its assembly register, drive_assembly_pool); the one sampled
// now joins.
void Machine::drive_lsu_read_pipe(const Wires& w) {
    const auto& cur = lsu_read_pipe_.cur();
    const bool arrived = !cur.empty() && cur.front().arrive_cycle == cycle_;
    if (!arrived && !w.lsu_rf_read.command.has_value()) return;
    auto next = cur;
    if (arrived) next.pop_front();
    if (w.lsu_rf_read.command.has_value()) next.push_back(w.lsu_rf_read);
    if (next.size() > config_.rf_read_latency_cycles) {
        throw SimError("internal: the LSU store-data read pipeline holds more than its Lr stages");
    }
    lsu_read_pipe_.set_next(std::move(next), "drive_lsu_read_pipe");
}

// ===========================================================================
// Helpers
// ===========================================================================

// The word pc names in the head packet: pc bit 3 selects its half.
Instruction Machine::head_instruction(const Warp& warp) const {
    const auto& slot = front_end_.cur().at(warp.id);
    if (slot.pc % 8 != 0) throw SimError("unaligned PC in warp " + std::to_string(warp.id));
    if (slot.pc / 8 >= kernel_.words.size()) {
        throw SimError(
            "warp " + std::to_string(warp.id) + " fell off kernel at pc " +
            std::to_string(slot.pc));
    }
    return decode(slot.fifo.front()[slot.pc / 8 % 2]);
}

std::vector<uint8_t> Machine::source_registers(const Instruction& inst) const {
    if (is_mma_async(inst.opcode)) return {};
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
    case Opcode::setp_ge:
    case Opcode::setp_lt_s32:
    case Opcode::setp_le_s32:
    case Opcode::setp_gt_s32:
    case Opcode::setp_ge_s32: return {inst.src0, inst.src1};
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
    case Opcode::ex2:
    case Opcode::rcp:
    case Opcode::setpi_eq:
    case Opcode::setpi_ne:
    case Opcode::setpi_lt:
    case Opcode::setpi_le:
    case Opcode::setpi_gt:
    case Opcode::setpi_ge:
    case Opcode::setpi_lt_s32:
    case Opcode::setpi_le_s32:
    case Opcode::setpi_gt_s32:
    case Opcode::setpi_ge_s32:
    case Opcode::r2ur: return {inst.src0};
    case Opcode::sts:
    case Opcode::stg: return {inst.src0, inst.src1};
    case Opcode::stg_v8: {
        std::vector<uint8_t> result{inst.src0};
        for (uint16_t j = 0; j < 8; ++j) result.push_back(static_cast<uint8_t>(inst.src1 + j));
        return result;
    }
    case Opcode::acc_st: return {inst.src1};
    case Opcode::acc_st_v8: {
        std::vector<uint8_t> result;
        for (uint16_t j = 0; j < 8; ++j) result.push_back(static_cast<uint8_t>(inst.src1 + j));
        return result;
    }
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
    case Opcode::ur2r:
    case Opcode::lds:
    case Opcode::lds_uni:
    case Opcode::ldg:
    case Opcode::ldg_v8:
    case Opcode::ldg_uni: return inst.dst;
    case Opcode::acc_ld:
    case Opcode::acc_ld_v8: return inst.dst;
    case Opcode::atom_add_uni:
    case Opcode::atom_and_uni:
    case Opcode::atom_xor_uni:
    case Opcode::atom_exch_uni: return inst.dst;
    default: return std::nullopt;
    }
}

// No scoreboard: a hazard is a PANIC, diagnosed by the sanitizer at the
// cycles src/rf.hpp gives (each source's sample; the destination's earliest
// landing), the same numbers qtxc schedules with. Only the destination range
// check is a property of the instruction itself.
void Machine::check_register_hazards(const Warp& warp, const Instruction& inst) const {
    std::vector<std::pair<uint8_t, uint64_t>> samples;
    for (const auto reg : source_registers(inst)) samples.emplace_back(reg, cycle_ + issue_to_sample);
    sanitizer_.check_rf_raw(site(warp, inst), samples);
    if (const auto dst = destination_register(inst); dst.has_value()) {
        const auto width = inst.opcode == Opcode::acc_ld_v8 || inst.opcode == Opcode::ldg_v8 ? 8u : 1u;
        if (static_cast<unsigned>(*dst) + width > kernel_.register_count)
            panic(warp, inst, "RF destination range out of bounds");
        sanitizer_.check_rf_waw_war(site(warp, inst), *dst, width, cycle_ + issue_to_visible_min(inst, config_));
    }
}

// A dispatched instruction's operand, as its RF read latched it at its
// sample (step_rf_reads). It is never read from the RF array here.
std::vector<uint32_t> Machine::capture(const DispatchEntry& op, uint8_t reg) const {
    for (std::size_t i = 0; i < op.reads.size(); ++i) {
        if (op.reads[i] != reg) continue;
        if (op.operands[i].size() != config_.lanes) {
            throw SimError("internal: r" + std::to_string(reg) + " used at dispatch before its RF sample");
        }
        return op.operands[i];
    }
    throw SimError("internal: r" + std::to_string(reg) + " used at dispatch but not read at issue");
}

std::vector<uint32_t> Machine::capture_predicate(const Warp& warp, uint8_t predicate) const {
    const auto& bits = warp.predicates.at(predicate);
    return std::vector<uint32_t>(bits.begin(), bits.end());
}

// A declared-uniform operand still taken from a VECTOR register (rfc09 pin
// 3) -- since the URF slice only the ldg.v8 / stg.v8 bases (the TMA's
// operands are uniform registers since rfc17): hardware reads lane 0,
// whatever the active mask (even when lane 0
// is inactive). Two things are UB and the sanitizer panics on them: a value
// that differs across lanes (named by the operand: "... is not
// warp-uniform"), and any uniform operation under a partial or empty mask
// (backlog D12.1, rfc13 section 2: a program with a partial mask uses the
// per-lane instruction). bra.uni checks the mask in resolve_branch.
uint32_t Machine::uniform_reg(const DispatchEntry& op, uint8_t reg, const std::string& name) const {
    sanitizer_.check_uniform_mask(site(op), op.active_mask, full_mask());
    const auto lanes = capture(op, reg);
    sanitizer_.check_warp_uniform(site(op), lanes, name);
    return lanes[0];
}

// A uniform-register operand (lds.uni / ldg.uni bases, atom.*.uni address
// and operand, acc.ld / acc.st tmem pointers, mbar addresses and values, MMA
// addresses and result pointer, mma.commit's mbar), as its round latched it
// at its sample (step_rf_reads). Uniform by construction: no lane check. The
// uniform operation's partial-mask rule stays (the sanitizer's PANIC).
uint32_t Machine::uniform_operand(const DispatchEntry& op, uint8_t ureg) const {
    sanitizer_.check_uniform_mask(site(op), op.active_mask, full_mask());
    for (std::size_t i = 0; i < op.uniform_reads.size(); ++i) {
        if (op.uniform_reads[i] == ureg) return op.uniform_operands[i];
    }
    throw SimError("internal: ur" + std::to_string(ureg) + " used at dispatch but not read at issue");
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
// An sts with two lanes on one word (UB: the sanitizer's PANIC) writes both,
// in round order, so the higher lane's word lands last; one over a live mbar
// (UB: the sanitizer's C10) simply writes it.
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

// A fixed result on its way to the RF write port it booked (port_cycle),
// landing Wd later. Its visibility was recorded at issue.
Machine::Writeback Machine::register_writeback(
    const Warp& warp,
    uint8_t dst,
    uint64_t port_cycle,
    std::vector<uint32_t> values,
    std::string reason) const {
    Writeback writeback;
    writeback.port_cycle = port_cycle;
    writeback.land_cycle = port_cycle + config_.rf_write_depth_cycles;
    writeback.warp = warp.id;
    writeback.dst = dst;
    writeback.values = std::move(values);
    writeback.reason = std::move(reason);
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

// An issuing warp's site: its FIFO head's pc.
IssueSite Machine::site(const Warp& warp, const Instruction& inst) const {
    return IssueSite{cycle_, warp.id, head_pc(warp.id), inst, &config_.mma_shapes};
}

// A dispatched instruction's site: the dispatch cycle, its issue's pc.
IssueSite Machine::site(const DispatchEntry& op) const {
    return IssueSite{cycle_, op.warp, op.pc, op.instruction, &config_.mma_shapes};
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
    uint32_t base, uint32_t spacing, bool a_operand, const MmaShape& shape) const {
    // Ask the operand geometry itself (the same functions the fetch uses):
    // the hull of every byte the command's shape reads. Under swizzle64 a
    // K = 8 operand reads one 16-byte piece per row, but the hull covers
    // the rows' other pieces too (a conservative owner).
    const auto [first, last] = mma_operand_hull(config_, shape, base, spacing, a_operand);
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
// its write is granted later). None when the mbar is outside shmem (the
// dispatch has panicked then).
// Sanitizer-only: the hardware reads no mbar at TMA start.
std::optional<uint64_t> Machine::pending_tma_completion_mbar(uint32_t mbar, const Wires& w) const {
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

[[noreturn]] void Machine::panic(const DispatchEntry& op, const std::string& reason) const {
    throw_issue_panic(site(op), reason);
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

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
        acc_transfers_.resize(config_.tmem_banks);
        mbar_wait_slots_.resize(config_.mbar_wait_slots);
        read_reservations_.assign(
            config_.rf_arithmetic_read_cycles, std::vector<uint32_t>(config_.rf_banks, 0));
        staging_.resize(config_.lsu_staging_vectors);
        for (auto& assembly : staging_) {
            assembly.words.assign(config_.lanes, 0);
            assembly.valid.assign(config_.lanes, 0);
        }
        free_id_tags_.assign(config_.lsu_sectors_in_flight, 0);
        tma_sector_.data.assign(config_.gmem_sector_bytes, 0);
        tma_store_sector_.data.assign(config_.gmem_sector_bytes, 0);
        warps_.reserve(launch.active_warps);
        for (uint32_t id = 0; id < launch.active_warps; ++id) {
            Warp warp;
            warp.id = id;
            warp.rf.resize(static_cast<std::size_t>(config_.lanes) * kernel_.register_count, 0);
            for (auto& predicate : warp.predicates) predicate.resize(config_.lanes, 0);
            warps_.push_back(std::move(warp));
        }
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

// One core cycle. The memory path of rfc09 3a runs CONSUMER BEFORE PRODUCER
// (reverse pipeline order), so a datum written into a named register in this
// tick cannot be read by the next stage until the next tick:
//
//   load path:  RF write (assembly -> RF)        advance_lsu_rf_writes
//               TMA placement (TMA reg -> shmem) advance_tma / advance_shmem
//               pluck (ingress FIFO -> TMA reg / assembly)  advance_ingress
//               bus (-> ingress FIFO)            advance_dram_clock
//   store path: egress write port (assembly / TMA store reg -> egress FIFO)
//                                                advance_egress_writes
//               TMA store read (shmem -> TMA store reg)     advance_shmem
//               RF read (RF -> assembly): scalar at issue, vector after it
//                                                issue_one / advance_lsu_rf_reads
//
// The two BRAM FIFOs carry a visibility cycle (write cycle + declared
// latency >= 1), which the egress side needs because the bus must run after
// the pluck. Every register also records its last write cycle and every
// consumer asserts it is in the past (tripwire, never a timing input).
// A register drained in cycle t may be refilled in cycle t (its consumer ran
// first in the tick: a normal pipeline handoff, never a read-through).
// Request channel: TMA store write requests, then the TMA walker, then the
// LSU. The LSU's RF write is booked at the start of the tick for this cycle
// (arithmetic booked at earlier issues keeps its port), its RF read after
// issue (this cycle's issue reads keep theirs).
void Machine::tick() {
    if (done()) return;
    sanitizer_.record_cycle_start();
    request_channel_used_ = false;
    ingress_freed_this_cycle_ = 0;
    return_ready_low_this_cycle_ = false;
    advance_lsu_rf_writes();
    commit_writebacks();
    advance_egress_writes();
    advance_tma_write_requests();
    advance_tma();
    advance_lsu_requests();
    advance_shmem();
    advance_mma();
    advance_acc_transfers();
    if (tma_event_command_.has_value() && !engine_mbar_event_.has_value()) {
        const auto& command = tma_commands_.at(*tma_event_command_);
        trace_->tma_command_end(
            cycle_ + 1, command.id, command.logical_bytes, command.useful_bytes);
        tma_commands_.erase(*tma_event_command_);
        tma_event_command_.reset();
        trace_->tma_depths(
            cycle_ + 1, tma_commands_.size(), tma_sector_credits_used_, tma_returns_.size());
    }
    advance_mbar_wait_timeouts();
    advance_ingress();
    advance_dram_clock();
    issue_one();
    advance_lsu_rf_reads();
    // Writeback precedes both old deferred captures and new issue-cycle reads.
    advance_rf_reads();
    ++cycle_;
}

void Machine::run(uint64_t cycle_limit) {
    while (!done() && cycle_ < cycle_limit) tick();
    if (!done()) throw SimError("cycle cap at cycle " + std::to_string(cycle_));
    trace_->finish(cycle_);
}

bool Machine::acc_slot_available(uint8_t bank, bool load) const {
    if (mma_bank_busy(bank)) return false;
    const auto& slots = acc_transfers_.at(bank);
    if (slots[load ? 0 : 1].has_value()) return false;
    return config_.tmem_simt_read_write_duplex ||
        (!slots[0].has_value() && !slots[1].has_value());
}

bool Machine::mma_bank_busy(uint8_t bank) const {
    for (const auto& command : mma_commands_) {
        if (!command.commit && (command.next_step != 0 || command.operands_consumed) &&
            kernel_.accumulators.at(command.accumulator).bank == bank) return true;
    }
    for (const auto& result : mma_results_)
        if (kernel_.accumulators.at(result.accumulator).bank == bank) return true;
    return false;
}

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

void Machine::advance_mma() {
    for (auto it = mma_results_.begin(); it != mma_results_.end();) {
        if (it->ready_cycle > cycle_) { ++it; continue; }
        const auto& acc = kernel_.accumulators.at(it->accumulator);
        for (uint32_t mi = 0; mi < config_.mma_core_m; ++mi)
            for (uint32_t ni = 0; ni < config_.mma_core_n; ++ni)
                tmem_[acc.bank][tmem_word_index(acc, it->m0 + mi, it->n0 + ni)] =
                    it->values[mi * config_.mma_core_n + ni];
        trace_->mma_result(cycle_, it->command, it->m0 * config_.mma_n + it->n0);
        const auto id = it->command;
        auto command = std::find_if(mma_commands_.begin(), mma_commands_.end(),
            [id](const auto& candidate) { return candidate.id == id; });
        if (command == mma_commands_.end() || command->outstanding_results == 0)
            throw SimError("internal MMA result lost its command");
        --command->outstanding_results;
        it = mma_results_.erase(it);
    }
    while (!mma_commands_.empty() && !mma_commands_.front().commit &&
           mma_commands_.front().operands_consumed && mma_commands_.front().outstanding_results == 0) {
        sanitizer_.record_mma_retired(mma_commands_.front().id); // last result retired
        mma_commands_.pop_front();
    }
    // Port service: one bank round per cycle for the in-flight fetch. A new
    // fetch may start only in a cycle with no fetch in service, so the next
    // step begins the cycle after the port frees; one-round layouts sustain
    // one core step per cycle.
    if (mma_fetch_.has_value()) {
        serve_mma_fetch_round();
        return;
    }
    if (mma_commands_.empty()) return;
    auto command_it = std::find_if(mma_commands_.begin(), mma_commands_.end(),
        [](const auto& command) { return !command.operands_consumed; });
    if (command_it == mma_commands_.end()) return;
    auto& command = *command_it;
    if (command.commit) {
        if (command_it != mma_commands_.begin() || engine_mbar_event_.has_value()) return;
        MbarRmwRequest request;
        request.operation = MbarRmwOperation::arrive;
        request.byte_address = command.mbar_address;
        engine_mbar_event_ = std::move(request);
        mma_commands_.pop_front();
        return;
    }
    const auto& acc = kernel_.accumulators.at(command.accumulator);
    if (acc_transfers_[acc.bank][0].has_value() || acc_transfers_[acc.bank][1].has_value()) return;
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
    for (const auto& pending : mma_results_) {
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
    mma_fetch_ = std::move(fetch);
    serve_mma_fetch_round();
}

void Machine::serve_mma_fetch_round() {
    auto& fetch = *mma_fetch_;
    sanitizer_.check_port_b_round(fetch.round_words.at(fetch.next_round));
    for (const auto address : fetch.round_elements.at(fetch.next_round)) {
        if (static_cast<uint64_t>(address) + 2 > shmem_.size())
            throw SimError("MMA operand footprint is outside shmem");
        uint16_t bits = 0;
        std::memcpy(&bits, shmem_.data() + address, 2);
        fetch.captured[address] = bits;
    }
    ++fetch.next_round;
    if (fetch.next_round != fetch.round_words.size()) return;
    complete_mma_fetch();
}

// Fetch completion is core entry: the accumulator input is read, the step's
// result is scheduled one pipeline latency later, and the last step's
// operand-consumed depbar clears here, at the final physical read.
void Machine::complete_mma_fetch() {
    const auto fetch = std::move(*mma_fetch_);
    mma_fetch_.reset();
    const auto id = fetch.command;
    auto command_it = std::find_if(mma_commands_.begin(), mma_commands_.end(),
        [id](const auto& candidate) { return candidate.id == id; });
    if (command_it == mma_commands_.end()) throw SimError("internal MMA fetch lost its command");
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
    mma_results_.push_back(std::move(result));
    ++command.outstanding_results;
    if (fetch.last_step) {
        command.operands_consumed = true;
        sanitizer_.record_mma_operands_consumed(command.id); // final operand read
        auto& counter = warps_.at(command.warp).depbar[command.depbar];
        if (counter == 0) throw SimError("internal MMA depbar underflow");
        --counter;
        trace_->depbar(cycle_, command.warp, command.depbar, counter);
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

void Machine::advance_acc_transfers() {
    for (uint32_t bank = 0; bank < acc_transfers_.size(); ++bank) {
        for (uint32_t direction = 0; direction < 2; ++direction) {
            auto& slot = acc_transfers_[bank][direction];
            if (!slot.has_value()) continue;
            auto& transfer = *slot;
            auto& warp = warps_.at(transfer.warp);
            const auto& acc = kernel_.accumulators.at(transfer.accumulator);
            const auto reg = static_cast<uint8_t>(transfer.rf_base + transfer.next_column);
            if (transfer.load) {
                if (!reserve_write(reg, cycle_ + 1)) continue;
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
                writebacks_.push_back(std::move(writeback));
                sanitizer_.record_reg_ready(warp.id, reg, cycle_ + 1);
            } else {
                const std::vector<RfRead> reads{{reg, 0}};
                if (!can_reserve_reads(reads)) continue;
                reserve_reads(reads);
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
                auto& counter = warp.depbar[transfer.depbar];
                if (counter == 0) throw SimError("internal acc.st depbar underflow");
                --counter;
                trace_->depbar(cycle_, warp.id, transfer.depbar, counter);
                // final tmem write: write-ownership ends
                sanitizer_.record_acc_transfer_final_beat(static_cast<uint8_t>(bank), false, cycle_);
            }
            slot.reset();
        }
    }
}

bool Machine::done() const {
    const bool warps_halted = std::all_of(warps_.begin(), warps_.end(), [](const Warp& warp) {
        return warp.halted;
    });
    const bool wait_slots_empty = std::all_of(
        mbar_wait_slots_.begin(), mbar_wait_slots_.end(),
        [](const auto& slot) { return !slot.has_value(); });
    return warps_halted && wait_slots_empty &&
        !port_a_.has_value() && !tma_port_a_.has_value() &&
        !engine_mbar_event_.has_value() &&
        !mbar_rmw_.has_value() && !mbar_read_.has_value() && gmem_queue_.empty() &&
        !gmem_bus_.has_value() && pending_write_acks_.empty() && lsu_commands_.empty() &&
        read_records_.empty() && ingress_fifo_.empty() && egress_fifo_.empty() &&
        !tma_sector_.request.has_value() && !tma_store_sector_.busy &&
        tma_commands_.empty() &&
        tma_returns_.empty() && !tma_event_command_.has_value() &&
        tma_sector_credits_used_ == 0 && writebacks_.empty() && arithmetic_reads_.empty() &&
        mma_commands_.empty() && mma_results_.empty() && !mma_fetch_.has_value() &&
        std::all_of(acc_transfers_.begin(), acc_transfers_.end(), [](const auto& slots) {
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
    if (engine_mbar_event_.has_value()) {
        throw SimError("engine mbar event slot is occupied; an engine holds one event at a time");
    }
    MbarRmwRequest request;
    request.operation = MbarRmwOperation::complete_tx;
    request.byte_address = byte_address;
    request.value = bytes;
    request.parity = parity;
    engine_mbar_event_ = std::move(request);
}

uint32_t Machine::read_reg(
    uint32_t warp, uint32_t lane, uint32_t reg) const {
    if (warp >= warps_.size() || lane >= config_.lanes || reg >= kernel_.register_count) {
        throw SimError("invalid host RF read");
    }
    return rf(warps_[warp], lane, reg);
}

void Machine::commit_writebacks() {
    for (auto it = writebacks_.begin(); it != writebacks_.end();) {
        if (it->ready_cycle > cycle_) {
            ++it;
            continue;
        }
        auto& warp = warps_.at(it->warp);
        if (it->dst.has_value()) {
            if (it->values.size() != config_.lanes) {
                throw SimError("internal writeback lane count mismatch");
            }
            for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
                rf(warp, lane, *it->dst) = it->values[lane];
            }
        }
        if (it->predicate_dst.has_value()) {
            if (it->values.size() != config_.lanes) {
                throw SimError("internal predicate writeback lane count mismatch");
            }
            for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
                warp.predicates[*it->predicate_dst][lane] = it->values[lane] != 0;
            }
        }
        if (it->clear_depbar.has_value()) {
            auto& counter = warp.depbar[*it->clear_depbar];
            if (counter == 0) {
                throw SimError("depbar underflow on sb" + std::to_string(*it->clear_depbar));
            }
            --counter;
            trace_->depbar(cycle_, warp.id, *it->clear_depbar, counter);
        }
        if (it->release_lsu_command.has_value()) release_lsu_command(*it->release_lsu_command);
        it = writebacks_.erase(it);
    }
    sanitizer_.record_writebacks_committed(cycle_); // acc.ld final RF writes end read-ownership
    write_reservations_.erase(cycle_);
}

void Machine::execute_arithmetic(ArithmeticRead& pending) {
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
        schedule_predicate_write(warp, inst.dst, result_cycle, std::move(result));
    } else {
        schedule_register_write(warp, inst.dst, result_cycle, std::move(result), opcode_name(inst.opcode));
    }
}

// Runs at end of tick, after writeback and issue: an offset-0 read issued
// this cycle captures here, in the same tick, and sees this cycle's writebacks.
void Machine::advance_rf_reads() {
    for (auto it = arithmetic_reads_.begin(); it != arithmetic_reads_.end();) {
        // Perform every capture the plan scheduled for this cycle.
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
            execute_arithmetic(*it);
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

void Machine::advance_tma() {
    present_tma_mbar_event();
    if (tma_port_a_.has_value() || port_a_busy()) return;

    if (!tma_returns_.empty()) {
        auto value = std::move(tma_returns_.front());
        tma_returns_.pop_front();
        // The placement's rounds read the TMA sector register directly; the
        // register stays occupied until the last round (no port-A data latch).
        trace_->tma_depths(
            cycle_, tma_commands_.size(), tma_sector_credits_used_, tma_returns_.size());
        TmaPortRequest request;
        request.kind = TmaPortKind::load_write;
        request.command = value.command;
        request.sector_request = value.sector_request;
        request.from_sector_register = value.from_sector_register;
        request.fragments = std::move(value.fragments);
        request.rounds = tma_port_rounds(request.fragments);
        trace_->tma_port_begin(
            cycle_, request.command, "tma load write", request.rounds.size());
        trace_->tma_data_port(static_cast<uint32_t>(request.rounds.size()));
        tma_port_a_ = std::move(request);
        return;
    }

    advance_tma_frontend();
}

void Machine::advance_shmem() {
    if (tma_port_a_.has_value()) {
        auto& request = *tma_port_a_;
        if (request.rounds.empty() || request.next_round >= request.rounds.size()) {
            throw SimError("internal zero-round tma port-A request");
        }
        serve_tma_port_round(request);
        ++request.next_round;
        trace_->shmem_round();
        if (request.next_round != request.rounds.size()) return;
        auto complete = std::move(request);
        tma_port_a_.reset();
        trace_->shmem_end(cycle_ + 1);
        complete_tma_port_request(std::move(complete));
        return;
    }
    if (!port_a_.has_value()) {
        if (mbar_read_.has_value()) {
            advance_mbar_read();
        } else if (mbar_rmw_.has_value()) {
            advance_mbar_rmw();
        } else if (engine_mbar_event_.has_value()) {
            // Engines line up ahead of new SIMT work: the held event takes the free
            // engine here, before this cycle's issue can see the port as idle.
            ActiveMbarRmw active;
            active.request = std::move(*engine_mbar_event_);
            engine_mbar_event_.reset();
            // Sanitizer: an engine's RMW is admitted here (SIMT ones at issue).
            // Engine events never carry a requesting warp.
            sanitizer_.check_engine_mbar_rmw(cycle_, active.request.byte_address,
                active.request.operation == MbarRmwOperation::complete_tx);
            mbar_rmw_ = std::move(active);
            advance_mbar_rmw();
        }
        return;
    }
    auto& request = *port_a_;
    const auto& round = request.rounds.at(request.next_round);
    for (const auto& access : round) {
        if (request.load) {
            request.load_values[access.lane] = read_shmem_u32(access.byte_address);
        } else {
            std::memcpy(shmem_.data() + access.byte_address, &access.value, sizeof(access.value));
            sanitizer_.record_port_a_write(access.byte_address, sizeof(access.value));
        }
    }
    trace_->shmem_round();
    ++request.next_round;
    if (request.next_round != request.rounds.size()) return;

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
    writebacks_.push_back(std::move(completion));
    trace_->shmem_end(cycle_ + 1);
    port_a_.reset();
}

void Machine::advance_mbar_rmw() {
    auto& active = *mbar_rmw_;
    if (!active.write_stage) {
        active.state = load_mbar(active.request.byte_address);
        active.write_stage = true;
        std::string operation;
        switch (active.request.operation) {
        case MbarRmwOperation::init: operation = "init"; break;
        case MbarRmwOperation::arrive: operation = "arrive"; break;
        case MbarRmwOperation::arrive_expect_tx: operation = "arrive.expect_tx"; break;
        case MbarRmwOperation::complete_tx: operation = "complete_tx"; break;
        }
        trace_->mbar_rmw_begin(cycle_, active.request.byte_address, std::move(operation));
        return; // read cycle
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
        for (const auto& wait_slot : mbar_wait_slots_) {
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
    store_mbar(request.byte_address, state);
    // C5: the RMW write stage is a port-A write like any other; a same-cycle
    // same-word MMA port-B round must see it.
    sanitizer_.record_port_a_write(request.byte_address, 8);
    if (phase_changed) {
        wake_mbar_waiters(request.byte_address, static_cast<uint8_t>(state.phase));
    }
    trace_->mbar_rmw_end(cycle_ + 1, phase_changed);
    mbar_rmw_.reset();
}

void Machine::advance_mbar_read() {
    auto request = std::move(*mbar_read_);
    mbar_read_.reset();
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
    if (!request.wait_slot.has_value() || !warp.mbar_wait_slot.has_value() ||
        *request.wait_slot != *warp.mbar_wait_slot ||
        *request.wait_slot >= mbar_wait_slots_.size() ||
        !mbar_wait_slots_[*request.wait_slot].has_value()) {
        throw SimError("internal mbar try-wait slot mismatch");
    }
    auto& slot = *mbar_wait_slots_[*request.wait_slot];
    if (slot.warp != warp.id || slot.byte_address != request.byte_address ||
        slot.parity != request.parity || slot.predicate_dst != request.predicate_dst) {
        throw SimError("internal mbar try-wait record mismatch");
    }
    trace_->mbar_probe(cycle_, warp.id, request.byte_address, request.parity, complete);
    if (complete) {
        complete_mbar_try_wait(*request.wait_slot, true);
    } else {
        slot.parked = true;
        slot.deadline_cycle = cycle_ > std::numeric_limits<uint64_t>::max() -
                config_.mbar_try_wait_timeout_cycles
            ? std::numeric_limits<uint64_t>::max()
            : cycle_ + config_.mbar_try_wait_timeout_cycles;
        trace_->mbar_sleep(cycle_, warp.id, request.byte_address, request.parity);
    }
}

void Machine::advance_mbar_wait_timeouts() {
    for (uint32_t index = 0; index < mbar_wait_slots_.size(); ++index) {
        const auto& slot = mbar_wait_slots_[index];
        if (!slot.has_value() || !slot->parked || cycle_ < slot->deadline_cycle) continue;
        trace_->mbar_timeout(
            cycle_, slot->warp, slot->byte_address, slot->parity);
        complete_mbar_try_wait(index, false);
    }
}

void Machine::advance_dram_clock() {
    const auto whole_ticks = config_.dram_clock_hz / config_.core_clock_hz;
    const auto remainder = config_.dram_clock_hz % config_.core_clock_hz;
    if (whole_ticks > 1024) throw SimError("DRAM/core clock ratio exceeds simulator limit");
    for (uint64_t i = 0; i < whole_ticks; ++i) {
        ++dram_tick_;
        tick_gmem_controller();
    }
    if (remainder == 0) return;
    if (dram_phase_ >= config_.core_clock_hz - remainder) {
        dram_phase_ -= config_.core_clock_hz - remainder;
        ++dram_tick_;
        tick_gmem_controller();
    } else {
        dram_phase_ += remainder;
    }
}

void Machine::complete_tma_port_request(TmaPortRequest request) {
    auto command_it = tma_commands_.find(request.command);
    if (command_it == tma_commands_.end()) throw SimError("orphaned tma port-A request");
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
            for (const auto& wait_slot : mbar_wait_slots_) {
                if (!wait_slot.has_value()) continue;
                const auto fragment_end = static_cast<uint64_t>(fragment.shmem_address) + fragment.size;
                const auto mbar_end = static_cast<uint64_t>(wait_slot->byte_address) + 8;
                if (fragment.shmem_address < mbar_end && wait_slot->byte_address < fragment_end) {
                    throw SimError("tma load repurposes an mbar with a live waiter");
                }
            }
        }
        if (request.from_sector_register) { // its last round has read it: free
            tma_sector_.request.reset();
            tma_sector_.filled = 0;
            tma_sector_.complete = false;
        }
        --command.pending_load_writes;
        release_tma_credit();
        if (command.walker_done && command.pending_load_writes == 0) {
            mark_tma_data_complete(command);
        }
        return;
    case TmaPortKind::store_read: {
        if (command.descriptor.direction != TmaDirection::store ||
            command.pending_store_captures == 0 || !tma_store_sector_.busy ||
            tma_store_sector_.complete || tma_store_sector_.command != command.id) {
            throw SimError("unexpected tma store capture");
        }
        // The source is captured: the TMA store sector register is complete
        // and goes to the egress FIFO from the next cycle on.
        tma_store_sector_.complete = true;
        --command.pending_store_captures;
        if (command.walker_done && command.pending_store_captures == 0) {
            mark_tma_data_complete(command);
        }
        return;
    }
    }
    throw SimError("unknown tma port-A request");
}

void Machine::advance_tma_frontend() {
    if (!tma_frontend_command_.has_value()) {
        tma_frontend_command_ = next_tma_frontend_command();
    }
    if (!tma_frontend_command_.has_value()) return;

    auto command_it = tma_commands_.find(*tma_frontend_command_);
    if (command_it == tma_commands_.end()) {
        throw SimError("tma frontend selected a missing command");
    }
    auto& command = command_it->second;

    if (command.state == TmaCommandState::descriptor) {
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
        trace_->tma_port_begin(
            cycle_, request.command, "tma descriptor read", request.rounds.size());
        tma_port_a_ = std::move(request);
        return;
    }
    if (command.state == TmaCommandState::phase) {
        TmaPortRequest request;
        request.kind = TmaPortKind::phase;
        request.command = command.id;
        for (uint32_t offset = 0; offset < 8; offset += 4) {
            GmemSectorRequest::TmaFragment fragment;
            fragment.shmem_address = command.descriptor.completion_mbar + offset;
            fragment.size = 4;
            request.fragments.push_back(fragment);
        }
        request.rounds = tma_port_rounds(request.fragments);
        trace_->tma_port_begin(
            cycle_, request.command, "tma mbar phase read", request.rounds.size());
        tma_port_a_ = std::move(request);
        return;
    }
    if (command.state != TmaCommandState::walking) {
        tma_frontend_command_.reset();
        return;
    }

    const bool progressed = tma_walker_step(command);
    if (command.walker_done) {
        if (command.descriptor.direction == TmaDirection::load &&
            command.pending_load_writes == 0) {
            mark_tma_data_complete(command);
        }
        if (command.descriptor.direction == TmaDirection::store &&
            command.pending_store_captures == 0) {
            mark_tma_data_complete(command);
        }
        tma_frontend_command_.reset();
    } else if (!progressed) {
        // Credit or controller pressure. Keep ownership of the walker and retry.
    }
}

void Machine::present_tma_mbar_event() {
    if (engine_mbar_event_.has_value() || tma_event_command_.has_value()) return;
    const auto command_id = next_data_complete_tma_command();
    if (!command_id.has_value()) return;
    auto& command = tma_commands_.at(*command_id);
    if (command.logical_bytes > std::numeric_limits<uint32_t>::max()) {
        throw SimError("tma completion byte count exceeds the mbar encoding");
    }
    MbarRmwRequest request;
    request.operation = MbarRmwOperation::complete_tx;
    request.byte_address = command.descriptor.completion_mbar;
    request.value = static_cast<uint32_t>(command.logical_bytes);
    request.parity = command.mbar_phase;
    engine_mbar_event_ = std::move(request);
    command.state = TmaCommandState::event_presented;
    tma_event_command_ = *command_id;
}

std::optional<uint64_t> Machine::next_tma_frontend_command() const {
    for (const auto& [id, command] : tma_commands_) {
        if (command.state == TmaCommandState::descriptor ||
            command.state == TmaCommandState::phase ||
            (command.state == TmaCommandState::walking && !command.walker_done)) {
            return id;
        }
    }
    return std::nullopt;
}

std::optional<uint64_t> Machine::next_data_complete_tma_command() const {
    for (const auto& [id, command] : tma_commands_) {
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

// One bank round of a TMA port-A transaction. A load placement reads the TMA
// sector register (fall-through tripwire: written in an earlier cycle) or
// writes zeros (OOB fill); a store read writes the TMA store sector register
// in the cycle its round reads shmem.
void Machine::serve_tma_port_round(TmaPortRequest& request) {
    const bool write = request.kind == TmaPortKind::load_write;
    const bool store_read = request.kind == TmaPortKind::store_read;
    if (write && request.from_sector_register) {
        if (!tma_sector_.complete) {
            throw SimError("internal fall-through: TMA sector register read in the cycle it was written");
        }
        sanitizer_.check_tma_sector_read(cycle_);
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
                source = tma_sector_.data.data() +
                    (fragment.gmem_address - tma_sector_.sector_address) + piece.offset;
            }
            std::memcpy(shmem_.data() + address, source, piece.size);
            sanitizer_.record_port_a_write(address, piece.size);
        } else if (store_read) {
            std::memcpy(tma_store_sector_.data.data() +
                    (fragment.gmem_address - tma_store_sector_.sector_address) + piece.offset,
                shmem_.data() + address, piece.size);
        } else {
            std::memcpy(fragment.data.data() + piece.offset, shmem_.data() + address, piece.size);
        }
    }
    if (write) probe(LsuEdge::tma_place_round, request.sector_request, request.next_round, bytes);
    if (store_read) {
        sanitizer_.record_tma_store_sector_write(cycle_);
        probe(LsuEdge::tma_store_round, request.command, request.next_round, bytes);
    }
}

bool Machine::tma_walker_step(TmaCommand& command) {
    if (command.walker_done) return false;
    const bool load = command.descriptor.direction == TmaDirection::load;

    // Store OOB elements have no source or destination transaction.
    while (!load && !command.walker_done && !tma_coordinate_in_bounds(command)) {
        (void)tma_advance_coordinate(command);
    }
    if (command.walker_done) return true;
    if (tma_sector_credits_used_ == config_.tma_sector_credits) return false;
    // A store sector's source read writes the TMA store sector register,
    // which must have drained into the egress FIFO.
    if (!load && tma_store_sector_.busy) return false;

    std::vector<GmemSectorRequest::TmaFragment> fragments;
    const bool synthetic_zero = load && !tma_coordinate_in_bounds(command);
    if (load && !synthetic_zero &&
        (!controller_request_ready() || !request_channel_grants(GmemClient::tma))) {
        return false;
    }
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
            if (!fragments.empty() &&
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
        fragments.push_back(fragment);
        bytes += fragment.size;
        if (in_bounds) command.useful_bytes += fragment.size;
        (void)tma_advance_coordinate(command);
    }
    if (fragments.empty()) throw SimError("tma walker failed to emit an element");

    ++tma_sector_credits_used_;
    trace_->tma_depths(
        cycle_, tma_commands_.size(), tma_sector_credits_used_, tma_returns_.size());
    if (synthetic_zero) {
        TmaReturn returned;
        returned.command = command.id;
        returned.sector_request = next_gmem_request_id_++;
        returned.fragments = std::move(fragments);
        ++command.pending_load_writes;
        tma_returns_.push_back(std::move(returned));
        trace_->tma_depths(
            cycle_, tma_commands_.size(), tma_sector_credits_used_, tma_returns_.size());
        return true;
    }
    trace_->tma_sector(config_.gmem_sector_bytes);
    if (load) {
        GmemSectorRequest request;
        request.id = next_gmem_request_id_++;
        request.command = command.id;
        request.stream = tma_stream;
        request.sector_address = sector_address;
        request.ready_dram_tick = dram_tick_ + config_.gmem_latency_dram_ticks;
        request.load = true;
        request.owner = GmemSectorRequest::Owner::tma_load;
        request.beats = static_cast<uint32_t>(
            (config_.gmem_sector_bytes + config_.gmem_bus_bytes_per_dram_tick - 1) /
            config_.gmem_bus_bytes_per_dram_tick);
        // The placement description rides in the SM's ID table; the data
        // returns through the ingress FIFO and the TMA sector register.
        ReadRecord record;
        record.kind = ReadKind::tma;
        record.stream = tma_stream;
        record.command = command.id;
        record.sector_address = sector_address;
        record.fragments = fragments;
        read_records_.emplace(request.id, std::move(record));
        sanitizer_.record_ordered_request(tma_stream, request.id);
        accept_gmem_request(std::move(request), GmemClient::tma);
        ++command.pending_load_writes;
    } else {
        // No controller request yet: the write's request is issued from the
        // egress FIFO once this sector's data is there (one write contract).
        ++command.pending_store_captures;
        tma_store_sector_.busy = true;
        tma_store_sector_.command = command.id;
        tma_store_sector_.sector_address = sector_address;
        std::fill(tma_store_sector_.data.begin(), tma_store_sector_.data.end(), 0);
        tma_store_sector_.byte_enable = 0;
        for (const auto& fragment : fragments) {
            const auto offset = fragment.gmem_address - sector_address;
            for (uint32_t byte = 0; byte < fragment.size; ++byte)
                tma_store_sector_.byte_enable |= 1ull << (offset + byte);
        }
        tma_store_sector_.complete = false;
        tma_store_sector_.words_out = 0;
        TmaPortRequest port_request;
        port_request.kind = TmaPortKind::store_read;
        port_request.command = command.id;
        port_request.fragments = std::move(fragments);
        port_request.rounds = tma_port_rounds(port_request.fragments);
        trace_->tma_port_begin(
            cycle_, port_request.command, "tma store read", port_request.rounds.size());
        trace_->tma_data_port(static_cast<uint32_t>(port_request.rounds.size()));
        tma_port_a_ = std::move(port_request);
    }
    return true;
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

void Machine::release_tma_credit() {
    if (tma_sector_credits_used_ == 0) throw SimError("tma sector credit underflow");
    --tma_sector_credits_used_;
    trace_->tma_depths(
        cycle_, tma_commands_.size(), tma_sector_credits_used_, tma_returns_.size());
}

void Machine::issue_one() {
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
        if (warp.mbar_wait_slot.has_value()) {
            states[index] = {WarpState::mbar_wait, "mbar phase"};
            continue;
        }
        if (cycle_ < warp.stall_until) {
            states[index] = {WarpState::stall_count, "stall count"};
            continue;
        }
        const auto attempt = try_issue(warp);
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

Machine::IssueAttempt Machine::try_issue(Warp& warp) {
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
    for (uint8_t bar = 0; bar < 6; ++bar) {
        if ((inst.control.wait_mask & (1u << bar)) != 0 && warp.depbar[bar] != 0) {
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
        if (destination.has_value() && !reserve_write(*destination, result_cycle)) {
            return {false, WarpState::structural, "RF write reservation"};
        }
        reserve_reads(reads);
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
    if (is_mbar_try_wait(inst.opcode) && !free_mbar_wait_slot().has_value()) {
        return {false, WarpState::mbar_wait_slot, "mbar wait slots full"};
    }
    if (is_tma(inst.opcode) && tma_commands_.size() == config_.tma_queue_commands) {
        return {false, WarpState::structural, "tma command queue full"};
    }
    if ((inst.opcode == Opcode::mma_async || inst.opcode == Opcode::mma_async_init ||
         inst.opcode == Opcode::mma_commit) && mma_commands_.size() >= config_.mma_queue_commands) {
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
        if (!acc_slot_available(kernel_.accumulators[acc_id].bank, acc_load)) {
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
    // Port-A arbiter, SIMT side. A started access (SIMT rounds, a TMA port
    // request, the mbar engine) is never preempted. For the next free grant
    // the order is: a held engine mbar event (claims the engine in
    // advance_shmem), then the TMA (advance_tma, at the start of the next
    // cycle), then SIMT. So SIMT may not take a free port while the TMA has
    // a port-A request waiting: that is TMA priority, counted per warp-cycle.
    if (inst.opcode == Opcode::lds || inst.opcode == Opcode::sts ||
        is_mbar_rmw(inst.opcode) ||
        is_mbar_test_wait(inst.opcode) || is_mbar_try_wait(inst.opcode)) {
        if (port_a_busy()) return {false, WarpState::structural, "port A busy"};
        if (tma_wants_port_a()) {
            trace_->shmem_tma_priority_wait();
            return {false, WarpState::structural, "port A: TMA priority"};
        }
    }

    // LSU admission: a command slot AND an assembly register, or the warp is
    // ineligible this cycle. Addresses are checked first: a malformed access
    // panics as soon as the warp tries it.
    std::optional<LsuCommand> lsu_plan;
    std::optional<uint32_t> lsu_assembly;
    if (is_lsu(inst.opcode)) {
        lsu_plan = plan_lsu_command(warp, inst);
        if (lsu_commands_.size() >= config_.lsu_active_commands) {
            ++trace_->lsu_counters().lsu_no_slot_cycles;
            return {false, WarpState::structural, "LSU command slot"};
        }
        lsu_assembly = free_assembly_register();
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
        ++warp.depbar[inst.control.set_depbar];
        trace_->depbar(cycle_, warp.id, inst.control.set_depbar, warp.depbar[inst.control.set_depbar]);
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
        mma_commands_.push_back(std::move(command));
        trace_->mma_command(cycle_, mma_commands_.back().id, warp.id,
            inst.opcode == Opcode::mma_async_init);
        break;
    }
    case Opcode::mma_commit: {
        MmaCommand command;
        command.id = next_mma_command_id_++;
        command.commit = true;
        command.warp = warp.id;
        command.mbar_address = mbar_address();
        mma_commands_.push_back(std::move(command));
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
        ++warp.depbar[inst.control.set_depbar];
        trace_->depbar(cycle_, warp.id, inst.control.set_depbar, warp.depbar[inst.control.set_depbar]);
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
        acc_transfers_[acc.bank][load ? 0 : 1] = std::move(transfer);
        break;
    }
    case Opcode::lds: {
        if (inst.control.set_depbar >= 6) panic(warp, inst, "lds has no depbar");
        auto rounds = build_shmem_rounds(warp, inst, true);
        const auto complete_cycle = cycle_ + rounds.size() + 1;
        if (!reserve_write(inst.dst, complete_cycle)) {
            return {false, WarpState::structural, "RF write reservation"};
        }
        // admitted: sanitizer check
        sanitizer_.check_simt_shmem_access(site(warp, inst), access_addresses(rounds), true);
        sanitizer_.record_reg_ready(warp.id, inst.dst, complete_cycle);
        ++warp.depbar[inst.control.set_depbar];
        trace_->depbar(cycle_, warp.id, inst.control.set_depbar, warp.depbar[inst.control.set_depbar]);
        ShmemRequest request;
        request.load = true;
        request.warp = warp.id;
        request.dst = inst.dst;
        request.depbar = inst.control.set_depbar;
        request.rounds = std::move(rounds);
        request.load_values.resize(config_.lanes, 0);
        trace_->shmem_begin(
            cycle_, warp.id, true, static_cast<uint32_t>(request.rounds.size()));
        port_a_ = std::move(request);
        break;
    }
    case Opcode::sts: {
        if (inst.control.set_depbar >= 6) panic(warp, inst, "sts has no depbar");
        auto rounds = build_shmem_rounds(warp, inst, false);
        sanitizer_.check_simt_shmem_access(site(warp, inst), access_addresses(rounds), false);
        ++warp.depbar[inst.control.set_depbar];
        trace_->depbar(cycle_, warp.id, inst.control.set_depbar, warp.depbar[inst.control.set_depbar]);
        ShmemRequest request;
        request.load = false;
        request.warp = warp.id;
        request.depbar = inst.control.set_depbar;
        request.rounds = std::move(rounds);
        trace_->shmem_begin(
            cycle_, warp.id, false, static_cast<uint32_t>(request.rounds.size()));
        port_a_ = std::move(request);
        break;
    }
    case Opcode::ldg:
    case Opcode::stg:
    case Opcode::ldg_v8:
    case Opcode::stg_v8:
    case Opcode::ldg_uni:
        if (inst.control.set_depbar >= 6) panic(warp, inst, "global memory operation has no depbar");
        admit_lsu_command(warp, inst, std::move(*lsu_plan), *lsu_assembly);
        break;
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
            pending_tma_descriptor_bytes(command.descriptor_address));
        tma_commands_.emplace(command_id, std::move(command));
        trace_->tma_command_begin(
            cycle_, command_id, warp.id, static_cast<uint32_t>(inst.imm), tma_commands_.size());
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
        // The engine was free (port A busy is a structural stall above), so the
        // accepted request claims it directly; its read cycle is next advance.
        ActiveMbarRmw active;
        active.request = std::move(request);
        mbar_rmw_ = std::move(active);
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
            const auto slot_index = free_mbar_wait_slot();
            if (!slot_index.has_value()) {
                throw SimError("internal mbar try-wait slot disappeared during issue");
            }
            MbarWaitSlot slot;
            slot.warp = warp.id;
            slot.byte_address = request.byte_address;
            slot.parity = request.parity;
            slot.predicate_dst = request.predicate_dst;
            mbar_wait_slots_[*slot_index] = slot;
            warp.mbar_wait_slot = *slot_index;
            sanitizer_.record_pred_ready(warp.id, inst.dst, std::numeric_limits<uint64_t>::max());
            request.wait_slot = *slot_index;
        }
        mbar_read_ = std::move(request);
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

    reserve_reads(reads);
    for (const auto& read : reads) trace_->rf_read(cycle_, warp.id, read.reg, read.reg % config_.rf_banks);
    warp.stall_until = cycle_ + inst.control.stall;
    const auto old_pc = warp.pc;
    warp.pc = next_pc;
    trace_->issue(cycle_, warp.id, old_pc, inst);
    return {true, WarpState::issued, {}};
}

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

void Machine::reserve_reads(const std::vector<RfRead>& reads) {
    for (const auto& read : reads) {
        ++read_reservations_[(cycle_ + read.offset) % read_reservations_.size()][read.reg % config_.rf_banks];
    }
}

bool Machine::reserve_write(uint8_t reg, uint64_t at_cycle) {
    auto& slots = write_reservations_[at_cycle];
    if (slots.empty()) slots.resize(config_.rf_banks, 0);
    const auto bank = reg % config_.rf_banks;
    if (slots[bank] == config_.rf_write_ports_per_bank) return false;
    ++slots[bank];
    return true;
}

bool Machine::port_a_busy() const {
    return port_a_.has_value() || tma_port_a_.has_value() ||
        mbar_read_.has_value() || mbar_rmw_.has_value() ||
        engine_mbar_event_.has_value();
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

void Machine::store_mbar(uint32_t byte_address, const MbarState& state) {
    const auto word = encode_mbar_state(state);
    const auto low = static_cast<uint32_t>(word);
    const auto high = static_cast<uint32_t>(word >> 32);
    std::memcpy(shmem_.data() + byte_address, &low, sizeof(low));
    std::memcpy(shmem_.data() + byte_address + 4, &high, sizeof(high));
}

void Machine::wake_mbar_waiters(uint32_t byte_address, uint8_t new_phase) {
    for (uint32_t index = 0; index < mbar_wait_slots_.size(); ++index) {
        const auto& slot = mbar_wait_slots_[index];
        if (!slot.has_value() || slot->byte_address != byte_address ||
            slot->parity == new_phase) {
            continue;
        }
        trace_->mbar_wake(cycle_, slot->warp, byte_address, new_phase);
        complete_mbar_try_wait(index, true);
    }
}

std::optional<uint32_t> Machine::free_mbar_wait_slot() const {
    for (uint32_t index = 0; index < mbar_wait_slots_.size(); ++index) {
        if (!mbar_wait_slots_[index].has_value()) return index;
    }
    return std::nullopt;
}

void Machine::complete_mbar_try_wait(uint32_t index, bool result) {
    if (index >= mbar_wait_slots_.size() || !mbar_wait_slots_[index].has_value()) {
        throw SimError("internal completion of an empty mbar try-wait slot");
    }
    const auto slot = *mbar_wait_slots_[index];
    auto& warp = warps_.at(slot.warp);
    if (!warp.mbar_wait_slot.has_value() || *warp.mbar_wait_slot != index) {
        throw SimError("internal warp mbar try-wait slot mismatch");
    }
    for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
        warp.predicates[slot.predicate_dst][lane] = result;
    }
    sanitizer_.record_pred_ready(warp.id, slot.predicate_dst, cycle_);
    warp.mbar_wait_slot.reset();
    mbar_wait_slots_[index].reset();
}

uint32_t& Machine::rf(Warp& warp, uint32_t lane, uint32_t reg) {
    return warp.rf.at(static_cast<std::size_t>(lane) * kernel_.register_count + reg);
}

uint32_t Machine::rf(const Warp& warp, uint32_t lane, uint32_t reg) const {
    return warp.rf.at(static_cast<std::size_t>(lane) * kernel_.register_count + reg);
}

std::vector<std::vector<Machine::LaneAccess>> Machine::build_shmem_rounds(
    const Warp& warp, const Instruction& inst, bool load) const {
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
            for (const auto& wait_slot : mbar_wait_slots_) {
                if (wait_slot.has_value() &&
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

std::size_t Machine::gmem_outstanding_sectors() const {
    return gmem_queue_.size() + (gmem_bus_.has_value() ? 1 : 0);
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

void Machine::schedule_register_write(
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
    writebacks_.push_back(std::move(writeback));
}

void Machine::schedule_predicate_write(
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
    writebacks_.push_back(std::move(writeback));
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
    uint32_t address) const {
    // The descriptor the engine will read later. The engine's descriptor read
    // is a port-A request, so a SIMT sts still draining through port A is
    // ordered before it (one path, issue order): overlay its unserved rounds.
    // Writes accepted after this point would conflict with the descriptor's
    // read-ownership and panic.
    std::array<uint8_t, tma_descriptor_bytes> bytes{};
    std::memcpy(bytes.data(), shmem_.data() + address, bytes.size());
    if (port_a_.has_value() && !port_a_->load) {
        for (std::size_t round = port_a_->next_round; round < port_a_->rounds.size(); ++round) {
            for (const auto& access : port_a_->rounds[round]) {
                if (access.byte_address < address ||
                    access.byte_address + 4 > address + tma_descriptor_bytes) continue;
                std::memcpy(bytes.data() + (access.byte_address - address),
                    &access.value, sizeof(access.value));
            }
        }
    }
    return bytes;
}

// Hardware: the command's data movement is done. Sanitizer: its box
// ownership ends here.
void Machine::mark_tma_data_complete(TmaCommand& command) {
    command.state = TmaCommandState::data_complete;
    sanitizer_.record_tma_data_complete(command.id);
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

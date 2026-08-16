#include "sim.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <new>
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

namespace {

bool is_setp(Opcode opcode) {
    return (opcode >= Opcode::setp_eq && opcode <= Opcode::setp_ge) ||
        (opcode >= Opcode::setpi_eq && opcode <= Opcode::setpi_ge);
}

bool is_setp_immediate(Opcode opcode) {
    return opcode >= Opcode::setpi_eq && opcode <= Opcode::setpi_ge;
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
    if (launch.active_warps == 0) throw SimError("active warp count is zero");
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
        warps_.reserve(launch.active_warps);
        for (uint32_t id = 0; id < launch.active_warps; ++id) {
            Warp warp;
            warp.id = id;
            warp.rf.resize(static_cast<std::size_t>(config_.lanes) * kernel_.register_count, 0);
            warp.ready_cycle.resize(kernel_.register_count, 0);
            for (auto& predicate : warp.predicates) predicate.resize(config_.lanes, 0);
            warps_.push_back(std::move(warp));
        }
    } catch (const std::bad_alloc&) {
        throw SimError("machine state does not fit in host memory");
    } catch (const std::length_error&) {
        throw SimError("machine state exceeds host container limits");
    }
    trace_.emplace(launch.active_warps, config_.core_clock_hz, trace_options);
    apply_image(shmem_, launch.shmem_image, "shmem");
    apply_image(gmem_, launch.gmem_image, "gmem");
}

void Machine::tick() {
    if (done()) return;
    commit_writebacks();
    advance_shmem();
    advance_dram_clock();
    issue_one();
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
    return warps_halted && !port_a_.has_value() && gmem_queue_.empty() &&
        !gmem_bus_.has_value() && gmem_commands_.empty() && writebacks_.empty();
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
        if (it->clear_outstanding_ldg) {
            warp.outstanding_ldg = false;
        }
        it = writebacks_.erase(it);
    }
    write_reservations_.erase(cycle_);
}

void Machine::advance_shmem() {
    if (!port_a_.has_value()) return;
    auto& request = *port_a_;
    const auto& round = request.rounds.at(request.next_round);
    for (const auto& access : round) {
        if (request.load) {
            request.load_values[access.lane] = read_shmem_u32(access.byte_address);
        } else {
            std::memcpy(shmem_.data() + access.byte_address, &access.value, sizeof(access.value));
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

void Machine::tick_gmem_controller() {
    if (!gmem_bus_.has_value() && !gmem_queue_.empty() &&
        gmem_queue_.front().ready_dram_tick <= dram_tick_) {
        gmem_bus_ = std::move(gmem_queue_.front());
        gmem_queue_.pop_front();
        trace_->gmem_bus_begin(
            cycle_, gmem_bus_->id, gmem_bus_->sector_address, gmem_bus_->load);
        trace_->gmem_queue_depth(cycle_, gmem_outstanding_sectors());
    }
    if (!gmem_bus_.has_value()) return;

    auto& request = *gmem_bus_;
    if (request.beats_remaining == 0) throw SimError("gmem request has no bus beats");
    --request.beats_remaining;
    trace_->gmem_bus_beat(cycle_, dram_tick_, request.id);
    if (request.beats_remaining != 0) return;

    auto complete = std::move(request);
    gmem_bus_.reset();
    trace_->gmem_bus_end(cycle_ + 1, complete.id, config_.gmem_sector_bytes);
    trace_->gmem_queue_depth(cycle_, gmem_outstanding_sectors());
    complete_gmem_sector(std::move(complete));
}

void Machine::complete_gmem_sector(GmemSectorRequest request) {
    auto command_it = gmem_commands_.find(request.command);
    if (command_it == gmem_commands_.end()) throw SimError("orphaned gmem sector request");
    auto& command = command_it->second;
    for (const auto& access : request.accesses) {
        if (request.load) {
            uint32_t value = 0;
            std::memcpy(
                &value,
                gmem_.data() + static_cast<std::size_t>(access.byte_address),
                sizeof(value));
            command.load_values[access.lane] = value;
        } else {
            std::memcpy(
                gmem_.data() + static_cast<std::size_t>(access.byte_address),
                &access.value,
                sizeof(access.value));
        }
    }
    if (command.remaining_sectors == 0) throw SimError("gmem command sector underflow");
    --command.remaining_sectors;
    if (command.remaining_sectors != 0) return;

    Writeback completion;
    completion.warp = command.warp;
    completion.clear_depbar = command.depbar;
    completion.reason = command.load ? "ldg" : "stg";
    completion.ready_cycle = cycle_ + 1;
    if (command.load) {
        while (!reserve_write(command.dst, completion.ready_cycle)) ++completion.ready_cycle;
        completion.dst = command.dst;
        completion.values = std::move(command.load_values);
        completion.clear_outstanding_ldg = true;
        warps_.at(command.warp).ready_cycle[command.dst] = completion.ready_cycle;
    }
    writebacks_.push_back(std::move(completion));
    trace_->gmem_command_end(cycle_ + 1, request.command);
    gmem_commands_.erase(command_it);
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
    if (is_setp(inst.opcode)) {
        if (inst.dst >= 7) panic(warp, inst, "setp destination is not p0..p6");
        if (warp.predicate_ready_cycle[inst.dst] > cycle_) {
            panic(warp, inst, "predicate WAW hazard: missing stall for p" + std::to_string(inst.dst));
        }
    }
    if (inst.opcode == Opcode::bra_uni && inst.control.predicate != 7 &&
        warp.predicate_ready_cycle[inst.control.predicate] > cycle_) {
        panic(
            warp,
            inst,
            "predicate RAW hazard: missing stall for p" + std::to_string(inst.control.predicate));
    }
    check_register_hazards(warp, inst);
    check_rf_read_ports(inst);
    if ((inst.opcode == Opcode::lds || inst.opcode == Opcode::sts) && port_a_.has_value()) {
        return {false, WarpState::structural, "port A busy"};
    }

    std::vector<std::vector<LaneAccess>> gmem_sectors;
    if (inst.opcode == Opcode::ldg || inst.opcode == Opcode::stg) {
        if (inst.opcode == Opcode::ldg && warp.outstanding_ldg) {
            return {false, WarpState::structural, "one outstanding ldg"};
        }
        gmem_sectors = build_gmem_sectors(warp, inst, inst.opcode == Opcode::ldg);
        if (gmem_sectors.size() > config_.gmem_queue_sectors) {
            panic(warp, inst, "one gmem command exceeds controller queue capacity");
        }
        if (gmem_outstanding_sectors() + gmem_sectors.size() > config_.gmem_queue_sectors) {
            return {false, WarpState::structural, "gmem queue full"};
        }
    }

    std::vector<uint32_t> result(config_.lanes, 0);
    const auto latency = fixed_latency(inst, config_.timing);
    const auto result_cycle = cycle_ + latency;
    uint32_t next_pc = warp.pc + 8;

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
            result[lane] = rf(warp, lane, inst.src0) + rf(warp, lane, inst.src1);
        }
        break;
    case Opcode::addi:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = rf(warp, lane, inst.src0) + static_cast<uint32_t>(inst.imm);
        }
        break;
    case Opcode::shli:
        if (inst.imm < 0 || inst.imm > 31) panic(warp, inst, "shift outside 0..31");
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = rf(warp, lane, inst.src0) << inst.imm;
        }
        break;
    case Opcode::fadd:
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = float_bits(
                bits_float(rf(warp, lane, inst.src0)) + bits_float(rf(warp, lane, inst.src1)));
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
                : rf(warp, lane, inst.src1);
            result[lane] = compare_setp(inst.opcode, rf(warp, lane, inst.src0), rhs);
        }
        break;
    case Opcode::lds: {
        if (inst.control.set_depbar >= 6) panic(warp, inst, "lds has no depbar");
        auto rounds = build_shmem_rounds(warp, inst, true);
        const auto complete_cycle = cycle_ + rounds.size() + 1;
        if (!reserve_write(inst.dst, complete_cycle)) {
            return {false, WarpState::structural, "RF write reservation"};
        }
        warp.ready_cycle[inst.dst] = complete_cycle;
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
    case Opcode::stg: {
        if (inst.control.set_depbar >= 6) panic(warp, inst, "global memory operation has no depbar");
        const bool load = inst.opcode == Opcode::ldg;
        const auto command_id = next_gmem_command_id_++;
        GmemCommand command;
        command.load = load;
        command.warp = warp.id;
        command.dst = inst.dst;
        command.depbar = inst.control.set_depbar;
        command.remaining_sectors = static_cast<uint32_t>(gmem_sectors.size());
        if (load) {
            command.load_values.resize(config_.lanes, 0);
            warp.outstanding_ldg = true;
            warp.ready_cycle[inst.dst] = std::numeric_limits<uint64_t>::max();
        }
        gmem_commands_.emplace(command_id, std::move(command));
        trace_->gmem_command_begin(cycle_, command_id, warp.id, load);
        ++warp.depbar[inst.control.set_depbar];
        trace_->depbar(cycle_, warp.id, inst.control.set_depbar, warp.depbar[inst.control.set_depbar]);
        const auto beats = static_cast<uint32_t>(
            (config_.gmem_sector_bytes + config_.gmem_bus_bytes_per_dram_tick - 1) /
            config_.gmem_bus_bytes_per_dram_tick);
        for (auto& accesses : gmem_sectors) {
            GmemSectorRequest request;
            request.id = next_gmem_request_id_++;
            request.command = command_id;
            request.sector_address =
                (accesses.front().byte_address / config_.gmem_sector_bytes) * config_.gmem_sector_bytes;
            request.ready_dram_tick = dram_tick_ + config_.gmem_latency_dram_ticks;
            request.load = load;
            request.accesses = std::move(accesses);
            request.beats_remaining = beats;
            trace_->gmem_sector_enqueue();
            gmem_queue_.push_back(std::move(request));
        }
        trace_->gmem_queue_depth(cycle_, gmem_outstanding_sectors());
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

    if (is_setp(inst.opcode)) {
        if (latency == 0) panic(warp, inst, "zero-latency predicate writer is unsupported");
        schedule_predicate_write(warp, inst.dst, result_cycle, std::move(result));
    } else if (latency != 0) {
        if (!destination.has_value()) panic(warp, inst, "fixed-latency opcode has no destination");
        if (!reserve_write(*destination, result_cycle)) {
            return {false, WarpState::structural, "RF write reservation"};
        }
        schedule_register_write(warp, *destination, result_cycle, std::move(result), opcode_name(inst.opcode));
    } else if (inst.opcode != Opcode::lds && inst.opcode != Opcode::sts &&
               inst.opcode != Opcode::ldg && inst.opcode != Opcode::stg &&
               inst.opcode != Opcode::bra && inst.opcode != Opcode::bra_uni &&
               inst.opcode != Opcode::halt) {
        panic(warp, inst, "zero-latency RF writer is unsupported");
    }

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
    case Opcode::setp_eq:
    case Opcode::setp_ne:
    case Opcode::setp_lt:
    case Opcode::setp_le:
    case Opcode::setp_gt:
    case Opcode::setp_ge: return {inst.src0, inst.src1};
    case Opcode::addi:
    case Opcode::shli:
    case Opcode::lds:
    case Opcode::ldg:
    case Opcode::setpi_eq:
    case Opcode::setpi_ne:
    case Opcode::setpi_lt:
    case Opcode::setpi_le:
    case Opcode::setpi_gt:
    case Opcode::setpi_ge: return {inst.src0};
    case Opcode::sts:
    case Opcode::stg: return {inst.src0, inst.src1};
    default: return {};
    }
}

std::optional<uint8_t> Machine::destination_register(const Instruction& inst) const {
    switch (inst.opcode) {
    case Opcode::sread:
    case Opcode::mov:
    case Opcode::add:
    case Opcode::addi:
    case Opcode::shli:
    case Opcode::fadd:
    case Opcode::lds:
    case Opcode::ldg: return inst.dst;
    default: return std::nullopt;
    }
}

void Machine::check_register_hazards(const Warp& warp, const Instruction& inst) const {
    for (const auto reg : source_registers(inst)) {
        if (warp.ready_cycle.at(reg) > cycle_) {
            panic(warp, inst, "RF RAW hazard: missing stall or depbar wait for r" + std::to_string(reg));
        }
    }
    if (const auto dst = destination_register(inst); dst.has_value() && warp.ready_cycle[*dst] > cycle_) {
        panic(warp, inst, "RF WAW hazard: missing stall or depbar wait for r" + std::to_string(*dst));
    }
}

void Machine::check_rf_read_ports(const Instruction& inst) const {
    std::vector<std::vector<uint8_t>> bank_reads(config_.rf_banks);
    for (const auto reg : source_registers(inst)) {
        const auto bank = reg % config_.rf_banks;
        auto& reads = bank_reads[bank];
        if (std::find(reads.begin(), reads.end(), reg) != reads.end()) continue;
        if (reads.size() == config_.rf_read_ports_per_bank) {
            throw SimError("RF bank read conflict in '" + format_instruction(inst) + "'");
        }
        reads.push_back(reg);
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
        const auto bank = (byte_address / 4) % config_.shmem_banks;
        auto& bank_groups = groups[bank];
        auto found = std::find_if(bank_groups.begin(), bank_groups.end(), [&](const Group& group) {
            return group.address == byte_address;
        });
        if (found != bank_groups.end()) {
            if (!load) panic(warp, inst, "multiple lanes write the same shmem word");
            found->accesses.push_back({lane, byte_address, 0});
            continue;
        }
        Group group;
        group.address = byte_address;
        group.accesses.push_back({lane, byte_address, load ? 0 : rf(warp, lane, inst.src1)});
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

std::vector<std::vector<Machine::LaneAccess>> Machine::build_gmem_sectors(
    const Warp& warp, const Instruction& inst, bool load) const {
    std::map<uint64_t, std::vector<LaneAccess>> sectors;
    std::map<uint64_t, uint32_t> store_words;
    for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
        const auto base = static_cast<int64_t>(rf(warp, lane, inst.src0));
        const auto address = base + inst.imm;
        if (address < 0) panic(warp, inst, "gmem address arithmetic underflow");
        const auto byte_address = static_cast<uint64_t>(address);
        if (byte_address % 4 != 0) panic(warp, inst, "unaligned 32-bit gmem access");
        if (byte_address + 4 < byte_address || byte_address + 4 > gmem_.size()) {
            panic(warp, inst, "gmem access out of bounds");
        }
        if (!load && !store_words.emplace(byte_address, lane).second) {
            panic(warp, inst, "multiple lanes write the same gmem word");
        }
        const auto sector = (byte_address / config_.gmem_sector_bytes) * config_.gmem_sector_bytes;
        sectors[sector].push_back({lane, byte_address, load ? 0 : rf(warp, lane, inst.src1)});
    }
    std::vector<std::vector<LaneAccess>> result;
    result.reserve(sectors.size());
    for (auto& entry : sectors) {
        result.push_back(std::move(entry.second));
    }
    return result;
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
    warp.ready_cycle[dst] = ready_cycle;
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
    warp.predicate_ready_cycle[dst] = ready_cycle;
    Writeback writeback;
    writeback.ready_cycle = ready_cycle;
    writeback.warp = warp.id;
    writeback.predicate_dst = dst;
    writeback.values = std::move(values);
    writeback.reason = "setp";
    writebacks_.push_back(std::move(writeback));
}

[[noreturn]] void Machine::panic(
    const Warp& warp, const Instruction& inst, const std::string& reason) const {
    throw SimError(
        "PANIC cycle=" + std::to_string(cycle_) + " warp=" + std::to_string(warp.id) +
        " pc=" + std::to_string(warp.pc) + " reason=" + reason + " inst='" +
        format_instruction(inst) + "'");
}

} // namespace nanosm

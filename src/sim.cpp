#include "sim.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <sstream>
#include <utility>

namespace nanosm {

std::uint32_t float_bits(float value) {
    return std::bit_cast<std::uint32_t>(value);
}

float bits_float(std::uint32_t value) {
    return std::bit_cast<float>(value);
}

Machine::Machine(Config config, Kernel kernel, Launch launch)
    : config_(config), kernel_(std::move(kernel)), shmem_(config.shmem_bytes, 0) {
    if (config_.lanes == 0 || config_.registers == 0 || config_.registers > 64 ||
        config_.shmem_banks == 0 || kernel_.words.empty()) {
        throw SimError("invalid zero-sized or unencodable machine resource");
    }
    if (launch.active_warps == 0 || launch.active_warps > config_.max_warps) {
        throw SimError("invalid active warp count");
    }
    if (launch.shmem_image.size() > shmem_.size()) {
        throw SimError("launch shmem image does not fit");
    }
    std::copy(launch.shmem_image.begin(), launch.shmem_image.end(), shmem_.begin());
    warps_.reserve(launch.active_warps);
    for (std::uint32_t id = 0; id < launch.active_warps; ++id) {
        Warp warp;
        warp.id = id;
        warp.rf.resize(static_cast<std::size_t>(config_.lanes) * config_.registers, 0);
        warp.ready_cycle.resize(config_.registers, 0);
        warps_.push_back(std::move(warp));
    }
}

void Machine::tick() {
    if (done()) return;
    commit_writebacks();
    advance_shmem();
    issue_one();
    ++cycle_;
}

void Machine::run(std::uint64_t cycle_limit) {
    while (!done() && cycle_ < cycle_limit) tick();
    if (!done()) throw SimError("cycle cap at cycle " + std::to_string(cycle_));
}

bool Machine::done() const {
    const bool warps_halted = std::all_of(warps_.begin(), warps_.end(), [](const Warp& warp) {
        return warp.halted;
    });
    return warps_halted && !port_a_.has_value() && writebacks_.empty();
}

std::uint32_t Machine::read_shmem_u32(std::uint32_t byte_address) const {
    if (byte_address % 4 != 0 || static_cast<std::uint64_t>(byte_address) + 4 > shmem_.size()) {
        throw SimError("invalid host shmem u32 read");
    }
    std::uint32_t value = 0;
    std::memcpy(&value, shmem_.data() + byte_address, sizeof(value));
    return value;
}

std::uint32_t Machine::read_reg(
    std::uint32_t warp, std::uint32_t lane, std::uint32_t reg) const {
    if (warp >= warps_.size() || lane >= config_.lanes || reg >= config_.registers) {
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
            for (std::uint32_t lane = 0; lane < config_.lanes; ++lane) {
                rf(warp, lane, *it->dst) = it->values[lane];
            }
            trace(
                "rf.write w" + std::to_string(warp.id) + " r" +
                std::to_string(*it->dst) + " (" + it->reason + ")");
        }
        if (it->clear_depbar.has_value()) {
            auto& counter = warp.depbar[*it->clear_depbar];
            if (counter == 0) {
                throw SimError("depbar underflow on sb" + std::to_string(*it->clear_depbar));
            }
            --counter;
            trace(
                "depbar.dec w" + std::to_string(warp.id) + " sb" +
                std::to_string(*it->clear_depbar) + " -> " + std::to_string(counter));
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
    trace(
        "shmem.portA w" + std::to_string(request.warp) + " round " +
        std::to_string(request.next_round + 1) + "/" + std::to_string(request.rounds.size()));
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
    trace("shmem.complete w" + std::to_string(request.warp));
    port_a_.reset();
}

void Machine::issue_one() {
    std::vector<std::string> reasons;
    for (std::uint32_t offset = 0; offset < warps_.size(); ++offset) {
        const auto index = (scheduler_cursor_ + offset) % static_cast<std::uint32_t>(warps_.size());
        auto& warp = warps_[index];
        if (warp.halted) continue;
        if (cycle_ < warp.stall_until) {
            reasons.push_back("w" + std::to_string(index) + ":stall-count");
            continue;
        }
        const auto attempt = try_issue(warp);
        if (attempt.issued) {
            scheduler_cursor_ = (index + 1) % static_cast<std::uint32_t>(warps_.size());
            return;
        }
        reasons.push_back("w" + std::to_string(index) + ":" + attempt.reason);
    }

    std::ostringstream message;
    message << "issue.idle";
    if (!reasons.empty()) {
        message << " [";
        for (std::size_t i = 0; i < reasons.size(); ++i) {
            if (i != 0) message << ", ";
            message << reasons[i];
        }
        message << ']';
    }
    trace(message.str());
}

Machine::IssueAttempt Machine::try_issue(Warp& warp) {
    const auto inst = instruction(warp);
    if (inst.control.predicate != 7 || inst.control.predicate_invert) {
        panic(warp, inst, "predicated execution is not implemented yet");
    }
    for (std::uint8_t bar = 0; bar < 6; ++bar) {
        if ((inst.control.wait_mask & (1u << bar)) != 0 && warp.depbar[bar] != 0) {
            return {false, "depbar-wait(sb" + std::to_string(bar) + ")"};
        }
    }

    const auto sources = source_registers(inst);
    const auto destination = destination_register(inst);
    for (const auto reg : sources) {
        if (reg >= config_.registers) panic(warp, inst, "source register out of range");
    }
    if (destination.has_value() && *destination >= config_.registers) {
        panic(warp, inst, "destination register out of range");
    }
    check_register_hazards(warp, inst);
    check_rf_read_ports(inst);
    if ((inst.opcode == Opcode::lds || inst.opcode == Opcode::sts) && port_a_.has_value()) {
        return {false, "portA-busy"};
    }

    std::vector<std::uint32_t> result(config_.lanes, 0);
    const auto latency = fixed_latency(inst, config_.timing);
    const auto result_cycle = cycle_ + latency;

    switch (inst.opcode) {
    case Opcode::sread:
        for (std::uint32_t lane = 0; lane < config_.lanes; ++lane) {
            switch (static_cast<SpecialReg>(inst.imm)) {
            case SpecialReg::lane_id: result[lane] = lane; break;
            case SpecialReg::warp_id: result[lane] = warp.id; break;
            case SpecialReg::thread_id: result[lane] = warp.id * config_.lanes + lane; break;
            case SpecialReg::n_warps: result[lane] = static_cast<std::uint32_t>(warps_.size()); break;
            default: panic(warp, inst, "invalid special register");
            }
        }
        break;
    case Opcode::mov:
        std::fill(result.begin(), result.end(), static_cast<std::uint32_t>(inst.imm));
        break;
    case Opcode::add:
        for (std::uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = rf(warp, lane, inst.src0) + rf(warp, lane, inst.src1);
        }
        break;
    case Opcode::addi:
        for (std::uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = rf(warp, lane, inst.src0) + static_cast<std::uint32_t>(inst.imm);
        }
        break;
    case Opcode::shli:
        if (inst.imm < 0 || inst.imm > 31) panic(warp, inst, "shift outside 0..31");
        for (std::uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = rf(warp, lane, inst.src0) << inst.imm;
        }
        break;
    case Opcode::fadd:
        for (std::uint32_t lane = 0; lane < config_.lanes; ++lane) {
            result[lane] = float_bits(
                bits_float(rf(warp, lane, inst.src0)) + bits_float(rf(warp, lane, inst.src1)));
        }
        break;
    case Opcode::lds: {
        if (inst.control.set_depbar >= 6) panic(warp, inst, "lds has no depbar");
        auto rounds = build_shmem_rounds(warp, inst, true);
        const auto complete_cycle = cycle_ + rounds.size() + 1;
        if (!reserve_write(inst.dst, complete_cycle)) return {false, "rf-write-reservation"};
        warp.ready_cycle[inst.dst] = complete_cycle;
        ++warp.depbar[inst.control.set_depbar];
        trace(
            "depbar.inc w" + std::to_string(warp.id) + " sb" +
            std::to_string(inst.control.set_depbar) + " -> " +
            std::to_string(warp.depbar[inst.control.set_depbar]));
        ShmemRequest request;
        request.load = true;
        request.warp = warp.id;
        request.dst = inst.dst;
        request.depbar = inst.control.set_depbar;
        request.rounds = std::move(rounds);
        request.load_values.resize(config_.lanes, 0);
        port_a_ = std::move(request);
        break;
    }
    case Opcode::sts: {
        if (inst.control.set_depbar >= 6) panic(warp, inst, "sts has no depbar");
        auto rounds = build_shmem_rounds(warp, inst, false);
        ++warp.depbar[inst.control.set_depbar];
        trace(
            "depbar.inc w" + std::to_string(warp.id) + " sb" +
            std::to_string(inst.control.set_depbar) + " -> " +
            std::to_string(warp.depbar[inst.control.set_depbar]));
        ShmemRequest request;
        request.load = false;
        request.warp = warp.id;
        request.depbar = inst.control.set_depbar;
        request.rounds = std::move(rounds);
        port_a_ = std::move(request);
        break;
    }
    case Opcode::halt:
        warp.halted = true;
        break;
    default:
        panic(warp, inst, "unknown opcode");
    }

    if (latency != 0) {
        if (!destination.has_value()) panic(warp, inst, "fixed-latency opcode has no destination");
        if (!reserve_write(*destination, result_cycle)) return {false, "rf-write-reservation"};
        schedule_register_write(warp, *destination, result_cycle, std::move(result), opcode_name(inst.opcode));
    } else if (inst.opcode != Opcode::lds && inst.opcode != Opcode::sts && inst.opcode != Opcode::halt) {
        panic(warp, inst, "zero-latency RF writer is unsupported");
    }

    warp.stall_until = cycle_ + inst.control.stall;
    const auto old_pc = warp.pc;
    warp.pc += 8;
    trace(
        "issue w" + std::to_string(warp.id) + " pc=" + std::to_string(old_pc) + " " +
        format_instruction(inst));
    return {true, {}};
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

std::vector<std::uint8_t> Machine::source_registers(const Instruction& inst) const {
    switch (inst.opcode) {
    case Opcode::add:
    case Opcode::fadd: return {inst.src0, inst.src1};
    case Opcode::addi:
    case Opcode::shli:
    case Opcode::lds: return {inst.src0};
    case Opcode::sts: return {inst.src0, inst.src1};
    default: return {};
    }
}

std::optional<std::uint8_t> Machine::destination_register(const Instruction& inst) const {
    switch (inst.opcode) {
    case Opcode::sread:
    case Opcode::mov:
    case Opcode::add:
    case Opcode::addi:
    case Opcode::shli:
    case Opcode::fadd:
    case Opcode::lds: return inst.dst;
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
    std::array<std::optional<std::uint8_t>, 3> bank_owner;
    for (const auto reg : source_registers(inst)) {
        const auto bank = reg % 3;
        if (bank_owner[bank].has_value() && *bank_owner[bank] != reg) {
            throw SimError("RF bank read conflict in '" + format_instruction(inst) + "'");
        }
        bank_owner[bank] = reg;
    }
}

bool Machine::reserve_write(std::uint8_t reg, std::uint64_t at_cycle) {
    auto& slots = write_reservations_[at_cycle];
    const auto bank = reg % 3;
    if (slots[bank]) return false;
    slots[bank] = true;
    return true;
}

std::uint32_t& Machine::rf(Warp& warp, std::uint32_t lane, std::uint32_t reg) {
    return warp.rf.at(static_cast<std::size_t>(lane) * config_.registers + reg);
}

std::uint32_t Machine::rf(const Warp& warp, std::uint32_t lane, std::uint32_t reg) const {
    return warp.rf.at(static_cast<std::size_t>(lane) * config_.registers + reg);
}

std::vector<std::vector<Machine::LaneAccess>> Machine::build_shmem_rounds(
    const Warp& warp, const Instruction& inst, bool load) const {
    struct Group {
        std::uint32_t address{};
        std::vector<LaneAccess> accesses;
    };
    std::vector<std::vector<Group>> groups(config_.shmem_banks);
    for (std::uint32_t lane = 0; lane < config_.lanes; ++lane) {
        const auto base = static_cast<std::int64_t>(rf(warp, lane, inst.src0));
        const auto address = base + inst.imm;
        if (address < 0 || address > std::numeric_limits<std::uint32_t>::max()) {
            panic(warp, inst, "shmem address arithmetic overflow");
        }
        const auto byte_address = static_cast<std::uint32_t>(address);
        if (byte_address % 4 != 0) panic(warp, inst, "unaligned 32-bit shmem access");
        if (static_cast<std::uint64_t>(byte_address) + 4 > shmem_.size()) {
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

void Machine::schedule_register_write(
    Warp& warp,
    std::uint8_t dst,
    std::uint64_t ready_cycle,
    std::vector<std::uint32_t> values,
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

void Machine::trace(std::string message) {
    trace_.push_back({cycle_, std::move(message)});
}

[[noreturn]] void Machine::panic(
    const Warp& warp, const Instruction& inst, const std::string& reason) const {
    throw SimError(
        "PANIC cycle=" + std::to_string(cycle_) + " warp=" + std::to_string(warp.id) +
        " pc=" + std::to_string(warp.pc) + " reason=" + reason + " inst='" +
        format_instruction(inst) + "'");
}

} // namespace nanosm

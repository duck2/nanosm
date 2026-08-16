#include "compiler.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <optional>
#include <regex>
#include <sstream>
#include <unordered_map>

namespace nanosm {
namespace {

std::string trim(std::string value) {
    const auto first = std::find_if_not(value.begin(), value.end(), [](unsigned char c) {
        return std::isspace(c) != 0;
    });
    const auto last = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char c) {
        return std::isspace(c) != 0;
    }).base();
    if (first >= last) return {};
    return std::string(first, last);
}

std::string upper(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return value;
}

std::vector<std::string> split(const std::string& text, char separator) {
    std::vector<std::string> result;
    std::size_t begin = 0;
    for (std::size_t i = 0; i <= text.size(); ++i) {
        if (i == text.size() || text[i] == separator) {
            result.push_back(trim(text.substr(begin, i - begin)));
            begin = i + 1;
        }
    }
    return result;
}

std::vector<std::string> operands(const std::string& text) {
    std::vector<std::string> result;
    std::size_t begin = 0;
    int brackets = 0;
    for (std::size_t i = 0; i < text.size(); ++i) {
        brackets += text[i] == '[' ? 1 : 0;
        brackets -= text[i] == ']' ? 1 : 0;
        if (text[i] == ',' && brackets == 0) {
            result.push_back(trim(text.substr(begin, i - begin)));
            begin = i + 1;
        }
    }
    result.push_back(trim(text.substr(begin)));
    return result;
}

uint8_t parse_reg(const std::string& text, uint32_t line) {
    static const std::regex pattern(R"(^[rR]([0-9]+)$)");
    std::smatch match;
    if (!std::regex_match(text, match, pattern)) {
        throw SimError("line " + std::to_string(line) + ": expected register, got '" + text + "'");
    }
    const auto value = std::stoul(match[1].str());
    if (value > 255) {
        throw SimError("line " + std::to_string(line) + ": register must fit eight bits");
    }
    return static_cast<uint8_t>(value);
}

std::pair<uint8_t, bool> parse_predicate(const std::string& text, uint32_t line) {
    auto value = text;
    bool invert = false;
    if (!value.empty() && value.front() == '!') {
        invert = true;
        value.erase(value.begin());
    }
    if (value == "PT") return {7, invert};
    if (value.size() == 2 && value[0] == 'p' && value[1] >= '0' && value[1] <= '6') {
        return {static_cast<uint8_t>(value[1] - '0'), invert};
    }
    throw SimError("line " + std::to_string(line) + ": bad predicate '" + text + "'");
}

bool is_setp(Opcode opcode) {
    return (opcode >= Opcode::setp_eq && opcode <= Opcode::setp_ge) ||
        (opcode >= Opcode::setpi_eq && opcode <= Opcode::setpi_ge);
}

bool is_branch(Opcode opcode) {
    return opcode == Opcode::bra || opcode == Opcode::bra_uni;
}

Opcode parse_setp_opcode(std::string op, bool immediate, uint32_t line) {
    if (op.ends_with(".u32")) op.resize(op.size() - 4);
    if (op == "setp.eq") return immediate ? Opcode::setpi_eq : Opcode::setp_eq;
    if (op == "setp.ne") return immediate ? Opcode::setpi_ne : Opcode::setp_ne;
    if (op == "setp.lt") return immediate ? Opcode::setpi_lt : Opcode::setp_lt;
    if (op == "setp.le") return immediate ? Opcode::setpi_le : Opcode::setp_le;
    if (op == "setp.gt") return immediate ? Opcode::setpi_gt : Opcode::setp_gt;
    if (op == "setp.ge") return immediate ? Opcode::setpi_ge : Opcode::setp_ge;
    throw SimError("line " + std::to_string(line) + ": unknown setp comparison '" + op + "'");
}

int32_t parse_value(
    const std::string& text,
    const std::unordered_map<std::string, int32_t>& constants,
    uint32_t line) {
    const auto found = constants.find(text);
    if (found != constants.end()) return found->second;
    std::size_t consumed = 0;
    try {
        const auto value = std::stoll(text, &consumed, 0);
        if (consumed != text.size() || value < encoding::immediate_min || value > encoding::immediate_max) {
            throw std::invalid_argument("range or suffix");
        }
        return static_cast<int32_t>(value);
    } catch (const std::exception&) {
        throw SimError("line " + std::to_string(line) + ": bad or unknown immediate '" + text + "'");
    }
}

struct Address {
    uint8_t base = 0;
    int32_t offset = 0;
};

Address parse_address(
    const std::string& text,
    const std::unordered_map<std::string, int32_t>& constants,
    uint32_t line) {
    static const std::regex pattern(
        R"(^\[\s*([rR][0-9]+)\s*(?:([+-])\s*([A-Za-z_][A-Za-z0-9_]*|0[xX][0-9A-Fa-f]+|[0-9]+))?\s*\]$)");
    std::smatch match;
    if (!std::regex_match(text, match, pattern)) {
        throw SimError("line " + std::to_string(line) + ": bad address '" + text + "'");
    }
    Address result;
    result.base = parse_reg(match[1].str(), line);
    if (match[3].matched) {
        result.offset = parse_value(match[3].str(), constants, line);
        if (match[2].str() == "-") result.offset = -result.offset;
    }
    return result;
}

void expect_count(const std::vector<std::string>& args, std::size_t count, uint32_t line) {
    if (args.size() != count) {
        throw SimError(
            "line " + std::to_string(line) + ": expected " + std::to_string(count) +
            " operands, got " + std::to_string(args.size()));
    }
}

uint8_t parse_bar(const std::string& value, uint32_t line) {
    static const std::regex pattern(R"(^sb([0-5])$)");
    std::smatch match;
    if (!std::regex_match(value, match, pattern)) {
        throw SimError("line " + std::to_string(line) + ": expected sb0..sb5, got '" + value + "'");
    }
    return static_cast<uint8_t>(match[1].str()[0] - '0');
}

Control parse_control(const std::string& text, uint32_t line) {
    Control control;
    if (text.empty()) return control;
    if (text.front() != '{' || text.back() != '}') {
        throw SimError("line " + std::to_string(line) + ": malformed control block '" + text + "'");
    }
    for (const auto& item : split(text.substr(1, text.size() - 2), ',')) {
        const auto equals = item.find('=');
        if (equals == std::string::npos) {
            throw SimError("line " + std::to_string(line) + ": malformed control item '" + item + "'");
        }
        const auto name = trim(item.substr(0, equals));
        const auto value = trim(item.substr(equals + 1));
        if (name == "pred") {
            const auto [predicate, invert] = parse_predicate(value, line);
            control.predicate = predicate;
            control.predicate_invert = invert;
        } else if (name == "stall") {
            const auto stall = parse_value(value, {}, line);
            if (stall < 0 || stall > 31) {
                throw SimError("line " + std::to_string(line) + ": stall must be 0..31");
            }
            control.stall = static_cast<uint8_t>(stall);
        } else if (name == "set") {
            control.set_depbar = value == "-" ? 7 : parse_bar(value, line);
        } else if (name == "wait") {
            control.wait_mask = 0;
            if (value != "-") {
                for (const auto& bar : split(value, '|')) {
                    control.wait_mask |= static_cast<uint8_t>(1u << parse_bar(bar, line));
                }
            }
        } else {
            throw SimError("line " + std::to_string(line) + ": unknown control field '" + name + "'");
        }
    }
    return control;
}

AssemblyProgram parse_program(std::string_view source, bool allow_control) {
    AssemblyProgram program;
    std::unordered_map<std::string, int32_t> constants;
    std::unordered_map<std::string, std::size_t> labels;
    struct PendingBranch {
        std::size_t instruction = 0;
        std::string target;
        uint32_t line = 0;
    };
    std::vector<PendingBranch> pending_branches;
    std::istringstream input{std::string(source)};
    std::string raw;
    uint32_t line_number = 0;

    while (std::getline(input, raw)) {
        ++line_number;
        const auto comment = raw.find(';');
        auto clean = trim(raw.substr(0, comment));
        if (clean.empty()) continue;

        static const std::regex label_pattern(R"(^([A-Za-z_][A-Za-z0-9_]*):)");
        std::smatch label_match;
        while (std::regex_search(clean, label_match, label_pattern)) {
            const auto name = label_match[1].str();
            if (labels.contains(name) || constants.contains(name)) {
                throw SimError("line " + std::to_string(line_number) + ": duplicate symbol '" + name + "'");
            }
            labels[name] = program.instructions.size();
            program.labels.push_back({name, program.instructions.size()});
            clean = trim(clean.substr(label_match[0].length()));
            if (clean.empty()) break;
        }
        if (clean.empty()) continue;

        if (clean.starts_with(".equ")) {
            auto args = operands(trim(clean.substr(4)));
            expect_count(args, 2, line_number);
            const auto value = parse_value(args[1], constants, line_number);
            if (constants.contains(args[0]) || labels.contains(args[0])) {
                throw SimError("line " + std::to_string(line_number) + ": duplicate constant '" + args[0] + "'");
            }
            constants[args[0]] = value;
            program.constants.emplace_back(args[0], value);
            continue;
        }
        if (clean.starts_with(".profile")) {
            if (!allow_control) {
                throw SimError("line " + std::to_string(line_number) + ": qtx cannot select a profile");
            }
            if (!program.profile_name.empty()) {
                throw SimError("line " + std::to_string(line_number) + ": duplicate .profile");
            }
            const auto args = operands(trim(clean.substr(8)));
            expect_count(args, 2, line_number);
            std::size_t consumed = 0;
            try {
                program.profile_fingerprint = std::stoull(args[1], &consumed, 0);
                if (consumed != args[1].size()) throw std::invalid_argument("suffix");
            } catch (const std::exception&) {
                throw SimError("line " + std::to_string(line_number) + ": bad profile fingerprint");
            }
            program.profile_name = args[0];
            continue;
        }
        if (clean.front() == '.') {
            throw SimError("line " + std::to_string(line_number) + ": unsupported directive '" + clean + "'");
        }

        std::string control_text;
        const auto control_begin = clean.find('{');
        if (control_begin != std::string::npos) {
            if (!allow_control) {
                throw SimError("line " + std::to_string(line_number) + ": qtx cannot contain control bits");
            }
            control_text = trim(clean.substr(control_begin));
            clean = trim(clean.substr(0, control_begin));
        }

        std::optional<std::pair<uint8_t, bool>> prefix_predicate;
        if (!clean.empty() && clean.front() == '@') {
            const auto end = clean.find_first_of(" \t");
            if (end == std::string::npos) {
                throw SimError("line " + std::to_string(line_number) + ": predicate has no instruction");
            }
            prefix_predicate = parse_predicate(clean.substr(1, end - 1), line_number);
            clean = trim(clean.substr(end + 1));
        }

        const auto space = clean.find_first_of(" \t");
        const auto op = clean.substr(0, space);
        const auto tail = space == std::string::npos ? std::string{} : trim(clean.substr(space + 1));
        const auto args = tail.empty() ? std::vector<std::string>{} : operands(tail);
        Instruction inst;
        inst.line = line_number;
        inst.text = clean;
        inst.control = parse_control(control_text, line_number);
        if (prefix_predicate.has_value()) {
            if ((inst.control.predicate != 7 || inst.control.predicate_invert) &&
                (inst.control.predicate != prefix_predicate->first ||
                 inst.control.predicate_invert != prefix_predicate->second)) {
                throw SimError("line " + std::to_string(line_number) + ": conflicting predicate controls");
            }
            inst.control.predicate = prefix_predicate->first;
            inst.control.predicate_invert = prefix_predicate->second;
        }

        if (op == "sread") {
            expect_count(args, 2, line_number);
            inst.opcode = Opcode::sread;
            inst.dst = parse_reg(args[0], line_number);
            const auto name = upper(args[1]);
            if (name == "LANE_ID") inst.imm = static_cast<int32_t>(SpecialReg::lane_id);
            else if (name == "WARP_ID") inst.imm = static_cast<int32_t>(SpecialReg::warp_id);
            else if (name == "THREAD_ID") inst.imm = static_cast<int32_t>(SpecialReg::thread_id);
            else if (name == "N_WARPS") inst.imm = static_cast<int32_t>(SpecialReg::n_warps);
            else throw SimError("line " + std::to_string(line_number) + ": unknown special register '" + args[1] + "'");
        } else if (op == "mov") {
            expect_count(args, 2, line_number);
            inst.opcode = Opcode::mov;
            inst.dst = parse_reg(args[0], line_number);
            inst.imm = parse_value(args[1], constants, line_number);
        } else if (op == "add" || op == "fadd") {
            expect_count(args, 3, line_number);
            inst.opcode = op == "add" ? Opcode::add : Opcode::fadd;
            inst.dst = parse_reg(args[0], line_number);
            inst.src0 = parse_reg(args[1], line_number);
            inst.src1 = parse_reg(args[2], line_number);
        } else if (op.starts_with("setp.")) {
            expect_count(args, 3, line_number);
            const auto [predicate, invert] = parse_predicate(args[0], line_number);
            if (predicate == 7 || invert) {
                throw SimError("line " + std::to_string(line_number) + ": setp destination must be p0..p6");
            }
            inst.dst = predicate;
            inst.src0 = parse_reg(args[1], line_number);
            static const std::regex register_pattern(R"(^[rR][0-9]+$)");
            const bool immediate = !std::regex_match(args[2], register_pattern);
            inst.opcode = parse_setp_opcode(op, immediate, line_number);
            if (immediate) inst.imm = parse_value(args[2], constants, line_number);
            else inst.src1 = parse_reg(args[2], line_number);
        } else if (op == "addi" || op == "shli") {
            expect_count(args, 3, line_number);
            inst.opcode = op == "addi" ? Opcode::addi : Opcode::shli;
            inst.dst = parse_reg(args[0], line_number);
            inst.src0 = parse_reg(args[1], line_number);
            inst.imm = parse_value(args[2], constants, line_number);
        } else if (op == "lds" || op == "ldg") {
            expect_count(args, 2, line_number);
            inst.opcode = op == "lds" ? Opcode::lds : Opcode::ldg;
            inst.dst = parse_reg(args[0], line_number);
            const auto addr = parse_address(args[1], constants, line_number);
            inst.src0 = addr.base;
            inst.imm = addr.offset;
        } else if (op == "sts" || op == "stg") {
            expect_count(args, 2, line_number);
            inst.opcode = op == "sts" ? Opcode::sts : Opcode::stg;
            const auto addr = parse_address(args[0], constants, line_number);
            inst.src0 = addr.base;
            inst.imm = addr.offset;
            inst.src1 = parse_reg(args[1], line_number);
        } else if (op == "bra" || op == "bra.uni") {
            expect_count(args, 1, line_number);
            static const std::regex target_pattern(R"(^[A-Za-z_][A-Za-z0-9_]*$)");
            if (!std::regex_match(args[0], target_pattern)) {
                throw SimError("line " + std::to_string(line_number) + ": bad branch target '" + args[0] + "'");
            }
            inst.opcode = op == "bra" ? Opcode::bra : Opcode::bra_uni;
            pending_branches.push_back({program.instructions.size(), args[0], line_number});
        } else if (op == "halt") {
            expect_count(args, 0, line_number);
            inst.opcode = Opcode::halt;
        } else {
            throw SimError("line " + std::to_string(line_number) + ": unknown opcode '" + op + "'");
        }
        program.instructions.push_back(std::move(inst));
    }
    if (program.instructions.empty()) throw SimError("kernel contains no instructions");
    for (const auto& branch : pending_branches) {
        const auto target = labels.find(branch.target);
        if (target == labels.end()) {
            throw SimError("line " + std::to_string(branch.line) + ": unknown label '" + branch.target + "'");
        }
        if (target->second >= program.instructions.size()) {
            throw SimError("line " + std::to_string(branch.line) + ": branch target falls off kernel");
        }
        const auto target_pc = static_cast<int64_t>(target->second) * 8;
        const auto next_pc = static_cast<int64_t>(branch.instruction + 1) * 8;
        const auto displacement = target_pc - next_pc;
        if (displacement < encoding::immediate_min || displacement > encoding::immediate_max) {
            throw SimError("line " + std::to_string(branch.line) + ": branch target is out of range");
        }
        program.instructions[branch.instruction].imm = static_cast<int32_t>(displacement);
    }
    return program;
}

std::vector<uint8_t> sources(const Instruction& inst) {
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

std::optional<uint8_t> predicate_destination(const Instruction& inst) {
    if (is_setp(inst.opcode)) return inst.dst;
    return std::nullopt;
}

std::optional<uint8_t> destination(const Instruction& inst) {
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

uint16_t register_count(const AssemblyProgram& program) {
    uint16_t count = 0;
    for (const auto& inst : program.instructions) {
        for (const auto reg : sources(inst)) {
            count = std::max<uint16_t>(count, static_cast<uint16_t>(reg) + 1);
        }
        if (const auto dst = destination(inst); dst.has_value()) {
            count = std::max<uint16_t>(count, static_cast<uint16_t>(*dst) + 1);
        }
    }
    return count;
}

std::string read_file(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw SimError("cannot open '" + path + "'");
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

} // namespace

AssemblyProgram compile_qtx(std::string_view source, const DeviceProfile& profile) {
    validate_device_profile(profile);
    auto program = parse_program(source, false);
    program.register_count = register_count(program);
    program.profile_name = profile.name;
    program.profile_fingerprint = device_profile_fingerprint(profile);
    std::array<bool, 6> live{};
    std::array<std::optional<uint8_t>, 256> pending_register{};
    std::optional<uint8_t> store_barrier;
    std::array<int64_t, 7> predicate_ready{};

    // Virtual issue timeline. Consecutive same-warp issues are at least max(stall, 1)
    // apart, and real gaps only grow (scheduler contention, depbar waits, structural
    // stalls), so any spacing proven here holds on the machine. reg_ready is the vtime
    // a fixed-latency result commits; async results are covered by depbars instead.
    // A stall is paid only when a consumer sits closer than its producer's latency,
    // and it lands on the instruction just before the consumer.
    std::array<int64_t, 256> reg_ready{};
    int64_t vtime = 0;
    int64_t previous_vtime = 0;
    Instruction* previous = nullptr;

    const auto allocate = [&]() -> uint8_t {
        for (uint8_t bar = 0; bar < 6; ++bar) {
            if (!live[bar]) {
                live[bar] = true;
                return bar;
            }
        }
        throw SimError("qtx scheduler exhausted six live depbars");
    };

    for (auto& inst : program.instructions) {
        if (inst.opcode == Opcode::bra &&
            (inst.control.predicate != 7 || inst.control.predicate_invert)) {
            throw SimError("line " + std::to_string(inst.line) + ": bra is unconditional");
        }
        if ((inst.control.predicate != 7 || inst.control.predicate_invert) &&
            inst.opcode != Opcode::bra_uni) {
            throw SimError(
                "line " + std::to_string(inst.line) +
                ": only bra.uni may be predicated before divergent execution exists");
        }
        uint8_t wait_mask = 0;
        for (const auto source : sources(inst)) {
            if (pending_register[source].has_value()) {
                wait_mask |= static_cast<uint8_t>(1u << *pending_register[source]);
            }
        }
        if (const auto dst = destination(inst); dst.has_value() && pending_register[*dst].has_value()) {
            wait_mask |= static_cast<uint8_t>(1u << *pending_register[*dst]);
        }
        if (inst.opcode == Opcode::halt || is_branch(inst.opcode)) {
            for (uint8_t bar = 0; bar < 6; ++bar) {
                if (live[bar]) wait_mask |= static_cast<uint8_t>(1u << bar);
            }
        }
        inst.control.wait_mask = wait_mask;

        if (wait_mask != 0) {
            for (auto& pending : pending_register) {
                if (pending.has_value() && (wait_mask & (1u << *pending)) != 0) pending.reset();
            }
            for (uint8_t bar = 0; bar < 6; ++bar) {
                if ((wait_mask & (1u << bar)) != 0) live[bar] = false;
            }
            if (store_barrier.has_value() && (wait_mask & (1u << *store_barrier)) != 0) {
                store_barrier.reset();
            }
        }

        int64_t required = vtime;
        for (const auto source : sources(inst)) {
            required = std::max(required, reg_ready[source]);
        }
        if (const auto dst = destination(inst); dst.has_value()) {
            required = std::max(required, reg_ready[*dst]); // RF write order = issue order
        }
        if (inst.opcode == Opcode::bra_uni && inst.control.predicate != 7) {
            required = std::max(required, predicate_ready[inst.control.predicate]);
        }
        if (const auto dst = predicate_destination(inst); dst.has_value()) {
            required = std::max(required, predicate_ready[*dst]);
        }
        if (required > vtime) {
            if (previous == nullptr) {
                throw SimError("internal: first instruction cannot need a stall");
            }
            const auto stall = required - previous_vtime;
            if (stall > static_cast<int64_t>(encoding::stall_mask)) {
                throw SimError(
                    "line " + std::to_string(inst.line) +
                    ": required stall exceeds the five-bit field (nop insertion not implemented)");
            }
            previous->control.stall = static_cast<uint8_t>(stall);
            vtime = required;
        }
        inst.control.stall = 0;

        const auto latency = fixed_latency(inst, profile.timing);
        if (const auto dst = destination(inst); latency != 0 && dst.has_value()) {
            reg_ready[*dst] = vtime + latency;
        }
        if (const auto dst = predicate_destination(inst); latency != 0 && dst.has_value()) {
            predicate_ready[*dst] = vtime + latency;
        }
        if (inst.opcode == Opcode::lds || inst.opcode == Opcode::ldg) {
            const auto bar = allocate();
            inst.control.set_depbar = bar;
            pending_register[inst.dst] = bar;
        } else if (inst.opcode == Opcode::sts || inst.opcode == Opcode::stg) {
            if (!store_barrier.has_value()) store_barrier = allocate();
            inst.control.set_depbar = *store_barrier;
        }

        if (is_branch(inst.opcode)) {
            int64_t all_ready = vtime + 1;
            for (const auto ready : reg_ready) all_ready = std::max(all_ready, ready);
            for (const auto ready : predicate_ready) all_ready = std::max(all_ready, ready);
            if (all_ready > vtime + 1) {
                const auto stall = all_ready - vtime;
                if (stall > static_cast<int64_t>(encoding::stall_mask)) {
                    throw SimError(
                        "line " + std::to_string(inst.line) +
                        ": control-flow drain exceeds the five-bit stall field");
                }
                inst.control.stall = static_cast<uint8_t>(stall);
            }

            // Every branch is a conservative block boundary for v0: its wait mask
            // drains async work, and its stall covers outstanding fixed-latency writes.
            // Both the target and fallthrough therefore begin with clean timing state.
            live.fill(false);
            pending_register.fill(std::nullopt);
            store_barrier.reset();
            reg_ready.fill(0);
            predicate_ready.fill(0);
            previous = nullptr;
            previous_vtime = 0;
            vtime = 0;
            continue;
        }

        previous = &inst;
        previous_vtime = vtime;
        vtime += 1;
    }
    return program;
}

AssemblyProgram compile_qtx_file(const std::string& path, const DeviceProfile& profile) {
    return compile_qtx(read_file(path), profile);
}

std::string emit_asm(const AssemblyProgram& program) {
    std::ostringstream out;
    out << "; scheduled qtx assembly: controls are part of the instruction word\n";
    out << "; {pred, stall, set, wait}; six warp-local depbar counters sb0..sb5\n\n";
    out << ".profile " << program.profile_name << ", 0x"
        << format_profile_fingerprint(program.profile_fingerprint) << "\n\n";
    for (const auto& [name, value] : program.constants) {
        out << ".equ " << name << ", " << value << '\n';
    }
    if (!program.constants.empty()) out << '\n';
    for (std::size_t index = 0; index < program.instructions.size(); ++index) {
        for (const auto& label : program.labels) {
            if (label.instruction_index == index) out << label.name << ":\n";
        }
        const auto& inst = program.instructions[index];
        out << "        " << inst.text << ' ' << format_control(inst.control) << '\n';
    }
    return out.str();
}

Kernel assemble_asm(std::string_view source, const DeviceProfile& profile) {
    validate_device_profile(profile);
    const auto program = parse_program(source, true);
    const auto expected = device_profile_fingerprint(profile);
    if (program.profile_name.empty()) throw SimError("assembly has no .profile directive");
    if (program.profile_name != profile.name || program.profile_fingerprint != expected) {
        throw SimError(
            "assembly profile '" + program.profile_name + "@" +
            format_profile_fingerprint(program.profile_fingerprint) + "' does not match '" +
            profile.name + "@" + format_profile_fingerprint(expected) + "'");
    }
    Kernel kernel;
    kernel.register_count = register_count(program);
    kernel.profile_name = program.profile_name;
    kernel.profile_fingerprint = program.profile_fingerprint;
    kernel.words.reserve(program.instructions.size());
    for (std::size_t index = 0; index < program.instructions.size(); ++index) {
        const auto& inst = program.instructions[index];
        if ((inst.opcode == Opcode::lds || inst.opcode == Opcode::sts ||
             inst.opcode == Opcode::ldg || inst.opcode == Opcode::stg) && inst.control.set_depbar >= 6) {
            throw SimError("line " + std::to_string(inst.line) + ": memory operation requires set=sb0..sb5");
        }
        if (inst.opcode != Opcode::lds && inst.opcode != Opcode::sts &&
            inst.opcode != Opcode::ldg && inst.opcode != Opcode::stg && inst.control.set_depbar != 7) {
            throw SimError("line " + std::to_string(inst.line) + ": this opcode cannot set a depbar");
        }
        if (is_setp(inst.opcode) && inst.dst >= 7) {
            throw SimError("line " + std::to_string(inst.line) + ": setp destination must be p0..p6");
        }
        if ((inst.control.predicate != 7 || inst.control.predicate_invert) &&
            inst.opcode != Opcode::bra_uni) {
            throw SimError(
                "line " + std::to_string(inst.line) +
                ": only bra.uni may be predicated before divergent execution exists");
        }
        if (is_branch(inst.opcode)) {
            const auto target = static_cast<int64_t>(index + 1) * 8 + inst.imm;
            if (inst.imm % 8 != 0 || target < 0 ||
                static_cast<uint64_t>(target / 8) >= program.instructions.size()) {
                throw SimError("line " + std::to_string(inst.line) + ": invalid branch displacement");
            }
        }
        kernel.words.push_back(encode(inst));
    }
    return kernel;
}

Kernel assemble_asm_file(const std::string& path, const DeviceProfile& profile) {
    return assemble_asm(read_file(path), profile);
}

std::string emit_hex(const Kernel& kernel) {
    std::ostringstream out;
    out << ".profile " << kernel.profile_name << ", 0x"
        << format_profile_fingerprint(kernel.profile_fingerprint) << '\n';
    out << ".registers " << kernel.register_count << '\n';
    for (const auto word : kernel.words) {
        out << std::hex << std::setw(16) << std::setfill('0') << word
            << " ; " << format_instruction(decode(word)) << '\n';
    }
    return out.str();
}

Kernel load_hex(std::string_view source) {
    Kernel kernel;
    bool saw_register_count = false;
    std::istringstream input{std::string(source)};
    std::string raw;
    uint32_t line = 0;
    while (std::getline(input, raw)) {
        ++line;
        const auto comment = raw.find(';');
        const auto clean = trim(raw.substr(0, comment));
        if (clean.empty()) continue;
        if (clean.starts_with(".profile")) {
            if (!kernel.profile_name.empty()) {
                throw SimError("line " + std::to_string(line) + ": duplicate .profile");
            }
            const auto args = operands(trim(clean.substr(8)));
            expect_count(args, 2, line);
            std::size_t consumed = 0;
            try {
                kernel.profile_fingerprint = std::stoull(args[1], &consumed, 0);
                if (consumed != args[1].size()) throw std::invalid_argument("suffix");
            } catch (const std::exception&) {
                throw SimError("line " + std::to_string(line) + ": bad profile fingerprint");
            }
            kernel.profile_name = args[0];
            continue;
        }
        if (clean.starts_with(".registers")) {
            if (saw_register_count) {
                throw SimError("line " + std::to_string(line) + ": duplicate .registers");
            }
            saw_register_count = true;
            const auto value = trim(clean.substr(10));
            std::size_t consumed = 0;
            try {
                const auto count = std::stoul(value, &consumed, 0);
                if (consumed != value.size() || count > 256) {
                    throw std::invalid_argument("range or suffix");
                }
                kernel.register_count = static_cast<uint16_t>(count);
            } catch (const std::exception&) {
                throw SimError("line " + std::to_string(line) + ": bad register count");
            }
            continue;
        }
        std::size_t consumed = 0;
        try {
            const auto word = std::stoull(clean, &consumed, 16);
            if (consumed != clean.size()) throw std::invalid_argument("suffix");
            kernel.words.push_back(word);
        } catch (const std::exception&) {
            throw SimError("line " + std::to_string(line) + ": bad 64-bit hex word '" + clean + "'");
        }
    }
    if (kernel.profile_name.empty()) throw SimError("hex kernel has no .profile header");
    if (!saw_register_count) throw SimError("hex kernel has no .registers header");
    if (kernel.words.empty()) throw SimError("hex kernel contains no instructions");
    AssemblyProgram decoded;
    decoded.instructions.reserve(kernel.words.size());
    for (const auto word : kernel.words) decoded.instructions.push_back(decode(word));
    if (kernel.register_count != register_count(decoded)) {
        throw SimError("hex kernel register count does not match its instructions");
    }
    return kernel;
}

Kernel load_hex_file(const std::string& path) {
    return load_hex(read_file(path));
}

Kernel compile_and_assemble(std::string_view source, const DeviceProfile& profile) {
    return assemble_asm(emit_asm(compile_qtx(source, profile)), profile);
}

Kernel compile_and_assemble_file(const std::string& path, const DeviceProfile& profile) {
    return compile_and_assemble(read_file(path), profile);
}

} // namespace nanosm

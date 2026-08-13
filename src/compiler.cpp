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

std::uint8_t parse_reg(const std::string& text, std::uint32_t line) {
    static const std::regex pattern(R"(^[rR]([0-9]+)$)");
    std::smatch match;
    if (!std::regex_match(text, match, pattern)) {
        throw SimError("line " + std::to_string(line) + ": expected register, got '" + text + "'");
    }
    const auto value = std::stoul(match[1].str());
    if (value > 63) {
        throw SimError("line " + std::to_string(line) + ": register must fit six bits");
    }
    return static_cast<std::uint8_t>(value);
}

std::int32_t parse_value(
    const std::string& text,
    const std::unordered_map<std::string, std::int32_t>& constants,
    std::uint32_t line) {
    const auto found = constants.find(text);
    if (found != constants.end()) return found->second;
    std::size_t consumed = 0;
    try {
        const auto value = std::stoll(text, &consumed, 0);
        if (consumed != text.size() || value < encoding::immediate_min || value > encoding::immediate_max) {
            throw std::invalid_argument("range or suffix");
        }
        return static_cast<std::int32_t>(value);
    } catch (const std::exception&) {
        throw SimError("line " + std::to_string(line) + ": bad or unknown immediate '" + text + "'");
    }
}

struct Address {
    std::uint8_t base{};
    std::int32_t offset{};
};

Address parse_address(
    const std::string& text,
    const std::unordered_map<std::string, std::int32_t>& constants,
    std::uint32_t line) {
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

void expect_count(const std::vector<std::string>& args, std::size_t count, std::uint32_t line) {
    if (args.size() != count) {
        throw SimError(
            "line " + std::to_string(line) + ": expected " + std::to_string(count) +
            " operands, got " + std::to_string(args.size()));
    }
}

std::uint8_t parse_bar(const std::string& value, std::uint32_t line) {
    static const std::regex pattern(R"(^sb([0-5])$)");
    std::smatch match;
    if (!std::regex_match(value, match, pattern)) {
        throw SimError("line " + std::to_string(line) + ": expected sb0..sb5, got '" + value + "'");
    }
    return static_cast<std::uint8_t>(match[1].str()[0] - '0');
}

Control parse_control(const std::string& text, std::uint32_t line) {
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
            auto pred = value;
            if (!pred.empty() && pred.front() == '!') {
                control.predicate_invert = true;
                pred.erase(pred.begin());
            }
            if (pred == "PT") control.predicate = 7;
            else if (pred.size() == 2 && pred[0] == 'p' && pred[1] >= '0' && pred[1] <= '6') {
                control.predicate = static_cast<std::uint8_t>(pred[1] - '0');
            } else {
                throw SimError("line " + std::to_string(line) + ": bad predicate '" + value + "'");
            }
        } else if (name == "stall") {
            const auto stall = parse_value(value, {}, line);
            if (stall < 0 || stall > 31) {
                throw SimError("line " + std::to_string(line) + ": stall must be 0..31");
            }
            control.stall = static_cast<std::uint8_t>(stall);
        } else if (name == "set") {
            control.set_depbar = value == "-" ? 7 : parse_bar(value, line);
        } else if (name == "wait") {
            control.wait_mask = 0;
            if (value != "-") {
                for (const auto& bar : split(value, '|')) {
                    control.wait_mask |= static_cast<std::uint8_t>(1u << parse_bar(bar, line));
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
    std::unordered_map<std::string, std::int32_t> constants;
    std::istringstream input{std::string(source)};
    std::string raw;
    std::uint32_t line_number = 0;

    while (std::getline(input, raw)) {
        ++line_number;
        const auto comment = raw.find(';');
        auto clean = trim(raw.substr(0, comment));
        if (clean.empty()) continue;

        if (clean.starts_with(".equ")) {
            auto args = operands(trim(clean.substr(4)));
            expect_count(args, 2, line_number);
            const auto value = parse_value(args[1], constants, line_number);
            if (constants.contains(args[0])) {
                throw SimError("line " + std::to_string(line_number) + ": duplicate constant '" + args[0] + "'");
            }
            constants[args[0]] = value;
            program.constants.emplace_back(args[0], value);
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

        const auto space = clean.find_first_of(" \t");
        const auto op = clean.substr(0, space);
        const auto tail = space == std::string::npos ? std::string{} : trim(clean.substr(space + 1));
        const auto args = tail.empty() ? std::vector<std::string>{} : operands(tail);
        Instruction inst;
        inst.line = line_number;
        inst.text = clean;
        inst.control = parse_control(control_text, line_number);

        if (op == "sread") {
            expect_count(args, 2, line_number);
            inst.opcode = Opcode::sread;
            inst.dst = parse_reg(args[0], line_number);
            const auto name = upper(args[1]);
            if (name == "LANE_ID") inst.imm = static_cast<std::int32_t>(SpecialReg::lane_id);
            else if (name == "WARP_ID") inst.imm = static_cast<std::int32_t>(SpecialReg::warp_id);
            else if (name == "THREAD_ID") inst.imm = static_cast<std::int32_t>(SpecialReg::thread_id);
            else if (name == "N_WARPS") inst.imm = static_cast<std::int32_t>(SpecialReg::n_warps);
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
        } else if (op == "addi" || op == "shli") {
            expect_count(args, 3, line_number);
            inst.opcode = op == "addi" ? Opcode::addi : Opcode::shli;
            inst.dst = parse_reg(args[0], line_number);
            inst.src0 = parse_reg(args[1], line_number);
            inst.imm = parse_value(args[2], constants, line_number);
        } else if (op == "lds") {
            expect_count(args, 2, line_number);
            inst.opcode = Opcode::lds;
            inst.dst = parse_reg(args[0], line_number);
            const auto addr = parse_address(args[1], constants, line_number);
            inst.src0 = addr.base;
            inst.imm = addr.offset;
        } else if (op == "sts") {
            expect_count(args, 2, line_number);
            inst.opcode = Opcode::sts;
            const auto addr = parse_address(args[0], constants, line_number);
            inst.src0 = addr.base;
            inst.imm = addr.offset;
            inst.src1 = parse_reg(args[1], line_number);
        } else if (op == "halt") {
            expect_count(args, 0, line_number);
            inst.opcode = Opcode::halt;
        } else {
            throw SimError("line " + std::to_string(line_number) + ": unknown opcode '" + op + "'");
        }
        program.instructions.push_back(std::move(inst));
    }
    if (program.instructions.empty()) throw SimError("kernel contains no instructions");
    return program;
}

std::vector<std::uint8_t> sources(const Instruction& inst) {
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

std::optional<std::uint8_t> destination(const Instruction& inst) {
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

std::string read_file(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw SimError("cannot open '" + path + "'");
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

} // namespace

AssemblyProgram compile_qtx(std::string_view source, const Timing& timing) {
    auto program = parse_program(source, false);
    std::array<bool, 6> live{};
    std::array<std::optional<std::uint8_t>, 64> pending_register{};
    std::optional<std::uint8_t> store_barrier;

    // Virtual issue timeline. Consecutive same-warp issues are at least max(stall, 1)
    // apart, and real gaps only grow (scheduler contention, depbar waits, structural
    // stalls), so any spacing proven here holds on the machine. reg_ready is the vtime
    // a fixed-latency result commits; async results are covered by depbars instead.
    // A stall is paid only when a consumer sits closer than its producer's latency,
    // and it lands on the instruction just before the consumer.
    std::array<std::int64_t, 64> reg_ready{};
    std::int64_t vtime = 0;
    std::int64_t previous_vtime = 0;
    Instruction* previous = nullptr;

    const auto allocate = [&]() -> std::uint8_t {
        for (std::uint8_t bar = 0; bar < 6; ++bar) {
            if (!live[bar]) {
                live[bar] = true;
                return bar;
            }
        }
        throw SimError("qtx scheduler exhausted six live depbars");
    };

    for (auto& inst : program.instructions) {
        std::uint8_t wait_mask = 0;
        for (const auto source : sources(inst)) {
            if (pending_register[source].has_value()) {
                wait_mask |= static_cast<std::uint8_t>(1u << *pending_register[source]);
            }
        }
        if (const auto dst = destination(inst); dst.has_value() && pending_register[*dst].has_value()) {
            wait_mask |= static_cast<std::uint8_t>(1u << *pending_register[*dst]);
        }
        if (inst.opcode == Opcode::halt) {
            for (std::uint8_t bar = 0; bar < 6; ++bar) {
                if (live[bar]) wait_mask |= static_cast<std::uint8_t>(1u << bar);
            }
        }
        inst.control.wait_mask = wait_mask;

        if (wait_mask != 0) {
            for (auto& pending : pending_register) {
                if (pending.has_value() && (wait_mask & (1u << *pending)) != 0) pending.reset();
            }
            for (std::uint8_t bar = 0; bar < 6; ++bar) {
                if ((wait_mask & (1u << bar)) != 0) live[bar] = false;
            }
            if (store_barrier.has_value() && (wait_mask & (1u << *store_barrier)) != 0) {
                store_barrier.reset();
            }
        }

        std::int64_t required = vtime;
        for (const auto source : sources(inst)) {
            required = std::max(required, reg_ready[source]);
        }
        if (const auto dst = destination(inst); dst.has_value()) {
            required = std::max(required, reg_ready[*dst]); // RF write order = issue order
        }
        if (required > vtime) {
            if (previous == nullptr) {
                throw SimError("internal: first instruction cannot need a stall");
            }
            const auto stall = required - previous_vtime;
            if (stall > static_cast<std::int64_t>(encoding::stall_mask)) {
                throw SimError(
                    "line " + std::to_string(inst.line) +
                    ": required stall exceeds the five-bit field (nop insertion not implemented)");
            }
            previous->control.stall = static_cast<std::uint8_t>(stall);
            vtime = required;
        }
        inst.control.stall = 0;

        const auto latency = fixed_latency(inst, timing);
        if (const auto dst = destination(inst); latency != 0 && dst.has_value()) {
            reg_ready[*dst] = vtime + latency;
        }
        if (inst.opcode == Opcode::lds) {
            const auto bar = allocate();
            inst.control.set_depbar = bar;
            pending_register[inst.dst] = bar;
        } else if (inst.opcode == Opcode::sts) {
            if (!store_barrier.has_value()) store_barrier = allocate();
            inst.control.set_depbar = *store_barrier;
        }

        previous = &inst;
        previous_vtime = vtime;
        vtime += 1;
    }
    return program;
}

AssemblyProgram compile_qtx_file(const std::string& path, const Timing& timing) {
    return compile_qtx(read_file(path), timing);
}

std::string emit_asm(const AssemblyProgram& program) {
    std::ostringstream out;
    out << "; scheduled qtx assembly: controls are part of the instruction word\n";
    out << "; {pred, stall, set, wait}; six warp-local depbar counters sb0..sb5\n\n";
    for (const auto& [name, value] : program.constants) {
        out << ".equ " << name << ", " << value << '\n';
    }
    if (!program.constants.empty()) out << '\n';
    for (const auto& inst : program.instructions) {
        out << inst.text << ' ' << format_control(inst.control) << '\n';
    }
    return out.str();
}

Kernel assemble_asm(std::string_view source) {
    const auto program = parse_program(source, true);
    Kernel kernel;
    kernel.words.reserve(program.instructions.size());
    for (const auto& inst : program.instructions) {
        if ((inst.opcode == Opcode::lds || inst.opcode == Opcode::sts) && inst.control.set_depbar >= 6) {
            throw SimError("line " + std::to_string(inst.line) + ": lds/sts requires set=sb0..sb5");
        }
        if (inst.opcode != Opcode::lds && inst.opcode != Opcode::sts && inst.control.set_depbar != 7) {
            throw SimError("line " + std::to_string(inst.line) + ": this opcode cannot set a depbar");
        }
        kernel.words.push_back(encode(inst));
    }
    return kernel;
}

Kernel assemble_asm_file(const std::string& path) {
    return assemble_asm(read_file(path));
}

std::string emit_hex(const Kernel& kernel) {
    std::ostringstream out;
    for (const auto word : kernel.words) {
        out << std::hex << std::setw(16) << std::setfill('0') << word
            << " ; " << format_instruction(decode(word)) << '\n';
    }
    return out.str();
}

Kernel load_hex(std::string_view source) {
    Kernel kernel;
    std::istringstream input{std::string(source)};
    std::string raw;
    std::uint32_t line = 0;
    while (std::getline(input, raw)) {
        ++line;
        const auto comment = raw.find(';');
        const auto clean = trim(raw.substr(0, comment));
        if (clean.empty()) continue;
        std::size_t consumed = 0;
        try {
            const auto word = std::stoull(clean, &consumed, 16);
            if (consumed != clean.size()) throw std::invalid_argument("suffix");
            kernel.words.push_back(word);
        } catch (const std::exception&) {
            throw SimError("line " + std::to_string(line) + ": bad 64-bit hex word '" + clean + "'");
        }
    }
    if (kernel.words.empty()) throw SimError("hex kernel contains no instructions");
    return kernel;
}

Kernel load_hex_file(const std::string& path) {
    return load_hex(read_file(path));
}

Kernel compile_and_assemble(std::string_view source, const Timing& timing) {
    return assemble_asm(emit_asm(compile_qtx(source, timing)));
}

Kernel compile_and_assemble_file(const std::string& path, const Timing& timing) {
    return compile_and_assemble(read_file(path), timing);
}

} // namespace nanosm

// The text parser qtx and assembly share (src/qtxc/): line by line, labels,
// the .equ / .profile directives, the predicate prefix, the control
// block (assembly only) and every instruction's operands, into an
// AssemblyProgram with its branch displacements resolved. It does not
// schedule (schedule.cpp), apply the rules an instruction must pass
// (checks.cpp) or encode (assemble.cpp). read_file, the file reader every
// *_file entry point uses, is here too.

#include "qtxc/internal.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <optional>
#include <regex>
#include <sstream>
#include <unordered_map>

namespace nanosm {
namespace qtxc_internal {

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

void expect_count(const std::vector<std::string>& args, std::size_t count, uint32_t line) {
    if (args.size() != count) {
        throw SimError(
            "line " + std::to_string(line) + ": expected " + std::to_string(count) +
            " operands, got " + std::to_string(args.size()));
    }
}

} // namespace qtxc_internal

namespace {

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

bool is_uniform_register(const std::string& text) {
    static const std::regex pattern(R"(^[uU][rR][0-9]+$)");
    return std::regex_match(text, pattern);
}

// A uniform register, ur0 .. ur(urf.registers - 1).
uint8_t parse_ureg(const std::string& text, uint32_t line, const DeviceProfile& profile) {
    static const std::regex pattern(R"(^[uU][rR]([0-9]+)$)");
    std::smatch match;
    if (!std::regex_match(text, match, pattern)) {
        throw SimError("line " + std::to_string(line) + ": expected uniform register (ur0..ur" +
            std::to_string(profile.urf_registers - 1) + "), got '" + text + "'");
    }
    const auto value = std::stoul(match[1].str());
    if (value >= profile.urf_registers) {
        throw SimError("line " + std::to_string(line) + ": uniform register '" + text +
            "' is outside this profile's " + std::to_string(profile.urf_registers) + " (urf.registers)");
    }
    return static_cast<uint8_t>(value);
}

int32_t parse_value(
    const std::string& text, const std::unordered_map<std::string, int32_t>& constants, uint32_t line);
bool is_register(const std::string& text);

// A tmem operand of acc.ld / acc.st (src/layout.hpp): [urT, S], the pointer
// in a uniform register and the region's words per row, a constant (which
// values are legal is check_tmem_operand's business).
std::pair<uint8_t, int32_t> parse_tmem_operand(
    const std::string& text,
    const std::unordered_map<std::string, int32_t>& constants,
    uint32_t line,
    const DeviceProfile& profile) {
    const auto inner = text.size() >= 2 && text.front() == '[' && text.back() == ']'
        ? split(text.substr(1, text.size() - 2), ',') : std::vector<std::string>{};
    if (inner.size() != 2) {
        throw SimError("line " + std::to_string(line) + ": bad tmem operand '" + text +
            "' (expected [urT, words per row])");
    }
    return {parse_ureg(inner[0], line, profile), parse_value(inner[1], constants, line)};
}

// An mma.async's result operand: [urC], the tmem pointer alone (its rows are
// the shape's n words).
uint8_t parse_mma_result(const std::string& text, uint32_t line, const DeviceProfile& profile) {
    const auto inner = text.size() >= 2 && text.front() == '[' && text.back() == ']'
        ? split(text.substr(1, text.size() - 2), ',') : std::vector<std::string>{};
    if (inner.size() == 2) {
        throw SimError("line " + std::to_string(line) + ": mma.async result '" + text +
            "': its rows are the shape's n words; write [urC] without a row length");
    }
    if (inner.size() != 1 || is_register(inner[0]) || !is_uniform_register(inner[0])) {
        throw SimError("line " + std::to_string(line) + ": bad mma.async result '" + text +
            "' (expected [urC], a uniform register holding its tmem pointer)");
    }
    return parse_ureg(inner[0], line, profile);
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

// "mma.async.m64n64k8", "mma.async.init.m32n64k8", ...: the mma.async
// block's opcode for that shape on this profile (src/isa.hpp: the shape's
// place in the profile's list). nullopt for a mnemonic that is not an
// mma.async; a shape the profile does not list is an error.
std::optional<Opcode> mma_async_opcode(const std::string& op, const DeviceProfile& profile, uint32_t line) {
    static const std::string plain = "mma.async.";
    static const std::string init = "mma.async.init.";
    const bool is_init = op.starts_with(init);
    if (!is_init && !op.starts_with(plain)) return std::nullopt;
    const auto suffix = op.substr(is_init ? init.size() : plain.size());
    const auto shape = parse_mma_shape_name(suffix);
    if (!shape.has_value()) return std::nullopt;
    const auto found = std::find(profile.mma_shapes.begin(), profile.mma_shapes.end(), *shape);
    if (found == profile.mma_shapes.end()) {
        throw SimError("line " + std::to_string(line) + ": " + op + ": shape " + suffix +
            " is not one of this device profile's mma shapes");
    }
    return mma_opcode(static_cast<uint32_t>(found - profile.mma_shapes.begin()), is_init);
}

// setp.xx.TYPE / usetp.xx.TYPE: the type is required, as in PTX. .u32
// compares unsigned 32-bit, .s32 signed. eq and ne are one comparison either
// way, so .s32 on them names the .u32 opcode. Each of the four unsigned
// opcode blocks lists eq, ne, lt, le, gt, ge in that order, each signed block
// lt, le, gt, ge.
Opcode parse_setp_opcode(const std::string& written, bool uniform, bool immediate, uint32_t line) {
    const std::string family = uniform ? "usetp" : "setp";
    const auto where = "line " + std::to_string(line) + ": ";
    static const std::array<std::string, 6> comparisons = {"eq", "ne", "lt", "le", "gt", "ge"};
    const auto parts = split(written, '.');
    const auto found = parts.size() >= 2 ? std::find(comparisons.begin(), comparisons.end(), parts[1])
                                         : comparisons.end();
    if (found == comparisons.end() || parts.size() > 3) {
        throw SimError(where + "unknown " + family + " comparison '" + written + "'");
    }
    const auto untyped = family + "." + parts[1];
    if (parts.size() == 2 || (parts[2] != "u32" && parts[2] != "s32")) {
        throw SimError(where + "'" + written + "' needs its type: " + untyped + ".u32 (unsigned) or " + untyped +
            ".s32 (signed)");
    }
    const auto i = static_cast<uint8_t>(found - comparisons.begin());
    if (parts[2] == "s32" && i >= 2) {
        const auto first = uniform ? (immediate ? Opcode::usetpi_lt_s32 : Opcode::usetp_lt_s32)
                                   : (immediate ? Opcode::setpi_lt_s32 : Opcode::setp_lt_s32);
        return static_cast<Opcode>(static_cast<uint8_t>(first) + i - 2);
    }
    const auto first = uniform ? (immediate ? Opcode::usetpi_eq : Opcode::usetp_eq)
                               : (immediate ? Opcode::setpi_eq : Opcode::setp_eq);
    return static_cast<Opcode>(static_cast<uint8_t>(first) + i);
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

// [urA] or [urA + / - offset]: a uniform register base.
Address parse_uniform_address(
    const std::string& text,
    const std::unordered_map<std::string, int32_t>& constants,
    uint32_t line,
    const DeviceProfile& profile) {
    static const std::regex pattern(
        R"(^\[\s*([A-Za-z]+[0-9]+)\s*(?:([+-])\s*([A-Za-z_][A-Za-z0-9_]*|0[xX][0-9A-Fa-f]+|[0-9]+))?\s*\]$)");
    std::smatch match;
    if (!std::regex_match(text, match, pattern)) {
        throw SimError("line " + std::to_string(line) + ": bad address '" + text + "'");
    }
    Address result;
    result.base = parse_ureg(match[1].str(), line, profile);
    if (match[3].matched) {
        result.offset = parse_value(match[3].str(), constants, line);
        if (match[2].str() == "-") result.offset = -result.offset;
    }
    return result;
}

// An mbar or MMA operand address: [urA], no encoded offset.
uint8_t parse_mbar_address(const std::string& text, uint32_t line, const DeviceProfile& profile) {
    const auto address = parse_uniform_address(text, {}, line, profile);
    if (address.offset != 0) {
        throw SimError(
            "line " + std::to_string(line) +
            ": mbar v0 address must be [uniform register] without an encoded offset");
    }
    return address.base;
}

// An mma.async operand (src/isa.hpp): [urA] or [urA, spacing], the spacing a
// constant in bytes (default 16), a multiple of 16 that fits its 7-bit field.
// Which spacings the profile's layout takes is check_mma_shape's business.
std::pair<uint8_t, uint32_t> parse_mma_operand(
    const std::string& text,
    const std::unordered_map<std::string, int32_t>& constants,
    uint32_t line,
    const DeviceProfile& profile) {
    const auto inner = text.size() >= 2 && text.front() == '[' && text.back() == ']'
        ? split(text.substr(1, text.size() - 2), ',') : std::vector<std::string>{};
    if (inner.empty() || inner.size() > 2) {
        throw SimError("line " + std::to_string(line) + ": bad mma operand '" + text +
            "' (expected [urA] or [urA, row spacing])");
    }
    const auto base = parse_ureg(inner[0], line, profile);
    if (inner.size() == 1) return {base, mma_default_spacing};
    const auto spacing = parse_value(inner[1], constants, line);
    if (spacing <= 0 || spacing % static_cast<int32_t>(mma_spacing_unit) != 0 ||
        static_cast<uint32_t>(spacing) > mma_max_spacing) {
        throw SimError("line " + std::to_string(line) + ": mma operand row spacing " + std::to_string(spacing) +
            " must be a multiple of 16 bytes in 16.." + std::to_string(mma_max_spacing));
    }
    return {base, static_cast<uint32_t>(spacing)};
}

// "tma.load.2d" -> (store false, 2 coordinates); nullopt for anything else.
std::optional<std::pair<bool, uint32_t>> tma_mnemonic(const std::string& op) {
    static const std::regex pattern(R"(^tma\.(load|store)\.([1-4])d$)");
    std::smatch match;
    if (!std::regex_match(op, match, pattern)) return std::nullopt;
    return std::pair<bool, uint32_t>{match[1].str() == "store", static_cast<uint32_t>(match[2].str()[0] - '0')};
}

// A tma's buffer or barrier operand: [urA], no offset.
uint8_t parse_tma_address(const std::string& text, const std::string& what, uint32_t line, const DeviceProfile& profile) {
    const auto address = parse_uniform_address(text, {}, line, profile);
    if (address.offset != 0) {
        throw SimError("line " + std::to_string(line) + ": the tma " + what +
            " is [uniform register] without an offset");
    }
    return address.base;
}

// A tma's tile operand [DESC, urC]: the descriptor's shared address (a
// constant, 64-byte aligned, below 1 MiB: the instruction holds it in 64-byte
// units) and the first coordinate register.
std::pair<int32_t, uint8_t> parse_tma_tile(
    const std::string& text,
    const std::unordered_map<std::string, int32_t>& constants,
    uint32_t line,
    const DeviceProfile& profile) {
    const auto inner = text.size() >= 2 && text.front() == '[' && text.back() == ']'
        ? split(text.substr(1, text.size() - 2), ',') : std::vector<std::string>{};
    if (inner.size() != 2) {
        throw SimError("line " + std::to_string(line) + ": bad tma tile operand '" + text +
            "' (expected [DESC, urC])");
    }
    const auto descriptor = parse_value(inner[0], constants, line);
    if (descriptor < 0 || descriptor % encoding::tma_descriptor_field_unit != 0 ||
        static_cast<uint64_t>(descriptor) / encoding::tma_descriptor_field_unit > encoding::tma_descriptor_field_mask) {
        throw SimError("line " + std::to_string(line) + ": tma descriptor address " + std::to_string(descriptor) +
            " must be 64-byte aligned and below 1 MiB");
    }
    return {descriptor, parse_ureg(inner[1], line, profile)};
}

bool is_register(const std::string& text) {
    static const std::regex pattern(R"(^[rR][0-9]+$)");
    return std::regex_match(text, pattern);
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

} // namespace

namespace qtxc_internal {

AssemblyProgram parse_program(std::string_view source, bool allow_control, const DeviceProfile& profile) {
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
        if (clean.starts_with(".acc") && (clean.size() == 4 || std::isspace(static_cast<unsigned char>(clean[4])))) {
            throw SimError("line " + std::to_string(line_number) +
                ": there is no .acc directive: an accumulator is a tmem pointer in a uniform register "
                "(acc.ld rD, [urT, words per row]; mma.async [urC], [urA, ...], [urB, ...])");
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
            // The per-lane special registers. The warp-uniform ones live on the
            // uniform side: usread, then ur2r for a vector copy.
            expect_count(args, 2, line_number);
            inst.opcode = Opcode::sread;
            inst.dst = parse_reg(args[0], line_number);
            const auto name = upper(args[1]);
            if (name == "LANE_ID") inst.imm = static_cast<int32_t>(SpecialReg::lane_id);
            else if (name == "THREAD_ID") inst.imm = static_cast<int32_t>(SpecialReg::thread_id);
            else if (name == "WARP_ID" || name == "N_WARPS") {
                throw SimError("line " + std::to_string(line_number) + ": " + name +
                    " is warp-uniform: read it with usread into a uniform register (ur2r for a vector copy)");
            } else throw SimError("line " + std::to_string(line_number) + ": unknown special register '" + args[1] + "'");
        } else if (op == "usread") {
            expect_count(args, 2, line_number);
            inst.opcode = Opcode::usread;
            inst.dst = parse_ureg(args[0], line_number, profile);
            const auto name = upper(args[1]);
            if (name == "WARP_ID") inst.imm = static_cast<int32_t>(SpecialReg::warp_id);
            else if (name == "N_WARPS") inst.imm = static_cast<int32_t>(SpecialReg::n_warps);
            else if (name == "LANE_ID" || name == "THREAD_ID") {
                throw SimError("line " + std::to_string(line_number) + ": " + name +
                    " differs across lanes: read it with sread into a vector register");
            } else throw SimError("line " + std::to_string(line_number) + ": unknown special register '" + args[1] + "'");
        } else if (op == "umov") {
            expect_count(args, 2, line_number);
            inst.dst = parse_ureg(args[0], line_number, profile);
            if (is_uniform_register(args[1])) {
                inst.opcode = Opcode::umov;
                inst.src0 = parse_ureg(args[1], line_number, profile);
            } else if (is_register(args[1])) {
                throw SimError("line " + std::to_string(line_number) +
                    ": umov takes a uniform register or an immediate (r2ur moves lane 0 of a vector register)");
            } else {
                inst.opcode = Opcode::umovi;
                inst.imm = parse_value(args[1], constants, line_number);
            }
        } else if (op == "uadd" || op == "usub" || op == "ushl" || op == "ushr" ||
                   op == "uand" || op == "uor" || op == "uxor") {
            expect_count(args, 3, line_number);
            inst.opcode = op == "uadd" ? Opcode::uadd
                : op == "usub" ? Opcode::usub
                : op == "ushl" ? Opcode::ushl
                : op == "ushr" ? Opcode::ushr
                : op == "uand" ? Opcode::uand
                : op == "uor" ? Opcode::uor
                : Opcode::uxor;
            inst.dst = parse_ureg(args[0], line_number, profile);
            inst.src0 = parse_ureg(args[1], line_number, profile);
            inst.src1 = parse_ureg(args[2], line_number, profile);
        } else if (op == "uaddi" || op == "ushli" || op == "ushri" || op == "uandi" ||
                   op == "uori" || op == "uxori") {
            expect_count(args, 3, line_number);
            inst.opcode = op == "uaddi" ? Opcode::uaddi
                : op == "ushli" ? Opcode::ushli
                : op == "ushri" ? Opcode::ushri
                : op == "uandi" ? Opcode::uandi
                : op == "uori" ? Opcode::uori
                : Opcode::uxori;
            inst.dst = parse_ureg(args[0], line_number, profile);
            inst.src0 = parse_ureg(args[1], line_number, profile);
            inst.imm = parse_value(args[2], constants, line_number);
        } else if (op == "r2ur") {
            // Lane 0 of a vector register into a uniform register.
            expect_count(args, 2, line_number);
            inst.opcode = Opcode::r2ur;
            inst.dst = parse_ureg(args[0], line_number, profile);
            inst.src0 = parse_reg(args[1], line_number);
        } else if (op == "ur2r") {
            // A uniform register into every lane of a vector register.
            expect_count(args, 2, line_number);
            inst.opcode = Opcode::ur2r;
            inst.dst = parse_reg(args[0], line_number);
            inst.src0 = parse_ureg(args[1], line_number, profile);
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
        } else if (op == "and" || op == "or" || op == "xor" || op == "shl" || op == "shr") {
            expect_count(args, 3, line_number);
            inst.opcode = op == "and" ? Opcode::and_
                : op == "or" ? Opcode::or_
                : op == "xor" ? Opcode::xor_
                : op == "shl" ? Opcode::shl
                : Opcode::shr;
            inst.dst = parse_reg(args[0], line_number);
            inst.src0 = parse_reg(args[1], line_number);
            inst.src1 = parse_reg(args[2], line_number);
        } else if (op == "fmul" || op == "fsub" || op == "fmax" || op == "cvt.pack.f16x2") {
            expect_count(args, 3, line_number);
            inst.opcode = op == "fmul" ? Opcode::fmul
                : op == "fsub" ? Opcode::fsub
                : op == "fmax" ? Opcode::fmax
                : Opcode::cvt_pack;
            inst.dst = parse_reg(args[0], line_number);
            inst.src0 = parse_reg(args[1], line_number);
            inst.src1 = parse_reg(args[2], line_number);
        } else if (op == "fma") {
            expect_count(args, 4, line_number);
            inst.opcode = Opcode::fma;
            inst.dst = parse_reg(args[0], line_number);
            inst.src0 = parse_reg(args[1], line_number);
            inst.src1 = parse_reg(args[2], line_number);
            inst.src2 = parse_reg(args[3], line_number);
        } else if (op == "ex2" || op == "rcp") {
            expect_count(args, 2, line_number);
            inst.opcode = op == "ex2" ? Opcode::ex2 : Opcode::rcp;
            inst.dst = parse_reg(args[0], line_number);
            inst.src0 = parse_reg(args[1], line_number);
        } else if (op.starts_with("setp.")) {
            expect_count(args, 3, line_number);
            const auto [predicate, invert] = parse_predicate(args[0], line_number);
            if (predicate == 7 || invert) {
                throw SimError("line " + std::to_string(line_number) + ": setp destination must be p0..p6");
            }
            inst.dst = predicate;
            inst.src0 = parse_reg(args[1], line_number);
            const bool immediate = !is_register(args[2]);
            inst.opcode = parse_setp_opcode(op, false, immediate, line_number);
            // The text names the opcode: eq / ne written .s32 reads .u32.
            inst.text = opcode_name(inst.opcode) + clean.substr(op.size());
            if (immediate) inst.imm = parse_value(args[2], constants, line_number);
            else inst.src1 = parse_reg(args[2], line_number);
        } else if (op.starts_with("usetp.")) {
            // The uniform compare: uniform registers (a vector register is
            // refused by parse_ureg), the same comparisons as setp.
            expect_count(args, 3, line_number);
            const auto [predicate, invert] = parse_predicate(args[0], line_number);
            if (predicate == 7 || invert) {
                throw SimError("line " + std::to_string(line_number) + ": usetp destination must be p0..p6");
            }
            inst.dst = predicate;
            inst.src0 = parse_ureg(args[1], line_number, profile);
            const bool immediate = !is_uniform_register(args[2]) && !is_register(args[2]);
            inst.opcode = parse_setp_opcode(op, true, immediate, line_number);
            inst.text = opcode_name(inst.opcode) + clean.substr(op.size());
            if (immediate) inst.imm = parse_value(args[2], constants, line_number);
            else inst.src1 = parse_ureg(args[2], line_number, profile);
        } else if (op == "addi" || op == "shli" || op == "shri" || op == "xori" ||
                   op == "andi" || op == "ori") {
            expect_count(args, 3, line_number);
            inst.opcode = op == "addi" ? Opcode::addi
                : op == "shli" ? Opcode::shli
                : op == "shri" ? Opcode::shri
                : op == "andi" ? Opcode::andi
                : op == "ori" ? Opcode::ori
                : Opcode::xori;
            inst.dst = parse_reg(args[0], line_number);
            inst.src0 = parse_reg(args[1], line_number);
            inst.imm = parse_value(args[2], constants, line_number);
        } else if (op == "lds" || op == "lds.uni" || op == "ldg" || op == "ldg.v8" || op == "ldg.uni") {
            expect_count(args, 2, line_number);
            inst.opcode = op == "lds" ? Opcode::lds
                : op == "lds.uni" ? Opcode::lds_uni
                : op == "ldg" ? Opcode::ldg
                : op == "ldg.uni" ? Opcode::ldg_uni
                : Opcode::ldg_v8;
            inst.dst = parse_reg(args[0], line_number);
            // lds.uni / ldg.uni: the base is a uniform register.
            const bool uniform = op == "lds.uni" || op == "ldg.uni";
            const auto addr = uniform ? parse_uniform_address(args[1], constants, line_number, profile)
                                      : parse_address(args[1], constants, line_number);
            inst.src0 = addr.base;
            inst.imm = addr.offset;
            if (op == "ldg.v8" && static_cast<unsigned>(inst.dst) + 8 > 256)
                throw SimError("line " + std::to_string(line_number) + ": global vector register range wraps");
        } else if (op == "sts" || op == "stg" || op == "stg.v8") {
            expect_count(args, 2, line_number);
            inst.opcode = op == "sts" ? Opcode::sts : op == "stg" ? Opcode::stg : Opcode::stg_v8;
            const auto addr = parse_address(args[0], constants, line_number);
            inst.src0 = addr.base;
            inst.imm = addr.offset;
            inst.src1 = parse_reg(args[1], line_number);
            if (op == "stg.v8" && static_cast<unsigned>(inst.src1) + 8 > 256)
                throw SimError("line " + std::to_string(line_number) + ": global vector register range wraps");
        } else if (op == "atom.add.uni" || op == "atom.and.uni" || op == "atom.xor.uni" || op == "atom.exch.uni") {
            // trash/rfc13.md: uniform shared atomics on signed int32. One word
            // per warp instruction (its address and operand are uniform
            // registers); its old value comes back to every lane's rD (async,
            // through a load slot: a depbar like lds.uni). The offset is a
            // signed 14-bit field.
            expect_count(args, 3, line_number);
            inst.opcode = op == "atom.add.uni" ? Opcode::atom_add_uni
                : op == "atom.and.uni" ? Opcode::atom_and_uni
                : op == "atom.xor.uni" ? Opcode::atom_xor_uni
                : Opcode::atom_exch_uni;
            inst.dst = parse_reg(args[0], line_number);
            const auto addr = parse_uniform_address(args[1], constants, line_number, profile);
            inst.src0 = addr.base;
            inst.imm = addr.offset;
            inst.src1 = parse_ureg(args[2], line_number, profile);
            if (inst.imm < encoding::atomic_immediate_min || inst.imm > encoding::atomic_immediate_max) {
                throw SimError("line " + std::to_string(line_number) + ": atom offset must fit a signed 14-bit immediate");
            }
        } else if (op == "mbar.init") {
            expect_count(args, 2, line_number);
            inst.src0 = parse_mbar_address(args[0], line_number, profile);
            if (is_uniform_register(args[1]) || is_register(args[1])) {
                inst.opcode = Opcode::mbar_init_r;
                inst.src1 = parse_ureg(args[1], line_number, profile);
            } else {
                inst.opcode = Opcode::mbar_init_i;
                inst.imm = parse_value(args[1], constants, line_number);
            }
        } else if (op == "mbar.arrive") {
            expect_count(args, 1, line_number);
            inst.opcode = Opcode::mbar_arrive;
            inst.src0 = parse_mbar_address(args[0], line_number, profile);
        } else if (op == "mbar.arrive.expect_tx") {
            expect_count(args, 2, line_number);
            inst.src0 = parse_mbar_address(args[0], line_number, profile);
            if (is_uniform_register(args[1]) || is_register(args[1])) {
                inst.opcode = Opcode::mbar_arrive_expect_tx_r;
                inst.src1 = parse_ureg(args[1], line_number, profile);
            } else {
                inst.opcode = Opcode::mbar_arrive_expect_tx_i;
                inst.imm = parse_value(args[1], constants, line_number);
            }
        } else if (op == "mbar.test_wait") {
            expect_count(args, 3, line_number);
            const auto [predicate, invert] = parse_predicate(args[0], line_number);
            if (predicate == 7 || invert) {
                throw SimError("line " + std::to_string(line_number) + ": mbar.test_wait destination must be p0..p6");
            }
            inst.dst = predicate;
            inst.src0 = parse_mbar_address(args[1], line_number, profile);
            if (is_uniform_register(args[2]) || is_register(args[2])) {
                inst.opcode = Opcode::mbar_test_wait_r;
                inst.src1 = parse_ureg(args[2], line_number, profile);
            } else {
                inst.opcode = Opcode::mbar_test_wait_i;
                inst.imm = parse_value(args[2], constants, line_number);
            }
        } else if (op == "mbar.try_wait") {
            expect_count(args, 3, line_number);
            const auto [predicate, invert] = parse_predicate(args[0], line_number);
            if (predicate == 7 || invert) {
                throw SimError("line " + std::to_string(line_number) + ": mbar.try_wait destination must be p0..p6");
            }
            inst.dst = predicate;
            inst.src0 = parse_mbar_address(args[1], line_number, profile);
            if (is_uniform_register(args[2]) || is_register(args[2])) {
                inst.opcode = Opcode::mbar_try_wait_r;
                inst.src1 = parse_ureg(args[2], line_number, profile);
            } else {
                inst.opcode = Opcode::mbar_try_wait_i;
                inst.imm = parse_value(args[2], constants, line_number);
            }
        } else if (op == "tma.load" || op == "tma.store") {
            throw SimError("line " + std::to_string(line_number) + ": " + op +
                " needs its coordinate count: " + op + ".1d .. " + op + ".4d");
        } else if (const auto tma = tma_mnemonic(op); tma.has_value()) {
            // tma.load.Nd [urS], [DESC, urC], [urM] / tma.store.Nd [DESC, urC],
            // [urS], [urM] (src/tma.hpp): the buffer, the descriptor (a
            // constant) with the first of N coordinate registers, the barrier.
            expect_count(args, 3, line_number);
            const auto [store, coordinates] = *tma;
            inst.opcode = tma_opcode(store, coordinates);
            const auto& tile = store ? args[0] : args[1];
            const auto& buffer = store ? args[1] : args[0];
            const auto [descriptor, first] = parse_tma_tile(tile, constants, line_number, profile);
            inst.imm = descriptor;
            inst.src1 = first;
            inst.src0 = parse_tma_address(buffer, "buffer", line_number, profile);
            inst.src2 = parse_tma_address(args[2], "barrier", line_number, profile);
            if (static_cast<uint32_t>(first) + coordinates > profile.urf_registers) {
                throw SimError("line " + std::to_string(line_number) + ": the coordinates ur" +
                    std::to_string(first) + " .. ur" + std::to_string(first + coordinates - 1) +
                    " pass this profile's last uniform register, ur" + std::to_string(profile.urf_registers - 1));
            }
        } else if (op == "acc.ld" || op == "acc.ld.v8") {
            // acc.ld rD, [urT, S] (src/layout.hpp): lane l reads row l of the
            // region at the pointer in urT, whose rows are S words; .v8 reads
            // eight columns into rD .. rD + 7.
            expect_count(args, 2, line_number);
            inst.opcode = op == "acc.ld" ? Opcode::acc_ld : Opcode::acc_ld_v8;
            inst.dst = parse_reg(args[0], line_number);
            const auto [pointer, row_words] = parse_tmem_operand(args[1], constants, line_number, profile);
            inst.src0 = pointer;
            inst.imm = row_words;
            const auto width = op == "acc.ld.v8" ? 8u : 1u;
            if (static_cast<unsigned>(inst.dst) + width > 256) {
                throw SimError("line " + std::to_string(line_number) + ": accumulator register range wraps");
            }
        } else if (op == "acc.st" || op == "acc.st.v8") {
            // acc.st [urT, S], rS: the same region, vector data.
            expect_count(args, 2, line_number);
            inst.opcode = op == "acc.st" ? Opcode::acc_st : Opcode::acc_st_v8;
            const auto [pointer, row_words] = parse_tmem_operand(args[0], constants, line_number, profile);
            inst.src0 = pointer;
            inst.imm = row_words;
            inst.src1 = parse_reg(args[1], line_number);
            const auto width = op == "acc.st.v8" ? 8u : 1u;
            if (static_cast<unsigned>(inst.src1) + width > 256) {
                throw SimError("line " + std::to_string(line_number) + ": accumulator register range wraps");
            }
        } else if (const auto mma = mma_async_opcode(op, profile, line_number); mma.has_value()) {
            expect_count(args, 3, line_number);
            // mma.async [urC], [urA, spacing], [urB, spacing] (src/isa.hpp):
            // urC the result's tmem pointer, its rows n words.
            inst.opcode = *mma;
            inst.src2 = parse_mma_result(args[0], line_number, profile);
            const auto [a, a_spacing] = parse_mma_operand(args[1], constants, line_number, profile);
            const auto [b, b_spacing] = parse_mma_operand(args[2], constants, line_number, profile);
            inst.src0 = a;
            inst.src1 = b;
            inst.imm = mma_spacing_immediate(a_spacing, b_spacing);
        } else if (op == "mma.commit") {
            expect_count(args, 1, line_number);
            inst.opcode = Opcode::mma_commit;
            inst.src0 = parse_mbar_address(args[0], line_number, profile);
        } else if (op == "bra" || op == "bra.uni" || op == "ssy") {
            // bra / bra.uni TARGET; @p bra TARGET is the divergent branch.
            // ssy LABEL (trash/divergence-2026-11-21.md): the pc its entry
            // holds, where a sync that pops it sends the warp; any instruction.
            expect_count(args, 1, line_number);
            static const std::regex target_pattern(R"(^[A-Za-z_][A-Za-z0-9_]*$)");
            if (!std::regex_match(args[0], target_pattern)) {
                throw SimError("line " + std::to_string(line_number) + ": bad branch target '" + args[0] + "'");
            }
            inst.opcode = op == "bra" ? Opcode::bra : op == "bra.uni" ? Opcode::bra_uni : Opcode::ssy;
            pending_branches.push_back({program.instructions.size(), args[0], line_number});
        } else if (op == "sync") {
            expect_count(args, 0, line_number);
            inst.opcode = Opcode::sync;
        } else if (op == "fence.sh") {
            // rfc12 section 10: explicit only. qtxc schedules it like any
            // instruction and never inserts one.
            expect_count(args, 0, line_number);
            inst.opcode = Opcode::fence_sh;
        } else if (op == "nop") {
            // The machine instruction that only carries control fields
            // (src/isa.hpp). qtx has no control fields to give it: it says
            // wait, which the scheduler emits as a nop.
            if (!allow_control) {
                throw SimError("line " + std::to_string(line_number) +
                    ": qtx has no nop; wait waits for the warp's outstanding asynchronous work");
            }
            expect_count(args, 0, line_number);
            inst.opcode = Opcode::nop;
        } else if (op == "wait") {
            // qtx: issue nothing further until every outstanding asynchronous
            // operation of this warp has completed. The scheduler gives the
            // nop a wait for every live depbar (schedule.cpp).
            if (allow_control) {
                throw SimError("line " + std::to_string(line_number) +
                    ": wait is qtx; assembly writes nop {wait=sb0|...}");
            }
            expect_count(args, 0, line_number);
            inst.opcode = Opcode::nop;
            inst.text = "nop";
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

std::string read_file(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw SimError("cannot open '" + path + "'");
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

} // namespace qtxc_internal
} // namespace nanosm

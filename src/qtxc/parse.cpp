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
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <optional>
#include <regex>
#include <sstream>
#include <tuple>
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

// qtx's register names (docs/qtx.md "Registers"; registers.cpp). While
// parse_program reads qtx, `names` points at the program being built: a
// register operand is then a name, %x, and what the instruction's field
// holds is the name's place in its file's list -- vector_registers or
// uniform_registers, whichever the operand position asks for -- until
// allocate_registers gives it a number. A name used in both files, and a raw
// r5 / ur5, are refused here. A predicate may be a name too (%odd, in a
// predicate position; %p0 .. %p6 there are p0 .. p6 themselves): its place
// is in predicate_names, and the field holds 8 + place until it has a
// number. Null while assembly is read: registers are numbers there.
struct Names {
    AssemblyProgram* program = nullptr;
    // name -> (kind: 0 vector register, 1 uniform register, 2 predicate; place)
    std::unordered_map<std::string, std::pair<int, uint32_t>> index;
    std::string text; // the instruction being read, for a later message
    bool own = false; // qtxc is naming a register of its own (ld.param's), not reading the writer's
    // Vector names first met as a source of an instruction that takes a
    // uniform register only in another slot (read_source): a later use of
    // one as a uniform register is told where a uniform register may go.
    std::unordered_map<std::string, bool> fixed_vector_source;
};

const char* kind_name(int kind) {
    return kind == 0 ? "vector register" : kind == 1 ? "uniform register" : "predicate";
}

// The message for a name already used as another kind.
SimError kind_conflict(
    const AssemblyProgram& program, const std::string& where, const std::string& name, int here,
    std::pair<int, uint32_t> there) {
    const auto& other = (there.first == 0 ? program.vector_registers
        : there.first == 1 ? program.uniform_registers : program.predicate_names)[there.second];
    const std::string advice = here == 2 || there.first == 2
        ? "a name is a register or a predicate, not both"
        : "a name is one register, in one file (ur2r and r2ur carry a value from one file to the other)";
    return SimError(where + "%" + name + " is used here as a " + kind_name(here) + " and on line " +
        std::to_string(other.line) + " as a " + kind_name(there.first) + " ('" + other.where + "'); " + advice);
}
thread_local Names* names = nullptr;

const char* file_name(bool uniform) { return uniform ? "uniform" : "vector"; }

// Names beginning qtxc_ are the compiler's own: the uniform register an
// ld.param reads through (%qtxc_param) and the copies it inserts
// (%qtxc_copyN, registers.cpp). A writer who used one would share a register
// with the compiler and see its value change, so the name is refused where
// the writer writes it.
bool reserved_name(const std::string& name) { return name.rfind("qtxc_", 0) == 0; }
void refuse_reserved(const std::string& name, const std::string& where) {
    if (!reserved_name(name) || (names != nullptr && names->own)) return;
    throw SimError(where + "%" + name + ": names beginning %qtxc_ are qtxc's own (the register an ld.param reads "
        "through, the copies it inserts); choose another name");
}

uint8_t named_register(const std::string& text, bool uniform, uint32_t line) {
    static const std::regex raw(R"(^[uU]?[rR][0-9]+$)");
    static const std::regex pattern(R"(^%([A-Za-z_][A-Za-z0-9_]*)$)");
    const auto where = "line " + std::to_string(line) + ": ";
    std::smatch match;
    if (std::regex_match(text, raw)) {
        throw SimError(where + "qtx names its registers: write a name such as %x instead of '" + text +
            "' (qtxc picks the number)");
    }
    if (!std::regex_match(text, match, pattern)) {
        throw SimError(where + "expected a " + (uniform ? std::string("uniform ") : std::string()) +
            "register name (%name), got '" + text + "'");
    }
    const auto name = match[1].str();
    if (name.size() == 2 && name[0] == 'p' && name[1] >= '0' && name[1] <= '6') {
        throw SimError(where + "%" + name + " is the predicate " + name + ", not a register");
    }
    refuse_reserved(name, where);
    auto& list = uniform ? names->program->uniform_registers : names->program->vector_registers;
    const int kind = uniform ? 1 : 0;
    if (const auto found = names->index.find(name); found != names->index.end()) {
        if (found->second.first != kind) {
            auto error = kind_conflict(*names->program, where, name, kind, found->second);
            if (kind == 1 && found->second.first == 0 && names->fixed_vector_source.contains(name)) {
                throw SimError(std::string(error.what()) + "; a per-lane instruction takes a uniform register only "
                    "as its second source (fma: as a multiplicand)");
            }
            throw error;
        }
        return static_cast<uint8_t>(found->second.second);
    }
    if (list.size() >= 256) {
        throw SimError(where + "more than 256 " + file_name(uniform) + " register names");
    }
    AssemblyProgram::QtxRegister entry;
    entry.name = name;
    entry.line = line;
    entry.where = names->text;
    list.push_back(std::move(entry));
    names->index[name] = {kind, static_cast<uint32_t>(list.size() - 1)};
    return static_cast<uint8_t>(list.size() - 1);
}

// An instruction that names the first of `count` neighbouring registers
// (ldg.v8 and its kin: eight; a tma's coordinates): in qtx the name must be
// a member of a declared group with that many members from it on.
void require_group(uint8_t place, bool uniform, uint32_t count, uint32_t line, const std::string& what) {
    if (names == nullptr || count < 2) return;
    const auto& program = *names->program;
    const auto& entry = (uniform ? program.uniform_registers : program.vector_registers)[place];
    const std::string directive = uniform ? ".ureg" : ".reg";
    const auto where = "line " + std::to_string(line) + ": " + what + " " + std::to_string(count) +
        " neighbouring registers starting at %" + entry.name;
    if (entry.group < 0) {
        throw SimError(where + ", which is a single register: declare a group, `" + directive + " %" + entry.name +
            "<" + std::to_string(count) + ">`, and name its first member, %" + entry.name + "0");
    }
    const auto& group = program.register_groups[static_cast<std::size_t>(entry.group)];
    if (entry.member + count > group.size) {
        throw SimError(where + ", and `" + directive + " %" + group.stem + "<" + std::to_string(group.size) +
            ">` (line " + std::to_string(group.line) + ") has only " + std::to_string(group.size - entry.member) +
            " from there on");
    }
}

uint8_t parse_reg(const std::string& text, uint32_t line) {
    if (names != nullptr) return named_register(text, false, line);
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

// Where an operand may be a register or an immediate (umov, usetp, an mbar's
// count), is it a register? In qtx any %name is.
bool is_uniform_register(const std::string& text) {
    if (names != nullptr && !text.empty() && text.front() == '%') return true;
    static const std::regex pattern(R"(^[uU][rR][0-9]+$)");
    return std::regex_match(text, pattern);
}

// A uniform register, ur0 .. ur(urf.registers - 1).
uint8_t parse_ureg(const std::string& text, uint32_t line, const DeviceProfile& profile) {
    if (names != nullptr) return named_register(text, true, line);
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

// A register source of an instruction that takes a uniform source (src/isa.hpp
// takes_uniform_source): its file and its number (assembly: `r5` or `ur5`)
// or its place in that file's list (qtx). In qtx a name has the file the rest
// of the kernel gives it -- a write, or a use where only one file can go. A
// name read in a slot where a uniform register may go (`either`) and not met
// before waits for the end of the kernel (kind -1, place_pending_sources); a
// name met first in any other source slot is a vector register, as everywhere.
struct Source {
    std::string text;
    int kind = 0; // 0 vector register, 1 uniform register, -1 not decided yet
    uint8_t place = 0;
};

Source read_source(const std::string& text, bool either, uint32_t line, const DeviceProfile& profile) {
    Source source;
    source.text = text;
    if (names == nullptr) {
        static const std::regex uniform(R"(^[uU][rR][0-9]+$)");
        if (std::regex_match(text, uniform)) {
            source.kind = 1;
            source.place = parse_ureg(text, line, profile);
        } else {
            source.place = parse_reg(text, line);
        }
        return source;
    }
    static const std::regex pattern(R"(^%([A-Za-z_][A-Za-z0-9_]*)$)");
    std::smatch match;
    if (std::regex_match(text, match, pattern)) {
        const auto name = match[1].str();
        // Before the lookup: qtxc's own names are in the index too (ld.param's).
        refuse_reserved(name, "line " + std::to_string(line) + ": ");
        const auto found = names->index.find(name);
        if (found != names->index.end() && found->second.first != 2) {
            source.kind = found->second.first;
            source.place = static_cast<uint8_t>(found->second.second);
            return source;
        }
        const bool predicate = name.size() == 2 && name[0] == 'p' && name[1] >= '0' && name[1] <= '6';
        if (either && found == names->index.end() && !predicate) {
            source.kind = -1;
            return source;
        }
        if (found == names->index.end() && !predicate) names->fixed_vector_source[name] = true;
    }
    source.place = named_register(text, false, line); // refuses what is not a vector register name
    return source;
}

// The sources of an instruction that takes a uniform source, all decided, into
// its fields: at most one uniform register, as the second source -- of fma
// either multiplicand, moved into that slot (a x b commutes; the text follows
// so the assembly reads as the machine does). A uniform first source, a
// uniform addend and two uniform sources are refused. `args` are the
// operands as written, the destination first.
void place_sources(Instruction& inst, const std::vector<Source>& sources, const std::vector<std::string>& args) {
    const auto where = "line " + std::to_string(inst.line) + ": '" + inst.text + "' reads ";
    const auto name = opcode_name(inst.opcode);
    const auto form = name + (is_setp(inst.opcode) ? " p, x, U" : " d, x, U");
    const bool fma = inst.opcode == Opcode::fma;
    std::vector<std::string> uniform;
    for (std::size_t i = 0; i < (fma ? 2u : sources.size()); ++i) {
        if (sources[i].kind == 1) uniform.push_back(sources[i].text);
    }
    const std::string copy = "; ur2r copies a uniform register into a vector register for every other place";
    if (fma && sources[2].kind == 1) {
        throw SimError(where + "the uniform register " + sources[2].text + " as its addend; fma takes a uniform "
            "register only as a multiplicand (fma d, a, U, c or fma d, U, b, c)" + copy);
    }
    if (uniform.size() > 1) {
        throw SimError(where + "two uniform registers, " + uniform[0] + " and " + uniform[1] + "; a per-lane "
            "instruction takes one, as its second source (" + (fma ? std::string("fma: a multiplicand") : form) + ")" +
            copy);
    }
    if (!fma && sources[0].kind == 1) {
        throw SimError(where + "the uniform register " + sources[0].text + " as its first source; " + name +
            " takes a uniform register only as its second source (" + form +
            (inst.opcode == Opcode::fsub ? ": x minus U" : "") + ")" + copy);
    }
    std::size_t a = 0;
    std::size_t b = 1;
    if (fma && sources[0].kind == 1) {
        std::swap(a, b);
        inst.text = "fma " + args[0] + ", " + args[2] + ", " + args[1] + ", " + args[3];
    }
    inst.src0 = sources[a].place;
    inst.src1 = sources[b].place;
    inst.src1_uniform = sources[b].kind == 1;
    if (fma) inst.src2 = sources[2].place;
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
    if (inner.size() != 1 || (names == nullptr && is_register(inner[0])) || !is_uniform_register(inner[0])) {
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
    // qtx: %p0 .. %p6 are p0 .. p6; any other %name is a predicate qtxc
    // places (registers.cpp). The field holds 8 + its place meanwhile.
    static const std::regex name_pattern(R"(^%([A-Za-z_][A-Za-z0-9_]*)$)");
    std::smatch match;
    if (names != nullptr && std::regex_match(value, match, name_pattern)) {
        const auto name = match[1].str();
        const auto where = "line " + std::to_string(line) + ": ";
        if (name.size() == 2 && name[0] == 'p' && name[1] >= '0' && name[1] <= '6') {
            return {static_cast<uint8_t>(name[1] - '0'), invert};
        }
        refuse_reserved(name, where);
        auto& list = names->program->predicate_names;
        if (const auto found = names->index.find(name); found != names->index.end()) {
            if (found->second.first != 2) throw kind_conflict(*names->program, where, name, 2, found->second);
            return {static_cast<uint8_t>(8 + found->second.second), invert};
        }
        if (list.size() >= 240) throw SimError(where + "more than 240 predicate names");
        AssemblyProgram::QtxRegister entry;
        entry.name = name;
        entry.line = line;
        entry.where = names->text;
        list.push_back(std::move(entry));
        names->index[name] = {2, static_cast<uint32_t>(list.size() - 1)};
        return {static_cast<uint8_t>(8 + list.size() - 1), invert};
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

// A constant where an instruction or a directive takes one: a constant
// expression (expressions.cpp) whose value fits the immediate field.
int32_t parse_value(
    const std::string& text,
    const std::unordered_map<std::string, int32_t>& constants,
    uint32_t line) {
    std::string problem;
    const auto value = evaluate_expression(text, constants, problem);
    if (!value.has_value()) {
        throw SimError("line " + std::to_string(line) + ": bad or unknown immediate '" + text + "'" +
            (problem.empty() ? std::string() : " (" + problem + ")"));
    }
    if (*value < encoding::immediate_min || *value > encoding::immediate_max) {
        throw SimError("line " + std::to_string(line) + ": bad or unknown immediate '" + text + "' (its value " +
            std::to_string(*value) + " does not fit the immediate field, " + std::to_string(encoding::immediate_min) +
            " .. " + std::to_string(encoding::immediate_max) + ")");
    }
    return static_cast<int32_t>(*value);
}

// umov's immediate: any 32-bit word (src/isa.hpp, the umovi payload). A name;
// an integer, decimal or 0x, from -2^31 to 2^32 - 1; or a float, written with
// a point or an exponent, or inf, -inf, nan: its IEEE bits. The result is the
// 32 bits (Instruction::imm holds them as written).
int32_t parse_word(
    const std::string& text,
    const std::unordered_map<std::string, int32_t>& constants,
    uint32_t line) {
    const auto bad = [&](const std::string& why) -> SimError {
        return SimError("line " + std::to_string(line) + ": bad or unknown 32-bit immediate '" + text +
            "' (a constant expression from -2147483648 to 4294967295, or a float such as 1.5, -inf)" +
            (why.empty() ? std::string() : ": " + why));
    };
    if (text.empty()) throw bad("");
    // A float literal: digits with a point or an exponent, or inf / nan, with
    // an optional sign. (Hex floats and suffixes are not accepted.) A plain
    // integer is an expression, below.
    static const std::regex float_literal(
        R"(^[+-]?(([0-9]+\.[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?|[0-9]+[eE][+-]?[0-9]+|inf|nan)$)");
    if (std::regex_match(text, float_literal) && !constants.contains(text)) {
        char* end = nullptr;
        const float value = std::strtof(text.c_str(), &end);
        if (end != text.c_str() + text.size()) throw bad("");
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        return static_cast<int32_t>(bits);
    }
    std::string problem;
    const auto value = evaluate_expression(text.front() == '+' ? text.substr(1) : text, constants, problem);
    if (!value.has_value()) throw bad(problem);
    if (*value < -(int64_t{1} << 31) || *value > (int64_t{1} << 32) - 1) throw bad("the value is out of range");
    return static_cast<int32_t>(static_cast<uint32_t>(static_cast<uint64_t>(*value)));
}

struct Address {
    uint8_t base = 0;
    int32_t offset = 0;
};

// "[REG]" or "[REG + / - expression]" -> the register's text and the offset,
// a constant expression (so a named object reads where it is used: lds %a,
// [%byte + A_TILE[1]]). nullopt when the text is not that shape.
std::optional<std::pair<std::string, int32_t>> split_address(
    const std::string& text,
    const std::unordered_map<std::string, int32_t>& constants,
    uint32_t line) {
    if (text.size() < 2 || text.front() != '[' || text.back() != ']') return std::nullopt;
    const auto inner = trim(text.substr(1, text.size() - 2));
    std::size_t end = 0;
    while (end < inner.size() &&
           (std::isalnum(static_cast<unsigned char>(inner[end])) != 0 || inner[end] == '_' || inner[end] == '%')) {
        ++end;
    }
    if (end == 0) return std::nullopt;
    const auto rest = trim(inner.substr(end));
    if (rest.empty()) return std::pair<std::string, int32_t>{inner.substr(0, end), 0};
    if (rest.front() != '+' && rest.front() != '-') return std::nullopt;
    // "0 + rest": the sign in front belongs to the offset.
    return std::pair<std::string, int32_t>{inner.substr(0, end), parse_value("0 " + rest, constants, line)};
}

Address parse_address(
    const std::string& text,
    const std::unordered_map<std::string, int32_t>& constants,
    uint32_t line) {
    const auto parts = split_address(text, constants, line);
    static const std::regex base_pattern(R"(^([rR][0-9]+|%[A-Za-z_][A-Za-z0-9_]*)$)");
    if (!parts.has_value() || !std::regex_match(parts->first, base_pattern)) {
        throw SimError("line " + std::to_string(line) + ": bad address '" + text + "'");
    }
    Address result;
    result.base = parse_reg(parts->first, line);
    result.offset = parts->second;
    return result;
}

// [urA] or [urA + / - offset]: a uniform register base.
Address parse_uniform_address(
    const std::string& text,
    const std::unordered_map<std::string, int32_t>& constants,
    uint32_t line,
    const DeviceProfile& profile) {
    const auto parts = split_address(text, constants, line);
    static const std::regex base_pattern(R"(^([A-Za-z]+[0-9]+|%[A-Za-z_][A-Za-z0-9_]*)$)");
    if (!parts.has_value() || !std::regex_match(parts->first, base_pattern)) {
        throw SimError("line " + std::to_string(line) + ": bad address '" + text + "'");
    }
    Address result;
    result.base = parse_ureg(parts->first, line, profile);
    result.offset = parts->second;
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
    if (names != nullptr && !text.empty() && text.front() == '%') return true;
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
        } else if (name == "read") {
            control.read_depbar = value == "-" ? 7 : parse_bar(value, line);
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
    // The instructions with a source whose file the rest of the kernel
    // decides (read_source), placed after the last line.
    struct PendingSources {
        std::size_t instruction = 0;
        std::vector<Source> sources;
        std::vector<std::string> args;
    };
    std::vector<PendingSources> pending_sources;
    const auto with_uniform_source = [&](Instruction& inst, std::vector<Source> sources,
                                         const std::vector<std::string>& args) {
        const bool decided = std::all_of(sources.begin(), sources.end(), [](const Source& s) { return s.kind >= 0; });
        if (decided) {
            place_sources(inst, sources, args);
        } else {
            pending_sources.push_back({program.instructions.size(), std::move(sources), args});
        }
    };
    std::istringstream input{std::string(source)};
    std::string raw;
    uint32_t line_number = 0;

    // qtx: registers are names (see Names). The pointer is cleared however
    // this function is left.
    Names name_table;
    name_table.program = &program;
    struct NamesScope {
        explicit NamesScope(Names* table) { names = table; }
        ~NamesScope() { names = nullptr; }
    } names_scope(allow_control ? nullptr : &name_table);
    // `.reg %a<8>` / `.ureg %c<2>` (qtx): N neighbouring registers, %a0 ..
    // %a7. Read before the first line, like the sections, so a group may be
    // declared anywhere in the file.
    const auto declare_groups = [&](
        const std::string& tail, bool uniform, uint32_t line,
        const std::unordered_map<std::string, int32_t>& known_constants) {
        // The count is a constant expression (a number, an `.equ` declared above).
        static const std::regex pattern(R"(^%([A-Za-z_][A-Za-z0-9_]*)<(.+)>$)");
        const std::string directive = uniform ? ".ureg" : ".reg";
        const auto where = "line " + std::to_string(line) + ": ";
        if (tail.empty()) throw SimError(where + directive + " needs a group: " + directive + " %name<count>");
        for (const auto& item : operands(tail)) {
            std::smatch match;
            if (!std::regex_match(item, match, pattern)) {
                throw SimError(where + "expected " + directive + " %name<count> (count neighbouring " +
                    file_name(uniform) + " registers, %name0 .. ), got '" + item + "'");
            }
            const auto stem = match[1].str();
            const auto digits = trim(match[2].str());
            refuse_reserved(stem, where + directive + " " + item + ": ");
            std::string problem;
            const auto wide = evaluate_expression(digits, known_constants, problem);
            if (!wide.has_value()) {
                throw SimError(where + directive + " %" + stem + "<" + digits + ">: bad count (" + problem + ")");
            }
            const auto count = *wide < 0 || *wide > 256 ? 0u : static_cast<uint32_t>(*wide);
            auto& list = uniform ? program.uniform_registers : program.vector_registers;
            if (count < 2 || list.size() + count > 256) {
                throw SimError(where + directive + " %" + stem + "<" + digits + ">: a group has 2 or more registers, " +
                    "and a file at most 256 names");
            }
            AssemblyProgram::QtxGroup group;
            group.stem = stem;
            group.uniform = uniform;
            group.first = static_cast<uint32_t>(list.size());
            group.size = count;
            group.line = line;
            for (uint32_t member = 0; member < count; ++member) {
                AssemblyProgram::QtxRegister entry;
                entry.name = stem + std::to_string(member);
                entry.line = line;
                entry.where = directive + " " + item;
                entry.group = static_cast<int32_t>(program.register_groups.size());
                entry.member = member;
                if (name_table.index.contains(entry.name)) {
                    throw SimError(where + directive + " " + item + ": %" + entry.name +
                        " is already a member of another group");
                }
                name_table.index[entry.name] = {uniform ? 1 : 0, static_cast<uint32_t>(list.size())};
                list.push_back(std::move(entry));
            }
            program.register_groups.push_back(std::move(group));
        }
    };

    // The sections first (`.param`, `.mbar`, `.data` lines, docs/qtx.md "Launch"):
    // a section's address depends on the whole list (src/isa.hpp
    // launch_block), and its name is a constant, the address, from the first
    // line on. This pass takes each section's name, kind and size; the lines
    // are read in full below, in order, where their values may use `.equ`
    // names. A `.data` line's size is its value count.
    std::unordered_map<std::string, uint8_t> section_flags; // qtx: name -> its launch flag's depbar bit
    // The launch flags of the sections a text names (their aliases too): an
    // instruction whose operands, or an `.equ` whose expression, mentions a
    // section waits for it. A %register or an object's .property is not a name.
    const auto flags_named = [&](const std::string& text) {
        uint8_t named = 0;
        if (section_flags.empty()) return named;
        static const std::regex word(R"([A-Za-z_][A-Za-z0-9_]*)");
        for (std::sregex_iterator it(text.begin(), text.end(), word), end; it != end; ++it) {
            const auto before = it->position() > 0 ? text[static_cast<std::size_t>(it->position()) - 1] : ' ';
            if (before == '%' || before == '.') continue;
            if (const auto found = section_flags.find(it->str()); found != section_flags.end()) {
                named |= found->second;
            }
        }
        return named;
    };
    // What a constant expression may name beyond the `.equ` constants: the
    // kernel's memory objects and its `.mbar` arrays (expressions.cpp).
    ExpressionNames expression_names;
    expression_names.objects = &program.objects;
    expression_names.sections = &program.sections;
    ExpressionScope expression_scope(&expression_names);
    std::size_t next_section = 0;
    {
        std::istringstream scan{std::string(source)};
        uint32_t scan_line = 0;
        uint32_t last_section_line = 0;
        // The `.equ` lines that need no address (numbers and other such
        // constants) are worked out here too, in order, so that an object's
        // size or count may be a constant declared above it. They are read
        // again, like every `.equ`, in the pass below.
        std::unordered_map<std::string, int32_t> early_constants;
        while (std::getline(scan, raw)) {
            ++scan_line;
            auto clean = trim(raw.substr(0, raw.find(';')));
            KernelSection section;
            if (!allow_control && (is_directive(clean, ".reg") || is_directive(clean, ".ureg"))) {
                const bool uniform = is_directive(clean, ".ureg");
                declare_groups(trim(clean.substr(uniform ? 5 : 4)), uniform, scan_line, early_constants);
                continue;
            }
            if (is_directive(clean, ".equ")) {
                const auto args = operands(trim(clean.substr(4)));
                std::string problem;
                if (args.size() == 2) {
                    if (const auto value = evaluate_expression(args[1], early_constants, problem);
                        value.has_value() && *value >= -(int64_t{1} << 31) && *value < (int64_t{1} << 31)) {
                        early_constants.emplace(args[0], static_cast<int32_t>(*value));
                    }
                }
                continue;
            }
            if (is_directive(clean, ".shmem") || is_directive(clean, ".tmem")) {
                const bool tmem = is_directive(clean, ".tmem");
                parse_object_directive(
                    trim(clean.substr(tmem ? 5 : 6)), tmem, early_constants, scan_line, allow_control, program.objects);
                continue;
            }
            if (is_directive(clean, ".param")) {
                section = parse_param_directive(trim(clean.substr(6)), scan_line, false);
            } else if (is_directive(clean, ".mbar")) {
                // "expect N" is read below, where N may be an .equ name.
                const auto args = operands(trim(clean.substr(5)));
                section.kind = SectionKind::mbar;
                std::tie(section.name, section.count) =
                    mbar_name_and_count(args.empty() ? std::string() : args[0], early_constants, scan_line);
            } else if (is_directive(clean, ".data")) {
                const auto args = operands(trim(clean.substr(5)));
                section.kind = SectionKind::data;
                section.name = section_name(args.empty() ? std::string() : args[0], scan_line, "data");
                section.bytes.assign(args.size() < 2 ? 0 : 4 * (args.size() - 1), 0);
            } else {
                continue;
            }
            for (const auto& earlier : program.sections) {
                if (earlier.name == section.name) {
                    throw SimError("line " + std::to_string(scan_line) + ": duplicate symbol '" + section.name + "'");
                }
            }
            program.sections.push_back(std::move(section));
            last_section_line = scan_line;
        }
        const auto block = launch_block(program.sections, profile.shmem_bytes);
        if (!block.has_value()) {
            throw SimError("line " + std::to_string(last_section_line) +
                ": the launch block (.param, .mbar, .data) does not fit shared memory (" +
                std::to_string(profile.shmem_bytes) + " bytes)");
        }
        for (std::size_t index = 0; index < program.sections.size(); ++index) {
            auto& section = program.sections[index];
            section.address = block->addresses[index];
            constants[section.name] = static_cast<int32_t>(section.address);
        }
        expression_names.addresses_known = true;
        // The kernel's own memory (objects.cpp): below the launch block, from
        // address 0 up; tmem from word 0. An object's name is its address.
        place_objects(
            program.objects, program.sections.empty() ? profile.shmem_bytes : block->start,
            uint64_t{profile.tmem_fine_banks} * profile.tmem_words_per_fine_bank);
        for (const auto& object : program.objects) {
            if (constants.contains(object.name)) {
                throw SimError("line " + std::to_string(object.line) + ": duplicate symbol '" + object.name + "'");
            }
            constants[object.name] = static_cast<int32_t>(object.address);
        }
        // qtx: the launch flags are the compiler's (the policy: src/isa.hpp
        // default_launch_flags); assembly states its own (`.flag`).
        if (!allow_control) {
            program.flags = default_launch_flags(program.sections, *block, profile.core_launch_flags);
            for (const auto& section : program.sections) {
                const auto end = section.address - block->start + section_bytes(section);
                if (const auto bar = launch_flag_covering(program.flags, end); bar.has_value()) {
                    section_flags[section.name] = static_cast<uint8_t>(1u << *bar);
                }
            }
        }
    }

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

        if (is_directive(clean, ".equ")) {
            auto args = operands(trim(clean.substr(4)));
            expect_count(args, 2, line_number);
            // A constant expression; any 32-bit value (where it is used
            // decides whether it fits).
            std::string problem;
            const auto wide = evaluate_expression(args[1], constants, problem);
            if (!wide.has_value() || *wide < -(int64_t{1} << 31) || *wide >= (int64_t{1} << 31)) {
                throw SimError("line " + std::to_string(line_number) + ": bad or unknown immediate '" + args[1] + "'" +
                    (problem.empty() ? std::string(wide.has_value() ? " (out of range)" : "") : " (" + problem + ")"));
            }
            const auto value = static_cast<int32_t>(*wide);
            if (constants.contains(args[0]) || labels.contains(args[0])) {
                throw SimError("line " + std::to_string(line_number) + ": duplicate constant '" + args[0] + "'");
            }
            constants[args[0]] = value;
            program.constants.emplace_back(args[0], value);
            // A constant whose expression names a section (`.equ XX, X`,
            // `.equ SECOND, FULL[1]`, through any number of such lines) is an
            // address in that section: an instruction naming it waits on the
            // same launch flag.
            if (const auto flags = flags_named(args[1]); flags != 0) section_flags[args[0]] = flags;
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
        if (is_directive(clean, ".param")) { // read above, before the first line
            ++next_section;
            continue;
        }
        if (is_directive(clean, ".flag")) {
            if (!allow_control) {
                throw SimError("line " + std::to_string(line_number) +
                    ": qtx has no .flag line (qtxc chooses the launch flags)");
            }
            program.flags.push_back(parse_flag_directive(trim(clean.substr(5)), line_number));
            continue;
        }
        if (!allow_control && (is_directive(clean, ".reg") || is_directive(clean, ".ureg"))) {
            continue; // read above, before the first line
        }
        if (is_directive(clean, ".shmem") || is_directive(clean, ".tmem")) {
            continue; // read and placed above, before the first line
        }
        if (is_directive(clean, ".name")) {
            // Assembly: a qtx register's name and its number, for the reader
            // and for whoever asks the kernel (Kernel::register_names).
            if (!allow_control) {
                throw SimError("line " + std::to_string(line_number) +
                    ": qtx has no .name line (a register is a name there already: %x)");
            }
            program.register_names.push_back(parse_name_directive(trim(clean.substr(5)), line_number));
            continue;
        }
        if (is_directive(clean, ".warps")) {
            if (program.warps != 0) throw SimError("line " + std::to_string(line_number) + ": duplicate .warps");
            program.warps = parse_warps_directive(trim(clean.substr(6)), line_number);
            if (program.warps > profile.core_warps) {
                throw SimError("line " + std::to_string(line_number) + ": .warps " + std::to_string(program.warps) +
                    " exceeds this profile's " + std::to_string(profile.core_warps) + " warp slots (core.warps)");
            }
            continue;
        }
        if (is_directive(clean, ".mbar") || is_directive(clean, ".data")) {
            // Its place was settled above; its values are read here. (Its
            // own name is a constant by now; the directive wants a name.)
            auto& section = program.sections.at(next_section++);
            auto read = is_directive(clean, ".mbar")
                ? parse_mbar_directive(trim(clean.substr(5)), constants, line_number, false)
                : parse_data_directive(trim(clean.substr(5)), constants, line_number, false);
            if (read.name != section.name || section_bytes(read) != section_bytes(section)) {
                throw SimError("internal: line " + std::to_string(line_number) + " is not the section the first pass saw");
            }
            section.expected = read.expected;
            section.bytes = std::move(read.bytes);
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
        name_table.text = clean;
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
        std::string names_text = tail; // what the launch-flag scan below reads
        inst.line = line_number;
        inst.text = clean;
        name_table.text = clean;
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
        // The opcode the mnemonic names (src/isa.hpp; the first where a register
        // and an immediate form share one) and its form, by which the regular
        // instructions are read.
        const auto mnemonic_opcode = opcode_for_mnemonic(op);
        const auto named_form = mnemonic_opcode.has_value() ? opcode_info(*mnemonic_opcode).form : Form::no_operands;

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
                inst.imm = parse_word(args[1], constants, line_number);
            }
        } else if (named_form == Form::uniform_binary) {
            expect_count(args, 3, line_number);
            inst.opcode = *mnemonic_opcode;
            inst.dst = parse_ureg(args[0], line_number, profile);
            inst.src0 = parse_ureg(args[1], line_number, profile);
            inst.src1 = parse_ureg(args[2], line_number, profile);
        } else if (named_form == Form::uniform_binary_imm) {
            expect_count(args, 3, line_number);
            inst.opcode = *mnemonic_opcode;
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
            // mov rD, IMM is the mov instruction. mov rD, rS is written for
            // the reader: it is addi rD, rS, 0, and disassembles as that.
            expect_count(args, 2, line_number);
            inst.dst = parse_reg(args[0], line_number);
            if (is_register(args[1])) {
                inst.opcode = Opcode::addi;
                inst.src0 = parse_reg(args[1], line_number);
                inst.imm = 0;
            } else {
                inst.opcode = Opcode::mov;
                inst.imm = parse_value(args[1], constants, line_number);
            }
        } else if (named_form == Form::vector_binary) {
            expect_count(args, 3, line_number);
            inst.opcode = *mnemonic_opcode;
            inst.dst = parse_reg(args[0], line_number);
            with_uniform_source(inst, {read_source(args[1], false, line_number, profile),
                read_source(args[2], true, line_number, profile)}, args);
        } else if (op == "fma") {
            // fma d, a, b, c (a x b + c): a uniform multiplicand may be
            // written as a or as b; it goes in the b slot (place_sources).
            expect_count(args, 4, line_number);
            inst.opcode = Opcode::fma;
            inst.dst = parse_reg(args[0], line_number);
            with_uniform_source(inst, {read_source(args[1], true, line_number, profile),
                read_source(args[2], true, line_number, profile), read_source(args[3], false, line_number, profile)},
                args);
        } else if (named_form == Form::vector_unary) {
            expect_count(args, 2, line_number);
            inst.opcode = *mnemonic_opcode;
            inst.dst = parse_reg(args[0], line_number);
            inst.src0 = parse_reg(args[1], line_number);
        } else if (op.starts_with("setp.")) {
            expect_count(args, 3, line_number);
            const auto [predicate, invert] = parse_predicate(args[0], line_number);
            if (predicate == 7 || invert) {
                throw SimError("line " + std::to_string(line_number) + ": setp destination must be p0..p6");
            }
            inst.dst = predicate;
            const bool immediate = !is_register(args[2]) && !is_uniform_register(args[2]);
            inst.opcode = parse_setp_opcode(op, false, immediate, line_number);
            // The text names the opcode: eq / ne written .s32 reads .u32.
            inst.text = opcode_name(inst.opcode) + clean.substr(op.size());
            if (immediate) {
                inst.src0 = parse_reg(args[1], line_number);
                inst.imm = parse_value(args[2], constants, line_number);
            } else {
                with_uniform_source(inst, {read_source(args[1], false, line_number, profile),
                    read_source(args[2], true, line_number, profile)}, args);
            }
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
        } else if (named_form == Form::vector_binary_imm) {
            expect_count(args, 3, line_number);
            inst.opcode = *mnemonic_opcode;
            inst.dst = parse_reg(args[0], line_number);
            inst.src0 = parse_reg(args[1], line_number);
            inst.imm = parse_value(args[2], constants, line_number);
        } else if (op == "ld.param.u32" || op == "ld.param.f32") {
            // ld.param.u32 %x, [NAME] (qtx; docs/qtx.md "Launch"): a parameter's
            // value into every lane of a vector register. It is written for
            // the reader and stands for the two instructions a parameter read
            // is: umov of the parameter's shared address into a uniform
            // register qtxc names (%qtxc_param), then lds.uni through it. The
            // lds.uni is the read, so it is the one that waits for the
            // launch's bytes (the parameter's launch flag); the umov only
            // loads a constant and goes ahead.
            if (allow_control) {
                throw SimError("line " + std::to_string(line_number) +
                    ": ld.param is qtx; assembly writes the umov and the lds.uni it stands for");
            }
            expect_count(args, 2, line_number);
            if (prefix_predicate.has_value()) {
                throw SimError("line " + std::to_string(line_number) + ": ld.param cannot be masked (@p)");
            }
            static const std::regex operand(R"(^\[\s*([A-Za-z_][A-Za-z0-9_]*)\s*\]$)");
            std::smatch match;
            const KernelSection* param = nullptr;
            if (std::regex_match(args[1], match, operand)) {
                for (const auto& section : program.sections) {
                    if (section.name == match[1].str() && section.kind == SectionKind::param) param = &section;
                }
            }
            if (param == nullptr) {
                throw SimError("line " + std::to_string(line_number) + ": " + op + " reads a parameter: expected " + op +
                    " %x, [NAME] with NAME a .param of this kernel, got '" + args[1] + "'");
            }
            const bool wants_float = op == "ld.param.f32";
            if (param->type == ParamType::desc || (param->type == ParamType::f32) != wants_float) {
                throw SimError("line " + std::to_string(line_number) + ": " + op + " [" + param->name + "]: " +
                    param->name + " is a .param " + param_type_name(param->type) +
                    (param->type == ParamType::desc
                        ? " (a descriptor is named in a tma instruction's tile operand, not read)"
                        : param->type == ParamType::f32 ? " (read it with ld.param.f32)"
                        : " (read it with ld.param.u32; of a ptr that is its low 32 bits)"));
            }
            Instruction address;
            address.line = line_number;
            address.opcode = Opcode::umovi;
            name_table.own = true;
            address.dst = parse_ureg("%qtxc_param", line_number, profile);
            name_table.own = false;
            address.imm = static_cast<int32_t>(param->address);
            address.text = "umov %qtxc_param, " + param->name;
            program.named_flags.push_back(0);
            program.instructions.push_back(std::move(address));
            inst.opcode = Opcode::lds_uni;
            inst.dst = parse_reg(args[0], line_number);
            name_table.own = true;
            inst.src0 = parse_ureg("%qtxc_param", line_number, profile);
            name_table.own = false;
            inst.imm = 0;
            inst.text = "lds.uni " + args[0] + ", [%qtxc_param]";
            names_text = param->name; // its launch flag, below
        } else if (named_form == Form::load || named_form == Form::uniform_load) {
            expect_count(args, 2, line_number);
            inst.opcode = *mnemonic_opcode;
            inst.dst = parse_reg(args[0], line_number);
            // lds.uni / ldg.uni: the base is a uniform register.
            const auto addr = named_form == Form::uniform_load
                ? parse_uniform_address(args[1], constants, line_number, profile)
                : parse_address(args[1], constants, line_number);
            inst.src0 = addr.base;
            inst.imm = addr.offset;
            if (register_span(inst.opcode) > 1) {
                require_group(inst.dst, false, 8, line_number, op + " writes");
                if (static_cast<unsigned>(inst.dst) + 8 > 256)
                    throw SimError("line " + std::to_string(line_number) + ": global vector register range wraps");
            }
        } else if (named_form == Form::store) {
            expect_count(args, 2, line_number);
            inst.opcode = *mnemonic_opcode;
            const auto addr = parse_address(args[0], constants, line_number);
            inst.src0 = addr.base;
            inst.imm = addr.offset;
            inst.src1 = parse_reg(args[1], line_number);
            if (register_span(inst.opcode) > 1) {
                require_group(inst.src1, false, 8, line_number, op + " stores");
                if (static_cast<unsigned>(inst.src1) + 8 > 256)
                    throw SimError("line " + std::to_string(line_number) + ": global vector register range wraps");
            }
        } else if (named_form == Form::atomic) {
            // Uniform shared atomics on signed int32. One word
            // per warp instruction (its address and operand are uniform
            // registers); its old value comes back to every lane's rD (async,
            // through a load slot: a depbar like lds.uni). The offset is a
            // signed 14-bit field.
            expect_count(args, 3, line_number);
            inst.opcode = *mnemonic_opcode;
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
            // [urS], [urM] (src/tma_descriptor.hpp): the buffer, the descriptor (a
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
            require_group(first, true, coordinates, line_number, op + " reads its coordinates from");
            if (names == nullptr && static_cast<uint32_t>(first) + coordinates > profile.urf_registers) {
                throw SimError("line " + std::to_string(line_number) + ": the coordinates ur" +
                    std::to_string(first) + " .. ur" + std::to_string(first + coordinates - 1) +
                    " pass this profile's last uniform register, ur" + std::to_string(profile.urf_registers - 1));
            }
        } else if (named_form == Form::acc_load) {
            // acc.ld rD, [urT, S] (src/layout.hpp): lane l reads row l of the
            // region at the pointer in urT, whose rows are S words; .v8 reads
            // eight columns into rD .. rD + 7.
            expect_count(args, 2, line_number);
            inst.opcode = *mnemonic_opcode;
            inst.dst = parse_reg(args[0], line_number);
            const auto [pointer, row_words] = parse_tmem_operand(args[1], constants, line_number, profile);
            inst.src0 = pointer;
            inst.imm = row_words;
            const auto width = register_span(inst.opcode);
            require_group(inst.dst, false, width, line_number, "acc.ld.v8 writes");
            if (static_cast<unsigned>(inst.dst) + width > 256) {
                throw SimError("line " + std::to_string(line_number) + ": accumulator register range wraps");
            }
        } else if (named_form == Form::acc_store) {
            // acc.st [urT, S], rS: the same region, vector data.
            expect_count(args, 2, line_number);
            inst.opcode = *mnemonic_opcode;
            const auto [pointer, row_words] = parse_tmem_operand(args[0], constants, line_number, profile);
            inst.src0 = pointer;
            inst.imm = row_words;
            inst.src1 = parse_reg(args[1], line_number);
            const auto width = register_span(inst.opcode);
            require_group(inst.src1, false, width, line_number, "acc.st.v8 stores");
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
        } else if (named_form == Form::branch) {
            // bra / bra.uni TARGET; @p bra TARGET is the divergent branch.
            // ssy LABEL: the pc its entry
            // holds, where a sync that pops it sends the warp; any instruction.
            expect_count(args, 1, line_number);
            static const std::regex target_pattern(R"(^[A-Za-z_][A-Za-z0-9_]*$)");
            if (!std::regex_match(args[0], target_pattern)) {
                throw SimError("line " + std::to_string(line_number) + ": bad branch target '" + args[0] + "'");
            }
            inst.opcode = *mnemonic_opcode;
            pending_branches.push_back({program.instructions.size(), args[0], line_number});
        } else if (op == "sync") {
            expect_count(args, 0, line_number);
            inst.opcode = Opcode::sync;
        } else if (op == "fence.sh") {
            // Explicit only. qtxc schedules it like any
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
        // The sections this instruction's operands name (qtx): their launch
        // flags, for the scheduler.
        const uint8_t named = flags_named(names_text);
        program.named_flags.push_back(named);
        program.instructions.push_back(std::move(inst));
    }
    if (program.instructions.empty()) throw SimError("kernel contains no instructions");
    // A name read where a vector or a uniform register may go has, by now,
    // the file the rest of the kernel gave it -- or none, which is refused.
    for (auto& pending : pending_sources) {
        auto& inst = program.instructions[pending.instruction];
        for (auto& source : pending.sources) {
            if (source.kind >= 0) continue;
            const auto name = source.text.substr(1);
            const auto found = name_table.index.find(name);
            const auto where = "line " + std::to_string(inst.line) + ": ";
            if (found == name_table.index.end()) {
                throw SimError(where + "'" + inst.text + "' reads " + source.text + " where a vector or a uniform "
                    "register may go, and nothing else in the kernel says which it is: a write to it or a use where "
                    "only one of them can go decides (umov makes a uniform register, mov a vector register)");
            }
            if (found->second.first == 2) throw kind_conflict(program, where, name, 0, found->second);
            source.kind = found->second.first;
            source.place = static_cast<uint8_t>(found->second.second);
        }
        place_sources(inst, pending.sources, pending.args);
    }
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

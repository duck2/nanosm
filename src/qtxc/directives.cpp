// What a source declares beside its code (src/qtxc/): `.warps`, and the
// sections of the launch block -- the `.param` signature, `.mbar` and `.data`
// -- with the launch flags over them, `.flag` (src/isa.hpp Kernel; README
// "Launch"). qtx, assembly and the hex text all write these lines, so their
// parsers and formatters are here once; parse.cpp and hex.cpp call them, and
// the rule that needs the whole kernel (the block fits, every section is
// where the block rule puts it, the flags cover it) is check_kernel_image at
// the end.

#include "qtxc/internal.hpp"
#include "mbar.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <regex>
#include <sstream>
#include <tuple>

namespace nanosm {
namespace qtxc_internal {

namespace {

std::string at(uint32_t line) {
    return "line " + std::to_string(line) + ": ";
}

// "word rest" split at the first white space.
std::pair<std::string, std::string> first_word(const std::string& text) {
    const auto end = std::find_if(text.begin(), text.end(), [](unsigned char c) { return std::isspace(c) != 0; });
    return {std::string(text.begin(), end), trim(std::string(end, text.end()))};
}

// A directive's argument: a constant expression (expressions.cpp) from 0 to
// `maximum`.
uint64_t unsigned_value(
    const std::string& text,
    const std::unordered_map<std::string, int32_t>& constants,
    uint64_t maximum,
    uint32_t line,
    const char* what) {
    std::string problem;
    const auto value = evaluate_expression(text, constants, problem);
    if (!value.has_value()) {
        throw SimError(at(line) + "bad or unknown " + what + " '" + text + "'" +
            (problem.empty() ? std::string() : " (" + problem + ")"));
    }
    // A constant holds its 32 bits signed: 0xffffffff written as a name reads -1.
    const auto wide = *value < 0 && maximum == 0xffffffffu ? *value + (int64_t{1} << 32) : *value;
    if (wide < 0 || static_cast<uint64_t>(wide) > maximum) {
        throw SimError(at(line) + what + " '" + text + "' is out of range");
    }
    return static_cast<uint64_t>(wide);
}

void append_u32(std::vector<uint8_t>& bytes, uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8) bytes.push_back(static_cast<uint8_t>(value >> shift));
}

} // namespace

// A section's name: an identifier that is not a register, a predicate or a
// depbar.
std::string section_name(const std::string& name, uint32_t line, const char* what) {
    static const std::regex identifier(R"(^[A-Za-z_][A-Za-z0-9_]*$)");
    static const std::regex reserved(R"(^([rR][0-9]+|[uU][rR][0-9]+|p[0-6]|PT|sb[0-5])$)");
    if (!std::regex_match(name, identifier) || std::regex_match(name, reserved)) {
        throw SimError(at(line) + "bad " + what + " name '" + name +
            "' (a section is named; its address comes from the launch block)");
    }
    return name;
}

// An mbar's name with its optional element count: "NAME" (1) or "NAME[C]".
std::pair<std::string, uint32_t> mbar_name_and_count(
    const std::string& text, const std::unordered_map<std::string, int32_t>& constants, uint32_t line) {
    // NAME[COUNT]: the count is a constant expression (a number, an `.equ`
    // declared above the line), as a .shmem array's is.
    static const std::regex array(R"(^([^\[\]\s]+)\s*\[(.+)\]$)");
    std::smatch match;
    if (!std::regex_match(text, match, array)) return {section_name(text, line, "mbar"), 1};
    const auto name = section_name(match[1].str(), line, "mbar");
    const auto written = trim(match[2].str());
    std::string problem;
    const auto count = evaluate_expression(written, constants, problem);
    if (!count.has_value()) {
        throw SimError(at(line) + ".mbar " + name + "[" + written + "]: bad barrier count (" + problem + ")");
    }
    if (*count < 1 || *count > max_mbar_array) {
        throw SimError(at(line) + ".mbar " + name + "[" + written + "]: an array has 1.." +
            std::to_string(max_mbar_array) + " barriers");
    }
    return {name, static_cast<uint32_t>(*count)};
}

bool is_directive(const std::string& clean, std::string_view name) {
    return clean.starts_with(name) &&
        (clean.size() == name.size() || std::isspace(static_cast<unsigned char>(clean[name.size()])) != 0);
}

// `.warps N`
uint32_t parse_warps_directive(const std::string& tail, uint32_t line) {
    const auto value = unsigned_value(tail, {}, 0xffff, line, "warp count");
    if (value == 0) throw SimError(at(line) + ".warps must be at least 1");
    return static_cast<uint32_t>(value);
}

// `.param TYPE NAME` (qtx, assembly); the hex text writes the address the
// assembler gave it too: `.param TYPE NAME, ADDRESS` (with_address).
KernelSection parse_param_directive(const std::string& tail, uint32_t line, bool with_address) {
    const auto args = operands(tail);
    if (args.size() != (with_address ? 2u : 1u)) {
        throw SimError(at(line) + (with_address
            ? "expected .param TYPE NAME, ADDRESS"
            : "expected .param TYPE NAME (ptr, u32, f32 or desc)"));
    }
    const auto [type_name, name] = first_word(args[0]);
    KernelSection param;
    param.kind = SectionKind::param;
    if (type_name == "ptr") param.type = ParamType::ptr;
    else if (type_name == "u32") param.type = ParamType::u32;
    else if (type_name == "f32") param.type = ParamType::f32;
    else if (type_name == "desc") param.type = ParamType::desc;
    else {
        throw SimError(at(line) + "unknown parameter type '" + type_name +
            "' (ptr: a global address, 8 bytes; u32; f32; desc: a TMA descriptor, 64 bytes)");
    }
    param.name = section_name(name, line, "parameter");
    if (with_address) param.address = static_cast<uint32_t>(unsigned_value(args[1], {}, 0xffffffffu, line, "address"));
    return param;
}

// `.mbar NAME, expect N`: a barrier in phase 0 that expects N arrivals; NAME
// becomes its address in the launch block. `.mbar NAME[C], expect N`: C such
// barriers side by side, one section, element i at NAME + 8 i. The hex text:
// `.mbar NAME, ADDRESS, expect N` (NAME[C] for an array).
KernelSection parse_mbar_directive(
    const std::string& tail, const std::unordered_map<std::string, int32_t>& constants, uint32_t line,
    bool with_address) {
    const auto args = operands(tail);
    const std::size_t count = with_address ? 3 : 2;
    const auto last = args.size() == count ? first_word(args.back()) : std::pair<std::string, std::string>{};
    if (args.size() != count || last.first != "expect" || last.second.empty()) {
        throw SimError(at(line) + (with_address
            ? "expected .mbar NAME, ADDRESS, expect N"
            : "expected .mbar NAME, expect N or .mbar NAME[COUNT], expect N (the name is the first "
              "barrier's address, given by the launch block)"));
    }
    KernelSection mbar;
    mbar.kind = SectionKind::mbar;
    std::tie(mbar.name, mbar.count) = mbar_name_and_count(args[0], constants, line);
    if (with_address) mbar.address = static_cast<uint32_t>(unsigned_value(args[1], {}, 0xffffffffu, line, "address"));
    const auto expected = unsigned_value(last.second, constants, mbar_encoding::max_count, line, "expected count");
    if (expected == 0) throw SimError(at(line) + "an mbar's expected count must be 1..32767");
    mbar.expected = static_cast<uint16_t>(expected);
    return mbar;
}

// `.data NAME, f32 V, V, ...` or `.data NAME, u32 V, V, ...`: words of the
// launch block, little-endian; NAME becomes the first one's address. f32
// values are decimal numbers, inf, -inf or nan; u32 values are numbers or
// .equ names. The hex text: `.data NAME, ADDRESS, u32 ...`.
KernelSection parse_data_directive(
    const std::string& tail, const std::unordered_map<std::string, int32_t>& constants, uint32_t line,
    bool with_address) {
    auto args = operands(tail);
    const std::size_t first_value = with_address ? 2 : 1;
    if (args.size() < first_value + 1) {
        throw SimError(at(line) + (with_address
            ? "expected .data NAME, ADDRESS, u32 VALUE, ..."
            : "expected .data NAME, f32|u32 VALUE, ... (the name is the data's address, given by the launch block)"));
    }
    KernelSection data;
    data.kind = SectionKind::data;
    data.name = section_name(args[0], line, "data");
    if (with_address) data.address = static_cast<uint32_t>(unsigned_value(args[1], {}, 0xffffffffu, line, "address"));
    const auto [type, first] = first_word(args[first_value]);
    if (type != "f32" && type != "u32") {
        throw SimError(at(line) + "a .data line names its element type: f32 or u32");
    }
    args[first_value] = first;
    for (std::size_t index = first_value; index < args.size(); ++index) {
        const auto& text = args[index];
        if (text.empty()) throw SimError(at(line) + "empty .data value");
        if (type == "u32") {
            append_u32(data.bytes, static_cast<uint32_t>(unsigned_value(text, constants, 0xffffffffu, line, "u32 value")));
            continue;
        }
        char* end = nullptr;
        const float value = std::strtof(text.c_str(), &end);
        if (end != text.c_str() + text.size()) throw SimError(at(line) + "bad f32 value '" + text + "'");
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        append_u32(data.bytes, bits);
    }
    return data;
}

// `.flag sbN, BYTES` (assembly, hex): depbar sbN is a launch flag, cleared
// when the first BYTES bytes of the launch block have landed. qtxc writes
// these lines (src/isa.hpp default_launch_flags); qtx has no such line.
LaunchFlag parse_flag_directive(const std::string& tail, uint32_t line) {
    const auto args = operands(tail);
    static const std::regex bar(R"(^sb([0-5])$)");
    std::smatch match;
    if (args.size() != 2 || !std::regex_match(args[0], match, bar)) {
        throw SimError(at(line) + "expected .flag sb0..sb5, BYTES");
    }
    LaunchFlag flag;
    flag.bar = static_cast<uint8_t>(match[1].str()[0] - '0');
    flag.bytes = static_cast<uint32_t>(unsigned_value(args[1], {}, 0xffffffffu, line, "byte count"));
    return flag;
}

std::string format_section_directive(const KernelSection& section, bool with_address) {
    const auto address = with_address ? ", " + std::to_string(section.address) : std::string();
    switch (section.kind) {
    case SectionKind::param:
        return std::string(".param ") + param_type_name(section.type) + ' ' + section.name + address;
    case SectionKind::mbar:
        return ".mbar " + section.name + (section.count == 1 ? "" : "[" + std::to_string(section.count) + "]") +
            address + ", expect " + std::to_string(section.expected);
    case SectionKind::data: {
        // Always as u32 words in hex: the bits, whatever type the source wrote.
        std::ostringstream out;
        out << ".data " << section.name << address << ", u32";
        for (std::size_t offset = 0; offset + 4 <= section.bytes.size(); offset += 4) {
            uint32_t word = 0;
            std::memcpy(&word, section.bytes.data() + offset, sizeof(word));
            out << (offset == 0 ? " " : ", ") << "0x" << std::hex << std::setw(8) << std::setfill('0') << word
                << std::dec;
        }
        return out.str();
    }
    }
    return {};
}

// `.name %x, r5` / `.name %blk, ur6` (assembly, hex): the name a qtx gave a
// register and the number qtxc gave it (registers.cpp). It changes nothing
// about the code; a reader, a debugger or a test asks the kernel for it
// (Kernel::register_names).
RegisterName parse_name_directive(const std::string& tail, uint32_t line) {
    const auto args = operands(tail);
    static const std::regex name_pattern(R"(^%([A-Za-z_][A-Za-z0-9_]*)$)");
    static const std::regex number_pattern(R"(^(ur|r|p)([0-9]+)$)");
    std::smatch name_match, number_match;
    if (args.size() != 2 || !std::regex_match(args[0], name_match, name_pattern) ||
        !std::regex_match(args[1], number_match, number_pattern) || number_match[2].str().size() > 3 ||
        std::stoul(number_match[2].str()) > 255) {
        throw SimError("line " + std::to_string(line) + ": expected .name %name, rN (or urN, or pN)");
    }
    return {name_match[1].str(), number_match[1].str() == "ur",
        static_cast<uint8_t>(std::stoul(number_match[2].str())), number_match[1].str() == "p"};
}

std::string format_name_directive(const RegisterName& entry) {
    return ".name %" + entry.name + ", " + (entry.predicate ? "p" : entry.uniform ? "ur" : "r") +
        std::to_string(entry.number);
}

std::string format_flag_directive(const LaunchFlag& flag) {
    return ".flag sb" + std::to_string(flag.bar) + ", " + std::to_string(flag.bytes);
}

} // namespace qtxc_internal

// Kernel load and assembly: the kernel's sections are whole, named once, fit
// this profile's shared memory and sit where the block rule puts them
// (src/isa.hpp launch_block), so a kernel cannot carry its own idea of where
// the launch's bytes land; its launch flags are distinct depbars, no more
// than the profile reserves (core.launch_flags), and together cover the
// block.
void check_kernel_image(const Kernel& kernel, const DeviceProfile& profile) {
    if (kernel.warps > profile.core_warps) {
        throw SimError("kernel is written for " + std::to_string(kernel.warps) + " warps (.warps); the SM has " +
            std::to_string(profile.core_warps) + " warp slots (core.warps)");
    }
    if (kernel.sections.empty()) {
        if (!kernel.flags.empty()) throw SimError("kernel has launch flags (.flag) and nothing to launch");
        return;
    }
    for (const auto& section : kernel.sections) {
        if (std::count_if(kernel.sections.begin(), kernel.sections.end(),
                [&](const KernelSection& other) { return other.name == section.name; }) != 1) {
            throw SimError("kernel declares '" + section.name + "' twice");
        }
        if (section.kind == SectionKind::data && (section.bytes.empty() || section.bytes.size() % 4 != 0)) {
            throw SimError(".data " + section.name + " is not whole 4-byte words");
        }
        if (section.kind == SectionKind::mbar &&
            (section.expected == 0 || section.expected > mbar_encoding::max_count)) {
            throw SimError(".mbar " + section.name + " must expect 1..32767 arrivals");
        }
        if (section.count == 0 || section.count > max_mbar_array ||
            (section.kind != SectionKind::mbar && section.count != 1)) {
            throw SimError("kernel section '" + section.name + "' has a bad element count (" +
                std::to_string(section.count) + "); only an .mbar is an array, of 1.." +
                std::to_string(max_mbar_array));
        }
    }
    const auto block = launch_block(kernel.sections, profile.shmem_bytes);
    if (!block.has_value()) {
        throw SimError("the kernel's launch block (.param, .mbar, .data) does not fit shared memory (" +
            std::to_string(profile.shmem_bytes) + " bytes)");
    }
    for (std::size_t index = 0; index < kernel.sections.size(); ++index) {
        if (kernel.sections[index].address != block->addresses[index]) {
            throw SimError("kernel section '" + kernel.sections[index].name + "' is at " +
                std::to_string(kernel.sections[index].address) + "; the launch block puts it at " +
                std::to_string(block->addresses[index]) + " on this profile (rebuild the kernel)");
        }
    }
    if (kernel.flags.size() > profile.core_launch_flags) {
        throw SimError("kernel has " + std::to_string(kernel.flags.size()) + " launch flags; this profile reserves " +
            std::to_string(profile.core_launch_flags) + " (core.launch_flags; rebuild the kernel)");
    }
    if (profile.tma_queue_commands == 0) {
        throw SimError("this profile has no TMA to copy the kernel's launch block (.param, .mbar, .data) "
            "into shared memory; it runs only kernels that declare none");
    }
    bool covered = false;
    for (std::size_t index = 0; index < kernel.flags.size(); ++index) {
        const auto& flag = kernel.flags[index];
        if (flag.bar >= 6 || flag.bytes == 0 || flag.bytes > block->bytes) {
            throw SimError("launch flag sb" + std::to_string(flag.bar) + ", " + std::to_string(flag.bytes) +
                " is not a depbar and a byte count inside the launch block (" + std::to_string(block->bytes) +
                " bytes)");
        }
        for (std::size_t other = 0; other < index; ++other) {
            if (kernel.flags[other].bar == flag.bar) {
                throw SimError("launch flag sb" + std::to_string(flag.bar) + " is given twice");
            }
        }
        covered = covered || flag.bytes == block->bytes;
    }
    if (!covered) {
        throw SimError("the kernel's launch flags (.flag) do not cover its launch block: one must have the "
            "block's " + std::to_string(block->bytes) + " bytes, or nothing could wait for the last section");
    }
}

} // namespace nanosm

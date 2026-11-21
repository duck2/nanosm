// The hex kernel text (src/qtxc/): emit_hex writes a Kernel as a .profile /
// .registers header, what the kernel declares beside its code (.warps, its
// sections .param / .mbar / .data each with its address, its launch flags
// .flag: directives.cpp) and one 64-bit word per line; load_hex reads it back and runs validate_kernel
// (checks.cpp). The qtx's symbols are not in it. The encoding of a word is
// src/isa.cpp's; this is only the text around the words.

#include "qtxc/internal.hpp"

#include <iomanip>
#include <sstream>

namespace nanosm {

std::string emit_hex(const Kernel& kernel, const std::vector<MmaShape>* mma_shapes) {
    std::ostringstream out;
    out << ".profile " << kernel.profile_name << ", 0x"
        << format_profile_fingerprint(kernel.profile_fingerprint) << '\n';
    out << ".registers " << kernel.register_count << '\n';
    if (kernel.warps != 0) out << ".warps " << kernel.warps << '\n';
    for (const auto& section : kernel.sections) out << format_section_directive(section, true) << '\n';
    for (const auto& flag : kernel.flags) out << format_flag_directive(flag) << '\n';
    for (const auto word : kernel.words) {
        out << std::hex << std::setw(16) << std::setfill('0') << word
            << " ; " << format_instruction(decode(word), true, mma_shapes) << '\n';
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
        if (is_directive(clean, ".warps")) {
            if (kernel.warps != 0) throw SimError("line " + std::to_string(line) + ": duplicate .warps");
            kernel.warps = parse_warps_directive(trim(clean.substr(6)), line);
            continue;
        }
        if (is_directive(clean, ".param")) {
            kernel.sections.push_back(parse_param_directive(trim(clean.substr(6)), line, true));
            continue;
        }
        if (is_directive(clean, ".mbar")) {
            kernel.sections.push_back(parse_mbar_directive(trim(clean.substr(5)), {}, line, true));
            continue;
        }
        if (is_directive(clean, ".data")) {
            kernel.sections.push_back(parse_data_directive(trim(clean.substr(5)), {}, line, true));
            continue;
        }
        if (is_directive(clean, ".flag")) {
            kernel.flags.push_back(parse_flag_directive(trim(clean.substr(5)), line));
            continue;
        }
        if (clean.starts_with(".acc")) {
            throw SimError("line " + std::to_string(line) +
                ": a hex kernel has no .acc header since 2026-11-19 (an accumulator is a tmem pointer in a "
                "uniform register); rebuild it");
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
    if (!saw_register_count) throw SimError("hex kernel has no .registers header");
    validate_kernel(kernel);
    return kernel;
}

Kernel load_hex_file(const std::string& path) {
    return load_hex(read_file(path));
}

} // namespace nanosm

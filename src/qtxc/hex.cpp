// The hex kernel text (src/qtxc/): emit_hex writes a Kernel as a .profile /
// .registers / .acc header and one 64-bit word per line; load_hex reads it
// back and runs validate_kernel (checks.cpp). The encoding of a word is
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
    for (const auto& acc : kernel.accumulators) {
        out << ".acc " << acc.name << ", " << acc.base_word << ", " << acc.rows << ", " << acc.cols << '\n';
    }
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
        if (clean.starts_with(".acc")) {
            const auto args = operands(trim(clean.substr(4)));
            expect_count(args, 4, line);
            Kernel::Accumulator acc;
            acc.name = args[0];
            try {
                const auto base = std::stoul(args[1], nullptr, 0);
                const auto rows = std::stoul(args[2], nullptr, 0);
                const auto cols = std::stoul(args[3], nullptr, 0);
                if (rows == 0 || rows > 65535 || cols == 0 || cols > 65535 ||
                    kernel.accumulators.size() >= 256) throw std::invalid_argument("range");
                acc.base_word = static_cast<uint32_t>(base);
                acc.rows = static_cast<uint16_t>(rows);
                acc.cols = static_cast<uint16_t>(cols);
            } catch (const std::exception&) {
                throw SimError("line " + std::to_string(line) + ": bad accumulator metadata");
            }
            kernel.accumulators.push_back(std::move(acc));
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
    if (!saw_register_count) throw SimError("hex kernel has no .registers header");
    validate_kernel(kernel);
    return kernel;
}

Kernel load_hex_file(const std::string& path) {
    return load_hex(read_file(path));
}

} // namespace nanosm

#pragma once

#include "isa.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace nanosm {

class SimError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct AssemblyProgram {
    std::vector<std::pair<std::string, std::int32_t>> constants;
    std::vector<Instruction> instructions;
};

AssemblyProgram compile_qtx(std::string_view source, const Timing& timing = {});
AssemblyProgram compile_qtx_file(const std::string& path, const Timing& timing = {});
std::string emit_asm(const AssemblyProgram& program);

Kernel assemble_asm(std::string_view source);
Kernel assemble_asm_file(const std::string& path);
std::string emit_hex(const Kernel& kernel);
Kernel load_hex(std::string_view source);
Kernel load_hex_file(const std::string& path);

Kernel compile_and_assemble(std::string_view source, const Timing& timing = {});
Kernel compile_and_assemble_file(const std::string& path, const Timing& timing = {});

} // namespace nanosm

#pragma once

#include "isa.hpp"
#include "profile.hpp"

#include <cstddef>
#include <stdint.h>
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

struct AssemblyLabel {
    std::string name;
    std::size_t instruction_index = 0;
};

struct AssemblyProgram {
    std::string profile_name;
    uint64_t profile_fingerprint = 0;
    uint16_t register_count = 0;
    std::vector<std::pair<std::string, int32_t>> constants;
    std::vector<AssemblyLabel> labels;
    std::vector<Instruction> instructions;
};

AssemblyProgram compile_qtx(std::string_view source, const DeviceProfile& profile);
AssemblyProgram compile_qtx_file(const std::string& path, const DeviceProfile& profile);
std::string emit_asm(const AssemblyProgram& program);

Kernel assemble_asm(std::string_view source, const DeviceProfile& profile);
Kernel assemble_asm_file(const std::string& path, const DeviceProfile& profile);
std::string emit_hex(const Kernel& kernel);
Kernel load_hex(std::string_view source);
Kernel load_hex_file(const std::string& path);

Kernel compile_and_assemble(std::string_view source, const DeviceProfile& profile);
Kernel compile_and_assemble_file(const std::string& path, const DeviceProfile& profile);

} // namespace nanosm

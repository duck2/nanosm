#pragma once

// The toolchain's public interface (src/qtxc/): qtx -> scheduled assembly
// (compile_qtx), assembly -> Kernel (assemble_asm), the hex text both ways,
// and the checks a kernel passes at load. The simulator includes it for those
// checks and for SimError, which lives here for now; src/compiler.hpp is a
// one-line include of this file. How each step works is in the .cpp files
// next to it; internal.hpp is theirs, not part of this interface.

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
// The hex text; each word's comment names its instruction, an mma.async by
// its shape when the profile's shape list is given.
std::string emit_hex(const Kernel& kernel, const std::vector<MmaShape>* mma_shapes = nullptr);
Kernel load_hex(std::string_view source);
Kernel load_hex_file(const std::string& path);
void validate_kernel(const Kernel& kernel);
// Kernel load: every instruction's distinct sources fit its RF banks' read
// ports in one round (src/rf.hpp), else a SimError naming the pc, the
// instruction and the registers. qtxc and the assembler apply the same rule
// per source line.
void check_rf_read_rounds(const Kernel& kernel, const DeviceProfile& profile);
// Kernel load: every mma.async names a shape the profile lists (with legal
// row spacings), and every acc.ld / acc.st a legal words per row (qtxc and
// the assembler apply the same rules per source line).
void check_mma_shapes(const Kernel& kernel, const DeviceProfile& profile);
// Kernel load: every uniform register an instruction names (src/isa.hpp
// uniform_sources -- a tma's last coordinate register included -- and
// uniform_destination) is below the profile's urf.registers, else a SimError
// naming the pc and the instruction (qtxc and the assembler refuse it per
// source line).
void check_uniform_registers(const Kernel& kernel, const DeviceProfile& profile);

Kernel compile_and_assemble(std::string_view source, const DeviceProfile& profile);
Kernel compile_and_assemble_file(const std::string& path, const DeviceProfile& profile);

// Hand assembly: raise each stall to what the kernel's fixed-latency register
// and predicate dependencies need under `profile` (the distances qtxc
// schedules with, src/rf.hpp); depbars and waits stay the author's.
Kernel schedule_fixed_stalls(Kernel kernel, const DeviceProfile& profile);

} // namespace nanosm

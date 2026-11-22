#pragma once

// The toolchain's public interface (src/qtxc/): qtx -> scheduled assembly
// (compile_qtx), assembly -> Kernel (assemble_asm), the hex text both ways,
// and the checks a kernel passes at load. The simulator includes it for those
// checks and for SimError, which lives here for now. How each step works is in the .cpp files
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
    // What the source declares beside the code (src/isa.hpp Kernel): `.warps`,
    // the sections of the launch block in declaration order (`.param`,
    // `.mbar`, `.data`, each with its shared-memory address) and the launch
    // flags over them. Assembly text carries the same directives and `.flag`.
    uint32_t warps = 0;
    std::vector<KernelSection> sections;
    std::vector<LaunchFlag> flags;
    // qtx only, one per instruction: the launch flags (a depbar mask) of the
    // sections the instruction's text names. The scheduler turns them into
    // waits (schedule.cpp); assembly's waits are the author's.
    std::vector<uint8_t> named_flags;
    // qtx's registers (registers.cpp). A qtx register is a name, %x; the
    // parser gives each a place in its file's list (vector or uniform) and
    // the instructions hold those places until allocate_registers turns
    // them into numbers. `group`: the `.reg %a<8>` / `.ureg %c<2>` it is a
    // member of (neighbouring registers), or none. `scratch`: one qtxc made
    // for a copy.
    struct QtxRegister {
        std::string name; // without the %
        uint32_t line = 0;
        std::string where; // the instruction or directive that first named it
        int32_t group = -1;
        uint32_t member = 0;
        bool scratch = false;
    };
    struct QtxGroup {
        std::string stem;
        bool uniform = false;
        uint32_t first = 0; // the place of its member 0 in its file's list
        uint32_t size = 0;
        uint32_t line = 0;
    };
    std::vector<QtxRegister> vector_registers;
    std::vector<QtxRegister> uniform_registers;
    // Predicates with a name (%odd): an instruction's predicate field holds
    // 8 + the name's place here until allocation; p0 .. p6 (and %p0 .. %p6)
    // are themselves.
    std::vector<QtxRegister> predicate_names;
    std::vector<QtxGroup> register_groups;
    // After allocation (qtx), or as an assembly's `.name` lines say: each
    // name and its number. `notes`: one per instruction, a comment the
    // assembly carries (a copy qtxc inserted); `warnings`: what the compiler
    // has to tell the author; `copies`: the copies it inserted.
    std::vector<RegisterName> register_names;
    std::vector<std::string> notes;
    std::vector<std::string> warnings;
    uint32_t copies = 0;
    // The kernel's own memory by name (`.shmem` / `.tmem`, objects.cpp), in
    // declaration order, each with its address.
    std::vector<MemoryObject> objects;
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
// Kernel load: a write mask (@p / @!p) only on an instruction that can be
// masked (src/isa.hpp can_be_masked; bra and bra.uni have their own
// predicate), else a SimError naming the pc (qtxc and the assembler refuse it
// per source line).
void check_write_masks(const Kernel& kernel, const DeviceProfile& profile);
// Kernel load: an ldg.v8 / stg.v8 only on a profile with 32-byte sectors (a
// lane's eight words are one sector), else a SimError naming the pc (qtxc
// and the assembler refuse it per source line).
void check_v8_sectors(const Kernel& kernel, const DeviceProfile& profile);
// Kernel load: every uniform register an instruction names (src/isa.hpp
// uniform_sources -- a tma's last coordinate register included -- and
// uniform_destination) is below the profile's urf.registers, else a SimError
// naming the pc and the instruction (qtxc and the assembler refuse it per
// source line).
void check_uniform_registers(const Kernel& kernel, const DeviceProfile& profile);

// Kernel load and assembly: what the kernel declares beside its code
// (`.warps`, its sections `.param` / `.mbar` / `.data`, its launch flags)
// fits this profile: the launch block is inside shared memory, every section
// sits where the block rule puts it (src/isa.hpp launch_block) and the flags
// are the profile's and cover the block.
void check_kernel_image(const Kernel& kernel, const DeviceProfile& profile);

Kernel compile_and_assemble(std::string_view source, const DeviceProfile& profile);
Kernel compile_and_assemble_file(const std::string& path, const DeviceProfile& profile);

// Hand assembly: raise each stall to what the kernel's fixed-latency register
// and predicate dependencies need under `profile` (the distances qtxc
// schedules with, src/rf.hpp); depbars and waits stay the author's.
Kernel schedule_fixed_stalls(Kernel kernel, const DeviceProfile& profile);

} // namespace nanosm

#pragma once

// Shared between the toolchain's own files (src/qtxc/*.cpp) and included
// nowhere else: each is used by more than one file. Not part of the public
// interface, which is qtxc.hpp. The comment on each function sits with its
// definition.

#include "qtxc/qtxc.hpp"

#include <cstddef>
#include <optional>
#include <stdint.h>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace nanosm {
namespace qtxc_internal {

// Text helpers (parse.cpp); the hex reader uses them too.
std::string trim(std::string value);
std::vector<std::string> operands(const std::string& text);
void expect_count(const std::vector<std::string>& args, std::size_t count, uint32_t line);
std::string read_file(const std::string& path);

// The parser qtx and assembly share (parse.cpp). allow_control: assembly,
// where control blocks and .profile are allowed; qtx refuses both.
AssemblyProgram parse_program(std::string_view source, bool allow_control, const DeviceProfile& profile);

// The toolchain's opcode groups and instruction metadata (metadata.cpp; the
// opcode classes the machine also uses are src/isa.hpp's).
bool is_branch(Opcode opcode);
bool transfers_control(Opcode opcode);
bool has_displacement(Opcode opcode);
std::vector<uint8_t> sources(const Instruction& inst);
std::optional<uint8_t> predicate_destination(const Instruction& inst);
std::optional<uint8_t> destination(const Instruction& inst);
uint16_t register_count(const AssemblyProgram& program);
std::vector<uint8_t> issue_sources(const Instruction& inst);

// The directives beside the code (directives.cpp): `.warps`, the sections
// `.param`, `.mbar`, `.data`, and `.flag`. `tail` is the line after the
// directive's name; with_address: the hex text's form, which carries each
// section's address.
bool is_directive(const std::string& clean, std::string_view name);
std::string section_name(const std::string& name, uint32_t line, const char* what);
std::pair<std::string, uint32_t> mbar_name_and_count(
    const std::string& text, const std::unordered_map<std::string, int32_t>& constants, uint32_t line);
uint32_t parse_warps_directive(const std::string& tail, uint32_t line);
KernelSection parse_param_directive(const std::string& tail, uint32_t line, bool with_address);
KernelSection parse_mbar_directive(
    const std::string& tail, const std::unordered_map<std::string, int32_t>& constants, uint32_t line,
    bool with_address);
KernelSection parse_data_directive(
    const std::string& tail, const std::unordered_map<std::string, int32_t>& constants, uint32_t line,
    bool with_address);
LaunchFlag parse_flag_directive(const std::string& tail, uint32_t line);
std::string format_section_directive(const KernelSection& section, bool with_address);
std::string format_flag_directive(const LaunchFlag& flag);
// `.name %x, r5` (assembly, hex): a qtx register's name and its number.
RegisterName parse_name_directive(const std::string& tail, uint32_t line);
std::string format_name_directive(const RegisterName& entry);

// Constant expressions (expressions.cpp): + - * / << >>, unary minus,
// parentheses, log2(x), over numbers, `.equ` constants (`constants`) and the
// names of whoever is parsing: memory objects (NAME, NAME[i], NAME.bytes,
// .stride, .row_bytes, .row_words) and `.mbar` arrays (NAME[i]). The value,
// or nullopt with `problem` saying why; the caller checks its range.
struct ExpressionNames {
    const std::vector<MemoryObject>* objects = nullptr;
    const std::vector<KernelSection>* sections = nullptr;
    bool addresses_known = false; // the sections have their addresses
};
class ExpressionScope { // while it lives, expressions see these names
public:
    explicit ExpressionScope(const ExpressionNames* names);
    ~ExpressionScope();
    ExpressionScope(const ExpressionScope&) = delete;
    ExpressionScope& operator=(const ExpressionScope&) = delete;
private:
    const ExpressionNames* previous_;
};
std::optional<int64_t> evaluate_expression(
    const std::string& text, const std::unordered_map<std::string, int32_t>& constants, std::string& problem);

// The kernel's own memory by name (objects.cpp): a `.shmem` / `.tmem` line
// (appended to `objects`; allow_at: assembly, which carries each address),
// where qtxc puts the objects, and the line as assembly carries it.
void parse_object_directive(
    const std::string& tail, bool tmem, const std::unordered_map<std::string, int32_t>& constants, uint32_t line,
    bool allow_at, std::vector<MemoryObject>& objects);
void place_objects(std::vector<MemoryObject>& objects, uint32_t shmem_limit, uint64_t tmem_words);
std::string format_object_directive(const MemoryObject& object);

// qtx's registers (registers.cpp): every %name of a parsed qtx program gets
// its number (banks, copies where the banks cannot fit, who may share); the
// program then holds physical registers, as the scheduler expects.
void allocate_registers(AssemblyProgram& program, const DeviceProfile& profile);
const char* file_kind_name(bool uniform);

// The per-instruction rules (checks.cpp): all of them, and the one-round RF
// read rule alone (register allocation checks its own numbers with it).
void check_instruction(const Instruction& inst, const DeviceProfile& profile, const std::string& where);
void check_rf_read_round(const Instruction& inst, const DeviceProfile& profile, const std::string& where);

} // namespace qtxc_internal

// An anonymous namespace is a unique namespace plus a using-directive; this is
// the same with a name the files can share, so a call needs no qualifier.
using namespace qtxc_internal;

} // namespace nanosm

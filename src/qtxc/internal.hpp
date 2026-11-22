#pragma once

// Shared between the toolchain's own files (src/qtxc/*.cpp) and included
// nowhere else. These lived in compiler.cpp's anonymous namespace; each is
// now used by more than one file. Not part of the public interface, which is
// qtxc.hpp. The comment on each function sits with its definition.

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

// Opcode groups and instruction metadata (metadata.cpp; is_setp and is_usetp
// are src/isa.hpp's).
bool is_branch(Opcode opcode);
bool transfers_control(Opcode opcode);
bool has_displacement(Opcode opcode);
bool is_mbar_test_wait(Opcode opcode);
bool is_mbar_try_wait(Opcode opcode);
bool is_mbar_predicate_writer(Opcode opcode);
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
std::pair<std::string, uint32_t> mbar_name_and_count(const std::string& text, uint32_t line);
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

// The per-instruction rules (checks.cpp).
void check_rf_read_round(const Instruction& inst, const DeviceProfile& profile, const std::string& where);
void check_mma_shape(const Instruction& inst, const DeviceProfile& profile, const std::string& where);
void check_tmem_operand(const Instruction& inst, const DeviceProfile& profile, const std::string& where);
void check_divergence(const Instruction& inst, const DeviceProfile& profile, const std::string& where);
void check_write_mask(const Instruction& inst, const DeviceProfile& profile, const std::string& where);
void check_v8_sector(const Instruction& inst, const DeviceProfile& profile, const std::string& where);

} // namespace qtxc_internal

// An anonymous namespace is a unique namespace plus a using-directive; this is
// the same with a name the files can share, so every call reads as it did in
// compiler.cpp.
using namespace qtxc_internal;

} // namespace nanosm

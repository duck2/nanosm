// Assembly text both ways (src/qtxc/): emit_asm writes a scheduled program;
// assemble_asm reads assembly back (qtxc's or hand-written), checks it
// against the profile and encodes it into a Kernel. compile_and_assemble is
// qtx -> Kernel through exactly that text. Nothing here schedules: an
// assembly's controls are taken as written.

#include "qtxc/internal.hpp"

#include <sstream>

namespace nanosm {

std::string emit_asm(const AssemblyProgram& program) {
    std::ostringstream out;
    out << "; scheduled qtx assembly: controls are part of the instruction word\n";
    out << "; {pred, stall, set, read, wait}; six warp-local depbar counters sb0..sb5\n\n";
    out << ".profile " << program.profile_name << ", 0x"
        << format_profile_fingerprint(program.profile_fingerprint) << "\n\n";
    // What the source declares beside the code (directives.cpp): the
    // sections of the launch block in their order, the launch flags the
    // scheduler chose over them, then the constants.
    if (program.warps != 0) out << ".warps " << program.warps << '\n';
    for (const auto& section : program.sections) out << format_section_directive(section, false) << '\n';
    for (const auto& flag : program.flags) out << format_flag_directive(flag) << '\n';
    // The kernel's own memory by name, each object where qtxc put it (objects.cpp).
    for (const auto& object : program.objects) out << format_object_directive(object) << '\n';
    for (const auto& [name, value] : program.constants) {
        out << ".equ " << name << ", " << value << '\n';
    }
    if (program.warps != 0 || !program.sections.empty() || !program.constants.empty() || !program.objects.empty()) {
        out << '\n';
    }
    // The qtx's register names and the numbers qtxc gave them (registers.cpp).
    for (const auto& entry : program.register_names) out << format_name_directive(entry) << '\n';
    if (!program.register_names.empty()) out << '\n';
    for (std::size_t index = 0; index < program.instructions.size(); ++index) {
        for (const auto& label : program.labels) {
            if (label.instruction_index == index) out << label.name << ":\n";
        }
        const auto& inst = program.instructions[index];
        out << "        " << inst.text << ' ' << format_control(inst.control);
        if (index < program.notes.size() && !program.notes[index].empty()) out << " ; " << program.notes[index];
        out << '\n';
    }
    return out.str();
}

Kernel assemble_asm(std::string_view source, const DeviceProfile& profile) {
    validate_device_profile(profile);
    const auto program = parse_program(source, true, profile);
    const auto expected = device_profile_fingerprint(profile);
    if (program.profile_name.empty()) throw SimError("assembly has no .profile directive");
    if (program.profile_name != profile.name || program.profile_fingerprint != expected) {
        throw SimError(
            "assembly profile '" + program.profile_name + "@" +
            format_profile_fingerprint(program.profile_fingerprint) + "' does not match '" +
            profile.name + "@" + format_profile_fingerprint(expected) + "' (the fingerprint stamps the device "
            "profile and the instruction-set revision, " + std::to_string(isa_revision) + " here)");
    }
    Kernel kernel;
    kernel.register_count = register_count(program);
    kernel.profile_name = program.profile_name;
    kernel.profile_fingerprint = program.profile_fingerprint;
    kernel.warps = program.warps;
    kernel.sections = program.sections;
    kernel.flags = program.flags;
    // The names a debugger or a test may ask for: the .equ constants and each
    // section's address.
    kernel.symbols = program.constants;
    for (const auto& section : program.sections) {
        kernel.symbols.emplace_back(section.name, static_cast<int32_t>(section.address));
    }
    // ... and each memory object's address (its elements and sizes are in
    // kernel.objects).
    for (const auto& object : program.objects) {
        kernel.symbols.emplace_back(object.name, static_cast<int32_t>(object.address));
    }
    kernel.objects = program.objects;
    kernel.register_names = program.register_names;
    check_kernel_image(kernel, profile);
    kernel.words.reserve(program.instructions.size());
    for (std::size_t index = 0; index < program.instructions.size(); ++index) {
        const auto& inst = program.instructions[index];
        if (sets_depbar(inst.opcode) && inst.control.set_depbar >= 6) {
            throw SimError("line " + std::to_string(inst.line) + ": memory operation requires set=sb0..sb5");
        }
        if (!sets_depbar(inst.opcode) && inst.control.set_depbar != 7) {
            throw SimError("line " + std::to_string(inst.line) + ": this opcode cannot set a depbar");
        }
        if ((is_setp(inst.opcode) || is_usetp(inst.opcode)) && inst.dst >= 7) {
            throw SimError("line " + std::to_string(inst.line) + ": " + opcode_name(inst.opcode) +
                " destination must be p0..p6");
        }
        if (is_mbar_predicate_writer(inst.opcode) && inst.dst >= 7) {
            throw SimError(
                "line " + std::to_string(inst.line) +
                ": mbar wait predicate destination must be p0..p6");
        }
        check_instruction(inst, profile, "line " + std::to_string(inst.line));
        if (is_tma_opcode(inst.opcode) &&
            (inst.imm < 0 || inst.imm % 64 != 0 || static_cast<uint64_t>(inst.imm) + 64 > profile.shmem_bytes)) {
            throw SimError(
                "line " + std::to_string(inst.line) +
                ": tma descriptor address must be 64-byte aligned and in shmem");
        }
        if (inst.opcode == Opcode::mbar_init_i && (inst.imm <= 0 || inst.imm > 0x7fff)) {
            throw SimError("line " + std::to_string(inst.line) + ": mbar expected count must be 1..32767");
        }
        if (inst.opcode == Opcode::mbar_arrive_expect_tx_i && inst.imm < 0) {
            throw SimError("line " + std::to_string(inst.line) + ": mbar tx byte count cannot be negative");
        }
        if ((inst.opcode == Opcode::mbar_test_wait_i || inst.opcode == Opcode::mbar_try_wait_i) &&
            inst.imm != 0 && inst.imm != 1) {
            throw SimError("line " + std::to_string(inst.line) + ": mbar parity must be 0 or 1");
        }
        if (has_displacement(inst.opcode)) {
            const auto target = static_cast<int64_t>(index + 1) * 8 + inst.imm;
            if (inst.imm % 8 != 0 || target < 0 ||
                static_cast<uint64_t>(target / 8) >= program.instructions.size()) {
                throw SimError("line " + std::to_string(inst.line) + ": invalid branch displacement");
            }
        }
        kernel.words.push_back(encode(inst));
    }
    return kernel;
}

Kernel assemble_asm_file(const std::string& path, const DeviceProfile& profile) {
    return assemble_asm(read_file(path), profile);
}

Kernel compile_and_assemble(std::string_view source, const DeviceProfile& profile) {
    const auto program = compile_qtx(source, profile);
    auto kernel = assemble_asm(emit_asm(program), profile);
    kernel.warnings = program.warnings;
    return kernel;
}

Kernel compile_and_assemble_file(const std::string& path, const DeviceProfile& profile) {
    return compile_and_assemble(read_file(path), profile);
}

} // namespace nanosm

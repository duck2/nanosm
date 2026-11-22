// The rules an instruction must pass whoever wrote it (src/qtxc/): the
// one-round RF read rule, the mma.async shape rule and the acc.ld / acc.st
// words-per-row rule, applied per source
// line by qtxc and the assembler and per pc at kernel load, and the other
// kernel-load checks (validate_kernel, check_uniform_registers). It does not
// parse or schedule. The refusals that depend on what a profile supports
// (try_wait, tma, MMA) stay where they were, in schedule.cpp and
// assemble.cpp.

#include "qtxc/internal.hpp"
#include "rf.hpp"

#include <string>

namespace nanosm {
namespace qtxc_internal {

// The one-round read rule (src/rf.hpp): distinct sources beyond a bank's read
// ports make the instruction malformed. `where` is "line N" or "pc N".
void check_rf_read_round(const Instruction& inst, const DeviceProfile& profile, const std::string& where) {
    if (const auto conflict = rf_read_port_conflict(inst, profile, issue_sources(inst)); conflict.has_value()) {
        throw SimError(where + ": " + *conflict);
    }
}

// An acc.ld / acc.st names the words per row of its region: a power of two,
// 4 or more, and at most tmem's words (src/layout.hpp). A property of the
// instruction, refused at assembly and at kernel load. Where its pointer
// points is a runtime value: the machine's and the sanitizer's business.
void check_tmem_operand(const Instruction& inst, const DeviceProfile& profile, const std::string& where) {
    if (inst.opcode != Opcode::acc_ld && inst.opcode != Opcode::acc_ld_v8 &&
        inst.opcode != Opcode::acc_st && inst.opcode != Opcode::acc_st_v8) {
        return;
    }
    const uint64_t tmem_words = uint64_t{profile.tmem_fine_banks} * profile.tmem_words_per_fine_bank;
    if (inst.imm <= 0 || !valid_tmem_row_words(static_cast<uint32_t>(inst.imm)) ||
        static_cast<uint64_t>(inst.imm) > tmem_words) {
        throw SimError(where + ": " + opcode_name(inst.opcode) + ": words per row " + std::to_string(inst.imm) +
            " must be a power of two, 4 or more and at most tmem's " + std::to_string(tmem_words) + " words");
    }
}

// A write mask (@p / @!p) on an instruction that cannot be masked
// (src/isa.hpp can_be_masked): only the per-lane instructions can; bra and
// bra.uni have their own predicate. Applied by qtxc and the assembler per
// line and at kernel load per pc (check_write_masks).
void check_write_mask(const Instruction& inst, const DeviceProfile& profile, const std::string& where) {
    if (!has_write_mask(inst) || inst.opcode == Opcode::bra || inst.opcode == Opcode::bra_uni ||
        can_be_masked(inst.opcode)) {
        return;
    }
    throw SimError(where + ": " + opcode_name(inst.opcode, &profile.mma_shapes) +
        " cannot be masked (@p): only per-lane instructions can (arithmetic, mov, sread, ur2r, the "
        "special functions, cvt, setp, lds, sts, ldg, stg, ldg.v8, stg.v8)");
}

// ldg.v8 / stg.v8 move one sector per lane, a lane's eight words: they need
// a profile whose sectors are 32 bytes. A property of the instruction and
// the profile, so refused by qtxc and the assembler per line and at kernel
// load per pc (check_v8_sectors), as an mma.async of an unlisted shape is.
// Scalar ldg / stg work with any sector size.
void check_v8_sector(const Instruction& inst, const DeviceProfile& profile, const std::string& where) {
    if (inst.opcode != Opcode::ldg_v8 && inst.opcode != Opcode::stg_v8) return;
    if (profile.gmem_sector_bytes == 32) return;
    throw SimError(where + ": " + opcode_name(inst.opcode) +
        " needs 32-byte sectors (a lane's eight words are one sector); this device profile has " +
        std::to_string(profile.gmem_sector_bytes) + "-byte sectors (gmem.sector_bytes)");
}

// Predication and the divergence stack (trash/divergence-2026-11-21.md):
// a write mask only on an instruction that can be masked (check_write_mask), and ssy, sync
// and a predicated bra need a stack, which a profile with
// core.divergence_stack_entries = 0 does not have. Kernel load refuses both
// too (src/sim.cpp).
void check_divergence(const Instruction& inst, const DeviceProfile& profile, const std::string& where) {
    check_write_mask(inst, profile, where);
    if (profile.core_divergence_stack_entries == 0 && uses_divergence_stack(inst)) {
        throw SimError(where + ": " + (inst.opcode == Opcode::bra ? "a predicated bra" : opcode_name(inst.opcode)) +
            " needs the divergence stack, which this device profile does not have "
            "(core.divergence_stack_entries = 0)");
    }
}

// An mma.async names a shape the profile lists: a property of the
// instruction, so refused at assembly and at kernel load (both builds). Its
// result is a tmem pointer whose rows are the shape's n words; where it
// points is a runtime value. `where` is "line N" or "pc N".
void check_mma_shape(const Instruction& inst, const DeviceProfile& profile, const std::string& where) {
    if (!is_mma_async(inst.opcode)) return;
    const auto shape = mma_shape_of(profile, inst.opcode);
    const auto name = opcode_name(inst.opcode, &profile.mma_shapes);
    if (!shape.has_value()) {
        throw SimError(where + ": " + name + ": this device profile defines no mma shape for it (its list has " +
            std::to_string(profile.mma_shapes.size()) + " shapes)");
    }
    // The row spacings (src/isa.hpp). swizzle64: a spacing of the layout (a
    // power of two, 16 or more; the field reaches 1024) that holds the 16-byte
    // pieces the shape reads in each row (k / 8, rounded up). The dense
    // diagnostic layout is compact in the command's shape and takes none.
    const uint32_t row_bytes = (shape->k + 7) / 8 * 16;
    for (const bool a_operand : {true, false}) {
        const auto spacing = mma_row_spacing(inst, a_operand);
        const std::string operand = a_operand ? "A" : "B";
        if (profile.mma_layout == TileLayout::dense) {
            if (spacing != mma_default_spacing) {
                throw SimError(where + ": " + name + ": the dense mma layout takes no row spacing (" + operand +
                    " names " + std::to_string(spacing) + ")");
            }
        } else if (!valid_swizzle64_spacing(spacing) || spacing < row_bytes) {
            throw SimError(where + ": " + name + ": " + operand + "'s row spacing " + std::to_string(spacing) +
                " must be a power of two bytes, 16 or more, and at least the " + std::to_string(row_bytes) +
                " bytes it reads per row");
        }
    }
}

} // namespace qtxc_internal

void validate_kernel(const Kernel& kernel) {
    if (kernel.profile_name.empty()) throw SimError("kernel has no profile name");
    if (kernel.register_count > 256) throw SimError("kernel register count exceeds 256");
    if (kernel.words.empty()) throw SimError("kernel contains no instructions");
    AssemblyProgram decoded;
    decoded.instructions.reserve(kernel.words.size());
    for (const auto word : kernel.words) decoded.instructions.push_back(decode(word));
    if (kernel.register_count != register_count(decoded)) {
        throw SimError("kernel register count does not match its instructions");
    }
}

void check_mma_shapes(const Kernel& kernel, const DeviceProfile& profile) {
    for (std::size_t index = 0; index < kernel.words.size(); ++index) {
        const auto inst = decode(kernel.words[index]);
        const auto where = "kernel pc " + std::to_string(8 * index);
        check_mma_shape(inst, profile, where);
        check_tmem_operand(inst, profile, where);
    }
}

void check_uniform_registers(const Kernel& kernel, const DeviceProfile& profile) {
    for (std::size_t index = 0; index < kernel.words.size(); ++index) {
        const auto inst = decode(kernel.words[index]);
        if (is_tma_opcode(inst.opcode) &&
            uint32_t{inst.src1} + tma_coordinate_count(inst.opcode) > profile.urf_registers) {
            throw SimError("kernel pc " + std::to_string(8 * index) + ": '" + format_instruction(inst, false) +
                "' names coordinates past this profile's last uniform register, ur" +
                std::to_string(profile.urf_registers - 1));
        }
        auto named = uniform_sources(inst);
        if (const auto dst = uniform_destination(inst); dst.has_value()) named.push_back(*dst);
        for (const auto reg : named) {
            if (reg < profile.urf_registers) continue;
            throw SimError("kernel pc " + std::to_string(8 * index) + ": '" +
                format_instruction(inst, false, &profile.mma_shapes) + "' names uniform register ur" +
                std::to_string(reg) + ", outside this profile's " + std::to_string(profile.urf_registers) +
                " (urf.registers)");
        }
    }
}

void check_v8_sectors(const Kernel& kernel, const DeviceProfile& profile) {
    for (std::size_t index = 0; index < kernel.words.size(); ++index) {
        check_v8_sector(decode(kernel.words[index]), profile, "kernel pc " + std::to_string(8 * index));
    }
}

void check_write_masks(const Kernel& kernel, const DeviceProfile& profile) {
    for (std::size_t index = 0; index < kernel.words.size(); ++index) {
        check_write_mask(decode(kernel.words[index]), profile, "kernel pc " + std::to_string(8 * index));
    }
}

void check_rf_read_rounds(const Kernel& kernel, const DeviceProfile& profile) {
    for (std::size_t index = 0; index < kernel.words.size(); ++index) {
        check_rf_read_round(decode(kernel.words[index]), profile, "kernel pc " + std::to_string(8 * index));
    }
}

} // namespace nanosm

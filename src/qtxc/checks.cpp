// The rules an instruction must pass whoever wrote it (src/qtxc/), in one
// place: check_instruction, applied per source line by qtxc's scheduler and
// the assembler and per pc at kernel load (check_kernel), and the other
// kernel-load checks (validate_kernel, the reserved bits of each word, the
// uniform registers). It does not parse or schedule.

#include "qtxc/internal.hpp"
#include "timing.hpp"

#include <iomanip>
#include <optional>
#include <sstream>
#include <string>

namespace nanosm {
namespace qtxc_internal {

// The one-round read rule (src/timing.hpp): distinct sources beyond a bank's read
// ports make the instruction malformed. `where` is "line N" or "pc N".
void check_rf_read_round(const Instruction& inst, const DeviceProfile& profile, const std::string& where) {
    if (const auto conflict = rf_read_port_conflict(inst, profile, issue_sources(inst)); conflict.has_value()) {
        throw SimError(where + ": " + *conflict);
    }
}

namespace {

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
// bra.uni have their own predicate.
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
// the profile, as an mma.async of an unlisted shape is. Scalar ldg / stg
// work with any sector size.
void check_v8_sector(const Instruction& inst, const DeviceProfile& profile, const std::string& where) {
    if (inst.opcode != Opcode::ldg_v8 && inst.opcode != Opcode::stg_v8) return;
    if (profile.gmem_sector_bytes == 32) return;
    throw SimError(where + ": " + opcode_name(inst.opcode) +
        " needs 32-byte sectors (a lane's eight words are one sector); this device profile has " +
        std::to_string(profile.gmem_sector_bytes) + "-byte sectors (gmem.sector_bytes)");
}

// ssy, sync and a predicated bra need a divergence stack, which a profile
// with core.divergence_stack_entries = 0 does not have.
void check_divergence_stack(const Instruction& inst, const DeviceProfile& profile, const std::string& where) {
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

// An instruction this profile has no unit for: mbar.try_wait needs wait slots
// (sync.mbar_wait_slots), a tma the TMA queue (tma.queue_commands). (A profile
// always has an MMA queue: validate_device_profile refuses none.)
std::optional<std::string> unsupported_by_profile(const Instruction& inst, const DeviceProfile& profile) {
    if (is_mbar_try_wait(inst.opcode) && profile.mbar_wait_slots == 0) {
        return "mbar.try_wait is unsupported by this device profile";
    }
    if (is_tma_opcode(inst.opcode) && profile.tma_queue_commands == 0) {
        return "tma.load / tma.store are unsupported by this device profile";
    }
    return std::nullopt;
}

// Every rule but what the profile supports, in order.
void check_rules(const Instruction& inst, const DeviceProfile& profile, const std::string& where) {
    check_write_mask(inst, profile, where);
    check_divergence_stack(inst, profile, where);
    check_rf_read_round(inst, profile, where);
    check_mma_shape(inst, profile, where);
    check_tmem_operand(inst, profile, where);
    check_v8_sector(inst, profile, where);
}

} // namespace

// The rules every instruction must pass on this profile, whoever wrote it,
// in this order: the profile has its unit (mbar.try_wait, tma); a write mask
// only where it can be (check_write_mask); the divergence stack; the
// one-round RF read; an mma.async's shape and row spacings; an acc.ld /
// acc.st's words per row; ldg.v8 / stg.v8's sectors. qtxc's scheduler and
// the assembler apply it per source line (`where` is "line N"), kernel load
// per pc (check_kernel).
void check_instruction(const Instruction& inst, const DeviceProfile& profile, const std::string& where) {
    if (const auto refusal = unsupported_by_profile(inst, profile); refusal.has_value()) {
        throw SimError(where + ": " + *refusal);
    }
    check_rules(inst, profile, where);
}

} // namespace qtxc_internal

namespace {

// A word with a bit set that its form does not define (src/isa.hpp
// reserved_bit): refused, naming the pc and the bit, before anything reads
// the word's fields.
void check_reserved_bits(const Kernel& kernel) {
    for (std::size_t index = 0; index < kernel.words.size(); ++index) {
        const auto word = kernel.words[index];
        const auto bit = reserved_bit(word);
        if (!bit.has_value()) continue;
        std::ostringstream hex;
        hex << std::hex << std::setw(16) << std::setfill('0') << word;
        throw SimError("kernel pc " + std::to_string(8 * index) + ": word 0x" + hex.str() + " ('" +
            format_instruction(decode(word), false) + "') sets bit " + std::to_string(*bit) +
            ", which its form does not define; a reserved bit must be zero (this reader is instruction-set "
            "revision " + std::to_string(isa_revision) + ")");
    }
}

} // namespace

void validate_kernel(const Kernel& kernel) {
    if (kernel.profile_name.empty()) throw SimError("kernel has no profile name");
    if (kernel.register_count > 256) throw SimError("kernel register count exceeds 256");
    if (kernel.words.empty()) throw SimError("kernel contains no instructions");
    check_reserved_bits(kernel);
    AssemblyProgram decoded;
    decoded.instructions.reserve(kernel.words.size());
    for (const auto word : kernel.words) decoded.instructions.push_back(decode(word));
    if (kernel.register_count != register_count(decoded)) {
        throw SimError("kernel register count does not match its instructions");
    }
}

void check_kernel(const Kernel& kernel, const DeviceProfile& profile) {
    check_reserved_bits(kernel);
    // What the profile has no unit for is refused for the kernel as a whole.
    for (const auto word : kernel.words) {
        if (const auto refusal = unsupported_by_profile(decode(word), profile); refusal.has_value()) {
            throw SimError(*refusal);
        }
    }
    for (std::size_t index = 0; index < kernel.words.size(); ++index) {
        const auto inst = decode(kernel.words[index]);
        check_rules(inst, profile, "kernel pc " + std::to_string(8 * index));
        // Every uniform register it names is one the profile's URF has.
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

} // namespace nanosm

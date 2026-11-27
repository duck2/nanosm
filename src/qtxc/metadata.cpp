// Instruction metadata for the toolchain (src/qtxc/): the opcode groups the
// scheduler, the assembler and the checks test for, and a program's register
// count. Which registers an instruction reads and writes is src/isa.hpp's, the
// timing of a read or a write src/timing.hpp's. Nothing here parses,
// schedules or refuses.

#include "qtxc/internal.hpp"

#include <algorithm>

namespace nanosm {
namespace qtxc_internal {

// bra and bra.uni: a target (a displacement) and a control transfer.
bool is_branch(Opcode opcode) {
    return opcode == Opcode::bra || opcode == Opcode::bra_uni;
}

// Where control may go somewhere other than the next instruction: the
// branches, and sync, whose implicit edge
// goes to the pc of the entry it pops: a divergent bra's target or an ssy's
// label (the instruction after the sync when that is the pc). The
// scheduler drains at each of them.
bool transfers_control(Opcode opcode) {
    return is_branch(opcode) || opcode == Opcode::sync;
}

// The instructions whose immediate is a displacement to a label: the
// branches and ssy (its label).
bool has_displacement(Opcode opcode) {
    return is_branch(opcode) || opcode == Opcode::ssy;
}

uint16_t register_count(const AssemblyProgram& program) {
    uint16_t count = 0;
    for (const auto& inst : program.instructions) {
        for (const auto reg : vector_sources(inst)) {
            count = std::max<uint16_t>(count, static_cast<uint16_t>(reg) + 1);
        }
        if (const auto dst = vector_destination(inst); dst.has_value()) {
            count = std::max<uint16_t>(count, static_cast<uint16_t>(*dst + register_span(inst.opcode)));
        }
    }
    return count;
}

} // namespace qtxc_internal
} // namespace nanosm

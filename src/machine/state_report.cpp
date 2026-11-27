// The state report: every piece of clocked state by unit, with what the code knows of its size,
// and every black box's contract.
// (Machine is one class, src/machine/machine.hpp; the map of its files is at the top of
// src/machine/machine.cpp.)

#include "machine/machine.hpp"

#include <algorithm>
#include <cstdio>
#include <sstream>

namespace nanosm {

namespace {

// A number with its thousands grouped: 1,442,878.
std::string number(uint64_t value) {
    std::string digits = std::to_string(value);
    for (auto at = static_cast<std::ptrdiff_t>(digits.size()) - 3; at > 0; at -= 3) digits.insert(at, ",");
    return digits;
}

// What the code knows of one piece of state's size, as the report prints it.
std::string size_text(const StateSize& size) {
    const std::string kind = size.kind;
    if (kind == "group") return number(size.entries) + (size.entries == 1 ? " member" : " members");
    if (kind == "pipeline") return number(size.entries) + (size.entries == 1 ? " stage" : " stages");
    if (kind == "FIFO") {
        return "capacity " + number(size.entries) + ", write " + std::to_string(size.write_ports) + " / read " +
            std::to_string(size.read_ports) + " entries wide, read latency " + std::to_string(size.read_delay);
    }
    if (kind == "memory") {
        return "banks x words x word bits " + number(size.banks) + " x " + number(size.entries / size.banks) + " x " +
            number(size.word_bits) + "; " + std::to_string(size.read_ports) + "R " + std::to_string(size.write_ports) +
            "W" + (size.banks > 1 ? " per bank" : "") + ", read delay " + std::to_string(size.read_delay) + ", " +
            size.same_cycle;
    }
    if (kind == "black box") return "its contract below";
    return "";
}

} // namespace

// The black boxes (blackbox.hpp), each declared where it is defined: the
// arithmetic units (arith.cpp), the tensor core's
// arithmetic (tensor_core.cpp), and the memories, beside each in machine.hpp.
// Listed by unit.
void Machine::declare_black_boxes() {
    declare_arithmetic_black_boxes();
    declare_mma_core_black_box();
    black_boxes_.push_back(instruction_memory_black_box());
    black_boxes_.push_back(divergence_stack_black_box());
    black_boxes_.push_back(rf_black_box());
    black_boxes_.push_back(urf_black_box());
    black_boxes_.push_back(shmem_black_box());
    black_boxes_.push_back(tmem_black_box());
    std::stable_sort(black_boxes_.begin(), black_boxes_.end(),
        [](const BlackBoxDeclaration& a, const BlackBoxDeclaration& b) { return a.unit < b.unit; });
}

// ===========================================================================
// The report. Per unit: each piece of clocked state (AllClocked, in
// declaration order) with its kind, its name and its size -- a group's
// members, a pipeline's stages, a FIFO's capacity and ports, a memory's shape.
// Then each black box's contract: its latency and the profile settings it
// comes from, its interval, an arithmetic unit's operations, its inputs and
// outputs.
// ===========================================================================

std::string Machine::state_report() const {
    std::ostringstream out;
    const auto sizes = clocked_.sizes();
    out << "state report: " << config_.name << "\n";
    out << "Every piece of clocked state, by unit: its kind, its name and its size. Then every black box's\n"
           "contract.\n";
    for (auto unit = static_cast<uint8_t>(Unit::front_end); unit <= static_cast<uint8_t>(last_unit); ++unit) {
        const auto which = static_cast<Unit>(unit);
        out << "\n" << unit_name(which) << "\n";
        for (const auto& size : sizes) {
            if (size.unit != which) continue;
            char line[160];
            std::snprintf(line, sizeof(line), "  %-9s %-52s  ", size.kind, size.name.c_str());
            std::string text = line + size_text(size);
            text.erase(text.find_last_not_of(' ') + 1);
            out << text << "\n";
        }
    }
    out << "\nblack boxes (a contract outside, anything inside)\n";
    if (black_boxes_.empty()) out << "  none\n";
    for (const auto& black_box : black_boxes_) {
        out << "  " << black_box.name << " (" << unit_name(black_box.unit) << ")\n    latency " << black_box.latency
            << " (" << black_box.latency_from << "), interval " << black_box.interval << "\n";
        if (!black_box.operations.empty()) out << "    ops  " << black_box.operations << "\n";
        for (const auto* ports : {&black_box.inputs, &black_box.outputs}) {
            for (const auto& port : *ports) {
                out << (ports == &black_box.inputs ? "    in   " : "    out  ") << port.name << ": " << number(port.bits)
                    << (port.bits == 1 ? " bit, " : " bits, ") << port.held_in << "\n";
            }
        }
    }
    out << "\noutside the SM (not in the report)\n";
    out << "  global memory; the memory controller's queue, bus and write acknowledgements\n";
    return out.str();
}

} // namespace nanosm

// qtx's registers (src/qtxc/): a qtx register is a name, %x, and
// allocate_registers gives every name a number. It runs between the parser,
// which leaves each name's place in its file's list (vector_registers or
// uniform_registers) in the instruction's fields, and the scheduler, which
// sees numbers only.
//
// What it decides, in order:
//   1. Banks (vector file). An instruction's issue reads at most
//      rf.read_ports_per_bank different registers from one bank (register
//      number mod rf.banks; src/timing.hpp). Every name or group (`.reg %a<8>`:
//      neighbours, so one choice for all eight) that is read beside another
//      gets a bank: a search for a choice with no conflict, and when there
//      is none (or the search runs out) the choice with the least conflict,
//      an instruction inside a loop counting more.
//   2. Copies. For an instruction still in conflict, a copy of one source
//      into a scratch register of a free bank, immediately before it; a
//      label on the instruction moves to the copy. Inside a loop the source
//      copied is one the innermost loop does not write, when there is one. A
//      warning says so, and whether a copy made before the loop would do.
//   3. Who may share a number. A name holds its value from a write to the
//      last read that needs it (liveness over the kernel's control flow);
//      two names that never hold a value at the same time may share.
//   4. Numbers. Every name gets a number of its own while the file has
//      room (a shared number makes one name's write wait for the other's
//      reads, which the scheduler then pays for); names share only when it
//      would otherwise overflow.
// Then the fields and the text get the numbers, and the program carries the
// table (register_names) for the assembly's `.name` lines.

#include "qtxc/internal.hpp"
#include "timing.hpp"

#include <algorithm>
#include <array>
#include <map>
#include <numeric>
#include <regex>
#include <set>

namespace nanosm {
namespace {

using QtxRegister = AssemblyProgram::QtxRegister;

// ---- which fields of an instruction are registers, and of which file ----

enum class Kind : uint8_t { none, vector, uniform };

struct Slot {
    Kind kind = Kind::none;
    uint32_t width = 1; // 8 for a v8's register operand, N for a tma's coordinates
    bool read = false;
    bool written = false;
};

// dst, src0, src1, src2.
using Slots = std::array<Slot, 4>;

uint8_t& field(Instruction& inst, std::size_t slot) {
    switch (slot) {
    case 0: return inst.dst;
    case 1: return inst.src0;
    case 2: return inst.src1;
    default: return inst.src2;
    }
}

uint8_t field(const Instruction& inst, std::size_t slot) {
    switch (slot) {
    case 0: return inst.dst;
    case 1: return inst.src0;
    case 2: return inst.src1;
    default: return inst.src2;
    }
}

// Asked of the functions that already know (metadata.cpp sources /
// destination, src/isa.hpp uniform_sources / uniform_destination) by giving
// a copy of the instruction four far-apart field values and seeing which
// come back, so there is no second table to keep in step.
Slots register_slots(const Instruction& inst) {
    Instruction probe = inst;
    static constexpr std::array<uint8_t, 4> marks = {0x10, 0x30, 0x50, 0x70};
    for (std::size_t slot = 0; slot < 4; ++slot) field(probe, slot) = marks[slot];
    Slots slots;
    const auto note = [&](uint8_t value, Kind kind, bool written) {
        const std::size_t slot = (value >> 5) & 3;
        auto& entry = slots[slot];
        entry.kind = kind;
        entry.width = std::max<uint32_t>(entry.width, static_cast<uint32_t>(value - marks[slot]) + 1);
        (written ? entry.written : entry.read) = true;
    };
    for (const auto value : sources(probe)) note(value, Kind::vector, false);
    for (const auto value : uniform_sources(probe)) note(value, Kind::uniform, false);
    if (const auto dst = destination(probe); dst.has_value()) {
        note(*dst, Kind::vector, true);
        if (inst.opcode == Opcode::acc_ld_v8 || inst.opcode == Opcode::ldg_v8) slots[(*dst >> 5) & 3].width = 8;
    }
    if (const auto dst = uniform_destination(probe); dst.has_value()) note(*dst, Kind::uniform, true);
    return slots;
}

// ---- small sets of places ----

struct Bits {
    std::vector<uint64_t> words;
    explicit Bits(std::size_t count = 0) : words((count + 63) / 64, 0) {}
    void set(std::size_t i) { words[i / 64] |= uint64_t{1} << (i % 64); }
    void clear(std::size_t i) { words[i / 64] &= ~(uint64_t{1} << (i % 64)); }
    [[nodiscard]] bool test(std::size_t i) const { return (words[i / 64] >> (i % 64)) & 1; }
    bool merge(const Bits& other) { // |=, true when something was added
        bool changed = false;
        for (std::size_t w = 0; w < words.size(); ++w) {
            const auto next = words[w] | other.words[w];
            changed = changed || next != words[w];
            words[w] = next;
        }
        return changed;
    }
};

// ---- one register file's names ----

// A name, or a group (which is placed as one piece).
struct Node {
    uint32_t first = 0; // place of its first member in the file's list
    uint32_t size = 1;
    bool scratch = false;
    int32_t bank = -1; // vector file: the bank of its first member, when it matters
    int32_t base = -1; // the number of its first member
};

struct File {
    bool uniform = false;
    std::vector<QtxRegister>* registers = nullptr;
    std::vector<Node> nodes;
    std::vector<uint32_t> node_of; // per place
};

File make_file(AssemblyProgram& program, bool uniform) {
    File file;
    file.uniform = uniform;
    file.registers = uniform ? &program.uniform_registers : &program.vector_registers;
    file.node_of.assign(file.registers->size(), 0);
    for (uint32_t place = 0; place < file.registers->size();) {
        const auto& entry = (*file.registers)[place];
        Node node;
        node.first = place;
        node.scratch = entry.scratch;
        if (entry.group >= 0) node.size = program.register_groups[static_cast<std::size_t>(entry.group)].size;
        for (uint32_t member = 0; member < node.size; ++member) {
            file.node_of[place + member] = static_cast<uint32_t>(file.nodes.size());
        }
        place += node.size;
        file.nodes.push_back(node);
    }
    return file;
}

std::string percent(const QtxRegister& entry) { return "%" + entry.name; }

// How deep in loops each instruction sits: a loop is the stretch from a
// backward branch's target to the branch.
std::vector<uint32_t> loop_depths(const std::vector<Instruction>& instructions) {
    std::vector<uint32_t> depth(instructions.size(), 0);
    for (std::size_t index = 0; index < instructions.size(); ++index) {
        const auto& inst = instructions[index];
        if (!is_branch(inst.opcode)) continue;
        const auto target = static_cast<int64_t>(index) + 1 + inst.imm / 8;
        if (target < 0 || target > static_cast<int64_t>(index)) continue;
        for (auto i = static_cast<std::size_t>(target); i <= index; ++i) ++depth[i];
    }
    return depth;
}

// The innermost loop around each instruction, as [first, last]. A loop is a
// header and every backward branch to it: the stretch from the header to the
// last of them (an earlier one, a `continue`, is inside the same loop). The
// innermost loop around an instruction is the shortest such stretch that
// holds it. nullopt outside every loop.
std::vector<std::optional<std::pair<std::size_t, std::size_t>>> innermost_loops(
    const std::vector<Instruction>& instructions) {
    // Per header: the last backward branch to it.
    std::vector<std::optional<std::size_t>> last_back_edge(instructions.size());
    for (std::size_t index = 0; index < instructions.size(); ++index) {
        const auto& inst = instructions[index];
        if (!is_branch(inst.opcode)) continue;
        const auto target = static_cast<int64_t>(index) + 1 + inst.imm / 8;
        if (target < 0 || target > static_cast<int64_t>(index)) continue;
        last_back_edge[static_cast<std::size_t>(target)] = index;
    }
    std::vector<std::optional<std::pair<std::size_t, std::size_t>>> loops(instructions.size());
    for (std::size_t first = 0; first < instructions.size(); ++first) {
        if (!last_back_edge[first].has_value()) continue;
        const auto last = *last_back_edge[first];
        for (auto i = first; i <= last; ++i) {
            if (!loops[i].has_value() || last - first < loops[i]->second - loops[i]->first) loops[i] = {{first, last}};
        }
    }
    return loops;
}

// The different vector registers an instruction's issue reads, as places.
std::vector<uint8_t> issue_places(const Instruction& inst) { return rf_read_round(issue_sources(inst)); }

// ---- 1. banks ----

// One instruction's read round: its members must spread over the banks.
struct Round {
    std::vector<std::pair<uint32_t, uint32_t>> members; // (node, offset in the node)
    uint64_t weight = 1;
};

struct Banks {
    uint32_t banks = 1;
    uint32_t ports = 1;
    std::vector<Round> rounds;
    std::vector<std::vector<uint32_t>> rounds_of; // per node
    std::vector<int32_t> bank;                    // per node, -1: not chosen (or it does not matter)

    // How many registers too many `round` asks of some bank, among the nodes
    // that have a bank.
    [[nodiscard]] uint32_t excess(const Round& round) const {
        std::vector<uint32_t> count(banks, 0);
        for (const auto& [node, offset] : round.members) {
            if (bank[node] >= 0) ++count[(static_cast<uint32_t>(bank[node]) + offset) % banks];
        }
        uint32_t result = 0;
        for (const auto c : count) result += c > ports ? c - ports : 0;
        return result;
    }
    [[nodiscard]] uint64_t cost_of(uint32_t node) const {
        uint64_t cost = 0;
        for (const auto index : rounds_of[node]) cost += rounds[index].weight * excess(rounds[index]);
        return cost;
    }
};

void choose_banks(Banks& state) {
    const auto node_count = state.rounds_of.size();
    std::vector<uint32_t> order;
    std::vector<uint64_t> weight(node_count, 0);
    for (uint32_t node = 0; node < node_count; ++node) {
        if (state.rounds_of[node].empty()) continue;
        order.push_back(node);
        for (const auto index : state.rounds_of[node]) weight[node] += state.rounds[index].weight;
    }
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return weight[a] > weight[b]; });
    std::vector<uint32_t> used(state.banks, 0); // nodes per bank, to spread them
    const auto preference = [&]() {
        std::vector<uint32_t> banks(state.banks);
        std::iota(banks.begin(), banks.end(), 0u);
        std::stable_sort(banks.begin(), banks.end(), [&](uint32_t a, uint32_t b) { return used[a] < used[b]; });
        return banks;
    };

    // A search for a choice without any conflict, bounded.
    uint64_t steps = 0;
    const uint64_t budget = 200000;
    bool gave_up = false;
    const auto search = [&](const auto& self, std::size_t position) -> bool {
        if (position == order.size()) return true;
        if (++steps > budget) {
            gave_up = true;
            return false;
        }
        const auto node = order[position];
        for (const auto candidate : preference()) {
            state.bank[node] = static_cast<int32_t>(candidate);
            if (state.cost_of(node) == 0) {
                ++used[candidate];
                if (self(self, position + 1)) return true;
                --used[candidate];
                if (gave_up) break;
            }
        }
        state.bank[node] = -1;
        return false;
    };
    if (search(search, 0)) return;

    // None (or none found in time): each node in turn takes the bank that
    // costs least, then single moves while one helps.
    std::fill(used.begin(), used.end(), 0u);
    for (const auto node : order) state.bank[node] = -1;
    for (const auto node : order) {
        uint32_t best = 0;
        uint64_t best_cost = ~uint64_t{0};
        for (const auto candidate : preference()) {
            state.bank[node] = static_cast<int32_t>(candidate);
            const auto cost = state.cost_of(node);
            if (cost < best_cost) {
                best_cost = cost;
                best = candidate;
            }
        }
        state.bank[node] = static_cast<int32_t>(best);
        ++used[best];
    }
    for (int pass = 0; pass < 64; ++pass) {
        bool moved = false;
        for (const auto node : order) {
            const auto current = static_cast<uint32_t>(state.bank[node]);
            uint32_t best = current;
            uint64_t best_cost = state.cost_of(node);
            for (uint32_t candidate = 0; candidate < state.banks; ++candidate) {
                state.bank[node] = static_cast<int32_t>(candidate);
                const auto cost = state.cost_of(node);
                if (cost < best_cost) {
                    best_cost = cost;
                    best = candidate;
                }
            }
            state.bank[node] = static_cast<int32_t>(best);
            moved = moved || best != current;
        }
        if (!moved) break;
    }
}

// ---- 2. copies ----

// `name` as a whole token in `text` -> `replacement`.
std::string replace_token(const std::string& text, const std::string& name, const std::string& replacement) {
    std::string result;
    std::size_t at = 0;
    while (at < text.size()) {
        const auto found = text.find(name, at);
        if (found == std::string::npos) break;
        const auto end = found + name.size();
        const bool whole = end == text.size() ||
            !(std::isalnum(static_cast<unsigned char>(text[end])) != 0 || text[end] == '_');
        result += text.substr(at, found - at);
        result += whole ? replacement : name;
        at = end;
    }
    result += text.substr(at);
    return result;
}

// The instruction's text with source `name` replaced: its destination (the
// first operand, when it writes a vector register) is left alone, and of a
// stg.v8 only the address is touched (its data is a group).
std::string replace_source(const Instruction& inst, const std::string& name, const std::string& replacement) {
    const auto space = inst.text.find_first_of(" \t");
    if (space == std::string::npos) return inst.text;
    auto head = inst.text.substr(0, space + 1);
    auto tail = inst.text.substr(space + 1);
    std::string kept;
    if (destination(inst).has_value()) {
        const auto comma = tail.find(',');
        if (comma == std::string::npos) return inst.text;
        head += tail.substr(0, comma + 1);
        tail = tail.substr(comma + 1);
    } else if (inst.opcode == Opcode::stg_v8) {
        const auto close = tail.find(']');
        if (close == std::string::npos) return inst.text;
        kept = tail.substr(close + 1);
        tail = tail.substr(0, close + 1);
    }
    return head + replace_token(tail, name, replacement) + kept;
}

// ---- 3. who holds a value when ----

// Where control may go from instruction `index`.
std::vector<std::size_t> successors(
    const std::vector<Instruction>& instructions, std::size_t index, const std::vector<std::size_t>& popped) {
    const auto& inst = instructions[index];
    std::vector<std::size_t> result;
    const bool has_next = index + 1 < instructions.size();
    if (inst.opcode == Opcode::halt) return result;
    if (is_branch(inst.opcode)) {
        result.push_back(static_cast<std::size_t>(static_cast<int64_t>(index) + 1 + inst.imm / 8));
        // A predicate makes it conditional (bra.uni) or a split (bra).
        if (has_write_mask(inst) && has_next) result.push_back(index + 1);
        return result;
    }
    if (inst.opcode == Opcode::sync) {
        // Wherever the entry it pops says: any label an ssy or a masked bra
        // names (the instruction after the sync, when that is the label).
        result = popped;
    }
    if (has_next) result.push_back(index + 1);
    return result;
}

struct Liveness {
    std::vector<Bits> in;
    std::vector<Bits> out;
    std::vector<Bits> defined;
};

// uses(i) / defines(i): places of this file an instruction reads and writes;
// kept(i): written only in part, so the old value stays wanted.
template <typename Uses, typename Defines>
Liveness liveness(
    const std::vector<Instruction>& instructions, std::size_t places, const Uses& uses, const Defines& defines) {
    std::vector<std::size_t> popped;
    for (std::size_t index = 0; index < instructions.size(); ++index) {
        const auto& inst = instructions[index];
        if (inst.opcode == Opcode::ssy || is_divergent_branch(inst)) {
            popped.push_back(static_cast<std::size_t>(static_cast<int64_t>(index) + 1 + inst.imm / 8));
        }
    }
    Liveness result;
    result.in.assign(instructions.size(), Bits(places));
    result.out.assign(instructions.size(), Bits(places));
    result.defined.assign(instructions.size(), Bits(places));
    std::vector<Bits> used(instructions.size(), Bits(places));
    std::vector<Bits> killed(instructions.size(), Bits(places));
    std::vector<std::vector<std::size_t>> next(instructions.size());
    for (std::size_t index = 0; index < instructions.size(); ++index) {
        next[index] = successors(instructions, index, popped);
        for (const auto place : uses(index)) used[index].set(place);
        for (const auto& [place, whole] : defines(index)) {
            result.defined[index].set(place);
            if (whole) killed[index].set(place);
            else used[index].set(place);
        }
    }
    for (bool changed = true; changed;) {
        changed = false;
        for (std::size_t index = instructions.size(); index-- > 0;) {
            for (const auto successor : next[index]) {
                if (successor < instructions.size()) changed = result.out[index].merge(result.in[successor]) || changed;
            }
            Bits in = result.out[index];
            for (std::size_t w = 0; w < in.words.size(); ++w) {
                in.words[w] = (in.words[w] & ~killed[index].words[w]) | used[index].words[w];
            }
            changed = result.in[index].merge(in) || changed;
        }
    }
    return result;
}

// Two nodes may not share a number when both hold a value at some point: at
// each instruction, everything alive after it and everything it writes; and
// everything alive at the kernel's first instruction (read before written).
std::vector<std::vector<bool>> interference(const File& file, const Liveness& live) {
    const auto count = file.nodes.size();
    std::vector<std::vector<bool>> result(count, std::vector<bool>(count, false));
    const auto mark = [&](const std::vector<uint32_t>& nodes) {
        for (const auto a : nodes) {
            for (const auto b : nodes) {
                if (a != b) result[a][b] = true;
            }
        }
    };
    const auto nodes_of = [&](const Bits& places_a, const Bits* places_b) {
        std::vector<bool> seen(count, false);
        std::vector<uint32_t> nodes;
        for (std::size_t place = 0; place < file.node_of.size(); ++place) {
            if (!places_a.test(place) && (places_b == nullptr || !places_b->test(place))) continue;
            const auto node = file.node_of[place];
            if (!seen[node]) nodes.push_back(node);
            seen[node] = true;
        }
        return nodes;
    };
    for (std::size_t index = 0; index < live.out.size(); ++index) {
        mark(nodes_of(live.out[index], &live.defined[index]));
    }
    if (!live.in.empty()) mark(nodes_of(live.in[0], nullptr));
    return result;
}

// ---- 4. numbers ----

void number_file(
    File& file, const std::vector<std::vector<bool>>& interferes, uint32_t banks, uint32_t soft, uint32_t hard,
    const std::string& capacity) {
    // Groups first (the largest first), then the names whose bank matters in
    // the order they appear, then the rest, which fill what is left.
    std::vector<uint32_t> order(file.nodes.size());
    std::iota(order.begin(), order.end(), 0u);
    const auto rank = [&](uint32_t node) {
        const auto& n = file.nodes[node];
        return n.size > 1 ? 0 : n.bank >= 0 ? 1 : 2;
    };
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        if (rank(a) != rank(b)) return rank(a) < rank(b);
        if (rank(a) == 0 && file.nodes[a].size != file.nodes[b].size) return file.nodes[a].size > file.nodes[b].size;
        return false;
    });
    std::vector<uint32_t> placed;
    const auto fits = [&](uint32_t node, uint32_t base, bool share) {
        const auto end = base + file.nodes[node].size;
        for (const auto other : placed) {
            if (share && !interferes[node][other]) continue;
            const auto other_base = static_cast<uint32_t>(file.nodes[other].base);
            if (base < other_base + file.nodes[other].size && other_base < end) return false;
        }
        return true;
    };
    for (const auto node : order) {
        auto& n = file.nodes[node];
        const uint32_t step = n.bank >= 0 ? banks : 1;
        const uint32_t start = n.bank >= 0 ? static_cast<uint32_t>(n.bank) : 0;
        // A number of its own within the usual room; else a shared one there;
        // else a shared one anywhere the file reaches.
        for (const auto& [share, limit] : {std::pair{false, soft}, std::pair{true, soft}, std::pair{true, hard}}) {
            for (uint32_t base = start; n.base < 0 && base + n.size <= limit; base += step) {
                if (fits(node, base, share)) n.base = static_cast<int32_t>(base);
            }
        }
        if (n.base < 0) {
            const auto& entry = (*file.registers)[n.first];
            throw SimError("line " + std::to_string(entry.line) + ": no " + file_kind_name(file.uniform) +
                " register is left for " + percent(entry) + ": more values are held at once than " + capacity);
        }
        placed.push_back(node);
    }
}

} // namespace

namespace qtxc_internal {

const char* file_kind_name(bool uniform) { return uniform ? "uniform" : "vector"; }

void allocate_registers(AssemblyProgram& program, const DeviceProfile& profile) {
    const uint32_t banks = profile.rf_banks;
    const uint32_t ports = profile.rf_read_ports_per_bank;
    auto& instructions = program.instructions;
    program.notes.assign(instructions.size(), std::string());

    // 1. Banks for the vector names that are read beside another.
    {
        const auto vectors = make_file(program, false);
        const auto depth = loop_depths(instructions);
        Banks state;
        state.banks = banks;
        state.ports = ports;
        state.rounds_of.assign(vectors.nodes.size(), {});
        state.bank.assign(vectors.nodes.size(), -1);
        for (std::size_t index = 0; index < instructions.size(); ++index) {
            const auto places = issue_places(instructions[index]);
            if (places.size() <= ports) continue;
            if (places.size() > uint64_t{banks} * ports) {
                throw SimError("line " + std::to_string(instructions[index].line) + ": '" + instructions[index].text +
                    "' reads " + std::to_string(places.size()) + " different registers at once; this profile's " +
                    "register file serves " + std::to_string(uint64_t{banks} * ports) + " (rf.banks x rf.read_ports_per_bank)");
            }
            Round round;
            round.weight = uint64_t{1} << (3 * std::min<uint32_t>(depth[index], 6));
            for (const auto place : places) {
                const auto node = vectors.node_of[place];
                round.members.emplace_back(node, place - vectors.nodes[node].first);
                state.rounds_of[node].push_back(static_cast<uint32_t>(state.rounds.size()));
            }
            state.rounds.push_back(std::move(round));
        }
        choose_banks(state);

        // 2. A copy for each source still in a bank that is asked too much.
        const auto bank_of = [&](uint8_t place, const std::map<uint8_t, uint32_t>& scratch_banks) {
            if (const auto found = scratch_banks.find(place); found != scratch_banks.end()) return found->second;
            const auto node = vectors.node_of[place];
            // A name read beside nothing, anywhere, has no bank; it never
            // shows up here (this instruction would have given it one).
            return (static_cast<uint32_t>(state.bank[node]) + place - vectors.nodes[node].first) % banks;
        };
        std::map<std::pair<uint32_t, uint32_t>, uint8_t> scratch; // (bank, how many this instruction took) -> place
        std::map<uint8_t, uint32_t> scratch_banks;                 // place -> bank
        std::set<std::string> taken;
        for (const auto& entry : program.vector_registers) taken.insert(entry.name);
        for (const auto& entry : program.uniform_registers) taken.insert(entry.name);
        const auto scratch_place = [&](uint32_t bank, uint32_t nth, uint32_t line) {
            if (const auto found = scratch.find({bank, nth}); found != scratch.end()) return found->second;
            if (program.vector_registers.size() >= 256) {
                throw SimError("line " + std::to_string(line) + ": more than 256 vector register names (with qtxc's copies)");
            }
            QtxRegister entry;
            entry.name = "qtxc_copy" + std::to_string(scratch.size());
            while (taken.contains(entry.name)) entry.name += "_";
            taken.insert(entry.name);
            entry.line = line;
            entry.where = "a copy qtxc inserted";
            entry.scratch = true;
            program.vector_registers.push_back(std::move(entry));
            const auto place = static_cast<uint8_t>(program.vector_registers.size() - 1);
            scratch[{bank, nth}] = place;
            scratch_banks[place] = bank;
            return place;
        };

        std::vector<Instruction> rebuilt;
        std::vector<uint8_t> rebuilt_flags;
        std::vector<std::string> rebuilt_notes;
        std::vector<std::size_t> new_index(instructions.size(), 0); // of an old instruction's first new one
        std::vector<int64_t> target(instructions.size(), -1);       // a displacement's old target
        const auto loops = innermost_loops(instructions);
        for (std::size_t index = 0; index < instructions.size(); ++index) {
            if (has_displacement(instructions[index].opcode)) {
                target[index] = static_cast<int64_t>(index) + 1 + instructions[index].imm / 8;
            }
        }
        for (std::size_t index = 0; index < instructions.size(); ++index) {
            auto inst = instructions[index];
            new_index[index] = rebuilt.size();
            std::vector<uint32_t> taken_here(banks, 0);
            for (;;) {
                const auto places = issue_places(inst);
                std::vector<std::vector<uint8_t>> by_bank(banks);
                if (places.size() > ports) {
                    for (const auto place : places) by_bank[bank_of(place, scratch_banks)].push_back(place);
                }
                uint32_t crowded = banks;
                uint32_t free_bank = 0;
                for (uint32_t bank = 0; bank < banks; ++bank) {
                    if (by_bank[bank].size() > ports && crowded == banks) crowded = bank;
                    if (by_bank[bank].size() < by_bank[free_bank].size()) free_bank = bank;
                }
                if (crowded == banks) break;
                // One of the crowded bank's registers moves; the others stay.
                // Inside a loop, the last in operand order that the innermost
                // loop around the instruction does not write (its copy could
                // be made once, before the loop); else the last.
                const auto& loop = loops[index];
                const auto written_in_loop = [&](uint8_t place) {
                    if (program.vector_registers[place].scratch) return true;
                    for (auto i = loop->first; i <= loop->second; ++i) {
                        const auto slots = register_slots(instructions[i]);
                        for (std::size_t slot = 0; slot < 4; ++slot) {
                            if (slots[slot].kind != Kind::vector || !slots[slot].written) continue;
                            const uint32_t first = field(instructions[i], slot);
                            if (place >= first && place < first + slots[slot].width) return true;
                        }
                    }
                    return false;
                };
                auto victim = by_bank[crowded].back();
                bool invariant = false;
                if (loop.has_value()) {
                    for (auto it = by_bank[crowded].rbegin(); it != by_bank[crowded].rend(); ++it) {
                        if (written_in_loop(*it)) continue;
                        victim = *it;
                        invariant = true;
                        break;
                    }
                }
                const auto place = scratch_place(free_bank, taken_here[free_bank]++, inst.line);
                const auto& victim_entry = program.vector_registers[victim];
                const auto& scratch_entry = program.vector_registers[place];
                std::string beside;
                for (const auto other : by_bank[crowded]) {
                    if (other == victim) continue;
                    beside += (beside.empty() ? "" : ", ") + percent(program.vector_registers[other]);
                }
                Instruction copy;
                copy.opcode = Opcode::addi;
                copy.dst = place;
                copy.src0 = victim;
                copy.imm = 0;
                copy.line = inst.line;
                copy.text = "addi " + percent(scratch_entry) + ", " + percent(victim_entry) + ", 0";
                std::string advice;
                if (invariant) {
                    advice = "; it is inside a loop that does not write " + percent(victim_entry) + ": copying it into "
                        "a second name before the loop, and reading that name here, avoids it";
                } else if (loop.has_value()) {
                    advice = "; it is inside a loop that writes " + percent(victim_entry) + " and " + beside +
                        ", so a copy made before the loop would not hold their values here: the copy stays";
                }
                program.warnings.push_back(
                    "line " + std::to_string(inst.line) + ": '" + inst.text + "' reads " + percent(victim_entry) +
                    " and " + beside + " from one register-file bank; qtxc inserted a copy of " +
                    percent(victim_entry) + " before it (one more instruction each time it runs" + advice + ")");
                rebuilt.push_back(std::move(copy));
                rebuilt_flags.push_back(0);
                rebuilt_notes.push_back("qtxc: a copy of " + percent(victim_entry) + ", which shares a bank with " +
                    beside + " in the next instruction");
                ++program.copies;
                // The instruction reads the scratch where its issue read the victim.
                const auto slots = register_slots(inst);
                for (std::size_t slot = 1; slot < 4; ++slot) {
                    if (slots[slot].kind == Kind::vector && slots[slot].width == 1 && slots[slot].read &&
                        field(inst, slot) == victim) {
                        field(inst, slot) = place;
                    }
                }
                inst.text = replace_source(inst, percent(victim_entry), percent(scratch_entry));
            }
            rebuilt.push_back(std::move(inst));
            rebuilt_flags.push_back(program.named_flags[index]);
            rebuilt_notes.push_back(std::string());
        }
        if (program.copies != 0) {
            for (std::size_t index = 0; index < instructions.size(); ++index) {
                if (target[index] < 0) continue;
                // The branch is the last of its instruction's new ones.
                const auto self = (index + 1 < instructions.size() ? new_index[index + 1] : rebuilt.size()) - 1;
                const auto to = new_index[static_cast<std::size_t>(target[index])];
                rebuilt[self].imm = static_cast<int32_t>((static_cast<int64_t>(to) - static_cast<int64_t>(self) - 1) * 8);
            }
            for (auto& label : program.labels) {
                if (label.instruction_index < new_index.size()) label.instruction_index = new_index[label.instruction_index];
            }
            instructions = std::move(rebuilt);
            program.named_flags = std::move(rebuilt_flags);
            program.notes = std::move(rebuilt_notes);
        }

        // Banks into the file the numbering sees (scratches included).
        auto file = make_file(program, false);
        for (uint32_t node = 0; node < file.nodes.size(); ++node) {
            const auto place = static_cast<uint8_t>(file.nodes[node].first);
            if (const auto found = scratch_banks.find(place); found != scratch_banks.end()) {
                file.nodes[node].bank = static_cast<int32_t>(found->second);
            } else if (node < state.bank.size()) {
                file.nodes[node].bank = state.bank[node];
            }
        }

        // 3. Liveness. A vector write is partial when the instruction is
        // masked, and -- since the active mask may be partial anywhere in a
        // kernel that uses the divergence stack -- everywhere in such a
        // kernel; a copy's scratch is written and read under one mask, so it
        // is whole.
        bool diverges = false;
        for (const auto& inst : instructions) diverges = diverges || uses_divergence_stack(inst);
        std::vector<Slots> slots;
        for (const auto& inst : instructions) slots.push_back(register_slots(inst));
        const auto places_of = [&](std::size_t index, Kind kind, bool written) {
            std::vector<std::pair<uint32_t, bool>> result;
            const auto& inst = instructions[index];
            for (std::size_t slot = 0; slot < 4; ++slot) {
                const auto& entry = slots[index][slot];
                if (entry.kind != kind || (written ? !entry.written : !entry.read)) continue;
                for (uint32_t j = 0; j < entry.width; ++j) {
                    const uint32_t place = uint32_t{field(inst, slot)} + j;
                    bool whole = true;
                    if (written && kind == Kind::vector) {
                        const bool masked = has_write_mask(inst) && can_be_masked(inst.opcode);
                        whole = program.vector_registers[place].scratch || !(masked || diverges);
                    }
                    result.emplace_back(place, whole);
                }
            }
            return result;
        };
        const auto run = [&](File& f, Kind kind, uint32_t soft, uint32_t hard, const std::string& capacity) {
            const auto uses = [&](std::size_t index) {
                std::vector<uint32_t> result;
                for (const auto& [place, whole] : places_of(index, kind, false)) result.push_back(place);
                return result;
            };
            const auto defines = [&](std::size_t index) { return places_of(index, kind, true); };
            const auto live = liveness(instructions, f.registers->size(), uses, defines);
            number_file(f, interference(f, live), banks, soft, hard, capacity);
        };

        // 4. Numbers. The vector file's usual room: what every warp can have
        // when the kernel's warps (`.warps`, else every warp slot) share it.
        const uint32_t warps = program.warps != 0 ? program.warps : profile.core_warps;
        const uint32_t usual = std::min<uint32_t>(256, banks * (profile.rf_rows_per_bank / std::max<uint32_t>(warps, 1)));
        run(file, Kind::vector, usual, 256, "the 256 a kernel can name");
        auto uniform = make_file(program, true);
        run(uniform, Kind::uniform, profile.urf_registers, profile.urf_registers,
            "this profile's " + std::to_string(profile.urf_registers) + " (urf.registers)");

        // Predicates with a name (%odd): each gets one of p0 .. p6 that the
        // kernel does not name by number, its own while there are enough,
        // shared otherwise between names that never hold a value at the
        // same time (the same liveness; a masked setp, and any predicate
        // write in a kernel that uses the divergence stack, keeps the old
        // value alive in the lanes it skips).
        std::vector<uint8_t> predicate_number(program.predicate_names.size(), 0);
        if (!program.predicate_names.empty()) {
            const auto count = program.predicate_names.size();
            std::array<bool, 7> named{};
            for (const auto& inst : instructions) {
                if (inst.control.predicate < 7) named[inst.control.predicate] = true;
                if (const auto dst = predicate_destination(inst); dst.has_value() && *dst < 7) named[*dst] = true;
            }
            const auto uses = [&](std::size_t index) {
                std::vector<uint32_t> result;
                if (instructions[index].control.predicate >= 8) result.push_back(instructions[index].control.predicate - 8u);
                return result;
            };
            const auto defines = [&](std::size_t index) {
                std::vector<std::pair<uint32_t, bool>> result;
                const auto& inst = instructions[index];
                if (const auto dst = predicate_destination(inst); dst.has_value() && *dst >= 8) {
                    const bool masked = has_write_mask(inst) && can_be_masked(inst.opcode);
                    result.emplace_back(*dst - 8u, !(masked || diverges));
                }
                return result;
            };
            File predicates;
            predicates.nodes.resize(count);
            predicates.node_of.resize(count);
            for (uint32_t place = 0; place < count; ++place) {
                predicates.nodes[place].first = place;
                predicates.node_of[place] = place;
            }
            const auto interferes = interference(predicates, liveness(instructions, count, uses, defines));
            std::vector<int32_t> given(count, -1);
            uint32_t free_predicates = 0;
            for (const auto taken : named) free_predicates += taken ? 0 : 1;
            for (uint32_t place = 0; place < count; ++place) {
                for (const bool share : {false, true}) {
                    for (uint8_t candidate = 0; candidate < 7 && given[place] < 0; ++candidate) {
                        if (named[candidate]) continue;
                        bool fits = true;
                        for (uint32_t other = 0; other < place; ++other) {
                            if (given[other] == candidate && (!share || interferes[place][other])) fits = false;
                        }
                        if (fits) given[place] = candidate;
                    }
                }
                if (given[place] < 0) {
                    const auto& entry = program.predicate_names[place];
                    throw SimError("line " + std::to_string(entry.line) + ": no predicate is left for " + percent(entry) +
                        ": more predicate values are held at once than the " + std::to_string(free_predicates) +
                        " of p0..p6 the kernel does not name by number");
                }
                predicate_number[place] = static_cast<uint8_t>(given[place]);
            }
        }

        // The numbers into the fields and the text, and the table.
        const auto number = [&](const File& f, uint32_t place) {
            const auto& node = f.nodes[f.node_of[place]];
            return static_cast<uint8_t>(static_cast<uint32_t>(node.base) + place - node.first);
        };
        std::vector<bool> vector_used(program.vector_registers.size(), false);
        std::vector<bool> uniform_used(program.uniform_registers.size(), false);
        std::map<std::string, std::string> written;
        for (std::size_t place = 0; place < program.vector_registers.size(); ++place) {
            written[program.vector_registers[place].name] = "r" + std::to_string(number(file, static_cast<uint32_t>(place)));
        }
        for (std::size_t place = 0; place < program.uniform_registers.size(); ++place) {
            written[program.uniform_registers[place].name] =
                "ur" + std::to_string(number(uniform, static_cast<uint32_t>(place)));
        }
        for (uint8_t p = 0; p < 7; ++p) written["p" + std::to_string(p)] = "p" + std::to_string(p); // %p0 is p0
        for (std::size_t place = 0; place < program.predicate_names.size(); ++place) {
            written[program.predicate_names[place].name] = "p" + std::to_string(predicate_number[place]);
        }
        static const std::regex name_pattern(R"(%([A-Za-z_][A-Za-z0-9_]*))");
        for (std::size_t index = 0; index < instructions.size(); ++index) {
            auto& inst = instructions[index];
            if (inst.control.predicate >= 8) inst.control.predicate = predicate_number[inst.control.predicate - 8u];
            if (const auto dst = predicate_destination(inst); dst.has_value() && *dst >= 8) {
                inst.dst = predicate_number[*dst - 8u];
            }
            for (std::size_t slot = 0; slot < 4; ++slot) {
                const auto& entry = slots[index][slot];
                if (entry.kind == Kind::none) continue;
                const auto place = field(inst, slot);
                const bool is_uniform = entry.kind == Kind::uniform;
                for (uint32_t j = 0; j < entry.width; ++j) {
                    (is_uniform ? uniform_used : vector_used)[place + j] = true;
                }
                field(inst, slot) = number(is_uniform ? uniform : file, place);
            }
            std::string text;
            auto begin = inst.text.cbegin();
            for (std::sregex_iterator it(inst.text.begin(), inst.text.end(), name_pattern), end; it != end; ++it) {
                text.append(begin, inst.text.cbegin() + it->position());
                text += written.at((*it)[1].str());
                begin = inst.text.cbegin() + it->position() + it->length();
            }
            text.append(begin, inst.text.cend());
            inst.text = std::move(text);
            check_rf_read_round(inst, profile, "internal: qtxc's register numbers, line " + std::to_string(inst.line));
        }
        for (std::size_t place = 0; place < program.vector_registers.size(); ++place) {
            if (!vector_used[place]) continue;
            program.register_names.push_back(
                {program.vector_registers[place].name, false, number(file, static_cast<uint32_t>(place))});
        }
        for (std::size_t place = 0; place < program.uniform_registers.size(); ++place) {
            if (!uniform_used[place]) continue;
            program.register_names.push_back(
                {program.uniform_registers[place].name, true, number(uniform, static_cast<uint32_t>(place))});
        }
        for (std::size_t place = 0; place < program.predicate_names.size(); ++place) {
            program.register_names.push_back({program.predicate_names[place].name, false, predicate_number[place], true});
        }
    }
}

} // namespace qtxc_internal
} // namespace nanosm

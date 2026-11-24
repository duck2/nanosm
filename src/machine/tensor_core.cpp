// The tensor core: the MMA engine on port B, tensor memory, and the accumulator transfers.
// (Machine is one class, src/machine/machine.hpp; the map of its files is at the top of
// src/machine/machine.cpp.)

#include "machine/machine.hpp"
#include "machine/mma_numeric.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

namespace nanosm {

// ===========================================================================
// MMA engine (port B) and accumulator transfers
// ===========================================================================

namespace {

// An operand element's byte address, in the profile's layout through the
// layout rule the TMA places tiles with (src/layout.hpp); the same rule for
// A and B. Under swizzle64 both operands are rows of k -- A m rows, B n rows
// -- `spacing` bytes apart (the instruction's), and `base` is the operand's
// first 16-byte piece in its row 0 (tile + 16 x piece: a logical position,
// not a shuffled one). Element (row, k) is at the logical address base + row
// x spacing + 2k, shuffled by that address's own row number (address /
// spacing, src/layout.hpp): a K = 8 command reads one piece of every row; a
// shape with k > 8 reads k / 8 consecutive pieces of the same rows. The
// address generator walks the rows in order and adds the spacing once per
// row (an adder, no multiplier); `row * spacing` here is the simulator's
// shorthand for that running sum, and the row number is the running
// address's bits above the spacing (wires, no counter). An operand may start
// at any row of a taller tile. The dense diagnostic layout is row-major and
// compact in the command's shape: A m rows of k, B k rows of n (no spacing).
uint32_t mma_a_address(const Config& config, const MmaShape& shape, uint32_t base, uint32_t spacing,
    uint32_t m, uint32_t k) {
    if (config.mma_layout == TileLayout::dense) return dense_address(base, m, k, shape.k, 2);
    return swizzle64_shuffle(base + m * spacing + 2 * k, spacing);
}

uint32_t mma_b_address(const Config& config, const MmaShape& shape, uint32_t base, uint32_t spacing,
    uint32_t k, uint32_t n) {
    if (config.mma_layout == TileLayout::dense) return dense_address(base, k, n, shape.n, 2);
    return swizzle64_shuffle(base + n * spacing + 2 * k, spacing);
}

uint32_t mma_operand_element(const Config& config, const MmaShape& shape, uint32_t base, uint32_t spacing,
    bool a_operand, uint32_t outer, uint32_t k) {
    return a_operand ? mma_a_address(config, shape, base, spacing, outer, k)
                     : mma_b_address(config, shape, base, spacing, k, outer);
}

// The bytes an operand of `shape` reads, as [first, end): the hull of every
// element it reads (under swizzle64 a later row's piece can sit below or
// above its logical place, so the ends come from the addresses themselves).
std::pair<uint64_t, uint64_t> mma_operand_hull(const Config& config, const MmaShape& shape, uint32_t base,
    uint32_t spacing, bool a_operand) {
    uint64_t first = std::numeric_limits<uint64_t>::max();
    uint64_t end = 0;
    const uint32_t outer = a_operand ? shape.m : shape.n;
    for (uint32_t i = 0; i < outer; ++i) {
        for (uint32_t k = 0; k < shape.k; ++k) {
            const uint64_t address = mma_operand_element(config, shape, base, spacing, a_operand, i, k);
            first = std::min(first, address);
            end = std::max(end, address + 2);
        }
    }
    return {first, end};
}

uint32_t mma_total_steps(const Config& config, const MmaShape& shape) {
    return (shape.m / config.mma_core_m) * (shape.n / config.mma_core_n) * (shape.k / config.mma_core_k);
}

} // namespace

uint32_t mma_operand_address(const Config& config, const MmaShape& shape, uint32_t base, uint32_t spacing,
    bool a_operand, uint32_t outer, uint32_t k) {
    return mma_operand_element(config, shape, base, spacing, a_operand, outer, k);
}

// The acc.ld (load) or acc.st engine, as this cycle leaves it (look-ahead
// (c)): free if it runs no transfer. The MMA never matters here -- it has
// tmem services of its own; a transfer racing MMA
// work is the sanitizer's business.
bool Machine::acc_engine_available(const Wires& w, bool load) const {
    return !w.acc_next[load ? 0 : 1].has_value();
}

// The MMA engine's cycle, computed on a copy of its queue and pipeline:
// issue looks ahead at the queue entries it frees (c), so the next state is a
// wire; drive_mma commits it with issue's new command. Its tmem reads (C) and
// writes (results) use its own tmem services; the writes land at this
// cycle's edge (w.mma_tmem_writes).
void Machine::step_mma(Wires& w) {
    auto mma = mma_.cur();
    advance_mma(mma, w);
    w.mma_next = std::move(mma);
}

// One cycle of the MMA engine, applied to `mma` (a copy of its Q):
//   1. the steps in the core: one whose C read is at the tmem RAM now reads
//      it for the core; one whose result is in the core's output register
//      now leaves for tmem; one whose result is at the tmem RAM now writes it
//      (at this cycle's edge) and retires; retired commands leave the queue
//      head;
//   2. the port-B read pipeline: the round at the RAM samples its elements;
//      the round at the port output adds them to the core's operand latch,
//      and a step's last round enters the core (C is read from tmem then);
//   3. port B grants one round: the next round of the step in service, else
//      a new step's first round (or the head commit's presentation);
//   4. the read pipeline moves one stage; the granted round enters stage 0.
// A step granted its last round at g samples it at g + D and enters the core
// at g + D + Lb (the same D and Lb as port A).
void Machine::advance_mma(MmaPipeline& mma, Wires& w) {
    if (const auto& at_c = mma_tags_.stage(config_.tmem_port_depth - 1); at_c.has_value()) read_mma_c(*at_c);
    if (mma_tags_.last().has_value() != mma_core_.output().has_value()) {
        throw SimError("internal: an MMA step's tag and its result left the core apart");
    }
    if (const auto& tag = mma_tags_.last(); tag.has_value()) w.mma_write_entering = MmaWrite{*tag, *mma_core_.output()};
    if (const auto& landing = mma_writes_.last(); landing.has_value()) {
        const auto& tag = landing->tag;
        const auto words = mma_block_words(tag);
        for (std::size_t i = 0; i < words.size(); ++i) w.mma_tmem_writes.emplace_back(words[i], landing->values[i]);
        trace_->mma_result(cycle_, tag.command, tag.c_pointer + tag.m0 * tag.row_words + tag.n0);
        const auto id = tag.command;
        auto command = std::find_if(mma.slots.begin(), mma.slots.end(),
            [id](const auto& candidate) { return candidate.id == id; });
        if (command == mma.slots.end() || command->outstanding_results == 0)
            throw SimError("internal MMA result lost its command");
        --command->outstanding_results;
    }
    while (!mma.slots.empty() && !mma.slots.front().commit &&
           mma.slots.front().operands_consumed && mma.slots.front().outstanding_results == 0) {
        sanitizer_.record_mma_retired(mma.slots.front().id); // last result retired
        mma.slots.pop_front();
    }
    if (const auto& at_ram = mma_reads_.stage(config_.shmem_port_a_depth - 1); at_ram.has_value()) {
        w.mma_read_sampled = *at_ram;
        sample_mma_read(mma, *w.mma_read_sampled, w);
    }
    if (const auto& output = mma_reads_.last(); output.has_value()) {
        for (std::size_t element = 0; element < mma_step_elements; ++element) {
            if (output->in_round[element]) mma.latch[element] = output->values[element];
        }
        if (output->last_round) enter_mma_core(mma, output->step, w);
    }
    if (!mma.fetch.has_value()) mma.fetch = next_mma_step(mma, w);
    if (mma.fetch.has_value()) {
        w.mma_read_granted = grant_mma_round(mma);
        if (mma.fetch->next_round == mma.fetch->rounds) mma.fetch.reset();
    }
}

// Element `element` of a step (MmaFetch): its shared-memory byte address, or
// none for an A element of a step that uses the held A block.
std::optional<uint32_t> Machine::mma_element_address(const MmaStep& step, std::size_t element) const {
    const std::size_t a_elements = std::size_t{config_.mma_core_m} * config_.mma_core_k;
    if (element < a_elements) {
        if (!step.reload_a) return std::nullopt;
        const auto mi = static_cast<uint32_t>(element / config_.mma_core_k);
        const auto kk = static_cast<uint32_t>(element % config_.mma_core_k);
        return mma_a_address(config_, step.shape, step.a_base, step.a_spacing, step.m0 + mi, step.k0 + kk);
    }
    const auto at = element - a_elements;
    const auto kk = static_cast<uint32_t>(at / config_.mma_core_n);
    const auto ni = static_cast<uint32_t>(at % config_.mma_core_n);
    return mma_b_address(config_, step.shape, step.b_base, step.b_spacing, step.k0 + kk, step.n0 + ni);
}

// A step's elements: core_m x core_k of A and core_k x core_n of B.
std::size_t Machine::mma_elements() const {
    return std::size_t{config_.mma_core_m} * config_.mma_core_k + std::size_t{config_.mma_core_k} * config_.mma_core_n;
}

// The next step port B may start now, if any; or the head commit presents
// into the MMA's event register. Steps walk hold-A / stream-B: for each K
// block, for each A block (m), every B block (n) in order; the
// first step of each A block reads A and B, the rest read B only and use the
// held A block. A command's steps may start once the previous command's last
// step has been granted (commands overlap in the pipelines).
std::optional<Machine::MmaFetch> Machine::next_mma_step(MmaPipeline& mma, Wires& w) {
    auto command_it = std::find_if(mma.slots.begin(), mma.slots.end(), [this](const auto& command) {
        return command.commit || command.next_step < mma_total_steps(config_, command.shape);
    });
    if (command_it == mma.slots.end()) return std::nullopt;
    auto& command = *command_it;
    if (command.commit) {
        // A commit at the queue head presents into the MMA's own event
        // register: free if empty in Q, or claimed by the RMW engine now (d).
        const bool register_free = !mma_mbar_event_.cur().has_value() || w.shmem.mma_event_claimed;
        if (command_it != mma.slots.begin() || !register_free) return std::nullopt;
        MbarRmwRequest request;
        request.operation = MbarRmwOperation::arrive;
        request.byte_address = command.mbar_address;
        w.mma_present = std::move(request);
        mma.slots.pop_front();
        return std::nullopt;
    }
    // The command's shape sets the walk: rows in core_m blocks (8 for m32,
    // 16 for m64), the n span in core_n blocks, k in core_k steps.
    const uint32_t blocks_n = command.shape.n / config_.mma_core_n;
    const uint32_t blocks_per_k = (command.shape.m / config_.mma_core_m) * blocks_n;
    const uint32_t total_steps = mma_total_steps(config_, command.shape);
    MmaStep step;
    const uint32_t k_block = command.next_step / blocks_per_k;
    const uint32_t block = command.next_step % blocks_per_k;
    const uint32_t m_block = block / blocks_n;
    const uint32_t n_block = block % blocks_n;
    step.command = command.id;
    step.c_pointer = command.c_pointer;
    step.m0 = m_block * config_.mma_core_m;
    step.n0 = n_block * config_.mma_core_n;
    step.k0 = k_block * config_.mma_core_k;
    step.a_base = command.a_base;
    step.b_base = command.b_base;
    step.a_spacing = command.a_spacing;
    step.b_spacing = command.b_spacing;
    step.shape = command.shape;
    step.init_step = command.init && step.k0 == 0;
    step.reload_a = n_block == 0;
    step.last_step = command.next_step + 1 == total_steps;
    // A step whose tile is still in the core or on its way to tmem (a short
    // sweep) waits: its C must be the earlier step's result. Tiles are told
    // apart by the tmem words they write (any shared word: the same tile, or
    // two commands' overlapping results), so pointers may alias freely.
    const auto words = mma_block_words(mma_tag(step));
    const auto shares_words = [&](const MmaTag& other) {
        const auto other_words = mma_block_words(other);
        return std::any_of(other_words.begin(), other_words.end(), [&](uint32_t word) {
            return std::find(words.begin(), words.end(), word) != words.end();
        });
    };
    // In flight: the steps in the core (their tags; the one entering now
    // too), the results on their way to tmem whose write has not landed (the
    // one in the write pipeline's last stage lands at this edge), and the
    // rounds in the read pipeline.
    const auto& tags = mma_tags_.cur();
    const auto& writes = mma_writes_.cur();
    const auto& reads = mma_reads_.cur();
    const bool tile_in_flight =
        std::any_of(tags.begin(), tags.end(), [&](const std::optional<MmaTag>& tag) {
            return tag.has_value() && shares_words(*tag);
        }) ||
        std::any_of(writes.begin(), writes.end() - 1, [&](const std::optional<MmaWrite>& write) {
            return write.has_value() && shares_words(write->tag);
        }) ||
        (w.mma_tag_entering.has_value() && shares_words(*w.mma_tag_entering)) ||
        std::any_of(reads.begin(), reads.end(), [&](const std::optional<MmaRead>& read) {
            return read.has_value() && shares_words(mma_tag(read->step));
        });
    if (tile_in_flight) {
        trace_->mma_recurrence_stall();
        return std::nullopt; // explicit recurrence bubble for a profile whose sweep is too short
    }
    // Bank rounds: round r reads every bank's r-th distinct word, in
    // ascending order; each element is in its word's round. (At least one
    // round: a step reads B.)
    MmaFetch fetch;
    fetch.step = step;
    Bounded<uint32_t, mma_step_elements> distinct; // the step's distinct words
    for (std::size_t element = 0; element < mma_elements(); ++element) {
        if (const auto address = mma_element_address(step, element); address.has_value()) {
            if (std::find(distinct.begin(), distinct.end(), *address / 4) == distinct.end()) distinct.push_back(*address / 4);
        }
    }
    for (std::size_t element = 0; element < mma_elements(); ++element) {
        const auto address = mma_element_address(step, element);
        if (!address.has_value()) continue;
        const uint32_t word = *address / 4;
        const auto round = static_cast<uint32_t>(std::count_if(distinct.begin(), distinct.end(), [&](uint32_t other) {
            return other % config_.shmem_banks == word % config_.shmem_banks && other < word;
        }));
        fetch.element_round[element] = static_cast<uint8_t>(round);
        fetch.rounds = std::max(fetch.rounds, round + 1);
    }
    ++command.next_step;
    ++command.outstanding_results; // retires at its tmem write
    sanitizer_.record_mma_fetch_start(command.id);
    trace_->mma_input(cycle_, command.id, fetch.rounds, command.next_step - 1, step.reload_a);
    return fetch;
}

// Port B grants the step in service its next round: the elements in it.
Machine::MmaRead Machine::grant_mma_round(MmaPipeline& mma) {
    auto& fetch = *mma.fetch;
    MmaRead read;
    read.step = fetch.step;
    for (std::size_t element = 0; element < mma_elements(); ++element) {
        read.in_round[element] = mma_element_address(fetch.step, element).has_value() &&
            fetch.element_round[element] == fetch.next_round;
    }
    ++fetch.next_round;
    read.last_round = fetch.next_round == fetch.rounds;
    return read;
}

// A round at the RAM reads its elements from shmem as it is now, before this
// cycle's writes land at the clock edge (READ_FIRST). The
// command's last step's last round is its final physical operand read: the
// operands are consumed and its depbar clears here.
void Machine::sample_mma_read(MmaPipeline& mma, MmaRead& read, Wires& w) {
    for (std::size_t element = 0; element < mma_elements(); ++element) {
        if (!read.in_round[element]) continue;
        const auto address = *mma_element_address(read.step, element);
        if (static_cast<uint64_t>(address) + 2 > shmem_bytes())
            throw SimError("MMA operand footprint is outside shmem");
        // Its two bytes, from the word(s) port B reads in their banks now (a
        // round reads one word per bank; its elements share it).
        uint16_t bits = 0;
        for (uint32_t byte = 0; byte < 2; ++byte) {
            const auto at = shmem_word(address + byte);
            const auto bytes = shmem_.read(at.bank, at.row, "port-B read sample", shmem_port_b);
            bits = static_cast<uint16_t>(bits | uint32_t{bytes[at.byte]} << (8 * byte));
        }
        read.values[element] = bits;
    }
    if (!read.last_round || !read.step.last_step) return;
    const auto id = read.step.command;
    auto command = std::find_if(mma.slots.begin(), mma.slots.end(),
        [id](const auto& candidate) { return candidate.id == id; });
    if (command == mma.slots.end()) throw SimError("internal MMA read lost its command");
    command->operands_consumed = true;
    sanitizer_.record_mma_operands_consumed(command->id); // final operand read
    record_depbar_completion(w, command->warp, command->depbar, "internal MMA depbar underflow");
}

// A step's operands are all at the port output: it enters the core. A
// reloading step's A block replaces the held one; the others use it. The
// core's black box takes A and B now; the step's tag enters the tag pipeline, and
// its C read leaves for tmem with it (at the RAM tmem.port_depth later,
// read_mma_c); its result is in the core's output register result_latency
// after entry and lands in tmem tmem.port_depth after that.
void Machine::enter_mma_core(MmaPipeline& mma, const MmaStep& step, Wires& w) {
    const std::size_t a_elements = std::size_t{config_.mma_core_m} * config_.mma_core_k;
    if (step.reload_a) mma.a_held.assign(mma.latch.begin(), mma.latch.begin() + static_cast<std::ptrdiff_t>(a_elements));
    if (mma.a_held.size() != a_elements) throw SimError("internal MMA step found no held A block");
    MmaCoreStep entering;
    entering.entered = cycle_;
    entering.a = mma.a_held;
    // B is transposed here only in the core's own copy (the latch holds it
    // k-major).
    entering.b.resize(std::size_t{config_.mma_core_n} * config_.mma_core_k);
    for (uint32_t ni = 0; ni < config_.mma_core_n; ++ni)
        for (uint32_t kk = 0; kk < config_.mma_core_k; ++kk)
            entering.b[ni * config_.mma_core_k + kk] = mma.latch[a_elements + kk * config_.mma_core_n + ni];
    mma_core_.inside().push_back(entering);
    w.mma_tag_entering = mma_tag(step);
}

Machine::MmaTag Machine::mma_tag(const MmaStep& step) {
    return MmaTag{step.command, step.c_pointer, step.m0, step.n0, step.shape.n, step.init_step};
}

// A step's C read is at the tmem RAM now: its sixteen words are read (one
// round of the MMA's tmem read service; READ_FIRST, before this cycle's
// writes land), or zero for a step that starts its tile. The core takes
// them at tmem's output, tmem.read_latency_cycles later; the simulator hands
// them over now, as it carries every read through a memory's delay.
void Machine::read_mma_c(const MmaTag& tag) {
    MmaBlock c(std::size_t{config_.mma_core_m} * config_.mma_core_n, 0u);
    if (!tag.init_step) {
        const auto words = mma_block_words(tag);
        for (std::size_t i = 0; i < words.size(); ++i) c[i] = read_tmem(words[i], tmem_port_mma, "MMA C read");
    }
    finish_mma_step(c);
}

// Inside the core's black box: the oldest step's A and B have waited for
// this C; each of its core_m x core_n words is the fused dot-plus-C of its
// row of A and column of B (mma_numeric.cpp), computed at once and delivered
// for the black box's output register result_latency after the step entered.
void Machine::finish_mma_step(const MmaBlock& c) {
    auto& waiting = mma_core_.inside();
    if (waiting.empty() || waiting.front().entered + config_.tmem_port_depth != cycle_) {
        throw SimError("internal: an MMA C read found no step waiting for it");
    }
    const auto& step = waiting.front();
    MmaBlock values(std::size_t{config_.mma_core_m} * config_.mma_core_n, 0u);
    for (uint32_t mi = 0; mi < config_.mma_core_m; ++mi) {
        for (uint32_t ni = 0; ni < config_.mma_core_n; ++ni) {
            values[mi * config_.mma_core_n + ni] = mma_dot_f16(
                std::span<const uint16_t>(step.a).subspan(mi * config_.mma_core_k, config_.mma_core_k),
                std::span<const uint16_t>(step.b).subspan(ni * config_.mma_core_k, config_.mma_core_k),
                c[mi * config_.mma_core_n + ni]);
        }
    }
    mma_core_.deliver(values, step.entered, "the MMA core");
    waiting.pop_front();
}

// The core's arithmetic as a black box (src/machine/blackbox.hpp). It takes a
// step's A and B blocks at its entry and its C at tmem's output, W =
// tmem.port_depth + tmem.read_latency_cycles later, and has the result in its
// output register L = mma.result_latency_cycles after entry; a step can enter
// every cycle. L > W (validate_device_profile refuses a profile that breaks
// it), so a clock edge is left for the arithmetic after C is at tmem's
// output. (Its inside: enter_mma_core, read_mma_c and finish_mma_step
// above, the dot products in mma_numeric.cpp.)
void Machine::declare_mma_core_black_box() {
    const auto& c = config_;
    const uint64_t m = c.mma_core_m;
    const uint64_t n = c.mma_core_n;
    const uint64_t k = c.mma_core_k;
    const uint64_t wait = uint64_t{c.tmem_port_depth} + c.tmem_read_latency_cycles;
    BlackBoxDeclaration black_box("MMA core");
    black_box.unit = Unit::tensor_core;
    black_box.latency = c.mma_result_latency_cycles;
    black_box.latency_from = "mma.result_latency_cycles; C at entry + tmem.port_depth + tmem.read_latency_cycles = " +
        std::to_string(wait);
    const std::string operands = "the MMA's operand latch and the port-B read pipeline's last stage (in the open)";
    black_box.inputs = {BlackBoxPort{"A block (core_m x core_k fp16)", m * k * 16, operands},
        BlackBoxPort{"B block (core_k x core_n fp16)", k * n * 16, operands},
        BlackBoxPort{"C (core_m x core_n words), at entry + " + std::to_string(wait), m * n * 32,
            "tensor memory's MMA read registers (the memory's)"},
        BlackBoxPort{"C is zero (the tile's first step)", 1, "MMA core: step tags (in the open)"}};
    black_box.outputs = {BlackBoxPort{"result block (core_m x core_n words)", m * n * 32 + 1,
        mma_core_.name() + " (in the open)"}};
    mma_core_.build(black_box, [](const std::deque<MmaCoreStep>& waiting) { return waiting.empty(); });
    black_boxes_.push_back(std::move(black_box));
}

// The tmem writes at the RAM now: the MMA's result block through each fine
// bank's write port 0, an acc.st beat through write port 1 (one word per fine
// bank each). They land at the clock edge, the acc.st beat's after the MMA's.
void Machine::land_tmem_writes(const Wires& w) {
    for (const auto* writes : {&w.mma_tmem_writes, &w.acc_st_tmem_writes}) {
        const bool mma = writes == &w.mma_tmem_writes;
        for (const auto& [word, value] : *writes) {
            tmem_.write(word % config_.tmem_fine_banks, word / config_.tmem_fine_banks,
                std::span<const uint32_t>(&value, 1), Memory<uint32_t>::every_element,
                mma ? "MMA result write" : "acc.st write", mma ? tmem_port_mma : tmem_port_acc);
        }
    }
}

std::size_t Machine::tmem_index(uint64_t logical, uint32_t row_words) const {
    const uint64_t index = tmem_physical_word(logical, row_words);
    if (index >= tmem_words()) throw SimError("internal tmem layout overflow");
    return static_cast<std::size_t>(index);
}

std::optional<std::vector<uint32_t>> Machine::tmem_region_words(
    uint32_t pointer, uint32_t row_words, uint32_t rows, uint32_t columns) const {
    std::vector<uint32_t> words;
    words.reserve(static_cast<std::size_t>(rows) * columns);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t column = 0; column < columns; ++column) {
            const uint64_t physical = tmem_physical_word(
                uint64_t{pointer} + uint64_t{row} * row_words + column, row_words);
            if (physical >= tmem_words()) return std::nullopt;
            words.push_back(static_cast<uint32_t>(physical));
        }
    }
    return words;
}

// The tmem words of a matmul step's core_m x core_n result block, row-major:
// rows m0.. and columns n0.. of the command's result region (rows of n words).
std::vector<uint32_t> Machine::mma_block_words(const MmaTag& tag) const {
    std::vector<uint32_t> words;
    words.reserve(config_.mma_core_m * config_.mma_core_n);
    for (uint32_t mi = 0; mi < config_.mma_core_m; ++mi)
        for (uint32_t ni = 0; ni < config_.mma_core_n; ++ni)
            words.push_back(static_cast<uint32_t>(tmem_index(
                uint64_t{tag.c_pointer} + uint64_t{tag.m0 + mi} * tag.row_words + tag.n0 + ni, tag.row_words)));
    return words;
}

// The two accumulator transfer engines (one acc.ld
// engine, one acc.st engine for the whole machine, each with its own tmem
// service in every fine bank), computed on copies of their registers: issue
// looks ahead at them (c) and may give a freed engine its next transfer now
// (d). The acc.ld pipeline before the acc.st beat.
void Machine::step_acc_transfers(Wires& w) {
    auto engines = acc_engines_.cur();
    step_acc_ld(engines[0], w);
    step_acc_st(engines[1], w);
    w.acc_next = std::move(engines);
}

// The acc.ld engine's pipeline. The beat at the tmem RAM stage samples its
// words now (once).
// Its output asks the RF write decision for service; this cycle's grant was
// decided at the top of the cycle.
//   output empty, or granted -> advance: every stage moves one toward the
//                               output; stage 0 takes the engine's next read
//                               if its transfer has a beat left, else stays
//                               empty
//   output valid and denied  -> hold: no stage moves, nothing launches
// A granted beat belongs to the RF write path from its grant on.
void Machine::step_acc_ld(std::optional<AccTransfer>& engine, Wires& w) {
    if (const auto& at_ram = acc_ld_pipe_.stage(config_.tmem_port_depth - 1); at_ram.has_value() && !at_ram->sampled) {
        w.acc_ld_sampled = *at_ram;
        sample_acc_ld_beat(*w.acc_ld_sampled);
    }
    const bool output_valid = acc_ld_pipe_.last().has_value();
    if (output_valid && !w.rf_write.granted_to(AsyncWriter::acc_ld, 0)) {
        w.acc_ld_hold = true;
        return;
    }
    w.acc_ld_launch = launch_acc_ld_beat(engine);
}

// The engine's next read leaves for tmem now: the transfer's next column, all
// lanes. The engine frees with its last read's launch; each beat carries its
// own destination and context down the pipeline.
std::optional<Machine::AccLdBeat> Machine::launch_acc_ld_beat(std::optional<AccTransfer>& engine) {
    if (!engine.has_value()) return std::nullopt;
    auto& transfer = *engine;
    AccLdBeat beat;
    beat.warp = transfer.warp;
    beat.reg = static_cast<uint8_t>(transfer.rf_base + transfer.next_column);
    beat.address = transfer.pointer + transfer.next_column;
    beat.row_words = transfer.row_words;
    beat.last = transfer.next_column + 1 == transfer.width;
    beat.depbar = transfer.depbar;
    ++transfer.next_column;
    if (beat.last) engine.reset();
    return beat;
}

// A beat at the tmem RAM: its sixteen words (sixteen distinct fine banks) are
// read now, as tmem is before this cycle's writes land (READ_FIRST). The
// transfer's last sample ends its tmem read-ownership.
void Machine::sample_acc_ld_beat(AccLdBeat& beat) {
    beat.values.resize(config_.lanes);
    for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
        const auto word = tmem_index(uint64_t{beat.address} + uint64_t{lane} * beat.row_words, beat.row_words);
        beat.values[lane] = read_tmem(static_cast<uint32_t>(word), tmem_port_acc, "acc.ld sample");
    }
    beat.sampled = true;
    trace_->tmem_beat(cycle_, beat.warp, true, beat.address);
    if (beat.last) sanitizer_.record_acc_transfer_final_beat(true);
}

// The acc.st engine's transfer: its beats are read from the RF one per cycle
// when the RF read decision grants them (decide_rf_reads: sampled at the
// grant, at the RF output Lr later, at the tmem RAM tmem.port_depth after
// that); a beat at the tmem RAM now is written at this cycle's edge. The last
// beat's write ends the transfer: its depbar clears, its tmem write-ownership
// ends and the engine frees.
void Machine::step_acc_st(std::optional<AccTransfer>& engine, Wires& w) {
    if (!engine.has_value()) return;
    auto& transfer = *engine;
    if (const auto& at_ram = acc_st_pipe_.last(); at_ram.has_value()) {
        const auto& beat = *at_ram;
        const uint64_t address = uint64_t{transfer.pointer} + beat.column;
        for (uint32_t lane = 0; lane < config_.lanes; ++lane) {
            w.acc_st_tmem_writes.emplace_back(
                static_cast<uint32_t>(tmem_index(address + uint64_t{lane} * transfer.row_words, transfer.row_words)),
                beat.values[lane]);
        }
        trace_->tmem_beat(cycle_, transfer.warp, false, static_cast<uint32_t>(address));
        ++transfer.landed;
    }
    if (transfer.landed != transfer.width) return;
    record_depbar_completion(w, transfer.warp, transfer.depbar, "internal acc.st depbar underflow");
    // final tmem write: write-ownership ends
    sanitizer_.record_acc_transfer_final_beat(false);
    engine.reset();
}

// What a transfer touches, for the sanitizer: the region and its tmem words.
TmemRegion Machine::tmem_region(uint32_t pointer, const Instruction& inst, std::vector<uint32_t> words) const {
    TmemRegion region;
    region.pointer = pointer;
    region.row_words = static_cast<uint32_t>(inst.imm);
    region.rows = config_.lanes;
    region.columns = inst.opcode == Opcode::acc_ld_v8 || inst.opcode == Opcode::acc_st_v8 ? 8 : 1;
    std::sort(words.begin(), words.end());
    region.words = std::move(words);
    return region;
}

void Machine::dispatch_mma(const DispatchEntry& op, Wires& w) {
    const auto& inst = op.instruction;
    if (inst.opcode == Opcode::mma_commit) {
        MmaCommand command;
        command.id = mma_command_number_.cur();
        command.commit = true;
        command.warp = op.warp;
        command.mbar_address = mbar_address_operand(op);
        w.dispatch.mma = std::move(command);
        return;
    }
    // The shape is the profile's (its list gives each opcode pair a shape);
    // its existence was checked at kernel load (check_kernel).
    const auto shape = *mma_shape_of(config_, inst.opcode);
    const auto a = uniform_operand(op, inst.src0);
    const auto b = uniform_operand(op, inst.src1);
    // urC, from the round counter's round 1 (src/timing.hpp uniform_read_rounds):
    // the result's tmem pointer, its rows the shape's n words. Every word it
    // writes must be in tmem (the machine's PANIC, as an operand outside
    // shmem); a pointer that breaks the alignment rule is the sanitizer's
    // (src/layout.hpp tmem_pointer_aligned). The machine writes whatever
    // words the layout names.
    const auto c = uniform_operand(op, inst.src2);
    auto c_words = tmem_region_words(c, shape.n, shape.m, shape.n);
    if (!c_words.has_value()) panic(op, "MMA result region is outside tmem");
    sanitizer_.check_tmem_pointer(site(op), opcode_name(inst.opcode, &config_.mma_shapes) + " result", c,
        shape.n, shape.n);
    // The spacings are instruction constants, checked at kernel load
    // (check_kernel): under swizzle64 a power of two, 16 or more.
    const auto a_spacing = mma_row_spacing(inst, true);
    const auto b_spacing = mma_row_spacing(inst, false);
    // A swizzle64 operand starts at a 16-byte piece. Every byte it reads must
    // be in shmem: the hull of the addresses the walk reads (under swizzle64
    // a row's piece moves inside its row, so not simply base + rows x spacing).
    const auto inside = [&](uint32_t base, uint32_t spacing, bool a_operand) {
        return base < shmem_bytes() &&
            mma_operand_hull(config_, shape, base, spacing, a_operand).second <= shmem_bytes();
    };
    if ((config_.mma_layout == TileLayout::swizzle64 && ((a | b) & 15u) != 0) ||
        !inside(a, a_spacing, true) || !inside(b, b_spacing, false))
        panic(op, "MMA operand alignment or footprint is outside shmem");
    // The machine reads whatever the address-keyed rule names; an operand
    // whose row-0 bytes are not inside one legally placed row is the
    // sanitizer's PANIC (src/layout.hpp: a row-0 line on a multiple of the
    // spacing and of 64).
    if (config_.mma_layout == TileLayout::swizzle64) {
        sanitizer_.check_swizzle64_operand(site(op), "A", a, a_spacing, 2 * shape.k);
        sanitizer_.check_swizzle64_operand(site(op), "B", b, b_spacing, 2 * shape.k);
    }
    MmaCommand command;
    command.id = mma_command_number_.cur();
    command.init = is_mma_init(inst.opcode);
    command.shape = shape;
    command.warp = op.warp;
    command.c_pointer = c;
    command.depbar = inst.control.set_depbar;
    command.a_base = a;
    command.b_base = b;
    command.a_spacing = a_spacing;
    command.b_spacing = b_spacing;
    TmemRegion result;
    result.pointer = c;
    result.row_words = shape.n;
    result.rows = shape.m;
    result.columns = shape.n;
    std::sort(c_words->begin(), c_words->end());
    result.words = std::move(*c_words);
    if constexpr (NANOSM_SANITIZER != 0) {
        // The shmem words each operand reads, for the row spacing check
        // (computed for the sanitizer only). The dense diagnostic layout
        // reads no spacing: none.
        const auto operand_words = [&](uint32_t base, uint32_t spacing, bool a_operand) {
            std::vector<uint32_t> words;
            if (config_.mma_layout != TileLayout::swizzle64) return words;
            for (uint32_t outer = 0; outer < (a_operand ? shape.m : shape.n); ++outer) {
                for (uint32_t k = 0; k < shape.k; ++k) {
                    words.push_back(mma_operand_address(config_, shape, base, spacing, a_operand, outer, k) / 4);
                }
            }
            std::sort(words.begin(), words.end());
            words.erase(std::unique(words.begin(), words.end()), words.end());
            return words;
        };
        sanitizer_.record_mma_row_spacing(site(op), command.id, command.init, result,
            operand_words(a, a_spacing, true), a_spacing, operand_words(b, b_spacing, false), b_spacing);
    }
    sanitizer_.record_mma_accept(site(op), command.id, result, command.depbar,
        mma_operand_span(a, a_spacing, true, shape), mma_operand_span(b, b_spacing, false, shape));
    trace_->mma_command(cycle_, command.id, op.warp, command.init);
    w.dispatch.mma = std::move(command);
}

// acc.ld / acc.st: the pointer arrives now; the transfer runs one beat per
// column in its engine (step_acc_transfers, and decide_rf_reads for acc.st):
// lane l moves row l of the region at the pointer (rows of S words, the
// instruction's), beat j column j. Every word it touches must be in tmem
// (the machine's PANIC); the alignment rule is the sanitizer's
// (src/layout.hpp tmem_pointer_aligned).
void Machine::dispatch_acc_transfer(const DispatchEntry& op, Wires& w) {
    const auto& inst = op.instruction;
    const bool load = is_acc_load(inst.opcode);
    const uint32_t width = inst.opcode == Opcode::acc_ld_v8 || inst.opcode == Opcode::acc_st_v8 ? 8 : 1;
    const uint8_t rf_base = load ? inst.dst : inst.src1;
    const auto row_words = static_cast<uint32_t>(inst.imm);
    const auto pointer = uniform_operand(op, inst.src0);
    auto words = tmem_region_words(pointer, row_words, config_.lanes, width);
    if (!words.has_value() || static_cast<unsigned>(rf_base) + width > kernel_.register_count) {
        panic(op, "accumulator transfer region outside tmem or RF range out of bounds");
    }
    const auto region = tmem_region(pointer, inst, std::move(*words));
    sanitizer_.check_tmem_pointer(site(op), opcode_name(inst.opcode), pointer, row_words, width);
    // Racing accepted MMA work (check_issue_hazards): an MMA command issued
    // before this transfer was accepted by the MMA before this dispatch
    // (dispatch keeps issue order).
    sanitizer_.check_acc_transfer_vs_mma(site(op), region);
    sanitizer_.check_acc_transfer_overlap(site(op), region, load);
    AccTransfer transfer;
    transfer.load = load;
    transfer.warp = op.warp;
    transfer.rf_base = rf_base;
    transfer.depbar = inst.control.set_depbar;
    transfer.pointer = pointer;
    transfer.row_words = row_words;
    transfer.width = width;
    sanitizer_.record_acc_transfer_accept(site(op), load, region, inst.control.set_depbar);
    w.dispatch.acc = DispatchDecision::AccAdmission{load, std::move(transfer)};
}

// The MMA's event register: a claim and a new commit's presentation may
// happen in one cycle (d).
void Machine::drive_mma_event(const Wires& w) {
    if (!w.shmem.mma_event_claimed && !w.mma_present.has_value()) return;
    auto next = w.shmem.mma_event_claimed ? std::nullopt : mma_mbar_event_.cur();
    if (w.mma_present.has_value()) {
        if (next.has_value()) throw SimError("internal MMA event presented into a full register");
        next = w.mma_present;
    }
    mma_mbar_event_.set_next(std::move(next), "drive_mma_event");
}

// The MMA queue and pipeline: step_mma's result, plus issue's new command.
void Machine::drive_mma(const Wires& w) {
    auto next = w.mma_next;
    if (w.dispatch.mma.has_value()) {
        next.slots.push_back(*w.dispatch.mma);
        mma_command_number_.set_next(w.dispatch.mma->id + 1, "drive_mma");
    }
    if (next.slots.size() > config_.mma_queue_commands) {
        throw SimError("internal: the tensor core holds more than mma.queue_commands commands");
    }
    mma_.set_next(std::move(next), "drive_mma");
    if (w.mma_read_sampled.has_value()) {
        mma_reads_.replace(config_.shmem_port_a_depth - 1, w.mma_read_sampled, "drive_mma: the shmem RAM's elements");
    }
    if (w.mma_read_granted.has_value()) mma_reads_.push(*w.mma_read_granted, "drive_mma: the granted round");
    if (w.mma_tag_entering.has_value()) mma_tags_.push(*w.mma_tag_entering, "drive_mma: a step enters the core");
    if (w.mma_write_entering.has_value()) mma_writes_.push(*w.mma_write_entering, "drive_mma: a result leaves the core");
}

// The two acc transfer engines: step_acc_transfers' result (acc.ld launches,
// acc.st landings), plus this cycle's acc.st RF read (the beat enters the
// acc.st pipeline) and dispatch's new transfer (into an engine freed this
// cycle, possibly).
void Machine::drive_acc_transfers(const Wires& w) {
    auto next = w.acc_next;
    if (const auto& beat = w.acc_st_read; beat.has_value()) {
        auto& engine = next[1];
        if (!engine.has_value()) throw SimError("internal acc.st beat read with no acc.st transfer");
        acc_st_pipe_.push(*beat, "drive_acc_transfers: acc.st beat");
        ++engine->next_column;
    }
    if (const auto& admission = w.dispatch.acc; admission.has_value()) {
        auto& engine = next[admission->load ? 0 : 1];
        if (engine.has_value()) throw SimError("internal acc transfer engine given two transfers");
        engine = admission->transfer;
    }
    acc_engines_.set_next(std::move(next), "drive_acc_transfers");
}

// The acc.ld pipeline: step_acc_transfers' advance or hold. The beat at the
// tmem RAM stage carries its words from this edge on, held or not.
void Machine::drive_acc_ld_pipe(const Wires& w) {
    if (w.acc_ld_sampled.has_value()) {
        acc_ld_pipe_.replace(config_.tmem_port_depth - 1, w.acc_ld_sampled, "drive_acc_ld_pipe: the tmem words");
    }
    if (w.acc_ld_hold) {
        acc_ld_pipe_.hold("drive_acc_ld_pipe: the output is denied");
    } else if (w.acc_ld_launch.has_value()) {
        acc_ld_pipe_.push(*w.acc_ld_launch, "drive_acc_ld_pipe");
    }
}

std::optional<std::pair<uint32_t, uint32_t>> Machine::mma_operand_span(
    uint32_t base, uint32_t spacing, bool a_operand, const MmaShape& shape) const {
    // Ask the operand geometry itself (the same functions the fetch uses):
    // the hull of every byte the command's shape reads. Under swizzle64 a
    // K = 8 operand reads one 16-byte piece per row, but the hull covers
    // the rows' other pieces too (a conservative owner).
    const auto [first, last] = mma_operand_hull(config_, shape, base, spacing, a_operand);
    if (last > std::numeric_limits<uint32_t>::max()) return std::nullopt;
    return std::pair<uint32_t, uint32_t>{static_cast<uint32_t>(first), static_cast<uint32_t>(last)};
}

} // namespace nanosm

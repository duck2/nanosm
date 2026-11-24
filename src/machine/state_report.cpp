// The state report: every primitive's declared hardware bits (declare_state_bits) and the
// report of them by unit (state_report).
// (Machine is one class, src/machine/machine.hpp; the map of its files is at the top of
// src/machine/machine.cpp.)

#include "machine/machine.hpp"

#include <algorithm>
#include <cstdio>
#include <sstream>

namespace nanosm {

// ===========================================================================
// Declared bits. Each payload's bits follow from the architecture and the
// profile's sizes, not from what C++ stores: a field with a maximum and a
// count (Bounded) holds the profile's count, a lane mask is core.lanes bits,
// an address is as wide as the memory it addresses, a register number as wide
// as its file. Fields marked (S) (read only for a PANIC or its text) or (T)
// (only for the trace) in machine.hpp count zero, and so do fields nothing
// reads (named below). An optional payload has one valid bit; a pipeline
// stage's valid bit is the pipeline's own (pipe.hpp). Q and D are one
// register: each is counted once.
// ===========================================================================

namespace {

// A number with its thousands grouped: 1,442,878.
std::string number(uint64_t value) {
    std::string digits = std::to_string(value);
    for (auto at = static_cast<std::ptrdiff_t>(digits.size()) - 3; at > 0; at -= 3) digits.insert(at, ",");
    return digits;
}

// ===========================================================================
// What the black boxes cost. A cost is not part of a black box's contract and
// changes nothing the machine does; every cost is a first-order estimate that
// depends on the implementation. It is costed for an FPGA in a consumer's
// reach (7 series, UltraScale+): flip-flops of the fabric, memory bits and
// basic DSPs. A basic DSP is the plain hard arithmetic block every family in
// reach has in some form: about one 25 x 18 multiply with an add and its own
// registers (a 24-bit significand product takes two, an 11-bit one one);
// registers a basic DSP or a block RAM has of its own are not flip-flops of
// the fabric. Logic is not counted: take a device's LUTs to be about its
// flip-flops. A profile that stands for an ASIC has no use for these.
// The profile's cost section gives a unit's cost as plain numbers; a unit it
// leaves out takes the default guess below. A memory's size is no estimate:
// its words, from its shape.
// ===========================================================================

using Cost = Machine::BlackBoxCost;

// A result's tag inside a unit: warp, destination, lane enables, a valid bit.
uint64_t result_tag(const Config& c) { return bits_for(c.core_warps) + 8 + c.lanes + 1; }

// ALU: per lane the add, the logic ops and the compare sit in a basic DSP
// whose own registers hold the result; the barrel shifter does not, so each
// stage after the first holds every lane's word, the tag and the setp bit.
Cost guess_alu(const Config& c) {
    return {(uint64_t{c.timing.alu} - 1) * (uint64_t{c.lanes} * 32 + result_tag(c) + 1), 0, c.lanes};
}

// FPU: per lane a single-precision fused multiply-add (fadd, fmul and fmax
// are its special cases; cvt.pack is not counted separately), its 24 x 24
// product on two basic DSPs, about 64 bits a lane and the tag a stage.
Cost guess_fpu(const Config& c) {
    return {(uint64_t{c.timing.fadd} - 1) * (uint64_t{c.lanes} * 64 + result_tag(c)), 0, 2 * uint64_t{c.lanes}};
}

// Special-function unit: per lane served, a 128-entry table of three
// coefficients (64 bits) per function and a quadratic on two basic DSPs; the
// flip-flops are the G - 1 waiting operand groups and result groups, the
// Ls - 1 function stages (lanes, valid, group) and the tags inside.
Cost guess_sfu(const Config& c) {
    const uint64_t groups = (c.lanes + c.sfu_lanes - 1) / c.sfu_lanes;
    const uint64_t inside = (c.timing.sfu - 1 + groups - 1) / groups;
    return {2 * (groups - 1) * c.sfu_lanes * 32 +
            (c.sfu_latency_cycles - 1) * (uint64_t{c.sfu_lanes} * 64 + 1 + bits_for(groups)) + inside * result_tag(c),
        uint64_t{c.sfu_lanes} * 2 * 128 * 64, 2 * uint64_t{c.sfu_lanes}};
}

// MMA core: a basic DSP per product of a step. Until C is at the core (W =
// tmem.port_depth + read_latency) its stages hold A and B; from C on, per
// result word, the k products (28 bits) and C (27) with the window's 8-bit
// exponent, summed by a tree halving the terms a stage; past the one sum a
// stage only delays the packed 32-bit word (so a long latency adds 32 a word).
Cost guess_mma_core(const Config& c) {
    const uint64_t m = c.mma_core_m;
    const uint64_t n = c.mma_core_n;
    const uint64_t k = c.mma_core_k;
    const uint64_t inner = c.mma_result_latency_cycles - 1;
    const uint64_t before_c = std::min<uint64_t>(uint64_t{c.tmem_port_depth} + c.tmem_read_latency_cycles, inner);
    uint64_t after_c = 0;
    uint64_t terms = k + 1;
    bool summed = false;
    for (uint64_t stage = 0; before_c + stage < inner; ++stage) {
        uint64_t word = 32;
        if (!summed && terms > 1) {
            word = (stage == 0 ? k * 28 + 27 : terms * (28 + stage)) + 8;
        } else if (!summed) {
            word = 28 + stage + 8;
            summed = true;
        }
        after_c += m * n * word;
        terms = (terms + 1) / 2;
    }
    return {before_c * (m * k + k * n) * 16 + after_c, 0, m * n * k};
}

struct CostedUnit {
    const char* black_box;
    std::optional<BlackBoxCostEntry> BlackBoxCosts::*entry;
    uint64_t lanes; // what a per-lane amount is multiplied by
    Cost (*guess)(const Config&);
};

} // namespace

Machine::BlackBoxCost Machine::black_box_cost(const BlackBoxDeclaration& black_box) const {
    if (black_box.memory.has_value()) {
        const auto& shape = *black_box.memory;
        return {0, uint64_t{shape.banks} * shape.rows * shape.word_bits, 0, "its words"};
    }
    const CostedUnit units[] = {{"ALU", &BlackBoxCosts::alu, config_.lanes, guess_alu},
        {"FPU", &BlackBoxCosts::fpu, config_.lanes, guess_fpu},
        {"special-function unit", &BlackBoxCosts::sfu, config_.sfu_lanes, guess_sfu},
        {"MMA core", &BlackBoxCosts::mma_core, 0, guess_mma_core}};
    for (const auto& unit : units) {
        if (black_box.name != unit.black_box) continue;
        const auto& entry = config_.costs.*(unit.entry);
        if (!entry.has_value()) {
            auto guess = unit.guess(config_);
            guess.source = "default guess";
            return guess;
        }
        return {entry->flip_flops + entry->flip_flops_per_lane * unit.lanes,
            entry->memory_bits + entry->memory_bits_per_lane * unit.lanes,
            entry->basic_dsps + entry->basic_dsps_per_lane * unit.lanes, "from the profile"};
    }
    throw SimError("internal: black box '" + black_box.name + "' has no cost (state_report.cpp)");
}

void Machine::declare_state_bits() {
    const auto& c = config_;
    const uint64_t lanes = c.lanes;
    const uint64_t warp = bits_for(c.core_warps);
    const uint64_t pc = 32;                  // a byte pc (src/timing.hpp)
    const uint64_t word = 64;                // an instruction word: its decoded fields come from it
    const uint64_t rf_register = 8;          // r0..r255
    const uint64_t uniform_register = bits_for(c.urf_registers);
    const uint64_t predicate = 3;            // p0..p6, PT
    const uint64_t depbar = 3;               // sb0..sb5
    const uint64_t shared = bits_for(c.shmem_bytes);
    const uint64_t shared_count = count_bits(c.shmem_bytes);
    const uint64_t global = bits_for(c.gmem_bytes);
    const uint64_t tmem_word = bits_for(uint64_t{c.tmem_fine_banks} * c.tmem_words_per_fine_bank);
    const uint64_t tmem_row_words = count_bits(bits_for(uint64_t{c.tmem_fine_banks} * c.tmem_words_per_fine_bank));
    const uint64_t shmem_row = bits_for(shmem_.rows());
    const uint64_t sector_bits = uint64_t{8} * c.gmem_sector_bytes;
    const uint64_t sector_words = c.gmem_sector_bytes / 4;
    const uint64_t lsu_slot = bits_for(c.lsu_active_commands);
    const uint64_t tma_slot = bits_for(c.tma_queue_commands);
    const uint64_t mma_slot = bits_for(c.mma_queue_commands);
    const uint64_t load_slot = bits_for(c.shmem_load_slots);
    const uint64_t vector_width = count_bits(c.tmem_max_vector_width);
    const uint64_t tma_rounds = count_bits(tma_max_rounds());
    const uint64_t credits = count_bits(c.tma_sector_credits);
    // An LSU command's requests: a lane's word each, or a v8's sectors.
    const uint64_t lsu_requests =
        count_bits(std::max<uint64_t>(lanes, uint64_t{8} * lanes * 4 / c.gmem_sector_bytes));

    // ---- payloads shared by several primitives ----
    // A run (TmaRun): slot, element 0's global and unshuffled shared address,
    // elements (1 .. a sector's), element size, zero fill, layout, transposed,
    // the spacing's log2.
    const uint64_t tma_run = tma_slot + global + shared + count_bits(c.gmem_sector_bytes) + 2 + 1 + 1 + 1 + 5;
    // MbarRmwRequest: operation, address, operand, warp (valid + warp), its
    // return load slot (valid + slot).
    const uint64_t rmw_request = 3 + shared + 32 + (1 + warp) + (1 + load_slot);
    // MbarReadRequest: test, warp, predicate, its wait slot (valid + slot),
    // address, parity.
    const uint64_t read_request = 1 + warp + predicate + (1 + bits_for(c.mbar_wait_slots)) + shared + 1;
    // A write round (WriteBeat): per bank a word, its byte enables and its row
    // (the bank is its place); an sts's last round (valid, warp, depbar); an
    // RMW's wake (valid, address, phase); a placement's end (valid, slot, the
    // run's end in shared memory).
    const uint64_t write_beat = uint64_t{c.shmem_banks} * (32 + 4 + shmem_row) + (1 + warp + depbar) +
        (1 + shared + 1) + (1 + tma_slot + shared_count);
    // A read round (ReadBeat): its client, slot and round, per bank a valid
    // row and the word read, and a probe's request (valid + request).
    // (Its round: a load slot's, at most one per lane, or a TMA read's.)
    const uint64_t read_beat = 3 + bits_for(std::max(c.shmem_load_slots, c.tma_store_sector_registers)) +
        bits_for(std::max<uint64_t>(lanes, tma_max_rounds())) + uint64_t{c.shmem_banks} * (1 + shmem_row + 32) +
        (1 + read_request);
    // UrfReadHeader: the dispatch stage its later round joins (the C++ keeps
    // the issue cycle), the warp, the instruction word.
    const uint64_t urf_header = bits_for(issue_to_dispatch(c)) + warp + word;
    // TmaWalker: the row steps (global, shared), x / y and their bounds,
    // the tile's length and rows (16 bits each), element size, layout,
    // transposed, the spacing's log2, row 0's global and shared address.
    const uint64_t tma_walker = global + (shared + 2) + 8 * 16 + 2 + 1 + 1 + 5 + global + shared;
    // TmaPortRequest: kind, run, rounds and rounds granted.
    const uint64_t tma_port_request = 2 + tma_run + 2 * tma_rounds;
    // MmaStep: the command's slot, the result pointer, last / init / reload,
    // the block's m0 / n0 / k0, both operands' base and row spacing, the shape.
    uint64_t largest_m = 0;
    uint64_t largest_n = 0;
    uint64_t largest_k = 0;
    for (const auto& shape : c.mma_shapes) {
        largest_m = std::max<uint64_t>(largest_m, shape.m);
        largest_n = std::max<uint64_t>(largest_n, shape.n);
        largest_k = std::max<uint64_t>(largest_k, shape.k);
    }
    const uint64_t mma_step = mma_slot + tmem_word + 3 + bits_for(largest_m) + bits_for(largest_n) +
        bits_for(largest_k) + 4 * shared + bits_for(c.mma_shapes.size());
    const uint64_t operand_words = 4; // a round's vector registers (limits::round_reads)
    const uint64_t uniform_operands = 4; // an entry's uniform registers: two rounds of the URF's two ports
    // A pipeline that carries what a memory read returns (the A-read and
    // port-B read from shared memory, the store-data and acc.st reads from
    // the RF, the TMA operand read from the URF): in hardware the words are
    // inside the memory until its read delay has passed, in its own read
    // registers (blackbox.hpp), so they count only in the stages after the one
    // where they are at its output -- none where the pipeline ends there.
    // `payload`: a stage's bits without the words; `at_output`: the stage the
    // words are at the memory's output in.
    const auto read_pipeline = [](auto& pipe, uint64_t payload, uint64_t word_bits, uint64_t at_output,
                                   const std::string& memory) {
        const uint64_t stages = pipe.stages();
        const uint64_t past_delay = stages > at_output + 1 ? (stages - 1 - at_output) * word_bits : 0;
        pipe.declare_total_bits(stages * (1 + payload) + past_delay);
        pipe.declare_note("its words count only in stages past " + memory + "'s read delay: " + number(past_delay) +
            " bits");
    };

    // ---- front end ----
    // A warp's front end: its packets (two instruction words each) and their
    // count, pc, fetch pc, halted.
    front_end_.declare_member_bits(uint64_t{c.core_instruction_buffer_packets} * 2 * word +
        count_bits(c.core_instruction_buffer_packets) + pc + pc + 1);
    fetch_read_.declare_bits(1 + warp + pc);
    fetch_cursor_.declare_bits(warp);
    // ---- branches ----
    issue_branch_.declare_bits(1 + warp + pc + word);
    divergence_refill_.declare_bits(1 + warp + bits_for(c.core_divergence_stack_entries));
    // ---- issue: depbar_ and issue_read_banks_ are declared where they are built ----
    // ---- barriers ----
    // A wait slot: valid, warp, address, parity, predicate, parked, and its
    // deadline as a count of the try-wait timeout (the C++ keeps a cycle).
    mbar_wait_slots_.declare_member_bits(
        1 + warp + shared + 1 + predicate + 1 + count_bits(c.mbar_try_wait_timeout_cycles));
    mbar_probe_.declare_bits(1 + read_request);
    simt_rmw_.declare_bits(1 + rmw_request);
    tma_mbar_event_.declare_bits(1 + rmw_request);
    mma_mbar_event_.declare_bits(1 + rmw_request);
    // The operation in flight (valid, request, read granted), the engine
    // served last, the class started last.
    rmw_.declare_bits((1 + rmw_request + 1) + 1 + 1);
    // An update: the word written back (an mbar's 64 bits; an atomic's 32 in
    // the low half) and a phase flip's wake (valid + phase). Its address and
    // kind are the operation's, in the engine's register.
    rmw_alu_.declare_payload_bits(64 + 1 + 1);
    // ---- shared memory ----
    // A slot's access (SlotRounds): the lanes that access, per lane its
    // address and its round, the rounds and the rounds granted.
    const uint64_t slot_rounds = lanes + lanes * (shared + bits_for(lanes)) + 2 * count_bits(lanes);
    // A load slot: valid, warp, uniform, lane mask, its access, destination,
    // depbar, the gather register, rounds back, engine-filled.
    load_slots_.declare_member_bits(
        1 + warp + 1 + lanes + slot_rounds + rf_register + depbar + lanes * 32 + count_bits(lanes) + 1);
    // A store slot: valid, warp, its access, the lanes' words, depbar.
    store_slots_.declare_member_bits(1 + warp + slot_rounds + lanes * 32 + depbar);
    // A-read: its words are sampled at the RAM (stage D - 1) and at the port
    // output Lb later, its last stage.
    read_pipeline(a_read_pipe_, read_beat - uint64_t{c.shmem_banks} * 32, uint64_t{c.shmem_banks} * 32,
        c.shmem_port_a_depth - 1 + c.shmem_read_latency_cycles, "shared memory");
    a_write_pipe_.declare_payload_bits(write_beat);
    b_write_pipe_.declare_payload_bits(write_beat);
    // A-read's pointer over the load slots, the probe and the TMA; A-write's
    // over the store slots; the alternation bit.
    shmem_arbiters_.declare_bits(bits_for(c.shmem_load_slots + 2) + bits_for(c.shmem_store_slots) + 1);
    // ---- tensor core ----
    // A step's operand elements (A and B) and its bank rounds: at most one
    // distinct word per element.
    const uint64_t mma_elements = uint64_t{c.mma_core_m} * c.mma_core_k + uint64_t{c.mma_core_k} * c.mma_core_n;
    // A port-B round: its step, last round, which of the step's elements it
    // reads, per bank a valid row and the word (sampled at the RAM, stage
    // D - 1; at the port output, the last stage).
    read_pipeline(mma_reads_, mma_step + 1 + mma_elements + uint64_t{c.shmem_banks} * (1 + shmem_row),
        uint64_t{c.shmem_banks} * 32, c.shmem_port_a_depth - 1 + c.shmem_read_latency_cycles, "shared memory");
    // The MMA queue and pipeline. A command: commit, init, shape, warp, result
    // pointer, depbar, the operands' bases and row spacings, its barrier,
    // steps started and results outstanding (at most its steps), operands
    // consumed. The slots (valid + command each), their count and the head:
    // the slots are a ring in age order, so a command's slot (a result's name
    // for it; the C++ keeps its number) is its place in the ring, not a field.
    // The fetch (valid, step, rounds, next round, each element's round); the
    // operand latch (an fp16 per element); the held A block.
    uint64_t largest_steps = 0;
    for (const auto& shape : c.mma_shapes) {
        largest_steps = std::max<uint64_t>(largest_steps,
            uint64_t{shape.m / c.mma_core_m} * (shape.n / c.mma_core_n) * (shape.k / c.mma_core_k));
    }
    const uint64_t mma_command = 1 + 1 + bits_for(c.mma_shapes.size()) + warp + tmem_word + depbar +
        4 * shared + shared + 2 * count_bits(largest_steps) + 1;
    mma_.declare_bits(uint64_t{c.mma_queue_commands} * (1 + mma_command) + count_bits(c.mma_queue_commands) + mma_slot +
        (1 + mma_step + 2 * count_bits(mma_elements) + mma_elements * bits_for(mma_elements)) + mma_elements * 16 +
        uint64_t{c.mma_core_m} * c.mma_core_k * 16);
    // A step's tag beside the core: its command's slot, its result block's
    // place (the result pointer, m0, n0, the shape), C is zero. With the
    // result on its way to tmem: the same place (C is not read again) and the
    // block's words. (The core's output register is its black box's, tensor_core.cpp.)
    const uint64_t block = uint64_t{c.mma_core_m} * c.mma_core_n;
    const uint64_t block_place = mma_slot + tmem_word + bits_for(largest_m) + bits_for(largest_n) +
        bits_for(c.mma_shapes.size());
    mma_tags_.declare_payload_bits(block_place + 1);
    mma_writes_.declare_payload_bits(block_place + block * 32);
    // Each engine: valid, load, warp, register base, depbar, pointer, words
    // per row (its log2), width, columns launched, beats landed.
    acc_engines_.declare_bits(
        2 * (1 + 1 + warp + rf_register + depbar + tmem_word + tmem_row_words + 3 * vector_width));
    // A beat: warp, register, address, words per row, sampled, last, depbar;
    // and the lanes' words from the tmem RAM stage (port_depth - 1) on. Not
    // the read-delay rule: the pipeline holds while its output waits for the
    // RF, and a held beat keeps its words in flip-flops of its own (the read
    // at the RAM in the cycle a hold begins replaces what tmem's read
    // registers held), so every stage from the sample on counts them.
    {
        const uint64_t stages = acc_ld_pipe_.stages();
        const uint64_t words = (stages - (c.tmem_port_depth - 1)) * lanes * 32;
        acc_ld_pipe_.declare_total_bits(stages * (1 + warp + rf_register + tmem_word + tmem_row_words + 1 + 1 + depbar) +
            words);
        acc_ld_pipe_.declare_note("its words count from the tmem RAM stage on (a held beat keeps them): " +
            number(words) + " bits");
    }
    // A beat: its column (0 .. width - 1), the lanes' words (read at its
    // grant, at the RF's output in stage Lr - 1, then tmem.port_depth stages
    // to the tmem RAM).
    read_pipeline(acc_st_pipe_, bits_for(c.tmem_max_vector_width), lanes * 32, c.rf_read_latency_cycles - 1,
        "the RF");
    // ---- TMA ----
    // A command: slot, direction, descriptor address, four coordinates, the
    // buffer and the barrier, state, logical bytes, pending placements and
    // captures, walker done, launch, launch bytes landed. (Its warp, pc and
    // instruction word are written and never read: zero.)
    const uint64_t tma_command = tma_slot + 1 + shared + 4 * 32 + shared + shared + 3 + 32 + credits + credits + 1 +
        1 + shared_count;
    // The slots (valid + command each), their age order and its count, the
    // frontend's and the event's command (valid + slot; the C++ keeps their
    // numbers), the credits used, the walker.
    tma_.declare_bits(uint64_t{c.tma_queue_commands} * (1 + tma_command) + uint64_t{c.tma_queue_commands} * tma_slot +
        count_bits(c.tma_queue_commands) + 2 * (1 + tma_slot) + credits + tma_walker);
    tma_walker_start_pipe_.declare_payload_bits(tma_slot + tma_walker);
    tma_write_txn_.declare_bits(1 + tma_port_request);
    // The transaction, its rounds back, its store register, the frontend
    // latch (the descriptor's 64 bytes).
    tma_read_txn_.declare_bits(1 + tma_port_request + tma_rounds + bits_for(c.tma_store_sector_registers) +
        8 * tma_descriptor_bytes);
    // The sector register: valid (the C++ keeps the request's number), its
    // sector address, data, bytes in, complete.
    tma_sector_.declare_bits(1 + global + sector_bits + count_bits(c.gmem_sector_bytes) + 1);
    tma_returns_.declare_entry_bits(tma_run);
    // A store register: valid, run, rounds, data, rounds back, words out.
    tma_store_registers_.declare_member_bits(1 + tma_run + tma_rounds + sector_bits + tma_rounds +
        count_bits(sector_words));
    tma_store_head_.declare_bits(bits_for(c.tma_store_sector_registers));
    // A pair: its round, last, round 0's header (valid + header), the two
    // words (sampled before stage 0, at the URF's output in the last stage)
    // and their count.
    read_pipeline(tma_read_pairs_, 2 + 1 + (1 + urf_header) + 2, 2 * 32, c.urf_read_latency_cycles - 1, "the URF");
    // The holding register: valid, header, buffer, barrier, four coordinates.
    tma_holding_.declare_bits(1 + urf_header + 32 + 32 + 4 * 32);
    tma_reads_.declare_entry_bits(tma_run);
    tma_read_received_.declare_bits(bits_for(c.gmem_sector_bytes));
    // ---- uniform register file ----
    // The round counter: valid, header, the next round.
    urf_read_rounds_.declare_bits(1 + urf_header + bits_for(tma_operand_rounds(tma_max_coordinates)));
    // A write: warp, lane mask, instruction word, its operands' count, the
    // value (valid + 32). Its two operand latches count zero: the uniform ALU
    // takes its operands at the URF's output (stage Lu) and they go no
    // further, and r2ur's lane 0 is its value from the sample on.
    urf_pipe_.declare_payload_bits(warp + lanes + word + 2 + (1 + 32));
    urf_pipe_.declare_note("its operands are taken at the URF's output: no latches");
    // ---- load / store unit ----
    // A command: slot, kind, warp, register, depbar, lanes, a sector address,
    // lane and word offset per lane and their count, requests total and
    // emitted, lanes written, a v8 store's beats read, sectors pushed, words
    // pushed (a v8 store's sector words, or a scalar store's lanes),
    // acknowledgements. (Its pc and instruction word are written and never
    // read, its buffer is the trace's, its stream a constant: zero.)
    const uint64_t lsu_command = lsu_slot + 3 + warp + rf_register + depbar + lanes +
        lanes * (global + bits_for(lanes) + bits_for(sector_words)) + count_bits(lanes) + 3 * lsu_requests +
        count_bits(8) + count_bits(lanes) + count_bits(std::max(sector_words, lanes)) + lsu_requests;
    lsu_commands_.declare_bits(uint64_t{c.lsu_active_commands} * (1 + lsu_command) +
        uint64_t{c.lsu_active_commands} * lsu_slot + count_bits(c.lsu_active_commands));
    // A buffer: owner (valid + slot), group, the group's lanes, its words and
    // their valid bits, valid and expected counts, beats and beats total.
    const auto buffer_bits = [&](uint64_t words) {
        return (1 + lsu_slot) + bits_for(lanes) + lanes + words * 32 + words + 2 * count_bits(words) + 2 * count_bits(8);
    };
    lsu_buffers_.declare_total_bits(buffer_bits(8 * uint64_t{c.lsu_load_group_lanes}) +
        buffer_bits(std::max<uint64_t>(8 * uint64_t{c.lsu_store_group_lanes}, lanes)));
    // The open egress slot: valid, client, owner, request index, sector
    // address, byte enables, words to write and written. Its words are in the
    // egress RAM at the write pointer already (egress_data_): zero here.
    egress_open_.declare_bits(1 + 1 + lsu_slot + lsu_requests + global + c.gmem_sector_bytes +
        2 * count_bits(sector_words));
    egress_requests_.declare_entry_bits(1 + lsu_slot + lsu_requests + global + c.gmem_sector_bytes);
    // Its second entry is read only by the LSU's request-wait counter (an
    // observer): the hardware's read port is one entry wide.
    egress_requests_.declare_read_width(1);
    lsu_reads_.declare_entry_bits(2 + lsu_slot + bits_for(lanes) + bits_for(sector_words));
    lsu_read_received_.declare_bits(bits_for(c.gmem_sector_bytes));
    // A store-data read: its command (valid + slot), buffer, group, beat, the
    // group's lanes; and the register's lanes (read at its grant, at the RF's
    // output in its last stage).
    read_pipeline(lsu_read_pipe_, (1 + lsu_slot) + 1 + bits_for(lanes) + 3 + lanes, lanes * 32,
        c.rf_read_latency_cycles - 1, "the RF");
    // ---- register file ----
    // Its policies: the owed bit and a cursor over the async writers (the
    // load slots, the LSU's two buffers, the acc.ld pipeline) or readers
    // (acc.st, the LSU's store data).
    rf_write_policy_.declare_member_bits(1 + bits_for(c.shmem_load_slots + 2 + 1));
    rf_read_policy_.declare_member_bits(1 + bits_for(2));
    // A write port's write: its row in the bank, the lanes' words and their
    // enables, the depbar it counts down (valid + depbar) and that depbar's
    // warp.
    for (auto& pipe : rf_write_pipes_) {
        pipe.declare_payload_bits(bits_for(c.rf_rows_per_bank) + lanes * 32 + lanes + (1 + depbar) + warp);
    }
    // ---- dispatch and arithmetic: the operands in flight ----
    // An entry samples its registers in a stage s and the simulator carries
    // their words from stage s + 1 on. In hardware the words are inside the
    // memory until its read delay has passed: at its output (register) in
    // stage s + delay. So the stages up to that one hold no flip-flops for
    // them -- they are the memory's: what its read ports have in flight, at
    // most banks x read ports x word bits x read delay, whoever reads (the
    // memory's line says so) -- and a stage after it holds them: a latch of
    // real flip-flops (the uniform operands when the URF's delay is shorter
    // than the RF's; the vector operands' holding stage where the dispatch
    // waits for a slower uniform round), as wide as the words the path takes
    // in that round. The held bits of `words` words of `bits` each:
    const auto held = [](uint64_t stages, uint64_t sampled_at, uint64_t delay, uint64_t words, uint64_t bits) {
        const uint64_t last = stages - 1;
        const uint64_t at_output = sampled_at + delay;
        return (last > at_output ? last - at_output : 0) * words * bits;
    };
    // The most words each path's rounds take (issue_plan, uniform_read_rounds;
    // at most the ports): a dispatched instruction's vector round two (sts,
    // stg: address and data; stg.v8 reads only its address there, its data
    // through the LSU's store-data read; acc.st none), its uniform round 0
    // two, round 1 one (an mma.async's urC); an arithmetic instruction's
    // vector round three (fma), its uniform round one (ur2r).
    const uint64_t rf_ports = uint64_t{c.rf_banks} * c.rf_read_ports_per_bank;
    const uint64_t lane_bits = lanes * 32;
    // An entry: warp, pc, lane mask, instruction word, its round's registers
    // and their count, its uniform registers (two rounds) and their count;
    // then its operands' words as they are held: the vector round sampled in
    // stage 0, uniform round 0 in stage 0 and round 1 in stage 1.
    {
        const uint64_t stages = dispatch_pipe_.stages();
        const uint64_t words = held(stages, 0, c.rf_read_latency_cycles, std::min<uint64_t>(2, rf_ports), lane_bits) +
            held(stages, 0, c.urf_read_latency_cycles, std::min<uint64_t>(2, urf_read_ports), 32) +
            held(stages, 1, c.urf_read_latency_cycles, std::min<uint64_t>(1, urf_read_ports), 32);
        const uint64_t entry = warp + pc + lanes + word + operand_words * rf_register + count_bits(operand_words) +
            uniform_operands * uniform_register + count_bits(uniform_operands);
        dispatch_pipe_.declare_total_bits(stages * (1 + entry) + words);
        dispatch_pipe_.declare_note("its operands' words count only where they are held after the RF's and URF's "
            "read delays: " + number(words) + " bits");
    }
    // The operand latches: warp, lane mask, instruction word, the round's
    // registers and count, its uniform registers (one round) and count; the
    // words as they are held (sampled in stage 0; it executes from the last).
    {
        const uint64_t stages = arithmetic_reads_.stages();
        const uint64_t words = held(stages, 0, c.rf_read_latency_cycles, std::min<uint64_t>(3, rf_ports), lane_bits) +
            held(stages, 0, c.urf_read_latency_cycles, std::min<uint64_t>(1, urf_read_ports), 32);
        const uint64_t entry = warp + lanes + word + operand_words * rf_register + count_bits(operand_words) +
            2 * uniform_register + count_bits(2);
        arithmetic_reads_.declare_total_bits(stages * (1 + entry) + words);
        arithmetic_reads_.declare_note("its operands' words count only where they are held after the RF's and URF's "
            "read delays: " + number(words) + " bits");
    }
    // A memory's read data in flight: every read port's word for each cycle
    // of the read delay. (Its black box's report prints it: memory_blackbox.)
    const auto in_flight = [](auto& memory) {
        const auto& shape = memory.shape();
        memory.declare_note("read data in flight: at most " +
            number(uint64_t{shape.banks} * shape.read_ports * shape.word_bits * shape.read_delay) +
            " bits (banks x read ports x word bits x read delay), in its own read registers: the words its readers' "
            "pipelines carry inside the read delay are these");
    };
    in_flight(rf_);
    in_flight(urf_);
    in_flight(shmem_);
    in_flight(tmem_);
    instruction_memory_.declare_note("the kernel's words (no profile size)");
    // ---- launch ----
    // The launch record: valid, the block's global and shared address and
    // bytes, and per launch flag a valid bit, its depbar (a kernel's .flag
    // names any of sb0..sb5) and its byte count.
    launch_record_.declare_bits(
        1 + global + shared + shared_count + uint64_t{c.core_launch_flags} * (1 + depbar + shared_count));
}

// The black boxes (blackbox.hpp), each declared where it is defined: the
// arithmetic units (arith.cpp), the tensor core's
// arithmetic (tensor_core.cpp), and the memories, beside each in machine.hpp
// (their size is their words). Listed by unit.
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

// The primitives whose payload cannot declare honest bits yet (the list at
// the end of machine.hpp), each with what bounds it.
// There is none: every payload is fixed.
std::vector<std::pair<const Clocked*, std::string>> Machine::not_yet_declared() const {
    return {};
}

// ===========================================================================
// The report. Per unit: what is in the open -- its flip-flops (registers,
// groups, pipelines, a FIFO's pointers, the black boxes' output registers)
// and its FIFO storage, counted -- and, apart, its black boxes (the memories
// among them) with their costs. Then each black box's contract, and the
// totals: counted, behind walls, and their sum. Not in the totals: what
// cannot declare its bits yet, and plain state that is not a primitive.
// ===========================================================================

std::string Machine::state_report() const {
    std::ostringstream out;
    const auto sizes = clocked_.sizes();
    const auto missing = not_yet_declared();
    out << "state report: " << config_.name << " (declared hardware bits)\n";
    out << "In the open: registers, groups, pipelines, FIFOs and the black boxes' output registers, counted. Behind\n"
           "walls: black boxes, the memories among them, each a contract, with its cost: from the profile, a\n"
           "default guess, or a memory's words.\n";
    uint64_t total_flops = 0;
    uint64_t total_fifo = 0;
    for (auto unit = static_cast<uint8_t>(Unit::front_end); unit <= static_cast<uint8_t>(last_unit); ++unit) {
        const auto which = static_cast<Unit>(unit);
        uint64_t flops = 0;
        uint64_t fifo = 0;
        out << "\n" << unit_name(which) << "\n";
        for (const auto& size : sizes) {
            if (size.unit != which) continue;
            const std::string kind = size.kind;
            if (kind == "memory") continue; // a black box: below
            const auto bits = size.bits();
            char line[160];
            if (!bits.has_value()) {
                std::snprintf(line, sizeof(line), "  %-9s %-52s not yet declared\n", size.kind, size.name.c_str());
                out << line;
                continue;
            }
            std::string shape;
            if (kind == "FIFO") {
                shape = std::to_string(size.entries) + " x " + std::to_string(*size.bits_per_entry) + " bits, write " +
                    std::to_string(size.write_ports) + " / read " + std::to_string(size.read_ports) +
                    " entries wide, latency " + std::to_string(size.read_delay) + "; pointers " +
                    std::to_string(size.control_bits) + " flip-flops";
                fifo += size.entries * *size.bits_per_entry;
                flops += size.control_bits;
            } else {
                shape = size.total_bits.has_value()
                    ? std::to_string(size.entries) + ", declared as a whole"
                    : std::to_string(size.entries) + " x " + std::to_string(*size.bits_per_entry);
                flops += *bits;
            }
            if (!size.note.empty()) shape += "; " + size.note;
            std::snprintf(line, sizeof(line), "  %-9s %-52s %12s  ", size.kind, size.name.c_str(), number(*bits).c_str());
            out << line << shape << "\n";
        }
        out << "  in the open: flip-flops " << number(flops) << ", FIFO " << number(fifo) << ", total "
            << number(flops + fifo) << "\n";
        BlackBoxCost walls;
        bool any = false;
        for (const auto& black_box : black_boxes_) {
            if (black_box.unit != which) continue;
            any = true;
            const auto cost = black_box_cost(black_box);
            char line[200];
            std::snprintf(line, sizeof(line), "  %-9s %-52s %12s  %s\n", "black box", black_box.name.c_str(),
                number(cost.flip_flops + cost.memory_bits).c_str(), cost.source);
            out << line;
            walls.flip_flops += cost.flip_flops;
            walls.memory_bits += cost.memory_bits;
            walls.basic_dsps += cost.basic_dsps;
        }
        if (any) {
            out << "  behind walls: flip-flops " << number(walls.flip_flops) << ", memory " << number(walls.memory_bits)
                << ", basic DSPs " << number(walls.basic_dsps) << " (declared)\n";
        }
        total_flops += flops;
        total_fifo += fifo;
    }
    // The black boxes: each one's contract, and its cost and where that
    // comes from.
    BlackBoxCost declared;
    out << "\nblack boxes (a contract outside, anything inside: the cost is estimated, not counted)\n";
    if (black_boxes_.empty()) out << "  none\n";
    for (const auto& black_box : black_boxes_) {
        out << "  " << black_box.name << " (" << unit_name(black_box.unit) << ")\n    latency " << black_box.latency
            << " ("
            << black_box.latency_from << ")";
        if (black_box.interval > 1) out << ", a new input at most every " << black_box.interval << " cycles";
        out << "\n";
        for (const auto* ports : {&black_box.inputs, &black_box.outputs}) {
            for (const auto& port : *ports) {
                out << (ports == &black_box.inputs ? "    in   " : "    out  ") << port.name << ": " << number(port.bits)
                    << (port.bits == 1 ? " bit, " : " bits, ") << port.held_in << "\n";
            }
        }
        if (black_box.memory.has_value()) {
            const auto& shape = *black_box.memory;
            out << "    shape: banks x words x word bits " << shape.banks << " x " << shape.rows << " x "
                << shape.word_bits << "; " << shape.read_ports << "R " << shape.write_ports << "W"
                << (shape.banks > 1 ? " per bank" : "") << ", " << same_cycle_text(shape.same_cycle) << "\n";
        }
        if (!black_box.note.empty()) out << "    " << black_box.note << "\n";
        const auto cost = black_box_cost(black_box);
        out << "    cost " << number(cost.flip_flops) << " flip-flops, " << number(cost.memory_bits) << " memory bits, "
            << number(cost.basic_dsps) << " basic DSPs (" << cost.source << ")\n";
        declared.flip_flops += cost.flip_flops;
        declared.memory_bits += cost.memory_bits;
        declared.basic_dsps += cost.basic_dsps;
    }
    const uint64_t bram18 = 18 * 1024;
    char line[160];
    out << "\ntotals\n";
    out << "  in the open (counted)\n";
    out << "    flip-flops       " << number(total_flops) << "\n";
    out << "    FIFO storage     " << number(total_fifo) << "\n";
    out << "    total            " << number(total_flops + total_fifo) << "\n";
    out << "  behind walls (" << black_boxes_.size() << " black boxes)\n";
    out << "    flip-flops       " << number(declared.flip_flops) << "\n";
    out << "    memory bits      " << number(declared.memory_bits) << "\n";
    out << "    total            " << number(declared.flip_flops + declared.memory_bits) << "\n";
    out << "    basic DSPs       " << number(declared.basic_dsps) << "\n";
    const uint64_t all_flops = total_flops + declared.flip_flops;
    out << "  sum\n";
    out << "    flip-flops       " << number(all_flops) << "\n";
    out << "    memory bits      " << number(declared.memory_bits) << "\n";
    out << "    FIFO storage     " << number(total_fifo) << "\n";
    out << "    total            " << number(all_flops + declared.memory_bits + total_fifo) << "\n";
    out << "    basic DSPs       " << number(declared.basic_dsps) << "\n";
    std::snprintf(line, sizeof(line), "  memory + FIFO in 18-kilobit block RAMs: %s / %s = %.1f\n",
        number(declared.memory_bits + total_fifo).c_str(), number(bram18).c_str(),
        static_cast<double>(declared.memory_bits + total_fifo) / static_cast<double>(bram18));
    out << line;
    out << "\nnot yet declared (not in the totals)\n";
    if (missing.empty()) out << "  none\n";
    for (const auto& [reg, why] : missing) {
        out << "  " << reg->name() << " (" << unit_name(reg->unit()) << "): " << why << "\n";
    }
    out << "\nplain state, not a primitive yet (not in the totals)\n";
    out << "  issue: the scheduler's cursor, the SFU's busy-until cycle, each warp's stall-until cycle\n";
    out << "\noutside the SM (not in the report)\n";
    out << "  global memory; the memory controller's queue, bus and write acknowledgements\n";
    out << "\n(S) and (T) fields count zero (sanitizer text, trace numbers); see src/machine/state_report.cpp\n";
    return out.str();
}

} // namespace nanosm

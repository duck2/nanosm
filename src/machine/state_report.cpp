// The state report: every primitive's declared hardware bits (declare_state_bits) and the
// report of them by unit (state_report).
// (Machine is one class, src/machine/machine.hpp; the map of its files is at the top of
// src/machine/machine.cpp.)

#include "machine/machine.hpp"

#include <algorithm>
#include <bit>
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

// Bits to tell `n` things apart (0 for one or none).
uint64_t bits_for(uint64_t n) { return n <= 1 ? 0 : static_cast<uint64_t>(std::bit_width(n - 1)); }
// Bits to count from 0 to `n`.
uint64_t count_bits(uint64_t n) { return static_cast<uint64_t>(std::bit_width(n)); }

} // namespace

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
    rmw_alu_.declare_payload_bits(write_beat);
    // ---- shared memory ----
    a_read_pipe_.declare_payload_bits(read_beat);
    a_write_pipe_.declare_payload_bits(write_beat);
    b_write_pipe_.declare_payload_bits(write_beat);
    // A-read's pointer over the load slots, the probe and the TMA; A-write's
    // over the store slots; the alternation bit.
    shmem_arbiters_.declare_bits(bits_for(c.shmem_load_slots + 2) + bits_for(c.shmem_store_slots) + 1);
    // ---- tensor core ----
    // A port-B round: its step, last round, per bank a valid row and the word.
    mma_reads_.declare_payload_bits(mma_step + 1 + uint64_t{c.shmem_banks} * (1 + shmem_row + 32));
    // A step in the core: its command's slot, its result block's tmem words,
    // init, the A and B blocks, C sampled, the result.
    const uint64_t block = uint64_t{c.mma_core_m} * c.mma_core_n;
    mma_results_.declare_payload_bits(mma_slot + block * tmem_word + 1 +
        uint64_t{c.mma_core_m} * c.mma_core_k * 16 + uint64_t{c.mma_core_n} * c.mma_core_k * 16 + 1 + block * 32);
    // Each engine: valid, load, warp, register base, depbar, pointer, words
    // per row (its log2), width, columns launched, beats landed.
    acc_engines_.declare_bits(
        2 * (1 + 1 + warp + rf_register + depbar + tmem_word + tmem_row_words + 3 * vector_width));
    // A beat: warp, register, address, words per row, sampled, the lanes'
    // words, last, depbar.
    acc_ld_pipe_.declare_payload_bits(warp + rf_register + tmem_word + tmem_row_words + 1 + lanes * 32 + 1 + depbar);
    // A beat: its column (0 .. width - 1), the lanes' words.
    acc_st_pipe_.declare_payload_bits(bits_for(c.tmem_max_vector_width) + lanes * 32);
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
    // words and their count.
    tma_read_pairs_.declare_payload_bits(2 + 1 + (1 + urf_header) + 2 * 32 + 2);
    // The holding register: valid, header, buffer, barrier, four coordinates.
    tma_holding_.declare_bits(1 + urf_header + 32 + 32 + 4 * 32);
    tma_reads_.declare_entry_bits(tma_run);
    tma_read_received_.declare_bits(bits_for(c.gmem_sector_bytes));
    // ---- uniform register file ----
    // The round counter: valid, header, the next round.
    urf_read_rounds_.declare_bits(1 + urf_header + bits_for(tma_operand_rounds(tma_max_coordinates)));
    // A write: warp, lane mask, instruction word, its two operand latches and
    // their count, the value (valid + 32).
    urf_pipe_.declare_payload_bits(warp + lanes + word + 2 * 32 + 2 + (1 + 32));
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
    // A store-data read: its command (valid + slot), buffer, the register's
    // lanes, group, beat, the group's lanes.
    lsu_read_pipe_.declare_payload_bits((1 + lsu_slot) + 1 + lanes * 32 + bits_for(lanes) + 3 + lanes);
    // ---- register file ----
    // Its policies: the owed bit and a cursor over the async writers (the
    // load slots, the LSU's two buffers, the acc.ld pipeline) or readers
    // (acc.st, the LSU's store data).
    rf_write_policy_.declare_member_bits(1 + bits_for(c.shmem_load_slots + 2 + 1));
    rf_read_policy_.declare_member_bits(1 + bits_for(2));
    // ---- dispatch ----
    // An entry: warp, pc, lane mask, instruction word, its round's registers
    // and their count, their lanes' words, its uniform registers, count and
    // words.
    dispatch_pipe_.declare_payload_bits(warp + pc + lanes + word + operand_words * rf_register +
        count_bits(operand_words) + operand_words * lanes * 32 + uniform_operands * uniform_register +
        count_bits(uniform_operands) + uniform_operands * 32);
    // ---- arithmetic ----
    // The operand latches: warp, lane mask, instruction word, the round's
    // registers, count and words, its uniform registers (one round), count
    // and words.
    arithmetic_reads_.declare_payload_bits(warp + lanes + word + operand_words * rf_register +
        count_bits(operand_words) + operand_words * lanes * 32 + 2 * uniform_register + count_bits(2) + 2 * 32);
    // ---- launch ----
    // The launch record: valid, the block's global and shared address and
    // bytes, and per launch flag a valid bit, its depbar (a kernel's .flag
    // names any of sb0..sb5) and its byte count.
    launch_record_.declare_bits(
        1 + global + shared + shared_count + uint64_t{c.core_launch_flags} * (1 + depbar + shared_count));
}

// The primitives whose payload cannot declare honest bits yet (the list at
// the end of machine.hpp), each with what bounds it.
std::vector<std::pair<const RegisterBase*, std::string>> Machine::not_yet_declared() const {
    const auto lanes = std::to_string(config_.lanes);
    const auto operand_elements = std::to_string(
        config_.mma_core_m * config_.mma_core_k + config_.mma_core_k * config_.mma_core_n);
    return {
        {&load_slots_, "its stored rounds (SlotRounds::rounds, a list of rounds of lanes); C++ worst case: " +
                lanes + " lane accesses in all its rounds, one per active lane"},
        {&store_slots_, "the same, for a store's rounds; C++ worst case: " + lanes + " lane accesses"},
        {&mma_, "the fetch's rounds of element addresses and the arriving operands (keyed by byte address); "
                "C++ worst case: " + operand_elements + " fp16 elements of one step (A and B) in each"},
        {&writebacks_, "results of every latency in one deque, removed from the middle; no C++ bound "
                       "(the write calendar's bookings bound the fixed results)"},
    };
}

// ===========================================================================
// The report: per unit its flip-flops (registers, groups, pipelines, a
// FIFO's pointers), its memories (words x width, ports) and its FIFO storage,
// then the totals. Not in the totals: what cannot declare its bits yet, and
// plain state that is not a primitive.
// ===========================================================================

std::string Machine::state_report() const {
    std::ostringstream out;
    const auto sizes = registers_.sizes();
    const auto missing = not_yet_declared();
    const auto number = [](uint64_t value) {
        std::string digits = std::to_string(value);
        for (auto at = static_cast<std::ptrdiff_t>(digits.size()) - 3; at > 0; at -= 3) digits.insert(at, ",");
        return digits;
    };
    out << "state report: " << config_.name << " (declared hardware bits)\n";
    uint64_t total_flops = 0;
    uint64_t total_memory = 0;
    uint64_t total_fifo = 0;
    for (auto unit = static_cast<uint8_t>(Unit::front_end); unit <= static_cast<uint8_t>(last_unit); ++unit) {
        const auto which = static_cast<Unit>(unit);
        uint64_t flops = 0;
        uint64_t memory = 0;
        uint64_t fifo = 0;
        out << "\n" << unit_name(which) << "\n";
        for (const auto& size : sizes) {
            if (size.unit != which) continue;
            const std::string kind = size.kind;
            const auto bits = size.bits();
            char line[160];
            if (!bits.has_value()) {
                std::snprintf(line, sizeof(line), "  %-9s %-52s not yet declared\n", size.kind, size.name.c_str());
                out << line;
                continue;
            }
            std::string shape;
            if (kind == "memory") {
                const auto words = size.entries / size.banks;
                shape = (size.banks > 1 ? std::to_string(size.banks) + " banks x " : std::string()) +
                    std::to_string(words) + " words x " + std::to_string(*size.bits_per_entry) + " bits, " +
                    std::to_string(size.read_ports) + "R " + std::to_string(size.write_ports) + "W" +
                    (size.banks > 1 ? " per bank" : "") + ", read delay " + std::to_string(size.read_delay) + ", " +
                    size.same_cycle;
                if (size.name == instruction_memory_.name()) shape += "; the kernel's words (no profile size)";
                memory += *bits;
            } else if (kind == "FIFO") {
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
            std::snprintf(line, sizeof(line), "  %-9s %-52s %12s  ", size.kind, size.name.c_str(), number(*bits).c_str());
            out << line << shape << "\n";
        }
        out << "  flip-flops " << number(flops) << ", memory " << number(memory) << ", FIFO " << number(fifo)
            << ", total " << number(flops + memory + fifo) << "\n";
        total_flops += flops;
        total_memory += memory;
        total_fifo += fifo;
    }
    const uint64_t bram18 = 18 * 1024;
    char line[160];
    out << "\ntotals\n";
    out << "  flip-flops       " << number(total_flops) << "\n";
    out << "  memory bits      " << number(total_memory) << "\n";
    out << "  FIFO storage     " << number(total_fifo) << "\n";
    out << "  total            " << number(total_flops + total_memory + total_fifo) << "\n";
    std::snprintf(line, sizeof(line), "  memory + FIFO in 18-kilobit block RAMs: %s / %s = %.1f\n",
        number(total_memory + total_fifo).c_str(), number(bram18).c_str(),
        static_cast<double>(total_memory + total_fifo) / static_cast<double>(bram18));
    out << line;
    out << "\nnot yet declared (not in the totals)\n";
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

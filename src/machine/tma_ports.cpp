// The TMA's rounds on the shared-memory ports: descriptor and store reads, placement
// writes, the sector register and the store registers.
// (Machine is one class, src/machine/machine.hpp; the map of its files is at the top of
// src/machine/machine.cpp.)

#include "machine/machine.hpp"

#include <algorithm>
#include <cstring>
#include <tuple>
#include <utility>

namespace nanosm {

bool Machine::tma_read_in_service() const {
    const auto& transaction = tma_read_txn_.cur();
    return transaction.has_value() && transaction->request.next_round < transaction->request.rounds;
}

namespace {

const char* tma_port_label(int kind) {
    static const char* const labels[] = {"tma descriptor read", "tma load write", "tma store read"};
    return labels[kind];
}

} // namespace

// A TMA read's next round (round 0 when it starts now). A store-source round
// carries its store register's index; descriptor rounds return to the
// frontend latch.
Machine::ReadBeat Machine::tma_read_round(const TmaPortRequest& request, bool start, uint32_t store_register) {
    const auto rounds = request.rounds;
    const auto command = request.run.command;
    if (start) {
        trace_->shmem_begin(ShmemService::a_read, cycle_, tma_port_label(static_cast<int>(request.kind)),
            "command=" + std::to_string(command) + ", rounds=" + std::to_string(rounds),
            TraceCategory::tma, command);
        if (request.kind == TmaPortKind::store_read) trace_->tma_data_port(rounds);
    }
    trace_->shmem_grant(cycle_, ShmemService::a_read, ShmemClient::tma);
    if (request.next_round + 1 == rounds) trace_->shmem_end(ShmemService::a_read, cycle_ + 1);
    ReadBeat beat;
    const bool store = request.kind == TmaPortKind::store_read;
    beat.client = store ? ReadClient::tma_store : ReadClient::tma_frontend;
    beat.slot = store ? store_register : 0;
    beat.round = request.next_round;
    beat.addresses = tma_round_addresses(request, tma_round_plan(request).at(request.next_round));
    // (S) the words a swizzle64 store reads, against their writer's row spacing
    if (store) sanitizer_.check_tma_store_source(cycle_, command, beat.addresses);
    return beat;
}

// The distinct words one TMA round touches: at most one per bank (checked).
std::vector<uint32_t> Machine::tma_round_addresses(
    const TmaPortRequest& request, const std::vector<TmaPortRequest::Piece>& pieces) const {
    std::vector<uint32_t> addresses;
    for (const auto& piece : pieces) {
        const uint32_t address = tma_piece_address(request.run, piece);
        if (static_cast<uint64_t>(address) + request.run.element_bytes > shmem_.size()) {
            throw SimError("tma port service is outside shmem");
        }
        const uint32_t word = address / 4 * 4;
        if (std::find(addresses.begin(), addresses.end(), word) == addresses.end()) addresses.push_back(word);
    }
    if (addresses.size() > config_.shmem_banks) throw SimError("internal: a tma round reads more words than banks");
    return addresses;
}

// The TMA read transaction after this cycle: a round granted (a new
// transaction starts with round 0), a descriptor round's words back into the
// frontend latch. A descriptor read ends when its last round's words are
// back, and the TMA takes its data (step_tma); a store-source read ends at
// its last round's grant (its words go to its store register:
// fill_tma_store_register).
void Machine::advance_tma_read(Wires& w) {
    auto& shmem = w.shmem;
    const bool granted = shmem.a_read.user == AReadUser::tma;
    const bool returned = shmem.delivered.has_value() && shmem.delivered->client == ReadClient::tma_frontend;
    if (!granted && !returned) return;
    auto transaction = shmem.a_read.tma_start
        ? TmaReadTransaction{*w.tma.a_read, 0, shmem.tma_store_reserved.value_or(0), {}}
        : *tma_read_txn_.cur();
    auto& request = transaction.request;
    shmem.tma_read_changed = true;
    if (granted) ++request.next_round;
    if (request.kind == TmaPortKind::store_read) {
        if (request.next_round == request.rounds) {
            shmem.tma_read_after.reset(); // A-read is released at the last round's grant
        } else {
            shmem.tma_read_after = std::move(transaction);
        }
        return;
    }
    if (returned) {
        // The descriptor's words into the frontend latch: element k is bytes
        // 4k .. 4k + 3.
        const auto& beat = *shmem.delivered;
        const auto plan = tma_round_plan(request);
        for (const auto& piece : plan.at(beat.round)) {
            const uint32_t address = tma_piece_address(request.run, piece);
            const auto at = std::find(beat.addresses.begin(), beat.addresses.end(), address / 4 * 4);
            const auto word = beat.words.at(static_cast<std::size_t>(at - beat.addresses.begin()));
            std::memcpy(transaction.latch.data() + 4 * piece.element, &word, 4);
        }
        ++transaction.returned;
    }
    if (transaction.returned == request.rounds) {
        shmem.tma_read_done = std::move(transaction);
        shmem.tma_read_after.reset();
        return;
    }
    shmem.tma_read_after = std::move(transaction);
}

// A store-source round's words at the port output, for the TMA store
// register the round names (its return context). The register's own source
// map says where each byte goes in the sector; the last round completes it.
void Machine::fill_tma_store_register(const ReadBeat& beat, Wires& w) {
    const auto& reg = tma_store_registers_.cur().ring.at(beat.slot);
    if (!reg.has_value() || reg->complete()) throw SimError("internal: a TMA store round returned to a free register");
    TmaStoreFill fill;
    fill.store_register = beat.slot;
    fill.command = reg->run.command;
    // The store-source read's request as its rounds were planned: the
    // register's run (a store read: whole words, no rotation rule).
    TmaPortRequest source;
    source.kind = TmaPortKind::store_read;
    source.run = reg->run;
    const uint32_t bytes = reg->run.element_bytes;
    const uint32_t first = run_sector_offset(reg->run);
    const auto plan = tma_round_plan(source);
    for (const auto& piece : plan.at(beat.round)) {
        const uint32_t address = tma_piece_address(reg->run, piece);
        const auto at = std::find(beat.addresses.begin(), beat.addresses.end(), address / 4 * 4);
        const auto word = beat.words.at(static_cast<std::size_t>(at - beat.addresses.begin()));
        const uint32_t offset = first + uint32_t{piece.element} * bytes;
        for (uint32_t byte = 0; byte < bytes; ++byte) {
            fill.bytes.emplace_back(offset + byte, static_cast<uint8_t>(word >> (8 * (address % 4 + byte))));
        }
    }
    if (fill.bytes.size() > config_.gmem_sector_bytes) throw SimError("internal: a tma store round fills more than a sector");
    fill.last = reg->returned + 1 == reg->rounds;
    probe(LsuEdge::tma_store_round, reg->run.command, beat.round, static_cast<uint32_t>(4 * beat.addresses.size()));
    w.shmem.tma_store_fill = std::move(fill);
}

// One placement round: its bytes from the TMA sector register's Q (filled in
// earlier cycles), or zeros for an OOB fill. The last round carries the
// placement to the RAM, where its load write ends.
Machine::WriteBeat Machine::placement_round(const TmaPortRequest& request, bool start) {
    const auto rounds = request.rounds;
    const auto& run = request.run;
    if (rounds == 0 || request.next_round >= rounds) {
        throw SimError("internal zero-round tma placement");
    }
    if (start) {
        trace_->shmem_begin(ShmemService::b_write, cycle_, tma_port_label(static_cast<int>(request.kind)),
            "command=" + std::to_string(run.command) + ", rounds=" + std::to_string(rounds),
            TraceCategory::tma, run.command);
        trace_->tma_data_port(rounds);
    }
    const auto& sector = tma_sector_.cur();
    if (!run.zero_fill && (!sector.complete || sector.sector_address != run_sector_address(run))) {
        throw SimError("internal: TMA placement started before its sector register held its sector");
    }
    // Each piece: its element's bytes from the sector register (the run's
    // offset in the sector plus element x size), or zeros.
    WriteBeat beat;
    uint32_t bytes = 0;
    const uint32_t size = run.element_bytes;
    const uint32_t first = run.zero_fill ? 0 : run_sector_offset(run);
    const auto plan = tma_round_plan(request);
    for (const auto& piece : plan.at(request.next_round)) {
        const uint32_t address = tma_piece_address(run, piece);
        if (static_cast<uint64_t>(address) + size > shmem_.size()) {
            throw SimError("tma port service is outside shmem");
        }
        std::vector<uint8_t> data(size, 0);
        if (!run.zero_fill) {
            const uint8_t* source = sector.data.data() + first + uint32_t{piece.element} * size;
            std::copy(source, source + size, data.begin());
        }
        beat.writes.push_back({address, std::move(data)});
        bytes += size;
    }
    if (bytes > 4 * config_.shmem_banks) throw SimError("internal: a tma placement round writes more than its banks");
    // The sanitizer's live-mbar check: a placement round must not write a live mbar.
    std::vector<std::pair<uint32_t, uint32_t>> written;
    for (const auto& write : beat.writes) {
        written.emplace_back(write.address, write.address + static_cast<uint32_t>(write.bytes.size()));
    }
    // (The launch copy writes the kernel's `.mbar` states: that is how they
    // get there.)
    const auto* placing = tma_.cur().commands.find(run.command);
    if (placing == nullptr || !placing->launch) {
        sanitizer_.check_live_mbar_write(cycle_, "tma load #" + std::to_string(run.command) + " placement", written);
    }
    probe(LsuEdge::tma_place_round, request.sector_request, request.next_round, bytes);
    trace_->shmem_grant(cycle_, ShmemService::b_write, ShmemClient::tma);
    if (request.next_round + 1 == rounds) {
        beat.placed = PlacementEnd{run.command,
            run.shared_address + uint32_t{run.elements} * run.element_bytes};
        trace_->shmem_end(ShmemService::b_write, cycle_ + 1);
    }
    return beat;
}

// The bank rounds of one TMA shmem transaction, from its run: each round,
// every bank serves one word. A piece is one element: its bytes lie in one
// shared word and (a sector run) one sector word, because both addresses are
// multiples of the element size (the descriptor's base is; the strides count
// elements; the buffer is 64-byte aligned) and the shuffle moves whole words.
//
// A load's placement from the sector register has one more rule: in a
// round a bank takes its bytes from ONE 4-byte word of the
// sector register, rotated by one byte amount, and writes them with byte
// enables. That is all the wiring the path has per bank: a word select, a
// four-way byte rotate, four byte enables. A shared word whose bytes come
// from two sector words (a row that starts off a 4-byte boundary) is
// therefore two partial writes in two rounds; a word that straddles two
// sectors gets its halves from the two sectors' placements. Crooked data is
// slower, never refused. Reads (the descriptor, a store's source) and zero
// fills take whole words and have no such rule. In each bank the sources
// take rounds in (shared word, sector word, rotation) order.
//
// Worst case per sector (tma_max_rounds, tma_max_pieces; checked here):
//   pieces  one per element: at most sector bytes / element bytes (32 at the
//           canonical profile, a run of 1-byte elements; the descriptor
//           read: 16);
//   rounds  with 16 or more banks, a power of two (the canonical 16): a
//           row's run covers at most sector bytes / 4 + 1 consecutive
//           unshuffled words (9), shuffled by one row number, so at most
//           ceil(9 / 16) = 1 per bank, each with at most two sources: 2
//           rounds; a transposed run's elements sit in consecutive stored
//           rows, any 16 of which put the same word on 16 banks: 1 round;
//           the descriptor's 16 aligned words: 1 round. So 2 x ceil((sector
//           bytes / 4 + 1) / 16) = 2 rounds at the canonical profile. Any
//           other bank count: at most one round per piece.
uint32_t Machine::tma_max_pieces() const {
    return std::max<uint32_t>(config_.gmem_sector_bytes, tma_descriptor_bytes / 4);
}

uint32_t Machine::tma_max_rounds() const {
    const uint32_t banks = config_.shmem_banks;
    if (banks >= 16 && (banks & (banks - 1)) == 0) return 2 * ((config_.gmem_sector_bytes / 4 + 1 + 15) / 16);
    return tma_max_pieces();
}

// Where a piece (element) starts in shared memory: the run's unshuffled
// address plus the element's step (a shift), through the layout's shuffle.
uint32_t Machine::tma_piece_address(const TmaRun& run, const TmaPortRequest::Piece& piece) const {
    const uint32_t step = run.transposed ? run.spacing : run.element_bytes;
    const uint32_t natural = run.shared_address + uint32_t{piece.element} * step;
    return run.layout == TileLayout::swizzle64 ? swizzle64_shuffle(natural, run.spacing) : natural;
}

Machine::TmaRoundPlan Machine::tma_round_plan(const TmaPortRequest& request) const {
    using Piece = TmaPortRequest::Piece;
    const auto& run = request.run;
    const bool from_sector_register = request.kind == TmaPortKind::load_write && !run.zero_fill;
    const uint32_t bytes = run.element_bytes;
    if (run.elements == 0 || run.elements > tma_max_pieces() || (bytes != 1 && bytes != 2 && bytes != 4)) {
        throw SimError("internal: a tma run of " + std::to_string(run.elements) + " elements of " +
            std::to_string(bytes) + " bytes");
    }
    // Per bank: (shared word, sector word, byte rotation) -> its pieces.
    using Source = std::tuple<uint32_t, uint64_t, uint8_t>;
    std::vector<std::map<Source, std::vector<Piece>>> banks(config_.shmem_banks);
    for (uint32_t element = 0; element < run.elements; ++element) {
        const Piece piece{static_cast<uint8_t>(element)};
        const uint32_t address = tma_piece_address(run, piece);
        if (address % bytes != 0) throw SimError("internal: a tma element is not aligned in shmem");
        Source source{address / 4, 0, 0};
        if (from_sector_register) {
            const uint64_t from = run.global_address + uint64_t{element} * bytes;
            if (from % bytes != 0) throw SimError("internal: a tma element is not aligned in gmem");
            source = {address / 4, from / 4, static_cast<uint8_t>((address % 4 + 4 - from % 4) % 4)};
        }
        banks[(address / 4) % config_.shmem_banks][source].push_back(piece);
    }
    std::size_t rounds = 0;
    for (const auto& bank : banks) rounds = std::max(rounds, bank.size());
    if (rounds == 0 || rounds > tma_max_rounds()) {
        throw SimError("internal: a tma transaction of " + std::to_string(rounds) + " bank rounds (the bound is " +
            std::to_string(tma_max_rounds()) + ")");
    }
    TmaRoundPlan plan(rounds);
    for (const auto& bank : banks) {
        std::size_t round = 0;
        for (const auto& [source, pieces] : bank) {
            (void)source;
            plan[round].insert(plan[round].end(), pieces.begin(), pieces.end());
            ++round;
        }
    }
    return plan;
}

Machine::TmaPortRequest Machine::tma_port_request(TmaPortKind kind, const TmaRun& run, uint64_t sector_request) const {
    TmaPortRequest request;
    request.kind = kind;
    request.sector_request = sector_request;
    request.run = run;
    request.rounds = static_cast<uint32_t>(tma_round_plan(request).size());
    return request;
}

// The TMA's shmem transactions after this cycle's grants and returns.
void Machine::drive_tma_transactions(const Wires& w) {
    if (w.shmem.tma_read_changed) tma_read_txn_.set_next(w.shmem.tma_read_after, "shmem: TMA read");
    if (w.shmem.tma_write_changed) tma_write_txn_.set_next(w.shmem.tma_write_after, "shmem: TMA placement");
}

// The TMA sector register: freed by its placement's last round, refilled by
// the ingress FIFO (the pluck's copy already starts from the freed register,
// so a same-cycle refill is one calculation (d)).
void Machine::drive_tma_sector(const Wires& w) {
    const bool placed = w.shmem.tma_sector_released;
    if (w.ingress.tma_sector_after.has_value()) {
        tma_sector_.set_next(*w.ingress.tma_sector_after, "ingress: TMA sector fill");
    } else if (placed) {
        auto next = tma_sector_.cur();
        next.request.reset();
        next.filled = 0;
        next.complete = false;
        tma_sector_.set_next(std::move(next), "port-B write: TMA sector placed");
    }
}

// The TMA store sector registers, in one calculation: the head register's
// words drain into the egress FIFO and its last word frees it (the head
// moves on); a source read granted now reserves the tail register -- the
// one just freed, when the ring was full (d); a source round's words land in
// the register the round names.
void Machine::drive_tma_store_registers(const Wires& w) {
    const bool draining = w.egress.client == GmemClient::tma;
    const auto& reserved = w.shmem.tma_store_reserved;
    const auto& fill = w.shmem.tma_store_fill;
    if (!draining && !reserved.has_value() && !fill.has_value()) return;
    auto next = tma_store_registers_.cur();
    const auto size = static_cast<uint32_t>(next.ring.size());
    if (draining) {
        auto& head = next.ring.at(next.head);
        head->words_out += static_cast<uint32_t>(w.egress.words.size());
        if (w.egress.tma_store_drained) {
            head.reset();
            next.head = (next.head + 1) % size;
        }
    }
    if (reserved.has_value()) {
        auto& reg = next.ring.at(*reserved);
        if (reg.has_value()) throw SimError("internal: a TMA store register reserved twice");
        TmaStoreRegister fresh;
        fresh.run = w.tma.walker->run;
        fresh.rounds = w.tma.a_read->rounds;
        fresh.data.assign(config_.gmem_sector_bytes, 0);
        reg = std::move(fresh);
    }
    if (fill.has_value()) {
        auto& reg = next.ring.at(fill->store_register);
        for (const auto& [offset, byte] : fill->bytes) reg->data.at(offset) = byte;
        ++reg->returned;
    }
    tma_store_registers_.set_next(std::move(next), "drive_tma_store_registers");
}

} // namespace nanosm

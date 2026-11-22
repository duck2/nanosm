// The TMA: its operand read, the command queue, the walker, requests and completions.
// (Machine is one class, src/machine/machine.hpp; the map of its files is at the top of
// src/machine/machine.cpp.)

#include "machine/machine.hpp"

#include <algorithm>
#include <limits>
#include <utility>

namespace nanosm {

// ===========================================================================
// TMA engine
// ===========================================================================

// What the TMA wants this cycle, from Q alone (no other unit's wants).
//   1. Present its completion event: into its own register, when that is
//      empty and the previous command has ended. The event is visible to the
//      RMW engine from the next cycle (like the MMA's).
//   2. A TMA store sector waiting, complete and visible, at the egress FIFO's
//      request pointer wants the request channel.
//   3. Port-B write: the front ticket's placement, when no placement is
//      under way.
//   4. The frontend command: its descriptor read (A-read), or the walker's
//      step (plan_tma_walker) -- a load sector wants the request channel, a
//      store sector the A-read of its source (and a free TMA store register
//      for its words); a zero fill or a skipped store stretch wants no shared
//      resource. One TMA read in service at a time: none while a descriptor
//      read's words are still coming back, or while a store-source read has
//      rounds left to be granted. There is no mbar read at the start: the
//      walk begins once the descriptor is decoded and the
//      walker's start has left its pipeline (nothing is wanted meanwhile).
Machine::TmaRequests Machine::tma_requests(const EgressWrite& egress) const {
    TmaRequests want;
    const auto& tma = tma_.cur();
    if (!tma_mbar_event_.cur().has_value() && !tma.event_command.has_value()) {
        want.present = next_data_complete_tma_command(tma);
        if (want.present.has_value()) {
            const auto& command = tma.commands.at(*want.present);
            if (command.logical_bytes > std::numeric_limits<uint32_t>::max()) {
                throw SimError("tma completion byte count exceeds the mbar encoding");
            }
            MbarRmwRequest event;
            event.operation = MbarRmwOperation::complete_tx;
            event.byte_address = command.mbar_address;
            event.value = static_cast<uint32_t>(command.logical_bytes);
            event.tma_command = command.id; // (S) the sanitizer's accept-time phase peek
            want.present_event = std::move(event);
        }
    }
    const auto* slot = egress_request_slot(std::nullopt);
    want.write_request = slot != nullptr && slot->client == GmemClient::tma && egress_slot_visible(*slot);

    if (!tma_write_txn_.cur().has_value() && !tma.returns.empty()) {
        const auto& ticket = tma.returns.front();
        want.placement = tma_port_request(TmaPortKind::load_write, ticket.run, ticket.sector_request);
    }

    want.frontend = tma.frontend.has_value() ? tma.frontend : next_tma_frontend_command(tma);
    if (!want.frontend.has_value()) return want;
    const auto& command = tma.commands.at(*want.frontend);
    const bool read_free = !tma_read_txn_.cur().has_value();
    if (command.state == TmaCommandState::descriptor) {
        if (!read_free) return want;
        // The descriptor read: the run of its sixteen words (dense).
        TmaRun run;
        run.command = command.id;
        run.shared_address = command.descriptor_address;
        run.elements = static_cast<uint8_t>(tma_descriptor_bytes / 4);
        run.element_bytes = 4;
        want.a_read = tma_port_request(TmaPortKind::descriptor, run);
        return want;
    }
    // The walker's start is in its pipeline: the frontend waits for it.
    if (command.state == TmaCommandState::starting) return want;
    if (command.state != TmaCommandState::walking) {
        want.frontend_reset = true;
        return want;
    }
    // A store sector's source read needs a TMA store register (free in Q, or
    // the head one drained into the egress FIFO this cycle (d)) and no TMA
    // read in service.
    const auto store_register = read_free ? tma_store_register_free(egress) : std::nullopt;
    want.walker = plan_tma_walker(command, tma.walker, tma.credits_used, store_register.has_value());
    if (want.walker->sector == TmaWalkerPlan::Sector::store) {
        want.a_read = tma_port_request(TmaPortKind::store_read, want.walker->run);
        want.store_register = store_register;
    }
    return want;
}

// The TMA store register a source read granted now would reserve: the ring's
// tail, when a register is free in Q or the head one drains this cycle (d).
std::optional<uint32_t> Machine::tma_store_register_free(const EgressWrite& egress) const {
    const auto& stores = tma_store_registers_.cur();
    const auto size = static_cast<uint32_t>(stores.ring.size());
    const auto held = stores.held();
    if (held == size && !egress.tma_store_drained) return std::nullopt;
    return (stores.head + held) % size; // full and draining: the head itself
}

// The walker's step this cycle, planned from Q (TmaWalker, machine.hpp): ONE
// run, one zero fill or one skipped stretch, found in closed form from the
// walker's position and its bounds -- never a loop over elements. From (x, y):
//   in range   a run to the first of the sector's end ((sector bytes -
//              address mod sector bytes) / element bytes elements on) and
//              x_high (the tensor's edge, or the row's end): a load or store
//              sector;
//   a load's   a zero fill to the first of x_low (the in-range part) or the
//   stretch    row's end, and a sector's bytes;
//   a store's  a skip to x_low or the row's end: no sector, no credit, one
//   stretch    cycle.
// A run never leaves its row: a tile whose rows are contiguous in gmem and
// shorter than a sector takes a sector request per row.
// Blocked (Sector::none; nothing changes): no credit; a store sector without
// a free store register (and A-read). A load sector also needs the request
// channel, which select_request_channel decides.
Machine::TmaWalkerPlan Machine::plan_tma_walker(const TmaCommand& command, const TmaWalker& walker,
    uint32_t credits_used, bool store_register_free) const {
    using Sector = TmaWalkerPlan::Sector;
    TmaWalkerPlan plan;
    if (command.walker_done) return plan;
    const bool load = command.direction == TmaDirection::load;
    const uint32_t bytes = walker.element_bytes;
    const uint32_t sector_bytes = config_.gmem_sector_bytes;
    const uint32_t x = walker.x;
    const bool row_in_range = walker.y >= walker.y_low && walker.y < walker.y_high;
    const bool in_range = row_in_range && x >= walker.x_low && x < walker.x_high;
    const uint64_t global = walker.row_global + uint64_t{x} * bytes; // x x bytes: a shift
    uint32_t end = 0;
    if (in_range) {
        const uint32_t to_sector_end = (sector_bytes - static_cast<uint32_t>(global % sector_bytes)) / bytes;
        end = std::min<uint32_t>(walker.x_high, x + to_sector_end);
        plan.sector = load ? Sector::load : Sector::store;
    } else {
        end = row_in_range && x < walker.x_low ? walker.x_low : walker.tile_length;
        if (load) end = std::min<uint32_t>(end, x + sector_bytes / bytes);
        plan.sector = load ? Sector::zero_fill : Sector::skip;
    }
    if (end <= x || end > walker.tile_length) throw SimError("internal: the tma walker made no progress");
    if (plan.sector != Sector::skip) {
        if (credits_used >= config_.tma_sector_credits ||
            (plan.sector == Sector::store && !store_register_free)) {
            plan.sector = Sector::none;
            return plan;
        }
        auto& run = plan.run;
        run.command = command.id;
        run.elements = static_cast<uint8_t>(end - x);
        run.element_bytes = static_cast<uint8_t>(bytes);
        run.zero_fill = !in_range;
        run.layout = walker.layout;
        run.transposed = walker.transposed;
        run.spacing = walker.spacing;
        // Element x's unshuffled shared address: x x bytes, or (transposed) x x
        // the stored row's bytes -- a shift either way.
        run.shared_address = walker.row_shared + x * (walker.transposed ? walker.spacing : bytes);
        const uint64_t step = walker.transposed ? walker.spacing : bytes;
        if (uint64_t{run.shared_address} + (run.elements - 1) * step + bytes > shmem_.size()) {
            throw SimError("internal: a tma run is outside shmem (the tile was checked at its decode)");
        }
        if (in_range) {
            run.global_address = global;
            const uint64_t run_bytes = uint64_t{run.elements} * bytes;
            if (global > gmem_.size() || run_bytes > gmem_.size() - global) {
                throw SimError("tma global access is outside gmem");
            }
            plan.sector_address = run_sector_address(run);
        }
    }
    // The walker after the step: along the row, or to the next row (one add
    // of each row step).
    auto& next = plan.walker_after;
    next = walker;
    next.x = static_cast<uint16_t>(end);
    if (end == walker.tile_length) {
        next.x = 0;
        next.y = static_cast<uint16_t>(walker.y + 1);
        next.row_global += walker.row_step_global;
        next.row_shared += walker.row_step_shared;
        plan.done_after = next.y == walker.tile_rows;
    }
    return plan;
}

// The TMA's cycle, computed on a copy of its queue: the command end (the old
// event's command leaves before a presentation names a new one), its own
// requests, its shmem transactions' ends, the ingress ticket, the write acks.
// Issue looks ahead at the result (queue entries (c)); drive_tma_queue
// commits it with issue's new command.
void Machine::step_tma(Wires& w) {
    auto tma = tma_.cur();
    // The launch copy, done last cycle, leaves the queue: it has no
    // completion event.
    for (auto it = tma.commands.begin(); it != tma.commands.end();) {
        if (it->second.state != TmaCommandState::launch_done) {
            ++it;
            continue;
        }
        trace_->tma_command_end(cycle_, it->second.id, it->second.logical_bytes, it->second.useful_bytes);
        it = tma.commands.erase(it);
        trace_tma_depths(tma, cycle_);
    }
    // The command whose event the register holds ends in the cycle the RMW
    // engine claims that event (the event's RMW starts: serve_rmw_start).
    // (Today the TMA never presents in that cycle:
    // it presents only when event_command is empty in Q.)
    if (tma.event_command.has_value() && w.shmem.tma_event_claimed) {
        const auto& command = tma.commands.at(*tma.event_command);
        trace_->tma_command_end(cycle_ + 1, command.id, command.logical_bytes, command.useful_bytes);
        tma.commands.erase(*tma.event_command);
        tma.event_command.reset();
        trace_tma_depths(tma, cycle_ + 1);
    }
    apply_tma_requests(tma, w);
    // The walker's start at the start pipeline's last stage loads the walker:
    // the command walks from the next cycle.
    const auto& start_pipe = tma_walker_start_pipe_.cur();
    if (!start_pipe.empty() && start_pipe.back().has_value()) {
        const auto& start = *start_pipe.back();
        auto& command = tma.commands.at(start.command);
        if (command.state != TmaCommandState::starting || tma.frontend != start.command) {
            throw SimError("internal: a tma walker start left its pipeline for a command not starting");
        }
        tma.walker = start.walker;
        command.state = TmaCommandState::walking;
    }
    if (w.shmem.tma_read_done.has_value()) complete_tma_frontend_read(tma, *w.shmem.tma_read_done, w);
    if (w.shmem.tma_store_fill.has_value() && w.shmem.tma_store_fill->last) {
        complete_tma_store_capture(tma, w.shmem.tma_store_fill->command);
    }
    if (w.shmem.tma_placed.has_value()) {
        land_launch_bytes(tma, *w.shmem.tma_placed, w);
        complete_tma_load_write(tma, *w.shmem.tma_placed);
    }
    if (w.ingress.tma_ticket.has_value()) {
        tma.returns.push_back(*w.ingress.tma_ticket);
        trace_tma_depths(tma, cycle_);
    }
    for (uint32_t release = 0; release < w.controller.tma_credit_releases; ++release) release_tma_credit(tma);
    w.tma_next = std::move(tma);
}

void Machine::apply_tma_requests(TmaQueue& tma, Wires& w) {
    if (w.tma.present.has_value()) {
        tma.commands.at(*w.tma.present).state = TmaCommandState::event_presented;
        tma.event_command = w.tma.present;
    }
    if (w.shmem.b_write == BWriteUser::tma_start) {
        tma.returns.pop_front(); // its rounds read the sector register until the last one
        trace_tma_depths(tma, cycle_);
    }
    tma.frontend = w.tma.frontend;
    if (!tma.frontend.has_value()) return;
    if (w.tma.frontend_reset) {
        tma.frontend.reset();
        return;
    }
    if (w.tma.walker.has_value()) apply_tma_walker(tma, w);
}

void Machine::apply_tma_walker(TmaQueue& tma, Wires& w) {
    using Sector = TmaWalkerPlan::Sector;
    const auto& plan = *w.tma.walker;
    auto& command = tma.commands.at(*tma.frontend);
    // A store sector goes when A-read grants its source read; the others
    // when the request channel lets them through; a skip always goes.
    const bool store_read_granted = w.shmem.a_read.user == AReadUser::tma && w.shmem.a_read.tma_start;
    const bool emits = plan.sector == Sector::store ? store_read_granted
        : (plan.sector == Sector::load || plan.sector == Sector::zero_fill) && w.channel.walker_proceeds;
    if (plan.sector == Sector::skip || emits) {
        tma.walker = plan.walker_after;
        if (plan.done_after) command.walker_done = true;
    }
    if (emits) {
        ++tma.credits_used;
        if (tma.credits_used > config_.tma_sector_credits) throw SimError("internal: tma sector credits overrun");
        trace_tma_depths(tma, cycle_);
        if (!plan.run.zero_fill) command.useful_bytes += uint64_t{plan.run.elements} * plan.run.element_bytes;
        switch (plan.sector) {
        case Sector::zero_fill: {
            TmaReturn returned;
            returned.run = plan.run;
            returned.sector_request = w.channel.walker_id;
            ++command.pending_load_writes;
            tma.returns.push_back(std::move(returned));
            trace_tma_depths(tma, cycle_);
            break;
        }
        case Sector::load:
            trace_->tma_sector(config_.gmem_sector_bytes);
            ++command.pending_load_writes;
            break;
        case Sector::store: // its source read reserved a TMA store register (serve_a_read)
            trace_->tma_sector(config_.gmem_sector_bytes);
            ++command.pending_store_captures;
            break;
        case Sector::none:
        case Sector::skip:
            break;
        }
    }
    if (!command.walker_done) return; // credit or channel pressure: retry next cycle
    const bool load = command.direction == TmaDirection::load;
    if ((load && command.pending_load_writes == 0) || (!load && command.pending_store_captures == 0)) {
        mark_tma_data_complete(command);
    }
    tma.frontend.reset();
}

// A descriptor read's last round is back: its data is in `request` (the
// frontend latch). The walker's start enters its pipeline (or, with no
// stages, loads the walker now); the command walks after it (no mbar
// phase read and no "enough transaction bytes" check -- the tx count is
// signed).
void Machine::complete_tma_frontend_read(TmaQueue& tma, const TmaReadTransaction& read, Wires& w) {
    const auto& request = read.request;
    auto command_it = tma.commands.find(request.run.command);
    if (command_it == tma.commands.end()) throw SimError("orphaned tma shmem transaction");
    auto& command = command_it->second;

    switch (request.kind) {
    case TmaPortKind::descriptor: {
        if (command.state != TmaCommandState::descriptor) {
            throw SimError("tma descriptor read completed in the wrong command state");
        }
        if (tma.frontend != command.id) throw SimError("internal: a descriptor returned for a command not walked next");
        const auto descriptor = decode_tma_descriptor(read.latch.data(), read.latch.size());
        command.logical_bytes = tma_logical_bytes(descriptor);
        validate_tma_runtime_descriptor(command, descriptor);
        auto walker = tma_walker_start(command, descriptor);
        // releases the descriptor read-ownership; verifies the accept-time peek
        sanitizer_.record_tma_descriptor_read(command.id, descriptor, command.shmem_address,
            command.direction == TmaDirection::load);
        if (config_.tma_walker_start_cycles == 0) {
            tma.walker = std::move(walker);
            command.state = TmaCommandState::walking;
        } else {
            w.tma_walker_starting = TmaWalkerStart{command.id, std::move(walker)};
            command.state = TmaCommandState::starting;
        }
        return;
    }
    case TmaPortKind::load_write:
    case TmaPortKind::store_read:
        break;
    }
    throw SimError("unexpected tma frontend read");
}

// A placement's last write reached the RAM: its command's load write ends.
// A placement over an mbar with a live waiter is UB (as on NVIDIA: an
// mbarrier's memory serves nothing else until mbarrier.inval); the
// sanitizer PANICs at the placement write. The machine just wrote the
// bytes.
void Machine::complete_tma_load_write(TmaQueue& tma, const PlacementEnd& placed) {
    auto command_it = tma.commands.find(placed.command);
    if (command_it == tma.commands.end()) throw SimError("orphaned tma shmem transaction");
    auto& command = command_it->second;
    if (command.direction != TmaDirection::load || command.pending_load_writes == 0) {
        throw SimError("unexpected tma load return");
    }
    --command.pending_load_writes;
    release_tma_credit(tma);
    if (command.walker_done && command.pending_load_writes == 0) {
        mark_tma_data_complete(command);
    }
}

// A store sector's source is captured: its TMA store register is complete
// (from the next cycle it waits for the egress write port).
void Machine::complete_tma_store_capture(TmaQueue& tma, uint64_t command_id) {
    auto command_it = tma.commands.find(command_id);
    if (command_it == tma.commands.end()) throw SimError("orphaned tma shmem transaction");
    auto& command = command_it->second;
    if (command.direction != TmaDirection::store || command.pending_store_captures == 0) {
        throw SimError("unexpected tma store capture");
    }
    --command.pending_store_captures;
    if (command.walker_done && command.pending_store_captures == 0) {
        mark_tma_data_complete(command);
    }
}

std::optional<uint64_t> Machine::next_tma_frontend_command(const TmaQueue& tma) const {
    for (const auto& [id, command] : tma.commands) {
        if (command.state == TmaCommandState::descriptor || command.state == TmaCommandState::starting ||
            (command.state == TmaCommandState::walking && !command.walker_done)) {
            return id;
        }
    }
    return std::nullopt;
}

std::optional<uint64_t> Machine::next_data_complete_tma_command(const TmaQueue& tma) const {
    for (const auto& [id, command] : tma.commands) {
        if (command.state == TmaCommandState::data_complete) return id;
    }
    return std::nullopt;
}

// The run's sector, its first byte's offset in it, and the byte enables of
// its bytes (a store's AXI strobes): a decoder of (offset, length).
uint64_t Machine::run_sector_address(const TmaRun& run) const {
    return run.global_address - run.global_address % config_.gmem_sector_bytes;
}

uint32_t Machine::run_sector_offset(const TmaRun& run) const {
    return static_cast<uint32_t>(run.global_address % config_.gmem_sector_bytes);
}

uint64_t Machine::run_byte_enable(const TmaRun& run) const {
    const uint32_t length = uint32_t{run.elements} * run.element_bytes;
    const uint64_t mask = length >= 64 ? ~0ull : (1ull << length) - 1;
    return mask << run_sector_offset(run);
}

// The walker's registers for the command whose descriptor has just come
// back (TmaWalker, machine.hpp), computed in that cycle from the decoded
// descriptor and the command's coordinates; they travel through the walker
// start pipeline's tma.walker_start_cycles stages (TmaWalkerStart, canonical
// 2) before the walker is loaded, so the walk starts that many cycles after
// the next one (0 stages: in the next one, the timing before the stages).
// Three products, side by side, each a 32-bit coordinate x a 32-bit stride:
// the row dimension's (c_row x s_row) and the two other dimensions' (c_k x
// s_k; their in-range test is two compares each); the stages are theirs and
// the add chain's.
// Everything else is subtracts, compares and shifts:
//   x_low  = clamp(-c0, 0, length)        x_high = clamp(d0 - c0, 0, length)
//   y_low  = clamp(-c_row, 0, rows)       y_high = clamp(d_row - c_row, 0, rows)
//            (both 0 when another dimension's coordinate is outside)
//   row_global (row 0) = base + (c0 + c_row s_row + c_a s_a + c_b s_b) x element bytes,
//            modulo 2^64: exact wherever a row is in range.
void Machine::validate_tma_runtime_descriptor(const TmaCommand& command, const TmaDescriptor& descriptor) const {
    if (command.logical_bytes == 0 ||
        command.logical_bytes > std::numeric_limits<uint32_t>::max()) {
        throw SimError("tma logical byte count is outside the mbar encoding");
    }
    // The buffer address is 64-byte aligned (its low bits are not wired,
    // src/tma_descriptor.hpp); the whole tile must lie in shmem.
    if (command.shmem_address > shmem_.size() ||
        command.logical_bytes > shmem_.size() - command.shmem_address) {
        throw SimError("tma tile does not fit in shmem");
    }
    if (descriptor.base > gmem_.size()) {
        throw SimError("tma global base is outside gmem");
    }
}

Machine::TmaWalker Machine::tma_walker_start(const TmaCommand& command, const TmaDescriptor& descriptor) const {
    TmaWalker walker;
    const uint32_t bytes = descriptor.element_bytes;
    const uint32_t row = descriptor.row_dimension;
    const int64_t length = descriptor.tile_length;
    const int64_t rows = descriptor.tile_rows;
    const int64_t c0 = command.coordinates[0];
    const int64_t c_row = command.coordinates[row];
    const auto clamp = [](int64_t value, int64_t high) {
        return static_cast<uint16_t>(std::clamp<int64_t>(value, 0, high));
    };
    walker.tile_length = descriptor.tile_length;
    walker.tile_rows = descriptor.tile_rows;
    walker.element_bytes = static_cast<uint8_t>(bytes);
    walker.layout = descriptor.layout;
    walker.transposed = descriptor.transposed;
    walker.spacing = descriptor.layout == TileLayout::swizzle64 ? tma_swizzle64_row_bytes(descriptor) : 0;
    walker.x_low = clamp(-c0, length);
    walker.x_high = clamp(int64_t{descriptor.dimensions[0]} - c0, length);
    // The two dimensions the tile does not walk: inside, and their offset.
    bool others_in_range = true;
    uint64_t others = 0; // < 2 x 2^31 x 2^32: fits
    for (uint32_t dimension = 1; dimension < tma_max_coordinates; ++dimension) {
        if (dimension == row) continue;
        const int64_t coordinate = command.coordinates[dimension];
        if (coordinate < 0 || static_cast<uint64_t>(coordinate) >= descriptor.dimensions[dimension]) {
            others_in_range = false;
            continue;
        }
        others += static_cast<uint64_t>(coordinate) * tma_stride(descriptor, dimension);
    }
    if (others_in_range) {
        walker.y_low = clamp(-c_row, rows);
        walker.y_high = clamp(int64_t{descriptor.dimensions[row]} - c_row, rows);
    }
    const uint64_t s_row = tma_stride(descriptor, row);
    // Element (0, 0)'s index and address, modulo 2^64 (unsigned wrap): |c_row x
    // s_row| < 2^63.
    const uint64_t element0 = static_cast<uint64_t>(c0) + static_cast<uint64_t>(c_row * static_cast<int64_t>(s_row)) +
        others;
    walker.row_global = descriptor.base + element0 * bytes;
    walker.row_step_global = s_row * bytes;
    walker.row_shared = command.shmem_address;
    walker.row_step_shared = descriptor.transposed ? bytes : uint32_t{descriptor.tile_length} * bytes;
    // Simulator check, not machine behaviour: the highest in-range element's
    // address must not leave 64 bits (each in-range address is then exact).
    if (walker.x_low < walker.x_high && walker.y_low < walker.y_high) {
        const auto overflow = [] { throw SimError("tma global address arithmetic overflow"); };
        const uint64_t last_x = static_cast<uint64_t>(c0 + walker.x_high - 1);
        const uint64_t last_row = static_cast<uint64_t>(c_row + walker.y_high - 1);
        if (last_row != 0 && s_row > UINT64_MAX / last_row) overflow();
        uint64_t element = last_row * s_row;
        if (element > UINT64_MAX - last_x) overflow();
        element += last_x;
        if (element > UINT64_MAX - others) overflow();
        element += others;
        if (element >= UINT64_MAX / bytes) overflow();
        if (descriptor.base > UINT64_MAX - (element + 1) * bytes) overflow();
    }
    return walker;
}

void Machine::release_tma_credit(TmaQueue& tma) {
    if (tma.credits_used == 0) throw SimError("tma sector credit underflow");
    --tma.credits_used;
    trace_tma_depths(tma, cycle_);
}

void Machine::trace_tma_depths(const TmaQueue& tma, uint64_t at_cycle) {
    trace_->tma_depths(at_cycle, tma.commands.size(), tma.credits_used, tma.returns.size());
}

// Hardware: the command's data movement is done. Sanitizer: its box
// ownership ends here.
void Machine::mark_tma_data_complete(TmaCommand& command) {
    if (command.launch) {
        // The launch copy completes into the launch flags (land_launch_bytes),
        // not into a barrier: no event to present.
        command.state = TmaCommandState::launch_done;
        return;
    }
    command.state = TmaCommandState::data_complete;
    sanitizer_.record_tma_data_complete(command.id);
}

// ===========================================================================
// The TMA operand read's end (UrfReadRounds in src/machine/machine.hpp; the timing is
// src/timing.hpp tma_operand_rounds). The pair in the URF read stages' last stage
// is at the URF output now: it goes into the holding register (round 0 starts
// the command: its instruction's fields, the buffer and the barrier, the
// coordinates zero until supplied). With the command's last pair the command
// is pushed into the TMA queue now; the engine sees it from the next cycle.
// ===========================================================================

void Machine::arrive_tma_operands(Wires& w) {
    const auto& pairs = tma_read_pairs_.cur();
    if (pairs.empty() || !pairs.back().has_value()) return;
    const auto& pair = *pairs.back();
    auto holding = tma_holding_.cur();
    if (pair.round == 0) {
        if (holding.has_value()) {
            throw SimError("internal: a tma's operands arrived while the holding register held another command");
        }
        holding = TmaReadHolding{};
        holding->header = pair.header.value();
        holding->buffer = pair.words.at(0);
        holding->barrier = pair.words.at(1);
    } else {
        if (!holding.has_value()) throw SimError("internal: a tma's coordinates arrived without their command");
        for (std::size_t i = 0; i < pair.words.size(); ++i) {
            holding->coordinates.at(2 * (pair.round - 1) + i) = static_cast<int32_t>(pair.words[i]);
        }
    }
    if (pair.last) {
        w.tma_pushed = tma_command_from(*holding, w);
        holding.reset();
    }
    w.tma_holding_changed = true;
    w.tma_holding_after = std::move(holding);
}

// The holding register's command, complete now, as the TMA queue takes it:
// the machine's panics on its operands, and
// the sanitizer's checks -- the uniform operation's partial-mask rule, the
// address operands' low bits, the descriptor's and the tile's footprint.
Machine::TmaCommand Machine::tma_command_from(const TmaReadHolding& holding, Wires& w) {
    const auto& header = holding.header;
    const auto& inst = header.instruction;
    const IssueSite where{cycle_, header.warp, header.pc, inst, &config_.mma_shapes};
    if (static_cast<uint64_t>(inst.imm) + tma_descriptor_bytes > shmem_.size()) {
        throw_issue_panic(where, "tma descriptor is not in shmem");
    }
    sanitizer_.check_uniform_mask(where, header.active_mask, full_mask());
    sanitizer_.check_tma_operand_alignment(where, holding.buffer, holding.barrier);
    TmaCommand command;
    command.id = next_tma_command_id_++;
    command.warp = header.warp;
    command.pc = header.pc;
    command.instruction = inst;
    command.direction = is_tma_store(inst.opcode) ? TmaDirection::store : TmaDirection::load;
    command.descriptor_address = static_cast<uint32_t>(inst.imm);
    command.coordinates = holding.coordinates;
    // The two addresses' low bits are not wired (src/tma_descriptor.hpp).
    command.shmem_address = holding.buffer & ~(tma_buffer_alignment - 1);
    command.mbar_address = holding.barrier & ~(tma_barrier_alignment - 1);
    if (static_cast<uint64_t>(command.mbar_address) + 8 > shmem_.size()) {
        throw_issue_panic(where, "tma completion mbar is outside shmem");
    }
    // Sanitizer: the descriptor the engine will read -- memory as of this
    // cycle's edge; anything still travelling to it PANICs in
    // record_tma_accept (missing fence.sh / race) -- and the
    // phase of its completion mbar now, the accept-time peek a TMA's
    // complete_tx is checked against (the hardware reads no mbar). The
    // engine accepts the command now, at its push.
    const auto descriptor_bytes = tma_descriptor_at_accept(command.descriptor_address, w);
    sanitizer_.record_tma_accept(where, command.id, command.descriptor_address, descriptor_bytes,
        command.direction == TmaDirection::load, command.shmem_address, command.mbar_address,
        pending_tma_completion_mbar(command.mbar_address, w));
    trace_->tma_command_begin(cycle_, command.id, header.warp, command.descriptor_address,
        w.tma_next.commands.size() + 1);
    return command;
}

// The commands the TMA operand read holds after this cycle, each counted as
// a slot of the TMA queue since its accept: the one in the round counter, and
// each whose last pair is still in the URF read stages (the pair in the last
// stage arrives now: tma_pushed).
uint32_t Machine::tma_commands_being_read() const {
    const auto& rounds = urf_read_rounds_.cur();
    uint32_t commands = rounds.has_value() && is_tma_opcode(rounds->header.instruction.opcode) ? 1 : 0;
    const auto& pairs = tma_read_pairs_.cur();
    for (std::size_t stage = 0; stage + 1 < pairs.size(); ++stage) {
        if (pairs[stage].has_value() && pairs[stage]->last) ++commands;
    }
    return commands;
}

// The TMA's event register: emptied by the RMW engine's claim, filled by the
// TMA's presentation (never both in one cycle: the TMA presents only into an
// empty register).
void Machine::drive_tma_event(const Wires& w) {
    if (!w.shmem.tma_event_claimed && !w.tma.present_event.has_value()) return;
    auto next = w.shmem.tma_event_claimed ? std::nullopt : tma_mbar_event_.cur();
    if (w.tma.present_event.has_value()) {
        if (next.has_value()) throw SimError("internal TMA event presented into a full register");
        next = w.tma.present_event;
    }
    tma_mbar_event_.set_next(std::move(next), "drive_tma_event");
}

// The TMA command queue: step_tma's result, plus issue's new command.
void Machine::drive_tma_queue(const Wires& w) {
    auto next = w.tma_next;
    if (w.tma_pushed.has_value()) next.commands.emplace(w.tma_pushed->id, *w.tma_pushed);
    // The hardware sizes (TmaQueue, machine.hpp): issue's admission and the
    // credits keep them; exceeding one is a machine bug.
    if (next.commands.size() > config_.tma_queue_commands) {
        throw SimError("internal: the TMA queue holds more than tma.queue_commands commands");
    }
    if (next.credits_used > config_.tma_sector_credits || next.returns.size() > next.credits_used) {
        throw SimError("internal: TMA placement tickets exceed the sector credits");
    }
    tma_.set_next(std::move(next), "drive_tma_queue");
}

// The walker start pipeline moves one stage: a start whose descriptor came
// back now enters stage 0; the one in the last stage loaded the walker now
// (step_tma) and leaves. Only the frontend command starts, so at most one
// stage is occupied (checked).
void Machine::drive_tma_walker_start(const Wires& w) {
    const auto& cur = tma_walker_start_pipe_.cur();
    if (cur.empty()) {
        if (w.tma_walker_starting.has_value()) throw SimError("internal: a tma walker start with no start stages");
        return;
    }
    const auto occupied = [](const TmaWalkerStartPipeline& pipe) {
        return std::count_if(pipe.begin(), pipe.end(), [](const auto& stage) { return stage.has_value(); });
    };
    if (!w.tma_walker_starting.has_value() && occupied(cur) == 0) return;
    TmaWalkerStartPipeline next(cur.size());
    next.front() = w.tma_walker_starting;
    for (std::size_t stage = 1; stage < cur.size(); ++stage) next[stage] = cur[stage - 1];
    if (occupied(next) > 1) throw SimError("internal: two tma walker starts in the start pipeline");
    tma_walker_start_pipe_.set_next(std::move(next), "drive_tma_walker_start");
}

// The URF's round counter moves to its next round (it empties after the
// last) or takes issue's new instruction; the TMA operand read's sampled
// pairs move one URF read stage on (the one in the last stage arrived now);
// its holding register takes this cycle's arrival.
void Machine::drive_tma_operand_read(const Wires& w) {
    const auto& rounds = urf_read_rounds_.cur();
    std::optional<UrfReadRounds> next_rounds;
    if (rounds.has_value() && urf_read_ports_held()) {
        UrfReadRounds advanced = *rounds;
        ++advanced.round;
        next_rounds = std::move(advanced);
    }
    if (w.issue.urf_rounds.has_value()) {
        if (next_rounds.has_value()) {
            throw SimError("internal: a multi-round uniform reader was accepted while the counter had rounds left");
        }
        next_rounds = w.issue.urf_rounds;
    }
    if (rounds.has_value() || next_rounds.has_value()) {
        urf_read_rounds_.set_next(std::move(next_rounds), "drive_tma_operand_read: URF round counter");
    }
    const auto& pairs = tma_read_pairs_.cur();
    const bool in_flight = std::any_of(pairs.begin(), pairs.end(), [](const auto& stage) { return stage.has_value(); });
    if (in_flight || w.tma_sampled.has_value()) {
        TmaReadPairs next(pairs.size());
        next.front() = w.tma_sampled;
        for (std::size_t stage = 0; stage + 1 < pairs.size(); ++stage) next[stage + 1] = pairs[stage];
        tma_read_pairs_.set_next(std::move(next), "drive_tma_operand_read: URF read stages");
    }
    if (w.tma_holding_changed) tma_holding_.set_next(w.tma_holding_after, "drive_tma_operand_read: holding");
}

} // namespace nanosm

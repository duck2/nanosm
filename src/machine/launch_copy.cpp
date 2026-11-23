// The launch copy: the block the host placed in global memory is copied into the
// parameter area of shared memory while the warps already run.
// (Machine is one class, src/machine/machine.hpp; the map of its files is at the top of
// src/machine/machine.cpp.)

#include "machine/machine.hpp"

#include <utility>

namespace nanosm {

// The launch copy (README "Launch"). The
// host wrote the kernel's launch block into global memory and the launch
// record says where it is, where it goes in shared memory and how long it
// is. Nothing of it is in shared memory at the first cycle. The copy is the
// TMA's engine doing one dense load with no instruction behind it: at reset
// the TMA's queue holds one command and the walker registers are loaded from
// the launch record (one row of 4-byte elements from the block's global
// address to its shared address), so there is no descriptor fetch and no
// barrier; its sectors go through the request channel, the bus, the ingress
// FIFO, the TMA sector register and the port-B write like any load's, and
// share them with the kernel's own work. The warps start at the first cycle
// too. Each launch flag starts counted once in every warp (the depbar's
// reset value) and is decremented in every warp when the copy has landed its
// bytes (land_launch_bytes). When the last sector has landed the command
// leaves the queue; a kernel's own TMA commands queue behind it meanwhile.
// Hardware beyond the TMA's: the launch record (src/image.hpp), a 17-bit
// landed counter, one comparator per flag, and one more decrement source on
// each flag's counter in every warp.
void Machine::start_launch_copy(const LaunchRecord& record) {
    if (config_.tma_queue_commands == 0) {
        throw SimError("internal: a launch record on a profile with no TMA (check_kernel_image refuses it)");
    }
    if (record.bytes == 0 || record.bytes % 4 != 0 || record.global_address % 4 != 0 ||
        record.shared_address % 4 != 0 || record.bytes / 4 > 0xffff ||
        uint64_t{record.shared_address} + record.bytes > shmem_.size() ||
        record.global_address > gmem_.size() || record.bytes > gmem_.size() - record.global_address) {
        throw SimError("launch record: the launch block is not whole 4-byte words inside global and shared memory");
    }
    launch_record_ = record;
    auto tma = tma_.cur();
    TmaCommand command;
    command.id = next_tma_command_id_++;
    command.launch = true;
    command.direction = TmaDirection::load;
    command.shmem_address = record.shared_address;
    command.state = TmaCommandState::walking;
    command.logical_bytes = record.bytes;
    TmaWalker walker;
    walker.element_bytes = 4;
    walker.tile_length = static_cast<uint16_t>(record.bytes / 4);
    walker.tile_rows = 1;
    walker.x_high = walker.tile_length;
    walker.y_high = 1;
    walker.row_global = record.global_address;
    walker.row_shared = record.shared_address;
    tma.walker = walker;
    tma.frontend = command.id;
    trace_->launch_copy_begin(0, command.id, record.global_address, record.shared_address, record.bytes);
    tma.commands.take(command, config_.tma_queue_commands);
    tma_.poke(std::move(tma));
    for (uint32_t warp = 0; warp < warps_.size(); ++warp) {
        auto counters = depbar_.cur().at(warp);
        for (const auto& flag : record.flags) counters.at(flag.bar) = 1;
        depbar_.poke(warp, counters);
    }
    for (const auto& flag : record.flags) {
        for (uint32_t warp = 0; warp < warps_.size(); ++warp) trace_->depbar(0, warp, flag.bar, 1);
    }
    sanitizer_.record_launch_copy(record.shared_address, record.shared_address + record.bytes);
}

// A sector of the launch copy has reached the RAM (placements are in order:
// one sector register, one write transaction): the block's landed bytes
// grow, and every launch flag whose cut they now reach is decremented in
// every warp. Issue reads the counters' Q, so a waiting instruction can
// issue from the next cycle, when the bytes are in the RAM.
void Machine::land_launch_bytes(TmaQueue& tma, const PlacementEnd& placed, Wires& w) {
    auto* found = tma.commands.find_in_slot(placed.slot, placed.command);
    if (found == nullptr || !found->launch) return;
    auto& command = *found;
    const auto& record = *launch_record_;
    const uint32_t landed = placed.shared_end - record.shared_address;
    if (placed.shared_end <= record.shared_address + command.launch_landed || landed > record.bytes) {
        throw SimError("internal: the launch copy's sectors did not land in order");
    }
    for (const auto& flag : record.flags) {
        if (flag.bytes <= command.launch_landed || flag.bytes > landed) continue;
        for (uint32_t warp = 0; warp < warps_.size(); ++warp) {
            record_depbar_completion(w, warp, flag.bar, "internal: a launch flag's counter was already zero");
        }
        trace_->launch_flag(cycle_ + 1, flag.bar, flag.bytes);
    }
    command.launch_landed = landed;
    sanitizer_.record_launch_landed(record.shared_address + landed);
}

} // namespace nanosm

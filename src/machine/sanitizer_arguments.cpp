// What the machine hands the sanitizer: pure reads of hardware state that feed
// nothing in the machine.
// (Machine is one class, src/machine/machine.hpp; the map of its files is at the top of
// src/machine/machine.cpp.)

#include "machine/machine.hpp"

#include <algorithm>
#include <cstring>

namespace nanosm {

// ---------------------------------------------------------------------------
// Sanitizer argument builders. Pure reads of hardware state; their results go
// only to sanitizer_ (src/machine/sanitizer.hpp). Nothing here feeds the machine.
// (One more is beside the code it reads: mma_operand_span, in
// src/machine/tensor_core.cpp with the tensor core's address helpers.)
// ---------------------------------------------------------------------------

// An issuing warp's site: its FIFO head's pc.
IssueSite Machine::site(const Warp& warp, const Instruction& inst) const {
    return IssueSite{cycle_, warp.id, head_pc(warp.id), inst, &config_.mma_shapes};
}

// A dispatched instruction's site: the dispatch cycle, its issue's pc.
IssueSite Machine::site(const DispatchEntry& op) const {
    return IssueSite{cycle_, op.warp, op.pc, op.instruction, &config_.mma_shapes};
}

// A queued instruction's site, and a global-memory one's command's once it
// has left the queue: this cycle, its issue's pc.
IssueSite Machine::site(const QueuedEntry& entry) const {
    return IssueSite{cycle_, entry.warp, entry.pc, entry.instruction, &config_.mma_shapes};
}

IssueSite Machine::site(const LsuCommand& command) const {
    return IssueSite{cycle_, command.warp, command.pc, command.instruction, &config_.mma_shapes};
}

// A register as a sample now reads it, every lane, through no port: a write
// lands at its RAM cycle's edge, so the array holds what is visible now.
Machine::LaneWords Machine::rf_peek(uint32_t warp, uint8_t reg) const {
    const auto address = rf_address(warp, reg);
    const auto word = rf_.peek(address.bank, address.row);
    LaneWords lanes;
    for (uint32_t lane = 0; lane < config_.lanes; ++lane) lanes.push_back(word[lane]);
    return lanes;
}

// Each bank round's word addresses, round by round in bank order (the
// sanitizer's shadow of an access), and all of them in that order.
std::vector<std::vector<uint32_t>> Machine::round_addresses(const SlotRounds& access) const {
    std::vector<std::vector<uint32_t>> addresses;
    for (uint32_t round = 0; round < access.rounds; ++round) {
        auto& words = addresses.emplace_back();
        for (const auto lane : round_lanes(access, round)) words.push_back(access.address[lane]);
    }
    return addresses;
}

std::vector<uint32_t> Machine::access_addresses(const SlotRounds& access) const {
    std::vector<uint32_t> addresses;
    for (const auto& round : round_addresses(access)) addresses.insert(addresses.end(), round.begin(), round.end());
    return addresses;
}

// shmem bytes [address, address + size) as they stand after this cycle's
// clock edge: memory, plus the write rounds at the RAM now (they land at the
// edge). An access at the RAM in a command's accept cycle has REACHED it,
// so this is what an engine command accepted now finds.
std::vector<uint8_t> Machine::shmem_after_edge(uint32_t address, uint32_t size, const Wires& w) const {
    std::vector<uint8_t> bytes(shmem_.contents().begin() + address, shmem_.contents().begin() + address + size);
    for (const auto& write : w.shmem.writes) {
        for (std::size_t byte = 0; byte < write.bytes.size(); ++byte) {
            const uint64_t at = static_cast<uint64_t>(write.address) + byte;
            if (at >= address && at < static_cast<uint64_t>(address) + size) bytes[at - address] = write.bytes[byte];
        }
    }
    return bytes;
}

// The descriptor the engine will read: memory as of this cycle's edge. No
// overlay of in-flight stores is needed:
// a write still travelling to the descriptor at accept is a missing-fence
// / race PANIC (record_tma_accept), and the descriptor's read-ownership
// covers every write after it, so the engine reads exactly these bytes.
std::array<uint8_t, tma_descriptor_bytes> Machine::tma_descriptor_at_accept(uint32_t address, const Wires& w) const {
    const auto memory = shmem_after_edge(address, static_cast<uint32_t>(tma_descriptor_bytes), w);
    std::array<uint8_t, tma_descriptor_bytes> bytes{};
    std::copy(memory.begin(), memory.end(), bytes.begin());
    return bytes;
}

// The accept-time peek of a TMA's completion mbar: its word
// as the RMW engine will leave it -- memory as of this cycle's edge, then the
// writes already granted and still travelling (the write pipelines' earlier
// stages, oldest first, then this cycle's grants: an RMW's write among them),
// then the RMW's computed update (its ALU stages and this cycle's update:
// its write is granted later). None when the mbar is outside shmem (the
// dispatch has panicked then).
// Sanitizer-only: the hardware reads no mbar at TMA start.
std::optional<uint64_t> Machine::pending_tma_completion_mbar(uint32_t mbar, const Wires& w) const {
    if (mbar % 8 != 0 || static_cast<uint64_t>(mbar) + 8 > shmem_bytes()) return std::nullopt;
    auto bytes = shmem_after_edge(mbar, 8, w);
    const auto overlay_beat = [&](const std::optional<WriteBeat>& beat) {
        if (!beat.has_value()) return;
        for (const auto& write : beat->writes) {
            for (std::size_t byte = 0; byte < write.bytes.size(); ++byte) {
                const uint64_t at = static_cast<uint64_t>(write.address) + byte;
                if (at >= mbar && at < mbar + 8ull) bytes[at - mbar] = write.bytes[byte];
            }
        }
    };
    const auto& a_writes = a_write_pipe_.cur();
    const auto& b_writes = b_write_pipe_.cur();
    for (std::size_t stage = a_writes.size() - 1; stage-- > 0;) { // the last stage is at the RAM now
        overlay_beat(a_writes[stage]);
        overlay_beat(b_writes[stage]);
    }
    overlay_beat(w.shmem.a_write_beat);
    overlay_beat(w.shmem.b_write_beat);
    const auto overlay_update = [&](const std::optional<RmwUpdate>& update) {
        if (update.has_value()) overlay_beat(rmw_write_round(*update));
    };
    const auto& alu = rmw_alu_.cur();
    for (std::size_t stage = alu.size(); stage-- > 0;) overlay_update(alu[stage]);
    overlay_update(w.shmem.rmw_update);
    uint64_t word = 0;
    std::memcpy(&word, bytes.data(), sizeof(word));
    return word;
}

} // namespace nanosm

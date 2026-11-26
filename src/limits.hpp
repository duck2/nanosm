#pragma once

// Architectural maxima: the largest value a profile may give the knobs that
// size a piece of state. A register's payload has a fixed size (a field whose
// length depends on the profile is declared at its maximum here, with a
// count: src/machine/fixed.hpp), so these are the bounds the machine is built
// for. A profile past one is refused when it is loaded, naming the knob
// (check_architectural_limits, src/profile.cpp).

#include <cstddef>

namespace nanosm::limits {

inline constexpr std::size_t lanes = 64;                      // core "lanes": a lane mask is 64 bits
inline constexpr std::size_t warps = 64;                      // core "warps"
inline constexpr std::size_t rf_banks = 16;                   // rf "banks"
inline constexpr std::size_t shmem_banks = 64;                // shmem "banks"
inline constexpr std::size_t sector_bytes = 64;               // gmem "sector_bytes"
inline constexpr std::size_t instruction_buffer_packets = 16; // core "instruction_buffer_packets"
inline constexpr std::size_t mma_queue_commands = 8;          // mma "queue_commands"
inline constexpr std::size_t tma_queue_commands = 8;          // tma "queue_commands"
inline constexpr std::size_t lsu_active_commands = 8;         // lsu "active_commands"
inline constexpr std::size_t shmem_queue_entries = 16;        // shmem "read_queue_entries", "write_queue_entries"
inline constexpr std::size_t mma_block = 64;                  // mma core_m x core_n, core_m x core_k, core_n x core_k

// Fixed by the instruction set, not by a profile.
inline constexpr std::size_t round_reads = 4;                 // vector registers one instruction reads in a round
inline constexpr std::size_t round_uniform_reads = 4;         // uniform registers one round reads (two read ports)
inline constexpr std::size_t v8_registers = 8;                // registers of a v8 load, store or accumulator transfer

} // namespace nanosm::limits

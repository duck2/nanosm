#pragma once

// What a piece of clocked state may hold.
//
// A register, a group member and a pipeline stage are fixed bundles of bits:
// their payload is a plain value of a fixed size (integers, small structs,
// std::array, std::optional of those, and fixed.hpp's Bounded for a field
// with a maximum and a count). A payload with a std::vector, deque, map or
// string in it has no declared size, so it is refused when it is compiled:
// FixedPayload below is checked by Reg, Regs, Pipe, Fifo and Memory, and by
// a black box's output register (blackbox.hpp; its inside is not state of this
// kind).

#include <type_traits>

namespace nanosm {

template <typename T>
concept FixedPayload = std::is_trivially_copyable_v<T>;

} // namespace nanosm

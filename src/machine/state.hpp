#pragma once

// What a piece of clocked state may hold.
//
// A register, a group member and a pipeline stage are fixed bundles of bits:
// their payload is a plain value of a fixed size (integers, small structs,
// std::array, std::optional of those, and fixed.hpp's Bounded for a field
// with a maximum and a count). A payload with a std::vector, deque, map or
// string in it has no declared size, so it is refused when it is compiled:
// FixedPayload below is checked by Reg, Regs, Pipe, Fifo and Memory.
//
// The payloads that are not there yet are named one by one, in one place:
// the list at the end of machine.hpp, each with what is still in it. That
// list is what remains to be made honest.

#include <type_traits>

namespace nanosm {

// Specialized (to true) only in machine.hpp's list of exceptions.
template <typename T>
struct NotYetFixedPayload : std::false_type {};

template <typename T>
concept FixedPayload = std::is_trivially_copyable_v<T> || NotYetFixedPayload<T>::value;

} // namespace nanosm

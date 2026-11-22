#pragma once

// Reading an integer out of JSON without narrowing it silently. Used by the
// profile reader (src/profile.cpp) and the launch image reader
// (src/image.cpp).

#include <nlohmann/json.hpp>

#include <limits>
#include <optional>
#include <stdint.h>

namespace nanosm {

// A JSON value as an unsigned integer that fits T, or nullopt. Checked BEFORE
// it is narrowed: nlohmann's get<uint32_t> is a bare cast (4294967296 would
// become 0, 4160.9 4160 and -8 0xfffffff8).
template <typename T>
std::optional<T> unsigned_field(const nlohmann::json& value) {
    if (!value.is_number_unsigned()) return std::nullopt;
    const auto wide = value.get<uint64_t>();
    if (wide > std::numeric_limits<T>::max()) return std::nullopt;
    return static_cast<T>(wide);
}

} // namespace nanosm

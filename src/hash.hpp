#pragma once

#include <stdint.h>
#include <string_view>

namespace nanosm {

constexpr uint64_t fnv1a_64(std::string_view bytes) {
    constexpr uint64_t offset_basis = 14695981039346656037ull;
    constexpr uint64_t prime = 1099511628211ull;

    uint64_t hash = offset_basis;
    for (const unsigned char byte : bytes) {
        hash ^= byte;
        hash *= prime;
    }
    return hash;
}

} // namespace nanosm

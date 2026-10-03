#pragma once

#include "isa.hpp"

#include <stdint.h>
#include <string>
#include <string_view>
#include <vector>

namespace nanosm {

struct ImageSegment {
    uint64_t address = 0;
    std::vector<uint8_t> bytes;
};

struct Launch {
    uint32_t active_warps = 0;
    std::vector<ImageSegment> shmem_image;
    std::vector<ImageSegment> gmem_image;
};

struct LaunchImage {
    Kernel kernel;
    Launch launch;
};

std::string emit_launch_image(const LaunchImage& image);
LaunchImage load_launch_image(std::string_view source);
LaunchImage load_launch_image_file(const std::string& path);
std::string emit_launch_state(const Launch& launch);
Launch load_launch_state(std::string_view source);

} // namespace nanosm

#include "sim.hpp"

#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t elements = 64;
constexpr std::uint32_t a_base = 0;
constexpr std::uint32_t b_base = elements * 4;
constexpr std::uint32_t c_base = elements * 8;

void write_float(std::vector<std::uint8_t>& image, std::uint32_t address, float value) {
    const auto bits = nanosm::float_bits(value);
    std::memcpy(image.data() + address, &bits, sizeof(bits));
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::string kernel_path = argc > 1 ? argv[1] : "build/generated/vadd.hex";
        auto kernel = nanosm::load_hex_file(kernel_path);

        nanosm::Config config;
        nanosm::Launch launch;
        launch.active_warps = elements / config.lanes;
        launch.shmem_image.resize(c_base + elements * 4, 0);
        for (std::uint32_t i = 0; i < elements; ++i) {
            write_float(launch.shmem_image, a_base + i * 4, static_cast<float>(i) * 0.5f);
            write_float(launch.shmem_image, b_base + i * 4, 100.0f - static_cast<float>(i));
        }

        nanosm::Machine machine(config, std::move(kernel), std::move(launch));
        machine.run();

        for (std::uint32_t i = 0; i < elements; ++i) {
            const auto actual = nanosm::bits_float(machine.read_shmem_u32(c_base + i * 4));
            const auto expected = static_cast<float>(i) * 0.5f + 100.0f - static_cast<float>(i);
            if (std::fabs(actual - expected) > 1e-6f) {
                std::cerr << "mismatch C[" << i << "]: got " << actual
                          << ", expected " << expected << '\n';
                return 1;
            }
        }

        std::cout << "vadd ok: " << elements << " elements in " << machine.cycle() << " cycles\n";
        for (const auto& event : machine.trace()) {
            std::cout << event.cycle << "\t" << event.message << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

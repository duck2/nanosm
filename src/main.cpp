#include "sim.hpp"
#include "log.hpp"

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr uint32_t elements = 64;
constexpr uint32_t a_base = 0;
constexpr uint32_t b_base = elements * 4;
constexpr uint32_t c_base = elements * 8;

void write_float(std::vector<uint8_t>& image, uint32_t address, float value) {
    const auto bits = nanosm::float_bits(value);
    std::memcpy(image.data() + address, &bits, sizeof(bits));
}

} // namespace

int main(int argc, char** argv) {
    return nanosm::run_main([&] {
        if (argc < 3) {
            nanosm::err(
                "usage: nanosm_demo DEVICE.json KERNEL.hex "
                "[--summary] [--trace FILE.json] [--trace-debug] [--trace-max-events N]");
            return 2;
        }
        nanosm::TraceOptions trace_options;
        std::string trace_path;
        bool show_summary = true;
        for (int i = 3; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--summary") {
                show_summary = true;
            } else if (option == "--trace" && i + 1 < argc) {
                trace_options.timeline = true;
                trace_path = argv[++i];
            } else if (option == "--trace-debug") {
                trace_options.timeline = true;
                trace_options.debug = true;
            } else if (option == "--trace-max-events" && i + 1 < argc) {
                trace_options.max_events = std::stoull(argv[++i]);
            } else {
                nanosm::err("unknown or incomplete option '{}'", option);
                return 2;
            }
        }
        const auto config = nanosm::load_device_profile(argv[1]);
        const std::string kernel_path = argv[2];
        auto kernel = nanosm::load_hex_file(kernel_path);

        nanosm::Launch launch;
        launch.active_warps = elements / config.lanes;
        nanosm::ImageSegment gmem;
        gmem.address = 0;
        gmem.bytes.resize(c_base + elements * 4, 0);
        for (uint32_t i = 0; i < elements; ++i) {
            write_float(gmem.bytes, a_base + i * 4, static_cast<float>(i) * 0.5f);
            write_float(gmem.bytes, b_base + i * 4, 100.0f - static_cast<float>(i));
        }
        launch.gmem_image.push_back(std::move(gmem));

        nanosm::Machine machine(config, std::move(kernel), std::move(launch), trace_options);
        machine.run();

        for (uint32_t i = 0; i < elements; ++i) {
            const auto actual = nanosm::bits_float(machine.read_gmem_u32(c_base + i * 4));
            const auto expected = static_cast<float>(i) * 0.5f + 100.0f - static_cast<float>(i);
            if (std::fabs(actual - expected) > 1e-6f) {
                nanosm::err("mismatch C[{}]: got {}, expected {}", i, actual, expected);
                return 1;
            }
        }

        nanosm::log("vadd ok: {} elements", elements);
        if (show_summary) nanosm::log("{}", machine.trace().summary_text());
        if (!trace_path.empty()) {
            machine.trace().write_perfetto(trace_path);
            nanosm::log(
                "wrote {} timeline events to {}{}",
                machine.trace().events().size(),
                trace_path,
                machine.trace().truncated() ? " (truncated)" : "");
        }
        return 0;
    });
}

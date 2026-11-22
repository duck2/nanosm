#include "machine/machine.hpp"
#include "log.hpp"

#include <string>
#include <utility>

int main(int argc, char** argv) {
    return nanosm::run_main([&] {
        if (argc < 3) {
            nanosm::err(
                "usage: nanosm_run DEVICE.json INPUT.launch.json "
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
        auto input = nanosm::load_launch_image_file(argv[2]);
        nanosm::Machine machine(
            config, std::move(input.kernel), std::move(input.launch), trace_options);
        machine.run();

        nanosm::log("launch completed");
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

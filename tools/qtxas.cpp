#include "compiler.hpp"
#include "log.hpp"

#include <fstream>

int main(int argc, char** argv) {
    if (argc != 4) {
        nanosm::err("usage: qtxas DEVICE.json INPUT.asm OUTPUT.hex");
        return 2;
    }
    return nanosm::run_main([&] {
        const auto profile = nanosm::load_device_profile(argv[1]);
        const auto kernel = nanosm::assemble_asm_file(argv[2], profile);
        std::ofstream output(argv[3]);
        if (!output) nanosm::die("cannot create output hex '{}'", argv[3]);
        output << nanosm::emit_hex(kernel);
        return 0;
    });
}

#include "compiler.hpp"
#include "log.hpp"

#include <fstream>

int main(int argc, char** argv) {
    if (argc != 4) {
        nanosm::err("usage: qtxc DEVICE.json INPUT.qtx OUTPUT.asm");
        return 2;
    }
    return nanosm::run_main([&] {
        const auto profile = nanosm::load_device_profile(argv[1]);
        const auto program = nanosm::compile_qtx_file(argv[2], profile);
        std::ofstream output(argv[3]);
        if (!output) nanosm::die("cannot create output asm '{}'", argv[3]);
        output << nanosm::emit_asm(program);
        return 0;
    });
}

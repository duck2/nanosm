#include "compiler.hpp"

#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: qtxc INPUT.qtx OUTPUT.asm\n";
        return 2;
    }
    try {
        const auto program = nanosm::compile_qtx_file(argv[1]);
        std::ofstream output(argv[2]);
        if (!output) throw nanosm::SimError("cannot create output asm");
        output << nanosm::emit_asm(program);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

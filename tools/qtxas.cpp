#include "compiler.hpp"

#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: qtxas INPUT.asm OUTPUT.hex\n";
        return 2;
    }
    try {
        const auto kernel = nanosm::assemble_asm_file(argv[1]);
        std::ofstream output(argv[2]);
        if (!output) throw nanosm::SimError("cannot create output hex");
        output << nanosm::emit_hex(kernel);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

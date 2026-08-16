#pragma once

#include <cstdlib>
#include <exception>
#include <format>
#include <iostream>
#include <utility>

namespace nanosm {

template<typename... Args>
void log(std::format_string<Args...> format, Args&&... args) {
    std::cout << std::format(format, std::forward<Args>(args)...) << '\n';
}

template<typename... Args>
void err(std::format_string<Args...> format, Args&&... args) {
    std::cerr << std::format(format, std::forward<Args>(args)...) << '\n';
}

template<typename... Args>
[[noreturn]] void die(std::format_string<Args...> format, Args&&... args) {
    err(format, std::forward<Args>(args)...);
    std::exit(EXIT_FAILURE);
}

template<typename Function>
int run_main(Function&& function) {
    try {
        return function();
    } catch (const std::exception& error) {
        err("{}", error.what());
        return EXIT_FAILURE;
    } catch (...) {
        err("unknown fatal error");
        return EXIT_FAILURE;
    }
}

} // namespace nanosm

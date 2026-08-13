cmake_minimum_required(VERSION 3.20)
project(nanosm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(nanosm
    src/isa.cpp
    src/compiler.cpp
    src/sim.cpp
)
target_include_directories(nanosm PUBLIC src)

if(MSVC)
    target_compile_options(nanosm PRIVATE /W4 /permissive-)
else()
    target_compile_options(nanosm PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

add_executable(qtxc tools/qtxc.cpp)
target_link_libraries(qtxc PRIVATE nanosm)

add_executable(qtxas tools/qtxas.cpp)
target_link_libraries(qtxas PRIVATE nanosm)

set(GENERATED_DIR ${CMAKE_BINARY_DIR}/generated)
set(VADD_ASM ${GENERATED_DIR}/vadd.asm)
set(VADD_HEX ${GENERATED_DIR}/vadd.hex)

add_custom_command(
    OUTPUT ${VADD_ASM}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${GENERATED_DIR}
    COMMAND $<TARGET_FILE:qtxc> ${CMAKE_SOURCE_DIR}/kernels/vadd.qtx ${VADD_ASM}
    DEPENDS qtxc ${CMAKE_SOURCE_DIR}/kernels/vadd.qtx
    VERBATIM
)
add_custom_command(
    OUTPUT ${VADD_HEX}
    COMMAND $<TARGET_FILE:qtxas> ${VADD_ASM} ${VADD_HEX}
    DEPENDS qtxas ${VADD_ASM}
    VERBATIM
)
add_custom_target(qtx_kernels ALL DEPENDS ${VADD_HEX})

add_executable(nanosm_demo src/main.cpp)
target_link_libraries(nanosm_demo PRIVATE nanosm)
add_dependencies(nanosm_demo qtx_kernels)

add_executable(nanosm_tests tests/tests.cpp)
target_link_libraries(nanosm_tests PRIVATE nanosm)

enable_testing()
add_test(NAME nanosm_tests COMMAND nanosm_tests)
set_tests_properties(nanosm_tests PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_custom_target(run
    COMMAND $<TARGET_FILE:nanosm_demo> ${VADD_HEX}
    DEPENDS nanosm_demo qtx_kernels
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    USES_TERMINAL
)

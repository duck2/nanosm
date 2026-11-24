cmake_minimum_required(VERSION 3.20)
project(nanosm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Optimized by default: the simulator steps every core cycle, and an unoptimized
# build is several times slower. Pass -DCMAKE_BUILD_TYPE=Debug for a debug build.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

# The Sanitizer (src/machine/sanitizer.hpp) holds shadow state that only decides
# whether to PANIC. OFF compiles it to no-ops; the machine must compute exactly
# the same thing either way. With ON (the default) the build also produces an
# OFF twin of the library, C API and unit tests, and ctest proves the two
# byte-identical (scripts/sanitizer_off_check.py).
option(NANOSM_SANITIZER "Build the ownership/hazard sanitizer (OFF: compiled to no-ops)" ON)

set(NANOSM_SOURCES
    src/isa.cpp
    src/image.cpp
    src/profile.cpp
    src/qtxc/parse.cpp
    src/qtxc/directives.cpp
    src/qtxc/expressions.cpp
    src/qtxc/objects.cpp
    src/qtxc/metadata.cpp
    src/qtxc/registers.cpp
    src/qtxc/checks.cpp
    src/qtxc/schedule.cpp
    src/qtxc/assemble.cpp
    src/qtxc/hex.cpp
    src/machine/machine.cpp
    src/machine/launch_copy.cpp
    src/machine/front_end.cpp
    src/machine/branch.cpp
    src/machine/issue.cpp
    src/machine/dispatch.cpp
    src/machine/rf.cpp
    src/machine/urf.cpp
    src/machine/arith.cpp
    src/machine/shmem.cpp
    src/machine/barriers.cpp
    src/machine/tma.cpp
    src/machine/tma_ports.cpp
    src/machine/tensor_core.cpp
    src/machine/sanitizer_arguments.cpp
    src/machine/state_report.cpp
    src/machine/common.cpp
    src/machine/sanitizer.cpp
    src/machine/mma_numeric.cpp
    src/machine/lsu.cpp
    src/machine/gmem.cpp
    src/machine/trace.cpp
)

function(add_nanosm_library NAME SANITIZER)
    add_library(${NAME} ${NANOSM_SOURCES})
    target_include_directories(${NAME} PUBLIC src PRIVATE third_party)
    target_compile_definitions(${NAME} PUBLIC NANOSM_SANITIZER=${SANITIZER})
    set_target_properties(${NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    if(MSVC)
        target_compile_options(${NAME} PRIVATE /W4 /permissive-)
    else()
        target_compile_options(${NAME} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
endfunction()

function(add_nanosm_capi NAME LIBRARY)
    add_library(${NAME} SHARED src/capi.cpp)
    target_link_libraries(${NAME} PRIVATE ${LIBRARY})
    target_include_directories(${NAME} PUBLIC src)
    target_compile_definitions(${NAME} PRIVATE NANOSM_CAPI_BUILD)
    set_target_properties(${NAME} PROPERTIES C_VISIBILITY_PRESET hidden CXX_VISIBILITY_PRESET hidden)
endfunction()

if(NANOSM_SANITIZER)
    add_nanosm_library(nanosm 1)
else()
    add_nanosm_library(nanosm 0)
endif()
add_nanosm_capi(nanosm_capi nanosm)

add_executable(qtxc tools/qtxc.cpp)
target_link_libraries(qtxc PRIVATE nanosm)

add_executable(qtxas tools/qtxas.cpp)
target_link_libraries(qtxas PRIVATE nanosm)

set(DEVICE_PROFILE ${CMAKE_SOURCE_DIR}/profiles/nanosm-v0.json)

# Replays a .launch.json (nanosm.launch's write_image) without Python.
add_executable(nanosm_run src/main.cpp)
target_link_libraries(nanosm_run PRIVATE nanosm)

# GoogleTest, vendored (third_party/googletest/VENDORED.md): no network at
# build time.
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/googletest EXCLUDE_FROM_ALL)

# The unit tests, one file per area; tests/test_support.hpp holds the shared
# helpers. Suites ending in "Slow" are the slow ones (label slow, see below).
set(NANOSM_TEST_SOURCES
    tests/basics_test.cpp
    tests/lsu_test.cpp
    tests/tma_test.cpp
    tests/tma_operands_test.cpp
    tests/machine_test.cpp
    tests/rf_test.cpp
    tests/mma_test.cpp
    tests/sanitizer_test.cpp
    tests/alu_test.cpp
    tests/stepping_test.cpp
    tests/shmem_test.cpp
    tests/rmw_test.cpp
    tests/frontend_test.cpp
    tests/urf_test.cpp
    tests/divergence_test.cpp
    tests/registers_test.cpp
    tests/memory_test.cpp
)

function(add_nanosm_tests NAME LIBRARY)
    add_executable(${NAME} ${NANOSM_TEST_SOURCES})
    target_link_libraries(${NAME} PRIVATE ${LIBRARY} GTest::gtest_main)
    target_include_directories(${NAME} PRIVATE third_party tests)
endfunction()

add_nanosm_tests(nanosm_tests nanosm)

add_executable(nanosm_capi_tests tests/capi_tests.cpp)
target_link_libraries(nanosm_capi_tests PRIVATE nanosm_capi)

if(NANOSM_SANITIZER)
    # The sanitizer-OFF twin, for the regression check below.
    add_nanosm_library(nanosm_nosan 0)
    add_nanosm_capi(nanosm_capi_nosan nanosm_nosan)
    add_nanosm_tests(nanosm_tests_nosan nanosm_nosan)
endif()

enable_testing()
# No test writes a fixed path: the unit tests write into a per-process temp
# file, launches write a trace only under --trace, and the sanitizer check
# passes --trace into its own temp dir. So the targets need no RESOURCE_LOCK
# under ctest -j, and build/ and build_nosan/ may run their tests at once.
#
# Every unit test is its own ctest entry (gtest_discover_tests, listed when
# ctest runs), so ctest -j spreads them over the cores. Labels: "unit" for
# the fast unit tests, "slow" for suites ending in "Slow" and for the heavy
# targets below; `ctest -LE slow` is the fast set (scripts/build.ps1 -Fast).
include(GoogleTest)
function(discover_nanosm_tests NAME PREFIX)
    gtest_discover_tests(${NAME}
        TEST_PREFIX "${PREFIX}"
        TEST_FILTER "-*Slow.*"
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        DISCOVERY_MODE PRE_TEST
        PROPERTIES LABELS "unit")
    gtest_discover_tests(${NAME}
        TEST_PREFIX "${PREFIX}"
        TEST_FILTER "*Slow.*"
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        DISCOVERY_MODE PRE_TEST
        PROPERTIES LABELS slow)
endfunction()
discover_nanosm_tests(nanosm_tests "")
add_test(NAME nanosm_capi_tests COMMAND nanosm_capi_tests)
set_tests_properties(nanosm_capi_tests PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# The command-line compiler and assembler: kernels/vadd.qtx to scheduled
# assembly to instruction words, into the build directory.
file(WRITE ${CMAKE_BINARY_DIR}/qtx_tools_check.cmake [=[
execute_process(COMMAND ${QTXC} ${PROFILE} ${SOURCE} ${OUT}.asm COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND ${QTXAS} ${PROFILE} ${OUT}.asm ${OUT}.hex COMMAND_ERROR_IS_FATAL ANY)
]=])
add_test(
    NAME nanosm_qtx_tools
    COMMAND ${CMAKE_COMMAND} -DQTXC=$<TARGET_FILE:qtxc> -DQTXAS=$<TARGET_FILE:qtxas>
        -DPROFILE=${DEVICE_PROFILE} -DSOURCE=${CMAKE_SOURCE_DIR}/kernels/vadd.qtx
        -DOUT=${CMAKE_BINARY_DIR}/vadd -P ${CMAKE_BINARY_DIR}/qtx_tools_check.cmake
)

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_test(
        NAME nanosm_python_pipeline
        COMMAND ${CMAKE_COMMAND} -E env
            "NANOSM_LIBRARY=$<TARGET_FILE:nanosm_capi>"
            ${Python3_EXECUTABLE} -B -m nanosm kernels/pipeline.launch.py
    )
    set_tests_properties(nanosm_python_pipeline PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    add_test(
        NAME nanosm_python_pipeline_tma
        COMMAND ${CMAKE_COMMAND} -E env
            "NANOSM_LIBRARY=$<TARGET_FILE:nanosm_capi>"
            ${Python3_EXECUTABLE} -B -m nanosm kernels/pipeline_tma.launch.py
    )
    set_tests_properties(nanosm_python_pipeline_tma PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    add_test(
        NAME nanosm_python_pipeline_tma_wide
        COMMAND ${CMAKE_COMMAND} -E env
            "NANOSM_LIBRARY=$<TARGET_FILE:nanosm_capi>"
            ${Python3_EXECUTABLE} -B -m nanosm kernels/pipeline_tma_wide.launch.py
    )
    set_tests_properties(
        nanosm_python_pipeline_tma_wide PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    add_test(
        NAME nanosm_python_fa
        COMMAND ${CMAKE_COMMAND} -E env
            "NANOSM_LIBRARY=$<TARGET_FILE:nanosm_capi>"
            ${Python3_EXECUTABLE} -B -m nanosm kernels/fa.launch.py
    )
    set_tests_properties(nanosm_python_fa PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} LABELS slow)
    add_test(
        NAME nanosm_python_fa32
        COMMAND ${CMAKE_COMMAND} -E env
            "NANOSM_LIBRARY=$<TARGET_FILE:nanosm_capi>"
            ${Python3_EXECUTABLE} -B -m nanosm kernels/fa32.launch.py
    )
    set_tests_properties(nanosm_python_fa32 PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} LABELS slow)
    add_test(
        NAME nanosm_python_vadd
        COMMAND ${CMAKE_COMMAND} -E env
            "NANOSM_LIBRARY=$<TARGET_FILE:nanosm_capi>"
            ${Python3_EXECUTABLE} -B -m nanosm kernels/vadd.launch.py
    )
    set_tests_properties(nanosm_python_vadd PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    add_test(
        NAME nanosm_python_diverge
        COMMAND ${CMAKE_COMMAND} -E env
            "NANOSM_LIBRARY=$<TARGET_FILE:nanosm_capi>"
            ${Python3_EXECUTABLE} -B -m nanosm kernels/diverge.launch.py
    )
    set_tests_properties(nanosm_python_diverge PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    add_test(
        NAME nanosm_python_diverge_masked
        COMMAND ${CMAKE_COMMAND} -E env
            "NANOSM_LIBRARY=$<TARGET_FILE:nanosm_capi>"
            ${Python3_EXECUTABLE} -B -m nanosm kernels/diverge_masked.launch.py
    )
    set_tests_properties(nanosm_python_diverge_masked PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    # The stepping discipline's regression check: register drivers run in a
    # random order every cycle (NANOSM_DRIVER_SHUFFLE_SEED); every launch's
    # stdout and Perfetto trace must be byte-identical to the unshuffled run.
    add_test(
        NAME nanosm_driver_shuffle_identical
        COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_SOURCE_DIR}/scripts/driver_shuffle_check.py
            --lib $<TARGET_FILE:nanosm_capi>
    )
    set_tests_properties(nanosm_driver_shuffle_identical PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} LABELS slow)
endif()

if(NANOSM_SANITIZER)
    # Sanitizer OFF: every test that does not expect a sanitizer panic passes
    # (those are skipped: GTEST_SKIP, reported by ctest as not run).
    discover_nanosm_tests(nanosm_tests_nosan "nosan.")
    if(Python3_Interpreter_FOUND)
        # Sanitizer ON vs OFF: every launch's stdout and Perfetto trace and the
        # unit-test output must be byte-identical. Computation that depends on
        # sanitizer state fails here.
        add_test(
            NAME nanosm_sanitizer_off_identical
            COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_SOURCE_DIR}/scripts/sanitizer_off_check.py
                --on-lib $<TARGET_FILE:nanosm_capi> --off-lib $<TARGET_FILE:nanosm_capi_nosan>
                --on-tests $<TARGET_FILE:nanosm_tests> --off-tests $<TARGET_FILE:nanosm_tests_nosan>
        )
        set_tests_properties(nanosm_sanitizer_off_identical PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} LABELS slow)
    endif()
endif()

cmake_minimum_required(VERSION 3.20)
project(nanosm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(nanosm
    src/isa.cpp
    src/image.cpp
    src/profile.cpp
    src/trace.cpp
    src/compiler.cpp
    src/sim.cpp
    src/mma_numeric.cpp
    src/lsu.cpp
)
target_include_directories(nanosm PUBLIC src PRIVATE third_party)
set_target_properties(nanosm PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    target_compile_options(nanosm PRIVATE /W4 /permissive-)
else()
    target_compile_options(nanosm PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

add_library(nanosm_capi SHARED src/capi.cpp)
target_link_libraries(nanosm_capi PRIVATE nanosm)
target_include_directories(nanosm_capi PUBLIC src)
target_compile_definitions(nanosm_capi PRIVATE NANOSM_CAPI_BUILD)
set_target_properties(nanosm_capi PROPERTIES C_VISIBILITY_PRESET hidden CXX_VISIBILITY_PRESET hidden)

add_executable(qtxc tools/qtxc.cpp)
target_link_libraries(qtxc PRIVATE nanosm)

add_executable(qtxas tools/qtxas.cpp)
target_link_libraries(qtxas PRIVATE nanosm)

set(GENERATED_DIR ${CMAKE_BINARY_DIR}/generated)
set(DEVICE_PROFILE ${CMAKE_SOURCE_DIR}/profiles/nanosm-v0.json)

function(add_qtx_kernel NAME SOURCE)
    set(ASM_FILE ${GENERATED_DIR}/${NAME}.asm)
    set(HEX_FILE ${GENERATED_DIR}/${NAME}.hex)
    add_custom_command(
        OUTPUT ${ASM_FILE}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${GENERATED_DIR}
        COMMAND $<TARGET_FILE:qtxc> ${DEVICE_PROFILE} ${SOURCE} ${ASM_FILE}
        DEPENDS qtxc ${DEVICE_PROFILE} ${SOURCE}
        VERBATIM
    )
    add_custom_command(
        OUTPUT ${HEX_FILE}
        COMMAND $<TARGET_FILE:qtxas> ${DEVICE_PROFILE} ${ASM_FILE} ${HEX_FILE}
        DEPENDS qtxas ${DEVICE_PROFILE} ${ASM_FILE}
        VERBATIM
    )
    set(${NAME}_HEX ${HEX_FILE} PARENT_SCOPE)
endfunction()

add_qtx_kernel(vadd ${CMAKE_SOURCE_DIR}/kernels/vadd.qtx)
add_qtx_kernel(pipeline ${CMAKE_SOURCE_DIR}/kernels/pipeline.qtx)
add_custom_target(qtx_kernels ALL DEPENDS ${vadd_HEX} ${pipeline_HEX})

add_executable(make_demo_image tools/demo_image.cpp)
target_link_libraries(make_demo_image PRIVATE nanosm)

set(VADD_LAUNCH ${GENERATED_DIR}/vadd.launch.json)
set(PIPELINE_LAUNCH ${GENERATED_DIR}/pipeline.launch.json)
add_custom_command(
    OUTPUT ${VADD_LAUNCH}
    COMMAND $<TARGET_FILE:make_demo_image> ${DEVICE_PROFILE} vadd ${vadd_HEX} ${VADD_LAUNCH}
    DEPENDS make_demo_image ${DEVICE_PROFILE} ${vadd_HEX}
    VERBATIM
)
add_custom_command(
    OUTPUT ${PIPELINE_LAUNCH}
    COMMAND $<TARGET_FILE:make_demo_image> ${DEVICE_PROFILE} pipeline ${pipeline_HEX} ${PIPELINE_LAUNCH}
    DEPENDS make_demo_image ${DEVICE_PROFILE} ${pipeline_HEX}
    VERBATIM
)
add_custom_target(launch_images ALL DEPENDS ${VADD_LAUNCH} ${PIPELINE_LAUNCH})

add_executable(nanosm_run src/main.cpp)
target_link_libraries(nanosm_run PRIVATE nanosm)
add_dependencies(nanosm_run launch_images)

add_executable(nanosm_tests tests/tests.cpp)
target_link_libraries(nanosm_tests PRIVATE nanosm)
target_include_directories(nanosm_tests PRIVATE third_party)

add_executable(nanosm_capi_tests tests/capi_tests.cpp)
target_link_libraries(nanosm_capi_tests PRIVATE nanosm_capi)

enable_testing()
add_test(NAME nanosm_tests COMMAND nanosm_tests)
set_tests_properties(nanosm_tests PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME nanosm_capi_tests COMMAND nanosm_capi_tests)
set_tests_properties(nanosm_capi_tests PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

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
    set_tests_properties(nanosm_python_fa PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    add_test(
        NAME nanosm_python_vadd
        COMMAND ${CMAKE_COMMAND} -E env
            "NANOSM_LIBRARY=$<TARGET_FILE:nanosm_capi>"
            ${Python3_EXECUTABLE} -B -m nanosm kernels/vadd.launch.py
    )
    set_tests_properties(nanosm_python_vadd PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()

add_custom_target(run
    COMMAND $<TARGET_FILE:nanosm_run> ${DEVICE_PROFILE} ${VADD_LAUNCH}
    DEPENDS nanosm_run launch_images
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    USES_TERMINAL
)

add_custom_target(trace_pipeline
    COMMAND $<TARGET_FILE:nanosm_run> ${DEVICE_PROFILE} ${PIPELINE_LAUNCH}
        --trace ${CMAKE_BINARY_DIR}/pipeline.perfetto.json
    DEPENDS nanosm_run launch_images
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    USES_TERMINAL
)

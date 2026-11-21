#include "capi.h"

#include <array>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string read_file(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open " + path);
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

std::string last_error() {
    const auto size = nanosm_last_error(nullptr, 0);
    std::vector<char> buffer(size + 1);
    nanosm_last_error(buffer.data(), buffer.size());
    return buffer.data();
}

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message + ": " + last_error());
}

std::string get_string(
    uint64_t (*function)(uint32_t, char*, uint64_t), uint32_t handle) {
    const auto size = function(handle, nullptr, 0);
    require(size != NANOSM_SIZE_ERROR, "C API string size query failed");
    std::vector<char> buffer(size + 1);
    require(function(handle, buffer.data(), buffer.size()) == size, "C API string copy failed");
    return buffer.data();
}

} // namespace

int main() {
    try {
        uint64_t mbar = 0;
        require(
            nanosm_mbar_encode(1, 4, 3, 128, &mbar) == NANOSM_OK,
            "C API could not encode an mbar");
        require(mbar != 0, "C API encoded an empty mbar");
        require(
            nanosm_mbar_encode(2, 1, 1, 0, &mbar) == NANOSM_ERROR,
            "C API accepted an invalid mbar phase");

        // base, element_bytes, d0..d3, s1..s3, tile length, tile rows, row
        // dimension, layout, transposed.
        std::array<uint8_t, 64> descriptor{};
        require(
            nanosm_tma_descriptor_encode(
                0x1122334455667788ull, 4, 16, 1, 1, 1, 0, 0, 0, 16, 1, 1, NANOSM_TMA_LAYOUT_DENSE, 0,
                descriptor.data(), descriptor.size()) == NANOSM_OK,
            "C API could not encode a tma descriptor");
        require(
            descriptor[0] == 4 && descriptor[4] == 16 && descriptor[8] == 0x88 && descriptor[15] == 0x11,
            "C API tma descriptor bytes are wrong");
        require(
            nanosm_tma_descriptor_encode(
                0, 2, 64, 64, 1, 1, 64, 0, 0, 64, 8, 1, NANOSM_TMA_LAYOUT_SWIZZLE64, 1,
                descriptor.data(), descriptor.size()) == NANOSM_OK &&
                descriptor[2] == NANOSM_TMA_LAYOUT_SWIZZLE64 && descriptor[3] == 1,
            "C API could not encode a transposed swizzle64 tile");
        require(
            nanosm_tma_descriptor_encode(
                0, 2, 64, 64, 1, 1, 64, 0, 0, 64, 8, 1, NANOSM_TMA_LAYOUT_DENSE, 1,
                descriptor.data(), descriptor.size()) == NANOSM_ERROR,
            "C API accepted a transposed dense tile");
        require(
            nanosm_tma_descriptor_encode(
                0, 4, 16, 1, 1, 1, 0, 0, 0, 70000, 1, 1, NANOSM_TMA_LAYOUT_DENSE, 0,
                descriptor.data(), descriptor.size()) == NANOSM_ERROR,
            "C API truncated an oversized tile length before validation");

        const auto profile = read_file("profiles/nanosm-v0.json");
        const std::string qtx = "halt\n";
        const std::string state = R"({
            "format": "nanosm-launch-state-v0",
            "active_warps": 1,
            "shmem": [],
            "gmem": []
        })";
        auto handle = nanosm_session_create_qtx(
            profile.data(), profile.size(), qtx.data(), qtx.size(), state.data(), state.size(),
            NANOSM_TRACE_TIMELINE, 1000);
        require(handle != NANOSM_INVALID_HANDLE, "C API could not create qtx session");
        require(nanosm_session_run(handle, 100) == NANOSM_OK, "C API run failed");
        // The front end (src/rf.hpp, two-word packets): the halt's packet,
        // fetched at c0, issues at c2 (fetch_to_issue); the next packet (past
        // the kernel's end), granted at c1 while the FIFO had room, is at the
        // memory output at c2 and dropped by the halt; at c2 the FIFO and the
        // read in flight fill both entries, so nothing more is granted, and
        // done() holds at c3: the run is 3 cycles.
        require(nanosm_session_cycle(handle) == 3, "halt kernel took the wrong cycle count");
        require(
            get_string(nanosm_session_summary, handle).find("instructions issued") != std::string::npos,
            "C API summary is missing");
        require(
            get_string(nanosm_session_trace_json, handle).find("traceEvents") != std::string::npos,
            "C API trace JSON is missing");
        const auto image = get_string(nanosm_session_launch_image, handle);
        std::array<uint8_t, 4> bytes{1, 1, 1, 1};
        require(
            nanosm_session_read_gmem(handle, 0, bytes.data(), bytes.size()) == NANOSM_OK,
            "C API gmem read failed");
        require(bytes == std::array<uint8_t, 4>{}, "zeroed launch gmem was not zero");
        require(nanosm_session_destroy(handle) == NANOSM_OK, "C API destroy failed");

        handle = nanosm_session_create_image(
            profile.data(), profile.size(), image.data(), image.size(), 0, 0);
        require(handle != NANOSM_INVALID_HANDLE, "C API could not replay launch image");
        require(nanosm_session_run(handle, 100) == NANOSM_OK, "replayed C API run failed");
        require(nanosm_session_destroy(handle) == NANOSM_OK, "replayed C API destroy failed");

        // A kernel with a signature: the host learns it (and the symbols a
        // test may want) from the kernel, passes arguments by name, and a
        // missing one is refused by name.
        const std::string signed_qtx = ".warps 1\n.param ptr OUT\n.equ SPOT, 256\nhalt\n";
        const auto info_size = nanosm_qtx_kernel_info(
            profile.data(), profile.size(), signed_qtx.data(), signed_qtx.size(), nullptr, 0);
        require(info_size != NANOSM_SIZE_ERROR, "C API could not describe a kernel");
        std::vector<char> info(info_size + 1);
        require(
            nanosm_qtx_kernel_info(
                profile.data(), profile.size(), signed_qtx.data(), signed_qtx.size(), info.data(), info.size()) ==
                info_size,
            "C API kernel info copy failed");
        const std::string info_text = info.data();
        require(
            info_text.find("\"name\": \"OUT\"") != std::string::npos &&
                info_text.find("\"type\": \"ptr\"") != std::string::npos &&
                info_text.find("\"SPOT\": 256") != std::string::npos &&
                info_text.find("\"warps\": 1") != std::string::npos &&
                info_text.find("\"kind\": \"param\"") != std::string::npos &&
                info_text.find("\"flag\": 5") != std::string::npos &&
                info_text.find("\"bar\": 5") != std::string::npos,
            "C API kernel info lacks the signature, a symbol, the warp count, the sections or the launch flags");
        const std::string with_argument = R"({
            "format": "nanosm-launch-state-v0",
            "args": {"OUT": 4096},
            "shmem": [],
            "gmem": []
        })";
        handle = nanosm_session_create_qtx(
            profile.data(), profile.size(), signed_qtx.data(), signed_qtx.size(),
            with_argument.data(), with_argument.size(), 0, 0);
        require(handle != NANOSM_INVALID_HANDLE, "C API refused a launch that gives its argument");
        // The kernel halts at once, but the machine is done only when the
        // launch copy has brought the argument into shared memory: global
        // memory's latency (100 cycles) and one sector.
        require(nanosm_session_run(handle, 100) != NANOSM_OK, "the launch copy took no time");
        require(nanosm_session_run(handle, 1000) == NANOSM_OK, "C API run with an argument failed");
        require(nanosm_session_destroy(handle) == NANOSM_OK, "C API destroy failed");
        handle = nanosm_session_create_qtx(
            profile.data(), profile.size(), signed_qtx.data(), signed_qtx.size(),
            state.data(), state.size(), 0, 0);
        require(
            handle == NANOSM_INVALID_HANDLE && last_error().find("'OUT' (ptr) is missing") != std::string::npos,
            "C API did not refuse a missing argument by name");

        const std::string bad_qtx = "not_an_opcode\n";
        handle = nanosm_session_create_qtx(
            profile.data(), profile.size(), bad_qtx.data(), bad_qtx.size(),
            state.data(), state.size(), 0, 0);
        require(handle == NANOSM_INVALID_HANDLE, "C API accepted an invalid qtx program");
        require(!last_error().empty(), "C API did not preserve its error message");
        std::cout << "C API tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "C API test failure: " << error.what() << '\n';
        return 1;
    }
}

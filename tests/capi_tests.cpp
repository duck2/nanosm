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

        std::array<uint8_t, 64> descriptor{};
        require(
            nanosm_tma_descriptor_encode(
                1, 0, 0, 0x1122334455667788ull,
                16, 1, 1, 0, 0, 16, 1, 1, 4, 0, 256, 128,
                descriptor.data(), descriptor.size()) == NANOSM_OK,
            "C API could not encode a tma descriptor");
        require(
            descriptor[0] == 1 && descriptor[3] == 4 && descriptor[8] == 0x88 &&
                descriptor[15] == 0x11,
            "C API tma descriptor bytes are wrong");
        require(
            nanosm_tma_descriptor_encode(
                2, 0, 1, 0, 64, 64, 1, 64, 0, 64, 8, 1, 2,
                NANOSM_TMA_SWIZZLE_XOR64_B, 1024, 128,
                descriptor.data(), descriptor.size()) == NANOSM_OK &&
                descriptor[2] == 1 && descriptor[4] == NANOSM_TMA_SWIZZLE_XOR64_B,
            "C API could not encode the named xor64 transpose layout");
        require(
            nanosm_tma_descriptor_encode(
                257, 0, 0, 0, 16, 1, 1, 0, 0, 16, 1, 1, 4, 0, 256, 128,
                descriptor.data(), descriptor.size()) == NANOSM_ERROR,
            "C API truncated an oversized tma rank before validation");

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
        require(nanosm_session_cycle(handle) == 1, "halt kernel took the wrong cycle count");
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

#include "compiler.hpp"
#include "image.hpp"
#include "log.hpp"
#include "mbar.hpp"
#include "profile.hpp"

#include <bit>
#include <cstring>
#include <fstream>
#include <string>

namespace {

void write_u32(std::vector<uint8_t>& bytes, uint32_t offset, uint32_t value) {
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

nanosm::ImageSegment f32_sequence(
    uint64_t address, uint32_t count, const auto& value) {
    nanosm::ImageSegment segment;
    segment.address = address;
    segment.bytes.resize(static_cast<std::size_t>(count) * 4);
    for (uint32_t index = 0; index < count; ++index) {
        write_u32(
            segment.bytes,
            index * 4,
            std::bit_cast<uint32_t>(static_cast<float>(value(index))));
    }
    return segment;
}

nanosm::Launch make_vadd(const nanosm::DeviceProfile& profile) {
    // kernels/vadd.qtx: A at 0, B at 32 KiB, C at 64 KiB, 8192 fp32 each,
    // four warps streaming 512 B blocks with ldg.v8 / stg.v8.
    constexpr uint32_t elements = 8192;
    if (profile.lanes != 16) {
        throw nanosm::SimError("vadd demo requires the canonical 16-lane profile");
    }
    nanosm::Launch launch;
    launch.active_warps = 4;
    launch.gmem_image.push_back(f32_sequence(0, elements, [](uint32_t i) { return i; }));
    launch.gmem_image.push_back(f32_sequence(32768, elements, [](uint32_t i) { return 2 * i; }));
    return launch;
}

nanosm::Launch make_pipeline(const nanosm::DeviceProfile& profile) {
    constexpr uint32_t tiles = 8;
    constexpr uint32_t elements = tiles * 16;
    if (profile.lanes != 16) {
        throw nanosm::SimError("pipeline demo requires the canonical 16-lane profile");
    }
    nanosm::Launch launch;
    launch.active_warps = 2;
    nanosm::MbarState barrier;
    barrier.valid = true;
    barrier.expected = 1;
    barrier.remaining = 1;
    const auto encoded = nanosm::encode_mbar_state(barrier);
    for (const uint64_t address : {1024ull, 1032ull, 1040ull, 1048ull}) {
        nanosm::ImageSegment segment;
        segment.address = address;
        segment.bytes.resize(8);
        std::memcpy(segment.bytes.data(), &encoded, sizeof(encoded));
        launch.shmem_image.push_back(std::move(segment));
    }
    launch.gmem_image.push_back(f32_sequence(0, elements, [](uint32_t i) { return i; }));
    launch.gmem_image.push_back(f32_sequence(4096, elements, [](uint32_t i) { return 2 * i; }));
    return launch;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        nanosm::err(
            "usage: make_demo_image DEVICE.json vadd|pipeline KERNEL.hex OUTPUT.launch.json");
        return 2;
    }
    return nanosm::run_main([&] {
        const auto profile = nanosm::load_device_profile(argv[1]);
        auto kernel = nanosm::load_hex_file(argv[3]);
        if (kernel.profile_name != profile.name ||
            kernel.profile_fingerprint != nanosm::device_profile_fingerprint(profile)) {
            throw nanosm::SimError("kernel and device profile do not match");
        }
        nanosm::Launch launch;
        const std::string workload = argv[2];
        if (workload == "vadd") launch = make_vadd(profile);
        else if (workload == "pipeline") launch = make_pipeline(profile);
        else nanosm::die("unknown demo workload '{}'", workload);

        std::ofstream output(argv[4]);
        if (!output) nanosm::die("cannot create launch image '{}'", argv[4]);
        output << nanosm::emit_launch_image({std::move(kernel), std::move(launch)});
        return 0;
    });
}

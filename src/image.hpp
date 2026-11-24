#pragma once

#include "isa.hpp"

#include <optional>
#include <stdint.h>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace nanosm {

struct DeviceProfile;

struct ImageSegment {
    uint64_t address = 0;
    std::vector<uint8_t> bytes;
};

// One argument of a launch (README "Launch"): the value for the kernel's
// `.param` of that name. The parameter's type says what fits: an unsigned
// integer for ptr and u32, a number for f32, 64 bytes for desc.
struct LaunchArgument {
    std::string name;
    std::variant<uint64_t, double, std::vector<uint8_t>> value;
};

// The launch record (README "Launch"): the few registers the host writes to
// start a kernel, in place before the first cycle like the warp count. They
// say where the kernel's launch block is in global memory, where it goes in
// shared memory, how long it is, and after how many delivered bytes each
// launch flag is decremented in every warp. Size at the canonical profile:
// warps 5 + global address 64 + shared address 16 + bytes 17 + four flags x
// (valid 1 + bytes 17) = 174 bits. bind_launch fills it; a kernel with no
// sections has none.
struct LaunchRecord {
    uint64_t global_address = 0;
    uint32_t shared_address = 0;
    uint32_t bytes = 0;
    std::vector<LaunchFlag> flags;
};

// What the host gives a kernel: how many warps (0: as many as the kernel's
// `.warps` says), one argument per parameter, and global memory. The host
// needs no address inside shared memory for any of it.
// test_shmem and mbars are a TEST's access, for tests that poison a region or
// set up a machine by hand: bytes written into shared memory before the first
// cycle, untimed, and barriers declared live. A launch that uses test_shmem
// says so in its run summary (`test_preload:`), so a harness can refuse it,
// and it may not touch the kernel's launch block (bind_launch).
struct Launch {
    uint32_t active_warps = 0;
    std::vector<ImageSegment> test_shmem;
    std::vector<ImageSegment> gmem_image;
    // Barriers the shmem image initializes by hand (byte addresses,
    // 8-aligned). Declared, not detected: the sanitizer treats them as LIVE
    // from launch, like a SIMT mbar.init's
    // and like a kernel's `.mbar`. Optional in the launch formats.
    std::vector<uint32_t> mbars;
    // Last, so that tests written as Launch{warps, shmem, gmem, mbars} stand.
    std::vector<LaunchArgument> args;
    // Where in global memory the host puts the kernel's launch block (a
    // multiple of 4). Absent: the top of global memory, rounded down to a
    // sector.
    std::optional<uint64_t> block_address;
    // The launch record, written by bind_launch (not by the host's caller).
    std::optional<LaunchRecord> record;
};

// The host's part of a launch, done once when a machine is built: the warp
// count settled against the kernel's `.warps`; the launch block's bytes made
// (each argument at its parameter, each `.mbar` state, each `.data`, in the
// kernel's section order: src/isa.hpp launch_block) and written into GLOBAL
// memory at block_address, with the launch record saying where; the kernel's
// barriers added to the live list. The machine copies the block into shared
// memory from the first cycle on (src/machine/launch_copy.cpp); there is no
// other way for it to get there. A missing, unknown or wrongly typed argument
// is refused by name. The result has no args left.
Launch bind_launch(const Kernel& kernel, Launch launch, const DeviceProfile& profile);

// What a kernel declares, as JSON, for a host and for a test that looks
// inside (the C API's nanosm_qtx_kernel_info):
//   {"warps": N,                      0: the kernel does not say
//    "params": [{"name", "type", "address", "bytes"}, ...],   the signature, in order
//    "sections": [{"name", "kind", "address", "bytes", "flag"}, ...],  the launch block, in order
//    "block": {"address", "bytes"},    where the block sits in shared memory
//    "flags": [{"bar", "bytes"}, ...], the launch flags
//    "symbols": {"NAME": value, ...},  .equ constants, section and object addresses
//    "objects": [{"name", "memory" ("shmem" | "tmem"), "address", "count",
//                 "size", "stride", "alignment", "rows", "cols", "elem",
//                 "over"}, ...],       the .shmem / .tmem objects: element i
//                                      is at address + i x stride
//    "registers": {"%x": "r5", ...},   the qtx's register names and qtxc's numbers
//    "warnings": ["line 12: ...", ...]} what qtxc said (a copy it inserted)
std::string emit_kernel_info(const Kernel& kernel);

// A launch image (a `.launch.json` file: nanosm.launch's write_image writes
// one, nanosm_run replays it; "format": "nanosm-launch-v0") is the complete
// cycle-0 input: the profile-stamped kernel words and register count, what
// the kernel declares
// (`warps`, its `sections` in order, its launch `flags`), the launch's
// arguments (`args`), its warp count and its launch block's global address
// (`block_address`) when it gives them, and ordered sparse `gmem` and
// `test_shmem` byte segments (`test_shmem`: a test's own untimed writes, as
// in Launch above). Bytes and 64-bit words are hexadecimal strings, addresses
// JSON integers. Segments are applied in array order, later overlapping bytes
// winning. Nothing in it says which workload wrote it.
struct LaunchImage {
    Kernel kernel;
    Launch launch;
};

std::string emit_launch_image(const LaunchImage& image);
LaunchImage load_launch_image(std::string_view source);
LaunchImage load_launch_image_file(const std::string& path);
std::string emit_launch_state(const Launch& launch);
Launch load_launch_state(std::string_view source);

} // namespace nanosm

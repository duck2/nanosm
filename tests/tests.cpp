#include "sim.hpp"
#include "hash.hpp"
#include "log.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void write_u32(std::vector<uint8_t>& image, uint32_t address, uint32_t value) {
    std::memcpy(image.data() + address, &value, sizeof(value));
}

std::vector<uint8_t>& image_bytes(
    std::vector<nanosm::ImageSegment>& image, std::size_t size) {
    image.push_back({0, std::vector<uint8_t>(size, 0)});
    return image.back().bytes;
}

const nanosm::DeviceProfile& canonical_profile() {
    static const auto profile = nanosm::load_device_profile("profiles/nanosm-v0.json");
    return profile;
}

nanosm::Kernel compile(
    std::string_view source, const nanosm::DeviceProfile& profile = canonical_profile()) {
    return nanosm::compile_and_assemble(source, profile);
}

void test_encoding_round_trip() {
    nanosm::Instruction inst;
    inst.opcode = nanosm::Opcode::add;
    inst.control.stall = 21; // needs the fifth stall bit
    inst.control.set_depbar = 4;
    inst.control.wait_mask = 0b100101;
    inst.dst = 200;
    inst.src0 = 129;
    inst.src1 = 255;
    const auto decoded = nanosm::decode(nanosm::encode(inst));
    require(decoded.opcode == inst.opcode, "opcode encoding mismatch");
    require(decoded.control.stall == 21, "stall encoding mismatch");
    require(decoded.control.set_depbar == 4, "set-depbar encoding mismatch");
    require(decoded.control.wait_mask == 0b100101, "wait-mask encoding mismatch");
    require(
        decoded.dst == 200 && decoded.src0 == 129 && decoded.src1 == 255,
        "eight-bit register encoding mismatch");

    inst.opcode = nanosm::Opcode::addi;
    inst.imm = -1234567;
    const auto immediate_decoded = nanosm::decode(nanosm::encode(inst));
    require(immediate_decoded.dst == 200 && immediate_decoded.src0 == 129, "RRI register mismatch");
    require(immediate_decoded.imm == -1234567, "signed 22-bit immediate mismatch");
}

void test_fnv1a_64_vectors() {
    require(nanosm::fnv1a_64("") == 0xcbf29ce484222325ull, "FNV-1a empty vector mismatch");
    require(nanosm::fnv1a_64("hello") == 0xa430d84680aabd0bull, "FNV-1a hello vector mismatch");
}

void test_high_register_kernel_metadata() {
    const auto kernel = compile("mov r128, 7\nhalt");
    require(kernel.register_count == 129, "qtxas did not derive the contiguous RF allocation");
    require(nanosm::decode(kernel.words.front()).dst == 128, "r128 did not survive assembly");
    const auto loaded = nanosm::load_hex(nanosm::emit_hex(kernel));
    require(loaded.register_count == 129, "hex lost the kernel register count");
}

void test_rf_capacity_determines_residency() {
    auto profile = canonical_profile();
    profile.rf_rows_per_bank = 1;
    const auto kernel = compile("mov r0, 1\nhalt", profile);
    nanosm::Launch launch;
    launch.active_warps = 2;
    try {
        nanosm::Machine machine(profile, kernel, std::move(launch));
        throw std::runtime_error("launch ignored physical RF capacity");
    } catch (const nanosm::SimError& error) {
        require(
            std::string(error.what()).find("RF allocation cannot fit") != std::string::npos,
            "RF residency failure produced the wrong error");
    }
}

void test_compiler_controls() {
    const auto program = nanosm::compile_qtx(R"(
        mov  r0, 0
        lds  r1, [r0]
        lds  r2, [r0 + 64]
        fadd r4, r1, r2
        sts  [r0 + 128], r4
        halt
    )", canonical_profile());
    require(program.instructions[0].control.stall == 0, "mov stalled without a close consumer");
    require(program.instructions[1].control.set_depbar == 0, "first load did not allocate sb0");
    require(program.instructions[2].control.set_depbar == 1, "second load did not allocate sb1");
    require(program.instructions[3].control.wait_mask == 0b000011, "fadd did not wait for both loads");
    require(program.instructions[3].control.stall == 4, "fadd did not cover its one-away consumer");
    require(program.instructions[4].control.set_depbar == 0, "store did not reuse a released depbar");
    require(program.instructions[5].control.wait_mask == 0b000001, "halt did not wait for store completion");
}

void test_stalls_track_consumer_distance() {
    // fadd's result is consumed three instructions later, so of its 4-cycle latency
    // only one cycle must be bought with a stall, and it lands just before the consumer.
    const auto program = nanosm::compile_qtx(R"(
        mov  r0, 1
        fadd r1, r0, r0
        mov  r2, 5
        mov  r3, 6
        add  r4, r1, r2
        halt
    )", canonical_profile());
    require(program.instructions[1].control.stall == 0, "fadd stalled its own latency");
    require(program.instructions[2].control.stall == 0, "independent mov stalled");
    require(program.instructions[3].control.stall == 2, "pre-consumer stall wrong");
    nanosm::Launch launch;
    launch.active_warps = 1;
    image_bytes(launch.shmem_image, 4);
    nanosm::Machine machine(
        canonical_profile(),
        nanosm::assemble_asm(nanosm::emit_asm(program), canonical_profile()),
        std::move(launch));
    machine.run(); // the hazard sanitizer accepts the minimal schedule
}

void test_counted_loop_and_label_relocation() {
    const auto program = nanosm::compile_qtx(R"(
        mov r0, 0
        mov r1, 4
    loop:
        addi r0, r0, 1
        setp.lt.u32 p0, r0, r1
        @p0 bra.uni loop
        halt
    )", canonical_profile());
    require(program.instructions[4].imm == -24, "backedge relocation is not relative to the next PC");
    require(program.instructions[4].control.predicate == 0, "branch lost its qtx predicate");
    const auto assembly = nanosm::emit_asm(program);
    require(assembly.find("loop:") != std::string::npos, "scheduled assembly lost its label");

    nanosm::Launch launch;
    launch.active_warps = 1;
    nanosm::Machine machine(
        canonical_profile(),
        nanosm::assemble_asm(assembly, canonical_profile()),
        std::move(launch));
    machine.run();
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(machine.read_reg(0, lane, 0) == 4, "counted loop produced the wrong trip count");
    }
}

void test_uniform_role_dispatch() {
    nanosm::Launch launch;
    launch.active_warps = 2;
    nanosm::Machine machine(canonical_profile(), compile(R"(
        sread r0, WARP_ID
        setp.eq p0, r0, 0
        @p0 bra.uni warp_zero
        mov r1, 11
        bra done
    warp_zero:
        mov r1, 22
    done:
        halt
    )"), std::move(launch));
    machine.run();
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(machine.read_reg(0, lane, 1) == 22, "warp-zero role branch was not taken");
        require(machine.read_reg(1, lane, 1) == 11, "nonzero warp did not take fallthrough");
    }
}

void test_nonuniform_bra_uni_panics() {
    nanosm::Launch launch;
    launch.active_warps = 1;
    nanosm::Machine machine(canonical_profile(), compile(R"(
        sread r0, LANE_ID
        setp.lt p0, r0, 8
        @p0 bra.uni split
        halt
    split:
        halt
    )"), std::move(launch));
    try {
        machine.run();
        throw std::runtime_error("nonuniform bra.uni executed without divergence support");
    } catch (const nanosm::SimError& error) {
        require(
            std::string(error.what()).find("bra.uni predicate is not warp-uniform") != std::string::npos,
            "nonuniform bra.uni produced the wrong PANIC");
    }
}

void test_branch_boundary_drains_async_work() {
    const auto program = nanosm::compile_qtx(R"(
        mov r0, 0
        ldg r1, [r0]
        bra done
    done:
        halt
    )", canonical_profile());
    require(program.instructions[2].control.wait_mask == 1, "branch did not drain its live load depbar");

    const auto fixed = nanosm::compile_qtx(R"(
        mov r0, 1
        fadd r1, r0, r0
        bra done
    done:
        halt
    )", canonical_profile());
    require(fixed.instructions[2].control.stall == 3, "branch did not drain a live fadd writeback");
}

void test_vadd() {
    constexpr uint32_t n = 64;
    constexpr uint32_t b = n * 4;
    constexpr uint32_t c = n * 8;
    auto config = canonical_profile();
    nanosm::Launch launch;
    launch.active_warps = n / config.lanes;
    auto& gmem = image_bytes(launch.gmem_image, c + n * 4);
    for (uint32_t i = 0; i < n; ++i) {
        write_u32(gmem, i * 4, nanosm::float_bits(static_cast<float>(i)));
        write_u32(gmem, b + i * 4, nanosm::float_bits(static_cast<float>(2 * i)));
    }
    nanosm::Machine machine(
        config, nanosm::compile_and_assemble_file("kernels/vadd.qtx", config), std::move(launch));
    machine.run();
    for (uint32_t i = 0; i < n; ++i) {
        require(
            nanosm::bits_float(machine.read_gmem_u32(c + i * 4)) == static_cast<float>(3 * i),
            "vadd output mismatch at " + std::to_string(i));
    }
}

void test_sparse_launch_images() {
    nanosm::Launch launch;
    launch.active_warps = 1;
    launch.shmem_image.push_back({64, std::vector<uint8_t>(4, 0)});
    launch.gmem_image.push_back({128, std::vector<uint8_t>(4, 0)});
    write_u32(launch.shmem_image.back().bytes, 0, 0x11223344u);
    write_u32(launch.gmem_image.back().bytes, 0, 0x55667788u);
    nanosm::Machine machine(canonical_profile(), compile("halt"), std::move(launch));
    machine.run();
    require(machine.read_shmem_u32(64) == 0x11223344u, "sparse shmem image landed incorrectly");
    require(machine.read_gmem_u32(128) == 0x55667788u, "sparse gmem image landed incorrectly");
    require(machine.read_shmem_u32(0) == 0, "unspecified shmem was not zero");
    require(machine.read_gmem_u32(0) == 0, "unspecified gmem was not zero");
}

void test_gmem_coalesces_to_sectors() {
    nanosm::Launch launch;
    launch.active_warps = 1;
    image_bytes(launch.gmem_image, 64);
    nanosm::Machine machine(canonical_profile(), compile(R"(
        sread r0, LANE_ID
        shli  r0, r0, 2
        ldg   r1, [r0]
        halt
    )"), std::move(launch));
    machine.run();
    require(
        machine.trace().summary().gmem_sectors == 2,
        "16 contiguous u32 lanes did not coalesce to two 32B sectors");
}

void test_gmem_command_must_fit_queue() {
    auto config = canonical_profile();
    config.gmem_queue_sectors = 1;
    nanosm::Launch launch;
    launch.active_warps = 1;
    image_bytes(launch.gmem_image, 64);
    nanosm::Machine machine(config, compile(R"(
        sread r0, LANE_ID
        shli  r0, r0, 2
        ldg   r1, [r0]
        halt
    )", config), std::move(launch));
    try {
        machine.run();
        throw std::runtime_error("oversized gmem command stalled forever instead of panicking");
    } catch (const nanosm::SimError& error) {
        require(
            std::string(error.what()).find("exceeds controller queue capacity") != std::string::npos,
            "oversized gmem command produced the wrong panic");
    }
}

void test_gmem_queue_pressure_structurally_stalls() {
    auto config = canonical_profile();
    config.gmem_queue_sectors = 2;
    config.gmem_latency_dram_ticks = 4;
    nanosm::Launch launch;
    launch.active_warps = 2;
    image_bytes(launch.gmem_image, 128);
    nanosm::Machine machine(config, compile(R"(
        sread r0, THREAD_ID
        shli  r0, r0, 2
        ldg   r1, [r0]
        halt
    )", config), std::move(launch));
    machine.run();
    require(
        machine.trace().summary().warp_cycles[static_cast<std::size_t>(nanosm::WarpState::structural)] != 0,
        "finite gmem queue never produced a structural issue stall");
}

void test_profile_stamp_rejects_mismatch() {
    const auto profile = nanosm::load_device_profile("profiles/nanosm-v0.json");
    const auto program = nanosm::compile_qtx("mov r0, 1\nhalt", profile);
    const auto assembly = nanosm::emit_asm(program);
    const auto kernel = nanosm::assemble_asm(assembly, profile);
    require(kernel.profile_name == profile.name, "profile name was not stamped into kernel");
    require(
        kernel.profile_fingerprint == nanosm::device_profile_fingerprint(profile),
        "profile fingerprint was not stamped into kernel");

    auto other = profile;
    ++other.timing.fadd;
    try {
        (void)nanosm::assemble_asm(assembly, other);
        throw std::runtime_error("qtxas accepted assembly scheduled for another profile");
    } catch (const nanosm::SimError& error) {
        require(
            std::string(error.what()).find("does not match") != std::string::npos,
            "profile mismatch produced the wrong error");
    }

    nanosm::Launch launch;
    launch.active_warps = 1;
    try {
        nanosm::Machine machine(other, kernel, std::move(launch));
        throw std::runtime_error("simulator accepted a kernel for another profile");
    } catch (const nanosm::SimError& error) {
        require(
            std::string(error.what()).find("does not match simulator") != std::string::npos,
            "simulator profile mismatch produced the wrong error");
    }
}

void test_dram_uses_its_own_clock() {
    auto config = canonical_profile();
    config.core_clock_hz = 4;
    config.dram_clock_hz = 1;
    config.gmem_latency_dram_ticks = 0;
    config.gmem_bus_bytes_per_dram_tick = 16;
    nanosm::Launch launch;
    launch.active_warps = 1;
    image_bytes(launch.gmem_image, 64);
    nanosm::Machine machine(config, compile(R"(
        sread r0, LANE_ID
        shli  r0, r0, 2
        ldg   r1, [r0]
        halt
    )", config), std::move(launch), {.timeline = true, .debug = true});
    machine.run();
    std::vector<uint64_t> beat_cycles;
    for (const auto& event : machine.trace().events()) {
        if (event.category == nanosm::TraceCategory::gmem && event.name == "bus beat") {
            beat_cycles.push_back(event.start_cycle);
        }
    }
    require(beat_cycles.size() == 4, "two sectors did not take four 16-byte DRAM beats");
    require(beat_cycles[1] - beat_cycles[0] == 4, "DRAM beats advanced at the core clock");
}

void test_lds_is_nonblocking_and_depbar_waits() {
    const auto program = nanosm::compile_qtx(R"(
        mov  r0, 0
        lds  r1, [r0]
        lds  r2, [r0 + 64]
        fadd r4, r1, r2
        halt
    )", canonical_profile());
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.shmem_image, 128);
    write_u32(shmem, 0, nanosm::float_bits(2.0f));
    write_u32(shmem, 64, nanosm::float_bits(3.0f));
    nanosm::Machine machine(
        canonical_profile(),
        nanosm::assemble_asm(nanosm::emit_asm(program), canonical_profile()),
        std::move(launch),
        {.timeline = true});
    machine.run();

    std::vector<uint64_t> load_issue_cycles;
    bool saw_depbar_wait = false;
    for (const auto& event : machine.trace().events()) {
        if (event.category == nanosm::TraceCategory::instruction && event.name == "lds") {
            load_issue_cycles.push_back(event.start_cycle);
        }
    }
    saw_depbar_wait = machine.trace().summary().warp_cycles[
        static_cast<std::size_t>(nanosm::WarpState::depbar)] != 0;
    require(load_issue_cycles.size() == 2, "did not issue two loads");
    require(load_issue_cycles[1] == load_issue_cycles[0] + 1, "lds blocked the warp instead of retiring");
    require(saw_depbar_wait, "consumer never stalled on its depbar mask");
}

void test_missing_wait_panics_instead_of_scoreboarding() {
    const auto& profile = canonical_profile();
    const auto assembly =
        ".profile " + profile.name + ", 0x" +
        nanosm::format_profile_fingerprint(nanosm::device_profile_fingerprint(profile)) + R"(
        mov  r0, 0 {pred=PT, stall=1, set=-, wait=-}
        lds  r1, [r0] {pred=PT, stall=0, set=sb0, wait=-}
        fadd r4, r1, r2 {pred=PT, stall=4, set=-, wait=-}
        halt {pred=PT, stall=0, set=-, wait=sb0}
    )";
    const auto kernel = nanosm::assemble_asm(assembly, profile);
    nanosm::Launch launch;
    launch.active_warps = 1;
    image_bytes(launch.shmem_image, 64);
    nanosm::Machine machine(canonical_profile(), kernel, std::move(launch));
    try {
        machine.run();
        throw std::runtime_error("missing depbar wait was silently scoreboard-stalled");
    } catch (const nanosm::SimError& error) {
        require(
            std::string(error.what()).find("RF RAW hazard") != std::string::npos,
            "missing wait produced the wrong panic");
    }
}

void test_broadcast_is_one_round() {
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.shmem_image, 4);
    write_u32(shmem, 0, 0x12345678u);
    nanosm::Machine machine(canonical_profile(), compile(R"(
        mov  r0, 0
        lds  r1, [r0]
        halt
    )"), std::move(launch));
    machine.run();
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(machine.read_reg(0, lane, 1) == 0x12345678u, "broadcast value mismatch");
    }
    require(machine.trace().summary().shmem_rounds == 1, "broadcast did not use exactly one bank round");
}

void test_bank_conflict_serializes() {
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.shmem_image, 16 * 64);
    for (uint32_t lane = 0; lane < 16; ++lane) write_u32(shmem, lane * 64, lane + 10);
    nanosm::Machine machine(canonical_profile(), compile(R"(
        sread r0, LANE_ID
        shli  r0, r0, 6
        lds   r1, [r0]
        halt
    )"), std::move(launch));
    machine.run();
    require(
        machine.trace().summary().shmem_rounds == 16,
        "sixteen-way bank conflict did not serialize to sixteen rounds");
}

void test_write_collision_panics() {
    nanosm::Launch launch;
    launch.active_warps = 1;
    image_bytes(launch.shmem_image, 4);
    nanosm::Machine machine(canonical_profile(), compile(R"(
        mov r0, 0
        mov r1, 7
        sts [r0], r1
        halt
    )"), std::move(launch));
    try {
        machine.run();
        throw std::runtime_error("same-word write collision did not panic");
    } catch (const nanosm::SimError& error) {
        require(
            std::string(error.what()).find("multiple lanes write the same shmem word") != std::string::npos,
            "write collision produced the wrong panic");
    }
}

void test_gmem_write_collision_panics_without_writing() {
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& gmem = image_bytes(launch.gmem_image, 4);
    write_u32(gmem, 0, 0x12345678);
    nanosm::Machine machine(canonical_profile(), compile(R"(
        mov   r0, 0
        sread r1, LANE_ID
        stg   [r0], r1
        halt
    )"), std::move(launch));
    try {
        machine.run();
        throw std::runtime_error("same-word gmem write collision did not panic");
    } catch (const nanosm::SimError& error) {
        require(
            std::string(error.what()).find("multiple lanes write the same gmem word") != std::string::npos,
            "gmem write collision produced the wrong panic");
    }
    require(machine.read_gmem_u32(0) == 0x12345678, "colliding gmem store changed memory");
}

void test_profile_defaults_are_invalid() {
    nanosm::DeviceProfile profile;
    require(profile.timing.alu == 0 && profile.timing.fadd == 0, "profile inherited ISA timing defaults");
    try {
        nanosm::validate_device_profile(profile);
        throw std::runtime_error("default-constructed profile passed validation");
    } catch (const nanosm::ProfileError&) {
    }
}

void test_profile_is_validated_before_machine_allocation() {
    auto profile = canonical_profile();
    profile.timing.alu = 0;
    profile.timing.fadd = 0;
    profile.gmem_bytes = std::numeric_limits<uint64_t>::max();
    nanosm::Launch launch;
    launch.active_warps = 1;
    try {
        nanosm::Machine machine(profile, compile("halt"), std::move(launch));
        throw std::runtime_error("invalid profile reached machine allocation");
    } catch (const nanosm::SimError& error) {
        require(
            std::string(error.what()).find("device profile") != std::string::npos,
            "invalid profile did not produce deterministic profile PANIC");
    }
}

void test_host_allocation_failure_is_a_panic() {
    auto profile = canonical_profile();
    profile.gmem_bytes = std::numeric_limits<uint64_t>::max();
    const auto kernel = compile("halt", profile);
    nanosm::Launch launch;
    launch.active_warps = 1;
    try {
        nanosm::Machine machine(profile, kernel, std::move(launch));
        throw std::runtime_error("impossible machine allocation unexpectedly succeeded");
    } catch (const nanosm::SimError& error) {
        const std::string message = error.what();
        require(
            message.find("host memory") != std::string::npos ||
                message.find("host container limits") != std::string::npos,
            "host allocation failure escaped deterministic PANIC conversion");
    }
}

void test_trace_attribution_and_limit() {
    nanosm::Launch launch;
    launch.active_warps = 1;
    nanosm::Machine machine(
        canonical_profile(),
        compile("mov r0, 1\nhalt"),
        std::move(launch),
        {.timeline = true, .max_events = 1});
    machine.run();
    const auto& trace = machine.trace();
    const auto& summary = trace.summary();
    require(
        summary.scheduler_issued_cycles + summary.scheduler_idle_cycles +
                summary.scheduler_draining_cycles ==
            summary.cycles,
        "scheduler attribution does not sum to total cycles");
    uint64_t warp_cycles = 0;
    for (const auto cycles : summary.warp_cycles) warp_cycles += cycles;
    require(warp_cycles == summary.cycles, "warp attribution does not sum to cycles * warps");
    require(trace.events().size() == 1, "timeline exceeded its hard event limit");
    require(trace.truncated(), "timeline limit did not report truncation");
}

void test_perfetto_overlapping_commands_are_async() {
    nanosm::Launch launch;
    launch.active_warps = 2;
    image_bytes(launch.gmem_image, 128);
    nanosm::Machine machine(canonical_profile(), compile(R"(
        sread r0, THREAD_ID
        shli  r0, r0, 2
        ldg   r1, [r0]
        halt
    )"), std::move(launch), {.timeline = true});
    machine.run();
    const std::string path = "build/test_async_trace.json";
    machine.trace().write_perfetto(path);

    std::ifstream input(path);
    require(static_cast<bool>(input), "trace file was not written");
    nlohmann::json document;
    input >> document;

    // Complete 'X' slices that partially overlap on one tid are ambiguous to nest and
    // trigger Perfetto's slice_spill_overlapping_complete_event; overlapping lifetimes
    // must ship as async begin/end pairs instead.
    std::map<uint64_t, std::vector<std::pair<uint64_t, uint64_t>>> spans;
    std::size_t begins = 0;
    std::size_t ends = 0;
    for (const auto& event : document.at("traceEvents")) {
        const auto ph = event.at("ph").get<std::string>();
        if (ph == "b") ++begins;
        if (ph == "e") ++ends;
        if (ph != "X") continue;
        // Compare the exact cycle integers from args: recomputing spans from the
        // floating-point ts/dur pair manufactures 1-ULP overlaps that do not exist.
        const auto start = event.at("args").at("start_cycle").get<uint64_t>();
        const auto duration = event.at("args").at("duration_cycles").get<uint64_t>();
        spans[event.at("tid").get<uint64_t>()].emplace_back(start, start + duration);
    }
    for (auto& [tid, intervals] : spans) {
        std::sort(intervals.begin(), intervals.end());
        for (std::size_t i = 1; i < intervals.size(); ++i) {
            require(
                intervals[i].first >= intervals[i - 1].second,
                "overlapping complete slices on tid " + std::to_string(tid));
        }
    }
    require(begins >= 2 && begins == ends, "concurrent gmem commands did not emit async pairs");
}

} // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"encoding round trip", test_encoding_round_trip},
        {"FNV-1a vectors", test_fnv1a_64_vectors},
        {"high register metadata", test_high_register_kernel_metadata},
        {"RF-derived residency", test_rf_capacity_determines_residency},
        {"compiler controls", test_compiler_controls},
        {"consumer-distance stalls", test_stalls_track_consumer_distance},
        {"counted loop and relocation", test_counted_loop_and_label_relocation},
        {"uniform role dispatch", test_uniform_role_dispatch},
        {"nonuniform bra.uni PANIC", test_nonuniform_bra_uni_panics},
        {"branch drains async work", test_branch_boundary_drains_async_work},
        {"vadd", test_vadd},
        {"sparse launch images", test_sparse_launch_images},
        {"gmem coalescing", test_gmem_coalesces_to_sectors},
        {"gmem command capacity", test_gmem_command_must_fit_queue},
        {"gmem queue pressure", test_gmem_queue_pressure_structurally_stalls},
        {"profile stamp", test_profile_stamp_rejects_mismatch},
        {"DRAM clock domain", test_dram_uses_its_own_clock},
        {"nonblocking lds", test_lds_is_nonblocking_and_depbar_waits},
        {"missing wait panic", test_missing_wait_panics_instead_of_scoreboarding},
        {"broadcast", test_broadcast_is_one_round},
        {"bank conflict", test_bank_conflict_serializes},
        {"write collision", test_write_collision_panics},
        {"gmem write collision", test_gmem_write_collision_panics_without_writing},
        {"invalid profile defaults", test_profile_defaults_are_invalid},
        {"validate before allocation", test_profile_is_validated_before_machine_allocation},
        {"allocation failure PANIC", test_host_allocation_failure_is_a_panic},
        {"trace attribution and limit", test_trace_attribution_and_limit},
        {"perfetto async commands", test_perfetto_overlapping_commands_are_async},
    };
    std::size_t passed = 0;
    for (const auto& [name, test] : tests) {
        try {
            test();
            nanosm::log("PASS {}", name);
            ++passed;
        } catch (const std::exception& error) {
            nanosm::err("FAIL {}: {}", name, error.what());
        }
    }
    nanosm::log("{}/{} tests passed", passed, tests.size());
    return passed == tests.size() ? 0 : 1;
}

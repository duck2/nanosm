#include "sim.hpp"
#include "mma_numeric.hpp"
#include "hash.hpp"
#include "log.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

// Every tick-by-tick loop here ticks through this: a machine that stops making
// progress fails its test at the cap instead of hanging the binary (with the
// sanitizer compiled out, a bug it would have caught can loop forever).
constexpr uint64_t test_cycle_cap = 200000;

void tick_capped(nanosm::Machine& machine) {
    if (machine.cycle() >= test_cycle_cap) {
        throw std::runtime_error("cycle cap " + std::to_string(test_cycle_cap) + ": the machine stopped making progress");
    }
    machine.tick();
}


void write_u32(std::vector<uint8_t>& image, uint32_t address, uint32_t value) {
    std::memcpy(image.data() + address, &value, sizeof(value));
}

void write_u16(std::vector<uint8_t>& image, uint32_t address, uint16_t value) {
    std::memcpy(image.data() + address, &value, sizeof(value));
}

void write_u64(std::vector<uint8_t>& image, uint32_t address, uint64_t value) {
    std::memcpy(image.data() + address, &value, sizeof(value));
}

void write_tma_descriptor(
    std::vector<uint8_t>& image, uint32_t address, const nanosm::TmaDescriptor& descriptor) {
    const auto bytes = nanosm::encode_tma_descriptor(descriptor);
    std::memcpy(image.data() + address, bytes.data(), bytes.size());
}

std::vector<uint8_t>& image_bytes(
    std::vector<nanosm::ImageSegment>& image, std::size_t size) {
    image.push_back({0, std::vector<uint8_t>(size, 0)});
    return image.back().bytes;
}

// A test that must write a file writes it here: the system temp directory
// plus this process's id and a per-process counter, so overlapping runs of
// this binary (ctest -j, the sanitizer ON/OFF twin, build and build_nosan at
// once) never share a path. The file is removed when the guard goes away.
class ScratchFile {
public:
    explicit ScratchFile(const std::string& stem) {
        static uint64_t counter = 0;
#ifdef _WIN32
        const auto pid = static_cast<long long>(_getpid());
#else
        const auto pid = static_cast<long long>(getpid());
#endif
        path_ = std::filesystem::temp_directory_path() /
            ("nanosm_tests_" + stem + "_" + std::to_string(pid) + "_" +
                std::to_string(counter++) + ".json");
    }
    ScratchFile(const ScratchFile&) = delete;
    ScratchFile& operator=(const ScratchFile&) = delete;
    ~ScratchFile() {
        std::error_code ignored;
        std::filesystem::remove(path_, ignored);
    }
    [[nodiscard]] std::string string() const { return path_.string(); }

private:
    std::filesystem::path path_;
};

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

void test_mbar_word_encoding() {
    nanosm::MbarState state;
    state.phase = true;
    state.valid = true;
    state.expected = 0x1234;
    state.remaining = 0x2345;
    state.tx_bytes = 0x89abcdef;
    const auto word = nanosm::encode_mbar_state(state);
    require(
        word == 0x89abcdef468a48d3ull,
        "mbar word layout changed without updating its architectural encoding test");
    const auto decoded = nanosm::decode_mbar_state(word);
    require(
        decoded.phase == state.phase && decoded.valid == state.valid &&
            decoded.expected == state.expected && decoded.remaining == state.remaining &&
            decoded.tx_bytes == state.tx_bytes,
        "mbar word encoding did not round trip");
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

void test_mbar_try_wait_sleeps_and_engine_completion_wakes() {
    nanosm::Launch launch;
    launch.active_warps = 1;
    image_bytes(launch.shmem_image, 128);
    nanosm::Machine machine(canonical_profile(), compile(R"(
        mov r0, 64
        mbar.init [r0], 1
        mbar.arrive.expect_tx [r0], 8
    wait:
        mbar.try_wait p0, [r0], 0
        @!p0 bra.uni wait
        mov r1, 9
        halt
    )"), std::move(launch), {.timeline = true});

    while (machine.trace().summary().mbar_sleeps == 0 && machine.cycle() < 32) machine.tick();
    require(machine.trace().summary().mbar_sleeps == 1, "mbar.try_wait never parked the warp");
    machine.complete_mbar_tx(64, 8, 0);
    machine.run();

    const auto state = machine.read_mbar_state(64);
    require(state.valid, "mbar lost its valid bit");
    require(state.phase, "complete_tx did not flip the mbar phase");
    require(state.expected == 1 && state.remaining == 1, "mbar did not reload its arrival count");
    require(state.tx_bytes == 0, "complete_tx did not consume transaction bytes");
    require(machine.read_reg(0, 0, 1) == 9, "woken warp did not resume after mbar.try_wait");
    require(machine.trace().summary().mbar_rmw_requests == 3, "wrong number of mbar RMW requests");
    require(machine.trace().summary().mbar_phase_changes == 1, "phase transition was not traced");
    require(machine.trace().summary().mbar_wakes == 1, "phase broadcast did not wake the warp");
    require(machine.trace().summary().mbar_timeouts == 0, "exact wake incorrectly timed out");
    std::size_t rmw_slices = 0;
    for (const auto& event : machine.trace().events()) {
        if (event.name != "mbar RMW") continue;
        ++rmw_slices;
        require(event.duration_cycles == 2, "mbar RMW did not occupy exactly two port-A cycles");
    }
    require(rmw_slices == 3, "trace did not contain every mbar RMW service interval");
}

void test_mbar_test_wait_broadcasts_predicate() {
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.shmem_image, 128);
    nanosm::MbarState initial;
    initial.valid = true;
    initial.phase = true;
    initial.expected = 1;
    initial.remaining = 1;
    write_u64(shmem, 64, nanosm::encode_mbar_state(initial));
    nanosm::Machine machine(canonical_profile(), compile(R"(
        mov r0, 64
        mbar.test_wait p0, [r0], 0
        @p0 bra.uni complete
        mov r1, 1
        halt
    complete:
        mov r1, 2
        halt
    )"), std::move(launch));
    machine.run();
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(machine.read_reg(0, lane, 1) == 2, "mbar.test_wait did not broadcast true");
    }
}

void test_mbar_try_wait_immediate_success() {
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.shmem_image, 128);
    nanosm::MbarState initial;
    initial.valid = true;
    initial.phase = true;
    initial.expected = 1;
    initial.remaining = 1;
    write_u64(shmem, 64, nanosm::encode_mbar_state(initial));
    nanosm::Machine machine(canonical_profile(), compile(R"(
        mov r0, 64
        mov r2, 0
        mbar.try_wait p0, [r0], r2
        @p0 bra.uni complete
        mov r1, 1
        halt
    complete:
        mov r1, 2
        halt
    )"), std::move(launch));
    machine.run();
    require(machine.read_reg(0, 0, 1) == 2, "immediate try_wait did not write true");
    const auto& summary = machine.trace().summary();
    require(summary.mbar_initial_probes == 1, "try_wait initial probe was not traced");
    require(summary.mbar_immediate_successes == 1, "immediate try_wait was not counted");
    require(summary.mbar_sleeps == 0, "complete try_wait incorrectly parked");
}

void test_mbar_try_wait_times_out_false() {
    auto profile = canonical_profile();
    profile.mbar_try_wait_timeout_cycles = 3;
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.shmem_image, 128);
    nanosm::MbarState initial;
    initial.valid = true;
    initial.expected = 1;
    initial.remaining = 1;
    write_u64(shmem, 64, nanosm::encode_mbar_state(initial));
    nanosm::Machine machine(profile, compile(R"(
        mov r0, 64
        mbar.try_wait p0, [r0], 0
        @p0 bra.uni unexpected
        mov r1, 7
        halt
    unexpected:
        mov r1, 9
        halt
    )", profile), std::move(launch));
    machine.run();
    require(machine.read_reg(0, 0, 1) == 7, "timed-out try_wait did not write false");
    require(machine.trace().summary().mbar_timeouts == 1, "try_wait timeout was not traced");
    require(machine.trace().summary().mbar_wakes == 0, "timeout was counted as exact wake");
}

void test_mbar_phase_flip_wins_on_try_wait_deadline() {
    auto profile = canonical_profile();
    profile.mbar_try_wait_timeout_cycles = 2;
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.shmem_image, 128);
    nanosm::MbarState initial;
    initial.valid = true;
    initial.expected = 1;
    initial.remaining = 0;
    initial.tx_bytes = 8;
    write_u64(shmem, 64, nanosm::encode_mbar_state(initial));
    nanosm::Machine machine(profile, compile(R"(
        mov r0, 64
        mbar.try_wait p0, [r0], 0
        @p0 bra.uni complete
        mov r1, 1
        halt
    complete:
        mov r1, 2
        halt
    )", profile), std::move(launch));

    while (machine.trace().summary().mbar_sleeps == 0 && machine.cycle() < 32) machine.tick();
    require(machine.trace().summary().mbar_sleeps == 1, "deadline test never parked");
    machine.complete_mbar_tx(64, 8, 0);
    machine.run();
    require(machine.read_reg(0, 0, 1) == 2, "phase flip on deadline did not return true");
    require(machine.trace().summary().mbar_wakes == 1, "deadline phase flip was not an exact wake");
    require(machine.trace().summary().mbar_timeouts == 0, "deadline phase flip lost to timeout");
}

void test_mbar_try_wait_slots_limit_parked_warps() {
    auto profile = canonical_profile();
    profile.mbar_wait_slots = 1;
    profile.mbar_try_wait_timeout_cycles = 4;
    nanosm::Launch launch;
    launch.active_warps = 2;
    auto& shmem = image_bytes(launch.shmem_image, 128);
    nanosm::MbarState initial;
    initial.valid = true;
    initial.expected = 1;
    initial.remaining = 1;
    write_u64(shmem, 64, nanosm::encode_mbar_state(initial));
    nanosm::Machine machine(profile, compile(R"(
        mov r0, 64
        mbar.try_wait p0, [r0], 0
        halt
    )", profile), std::move(launch));
    machine.run();
    const auto& summary = machine.trace().summary();
    require(summary.mbar_initial_probes == 2, "not every warp eventually obtained a wait slot");
    require(summary.mbar_timeouts == 2, "parked warps did not release their wait slots");
    require(summary.mbar_slot_full_stalls != 0, "wait-slot exhaustion was not attributed");
    require(
        summary.warp_cycles[static_cast<std::size_t>(nanosm::WarpState::mbar_wait_slot)] != 0,
        "slot-full cycles were not a distinct warp state");
}

void test_staged_pipeline_kernel() {
    // The everything-so-far kernel: warp 0 stages gmem tiles into double-buffered shmem
    // through the RF, warp 1 consumes and stores back; full/empty barrier pairs, try_wait
    // retry loops with xori parity toggling, counted loops, role dispatch.
    constexpr uint32_t tiles = 8;
    constexpr uint32_t elements = tiles * 16; // one lane-width tile per pass
    nanosm::Launch launch;
    launch.active_warps = 2;
    auto& shmem = image_bytes(launch.shmem_image, 2048);
    nanosm::MbarState barrier;
    barrier.valid = true;
    barrier.expected = 1;
    barrier.remaining = 1;
    for (const uint32_t address : {1024u, 1032u, 1040u, 1048u}) {
        write_u64(shmem, address, nanosm::encode_mbar_state(barrier));
    }
    auto& gmem = image_bytes(launch.gmem_image, 8192 + elements * 4);
    for (uint32_t i = 0; i < elements; ++i) {
        write_u32(gmem, i * 4, nanosm::float_bits(static_cast<float>(i)));
        write_u32(gmem, 4096 + i * 4, nanosm::float_bits(static_cast<float>(2 * i)));
    }
    nanosm::LaunchImage input{
        nanosm::compile_and_assemble_file("kernels/pipeline.qtx", canonical_profile()),
        std::move(launch),
    };
    const auto serialized = nanosm::emit_launch_image(input);
    auto loaded = nanosm::load_launch_image(serialized);
    require(loaded.launch.active_warps == 2, "launch image lost its active warp count");
    require(loaded.launch.shmem_image.size() == 1, "launch image lost its shmem segment");
    require(loaded.launch.gmem_image.size() == 1, "launch image lost its gmem segment");
    require(
        nanosm::emit_launch_image(loaded) == serialized,
        "launch image serialization is not canonical");
    nanosm::Machine machine(
        canonical_profile(), std::move(loaded.kernel), std::move(loaded.launch));
    machine.run();
    for (uint32_t i = 0; i < elements; ++i) {
        require(
            nanosm::bits_float(machine.read_gmem_u32(8192 + i * 4)) == static_cast<float>(3 * i),
            "pipeline output mismatch at " + std::to_string(i));
    }
    const auto& summary = machine.trace().summary();
    require(summary.mbar_sleeps != 0, "pipeline never parked a warp");
    require(summary.mbar_wakes != 0, "pipeline never took an exact wake");
    require(summary.mbar_timeouts != 0, "gmem latency never outlasted the try_wait timeout");
    require(summary.gmem_commands == 3 * tiles, "wrong gmem command count for the pipeline");
    uint64_t warp_cycles = 0;
    for (const auto cycles : summary.warp_cycles) warp_cycles += cycles;
    require(warp_cycles == summary.cycles * 2, "pipeline warp attribution does not sum");
}

void test_launch_image_rejects_bad_segment_hex() {
    nanosm::LaunchImage input;
    input.kernel = compile("halt");
    input.launch.active_warps = 1;
    input.launch.shmem_image.push_back({0, {0xab}});
    auto serialized = nanosm::emit_launch_image(input);
    const auto field = serialized.find("\"bytes\": \"ab\"");
    const auto byte = serialized.find("ab", field);
    require(byte != std::string::npos, "launch test could not find its byte payload");
    serialized[byte] = 'z';
    try {
        nanosm::load_launch_image(serialized);
        throw std::runtime_error("launch image accepted malformed segment hex");
    } catch (const nanosm::SimError& error) {
        require(
            std::string(error.what()).find("non-hex") != std::string::npos,
            "malformed segment produced the wrong launch-image error");
    }
}

void test_profile_can_disable_mbar_try_wait() {
    auto profile = canonical_profile();
    profile.mbar_wait_slots = 0;
    profile.mbar_try_wait_timeout_cycles = 0;
    nanosm::validate_device_profile(profile);
    try {
        nanosm::compile_qtx("mov r0, 64\nmbar.try_wait p0, [r0], 0\nhalt", profile);
        throw std::runtime_error("qtxc accepted try_wait with no wait slots");
    } catch (const nanosm::SimError& error) {
        require(
            std::string(error.what()).find("unsupported") != std::string::npos,
            "disabled try_wait produced the wrong error");
    }

    profile.mbar_wait_slots = 1;
    try {
        nanosm::validate_device_profile(profile);
        throw std::runtime_error("profile accepted wait slots without a timeout");
    } catch (const nanosm::ProfileError&) {
    }
}

void test_warp_handoff_through_mbar() {
    // The slice's acceptance kernel: producer writes a tile, arrives; consumer sleeps,
    // wakes on the producer's arrival, reads, transforms, writes back. The barrier is a
    // launch-time object (poked via the image) because runtime init has a use-before-init
    // race between warps -- exactly why design.md makes barriers launch-time.
    nanosm::Launch launch;
    launch.active_warps = 2;
    auto& shmem = image_bytes(launch.shmem_image, 512);
    nanosm::MbarState initial;
    initial.valid = true;
    initial.expected = 1;
    initial.remaining = 1;
    write_u64(shmem, 256, nanosm::encode_mbar_state(initial));
    nanosm::Machine machine(canonical_profile(), compile(R"(
        sread r0, WARP_ID
        setp.eq p0, r0, 0
        @p0 bra.uni producer
        mov r1, 256
    consume_wait:
        mbar.try_wait p1, [r1], 0
        @!p1 bra.uni consume_wait
        sread r2, LANE_ID
        shli r2, r2, 2
        lds r3, [r2]
        addi r3, r3, 100
        sts [r2 + 320], r3
        halt
    producer:
        sread r2, LANE_ID
        shli r2, r2, 2
        sread r4, LANE_ID
        addi r4, r4, 7
        sts [r2], r4
        mov r1, 256
        mbar.arrive [r1]
        halt
    )"), std::move(launch));
    machine.run();
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(
            machine.read_shmem_u32(320 + lane * 4) == lane + 107,
            "handoff produced wrong data at lane " + std::to_string(lane));
    }
    const auto& summary = machine.trace().summary();
    require(summary.mbar_sleeps == 1, "consumer never slept on the barrier");
    require(summary.mbar_wakes == 1, "producer's arrival did not wake the consumer");
    require(
        summary.warp_cycles[static_cast<std::size_t>(nanosm::WarpState::mbar_wait)] != 0,
        "sleeping consumer cycles were not attributed to mbar wait");
    uint64_t warp_cycles = 0;
    for (const auto cycles : summary.warp_cycles) warp_cycles += cycles;
    require(
        warp_cycles == summary.cycles * 2,
        "warp attribution does not sum to total with a sleeping warp");
}

void test_mbar_requires_uniform_address() {
    nanosm::Launch launch;
    launch.active_warps = 1;
    image_bytes(launch.shmem_image, 128);
    nanosm::Machine machine(canonical_profile(), compile(R"(
        sread r0, LANE_ID
        shli r0, r0, 3
        mbar.init [r0], 1
        halt
    )"), std::move(launch));
    try {
        machine.run();
        throw std::runtime_error("nonuniform mbar address was accepted");
    } catch (const nanosm::SimError& error) {
        require(
            std::string(error.what()).find("mbar address is not warp-uniform") != std::string::npos,
            "nonuniform mbar address produced the wrong PANIC");
    }
}

void test_mbar_arrival_underflow_panics() {
    nanosm::Launch launch;
    launch.active_warps = 1;
    image_bytes(launch.shmem_image, 128);
    nanosm::Machine machine(canonical_profile(), compile(R"(
        mov r0, 64
        mbar.init [r0], 1
        mbar.arrive.expect_tx [r0], 8
        mbar.arrive [r0]
        halt
    )"), std::move(launch));
    try {
        machine.run();
        throw std::runtime_error("mbar arrival underflow was accepted");
    } catch (const nanosm::SimError& error) {
        require(
            std::string(error.what()).find("arrival count underflow") != std::string::npos,
            "mbar arrival underflow produced the wrong PANIC");
    }
}

void test_mbar_reinit_with_waiter_panics() {
    nanosm::Launch launch;
    launch.active_warps = 2;
    auto& shmem = image_bytes(launch.shmem_image, 128);
    nanosm::MbarState initial;
    initial.valid = true;
    initial.expected = 1;
    initial.remaining = 1;
    write_u64(shmem, 64, nanosm::encode_mbar_state(initial));
    nanosm::Machine machine(canonical_profile(), compile(R"(
        mov r1, 64
        sread r0, WARP_ID
        setp.eq p0, r0, 0
        @p0 bra.uni waiter
        mbar.init [r1], 1
        halt
    waiter:
        mbar.try_wait p1, [r1], 0
        halt
    )"), std::move(launch));
    try {
        machine.run();
        throw std::runtime_error("mbar reinit with a live waiter was accepted");
    } catch (const nanosm::SimError& error) {
        require(
            std::string(error.what()).find("reinitializing an mbar with a live waiter") !=
                std::string::npos,
            "mbar reinit with a waiter produced the wrong PANIC");
    }
}

void test_mbar_complete_wrong_phase_panics() {
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.shmem_image, 128);
    nanosm::MbarState initial;
    initial.valid = true;
    initial.phase = true;
    initial.expected = 1;
    initial.remaining = 0;
    initial.tx_bytes = 8;
    write_u64(shmem, 64, nanosm::encode_mbar_state(initial));
    nanosm::Machine machine(canonical_profile(), compile("halt"), std::move(launch));
    machine.complete_mbar_tx(64, 8, 0);
    try {
        machine.run();
        throw std::runtime_error("complete_tx against the wrong phase was accepted");
    } catch (const nanosm::SimError& error) {
        require(
            std::string(error.what()).find("wrong phase") != std::string::npos,
            "wrong-phase complete_tx produced the wrong PANIC");
    }
}

void test_store_cannot_repurpose_mbar_with_waiter() {
    nanosm::Launch launch;
    launch.active_warps = 2;
    auto& shmem = image_bytes(launch.shmem_image, 128);
    nanosm::MbarState initial;
    initial.valid = true;
    initial.expected = 1;
    initial.remaining = 1;
    write_u64(shmem, 64, nanosm::encode_mbar_state(initial));
    nanosm::Machine machine(canonical_profile(), compile(R"(
        mov r1, 64
        sread r0, WARP_ID
        setp.eq p0, r0, 0
        @p0 bra.uni waiter
        mov r2, 5
        sts [r1], r2
        halt
    waiter:
        mbar.try_wait p1, [r1], 0
        halt
    )"), std::move(launch));
    try {
        machine.run();
        throw std::runtime_error("ordinary store repurposed an mbar with a live waiter");
    } catch (const nanosm::SimError& error) {
        require(
            std::string(error.what()).find("repurposing an mbar with a live waiter") !=
                std::string::npos,
            "ordinary mbar repurpose produced the wrong PANIC");
    }
}

// ---------------------------------------------------------------------------
// LSU (rfc09 B2): interleaved v8, scalar gather/scatter, ldg.uni, the staging
// pool, the backpressure chain, TMA port-A priority, read-stream separation.
// trash/lsu-b2-2026-10-04.md lists which tests replaced which.
// ---------------------------------------------------------------------------

// Coprime periods (31, 29): unique for 899 consecutive words, so a word in
// the wrong lane, register or sector cannot hide behind a repeat.
uint32_t pattern_word(uint32_t index) {
    return 0x80000000u | ((index % 31) << 16) | (index % 29);
}

void fill_pattern(std::vector<uint8_t>& image, uint32_t byte_address, uint32_t words, uint32_t first_index) {
    for (uint32_t i = 0; i < words; ++i) write_u32(image, byte_address + 4 * i, pattern_word(first_index + i));
}

void expect_sim_panic(nanosm::Machine& machine, const std::string& needle, const std::string& what) {
    try {
        machine.run();
    } catch (const nanosm::SimError& error) {
        require(std::string(error.what()).find(needle) != std::string::npos,
            what + " produced the wrong panic: " + error.what());
        return;
    }
    throw std::runtime_error(what + " did not panic");
}

struct LsuEventSpan {
    std::string name;
    std::string detail;
    uint64_t start = 0;
    uint64_t end = 0;
};

std::vector<LsuEventSpan> trace_spans(
    const nanosm::Machine& machine, nanosm::TraceCategory category, const std::string& name) {
    std::vector<LsuEventSpan> spans;
    for (const auto& event : machine.trace().events()) {
        if (event.kind != nanosm::TraceEventKind::slice || event.category != category) continue;
        if (!name.empty() && event.name != name) continue;
        spans.push_back({event.name, event.detail, event.start_cycle, event.start_cycle + event.duration_cycles});
    }
    std::sort(spans.begin(), spans.end(), [](const auto& a, const auto& b) { return a.start < b.start; });
    return spans;
}

void test_vadd() {
    // kernels/vadd.qtx: 8192 fp32 per vector, four warps, ldg.v8/fadd/stg.v8.
    constexpr uint32_t n = 8192;
    constexpr uint32_t b = 32768;
    constexpr uint32_t c = 65536;
    auto config = canonical_profile();
    nanosm::Launch launch;
    launch.active_warps = 4;
    auto& gmem = image_bytes(launch.gmem_image, c + n * 4 + 64);
    for (uint32_t i = 0; i < n; ++i) {
        write_u32(gmem, i * 4, nanosm::float_bits(static_cast<float>(i % 97)));
        write_u32(gmem, b + i * 4, nanosm::float_bits(static_cast<float>(2 * (i % 89))));
    }
    std::fill(gmem.begin() + c + n * 4, gmem.end(), 0xA5);
    nanosm::Machine machine(
        config, nanosm::compile_and_assemble_file("kernels/vadd.qtx", config), std::move(launch));
    machine.run();
    for (uint32_t i = 0; i < n; ++i) {
        require(
            nanosm::bits_float(machine.read_gmem_u32(c + i * 4)) ==
                static_cast<float>(i % 97 + 2 * (i % 89)),
            "vadd output mismatch at " + std::to_string(i));
    }
    for (uint32_t byte = c + n * 4; byte < c + n * 4 + 64; ++byte)
        require(machine.gmem()[byte] == 0xA5, "vadd wrote past C");
    const auto& summary = machine.trace().summary();
    require(summary.lsu_vector_loads == 128 && summary.lsu_vector_stores == 64 &&
            summary.lsu_sectors == 3 * n * 4 / 32 && summary.lsu_bytes == 3 * n * 4 &&
            summary.lsu_useful_bytes == summary.lsu_bytes && summary.gmem_bytes == 3 * n * 4,
        "vadd traffic is not exactly 3 x 32 KiB of whole sectors");
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

// Replaces "gmem coalescing" (16 contiguous lanes -> 2 sectors): there is no
// coalescer, so the same access is sixteen requests.
void test_lsu_scalar_ldg_is_one_request_per_lane() {
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& gmem = image_bytes(launch.gmem_image, 64);
    fill_pattern(gmem, 0, 16, 0);
    nanosm::Machine machine(canonical_profile(), compile(R"(
        sread r0, LANE_ID
        shli  r0, r0, 2
        ldg   r1, [r0]
        halt
    )"), std::move(launch));
    machine.run();
    for (uint32_t lane = 0; lane < 16; ++lane)
        require(machine.read_reg(0, lane, 1) == pattern_word(lane), "scalar ldg lane value mismatch");
    const auto& summary = machine.trace().summary();
    require(summary.gmem_sectors == 16 && summary.lsu_sectors == 16 && summary.lsu_scalar_loads == 1,
        "16 contiguous lanes were merged instead of issuing 16 sector requests");
    require(summary.lsu_useful_bytes == 64 && summary.lsu_bytes == 512,
        "scalar ldg useful/sector byte accounting is wrong");
}

// (a) Replaces "gmem v8 roundtrip" (row-per-lane): register rD+j, lane l
// <-> byte rA + 4*(16*j + l), both directions, with guard bytes.
void test_lsu_v8_interleaved_mapping() {
    constexpr uint32_t source = 1024;
    constexpr uint32_t target = 4096;
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& gmem = image_bytes(launch.gmem_image, 8192);
    fill_pattern(gmem, source, 128, 0);
    std::fill(gmem.begin() + target - 64, gmem.begin() + target + 512 + 64, 0xA5);
    std::string source_text = R"(
        mov r30, 1024
        ldg.v8 r8, [r30]
        sread r1, LANE_ID
        shli r2, r1, 8
    )";
    for (uint32_t j = 0; j < 8; ++j)
        source_text += "addi r" + std::to_string(16 + j) + ", r2, " + std::to_string(0x100000 + j * 0x10000) + "\n";
    source_text += "mov r31, 4096\nstg.v8 [r31], r16\nhalt\n";
    nanosm::Machine machine(canonical_profile(), compile(source_text), std::move(launch));
    machine.run();
    for (uint32_t j = 0; j < 8; ++j) {
        for (uint32_t lane = 0; lane < 16; ++lane) {
            // Written-out contract, independent of the implementation.
            const uint32_t word = 16 * j + lane;
            require(machine.read_reg(0, lane, 8 + j) == pattern_word(word),
                "ldg.v8 r" + std::to_string(8 + j) + " lane " + std::to_string(lane) +
                    " is not word 16*j + l of the block");
            require(machine.read_gmem_u32(target + 4 * word) == lane * 256 + 0x100000 + j * 0x10000,
                "stg.v8 r" + std::to_string(16 + j) + " lane " + std::to_string(lane) +
                    " did not land at word 16*j + l");
        }
    }
    for (uint32_t byte = target - 64; byte < target; ++byte)
        require(machine.gmem()[byte] == 0xA5, "stg.v8 damaged the guard before its block");
    for (uint32_t byte = target + 512; byte < target + 576; ++byte)
        require(machine.gmem()[byte] == 0xA5, "stg.v8 damaged the guard after its block");
    const auto& summary = machine.trace().summary();
    require(summary.lsu_vector_loads == 1 && summary.lsu_vector_stores == 1 &&
            summary.lsu_sectors == 32 && summary.gmem_sectors == 32 && summary.lsu_useful_bytes == 1024,
        "a v8 command is not exactly sixteen whole sectors");
}

// Replaces the old "gmem v8 source lifetime" with a uniform base.
void test_lsu_v8_store_source_lifetime() {
    const std::string text = R"(
        mov r0, 0
        mov r8, 17
        mov r9, 18
        mov r10, 19
        mov r11, 20
        mov r12, 21
        mov r13, 22
        mov r14, 23
        mov r15, 24
        stg.v8 [r0], r8
        mov r8, 99
        halt
    )";
    const auto scheduled = nanosm::compile_qtx(text, canonical_profile());
    require(scheduled.instructions[10].control.wait_mask != 0,
        "qtxc did not protect the deferred stg.v8 source tuple");
    nanosm::Launch launch;
    launch.active_warps = 1;
    image_bytes(launch.gmem_image, 512);
    nanosm::Machine machine(canonical_profile(), compile(text), std::move(launch));
    machine.run();
    for (uint32_t j = 0; j < 8; ++j)
        for (uint32_t lane = 0; lane < 16; ++lane)
            require(machine.read_gmem_u32(4 * (16 * j + lane)) == 17 + j,
                "younger overwrite changed a deferred stg.v8 source");
    require(machine.read_reg(0, 3, 8) == 99, "the younger overwrite was lost");
}

void test_gmem_v8_predication_is_rejected() {
    try {
        (void)compile("@p0 stg.v8 [r0], r8\nhalt\n");
        throw std::runtime_error("predicated stg.v8 was accepted without divergent execution");
    } catch (const nanosm::SimError& error) {
        require(std::string(error.what()).find("only bra.uni may be predicated") != std::string::npos,
            "predicated stg.v8 produced the wrong rejection");
    }
}

// Replaces "gmem v8 shared return": LSU RF writes are full-width beats through
// the reservations, at most one per cycle machine-wide, never single lanes.
void test_lsu_rf_writes_are_full_width_beats() {
    auto config = canonical_profile();
    config.gmem_latency_dram_ticks = 0;
    config.gmem_bus_bytes_per_dram_tick = config.gmem_sector_bytes;
    config.dram_clock_hz = config.core_clock_hz * 64;
    nanosm::Launch launch;
    launch.active_warps = 2;
    auto& image = image_bytes(launch.gmem_image, 1024);
    fill_pattern(image, 0, 256, 0);
    nanosm::Machine machine(config, compile(R"(
        sread r0, WARP_ID
        shli r0, r0, 9
        ldg.v8 r8, [r0]
        halt
    )", config), std::move(launch));
    std::vector<uint32_t> previous(2 * 8 * 16, 0);
    uint32_t writes = 0;
    while (!machine.done()) {
        tick_capped(machine);
        uint32_t registers_changed = 0;
        for (uint32_t warp = 0; warp < 2; ++warp) {
            for (uint32_t j = 0; j < 8; ++j) {
                uint32_t lanes_changed = 0;
                for (uint32_t lane = 0; lane < 16; ++lane) {
                    auto& old = previous[(warp * 8 + j) * 16 + lane];
                    const auto value = machine.read_reg(warp, lane, 8 + j);
                    if (value != old) ++lanes_changed;
                    old = value;
                }
                require(lanes_changed == 0 || lanes_changed == 16, "an LSU RF write touched a subset of lanes");
                if (lanes_changed == 16) ++registers_changed;
            }
        }
        require(registers_changed <= 1, "the LSU wrote more than one RF beat in a cycle");
        writes += registers_changed;
    }
    require(writes == 16, "two ldg.v8 commands did not make sixteen RF write beats");
    for (uint32_t warp = 0; warp < 2; ++warp)
        for (uint32_t j = 0; j < 8; ++j)
            for (uint32_t lane = 0; lane < 16; ++lane)
                require(machine.read_reg(warp, lane, 8 + j) == pattern_word(warp * 128 + 16 * j + lane),
                    "overlapping ldg.v8 commands lost or swapped a word");
}

// (b) Replaces "gmem v8 invalid footprints": the v8/uni/scalar contracts.
void test_lsu_contract_panics() {
    {
        auto profile = canonical_profile();
        profile.timing.alu = 4;
        const auto scheduled = nanosm::compile_qtx("mov r0, 0\nldg.v8 r8, [r0]\nhalt\n", profile);
        require(scheduled.instructions.front().control.stall != 0,
            "qtxc did not schedule the ldg.v8 address RAW dependency");
    }
    const auto run = [](const std::string& text, uint64_t gmem_bytes, const std::string& needle,
                        const std::string& what) {
        auto config = canonical_profile();
        config.gmem_bytes = gmem_bytes;
        nanosm::Launch launch;
        launch.active_warps = 1;
        nanosm::Machine machine(config, compile(text, config), std::move(launch));
        expect_sim_panic(machine, needle, what);
    };
    // One lane's base differs (lane 15 = 128, the others 64): both are 64 B
    // aligned, so only the uniformity rule can fire.
    const std::string one_lane_differs = R"(
        sread r1, LANE_ID
        addi r1, r1, 1
        shri r1, r1, 4
        shli r1, r1, 6
        addi r0, r1, 64
    )";
    run(one_lane_differs + "ldg.v8 r8, [r0]\nhalt\n", 4096,
        "ldg.v8 base address is not warp-uniform", "non-uniform ldg.v8 base");
    run(one_lane_differs + "stg.v8 [r0], r8\nhalt\n", 4096,
        "stg.v8 base address is not warp-uniform", "non-uniform stg.v8 base");
    run(one_lane_differs + "ldg.uni r8, [r0]\nhalt\n", 4096,
        "ldg.uni base address is not warp-uniform", "non-uniform ldg.uni base");
    run("mov r0, 32\nldg.v8 r8, [r0]\nhalt\n", 4096,
        "ldg.v8 address 32 is not 64-byte aligned", "rA = 32 mod 64 ldg.v8");
    run("mov r0, 64\nstg.v8 [r0 + 32], r8\nhalt\n", 4096,
        "stg.v8 address 96 is not 64-byte aligned", "rA + imm = 32 mod 64 stg.v8");
    run("sread r0, LANE_ID\nshli r0, r0, 2\nldg r1, [r0 + 2]\nhalt\n", 4096,
        "unaligned 32-bit gmem access", "2-byte aligned scalar ldg");
    run("mov r0, 6\nldg.uni r1, [r0]\nhalt\n", 4096,
        "unaligned 32-bit gmem access", "2-byte aligned ldg.uni");
    run("mov r0, 576\nldg.v8 r8, [r0]\nhalt\n", 1024, "out of bounds", "out-of-bounds ldg.v8 block");
}

// Replaces "gmem command capacity" (a footprint larger than the controller
// queue used to PANIC): commands now stream through any queue >= 1 entry.
void test_lsu_commands_stream_through_small_queue() {
    auto config = canonical_profile();
    config.tma_queue_commands = 0;
    config.tma_sector_credits = 0;
    config.gmem_queue_sectors = 2;
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& gmem = image_bytes(launch.gmem_image, 2048);
    fill_pattern(gmem, 0, 128, 0);
    nanosm::Machine machine(config, compile(R"(
        mov r0, 0
        ldg.v8 r8, [r0]
        stg.v8 [r0 + 1024], r8
        sread r1, LANE_ID
        shli r1, r1, 2
        ldg r16, [r1]
        halt
    )", config), std::move(launch));
    machine.run();
    for (uint32_t word = 0; word < 128; ++word)
        require(machine.read_gmem_u32(1024 + 4 * word) == pattern_word(word), "streamed v8 copy mismatch");
    for (uint32_t lane = 0; lane < 16; ++lane)
        require(machine.read_reg(0, lane, 16) == pattern_word(lane), "streamed scalar ldg mismatch");
    const auto& summary = machine.trace().summary();
    require(summary.lsu_request_wait_cycles != 0, "a two-entry queue never lowered request-ready");
    require(summary.gmem_max_queue_depth <= 2, "the controller queue exceeded its capacity");
}

// Replaces "gmem queue pressure": the LSU command slot is the structural
// wait point at issue.
void test_lsu_command_slot_structurally_stalls() {
    auto config = canonical_profile();
    config.lsu_active_commands = 1;
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
    const auto& summary = machine.trace().summary();
    require(summary.lsu_no_slot_cycles != 0 && summary.lsu_no_staging_cycles == 0,
        "one command slot never made the second ldg ineligible");
    require(summary.warp_cycles[static_cast<std::size_t>(nanosm::WarpState::structural)] >=
            summary.lsu_no_slot_cycles,
        "LSU ineligibility was not a structural warp state");
}

// (c) Gathers: one request per lane, no merging, any lane-address relation;
// ldg.uni is the declared broadcast (one request).
void test_lsu_gather_requests() {
    const auto gather = [](const std::string& address_code, uint32_t expected_sectors,
                           const std::function<uint32_t(uint32_t)>& word_of_lane,
                           nanosm::ControllerTestOrder order = nanosm::ControllerTestOrder::fifo) {
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& gmem = image_bytes(launch.gmem_image, 16384);
        fill_pattern(gmem, 0, 4096, 0);
        nanosm::Machine machine(canonical_profile(),
            compile("sread r1, LANE_ID\n" + address_code + "halt\n"), std::move(launch),
            {.timeline = true});
        machine.set_controller_test_order(order);
        machine.run();
        for (uint32_t lane = 0; lane < 16; ++lane)
            require(machine.read_reg(0, lane, 10) == pattern_word(word_of_lane(lane)),
                "gathered word mismatch in lane " + std::to_string(lane));
        require(machine.trace().summary().lsu_sectors == expected_sectors,
            "gather issued " + std::to_string(machine.trace().summary().lsu_sectors) +
                " sector requests, expected " + std::to_string(expected_sectors));
        std::vector<uint64_t> served;
        for (const auto& span : trace_spans(machine, nanosm::TraceCategory::gmem, "read sector")) {
            const auto at = span.detail.find("address=");
            served.push_back(std::stoull(span.detail.substr(at + 8)));
        }
        return served;
    };
    // Sixteen distinct sectors: lane*96 + 4*(lane & 7).
    const std::string distinct = R"(
        shli r2, r1, 5
        shli r3, r1, 6
        add r4, r2, r3
        andi r5, r1, 7
        shli r5, r5, 2
        add r6, r4, r5
        ldg r10, [r6]
    )";
    const auto distinct_word = [](uint32_t lane) { return (lane * 96 + 4 * (lane & 7)) / 4; };
    const auto in_order = gather(distinct, 16, distinct_word);
    require(std::is_sorted(in_order.begin(), in_order.end()),
        "the placeholder FIFO controller reordered gather returns");
    // Same gather with the legal free-ID reorder: returns come back out of
    // request order and every lane still gets its own word.
    const auto reordered = gather(distinct, 16, distinct_word, nanosm::ControllerTestOrder::reverse_free_ids);
    require(!std::is_sorted(reordered.begin(), reordered.end()),
        "the reverse_free_ids controller did not reorder gather returns");
    // Two lanes per sector (lanes 2k, 2k+1): still sixteen requests.
    gather(R"(
        shli r7, r1, 2
        shri r8, r1, 1
        shli r8, r8, 8
        add r9, r7, r8
        ldg r10, [r9 + 4096]
    )", 16, [](uint32_t lane) { return (4096 + 4 * lane + 256 * (lane >> 1)) / 4; });
    // All sixteen lanes in one sector: sixteen requests.
    gather(R"(
        andi r12, r1, 7
        shli r12, r12, 2
        ldg r10, [r12 + 8192]
    )", 16, [](uint32_t lane) { return (8192 + 4 * (lane & 7)) / 4; });
    // The same word for every lane, declared: one request, fanned out.
    gather("mov r0, 0\nldg.uni r10, [r0 + 8200]\n", 1, [](uint32_t) { return 8200 / 4; });
}

// (d) Scatters: sixteen masked sector writes; byte enables keep every
// unselected byte, including other lanes' words in a shared sector.
void test_lsu_scatter_byte_enables() {
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& gmem = image_bytes(launch.gmem_image, 2048);
    std::fill(gmem.begin(), gmem.end(), 0xA5);
    nanosm::Machine machine(canonical_profile(), compile(R"(
        sread r1, LANE_ID
        shli r6, r1, 6
        add r5, r1, r1
        add r5, r5, r1
        andi r5, r5, 7
        shli r5, r5, 2
        add r2, r6, r5
        addi r3, r1, 4096
        stg [r2], r3
        shli r7, r1, 2
        addi r9, r1, 8192
        stg [r7 + 1024], r9
        halt
    )"), std::move(launch));
    machine.run();
    std::vector<bool> written(2048, false);
    for (uint32_t lane = 0; lane < 16; ++lane) {
        const uint32_t own = lane * 64 + 4 * ((lane * 3) & 7);
        require(machine.read_gmem_u32(own) == 4096 + lane, "scatter lane word mismatch");
        require(machine.read_gmem_u32(1024 + 4 * lane) == 8192 + lane, "shared-sector scatter word mismatch");
        for (uint32_t byte = 0; byte < 4; ++byte) {
            written[own + byte] = true;
            written[1024 + 4 * lane + byte] = true;
        }
    }
    for (uint32_t byte = 0; byte < 2048; ++byte)
        if (!written[byte]) require(machine.gmem()[byte] == 0xA5, "a masked write damaged byte " + std::to_string(byte));
    const auto& summary = machine.trace().summary();
    require(summary.lsu_scalar_stores == 2 && summary.lsu_sectors == 32,
        "each scatter is not sixteen masked sector writes");
}

// (e) The staging pool is owned at admission. With one assembly register the
// second warp's ldg.v8 is ineligible until the first command's eighth RF
// write commits; with two it is admitted at once.
void test_lsu_staging_pool_admission() {
    const auto run = [](uint32_t staging) {
        auto config = canonical_profile();
        config.lsu_staging_vectors = staging;
        nanosm::Launch launch;
        launch.active_warps = 2;
        auto& gmem = image_bytes(launch.gmem_image, 1024);
        fill_pattern(gmem, 0, 256, 0);
        nanosm::Machine machine(config, compile(R"(
            sread r0, WARP_ID
            shli r0, r0, 9
            ldg.v8 r8, [r0]
            halt
        )", config), std::move(launch), {.timeline = true});
        machine.run();
        for (uint32_t warp = 0; warp < 2; ++warp)
            for (uint32_t j = 0; j < 8; ++j)
                for (uint32_t lane = 0; lane < 16; ++lane)
                    require(machine.read_reg(warp, lane, 8 + j) == pattern_word(warp * 128 + 16 * j + lane),
                        "pool admission corrupted a vector load");
        const auto spans = trace_spans(machine, nanosm::TraceCategory::gmem, "ldg.v8");
        require(spans.size() == 2, "expected two ldg.v8 command events");
        return std::make_tuple(spans[0], spans[1], machine.trace().summary().lsu_no_staging_cycles,
            machine.trace().summary().lsu_no_slot_cycles);
    };
    const auto [first, second, no_staging, no_slot] = run(1);
    require(second.start == first.end,
        "with one assembly register the second ldg.v8 was admitted at cycle " + std::to_string(second.start) +
            ", not at the first command's eighth write (" + std::to_string(first.end) + ")");
    // Warp 1 is tried every cycle from its ldg.v8 until admission.
    require(no_staging + 4 >= second.start - first.start,
        "no_staging did not count the ineligible cycles");
    require(no_staging != 0 && no_slot == 0, "the wait was not attributed to the staging pool");
    const auto [first2, second2, no_staging2, no_slot2] = run(2);
    require(second2.start <= first2.start + 2 && no_staging2 == 0 && no_slot2 == 0,
        "with two assembly registers the second ldg.v8 was not admitted at once");
}

// (f) The backpressure chain: RF write port blocked -> assembly register full
// -> (B2.1: no vector sector register) ingress FIFO head blocked -> FIFO full
// -> return-ready low ->
// the controller holds beats -> queue full -> request-ready low -> both
// command slots held -> the next ldg is ineligible. Nothing is lost.
void test_lsu_backpressure_chain() {
    const auto run = [](bool storm) {
        auto config = canonical_profile();
        config.tma_queue_commands = 0;
        config.tma_sector_credits = 0;
        config.gmem_queue_sectors = 8;
        config.lsu_ingress_sectors = 2;
        nanosm::Launch launch;
        launch.active_warps = 2;
        auto& gmem = image_bytes(launch.gmem_image, 4096);
        fill_pattern(gmem, 0, 1024, 0);
        // Warp 1 writes bank 0 (r3) every cycle: arithmetic books its RF write
        // at issue, so the LSU's write of r0 (bank 0) never finds a free port.
        std::string text = R"(
            sread r29, WARP_ID
            setp.ne p0, r29, 0
            @p0 bra.uni other
            mov r30, 0
            mov r31, 512
            mov r28, 2048
            ldg.v8 r0, [r30]
            ldg.v8 r8, [r31]
            ldg r16, [r28]
            halt
        other:
        )";
        if (storm) for (int i = 0; i < 500; ++i) text += "mov r3, 1\n";
        text += "halt\n";
        nanosm::Machine machine(config, compile(text, config), std::move(launch), {.timeline = true});
        bool linked = false;
        uint64_t ingress = 0, request = 0, slot = 0;
        while (!machine.done()) {
            tick_capped(machine);
            const auto& s = machine.trace().summary();
            if (s.lsu_ingress_full_cycles > ingress && s.lsu_request_wait_cycles > request &&
                s.lsu_no_slot_cycles > slot) linked = true;
            ingress = s.lsu_ingress_full_cycles;
            request = s.lsu_request_wait_cycles;
            slot = s.lsu_no_slot_cycles;
        }
        for (uint32_t j = 0; j < 16; ++j)
            for (uint32_t lane = 0; lane < 16; ++lane)
                require(machine.read_reg(0, lane, j) == pattern_word(16 * j + lane),
                    "a byte was lost or corrupted under backpressure");
        for (uint32_t lane = 0; lane < 16; ++lane)
            require(machine.read_reg(0, lane, 16) == pattern_word(512), "the stalled scalar ldg read the wrong word");
        return std::make_tuple(linked, machine.trace().summary());
    };
    const auto [linked, summary] = run(true);
    require(summary.lsu_rf_write_wait_cycles >= 300, "the RF write port was never the blocked link");
    require(summary.lsu_ingress_full_cycles != 0, "return-ready never dropped (ingress FIFO never full)");
    require(summary.lsu_request_wait_cycles != 0, "request-ready never dropped");
    require(summary.lsu_no_slot_cycles != 0, "the next ldg was never ineligible");
    require(linked, "no cycle showed ingress full, request-ready low and the slot wait together");
    const auto [control_linked, control] = run(false);
    require(!control_linked && control.lsu_ingress_full_cycles == 0 && control.lsu_rf_write_wait_cycles == 0,
        "the control run (free RF write port) still filled the ingress FIFO");
}

void write_rank1_load(std::vector<uint8_t>& shmem, uint32_t descriptor_address, uint64_t gmem_base,
                      uint32_t elements, uint32_t tile, uint32_t mbar_address) {
    nanosm::TmaDescriptor descriptor;
    descriptor.gmem_base = gmem_base;
    descriptor.dimensions = {elements, 1, 1};
    descriptor.box = {elements, 1, 1};
    descriptor.shmem_base = tile;
    descriptor.completion_mbar = mbar_address;
    write_tma_descriptor(shmem, descriptor_address, descriptor);
    nanosm::MbarState mbar;
    mbar.valid = true;
    mbar.expected = 1;
    mbar.tx_bytes = elements * 4;
    write_u64(shmem, mbar_address, nanosm::encode_mbar_state(mbar));
}

// (g) TMA priority on port A: a returned TMA sector waits only for the SIMT
// access already in port A, never for one granted after it became ready.
void test_tma_port_a_priority_over_sts() {
    nanosm::Launch launch;
    launch.active_warps = 2;
    auto& shmem = image_bytes(launch.shmem_image, 8192);
    write_rank1_load(shmem, 0, 0, 64, 1024, 128);
    auto& gmem = image_bytes(launch.gmem_image, 256);
    fill_pattern(gmem, 0, 64, 0);
    std::string text = R"(
        sread r29, WARP_ID
        setp.ne p0, r29, 0
        @p0 bra.uni hammer
        mov r0, 0
        mov r1, 0
        mov r2, 0
        mov r3, 128
        tma.async [0], r0
    wait:
        mbar.try_wait p1, [r3], 0
        @!p1 bra.uni wait
        halt
    hammer:
        sread r4, LANE_ID
        shli r4, r4, 2
    )";
    for (int i = 0; i < 400; ++i) text += "sts [r4 + 4096], r4\n";
    text += "halt\n";
    nanosm::Machine machine(canonical_profile(), compile(text), std::move(launch), {.timeline = true});
    machine.enable_edge_probe();
    machine.run();
    for (uint32_t i = 0; i < 64; ++i)
        require(machine.read_shmem_u32(1024 + 4 * i) == pattern_word(i), "TMA load data mismatch under sts pressure");
    const auto placements = trace_spans(machine, nanosm::TraceCategory::tma, "tma load write");
    // B2.1: the sector is complete in the TMA sector register at the cycle of
    // its last plucked byte (bus arrival + the ingress FIFO latency), no
    // longer at its last bus beat (B2's fall-through FIFO). Taken from the
    // edge probe; the assertion below is unchanged.
    std::vector<uint64_t> completed;
    std::map<uint64_t, uint32_t> filled;
    for (const auto& event : machine.edge_events()) {
        if (event.edge != nanosm::LsuEdge::tma_reg_write) continue;
        if ((filled[event.id] += event.bytes) == 32) completed.push_back(event.cycle);
    }
    const auto stores = trace_spans(machine, nanosm::TraceCategory::shmem, "sts");
    require(placements.size() == 8 && completed.size() == 8, "expected eight TMA sectors");
    uint32_t contended = 0;
    for (std::size_t k = 0; k < 8; ++k) {
        // The sector is in the TMA sector register at the end of its
        // completion cycle; from that cycle's issue on, SIMT may not be granted.
        const auto ready = completed[k];
        for (const auto& store : stores) {
            require(!(store.start >= ready && store.start < placements[k].start),
                "sts granted at cycle " + std::to_string(store.start) + " after TMA sector " +
                    std::to_string(k) + " became ready (" + std::to_string(ready) +
                    ") but before its placement (" + std::to_string(placements[k].start) + ")");
            if (store.start < ready && store.end > ready) ++contended;
        }
    }
    require(contended != 0, "the sts hammer never overlapped a TMA return; the test proves nothing");
    require(machine.trace().summary().shmem_tma_priority_waits != 0, "SIMT waits behind TMA were not counted");
}

// (h) Read streams: TMA (ID 0) and vector (ID 1) sectors interleave in the
// controller's global order at sector granularity; each lands correctly.
void test_lsu_tma_and_vector_streams_interleave() {
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.shmem_image, 4096);
    write_rank1_load(shmem, 0, 0, 128, 1024, 128);
    auto& gmem = image_bytes(launch.gmem_image, 1024);
    fill_pattern(gmem, 0, 256, 0);
    nanosm::Machine machine(canonical_profile(), compile(R"(
        mov r0, 0
        mov r1, 0
        mov r2, 0
        mov r3, 128
        mov r30, 512
        tma.async [0], r0
        ldg.v8 r8, [r30]
    wait:
        mbar.try_wait p1, [r3], 0
        @!p1 bra.uni wait
        halt
    )"), std::move(launch), {.timeline = true});
    machine.run();
    for (uint32_t i = 0; i < 128; ++i)
        require(machine.read_shmem_u32(1024 + 4 * i) == pattern_word(i), "TMA stream data mismatch");
    for (uint32_t j = 0; j < 8; ++j)
        for (uint32_t lane = 0; lane < 16; ++lane)
            require(machine.read_reg(0, lane, 8 + j) == pattern_word(128 + 16 * j + lane),
                "vector stream data mismatch with TMA returns interleaved");
    uint32_t switches = 0;
    std::string previous;
    for (const auto& span : trace_spans(machine, nanosm::TraceCategory::gmem, "read sector")) {
        const auto owner = span.detail.substr(span.detail.find("owner="));
        if (!previous.empty() && owner != previous) ++switches;
        previous = owner;
    }
    require(switches >= 8, "TMA and vector returns did not interleave at sector granularity (" +
        std::to_string(switches) + " owner switches)");
}

// The boundary tripwire: a controller that returns an ordered ID out of
// request order is a contract violation the SM refuses, by PANIC.
void test_lsu_controller_contract_tripwire() {
    nanosm::Launch launch;
    launch.active_warps = 1;
    image_bytes(launch.gmem_image, 512);
    nanosm::Machine machine(canonical_profile(), compile("mov r0, 0\nldg.v8 r8, [r0]\nhalt\n"), std::move(launch));
    machine.set_controller_test_order(nanosm::ControllerTestOrder::illegal_ordered_reorder);
    expect_sim_panic(machine, "contract violation: read stream 1 (vector) returned request",
        "an out-of-order vector stream");
}

void test_lsu_profile_and_encoding() {
    const auto profile = canonical_profile();
    require(profile.lsu_active_commands == 2 && profile.lsu_staging_vectors == 2 &&
            profile.lsu_read_streams == 2 && profile.lsu_pluck_words_per_cycle == 2 &&
            profile.lsu_ingress_sectors == 8 && profile.lsu_egress_sectors == 8 &&
            profile.lsu_sectors_in_flight == 16 && profile.lsu_ingress_latency_cycles == 1 &&
            profile.lsu_egress_latency_cycles == 1,
        "canonical lsu profile values changed");
    auto other = profile;
    other.lsu_staging_vectors = 1;
    require(nanosm::device_profile_fingerprint(profile) != nanosm::device_profile_fingerprint(other),
        "lsu sizing was omitted from the profile fingerprint");
    for (const auto field : {&nanosm::DeviceProfile::lsu_ingress_latency_cycles,
                             &nanosm::DeviceProfile::lsu_egress_latency_cycles}) {
        auto latency = profile;
        latency.*field = 3;
        require(nanosm::device_profile_fingerprint(profile) != nanosm::device_profile_fingerprint(latency),
            "an lsu FIFO latency was omitted from the profile fingerprint");
    }
    std::ifstream input("profiles/nanosm-v0.json");
    const auto document = nlohmann::json::parse(input);
    const auto rejects = [&](const std::function<void(nlohmann::json&)>& edit, const std::string& needle) {
        auto changed = document;
        edit(changed);
        try {
            (void)nanosm::parse_device_profile(changed.dump());
        } catch (const nanosm::ProfileError& error) {
            require(std::string(error.what()).find(needle) != std::string::npos,
                std::string("wrong lsu profile rejection: ") + error.what());
            return;
        }
        throw std::runtime_error("invalid lsu profile accepted: " + needle);
    };
    rejects([](auto& j) { j.erase("lsu"); }, "missing 'lsu'");
    // B2.1: read_streams generalized (stream 0 TMA, 1..n-1 vector), >= 2.
    rejects([](auto& j) { j["lsu"]["read_streams"] = 1; }, "read_streams must be in 2..64");
    rejects([](auto& j) { j["lsu"]["ingress_sectors"] = 1; }, "at least two sectors");
    rejects([](auto& j) { j["lsu"]["egress_sectors"] = 1; }, "at least two sectors");
    rejects([](auto& j) { j["lsu"]["staging_vectors"] = 0; }, "must be nonzero");
    // B2.1: a BRAM FIFO is never fall-through; latency 0 is rejected.
    rejects([](auto& j) { j["lsu"]["ingress_latency_cycles"] = 0; }, "latency must be in 1..64");
    rejects([](auto& j) { j["lsu"]["egress_latency_cycles"] = 0; }, "latency must be in 1..64");
    rejects([](auto& j) { j["lsu"].erase("egress_latency_cycles"); }, "egress_latency_cycles");
    // active_commands > staging_vectors is a legal profile: the pool makes
    // the extra slot ineligible (no_staging), the sanitizer asserts the bound.
    auto loose = document;
    loose["lsu"]["staging_vectors"] = 1;
    (void)nanosm::parse_device_profile(loose.dump());
    auto streams = document;
    streams["lsu"]["read_streams"] = 3;
    require(nanosm::parse_device_profile(streams.dump()).lsu_read_streams == 3, "read_streams = 3 rejected");

    const auto kernel = compile("mov r3, 0\nldg.uni r5, [r3 + 8]\nhalt\n");
    const auto decoded = nanosm::decode(kernel.words[1]);
    require(decoded.opcode == nanosm::Opcode::ldg_uni && decoded.dst == 5 && decoded.src0 == 3 &&
            decoded.imm == 8 && decoded.control.set_depbar < 6,
        "ldg.uni did not survive qtxc + qtxas");
    require(nanosm::format_instruction(decoded, false) == "ldg.uni r5, [r3 + 8]", "ldg.uni formatting changed");
    require(nanosm::decode(nanosm::encode(decoded)).opcode == nanosm::Opcode::ldg_uni, "ldg.uni encoding round trip");
}

// ---------------------------------------------------------------------------
// B2.1 (rfc09 section 3a): the handoff table edge by edge. Every edge has a
// declared width per cycle; every named stage is a register stage (BRAM
// FIFOs: + their declared latency); no datum crosses two stages in a cycle;
// one write contract; generalized read streams. Observed through the
// machine's edge probe (a test hook, never consulted by the machine).
// trash/lsu-b21-2026-10-04.md.
// ---------------------------------------------------------------------------

using nanosm::LsuEdge;

void write_rank1_tma(std::vector<uint8_t>& shmem, uint32_t descriptor_address, bool store,
                     uint64_t gmem_base, uint32_t elements, uint32_t tile, uint32_t mbar_address) {
    nanosm::TmaDescriptor descriptor;
    descriptor.direction = store ? nanosm::TmaDirection::store : nanosm::TmaDirection::load;
    descriptor.gmem_base = gmem_base;
    descriptor.dimensions = {elements, 1, 1};
    descriptor.box = {elements, 1, 1};
    descriptor.shmem_base = tile;
    descriptor.completion_mbar = mbar_address;
    write_tma_descriptor(shmem, descriptor_address, descriptor);
    nanosm::MbarState mbar;
    mbar.valid = true;
    mbar.expected = 1;
    mbar.tx_bytes = elements * 4;
    write_u64(shmem, mbar_address, nanosm::encode_mbar_state(mbar));
}

std::string tma_and_wait(uint32_t descriptor_address, uint32_t mbar_address) {
    return "mov r0, 0\nmov r1, 0\nmov r2, 0\nmov r3, " + std::to_string(mbar_address) +
        "\ntma.async [" + std::to_string(descriptor_address) + "], r0\n"
        "wait:\nmbar.try_wait p1, [r3], 0\n@!p1 bra.uni wait\nhalt\n";
}

// A bus far faster than any pluck rate (a whole sector per core cycle, no
// latency), so the declared SM-side widths are what binds.
nanosm::DeviceProfile fast_bus_profile(uint32_t pluck) {
    auto config = canonical_profile();
    config.dram_clock_hz = config.core_clock_hz;
    config.gmem_bus_bytes_per_dram_tick = 32;
    config.gmem_latency_dram_ticks = 0;
    config.lsu_pluck_words_per_cycle = pluck;
    return config;
}

struct EdgeLoad {
    uint32_t bytes = 0;
    uint32_t events = 0;
};

std::map<std::pair<uint64_t, LsuEdge>, EdgeLoad> edge_load(const nanosm::Machine& machine) {
    std::map<std::pair<uint64_t, LsuEdge>, EdgeLoad> load;
    for (const auto& event : machine.edge_events()) {
        auto& entry = load[{event.cycle, event.edge}];
        entry.bytes += event.bytes;
        ++entry.events;
    }
    return load;
}

uint32_t max_bytes_per_cycle(const nanosm::Machine& machine, LsuEdge edge) {
    uint32_t most = 0;
    for (const auto& [key, entry] : edge_load(machine))
        if (key.second == edge) most = std::max(most, entry.bytes);
    return most;
}

// Every edge of the 3a table, every cycle of the run: never more than its
// declared width.
void check_edge_widths(const nanosm::Machine& machine, const nanosm::DeviceProfile& config,
                       const std::string& what) {
    const uint32_t pluck = config.lsu_pluck_words_per_cycle * 4;
    std::map<uint64_t, uint32_t> beats_per_tick;
    std::map<uint64_t, uint64_t> slot_client;
    for (const auto& event : machine.edge_events()) {
        if (event.edge == LsuEdge::bus_read_beat || event.edge == LsuEdge::bus_write_beat) {
            require(event.bytes <= config.gmem_bus_bytes_per_dram_tick, what + ": a beat wider than the bus");
            require(++beats_per_tick[event.dram_tick] == 1,
                what + ": two bus beats in DRAM tick " + std::to_string(event.dram_tick));
        }
        if (event.edge == LsuEdge::egress_write) {
            const auto [it, fresh] = slot_client.emplace(event.id, event.aux);
            require(it->second == event.aux, what + ": an egress slot written by two clients");
        }
    }
    const auto load = edge_load(machine);
    for (const auto& [key, entry] : load) {
        const auto [cycle, edge] = key;
        const auto at = what + ", cycle " + std::to_string(cycle) + ": ";
        switch (edge) {
        case LsuEdge::ingress_read:
            require(entry.bytes <= pluck, at + "ingress FIFO read " + std::to_string(entry.bytes) +
                " B, over lsu.pluck_words_per_cycle");
            break;
        case LsuEdge::tma_reg_write:
            require(entry.bytes <= pluck, at + "TMA sector register filled with " +
                std::to_string(entry.bytes) + " B, over the pluck width");
            break;
        case LsuEdge::assembly_write:
            require(entry.bytes <= pluck, at + "assembly register filled with " +
                std::to_string(entry.bytes) + " B, over the pluck width");
            break;
        case LsuEdge::egress_write:
            require(entry.bytes <= pluck && entry.events == 1,
                at + "egress write port over its width or shared by two clients");
            break;
        case LsuEdge::rf_write_beat:
        case LsuEdge::rf_read_beat:
            require(entry.events <= 1, at + "more than one full-width LSU RF beat");
            break;
        case LsuEdge::tma_place_round:
        case LsuEdge::tma_store_round: {
            const auto other = edge == LsuEdge::tma_place_round ? LsuEdge::tma_store_round : LsuEdge::tma_place_round;
            require(entry.events <= 1 && load.count({cycle, other}) == 0,
                at + "more than one TMA port-A round");
            break;
        }
        default:
            break;
        }
    }
}

// E1: per-edge widths. The vector path at pluck 1 and 2, gathers, scatters
// and TMA (load and store) at pluck 1, all under a bus 8-32x faster than the
// pluck, so that the declared SM widths are the binding ones (and the test
// requires they are reached, so it is not vacuous).
void test_lsu_edge_widths() {
    for (const uint32_t pluck : {1u, 2u}) {
        const auto config = fast_bus_profile(pluck);
        nanosm::Launch launch;
        launch.active_warps = 2;
        auto& gmem = image_bytes(launch.gmem_image, 4096);
        fill_pattern(gmem, 0, 256, 0);
        nanosm::Machine machine(config, compile(R"(
            sread r0, WARP_ID
            shli r0, r0, 9
            ldg.v8 r8, [r0]
            stg.v8 [r0 + 2048], r8
            halt
        )", config), std::move(launch));
        machine.enable_edge_probe();
        machine.run();
        for (uint32_t word = 0; word < 256; ++word)
            require(machine.read_gmem_u32(2048 + 4 * word) == pattern_word(word), "vector copy mismatch");
        const auto what = "vector path, pluck " + std::to_string(pluck);
        check_edge_widths(machine, config, what);
        require(max_bytes_per_cycle(machine, LsuEdge::assembly_write) == pluck * 4 &&
                max_bytes_per_cycle(machine, LsuEdge::ingress_read) == pluck * 4 &&
                max_bytes_per_cycle(machine, LsuEdge::egress_write) == pluck * 4,
            what + ": the pluck width was never reached; the test proves nothing");
        require(machine.cycle() >= 1024 / (4 * pluck), what + ": faster than the ingress read port allows");
    }
    { // gathers and scatters at pluck 1
        const auto config = fast_bus_profile(1);
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& gmem = image_bytes(launch.gmem_image, 8192);
        fill_pattern(gmem, 0, 1024, 0);
        nanosm::Machine machine(config, compile(R"(
            sread r1, LANE_ID
            shli r2, r1, 7
            ldg r10, [r2 + 4]
            stg [r2 + 4096], r10
            halt
        )", config), std::move(launch));
        machine.enable_edge_probe();
        machine.run();
        for (uint32_t lane = 0; lane < 16; ++lane)
            require(machine.read_gmem_u32(4096 + 128 * lane) == pattern_word(32 * lane + 1), "gather/scatter mismatch");
        check_edge_widths(machine, config, "gather/scatter, pluck 1");
        // A whole 32 B sector is read through a 4 B port per gathered word.
        require(max_bytes_per_cycle(machine, LsuEdge::ingress_read) == 4, "gather never reached the pluck width");
    }
    for (const bool store : {false, true}) { // TMA at pluck 1: 2 KiB
        const auto config = fast_bus_profile(1);
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.shmem_image, 4096);
        write_rank1_tma(shmem, 0, store, 0, 512, 1024, 128);
        auto& gmem = image_bytes(launch.gmem_image, 4096);
        if (store) fill_pattern(shmem, 1024, 512, 0); else fill_pattern(gmem, 0, 512, 0);
        nanosm::Machine machine(config, compile(tma_and_wait(0, 128), config), std::move(launch));
        machine.enable_edge_probe();
        machine.run();
        for (uint32_t i = 0; i < 512; ++i)
            require((store ? machine.read_gmem_u32(4 * i) : machine.read_shmem_u32(1024 + 4 * i)) == pattern_word(i),
                "TMA data mismatch at pluck 1");
        const std::string what = store ? "TMA store, pluck 1" : "TMA load, pluck 1";
        check_edge_widths(machine, config, what);
        require(max_bytes_per_cycle(machine, store ? LsuEdge::egress_write : LsuEdge::tma_reg_write) == 4,
            what + ": never reached the pluck width");
        require(machine.cycle() >= 2048 / 4, what + ": faster than one word per cycle");
    }
}

// Codex's B2 probe (chan 2026-10-04, finding 3): a 16 KiB rank-1 linear TMA
// store, DRAM = core clock, 32 B per tick, latency 0, egress_sectors 2. B2
// took 520 cycles at pluck 1, 2 and 8 (the store bypassed egress). Now every
// TMA store sector crosses the shared egress write port, so the run is
// pluck-bound: 16384 / (4 x pluck) cycles plus a short tail.
void test_tma_store_egress_is_pluck_bound() {
    for (const uint32_t pluck : {1u, 2u, 8u}) {
        auto config = fast_bus_profile(pluck);
        config.lsu_egress_sectors = 2;
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.shmem_image, 16384 + 256);
        fill_pattern(shmem, 0, 4096, 0);
        write_rank1_tma(shmem, 16384, true, 0, 4096, 0, 16384 + 128);
        image_bytes(launch.gmem_image, 16384);
        nanosm::Machine machine(config, compile(tma_and_wait(16384, 16384 + 128), config), std::move(launch));
        machine.run();
        for (uint32_t i = 0; i < 4096; ++i)
            require(machine.read_gmem_u32(4 * i) == pattern_word(i), "16 KiB TMA store data mismatch");
        const uint64_t bound = 16384 / (4 * pluck);
        require(machine.cycle() >= bound && machine.cycle() <= bound + 100,
            "16 KiB TMA store at pluck " + std::to_string(pluck) + " took " + std::to_string(machine.cycle()) +
                " cycles; the egress write port bounds it at " + std::to_string(bound));
    }
}

// Timelines of single data under an uncontended profile: DRAM = core clock,
// 8 B beats, gmem latency 4 ticks, pluck 2 (= the bus rate), FIFO latencies
// as given. Each entry is (stage-to-stage) cycle deltas; E2 asserts them,
// E5 compares two latency settings.
struct Timelines {
    std::vector<int64_t> vector_bus_to_assembly;   // per beat
    std::vector<int64_t> vector_assembly_to_rf;    // per chunk
    std::vector<uint64_t> vector_rf_writes;
    std::vector<int64_t> tma_bus_to_register;      // per beat
    int64_t tma_register_to_shmem = 0;
    uint64_t tma_place = 0;
    int64_t uni_bus_to_assembly = 0;
    int64_t uni_assembly_to_rf = 0;
    uint64_t uni_rf_write = 0;
    std::vector<int64_t> gather_bus_to_read;       // per beat
    int64_t gather_assembly_to_rf = 0;
    uint64_t gather_rf_write = 0;
    std::vector<int64_t> store_rf_to_egress;       // per chunk
    std::vector<int64_t> store_drain_to_next_rf;   // per chunk boundary
    std::vector<int64_t> store_slot_to_request;    // per slot
    std::vector<int64_t> store_request_to_bus;     // per slot
    std::vector<int64_t> store_ack_after_last_beat; // per slot, DRAM ticks
    uint64_t store_first_beat = 0;
    std::array<uint64_t, 5> total_cycles{};
};

Timelines lsu_timelines(uint32_t ingress_latency, uint32_t egress_latency) {
    auto config = canonical_profile();
    config.dram_clock_hz = config.core_clock_hz;
    config.gmem_bus_bytes_per_dram_tick = 8;
    config.gmem_latency_dram_ticks = 4;
    config.lsu_pluck_words_per_cycle = 2;
    config.lsu_ingress_latency_cycles = ingress_latency;
    config.lsu_egress_latency_cycles = egress_latency;
    Timelines t;
    const auto run = [&](const std::string& text, bool tma, std::size_t slot) {
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& gmem = image_bytes(launch.gmem_image, 8192);
        fill_pattern(gmem, 0, 2048, 0);
        if (tma) write_rank1_tma(image_bytes(launch.shmem_image, 2048), 0, false, 0, 8, 1024, 128);
        auto machine = std::make_unique<nanosm::Machine>(config, compile(text, config), std::move(launch));
        machine->enable_edge_probe();
        machine->run();
        t.total_cycles[slot] = machine->cycle();
        return machine;
    };
    const auto of = [](const nanosm::Machine& machine, LsuEdge edge) {
        std::vector<nanosm::LsuEdgeEvent> events;
        for (const auto& event : machine.edge_events()) if (event.edge == edge) events.push_back(event);
        return events;
    };
    const auto read_cycle = [&](const nanosm::Machine& machine, uint64_t request, uint64_t offset) {
        for (const auto& event : of(machine, LsuEdge::ingress_read))
            if (event.id == request && event.aux == offset) return event.cycle;
        throw std::runtime_error("a returned beat was never read from the ingress FIFO");
    };
    { // vector load: bus -> ingress FIFO -> assembly -> RF
        const auto machine = run("mov r0, 0\nldg.v8 r8, [r0]\nhalt\n", false, 0);
        std::map<uint64_t, uint64_t> last_word; // request -> cycle of its last assembly write
        for (const auto& event : of(*machine, LsuEdge::assembly_write)) last_word[event.aux] = event.cycle;
        for (const auto& beat : of(*machine, LsuEdge::bus_read_beat)) {
            const auto read = read_cycle(*machine, beat.id, beat.aux);
            bool assembled = false;
            for (const auto& event : of(*machine, LsuEdge::assembly_write))
                assembled |= event.aux == beat.id && event.cycle == read;
            require(assembled, "a vector beat read from the FIFO did not land in the assembly register that cycle");
            t.vector_bus_to_assembly.push_back(static_cast<int64_t>(read) - static_cast<int64_t>(beat.cycle));
        }
        std::vector<uint64_t> requests;
        for (const auto& [request, cycle] : last_word) requests.push_back(request);
        const auto writes = of(*machine, LsuEdge::rf_write_beat);
        require(writes.size() == 8 && requests.size() == 16, "vector load: expected 8 RF beats from 16 sectors");
        for (uint32_t j = 0; j < 8; ++j) {
            const auto chunk_done = std::max(last_word[requests[2 * j]], last_word[requests[2 * j + 1]]);
            t.vector_assembly_to_rf.push_back(static_cast<int64_t>(writes[j].cycle) - static_cast<int64_t>(chunk_done));
            t.vector_rf_writes.push_back(writes[j].cycle);
        }
    }
    { // TMA load, one sector: bus -> ingress FIFO -> TMA sector register -> shmem
        const auto machine = run(tma_and_wait(0, 128), true, 1);
        uint64_t last = 0;
        for (const auto& beat : of(*machine, LsuEdge::bus_read_beat)) {
            for (const auto& event : of(*machine, LsuEdge::tma_reg_write)) {
                if (event.id != beat.id || event.aux != beat.aux) continue;
                t.tma_bus_to_register.push_back(static_cast<int64_t>(event.cycle) - static_cast<int64_t>(beat.cycle));
                last = std::max(last, event.cycle);
            }
        }
        const auto rounds = of(*machine, LsuEdge::tma_place_round);
        require(rounds.size() == 1 && t.tma_bus_to_register.size() == 4, "TMA load: expected 4 beats, 1 round");
        t.tma_register_to_shmem = static_cast<int64_t>(rounds[0].cycle) - static_cast<int64_t>(last);
        t.tma_place = rounds[0].cycle;
        for (uint32_t i = 0; i < 8; ++i)
            require(machine->read_shmem_u32(1024 + 4 * i) == pattern_word(i), "TMA timeline data mismatch");
    }
    { // ldg.uni of word 5 (byte 20, in the beat at offset 16)
        const auto machine = run("mov r0, 0\nldg.uni r1, [r0 + 20]\nhalt\n", false, 2);
        const auto words = of(*machine, LsuEdge::assembly_write);
        const auto writes = of(*machine, LsuEdge::rf_write_beat);
        require(words.size() == 1 && writes.size() == 1, "ldg.uni: one word, one RF beat");
        for (const auto& beat : of(*machine, LsuEdge::bus_read_beat))
            if (beat.aux == 16) t.uni_bus_to_assembly = static_cast<int64_t>(words[0].cycle) - static_cast<int64_t>(beat.cycle);
        t.uni_assembly_to_rf = static_cast<int64_t>(writes[0].cycle) - static_cast<int64_t>(words[0].cycle);
        t.uni_rf_write = writes[0].cycle;
        require(machine->read_reg(0, 7, 1) == pattern_word(5), "ldg.uni timeline data mismatch");
    }
    { // gather: 16 lanes, 16 sectors
        const auto machine = run("sread r1, LANE_ID\nshli r2, r1, 7\nldg r10, [r2 + 8]\nhalt\n", false, 3);
        for (const auto& beat : of(*machine, LsuEdge::bus_read_beat))
            t.gather_bus_to_read.push_back(
                static_cast<int64_t>(read_cycle(*machine, beat.id, beat.aux)) - static_cast<int64_t>(beat.cycle));
        uint64_t last = 0;
        for (const auto& event : of(*machine, LsuEdge::assembly_write)) last = std::max(last, event.cycle);
        const auto writes = of(*machine, LsuEdge::rf_write_beat);
        require(writes.size() == 1, "gather: one RF beat");
        t.gather_assembly_to_rf = static_cast<int64_t>(writes[0].cycle) - static_cast<int64_t>(last);
        t.gather_rf_write = writes[0].cycle;
        for (uint32_t lane = 0; lane < 16; ++lane)
            require(machine->read_reg(0, lane, 10) == pattern_word(32 * lane + 2), "gather timeline data mismatch");
    }
    { // stg.v8: RF -> assembly -> egress FIFO -> bus -> acknowledgement
        std::string text = "mov r0, 4096\n";
        for (uint32_t j = 0; j < 8; ++j) text += "mov r" + std::to_string(8 + j) + ", " + std::to_string(100 + j) + "\n";
        text += "stg.v8 [r0], r8\nhalt\n";
        const auto machine = run(text, false, 4);
        const auto reads = of(*machine, LsuEdge::rf_read_beat);
        require(reads.size() == 8, "stg.v8: eight RF read beats");
        std::map<uint64_t, std::pair<uint64_t, uint64_t>> slot_span; // slot -> first, last egress write
        for (const auto& event : of(*machine, LsuEdge::egress_write)) {
            auto [it, fresh] = slot_span.emplace(event.id, std::make_pair(event.cycle, event.cycle));
            it->second.second = event.cycle;
        }
        std::vector<std::pair<uint64_t, uint64_t>> slots;
        for (const auto& [slot, span] : slot_span) slots.push_back(span);
        require(slots.size() == 16, "stg.v8: sixteen egress slots");
        for (uint32_t j = 0; j < 8; ++j) {
            t.store_rf_to_egress.push_back(static_cast<int64_t>(slots[2 * j].first) - static_cast<int64_t>(reads[j].cycle));
            if (j != 0)
                t.store_drain_to_next_rf.push_back(
                    static_cast<int64_t>(reads[j].cycle) - static_cast<int64_t>(slots[2 * j - 1].second));
        }
        std::map<uint64_t, uint64_t> request_of_slot, request_cycle, first_beat, last_beat_tick, ack_tick;
        for (const auto& event : of(*machine, LsuEdge::write_request)) {
            request_of_slot[event.aux] = event.id;
            request_cycle[event.id] = event.cycle;
        }
        for (const auto& event : of(*machine, LsuEdge::bus_write_beat)) {
            first_beat.emplace(event.id, event.cycle);
            last_beat_tick[event.id] = event.dram_tick;
        }
        for (const auto& event : of(*machine, LsuEdge::write_ack)) ack_tick[event.id] = event.dram_tick;
        uint32_t index = 0;
        for (const auto& [slot, span] : slot_span) {
            const auto request = request_of_slot.at(slot);
            t.store_slot_to_request.push_back(
                static_cast<int64_t>(request_cycle.at(request)) - static_cast<int64_t>(span.second));
            t.store_request_to_bus.push_back(
                static_cast<int64_t>(first_beat.at(request)) - static_cast<int64_t>(request_cycle.at(request)));
            t.store_ack_after_last_beat.push_back(
                static_cast<int64_t>(ack_tick.at(request)) - static_cast<int64_t>(last_beat_tick.at(request)));
            if (index++ == 0) t.store_first_beat = first_beat.at(request);
        }
        for (uint32_t word = 0; word < 128; ++word)
            require(machine->read_gmem_u32(4096 + 4 * word) == 100 + word / 16, "stg.v8 timeline data mismatch");
    }
    return t;
}

bool all_equal(const std::vector<int64_t>& values, int64_t expected) {
    return !values.empty() && std::all_of(values.begin(), values.end(), [&](int64_t v) { return v == expected; });
}

// E2: no fall-through. Each datum spends exactly one cycle per register
// stage plus the declared FIFO latency, uncontended.
void test_lsu_no_fall_through_timelines() {
    const auto t = lsu_timelines(1, 1);
    require(all_equal(t.vector_bus_to_assembly, 1),
        "vector: bus beat -> assembly register is not exactly the ingress FIFO latency (1)");
    require(all_equal(t.vector_assembly_to_rf, 1),
        "vector: the RF write is not the cycle after the 16th word lands");
    require(all_equal(t.tma_bus_to_register, 1), "TMA: bus beat -> TMA sector register is not 1 cycle");
    require(t.tma_register_to_shmem == 1, "TMA: sector register -> shmem round is not the next cycle");
    require(t.uni_bus_to_assembly == 1 && t.uni_assembly_to_rf == 1, "ldg.uni: bus -> assembly -> RF is not 1 + 1");
    require(all_equal(t.gather_bus_to_read, 1) && t.gather_assembly_to_rf == 1,
        "gather: bus -> FIFO read -> RF is not 1 + 1");
    require(all_equal(t.store_rf_to_egress, 1), "stg.v8: RF read beat -> first egress word is not 1 cycle");
    // The assembly register is refilled in the cycle its last word drained
    // (its consumer runs first in the tick): a handoff, not a read-through.
    require(all_equal(t.store_drain_to_next_rf, 0), "stg.v8: drained assembly register not refilled at once");
    require(all_equal(t.store_slot_to_request, 1), "stg.v8: complete slot -> write request is not the egress latency (1)");
    require(all_equal(t.store_request_to_bus, 0), "stg.v8: an accepted write did not start on the idle bus");
    require(all_equal(t.store_ack_after_last_beat, 4), "stg.v8: the ack is not latency_dram_ticks after the last beat");
}

// E5: the two FIFO latencies are real stages: raising both from 1 to 3
// shifts every timeline that crosses one FIFO by exactly 2 cycles.
void test_lsu_fifo_latency_fields() {
    const auto one = lsu_timelines(1, 1);
    const auto three = lsu_timelines(3, 3);
    require(all_equal(three.vector_bus_to_assembly, 3) && all_equal(three.tma_bus_to_register, 3) &&
            three.uni_bus_to_assembly == 3 && all_equal(three.gather_bus_to_read, 3),
        "ingress latency 3: bus -> plucked is not 3 cycles");
    require(all_equal(three.store_slot_to_request, 3), "egress latency 3: slot -> write request is not 3 cycles");
    for (std::size_t j = 0; j < 8; ++j)
        require(three.vector_rf_writes[j] == one.vector_rf_writes[j] + 2, "vector RF write did not shift by exactly 2");
    require(three.tma_place == one.tma_place + 2 && three.uni_rf_write == one.uni_rf_write + 2 &&
            three.gather_rf_write == one.gather_rf_write + 2 && three.store_first_beat == one.store_first_beat + 2,
        "a FIFO-crossing timeline did not shift by exactly 2");
    for (std::size_t k = 0; k < one.total_cycles.size(); ++k)
        require(three.total_cycles[k] == one.total_cycles[k] + 2,
            "scenario " + std::to_string(k) + " ran " + std::to_string(three.total_cycles[k]) + " vs " +
                std::to_string(one.total_cycles[k]) + " cycles: not exactly +2");
}

// E3: one write contract. A TMA store and an stg.v8 of the same 512 B under
// the same (canonical) profile: every write is acknowledged exactly
// gmem.latency_dram_ticks after its last bus beat; TMA credits are released
// in the ack's cycle; the LSU slot at the commit of the ack (the next cycle,
// with the depbar), so both are reusable from the cycle after the ack.
void test_lsu_one_write_contract() {
    const auto config = canonical_profile();
    const auto ack_lags = [](const nanosm::Machine& machine) {
        std::map<uint64_t, uint64_t> last_beat;
        std::vector<int64_t> lags;
        for (const auto& event : machine.edge_events()) {
            if (event.edge == LsuEdge::bus_write_beat) last_beat[event.id] = event.dram_tick;
            if (event.edge == LsuEdge::write_ack)
                lags.push_back(static_cast<int64_t>(event.dram_tick) - static_cast<int64_t>(last_beat.at(event.id)));
        }
        return lags;
    };
    std::vector<int64_t> tma_lags, lsu_lags;
    { // TMA store, 128 fp32
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.shmem_image, 2048);
        write_rank1_tma(shmem, 0, true, 1024, 128, 1024, 128);
        fill_pattern(shmem, 1024, 128, 0);
        image_bytes(launch.gmem_image, 2048);
        nanosm::Machine machine(config, compile(tma_and_wait(0, 128)), std::move(launch));
        machine.enable_edge_probe();
        std::multiset<uint64_t> releases;
        uint32_t credits = 0;
        while (!machine.done()) {
            tick_capped(machine);
            const auto now = machine.tma_sector_credits_used();
            for (uint32_t k = now; k < credits; ++k) releases.insert(machine.cycle() - 1);
            credits = now;
        }
        std::multiset<uint64_t> acks;
        for (const auto& event : machine.edge_events()) if (event.edge == LsuEdge::write_ack) acks.insert(event.cycle);
        require(acks.size() == 16 && releases == acks, "TMA store credits were not released exactly at the acks");
        tma_lags = ack_lags(machine);
        for (uint32_t i = 0; i < 128; ++i) require(machine.read_gmem_u32(1024 + 4 * i) == pattern_word(i), "TMA store data");
    }
    { // stg.v8, 128 words
        nanosm::Launch launch;
        launch.active_warps = 1;
        image_bytes(launch.gmem_image, 2048);
        std::string text = "mov r0, 1024\nsread r1, LANE_ID\n";
        for (uint32_t j = 0; j < 8; ++j) text += "addi r" + std::to_string(8 + j) + ", r1, " + std::to_string(16 * j) + "\n";
        text += "stg.v8 [r0], r8\nhalt\n";
        nanosm::Machine machine(config, compile(text), std::move(launch));
        machine.enable_edge_probe();
        uint64_t released = 0;
        std::size_t in_flight = 0;
        while (!machine.done()) {
            tick_capped(machine);
            const auto now = machine.lsu_commands_in_flight();
            if (now < in_flight) released = machine.cycle() - 1;
            in_flight = now;
        }
        uint64_t last_ack = 0;
        for (const auto& event : machine.edge_events()) if (event.edge == LsuEdge::write_ack) last_ack = event.cycle;
        require(released == last_ack + 1, "the stg.v8 slot was not released at the commit of its last ack");
        lsu_lags = ack_lags(machine);
        for (uint32_t i = 0; i < 128; ++i) require(machine.read_gmem_u32(1024 + 4 * i) == i, "stg.v8 data");
    }
    require(tma_lags.size() == 16 && lsu_lags.size() == 16, "expected sixteen writes per client");
    require(all_equal(tma_lags, config.gmem_latency_dram_ticks) && tma_lags == lsu_lags,
        "TMA store and stg.v8 acks are not both gmem.latency_dram_ticks after the last beat");
}

// E4: lsu.read_streams = 3 (TMA + two vector streams). Two ldg.v8 commands
// land on streams 1 and 2 (round-robin); the legal interleave order
// reorders their sectors ACROSS the streams, never within one; data is
// correct either way. With two streams (both commands on stream 1) the same
// hook cannot interleave them, and the illegal reorder still panics.
void test_lsu_read_streams_generalized() {
    const auto run = [](uint32_t streams, nanosm::ControllerTestOrder order) {
        auto config = canonical_profile();
        config.lsu_read_streams = streams;
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& gmem = image_bytes(launch.gmem_image, 1024);
        fill_pattern(gmem, 0, 256, 0);
        nanosm::Machine machine(config, compile(R"(
            mov r0, 0
            mov r1, 512
            ldg.v8 r8, [r0]
            ldg.v8 r16, [r1]
            halt
        )", config), std::move(launch), {.timeline = true});
        machine.set_controller_test_order(order);
        machine.run();
        for (uint32_t j = 0; j < 8; ++j)
            for (uint32_t lane = 0; lane < 16; ++lane) {
                require(machine.read_reg(0, lane, 8 + j) == pattern_word(16 * j + lane), "stream A data mismatch");
                require(machine.read_reg(0, lane, 16 + j) == pattern_word(128 + 16 * j + lane), "stream B data mismatch");
            }
        std::vector<uint64_t> served;
        for (const auto& span : trace_spans(machine, nanosm::TraceCategory::gmem, "read sector"))
            served.push_back(std::stoull(span.detail.substr(span.detail.find("address=") + 8)));
        return served;
    };
    const auto switches = [](const std::vector<uint64_t>& served) {
        uint32_t count = 0;
        for (std::size_t k = 1; k < served.size(); ++k) count += (served[k] < 512) != (served[k - 1] < 512);
        return count;
    };
    const auto in_order_within = [](const std::vector<uint64_t>& served) {
        uint64_t a = 0, b = 512;
        for (const auto address : served) {
            auto& next = address < 512 ? a : b;
            if (address != next) return false;
            next += 32;
        }
        return true;
    };
    const auto fifo = run(3, nanosm::ControllerTestOrder::fifo);
    require(switches(fifo) == 1 && in_order_within(fifo), "the FIFO controller reordered vector sectors");
    const auto interleaved = run(3, nanosm::ControllerTestOrder::interleave_ordered_streams);
    require(switches(interleaved) >= 16, "read_streams 3: the commands' sectors did not interleave across streams (" +
        std::to_string(switches(interleaved)) + " switches)");
    require(in_order_within(interleaved), "read_streams 3: a stream was reordered within its ID");
    const auto same_stream = run(2, nanosm::ControllerTestOrder::interleave_ordered_streams);
    require(switches(same_stream) == 1 && in_order_within(same_stream),
        "read_streams 2: the legal hook reordered within one ID");
    auto config = canonical_profile();
    config.lsu_read_streams = 3;
    nanosm::Launch launch;
    launch.active_warps = 1;
    image_bytes(launch.gmem_image, 1024);
    nanosm::Machine machine(config, compile("mov r0, 0\nmov r1, 512\nldg.v8 r8, [r0]\nldg.v8 r16, [r1]\nhalt\n", config),
        std::move(launch));
    machine.set_controller_test_order(nanosm::ControllerTestOrder::illegal_ordered_reorder);
    expect_sim_panic(machine, "contract violation: read stream 1 (vector) returned request",
        "an out-of-order vector stream with three read streams");
}

void test_tma_descriptor_encoding() {
    nanosm::TmaDescriptor descriptor;
    descriptor.rank = 3;
    descriptor.direction = nanosm::TmaDirection::store;
    descriptor.gmem_base = 0x1122334455667788ull;
    descriptor.dimensions = {17, 19, 23};
    descriptor.strides = {32, 1024};
    descriptor.box = {4, 5, 6};
    descriptor.element_bytes = 2;
    descriptor.shmem_base = 0x200;
    descriptor.completion_mbar = 0x180;
    const auto bytes = nanosm::encode_tma_descriptor(descriptor);
    require(bytes[0] == 3 && bytes[1] == 1 && bytes[3] == 2, "tma header bytes changed");
    require(bytes[8] == 0x88 && bytes[15] == 0x11, "tma gmem base is not little-endian");
    const auto decoded = nanosm::decode_tma_descriptor(bytes.data(), bytes.size());
    require(
        decoded.rank == descriptor.rank && decoded.direction == descriptor.direction &&
            decoded.gmem_base == descriptor.gmem_base &&
            decoded.dimensions == descriptor.dimensions && decoded.strides == descriptor.strides &&
            decoded.box == descriptor.box && decoded.element_bytes == descriptor.element_bytes &&
            decoded.shmem_base == descriptor.shmem_base &&
            decoded.completion_mbar == descriptor.completion_mbar,
        "tma descriptor did not round trip");

    auto corrupt = bytes;
    corrupt[56] = 1;
    try {
        (void)nanosm::decode_tma_descriptor(corrupt.data(), corrupt.size());
        throw std::runtime_error("tma decoder accepted a nonzero reserved byte");
    } catch (const nanosm::SimError& error) {
        require(
            std::string(error.what()).find("reserved tail") != std::string::npos,
            "reserved tma byte produced the wrong panic");
    }

    descriptor.transpose = true;
    try {
        (void)nanosm::encode_tma_descriptor(descriptor);
        throw std::runtime_error("v0 tma descriptor accepted transpose");
    } catch (const nanosm::SimError& error) {
        require(
            std::string(error.what()).find("transpose") != std::string::npos,
            "unsupported tma transpose produced the wrong panic");
    }
    descriptor.transpose = false;
    descriptor.swizzle = 3;
    try {
        (void)nanosm::encode_tma_descriptor(descriptor);
        throw std::runtime_error("v0 tma descriptor accepted swizzle");
    } catch (const nanosm::SimError& error) {
        require(
            std::string(error.what()).find("swizzle") != std::string::npos,
            "unsupported tma swizzle produced the wrong panic");
    }

    for (uint8_t layout : {uint8_t{1}, uint8_t{2}}) for (bool transpose : {false, true}) {
        nanosm::TmaDescriptor operand;
        operand.rank = 2;
        operand.transpose = transpose;
        operand.element_bytes = 2;
        operand.swizzle = layout;
        operand.dimensions = {96, 96, 1};
        operand.strides = {101, 0};
        operand.box = layout == 1
            ? std::array<uint32_t, 3>{8, 64, 1}
            : std::array<uint32_t, 3>{64, 8, 1};
        operand.shmem_base = 1024;
        operand.completion_mbar = 128;
        const auto encoded = nanosm::encode_tma_descriptor(operand);
        const auto round_trip = nanosm::decode_tma_descriptor(encoded.data(), encoded.size());
        require(round_trip.swizzle == layout && round_trip.transpose == transpose &&
                round_trip.box == operand.box,
            "xor64 descriptor layout/transpose did not round trip");
    }

    nanosm::TmaDescriptor invalid;
    invalid.rank = 2; invalid.element_bytes = 2; invalid.swizzle = 1;
    invalid.dimensions = {64, 64, 1}; invalid.strides = {64, 0};
    invalid.box = {8, 63, 1}; invalid.shmem_base = 1024;
    try {
        (void)nanosm::encode_tma_descriptor(invalid);
        throw std::runtime_error("xor64 descriptor accepted the wrong geometry");
    } catch (const nanosm::SimError& error) {
        require(std::string(error.what()).find("geometry") != std::string::npos,
            "wrong xor64 geometry produced the wrong panic");
    }
    invalid.box = {8, 64, 1}; invalid.shmem_base = 1026;
    try {
        (void)nanosm::encode_tma_descriptor(invalid);
        throw std::runtime_error("xor64 descriptor accepted an unaligned object");
    } catch (const nanosm::SimError& error) {
        require(std::string(error.what()).find("64-byte") != std::string::npos,
            "unaligned xor64 object produced the wrong panic");
    }
}

uint32_t reference_xor64_address(
    uint32_t base, uint8_t layout, uint32_t u, uint32_t v) {
    const uint32_t outer = layout == 1 ? v : u;
    const uint32_t k = layout == 1 ? u : v;
    const uint32_t phase = layout == 1 ? 0 : 2;
    return base + 4 * (4 * outer + ((k >> 1) ^ ((outer >> 2) & 1) ^ phase)) +
        2 * (k & 1);
}

void test_tma_xor64_transpose_load_store() {
    constexpr uint32_t mbar_address = 128;
    constexpr uint32_t tile_address = 1024;
    constexpr uint32_t stride = 83;
    for (uint8_t layout : {uint8_t{1}, uint8_t{2}}) {
        const uint32_t box_x = layout == 1 ? 8 : 64;
        const uint32_t box_y = layout == 1 ? 64 : 8;
        for (bool transpose : {false, true}) {
            const uint32_t footprint_x = transpose ? box_y : box_x;
            const uint32_t footprint_y = transpose ? box_x : box_y;
            const uint32_t dim_x = 3 + footprint_x + 5;
            const uint32_t dim_y = 2 + footprint_y + 3;
            nanosm::TmaDescriptor descriptor;
            descriptor.rank = 2;
            descriptor.transpose = transpose;
            descriptor.element_bytes = 2;
            descriptor.swizzle = layout;
            descriptor.dimensions = {dim_x, dim_y, 1};
            descriptor.strides = {stride, 0};
            descriptor.box = {box_x, box_y, 1};
            descriptor.shmem_base = tile_address;
            descriptor.completion_mbar = mbar_address;

            nanosm::Launch launch;
            launch.active_warps = 1;
            auto& shmem = image_bytes(launch.shmem_image, tile_address + 1088);
            auto& gmem = image_bytes(launch.gmem_image, stride * dim_y * 2);
            for (uint32_t y = 0; y < dim_y; ++y)
                for (uint32_t x = 0; x < dim_x; ++x)
                    write_u16(gmem, 2 * (y * stride + x),
                        static_cast<uint16_t>(0x100 + 3 * y + 5 * x));
            write_tma_descriptor(shmem, 0, descriptor);
            nanosm::MbarState mbar{false, true, 1, 0, 1024};
            write_u64(shmem, mbar_address, nanosm::encode_mbar_state(mbar));
            nanosm::Machine machine(canonical_profile(), compile(R"(
                mov r0, 3
                mov r1, 2
                mov r2, 0
                mov r3, 128
                tma.async [0], r0
            wait:
                mbar.try_wait p0, [r3], 0
                @!p0 bra.uni wait
                halt
            )"), std::move(launch));
            machine.run(20000);
            std::set<uint32_t> addresses;
            for (uint32_t v = 0; v < box_y; ++v) for (uint32_t u = 0; u < box_x; ++u) {
                const uint32_t address = reference_xor64_address(tile_address, layout, u, v);
                require(addresses.insert(address).second, "xor64 map is not bijective");
                const uint32_t gx = 3 + (transpose ? v : u);
                const uint32_t gy = 2 + (transpose ? u : v);
                const uint16_t actual = static_cast<uint16_t>(
                    machine.read_shmem_u32(address & ~3u) >> (8 * (address & 3u)));
                require(actual ==
                    static_cast<uint16_t>(0x100 + 3 * gy + 5 * gx),
                    "xor64 transpose load mapped the wrong element");
            }
            require(addresses.size() == 512, "xor64 map did not cover the operand");

            descriptor.direction = nanosm::TmaDirection::store;
            nanosm::Launch store_launch;
            store_launch.active_warps = 1;
            auto& store_shmem = image_bytes(store_launch.shmem_image, tile_address + 1088);
            auto& store_gmem = image_bytes(store_launch.gmem_image, stride * dim_y * 2);
            std::fill(store_gmem.begin(), store_gmem.end(), 0x5a);
            for (uint32_t v = 0; v < box_y; ++v) for (uint32_t u = 0; u < box_x; ++u)
                write_u16(store_shmem, reference_xor64_address(tile_address, layout, u, v),
                    static_cast<uint16_t>(0x400 + 7 * v + 11 * u));
            write_tma_descriptor(store_shmem, 0, descriptor);
            write_u64(store_shmem, mbar_address, nanosm::encode_mbar_state(mbar));
            nanosm::Machine store_machine(canonical_profile(), compile(R"(
                mov r0, 3
                mov r1, 2
                mov r2, 0
                mov r3, 128
                tma.async [0], r0
            wait:
                mbar.try_wait p0, [r3], 0
                @!p0 bra.uni wait
                halt
            )"), std::move(store_launch));
            store_machine.run(20000);
            for (uint32_t v = 0; v < box_y; ++v) for (uint32_t u = 0; u < box_x; ++u) {
                const uint32_t gx = 3 + (transpose ? v : u);
                const uint32_t gy = 2 + (transpose ? u : v);
                uint16_t actual = 0;
                const uint32_t address = 2 * (gy * stride + gx);
                std::memcpy(&actual, store_machine.gmem().data() + address, sizeof(actual));
                require(actual ==
                    static_cast<uint16_t>(0x400 + 7 * v + 11 * u),
                    "xor64 transpose store mapped the wrong element");
            }
        }
    }
}

void test_tma_xor64_feeds_mma_orientations() {
    // Period 7: coprime to every origin/box offset in play (8, 24, 56), so a
    // misplaced origin cannot alias to the correct values the way a power-of-two
    // period would. All values are exact halfs with exact fp32 products/sums.
    const auto half_bits = [](uint32_t selector) {
        constexpr std::array<uint16_t, 7> bits{
            0x3800, 0x3c00, 0x4000, 0xbc00, 0x3e00, 0xb800, 0x4200};
        return bits[selector % 7];
    };
    const auto half_value = [](uint32_t selector) {
        constexpr std::array<float, 7> values{0.5f, 1.0f, 2.0f, -1.0f, 1.5f, -0.5f, 3.0f};
        return values[selector % 7];
    };
    for (bool pv : {false, true}) {
        constexpr uint32_t stride_a = 21;
        constexpr uint32_t stride_b_qk = 23;
        constexpr uint32_t stride_b_pv = 71;
        constexpr uint64_t base_a = 0;
        constexpr uint64_t base_b = 8192;
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.shmem_image, 7168);
        auto& gmem = image_bytes(launch.gmem_image, 20000);
        for (uint32_t m = 0; m < 64; ++m)
            for (uint32_t k = 0; k < 16; ++k)
                write_u16(gmem, base_a + 2 * (m * stride_a + k), half_bits(m + 2 * k));
        if (!pv) {
            for (uint32_t n = 0; n < 64; ++n)
                for (uint32_t k = 0; k < 16; ++k)
                    write_u16(gmem, base_b + 2 * (n * stride_b_qk + k), half_bits(3 * n + k + 1));
        } else {
            for (uint32_t k = 0; k < 16; ++k)
                for (uint32_t n = 0; n < 64; ++n)
                    write_u16(gmem, base_b + 2 * (k * stride_b_pv + n), half_bits(3 * n + k + 1));
        }
        for (uint32_t chunk = 0; chunk < 2; ++chunk) {
            nanosm::TmaDescriptor a;
            a.rank = 2; a.element_bytes = 2; a.swizzle = 1;
            a.gmem_base = base_a; a.dimensions = {16, 64, 1};
            a.strides = {stride_a, 0}; a.box = {8, 64, 1};
            a.shmem_base = 1024 + 2048 * chunk; a.completion_mbar = 512 + 16 * chunk;
            nanosm::TmaDescriptor b;
            b.rank = 2; b.element_bytes = 2; b.swizzle = 2; b.gmem_base = base_b;
            b.box = {64, 8, 1}; b.shmem_base = 2048 + 2048 * chunk;
            b.completion_mbar = 520 + 16 * chunk;
            if (!pv) {
                b.transpose = true; b.dimensions = {16, 64, 1}; b.strides = {stride_b_qk, 0};
            } else {
                b.dimensions = {64, 16, 1}; b.strides = {stride_b_pv, 0};
            }
            write_tma_descriptor(shmem, 64 * (2 * chunk), a);
            write_tma_descriptor(shmem, 64 * (2 * chunk + 1), b);
            nanosm::MbarState ready{false, true, 1, 0, 1024};
            write_u64(shmem, a.completion_mbar, nanosm::encode_mbar_state(ready));
            write_u64(shmem, b.completion_mbar, nanosm::encode_mbar_state(ready));
        }
        nanosm::MbarState commit{false, true, 1, 1, 0};
        write_u64(shmem, 544, nanosm::encode_mbar_state(commit));
        // Chunk 1 origins per rfc07's table: A advances k in global x for both
        // orientations; B advances k in x under QK^T (transpose=1) but in y
        // under PV (transpose=0) â€” V's rows are keys.
        const std::string chunk1_b = pv
            ? "mov r0, 0\nmov r1, 8\ntma.async [192], r0\n"
            : "tma.async [192], r0\n";
        const auto kernel = compile(std::string(R"(
            .acc C, 0, 0, 64, 64
            mov r0, 0
            mov r1, 0
            mov r2, 0
            tma.async [0], r0
            tma.async [64], r0
            mov r0, 8
            tma.async [128], r0
)") + chunk1_b + R"(
            mov r4, 512
        wait0:
            mbar.try_wait p0, [r4], 0
            @!p0 bra.uni wait0
            mov r4, 520
        wait1:
            mbar.try_wait p0, [r4], 0
            @!p0 bra.uni wait1
            mov r4, 528
        wait2:
            mbar.try_wait p0, [r4], 0
            @!p0 bra.uni wait2
            mov r4, 536
        wait3:
            mbar.try_wait p0, [r4], 0
            @!p0 bra.uni wait3
            mov r10, 1024
            mov r11, 2048
            mma.async.init C, [r10], [r11]
            mov r10, 3072
            mov r11, 4096
            mma.async C, [r10], [r11]
            mov r4, 544
            mma.commit [r4]
        wait_commit:
            mbar.try_wait p0, [r4], 0
            @!p0 bra.uni wait_commit
            mov r8, 0
            mov r9, 0
            acc.ld.v8 r20, C, r8
            mov r12, 48
            mov r13, 56
            acc.ld.v8 r28, C, r12
            halt
        )");
        nanosm::Machine machine(canonical_profile(), kernel, std::move(launch), {.timeline = true});
        machine.run(50000);
        const auto expected_c = [&](uint32_t m, uint32_t n) {
            float sum = 0;
            for (uint32_t k = 0; k < 16; ++k)
                sum += half_value(m + 2 * k) * half_value(3 * n + k + 1);
            return sum;
        };
        for (uint32_t lane = 0; lane < 16; ++lane) {
            for (uint32_t j = 0; j < 8; ++j) {
                require(machine.read_reg(0, lane, 20 + j) ==
                        nanosm::float_bits(expected_c(lane, j)),
                    pv ? "TMA-fed PV orientation produced the wrong MMA value"
                       : "TMA-fed QK transpose produced the wrong MMA value");
                require(machine.read_reg(0, lane, 28 + j) ==
                        nanosm::float_bits(expected_c(48 + lane, 56 + j)),
                    "TMA-fed MMA far-corner block is wrong");
            }
        }
        const auto& summary = machine.trace().summary();
        require(summary.tma_commands == 4 && summary.tma_command_completions == 4,
            "TMA-fed MMA proof did not complete all operand commands");
        require(summary.mma_commands == 2 && summary.mma_shmem_rounds == summary.mma_input_steps,
            "TMA-authored operands did not retain one combined MMA bank round per step");
    }
}

// Independent restatement of the stage-granular map: logical K picks the
// 1024-byte operand object and the position inside it. Never calls the
// production helper.
uint32_t reference_wide_xor64(uint32_t base, uint8_t layout, uint32_t u, uint32_t v) {
    const uint32_t k = layout == 1 ? u : v;
    const uint32_t outer = layout == 1 ? v : u;
    const uint32_t phase = layout == 1 ? 0u : 2u;
    return base + 1024 * (k / 8) +
        4 * (4 * outer + (((k % 8) >> 1) ^ ((outer >> 2) & 1) ^ phase)) +
        2 * (k % 2);
}

void test_tma_wide_xor64_validation() {
    nanosm::TmaDescriptor descriptor;
    descriptor.rank = 2;
    descriptor.element_bytes = 2;
    descriptor.dimensions = {128, 128, 1};
    descriptor.strides = {128, 0};
    descriptor.shmem_base = 1024;
    const auto expect = [&](uint8_t swizzle, std::array<uint32_t, 3> box, bool ok) {
        descriptor.swizzle = swizzle;
        descriptor.box = box;
        bool accepted = true;
        try {
            (void)nanosm::encode_tma_descriptor(descriptor);
        } catch (const nanosm::SimError&) {
            accepted = false;
        }
        require(accepted == ok, "wide xor64 box validation disagreed with the contract");
    };
    expect(1, {16, 64, 1}, true);
    expect(1, {64, 64, 1}, true);
    expect(2, {64, 16, 1}, true);
    expect(2, {64, 64, 1}, true);
    expect(1, {12, 64, 1}, false);
    expect(1, {72, 64, 1}, false);
    expect(1, {16, 32, 1}, false);
    expect(2, {32, 16, 1}, false);

    nanosm::TmaDescriptor big;
    big.rank = 2;
    big.element_bytes = 2;
    big.swizzle = 1;
    big.dimensions = {128, 128, 1};
    big.strides = {128, 0};
    big.box = {64, 64, 1};
    big.shmem_base = 61440; // 8192-byte span ends past the 64 KiB shmem
    big.completion_mbar = 33024;
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.shmem_image, 33032);
    image_bytes(launch.gmem_image, 65536);
    write_tma_descriptor(shmem, 0, big);
    write_u64(shmem, 33024, nanosm::encode_mbar_state({false, true, 1, 0, 8192}));
    nanosm::Machine machine(canonical_profile(),
        compile("mov r0, 0\nmov r1, 0\nmov r2, 0\ntma.async [0], r0\nhalt\n"),
        std::move(launch));
    try {
        machine.run(20000);
        throw std::runtime_error("oversized xor64 span was accepted");
    } catch (const nanosm::SimError& error) {
        require(std::string(error.what()).find("does not fit in shmem") != std::string::npos,
            "oversized xor64 span rejected with the wrong diagnostic");
    }
}

void test_tma_wide_xor64_matches_narrow() {
    constexpr uint32_t mbar_address = 576;
    constexpr uint32_t tile_address = 1024;
    constexpr uint32_t stride = 83;
    constexpr uint32_t dim_x = 80;
    constexpr uint32_t dim_y = 70;
    struct Case {
        uint8_t layout;
        bool transpose;
        char step_axis; // global axis the K origin advances along, per k8 chunk
    };
    for (const auto& test_case : {Case{1, false, 'x'}, Case{2, false, 'y'}, Case{2, true, 'x'}}) {
        const auto run_span = [&](uint32_t commands) {
            nanosm::Launch launch;
            launch.active_warps = 1;
            auto& shmem = image_bytes(launch.shmem_image, tile_address + 8192);
            auto& gmem = image_bytes(launch.gmem_image, 2 * stride * dim_y + 64);
            for (uint32_t y = 0; y < dim_y; ++y)
                for (uint32_t x = 0; x < dim_x; ++x)
                    write_u16(gmem, 2 * (y * stride + x),
                        static_cast<uint16_t>(0x100 + 3 * y + 5 * x));
            const uint32_t k_extent = 64 / commands;
            for (uint32_t i = 0; i < commands; ++i) {
                nanosm::TmaDescriptor descriptor;
                descriptor.rank = 2;
                descriptor.transpose = test_case.transpose;
                descriptor.element_bytes = 2;
                descriptor.swizzle = test_case.layout;
                descriptor.dimensions = {dim_x, dim_y, 1};
                descriptor.strides = {stride, 0};
                descriptor.box = test_case.layout == 1
                    ? std::array<uint32_t, 3>{k_extent, 64, 1}
                    : std::array<uint32_t, 3>{64, k_extent, 1};
                descriptor.shmem_base = tile_address + 1024 * (k_extent / 8) * i;
                descriptor.completion_mbar = mbar_address;
                write_tma_descriptor(shmem, 64 * i, descriptor);
            }
            write_u64(shmem, mbar_address,
                nanosm::encode_mbar_state({false, true, 1, 0, 8192}));
            std::string source = "mov r0, 3\nmov r1, 2\nmov r2, 0\nmov r3, " +
                std::to_string(mbar_address) + "\n";
            for (uint32_t i = 0; i < commands; ++i) {
                if (i != 0) {
                    source += test_case.step_axis == 'x'
                        ? "addi r0, r0, 8\n" : "addi r1, r1, 8\n";
                }
                source += "tma.async [" + std::to_string(64 * i) + "], r0\n";
            }
            source += "wait:\nmbar.try_wait p0, [r3], 0\n@!p0 bra.uni wait\nhalt\n";
            nanosm::Machine machine(canonical_profile(), compile(source), std::move(launch));
            machine.run(200000);
            std::vector<uint32_t> words;
            words.reserve(2048);
            for (uint32_t offset = 0; offset < 8192; offset += 4)
                words.push_back(machine.read_shmem_u32(tile_address + offset));
            return words;
        };
        const auto wide = run_span(1);
        const auto narrow = run_span(8);
        require(wide == narrow, "wide xor64 command disagreed with eight narrow commands");
        std::set<uint32_t> addresses;
        for (uint32_t v = 0; v < 64; ++v) {
            for (uint32_t u = 0; u < 64; ++u) {
                const auto address =
                    reference_wide_xor64(tile_address, test_case.layout, u, v);
                require(address >= tile_address && address < tile_address + 8192,
                    "wide xor64 address left the span");
                require(addresses.insert(address).second, "wide xor64 map is not bijective");
                const uint32_t gx = 3 + (test_case.transpose ? v : u);
                const uint32_t gy = 2 + (test_case.transpose ? u : v);
                const auto word = wide[(address - tile_address) / 4];
                const auto actual = static_cast<uint16_t>(word >> (8 * (address & 3u)));
                require(actual == static_cast<uint16_t>(0x100 + 3 * gy + 5 * gx),
                    "wide xor64 load mapped the wrong element");
            }
        }
        require(addresses.size() == 4096, "wide xor64 map did not cover the span");
    }
}

void test_tma_wide_xor64_sector_economics() {
    constexpr uint32_t mbar_address = 576;
    constexpr uint32_t tile_address = 1024;
    // Natural head-innermost K: 64 keys x 64 fp16, dense 128-byte rows.
    const auto run_summary = [&](uint32_t commands) {
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.shmem_image, tile_address + 8192);
        auto& gmem = image_bytes(launch.gmem_image, 8192);
        for (uint32_t i = 0; i < 4096; ++i)
            write_u16(gmem, 2 * i, static_cast<uint16_t>((i * 2654435761u) >> 16));
        const uint32_t k_extent = 64 / commands;
        for (uint32_t i = 0; i < commands; ++i) {
            nanosm::TmaDescriptor descriptor;
            descriptor.rank = 2;
            descriptor.transpose = true; // the QK^T orientation on the natural layout
            descriptor.element_bytes = 2;
            descriptor.swizzle = 2;
            descriptor.dimensions = {64, 64, 1};
            descriptor.strides = {64, 0};
            descriptor.box = {64, k_extent, 1};
            descriptor.shmem_base = tile_address + 1024 * (k_extent / 8) * i;
            descriptor.completion_mbar = mbar_address;
            write_tma_descriptor(shmem, 64 * i, descriptor);
        }
        write_u64(shmem, mbar_address, nanosm::encode_mbar_state({false, true, 1, 0, 8192}));
        std::string source = "mov r0, 0\nmov r1, 0\nmov r2, 0\nmov r3, " +
            std::to_string(mbar_address) + "\n";
        for (uint32_t i = 0; i < commands; ++i) {
            if (i != 0) source += "addi r0, r0, 8\n";
            source += "tma.async [" + std::to_string(64 * i) + "], r0\n";
        }
        source += "wait:\nmbar.try_wait p0, [r3], 0\n@!p0 bra.uni wait\nhalt\n";
        nanosm::Machine machine(canonical_profile(), compile(source), std::move(launch));
        machine.run(200000);
        return machine.trace().summary();
    };
    const auto wide = run_summary(1);
    require(wide.tma_command_completions == 1 && wide.tma_useful_bytes == 8192 &&
            wide.tma_sector_bytes == 8192,
        "wide natural-layout load was not 1.00x sector traffic");
    const auto narrow = run_summary(8);
    require(narrow.tma_command_completions == 8 && narrow.tma_useful_bytes == 8192 &&
            narrow.tma_sector_bytes == 16384,
        "narrow chunked load did not show the 2x half-sector refetch");
}

// The pre-2026-10-03 fp16 operand formula, written out literally. The byte
// primitive must reproduce it for every MMA/TMA fp16 address in play.
void test_xor64_byte_primitive_matches_fp16_formula() {
    for (uint32_t base : {0u, 1024u, 40960u, 65472u}) {
        for (uint32_t phase : {0u, 2u}) {
            for (uint32_t outer = 0; outer < 64; ++outer) {
                for (uint32_t k = 0; k < 8; ++k) {
                    const uint32_t old = base +
                        4 * (4 * outer + ((k >> 1) ^ ((outer >> 2) & 1) ^ phase)) +
                        2 * (k & 1);
                    require(nanosm::xor64_operand_address(base, outer, k, phase) == old,
                        "xor64 operand wrapper diverged from the fp16 formula");
                    require(nanosm::xor64_byte_address(base, outer, 2 * k, phase) == old &&
                            nanosm::xor64_byte_address(base, outer, 2 * k + 1, phase) == old + 1,
                        "xor64 byte primitive diverged from the fp16 formula");
                }
            }
        }
    }
}

void test_tma_fp32_xor64_validation() {
    const auto accepted = [](uint8_t element_bytes, uint8_t swizzle,
                             std::array<uint32_t, 3> box, nanosm::TmaDirection direction) {
        nanosm::TmaDescriptor descriptor;
        descriptor.rank = 2;
        descriptor.direction = direction;
        descriptor.element_bytes = element_bytes;
        descriptor.swizzle = swizzle;
        descriptor.dimensions = {128, 128, 1};
        descriptor.strides = {128, 0};
        descriptor.box = box;
        descriptor.shmem_base = 1024;
        descriptor.completion_mbar = 128;
        try {
            (void)nanosm::encode_tma_descriptor(descriptor);
            return std::string();
        } catch (const nanosm::SimError& error) {
            return std::string(error.what());
        }
    };
    const auto load = nanosm::TmaDirection::load;
    const auto store = nanosm::TmaDirection::store;
    // fp32: four elements per 16-byte row; K extent a multiple of 4, up to 64.
    for (auto direction : {load, store}) {
        require(accepted(4, 1, {64, 64, 1}, direction).empty(),
            "fp32 xor64_a 64x64 tile was rejected");
        require(accepted(4, 2, {64, 64, 1}, direction).empty(),
            "fp32 xor64_b 64x64 tile was rejected");
        require(accepted(4, 1, {4, 64, 1}, direction).empty() &&
                accepted(4, 1, {12, 64, 1}, direction).empty(),
            "fp32 xor64 single/odd object counts were rejected");
    }
    for (std::array<uint32_t, 3> box : {std::array<uint32_t, 3>{6, 64, 1},
                                        std::array<uint32_t, 3>{2, 64, 1},
                                        std::array<uint32_t, 3>{68, 64, 1},
                                        std::array<uint32_t, 3>{64, 32, 1}}) {
        require(accepted(4, 1, box, store).find("geometry") != std::string::npos,
            "fp32 xor64 accepted a K extent that is not whole objects");
    }
    require(accepted(4, 2, {64, 6, 1}, load).find("geometry") != std::string::npos,
        "fp32 xor64_b accepted a K extent that is not whole objects");
    require(!accepted(8, 1, {64, 64, 1}, store).empty(), "8-byte xor64 elements were accepted");
    require(accepted(1, 1, {64, 64, 1}, store).find("requires rank 2") != std::string::npos,
        "1-byte xor64 elements were accepted");
    // fp16 keeps its eight-element rows: 12 is whole fp32 objects, not fp16.
    for (std::array<uint32_t, 3> box : {std::array<uint32_t, 3>{12, 64, 1},
                                        std::array<uint32_t, 3>{4, 64, 1},
                                        std::array<uint32_t, 3>{72, 64, 1},
                                        std::array<uint32_t, 3>{16, 32, 1}}) {
        require(accepted(2, 1, box, load).find("geometry") != std::string::npos,
            "fp16 xor64 wrong geometry was accepted after the fp32 generalization");
    }
}

// Independent fp32 xor64_a address: object = col / 4 (four floats per
// 16-byte row), row = the 16-byte row, word-in-row = (col % 4) XOR bit 2 of
// the row. Never calls the production primitive.
uint32_t reference_fp32_xor64_a(uint32_t base, uint32_t row, uint32_t col) {
    return base + 1024 * (col / 4) + 16 * row + 4 * ((col % 4) ^ ((row / 4) % 2));
}

float fp32_tile_value(uint32_t row, uint32_t col) {
    // Periods 31 in both axes: coprime to 4, 8, 16, 64. Small integers, exact.
    return static_cast<float>(static_cast<int>((row * 17 + col * 11) % 31) - 15);
}

void test_tma_fp32_xor64_store_and_load() {
    constexpr uint32_t mbar_address = 128;
    constexpr uint32_t tile_address = 1024;
    constexpr uint32_t tile_bytes = 16384;
    constexpr uint64_t gmem_base = 512;
    constexpr uint32_t gmem_size = 512 + tile_bytes + 512;

    std::set<uint32_t> addresses;
    for (uint32_t row = 0; row < 64; ++row) {
        for (uint32_t col = 0; col < 64; ++col) {
            const auto address = reference_fp32_xor64_a(tile_address, row, col);
            require(address % 4 == 0 && address >= tile_address &&
                    address < tile_address + tile_bytes,
                "fp32 xor64 address left the 16-object span");
            require(addresses.insert(address).second, "fp32 xor64 map is not bijective");
        }
    }
    require(addresses.size() == 4096, "fp32 xor64 map did not cover the 16 objects");

    const auto make_descriptor = [&](nanosm::TmaDirection direction, uint8_t swizzle) {
        nanosm::TmaDescriptor descriptor;
        descriptor.rank = 2;
        descriptor.direction = direction;
        descriptor.element_bytes = 4;
        descriptor.swizzle = swizzle;
        descriptor.gmem_base = gmem_base;
        descriptor.dimensions = {64, 64, 1};
        descriptor.strides = {64, 0};
        descriptor.box = {64, 64, 1};
        descriptor.shmem_base = tile_address;
        descriptor.completion_mbar = mbar_address;
        return descriptor;
    };
    const char* program = R"(
        mov r0, 0
        mov r1, 0
        mov r2, 0
        mov r3, 128
        tma.async [0], r0
    wait:
        mbar.try_wait p0, [r3], 0
        @!p0 bra.uni wait
        halt
    )";
    const nanosm::MbarState mbar{false, true, 1, 0, tile_bytes};

    // STORE: shmem tile authored at the independent address -> row-major gmem.
    const auto run_store = [&](uint8_t swizzle) {
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.shmem_image, tile_address + tile_bytes);
        auto& gmem = image_bytes(launch.gmem_image, gmem_size);
        std::fill(gmem.begin(), gmem.end(), 0x5a);
        for (uint32_t row = 0; row < 64; ++row)
            for (uint32_t col = 0; col < 64; ++col)
                write_u32(shmem, swizzle == 1
                        ? reference_fp32_xor64_a(tile_address, row, col)
                        : tile_address + 4 * (64 * row + col),
                    nanosm::float_bits(fp32_tile_value(row, col)));
        write_tma_descriptor(shmem, 0, make_descriptor(nanosm::TmaDirection::store, swizzle));
        write_u64(shmem, mbar_address, nanosm::encode_mbar_state(mbar));
        nanosm::Machine machine(canonical_profile(), compile(program), std::move(launch));
        machine.run(200000);
        for (uint32_t row = 0; row < 64; ++row) {
            for (uint32_t col = 0; col < 64; ++col) {
                uint32_t actual = 0;
                std::memcpy(&actual, machine.gmem().data() + gmem_base + 4 * (64 * row + col),
                    sizeof(actual));
                require(actual == nanosm::float_bits(fp32_tile_value(row, col)),
                    "fp32 xor64 TMA store mapped the wrong element at row " +
                        std::to_string(row) + " col " + std::to_string(col));
            }
        }
        for (uint32_t offset = 0; offset < gmem_size; ++offset) {
            if (offset >= gmem_base && offset < gmem_base + tile_bytes) continue;
            require(machine.gmem()[offset] == 0x5a, "fp32 TMA store wrote outside its tile");
        }
        return machine.trace().summary();
    };
    const auto swizzled_store = run_store(1);
    // A 32-byte sector is eight consecutive fp32 of one row: four in object
    // col/4 and four in the next. Both halves sit in the same row of their
    // objects, so on the same four banks (1024 B = 256 words = 0 mod 16) at
    // different words: two port-A rounds per sector. Linear placement of the
    // same sector is eight adjacent words: one round.
    require(swizzled_store.tma_command_completions == 1 &&
            swizzled_store.tma_useful_bytes == tile_bytes &&
            swizzled_store.tma_sector_bytes == tile_bytes,
        "fp32 xor64 TMA store was not 1.00x sector traffic");
    require(swizzled_store.tma_port_requests == 512 && swizzled_store.tma_port_rounds == 1024,
        "fp32 xor64 TMA store did not read two port-A rounds per sector");
    const auto linear_store = run_store(0);
    require(linear_store.tma_port_requests == 512 && linear_store.tma_port_rounds == 512,
        "linear fp32 TMA store did not read one port-A round per sector");

    // LOAD: row-major gmem -> shmem; every element checked at the independent address.
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.shmem_image, tile_address + tile_bytes);
    auto& gmem = image_bytes(launch.gmem_image, gmem_size);
    std::fill(shmem.begin() + tile_address, shmem.end(), 0x7f);
    for (uint32_t row = 0; row < 64; ++row)
        for (uint32_t col = 0; col < 64; ++col)
            write_u32(gmem, static_cast<uint32_t>(gmem_base) + 4 * (64 * row + col),
                nanosm::float_bits(fp32_tile_value(row, col)));
    write_tma_descriptor(shmem, 0, make_descriptor(nanosm::TmaDirection::load, 1));
    write_u64(shmem, mbar_address, nanosm::encode_mbar_state(mbar));
    nanosm::Machine machine(canonical_profile(), compile(program), std::move(launch));
    machine.run(200000);
    for (uint32_t row = 0; row < 64; ++row) {
        for (uint32_t col = 0; col < 64; ++col) {
            require(machine.read_shmem_u32(reference_fp32_xor64_a(tile_address, row, col)) ==
                    nanosm::float_bits(fp32_tile_value(row, col)),
                "fp32 xor64 TMA load mapped the wrong element at row " +
                    std::to_string(row) + " col " + std::to_string(col));
        }
    }
    const auto& load_summary = machine.trace().summary();
    require(load_summary.tma_port_requests == 512 && load_summary.tma_port_rounds == 1024,
        "fp32 xor64 TMA load did not write two port-A rounds per sector");
}

void test_tma_rank1_load_and_oob_zero_fill() {
    constexpr uint32_t descriptor_address = 0;
    constexpr uint32_t mbar_address = 128;
    constexpr uint32_t tile_address = 256;
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.shmem_image, tile_address + 32);
    auto& gmem = image_bytes(launch.gmem_image, 32);
    for (uint32_t i = 0; i < 4; ++i) write_u32(gmem, i * 4, 0x100u + i);

    nanosm::TmaDescriptor descriptor;
    descriptor.gmem_base = 0;
    descriptor.dimensions = {4, 1, 1};
    descriptor.box = {8, 1, 1};
    descriptor.shmem_base = tile_address;
    descriptor.completion_mbar = mbar_address;
    write_tma_descriptor(shmem, descriptor_address, descriptor);
    nanosm::MbarState mbar;
    mbar.valid = true;
    mbar.expected = 1;
    mbar.tx_bytes = 32;
    write_u64(shmem, mbar_address, nanosm::encode_mbar_state(mbar));

    nanosm::Machine machine(canonical_profile(), compile(R"(
        mov r0, -4
        mov r1, 0
        mov r2, 0
        mov r3, 128
        tma.async [0], r0
    wait:
        mbar.try_wait p0, [r3], 0
        @!p0 bra.uni wait
        halt
    )"), std::move(launch));
    machine.run();
    for (uint32_t i = 0; i < 4; ++i) {
        require(machine.read_shmem_u32(tile_address + i * 4) == 0, "tma OOB load was not zero");
        require(
            machine.read_shmem_u32(tile_address + (i + 4) * 4) == 0x100u + i,
            "tma rank-1 load mapped the wrong global element");
    }
    const auto completed = machine.read_mbar_state(mbar_address);
    require(completed.phase && completed.tx_bytes == 0, "tma load completed the mbar incorrectly");
    const auto& summary = machine.trace().summary();
    require(
        summary.tma_commands == 1 && summary.tma_command_completions == 1,
        "tma load command lifetime was not counted");
    require(
        summary.tma_logical_bytes == 32 && summary.tma_useful_bytes == 16 &&
            summary.tma_sector_bytes == 32,
        "tma load logical/useful/physical byte accounting is wrong");
}

void test_tma_store_preserves_other_sector_bytes() {
    constexpr uint32_t descriptor_address = 64;
    constexpr uint32_t mbar_address = 128;
    constexpr uint32_t tile_address = 256;
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.shmem_image, tile_address + 16);
    auto& gmem = image_bytes(launch.gmem_image, 64);
    std::fill(gmem.begin(), gmem.end(), 0x5a);
    for (uint32_t i = 0; i < 4; ++i) write_u32(shmem, tile_address + i * 4, 0x200u + i);

    nanosm::TmaDescriptor descriptor;
    descriptor.direction = nanosm::TmaDirection::store;
    descriptor.gmem_base = 12;
    descriptor.dimensions = {4, 1, 1};
    descriptor.box = {4, 1, 1};
    descriptor.shmem_base = tile_address;
    descriptor.completion_mbar = mbar_address;
    write_tma_descriptor(shmem, descriptor_address, descriptor);
    nanosm::MbarState mbar;
    mbar.valid = true;
    mbar.expected = 1;
    mbar.tx_bytes = 16;
    write_u64(shmem, mbar_address, nanosm::encode_mbar_state(mbar));

    nanosm::Machine machine(canonical_profile(), compile(R"(
        mov r0, 0
        mov r1, 0
        mov r2, 0
        mov r3, 128
        tma.async [64], r0
    wait:
        mbar.try_wait p0, [r3], 0
        @!p0 bra.uni wait
        halt
    )"), std::move(launch));
    machine.run();
    for (uint32_t address = 0; address < 12; ++address) {
        require(machine.gmem()[address] == 0x5a, "tma store damaged the sector prefix");
    }
    for (uint32_t i = 0; i < 4; ++i) {
        require(machine.read_gmem_u32(12 + i * 4) == 0x200u + i, "tma store value mismatch");
    }
    for (uint32_t address = 28; address < 64; ++address) {
        require(machine.gmem()[address] == 0x5a, "tma store damaged the sector suffix");
    }
    const auto& summary = machine.trace().summary();
    require(
        summary.tma_logical_bytes == 16 && summary.tma_useful_bytes == 16 &&
            summary.tma_sector_bytes == 32,
        "tma store logical/useful/physical byte accounting is wrong");
}

void test_tma_rank3_strides() {
    constexpr uint32_t descriptor_address = 0;
    constexpr uint32_t mbar_address = 128;
    constexpr uint32_t tile_address = 256;
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.shmem_image, tile_address + 16);
    auto& gmem = image_bytes(launch.gmem_image, 64 * 4);
    for (uint32_t i = 0; i < 64; ++i) write_u32(gmem, i * 4, 1000 + i);

    nanosm::TmaDescriptor descriptor;
    descriptor.rank = 3;
    descriptor.dimensions = {3, 4, 2};
    descriptor.strides = {5, 30};
    descriptor.box = {2, 2, 1};
    descriptor.shmem_base = tile_address;
    descriptor.completion_mbar = mbar_address;
    write_tma_descriptor(shmem, descriptor_address, descriptor);
    nanosm::MbarState mbar;
    mbar.valid = true;
    mbar.expected = 1;
    mbar.tx_bytes = 16;
    write_u64(shmem, mbar_address, nanosm::encode_mbar_state(mbar));

    nanosm::Machine machine(canonical_profile(), compile(R"(
        mov r0, 1
        mov r1, 1
        mov r2, 1
        mov r3, 128
        tma.async [0], r0
    wait:
        mbar.try_wait p0, [r3], 0
        @!p0 bra.uni wait
        halt
    )"), std::move(launch));
    machine.run();
    const std::array<uint32_t, 4> expected{1036, 1037, 1041, 1042};
    for (uint32_t i = 0; i < expected.size(); ++i) {
        require(
            machine.read_shmem_u32(tile_address + i * 4) == expected[i],
            "tma rank-3 stride mapping is wrong");
    }
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
    // ldg.uni: one declared sector request, two 16-byte beats. (This test
    // used a 16-lane scalar ldg when the coalescer made that two sectors.)
    nanosm::Machine machine(config, compile(R"(
        mov r0, 0
        ldg.uni r1, [r0]
        halt
    )", config), std::move(launch), {.timeline = true, .debug = true});
    machine.run();
    std::vector<uint64_t> beat_cycles;
    for (const auto& event : machine.trace().events()) {
        if (event.category == nanosm::TraceCategory::gmem && event.name == "bus beat") {
            beat_cycles.push_back(event.start_cycle);
        }
    }
    require(beat_cycles.size() == 2, "one sector did not take two 16-byte DRAM beats");
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
    const ScratchFile file("async_trace");
    machine.trace().write_perfetto(file.string());

    nlohmann::json document;
    {
        std::ifstream input(file.string());
        require(static_cast<bool>(input), "trace file was not written");
        input >> document;
    }

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

std::vector<uint64_t> issued_at(const nanosm::Machine& machine, uint32_t pc) {
    std::vector<uint64_t> result;
    for (const auto& event : machine.trace().events()) {
        if (event.category == nanosm::TraceCategory::instruction &&
            event.name != "RF read" && event.id == pc) result.push_back(event.start_cycle);
    }
    return result;
}

void check_read_capacity(const nanosm::Machine& machine, const nanosm::DeviceProfile& profile) {
    std::map<std::pair<uint64_t, uint32_t>, uint32_t> counts;
    require(!machine.trace().truncated(), "RF capacity check needs an untruncated trace");
    for (const auto& event : machine.trace().events()) {
        if (event.name != "RF read") continue;
        const auto bank = static_cast<uint32_t>(event.value);
        require(bank == event.id % profile.rf_banks, "read trace has the wrong bank");
        require(++counts[{event.start_cycle, bank}] <= profile.rf_read_ports_per_bank,
            "actual RF captures exceeded per-bank read capacity");
    }
}

void test_rf_read_profile_and_plan() {
    auto profile = canonical_profile();
    auto other = profile;
    other.rf_arithmetic_read_cycles = 3;
    require(nanosm::device_profile_fingerprint(profile) != nanosm::device_profile_fingerprint(other),
        "RF read window was omitted from the profile fingerprint");
    std::ifstream input("profiles/nanosm-v0.json");
    nlohmann::json document;
    input >> document;
    for (const auto value : {nlohmann::json(0), nlohmann::json(33), nlohmann::json(-1),
             nlohmann::json(2.5), nlohmann::json(4294967299ull)}) {
        document["rf"]["arithmetic_read_cycles"] = value;
        try {
            (void)nanosm::parse_device_profile(document.dump());
            throw std::runtime_error("invalid RF read window was accepted");
        } catch (const nanosm::ProfileError&) {}
    }
    document["rf"].erase("arithmetic_read_cycles");
    require(nanosm::parse_device_profile(document.dump()).rf_arithmetic_read_cycles == 1,
        "omitted RF read window did not default to one");

    nanosm::Instruction add;
    add.opcode = nanosm::Opcode::add;
    add.src0 = 0;
    add.src1 = 3;
    auto reads = nanosm::rf_read_plan(add, other, {0, 3});
    require(reads.size() == 2 && reads[0].offset == 1 && reads[1].offset == 2,
        "same-bank reads were not right-aligned in the fixed window");
    reads = nanosm::rf_read_plan(add, other, {0, 0});
    require(reads.size() == 1 && reads[0].offset == 2, "duplicate operand did not broadcast");
    other.rf_read_ports_per_bank = 2;
    reads = nanosm::rf_read_plan(add, other, {0, 3});
    require(reads.size() == 2 && reads[0].offset == 2 && reads[1].offset == 2,
        "multiport profile did not use both bank ports");
    try {
        (void)nanosm::compile_qtx("add r2, r0, r3\nhalt", profile);
        throw std::runtime_error("impossible one-cycle operand schedule was accepted");
    } catch (const nanosm::ProfileError&) {}
}

void test_rf_deferred_fadd_and_halt() {
    auto profile = canonical_profile();
    profile.rf_arithmetic_read_cycles = 3;
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& image = image_bytes(launch.shmem_image, 8);
    write_u32(image, 0, nanosm::float_bits(1.25f));
    write_u32(image, 4, nanosm::float_bits(2.5f));
    nanosm::Machine machine(profile, compile(R"(
        lds r0, [r2]
        lds r3, [r2+4]
        fadd r4, r0, r3
        mov r8, 9
        halt
    )", profile), std::move(launch), {.timeline = true, .debug = true});
    uint64_t changed = 0;
    while (!machine.done() && machine.cycle() < 1000) {
        const auto tick = machine.cycle();
        machine.tick();
        if (machine.read_reg(0, 0, 4) != 0 && changed == 0) changed = tick;
    }
    require(machine.done(), "deferred FADD pipeline did not drain");
    const auto issue = issued_at(machine, 16).at(0);
    require(changed == issue + 2 + profile.timing.fadd, "accepted FADD latency varied");
    require(machine.read_reg(0, 0, 4) == nanosm::float_bits(3.75f), "deferred FADD lost its operands");
    std::vector<std::pair<uint64_t, uint64_t>> captures;
    for (const auto& event : machine.trace().events()) {
        if (event.name == "RF read" && (event.id == 0 || event.id == 3))
            captures.emplace_back(event.start_cycle, event.id);
    }
    require(captures == std::vector<std::pair<uint64_t, uint64_t>>{{issue + 1, 0}, {issue + 2, 3}},
        "FADD sources were not captured in their reserved cycles");
    const auto mov_issue = issued_at(machine, 24).at(0);
    require(mov_issue == issue + 1, "zero-source MOV could not follow pipelined FADD");
    check_read_capacity(machine, profile);

    nanosm::Launch predicate_launch;
    predicate_launch.active_warps = 1;
    nanosm::Machine predicate(profile, compile("setp.eq.u32 p0, r0, r0\nhalt", profile),
        std::move(predicate_launch), {.timeline = true, .debug = true});
    predicate.run();
    require(predicate.cycle() == 4, "HALT dropped a pending predicate-only read pipeline");
    uint32_t duplicate_reads = 0;
    for (const auto& event : predicate.trace().events()) {
        if (event.name == "RF read") {
            require(event.start_cycle == 2, "duplicate operand captured before its fixed slot");
            ++duplicate_reads;
        }
    }
    require(duplicate_reads == 1, "duplicate operand consumed multiple RF reads");
}

void test_rf_deferred_war_and_branch() {
    auto profile = canonical_profile();
    profile.rf_arithmetic_read_cycles = 4;
    const auto source = "add r2, r0, r3\nmov r3, 7\nhalt";
    auto program = nanosm::compile_qtx(source, profile);
    require(program.instructions[0].control.stall == 4, "qtxc did not protect deferred WAR");
    for (const uint8_t bad_stall : {uint8_t{0}, uint8_t{3}}) {
        auto invalid = program;
        invalid.instructions[0].control.stall = bad_stall;
        nanosm::Launch launch;
        launch.active_warps = 1;
        nanosm::Machine machine(profile, nanosm::assemble_asm(nanosm::emit_asm(invalid), profile),
            std::move(launch));
        try {
            machine.run();
            throw std::runtime_error("missing WAR stall was silently repaired");
        } catch (const nanosm::SimError& error) {
            require(std::string(error.what()).find("RF WAR hazard") != std::string::npos,
                "invalid WAR assembly produced the wrong diagnostic");
        }
    }
    nanosm::Launch launch;
    launch.active_warps = 1;
    nanosm::Machine valid(profile, compile(source, profile), std::move(launch));
    valid.run();
    require(valid.read_reg(0, 0, 2) == 0 && valid.read_reg(0, 0, 3) == 7,
        "WAR-correct schedule changed the captured value");

    nanosm::Launch loop_launch;
    loop_launch.active_warps = 4;
    nanosm::Machine loop(profile, compile(R"(
        mov r0, 0
        mov r3, 9
    loop:
        addi r0, r0, 1
        setp.lt.u32 p0, r0, r3
        @p0 bra.uni loop
        halt
    )", profile), std::move(loop_launch), {.timeline = true, .debug = true});
    loop.run();
    for (uint32_t warp = 0; warp < 4; ++warp)
        require(loop.read_reg(warp, 0, 0) == 9, "branch did not drain fixed read/predicate state");
    check_read_capacity(loop, profile);
}

void test_rf_reservation_retries() {
    auto profile = canonical_profile();
    profile.rf_arithmetic_read_cycles = 3;
    nanosm::Launch launch;
    launch.active_warps = 1;
    nanosm::Machine writes(profile, compile(R"(
        fadd r1, r0, r0
        mov r2, 0
        mov r5, 0
        addi r4, r6, 1
        lds r8, [r6]
        halt
    )", profile), std::move(launch), {.timeline = true, .debug = true});
    writes.run();
    require(issued_at(writes, 24) == std::vector<uint64_t>{4},
        "write conflict did not reject exactly one arithmetic admission");
    require(issued_at(writes, 32) == std::vector<uint64_t>{5},
        "failed write admission leaked a future RF read reservation");
    require(writes.read_reg(0, 0, 4) == 1, "write retry lost arithmetic result");
    check_read_capacity(writes, profile);

    nanosm::Launch read_launch;
    read_launch.active_warps = 1;
    nanosm::Machine reads(profile, compile(R"(
        add r2, r0, r3
        lds r4, [r6]
        halt
    )", profile), std::move(read_launch), {.timeline = true, .debug = true});
    reads.run();
    require(issued_at(reads, 8) == std::vector<uint64_t>{3},
        "issue-captured LDS did not contend with both deferred arithmetic reads");
    require(reads.trace().summary().shmem_requests == 1, "RF retries duplicated async admission");
    check_read_capacity(reads, profile);

    nanosm::Launch arithmetic_launch;
    arithmetic_launch.active_warps = 1;
    nanosm::Machine arithmetic(profile, compile(R"(
        add r2, r0, r3
        add r5, r6, r9
        lds r4, [r12]
        halt
    )", profile), std::move(arithmetic_launch), {.timeline = true, .debug = true});
    arithmetic.run(1000);
    require(issued_at(arithmetic, 8) == std::vector<uint64_t>{2},
        "arithmetic read collision did not defer acceptance by exactly one cycle");
    require(issued_at(arithmetic, 16) == std::vector<uint64_t>{5},
        "read reservation retry changed the promised fixed window or leaked slots");
    check_read_capacity(arithmetic, profile);
}

void test_rf_seeded_competing_warps() {
    for (const uint32_t banks : {2u, 3u}) {
        for (const uint32_t window : {2u, 3u, 4u}) {
            for (const uint32_t warps : {1u, 4u}) {
                auto profile = canonical_profile();
                profile.rf_banks = banks;
                profile.rf_arithmetic_read_cycles = window;
                std::string source;
                std::array<uint32_t, 12> expected{};
                for (uint32_t r = 0; r < expected.size(); ++r) {
                    expected[r] = r + 1;
                    source += "mov r" + std::to_string(r) + ", " + std::to_string(r + 1) + "\n";
                }
                uint32_t seed = banks * 100 + window * 10 + warps;
                const auto random = [&]() { seed = seed * 1664525u + 1013904223u; return seed; };
                for (uint32_t i = 0; i < 400; ++i) {
                    const auto dst = random() % expected.size();
                    const auto a = random() % expected.size();
                    const auto b = i % 3 == 0 ? a : random() % expected.size();
                    if (i % 5 == 0) {
                        const auto immediate = random() % 1024;
                        source += "xori r" + std::to_string(dst) + ", r" + std::to_string(a) +
                            ", " + std::to_string(immediate) + "\n";
                        expected[dst] = expected[a] ^ immediate;
                    } else {
                        source += "add r" + std::to_string(dst) + ", r" + std::to_string(a) +
                            ", r" + std::to_string(b) + "\n";
                        expected[dst] = expected[a] + expected[b];
                    }
                }
                source += "halt\n";
                nanosm::Launch launch;
                launch.active_warps = warps;
                nanosm::Machine machine(profile, compile(source, profile), std::move(launch),
                    {.timeline = true, .debug = true});
                machine.run(50000);
                for (uint32_t warp = 0; warp < warps; ++warp)
                    for (uint32_t lane = 0; lane < profile.lanes; ++lane)
                        for (uint32_t r = 0; r < expected.size(); ++r)
                            require(machine.read_reg(warp, lane, r) == expected[r],
                                "seeded RF contention changed the sequential result");
                check_read_capacity(machine, profile);
            }
        }
    }
}

void test_tmem_profile_metadata_and_round_trip() {
    auto profile = canonical_profile();
    auto duplex = profile;
    duplex.tmem_simt_read_write_duplex = true;
    require(nanosm::device_profile_fingerprint(profile) != nanosm::device_profile_fingerprint(duplex),
        "tmem transfer policy was omitted from the profile fingerprint");

    const std::string source = R"(
        .acc TILE, 0, 0, 64, 64
        mov r0, 2044
        shli r0, r0, 20
        mov r1, 1
        mov r2, 2
        mov r3, 3
        mov r4, 4
        mov r5, 5
        mov r6, 6
        mov r7, 7
        mov r10, 10
        mov r11, 11
        sread r30, WARP_ID
        shli r30, r30, 4
        mov r31, 0
        acc.st.v8 TILE, r30, r0
        add r20, r10, r11
        acc.ld.v8 r40, TILE, r30
        halt
    )";
    const auto scheduled = nanosm::compile_qtx(source, profile);
    require(scheduled.accumulators.size() == 1 && scheduled.accumulators[0].cols == 64,
        "qtx lost accumulator allocation metadata");
    require(scheduled.instructions[15].opcode == nanosm::Opcode::add,
        "directed tmem test instruction index changed unexpectedly");
    require(scheduled.instructions[15].control.wait_mask == 0,
        "independent arithmetic was unnecessarily serialized behind acc.st");
    const auto protected_store = nanosm::compile_qtx(R"(
        .acc A, 0, 0, 64, 64
        mov r0, 1
        mov r8, 0
        mov r9, 0
        acc.st A, r8, r0
        mov r0, 2
        halt
    )", profile);
    const auto store_bar = protected_store.instructions[3].control.set_depbar;
    require((protected_store.instructions[4].control.wait_mask & (1u << store_bar)) != 0,
        "qtxc did not protect the async acc.st source lifetime");
    const auto assembly = nanosm::emit_asm(scheduled);
    const auto kernel = nanosm::assemble_asm(assembly, profile);
    const auto loaded = nanosm::load_hex(nanosm::emit_hex(kernel));
    require(loaded.accumulators.size() == 1 && loaded.accumulators[0].name == "TILE",
        "hex lost accumulator allocation metadata");

    nanosm::Launch launch;
    launch.active_warps = 4;
    nanosm::Machine machine(profile, loaded, std::move(launch), {.timeline = true, .debug = true});
    machine.run(10000);
    require(machine.trace().summary().tmem_store_beats == 32 &&
            machine.trace().summary().tmem_load_beats == 32,
        "tmem trace did not expose every vector beat");
    for (uint32_t warp = 0; warp < 4; ++warp) {
        for (uint32_t lane = 0; lane < profile.lanes; ++lane) {
            require(machine.read_reg(warp, lane, 40) == 0x7fc00000u,
                "tmem transfer did not preserve raw NaN payload bits");
            for (uint32_t j = 1; j < 8; ++j) {
                require(machine.read_reg(warp, lane, 40 + j) == j,
                    "RF -> tmem -> RF vector round trip changed bits");
            }
            require(machine.read_reg(warp, lane, 20) == 21,
                "concurrent arithmetic made no forward progress");
        }
    }
}

void test_tmem_transfer_ordering_and_store_lifetime() {
    const auto& profile = canonical_profile();
    const std::string source = R"(
        .acc A, 0, 0, 16, 16
        .acc B, 1, 0, 16, 16
        mov r0, 17
        mov r1, 34
        mov r8, 0
        mov r9, 0
        acc.st.v8 A, r8, r0
        acc.st.v8 B, r8, r0
        mov r0, 51
        acc.ld r2, A, r8
        acc.ld r3, B, r8
        halt
    )";
    const auto scheduled = nanosm::compile_qtx(source, profile);
    const auto& first_store = scheduled.instructions[4];
    const auto& second_store = scheduled.instructions[5];
    const auto& overwrite = scheduled.instructions[6];
    const auto& first_load = scheduled.instructions[7];
    require(first_store.control.set_depbar < 6 &&
            second_store.control.set_depbar == first_store.control.set_depbar,
        "concurrent accumulator stores did not share a counted completion depbar");
    require((overwrite.control.wait_mask & (1u << first_store.control.set_depbar)) != 0,
        "overwrite did not wait for every older accumulator-store reader");
    require(first_load.control.wait_mask == 0,
        "a wait already paid by the overwrite was redundantly retained on the load");

    const auto ordered = nanosm::compile_qtx(R"(
        .acc A, 0, 0, 16, 16
        mov r0, 17
        mov r8, 0
        mov r9, 0
        acc.st A, r8, r0
        acc.ld r2, A, r8
        halt
    )", profile);
    nanosm::Launch ordered_launch;
    ordered_launch.active_warps = 1;
    nanosm::Machine ordered_machine(
        profile, nanosm::assemble_asm(nanosm::emit_asm(ordered), profile),
        std::move(ordered_launch), {.timeline = true, .debug = true});
    ordered_machine.run(1000);
    require(ordered_machine.read_reg(0, 0, 2) == 17,
        "same-bank structural ownership did not order an accumulator load after its store");
    uint64_t store_cycle = 0;
    uint64_t load_cycle = 0;
    for (const auto& event : ordered_machine.trace().events()) {
        if (event.name == "acc.st beat") store_cycle = event.start_cycle;
        if (event.name == "acc.ld beat") load_cycle = event.start_cycle;
    }
    require(store_cycle < load_cycle,
        "same-bank accumulator load began before the older store's final tmem write");

    nanosm::Launch launch;
    launch.active_warps = 1;
    nanosm::Machine machine(
        profile, nanosm::assemble_asm(nanosm::emit_asm(scheduled), profile),
        std::move(launch), {.timeline = true, .debug = true});
    machine.run(1000);
    for (uint32_t lane = 0; lane < profile.lanes; ++lane) {
        require(machine.read_reg(0, lane, 2) == 17 && machine.read_reg(0, lane, 3) == 17,
            "a younger overwrite changed a deferred accumulator-store source");
        require(machine.read_reg(0, lane, 0) == 51,
            "the source overwrite did not execute after both stores completed");
    }
    require(machine.trace().summary().tmem_store_beats == 16 &&
            machine.trace().summary().tmem_load_beats == 2,
        "scalar transfer completion did not account for each final tmem/RF beat");

    // Hand-written assembly is allowed to omit the compiler's wait, but the
    // simulator must diagnose the architectural WAR violation rather than
    // silently scoreboard or capture the source early.
    auto unsafe = nanosm::assemble_asm(nanosm::emit_asm(scheduled), profile);
    auto unsafe_overwrite = nanosm::decode(unsafe.words[6]);
    unsafe_overwrite.control.wait_mask = 0;
    unsafe.words[6] = nanosm::encode(unsafe_overwrite);
    nanosm::Launch unsafe_launch;
    unsafe_launch.active_warps = 1;
    try {
        nanosm::Machine unsafe_machine(
            profile, std::move(unsafe), std::move(unsafe_launch), {.timeline = true, .debug = true});
        unsafe_machine.run(1000);
        throw std::runtime_error("missing accumulator-store WAR wait did not panic");
    } catch (const nanosm::SimError& error) {
        require(std::string(error.what()).find("RF WAR hazard") != std::string::npos,
            "missing accumulator-store wait produced the wrong diagnostic");
    }
}

void test_mma_raw_window_numerics() {
    using nanosm::mma_dot_f16;
    const std::array<uint16_t, 4> zero{};
    const std::array<uint16_t, 4> one{0x3c00, 0, 0, 0};
    const auto check = [&](std::array<uint16_t, 4> a, std::array<uint16_t, 4> b,
                           uint32_t c, uint32_t expected) {
        require(mma_dot_f16(a, b, c) == expected, "MMA raw-window directed vector mismatch");
    };
    // Cancellation within a fusion group retains the unit that serial FMA loses.
    check({0x6c00, 0x3c00, 0xec00, 0}, {0x6c00, 0x3c00, 0x6c00, 0}, 0, 0x3f800000);
    check({0x6c00, 0xec00, 0x3c00, 0}, {0x6c00, 0x6c00, 0x3c00, 0}, 0, 0x3f800000);
    // But alignment can discard a small C before the large products cancel.
    check({0x6c00, 0xec00, 0, 0}, {0x6c00, 0x6c00, 0, 0}, 0x3e800000, 0);
    // Final truncation, not RNE: 1 + 1.5 * 2^-24 -> 1, for both signs.
    check({0x3c00, 0x3e00, 0, 0}, {0x3c00, 1, 0, 0}, 0, 0x3f800000);
    check({0xbc00, 0xbe00, 0, 0}, {0x3c00, 1, 0, 0}, 0, 0xbf800000);
    check(zero, zero, 1, 1); // FP32 C subnormals survive when there is no larger term
    check(zero, zero, 0x80000001, 0x80000001);
    check(zero, zero, 0x80000000, 0);
    check({0x7c00, 0, 0, 0}, one, 0, 0x7f800000);
    check({0xfc00, 0, 0, 0}, one, 0, 0xff800000);
    check({0x7c00, 0, 0, 0}, zero, 0, 0x7fffffff);
    check({0x7c00, 0, 0, 0}, one, 0xff800000, 0x7fffffff);
    check(zero, zero, 0x7f800001, 0x7fffffff);
    // All half encodings: identity, including denormals and signaling NaNs.
    for (uint32_t h = 0; h < 65536; ++h) {
        const uint32_t expected = (h & 0x7fff) == 0 ? 0
            : (h & 0x7c00) == 0x7c00 && (h & 0x3ff) != 0 ? 0x7fffffff
            : nanosm::half_to_float_bits(static_cast<uint16_t>(h));
        check({static_cast<uint16_t>(h), 0, 0, 0}, one, 0, expected);
    }
    // Frozen raw B200 FP16 K=16 vectors from SF Tensor validation.py at
    // d96b82180ac9041c09295f3a6a3c2c4e592ddce6 (seed 103). These validate
    // the arithmetic helper, not the emulator's smaller fusion boundary.
    const std::array<uint16_t, 16> a0{0xfaa3,0xb6dc,0x74cf,0xef28,0xb806,0xb24c,0xbc21,0xd179,
        0x3703,0xb067,0x13fa,0xe990,0x9a4d,0xbd38,0x1f36,0xa782};
    const std::array<uint16_t, 16> b0{0x61d9,0x760f,0xe3c3,0x568f,0xdf45,0xacdf,0x1e62,0x7d36,
        0x150d,0x21d3,0x38f4,0x954f,0xd3fd,0xf912,0x0654,0x0b47};
    const std::array<uint16_t, 16> a1{0xf638,0x1b73,0xb103,0x20ac,0x9b70,0x85ff,0x4b2e,0x2ef5,
        0xbef1,0x68d1,0xdc66,0xbbb2,0x94ac,0xb916,0xac3a,0x6a3f};
    const std::array<uint16_t, 16> b1{0x3a00,0xc940,0xa841,0x928f,0xc577,0xd3b3,0x9e19,0x3fb6,
        0x1608,0x5e7e,0x33db,0x9490,0x37f2,0xf226,0xb17e,0x9e4b};
    require(mma_dot_f16(a0, b0, 0x39a59521) == 0x7fffffff, "B200 NaN vector mismatch");
    require(mma_dot_f16(a1, b1, 0x82890b9b) == 0x49776bba, "B200 finite vector mismatch");

    // Compact offline regression against 20,000 pinned upstream oracle words.
    // xorshift32 order: four A, four B, C; FNV1a over little-endian result bytes.
    uint32_t seed = 0x12345678;
    const auto next = [&]() { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return seed; };
    uint64_t digest = 14695981039346656037ull;
    for (uint32_t i = 0; i < 20000; ++i) {
        std::array<uint16_t, 4> a, b;
        for (auto& h : a) h = static_cast<uint16_t>(next());
        for (auto& h : b) h = static_cast<uint16_t>(next());
        const uint32_t result = mma_dot_f16(a, b, next());
        for (unsigned shift = 0; shift < 32; shift += 8)
            digest = (digest ^ ((result >> shift) & 255)) * 1099511628211ull;
    }
    require(digest == 0xbc4eca1d7a740939ull, "MMA raw-window upstream corpus mismatch");

    std::ifstream input("profiles/nanosm-v0.json");
    const auto document = nlohmann::json::parse(input);
    require(document.at("mma").at("numeric_model") == nanosm::mma_numeric_model,
        "canonical profile does not name the implemented MMA arithmetic");
    // Old or invented arithmetic must not silently acquire the new semantics.
    for (const bool missing : {false, true}) {
        auto changed = document;
        if (missing) changed["mma"].erase("numeric_model");
        else changed["mma"]["numeric_model"] = "serial_fma_fp32_reference";
        bool rejected = false;
        try { (void)nanosm::parse_device_profile(changed.dump()); }
        catch (const nanosm::ProfileError&) { rejected = true; }
        require(rejected, "missing/unsupported MMA arithmetic accepted by profile parser");
    }
}

void test_mma_fusion_boundaries() {
    // Same eight products, different physical fusion extent. Four-wide must
    // round the first group to FP32 before cancellation in the second group.
    for (const uint32_t core_k : {4u, 8u}) {
        auto profile = canonical_profile();
        profile.mma_xor_layout = false;
        profile.mma_core_k = core_k;
        const auto scheduled = nanosm::compile_qtx(
            ".acc C, 0, 0, 64, 64\nmov r0, 0\nmov r1, 1024\nmov r4, 4096\nmov r8, 0\n"
            "mbar.init [r4], 1\nmma.async.init C, [r0], [r1]\nmma.commit [r4]\n"
            "wait:\nmbar.try_wait p0, [r4], 0\n@!p0 bra.uni wait\nacc.ld r10, C, r8\nhalt\n", profile);
        const auto kernel = nanosm::assemble_asm(nanosm::emit_asm(scheduled), profile);
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.shmem_image, 8192);
        const std::array<uint16_t, 8> a{0x6c00, 0x3c00, 0, 0, 0xec00, 0, 0, 0};
        const std::array<uint16_t, 8> b{0x6c00, 0x3c00, 0, 0, 0x6c00, 0, 0, 0};
        for (uint32_t row = 0; row < 64; ++row)
            for (uint32_t k = 0; k < 8; ++k) write_u16(shmem, 2 * (row * 8 + k), a[k]);
        for (uint32_t k = 0; k < 8; ++k)
            for (uint32_t col = 0; col < 64; ++col) write_u16(shmem, 1024 + 2 * (k * 64 + col), b[k]);
        nanosm::Machine machine(profile, kernel, std::move(launch));
        machine.run(20000);
        for (uint32_t lane = 0; lane < profile.lanes; ++lane)
            require(machine.read_reg(0, lane, 10) == (core_k == 4 ? 0u : 0x3f800000u),
                "MMA did not round at the physical core_k boundary");
    }
}

void test_autonomous_mma_gemm_and_commit() {
    auto profile = canonical_profile();
    auto dense = profile;
    dense.mma_xor_layout = false;
    require(nanosm::device_profile_fingerprint(profile) != nanosm::device_profile_fingerprint(dense),
        "MMA operand layout was omitted from the profile fingerprint");

    std::ostringstream qtx;
    qtx << ".acc C, 0, 0, 64, 64\n"
        << "mov r0, 0\nmov r1, 1024\nmov r4, 4096\nmov r8, 0\nmov r9, 0\n"
        << "mbar.init [r4], 1\n"
        << "mma.async.init C, [r0], [r1]\n";
    for (uint32_t chunk = 1; chunk < 64; ++chunk)
        qtx << "mma.async C, [r0], [r1]\n";
    // The GEMM proof passes because of the protocol: the commit mbar is
    // explicitly observed before any transfer touches C (cases 01/13).
    qtx << "mma.commit [r4]\n"
        << "wait_commit:\n"
        << "mbar.try_wait p0, [r4], 0\n"
        << "@!p0 bra.uni wait_commit\n"
        << "acc.ld r10, C, r8\nhalt\n";
    const auto scheduled = nanosm::compile_qtx(qtx.str(), profile);
    require(scheduled.instructions[6].control.set_depbar < 6 &&
            scheduled.instructions[7].control.set_depbar < 6,
        "qtxc did not allocate MMA operand-consumed depbars");
    const auto kernel = nanosm::assemble_asm(nanosm::emit_asm(scheduled), profile);
    require(nanosm::decode(kernel.words[6]).opcode == nanosm::Opcode::mma_async_init &&
            nanosm::decode(kernel.words[70]).opcode == nanosm::Opcode::mma_commit,
        "MMA opcodes did not survive assembly encoding");

    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.shmem_image, 8192);
    auto a_address = [](uint32_t m, uint32_t k) {
        return 4 * (4 * m + ((k >> 1) ^ ((m >> 2) & 1))) + 2 * (k & 1);
    };
    auto b_address = [](uint32_t k, uint32_t n) {
        return 1024 + 4 * (4 * n + ((k >> 1) ^ ((n >> 2) & 1) ^ 2)) + 2 * (k & 1);
    };
    for (uint32_t m = 0; m < 64; ++m)
        for (uint32_t k = 0; k < 8; ++k) write_u16(shmem, a_address(m, k), 0x3c00); // 1
    for (uint32_t k = 0; k < 8; ++k)
        for (uint32_t n = 0; n < 64; ++n) write_u16(shmem, b_address(k, n), 0x4000); // 2

    nanosm::Machine machine(profile, kernel, std::move(launch), {.timeline = true, .debug = true});
    machine.run(50000);
    for (uint32_t lane = 0; lane < profile.lanes; ++lane)
        require(machine.read_reg(0, lane, 10) == nanosm::float_bits(1024.0f),
            "64-command 64x512 MMA did not produce the ordered FP32 GEMM result");
    require(machine.read_mbar_state(4096).phase,
        "mma.commit did not publish the result-ready mbar after final tmem writes");
    const auto& summary = machine.trace().summary();
    require(summary.mma_commands == 64 && summary.mma_input_steps == 32768 &&
            summary.mma_result_steps == 32768 && summary.mma_shmem_rounds == 32768 &&
            summary.mma_recurrence_stalls == 0,
        "canonical MMA trace did not prove one-round pipelined physical steps");
    uint64_t second_admission = std::numeric_limits<uint64_t>::max();
    uint64_t first_last_write = 0;
    for (const auto& event : machine.trace().events()) {
        if (event.category != nanosm::TraceCategory::mma) continue;
        if (event.name == "mma.async" && event.id == 2) second_admission = event.start_cycle;
        if (event.name == "MMA tmem write" && event.id == 1)
            first_last_write = std::max(first_last_write, event.start_cycle);
    }
    require(second_admission < first_last_write,
        "adjacent MMA commands drained at the command boundary instead of overlapping");
}

void test_worker_acc_store_mbar_handoff() {
    const auto& profile = canonical_profile();
    const auto scheduled = nanosm::compile_qtx(R"(
        .acc C, 0, 0, 64, 64
        sread r0, THREAD_ID
        sread r8, WARP_ID
        shli r8, r8, 4
        mov r9, 0
        mov r12, 4096
        acc.st.v8 C, r8, r0
        mbar.arrive [r12]
        halt
    )", profile);
    const auto& store = scheduled.instructions[5];
    const auto& arrive = scheduled.instructions[6];
    require((arrive.control.wait_mask & (1u << store.control.set_depbar)) != 0,
        "worker mbar.arrive was not gated by its accumulator-store completion");
    nanosm::Launch launch;
    launch.active_warps = 2;
    auto& shmem = image_bytes(launch.shmem_image, 8192);
    nanosm::MbarState initial;
    initial.valid = true;
    initial.expected = 2;
    initial.remaining = 2;
    write_u64(shmem, 4096, nanosm::encode_mbar_state(initial));
    nanosm::Machine machine(profile,
        nanosm::assemble_asm(nanosm::emit_asm(scheduled), profile), std::move(launch));
    machine.run(2000);
    require(machine.read_mbar_state(4096).phase,
        "shared worker mbar did not receive exactly one post-store arrival per warp");
}

// ---- Per-engine mbar event registers (design.md Mbar; 2026-10-07) ----

using RmwService = std::pair<uint64_t, std::string>; // start cycle, trace detail

// The RMW engine's service order: every "mbar RMW" slice, by start cycle.
std::vector<RmwService> mbar_rmw_services(const nanosm::Machine& machine) {
    std::vector<RmwService> services;
    for (const auto& event : machine.trace().events())
        if (event.name == "mbar RMW") services.emplace_back(event.start_cycle, event.detail);
    std::sort(services.begin(), services.end());
    return services;
}

void write_mbar(std::vector<uint8_t>& shmem, uint32_t address, uint16_t remaining,
    uint32_t tx_bytes) {
    nanosm::MbarState state;
    state.valid = true;
    state.expected = 1;
    state.remaining = remaining;
    state.tx_bytes = tx_bytes;
    write_u64(shmem, address, nanosm::encode_mbar_state(state));
}

// A TMA complete_tx and an MMA commit become pending in the same cycle r, while
// the RMW engine is still busy (the write stage of an earlier TMA event). Both
// are served, alternating TMA, MMA, TMA, and the mma.async queued behind the
// commit is accepted in cycle r and starts fetching in r + 1, exactly as in a
// run with no TMA event at all: the commit presents into the MMA's own register.
// (The alternation here follows the last-served bit: T1 was served last, so
// the tie at r + 1 goes to the MMA.)
// Before 2026-10-07 the two engines shared ONE event slot. In cycle r the
// commit found it holding the TMA event and stayed at the head of the MMA
// queue: the order was TMA, TMA, MMA, and the mma.async behind it (accepted
// at r into the entry the retired init command freed) could not start
// fetching until r + 2, one cycle late -- one cycle for every cycle the TMA
// event sat in the slot. (Measured on 29e320f: fetch 525 instead of 524.)
void test_mbar_engine_tma_event_does_not_stall_mma_queue() {
    constexpr uint32_t commit_mbar = 4096;
    constexpr uint32_t tx_mbar = 4104;
    const auto kernel = compile(R"(
        .acc C, 0, 0, 64, 64
        mov r0, 0
        mov r1, 1024
        mov r4, 4096
        mma.async.init C, [r0], [r1]
        mma.commit [r4]
        mma.async C, [r0], [r1]
        halt
    )");
    const auto make_launch = [&] {
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.shmem_image, 8192);
        write_mbar(shmem, commit_mbar, 1, 0);
        write_mbar(shmem, tx_mbar, 0, 128); // arrived; waits for 2 x 64 bytes
        return launch;
    };
    // The queued mma.async is command 3 (1 = the init command, 2 = the commit).
    const auto queued_accept_and_fetch = [](const nanosm::Machine& machine) {
        uint64_t accept = 0;
        uint64_t fetch = std::numeric_limits<uint64_t>::max();
        for (const auto& event : machine.trace().events()) {
            if (event.category != nanosm::TraceCategory::mma || event.id != 3) continue;
            if (event.name == "mma.async") accept = event.start_cycle;
            if (event.name == "MMA input step") fetch = std::min(fetch, event.start_cycle);
        }
        return std::pair{accept, fetch};
    };

    // Reference, no TMA event: the commit presents in cycle r, the cycle its
    // predecessor retires, and frees the queue slot the mma.async waits for.
    nanosm::Machine reference(canonical_profile(), kernel, make_launch(), {.timeline = true});
    reference.run(5000);
    const auto [r, fetch] = queued_accept_and_fetch(reference);
    require(r > 2 && fetch == r + 1, "reference MMA queue timing changed; re-derive the test");
    require(mbar_rmw_services(reference) ==
            std::vector<RmwService>{{r + 1, "arrive (mma.commit), address=4096"}},
        "reference commit was not served at r + 1");

    nanosm::Machine machine(canonical_profile(), kernel, make_launch(), {.timeline = true});
    while (machine.cycle() + 1 < r) machine.tick();
    machine.complete_mbar_tx(tx_mbar, 64, 0); // T1, pending in r - 1: the engine is free
    machine.tick();
    require(!machine.tma_mbar_event_held(), "T1 did not take the free RMW engine");
    machine.complete_mbar_tx(tx_mbar, 64, 0); // T2, pending in r: the engine is in T1's write
    require(!machine.mma_mbar_event_held(), "the commit presented before cycle r");
    machine.tick(); // cycle r
    require(machine.tma_mbar_event_held() && machine.mma_mbar_event_held(),
        "the TMA event and the commit were not pending together in cycle r");
    machine.run(5000);

    require(queued_accept_and_fetch(machine) == std::pair{r, fetch},
        "a held TMA event stalled the MMA queue behind the commit");
    require(mbar_rmw_services(machine) == std::vector<RmwService>{
                {r - 1, "complete_tx (tma), address=4104"},
                {r + 1, "arrive (mma.commit), address=4096"},
                {r + 3, "complete_tx (tma), address=4104"}},
        "engine events were not served once each, alternating TMA, MMA, TMA");
    require(machine.trace().summary().mbar_engine_waits_on_other == 2,
        "T2 should wait exactly the two cycles of the commit's RMW");
    require(machine.read_mbar_state(commit_mbar).phase, "the commit's arrive was lost");
    const auto tx = machine.read_mbar_state(tx_mbar);
    require(tx.phase && tx.tx_bytes == 0, "a TMA completion was lost");
}

// Both engines hold an event at the first free point: the TMA goes first (the
// last-served bit starts at MMA). The TMA presents again before the next free
// point, so both are pending again: the MMA goes first. Then the second
// commit and that TMA event tie, and the TMA goes. complete_mbar_tx's
// occupied PANIC belongs to the TMA's register alone.
void test_mbar_engine_events_alternate_tma_first() {
    constexpr uint32_t commit0 = 4096;
    constexpr uint32_t tx_mbar = 4104;
    constexpr uint32_t commit1 = 4112;
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.shmem_image, 8192);
    write_mbar(shmem, commit0, 1, 0);
    write_mbar(shmem, commit1, 1, 0);
    write_mbar(shmem, tx_mbar, 0, 128);
    nanosm::Machine machine(canonical_profile(), compile(R"(
        mov r4, 4096
        mov r6, 4112
        mma.commit [r4]
        mma.commit [r6]
        halt
    )"), std::move(launch), {.timeline = true});

    while (!machine.mma_mbar_event_held()) tick_capped(machine); // the first commit presented
    require(!machine.tma_mbar_event_held() && mbar_rmw_services(machine).empty(),
        "an engine event was served before the first commit");
    const auto first = machine.cycle(); // the first free point with an engine event
    machine.complete_mbar_tx(tx_mbar, 64, 0); // accepted although the MMA register is full
    try {
        machine.complete_mbar_tx(tx_mbar, 64, 0);
        throw std::runtime_error("a second event entered the full TMA register");
    } catch (const nanosm::SimError& error) {
        require(std::string(error.what()).find("TMA mbar event register is occupied") !=
                std::string::npos, "wrong full-register PANIC");
    }
    machine.tick(); // first: both held
    require(!machine.tma_mbar_event_held() && machine.mma_mbar_event_held(),
        "the first tie did not serve the TMA");
    machine.tick(); // the TMA event's write stage; the commit still waits
    machine.complete_mbar_tx(tx_mbar, 64, 0); // both pending again at first + 2
    machine.run(1000);

    require(mbar_rmw_services(machine) == std::vector<RmwService>{
                {first, "complete_tx (tma), address=4104"},
                {first + 2, "arrive (mma.commit), address=4096"},
                {first + 4, "complete_tx (tma), address=4104"},
                {first + 6, "arrive (mma.commit), address=4112"}},
        "tied engine events did not alternate TMA, MMA, TMA, MMA");
    // Each of the three ties holds the loser through both cycles of the
    // winner's RMW.
    require(machine.trace().summary().mbar_engine_waits_on_other == 6,
        "engine_waits_on_other does not count 2 cycles per tie");
    require(machine.read_mbar_state(commit0).phase && machine.read_mbar_state(commit1).phase,
        "a commit's arrive was lost");
    const auto tx = machine.read_mbar_state(tx_mbar);
    require(tx.phase && tx.tx_bytes == 0, "a TMA completion was lost");
}

// The same-cycle case with a real TMA load. Filler movs before tma.async set
// its issue delay; the sweep moves its completion across cycle r (the commit's
// presentation) in two-cycle steps (DRAM clock), and the test finds the delay
// at which both present in cycle r. Each event is visible to the RMW engine
// from the next cycle, so at r + 1 they tie: the TMA goes first (the
// last-served bit starts at MMA) and the commit follows at r + 3, after
// waiting the two cycles of the TMA's RMW. At every delay the queued
// mma.async is still accepted at r (the MMA queue never waits on the TMA) and
// the TMA command ends the cycle after its event is claimed.
// Before 2026-10-08 (stepping discipline) the TMA's event could be claimed in
// the cycle it was presented, an artifact of tick()'s call order, while the
// MMA's waited a cycle: at the aligned delay the TMA was served at r and the
// commit at r + 2 (one cycle of waiting). Before 2026-10-07, at the aligned
// delay the commit filled the shared slot right behind the TMA's claimed
// event, and the TMA command could not end (nor free its queue entry) until
// that slot emptied (measured on 29e320f: 529 instead of 527).
void test_mbar_engine_real_tma_and_commit_same_cycle() {
    constexpr uint32_t commit_mbar = 4096;
    constexpr uint32_t tx_mbar = 4104;
    constexpr uint32_t tile = 2048;
    constexpr uint32_t descriptor_address = 6144;
    struct Observed {
        uint64_t accept = 0;  // the queued mma.async (command 3)
        uint64_t tma_end = 0; // end of the "tma command" slice
        std::vector<RmwService> services;
        uint64_t waits = 0;
    };
    const auto run = [&](std::optional<uint32_t> delay) {
        std::ostringstream qtx;
        qtx << ".acc C, 0, 0, 64, 64\nmov r0, 0\nmov r1, 1024\nmov r4, 4096\n"
            << "mov r8, 0\nmov r9, 0\nmov r10, 0\n"
            << "mma.async.init C, [r0], [r1]\nmma.commit [r4]\n";
        if (delay.has_value()) {
            for (uint32_t i = 0; i < *delay; ++i) qtx << "mov r" << 12 + i % 8 << ", " << i << '\n';
            qtx << "tma.async [" << descriptor_address << "], r8\n";
        }
        qtx << "mma.async C, [r0], [r1]\nhalt\n";
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.shmem_image, 8192);
        auto& gmem = image_bytes(launch.gmem_image, 64);
        for (uint32_t i = 0; i < 16; ++i) write_u32(gmem, 4 * i, 0x100u + i);
        write_mbar(shmem, commit_mbar, 1, 0);
        write_mbar(shmem, tx_mbar, 0, 64);
        nanosm::TmaDescriptor descriptor;
        descriptor.dimensions = {16, 1, 1};
        descriptor.box = {16, 1, 1};
        descriptor.shmem_base = tile;
        descriptor.completion_mbar = tx_mbar;
        write_tma_descriptor(shmem, descriptor_address, descriptor);
        nanosm::Machine machine(canonical_profile(), compile(qtx.str()), std::move(launch),
            {.timeline = true});
        machine.run(5000);
        Observed observed;
        for (const auto& event : machine.trace().events()) {
            if (event.category == nanosm::TraceCategory::mma && event.name == "mma.async" &&
                event.id == 3) observed.accept = event.start_cycle;
            if (event.name == "tma command") observed.tma_end = event.start_cycle + event.duration_cycles;
        }
        observed.services = mbar_rmw_services(machine);
        observed.waits = machine.trace().summary().mbar_engine_waits_on_other;
        if (delay.has_value()) {
            require(machine.read_mbar_state(tx_mbar).phase &&
                    machine.read_shmem_u32(tile + 60) == 0x10fu, "the TMA load did not complete");
        }
        return observed;
    };
    const RmwService tma_service{0, "complete_tx (tma), address=4104"};
    const RmwService mma_service{0, "arrive (mma.commit), address=4096"};
    const auto at = [](RmwService service, uint64_t cycle) { service.first = cycle; return service; };

    const auto reference = run(std::nullopt); // no TMA
    const auto r = reference.accept;
    require(reference.services == std::vector<RmwService>{at(mma_service, r + 1)},
        "reference commit was not served at r + 1");
    const auto early = run(0u); // the TMA completes long before r
    require(early.services.size() == 2 && early.services[0].second == tma_service.second &&
            early.services[0].first < r, "a delay-0 TMA did not complete before the commit");
    // The fillers issue one per cycle, so this delay lands the TMA near r.
    const auto estimate = static_cast<uint32_t>(r - early.services[0].first);
    bool aligned = false;
    for (uint32_t delay = estimate - 6; delay <= estimate + 6; ++delay) {
        const auto observed = run(delay);
        const auto where = " (delay " + std::to_string(delay) + ")";
        require(observed.accept == r, "the TMA's completion delayed the MMA queue" + where);
        require(observed.services.size() == 2, "an engine event was lost" + where);
        const auto tma_claim = observed.services[0].second == tma_service.second
            ? observed.services[0].first : observed.services[1].first;
        require(observed.tma_end == tma_claim + 1,
            "the TMA command outlived its claimed event (held by the MMA's?)" + where);
        if (tma_claim != r + 1) continue; // the TMA presented in r too
        aligned = true;
        require(observed.services ==
                std::vector<RmwService>{at(tma_service, r + 1), at(mma_service, r + 3)},
            "same-cycle TMA completion and commit were not served TMA, then MMA" + where);
        require(observed.waits == 2, "the commit should wait two cycles (the TMA's RMW)" + where);
    }
    require(aligned, "no delay made the TMA completion and the commit present in one cycle");
}

void test_mma_dense_profile_recurrence_bubbles() {
    auto profile = canonical_profile();
    profile.mma_m = 16;
    profile.mma_n = 4;
    profile.mma_k = 8;
    profile.mma_core_m = 4;
    profile.mma_core_n = 4;
    profile.mma_core_k = 4;
    profile.mma_xor_layout = false;
    profile.mma_result_latency_cycles = 12;
    nanosm::validate_device_profile(profile);
    auto unsupported = profile;
    unsupported.mma_xor_layout = true;
    try {
        nanosm::validate_device_profile(unsupported);
        throw std::runtime_error("unsupported generalized XOR MMA layout was accepted");
    } catch (const nanosm::ProfileError&) {}
    const auto kernel = compile(R"(
        .acc C, 0, 0, 16, 4
        mov r0, 0
        mov r1, 512
        mov r4, 896
        mbar.init [r4], 1
        mov r8, 0
        mov r9, 0
        mma.async.init C, [r0], [r1]
        mma.commit [r4]
    wait_done:
        mbar.try_wait p0, [r4], 0
        @!p0 bra.uni wait_done
        acc.ld r10, C, r8
        halt
    )", profile);
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.shmem_image, 1024);
    for (uint32_t m = 0; m < 16; ++m)
        for (uint32_t k = 0; k < 8; ++k) write_u16(shmem, 2 * (m * 8 + k), 0x3c00);
    for (uint32_t k = 0; k < 8; ++k)
        for (uint32_t n = 0; n < 4; ++n) write_u16(shmem, 512 + 2 * (k * 4 + n), 0x3c00);
    nanosm::Machine machine(profile, kernel, std::move(launch), {.timeline = true});
    machine.run(1000);
    require(machine.read_reg(0, 0, 10) == nanosm::float_bits(8.0f),
        "dense diagnostic MMA profile produced the wrong result");
    require(machine.trace().summary().mma_recurrence_stalls > 0,
        "small dense MMA profile did not expose its recurrence bubbles");
    require(machine.trace().summary().mma_shmem_rounds >
            machine.trace().summary().mma_input_steps,
        "dense diagnostic footprint did not charge its shmem bank conflicts");
}

nanosm::DeviceProfile dense_one_bank_profile(uint32_t result_latency) {
    auto profile = canonical_profile();
    profile.mma_m = 16;
    profile.mma_n = 4;
    profile.mma_k = 8;
    profile.mma_core_m = 4;
    profile.mma_core_n = 4;
    profile.mma_core_k = 4;
    profile.mma_xor_layout = false;
    profile.mma_result_latency_cycles = result_latency;
    profile.shmem_banks = 1;
    nanosm::validate_device_profile(profile);
    return profile;
}

void test_mma_bare_transfer_race_panics() {
    // The FAIL.asm shape (cases 01/02): three commands, a commit, and an
    // acc.ld with no wait on the commit mbar. Racing accepted MMA work is UB
    // and the simulator diagnoses it instead of returning stale columns.
    const auto profile = dense_one_bank_profile(1);
    const auto kernel = compile(R"(
        .acc C, 0, 0, 16, 4
        mov r4, 14000
        mbar.init [r4], 1
        mov r0, 0
        mov r1, 2048
        mma.async.init C, [r0], [r1]
        mov r0, 4096
        mov r1, 6144
        mma.async C, [r0], [r1]
        mov r0, 8192
        mov r1, 10240
        mma.async C, [r0], [r1]
        mma.commit [r4]
        mov r8, 0
        mov r9, 0
        acc.ld r10, C, r8
        halt
    )", profile);
    nanosm::Launch launch;
    launch.active_warps = 1;
    nanosm::Machine machine(profile, kernel, std::move(launch));
    try {
        machine.run(10000);
        throw std::runtime_error("bare commit -> acc.ld race was not diagnosed");
    } catch (const nanosm::SimError& error) {
        require(std::string(error.what()).find("races accepted MMA work") != std::string::npos,
            "the transfer/MMA race produced the wrong diagnostic");
    }
}

void test_mma_multi_round_fetch_timing() {
    // Case 03/05/06: one bank makes every 4x4x4 step a 16-round fetch. A step
    // fetching at t completes (enters the core) at t+15, its result lands at
    // t+15+latency, the next fetch starts at t+16, and the operand-consumed
    // depbar clears at the LAST PHYSICAL READ of the final step.
    const auto profile = dense_one_bank_profile(12);
    const std::string source = R"(
        .acc C, 0, 0, 16, 4
        mov r0, 0
        mov r1, 512
        mov r4, 896
        mbar.init [r4], 1
        mov r8, 0
        mov r9, 0
        mma.async.init C, [r0], [r1]
        mma.commit [r4]
    wait_done:
        mbar.try_wait p0, [r4], 0
        @!p0 bra.uni wait_done
        acc.ld r10, C, r8
        halt
    )";
    const auto scheduled = nanosm::compile_qtx(source, profile);
    require(scheduled.instructions[6].opcode == nanosm::Opcode::mma_async_init,
        "multi-round timing test instruction index changed unexpectedly");
    const auto bar = scheduled.instructions[6].control.set_depbar;
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.shmem_image, 1024);
    for (uint32_t m = 0; m < 16; ++m)
        for (uint32_t k = 0; k < 8; ++k) write_u16(shmem, 2 * (m * 8 + k), 0x3c00);
    for (uint32_t k = 0; k < 8; ++k)
        for (uint32_t n = 0; n < 4; ++n) write_u16(shmem, 512 + 2 * (k * 4 + n), 0x3c00);
    nanosm::Machine machine(profile,
        nanosm::assemble_asm(nanosm::emit_asm(scheduled), profile),
        std::move(launch), {.timeline = true, .debug = true});
    machine.run(4000);
    std::vector<uint64_t> inputs;
    uint64_t first_result = 0;
    uint64_t depbar_cleared = 0;
    for (const auto& event : machine.trace().events()) {
        if (event.name == "MMA input step") inputs.push_back(event.start_cycle);
        if (event.name == "MMA tmem write" && first_result == 0) first_result = event.start_cycle;
        if (depbar_cleared == 0 && event.name == "depbar sb" + std::to_string(bar) &&
            event.detail == "value=0") {
            depbar_cleared = event.start_cycle;
        }
    }
    require(inputs.size() == 8, "one dense 16x4x8 command is eight core steps");
    for (std::size_t i = 0; i < inputs.size(); ++i)
        require(inputs[i] == inputs[0] + 16 * i,
            "16-round fetches did not serialize the port at one round per cycle");
    require(first_result == inputs[0] + 15 + 12,
        "the result did not launch from fetch completion plus the pipeline latency");
    require(depbar_cleared == inputs[7] + 15,
        "operand-consumed depbar did not clear at the final physical read");
    require(machine.trace().summary().mma_recurrence_stalls == 0,
        "16-round spacing already covers the result latency; no bubbles expected");
    for (uint32_t lane = 0; lane < profile.lanes; ++lane)
        require(machine.read_reg(0, lane, 10) == nanosm::float_bits(8.0f),
            "per-round operand capture changed the numerical result");
}

void test_mma_port_collision_sweep() {
    // Case 08: a port-A write and an MMA port-B round touching the same shmem
    // word in the same cycle is UB and panics. Both streams ascend one word
    // per cycle, but the fetch's step-0 footprint skips words, so word w is
    // read rank(w) rounds in â€” w - rank(w) cycles earlier than an aligned
    // writer reaches it ({0,2,4,6} for the A block). Padding between the sts
    // and the MMA sweeps that offset: some paddings collide on one cycle,
    // others miss. Writing far from the operands never panics at any padding.
    const auto profile = dense_one_bank_profile(4);
    uint32_t panics = 0;
    for (uint32_t base : {0u, 8192u}) {
        uint32_t base_panics = 0;
        for (uint32_t pad = 0; pad <= 8; ++pad) {
            std::ostringstream qtx;
            qtx << ".acc C, 0, 0, 16, 4\n"
                << "mov r0, 0\nmov r1, 512\n"
                << "sread r20, LANE_ID\nshli r20, r20, 2\n"
                << "addi r20, r20, " << base << "\n"
                << "mov r21, 7\n"
                << "sts [r20], r21\n";
            for (uint32_t i = 0; i < pad; ++i) qtx << "mov r22, 1\n";
            qtx << "mma.async.init C, [r0], [r1]\nhalt\n";
            nanosm::Launch launch;
            launch.active_warps = 1;
            nanosm::Machine machine(profile, compile(qtx.str(), profile), std::move(launch));
            try {
                machine.run(4000);
            } catch (const nanosm::SimError& error) {
                require(std::string(error.what()).find("in the same cycle") != std::string::npos,
                    "the port collision produced the wrong diagnostic");
                ++base_panics;
            }
        }
        if (base == 0) panics = base_panics;
        else require(base_panics == 0, "a write far from the operands must never collide");
    }
    require(panics >= 1, "no padding aligned the crossing streams onto one cycle");
    require(panics <= 8, "different-cycle same-word overlap must not be diagnosed");
}

void test_mma_varying_round_service() {
    // Unswizzled dense 64x64x8 on 16 banks has steps of different conflict
    // costs (the 2026-09-04 audit counted four- and five-round steps). Each
    // fetch must occupy the port for exactly its own rounds: the span of all
    // input steps equals the sum of served rounds plus counted stalls.
    auto profile = canonical_profile();
    profile.mma_xor_layout = false;
    nanosm::validate_device_profile(profile);
    const auto kernel = compile(R"(
        .acc C, 0, 0, 64, 64
        mov r0, 0
        mov r1, 2048
        mov r4, 8192
        mbar.init [r4], 1
        mov r8, 0
        mov r9, 0
        mma.async.init C, [r0], [r1]
        mma.commit [r4]
    wait_done:
        mbar.try_wait p0, [r4], 0
        @!p0 bra.uni wait_done
        acc.ld r10, C, r8
        halt
    )", profile);
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.shmem_image, 4096);
    for (uint32_t i = 0; i < 512; ++i) write_u16(shmem, 2 * i, 0x3c00);
    for (uint32_t i = 0; i < 512; ++i) write_u16(shmem, 2048 + 2 * i, 0x3c00);
    nanosm::Machine machine(profile, kernel, std::move(launch), {.timeline = true});
    machine.run(8000);
    std::vector<std::pair<uint64_t, uint64_t>> inputs; // cycle, rounds
    for (const auto& event : machine.trace().events()) {
        if (event.name != "MMA input step") continue;
        inputs.emplace_back(event.start_cycle, std::stoull(event.detail.substr(7)));
    }
    require(inputs.size() == 512, "one canonical-shape command is 512 core steps");
    uint64_t served = 0;
    std::set<uint64_t> distinct;
    for (std::size_t i = 0; i + 1 < inputs.size(); ++i) {
        require(inputs[i + 1].first - inputs[i].first >= inputs[i].second,
            "a fetch began before the previous step's rounds freed the port");
        served += inputs[i].second;
        distinct.insert(inputs[i].second);
    }
    distinct.insert(inputs.back().second);
    require(distinct.size() >= 2, "the dense walk was expected to vary its conflict rounds");
    require(inputs.back().first - inputs.front().first ==
            served + machine.trace().summary().mma_recurrence_stalls,
        "input-step spacing did not equal served rounds plus counted stalls");
    for (uint32_t lane = 0; lane < profile.lanes; ++lane)
        require(machine.read_reg(0, lane, 10) == nanosm::float_bits(8.0f),
            "varying-round service changed the numerical result");
}

// ---- Race sanitizer: shmem range ownership and tmem ownership ----
// (trash/shmem-ownership-sanitizer-2026-10-03.md)

// Runs the machine and returns its PANIC text; a clean run is a test failure.
std::string run_expecting_panic(
    const nanosm::DeviceProfile& profile, const nanosm::Kernel& kernel, nanosm::Launch launch,
    const std::string& what, uint64_t cycles = 20000) {
    nanosm::Machine machine(profile, kernel, std::move(launch));
    try {
        machine.run(cycles);
    } catch (const nanosm::SimError& error) {
        return error.what();
    }
    throw std::runtime_error(what + " did not panic");
}

void require_mentions(
    const std::string& message, std::initializer_list<std::string_view> needles,
    const std::string& what) {
    for (const auto needle : needles) {
        if (message.find(needle) == std::string::npos)
            throw std::runtime_error(what + ": diagnostic lacks '" + std::string(needle) +
                "': " + message);
    }
}

// Rank-1 linear fp32 TMA of `bytes` between shmem `shmem_base` and gmem 0.
nanosm::TmaDescriptor linear_tma(bool store, uint32_t shmem_base, uint32_t bytes, uint32_t mbar) {
    nanosm::TmaDescriptor descriptor;
    descriptor.direction = store ? nanosm::TmaDirection::store : nanosm::TmaDirection::load;
    descriptor.dimensions = {bytes / 4, 1, 1};
    descriptor.box = {bytes / 4, 1, 1};
    descriptor.shmem_base = shmem_base;
    descriptor.completion_mbar = mbar;
    return descriptor;
}

// An mbar already expecting `tx` bytes and no arrivals: one complete_tx flips it.
void write_tx_mbar(std::vector<uint8_t>& shmem, uint32_t address, uint32_t tx) {
    nanosm::MbarState state;
    state.valid = true;
    state.expected = 1;
    state.tx_bytes = tx;
    write_u64(shmem, address, nanosm::encode_mbar_state(state));
}

void write_count_mbar(std::vector<uint8_t>& shmem, uint32_t address, uint16_t count) {
    nanosm::MbarState state;
    state.valid = true;
    state.expected = count;
    state.remaining = count;
    write_u64(shmem, address, nanosm::encode_mbar_state(state));
}

void test_sanitizer_sts_into_tma_store_source() {
    // tma-out break (1): the issuer fires the store before the workers finished
    // staging. The store read-owns its source from accept until its last source
    // read, so the late sts panics whatever the port timing.
    const auto& profile = canonical_profile();
    const auto launch_with_store = [] {
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.shmem_image, 8192);
        image_bytes(launch.gmem_image, 2048);
        for (uint32_t i = 0; i < 256; ++i) write_u32(shmem, 4 * i, 3 * i + 1);
        write_tma_descriptor(shmem, 4096, linear_tma(true, 0, 1024, 4160));
        write_tx_mbar(shmem, 4160, 1024);
        return launch;
    };
    const std::string prologue = R"(
        mov r0, 0
        mov r1, 0
        mov r2, 0
        mov r4, 4160
        tma.async [4096], r0
    )";
    const std::string stage = R"(
        sread r20, LANE_ID
        shli r20, r20, 2
        mov r21, 7
        sts [r20], r21
        halt
    )";
    const auto message = run_expecting_panic(profile, compile(prologue + stage),
        launch_with_store(), "sts into a live TMA store source");
    require_mentions(message, {"shmem race: sts writes 16 words, bytes [0, 64)",
        "tma store #0", "read-owner of its source [0, 1024) until its last source read",
        "wait the TMA completion mbar [4160]", "inst='sts"}, "TMA store source race");

    // Control: waiting the store's completion mbar orders the overwrite.
    nanosm::Machine machine(profile, compile(prologue + R"(
    wait:
        mbar.try_wait p0, [r4], 0
        @!p0 bra.uni wait
    )" + stage), launch_with_store());
    machine.run(20000);
    for (uint32_t i = 0; i < 256; ++i)
        require(machine.read_gmem_u32(4 * i) == 3 * i + 1, "the waited store captured a late sts");
    require(machine.read_shmem_u32(0) == 7, "the waited sts did not execute");
}

void test_sanitizer_sts_into_mma_operand_stage() {
    // tma-out break (2): a write into a stage that a queued or active MMA will
    // still read. Queue depth 4 lets a command wait behind an unfired commit.
    auto profile = canonical_profile();
    profile.mma_queue_commands = 4;
    nanosm::validate_device_profile(profile);
    for (const bool queued : {true, false}) {
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.shmem_image, 8192 + 8);
        write_count_mbar(shmem, 8192, 1);
        const auto kernel = compile(std::string(R"(
            .acc C, 0, 0, 64, 64
            mov r0, 0
            mov r1, 1024
            mov r2, 2048
            mov r3, 3072
            mov r4, 8192
            mma.async.init C, [r0], [r1]
            mma.commit [r4]
            mma.async C, [r2], [r3]
            sread r20, LANE_ID
            shli r20, r20, 2
            addi r20, r20, )") + (queued ? "2048" : "0") + R"(
            mov r21, 0
            sts [r20], r21
            halt
        )", profile);
        const auto message = run_expecting_panic(profile, kernel, std::move(launch),
            queued ? "sts into a queued MMA's stage" : "sts into an active MMA's stage");
        if (queued) {
            require_mentions(message, {"sts writes 16 words, bytes [2048, 2112)",
                "mma.async #3 [queued]", "read-owner of its A operand [2048, 3072)",
                "until its final operand read", "wait the operand-consumed depbar sb",
                "/ commit mbar"}, "queued-MMA stage race");
        } else {
            require_mentions(message, {"sts writes 16 words, bytes [0, 64)",
                "mma.async #1 [fetching]", "read-owner of its A operand [0, 1024)",
                "wait the operand-consumed depbar sb"}, "active-MMA stage race");
        }
    }
}

void test_sanitizer_lds_from_tma_load_destination() {
    const auto& profile = canonical_profile();
    const auto launch_with_load = [] {
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.shmem_image, 8192);
        auto& gmem = image_bytes(launch.gmem_image, 1024);
        for (uint32_t i = 0; i < 256; ++i) write_u32(gmem, 4 * i, 5 * i + 2);
        write_tma_descriptor(shmem, 4096, linear_tma(false, 0, 1024, 4160));
        write_tx_mbar(shmem, 4160, 1024);
        return launch;
    };
    const std::string prologue = R"(
        mov r0, 0
        mov r1, 0
        mov r2, 0
        mov r4, 4160
        sread r20, LANE_ID
        shli r20, r20, 2
        tma.async [4096], r0
    )";
    const auto message = run_expecting_panic(profile, compile(prologue + R"(
        lds r21, [r20]
        halt
    )"), launch_with_load(), "lds from a pending TMA load destination");
    require_mentions(message, {"shmem race: lds reads 16 words, bytes [0, 64)",
        "tma load #0", "write-owner of its destination [0, 1024) until its last shmem write commits",
        "wait the TMA completion mbar [4160]"}, "TMA load destination read race");

    nanosm::Machine machine(profile, compile(prologue + R"(
    wait:
        mbar.try_wait p0, [r4], 0
        @!p0 bra.uni wait
        lds r21, [r20]
        halt
    )"), launch_with_load());
    machine.run(20000);
    for (uint32_t lane = 0; lane < profile.lanes; ++lane)
        require(machine.read_reg(0, lane, 21) == 5 * lane + 2, "the waited lds read the wrong value");
}

void test_sanitizer_mbar_rmw_into_owned_range() {
    const auto& profile = canonical_profile();
    {
        // A SIMT arrive whose mbar sits inside a pending TMA load destination.
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.shmem_image, 8192);
        image_bytes(launch.gmem_image, 1024);
        write_tma_descriptor(shmem, 4096, linear_tma(false, 0, 1024, 4160));
        write_tx_mbar(shmem, 4160, 1024);
        write_count_mbar(shmem, 512, 2);
        const auto message = run_expecting_panic(profile, compile(R"(
            mov r0, 0
            mov r1, 0
            mov r2, 0
            mov r4, 512
            tma.async [4096], r0
            mbar.arrive [r4]
            halt
        )"), std::move(launch), "mbar.arrive inside a TMA load destination");
        require_mentions(message, {"shmem race: mbar.arrive writes mbar bytes [512, 520)",
            "tma load #0", "wait the TMA completion mbar [4160]", "warp=0"},
            "SIMT RMW ownership race");
    }
    {
        // An engine RMW (TMA complete_tx) whose completion mbar sits inside an
        // operand stage two queued MMAs still read.
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.shmem_image, 8192);
        image_bytes(launch.gmem_image, 1024);
        write_tma_descriptor(shmem, 4096, linear_tma(false, 2048, 1024, 512));
        write_tx_mbar(shmem, 512, 1024);
        const auto message = run_expecting_panic(profile, compile(R"(
            .acc C, 0, 0, 64, 64
            mov r0, 0
            mov r1, 1024
            mov r8, 0
            mov r9, 0
            mov r10, 0
            mma.async.init C, [r0], [r1]
            mma.async C, [r0], [r1]
            tma.async [4096], r8
            halt
        )"), std::move(launch), "engine complete_tx into an MMA operand stage");
        require_mentions(message, {"engine complete_tx writes mbar bytes [512, 520)",
            "read-owner of its A operand [0, 1024)", "source=engine"},
            "engine RMW ownership race");
    }
}

void test_sanitizer_tma_and_mma_accept_conflicts() {
    const auto& profile = canonical_profile();
    const auto launch_with = [](bool second_store) {
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.shmem_image, 8192);
        image_bytes(launch.gmem_image, 1024);
        write_tma_descriptor(shmem, 4096, linear_tma(false, 0, 1024, 4160));
        write_tma_descriptor(shmem, 4224, linear_tma(second_store, 1024, 1024, 4168));
        write_tma_descriptor(shmem, 4288, linear_tma(true, 0, 1024, 4168));
        write_tx_mbar(shmem, 4160, 1024);
        write_tx_mbar(shmem, 4168, 1024);
        return launch;
    };
    // TMA load accepted over a stage an active MMA reads.
    auto message = run_expecting_panic(profile, compile(R"(
        .acc C, 0, 0, 64, 64
        mov r0, 0
        mov r1, 1024
        mov r8, 0
        mov r9, 0
        mov r10, 0
        mma.async.init C, [r0], [r1]
        tma.async [4096], r8
        halt
    )"), launch_with(false), "TMA load over an active MMA operand");
    require_mentions(message, {"shmem race: tma load #0 destination bytes [0, 1024)",
        "owned by mma.async #1 [fetching]", "A operand [0, 1024)",
        "wait the operand-consumed depbar sb", "inst='tma.async"}, "TMA load vs MMA");
    // MMA accepted over a pending TMA load destination (the other direction).
    message = run_expecting_panic(profile, compile(R"(
        .acc C, 0, 0, 64, 64
        mov r0, 0
        mov r1, 1024
        mov r8, 0
        mov r9, 0
        mov r10, 0
        tma.async [4224], r8
        mma.async.init C, [r0], [r1]
        halt
    )"), launch_with(false), "MMA over a pending TMA load destination");
    require_mentions(message, {"shmem race: mma.async #1 B operand bytes [1024, 2048)",
        "owned by tma load #0", "wait the TMA completion mbar [4168]"}, "MMA vs TMA load");
    // TMA store accepted whose source a pending TMA load still writes.
    message = run_expecting_panic(profile, compile(R"(
        mov r8, 0
        mov r9, 0
        mov r10, 0
        tma.async [4096], r8
        tma.async [4288], r8
        halt
    )"), launch_with(false), "TMA store of a pending TMA load destination");
    require_mentions(message, {"shmem race: tma store #1 source bytes [0, 1024)",
        "owned by tma load #0", "wait the TMA completion mbar [4160]"}, "TMA store vs TMA load");
    // Two readers never conflict: a TMA store of an MMA operand stage is legal.
    nanosm::Machine readers(profile, compile(R"(
        .acc C, 0, 0, 64, 64
        mov r0, 0
        mov r1, 1024
        mov r4, 4168
        mov r8, 0
        mov r9, 0
        mov r10, 0
        mma.async.init C, [r0], [r1]
        tma.async [4224], r8
    wait:
        mbar.try_wait p0, [r4], 0
        @!p0 bra.uni wait
        halt
    )"), launch_with(true));
    readers.run(20000);
}

void test_sanitizer_tmem_ownership_both_directions() {
    auto duplex = canonical_profile();
    duplex.tmem_simt_read_write_duplex = true;
    nanosm::validate_device_profile(duplex);
    const auto one_warp = [] {
        nanosm::Launch launch;
        launch.active_warps = 1;
        return launch;
    };
    const std::string header = R"(
        .acc C, 0, 0, 64, 64
        mov r0, 0
        mov r1, 1024
        mov r8, 0
        mov r9, 0
        mov r10, 16
        mov r11, 0
    )";
    // C4, new direction: an MMA accepted while an acc.st on its accumulator
    // is still writing tmem.
    auto message = run_expecting_panic(canonical_profile(), compile(header + R"(
        acc.st.v8 C, r8, r16
        mma.async.init C, [r0], [r1]
        halt
    )"), one_warp(), "MMA over an in-flight acc.st");
    require_mentions(message, {"tmem race: mma.async #1 on 'C'",
        "acc.st #0", "write-owner of 'C' rows [0, 16) cols [0, 8) until its final tmem write",
        "wait that transfer's depbar sb"}, "MMA vs acc.st");
    // ... and while an acc.ld still reads it (WAR on tmem).
    message = run_expecting_panic(canonical_profile(), compile(header + R"(
        acc.ld.v8 r24, C, r8
        mma.async.init C, [r0], [r1]
        halt
    )"), one_warp(), "MMA over an in-flight acc.ld");
    require_mentions(message, {"tmem race: mma.async #1 on 'C'", "acc.ld #0",
        "read-owner of 'C' rows [0, 16) cols [0, 8) until its final RF write"}, "MMA vs acc.ld");
    // Existing direction, same text plus the owner.
    message = run_expecting_panic(canonical_profile(), compile(header + R"(
        mma.async.init C, [r0], [r1]
        acc.ld.v8 r24, C, r8
        halt
    )"), one_warp(), "acc.ld over accepted MMA work");
    require_mentions(message, {"accumulator transfer races accepted MMA work on 'C' owned by "
        "mma.async #1", "write-owner of 'C' rows [0, 64) cols [0, 64) until its last result retires",
        ": wait on the mma.commit mbar phase first"}, "acc.ld vs MMA");
    // Transfer vs transfer. Whole-transfer slots order same-bank transfers
    // structurally (test "tmem ordering and store lifetime"), so only the
    // duplex profile can admit an overlapping pair.
    message = run_expecting_panic(duplex, compile(header + R"(
        acc.st.v8 C, r8, r16
        acc.ld.v8 r24, C, r8
        halt
    )", duplex), one_warp(), "acc.ld over an in-flight acc.st");
    require_mentions(message, {"tmem race: acc.ld reads 'C' rows [0, 16) cols [0, 8)",
        "overlapping acc.st #0", "wait that transfer's depbar sb"}, "acc.ld vs acc.st");
    message = run_expecting_panic(duplex, compile(header + R"(
        acc.ld.v8 r24, C, r8
        acc.st.v8 C, r8, r16
        halt
    )", duplex), one_warp(), "acc.st over an in-flight acc.ld");
    require_mentions(message, {"tmem race: acc.st writes 'C' rows [0, 16) cols [0, 8)",
        "overlapping acc.ld #0"}, "acc.st vs acc.ld");
    // Disjoint rows in duplex mode: concurrent, legal, and exact.
    nanosm::Machine disjoint(duplex, compile(header + R"(
        mov r16, 41
        acc.st.v8 C, r10, r16
        acc.ld.v8 r24, C, r8
        halt
    )", duplex), one_warp());
    disjoint.run(2000);
    require(disjoint.read_reg(0, 0, 24) == 0, "disjoint acc.ld read the concurrent store's rows");
}

void test_sanitizer_double_buffer_and_p_tile_are_clean() {
    const auto& profile = canonical_profile();
    {
        // TMA fills stage 1 while MMA reads stage 0, then the roles swap after
        // the stage mbar. Disjoint ranges: no panic, exact result.
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.shmem_image, 8192 + 24);
        auto& gmem = image_bytes(launch.gmem_image, 1024);
        for (uint32_t i = 0; i < 512; ++i) write_u16(gmem, 2 * i, 0x3c00);
        for (uint32_t stage = 0; stage < 4; ++stage) {
            nanosm::TmaDescriptor descriptor;
            descriptor.element_bytes = 2;
            descriptor.dimensions = {512, 1, 1};
            descriptor.box = {512, 1, 1};
            descriptor.shmem_base = 1024 * stage;
            descriptor.completion_mbar = stage < 2 ? 8192 : 8200;
            write_tma_descriptor(shmem, 4096 + 64 * stage, descriptor);
        }
        for (uint32_t mbar : {8192u, 8200u, 8208u}) write_count_mbar(shmem, mbar, 1);
        nanosm::Machine machine(profile, compile(R"(
            .acc C, 0, 0, 64, 64
            mov r8, 0
            mov r9, 0
            mov r10, 0
            mov r2, 8192
            mov r3, 8200
            mov r4, 8208
            mbar.arrive.expect_tx [r2], 2048
            tma.async [4096], r8
            tma.async [4160], r8
        L0:
            mbar.try_wait p1, [r2], 0
            @!p1 bra.uni L0
            mov r0, 0
            mov r1, 1024
            mma.async.init C, [r0], [r1]
            mbar.arrive.expect_tx [r3], 2048
            tma.async [4224], r8
            tma.async [4288], r8
        L1:
            mbar.try_wait p1, [r3], 0
            @!p1 bra.uni L1
            mov r0, 2048
            mov r1, 3072
            mma.async C, [r0], [r1]
            mbar.arrive.expect_tx [r2], 2048
            tma.async [4096], r8
            tma.async [4160], r8
        L2:
            mbar.try_wait p1, [r2], 1
            @!p1 bra.uni L2
            mov r0, 0
            mov r1, 1024
            mma.async C, [r0], [r1]
            mma.commit [r4]
        L3:
            mbar.try_wait p1, [r4], 0
            @!p1 bra.uni L3
            mov r12, 0
            mov r13, 0
            acc.ld.v8 r16, C, r12
            halt
        )"), std::move(launch), {.timeline = true});
        machine.run(50000);
        for (uint32_t lane = 0; lane < profile.lanes; ++lane)
            for (uint32_t j = 0; j < 8; ++j)
                require(machine.read_reg(0, lane, 16 + j) == nanosm::float_bits(24.0f),
                    "double-buffered MMA chain produced the wrong value");
        // The overlap really happened: stage 1's fill began before MMA #1 (on
        // stage 0) finished its operand reads, and stage 0's refill began
        // before MMA #2 (on stage 1) finished.
        std::map<uint64_t, uint64_t> tma_begin;
        std::map<uint64_t, uint64_t> mma_last_input;
        for (const auto& event : machine.trace().events()) {
            if (event.name == "tma command") tma_begin[event.id] = event.start_cycle;
            if (event.name == "MMA input step")
                mma_last_input[event.id] = std::max(mma_last_input[event.id], event.start_cycle);
        }
        require(tma_begin.size() == 6 && mma_last_input.size() == 3,
            "double-buffer trace lost its commands");
        require(tma_begin.at(2) < mma_last_input.at(1) && tma_begin.at(4) < mma_last_input.at(2),
            "double-buffer test did not overlap a TMA fill with an MMA read");
    }
    {
        // FA's P-tile handoff, twice: worker sts P -> mbar -> MMA reads P ->
        // commit mbar -> worker overwrites P -> mbar -> MMA reads it again.
        nanosm::Launch launch;
        launch.active_warps = 2;
        auto& shmem = image_bytes(launch.shmem_image, 8192 + 16);
        for (uint32_t i = 0; i < 512; ++i) write_u16(shmem, 1024 + 2 * i, 0x3c00);
        write_count_mbar(shmem, 8192, 1);
        write_count_mbar(shmem, 8200, 1);
        nanosm::Machine machine(profile, compile(R"(
            .acc C, 0, 0, 64, 64
            mov r10, 8192
            mov r11, 8200
            sread r0, WARP_ID
            setp.eq p0, r0, 0
            @p0 bra.uni ISSUER
            sread r20, LANE_ID
            shli r20, r20, 2
            mov r21, 15360
            shli r22, r21, 16
            or r21, r21, r22
            mov r25, 0
        FILL1:
            add r26, r20, r25
            sts [r26], r21
            addi r25, r25, 64
            setp.lt p1, r25, 1024
            @p1 bra.uni FILL1
            mbar.arrive [r10]
        DONE0:
            mbar.try_wait p1, [r11], 0
            @!p1 bra.uni DONE0
            mov r21, 16384
            shli r22, r21, 16
            or r21, r21, r22
            mov r25, 0
        FILL2:
            add r26, r20, r25
            sts [r26], r21
            addi r25, r25, 64
            setp.lt p1, r25, 1024
            @p1 bra.uni FILL2
            mbar.arrive [r10]
            halt
        ISSUER:
            mov r0, 0
            mov r1, 1024
        READY0:
            mbar.try_wait p1, [r10], 0
            @!p1 bra.uni READY0
            mma.async.init C, [r0], [r1]
            mma.commit [r11]
        WAIT0:
            mbar.try_wait p1, [r11], 0
            @!p1 bra.uni WAIT0
        READY1:
            mbar.try_wait p1, [r10], 1
            @!p1 bra.uni READY1
            mma.async C, [r0], [r1]
            mma.commit [r11]
        WAIT1:
            mbar.try_wait p1, [r11], 1
            @!p1 bra.uni WAIT1
            mov r12, 0
            mov r13, 0
            acc.ld.v8 r16, C, r12
            halt
        )"), std::move(launch));
        machine.run(50000);
        for (uint32_t lane = 0; lane < profile.lanes; ++lane)
            for (uint32_t j = 0; j < 8; ++j)
                require(machine.read_reg(0, lane, 16 + j) == nanosm::float_bits(24.0f),
                    "P-tile handoff produced the wrong value");
    }
}

void test_sanitizer_rmw_port_collision() {
    // C5: the RMW engine's write stage joins the same-cycle port-A record. An
    // mbar.init over the first word of an MMA's operand footprint, then the
    // MMA right behind it: with no padding, fetch round 0 reads word 0 in the
    // RMW's write cycle. (SIMT RMWs are ownership-checked at issue, like sts,
    // so this same-cycle record is what sees an MMA accepted in between.)
    const auto profile = dense_one_bank_profile(4);
    uint32_t panics = 0;
    for (uint32_t mbar : {0u, 896u}) {
        uint32_t mbar_panics = 0;
        for (uint32_t pad = 0; pad <= 4; ++pad) {
            std::ostringstream qtx;
            qtx << ".acc C, 0, 0, 16, 4\n"
                << "mov r0, 0\nmov r1, 512\nmov r4, " << mbar << "\n"
                << "mbar.init [r4], 1\n";
            for (uint32_t i = 0; i < pad; ++i) qtx << "mov r22, 1\n";
            qtx << "mma.async.init C, [r0], [r1]\nhalt\n";
            nanosm::Launch launch;
            launch.active_warps = 1;
            nanosm::Machine machine(profile, compile(qtx.str(), profile), std::move(launch));
            try {
                machine.run(4000);
            } catch (const nanosm::SimError& error) {
                require_mentions(error.what(), {"in the same cycle"}, "RMW/port-B collision");
                ++mbar_panics;
            }
        }
        if (mbar == 0) panics = mbar_panics;
        else require(mbar_panics == 0, "an mbar outside the operands must never collide");
    }
    require(panics >= 1, "the RMW write stage is missing from the port-A collision record");
    require(panics < 5, "different-cycle RMW/port-B overlap must not be diagnosed");
}

void test_integer_alu_menu() {
    // Toolchain round trip for the eight integer ALU ops.
    const auto program = nanosm::compile_qtx(R"(
        sread r0, LANE_ID
        lds r1, [r0]
        lds r2, [r0+64]
        and r4, r1, r2
        or  r5, r1, r2
        xor r7, r1, r2
        shl r8, r1, r2
        shr r10, r1, r2
        andi r11, r1, 255
        ori r13, r1, 4080
        shri r14, r1, 7
        halt
    )", canonical_profile());
    const auto assembled = nanosm::assemble_asm(nanosm::emit_asm(program), canonical_profile());
    const auto reloaded = nanosm::load_hex(nanosm::emit_hex(assembled));
    require(assembled.words == reloaded.words, "asm/hex round trip changed integer ALU words");
    std::set<nanosm::Opcode> seen;
    for (const auto word : assembled.words) seen.insert(nanosm::decode(word).opcode);
    for (const auto opcode : {nanosm::Opcode::and_, nanosm::Opcode::or_, nanosm::Opcode::xor_,
             nanosm::Opcode::shl, nanosm::Opcode::shr, nanosm::Opcode::andi,
             nanosm::Opcode::ori, nanosm::Opcode::shri}) {
        require(seen.count(opcode) == 1, "an integer ALU opcode did not survive the toolchain");
    }

    // Per-lane numeric vectors. B holds register shift amounts including 32
    // and 45: the 5-bit shifter uses amount mod 32, so 32 acts as 0 and 45 as 13.
    const std::array<uint32_t, 16> a{
        0xffffffffu, 0x80000000u, 0xdeadbeefu, 0x00000001u,
        0x12345678u, 0xa5a5a5a5u, 0x0f0f0f0fu, 0xfffffffeu,
        0x00000000u, 0x7fffffffu, 0xcafebabeu, 0x55555555u,
        0x89abcdefu, 0x40000000u, 0x00010000u, 0xffff0000u};
    const std::array<uint32_t, 16> b{
        0, 31, 1, 31,
        4, 8, 16, 1,
        13, 30, 32, 45,   // 32 -> 0, 45 -> 13 under the 5-bit shifter
        2, 62, 33, 95};   // 62 -> 30, 33 -> 1, 95 -> 31
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& image = image_bytes(launch.shmem_image, 128);
    for (uint32_t lane = 0; lane < 16; ++lane) {
        write_u32(image, lane * 4, a[lane]);
        write_u32(image, 64 + lane * 4, b[lane]);
    }
    nanosm::Machine machine(canonical_profile(), compile(R"(
        sread r0, LANE_ID
        shli r0, r0, 2
        lds r1, [r0]
        lds r2, [r0+64]
        and r4, r1, r2
        or  r5, r1, r2
        xor r7, r1, r2
        shl r8, r1, r2
        shr r10, r1, r2
        andi r11, r1, 255
        ori r13, r1, 4080
        shri r14, r1, 5
        shri r16, r1, 0
        shri r17, r1, 31
        halt
    )"), std::move(launch));
    machine.run(20000);
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(machine.read_reg(0, lane, 4) == (a[lane] & b[lane]), "and lane mismatch");
        require(machine.read_reg(0, lane, 5) == (a[lane] | b[lane]), "or lane mismatch");
        require(machine.read_reg(0, lane, 7) == (a[lane] ^ b[lane]), "xor lane mismatch");
        require(machine.read_reg(0, lane, 8) == (a[lane] << (b[lane] & 31)),
            "shl lane mismatch (5-bit amount rule)");
        require(machine.read_reg(0, lane, 10) == (a[lane] >> (b[lane] & 31)),
            "shr lane mismatch (5-bit amount rule)");
        require(machine.read_reg(0, lane, 11) == (a[lane] & 255u), "andi lane mismatch");
        require(machine.read_reg(0, lane, 13) == (a[lane] | 4080u), "ori lane mismatch");
        require(machine.read_reg(0, lane, 14) == (a[lane] >> 5), "shri lane mismatch");
        require(machine.read_reg(0, lane, 16) == a[lane], "shri by 0 changed the value");
        require(machine.read_reg(0, lane, 17) == (a[lane] >> 31), "shri by 31 mismatch");
    }

    // Immediate shifts keep the shli discipline: out-of-range imm panics.
    nanosm::Launch bad_launch;
    bad_launch.active_warps = 1;
    nanosm::Machine bad(canonical_profile(), compile("mov r0, 1\nshri r2, r0, 32\nhalt"),
        std::move(bad_launch));
    try {
        bad.run(20000);
        throw std::runtime_error("out-of-range shri immediate did not panic");
    } catch (const nanosm::SimError& error) {
        require(std::string(error.what()).find("shift outside 0..31") != std::string::npos,
            "shri range panic carried the wrong diagnostic");
    }

    // The idiom that motivated the slice: ((row >> 2) & 1) and a lane-vs-row
    // XOR bank skew, previously inexpressible without a host-side table.
    nanosm::Launch idiom_launch;
    idiom_launch.active_warps = 1;
    nanosm::Machine idiom(canonical_profile(), compile(R"(
        sread r0, LANE_ID
        mov  r1, 48
        add  r1, r0, r1
        shri r2, r1, 2
        andi r2, r2, 1
        mov  r4, 21
        andi r5, r1, 15
        add  r7, r4, r0
        xor  r8, r7, r5
        halt
    )"), std::move(idiom_launch));
    idiom.run(20000);
    for (uint32_t lane = 0; lane < 16; ++lane) {
        const uint32_t row = 48 + lane;
        require(idiom.read_reg(0, lane, 2) == ((row >> 2) & 1u), "((row >> 2) & 1) mismatch");
        require(idiom.read_reg(0, lane, 8) == ((21u + lane) ^ (row & 15u)),
            "lane-vs-row xor skew mismatch");
    }

    // Same-bank dual sources stay rejected at the canonical one-cycle window.
    try {
        (void)nanosm::compile_qtx("and r2, r0, r3\nhalt", canonical_profile());
        throw std::runtime_error("same-bank and sources were accepted at window 1");
    } catch (const nanosm::ProfileError&) {}
}

void test_fa_menu_encoding_round_trips() {
    nanosm::Instruction fma;
    fma.opcode = nanosm::Opcode::fma;
    fma.control.stall = 19;
    fma.dst = 8;
    fma.src0 = 250;
    fma.src1 = 13;
    fma.src2 = 201;
    const auto decoded = nanosm::decode(nanosm::encode(fma));
    require(decoded.opcode == nanosm::Opcode::fma && decoded.dst == 8 &&
            decoded.src0 == 250 && decoded.src1 == 13 && decoded.src2 == 201,
        "fma four-register encoding mismatch");
    nanosm::Instruction unary;
    unary.opcode = nanosm::Opcode::ex2;
    unary.dst = 100;
    unary.src0 = 55;
    const auto unary_decoded = nanosm::decode(nanosm::encode(unary));
    require(unary_decoded.opcode == nanosm::Opcode::ex2 && unary_decoded.dst == 100 &&
            unary_decoded.src0 == 55, "ex2 encoding mismatch");

    const auto program = nanosm::compile_qtx(R"(
        sread r0, LANE_ID
        lds r1, [r0]
        lds r2, [r0+64]
        lds r3, [r0+128]
        fmul r5, r1, r2
        fsub r13, r1, r2
        fmax r7, r1, r2
        fma r8, r1, r2, r3
        cvt.pack.f16x2 r14, r1, r2
        ex2 r10, r1
        rcp r11, r1
        halt
    )", canonical_profile());
    const auto assembled = nanosm::assemble_asm(nanosm::emit_asm(program), canonical_profile());
    const auto reloaded = nanosm::load_hex(nanosm::emit_hex(assembled));
    require(assembled.words == reloaded.words, "asm/hex round trip changed the FA menu words");
    std::set<nanosm::Opcode> seen;
    for (const auto word : assembled.words) seen.insert(nanosm::decode(word).opcode);
    for (const auto opcode : {nanosm::Opcode::fmul, nanosm::Opcode::fsub, nanosm::Opcode::fmax,
             nanosm::Opcode::fma, nanosm::Opcode::cvt_pack, nanosm::Opcode::ex2,
             nanosm::Opcode::rcp}) {
        require(seen.count(opcode) == 1, "an FA menu opcode did not survive the toolchain");
    }
}

void test_half_conversion_and_pack_vectors() {
    // Every half value round-trips through fp32 exactly: finite and infinite
    // by value, NaNs by preserved payload (<<13 then >>13).
    for (uint32_t h = 0; h < 0x10000; ++h) {
        const auto half = static_cast<uint16_t>(h);
        const auto back = nanosm::float_to_half_bits(nanosm::half_to_float_bits(half));
        require(back == half, "half -> float -> half round trip failed");
    }
    // Directed fp32 -> fp16 pins: RNE ties, overflow, subnormals, NaN, zeros.
    const std::array<std::pair<uint32_t, uint16_t>, 12> pins{{
        {0x3f800000u, 0x3c00}, // 1.0
        {0x3f801000u, 0x3c00}, // 1 + 2^-11: tie to even (down)
        {0x3f801800u, 0x3c01}, // 1 + 3*2^-12: rounds up
        {0x477fe000u, 0x7bff}, // 65504, half max
        {0x477ff000u, 0x7c00}, // 65520: tie rounds to infinity
        {0x33000000u, 0x0000}, // 2^-25: tie to even zero
        {0x33400000u, 0x0001}, // 1.5 * 2^-25 rounds to smallest subnormal
        {0xb3800000u, 0x8001}, // -2^-24
        {0x7f800000u, 0x7c00}, // +inf
        {0x7fc00000u, 0x7e00}, // canonical qNaN
        {0x7f800001u, 0x7e00}, // signaling payload truncates to zero: quiet bit forced
        {0x80000000u, 0x8000}, // -0
    }};
    for (const auto& [bits, expected] : pins) {
        require(nanosm::float_to_half_bits(bits) == expected,
            "fp32 -> fp16 pin mismatch for " + std::to_string(bits));
    }
}

void test_fa_menu_numeric_vectors() {
    // Lane-per-vector machine test. Expected values recompute the spec
    // algorithms locally; the marked lanes also pin literal result bits so a
    // platform libm drift in exp2 breaks loudly.
    constexpr uint32_t nan = 0x7fc00000u;
    const std::array<uint32_t, 16> a{
        0x3f800000u, 0xbf800000u, 0x3f000000u, 0xbf000000u, // 1, -1, 0.5, -0.5
        0x40400000u, 0x477fe000u, 0x33800000u, 0x00000000u, // 3, 65504, 2^-24, +0
        0x80000000u, 0x7f800000u, 0xff800000u, nan,          // -0, +inf, -inf, NaN
        0x40200000u, 0xc30c0000u, 0xc3160000u, 0x3f800001u}; // 2.5, -140, -150, 1+2^-23
    const std::array<uint32_t, 16> b{
        0x40000000u, 0xc0400000u, 0x3f000000u, 0xbf000000u,
        0x3fc00000u, 0x3f800000u, 0x42000000u, 0x80000000u, // ... fmax(+0, -0)
        0x00000000u, nan, 0xc0000000u, 0x40a00000u,          // fmax(NaN, 5) = 5
        0xbf000000u, 0x3f800000u, 0x3f800000u, 0x3f800001u};
    const std::array<uint32_t, 16> c{
        0x3f800000u, 0x3f800000u, 0x3f800000u, 0x3f800000u,
        0x3f800000u, 0x3f800000u, 0x3f800000u, 0x3f800000u,
        0x3f800000u, 0x3f800000u, 0x3f800000u, 0x3f800000u,
        0x3f800000u, 0x3f800000u, 0x3f800000u, 0xbf800002u}; // fma single-rounding case

    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.shmem_image, 256);
    for (uint32_t lane = 0; lane < 16; ++lane) {
        write_u32(shmem, lane * 4, a[lane]);
        write_u32(shmem, 64 + lane * 4, b[lane]);
        write_u32(shmem, 128 + lane * 4, c[lane]);
    }
    nanosm::Machine machine(canonical_profile(), compile(R"(
        sread r0, LANE_ID
        shli r0, r0, 2
        lds r1, [r0]
        lds r2, [r0+64]
        lds r3, [r0+128]
        fmul r5, r1, r2
        fsub r13, r1, r2
        fmax r7, r1, r2
        fma r8, r1, r2, r3
        cvt.pack.f16x2 r14, r1, r2
        ex2 r10, r1
        rcp r11, r1
        halt
    )"), std::move(launch));
    machine.run(10000);

    const auto canonical = [](float value) {
        return std::isnan(value) ? nan : nanosm::float_bits(value);
    };
    const auto as_float = nanosm::bits_float;
    for (uint32_t lane = 0; lane < 16; ++lane) {
        const float fa = as_float(a[lane]);
        const float fb = as_float(b[lane]);
        const float fc = as_float(c[lane]);
        require(machine.read_reg(0, lane, 5) == canonical(fa * fb), "fmul vector mismatch");
        require(machine.read_reg(0, lane, 13) == canonical(fa - fb), "fsub vector mismatch");
        const uint32_t expected_max = (std::isnan(fa) && std::isnan(fb)) ? nan
            : std::isnan(fa) ? b[lane]
            : std::isnan(fb) ? a[lane]
            : (fa == 0.0f && fb == 0.0f)
                ? ((std::signbit(fa) && std::signbit(fb)) ? a[lane] : 0u)
                : (fa > fb ? a[lane] : b[lane]);
        require(machine.read_reg(0, lane, 7) == expected_max, "fmax vector mismatch");
        require(machine.read_reg(0, lane, 8) == canonical(std::fma(fa, fb, fc)),
            "fma vector mismatch");
        const uint32_t expected_pack =
            static_cast<uint32_t>(nanosm::float_to_half_bits(a[lane])) |
            static_cast<uint32_t>(nanosm::float_to_half_bits(b[lane])) << 16;
        require(machine.read_reg(0, lane, 14) == expected_pack, "cvt.pack vector mismatch");
        const uint32_t expected_ex2 = std::isnan(fa) ? nan
            : canonical(static_cast<float>(std::exp2(static_cast<double>(fa))));
        require(machine.read_reg(0, lane, 10) == expected_ex2, "ex2 vector mismatch");
        const uint32_t expected_rcp = std::isnan(fa) ? nan
            : canonical(static_cast<float>(1.0 / static_cast<double>(fa)));
        require(machine.read_reg(0, lane, 11) == expected_rcp, "rcp vector mismatch");
    }
    // Literal pins, independent of this test's own reference arithmetic.
    require(machine.read_reg(0, 0, 10) == 0x40000000u, "ex2(1) != 2");
    require(machine.read_reg(0, 2, 10) == 0x3fb504f3u, "ex2(0.5) != sqrt(2) bits");
    require(machine.read_reg(0, 3, 10) == 0x3f3504f3u, "ex2(-0.5) != 1/sqrt(2) bits");
    require(machine.read_reg(0, 13, 10) == 0x00000200u, "ex2(-140) != subnormal 2^-140");
    require(machine.read_reg(0, 14, 10) == 0x00000000u, "ex2(-150) did not flush the tie to zero");
    require(machine.read_reg(0, 10, 10) == 0x00000000u, "ex2(-inf) != +0");
    require(machine.read_reg(0, 7, 11) == 0x7f800000u, "rcp(+0) != +inf");
    require(machine.read_reg(0, 8, 11) == 0xff800000u, "rcp(-0) != -inf");
    require(machine.read_reg(0, 9, 11) == 0x00000000u, "rcp(+inf) != +0");
    require(machine.read_reg(0, 7, 7) == 0x00000000u, "fmax(+0, -0) != +0");
    require(machine.read_reg(0, 11, 7) == 0x40a00000u, "fmax(NaN, 5) != 5");
    require(machine.read_reg(0, 15, 8) == 0x28800000u,
        "fma lost its single rounding: (1+2^-23)^2 - (1+2^-22) != 2^-46");
}

void test_sfu_unit_timing() {
    const auto& profile = canonical_profile();
    require(profile.timing.sfu == 11, "canonical effective SFU latency is not 3 + 8");
    const auto scheduled = nanosm::compile_qtx(R"(
        sread r1, LANE_ID
        ex2 r2, r1
        fadd r4, r2, r2
        halt
    )", profile);
    require(scheduled.instructions[1].control.stall == 11,
        "qtxc did not schedule the effective SFU latency");

    nanosm::Launch launch;
    launch.active_warps = 2;
    nanosm::Machine machine(profile, compile(R"(
        sread r1, LANE_ID
        ex2 r2, r1
        add r5, r1, r1
        ex2 r3, r1
        halt
    )"), std::move(launch), {.timeline = true});
    machine.run(10000);
    // The single unit serializes all four ex2 admissions by its occupancy.
    auto first = issued_at(machine, 8);
    auto second = issued_at(machine, 24);
    require(first.size() == 2 && second.size() == 2, "expected four SFU issues");
    std::vector<uint64_t> sfu_issues{first[0], first[1], second[0], second[1]};
    std::sort(sfu_issues.begin(), sfu_issues.end());
    for (std::size_t i = 1; i < sfu_issues.size(); ++i) {
        require(sfu_issues[i] - sfu_issues[i - 1] >= 4,
            "SFU admissions were not spaced by the unit occupancy");
    }
    // The ALU op right after each warp's first ex2 issued while the unit was
    // still busy: lanes and the SFU are independent throughputs.
    const auto adds = issued_at(machine, 16);
    require(adds.size() == 2, "expected one add per warp");
    for (uint32_t warp = 0; warp < 2; ++warp) {
        require(adds[warp] > first[warp] && adds[warp] < first[warp] + 4,
            "an independent ALU op waited on the busy SFU");
    }
    const auto& summary = machine.trace().summary();
    require(summary.sfu_ops == 4 && summary.sfu_busy_cycles == 16,
        "SFU counters did not record four quarter-rate ops");
    require(summary.sfu_unit_stalls > 0, "SFU contention produced no counted stalls");
    for (uint32_t warp = 0; warp < 2; ++warp)
        for (uint32_t lane = 0; lane < 16; ++lane)
            require(machine.read_reg(warp, lane, 5) == 2 * lane,
                "the interleaved ALU op computed the wrong value");
}

void test_fma_bank_discipline() {
    try {
        (void)nanosm::compile_qtx("fma r2, r0, r3, r1\nhalt", canonical_profile());
        throw std::runtime_error("same-bank fma sources were accepted at a one-cycle window");
    } catch (const nanosm::ProfileError&) {}
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.shmem_image, 64);
    for (uint32_t lane = 0; lane < 16; ++lane) write_u32(shmem, lane * 4, nanosm::float_bits(2.0f));
    nanosm::Machine machine(canonical_profile(), compile(R"(
        sread r0, LANE_ID
        shli r0, r0, 2
        lds r1, [r0]
        lds r2, [r0+0]
        lds r3, [r0+0]
        fma r5, r1, r2, r3
        halt
    )"), std::move(launch));
    machine.run(10000);
    for (uint32_t lane = 0; lane < 16; ++lane)
        require(machine.read_reg(0, lane, 5) == nanosm::float_bits(6.0f),
            "mod-3-distinct fma produced the wrong value");
}

void test_mini_softmax_smoke() {
    // One warp, four scores per lane. The kernel runs max-reduce -> subtract
    // -> ex2 -> sum-reduce -> reciprocal -> scale -> half2 pack; the expected
    // values recompute the same spec algorithms in the same association.
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.shmem_image, 512);
    std::array<std::array<float, 4>, 16> scores{};
    for (uint32_t lane = 0; lane < 16; ++lane) {
        for (uint32_t j = 0; j < 4; ++j) {
            scores[lane][j] = lane == 5 ? 1.25f : (lane * 4 + j) * 0.37f - 3.1f;
            write_u32(shmem, lane * 16 + j * 4, nanosm::float_bits(scores[lane][j]));
        }
    }
    nanosm::Machine machine(canonical_profile(), compile(R"(
        sread r0, LANE_ID
        shli r0, r0, 4
        lds r1, [r0]
        lds r2, [r0+4]
        lds r16, [r0+8]
        lds r17, [r0+12]
        fmax r5, r1, r2
        fmax r7, r16, r17
        fmax r9, r5, r7
        fsub r10, r1, r9
        fsub r11, r2, r9
        fsub r13, r16, r9
        fsub r14, r17, r9
        ex2 r19, r10
        ex2 r20, r11
        ex2 r22, r13
        ex2 r23, r14
        fadd r25, r19, r20
        fadd r26, r22, r23
        fadd r27, r25, r26
        rcp r30, r27
        fmul r34, r19, r30
        fmul r35, r20, r30
        fmul r37, r22, r30
        fmul r38, r23, r30
        cvt.pack.f16x2 r40, r34, r35
        cvt.pack.f16x2 r41, r37, r38
        halt
    )"), std::move(launch));
    machine.run(20000);
    for (uint32_t lane = 0; lane < 16; ++lane) {
        const auto& s = scores[lane];
        const float m = std::max(std::max(s[0], s[1]), std::max(s[2], s[3]));
        std::array<float, 4> e{};
        for (uint32_t j = 0; j < 4; ++j) {
            e[j] = static_cast<float>(std::exp2(static_cast<double>(s[j] - m)));
        }
        const float sum = (e[0] + e[1]) + (e[2] + e[3]);
        const float inverse = static_cast<float>(1.0 / static_cast<double>(sum));
        std::array<uint16_t, 4> packed{};
        for (uint32_t j = 0; j < 4; ++j) {
            packed[j] = nanosm::float_to_half_bits(nanosm::float_bits(e[j] * inverse));
        }
        const uint32_t low = static_cast<uint32_t>(packed[0]) |
            static_cast<uint32_t>(packed[1]) << 16;
        const uint32_t high = static_cast<uint32_t>(packed[2]) |
            static_cast<uint32_t>(packed[3]) << 16;
        require(machine.read_reg(0, lane, 40) == low, "softmax packed low pair mismatch");
        require(machine.read_reg(0, lane, 41) == high, "softmax packed high pair mismatch");
    }
}

void test_sfu_profile_validation() {
    std::ifstream input("profiles/nanosm-v0.json");
    nlohmann::json document;
    input >> document;
    require(nanosm::parse_device_profile(document.dump()).timing.sfu == 11,
        "canonical sfu occupancy + latency did not derive 11");
    for (const auto value : {nlohmann::json(0), nlohmann::json(17), nlohmann::json(2.5),
             nlohmann::json(-1)}) {
        auto modified = document;
        modified["sfu"]["lanes"] = value;
        try {
            (void)nanosm::parse_device_profile(modified.dump());
            throw std::runtime_error("invalid sfu lane count was accepted");
        } catch (const nanosm::ProfileError&) {}
    }
    for (const auto value : {nlohmann::json(0), nlohmann::json(25), nlohmann::json(1.5)}) {
        auto modified = document;
        modified["sfu"]["latency_cycles"] = value;
        try {
            (void)nanosm::parse_device_profile(modified.dump());
            throw std::runtime_error("invalid sfu latency was accepted");
        } catch (const nanosm::ProfileError&) {}
    }
    auto faster = document;
    faster["sfu"]["lanes"] = 16;
    require(nanosm::device_profile_fingerprint(nanosm::parse_device_profile(faster.dump())) !=
            nanosm::device_profile_fingerprint(nanosm::parse_device_profile(document.dump())),
        "sfu throughput was omitted from the profile fingerprint");
}

} // namespace

// ===========================================================================
// Stepping discipline (trash/stepping-discipline-2026-10-08.md)
// ===========================================================================

// ---- Reg<T> (src/reg.hpp) ----

void test_reg_hold_write_commit() {
    nanosm::Sanitizer sanitizer;
    uint64_t cycle = 7;
    nanosm::RegisterFile file(sanitizer, cycle);
    nanosm::Reg<int> reg(file, "test register", 5);
    sanitizer.record_cycle_start();
    file.commit();
    require(reg.cur() == 5, "a register nobody wrote did not hold its value");
    sanitizer.record_cycle_start();
    reg.set_next(9, "writer");
    require(reg.cur() == 5, "set_next() changed Q before the clock edge");
    file.commit();
    require(reg.cur() == 9, "commit did not publish D");
    sanitizer.record_cycle_start();
    file.commit();
    require(reg.cur() == 9, "the tick after a write did not hold");
}

// A second write in one tick is a PANIC naming the register, the cycle and
// both writers, even when the same writer writes twice.
void test_reg_duplicate_write_panics() {
    nanosm::Sanitizer sanitizer;
    uint64_t cycle = 7;
    nanosm::RegisterFile file(sanitizer, cycle);
    nanosm::Reg<int> reg(file, "test register", 0);
    const auto expect_panic = [&](const char* first, const char* second) {
        sanitizer.record_cycle_start();
        reg.set_next(1, first);
        try {
            reg.set_next(2, second);
        } catch (const nanosm::SimError& error) {
            const std::string message = error.what();
            for (const std::string needle : {std::string("register 'test register'"), std::string("cycle=7"),
                     std::string("first by '") + first + "'", std::string("then by '") + second + "'"}) {
                require(message.find(needle) != std::string::npos, "duplicate-write PANIC lacks '" + needle + "': " + message);
            }
            file.commit();
            return;
        }
        throw std::runtime_error("a second write in one tick did not panic");
    };
    expect_panic("drain", "fill");
    expect_panic("drain", "drain"); // the same writer twice
    sanitizer.record_cycle_start();
    reg.set_next(3, "next tick");    // a new tick: one write is fine again
    file.commit();
    require(reg.cur() == 3, "the register did not recover after a duplicate-write PANIC");
}

// ---- The continuing vector load's assembly register (Codex's four cases + the final chunk) ----

// One ldg.v8 of the 512 B block at gmem 0 (eight 64 B chunks; chunk k lands
// in r(8+k)), seen through the edge probe.
struct VectorLoadTimeline {
    std::array<uint64_t, 8> first_word{}; // the chunk's first word enters the assembly register
    std::array<uint64_t, 8> last_word{};  // its sixteenth
    std::array<uint64_t, 8> drain{};      // its RF write beat
    std::array<uint32_t, 8> drains{};     // RF write beats of r(8+k): must be exactly one
    std::map<uint64_t, std::vector<uint32_t>> chunks_written_at; // cycle -> chunks receiving words
};

VectorLoadTimeline vector_load_timeline(const nanosm::Machine& machine, uint64_t command) {
    VectorLoadTimeline t;
    std::set<uint64_t> requests; // sector order = request order within the command
    for (const auto& event : machine.edge_events())
        if (event.edge == LsuEdge::assembly_write && event.id == command) requests.insert(event.aux);
    require(requests.size() == 16, "vector load: expected 16 sectors");
    const std::vector<uint64_t> order(requests.begin(), requests.end());
    const auto chunk_of = [&](uint64_t request) {
        return static_cast<uint32_t>((std::find(order.begin(), order.end(), request) - order.begin()) / 2);
    };
    t.first_word.fill(std::numeric_limits<uint64_t>::max());
    for (const auto& event : machine.edge_events()) {
        if (event.edge == LsuEdge::assembly_write && event.id == command) {
            const auto chunk = chunk_of(event.aux);
            t.first_word[chunk] = std::min(t.first_word[chunk], event.cycle);
            t.last_word[chunk] = std::max(t.last_word[chunk], event.cycle);
            t.chunks_written_at[event.cycle].push_back(chunk);
        }
        if (event.edge == LsuEdge::rf_write_beat && event.id == command && event.aux >= 8 && event.aux < 16) {
            t.drain[event.aux - 8] = event.cycle;
            ++t.drains[event.aux - 8];
        }
    }
    return t;
}

void require_vector_data(const nanosm::Machine& machine, uint32_t warp, uint32_t first_reg, uint32_t first_word,
    const std::string& what) {
    for (uint32_t k = 0; k < 8; ++k)
        for (uint32_t lane = 0; lane < 16; ++lane)
            require(machine.read_reg(warp, lane, first_reg + k) == pattern_word(first_word + 16 * k + lane),
                what + ": r" + std::to_string(first_reg + k) + " lane " + std::to_string(lane) +
                    " holds a word of another chunk");
}

std::unique_ptr<nanosm::Machine> run_vector_load(const nanosm::DeviceProfile& config) {
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& gmem = image_bytes(launch.gmem_image, 4096);
    fill_pattern(gmem, 0, 1024, 0);
    auto machine = std::make_unique<nanosm::Machine>(
        config, compile("mov r0, 0\nldg.v8 r8, [r0]\nhalt\n", config), std::move(launch));
    machine->enable_edge_probe();
    machine->run();
    return machine;
}

// (1) The RF drain is granted: the old vector is written exactly once, from
// Q; the next chunk's refill starts in that same cycle (the chunk being
// filled is chunks_done + the drain now); no new word joins the old write.
void test_assembly_contract_drain_granted() {
    const auto machine = run_vector_load(fast_bus_profile(2));
    const auto t = vector_load_timeline(*machine, 0);
    for (uint32_t k = 0; k < 8; ++k) {
        require(t.drains[k] == 1, "r" + std::to_string(8 + k) + " was not written exactly once");
        if (k == 7) continue;
        require(t.first_word[k + 1] == t.drain[k],
            "chunk " + std::to_string(k + 1) + " did not start filling in the cycle chunk " +
                std::to_string(k) + " drained (" + std::to_string(t.first_word[k + 1]) + " vs " +
                std::to_string(t.drain[k]) + ")");
        for (const auto chunk : t.chunks_written_at.at(t.drain[k]))
            require(chunk == k + 1, "a word of chunk " + std::to_string(chunk) + " entered while chunk " +
                std::to_string(k) + " drained");
    }
    require_vector_data(*machine, 0, 8, 0, "drain granted");
}

// (2) The RF drain is denied by an existing booking (warp 1's movs book the
// same bank's write port at every issue): the old vector stays put, the
// ingress FIFO is not read, and no chunk, depbar or command state changes.
// When the bookings stop, it drains, intact.
void test_assembly_contract_drain_denied() {
    auto config = fast_bus_profile(2);
    nanosm::Launch launch;
    launch.active_warps = 2;
    auto& gmem = image_bytes(launch.gmem_image, 4096);
    fill_pattern(gmem, 0, 1024, 0);
    // r8 and r2 share RF bank 2: warp 1's mov at t books bank 2 for t + 1.
    std::string text = R"(
        sread r29, WARP_ID
        setp.ne p0, r29, 0
        @p0 bra.uni other
        mov r30, 0
        ldg.v8 r8, [r30]
        halt
    other:
    )";
    for (int i = 0; i < 200; ++i) text += "mov r2, 1\n";
    text += "halt\n";
    nanosm::Machine machine(config, compile(text, config), std::move(launch));
    machine.enable_edge_probe();
    uint64_t waits = machine.trace().summary().lsu_rf_write_wait_cycles;
    uint32_t denied = 0;
    std::size_t seen = 0;
    while (!machine.done()) {
        const auto cycle = machine.cycle();
        tick_capped(machine);
        const auto now = machine.trace().summary().lsu_rf_write_wait_cycles;
        const bool was_denied = now > waits;
        waits = now;
        std::vector<LsuEdge> edges;
        for (; seen < machine.edge_events().size(); ++seen) edges.push_back(machine.edge_events()[seen].edge);
        if (!was_denied) continue;
        ++denied;
        const auto where = " in denied cycle " + std::to_string(cycle);
        for (const auto edge : edges) {
            require(edge != LsuEdge::rf_write_beat, "an RF write beat" + where);
            require(edge != LsuEdge::assembly_write, "a word joined the full assembly register" + where);
            require(edge != LsuEdge::ingress_read, "the ingress FIFO was read" + where);
        }
        uint32_t pending = 0;
        for (uint8_t bar = 0; bar < 6; ++bar) pending += machine.depbar_value(0, bar);
        require(pending == 1, "warp 0's depbar changed" + where);
        require(machine.lsu_commands_in_flight() == 1, "the command slot changed" + where);
    }
    require(denied >= 100, "the drain was denied only " + std::to_string(denied) + " cycles; the test proves little");
    const auto t = vector_load_timeline(machine, 0);
    for (uint32_t k = 0; k < 8; ++k) require(t.drains[k] == 1, "a chunk was written twice or never");
    require_vector_data(machine, 0, 8, 0, "drain denied");
}

// (3) A fill that completes in a cycle does not drain in it: the RF write is
// exactly the next cycle (uncontended).
void test_assembly_contract_fill_completes() {
    const auto machine = run_vector_load(fast_bus_profile(2));
    const auto t = vector_load_timeline(*machine, 0);
    for (uint32_t k = 0; k < 8; ++k)
        require(t.drain[k] == t.last_word[k] + 1,
            "chunk " + std::to_string(k) + ": sixteenth word at " + std::to_string(t.last_word[k]) +
                ", RF write at " + std::to_string(t.drain[k]) + " (not the next cycle)");
}

// (4) An empty register under a wide pluck: words reach the RF no earlier
// than the next cycle. At 8 words per cycle a chunk fills in two cycles. At
// 16 words per cycle, once the ingress FIFO has backed up (warp 1 holds chunk
// 0's RF write port for a while), a whole chunk enters the register emptied
// by the previous drain in one cycle, and still drains only the next.
void test_assembly_contract_wide_pluck() {
    {
        const auto machine = run_vector_load(fast_bus_profile(8));
        const auto t = vector_load_timeline(*machine, 0);
        for (uint32_t k = 0; k < 8; ++k)
            require(t.drain[k] == t.last_word[k] + 1, "a chunk passed through to the RF (pluck 8)");
        require_vector_data(*machine, 0, 8, 0, "wide pluck 8");
    }
    auto config = fast_bus_profile(16);
    config.dram_clock_hz = 2 * config.core_clock_hz;
    nanosm::Launch launch;
    launch.active_warps = 2;
    fill_pattern(image_bytes(launch.gmem_image, 4096), 0, 1024, 0);
    std::string text = R"(
        sread r29, WARP_ID
        setp.ne p0, r29, 0
        @p0 bra.uni other
        mov r30, 0
        ldg.v8 r8, [r30]
        halt
    other:
    )";
    for (int i = 0; i < 40; ++i) text += "mov r2, 1\n"; // r2 shares r8's bank
    text += "halt\n";
    nanosm::Machine machine(config, compile(text, config), std::move(launch));
    machine.enable_edge_probe();
    machine.run();
    const auto t = vector_load_timeline(machine, 0);
    uint32_t whole = 0;
    for (uint32_t k = 0; k < 8; ++k) {
        require(t.drain[k] > t.last_word[k], "a chunk passed through to the RF (pluck 16)");
        if (k == 0) continue; // chunk 0 waited on warp 1's bookings
        require(t.drain[k] == t.last_word[k] + 1, "an uncontended chunk did not drain the next cycle (pluck 16)");
        if (t.first_word[k] == t.last_word[k]) ++whole;
    }
    require(whole != 0, "no chunk entered its register in one cycle; the test proves little");
    require_vector_data(machine, 0, 8, 0, "wide pluck 16");
}

// (5) The final chunk: its RF write releases the command slot and the
// assembly register, and issue reallocates both to the next ldg.v8 in that
// same cycle (lsu.active_commands = staging_vectors = 1). Never two owners;
// the new command's words start after its own requests; no data mixes.
void test_assembly_contract_final_chunk() {
    auto config = fast_bus_profile(2);
    config.lsu_staging_vectors = 1;
    config.lsu_active_commands = 1;
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& gmem = image_bytes(launch.gmem_image, 4096);
    fill_pattern(gmem, 0, 1024, 0);
    nanosm::Machine machine(config,
        compile("mov r0, 0\nmov r1, 512\nldg.v8 r8, [r0]\nldg.v8 r16, [r1]\nhalt\n", config),
        std::move(launch), {.timeline = true});
    machine.enable_edge_probe();
    while (!machine.done()) {
        tick_capped(machine);
        require(machine.lsu_commands_in_flight() <= 1, "two LSU commands held the one slot");
    }
    std::map<uint64_t, std::pair<uint64_t, uint64_t>> life; // command -> admitted, released
    for (const auto& event : machine.trace().events())
        if (event.category == nanosm::TraceCategory::gmem && event.name == "ldg.v8")
            life[event.id] = {event.start_cycle, event.start_cycle + event.duration_cycles};
    require(life.size() == 2, "expected two ldg.v8 commands");
    const auto first = vector_load_timeline(machine, 0);
    const auto final_drain = first.drain[7];
    require(life[0].second == final_drain, "the first command did not end at its final RF write");
    require(life[1].first == final_drain,
        "the second ldg.v8 was not admitted in the cycle the first released its slot (" +
            std::to_string(life[1].first) + " vs " + std::to_string(final_drain) + ")");
    const auto second = vector_load_timeline(machine, 1);
    require(second.first_word[0] > final_drain, "the new command's words entered in the reallocation cycle");
    require_vector_data(machine, 0, 8, 0, "final chunk (old command)");
    require_vector_data(machine, 0, 16, 128, "final chunk (new command)");
}

// ---- Same-cycle reuse after a load's final RF drain (staging = active = 1) ----

// The one assembly register at the end of a cycle.
struct AssemblyAfter {
    std::optional<uint64_t> owner;
    std::vector<uint8_t> valid;
    uint32_t valid_count = 0;
};

struct ReuseRun {
    std::unique_ptr<nanosm::Machine> machine;
    uint64_t reuse_cycle = 0; // the old load's final RF write, and the second command's admission
    AssemblyAfter after;      // the register after that cycle
};

// `old_load` is LSU command 0, `second` (opcode `second_name`) command 1. With
// one command slot and one assembly register, the second waits for both and
// is admitted in the cycle the load's final RF write releases them (d).
// gmem holds pattern words; r5 = 4 * lane, r6 = 2048 + 4 * lane, r24..r31 =
// 101..108.
ReuseRun run_reuse_after_final_drain(const std::string& old_load, const std::string& second,
    const std::string& second_name) {
    auto config = fast_bus_profile(2);
    config.lsu_staging_vectors = 1;
    config.lsu_active_commands = 1;
    nanosm::Launch launch;
    launch.active_warps = 1;
    fill_pattern(image_bytes(launch.gmem_image, 4096), 0, 1024, 0);
    std::string text = "sread r4, LANE_ID\nshli r5, r4, 2\nori r6, r5, 2048\nmov r0, 0\nmov r2, 2048\n";
    for (uint32_t j = 0; j < 8; ++j) text += "mov r" + std::to_string(24 + j) + ", " + std::to_string(101 + j) + "\n";
    text += old_load + "\n" + second + "\nhalt\n";
    ReuseRun run;
    run.machine = std::make_unique<nanosm::Machine>(
        config, compile(text, config), std::move(launch), nanosm::TraceOptions{.timeline = true});
    auto& machine = *run.machine;
    machine.enable_edge_probe();
    std::map<uint64_t, AssemblyAfter> after;
    while (!machine.done()) {
        const auto cycle = machine.cycle();
        tick_capped(machine);
        after[cycle] = {machine.assembly_owner(0), machine.assembly_valid(0), machine.assembly_valid_count(0)};
    }
    uint64_t final_drain = 0;
    for (const auto& event : machine.edge_events())
        if (event.edge == LsuEdge::rf_write_beat && event.id == 0) final_drain = std::max(final_drain, event.cycle);
    std::optional<uint64_t> admitted;
    for (const auto& event : machine.trace().events())
        if (event.category == nanosm::TraceCategory::gmem && event.name == second_name && event.id == 1)
            admitted = event.start_cycle;
    require(admitted == final_drain,
        second_name + " was not admitted in the cycle '" + old_load + "' made its final RF write");
    run.reuse_cycle = final_drain;
    run.after = after.at(final_drain);
    return run;
}

void require_old_load_data(const nanosm::Machine& machine, const std::string& old_load) {
    if (old_load.starts_with("ldg.v8")) {
        require_vector_data(machine, 0, 8, 0, old_load);
        return;
    }
    for (uint32_t lane = 0; lane < 16; ++lane)
        require(machine.read_reg(0, lane, 8) == pattern_word(lane), old_load + ": lane " + std::to_string(lane));
}

// A scalar store admitted in a load's final-drain cycle: after the cycle it
// owns the register with all sixteen lanes valid (Codex 2026-10-08: the old
// load's drain used to clear them after the store's fill). The output alone
// cannot show it: scalar-store egress reads the words without valid_count.
void test_assembly_reuse_load_to_scalar_store() {
    for (const std::string old_load : {"ldg.v8 r8, [r0]", "ldg r8, [r5]"}) {
        const auto run = run_reuse_after_final_drain(old_load, "stg [r5 + 2048], r6", "stg");
        const auto where = " after the reuse cycle " + std::to_string(run.reuse_cycle) + " (" + old_load + ")";
        require(run.after.owner == 1u, "the store does not own the register" + where);
        require(run.after.valid_count == 16 && run.after.valid == std::vector<uint8_t>(16, 1),
            "the store's valid bits were lost" + where);
        for (uint32_t lane = 0; lane < 16; ++lane)
            require(run.machine->read_gmem_u32(2048 + 4 * lane) == 2048 + 4 * lane,
                "the scalar store wrote the wrong word for lane " + std::to_string(lane));
        require_old_load_data(*run.machine, old_load);
    }
}

// A stg.v8 admitted in that cycle: its first RF read beat (bypass (e)) fills
// the register in the same cycle, after the old load's drain emptied it.
void test_assembly_reuse_load_to_vector_store() {
    const std::string old_load = "ldg.v8 r8, [r0]";
    const auto run = run_reuse_after_final_drain(old_load, "stg.v8 [r2], r24", "stg.v8");
    bool first_read_now = false;
    for (const auto& event : run.machine->edge_events())
        if (event.edge == LsuEdge::rf_read_beat && event.id == 1 && event.cycle == run.reuse_cycle) first_read_now = true;
    require(first_read_now, "the stg.v8's first RF read was not in the reuse cycle; the test proves little");
    require(run.after.owner == 1u, "the stg.v8 does not own the register after the reuse cycle");
    require(run.after.valid_count == 16 && run.after.valid == std::vector<uint8_t>(16, 1),
        "the stg.v8's first chunk is not valid after the reuse cycle");
    for (uint32_t j = 0; j < 8; ++j)
        for (uint32_t lane = 0; lane < 16; ++lane)
            require(run.machine->read_gmem_u32(2048 + 4 * (16 * j + lane)) == 101 + j,
                "the stg.v8 wrote the wrong word: r" + std::to_string(24 + j) + " lane " + std::to_string(lane));
    require_old_load_data(*run.machine, old_load);
}

// A new load admitted in that cycle starts empty. (Assembly contract 5
// covers ldg.v8 -> ldg.v8 through the data path.)
void test_assembly_reuse_load_to_load() {
    const std::string old_load = "ldg.v8 r8, [r0]";
    const auto run = run_reuse_after_final_drain(old_load, "ldg r16, [r5 + 1024]", "ldg");
    require(run.after.owner == 1u, "the new load does not own the register after the reuse cycle");
    require(run.after.valid_count == 0 && run.after.valid == std::vector<uint8_t>(16, 0),
        "the new load's register is not empty after the reuse cycle");
    for (uint32_t lane = 0; lane < 16; ++lane)
        require(run.machine->read_reg(0, lane, 16) == pattern_word(256 + lane),
            "the new load returned the wrong word for lane " + std::to_string(lane));
    require_old_load_data(*run.machine, old_load);
}

// The sanitizer's guard for the case above: a reader of an assembly register
// that takes a lane whose valid bit is clear panics.
void test_sanitizer_assembly_reader_takes_valid_lanes() {
    nanosm::Sanitizer sanitizer;
    std::vector<uint8_t> valid(16, 1);
    sanitizer.check_assembly_lanes_valid(5, 0, valid, 0, 16, "RF drain");
    valid[3] = 0;
    sanitizer.check_assembly_lanes_valid(5, 0, valid, 4, 12, "egress write port"); // lanes 4..15 only
    try {
        sanitizer.check_assembly_lanes_valid(5, 0, valid, 3, 1, "egress write port");
    } catch (const nanosm::SimError& error) {
        const std::string message = error.what();
        require(message.find("egress write port read lane 3 of assembly register 0") != std::string::npos,
            "the lane check's PANIC does not name the reader, lane and register: " + message);
        return;
    }
    throw std::runtime_error("a read of an invalid assembly lane did not panic");
}

// ---- The mbar event registers: both engines alike ----

// A TMA load and an MMA commit, far apart: each engine's event, presented in
// cycle p, is first claimable in p + 1, and the free RMW engine claims it
// then. (Before 2026-10-08 a TMA event could be claimed in p itself, an
// accident of the call order in tick().)
void test_mbar_engine_events_visible_next_cycle() {
    constexpr uint32_t commit_mbar = 4096;
    constexpr uint32_t tx_mbar = 4104;
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.shmem_image, 8192);
    auto& gmem = image_bytes(launch.gmem_image, 64);
    for (uint32_t i = 0; i < 16; ++i) write_u32(gmem, 4 * i, 0x100u + i);
    write_mbar(shmem, commit_mbar, 1, 0);
    write_mbar(shmem, tx_mbar, 0, 64);
    nanosm::TmaDescriptor descriptor;
    descriptor.dimensions = {16, 1, 1};
    descriptor.box = {16, 1, 1};
    descriptor.shmem_base = 2048;
    descriptor.completion_mbar = tx_mbar;
    write_tma_descriptor(shmem, 6144, descriptor);
    std::string text = ".acc C, 0, 0, 64, 64\nmov r0, 0\nmov r1, 1024\nmov r4, 4096\n"
        "mov r8, 0\nmov r9, 0\nmov r10, 0\ntma.async [6144], r8\n";
    for (uint32_t i = 0; i < 300; ++i) text += "mov r" + std::to_string(12 + i % 8) + ", " + std::to_string(i) + "\n";
    text += "mma.async.init C, [r0], [r1]\nmma.commit [r4]\nhalt\n";
    nanosm::Machine machine(canonical_profile(), compile(text), std::move(launch), {.timeline = true});
    std::optional<uint64_t> tma_presented;
    std::optional<uint64_t> mma_presented;
    while (!machine.done()) {
        tick_capped(machine);
        if (!tma_presented && machine.tma_mbar_event_held()) tma_presented = machine.cycle() - 1;
        if (!mma_presented && machine.mma_mbar_event_held()) mma_presented = machine.cycle() - 1;
    }
    require(tma_presented && mma_presented, "an engine never presented its event");
    require(*tma_presented + 20 < *mma_presented, "the two events were meant to be far apart");
    require(mbar_rmw_services(machine) == std::vector<RmwService>{
                {*tma_presented + 1, "complete_tx (tma), address=4104"},
                {*mma_presented + 1, "arrive (mma.commit), address=4096"}},
        "an engine event was not claimed exactly one cycle after it was presented");
}

// ---- Simultaneous events combine ----

// A depbar completion and an admission on the same counter in one cycle: the
// lds on sb0 completes (its depbar clear lands at the RF write port) and the
// ldg that waits on sb0 and sets it again issues in that same cycle: issue
// sees the completion at once (bypass (b)), and the counter is one
// calculation, 1 - 1 + 1 = 1. Neither event is deferred.
void test_simultaneous_depbar_completion_and_admission() {
    const auto& profile = canonical_profile();
    const auto assembly =
        ".profile " + profile.name + ", 0x" +
        nanosm::format_profile_fingerprint(nanosm::device_profile_fingerprint(profile)) + R"(
        mov  r0, 0 {pred=PT, stall=0, set=-, wait=-}
        lds  r1, [r0] {pred=PT, stall=0, set=sb0, wait=-}
        mov  r5, 7 {pred=PT, stall=0, set=-, wait=-}
        ldg  r3, [r0] {pred=PT, stall=0, set=sb0, wait=sb0}
        halt {pred=PT, stall=0, set=-, wait=sb0}
    )";
    nanosm::Launch launch;
    launch.active_warps = 1;
    write_u32(image_bytes(launch.shmem_image, 64), 0, 0xabcd);
    write_u32(image_bytes(launch.gmem_image, 64), 0, 0x1234);
    nanosm::Machine machine(profile, nanosm::assemble_asm(assembly, profile), std::move(launch), {.timeline = true});
    std::map<uint64_t, uint32_t> sb0_after; // cycle -> sb0 at the next cycle's start
    while (!machine.done()) {
        const auto cycle = machine.cycle();
        tick_capped(machine);
        sb0_after[cycle] = machine.depbar_value(0, 0);
    }
    std::vector<std::pair<uint64_t, std::string>> sb0_events;
    uint64_t ldg_issue = 0;
    for (const auto& event : machine.trace().events()) {
        if (event.name == "depbar sb0" && event.track == 100) sb0_events.emplace_back(event.start_cycle, event.detail);
        if (event.name == "ldg") ldg_issue = event.start_cycle;
    }
    const auto completion = std::find(sb0_events.begin(), sb0_events.end(),
        std::make_pair(ldg_issue, std::string("value=0")));
    require(completion != sb0_events.end(), "the lds did not complete in the ldg's issue cycle");
    require(completion + 1 != sb0_events.end() && *(completion + 1) == std::make_pair(ldg_issue, std::string("value=1")),
        "the admission did not follow the completion in the same cycle");
    require(sb0_after.at(ldg_issue) == 1, "sb0 is not 1 - 1 + 1 after the cycle");
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(machine.read_reg(0, lane, 1) == 0xabcd && machine.read_reg(0, lane, 3) == 0x1234,
            "a load's value was lost");
    }
}

// A full FIFO popped and pushed in one cycle. The MMA queue (two entries) is
// full; its head retires in the cycle a third mma.async issues: the pop and
// the push combine (depth' = 2 - 1 + 1), the issue is not deferred.
void test_simultaneous_full_queue_pop_and_push() {
    nanosm::Launch launch;
    launch.active_warps = 1;
    image_bytes(launch.shmem_image, 4096);
    nanosm::Machine machine(canonical_profile(), compile(R"(
        .acc C, 0, 0, 64, 64
        mov r0, 0
        mov r1, 1024
        mma.async.init C, [r0], [r1]
        mma.async C, [r0], [r1]
        mma.async C, [r0], [r1]
        halt
    )"), std::move(launch), {.timeline = true});
    std::map<uint64_t, std::size_t> depth_after;
    bool waited_full = false;
    while (!machine.done()) {
        const auto cycle = machine.cycle();
        tick_capped(machine);
        depth_after[cycle] = machine.mma_queue_depth();
    }
    uint64_t third_issue = 0;
    uint64_t first_last_result = 0;
    for (const auto& event : machine.trace().events()) {
        if (event.category != nanosm::TraceCategory::mma && event.category != nanosm::TraceCategory::warp) continue;
        if (event.name == "mma.async" && event.id == 3) third_issue = event.start_cycle;
        if (event.name == "MMA tmem write" && event.id == 1) first_last_result = std::max(first_last_result, event.start_cycle);
        if (event.category == nanosm::TraceCategory::warp && event.detail == "MMA command queue full") waited_full = true;
    }
    require(waited_full, "the third mma.async never found the queue full");
    require(third_issue == first_last_result,
        "the third mma.async did not issue in the cycle the head retired (" + std::to_string(third_issue) +
            " vs " + std::to_string(first_last_result) + ")");
    require(depth_after.at(third_issue - 1) == 2 && depth_after.at(third_issue) == 2,
        "the full queue's pop and push did not combine");
}

// The ingress FIFO's declared full flag is registered (B2.1 implementer
// choice 3, kept): when the FIFO is full at the start of a cycle, the bus
// cannot write a beat in that cycle even if the read port frees room in it;
// the beat lands the next cycle. A FIFO that is not full is read and written
// in one cycle (count' = count + push - pop).
void test_simultaneous_ingress_push_and_pop() {
    auto config = fast_bus_profile(8); // one 32 B beat per pop: a pop frees a whole beat's room
    config.lsu_ingress_sectors = 2;
    nanosm::Launch launch;
    launch.active_warps = 2;
    auto& gmem = image_bytes(launch.gmem_image, 4096);
    fill_pattern(gmem, 0, 1024, 0);
    std::string text = R"(
        sread r29, WARP_ID
        setp.ne p0, r29, 0
        @p0 bra.uni other
        mov r30, 0
        ldg.v8 r8, [r30]
        halt
    other:
    )";
    for (int i = 0; i < 100; ++i) text += "mov r2, 1\n";
    text += "halt\n";
    nanosm::Machine machine(config, compile(text, config), std::move(launch));
    machine.enable_edge_probe();
    std::map<uint64_t, bool> held; // cycle -> a read beat was held (FIFO full at the cycle's start)
    uint64_t full = 0;
    while (!machine.done()) {
        const auto cycle = machine.cycle();
        tick_capped(machine);
        held[cycle] = machine.trace().summary().lsu_ingress_full_cycles > full;
        full = machine.trace().summary().lsu_ingress_full_cycles;
    }
    std::map<uint64_t, std::pair<uint32_t, uint32_t>> moves; // cycle -> (pushes, pops)
    for (const auto& event : machine.edge_events()) {
        if (event.edge == LsuEdge::bus_read_beat) ++moves[event.cycle].first;
        if (event.edge == LsuEdge::ingress_read) ++moves[event.cycle].second;
    }
    bool both = false;
    bool registered = false;
    for (const auto& [cycle, move] : moves) {
        if (move.first != 0 && move.second != 0) both = true;
        // Full at the start of the cycle (a beat was held the cycle before
        // and this one), the read port pops: the beat still waits a cycle.
        if (move.second != 0 && held.count(cycle - 1) && held.at(cycle - 1) && held.at(cycle)) {
            registered = true;
            require(move.first == 0, "a beat entered the full FIFO in the cycle it was popped");
            require(moves.count(cycle + 1) && moves.at(cycle + 1).first != 0,
                "the held beat did not land the cycle after the pop");
        }
    }
    require(both, "no cycle read and wrote the ingress FIFO together");
    require(registered, "the FIFO was never full when its read port popped; the test proves little");
    require_vector_data(machine, 0, 8, 0, "ingress push/pop");
}

// ---- The driver shuffle (regression guard; the launches run under ctest
// via scripts/driver_shuffle_check.py) ----

// Register drivers read only Q and the wires, so permuting their order (and
// the independent request generators') every cycle changes nothing: not a
// cycle, a counter, a trace byte or a memory byte. A guard against a future
// driver that reads another's D or shared state, not a proof of independence.
void test_driver_shuffle_unit_kernels() {
    struct Scenario {
        std::string name;
        std::function<std::unique_ptr<nanosm::Machine>()> make;
    };
    const auto backpressure = [] {
        auto config = canonical_profile();
        config.tma_queue_commands = 0;
        config.tma_sector_credits = 0;
        config.gmem_queue_sectors = 8;
        config.lsu_ingress_sectors = 2;
        nanosm::Launch launch;
        launch.active_warps = 2;
        fill_pattern(image_bytes(launch.gmem_image, 4096), 0, 1024, 0);
        std::string text = "sread r29, WARP_ID\nsetp.ne p0, r29, 0\n@p0 bra.uni other\nmov r30, 0\nmov r31, 512\n"
            "mov r28, 2048\nldg.v8 r0, [r30]\nldg.v8 r8, [r31]\nldg r16, [r28]\nstg.v8 [r28], r0\nhalt\nother:\n";
        for (int i = 0; i < 60; ++i) text += "mov r3, 1\n";
        text += "halt\n";
        return std::make_unique<nanosm::Machine>(config, compile(text, config), std::move(launch),
            nanosm::TraceOptions{.timeline = true, .debug = true});
    };
    const auto tma_and_mma = [] {
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.shmem_image, 8192);
        auto& gmem = image_bytes(launch.gmem_image, 64);
        for (uint32_t i = 0; i < 16; ++i) write_u32(gmem, 4 * i, 0x100u + i);
        write_mbar(shmem, 4096, 1, 0);
        write_mbar(shmem, 4104, 0, 64);
        nanosm::TmaDescriptor descriptor;
        descriptor.dimensions = {16, 1, 1};
        descriptor.box = {16, 1, 1};
        descriptor.shmem_base = 2048;
        descriptor.completion_mbar = 4104;
        write_tma_descriptor(shmem, 6144, descriptor);
        return std::make_unique<nanosm::Machine>(canonical_profile(), compile(R"(
            .acc C, 0, 0, 64, 64
            mov r0, 0
            mov r1, 1024
            mov r4, 4096
            mov r8, 0
            mov r9, 0
            mov r10, 0
            mma.async.init C, [r0], [r1]
            mma.commit [r4]
            tma.async [6144], r8
            mma.async C, [r0], [r1]
            halt
        )"), std::move(launch), nanosm::TraceOptions{.timeline = true, .debug = true});
    };
    const auto acc_round_trip = [] {
        nanosm::Launch launch;
        launch.active_warps = 2;
        return std::make_unique<nanosm::Machine>(canonical_profile(), compile(R"(
            .acc C, 0, 0, 64, 64
            sread r0, THREAD_ID
            sread r8, WARP_ID
            shli r8, r8, 4
            mov r9, 0
            acc.st.v8 C, r8, r0
            acc.ld.v8 r16, C, r8
            fadd r24, r16, r17
            halt
        )"), std::move(launch), nanosm::TraceOptions{.timeline = true, .debug = true});
    };
    const auto scatter_and_tma_store = [] {
        nanosm::Launch launch;
        launch.active_warps = 2;
        auto& shmem = image_bytes(launch.shmem_image, 4096);
        fill_pattern(shmem, 1024, 64, 0);
        write_rank1_tma(shmem, 0, true, 2048, 64, 1024, 128);
        image_bytes(launch.gmem_image, 8192);
        return std::make_unique<nanosm::Machine>(canonical_profile(), compile(R"(
            sread r29, WARP_ID
            setp.ne p0, r29, 0
            @p0 bra.uni other
            mov r0, 0
            mov r1, 0
            mov r2, 0
            mov r3, 128
            tma.async [0], r0
        wait:
            mbar.try_wait p1, [r3], 0
            @!p1 bra.uni wait
            halt
        other:
            sread r4, LANE_ID
            shli r5, r4, 7
            stg [r5 + 4096], r4
            halt
        )"), std::move(launch), nanosm::TraceOptions{.timeline = true, .debug = true});
    };
    const std::vector<Scenario> scenarios = {
        {"vector loads under RF backpressure", backpressure},
        {"TMA load and MMA commit", tma_and_mma},
        {"acc.st / acc.ld round trip", acc_round_trip},
        {"scatter and TMA store", scatter_and_tma_store},
    };
    const auto fingerprint = [](nanosm::Machine& machine) {
        machine.run(20000);
        return std::make_tuple(machine.cycle(), machine.trace().summary_text(), machine.trace().perfetto_json(),
            machine.gmem(), machine.shmem());
    };
    for (const auto& scenario : scenarios) {
        auto reference_machine = scenario.make();
        const auto reference = fingerprint(*reference_machine);
        for (const uint64_t seed : {1ull, 2ull, 3ull, 0x9e3779b97f4a7c15ull}) {
            auto machine = scenario.make();
            machine->set_driver_shuffle_seed(seed);
            require(fingerprint(*machine) == reference,
                scenario.name + ": driver order changed the run (seed " + std::to_string(seed) + ")");
        }
    }
}

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"encoding round trip", test_encoding_round_trip},
        {"FNV-1a vectors", test_fnv1a_64_vectors},
        {"mbar word encoding", test_mbar_word_encoding},
        {"high register metadata", test_high_register_kernel_metadata},
        {"RF-derived residency", test_rf_capacity_determines_residency},
        {"compiler controls", test_compiler_controls},
        {"consumer-distance stalls", test_stalls_track_consumer_distance},
        {"RF read profile and plan", test_rf_read_profile_and_plan},
        {"RF deferred FADD and HALT", test_rf_deferred_fadd_and_halt},
        {"RF deferred WAR and branch", test_rf_deferred_war_and_branch},
        {"RF reservation retries", test_rf_reservation_retries},
        {"RF seeded competing warps", test_rf_seeded_competing_warps},
        {"integer ALU menu", test_integer_alu_menu},
        {"FA menu encoding round trips", test_fa_menu_encoding_round_trips},
        {"half conversion and pack vectors", test_half_conversion_and_pack_vectors},
        {"FA menu numeric vectors", test_fa_menu_numeric_vectors},
        {"SFU unit timing", test_sfu_unit_timing},
        {"fma bank discipline", test_fma_bank_discipline},
        {"mini softmax smoke", test_mini_softmax_smoke},
        {"SFU profile validation", test_sfu_profile_validation},
        {"tmem metadata and round trip", test_tmem_profile_metadata_and_round_trip},
        {"tmem ordering and store lifetime", test_tmem_transfer_ordering_and_store_lifetime},
        {"MMA raw-window numerics", test_mma_raw_window_numerics},
        {"MMA fusion boundaries", test_mma_fusion_boundaries},
        {"autonomous MMA GEMM and commit", test_autonomous_mma_gemm_and_commit},
        {"worker acc.store mbar handoff", test_worker_acc_store_mbar_handoff},
        {"mbar engines: TMA event does not stall MMA queue",
            test_mbar_engine_tma_event_does_not_stall_mma_queue},
        {"mbar engines: tie alternates, TMA first", test_mbar_engine_events_alternate_tma_first},
        {"mbar engines: real TMA and commit in one cycle",
            test_mbar_engine_real_tma_and_commit_same_cycle},
        {"MMA dense profile recurrence bubbles", test_mma_dense_profile_recurrence_bubbles},
        {"MMA bare transfer race panics", test_mma_bare_transfer_race_panics},
        {"MMA multi-round fetch timing", test_mma_multi_round_fetch_timing},
        {"MMA port collision sweep", test_mma_port_collision_sweep},
        {"MMA varying round service", test_mma_varying_round_service},
        {"sanitizer: sts into TMA store source", test_sanitizer_sts_into_tma_store_source},
        {"sanitizer: sts into MMA operand stage", test_sanitizer_sts_into_mma_operand_stage},
        {"sanitizer: lds from TMA load destination", test_sanitizer_lds_from_tma_load_destination},
        {"sanitizer: mbar RMW into owned range", test_sanitizer_mbar_rmw_into_owned_range},
        {"sanitizer: TMA/MMA accept conflicts", test_sanitizer_tma_and_mma_accept_conflicts},
        {"sanitizer: tmem ownership both directions", test_sanitizer_tmem_ownership_both_directions},
        {"sanitizer: double buffer and P tile clean", test_sanitizer_double_buffer_and_p_tile_are_clean},
        {"sanitizer: RMW port collision (C5)", test_sanitizer_rmw_port_collision},
        {"counted loop and relocation", test_counted_loop_and_label_relocation},
        {"uniform role dispatch", test_uniform_role_dispatch},
        {"nonuniform bra.uni PANIC", test_nonuniform_bra_uni_panics},
        {"branch drains async work", test_branch_boundary_drains_async_work},
        {"mbar sleep and wake", test_mbar_try_wait_sleeps_and_engine_completion_wakes},
        {"mbar test_wait predicate", test_mbar_test_wait_broadcasts_predicate},
        {"mbar try_wait immediate", test_mbar_try_wait_immediate_success},
        {"mbar try_wait timeout", test_mbar_try_wait_times_out_false},
        {"mbar deadline phase priority", test_mbar_phase_flip_wins_on_try_wait_deadline},
        {"mbar try_wait slots", test_mbar_try_wait_slots_limit_parked_warps},
        {"profile disables try_wait", test_profile_can_disable_mbar_try_wait},
        {"staged pipeline kernel", test_staged_pipeline_kernel},
        {"launch image validation", test_launch_image_rejects_bad_segment_hex},
        {"mbar warp handoff", test_warp_handoff_through_mbar},
        {"mbar uniform address", test_mbar_requires_uniform_address},
        {"mbar arrival underflow", test_mbar_arrival_underflow_panics},
        {"mbar reinit with waiter", test_mbar_reinit_with_waiter_panics},
        {"mbar wrong-phase completion", test_mbar_complete_wrong_phase_panics},
        {"mbar store repurpose", test_store_cannot_repurpose_mbar_with_waiter},
        {"vadd", test_vadd},
        {"sparse launch images", test_sparse_launch_images},
        {"lsu scalar ldg: one request per lane", test_lsu_scalar_ldg_is_one_request_per_lane},
        {"lsu v8 interleaved mapping", test_lsu_v8_interleaved_mapping},
        {"lsu v8 store source lifetime", test_lsu_v8_store_source_lifetime},
        {"gmem v8 predication rejection", test_gmem_v8_predication_is_rejected},
        {"lsu RF writes are full-width beats", test_lsu_rf_writes_are_full_width_beats},
        {"lsu contract panics", test_lsu_contract_panics},
        {"lsu commands stream through a small queue", test_lsu_commands_stream_through_small_queue},
        {"lsu command slot stalls issue", test_lsu_command_slot_structurally_stalls},
        {"lsu gather requests", test_lsu_gather_requests},
        {"lsu scatter byte enables", test_lsu_scatter_byte_enables},
        {"lsu staging pool admission", test_lsu_staging_pool_admission},
        {"lsu backpressure chain", test_lsu_backpressure_chain},
        {"tma port-A priority over sts", test_tma_port_a_priority_over_sts},
        {"lsu TMA/vector stream interleave", test_lsu_tma_and_vector_streams_interleave},
        {"lsu controller contract tripwire", test_lsu_controller_contract_tripwire},
        {"lsu profile and ldg.uni encoding", test_lsu_profile_and_encoding},
        {"lsu B2.1: per-edge widths", test_lsu_edge_widths},
        {"lsu B2.1: TMA store egress is pluck-bound", test_tma_store_egress_is_pluck_bound},
        {"lsu B2.1: no fall-through timelines", test_lsu_no_fall_through_timelines},
        {"lsu B2.1: FIFO latency fields", test_lsu_fifo_latency_fields},
        {"lsu B2.1: one write contract", test_lsu_one_write_contract},
        {"lsu B2.1: read_streams generalized", test_lsu_read_streams_generalized},
        {"tma descriptor encoding", test_tma_descriptor_encoding},
        {"tma xor64 transpose load/store", test_tma_xor64_transpose_load_store},
        {"tma xor64 feeds MMA", test_tma_xor64_feeds_mma_orientations},
        {"tma wide xor64 validation", test_tma_wide_xor64_validation},
        {"tma wide xor64 vs narrow", test_tma_wide_xor64_matches_narrow},
        {"tma wide xor64 sectors", test_tma_wide_xor64_sector_economics},
        {"xor64 byte primitive == fp16 formula", test_xor64_byte_primitive_matches_fp16_formula},
        {"tma fp32 xor64 validation", test_tma_fp32_xor64_validation},
        {"tma fp32 xor64 store/load", test_tma_fp32_xor64_store_and_load},
        {"tma rank-1 load and OOB", test_tma_rank1_load_and_oob_zero_fill},
        {"tma store masked sector", test_tma_store_preserves_other_sector_bytes},
        {"tma rank-3 strides", test_tma_rank3_strides},
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
        {"reg: hold, write, commit", test_reg_hold_write_commit},
        {"reg: duplicate write PANIC", test_reg_duplicate_write_panics},
        {"assembly contract 1: drain granted, refill at once", test_assembly_contract_drain_granted},
        {"assembly contract 2: drain denied by a booking", test_assembly_contract_drain_denied},
        {"assembly contract 3: no drain in the filling cycle", test_assembly_contract_fill_completes},
        {"assembly contract 4: wide pluck, no pass-through", test_assembly_contract_wide_pluck},
        {"assembly contract 5: final chunk reallocation", test_assembly_contract_final_chunk},
        {"assembly reuse: final load drain -> scalar store", test_assembly_reuse_load_to_scalar_store},
        {"assembly reuse: final load drain -> vector store", test_assembly_reuse_load_to_vector_store},
        {"assembly reuse: final load drain -> load", test_assembly_reuse_load_to_load},
        {"sanitizer: assembly reader takes only valid lanes", test_sanitizer_assembly_reader_takes_valid_lanes},
        {"mbar engines: events visible the next cycle", test_mbar_engine_events_visible_next_cycle},
        {"simultaneous: depbar completion and admission", test_simultaneous_depbar_completion_and_admission},
        {"simultaneous: full queue pop and push", test_simultaneous_full_queue_pop_and_push},
        {"simultaneous: ingress FIFO push and pop", test_simultaneous_ingress_push_and_pop},
        {"driver shuffle: unit kernels unchanged", test_driver_shuffle_unit_kernels},
    };
    // Tests that deliberately provoke a sanitizer PANIC (src/sanitizer.hpp).
    // A NANOSM_SANITIZER=0 build compiles the sanitizer to no-ops, so it skips
    // and counts them; every other test must pass there unchanged.
    const std::set<std::string> expects_sanitizer_panic = {
        "RF deferred WAR and branch",
        "tmem ordering and store lifetime",
        "MMA bare transfer race panics",
        "MMA port collision sweep",
        "sanitizer: sts into TMA store source",
        "sanitizer: sts into MMA operand stage",
        "sanitizer: lds from TMA load destination",
        "sanitizer: mbar RMW into owned range",
        "sanitizer: TMA/MMA accept conflicts",
        "sanitizer: tmem ownership both directions",
        "sanitizer: RMW port collision (C5)",
        "lsu controller contract tripwire",
        "lsu B2.1: read_streams generalized",
        "missing wait panic",
        "reg: duplicate write PANIC",
        "sanitizer: assembly reader takes only valid lanes",
    };
    for (const auto& name : expects_sanitizer_panic) {
        if (std::none_of(tests.begin(), tests.end(), [&](const auto& test) { return test.first == name; })) {
            nanosm::err("FAIL sanitizer-panic tag names no test: {}", name);
            return 1;
        }
    }
    std::size_t passed = 0;
    std::size_t skipped = 0;
    for (const auto& [name, test] : tests) {
#if !NANOSM_SANITIZER
        if (expects_sanitizer_panic.contains(name)) {
            nanosm::log("SKIP {} (expects a sanitizer panic; NANOSM_SANITIZER=0)", name);
            ++skipped;
            continue;
        }
#endif
        try {
            test();
            nanosm::log("PASS {}", name);
            ++passed;
        } catch (const std::exception& error) {
            nanosm::err("FAIL {}: {}", name, error.what());
        }
    }
    if (skipped == 0) {
        nanosm::log("{}/{} tests passed", passed, tests.size());
    } else {
        nanosm::log("{}/{} tests passed ({} skipped: they expect a sanitizer panic)",
            passed, tests.size() - skipped, skipped);
    }
    return passed + skipped == tests.size() ? 0 : 1;
}

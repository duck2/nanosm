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

// I2 / I2.1 / I3 (rfc12 sections 2 and 6): an RMW's write is granted D + Lb
// + C cycles after its read (the read's words reach the port output, then the
// update crosses the C ALU stages). Its "mbar RMW" slice runs from its START
// (I3: the store pause begins) to its write grant; uncontended -- no store or
// A-read transaction under way -- the read is granted in the start cycle, so
// the engine is busy D + Lb + C + 1 cycles (2 before I2; D + Lb + 1 in I2,
// which had no ALU stage). Back-to-back services are that far apart.
uint64_t rmw_busy_cycles(const nanosm::DeviceProfile& profile = canonical_profile()) {
    return profile.shmem_port_a_depth + profile.shmem_read_latency_cycles + profile.shmem_rmw_compute_cycles + 1;
}

nanosm::Kernel compile(
    std::string_view source, const nanosm::DeviceProfile& profile = canonical_profile()) {
    return nanosm::compile_and_assemble(source, profile);
}

// Hand assembly: lines without a {...} control block get the default one.
std::string i1_asm(const nanosm::DeviceProfile& profile, const std::string& body) {
    std::string text = ".profile " + profile.name + ", 0x" +
        nanosm::format_profile_fingerprint(nanosm::device_profile_fingerprint(profile)) + "\n";
    std::istringstream lines(body);
    std::string line;
    while (std::getline(lines, line)) {
        const auto first = line.find_first_not_of(" \t");
        if (first == std::string::npos) continue;
        text += line;
        if (line[first] != '.' && line.find('{') == std::string::npos) text += " {pred=PT, stall=0, set=-, wait=-}";
        text += "\n";
    }
    return text;
}

// Hand assembly with the fixed-latency stalls its register and predicate
// dependencies need under `profile` (src/rf.hpp, the distances qtxc
// schedules with): every authored stall is kept when it is at least that,
// raised when not; depbars and waits stay as written (I6a).
nanosm::Kernel hand_assemble(const std::string& text, const nanosm::DeviceProfile& profile) {
    return nanosm::schedule_fixed_stalls(nanosm::assemble_asm(text, profile), profile);
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
    // I3 (rfc12 section 8): the tx field is a signed 32-bit count (two's
    // complement), same bits 32-63: 0x89abcdef is a negative count.
    state.tx_bytes = static_cast<int32_t>(0x89abcdefu);
    const auto word = nanosm::encode_mbar_state(state);
    require(
        word == 0x89abcdef468a48d3ull,
        "mbar word layout changed without updating its architectural encoding test");
    const auto decoded = nanosm::decode_mbar_state(word);
    require(
        decoded.phase == state.phase && decoded.valid == state.valid &&
            decoded.expected == state.expected && decoded.remaining == state.remaining &&
            decoded.tx_bytes == state.tx_bytes && decoded.tx_bytes < 0,
        "mbar word encoding did not round trip");
    state.tx_bytes = -128; // complete_tx ran 128 bytes ahead of its expect_tx
    require(nanosm::encode_mbar_state(state) >> 32 == 0xffffff80u &&
            nanosm::decode_mbar_state(nanosm::encode_mbar_state(state)).tx_bytes == -128,
        "a negative tx count did not round trip");
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
    // I6a (src/rf.hpp): a fixed result is visible issue_to_visible after its
    // issue; its consumer samples issue_to_sample (1, the issue register)
    // after its own. The mov's result feeds the very next lds (7 - 1 = 6 at
    // canonical); the fadd's the sts after it (10 - 1 = 9).
    const auto& profile = canonical_profile();
    const auto distance = [&](const nanosm::Instruction& producer) {
        return nanosm::issue_to_visible(producer, profile) - nanosm::issue_register_stages;
    };
    require(program.instructions[0].control.stall == distance(program.instructions[0]),
        "mov did not cover the lds right after it");
    require(program.instructions[1].control.set_depbar == 0, "first load did not allocate sb0");
    require(program.instructions[2].control.set_depbar == 1, "second load did not allocate sb1");
    require(program.instructions[3].control.wait_mask == 0b000011, "fadd did not wait for both loads");
    require(program.instructions[3].control.stall == distance(program.instructions[3]),
        "fadd did not cover its one-away consumer");
    require(program.instructions[4].control.set_depbar == 0, "store did not reuse a released depbar");
    require(program.instructions[5].control.wait_mask == 0b000001, "halt did not wait for store completion");
}

void test_stalls_track_consumer_distance() {
    // fadd's result is consumed three instructions later, so only what its
    // distance (src/rf.hpp: issue_to_visible - the consumer's issue_to_sample)
    // exceeds those three issues is bought with a stall, and it lands just
    // before the consumer. The mov feeding the fadd stalls the full distance.
    const auto& profile = canonical_profile();
    const auto program = nanosm::compile_qtx(R"(
        mov  r0, 1
        fadd r1, r0, r0
        mov  r2, 5
        mov  r3, 6
        add  r4, r1, r2
        halt
    )", profile);
    const auto& fadd = program.instructions[1];
    const auto fadd_distance = nanosm::issue_to_visible(fadd, profile) - nanosm::issue_register_stages;
    require(program.instructions[0].control.stall ==
            nanosm::issue_to_visible(program.instructions[0], profile) - nanosm::issue_register_stages,
        "mov did not cover the fadd right after it");
    require(fadd.control.stall == 0, "fadd stalled its own latency");
    require(program.instructions[2].control.stall == 0, "independent mov stalled");
    require(program.instructions[3].control.stall == fadd_distance - 2, "pre-consumer stall wrong");
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
    // The fadd issued one cycle before the branch; the block after it must
    // issue no earlier than one cycle before the fadd's result is visible (its
    // first sample is after the issue register): issue_to_visible - 1 cycles
    // after the fadd, issue_to_visible - 2 after the branch (src/rf.hpp).
    require(fixed.instructions[2].control.stall ==
            nanosm::issue_to_visible(fixed.instructions[1], canonical_profile()) -
                nanosm::issue_register_stages - 1,
        "branch did not drain a live fadd writeback");
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

    // A guard only: the warp parks once its init, arrive and probe have
    // crossed issue, dispatch (I6a) and the RMW engine.
    while (machine.trace().summary().mbar_sleeps == 0 && machine.cycle() < 200) machine.tick();
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
        // I2: read grant to write grant, D + Lb + 1 cycles (was 2); I2.1: + C.
        require(event.duration_cycles == rmw_busy_cycles(), "mbar RMW did not hold A for D + Lb + C + 1 cycles");
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

// I2: the wake comes when the RMW's write reaches the RAM (rfc12 section 6),
// 2D + Lb (I2.1: + C) cycles after the event's claim, which is the cycle after
// the park. The timeout puts the deadline in that cycle (was 2: claim, then
// write).
void test_mbar_phase_flip_wins_on_try_wait_deadline() {
    auto profile = canonical_profile();
    // The wake lands exactly on the deadline: the claim is the cycle after the
    // park, the wake the RMW write's commit, D + Lb + C + D later (I2.1: + C,
    // the RMW's ALU stages).
    profile.mbar_try_wait_timeout_cycles = 2 * profile.shmem_port_a_depth + profile.shmem_read_latency_cycles +
        profile.shmem_rmw_compute_cycles + 1;
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

// I3 (rfc12 section 8): complete_tx carries no phase in hardware; the phase
// a sender meant (here the host's stated parity) is sanitizer metadata, so
// this is now a sanitizer PANIC (listed in expects_sanitizer_panic).
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
    // (The non-uniform bases moved to "uniformity: hardware uses lane 0":
    // since I1 they are sanitizer panics, and this test runs in both builds.)
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
    // I6a: the second ldg.v8 issues in the cycle the first command's eighth RF
    // write releases the register (its grant: first.end) and reaches the LSU
    // issue_to_dispatch later (the command slice starts at dispatch).
    const auto dispatch = nanosm::issue_to_dispatch(canonical_profile());
    const auto [first, second, no_staging, no_slot] = run(1);
    require(second.start == first.end + dispatch,
        "with one assembly register the second ldg.v8 was admitted at cycle " + std::to_string(second.start) +
            ", not at the first command's eighth write (" + std::to_string(first.end) + ") + dispatch");
    // Warp 1 is tried every cycle from its ldg.v8 until admission.
    require(no_staging + 4 >= second.start - first.start,
        "no_staging did not count the ineligible cycles");
    require(no_staging != 0 && no_slot == 0, "the wait was not attributed to the staging pool");
    const auto [first2, second2, no_staging2, no_slot2] = run(2);
    require(second2.start <= first2.start + 2 && no_staging2 == 0 && no_slot2 == 0,
        "with two assembly registers the second ldg.v8 was not admitted at once");
}

// The RF-starvation setup of the LSU backpressure tests. Before I1 a stream
// of movs (latency 1) booked an async writer's bank every cycle and starved
// it indefinitely. Since I1 (trash/rfc10.md section 3) one denial with service
// owed inhibits new bookings to the bank, so a ready async writer waits at
// most for the bookings already made. To hold one for a while, the stream
// books booking_horizon cycles ahead: fadds whose result reaches the write
// port that far after their issue (I6a: the issue register, Lr and the fadd
// latency set by with_long_bookings, src/rf.hpp), one per cycle, rotating
// over booking_horizon destination registers of `bank`. booking_latency is
// as long as the five-bit stall field allows at canonical RF depths (window
// 1, Lr 2, Wd 2: 31 - 2 - 2 - 1 = 26), less a margin.
constexpr uint32_t booking_latency = 24;
constexpr uint32_t booking_horizon = 1 + 2 + booking_latency; // issue register + canonical Lr + latency

void with_long_bookings(nanosm::DeviceProfile& config) {
    config.timing.fadd = booking_latency;
}

std::string booking_stream(uint32_t bank, uint32_t count) {
    std::string text;
    for (uint32_t i = 0; i < count; ++i) {
        text += "fadd r" + std::to_string(bank + 3 * (i % booking_horizon)) + ", r1, r1\n";
    }
    return text;
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
        with_long_bookings(config);
        nanosm::Launch launch;
        launch.active_warps = 2;
        auto& gmem = image_bytes(launch.gmem_image, 4096);
        fill_pattern(gmem, 0, 1024, 0);
        // Warp 1 books RF bank 0 every cycle, booking_horizon cycles ahead
        // (I1: a ready assembly then waits out up to that many old bookings
        // per denial; a mov storm no longer starves it).
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
        if (storm) text += booking_stream(0, 500);
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
    // I1: each of the six chunks bound for bank 0 (r0, r3, r6, r9, r12, r15)
    // waits out at most the bookings already made when it was denied (the
    // owed bit then inhibits warp 1): blocked, but never starved.
    require(summary.lsu_rf_write_wait_cycles >= 5 * booking_horizon, "the RF write port was never the blocked link");
    require(summary.lsu_rf_write_wait_cycles <= 6 * (booking_horizon + 1), "an assembly waited past the booking horizon");
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
        // I2 (rfc12 section 2): a sector's source read reaches the one TMA
        // store sector register D + Lb cycles after its grant, and the
        // register holds the sector until its last word is in the egress
        // FIFO, so each of the 512 sectors costs D + Lb plus its 8 / pluck
        // drain cycles (before I2: the drain alone, 16384 / (4 * pluck)).
        const uint64_t bound = 512 * (config.shmem_port_a_depth + config.shmem_read_latency_cycles + 8 / pluck);
        require(machine.cycle() >= bound && machine.cycle() <= bound + 100,
            "16 KiB TMA store at pluck " + std::to_string(pluck) + " took " + std::to_string(machine.cycle()) +
                " cycles; the store register's round trip bounds it at " + std::to_string(bound));
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
    // I6a: the beat sampled at t is at the RF output, filling the assembly
    // register, at t + Lr; the egress port reads it (Q) one cycle later.
    require(all_equal(t.store_rf_to_egress, canonical_profile().rf_read_latency_cycles + 1),
        "stg.v8: RF read beat -> first egress word is not Lr + 1 cycles");
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
        // under PV (transpose=0) ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â V's rows are keys.
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
        // Hold A / stream B on a 4x4x8 core: the step that reads an A block
        // reads A and B (two bank rounds), the other fifteen read one B block
        // (one round), and each of the 16 A blocks is read once per command.
        require(summary.mma_commands == 2 && summary.mma_input_steps == 2 * 256 &&
                summary.mma_a_block_reads == 2 * 16 &&
                summary.mma_shmem_rounds == summary.mma_input_steps + summary.mma_a_block_reads,
            "TMA-authored operands did not keep one bank round per B block and per A block");
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
    // Re-pinned in B3: was `mov r0, 0` (all sixteen lanes on one word, one
    // round by the old same-word merge). Without the merge that is sixteen
    // rounds (sixteen lanes on bank 0), which holds port A and would block the
    // second lds structurally. Lane-strided addresses keep each lds at one
    // round (one lane per bank), the premise of this test.
    const auto program = nanosm::compile_qtx(R"(
        sread r0, LANE_ID
        shli r0, r0, 2
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

// Re-pinned in B3 (was "broadcast", 1 round): sixteen lanes reading one word
// were merged into one round. Uniformity is now declared, never detected:
// they are sixteen lanes on bank 0, served one per round, 16 rounds. Every
// lane still gets the word. The one-round broadcast is lds.uni (B3 tests).
void test_plain_lds_of_one_word_is_sixteen_rounds() {
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
    require(machine.trace().summary().shmem_rounds == 16,
        "sixteen lanes on one word did not take sixteen bank rounds");
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

// An issue event (the instruction category also carries the debug trace's
// "RF read" samples and "RF write" port presentations).
bool is_issue_event(const nanosm::TimelineEvent& event) {
    return event.category == nanosm::TraceCategory::instruction && event.name != "RF read" &&
        event.name != "RF write";
}

std::vector<uint64_t> issued_at(const nanosm::Machine& machine, uint32_t pc) {
    std::vector<uint64_t> result;
    for (const auto& event : machine.trace().events()) {
        if (is_issue_event(event) && event.id == pc) result.push_back(event.start_cycle);
    }
    return result;
}

// Every issue, in order: " c<cycle>:w<warp>:<pc>:<opcode>" (a debugging aid
// for failure messages).
std::string issue_log(const nanosm::Machine& machine) {
    std::string text;
    for (const auto& event : machine.trace().events()) {
        if (!is_issue_event(event)) continue;
        const auto warp = event.detail.substr(5, event.detail.find(',') - 5);
        text += " c" + std::to_string(event.start_cycle) + ":w" + warp + ":" + std::to_string(event.id) + ":" + event.name;
    }
    return text;
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

// trash/rfc14.md section 3: every instruction reads its operands in ONE
// round, sampled at issue + issue_to_sample, so the profile has no read
// window. A profile that still names rf.arithmetic_read_cycles is refused,
// whatever the value (the old default 1 included); the canonical profile has
// none and holds about as much RF as shmem (352 rows per bank). A round reads
// each distinct source once, in source order.
void test_rf_read_profile_and_round() {
    const auto& profile = canonical_profile();
    std::ifstream input("profiles/nanosm-v0.json");
    nlohmann::json document;
    input >> document;
    require(!document["rf"].contains("arithmetic_read_cycles"), "the canonical profile still names a read window");
    require(profile.rf_rows_per_bank == 352 &&
            uint64_t{profile.rf_banks} * profile.rf_rows_per_bank * profile.lanes * 4 == 66 * 1024,
        "the canonical RF is not 3 x 352 rows of 64 B (66 KiB)");
    for (const auto value : {nlohmann::json(1), nlohmann::json(3)}) {
        auto old = document;
        old["rf"]["arithmetic_read_cycles"] = value;
        try {
            (void)nanosm::parse_device_profile(old.dump());
            throw std::runtime_error("a profile naming rf.arithmetic_read_cycles was accepted");
        } catch (const nanosm::ProfileError& error) {
            require(std::string(error.what()).find("rf.arithmetic_read_cycles was removed") != std::string::npos,
                std::string("the removed field's refusal is unclear: ") + error.what());
        }
    }
    require(nanosm::rf_read_round({0, 3}) == std::vector<uint8_t>{0, 3}, "a round lost a source");
    require(nanosm::rf_read_round({4, 4}) == std::vector<uint8_t>{4}, "a repeated register was read twice");
    require(nanosm::rf_read_round({5, 2, 5, 8}) == std::vector<uint8_t>{5, 2, 8}, "a round's order or broadcast");
    require(nanosm::issue_to_sample == nanosm::issue_register_stages &&
            nanosm::issue_to_operands(profile) == nanosm::issue_to_sample + profile.rf_read_latency_cycles &&
            nanosm::issue_to_dispatch(profile) == nanosm::issue_to_operands(profile),
        "every instruction's operands are not at the RF output issue + 1 + Lr");
}

// One round (src/rf.hpp): the fadd's two sources (r0 in bank 0, r1 in bank
// 1) are both sampled at issue + issue_to_sample; its operands are at the RF
// output Lr later, its result at the write port fadd later, landing at the
// edge Wd after that -- read_reg sees it after that cycle's tick. The
// source-free mov issues right behind it. setp p0, r0, r0 issues at c2 (its
// word fetched at c0: fetch_to_issue, src/rf.hpp) and reads r0 once, at c3;
// its predicate reaches the file at its port cycle c2 + 1 + Lr + alu (c6), at
// that edge, and done() waits for it: the run ends at c7.
void test_rf_one_round_fadd_and_halt() {
    const auto& profile = canonical_profile();
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& image = image_bytes(launch.shmem_image, 8);
    write_u32(image, 0, nanosm::float_bits(1.25f));
    write_u32(image, 4, nanosm::float_bits(2.5f));
    nanosm::Machine machine(profile, compile(R"(
        lds r0, [r2]
        lds r1, [r2+4]
        fadd r4, r0, r1
        mov r8, 9
        halt
    )", profile), std::move(launch), {.timeline = true, .debug = true});
    uint64_t changed = 0;
    while (!machine.done() && machine.cycle() < 1000) {
        const auto tick = machine.cycle();
        machine.tick();
        if (machine.read_reg(0, 0, 4) != 0 && changed == 0) changed = tick;
    }
    require(machine.done(), "the FADD pipeline did not drain");
    const auto issue = issued_at(machine, 16).at(0);
    require(changed == issue + nanosm::issue_to_sample + profile.rf_read_latency_cycles + profile.timing.fadd +
                profile.rf_write_depth_cycles,
        "accepted FADD latency varied");
    require(machine.read_reg(0, 0, 4) == nanosm::float_bits(3.75f), "the FADD lost its operands");
    std::vector<std::pair<uint64_t, uint64_t>> captures;
    for (const auto& event : machine.trace().events()) {
        if (event.name == "RF read" && event.start_cycle > issue && (event.id == 0 || event.id == 1))
            captures.emplace_back(event.start_cycle, event.id);
    }
    require(captures == std::vector<std::pair<uint64_t, uint64_t>>{{issue + 1, 0}, {issue + 1, 1}},
        "the FADD's sources were not both sampled at issue + 1");
    require(issued_at(machine, 24).at(0) == issue + 1, "the source-free MOV did not follow the FADD");
    check_read_capacity(machine, profile);

    nanosm::Launch predicate_launch;
    predicate_launch.active_warps = 1;
    nanosm::Machine predicate(profile, compile("setp.eq.u32 p0, r0, r0\nhalt", profile),
        std::move(predicate_launch), {.timeline = true, .debug = true});
    predicate.run();
    require(predicate.cycle() == nanosm::fetch_to_issue + nanosm::issue_to_sample + profile.rf_read_latency_cycles +
                profile.timing.alu + 1,
        "HALT dropped a pending predicate-only read pipeline");
    uint32_t duplicate_reads = 0;
    for (const auto& event : predicate.trace().events()) {
        if (event.name == "RF read") {
            require(event.start_cycle == nanosm::fetch_to_issue + nanosm::issue_to_sample,
                "the repeated operand was not sampled at c3");
            ++duplicate_reads;
        }
    }
    require(duplicate_reads == 1, "a repeated operand consumed several RF reads");
}

// One round ends a reader's samples at issue + issue_to_sample, before the
// earliest any later write of the warp can land (earliest_rf_visible: 7 at
// canonical). So qtxc buys no WAR stall, neither for a fixed writer nor for
// an async one, and the reader captures the old value. (With a read window
// of 8, before RFC14, the async writer needed one.) A counted loop in four
// warps still drains its fixed reads and predicates at every branch.
void test_rf_one_round_war_and_branch() {
    const auto& profile = canonical_profile();
    require(nanosm::issue_to_sample + 1 < nanosm::earliest_rf_visible(profile), "a younger write could land first");
    const auto fixed_writer = nanosm::compile_qtx("add r2, r0, r1\nmov r1, 7\nhalt", profile);
    require(fixed_writer.instructions[0].control.stall == 0, "qtxc bought a WAR stall a fixed writer cannot need");
    const auto source = "add r2, r0, r1\nlds r1, [r5]\nhalt";
    const auto async_writer = nanosm::compile_qtx(source, profile);
    require(async_writer.instructions[0].control.stall == 0, "qtxc bought a WAR stall an async writer cannot need");
    nanosm::Launch launch;
    launch.active_warps = 1;
    write_u32(image_bytes(launch.shmem_image, 64), 0, 7);
    nanosm::Machine war(profile, compile(source, profile), std::move(launch));
    war.run();
    require(war.read_reg(0, 0, 2) == 0 && war.read_reg(0, 0, 1) == 7,
        "the reader did not capture the old r1, or the lds did not land");

    nanosm::Launch loop_launch;
    loop_launch.active_warps = 4;
    nanosm::Machine loop(profile, compile(R"(
        mov r0, 0
        mov r4, 9
    loop:
        addi r0, r0, 1
        setp.lt.u32 p0, r0, r4
        @p0 bra.uni loop
        halt
    )", profile), std::move(loop_launch), {.timeline = true, .debug = true});
    loop.run();
    for (uint32_t warp = 0; warp < 4; ++warp)
        require(loop.read_reg(warp, 0, 0) == 9, "branch did not drain fixed read/predicate state");
    check_read_capacity(loop, profile);
}

// Seeded random adds and xoris over twelve registers, in one, four and
// sixteen warps and two or three banks: every add's sources are one
// register twice or two registers in different banks (one round, one read
// port per bank). Every warp ends with the sequential result, and no cycle
// samples a bank more than its ports allow.
void test_rf_seeded_competing_warps() {
    for (const uint32_t banks : {2u, 3u}) {
        for (const uint32_t warps : {1u, 4u, 16u}) {
            auto profile = canonical_profile();
            profile.rf_banks = banks;
            std::string source;
            std::array<uint32_t, 12> expected{};
            for (uint32_t r = 0; r < expected.size(); ++r) {
                expected[r] = r + 1;
                source += "mov r" + std::to_string(r) + ", " + std::to_string(r + 1) + "\n";
            }
            uint32_t seed = banks * 100 + warps;
            const auto random = [&]() { seed = seed * 1664525u + 1013904223u; return seed; };
            for (uint32_t i = 0; i < 400; ++i) {
                const auto dst = random() % expected.size();
                const auto a = random() % expected.size();
                std::vector<uint32_t> other_banks;
                for (uint32_t r = 0; r < expected.size(); ++r) {
                    if (r % banks != a % banks) other_banks.push_back(r);
                }
                const auto b = i % 3 == 0 ? a : other_banks[random() % other_banks.size()];
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
    // 4x4x8 core: 256 steps per command; each command reads its 16 A blocks
    // once, each with its first B block (two rounds), and 240 B-only steps
    // (one round each): 16 x 17 = 272 rounds per command.
    require(summary.mma_commands == 64 && summary.mma_input_steps == 64 * 256 &&
            summary.mma_result_steps == 64 * 256 && summary.mma_a_block_reads == 64 * 16 &&
            summary.mma_shmem_rounds == 64 * 272 && summary.mma_recurrence_stalls == 0,
        "canonical MMA trace did not show hold-A / stream-B steps at one round per B block");
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
    int32_t tx_bytes) {
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
// the tie at r - 1 + busy goes to the MMA.) I2: services are busy = D + Lb + 1
// cycles apart (was 2; I2.1: D + Lb + C + 1); the commit, held from r + 1, waits busy - 2 cycles
// for T1 and T2 waits busy cycles for the commit (was 0 + 2).
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
    // predecessor retires, and frees the queue slot the mma.async waits for:
    // the mma.async issues in r and reaches the MMA queue at its dispatch,
    // accept = r + issue_to_dispatch (I6a), fetching from the next cycle.
    nanosm::Machine reference(canonical_profile(), kernel, make_launch(), {.timeline = true});
    reference.run(5000);
    const auto [accept, fetch] = queued_accept_and_fetch(reference);
    const auto r = accept - nanosm::issue_to_dispatch(canonical_profile());
    require(r > 2 && fetch == accept + 1, "reference MMA queue timing changed; re-derive the test");
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

    require(queued_accept_and_fetch(machine) == std::pair{accept, fetch},
        "a held TMA event stalled the MMA queue behind the commit");
    const auto busy = rmw_busy_cycles();
    require(mbar_rmw_services(machine) == std::vector<RmwService>{
                {r - 1, "complete_tx (tma), address=4104"},
                {r - 1 + busy, "arrive (mma.commit), address=4096"},
                {r - 1 + 2 * busy, "complete_tx (tma), address=4104"}},
        "engine events were not served once each, alternating TMA, MMA, TMA");
    require(machine.trace().summary().mbar_engine_waits_on_other == 2 * busy - 2,
        "the commit should wait out T1's RMW from r + 1, and T2 the whole of the commit's");
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
    machine.tick(); // the TMA event's RMW is under way; the commit still waits
    machine.complete_mbar_tx(tx_mbar, 64, 0); // pending again from first + 2, before the next free point
    machine.run(1000);

    // I2: services are busy = D + Lb + 1 cycles apart (was 2); I2.1: + C.
    const auto busy = rmw_busy_cycles();
    require(mbar_rmw_services(machine) == std::vector<RmwService>{
                {first, "complete_tx (tma), address=4104"},
                {first + busy, "arrive (mma.commit), address=4096"},
                {first + 2 * busy, "complete_tx (tma), address=4104"},
                {first + 3 * busy, "arrive (mma.commit), address=4112"}},
        "tied engine events did not alternate TMA, MMA, TMA, MMA");
    // Each of the three ties holds the loser through all of the winner's RMW.
    require(machine.trace().summary().mbar_engine_waits_on_other == 3 * busy,
        "engine_waits_on_other does not count the winner's whole RMW per tie");
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
    // The commit presents in r; the mma.async behind it issues then and
    // reaches the queue at its dispatch (I6a).
    const auto r = reference.accept - nanosm::issue_to_dispatch(canonical_profile());
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
        require(observed.accept == reference.accept, "the TMA's completion delayed the MMA queue" + where);
        require(observed.services.size() == 2, "an engine event was lost" + where);
        const auto tma_claim = observed.services[0].second == tma_service.second
            ? observed.services[0].first : observed.services[1].first;
        require(observed.tma_end == tma_claim + 1,
            "the TMA command outlived its claimed event (held by the MMA's?)" + where);
        if (tma_claim != r + 1) continue; // the TMA presented in r too
        aligned = true;
        const auto busy = rmw_busy_cycles(); // I2: D + Lb + 1 (was 2); I2.1: + C
        require(observed.services ==
                std::vector<RmwService>{at(tma_service, r + 1), at(mma_service, r + 1 + busy)},
            "same-cycle TMA completion and commit were not served TMA, then MMA" + where);
        require(observed.waits == busy, "the commit should wait out the TMA's RMW" + where);
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
    // Case 03/05/06: one bank makes every 4x4x4 step a 16-round fetch (with
    // one N block every step reads its A block too). A step granted its first
    // round at t is granted its last at t+15; that round reaches the RAM at
    // t+15+D and the port output at t+15+D+Lb, where the step enters the core
    // (the port-B read pipeline has port A's D and Lb); its result lands at
    // entry + latency; the next step's first round is granted at t+16; and the
    // operand-consumed depbar clears at the LAST PHYSICAL READ of the final
    // step: its last round's RAM sample.
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
    const auto d = profile.shmem_port_a_depth;
    const auto lb = profile.shmem_read_latency_cycles;
    require(first_result == inputs[0] + 15 + d + lb + 12,
        "the result did not launch from core entry (last round + D + Lb) plus the pipeline latency");
    require(depbar_cleared == inputs[7] + 15 + d,
        "operand-consumed depbar did not clear at the final physical read (the last round's sample)");
    require(machine.trace().summary().mma_recurrence_stalls == 0,
        "16-round spacing already covers the result latency; no bubbles expected");
    for (uint32_t lane = 0; lane < profile.lanes; ++lane)
        require(machine.read_reg(0, lane, 10) == nanosm::float_bits(8.0f),
            "per-round operand capture changed the numerical result");
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
    require(inputs.size() == 256, "one canonical-shape command is 256 core steps (4x4x8 core)");
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

// ---- I7: tensor core 4x4x8, hold A / stream B (trash/rfc11.md section 0) ----

// Canonical xor64 operand addresses, restated independently of the
// production helper: A element (m, k) and B element (k, n) of 64x64x8 tiles
// at 0 and 1024.
uint32_t i7_a_address(uint32_t m, uint32_t k) {
    return 4 * (4 * m + ((k >> 1) ^ ((m >> 2) & 1))) + 2 * (k & 1);
}
uint32_t i7_b_address(uint32_t k, uint32_t n) {
    return 1024 + 4 * (4 * n + ((k >> 1) ^ ((n >> 2) & 1) ^ 2)) + 2 * (k & 1);
}

// One 64x64x8 tile pair (A = a_k on every row, B = b_k on every column) and
// `commands` MMA commands into one accumulator, then commit, wait, acc.ld.
std::unique_ptr<nanosm::Machine> i7_machine(const std::array<uint16_t, 8>& a,
    const std::array<uint16_t, 8>& b, uint32_t commands) {
    const auto& profile = canonical_profile();
    std::ostringstream qtx;
    qtx << ".acc C, 0, 0, 64, 64\nmov r0, 0\nmov r1, 1024\nmov r4, 4096\nmov r8, 0\nmov r9, 0\n"
        << "mbar.init [r4], 1\nmma.async.init C, [r0], [r1]\n";
    for (uint32_t command = 1; command < commands; ++command) qtx << "mma.async C, [r0], [r1]\n";
    qtx << "mma.commit [r4]\nwait:\nmbar.try_wait p0, [r4], 0\n@!p0 bra.uni wait\n"
        << "acc.ld r10, C, r8\nhalt\n";
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.shmem_image, 8192);
    for (uint32_t m = 0; m < 64; ++m)
        for (uint32_t k = 0; k < 8; ++k) write_u16(shmem, i7_a_address(m, k), a[k]);
    for (uint32_t k = 0; k < 8; ++k)
        for (uint32_t n = 0; n < 64; ++n) write_u16(shmem, i7_b_address(k, n), b[k]);
    return std::make_unique<nanosm::Machine>(profile, compile(qtx.str(), profile), std::move(launch),
        nanosm::TraceOptions{.timeline = true, .debug = true});
}

// A 4x4x8 step is ONE fused group of eight products plus C (rfc11 section 0),
// not two successive four-product groups. These eight products tell them
// apart: 4096 * 4096 + 1 * 1 is 2^24 + 1, which a four-product group
// truncates to 2^24 in FP32 before the second group's -2^24 cancels it (0);
// one eight-product group keeps the unit (1.0).
void test_i7_dot8_group_boundary() {
    const std::array<uint16_t, 8> a{0x6c00, 0x3c00, 0, 0, 0xec00, 0, 0, 0}; // 4096, 1, 0, 0, -4096, 0...
    const std::array<uint16_t, 8> b{0x6c00, 0x3c00, 0, 0, 0x6c00, 0, 0, 0}; // 4096, 1, 0, 0, 4096, 0...
    const auto eight = nanosm::mma_dot_f16(a, b, 0);
    const auto first_four = nanosm::mma_dot_f16(std::span(a).first(4), std::span(b).first(4), 0);
    const auto two_fours = nanosm::mma_dot_f16(std::span(a).last(4), std::span(b).last(4), first_four);
    require(eight == 0x3f800000u && first_four == 0x4b800000u && two_fours == 0,
        "the eight-product group and two four-product groups do not differ as derived");
    require(canonical_profile().mma_core_k == 8, "the canonical MMA core is not 4x4x8");
    auto machine = i7_machine(a, b, 1);
    machine->run(20000);
    for (uint32_t lane = 0; lane < 16; ++lane)
        require(machine->read_reg(0, lane, 10) == 0x3f800000u,
            "the canonical 4x4x8 core did not compute one eight-product group");
}

// One canonical command, step by step. Hold A / stream B: for each of the
// 16 A blocks (m0 = 4i), every B block (n0 = 4j) in order; the step with
// j = 0 reads its A block and its B block (2 bank rounds), the other 15 read
// one B block (1 round) and use the held A block. Port B grants one round
// per cycle and a step its first round the cycle after the previous step's
// last, so step s + 1 starts rounds(s) cycles after step s. A step whose
// last round is granted at g enters the core at g + D + Lb and writes tmem
// result_latency later.
void test_i7_hold_a_stream_b_walker() {
    const auto& profile = canonical_profile();
    std::array<uint16_t, 8> a{}, b{};
    a.fill(0x3c00); // 1
    b.fill(0x4000); // 2
    auto machine = i7_machine(a, b, 1);
    machine->run(20000);
    for (uint32_t lane = 0; lane < 16; ++lane)
        require(machine->read_reg(0, lane, 10) == nanosm::float_bits(16.0f),
            "hold-A walker computed the wrong value");
    std::vector<const nanosm::TimelineEvent*> inputs, writes;
    for (const auto& event : machine->trace().events()) {
        if (event.name == "MMA input step") inputs.push_back(&event);
        if (event.name == "MMA tmem write") writes.push_back(&event);
    }
    require(inputs.size() == 256 && writes.size() == 256, "one 64x64x8 command is 256 4x4x8 steps");
    const auto d = profile.shmem_port_a_depth;
    const auto lb = profile.shmem_read_latency_cycles;
    for (uint32_t s = 0; s < 256; ++s) {
        const bool reads_a = s % 16 == 0;
        const uint32_t rounds = reads_a ? 2 : 1;
        require(static_cast<uint32_t>(inputs[s]->value) == s &&
                inputs[s]->detail == (reads_a ? "rounds=2, reads A" : "rounds=1"),
            "step " + std::to_string(s) + " read its A block at the wrong step: " + inputs[s]->detail);
        if (s + 1 < 256)
            require(inputs[s + 1]->start_cycle == inputs[s]->start_cycle + rounds,
                "step " + std::to_string(s + 1) + " did not start the cycle after the previous step's last round");
        const uint32_t tile = 64 * (4 * (s / 16)) + 4 * (s % 16); // m0 * 64 + n0, n fastest
        require(static_cast<uint32_t>(writes[s]->value) == tile,
            "step " + std::to_string(s) + " wrote tile " + std::to_string(writes[s]->value) +
                ", not the hold-A / stream-B order's " + std::to_string(tile));
        require(writes[s]->start_cycle == inputs[s]->start_cycle + (rounds - 1) + d + lb +
                profile.mma_result_latency_cycles,
            "step " + std::to_string(s) + ": tmem write is not last round + D + Lb + result latency");
    }
    const auto& summary = machine->trace().summary();
    require(summary.mma_a_block_reads == 16 && summary.mma_shmem_rounds == 16 * 17,
        "one command did not read 16 A blocks in 16 x 17 bank rounds");
}

// Back-to-back commands into one accumulator: the next command's first step
// is granted the cycle after the previous command's last round, so N
// commands take 272 N rounds = 272 N cycles of port B from the first input
// step: 256 steps per 272 cycles, 16/17 of one step per cycle (the rfc11
// section 0 ceiling, here measured, with no recurrence or transfer stalls).
void test_i7_back_to_back_throughput() {
    constexpr uint32_t commands = 8;
    std::array<uint16_t, 8> a{}, b{};
    a.fill(0x3c00);
    b.fill(0x3c00);
    auto machine = i7_machine(a, b, commands);
    machine->run(100000);
    for (uint32_t lane = 0; lane < 16; ++lane)
        require(machine->read_reg(0, lane, 10) == nanosm::float_bits(8.0f * commands),
            "back-to-back commands accumulated the wrong value");
    uint64_t first = std::numeric_limits<uint64_t>::max(), last = 0, steps = 0;
    for (const auto& event : machine->trace().events()) {
        if (event.name != "MMA input step") continue;
        first = std::min(first, event.start_cycle);
        last = std::max(last, event.start_cycle);
        ++steps;
    }
    require(steps == 256 * commands, "back-to-back commands lost steps");
    // The last step reads one round, so port B is busy through `last`.
    require(last - first + 1 == 272 * commands,
        "back-to-back commands did not keep port B busy one round per cycle: " +
            std::to_string(last - first + 1) + " cycles for " + std::to_string(272 * commands) + " rounds");
    require(machine->trace().summary().mma_recurrence_stalls == 0, "back-to-back commands hit recurrence stalls");
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
    // ... and while an acc.ld still reads it (WAR on tmem). I2.1 (F6): the
    // read-ownership now ends at the final tmem read, not the final RF write;
    // the text says so ("I2.1: acc.ld read-ownership" tests the boundary).
    message = run_expecting_panic(canonical_profile(), compile(header + R"(
        acc.ld.v8 r24, C, r8
        mma.async.init C, [r0], [r1]
        halt
    )"), one_warp(), "MMA over an in-flight acc.ld");
    require_mentions(message, {"tmem race: mma.async #1 on 'C'", "acc.ld #0",
        "read-owner of 'C' rows [0, 16) cols [0, 8) until its final tmem read"}, "MMA vs acc.ld");
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

    // Two distinct sources in one bank are a malformed instruction (one read
    // round, one read port per bank: src/rf.hpp).
    try {
        (void)nanosm::compile_qtx("and r2, r0, r3\nhalt", canonical_profile());
        throw std::runtime_error("same-bank and sources were accepted");
    } catch (const nanosm::SimError& error) {
        require(std::string(error.what()).find("line 1: malformed instruction 'and r2, r0, r3'") != std::string::npos,
            std::string("same-bank and: ") + error.what());
    }
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
    // The ex2's result feeds the very next fadd: its whole distance (src/rf.hpp),
    // the effective SFU latency inside it.
    require(scheduled.instructions[1].control.stall ==
            nanosm::issue_to_visible(scheduled.instructions[1], profile) - nanosm::issue_register_stages,
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
        throw std::runtime_error("same-bank fma sources were accepted");
    } catch (const nanosm::SimError& error) {
        require(std::string(error.what()).find("its distinct sources r0 and r3 are in one RF bank (0)") !=
                std::string::npos,
            std::string("same-bank fma: ") + error.what());
    }
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

// (2) The RF drain is denied by existing bookings (warp 1's fadds book the
// same bank's write port booking_horizon cycles ahead, one per cycle): the
// old vector stays put, the ingress FIFO is not read, and no chunk, depbar
// or command state changes. Since I1 the denial is bounded: the denied drain
// with service owed inhibits warp 1's new bookings to the bank, so it drains,
// intact, once the old ones have passed (no denial outlasts the horizon).
void test_assembly_contract_drain_denied() {
    auto config = fast_bus_profile(2);
    config.gmem_latency_dram_ticks = 2 * booking_horizon; // the stream fills the calendar first
    with_long_bookings(config);
    nanosm::Launch launch;
    launch.active_warps = 2;
    auto& gmem = image_bytes(launch.gmem_image, 4096);
    fill_pattern(gmem, 0, 1024, 0);
    // r8 and warp 1's stream (r2, r5, ...) share RF bank 2.
    std::string text = R"(
        sread r29, WARP_ID
        setp.ne p0, r29, 0
        @p0 bra.uni other
        mov r30, 0
        ldg.v8 r8, [r30]
        halt
    other:
    )";
    text += booking_stream(2, 200);
    text += "halt\n";
    nanosm::Machine machine(config, compile(text, config), std::move(launch));
    machine.enable_edge_probe();
    uint64_t waits = machine.trace().summary().lsu_rf_write_wait_cycles;
    uint32_t denied = 0;
    uint32_t episode = 0;
    uint32_t longest = 0;
    std::size_t seen = 0;
    while (!machine.done()) {
        const auto cycle = machine.cycle();
        tick_capped(machine);
        const auto now = machine.trace().summary().lsu_rf_write_wait_cycles;
        const bool was_denied = now > waits;
        waits = now;
        std::vector<LsuEdge> edges;
        for (; seen < machine.edge_events().size(); ++seen) edges.push_back(machine.edge_events()[seen].edge);
        episode = was_denied ? episode + 1 : 0;
        longest = std::max(longest, episode);
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
    require(denied >= booking_horizon - 5,
        "the drain was denied only " + std::to_string(denied) + " cycles; the test proves little");
    require(longest <= booking_horizon,
        "a denial lasted " + std::to_string(longest) + " cycles: the owed bit did not stop new bookings");
    require(machine.trace().summary().rf_simt_inhibited != 0, "warp 1's bookings were never inhibited");
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
// 16 words per cycle, once the ingress FIFO has backed up (warp 1's bookings
// hold chunk 0's RF write port for a while), a whole chunk enters the register
// emptied by the previous drain in one cycle, and still drains only the next.
// (I1: the bookings warp 1 makes after chunk 0's drain land booking_horizon
// cycles later, after the last chunk: chunks 1..7 are uncontended.)
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
    config.gmem_latency_dram_ticks = 4 * booking_horizon; // the stream fills the calendar first
    with_long_bookings(config);
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
    text += booking_stream(2, 100); // bank 2: r8's
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
    // I6a: the second ldg.v8 issues in the cycle the first releases its slot
    // (look-ahead (c)) and the LSU admits it issue_to_dispatch later.
    const auto admitted = final_drain + nanosm::issue_to_dispatch(config);
    require(life[1].first == admitted,
        "the second ldg.v8 was not admitted issue_to_dispatch after the first released its slot (" +
            std::to_string(life[1].first) + " vs " + std::to_string(admitted) + ")");
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
    uint64_t reuse_cycle = 0; // the second command's admission (dispatch)
    AssemblyAfter after;      // the register after that cycle
    std::map<uint64_t, AssemblyAfter> by_cycle; // the register after every cycle
};

// `old_load` is LSU command 0, `second` (opcode `second_name`) command 1. With
// one command slot and one assembly register, the second waits for both: it
// issues in the cycle the load's final RF write releases them (look-ahead
// (c)) and the LSU admits it issue_to_dispatch later (I6a), into the register
// free since then. (Before I6a the admission was that same cycle, a reuse
// (d) of the register the drain was emptying; a claim now waits for the
// release, so issue cannot line the two up.)
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
    const auto dispatch = final_drain + nanosm::issue_to_dispatch(config);
    require(admitted == dispatch,
        second_name + " was not admitted issue_to_dispatch after '" + old_load + "' made its final RF write");
    run.reuse_cycle = dispatch;
    run.after = after.at(dispatch);
    run.by_cycle = std::move(after);
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

// A stg.v8 admitted then: its first RF read beat is sampled in its admission
// cycle (bypass (e)) and fills the register Lr later (I6a).
void test_assembly_reuse_load_to_vector_store() {
    const std::string old_load = "ldg.v8 r8, [r0]";
    const auto run = run_reuse_after_final_drain(old_load, "stg.v8 [r2], r24", "stg.v8");
    bool first_read_now = false;
    for (const auto& event : run.machine->edge_events())
        if (event.edge == LsuEdge::rf_read_beat && event.id == 1 && event.cycle == run.reuse_cycle) first_read_now = true;
    require(first_read_now, "the stg.v8's first RF read was not in the reuse cycle; the test proves little");
    require(run.after.owner == 1u && run.after.valid_count == 0,
        "the stg.v8 does not own an empty register after the reuse cycle");
    const auto filled = run.by_cycle.at(run.reuse_cycle + canonical_profile().rf_read_latency_cycles);
    require(filled.owner == 1u && filled.valid_count == 16 && filled.valid == std::vector<uint8_t>(16, 1),
        "the stg.v8's first chunk is not valid Lr after the reuse cycle");
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
// I6a: a completion reaches issue through the counter (Q), so an instruction
// WAITING on the depbar issues the cycle after its completion. The one
// calculation still combines a completion and an admission that land in one
// cycle: here the ldg sets sb0 without waiting on it, issued exactly when the
// lds.uni's RF write lands. The lds.uni at L (one round): dispatch L +
// issue_to_dispatch, its slot visible to A-read one cycle later, the RAM
// sample D after that grant, its word Lb later, the gather register one cycle
// later (the RF write decision's grant), the write lands Wd after it: its
// depbar completion is presented then. `pad` independent movs put the ldg
// there. (The mov issues at fetch_to_issue, its word fetched at c0; the
// three-word FIFO then supplies one instruction per cycle.)
void test_simultaneous_depbar_completion_and_admission() {
    const auto& profile = canonical_profile();
    const uint64_t lds_issue = nanosm::fetch_to_issue +
        nanosm::issue_to_visible(nanosm::decode(compile("mov r0, 0\nhalt").words[0]), profile) -
        nanosm::issue_register_stages; // mov r0 at c2, the lds that reads it at its distance
    const auto completion_cycle = lds_issue + nanosm::issue_to_dispatch(profile) + 1 + profile.shmem_port_a_depth +
        profile.shmem_read_latency_cycles + 1 + profile.rf_write_depth_cycles;
    const auto pad = completion_cycle - lds_issue - 1;
    std::string body = "mov r0, 0\nlds.uni r1, [r0] {pred=PT, stall=0, set=sb0, wait=-}\n";
    for (uint64_t i = 0; i < pad; ++i) body += "mov r5, 7\n";
    body += "ldg r3, [r0] {pred=PT, stall=0, set=sb0, wait=-}\nhalt {pred=PT, stall=0, set=-, wait=sb0}\n";
    nanosm::Launch launch;
    launch.active_warps = 1;
    write_u32(image_bytes(launch.shmem_image, 64), 0, 0xabcd);
    write_u32(image_bytes(launch.gmem_image, 64), 0, 0x1234);
    nanosm::Machine machine(profile, hand_assemble(i1_asm(profile, body), profile), std::move(launch), {.timeline = true});
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
        if (event.category == nanosm::TraceCategory::instruction && event.name == "ldg") ldg_issue = event.start_cycle;
    }
    require(ldg_issue == completion_cycle, "the ldg did not issue at the lds's completion; re-derive the test");
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
    // I6a: the "mma.async" event is the MMA's acceptance, at dispatch. The
    // third issues in the cycle the head retires (look-ahead (c): the queue
    // as that cycle leaves it) and reaches the queue issue_to_dispatch later:
    // the pop and the push no longer share a cycle (a claim waits for the
    // pop), so the depth is 2, then 1 until the dispatch, then 2.
    const auto dispatch = nanosm::issue_to_dispatch(canonical_profile());
    const auto third_accept = third_issue;
    require(waited_full, "the third mma.async never found the queue full");
    require(third_accept == first_last_result + dispatch,
        "the third mma.async did not issue in the cycle the head retired (accepted " +
            std::to_string(third_accept) + " vs retired " + std::to_string(first_last_result) + ")");
    require(depth_after.at(first_last_result - 1) == 2 && depth_after.at(first_last_result) == 1 &&
            depth_after.at(third_accept - 1) == 1 && depth_after.at(third_accept) == 2,
        "the queue's pop and the dispatched push did not land in their cycles");
}

// The ingress FIFO's declared full flag is registered (B2.1 implementer
// choice 3, kept): when the FIFO is full at the start of a cycle, the bus
// cannot write a beat in that cycle even if the read port frees room in it;
// the beat lands the next cycle. A FIFO that is not full is read and written
// in one cycle (count' = count + push - pop).
void test_simultaneous_ingress_push_and_pop() {
    auto config = fast_bus_profile(8); // one 32 B beat per pop: a pop frees a whole beat's room
    config.lsu_ingress_sectors = 2;
    // Warp 1's long bookings hold chunk 0's RF write for a while, so the
    // FIFO fills (I1: a mov storm no longer does; see booking_stream).
    config.gmem_latency_dram_ticks = 2 * booking_horizon;
    with_long_bookings(config);
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
    text += booking_stream(2, 100); // bank 2: r8's
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
        with_long_bookings(config); // I1: a mov storm no longer holds the RF write port
        nanosm::Launch launch;
        launch.active_warps = 2;
        fill_pattern(image_bytes(launch.gmem_image, 4096), 0, 1024, 0);
        std::string text = "sread r29, WARP_ID\nsetp.ne p0, r29, 0\n@p0 bra.uni other\nmov r30, 0\nmov r31, 512\n"
            "mov r28, 2048\nldg.v8 r0, [r30]\nldg.v8 r8, [r31]\nldg r16, [r28]\nstg.v8 [r28], r0\nhalt\nother:\n";
        text += booking_stream(0, 150);
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

// ---------------------------------------------------------------------------
// B3 (rfc09 pin 3): uniformity is declared, never detected.
// - lds / sts: rounds = the most lanes on any one bank, bank = (address / 4)
//   % 16. Two lanes on one word are two lanes on one bank. Each bank serves
//   its lanes in ascending lane order, one per round.
// - lds.uni: one access at lane 0's address, the word fanned out to every
//   lane, one RF write. A nonuniform address is UB: a sanitizer PANIC.
// trash/b3-uniformity-2026-10-08.md.
// ---------------------------------------------------------------------------

// Below b3_table, the shmem word at byte address a holds b3_word(a). The
// table holds one address per lane (lane l at b3_table + 4l: one lane per
// bank, so loading it is one round).
constexpr uint32_t b3_table = 4096;

uint32_t b3_word(uint32_t address) {
    return 0xb3000000u | address;
}

nanosm::Launch b3_launch(const std::vector<uint32_t>& lane_addresses = {}) {
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.shmem_image, b3_table + 64);
    for (uint32_t address = 0; address < b3_table; address += 4) write_u32(shmem, address, b3_word(address));
    for (uint32_t lane = 0; lane < lane_addresses.size(); ++lane) {
        write_u32(shmem, b3_table + 4 * lane, lane_addresses[lane]);
    }
    return launch;
}

std::unique_ptr<nanosm::Machine> b3_machine(const std::string& program, nanosm::Launch launch,
    const nanosm::DeviceProfile& profile = canonical_profile()) {
    return std::make_unique<nanosm::Machine>(
        profile, compile(program, profile), std::move(launch), nanosm::TraceOptions{.timeline = true});
}

// Loads lane l's address from the table into r0, then runs `operation` on [r0].
std::string b3_table_program(const std::string& operation) {
    return "sread r2, LANE_ID\nshli r2, r2, 2\nlds r0, [r2 + " + std::to_string(b3_table) + "]\n" +
        operation + "\nhalt\n";
}

// Every SIMT shmem request's round count, in issue order (its trace slice).
std::vector<uint32_t> shmem_request_rounds(const nanosm::Machine& machine) {
    std::vector<std::pair<uint64_t, uint32_t>> requests;
    for (const auto& event : machine.trace().events()) {
        if (event.category != nanosm::TraceCategory::shmem || event.kind != nanosm::TraceEventKind::slice) continue;
        const auto rounds = event.detail.substr(event.detail.find("rounds=") + 7);
        requests.emplace_back(event.start_cycle, static_cast<uint32_t>(std::stoul(rounds)));
    }
    std::sort(requests.begin(), requests.end());
    std::vector<uint32_t> result;
    for (const auto& request : requests) result.push_back(request.second);
    return result;
}

// Every write of warp 0's register `reg`, seen tick by tick: the cycle that
// wrote it, how many lanes changed, and the lane values after it.
struct RegisterWrite {
    uint64_t cycle = 0;
    uint32_t lanes_changed = 0;
    std::vector<uint32_t> values;
};

std::vector<RegisterWrite> watch_register(nanosm::Machine& machine, uint32_t reg) {
    const auto lanes = [&] {
        std::vector<uint32_t> values(canonical_profile().lanes);
        for (uint32_t lane = 0; lane < values.size(); ++lane) values[lane] = machine.read_reg(0, lane, reg);
        return values;
    };
    std::vector<RegisterWrite> writes;
    auto before = lanes();
    while (!machine.done()) {
        const auto cycle = machine.cycle();
        tick_capped(machine);
        const auto after = lanes();
        uint32_t changed = 0;
        for (uint32_t lane = 0; lane < after.size(); ++lane) changed += after[lane] != before[lane] ? 1 : 0;
        if (changed != 0) writes.push_back({cycle, changed, after});
        before = after;
    }
    return writes;
}

// lds.uni: one port-A round, every lane gets the word, one full-width RF
// write at issue + 2 (the round, then the write port). The plain lds of the
// same word takes sixteen rounds: still one RF write, at issue + 17.
void test_b3_lds_uni_one_round_one_write() {
    for (const std::string operation : {"lds.uni", "lds"}) {
        const uint32_t rounds = operation == "lds.uni" ? 1 : 16;
        // r1 differs in every lane before the load, so one write of the load
        // changes all sixteen lanes in one cycle.
        auto machine = b3_machine(
            "sread r1, LANE_ID\nmov r0, 60\n" + operation + " r1, [r0 + 16]\nhalt\n", b3_launch());
        const auto writes = watch_register(*machine, 1);
        require(writes.size() == 2, operation + ": r1 was not written exactly twice (sread, load)");
        const auto& load = writes[1];
        require(load.lanes_changed == 16, operation + ": the load did not write all sixteen lanes at once");
        for (const auto value : load.values) require(value == b3_word(76), operation + ": a lane missed the word");
        require(shmem_request_rounds(*machine) == std::vector<uint32_t>{rounds}, operation + ": wrong round count");
        require(machine->trace().summary().shmem_rounds == rounds, operation + ": port A served extra rounds");
        // I2: the last round's words reach the slot D + Lb after its grant.
        // I6a: the first round is granted the cycle after dispatch (issue +
        // issue_to_dispatch, the slot register), the gather register is
        // complete the cycle after the last words, the RF write is granted
        // then and lands Wd later.
        const auto issue = issued_at(*machine, 16);
        const auto& profile = canonical_profile();
        require(issue.size() == 1 && load.cycle == issue[0] + nanosm::issue_to_dispatch(profile) + rounds + 1 +
                profile.shmem_port_a_depth + profile.shmem_read_latency_cycles + profile.rf_write_depth_cycles,
            operation + ": the RF write did not land D + Lb + 1 + Wd cycles after the last round");
    }
}

// k lanes on each word, one word per bank (lane l reads word (l / k) * k):
// k rounds, one lane per round per bank (the old same-word merge: 1 round).
void test_b3_lds_k_lanes_on_one_word() {
    for (const auto& [k, log_k] : {std::pair{1u, 0u}, std::pair{2u, 1u}, std::pair{16u, 4u}}) {
        const auto what = "lds, " + std::to_string(k) + " lanes per word";
        auto machine = b3_machine("sread r0, LANE_ID\nshri r0, r0, " + std::to_string(log_k) +
            "\nshli r0, r0, " + std::to_string(log_k + 2) + "\nlds r1, [r0]\nhalt\n", b3_launch());
        machine->run();
        require(shmem_request_rounds(*machine) == std::vector<uint32_t>{k}, what + ": rounds != k");
        for (uint32_t lane = 0; lane < 16; ++lane) {
            require(machine->read_reg(0, lane, 1) == b3_word(lane / k * k * 4), what + ": lane value");
        }
    }
}

// Lanes spread over banks: rounds = the most lanes on one bank, counting
// lanes, not words.
void test_b3_rounds_are_max_lanes_per_bank() {
    const auto spread = [](uint32_t stride) {
        std::vector<uint32_t> addresses(16);
        for (uint32_t lane = 0; lane < 16; ++lane) addresses[lane] = stride * lane;
        return addresses;
    };
    // Lanes 0..3 on byte 0, lanes 4..7 on byte 64 (both bank 0); 8..15 alone.
    std::vector<uint32_t> two_words(16);
    // Bank 1: lanes 0..2 on three words; bank 2: lanes 3..7 on one word; 8..15 alone.
    std::vector<uint32_t> mixed(16);
    for (uint32_t lane = 0; lane < 16; ++lane) {
        two_words[lane] = lane < 4 ? 0 : lane < 8 ? 64 : 4 * lane;
        mixed[lane] = lane < 3 ? 4 + 64 * lane : lane < 8 ? 8 : 4 * lane;
    }
    struct Case {
        std::string what;
        std::vector<uint32_t> addresses;
        uint32_t rounds;
        bool distinct_words; // run as sts too (two lanes on one word is a store PANIC)
    };
    const std::vector<Case> cases = {
        {"stride 4 B, one lane per bank", spread(4), 1, true},
        {"stride 8 B, two lanes per even bank", spread(8), 2, true},
        {"stride 64 B, sixteen lanes on bank 0", spread(64), 16, true},
        {"two bank-0 words, four lanes each (old merge: 2)", two_words, 8, false},
        {"five lanes on one bank-2 word, three on bank 1 (old merge: 3)", mixed, 5, false},
    };
    for (const auto& c : cases) {
        auto load = b3_machine(b3_table_program("lds r1, [r0]"), b3_launch(c.addresses));
        load->run();
        require(shmem_request_rounds(*load) == std::vector<uint32_t>{1, c.rounds}, c.what + ": lds rounds");
        for (uint32_t lane = 0; lane < 16; ++lane) {
            require(load->read_reg(0, lane, 1) == b3_word(c.addresses[lane]), c.what + ": lds lane value");
        }
        if (!c.distinct_words) continue;
        auto store = b3_machine(b3_table_program("addi r1, r2, 500\nsts [r0], r1"), b3_launch(c.addresses));
        store->run();
        require(shmem_request_rounds(*store) == std::vector<uint32_t>{1, c.rounds}, c.what + ": sts rounds");
        for (uint32_t lane = 0; lane < 16; ++lane) {
            require(store->read_shmem_u32(c.addresses[lane]) == 500 + 4 * lane, c.what + ": sts lane value");
        }
    }
}

// Each bank serves its lanes in ascending lane order. Lanes 0 and 1 share
// bank 0, lane 0 at byte 64 and lane 1 at byte 0; every other lane is alone
// on its bank. Round 0 writes lane 0's word and lanes 2..15; round 1 lane 1's
// word (the old grouping served the lower address, lane 1's, first).
void test_b3_sts_bank_serves_lanes_in_order() {
    std::vector<uint32_t> addresses(16);
    for (uint32_t lane = 0; lane < 16; ++lane) addresses[lane] = 4 * lane;
    addresses[0] = 64;
    addresses[1] = 0;
    auto machine = b3_machine(b3_table_program("addi r1, r2, 1000\nsts [r0], r1"), b3_launch(addresses));
    std::map<uint32_t, uint64_t> written; // byte address -> the cycle whose clock edge wrote it
    while (!machine->done()) {
        const auto cycle = machine->cycle();
        tick_capped(*machine);
        for (const auto address : addresses) {
            if (!written.contains(address) && machine->read_shmem_u32(address) != b3_word(address)) {
                written[address] = cycle;
            }
        }
    }
    require(shmem_request_rounds(*machine) == std::vector<uint32_t>{1, 2}, "sts: bank 0 holds two lanes");
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(machine->read_shmem_u32(addresses[lane]) == 1000 + 4 * lane, "sts: lane value");
    }
    require(written.at(0) == written.at(64) + 1, "bank 0 did not serve lane 0 before lane 1");
    for (uint32_t lane = 2; lane < 16; ++lane) {
        require(written.at(addresses[lane]) == written.at(64), "a lane alone on its bank missed round 0");
    }
}

// lds.uni composes with the RF read round, WAR, WAW and depbars exactly as
// a one-round lds does: the same compiled controls, the same issue timeline.
// The plain lds reads word 16 + l in lane l (one lane per bank); lds.uni
// reads word 16 in every lane.
void test_b3_lds_uni_composes_like_lds() {
    const auto program = [](const std::string& address_source, const std::string& load) {
        return address_source + "\nshli r0, r0, 2\n" + load + " r1, [r0 + 64]\n" +
            "mov r0, 7\n"        // WAR on the load's address register
            "fadd r2, r1, r1\n"  // RAW through the load's depbar
            "mov r1, 3\n"        // WAW on the load's destination
            "halt\n";
    };
    const auto plain = program("sread r0, LANE_ID", "lds");
    const auto uniform = program("mov r0, 0", "lds.uni");
    const auto timeline = [](const nanosm::Machine& machine) {
        std::vector<std::pair<uint64_t, uint64_t>> issues; // cycle, pc
        for (const auto& event : machine.trace().events()) {
            if (is_issue_event(event)) {
                issues.emplace_back(event.start_cycle, event.id);
            }
        }
        return issues;
    };
    const auto twice = [](uint32_t word) {
        return nanosm::float_bits(nanosm::bits_float(word) + nanosm::bits_float(word));
    };
    {
        const std::string what = "canonical";
        const auto& profile = canonical_profile();
        const auto plain_asm = nanosm::compile_qtx(plain, profile);
        const auto uniform_asm = nanosm::compile_qtx(uniform, profile);
        for (std::size_t i = 0; i < plain_asm.instructions.size(); ++i) {
            const auto& a = plain_asm.instructions[i].control;
            const auto& b = uniform_asm.instructions[i].control;
            require(a.stall == b.stall && a.set_depbar == b.set_depbar && a.wait_mask == b.wait_mask,
                what + ": qtxc scheduled lds.uni differently from lds");
        }
        auto plain_run = b3_machine(plain, b3_launch(), profile);
        auto uniform_run = b3_machine(uniform, b3_launch(), profile);
        plain_run->run();
        uniform_run->run();
        require(timeline(*plain_run) == timeline(*uniform_run), what + ": issue timelines differ");
        require(plain_run->cycle() == uniform_run->cycle(), what + ": run lengths differ");
        for (uint32_t lane = 0; lane < 16; ++lane) {
            require(plain_run->read_reg(0, lane, 2) == twice(b3_word(64 + 4 * lane)), what + ": lds value");
            require(uniform_run->read_reg(0, lane, 2) == twice(b3_word(64)), what + ": lds.uni value");
            require(uniform_run->read_reg(0, lane, 0) == 7 && uniform_run->read_reg(0, lane, 1) == 3,
                what + ": lds.uni WAR/WAW order");
        }
    }
}

// Sanitizer ON only (tagged): compiled out, hardware reads lane 0's address
// and the program runs.
void test_b3_lds_uni_nonuniform_panics() {
    // Lane 15's base is 128, the others 64: both aligned and in bounds, so
    // only the uniformity rule can fire (the ldg.uni contract test's setup).
    auto machine = b3_machine(R"(
        sread r1, LANE_ID
        addi r1, r1, 1
        shri r1, r1, 4
        shli r1, r1, 6
        addi r0, r1, 64
        lds.uni r8, [r0]
        halt
    )", b3_launch());
    expect_sim_panic(*machine, "lds.uni base address is not warp-uniform", "non-uniform lds.uni base");
}

// Alignment and bounds: the machine's own panics, as for lds, in both builds.
void test_b3_lds_uni_alignment_and_bounds() {
    const auto expect = [](const std::string& program, const std::string& needle, const std::string& what) {
        auto machine = b3_machine(program, b3_launch());
        expect_sim_panic(*machine, needle, what);
    };
    expect("mov r0, 6\nlds.uni r1, [r0]\nhalt\n", "unaligned 32-bit shmem access", "2-byte aligned lds.uni");
    expect("mov r0, 65532\nlds.uni r1, [r0 + 4]\nhalt\n", "shmem access out of bounds", "lds.uni past shmem");
    expect("mov r0, 0\nlds.uni r1, [r0 - 4]\nhalt\n", "shmem address arithmetic overflow",
        "negative lds.uni address");
    auto last_word = b3_machine("mov r0, 65532\nlds.uni r1, [r0]\nhalt\n", b3_launch());
    last_word->run();
    require(last_word->read_reg(0, 9, 1) == 0, "lds.uni of the last shmem word");
}

// ---------------------------------------------------------------------------
// I1: declared-uniform operands are a sanitizer rule (rfc09 pin 3). Hardware
// reads lane 0's value; with the sanitizer, a value that differs across lanes
// PANICs with the operand's name, as before. One test for both builds.
// ---------------------------------------------------------------------------

nanosm::Launch uniformity_launch() {
    nanosm::Launch launch;
    launch.active_warps = 1;
    fill_pattern(image_bytes(launch.gmem_image, 4096), 0, 1024, 0);
    auto& shmem = image_bytes(launch.shmem_image, 256);
    for (uint32_t address = 0; address < 256; address += 4) write_u32(shmem, address, 0x5000 + address);
    return launch;
}

void test_uniformity_uses_lane_zero() {
    // Lane 0's base is 64, lane 15's 128: both aligned and in bounds, so
    // only the uniformity rule could fire.
    const std::string one_lane_differs = R"(
        sread r1, LANE_ID
        addi r1, r1, 1
        shri r1, r1, 4
        shli r1, r1, 6
        addi r0, r1, 64
    )";
    const auto all_lanes = [](const nanosm::Machine& machine, uint32_t reg, uint32_t value, const std::string& what) {
        for (uint32_t lane = 0; lane < 16; ++lane) {
            require(machine.read_reg(0, lane, reg) == value, what + ": lane " + std::to_string(lane));
        }
    };
    struct Case {
        std::string name;
        std::string program;
        std::string message;                                    // sanitizer ON: the PANIC
        std::function<void(const nanosm::Machine&)> lane_zero;  // sanitizer OFF: lane 0's value was used
    };
    const std::vector<Case> cases = {
        {"ldg.v8 base", one_lane_differs + "ldg.v8 r8, [r0]\nhalt\n", "ldg.v8 base address is not warp-uniform",
            [](const nanosm::Machine& machine) {
                for (uint32_t k = 0; k < 8; ++k)
                    for (uint32_t lane = 0; lane < 16; ++lane)
                        require(machine.read_reg(0, lane, 8 + k) == pattern_word(16 + 16 * k + lane),
                            "ldg.v8 did not load lane 0's block");
            }},
        {"stg.v8 base", one_lane_differs + "stg.v8 [r0], r8\nhalt\n", "stg.v8 base address is not warp-uniform",
            [](const nanosm::Machine& machine) {
                require(machine.read_gmem_u32(64) == 0 && machine.read_gmem_u32(60) == pattern_word(15),
                    "stg.v8 did not store at lane 0's block");
            }},
        {"ldg.uni base", one_lane_differs + "ldg.uni r8, [r0]\nhalt\n", "ldg.uni base address is not warp-uniform",
            [&](const nanosm::Machine& machine) { all_lanes(machine, 8, pattern_word(16), "ldg.uni"); }},
        {"lds.uni base", one_lane_differs + "lds.uni r8, [r0]\nhalt\n", "lds.uni base address is not warp-uniform",
            [&](const nanosm::Machine& machine) { all_lanes(machine, 8, 0x5000 + 64, "lds.uni"); }},
        {"mbar address", "sread r0, LANE_ID\nshli r0, r0, 3\nmbar.init [r0], 1\nhalt\n",
            "mbar address is not warp-uniform",
            [](const nanosm::Machine& machine) {
                const auto state = machine.read_mbar_state(0);
                require(state.valid && state.expected == 1, "mbar.init did not use lane 0's address");
            }},
        {"bra.uni predicate",
            "sread r0, LANE_ID\nsetp.lt p0, r0, 8\n@p0 bra.uni taken\nmov r5, 1\nhalt\ntaken:\nmov r5, 2\nhalt\n",
            "bra.uni predicate is not warp-uniform",
            [&](const nanosm::Machine& machine) { all_lanes(machine, 5, 2, "bra.uni (lane 0 takes it)"); }},
    };
    for (const auto& c : cases) {
        nanosm::Machine machine(canonical_profile(), compile(c.program), uniformity_launch());
#if NANOSM_SANITIZER
        expect_sim_panic(machine, c.message, "non-uniform " + c.name);
#else
        machine.run();
        c.lane_zero(machine);
#endif
    }
}

// qtxc and qtxas: opcode 0x26, the RRI payload, a depbar like every memory op.
void test_b3_lds_uni_encoding() {
    const auto& profile = canonical_profile();
    const auto kernel = compile("mov r3, 0\nlds.uni r5, [r3 + 8]\nhalt\n");
    const auto decoded = nanosm::decode(kernel.words[1]);
    require(static_cast<unsigned>(decoded.opcode) == 0x26 && decoded.dst == 5 && decoded.src0 == 3 &&
            decoded.imm == 8 && decoded.control.set_depbar < 6,
        "lds.uni did not survive qtxc + qtxas");
    require(nanosm::format_instruction(decoded, false) == "lds.uni r5, [r3 + 8]", "lds.uni formatting");
    auto negative = decoded;
    negative.imm = -2048;
    require(nanosm::decode(nanosm::encode(negative)).imm == -2048, "lds.uni negative immediate");
    const auto header = ".profile " + profile.name + ", 0x" +
        nanosm::format_profile_fingerprint(nanosm::device_profile_fingerprint(profile)) + "\n";
    (void)nanosm::assemble_asm(header + R"(
        mov r0, 0 {pred=PT, stall=1, set=-, wait=-}
        lds.uni r1, [r0] {pred=PT, stall=0, set=sb0, wait=-}
        halt {pred=PT, stall=0, set=-, wait=sb0}
    )", profile);
    try {
        (void)nanosm::assemble_asm(header + R"(
            lds.uni r1, [r0] {pred=PT, stall=0, set=-, wait=-}
            halt {pred=PT, stall=0, set=-, wait=-}
        )", profile);
        throw std::runtime_error("qtxas accepted lds.uni without a depbar");
    } catch (const nanosm::SimError& error) {
        require(std::string(error.what()).find("memory operation requires set=sb0..sb5") != std::string::npos,
            std::string("lds.uni without a depbar: wrong error: ") + error.what());
    }
}

// ---------------------------------------------------------------------------
// I1: the admission/execution split (trash/rfc10.md section 1), load slots
// (rfc12 section 3), the RF write decision (rfc10 section 3: async_owed plus
// round-robin among ready async holders) and the freezeable acc.ld pipeline.
// trash/i1-admission-c8-rf-2026-10-10.md. The timelines below are hand
// assembly with explicit controls, so every cycle is the program's own.
// ---------------------------------------------------------------------------

// I2 (rfc12 section 2): an lds's words come back D + Lb cycles after its
// grant -- the I1 timelines below were written for D = Lb = 0. They keep
// their RF scenarios by moving the fadd bookings and the instructions after
// each lds.uni by the same amount (fadd latency + X, the lds.uni's stall
// 1 + X), so every cycle after the lds.uni is the old one + X.
// I6a: the lds.uni also reaches its slot at dispatch, issue_to_dispatch after
// its issue, so X = D + Lb + issue_to_dispatch. A fadd's booking moves
// issue_to_dispatch (the issue register and Lr ahead of its execution) by
// itself, so its latency only adds D + Lb (i1_fadd_latency). And a result
// written at the RF port in cycle t lands at t + Wd: a test that watches the
// register sees it then. The scenarios' address register r1 is zero from
// launch: a `mov r1, 0` feeding the lds.uni would now need six cycles before
// it, which would move the scenario, so the first instruction writes r61.
// I6b-2 (the front end): the first instruction issues at fetch_to_issue
// (c2: its word fetched at c0, src/rf.hpp), and the three-word FIFO then
// supplies one instruction per cycle -- through every stall and refusal, the
// FIFO fills behind them. So a scenario's cycles (the comments' c0, c1, ...)
// all move by i1_launch_shift; the tests add it to every absolute cycle.
constexpr uint64_t i1_launch_shift = nanosm::fetch_to_issue;

uint32_t i1_lds_shift(const nanosm::DeviceProfile& profile = canonical_profile()) {
    return profile.shmem_port_a_depth + profile.shmem_read_latency_cycles + nanosm::issue_to_dispatch(profile);
}

uint8_t i1_fadd_latency(const nanosm::DeviceProfile& profile, uint32_t base) {
    return static_cast<uint8_t>(base + profile.shmem_port_a_depth + profile.shmem_read_latency_cycles);
}

std::unique_ptr<nanosm::Machine> i1_machine(
    const nanosm::DeviceProfile& profile, const std::string& body, uint32_t warps = 1) {
    nanosm::Launch launch;
    launch.active_warps = warps;
    auto& shmem = image_bytes(launch.shmem_image, 4096);
    for (uint32_t word = 0; word < 1024; ++word) write_u32(shmem, 4 * word, 0x100 + word);
    return std::make_unique<nanosm::Machine>(profile, hand_assemble(i1_asm(profile, body), profile),
        std::move(launch), nanosm::TraceOptions{.timeline = true});
}

// One run, tick by tick: the cycle in which each watched register of warp 0
// changed (lane 0), and `observe(cycle)` after every tick.
std::map<uint32_t, std::vector<uint64_t>> i1_run(nanosm::Machine& machine, const std::vector<uint32_t>& regs,
    const std::function<void(uint64_t)>& observe = {}) {
    std::map<uint32_t, uint32_t> before;
    for (const auto reg : regs) before[reg] = machine.read_reg(0, 0, reg);
    std::map<uint32_t, std::vector<uint64_t>> written;
    while (!machine.done()) {
        const auto cycle = machine.cycle();
        tick_capped(machine);
        for (const auto reg : regs) {
            const auto now = machine.read_reg(0, 0, reg);
            if (now != before[reg]) written[reg].push_back(cycle);
            before[reg] = now;
        }
        if (observe) observe(cycle);
    }
    return written;
}

// Warp `warp`'s refusal slices with this reason: (first cycle, length).
std::vector<std::pair<uint64_t, uint64_t>> refusals(
    const nanosm::Machine& machine, uint32_t warp, const std::string& reason) {
    std::vector<std::pair<uint64_t, uint64_t>> result;
    for (const auto& event : machine.trace().events()) {
        if (event.category == nanosm::TraceCategory::warp && event.track == 100 + warp && event.detail == reason) {
            result.emplace_back(event.start_cycle, event.duration_cycles);
        }
    }
    return result;
}

std::string describe(const std::vector<std::pair<uint64_t, uint64_t>>& slices) {
    std::string text;
    for (const auto& [start, length] : slices) text += " {" + std::to_string(start) + ", " + std::to_string(length) + "}";
    return text;
}

// C8, Fable's probe A (trash/causality-audit-fable-2026-10-08.md section 7):
// one instruction stream; shmem word 0 is the lane mask the plain lds's
// addresses are built from. Mask 15: sixteen lanes on sixteen banks, one
// round. Mask 7: lanes l and l + 8 share a bank, two rounds, and the result
// would meet the fadd's booked write of bank 0. Before I1 the lds issued at c7
// with mask 15 and was refused at c7 with mask 7 (its own addresses decided
// its issue). Now both issue in the same cycle; the rounds move only the
// completion.
// I6a: the lds at L reaches its slot at dispatch, its first round is granted
// one cycle later, its words come back D + Lb after its last grant and its
// gather register is complete one cycle after that: one round -> its RF write
// port cycle is L + one_round_port, two rounds -> one more. The fadd, issued
// shli_distance - 1 before the lds (the stall on it waits out shli -> lds),
// books bank 0's port at its issue + issue register + Lr + latency: the
// latency below puts it on the two-round result's port cycle, which is denied
// and written one cycle later. Every result lands Wd after its port.
void test_i1_lds_issue_ignores_its_rounds() {
    auto profile = canonical_profile();
    const auto dispatch = nanosm::issue_to_dispatch(profile);
    const auto one_round_port = dispatch + 1 + profile.shmem_port_a_depth + profile.shmem_read_latency_cycles + 1;
    const auto alu_distance = nanosm::issue_to_visible(nanosm::decode(compile("mov r0, 0\nhalt").words[0]), profile) -
        nanosm::issue_register_stages;
    const auto fadd_before_lds = alu_distance - 1; // shli, fadd, (stall), lds
    profile.timing.fadd = static_cast<uint8_t>(fadd_before_lds + one_round_port + 1 -
        nanosm::issue_register_stages - profile.rf_read_latency_cycles);
    const std::string body = R"(
        sread r0, LANE_ID
        mov r4, 0
        lds.uni r2, [r4] {pred=PT, stall=0, set=sb1, wait=-}
        and r0, r0, r2 {pred=PT, stall=0, set=-, wait=sb1}
        shli r0, r0, 2
        fadd r3, r1, r1
        lds r6, [r0] {pred=PT, stall=0, set=sb0, wait=-}
        halt {pred=PT, stall=0, set=-, wait=sb0}
    )";
    struct Outcome {
        uint64_t issue = 0;
        uint64_t written = 0;
        std::vector<uint32_t> rounds;
    };
    const auto run = [&](uint32_t mask) {
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.shmem_image, 256);
        write_u32(shmem, 0, mask);
        for (uint32_t word = 1; word < 64; ++word) write_u32(shmem, 4 * word, 0x100 + word);
        nanosm::Machine machine(profile, hand_assemble(i1_asm(profile, body), profile), std::move(launch),
            nanosm::TraceOptions{.timeline = true});
        const auto written = i1_run(machine, {6});
        for (uint32_t lane = 0; lane < 16; ++lane) {
            const auto word = lane & mask;
            require(machine.read_reg(0, lane, 6) == (word == 0 ? mask : 0x100 + word), "lds lane value");
        }
        const auto issue = issued_at(machine, 6 * 8);
        require(issue.size() == 1 && written.at(6).size() == 1, "the lds issued or wrote more than once");
        return Outcome{issue[0], written.at(6)[0], shmem_request_rounds(machine)};
    };
    const auto one_round = run(15);
    const auto two_rounds = run(7);
    require(one_round.rounds == std::vector<uint32_t>{1, 1} && two_rounds.rounds == std::vector<uint32_t>{1, 2},
        "the masks did not give one and two bank rounds");
    require(one_round.issue == two_rounds.issue,
        "the lds's issue cycle depends on its own addresses (C8): " + std::to_string(one_round.issue) + " vs " +
            std::to_string(two_rounds.issue));
    const auto wd = profile.rf_write_depth_cycles;
    require(one_round.written == one_round.issue + one_round_port + wd,
        "one round: the result did not land at issue + one_round_port + Wd");
    require(two_rounds.written == two_rounds.issue + one_round_port + 2 + wd,
        "two rounds: the result (ready when the fadd's booking holds bank 0) was not written one cycle later");
}

// The tripwire behind "a refused attempt reads nothing": a capture inside an
// attempt the admission has not accepted is an internal sanitizer PANIC.
void test_sanitizer_issue_capture_before_acceptance() {
    nanosm::Sanitizer sanitizer;
    sanitizer.record_cycle_start();
    sanitizer.record_issue_attempt(3);
    try {
        sanitizer.check_issue_capture(12, 3, 7, false);
        throw std::runtime_error("a capture before acceptance did not panic");
    } catch (const nanosm::SimError& error) {
        const std::string message = error.what();
        require(message.find("warp 3 read r7") != std::string::npos &&
                message.find("not accepted") != std::string::npos,
            "wrong tripwire message: " + message);
    }
    sanitizer.record_issue_accepted();
    sanitizer.check_issue_capture(12, 3, 7, false); // accepted: fine
    sanitizer.record_issue_attempt(4);              // the next attempt starts unaccepted again
    try {
        sanitizer.check_issue_capture(12, 4, 1, true);
        throw std::runtime_error("a predicate capture before acceptance did not panic");
    } catch (const nanosm::SimError& error) {
        require(std::string(error.what()).find("read p1") != std::string::npos, "wrong predicate tripwire message");
    }
}

// Refused attempts of every kind that carries operands complete without the
// tripwire firing (sanitizer ON), and the refusals really happened. Admission
// itself (depbar_wait, structural_refusal) takes no register at all.
void test_i1_refused_attempts_read_nothing() {
    std::set<std::string> reasons;
    const auto collect = [&](const nanosm::Machine& machine) {
        for (const auto& event : machine.trace().events()) {
            // I2: shmem refusals have their own warp states.
            const bool refused = event.name == "structural" || event.name == "no free slot" ||
                event.name == "ordered behind own access";
            if (event.category == nanosm::TraceCategory::warp && refused) reasons.insert(event.detail);
        }
    };
    {   // load slots full: both slots hold results the booked RF bank 0 denies
        auto profile = canonical_profile();
        profile.timing.fadd = 7;
        auto machine = i1_machine(profile, R"(
            mov r1, 0
            fadd r30, r4, r4
            fadd r33, r4, r4
            fadd r36, r4, r4
            fadd r39, r4, r4
            fadd r42, r4, r4
            fadd r45, r4, r4
            fadd r48, r4, r4
            lds.uni r3, [r1] {pred=PT, stall=0, set=sb0, wait=-}
            lds.uni r6, [r1 + 4] {pred=PT, stall=0, set=sb1, wait=-}
            lds.uni r9, [r1 + 8] {pred=PT, stall=0, set=sb2, wait=-}
            halt {pred=PT, stall=0, set=-, wait=sb0|sb1|sb2}
        )");
        machine->run();
        collect(*machine);
        require(machine->read_reg(0, 5, 9) == 0x102, "the refused lds.uni loaded the wrong word");
    }
    {   // RF write inhibited: a fadd while a load result is owed service
        auto profile = canonical_profile();
        const auto x = i1_lds_shift(profile);
        profile.timing.fadd = i1_fadd_latency(profile, 4);
        auto machine = i1_machine(profile, R"(
            mov r61, 0
            fadd r6, r4, r4
            fadd r9, r4, r4
            lds.uni r3, [r1] {pred=PT, stall=)" + std::to_string(1 + x) + R"(, set=sb0, wait=-}
            fadd r12, r4, r4
            fadd r15, r4, r4
            halt {pred=PT, stall=0, set=-, wait=sb0}
        )");
        machine->run();
        collect(*machine);
    }
    {   // ordered behind own access (I2; was "port A busy"): an lds behind a two-round sts
        auto machine = i1_machine(canonical_profile(), R"(
            sread r0, LANE_ID
            shli r0, r0, 3
            mov r2, 9
            sts [r0], r2 {pred=PT, stall=0, set=sb0, wait=-}
            lds r5, [r0 + 4] {pred=PT, stall=0, set=sb1, wait=-}
            halt {pred=PT, stall=0, set=-, wait=sb0|sb1}
        )");
        machine->run();
        collect(*machine);
        require(machine->read_reg(0, 3, 5) == 0x100 + 7, "the refused lds loaded the wrong word");
    }
    {   // LSU command slot; tmem bank transfer busy (two accumulators in one tmem bank)
        auto profile = canonical_profile();
        profile.lsu_active_commands = 1;
        nanosm::Launch launch;
        launch.active_warps = 1;
        fill_pattern(image_bytes(launch.gmem_image, 4096), 0, 1024, 0);
        nanosm::Machine machine(profile, hand_assemble(i1_asm(profile, R"(
            .acc C0, 0, 0, 16, 16
            .acc C1, 0, 256, 16, 16
            sread r0, LANE_ID
            shli r0, r0, 2
            mov r8, 0
            mov r9, 0
            ldg r11, [r0] {pred=PT, stall=0, set=sb0, wait=-}
            ldg r12, [r0 + 64] {pred=PT, stall=0, set=sb1, wait=-}
            acc.st.v8 C0, r8, r0 {pred=PT, stall=0, set=sb2, wait=-}
            acc.ld r10, C1, r8 {pred=PT, stall=0, set=sb3, wait=-}
            halt {pred=PT, stall=0, set=-, wait=sb0|sb1|sb2|sb3}
        )"), profile), std::move(launch), nanosm::TraceOptions{.timeline = true});
        machine.run();
        collect(machine);
        require(machine.read_reg(0, 4, 12) == pattern_word(20), "the refused ldg loaded the wrong word");
    }
    for (const std::string reason : {"load slots full", "RF write inhibited", "ordered behind own access",
             "LSU command slot", "tmem bank transfer busy"}) {
        require(reasons.contains(reason), "no attempt was refused for '" + reason + "'; the test proves little");
    }
}

// (a) A ready lds result against dense arithmetic writing its bank (fadd
// latency 4). The fadds at c1, c2 book bank 0 for c5, c6 and set async_owed;
// the lds.uni (c3) has its word at c5: denied with service owed, so bank 0
// is inhibited and the fadd at c5 is refused (c5, c6); at c7 the old
// bookings have drained and the result is written. In that cycle the fadd
// books c11: owed is set again (a booking and a grant in one cycle: set wins).
void test_i1_rf_owed_inhibits_after_denial() {
    auto profile = canonical_profile();
    const auto x = i1_lds_shift(profile);
    const auto wd = profile.rf_write_depth_cycles;
    profile.timing.fadd = i1_fadd_latency(profile, 4);
    auto machine = i1_machine(profile, R"(
        mov r61, 0
        fadd r6, r4, r4
        fadd r9, r4, r4
        lds.uni r3, [r1] {pred=PT, stall=)" + std::to_string(1 + x) + R"(, set=sb0, wait=-}
        fadd r12, r4, r4
        fadd r15, r4, r4
        halt {pred=PT, stall=0, set=-, wait=sb0}
    )");
    require(!machine->rf_write_owed(0), "async_owed did not start clear");
    const auto s = i1_launch_shift;
    std::map<uint64_t, std::pair<bool, uint32_t>> bank0; // cycle -> owed, cursor after it
    const auto written = i1_run(*machine, {3}, [&](uint64_t cycle) {
        bank0[cycle] = {machine->rf_write_owed(0), machine->rf_write_cursor(0)};
    });
    require(!bank0.at(s).first && bank0.at(s + 1).first, "a successful booking of bank 0 did not set async_owed");
    require(issued_at(*machine, 3 * 8) == std::vector<uint64_t>{s + 3}, "lds.uni issue");
    require(written.at(3) == std::vector<uint64_t>{s + 7 + x + wd},
        "the result was not written when the old bookings drained (c7 + x, landing Wd later)");
    require(refusals(*machine, 0, "RF write inhibited") == std::vector<std::pair<uint64_t, uint64_t>>{{s + 5 + x, 2}},
        "bank 0 was not inhibited exactly while the result was denied with service owed (c5 + x, c6 + x)");
    require(issued_at(*machine, 5 * 8) == std::vector<uint64_t>{s + 7 + x},
        "the inhibited fadd did not issue in the grant cycle");
    require(bank0.at(s + 7 + x).first, "a grant and a booking in one cycle cleared async_owed (set must win)");
    require(bank0.at(s + 6 + x).second == 0 && bank0.at(s + 7 + x).second == 1,
        "the cursor did not move past load slot 0 on its grant");
    const auto& summary = machine->trace().summary();
    require(summary.rf_async_waits == 2 && summary.rf_simt_inhibited == 2 && summary.rf_async_writes == 1,
        "rf counters");
}

// (a, cont.) async_owed = 0 permits one more booking; a refused candidate
// leaves the bit alone. fadd latency 5: the first lds.uni's result (c5) is
// granted on a free port and clears owed; the second's (c6) is denied with
// owed = 0, so the fadd at c6 still books (and sets owed); at c7 the result
// is denied again and the next fadd is inhibited; c8 it is written.
void test_i1_rf_owed_one_more_booking() {
    auto profile = canonical_profile();
    const auto x = i1_lds_shift(profile);
    const auto wd = profile.rf_write_depth_cycles;
    profile.timing.fadd = i1_fadd_latency(profile, 5);
    {
        auto machine = i1_machine(profile, R"(
            mov r61, 0
            fadd r6, r4, r4
            fadd r9, r4, r4
            lds.uni r3, [r1] {pred=PT, stall=0, set=sb0, wait=-}
            lds.uni r12, [r1] {pred=PT, stall=)" + std::to_string(1 + x) + R"(, set=sb1, wait=-}
            mov r2, 1
            fadd r15, r4, r4
            fadd r18, r4, r4
            halt {pred=PT, stall=0, set=-, wait=sb0|sb1}
        )");
        const auto s = i1_launch_shift;
        std::map<uint64_t, std::pair<bool, uint32_t>> bank0;
        const auto written = i1_run(*machine, {3, 12}, [&](uint64_t cycle) {
            bank0[cycle] = {machine->rf_write_owed(0), machine->rf_write_cursor(0)};
        });
        require(written.at(3) == std::vector<uint64_t>{s + 5 + x + wd} &&
                written.at(12) == std::vector<uint64_t>{s + 8 + x + wd},
            "write cycles");
        require(!bank0.at(s + 5 + x).first, "an async grant with no booking in its cycle did not clear async_owed");
        require(issued_at(*machine, 6 * 8) == std::vector<uint64_t>{s + 6 + x},
            "with async_owed = 0 a denied writer did not let one more booking through");
        require(bank0.at(s + 6 + x).first, "that booking did not set async_owed");
        require(refusals(*machine, 0, "RF write inhibited") ==
                std::vector<std::pair<uint64_t, uint64_t>>{{s + 7 + x, 1}},
            "the next booking was not inhibited (c7 + x)");
        require(bank0.at(s + 5 + x).second == 1 && bank0.at(s + 6 + x).second == 1 &&
                bank0.at(s + 7 + x).second == 1 && bank0.at(s + 8 + x).second == 2,
            "the cursor moved without a grant, or not on one");
    }
    {   // A refused booking (its port taken) does not set the bit. I6a: the add
        // after the lds.uni moved X with it, but its booking is only
        // issue_to_dispatch + alu ahead of its issue: the fadd latency that
        // puts their ports on one cycle is 5 + X here.
        auto refusing = profile;
        refusing.timing.fadd = static_cast<uint8_t>(5 + x);
        auto machine = i1_machine(refusing, R"(
            mov r61, 0
            fadd r6, r4, r4
            lds.uni r3, [r1] {pred=PT, stall=)" + std::to_string(1 + x) + R"(, set=sb0, wait=-}
            mov r2, 1
            mov r5, 1
            add r9, r4, r4
            halt {pred=PT, stall=0, set=-, wait=sb0}
        )");
        const auto s = i1_launch_shift;
        std::map<uint64_t, bool> owed;
        (void)i1_run(*machine, {}, [&](uint64_t cycle) { owed[cycle] = machine->rf_write_owed(0); });
        require(refusals(*machine, 0, "RF write reservation") ==
                std::vector<std::pair<uint64_t, uint64_t>>{{s + 5 + x, 1}},
            "the add was not refused at c5 + x (its write cycle is the fadd's)");
        require(!owed.at(s + 4 + x) && !owed.at(s + 5 + x) && owed.at(s + 6 + x),
            "a refused booking changed async_owed");
    }
}

// (d) Inhibition refuses only fixed RF-write bookings to the inhibited bank:
// while bank 0 is inhibited (c5, c6) a mov writing bank 1 issues, and so does
// an lds.uni whose destination is in bank 0 (a destination is not a booking).
// (e) With no async writer ready, a set async_owed inhibits nothing.
void test_i1_rf_inhibition_is_bank_and_booking_local() {
    {
        auto profile = canonical_profile();
        const auto x = i1_lds_shift(profile);
        const auto wd = profile.rf_write_depth_cycles;
        profile.timing.fadd = i1_fadd_latency(profile, 4);
        auto machine = i1_machine(profile, R"(
            mov r61, 0
            fadd r6, r4, r4
            fadd r9, r4, r4
            lds.uni r3, [r1] {pred=PT, stall=)" + std::to_string(1 + x) + R"(, set=sb0, wait=-}
            fadd r12, r4, r4
            mov r13, 1
            lds.uni r21, [r1] {pred=PT, stall=0, set=sb1, wait=-}
            fadd r15, r4, r4
            halt {pred=PT, stall=0, set=-, wait=sb0|sb1}
        )");
        const auto s = i1_launch_shift;
        const auto written = i1_run(*machine, {3, 13, 21});
        require(written.at(3) == std::vector<uint64_t>{s + 7 + x + wd},
            "the first result was not written at c7 + x (+ Wd)");
        require(issued_at(*machine, 5 * 8) == std::vector<uint64_t>{s + 5 + x},
            "a bank-1 write was held by bank 0's inhibition");
        require(issued_at(*machine, 6 * 8) == std::vector<uint64_t>{s + 6 + x},
            "an lds.uni into bank 0 was held by bank 0's inhibition");
        // The second lds.uni's own result is x later too (it meets fadd r12's booking).
        require(written.at(21) == std::vector<uint64_t>{s + 9 + 2 * x + wd}, "the second result");
        require(machine->trace().summary().rf_simt_inhibited == 0, "nothing should have been inhibited");
    }
    {
        auto machine = i1_machine(canonical_profile(), R"(
            mov r1, 0
            fadd r6, r4, r4
            fadd r9, r4, r4
            fadd r12, r4, r4
            fadd r15, r4, r4
            fadd r18, r4, r4
            halt
        )");
        const auto s = i1_launch_shift;
        std::map<uint64_t, bool> owed;
        (void)i1_run(*machine, {}, [&](uint64_t cycle) { owed[cycle] = machine->rf_write_owed(0); });
        for (uint32_t pc = 1; pc <= 5; ++pc) {
            require(issued_at(*machine, pc * 8) == std::vector<uint64_t>{s + pc},
                "a booking was refused with no writer ready");
        }
        require(owed.at(s + 1) && owed.at(s + 5), "async_owed was not set by the bookings");
        require(machine->trace().summary().rf_simt_inhibited == 0, "inhibited with no async writer ready");
    }
}

// (c) Round-robin among ready holders. One RF bank (two read ports), so every
// holder targets it: two acc.ld.v8 pipelines (tmem banks 0 and 1) present a
// beat every cycle and take turns -- each pipeline holds every other cycle,
// every beat is written once and in order. In c32 a mov's booked write takes
// the port: no grant, and the cursor stays where it was.
void test_i1_rf_round_robin_among_holders() {
    auto profile = canonical_profile();
    profile.rf_banks = 1;
    profile.rf_read_ports_per_bank = 2;
    auto machine = i1_machine(profile, R"(
        .acc C0, 0, 0, 64, 64
        .acc C1, 1, 0, 64, 64
        sread r0, LANE_ID
        addi r0, r0, 1
        addi r1, r0, 100
        addi r2, r0, 200
        addi r3, r0, 300
        addi r4, r0, 400
        addi r5, r0, 500
        addi r6, r0, 600
        addi r7, r0, 700
        mov r8, 0
        mov r9, 0
        acc.st.v8 C0, r8, r0 {pred=PT, stall=0, set=sb0, wait=-}
        acc.st.v8 C1, r8, r0 {pred=PT, stall=0, set=sb1, wait=-}
        acc.ld.v8 r16, C0, r8 {pred=PT, stall=0, set=sb2, wait=sb0}
        acc.ld.v8 r24, C1, r8 {pred=PT, stall=0, set=sb3, wait=sb1}
        mov r50, 1
        halt {pred=PT, stall=0, set=-, wait=sb2|sb3}
    )");
    std::vector<uint32_t> regs{50};
    for (uint32_t reg = 16; reg < 32; ++reg) regs.push_back(reg);
    std::map<uint64_t, uint32_t> cursor;
    const auto written = i1_run(*machine, regs, [&](uint64_t cycle) { cursor[cycle] = machine->rf_write_cursor(0); });
    std::vector<std::pair<uint64_t, uint32_t>> order; // (cycle, register)
    for (const auto& [reg, cycles] : written) {
        require(cycles.size() == 1, "r" + std::to_string(reg) + " was not written exactly once");
        order.emplace_back(cycles[0], reg);
    }
    std::sort(order.begin(), order.end());
    // Holders: load slots 0-1, assembly registers 2-3, acc.ld pipelines 4 (tmem bank 0) and 5.
    const std::vector<uint32_t> expected{16, 50, 24, 17, 25, 18, 26, 19, 27, 20, 28, 21, 29, 22, 30, 23, 31};
    for (std::size_t i = 0; i < order.size(); ++i) {
        require(order[i].second == expected[i], "write " + std::to_string(i) + " was r" +
            std::to_string(order[i].second) + ", not r" + std::to_string(expected[i]));
        require(order[i].first == order.front().first + i, "a cycle passed without a write (a refill bubble)");
    }
    // I6a: a write is granted (at the port) Wd cycles before the register
    // changes; the cursor moves at the grant.
    const auto wd = profile.rf_write_depth_cycles;
    const auto mov_port = order[1].first - wd;
    require(cursor.at(mov_port - 1) == 5 && cursor.at(mov_port) == 5,
        "the cursor moved in the cycle no holder was granted");
    for (std::size_t i = 2; i < order.size(); ++i) {
        const bool pipeline1 = order[i].second >= 24;
        require(cursor.at(order[i].first - wd) == (pipeline1 ? 0u : 5u),
            "the cursor did not move past the holder granted");
    }
}

// The round-robin choice itself (one bank): scan from the cursor, skip holders
// not ready, grant up to the free ports, move past the last one granted.
void test_i1_round_robin_grant() {
    using Grant = nanosm::RoundRobinGrant;
    const auto grant = [](std::vector<uint8_t> ready, uint32_t cursor, uint32_t ports) {
        return nanosm::round_robin_grant(ready, cursor, ports);
    };
    const auto same = [](const Grant& got, std::vector<uint32_t> granted, uint32_t cursor) {
        return got.granted == granted && got.cursor_after == cursor;
    };
    require(same(grant({1, 0, 1, 0}, 0, 1), {0}, 1), "from cursor 0");
    require(same(grant({1, 0, 1, 0}, 1, 1), {2}, 3), "skips a holder that is not ready");
    require(same(grant({1, 0, 1, 0}, 3, 1), {0}, 1), "wraps around");
    require(same(grant({1, 0, 1, 0}, 2, 0), {}, 2), "no free port: nobody granted, the cursor stays");
    require(same(grant({0, 0, 0, 0}, 2, 1), {}, 2), "nobody ready: the cursor stays");
    require(same(grant({1, 1, 1, 0}, 1, 2), {1, 2}, 3), "two ports: two grants, past the last");
    // Two holders that stay ready alternate.
    uint32_t cursor = 0;
    std::vector<uint32_t> turns;
    for (int i = 0; i < 6; ++i) {
        const auto result = grant({0, 1, 0, 1}, cursor, 1);
        turns.push_back(result.granted.at(0));
        cursor = result.cursor_after;
    }
    require(turns == std::vector<uint32_t>{1, 3, 1, 3, 1, 3}, "two persistently ready holders did not alternate");
}

// The acc.ld transfer pipeline freezes and resumes (rfc10 section 2). The six
// fadds before the acc.st book RF bank 1 for six consecutive cycles from P,
// the cycle beat 0 (r16, bank 1) is first the pipeline's output. Denied, the
// whole pipeline holds -- same destination and data, no tmem read, depbar
// set, done() false -- while the add that wants bank 1 is held back first by
// its own booked port, then by the inhibition. At P + 6 the beat is written
// and the pipeline refills with no bubble; later beats hold one cycle behind
// each add's booking. The depbar clears as the last beat's write lands.
// I6a (src/rf.hpp): the acc.ld at A reaches its slot at dispatch (A +
// issue_to_dispatch), its first tmem read one cycle later, its output stage
// one more: P = A + issue_to_dispatch + 2. A calibration run (fadd latency 1,
// bookings long gone) measures A and the first fadd's issue F0; the fadd
// latency that puts F0's booking (F0 + issue register + Lr + latency) on P is
// then used. The add r46 issues two cycles after the acc.ld, at P - 3, and
// books issue + 4 (issue register, Lr, alu): its ports P + 1 .. P + 5 are
// the fadds', then (P + 2 .. P + 5) bank 1 is inhibited while beat 0 is
// denied with service owed; it issues at P + 6 and books P + 10. The add
// r43 issues at P + 8 and books P + 12. The beats r16..r23 sit in banks 1,
// 2, 0, 1, 2, 0, 1, 2: beats 0-5 are written at P + 6 .. P + 11 (r46's
// bank-1 booking at P + 10 is not beat 4's bank), beat 6 (bank 1) meets
// r43's booking at P + 12 and goes at P + 13, beat 7 at P + 14. Every
// register changes Wd after its write.
void test_i1_acc_ld_pipeline_freeze() {
    const std::string body = R"(
        .acc C, 0, 0, 64, 64
        sread r0, LANE_ID
        addi r0, r0, 1
        addi r1, r0, 100
        addi r2, r0, 200
        addi r3, r0, 300
        addi r4, r0, 400
        addi r5, r0, 500
        addi r6, r0, 600
        addi r7, r0, 700
        mov r8, 0
        mov r9, 0
        fadd r25, r10, r10
        fadd r28, r10, r10
        fadd r31, r10, r10
        fadd r34, r10, r10
        fadd r37, r10, r10
        fadd r40, r10, r10
        acc.st.v8 C, r8, r0 {pred=PT, stall=0, set=sb0, wait=-}
        acc.ld.v8 r16, C, r8 {pred=PT, stall=2, set=sb1, wait=sb0}
        add r46, r10, r10 {pred=PT, stall=2, set=-, wait=-}
        add r43, r10, r10
        halt {pred=PT, stall=0, set=-, wait=sb1}
    )";
    auto profile = canonical_profile();
    profile.timing.fadd = 1;
    uint64_t acc_ld_issue = 0;
    uint64_t first_fadd = 0;
    {
        auto calibration = i1_machine(profile, body);
        calibration->run();
        acc_ld_issue = issued_at(*calibration, 18 * 8).at(0);
        first_fadd = issued_at(*calibration, 11 * 8).at(0);
    }
    const auto p = acc_ld_issue + nanosm::issue_to_dispatch(profile) + 2;
    profile.timing.fadd = static_cast<uint8_t>(p - first_fadd - nanosm::issue_register_stages -
        profile.rf_read_latency_cycles);
    auto machine = i1_machine(profile, body);
    require(nanosm::Machine::acc_ld_pipeline_stages() == 1, "this test is written for a one-stage pipeline");
    struct Seen {
        std::optional<nanosm::Machine::AccLdStageView> output;
        uint64_t tmem_reads = 0;
        uint32_t sb1 = 0;
        bool done = false;
    };
    std::map<uint64_t, Seen> seen;
    std::vector<uint32_t> regs;
    for (uint32_t reg = 16; reg < 24; ++reg) regs.push_back(reg);
    const auto written = i1_run(*machine, regs, [&](uint64_t cycle) {
        seen[cycle] = {machine->acc_ld_stage(0, 0), machine->trace().summary().tmem_load_beats,
            machine->depbar_value(0, 1), machine->done()};
    });
    require(issued_at(*machine, 18 * 8) == std::vector<uint64_t>{acc_ld_issue},
        "acc.ld issue moved with the fadd latency; the calibration does not hold");
    // Held: the output after the ticks of P - 1 .. P + 5 (Q of P .. P + 6) is beat 0, unchanged.
    for (uint64_t cycle = p - 1; cycle <= p + 5; ++cycle) {
        const auto& at = seen.at(cycle);
        require(at.output.has_value() && at.output->reg == 16 && !at.output->last,
            "the output stage did not hold beat 0 after c" + std::to_string(cycle));
        require(at.output->values == seen.at(p - 1).output->values, "the held beat's data changed");
        require(at.tmem_reads == 1, "a tmem read launched while the output was held (c" + std::to_string(cycle) + ")");
        require(at.sb1 == 1 && !at.done, "the depbar or done() moved while a beat was held");
    }
    const auto wd = profile.rf_write_depth_cycles;
    const std::vector<uint64_t> port{p + 6, p + 7, p + 8, p + 9, p + 10, p + 11, p + 13, p + 14};
    for (uint32_t beat = 0; beat < 8; ++beat) {
        const auto reg = 16 + beat;
        std::string seen_writes;
        for (const auto cycle : written.at(reg)) seen_writes += " " + std::to_string(cycle);
        require(written.at(reg) == std::vector<uint64_t>{port[beat] + wd},
            "beat " + std::to_string(beat) + " was not written exactly once at c" + std::to_string(port[beat]) +
                " (landing Wd later; P = " + std::to_string(p) + ", landed" + seen_writes + "; inhibited" +
                describe(refusals(*machine, 0, "RF write inhibited")) + "; booked" +
                describe(refusals(*machine, 0, "RF write reservation")) + "; issues" + issue_log(*machine) + ")");
        for (uint32_t lane = 0; lane < 16; ++lane) {
            require(machine->read_reg(0, lane, reg) == lane + 1 + 100 * beat, "beat " + std::to_string(beat) + " data");
        }
    }
    require(seen.at(p + 12).output.has_value() && seen.at(p + 12).output->reg == 22,
        "beat 6 did not hold through P + 12 after beats 0-5 were written");
    require(seen.at(port[7] + wd - 1).sb1 == 1 && seen.at(port[7] + wd).sb1 == 0,
        "the depbar did not clear exactly as the last beat's write landed");
    require(refusals(*machine, 0, "RF write inhibited") == std::vector<std::pair<uint64_t, uint64_t>>{{p + 2, 4}},
        "the add was not inhibited while beat 0 was denied with service owed");
    require(machine->trace().summary().tmem_load_beats == 8 && machine->trace().summary().rf_async_writes == 8,
        "beats launched or written more than once");
}

// Load slots (rfc12 section 3; shmem.load_slots = 2). Two lds.uni hold both
// slots, their results complete and waiting for the RF (eight fadds keep RF
// bank 0 booked over them); a third is refused (structural, counted) until
// the RF write decision grants slot 0 and issues in that same cycle.
// Meanwhile port A serves a store while both slots hold results.
// I6a re-derivation (was two warps, qtxc): one warp, hand assembly, so the
// fadd bookings are contiguous (a second warp's issues used to leave holes in
// them, and a hole under a ready result grants it). Relative to the first
// lds.uni at L0: the eight fadds issue at L0 - 8 .. L0 - 1 and book bank 0
// for L0 - 5 + latency .. L0 + 2 + latency (issue register, Lr, latency:
// src/rf.hpp); latency 13 covers L0 + 8 .. L0 + 15. Each lds.uni's result is
// at its RF port x + 2 after its issue (x = i1_lds_shift): L0 + 9 and
// L0 + 10, denied until the bookings end, so slot 0's write is granted at
// G0 = L0 + 16. The sts (a write) is ordered behind the two loads until the
// second one's round is granted (L0 + 5), issues then and is granted at
// L0 + 9 (dispatch, slot register), reaching the RAM at L0 + 11 while both
// results wait. The third lds.uni is ordered behind the sts until that grant
// (L0 + 6 .. L0 + 8; before I6a the sts was granted the cycle after its
// issue, so no order wait showed), then refused for load slots from L0 + 9
// to G0, and issues at G0. Results land Wd after their grants.
void test_i1_load_slots() {
    auto profile = canonical_profile();
    const auto x = i1_lds_shift(profile);
    const auto wd = profile.rf_write_depth_cycles;
    profile.timing.fadd = i1_fadd_latency(profile, 9);
    auto machine = i1_machine(profile, R"(
        sread r20, LANE_ID
        mov r21, 7
        shli r20, r20, 2
        fadd r30, r4, r4
        fadd r33, r4, r4
        fadd r36, r4, r4
        fadd r39, r4, r4
        fadd r42, r4, r4
        fadd r45, r4, r4
        fadd r48, r4, r4
        fadd r51, r4, r4
        lds.uni r3, [r1] {pred=PT, stall=0, set=sb0, wait=-}
        lds.uni r6, [r1 + 4] {pred=PT, stall=0, set=sb1, wait=-}
        sts [r20 + 1024], r21 {pred=PT, stall=0, set=sb2, wait=-}
        lds.uni r9, [r1 + 8] {pred=PT, stall=0, set=sb3, wait=-}
        halt {pred=PT, stall=0, set=-, wait=sb0|sb1|sb2|sb3}
    )");
    // The slots as they stand during each cycle (Q: observed after the
    // previous tick), and the cycle the store's round lands.
    bool both_waiting = false;
    std::optional<uint64_t> store_cycle;
    bool store_while_both_waiting = false;
    const auto written = i1_run(*machine, {3, 6, 9}, [&](uint64_t cycle) {
        if (!store_cycle.has_value() && machine->read_shmem_u32(1024) == 7) {
            store_cycle = cycle;
            store_while_both_waiting = both_waiting;
        }
        both_waiting = machine->load_slot_complete(0) && machine->load_slot_complete(1);
    });
    const auto lds = [&](uint32_t n) { return issued_at(*machine, (11 + n) * 8 + (n == 2 ? 8 : 0)); };
    const auto at = [](const std::vector<uint64_t>& cycles) { return cycles.empty() ? std::string("-") : std::to_string(cycles.front()); };
    const auto observed = " lds " + at(lds(0)) + " " + at(lds(1)) + " " + at(lds(2)) + "; written " + at(written.at(3)) +
        " " + at(written.at(6)) + " " + at(written.at(9)) + "; refused" + describe(refusals(*machine, 0, "load slots full")) +
        "; store c" + (store_cycle.has_value() ? std::to_string(*store_cycle) : std::string("-")) +
        "; issues" + issue_log(*machine);
    require(lds(0).size() == 1 && lds(1) == std::vector<uint64_t>{lds(0)[0] + 1}, "the first two lds.uni:" + observed);
    const auto l0 = lds(0)[0];
    const auto g0 = l0 + 16;
    require(issued_at(*machine, 13 * 8) == std::vector<uint64_t>{l0 + 5},
        "the sts did not issue as the second load's round was granted:" + observed);
    // The sts behind the second load (L0 + 2 .. L0 + 4), the third load behind
    // the sts (L0 + 6 .. L0 + 8): issue_to_dispatch cycles each.
    const auto t = nanosm::issue_to_dispatch(profile);
    require(refusals(*machine, 0, "ordered behind own access") ==
            std::vector<std::pair<uint64_t, uint64_t>>{{l0 + 2, t}, {l0 + 6, t}},
        "the sts was not ordered behind the second load until its grant (L0 + 5), or the third lds.uni behind "
        "the sts until its grant (L0 + 9):" + observed + "; ordered" +
            describe(refusals(*machine, 0, "ordered behind own access")));
    const auto refused = refusals(*machine, 0, "load slots full");
    require(refused == std::vector<std::pair<uint64_t, uint64_t>>{{l0 + 9, g0 - l0 - 9}},
        "the third lds.uni was not refused for load slots from L0 + 9 until slot 0's write (G0):" + observed);
    require(machine->trace().summary().shmem_load_slot_waits == g0 - l0 - 9, "the refusals were not counted");
    require(lds(2) == std::vector<uint64_t>{g0} && written.at(3) == std::vector<uint64_t>{g0 + wd},
        "the third lds.uni did not take slot 0 in the cycle its write was granted:" + observed);
    require(written.at(6) == std::vector<uint64_t>{g0 + 1 + wd} &&
            written.at(9) == std::vector<uint64_t>{g0 + x + 2 + wd},
        "results:" + observed);
    require(machine->read_reg(0, 7, 9) == 0x102, "the third load's word");
    require(store_cycle == l0 + 11 && store_while_both_waiting,
        "port A did not serve the sts while both load slots held results waiting for the RF:" + observed);
    for (uint32_t lane = 0; lane < 16; ++lane) require(machine->read_shmem_u32(1024 + 4 * lane) == 7, "sts data");
}

// shmem.load_slots: optional in the profile JSON (default 2), an integer of
// at least 1, part of the fingerprint. With one slot, back-to-back lds.uni
// serialize on it: the second takes it in the cycle the first's write is
// granted.
void test_i1_load_slots_profile_field() {
    std::ifstream input("profiles/nanosm-v0.json");
    nlohmann::json document;
    input >> document;
    require(nanosm::parse_device_profile(document.dump()).shmem_load_slots == 2,
        "the canonical profile does not declare two load slots");
    auto defaulted = document;
    defaulted["shmem"].erase("load_slots");
    require(nanosm::parse_device_profile(defaulted.dump()).shmem_load_slots == 2, "the default is not 2");
    for (const auto value : {nlohmann::json(0), nlohmann::json(-1), nlohmann::json(1.5)}) {
        auto modified = document;
        modified["shmem"]["load_slots"] = value;
        try {
            (void)nanosm::parse_device_profile(modified.dump());
            throw std::runtime_error("invalid load_slots " + value.dump() + " was accepted");
        } catch (const nanosm::ProfileError&) {}
    }
    auto zero = canonical_profile();
    zero.shmem_load_slots = 0;
    try {
        nanosm::validate_device_profile(zero);
        throw std::runtime_error("zero load slots passed validation");
    } catch (const nanosm::ProfileError&) {}
    auto three = canonical_profile();
    three.shmem_load_slots = 3;
    require(nanosm::device_profile_fingerprint(three) != nanosm::device_profile_fingerprint(canonical_profile()),
        "load slots were omitted from the profile fingerprint");
    auto one = canonical_profile();
    one.shmem_load_slots = 1;
    auto machine = i1_machine(one, R"(
        mov r61, 0
        lds.uni r3, [r1] {pred=PT, stall=0, set=sb0, wait=-}
        lds.uni r6, [r1 + 4] {pred=PT, stall=0, set=sb1, wait=-}
        halt {pred=PT, stall=0, set=-, wait=sb0|sb1}
    )");
    const auto written = i1_run(*machine, {3, 6});
    // I2: the slot is held while the first load's words come back (D + Lb).
    const auto refused = refusals(*machine, 0, "load slots full");
    require(refused == std::vector<std::pair<uint64_t, uint64_t>>{{i1_launch_shift + 2, 1 + i1_lds_shift(one)}},
        "the second lds.uni was not refused until the one slot freed:" + describe(refused));
    require(written.at(3).size() == 1 &&
            issued_at(*machine, 2 * 8) == std::vector<uint64_t>{written.at(3)[0] - one.rf_write_depth_cycles},
        "the second lds.uni did not take the slot as it freed (the first's write grant, Wd before it lands)");
    require(machine->read_reg(0, 0, 6) == 0x101, "the second load's word");
}

// I1.1: the LSU has one RF write path (rfc09 section 3a: one lane-wide vector
// per cycle), so its output mux presents ONE assembly register per cycle to the
// RF write decision, round-robin from the LSU cursor. Warp 0 issues two ldg.uni:
// r9 (bank 0) then r10 (bank 1). Warp 1 keeps bank 0 booked with fadds, so the
// first load waits for its bank while the second completes: both assembly
// registers hold complete vectors at once, with bank 1 free. The LSU still
// writes only the register at its cursor (the bank-0 load, when bank 0 opens)
// and the other one afterwards -- never the second one past the first, and
// never two in one cycle.
void test_i1_lsu_one_rf_write_per_cycle() {
    auto config = canonical_profile();
    config.gmem_latency_dram_ticks = 64 * 25; // the loads return while warp 1's fadds book bank 0 every cycle
    config.gmem_bus_bytes_per_dram_tick = config.gmem_sector_bytes;
    config.dram_clock_hz = config.core_clock_hz * 64;
    config.lsu_pluck_words_per_cycle = 64; // a load completes as soon as its sector arrives
    nanosm::Launch launch;
    launch.active_warps = 2;
    auto& image = image_bytes(launch.gmem_image, 1024);
    fill_pattern(image, 0, 256, 0);
    nanosm::Machine machine(config, compile(R"(
        sread r2, WARP_ID
        setp.ne p0, r2, 0
@p0     bra.uni ARITH
        mov r0, 0
        mov r1, 512
        ldg.uni r9, [r0]
        ldg.uni r10, [r1]
        halt
ARITH:
        mov r4, 1
        mov r5, 2
        fadd r3, r4, r5
        fadd r6, r4, r5
        fadd r24, r4, r5
        fadd r27, r4, r5
        fadd r30, r4, r5
        fadd r33, r4, r5
        fadd r36, r4, r5
        fadd r39, r4, r5
        fadd r3, r4, r5
        fadd r6, r4, r5
        fadd r24, r4, r5
        fadd r27, r4, r5
        fadd r30, r4, r5
        fadd r33, r4, r5
        fadd r36, r4, r5
        fadd r39, r4, r5
        fadd r3, r4, r5
        fadd r6, r4, r5
        fadd r24, r4, r5
        fadd r27, r4, r5
        fadd r30, r4, r5
        fadd r33, r4, r5
        fadd r36, r4, r5
        fadd r39, r4, r5
        fadd r3, r4, r5
        fadd r6, r4, r5
        fadd r24, r4, r5
        fadd r27, r4, r5
        fadd r30, r4, r5
        fadd r33, r4, r5
        fadd r36, r4, r5
        fadd r39, r4, r5
        fadd r3, r4, r5
        fadd r6, r4, r5
        fadd r24, r4, r5
        fadd r27, r4, r5
        fadd r30, r4, r5
        fadd r33, r4, r5
        fadd r36, r4, r5
        fadd r39, r4, r5
        fadd r3, r4, r5
        fadd r6, r4, r5
        fadd r24, r4, r5
        fadd r27, r4, r5
        fadd r30, r4, r5
        fadd r33, r4, r5
        fadd r36, r4, r5
        fadd r39, r4, r5
        fadd r3, r4, r5
        fadd r6, r4, r5
        fadd r24, r4, r5
        fadd r27, r4, r5
        fadd r30, r4, r5
        fadd r33, r4, r5
        fadd r36, r4, r5
        fadd r39, r4, r5
        fadd r3, r4, r5
        fadd r6, r4, r5
        fadd r24, r4, r5
        fadd r27, r4, r5
        fadd r30, r4, r5
        fadd r33, r4, r5
        fadd r36, r4, r5
        fadd r39, r4, r5
        halt
    )", config), std::move(launch));
    machine.enable_edge_probe();
    std::vector<std::tuple<uint64_t, uint64_t, uint64_t, uint32_t>> both_ready; // (cycle, owner 0, owner 1, cursor)
    while (!machine.done()) {
        const auto cycle = machine.cycle();
        if (machine.assembly_valid_count(0) == 16 && machine.assembly_valid_count(1) == 16 &&
            machine.assembly_owner(0).has_value() && machine.assembly_owner(1).has_value()) {
            both_ready.emplace_back(cycle, *machine.assembly_owner(0), *machine.assembly_owner(1),
                machine.rf_write_lsu_cursor());
        }
        tick_capped(machine);
    }
    std::map<uint64_t, std::vector<uint64_t>> writes_at; // cycle -> command ids written
    std::map<uint64_t, uint64_t> write_cycle;            // command id -> its RF write cycle
    for (const auto& event : machine.edge_events()) {
        if (event.edge != LsuEdge::rf_write_beat) continue;
        writes_at[event.cycle].push_back(event.id);
        write_cycle[event.id] = event.cycle;
    }
    for (const auto& [cycle, ids] : writes_at) {
        require(ids.size() == 1, "the LSU wrote " + std::to_string(ids.size()) + " RF beats in cycle " +
            std::to_string(cycle));
    }
    require(write_cycle.size() == 2, "two ldg.uni did not make two LSU RF writes");
    require(!both_ready.empty(), "the two assembly registers were never ready together (the test is vacuous)");
    for (const auto& [cycle, owner0, owner1, cursor] : both_ready) {
        const auto presented = cursor == 0 ? owner0 : owner1;
        const auto other = cursor == 0 ? owner1 : owner0;
        if (writes_at.count(cycle) == 1) {
            require(writes_at.at(cycle).front() == presented,
                "cycle " + std::to_string(cycle) + ": a register other than the one at the LSU cursor was written");
        }
        require(write_cycle.at(other) > write_cycle.at(presented),
            "the register behind the LSU cursor was written before the one at the cursor");
    }
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(machine.read_reg(0, lane, 9) == pattern_word(0), "the first ldg.uni's word");
        require(machine.read_reg(0, lane, 10) == pattern_word(128), "the second ldg.uni's word");
    }
}

// ---------------------------------------------------------------------------
// I2: shmem services and service slots (trash/rfc12.md sections 1-5 and 13;
// trash/i2-portA-slots-2026-10-11.md). Exact timelines are hand assembly
// (i1_asm) under a debug trace, where every granted round is an instant
// "<service> grant" (detail = client, value = slot). Several timelines use an
// engine event the host places into the TMA's event register
// (host_rmw_event). I3 re-derived every one of them: in I2 that RMW held
// A-read AND A-write from its read to its write; since I3 (rfc12 section 6)
// it pauses only new STORE transactions, from its start to its write grant,
// and holds A-read only while it waits for its read (i3_with_blocker below).
// ---------------------------------------------------------------------------

// A read's words reach the port output D + Lb cycles after its grant.
uint32_t i2_return(const nanosm::DeviceProfile& profile) {
    return profile.shmem_port_a_depth + profile.shmem_read_latency_cycles;
}

// I6a: issue to dependent issue behind a one-cycle ALU result (mov, sread,
// shli, ...): visible issue_to_visible after its producer's issue, sampled
// issue_register_stages after the consumer's (src/rf.hpp; 6 at canonical).
// The hand-assembled timelines below are built from it and from
// issue_to_dispatch (an lds / sts / mbar operation reaches its slot or
// register that long after its issue, and is served from the next cycle).
uint64_t alu_distance(const nanosm::DeviceProfile& profile) {
    nanosm::Instruction mov;
    mov.opcode = nanosm::Opcode::mov;
    return nanosm::issue_to_visible(mov, profile) - nanosm::issue_register_stages;
}

struct I2Grant {
    uint64_t cycle = 0;
    std::string service; // "A-read", "A-write", "B-write"
    std::string client;  // "simt", "probe", "rmw", "tma"
    uint32_t slot = 0;
    bool operator==(const I2Grant&) const = default;
};

std::string describe(const std::vector<I2Grant>& grants) {
    std::string text;
    for (const auto& grant : grants) {
        text += " c" + std::to_string(grant.cycle) + ":" + grant.service + "/" + grant.client +
            (grant.client == "simt" ? std::to_string(grant.slot) : "");
    }
    return text;
}

// The rounds granted (debug traces), by cycle; one service, or all.
std::vector<I2Grant> i2_grants(const nanosm::Machine& machine, const std::string& service = {}) {
    std::vector<I2Grant> grants;
    for (const auto& event : machine.trace().events()) {
        if (event.category != nanosm::TraceCategory::shmem || event.kind != nanosm::TraceEventKind::instant) continue;
        const auto name = event.name.substr(0, event.name.find(" grant"));
        if (!service.empty() && name != service) continue;
        grants.push_back({event.start_cycle, name, event.detail, static_cast<uint32_t>(event.value)});
    }
    std::stable_sort(grants.begin(), grants.end(), [](const auto& a, const auto& b) { return a.cycle < b.cycle; });
    return grants;
}

constexpr uint32_t i2_event_mbar = 4096; // a TMA-completion mbar for the host's events

// shmem words 0..2047 hold 0x100 + word; an mbar at i2_event_mbar waits for
// 64 transaction bytes (one host event completes its phase).
std::unique_ptr<nanosm::Machine> i2_machine(const nanosm::DeviceProfile& profile, const std::string& program,
    uint32_t warps = 1, bool hand_assembly = true) {
    nanosm::Launch launch;
    launch.active_warps = warps;
    auto& shmem = image_bytes(launch.shmem_image, 8192);
    for (uint32_t word = 0; word < 2048; ++word) write_u32(shmem, 4 * word, 0x100 + word);
    nanosm::MbarState mbar;
    mbar.valid = true;
    mbar.expected = 1;
    mbar.tx_bytes = 64;
    write_u64(shmem, i2_event_mbar, nanosm::encode_mbar_state(mbar));
    const auto kernel = hand_assembly ? hand_assemble(i1_asm(profile, program), profile)
                                      : compile(program, profile);
    return std::make_unique<nanosm::Machine>(profile, kernel, std::move(launch),
        nanosm::TraceOptions{.timeline = true, .debug = true});
}

// The host's event (call between ticks): a complete_tx of 64 bytes into the
// TMA's event register, held from the next cycle; it completes
// i2_event_mbar's phase. (Was i2_only_hold_both_a_services.) Since I3 an RMW
// STARTS as soon as the engine is idle and A-write's alternation lets it (no
// store waiting for its turn): from that cycle to its write grant the store
// pause holds every new store transaction; A-read is held only while the RMW
// waits for its read -- behind a store transaction under way on A-write
// (i3_with_blocker) or an A-read transaction under way -- and serves everyone
// from the read on.
void host_rmw_event(nanosm::Machine& machine) {
    machine.complete_mbar_tx(i2_event_mbar, 64, machine.read_mbar_state(i2_event_mbar).phase ? 1 : 0);
}

// I3: holding A-read for a few cycles takes a second warp. Warp 1 runs one
// sixteen-round sts (all lanes on one bank: words 6144 + 64 * lane);
// i3_hold_a_read places the host's event while that store is under way. The
// RMW starts the next cycle (the store started last, so the alternation bit
// gives the RMW the next start) and waits for its read until the cycle after
// the store's last round, starting nothing new on A-read meanwhile
// (select_a_read step 2). Warp 0 runs `body` after the dispatch (p6, r20,
// r21, r29 are the dispatch's and warp 1's).
std::string i3_with_blocker(const std::string& body) {
    return R"(
        sread r29, WARP_ID
        setp.ne p6, r29, 0
        bra.uni i3_blocker {pred=p6, stall=0, set=-, wait=-}
    )" + body + R"(
    i3_blocker: sread r20, LANE_ID
        shli r20, r20, 6
        mov r21, 0x66
        sts [r20 + 6144], r21 {pred=PT, stall=0, set=sb0, wait=-}
        halt {pred=PT, stall=0, set=-, wait=sb0}
    )";
}

// Runs until warp 1's blocking store has its first round granted (at g), then
// places the host's event: held from g + 1, it starts an RMW there. Returns g.
uint64_t i3_hold_a_read(nanosm::Machine& machine) {
    while (i2_grants(machine, "A-write").empty()) tick_capped(machine);
    host_rmw_event(machine);
    return machine.cycle() - 1;
}

void i2_run_until(nanosm::Machine& machine, uint64_t cycle) {
    while (machine.cycle() < cycle) tick_capped(machine);
}

// Same-warp order through shmem with the warp's reads serviced out of order
// (rfc12 section 4). Canonical D = Lb = C = 2, two load slots.
// I3 re-derivation: in I2 one warp's host event held A-read from the RMW's
// read to its write. Since I3 an RMW holds A-read only while it waits for
// its read, so warp 1's sixteen-round store gives it something to wait for
// (i3_with_blocker), and the slots are arranged by a depbar instead of RF
// write timing. The scenario is unchanged: two of warp 0's lds become ready
// while A-read is held, and the round-robin pointer takes the YOUNGER first.
//   c10 lds.uni P takes slot 0 and is granted at c11: the pointer moves past
//   slot 0. Warp 1's sts is admitted at c12, its rounds granted c13..c28; the
//   host's event starts an RMW at c14. P's RF write (c16) clears sb4 and
//   frees slot 0: OLD (waiting on sb4) takes slot 0 at c16, YOUNG slot 1 at
//   c17. The RMW's read is granted at c29 (r), the cycle after the store's
//   last round; then YOUNG (slot 1) at r + 1, OLD (slot 0) at r + 2 --
//   reordered reads.
//   WAR: the sts over OLD's words is ordered behind OLD (refused from c19
//   until OLD's grant), admitted at r + 2, and -- the store pause lasting to the RMW's
//   write grant w = r + D + Lb + C -- granted at w + 1: it reaches the RAM at
//   w + 1 + D, long after OLD's sample (r + 2 + D). The sts over YOUNG's
//   words, ordered behind it, follows at w + 2.
// Then a tight RAW (an lds granted the cycle after the sts's last grant sees
// the new words) and a WAW of two two-round stores.
// I6a re-derivation, the scenario unchanged; every access reaches its slot
// issue_to_dispatch (t) after its issue and is served from the next cycle,
// and the cycles are relative to issues in the trace:
//   lds.uni P issues at p, is granted at p + t + 1 (the pointer moves past
//   slot 0); its result is at the RF port x + 2 after its issue (x =
//   i1_lds_shift) and visible Wd + 1 later, when sb4 clears: OLD issues
//   then, YOUNG the cycle after; they take slots 0 and 1 at their dispatches.
//   Warp 1's sts is granted b .. b + 15 (b: its issue + t + 1); the RMW,
//   started at b + 1, reads at r = b + 16, then YOUNG at r + 1, OLD at r + 2.
//   WAR: the sts over OLD's words (tried from the cycle its r5 is visible)
//   is ordered behind OLD until OLD's grant r + 2; in that cycle warp 1's
//   halt (its sts's last round at the RAM) takes the issue slot, so the sts
//   issues at r + 3, is in its slot by the end of the store pause w = r + D
//   + Lb + C, and is granted at w + 1: it reaches the RAM long after OLD's
//   sample (r + 2 + D). The sts over YOUNG's words, ordered behind it,
//   issues at that grant and is granted t + 1 later.
// The tight RAW is now t + 1 cycles wide: the lds, ordered behind the sts
// until its grant, issues there and reaches A-read t + 1 later (a same-warp
// read can no longer be granted the cycle after the store). The WAW's second
// sts likewise follows the first's last round t + 1 later.
void test_i2_same_warp_order_with_reordered_reads() {
    const auto& profile = canonical_profile();
    require(i2_return(profile) == 4 && profile.shmem_load_slots == 2, "the timeline is derived for D = Lb = 2");
    const uint64_t t = nanosm::issue_to_dispatch(profile);
    const uint64_t c = profile.shmem_rmw_compute_cycles;
    auto machine = i2_machine(profile, i3_with_blocker(R"(
        sread r0, LANE_ID
        shli r0, r0, 2
        lds.uni r3, [r1 + 256] {pred=PT, stall=0, set=sb4, wait=-}
        lds r7, [r0 + 128] {pred=PT, stall=0, set=sb0, wait=sb4}
        lds r10, [r0 + 192] {pred=PT, stall=0, set=sb1, wait=-}
        mov r5, 0x5150
        sts [r0 + 128], r5 {pred=PT, stall=0, set=sb2, wait=-}
        sts [r0 + 192], r5 {pred=PT, stall=0, set=sb3, wait=-}
        halt {pred=PT, stall=0, set=-, wait=sb0|sb1|sb2|sb3}
    )"), 2);
    const auto blocker = i3_hold_a_read(*machine);
    machine->run();
    const uint64_t r = blocker + 16;            // the RMW's read: the cycle after the blocker's last round
    const uint64_t w = r + i2_return(profile) + c; // its write
    const auto reads = i2_grants(*machine, "A-read");
    constexpr uint32_t p_pc = 5 * 8, blocker_pc = 15 * 8, r5_pc = 8 * 8;
    const auto p = issued_at(*machine, p_pc).at(0);
    require(issued_at(*machine, blocker_pc) == std::vector<uint64_t>{blocker - t - 1} &&
            reads == std::vector<I2Grant>{{p + t + 1, "A-read", "simt", 0}, {r, "A-read", "rmw", 0},
                         {r + 1, "A-read", "simt", 1}, {r + 2, "A-read", "simt", 0}},
        "A-read grants (blocker from c" + std::to_string(blocker) + "):" + describe(reads) + " | all" +
            describe(i2_grants(*machine)) + " | issues" + issue_log(*machine) + " | ordered" +
            describe(refusals(*machine, 0, "ordered behind own access")) + " slots" +
            describe(refusals(*machine, 0, "load slots full")));
    constexpr uint32_t old_pc = 6 * 8, young_pc = 7 * 8, war_pc = 9 * 8;
    const auto old_issue = p + i1_lds_shift(profile) + 2 + profile.rf_write_depth_cycles + 1; // P visible: sb4 clear
    require(issued_at(*machine, old_pc) == std::vector<uint64_t>{old_issue} &&
            issued_at(*machine, young_pc) == std::vector<uint64_t>{old_issue + 1},
        "OLD did not issue as P's result became visible (sb4), or YOUNG the cycle after");
    // (At r + 2 warp 1's halt -- its sts's last round reached the RAM -- takes
    // the issue slot, so the WAR sts, admissible at OLD's grant, issues at
    // r + 3 and its refusal slice ends at r + 1.)
    const auto ordered = refusals(*machine, 0, "ordered behind own access");
    const auto war_first_try = issued_at(*machine, r5_pc).at(0) + alu_distance(profile);
    require(issued_at(*machine, war_pc) == std::vector<uint64_t>{r + 3} && ordered.size() == 2 &&
            ordered.front() == std::pair<uint64_t, uint64_t>{war_first_try, r + 2 - war_first_try},
        "the sts over OLD's words was not ordered behind OLD until OLD's grant:" +
            describe(ordered) + "; grants:" + (describe(i2_grants(*machine)) + " | issues" + issue_log(*machine)));
    std::vector<I2Grant> writes;
    for (uint64_t round = 0; round < 16; ++round) writes.push_back({blocker + round, "A-write", "simt", 0});
    writes.push_back({w, "A-write", "rmw", 0});
    writes.push_back({w + 1, "A-write", "simt", 0});
    writes.push_back({w + t + 2, "A-write", "simt", 0});
    require(i2_grants(*machine, "A-write") == writes, "A-write grants:" + describe(i2_grants(*machine, "A-write")));
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(machine->read_reg(0, lane, 7) == 0x100 + 32 + lane, "OLD did not read the words before the sts");
        require(machine->read_reg(0, lane, 10) == 0x100 + 48 + lane, "YOUNG did not read the words before the sts");
        require(machine->read_shmem_u32(128 + 4 * lane) == 0x5150 && machine->read_shmem_u32(192 + 4 * lane) == 0x5150,
            "sts data");
    }

    // Tight RAW: the sts (issued at s = fetch_to_issue + 2 alu_distance + 1:
    // the first instruction issues at fetch_to_issue, and the sts needs the
    // mov's r5) has its one round granted at s + t + 1 and reaching the RAM D
    // later; the lds, admitted at that grant, is granted t + 1 later and
    // samples D after that.
    auto raw = i2_machine(profile, R"(
        sread r0, LANE_ID
        shli r0, r0, 2
        mov r5, 0x777
        sts [r0 + 256], r5 {pred=PT, stall=0, set=sb0, wait=-}
        lds r6, [r0 + 256] {pred=PT, stall=0, set=sb1, wait=-}
        halt {pred=PT, stall=0, set=-, wait=sb0|sb1}
    )");
    raw->run();
    const uint64_t raw_store = nanosm::fetch_to_issue + 2 * alu_distance(profile) + 1 + t + 1;
    require(i2_grants(*raw) == std::vector<I2Grant>{{raw_store, "A-write", "simt", 0},
                                   {raw_store + t + 1, "A-read", "simt", 0}},
        "the lds was not granted t + 1 after the sts's last grant:" + describe(i2_grants(*raw)));
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(raw->read_reg(0, lane, 6) == 0x777, "RAW: the lds did not see the sts (write path deeper than read?)");
    }

    // WAW: lanes l and l + 8 share a bank (two rounds). The second sts waits
    // for the first's last round, issues there and its rounds follow t + 1
    // later.
    auto waw = i2_machine(profile, R"(
        sread r0, LANE_ID
        andi r1, r0, 7
        shri r2, r0, 3
        shli r1, r1, 2
        shli r2, r2, 6
        add r1, r1, r2
        mov r5, 0x111
        mov r6, 0x222
        sts [r1 + 3072], r5 {pred=PT, stall=0, set=sb0, wait=-}
        sts [r1 + 3072], r6 {pred=PT, stall=0, set=sb1, wait=-}
        halt {pred=PT, stall=0, set=-, wait=sb0|sb1}
    )");
    waw->run();
    const auto stores = i2_grants(*waw, "A-write");
    require(stores.size() == 4 && stores[1].cycle == stores[0].cycle + 1 && stores[2].cycle == stores[1].cycle + t + 1 &&
            stores[3].cycle == stores[2].cycle + 1,
        "the second sts did not follow the first's last round:" + describe(stores));
    require(refusals(*waw, 0, "ordered behind own access").size() == 1, "the second sts was not ordered once");
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(waw->read_shmem_u32((lane % 8) * 4 + (lane / 8) * 64 + 3072) == 0x222, "WAW: the older store won");
    }
}

// Divergence (rfc12 section 3): rounds come from active lanes only; a masked
// lane is no access (its word, and an lds's destination lane, are
// untouched); an lds / sts with no active lane takes no slot, sets no depbar
// and makes no request; a uniform operation under a partial mask is the
// sanitizer's PANIC (backlog D12; the machine's quiet lane-0 behavior:
// "D12: uniform ops under a partial mask").
void test_i2_divergence() {
    const std::string body = R"(
        sread r0, LANE_ID
        shli r0, r0, 6
        mov r3, 0x77
        mov r5, 0x55
        sts [r0], r5 {pred=PT, stall=0, set=sb0, wait=-}
        lds r3, [r0 + 4] {pred=PT, stall=0, set=sb1, wait=-}
        halt {pred=PT, stall=0, set=-, wait=sb0|sb1}
    )"; // every lane on bank 0: a full mask is sixteen rounds
    const auto& profile = canonical_profile();
    {
        auto machine = i2_machine(profile, body);
        machine->test_set_active_mask(0, 0x00f0); // lanes 4..7
        machine->run();
        require(shmem_request_rounds(*machine) == std::vector<uint32_t>{4, 4}, "four active lanes are four rounds");
        for (uint32_t lane = 0; lane < 16; ++lane) {
            const bool active = lane >= 4 && lane < 8;
            require(machine->read_shmem_u32(64 * lane) == (active ? 0x55 : 0x100 + 16 * lane), "sts lane");
            require(machine->read_reg(0, lane, 3) == (active ? 0x100 + 16 * lane + 1 : 0x77), "lds lane");
        }
    }
    {
        auto machine = i2_machine(profile, body);
        machine->test_set_active_mask(0, 0);
        bool slot_taken = false;
        while (!machine->done()) {
            tick_capped(*machine);
            slot_taken |= machine->load_slot_held(0) || machine->load_slot_held(1) || machine->store_slot_held(0) ||
                machine->store_slot_held(1);
        }
        require(!slot_taken, "an access with no active lane took a slot");
        require(shmem_request_rounds(*machine).empty() && i2_grants(*machine).empty(), "an absent access was served");
        const auto lds_issue = issued_at(*machine, 5 * 8);
        require(lds_issue.size() == 1 && issued_at(*machine, 6 * 8) == std::vector<uint64_t>{lds_issue[0] + 1},
            "the halt waited on a depbar nothing set");
        for (uint32_t lane = 0; lane < 16; ++lane) {
            require(machine->read_shmem_u32(64 * lane) == 0x100 + 16 * lane && machine->read_reg(0, lane, 3) == 0x77,
                "an absent access changed memory or its destination");
        }
    }
#if NANOSM_SANITIZER
    for (const std::string uniform :
             {"lds.uni r3, [r1] {pred=PT, stall=0, set=sb0, wait=-}", "mbar.arrive [r2]", "mbar.test_wait p0, [r2], 0"}) {
        auto machine = i2_machine(profile, "mov r2, 4096\n" + uniform + "\nhalt\n");
        machine->test_set_active_mask(0, 0xfffe);
        try {
            machine->run();
            throw std::runtime_error(uniform + " ran under a partial mask");
        } catch (const nanosm::SimError& error) {
            require_mentions(error.what(), {"uniform operation under a partial active mask"}, uniform);
        }
    }
#endif
}

// shmem.store_slots: 1 and 2 both work (rfc12 section 2). Two warps each
// store two rounds; with one slot the second warp's sts waits for the slot
// (refused "store slots full") until the first's last round is granted,
// then takes the slot in that cycle; with two it waits in its own slot. The
// memory is the same either way. With two slots the rounds are back to back;
// with one, since I6a, the second sts issues as the first's last round frees
// the slot and reaches it at dispatch, so its first round is
// issue_to_dispatch + 1 after that.
void test_i2_store_slots_one_and_two() {
    const std::string program = R"(
        sread r29, WARP_ID
        sread r0, LANE_ID
        andi r1, r0, 7
        shri r2, r0, 3
        shli r1, r1, 2
        shli r2, r2, 6
        add r1, r1, r2
        shli r3, r29, 7
        add r1, r1, r3
        addi r5, r29, 0x300
        sts [r1 + 2048], r5
        halt
    )";
    std::vector<std::vector<uint32_t>> memory;
    for (const uint32_t slots : {1u, 2u}) {
        auto profile = canonical_profile();
        profile.shmem_store_slots = slots;
        auto machine = i2_machine(profile, program, 2, false);
        machine->run();
        const auto stores = i2_grants(*machine, "A-write");
        require(stores.size() == 4, "four store rounds");
        for (std::size_t round = 1; round < 4; ++round) {
            const auto gap = slots == 1 && round == 2 ? 1 + nanosm::issue_to_dispatch(profile) : 1;
            require(stores[round].cycle == stores[round - 1].cycle + gap,
                std::to_string(slots) + " store slots: the stores' rounds were not where expected:" + describe(stores));
        }
        const auto waits = machine->trace().summary().shmem_store_slot_waits;
        require(slots == 1 ? waits != 0 : waits == 0, std::to_string(slots) + " store slots: store_slot_full=" +
            std::to_string(waits));
        std::vector<uint32_t> words;
        for (uint32_t word = 0; word < 64; ++word) words.push_back(machine->read_shmem_u32(2048 + 4 * word));
        memory.push_back(words);
    }
    require(memory[0] == memory[1], "one and two store slots stored different data");
    for (uint32_t warp = 0; warp < 2; ++warp) {
        for (uint32_t lane = 0; lane < 16; ++lane) {
            require(memory[0][(lane % 8) + (lane / 8) * 16 + warp * 32] == 0x300 + warp, "store data");
        }
    }
}

// One store slot, reused while the previous store is still in the A-write
// pipeline (rfc12 section 3), and done() when the last instruction is a
// store. sts X (issued at x) reaches the slot at its dispatch (x +
// issue_to_dispatch, I6a) and is granted the next cycle, gx, which frees the
// slot; sts Y issues then and is granted gy = gx + issue_to_dispatch + 1.
// X reaches the RAM at gx + D, Y at gy + D; the halt issues at gx + 1, and
// the machine is done only after Y's write edge.
void test_i2_store_slot_reuse_and_done() {
    auto profile = canonical_profile();
    profile.shmem_store_slots = 1;
    const auto d = profile.shmem_port_a_depth;
    const auto dispatch = nanosm::issue_to_dispatch(profile);
    auto machine = i2_machine(profile, R"(
        sread r0, LANE_ID
        shli r0, r0, 2
        mov r5, 0xa1
        sts [r0 + 512], r5 {pred=PT, stall=0, set=sb0, wait=-}
        sts [r0 + 576], r5 {pred=PT, stall=0, set=sb1, wait=-}
        halt
    )");
    std::map<uint64_t, std::tuple<bool, bool, bool, bool>> after; // cycle -> slot held, X landed, Y landed, done
    while (!machine->done()) {
        const auto cycle = machine->cycle();
        tick_capped(*machine);
        after[cycle] = {machine->store_slot_held(0), machine->read_shmem_u32(512) == 0xa1,
            machine->read_shmem_u32(576) == 0xa1, machine->done()};
    }
    const auto x = issued_at(*machine, 3 * 8).at(0);
    const auto gx = x + dispatch + 1;
    const auto gy = gx + dispatch + 1;
    require(i2_grants(*machine, "A-write") == std::vector<I2Grant>{{gx, "A-write", "simt", 0}, {gy, "A-write", "simt", 0}},
        "the two stores were not granted at gx and gy:" + (describe(i2_grants(*machine)) + " | issues" + issue_log(*machine)));
    require(issued_at(*machine, 4 * 8) == std::vector<uint64_t>{gx}, "Y did not issue as X's last round freed the slot");
    require(std::get<0>(after.at(gx - 1)) && !std::get<0>(after.at(gx)) && std::get<0>(after.at(gy - 1)) &&
            !std::get<0>(after.at(gy)),
        "the slot was not held from each store's dispatch to its last round");
    require(!std::get<1>(after.at(gx + d - 1)) && std::get<1>(after.at(gx + d)), "X did not reach the RAM at gx + D");
    require(!std::get<2>(after.at(gy + d - 1)) && std::get<2>(after.at(gy + d)), "Y did not reach the RAM at gy + D");
    require(issued_at(*machine, 5 * 8) == std::vector<uint64_t>{gx + 1}, "halt");
    require(!std::get<3>(after.at(gy + d - 1)) && std::get<3>(after.at(gy + d)),
        "done() did not cover the store travelling to its write edge");
    for (uint32_t lane = 0; lane < 16; ++lane) require(machine->read_shmem_u32(576 + 4 * lane) == 0xa1, "Y's data");
}

// A-read and A-write are independent services: one warp's sixteen-round lds
// and another warp's sixteen-round sts are granted in the same cycles.
// I6b-2: the two warps issue their bra.uni one cycle apart; the reader's lds
// follows its not-taken branch by branch_not_taken_interval, the writer's sts
// its mov at the branch target by alu_distance, and the target the taken
// branch by branch_taken_interval (src/rf.hpp) plus one cycle of fetch
// contention (two-word packets): the reader's lds is the second half of its
// packet, so issuing it pops the packet and the reader asks for a fetch in
// the same cycle the writer first asks for its target's packet; the
// round-robin cursor, last moved past the writer by the writer's own
// previous fetch, serves the reader first. So the sts issues
// 1 + (taken + 1) - not taken + alu_distance after the lds (10 at
// canonical), each is granted t + 1 after its issue, and the two
// sixteen-cycle windows share 16 minus that many cycles -- in every one of
// which both services grant.
void test_i2_a_read_and_a_write_same_cycle() {
    auto machine = i2_machine(canonical_profile(), R"(
        sread r29, WARP_ID
        sread r0, LANE_ID
        shli r0, r0, 6
        setp.ne p0, r29, 0
        @p0 bra.uni writer
        lds r3, [r0]
        halt
    writer:
        mov r5, 0x44
        sts [r0 + 32], r5
        halt
    )", 2, false);
    machine->run();
    std::set<uint64_t> read_cycles;
    for (const auto& grant : i2_grants(*machine, "A-read")) read_cycles.insert(grant.cycle);
    uint32_t together = 0;
    for (const auto& grant : i2_grants(*machine, "A-write")) together += read_cycles.contains(grant.cycle) ? 1 : 0;
    const auto apart = 1 + (nanosm::branch_taken_interval + 1) - nanosm::branch_not_taken_interval +
        alu_distance(canonical_profile());
    require(issued_at(*machine, 8 * 8).at(0) == issued_at(*machine, 5 * 8).at(0) + apart,
        "the sts did not issue 1 + (taken + 1) - not taken + alu_distance after the lds:" + issue_log(*machine));
    require(read_cycles.size() == 16 && together == 16 - apart,
        "A-read and A-write were not granted in the same cycles (" + std::to_string(together) + ")");
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(machine->read_reg(0, lane, 3) == 0x100 + 16 * lane, "the load's words");
        require(machine->read_shmem_u32(64 * lane + 32) == 0x44, "the store's words");
    }
}

// rfc12 section 5: ONE round-robin pointer over the load slots, the probe
// register and the TMA's reads, moved only by its own grants (not by the
// RMW's). A-read is held while two lds.uni and the TMA's descriptor read
// become ready. I3 re-derivation: the hold is an RMW waiting for its read
// behind warp 1's sixteen-round store (i3_with_blocker; in I2 one warp's RMW
// held A-read from its read to its write); warp 0's stall puts its three
// requests inside it. Everything is relative to the RMW's read r.
//   pointer at 0 (nothing granted yet): slot 0, slot 1, then the TMA;
//   pointer past slot 1 (two earlier lds.uni, granted before the hold): the
//   TMA, then slot 0, slot 1.
// I6a: a request becomes ready for A-read at its dispatch + 1, issue +
// issue_to_dispatch + 1, and warp 1's blocking store starts later (its
// sources' fixed distances, then its dispatch). The stall (was 6 / 5)
// re-places warp 0's three requests: the tma.async issues at c31 in both
// runs (b = c32: ready from b + 3, inside the hold that starts at b + 1).
// Two-word packets: in the primed run warp 0's first priming lds.uni (at
// c14) is the second half of its packet, so warp 0 asks for a fetch at c15,
// when warp 1 first asks for its branch target's packet; the round-robin
// serves warp 0 first, warp 1's target issues a cycle later and its
// blocking store starts at b = c36. The primed stall is therefore 17 (was
// 15): the tma.async issues at c33 and is ready at c37 = b + 1.
void test_i2_round_robin_tma_reads_and_loads() {
    const auto& profile = canonical_profile();
    require(i2_return(profile) == 4, "the timeline is derived for D = Lb = 2");
    const auto run = [&](bool primed) {
        nanosm::Launch launch;
        launch.active_warps = 2;
        auto& shmem = image_bytes(launch.shmem_image, 8192);
        nanosm::MbarState mbar;
        mbar.valid = true;
        mbar.expected = 1;
        mbar.tx_bytes = 64;
        write_u64(shmem, i2_event_mbar, nanosm::encode_mbar_state(mbar));
        write_rank1_load(shmem, 5120, 0, 16, 2048, 4104);
        fill_pattern(image_bytes(launch.gmem_image, 64), 0, 16, 0);
        const std::string priming = primed ? "lds.uni r5, [r1 + 8] {pred=PT, stall=0, set=sb3, wait=-}\n"
                                             "lds.uni r6, [r1 + 12] {pred=PT, stall=0, set=sb4, wait=-}\n"
                                           : "";
        const std::string stall = primed ? "17" : "18";
        nanosm::Machine machine(profile, hand_assemble(i1_asm(profile, i3_with_blocker(priming +
            "mov r31, 0 {pred=PT, stall=" + stall + ", set=-, wait=-}\n" + R"(
            tma.async [5120], r8
            lds.uni r3, [r1] {pred=PT, stall=0, set=sb0, wait=-}
            lds.uni r4, [r1 + 4] {pred=PT, stall=0, set=sb1, wait=-}
            halt {pred=PT, stall=0, set=-, wait=sb0|sb1}
        )")), profile), std::move(launch), nanosm::TraceOptions{.timeline = true, .debug = true});
        const auto blocker = i3_hold_a_read(machine);
        machine.run();
        const uint64_t r = blocker + 16; // the RMW's read
        // warp 0's tma.async and two lds.uni (the last three before halt)
        // became ready (dispatch + 1) while A-read was held
        const uint32_t tma_pc = (primed ? 6 : 4) * 8;
        const auto ready = [&](uint32_t pc) { return issued_at(machine, pc).front() + nanosm::issue_to_dispatch(profile) + 1; };
        for (const uint32_t pc : {tma_pc, tma_pc + 8, tma_pc + 16}) {
            require(issued_at(machine, pc).size() == 1 && ready(pc) < r && ready(pc) > blocker,
                "a request did not become ready inside the hold (blocker c" + std::to_string(blocker) + "):" +
                    describe(i2_grants(machine)) + " | issues" + issue_log(machine));
        }
        auto reads = i2_grants(machine, "A-read");
        reads.resize(std::min<std::size_t>(reads.size(), primed ? 6 : 4));
        return std::pair{reads, r};
    };
    const auto [plain, r0] = run(false);
    require(plain == std::vector<I2Grant>{{r0, "A-read", "rmw", 0}, {r0 + 1, "A-read", "simt", 0},
                         {r0 + 2, "A-read", "simt", 1}, {r0 + 3, "A-read", "tma", 0}},
        "pointer at 0: not slot 0, slot 1, TMA:" + describe(plain));
    const auto [primed, r1] = run(true);
    require(primed.size() == 6 && primed[0].slot == 0 && primed[1].slot == 1 && primed[1].cycle < r1 - 16 &&
            std::vector<I2Grant>(primed.begin() + 2, primed.end()) ==
                std::vector<I2Grant>{{r1, "A-read", "rmw", 0}, {r1 + 1, "A-read", "tma", 0},
                    {r1 + 2, "A-read", "simt", 0}, {r1 + 3, "A-read", "simt", 1}},
        "pointer past slot 1: not TMA, slot 0, slot 1:" + describe(primed));
}

// TMA placement is the port-B write (rfc11 section 0): it runs in the same
// cycles as another warp's sts stream (A-write) and the MMA's operand fetch
// (port-B read). Everything lands correctly. I6b-2: the hammer's stores are
// sixteen rounds each (all lanes on one bank), so A-write is busy 16 of
// every t + 16 cycles; a stream of one-round stores (one per t + 1 cycles,
// each ordered behind the last) met the placements (one per 8 cycles) only
// at a lucky phase, which the front end's branch timing moved.
void test_i2_tma_placement_on_port_b() {
    nanosm::Launch launch;
    launch.active_warps = 2;
    auto& shmem = image_bytes(launch.shmem_image, 8192);
    for (uint32_t i = 0; i < 1024; ++i) write_u16(shmem, 2 * i, 0x3c00); // A and B: 1.0
    write_rank1_load(shmem, 5120, 0, 256, 2048, 4104);
    fill_pattern(image_bytes(launch.gmem_image, 1024), 0, 256, 0);
    std::string program = R"(
        .acc C, 0, 0, 64, 64
        sread r29, WARP_ID
        setp.ne p0, r29, 0
        @p0 bra.uni hammer
        mov r0, 0
        mov r1, 1024
        mov r4, 4096
        mbar.init [r4], 1
        mma.async.init C, [r0], [r1]
        mma.commit [r4]
        mov r8, 0
        mov r9, 0
        mov r10, 0
        mov r11, 4104
        tma.async [5120], r8
    tma_wait:
        mbar.try_wait p1, [r11], 0
        @!p1 bra.uni tma_wait
    mma_wait:
        mbar.try_wait p2, [r4], 0
        @!p2 bra.uni mma_wait
        mov r12, 0
        mov r13, 0
        acc.ld r16, C, r12
        halt
    hammer:
        sread r20, LANE_ID
        shli r20, r20, 6
    )";
    for (int i = 0; i < 50; ++i) program += "sts [r20 + 6144], r20\n";
    program += "halt\n";
    nanosm::Machine machine(canonical_profile(), compile(program), std::move(launch),
        nanosm::TraceOptions{.timeline = true, .debug = true});
    machine.run(20000);
    std::set<uint64_t> placements, stores, fetches;
    for (const auto& grant : i2_grants(machine)) {
        if (grant.service == "B-write") placements.insert(grant.cycle);
        if (grant.service == "A-write" && grant.client == "simt") stores.insert(grant.cycle);
    }
    for (const auto& event : machine.trace().events()) {
        if (event.name == "MMA input step") fetches.insert(event.start_cycle);
    }
    uint32_t all_three = 0;
    for (const auto cycle : placements) all_three += stores.contains(cycle) && fetches.contains(cycle) ? 1 : 0;
    require(placements.size() == 32 && all_three != 0,
        "no cycle had a placement, an sts round and an MMA fetch together (" + std::to_string(all_three) + ")");
    for (uint32_t i = 0; i < 256; ++i) require(machine.read_shmem_u32(2048 + 4 * i) == pattern_word(i), "TMA data");
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(machine.read_shmem_u32(6144 + 64 * lane) == 64 * lane, "sts data");
        require(machine.read_reg(0, lane, 16) == nanosm::float_bits(8.0f), "MMA result");
    }
}

// Was "I2: RMW holds both A services" (I2's conservative RMW: nothing else on
// A from its read to its write). Re-derived for I3's global store pause
// (rfc12 section 6) on the same program. Both warps: a sixteen-round sts
// (all lanes on one bank), then an lds of other words, ordered behind it.
// Warp 0's sts is granted c7..c22; warp 1's waits in its slot. The host's
// event is held from c9: the RMW starts there -- the store pause begins,
// warp 0's sts keeps its rounds -- and its read is granted at c23, strictly
// after that sts's last round. Reads continue: warp 0's lds (admitted at
// its sts's last grant) is granted from c24, during the RMW. The RMW's write
// is granted at w = c23 + D + Lb + C; warp 1's sts, paused since c10, at
// w + 1. No store round is granted between the RMW's read and its write.
void test_i3_store_pause_two_warps() {
    const auto& profile = canonical_profile();
    const uint64_t c = profile.shmem_rmw_compute_cycles;
    auto machine = i2_machine(profile, R"(
        sread r0, LANE_ID
        shli r0, r0, 6
        mov r5, 0x5a
        sts [r0 + 4], r5 {pred=PT, stall=0, set=sb0, wait=-}
        lds r6, [r0 + 8] {pred=PT, stall=0, set=sb1, wait=-}
        halt {pred=PT, stall=0, set=-, wait=sb0|sb1}
    )", 2);
    // The host event arrives while warp 0's sts is under way.
    while (i2_grants(*machine, "A-write").empty()) tick_capped(*machine);
    tick_capped(*machine);
    host_rmw_event(*machine);
    const auto start = machine->cycle();
    machine->run();
    const auto grants = i2_grants(*machine);
    std::vector<uint64_t> store0, store1, load0;
    uint64_t read = 0;
    uint64_t write = 0;
    for (const auto& grant : grants) {
        if (grant.client == "rmw") (grant.service == "A-read" ? read : write) = grant.cycle;
        if (grant.client != "simt") continue;
        if (grant.service == "A-write") (store0.size() < 16 ? store0 : store1).push_back(grant.cycle);
        if (grant.service == "A-read" && load0.size() < 16) load0.push_back(grant.cycle);
    }
    const auto rounds = [](uint64_t first) {
        std::vector<uint64_t> cycles;
        for (uint64_t round = 0; round < 16; ++round) cycles.push_back(first + round);
        return cycles;
    };
    // Relative to warp 0's first store round g (I6a moved the program, not
    // the scenario): the event is held from g + 2 and starts the RMW there;
    // the RMW reads the cycle after the store's last round (g + 16). Warp 0's
    // lds is ordered behind its sts until that last round, issues then and
    // reaches its slot at dispatch: its rounds start issue_to_dispatch + 1
    // after the store's last round, i.e. read + issue_to_dispatch.
    require(store0.size() == 16, "warp 0's sts:" + describe(grants));
    const auto g = store0.front();
    require(start == g + 2 && mbar_rmw_services(*machine).front().first == start,
        "the RMW did not start in the cycle the event was held (g + 2)");
    require(store0 == rounds(g) && read == g + 16 && write == read + i2_return(profile) + c,
        "the RMW's read was not granted the cycle after the store's last round, or its write D + Lb + C "
        "later:" + describe(grants));
    require(load0 == rounds(read + nanosm::issue_to_dispatch(profile)),
        "reads did not continue during the RMW: warp 0's lds (ordered behind its sts, then dispatched) was not "
        "granted from read + issue_to_dispatch:" + describe(grants));
    require(store1 == rounds(write + 1), "warp 1's paused sts was not granted from the cycle after the RMW's "
        "write grant:" + describe(grants));
    require(machine->trace().summary().shmem_store_paused == write - start,
        "store_paused should count every pause cycle with warp 1's sts waiting (c10..w)");
    require(machine->read_mbar_state(i2_event_mbar).phase, "the RMW's update was lost");
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(machine->read_shmem_u32(64 * lane + 4) == 0x5a, "sts data");
        require(machine->read_reg(0, lane, 6) == 0x100 + 16 * lane + 2, "lds data");
    }
}

// shmem.port_a_depth (D), shmem.read_latency_cycles (Lb) and
// shmem.store_slots: optional in the JSON (default 2), never 0 (every latency
// is a nonzero number of register stages), part of the fingerprint. A
// deeper read latency moves an lds's result by exactly the difference.
void test_i2_profile_fields() {
    std::ifstream input("profiles/nanosm-v0.json");
    nlohmann::json document;
    input >> document;
    const auto canonical = nanosm::parse_device_profile(document.dump());
    require(canonical.shmem_port_a_depth == 2 && canonical.shmem_read_latency_cycles == 2 &&
            canonical.shmem_store_slots == 2, "the canonical profile does not declare D = Lb = 2 and two store slots");
    for (const char* field : {"port_a_depth", "read_latency_cycles", "store_slots"}) {
        auto defaulted = document;
        defaulted["shmem"].erase(field);
        require(nanosm::device_profile_fingerprint(nanosm::parse_device_profile(defaulted.dump())) ==
                nanosm::device_profile_fingerprint(canonical), std::string(field) + ": the default is not 2");
        for (const auto value : {nlohmann::json(0), nlohmann::json(-1), nlohmann::json(1.5)}) {
            auto modified = document;
            modified["shmem"][field] = value;
            try {
                (void)nanosm::parse_device_profile(modified.dump());
                throw std::runtime_error(std::string(field) + " = " + value.dump() + " was accepted");
            } catch (const nanosm::ProfileError&) {}
        }
    }
    for (const auto field : {&nanosm::DeviceProfile::shmem_port_a_depth,
             &nanosm::DeviceProfile::shmem_read_latency_cycles, &nanosm::DeviceProfile::shmem_store_slots}) {
        auto zero = canonical_profile();
        zero.*field = 0;
        try {
            nanosm::validate_device_profile(zero);
            throw std::runtime_error("a zero shmem field passed validation");
        } catch (const nanosm::ProfileError&) {}
        auto three = canonical_profile();
        three.*field = 3;
        require(nanosm::device_profile_fingerprint(three) != nanosm::device_profile_fingerprint(canonical_profile()),
            "a shmem field was omitted from the fingerprint");
    }
    const auto result_cycle = [](const nanosm::DeviceProfile& profile) {
        auto machine = i2_machine(profile, "lds.uni r3, [r1] {pred=PT, stall=0, set=sb0, wait=-}\n"
                                           "halt {pred=PT, stall=0, set=-, wait=sb0}\n");
        const auto written = i1_run(*machine, {3});
        return written.at(3).front();
    };
    // I6a: the lds.uni at c0 reaches its slot at dispatch, its round is
    // granted the cycle after, its word comes back D + Lb later, the gather
    // register is complete one cycle later (its RF grant), and the write lands
    // Wd after that. (I6b-2: it issues at fetch_to_issue, not c0.)
    auto slow = canonical_profile();
    slow.shmem_read_latency_cycles = 7;
    const auto& canonical_run = canonical_profile();
    require(result_cycle(canonical_run) == nanosm::fetch_to_issue + nanosm::issue_to_dispatch(canonical_run) + 1 +
                i2_return(canonical_run) + 1 + canonical_run.rf_write_depth_cycles &&
            result_cycle(slow) == result_cycle(canonical_profile()) + 5,
        "an lds result did not arrive dispatch + 1 + D + Lb + 1 (the gather register) + Wd after its issue");
}

// READ_FIRST (rfc12 section 1): a read and a write of one word reaching the
// RAM in the same cycle -- on different services -- read the old word, with
// no collision machinery and no PANIC. I3 re-derivation: the host's event
// starts an RMW at c0 whose store pause (a long Lb stretches it) holds warp
// 0's sts in its slot until the RMW's write grant w; the sts is granted at
// w + 1. Reads are not paused, so warp 1's lds of the same words is placed by
// its stall count (in I2 the RMW held both services and released both at
// once). Swept over that stall: the lds sees the new words iff it is granted
// after the sts (it samples after the sts's write edge); granted in the same
// cycle it sees the old words (READ_FIRST). The sweep must include that tie.
// The MMA's port-B read crossing an A-write: no PANIC at any alignment (the
// old same-cycle C5 diagnostic is gone). I2.1 (Fable F3: the sweep asserted
// only no-PANIC and a range): every pad's row-0 result is now the EXACT sum
// of the A words the fetch sampled -- one bank, so the sts's lane w writes
// word w in round w, and steps 0 and 4 read row 0's words 0, 1 and 2, 3 in
// their first two rounds. A word is new iff its sts round reached the RAM in
// an earlier cycle than the fetch round's RAM sample (the same cycle is
// READ_FIRST: old). I7: that sample is the round's grant + D. The sweep must
// cross that boundary, a tie included.
// I4 (rfc12 sections 10 and 11): the MMA sweep issues the sts and then the
// mma.async.init reading its words with NO fence.sh -- a deliberate missing
// fence, kept to measure READ_FIRST. The hardware is unchanged, so with the
// sanitizer compiled out the sweep asserts exactly what it did. With the
// sanitizer, the missing-fence check PANICs at the MMA's accept exactly when
// a store round has not reached the RAM by that cycle (every old-word
// outcome and the tie are among those); the other pads run and read only
// new words. The sweep must have both outcomes.
void test_i2_read_first_across_services() {
    auto profile = canonical_profile();
    profile.shmem_read_latency_cycles = 24;
    const uint64_t w = profile.shmem_port_a_depth + profile.shmem_read_latency_cycles + profile.shmem_rmw_compute_cycles;
    std::set<std::string> sides;
    for (uint32_t stall = 0; stall <= 31; ++stall) {
        auto machine = i2_machine(profile, R"(
            sread r29, WARP_ID
            sread r0, LANE_ID
            shli r0, r0, 2
            setp.ne p6, r29, 0
            bra.uni reader {pred=p6, stall=0, set=-, wait=-}
            mov r5, 0xee
            sts [r0 + 512], r5 {pred=PT, stall=0, set=sb0, wait=-}
            halt {pred=PT, stall=0, set=-, wait=sb0}
        reader: mov r30, 0 {pred=PT, stall=)" + std::to_string(stall) + R"(, set=-, wait=-}
            lds r6, [r0 + 512] {pred=PT, stall=0, set=sb0, wait=-}
            halt {pred=PT, stall=0, set=-, wait=sb0}
        )", 2);
        host_rmw_event(*machine);
        machine->run();
        const auto grants = i2_grants(*machine);
        std::map<std::string, uint64_t> simt;
        for (const auto& grant : grants) if (grant.client == "simt") simt[grant.service] = grant.cycle;
        const auto what = "stall " + std::to_string(stall) + ":" + describe(grants);
        require(simt.size() == 2 && simt.at("A-write") == w + 1, what + ": the sts was not granted at w + 1");
        const bool after = simt.at("A-read") > simt.at("A-write");
        sides.insert(after ? "after" : simt.at("A-read") == simt.at("A-write") ? "tie" : "before");
        for (uint32_t lane = 0; lane < 16; ++lane) {
            require(machine->read_reg(1, lane, 6) == (after ? 0xee : 0x100 + 128 + lane),
                what + (after ? ": the lds missed the sts" : ": READ_FIRST: the lds did not read the old word"));
            require(machine->read_shmem_u32(512 + 4 * lane) == 0xee, what + ": the sts was lost");
        }
    }
    require(sides == std::set<std::string>{"before", "tie", "after"}, "the sweep did not cross the same-cycle grant");

    // I7: port B now has port A's depth, so a store round and a fetch round
    // reach the RAM in grant order and meet there only when granted in one
    // cycle. A program-order sweep alone (the sts, then pads, then the MMA)
    // never granted the fetch first. A host event's RMW now pauses the sts in
    // its slot until w + 1 (w = D + Lb + C; Lb 41 makes w + 1 later than the
    // first pads' fetch) while the pads move the fetch across that grant.
    auto dense = dense_one_bank_profile(4);
    dense.shmem_read_latency_cycles = 41;
    nanosm::validate_device_profile(dense);
    const uint64_t depth = dense.shmem_port_a_depth;
    const uint64_t paused_until = depth + dense.shmem_read_latency_cycles + dense.shmem_rmw_compute_cycles;
    constexpr uint32_t event_mbar = 960;
    std::set<uint32_t> outcomes;
    bool tie = false;
    uint32_t panicked = 0;
    uint32_t clean = 0;
    // I4: up to 20 pads (was 8); I7: up to 40, so the sanitizer build sees
    // the paused sixteen-round sts reach the RAM in full before an accept.
    for (uint32_t pad = 0; pad <= 40; ++pad) {
        std::ostringstream qtx;
        // I7: the commit mbar is in the image (an mbar.init would be an RMW
        // behind the host's, ordering the sts after its long write).
        qtx << ".acc C, 0, 0, 16, 4\n"
            << "mov r0, 0\nmov r1, 512\nmov r4, 896\n"
            << "sread r20, LANE_ID\nshli r20, r20, 2\n"
            << "mov r21, 16384\nshli r22, r21, 16\nor r21, r21, r22\n" // two fp16 2.0 halves
            << "sts [r20], r21\n";
        for (uint32_t i = 0; i < pad; ++i) qtx << "mov r22, 1\n";
        qtx << "mma.async.init C, [r0], [r1]\nmma.commit [r4]\n"
            << "wait:\nmbar.try_wait p0, [r4], 0\n@!p0 bra.uni wait\n"
            << "mov r8, 0\nmov r9, 0\nacc.ld r10, C, r8\nhalt\n";
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.shmem_image, 1024);
        for (uint32_t i = 0; i < 128; ++i) write_u16(shmem, 2 * i, 0x3c00);       // A: 1.0
        for (uint32_t i = 0; i < 32; ++i) write_u16(shmem, 512 + 2 * i, 0x3c00);  // B: 1.0
        nanosm::MbarState event;
        event.valid = true;
        event.expected = 1;
        event.tx_bytes = 64;
        write_u64(shmem, event_mbar, nanosm::encode_mbar_state(event));
        nanosm::MbarState commit;
        commit.valid = true;
        commit.expected = 1;
        commit.remaining = 1;
        write_u64(shmem, 896, nanosm::encode_mbar_state(commit));
        nanosm::Machine mma(dense, compile(qtx.str(), dense), std::move(launch),
            nanosm::TraceOptions{.timeline = true, .debug = true});
        mma.complete_mbar_tx(event_mbar, 64, 0); // the RMW that pauses the sts
        std::optional<std::string> panic;
        try {
            mma.run(4000);
        } catch (const nanosm::SimError& error) {
            panic = error.what();
        }
#if NANOSM_SANITIZER
        {
            // The MMA's accept: its dispatch (I6a: its issue, pc = 8 * (9 + pad), +
            // issue_to_dispatch), or the PANIC it raised there.
            std::vector<I2Grant> stores;
            for (const auto& grant : i2_grants(mma, "A-write")) {
                if (grant.client == "simt") stores.push_back(grant);
            }
            const auto what = "pad " + std::to_string(pad) + ":" + describe(stores);
            const uint64_t accept = panic.has_value()
                ? std::stoull(panic->substr(panic->find("cycle=") + 6))
                : issued_at(mma, 8 * (9 + pad)).at(0) + nanosm::issue_to_dispatch(dense);
            const bool in_flight = stores.size() < 16 || stores.back().cycle + depth > accept;
            require(panic.has_value() == in_flight, what + ": accepted at c" + std::to_string(accept) +
                (panic.has_value() ? " with every store round at the RAM, yet: " + *panic
                                   : " with a store round still travelling, and no missing-fence PANIC"));
            if (panic.has_value()) {
                require_mentions(*panic, {"missing fence.sh: mma.async #1 A operand bytes",
                    "overlap this warp's earlier sts [r20], r21", "whose write has not reached the RAM"}, what);
                ++panicked;
                continue;
            }
            require(nanosm::bits_float(mma.read_reg(0, 0, 10)) == 16.0f, what + ": not every word was new");
            ++clean;
            continue;
        }
#endif
        require(!panic.has_value(), "pad " + std::to_string(pad) + ": " + panic.value_or(""));
        std::map<int64_t, uint64_t> step_start;
        for (const auto& event : mma.trace().events()) {
            if (event.name == "MMA input step") step_start[event.value] = event.start_cycle;
        }
        // I7: a port-B round granted at g samples the RAM at g + D (the read
        // pipeline has port A's depth); before I7 it read the RAM at g.
        const std::array<uint64_t, 4> sampled{step_start.at(0) + depth, step_start.at(0) + 1 + depth,
            step_start.at(4) + depth, step_start.at(4) + 1 + depth};
        std::vector<I2Grant> stores;
        for (const auto& grant : i2_grants(mma, "A-write")) {
            if (grant.client == "simt") stores.push_back(grant);
        }
        require(stores.size() == 16 && stores[0].cycle == paused_until + 1,
            "the sts was not sixteen rounds granted from w + 1:" + describe(stores));
        float expected = 0.0f;
        for (uint32_t word = 0; word < 4; ++word) {
            const auto landed = stores[word].cycle + depth; // its bytes land at this cycle's clock edge
            tie |= landed == sampled[word];
            expected += 2.0f * (landed < sampled[word] ? 2.0f : 1.0f); // two fp16 halves per word, B = 1.0
        }
        const auto row0 = nanosm::bits_float(mma.read_reg(0, 0, 10));
        require(row0 == expected, "pad " + std::to_string(pad) + ": row 0 = " + std::to_string(row0) +
            ", but the words the fetch sampled sum to " + std::to_string(expected));
        outcomes.insert(mma.read_reg(0, 0, 10));
    }
#if NANOSM_SANITIZER
    require(panicked != 0 && clean != 0, "the sweep did not cross the missing-fence boundary (" +
        std::to_string(panicked) + " PANICs, " + std::to_string(clean) + " clean)");
#else
    require(outcomes.size() >= 2 && tie && panicked == 0 && clean == 0,
        "the sweep did not cross the READ_FIRST boundary (a tie included)");
#endif
}

// ---------------------------------------------------------------------------
// I2.1: follow-ups to the port-A slice (trash/i2-1-followups-2026-10-11.md;
// Fable's review trash/i1-i2-review-fable-2026-10-11.md F1-F8, and the TMA
// store-register derivation in trash/chan.txt). Timelines are hand assembly
// under a debug trace, as in the I2 section.
// ---------------------------------------------------------------------------

// An I2-style machine (shmem words 0..2047 hold 0x100 + word) whose image
// also holds the given mbars.
std::unique_ptr<nanosm::Machine> i21_machine(const nanosm::DeviceProfile& profile, const std::string& program,
    const std::vector<std::pair<uint32_t, nanosm::MbarState>>& mbars, uint32_t warps = 1) {
    nanosm::Launch launch;
    launch.active_warps = warps;
    auto& shmem = image_bytes(launch.shmem_image, 8192);
    for (uint32_t word = 0; word < 2048; ++word) write_u32(shmem, 4 * word, 0x100 + word);
    for (const auto& [address, state] : mbars) write_u64(shmem, address, nanosm::encode_mbar_state(state));
    return std::make_unique<nanosm::Machine>(profile, hand_assemble(i1_asm(profile, program), profile),
        std::move(launch), nanosm::TraceOptions{.timeline = true, .debug = true});
}

nanosm::MbarState i21_mbar(uint16_t expected, uint16_t remaining, int32_t tx_bytes = 0, bool phase = false) {
    nanosm::MbarState state;
    state.valid = true;
    state.phase = phase;
    state.expected = expected;
    state.remaining = remaining;
    state.tx_bytes = tx_bytes;
    return state;
}

// An optional profile field, I2 style: the canonical JSON declares
// `canonical`; absent, it defaults to the same value (same fingerprint); 0,
// negative, fractional and 65 are rejected by parsing and 0 by validation;
// another value changes the fingerprint.
void i21_check_optional_field(const char* section, const char* field, uint32_t canonical,
    uint32_t nanosm::DeviceProfile::*member) {
    std::ifstream input("profiles/nanosm-v0.json");
    nlohmann::json document;
    input >> document;
    const auto parsed = nanosm::parse_device_profile(document.dump());
    const auto name = std::string(section) + "." + field;
    require(parsed.*member == canonical && document[section].contains(field),
        name + ": the canonical profile does not declare " + std::to_string(canonical));
    auto defaulted = document;
    defaulted[section].erase(field);
    require(nanosm::device_profile_fingerprint(nanosm::parse_device_profile(defaulted.dump())) ==
            nanosm::device_profile_fingerprint(parsed), name + ": the default is not " + std::to_string(canonical));
    for (const auto value : {nlohmann::json(0), nlohmann::json(-1), nlohmann::json(1.5), nlohmann::json(65)}) {
        auto modified = document;
        modified[section][field] = value;
        try {
            (void)nanosm::parse_device_profile(modified.dump());
            throw std::runtime_error(name + " = " + value.dump() + " was accepted");
        } catch (const nanosm::ProfileError&) {}
    }
    auto zero = canonical_profile();
    zero.*member = 0;
    try {
        nanosm::validate_device_profile(zero);
        throw std::runtime_error(name + " = 0 passed validation");
    } catch (const nanosm::ProfileError&) {}
    auto other = canonical_profile();
    other.*member = canonical + 1;
    require(nanosm::device_profile_fingerprint(other) != nanosm::device_profile_fingerprint(canonical_profile()),
        name + " was omitted from the profile fingerprint");
}

// F1 (rfc12 section 2: every latency is a nonzero number of register
// stages): the RMW engine's update crosses shmem.rmw_compute_cycles (C) ALU
// stages. A SIMT mbar.arrive issued at i reaches the SIMT RMW register at its
// dispatch (I6a: i + issue_to_dispatch) and has its read granted the cycle
// after, r; its words reach the port output at r + D + Lb; the update is in
// the ALU's last stage C cycles later and its write is granted then; the
// write reaches the RAM -- the phase flips at that clock edge -- D cycles
// after that. The engine is busy (its trace slice) D + Lb + C + 1 cycles.
// (Before I6a the arrive issued at c1 and was read at c2; now it waits for
// the mov's r4 -- src/rf.hpp -- and for its dispatch.)
void test_i21_rmw_compute_stage() {
    i21_check_optional_field("shmem", "rmw_compute_cycles", 2, &nanosm::DeviceProfile::shmem_rmw_compute_cycles);
    for (const uint32_t c : {1u, 2u, 3u}) {
        auto profile = canonical_profile();
        profile.shmem_rmw_compute_cycles = c;
        const uint64_t d = profile.shmem_port_a_depth;
        const uint64_t lb = profile.shmem_read_latency_cycles;
        auto machine = i21_machine(profile, R"(
            mov r4, 4160
            mbar.arrive [r4]
            halt
        )", {{4160, i21_mbar(1, 1)}});
        std::optional<uint64_t> flipped; // the cycle whose clock edge landed the new phase
        while (!machine->done()) {
            const auto cycle = machine->cycle();
            tick_capped(*machine);
            if (!flipped.has_value() && machine->read_mbar_state(4160).phase) flipped = cycle;
        }
        const auto what = "C = " + std::to_string(c) + ": ";
        const auto grants = i2_grants(*machine);
        const auto arrive = issued_at(*machine, 8);
        require(arrive.size() == 1, what + "the arrive did not issue once");
        const auto r = arrive[0] + nanosm::issue_to_dispatch(profile) + 1;
        require(grants == std::vector<I2Grant>{{r, "A-read", "rmw", 0}, {r + d + lb + c, "A-write", "rmw", 0}},
            what + "the read was not granted the cycle after the arrive's dispatch, or the write D + Lb + C after "
            "it:" + describe(grants));
        require(flipped == r + d + lb + c + d, what + "the update did not reach the RAM D cycles after its write grant");
        require(mbar_rmw_services(*machine).size() == 1 && rmw_busy_cycles(profile) == d + lb + c + 1,
            what + "one RMW");
        for (const auto& event : machine->trace().events()) {
            if (event.name == "mbar RMW") {
                require(event.duration_cycles == d + lb + c + 1, what + "the engine was not busy D + Lb + C + 1 cycles");
            }
        }
        const auto state = machine->read_mbar_state(4160);
        require(state.phase && state.remaining == 1, what + "the arrive's update was lost");
    }
}

// F2: admission of a SIMT RMW reads registered state and this cycle's grants
// only. The MMA presents its commit's arrive into its own event register at
// c3 (the commit reached the queue's head at the edge of c2); the warp's
// mbar.arrive, attempted at c3, is accepted at c3 (the event reaches the RMW
// engine, and admission, from c4) and is served first, at c4; the held MMA
// event is claimed when the engine frees, D + Lb + C + 1 cycles later.
// Before I2.1 issue also read the presentation of c3 and refused the arrive
// until the engine had served the commit (arrive at c4 + busy - 1).
// I3: the arrive enters the SIMT RMW register at c3 (it only needs that
// register free); at c4 it and the MMA's event are both held and the
// engine's class rule -- the class that did not start last; at the first
// tie, SIMT -- starts the arrive. Same cycles as in I2.1.
// I6a: the same scenario, moved: the commit issues at m (it needs the
// first mov's r4: alu_distance), reaches the MMA queue at its dispatch
// m + issue_to_dispatch and presents at P = m + issue_to_dispatch + 1. The
// arrive issues at m + 1 and reaches the SIMT RMW register at its own
// dispatch, P: the cycle the commit is presented, as before. Both are held
// at P + 1 and the arrive is served first. (I6b-2: the first mov issues at
// fetch_to_issue, so m is that + alu_distance.)
void test_i21_simt_rmw_admission_registered_state() {
    const auto& profile = canonical_profile();
    auto machine = i21_machine(profile, R"(
        mov r4, 4160
        mov r5, 4168
        mma.commit [r4]
        mbar.arrive [r5]
        halt
    )", {{4160, i21_mbar(1, 1)}, {4168, i21_mbar(2, 2)}});
    const auto m = nanosm::fetch_to_issue + alu_distance(profile);
    const auto p = m + nanosm::issue_to_dispatch(profile) + 1;
    i2_run_until(*machine, p + 1);
    require(machine->mma_mbar_event_held(), "the commit was not presented at P (held from P + 1)");
    machine->run();
    require(issued_at(*machine, 2 * 8) == std::vector<uint64_t>{m} &&
            issued_at(*machine, 3 * 8) == std::vector<uint64_t>{m + 1},
        "the commit did not issue at m or the arrive at m + 1 (dispatched in the cycle the commit was presented)");
    const auto busy = rmw_busy_cycles(profile);
    const auto services = mbar_rmw_services(*machine);
    require(services == std::vector<RmwService>{{p + 1, "arrive, address=4168"},
                            {p + 1 + busy, "arrive (mma.commit), address=4160"}},
        "the SIMT arrive was not served at P + 1 and the commit after it");
    require(machine->read_mbar_state(4160).phase && machine->read_mbar_state(4168).remaining == 1,
        "an arrive was lost");
}

// F3 (rfc12 section 6 / 13): a SIMT RMW waits for its own warp's lds and sts
// before it takes the engine. All lanes on one bank: the sts is sixteen
// rounds (c5..c20); the lds over the same words is ordered behind it
// (refused c5..c19) and admitted at its last grant (c20), granted
// c21..c36, and reads the stored words; the mbar.arrive is ordered behind
// the lds (refused c21..c35), admitted at the lds's last grant (c36), read at
// c37 and written D + Lb + C later. The same at canonical and at two
// noncanonical depths (D = 3, Lb = 1, C = 1; D = 1, Lb = 3, C = 3): the
// grants, the refusals and the RAW data do not move.
// I6a re-derivation, relative to the sts's issue s (it needs r5, written by
// the mov at s - alu_distance) and t = issue_to_dispatch: each access
// reaches its slot or register t after its issue and is served from the
// next cycle, so the sts's rounds are s + t + 1 .. s + t + 16; the lds,
// ordered behind it from s + 1 (refused t + 15 cycles), is admitted at its
// last grant and granted from s + 2t + 17; the arrive, ordered behind the
// lds (again t + 15 cycles), is admitted at the lds's last grant (s + 2t +
// 32) and read at s + 3t + 33. (I6b-2: the sread issues at fetch_to_issue.)
void test_i21_simt_rmw_behind_own_lds_and_sts() {
    for (const auto& [d, lb, c] : std::vector<std::tuple<uint32_t, uint32_t, uint32_t>>{{2, 2, 2}, {3, 1, 1}, {1, 3, 3}}) {
        auto profile = canonical_profile();
        profile.shmem_port_a_depth = d;
        profile.shmem_read_latency_cycles = lb;
        profile.shmem_rmw_compute_cycles = c;
        const auto what = "D=" + std::to_string(d) + " Lb=" + std::to_string(lb) + " C=" + std::to_string(c) + ": ";
        auto machine = i21_machine(profile, R"(
            sread r0, LANE_ID
            shli r0, r0, 6
            mov r4, 4160
            mov r5, 0x5a
            sts [r0 + 8], r5 {pred=PT, stall=0, set=sb0, wait=-}
            lds r6, [r0 + 8] {pred=PT, stall=0, set=sb1, wait=-}
            mbar.arrive [r4]
            halt {pred=PT, stall=0, set=-, wait=sb0|sb1}
        )", {{4160, i21_mbar(2, 2)}});
        machine->run();
        std::vector<uint64_t> store_rounds, load_rounds;
        uint64_t rmw_read = 0, rmw_write = 0;
        for (const auto& grant : i2_grants(*machine)) {
            if (grant.client == "simt") (grant.service == "A-write" ? store_rounds : load_rounds).push_back(grant.cycle);
            if (grant.client == "rmw") (grant.service == "A-read" ? rmw_read : rmw_write) = grant.cycle;
        }
        const auto consecutive = [](const std::vector<uint64_t>& cycles, uint64_t first) {
            if (cycles.size() != 16) return false;
            for (std::size_t round = 0; round < 16; ++round) {
                if (cycles[round] != first + round) return false;
            }
            return true;
        };
        const auto t = nanosm::issue_to_dispatch(profile);
        const auto s = nanosm::fetch_to_issue + 2 + 2 * alu_distance(profile); // sread, shli; mov r4, mov r5; the sts
        const auto issues = " | issues" + issue_log(*machine);
        require(issued_at(*machine, 4 * 8) == std::vector<uint64_t>{s}, what + "the sts did not issue at s:" + issues);
        require(consecutive(store_rounds, s + t + 1) && consecutive(load_rounds, s + 2 * t + 17),
            what + "sts rounds not from s + t + 1 or lds rounds not from s + 2t + 17:" + describe(i2_grants(*machine)) +
                issues);
        require(issued_at(*machine, 5 * 8) == std::vector<uint64_t>{s + t + 16} &&
                issued_at(*machine, 6 * 8) == std::vector<uint64_t>{s + 2 * t + 32},
            what + "the lds was not admitted at the sts's last grant, or the arrive at the lds's");
        const auto ordered = refusals(*machine, 0, "ordered behind own access");
        require(ordered == std::vector<std::pair<uint64_t, uint64_t>>{{s + 1, t + 15}, {s + t + 17, t + 15}},
            what + "ordered refusals:" + describe(ordered));
        require(rmw_read == s + 3 * t + 33 && rmw_write == rmw_read + d + lb + c,
            what + "the RMW was not read at s + 3t + 33 and written D + Lb + C later");
        for (uint32_t lane = 0; lane < 16; ++lane) require(machine->read_reg(0, lane, 6) == 0x5a, what + "RAW: the lds missed the sts");
        require(machine->read_mbar_state(4160).remaining == 1, what + "the arrive was lost");
    }
}

// F3: a probe waits for its own warp's sts, and sees it. The sts's sixteen
// lanes are on one bank (sixteen rounds, c7..c22); lane 15 -- the LAST round
// -- writes the mbar's low word with its phase flipped. mbar.test_wait is a
// read: ordered behind the sts (refused c7..c21), admitted at its last grant
// (c22), its probe granted c23, sampling D cycles later, after the last
// round reached the RAM (c22 + D): the predicate says "phase complete".
// I6a re-derivation, relative to the sts's issue s (it needs r6, written by
// the mov at s - alu_distance, after sread -> xori -> shli) and t =
// issue_to_dispatch: rounds s + t + 1 .. s + t + 16; the test_wait refused
// from s + 1 (t + 15 cycles), admitted at the last grant s + t + 16 and
// granted its probe at s + 2t + 17, after its dispatch. (I6b-2: the sread
// issues at fetch_to_issue.)
void test_i21_probe_behind_own_sts() {
    auto flipped = i21_mbar(1, 1);
    flipped.phase = true;
    const auto new_low = static_cast<uint32_t>(nanosm::encode_mbar_state(flipped));
    for (const auto& [d, lb] : std::vector<std::pair<uint32_t, uint32_t>>{{2, 2}, {3, 1}}) {
        auto profile = canonical_profile();
        profile.shmem_port_a_depth = d;
        profile.shmem_read_latency_cycles = lb;
        const auto what = "D=" + std::to_string(d) + " Lb=" + std::to_string(lb) + ": ";
        auto machine = i21_machine(profile, R"(
            sread r1, LANE_ID
            xori r2, r1, 15
            shli r2, r2, 6
            mov r6, )" + std::to_string(new_low) + R"(
            mov r4, 6144
            mov r9, 0
            sts [r2 + 6144], r6 {pred=PT, stall=0, set=sb0, wait=-}
            mbar.test_wait p0, [r4], 0
            bra.uni done {pred=!p0, stall=0, set=-, wait=-}
            mov r9, 1
        done: halt {pred=PT, stall=0, set=-, wait=sb0}
        )", {{6144, i21_mbar(1, 1)}});
        machine->run();
        const auto t = nanosm::issue_to_dispatch(profile);
        const auto s = nanosm::fetch_to_issue + 3 * alu_distance(profile) + 1; // sread, xori, shli; mov r6; the sts
        require(issued_at(*machine, 6 * 8) == std::vector<uint64_t>{s},
            what + "the sts did not issue at s:" + issue_log(*machine));
        require(issued_at(*machine, 7 * 8) == std::vector<uint64_t>{s + t + 16},
            what + "the probe was not admitted at the sts's last grant");
        require(refusals(*machine, 0, "ordered behind own access") ==
                std::vector<std::pair<uint64_t, uint64_t>>{{s + 1, t + 15}},
            what + "the probe was not ordered behind the sts (s + 1 .. s + t + 15)");
        const auto reads = i2_grants(*machine, "A-read");
        require(reads == std::vector<I2Grant>{{s + 2 * t + 17, "A-read", "probe", 0}},
            what + "probe grant:" + describe(reads));
        for (uint32_t lane = 0; lane < 16; ++lane) {
            require(machine->read_reg(0, lane, 9) == 1, what + "the probe did not see its own sts's last round");
        }
    }
}

// F3 / rfc12 section 9: a wake that arrives while a try_wait's probe is in
// flight ends the wait; the probe, when it returns, is stale and does
// nothing. Warp 0's try_wait installs its wait slot and its probe is granted
// the next cycle (I6a: both at its dispatch, issue_to_dispatch after its
// issue, so the probe's grant is issue + issue_to_dispatch + 1, and warp 1's
// try_wait takes its slot at ITS dispatch). I3 re-derivation: in I2 the host's event, placed then,
// held A until its write, so the probe was granted right after the write.
// Reads are not paused any more, so the event is placed D + Lb + C cycles
// EARLIER: the RMW's write is granted in the probe's grant cycle w. The
// probe samples at w + D, the cycle the write reaches the RAM -- READ_FIRST:
// the old phase -- and the write's commit wakes slot 0 in that cycle; the
// probe returns at w + D + Lb to a slot that is gone -- or, with a second
// warp, to the same slot index recycled by warp 1's try_wait on another
// mbar. The stale probe must not complete warp 1's wait: warp 1 times out
// (r10 = 0), warp 0 woke (r9 = 1), and only warp 1's probe is acted on.
void test_i21_stale_probe_after_wake() {
    const auto& profile = canonical_profile();
    const uint64_t d = profile.shmem_port_a_depth;
    const uint64_t lb = profile.shmem_read_latency_cycles;
    const uint64_t c = profile.shmem_rmw_compute_cycles;
    const uint64_t t = nanosm::issue_to_dispatch(profile);
    const auto program = [](uint32_t warp1_stall) {
        return R"(
            sread r29, WARP_ID
            setp.ne p1, r29, 0
            bra.uni w1 {pred=p1, stall=0, set=-, wait=-}
            mov r4, 4096
            mov r9, 0
            mbar.try_wait p0, [r4], 0
            bra.uni w0done {pred=!p0, stall=0, set=-, wait=-}
            mov r9, 1
        w0done: halt
        w1: mov r5, 4176
            mov r10, 0 {pred=PT, stall=)" + std::to_string(warp1_stall) + R"(, set=-, wait=-}
            mbar.try_wait p2, [r5], 0
            bra.uni w1done {pred=!p2, stall=0, set=-, wait=-}
            mov r10, 1
        w1done: halt
        )";
    };
    const std::vector<std::pair<uint32_t, nanosm::MbarState>> mbars{
        {i2_event_mbar, i21_mbar(1, 0, 64)}, {4176, i21_mbar(1, 1)}};
    constexpr uint32_t warp0_try_wait = 5 * 8;
    constexpr uint32_t warp1_try_wait = 11 * 8;
    // When warp 0's try_wait issues (independent of the host's event).
    const auto try_wait_issue = [&](uint32_t warps, uint32_t warp1_stall) {
        auto dry = i21_machine(profile, program(warp1_stall), mbars, warps);
        dry->run();
        return issued_at(*dry, warp0_try_wait).front();
    };
    struct Seen {
        uint64_t warp0_issue = 0;
        uint64_t warp1_issue = 0;
        std::vector<I2Grant> reads;
        std::vector<I2Grant> writes;
        nanosm::TraceSummary summary;
        uint32_t r9 = 0;
        uint32_t r10 = 0;
    };
    const auto run = [&](uint32_t warps, uint32_t warp1_stall) {
        const auto t0 = try_wait_issue(warps, warp1_stall);
        require(t0 + t + 1 >= d + lb + c, "warp 0's try_wait issues too early to place the event before it");
        auto machine = i21_machine(profile, program(warp1_stall), mbars, warps);
        // The probe is granted at t0 + t + 1; the RMW, read in its start
        // cycle, has its write granted D + Lb + C later: start it at
        // t0 + t + 1 - (D + Lb + C).
        i2_run_until(*machine, t0 + t + 1 - (d + lb + c));
        host_rmw_event(*machine);
        machine->run();
        Seen seen;
        seen.warp0_issue = issued_at(*machine, warp0_try_wait).front();
        if (warps == 2) {
            const auto issues = issued_at(*machine, warp1_try_wait);
            require(issues.size() == 1, "warp 1's try_wait did not issue once");
            seen.warp1_issue = issues.front();
        }
        seen.reads = i2_grants(*machine, "A-read");
        seen.writes = i2_grants(*machine, "A-write");
        seen.summary = machine->trace().summary();
        seen.r9 = machine->read_reg(0, 0, 9);
        if (warps == 2) seen.r10 = machine->read_reg(1, 0, 10);
        return seen;
    };
    // One warp: the slot is gone when the probe returns.
    const auto alone = run(1, 0);
    const auto w = alone.warp0_issue + t + 1; // the probe's grant and the RMW's write grant
    require(alone.reads.size() >= 2 && alone.reads[0] == I2Grant{w - (d + lb + c), "A-read", "rmw", 0} &&
            alone.reads[1] == I2Grant{w, "A-read", "probe", 0} &&
            alone.writes == std::vector<I2Grant>{{w, "A-write", "rmw", 0}},
        "the probe was not granted in the RMW's write-grant cycle:" + describe(alone.reads) + describe(alone.writes));
    require(alone.summary.mbar_wakes == 1 && alone.summary.mbar_initial_probes == 0 && alone.summary.mbar_sleeps == 0 &&
            alone.summary.mbar_timeouts == 0, "the stale probe was acted on (or the wake was lost)");
    require(alone.r9 == 1, "the woken try_wait did not return true");
    // Two warps (warp 1's stall count swept): warp 1's try_wait takes slot 0
    // between the wake (w + D) and the stale probe's return (w + D + Lb), with
    // w the probe's grant (the cycle after warp 0's try_wait's dispatch) and
    // the RMW's write grant.
    uint32_t recycled = 0;
    std::string schedules;
    for (uint32_t stall = 0; stall < 32; ++stall) {
        const auto seen = run(2, stall);
        const auto grant = seen.warp0_issue + t + 1;
        const auto wake = grant + d;
        const auto probe_return = grant + d + lb;
        schedules += " stall " + std::to_string(stall) + ": warp 0 c" + std::to_string(seen.warp0_issue) +
            ", warp 1 c" + std::to_string(seen.warp1_issue) + ";";
        const auto warp1_slot = seen.warp1_issue + t; // its dispatch takes the lowest free slot
        if (warp1_slot < wake || warp1_slot >= probe_return) continue;
        ++recycled;
        const auto what = "warp 1 try_wait at c" + std::to_string(seen.warp1_issue) + ": ";
        require(seen.r9 == 1, what + "warp 0's woken try_wait did not return true");
        require(seen.r10 == 0, what + "warp 0's stale probe completed warp 1's recycled wait slot");
        require(seen.summary.mbar_initial_probes == 1 && seen.summary.mbar_wakes == 1 &&
                seen.summary.mbar_sleeps == 1 && seen.summary.mbar_timeouts == 1,
            what + "only warp 1's probe should act (it parks, then times out)");
    }
    require(recycled != 0, "no warp-1 schedule recycled the slot while warp 0's probe travelled:" + schedules);
}

// F3 / rfc12 section 5: an RMW waiting for its read does not preempt an
// A-read transaction under way (select_a_read step 2 after step 1). A
// sixteen-round lds is granted c3..c18; the host's event is held from c5.
// I3: the RMW STARTS at c5 (its store pause begins; there is no store) and
// waits for its read; the read is granted at c19 and the write D + Lb + C
// later -- the same grants as in I2, where the event was claimed only at its
// read grant. Also at a noncanonical depth.
// I6a: the same scenario relative to the lds's first grant g (its issue,
// after sread -> shli, at 2 alu_distance, plus issue_to_dispatch + 1): the
// event is held from g + 2, the RMW reads at g + 16. (I6b-2: the sread
// issues at fetch_to_issue.)
void test_i21_rmw_read_behind_a_read_transaction() {
    for (const auto& [d, lb, c] : std::vector<std::tuple<uint32_t, uint32_t, uint32_t>>{{2, 2, 2}, {3, 1, 1}}) {
        auto profile = canonical_profile();
        profile.shmem_port_a_depth = d;
        profile.shmem_read_latency_cycles = lb;
        profile.shmem_rmw_compute_cycles = c;
        const auto what = "D=" + std::to_string(d) + " Lb=" + std::to_string(lb) + " C=" + std::to_string(c) + ": ";
        auto machine = i2_machine(profile, R"(
            sread r0, LANE_ID
            shli r0, r0, 6
            lds r6, [r0 + 12] {pred=PT, stall=0, set=sb0, wait=-}
            halt {pred=PT, stall=0, set=-, wait=sb0}
        )");
        const auto g = nanosm::fetch_to_issue + 2 * alu_distance(profile) + nanosm::issue_to_dispatch(profile) + 1;
        i2_run_until(*machine, g + 2);
        host_rmw_event(*machine);
        machine->run();
        std::vector<I2Grant> expected;
        for (uint64_t round = 0; round < 16; ++round) expected.push_back({g + round, "A-read", "simt", 0});
        expected.push_back({g + 16, "A-read", "rmw", 0});
        expected.push_back({g + 16 + d + lb + c, "A-write", "rmw", 0});
        const auto grants = i2_grants(*machine);
        require(grants == expected, what + "the RMW's read preempted the lds or waited too long:" + describe(grants));
        require(mbar_rmw_services(*machine).front().first == g + 2, what + "the RMW did not start at g + 2");
        for (uint32_t lane = 0; lane < 16; ++lane) {
            require(machine->read_reg(0, lane, 6) == 0x100 + 16 * lane + 3, what + "lds data");
        }
    }
}

// TMA store sector registers (tma.store_sector_registers, canonical 1).
// A rank-1 linear TMA store of `sectors` 32 B sectors (one A-read round
// each) under a fast bus: the cycles its source rounds' words returned
// (edge probe tma_store_round), in order, and the run's length. The warp
// halts instead of polling the mbar: a try_wait probe would share A-read's
// round-robin with the store reads (legal contention, not what is measured).
std::pair<std::vector<uint64_t>, uint64_t> i21_tma_store_returns(const nanosm::DeviceProfile& config, uint32_t sectors) {
    const uint32_t bytes = 32 * sectors;
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.shmem_image, bytes + 256);
    fill_pattern(shmem, 0, bytes / 4, 0);
    write_rank1_tma(shmem, bytes, true, 0, bytes / 4, 0, bytes + 128);
    image_bytes(launch.gmem_image, bytes);
    const auto program = "mov r0, 0\nmov r1, 0\nmov r2, 0\ntma.async [" + std::to_string(bytes) + "], r0\nhalt\n";
    nanosm::Machine machine(config, compile(program, config), std::move(launch));
    machine.enable_edge_probe();
    machine.run(200000);
    for (uint32_t i = 0; i < bytes / 4; ++i) require(machine.read_gmem_u32(4 * i) == pattern_word(i), "TMA store data");
    std::vector<uint64_t> returns;
    for (const auto& event : machine.edge_events()) {
        if (event.edge == LsuEdge::tma_store_round) returns.push_back(event.cycle);
    }
    require(returns.size() == sectors, "not one source round per sector");
    return {returns, machine.cycle()};
}

// With L = D + Lb and d = ceil(8 / pluck) (chan 2026-10-11, the derivation):
// a store-source read releases A-read at its (one) round's grant and its
// words carry their register's index back; a register turns over every
// L + d cycles (reserved at the read's grant, complete L + 1 later, drained
// in d cycles, reusable in its last drain cycle). So N registers and a fast
// bus give N sectors per max(L + d, N d) cycles in steady state:
// min(N x 32 / (L + d), 32 / d) bytes per cycle. Checked exactly, for every
// steady-state sector, at N = 1..3 and pluck 1, 2, 8; and the canonical
// cases by name (D = Lb = 2): N = 1, pluck 2 -> one sector every 8 cycles;
// N = 2, pluck 2 -> egress-bound, one every 4. With N = ceil((L + d) / d)
// registers a 16 KiB store is pluck-bound again (rfc09 3a's egress bound).
void test_i21_tma_store_sector_registers() {
    i21_check_optional_field("tma", "store_sector_registers", 1, &nanosm::DeviceProfile::tma_store_sector_registers);
    const auto& canonical = canonical_profile();
    const uint64_t l = canonical.shmem_port_a_depth + canonical.shmem_read_latency_cycles;
    for (const uint32_t pluck : {1u, 2u, 8u}) {
        const uint64_t d = (8 + pluck - 1) / pluck;
        for (const uint32_t n : {1u, 2u, 3u}) {
            auto config = fast_bus_profile(pluck);
            config.tma_store_sector_registers = n;
            const auto what = "N=" + std::to_string(n) + " pluck " + std::to_string(pluck) + ": ";
            const auto [returns, cycles] = i21_tma_store_returns(config, 64);
            (void)cycles;
            const auto period = std::max(l + d, n * d);
            for (std::size_t k = n; k + n < returns.size(); ++k) {
                require(returns[k + n] - returns[k] == period, what + "sector " + std::to_string(k) + " -> " +
                    std::to_string(k + n) + " took " + std::to_string(returns[k + n] - returns[k]) +
                    " cycles, not max(L + d, N d) = " + std::to_string(period));
            }
        }
    }
    require(l == 4 && canonical.lsu_pluck_words_per_cycle == 2, "the named cases are derived for D = Lb = 2, pluck 2");
    {
        auto one = fast_bus_profile(2);
        const auto returns = i21_tma_store_returns(one, 16).first;
        for (std::size_t k = 1; k < returns.size(); ++k) {
            require(returns[k] - returns[k - 1] == 8, "N=1 pluck 2: one register does not turn over every L + d = 8 cycles");
        }
        auto two = fast_bus_profile(2);
        two.tma_store_sector_registers = 2;
        const auto pair = i21_tma_store_returns(two, 16).first;
        for (std::size_t k = 3; k < pair.size(); ++k) {
            require(pair[k] - pair[k - 1] == 4, "N=2 pluck 2: not egress-bound (one sector every d = 4 cycles)");
        }
    }
    for (const uint32_t pluck : {1u, 2u, 8u}) {
        const uint64_t d = (8 + pluck - 1) / pluck;
        auto config = fast_bus_profile(pluck);
        config.lsu_egress_sectors = 2;
        config.tma_store_sector_registers = static_cast<uint32_t>((l + d + d - 1) / d);
        const auto cycles = i21_tma_store_returns(config, 512).second;
        const uint64_t egress_bound = 16384 / (4 * pluck);
        require(cycles >= egress_bound && cycles <= egress_bound + 100, "pluck " + std::to_string(pluck) + ", N = " +
            std::to_string(config.tma_store_sector_registers) + ": 16 KiB took " + std::to_string(cycles) +
            " cycles, not the egress bound " + std::to_string(egress_bound));
    }
}

// F6: an acc.ld's tmem read-ownership (sanitizer) ends at its FINAL TMEM READ
// -- the hardware truth: its values are in its pipeline -- while its
// destination registers stay not-ready until each beat's RF write. Here the
// acc.ld.v8's last beat (r23, bank 2) is denied the RF write port by four
// fadds booked on bank 2, so it is written a few cycles after its tmem read.
// An mma.async.init on the accumulator accepted after the last tmem read but
// before that RF write is legal (no PANIC; the acc.ld still returns the old
// values); one accepted before the last tmem read is a tmem race (PANIC).
// I6a: the MMA accepts the command (and the sanitizer judges it) at its
// dispatch, issue_to_dispatch after its issue.
void test_i21_acc_ld_read_ownership() {
    auto profile = canonical_profile();
    profile.timing.fadd = 16;
    const auto program = [](uint32_t gap) {
        std::string body = R"(
            .acc C, 0, 0, 64, 64
            sread r0, LANE_ID
            mov r8, 0
            mov r9, 0
            mov r1, 0
            mov r2, 1024
        )";
        body += "mov r23, 77\n"; // nonzero: the acc.ld's write of the old (zero) tmem value shows
        for (uint32_t i = 0; i < 6; ++i) body += "setp.eq p" + std::to_string(5 + i % 2) + ", r0, 0\n";
        body += R"(
            fadd r26, r10, r10
            fadd r29, r10, r10
            fadd r32, r10, r10
            fadd r35, r10, r10
            setp.eq p5, r0, 0
            setp.eq p6, r0, 0
            setp.eq p5, r0, 0
            acc.ld.v8 r16, C, r8 {pred=PT, stall=0, set=sb0, wait=-}
        )";
        // One filler per cycle: they rotate over six predicates (I6a: a setp
        // waits until the previous write of its predicate is in the file,
        // issue_to_predicate_visible after it, so two alternating predicates
        // would issue in pairs five cycles apart).
        for (uint32_t i = 0; i < gap; ++i) body += "setp.eq p" + std::to_string(1 + i % 6) + ", r0, 0\n";
        body += R"(
            mma.async.init C, [r1], [r2] {pred=PT, stall=0, set=sb1, wait=-}
            halt {pred=PT, stall=0, set=-, wait=sb0|sb1}
        )";
        return body;
    };
    const auto machine_for = [&](uint32_t gap) {
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.shmem_image, 4096);
        for (uint32_t i = 0; i < 1024; ++i) write_u16(shmem, 2 * i, 0x3c00); // A and B: 1.0
        return std::make_unique<nanosm::Machine>(profile, hand_assemble(i1_asm(profile, program(gap)), profile),
            std::move(launch), nanosm::TraceOptions{.timeline = true});
    };
    constexpr uint32_t acc_ld_pc = 19 * 8;
    // Reference: the MMA long after; the final tmem read and final RF write.
    auto reference = machine_for(24);
    const auto written = i1_run(*reference, {23});
    uint64_t final_read = 0;
    for (const auto& event : reference->trace().events()) {
        if (event.name == "acc.ld beat") final_read = std::max(final_read, event.start_cycle);
    }
    const auto final_write = written.at(23).back();
    require(issued_at(*reference, acc_ld_pc).size() == 1 && final_write >= final_read + 2,
        "the last beat's RF write is not delayed past its tmem read; the test proves nothing (read c" +
            std::to_string(final_read) + ", write c" + std::to_string(final_write) + ")");
    const auto acc_ld_issue = issued_at(*reference, acc_ld_pc).front();
    uint32_t legal_between = 0;
    uint32_t races = 0;
    const auto dispatch = nanosm::issue_to_dispatch(profile);
    for (uint32_t gap = 0; acc_ld_issue + 1 + gap + dispatch < final_write; ++gap) {
        auto machine = machine_for(gap);
        const auto mma_issue = acc_ld_issue + 1 + gap;
        const auto mma_accept = mma_issue + dispatch;
        const auto where = "mma.async.init issued c" + std::to_string(mma_issue) + ", accepted c" +
            std::to_string(mma_accept) + " (last tmem read c" + std::to_string(final_read) + ", last RF write c" +
            std::to_string(final_write) + "): ";
        if (mma_accept < final_read) {
            try {
                machine->run(5000);
                throw std::runtime_error(where + "accepted while the acc.ld still reads its region; issues" +
                    issue_log(*machine));
            } catch (const nanosm::SimError& error) {
                require_mentions(error.what(), {"tmem race: mma.async", "acc.ld #0", "until its final tmem read"}, where);
            }
            ++races;
            continue;
        }
        machine->run(5000); // no PANIC
        require(issued_at(*machine, (19 + 1 + gap) * 8) == std::vector<uint64_t>{mma_issue}, where + "issue cycle moved");
        for (uint32_t reg = 16; reg < 24; ++reg) {
            require(machine->read_reg(0, 0, reg) == 0, where + "the acc.ld returned the MMA's result, not the old values");
        }
        ++legal_between;
    }
    require(races != 0 && legal_between >= 2, "the sweep did not cover both sides of the last tmem read");
}


// ---------------------------------------------------------------------------
// I3: the RMW engine's global store pause, the RMW / store alternation bit,
// fair admission between SIMT RMWs and engine events, and the signed mbar tx
// count (trash/rfc12.md sections 5, 6 and 8;
// trash/i3-rmw-pause-signed-tx-2026-10-11.md). Hand assembly under a debug
// trace, as in the I2 sections. An RMW's START is its "mbar RMW" slice's
// start (the store pause begins at that cycle's edge); its read and write are
// the "rmw" grants on A-read and A-write.
// ---------------------------------------------------------------------------

struct I3Rmw {
    uint64_t start = 0;
    uint64_t read = 0;
    uint64_t write = 0;
    std::string detail;
};

std::string describe(const std::vector<I3Rmw>& rmws) {
    std::string text;
    for (const auto& rmw : rmws) {
        text += " {" + rmw.detail + ": start c" + std::to_string(rmw.start) + ", read c" + std::to_string(rmw.read) +
            ", write c" + std::to_string(rmw.write) + "}";
    }
    return text;
}

// Every RMW, in order (the engine runs one at a time): start, read, write.
std::vector<I3Rmw> i3_rmws(const nanosm::Machine& machine) {
    std::vector<I3Rmw> rmws;
    for (const auto& [start, detail] : mbar_rmw_services(machine)) rmws.push_back({start, 0, 0, detail});
    std::size_t reads = 0;
    std::size_t writes = 0;
    for (const auto& grant : i2_grants(machine)) {
        if (grant.client != "rmw") continue;
        auto& next = grant.service == "A-read" ? reads : writes;
        require(next < rmws.size(), "an RMW grant without an RMW slice");
        (grant.service == "A-read" ? rmws[next].read : rmws[next].write) = grant.cycle;
        ++next;
    }
    return rmws;
}

// The cycles store transactions started (their first round's grant).
std::vector<uint64_t> i3_store_starts(const nanosm::Machine& machine) {
    std::vector<uint64_t> starts;
    for (const auto& event : machine.trace().events()) {
        if (event.category == nanosm::TraceCategory::shmem && event.kind == nanosm::TraceEventKind::slice &&
            event.name == "sts") {
            starts.push_back(event.start_cycle);
        }
    }
    std::sort(starts.begin(), starts.end());
    return starts;
}

// rfc12 section 6, the pause sequence with a store transaction under way
// when the RMW starts; at D = 2 and D = 3. All lanes on one bank: sts X is
// sixteen rounds, granted c4..c19. The host's event is held from c8: the RMW
// starts there (the store pause begins at that edge; X, under way, keeps its
// rounds); its read is granted at c20, strictly after X's last round (matched
// D: X's last write lands at c19 + D, before the RMW samples at c20 + D); its
// write at w = c20 + D + Lb + C. The younger sts Y, admitted at X's last
// grant, waits in its slot and is granted at w + 1 -- one cycle after the
// RMW's final write GRANT, not after its commit -- and still reaches the RAM
// after the RMW's write: the mbar lands at the edge of w + D, Y's first round
// at w + 1 + D. No store round between the RMW's read and its write.
// I6a: the same sequence relative to X's first grant g (X issues at s =
// 2 alu_distance + 1 -- it needs the mov's r5 -- and reaches its slot
// issue_to_dispatch later, t): X g .. g + 15, the event held from g + 4,
// the RMW read at g + 16; Y admitted at g + 15 waits in its slot from
// g + 16 + t to w (store_paused w - (g + 15 + t)). (I6b-2: the sread
// issues at fetch_to_issue, so s moves by that.)
void test_i3_store_pause_sequence() {
    for (const uint32_t d : {2u, 3u}) {
        auto profile = canonical_profile();
        profile.shmem_port_a_depth = d;
        const uint64_t lb = profile.shmem_read_latency_cycles;
        const uint64_t c = profile.shmem_rmw_compute_cycles;
        const auto what = "D=" + std::to_string(d) + ": ";
        auto machine = i2_machine(profile, R"(
            sread r0, LANE_ID
            shli r0, r0, 6
            mov r5, 0x5a
            sts [r0 + 8], r5 {pred=PT, stall=0, set=sb0, wait=-}
            sts [r0 + 12], r5 {pred=PT, stall=0, set=sb1, wait=-}
            halt {pred=PT, stall=0, set=-, wait=sb0|sb1}
        )");
        const auto t = nanosm::issue_to_dispatch(profile);
        const auto g = nanosm::fetch_to_issue + 2 * alu_distance(profile) + 1 + t + 1;
        i2_run_until(*machine, g + 4);
        host_rmw_event(*machine);
        std::optional<uint64_t> mbar_landed; // the cycle at whose clock edge the RMW's write landed
        std::optional<uint64_t> y_landed;    // ... Y's round 0 (lane 0's word)
        while (!machine->done()) {
            const auto cycle = machine->cycle();
            tick_capped(*machine);
            if (!mbar_landed && machine->read_mbar_state(i2_event_mbar).phase) mbar_landed = cycle;
            if (!y_landed && machine->read_shmem_u32(12) == 0x5a) y_landed = cycle;
        }
        const uint64_t w = g + 16 + d + lb + c;
        const auto rmws = i3_rmws(*machine);
        require(rmws.size() == 1 && rmws[0].start == g + 4 && rmws[0].read == g + 16 && rmws[0].write == w,
            what + "the RMW did not start at g + 4, read at g + 16 (X's last round + 1), write at g + 16 + D + Lb + C:" +
                describe(rmws) + " | issues" + issue_log(*machine));
        std::vector<uint64_t> stores;
        for (const auto& grant : i2_grants(*machine, "A-write")) {
            if (grant.client == "simt") stores.push_back(grant.cycle);
        }
        std::vector<uint64_t> expected;
        for (uint64_t round = 0; round < 16; ++round) expected.push_back(g + round);     // X
        for (uint64_t round = 0; round < 16; ++round) expected.push_back(w + 1 + round); // Y
        require(stores == expected, what + "store rounds (X g..g + 15, Y from w + 1):" +
            describe(i2_grants(*machine, "A-write")));
        require(issued_at(*machine, 4 * 8) == std::vector<uint64_t>{g + 15}, what + "Y was not admitted at X's last grant");
        require(mbar_landed == w + d && y_landed == w + 1 + d,
            what + "the RMW's write did not land at w + D, or Y's first round at w + 1 + D");
        require(machine->trace().summary().shmem_store_paused == w - (g + 15 + t),
            what + "store_paused should count g + 16 + t .. w (Y waiting in its slot under the pause)");
        for (uint32_t lane = 0; lane < 16; ++lane) {
            require(machine->read_shmem_u32(64 * lane + 8) == 0x5a && machine->read_shmem_u32(64 * lane + 12) == 0x5a,
                what + "store data");
        }
    }
}

// rfc12 section 6: reads continue during an RMW. The host's event is held
// from c5 and read at c5 (nothing waits on A); its write is granted at c5 +
// D + Lb + C = c11. In between A-read serves warp 0's lds.uni (c6), the
// TMA's descriptor read (c7) and warp 0's probe (c8).
// I6a: the three requests issue at c5, c6, c7 as before but reach A-read
// issue_to_dispatch later, so the event is placed at the lds.uni's dispatch
// e = 5 + issue_to_dispatch: the RMW reads at e, the lds.uni, the TMA and
// the probe are granted at e + 1, e + 2, e + 3, before its write at
// e + D + Lb + C. (I6b-2: the program starts at fetch_to_issue, so the
// lds.uni issues at fetch_to_issue + 5.)
void test_i3_reads_during_an_rmw() {
    const auto& profile = canonical_profile();
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.shmem_image, 8192);
    for (uint32_t word = 0; word < 512; ++word) write_u32(shmem, 4 * word, 0x100 + word);
    write_u64(shmem, i2_event_mbar, nanosm::encode_mbar_state(i21_mbar(1, 0, 64)));
    write_rank1_load(shmem, 5120, 0, 16, 2048, 4104);
    fill_pattern(image_bytes(launch.gmem_image, 64), 0, 16, 0);
    nanosm::Machine machine(profile, hand_assemble(i1_asm(profile, R"(
        mov r12, 4104
        mov r24, 0
        mov r25, 0
        mov r26, 0
        mov r27, 0
        lds.uni r3, [r1 + 64] {pred=PT, stall=0, set=sb0, wait=-}
        tma.async [5120], r8
        mbar.test_wait p0, [r12], 0
        halt {pred=PT, stall=0, set=-, wait=sb0}
    )"), profile), std::move(launch), nanosm::TraceOptions{.timeline = true, .debug = true});
    const uint64_t e = nanosm::fetch_to_issue + 5 + nanosm::issue_to_dispatch(profile);
    i2_run_until(machine, e);
    host_rmw_event(machine);
    machine.run();
    const uint64_t w = e + i2_return(profile) + profile.shmem_rmw_compute_cycles;
    const auto rmws = i3_rmws(machine);
    require(rmws.size() == 2 && rmws[0].start == e && rmws[0].read == e && rmws[0].write == w,
        "the host event's RMW:" + describe(rmws));
    auto reads = i2_grants(machine, "A-read");
    reads.resize(std::min<std::size_t>(reads.size(), 4));
    require(reads == std::vector<I2Grant>{{e, "A-read", "rmw", 0}, {e + 1, "A-read", "simt", 0},
                         {e + 2, "A-read", "tma", 0}, {e + 3, "A-read", "probe", 0}} && reads.back().cycle < w,
        "A-read did not serve an lds, the TMA and a probe between the RMW's read and its write:" + describe(reads));
    for (uint32_t lane = 0; lane < 16; ++lane) require(machine.read_reg(0, lane, 3) == 0x100 + 16, "lds.uni data");
    for (uint32_t i = 0; i < 16; ++i) require(machine.read_shmem_u32(2048 + 4 * i) == pattern_word(i), "TMA data");
}

// rfc12 sections 1 and 6: a read of the mbar that samples in the cycle the
// RMW's write reaches the RAM sees the OLD word (READ_FIRST). The host's
// event is read at c0; its write is granted at w = D + Lb + C and lands at
// w + D, completing the phase. Warp 0's test_wait, placed by its stall count,
// is granted at g and samples at g + D: it sees the new phase iff g > w --
// granted at w itself (the write-grant cycle) it reads the old phase.
// I6a: the test_wait needs the first mov's r4 (alu_distance) and reaches
// the probe register at its dispatch, so its probe is granted at
// alu_distance + issue_to_dispatch + 1 at the earliest (stalls 0..5), one
// cycle later per stall beyond. The host's event is placed (read at
// k = w - D - Lb - C) so that its write grant w is one cycle after that
// earliest grant: stalls 0..8 still cross w. (I6b-2: the first mov issues at
// fetch_to_issue.)
void test_i3_read_first_at_rmw_write() {
    const auto& profile = canonical_profile();
    const uint64_t rmw = i2_return(profile) + profile.shmem_rmw_compute_cycles;
    const uint64_t w = nanosm::fetch_to_issue + alu_distance(profile) + nanosm::issue_to_dispatch(profile) + 2;
    const uint64_t k = w - rmw;
    std::set<int> sides;
    for (uint32_t stall = 0; stall <= 8; ++stall) {
        auto machine = i2_machine(profile, R"(
            mov r4, 4096 {pred=PT, stall=)" + std::to_string(stall) + R"(, set=-, wait=-}
            mov r9, 0
            mbar.test_wait p0, [r4], 0
            bra.uni done {pred=!p0, stall=0, set=-, wait=-}
            mov r9, 1
        done: halt
        )");
        i2_run_until(*machine, k);
        host_rmw_event(*machine);
        machine->run();
        uint64_t probe = 0;
        for (const auto& grant : i2_grants(*machine, "A-read")) {
            if (grant.client == "probe") probe = grant.cycle;
        }
        const auto rmws = i3_rmws(*machine);
        const auto what = "probe granted c" + std::to_string(probe) + "," + describe(rmws) + ": ";
        require(probe != 0 && rmws.size() == 1 && rmws[0].read == k && rmws[0].write == w, what + "timeline");
        require(machine->read_reg(0, 0, 9) == (probe > w ? 1u : 0u),
            what + (probe > w ? "the probe missed the RMW's write" : "READ_FIRST: the probe did not read the old phase"));
        sides.insert(probe < w ? -1 : probe == w ? 0 : 1);
    }
    require(sides == std::set<int>{-1, 0, 1}, "the sweep did not cross the RMW's write-grant cycle");
}

// rfc12 section 5: the alternation bit between RMW starts and store
// transactions. Warp 0 issues eight mbar.arrive back to back (each waits at
// issue for the previous one's final write grant w, so the next RMW waits to
// start from w + 1); warp 1 issues eight one-round sts on the same words
// (each waits for the previous one's grant). Whenever both wait, the one
// that did not start last goes: the waiting store starts at w + 1 and the
// next RMW at w + 2 (its read at once: the store's one round was granted at
// w + 1). So the starts go R, S, R, S, ... -- back-to-back RMWs do not starve
// the stores -- and no store round is granted between an RMW's read and its
// write.
// I6a re-derivation: an arrive now reaches the SIMT RMW register
// issue_to_dispatch after its issue, so one warp's next arrive (issued at
// the previous one's write grant w) is there only at w + issue_to_dispatch
// + 1 -- after the waiting store already started alone at w + 1: the two
// never waited together and the alternation went unexercised. The arrives
// now come from two warps (0 and 2, four each): while one's RMW runs, the
// other's waits in the register, so at every w + 1 both classes wait, as in
// the one-warp scenario before I6a. Warp 1 issues the eight stores. The
// arrive warps' mov r4 stalls 10 cycles (its result needs alu_distance, 6)
// so that warp 1's first store reaches its slot while the first RMW runs:
// it waits under that RMW's pause, as every later one does.
void test_i3_rmw_store_alternation() {
    const auto& profile = canonical_profile();
    std::string arrives;
    std::string stores;
    for (int i = 0; i < 8; ++i) {
        if (i < 4) arrives += "mbar.arrive [r4]\n";
        stores += "sts [r20 + 2048], r21 {pred=PT, stall=0, set=sb0, wait=-}\n";
    }
    auto machine = i21_machine(profile, R"(
        sread r29, WARP_ID
        setp.eq p6, r29, 1
        bra.uni stores {pred=p6, stall=0, set=-, wait=-}
        mov r4, 4160 {pred=PT, stall=10, set=-, wait=-}
    )" + arrives + R"(
        halt
    stores: sread r20, LANE_ID
        shli r20, r20, 2
        mov r21, 0x33
    )" + stores + R"(
        halt {pred=PT, stall=0, set=-, wait=sb0}
    )", {{4160, i21_mbar(8, 8)}}, 3);
    machine->run();
    const auto rmws = i3_rmws(*machine);
    const auto store_starts = i3_store_starts(*machine);
    std::vector<std::pair<uint64_t, char>> starts;
    for (const auto& rmw : rmws) starts.emplace_back(rmw.start, 'R');
    for (const auto start : store_starts) starts.emplace_back(start, 'S');
    std::sort(starts.begin(), starts.end());
    std::string order;
    for (const auto& [cycle, kind] : starts) order += kind;
    require(order == "RSRSRSRSRSRSRSRS", "RMW (R) and store (S) starts did not alternate: " + order + describe(rmws));
    for (std::size_t k = 0; k < 8; ++k) {
        require(store_starts[k] == rmws[k].write + 1, "store " + std::to_string(k) +
            " did not start the cycle after RMW " + std::to_string(k) + "'s write grant:" + describe(rmws));
        if (k + 1 < 8) {
            require(rmws[k + 1].start == rmws[k].write + 2 && rmws[k + 1].read == rmws[k + 1].start,
                "RMW " + std::to_string(k + 1) + " did not start (and read) at w + 2:" + describe(rmws));
        }
    }
    for (const auto& grant : i2_grants(*machine, "A-write")) {
        for (const auto& rmw : rmws) {
            require(grant.client != "simt" || grant.cycle < rmw.read || grant.cycle > rmw.write,
                "a store round was granted between an RMW's read and its write, c" + std::to_string(grant.cycle) +
                    ":" + describe(rmws));
        }
    }
    require(machine->read_mbar_state(4160).phase, "the eight arrives did not complete the phase");
    for (uint32_t lane = 0; lane < 16; ++lane) require(machine->read_shmem_u32(2048 + 4 * lane) == 0x33, "store data");
}

// rfc12 section 6: a SIMT RMW takes the engine only when its warp has no
// unserviced lds / sts -- it waits for its own stores at issue, BEFORE the
// SIMT RMW register and the engine, so its store pause can never hold them
// (no deadlock). The host's event starts an RMW at c0 (read c0, write c6):
// its pause holds warp 0's sixteen-round sts, admitted at c4, until c7. The
// warp's mbar.arrive is ordered behind that sts (refused c5..c21), admitted
// at its last grant (c22), starts the next cycle with nothing to wait for
// (read c23) and is written D + Lb + C later.
// I6a: the sts issues at s (it needs the mov's r5: 2 alu_distance + 2) and
// reaches its slot at s + issue_to_dispatch (t). The host's event is placed
// as the sts issues (held, started and read at k = s + 1; written at w =
// k + D + Lb + C), so its pause holds the sts in its slot until w; the sts
// is granted w + 1 .. w + 16. The arrive is ordered behind it from s + 1,
// admitted at its last grant (w + 16), reaches the register at its dispatch
// and starts (reads) the cycle after, w + t + 17. (I6b-2: the sread issues
// at fetch_to_issue.)
void test_i3_simt_rmw_waits_for_own_stores() {
    const auto& profile = canonical_profile();
    const uint64_t rmw = i2_return(profile) + profile.shmem_rmw_compute_cycles; // read to write
    const uint64_t t = nanosm::issue_to_dispatch(profile);
    auto machine = i21_machine(profile, R"(
        sread r0, LANE_ID
        shli r0, r0, 6
        mov r4, 4160
        mov r5, 0x5a
        sts [r0 + 8], r5 {pred=PT, stall=0, set=sb0, wait=-}
        mbar.arrive [r4]
        halt {pred=PT, stall=0, set=-, wait=sb0}
    )", {{i2_event_mbar, i21_mbar(1, 0, 64)}, {4160, i21_mbar(2, 2)}});
    const uint64_t s = nanosm::fetch_to_issue + 2 * alu_distance(profile) + 2;
    const uint64_t k = s + 1;
    const uint64_t w = k + rmw;
    require(s + t < w, "the sts would not reach its slot under the pause");
    i2_run_until(*machine, k);
    host_rmw_event(*machine);
    machine->run();
    const auto rmws = i3_rmws(*machine);
    require(issued_at(*machine, 4 * 8) == std::vector<uint64_t>{s}, "the sts did not issue at s:" + issue_log(*machine));
    require(rmws.size() == 2 && rmws[0].start == k && rmws[0].read == k && rmws[0].write == w,
        "the host event's RMW:" + describe(rmws));
    std::vector<uint64_t> stores;
    for (const auto& grant : i2_grants(*machine, "A-write")) {
        if (grant.client == "simt") stores.push_back(grant.cycle);
    }
    std::vector<uint64_t> expected;
    for (uint64_t round = 0; round < 16; ++round) expected.push_back(w + 1 + round);
    require(stores == expected, "the sts did not resume at the event's write grant + 1:" + describe(i2_grants(*machine)));
    require(issued_at(*machine, 5 * 8) == std::vector<uint64_t>{w + 16} &&
            refusals(*machine, 0, "ordered behind own access") ==
                std::vector<std::pair<uint64_t, uint64_t>>{{s + 1, w + 16 - (s + 1)}},
        "the arrive was not ordered behind its own sts until the sts's last grant (s + 1 .. w + 15, admitted w + 16)");
    require(rmws[1].detail == "arrive, address=4160" && rmws[1].start == w + t + 17 && rmws[1].read == w + t + 17 &&
            rmws[1].write == w + t + 17 + rmw, "the arrive's RMW did not start and read at w + t + 17:" + describe(rmws));
    require(machine->read_mbar_state(4160).remaining == 1, "the arrive was lost");
}

// The RMW engine's admission (implementer choice, I3): the SIMT RMW register
// and the engines' event registers alternate by class. The host keeps the
// TMA's event register full (zero-byte complete_tx events on 4168, which
// change nothing) while warp 0 issues four mbar.arrive on 4160; each waits in
// the SIMT RMW register while the engine works. Whenever both classes hold a
// request, the class that did not start last goes, so the starts alternate
// TMA, SIMT, TMA, SIMT, ... one busy period apart: an endless supply of
// engine events cannot starve the warp. (Before I3 a held engine event
// refused a SIMT RMW at issue: engine events always went first.)
// I6a re-derivation: an arrive reaches the register issue_to_dispatch after
// its issue, and one warp issues its next arrive only at the previous one's
// write grant, so one warp's arrive would miss every other engine start
// (the TMA event then went alone, untested). Two warps issue two arrives
// each: while one's arrive runs, the other's waits in the register. Warp
// 0's first arrive issues at fetch_to_issue + alu_distance (its mov issues
// first, at fetch_to_issue; the arrive needs its r4) and is in the register
// from its dispatch + 1, a = that + issue_to_dispatch + 1; the host starts
// feeding one busy period before (f = a - busy), so the first TMA event
// ends as that arrive is held, and from then on both classes always hold a
// request.
void test_i3_rmw_admission_fairness() {
    const auto& profile = canonical_profile();
    const auto busy = rmw_busy_cycles(profile);
    auto machine = i21_machine(profile, R"(
        mov r4, 4160
        mbar.arrive [r4]
        mbar.arrive [r4]
        halt
    )", {{4160, i21_mbar(5, 5)}, {4168, i21_mbar(1, 1)}}, 2);
    const uint64_t f =
        nanosm::fetch_to_issue + alu_distance(profile) + nanosm::issue_to_dispatch(profile) + 1 - busy;
    while (!machine->done()) {
        const auto cycle = machine->cycle();
        if (cycle >= f && cycle < f + 9 * busy && !machine->tma_mbar_event_held()) {
            machine->complete_mbar_tx(4168, 0, 0);
        }
        tick_capped(*machine);
    }
    const auto services = mbar_rmw_services(*machine);
    std::vector<RmwService> expected;
    for (uint64_t k = 0; k < 9; ++k) {
        expected.emplace_back(f + k * busy, k % 2 == 0 ? "complete_tx (tma), address=4168" : "arrive, address=4160");
    }
    std::string seen;
    for (const auto& [cycle, detail] : services) seen += " c" + std::to_string(cycle) + ": " + detail + ";";
    require(services.size() >= 9 && std::vector<RmwService>(services.begin(), services.begin() + 9) == expected,
        "SIMT RMWs and engine events did not alternate:" + seen);
    const auto state = machine->read_mbar_state(4160);
    require(state.remaining == 1 && !state.phase, "an arrive was lost");
}

// rfc12 section 8: the signed tx count, in both orders ((pending arrivals,
// tx); the host's complete_tx and warp 0's arrive.expect_tx):
//   completion first:  (1, 0) -complete_tx 128-> (1, -128) -arrive.expect_tx 128-> (0, 0): complete
//   expectation first: (1, 0) -arrive.expect_tx 128-> (0, 128) -complete_tx 128-> (0, 0): complete
// At c18 the mbar holds the intermediate count; at the end the phase has
// flipped (arrivals reloaded, tx 0) and the warp's try_wait returned true.
// I6a: the observation moves to c20. Expectation first, the arrive issues at
// alu_distance (6: it needs the mov's r4), starts at its dispatch + 1 (10),
// is written D + Lb + C later (16) and lands D after that, at the edge of
// c18: seen from c19. Completion first, the arrive (behind the stall of 20)
// issues at c20, so nothing of it has landed by then.
// I6b-2: the program starts at fetch_to_issue (c2), so all of it -- and the
// observation -- moves by that: c22.
void test_i3_signed_tx_both_orders() {
    const auto& profile = canonical_profile();
    for (const bool completion_first : {true, false}) {
        const auto what = std::string(completion_first ? "completion first: " : "expectation first: ");
        auto machine = i21_machine(profile, std::string(completion_first ? R"(
            mov r4, 4160 {pred=PT, stall=20, set=-, wait=-}
            mbar.arrive.expect_tx [r4], 128
        )" : R"(
            mov r4, 4160
            mbar.arrive.expect_tx [r4], 128
            mov r9, 0 {pred=PT, stall=20, set=-, wait=-}
        )") + R"(
            mov r9, 0
        wait: mbar.try_wait p0, [r4], 0
            bra.uni wait {pred=!p0, stall=0, set=-, wait=-}
            mov r9, 1
            halt
        )", {{4160, i21_mbar(1, 1)}});
        if (completion_first) machine->complete_mbar_tx(4160, 128, 0);
        i2_run_until(*machine, nanosm::fetch_to_issue + 20);
        const auto middle = machine->read_mbar_state(4160);
        require(!middle.phase && middle.remaining == (completion_first ? 1 : 0) &&
                middle.tx_bytes == (completion_first ? -128 : 128),
            what + "intermediate (arrivals, tx) = (" + std::to_string(middle.remaining) + ", " +
                std::to_string(middle.tx_bytes) + ")");
        if (!completion_first) machine->complete_mbar_tx(4160, 128, 0);
        machine->run();
        const auto end = machine->read_mbar_state(4160);
        require(end.phase && end.remaining == 1 && end.tx_bytes == 0 && machine->read_reg(0, 0, 9) == 1,
            what + "the phase did not complete at (0, 0)");
    }
}

// rfc12 section 8: the TMA reads no mbar at its start (I2.1: after the
// descriptor, a one-round A-read of the completion mbar, which also checked
// that enough bytes were expected). A 64-byte TMA load completes into an mbar
// that expects one arrival and NO bytes yet -- the old check would have
// refused it -- and its complete_tx runs ahead: (1, 0) -> (1, -64). The
// warp's arrive.expect_tx of 64, long after, completes the phase. A-read
// serves the TMA exactly one round (its descriptor's), and shmem counts three
// requests (the descriptor read and two sector placements), one fewer than
// before.
void test_i3_tma_start_reads_no_mbar() {
    const auto& profile = canonical_profile();
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.shmem_image, 8192);
    write_rank1_load(shmem, 5120, 0, 16, 2048, 4104);
    write_u64(shmem, 4104, nanosm::encode_mbar_state(i21_mbar(1, 1)));
    fill_pattern(image_bytes(launch.gmem_image, 64), 0, 16, 0);
    std::string program = "mov r12, 4104\ntma.async [5120], r8\n";
    for (int i = 0; i < 10; ++i) program += "mov r9, 0 {pred=PT, stall=31, set=-, wait=-}\n";
    program += R"(
        mbar.arrive.expect_tx [r12], 64
    wait: mbar.try_wait p0, [r12], 0
        bra.uni wait {pred=!p0, stall=0, set=-, wait=-}
        mov r9, 1
        halt
    )";
    nanosm::Machine machine(profile, hand_assemble(i1_asm(profile, program), profile), std::move(launch),
        nanosm::TraceOptions{.timeline = true, .debug = true});
    std::vector<int32_t> counts{0}; // every tx count the mbar held, in order
    while (!machine.done()) {
        tick_capped(machine);
        const auto tx = machine.read_mbar_state(4104).tx_bytes;
        if (tx != counts.back()) counts.push_back(tx);
    }
    require(counts == std::vector<int32_t>{0, -64, 0} && machine.read_mbar_state(4104).phase &&
            machine.read_reg(0, 0, 9) == 1, "the TMA's complete_tx did not run ahead of the expect_tx (tx 0, -64, 0)");
    const auto& summary = machine.trace().summary();
    require(summary.shmem_a_read_tma == 1 && summary.shmem_requests == 3,
        "the TMA read more than its descriptor on A-read (a_read_tma=" + std::to_string(summary.shmem_a_read_tma) +
            ", requests=" + std::to_string(summary.shmem_requests) + ")");
    for (uint32_t i = 0; i < 16; ++i) require(machine.read_shmem_u32(2048 + 4 * i) == pattern_word(i), "TMA data");
}

// rfc12 section 8: tx arithmetic is checked (an atomic add wraps; this does
// not). Machine PANICs, both sanitizer builds: an operand above 2^20 - 1; a
// count rising above 2^20 - 1 (two arrive.expect_tx of 2^19); a count
// falling below -(2^20 - 1) (two host complete_tx of 2^20 - 1: the first
// reaches the limit exactly, which is legal; the second leaves it).
void test_i3_tx_range_panics() {
    const auto& profile = canonical_profile();
    {
        auto machine = i21_machine(profile, R"(
            mov r4, 4160
            mov r2, 524288
            mbar.arrive.expect_tx [r4], r2
            mbar.arrive.expect_tx [r4], r2
            halt
        )", {{4160, i21_mbar(2, 2)}});
        expect_sim_panic(*machine, "transaction byte count overflow", "a tx count above 2^20 - 1");
    }
    {
        auto machine = i21_machine(profile, R"(
            mov r4, 4160
            mov r2, 1048576
            mbar.arrive.expect_tx [r4], r2
            halt
        )", {{4160, i21_mbar(2, 2)}});
        expect_sim_panic(*machine, "transaction byte count operand exceeds 2^20 - 1", "a tx operand of 2^20");
    }
    {
        // The warp stays alive past the first event: kernel end would PANIC on tx != 0.
        auto machine = i21_machine(profile, "mov r9, 0 {pred=PT, stall=31, set=-, wait=-}\nhalt\n",
            {{4160, i21_mbar(1, 1)}});
        machine->complete_mbar_tx(4160, 1048575, 0);
        while (machine->read_mbar_state(4160).tx_bytes == 0) tick_capped(*machine);
        require(machine->read_mbar_state(4160).tx_bytes == -1048575, "the count did not reach -(2^20 - 1)");
        while (machine->tma_mbar_event_held()) tick_capped(*machine);
        machine->complete_mbar_tx(4160, 1048575, 0);
        expect_sim_panic(*machine, "transaction byte count underflow", "a tx count below -(2^20 - 1)");
    }
}

// SANITIZER (rfc12 section 8): the TMA's start-time mbar read is gone; the
// sanitizer peeks the completion mbar's phase when tma.async is accepted and
// compares it when the complete_tx reaches the mbar ("wrong phase"). Two ways
// a TMA's bytes land in a later phase:
//   plain arrive: the mbar expects one arrival and no bytes; the TMA is
//   accepted in phase 0, then a PLAIN mbar.arrive completes phase 0 at once
//   (tx still 0) -- rfc12's unsafe case -- and the TMA's complete_tx reaches
//   phase 1;
//   reuse: one arrive.expect_tx of 64 bytes but two 64-byte TMAs, both
//   accepted in phase 0; the first completes the phase, the second's
//   complete_tx reaches phase 1.
void test_i3_sanitizer_wrong_phase_complete_tx() {
    const auto& profile = canonical_profile();
    const auto machine_for = [&](const std::string& program) {
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.shmem_image, 8192);
        write_rank1_load(shmem, 5120, 0, 16, 2048, 4104);
        write_rank1_load(shmem, 5184, 0, 16, 2112, 4104);
        write_u64(shmem, 4104, nanosm::encode_mbar_state(i21_mbar(1, 1)));
        fill_pattern(image_bytes(launch.gmem_image, 64), 0, 16, 0);
        return std::make_unique<nanosm::Machine>(profile, hand_assemble(i1_asm(profile, program), profile),
            std::move(launch));
    };
    auto plain = machine_for(R"(
        mov r12, 4104
        tma.async [5120], r8
        mbar.arrive [r12]
        halt
    )");
    expect_sim_panic(*plain, "wrong phase (tma #0 was accepted in phase 0; the mbar is in phase 1)",
        "a TMA completing after a plain arrive ended its phase");
    auto reuse = machine_for(R"(
        mov r12, 4104
        mbar.arrive.expect_tx [r12], 64
        tma.async [5120], r8
        tma.async [5184], r8
        halt
    )");
    expect_sim_panic(*reuse, "wrong phase (tma #1 was accepted in phase 0; the mbar is in phase 1)",
        "a second TMA completing into the reused mbar's next phase");
}

// SANITIZER (rfc12 section 8): a tx count nonzero at kernel end -- a missing
// complete_tx (tx 64 left) or a missing expect_tx (tx -32 left) -- would hang
// a waiter: PANIC at the end of the run.
void test_i3_sanitizer_nonzero_tx_at_kernel_end() {
    const auto& profile = canonical_profile();
    {
        auto machine = i21_machine(profile, "mov r4, 4160\nmbar.arrive.expect_tx [r4], 64\nhalt\n", {{4160, i21_mbar(2, 2)}});
        expect_sim_panic(*machine, "transaction count is nonzero at kernel end (tx=64", "a missing complete_tx");
    }
    {
        auto machine = i21_machine(profile, "halt\n", {{4160, i21_mbar(1, 1)}});
        machine->complete_mbar_tx(4160, 32, 0);
        expect_sim_panic(*machine, "transaction count is nonzero at kernel end (tx=-32", "a missing expect_tx");
    }
}

// ---------------------------------------------------------------------------
// I4: fence.sh, the missing-fence / race sanitizer and C10 live-mbar opacity
// (trash/rfc12.md sections 10 and 11; trash/i4-fence-sanitizer-c10-2026-10-11.md).
// Timelines are hand assembly under a debug trace, as in the I2 section; the
// sanitizer cases are qtxc where the exact cycles do not matter.
// ---------------------------------------------------------------------------

// An I2-style machine (shmem words 0..2047 hold 0x100 + word; gmem words
// 0..1023 hold 5 * word + 2) whose shmem image `setup` adds to.
std::unique_ptr<nanosm::Machine> i4_machine(const nanosm::DeviceProfile& profile, const std::string& program,
    const std::function<void(std::vector<uint8_t>&)>& setup = {}, uint32_t warps = 1, bool hand_assembly = true) {
    nanosm::Launch launch;
    launch.active_warps = warps;
    auto& shmem = image_bytes(launch.shmem_image, 8192);
    for (uint32_t word = 0; word < 2048; ++word) write_u32(shmem, 4 * word, 0x100 + word);
    if (setup) setup(shmem);
    auto& gmem = image_bytes(launch.gmem_image, 4096);
    for (uint32_t word = 0; word < 1024; ++word) write_u32(gmem, 4 * word, 5 * word + 2);
    const auto kernel = hand_assembly ? hand_assemble(i1_asm(profile, program), profile)
                                      : compile(program, profile);
    return std::make_unique<nanosm::Machine>(profile, kernel, std::move(launch),
        nanosm::TraceOptions{.timeline = true, .debug = true});
}

// A rank-1 TMA descriptor at 5120 moving `bytes` between shmem `shmem_base`
// and gmem 0, completing into the mbar at 4104 (which expects those bytes).
std::function<void(std::vector<uint8_t>&)> i4_tma(bool store, uint32_t shmem_base, uint32_t bytes) {
    return [=](std::vector<uint8_t>& shmem) {
        write_tma_descriptor(shmem, 5120, linear_tma(store, shmem_base, bytes, 4104));
        write_tx_mbar(shmem, 4104, bytes);
    };
}

// The machine's PANIC text (a clean run is a test failure).
std::string i4_panic(nanosm::Machine& machine, const std::string& what) {
    try {
        machine.run();
    } catch (const nanosm::SimError& error) {
        return error.what();
    }
    throw std::runtime_error(what + " did not panic");
}

const std::string i4_fence_wait = "fence.sh: own accesses not at the RAM";

// The ISA side: fence.sh is opcode 0x27 with no operands; qtxc and the hand
// assembler parse it (and refuse operands or a depbar); hex round-trips it;
// qtxc never inserts one (rfc12 section 10: explicit only).
void test_i4_fence_encoding_qtxc_and_assembly() {
    nanosm::Instruction fence;
    fence.opcode = nanosm::Opcode::fence_sh;
    fence.control.stall = 3;
    const auto word = nanosm::encode(fence);
    require(word >> 56 == 0x27 && (word & ((1ull << 38) - 1)) == 0, "fence.sh is not opcode 0x27 with an empty payload");
    const auto decoded = nanosm::decode(word);
    require(decoded.opcode == nanosm::Opcode::fence_sh && decoded.control.stall == 3 &&
            nanosm::format_instruction(decoded, false) == "fence.sh",
        "fence.sh did not round-trip: " + nanosm::format_instruction(decoded));
    require(nanosm::fixed_latency(decoded, canonical_profile().timing) == 0, "fence.sh has a fixed latency");

    const auto& profile = canonical_profile();
    const auto program = nanosm::compile_qtx("mov r0, 0\nsts [r0], r0\nfence.sh\nhalt\n", profile);
    require(program.instructions.size() == 4 && program.instructions[2].opcode == nanosm::Opcode::fence_sh &&
            program.instructions[2].control.set_depbar == 7,
        "qtxc did not schedule the written fence.sh in place");
    const auto kernel = nanosm::assemble_asm(nanosm::emit_asm(program), profile);
    require(nanosm::load_hex(nanosm::emit_hex(kernel)).words == kernel.words, "fence.sh did not survive hex");
    for (const auto* bad : {"fence.sh r1\nhalt\n", "fence.sh [r0]\nhalt\n"}) {
        try {
            (void)nanosm::compile_qtx(bad, profile);
            throw std::runtime_error(std::string("qtxc accepted '") + bad + "'");
        } catch (const nanosm::SimError& error) {
            require(std::string(error.what()).find("expected 0 operands") != std::string::npos,
                std::string("fence.sh with an operand: wrong error: ") + error.what());
        }
    }
    try {
        (void)hand_assemble(i1_asm(profile, "fence.sh {pred=PT, stall=0, set=sb0, wait=-}\nhalt\n"), profile);
        throw std::runtime_error("the hand assembler let fence.sh set a depbar");
    } catch (const nanosm::SimError& error) {
        require(std::string(error.what()).find("this opcode cannot set a depbar") != std::string::npos,
            std::string("fence.sh with a depbar: wrong error: ") + error.what());
    }
    // No auto-insertion: an sts the TMA then stores, with no fence written, gets none.
    const auto unfenced = nanosm::compile_qtx(
        "mov r8, 0\nmov r9, 0\nmov r10, 0\nsts [r8], r8\ntma.async [5120], r8\nhalt\n", profile);
    require(std::none_of(unfenced.instructions.begin(), unfenced.instructions.end(),
                [](const auto& inst) { return inst.opcode == nanosm::Opcode::fence_sh; }) &&
            unfenced.instructions.size() == 6,
        "qtxc inserted a fence.sh");
}

// rfc12 section 10: fence.sh is admitted in the cycle its warp's LAST earlier
// shmem access reaches the RAM -- a store round's write edge, a load round's
// sample: its grant + D -- so the instruction after it issues once that edge
// has passed. At D = 2 and 3, hand assembly, one instruction per cycle:
//   one-round sts: issued c3, granted c4; fence.sh refused c4 .. c3 + D,
//     admitted at c4 + D; the next instruction at c5 + D;
//   two-round sts (lanes l and l + 8 share a bank): rounds at c8 and c9;
//     fence.sh admitted at c9 + D (the last round);
//   lds: granted c3, samples at c3 + D: fence.sh is admitted then, while the
//     words are still on their way to the RF (its write comes later);
//   a TMA store of the sts's words right after the fence: accepted with the
//     store at the RAM, so the missing-fence check passes (sanitizer build)
//     and gmem gets the new words.
// (Mutation: admitting fence.sh at "no unserviced slot" alone -- without the
// + D -- puts it at the last grant, c4 / c9 / c3, and fails every case.)
// I6a re-derivation: the access issues at s (its register sources' fixed
// distances, alu_distance each, decide it) and reaches its slot
// issue_to_dispatch (t) later, so its first grant is g = s + t + 1; the
// fence is refused from s + 1 (t + D cycles for one round) and admitted at
// the last grant + D as before. (I6b-2: every program starts at
// fetch_to_issue.)
void test_i4_fence_waits_until_own_accesses_reach_ram() {
    for (const uint32_t d : {2u, 3u}) {
        auto profile = canonical_profile();
        profile.shmem_port_a_depth = d;
        const auto what = "D=" + std::to_string(d) + ": ";
        using Slices = std::vector<std::pair<uint64_t, uint64_t>>;
        const uint64_t t = nanosm::issue_to_dispatch(profile);
        const uint64_t distance = alu_distance(profile);
        const uint64_t first = nanosm::fetch_to_issue; // I6b-2: the first instruction's issue

        auto one = i4_machine(profile, R"(
            sread r0, LANE_ID
            shli r0, r0, 2
            mov r5, 0x777
            sts [r0 + 256], r5 {pred=PT, stall=0, set=sb0, wait=-}
            fence.sh
            mov r9, 1
            halt {pred=PT, stall=0, set=-, wait=sb0}
        )");
        one->run();
        const uint64_t s1 = first + 2 * distance + 1; // sread, shli, mov r5; the sts needs r5
        const uint64_t g1 = s1 + t + 1;
        require(issued_at(*one, 3 * 8) == std::vector<uint64_t>{s1} &&
                i2_grants(*one) == std::vector<I2Grant>{{g1, "A-write", "simt", 0}},
            what + "one-round sts:" + describe(i2_grants(*one)) + " | issues" + issue_log(*one));
        require(issued_at(*one, 4 * 8) == std::vector<uint64_t>{g1 + d} &&
                issued_at(*one, 5 * 8) == std::vector<uint64_t>{g1 + 1 + d} &&
                refusals(*one, 0, i4_fence_wait) == Slices{{s1 + 1, t + d}},
            what + "one-round sts: fence.sh not admitted exactly at its grant + D (c" + std::to_string(g1 + d) +
                "); waits:" + describe(refusals(*one, 0, i4_fence_wait)));

        auto two = i4_machine(profile, R"(
            sread r0, LANE_ID
            andi r1, r0, 7
            shri r2, r0, 3
            shli r1, r1, 2
            shli r2, r2, 6
            add r1, r1, r2
            mov r5, 0x222
            sts [r1 + 3072], r5 {pred=PT, stall=0, set=sb0, wait=-}
            fence.sh
            mov r9, 1
            halt {pred=PT, stall=0, set=-, wait=sb0}
        )");
        two->run();
        // sread; andi / shri; shli / shli; add (needs the second shli); mov r5;
        // the sts needs r5.
        const uint64_t s2 = first + 4 * distance + 2;
        const uint64_t g2 = s2 + t + 1;
        require(issued_at(*two, 7 * 8) == std::vector<uint64_t>{s2} &&
                i2_grants(*two) == std::vector<I2Grant>{{g2, "A-write", "simt", 0}, {g2 + 1, "A-write", "simt", 0}},
            what + "two-round sts:" + describe(i2_grants(*two)) + " | issues" + issue_log(*two));
        require(issued_at(*two, 8 * 8) == std::vector<uint64_t>{g2 + 1 + d} &&
                refusals(*two, 0, i4_fence_wait) == Slices{{s2 + 1, t + 1 + d}},
            what + "two-round sts: fence.sh not admitted at its LAST round's grant + D (c" + std::to_string(g2 + 1 + d) +
                "); waits:" + describe(refusals(*two, 0, i4_fence_wait)));

        auto load = i4_machine(profile, R"(
            sread r0, LANE_ID
            shli r0, r0, 2
            lds r6, [r0 + 256] {pred=PT, stall=0, set=sb0, wait=-}
            fence.sh
            mov r9, 1
            halt {pred=PT, stall=0, set=-, wait=sb0}
        )");
        const auto written = i1_run(*load, {6});
        const uint64_t l = first + 2 * distance; // sread, shli; the lds needs r0
        const uint64_t gl = l + t + 1;
        require(issued_at(*load, 2 * 8) == std::vector<uint64_t>{l} &&
                i2_grants(*load) == std::vector<I2Grant>{{gl, "A-read", "simt", 0}},
            what + "lds:" + describe(i2_grants(*load)) + " | issues" + issue_log(*load));
        require(issued_at(*load, 3 * 8) == std::vector<uint64_t>{gl + d} &&
                refusals(*load, 0, i4_fence_wait) == Slices{{l + 1, t + d}},
            what + "lds: fence.sh not admitted at its sample (grant + D = c" + std::to_string(gl + d) + ")");
        require(written.at(6).size() == 1 && written.at(6).front() > gl + d && load->read_reg(0, 5, 6) == 0x100 + 64 + 5,
            what + "lds: the fence waited for the RF write, or the lds read the wrong word");

        // The TMA's coordinates and mbar are written first (I6a): their fixed
        // distances are long met when the TMA follows the fence, so qtxc's
        // static schedule puts no stall between them.
        auto engine = i4_machine(profile, R"(
            mov r8, 0
            mov r9, 0
            mov r10, 0
            mov r12, 4104
            sread r0, LANE_ID
            shli r0, r0, 2
            mov r5, 0x777
            sts [r0], r5 {pred=PT, stall=0, set=sb0, wait=-}
            fence.sh
            tma.async [5120], r8
        wait: mbar.try_wait p0, [r12], 0
            bra.uni wait {pred=!p0, stall=0, set=-, wait=-}
            halt {pred=PT, stall=0, set=-, wait=sb0}
        )", i4_tma(true, 0, 1024));
        engine->run(); // the sanitizer build: no missing-fence PANIC
        const uint64_t ge = first + 4 + 2 * distance + 1 + t + 1; // four movs, then as in `one`
        require(i2_grants(*engine, "A-write").front().cycle == ge && issued_at(*engine, 8 * 8) == std::vector<uint64_t>{ge + d} &&
                issued_at(*engine, 9 * 8) == std::vector<uint64_t>{ge + 1 + d},
            what + "the TMA store did not follow the fence at the sts's grant + D + 1:" + describe(i2_grants(*engine)) +
                " | issues" + issue_log(*engine));
        for (uint32_t i = 0; i < 256; ++i) {
            require(engine->read_gmem_u32(4 * i) == (i < 16 ? 0x777u : 0x100 + i), what + "TMA store word " + std::to_string(i));
        }
    }
}

// SANITIZER (rfc12 sections 10 and 11): each case a warp's own access that
// has NOT reached the RAM when an engine command over the same bytes is
// accepted -- one write involved -- PANICs "missing fence.sh"; the same
// program with fence.sh between them runs and computes the fenced result:
//   sts -> TMA store of those words (its source reads);
//   sts -> tma.async whose DESCRIPTOR the sts just wrote (copied in through
//     lds / sts; before I4 the descriptor peek overlaid in-flight stores);
//   sts -> mma.async reading that tile (port B), one bank: sixteen rounds;
//   lds -> TMA load overwriting the words the lds reads (it has not sampled).
// The PANIC is deliberate: the conflicting access is still outstanding (the
// command issues the cycle after it).
void test_i4_missing_fence_panics() {
    const auto& profile = canonical_profile();
    const auto program = [](const std::string& before, bool fenced, const std::string& command) {
        return before + (fenced ? "fence.sh\n" : "") + command + R"(
        wait:
            mbar.try_wait p0, [r4], 0
            @!p0 bra.uni wait
            halt
        )";
    };
    const std::string prologue = R"(
        mov r8, 0
        mov r9, 0
        mov r10, 0
        mov r4, 4104
        sread r20, LANE_ID
        shli r20, r20, 2
    )";

    // sts -> TMA store source.
    const auto store_source = prologue + "mov r21, 7\nsts [r20], r21\n";
    auto message = i4_panic(*i4_machine(profile, program(store_source, false, "tma.async [5120], r8\n"),
        i4_tma(true, 0, 1024), 1, false), "sts -> TMA store source");
    require_mentions(message, {"reason=missing fence.sh: tma store #0 source bytes [0, 64) overlap this warp's earlier "
        "sts [r20], r21 (pc 56, issued cycle", "whose write has not reached the RAM; issue fence.sh between them",
        "inst='tma.async [5120], r8"}, "sts -> TMA store source");
    auto fenced = i4_machine(profile, program(store_source, true, "tma.async [5120], r8\n"), i4_tma(true, 0, 1024), 1, false);
    fenced->run();
    for (uint32_t i = 0; i < 256; ++i) {
        require(fenced->read_gmem_u32(4 * i) == (i < 16 ? 7u : 0x100 + i), "fenced TMA store word " + std::to_string(i));
    }

    // sts -> the descriptor: copied from 6144 into 5120 (zeros: invalid).
    const auto descriptor_copy = [](std::vector<uint8_t>& shmem) {
        write_tma_descriptor(shmem, 6144, linear_tma(false, 1024, 64, 4104));
        for (uint32_t i = 0; i < 16; ++i) write_u32(shmem, 5120 + 4 * i, 0);
        write_tx_mbar(shmem, 4104, 64);
    };
    const auto copy = prologue + "lds r21, [r20 + 6144]\nsts [r20 + 5120], r21\n";
    message = i4_panic(*i4_machine(profile, program(copy, false, "tma.async [5120], r8\n"), descriptor_copy, 1, false),
        "sts -> TMA descriptor");
    require_mentions(message, {"reason=missing fence.sh: tma #0 descriptor bytes [5120, 5184) overlap this warp's earlier "
        "sts [r20 + 5120], r21", "whose write has not reached the RAM"}, "sts -> TMA descriptor");
    fenced = i4_machine(profile, program(copy, true, "tma.async [5120], r8\n"), descriptor_copy, 1, false);
    fenced->run();
    for (uint32_t i = 0; i < 16; ++i) {
        require(fenced->read_shmem_u32(1024 + 4 * i) == 5 * i + 2, "the TMA with the copied descriptor, word " + std::to_string(i));
    }

    // sts -> mma.async on that tile (one bank: the sts is sixteen rounds).
    const auto dense = dense_one_bank_profile(4);
    const auto mma_machine = [&](bool fence) {
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.shmem_image, 1024);
        for (uint32_t i = 0; i < 128; ++i) write_u16(shmem, 2 * i, 0x3c00);       // A: 1.0
        for (uint32_t i = 0; i < 32; ++i) write_u16(shmem, 512 + 2 * i, 0x3c00);  // B: 1.0
        write_count_mbar(shmem, 896, 1);
        return std::make_unique<nanosm::Machine>(dense, compile(std::string(R"(
            .acc C, 0, 0, 16, 4
            mov r0, 0
            mov r1, 512
            mov r4, 896
            sread r20, LANE_ID
            shli r20, r20, 2
            mov r21, 16384
            shli r22, r21, 16
            or r21, r21, r22
            sts [r20], r21
        )") + (fence ? "fence.sh\n" : "") + R"(
            mma.async.init C, [r0], [r1]
            mma.commit [r4]
        wait:
            mbar.try_wait p0, [r4], 0
            @!p0 bra.uni wait
            mov r8, 0
            mov r9, 0
            acc.ld r10, C, r8
            halt
        )", dense), std::move(launch));
    };
    message = i4_panic(*mma_machine(false), "sts -> mma.async");
    require_mentions(message, {"reason=missing fence.sh: mma.async #1 A operand bytes [0, 64) overlap this warp's earlier "
        "sts [r20], r21", "inst='mma.async.init ACC0, [r0], [r1]"}, "sts -> mma.async");
    auto mma = mma_machine(true);
    mma->run(4000);
    require(nanosm::bits_float(mma->read_reg(0, 0, 10)) == 16.0f, "the fenced MMA did not read the stored 2.0 halves");

    // lds -> TMA load over the words it reads.
    const auto read_then_load = prologue + "lds r21, [r20]\n";
    message = i4_panic(*i4_machine(profile, program(read_then_load, false, "tma.async [5120], r8\n"),
        i4_tma(false, 0, 1024), 1, false), "lds -> TMA load");
    require_mentions(message, {"reason=missing fence.sh: tma load #0 destination bytes [0, 64) overlap this warp's earlier "
        "lds r21, [r20] (pc 48, issued cycle", "whose read has not sampled the RAM; issue fence.sh between them"},
        "lds -> TMA load");
    fenced = i4_machine(profile, program(read_then_load, true, "tma.async [5120], r8\n"), i4_tma(false, 0, 1024), 1, false);
    fenced->run();
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(fenced->read_reg(0, lane, 21) == 0x100 + lane, "the fenced lds did not read the old word");
    }
    for (uint32_t i = 0; i < 256; ++i) require(fenced->read_shmem_u32(4 * i) == 5 * i + 2, "the fenced TMA load");
}

// SANITIZER (rfc12 section 11): another warp's access still travelling to
// the RAM is a RACE, not a missing fence (cross-warp ordering is by mbar).
// Warp 1 runs a sixteen-round access (all lanes on bank 0: 6144 + 64 * lane);
// warp 0, after a stall, accepts a TMA over those words while it is under way:
//   warp 1 sts, warp 0 TMA store (the engine reads, the access writes);
//   warp 1 lds, warp 0 TMA load (the engine writes, the access reads).
// Control, the FA handoff: warp 1's sts, then mbar.arrive (admitted once the
// sts is serviced; no depbar wait), warp 0 waits on the mbar and stores the
// words: the arrive's RMW commits after the store's last round reached the
// RAM, so the TMA's accept finds nothing in flight.
// I6a: the TMA is accepted at its dispatch and warp 1's access reaches its
// slot at its own; the stall (was 8) is re-chosen so the TMA's accept falls
// inside warp 1's rounds (before, warp 1's sts was dispatched into the TMA's
// bytes and PANICked from warp 1's side).
void test_i4_cross_warp_race() {
    const auto& profile = canonical_profile();
    const auto two_warps = [](const std::string& access) {
        return R"(
            sread r29, WARP_ID
            setp.ne p6, r29, 0
            bra.uni other {pred=p6, stall=0, set=-, wait=-}
            mov r8, 0
            mov r9, 0
            mov r10, 0 {pred=PT, stall=20, set=-, wait=-}
            tma.async [5120], r8
            mov r4, 4104
        wait: mbar.try_wait p0, [r4], 0
            bra.uni wait {pred=!p0, stall=0, set=-, wait=-}
            halt
        other: sread r20, LANE_ID
            shli r20, r20, 6
            mov r21, 0x66
        )" + access + R"(
            halt {pred=PT, stall=0, set=-, wait=sb0}
        )";
    };
    // The TMA's accept lies inside the sixteen rounds' travel (first grant .. last grant + D).
    const auto under_way = [&](nanosm::Machine& machine, const std::string& message, const std::string& service) {
        const auto rounds = i2_grants(machine, service);
        const auto accept = std::stoull(message.substr(message.find("cycle=") + 6));
        require(!rounds.empty() && rounds.front().cycle <= accept &&
                accept < rounds.back().cycle + profile.shmem_port_a_depth,
            "the TMA was not accepted while warp 1's access was under way (c" + std::to_string(accept) + "):" +
                describe(rounds));
    };
    auto machine = i4_machine(profile, two_warps("sts [r20 + 6144], r21 {pred=PT, stall=0, set=sb0, wait=-}"),
        i4_tma(true, 6144, 1024), 2);
    auto message = i4_panic(*machine, "warp 1 sts vs warp 0 TMA store");
    require_mentions(message, {"warp=0", "reason=shmem race: tma store #0 source bytes [",
        "overlap warp 1's sts [r20 + 6144], r21 (pc", "whose write has not reached the RAM; order them with an mbar"},
        "cross-warp sts vs TMA store");
    under_way(*machine, message, "A-write");
    machine = i4_machine(profile, two_warps("lds r22, [r20 + 6144] {pred=PT, stall=0, set=sb0, wait=-}"),
        i4_tma(false, 6144, 1024), 2);
    message = i4_panic(*machine, "warp 1 lds vs warp 0 TMA load");
    require_mentions(message, {"warp=0", "reason=shmem race: tma load #0 destination bytes [",
        "overlap warp 1's lds r22, [r20 + 6144] (pc", "whose read has not sampled the RAM; order them with an mbar"},
        "cross-warp lds vs TMA load");
    under_way(*machine, message, "A-read");

    auto handoff = i4_machine(profile, R"(
            sread r29, WARP_ID
            setp.ne p6, r29, 0
            bra.uni producer {pred=p6, stall=0, set=-, wait=-}
            mov r8, 0
            mov r9, 0
            mov r10, 0
            mov r4, 4104
            mov r5, 4112
        ready: mbar.try_wait p0, [r5], 0
            bra.uni ready {pred=!p0, stall=0, set=-, wait=-}
            tma.async [5120], r8
        stored: mbar.try_wait p0, [r4], 0
            bra.uni stored {pred=!p0, stall=0, set=-, wait=-}
            halt
        producer: sread r20, LANE_ID
            shli r20, r20, 6
            mov r21, 0x66
            mov r5, 4112
            sts [r20 + 6144], r21 {pred=PT, stall=0, set=sb0, wait=-}
            mbar.arrive [r5]
            halt {pred=PT, stall=0, set=-, wait=sb0}
        )", [](std::vector<uint8_t>& shmem) {
            i4_tma(true, 6144, 1024)(shmem);
            write_count_mbar(shmem, 4112, 1);
        }, 2);
    handoff->run();
    for (uint32_t i = 0; i < 256; ++i) {
        require(handoff->read_gmem_u32(4 * i) == (i % 16 == 0 ? 0x66u : 0x100 + 1536 + i), "handoff word " + std::to_string(i));
    }
}

// rfc12 section 11: read / read overlap never PANICs. An lds still under way
// (one bank: sixteen rounds) when an mma.async.init reading the same words is
// accepted; an lds that has not sampled when a TMA store of its words is
// accepted. Both run, and the asserted timing shows the lds really was in
// flight at the accept.
void test_i4_read_read_overlap_is_clean() {
    const auto dense = dense_one_bank_profile(4);
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.shmem_image, 1024);
    for (uint32_t i = 0; i < 128; ++i) write_u16(shmem, 2 * i, 0x3c00);       // A: 1.0
    for (uint32_t i = 0; i < 32; ++i) write_u16(shmem, 512 + 2 * i, 0x3c00);  // B: 1.0
    write_count_mbar(shmem, 896, 1);
    nanosm::Machine mma(dense, compile(R"(
        .acc C, 0, 0, 16, 4
        mov r0, 0
        mov r1, 512
        mov r4, 896
        sread r20, LANE_ID
        shli r20, r20, 2
        lds r21, [r20]
        mma.async.init C, [r0], [r1]
        mma.commit [r4]
    wait:
        mbar.try_wait p0, [r4], 0
        @!p0 bra.uni wait
        mov r8, 0
        mov r9, 0
        acc.ld r10, C, r8
        halt
    )", dense), std::move(launch), nanosm::TraceOptions{.timeline = true, .debug = true});
    mma.run(4000);
    const auto reads = i2_grants(mma, "A-read");
    const auto accept = issued_at(mma, 6 * 8);
    require(reads.size() >= 16 && accept.size() == 1 && reads[15].cycle + dense.shmem_port_a_depth > accept.front(),
        "the lds was not still in flight when the MMA was accepted:" + describe(reads));
    require(nanosm::bits_float(mma.read_reg(0, 0, 10)) == 8.0f && mma.read_reg(0, 3, 21) == 0x3c003c00u,
        "lds + MMA read/read results");

    const auto& profile = canonical_profile();
    auto store = i4_machine(profile, R"(
        mov r8, 0
        mov r9, 0
        mov r10, 0
        mov r4, 4104
        sread r20, LANE_ID
        shli r20, r20, 2
        lds r21, [r20] {pred=PT, stall=0, set=sb0, wait=-}
        tma.async [5120], r8
    wait: mbar.try_wait p0, [r4], 0
        bra.uni wait {pred=!p0, stall=0, set=-, wait=-}
        halt {pred=PT, stall=0, set=-, wait=sb0}
    )", i4_tma(true, 0, 1024));
    store->run();
    const auto lds = i2_grants(*store, "A-read");
    require(!lds.empty() && lds.front().client == "simt" &&
            lds.front().cycle + profile.shmem_port_a_depth > issued_at(*store, 7 * 8).at(0),
        "the lds had sampled before the TMA store's accept:" + describe(lds));
    for (uint32_t i = 0; i < 256; ++i) require(store->read_gmem_u32(4 * i) == 0x100 + i, "lds + TMA store word");
    require(store->read_reg(0, 7, 21) == 0x100 + 7, "lds + TMA store: the lds");
}

// SANITIZER, C10 live-mbar opacity (rfc12 section 11): an mbar.init makes its
// 8 bytes a live mbar until kernel end; only mbar operations may touch it.
//   sts onto it after the init PANICs (at the sts round's grant) naming the
//   mbar; the same sts BEFORE the init is legal (the init repurposes it);
//   a TMA load whose placement writes it PANICs at that placement round. The
//   init must reach the RAM before the TMA's accept (fence.sh); without the
//   fence the in-flight init itself is the missing-fence PANIC;
//   mbar.init of a live mbar PANICs (implementer choice: no mbar.inval in
//   v0, so an mbar is initialized once per kernel).
// mbar operations on a live mbar stay legal (e.g. "mbar sleep and wake":
// init, arrive.expect_tx, try_wait, engine complete_tx). The machine's
// live-waiter PANICs are gone (backlog D12 / D12.1): C10 is the only check.
void test_i4_live_mbar_opacity() {
    const auto& profile = canonical_profile();
    const std::string lanes = R"(
        sread r20, LANE_ID
        shli r20, r20, 2
        add r20, r20, r4
        mov r21, 9
    )";
    auto message = i4_panic(*i4_machine(profile, "mov r4, 4160\nmbar.init [r4], 1\n" + lanes + "sts [r20], r21\nhalt\n",
        {}, 1, false), "sts onto a live mbar");
    require_mentions(message, {"reason=live mbar opacity: sts writes bytes [4160, 4168) of the live mbar [4160, 4168) "
        "(mbar.init by warp 0 pc 8 at cycle ", "until kernel end only mbar operations may touch it",
        "inst='sts [r20], r21"}, "sts onto a live mbar");
    auto repurpose = i4_machine(profile, "mov r4, 4160\n" + lanes + "sts [r20], r21\nmbar.init [r4], 1\nhalt\n", {}, 1, false);
    repurpose->run();
    const auto state = repurpose->read_mbar_state(4160);
    require(state.valid && state.expected == 1 && repurpose->read_shmem_u32(4168) == 9,
        "an sts before the mbar.init was refused, or the init did not land after it");

    const std::string placement = R"(
        mov r8, 0
        mov r9, 0
        mov r10, 0
        mov r4, 4104
        mov r5, 1032
        mbar.init [r5], 1
    )";
    const std::string load = R"(
        tma.async [5120], r8
    wait:
        mbar.try_wait p0, [r4], 0
        @!p0 bra.uni wait
        halt
    )";
    message = i4_panic(*i4_machine(profile, placement + "fence.sh\n" + load, i4_tma(false, 1024, 64), 1, false),
        "TMA placement over a live mbar");
    require_mentions(message, {"mbar address=1032 reason=live mbar opacity: tma load #0 placement writes bytes "
        "[1032, 1040) of the live mbar [1032, 1040) (mbar.init by warp 0 pc 40 at cycle "}, "TMA placement over a live mbar");
    message = i4_panic(*i4_machine(profile, placement + load, i4_tma(false, 1024, 64), 1, false),
        "TMA load over an mbar.init in flight");
    require_mentions(message, {"reason=missing fence.sh: tma load #0 destination bytes [1032, 1040) overlap this warp's "
        "earlier mbar.init [r5], 1 (pc 40"}, "TMA load over an mbar.init in flight");

    message = i4_panic(*i4_machine(profile, "mov r4, 4160\nmbar.init [r4], 1\nmbar.init [r4], 2\nhalt\n", {}, 1, false),
        "mbar.init over a live mbar");
    require_mentions(message, {"reason=mbar.init over the live mbar [4160, 4168) (mbar.init by warp 0 pc 8 at cycle ",
        "(v0 has no mbar.inval: an mbar is initialized once per kernel)", "inst='mbar.init [r4], 2"},
        "mbar.init over a live mbar");
}

// ---------------------------------------------------------------------------
// I5: uniform shared atomics (trash/rfc13.md; trash/i5r-uniform-atomics-
// 2026-10-12.md), backlog D12 (machine quiet, sanitizer PANIC) and the C10
// follow-up (launch-declared mbars are live; C11: their addresses are checked
// before narrowing). Hand assembly under a debug trace, as in the I2 - I4
// sections. An atomic is an RMW-engine operation: its "atomic RMW" slice runs
// from its START (the store pause begins) to its write grant + 1; inside it,
// its read and its write are the "rmw" grants on A-read and A-write.
// ---------------------------------------------------------------------------

struct I5Atomic {
    uint64_t start = 0;
    uint64_t end = 0;             // the write grant + 1
    std::vector<uint64_t> reads;  // its read grant
    std::vector<uint64_t> writes; // its write grant
    std::string detail;
};

std::string describe(const I5Atomic& atomic) {
    std::string text = " {" + atomic.detail + ": start c" + std::to_string(atomic.start) + ", end c" +
        std::to_string(atomic.end) + ", reads";
    for (const auto cycle : atomic.reads) text += " " + std::to_string(cycle);
    text += ", writes";
    for (const auto cycle : atomic.writes) text += " " + std::to_string(cycle);
    return text + "}";
}

std::vector<I5Atomic> i5_atomics(const nanosm::Machine& machine) {
    std::vector<I5Atomic> atomics;
    for (const auto& event : machine.trace().events()) {
        if (event.name != "atomic RMW") continue;
        atomics.push_back({event.start_cycle, event.start_cycle + event.duration_cycles, {}, {}, event.detail});
    }
    std::sort(atomics.begin(), atomics.end(), [](const auto& a, const auto& b) { return a.start < b.start; });
    for (const auto& grant : i2_grants(machine)) {
        if (grant.client != "rmw") continue;
        for (auto& atomic : atomics) {
            if (grant.cycle < atomic.start || grant.cycle >= atomic.end) continue;
            (grant.service == "A-read" ? atomic.reads : atomic.writes).push_back(grant.cycle);
        }
    }
    return atomics;
}

// The one atomic a run made.
I5Atomic i5_atomic(const nanosm::Machine& machine, const std::string& what) {
    const auto atomics = i5_atomics(machine);
    require(atomics.size() == 1, what + ": expected one atomic, found " + std::to_string(atomics.size()));
    return atomics.front();
}

// rfc13 section 6: one read at g, the old word at the port output at
// a = g + D + Lb, the write granted at w = a + C, the slice ending at w + 1.
// `read` is the expected g.
void i5_require_timeline(const I5Atomic& atomic, const nanosm::DeviceProfile& profile, uint64_t read,
    const std::string& what) {
    const uint64_t w = read + profile.shmem_port_a_depth + profile.shmem_read_latency_cycles +
        profile.shmem_rmw_compute_cycles;
    require(atomic.reads == std::vector<uint64_t>{read} && atomic.writes == std::vector<uint64_t>{w} &&
            atomic.end == w + 1,
        what + ": not read c" + std::to_string(read) + ", write c" + std::to_string(w) + ":" + describe(atomic));
}

// The atomic ALU, written independently of the machine's: 32-bit wrap through
// 64-bit arithmetic.
uint32_t i5_expected(nanosm::Opcode opcode, uint32_t old_word, uint32_t operand) {
    switch (opcode) {
    case nanosm::Opcode::atom_add_uni: return static_cast<uint32_t>((uint64_t{old_word} + operand) & 0xffffffffull);
    case nanosm::Opcode::atom_and_uni: return old_word & operand;
    case nanosm::Opcode::atom_xor_uni: return old_word ^ operand;
    default: return operand; // exch
    }
}

// An I2-style machine (shmem words 0..2047 hold 0x100 + word; gmem words
// 0..1023 hold 5 * word + 2); `setup` may change the launch (its image's
// shmem bytes are launch.shmem_image[0].bytes; its declared mbars).
std::unique_ptr<nanosm::Machine> i5_machine(const nanosm::DeviceProfile& profile, const std::string& program,
    uint32_t warps = 1, const std::function<void(nanosm::Launch&)>& setup = {}, bool hand_assembly = true) {
    nanosm::Launch launch;
    launch.active_warps = warps;
    auto& shmem = image_bytes(launch.shmem_image, 8192);
    for (uint32_t word = 0; word < 2048; ++word) write_u32(shmem, 4 * word, 0x100 + word);
    auto& gmem = image_bytes(launch.gmem_image, 4096);
    for (uint32_t word = 0; word < 1024; ++word) write_u32(gmem, 4 * word, 5 * word + 2);
    if (setup) setup(launch);
    const auto kernel = hand_assembly ? hand_assemble(i1_asm(profile, program), profile)
                                      : compile(program, profile);
    return std::make_unique<nanosm::Machine>(profile, kernel, std::move(launch),
        nanosm::TraceOptions{.timeline = true, .debug = true});
}

const std::array<std::pair<nanosm::Opcode, const char*>, 4> i5_ops = {{
    {nanosm::Opcode::atom_add_uni, "atom.add.uni"},
    {nanosm::Opcode::atom_and_uni, "atom.and.uni"},
    {nanosm::Opcode::atom_xor_uni, "atom.xor.uni"},
    {nanosm::Opcode::atom_exch_uni, "atom.exch.uni"},
}};

// The ISA side: four opcodes 0x39..0x3c, payload dst[8] base[8] operand[8]
// simm[14]; qtxc and the hand assembler parse "atom.op.uni rD, [rA + imm],
// rB"; the offset is a signed 14-bit field; an atomic returns its old value
// through a depbar (qtxc allocates one, a consumer of rD waits on it; the hand
// assembler requires one); latency 0 (an async RF writer); hex round trip.
// The per-lane spelling "atom.add" is gone.
void test_i5_atomic_encoding_and_assembly() {
    const auto& profile = canonical_profile();
    uint32_t code = 0x39;
    for (const auto& [opcode, name] : i5_ops) {
        for (const int32_t offset : {-8192, 8191, 0, -4}) {
            nanosm::Instruction inst;
            inst.opcode = opcode;
            inst.dst = 200;
            inst.src0 = 129;
            inst.src1 = 255;
            inst.imm = offset;
            inst.control.set_depbar = 3;
            const auto word = nanosm::encode(inst);
            const auto decoded = nanosm::decode(word);
            require(word >> 56 == code && decoded.opcode == opcode && decoded.dst == 200 && decoded.src0 == 129 &&
                    decoded.src1 == 255 && decoded.imm == offset && decoded.control.set_depbar == 3,
                std::string(name) + " did not round-trip with offset " + std::to_string(offset));
            require(nanosm::fixed_latency(decoded, profile.timing) == 0, std::string(name) + " has a fixed latency");
        }
        nanosm::Instruction text;
        text.opcode = opcode;
        text.dst = 7;
        text.src0 = 1;
        text.src1 = 5;
        text.imm = 64;
        require(nanosm::format_instruction(text, false) == std::string(name) + " r7, [r1 + 64], r5",
            "format: " + nanosm::format_instruction(text, false));
        text.imm = 8192;
        try {
            (void)nanosm::encode(text);
            throw std::runtime_error(std::string(name) + " encoded a 15-bit offset");
        } catch (const nanosm::SimError& error) {
            require(std::string(error.what()).find("signed 14-bit") != std::string::npos, error.what());
        }
        ++code;
    }
    const auto program = nanosm::compile_qtx(
        "mov r1, 0\nmov r2, 5\natom.xor.uni r3, [r1 - 64], r2\nadd r4, r3, r3\nhalt\n", profile);
    const auto& atom = program.instructions.at(2);
    const auto& consumer = program.instructions.at(3);
    require(atom.opcode == nanosm::Opcode::atom_xor_uni && atom.imm == -64 && atom.control.set_depbar < 6 &&
            (consumer.control.wait_mask >> atom.control.set_depbar & 1) != 0,
        "qtxc did not give the atomic a depbar its rD consumer waits on");
    const auto kernel = nanosm::assemble_asm(nanosm::emit_asm(program), profile);
    require(nanosm::load_hex(nanosm::emit_hex(kernel)).words == kernel.words, "atom did not survive hex");
    const auto rejects = [&](const std::string& source, bool hand, const std::string& needle) {
        try {
            if (hand) (void)hand_assemble(i1_asm(profile, source), profile);
            else (void)nanosm::compile_qtx(source, profile);
        } catch (const nanosm::SimError& error) {
            require(std::string(error.what()).find(needle) != std::string::npos,
                "'" + source + "': wrong error: " + error.what());
            return;
        }
        throw std::runtime_error("'" + source + "' was accepted");
    };
    rejects("atom.add.uni r3, [r1 + 8192], r2\nhalt\n", false, "atom offset must fit a signed 14-bit immediate");
    rejects("atom.add.uni r3, [r1 - 8193], r2\nhalt\n", false, "atom offset must fit a signed 14-bit immediate");
    rejects("atom.add.uni r3, [r1]\nhalt\n", false, "expected 3 operands");
    rejects("atom.min.uni r3, [r1], r2\nhalt\n", false, "unknown opcode 'atom.min.uni'");
    rejects("atom.add r3, [r1], r2\nhalt\n", false, "unknown opcode 'atom.add'");
    rejects("atom.exch.uni r3, [r1], r2\nhalt\n", true, "memory operation requires set=sb0..sb5");
}

// rfc13 sections 1 and 6: ONE word per warp instruction, the update applied
// ONCE (all sixteen lanes hold the operand 0x5a5: add adds it once), every
// lane's rD gets the old word. One read, granted when the engine starts it
// (the cycle after the atomic's issue: nothing else holds A-read); its write
// D + Lb + C later; the slice ends at that write + 1. The old word reaches rD
// through the return slot at a + 1 (a = the read + D + Lb), BEFORE the write
// grant a + C; the warp's next shmem access (an lds.uni of the word) is
// admitted only at the write grant and reads the new word. Each op at the
// canonical D = Lb = C = 2 (7 cycles from start to the end of the pause);
// add also at D = 3, C = 3.
// I6a: the atomic reaches the SIMT RMW register at its dispatch, so it
// starts issue_to_dispatch + 1 after its issue; rD's RF write is granted at
// a + 1 (still before the word's write grant) and lands Wd later; the
// lds.uni, admitted at the write grant w, is granted at its dispatch + 1,
// w + issue_to_dispatch + 1.
void test_i5_one_word_timeline() {
    const std::string program_tail = R"( r7, [r1 + 2048], r5 {pred=PT, stall=0, set=sb0, wait=-}
        lds.uni r8, [r1 + 2048] {pred=PT, stall=0, set=sb1, wait=-}
        halt {pred=PT, stall=0, set=-, wait=sb0|sb1}
    )";
    const std::string program_head = R"(
        mov r1, 0
        mov r5, 0x5a5
        )";
    struct Case {
        nanosm::Opcode opcode;
        const char* name;
        uint32_t d;
        uint32_t c;
    };
    std::vector<Case> cases;
    for (const auto& [opcode, name] : i5_ops) cases.push_back({opcode, name, 2, 2});
    cases.push_back({nanosm::Opcode::atom_add_uni, "atom.add.uni", 3, 3});
    for (const auto& test : cases) {
        auto profile = canonical_profile();
        profile.shmem_port_a_depth = test.d;
        profile.shmem_rmw_compute_cycles = test.c;
        const auto what = std::string(test.name) + " D=" + std::to_string(test.d) + " C=" + std::to_string(test.c);
        auto machine = i5_machine(profile, program_head + test.name + program_tail);
        const auto written = i1_run(*machine, {7});
        const auto atomic = i5_atomic(*machine, what);
        const auto t = nanosm::issue_to_dispatch(profile);
        const auto issue = issued_at(*machine, 16).at(0);
        require(atomic.start == issue + t + 1, what + ": did not start the cycle after its dispatch");
        i5_require_timeline(atomic, profile, atomic.start, what);
        const auto a = atomic.reads.at(0) + profile.shmem_port_a_depth + profile.shmem_read_latency_cycles;
        const auto w = atomic.writes.at(0);
        require(written.at(7) == std::vector<uint64_t>{a + 1 + profile.rf_write_depth_cycles} && a + 1 < w,
            what + ": rD's RF write was not granted at a + 1 (c" + std::to_string(a + 1) + "), before the write grant c" +
                std::to_string(w) + ", landing Wd later");
        require(issued_at(*machine, 24) == std::vector<uint64_t>{w}, what + ": the lds.uni was not admitted at the write grant");
        const auto lds = i2_grants(*machine, "A-read");
        require(lds.size() == 2 && lds[1] == I2Grant{w + t + 1, "A-read", "simt", lds[1].slot},
            what + ": the lds.uni's read was not granted at w + issue_to_dispatch + 1:" + describe(lds));
        const uint32_t updated = i5_expected(test.opcode, 0x300, 0x5a5);
        require(machine->read_shmem_u32(2048) == updated, what + ": the word was not updated once");
        for (uint32_t lane = 0; lane < 16; ++lane) {
            require(machine->read_reg(0, lane, 7) == 0x300 && machine->read_reg(0, lane, 8) == updated,
                what + ": lane " + std::to_string(lane) + " rD or the lds.uni");
        }
        require(atomic.detail == std::string(test.name) + ", warp=0", what + ": slice detail " + atomic.detail);
        if (test.d == 2 && test.c == 2) require(atomic.end - atomic.start == 7, what + ": not 7 cycles canonical");
    }
}

// int32 add WRAPS (rfc13 section 1: defined 32-bit two's-complement wrap,
// unlike the checked mbar tx count): one atomic each,
//   0x7fffffff + 1 = 0x80000000      0x80000000 + (-1) = 0x7fffffff
//   0xffffffff + 1 = 0               0x80000000 + 0x80000000 = 0
//   0x7fffffff + 0x7fffffff = 0xfffffffe
// and a chain across the boundary on one word: 0x7ffffffe + 1 + 1 + 1 + 1,
// each atomic reading the previous sum. Operands come from a table (lds.uni),
// in registers whose RF bank differs from the base's (r1: bank 1), so each
// atomic's two RF reads fit its one read round (one read port per bank).
void test_i5_add_wraps_at_the_int32_boundary() {
    const auto& profile = canonical_profile();
    const std::array<std::pair<uint32_t, uint32_t>, 5> cases = {{
        {0x7fffffffu, 1u}, {0x80000000u, 0xffffffffu}, {0xffffffffu, 1u}, {0x80000000u, 0x80000000u},
        {0x7fffffffu, 0x7fffffffu},
    }};
    auto machine = i5_machine(profile, R"(
        mov r1, 0
        mov r20, 1
        lds.uni r12, [r1 + 7232] {pred=PT, stall=0, set=sb1, wait=-}
        lds.uni r14, [r1 + 7236] {pred=PT, stall=0, set=sb1, wait=-}
        lds.uni r15, [r1 + 7240] {pred=PT, stall=0, set=sb1, wait=-}
        lds.uni r17, [r1 + 7244] {pred=PT, stall=0, set=sb1, wait=-}
        lds.uni r18, [r1 + 7248] {pred=PT, stall=0, set=sb1, wait=-}
        atom.add.uni r7, [r1 + 2048], r12 {pred=PT, stall=0, set=sb0, wait=sb1}
        atom.add.uni r8, [r1 + 2052], r14 {pred=PT, stall=0, set=sb0, wait=-}
        atom.add.uni r9, [r1 + 2056], r15 {pred=PT, stall=0, set=sb0, wait=-}
        atom.add.uni r10, [r1 + 2060], r17 {pred=PT, stall=0, set=sb0, wait=-}
        atom.add.uni r11, [r1 + 2064], r18 {pred=PT, stall=0, set=sb0, wait=-}
        atom.add.uni r21, [r1 + 2092], r20 {pred=PT, stall=0, set=sb2, wait=-}
        atom.add.uni r22, [r1 + 2092], r20 {pred=PT, stall=0, set=sb2, wait=-}
        atom.add.uni r23, [r1 + 2092], r20 {pred=PT, stall=0, set=sb2, wait=-}
        atom.add.uni r24, [r1 + 2092], r20 {pred=PT, stall=0, set=sb2, wait=-}
        halt {pred=PT, stall=0, set=-, wait=sb0|sb2}
    )", 1, [&](nanosm::Launch& launch) {
        auto& shmem = launch.shmem_image.at(0).bytes;
        for (uint32_t k = 0; k < 5; ++k) {
            write_u32(shmem, 2048 + 4 * k, cases[k].first);
            write_u32(shmem, 7232 + 4 * k, cases[k].second);
        }
        write_u32(shmem, 2092, 0x7ffffffeu);
    });
    machine->run();
    for (uint32_t k = 0; k < 5; ++k) {
        const auto [old_word, operand] = cases[k];
        require(machine->read_shmem_u32(2048 + 4 * k) == static_cast<uint32_t>(uint64_t{old_word} + operand) &&
                machine->read_reg(0, 5, 7 + k) == old_word,
            "case " + std::to_string(k) + ": " + std::to_string(old_word) + " + " + std::to_string(operand));
    }
    for (uint32_t k = 0; k < 4; ++k) {
        require(machine->read_reg(0, 9, 21 + k) == 0x7ffffffeu + k, "chain: atomic " + std::to_string(k));
    }
    require(machine->read_shmem_u32(2092) == 0x80000002u, "chain: the sum did not wrap to 0x80000002");
}

// rfc13 section 4: an atomic reserves its RETURN load slot at admission; with
// none free it is not admitted (structural, counted as a load-slot wait), so
// it never takes the store pause with nowhere to put its old value. One load
// slot: the lds holds it from its grant G until its RF write at G + D + Lb +
// 1; the atomic (not ordered: the lds is serviced at its grant) is refused
// "load slots full" G .. G + D + Lb and admitted in the cycle the lds's RF
// write frees the slot (same-cycle reuse), then runs.
// I6a: the lds issues at l (it needs the shli's r1: 2 alu_distance) and is
// granted at G = l + issue_to_dispatch + 1; the atomic, first tried at l + 2
// (it needs the second mov's r3), is ordered behind the unserviced lds
// until G, then refused for its return slot exactly as before. (I6b-2: the
// sread issues at fetch_to_issue.)
void test_i5_return_slot_admission() {
    auto profile = canonical_profile();
    profile.shmem_load_slots = 1;
    const uint64_t back = profile.shmem_port_a_depth + profile.shmem_read_latency_cycles;
    auto machine = i5_machine(profile, R"(
        sread r0, LANE_ID
        shli r1, r0, 2
        mov r2, 0
        mov r3, 5
        lds r6, [r1 + 256] {pred=PT, stall=0, set=sb1, wait=-}
        atom.add.uni r7, [r2 + 1024], r3 {pred=PT, stall=0, set=sb0, wait=-}
        halt {pred=PT, stall=0, set=-, wait=sb0|sb1}
    )");
    machine->run();
    const auto g = i2_grants(*machine, "A-read").at(0);
    const auto t = nanosm::issue_to_dispatch(profile);
    const auto l = nanosm::fetch_to_issue + 2 * alu_distance(profile);
    require(g.client == "simt" && issued_at(*machine, 32) == std::vector<uint64_t>{l} && g.cycle == l + t + 1,
        "the lds was not granted the cycle after its dispatch:" + issue_log(*machine));
    require(refusals(*machine, 0, "ordered behind own access") == std::vector<std::pair<uint64_t, uint64_t>>{{l + 2, t - 1}},
        "the atomic was not ordered behind the lds until its grant:" +
            describe(refusals(*machine, 0, "ordered behind own access")));
    require(refusals(*machine, 0, "load slots full") == std::vector<std::pair<uint64_t, uint64_t>>{{g.cycle, back + 1}} &&
            issued_at(*machine, 40) == std::vector<uint64_t>{g.cycle + back + 1},
        "the atomic was not refused for its return slot c" + std::to_string(g.cycle) + " .. c" +
            std::to_string(g.cycle + back) + " and admitted at c" + std::to_string(g.cycle + back + 1) + ":" +
            describe(refusals(*machine, 0, "load slots full")));
    require(machine->trace().summary().shmem_load_slot_waits == back + 1, "the return-slot refusals were not counted");
    const auto atomic = i5_atomic(*machine, "after the refusal");
    i5_require_timeline(atomic, profile, atomic.start, "after the refusal");
    require(machine->read_shmem_u32(1024) == 0x205, "the atomic's word");
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(machine->read_reg(0, lane, 6) == 0x100 + 64 + lane && machine->read_reg(0, lane, 7) == 0x200,
            "lane " + std::to_string(lane) + ": the lds or the atomic returned the wrong word");
    }
}

// rfc13 section 4: the SIMT RMW register holds ONE operation machine-wide.
// Three warps, one atomic each on one word (operand 1): the first runs in the
// engine, the second waits in the register, the third is refused "SIMT RMW
// register busy" until the register empties. All three apply, one after
// another (each reads the previous sum), and the engine runs them back to
// back with no overlap.
void test_i5_simt_rmw_register_busy() {
    auto profile = canonical_profile();
    profile.shmem_load_slots = 3; // a return slot for each: the third waits only for the register
    auto machine = i5_machine(profile, R"(
        mov r1, 0
        mov r5, 1
        atom.add.uni r7, [r1 + 2048], r5 {pred=PT, stall=0, set=sb0, wait=-}
        halt {pred=PT, stall=0, set=-, wait=sb0}
    )", 3);
    machine->run();
    const auto atomics = i5_atomics(*machine);
    require(atomics.size() == 3, "expected three atomics");
    for (std::size_t k = 1; k < atomics.size(); ++k) {
        require(atomics[k].start >= atomics[k - 1].end, "two atomics overlapped in the engine:" + describe(atomics[k]));
    }
    uint32_t busy = 0;
    for (uint32_t warp = 0; warp < 3; ++warp) busy += static_cast<uint32_t>(refusals(*machine, warp, "SIMT RMW register busy").size());
    require(busy != 0, "no warp was refused for the busy SIMT RMW register");
    std::set<uint32_t> old_values;
    for (uint32_t warp = 0; warp < 3; ++warp) old_values.insert(machine->read_reg(warp, 0, 7));
    require(old_values == std::set<uint32_t>{0x300, 0x301, 0x302} && machine->read_shmem_u32(2048) == 0x303,
        "the three atomics did not apply one after another");
}

// rfc13 section 6 with rfc12 section 6: the global store pause covers the
// atomic from its start to its write grant w; stores start again at w + 1;
// reads continue. Warp 0: one atomic. Warp 1: an lds (granted while the
// atomic runs) and an sts on other words, admitted before w and granted
// exactly at w + 1; store_paused counts its wait.
// (Mutation: the pause ends at the atomic's read grant -> the sts goes in
// before w + 1 and this test fails.)
// I6a: every access now reaches its slot or register issue_to_dispatch after
// its issue. The stall on warp 0's mov r5 (11) re-places the atomic: it
// issues at c26 and starts at c30 (w = c36); warp 1's lds (issued c27) is
// granted at c31, during it; its sts, ordered behind the lds, issues at c31
// and is in its slot from c35, paused until w.
// Two-word packets: warp 0's mov r1 (c14) is the second half of its packet,
// so warp 0 asks for a fetch at c15, the cycle warp 1 first asks for its
// branch target's packet; the round-robin serves warp 0 first and warp 1's
// path runs a cycle later (its sread at c18, its lds at c30, granted c34;
// its sts at c34, in its slot from c38). The stall is 14 (was 11): the
// atomic issues at c29 and starts at c33 (w = c39), so the lds is granted
// during it and the sts waits in its slot c38 .. c39, as before.
void test_i5_store_pause_and_resume() {
    const auto& profile = canonical_profile();
    auto machine = i5_machine(profile, R"(
        sread r29, WARP_ID
        setp.ne p6, r29, 0
        bra.uni other {pred=p6, stall=0, set=-, wait=-}
        mov r1, 0
        mov r5, 1 {pred=PT, stall=14, set=-, wait=-}
        atom.add.uni r7, [r1 + 2048], r5 {pred=PT, stall=0, set=sb0, wait=-}
        halt {pred=PT, stall=0, set=-, wait=sb0}
    other: sread r20, LANE_ID
        shli r20, r20, 2
        mov r21, 0x77
        lds r23, [r20 + 256] {pred=PT, stall=0, set=sb1, wait=-}
        sts [r20 + 3072], r21 {pred=PT, stall=0, set=sb2, wait=-}
        halt {pred=PT, stall=0, set=-, wait=sb1|sb2}
    )", 2);
    machine->run();
    const auto atomic = i5_atomic(*machine, "store pause");
    i5_require_timeline(atomic, profile, atomic.start, "store pause");
    std::vector<uint64_t> loads;
    std::vector<uint64_t> stores;
    for (const auto& grant : i2_grants(*machine)) {
        if (grant.client != "simt") continue;
        (grant.service == "A-read" ? loads : stores).push_back(grant.cycle);
    }
    const auto w = atomic.writes.at(0);
    const auto sts_issue = issued_at(*machine, 11 * 8);
    require(loads.size() == 1 && loads[0] > atomic.start && loads[0] < w,
        "warp 1's lds was not granted while the atomic ran (reads continue):" + (describe(i2_grants(*machine)) + " | issues" + issue_log(*machine)) +
            describe(atomic));
    require(sts_issue.size() == 1 && sts_issue[0] < w, "warp 1's sts was not admitted before the atomic's write");
    require(stores == std::vector<uint64_t>{w + 1},
        "the sts was not granted at the atomic's write + 1 (c" + std::to_string(w + 1) + "):" +
            describe(i2_grants(*machine, "A-write")) + describe(atomic));
    require(machine->trace().summary().shmem_store_paused == w - (sts_issue[0] + nanosm::issue_to_dispatch(profile)),
        "store_paused should count the sts's wait in its slot, its dispatch + 1 .. the atomic's write");
    require(machine->read_shmem_u32(2048) == 0x301, "the atomic's word");
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(machine->read_reg(0, lane, 7) == 0x300 && machine->read_reg(1, lane, 23) == 0x100 + 64 + lane &&
                machine->read_shmem_u32(3072 + 4 * lane) == 0x77,
            "lane " + std::to_string(lane) + ": wrong atomic, lds or sts data");
    }
}

// rfc12 section 6 step 2: an atomic that STARTS while another warp's store
// transaction is under way reads strictly after that store's last round
// (matched D then lands the store before the atomic's sample). Warp 1: one
// sts of sixteen rounds (every lane on bank 0); its LAST round writes the
// atomic's word. Warp 0's atomic starts during those rounds; its read is
// granted the cycle after the last round, and it reads the stored word.
// (Mutation: the read granted while the store is under way -> it is early
// and returns the old word.)
// I6a: warp 1's sts now reaches its slot issue_to_dispatch after its issue
// (rounds c32..c47) and the atomic starts issue_to_dispatch + 1 after its
// own; the stall on the last filler mov (14) re-places the atomic's start
// inside those rounds.
void test_i5_read_after_the_store_under_way() {
    const auto& profile = canonical_profile();
    auto machine = i5_machine(profile, R"(
        sread r29, WARP_ID
        setp.ne p6, r29, 0
        bra.uni other {pred=p6, stall=0, set=-, wait=-}
        mov r1, 0
        mov r5, 1
        mov r6, 0
        mov r6, 1
        mov r6, 2 {pred=PT, stall=14, set=-, wait=-}
        atom.add.uni r7, [r1 + 960], r5 {pred=PT, stall=0, set=sb0, wait=-}
        halt {pred=PT, stall=0, set=-, wait=sb0}
    other: sread r20, LANE_ID
        shli r20, r20, 6
        mov r21, 0x77
        sts [r20], r21 {pred=PT, stall=0, set=sb2, wait=-}
        halt {pred=PT, stall=0, set=-, wait=sb2}
    )", 2);
    machine->run();
    const auto atomic = i5_atomic(*machine, "read after the store");
    std::vector<uint64_t> rounds;
    for (const auto& grant : i2_grants(*machine, "A-write")) {
        if (grant.client == "simt") rounds.push_back(grant.cycle);
    }
    require(rounds.size() == 16, "the sts was not sixteen rounds");
    require(atomic.start > rounds.front() && atomic.start <= rounds.back(),
        "the atomic did not start while the sts was under way (the test needs that):" + describe(atomic) +
            " sts rounds c" + std::to_string(rounds.front()) + "..c" + std::to_string(rounds.back()));
    i5_require_timeline(atomic, profile, rounds.back() + 1, "read after the store");
    require(machine->read_reg(0, 0, 7) == 0x77 && machine->read_shmem_u32(960) == 0x78,
        "the atomic did not read the store's word (rD " + std::to_string(machine->read_reg(0, 0, 7)) + ")");
}

// rfc13 section 4 with rfc12 section 4: an atomic is a WRITE for its warp's
// order and counts as unserviced until its write grant.
//   sts B; atom B: the atomic waits for the sts's grant and reads its word;
//   atom B; lds B: the lds waits until the atomic's write grant w (admitted
//     at w) and reads the result in lane 0's word;
//   lds A; atom A: the atomic waits for the lds's grant; the lds reads the
//     old word (and so does the atomic: it applies after);
//   atom A; fence.sh: admitted at the atomic's write grant + D.
// Words B: 384 + l (sixteen banks, one round each); A: 256 + l. Three load
// slots, so atom A's return slot is free while both lds hold theirs: only
// the warp's order decides.
void test_i5_same_warp_order() {
    auto profile = canonical_profile();
    profile.shmem_load_slots = 3;
    auto machine = i5_machine(profile, R"(
        sread r0, LANE_ID
        shli r1, r0, 2
        mov r2, 0
        mov r3, 3
        mov r12, 0x1010
        sts [r1 + 1536], r12 {pred=PT, stall=0, set=sb1, wait=-}
        atom.add.uni r11, [r2 + 1536], r3 {pred=PT, stall=0, set=sb2, wait=-}
        lds r8, [r1 + 1536] {pred=PT, stall=0, set=sb3, wait=-}
        lds r6, [r1 + 1024] {pred=PT, stall=0, set=sb4, wait=-}
        atom.add.uni r7, [r2 + 1024], r3 {pred=PT, stall=0, set=sb0, wait=-}
        fence.sh
        halt {pred=PT, stall=0, set=-, wait=sb0|sb1|sb2|sb3|sb4}
    )");
    machine->run();
    const auto atomics = i5_atomics(*machine);
    require(atomics.size() == 2, "expected two atomics");
    std::vector<I2Grant> simt;
    for (const auto& grant : i2_grants(*machine)) {
        if (grant.client == "simt") simt.push_back(grant);
    }
    // sts B, lds B, lds A, in that order.
    require(simt.size() == 3 && simt[0].service == "A-write" && simt[1].service == "A-read" &&
            simt[2].service == "A-read",
        "the SIMT rounds:" + describe(simt));
    require(issued_at(*machine, 48) == std::vector<uint64_t>{simt[0].cycle} && atomics[0].reads.at(0) > simt[0].cycle,
        "atom B was not admitted at the sts's grant, reading after it");
    const auto w_b = atomics[0].writes.at(0);
    require(issued_at(*machine, 56) == std::vector<uint64_t>{w_b} && simt[1].cycle > w_b,
        "lds B was not admitted at atom B's write grant c" + std::to_string(w_b) + ":" + describe(atomics[0]));
    require(issued_at(*machine, 72) == std::vector<uint64_t>{simt[2].cycle} && atomics[1].reads.at(0) > simt[2].cycle,
        "atom A was not admitted at lds A's grant");
    require(issued_at(*machine, 80) == std::vector<uint64_t>{atomics[1].writes.at(0) + profile.shmem_port_a_depth},
        "fence.sh was not admitted at atom A's write grant + D");
    require(machine->read_shmem_u32(1536) == 0x1013 && machine->read_shmem_u32(1024) == 0x203,
        "the atomics' words");
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(machine->read_reg(0, lane, 11) == 0x1010 && machine->read_reg(0, lane, 7) == 0x200,
            "lane " + std::to_string(lane) + ": an atomic's old value");
        require(machine->read_reg(0, lane, 8) == (lane == 0 ? 0x1013u : 0x1010u) &&
                machine->read_reg(0, lane, 6) == 0x200 + lane,
            "lane " + std::to_string(lane) + ": an lds around the atomics");
    }
}

// Backlog D12 (HUMAN 2026-10-12): a uniform operation takes lane 0's
// operands whatever the active mask -- the mask gates only its destination's
// lane write enables -- and a partial (or empty) mask is UB: the SANITIZER
// PANICs; the machine runs quietly, and what it does then is no contract
// (rfc13 section 2: the asserts below pin today's behavior so that it stays
// deterministic, nothing more). Lane 0 is inactive here (0xfffe), and
// lane l's registers hold l-dependent values, so lane 0's are visible:
//   atom.add.uni [lane 0's r1 + 2048] += lane 0's r5 (1): word 512 becomes
//     0x301; rD lanes 1..15 get 0x300, lane 0 keeps 0xdead;
//   lds.uni r3 from lane 0's address: lanes 1..15 get 0x300;
//   mbar.arrive and mbar.test_wait on a count-1 mbar: the phase completes.
// An EMPTY mask still performs the atomic (word 512 = 0x301) and writes no
// lane of rD.
void test_d12_uniform_ops_under_a_partial_mask() {
    const auto& profile = canonical_profile();
    const std::string program = R"(
        sread r0, LANE_ID
        shli r1, r0, 2
        addi r5, r0, 1
        mov r7, 0xdead
        mov r3, 0xbeef
        mov r4, 4160
        atom.add.uni r7, [r1 + 2048], r5 {pred=PT, stall=0, set=sb0, wait=-}
        lds.uni r3, [r1 + 2048] {pred=PT, stall=0, set=sb1, wait=sb0}
        mbar.arrive [r4]
        mbar.test_wait p0, [r4], 0
        halt {pred=PT, stall=0, set=-, wait=sb0|sb1}
    )";
    const auto mbar = [](nanosm::Launch& launch) { write_count_mbar(launch.shmem_image.at(0).bytes, 4160, 1); };
    for (const uint64_t mask : {uint64_t{0xfffe}, uint64_t{0}}) {
        const auto what = "mask " + std::to_string(mask);
        auto machine = i5_machine(profile, program, 1, mbar);
        machine->test_set_active_mask(0, mask);
#if NANOSM_SANITIZER
        const auto message = i4_panic(*machine, what);
        require_mentions(message, {"atom.add.uni is a uniform operation under a partial active mask"}, what);
#else
        while (!machine->done()) tick_capped(*machine);
        require(machine->read_shmem_u32(2048) == 0x301, what + ": the atomic did not use lane 0's address and operand");
        require(machine->read_mbar_state(4160).phase, what + ": the arrive did not complete the phase");
        for (uint32_t lane = 0; lane < 16; ++lane) {
            const bool active = (mask >> lane & 1) != 0;
            require(machine->read_reg(0, lane, 7) == (active ? 0x300u : 0xdeadu) &&
                    machine->read_reg(0, lane, 3) == (active ? 0x301u : 0xbeefu),
                what + ": lane " + std::to_string(lane) + " rD (the mask gates only the write enables)");
        }
#endif
    }
#if NANOSM_SANITIZER
    for (const std::string uniform :
             {"lds.uni r3, [r1] {pred=PT, stall=0, set=sb0, wait=-}", "mbar.arrive [r2]", "mbar.test_wait p0, [r2], 0"}) {
        auto machine = i5_machine(profile, "mov r1, 0\nmov r2, 4160\n" + uniform + "\nhalt\n", 1, mbar);
        machine->test_set_active_mask(0, 0xfffe);
        require_mentions(i4_panic(*machine, uniform), {"uniform operation under a partial active mask"}, uniform);
    }
#endif
}

// Backlog D12 (HUMAN 2026-10-12, as on NVIDIA): two lanes of one sts naming
// one word is UB -- the SANITIZER PANICs at issue; the machine just writes
// every lane, in round order (each lane on the word is its own bank round),
// so the highest ACTIVE lane's word lands. Lane l stores l + 1 to word 0.
void test_d12_sts_two_lanes_on_one_word() {
    for (const uint64_t mask : {uint64_t{0xffff}, uint64_t{0x00f6}}) {
        nanosm::Launch launch;
        launch.active_warps = 1;
        image_bytes(launch.shmem_image, 4);
        nanosm::Machine machine(canonical_profile(), compile(R"(
            sread r1, LANE_ID
            addi r2, r1, 1
            mov r0, 0
            sts [r0], r2
            halt
        )"), std::move(launch));
        machine.test_set_active_mask(0, mask);
#if NANOSM_SANITIZER
        require_mentions(i4_panic(machine, "same-word sts"), {"multiple lanes write the same shmem word"},
            "same-word sts");
#else
        while (!machine.done()) tick_capped(machine);
        const uint32_t highest = mask == 0xffff ? 15 : 7;
        require(machine.read_shmem_u32(0) == highest + 1,
            "the highest active lane's word did not land (" + std::to_string(machine.read_shmem_u32(0)) + ")");
#endif
    }
}

// Backlog D12, gmem: two lanes of one stg naming one word is UB -- the
// SANITIZER PANICs at issue, before anything is written; the machine sends
// every lane's write, one request per lane in lane order, and with the
// placeholder controller's queue-front order the highest lane's lands last.
void test_d12_stg_two_lanes_on_one_word() {
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
#if NANOSM_SANITIZER
    require_mentions(i4_panic(machine, "same-word stg"), {"multiple lanes write the same gmem word"}, "same-word stg");
    require(machine.read_gmem_u32(0) == 0x12345678, "the PANIC came after a write");
#else
    while (!machine.done()) tick_capped(machine);
    require(machine.read_gmem_u32(0) == 15, "lane 15's word did not land last (" +
        std::to_string(machine.read_gmem_u32(0)) + ")");
#endif
}

// Backlog D12: an sts over an mbar with a live waiter is UB (as on NVIDIA:
// an mbarrier's memory is used for nothing else until mbarrier.inval). The
// machine no longer checks it: the sts just writes the word. The SANITIZER's
// C10 PANICs (the mbar is declared by the launch, so it is live from launch).
// Warp 0 waits on the mbar (one try_wait, then halt); warp 1's sts writes 5
// into words 16 + l (lanes 0 and 1: the mbar's two words). Without the
// sanitizer the waiter's try_wait simply times out (no wake); the run is
// capped.
void test_d12_sts_over_an_mbar_with_a_live_waiter() {
    nanosm::Launch launch;
    launch.active_warps = 2;
    auto& shmem = image_bytes(launch.shmem_image, 128);
    write_count_mbar(shmem, 64, 1);
    launch.mbars = {64};
    nanosm::Machine machine(canonical_profile(), compile(R"(
        mov r1, 64
        sread r0, WARP_ID
        setp.eq p0, r0, 0
        @p0 bra.uni waiter
        sread r3, LANE_ID
        shli r3, r3, 2
        add r3, r3, r1
        mov r2, 5
        sts [r3], r2
        halt
    waiter:
        mbar.try_wait p1, [r1], 0
        halt
    )"), std::move(launch));
#if NANOSM_SANITIZER
    require_mentions(i4_panic(machine, "sts over a live mbar"), {"reason=live mbar opacity: sts writes bytes [64, 72) "
        "of the live mbar [64, 72) (declared by the launch image)"}, "sts over a live mbar");
#else
    while (!machine.done()) tick_capped(machine);
    require(machine.read_shmem_u32(64) == 5 && machine.read_shmem_u32(68) == 5, "the sts did not write the mbar's words");
#endif
}

// Backlog D12.1 (HUMAN 2026-10-12): EVERY uniform operation -- each
// instruction with a lane-0-sourced operand, and bra.uni in every form -- is
// UB under a partial (or empty) active mask: the SANITIZER PANICs naming it;
// the machine reads lane 0 and runs quietly (what it does then is no
// contract; a program with a partial mask uses the per-lane instruction).
// One program per uniform opcode family, each of which runs to completion
// under a full mask, run under 0xfffe (lane 0 inactive): the named opcode is
// the program's first uniform operation.
void test_d12_1_every_uniform_op_under_a_partial_mask() {
    const auto& profile = canonical_profile();
    const auto setup = [](nanosm::Launch& launch) {
        auto& shmem = image_bytes(launch.shmem_image, 8192);
        write_count_mbar(shmem, 4096, 1);
        write_tma_descriptor(shmem, 5120, linear_tma(false, 1024, 64, 4104));
        write_tx_mbar(shmem, 4104, 64);
        image_bytes(launch.gmem_image, 4096);
    };
    const std::string mma_tail = "mov r4, 4160\nmbar.init [r4], 1\nmma.commit [r4]\n"
                                 "wait:\nmbar.try_wait p0, [r4], 0\n@!p0 bra.uni wait\nhalt\n";
    const std::vector<std::pair<std::string, std::string>> cases = {
        {"lds.uni", "mov r0, 0\nlds.uni r1, [r0]\nhalt\n"},
        {"ldg.uni", "mov r0, 0\nldg.uni r1, [r0]\nhalt\n"},
        {"ldg.v8", "mov r0, 0\nldg.v8 r8, [r0]\nhalt\n"},
        {"stg.v8", "mov r0, 0\nstg.v8 [r0], r8\nhalt\n"},
        {"mbar.init", "mov r4, 4160\nmbar.init [r4], 1\nhalt\n"},
        {"mbar.arrive", "mov r4, 4096\nmbar.arrive [r4]\nhalt\n"},
        {"mbar.arrive.expect_tx", "mov r4, 4096\nmbar.arrive.expect_tx [r4], 0\nhalt\n"},
        {"mbar.test_wait", "mov r4, 4096\nmbar.test_wait p0, [r4], 0\nhalt\n"},
        {"mbar.try_wait", "mov r4, 4096\nmbar.try_wait p0, [r4], 0\nhalt\n"},
        {"atom.add.uni", "mov r0, 0\nmov r1, 1\natom.add.uni r2, [r0], r1\nhalt\n"},
        {"tma.async", "mov r8, 0\nmov r9, 0\nmov r10, 0\nmov r4, 4104\ntma.async [5120], r8\n"
                      "wait:\nmbar.try_wait p0, [r4], 0\n@!p0 bra.uni wait\nhalt\n"},
        {"mma.async.init", ".acc C, 0, 0, 64, 64\nmov r0, 0\nmov r1, 1024\nmma.async.init C, [r0], [r1]\n" + mma_tail},
        {"mma.async", ".acc C, 0, 0, 64, 64\nmov r0, 0\nmov r1, 1024\nmma.async C, [r0], [r1]\n" + mma_tail},
        {"mma.commit", "mov r4, 4096\nmma.commit [r4]\nhalt\n"},
        {"acc.ld", ".acc C, 0, 0, 64, 64\nmov r8, 0\nmov r9, 0\nacc.ld r10, C, r8\nhalt\n"},
        {"acc.st.v8", ".acc C, 0, 0, 64, 64\nmov r8, 0\nmov r9, 0\nmov r0, 1\nacc.st.v8 C, r8, r0\nhalt\n"},
        {"bra.uni", "sread r0, WARP_ID\nsetp.eq p0, r0, 0\n@p0 bra.uni out\nmov r1, 1\nout:\nhalt\n"},
        {"bra.uni", "bra.uni out\nmov r1, 1\nout:\nhalt\n"}, // @PT: no operand, still uniform
    };
    for (const auto& [name, program] : cases) {
        for (const uint64_t mask : {uint64_t{0xffff}, uint64_t{0xfffe}}) {
            const auto what = name + " under mask " + std::to_string(mask) + ": " + program;
            nanosm::Launch launch;
            launch.active_warps = 1;
            setup(launch);
            nanosm::Machine machine(profile, compile(program, profile), std::move(launch));
            machine.test_set_active_mask(0, mask);
#if NANOSM_SANITIZER
            if (mask != 0xffff) {
                require_mentions(i4_panic(machine, what), {name + " is a uniform operation under a partial active mask"},
                    what);
                continue;
            }
#endif
            try {
                while (!machine.done()) tick_capped(machine);
            } catch (const nanosm::SimError& error) {
                throw std::runtime_error(what + ": the machine panicked: " + error.what());
            }
        }
    }
}

// Backlog D12.1: an mbar.init over an mbar with a live waiter is UB (as on
// NVIDIA: an mbarrier is reinitialized only after mbarrier.inval). The
// machine no longer checks it: the init just writes the new state and wakes
// nobody. The SANITIZER's C10 PANICs (the mbar is declared by the launch, so
// it is live from launch). Warp 0 parks in one try_wait on the count-1 mbar
// (nobody arrives); warp 1 re-initializes it with count 3 while warp 0 is
// parked. Without the sanitizer warp 0's try_wait times out (false: r9 stays
// 0) and both halt; the run is capped.
void test_d12_1_mbar_init_over_a_live_waiter() {
    nanosm::Launch launch;
    launch.active_warps = 2;
    auto& shmem = image_bytes(launch.shmem_image, 128);
    write_count_mbar(shmem, 64, 1);
    launch.mbars = {64};
    nanosm::Machine machine(canonical_profile(), compile(R"(
        mov r1, 64
        sread r0, WARP_ID
        setp.eq p0, r0, 0
        @p0 bra.uni waiter
        mbar.init [r1], 3
        halt
    waiter:
        mov r9, 0
        mbar.try_wait p1, [r1], 0
        @p1 bra.uni woken
        halt
    woken:
        mov r9, 7
        halt
    )"), std::move(launch));
#if NANOSM_SANITIZER
    require_mentions(i4_panic(machine, "mbar.init over a live waiter"),
        {"reason=mbar.init over the live mbar [64, 72) (declared by the launch image)", "inst='mbar.init [r1], 3"},
        "mbar.init over a live waiter");
#else
    while (!machine.done()) tick_capped(machine);
    const auto state = machine.read_mbar_state(64);
    require(state.valid && state.expected == 3 && state.remaining == 3 && !state.phase,
        "the mbar.init did not write its state over the waited-on mbar");
    require(machine.read_reg(0, 0, 9) == 0, "the waiter was woken; a re-initialization wakes nobody");
#endif
}

// Backlog D12.1: a TMA load placing data over an mbar with a live waiter is
// UB (as on NVIDIA: an mbarrier's memory serves nothing else until
// mbarrier.inval). The machine no longer checks it: the placement just writes
// the bytes. The SANITIZER's C10 PANICs at the placement write (the mbar at
// 1032 is declared by the launch). Warp 0 spins in try_wait on that count-1
// mbar (nobody arrives); warp 1, after a delay that lets warp 0's first probe
// return and park, loads 64 bytes from gmem 0 over shmem [1024, 1088). The
// gmem bytes that land on the mbar are the same mbar with its phase flipped
// (only bit 0 differs, so any probe sees a valid mbar): without the
// sanitizer warp 0's next probe sees phase 1 and leaves, and warp 1 waits
// for the TMA's own mbar at 4104.
void test_d12_1_tma_load_over_a_live_waiter() {
    nanosm::Launch launch;
    launch.active_warps = 2;
    auto& shmem = image_bytes(launch.shmem_image, 8192);
    write_count_mbar(shmem, 1032, 1);
    write_tma_descriptor(shmem, 5120, linear_tma(false, 1024, 64, 4104));
    write_tx_mbar(shmem, 4104, 64);
    launch.mbars = {1032};
    auto& gmem = image_bytes(launch.gmem_image, 64);
    for (uint32_t word = 0; word < 16; ++word) write_u32(gmem, 4 * word, 0x500 + word);
    nanosm::MbarState flipped;
    flipped.phase = true;
    flipped.valid = true;
    flipped.expected = 1;
    flipped.remaining = 1;
    write_u64(gmem, 8, nanosm::encode_mbar_state(flipped));
    nanosm::Machine machine(canonical_profile(), compile(R"(
        sread r0, WARP_ID
        setp.eq p0, r0, 0
        @p0 bra.uni waiter
        mov r20, 0
        addi r20, r20, 1
        addi r20, r20, 1
        addi r20, r20, 1
        addi r20, r20, 1
        addi r20, r20, 1
        addi r20, r20, 1
        addi r20, r20, 1
        addi r20, r20, 1
        mov r8, 0
        mov r9, 0
        mov r10, 0
        mov r4, 4104
        tma.async [5120], r8
    loaded:
        mbar.try_wait p1, [r4], 0
        @!p1 bra.uni loaded
        halt
    waiter:
        mov r1, 1032
    spin:
        mbar.try_wait p2, [r1], 0
        @!p2 bra.uni spin
        halt
    )"), std::move(launch));
#if NANOSM_SANITIZER
    require_mentions(i4_panic(machine, "TMA load over a live waiter"),
        {"reason=live mbar opacity: tma load #0 placement writes bytes [1032, 1040) of the live mbar [1032, 1040) "
         "(declared by the launch image)"},
        "TMA load over a live waiter");
#else
    while (!machine.done()) tick_capped(machine);
    const auto state = machine.read_mbar_state(1032);
    require(state.valid && state.phase && state.expected == 1 && state.remaining == 1,
        "the TMA did not write its bytes over the waited-on mbar");
    require(machine.read_shmem_u32(1024) == 0x500 && machine.read_shmem_u32(1084) == 0x50f,
        "the TMA's other bytes did not land");
#endif
}

// SANITIZER (rfc09 pin 3, rfc13 section 2): an atomic's address and operand
// are lane 0's in hardware; one that differs across lanes PANICs.
void test_i5_atomic_nonuniform_operands() {
    const auto& profile = canonical_profile();
    for (const auto& [program, needle] : {
             std::pair<std::string, std::string>{"sread r0, LANE_ID\nshli r1, r0, 2\nmov r5, 1\n"
                 "atom.add.uni r7, [r1], r5\nhalt\n", "atom base address is not warp-uniform"},
             {"sread r0, LANE_ID\nmov r1, 0\natom.add.uni r7, [r1], r0\nhalt\n", "atom operand is not warp-uniform"},
         }) {
        require_mentions(i4_panic(*i5_machine(profile, program, 1, {}, false), needle), {needle}, needle);
    }
}

// SANITIZER (rfc13 section 7, C10): an atomic writing a LIVE mbar PANICs at
// its write grant (using the RMW engine does not authorize it) -- an mbar
// made live by a SIMT mbar.init, and one the launch declared (live from
// launch). The launch-declared mbar is opaque to an sts and to an mbar.init
// as well, and stays usable by mbar operations. The atomic's other sanitizer
// checks: issued over a TMA load's destination it is a race; its word still
// travelling under a TMA store of it, without a fence.sh, a missing fence
// (with one: clean, and gmem gets its result).
void test_i5_atomic_sanitizer() {
    const auto& profile = canonical_profile();
    auto message = i4_panic(*i5_machine(profile, "mov r4, 4160\nmbar.init [r4], 1\nmov r21, 9\n"
        "atom.add.uni r7, [r4], r21\nhalt\n", 1, {}, false), "an atomic on a SIMT-initialized mbar");
    require_mentions(message, {"mbar address=4160 reason=live mbar opacity: atom.add.uni r7, [r4], r21 (warp 0 pc ",
        "writes bytes [4160, 4164) of the live mbar [4160, 4168) (mbar.init by warp 0 pc 8 at cycle ",
        "until kernel end only mbar operations may touch it"}, "an atomic on a SIMT-initialized mbar");

    const auto declared = [](nanosm::Launch& launch) {
        write_count_mbar(launch.shmem_image.at(0).bytes, 4160, 1);
        launch.mbars = {4160};
    };
    message = i4_panic(*i5_machine(profile, "mov r4, 4164\nmov r21, 9\natom.exch.uni r7, [r4], r21\nhalt\n", 1,
        declared, false), "an atomic on a launch-declared mbar");
    require_mentions(message, {"mbar address=4160 reason=live mbar opacity: atom.exch.uni r7, [r4], r21 (warp 0 pc ",
        "writes bytes [4164, 4168) of the live mbar [4160, 4168) (declared by the launch image)"},
        "an atomic on a launch-declared mbar");
    message = i4_panic(*i5_machine(profile, "mov r4, 4160\nsread r20, LANE_ID\nshli r20, r20, 2\nadd r20, r20, r4\n"
        "mov r21, 9\nsts [r20], r21\nhalt\n", 1, declared, false), "an sts on a launch-declared mbar");
    require_mentions(message, {"reason=live mbar opacity: sts writes bytes [4160, 4168) of the live mbar [4160, 4168) "
        "(declared by the launch image)"}, "an sts on a launch-declared mbar");
    message = i4_panic(*i5_machine(profile, "mov r4, 4160\nmbar.init [r4], 1\nhalt\n", 1, declared, false),
        "an mbar.init over a launch-declared mbar");
    require_mentions(message, {"reason=mbar.init over the live mbar [4160, 4168) (declared by the launch image)"},
        "an mbar.init over a launch-declared mbar");
    auto legal = i5_machine(profile, "mov r4, 4160\nmbar.arrive [r4]\nwait:\nmbar.try_wait p0, [r4], 0\n"
        "@!p0 bra.uni wait\nhalt\n", 1, declared, false);
    legal->run();
    require(legal->read_mbar_state(4160).phase, "an arrive on a launch-declared mbar did not complete its phase");

    const auto tma = [](bool store) {
        return [store](nanosm::Launch& launch) {
            auto& shmem = launch.shmem_image.at(0).bytes;
            write_tma_descriptor(shmem, 5120, linear_tma(store, store ? 0 : 1024, 64, 4104));
            write_tx_mbar(shmem, 4104, 64);
        };
    };
    const std::string tail = R"(
    wait:
        mbar.try_wait p0, [r4], 0
        @!p0 bra.uni wait
        halt
    )";
    const std::string prologue = "mov r8, 0\nmov r9, 0\nmov r10, 0\nmov r4, 4104\n";
    message = i4_panic(*i5_machine(profile, prologue + "tma.async [5120], r8\nmov r20, 1032\nmov r21, 1\n"
        "atom.add.uni r7, [r20], r21\n" + tail, 1, tma(false), false), "an atomic over a TMA load's destination");
    require_mentions(message, {"reason=shmem race: atom.add.uni writes 1 word, bytes [1032, 1036) owned by tma load #0"},
        "an atomic over a TMA load's destination");
    const std::string atomic = "mov r20, 8\nmov r21, 0x1000\natom.add.uni r7, [r20], r21\n";
    message = i4_panic(*i5_machine(profile, prologue + atomic + "tma.async [5120], r8\n" + tail, 1, tma(true), false),
        "a TMA store of an atomic's word without fence.sh");
    require_mentions(message, {"reason=missing fence.sh: tma store #0 source bytes [", "overlap this warp's earlier "
        "atom.add.uni r7, [r20], r21 (pc ", "whose write has not reached the RAM; issue fence.sh between them"},
        "a TMA store of an atomic's word without fence.sh");
    auto fenced = i5_machine(profile, prologue + atomic + "fence.sh\ntma.async [5120], r8\n" + tail, 1, tma(true), false);
    fenced->run();
    for (uint32_t i = 0; i < 16; ++i) {
        require(fenced->read_gmem_u32(4 * i) == (i == 2 ? 0x1102u : 0x100u + i),
            "the fenced TMA store did not carry the atomic's result");
    }
}

// C10 follow-up (both builds): a launch DECLARES the mbars the host
// initialized in its shmem image (Launch::mbars; the Python harness's
// shmem.mbar() declares, so FA's are declared). The launch formats carry the
// list ("mbars", absent: none, emitted only when there is one); the machine
// rejects a declaration that is not an 8-byte aligned shmem slot. C11 / C2:
// an "mbars" entry or "active_warps" that is not an unsigned 32-bit integer
// is rejected before it is narrowed (4294967296 is not address 0, 4160.9 is
// not 4160, -8 is not 0xfffffff8).
void test_i5_launch_declared_mbars_format() {
    nanosm::LaunchImage input;
    input.kernel = compile("halt");
    input.launch.active_warps = 1;
    input.launch.shmem_image.push_back({4160, std::vector<uint8_t>(8, 0)});
    const auto plain = nanosm::emit_launch_image(input);
    require(plain.find("mbars") == std::string::npos, "a launch with no declared mbar emitted an mbars field");
    input.launch.mbars = {4160, 4168};
    const auto declared = nanosm::emit_launch_image(input);
    const auto loaded = nanosm::load_launch_image(declared);
    require(loaded.launch.mbars == std::vector<uint32_t>{4160, 4168} && nanosm::emit_launch_image(loaded) == declared,
        "the launch image lost its declared mbars");
    const auto state_text = nanosm::emit_launch_state(input.launch);
    const auto state = nanosm::load_launch_state(state_text);
    require(state.mbars == input.launch.mbars, "the launch state lost its declared mbars");
    require(nanosm::load_launch_state(nanosm::emit_launch_state(nanosm::Launch{1, {}, {}, {}})).mbars.empty(),
        "an absent mbars field is not an empty list");
    for (const uint32_t bad : {4161u, 65536u}) {
        auto launch = input.launch;
        launch.mbars = {bad};
        try {
            nanosm::Machine machine(canonical_profile(), compile("halt"), std::move(launch));
            throw std::runtime_error("a declared mbar at " + std::to_string(bad) + " was accepted");
        } catch (const nanosm::SimError& error) {
            require(std::string(error.what()).find("launch declares an mbar at " + std::to_string(bad)) != std::string::npos,
                error.what());
        }
    }
    const auto rejected = [](const std::function<std::string()>& load, const std::string& needle, const std::string& what) {
        try {
            (void)load();
        } catch (const nanosm::SimError& error) {
            require(std::string(error.what()).find(needle) != std::string::npos, what + ": " + error.what());
            return;
        }
        throw std::runtime_error(what + " was accepted");
    };
    for (const auto& bad : {nlohmann::json(4294967296ull), nlohmann::json(4160.9), nlohmann::json(-8), nlohmann::json("64")}) {
        for (const bool image : {false, true}) {
            auto document = nlohmann::json::parse(image ? declared : state_text);
            document["mbars"] = {4160, bad};
            const auto text = document.dump();
            rejected([&] { return image ? nanosm::emit_launch_image(nanosm::load_launch_image(text))
                                        : nanosm::emit_launch_state(nanosm::load_launch_state(text)); },
                "launch: mbars[1] is not a shmem address", "mbars entry " + bad.dump());
        }
    }
    for (const auto& bad : {nlohmann::json(4294967297ull), nlohmann::json(1.5), nlohmann::json(-1)}) {
        for (const bool image : {false, true}) {
            auto document = nlohmann::json::parse(image ? declared : state_text);
            document["active_warps"] = bad;
            const auto text = document.dump();
            rejected([&] { return image ? nanosm::emit_launch_image(nanosm::load_launch_image(text))
                                        : nanosm::emit_launch_state(nanosm::load_launch_state(text)); },
                std::string(image ? "launch image" : "launch state") + ": active warp count is not an unsigned 32-bit integer",
                "active_warps " + bad.dump());
        }
    }
}

// ---------------------------------------------------------------------------
// I6a: the RF's read latency Lr and write depth Wd, the issue register, and
// src/rf.hpp as the ONE timing helper of qtxc, the simulator and the
// sanitizer (trash/rfc10.md section 5, trash/i6a-rf-pipeline-2026-10-13.md).
// Port A's edge convention: a read SAMPLED at S has its operand at S + Lr; a
// write PRESENTED at the RF write port at W is at the RAM at W + Wd, lands at
// that edge and is visible to samples from W + Wd + 1 -- no forwarding.
// The debug trace shows the samples ("RF read") and the presentations ("RF
// write"); the landing is the cycle whose tick changed the register.
// ---------------------------------------------------------------------------

nanosm::DeviceProfile i6a_profile(uint32_t lr, uint32_t wd) {
    auto profile = canonical_profile();
    profile.rf_read_latency_cycles = lr;
    profile.rf_write_depth_cycles = wd;
    nanosm::validate_device_profile(profile);
    return profile;
}

const std::vector<std::pair<uint32_t, uint32_t>> i6a_depths = {{1, 1}, {2, 2}, {3, 1}};

std::string i6a_what(const nanosm::DeviceProfile& profile) {
    return "Lr/Wd " + std::to_string(profile.rf_read_latency_cycles) + "/" +
        std::to_string(profile.rf_write_depth_cycles) + ": ";
}

// I2-style shmem (word w holds 0x100 + w), a debug trace.
std::unique_ptr<nanosm::Machine> i6a_machine(
    const nanosm::DeviceProfile& profile, const nanosm::Kernel& kernel, uint32_t warps = 1) {
    nanosm::Launch launch;
    launch.active_warps = warps;
    auto& shmem = image_bytes(launch.shmem_image, 8192);
    for (uint32_t word = 0; word < 2048; ++word) write_u32(shmem, 4 * word, 0x100 + word);
    return std::make_unique<nanosm::Machine>(profile, kernel, std::move(launch),
        nanosm::TraceOptions{.timeline = true, .debug = true});
}

// Warp 0's debug RF events of one register (its warp track): "RF read"
// samples or "RF write" port presentations, by cycle.
std::vector<uint64_t> i6a_rf_events(const nanosm::Machine& machine, const std::string& name, uint8_t reg) {
    std::vector<uint64_t> cycles;
    for (const auto& event : machine.trace().events()) {
        if (event.category == nanosm::TraceCategory::instruction && event.name == name && event.id == reg &&
            event.track == 100) {
            cycles.push_back(event.start_cycle);
        }
    }
    return cycles;
}

// The helper contract: for an add, an fadd, an SFU op, an lds, an sts and
// an acc.ld, the cycles the trace shows -- the samples, the port, the RAM
// (the register's landing edge), the visibility (a depbar waiter's issue) --
// are the helper's, relative to each instruction's accepted issue i, at
// Lr/Wd 1/1, 2/2 and 3/1 (t = issue_to_dispatch):
//   fixed:  samples i + issue_to_sample, port i + issue_to_port, RAM port +
//           Wd, visible i + issue_to_visible = RAM + 1;
//   lds:    address sampled at i + 1; A-read grant i + t + 1; the words at
//           port A's output D + Lb later; the RF port the cycle after; RAM
//           + Wd; its depbar clears at the visibility, when the waiting add
//           issues (the counter is 1 through the RAM cycle, 0 from RAM + 1);
//   sts:    address and data sampled at i + 1; A-write grant i + t + 1, the
//           word at the shmem RAM D later;
//   acc.ld: coordinates sampled at i + 1; tmem read (beat 0) at i + t + 1;
//           RF port i + t + 2; RAM + Wd; the waiting add at RAM + 1.
// Every RF write's visibility is at least earliest_rf_visible after its
// issue. The shli's stall lets every fixed result reach the RAM before the
// async section, and the acc.ld goes before the lds (its waiter first), so
// no write port or issue slot is contested: the uncontended pipeline.
void test_i6a_helper_contract() {
    for (const auto& [lr, wd] : i6a_depths) {
        const auto profile = i6a_profile(lr, wd);
        const auto what = i6a_what(profile);
        const auto kernel = hand_assemble(i1_asm(profile, R"(
            .acc C, 0, 0, 64, 64
            mov r1, 7
            mov r2, 0x3f8
            shli r2, r2, 20
            mov r11, 77
            mov r8, 0
            mov r9, 0
            sread r0, LANE_ID
            add r3, r0, r1
            fadd r4, r2, r2
            ex2 r5, r2
            shli r6, r0, 2 {pred=PT, stall=16, set=-, wait=-}
            acc.ld r11, C, r8 {pred=PT, stall=0, set=sb2, wait=-}
            lds r7, [r6 + 256] {pred=PT, stall=0, set=sb0, wait=-}
            sts [r6 + 512], r4 {pred=PT, stall=0, set=sb1, wait=-}
            add r14, r11, r11 {pred=PT, stall=0, set=-, wait=sb2}
            add r12, r7, r7 {pred=PT, stall=0, set=-, wait=sb0}
            halt {pred=PT, stall=0, set=-, wait=sb1}
        )"), profile);
        auto machine = i6a_machine(profile, kernel);
        std::optional<uint64_t> stored; // the cycle whose edge put the sts's lane-0 word in the RAM
        std::map<uint64_t, uint32_t> sb0;  // depbar sb0 (Q) during each cycle
        const auto landed = i1_run(*machine, {3, 4, 5, 7, 11}, [&](uint64_t cycle) {
            if (!stored.has_value() && machine->read_shmem_u32(512) == nanosm::float_bits(2.0f)) stored = cycle;
            sb0[cycle + 1] = machine->depbar_value(0, 0);
        });
        const auto issue = [&](uint32_t index) {
            const auto cycles = issued_at(*machine, index * 8);
            require(cycles.size() == 1, what + "instruction " + std::to_string(index) + " did not issue once");
            return cycles[0];
        };
        const auto inst = [&](uint32_t index) { return nanosm::decode(kernel.words.at(index)); };
        const auto t = nanosm::issue_to_dispatch(profile);
        const auto floor = nanosm::earliest_rf_visible(profile);
        const auto log = " | issues" + issue_log(*machine);
        const auto sampled_once = [&](uint8_t reg, uint64_t cycle) {
            const auto samples = i6a_rf_events(*machine, "RF read", reg);
            return std::count(samples.begin(), samples.end(), cycle) == 1;
        };

        // Fixed results: the add (sources in banks 0 and 1), the fadd, the ex2.
        for (const auto& [index, dst, sources] : std::vector<std::tuple<uint32_t, uint8_t, std::vector<uint8_t>>>{
                 {7, 3, {0, 1}}, {8, 4, {2}}, {9, 5, {2}}}) {
            const auto i = issue(index);
            const auto op = inst(index);
            const auto name = what + nanosm::opcode_name(op.opcode) + ": ";
            for (const auto reg : nanosm::rf_read_round(sources)) {
                require(sampled_once(reg, i + nanosm::issue_to_sample),
                    name + "r" + std::to_string(reg) + " was not sampled at issue + issue_to_sample" + log);
            }
            const auto port = i + nanosm::issue_to_port(op, profile);
            require(i6a_rf_events(*machine, "RF write", dst) == std::vector<uint64_t>{port},
                name + "not presented at the RF write port at issue + issue_to_port" + log);
            require(landed.at(dst) == std::vector<uint64_t>{port + wd}, name + "did not land at port + Wd");
            require(nanosm::issue_to_visible(op, profile) == nanosm::issue_to_port(op, profile) + wd + 1 &&
                    nanosm::issue_to_visible(op, profile) >= floor,
                name + "visibility is not the landing + 1, or under the floor");
        }

        // lds r7, [r6 + 256]: one round; its depbar waiter is the add r12.
        {
            const auto i = issue(12);
            const auto name = what + "lds: ";
            require(sampled_once(6, i + nanosm::issue_register_stages),
                name + "the address was not sampled after the issue register");
            const auto reads = i2_grants(*machine, "A-read");
            require(reads.size() == 1 && reads[0].cycle == i + t + 1, name + "A-read grant:" + describe(reads));
            const auto port = reads[0].cycle + i2_return(profile) + 1;
            require(i6a_rf_events(*machine, "RF write", 7) == std::vector<uint64_t>{port},
                name + "not presented at the RF port the cycle after its words" + log);
            require(landed.at(7) == std::vector<uint64_t>{port + wd}, name + "did not land at port + Wd");
            require(sb0.at(port + wd) == 1 && sb0.at(port + wd + 1) == 0,
                name + "the depbar did not clear at the visibility (W + Wd + 1)");
            require(issue(15) == port + wd + 1, name + "the waiting add did not issue at the visibility" + log);
            require(port + wd + 1 - i >= floor, name + "visible before the floor");
        }

        // sts [r6 + 512], r4: one round.
        {
            const auto i = issue(13);
            const auto name = what + "sts: ";
            require(sampled_once(4, i + nanosm::issue_register_stages) && sampled_once(6, i + nanosm::issue_register_stages),
                name + "the address and data were not sampled after the issue register" + log);
            const auto writes = i2_grants(*machine, "A-write");
            require(writes.size() == 1 && writes[0].cycle == i + t + 1, name + "A-write grant:" + describe(writes));
            require(stored == writes[0].cycle + profile.shmem_port_a_depth, name + "the word did not reach the RAM D after the grant");
        }

        // acc.ld r11, C, r8: one beat; its depbar waiter is the add r14.
        {
            const auto i = issue(11);
            const auto name = what + "acc.ld: ";
            require(sampled_once(8, i + nanosm::issue_register_stages) && sampled_once(9, i + nanosm::issue_register_stages),
                name + "the coordinates were not sampled after the issue register");
            std::vector<uint64_t> beats;
            for (const auto& event : machine->trace().events()) {
                if (event.name == "acc.ld beat") beats.push_back(event.start_cycle);
            }
            require(beats == std::vector<uint64_t>{i + t + 1}, name + "the tmem read was not at the dispatch + 1");
            const auto port = i + t + 2;
            require(i6a_rf_events(*machine, "RF write", 11).back() == port, // after the mov's 77
                name + "not presented at the RF port after the pipeline stage" + log);
            require(landed.at(11).size() == 2 && landed.at(11).back() == port + wd,
                name + "did not land at port + Wd (after the mov's 77)");
            require(issue(14) == port + wd + 1, name + "the waiting add did not issue at the visibility" + log);
            require(port + wd + 1 - i >= floor, name + "visible before the floor");
        }
        require(machine->read_reg(0, 5, 3) == 12 && machine->read_reg(0, 5, 7) == 0x100 + 64 + 5 &&
                machine->read_reg(0, 5, 11) == 0 && machine->read_reg(0, 0, 5) == nanosm::float_bits(2.0f),
            what + "data");
    }
}

// The qtxc distance for a dependent add -> add is issue_to_visible -
// issue_register_stages (the consumer samples one cycle after its issue);
// hand assembly with exactly that stall runs and reads the new value. One
// cycle less puts the consumer's sample at the producer's RAM cycle W + Wd:
// the sanitizer PANICs (RF RAW); without it the machine is quiet and the
// sample sees the OLD word -- the write lands at that edge, after every read
// of the cycle (READ_FIRST, no forwarding).
void test_i6a_dependent_add_distance() {
    for (const auto& [lr, wd] : i6a_depths) {
        const auto profile = i6a_profile(lr, wd);
        const auto what = i6a_what(profile);
        const auto scheduled = nanosm::compile_qtx("add r1, r0, r0\nadd r2, r1, r1\nhalt\n", profile);
        const auto distance = nanosm::issue_to_visible(scheduled.instructions[0], profile) - nanosm::issue_register_stages;
        require(scheduled.instructions[0].control.stall == distance, what + "qtxc's stall is not the helper's distance");
        const auto run = [&](uint64_t stall) {
            const auto text = i1_asm(profile, "sread r0, LANE_ID {pred=PT, stall=" +
                std::to_string(alu_distance(profile)) + ", set=-, wait=-}\nadd r1, r0, r0 {pred=PT, stall=" +
                std::to_string(stall) + ", set=-, wait=-}\nadd r2, r1, r1\nhalt\n");
            return i6a_machine(profile, nanosm::assemble_asm(text, profile));
        };
        auto exact = run(distance);
        exact->run();
        require(issued_at(*exact, 16).at(0) - issued_at(*exact, 8).at(0) == distance, what + "the consumer moved");
        require(exact->read_reg(0, 5, 2) == 20, what + "at the qtxc distance the add did not read the new r1");
        auto early = run(distance - 1);
#if NANOSM_SANITIZER
        require_mentions(i4_panic(*early, what + "one cycle early"),
            {"RF RAW hazard: missing stall or depbar wait for r1", "inst='add r2, r1, r1"}, what + "one cycle early");
#else
        early->run();
        require(early->read_reg(0, 5, 1) == 10 && early->read_reg(0, 5, 2) == 0,
            what + "one cycle early, the sample at the landing edge did not see the OLD r1 (write-first forwarding?)");
#endif
    }
}

// WAW: an add after an fadd to the same register must land after it. qtxc
// stalls the fadd fadd - alu + 1 cycles (the younger's landing one past the
// older's); then r3 holds the add's value. Two cycles less the add lands
// first: the sanitizer PANICs (RF WAW); without it the older fadd's value
// lands last and stays. (One cycle less the two results meet at one write
// port in one cycle and the add is refused for the port: structurally
// safe.)
void test_i6a_waw_order() {
    for (const auto& [lr, wd] : i6a_depths) {
        const auto profile = i6a_profile(lr, wd);
        const auto what = i6a_what(profile);
        const auto scheduled = nanosm::compile_qtx("fadd r3, r1, r1\nadd r3, r2, r2\nhalt\n", profile);
        const uint64_t stall = profile.timing.fadd - profile.timing.alu + 1;
        require(scheduled.instructions[0].control.stall == stall, what + "qtxc's WAW stall");
        // r1 = 1.0f (0x3f8 << 20), r2 = 3; each mov / shli stalls its full
        // distance (assembled as written: the stalls under test are not raised).
        const auto run = [&](uint64_t fadd_stall) {
            const auto distance = std::to_string(alu_distance(profile));
            const auto text = i1_asm(profile, "mov r1, 0x3f8 {pred=PT, stall=" + distance + ", set=-, wait=-}\n"
                "shli r1, r1, 20\nmov r2, 3 {pred=PT, stall=" + distance + ", set=-, wait=-}\n"
                "fadd r3, r1, r1 {pred=PT, stall=" + std::to_string(fadd_stall) + ", set=-, wait=-}\n"
                "add r3, r2, r2\nhalt\n");
            return i6a_machine(profile, nanosm::assemble_asm(text, profile));
        };
        auto ordered = run(stall);
        ordered->run();
        require(ordered->read_reg(0, 0, 3) == 6, what + "the younger add's value did not land last");
        auto early = run(stall - 2);
#if NANOSM_SANITIZER
        require_mentions(i4_panic(*early, what + "WAW"), {"RF WAW hazard: missing stall or depbar wait for r3"},
            what + "WAW");
#else
        early->run();
        require(early->read_reg(0, 0, 3) == nanosm::float_bits(2.0f), what + "the add did not land before the fadd");
#endif
    }
}

// An async writer against a fixed booking: an lds.uni's result reaches the
// RF write port at P (issue + i1_lds_shift + 2) in bank 0, where an add
// accepted earlier (at P - issue_to_port) booked bank 0's one port. The
// booking wins; the lds's write is denied (one wait), granted at P + 1 and
// lands Wd later; the add's lands at P + Wd. (I6b-2: the lds.uni issues at
// fetch_to_issue; its stall places the add.)
void test_i6a_async_writer_vs_fixed_booking() {
    for (const auto& [lr, wd] : i6a_depths) {
        const auto profile = i6a_profile(lr, wd);
        const auto what = i6a_what(profile);
        nanosm::Instruction add;
        add.opcode = nanosm::Opcode::add;
        const auto lds_issue = nanosm::fetch_to_issue;
        const auto p = lds_issue + i1_lds_shift(profile) + 2;
        const auto add_issue = p - nanosm::issue_to_port(add, profile);
        auto machine = i6a_machine(profile, hand_assemble(i1_asm(profile,
            "lds.uni r3, [r1] {pred=PT, stall=" + std::to_string(add_issue - lds_issue) + ", set=sb0, wait=-}\n"
            "add r6, r1, r1\n"
            "halt {pred=PT, stall=0, set=-, wait=sb0}\n"), profile));
        const auto landed = i1_run(*machine, {3});
        require(issued_at(*machine, 8) == std::vector<uint64_t>{add_issue}, what + "the add moved" + issue_log(*machine));
        require(i6a_rf_events(*machine, "RF write", 6) == std::vector<uint64_t>{p} &&
                i6a_rf_events(*machine, "RF write", 3) == std::vector<uint64_t>{p + 1},
            what + "the booked add did not keep the port at P, or the lds was not written at P + 1");
        require(landed.at(3) == std::vector<uint64_t>{p + 1 + wd} && machine->trace().summary().rf_async_waits == 1,
            what + "the lds's write did not land at P + 1 + Wd after one wait");
        require(machine->read_reg(0, 0, 3) == 0x100, what + "lds data");
    }
}

// Slots are claimed at admission and held while the operands travel (the
// coupled lds / sts slots stay reserved from accept; the unit takes the
// instruction issue_to_dispatch later). One load slot: lds.uni A accepted at
// a; lds.uni B, tried from a + 1, is refused "load slots full" although the
// slot register is still empty (A is in the dispatch pipeline). One store
// slot, two warps: warp 1's sts, tried the cycle after warp 0's, is refused
// "store slots full" while warp 0's sts travels and its slot is still empty.
void test_i6a_slots_claimed_while_operands_travel() {
    auto profile = canonical_profile();
    profile.shmem_load_slots = 1;
    profile.shmem_store_slots = 1;
    const auto t = nanosm::issue_to_dispatch(profile);
    {
        auto machine = i6a_machine(profile, hand_assemble(i1_asm(profile, R"(
            lds.uni r3, [r1] {pred=PT, stall=0, set=sb0, wait=-}
            lds.uni r6, [r1 + 4] {pred=PT, stall=0, set=sb1, wait=-}
            halt {pred=PT, stall=0, set=-, wait=sb0|sb1}
        )"), profile));
        std::map<uint64_t, bool> held; // the slot register (Q) during each cycle
        while (!machine->done()) {
            held[machine->cycle()] = machine->load_slot_held(0);
            tick_capped(*machine);
        }
        const auto a = issued_at(*machine, 0).at(0);
        const auto refused = refusals(*machine, 0, "load slots full");
        require(!refused.empty() && refused.front().first == a + 1, "B was not refused from a + 1:" + describe(refused));
        for (uint64_t cycle = a + 1; cycle <= a + t; ++cycle) {
            require(!held.at(cycle), "the slot register was not empty while A's operands travelled");
        }
        require(held.at(a + t + 1), "A did not take the slot at its dispatch");
    }
    {
        auto machine = i6a_machine(profile, hand_assemble(i1_asm(profile, R"(
            sread r0, LANE_ID
            sread r1, WARP_ID
            shli r0, r0, 2
            shli r1, r1, 6
            add r0, r0, r1
            sts [r0 + 1024], r1 {pred=PT, stall=0, set=sb0, wait=-}
            halt {pred=PT, stall=0, set=-, wait=sb0}
        )"), profile), 2);
        std::map<uint64_t, bool> held;
        while (!machine->done()) {
            held[machine->cycle()] = machine->store_slot_held(0);
            tick_capped(*machine);
        }
        const auto issues = issued_at(*machine, 5 * 8);
        require(issues.size() == 2 && issues[1] > issues[0], "two sts:" + issue_log(*machine));
        const auto refused = refusals(*machine, 1, "store slots full");
        require(!refused.empty() && refused.front().first == issues[0] + 1 && !held.at(issues[0] + 1),
            "warp 1's sts was not refused for the slot warp 0's travelling sts claimed:" + describe(refused) +
                issue_log(*machine));
        for (uint32_t lane = 0; lane < 16; ++lane) {
            require(machine->read_shmem_u32(1024 + 4 * lane) == 0 && machine->read_shmem_u32(1024 + 64 + 4 * lane) == 64,
                "sts data");
        }
    }
}

// rf.read_latency_cycles (Lr) and rf.write_depth_cycles (Wd): optional in
// the JSON (absent: the canonical 2), nonzero register stages (1..8), part of
// the fingerprint. And the longest fixed dependency must fit the five-bit
// stall field (src/profile.cpp): Lr + E + Wd + 1 <= 31, E the longest fixed
// execution (canonical: the effective SFU latency, 11).
void test_i6a_profile_fields() {
    i21_check_optional_field("rf", "read_latency_cycles", 2, &nanosm::DeviceProfile::rf_read_latency_cycles);
    i21_check_optional_field("rf", "write_depth_cycles", 2, &nanosm::DeviceProfile::rf_write_depth_cycles);
    auto profile = canonical_profile();
    profile.rf_read_latency_cycles = 8;
    profile.rf_write_depth_cycles = 8;
    profile.timing.sfu = static_cast<uint8_t>(31 - (8 + 8 + 1)); // exactly 31
    nanosm::validate_device_profile(profile);
    ++profile.timing.sfu; // 32
    try {
        nanosm::validate_device_profile(profile);
        throw std::runtime_error("a fixed dependency longer than the stall field passed validation");
    } catch (const nanosm::ProfileError& error) {
        require(std::string(error.what()).find("five-bit stall field") != std::string::npos,
            "the stall-field refusal has the wrong text");
    }
}

// ---------------------------------------------------------------------------
// I6b-1: the RF read side of trash/rfc14.md (sections 3-5) and 16 warps.
// Every instruction reads its operands in ONE round, sampled at issue +
// issue_to_sample; one instruction is accepted per cycle, so reads of
// different instructions never meet. The issue register's bank use owns the
// RF read ports in its sample cycle; acc.st beats and the LSU's store-data
// reads take the rest, with read_owed / inhibit making a hole for a denied
// reader. trash/rf-read-side-16-warps-2026-10-18.md.
// ---------------------------------------------------------------------------

// Every debug "RF read" sample of warp 0, by cycle: (register, bank).
std::map<uint64_t, std::vector<std::pair<uint32_t, uint32_t>>> i6b1_reads(const nanosm::Machine& machine) {
    std::map<uint64_t, std::vector<std::pair<uint32_t, uint32_t>>> reads;
    for (const auto& event : machine.trace().events()) {
        if (event.name == "RF read" && event.track == 100) {
            reads[event.start_cycle].emplace_back(static_cast<uint32_t>(event.id), static_cast<uint32_t>(event.value));
        }
    }
    return reads;
}

// Back-to-back same-bank reads. Nine independent instructions that all read
// RF bank 0 (r0, r3, r6, r9: two-source adds, a repeated register, xori,
// addi, lds.uni, stg, lds, setp) issue in consecutive cycles: no refusal,
// no bubble; bank 0 is sampled in every cycle of the run, each
// instruction's sources at its own issue + issue_to_sample; the data are
// right. At Lr 1, 2 and 3 -- the RF's output latency changes when operands
// arrive, never when or whether an instruction is accepted. Then four warps
// run the arithmetic part of it: their issues interleave one per cycle with
// no gap, and bank 0 is sampled once per cycle.
void test_i6b1_back_to_back_same_bank_reads() {
    const std::string setup = R"(
        sread r9, LANE_ID
        sread r4, WARP_ID
        shli r9, r9, 2
        shli r4, r4, 6
        add r9, r9, r4
        mov r0, 4
        mov r3, 8
        mov r6, 12
        mov r1, 100
        mov r2, 200
    )";
    // The run, from pc 80: (each instruction's distinct sources).
    const std::vector<std::vector<uint32_t>> mixed_sources{
        {0, 1}, {3, 2}, {6}, {9}, {0}, {9, 1}, {6}, {9}, {9}};
    const std::string mixed = setup + R"(
        add r10, r0, r1
        add r11, r3, r2
        xori r13, r6, 5
        addi r14, r9, 1
        lds.uni r16, [r0 + 512] {pred=PT, stall=0, set=sb0, wait=-}
        stg [r9 + 4096], r1 {pred=PT, stall=0, set=sb1, wait=-}
        add r17, r6, r6
        lds r19, [r9 + 2048] {pred=PT, stall=0, set=sb2, wait=-}
        setp.ne p0, r9, 0
        halt {pred=PT, stall=0, set=-, wait=sb0|sb1|sb2}
    )";
    const std::vector<std::vector<uint32_t>> arithmetic_sources{{0, 1}, {3, 2}, {6}, {9}, {6}, {9}};
    const std::string arithmetic = setup + R"(
        add r10, r0, r1
        add r11, r3, r2
        xori r13, r6, 5
        addi r14, r9, 1
        add r17, r6, r6
        setp.ne p0, r9, 0
        halt
    )";
    for (const uint32_t lr : {1u, 2u, 3u}) {
        const auto profile = i6a_profile(lr, 2);
        for (const auto& [warps, body, sources] : std::vector<std::tuple<uint32_t, std::string,
                 std::vector<std::vector<uint32_t>>>>{{1, mixed, mixed_sources}, {4, arithmetic, arithmetic_sources}}) {
            const auto what = i6a_what(profile) + std::to_string(warps) + " warp(s): ";
            auto machine = i6a_machine(profile, hand_assemble(i1_asm(profile, body), profile), warps);
            machine->run();
            const auto log = " | issues" + issue_log(*machine);
            require(machine->trace().summary().rf_simt_read_inhibited == 0, what + "a read was inhibited");
            std::set<uint64_t> run_issues;
            std::map<uint64_t, uint32_t> bank0_samples; // SIMT samples of bank 0 by cycle
            for (const auto& event : machine->trace().events()) {
                if (event.name == "RF read" && event.value == 0) ++bank0_samples[event.start_cycle];
                if (event.category == nanosm::TraceCategory::warp && event.detail.rfind("RF", 0) == 0) {
                    throw std::runtime_error(what + "an RF refusal: " + event.detail + log);
                }
            }
            for (uint32_t warp = 0; warp < warps; ++warp) {
                std::vector<uint64_t> issues;
                for (std::size_t index = 0; index < sources.size(); ++index) {
                    uint64_t cycle = 0;
                    uint32_t found = 0;
                    for (const auto& event : machine->trace().events()) {
                        if (is_issue_event(event) && event.id == 80 + 8 * index &&
                            event.detail.rfind("warp=" + std::to_string(warp) + ",", 0) == 0) {
                            cycle = event.start_cycle;
                            ++found;
                        }
                    }
                    require(found == 1, what + "run instruction " + std::to_string(index) + " did not issue once" + log);
                    issues.push_back(cycle);
                    run_issues.insert(cycle);
                    // Its sources, sampled at its issue + 1, in their banks.
                    for (const auto reg : sources[index]) {
                        uint32_t samples = 0;
                        for (const auto& event : machine->trace().events()) {
                            if (event.name == "RF read" && event.id == reg && event.start_cycle == cycle + 1 &&
                                event.track == 100 + warp) {
                                ++samples;
                            }
                        }
                        require(samples == 1, what + "r" + std::to_string(reg) + " of run instruction " +
                            std::to_string(index) + " was not sampled at issue + 1" + log);
                    }
                }
                if (warps == 1) {
                    for (std::size_t i = 1; i < issues.size(); ++i) {
                        require(issues[i] == issues[0] + i, what + "the run did not issue back to back" + log);
                    }
                }
                for (uint32_t lane = 0; lane < profile.lanes; ++lane) {
                    const auto base = 4 * lane + 64 * warp;
                    require(machine->read_reg(warp, lane, 10) == 104 && machine->read_reg(warp, lane, 11) == 208 &&
                            machine->read_reg(warp, lane, 13) == (12u ^ 5u) &&
                            machine->read_reg(warp, lane, 14) == base + 1 && machine->read_reg(warp, lane, 17) == 24,
                        what + "arithmetic data");
                    if (warps == 1) {
                        require(machine->read_reg(warp, lane, 16) == 0x100 + 516 / 4 &&
                                machine->read_reg(warp, lane, 19) == 0x100 + (2048 + base) / 4 &&
                                machine->read_gmem_u32(4096 + base) == 100,
                            what + "memory data");
                    }
                }
            }
            // Full sampling rate: from the run's first issue to its last, an
            // instruction is accepted every cycle and bank 0 is sampled in
            // every following cycle, once.
            const auto first = *run_issues.begin();
            const auto last = *run_issues.rbegin();
            require(run_issues.size() == last - first + 1, what + "the run's issues have a gap" + log);
            for (auto cycle = first + 1; cycle <= last + 1; ++cycle) {
                require(bank0_samples[cycle] == 1, what + "bank 0 was not sampled once at c" + std::to_string(cycle) + log);
            }
            check_read_capacity(*machine, profile);
        }
    }
}

// Two DISTINCT source registers of one instruction in one RF bank are a
// malformed instruction (one read round, one read port per bank), not a
// stall: qtxc and the hand assembler reject the line, kernel load rejects
// the word -- in both builds, naming the instruction and the registers.
// Every operand form: the two- and three-source arithmetic, setp, sts's
// base and value, stg, an atomic, the mbar register forms, mma.async's two
// addresses, and (with two banks) tma.async's coordinate group r0..r2. The
// kernel-load case carries words a looser profile (more read ports)
// assembled, restamped for the profile under test.
void test_i6b1_same_bank_sources_are_malformed() {
    const auto& canonical = canonical_profile();
    auto two_banks = canonical;
    two_banks.rf_banks = 2;
    struct Case {
        std::string line;
        std::string registers;
        const nanosm::DeviceProfile* profile;
    };
    const std::vector<Case> cases{
        {"add r2, r0, r3", "r0 and r3", &canonical},
        {"fadd r2, r1, r4", "r1 and r4", &canonical},
        {"fmul r2, r5, r8", "r5 and r8", &canonical},
        {"fsub r2, r0, r6", "r0 and r6", &canonical},
        {"fmax r1, r2, r8", "r2 and r8", &canonical},
        {"cvt.pack.f16x2 r2, r4, r7", "r4 and r7", &canonical},
        {"and r2, r0, r3", "r0 and r3", &canonical},
        {"or r2, r0, r3", "r0 and r3", &canonical},
        {"xor r2, r0, r3", "r0 and r3", &canonical},
        {"shl r2, r0, r3", "r0 and r3", &canonical},
        {"shr r2, r0, r3", "r0 and r3", &canonical},
        {"fma r1, r2, r5, r8", "r2, r5 and r8", &canonical},
        {"fma r1, r0, r3, r4", "r0 and r3", &canonical},
        {"setp.eq p0, r0, r3", "r0 and r3", &canonical},
        {"sts [r3 + 64], r6", "r3 and r6", &canonical},
        {"stg [r3], r6", "r3 and r6", &canonical},
        {"atom.add.uni r1, [r3], r6", "r3 and r6", &canonical},
        {"mbar.init [r3], r6", "r3 and r6", &canonical},
        {"mbar.arrive.expect_tx [r3], r6", "r3 and r6", &canonical},
        {"mbar.test_wait p0, [r3], r6", "r3 and r6", &canonical},
        {"mbar.try_wait p0, [r3], r6", "r3 and r6", &canonical},
        {"mma.async C, [r3], [r6]", "r3 and r6", &canonical},
        {"tma.async [1024], r0", "r0 and r2", &two_banks},
    };
    const auto mentions = [](const std::string& message, const std::string& where, const Case& c) {
        return message.rfind(where, 0) == 0 && message.find(": malformed instruction '") != std::string::npos &&
            message.find("its distinct sources " + c.registers + " are in one RF bank") != std::string::npos;
    };
    for (const auto& c : cases) {
        const auto& profile = *c.profile;
        const auto source = ".acc C, 0, 0, 64, 64\n" + c.line + "\nhalt\n";
        try {
            (void)nanosm::compile_qtx(source, profile);
            throw std::runtime_error("qtxc accepted '" + c.line + "'");
        } catch (const nanosm::SimError& error) {
            require(mentions(error.what(), "line 2", c), "qtxc's refusal of '" + c.line + "': " + error.what());
        }
        // The same line through a looser profile (one read port per source),
        // emitted as the profile under test's assembly.
        auto loose = profile;
        loose.rf_read_ports_per_bank = 3;
        auto program = nanosm::compile_qtx(source, loose);
        program.profile_fingerprint = nanosm::device_profile_fingerprint(profile);
        try {
            (void)nanosm::assemble_asm(nanosm::emit_asm(program), profile);
            throw std::runtime_error("the assembler accepted '" + c.line + "'");
        } catch (const nanosm::SimError& error) {
            require(mentions(error.what(), "line ", c), "the assembler's refusal of '" + c.line + "': " + error.what());
        }
        auto kernel = nanosm::assemble_asm(nanosm::emit_asm(nanosm::compile_qtx(source, loose)), loose);
        kernel.profile_fingerprint = nanosm::device_profile_fingerprint(profile);
        nanosm::Launch launch;
        launch.active_warps = 1;
        try {
            nanosm::Machine machine(profile, kernel, std::move(launch));
            throw std::runtime_error("kernel load accepted '" + c.line + "'");
        } catch (const nanosm::SimError& error) {
            require(mentions(error.what(), "kernel pc 0", c), "kernel load's refusal of '" + c.line + "': " + error.what());
        }
    }
    // The same register in several operand slots is one read: qtxc, the
    // assembler and kernel load accept it, and it computes.
    nanosm::Launch launch;
    launch.active_warps = 1;
    image_bytes(launch.shmem_image, 256);
    nanosm::Machine machine(canonical, compile(R"(
        sread r3, LANE_ID
        shli r3, r3, 2
        add r2, r3, r3
        mov r4, 0x3f8
        shli r4, r4, 20
        fma r5, r4, r4, r4
        setp.eq p0, r3, r3
        sts [r3 + 64], r3
        halt
    )", canonical), std::move(launch), {.timeline = true, .debug = true});
    machine.run();
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(machine.read_reg(0, lane, 2) == 8 * lane && machine.read_reg(0, lane, 5) == nanosm::float_bits(2.0f) &&
                machine.read_shmem_u32(64 + 4 * lane) == 4 * lane,
            "a repeated source register computed the wrong value");
    }
    check_read_capacity(machine, canonical);
}

// One async reader under continuous SIMT reads of its bank (RFC14 section
// 5). An acc.st's data beat (r0, bank 0) against addis that read r27 (bank
// 0) every cycle. The acc.st, accepted at a, reaches its unit at a + t
// (issue_to_dispatch) and its beat wants bank 0 from a + t + 1 = a + 4.
// There the issue register owns bank 0 (the addi accepted at a + 3): the
// beat is denied, and read_owed (set by that acceptance) makes the denial an
// inhibition -- the next candidate, an add reading banks 0 AND 1, is refused
// at a + 4, all or none: nothing of it is read in a + 5, bank 1 included. At
// a + 5 the issue register is empty, the beat takes bank 0 and the add is
// accepted in the same cycle (read_owed: the acceptance sets it, the grant
// clears it, set wins; the cursor moves past the acc.st). Every accepted
// instruction samples at its issue + 1; the stream loses exactly one cycle.
void test_i6b1_async_reader_gets_the_hole() {
    const auto& profile = canonical_profile();
    auto machine = i6a_machine(profile, hand_assemble(i1_asm(profile, R"(
        .acc C, 0, 0, 64, 64
        mov r30, 0
        mov r31, 0
        mov r28, 7
        sread r0, LANE_ID
        mov r27, 5 {pred=PT, stall=8, set=-, wait=-}
        acc.st C, r30, r0 {pred=PT, stall=0, set=sb0, wait=-}
        addi r1, r27, 1
        addi r2, r27, 2
        addi r5, r27, 3
        add r4, r27, r28
        addi r7, r27, 4
        acc.ld r10, C, r30 {pred=PT, stall=0, set=sb1, wait=sb0}
        halt {pred=PT, stall=0, set=-, wait=sb1}
    )"), profile));
    std::map<uint64_t, bool> owed;       // Q during each cycle
    std::map<uint64_t, uint32_t> cursor;
    std::map<uint64_t, std::pair<uint32_t, uint32_t>> bank_use; // the issue register's banks 0 and 1
    (void)i1_run(*machine, {}, [&](uint64_t cycle) {
        owed[cycle + 1] = machine->rf_read_owed(0);
        cursor[cycle + 1] = machine->rf_read_cursor(0);
        bank_use[cycle + 1] = {machine->issue_read_bank_use(0), machine->issue_read_bank_use(1)};
    });
    const auto log = " | issues" + issue_log(*machine);
    const auto a = issued_at(*machine, 5 * 8).at(0);
    const std::vector<uint64_t> expected{a + 1, a + 2, a + 3, a + 5, a + 6};
    for (uint32_t i = 0; i < expected.size(); ++i) {
        require(issued_at(*machine, (6 + i) * 8) == std::vector<uint64_t>{expected[i]},
            "SIMT instruction " + std::to_string(i) + " did not issue at its cycle" + log);
    }
    require(refusals(*machine, 0, "RF read inhibited") == std::vector<std::pair<uint64_t, uint64_t>>{{a + 4, 1}},
        "the refusal:" + describe(refusals(*machine, 0, "RF read inhibited")) + log);
    const auto reads = i6b1_reads(*machine);
    require(reads.at(a + 5) == std::vector<std::pair<uint32_t, uint32_t>>{{0, 0}},
        "in the hole cycle the beat was not the only read (a refused instruction read a register?)");
    require(i6a_rf_events(*machine, "RF read", 27) ==
                std::vector<uint64_t>{a + 2, a + 3, a + 4, a + 6, a + 7} &&
            i6a_rf_events(*machine, "RF read", 28) == std::vector<uint64_t>{a + 6},
        "an accepted instruction was not sampled at its issue + 1");
    require(bank_use.at(a + 4) == std::pair<uint32_t, uint32_t>{1, 0} &&
            bank_use.at(a + 5) == std::pair<uint32_t, uint32_t>{0, 0} &&
            bank_use.at(a + 6) == std::pair<uint32_t, uint32_t>{1, 1},
        "the issue register's bank use");
    require(owed.at(a + 4) && owed.at(a + 5) && owed.at(a + 6),
        "read_owed was not held through the denial, or the grant beat the acceptance (set wins)");
    require(cursor.at(a + 5) == 0 && cursor.at(a + 6) == 1, "the cursor did not move past the acc.st at its grant");
    require(machine->trace().summary().rf_simt_read_inhibited == 1, "inhibited cycles");
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(machine->read_reg(0, lane, 4) == 12 && machine->read_reg(0, lane, 10) == lane,
            "the add or the acc.st -> acc.ld round trip");
    }
    check_read_capacity(*machine, profile);
}

// Three async readers of one bank take the holes in round-robin order:
// acc.st C0 (tmem bank 0, holder 0), acc.st C1 (tmem bank 1, holder 1) and
// stg.v8's first store-data beat (the LSU, holder 2), all wanting bank 0
// (r0, r9, r18) while addis read r27 (bank 0) every cycle they may. The
// acc.sts (a, a + 1) reach their units at a + 3, a + 4, the stg.v8 (a + 2)
// at a + 5 (its first beat may read in its dispatch cycle). Each denial is
// an inhibition and each hole one grant: refusals at a + 4, a + 6, a + 8,
// grants C0 at a + 5, C1 at a + 7, the LSU at a + 9. The LSU's later beats
// in bank 0 (r21, r24) wait at most one cycle each. Data: tmem (read back)
// and gmem.
void test_i6b1_async_readers_round_robin() {
    const auto& profile = canonical_profile();
    std::string setup = "sread r0, LANE_ID\naddi r9, r0, 100\n";
    for (uint32_t j = 0; j < 8; ++j) setup += "addi r" + std::to_string(18 + j) + ", r0, " + std::to_string(200 + 10 * j) + "\n";
    auto machine = i6a_machine(profile, hand_assemble(i1_asm(profile, R"(
        .acc C0, 0, 0, 64, 64
        .acc C1, 1, 0, 64, 64
        mov r30, 0
        mov r31, 0
        mov r40, 0
    )" + setup + R"(
        mov r27, 5 {pred=PT, stall=8, set=-, wait=-}
        acc.st C0, r30, r0 {pred=PT, stall=0, set=sb0, wait=-}
        acc.st C1, r30, r9 {pred=PT, stall=0, set=sb1, wait=-}
        stg.v8 [r40], r18 {pred=PT, stall=0, set=sb2, wait=-}
        addi r1, r27, 1
        addi r2, r27, 2
        addi r4, r27, 3
        addi r5, r27, 4
        addi r7, r27, 5
        addi r8, r27, 6
        addi r10, r27, 7
        addi r11, r27, 8
        addi r13, r27, 9
        addi r14, r27, 10
        addi r16, r27, 11
        addi r17, r27, 12
        addi r1, r27, 1
        addi r2, r27, 2
        addi r4, r27, 3
        addi r5, r27, 4
        addi r7, r27, 5
        addi r8, r27, 6
        addi r10, r27, 7
        addi r11, r27, 8
        addi r13, r27, 9
        addi r14, r27, 10
        addi r16, r27, 11
        addi r17, r27, 12
        acc.ld r50, C0, r30 {pred=PT, stall=0, set=sb3, wait=sb0}
        acc.ld r51, C1, r30 {pred=PT, stall=0, set=sb4, wait=sb1}
        halt {pred=PT, stall=0, set=-, wait=sb2|sb3|sb4}
    )"), profile));
    std::map<uint64_t, uint32_t> cursor;
    (void)i1_run(*machine, {}, [&](uint64_t cycle) { cursor[cycle + 1] = machine->rf_read_cursor(0); });
    const auto log = " | issues" + issue_log(*machine);
    const auto a = issued_at(*machine, 14 * 8).at(0);
    const auto refused = refusals(*machine, 0, "RF read inhibited");
    require(refused.size() >= 3 && refused[0] == std::pair<uint64_t, uint64_t>{a + 4, 1} &&
            refused[1] == std::pair<uint64_t, uint64_t>{a + 6, 1} && refused[2] == std::pair<uint64_t, uint64_t>{a + 8, 1},
        "the first three refusals:" + describe(refused) + log);
    require(i6a_rf_events(*machine, "RF read", 0).back() == a + 5 && i6a_rf_events(*machine, "RF read", 9).back() == a + 7 &&
            i6a_rf_events(*machine, "RF read", 18).back() == a + 9,
        "the holes did not go C0, C1, LSU in turn" + log);
    require(cursor.at(a + 6) == 1 && cursor.at(a + 8) == 2 && cursor.at(a + 10) == 0,
        "the cursor did not move past each reader at its grant");
    // Every inhibited cycle's hole goes to a waiting reader the next cycle.
    const auto reads = i6b1_reads(*machine);
    for (const auto& [start, length] : refused) {
        require(length == 1, "a refusal lasted more than one cycle:" + describe(refused));
        bool hole_used = false;
        for (const auto& [reg, bank] : reads.at(start + 1)) hole_used = hole_used || (bank == 0 && reg != 27);
        require(hole_used, "the hole after the refusal at c" + std::to_string(start) + " went unused");
    }
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(machine->read_reg(0, lane, 50) == lane && machine->read_reg(0, lane, 51) == lane + 100,
            "tmem data (acc.st -> acc.ld)");
        for (uint32_t j = 0; j < 8; ++j) {
            require(machine->read_gmem_u32(4 * (16 * j + lane)) == lane + 200 + 10 * j, "stg.v8 data");
        }
    }
    check_read_capacity(*machine, profile);
}

// Continuous SIMT pressure on bank 0 (60 addis reading r27) with three
// persistent async readers whose beats come back to bank 0 every third
// register: acc.st.v8 C0 (r0..r7), acc.st.v8 C1 (r9..r16) and stg.v8
// (r18..r25). No accepted read is ever stopped (each addi samples r27 at its
// issue + 1); every acc.st beat is granted within 2 x 3 - 1 = 5 cycles of
// wanting its bank (in bank 0 one hole per two cycles, the other two
// readers at most ahead of it; in banks 1 and 2 only the readers compete);
// every inhibited cycle's hole goes to an async reader the next cycle; and
// the data (tmem read back, gmem) are right.
void test_i6b1_persistent_readers_under_pressure() {
    const auto& profile = canonical_profile();
    std::string setup = "sread r0, LANE_ID\n";
    for (uint32_t j = 1; j < 8; ++j) setup += "addi r" + std::to_string(j) + ", r0, " + std::to_string(10 * j) + "\n";
    for (uint32_t j = 0; j < 8; ++j) setup += "addi r" + std::to_string(9 + j) + ", r0, " + std::to_string(100 + 10 * j) + "\n";
    for (uint32_t j = 0; j < 8; ++j) setup += "addi r" + std::to_string(18 + j) + ", r0, " + std::to_string(200 + 10 * j) + "\n";
    std::string pressure;
    for (uint32_t i = 0; i < 60; ++i) pressure += "addi r41, r27, " + std::to_string(i) + "\n";
    const auto body = R"(
        .acc C0, 0, 0, 64, 64
        .acc C1, 1, 0, 64, 64
        mov r30, 0
        mov r31, 0
        mov r40, 0
    )" + setup + R"(
        mov r27, 5 {pred=PT, stall=8, set=-, wait=-}
        acc.st.v8 C0, r30, r0 {pred=PT, stall=0, set=sb0, wait=-}
        acc.st.v8 C1, r30, r9 {pred=PT, stall=0, set=sb1, wait=-}
        stg.v8 [r40], r18 {pred=PT, stall=0, set=sb2, wait=-}
    )" + pressure + R"(
        acc.ld.v8 r50, C0, r30 {pred=PT, stall=0, set=sb3, wait=sb0}
        acc.ld.v8 r58, C1, r30 {pred=PT, stall=0, set=sb4, wait=sb1}
        halt {pred=PT, stall=0, set=-, wait=sb2|sb3|sb4}
    )";
    const auto kernel = hand_assemble(i1_asm(profile, body), profile);
    auto machine = i6a_machine(profile, kernel);
    machine->run();
    const auto log = " | issues" + issue_log(*machine);
    const uint32_t first_acc_st = 3 + 1 + 7 + 8 + 8 + 1; // index of acc.st.v8 C0 (after 28 setup instructions)
    const auto a = issued_at(*machine, first_acc_st * 8).at(0);
    const auto t = nanosm::issue_to_dispatch(profile);
    // Accepted reads are never stopped.
    std::set<uint64_t> expected_r27;
    for (uint32_t i = 0; i < 60; ++i) expected_r27.insert(issued_at(*machine, (first_acc_st + 3 + i) * 8).at(0) + 1);
    const auto r27 = i6a_rf_events(*machine, "RF read", 27);
    require(std::set<uint64_t>(r27.begin(), r27.end()) == expected_r27 && r27.size() == 60,
        "an accepted addi was not sampled at its issue + 1" + log);
    // Bounded waits of the acc.st beats: beat j wants its bank from its
    // transfer's dispatch + 1 (beat 0) or the cycle after beat j - 1's grant.
    uint32_t longest = 0;
    for (const auto& [base, issue] : std::vector<std::pair<uint32_t, uint64_t>>{{0, a}, {9, a + 1}}) {
        auto ready = issue + t + 1;
        for (uint32_t j = 0; j < 8; ++j) {
            const auto grants = i6a_rf_events(*machine, "RF read", static_cast<uint8_t>(base + j));
            const auto grant = grants.back();
            require(grant >= ready, "a beat was read before it was wanted");
            longest = std::max(longest, static_cast<uint32_t>(grant - ready));
            ready = grant + 1;
        }
    }
    require(longest <= 5, "an acc.st beat waited " + std::to_string(longest) + " cycles" + log);
    require(machine->trace().summary().rf_simt_read_inhibited > 0, "the pressure never needed an inhibition");
    const auto reads = i6b1_reads(*machine);
    for (const auto& [start, length] : refusals(*machine, 0, "RF read inhibited")) {
        require(length == 1, "an inhibition lasted more than one cycle");
        bool hole_used = false;
        for (const auto& [reg, bank] : reads.at(start + 1)) hole_used = hole_used || (bank == 0 && reg != 27);
        require(hole_used, "the hole after the refusal at c" + std::to_string(start) + " went unused");
    }
    for (uint32_t lane = 0; lane < 16; ++lane) {
        for (uint32_t j = 0; j < 8; ++j) {
            require(machine->read_reg(0, lane, 50 + j) == lane + 10 * j &&
                    machine->read_reg(0, lane, 58 + j) == lane + 100 + 10 * j,
                "tmem data (acc.st.v8 -> acc.ld.v8)");
            require(machine->read_gmem_u32(4 * (16 * j + lane)) == lane + 200 + 10 * j, "stg.v8 data");
        }
    }
    check_read_capacity(*machine, profile);
}

// Unrelated banks proceed. Three warps: warps 0 and 1 read bank 0 (r27)
// every cycle they issue, warp 2 reads only bank 1 (r28); warp 0 issues an
// acc.st of r0 (bank 0) amid its stream. Round-robin in steady state is
// w0, w1, w2: the beat wants bank 0 at a + 4, right after warp 0's addi at
// a + 3 -- denied, inhibited: warp 1's bank-0 addi is refused at a + 4, and
// warp 2's bank-1 addi is accepted in that same cycle. At a + 5 the beat
// takes the hole.
// I6b-2: warp 2 can take the cycle only if it has an instruction then. Its
// stream starts after a taken branch (an empty FIFO), and three warps
// issuing every cycle get one fetch per cycle between them: the words
// buffered stay as few as when the streams started (they grow only in
// cycles nothing issues), so warp 2's next word would arrive just in time
// for its own turn, not a cycle early. So the three warps first meet at an
// mbar barrier (arrive, try_wait) at the head of their streams: while they
// wait, fetch fills their FIFOs, and each stream starts with words in hand.
// (warp 0's bank-0 stream is its own copy: a branch to warp 1's would
// drain the acc.st's depbar first.)
void test_i6b1_unrelated_banks_proceed() {
    const auto& profile = canonical_profile();
    std::string pad;
    for (uint32_t i = 0; i < 12; ++i) pad += "addi r1, r27, 1\n";
    std::string bank0;
    for (uint32_t i = 0; i < 24; ++i) bank0 += "addi r2, r27, 2\n";
    std::string bank1;
    for (uint32_t i = 0; i < 48; ++i) bank1 += "addi r4, r28, 3\n";
    const auto meet = [](const std::string& label) {
        return "mbar.arrive [r6]\n" + label + ": mbar.try_wait p3, [r6], 0\n@!p3 bra.uni " + label + "\n";
    };
    nanosm::Launch launch;
    launch.active_warps = 3;
    nanosm::MbarState barrier;
    barrier.valid = true;
    barrier.expected = 3;
    barrier.remaining = 3;
    write_u64(image_bytes(launch.shmem_image, 8192), 4096, nanosm::encode_mbar_state(barrier));
    nanosm::Machine machine(profile, compile(R"(
        .acc C, 0, 0, 64, 64
        sread r50, WARP_ID
        mov r27, 5
        mov r28, 7
        mov r30, 0
        mov r31, 0
        mov r6, 4096
        sread r0, LANE_ID
        setp.eq p0, r50, 2
        @p0 bra.uni other
        setp.eq p1, r50, 1
        @p1 bra.uni plain
    )" + meet("meet0") + pad + R"(
        acc.st C, r30, r0
    )" + bank0 + R"(
        halt
    plain:
    )" + meet("meet1") + bank0 + R"(
        halt
    other:
    )" + meet("meet2") + bank1 + R"(
        halt
    )", profile), std::move(launch), {.timeline = true, .debug = true});
    machine.run();
    const auto log = " | issues" + issue_log(machine);
    uint32_t acc_st_pc = 0;
    for (const auto& event : machine.trace().events()) {
        if (is_issue_event(event) && event.name == "acc.st") acc_st_pc = static_cast<uint32_t>(event.id);
    }
    const auto a = issued_at(machine, acc_st_pc).at(0);
    const auto warp_issued_at = [&](uint32_t warp, uint64_t cycle) {
        for (const auto& event : machine.trace().events()) {
            if (is_issue_event(event) && event.start_cycle == cycle &&
                event.detail.rfind("warp=" + std::to_string(warp) + ",", 0) == 0) {
                return true;
            }
        }
        return false;
    };
    require(warp_issued_at(0, a + 3) && warp_issued_at(1, a + 1) && warp_issued_at(2, a + 2),
        "the three warps were not in steady round-robin around the acc.st" + log);
    require(refusals(machine, 1, "RF read inhibited") == std::vector<std::pair<uint64_t, uint64_t>>{{a + 4, 1}},
        "warp 1's bank-0 addi was not refused exactly at a + 4:" + describe(refusals(machine, 1, "RF read inhibited")) + log);
    require(warp_issued_at(2, a + 4), "warp 2's bank-1 addi did not issue in the inhibited cycle" + log);
    require(refusals(machine, 2, "RF read inhibited").empty(), "warp 2 was inhibited for a bank it does not read");
    require(i6a_rf_events(machine, "RF read", 0).back() == a + 5, "the beat did not take the hole at a + 5" + log);
    for (uint32_t warp = 0; warp < 3; ++warp) {
        require(machine.read_reg(warp, 0, warp == 2 ? 4 : 2) == (warp == 2 ? 10u : 7u), "data");
    }
    check_read_capacity(machine, profile);
}

// 16 warps (HUMAN 10-17: "lets do 16 warps"). rf.rows_per_bank (352 at
// canonical, about as much RF as shmem) = 16 warps x 22 rows, so a
// 66-register kernel fits 16 warps and not 17. (I6b-2: core.warps, 16, now
// refuses a 17th warp first; with 32 warp slots declared, the RF rows still
// refuse it.) A 16-warp run in which every warp touches its own state:
// its RF rows and predicates, WARP_ID / THREAD_ID, a depbar (lds), its shmem
// order (lds behind its own sts), the stall count, an mbar arrival and a
// try_wait on a barrier all 16 arrive at (more waiters than the 8 wait
// slots), and its trace track.
void test_i6b1_sixteen_warps() {
    const auto& profile = canonical_profile();
    auto more_slots = profile;
    more_slots.core_warps = 32;
    for (const auto* slots : std::vector<const nanosm::DeviceProfile*>{&profile, &more_slots}) {
        const auto wide = compile("mov r65, 1\nhalt", *slots);
        require(wide.register_count == 66, "the residency kernel is not 66 registers");
        for (const uint32_t warps : {16u, 17u}) {
            nanosm::Launch launch;
            launch.active_warps = warps;
            try {
                nanosm::Machine machine(*slots, wide, std::move(launch));
                require(warps == 16, "17 warps of 66 registers were resident in 352 rows");
            } catch (const nanosm::SimError& error) {
                const auto expected = slots->core_warps == 16 ? "launch of 17 warps exceeds the SM's 16 warp slots"
                                                               : "RF allocation cannot fit 17 active warps";
                require(warps == 17 && std::string(error.what()).find(expected) != std::string::npos,
                    std::string("residency: ") + error.what());
            }
        }
    }
    nanosm::Launch launch;
    launch.active_warps = 16;
    auto& shmem = image_bytes(launch.shmem_image, 8192);
    nanosm::MbarState initial;
    initial.valid = true;
    initial.expected = 16;
    initial.remaining = 16;
    write_u64(shmem, 4096, nanosm::encode_mbar_state(initial));
    nanosm::Machine machine(profile, compile(R"(
        sread r0, WARP_ID
        sread r1, LANE_ID
        sread r10, THREAD_ID
        shli r2, r0, 6
        shli r4, r1, 2
        add r5, r2, r4
        addi r7, r0, 100
        sts [r5 + 1024], r7
        lds r8, [r5 + 1024]
        mov r6, 4096
        mbar.arrive [r6]
    wait:
        mbar.try_wait p0, [r6], 0
        @!p0 bra.uni wait
        addi r65, r8, 1
        halt
    )", profile), std::move(launch), {.timeline = true});
    machine.run();
    std::set<uint32_t> tracks;
    for (const auto& event : machine.trace().events()) {
        if (event.category == nanosm::TraceCategory::warp) tracks.insert(event.track);
    }
    require(tracks.size() == 16 && *tracks.begin() == 100 && *tracks.rbegin() == 115, "16 warp tracks");
    for (uint32_t warp = 0; warp < 16; ++warp) {
        for (uint32_t lane = 0; lane < 16; ++lane) {
            require(machine.read_reg(warp, lane, 10) == 16 * warp + lane && machine.read_reg(warp, lane, 8) == warp + 100 &&
                    machine.read_reg(warp, lane, 65) == warp + 101 &&
                    machine.read_shmem_u32(1024 + 64 * warp + 4 * lane) == warp + 100,
                "warp " + std::to_string(warp) + " lane " + std::to_string(lane) + " data");
        }
        for (uint8_t bar = 0; bar < 6; ++bar) require(machine.depbar_value(warp, bar) == 0, "a depbar left set");
    }
    const auto state = machine.read_mbar_state(4096);
    require(state.phase && state.remaining == 16, "the barrier did not complete one phase with 16 arrivals");
    const auto& summary = machine.trace().summary();
    uint64_t warp_cycles = 0;
    for (const auto cycles : summary.warp_cycles) warp_cycles += cycles;
    require(warp_cycles == summary.cycles * 16, "warp attribution does not cover 16 warps");
    require(summary.warp_cycles[static_cast<std::size_t>(nanosm::WarpState::mbar_wait_slot)] != 0,
        "16 waiters never found the 8 wait slots full");
}

// ---------------------------------------------------------------------------
// I6b-2: the front end (trash/front-end-2026-10-18.md, two-word packets
// since trash/front-end-packets-2026-10-18.md; the cycles in src/rf.hpp).
// Per warp an instruction FIFO of core.instruction_buffer_packets fetch
// packets (2 at canonical; a packet is the two words at an aligned 16-byte
// address); one instruction-memory read of one packet per cycle, fixed at
// one cycle; round-robin fetch over the warps with room, counting the read in
// flight, from registered state only; issuing a packet's second word pops
// it; a branch resolves in the issue register's cycle and a taken one clears
// its warp's FIFO and read in flight.
// ---------------------------------------------------------------------------

// Every issue of the run, in order: (cycle, warp, pc).
std::vector<std::tuple<uint64_t, uint32_t, uint32_t>> i6b2_issues(const nanosm::Machine& machine) {
    std::vector<std::tuple<uint64_t, uint32_t, uint32_t>> issues;
    for (const auto& event : machine.trace().events()) {
        if (!is_issue_event(event)) continue;
        const auto warp = std::stoul(event.detail.substr(5, event.detail.find(',') - 5));
        issues.emplace_back(event.start_cycle, static_cast<uint32_t>(warp), static_cast<uint32_t>(event.id));
    }
    return issues;
}

// The fetches granted (debug traces): (cycle, warp, pc).
std::vector<std::tuple<uint64_t, uint32_t, uint32_t>> i6b2_fetches(const nanosm::Machine& machine) {
    std::vector<std::tuple<uint64_t, uint32_t, uint32_t>> fetches;
    for (const auto& event : machine.trace().events()) {
        if (event.category == nanosm::TraceCategory::warp && event.name == "fetch") {
            fetches.emplace_back(event.start_cycle, event.track - 100, static_cast<uint32_t>(event.id));
        }
    }
    return fetches;
}

// Raw hand assembly by default: every stall is the one written (no
// block-boundary drain on branches), so each scenario's cycles are its own.
// `scheduled`: the fixed-latency stalls raised as qtxc would (hand_assemble).
std::unique_ptr<nanosm::Machine> i6b2_machine(
    const nanosm::DeviceProfile& profile, const std::string& body, uint32_t warps = 1, bool scheduled = false) {
    const auto text = i1_asm(profile, body);
    return i6a_machine(profile, scheduled ? hand_assemble(text, profile) : nanosm::assemble_asm(text, profile), warps);
}

std::string i6b2_independent_movs(uint32_t count) {
    std::string body;
    for (uint32_t i = 0; i < count; ++i) body += "mov r" + std::to_string(i % 16) + ", " + std::to_string(i) + "\n";
    return body;
}

// A lone warp of independent instructions from launch. Its first packet is
// fetched at c0 and its first word issues at fetch_to_issue (c2). Each FIFO
// entry is a credit that goes round in fetch_to_issue + 2 cycles carrying two
// words: fetched at f, its words issued at f + 2 and f + 3 at the earliest,
// the packet popped at f + 3's edge and its entry seen by fetch at f + 4
// (fetch reads the registered valid packets, not this cycle's pop). So N
// entries carry 2N words per 4 cycles: one entry issues two words in every
// four cycles, two entries (canonical) one word per cycle from c2, three or
// four no faster. With C = min(N, 2) credits doing work, instruction k issues
// at fetch_to_issue + (k / 2C) x 4 + k mod 2C.
void test_i6b2_lone_warp_issue_rate() {
    for (const uint32_t packets : {1u, 2u, 3u, 4u}) {
        auto profile = canonical_profile();
        profile.core_instruction_buffer_packets = packets;
        auto machine = i6b2_machine(profile, i6b2_independent_movs(12) + "halt\n");
        machine->run();
        const auto issues = i6b2_issues(*machine);
        const uint64_t round_trip = nanosm::fetch_to_issue + 2;
        const uint64_t words_per_trip = 2 * std::min<uint64_t>(packets, round_trip / 2);
        require(issues.size() == 13, std::to_string(packets) + " packets: not 13 issues");
        for (uint64_t k = 0; k < issues.size(); ++k) {
            const auto expected = nanosm::fetch_to_issue + k / words_per_trip * round_trip + k % words_per_trip;
            require(std::get<0>(issues[k]) == expected && std::get<2>(issues[k]) == 8 * k,
                std::to_string(packets) + " packets: instruction " + std::to_string(k) + " did not issue at " +
                    std::to_string(expected) + ":" + issue_log(*machine));
        }
        for (uint32_t reg = 0; reg < 12; ++reg) require(machine->read_reg(0, 3, reg) == reg, "mov data");
    }
}

// Several warps of independent instructions. Fetch serves the warps with room
// in turn and issue takes the heads in turn: two warps alternate from c2
// (warp w's k-th instruction at fetch_to_issue + w + 2k), one issue per
// cycle between them. Sixteen warps: warp w's first packet is fetched at cw
// and its first word issues at fetch_to_issue + w; every warp then issues
// once in sixteen cycles, k-th at fetch_to_issue + w + 16k, the machine
// issuing every cycle (each warp needs a packet every 32 cycles: the fetch
// port has room to spare). From launch every warp has room for both its
// packet entries, so the first 2W fetches go round the W warps twice: fetch
// i at ci, warp i mod W, packet 16 x (i / W).
void test_i6b2_warps_interleave() {
    for (const uint32_t warps : {2u, 16u}) {
        const auto& profile = canonical_profile();
        auto machine = i6b2_machine(profile, i6b2_independent_movs(8) + "halt\n", warps);
        machine->run();
        const auto issues = i6b2_issues(*machine);
        require(issues.size() == 9u * warps, std::to_string(warps) + " warps: issue count");
        for (std::size_t i = 0; i < issues.size(); ++i) {
            const auto [cycle, warp, pc] = issues[i];
            const uint64_t k = pc / 8;
            require(cycle == nanosm::fetch_to_issue + i && warp == i % warps &&
                    cycle == nanosm::fetch_to_issue + warp + warps * k,
                std::to_string(warps) + " warps: issue " + std::to_string(i) + " at c" + std::to_string(cycle) +
                    " warp " + std::to_string(warp) + " pc " + std::to_string(pc));
        }
        const auto fetches = i6b2_fetches(*machine);
        for (std::size_t i = 0; i < 2u * warps; ++i) {
            const auto [cycle, warp, pc] = fetches.at(i);
            require(cycle == i && warp == i % warps && pc == 16 * (i / warps),
                std::to_string(warps) + " warps: fetch " + std::to_string(i) + " was not round-robin");
        }
        for (uint32_t warp = 0; warp < warps; ++warp) require(machine->read_reg(warp, 0, 7) == 7, "mov data");
    }
}

// Fetch round-robin with a hot warp and three waiting warps (core.warps 16,
// launched 4). Warp 0 runs thirty independent movs; warps 1..3 sit on a
// stall count of 31 twice. Fetch ignores the waits: while the waiters' FIFOs
// have room the grants go round all four (the hot warp gets one fetch in
// four and issues at that rate); a waiter whose FIFO is full stops asking.
// Once all three are full the hot warp gets every fetch and, within
// fetch_to_issue of that, issues one per cycle until a waiter's stall ends.
// No priority policy: just the round-robin over warps with room.
// Two-word packets: before their branches the waiters' FIFOs are already
// full of their fall-through (prefetch runs through the setp -> bra.uni
// wait), so what counts is after the redirects: from the first waiter's
// first request (its branch's resolve + 1) to the first cycle after the
// last waiter's resolve in which all three are full. In that window every
// warp asks and each waiter is granted exactly its two packet entries, with
// the hot warp's turn between: at least 2 x 3 + 1 grants, in rotation.
void test_i6b2_fetch_round_robin_hot_and_waiting_warps() {
    const auto& profile = canonical_profile();
    const auto depth = profile.core_instruction_buffer_packets;
    auto machine = i6b2_machine(profile, R"(
        sread r29, WARP_ID
        setp.ne p0, r29, 0
        bra.uni waiter {pred=p0, stall=0, set=-, wait=-}
    )" + i6b2_independent_movs(30) + R"(
        halt
    waiter: mov r20, 1 {pred=PT, stall=31, set=-, wait=-}
        mov r21, 2 {pred=PT, stall=31, set=-, wait=-}
        mov r22, 3
        halt
    )", 4, true);
    std::map<uint64_t, bool> waiters_full; // per cycle (Q): every waiter's FIFO full
    while (!machine->done()) {
        waiters_full[machine->cycle()] = machine->instruction_fifo_packets(1) == depth &&
            machine->instruction_fifo_packets(2) == depth && machine->instruction_fifo_packets(3) == depth;
        tick_capped(*machine);
    }
    const auto fetches = i6b2_fetches(*machine);
    const auto issues = i6b2_issues(*machine);
    // The waiters' branches (pc 16), their first stalled mov (pc 34 * 8:
    // after sread, setp, bra.uni, thirty movs and the halt) and their second.
    uint64_t first_branch = std::numeric_limits<uint64_t>::max();
    uint64_t last_branch = 0;
    uint64_t waiter_start = 0;
    uint64_t waiter_resume = std::numeric_limits<uint64_t>::max();
    for (const auto& [cycle, warp, pc] : issues) {
        if (warp != 0 && pc == 16) {
            first_branch = std::min(first_branch, cycle);
            last_branch = std::max(last_branch, cycle);
        }
        if (warp == 3 && pc == 34 * 8) waiter_start = cycle;
        if (warp != 0 && pc == 35 * 8) waiter_resume = std::min(waiter_resume, cycle);
    }
    require(waiter_start != 0 && waiter_resume != std::numeric_limits<uint64_t>::max(), "the waiters did not run");
    std::optional<uint64_t> full;
    for (const auto& [cycle, all] : waiters_full) {
        if (!full.has_value() && all && cycle > last_branch + nanosm::issue_to_branch_resolve) full = cycle;
    }
    require(full.has_value(), "the waiters' FIFOs never filled after their branches");
    // While all four ask, the grants rotate w0, w1, w2, w3.
    std::vector<uint32_t> rotation;
    for (const auto& [cycle, warp, pc] : fetches) {
        if (cycle > first_branch + nanosm::issue_to_branch_resolve && cycle < *full) rotation.push_back(warp);
    }
    require(rotation.size() >= 2 * 3 + 1, "the window where all four warps ask is too short to show the rotation");
    for (std::size_t i = 1; i < rotation.size(); ++i) {
        require(rotation[i] == (rotation[i - 1] + 1) % 4, "fetch did not rotate over the warps with room");
    }
    // From then until the waiters ask again (their stall ends), every fetch
    // is the hot warp's, and it issues every cycle from full + 1.
    for (const auto& [cycle, warp, pc] : fetches) {
        if (cycle >= *full && cycle < waiter_resume) require(warp == 0, "a full waiter was granted a fetch");
    }
    std::vector<uint64_t> hot;
    for (const auto& [cycle, warp, pc] : issues) {
        if (warp == 0 && cycle > *full && cycle < waiter_resume) hot.push_back(cycle);
    }
    require(hot.size() >= 10 && hot.front() <= *full + nanosm::fetch_to_issue,
        "the hot warp did not issue within fetch_to_issue of the waiters' FIFOs filling");
    for (std::size_t i = 1; i < hot.size(); ++i) {
        require(hot[i] == hot[i - 1] + 1, "the hot warp lost a cycle while the waiters were full:" + issue_log(*machine));
    }
    for (uint32_t warp = 1; warp < 4; ++warp) require(machine->read_reg(warp, 0, 22) == 3, "waiter data");
    require(machine->read_reg(0, 0, 13) == 29, "hot warp data");
}

// Conditional branches (HUMAN 10-18: the predicate is read one cycle after
// issue), with two-word packets. Words: 0 setp, 1..5 movs, 6 bra.uni (the
// first half of packet 48), 7 mov r6 (its second half), 8 mov r7 and 9 the
// target mov r8 (packet 64: the target is a second half), 10 halt. A lone
// warp issues one word per cycle from c2 (two packet entries), so the
// bra.uni issues at t = c8. Its fetches: packet 0 at c0, 16 at c1, then each
// packet the cycle after the one two ahead pops (32 at c4, 48 at c6, 64 at
// c8 = t, granted from registered state as the bra.uni issues). So at t + 1
// (Q after t): packet 48 in the FIFO with pc at its second half (56),
// packet 64 in flight, fetch pc 80. The branch resolves at t + 1:
//   taken:      the FIFO (packet 48, whose second half was never issued) is
//               cleared and packet 64 landing at t + 1 is dropped -- a
//               redirect and an old-path landing in one cycle, the redirect
//               wins -- although it holds the target: pc takes the target
//               (72) and fetch pc its packet (64), which is read again at
//               t + 2 and its second half issues at t + 4
//               (branch_taken_interval). Its first half (mov r7) never
//               issues, nor does the cleared mov r6.
//   not taken:  the landing joins the kept packet 48; pc + 8 (its second
//               half) issues at t + 2 (branch_not_taken_interval) and packet
//               64's first half at t + 3; nothing is fetched again.
// Nothing of the warp issues or is fetched at t + 1.
void test_i6b2_conditional_branch() {
    const auto& profile = canonical_profile();
    for (const bool taken : {true, false}) {
        const auto what = std::string(taken ? "taken: " : "not taken: ");
        auto machine = i6b2_machine(profile, std::string("setp.eq p0, r0, ") + (taken ? "0" : "1") + R"(
            mov r1, 1
            mov r2, 2
            mov r3, 3
            mov r4, 4
            mov r5, 5
            bra.uni target {pred=p0, stall=0, set=-, wait=-}
            mov r6, 6
            mov r7, 7
        target: mov r8, 8
            halt
        )");
        constexpr uint32_t bra_pc = 6 * 8, target_pc = 9 * 8, target_packet = target_pc / 16 * 16;
        // setp at c2 (its p0 in the file from c2 + 5); the movs keep the warp
        // busy, so the bra.uni issues at c8 and samples p0 at c9.
        const uint64_t t = nanosm::fetch_to_issue + 6;
        require(t + nanosm::issue_to_predicate_sample >=
                    nanosm::fetch_to_issue + nanosm::issue_to_predicate_visible(
                        nanosm::decode(nanosm::assemble_asm(i1_asm(profile, "setp.eq p0, r0, 0\nhalt"), profile).words[0]),
                        profile),
            "the scenario's bra.uni would sample p0 before the setp wrote it");
        std::map<uint64_t, std::tuple<std::size_t, std::optional<uint32_t>, uint32_t, uint32_t>> after; // Q after each tick
        while (!machine->done()) {
            const auto cycle = machine->cycle();
            tick_capped(*machine);
            after[cycle] = {machine->instruction_fifo_packets(0), machine->fetch_in_flight(), machine->warp_pc(0),
                machine->warp_fetch_pc(0)};
        }
        require(issued_at(*machine, bra_pc) == std::vector<uint64_t>{t}, what + "bra.uni issue:" + issue_log(*machine));
        // At t + 1 (Q after t): packet 48 in the FIFO, packet 64 in flight.
        require(after.at(t) == std::tuple<std::size_t, std::optional<uint32_t>, uint32_t, uint32_t>{
                    1, 0u, bra_pc + 8, bra_pc + 32},
            what + "the FIFO and the read in flight at the resolve");
        const auto fetched = i6b2_fetches(*machine);
        const auto fetches_of = [&](uint32_t packet) {
            std::vector<uint64_t> cycles;
            for (const auto& [cycle, warp, fetched_pc] : fetched) if (fetched_pc == packet) cycles.push_back(cycle);
            return cycles;
        };
        for (const auto& [cycle, warp, pc] : fetched) {
            require(cycle != t + nanosm::issue_to_branch_resolve, what + "a fetch was granted while the branch was unresolved");
        }
        if (taken) {
            require(after.at(t + 1) == std::tuple<std::size_t, std::optional<uint32_t>, uint32_t, uint32_t>{
                        0, std::nullopt, target_pc, target_packet},
                what + "the redirect did not clear the FIFO and the read in flight, or set the pcs");
            require(issued_at(*machine, target_pc) == std::vector<uint64_t>{t + nanosm::branch_taken_interval} &&
                    fetches_of(target_packet) ==
                        std::vector<uint64_t>{t, t + nanosm::branch_taken_interval - nanosm::fetch_to_issue},
                what + "the target's packet was not dropped at t + 1 and read again at t + 2, its second half "
                       "issuing at t + 4:" + issue_log(*machine));
            require(issued_at(*machine, bra_pc + 8).empty() && issued_at(*machine, bra_pc + 16).empty() &&
                    machine->read_reg(0, 0, 6) == 0 && machine->read_reg(0, 0, 7) == 0,
                what + "an old-path word, or the first half of the target's packet, had an effect");
        } else {
            require(after.at(t + 1) == std::tuple<std::size_t, std::optional<uint32_t>, uint32_t, uint32_t>{
                        2, std::nullopt, bra_pc + 8, bra_pc + 32},
                what + "the buffered fall-through was not kept (with the packet landing at the resolve)");
            require(issued_at(*machine, bra_pc + 8) == std::vector<uint64_t>{t + nanosm::branch_not_taken_interval} &&
                    issued_at(*machine, bra_pc + 16) == std::vector<uint64_t>{t + nanosm::branch_not_taken_interval + 1},
                what + "the fall-through did not issue at t + 2:" + issue_log(*machine));
            require(fetches_of(bra_pc).size() == 1 && fetches_of(target_packet).size() == 1,
                what + "a kept fall-through packet was fetched again");
            require(machine->read_reg(0, 0, 6) == 6 && machine->read_reg(0, 0, 7) == 7, what + "fall-through data");
        }
        require(machine->read_reg(0, 0, 8) == 8, what + "target data");
    }
}

// The other branch forms resolve at the same point (implementer choice: one
// rule for every branch, one target adder): bra and bra.uni @PT are taken
// (the next instruction at t + branch_taken_interval), bra.uni @!PT is not
// (t + branch_not_taken_interval).
void test_i6b2_unconditional_branches() {
    const auto& profile = canonical_profile();
    for (const auto& [form, taken] : std::vector<std::pair<std::string, bool>>{
             {"bra target", true}, {"bra.uni target", true}, {"bra.uni target {pred=!PT, stall=0, set=-, wait=-}", false}}) {
        auto machine = i6b2_machine(profile, R"(
            mov r1, 1
            mov r2, 2
            mov r3, 3
        )" + form + R"(
            mov r4, 4
        target: mov r5, 5
            halt
        )");
        machine->run();
        const uint64_t t = nanosm::fetch_to_issue + 3;
        require(issued_at(*machine, 3 * 8) == std::vector<uint64_t>{t}, form + ": the branch did not issue at c5");
        const uint32_t next_pc = taken ? 5 * 8 : 4 * 8;
        const auto interval = taken ? nanosm::branch_taken_interval : nanosm::branch_not_taken_interval;
        require(issued_at(*machine, next_pc) == std::vector<uint64_t>{t + interval},
            form + ": the next instruction did not issue at t + " + std::to_string(interval) + ":" + issue_log(*machine));
        require(machine->read_reg(0, 0, 4) == (taken ? 0u : 4u) && machine->read_reg(0, 0, 5) == 5, form + ": data");
    }
}

// Fetch runs ahead of issue: past a taken branch, past the kernel's last
// word. Such words have no effect unless one reaches issue. The kernel's
// last word is a taken branch: the words after it, past the end, are
// fetched (the memory reads zeros there) and cleared by its redirect; the
// halt after `bra skip` is fetched as its fall-through and cleared too (if
// it had issued, r2 would be 0). A warp that really runs past its last word
// is reported when that word reaches issue: "fell off kernel", nothing
// earlier.
void test_i6b2_words_past_the_end() {
    const auto& profile = canonical_profile();
    auto machine = i6b2_machine(profile, R"(
            mov r1, 7
            bra skip
        done: halt
        skip: mov r2, 5
            bra done
    )");
    machine->run();
    require(machine->read_reg(0, 0, 1) == 7 && machine->read_reg(0, 0, 2) == 5, "a cleared word had an effect");
    bool past_the_end = false;
    for (const auto& [cycle, warp, pc] : i6b2_fetches(*machine)) past_the_end = past_the_end || pc >= 5 * 8;
    require(past_the_end, "no word past the kernel's end was fetched; the test proves little");
    require(issued_at(*machine, 2 * 8).size() == 1, "the halt did not issue exactly once (from the second branch)");
    require(machine->trace().summary().fetches_discarded >= 3, "the cleared words were not counted");

    auto falls = i6b2_machine(profile, "mov r1, 7\nmov r2, 8\n");
    try {
        falls->run();
        throw std::runtime_error("a warp ran past its last word without a report");
    } catch (const nanosm::SimError& error) {
        require(std::string(error.what()).find("warp 0 fell off kernel at pc 16") != std::string::npos,
            std::string("wrong fall-off report: ") + error.what());
    }
    // The word past the end was fetched at c2 and would issue at c4: the
    // report comes then, not at its fetch.
    require(falls->cycle() == nanosm::fetch_to_issue + 2, "the fall-off was not reported as the word reached issue");
}

// Prefetch goes on while the warp waits: through a stall count, a depbar
// wait and a refused head, the FIFO fills (both packet entries, Q: the
// waiting word's packet and the next), so when the wait ends the waiting
// instruction and the next two issue on consecutive cycles, and the warp is
// never "waiting for instruction" after its first two cycles.
void test_i6b2_prefetch_through_waits() {
    const auto check = [](nanosm::Machine& machine, uint32_t waiting_pc, uint64_t release, const std::string& what) {
        const auto depth = canonical_profile().core_instruction_buffer_packets;
        std::map<uint64_t, std::size_t> packets; // the FIFO's valid packets (Q) in each cycle
        while (!machine.done()) {
            packets[machine.cycle()] = machine.instruction_fifo_packets(0);
            tick_capped(machine);
        }
        require(packets.at(release) == depth, what + ": the FIFO was not full when the wait ended");
        for (uint32_t i = 0; i < 3; ++i) {
            require(issued_at(machine, waiting_pc + 8 * i) == std::vector<uint64_t>{release + i},
                what + ": instruction " + std::to_string(i) + " after the wait did not issue at release + " +
                    std::to_string(i) + ":" + issue_log(machine));
        }
        require(machine.trace().summary().warp_cycles[static_cast<std::size_t>(nanosm::WarpState::no_instruction)] ==
                nanosm::fetch_to_issue,
            what + ": the warp waited for an instruction after launch");
    };
    const auto& profile = canonical_profile();
    {   // A stall count of 12 on the first mov (c2): the next issues at c14.
        auto machine = i6b2_machine(profile, "mov r1, 1 {pred=PT, stall=12, set=-, wait=-}\n" +
            i6b2_independent_movs(4) + "halt\n");
        check(*machine, 8, nanosm::fetch_to_issue + 12, "stall count");
    }
    {   // A depbar: the lds.uni at c2 completes at c2 + dispatch + 1 + D + Lb
        // + 1 + Wd (its RF write lands); the waiting mov sees the counter at 0
        // the cycle after.
        auto machine = i6b2_machine(profile, R"(
            lds.uni r20, [r21] {pred=PT, stall=0, set=sb0, wait=-}
            mov r1, 1 {pred=PT, stall=0, set=-, wait=sb0}
        )" + i6b2_independent_movs(3) + "halt\n");
        const auto release = nanosm::fetch_to_issue + nanosm::issue_to_dispatch(profile) + 1 +
            profile.shmem_port_a_depth + profile.shmem_read_latency_cycles + 1 + profile.rf_write_depth_cycles + 1;
        check(*machine, 8, release, "depbar");
    }
    {   // A refused head: one load slot; the second lds.uni (c3) is refused
        // until the first's RF write frees the slot (the I1 timeline: at its
        // grant, 1 + i1_lds_shift cycles after the refusal starts).
        auto one = profile;
        one.shmem_load_slots = 1;
        auto machine = i6b2_machine(one, R"(
            lds.uni r20, [r21] {pred=PT, stall=0, set=sb0, wait=-}
            lds.uni r22, [r21 + 4] {pred=PT, stall=0, set=sb1, wait=-}
        )" + i6b2_independent_movs(3) + "halt {pred=PT, stall=0, set=-, wait=sb0|sb1}\n");
        const auto release = nanosm::fetch_to_issue + 1 + 1 + i1_lds_shift(one);
        check(*machine, 8, release, "refused head");
        require(refusals(*machine, 0, "load slots full") ==
                std::vector<std::pair<uint64_t, uint64_t>>{{nanosm::fetch_to_issue + 1, 1 + i1_lds_shift(one)}},
            "refused head: the second lds.uni was not refused from c3 until the slot freed");
    }
}

// setp -> bra.uni (src/rf.hpp): a conditional branch samples its predicate
// issue_to_predicate_sample after its issue, so it may issue that much
// before the setp's predicate is in the file: issue_to_predicate_visible -
// issue_to_predicate_sample after the setp (4 at canonical; 5 before the
// front end). qtxc, the hand assembler's stall scheduling and the sanitizer
// all use that distance. At it, the branch sees the new predicate; one cycle
// less is a sanitizer PANIC (without the sanitizer the machine quietly reads
// the old predicate).
void test_i6b2_setp_to_branch_distance() {
    const auto& profile = canonical_profile();
    const auto setp = nanosm::decode(nanosm::assemble_asm(i1_asm(profile, "setp.eq p0, r0, r0\nhalt"), profile).words[0]);
    const auto distance = nanosm::issue_to_predicate_visible(setp, profile) - nanosm::issue_to_predicate_sample;
    require(distance == 1 + profile.rf_read_latency_cycles + profile.timing.alu + 1 - 1 && distance == 4,
        "the canonical setp -> bra.uni distance is not 4");
    const auto qtx = nanosm::compile_qtx("setp.eq p0, r0, r0\n@p0 bra.uni target\nmov r5, 1\ntarget: halt\n", profile);
    require(qtx.instructions[0].control.stall == distance, "qtxc did not place the bra.uni at the distance");
    const auto program = [&](uint32_t stall) {
        return "setp.eq p0, r0, r0 {pred=PT, stall=" + std::to_string(stall) + R"(, set=-, wait=-}
            bra.uni target {pred=p0, stall=0, set=-, wait=-}
            mov r5, 1
            halt
        target: mov r5, 2
            halt
        )";
    };
    const auto scheduled = hand_assemble(i1_asm(profile, program(0)), profile);
    require(nanosm::decode(scheduled.words[0]).control.stall == distance,
        "the hand assembler's stall scheduling did not raise the setp's stall to the distance");
    auto at = i6b2_machine(profile, program(static_cast<uint32_t>(distance)));
    at->run();
    require(at->read_reg(0, 0, 5) == 2, "at the distance the bra.uni did not see the new predicate");
    auto early = i6b2_machine(profile, program(static_cast<uint32_t>(distance - 1)));
#if NANOSM_SANITIZER
    require_mentions(i4_panic(*early, "one cycle early"), {"predicate RAW hazard: missing stall for p0"},
        "one cycle early");
#else
    early->run();
    require(early->read_reg(0, 0, 5) == 1, "one cycle early, the machine did not read the old predicate");
#endif
}

// core.warps (16 at canonical) declares the SM's warp slots and is enforced
// at launch, before the RF rows: a one-register kernel launches 16 warps
// and is refused at 17 and 32 (before I6b-2 the RF alone decided, and
// accepted both). core.instruction_buffer_packets (2) is the FIFO depth in
// two-word packets. Both are optional in the JSON (absent: 16, 2), checked
// (warps 1..64, packets 1..16) and part of the fingerprint.
void test_i6b2_warp_slots_and_profile_fields() {
    const auto& profile = canonical_profile();
    require(profile.core_warps == 16 && profile.core_instruction_buffer_packets == 2,
        "the canonical profile does not declare 16 warps and 2 instruction packets");
    const auto kernel = compile("mov r0, 1\nhalt", profile);
    for (const uint32_t warps : {16u, 17u, 32u}) {
        nanosm::Launch launch;
        launch.active_warps = warps;
        try {
            nanosm::Machine machine(profile, kernel, std::move(launch));
            require(warps == 16, std::to_string(warps) + " warps were accepted on 16 warp slots");
            machine.run();
        } catch (const nanosm::SimError& error) {
            require(warps != 16 && std::string(error.what()).find("launch of " + std::to_string(warps) +
                    " warps exceeds the SM's 16 warp slots (core.warps)") != std::string::npos,
                std::string("warp slots: ") + error.what());
        }
    }
    std::ifstream input("profiles/nanosm-v0.json");
    nlohmann::json document;
    input >> document;
    for (const auto& [field, canonical, bad] : std::vector<std::tuple<std::string, uint32_t, std::vector<int>>>{
             {"warps", 16, {0, 65, -1}}, {"instruction_buffer_packets", 2, {0, 17, -1}}}) {
        auto defaulted = document;
        defaulted["core"].erase(field);
        require(nanosm::device_profile_fingerprint(nanosm::parse_device_profile(defaulted.dump())) ==
                nanosm::device_profile_fingerprint(profile),
            field + ": the default is not the canonical " + std::to_string(canonical));
        for (const auto value : bad) {
            auto modified = document;
            modified["core"][field] = value;
            try {
                (void)nanosm::parse_device_profile(modified.dump());
                throw std::runtime_error(field + " = " + std::to_string(value) + " was accepted");
            } catch (const nanosm::ProfileError&) {}
        }
        auto other = document;
        other["core"][field] = canonical + 1;
        require(nanosm::device_profile_fingerprint(nanosm::parse_device_profile(other.dump())) !=
                nanosm::device_profile_fingerprint(profile),
            field + " was omitted from the profile fingerprint");
    }
    auto zero = profile;
    zero.core_instruction_buffer_packets = 0;
    try {
        nanosm::validate_device_profile(zero);
        throw std::runtime_error("a zero-packet instruction FIFO passed validation");
    } catch (const nanosm::ProfileError&) {}
}

// The trace's instruction-supply state: "waiting for instruction" -- detail
// "instruction FIFO empty" or "branch unresolved" -- apart from the
// dependency and resource waits, counted in the summary's warp cycles and
// its front-end line (fetches, discarded words, those warp-cycles, and the
// FIFO-empty observation). A taken branch at t = c8 (the conditional-branch
// scenario): launch c0..c1 FIFO empty, t + 1 branch unresolved, t + 2 .. t + 3
// FIFO empty: 5 warp-cycles waiting. The FIFO itself is empty at c0, c1,
// t + 2, t + 3 but not at t + 1 (packet 48 is cleared at t + 1's edge): 4.
// Words discarded: at the redirect, packet 48's unissued second half (1) and
// the landing packet 64 (2); at the halt (the first half of packet 80, at
// t + 5) its unissued second half (1), and packet 96, granted in the halt's
// cycle and dropped as it lands (2): 6.
void test_i6b2_trace_supply_state() {
    const auto& profile = canonical_profile();
    auto machine = i6b2_machine(profile, R"(
        setp.eq p0, r0, 0
        mov r1, 1
        mov r2, 2
        mov r3, 3
        mov r4, 4
        mov r5, 5
        bra.uni target {pred=p0, stall=0, set=-, wait=-}
        mov r6, 6
        mov r7, 7
    target: mov r8, 8
        halt
    )");
    machine->run();
    const uint64_t t = nanosm::fetch_to_issue + 6;
    std::vector<std::tuple<uint64_t, uint64_t, std::string>> waits;
    for (const auto& event : machine->trace().events()) {
        if (event.category == nanosm::TraceCategory::warp && event.kind == nanosm::TraceEventKind::slice &&
            event.name == "waiting for instruction") {
            waits.emplace_back(event.start_cycle, event.duration_cycles, event.detail);
        }
    }
    const std::vector<std::tuple<uint64_t, uint64_t, std::string>> expected{
        {0, nanosm::fetch_to_issue, "instruction FIFO empty"},
        {t + 1, 1, "branch unresolved"},
        {t + 2, nanosm::fetch_to_issue, "instruction FIFO empty"}};
    require(waits == expected, "the supply waits were not traced as launch, branch, refill");
    const auto& summary = machine->trace().summary();
    const auto supply = summary.warp_cycles[static_cast<std::size_t>(nanosm::WarpState::no_instruction)];
    require(supply == 5 && summary.instruction_fifo_empty == 4 && summary.fetches_discarded == 6,
        "the supply counters: waiting " + std::to_string(supply) + ", FIFO empty " +
            std::to_string(summary.instruction_fifo_empty) + ", discarded " +
            std::to_string(summary.fetches_discarded));
    require(nanosm::warp_state_name(nanosm::WarpState::no_instruction) == "waiting for instruction",
        "the state's name");
    const auto text = machine->trace().summary_text();
    require(text.find("  waiting for instruction    5\n") != std::string::npos &&
            text.find("front end: fetches=" + std::to_string(summary.fetches) +
                " discarded=6 waiting_for_instruction=5 fifo_empty=4\n") != std::string::npos,
        "the summary text lacks the supply state or the front-end line:\n" + text);
}

// Every warp's trace state comes from its own registers, however far the
// scheduler scanned (Codex's reproducer: two warps, mov, mov, halt). Warp 0's
// first packet is fetched at c0 and warp 1's at c1; warp 0 issues at c2,
// warp 1 at c3, then they alternate. At c2 the scheduler stops at warp 0,
// the winner, but warp 1's FIFO is empty (its packet lands at c2's edge): it
// is "waiting for instruction" at c0, c1 and c2, warp 0 at c0 and c1 -- 5
// warp-cycles, and the FIFO-empty observation is 5 too. A warp past the
// winner with an instruction is "not examined (another warp issued)", never
// "ready": warp 0 at c3 and c5, warp 1 at c4 and c6 (warp 0's halt at c6,
// warp 1's at c7).
void test_i6b2_trace_counts_every_warp() {
    const auto& profile = canonical_profile();
    auto machine = i6b2_machine(profile, "mov r0, 1\nmov r1, 2\nhalt\n", 2);
    machine->run();
    std::map<uint32_t, std::vector<std::pair<uint64_t, uint64_t>>> waits;
    std::map<uint32_t, std::vector<uint64_t>> not_examined;
    for (const auto& event : machine->trace().events()) {
        if (event.category != nanosm::TraceCategory::warp || event.kind != nanosm::TraceEventKind::slice) continue;
        const auto warp = static_cast<uint32_t>(event.track - 100);
        if (event.name == "waiting for instruction") waits[warp].emplace_back(event.start_cycle, event.duration_cycles);
        if (event.name == "not examined (another warp issued)") {
            for (uint64_t c = 0; c < event.duration_cycles; ++c) not_examined[warp].push_back(event.start_cycle + c);
        }
    }
    require(waits[0] == std::vector<std::pair<uint64_t, uint64_t>>{{0, 2}} &&
            waits[1] == std::vector<std::pair<uint64_t, uint64_t>>{{0, 3}},
        "warp 1's empty FIFO at c2 (past the winner) was not traced as waiting for instruction");
    require(not_examined[0] == std::vector<uint64_t>{3, 5} && not_examined[1] == std::vector<uint64_t>{4, 6},
        "the warps past the winner were not traced as not examined:" + issue_log(*machine));
    const auto& summary = machine->trace().summary();
    require(summary.warp_cycles[static_cast<std::size_t>(nanosm::WarpState::no_instruction)] == 5 &&
            summary.instruction_fifo_empty == 5 &&
            summary.warp_cycles[static_cast<std::size_t>(nanosm::WarpState::not_examined)] == 4,
        "the summary undercounts the warps past the winner");
    require(machine->read_reg(1, 0, 1) == 2, "data");
}

// Two warps, the same code, alternating one issue per cycle from c2 (warp w's
// k-th word at c(2 + w + 2k)). Words: 0..3 movs, 4 'bra target' (the first
// half of packet 32), 5 mov r5, 6 mov r6 (packet 48's first half), 7 the
// target mov r7 (its second half), 8 halt. Fetch (two entries each): packet 0
// at c0 (w0) and c1 (w1), 16 at c2 and c3, 32 at c5 (w0, after its packet 0
// popped at c4) and c6 (w1), 48 for w0 at c9 (its packet 16 popped at c8;
// w1 is full), 48 for w1 at c10 (popped at c9; w0 is full, its packet 48 in
// flight). The branches issue on consecutive cycles: warp 0's at c10, warp
// 1's at c11. So:
//   - warp 0 redirects at c11 while warp 1's packet 48 is at the memory
//     output: warp 1 keeps it (FIFO 2 packets after c11), warp 0's FIFO is
//     cleared, its pc the target 56, its fetch pc packet 48;
//   - warp 1 redirects at c12, the next cycle (consecutive redirects of
//     different warps: each clears only its own front end);
//   - fetch: warp 0's packet 48 again at c12 (warp 1's branch is still in the
//     issue register), warp 1's at c13; the targets issue at c14 and c15,
//     each branch's t + 4 (branch_taken_interval).
void test_i6b2_redirects_across_warps() {
    const auto& profile = canonical_profile();
    auto machine = i6b2_machine(profile, R"(
            mov r1, 1
            mov r2, 2
            mov r3, 3
            mov r4, 4
            bra target
            mov r5, 5
            mov r6, 6
        target: mov r7, 7
            halt
    )", 2);
    std::map<uint64_t, std::tuple<std::optional<uint32_t>, std::size_t, std::size_t, uint32_t, uint32_t>> after;
    while (!machine->done()) {
        const auto cycle = machine->cycle();
        tick_capped(*machine);
        after[cycle] = {machine->fetch_in_flight(), machine->instruction_fifo_packets(0),
            machine->instruction_fifo_packets(1), machine->warp_pc(0), machine->warp_fetch_pc(0)};
    }
    constexpr uint32_t bra_pc = 4 * 8, target_pc = 7 * 8;
    const uint64_t t0 = nanosm::fetch_to_issue + 2 * 4, t1 = t0 + 1;
    const auto issues = i6b2_issues(*machine);
    const auto issued = [&](uint32_t warp, uint32_t pc) {
        std::vector<uint64_t> cycles;
        for (const auto& [cycle, w, issued_pc] : issues) if (w == warp && issued_pc == pc) cycles.push_back(cycle);
        return cycles;
    };
    require(issued(0, bra_pc) == std::vector<uint64_t>{t0} && issued(1, bra_pc) == std::vector<uint64_t>{t1},
        "the branches did not issue at c10 and c11:" + issue_log(*machine));
    require(std::get<0>(after.at(t0)) == std::optional<uint32_t>(1),
        "warp 1's packet was not at the memory output during warp 0's redirect");
    require(after.at(t1) == std::tuple<std::optional<uint32_t>, std::size_t, std::size_t, uint32_t, uint32_t>{
                std::nullopt, 0, 2, target_pc, target_pc / 16 * 16},
        "warp 0's redirect dropped warp 1's landing, or did not clear its own front end");
    const auto fetches = i6b2_fetches(*machine);
    const auto fetched = [&](uint32_t warp, uint32_t pc) {
        std::vector<uint64_t> cycles;
        for (const auto& [cycle, w, fetched_pc] : fetches) if (w == warp && fetched_pc == pc) cycles.push_back(cycle);
        return cycles;
    };
    require(fetched(0, 48) == std::vector<uint64_t>{t0 - 1, t1 + 1} && fetched(1, 48) == std::vector<uint64_t>{t0, t1 + 2},
        "the target packets were not read again at c12 (warp 0) and c13 (warp 1)");
    require(issued(0, target_pc) == std::vector<uint64_t>{t0 + nanosm::branch_taken_interval} &&
            issued(1, target_pc) == std::vector<uint64_t>{t1 + nanosm::branch_taken_interval},
        "the targets did not issue at each branch's t + 4:" + issue_log(*machine));
    for (uint32_t warp = 0; warp < 2; ++warp) {
        require(machine->read_reg(warp, 0, 4) == 4 && machine->read_reg(warp, 0, 5) == 0 &&
                machine->read_reg(warp, 0, 6) == 0 && machine->read_reg(warp, 0, 7) == 7,
            "warp " + std::to_string(warp) + ": an old-path word had an effect, or the target did not run");
    }
}

// A redirected warp refills behind the warps already asking. Sixteen warps
// start with 'bra target' (packet 0's first half; the target is packet 16's
// first half). Fetch round-robin gives warp w its packet 0 at cw; issue gives
// warp w its branch at c(2 + w) (each warp's head becomes valid just in time
// for its turn), which resolves at c(3 + w). Warp 0's redirect at c3 makes it
// ask from c4, but the cursor is past it: warps 4..15, still empty, get
// their packet 0 at c4..c15, and the cursor comes back to warp 0 at c16 (its
// target packet), so its target issues at c18 -- fifteen cycles after the
// resolve instead of branch_taken_interval's three. No priority: just the
// round-robin.
void test_i6b2_redirect_behind_many_fetchers() {
    const auto& profile = canonical_profile();
    auto machine = i6b2_machine(profile, R"(
            bra target
            mov r1, 1
        target: mov r2, 2
            halt
    )", 16);
    machine->run();
    const auto fetches = i6b2_fetches(*machine);
    for (uint32_t warp = 0; warp < 16; ++warp) {
        const auto [cycle, w, pc] = fetches.at(warp);
        require(cycle == warp && w == warp && pc == 0, "packet 0 fetches were not c0..c15, warps 0..15");
    }
    const auto [cycle16, warp16, pc16] = fetches.at(16);
    require(cycle16 == 16 && warp16 == 0 && pc16 == 16,
        "warp 0's target packet was not read when the round-robin came back to it at c16");
    std::vector<uint64_t> target_issue;
    for (const auto& [cycle, warp, pc] : i6b2_issues(*machine)) if (warp == 0 && pc == 16) target_issue.push_back(cycle);
    require(target_issue == std::vector<uint64_t>{16 + nanosm::fetch_to_issue},
        "warp 0's target did not issue at c18:" + issue_log(*machine));
    for (uint32_t warp = 0; warp < 16; ++warp) {
        require(machine->read_reg(warp, 0, 1) == 0 && machine->read_reg(warp, 0, 2) == 2,
            "warp " + std::to_string(warp) + ": data");
    }
}

// The front end's consistency check: while a warp runs, its fetch pc is pc
// rounded down to 16 plus 16 per packet held or in flight. Breaking it
// between ticks is caught at the next tick, in both builds.
void test_i6b2_front_end_consistency_check() {
    const auto& profile = canonical_profile();
    auto machine = i6b2_machine(profile, i6b2_independent_movs(8) + "halt\n");
    for (int i = 0; i < 3; ++i) tick_capped(*machine);
    machine->test_set_fetch_pc(0, machine->warp_fetch_pc(0) + 16);
    try {
        tick_capped(*machine);
        throw std::runtime_error("a broken fetch pc went unnoticed");
    } catch (const nanosm::SimError& error) {
        require(std::string(error.what()).find("warp 0's front end is inconsistent") != std::string::npos,
            std::string("wrong report: ") + error.what());
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
        {"RF read profile and one round", test_rf_read_profile_and_round},
        {"RF one-round FADD and HALT", test_rf_one_round_fadd_and_halt},
        {"RF one-round WAR and branch", test_rf_one_round_war_and_branch},
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
        {"MMA varying round service", test_mma_varying_round_service},
        {"I7: one eight-product group at the 4x4x8 boundary", test_i7_dot8_group_boundary},
        {"I7: hold A / stream B walker", test_i7_hold_a_stream_b_walker},
        {"I7: back-to-back MMA throughput", test_i7_back_to_back_throughput},
        {"sanitizer: sts into TMA store source", test_sanitizer_sts_into_tma_store_source},
        {"sanitizer: sts into MMA operand stage", test_sanitizer_sts_into_mma_operand_stage},
        {"sanitizer: lds from TMA load destination", test_sanitizer_lds_from_tma_load_destination},
        {"sanitizer: mbar RMW into owned range", test_sanitizer_mbar_rmw_into_owned_range},
        {"sanitizer: TMA/MMA accept conflicts", test_sanitizer_tma_and_mma_accept_conflicts},
        {"sanitizer: tmem ownership both directions", test_sanitizer_tmem_ownership_both_directions},
        {"sanitizer: double buffer and P tile clean", test_sanitizer_double_buffer_and_p_tile_are_clean},
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
        {"mbar wrong-phase completion", test_mbar_complete_wrong_phase_panics},
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
        {"plain lds of one word: 16 rounds", test_plain_lds_of_one_word_is_sixteen_rounds},
        {"bank conflict", test_bank_conflict_serializes},
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
        {"B3: lds.uni one round, one RF write", test_b3_lds_uni_one_round_one_write},
        {"B3: lds k lanes on one word", test_b3_lds_k_lanes_on_one_word},
        {"B3: rounds = max lanes per bank", test_b3_rounds_are_max_lanes_per_bank},
        {"B3: sts bank serves lanes in order", test_b3_sts_bank_serves_lanes_in_order},
        {"B3: lds.uni composes like lds", test_b3_lds_uni_composes_like_lds},
        {"B3: lds.uni nonuniform PANIC", test_b3_lds_uni_nonuniform_panics},
        {"B3: lds.uni alignment and bounds", test_b3_lds_uni_alignment_and_bounds},
        {"B3: lds.uni encoding", test_b3_lds_uni_encoding},
        {"uniformity: hardware uses lane 0", test_uniformity_uses_lane_zero},
        {"I1: lds issue ignores its own rounds (C8 probe A)", test_i1_lds_issue_ignores_its_rounds},
        {"sanitizer: issue capture before acceptance", test_sanitizer_issue_capture_before_acceptance},
        {"I1: refused attempts read nothing", test_i1_refused_attempts_read_nothing},
        {"I1: RF owed inhibits after a denial", test_i1_rf_owed_inhibits_after_denial},
        {"I1: RF owed permits one more booking", test_i1_rf_owed_one_more_booking},
        {"I1: RF inhibition is bank- and booking-local", test_i1_rf_inhibition_is_bank_and_booking_local},
        {"I1: RF round-robin among ready holders", test_i1_rf_round_robin_among_holders},
        {"I1: round_robin_grant", test_i1_round_robin_grant},
        {"I1: acc.ld pipeline freeze", test_i1_acc_ld_pipeline_freeze},
        {"I1: load slots", test_i1_load_slots},
        {"I1: load slots profile field", test_i1_load_slots_profile_field},
        {"I1.1: one LSU RF write per cycle", test_i1_lsu_one_rf_write_per_cycle},
        {"I2: same-warp RAW/WAR/WAW with reordered reads", test_i2_same_warp_order_with_reordered_reads},
        {"I2: divergence", test_i2_divergence},
        {"I2: store slots 1 and 2", test_i2_store_slots_one_and_two},
        {"I2: store slot reuse and done()", test_i2_store_slot_reuse_and_done},
        {"I2: A-read and A-write in one cycle", test_i2_a_read_and_a_write_same_cycle},
        {"I2: one round-robin over TMA reads and loads", test_i2_round_robin_tma_reads_and_loads},
        {"I2: TMA placement on the port-B write", test_i2_tma_placement_on_port_b},
        {"I3: store pause, two warps (was I2: RMW holds both A services)", test_i3_store_pause_two_warps},
        {"I2: profile fields D, Lb, store slots", test_i2_profile_fields},
        {"I2: READ_FIRST across services", test_i2_read_first_across_services},
        {"I2.1: RMW compute stage C", test_i21_rmw_compute_stage},
        {"I2.1: SIMT RMW admission reads registered state", test_i21_simt_rmw_admission_registered_state},
        {"I2.1: SIMT RMW behind own lds and sts", test_i21_simt_rmw_behind_own_lds_and_sts},
        {"I2.1: probe behind own sts", test_i21_probe_behind_own_sts},
        {"I2.1: stale probe after a wake", test_i21_stale_probe_after_wake},
        {"I2.1: RMW read behind an A-read transaction", test_i21_rmw_read_behind_a_read_transaction},
        {"I2.1: TMA store sector registers", test_i21_tma_store_sector_registers},
        {"I2.1: acc.ld read-ownership ends at the last tmem read", test_i21_acc_ld_read_ownership},
        {"I3: store pause sequence (D = 2, 3)", test_i3_store_pause_sequence},
        {"I3: reads during an RMW", test_i3_reads_during_an_rmw},
        {"I3: READ_FIRST at the RMW's write", test_i3_read_first_at_rmw_write},
        {"I3: RMW / store alternation", test_i3_rmw_store_alternation},
        {"I3: SIMT RMW waits for its own stores", test_i3_simt_rmw_waits_for_own_stores},
        {"I3: RMW admission fairness", test_i3_rmw_admission_fairness},
        {"I3: signed tx in both orders", test_i3_signed_tx_both_orders},
        {"I3: TMA start reads no mbar", test_i3_tma_start_reads_no_mbar},
        {"I3: tx range PANICs", test_i3_tx_range_panics},
        {"I3: sanitizer wrong-phase complete_tx", test_i3_sanitizer_wrong_phase_complete_tx},
        {"I3: sanitizer nonzero tx at kernel end", test_i3_sanitizer_nonzero_tx_at_kernel_end},
        {"I4: fence.sh encoding, qtxc and hand assembly", test_i4_fence_encoding_qtxc_and_assembly},
        {"I4: fence.sh waits until its warp's accesses reach the RAM", test_i4_fence_waits_until_own_accesses_reach_ram},
        {"I4: missing fence.sh PANICs; fence.sh passes", test_i4_missing_fence_panics},
        {"I4: cross-warp race; the mbar handoff is clean", test_i4_cross_warp_race},
        {"I4: read/read overlap never PANICs", test_i4_read_read_overlap_is_clean},
        {"I4: C10 live-mbar opacity", test_i4_live_mbar_opacity},
        {"I5: atom.*.uni encoding, qtxc and hand assembly", test_i5_atomic_encoding_and_assembly},
        {"I5: one word, applied once; the timeline (D, C = 2, 3)", test_i5_one_word_timeline},
        {"I5: int32 add wraps at the boundary", test_i5_add_wraps_at_the_int32_boundary},
        {"I5: return slot admission", test_i5_return_slot_admission},
        {"I5: SIMT RMW register busy", test_i5_simt_rmw_register_busy},
        {"I5: store pause and resume", test_i5_store_pause_and_resume},
        {"I5: read after the store under way", test_i5_read_after_the_store_under_way},
        {"I5: same-warp order around an atomic", test_i5_same_warp_order},
        {"I5: launch-declared mbars (format; C11, C2)", test_i5_launch_declared_mbars_format},
        {"I5: atom.*.uni non-uniform address or operand PANIC", test_i5_atomic_nonuniform_operands},
        {"I5: atomic sanitizer: C10, race, missing fence", test_i5_atomic_sanitizer},
        {"D12: uniform ops under a partial mask", test_d12_uniform_ops_under_a_partial_mask},
        {"D12: sts with two lanes on one word", test_d12_sts_two_lanes_on_one_word},
        {"D12: stg with two lanes on one word", test_d12_stg_two_lanes_on_one_word},
        {"D12: sts over an mbar with a live waiter", test_d12_sts_over_an_mbar_with_a_live_waiter},
        {"D12.1: every uniform op under a partial mask", test_d12_1_every_uniform_op_under_a_partial_mask},
        {"D12.1: mbar.init over an mbar with a live waiter", test_d12_1_mbar_init_over_a_live_waiter},
        {"D12.1: TMA load over an mbar with a live waiter", test_d12_1_tma_load_over_a_live_waiter},
        {"I6a: the RF timing helper is the trace's (Lr/Wd 1/1, 2/2, 3/1)", test_i6a_helper_contract},
        {"I6a: dependent add at the qtxc distance; one less is a RAW", test_i6a_dependent_add_distance},
        {"I6a: WAW order", test_i6a_waw_order},
        {"I6a: async writer vs a fixed booking", test_i6a_async_writer_vs_fixed_booking},
        {"I6a: slots claimed while operands travel", test_i6a_slots_claimed_while_operands_travel},
        {"I6a: profile fields Lr, Wd; the stall-field bound", test_i6a_profile_fields},
        {"I6b-1: back-to-back same-bank reads (Lr 1, 2, 3)", test_i6b1_back_to_back_same_bank_reads},
        {"I6b-1: same-bank distinct sources are malformed", test_i6b1_same_bank_sources_are_malformed},
        {"I6b-1: an async reader gets the hole", test_i6b1_async_reader_gets_the_hole},
        {"I6b-1: async readers round-robin", test_i6b1_async_readers_round_robin},
        {"I6b-1: persistent readers under SIMT pressure", test_i6b1_persistent_readers_under_pressure},
        {"I6b-1: unrelated banks proceed", test_i6b1_unrelated_banks_proceed},
        {"I6b-1: 16 warps", test_i6b1_sixteen_warps},
        {"I6b-2: lone warp issue rate at 1-4 packets", test_i6b2_lone_warp_issue_rate},
        {"I6b-2: warps interleave (2, 16)", test_i6b2_warps_interleave},
        {"I6b-2: fetch round-robin, hot and waiting warps", test_i6b2_fetch_round_robin_hot_and_waiting_warps},
        {"I6b-2: conditional branch taken / not taken", test_i6b2_conditional_branch},
        {"I6b-2: unconditional branches", test_i6b2_unconditional_branches},
        {"I6b-2: words past the end and on a cleared path", test_i6b2_words_past_the_end},
        {"I6b-2: prefetch through waits", test_i6b2_prefetch_through_waits},
        {"I6b-2: setp -> bra.uni distance", test_i6b2_setp_to_branch_distance},
        {"I6b-2: warp slots and profile fields", test_i6b2_warp_slots_and_profile_fields},
        {"I6b-2: trace supply state", test_i6b2_trace_supply_state},
        {"I6b-2: trace counts every warp", test_i6b2_trace_counts_every_warp},
        {"I6b-2: redirects across warps", test_i6b2_redirects_across_warps},
        {"I6b-2: redirect behind many fetchers", test_i6b2_redirect_behind_many_fetchers},
        {"I6b-2: front-end consistency check", test_i6b2_front_end_consistency_check},
    };
    // Tests that deliberately provoke a sanitizer PANIC (src/sanitizer.hpp).
    // A NANOSM_SANITIZER=0 build compiles the sanitizer to no-ops, so it skips
    // and counts them; every other test must pass there unchanged.
    const std::set<std::string> expects_sanitizer_panic = {
        "tmem ordering and store lifetime",
        "MMA bare transfer race panics",
        "sanitizer: sts into TMA store source",
        "sanitizer: sts into MMA operand stage",
        "sanitizer: lds from TMA load destination",
        "sanitizer: mbar RMW into owned range",
        "sanitizer: TMA/MMA accept conflicts",
        "sanitizer: tmem ownership both directions",
        "lsu controller contract tripwire",
        "lsu B2.1: read_streams generalized",
        "missing wait panic",
        "reg: duplicate write PANIC",
        "sanitizer: assembly reader takes only valid lanes",
        "B3: lds.uni nonuniform PANIC",
        // I1: uniformity is a sanitizer rule; without it hardware uses lane 0.
        "nonuniform bra.uni PANIC",
        "mbar uniform address",
        "sanitizer: issue capture before acceptance",
        // I2.1 (F6): the races on the early side of the boundary are sanitizer PANICs.
        "I2.1: acc.ld read-ownership ends at the last tmem read",
        // I3 (rfc12 section 8): the phase a complete_tx was meant for and the
        // tx count at kernel end are sanitizer checks.
        "mbar wrong-phase completion",
        "I3: sanitizer wrong-phase complete_tx",
        "I3: sanitizer nonzero tx at kernel end",
        // I4 (rfc12 section 11): missing fence.sh / race and C10 are sanitizer checks.
        "I4: missing fence.sh PANICs; fence.sh passes",
        "I4: cross-warp race; the mbar handoff is clean",
        "I4: C10 live-mbar opacity",
        // I5: uniformity, C10 (live mbars, now also launch-declared ones),
        // the race and the missing fence are sanitizer checks. (The D12 tests
        // run in both builds: quiet machine without the sanitizer, its PANIC
        // with it.)
        "I5: atom.*.uni non-uniform address or operand PANIC",
        "I5: atomic sanitizer: C10, race, missing fence",
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

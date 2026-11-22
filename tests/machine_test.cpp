// Machine-level basics: nonblocking lds, depbars, bank rounds, profile and allocation errors, the trace.
// Each test's TEST wrapper is at the end.
#include "test_support.hpp"

namespace {

using namespace nanosm_test;

// A test that must write a file writes it here: the system temp directory
// plus this process's id and a per-process counter, so overlapping runs of
// this binary (ctest -j, the sanitizer ON/OFF twin, build and build_nosan at
// once) never share a path. The file is removed when the object goes away.
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

void test_lds_is_nonblocking_and_depbar_waits() {
    // Not `mov r0, 0`: all sixteen lanes on one word are sixteen
    // rounds (sixteen lanes on bank 0), which holds port A and would block the
    // second lds structurally. Lane-strided addresses keep each lds at one
    // round (one lane per bank), the premise of this test.
    const auto program = qtx_program(R"(
        sread r0, LANE_ID
        shli r0, r0, 2
        lds  r1, [r0]
        lds  r2, [r0 + 64]
        fadd r4, r1, r2
        halt
    )", canonical_profile());
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.test_shmem, 128);
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
    image_bytes(launch.test_shmem, 64);
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

// Sixteen lanes reading one word are not merged into one round.
// Uniformity is declared, never detected:
// they are sixteen lanes on bank 0, served one per round, 16 rounds. Every
// lane still gets the word. The one-round broadcast is lds.uni.
void test_plain_lds_of_one_word_is_sixteen_rounds() {
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.test_shmem, 4);
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
    auto& shmem = image_bytes(launch.test_shmem, 16 * 64);
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

// Every instruction reads its operands in ONE
// round, sampled at issue + issue_to_sample, so the profile has no read
// window; the canonical profile holds about as much RF as shmem (352 rows
// per bank). A round reads each distinct source once, in source order.
void test_rf_read_profile_and_round() {
    const auto& profile = canonical_profile();
    require(profile.rf_rows_per_bank == 352 &&
            uint64_t{profile.rf_banks} * profile.rf_rows_per_bank * profile.lanes * 4 == 66 * 1024,
        "the canonical RF is not 3 x 352 rows of 64 B (66 KiB)");
    require(nanosm::rf_read_round({0, 3}) == std::vector<uint8_t>{0, 3}, "a round lost a source");
    require(nanosm::rf_read_round({4, 4}) == std::vector<uint8_t>{4}, "a repeated register was read twice");
    require(nanosm::rf_read_round({5, 2, 5, 8}) == std::vector<uint8_t>{5, 2, 8}, "a round's order or broadcast");
    require(nanosm::issue_to_sample == nanosm::issue_register_stages &&
            nanosm::issue_to_operands(profile) == nanosm::issue_to_sample + profile.rf_read_latency_cycles &&
            nanosm::issue_to_dispatch(profile) == nanosm::issue_to_operands(profile),
        "every instruction's operands are not at the RF output issue + 1 + Lr");
}

} // namespace

// RF read profile and one round
TEST(Machine, rf_read_profile_and_round) {
    test_rf_read_profile_and_round();
}

// nonblocking lds
TEST(Machine, lds_is_nonblocking_and_depbar_waits) {
    test_lds_is_nonblocking_and_depbar_waits();
}

// missing wait panic
TEST(Machine, missing_wait_panics_instead_of_scoreboarding) {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    test_missing_wait_panics_instead_of_scoreboarding();
}

// plain lds of one word: 16 rounds
TEST(Machine, plain_lds_of_one_word_is_sixteen_rounds) {
    test_plain_lds_of_one_word_is_sixteen_rounds();
}

// bank conflict
TEST(Machine, bank_conflict_serializes) {
    test_bank_conflict_serializes();
}

// invalid profile defaults
TEST(Machine, profile_defaults_are_invalid) {
    test_profile_defaults_are_invalid();
}

// validate before allocation
TEST(Machine, profile_is_validated_before_machine_allocation) {
    test_profile_is_validated_before_machine_allocation();
}

// allocation failure PANIC
TEST(Machine, host_allocation_failure_is_a_panic) {
    test_host_allocation_failure_is_a_panic();
}

// trace attribution and limit
TEST(Machine, trace_attribution_and_limit) {
    test_trace_attribution_and_limit();
}

// perfetto async commands
TEST(Machine, perfetto_overlapping_commands_are_async) {
    test_perfetto_overlapping_commands_are_async();
}

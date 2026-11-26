// Machine-level basics: nonblocking lds, bank rounds, the device profile (validation, its optional fields, the
// stamp a kernel carries), the instruction words' reserved bits, the trace.
#include "test_support.hpp"

#include "image.hpp"

#include <iomanip>

namespace {

using namespace nanosm_test;

std::string replaced(std::string text, const std::string& from, const std::string& to) {
    const auto found = text.find(from);
    require(found != std::string::npos, "the text has no '" + from + "'");
    return text.replace(found, from.size(), to);
}

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

nlohmann::json canonical_document() {
    std::ifstream input("profiles/nanosm-v0.json");
    nlohmann::json document;
    input >> document;
    return document;
}

} // namespace

// Lane-strided addresses keep each lds at one round (one lane per bank), so
// port A does not block the second lds: it issues the cycle after the first,
// and the consumer waits on its depbar.
TEST(Machine, lds_is_nonblocking_and_depbar_waits) {
    const auto program = qtx_program(R"(
        sread r0, LANE_ID
        shli r0, r0, 2
        lds  r1, [r0]
        lds  r2, [r0 + 64]
        fadd r4, r1, r2
        halt
    )", canonical_profile());
    auto launch = launch_warps(1);
    auto& shmem = image_bytes(launch.test_shmem, 128);
    write_u32(shmem, 0, nanosm::float_bits(2.0f));
    write_u32(shmem, 64, nanosm::float_bits(3.0f));
    nanosm::Machine machine(canonical_profile(),
        nanosm::assemble_asm(nanosm::emit_asm(program), canonical_profile()), std::move(launch), {.timeline = true});
    machine.run();
    std::vector<uint64_t> load_issue_cycles;
    for (const auto& event : events_named(machine, "lds")) {
        if (event.category == nanosm::TraceCategory::instruction) load_issue_cycles.push_back(event.start_cycle);
    }
    require(load_issue_cycles.size() == 2, "did not issue two loads");
    require(load_issue_cycles[1] == load_issue_cycles[0] + 1, "lds blocked the warp instead of retiring");
    require(machine.trace().summary().warp_cycles[static_cast<std::size_t>(nanosm::WarpState::depbar)] != 0,
        "consumer never stalled on its depbar mask");
}

// Sixteen lanes on one bank take sixteen rounds, one lane a round, whether
// they read one word (uniformity is declared with lds.uni, never detected) or
// sixteen words 64 bytes apart; every lane gets its word.
TEST(Machine, bank_conflict_serializes) {
    struct Row {
        std::string what;
        std::string kernel;
        uint32_t stride; // bytes between the lanes' words (0: one word)
    };
    const std::vector<Row> rows = {
        {"sixteen lanes on one word", "mov r0, 0\nlds r1, [r0]\nhalt\n", 0},
        {"sixteen words on one bank", "sread r0, LANE_ID\nshli r0, r0, 6\nlds r1, [r0]\nhalt\n", 64},
    };
    for (const auto& row : rows) {
        auto launch = launch_warps(1);
        auto& shmem = image_bytes(launch.test_shmem, 16 * 64);
        const auto word = [&](uint32_t lane) { return row.stride == 0 ? 0x12345678u : 10 + lane; };
        for (uint32_t lane = 0; lane < 16; ++lane) write_u32(shmem, lane * row.stride, word(lane));
        nanosm::Machine machine(canonical_profile(), compile(row.kernel), std::move(launch));
        machine.run();
        for (uint32_t lane = 0; lane < 16; ++lane) {
            require(machine.read_reg(0, lane, 1) == word(lane),
                row.what + ": lane " + std::to_string(lane) + " has the wrong word");
        }
        require(machine.trace().summary().shmem_rounds == 16, row.what + ": not sixteen bank rounds");
    }
}

// A default-constructed profile carries no ISA timing and is refused; an
// invalid profile is refused before the machine allocates anything; an
// allocation the host cannot make is a PANIC; a key the parser does not read
// is refused by name.
TEST(Machine, profile_is_validated_before_machine_allocation) {
    nanosm::DeviceProfile defaults;
    require(defaults.timing.alu == 0 && defaults.timing.fadd == 0, "profile inherited ISA timing defaults");
    (void)error_of<nanosm::ProfileError>(
        [&] { nanosm::validate_device_profile(defaults); }, "a default-constructed profile");

    auto invalid = canonical_profile();
    invalid.timing.alu = 0;
    invalid.timing.fadd = 0;
    invalid.gmem_bytes = std::numeric_limits<uint64_t>::max();
    expect_error<nanosm::SimError>([&] { nanosm::Machine machine(invalid, compile("halt"), launch_warps(1)); },
        {"device profile"}, "an invalid profile at machine allocation");

    auto huge = canonical_profile();
    huge.gmem_bytes = std::numeric_limits<uint64_t>::max();
    const auto kernel = compile("halt", huge);
    const auto message = error_of<nanosm::SimError>(
        [&] { nanosm::Machine machine(huge, kernel, launch_warps(1)); }, "an impossible machine allocation");
    require(message.find("host memory") != std::string::npos || message.find("host container limits") != std::string::npos,
        "host allocation failure escaped deterministic PANIC conversion: " + message);

    auto top = canonical_document();
    top["bogus"] = 1;
    expect_error<nanosm::ProfileError>([&] { (void)nanosm::parse_device_profile(top.dump()); },
        {"unknown key 'bogus' at the top level"}, "an unknown top-level key");
    auto inner = canonical_document();
    inner["shmem"]["bogus_field"] = 1;
    expect_error<nanosm::ProfileError>([&] { (void)nanosm::parse_device_profile(inner.dump()); },
        {"unknown key 'bogus_field' in section 'shmem'"}, "an unknown key in a section");
    // A section present as anything but an object is refused by name, an
    // optional one (urf) as much as a required one (sfu).
    const std::vector<std::pair<const char*, nlohmann::json>> not_objects = {
        {"urf", nullptr},
        {"urf", 42},
        {"urf", nlohmann::json::parse(R"([{"registers": 4}])")},
        {"sfu", 42},
    };
    for (const auto& [section, value] : not_objects) {
        auto document = canonical_document();
        document[section] = value;
        expect_error<nanosm::ProfileError>([&] { (void)nanosm::parse_device_profile(document.dump()); },
            {std::string("section '") + section + "' must be an object"},
            std::string("section ") + section + " = " + value.dump());
    }
}

// Every optional profile field: the canonical JSON declares it, its default is
// the canonical value, its range is enforced, and it is in the fingerprint.
TEST(Machine, optional_profile_fields) {
    using P = nanosm::DeviceProfile;
    // A slower RF or URF read, or a slower uniform ALU, needs a longer urf
    // issue_to_port_cycles (the uniform write port's offset covers every writer).
    const auto later_uniform_port = [](P& profile) { profile.urf_issue_to_port_cycles = 4; };
    check_optional_field("rf", "read_latency_cycles", 2, &P::rf_read_latency_cycles, 1, 8, later_uniform_port);
    check_optional_field("rf", "write_depth_cycles", 2, &P::rf_write_depth_cycles, 1, 8);
    check_optional_field("urf", "registers", 32, &P::urf_registers, 1, 64);
    check_optional_field("urf", "read_latency_cycles", 1, &P::urf_read_latency_cycles, 1, 8, later_uniform_port);
    check_optional_field("urf", "write_depth_cycles", 1, &P::urf_write_depth_cycles, 1, 8);
    check_optional_field("urf", "alu_latency_cycles", 1, &P::urf_alu_latency_cycles, 1, 8, later_uniform_port);
    check_optional_field("urf", "issue_to_port_cycles", 3, &P::urf_issue_to_port_cycles, 1, 30);
    check_optional_field("shmem", "read_queue_entries", 4, &P::shmem_read_queue_entries, 1, 16);
    check_optional_field("shmem", "write_queue_entries", 4, &P::shmem_write_queue_entries, 1, 16);
    check_optional_field("shmem", "writeback_entries", 2, &P::shmem_writeback_entries, 1, 1024);
    check_optional_field("shmem", "port_a_depth", 2, &P::shmem_port_a_depth, 1, 64);
    check_optional_field("shmem", "read_latency_cycles", 2, &P::shmem_read_latency_cycles, 1, 64);
    check_optional_field("shmem", "rmw_compute_cycles", 2, &P::shmem_rmw_compute_cycles, 1, 64);
    check_optional_field("tmem", "port_depth", 2, &P::tmem_port_depth, 1, 64);
    check_optional_field("tmem", "read_latency_cycles", 2, &P::tmem_read_latency_cycles, 1, 64);
    check_optional_field("lsu", "queue_entries", 4, &P::lsu_queue_entries, 1, 16);
    check_optional_field("tma", "store_sector_registers", 1, &P::tma_store_sector_registers, 1, 64);
    check_optional_field("tma", "walker_start_cycles", 2, &P::tma_walker_start_cycles, 0, 8);
    check_optional_field("core", "warps", 16, &P::core_warps, 1, 64);
    check_optional_field("core", "instruction_buffer_packets", 2, &P::core_instruction_buffer_packets, 1, 16);
    check_optional_field("core", "divergence_stack_entries", 8, &P::core_divergence_stack_entries, 0, 64);
    check_optional_field("core", "launch_flags", 4, &P::core_launch_flags, 1, 6);
}

// A kernel carries the name and fingerprint of the profile it was scheduled
// for; the assembler and the machine refuse it under another profile.
TEST(Machine, profile_stamp_rejects_mismatch) {
    const auto& profile = canonical_profile();
    const auto assembly = nanosm::emit_asm(qtx_program("mov r0, 1\nhalt", profile));
    const auto kernel = nanosm::assemble_asm(assembly, profile);
    require(kernel.profile_name == profile.name, "profile name was not stamped into kernel");
    require(kernel.profile_fingerprint == nanosm::device_profile_fingerprint(profile),
        "profile fingerprint was not stamped into kernel");
    auto other = profile;
    ++other.timing.fadd;
    expect_error<nanosm::SimError>([&] { (void)nanosm::assemble_asm(assembly, other); }, {"does not match"},
        "assembly scheduled for another profile");
    expect_error<nanosm::SimError>([&] { nanosm::Machine machine(other, kernel, launch_warps(1)); },
        {"does not match simulator"}, "a kernel for another profile in the simulator");
}

// The stamp is also the instruction set's revision (src/isa.hpp
// isa_revision): a kernel stamped for another revision, in assembly, hex or a
// saved launch image, is refused under the same profile.
TEST(Machine, isa_revision_in_the_stamp) {
    const auto& profile = canonical_profile();
    const auto here = nanosm::device_profile_fingerprint(profile);
    const auto elsewhere = nanosm::device_profile_fingerprint(profile, nanosm::isa_revision + 1);
    require(here != elsewhere, "the revision is not in the fingerprint");
    const auto assembly = nanosm::emit_asm(qtx_program("mov r0, 1\nhalt", profile));
    const auto stamp = "0x" + nanosm::format_profile_fingerprint(here);
    const auto restamped_asm = replaced(assembly, stamp, "0x" + nanosm::format_profile_fingerprint(elsewhere));
    expect_error<nanosm::SimError>([&] { (void)nanosm::assemble_asm(restamped_asm, profile); },
        {"does not match", "instruction-set revision"}, "assembly for another revision");
    auto kernel = nanosm::assemble_asm(assembly, profile);
    kernel.profile_fingerprint = elsewhere;
    const auto from_hex = nanosm::load_hex(nanosm::emit_hex(kernel));
    expect_error<nanosm::SimError>([&] { nanosm::Machine machine(profile, from_hex, launch_warps(1)); },
        {"does not match simulator", "instruction-set revision"}, "hex for another revision");
    const auto image = nanosm::load_launch_image(nanosm::emit_launch_image({kernel, launch_warps(1)}));
    expect_error<nanosm::SimError>([&] { nanosm::Machine machine(profile, image.kernel, image.launch); },
        {"does not match simulator"}, "a launch image for another revision");
}

// A bit a word's form does not define (src/isa.hpp reserved_bit) is refused
// at kernel load, from hex and from a launch image, naming the pc and the
// bit. The bits each form does define are accepted.
TEST(Machine, reserved_bits_refused) {
    const auto& profile = canonical_profile();
    const auto kernel = compile("mov r0, 1\nex2 r1, r0\nhalt\n", profile);
    require(!nanosm::reserved_bit(kernel.words.at(1)).has_value(), "a clean ex2 has a reserved bit");
    auto stray = kernel;
    stray.words.at(1) |= 1;  // ex2 has no U bit: bit 0 is reserved
    const std::vector<std::string_view> needles = {"kernel pc 8", "sets bit 0, which its form does not define"};
    expect_error<nanosm::SimError>([&] { nanosm::Machine machine(profile, stray, launch_warps(1)); },
        {needles[0], needles[1]}, "a stray bit at kernel load");
    expect_error<nanosm::SimError>([&] { (void)nanosm::load_hex(nanosm::emit_hex(stray)); },
        {needles[0], needles[1]}, "a stray bit in hex");
    const auto clean_image = nanosm::emit_launch_image({kernel, launch_warps(1)});
    const auto hex_word = [](uint64_t word) {
        std::ostringstream out;
        out << "0x" << std::hex << std::setw(16) << std::setfill('0') << word;
        return out.str();
    };
    expect_error<nanosm::SimError>(
        [&] { (void)nanosm::load_launch_image(replaced(clean_image, hex_word(kernel.words.at(1)),
                  hex_word(stray.words.at(1)))); },
        {needles[0], needles[1]}, "a stray bit in a launch image");

    const auto word_of = [](nanosm::Opcode opcode) { return uint64_t{static_cast<uint8_t>(opcode)} << 56; };
    // Reserved: add's bits 1..13, fma's 1..5, mbar.arrive's second register
    // field, anything in a nop's payload.
    require(nanosm::reserved_bit(word_of(nanosm::Opcode::add) | 2) == 1u, "add bit 1");
    require(nanosm::reserved_bit(word_of(nanosm::Opcode::add) | (1u << 13)) == 13u, "add bit 13");
    require(nanosm::reserved_bit(word_of(nanosm::Opcode::fma) | (1u << 5)) == 5u, "fma bit 5");
    require(nanosm::reserved_bit(word_of(nanosm::Opcode::mbar_arrive) | (1u << 22)) == 22u, "mbar.arrive bit 22");
    require(nanosm::reserved_bit(word_of(nanosm::Opcode::nop) | (uint64_t{1} << 37)) == 37u, "nop bit 37");
    // Defined: the U bit, fma's addend, the mma.async spacings, umov's 32-bit
    // word, a tma's descriptor field, an atomic's offset, the control fields.
    const uint64_t payload = (uint64_t{1} << 38) - 1;
    for (const auto opcode : {nanosm::Opcode::umovi, nanosm::Opcode::tma_load_4d, nanosm::Opcode::atom_add_uni,
             nanosm::Opcode::addi, nanosm::Opcode::sts, nanosm::mma_opcode(0, false)}) {
        require(!nanosm::reserved_bit(word_of(opcode) | payload).has_value(),
            nanosm::opcode_name(opcode) + ": a defined bit taken for reserved");
    }
    require(!nanosm::reserved_bit(word_of(nanosm::Opcode::add) | 1 | (payload & ~uint64_t{0x3fff})).has_value(),
        "add's fields and U bit taken for reserved");
    require(!nanosm::reserved_bit(word_of(nanosm::Opcode::fma) | 1 | (payload & ~uint64_t{0x3e})).has_value(),
        "fma's fields and U bit taken for reserved");
    require(!nanosm::reserved_bit(word_of(nanosm::Opcode::halt) | (~uint64_t{0} >> 8 & ~payload)).has_value(),
        "a control field taken for reserved");
}

TEST(Machine, trace_attribution_and_limit) {
    nanosm::Machine machine(canonical_profile(), compile("mov r0, 1\nhalt"), launch_warps(1),
        {.timeline = true, .max_events = 1});
    machine.run();
    const auto& trace = machine.trace();
    const auto& summary = trace.summary();
    require(summary.scheduler_issued_cycles + summary.scheduler_idle_cycles + summary.scheduler_draining_cycles ==
            summary.cycles,
        "scheduler attribution does not sum to total cycles");
    uint64_t warp_cycles = 0;
    for (const auto cycles : summary.warp_cycles) warp_cycles += cycles;
    require(warp_cycles == summary.cycles, "warp attribution does not sum to cycles * warps");
    require(trace.events().size() == 1, "timeline exceeded its hard event limit");
    require(trace.truncated(), "timeline limit did not report truncation");
}

// Complete 'X' slices that partially overlap on one tid are ambiguous to nest
// and trigger Perfetto's slice_spill_overlapping_complete_event; overlapping
// lifetimes ship as async begin/end pairs instead.
TEST(Machine, perfetto_overlapping_commands_are_async) {
    auto launch = launch_warps(2);
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
    std::map<uint64_t, std::vector<std::pair<uint64_t, uint64_t>>> spans;
    std::size_t begins = 0;
    std::size_t ends = 0;
    for (const auto& event : document.at("traceEvents")) {
        const auto ph = event.at("ph").get<std::string>();
        if (ph == "b") ++begins;
        if (ph == "e") ++ends;
        if (ph != "X") continue;
        // The exact cycle integers from args: spans recomputed from the
        // floating-point ts/dur pair show 1-ULP overlaps that do not exist.
        const auto start = event.at("args").at("start_cycle").get<uint64_t>();
        const auto duration = event.at("args").at("duration_cycles").get<uint64_t>();
        spans[event.at("tid").get<uint64_t>()].emplace_back(start, start + duration);
    }
    for (auto& [tid, intervals] : spans) {
        std::sort(intervals.begin(), intervals.end());
        for (std::size_t i = 1; i < intervals.size(); ++i) {
            require(intervals[i].first >= intervals[i - 1].second, "overlapping complete slices on tid " + std::to_string(tid));
        }
    }
    require(begins >= 2 && begins == ends, "concurrent gmem commands did not emit async pairs");
}

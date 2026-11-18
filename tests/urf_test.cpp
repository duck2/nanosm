// The uniform register file (trash/rfc16.md, trash/urf-slice-2026-11-08.md):
// uniform arithmetic, r2ur / ur2r, usread, the consumers that read uniform
// registers, the fixed write latency L, the profile fields and the trace.
// Smoke tests: things run, the data is right, and the few timing facts the
// design rests on (every writer presents at issue + L, a dependent uniform
// reader issues L + Wu later, one write per cycle).
#include "test_support.hpp"

namespace {

using namespace nanosm_test;

// (cycle, warp) of every issue whose opcode name is `name`, in trace order.
std::vector<std::pair<uint64_t, uint32_t>> issues_of(const nanosm::Machine& machine, const std::string& name) {
    std::vector<std::pair<uint64_t, uint32_t>> result;
    for (const auto& event : machine.trace().events()) {
        if (!is_issue_event(event) || event.name != name) continue;
        const auto warp = std::stoul(event.detail.substr(5, event.detail.find(',') - 5));
        result.emplace_back(event.start_cycle, static_cast<uint32_t>(warp));
    }
    return result;
}

// (cycle, warp, register, value) of every "UR write" instant (a write at the
// URF write port), in trace order.
struct UrWrite {
    uint64_t cycle = 0;
    uint32_t warp = 0;
    uint64_t reg = 0;
    int64_t value = 0;
};

std::vector<UrWrite> ur_writes(const nanosm::Machine& machine) {
    std::vector<UrWrite> result;
    for (const auto& event : machine.trace().events()) {
        if (event.name != "UR write") continue;
        result.push_back({event.start_cycle, event.track - 100, event.id, event.value});
    }
    return result;
}

std::string compile_error(const std::string& source, const nanosm::DeviceProfile& profile = canonical_profile()) {
    try {
        (void)compile(source, profile);
    } catch (const nanosm::SimError& error) {
        return error.what();
    }
    return {};
}

// Every uniform ALU opcode, usread, r2ur (lane 0) and ur2r (every lane), on
// three warps: the values in the URF and in the vector copies.
void test_uniform_arithmetic_and_moves() {
    nanosm::Launch launch;
    launch.active_warps = 3;
    nanosm::Machine machine(canonical_profile(), compile(R"(
        usread ur0, WARP_ID
        usread ur1, N_WARPS
        umov  ur2, 1000
        uaddi ur3, ur2, -7
        uadd  ur4, ur3, ur0
        usub  ur5, ur2, ur3
        umov  ur6, 3
        ushl  ur7, ur2, ur6
        ushr  ur8, ur2, ur6
        ushli ur9, ur2, 4
        ushri ur10, ur2, 2
        uand  ur11, ur2, ur3
        uor   ur12, ur2, ur3
        uxor  ur13, ur2, ur3
        uandi ur14, ur2, 255
        uori  ur15, ur0, 64
        uxori ur6, ur6, 5
        umov  ur3, ur1
        sread r0, LANE_ID
        addi  r1, r0, 500
        r2ur  ur2, r1
        ur2r  r3, ur4
        ur2r  r4, ur2
        halt
    )"), std::move(launch));
    machine.run(20000);
    for (uint32_t warp = 0; warp < 3; ++warp) {
        const auto ur = [&](uint32_t reg) { return machine.read_ureg(warp, reg); };
        const std::string where = "warp " + std::to_string(warp) + ": ";
        require(ur(0) == warp, where + "usread WARP_ID");
        require(ur(1) == 3, where + "usread N_WARPS");
        require(ur(2) == 500, where + "r2ur takes lane 0");
        require(ur(3) == 3, where + "umov from a uniform register");
        require(ur(4) == 993 + warp, where + "uadd / uaddi with a negative immediate");
        require(ur(5) == 7, where + "usub");
        require(ur(6) == 6, where + "uxori");
        require(ur(7) == 8000 && ur(8) == 125, where + "ushl / ushr");
        require(ur(9) == 16000 && ur(10) == 250, where + "ushli / ushri");
        require(ur(11) == (1000u & 993u) && ur(12) == (1000u | 993u) && ur(13) == (1000u ^ 993u),
            where + "uand / uor / uxor");
        require(ur(14) == (1000u & 255u) && ur(15) == (warp | 64u), where + "uandi / uori");
        for (uint32_t lane = 0; lane < 16; ++lane) {
            require(machine.read_reg(warp, lane, 3) == 993 + warp && machine.read_reg(warp, lane, 4) == 500,
                where + "ur2r writes every lane");
        }
    }
}

// The timing facts (canonical: L 3, Wu 1): qtxc places a dependent uniform
// reader L + Wu = 4 after its writer and a vector reader of ur2r 6 after it
// (like mov); each write is presented at the URF write port at its issue + L.
// Four warps writing back to back present in four different cycles, each at
// its own issue + L: one issue per cycle is one write per cycle.
void test_write_and_read_timing() {
    const auto& profile = canonical_profile();
    require(nanosm::issue_to_urf_port(profile) == 3 && nanosm::issue_to_urf_visible(profile) == 5,
        "the canonical URF timing is L 3, visible from issue + 5");
    const auto scheduled = nanosm::compile_qtx(R"(
        umov  ur1, 7
        uaddi ur2, ur1, 1
        ur2r  r1, ur2
        addi  r2, r1, 1
        halt
    )", profile);
    require(scheduled.instructions[0].control.stall == 4, "qtxc: uniform write -> uniform read is not 4 cycles");
    require(scheduled.instructions[1].control.stall == 4, "qtxc: uniform write -> ur2r is not 4 cycles");
    require(scheduled.instructions[2].control.stall == 6, "qtxc: ur2r -> vector read is not 6 cycles (like mov)");

    nanosm::Launch launch;
    launch.active_warps = 4;
    nanosm::Machine machine(profile, compile(R"(
        usread ur0, WARP_ID
        uaddi ur1, ur0, 10
        halt
    )"), std::move(launch), {.timeline = true});
    machine.run(20000);
    const auto usreads = issues_of(machine, "usread");
    const auto uaddis = issues_of(machine, "uaddi");
    const auto writes = ur_writes(machine);
    require(usreads.size() == 4 && uaddis.size() == 4 && writes.size() == 8, "wrong number of issues or UR writes");
    std::set<uint64_t> write_cycles;
    for (const auto& write : writes) {
        require(write_cycles.insert(write.cycle).second, "two UR writes presented in one cycle");
        const auto& issues = write.reg == 0 ? usreads : uaddis;
        const bool matched = std::any_of(issues.begin(), issues.end(), [&](const auto& issue) {
            return issue.second == write.warp && issue.first + 3 == write.cycle;
        });
        require(matched, "a UR write was not presented at its writer's issue + L");
        require(write.value == static_cast<int64_t>(write.reg == 0 ? write.warp : write.warp + 10),
            "a UR write carried the wrong value");
    }
    for (uint32_t warp = 0; warp < 4; ++warp) {
        const auto first = std::find_if(usreads.begin(), usreads.end(), [&](const auto& i) { return i.second == warp; });
        const auto second = std::find_if(uaddis.begin(), uaddis.end(), [&](const auto& i) { return i.second == warp; });
        require(second->first >= first->first + 4, "a dependent uniform reader issued before L + Wu");
        require(machine.read_ureg(warp, 1) == warp + 10, "uaddi read its source too early");
    }
}

// A uniform reader issued before its source is visible is the sanitizer's
// PANIC (no scoreboard), like a vector RAW.
void test_missing_uniform_stall_panics() {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    const auto& profile = canonical_profile();
    const auto kernel = nanosm::assemble_asm(i1_asm(profile, R"(
        umov ur1, 7 {pred=PT, stall=3, set=-, wait=-}
        uaddi ur2, ur1, 1
        halt
    )"), profile);
    nanosm::Launch launch;
    launch.active_warps = 1;
    nanosm::Machine machine(profile, kernel, std::move(launch));
    expect_sim_panic(machine, "URF RAW hazard: missing stall for ur1", "uniform read three cycles after its writer");
}

// r2ur reads lane 0, so under a partial active mask it is UB like every
// lane-0 operation (backlog D12.1): the sanitizer's PANIC.
void test_r2ur_under_a_partial_mask_panics() {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    nanosm::Launch launch;
    launch.active_warps = 1;
    nanosm::Machine machine(canonical_profile(), compile("sread r0, LANE_ID\nr2ur ur1, r0\nhalt\n"),
        std::move(launch));
    machine.test_set_active_mask(0, 0xfffe);
    expect_sim_panic(machine, "r2ur is a uniform operation under a partial active mask", "r2ur under a partial mask");
}

// The consumers read uniform registers: lds.uni and ldg.uni bases computed by
// uniform arithmetic, an atomic's address and operand, the mbar address and
// parity, and the acc.st / acc.ld coordinate pair.
void test_consumers_take_uniform_registers() {
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.shmem_image, 1024);
    write_u32(shmem, 256, 40);
    write_u32(shmem, 324, 0x1234);
    auto& gmem = image_bytes(launch.gmem_image, 128);
    write_u32(gmem, 68, 0xbeef);
    nanosm::Machine machine(canonical_profile(), compile(R"(
        .acc T, 0, 16, 8
        umov  ur1, 256
        uaddi ur2, ur1, 64
        lds.uni r1, [ur2 + 4]
        umov  ur3, 5
        atom.add.uni r2, [ur1], ur3
        umov  ur8, 64
        ldg.uni r8, [ur8 + 4]
        umov  ur4, 512
        umov  ur5, 0
        mbar.init [ur4], 1
        mbar.arrive [ur4]
        mbar.test_wait p1, [ur4], ur5
@!p1    bra.uni skip
        mov   r5, 1
skip:
        sread r0, LANE_ID
        addi  r6, r0, 100
        umov  ur6, 0
        umov  ur7, 3
        acc.st T, ur6, r6
        acc.ld r7, T, ur6
        halt
    )"), std::move(launch));
    machine.run(20000);
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(machine.read_reg(0, lane, 1) == 0x1234, "lds.uni from a uniform base");
        require(machine.read_reg(0, lane, 2) == 40, "atom.add.uni returned the old word");
        require(machine.read_reg(0, lane, 8) == 0xbeef, "ldg.uni from a uniform base");
        require(machine.read_reg(0, lane, 5) == 1, "mbar.test_wait with a uniform parity");
        require(machine.read_reg(0, lane, 7) == lane + 100, "acc.st / acc.ld with uniform coordinates");
    }
    require(machine.read_shmem_u32(256) == 45, "atom.add.uni added its uniform operand");
    require(machine.read_mbar_state(512).phase, "mbar.arrive at a uniform address did not complete the phase");
}

// The toolchain: the new forms encode, decode and print; the old lane-0
// forms, vector sread of WARP_ID and registers past the URF are refused; the
// uniform registers count neither toward the vector register count nor
// toward the RF bank read rule; kernel load refuses a uniform register past
// the profile's count.
void test_toolchain_forms_and_refusals() {
    const auto& profile = canonical_profile();
    const auto kernel = compile(R"(
        .acc T, 0, 16, 8
        usread ur0, WARP_ID
        umov  ur1, ur0
        umov  ur2, -5
        uadd  ur3, ur1, ur2
        uxori ur4, ur3, 9
        r2ur  ur5, r2
        ur2r  r1, ur5
        lds.uni r0, [ur3 + 8]
        atom.add.uni r2, [ur0 - 4], ur3
        acc.ld r1, T, ur6
        acc.st T, ur6, r1
        mbar.try_wait p1, [ur4], ur5
        halt
    )");
    const std::vector<std::string> expected = {
        "usread ur0, WARP_ID", "umov ur1, ur0", "umov ur2, -5", "uadd ur3, ur1, ur2", "uxori ur4, ur3, 9",
        "r2ur ur5, r2", "ur2r r1, ur5", "lds.uni r0, [ur3 + 8]", "atom.add.uni r2, [ur0 - 4], ur3",
        "acc.ld r1, ACC0, ur6", "acc.st ACC0, ur6, r1", "mbar.try_wait p1, [ur4], ur5", "halt"};
    require(kernel.words.size() == expected.size(), "wrong instruction count");
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const auto text = nanosm::format_instruction(nanosm::decode(kernel.words[i]), false);
        require(text == expected[i], "format: '" + text + "', expected '" + expected[i] + "'");
        require(nanosm::encode(nanosm::decode(kernel.words[i])) == kernel.words[i], "encode / decode: " + text);
    }
    // r0..r2 only: ur0..ur6 are not vector rows. The atomic reads no RF bank
    // (ur0 and ur3 would be one bank as r0 / r3).
    require(kernel.register_count == 3, "uniform registers counted as vector registers");
    require(nanosm::load_hex(nanosm::emit_hex(kernel)).words == kernel.words, "hex round trip");

    const auto refused = [&](const std::string& source, const std::string& needle, const std::string& what) {
        const auto error = compile_error(source);
        require(error.find(needle) != std::string::npos, what + ": wrong or no error: '" + error + "'");
    };
    refused("sread r0, WARP_ID\nhalt\n", "usread", "vector sread of WARP_ID");
    refused("sread r0, N_WARPS\nhalt\n", "usread", "vector sread of N_WARPS");
    refused("usread ur0, LANE_ID\nhalt\n", "sread", "usread of LANE_ID");
    refused("umov ur16, 1\nhalt\n", "outside this profile's 16", "ur16");
    refused("mov r2, 0\nlds.uni r1, [r2]\nhalt\n", "expected uniform register", "a vector lds.uni base");
    refused("mov r2, 0\nmbar.arrive [r2]\nhalt\n", "expected uniform register", "a vector mbar address");
    refused("mbar.try_wait p1, [ur0], r3\nhalt\n", "expected uniform register", "a vector mbar parity");
    refused(".acc T, 0, 16, 8\nacc.ld r1, T, ur15\nhalt\n", "coordinate pair", "an acc coordinate pair past ur15");
    refused("umov ur1, r2\nhalt\n", "r2ur", "umov from a vector register");

    nanosm::Instruction bad;
    bad.opcode = nanosm::Opcode::umovi;
    bad.dst = 16;
    bad.imm = 1;
    nanosm::Kernel image = compile("halt\n");
    image.words.insert(image.words.begin(), nanosm::encode(bad));
    nanosm::Launch launch;
    launch.active_warps = 1;
    try {
        nanosm::Machine machine(profile, image, std::move(launch));
        throw std::runtime_error("kernel load accepted ur16");
    } catch (const nanosm::SimError& error) {
        require(std::string(error.what()).find("uniform register ur16") != std::string::npos,
            std::string("kernel load: wrong error: ") + error.what());
    }
}

// The profile fields: declared in the canonical profile, optional with the
// canonical defaults, in the fingerprint; L must cover the longest writer and
// the URF may not read slower than the RF. A profile with L 4 schedules a
// dependent uniform reader 5 cycles later and presents each write at issue
// + 4.
void test_profile_fields() {
    const auto& canonical = canonical_profile();
    require(canonical.urf_registers == 16 && canonical.urf_read_latency_cycles == 1 &&
            canonical.urf_write_depth_cycles == 1 && canonical.urf_alu_latency_cycles == 1 &&
            canonical.urf_issue_to_port_cycles == 3, "canonical URF fields");
    require(nanosm::urf_longest_writer_path(canonical) == 3, "canonical writer path: 1 + max(1 + 1, 2)");
    i21_check_optional_field("urf", "registers", 16, &nanosm::DeviceProfile::urf_registers);
    i21_check_optional_field("urf", "write_depth_cycles", 1, &nanosm::DeviceProfile::urf_write_depth_cycles);
    i21_check_optional_field("urf", "issue_to_port_cycles", 3, &nanosm::DeviceProfile::urf_issue_to_port_cycles);
    const auto rejected = [](nanosm::DeviceProfile profile, const std::string& needle, const std::string& what) {
        try {
            nanosm::validate_device_profile(profile);
        } catch (const nanosm::ProfileError& error) {
            require(std::string(error.what()).find(needle) != std::string::npos,
                what + ": wrong error: " + error.what());
            return;
        }
        throw std::runtime_error(what + " passed validation");
    };
    auto slow_alu = canonical;
    slow_alu.urf_alu_latency_cycles = 2;
    rejected(slow_alu, "must cover the longest uniform-register writer, 4", "uniform ALU 2 with L 3");
    auto slow_rf = canonical;
    slow_rf.rf_read_latency_cycles = 3;
    rejected(slow_rf, "must cover the longest uniform-register writer, 4", "rf read latency 3 with L 3");
    auto slow_urf = canonical;
    slow_urf.urf_read_latency_cycles = 3;
    slow_urf.urf_issue_to_port_cycles = 5;
    rejected(slow_urf, "may not exceed rf read_latency_cycles", "URF read latency 3 above the RF's 2");
    slow_alu.urf_issue_to_port_cycles = 4;
    nanosm::validate_device_profile(slow_alu);
    require(nanosm::device_profile_fingerprint(slow_alu) != nanosm::device_profile_fingerprint(canonical),
        "urf alu_latency_cycles is not in the fingerprint");

    auto padded = canonical;
    padded.urf_issue_to_port_cycles = 4;
    const auto source = std::string("umov ur1, 7\nuaddi ur2, ur1, 1\nhalt\n");
    require(nanosm::compile_qtx(source, padded).instructions[0].control.stall == 5,
        "L 4: a dependent uniform reader is not 5 cycles later");
    nanosm::Launch launch;
    launch.active_warps = 1;
    nanosm::Machine machine(padded, compile(source, padded), std::move(launch), {.timeline = true});
    machine.run(20000);
    require(machine.read_ureg(0, 2) == 8, "L 4: wrong value");
    const auto writes = ur_writes(machine);
    const auto umov = issues_of(machine, "umov");
    require(writes.size() == 2 && umov.size() == 1 && writes[0].cycle == umov[0].first + 4,
        "L 4: the write was not presented at issue + 4");
}

// The trace: one "UR write" instant per uniform write, with the register and
// value, and the summary's urf line.
void test_trace_shows_writes() {
    nanosm::Launch launch;
    launch.active_warps = 1;
    nanosm::Machine machine(canonical_profile(), compile("umov ur1, 7\nuaddi ur2, ur1, 35\nhalt\n"),
        std::move(launch), {.timeline = true});
    machine.run(20000);
    require(machine.trace().summary().urf_writes == 2, "urf write count");
    require(machine.trace().summary_text().find("urf: writes=2 read_ports_held=0\n") != std::string::npos, "summary urf line");
    const auto json = machine.trace().perfetto_json();
    require(json.find("UR write") != std::string::npos && json.find("ur2 = 42") != std::string::npos,
        "Perfetto trace lacks the UR write instants");
}

} // namespace

// uniform arithmetic, usread, r2ur and ur2r: values
TEST(Urf, uniform_arithmetic_and_moves) {
    test_uniform_arithmetic_and_moves();
}

// fixed write latency, dependent distances, one write per cycle
TEST(Urf, write_and_read_timing) {
    test_write_and_read_timing();
}

// a missing uniform stall is a sanitizer panic
TEST(Urf, missing_uniform_stall_panics) {
    test_missing_uniform_stall_panics();
}

// r2ur reads lane 0: a partial mask is a sanitizer panic
TEST(Urf, r2ur_under_a_partial_mask_panics) {
    test_r2ur_under_a_partial_mask_panics();
}

// lds.uni / ldg.uni / atom / mbar / acc read uniform registers
TEST(Urf, consumers_take_uniform_registers) {
    test_consumers_take_uniform_registers();
}

// encoding, formatting, refusals, kernel load
TEST(Urf, toolchain_forms_and_refusals) {
    test_toolchain_forms_and_refusals();
}

// the urf profile section and L
TEST(Urf, profile_fields) {
    test_profile_fields();
}

// UR write instants and the summary line
TEST(Urf, trace_shows_writes) {
    test_trace_shows_writes();
}

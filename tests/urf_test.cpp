// The uniform register file:
// uniform arithmetic, r2ur / ur2r, usread, the consumers that read uniform
// registers, the fixed write latency L, the profile fields and the trace.
// Smoke tests: things run, the data is right, and the few timing facts the
// design rests on (every writer presents at issue + L, a dependent uniform
// reader issues L + Wu later, one write per cycle).
#include "test_support.hpp"

namespace {

using namespace nanosm_test;

// Every issue whose opcode name is `name`, in trace order.
std::vector<IssueRecord> issues_named(const nanosm::Machine& machine, const std::string& name) {
    std::vector<IssueRecord> result;
    for (const auto& issue : issue_records(machine)) {
        if (issue.opcode == name) result.push_back(issue);
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

} // namespace

// Every uniform ALU opcode, usread, r2ur (lane 0) and ur2r (every lane), on
// three warps: the values in the URF and in the vector copies.
TEST(Urf, uniform_arithmetic_and_moves) {
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
    )"), launch_warps(3));
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
TEST(Urf, write_and_read_timing) {
    const auto& profile = canonical_profile();
    require(nanosm::issue_to_urf_port(profile) == 3 && nanosm::issue_to_urf_visible(profile) == 5,
        "the canonical URF timing is L 3, visible from issue + 5");
    const auto scheduled = qtx_program(R"(
        umov  ur1, 7
        uaddi ur2, ur1, 1
        ur2r  r1, ur2
        addi  r2, r1, 1
        halt
    )", profile);
    require(scheduled.instructions[0].control.stall == 4, "qtxc: uniform write -> uniform read is not 4 cycles");
    require(scheduled.instructions[1].control.stall == 4, "qtxc: uniform write -> ur2r is not 4 cycles");
    require(scheduled.instructions[2].control.stall == 6, "qtxc: ur2r -> vector read is not 6 cycles (like mov)");
    // A uniform second source is read like ur2r's: 4 after its writer. One
    // cycle sooner is the sanitizer's PANIC.
    const auto uniform_b = qtx_program("umov ur2, 7\nadd r3, r1, ur2\nhalt\n", profile);
    require(uniform_b.instructions[0].control.stall == 4, "qtxc: uniform write -> add's uniform source is not 4 cycles");
    nanosm::Machine at_distance(profile, compile("umov ur2, 7\nadd r3, r1, ur2\nhalt\n"), launch_warps(1));
    at_distance.run(20000);
    require(at_distance.read_reg(0, 5, 3) == 7, "add read its uniform source wrong");
#if NANOSM_SANITIZER
    nanosm::Machine early(profile, nanosm::assemble_asm(hand_asm(profile,
        "umov ur2, 7 {pred=PT, stall=3, set=-, wait=-}\nadd r3, r1, ur2\nhalt"), profile), launch_warps(1));
    expect_panic(early, {"URF RAW hazard: missing stall for ur2"}, "add three cycles after its uniform source");
#endif

    nanosm::Machine machine(profile, compile(R"(
        usread ur0, WARP_ID
        uaddi ur1, ur0, 10
        halt
    )"), launch_warps(4), {.timeline = true});
    machine.run(20000);
    const auto usreads = issues_named(machine, "usread");
    const auto uaddis = issues_named(machine, "uaddi");
    const auto writes = ur_writes(machine);
    require(usreads.size() == 4 && uaddis.size() == 4 && writes.size() == 8, "wrong number of issues or UR writes");
    std::set<uint64_t> write_cycles;
    for (const auto& write : writes) {
        require(write_cycles.insert(write.cycle).second, "two UR writes presented in one cycle");
        const auto& issues = write.reg == 0 ? usreads : uaddis;
        const bool matched = std::any_of(issues.begin(), issues.end(), [&](const auto& issue) {
            return issue.warp == write.warp && issue.cycle + 3 == write.cycle;
        });
        require(matched, "a UR write was not presented at its writer's issue + L");
        require(write.value == static_cast<int64_t>(write.reg == 0 ? write.warp : write.warp + 10),
            "a UR write carried the wrong value");
    }
    for (uint32_t warp = 0; warp < 4; ++warp) {
        const auto of_warp = [&](const std::vector<IssueRecord>& issues) {
            const auto found = std::find_if(issues.begin(), issues.end(), [&](const auto& i) { return i.warp == warp; });
            require(found != issues.end(), "warp " + std::to_string(warp) + " did not issue it");
            return found->cycle;
        };
        require(of_warp(uaddis) == of_warp(usreads) + 4,
            "warp " + std::to_string(warp) + ": the dependent uniform reader did not issue L + Wu after its writer" +
                issue_log(machine));
        require(machine.read_ureg(warp, 1) == warp + 10, "uaddi read its source too early");
    }
}

// The consumers read uniform registers: lds.uni and ldg.uni bases computed by
// uniform arithmetic, an atomic's address and operand, the mbar address and
// parity, and the acc.st / acc.ld tmem pointer (here row 0, column 3 of a
// region of 8-word rows).
TEST(Urf, consumers_take_uniform_registers) {
    auto launch = launch_warps(1);
    auto& shmem = image_bytes(launch.test_shmem, 1024);
    write_u32(shmem, 256, 40);
    write_u32(shmem, 324, 0x1234);
    auto& gmem = image_bytes(launch.gmem_image, 128);
    write_u32(gmem, 68, 0xbeef);
    nanosm::Machine machine(canonical_profile(), compile(R"(
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
        umov  ur6, 3
        acc.st [ur6, 8], r6
        acc.ld r7, [ur6, 8]
        halt
    )"), std::move(launch));
    machine.run(20000);
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(machine.read_reg(0, lane, 1) == 0x1234, "lds.uni from a uniform base");
        require(machine.read_reg(0, lane, 2) == 40, "atom.add.uni returned the old word");
        require(machine.read_reg(0, lane, 8) == 0xbeef, "ldg.uni from a uniform base");
        require(machine.read_reg(0, lane, 5) == 1, "mbar.test_wait with a uniform parity");
        require(machine.read_reg(0, lane, 7) == lane + 100, "acc.st / acc.ld through a uniform tmem pointer");
    }
    require(machine.read_shmem_u32(256) == 45, "atom.add.uni added its uniform operand");
    require(machine.read_mbar_state(512).phase, "mbar.arrive at a uniform address did not complete the phase");
}

// The toolchain: the new forms encode, decode and print; the old lane-0
// forms, vector sread of WARP_ID and registers past the URF are refused; the
// uniform registers count neither toward the vector register count nor
// toward the RF bank read rule; kernel load refuses a uniform register past
// the profile's count.
TEST(Urf, toolchain_forms_and_refusals) {
    const auto& profile = canonical_profile();
    const auto kernel = compile(R"(
        usread ur0, WARP_ID
        umov  ur1, ur0
        umov  ur2, -5
        uadd  ur3, ur1, ur2
        uxori ur4, ur3, 9
        r2ur  ur5, r2
        ur2r  r1, ur5
        lds.uni r0, [ur3 + 8]
        atom.add.uni r2, [ur0 - 4], ur3
        acc.ld r1, [ur6, 8]
        acc.st [ur6, 8], r1
        mbar.try_wait p1, [ur4], ur5
        usetp.lt.u32 p3, ur3, ur4
        usetp.ge.u32 p4, ur3, -7
        usetp.lt.s32 p5, ur8, 0
        usetp.ne.s32 p6, ur3, ur4
        setp.gt.s32 p0, r1, -4
        halt
    )");
    const std::vector<std::string> expected = {
        "usread ur0, WARP_ID", "umov ur1, ur0", "umov ur2, -5", "uadd ur3, ur1, ur2", "uxori ur4, ur3, 9",
        "r2ur ur5, r2", "ur2r r1, ur5", "lds.uni r0, [ur3 + 8]", "atom.add.uni r2, [ur0 - 4], ur3",
        "acc.ld r1, [ur6, 8]", "acc.st [ur6, 8], r1", "mbar.try_wait p1, [ur4], ur5", "usetp.lt.u32 p3, ur3, ur4",
        "usetp.ge.u32 p4, ur3, -7", "usetp.lt.s32 p5, ur8, 0", "usetp.ne.u32 p6, ur3, ur4", "setp.gt.s32 p0, r1, -4",
        "halt"};
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
        expect_error([&] { (void)compile(source, profile); }, {needle}, what);
    };
    refused("sread r0, WARP_ID\nhalt\n", "usread", "vector sread of WARP_ID");
    refused("sread r0, N_WARPS\nhalt\n", "usread", "vector sread of N_WARPS");
    // The first register past the profile's URF and its last one.
    const auto past = "ur" + std::to_string(profile.urf_registers);
    const auto last = "ur" + std::to_string(profile.urf_registers - 1);
    refused("usread ur0, LANE_ID\nhalt\n", "sread", "usread of LANE_ID");
    // A register past the file's end has no name to be written with: the
    // file just runs out. Assembly's refusal of the number is below.
    refused("umov " + past + ", 1\nhalt\n", "this profile's " + std::to_string(profile.urf_registers), past);
    // A name is one register in one file.
    refused("mov r2, 0\nlds.uni r1, [r2]\nhalt\n", "is used here as a uniform register", "a vector lds.uni base");
    refused("mov r2, 0\nmbar.arrive [r2]\nhalt\n", "is used here as a uniform register", "a vector mbar address");
    refused("mbar.try_wait p1, [ur0], r3\nhalt\n", "is used here as a uniform register", "a vector mbar parity");
    refused("acc.ld r1, [" + past + ", 8]\nhalt\n", "this profile's", "an acc pointer past " + last);
    refused("acc.ld r1, [ur1, 6]\nhalt\n", "must be a power of two", "an acc row of 6 words");
    refused(".acc T, 0, 16, 8\nhalt\n", "there is no .acc directive", "the .acc directive");
    refused("umov ur1, r2\nhalt\n", "r2ur", "umov from a vector register");
    refused("usetp.eq.u32 p0, ur1, r2\nhalt\n", "is used here as a uniform register", "usetp of a vector register");
    expect_error([&] { (void)nanosm::assemble_asm(hand_asm(profile, "umov " + past + ", 1\nlds.uni r1, [r2]\nhalt"), profile); },
        {"outside this profile's " + std::to_string(profile.urf_registers)}, "assembly of " + past);
    refused("usetp.eq.u32 PT, ur1, 0\nhalt\n", "usetp destination must be p0..p6", "usetp into PT");
    refused("usetp.lo.u32 p0, ur1, 0\nhalt\n", "unknown usetp comparison", "an unknown usetp comparison");
    // Every compare says its type; the refusal names the two choices, in qtx
    // and in assembly alike.
    refused("setp.lt p0, r1, r2\nhalt\n", "'setp.lt' needs its type: setp.lt.u32 (unsigned) or setp.lt.s32 (signed)",
        "an untyped setp");
    refused("usetp.eq p0, ur1, 0\nhalt\n", "usetp.eq.u32 (unsigned) or usetp.eq.s32 (signed)", "an untyped usetp");
    refused("usetp.ge.f32 p0, ur1, 0\nhalt\n", "usetp.ge.u32 (unsigned) or usetp.ge.s32 (signed)", "a float usetp");
    expect_error([&] { (void)nanosm::assemble_asm(hand_asm(profile, "setp.ne p0, r1, r2\nhalt"), profile); },
        {"setp.ne.u32 (unsigned)"}, "assembly of an untyped setp");
    // eq / ne written .s32 is the .u32 opcode and is printed so in scheduled asm.
    const auto scheduled = nanosm::emit_asm(qtx_program("usetp.ne.s32 p6, ur3, ur4\nhalt\n", profile));
    require(scheduled.find("        usetp.ne.u32 p6, ur3, ur4 {") != std::string::npos,
        "scheduled asm does not print usetp.ne.s32 as usetp.ne.u32");

    nanosm::Instruction bad;
    bad.opcode = nanosm::Opcode::umovi;
    bad.dst = static_cast<uint8_t>(profile.urf_registers);
    bad.imm = 1;
    nanosm::Kernel image = compile("halt\n");
    image.words.insert(image.words.begin(), nanosm::encode(bad));
    expect_error([&] { nanosm::Machine machine(profile, image, launch_warps(1)); }, {"uniform register " + past},
        "kernel load of " + past);
}

// The rules between the URF fields: L must cover the longest writer and the
// URF may not read slower than the RF. A profile with L 4 schedules a
// dependent uniform reader 5 cycles later and presents each write at issue
// + 4.
TEST(Urf, profile_fields) {
    const auto& canonical = canonical_profile();
    require(canonical.urf_registers == 32 && canonical.urf_read_latency_cycles == 1 &&
            canonical.urf_write_depth_cycles == 1 && canonical.urf_alu_latency_cycles == 1 &&
            canonical.urf_issue_to_port_cycles == 3, "canonical URF fields");
    require(nanosm::urf_longest_writer_path(canonical) == 3, "canonical writer path: 1 + max(1 + 1, 2)");
    const auto rejected = [](const nanosm::DeviceProfile& profile, const std::string& needle, const std::string& what) {
        expect_error<nanosm::ProfileError>([&] { nanosm::validate_device_profile(profile); }, {needle}, what);
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

    auto padded = canonical;
    padded.urf_issue_to_port_cycles = 4;
    const auto source = std::string("umov ur1, 7\nuaddi ur2, ur1, 1\nhalt\n");
    require(qtx_program(source, padded).instructions[0].control.stall == 5,
        "L 4: a dependent uniform reader is not 5 cycles later");
    nanosm::Machine machine(padded, compile(source, padded), launch_warps(1), {.timeline = true});
    machine.run(20000);
    require(machine.read_ureg(0, 2) == 8, "L 4: wrong value");
    const auto writes = ur_writes(machine);
    const auto umov = issues_named(machine, "umov");
    require(writes.size() == 2 && umov.size() == 1 && writes[0].cycle == umov[0].cycle + 4,
        "L 4: the write was not presented at issue + 4");
}

// The trace: one "UR write" instant per uniform write, with the register and
// value, and the summary's urf line.
TEST(Urf, trace_shows_writes) {
    nanosm::Machine machine(canonical_profile(), compile("umov ur1, 7\nuaddi ur2, ur1, 35\nhalt\n"),
        launch_warps(1), {.timeline = true});
    machine.run(20000);
    require(machine.trace().summary().urf_writes == 2, "urf write count");
    require(machine.trace().summary_text().find("urf: writes=2 read_ports_held=0\n") != std::string::npos, "summary urf line");
    const auto json = machine.trace().perfetto_json();
    require(json.find("UR write") != std::string::npos && json.find("ur2 = 42") != std::string::npos,
        "Perfetto trace lacks the UR write instants");
}

// The uniform and vector compares: each of the six unsigned comparisons once
// in each form (register and immediate), and lt, le, gt, ge .s32 in both
// forms, on values whose signed and unsigned orders differ. Each usetp's
// predicate, gathered into a bit of ur31 through @!p1 bra.uni, and each
// setp's on the same values (a bit of ur30), equal the host's comparison of
// the row's type.
TEST(Urf, uniform_compare_values) {
    struct Case {
        std::string comparison;
        bool is_signed = false;
        int32_t lhs = 0;
        int32_t rhs = 0;
        bool immediate = false;
    };
    const std::vector<Case> cases = {
        {"eq", false, 7, 7, false},  {"eq", false, -1, -1, true},
        {"ne", false, 7, 8, false},  {"ne", false, 0, 0, true},
        {"lt", false, -3, 5, false}, {"lt", false, 5, -3, true}, // unsigned: 0xfffffffd is the larger
        {"le", false, 9, 9, false},  {"le", false, -1, 0, true},
        {"gt", false, -1, 1, false}, {"gt", false, 2, 3, true},
        {"ge", false, 0, -2, false}, {"ge", false, -2, -2, true},
        {"lt", true, -3, 5, false},  {"lt", true, 5, -3, true},
        {"le", true, -1, 0, false},  {"le", true, 1, -1, true},
        {"gt", true, -1, 1, false},  {"gt", true, 0, -2, true},
        {"ge", true, 0, -2, false},  {"ge", true, -5, 3, true},
    };
    const auto holds = [](const std::string& comparison, auto a, auto b) {
        return comparison == "eq" ? a == b
            : comparison == "ne" ? a != b
            : comparison == "lt" ? a < b
            : comparison == "le" ? a <= b
            : comparison == "gt" ? a > b
            : a >= b;
    };
    const auto row = [](const Case& c) {
        return c.comparison + (c.is_signed ? ".s32 " : ".u32 ") + std::to_string(c.lhs) + ", " +
            std::to_string(c.rhs) + (c.immediate ? " (immediate)" : "");
    };
    std::string source = "umov ur30, 0\numov ur31, 0\n";
    std::vector<bool> expected;
    for (std::size_t i = 0; i < cases.size(); ++i) {
        const auto& c = cases[i];
        const bool unsigned_holds = holds(c.comparison, static_cast<uint32_t>(c.lhs), static_cast<uint32_t>(c.rhs));
        if (c.is_signed) {
            require(holds(c.comparison, c.lhs, c.rhs) != unsigned_holds,
                row(c) + ": the row does not tell signed from unsigned");
        }
        expected.push_back(c.is_signed ? holds(c.comparison, c.lhs, c.rhs) : unsigned_holds);
        const auto n = std::to_string(i);
        const auto bit = std::to_string(1u << i);
        const auto rhs = std::to_string(c.rhs);
        const auto type = c.is_signed ? ".s32" : ".u32";
        source += "umov ur1, " + std::to_string(c.lhs) + "\numov ur2, " + rhs + "\n";
        source += "usetp." + c.comparison + type + " p1, ur1, " + (c.immediate ? rhs : "ur2") + "\n";
        source += "@!p1 bra.uni u" + n + "\nuori ur31, ur31, " + bit + "\nu" + n + ":\n";
        source += "ur2r r1, ur1\nur2r r2, ur2\n";
        source += "setp." + c.comparison + type + " p2, r1, " + (c.immediate ? rhs : "r2") + "\n";
        source += "@!p2 bra.uni v" + n + "\nuori ur30, ur30, " + bit + "\nv" + n + ":\n";
    }
    source += "halt\n";
    nanosm::Machine machine(canonical_profile(), compile(source), launch_warps(2));
    machine.run(20000);
    for (uint32_t warp = 0; warp < 2; ++warp) {
        for (std::size_t i = 0; i < cases.size(); ++i) {
            const auto where = "warp " + std::to_string(warp) + ", " + row(cases[i]) + ": ";
            require(((machine.read_ureg(warp, 31) >> i) & 1) == expected[i], where + "usetp's predicate is wrong");
            require(((machine.read_ureg(warp, 30) >> i) & 1) == expected[i], where + "setp's predicate is wrong");
        }
    }
}

// A loop counting in a uniform register, with no vector register at all.
// qtxc places the compare L + Wu = 4 after the counter's write (a uniform
// RAW) and the bra.uni L = 3 after the compare: its predicate is presented
// at issue + L and in the file from issue + L + 1, where the branch's sample
// (its issue + 1) meets it (a vector setp's is 4). The machine issues each
// branch 3 cycles after its compare.
TEST(Urf, uniform_compare_loop_and_timing) {
    const auto& profile = canonical_profile();
    require(nanosm::issue_to_uniform_predicate_visible(profile) == 4,
        "the canonical usetp predicate is not in the file from issue + 4");
    const std::string source = R"(
        umov  ur1, 0
        umov  ur2, 0
    loop:
        uaddi ur2, ur2, 3
        uaddi ur1, ur1, 1
        usetp.lt.u32 p0, ur1, 10
@p0     bra.uni loop
        halt
    )";
    const auto scheduled = qtx_program(source, profile);
    require(scheduled.instructions.at(3).control.stall == 4, "qtxc: uniform write -> usetp is not 4 cycles");
    require(scheduled.instructions.at(4).control.stall == 3, "qtxc: usetp -> bra.uni is not 3 cycles");
    const auto kernel = compile(source);
    require(kernel.register_count == 0, "the uniform loop allocated vector registers");
    nanosm::Machine machine(profile, kernel, launch_warps(1), {.timeline = true});
    machine.run(20000);
    require(machine.read_ureg(0, 1) == 10 && machine.read_ureg(0, 2) == 30, "the uniform loop ran the wrong count");
    const auto compares = issues_named(machine, "usetp.lt.u32");
    const auto branches = issues_named(machine, "bra.uni");
    require(compares.size() == 10 && branches.size() == 10, "ten iterations: ten compares and ten branches");
    for (std::size_t i = 0; i < compares.size(); ++i) {
        require(branches[i].cycle == compares[i].cycle + 3, "a bra.uni did not issue 3 cycles after its usetp");
    }
}

// A compare issued before its uniform source is visible, or a bra.uni issued
// before the compare's predicate is in the file, is the sanitizer's PANIC
// (no scoreboard), like a vector RAW. At the distances (4, then 3) the branch
// sees the new predicate, in both builds.
TEST(Urf, early_uniform_compare_or_branch_panics) {
    const auto& profile = canonical_profile();
    const auto machine = [&](uint32_t write_stall, uint32_t compare_stall) {
        const auto program = "umov ur1, 7 {pred=PT, stall=" + std::to_string(write_stall) + ", set=-, wait=-}\n" +
            "usetp.eq.u32 p0, ur1, 7 {pred=PT, stall=" + std::to_string(compare_stall) + ", set=-, wait=-}\n" + R"(
            bra.uni target {pred=p0, stall=0, set=-, wait=-}
            umov ur2, 1
            halt
        target: umov ur2, 2
            halt
        )";
        return std::make_unique<nanosm::Machine>(
            profile, nanosm::assemble_asm(hand_asm(profile, program), profile), launch_warps(1));
    };
    const auto at = machine(4, 3);
    at->run(20000);
    require(at->read_ureg(0, 2) == 2, "at the distances the bra.uni did not see the compare's predicate");
#if NANOSM_SANITIZER
    const auto early_compare = machine(3, 3);
    expect_panic(*early_compare, {"URF RAW hazard: missing stall for ur1"}, "usetp three cycles after its source");
    const auto early_branch = machine(4, 2);
    expect_panic(*early_branch, {"predicate RAW hazard: missing stall for p0"}, "bra.uni two cycles after usetp");
#endif
}

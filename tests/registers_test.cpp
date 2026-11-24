// qtx's registers (src/qtxc/registers.cpp, README "Registers"): a register
// is a name, qtxc picks its number. Smoke tests: names compute, the
// refusals read well, groups are neighbours, the bank rule is qtxc's (a copy
// and a warning where it cannot fit), and names share a number only when
// they never hold a value at the same time.
#include "test_support.hpp"

using namespace nanosm_test;

TEST(Registers, names_compute) {
    const auto& profile = canonical_profile();
    const std::string source = R"(
        sread  %lane, LANE_ID
        shli   %four, %lane, 2
        add    %sum, %lane, %four      ; 5 lane
        umov   %seven, 7
        ur2r   %bias, %seven
        add    %sum, %sum, %bias       ; 5 lane + 7
        mov    %kept, %sum
        halt
    )";
    const auto program = nanosm::compile_qtx(source, profile);
    const auto text = nanosm::emit_asm(program);
    // The assembly holds numbers and says which name got which.
    require(program.copies == 0 && program.warnings.empty(), "a copy where none is needed");
    require(text.find(".name %sum, r") != std::string::npos && text.find(".name %seven, ur") != std::string::npos,
        "the assembly does not say which number a name got:\n" + text);
    for (const auto& inst : program.instructions) {
        require(inst.text.find('%') == std::string::npos, "a name is left in the scheduled text: " + inst.text);
    }
    const auto kernel = compile(source, profile);
    require(nanosm::load_hex(nanosm::emit_hex(kernel)).register_names.size() == kernel.register_names.size(),
        "the hex text lost the register names");
    nanosm::Machine machine(profile, kernel, launch_warps(1));
    machine.run();
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(machine.read_reg(0, lane, number_of(kernel.register_names, "kept")) == 5 * lane + 7 &&
                machine.read_reg(0, lane, number_of(kernel.register_names, "sum")) == 5 * lane + 7,
            "named registers computed the wrong value");
    }
    require(machine.read_ureg(0, number_of(kernel.register_names, "seven")) == 7, "a named uniform register");
}

TEST(Registers, refusals) {
    qtx_refused("mov r5, 1\nhalt\n", "qtx names its registers: write a name such as %x instead of 'r5'", "a raw vector register");
    qtx_refused("umov ur3, 1\nhalt\n", "instead of 'ur3'", "a raw uniform register");
    qtx_refused("umov %a, 1\nadd %b, %a, %a\nhalt\n",
        "line 2: %a is used here as a vector register and on line 1 as a uniform register ('umov %a, 1')",
        "one name in both files");
    qtx_refused("mov %p, 0\nldg.v8 %x, [%p]\nhalt\n",
        "ldg.v8 writes 8 neighbouring registers starting at %x, which is a single register: declare a group, "
        "`.reg %x<8>`, and name its first member, %x0",
        "ldg.v8 into a lone name");
    qtx_refused(".reg %x<8>\nmov %p, 0\nldg.v8 %x3, [%p]\nhalt\n", "has only 5 from there on", "ldg.v8 past its group's end");
    qtx_refused("umov %s, 0\numov %m, 64\numov %c, 0\ntma.load.2d [%s], [4096, %c], [%m]\nhalt\n",
        "declare a group, `.ureg %c<2>`", "tma coordinates in a lone name");
    qtx_refused(".reg %a<8>\n.reg %a<4>\nhalt\n", "is already a member of another group", "two groups with one stem");
    qtx_refused(".reg %a\nhalt\n", "expected .reg %name<count>", "a group without a count");
    qtx_refused(".name %a, r1\nhalt\n", "qtx has no .name line", ".name in qtx");
    // Assembly's registers are numbers.
    expect_error([] { (void)nanosm::assemble_asm(hand_asm(canonical_profile(), "mov %a, 1\nhalt"), canonical_profile()); },
        {"expected register, got '%a'"}, "a name in assembly");
}

// The bank rule is qtxc's business. One register in several slots is one
// read; two groups read pairwise get different banks (a group has one choice,
// its base); a value read beside registers of every bank gets a
// copy just before the read, a warning, and the right answer.
TEST(Registers, banks_and_copies) {
    const auto& profile = canonical_profile();
    const auto once = nanosm::compile_qtx("mov %x, 3\nadd %x, %x, %x\nhalt\n", profile);
    require(once.copies == 0 && once.warnings.empty(), "add %x, %x, %x needed a copy");

    std::string groups = ".reg %a<8>, %b<8>\n";
    for (int j = 0; j < 8; ++j) {
        groups += "fadd %a" + std::to_string(j) + ", %a" + std::to_string(j) + ", %b" + std::to_string(j) + "\n";
    }
    const auto paired = nanosm::compile_qtx(groups + "halt\n", profile);
    const auto a0 = number_of(paired.register_names, "a0");
    const auto b0 = number_of(paired.register_names, "b0");
    require(paired.copies == 0 && a0 % profile.rf_banks != b0 % profile.rf_banks,
        "two groups read pairwise share a bank or needed copies");
    for (int j = 1; j < 8; ++j) {
        require(number_of(paired.register_names, "a" + std::to_string(j)) == a0 + j, "a group's members are not neighbours");
    }

    // %x, %y, %z are read pairwise, so they take one bank each (three banks,
    // one read port); %v is read beside each of them and has no bank left.
    const std::string source = R"(
        sread  %lane, LANE_ID
        addi   %x, %lane, 1
        addi   %y, %lane, 2
        addi   %z, %lane, 3
        mov    %v, 100
        add    %t0, %x, %y
        add    %t1, %y, %z
        add    %t2, %x, %z
A:      add    %r0, %v, %x
B:      add    %r1, %v, %y
C:      add    %r2, %v, %z
        add    %all, %r0, %r1
        add    %all, %all, %r2
        halt
    )";
    const auto program = nanosm::compile_qtx(source, profile);
    require(program.copies >= 1 && program.warnings.size() == program.copies,
        "a value read beside every bank got no copy");
    require(program.warnings.front().rfind("line ", 0) == 0 &&
            program.warnings.front().find("%v") != std::string::npos &&
            program.warnings.front().find("from one register-file bank; qtxc inserted a copy of %") !=
                std::string::npos,
        "the warning does not name the line, the registers and the copy: " + program.warnings.front());
    require(nanosm::emit_asm(program).find("; qtxc: a copy of %") != std::string::npos,
        "the copy is not marked in the assembly");
    // A label on the instruction moved to its copy.
    std::size_t copies_seen = 0;
    for (std::size_t index = 0; index < program.instructions.size(); ++index) {
        if (program.notes[index].empty()) continue;
        ++copies_seen;
        require(program.instructions[index].opcode == nanosm::Opcode::addi, "a copy is an addi");
        bool labelled = false;
        for (const auto& label : program.labels) labelled = labelled || label.instruction_index == index;
        require(labelled, "the label did not move to the copy");
    }
    require(copies_seen == program.copies, "copies and their notes disagree");
    const auto kernel = compile(source, profile);
    require(kernel.warnings.size() == program.copies, "the kernel lost qtxc's warnings");
    nanosm::Machine machine(profile, kernel, launch_warps(1));
    machine.run();
    // (Read through %all: a name's register holds its value only until the
    // last read that needs it.)
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(machine.read_reg(0, lane, number_of(kernel.register_names, "all")) == 306 + 3 * lane,
            "the repaired instructions computed the wrong value");
    }
}

// Names share a number only when they never hold a value at the same time.
// Two warps run two roles with twenty-one uniform names each: forty-three
// names in a file of thirty-two. And a write under a mask keeps the old
// value of the lanes it skips, so nothing may take that register meanwhile.
TEST(Registers, sharing) {
    const auto& profile = canonical_profile();
    require(profile.urf_registers == 32, "this test counts on 32 uniform registers");
    std::string source = "usread %w, WARP_ID\nusetp.eq.u32 p0, %w, 0\n@p0 bra.uni B\n";
    for (const std::string role : {"a", "b"}) {
        if (role == "b") source += "B:\n";
        const int first = role == "a" ? 1 : 101;
        for (int k = 0; k < 20; ++k) source += "umov %" + role + std::to_string(k) + ", " + std::to_string(first + k) + "\n";
        source += "uadd %sum_" + role + ", %" + role + "0, %" + role + "1\n";
        for (int k = 2; k < 20; ++k) source += "uadd %sum_" + role + ", %sum_" + role + ", %" + role + std::to_string(k) + "\n";
        source += "halt\n";
    }
    const auto kernel = compile(source, profile);
    nanosm::Machine machine(profile, kernel, launch_warps(2));
    machine.run();
    require(machine.read_ureg(0, number_of(kernel.register_names, "sum_b")) == 2210 &&
            machine.read_ureg(1, number_of(kernel.register_names, "sum_a")) == 210,
        "two roles' uniform registers were not kept apart");
    // Forty at once do not fit.
    std::string crowded;
    for (int k = 0; k < 40; ++k) crowded += "umov %u" + std::to_string(k) + ", 1\n";
    crowded += "uadd %s, %u0, %u1\n";
    for (int k = 2; k < 40; ++k) crowded += "uadd %s, %s, %u" + std::to_string(k) + "\n";
    qtx_refused(crowded + "halt\n", "no uniform register is left for %", "forty uniform values held at once");

    // A small register file (six vector registers for the one warp), so the
    // dozen temporaries below must share; %x's skipped lanes must survive.
    auto small = profile;
    small.rf_rows_per_bank = 2;
    std::string masked = R"(
        .warps 1
        sread  %lane, LANE_ID
        mov    %x, 7
        andi   %bit, %lane, 1
        setp.ne.u32 p0, %bit, 0
    )";
    for (int k = 0; k < 12; ++k) {
        masked += "mov %t" + std::to_string(k) + ", " + std::to_string(50 + k) + "\n";
        masked += "addi %t" + std::to_string(k) + ", %t" + std::to_string(k) + ", 1\n";
    }
    // Between %x's first write and its masked one: %x is not rewritten whole
    // there, so its register is not free in between.
    masked += "@p0 mov %x, 9\nadd %y, %x, %lane\nhalt\n";
    const auto shared = compile(masked, small);
    require(shared.register_count <= 6, "the temporaries did not share registers");
    nanosm::Machine small_machine(small, shared, launch_warps(1));
    small_machine.run();
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(small_machine.read_reg(0, lane, number_of(shared.register_names, "y")) ==
                (lane % 2 == 1 ? 9u : 7u) + lane,
            "a masked write's skipped lanes were lost to a shared register");
    }
}

// A predicate may be a name. %p0 .. %p6 are p0 .. p6 themselves; any other
// %name in a predicate position gets one of the predicates the kernel does
// not name by number. A name is a register or a predicate, not both, and
// only as many predicate values as there are predicates can be held at once.
TEST(Registers, predicate_names) {
    const auto& profile = canonical_profile();
    const std::string source = R"(
        sread  %lane, LANE_ID
        andi   %bit, %lane, 1
        setp.ne.u32 %odd, %bit, 0
        setp.lt.u32 %p0, %lane, 8
        setp.ge.u32 p1, %lane, 12
        mov    %x, 1
        @%odd  mov %x, 2
        @%p0   addi %x, %x, 10
        @!p1   addi %x, %x, 100
        halt
    )";
    const auto program = nanosm::compile_qtx(source, profile);
    const auto odd = number_of(program.register_names, "odd");
    require(odd >= 2 && odd < 7, "a named predicate took one the kernel names by number");
    require(nanosm::emit_asm(program).find(".name %odd, p" + std::to_string(odd)) != std::string::npos,
        "the assembly does not say which predicate a name got");
    const auto kernel = compile(source, profile);
    nanosm::Machine machine(profile, kernel, launch_warps(1));
    machine.run();
    for (uint32_t lane = 0; lane < 16; ++lane) {
        const uint32_t expected = (lane % 2 == 1 ? 2u : 1u) + (lane < 8 ? 10u : 0u) + (lane >= 12 ? 0u : 100u);
        require(machine.read_reg(0, lane, number_of(kernel.register_names, "x")) == expected,
            "masks by a named predicate, %p0 and p1 computed the wrong value");
    }
    qtx_refused("sread %a, LANE_ID\nsetp.ne.u32 %a, %a, 0\nhalt\n",
        "%a is used here as a predicate and on line 1 as a vector register", "one name as a register and a predicate");
    qtx_refused("mov %p0, 1\nhalt\n", "%p0 is the predicate p0, not a register", "%p0 as a register");
    // Eight predicate values held at once do not fit seven predicates; eight
    // names that are each done before the next begins share.
    std::string crowded = "sread %lane, LANE_ID\nmov %x, 0\n";
    std::string in_turn = crowded;
    for (int k = 0; k < 8; ++k) {
        crowded += "setp.eq.u32 %q" + std::to_string(k) + ", %lane, " + std::to_string(k) + "\n";
        in_turn += "setp.eq.u32 %q" + std::to_string(k) + ", %lane, " + std::to_string(k) + "\n@%q" +
            std::to_string(k) + " addi %x, %x, 1\n";
    }
    for (int k = 0; k < 8; ++k) crowded += "@%q" + std::to_string(k) + " addi %x, %x, 1\n";
    qtx_refused(crowded + "halt\n", "no predicate is left for %q7", "eight predicate values held at once");
    const auto shared = nanosm::compile_qtx(in_turn + "halt\n", profile);
    for (int k = 0; k < 8; ++k) {
        require(number_of(shared.register_names, "q" + std::to_string(k)) < 7, "a predicate name without a predicate");
    }
}

// qtx's registers (src/qtxc/registers.cpp, docs/qtx.md "Registers"): a register
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
    qtx_refused("umov %a, 1\naddi %b, %a, 1\nhalt\n",
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

    // Inside a loop the copy is of a source the loop does not write (%v,
    // though it comes first), and the warning's advice is that one; when the
    // loop writes every source of the crowded bank, the warning says the
    // copy stays.
    const std::string loop = R"(
        sread  %lane, LANE_ID
        addi   %x, %lane, 1
        addi   %y, %lane, 2
        addi   %z, %lane, 3
        mov    %v, 100
        umov   %n, 0
L:      add    %t0, %x, %y
        add    %t1, %y, %z
        add    %t2, %x, %z
        add    %t3, %x, %y
        add    %t4, %y, %z
        add    %t5, %x, %z
        add    %x, %v, %x
        add    %y, %v, %y
        add    %z, %v, %z
        uaddi  %n, %n, 1
        usetp.lt.u32 p0, %n, 2
@p0     bra.uni L
        add    %all, %x, %y
        add    %all, %all, %z
        halt
    )";
    const auto looped = nanosm::compile_qtx(loop, profile);
    require(looped.copies >= 1, "the loop's crowded read got no copy");
    for (const auto& warning : looped.warnings) {
        require(warning.find("qtxc inserted a copy of %v before it") != std::string::npos &&
                warning.find("it is inside a loop that does not write %v: copying it into a second name before the "
                             "loop, and reading that name here, avoids it") != std::string::npos,
            "the copy inside the loop is not of the loop-invariant %v: " + warning);
    }
    const auto loop_kernel = compile(loop, profile);
    nanosm::Machine loop_machine(profile, loop_kernel, launch_warps(1));
    loop_machine.run();
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(loop_machine.read_reg(0, lane, number_of(loop_kernel.register_names, "all")) == 606 + 3 * lane,
            "the loop's repaired instructions computed the wrong value");
    }
    std::string all_written = "mov %p, 1\nmov %q, 2\nmov %r, 3\nmov %s, 4\numov %n, 0\nL:\n";
    for (const auto& [left, right] : std::vector<std::pair<std::string, std::string>>{
             {"p", "q"}, {"p", "r"}, {"p", "s"}, {"q", "r"}, {"q", "s"}, {"r", "s"}}) {
        all_written += "add %" + left + right + ", %" + left + ", %" + right + "\n";
    }
    all_written += "addi %p, %p, 1\naddi %q, %q, 1\naddi %r, %r, 1\naddi %s, %s, 1\n"
                   "uaddi %n, %n, 1\nusetp.lt.u32 p0, %n, 2\n@p0 bra.uni L\nhalt\n";
    const auto stays = nanosm::compile_qtx(all_written, profile);
    require(stays.copies >= 1 && stays.warnings.front().find("so a copy made before the loop would not hold their "
                                                             "values here: the copy stays") != std::string::npos,
        "the warning for a loop that writes every source does not say the copy stays: " +
            (stays.warnings.empty() ? std::string("(none)") : stays.warnings.front()));

    // A loop with an early back edge (a `continue`) and its main one: %v is
    // written only between them, so it is not loop-invariant, and the copy's
    // advice must not say so. (The stretch to the early edge alone does not
    // write %v.)
    std::string early = loop;
    early.replace(early.find("@p0     bra.uni L"), std::string("@p0     bra.uni L").size(),
        "@p0     bra.uni L\n        addi   %v, %v, 1\n        usetp.lt.u32 p1, %n, 4\n@p1     bra.uni L");
    const auto continued = nanosm::compile_qtx(early, profile);
    require(continued.copies >= 1, "the early-continue loop's crowded read got no copy");
    bool names_v = false;
    for (const auto& warning : continued.warnings) {
        names_v = names_v || warning.find("%v") != std::string::npos;
        require(warning.find("inside a loop that does not write %v") == std::string::npos,
            "%v, written after the early back edge, was taken for loop-invariant: " + warning);
    }
    require(names_v, "no copy is beside %v, so the loop does not test the early back edge");
    const auto continued_kernel = compile(early, profile);
    nanosm::Machine continued_machine(profile, continued_kernel, launch_warps(1));
    continued_machine.run();
    for (uint32_t lane = 0; lane < 16; ++lane) {
        // x, y, z gain 100, 100, 101, 102 over the four passes.
        require(continued_machine.read_reg(0, lane, number_of(continued_kernel.register_names, "all")) ==
                1215 + 3 * lane, "the early-continue loop computed the wrong value");
    }
}

// One source of a per-lane ALU / FPU instruction may be a uniform register
// (src/isa.hpp takes_uniform_source). A name in that slot has the file the
// rest of the kernel gives it, also from a line below; a uniform multiplier
// of two groups of eight takes no bank, so fma needs no copy (and when its
// two vector sources share a bank, the copy is of one of them); and what is
// not allowed is refused with what is.
TEST(Registers, uniform_source) {
    const auto& profile = canonical_profile();
    // %step is made uniform by a line below its read (the code runs INIT first).
    const auto later = compile(R"(
        bra.uni INIT
BODY:   fadd %acc, %acc, %step
        halt
INIT:   umov %step, 1.5
        mov %acc, 0
        bra.uni BODY
    )", profile);
    nanosm::Machine machine(profile, later, launch_warps(1));
    machine.run();
    require(machine.read_reg(0, 3, number_of(later.register_names, "acc")) == nanosm::float_bits(1.5f),
        "fadd with a uniform source decided below it");

    std::string groups = ".reg %a<8>, %b<8>\numov %s, 0.5\n";
    for (int j = 0; j < 8; ++j) {
        const auto n = std::to_string(j);
        groups += "fma %a" + n + ", %a" + n + ", %s, %b" + n + "\n";
    }
    const auto scaled = nanosm::compile_qtx(groups + "fma %a0, %s, %b0, %a1\nhalt\n", profile);
    require(scaled.copies == 0 && scaled.warnings.empty(), "fma with a uniform multiplicand needed a copy");
    require(scaled.instructions.at(9).src1_uniform && scaled.instructions.at(9).text.find(", ur") != std::string::npos,
        "the uniform multiplicand written first did not move to the b slot: " + scaled.instructions.at(9).text);
    // %a0 beside %b0, %b1 and %b2 (three banks in a row): one of the fma's
    // shares %a0's bank. The copy is of a vector source; %s stays uniform.
    const auto crowded = nanosm::compile_qtx(".reg %a<8>, %b<8>\numov %s, 0.5\nfma %d0, %a0, %s, %b0\n"
        "fma %d1, %a0, %s, %b1\nfma %d2, %a0, %s, %b2\nhalt\n", profile);
    require(crowded.copies == 1 && crowded.warnings.size() == 1, "the fma whose vector sources share a bank got no copy");
    require(crowded.warnings.front().find("inserted a copy of %s") == std::string::npos &&
            (crowded.warnings.front().find("inserted a copy of %a0") != std::string::npos ||
             crowded.warnings.front().find("inserted a copy of %b") != std::string::npos),
        "the copy is not of a vector source: " + crowded.warnings.front());
    for (std::size_t index = 0; index + 1 < crowded.instructions.size(); ++index) {
        const auto& copy = crowded.instructions[index];
        if (copy.opcode != nanosm::Opcode::addi) continue;
        const auto& next = crowded.instructions[index + 1];
        require(next.opcode == nanosm::Opcode::fma && next.src1_uniform && next.text.find(", ur") != std::string::npos &&
                (next.src0 == copy.dst || next.src2 == copy.dst),
            "the fma behind the copy does not read it with its uniform multiplicand in place: " + next.text);
    }

    const std::string copy = "ur2r copies a uniform register into a vector register";
    qtx_refused("umov %u, 1\nsread %x, LANE_ID\nfsub %d, %u, %x\nhalt\n",
        "line 3: 'fsub %d, %u, %x' reads the uniform register %u as its first source; fsub takes a uniform register "
        "only as its second source (fsub d, x, U: x minus U); " + copy, "a uniform first source");
    qtx_refused("umov %u, 1\numov %v, 2\nadd %d, %u, %v\nhalt\n",
        "reads two uniform registers, %u and %v; a per-lane instruction takes one, as its second source", "two uniform sources");
    qtx_refused("umov %u, 1\numov %v, 2\nsread %x, LANE_ID\nfma %d, %v, %u, %x\nhalt\n",
        "reads two uniform registers, %v and %u", "two uniform multiplicands");
    qtx_refused("umov %u, 1\nsread %x, LANE_ID\nfma %d, %x, %x, %u\nhalt\n",
        "reads the uniform register %u as its addend; fma takes a uniform register only as a multiplicand", "a uniform addend");
    qtx_refused("sread %x, LANE_ID\nfadd %d, %x, %y\nhalt\n",
        "line 2: 'fadd %d, %x, %y' reads %y where a vector or a uniform register may go, and nothing else in the kernel "
        "says which it is", "a name nothing decides");
    qtx_refused("sread %x, LANE_ID\nsetp.lt.u32 p0, %u, %x\numov %u, 1\nhalt\n",
        "line 3: %u is used here as a uniform register and on line 2 as a vector register",
        "a uniform register met first as a first source");
    qtx_refused("umov %u, 1\nex2 %d, %u\nhalt\n", "is used here as a vector register", "a uniform ex2 source");
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

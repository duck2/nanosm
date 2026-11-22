// Shared memory: bank rounds and lds.uni (B3), uniformity (I1), port A (I2, I2.1), fence.sh and C10 (I4).
// Moved from the original tests/tests.cpp (trash/test-setup-2026-10-22.md);
// each test's body is unchanged, its TEST wrapper is at the end.
#include "test_support.hpp"

namespace {

using namespace nanosm_test;

// Issue to dependent issue behind a uniform-register writer (umov, usread,
// u* arithmetic, r2ur): visible issue_to_urf_visible after the writer's
// issue, sampled issue_to_urf_sample after the reader's (src/rf.hpp; 4 at
// canonical, where alu_distance is 6).
uint64_t uniform_distance(const nanosm::DeviceProfile& profile) {
    return nanosm::issue_to_urf_visible(profile) - nanosm::issue_to_urf_sample;
}

// ---------------------------------------------------------------------------
// B3 (rfc09 pin 3): uniformity is declared, never detected.
// - lds / sts: rounds = the most lanes on any one bank, bank = (address / 4)
//   % 16. Two lanes on one word are two lanes on one bank. Each bank serves
//   its lanes in ascending lane order, one per round.
// - lds.uni: one access at the address in a uniform register (uniform by
//   construction), the word fanned out to every lane, one RF write.
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
    auto& shmem = image_bytes(launch.test_shmem, b3_table + 64);
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
// same word takes sixteen rounds: still one RF write, at issue + 17. The
// lds.uni base is the uniform register ur0, the lds base the vector r0.
void test_b3_lds_uni_one_round_one_write() {
    for (const std::string operation : {"lds.uni", "lds"}) {
        const bool uniform = operation == "lds.uni";
        const uint32_t rounds = uniform ? 1 : 16;
        const std::string base = uniform ? "ur0" : "r0";
        // r1 differs in every lane before the load, so one write of the load
        // changes all sixteen lanes in one cycle.
        auto machine = b3_machine("sread r1, LANE_ID\n" + std::string(uniform ? "umov" : "mov") + " " + base +
            ", 60\n" + operation + " r1, [" + base + " + 16]\nhalt\n", b3_launch());
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
// a one-round lds does: from the load on, the same compiled controls and the
// same issue timeline. The plain lds reads word 16 + l in lane l (one lane
// per bank) through the vector base r0; lds.uni reads word 16 in every lane
// through the uniform base ur0, computed by uniform arithmetic. The address
// sources differ in timing (a uniform write -> read is 4 cycles at
// canonical, a vector one 6), so the controls before the load and the
// absolute cycles are not compared: the timeline is relative to the load's
// issue.
void test_b3_lds_uni_composes_like_lds() {
    const auto program = [](bool uniform) {
        const std::string base = uniform ? "ur0" : "r0";
        return std::string(uniform ? "umov ur0, 0\nushli " : "sread r0, LANE_ID\nshli ") + base + ", " + base +
            ", 2\n" + (uniform ? "lds.uni" : "lds") + " r1, [" + base + " + 64]\n" + (uniform ? "umov " : "mov ") +
            base + ", 7\n"       // WAR on the load's address register
            "fadd r2, r1, r1\n"  // RAW through the load's depbar
            "mov r1, 3\n"        // WAW on the load's destination
            "halt\n";
    };
    const auto plain = program(false);
    const auto uniform = program(true);
    constexpr uint32_t load_pc = 2 * 8;
    const auto timeline = [&](const nanosm::Machine& machine) {
        std::vector<std::pair<uint64_t, uint64_t>> issues; // cycle after the load's issue, pc
        const auto load = issued_at(machine, load_pc).at(0);
        for (const auto& event : machine.trace().events()) {
            if (is_issue_event(event) && event.id >= load_pc) {
                issues.emplace_back(event.start_cycle - load, event.id);
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
        require(plain_asm.instructions.size() == uniform_asm.instructions.size(), what + ": instruction counts differ");
        for (std::size_t i = load_pc / 8; i < plain_asm.instructions.size(); ++i) {
            const auto& a = plain_asm.instructions[i].control;
            const auto& b = uniform_asm.instructions[i].control;
            require(a.stall == b.stall && a.set_depbar == b.set_depbar && a.wait_mask == b.wait_mask,
                what + ": qtxc scheduled lds.uni differently from lds");
        }
        auto plain_run = b3_machine(plain, b3_launch(), profile);
        auto uniform_run = b3_machine(uniform, b3_launch(), profile);
        plain_run->run();
        uniform_run->run();
        require(timeline(*plain_run) == timeline(*uniform_run), what + ": issue timelines from the load on differ");
        require(plain_run->cycle() - issued_at(*plain_run, load_pc).at(0) ==
                uniform_run->cycle() - issued_at(*uniform_run, load_pc).at(0),
            what + ": run lengths after the load differ");
        require(uniform_run->read_ureg(0, 0) == 7, what + ": lds.uni WAR order");
        for (uint32_t lane = 0; lane < 16; ++lane) {
            require(plain_run->read_reg(0, lane, 2) == twice(b3_word(64 + 4 * lane)), what + ": lds value");
            require(uniform_run->read_reg(0, lane, 2) == twice(b3_word(64)), what + ": lds.uni value");
            require(uniform_run->read_reg(0, lane, 1) == 3, what + ": lds.uni WAW order");
        }
    }
}

// Alignment and bounds: the machine's own panics, as for lds, in both builds.
void test_b3_lds_uni_alignment_and_bounds() {
    const auto expect = [](const std::string& program, const std::string& needle, const std::string& what) {
        auto machine = b3_machine(program, b3_launch());
        expect_sim_panic(*machine, needle, what);
    };
    expect("umov ur0, 6\nlds.uni r1, [ur0]\nhalt\n", "unaligned 32-bit shmem access", "2-byte aligned lds.uni");
    expect("umov ur0, 65532\nlds.uni r1, [ur0 + 4]\nhalt\n", "shmem access out of bounds", "lds.uni past shmem");
    expect("umov ur0, 0\nlds.uni r1, [ur0 - 4]\nhalt\n", "shmem address arithmetic overflow",
        "negative lds.uni address");
    auto last_word = b3_machine("umov ur0, 65532\nlds.uni r1, [ur0]\nhalt\n", b3_launch());
    last_word->run();
    require(last_word->read_reg(0, 9, 1) == 0, "lds.uni of the last shmem word");
}

// ---------------------------------------------------------------------------
// I1: declared-uniform operands are a sanitizer rule (rfc09 pin 3). Hardware
// reads lane 0's value; with the sanitizer, a value that differs across lanes
// PANICs with the operand's name, as before. One test for both builds. Since
// the uniform register file only the operands still taken from a vector
// register is lane 0's: the bra.uni predicate (the lds.uni / ldg.uni bases
// and the mbar addresses are uniform registers, uniform by construction; and
// since 2026-11-21 ldg.v8 / stg.v8 take a per-lane address, so their cases
// are gone).
// ---------------------------------------------------------------------------

nanosm::Launch uniformity_launch() {
    nanosm::Launch launch;
    launch.active_warps = 1;
    fill_pattern(image_bytes(launch.gmem_image, 4096), 0, 1024, 0);
    auto& shmem = image_bytes(launch.test_shmem, 256);
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
        {"bra.uni predicate",
            "sread r0, LANE_ID\nsetp.lt.u32 p0, r0, 8\n@p0 bra.uni taken\nmov r5, 1\nhalt\ntaken:\nmov r5, 2\nhalt\n",
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

// qtxc and qtxas: opcode 0x26, the RRI payload (src0: the base's uniform
// register number), a depbar like every memory op.
void test_b3_lds_uni_encoding() {
    const auto& profile = canonical_profile();
    const auto kernel = compile("umov ur3, 0\nlds.uni r5, [ur3 + 8]\nhalt\n");
    const auto decoded = nanosm::decode(kernel.words[1]);
    require(static_cast<unsigned>(decoded.opcode) == 0x26 && decoded.dst == 5 && decoded.src0 == 3 &&
            decoded.imm == 8 && decoded.control.set_depbar < 6,
        "lds.uni did not survive qtxc + qtxas");
    require(nanosm::format_instruction(decoded, false) == "lds.uni r5, [ur3 + 8]", "lds.uni formatting");
    auto negative = decoded;
    negative.imm = -2048;
    require(nanosm::decode(nanosm::encode(negative)).imm == -2048, "lds.uni negative immediate");
    const auto header = ".profile " + profile.name + ", 0x" +
        nanosm::format_profile_fingerprint(nanosm::device_profile_fingerprint(profile)) + "\n";
    (void)nanosm::assemble_asm(header + R"(
        umov ur0, 0 {pred=PT, stall=1, set=-, wait=-}
        lds.uni r1, [ur0] {pred=PT, stall=0, set=sb0, wait=-}
        halt {pred=PT, stall=0, set=-, wait=sb0}
    )", profile);
    try {
        (void)nanosm::assemble_asm(header + R"(
            lds.uni r1, [ur0] {pred=PT, stall=0, set=-, wait=-}
            halt {pred=PT, stall=0, set=-, wait=-}
        )", profile);
        throw std::runtime_error("qtxas accepted lds.uni without a depbar");
    } catch (const nanosm::SimError& error) {
        require(std::string(error.what()).find("memory operation requires set=sb0..sb5") != std::string::npos,
            std::string("lds.uni without a depbar: wrong error: ") + error.what());
    }
}

// I3: holding A-read for a few cycles takes a second warp. Warp 1 runs one
// sixteen-round sts (all lanes on one bank: words 6144 + 64 * lane);
// i3_hold_a_read places the host's event while that store is under way. The
// RMW starts the next cycle (the store started last, so the alternation bit
// gives the RMW the next start) and waits for its read until the cycle after
// the store's last round, starting nothing new on A-read meanwhile
// (select_a_read step 2). Warp 0 runs `body` after the dispatch (p6, ur15,
// r20, r21, r29 are the dispatch's and warp 1's: usread reads the warp id,
// ur2r copies it to r29 for the setp).
std::string i3_with_blocker(const std::string& body) {
    return R"(
        usread ur15, WARP_ID
        ur2r r29, ur15
        setp.ne.u32 p6, r29, 0
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
// URF: P's base is the uniform ur1 (never written: 0), and the dispatch in
// i3_with_blocker is one instruction longer (usread + ur2r), so every pc is
// one instruction later; the cycles above are relative to issues and grants.
void test_i2_same_warp_order_with_reordered_reads() {
    const auto& profile = canonical_profile();
    require(i2_return(profile) == 4 && profile.shmem_load_slots == 2, "the timeline is derived for D = Lb = 2");
    const uint64_t t = nanosm::issue_to_dispatch(profile);
    const uint64_t c = profile.shmem_rmw_compute_cycles;
    auto machine = i2_machine(profile, i3_with_blocker(R"(
        sread r0, LANE_ID
        shli r0, r0, 2
        lds.uni r3, [ur1 + 256] {pred=PT, stall=0, set=sb4, wait=-}
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
    constexpr uint32_t p_pc = 6 * 8, blocker_pc = 16 * 8, r5_pc = 9 * 8;
    const auto p = issued_at(*machine, p_pc).at(0);
    require(issued_at(*machine, blocker_pc) == std::vector<uint64_t>{blocker - t - 1} &&
            reads == std::vector<I2Grant>{{p + t + 1, "A-read", "simt", 0}, {r, "A-read", "rmw", 0},
                         {r + 1, "A-read", "simt", 1}, {r + 2, "A-read", "simt", 0}},
        "A-read grants (blocker from c" + std::to_string(blocker) + "):" + describe(reads) + " | all" +
            describe(i2_grants(*machine)) + " | issues" + issue_log(*machine) + " | ordered" +
            describe(refusals(*machine, 0, "ordered behind own access")) + " slots" +
            describe(refusals(*machine, 0, "load slots full")));
    constexpr uint32_t old_pc = 7 * 8, young_pc = 8 * 8, war_pc = 10 * 8;
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

// Divergence (rfc12 section 3; trash/divergence-2026-11-21.md): rounds come
// from active lanes only; a masked lane is no access (its word, and an lds's
// destination lane, are untouched); a uniform operation under a partial
// mask is the sanitizer's PANIC (backlog D12; the machine's quiet lane-0
// behavior: "D12: uniform ops under a partial mask"). The mask is a real
// divergent region (masked_region). A warp's mask is never empty, so the
// old "no active lane: absent access" case is gone.
void test_i2_divergence() {
    const std::string body = R"(
        sread r0, LANE_ID
        shli r0, r0, 6
        mov r3, 0x77
        mov r5, 0x55
    )" + masked_region(0x00f0, R"(
        sts [r0], r5 {pred=PT, stall=0, set=sb0, wait=-}
        lds r3, [r0 + 4] {pred=PT, stall=0, set=sb1, wait=-}
    )") + R"(
        halt {pred=PT, stall=0, set=-, wait=sb0|sb1}
    )"; // every lane on bank 0: a full mask is sixteen rounds; lanes 4..7 active in the region
    const auto& profile = canonical_profile();
    {
        auto machine = i2_machine(profile, body);
        machine->run();
        require(shmem_request_rounds(*machine) == std::vector<uint32_t>{4, 4}, "four active lanes are four rounds");
        for (uint32_t lane = 0; lane < 16; ++lane) {
            const bool active = lane >= 4 && lane < 8;
            require(machine->read_shmem_u32(64 * lane) == (active ? 0x55 : 0x100 + 16 * lane), "sts lane");
            require(machine->read_reg(0, lane, 3) == (active ? 0x100 + 16 * lane + 1 : 0x77), "lds lane");
        }
    }
#if NANOSM_SANITIZER
    for (const std::string uniform : {"lds.uni r3, [ur1] {pred=PT, stall=0, set=sb0, wait=-}", "mbar.arrive [ur2]",
             "mbar.test_wait p0, [ur2], 0"}) {
        auto machine = i2_machine(profile, "umov ur2, 4096\n" + masked_region(0xfffe, uniform) + "halt\n");
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
        usread ur15, WARP_ID
        ur2r r29, ur15
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
// branch by branch_taken_interval (src/rf.hpp). (Before the URF the lds was
// the second half of its two-word packet: issuing it popped the packet and
// the reader's fetch contended with the writer's target fetch, one cycle
// more.) URF: the dispatch is usread + ur2r, one instruction more, so the lds
// is now the first half of its packet, with the halt: the reader asks for no
// fetch and the writer's target is fetched uncontended. So the sts issues
// 1 + taken - not taken + alu_distance after the lds (9 at canonical, was
// 10), each is granted t + 1 after its issue, and the two sixteen-cycle
// windows share 16 minus that many cycles -- in every one of which both
// services grant.
void test_i2_a_read_and_a_write_same_cycle() {
    auto machine = i2_machine(canonical_profile(), R"(
        usread ur15, WARP_ID
        ur2r r29, ur15
        sread r0, LANE_ID
        shli r0, r0, 6
        setp.ne.u32 p0, r29, 0
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
    const auto apart = 1 + nanosm::branch_taken_interval - nanosm::branch_not_taken_interval +
        alu_distance(canonical_profile());
    require(issued_at(*machine, 9 * 8).at(0) == issued_at(*machine, 6 * 8).at(0) + apart,
        "the sts did not issue 1 + taken - not taken + alu_distance after the lds:" + issue_log(*machine));
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
// re-places warp 0's three requests: the tma.load issues at c31 in both
// runs (b = c32: ready from b + 3, inside the hold that starts at b + 1).
// Two-word packets: in the primed run warp 0's first priming lds.uni (at
// c14) is the second half of its packet, so warp 0 asks for a fetch at c15,
// when warp 1 first asks for its branch target's packet; the round-robin
// serves warp 0 first, warp 1's target issues a cycle later and its
// blocking store starts at b = c36. The primed stall is therefore 17 (was
// 15): the tma.load issues at c33 and is ready at c37 = b + 1.
// URF: the dispatch (usread + ur2r) is one instruction longer and four
// cycles later, and the first priming lds.uni is now the first half of its
// packet, so that fetch contention is gone: warp 1's blocking store starts
// at b = c38 in both runs. The stalls are unchanged: the tma.load issues at
// c36 (plain; ready at c40 = b + 2) or c37 (primed; ready at c41 = b + 3),
// inside the hold that starts at b + 1.
// Uniform-register tma: three umovs (buffer, mbar, coordinate) replace the
// one mov, the last one stalling. Plain, they issue at c18..c20 (the mov was
// at c18); primed, at c20, c22, c23 (warp 1's first instruction after its
// branch takes c21; the mov was at c20). So the stalls are 16 / 14 (were
// 18 / 17) and the tma.load still issues at c36 / c37; its command reaches
// the TMA at issue + 3, the old dispatch, so it is ready at b + 2 / b + 3 as
// before. The lds.uni after it reads a uniform register, so issue refuses it
// in the cycle after the tma (the coordinate round holds the URF read ports):
// both lds.uni issue a cycle later, still ready inside the hold.
void test_i2_round_robin_tma_reads_and_loads() {
    const auto& profile = canonical_profile();
    require(i2_return(profile) == 4, "the timeline is derived for D = Lb = 2");
    const auto run = [&](bool primed) {
        nanosm::Launch launch;
        launch.active_warps = 2;
        auto& shmem = image_bytes(launch.test_shmem, 8192);
        nanosm::MbarState mbar;
        mbar.valid = true;
        mbar.expected = 1;
        mbar.tx_bytes = 64;
        write_u64(shmem, i2_event_mbar, nanosm::encode_mbar_state(mbar));
        write_rank1_load(shmem, 5120, 0, 16, 4104);
        fill_pattern(image_bytes(launch.gmem_image, 64), 0, 16, 0);
        const std::string priming = primed ? "lds.uni r5, [ur1 + 8] {pred=PT, stall=0, set=sb3, wait=-}\n"
                                             "lds.uni r6, [ur1 + 12] {pred=PT, stall=0, set=sb4, wait=-}\n"
                                           : "";
        const std::string stall = primed ? "14" : "16";
        // The TMA's operands (buffer at 1024 in ur10, completing into 4104 in
        // ur11, coordinate 0 in ur8); the last umov stalls.
        nanosm::Machine machine(profile, hand_assemble(i1_asm(profile, i3_with_blocker(priming +
            "umov ur10, 1024\numov ur11, 4104\numov ur8, 0 {pred=PT, stall=" + stall + ", set=-, wait=-}\n" + R"(
            tma.load.1d [ur10], [5120, ur8], [ur11]
            lds.uni r3, [ur1] {pred=PT, stall=0, set=sb0, wait=-}
            lds.uni r4, [ur1 + 4] {pred=PT, stall=0, set=sb1, wait=-}
            halt {pred=PT, stall=0, set=-, wait=sb0|sb1}
        )")), profile), std::move(launch), nanosm::TraceOptions{.timeline = true, .debug = true});
        const auto blocker = i3_hold_a_read(machine);
        machine.run();
        const uint64_t r = blocker + 16; // the RMW's read
        // warp 0's tma.load and two lds.uni (the last three before halt)
        // became ready (the tma's command in the TMA queue, an lds.uni's
        // dispatch, + 1) while A-read was held
        const uint32_t tma_pc = (primed ? 9 : 7) * 8;
        const auto ready = [&](uint32_t pc) {
            const auto reach = pc == tma_pc ? nanosm::issue_to_tma_command(1, profile) : nanosm::issue_to_dispatch(profile);
            return issued_at(machine, pc).front() + reach + 1;
        };
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
    auto& shmem = image_bytes(launch.test_shmem, 8192);
    for (uint32_t i = 0; i < 1024; ++i) write_u16(shmem, 2 * i, 0x3c00); // A and B: 1.0
    write_rank1_load(shmem, 5120, 0, 256, 4104);
    fill_pattern(image_bytes(launch.gmem_image, 1024), 0, 256, 0);
    std::string program = R"(
        usread ur15, WARP_ID
        ur2r r29, ur15
        setp.ne.u32 p0, r29, 0
        @p0 bra.uni hammer
        umov ur0, 0
        umov ur1, 1024
        umov ur4, 4096
        umov ur2, 0
        mbar.init [ur4], 1
        mma.async.init.m64n64k8 [ur2], [ur0], [ur1]
        mma.commit [ur4]
    )" + tma_operands_asm(10, 2048, 11, 4104, 8) + R"(
        tma.load.1d [ur10], [5120, ur8], [ur11]
    tma_wait:
        mbar.try_wait p1, [ur11], 0
        @!p1 bra.uni tma_wait
    mma_wait:
        mbar.try_wait p2, [ur4], 0
        @!p2 bra.uni mma_wait
        umov ur12, 0
        acc.ld r16, [ur12, 64]
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
        auto machine = i2_machine(profile, "lds.uni r3, [ur1] {pred=PT, stall=0, set=sb0, wait=-}\n"
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
// URF: the role dispatch is usread + ur2r, four cycles longer than the old
// sread, so warp 0's sts reaches its slot four cycles later (c31); Lb is 28
// (was 24) so the pause still ends after that, at w = 32.
void test_i2_read_first_across_services() {
    auto profile = canonical_profile();
    profile.shmem_read_latency_cycles = 28;
    const uint64_t w = profile.shmem_port_a_depth + profile.shmem_read_latency_cycles + profile.shmem_rmw_compute_cycles;
    std::set<std::string> sides;
    for (uint32_t stall = 0; stall <= 31; ++stall) {
        auto machine = i2_machine(profile, R"(
            usread ur15, WARP_ID
            ur2r r29, ur15
            sread r0, LANE_ID
            shli r0, r0, 2
            setp.ne.u32 p6, r29, 0
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
        qtx << "umov ur0, 0\numov ur1, 1024\numov ur4, 896\numov ur2, 0\n"
            << "sread r20, LANE_ID\nshli r20, r20, 2\n"
            << "mov r21, 16384\nshli r22, r21, 16\nor r21, r21, r22\n" // two fp16 2.0 halves
            << "sts [r20], r21\n";
        for (uint32_t i = 0; i < pad; ++i) qtx << "mov r22, 1\n";
        qtx << "mma.async.init.m32n64k8 [ur2], [ur0], [ur1]\nmma.commit [ur4]\n"
            << "wait:\nmbar.try_wait p0, [ur4], 0\n@!p0 bra.uni wait\n"
            << "umov ur8, 0\nacc.ld r10, [ur8, 64]\nhalt\n";
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.test_shmem, 2048);
        for (uint32_t i = 0; i < 256; ++i) write_u16(shmem, 2 * i, 0x3c00);        // A: 1.0
        for (uint32_t i = 0; i < 512; ++i) write_u16(shmem, 1024 + 2 * i, 0x3c00); // B: 1.0
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
            mma.run(10000);
        } catch (const nanosm::SimError& error) {
            panic = error.what();
        }
#if NANOSM_SANITIZER
        {
            // The MMA's accept: its dispatch (I6a: its issue, pc = 8 * (10 + pad), +
            // issue_to_dispatch), or the PANIC it raised there.
            std::vector<I2Grant> stores;
            for (const auto& grant : i2_grants(mma, "A-write")) {
                if (grant.client == "simt") stores.push_back(grant);
            }
            const auto what = "pad " + std::to_string(pad) + ":" + describe(stores);
            const uint64_t accept = panic.has_value()
                ? std::stoull(panic->substr(panic->find("cycle=") + 6))
                : issued_at(mma, 8 * (10 + pad)).at(0) + nanosm::issue_to_dispatch(dense);
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
        // pipeline has port A's depth); before I7 it read the RAM at g. Row 0's
        // four A words: words 0, 1 are rounds 0, 1 of step 0 (k block 0), words
        // 2, 3 rounds 0, 1 of step 128 (k block 1: 8 m x 16 n blocks later).
        const std::array<uint64_t, 4> sampled{step_start.at(0) + depth, step_start.at(0) + 1 + depth,
            step_start.at(128) + depth, step_start.at(128) + 1 + depth};
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

// F1 (rfc12 section 2: every latency is a nonzero number of register
// stages): the RMW engine's update crosses shmem.rmw_compute_cycles (C) ALU
// stages. A SIMT mbar.arrive issued at i reaches the SIMT RMW register at its
// dispatch (I6a: i + issue_to_dispatch) and has its read granted the cycle
// after, r; its words reach the port output at r + D + Lb; the update is in
// the ALU's last stage C cycles later and its write is granted then; the
// write reaches the RAM -- the phase flips at that clock edge -- D cycles
// after that. The engine is busy (its trace slice) D + Lb + C + 1 cycles.
// (Before I6a the arrive issued at c1 and was read at c2; now it waits for
// the umov's ur4 -- src/rf.hpp -- and for its dispatch.)
void test_i21_rmw_compute_stage() {
    i21_check_optional_field("shmem", "rmw_compute_cycles", 2, &nanosm::DeviceProfile::shmem_rmw_compute_cycles);
    for (const uint32_t c : {1u, 2u, 3u}) {
        auto profile = canonical_profile();
        profile.shmem_rmw_compute_cycles = c;
        const uint64_t d = profile.shmem_port_a_depth;
        const uint64_t lb = profile.shmem_read_latency_cycles;
        auto machine = i21_machine(profile, R"(
            umov ur4, 4160
            mbar.arrive [ur4]
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
// first umov's ur4: uniform_distance), reaches the MMA queue at its dispatch
// m + issue_to_dispatch and presents at P = m + issue_to_dispatch + 1. The
// arrive issues at m + 1 and reaches the SIMT RMW register at its own
// dispatch, P: the cycle the commit is presented, as before. Both are held
// at P + 1 and the arrive is served first. (I6b-2: the first umov issues at
// fetch_to_issue, so m is that + uniform_distance; before the URF the
// addresses were vector movs and m was that + alu_distance.)
void test_i21_simt_rmw_admission_registered_state() {
    const auto& profile = canonical_profile();
    auto machine = i21_machine(profile, R"(
        umov ur4, 4160
        umov ur5, 4168
        mma.commit [ur4]
        mbar.arrive [ur5]
        halt
    )", {{4160, i21_mbar(1, 1)}, {4168, i21_mbar(2, 2)}});
    const auto m = nanosm::fetch_to_issue + uniform_distance(profile);
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
            umov ur4, 4160
            mov r5, 0x5a
            sts [r0 + 8], r5 {pred=PT, stall=0, set=sb0, wait=-}
            lds r6, [r0 + 8] {pred=PT, stall=0, set=sb1, wait=-}
            mbar.arrive [ur4]
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
        const auto s = nanosm::fetch_to_issue + 2 + 2 * alu_distance(profile); // sread, shli; umov ur4, mov r5; the sts
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
            umov ur4, 6144
            mov r9, 0
            sts [r2 + 6144], r6 {pred=PT, stall=0, set=sb0, wait=-}
            mbar.test_wait p0, [ur4], 0
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
            usread ur15, WARP_ID
            ur2r r29, ur15
            setp.ne.u32 p1, r29, 0
            bra.uni w1 {pred=p1, stall=0, set=-, wait=-}
            umov ur4, 4096
            mov r9, 0
            mbar.try_wait p0, [ur4], 0
            bra.uni w0done {pred=!p0, stall=0, set=-, wait=-}
            mov r9, 1
        w0done: halt
        w1: umov ur5, 4176
            mov r10, 0 {pred=PT, stall=)" + std::to_string(warp1_stall) + R"(, set=-, wait=-}
            mbar.try_wait p2, [ur5], 0
            bra.uni w1done {pred=!p2, stall=0, set=-, wait=-}
            mov r10, 1
        w1done: halt
        )";
    };
    const std::vector<std::pair<uint32_t, nanosm::MbarState>> mbars{
        {i2_event_mbar, i21_mbar(1, 0, 64)}, {4176, i21_mbar(1, 1)}};
    constexpr uint32_t warp0_try_wait = 6 * 8;
    constexpr uint32_t warp1_try_wait = 12 * 8;
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
// A one-dimensional plain TMA store of `sectors` 32 B sectors (one A-read round
// each) under a fast bus: the cycles its source rounds' words returned
// (edge probe tma_store_round), in order, and the run's length. The warp
// halts instead of polling the mbar: a try_wait probe would share A-read's
// round-robin with the store reads (legal contention, not what is measured).
std::pair<std::vector<uint64_t>, uint64_t> i21_tma_store_returns(const nanosm::DeviceProfile& config, uint32_t sectors) {
    const uint32_t bytes = 32 * sectors;
    nanosm::Launch launch;
    launch.active_warps = 1;
    auto& shmem = image_bytes(launch.test_shmem, bytes + 256);
    fill_pattern(shmem, 0, bytes / 4, 0);
    write_rank1_tma(shmem, bytes, 0, bytes / 4, bytes + 128);
    image_bytes(launch.gmem_image, bytes);
    const auto program = tma_operands_asm(0, 0, 1, bytes + 128, 2) +
        "tma.store.1d [" + std::to_string(bytes) + ", ur2], [ur0], [ur1]\nhalt\n";
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
            sread r0, LANE_ID
            umov ur8, 0
            umov ur9, 0
            umov ur1, 0
            umov ur2, 1024
        )"; // ur8: the acc.ld's tmem pointer, ur9 the matmul's (both 0)
        body += "mov r23, 77\n"; // nonzero: the acc.ld's write of the old (zero) tmem value shows
        for (uint32_t i = 0; i < 6; ++i) body += "setp.eq.u32 p" + std::to_string(5 + i % 2) + ", r0, 0\n";
        body += R"(
            fadd r26, r10, r10
            fadd r29, r10, r10
            fadd r32, r10, r10
            fadd r35, r10, r10
            setp.eq.u32 p5, r0, 0
            setp.eq.u32 p6, r0, 0
            setp.eq.u32 p5, r0, 0
            acc.ld.v8 r16, [ur8, 64] {pred=PT, stall=0, set=sb0, wait=-}
        )";
        // One filler per cycle: they rotate over six predicates (I6a: a setp
        // waits until the previous write of its predicate is in the file,
        // issue_to_predicate_visible after it, so two alternating predicates
        // would issue in pairs five cycles apart).
        for (uint32_t i = 0; i < gap; ++i) body += "setp.eq.u32 p" + std::to_string(1 + i % 6) + ", r0, 0\n";
        body += R"(
            mma.async.init.m64n64k8 [ur9], [ur1], [ur2] {pred=PT, stall=0, set=sb1, wait=-}
            halt {pred=PT, stall=0, set=-, wait=sb0|sb1}
        )";
        return body;
    };
    const auto machine_for = [&](uint32_t gap) {
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.test_shmem, 4096);
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
    auto& shmem = image_bytes(launch.test_shmem, 8192);
    for (uint32_t word = 0; word < 2048; ++word) write_u32(shmem, 4 * word, 0x100 + word);
    if (setup) setup(shmem);
    auto& gmem = image_bytes(launch.gmem_image, 4096);
    for (uint32_t word = 0; word < 1024; ++word) write_u32(gmem, 4 * word, 5 * word + 2);
    const auto kernel = hand_assembly ? hand_assemble(i1_asm(profile, program), profile)
                                      : compile(program, profile);
    return std::make_unique<nanosm::Machine>(profile, kernel, std::move(launch),
        nanosm::TraceOptions{.timeline = true, .debug = true});
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
        "mov r8, 0\n" + i4_tma_operands(0) + "sts [r8], r8\ntma.store.1d [5120, ur8], [ur10], [ur11]\nhalt\n", profile);
    require(std::none_of(unfenced.instructions.begin(), unfenced.instructions.end(),
                [](const auto& inst) { return inst.opcode == nanosm::Opcode::fence_sh; }) &&
            unfenced.instructions.size() == 7,
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

        // The TMA's operand registers are written first (I6a): their fixed
        // distances are long met when the TMA follows the fence, so qtxc's
        // static schedule puts no stall between them.
        auto engine = i4_machine(profile, i4_tma_operands(0) + R"(
            umov ur12, 4104
            sread r0, LANE_ID
            shli r0, r0, 2
            mov r5, 0x777
            sts [r0], r5 {pred=PT, stall=0, set=sb0, wait=-}
            fence.sh
            tma.store.1d [5120, ur8], [ur10], [ur11]
        wait: mbar.try_wait p0, [ur12], 0
            bra.uni wait {pred=!p0, stall=0, set=-, wait=-}
            halt {pred=PT, stall=0, set=-, wait=sb0}
        )", i4_tma(1024));
        engine->run(); // the sanitizer build: no missing-fence PANIC
        const uint64_t ge = first + 4 + 2 * distance + 1 + t + 1; // four umovs, then as in `one`
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
//   sts -> tma.load whose DESCRIPTOR the sts just wrote (copied in through
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
            mbar.try_wait p0, [ur4], 0
            @!p0 bra.uni wait
            halt
        )";
    };
    // ur10, ur11, ur8: the TMA's operands (its buffer at `shared`, completing
    // into 4104, coordinate 0); ur4: that mbar, for the wait.
    const auto prologue = [](uint32_t shared) {
        return i4_tma_operands(shared) + R"(
        umov ur4, 4104
        sread r20, LANE_ID
        shli r20, r20, 2
    )";
    };
    const std::string tma_store = "tma.store.1d [5120, ur8], [ur10], [ur11]\n";
    const std::string tma_load = "tma.load.1d [ur10], [5120, ur8], [ur11]\n";

    // sts -> TMA store source.
    const auto store_source = prologue(0) + "mov r21, 7\nsts [r20], r21\n";
    auto message = i4_panic(*i4_machine(profile, program(store_source, false, tma_store),
        i4_tma(1024), 1, false), "sts -> TMA store source");
    require_mentions(message, {"reason=missing fence.sh: tma store #0 source bytes [0, 64) overlap this warp's earlier "
        "sts [r20], r21 (pc 56, issued cycle", "whose write has not reached the RAM; issue fence.sh between them",
        "inst='tma.store.1d [5120, ur8], [ur10], [ur11]"}, "sts -> TMA store source");
    auto fenced = i4_machine(profile, program(store_source, true, tma_store), i4_tma(1024), 1, false);
    fenced->run();
    for (uint32_t i = 0; i < 256; ++i) {
        require(fenced->read_gmem_u32(4 * i) == (i < 16 ? 7u : 0x100 + i), "fenced TMA store word " + std::to_string(i));
    }

    // sts -> the descriptor: copied from 6144 into 5120 (zeros: invalid).
    const auto descriptor_copy = [](std::vector<uint8_t>& shmem) {
        write_tma_descriptor(shmem, 6144, linear_tma(64));
        for (uint32_t i = 0; i < 16; ++i) write_u32(shmem, 5120 + 4 * i, 0);
        write_tx_mbar(shmem, 4104, 64);
    };
    const auto copy = prologue(1024) + "lds r21, [r20 + 6144]\nsts [r20 + 5120], r21\n";
    message = i4_panic(*i4_machine(profile, program(copy, false, tma_load), descriptor_copy, 1, false),
        "sts -> TMA descriptor");
    require_mentions(message, {"reason=missing fence.sh: tma #0 descriptor bytes [5120, 5184) overlap this warp's earlier "
        "sts [r20 + 5120], r21", "whose write has not reached the RAM"}, "sts -> TMA descriptor");
    fenced = i4_machine(profile, program(copy, true, tma_load), descriptor_copy, 1, false);
    fenced->run();
    for (uint32_t i = 0; i < 16; ++i) {
        require(fenced->read_shmem_u32(1024 + 4 * i) == 5 * i + 2, "the TMA with the copied descriptor, word " + std::to_string(i));
    }

    // sts -> mma.async on that tile (one bank: the sts is sixteen rounds).
    const auto dense = dense_one_bank_profile(4);
    const auto mma_machine = [&](bool fence) {
        nanosm::Launch launch;
        launch.active_warps = 1;
        auto& shmem = image_bytes(launch.test_shmem, 2048);
        for (uint32_t i = 0; i < 256; ++i) write_u16(shmem, 2 * i, 0x3c00);        // A: 1.0
        for (uint32_t i = 0; i < 512; ++i) write_u16(shmem, 1024 + 2 * i, 0x3c00); // B: 1.0
        write_count_mbar(shmem, 896, 1);
        return std::make_unique<nanosm::Machine>(dense, compile(std::string(R"(
            umov ur0, 0
            umov ur1, 1024
            umov ur4, 896
            umov ur2, 0
            sread r20, LANE_ID
            shli r20, r20, 2
            mov r21, 16384
            shli r22, r21, 16
            or r21, r21, r22
            sts [r20], r21
        )") + (fence ? "fence.sh\n" : "") + R"(
            mma.async.init.m32n64k8 [ur2], [ur0], [ur1]
            mma.commit [ur4]
        wait:
            mbar.try_wait p0, [ur4], 0
            @!p0 bra.uni wait
            umov ur8, 0
            acc.ld r10, [ur8, 64]
            halt
        )", dense), std::move(launch));
    };
    message = i4_panic(*mma_machine(false), "sts -> mma.async");
    require_mentions(message, {"reason=missing fence.sh: mma.async #1 A operand bytes [0, 64) overlap this warp's earlier "
        "sts [r20], r21", "inst='mma.async.init.m32n64k8 [ur2], [ur0], [ur1]"}, "sts -> mma.async");
    auto mma = mma_machine(true);
    mma->run(10000);
    require(nanosm::bits_float(mma->read_reg(0, 0, 10)) == 16.0f, "the fenced MMA did not read the stored 2.0 halves");

    // lds -> TMA load over the words it reads.
    const auto read_then_load = prologue(0) + "lds r21, [r20]\n";
    message = i4_panic(*i4_machine(profile, program(read_then_load, false, tma_load),
        i4_tma(1024), 1, false), "lds -> TMA load");
    require_mentions(message, {"reason=missing fence.sh: tma load #0 destination bytes [0, 64) overlap this warp's earlier "
        "lds r21, [r20] (pc 48, issued cycle", "whose read has not sampled the RAM; issue fence.sh between them"},
        "lds -> TMA load");
    fenced = i4_machine(profile, program(read_then_load, true, tma_load), i4_tma(1024), 1, false);
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
// bytes and PANICked from warp 1's side). Uniform-register tma: its three
// operand umovs take the slots of the three movs, and its command enters the
// TMA queue (the accept) at issue + 3, the old dispatch: the same timeline.
void test_i4_cross_warp_race() {
    const auto& profile = canonical_profile();
    // The words are 1024 + 64 * lane (bank 0); the TMA's buffer is [1024,
    // 2048). (They were 6144 + once; the timeline depends only on the bank.)
    // The last of the TMA's operand umovs stalls.
    const auto two_warps = [](const std::string& access, const std::string& command) {
        return R"(
            usread ur15, WARP_ID
            ur2r r29, ur15
            setp.ne.u32 p6, r29, 0
            bra.uni other {pred=p6, stall=0, set=-, wait=-}
            umov ur10, 1024
            umov ur11, 4104
            umov ur8, 0 {pred=PT, stall=20, set=-, wait=-}
            )" + command + R"(
            umov ur4, 4104
        wait: mbar.try_wait p0, [ur4], 0
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
    auto machine = i4_machine(profile, two_warps("sts [r20 + 1024], r21 {pred=PT, stall=0, set=sb0, wait=-}",
        "tma.store.1d [5120, ur8], [ur10], [ur11]"), i4_tma(1024), 2);
    auto message = i4_panic(*machine, "warp 1 sts vs warp 0 TMA store");
    require_mentions(message, {"warp=0", "reason=shmem race: tma store #0 source bytes [",
        "overlap warp 1's sts [r20 + 1024], r21 (pc", "whose write has not reached the RAM; order them with an mbar"},
        "cross-warp sts vs TMA store");
    under_way(*machine, message, "A-write");
    machine = i4_machine(profile, two_warps("lds r22, [r20 + 1024] {pred=PT, stall=0, set=sb0, wait=-}",
        "tma.load.1d [ur10], [5120, ur8], [ur11]"), i4_tma(1024), 2);
    message = i4_panic(*machine, "warp 1 lds vs warp 0 TMA load");
    require_mentions(message, {"warp=0", "reason=shmem race: tma load #0 destination bytes [",
        "overlap warp 1's lds r22, [r20 + 1024] (pc", "whose read has not sampled the RAM; order them with an mbar"},
        "cross-warp lds vs TMA load");
    under_way(*machine, message, "A-read");

    auto handoff = i4_machine(profile, R"(
            usread ur15, WARP_ID
            ur2r r29, ur15
            setp.ne.u32 p6, r29, 0
            bra.uni producer {pred=p6, stall=0, set=-, wait=-}
        )" + i4_tma_operands(1024) + R"(
            umov ur4, 4104
            umov ur5, 4112
        ready: mbar.try_wait p0, [ur5], 0
            bra.uni ready {pred=!p0, stall=0, set=-, wait=-}
            tma.store.1d [5120, ur8], [ur10], [ur11]
        stored: mbar.try_wait p0, [ur4], 0
            bra.uni stored {pred=!p0, stall=0, set=-, wait=-}
            halt
        producer: sread r20, LANE_ID
            shli r20, r20, 6
            mov r21, 0x66
            umov ur5, 4112
            sts [r20 + 1024], r21 {pred=PT, stall=0, set=sb0, wait=-}
            mbar.arrive [ur5]
            halt {pred=PT, stall=0, set=-, wait=sb0}
        )", [](std::vector<uint8_t>& shmem) {
            i4_tma(1024)(shmem);
            write_count_mbar(shmem, 4112, 1);
        }, 2);
    handoff->run();
    for (uint32_t i = 0; i < 256; ++i) {
        require(handoff->read_gmem_u32(4 * i) == (i % 16 == 0 ? 0x66u : 0x100 + 256 + i), "handoff word " + std::to_string(i));
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
    auto& shmem = image_bytes(launch.test_shmem, 2048);
    for (uint32_t i = 0; i < 256; ++i) write_u16(shmem, 2 * i, 0x3c00);        // A: 1.0
    for (uint32_t i = 0; i < 512; ++i) write_u16(shmem, 1024 + 2 * i, 0x3c00); // B: 1.0
    write_count_mbar(shmem, 896, 1);
    nanosm::Machine mma(dense, compile(R"(
        umov ur0, 0
        umov ur1, 1024
        umov ur4, 896
        umov ur2, 0
        sread r20, LANE_ID
        shli r20, r20, 2
        lds r21, [r20]
        mma.async.init.m32n64k8 [ur2], [ur0], [ur1]
        mma.commit [ur4]
    wait:
        mbar.try_wait p0, [ur4], 0
        @!p0 bra.uni wait
        umov ur8, 0
        acc.ld r10, [ur8, 64]
        halt
    )", dense), std::move(launch), nanosm::TraceOptions{.timeline = true, .debug = true});
    mma.run(10000);
    const auto reads = i2_grants(mma, "A-read");
    const auto accept = issued_at(mma, 7 * 8);
    require(reads.size() >= 16 && accept.size() == 1 && reads[15].cycle + dense.shmem_port_a_depth > accept.front(),
        "the lds was not still in flight when the MMA was accepted:" + describe(reads));
    require(nanosm::bits_float(mma.read_reg(0, 0, 10)) == 8.0f && mma.read_reg(0, 3, 21) == 0x3c003c00u,
        "lds + MMA read/read results");

    const auto& profile = canonical_profile();
    auto store = i4_machine(profile, i4_tma_operands(0) + R"(
        umov ur4, 4104
        sread r20, LANE_ID
        shli r20, r20, 2
        lds r21, [r20] {pred=PT, stall=0, set=sb0, wait=-}
        tma.store.1d [5120, ur8], [ur10], [ur11]
    wait: mbar.try_wait p0, [ur4], 0
        bra.uni wait {pred=!p0, stall=0, set=-, wait=-}
        halt {pred=PT, stall=0, set=-, wait=sb0}
    )", i4_tma(1024));
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
    // The mbar's address is ur4; ur2r copies it into r4 for the lanes' addresses.
    const std::string lanes = R"(
        sread r20, LANE_ID
        shli r20, r20, 2
        ur2r r4, ur4
        add r20, r20, r4
        mov r21, 9
    )";
    auto message = i4_panic(*i4_machine(profile, "umov ur4, 4160\nmbar.init [ur4], 1\n" + lanes + "sts [r20], r21\nhalt\n",
        {}, 1, false), "sts onto a live mbar");
    require_mentions(message, {"reason=live mbar opacity: sts writes bytes [4160, 4168) of the live mbar [4160, 4168) "
        "(mbar.init by warp 0 pc 8 at cycle ", "until kernel end only mbar operations may touch it",
        "inst='sts [r20], r21"}, "sts onto a live mbar");
    auto repurpose = i4_machine(profile, "umov ur4, 4160\n" + lanes + "sts [r20], r21\nmbar.init [ur4], 1\nhalt\n", {}, 1, false);
    repurpose->run();
    const auto state = repurpose->read_mbar_state(4160);
    require(state.valid && state.expected == 1 && repurpose->read_shmem_u32(4168) == 9,
        "an sts before the mbar.init was refused, or the init did not land after it");

    const std::string placement = i4_tma_operands(1024) + R"(
        umov ur4, 4104
        umov ur5, 1032
        mbar.init [ur5], 1
    )";
    const std::string load = R"(
        tma.load.1d [ur10], [5120, ur8], [ur11]
    wait:
        mbar.try_wait p0, [ur4], 0
        @!p0 bra.uni wait
        halt
    )";
    message = i4_panic(*i4_machine(profile, placement + "fence.sh\n" + load, i4_tma(64), 1, false),
        "TMA placement over a live mbar");
    require_mentions(message, {"mbar address=1032 reason=live mbar opacity: tma load #0 placement writes bytes "
        "[1032, 1040) of the live mbar [1032, 1040) (mbar.init by warp 0 pc 40 at cycle "}, "TMA placement over a live mbar");
    message = i4_panic(*i4_machine(profile, placement + load, i4_tma(64), 1, false),
        "TMA load over an mbar.init in flight");
    require_mentions(message, {"reason=missing fence.sh: tma load #0 destination bytes [1032, 1040) overlap this warp's "
        "earlier mbar.init [ur5], 1 (pc 40"}, "TMA load over an mbar.init in flight");

    message = i4_panic(*i4_machine(profile, "umov ur4, 4160\nmbar.init [ur4], 1\nmbar.init [ur4], 2\nhalt\n", {}, 1, false),
        "mbar.init over a live mbar");
    require_mentions(message, {"reason=mbar.init over the live mbar [4160, 4168) (mbar.init by warp 0 pc 8 at cycle ",
        "(v0 has no mbar.inval: an mbar is initialized once per kernel)", "inst='mbar.init [ur4], 2"},
        "mbar.init over a live mbar");
}

} // namespace

// B3: lds.uni one round, one RF write
TEST(Shmem, b3_lds_uni_one_round_one_write) {
    test_b3_lds_uni_one_round_one_write();
}

// B3: lds k lanes on one word
TEST(Shmem, b3_lds_k_lanes_on_one_word) {
    test_b3_lds_k_lanes_on_one_word();
}

// B3: rounds = max lanes per bank
TEST(Shmem, b3_rounds_are_max_lanes_per_bank) {
    test_b3_rounds_are_max_lanes_per_bank();
}

// B3: sts bank serves lanes in order
TEST(Shmem, b3_sts_bank_serves_lanes_in_order) {
    test_b3_sts_bank_serves_lanes_in_order();
}

// B3: lds.uni composes like lds
TEST(Shmem, b3_lds_uni_composes_like_lds) {
    test_b3_lds_uni_composes_like_lds();
}

// B3: lds.uni alignment and bounds
TEST(Shmem, b3_lds_uni_alignment_and_bounds) {
    test_b3_lds_uni_alignment_and_bounds();
}

// B3: lds.uni encoding
TEST(Shmem, b3_lds_uni_encoding) {
    test_b3_lds_uni_encoding();
}

// uniformity: hardware uses lane 0
TEST(Shmem, uniformity_uses_lane_zero) {
    test_uniformity_uses_lane_zero();
}

// I2: same-warp RAW/WAR/WAW with reordered reads
TEST(Shmem, i2_same_warp_order_with_reordered_reads) {
    test_i2_same_warp_order_with_reordered_reads();
}

// I2: divergence
TEST(Shmem, i2_divergence) {
    test_i2_divergence();
}

// I2: store slots 1 and 2
TEST(Shmem, i2_store_slots_one_and_two) {
    test_i2_store_slots_one_and_two();
}

// I2: store slot reuse and done()
TEST(Shmem, i2_store_slot_reuse_and_done) {
    test_i2_store_slot_reuse_and_done();
}

// I2: A-read and A-write in one cycle
TEST(Shmem, i2_a_read_and_a_write_same_cycle) {
    test_i2_a_read_and_a_write_same_cycle();
}

// I2: one round-robin over TMA reads and loads
TEST(Shmem, i2_round_robin_tma_reads_and_loads) {
    test_i2_round_robin_tma_reads_and_loads();
}

// I2: TMA placement on the port-B write
TEST(Shmem, i2_tma_placement_on_port_b) {
    test_i2_tma_placement_on_port_b();
}

// I3: store pause, two warps (was I2: RMW holds both A services)
TEST(Shmem, i3_store_pause_two_warps) {
    test_i3_store_pause_two_warps();
}

// I2: profile fields D, Lb, store slots
TEST(Shmem, i2_profile_fields) {
    test_i2_profile_fields();
}

// I2: READ_FIRST across services
TEST(ShmemSlow, i2_read_first_across_services) {
    test_i2_read_first_across_services();
}

// I2.1: RMW compute stage C
TEST(Shmem, i21_rmw_compute_stage) {
    test_i21_rmw_compute_stage();
}

// I2.1: SIMT RMW admission reads registered state
TEST(Shmem, i21_simt_rmw_admission_registered_state) {
    test_i21_simt_rmw_admission_registered_state();
}

// I2.1: SIMT RMW behind own lds and sts
TEST(Shmem, i21_simt_rmw_behind_own_lds_and_sts) {
    test_i21_simt_rmw_behind_own_lds_and_sts();
}

// I2.1: probe behind own sts
TEST(Shmem, i21_probe_behind_own_sts) {
    test_i21_probe_behind_own_sts();
}

// I2.1: stale probe after a wake
TEST(ShmemSlow, i21_stale_probe_after_wake) {
    test_i21_stale_probe_after_wake();
}

// I2.1: RMW read behind an A-read transaction
TEST(Shmem, i21_rmw_read_behind_a_read_transaction) {
    test_i21_rmw_read_behind_a_read_transaction();
}

// I2.1: TMA store sector registers
TEST(ShmemSlow, i21_tma_store_sector_registers) {
    test_i21_tma_store_sector_registers();
}

// I2.1: acc.ld read-ownership ends at the last tmem read
TEST(Shmem, i21_acc_ld_read_ownership) {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    test_i21_acc_ld_read_ownership();
}

// I4: fence.sh encoding, qtxc and hand assembly
TEST(Shmem, i4_fence_encoding_qtxc_and_assembly) {
    test_i4_fence_encoding_qtxc_and_assembly();
}

// I4: fence.sh waits until its warp's accesses reach the RAM
TEST(Shmem, i4_fence_waits_until_own_accesses_reach_ram) {
    test_i4_fence_waits_until_own_accesses_reach_ram();
}

// I4: missing fence.sh PANICs; fence.sh passes
TEST(Shmem, i4_missing_fence_panics) {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    test_i4_missing_fence_panics();
}

// I4: cross-warp race; the mbar handoff is clean
TEST(Shmem, i4_cross_warp_race) {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    test_i4_cross_warp_race();
}

// I4: read/read overlap never PANICs
TEST(Shmem, i4_read_read_overlap_is_clean) {
    test_i4_read_read_overlap_is_clean();
}

// I4: C10 live-mbar opacity
TEST(Shmem, i4_live_mbar_opacity) {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    test_i4_live_mbar_opacity();
}

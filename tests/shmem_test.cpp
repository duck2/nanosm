// Shared memory: bank rounds and lds.uni, the encodings of shmem's own
// opcodes, port A's services and the RMW engine against SIMT accesses,
// fence.sh, and the sanitizer's shmem checks (missing fence, races, live mbars).
#include "test_support.hpp"

namespace {

using namespace nanosm_test;

// ---------------------------------------------------------------------------
// Uniformity is declared, never detected.
// - lds / sts: rounds = the most lanes on any one bank, bank = (address / 4)
//   % 16. Two lanes on one word are two lanes on one bank. Each bank serves
//   its lanes in ascending lane order, one per round.
// - lds.uni: one access at the address in a uniform register (uniform by
//   construction), the word fanned out to every lane, one RF write.
// ---------------------------------------------------------------------------

// Below table_base, the shmem word at byte address a holds table_word(a). The
// table holds one address per lane (lane l at table_base + 4l: one lane per
// bank, so loading it is one round).
constexpr uint32_t table_base = 4096;

uint32_t table_word(uint32_t address) {
    return 0xb3000000u | address;
}

nanosm::Launch table_launch(const std::vector<uint32_t>& lane_addresses = {}) {
    auto launch = launch_warps(1);
    auto& shmem = image_bytes(launch.test_shmem, table_base + 64);
    for (uint32_t address = 0; address < table_base; address += 4) write_u32(shmem, address, table_word(address));
    for (uint32_t lane = 0; lane < lane_addresses.size(); ++lane) {
        write_u32(shmem, table_base + 4 * lane, lane_addresses[lane]);
    }
    return launch;
}

// A qtx machine over table_launch's image, with a timeline (no debug trace).
std::unique_ptr<nanosm::Machine> table_machine(const std::string& program, nanosm::Launch launch,
    const nanosm::DeviceProfile& profile = canonical_profile()) {
    return std::make_unique<nanosm::Machine>(
        profile, compile(program, profile), std::move(launch), nanosm::TraceOptions{.timeline = true});
}

// Loads lane l's address from the table into r0, then runs `operation` on [r0].
std::string table_program(const std::string& operation) {
    return "sread r2, LANE_ID\nshli r2, r2, 2\nlds r0, [r2 + " + std::to_string(table_base) + "]\n" +
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

// The warp-role prologue, in qtx and hand assembly alike: every warp but warp
// 0 branches to `label` (usread reads the warp id, ur2r copies it to r29 for
// the setp; ur15, r29 and `predicate` are the prologue's).
std::string by_warp(const std::string& label, const std::string& predicate = "p6") {
    return "usread ur15, WARP_ID\nur2r r29, ur15\nsetp.ne.u32 " + predicate + ", r29, 0\n@" + predicate +
        " bra.uni " + label + "\n";
}

// Holding A-read for a while takes a second warp: warp 1 runs one
// sixteen-round sts (all lanes on one bank: words 6144 + 64 * lane), warp 0
// runs `body` (hand assembly; r20, r21 are warp 1's). hold_a_read places the
// host's event while that store is under way: the RMW starts the next cycle
// (the store started last, so the alternation gives the RMW the next start)
// and waits for its read until the cycle after the store's last round,
// starting nothing new on A-read meanwhile.
std::string with_blocker(const std::string& body) {
    return by_warp("blocker") + body + R"(
    blocker: sread r20, LANE_ID
        shli r20, r20, 6
        mov r21, 0x66
        sts [r20 + 6144], r21 {pred=PT, stall=0, set=sb0, wait=-}
        halt {pred=PT, stall=0, set=-, wait=sb0}
    )";
}

// Runs until warp 1's blocking store has its first round granted (at b), then
// places the host's event: held from b + 1, it starts an RMW there. Returns b.
uint64_t hold_a_read(nanosm::Machine& machine) {
    while (grants_of(machine, "A-write").empty()) tick_capped(machine);
    host_rmw_event(machine);
    return machine.cycle() - 1;
}

// The two fp16 2.0 halves of every lane's word at shmem 4 * lane: on the
// one-bank profile, one sts of sixteen rounds (lane w writes word w in round w).
const std::string two_halves_sts = "sread r20, LANE_ID\nshli r20, r20, 2\n"
                                   "mov r21, 16384\nshli r22, r21, 16\nor r21, r21, r22\nsts [r20], r21\n";

// An m32n64k8 matmul on dense_one_bank_profile: A (1.0) at 0, B (1.0) at 1024,
// its commit into the mbar at 896; `before` runs between the operand umovs and
// the mma.async.init, and acc.ld brings row 0 of the result into r10. Under a
// debug trace; `mbars` are written into the image as well.
std::unique_ptr<nanosm::Machine> dense_mma_machine(const nanosm::DeviceProfile& dense, const std::string& before,
    const std::vector<std::pair<uint32_t, nanosm::MbarState>>& mbars = {}) {
    auto launch = launch_warps(1);
    auto& shmem = image_bytes(launch.test_shmem, 2048);
    for (uint32_t i = 0; i < 256; ++i) write_u16(shmem, 2 * i, 0x3c00);        // A: 1.0
    for (uint32_t i = 0; i < 512; ++i) write_u16(shmem, 1024 + 2 * i, 0x3c00); // B: 1.0
    write_mbar(shmem, 896, mbar_state(1, 1));
    for (const auto& [address, state] : mbars) write_mbar(shmem, address, state);
    const std::string text = "umov ur0, 0\numov ur1, 1024\numov ur4, 896\numov ur2, 0\n" + before +
        "mma.async.init.m32n64k8 [ur2], [ur0], [ur1]\nmma.commit [ur4]\n" + wait_on("ur4") +
        "umov ur8, 0\nacc.ld r10, [ur8, 64]\nhalt\n";
    return std::make_unique<nanosm::Machine>(dense, compile(text, dense), std::move(launch),
        nanosm::TraceOptions{.timeline = true, .debug = true});
}

const std::string fence_wait = "fence.sh: own accesses not at the RAM";

} // namespace

// lds.uni: one port-A round, every lane gets the word, one full-width RF
// write; the plain lds of the same word is sixteen rounds, still one RF
// write. The write lands issue + issue_to_dispatch + rounds + 1 + D + Lb + Wd
// after the load's issue (the first round is granted the cycle after
// dispatch, the last round's words are back D + Lb after its grant, the
// gather register is complete a cycle later, then the write port), so a
// longer read latency moves it by exactly the difference.
TEST(Shmem, lds_uni_one_round_one_write) {
    auto slow = canonical_profile();
    slow.shmem_read_latency_cycles = 7;
    const std::vector<std::pair<std::string, nanosm::DeviceProfile>> rows = {
        {"lds.uni", canonical_profile()}, {"lds", canonical_profile()}, {"lds.uni", slow}};
    for (const auto& row : rows) {
        const auto& operation = row.first;
        const auto& profile = row.second;
        const bool uniform = operation == "lds.uni";
        const uint32_t rounds = uniform ? 1 : 16;
        const std::string base = uniform ? "ur0" : "r0";
        const auto what = operation + " Lb=" + std::to_string(profile.shmem_read_latency_cycles);
        // r1 differs in every lane before the load, so the load's one write
        // changes all sixteen lanes in one cycle.
        auto machine = table_machine("sread r1, LANE_ID\n" + std::string(uniform ? "umov" : "mov") + " " + base +
            ", 60\n" + operation + " r1, [" + base + " + 16]\nhalt\n", table_launch(), profile);
        const auto writes = watch_register(*machine, 1);
        require(writes.size() == 2, what + ": r1 was not written exactly twice (sread, load)");
        const auto& load = writes[1];
        require(load.lanes_changed == 16, what + ": the load did not write all sixteen lanes at once");
        for (const auto value : load.values) require(value == table_word(76), what + ": a lane missed the word");
        require(shmem_request_rounds(*machine) == std::vector<uint32_t>{rounds}, what + ": wrong round count");
        require(machine->trace().summary().shmem_rounds == rounds, what + ": port A served extra rounds");
        const auto issue = issued_at(*machine, 16);
        require(issue.size() == 1 && load.cycle == issue[0] + nanosm::issue_to_dispatch(profile) + rounds + 1 +
                profile.shmem_port_a_depth + profile.shmem_read_latency_cycles + profile.rf_write_depth_cycles,
            what + ": the RF write did not land D + Lb + 1 + Wd cycles after the last round");
    }
}

// Lanes spread over banks: rounds = the most lanes on one bank, counting
// lanes, not words (k lanes on one word are k rounds).
TEST(Shmem, rounds_are_max_lanes_per_bank) {
    std::vector<uint32_t> stride4(16), stride8(16), stride64(16), two_per_word(16), one_word(16, 0);
    // Lanes 0..3 on byte 0, lanes 4..7 on byte 64 (both bank 0); 8..15 alone.
    std::vector<uint32_t> two_words(16);
    // Bank 1: lanes 0..2 on three words; bank 2: lanes 3..7 on one word; 8..15 alone.
    std::vector<uint32_t> mixed(16);
    for (uint32_t lane = 0; lane < 16; ++lane) {
        stride4[lane] = 4 * lane;
        stride8[lane] = 8 * lane;
        stride64[lane] = 64 * lane;
        two_per_word[lane] = 8 * (lane / 2);
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
        {"stride 4 B, one lane per bank", stride4, 1, true},
        {"stride 8 B, two lanes per even bank", stride8, 2, true},
        {"stride 64 B, sixteen lanes on bank 0", stride64, 16, true},
        {"two lanes on each word, one word per even bank", two_per_word, 2, false},
        {"sixteen lanes on one word", one_word, 16, false},
        {"two bank-0 words, four lanes each", two_words, 8, false},
        {"five lanes on one bank-2 word, three on bank 1", mixed, 5, false},
    };
    for (const auto& c : cases) {
        auto load = table_machine(table_program("lds r1, [r0]"), table_launch(c.addresses));
        load->run();
        require(shmem_request_rounds(*load) == std::vector<uint32_t>{1, c.rounds}, c.what + ": lds rounds");
        for (uint32_t lane = 0; lane < 16; ++lane) {
            require(load->read_reg(0, lane, 1) == table_word(c.addresses[lane]), c.what + ": lds lane value");
        }
        if (!c.distinct_words) continue;
        auto store = table_machine(table_program("addi r1, r2, 500\nsts [r0], r1"), table_launch(c.addresses));
        store->run();
        require(shmem_request_rounds(*store) == std::vector<uint32_t>{1, c.rounds}, c.what + ": sts rounds");
        for (uint32_t lane = 0; lane < 16; ++lane) {
            require(store->read_shmem_u32(c.addresses[lane]) == 500 + 4 * lane, c.what + ": sts lane value");
        }
    }
}

// Each bank serves its lanes in ascending lane order. Lanes 0 and 1 share
// bank 0, lane 0 at byte 64 and lane 1 at byte 0; every other lane is alone
// on its bank. Round 0 writes lane 0's word and lanes 2..15; round 1 lane 1's.
TEST(Shmem, sts_bank_serves_lanes_in_order) {
    std::vector<uint32_t> addresses(16);
    for (uint32_t lane = 0; lane < 16; ++lane) addresses[lane] = 4 * lane;
    addresses[0] = 64;
    addresses[1] = 0;
    auto machine = table_machine(table_program("addi r1, r2, 1000\nsts [r0], r1"), table_launch(addresses));
    std::map<uint32_t, uint64_t> written; // byte address -> the cycle whose clock edge wrote it
    while (!machine->done()) {
        const auto cycle = machine->cycle();
        tick_capped(*machine);
        for (const auto address : addresses) {
            if (!written.contains(address) && machine->read_shmem_u32(address) != table_word(address)) {
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

// Alignment and bounds: the machine's own panics, as for lds, in both builds.
TEST(Shmem, lds_uni_alignment_and_bounds) {
    struct Case {
        std::string program;
        std::string needle;
        std::string what;
    };
    const std::vector<Case> cases = {
        {"umov ur0, 6\nlds.uni r1, [ur0]\nhalt\n", "unaligned 32-bit shmem access", "2-byte aligned lds.uni"},
        {"umov ur0, 65532\nlds.uni r1, [ur0 + 4]\nhalt\n", "shmem access out of bounds", "lds.uni past shmem"},
        {"umov ur0, 0\nlds.uni r1, [ur0 - 4]\nhalt\n", "shmem address arithmetic overflow", "negative lds.uni address"},
    };
    for (const auto& c : cases) {
        auto machine = table_machine(c.program, table_launch());
        expect_panic(*machine, {c.needle}, c.what);
    }
    auto last_word = table_machine("umov ur0, 65532\nlds.uni r1, [ur0]\nhalt\n", table_launch());
    last_word->run();
    require(last_word->read_reg(0, 9, 1) == 0, "lds.uni of the last shmem word");
}

// The ISA side of shmem's own opcodes. Each encodes to its opcode in the top
// byte, decodes and prints back: lds.uni 0x26 (RRI: src0 the base's uniform
// register), fence.sh 0x27 (no payload), atom.{add,and,xor,exch}.uni
// 0x39..0x3c (dst[8] base[8] operand[8], a signed 14-bit offset; an async RF
// writer, latency 0). A memory operation needs a depbar (qtxc gives it one
// that a consumer of its result waits on; the hand assembler requires one);
// fence.sh may not set one, takes no operands, and qtxc never inserts it.
TEST(Shmem, fence_encoding_qtxc_and_assembly) {
    const auto& profile = canonical_profile();
    using nanosm::Opcode;
    struct Row {
        std::string what;
        Opcode opcode;
        uint32_t code; // the top byte
        uint8_t dst, src0, src1;
        int32_t imm;
        uint8_t stall, set_depbar;
        bool async;       // no fixed latency
        std::string text; // how it prints ("": not checked)
    };
    const std::vector<Row> rows = {
        {"lds.uni", Opcode::lds_uni, 0x26, 5, 3, 0, 8, 0, 0, false, "lds.uni r5, [ur3 + 8]"},
        {"lds.uni, negative offset", Opcode::lds_uni, 0x26, 5, 3, 0, -2048, 0, 0, false, ""},
        {"fence.sh", Opcode::fence_sh, 0x27, 0, 0, 0, 0, 3, 7, true, "fence.sh"},
        {"atom.add.uni", Opcode::atom_add_uni, 0x39, 7, 1, 5, 64, 0, 3, true, "atom.add.uni r7, [ur1 + 64], ur5"},
        {"atom.and.uni", Opcode::atom_and_uni, 0x3a, 7, 1, 5, 64, 0, 3, true, "atom.and.uni r7, [ur1 + 64], ur5"},
        {"atom.xor.uni", Opcode::atom_xor_uni, 0x3b, 7, 1, 5, 64, 0, 3, true, "atom.xor.uni r7, [ur1 + 64], ur5"},
        {"atom.exch.uni", Opcode::atom_exch_uni, 0x3c, 7, 1, 5, 64, 0, 3, true, "atom.exch.uni r7, [ur1 + 64], ur5"},
        {"atom.add.uni, lowest offset, widest fields", Opcode::atom_add_uni, 0x39, 200, 129, 255, -8192, 0, 3, true, ""},
        {"atom.exch.uni, highest offset, widest fields", Opcode::atom_exch_uni, 0x3c, 200, 129, 255, 8191, 0, 3, true,
            ""},
    };
    for (const auto& row : rows) {
        nanosm::Instruction inst;
        inst.opcode = row.opcode;
        inst.dst = row.dst;
        inst.src0 = row.src0;
        inst.src1 = row.src1;
        inst.imm = row.imm;
        inst.control.stall = row.stall;
        inst.control.set_depbar = row.set_depbar;
        const auto word = nanosm::encode(inst);
        const auto decoded = nanosm::decode(word);
        require(word >> 56 == row.code && decoded.opcode == row.opcode && decoded.dst == row.dst &&
                decoded.src0 == row.src0 && decoded.src1 == row.src1 && decoded.imm == row.imm &&
                decoded.control.stall == row.stall && decoded.control.set_depbar == row.set_depbar,
            row.what + " did not round-trip");
        if (row.opcode == Opcode::fence_sh) {
            require((word & ((1ull << 38) - 1)) == 0, "fence.sh has a payload");
        }
        if (row.async) require(nanosm::fixed_latency(decoded, profile.timing) == 0, row.what + " has a fixed latency");
        if (!row.text.empty()) {
            require(nanosm::format_instruction(decoded, false) == row.text,
                row.what + " prints as " + nanosm::format_instruction(decoded, false));
        }
    }
    nanosm::Instruction wide;
    wide.opcode = Opcode::atom_add_uni;
    wide.imm = 8192;
    expect_error([&] { (void)nanosm::encode(wide); }, {"signed 14-bit"}, "atom.add.uni with a 15-bit offset");

    const auto lds_uni = nanosm::decode(compile("umov ur3, 0\nlds.uni r5, [ur3 + 8]\nhalt\n").words[1]);
    require(lds_uni.opcode == Opcode::lds_uni && lds_uni.dst == 5 && lds_uni.src0 == 3 && lds_uni.imm == 8 &&
            lds_uni.control.set_depbar < 6,
        "lds.uni did not survive qtxc + qtxas");
    (void)nanosm::assemble_asm(hand_asm(profile, "umov ur0, 0 {pred=PT, stall=1, set=-, wait=-}\n"
                                                 "lds.uni r1, [ur0] {pred=PT, stall=0, set=sb0, wait=-}\n"
                                                 "halt {pred=PT, stall=0, set=-, wait=sb0}\n"),
        profile);
    const auto fenced = qtx_program("mov r0, 0\nsts [r0], r0\nfence.sh\nhalt\n", profile);
    require(fenced.instructions.size() == 4 && fenced.instructions[2].opcode == Opcode::fence_sh &&
            fenced.instructions[2].control.set_depbar == 7,
        "qtxc did not schedule the written fence.sh in place");
    const auto atomic =
        qtx_program("umov ur1, 0\numov ur2, 5\natom.xor.uni r3, [ur1 - 64], ur2\nadd r4, r3, r3\nhalt\n", profile);
    const auto& atom = atomic.instructions.at(2);
    const auto& consumer = atomic.instructions.at(3);
    require(atom.opcode == Opcode::atom_xor_uni && atom.imm == -64 && atom.control.set_depbar < 6 &&
            (consumer.control.wait_mask >> atom.control.set_depbar & 1) != 0,
        "qtxc did not give the atomic a depbar its rD consumer waits on");
    for (const auto* program : {&fenced, &atomic}) {
        const auto kernel = nanosm::assemble_asm(nanosm::emit_asm(*program), profile);
        require(nanosm::load_hex(nanosm::emit_hex(kernel)).words == kernel.words, "a memory operation did not survive hex");
    }
    // No auto-insertion: an sts the TMA then stores, with no fence written, gets none.
    const auto unfenced = qtx_program(
        "mov r8, 0\n" + one_tma_operands(0) + "sts [r8], r8\ntma.store.1d [5120, ur8], [ur10], [ur11]\nhalt\n", profile);
    require(std::none_of(unfenced.instructions.begin(), unfenced.instructions.end(),
                [](const auto& inst) { return inst.opcode == Opcode::fence_sh; }) &&
            unfenced.instructions.size() == 7,
        "qtxc inserted a fence.sh");

    struct Refused {
        std::string source;
        bool hand; // the hand assembler, else qtxc
        std::string needle;
    };
    const std::vector<Refused> refused = {
        {"fence.sh r1\nhalt\n", false, "expected 0 operands"},
        {"fence.sh [r0]\nhalt\n", false, "expected 0 operands"},
        {"fence.sh {pred=PT, stall=0, set=sb0, wait=-}\nhalt\n", true, "this opcode cannot set a depbar"},
        {"lds.uni r1, [ur0] {pred=PT, stall=0, set=-, wait=-}\nhalt\n", true, "memory operation requires set=sb0..sb5"},
        {"atom.exch.uni r3, [ur1], ur2\nhalt\n", true, "memory operation requires set=sb0..sb5"},
        {"atom.add.uni r3, [r1], r2\nhalt\n", true,
            "expected uniform register (ur0..ur" + std::to_string(profile.urf_registers - 1) + "), got 'r1'"},
        {"atom.add.uni r3, [ur1 + 8192], ur2\nhalt\n", false, "atom offset must fit a signed 14-bit immediate"},
        {"atom.add.uni r3, [ur1 - 8193], ur2\nhalt\n", false, "atom offset must fit a signed 14-bit immediate"},
        {"atom.add.uni r3, [ur1]\nhalt\n", false, "expected 3 operands"},
        {"atom.min.uni r3, [ur1], ur2\nhalt\n", false, "unknown opcode 'atom.min.uni'"},
        {"atom.add r3, [r1], r2\nhalt\n", false, "unknown opcode 'atom.add'"},
        {"atom.add.uni r3, [r1], r2\nhalt\n", false, "%r1 is used here as a uniform register"},
    };
    for (const auto& row : refused) {
        expect_error(
            [&] {
                if (row.hand) (void)hand_assemble(hand_asm(profile, row.source), profile);
                else (void)qtx_program(row.source, profile);
            },
            {row.needle}, std::string(row.hand ? "hand assembly '" : "qtxc '") + row.source + "'");
    }
}

// Same-warp order with the warp's reads serviced out of order. A host RMW
// waiting behind warp 1's sixteen-round store holds A-read (with_blocker)
// while two of warp 0's lds become ready; the round-robin pointer, moved past
// slot 0 by the lds.uni P, takes the YOUNGER (slot 1) first. WAR: the sts over
// OLD's words is ordered behind OLD until OLD's grant, then held by the store
// pause to the RMW's write grant w: granted at w + 1, it reaches the RAM long
// after OLD's sample; the sts over YOUNG's words, ordered behind it, issues at
// that grant and is granted t + 1 later. Relative to the blocker's first round
// b: the RMW reads at r = b + 16, YOUNG at r + 1, OLD at r + 2. (At r + 2 warp
// 1's halt takes the issue slot, so the WAR sts issues at r + 3.)
TEST(Shmem, same_warp_order_with_reordered_reads) {
    const auto& profile = canonical_profile();
    require(read_return(profile) == 4 && profile.shmem_load_slots == 2,
        "the scenario is built for D = Lb = 2 and two load slots");
    const uint64_t t = nanosm::issue_to_dispatch(profile);
    const uint64_t c = profile.shmem_rmw_compute_cycles;
    // P's base is the uniform ur1 (never written: 0).
    auto machine = debug_machine(profile, with_blocker(R"(
        sread r0, LANE_ID
        shli r0, r0, 2
        lds.uni r3, [ur1 + 256] {pred=PT, stall=0, set=sb4, wait=-}
        lds r7, [r0 + 128] {pred=PT, stall=0, set=sb0, wait=sb4}
        lds r10, [r0 + 192] {pred=PT, stall=0, set=sb1, wait=-}
        mov r5, 0x5150
        sts [r0 + 128], r5 {pred=PT, stall=0, set=sb2, wait=-}
        sts [r0 + 192], r5 {pred=PT, stall=0, set=sb3, wait=-}
        halt {pred=PT, stall=0, set=-, wait=sb0|sb1|sb2|sb3}
    )"), 2, event_mbars());
    const auto blocker = hold_a_read(*machine);
    machine->run();
    const uint64_t r = blocker + 16;                // the RMW's read: the cycle after the blocker's last round
    const uint64_t w = r + read_return(profile) + c; // its write
    const auto reads = grants_of(*machine, "A-read");
    constexpr uint32_t p_pc = 6 * 8, blocker_pc = 16 * 8, r5_pc = 9 * 8;
    const auto p = issued_at(*machine, p_pc).at(0);
    require(issued_at(*machine, blocker_pc) == std::vector<uint64_t>{blocker - t - 1} &&
            reads == std::vector<Grant>{{p + t + 1, "A-read", "simt", 0}, {r, "A-read", "rmw", 0},
                         {r + 1, "A-read", "simt", 1}, {r + 2, "A-read", "simt", 0}},
        "A-read grants (blocker from c" + std::to_string(blocker) + "):" + describe(reads) + " | all" +
            describe(grants_of(*machine)) + " | issues" + issue_log(*machine) + " | ordered" +
            describe(refusals(*machine, 0, "ordered behind own access")) + " slots" +
            describe(refusals(*machine, 0, "load slots full")));
    // OLD waits on sb4: it issues as P's result becomes visible (P's RF write
    // at its issue + lds_shift + 2, visible Wd + 1 later), YOUNG the cycle after.
    constexpr uint32_t old_pc = 7 * 8, young_pc = 8 * 8, war_pc = 10 * 8;
    const auto old_issue = p + lds_shift(profile) + 2 + profile.rf_write_depth_cycles + 1;
    require(issued_at(*machine, old_pc) == std::vector<uint64_t>{old_issue} &&
            issued_at(*machine, young_pc) == std::vector<uint64_t>{old_issue + 1},
        "OLD did not issue as P's result became visible (sb4), or YOUNG the cycle after");
    const auto ordered = refusals(*machine, 0, "ordered behind own access");
    const auto war_first_try = issued_at(*machine, r5_pc).at(0) + alu_distance(profile);
    require(issued_at(*machine, war_pc) == std::vector<uint64_t>{r + 3} && ordered.size() == 2 &&
            ordered.front() == std::pair<uint64_t, uint64_t>{war_first_try, r + 2 - war_first_try},
        "the sts over OLD's words was not ordered behind OLD until OLD's grant:" + describe(ordered) +
            "; grants:" + describe(grants_of(*machine)) + " | issues" + issue_log(*machine));
    std::vector<Grant> writes;
    for (uint64_t round = 0; round < 16; ++round) writes.push_back({blocker + round, "A-write", "simt", 0});
    writes.push_back({w, "A-write", "rmw", 0});
    writes.push_back({w + 1, "A-write", "simt", 0});
    writes.push_back({w + t + 2, "A-write", "simt", 0});
    require(grants_of(*machine, "A-write") == writes, "A-write grants:" + describe(grants_of(*machine, "A-write")));
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(machine->read_reg(0, lane, 7) == 0x100 + 32 + lane, "OLD did not read the words before the sts");
        require(machine->read_reg(0, lane, 10) == 0x100 + 48 + lane, "YOUNG did not read the words before the sts");
        require(machine->read_shmem_u32(128 + 4 * lane) == 0x5150 && machine->read_shmem_u32(192 + 4 * lane) == 0x5150,
            "sts data");
    }
}

// Rounds come from active lanes only; a masked lane is no access (its word,
// and an lds's destination lane, are untouched). The mask is a real divergent
// region (masked_region); every lane is on bank 0, so the four active lanes
// are four rounds.
TEST(Shmem, divergence) {
    auto machine = debug_machine(canonical_profile(), R"(
        sread r0, LANE_ID
        shli r0, r0, 6
        mov r3, 0x77
        mov r5, 0x55
    )" + masked_region(0x00f0, R"(
        sts [r0], r5 {pred=PT, stall=0, set=sb0, wait=-}
        lds r3, [r0 + 4] {pred=PT, stall=0, set=sb1, wait=-}
    )") + R"(
        halt {pred=PT, stall=0, set=-, wait=sb0|sb1}
    )");
    machine->run();
    require(shmem_request_rounds(*machine) == std::vector<uint32_t>{4, 4}, "four active lanes are four rounds");
    for (uint32_t lane = 0; lane < 16; ++lane) {
        const bool active = lane >= 4 && lane < 8;
        require(machine->read_shmem_u32(64 * lane) == (active ? 0x55 : 0x100 + 16 * lane), "sts lane");
        require(machine->read_reg(0, lane, 3) == (active ? 0x100 + 16 * lane + 1 : 0x77), "lds lane");
    }
}

// One store slot, reused while the previous store is still in the A-write
// pipeline, and done() when the last instruction is a store. sts X reaches
// the slot at its dispatch and is granted the next cycle, gx, which frees the
// slot; sts Y issues then and is granted gy = gx + issue_to_dispatch + 1. X
// reaches the RAM at gx + D, Y at gy + D; the halt issues at gx + 1, and the
// machine is done only after Y's write edge.
TEST(Shmem, store_slot_reuse_and_done) {
    auto profile = canonical_profile();
    profile.shmem_store_slots = 1;
    const auto d = profile.shmem_port_a_depth;
    const auto dispatch = nanosm::issue_to_dispatch(profile);
    auto machine = debug_machine(profile, R"(
        sread r0, LANE_ID
        shli r0, r0, 2
        mov r5, 0xa1
        sts [r0 + 512], r5 {pred=PT, stall=0, set=sb0, wait=-}
        sts [r0 + 576], r5 {pred=PT, stall=0, set=sb1, wait=-}
        halt
    )");
    struct After {
        bool x_landed = false;
        bool y_landed = false;
        bool done = false;
    };
    std::map<uint64_t, After> after; // the state after each cycle's clock edge
    while (!machine->done()) {
        const auto cycle = machine->cycle();
        tick_capped(*machine);
        after[cycle] = {machine->read_shmem_u32(512) == 0xa1, machine->read_shmem_u32(576) == 0xa1, machine->done()};
    }
    const auto x = issued_at(*machine, 3 * 8).at(0);
    const auto gx = x + dispatch + 1;
    const auto gy = gx + dispatch + 1;
    require(grants_of(*machine, "A-write") == std::vector<Grant>{{gx, "A-write", "simt", 0}, {gy, "A-write", "simt", 0}},
        "the two stores were not granted at gx and gy:" + describe(grants_of(*machine)) + " | issues" +
            issue_log(*machine));
    require(issued_at(*machine, 4 * 8) == std::vector<uint64_t>{gx}, "Y did not issue as X's last round freed the slot");
    require(!after.at(gx + d - 1).x_landed && after.at(gx + d).x_landed, "X did not reach the RAM at gx + D");
    require(!after.at(gy + d - 1).y_landed && after.at(gy + d).y_landed, "Y did not reach the RAM at gy + D");
    require(issued_at(*machine, 5 * 8) == std::vector<uint64_t>{gx + 1}, "halt");
    require(!after.at(gy + d - 1).done && after.at(gy + d).done,
        "done() did not cover the store travelling to its write edge");
    for (uint32_t lane = 0; lane < 16; ++lane) require(machine->read_shmem_u32(576 + 4 * lane) == 0xa1, "Y's data");
}

// A-read and A-write are independent services: one warp's sixteen-round lds
// and another warp's sixteen-round sts are granted in the same cycles. The
// warps' bra.uni issue a cycle apart; the reader's lds follows its not-taken
// branch by branch_not_taken_interval, the writer's sts its mov at the target
// by alu_distance, the target the taken branch by branch_taken_interval
// (src/timing.hpp). So the sts issues 1 + taken - not taken + alu_distance
// after the lds, each is granted t + 1 after its issue, and both services
// grant in every one of the cycles the two sixteen-cycle windows share.
TEST(Shmem, a_read_and_a_write_same_cycle) {
    const auto& profile = canonical_profile();
    auto machine = debug_machine(profile, compile(R"(
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
    )", profile), 2);
    machine->run();
    std::set<uint64_t> read_cycles;
    for (const auto& grant : grants_of(*machine, "A-read")) read_cycles.insert(grant.cycle);
    uint32_t together = 0;
    for (const auto& grant : grants_of(*machine, "A-write")) together += read_cycles.contains(grant.cycle) ? 1 : 0;
    const auto apart = 1 + nanosm::branch_taken_interval - nanosm::branch_not_taken_interval + alu_distance(profile);
    require(issued_at(*machine, 9 * 8).at(0) == issued_at(*machine, 6 * 8).at(0) + apart,
        "the sts did not issue 1 + taken - not taken + alu_distance after the lds:" + issue_log(*machine));
    require(read_cycles.size() == 16 && together == 16 - apart,
        "A-read and A-write were not granted in the same cycles (" + std::to_string(together) + ")");
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(machine->read_reg(0, lane, 3) == 0x100 + 16 * lane, "the load's words");
        require(machine->read_shmem_u32(64 * lane + 32) == 0x44, "the store's words");
    }
}

// ONE round-robin pointer over the load slots, the probe register and the
// TMA's reads, moved only by its own grants (not by the RMW's). A host RMW
// waiting behind warp 1's sixteen-round store holds A-read (with_blocker)
// while warp 0's tma.load and two lds.uni become ready; the last operand
// umov's stall places them inside the hold (the run checks that they are).
// Relative to the RMW's read r:
//   pointer at 0 (nothing granted yet): slot 0, slot 1, then the TMA;
//   pointer past slot 1 (two earlier lds.uni, granted before the hold): the
//   TMA, then slot 0, slot 1.
// A request is ready at its dispatch + 1 (the tma's command at issue +
// issue_to_tma_command + 1).
TEST(Shmem, round_robin_tma_reads_and_loads) {
    const auto& profile = canonical_profile();
    require(read_return(profile) == 4, "the stalls are tuned for D = Lb = 2");
    const auto run = [&](bool primed) {
        const std::string priming = primed ? "lds.uni r5, [ur1 + 8] {pred=PT, stall=0, set=sb3, wait=-}\n"
                                             "lds.uni r6, [ur1 + 12] {pred=PT, stall=0, set=sb4, wait=-}\n"
                                           : "";
        const std::string stall = primed ? "14" : "16";
        // The TMA's operands (buffer at 1024 in ur10, completing into 4104 in
        // ur11, coordinate 0 in ur8); the last umov stalls.
        auto machine = gmem_debug_machine(profile, with_blocker(priming +
            "umov ur10, 1024\numov ur11, 4104\numov ur8, 0 {pred=PT, stall=" + stall + ", set=-, wait=-}\n" + R"(
            tma.load.1d [ur10], [5120, ur8], [ur11]
            lds.uni r3, [ur1] {pred=PT, stall=0, set=sb0, wait=-}
            lds.uni r4, [ur1 + 4] {pred=PT, stall=0, set=sb1, wait=-}
            halt {pred=PT, stall=0, set=-, wait=sb0|sb1}
        )"), 2, [](std::vector<uint8_t>& shmem) {
            write_mbar(shmem, event_mbar, mbar_state(1, 0, 64));
            write_rank1_tma(shmem, 5120, 0, 16, 4104);
        });
        const auto blocker = hold_a_read(*machine);
        machine->run();
        const uint64_t r = blocker + 16; // the RMW's read
        // warp 0's tma.load and two lds.uni (the last three before halt)
        // became ready while A-read was held
        const uint32_t tma_pc = (primed ? 9 : 7) * 8;
        const auto ready = [&](uint32_t pc) {
            const auto reach = pc == tma_pc ? nanosm::issue_to_tma_command(1, profile) : nanosm::issue_to_dispatch(profile);
            return issued_at(*machine, pc).front() + reach + 1;
        };
        for (const uint32_t pc : {tma_pc, tma_pc + 8, tma_pc + 16}) {
            require(issued_at(*machine, pc).size() == 1 && ready(pc) < r && ready(pc) > blocker,
                "a request did not become ready inside the hold (blocker c" + std::to_string(blocker) + "):" +
                    describe(grants_of(*machine)) + " | issues" + issue_log(*machine));
        }
        auto reads = grants_of(*machine, "A-read");
        reads.resize(std::min<std::size_t>(reads.size(), primed ? 6 : 4));
        return std::pair{reads, r};
    };
    const auto [plain, r0] = run(false);
    require(plain == std::vector<Grant>{{r0, "A-read", "rmw", 0}, {r0 + 1, "A-read", "simt", 0},
                         {r0 + 2, "A-read", "simt", 1}, {r0 + 3, "A-read", "tma", 0}},
        "pointer at 0: not slot 0, slot 1, TMA:" + describe(plain));
    const auto [primed, r1] = run(true);
    require(primed.size() == 6 && primed[0].slot == 0 && primed[1].slot == 1 && primed[1].cycle < r1 - 16 &&
            std::vector<Grant>(primed.begin() + 2, primed.end()) ==
                std::vector<Grant>{{r1, "A-read", "rmw", 0}, {r1 + 1, "A-read", "tma", 0},
                    {r1 + 2, "A-read", "simt", 0}, {r1 + 3, "A-read", "simt", 1}},
        "pointer past slot 1: not TMA, slot 0, slot 1:" + describe(primed));
}

// READ_FIRST between the MMA's port-B read and an A-write. Port B has port
// A's depth, so a store round and a fetch round reach the RAM in grant order;
// a fetch round samples the RAM at its grant + D and sees a word iff that
// word's sts round reached the RAM in an EARLIER cycle (the same cycle is
// READ_FIRST: the old word). A host RMW's store pause holds the sixteen-round
// sts (one bank) until w + 1 (w = D + Lb + C; Lb 41 puts it after the first
// pads' fetch) while the pads move the fetch across it, a tie included; row 0
// of the result is the exact sum of the A words the fetch sampled. The
// mma.async.init has no fence.sh: with the sanitizer, the missing-fence check
// PANICs exactly when a store round has not reached the RAM at the MMA's
// accept, and the sweep must see both outcomes.
TEST(ShmemSlow, read_first_across_services) {
    auto dense = dense_one_bank_profile(5);
    dense.shmem_read_latency_cycles = 41;
    nanosm::validate_device_profile(dense);
    const uint64_t depth = dense.shmem_port_a_depth;
    // The host's event mbar: its RMW pauses the sts. (The commit mbar is in
    // the image: an mbar.init would be an RMW behind the host's, ordering the
    // sts after its long write.)
    constexpr uint32_t pause_mbar = 960;
#if NANOSM_SANITIZER
    uint32_t panicked = 0;
    uint32_t clean = 0;
#else
    const uint64_t paused_until = depth + dense.shmem_read_latency_cycles + dense.shmem_rmw_compute_cycles;
    std::set<uint32_t> outcomes;
    bool tie = false;
#endif
    // Up to 40 pads, so the sanitizer build sees the paused sts reach the RAM
    // in full before an accept.
    for (uint32_t pad = 0; pad <= 40; ++pad) {
        std::string pads;
        for (uint32_t i = 0; i < pad; ++i) pads += "mov r22, 1\n";
        auto mma = dense_mma_machine(dense, two_halves_sts + pads, {{pause_mbar, mbar_state(1, 0, 64)}});
        mma->complete_mbar_tx(pause_mbar, 64, 0);
        std::optional<std::string> panic;
        try {
            mma->run(10000);
        } catch (const nanosm::SimError& error) {
            panic = error.what();
        }
        std::vector<Grant> stores;
        for (const auto& grant : grants_of(*mma, "A-write")) {
            if (grant.client == "simt") stores.push_back(grant);
        }
        const auto what = "pad " + std::to_string(pad) + ":" + describe(stores);
#if NANOSM_SANITIZER
        // The MMA's accept: its dispatch (the mma.async.init is instruction
        // 10 + pad), or the PANIC it raised there.
        const uint64_t accept = panic.has_value()
            ? std::stoull(panic->substr(panic->find("cycle=") + 6))
            : issued_at(*mma, 8 * (10 + pad)).at(0) + nanosm::issue_to_dispatch(dense);
        const bool in_flight = stores.size() < 16 || stores.back().cycle + depth > accept;
        require(panic.has_value() == in_flight, what + ": accepted at c" + std::to_string(accept) +
            (panic.has_value() ? " with every store round at the RAM, yet: " + *panic
                               : " with a store round still travelling, and no missing-fence PANIC"));
        if (panic.has_value()) {
            require_mentions(*panic, {"missing fence.sh: mma.async #1 A operand bytes"}, what);
            ++panicked;
        } else {
            require(nanosm::bits_float(mma->read_reg(0, 0, 10)) == 16.0f, what + ": not every word was new");
            ++clean;
        }
#else
        require(!panic.has_value(), what + ": " + panic.value_or(""));
        std::map<int64_t, uint64_t> step_start;
        for (const auto& event : events_named(*mma, "MMA input step")) step_start[event.value] = event.start_cycle;
        // Row 0's four A words: words 0, 1 are rounds 0, 1 of step 0 (k block
        // 0), words 2, 3 rounds 0, 1 of step 128 (k block 1: 8 m x 16 n blocks
        // later).
        const std::array<uint64_t, 4> sampled{step_start.at(0) + depth, step_start.at(0) + 1 + depth,
            step_start.at(128) + depth, step_start.at(128) + 1 + depth};
        require(stores.size() == 16 && stores[0].cycle == paused_until + 1,
            what + ": the sts was not sixteen rounds granted from w + 1");
        float expected = 0.0f;
        for (uint32_t word = 0; word < 4; ++word) {
            const auto landed = stores[word].cycle + depth; // its bytes land at this cycle's clock edge
            tie |= landed == sampled[word];
            expected += 2.0f * (landed < sampled[word] ? 2.0f : 1.0f); // two fp16 halves per word, B = 1.0
        }
        const auto row0 = nanosm::bits_float(mma->read_reg(0, 0, 10));
        require(row0 == expected, what + ": row 0 = " + std::to_string(row0) +
            ", but the words the fetch sampled sum to " + std::to_string(expected));
        outcomes.insert(mma->read_reg(0, 0, 10));
#endif
    }
#if NANOSM_SANITIZER
    require(panicked != 0 && clean != 0, "the sweep did not cross the missing-fence boundary (" +
        std::to_string(panicked) + " PANICs, " + std::to_string(clean) + " clean)");
#else
    require(outcomes.size() >= 2 && tie, "the sweep did not cross the READ_FIRST boundary (a tie included)");
#endif
}

// The RMW engine's update crosses shmem.rmw_compute_cycles (C) ALU stages. A
// SIMT mbar.arrive reaches the SIMT RMW register at its dispatch and has its
// read granted the cycle after, r; its words reach the port output at r + D
// + Lb, its write is granted C cycles later, and the write reaches the RAM
// (the phase flips at that clock edge) D cycles after that. The engine is
// busy (its trace slice) D + Lb + C + 1 cycles.
TEST(Shmem, rmw_compute_stage) {
    for (const uint32_t c : {1u, 2u, 3u}) {
        auto profile = canonical_profile();
        profile.shmem_rmw_compute_cycles = c;
        const uint64_t d = profile.shmem_port_a_depth;
        const uint64_t lb = profile.shmem_read_latency_cycles;
        auto machine = debug_machine(profile, R"(
            umov ur4, 4160
            mbar.arrive [ur4]
            halt
        )", 1, {{4160, mbar_state(1, 1)}});
        std::optional<uint64_t> flipped; // the cycle whose clock edge landed the new phase
        while (!machine->done()) {
            const auto cycle = machine->cycle();
            tick_capped(*machine);
            if (!flipped.has_value() && machine->read_mbar_state(4160).phase) flipped = cycle;
        }
        const auto what = "C = " + std::to_string(c) + ": ";
        const auto grants = grants_of(*machine);
        const auto arrive = issued_at(*machine, 8);
        require(arrive.size() == 1, what + "the arrive did not issue once");
        const auto r = arrive[0] + nanosm::issue_to_dispatch(profile) + 1;
        require(grants == std::vector<Grant>{{r, "A-read", "rmw", 0}, {r + d + lb + c, "A-write", "rmw", 0}},
            what + "the read was not granted the cycle after the arrive's dispatch, or the write D + Lb + C after "
            "it:" + describe(grants));
        require(flipped == r + d + lb + c + d, what + "the update did not reach the RAM D cycles after its write grant");
        const auto slices = events_named(*machine, "mbar RMW");
        require(slices.size() == 1 && slices[0].duration_cycles == d + lb + c + 1,
            what + "not one RMW busy D + Lb + C + 1 cycles");
        const auto state = machine->read_mbar_state(4160);
        require(state.phase && state.remaining == 1, what + "the arrive's update was lost");
    }
}

// A probe waits for its own warp's sts, and sees it. The sts's sixteen lanes
// are on one bank (sixteen rounds); lane 15 -- the LAST round -- writes the
// mbar's low word with its phase flipped. mbar.test_wait is a read: ordered
// behind the sts, admitted at its last grant, its probe granted after its
// dispatch and sampling D cycles later, after the last round reached the
// RAM: the predicate says "phase complete". Relative to the sts's issue s and
// t = issue_to_dispatch: rounds s + t + 1 .. s + t + 16; the test_wait
// refused from s + 1 (t + 15 cycles), admitted at s + t + 16, its probe
// granted at s + 2t + 17.
TEST(Shmem, probe_behind_own_sts) {
    const auto new_low = static_cast<uint32_t>(nanosm::encode_mbar_state(mbar_state(1, 1, 0, true)));
    for (const auto& [d, lb] : std::vector<std::pair<uint32_t, uint32_t>>{{2, 2}, {3, 1}}) {
        auto profile = canonical_profile();
        profile.shmem_port_a_depth = d;
        profile.shmem_read_latency_cycles = lb;
        const auto what = "D=" + std::to_string(d) + " Lb=" + std::to_string(lb) + ": ";
        auto machine = debug_machine(profile, R"(
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
        )", 1, {{6144, mbar_state(1, 1)}});
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
        const auto reads = grants_of(*machine, "A-read");
        require(reads == std::vector<Grant>{{s + 2 * t + 17, "A-read", "probe", 0}}, what + "probe grant:" + describe(reads));
        for (uint32_t lane = 0; lane < 16; ++lane) {
            require(machine->read_reg(0, lane, 9) == 1, what + "the probe did not see its own sts's last round");
        }
    }
}

// A wake that arrives while a try_wait's probe is in flight ends the wait;
// the probe, when it returns, is stale and does nothing. Warp 0's try_wait
// installs its wait slot at its dispatch and its probe is granted the next
// cycle, w. The host's event is placed D + Lb + C earlier, so the RMW's write
// is granted at w too: the probe samples at w + D, the cycle the write
// reaches the RAM (READ_FIRST: the old phase), and the write's commit wakes
// slot 0 in that cycle; the probe returns at w + D + Lb to a slot that is gone
// -- or, with a second warp, to the same slot index recycled by warp 1's
// try_wait on another mbar. The stale probe must not complete warp 1's wait:
// warp 1 times out (r10 = 0), warp 0 woke (r9 = 1), only warp 1's probe acts.
TEST(ShmemSlow, stale_probe_after_wake) {
    const auto& profile = canonical_profile();
    const uint64_t d = profile.shmem_port_a_depth;
    const uint64_t lb = profile.shmem_read_latency_cycles;
    const uint64_t c = profile.shmem_rmw_compute_cycles;
    const uint64_t t = nanosm::issue_to_dispatch(profile);
    const auto program = [](uint32_t warp1_stall) {
        return by_warp("w1", "p1") + R"(
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
        {event_mbar, mbar_state(1, 0, 64)}, {4176, mbar_state(1, 1)}};
    constexpr uint32_t warp0_try_wait = 6 * 8;
    constexpr uint32_t warp1_try_wait = 12 * 8;
    // When warp 0's try_wait issues (independent of the host's event).
    const auto try_wait_issue = [&](uint32_t warps, uint32_t warp1_stall) {
        auto dry = debug_machine(profile, program(warp1_stall), warps, mbars);
        dry->run();
        return issued_at(*dry, warp0_try_wait).front();
    };
    struct Seen {
        uint64_t warp0_issue = 0;
        uint64_t warp1_issue = 0;
        std::vector<Grant> reads;
        std::vector<Grant> writes;
        nanosm::TraceSummary summary;
        uint32_t r9 = 0;
        uint32_t r10 = 0;
    };
    const auto run = [&](uint32_t warps, uint32_t warp1_stall) {
        const auto t0 = try_wait_issue(warps, warp1_stall);
        require(t0 + t + 1 >= d + lb + c, "warp 0's try_wait issues too early to place the event before it");
        auto machine = debug_machine(profile, program(warp1_stall), warps, mbars);
        // The probe is granted at t0 + t + 1; the RMW, read in its start
        // cycle, has its write granted D + Lb + C later.
        run_until(*machine, t0 + t + 1 - (d + lb + c));
        host_rmw_event(*machine);
        machine->run();
        Seen seen;
        seen.warp0_issue = issued_at(*machine, warp0_try_wait).front();
        if (warps == 2) {
            const auto issues = issued_at(*machine, warp1_try_wait);
            require(issues.size() == 1, "warp 1's try_wait did not issue once");
            seen.warp1_issue = issues.front();
        }
        seen.reads = grants_of(*machine, "A-read");
        seen.writes = grants_of(*machine, "A-write");
        seen.summary = machine->trace().summary();
        seen.r9 = machine->read_reg(0, 0, 9);
        if (warps == 2) seen.r10 = machine->read_reg(1, 0, 10);
        return seen;
    };
    // One warp: the slot is gone when the probe returns.
    const auto alone = run(1, 0);
    const auto w = alone.warp0_issue + t + 1; // the probe's grant and the RMW's write grant
    require(alone.reads.size() >= 2 && alone.reads[0] == Grant{w - (d + lb + c), "A-read", "rmw", 0} &&
            alone.reads[1] == Grant{w, "A-read", "probe", 0} &&
            alone.writes == std::vector<Grant>{{w, "A-write", "rmw", 0}},
        "the probe was not granted in the RMW's write-grant cycle:" + describe(alone.reads) + describe(alone.writes));
    require(alone.summary.mbar_wakes == 1 && alone.summary.mbar_initial_probes == 0 && alone.summary.mbar_sleeps == 0 &&
            alone.summary.mbar_timeouts == 0, "the stale probe was acted on (or the wake was lost)");
    require(alone.r9 == 1, "the woken try_wait did not return true");
    // Two warps (warp 1's stall count swept): warp 1's try_wait takes slot 0
    // between the wake (w + D) and the stale probe's return (w + D + Lb).
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

// An RMW waiting for its read does not preempt an A-read transaction under
// way. A sixteen-round lds is granted from g (after sread -> shli, 2
// alu_distance, then issue_to_dispatch + 1); the host's event is held from g
// + 2: the RMW STARTS there (its store pause begins; there is no store) and
// waits; its read is granted at g + 16, after the lds's last round, and its
// write D + Lb + C later. Also at a noncanonical depth.
TEST(Shmem, rmw_read_behind_a_read_transaction) {
    for (const auto& [d, lb, c] : std::vector<std::tuple<uint32_t, uint32_t, uint32_t>>{{2, 2, 2}, {3, 1, 1}}) {
        auto profile = canonical_profile();
        profile.shmem_port_a_depth = d;
        profile.shmem_read_latency_cycles = lb;
        profile.shmem_rmw_compute_cycles = c;
        const auto what = "D=" + std::to_string(d) + " Lb=" + std::to_string(lb) + " C=" + std::to_string(c) + ": ";
        auto machine = debug_machine(profile, R"(
            sread r0, LANE_ID
            shli r0, r0, 6
            lds r6, [r0 + 12] {pred=PT, stall=0, set=sb0, wait=-}
            halt {pred=PT, stall=0, set=-, wait=sb0}
        )", 1, event_mbars());
        const auto g = nanosm::fetch_to_issue + 2 * alu_distance(profile) + nanosm::issue_to_dispatch(profile) + 1;
        run_until(*machine, g + 2);
        host_rmw_event(*machine);
        machine->run();
        std::vector<Grant> expected;
        for (uint64_t round = 0; round < 16; ++round) expected.push_back({g + round, "A-read", "simt", 0});
        expected.push_back({g + 16, "A-read", "rmw", 0});
        expected.push_back({g + 16 + d + lb + c, "A-write", "rmw", 0});
        const auto grants = grants_of(*machine);
        require(grants == expected, what + "the RMW's read preempted the lds or waited too long:" + describe(grants));
        require(mbar_rmw_services(*machine).front().first == g + 2, what + "the RMW did not start at g + 2");
        for (uint32_t lane = 0; lane < 16; ++lane) {
            require(machine->read_reg(0, lane, 6) == 0x100 + 16 * lane + 3, what + "lds data");
        }
    }
}

// ---------------------------------------------------------------------------
// fence.sh, the missing-fence / race sanitizer and live mbars. Timelines are
// hand assembly under a debug trace, as in the port-A tests; the sanitizer
// cases are qtx where the exact cycles do not matter.
// ---------------------------------------------------------------------------

// fence.sh is admitted in the cycle its warp's LAST earlier shmem access
// reaches the RAM -- a store round's write edge, a load round's sample: its
// grant + D -- so the instruction after it issues once that edge has passed.
// At D = 2 and 3: a one-round sts; a two-round sts (lanes l and l + 8 share a
// bank: the last round decides); an lds (admitted at its sample, while its
// words are still on their way to the RF); a TMA store of the sts's words
// right after the fence (accepted with the store at the RAM, so the
// missing-fence check passes and gmem gets the new words). An access issued
// at s (its sources' fixed distances, alu_distance each) is granted from g =
// s + issue_to_dispatch + 1; the fence is refused from s + 1.
TEST(Shmem, fence_waits_until_own_accesses_reach_ram) {
    for (const uint32_t d : {2u, 3u}) {
        auto profile = canonical_profile();
        profile.shmem_port_a_depth = d;
        const auto what = "D=" + std::to_string(d) + ": ";
        using Slices = std::vector<std::pair<uint64_t, uint64_t>>;
        const uint64_t t = nanosm::issue_to_dispatch(profile);
        const uint64_t distance = alu_distance(profile);
        const uint64_t first = nanosm::fetch_to_issue; // the first instruction's issue

        auto one = gmem_debug_machine(profile, R"(
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
                grants_of(*one) == std::vector<Grant>{{g1, "A-write", "simt", 0}},
            what + "one-round sts:" + describe(grants_of(*one)) + " | issues" + issue_log(*one));
        require(issued_at(*one, 4 * 8) == std::vector<uint64_t>{g1 + d} &&
                issued_at(*one, 5 * 8) == std::vector<uint64_t>{g1 + 1 + d} &&
                refusals(*one, 0, fence_wait) == Slices{{s1 + 1, t + d}},
            what + "one-round sts: fence.sh not admitted exactly at its grant + D (c" + std::to_string(g1 + d) +
                "); waits:" + describe(refusals(*one, 0, fence_wait)));

        auto two = gmem_debug_machine(profile, R"(
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
                grants_of(*two) == std::vector<Grant>{{g2, "A-write", "simt", 0}, {g2 + 1, "A-write", "simt", 0}},
            what + "two-round sts:" + describe(grants_of(*two)) + " | issues" + issue_log(*two));
        require(issued_at(*two, 8 * 8) == std::vector<uint64_t>{g2 + 1 + d} &&
                refusals(*two, 0, fence_wait) == Slices{{s2 + 1, t + 1 + d}},
            what + "two-round sts: fence.sh not admitted at its LAST round's grant + D (c" +
                std::to_string(g2 + 1 + d) + "); waits:" + describe(refusals(*two, 0, fence_wait)));

        auto load = gmem_debug_machine(profile, R"(
            sread r0, LANE_ID
            shli r0, r0, 2
            lds r6, [r0 + 256] {pred=PT, stall=0, set=sb0, wait=-}
            fence.sh
            mov r9, 1
            halt {pred=PT, stall=0, set=-, wait=sb0}
        )");
        const auto written = watch_registers(*load, {6});
        const uint64_t l = first + 2 * distance; // sread, shli; the lds needs r0
        const uint64_t gl = l + t + 1;
        require(issued_at(*load, 2 * 8) == std::vector<uint64_t>{l} &&
                grants_of(*load) == std::vector<Grant>{{gl, "A-read", "simt", 0}},
            what + "lds:" + describe(grants_of(*load)) + " | issues" + issue_log(*load));
        require(issued_at(*load, 3 * 8) == std::vector<uint64_t>{gl + d} &&
                refusals(*load, 0, fence_wait) == Slices{{l + 1, t + d}},
            what + "lds: fence.sh not admitted at its sample (grant + D = c" + std::to_string(gl + d) + ")");
        require(written.at(6).size() == 1 && written.at(6).front() > gl + d && load->read_reg(0, 5, 6) == 0x100 + 64 + 5,
            what + "lds: the fence waited for the RF write, or the lds read the wrong word");

        // The TMA's operand registers are written first: their fixed
        // distances are long met when the TMA follows the fence.
        auto engine = gmem_debug_machine(profile, one_tma_operands(0) + R"(
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
        )", 1, one_tma_image(1024));
        engine->run(); // the sanitizer build: no missing-fence PANIC
        const uint64_t ge = first + 4 + 2 * distance + 1 + t + 1; // four umovs, then as in `one`
        const auto engine_stores = grants_of(*engine, "A-write");
        require(!engine_stores.empty() && engine_stores.front().cycle == ge &&
                issued_at(*engine, 8 * 8) == std::vector<uint64_t>{ge + d} &&
                issued_at(*engine, 9 * 8) == std::vector<uint64_t>{ge + 1 + d},
            what + "the TMA store did not follow the fence at the sts's grant + D + 1:" + describe(grants_of(*engine)) +
                " | issues" + issue_log(*engine));
        for (uint32_t i = 0; i < 256; ++i) {
            require(engine->read_gmem_u32(4 * i) == (i < 16 ? 0x777u : 0x100 + i), what + "TMA store word " + std::to_string(i));
        }
    }
}

// SANITIZER: a warp's own access that has NOT reached the RAM when an engine
// command over the same bytes is accepted -- one write involved -- PANICs
// "missing fence.sh"; the same program with fence.sh between them runs and
// computes the fenced result:
//   sts -> TMA store of those words (its source reads);
//   sts -> tma.load whose DESCRIPTOR the sts just wrote (copied in through
//     lds / sts);
//   sts -> mma.async reading that tile (port B), one bank: sixteen rounds;
//   lds -> TMA load overwriting the words the lds reads (it has not sampled).
TEST(Shmem, missing_fence_panics) {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    const auto& profile = canonical_profile();
    const auto machine = [&](const std::string& before, bool fenced, const std::string& command,
                             const std::function<void(std::vector<uint8_t>&)>& setup) {
        return gmem_debug_machine(profile, compile(before + (fenced ? "fence.sh\n" : "") + command + wait_on("ur4") +
            "halt\n", profile), 1, setup);
    };
    // ur10, ur11, ur8: the TMA's operands (its buffer at `shared`, completing
    // into 4104, coordinate 0); ur4: that mbar, for the wait.
    const auto prologue = [](uint32_t shared) {
        return one_tma_operands(shared) + "umov ur4, 4104\nsread r20, LANE_ID\nshli r20, r20, 2\n";
    };
    const std::string tma_store = "tma.store.1d [5120, ur8], [ur10], [ur11]\n";
    const std::string tma_load = "tma.load.1d [ur10], [5120, ur8], [ur11]\n";

    // sts -> TMA store source.
    const auto store_source = prologue(0) + "mov r21, 7\nsts [r20], r21\n";
    expect_panic(*machine(store_source, false, tma_store, one_tma_image(1024)),
        {"reason=missing fence.sh: tma store #0 source bytes [0, 64) overlap this warp's earlier "
         "sts [r20], r21 (pc 56, issued cycle", "whose write has not reached the RAM; issue fence.sh between them",
            "inst='tma.store.1d [5120, ur8], [ur10], [ur11]"},
        "sts -> TMA store source");
    auto fenced = machine(store_source, true, tma_store, one_tma_image(1024));
    fenced->run();
    for (uint32_t i = 0; i < 256; ++i) {
        require(fenced->read_gmem_u32(4 * i) == (i < 16 ? 7u : 0x100 + i), "fenced TMA store word " + std::to_string(i));
    }

    // sts -> the descriptor: copied from 6144 into 5120 (zeros: invalid).
    const auto descriptor_copy = [](std::vector<uint8_t>& shmem) {
        write_tma_descriptor(shmem, 6144, plain_tma_1d(16));
        for (uint32_t i = 0; i < 16; ++i) write_u32(shmem, 5120 + 4 * i, 0);
        write_mbar(shmem, 4104, mbar_state(1, 0, 64));
    };
    const auto copy = prologue(1024) + "lds r21, [r20 + 6144]\nsts [r20 + 5120], r21\n";
    expect_panic(*machine(copy, false, tma_load, descriptor_copy),
        {"reason=missing fence.sh: tma #0 descriptor bytes [5120, 5184) overlap this warp's earlier "
         "sts [r20 + 5120], r21", "whose write has not reached the RAM"},
        "sts -> TMA descriptor");
    fenced = machine(copy, true, tma_load, descriptor_copy);
    fenced->run();
    for (uint32_t i = 0; i < 16; ++i) {
        require(fenced->read_shmem_u32(1024 + 4 * i) == 5 * i + 2, "the TMA with the copied descriptor, word " + std::to_string(i));
    }

    // sts -> mma.async on that tile (one bank: the sts is sixteen rounds).
    const auto dense = dense_one_bank_profile(5);
    expect_panic(*dense_mma_machine(dense, two_halves_sts),
        {"reason=missing fence.sh: mma.async #1 A operand bytes [0, 64) overlap this warp's earlier "
         "sts [r20], r21", "inst='mma.async.init.m32n64k8 [ur2], [ur0], [ur1]"},
        "sts -> mma.async");
    auto mma = dense_mma_machine(dense, two_halves_sts + "fence.sh\n");
    mma->run(10000);
    require(nanosm::bits_float(mma->read_reg(0, 0, 10)) == 16.0f, "the fenced MMA did not read the stored 2.0 halves");

    // lds -> TMA load over the words it reads.
    const auto read_then_load = prologue(0) + "lds r21, [r20]\n";
    expect_panic(*machine(read_then_load, false, tma_load, one_tma_image(1024)),
        {"reason=missing fence.sh: tma load #0 destination bytes [0, 64) overlap this warp's earlier "
         "lds r21, [r20] (pc 48, issued cycle", "whose read has not sampled the RAM; issue fence.sh between them"},
        "lds -> TMA load");
    fenced = machine(read_then_load, true, tma_load, one_tma_image(1024));
    fenced->run();
    for (uint32_t lane = 0; lane < 16; ++lane) {
        require(fenced->read_reg(0, lane, 21) == 0x100 + lane, "the fenced lds did not read the old word");
    }
    for (uint32_t i = 0; i < 256; ++i) require(fenced->read_shmem_u32(4 * i) == 5 * i + 2, "the fenced TMA load");
}

// SANITIZER: another warp's access still travelling to the RAM is a RACE,
// not a missing fence (cross-warp ordering is by mbar). Warp 1 runs a
// sixteen-round access (all lanes on bank 0: 1024 + 64 * lane, inside the
// TMA's buffer [1024, 2048)); warp 0, after a stall, has a TMA over those
// words accepted while it is under way:
//   warp 1 sts, warp 0 TMA store (the engine reads, the access writes);
//   warp 1 lds, warp 0 TMA load (the engine writes, the access reads).
// Control, the FA handoff: warp 1's sts, then mbar.arrive (admitted once the
// sts is serviced; no depbar wait); warp 0 waits on the mbar and stores the
// words: the arrive's RMW commits after the store's last round reached the
// RAM, so the TMA's accept finds nothing in flight.
TEST(Shmem, cross_warp_race) {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    const auto& profile = canonical_profile();
    // The last of the TMA's operand umovs stalls, placing the accept inside
    // warp 1's rounds.
    const auto two_warps = [](const std::string& access, const std::string& command) {
        return by_warp("other") + R"(
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
        const auto rounds = grants_of(machine, service);
        const auto accept = std::stoull(message.substr(message.find("cycle=") + 6));
        require(!rounds.empty() && rounds.front().cycle <= accept &&
                accept < rounds.back().cycle + profile.shmem_port_a_depth,
            "the TMA was not accepted while warp 1's access was under way (c" + std::to_string(accept) + "):" +
                describe(rounds));
    };
    auto machine = gmem_debug_machine(profile, two_warps("sts [r20 + 1024], r21 {pred=PT, stall=0, set=sb0, wait=-}",
        "tma.store.1d [5120, ur8], [ur10], [ur11]"), 2, one_tma_image(1024));
    auto message = panic_of(*machine, "warp 1 sts vs warp 0 TMA store");
    require_mentions(message, {"warp=0", "reason=shmem race: tma store #0 source bytes [",
        "overlap warp 1's sts [r20 + 1024], r21 (pc", "whose write has not reached the RAM; order them with an mbar"},
        "cross-warp sts vs TMA store");
    under_way(*machine, message, "A-write");
    machine = gmem_debug_machine(profile, two_warps("lds r22, [r20 + 1024] {pred=PT, stall=0, set=sb0, wait=-}",
        "tma.load.1d [ur10], [5120, ur8], [ur11]"), 2, one_tma_image(1024));
    message = panic_of(*machine, "warp 1 lds vs warp 0 TMA load");
    require_mentions(message, {"warp=0", "reason=shmem race: tma load #0 destination bytes [",
        "overlap warp 1's lds r22, [r20 + 1024] (pc", "whose read has not sampled the RAM; order them with an mbar"},
        "cross-warp lds vs TMA load");
    under_way(*machine, message, "A-read");

    auto handoff = gmem_debug_machine(profile, by_warp("producer") + one_tma_operands(1024) + R"(
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
        )", 2, [](std::vector<uint8_t>& shmem) {
            one_tma_image(1024)(shmem);
            write_mbar(shmem, 4112, mbar_state(1, 1));
        });
    handoff->run();
    for (uint32_t i = 0; i < 256; ++i) {
        require(handoff->read_gmem_u32(4 * i) == (i % 16 == 0 ? 0x66u : 0x100 + 256 + i), "handoff word " + std::to_string(i));
    }
}

// Read / read overlap never PANICs: an lds still under way (one bank:
// sixteen rounds) when an mma.async.init reading the same words is accepted;
// an lds that has not sampled when a TMA store of its words is accepted. Both
// run, and the timing shows the lds really was in flight at the accept.
TEST(Shmem, read_read_overlap_is_clean) {
    const auto dense = dense_one_bank_profile(5);
    auto mma = dense_mma_machine(dense, "sread r20, LANE_ID\nshli r20, r20, 2\nlds r21, [r20]\n");
    mma->run(10000);
    const auto reads = grants_of(*mma, "A-read");
    const auto accept = issued_at(*mma, 7 * 8);
    require(reads.size() >= 16 && accept.size() == 1 && reads[15].cycle + dense.shmem_port_a_depth > accept.front(),
        "the lds was not still in flight when the MMA was accepted:" + describe(reads));
    require(nanosm::bits_float(mma->read_reg(0, 0, 10)) == 8.0f && mma->read_reg(0, 3, 21) == 0x3c003c00u,
        "lds + MMA read/read results");

    const auto& profile = canonical_profile();
    auto store = gmem_debug_machine(profile, one_tma_operands(0) + R"(
        umov ur4, 4104
        sread r20, LANE_ID
        shli r20, r20, 2
        lds r21, [r20] {pred=PT, stall=0, set=sb0, wait=-}
        tma.store.1d [5120, ur8], [ur10], [ur11]
    wait: mbar.try_wait p0, [ur4], 0
        bra.uni wait {pred=!p0, stall=0, set=-, wait=-}
        halt {pred=PT, stall=0, set=-, wait=sb0}
    )", 1, one_tma_image(1024));
    store->run();
    const auto lds = grants_of(*store, "A-read");
    require(!lds.empty() && lds.front().client == "simt" &&
            lds.front().cycle + profile.shmem_port_a_depth > issued_at(*store, 7 * 8).at(0),
        "the lds had sampled before the TMA store's accept:" + describe(lds));
    for (uint32_t i = 0; i < 256; ++i) require(store->read_gmem_u32(4 * i) == 0x100 + i, "lds + TMA store word");
    require(store->read_reg(0, 7, 21) == 0x100 + 7, "lds + TMA store: the lds");
}

// SANITIZER, live mbars are opaque: an mbar.init makes its 8 bytes a live
// mbar until kernel end, and an mbar a launch declares (Launch::mbars) is
// live from launch; only mbar operations may touch it.
//   sts onto it after the init PANICs (at the sts round's grant) naming the
//   mbar; the same sts BEFORE the init is legal (the init repurposes it);
//   a TMA load whose placement writes it PANICs at that placement round
//   (the init must reach the RAM before the TMA's accept: fence.sh; without
//   the fence the in-flight init itself is the missing-fence PANIC), and so
//   does one over a launch-declared mbar another warp is waiting on;
//   mbar.init of a live mbar PANICs (no mbar.inval in v0, so an mbar is
//   initialized once per kernel).
// mbar operations on a live mbar stay legal. The machine has no live-mbar
// PANIC of its own: this is the only check.
TEST(Shmem, live_mbar_opacity) {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    const auto& profile = canonical_profile();
    // The mbar's address is ur4; ur2r copies it into r4 for the lanes' addresses.
    const std::string lanes = R"(
        sread r20, LANE_ID
        shli r20, r20, 2
        ur2r r4, ur4
        add r20, r20, r4
        mov r21, 9
    )";
    const std::string placement = one_tma_operands(1024) + "umov ur4, 4104\numov ur5, 1032\nmbar.init [ur5], 1\n";
    const std::string load = "tma.load.1d [ur10], [5120, ur8], [ur11]\n" + wait_on("ur4") + "halt\n";
    // Warp 0 loads 64 bytes over shmem [1024, 1088) after a delay; warp 1 spins on the declared mbar at 1032.
    std::string declared_load = by_warp("waiter") + "mov r20, 0\n";
    for (int i = 0; i < 8; ++i) declared_load += "addi r20, r20, 1\n";
    declared_load += one_tma_operands(1024) + "umov ur4, 4104\ntma.load.1d [ur10], [5120, ur8], [ur11]\n" +
        wait_on("ur4", "loaded") + "halt\nwaiter:\numov ur1, 1032\n" + wait_on("ur1", "spin") + "halt\n";
    struct Case {
        std::string what;
        std::string program;
        uint32_t warps;
        std::function<void(std::vector<uint8_t>&)> setup;
        std::vector<uint32_t> declared;
        std::vector<std::string> needles;
    };
    const std::vector<Case> cases = {
        {"sts onto a live mbar", "umov ur4, 4160\nmbar.init [ur4], 1\n" + lanes + "sts [r20], r21\nhalt\n", 1, {}, {},
            {"reason=live mbar opacity: sts writes bytes [4160, 4168) of the live mbar [4160, 4168) "
             "(mbar.init by warp 0 pc 8 at cycle ",
                "until kernel end only mbar operations may touch it", "inst='sts [r20], r21"}},
        {"TMA placement over a live mbar", placement + "fence.sh\n" + load, 1, one_tma_image(64), {},
            {"mbar address=1032 reason=live mbar opacity: tma load #0 placement writes bytes "
             "[1032, 1040) of the live mbar [1032, 1040) (mbar.init by warp 0 pc 40 at cycle "}},
        {"TMA load over an mbar.init in flight", placement + load, 1, one_tma_image(64), {},
            {"reason=missing fence.sh: tma load #0 destination bytes [1032, 1040) overlap this warp's "
             "earlier mbar.init [ur5], 1 (pc 40"}},
        {"TMA load over a launch-declared mbar with a live waiter", declared_load, 2,
            [](std::vector<uint8_t>& shmem) {
                write_mbar(shmem, 1032, mbar_state(1, 1));
                one_tma_image(64)(shmem);
            },
            {1032},
            {"reason=live mbar opacity: tma load #0 placement writes bytes [1032, 1040) of the live mbar [1032, 1040) "
             "(declared by the launch image)"}},
        {"mbar.init over a live mbar", "umov ur4, 4160\nmbar.init [ur4], 1\nmbar.init [ur4], 2\nhalt\n", 1, {}, {},
            {"reason=mbar.init over the live mbar [4160, 4168) (mbar.init by warp 0 pc 8 at cycle ",
                "(v0 has no mbar.inval: an mbar is initialized once per kernel)", "inst='mbar.init [ur4], 2"}},
    };
    for (const auto& c : cases) {
        auto machine = gmem_debug_machine(profile, compile(c.program, profile), c.warps, c.setup, c.declared);
        const auto message = panic_of(*machine, c.what);
        for (const auto& needle : c.needles) require_mentions(message, {needle}, c.what);
    }
    auto repurpose = gmem_debug_machine(
        profile, compile("umov ur4, 4160\n" + lanes + "sts [r20], r21\nmbar.init [ur4], 1\nhalt\n", profile));
    repurpose->run();
    const auto state = repurpose->read_mbar_state(4160);
    require(state.valid && state.expected == 1 && repurpose->read_shmem_u32(4168) == 9,
        "an sts before the mbar.init was refused, or the init did not land after it");
}

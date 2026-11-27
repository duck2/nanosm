// qtx's memory objects (src/qtxc/objects.cpp, expressions.cpp; docs/qtx.md
// "Memory"): `.shmem` / `.tmem` name a piece of the kernel's own memory and
// qtxc places it; constants are expressions; ld.param reads a parameter.
// Smoke tests: objects land where the rule says and the code reaches them
// by name, an overlay lies on what it lists, and the refusals read well.
#include "test_support.hpp"

#include "image.hpp"

using namespace nanosm_test;

namespace {

const nanosm::MemoryObject& object_of(const std::vector<nanosm::MemoryObject>& objects, const std::string& name) {
    for (const auto& object : objects) {
        if (object.name == name) return object;
    }
    throw std::runtime_error("no memory object named " + name);
}

void asm_refused(const std::string& text, const std::string& needle, const std::string& what) {
    expect_error([&] { (void)nanosm::assemble_asm(text, canonical_profile()); }, {needle}, what);
}

std::string replaced(std::string text, const std::string& from, const std::string& to) {
    const auto found = text.find(from);
    require(found != std::string::npos, "the assembly has no '" + from + "':\n" + text);
    return text.replace(found, from.size(), to);
}

// An mbar's state, read from shared memory.
nanosm::MbarState mbar_at(const nanosm::Machine& machine, uint32_t address) {
    return nanosm::decode_mbar_state(machine.read_shmem_u32(address) |
        (static_cast<uint64_t>(machine.read_shmem_u32(address + 4)) << 32));
}

} // namespace

// Objects are placed one after another, each on its alignment; the code
// reaches them by name, element and property; a parameter is read with
// ld.param; assembly carries the objects with their addresses.
TEST(Memory, objects_place_and_compute) {
    const auto& profile = canonical_profile();
    const std::string source = R"(
        .param u32 BIAS
        .equ   N, 2
        .shmem TILE[N], rows=4, cols=16, elem=4     ; 64-byte rows, 256 bytes each: 0 and 256
        .shmem PAD, bytes=100, align=4              ; 512 .. 611
        .shmem BUF[2], bytes=64                     ; a plain buffer is on 64: 640 and 704
        .tmem  ACC[2], rows=6, cols=8               ; 48 words, on four rows (32): 0 and 64
        sread %lane, LANE_ID
        shli  %byte, %lane, log2(TILE.row_bytes / 16)
        ld.param.u32 %bias, [BIAS]
        add   %v, %lane, %bias
        sts   [%byte + BUF[1]], %v
        lds   %back, [%byte + BUF[1]]
        umov  %u, (TILE[1] - TILE) / TILE.row_bytes + log2(TILE.row_bytes) + (BUF.bytes >> 4) + ACC.stride + ACC[1].words
        umov  %a, ACC[1]
        halt
    )";
    const auto program = nanosm::compile_qtx(source, profile);
    const auto& tile = object_of(program.objects, "TILE");
    const auto& pad = object_of(program.objects, "PAD");
    const auto& buf = object_of(program.objects, "BUF");
    const auto& acc = object_of(program.objects, "ACC");
    require(tile.address == 0 && tile.count == 2 && tile.size == 256 && tile.stride == 256 && tile.alignment == 64,
        "a tile is rows x cols x elem bytes on a multiple of its row and of 64");
    require(pad.address == 512 && pad.alignment == 4 && pad.stride == 100, "align= replaces a plain buffer's 64");
    require(buf.address == 640 && buf.stride == 64, "a plain buffer starts on the next multiple of 64");
    require(acc.tmem && acc.address == 0 && acc.size == 48 && acc.stride == 64 && acc.alignment == 32,
        "a tmem tile is in words, on four of its rows, its stride padded to that");
    // The launch flag: ld.param waits at its lds.uni, the read (the umov
    // before it only loads the parameter's address, and goes ahead); an
    // instruction that names an object does not, whatever the expression.
    for (std::size_t index = 0; index < program.instructions.size(); ++index) {
        const bool reads_param = program.instructions[index].opcode == nanosm::Opcode::lds_uni;
        require((program.named_flags[index] != 0) == reads_param,
            "launch-flag wait on the wrong instruction: " + program.instructions[index].text);
    }
    // Assembly carries each object with its address and reads back to the same kernel.
    const auto text = nanosm::emit_asm(program);
    require(text.find(".shmem BUF[2], bytes=64, align=64, at=640") != std::string::npos &&
            text.find(".tmem ACC[2], rows=6, cols=8, align=32, at=0") != std::string::npos,
        "the assembly does not carry the objects:\n" + text);
    const auto kernel = nanosm::assemble_asm(text, profile);
    require(kernel.objects.size() == 4 && object_of(kernel.objects, "BUF").address == 640,
        "the assembled kernel lost its objects");
    require(nanosm::emit_kernel_info(kernel).find("\"name\": \"BUF\"") != std::string::npos,
        "the kernel info does not list the objects");
    auto launch = launch_warps(1);
    launch.args = {{"BIAS", uint64_t{1000}}};
    nanosm::Machine machine(profile, kernel, std::move(launch));
    machine.run();
    require_lanes(shmem_lanes(machine, 704), [](uint32_t lane) { return 1000 + lane; },
        "BUF[1] after a store through [%byte + BUF[1]]");
    require_lanes(machine, number_of(kernel.register_names, "back"), [](uint32_t lane) { return 1000 + lane; },
        "a load through [%byte + BUF[1]]");
    // 256 / 64 + 6 + (128 >> 4) + 64 + 48
    require(machine.read_ureg(0, number_of(kernel.register_names, "u")) == 4 + 6 + 8 + 64 + 48,
        "a constant expression over elements and properties");
    require(machine.read_ureg(0, number_of(kernel.register_names, "a")) == 64, "a tmem element's address");
}

// over= lays an object on the ones it lists; the first of them takes the
// overlay's alignment before anything is placed.
TEST(Memory, overlay) {
    const auto& profile = canonical_profile();
    const std::string objects = R"(
        .shmem A, bytes=64
        .shmem Q[2], rows=2, cols=16, elem=4            ; 128 bytes each
        .shmem K, rows=4, cols=16, elem=4               ; 256 bytes
    )";
    const auto program = nanosm::compile_qtx(objects + R"(
        .shmem STAGE[2], rows=1, cols=64, elem=4, over=Q, K   ; 256-byte rows: Q moves to 256
        halt
    )", profile);
    require(object_of(program.objects, "Q").address == 256 && object_of(program.objects, "K").address == 512,
        "the first object under an overlay did not take its alignment");
    const auto& stage = object_of(program.objects, "STAGE");
    require(stage.address == 256 && stage.stride == 256 && stage.over.size() == 2, "the overlay is not on what it lists");
    qtx_refused(objects + ".shmem STAGE[2], rows=1, cols=64, elem=4, over=K, Q\nhalt\n",
        "Q (at 64) does not follow K (which ends at 768)", "an overlay over objects out of order");
    qtx_refused(objects + ".shmem STAGE[2], rows=1, cols=64, elem=4, over=Q\nhalt\n",
        "it needs 512 bytes and the objects it lies over have 256", "an overlay larger than what it lies on");
    qtx_refused(objects + ".shmem STAGE, bytes=64, over=Q[2]\nhalt\n",
        "over=Q[2] is past the end of .shmem Q, which has 2 elements", "an overlay over an element that is not there");
    qtx_refused(objects + ".shmem STAGE, bytes=64, over=NOPE\nhalt\n", "there is no .shmem object 'NOPE'",
        "an overlay over nothing");
    qtx_refused(objects + ".tmem T, rows=4, cols=8\n.shmem STAGE, bytes=16, over=T\nhalt\n",
        "T is in tensor memory, not in shared memory", "an overlay across memories");
}

TEST(Memory, refusals) {
    const std::string tile = ".shmem QBUF[2], rows=32, cols=64, elem=2\n";
    qtx_refused(tile + "umov %a, QBUF[2]\nhalt\n", "QBUF[2] is past the end of .shmem QBUF, which has 2 elements",
        "an element past the end");
    qtx_refused(tile + "umov %a, QBUF.rows\nhalt\n", ".shmem QBUF has .bytes (QBUF[i].bytes for one element), .stride and .row_bytes",
        "an unknown property");
    qtx_refused(".shmem B, bytes=64, layout=dense\nhalt\n", "an object has no layout", "layout= on an object");
    qtx_refused(".shmem B, bytes=64, at=128\nhalt\n", "qtxc places objects", "at= in qtx");
    qtx_refused(".shmem B, rows=4, cols=16\nhalt\n", "expected .shmem NAME[COUNT], rows=R, cols=C, elem=BYTES",
        "a tile without its element size");
    qtx_refused(".tmem T, rows=4, cols=24\nhalt\n", "a tmem row is a power of two of words, 4 or more", "a tmem row of 24 words");
    qtx_refused(".tmem T, rows=4, cols=8, align=8\nhalt\n", "a multiple of four of its rows (32 words)",
        "a tmem object off its bands of four rows");
    qtx_refused(".shmem BIG[3], bytes=32768\nhalt\n",
        ".shmem BIG does not fit: it needs 98304 bytes at 0", "an object larger than shared memory");
    qtx_refused(".param desc D\n.shmem BIG, bytes=65536 - 32\nhalt\n",
        "shared memory below the launch block (.param / .mbar / .data, at the top) ends at 65472",
        "an object that reaches the launch block");
    qtx_refused(".equ B, 4\n.shmem B, bytes=64\nhalt\n", "duplicate constant 'B'", "an object named like a constant");
    qtx_refused(".shmem B, bytes=64\nB: halt\n", "duplicate symbol 'B'", "a label named like an object");
    qtx_refused("umov %a, log2(48)\nhalt\n", "log2(48): not a power of two", "log2 of 48");
    qtx_refused("umov %a, 8 / (4 - 4)\nhalt\n", "division by zero", "a zero divisor");
    qtx_refused(".equ N, 4\numov %a, N.bytes\nhalt\n", "only a .shmem or .tmem object has properties", "a property of a constant");
    qtx_refused(".equ N, 4\numov %a, N[1]\nhalt\n", "N is not an array", "an element of a constant");
    qtx_refused("mov %a, 1 << 30\nhalt\n", "does not fit the immediate field", "an expression too large for its field");
    // ld.param: a parameter of the right type, and nothing else.
    qtx_refused(".param u32 N\nld.param.f32 %x, [N]\nhalt\n", "N is a .param u32 (read it with ld.param.u32",
        "ld.param.f32 of a u32");
    qtx_refused(".param desc D\nld.param.u32 %x, [D]\nhalt\n", "a descriptor is named in a tma instruction's tile operand",
        "ld.param of a descriptor");
    qtx_refused(".shmem B, bytes=64\nld.param.u32 %x, [B]\nhalt\n", "with NAME a .param of this kernel", "ld.param of an object");
    // An `.equ` that names a section is an address in it: its users wait.
    const auto program = nanosm::compile_qtx(
        ".param u32 P\n.mbar M[2], expect 1\n.equ SECOND, M[1]\n.equ PLUS, P + 0\n.equ PLAIN, 4 * 2\n"
        "umov %a, SECOND\numov %b, PLUS\numov %c, PLAIN\nhalt\n", canonical_profile());
    require(program.named_flags.at(0) != 0 && program.named_flags.at(1) != 0 && program.named_flags.at(2) == 0,
        "a constant made from a section's address lost (or a plain constant gained) its launch-flag wait");
}

// Edge cases of placement, names and constant expressions.
TEST(Memory, placement_and_name_edge_cases) {
    const auto& profile = canonical_profile();
    // 1. An overlay may use its targets up to the last one's own size, not its
    // stride: the padding after an object belongs to whoever is declared next.
    qtx_refused(".shmem A, bytes=4\n.shmem B, bytes=4, align=4\n.shmem V, bytes=8, align=4, over=A\nhalt\n",
        "it needs 8 bytes and the objects it lies over have 4", "an overlay reaching into the next object");
    const std::string padded = ".shmem A, bytes=100, align=64\n.shmem B, bytes=64, align=4\n";
    qtx_refused(padded + ".shmem X, bytes=128, align=64, over=A\nhalt\n",
        "it needs 128 bytes and the objects it lies over have 100", "an overlay as large as its target's stride");
    {
        const auto program = nanosm::compile_qtx(
            padded + ".shmem Y, bytes=164, align=4, over=A, B\n"
                     ".shmem E[2], bytes=100, align=64\n"              // 192 and 320: a stride of 128
                     ".shmem Z, bytes=228, align=4, over=E[0], E[1]\nhalt\n", profile);
        require(object_of(program.objects, "B").address == 100 && object_of(program.objects, "Y").address == 0,
            "an overlay over an object and the one declared right after it");
        require(object_of(program.objects, "Z").address == 192,
            "an overlay over two elements of one array, listed one by one");
    }
    // 2. Names beginning qtxc_ are the compiler's (ld.param's register, its
    // copies): a writer's use is refused (it would share the register).
    qtx_refused(".param u32 VALUE\numov %qtxc_param, 123\nld.param.u32 %x, [VALUE]\nhalt\n",
        "%qtxc_param: names beginning %qtxc_ are qtxc's own", "a writer's register named like ld.param's");
    qtx_refused(".reg %qtxc_copy<8>\nhalt\n", "names beginning %qtxc_ are qtxc's own", "a group named like qtxc's copies");
    qtx_refused("mov %a, 1\nsetp.ne.u32 %qtxc_p, %a, 0\nhalt\n", "names beginning %qtxc_ are qtxc's own",
        "a predicate with qtxc's prefix");
    // ... also read as a source where a uniform register may go, before or
    // after an ld.param has named it (each source slot).
    for (const std::string read : {"add %ans, %v, %qtxc_param", "add %ans, %qtxc_param, %v",
             "fma %ans, %v, %v, %qtxc_param"}) {
        const std::string load = "ld.param.u32 %x, [N]\n";
        const std::string before = ".param u32 N\nsread %v, LANE_ID\n" + read + "\n" + load + "halt\n";
        const std::string after = ".param u32 N\n" + load + "sread %v, LANE_ID\n" + read + "\nhalt\n";
        qtx_refused(before, "%qtxc_param: names beginning %qtxc_ are qtxc's own", "'" + read + "' before an ld.param");
        qtx_refused(after, "%qtxc_param: names beginning %qtxc_ are qtxc's own", "'" + read + "' after an ld.param");
    }
    // 3. Nothing wraps: a product is checked before it is made, and an
    // array's extent is 64 bits wide until it has been checked to fit.
    qtx_refused(".equ N, (1 << 32) * (1 << 32)\nhalt\n", "the value is out of range", "a product past 64 bits");
    qtx_refused("umov %a, (1 << 32) << 32\nhalt\n", "the value is out of range", "a shift past 64 bits");
    qtx_refused(".shmem BIG[1024], bytes=1 << 22\n.shmem NEXT, bytes=64\nhalt\n",
        ".shmem BIG does not fit: it needs 4294967296 bytes at 0", "an array whose extent wraps 32 bits");
    // 4. An address written in assembly (at=) gets the checks placement gives.
    const auto text = nanosm::emit_asm(nanosm::compile_qtx(".shmem A, bytes=64\n.shmem B, bytes=4, align=4\nhalt\n", profile));
    const std::string line = ".shmem B, bytes=4, align=4, at=64";
    (void)nanosm::assemble_asm(replaced(text, line, ".shmem B, bytes=4, align=4, at=128"), profile);
    asm_refused(replaced(text, line, ".shmem B, bytes=4, align=4, at=65535"),
        "it is at 65535, which is not a multiple of its alignment, 4 bytes", "an object off its alignment");
    asm_refused(replaced(text, line, ".shmem B, bytes=4, align=4, at=65536"),
        ".shmem B: it does not fit: it needs 4 bytes at 65536", "an object past the end of shared memory");
    asm_refused(replaced(text, line, ".shmem B, bytes=4, align=4, at=0"),
        "overlaps .shmem A (0 .. 64); two objects share bytes only when one says over= the other",
        "an object written on top of another");
    asm_refused(replaced(text, line, ".shmem B, bytes=4, align=4, over=MISSING, at=8"),
        "there is no .shmem object 'MISSING'", "an overlay over nothing, with an address");
    asm_refused(replaced(text, line, ".shmem B, bytes=4, align=4, over=A, at=64"),
        "the objects it lies over (over=) are at 0 .. 64", "an overlay written outside what it names");
    (void)nanosm::assemble_asm(replaced(text, line, ".shmem B, bytes=4, align=4, over=A, at=60"), profile);
    // 5. A count is a constant expression wherever an array or a group has one.
    {
        const auto program = nanosm::compile_qtx(
            ".equ STAGES, 2\n.shmem BUF[STAGES], bytes=64\n.mbar FULL[STAGES], expect 1\n"
            ".reg %v<4 * STAGES>\n.ureg %c<STAGES>\n"
            "umov %c1, FULL[STAGES - 1]\nmov %v7, 1\nhalt\n", profile);
        uint32_t barriers = 0;
        for (const auto& section : program.sections) {
            if (section.name == "FULL") barriers = section.count;
        }
        require(barriers == 2 && object_of(program.objects, "BUF").count == 2,
            "an .mbar array and a .shmem array counted by one constant");
    }
    qtx_refused(".mbar FULL[STAGES], expect 1\nhalt\n", ".mbar FULL[STAGES]: bad barrier count (unknown name 'STAGES')",
        "an .mbar count that is no constant");
    qtx_refused(".reg %v<N>\nhalt\n", ".reg %v<N>: bad count (unknown name 'N')", "a group count that is no constant");
}

// `.equ XX, X` is another name for section X: an instruction that names XX
// waits on X's launch flag like one that names X (with no wait it would read
// zeros when the sanitizer is off, and PANIC when it is on). A chain of such
// lines too.
TEST(Memory, section_alias_waits_on_its_flag) {
    for (const std::string name : {"X", "XX", "XXX"}) {
        auto launch = launch_warps(1);
        launch.args = {{"X", uint64_t{0x12345678}}};
        nanosm::Machine machine(canonical_profile(), compile(
            ".param u32 X\n.equ XX, X\n.equ XXX, XX\n"
            "umov ur1, " + name + "\nlds.uni r1, [ur1]\nsread r2, LANE_ID\nshli r2, r2, 2\nsts [r2], r1\nhalt\n"),
            std::move(launch));
        machine.run();
        require(machine.read_shmem_u32(0) == 0x12345678u, "umov ur1, " + name + " read its parameter before it landed");
    }
}

// `.mbar NAME[C]` is one section of C barriers, element i at NAME + 8 i and
// written NAME[i]; naming any of it waits for the whole array, and an index
// past its end is refused.
TEST(Memory, mbar_array) {
    const auto& profile = canonical_profile();
    const auto kernel = compile(R"(
.warps 1
.param u32 A
.mbar PAIR[2], expect 3
.mbar ONE, expect 1
        umov ur1, PAIR
        umov ur2, PAIR[1]
        umov ur3, ONE
        mbar.arrive [ur2]
        halt
)");
    require(kernel.sections.size() == 3 && kernel.sections[1].count == 2 &&
            nanosm::section_bytes(kernel.sections[1]) == 16 &&
            kernel.sections[2].address == kernel.sections[1].address + 16,
        "an mbar array is not one section of 8 bytes per barrier");
    const auto text = nanosm::emit_hex(kernel);
    require(text.find(".mbar PAIR[2], ") != std::string::npos &&
            nanosm::load_hex(text).sections[1].count == 2, "the hex text lost the array");
    nanosm::LaunchImage image{kernel, {}};
    image.launch.args = {{"A", uint64_t{1}}};
    auto loaded = nanosm::load_launch_image(nanosm::emit_launch_image(image));
    require(loaded.kernel.sections[1].count == 2, "a saved image lost the array");
    nanosm::Machine machine(profile, std::move(loaded.kernel), std::move(loaded.launch));
    machine.run();
    const auto pair = kernel.sections[1].address;
    const auto first = mbar_at(machine, pair);
    const auto second = mbar_at(machine, pair + 8);
    require(first.valid && first.expected == 3 && first.remaining == 3, "element 0 of the array did not start expecting 3");
    require(second.valid && second.expected == 3 && second.remaining == 2,
        "element 1 of the array is not where NAME[1] says (the arrive went elsewhere)");
    for (const char* bad : {"umov ur1, PAIR[2]\nhalt\n", ".equ P, PAIR[7]\nhalt\n", "umov ur1, ONE[1]\nhalt\n"}) {
        expect_error([&] { (void)compile(std::string(".mbar PAIR[2], expect 1\n.mbar ONE, expect 1\n") + bad); },
            {"is past the end of .mbar"}, std::string("an index past an mbar array's end: ") + bad);
    }
    expect_error([] { (void)compile(".mbar PAIR[0], expect 1\nhalt\n"); }, {"an array has 1.."}, "an empty mbar array");
}

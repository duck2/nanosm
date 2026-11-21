// The TMA's instruction form, its operand read and its descriptor
// (trash/rfc17.md; trash/rfc17-slice-2026-11-17.md): tma.load.Nd /
// tma.store.Nd with uniform-register operands, read two per round over the
// URF's two read ports, and a tile cut from a tensor of up to four
// dimensions. Smoke tests: the data is right, plus the few timing facts of
// the read (one test).
#include "test_support.hpp"

namespace {

using namespace nanosm_test;

// A four-dimensional tensor of 32-bit words at gmem 0 (channel, head,
// sequence, batch), dense, each word 1000 + its element index.
constexpr std::array<uint32_t, 4> tensor_dims{8, 3, 6, 2};
constexpr std::array<uint32_t, 3> tensor_strides{8, 24, 144};
constexpr uint32_t tensor_elements = 288;
constexpr uint32_t descriptor_4d = 4096;
constexpr uint32_t descriptor_2d = 4160; // the same words as a two-dimensional tensor of rows of 8
constexpr uint32_t barrier = 4352;

uint32_t tensor_word(const std::array<int64_t, 4>& at) {
    int64_t index = at[0];
    for (uint32_t dimension = 0; dimension < 4; ++dimension) {
        if (at[dimension] < 0 || at[dimension] >= tensor_dims[dimension]) return 0; // zero-filled
        if (dimension > 0) index += at[dimension] * tensor_strides[dimension - 1];
    }
    return 1000u + static_cast<uint32_t>(index);
}

// Tiles of 8 x `rows` along the sequence (row dimension 2), and the 2-D view.
nanosm::TmaDescriptor sequence_tiles(uint16_t rows) {
    nanosm::TmaDescriptor descriptor;
    descriptor.dimensions = tensor_dims;
    descriptor.strides = tensor_strides;
    descriptor.tile_length = 8;
    descriptor.tile_rows = rows;
    descriptor.row_dimension = 2;
    return descriptor;
}

nanosm::TmaDescriptor rows_of_8(uint16_t rows) {
    nanosm::TmaDescriptor descriptor;
    descriptor.dimensions = {8, tensor_elements / 8, 1, 1};
    descriptor.strides = {8, 0, 0};
    descriptor.tile_length = 8;
    descriptor.tile_rows = rows;
    return descriptor;
}

// The launch: the tensor in gmem, both descriptors, `barriers` barriers from
// `barrier` on (8 bytes apart) each expecting `tx` bytes, shmem below 4096
// poisoned so a zero fill shows.
nanosm::Launch tensor_launch(uint32_t warps, uint16_t tile_rows, uint32_t barriers, uint32_t tx) {
    nanosm::Launch launch;
    launch.active_warps = warps;
    auto& gmem = image_bytes(launch.gmem_image, tensor_elements * 4);
    for (uint32_t i = 0; i < tensor_elements; ++i) write_u32(gmem, 4 * i, 1000u + i);
    auto& shmem = image_bytes(launch.test_shmem, 8192);
    for (uint32_t address = 0; address < descriptor_4d; address += 4) write_u32(shmem, address, 0xdeadbeefu);
    write_tma_descriptor(shmem, descriptor_4d, sequence_tiles(tile_rows));
    write_tma_descriptor(shmem, descriptor_2d, rows_of_8(tile_rows));
    for (uint32_t i = 0; i < barriers; ++i) write_tx_mbar(shmem, barrier + 8 * i, tx);
    return launch;
}

// The 8 x rows tile at `buffer` holds the tensor's elements from `origin`
// on, the rows stepping along `row_dimension`.
void require_tile(const nanosm::Machine& machine, uint32_t buffer, std::array<int64_t, 4> origin, uint32_t rows,
    uint32_t row_dimension, const std::string& what) {
    for (uint32_t y = 0; y < rows; ++y) {
        for (uint32_t x = 0; x < 8; ++x) {
            auto at = origin;
            at[0] += x;
            at[row_dimension] += y;
            const auto actual = machine.read_shmem_u32(buffer + 4 * (8 * y + x));
            require(actual == tensor_word(at), what + ": element (" + std::to_string(x) + ", " + std::to_string(y) +
                ") is " + std::to_string(actual) + ", expected " + std::to_string(tensor_word(at)));
        }
    }
}

// The same words seen as rows of 8 (descriptor_2d): rows first_row .. at
// `buffer`, row after row.
void require_rows(const nanosm::Machine& machine, uint32_t buffer, uint32_t first_row, uint32_t rows,
    const std::string& what) {
    for (uint32_t word = 0; word < 8 * rows; ++word) {
        require(machine.read_shmem_u32(buffer + 4 * word) == 1000u + 8 * first_row + word,
            what + ": word " + std::to_string(word));
    }
}

std::string wait_on(const std::string& ur, const std::string& label) {
    return label + ":\nmbar.try_wait p0, [" + ur + "], 0\n@!p0 bra.uni " + label + "\n";
}

// Issue cycles of the instructions named `name`, and the start cycles of the
// TMA's command slices (the cycle each command was pushed into the queue).
std::vector<uint64_t> issues_named(const nanosm::Machine& machine, const std::string& name) {
    std::vector<uint64_t> cycles;
    for (const auto& event : machine.trace().events()) {
        if (is_issue_event(event) && event.name == name) cycles.push_back(event.start_cycle);
    }
    return cycles;
}

std::vector<uint64_t> command_pushes(const nanosm::Machine& machine) {
    std::vector<uint64_t> cycles;
    for (const auto& event : machine.trace().events()) {
        if (event.name == "tma command") cycles.push_back(event.start_cycle);
    }
    std::sort(cycles.begin(), cycles.end());
    return cycles;
}

// N = 1 .. 4 from one descriptor: each reads its first N coordinates and
// takes the rest as zero (here c = 0, 1, 2, 1 for all four commands).
void test_four_coordinate_counts() {
    nanosm::Machine machine(canonical_profile(), compile(R"(
        umov ur0, 0
        umov ur1, 256
        umov ur2, 512
        umov ur12, 768
        umov ur3, 4352
        umov ur8, 0
        umov ur9, 1
        umov ur10, 2
        umov ur11, 1
        tma.load.1d [ur0], [4096, ur8], [ur3]
        tma.load.2d [ur1], [4096, ur8], [ur3]
        tma.load.3d [ur2], [4096, ur8], [ur3]
        tma.load.4d [ur12], [4096, ur8], [ur3]
    )" + wait_on("ur3", "loaded") + "halt\n"), tensor_launch(1, 2, 1, 4 * 64));
    machine.run(20000);
    require_tile(machine, 0, {0, 0, 0, 0}, 2, 2, ".1d (c0 only)");
    require_tile(machine, 256, {0, 1, 0, 0}, 2, 2, ".2d (c0, c1)");
    require_tile(machine, 512, {0, 1, 2, 0}, 2, 2, ".3d (c0 .. c2)");
    require_tile(machine, 768, {0, 1, 2, 1}, 2, 2, ".4d (c0 .. c3)");
    require(machine.trace().summary().tma_command_completions == 4, "four commands completed");
}

// A tile of 8 x 4 cut from the four-dimensional tensor at (-2, head 2,
// sequence 4, batch 1): the first two channels and the last two rows are
// outside it and zero-filled. A second command names head 3, outside the
// tensor's three heads: a dimension the tile does not walk is out, so the
// whole tile is zero.
void test_tile_from_a_four_dimensional_tensor() {
    nanosm::Machine machine(canonical_profile(), compile(R"(
        umov ur0, 0
        umov ur1, 512
        umov ur3, 4352
        umov ur8, -2
        umov ur9, 2
        umov ur10, 4
        umov ur11, 1
        tma.load.4d [ur0], [4096, ur8], [ur3]
        umov ur9, 3
        tma.load.4d [ur1], [4096, ur8], [ur3]
    )" + wait_on("ur3", "loaded") + "halt\n"), tensor_launch(1, 4, 1, 2 * 128));
    machine.run(20000);
    require_tile(machine, 0, {-2, 2, 4, 1}, 4, 2, "tile at (-2, 2, 4, 1)");
    for (uint32_t y : {2u, 3u}) require(machine.read_shmem_u32(4 * 8 * y) == 0, "an out-of-range row is not zero");
    require(machine.read_shmem_u32(0) == 0 && machine.read_shmem_u32(4 * 2) != 0, "the channel boundary is wrong");
    require_tile(machine, 512, {-2, 3, 4, 1}, 4, 2, "tile at head 3 (outside)");
    require(machine.read_shmem_u32(512 + 4 * 2) == 0, "a tile outside the heads was not zero-filled");
    const auto& summary = machine.trace().summary();
    require(summary.tma_logical_bytes == 256 && summary.tma_useful_bytes == 6 * 2 * 4,
        "logical / useful bytes: two 128-byte tiles, 12 elements in the tensor");
}

// Two warps issue two commands each, back to back, with different
// coordinates: no command takes another's words (the tag travels with each
// pair), and no two tma issues are closer than the first one's rounds.
void test_back_to_back_commands_from_two_warps() {
    nanosm::Machine machine(canonical_profile(), compile(R"(
        usread ur0, WARP_ID
        ushli ur1, ur0, 8
        uaddi ur2, ur1, 128
        ushli ur3, ur0, 3
        uaddi ur3, ur3, 4352
        umov  ur8, 0
        umov  ur9, ur0
        umov  ur10, 3
        tma.load.2d [ur1], [4096, ur8], [ur3]
        tma.load.3d [ur2], [4096, ur8], [ur3]
    )" + wait_on("ur3", "loaded") + "halt\n"), tensor_launch(2, 2, 2, 2 * 64), {.timeline = true});
    machine.run(20000);
    for (uint32_t warp = 0; warp < 2; ++warp) {
        const auto where = "warp " + std::to_string(warp);
        require_tile(machine, 256 * warp, {0, warp, 0, 0}, 2, 2, where + " .2d");
        require_tile(machine, 256 * warp + 128, {0, warp, 3, 0}, 2, 2, where + " .3d");
    }
    std::vector<std::pair<uint64_t, uint32_t>> issues; // (cycle, rounds)
    for (const auto cycle : issues_named(machine, "tma.load.2d")) issues.emplace_back(cycle, nanosm::tma_operand_rounds(2));
    for (const auto cycle : issues_named(machine, "tma.load.3d")) issues.emplace_back(cycle, nanosm::tma_operand_rounds(3));
    std::sort(issues.begin(), issues.end());
    require(issues.size() == 4, "four tma issues");
    for (std::size_t i = 1; i < issues.size(); ++i) {
        require(issues[i].first >= issues[i - 1].first + issues[i - 1].second,
            "a tma was accepted while the previous one still had rounds to start");
    }
}

// The read's cycles (canonical depths, Lu 1). A tma.load.4d accepted at T
// samples urS / urM at T + 1 and its coordinates at T + 2 and T + 3; the
// command is pushed into the queue at T + 4. A uniform reader behind it is
// refused at T + 1 and T + 2 ("uniform read ports held") and issues at
// T + 3. A .2d holds the ports one cycle and is pushed at T + 3. In a second
// machine a vector instruction and a uniform write that reads nothing issue
// right behind the .4d.
void test_uniform_reader_refused_during_the_rounds() {
    const auto& profile = canonical_profile();
    require(nanosm::tma_operand_rounds(1) == 2 && nanosm::tma_operand_rounds(2) == 2 &&
            nanosm::tma_operand_rounds(3) == 3 && nanosm::tma_operand_rounds(4) == 3,
        "rounds for N = 1 .. 4");
    require(nanosm::issue_to_tma_command(2, profile) == 3 && nanosm::issue_to_tma_command(4, profile) == 4,
        "a .2d is pushed at issue + 3, a .4d at issue + 4");
    const std::string setup = R"(
        umov ur0, 0
        umov ur4, 256
        umov ur1, 4352
        umov ur2, 5
        umov ur8, 0
        umov ur9, 0
        umov ur10, 0
        umov ur11, 0
    )";
    nanosm::Machine refused(profile, compile(setup + R"(
        tma.load.4d [ur0], [4096, ur8], [ur1]
        uaddi ur3, ur2, 1
        tma.load.2d [ur4], [4096, ur8], [ur1]
        uaddi ur5, ur2, 2
    )" + wait_on("ur1", "loaded") + "halt\n"), tensor_launch(1, 2, 1, 2 * 64), {.timeline = true});
    refused.run(20000);
    const auto four = issues_named(refused, "tma.load.4d");
    const auto two = issues_named(refused, "tma.load.2d");
    const auto readers = issues_named(refused, "uaddi");
    const auto pushes = command_pushes(refused);
    require(four.size() == 1 && two.size() == 1 && readers.size() == 2 && pushes.size() == 2, "the run's issues");
    require(readers[0] == four[0] + 3, "the reader behind the .4d issued at T + " + std::to_string(readers[0] - four[0]));
    require(readers[1] == two[0] + 2, "the reader behind the .2d issued at T + " + std::to_string(readers[1] - two[0]));
    require(pushes[0] == four[0] + 4 && pushes[1] == two[0] + 3, "the commands were pushed at the wrong cycles");
    require(refused.trace().summary().urf_read_ports_held == 3, "three refused warp-cycles");
    require(refused.trace().summary_text().find("urf: writes=10 read_ports_held=3\n") != std::string::npos,
        "the summary's urf line");
    require(refused.read_ureg(0, 3) == 6 && refused.read_ureg(0, 5) == 7, "the readers' results");
    require_tile(refused, 0, {0, 0, 0, 0}, 2, 2, ".4d");
    require_tile(refused, 256, {0, 0, 0, 0}, 2, 2, ".2d");

    nanosm::Machine free_ports(profile, compile(setup + R"(
        tma.load.4d [ur0], [4096, ur8], [ur1]
        mov  r1, 7
        umov ur6, 9
    )" + wait_on("ur1", "loaded") + "halt\n"), tensor_launch(1, 2, 1, 64), {.timeline = true});
    free_ports.run(20000);
    const auto tma = issues_named(free_ports, "tma.load.4d");
    require(issues_named(free_ports, "mov") == std::vector<uint64_t>{tma.at(0) + 1}, "the vector mov waited");
    require(issues_named(free_ports, "umov").back() == tma.at(0) + 2, "the uniform write that reads nothing waited");
    require(free_ports.trace().summary().urf_read_ports_held == 0, "nothing was refused");
}

// The operand registers may be rewritten by the very next instruction: a
// write issued at T + 1 lands at T + 1 + L + Wu, after the last round's
// sample. The first command uses the old buffer, barrier and row; the second
// the new ones.
void test_operand_registers_rewritten_on_the_next_instruction() {
    nanosm::Machine machine(canonical_profile(), compile(R"(
        umov ur0, 0
        umov ur1, 4352
        umov ur8, 0
        umov ur9, 1
        tma.load.2d [ur0], [4160, ur8], [ur1]
        umov ur9, 5
        umov ur0, 256
        umov ur1, 4360
        tma.load.2d [ur0], [4160, ur8], [ur1]
        umov ur2, 4352
    )" + wait_on("ur2", "first") + wait_on("ur1", "second") + "halt\n"), tensor_launch(1, 2, 2, 64),
        {.timeline = true});
    machine.run(20000);
    const auto tma = issues_named(machine, "tma.load.2d");
    const auto writes = issues_named(machine, "umov");
    require(tma.size() == 2 && std::find(writes.begin(), writes.end(), tma[0] + 1) != writes.end(),
        "the row coordinate was not rewritten the cycle after the tma");
    require_rows(machine, 0, 1, 2, "the first command (rows 1, 2)");
    require_rows(machine, 256, 5, 2, "the second command (rows 5, 6)");
    require(machine.read_mbar_state(4352).phase && machine.read_mbar_state(4360).phase, "both barriers completed");
}

// The buffer's low 6 bits and the barrier's low 3 bits are not wired: a
// program that sets them is the sanitizer's PANIC.
void test_unwired_low_bits_panic() {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    for (const auto& [buffer, mbar, needle] : {
             std::tuple{4u, 4352u, std::string("tma buffer address 4 is not 64-byte aligned")},
             std::tuple{0u, 4356u, std::string("tma barrier address 4356 is not 8-byte aligned")},
         }) {
        nanosm::Machine machine(canonical_profile(), compile(tma_operands_asm(0, buffer, 1, mbar, 8, {0, 0}) +
            "tma.load.2d [ur0], [4160, ur8], [ur1]\n" + wait_on("ur1", "loaded") + "halt\n"),
            tensor_launch(1, 2, 1, 64));
        expect_sim_panic(machine, needle, needle);
    }
}

std::string compile_error(const std::string& source) {
    try {
        (void)compile(source);
    } catch (const nanosm::SimError& error) {
        return error.what();
    }
    return {};
}

// The descriptor round-trips through its 64 bytes; reserved element sizes,
// reserved bytes and a layout byte past swizzle64 are refused. The assembler's
// forms print and parse back;
// it refuses an unaligned descriptor, coordinates past the last uniform
// register and the old form without a coordinate count.
void test_descriptor_and_toolchain() {
    nanosm::TmaDescriptor descriptor = sequence_tiles(4);
    descriptor.base = 0x123456780ull;
    descriptor.element_bytes = 2;
    descriptor.layout = nanosm::TileLayout::swizzle64;
    descriptor.transposed = true;
    descriptor.tile_length = 64;
    descriptor.tile_rows = 16;
    descriptor.dimensions = {64, 3, 100, 2};
    descriptor.strides = {64, 192, 19200};
    auto bytes = nanosm::encode_tma_descriptor(descriptor);
    const auto decoded = nanosm::decode_tma_descriptor(bytes.data(), bytes.size());
    require(nanosm::encode_tma_descriptor(decoded) == bytes && decoded.base == descriptor.base &&
            decoded.dimensions == descriptor.dimensions && decoded.strides == descriptor.strides &&
            decoded.tile_length == 64 && decoded.tile_rows == 16 && decoded.row_dimension == 2 &&
            decoded.layout == nanosm::TileLayout::swizzle64 && decoded.transposed,
        "the descriptor does not round-trip");
    const auto refused = [](const std::function<void()>& action, const std::string& needle, const std::string& what) {
        try {
            action();
        } catch (const nanosm::SimError& error) {
            require(std::string(error.what()).find(needle) != std::string::npos,
                what + ": wrong error: " + error.what());
            return;
        }
        throw std::runtime_error(what + " was accepted");
    };
    for (const uint8_t reserved_size : {8, 16}) {
        auto wide = rows_of_8(1);
        wide.element_bytes = reserved_size;
        refused([&] { (void)nanosm::encode_tma_descriptor(wide); }, "is reserved", "element size 8 / 16");
    }
    bytes[50] = 1;
    refused([&] { (void)nanosm::decode_tma_descriptor(bytes.data(), bytes.size()); }, "reserved bytes",
        "a nonzero reserved byte");
    bytes[50] = 0;
    bytes[nanosm::tma_encoding::layout] = 2; // the old "B operand" role byte
    refused([&] { (void)nanosm::decode_tma_descriptor(bytes.data(), bytes.size()); }, "layout must be",
        "layout byte 2");

    const auto kernel = compile("tma.store.3d [4096, ur4], [ur1], [ur2]\ntma.load.1d [ur0], [512, ur15], [ur3]\nhalt\n");
    const auto store = nanosm::decode(kernel.words.at(0));
    const auto load = nanosm::decode(kernel.words.at(1));
    require(store.opcode == nanosm::Opcode::tma_store_3d && load.opcode == nanosm::Opcode::tma_load_1d,
        "the opcodes");
    require(nanosm::format_instruction(store, false) == "tma.store.3d [4096, ur4], [ur1], [ur2]" &&
            nanosm::format_instruction(load, false) == "tma.load.1d [ur0], [512, ur15], [ur3]",
        "the printed forms: " + nanosm::format_instruction(store, false));
    require(compile_error("tma.load.1d [ur0], [4100, ur4], [ur1]\nhalt\n").find("must be 64-byte aligned") !=
            std::string::npos, "an unaligned descriptor address");
    // Four coordinates from three below the profile's last uniform register
    // pass it by one; one coordinate in the last register fits.
    const auto registers = canonical_profile().urf_registers;
    const auto from = "ur" + std::to_string(registers - 3);
    require(compile_error("tma.load.4d [ur0], [4096, " + from + "], [ur1]\nhalt\n")
                    .find("pass this profile's last uniform register") != std::string::npos,
        "coordinates " + from + " .. ur" + std::to_string(registers));
    require(compile_error("tma.load.1d [ur0], [4096, ur" + std::to_string(registers - 1) + "], [ur1]\nhalt\n").empty(),
        "one coordinate in the last uniform register");
    require(compile_error("tma.load [4096], r0, r1\nhalt\n").find("needs its coordinate count") != std::string::npos,
        "the old form");
}

} // namespace

TEST(TmaOperands, four_coordinate_counts) {
    test_four_coordinate_counts();
}

TEST(TmaOperands, tile_from_a_four_dimensional_tensor) {
    test_tile_from_a_four_dimensional_tensor();
}

TEST(TmaOperands, back_to_back_commands_from_two_warps) {
    test_back_to_back_commands_from_two_warps();
}

TEST(TmaOperands, uniform_reader_refused_during_the_rounds) {
    test_uniform_reader_refused_during_the_rounds();
}

TEST(TmaOperands, operand_registers_rewritten_on_the_next_instruction) {
    test_operand_registers_rewritten_on_the_next_instruction();
}

TEST(TmaOperands, unwired_low_bits_panic) {
    test_unwired_low_bits_panic();
}

TEST(TmaOperands, descriptor_and_toolchain) {
    test_descriptor_and_toolchain();
}

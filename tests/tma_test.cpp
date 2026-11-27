// The TMA's commands: swizzle64 placement and its census against the MMA's
// operand addressing, the limits of swizzle64 tiles, out-of-range handling,
// the walker's runs, the sector accounting and the store's sector registers.
// The descriptor (64 bytes in shared memory, src/tma_descriptor.hpp) holds
// the global tensor, the tile's shape and its shared-memory layout (dense, or
// swizzle64 and whether it is transposed; the rules are src/layout.hpp's,
// which the MMA reads its operands with). The instruction holds the
// descriptor's shared address as a constant and everything else in uniform
// registers: the buffer's shared address, the tile's coordinates (one
// register per dimension, from urC on) and the completion barrier's shared
// address:
//     tma.load.Nd  [urS], [DESC, urC], [urM]
//     tma.store.Nd [DESC, urC], [urS], [urM]
#include "test_support.hpp"

namespace {

using namespace nanosm_test;

// The barrier of most tests here (the descriptor is at shmem 0).
constexpr uint32_t tma_mbar = 128;

constexpr auto swizzle64 = nanosm::TileLayout::swizzle64;
constexpr auto dense = nanosm::TileLayout::dense;

// A tile of `length` x `rows` elements of a two-dimensional tensor `dim_x` x
// `dim_y` with row stride `stride` (elements), at gmem `base`.
nanosm::TmaDescriptor tile_2d(uint32_t dim_x, uint32_t dim_y, uint32_t stride, uint32_t length, uint32_t rows,
    uint8_t element_bytes = 4, nanosm::TileLayout layout = dense, bool transposed = false, uint64_t base = 0) {
    nanosm::TmaDescriptor descriptor;
    descriptor.base = base;
    descriptor.element_bytes = element_bytes;
    descriptor.layout = layout;
    descriptor.transposed = transposed;
    descriptor.dimensions = {dim_x, dim_y, 1, 1};
    descriptor.strides = {stride, 0, 0};
    descriptor.tile_length = static_cast<uint16_t>(length);
    descriptor.tile_rows = static_cast<uint16_t>(rows);
    return descriptor;
}

uint32_t tile_bytes(const nanosm::TmaDescriptor& descriptor) {
    return descriptor.element_bytes * descriptor.tile_length * descriptor.tile_rows;
}

using TmaFill = std::function<void(std::vector<uint8_t>& shmem, std::vector<uint8_t>& gmem)>;

// One warp; `shmem_bytes` of shared and `gmem_bytes` of global memory, zero
// until `fill` writes them; then `descriptor` at shmem 0 and at `mbar` the
// barrier expecting `tx_bytes`.
nanosm::Launch tma_launch(const nanosm::TmaDescriptor& descriptor, uint32_t tx_bytes, uint32_t shmem_bytes,
    uint32_t gmem_bytes, const TmaFill& fill = {}, uint32_t mbar = tma_mbar) {
    auto launch = launch_warps(1);
    auto& shmem = image_bytes(launch.test_shmem, shmem_bytes);
    auto& gmem = image_bytes(launch.gmem_image, gmem_bytes);
    if (fill) fill(shmem, gmem);
    write_tma_descriptor(shmem, 0, descriptor);
    write_mbar(shmem, mbar, mbar_state(1, 0, static_cast<int32_t>(tx_bytes)));
    return launch;
}

// The command on the descriptor at 0: the buffer in ur4, the barrier in ur3,
// the coordinates from ur8 (one per dimension).
std::string tma_command(bool store, std::size_t rank) {
    const auto shape = std::to_string(rank) + "d";
    return store ? "tma.store." + shape + " [0, ur8], [ur4], [ur3]\n" : "tma.load." + shape + " [ur4], [0, ur8], [ur3]\n";
}

// One command with its operands (the barrier at tma_mbar), the wait on its
// barrier, halt.
std::string one_tma(bool store, uint32_t buffer, const std::vector<int32_t>& coordinates) {
    return tma_operands_asm(4, buffer, 3, tma_mbar, 8, coordinates) + tma_command(store, coordinates.size()) +
        wait_on("ur3") + "halt\n";
}

// That kernel over tma_launch's images, the barrier expecting the tile's
// bytes; not yet run.
std::unique_ptr<nanosm::Machine> tma_machine(const nanosm::TmaDescriptor& descriptor, bool store, uint32_t buffer,
    const std::vector<int32_t>& coordinates, uint32_t shmem_bytes, uint32_t gmem_bytes, const TmaFill& fill = {},
    const nanosm::DeviceProfile& profile = canonical_profile()) {
    return std::make_unique<nanosm::Machine>(profile, compile(one_tma(store, buffer, coordinates), profile),
        tma_launch(descriptor, tile_bytes(descriptor), shmem_bytes, gmem_bytes, fill));
}

void require_completed(const nanosm::Machine& machine, const std::string& what) {
    const auto state = machine.read_mbar_state(tma_mbar);
    require(state.phase && state.tx_bytes == 0, what + ": the barrier did not complete with the tile's bytes");
}

// The pins of one run's TMA counters; an unset one is not checked.
struct TmaCounters {
    std::optional<uint64_t> commands, completions, logical, useful, sector, requests, rounds;
};

void require_counters(const nanosm::TraceSummary& summary, const TmaCounters& expected, const std::string& what) {
    const auto pin = [&](const std::optional<uint64_t>& value, uint64_t actual, const char* name) {
        if (value) require_eq(actual, *value, what + ": " + name);
    };
    pin(expected.commands, summary.tma_commands, "commands");
    pin(expected.completions, summary.tma_command_completions, "completions");
    pin(expected.logical, summary.tma_logical_bytes, "logical bytes");
    pin(expected.useful, summary.tma_useful_bytes, "useful bytes");
    pin(expected.sector, summary.tma_sector_bytes, "sector bytes");
    pin(expected.requests, summary.tma_port_requests, "port requests");
    pin(expected.rounds, summary.tma_port_rounds, "port rounds");
}

uint32_t element_at(const std::vector<uint8_t>& memory, uint32_t address, uint32_t bytes) {
    uint32_t value = 0;
    std::memcpy(&value, memory.data() + address, bytes);
    return value;
}

void put_element(std::vector<uint8_t>& memory, uint32_t address, uint32_t bytes, uint32_t value) {
    std::memcpy(memory.data() + address, &value, bytes);
}

// The shared address of tile element (x, y) (x along dimension 0, y the tile
// row) in a tile at `buffer`, by the rule written out (reference_swizzle64):
// a swizzle64 tile stores row y, or transposed position y of stored row x.
uint32_t tile_address(const nanosm::TmaDescriptor& descriptor, uint32_t buffer, uint32_t x, uint32_t y) {
    const uint32_t bytes = descriptor.element_bytes;
    if (descriptor.layout == dense) return buffer + bytes * (descriptor.tile_length * y + x);
    const uint32_t spacing = bytes * (descriptor.transposed ? descriptor.tile_rows : descriptor.tile_length);
    return descriptor.transposed ? reference_swizzle64(buffer, spacing, x, bytes * y)
                                 : reference_swizzle64(buffer, spacing, y, bytes * x);
}

// A tile of a two-dimensional tensor (fp16 or fp32; a one-dimensional one
// has one coordinate) at `origin`, its buffer at `buffer`, and the pins of
// the load's and the store's counters.
struct TileCase {
    std::string what;
    nanosm::TmaDescriptor descriptor;
    std::vector<int32_t> origin;
    uint32_t buffer = 1024;
    TmaCounters load{};
    TmaCounters store{};
};

// A load of the tile places every element where the layout rule puts it,
// zero outside the tensor, and writes nothing past the tile; a store writes
// exactly the tile's elements inside the tensor and leaves every other gmem
// byte alone. Both complete their barrier.
void check_tile_case(const TileCase& c) {
    const auto& d = c.descriptor;
    const uint32_t bytes = d.element_bytes;
    const uint32_t stride = d.dimensions[1] > 1 ? d.strides[0] : d.dimensions[0];
    const uint32_t base = static_cast<uint32_t>(d.base);
    const uint32_t gmem_bytes = base + bytes * (stride * (d.dimensions[1] - 1) + d.dimensions[0]) + 64;
    const uint32_t shmem_bytes = c.buffer + tile_bytes(d) + 64;
    const int64_t origin_x = c.origin[0];
    const int64_t origin_y = c.origin.size() > 1 ? c.origin[1] : 0;
    // Tensor element i holds value(i), the store's tile element k tag(k):
    // distinct, nonzero, and never 0x5a5a.
    const uint32_t high = bytes == 2 ? 0x8000u : 0x80000000u;
    const auto value = [&](uint32_t i) { return high | i; };
    const auto tag = [&](uint32_t k) { return (high >> 1) | k; };
    // Tile element (x, y)'s tensor index, if it is inside the tensor.
    const auto index_of = [&](uint32_t x, uint32_t y) -> std::optional<uint32_t> {
        const int64_t gx = origin_x + x;
        const int64_t gy = origin_y + y;
        if (gx < 0 || gy < 0 || gx >= d.dimensions[0] || gy >= d.dimensions[1]) return std::nullopt;
        return static_cast<uint32_t>(gy * stride + gx);
    };
    const auto at = [](uint32_t x, uint32_t y) { return " element (" + std::to_string(x) + ", " + std::to_string(y) + ")"; };
    {
        const auto what = c.what + ", load";
        auto machine = tma_machine(d, false, c.buffer, c.origin, shmem_bytes, gmem_bytes, [&](auto& shmem, auto& gmem) {
            std::fill(shmem.begin() + c.buffer, shmem.end(), 0xa5);
            for (uint32_t y = 0; y < d.dimensions[1]; ++y)
                for (uint32_t x = 0; x < d.dimensions[0]; ++x)
                    put_element(gmem, base + bytes * (y * stride + x), bytes, value(y * stride + x));
        });
        machine->run(200000);
        const auto shared = machine->shmem();
        for (uint32_t y = 0; y < d.tile_rows; ++y) for (uint32_t x = 0; x < d.tile_length; ++x) {
            const auto index = index_of(x, y);
            require(element_at(shared, tile_address(d, c.buffer, x, y), bytes) == (index ? value(*index) : 0u),
                what + ":" + at(x, y));
        }
        for (uint32_t address = c.buffer + tile_bytes(d); address < shmem_bytes; ++address)
            require(shared[address] == 0xa5, what + ": wrote past its tile");
        require_completed(*machine, what);
        require_counters(machine->trace().summary(), c.load, what);
    }
    {
        const auto what = c.what + ", store";
        std::vector<uint8_t> expected;
        auto machine = tma_machine(d, true, c.buffer, c.origin, shmem_bytes, gmem_bytes, [&](auto& shmem, auto& gmem) {
            std::fill(gmem.begin(), gmem.end(), 0x5a);
            expected = gmem;
            for (uint32_t y = 0; y < d.tile_rows; ++y) for (uint32_t x = 0; x < d.tile_length; ++x) {
                const uint32_t k = d.tile_length * y + x;
                put_element(shmem, tile_address(d, c.buffer, x, y), bytes, tag(k));
                if (const auto index = index_of(x, y)) put_element(expected, base + bytes * *index, bytes, tag(k));
            }
        });
        machine->run(200000);
        for (uint32_t address = 0; address < gmem_bytes; ++address)
            require(machine->gmem()[address] == expected[address], what + ": gmem byte " + std::to_string(address));
        require_completed(*machine, what);
        require_counters(machine->trace().summary(), c.store, what);
    }
}

// `commands` loads of 8192 / commands bytes each, one descriptor, the tile's
// coordinates in ur8, ur9 (set by the caller, moved on by `step` between
// commands): command i's buffer (ur4) starts at tile + (8192 / commands) x
// i. Every command completes into the mbar in ur3, which the wait reads.
std::string tile_piece_loads(uint32_t commands, uint32_t tile, uint32_t mbar, const std::string& step) {
    std::string source = "umov ur3, " + std::to_string(mbar) + "\n";
    for (uint32_t i = 0; i < commands; ++i) {
        if (i != 0) source += step;
        source += "umov ur4, " + std::to_string(tile + 8192 / commands * i) + "\n" + tma_command(false, 2);
    }
    return source + wait_on("ur3") + "halt\n";
}

// The cycle a command's completion flips the mbar at `mbar` (Q, between ticks).
uint64_t tick_until_phase(nanosm::Machine& machine, uint32_t mbar) {
    while (!machine.read_mbar_state(mbar).phase) {
        require(!machine.done() && machine.cycle() < 50000, "the tma command never completed its barrier");
        machine.tick();
    }
    return machine.cycle();
}

// A plain TMA store of `sectors` sectors (32 B each) under `config`: the
// cycles its source rounds' words returned (edge probe tma_store_round), in
// order, and the run's length. The warp halts instead of polling the mbar: a
// try_wait probe would share A-read's round-robin with the store reads
// (legal contention, not what is measured).
std::pair<std::vector<uint64_t>, uint64_t> tma_store_returns(const nanosm::DeviceProfile& config, uint32_t sectors) {
    const uint32_t bytes = 32 * sectors;
    auto launch = launch_warps(1);
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

} // namespace

// Swizzle64 tiles loaded and stored, every element where the rule written
// out puts it: fp16 16-byte stored rows, not transposed and transposed, at
// (3, 2) of a tensor of row stride 83; and a 64 x 64 fp32 tile of 256-byte
// rows, the whole tensor, at gmem 512.
TEST(Tma, tma_swizzle64_transposed_load_store) {
    const std::vector<TileCase> cases = {
        {"fp16, 16-byte stored rows", tile_2d(16, 69, 83, 8, 64, 2, swizzle64), {3, 2}},
        {"fp16, 16-byte stored rows, transposed", tile_2d(72, 13, 83, 64, 8, 2, swizzle64, true), {3, 2}},
        {"fp32, 256-byte stored rows", tile_2d(64, 64, 64, 64, 64, 4, swizzle64, false, 512), {0, 0}},
    };
    for (const auto& c : cases) check_tile_case(c);
}

// Out of range: a load zero-fills what lies outside the tensor (one fill per
// row-side band, at most a sector each), a store leaves it alone, and the
// sectors a command touches are whole ones.
TEST(Tma, tma_out_of_range_bands) {
    // A tile of 8 at -4 of a tensor of 4: half outside.
    auto rank1 = plain_tma_1d(4);
    rank1.tile_length = 8;
    const std::vector<TileCase> cases = {
        {"a tile of 8 at -4 of a tensor of 4", rank1, {-4}, 256,
            {.commands = 1, .completions = 1, .logical = 32, .useful = 16, .sector = 32}},
        // 16 bytes in the middle of one sector: the sector's other bytes stay.
        {"4 fp32 at gmem 12", plain_tma_1d(4, 12), {0}, 256, {}, {.logical = 16, .useful = 16, .sector = 32}},
        // 16 x 10 at (-4, -2) of 8 x 6: four columns out on each side, two
        // rows above and below: 4 rows x 2 + 6 rows x 2 = 20 fills, 6 sectors.
        {"bands on every side", tile_2d(8, 6, 8, 16, 10), {-4, -2}, 1024,
            {.useful = 192, .sector = 192, .requests = 26}},
        // 64 stored rows of 16 positions (32-byte rows) at (-8, -2) of 40 x 20.
        {"transposed swizzle64, 32-byte stored rows", tile_2d(40, 20, 40, 64, 16, 2, swizzle64, true), {-8, -2}, 2048},
    };
    for (const auto& c : cases) check_tile_case(c);
}

// A store whose tile is entirely outside its tensor (its rows, its columns,
// or a dimension it does not walk) writes nothing and completes its barrier
// with the tile's bytes; the walker skips one row per cycle, so a tile of 33
// rows completes exactly 32 cycles after a tile of 1.
TEST(Tma, tma_store_entirely_out_of_range) {
    constexpr uint32_t tile_address = 1024;
    struct Case { std::string what; std::vector<int32_t> coordinates; };
    for (const auto& c : {Case{"rows", {0, 100, 0}}, Case{"columns", {-64, 0, 0}}, Case{"third dimension", {0, 0, 5}}}) {
        std::vector<uint64_t> completion;
        for (const uint32_t rows : {1u, 33u}) {
            nanosm::TmaDescriptor descriptor;
            descriptor.dimensions = {16, 40, 3, 1};
            descriptor.strides = {16, 640, 0};
            descriptor.tile_length = 16;
            descriptor.tile_rows = static_cast<uint16_t>(rows);
            auto machine = tma_machine(descriptor, true, tile_address, c.coordinates, tile_address + 64 * 33, 8192,
                [&](auto& shmem, auto& gmem) {
                    std::fill(gmem.begin(), gmem.end(), 0x5a);
                    for (uint32_t i = 0; i < 16 * rows; ++i) write_u32(shmem, tile_address + 4 * i, 0x1000 + i);
                });
            completion.push_back(tick_until_phase(*machine, tma_mbar));
            machine->run(50000);
            require(std::all_of(machine->gmem().begin(), machine->gmem().begin() + 8192, [](uint8_t b) { return b == 0x5a; }),
                "an out-of-range store (" + c.what + ") wrote gmem");
            require_completed(*machine, "an out-of-range store (" + c.what + ")");
            const auto& summary = machine->trace().summary();
            require(summary.tma_command_completions == 1 && summary.tma_useful_bytes == 0 &&
                    summary.tma_sector_bytes == 0 && summary.tma_port_requests == 0,
                "an out-of-range store (" + c.what + ") moved data");
        }
        require_eq(completion[1], completion[0] + 32, "the completion of an out-of-range store (" + c.what +
            ") of 33 rows: 32 cycles after one of 1 row (one skip per row)");
    }
}

// The TMA's sector and port accounting for whole tiles. A 64 x 64 fp16 tile
// of 128-byte rows reads each sector once; cut into eight tiles of 16-byte
// rows each 32-byte sector is fetched twice, half used each time. A 32-byte
// sector is eight fp32 of one row (two 16-byte pieces the shuffle keeps in
// different slots) or, transposed, sixteen fp16 of neighbouring stored rows
// (spread over sixteen banks): one port-A round per sector either way, as
// dense placement.
TEST(TmaSlow, tma_swizzle64_sector_economics) {
    struct Row {
        std::string what;
        nanosm::TmaDescriptor descriptor;
        bool store = false;
        uint32_t commands = 1;
        TmaCounters counters;
    };
    const std::vector<Row> rows = {
        {"fp16 64 x 64, one load", tile_2d(64, 64, 64, 64, 64, 2, swizzle64), false, 1,
            {.completions = 1, .useful = 8192, .sector = 8192}},
        {"fp16 64 x 64 in eight loads of 16-byte rows", tile_2d(64, 64, 64, 8, 64, 2, swizzle64), false, 8,
            {.completions = 8, .useful = 8192, .sector = 16384}},
        {"fp32 swizzle64 store", tile_2d(64, 64, 64, 64, 64, 4, swizzle64, false, 512), true, 1,
            {.completions = 1, .useful = 16384, .sector = 16384, .requests = 512, .rounds = 512}},
        {"fp32 dense store", tile_2d(64, 64, 64, 64, 64, 4, dense, false, 512), true, 1, {.requests = 512, .rounds = 512}},
        {"fp32 swizzle64 load", tile_2d(64, 64, 64, 64, 64, 4, swizzle64, false, 512), false, 1,
            {.requests = 512, .rounds = 512}},
        {"fp16 transposed, 128-byte stored rows", tile_2d(64, 64, 64, 64, 64, 2, swizzle64, true), false, 1,
            {.requests = 256, .rounds = 256}},
    };
    constexpr uint32_t buffer = 1024;
    for (const auto& row : rows) {
        const auto& d = row.descriptor;
        const uint32_t bytes = row.commands * tile_bytes(d);
        const uint32_t tensor = d.element_bytes * d.dimensions[0] * d.dimensions[1];
        const auto program = row.commands == 1
            ? one_tma(row.store, buffer, {0, 0})
            : "umov ur8, 0\numov ur9, 0\n" + tile_piece_loads(row.commands, buffer, tma_mbar, "uaddi ur8, ur8, 8\n");
        nanosm::Machine machine(canonical_profile(), compile(program),
            tma_launch(d, bytes, buffer + bytes, static_cast<uint32_t>(d.base) + tensor, [&](auto& shmem, auto& gmem) {
                fill_pattern(shmem, buffer, bytes / 4, 0);
                fill_pattern(gmem, static_cast<uint32_t>(d.base), tensor / 4, 0);
            }));
        machine.run(200000);
        require_counters(machine.trace().summary(), row.counters, row.what);
    }
}

// The census, for whole rows: for stored-row counts R (4 .. 128) and fp16 K
// extents 8, 16, 32, 64 (rows of 16, 32, 64, 128 bytes), a TMA load of a
// swizzle64 R x K tile writes exactly what an MMA with m = R (as A) or n = R
// (as B), k = K and row spacing 2K reads: every element where the MMA's own
// operand addressing (nanosm::mma_operand_address) looks for it, and no byte
// outside the tile's R x K x 2 bytes. A's global is m rows of K; B's is K
// rows of n (transposed). The buffer is a multiple of every spacing here
// (128) but not of 16 rows of any of them, so the tile's row 0 is not row 0
// of the shuffle.
TEST(TmaSlow, tma_swizzle64_census_against_mma) {
    const auto& profile = canonical_profile();
    constexpr uint32_t base = 1152;
    constexpr uint32_t mbar_address = 64;
    constexpr uint32_t shmem_bytes = base + 128 * 64 * 2 + 1024;
    constexpr uint8_t poison = 0xa5;
    const auto tag = [](uint32_t outer, uint32_t k) { return static_cast<uint16_t>(0x8000u | outer << 6 | k); };
    const auto kernel = compile(tma_operands_asm(4, base, 3, mbar_address, 8, {0, 0}) + tma_command(false, 2) +
        wait_on("ur3") + "halt\n");
    for (const bool a_role : {true, false}) {
        for (uint32_t rows = 4; rows <= 128; rows *= 2) {
            for (const uint32_t k_extent : {8u, 16u, 32u, 64u}) {
                const auto what = std::string(a_role ? "A" : "B") + ", " + std::to_string(rows) + " rows, K " +
                    std::to_string(k_extent);
                // Row-major global, the tile the whole tensor: A is rows x K
                // (K along the row), B is K x n (K across the rows).
                const uint32_t length = a_role ? k_extent : rows;
                const uint32_t height = a_role ? rows : k_extent;
                const auto descriptor = tile_2d(length, height, length, length, height, 2, swizzle64, !a_role);
                nanosm::Machine machine(profile, kernel,
                    tma_launch(descriptor, rows * k_extent * 2, shmem_bytes, rows * k_extent * 2,
                        [&](auto& shmem, auto& gmem) {
                            std::fill(shmem.begin(), shmem.end(), poison);
                            std::fill(shmem.begin(), shmem.begin() + 128, 0);
                            for (uint32_t outer = 0; outer < rows; ++outer)
                                for (uint32_t k = 0; k < k_extent; ++k)
                                    write_u16(gmem, 2 * (a_role ? outer * k_extent + k : k * rows + outer), tag(outer, k));
                        },
                        mbar_address));
                machine.run(100000);
                const nanosm::MmaShape shape = a_role ? nanosm::MmaShape{rows, 16, k_extent}
                                                      : nanosm::MmaShape{16, rows, k_extent};
                const auto bytes = machine.shmem();
                std::set<uint32_t> addresses;
                for (uint32_t outer = 0; outer < rows; ++outer) {
                    for (uint32_t k = 0; k < k_extent; ++k) {
                        const auto address =
                            nanosm::mma_operand_address(profile, shape, base, 2 * k_extent, a_role, outer, k);
                        require(address >= base && address + 2 <= base + rows * k_extent * 2,
                            what + ": the MMA reads outside the tile's compact object");
                        require(addresses.insert(address).second, what + ": the MMA reads one address twice");
                        uint16_t value = 0;
                        std::memcpy(&value, bytes.data() + address, sizeof(value));
                        require(value == tag(outer, k), what + ": the MMA's element (" + std::to_string(outer) + ", " +
                            std::to_string(k) + ") is not where the TMA put it");
                    }
                }
                for (uint32_t address = 128; address < shmem_bytes; ++address) {
                    if (address >= base && address < base + rows * k_extent * 2) continue;
                    require(bytes[address] == poison, what + ": the TMA wrote byte " + std::to_string(address) +
                        " outside the object");
                }
            }
        }
    }
}

// A swizzle64 tile's stored row (the tile length, or the rows when
// transposed, times the element bytes) is a power of two bytes, 16 or more;
// it may have any number of stored rows (the other extent), 1 included; its
// elements are 2 or 4 bytes. A dense tile cannot be transposed. A tile whose
// packed span ends past shmem is the machine's PANIC.
TEST(Tma, tma_swizzle64_validation) {
    struct Row {
        uint8_t element_bytes;
        bool transposed;
        uint32_t length;
        uint32_t rows;
        const char* refusal; // nullptr: accepted; "": refused, any message
    };
    std::vector<Row> rows;
    for (const uint32_t elements : {1u, 4u, 8u, 12u, 16u, 24u, 32u, 40u, 64u, 96u, 128u, 255u, 256u}) {
        const uint32_t bytes = 2 * elements;
        const char* refusal = bytes >= 16 && (bytes & (bytes - 1)) == 0 ? nullptr : "stored row must be";
        rows.push_back({2, false, elements, 64, refusal});
        rows.push_back({2, true, 64, elements, refusal});
    }
    for (const uint32_t stored_rows : {1u, 3u, 5u, 17u, 200u}) {
        rows.push_back({2, false, 8, stored_rows, nullptr});
        rows.push_back({2, true, stored_rows, 16, nullptr});
    }
    // fp32: 4 (16 B), 8 (32 B), 16, 32, 64 ... elements per stored row.
    for (const Row row : {Row{4, false, 64, 64, nullptr}, Row{4, true, 64, 64, nullptr}, Row{4, false, 4, 64, nullptr},
             Row{4, false, 8, 64, nullptr}, Row{4, false, 128, 64, nullptr}, Row{4, false, 64, 48, nullptr},
             Row{4, true, 32, 8, nullptr}, Row{4, false, 64, 1, nullptr}, Row{8, false, 64, 64, ""},
             Row{1, false, 64, 64, "2- or 4-byte elements"}})
        rows.push_back(row);
    for (const uint32_t k : {2u, 6u, 12u, 20u, 48u}) {
        rows.push_back({4, false, k, 64, "stored row must be"});
        rows.push_back({4, true, 64, k, "stored row must be"});
    }
    for (const auto& row : rows) {
        const auto descriptor = tile_2d(128, 128, 128, row.length, row.rows, row.element_bytes, swizzle64, row.transposed);
        const auto what = "a swizzle64 tile of " + std::to_string(row.length) + " x " + std::to_string(row.rows) + " " +
            std::to_string(row.element_bytes) + "-byte elements" + (row.transposed ? ", transposed" : "");
        const auto encode = [&] { (void)nanosm::encode_tma_descriptor(descriptor); };
        if (row.refusal == nullptr) {
            const auto refused = [&] {
                try {
                    encode();
                } catch (const nanosm::SimError& error) {
                    return std::string(error.what());
                }
                return std::string();
            }();
            require(refused.empty(), what + " was refused: " + refused);
        } else {
            expect_error<nanosm::SimError>(encode, {row.refusal}, what);
        }
    }
    auto transposed_dense = tile_2d(128, 128, 128, 16, 64, 2);
    transposed_dense.transposed = true;
    expect_error<nanosm::SimError>([&] { (void)nanosm::encode_tma_descriptor(transposed_dense); },
        {"dense tiles cannot be transposed"}, "a transposed dense tile");

    // The span is the packed tile, 64 x 64 x 2 = 8192 bytes: at 61440 it
    // ends past the 64 KiB shmem.
    auto oversized = tma_machine(tile_2d(128, 128, 128, 64, 64, 2, swizzle64), false, 61440, {0, 0}, 4096, 65536);
    expect_panic(*oversized, {"does not fit in shmem"}, "an oversized swizzle64 span");
}

// The shuffle's row number is the address's, so pieces of a tile land
// anywhere: 16 loads of 4 stored rows, or 64 loads of ONE stored row, each
// landed right after the last, build exactly the 64-row tile one load makes:
// fp16, 128-byte stored rows, not transposed (the pieces step along the
// global rows) and transposed (they step along dimension 0, whose positions
// are the stored rows). The tile is at 0x4080, so its row 0 is row 1 of the
// shuffle (the one-row gather case): its first logical byte is stored at
// 0x4090 and its logical byte 16 at 0x4080, which a reader or writer still
// counting rows from the tile would miss.
TEST(TmaSlow, tma_swizzle64_pieces_land_anywhere) {
    constexpr uint32_t tile_address = 0x4080;
    constexpr uint32_t stride = 83;
    constexpr uint32_t dim_x = 80;
    constexpr uint32_t dim_y = 70;
    // Element (x, y) of the tile is global (3 + x, 2 + y).
    const auto value = [](uint32_t x, uint32_t y) { return static_cast<uint16_t>(0x100 + 3 * (2 + y) + 5 * (3 + x)); };
    for (const bool transposed : {false, true}) {
        const auto run_span = [&](uint32_t commands) {
            const uint32_t stored_rows = 64 / commands;
            const auto descriptor = tile_2d(dim_x, dim_y, stride, transposed ? stored_rows : 64,
                transposed ? 64 : stored_rows, 2, swizzle64, transposed);
            const std::string step = "uaddi ur" + std::string(transposed ? "8" : "9") + ", ur" +
                (transposed ? "8" : "9") + ", " + std::to_string(stored_rows) + "\n";
            nanosm::Machine machine(canonical_profile(),
                compile("umov ur8, 3\numov ur9, 2\n" + tile_piece_loads(commands, tile_address, tma_mbar, step)),
                tma_launch(descriptor, 8192, tile_address + 8192, 2 * stride * dim_y + 64, [&](auto&, auto& gmem) {
                    for (uint32_t y = 0; y < dim_y; ++y)
                        for (uint32_t x = 0; x < dim_x; ++x)
                            write_u16(gmem, 2 * (y * stride + x), static_cast<uint16_t>(0x100 + 3 * y + 5 * x));
                }));
            machine.run(200000);
            const auto shared = machine.shmem();
            return std::vector<uint8_t>(shared.begin() + tile_address, shared.begin() + tile_address + 8192);
        };
        const auto whole = run_span(1);
        require(whole == run_span(16), "one 64-row swizzle64 load disagreed with sixteen 4-row loads");
        require(whole == run_span(64), "one 64-row swizzle64 load disagreed with 64 one-row loads");
        const auto halfword = [&](uint32_t address) { return element_at(whole, address - tile_address, 2); };
        // Stored row 0's logical bytes 0 and 16, at their physical addresses
        // written out by hand: element (0, 0), and (8, 0) or transposed (0, 8).
        require(halfword(0x4090) == value(0, 0) && halfword(0x4080) == (transposed ? value(0, 8) : value(8, 0)),
            "stored row 0's logical bytes 0 and 16 are not at 0x4090 and 0x4080");
        for (uint32_t y = 0; y < 64; ++y) {
            for (uint32_t x = 0; x < 64; ++x) {
                const uint32_t position = transposed ? y : x;
                const uint32_t row = transposed ? x : y;
                require(halfword(reference_swizzle64(tile_address, 128, row, 2 * position)) == value(x, y),
                    "swizzle64 load mapped element (" + std::to_string(x) + ", " + std::to_string(y) + ") wrong" +
                        (transposed ? ", transposed" : ""));
            }
        }
    }
}

// The layout function (src/layout.hpp) against the rule written out
// (reference_swizzle64) for spacings 16 .. 512, at base 0 and at a legal base
// 17 x max(spacing, 64) that is not a multiple of 16 rows: the same address
// for every byte, every row keeps exactly its own bytes [base + row x
// spacing, base + (row + 1) x spacing), and the shuffle is a permutation.
// Keyed by address: a piece of the tile that starts at row j (base + j x
// spacing, any j) puts every byte where the tile does. The tensor core's
// operand pointer is a logical position: logical byte 16 of row 1 at spacing
// 128 is stored at 128 (its piece slot XORed with 1), not 16 past where byte
// 0 is. The legal spacings are the powers of two from 16.
TEST(Tma, swizzle64_layout_matches_the_rule) {
    for (const uint32_t spacing : {16u, 32u, 64u, 128u, 256u, 512u}) {
        for (const uint32_t base : {0u, 17 * std::max(spacing, 64u)}) {
            std::set<uint32_t> addresses;
            for (uint32_t row = 0; row < 64; ++row) {
                for (uint32_t byte = 0; byte < spacing; ++byte) {
                    const auto address = nanosm::swizzle64_address(base, spacing, row, byte);
                    const auto what = "spacing " + std::to_string(spacing) + " base " + std::to_string(base) +
                        " row " + std::to_string(row) + " byte " + std::to_string(byte);
                    require(address == reference_swizzle64(base, spacing, row, byte), what + ": not the rule");
                    require(address >= base + row * spacing && address < base + (row + 1) * spacing,
                        what + ": left its row");
                    require(addresses.insert(address).second, what + ": two bytes in one place");
                    for (const uint32_t first : {1u, 5u, 37u}) {
                        if (row < first) continue;
                        require(nanosm::swizzle64_address(base + first * spacing, spacing, row - first, byte) ==
                                address,
                            what + ": a piece starting at row " + std::to_string(first) + " puts it elsewhere");
                    }
                }
            }
        }
    }
    require(nanosm::swizzle64_address(0, 128, 1, 16) == 128 && nanosm::swizzle64_address(0, 128, 1, 0) == 144,
        "logical byte 16 of row 1 is not where the rule puts it");
    for (const uint32_t spacing : {0u, 1u, 8u, 15u, 16u, 17u, 24u, 31u, 32u, 48u, 64u, 96u, 128u, 256u, 512u, 1024u,
             2048u, 4095u, 4096u}) {
        const bool power = spacing >= 16 && (spacing & (spacing - 1)) == 0;
        require(nanosm::valid_swizzle64_spacing(spacing) == power,
            "spacing " + std::to_string(spacing) + (power ? " was refused" : " was accepted"));
    }
}

// The base rule (src/layout.hpp): a swizzle64 tile's buffer is a multiple of
// its spacing and of 64, and a tensor-core operand's row 0 lies in one row
// whose line is. The machine shuffles by address and checks neither; the
// sanitizer PANICs: a one-row tile of 128-byte rows at 1088 (a multiple of
// 64, not of 128), and an m64n64k8 A operand of 16-byte rows at 1040 (row
// 1 of a tile at 1024: the 64 rule).
TEST(Tma, swizzle64_base_rule_panics) {
    NANOSM_EXPECTS_SANITIZER_PANIC();
    auto tile = tma_machine(tile_2d(64, 1, 64, 64, 1, 2, swizzle64), false, 1088, {0, 0}, 4096, 128);
    expect_panic(*tile, {"swizzle64 buffer 1088 is not a multiple of 128"}, "a 128-byte-row tile at 1088");
    auto launch = launch_warps(1);
    image_bytes(launch.test_shmem, 8192);
    nanosm::Machine operand(canonical_profile(), compile(
        "umov ur0, 1040\numov ur1, 4096\numov ur2, 0\nmma.async.init.m64n64k8 [ur2], [ur0], [ur1]\nhalt\n"),
        std::move(launch));
    expect_panic(operand, {"mma operand A at 1040 reads 16 bytes per row at row spacing 16"},
        "an operand of 16-byte rows at 1040");
}

namespace {

// A(m, k) and B(k, n): small integers (exact sums), different in every
// 16-byte piece of a row (k and k + 8 differ by 24 = 4 mod 5 and 16 = 1 mod
// 3), so a command that reads the wrong piece is caught. Values 0..4 as fp16.
uint16_t whole_rows_fp16(uint32_t value) {
    static constexpr std::array<uint16_t, 5> bits{0x0000, 0x3c00, 0x4000, 0x4200, 0x4400};
    return bits.at(value);
}

uint32_t whole_rows_a(uint32_t m, uint32_t k) { return (m + 3 * k) % 5; }

uint32_t whole_rows_b(uint32_t k, uint32_t n) { return (2 * k + n) % 3; }

} // namespace

// The tensor core reads TMA tiles of 128-byte rows. A (64 x 64 fp16) and B
// (64 rows of 64 fp16: B's n rows of k, as a K cache) land whole, one load
// each. Eight m64n64k8 commands step both operand pointers by one 16-byte
// piece with row spacing 128 (FA's QK^T); eight m32n64k8 commands read A's
// bottom half, the pointer at row 32 (+ 32 x 128), and eight more read rows
// 21..52 (+ 21 x 128: an operand may start at any row, the shuffle keyed by
// address). A at 0x4080 starts at row 1 of the shuffle and B at 0x180 at
// row 3 (multiples of 128, not of 2048): a reader counting rows from the
// tile would read the wrong pieces. Bank rounds as with 16-byte rows: 17
// per A block and B sweep, and one TMA round per sector.
TEST(TmaSlow, swizzle64_mma_reads_a_tma_tile) {
    constexpr uint32_t a_tile = 0x4080;
    constexpr uint32_t b_tile = 0x180;
    auto launch = launch_warps(1);
    auto& shmem = image_bytes(launch.test_shmem, a_tile + 8192);
    auto& gmem = image_bytes(launch.gmem_image, 16384);
    for (uint32_t row = 0; row < 64; ++row) {
        for (uint32_t k = 0; k < 64; ++k) {
            write_u16(gmem, 2 * (64 * row + k), whole_rows_fp16(whole_rows_a(row, k)));
            write_u16(gmem, 8192 + 2 * (64 * row + k), whole_rows_fp16(whole_rows_b(k, row)));
        }
    }
    write_tma_descriptor(shmem, 0, tile_2d(64, 64, 64, 64, 64, 2, swizzle64));
    write_tma_descriptor(shmem, 64, tile_2d(64, 64, 64, 64, 64, 2, swizzle64, false, 8192));
    write_mbar(shmem, 128, mbar_state(1, 0, 16384));
    write_mbar(shmem, 136, mbar_state(1, 1, 0));
    // Three results in tmem: C (64x64) at 0, H and G (32x64) at 4096 and 6144.
    std::string source = "umov ur20, 0\numov ur21, 4096\numov ur22, 6144\n" +
        tma_operands_asm(1, a_tile, 2, 128, 8, {0, 0}) + "umov ur3, " + std::to_string(b_tile) + "\n" +
        "tma.load.2d [ur1], [0, ur8], [ur2]\ntma.load.2d [ur3], [64, ur8], [ur2]\n" + wait_on("ur2", "loaded");
    for (const auto& [accumulator, shape, a_start] :
         {std::tuple{"[ur20]", "m64n64k8", a_tile}, std::tuple{"[ur21]", "m32n64k8", a_tile + 32 * 128},
          std::tuple{"[ur22]", "m32n64k8", a_tile + 21 * 128}}) {
        source += "umov ur10, " + std::to_string(a_start) + "\numov ur11, " + std::to_string(b_tile) + "\n";
        for (uint32_t piece = 0; piece < 8; ++piece) {
            if (piece != 0) source += "uaddi ur10, ur10, 16\nuaddi ur11, ur11, 16\n";
            source += std::string("mma.async.") + (piece == 0 ? "init." : "") + shape + " " + accumulator +
                ", [ur10, 128], [ur11, 128]\n";
        }
    }
    source += "umov ur4, 136\nmma.commit [ur4]\n" + wait_on("ur4", "committed") +
        // C row 0 / row 48 column 56; H and G row 16 column 8 (4096 + 1024 + 8, 6144 + 1024 + 8).
        "acc.ld.v8 r20, [ur20, 64]\numov ur12, 3128\nacc.ld.v8 r28, [ur12, 64]\n"
        "umov ur14, 5128\numov ur15, 7176\nacc.ld.v8 r36, [ur14, 64]\nacc.ld.v8 r44, [ur15, 64]\nhalt\n";
    nanosm::Machine machine(canonical_profile(), compile(source), std::move(launch));
    machine.run(100000);
    const auto expected = [](uint32_t m, uint32_t n) {
        float sum = 0;
        for (uint32_t k = 0; k < 64; ++k) sum += static_cast<float>(whole_rows_a(m, k) * whole_rows_b(k, n));
        return nanosm::float_bits(sum);
    };
    for (uint32_t j = 0; j < 8; ++j) {
        require_lanes(machine, 20 + j, [&](uint32_t lane) { return expected(lane, j); },
            "C rows 0..15 (each operand's piece and row)");
        require_lanes(machine, 28 + j, [&](uint32_t lane) { return expected(48 + lane, 56 + j); }, "C's far corner");
        require_lanes(machine, 36 + j, [&](uint32_t lane) { return expected(48 + lane, 8 + j); },
            "the m32 command at A's row 32 (rows 32..63)");
        require_lanes(machine, 44 + j, [&](uint32_t lane) { return expected(37 + lane, 8 + j); },
            "the m32 command at A's row 21 (rows 21..52)");
    }
    const auto& summary = machine.trace().summary();
    require(summary.mma_commands == 24 && summary.mma_a_block_reads == 8 * 16 + 2 * 8 * 8 &&
            summary.mma_shmem_rounds == 8 * 272 + 2 * 8 * 136,
        "128-byte rows did not keep one bank round per B block and per A block: " +
            std::to_string(summary.mma_shmem_rounds) + " rounds");
    require(summary.tma_port_requests == 512 && summary.tma_port_rounds == 512,
        "the TMA did not place one sector per port-A round");
}

// A dense tile of 1-byte elements: the longest runs (32 elements, a whole
// sector) and crooked ones. 3 rows of 96 at (5, 1) of a 200-byte-row tensor:
// rows start at 205, 405, 605, so each row is 4 runs (19 + 32 + 32 + 13,
// 11 + 32 + 32 + 21, 3 + 32 + 32 + 29) and every placement, its shared and
// global offsets differing mod 4, takes at most two rounds.
TEST(Tma, tma_byte_tile_longest_run) {
    constexpr uint32_t tile_address = 1024;
    constexpr uint32_t length = 96;
    constexpr uint32_t rows = 3;
    const auto descriptor = tile_2d(200, 4, 200, length, rows, 1);
    const auto value = [](uint32_t address) { return static_cast<uint8_t>(address * 7 + 3); };
    {
        auto machine = tma_machine(descriptor, false, tile_address, {5, 1}, tile_address + length * rows, 800,
            [&](auto&, auto& gmem) { for (uint32_t i = 0; i < 800; ++i) gmem[i] = value(i); });
        machine->run(50000);
        const auto shared = machine->shmem();
        for (uint32_t y = 0; y < rows; ++y) for (uint32_t x = 0; x < length; ++x) {
            require(shared[tile_address + length * y + x] == value(200 * (y + 1) + 5 + x),
                "byte tile load: element (" + std::to_string(x) + ", " + std::to_string(y) + ")");
        }
        const auto& summary = machine->trace().summary();
        require(summary.tma_port_requests == 12 && summary.tma_sector_bytes == 384 &&
                summary.tma_port_rounds > summary.tma_port_requests && summary.tma_port_rounds <= 2 * summary.tma_port_requests,
            "byte tile load: " + std::to_string(summary.tma_port_requests) + " placements of " +
                std::to_string(summary.tma_port_rounds) + " rounds (expected 12 sectors, at most two rounds each)");
    }
    {
        auto machine = tma_machine(descriptor, true, tile_address, {5, 1}, tile_address + length * rows, 800,
            [&](auto& shmem, auto& gmem) {
                std::fill(gmem.begin(), gmem.end(), 0x5a);
                for (uint32_t i = 0; i < length * rows; ++i) shmem[tile_address + i] = value(i);
            });
        machine->run(50000);
        for (uint32_t address = 0; address < 800; ++address) {
            const uint32_t row = address / 200;
            const uint32_t column = address % 200;
            const bool tile = row >= 1 && row <= rows && column >= 5 && column < 5 + length;
            const uint8_t expected = tile ? value(length * (row - 1) + column - 5) : 0x5a;
            require(machine->gmem()[address] == expected, "byte tile store: gmem byte " + std::to_string(address));
        }
        require_eq(machine->trace().summary().tma_sector_bytes, 384, "byte tile store: sector bytes (12 sectors)");
    }
}

// The walker's start stages (tma.walker_start_cycles, 0..8): a lone 2 x 16
// fp32 load at (-4, 0) (zero fills and sectors) makes its first placement
// exactly one cycle later per stage, with the same bytes.
TEST(Tma, tma_walker_start_stages) {
    constexpr uint32_t tile_address = 1024;
    const auto descriptor = tile_2d(12, 2, 12, 16, 2);
    std::vector<uint64_t> first_placement;
    std::vector<std::vector<uint8_t>> landed;
    for (const uint32_t stages : {0u, 1u, 2u, 5u}) {
        auto profile = canonical_profile();
        profile.tma_walker_start_cycles = stages;
        auto machine = tma_machine(descriptor, false, tile_address, {-4, 0}, tile_address + 128, 96,
            [&](auto& shmem, auto& gmem) {
                for (uint32_t i = 0; i < 24; ++i) write_u32(gmem, 4 * i, 0x300 + i);
                std::fill(shmem.begin() + tile_address, shmem.end(), 0xa5);
            },
            profile);
        machine->enable_edge_probe();
        machine->run(50000);
        // The first placement round: the zero fill of x = 0..3, the walk's
        // first step (core cycles only; the end-to-end time also moves by
        // DRAM-tick alignment).
        const auto& events = machine->edge_events();
        const auto first = std::find_if(events.begin(), events.end(),
            [](const auto& event) { return event.edge == LsuEdge::tma_place_round; });
        require(first != events.end(), "no tma placement round");
        first_placement.push_back(first->cycle);
        const auto shared = machine->shmem();
        landed.emplace_back(shared.begin() + tile_address, shared.begin() + tile_address + 128);
        for (uint32_t i = 0; i < 32; ++i) {
            const uint32_t x = i % 16;
            const uint32_t expected = x < 4 ? 0 : 0x300 + 12 * (i / 16) + x - 4;
            require(machine->read_shmem_u32(tile_address + 4 * i) == expected,
                std::to_string(stages) + " start stages: tile word " + std::to_string(i));
        }
    }
    const uint64_t added[] = {0, 1, 2, 5};
    for (std::size_t k = 1; k < first_placement.size(); ++k) {
        require_eq(first_placement[k], first_placement[0] + added[k], std::to_string(added[k]) +
            " start stages: the first placement (that many cycles after none's)");
        require(landed[k] == landed[0], "the start stages changed the landed bytes");
    }
}

// With L = D + Lb and d = ceil(8 / pluck): a store-source read releases
// A-read at its (one) round's grant and its words carry their register's
// index back; a register turns over every L + d cycles (reserved at the
// read's grant, complete L + 1 later, drained in d cycles, reusable in its
// last drain cycle). So N registers and a fast bus give N sectors per
// max(L + d, N d) cycles in steady state: min(N x 32 / (L + d), 32 / d) bytes
// per cycle. Checked exactly, for every steady-state sector, at N = 1..3 and
// pluck 1, 2, 8; and the canonical cases by name (D = Lb = 2): N = 1, pluck
// 2 -> one sector every 8 cycles; N = 2, pluck 2 -> egress-bound, one every
// 4. With N = ceil((L + d) / d) registers a 16 KiB store is pluck-bound again
// (the egress bound).
TEST(TmaSlow, tma_store_sector_registers) {
    const auto& canonical = canonical_profile();
    const uint64_t l = canonical.shmem_port_a_depth + canonical.shmem_read_latency_cycles;
    for (const uint32_t pluck : {1u, 2u, 8u}) {
        const uint64_t d = (8 + pluck - 1) / pluck;
        for (const uint32_t n : {1u, 2u, 3u}) {
            auto config = fast_bus_profile(pluck);
            config.tma_store_sector_registers = n;
            const auto what = "N=" + std::to_string(n) + " pluck " + std::to_string(pluck) + ": ";
            const auto returns = tma_store_returns(config, 64).first;
            const auto period = std::max(l + d, n * d);
            for (std::size_t k = n; k + n < returns.size(); ++k) {
                require_eq(returns[k + n] - returns[k], period, what + "cycles from sector " + std::to_string(k) +
                    " to " + std::to_string(k + n) + " (max(L + d, N d))");
            }
        }
    }
    require(l == 4 && canonical.lsu_pluck_words_per_cycle == 2, "the named cases are derived for D = Lb = 2, pluck 2");
    {
        const auto returns = tma_store_returns(fast_bus_profile(2), 16).first;
        for (std::size_t k = 1; k < returns.size(); ++k) {
            require(returns[k] - returns[k - 1] == 8, "N=1 pluck 2: one register does not turn over every L + d = 8 cycles");
        }
        auto two = fast_bus_profile(2);
        two.tma_store_sector_registers = 2;
        const auto pair = tma_store_returns(two, 16).first;
        for (std::size_t k = 3; k < pair.size(); ++k) {
            require(pair[k] - pair[k - 1] == 4, "N=2 pluck 2: not egress-bound (one sector every d = 4 cycles)");
        }
    }
    for (const uint32_t pluck : {1u, 2u, 8u}) {
        const uint64_t d = (8 + pluck - 1) / pluck;
        auto config = fast_bus_profile(pluck);
        config.lsu_egress_sectors = 2;
        config.tma_store_sector_registers = static_cast<uint32_t>((l + d + d - 1) / d);
        const auto cycles = tma_store_returns(config, 512).second;
        const uint64_t egress_bound = 16384 / (4 * pluck);
        require(cycles >= egress_bound && cycles <= egress_bound + 100, "pluck " + std::to_string(pluck) + ", N = " +
            std::to_string(config.tma_store_sector_registers) + ": 16 KiB took " + std::to_string(cycles) +
            " cycles, not the egress bound " + std::to_string(egress_bound));
    }
}

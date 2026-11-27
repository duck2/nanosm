// Shared helpers of the unit tests. Inline so every test file can include them.
#pragma once

#include "machine/machine.hpp"
#include "machine/mma_numeric.hpp"
#include "hash.hpp"
#include "log.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

#include <gtest/gtest.h>

// A test that expects a sanitizer PANIC is skipped when the sanitizer is
// compiled out (NANOSM_SANITIZER=0): the machine is quiet there by design.
#if NANOSM_SANITIZER
#define NANOSM_EXPECTS_SANITIZER_PANIC() static_cast<void>(0)
#else
#define NANOSM_EXPECTS_SANITIZER_PANIC() \
    GTEST_SKIP() << "expects a sanitizer panic; NANOSM_SANITIZER=0"
#endif

namespace nanosm_test {

inline void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

// Every needle is in `message`.
inline void require_mentions(
    const std::string& message, std::initializer_list<std::string_view> needles, const std::string& what) {
    for (const auto needle : needles) {
        if (message.find(needle) == std::string::npos)
            throw std::runtime_error(what + ": diagnostic lacks '" + std::string(needle) + "': " + message);
    }
}

// ---- Errors: refused profiles, kernels, launches and machine PANICs ----

// The text of the error `action` throws: a ProfileError or a SimError (the
// profile parser and validation, qtxc, the assembler, kernel load, the
// machine), or only `Error` when one is named. `action` returning normally is
// a test failure naming `what`. A failed require inside `action` is not
// caught.
template <typename Error = void, typename Action>
std::string error_of(Action&& action, const std::string& what) {
    if constexpr (std::is_void_v<Error>) {
        try {
            std::forward<Action>(action)();
        } catch (const nanosm::ProfileError& error) {
            return error.what();
        } catch (const nanosm::SimError& error) {
            return error.what();
        }
    } else {
        try {
            std::forward<Action>(action)();
        } catch (const Error& error) {
            return error.what();
        }
    }
    throw std::runtime_error(what + " was accepted");
}

// `action` is refused with a message that mentions every needle.
template <typename Error = void, typename Action>
void expect_error(Action&& action, std::initializer_list<std::string_view> needles, const std::string& what) {
    require_mentions(error_of<Error>(std::forward<Action>(action), what), needles, what);
}

// The machine's PANIC text when it runs (a clean run is a test failure).
inline std::string panic_of(nanosm::Machine& machine, const std::string& what, uint64_t cycles = 100000) {
    return error_of<nanosm::SimError>([&] { machine.run(cycles); }, what + " (no panic)");
}

inline std::string panic_of(const nanosm::DeviceProfile& profile, const nanosm::Kernel& kernel, nanosm::Launch launch,
    const std::string& what, uint64_t cycles = 20000) {
    nanosm::Machine machine(profile, kernel, std::move(launch));
    return panic_of(machine, what, cycles);
}

// The machine PANICs, and its message mentions every needle.
inline void expect_panic(
    nanosm::Machine& machine, std::initializer_list<std::string_view> needles, const std::string& what) {
    require_mentions(panic_of(machine, what), needles, what);
}

// Every tick-by-tick loop here ticks through this: a machine that stops making
// progress fails its test at the cap instead of hanging the binary (with the
// sanitizer compiled out, a bug it would have caught can loop forever).
inline constexpr uint64_t test_cycle_cap = 200000;

inline void tick_capped(nanosm::Machine& machine) {
    if (machine.cycle() >= test_cycle_cap) {
        throw std::runtime_error("cycle cap " + std::to_string(test_cycle_cap) + ": the machine stopped making progress");
    }
    machine.tick();
}

inline void run_until(nanosm::Machine& machine, uint64_t cycle) {
    while (machine.cycle() < cycle) tick_capped(machine);
}

// ---- Images, launches, barriers ----

inline void write_u32(std::vector<uint8_t>& image, uint32_t address, uint32_t value) {
    std::memcpy(image.data() + address, &value, sizeof(value));
}

inline void write_u16(std::vector<uint8_t>& image, uint32_t address, uint16_t value) {
    std::memcpy(image.data() + address, &value, sizeof(value));
}

inline void write_u64(std::vector<uint8_t>& image, uint32_t address, uint64_t value) {
    std::memcpy(image.data() + address, &value, sizeof(value));
}

inline std::vector<uint8_t>& image_bytes(
    std::vector<nanosm::ImageSegment>& image, std::size_t size) {
    image.push_back({0, std::vector<uint8_t>(size, 0)});
    return image.back().bytes;
}

// A launch of `warps` warps and nothing else yet.
inline nanosm::Launch launch_warps(uint32_t warps = 1) {
    nanosm::Launch launch;
    launch.active_warps = warps;
    return launch;
}

// A live mbar: `expected` arrivals a phase, `remaining` of them pending, a
// transaction count of `tx_bytes`, in phase `phase`.
inline nanosm::MbarState mbar_state(uint16_t expected, uint16_t remaining, int32_t tx_bytes = 0, bool phase = false) {
    nanosm::MbarState state;
    state.valid = true;
    state.phase = phase;
    state.expected = expected;
    state.remaining = remaining;
    state.tx_bytes = tx_bytes;
    return state;
}

inline void write_mbar(std::vector<uint8_t>& shmem, uint32_t address, const nanosm::MbarState& state) {
    write_u64(shmem, address, nanosm::encode_mbar_state(state));
}

// A wait on the mbar whose address is in `ur` for phase 0, in qtx and hand
// assembly alike: `label` loops until it completes.
inline std::string wait_on(const std::string& ur, const std::string& label = "wait") {
    return label + ":\nmbar.try_wait p0, [" + ur + "], 0\n@!p0 bra.uni " + label + "\n";
}

inline const nanosm::DeviceProfile& canonical_profile() {
    static const auto profile = nanosm::load_device_profile("profiles/nanosm-v0.json");
    return profile;
}

// ---- TMA ----

inline void write_tma_descriptor(
    std::vector<uint8_t>& image, uint32_t address, const nanosm::TmaDescriptor& descriptor) {
    const auto bytes = nanosm::encode_tma_descriptor(descriptor);
    std::memcpy(image.data() + address, bytes.data(), bytes.size());
}

// A one-dimensional fp32 tensor of `elements` at gmem `base`, its tile the
// whole tensor, dense (src/tma_descriptor.hpp): the descriptor most tests use.
inline nanosm::TmaDescriptor plain_tma_1d(uint32_t elements, uint64_t base = 0) {
    nanosm::TmaDescriptor descriptor;
    descriptor.base = base;
    descriptor.dimensions = {elements, 1, 1, 1};
    descriptor.tile_length = static_cast<uint16_t>(elements);
    return descriptor;
}

// A plain_tma_1d descriptor of `elements` at gmem_base, and the mbar its
// command completes into, expecting those bytes. The buffer, the mbar and the
// direction are the instruction's (tma_operands_asm).
inline void write_rank1_tma(std::vector<uint8_t>& shmem, uint32_t descriptor_address, uint64_t gmem_base,
    uint32_t elements, uint32_t mbar_address) {
    write_tma_descriptor(shmem, descriptor_address, plain_tma_1d(elements, gmem_base));
    write_mbar(shmem, mbar_address, mbar_state(1, 0, static_cast<int32_t>(elements * 4)));
}

// A tma's uniform-register operands as assembly: the buffer's shared address
// into ur`buffer_ur`, the barrier's into ur`barrier_ur`, and the coordinates
// into ur`coordinate_ur` onwards. umov takes any 32-bit word.
inline std::string tma_operands_asm(uint32_t buffer_ur, uint32_t buffer, uint32_t barrier_ur, uint32_t barrier,
    uint32_t coordinate_ur, const std::vector<int32_t>& coordinates = {0}) {
    std::string text = "umov ur" + std::to_string(buffer_ur) + ", " + std::to_string(buffer) + "\n" +
        "umov ur" + std::to_string(barrier_ur) + ", " + std::to_string(barrier) + "\n";
    for (std::size_t i = 0; i < coordinates.size(); ++i) {
        text += "umov ur" + std::to_string(coordinate_ur + i) + ", " + std::to_string(coordinates[i]) + "\n";
    }
    return text;
}

// One one-dimensional TMA command at fixed addresses: its descriptor at 5120
// (`bytes` at gmem 0) and its mbar at 4104 expecting those bytes. The kernel's
// "tma.load.1d [ur10], [5120, ur8], [ur11]" (or the tma.store.1d form) takes
// its operands from one_tma_operands.
inline std::function<void(std::vector<uint8_t>&)> one_tma_image(uint32_t bytes) {
    return [=](std::vector<uint8_t>& shmem) {
        write_tma_descriptor(shmem, 5120, plain_tma_1d(bytes / 4));
        write_mbar(shmem, 4104, mbar_state(1, 0, static_cast<int32_t>(bytes)));
    };
}

// one_tma_image's command's operands, its buffer at shmem `shared`: ur10 the
// buffer, ur11 the mbar (4104), ur8 the coordinate (0).
inline std::string one_tma_operands(uint32_t shared) {
    return tma_operands_asm(10, shared, 11, 4104, 8);
}

// The swizzle64 layout (src/layout.hpp), written out independently of the
// production functions: byte `byte`
// of row `row` of a tile at `base` (a multiple of the spacing and of 64)
// whose rows are `spacing` bytes apart. The row stays whole at base + row x
// spacing; the row number that shuffles it is the unshuffled address's own,
// (base + row x spacing + byte) / spacing: the piece slot is XORed with it
// among a row's own slots, the word with (that row / 4) & 3.
inline uint32_t reference_swizzle64(uint32_t base, uint32_t spacing, uint32_t row, uint32_t byte) {
    const uint32_t natural = base + row * spacing + byte;
    const uint32_t absolute_row = natural / spacing;
    const uint32_t slots = std::min(spacing, 64u) / 16;
    const uint32_t rows_per_line = std::max(1u, 64 / spacing);
    const uint32_t slot_xor = (absolute_row / rows_per_line) & (slots - 1);
    const uint32_t word_xor = (absolute_row / 4) & 3;
    return natural ^ (slot_xor << 4) ^ (word_xor << 2);
}

// ---- Kernels ----

// Most test kernels are written at the level of physical registers (r5,
// ur3): they pin cycle counts that depend on which bank a register is in.
// qtx takes names only, so a kernel written that way is renamed here, r5 ->
// %r5 and ur3 -> %ur3, with one group declared for each file (`.reg %r<N>`,
// `.ureg %ur<M>`, appended, so line numbers stay). A group's members keep
// their distances and a lone group starts at number 0, so %r5 is r5 again;
// compile() checks that it is. A kernel written with %names passes through
// unchanged. renamed: whether anything was.
inline std::string name_raw_registers(
    std::string_view source, const nanosm::DeviceProfile& profile, bool* renamed = nullptr) {
    static const std::regex raw(R"(\b([uU]?)[rR]([0-9]+)\b)");
    const std::string text(source);
    std::string result;
    int max_vector = -1;
    int max_uniform = -1;
    auto begin = text.cbegin();
    for (std::sregex_iterator it(text.begin(), text.end(), raw), end; it != end; ++it) {
        const auto at = text.cbegin() + it->position();
        if (at != text.cbegin() && *(at - 1) == '%') continue; // a name already
        const bool uniform = (*it)[1].length() != 0;
        const int index = std::stoi((*it)[2].str().substr(0, 6));
        (uniform ? max_uniform : max_vector) = std::max(uniform ? max_uniform : max_vector, index);
        result.append(begin, at);
        result += (uniform ? "%ur" : "%r") + (*it)[2].str();
        begin = at + it->length();
    }
    result.append(begin, text.cend());
    if (renamed != nullptr) *renamed = max_vector >= 0 || max_uniform >= 0;
    if (max_vector < 0 && max_uniform < 0) return text;
    if (result.empty() || result.back() != '\n') result += '\n';
    // Room for a v8's eight registers and a tma's four coordinates after the
    // highest one named.
    if (max_vector >= 0) {
        result += ".reg %r<" + std::to_string(std::max(max_vector + 1, std::min(max_vector + 8, 256))) + ">\n";
    }
    if (max_uniform >= 0) {
        const int room = std::max(max_uniform + 1, std::min<int>(max_uniform + 4, static_cast<int>(profile.urf_registers)));
        result += ".ureg %ur<" + std::to_string(std::max(room, 2)) + ">\n";
    }
    return result;
}

inline void check_raw_registers_kept(const std::vector<nanosm::RegisterName>& names) {
    for (const auto& entry : names) {
        const auto expected = (entry.uniform ? "ur" : "r") + std::to_string(entry.number);
        if (entry.name.rfind("qtxc_copy", 0) == 0 || entry.predicate) continue; // a scratch qtxc made; a predicate
        if (entry.name != expected) {
            throw std::runtime_error("test kernel: %" + entry.name + " was given " + expected +
                "; this test's kernel is written in physical registers and needs them kept");
        }
    }
}

inline nanosm::Kernel compile(
    std::string_view source, const nanosm::DeviceProfile& profile = canonical_profile()) {
    bool renamed = false;
    const auto text = name_raw_registers(source, profile, &renamed);
    auto kernel = nanosm::compile_and_assemble(text, profile);
    if (renamed) check_raw_registers_kept(kernel.register_names);
    return kernel;
}

// nanosm::compile_qtx for a test kernel (renamed as above): the scheduled
// program, for a test that looks at stalls, depbars and waits.
inline nanosm::AssemblyProgram qtx_program(
    std::string_view source, const nanosm::DeviceProfile& profile = canonical_profile()) {
    bool renamed = false;
    const auto text = name_raw_registers(source, profile, &renamed);
    auto program = nanosm::compile_qtx(text, profile);
    if (renamed) check_raw_registers_kept(program.register_names);
    return program;
}

// Hand assembly: lines without a {...} control block get the default one.
inline std::string hand_asm(const nanosm::DeviceProfile& profile, const std::string& body) {
    std::string text = ".profile " + profile.name + ", 0x" +
        nanosm::format_profile_fingerprint(nanosm::device_profile_fingerprint(profile)) + "\n";
    std::istringstream lines(body);
    std::string line;
    while (std::getline(lines, line)) {
        const auto first = line.find_first_not_of(" \t");
        if (first == std::string::npos) continue;
        text += line;
        if (line[first] != '.' && line.find('{') == std::string::npos) text += " {pred=PT, stall=0, set=-, wait=-}";
        text += "\n";
    }
    return text;
}

// Hand assembly with the fixed-latency stalls its register and predicate
// dependencies need under `profile` (src/timing.hpp, the distances qtxc
// schedules with): every authored stall is kept when it is at least that,
// raised when not; depbars and waits stay as written.
inline nanosm::Kernel hand_assemble(const std::string& text, const nanosm::DeviceProfile& profile) {
    return nanosm::schedule_fixed_stalls(nanosm::assemble_asm(text, profile), profile);
}

// A divergent region: `body` runs with only
// the lanes of `mask` active; the others wait on the divergence stack. An
// if with an empty other arm: the body's sync pops the waiting lanes, whose
// arm is the second sync alone, which pops the ssy's entry (both pcs are the
// next instruction: no redirect). The warp goes on converged at an ordinary
// instruction. It uses r60, r61 and p6 (keep them out of `body`). The text
// suits qtx and hand_asm alike (every label shares its line with an
// instruction).
inline std::string masked_region(uint64_t mask, const std::string& body) {
    return "mov r60, " + std::to_string(mask) + "\n"
           "sread r61, LANE_ID\n"
           "shr r60, r60, r61\n"
           "andi r60, r60, 1\n"
           "setp.eq.u32 p6, r60, 0\n"
           "ssy masked_join\n"
           "@p6 bra masked_skip\n" +
        body + "\n"
               "sync\n"
               "masked_skip: sync\n"
               "masked_join: mov r60, 0\n";
}

// ---- Machines under a debug trace ----

// A machine whose shmem words 0..2047 hold 0x100 + word, with the given mbars
// written over them, under a debug trace (every granted shmem round is an
// instant "<service> grant", detail = client, value = slot; the RF's reads
// and write-port presentations are events too).
inline std::unique_ptr<nanosm::Machine> debug_machine(const nanosm::DeviceProfile& profile,
    const nanosm::Kernel& kernel, uint32_t warps = 1,
    const std::vector<std::pair<uint32_t, nanosm::MbarState>>& mbars = {}) {
    auto launch = launch_warps(warps);
    auto& shmem = image_bytes(launch.test_shmem, 8192);
    for (uint32_t word = 0; word < 2048; ++word) write_u32(shmem, 4 * word, 0x100 + word);
    for (const auto& [address, state] : mbars) write_mbar(shmem, address, state);
    return std::make_unique<nanosm::Machine>(profile, kernel, std::move(launch),
        nanosm::TraceOptions{.timeline = true, .debug = true});
}

// The same for a hand-assembled program (hand_asm, hand_assemble).
inline std::unique_ptr<nanosm::Machine> debug_machine(const nanosm::DeviceProfile& profile,
    const std::string& hand_assembly, uint32_t warps = 1,
    const std::vector<std::pair<uint32_t, nanosm::MbarState>>& mbars = {}) {
    return debug_machine(profile, hand_assemble(hand_asm(profile, hand_assembly), profile), warps, mbars);
}

// The host's events: a TMA-completion mbar at event_mbar waiting for 64
// transaction bytes (event_mbars(), for debug_machine). host_rmw_event (call
// between ticks) places a complete_tx of 64 bytes into the TMA's event
// register, held from the next cycle; it completes the mbar's phase. An RMW
// STARTS as soon as the engine is idle and A-write's alternation lets it (no
// store waiting for its turn): from that cycle to its write grant the store
// pause holds every new store transaction; A-read is held only while the RMW
// waits for its read -- behind a store transaction or an A-read transaction
// under way -- and serves everyone from the read on.
inline constexpr uint32_t event_mbar = 4096;

inline std::vector<std::pair<uint32_t, nanosm::MbarState>> event_mbars() {
    return {{event_mbar, mbar_state(1, 0, 64)}};
}

inline void host_rmw_event(nanosm::Machine& machine) {
    machine.complete_mbar_tx(event_mbar, 64, machine.read_mbar_state(event_mbar).phase ? 1 : 0);
}

// ---- Trace readers ----

inline std::vector<nanosm::TimelineEvent> events_named(const nanosm::Machine& machine, const std::string& name) {
    std::vector<nanosm::TimelineEvent> events;
    for (const auto& event : machine.trace().events()) {
        if (event.name == name) events.push_back(event);
    }
    return events;
}

// An issue event (the instruction category also carries the debug trace's
// "RF read" samples and "RF write" port presentations).
inline bool is_issue_event(const nanosm::TimelineEvent& event) {
    return event.category == nanosm::TraceCategory::instruction && event.name != "RF read" &&
        event.name != "RF write" && event.name != "UR write";
}

inline std::vector<uint64_t> issued_at(const nanosm::Machine& machine, uint32_t pc) {
    std::vector<uint64_t> result;
    for (const auto& event : machine.trace().events()) {
        if (is_issue_event(event) && event.id == pc) result.push_back(event.start_cycle);
    }
    return result;
}

// Every issue, in order: " c<cycle>:w<warp>:<pc>:<opcode>" (a debugging aid
// for failure messages).
inline std::string issue_log(const nanosm::Machine& machine) {
    std::string text;
    for (const auto& event : machine.trace().events()) {
        if (!is_issue_event(event)) continue;
        const auto warp = event.detail.substr(5, event.detail.find(',') - 5);
        text += " c" + std::to_string(event.start_cycle) + ":w" + warp + ":" + std::to_string(event.id) + ":" + event.name;
    }
    return text;
}

// Warp `warp`'s refusal slices with this reason: (first cycle, length).
inline std::vector<std::pair<uint64_t, uint64_t>> refusals(
    const nanosm::Machine& machine, uint32_t warp, const std::string& reason) {
    std::vector<std::pair<uint64_t, uint64_t>> result;
    for (const auto& event : machine.trace().events()) {
        if (event.category == nanosm::TraceCategory::warp && event.track == 100 + warp && event.detail == reason) {
            result.emplace_back(event.start_cycle, event.duration_cycles);
        }
    }
    return result;
}

inline std::string describe(const std::vector<std::pair<uint64_t, uint64_t>>& slices) {
    std::string text;
    for (const auto& [start, length] : slices) text += " {" + std::to_string(start) + ", " + std::to_string(length) + "}";
    return text;
}

// Warp 0's debug RF events of one register (its warp track): "RF read"
// samples or "RF write" port presentations, by cycle.
inline std::vector<uint64_t> rf_events(const nanosm::Machine& machine, const std::string& name, uint8_t reg) {
    std::vector<uint64_t> cycles;
    for (const auto& event : machine.trace().events()) {
        if (event.category == nanosm::TraceCategory::instruction && event.name == name && event.id == reg &&
            event.track == 100) {
            cycles.push_back(event.start_cycle);
        }
    }
    return cycles;
}

// One run, tick by tick: the cycle in which each watched register of warp 0
// changed (lane 0), and `observe(cycle)` after every tick.
inline std::map<uint32_t, std::vector<uint64_t>> watch_registers(nanosm::Machine& machine,
    const std::vector<uint32_t>& regs, const std::function<void(uint64_t)>& observe = {}) {
    std::map<uint32_t, uint32_t> before;
    for (const auto reg : regs) before[reg] = machine.read_reg(0, 0, reg);
    std::map<uint32_t, std::vector<uint64_t>> written;
    while (!machine.done()) {
        const auto cycle = machine.cycle();
        tick_capped(machine);
        for (const auto reg : regs) {
            const auto now = machine.read_reg(0, 0, reg);
            if (now != before[reg]) written[reg].push_back(cycle);
            before[reg] = now;
        }
        if (observe) observe(cycle);
    }
    return written;
}

// ---- Shared memory and the RMW engine ----

// A read's words reach the port output D + Lb cycles after its grant.
inline uint32_t read_return(const nanosm::DeviceProfile& profile) {
    return profile.shmem_port_a_depth + profile.shmem_read_latency_cycles;
}

// Accepted issue -> an uncontended queued shared-memory access's first grant
// (src/machine/shmem.cpp): it enters its queue at the issue's edge and may
// leave it at issue + 1 (its uniform operands, if any, at the URF output by
// the time its collect stage uses them: issue + Lu), is in the collect stage
// the cycle after leaving, and enters serve when its registers are at the RF
// output (at once without vector registers; Lr after its collect stage's
// sample for an lds / sts), granted the cycle after. Canonical: 3 for
// lds.uni, a probe, a barrier change, an atomic; 5 for lds and sts.
inline uint32_t issue_to_shmem_grant(const nanosm::DeviceProfile& profile, bool reads_registers = false) {
    return reads_registers ? 3 + profile.rf_read_latency_cycles : 2 + profile.urf_read_latency_cycles;
}

// An lds's result is written to the RF D + Lb + (issue to grant) - 1 after
// its issue (vector_address: an lds; else an lds.uni).
inline uint32_t lds_shift(const nanosm::DeviceProfile& profile = canonical_profile(), bool vector_address = false) {
    return profile.shmem_port_a_depth + profile.shmem_read_latency_cycles +
        issue_to_shmem_grant(profile, vector_address) - 1;
}

// An RMW's write is granted D + Lb + C cycles after its read (the read's
// words reach the port output, then the update crosses the C ALU stages).
// Its "mbar RMW" slice runs from its START (the store pause begins) to its
// write grant; uncontended -- no store or A-read transaction under way -- the
// read is granted in the start cycle, so the engine is busy D + Lb + C + 1
// cycles. Back-to-back services are that far apart.
inline uint64_t rmw_busy_cycles(const nanosm::DeviceProfile& profile = canonical_profile()) {
    return profile.shmem_port_a_depth + profile.shmem_read_latency_cycles + profile.shmem_rmw_compute_cycles + 1;
}

// Issue to dependent issue behind a one-cycle ALU result (mov, sread,
// shli, ...): visible issue_to_visible after its producer's issue, sampled
// issue_register_stages after the consumer's (src/timing.hpp; 6 at canonical).
inline uint64_t alu_distance(const nanosm::DeviceProfile& profile) {
    nanosm::Instruction mov;
    mov.opcode = nanosm::Opcode::mov;
    return nanosm::issue_to_visible(mov, profile) - nanosm::issue_register_stages;
}

// One granted shmem round of a debug trace.
struct Grant {
    uint64_t cycle = 0;
    std::string service; // "A-read", "A-write", "B-write"
    std::string client;  // "simt", "probe", "rmw", "tma"
    uint32_t slot = 0;
    bool operator==(const Grant&) const = default;
};

inline std::string describe(const std::vector<Grant>& grants) {
    std::string text;
    for (const auto& grant : grants) {
        text += " c" + std::to_string(grant.cycle) + ":" + grant.service + "/" + grant.client +
            (grant.client == "simt" ? std::to_string(grant.slot) : "");
    }
    return text;
}

// The rounds granted (debug traces), by cycle; one service, or all.
inline std::vector<Grant> grants_of(const nanosm::Machine& machine, const std::string& service = {}) {
    std::vector<Grant> grants;
    for (const auto& event : machine.trace().events()) {
        if (event.category != nanosm::TraceCategory::shmem || event.kind != nanosm::TraceEventKind::instant) continue;
        const auto name = event.name.substr(0, event.name.find(" grant"));
        if (!service.empty() && name != service) continue;
        grants.push_back({event.start_cycle, name, event.detail, static_cast<uint32_t>(event.value)});
    }
    std::stable_sort(grants.begin(), grants.end(), [](const auto& a, const auto& b) { return a.cycle < b.cycle; });
    return grants;
}

// The cycles of one service's grants to one client.
inline std::vector<uint64_t> grant_cycles(
    const nanosm::Machine& machine, const std::string& service, const std::string& client) {
    std::vector<uint64_t> cycles;
    for (const auto& grant : grants_of(machine, service)) {
        if (grant.client == client) cycles.push_back(grant.cycle);
    }
    return cycles;
}

// Every SIMT shmem request's round count, in issue order (its trace slice).
inline std::vector<uint32_t> shmem_request_rounds(const nanosm::Machine& machine) {
    std::vector<std::pair<uint64_t, uint32_t>> requests;
    for (const auto& event : machine.trace().events()) {
        if (event.category != nanosm::TraceCategory::shmem || event.kind != nanosm::TraceEventKind::slice) continue;
        const auto rounds = event.detail.substr(event.detail.find("rounds=") + 7);
        requests.emplace_back(event.start_cycle, static_cast<uint32_t>(std::stoul(rounds)));
    }
    std::sort(requests.begin(), requests.end());
    std::vector<uint32_t> result;
    for (const auto& request : requests) result.push_back(request.second);
    return result;
}

using RmwService = std::pair<uint64_t, std::string>; // start cycle, trace detail

// The RMW engine's service order: every "mbar RMW" slice, by start cycle.
inline std::vector<RmwService> mbar_rmw_services(const nanosm::Machine& machine) {
    std::vector<RmwService> services;
    for (const auto& event : machine.trace().events())
        if (event.name == "mbar RMW") services.emplace_back(event.start_cycle, event.detail);
    std::sort(services.begin(), services.end());
    return services;
}

// An RMW's START (its "mbar RMW" slice's start: the store pause begins at
// that cycle's edge), and its read and write (the "rmw" grants on A-read and
// A-write).
struct RmwSlice {
    uint64_t start = 0;
    uint64_t read = 0;
    uint64_t write = 0;
    std::string detail;
};

inline std::string describe(const std::vector<RmwSlice>& rmws) {
    std::string text;
    for (const auto& rmw : rmws) {
        text += " {" + rmw.detail + ": start c" + std::to_string(rmw.start) + ", read c" + std::to_string(rmw.read) +
            ", write c" + std::to_string(rmw.write) + "}";
    }
    return text;
}

// An atomic: an RMW-engine operation whose "atomic RMW" slice runs from its
// START (the store pause begins) to its write grant + 1; inside it, its read
// and its write are the "rmw" grants on A-read and A-write.
struct AtomicSlice {
    uint64_t start = 0;
    uint64_t end = 0;             // the write grant + 1
    std::vector<uint64_t> reads;  // its read grant
    std::vector<uint64_t> writes; // its write grant
    std::string detail;
};

inline std::string describe(const AtomicSlice& atomic) {
    std::string text = " {" + atomic.detail + ": start c" + std::to_string(atomic.start) + ", end c" +
        std::to_string(atomic.end) + ", reads";
    for (const auto cycle : atomic.reads) text += " " + std::to_string(cycle);
    text += ", writes";
    for (const auto cycle : atomic.writes) text += " " + std::to_string(cycle);
    return text + "}";
}

// The smallest ISA shape (m32n64k8: A 512 B, B 1024 B) on a 4x4x4 core,
// dense layout, one shmem bank: every port-B round is one word, so a step
// that reloads A is sixteen rounds (eight A words, eight B words) and every
// other step eight.
inline nanosm::DeviceProfile dense_one_bank_profile(uint32_t result_latency) {
    auto profile = canonical_profile();
    profile.mma_shapes = {{32, 64, 8}};
    profile.mma_core_m = 4;
    profile.mma_core_n = 4;
    profile.mma_core_k = 4;
    profile.mma_layout = nanosm::TileLayout::dense;
    profile.mma_result_latency_cycles = result_latency;
    profile.shmem_banks = 1;
    nanosm::validate_device_profile(profile);
    return profile;
}

// ---- Global memory and the LSU ----

// Coprime periods (31, 29): unique for 899 consecutive words, so a word in
// the wrong lane, register or sector cannot hide behind a repeat.
inline uint32_t pattern_word(uint32_t index) {
    return 0x80000000u | ((index % 31) << 16) | (index % 29);
}

inline void fill_pattern(std::vector<uint8_t>& image, uint32_t byte_address, uint32_t words, uint32_t first_index) {
    for (uint32_t i = 0; i < words; ++i) write_u32(image, byte_address + 4 * i, pattern_word(first_index + i));
}

using nanosm::LsuEdge;

// A bus far faster than any pluck rate (a whole sector per core cycle, no
// latency), so the declared SM-side widths are what binds.
inline nanosm::DeviceProfile fast_bus_profile(uint32_t pluck) {
    auto config = canonical_profile();
    config.dram_clock_hz = config.core_clock_hz;
    config.gmem_bus_bytes_per_dram_tick = 32;
    config.gmem_latency_dram_ticks = 0;
    config.lsu_pluck_words_per_cycle = pluck;
    return config;
}

// The RF-starvation setup of the LSU backpressure tests. A stream of movs
// (latency 1) cannot starve an async writer by booking its bank every cycle:
// one denial with service owed inhibits new bookings to the bank from the
// next cycle, so a ready async writer waits at most for the bookings made up
// to its denial. To hold one for
// a while, the stream books booking_horizon cycles ahead: fadds whose result
// reaches the write port that far after their issue (the issue register, Lr
// and the fadd latency set by with_long_bookings, src/timing.hpp), one per
// cycle, rotating over booking_horizon destination registers of `bank`. The
// fadd latency is as long as the five-bit stall field allows at canonical RF
// depths (window 1, Lr 2, Wd 2: 31 - 2 - 2 - 1 = 26), less a margin.
inline constexpr uint32_t booking_latency = 24;

inline constexpr uint32_t booking_horizon = 1 + 2 + booking_latency; // issue register + canonical Lr + latency

inline void with_long_bookings(nanosm::DeviceProfile& config) {
    config.timing.fadd = booking_latency;
}

inline std::string booking_stream(uint32_t bank, uint32_t count) {
    std::string text;
    for (uint32_t i = 0; i < count; ++i) {
        text += "fadd r" + std::to_string(bank + 3 * (i % booking_horizon)) + ", r1, r1\n";
    }
    return text;
}

// ---- Profile fields ----

// An optional profile field: the canonical JSON declares `canonical`;
// absent, it defaults to the same value (same fingerprint); negative,
// fractional and values outside [low, high] are refused by parsing, and 0 by
// validation when low is above 0; another value changes the fingerprint
// (`make_valid` adjusts that other profile where the value alone breaks a
// rule between fields; the comparison is then with the canonical profile
// under the same adjustment, so the field is the only difference).
inline void check_optional_field(const char* section, const char* field, uint32_t canonical,
    uint32_t nanosm::DeviceProfile::*member, uint32_t low = 1, uint32_t high = 64,
    const std::function<void(nanosm::DeviceProfile&)>& make_valid = {}) {
    std::ifstream input("profiles/nanosm-v0.json");
    nlohmann::json document;
    input >> document;
    const auto parsed = nanosm::parse_device_profile(document.dump());
    const auto name = std::string(section) + "." + field;
    require(parsed.*member == canonical && document[section].contains(field),
        name + ": the canonical profile does not declare " + std::to_string(canonical));
    auto defaulted = document;
    defaulted[section].erase(field);
    require(nanosm::device_profile_fingerprint(nanosm::parse_device_profile(defaulted.dump())) ==
            nanosm::device_profile_fingerprint(parsed), name + ": the default is not " + std::to_string(canonical));
    std::vector<nlohmann::json> refused = {nlohmann::json(-1), nlohmann::json(1.5), nlohmann::json(high + 1)};
    if (low > 0) refused.push_back(nlohmann::json(low - 1));
    for (const auto& value : refused) {
        auto modified = document;
        modified[section][field] = value;
        (void)error_of<nanosm::ProfileError>([&] { (void)nanosm::parse_device_profile(modified.dump()); },
            name + " = " + value.dump());
    }
    if (low > 0) {
        auto zero = canonical_profile();
        zero.*member = 0;
        (void)error_of<nanosm::ProfileError>([&] { nanosm::validate_device_profile(zero); }, name + " = 0 in validation");
    }
    auto base = canonical_profile();
    if (make_valid) make_valid(base);
    auto other = base;
    other.*member = canonical + 1;
    require(nanosm::device_profile_fingerprint(other) != nanosm::device_profile_fingerprint(base),
        name + " was omitted from the profile fingerprint");
}

// ---- tensor core, sanitizer, ALU (mma_test, sanitizer_test, alu_test) ----

// A tensor-core run in qtx: ur0 and ur1 hold the A and B objects' shmem
// addresses (rows a_spacing and b_spacing bytes apart), ur2 the result's tmem
// pointer, ur4 the commit's mbar (initialised by the kernel). `commands`
// mma.async of `shape` into the result, the first an init; mma.commit; a
// wait for the commit (label "wait"); `tail` (an acc.ld through ur2, say);
// halt.
struct MmaRun {
    std::string shape = "m64n64k8";
    uint32_t a = 0;
    uint32_t b = 1024;
    uint32_t acc = 0;
    uint32_t a_spacing = 16;
    uint32_t b_spacing = 16;
    uint32_t commands = 1;
    uint32_t mbar = 8192;
    std::string tail;
};

inline std::string mma_kernel(const MmaRun& run) {
    const auto operand = [](const std::string& ur, uint32_t spacing) {
        return "[" + ur + (spacing == 16 ? "" : ", " + std::to_string(spacing)) + "]";
    };
    const auto operands = " [ur2], " + operand("ur0", run.a_spacing) + ", " + operand("ur1", run.b_spacing) + "\n";
    std::string text = "umov ur0, " + std::to_string(run.a) + "\numov ur1, " + std::to_string(run.b) + "\numov ur2, " +
        std::to_string(run.acc) + "\numov ur4, " + std::to_string(run.mbar) + "\nmbar.init [ur4], 1\n";
    text += "mma.async.init." + run.shape + operands;
    for (uint32_t command = 1; command < run.commands; ++command) text += "mma.async." + run.shape + operands;
    return text + "mma.commit [ur4]\n" + wait_on("ur4") + run.tail + "halt\n";
}

// ---- shared memory and the RMW engine (shmem_test, rmw_test) ----

// Issue to dependent issue behind a uniform-register writer (umov, usread,
// u* arithmetic, r2ur): visible issue_to_urf_visible after the writer's
// issue, sampled issue_to_urf_sample after the reader's (src/timing.hpp; 4 at
// canonical, where alu_distance is 6).
inline uint64_t uniform_distance(const nanosm::DeviceProfile& profile) {
    return nanosm::issue_to_urf_visible(profile) - nanosm::issue_to_urf_sample;
}

// debug_machine's shmem (words 0..2047 hold 0x100 + word) changed by `setup`,
// gmem words 0..1023 holding 5 * word + 2, and `declared` the launch's
// declared mbars (Launch::mbars), under a debug trace.
inline std::unique_ptr<nanosm::Machine> gmem_debug_machine(const nanosm::DeviceProfile& profile,
    const nanosm::Kernel& kernel, uint32_t warps = 1, const std::function<void(std::vector<uint8_t>&)>& setup = {},
    const std::vector<uint32_t>& declared = {}) {
    auto launch = launch_warps(warps);
    auto& shmem = image_bytes(launch.test_shmem, 8192);
    for (uint32_t word = 0; word < 2048; ++word) write_u32(shmem, 4 * word, 0x100 + word);
    if (setup) setup(shmem);
    auto& gmem = image_bytes(launch.gmem_image, 4096);
    for (uint32_t word = 0; word < 1024; ++word) write_u32(gmem, 4 * word, 5 * word + 2);
    launch.mbars = declared;
    return std::make_unique<nanosm::Machine>(profile, kernel, std::move(launch),
        nanosm::TraceOptions{.timeline = true, .debug = true});
}

// The same for a hand-assembled program (hand_asm, hand_assemble).
inline std::unique_ptr<nanosm::Machine> gmem_debug_machine(const nanosm::DeviceProfile& profile,
    const std::string& hand_assembly, uint32_t warps = 1, const std::function<void(std::vector<uint8_t>&)>& setup = {},
    const std::vector<uint32_t>& declared = {}) {
    return gmem_debug_machine(
        profile, hand_assemble(hand_asm(profile, hand_assembly), profile), warps, setup, declared);
}

// ---- register files and front end (rf_test, frontend_test, urf_test) ----

// One issue of a run: its cycle, warp, pc and opcode name.
struct IssueRecord {
    uint64_t cycle = 0;
    uint32_t warp = 0;
    uint32_t pc = 0;
    std::string opcode;
};

// Every issue of the run, in trace order.
inline std::vector<IssueRecord> issue_records(const nanosm::Machine& machine) {
    std::vector<IssueRecord> issues;
    for (const auto& event : machine.trace().events()) {
        if (!is_issue_event(event)) continue;
        const auto warp = std::stoul(event.detail.substr(5, event.detail.find(',') - 5));
        issues.push_back({event.start_cycle, static_cast<uint32_t>(warp), static_cast<uint32_t>(event.id), event.name});
    }
    return issues;
}

// The cycles in which warp `warp` issued the instruction at `pc`.
inline std::vector<uint64_t> warp_issue_cycles(const nanosm::Machine& machine, uint32_t warp, uint32_t pc) {
    std::vector<uint64_t> cycles;
    for (const auto& issue : issue_records(machine)) {
        if (issue.warp == warp && issue.pc == pc) cycles.push_back(issue.cycle);
    }
    return cycles;
}

// ---- TMA and global memory (tma_test, tma_operands_test, lsu_test) ----

// ---- the rest (stepping, basics, divergence, machine, registers, memory) ----

// The number qtxc gave the register or predicate named %`name`.
inline uint8_t number_of(const std::vector<nanosm::RegisterName>& names, const std::string& name) {
    for (const auto& entry : names) {
        if (entry.name == name) return entry.number;
    }
    throw std::runtime_error("no register named %" + name);
}

// qtxc refuses `source` (canonical profile) with a message that mentions `needle`.
inline void qtx_refused(const std::string& source, const std::string& needle, const std::string& what) {
    expect_error([&] { (void)nanosm::compile_qtx(source, canonical_profile()); }, {needle}, what);
}

} // namespace nanosm_test

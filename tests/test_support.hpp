// Shared helpers of the unit tests (moved from the original tests/tests.cpp;
// trash/test-setup-2026-10-22.md). Inline so every test file can include them.
#pragma once

#include "sim.hpp"
#include "mma_numeric.hpp"
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
#include <set>
#include <sstream>
#include <string>
#include <tuple>
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

inline void write_u32(std::vector<uint8_t>& image, uint32_t address, uint32_t value) {
    std::memcpy(image.data() + address, &value, sizeof(value));
}

inline void write_u16(std::vector<uint8_t>& image, uint32_t address, uint16_t value) {
    std::memcpy(image.data() + address, &value, sizeof(value));
}

inline void write_u64(std::vector<uint8_t>& image, uint32_t address, uint64_t value) {
    std::memcpy(image.data() + address, &value, sizeof(value));
}

inline void write_tma_descriptor(
    std::vector<uint8_t>& image, uint32_t address, const nanosm::TmaDescriptor& descriptor) {
    const auto bytes = nanosm::encode_tma_descriptor(descriptor);
    std::memcpy(image.data() + address, bytes.data(), bytes.size());
}

// A one-dimensional fp32 tensor of `elements` at gmem `base`, its tile the
// whole tensor, dense (src/tma.hpp): the descriptor most tests use.
inline nanosm::TmaDescriptor plain_tma_1d(uint32_t elements, uint64_t base = 0) {
    nanosm::TmaDescriptor descriptor;
    descriptor.base = base;
    descriptor.dimensions = {elements, 1, 1, 1};
    descriptor.tile_length = static_cast<uint16_t>(elements);
    return descriptor;
}

// A tma's uniform-register operands as assembly: the buffer's shared address
// into ur`buffer_ur`, the barrier's into ur`barrier_ur`, and the coordinates
// into ur`coordinate_ur` onwards. umov's signed 22-bit immediate holds any
// shared address.
inline std::string tma_operands_asm(uint32_t buffer_ur, uint32_t buffer, uint32_t barrier_ur, uint32_t barrier,
    uint32_t coordinate_ur, const std::vector<int32_t>& coordinates = {0}) {
    std::string text = "umov ur" + std::to_string(buffer_ur) + ", " + std::to_string(buffer) + "\n" +
        "umov ur" + std::to_string(barrier_ur) + ", " + std::to_string(barrier) + "\n";
    for (std::size_t i = 0; i < coordinates.size(); ++i) {
        text += "umov ur" + std::to_string(coordinate_ur + i) + ", " + std::to_string(coordinates[i]) + "\n";
    }
    return text;
}

inline std::vector<uint8_t>& image_bytes(
    std::vector<nanosm::ImageSegment>& image, std::size_t size) {
    image.push_back({0, std::vector<uint8_t>(size, 0)});
    return image.back().bytes;
}

inline const nanosm::DeviceProfile& canonical_profile() {
    static const auto profile = nanosm::load_device_profile("profiles/nanosm-v0.json");
    return profile;
}

// I2 / I2.1 / I3 (rfc12 sections 2 and 6): an RMW's write is granted D + Lb
// + C cycles after its read (the read's words reach the port output, then the
// update crosses the C ALU stages). Its "mbar RMW" slice runs from its START
// (I3: the store pause begins) to its write grant; uncontended -- no store or
// A-read transaction under way -- the read is granted in the start cycle, so
// the engine is busy D + Lb + C + 1 cycles (2 before I2; D + Lb + 1 in I2,
// which had no ALU stage). Back-to-back services are that far apart.
inline uint64_t rmw_busy_cycles(const nanosm::DeviceProfile& profile = canonical_profile()) {
    return profile.shmem_port_a_depth + profile.shmem_read_latency_cycles + profile.shmem_rmw_compute_cycles + 1;
}

inline nanosm::Kernel compile(
    std::string_view source, const nanosm::DeviceProfile& profile = canonical_profile()) {
    return nanosm::compile_and_assemble(source, profile);
}

// Hand assembly: lines without a {...} control block get the default one.
inline std::string i1_asm(const nanosm::DeviceProfile& profile, const std::string& body) {
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
// dependencies need under `profile` (src/rf.hpp, the distances qtxc
// schedules with): every authored stall is kept when it is at least that,
// raised when not; depbars and waits stay as written (I6a).
inline nanosm::Kernel hand_assemble(const std::string& text, const nanosm::DeviceProfile& profile) {
    return nanosm::schedule_fixed_stalls(nanosm::assemble_asm(text, profile), profile);
}

// ---------------------------------------------------------------------------
// LSU (rfc09 B2): interleaved v8, scalar gather/scatter, ldg.uni, the staging
// pool, the backpressure chain, TMA port-A priority, read-stream separation.
// trash/lsu-b2-2026-10-04.md lists which tests replaced which.
// ---------------------------------------------------------------------------

// Coprime periods (31, 29): unique for 899 consecutive words, so a word in
// the wrong lane, register or sector cannot hide behind a repeat.
inline uint32_t pattern_word(uint32_t index) {
    return 0x80000000u | ((index % 31) << 16) | (index % 29);
}

inline void fill_pattern(std::vector<uint8_t>& image, uint32_t byte_address, uint32_t words, uint32_t first_index) {
    for (uint32_t i = 0; i < words; ++i) write_u32(image, byte_address + 4 * i, pattern_word(first_index + i));
}

inline void expect_sim_panic(nanosm::Machine& machine, const std::string& needle, const std::string& what) {
    try {
        machine.run();
    } catch (const nanosm::SimError& error) {
        require(std::string(error.what()).find(needle) != std::string::npos,
            what + " produced the wrong panic: " + error.what());
        return;
    }
    throw std::runtime_error(what + " did not panic");
}

// The RF-starvation setup of the LSU backpressure tests. Before I1 a stream
// of movs (latency 1) booked an async writer's bank every cycle and starved
// it indefinitely. Since I1 (trash/rfc10.md section 3) one denial with service
// owed inhibits new bookings to the bank, so a ready async writer waits at
// most for the bookings already made. To hold one for a while, the stream
// books booking_horizon cycles ahead: fadds whose result reaches the write
// port that far after their issue (I6a: the issue register, Lr and the fadd
// latency set by with_long_bookings, src/rf.hpp), one per cycle, rotating
// over booking_horizon destination registers of `bank`. booking_latency is
// as long as the five-bit stall field allows at canonical RF depths (window
// 1, Lr 2, Wd 2: 31 - 2 - 2 - 1 = 26), less a margin.
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

// A one-dimensional fp32 descriptor of `elements` at gmem_base
// (plain_tma_1d), and the mbar its command completes into, expecting those
// bytes. The buffer and the mbar are the instruction's operands
// (tma_operands_asm).
inline void write_rank1_load(std::vector<uint8_t>& shmem, uint32_t descriptor_address, uint64_t gmem_base,
                      uint32_t elements, uint32_t mbar_address) {
    write_tma_descriptor(shmem, descriptor_address, plain_tma_1d(elements, gmem_base));
    nanosm::MbarState mbar;
    mbar.valid = true;
    mbar.expected = 1;
    mbar.tx_bytes = elements * 4;
    write_u64(shmem, mbar_address, nanosm::encode_mbar_state(mbar));
}

// ---------------------------------------------------------------------------
// B2.1 (rfc09 section 3a): the handoff table edge by edge. Every edge has a
// declared width per cycle; every named stage is a register stage (BRAM
// FIFOs: + their declared latency); no datum crosses two stages in a cycle;
// one write contract; generalized read streams. Observed through the
// machine's edge probe (a test hook, never consulted by the machine).
// trash/lsu-b21-2026-10-04.md.
// ---------------------------------------------------------------------------

using nanosm::LsuEdge;

// The same for either direction (the direction is the instruction's).
inline void write_rank1_tma(std::vector<uint8_t>& shmem, uint32_t descriptor_address,
                     uint64_t gmem_base, uint32_t elements, uint32_t mbar_address) {
    write_tma_descriptor(shmem, descriptor_address, plain_tma_1d(elements, gmem_base));
    nanosm::MbarState mbar;
    mbar.valid = true;
    mbar.expected = 1;
    mbar.tx_bytes = elements * 4;
    write_u64(shmem, mbar_address, nanosm::encode_mbar_state(mbar));
}

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

// ---- Per-engine mbar event registers (design.md Mbar; 2026-10-07) ----

using RmwService = std::pair<uint64_t, std::string>; // start cycle, trace detail

// The RMW engine's service order: every "mbar RMW" slice, by start cycle.
inline std::vector<RmwService> mbar_rmw_services(const nanosm::Machine& machine) {
    std::vector<RmwService> services;
    for (const auto& event : machine.trace().events())
        if (event.name == "mbar RMW") services.emplace_back(event.start_cycle, event.detail);
    std::sort(services.begin(), services.end());
    return services;
}

inline void write_mbar(std::vector<uint8_t>& shmem, uint32_t address, uint16_t remaining,
    int32_t tx_bytes) {
    nanosm::MbarState state;
    state.valid = true;
    state.expected = 1;
    state.remaining = remaining;
    state.tx_bytes = tx_bytes;
    write_u64(shmem, address, nanosm::encode_mbar_state(state));
}

inline void require_mentions(
    const std::string& message, std::initializer_list<std::string_view> needles, const std::string& what);

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

// ---- Race sanitizer: shmem range ownership and tmem ownership ----
// (trash/shmem-ownership-sanitizer-2026-10-03.md)

// Runs the machine and returns its PANIC text; a clean run is a test failure.
inline std::string run_expecting_panic(
    const nanosm::DeviceProfile& profile, const nanosm::Kernel& kernel, nanosm::Launch launch,
    const std::string& what, uint64_t cycles = 20000) {
    nanosm::Machine machine(profile, kernel, std::move(launch));
    try {
        machine.run(cycles);
    } catch (const nanosm::SimError& error) {
        return error.what();
    }
    throw std::runtime_error(what + " did not panic");
}

inline void require_mentions(
    const std::string& message, std::initializer_list<std::string_view> needles,
    const std::string& what) {
    for (const auto needle : needles) {
        if (message.find(needle) == std::string::npos)
            throw std::runtime_error(what + ": diagnostic lacks '" + std::string(needle) +
                "': " + message);
    }
}

// A one-dimensional dense fp32 TMA of `bytes` at gmem 0 (the buffer, the
// mbar and the direction are the instruction's).
inline nanosm::TmaDescriptor linear_tma(uint32_t bytes) {
    return plain_tma_1d(bytes / 4);
}

// An mbar already expecting `tx` bytes and no arrivals: one complete_tx flips it.
inline void write_tx_mbar(std::vector<uint8_t>& shmem, uint32_t address, uint32_t tx) {
    nanosm::MbarState state;
    state.valid = true;
    state.expected = 1;
    state.tx_bytes = tx;
    write_u64(shmem, address, nanosm::encode_mbar_state(state));
}

inline void write_count_mbar(std::vector<uint8_t>& shmem, uint32_t address, uint16_t count) {
    nanosm::MbarState state;
    state.valid = true;
    state.expected = count;
    state.remaining = count;
    write_u64(shmem, address, nanosm::encode_mbar_state(state));
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

inline uint32_t i1_lds_shift(const nanosm::DeviceProfile& profile = canonical_profile()) {
    return profile.shmem_port_a_depth + profile.shmem_read_latency_cycles + nanosm::issue_to_dispatch(profile);
}

// One run, tick by tick: the cycle in which each watched register of warp 0
// changed (lane 0), and `observe(cycle)` after every tick.
inline std::map<uint32_t, std::vector<uint64_t>> i1_run(nanosm::Machine& machine, const std::vector<uint32_t>& regs,
    const std::function<void(uint64_t)>& observe = {}) {
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

// ---------------------------------------------------------------------------
// I2: shmem services and service slots (trash/rfc12.md sections 1-5 and 13;
// trash/i2-portA-slots-2026-10-11.md). Exact timelines are hand assembly
// (i1_asm) under a debug trace, where every granted round is an instant
// "<service> grant" (detail = client, value = slot). Several timelines use an
// engine event the host places into the TMA's event register
// (host_rmw_event). I3 re-derived every one of them: in I2 that RMW held
// A-read AND A-write from its read to its write; since I3 (rfc12 section 6)
// it pauses only new STORE transactions, from its start to its write grant,
// and holds A-read only while it waits for its read (i3_with_blocker below).
// ---------------------------------------------------------------------------

// A read's words reach the port output D + Lb cycles after its grant.
inline uint32_t i2_return(const nanosm::DeviceProfile& profile) {
    return profile.shmem_port_a_depth + profile.shmem_read_latency_cycles;
}

// I6a: issue to dependent issue behind a one-cycle ALU result (mov, sread,
// shli, ...): visible issue_to_visible after its producer's issue, sampled
// issue_register_stages after the consumer's (src/rf.hpp; 6 at canonical).
// The hand-assembled timelines below are built from it and from
// issue_to_dispatch (an lds / sts / mbar operation reaches its slot or
// register that long after its issue, and is served from the next cycle).
inline uint64_t alu_distance(const nanosm::DeviceProfile& profile) {
    nanosm::Instruction mov;
    mov.opcode = nanosm::Opcode::mov;
    return nanosm::issue_to_visible(mov, profile) - nanosm::issue_register_stages;
}

struct I2Grant {
    uint64_t cycle = 0;
    std::string service; // "A-read", "A-write", "B-write"
    std::string client;  // "simt", "probe", "rmw", "tma"
    uint32_t slot = 0;
    bool operator==(const I2Grant&) const = default;
};

inline std::string describe(const std::vector<I2Grant>& grants) {
    std::string text;
    for (const auto& grant : grants) {
        text += " c" + std::to_string(grant.cycle) + ":" + grant.service + "/" + grant.client +
            (grant.client == "simt" ? std::to_string(grant.slot) : "");
    }
    return text;
}

// The rounds granted (debug traces), by cycle; one service, or all.
inline std::vector<I2Grant> i2_grants(const nanosm::Machine& machine, const std::string& service = {}) {
    std::vector<I2Grant> grants;
    for (const auto& event : machine.trace().events()) {
        if (event.category != nanosm::TraceCategory::shmem || event.kind != nanosm::TraceEventKind::instant) continue;
        const auto name = event.name.substr(0, event.name.find(" grant"));
        if (!service.empty() && name != service) continue;
        grants.push_back({event.start_cycle, name, event.detail, static_cast<uint32_t>(event.value)});
    }
    std::stable_sort(grants.begin(), grants.end(), [](const auto& a, const auto& b) { return a.cycle < b.cycle; });
    return grants;
}

inline constexpr uint32_t i2_event_mbar = 4096; // a TMA-completion mbar for the host's events

// shmem words 0..2047 hold 0x100 + word; an mbar at i2_event_mbar waits for
// 64 transaction bytes (one host event completes its phase).
inline std::unique_ptr<nanosm::Machine> i2_machine(const nanosm::DeviceProfile& profile, const std::string& program,
    uint32_t warps = 1, bool hand_assembly = true) {
    nanosm::Launch launch;
    launch.active_warps = warps;
    auto& shmem = image_bytes(launch.shmem_image, 8192);
    for (uint32_t word = 0; word < 2048; ++word) write_u32(shmem, 4 * word, 0x100 + word);
    nanosm::MbarState mbar;
    mbar.valid = true;
    mbar.expected = 1;
    mbar.tx_bytes = 64;
    write_u64(shmem, i2_event_mbar, nanosm::encode_mbar_state(mbar));
    const auto kernel = hand_assembly ? hand_assemble(i1_asm(profile, program), profile)
                                      : compile(program, profile);
    return std::make_unique<nanosm::Machine>(profile, kernel, std::move(launch),
        nanosm::TraceOptions{.timeline = true, .debug = true});
}

// The host's event (call between ticks): a complete_tx of 64 bytes into the
// TMA's event register, held from the next cycle; it completes
// i2_event_mbar's phase. (Was i2_only_hold_both_a_services.) Since I3 an RMW
// STARTS as soon as the engine is idle and A-write's alternation lets it (no
// store waiting for its turn): from that cycle to its write grant the store
// pause holds every new store transaction; A-read is held only while the RMW
// waits for its read -- behind a store transaction under way on A-write
// (i3_with_blocker) or an A-read transaction under way -- and serves everyone
// from the read on.
inline void host_rmw_event(nanosm::Machine& machine) {
    machine.complete_mbar_tx(i2_event_mbar, 64, machine.read_mbar_state(i2_event_mbar).phase ? 1 : 0);
}

inline void i2_run_until(nanosm::Machine& machine, uint64_t cycle) {
    while (machine.cycle() < cycle) tick_capped(machine);
}

// ---------------------------------------------------------------------------
// I2.1: follow-ups to the port-A slice (trash/i2-1-followups-2026-10-11.md;
// Fable's review trash/i1-i2-review-fable-2026-10-11.md F1-F8, and the TMA
// store-register derivation in trash/chan.txt). Timelines are hand assembly
// under a debug trace, as in the I2 section.
// ---------------------------------------------------------------------------

// An I2-style machine (shmem words 0..2047 hold 0x100 + word) whose image
// also holds the given mbars.
inline std::unique_ptr<nanosm::Machine> i21_machine(const nanosm::DeviceProfile& profile, const std::string& program,
    const std::vector<std::pair<uint32_t, nanosm::MbarState>>& mbars, uint32_t warps = 1) {
    nanosm::Launch launch;
    launch.active_warps = warps;
    auto& shmem = image_bytes(launch.shmem_image, 8192);
    for (uint32_t word = 0; word < 2048; ++word) write_u32(shmem, 4 * word, 0x100 + word);
    for (const auto& [address, state] : mbars) write_u64(shmem, address, nanosm::encode_mbar_state(state));
    return std::make_unique<nanosm::Machine>(profile, hand_assemble(i1_asm(profile, program), profile),
        std::move(launch), nanosm::TraceOptions{.timeline = true, .debug = true});
}

inline nanosm::MbarState i21_mbar(uint16_t expected, uint16_t remaining, int32_t tx_bytes = 0, bool phase = false) {
    nanosm::MbarState state;
    state.valid = true;
    state.phase = phase;
    state.expected = expected;
    state.remaining = remaining;
    state.tx_bytes = tx_bytes;
    return state;
}

// An optional profile field, I2 style: the canonical JSON declares
// `canonical`; absent, it defaults to the same value (same fingerprint); 0,
// negative, fractional and 65 are rejected by parsing and 0 by validation;
// another value changes the fingerprint.
inline void i21_check_optional_field(const char* section, const char* field, uint32_t canonical,
    uint32_t nanosm::DeviceProfile::*member) {
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
    for (const auto value : {nlohmann::json(0), nlohmann::json(-1), nlohmann::json(1.5), nlohmann::json(65)}) {
        auto modified = document;
        modified[section][field] = value;
        try {
            (void)nanosm::parse_device_profile(modified.dump());
            throw std::runtime_error(name + " = " + value.dump() + " was accepted");
        } catch (const nanosm::ProfileError&) {}
    }
    auto zero = canonical_profile();
    zero.*member = 0;
    try {
        nanosm::validate_device_profile(zero);
        throw std::runtime_error(name + " = 0 passed validation");
    } catch (const nanosm::ProfileError&) {}
    auto other = canonical_profile();
    other.*member = canonical + 1;
    require(nanosm::device_profile_fingerprint(other) != nanosm::device_profile_fingerprint(canonical_profile()),
        name + " was omitted from the profile fingerprint");
}

// ---------------------------------------------------------------------------
// I3: the RMW engine's global store pause, the RMW / store alternation bit,
// fair admission between SIMT RMWs and engine events, and the signed mbar tx
// count (trash/rfc12.md sections 5, 6 and 8;
// trash/i3-rmw-pause-signed-tx-2026-10-11.md). Hand assembly under a debug
// trace, as in the I2 sections. An RMW's START is its "mbar RMW" slice's
// start (the store pause begins at that cycle's edge); its read and write are
// the "rmw" grants on A-read and A-write.
// ---------------------------------------------------------------------------

struct I3Rmw {
    uint64_t start = 0;
    uint64_t read = 0;
    uint64_t write = 0;
    std::string detail;
};

inline std::string describe(const std::vector<I3Rmw>& rmws) {
    std::string text;
    for (const auto& rmw : rmws) {
        text += " {" + rmw.detail + ": start c" + std::to_string(rmw.start) + ", read c" + std::to_string(rmw.read) +
            ", write c" + std::to_string(rmw.write) + "}";
    }
    return text;
}

// A one-dimensional TMA descriptor at 5120 of `bytes` at gmem 0, and the mbar
// at 4104 expecting those bytes. The kernel's tma.load.1d / tma.store.1d
// names the buffer and that mbar in uniform registers (i4_tma_operands).
inline std::function<void(std::vector<uint8_t>&)> i4_tma(uint32_t bytes) {
    return [=](std::vector<uint8_t>& shmem) {
        write_tma_descriptor(shmem, 5120, linear_tma(bytes));
        write_tx_mbar(shmem, 4104, bytes);
    };
}

// An i4_tma command's operands, its buffer at shmem `shared`: ur10 the
// buffer, ur11 the mbar (4104), ur8 the coordinate (0). For
// "tma.load.1d [ur10], [5120, ur8], [ur11]" (or the tma.store.1d form).
inline std::string i4_tma_operands(uint32_t shared) {
    return tma_operands_asm(10, shared, 11, 4104, 8);
}

// The machine's PANIC text (a clean run is a test failure).
inline std::string i4_panic(nanosm::Machine& machine, const std::string& what) {
    try {
        machine.run();
    } catch (const nanosm::SimError& error) {
        return error.what();
    }
    throw std::runtime_error(what + " did not panic");
}

// ---------------------------------------------------------------------------
// I5: uniform shared atomics (trash/rfc13.md; trash/i5r-uniform-atomics-
// 2026-10-12.md), backlog D12 (machine quiet, sanitizer PANIC) and the C10
// follow-up (launch-declared mbars are live; C11: their addresses are checked
// before narrowing). Hand assembly under a debug trace, as in the I2 - I4
// sections. An atomic is an RMW-engine operation: its "atomic RMW" slice runs
// from its START (the store pause begins) to its write grant + 1; inside it,
// its read and its write are the "rmw" grants on A-read and A-write.
// ---------------------------------------------------------------------------

struct I5Atomic {
    uint64_t start = 0;
    uint64_t end = 0;             // the write grant + 1
    std::vector<uint64_t> reads;  // its read grant
    std::vector<uint64_t> writes; // its write grant
    std::string detail;
};

inline std::string describe(const I5Atomic& atomic) {
    std::string text = " {" + atomic.detail + ": start c" + std::to_string(atomic.start) + ", end c" +
        std::to_string(atomic.end) + ", reads";
    for (const auto cycle : atomic.reads) text += " " + std::to_string(cycle);
    text += ", writes";
    for (const auto cycle : atomic.writes) text += " " + std::to_string(cycle);
    return text + "}";
}

// I2-style shmem (word w holds 0x100 + w), a debug trace.
inline std::unique_ptr<nanosm::Machine> i6a_machine(
    const nanosm::DeviceProfile& profile, const nanosm::Kernel& kernel, uint32_t warps = 1) {
    nanosm::Launch launch;
    launch.active_warps = warps;
    auto& shmem = image_bytes(launch.shmem_image, 8192);
    for (uint32_t word = 0; word < 2048; ++word) write_u32(shmem, 4 * word, 0x100 + word);
    return std::make_unique<nanosm::Machine>(profile, kernel, std::move(launch),
        nanosm::TraceOptions{.timeline = true, .debug = true});
}

// Warp 0's debug RF events of one register (its warp track): "RF read"
// samples or "RF write" port presentations, by cycle.
inline std::vector<uint64_t> i6a_rf_events(const nanosm::Machine& machine, const std::string& name, uint8_t reg) {
    std::vector<uint64_t> cycles;
    for (const auto& event : machine.trace().events()) {
        if (event.category == nanosm::TraceCategory::instruction && event.name == name && event.id == reg &&
            event.track == 100) {
            cycles.push_back(event.start_cycle);
        }
    }
    return cycles;
}

} // namespace nanosm_test

#include "sim.hpp"

#include <cstring>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void write_u32(std::vector<std::uint8_t>& image, std::uint32_t address, std::uint32_t value) {
    std::memcpy(image.data() + address, &value, sizeof(value));
}

nanosm::Kernel compile(std::string_view source) {
    return nanosm::compile_and_assemble(source);
}

void test_encoding_round_trip() {
    nanosm::Instruction inst;
    inst.opcode = nanosm::Opcode::lds;
    inst.control.stall = 21; // needs the fifth stall bit
    inst.control.set_depbar = 4;
    inst.control.wait_mask = 0b100101;
    inst.dst = 47;
    inst.src0 = 33;
    inst.src1 = 12;
    inst.imm = -12345;
    const auto decoded = nanosm::decode(nanosm::encode(inst));
    require(decoded.opcode == inst.opcode, "opcode encoding mismatch");
    require(decoded.control.stall == 21, "stall encoding mismatch");
    require(decoded.control.set_depbar == 4, "set-depbar encoding mismatch");
    require(decoded.control.wait_mask == 0b100101, "wait-mask encoding mismatch");
    require(decoded.dst == 47 && decoded.src0 == 33 && decoded.src1 == 12, "register encoding mismatch");
    require(decoded.imm == -12345, "signed immediate encoding mismatch");
}

void test_compiler_controls() {
    const auto program = nanosm::compile_qtx(R"(
        mov  r0, 0
        lds  r1, [r0]
        lds  r2, [r0 + 64]
        fadd r4, r1, r2
        sts  [r0 + 128], r4
        halt
    )");
    require(program.instructions[0].control.stall == 0, "mov stalled without a close consumer");
    require(program.instructions[1].control.set_depbar == 0, "first load did not allocate sb0");
    require(program.instructions[2].control.set_depbar == 1, "second load did not allocate sb1");
    require(program.instructions[3].control.wait_mask == 0b000011, "fadd did not wait for both loads");
    require(program.instructions[3].control.stall == 4, "fadd did not cover its one-away consumer");
    require(program.instructions[4].control.set_depbar == 0, "store did not reuse a released depbar");
    require(program.instructions[5].control.wait_mask == 0b000001, "halt did not wait for store completion");
}

void test_stalls_track_consumer_distance() {
    // fadd's result is consumed three instructions later, so of its 4-cycle latency
    // only one cycle must be bought with a stall, and it lands just before the consumer.
    const auto program = nanosm::compile_qtx(R"(
        mov  r0, 1
        fadd r1, r0, r0
        mov  r2, 5
        mov  r3, 6
        add  r4, r1, r2
        halt
    )");
    require(program.instructions[1].control.stall == 0, "fadd stalled its own latency");
    require(program.instructions[2].control.stall == 0, "independent mov stalled");
    require(program.instructions[3].control.stall == 2, "pre-consumer stall wrong");
    nanosm::Launch launch;
    launch.active_warps = 1;
    launch.shmem_image.resize(4, 0);
    nanosm::Machine machine(
        {}, nanosm::assemble_asm(nanosm::emit_asm(program)), std::move(launch));
    machine.run(); // the hazard sanitizer accepts the minimal schedule
}

void test_vadd() {
    constexpr std::uint32_t n = 64;
    constexpr std::uint32_t b = n * 4;
    constexpr std::uint32_t c = n * 8;
    nanosm::Config config;
    nanosm::Launch launch;
    launch.active_warps = n / config.lanes;
    launch.shmem_image.resize(c + n * 4, 0);
    for (std::uint32_t i = 0; i < n; ++i) {
        write_u32(launch.shmem_image, i * 4, nanosm::float_bits(static_cast<float>(i)));
        write_u32(launch.shmem_image, b + i * 4, nanosm::float_bits(static_cast<float>(2 * i)));
    }
    nanosm::Machine machine(
        config, nanosm::compile_and_assemble_file("kernels/vadd.qtx"), std::move(launch));
    machine.run();
    for (std::uint32_t i = 0; i < n; ++i) {
        require(
            nanosm::bits_float(machine.read_shmem_u32(c + i * 4)) == static_cast<float>(3 * i),
            "vadd output mismatch at " + std::to_string(i));
    }
}

void test_lds_is_nonblocking_and_depbar_waits() {
    const auto program = nanosm::compile_qtx(R"(
        mov  r0, 0
        lds  r1, [r0]
        lds  r2, [r0 + 64]
        fadd r4, r1, r2
        halt
    )");
    nanosm::Launch launch;
    launch.active_warps = 1;
    launch.shmem_image.resize(128, 0);
    write_u32(launch.shmem_image, 0, nanosm::float_bits(2.0f));
    write_u32(launch.shmem_image, 64, nanosm::float_bits(3.0f));
    nanosm::Machine machine({}, nanosm::assemble_asm(nanosm::emit_asm(program)), std::move(launch));
    machine.run();

    std::vector<std::uint64_t> load_issue_cycles;
    bool saw_depbar_wait = false;
    for (const auto& event : machine.trace()) {
        if (event.message.find("issue w0") == 0 && event.message.find("lds") != std::string::npos) {
            load_issue_cycles.push_back(event.cycle);
        }
        saw_depbar_wait |= event.message.find("depbar-wait") != std::string::npos;
    }
    require(load_issue_cycles.size() == 2, "did not issue two loads");
    require(load_issue_cycles[1] == load_issue_cycles[0] + 1, "lds blocked the warp instead of retiring");
    require(saw_depbar_wait, "consumer never stalled on its depbar mask");
}

void test_missing_wait_panics_instead_of_scoreboarding() {
    const auto kernel = nanosm::assemble_asm(R"(
        mov  r0, 0 {pred=PT, stall=1, set=-, wait=-}
        lds  r1, [r0] {pred=PT, stall=0, set=sb0, wait=-}
        fadd r4, r1, r2 {pred=PT, stall=4, set=-, wait=-}
        halt {pred=PT, stall=0, set=-, wait=sb0}
    )");
    nanosm::Launch launch;
    launch.active_warps = 1;
    launch.shmem_image.resize(64, 0);
    nanosm::Machine machine({}, kernel, std::move(launch));
    try {
        machine.run();
        throw std::runtime_error("missing depbar wait was silently scoreboard-stalled");
    } catch (const nanosm::SimError& error) {
        require(
            std::string(error.what()).find("RF RAW hazard") != std::string::npos,
            "missing wait produced the wrong panic");
    }
}

void test_broadcast_is_one_round() {
    nanosm::Launch launch;
    launch.active_warps = 1;
    launch.shmem_image.resize(4);
    write_u32(launch.shmem_image, 0, 0x12345678u);
    nanosm::Machine machine({}, compile(R"(
        mov  r0, 0
        lds  r1, [r0]
        halt
    )"), std::move(launch));
    machine.run();
    for (std::uint32_t lane = 0; lane < 16; ++lane) {
        require(machine.read_reg(0, lane, 1) == 0x12345678u, "broadcast value mismatch");
    }
    std::size_t rounds = 0;
    for (const auto& event : machine.trace()) rounds += event.message.starts_with("shmem.portA") ? 1 : 0;
    require(rounds == 1, "broadcast did not use exactly one bank round");
}

void test_bank_conflict_serializes() {
    nanosm::Launch launch;
    launch.active_warps = 1;
    launch.shmem_image.resize(16 * 64, 0);
    for (std::uint32_t lane = 0; lane < 16; ++lane) write_u32(launch.shmem_image, lane * 64, lane + 10);
    nanosm::Machine machine({}, compile(R"(
        sread r0, LANE_ID
        shli  r0, r0, 6
        lds   r1, [r0]
        halt
    )"), std::move(launch));
    machine.run();
    std::size_t rounds = 0;
    for (const auto& event : machine.trace()) rounds += event.message.starts_with("shmem.portA") ? 1 : 0;
    require(rounds == 16, "sixteen-way bank conflict did not serialize to sixteen rounds");
}

void test_write_collision_panics() {
    nanosm::Launch launch;
    launch.active_warps = 1;
    launch.shmem_image.resize(4, 0);
    nanosm::Machine machine({}, compile(R"(
        mov r0, 0
        mov r1, 7
        sts [r0], r1
        halt
    )"), std::move(launch));
    try {
        machine.run();
        throw std::runtime_error("same-word write collision did not panic");
    } catch (const nanosm::SimError& error) {
        require(
            std::string(error.what()).find("multiple lanes write the same shmem word") != std::string::npos,
            "write collision produced the wrong panic");
    }
}

} // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"encoding round trip", test_encoding_round_trip},
        {"compiler controls", test_compiler_controls},
        {"vadd", test_vadd},
        {"nonblocking lds", test_lds_is_nonblocking_and_depbar_waits},
        {"missing wait panic", test_missing_wait_panics_instead_of_scoreboarding},
        {"broadcast", test_broadcast_is_one_round},
        {"bank conflict", test_bank_conflict_serializes},
        {"write collision", test_write_collision_panics},
    };
    std::size_t passed = 0;
    for (const auto& [name, test] : tests) {
        try {
            test();
            std::cout << "PASS " << name << '\n';
            ++passed;
        } catch (const std::exception& error) {
            std::cerr << "FAIL " << name << ": " << error.what() << '\n';
        }
    }
    std::cout << passed << "/" << tests.size() << " tests passed\n";
    return passed == tests.size() ? 0 : 1;
}

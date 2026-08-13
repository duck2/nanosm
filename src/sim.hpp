#pragma once

#include "compiler.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace nanosm {

struct Config {
    std::uint32_t lanes{16};
    std::uint32_t max_warps{8};
    std::uint32_t registers{64};
    std::uint32_t shmem_bytes{64 * 1024};
    std::uint32_t shmem_banks{16};
    Timing timing{};
};

struct Launch {
    std::uint32_t active_warps{};
    std::vector<std::uint8_t> shmem_image;
};

struct TraceEvent {
    std::uint64_t cycle{};
    std::string message;
};

class Machine {
public:
    Machine(Config config, Kernel kernel, Launch launch);

    void tick();
    void run(std::uint64_t cycle_limit = 100000);

    [[nodiscard]] bool done() const;
    [[nodiscard]] std::uint64_t cycle() const { return cycle_; }
    [[nodiscard]] const std::vector<TraceEvent>& trace() const { return trace_; }
    [[nodiscard]] const std::vector<std::uint8_t>& shmem() const { return shmem_; }

    [[nodiscard]] std::uint32_t read_shmem_u32(std::uint32_t byte_address) const;
    [[nodiscard]] std::uint32_t read_reg(
        std::uint32_t warp, std::uint32_t lane, std::uint32_t reg) const;

private:
    struct Warp {
        std::uint32_t id{};
        std::uint32_t pc{}; // byte PC; every decoded instruction still occupies one 64-bit slot
        bool halted{};
        std::uint64_t stall_until{};
        std::array<std::uint32_t, 6> depbar{};
        std::vector<std::uint32_t> rf;
        std::vector<std::uint64_t> ready_cycle;
    };

    struct Writeback {
        std::uint64_t ready_cycle{};
        std::uint32_t warp{};
        std::optional<std::uint8_t> dst;
        std::vector<std::uint32_t> values;
        std::optional<std::uint8_t> clear_depbar;
        std::string reason;
    };

    struct LaneAccess {
        std::uint32_t lane{};
        std::uint32_t byte_address{};
        std::uint32_t value{};
    };

    struct ShmemRequest {
        bool load{};
        std::uint32_t warp{};
        std::uint8_t dst{};
        std::uint8_t depbar{};
        std::vector<std::vector<LaneAccess>> rounds;
        std::size_t next_round{};
        std::vector<std::uint32_t> load_values;
    };

    struct IssueAttempt {
        bool issued{};
        std::string reason;
    };

    Config config_;
    Kernel kernel_;
    std::vector<Warp> warps_;
    std::vector<std::uint8_t> shmem_;
    std::optional<ShmemRequest> port_a_;
    std::deque<Writeback> writebacks_;
    std::map<std::uint64_t, std::array<bool, 3>> write_reservations_;
    std::vector<TraceEvent> trace_;
    std::uint64_t cycle_{};
    std::uint32_t scheduler_cursor_{};

    void commit_writebacks();
    void advance_shmem();
    void issue_one();
    IssueAttempt try_issue(Warp& warp);

    [[nodiscard]] Instruction instruction(const Warp& warp) const;
    [[nodiscard]] std::vector<std::uint8_t> source_registers(const Instruction& inst) const;
    [[nodiscard]] std::optional<std::uint8_t> destination_register(const Instruction& inst) const;
    void check_register_hazards(const Warp& warp, const Instruction& inst) const;
    void check_rf_read_ports(const Instruction& inst) const;
    [[nodiscard]] bool reserve_write(std::uint8_t reg, std::uint64_t at_cycle);

    [[nodiscard]] std::uint32_t& rf(Warp& warp, std::uint32_t lane, std::uint32_t reg);
    [[nodiscard]] std::uint32_t rf(const Warp& warp, std::uint32_t lane, std::uint32_t reg) const;
    [[nodiscard]] std::vector<std::vector<LaneAccess>> build_shmem_rounds(
        const Warp& warp, const Instruction& inst, bool load) const;

    void schedule_register_write(
        Warp& warp,
        std::uint8_t dst,
        std::uint64_t ready_cycle,
        std::vector<std::uint32_t> values,
        std::string reason);
    void trace(std::string message);
    [[noreturn]] void panic(const Warp& warp, const Instruction& inst, const std::string& reason) const;
};

std::uint32_t float_bits(float value);
float bits_float(std::uint32_t value);

} // namespace nanosm

#pragma once

#include "compiler.hpp"
#include "profile.hpp"
#include "trace.hpp"

#include <array>
#include <cstddef>
#include <stdint.h>
#include <deque>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace nanosm {

struct ImageSegment {
    uint64_t address = 0;
    std::vector<uint8_t> bytes;
};

struct Launch {
    uint32_t active_warps = 0;
    std::vector<ImageSegment> shmem_image;
    std::vector<ImageSegment> gmem_image;
};

class Machine {
public:
    Machine(Config config, Kernel kernel, Launch launch, TraceOptions trace_options = {});

    void tick();
    void run(uint64_t cycle_limit = 100000);

    [[nodiscard]] bool done() const;
    [[nodiscard]] uint64_t cycle() const { return cycle_; }
    [[nodiscard]] const TraceRecorder& trace() const { return *trace_; }
    [[nodiscard]] const std::vector<uint8_t>& shmem() const { return shmem_; }
    [[nodiscard]] const std::vector<uint8_t>& gmem() const { return gmem_; }

    [[nodiscard]] uint32_t read_shmem_u32(uint32_t byte_address) const;
    [[nodiscard]] uint32_t read_gmem_u32(uint64_t byte_address) const;
    [[nodiscard]] uint32_t read_reg(
        uint32_t warp, uint32_t lane, uint32_t reg) const;

private:
    struct Warp {
        uint32_t id = 0;
        uint32_t pc = 0; // byte PC; every decoded instruction still occupies one 64-bit slot
        bool halted = false;
        uint64_t stall_until = 0;
        std::array<uint32_t, 6> depbar{};
        bool outstanding_ldg = false;
        std::vector<uint32_t> rf;
        std::vector<uint64_t> ready_cycle;
        std::array<std::vector<uint8_t>, 7> predicates;
        std::array<uint64_t, 7> predicate_ready_cycle{};
    };

    struct Writeback {
        uint64_t ready_cycle = 0;
        uint32_t warp = 0;
        std::optional<uint8_t> dst;
        std::optional<uint8_t> predicate_dst;
        std::vector<uint32_t> values;
        std::optional<uint8_t> clear_depbar;
        bool clear_outstanding_ldg = false;
        std::string reason;
    };

    struct LaneAccess {
        uint32_t lane = 0;
        uint64_t byte_address = 0;
        uint32_t value = 0;
    };

    struct ShmemRequest {
        bool load = false;
        uint32_t warp = 0;
        uint8_t dst = 0;
        uint8_t depbar = 0;
        std::vector<std::vector<LaneAccess>> rounds;
        std::size_t next_round = 0;
        std::vector<uint32_t> load_values;
    };

    struct GmemSectorRequest {
        uint64_t id = 0;
        uint64_t command = 0;
        uint64_t sector_address = 0;
        uint64_t ready_dram_tick = 0;
        bool load = false;
        std::vector<LaneAccess> accesses;
        uint32_t beats_remaining = 0;
    };

    struct GmemCommand {
        bool load = false;
        uint32_t warp = 0;
        uint8_t dst = 0;
        uint8_t depbar = 0;
        uint32_t remaining_sectors = 0;
        std::vector<uint32_t> load_values;
    };

    struct IssueAttempt {
        bool issued = false;
        WarpState state = WarpState::structural;
        std::string reason;
    };

    Config config_;
    Kernel kernel_;
    std::vector<Warp> warps_;
    std::vector<uint8_t> shmem_;
    std::vector<uint8_t> gmem_;
    std::optional<ShmemRequest> port_a_;
    std::deque<GmemSectorRequest> gmem_queue_;
    std::optional<GmemSectorRequest> gmem_bus_;
    std::map<uint64_t, GmemCommand> gmem_commands_;
    std::deque<Writeback> writebacks_;
    std::map<uint64_t, std::vector<uint32_t>> write_reservations_;
    std::optional<TraceRecorder> trace_;
    uint64_t cycle_ = 0;
    uint64_t dram_tick_ = 0;
    uint64_t dram_phase_ = 0;
    uint32_t scheduler_cursor_ = 0;
    uint64_t next_gmem_request_id_ = 0;
    uint64_t next_gmem_command_id_ = 0;

    void commit_writebacks();
    void advance_shmem();
    void advance_dram_clock();
    void tick_gmem_controller();
    void issue_one();
    IssueAttempt try_issue(Warp& warp);

    [[nodiscard]] Instruction instruction(const Warp& warp) const;
    [[nodiscard]] std::vector<uint8_t> source_registers(const Instruction& inst) const;
    [[nodiscard]] std::optional<uint8_t> destination_register(const Instruction& inst) const;
    void check_register_hazards(const Warp& warp, const Instruction& inst) const;
    void check_rf_read_ports(const Instruction& inst) const;
    [[nodiscard]] bool reserve_write(uint8_t reg, uint64_t at_cycle);

    [[nodiscard]] uint32_t& rf(Warp& warp, uint32_t lane, uint32_t reg);
    [[nodiscard]] uint32_t rf(const Warp& warp, uint32_t lane, uint32_t reg) const;
    [[nodiscard]] std::vector<std::vector<LaneAccess>> build_shmem_rounds(
        const Warp& warp, const Instruction& inst, bool load) const;
    [[nodiscard]] std::vector<std::vector<LaneAccess>> build_gmem_sectors(
        const Warp& warp, const Instruction& inst, bool load) const;
    [[nodiscard]] std::size_t gmem_outstanding_sectors() const;
    void complete_gmem_sector(GmemSectorRequest request);
    void apply_image(
        std::vector<uint8_t>& memory,
        const std::vector<ImageSegment>& image,
        const std::string& name);

    void schedule_register_write(
        Warp& warp,
        uint8_t dst,
        uint64_t ready_cycle,
        std::vector<uint32_t> values,
        std::string reason);
    void schedule_predicate_write(
        Warp& warp,
        uint8_t dst,
        uint64_t ready_cycle,
        std::vector<uint32_t> values);
    [[noreturn]] void panic(const Warp& warp, const Instruction& inst, const std::string& reason) const;
};

uint32_t float_bits(float value);
float bits_float(uint32_t value);

} // namespace nanosm

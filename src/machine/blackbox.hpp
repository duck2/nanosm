#pragma once

// A black box: a unit modelled opaquely, behind a wall.
//
// The machine's state is of two kinds. In the open: registers, groups,
// pipelines and FIFOs (reg.hpp, regs.hpp, pipe.hpp, fifo.hpp), whose payloads
// are plain data and whose every member has one writer a cycle. Behind a
// wall: black boxes. A black box's inside is ordinary code with no rules --
// it keeps what it likes in whatever C++ suits it (an arithmetic unit
// computes a whole result at once and holds it; nothing here builds its
// stages from gates) -- and its outside is a contract:
//   inputs     taken from registers (their Q) in the cycle the black box
//              takes them, each named with its bits and the register that
//              holds it, which is in the open where it is;
//   outputs    the black box's own registers, clocked with every other one:
//              a value whose inputs were taken at t is in the output register
//              at t + latency, latency >= 1, so nothing passes through a black
//              box in one cycle;
//   latency    fixed, from the profile's delays;
//   interval   a new input at most every `interval` cycles;
//   operations an arithmetic unit's: the instructions it executes (the
//              opcodes src/isa.hpp's table gives that unit).
// What it costs is not part of it. The state report prints every black box's
// contract.
// The memories (memory.hpp) are black boxes by the same contract --
// registered ports, a declared read delay, a shape from the profile -- so they
// are declared in the same list (memory_blackbox below, each memory's own
// declaration beside it in machine.hpp), and the words in flight during a
// read delay are the memory's business.

#include "machine/memory.hpp"
#include "machine/reg.hpp"

#include <bit>
#include <deque>
#include <optional>
#include <stdint.h>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace nanosm {

// Bits to tell `n` things apart (0 for one or none).
inline uint64_t bits_for(uint64_t n) { return n <= 1 ? 0 : static_cast<uint64_t>(std::bit_width(n - 1)); }

// An input or an output: its bits and the register that holds it.
struct BlackBoxPort {
    std::string name;
    uint64_t bits = 0;
    std::string held_in;
};

// A black box's contract, declared where it is defined.
struct BlackBoxDeclaration {
    explicit BlackBoxDeclaration(std::string name_) : name(std::move(name_)) {}

    std::string name;
    Unit unit = Unit::none;
    uint32_t latency = 1;
    std::string latency_from; // the profile's delays it is made of
    uint32_t interval = 1;
    std::vector<BlackBoxPort> inputs;
    std::vector<BlackBoxPort> outputs;
    std::string operations; // an arithmetic unit's, by name (none: not an arithmetic unit)
};

// A black box's output register, and what the inside has finished and not
// yet output. The inside delivers a value with the cycle its inputs were
// taken; the register takes it at the edge before taken + latency and holds
// it for that one cycle (nothing delivered: empty). Inside: whatever else the
// black box keeps (std::monostate: nothing); a black box that keeps anything
// there is built with the question "is it idle?" about it, so that empty()
// covers all its work.
template <typename Out, typename Inside = std::monostate>
class BlackBox final : public Clocked {
public:
    using InsideIdle = bool (*)(const Inside&);

    BlackBox(AllClocked& all, Unit unit, const std::string& name)
        : Clocked(all, unit, name + ": output register") {
        static_assert(FixedPayload<Out>, "a black box's output register holds a fixed bundle of bits (machine/state.hpp)");
    }

    // Its contract, once, when the machine is built; with the inside's idle
    // question when it keeps anything.
    void build(const BlackBoxDeclaration& declaration, InsideIdle inside_idle = nullptr) {
        if (built_) throw SimError("internal: black box '" + declaration.name + "' built twice");
        if (declaration.latency == 0 || declaration.interval == 0) {
            throw SimError("internal: black box '" + declaration.name + "' built with no latency or interval");
        }
        if (!std::is_same_v<Inside, std::monostate> && inside_idle == nullptr) {
            throw SimError("internal: black box '" + declaration.name + "' keeps an inside and was built without "
                "asking whether it is idle");
        }
        built_ = true;
        black_box_ = declaration.name;
        latency_ = declaration.latency;
        interval_ = declaration.interval;
        inside_idle_ = inside_idle;
    }
    [[nodiscard]] uint32_t latency() const { return latency_; }

    // The inside hands over a value whose inputs it took at cycle
    // `taken_at`: it is in the output register at taken_at + latency. At most
    // once per tick; inputs no closer than the interval. A refused value is
    // not a write.
    void deliver(Out value, uint64_t taken_at, const char* tag) {
        if (!built_) throw SimError("internal: a value delivered to a black box not yet built ('" + name() + "')");
        const uint64_t at = taken_at + latency_;
        if (taken_at > cycle() || at <= cycle() || (last_taken_.has_value() && taken_at < *last_taken_ + interval_)) {
            throw SimError("internal: black box '" + black_box_ + "' given a value taken at cycle " +
                std::to_string(taken_at) + " at cycle " + std::to_string(cycle()) + " (latency " +
                std::to_string(latency_) + ", interval " + std::to_string(interval_) + ")");
        }
        note_write(tag, delivered_);
        delivered_ = true;
        last_taken_ = taken_at;
        in_flight_.emplace_back(at, std::move(value));
    }

    // Q of the output register.
    [[nodiscard]] const std::optional<Out>& output() const { return q_; }
    // Nothing inside -- delivered or kept -- and nothing at the output (the
    // end of a run asks).
    [[nodiscard]] bool empty() const {
        return !q_.has_value() && in_flight_.empty() && (inside_idle_ == nullptr || inside_idle_(inside_));
    }
    // The black box's own code only.
    [[nodiscard]] Inside& inside() { return inside_; }

    void commit() override {
        delivered_ = false;
        q_.reset();
        if (in_flight_.empty()) return;
        if (in_flight_.front().first <= cycle()) {
            throw SimError("internal: black box '" + black_box_ + "' missed its output cycle");
        }
        if (in_flight_.front().first == cycle() + 1) {
            q_ = std::move(in_flight_.front().second);
            in_flight_.pop_front();
        }
    }

    [[nodiscard]] StateSize size() const override { return sized("black box", 1); }

private:
    std::string black_box_; // its name
    uint32_t latency_ = 1;
    uint32_t interval_ = 1;
    bool built_ = false;
    InsideIdle inside_idle_ = nullptr;
    std::optional<Out> q_;
    std::deque<std::pair<uint64_t, Out>> in_flight_; // (its output cycle, the value)
    std::optional<uint64_t> last_taken_;
    bool delivered_ = false;
    Inside inside_{};
};

// A memory as a black box: its read delay and its ports (its shape is the
// memory's own, MemoryShape). What it
// holds is its words (memory bits); a read's word in flight is in the
// memory's own read registers (a block RAM's output register), not
// flip-flops of the machine's.
template <typename T>
BlackBoxDeclaration memory_blackbox(const Memory<T>& memory, std::string latency_from, std::string readers,
    std::string writers) {
    const auto& shape = memory.shape();
    const uint64_t ports_read = uint64_t{shape.banks} * shape.read_ports;
    const uint64_t ports_write = uint64_t{shape.banks} * shape.write_ports;
    const uint64_t row = bits_for(shape.rows);
    const auto ports = [](uint64_t count, const char* kind) {
        return std::to_string(count) + " " + kind + " port" + (count == 1 ? "" : "s");
    };
    BlackBoxDeclaration blackbox(memory.name());
    blackbox.unit = memory.unit();
    blackbox.latency = shape.read_delay;
    blackbox.latency_from = std::move(latency_from);
    blackbox.inputs.push_back(BlackBoxPort{ports(ports_read, "read") + ": the address", ports_read * row, readers});
    if (ports_write != 0) {
        blackbox.inputs.push_back(BlackBoxPort{ports(ports_write, "write") + ": the address, word and element enables",
            ports_write * (row + shape.word_bits + shape.elements), std::move(writers)});
    }
    blackbox.outputs.push_back(BlackBoxPort{ports(ports_read, "read") + ": the word", ports_read * shape.word_bits,
        "the memory's read registers (its own, not flip-flops)"});
    return blackbox;
}

} // namespace nanosm

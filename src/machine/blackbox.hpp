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
//              holds it, which is counted in the open where it is;
//   outputs    the black box's own registers, clocked with every other one:
//              a value whose inputs were taken at t is in the output register
//              at t + latency, latency >= 1, so nothing passes through a black
//              box in one cycle;
//   latency    fixed, from the profile's delays;
//   interval   a new input at most every `interval` cycles;
//   cost       what the inside would probably use, as a function of the
//              unit's profile parameters, written beside the black box with
//              its reasoning;
//   signed off only by the human, on one line where the black box is
//              defined: SignOff::not_signed, which only the human changes to
//              SignOff::signed_by_human. A black box cannot be defined
//              without that line (BlackBoxDeclaration has no default for it).
//              The state report prints it.
// The memories (memory.hpp) are black boxes by the same contract --
// registered ports, a declared read delay, a size from the profile -- so they
// are declared in the same list (memory_blackbox below), and the words in
// flight during a read delay are the memory's business.
//
// A cost has three parts. Flip-flops and memory bits are what the state
// report counts in the open, so a black box's cost is comparable with them.
// Hard multipliers are what an arithmetic black box plainly has and neither
// of the others would show: a 25 x 18-bit multiplier block (a 24-bit
// significand product takes two, an 11-bit one takes one). Logic is not
// declared: the report takes a device's LUTs to be about its flip-flops.

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

// Whether the human has signed a black box off.
enum class SignOff : uint8_t { not_signed, signed_by_human };

// Bits to tell `n` things apart (0 for one or none).
inline uint64_t bits_for(uint64_t n) { return n <= 1 ? 0 : static_cast<uint64_t>(std::bit_width(n - 1)); }
// Bits to count from 0 to `n`.
inline uint64_t count_bits(uint64_t n) { return static_cast<uint64_t>(std::bit_width(n)); }

struct BlackBoxCost {
    uint64_t flip_flops = 0;  // registers inside the black box (its input and output registers are counted in the open)
    uint64_t memory_bits = 0; // RAM and ROM inside it (a memory: its words)
    uint64_t multipliers = 0; // hard 25 x 18-bit multipliers
};

// An input or an output: its bits and the register that holds it.
struct BlackBoxPort {
    std::string name;
    uint64_t bits = 0;
    std::string held_in;
};

struct BlackBoxDeclaration {
    // A black box is named and its sign-off given where it is defined; there
    // is no default.
    BlackBoxDeclaration(std::string name_, SignOff sign_off_) : name(std::move(name_)), sign_off(sign_off_) {}

    std::string name;
    SignOff sign_off;
    Unit unit = Unit::none;
    uint32_t latency = 1;
    std::string latency_from; // the profile's delays it is made of
    uint32_t interval = 1;
    std::vector<BlackBoxPort> inputs;
    std::vector<BlackBoxPort> outputs;
    BlackBoxCost cost;
    std::string cost_function; // the function, with its inputs' values
    std::string note;          // what else the report says of it
    [[nodiscard]] bool signed_off() const { return sign_off == SignOff::signed_by_human; }
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

    // The declared bits of the output register's payload (its valid bit is
    // counted beside it).
    void declare_output_bits(uint64_t bits) { output_bits_ = bits; }

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

    [[nodiscard]] StateSize size() const override {
        return sized("register", 1, output_bits_.has_value() ? std::optional<uint64_t>(*output_bits_ + 1) : std::nullopt);
    }

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
    std::optional<uint64_t> output_bits_;
    Inside inside_{};
};

// A memory as a black box: its read delay, its ports, its words. What it
// holds is its words (memory bits); a read's word in flight is in the
// memory's own read registers (a block RAM's output register), not
// flip-flops of the machine's. Its sign-off is given where it is declared,
// at the call.
template <typename T>
BlackBoxDeclaration memory_blackbox(const Memory<T>& memory, SignOff sign_off, std::string latency_from,
    std::string readers, std::string writers) {
    const auto& shape = memory.shape();
    const uint64_t ports_read = uint64_t{shape.banks} * shape.read_ports;
    const uint64_t ports_write = uint64_t{shape.banks} * shape.write_ports;
    const uint64_t row = bits_for(shape.rows);
    const auto ports = [](uint64_t count, const char* kind) {
        return std::to_string(count) + " " + kind + " port" + (count == 1 ? "" : "s");
    };
    BlackBoxDeclaration blackbox(memory.name(), sign_off);
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
    blackbox.cost.memory_bits = uint64_t{shape.banks} * shape.rows * shape.word_bits;
    blackbox.cost_function = "banks x words x word bits: " + std::to_string(shape.banks) + " x " + std::to_string(shape.rows) +
        " x " + std::to_string(shape.word_bits) + "; " + std::to_string(shape.read_ports) + "R " +
        std::to_string(shape.write_ports) + "W" + (shape.banks > 1 ? " per bank" : "") + ", " +
        same_cycle_text(shape.same_cycle);
    blackbox.note = memory.size().note;
    return blackbox;
}

} // namespace nanosm

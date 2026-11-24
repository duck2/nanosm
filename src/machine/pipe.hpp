#pragma once

// A pipeline: N stages in a row, clocked with the registers (reg.hpp).
//
// The question a pipeline answers is "where will a value that entered at
// cycle t be after n enabled edges": in stage n - 1, and gone after N. Nothing
// waits in it and nothing is addressed in it: a queue whose entries leave
// when a consumer is ready is a FIFO, not this.
//
// Each stage holds at most one value (a valid bit and the payload). Readers
// see Q for the whole tick, by stage number: stage 0 is the entry, stage
// N - 1 the output, whose value leaves at this tick's edge. A unit names the
// stage where it works ("stage D - 1 is at the RAM").
//
// At the clock edge, unless held:
//   stage k + 1 takes stage k's value (with this tick's replacement, if any),
//   stage 0 takes the value pushed this tick, or nothing,
//   the last stage's value leaves.
// hold() is the clock enable to every stage, low for this edge: no stage
// moves and stage 0 takes nothing (a push in a held tick is a bug).
//
// replace(k, v) is the logic a unit puts at stage k: the value there carries
// v from the edge on (a RAM's sampled words, an ALU's result), in stage
// k + 1 if the pipeline moves and in stage k if it is held; nothing as v
// means the value leaves the pipeline at stage k. An empty stage takes v the
// same way: a value entering at a tap, N - 1 - k edges from the output (the
// RF write calendar's bookings enter so, by how far ahead they are).
//
// Each write point -- the push, each stage's replacement, the hold -- is one
// net with one driver: a second write of the same point in a tick is the
// sanitizer's PANIC, as for Reg. With NANOSM_SANITIZER=0 the second write
// simply replaces the first; correct code never writes twice.
//
// N comes from the profile and is fixed when the machine is built (build()).
// Whether the stages become flip-flops or a shift-register primitive is the
// implementer's choice: nothing here depends on it. A zero-stage pipeline is
// a wire: it holds nothing and nothing may be pushed into it.

#include "machine/reg.hpp"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <stdint.h>
#include <string>
#include <utility>
#include <vector>

namespace nanosm {

template <typename T>
class Pipe final : public RegisterBase {
public:
    using Stage = std::optional<T>;

    // payload_bits: the declared hardware bits of one stage's payload (the
    // valid bit is counted beside it), or none where not declared yet.
    Pipe(RegisterFile& file, Unit unit, std::string name, std::optional<uint64_t> payload_bits = std::nullopt)
        : RegisterBase(file, unit, std::move(name)), payload_bits_(payload_bits) {
        static_assert(FixedPayload<T>, "a register's payload must be a fixed bundle of bits (machine/state.hpp); what is not there yet is listed at the end of machine.hpp");
        hold_point_ = new_write_point();
    }
    Pipe(RegisterFile& file, std::string name, std::optional<uint64_t> payload_bits = std::nullopt)
        : Pipe(file, Unit::none, std::move(name), payload_bits) {}

    // The number of stages, once, when the machine is built.
    void build(std::size_t stages) {
        if (built_) throw SimError("internal: pipeline '" + name() + "' built twice");
        built_ = true;
        q_.assign(stages, std::nullopt);
        replaced_.assign(stages, std::nullopt);
        stage_points_.reserve(stages);
        for (std::size_t stage = 0; stage < stages; ++stage) stage_points_.push_back(new_write_point());
    }

    // The declared hardware bits of one stage's payload, where they follow
    // from the profile (set when the machine is built).
    void declare_payload_bits(uint64_t bits) { payload_bits_ = bits; }
    // The declared bits of every stage together, valid bits included, where
    // the stages are not all alike (a stage that never holds a value).
    void declare_total_bits(uint64_t bits) { total_bits_ = bits; }

    [[nodiscard]] std::size_t stages() const { return q_.size(); }

    // Q, by stage: every reader's view for the whole tick.
    [[nodiscard]] const std::vector<Stage>& cur() const { return q_; }
    [[nodiscard]] const Stage& stage(std::size_t index) const { return q_.at(index); }
    // The output stage: its value leaves at this tick's edge (unless held).
    // (A zero-stage pipeline has no output: always nothing.)
    [[nodiscard]] const Stage& last() const { return q_.empty() ? none_ : q_.back(); }
    [[nodiscard]] bool empty() const { return occupied_ == 0; }

    // Stage 0's input at this edge. At most once per tick.
    void push(T value, const char* tag) {
        if (q_.empty()) throw SimError("internal: a value pushed into zero-stage pipeline '" + name() + "'");
        note_write(tag, pushed_.has_value());
        pushed_ = std::move(value);
    }

    // The value at stage `index` carries `value` from the edge on; nothing:
    // it leaves the pipeline there. At most once per stage per tick.
    void replace(std::size_t index, Stage value, const char* tag) {
        auto& slot = replaced_.at(index);
        // (The point's name is only built for the PANIC.)
        note_write(stage_points_.at(index),
            slot.has_value() ? name() + " (stage " + std::to_string(index) + ")" : std::string(), tag, slot.has_value());
        slot = std::move(value);
        edited_ = true;
    }

    // No stage moves at this edge.
    void hold(const char* tag) {
        note_write(hold_point_, held_ ? name() + " (hold)" : std::string(), tag, held_);
        held_ = true;
    }

    void commit() override {
        if (occupied_ == 0 && !pushed_.has_value() && !edited_) {
            held_ = false;
            return; // nothing in flight, nothing entering
        }
        if (held_) {
            if (pushed_.has_value()) {
                throw SimError("internal: a value pushed into pipeline '" + name() + "' in a tick it is held");
            }
            for (std::size_t stage = 0; stage < q_.size(); ++stage) {
                if (replaced_[stage].has_value()) q_[stage] = std::move(*replaced_[stage]);
            }
        } else {
            if (!q_.empty() && replaced_.back().has_value()) {
                throw SimError("internal: pipeline '" + name() +
                    "': its last stage was replaced in a tick its value leaves");
            }
            for (std::size_t stage = q_.size(); stage-- > 1;) {
                q_[stage] = replaced_[stage - 1].has_value() ? std::move(*replaced_[stage - 1]) : std::move(q_[stage - 1]);
            }
            if (!q_.empty()) q_.front() = std::move(pushed_);
        }
        pushed_.reset();
        for (auto& slot : replaced_) slot.reset();
        edited_ = false;
        held_ = false;
        occupied_ = static_cast<std::size_t>(
            std::count_if(q_.begin(), q_.end(), [](const Stage& stage) { return stage.has_value(); }));
    }

    [[nodiscard]] StateSize size() const override {
        auto size = sized("pipeline", q_.size(), std::nullopt);
        if (payload_bits_.has_value()) size.bits_per_entry = *payload_bits_ + 1; // + the valid bit
        size.total_bits = total_bits_;
        return size;
    }

private:
    std::vector<Stage> q_;
    Stage none_;
    std::optional<T> pushed_;
    std::vector<std::optional<Stage>> replaced_; // engaged: replaced this tick
    bool edited_ = false;
    bool held_ = false;
    bool built_ = false;
    std::size_t occupied_ = 0;
    std::optional<uint64_t> payload_bits_;
    std::optional<uint64_t> total_bits_;
    uint32_t hold_point_ = 0;
    std::vector<uint32_t> stage_points_;
};

} // namespace nanosm

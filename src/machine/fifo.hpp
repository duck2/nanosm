#pragma once

// A FIFO toward memory: entries pushed at the tail and popped at the head,
// clocked with the registers (reg.hpp). Only the deep, dynamic queues at the
// machine's edges toward global memory are FIFOs (a block RAM, or distributed
// RAM for a narrow one); everything that moves a fixed distance per cycle is a
// pipeline (pipe.hpp), and the small waiting structures are their units' own.
//
// The question a FIFO answers is "which entry will the next pop return": the
// oldest one pushed. Nothing is read, edited or removed anywhere else.
//
// Its shape is fixed when the machine is built (build()), from the profile:
//   capacity      entries it holds; a push past it is an internal error (the
//                 writer's logic must have looked at the room first).
//   write width   entries one push carries (a port wider than an entry: the
//                 bus's beats into a FIFO of words). One push per tick.
//   read width    entries the read port shows at the head (front(k), k below
//                 it) and the most one pop removes. One pop per tick.
//   read latency  the RAM's write-to-read delay: an entry pushed at the edge
//                 of cycle t is readable from cycle t + latency (1: the next
//                 cycle, as a register). The read side sees the write pointer
//                 through latency - 1 more register stages; count() is what
//                 the RAM holds, readable() what the read side sees of it.
//
// Readers see Q for the whole tick. push() and pop() are each one net with
// one driver: a second push, or a second pop, in one tick is the sanitizer's
// PANIC, as a second write of a Reg (with NANOSM_SANITIZER=0 the second one
// replaces the first; correct code never does it). At the edge the pop
// leaves first, then the push enters.

#include "machine/reg.hpp"

#include <cstddef>
#include <optional>
#include <stdint.h>
#include <string>
#include <utility>
#include <vector>

namespace nanosm {

struct FifoPorts {
    std::size_t write_width = 1;
    std::size_t read_width = 1;
    std::size_t read_latency = 1;
};

template <typename T>
class Fifo final : public Clocked {
public:
    Fifo(AllClocked& all, Unit unit, std::string name) : Clocked(all,unit, std::move(name)) {
        static_assert(FixedPayload<T>, "a register's payload must be a fixed bundle of bits (machine/state.hpp)");
        push_point_ = new_write_point();
        pop_point_ = new_write_point();
    }
    Fifo(AllClocked& all, std::string name) : Fifo(all,Unit::none, std::move(name)) {}

    // The FIFO's shape, once, when the machine is built.
    void build(std::size_t capacity, FifoPorts ports = {}) {
        if (built_) throw SimError("internal: FIFO '" + name() + "' built twice");
        // (A capacity of zero is a unit the profile leaves out: nothing may enter.)
        if (ports.write_width == 0 || ports.read_width == 0 || ports.read_latency == 0) {
            throw SimError("internal: FIFO '" + name() + "' built with a zero port or latency");
        }
        built_ = true;
        storage_.assign(capacity, T{});
        ports_ = ports;
        delay_.assign(ports.read_latency - 1, 0);
        pushed_.reserve(ports.write_width);
    }

    [[nodiscard]] std::size_t capacity() const { return storage_.size(); }
    [[nodiscard]] const FifoPorts& ports() const { return ports_; }
    // Entries in the RAM (Q), readable or not yet.
    [[nodiscard]] std::size_t count() const { return count_; }
    [[nodiscard]] bool empty() const { return count_ == 0; }
    [[nodiscard]] std::size_t room() const { return storage_.size() - count_; }
    // Entries the read side sees (their read latency has passed).
    [[nodiscard]] std::size_t readable() const { return readable_; }

    // An entry as the read port shows it: a reference to it, or the value
    // itself for a FIFO of single bits (they are packed; a bit has no address
    // to refer to).
    using Value = typename std::vector<T>::const_reference;

    // The read port: entry k from the head (0: the head), readable and inside
    // the port's width.
    [[nodiscard]] Value front(std::size_t k = 0) const {
        if (k >= readable_ || k >= ports_.read_width) {
            throw SimError("internal: FIFO '" + name() + "': read of entry " + std::to_string(k) + " with " +
                std::to_string(readable_) + " readable and a read port of " + std::to_string(ports_.read_width));
        }
        return std::as_const(storage_)[(head_ + k) % storage_.size()];
    }

    // The write port: the entries that enter at the tail at this edge, in
    // order, filled in place by the one writer. At most once per tick; the
    // write width is checked at the edge.
    [[nodiscard]] std::vector<T>& write_port(const char* tag) {
        note_write(push_point_, pushing_ ? name() + " (push)" : std::string(), tag, pushing_);
        pushing_ = true;
        pushed_.clear();
        return pushed_;
    }
    void push(const std::vector<T>& values, const char* tag) {
        auto& port = write_port(tag);
        port.insert(port.end(), values.begin(), values.end());
    }
    void push(T value, const char* tag) { write_port(tag).push_back(std::move(value)); }

    // The read port: `entries` leave the head at this edge. At most once per
    // tick, at most the read width, only readable entries.
    void pop(std::size_t entries, const char* tag) {
        note_write(pop_point_, popped_.has_value() ? name() + " (pop)" : std::string(), tag, popped_.has_value());
        if (entries > ports_.read_width || entries > readable_) {
            throw SimError("internal: FIFO '" + name() + "': a pop of " + std::to_string(entries) + " entries with " +
                std::to_string(readable_) + " readable and a read port of " + std::to_string(ports_.read_width));
        }
        popped_ = entries;
    }

    void commit() override {
        if (!pushing_ && !popped_.has_value() && in_delay_ == 0) return;
        // (A pop of nothing is still a drive of the read port; it moves no
        // pointer, and a zero-capacity FIFO has none to move.)
        if (popped_.has_value() && *popped_ != 0) {
            head_ = (head_ + *popped_) % storage_.size();
            count_ -= *popped_;
            readable_ -= *popped_;
        }
        // The write pointer as the read side sees it: what was pushed
        // latency - 1 edges ago becomes readable now.
        const std::size_t entering = pushing_ ? pushed_.size() : 0;
        if (entering > ports_.write_width) {
            throw SimError("internal: FIFO '" + name() + "': a push of " + std::to_string(entering) +
                " entries into a write port of " + std::to_string(ports_.write_width));
        }
        if (delay_.empty()) {
            readable_ += entering;
        } else {
            readable_ += delay_.back();
            in_delay_ -= delay_.back();
            for (std::size_t stage = delay_.size(); stage-- > 1;) delay_[stage] = delay_[stage - 1];
            delay_.front() = static_cast<uint32_t>(entering);
            in_delay_ += entering;
        }
        if (count_ + entering > storage_.size()) {
            throw SimError("internal: FIFO '" + name() + "' holds more than its " + std::to_string(storage_.size()) +
                " entries");
        }
        for (std::size_t index = 0; index < entering; ++index) {
            storage_[(head_ + count_) % storage_.size()] = std::move(pushed_[index]);
            ++count_;
        }
        pushed_.clear();
        pushing_ = false;
        popped_.reset();
    }

    // Its capacity, its ports' widths and its read latency.
    [[nodiscard]] StateSize size() const override {
        auto size = sized("FIFO", storage_.size());
        size.write_ports = static_cast<uint32_t>(ports_.write_width);
        size.read_ports = static_cast<uint32_t>(ports_.read_width);
        size.read_delay = static_cast<uint32_t>(ports_.read_latency);
        return size;
    }

private:
    std::vector<T> storage_;
    std::size_t head_ = 0;
    std::size_t count_ = 0;
    std::size_t readable_ = 0;
    std::vector<uint32_t> delay_; // entries pushed at each of the last latency - 1 edges, newest first
    std::size_t in_delay_ = 0;
    FifoPorts ports_;
    std::vector<T> pushed_;
    bool pushing_ = false;
    std::optional<std::size_t> popped_;
    bool built_ = false;
    uint32_t push_point_ = 0;
    uint32_t pop_point_ = 0;
};

} // namespace nanosm

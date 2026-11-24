#pragma once

// Clocked registers: the storage boundaries of the declared hardware.
//
// A Reg<T> holds Q, the value every reader sees for the whole tick (cur()),
// and D, the value it takes at the clock edge. set_next() supplies D. A
// register nobody writes holds Q. A second set_next() in one tick is a bug,
// even from the same writer: compute the complete next value in one place
// (ordinary locals may be edited freely) and write it once.
//
// Everything clocked (Clocked below: a register, a group, a pipeline, a
// FIFO, a memory, a black box's output) adds itself at construction to the
// machine's AllClocked, and AllClocked::commit() clocks all of it at the end
// of Machine::tick(), so a forgotten commit cannot happen.
//
// "D was written this tick" (the optional's engaged bit) is simulator
// bookkeeping that implements the register's hold / update: commit takes D if
// it was written and keeps Q otherwise, so it exists with the sanitizer
// compiled out. It is not a hardware state bit or a detection circuit: in
// hardware it is the register's enable (D = enable ? new value : Q). The
// writer tags and the duplicate-write PANIC are sanitizer bookkeeping
// (Sanitizer::record_register_write / check_register_rewrite); with
// NANOSM_SANITIZER=0 they compile to nothing and a second write would simply
// replace D. Correct code never writes twice, so both builds compute the same.

#include "machine/sanitizer.hpp"
#include "machine/state.hpp"

#include <cstddef>
#include <optional>
#include <stdint.h>
#include <string>
#include <utility>
#include <vector>

namespace nanosm {

class AllClocked;

// The machine's units, for the state report: every piece of clocked state
// names the one it belongs to when it is constructed.
enum class Unit : uint8_t {
    none, // a primitive outside a machine (a test's)
    front_end,
    branches,
    issue,
    dispatch,
    register_file,
    uniform_file,
    arithmetic,
    shared_memory,
    barriers,
    tma,
    tensor_core,
    load_store,
    launch,
};
inline constexpr Unit last_unit = Unit::launch;

inline const char* unit_name(Unit unit) {
    switch (unit) {
    case Unit::none: return "no unit";
    case Unit::front_end: return "front end";
    case Unit::branches: return "branches";
    case Unit::issue: return "issue";
    case Unit::dispatch: return "dispatch";
    case Unit::register_file: return "register file";
    case Unit::uniform_file: return "uniform register file";
    case Unit::arithmetic: return "arithmetic";
    case Unit::shared_memory: return "shared memory";
    case Unit::barriers: return "barriers";
    case Unit::tma: return "TMA";
    case Unit::tensor_core: return "tensor core";
    case Unit::load_store: return "load / store unit";
    case Unit::launch: return "launch";
    }
    return "?";
}

// What one piece of clocked state declares about its size, for a report of the
// machine's state by unit: how many entries it has (a register: 1; a group:
// its members; a pipeline: its stages; a FIFO: its capacity; a memory: its
// words) and the bits of one entry, when its declaration states them.
// Declared hardware bits, not what C++ stores; none when not declared yet.
struct StateSize {
    std::string name;
    Unit unit = Unit::none;
    const char* kind = "register";
    uint64_t entries = 1;
    std::optional<uint64_t> bits_per_entry;
    // Members of unequal widths (the LSU's two buffers): their sum.
    std::optional<uint64_t> total_bits;
    // Flip-flops beside the entries: a FIFO's pointers.
    uint64_t control_bits = 0;
    // A memory (memory.hpp): its entries are its words, in `banks` banks of
    // entries / banks words each, with these ports per bank; the read delay
    // and the rule for a read and a write of one word in one cycle. A FIFO:
    // its write and read ports' widths in entries and its read latency.
    uint64_t banks = 1;
    uint32_t read_ports = 0;
    uint32_t write_ports = 0;
    uint32_t read_delay = 0;
    const char* same_cycle = "";
    // What the report says beside the bits (where some of them are counted).
    std::string note;

    // The declared bits of the whole thing (entries and control), when
    // declared.
    [[nodiscard]] std::optional<uint64_t> bits() const {
        if (total_bits.has_value()) return *total_bits + control_bits;
        if (!bits_per_entry.has_value()) return std::nullopt;
        return entries * *bits_per_entry + control_bits;
    }
};

// The base of everything that takes the clock edge: a register, group,
// pipeline, FIFO, memory or black box (not all of them registers).
class Clocked {
public:
    Clocked(AllClocked& all, Unit unit, std::string name);
    Clocked(const Clocked&) = delete;
    Clocked& operator=(const Clocked&) = delete;
    virtual ~Clocked() = default;

    [[nodiscard]] const std::string& name() const { return name_; }
    [[nodiscard]] Unit unit() const { return unit_; }
    virtual void commit() = 0;
    [[nodiscard]] virtual StateSize size() const { return sized("register", 1, std::nullopt); }
    // A note for the state report, set when the machine is built.
    void declare_note(std::string note) { note_ = std::move(note); }

protected:
    // A size with this one's name, unit and note.
    [[nodiscard]] StateSize sized(const char* kind, uint64_t entries, std::optional<uint64_t> bits) const {
        StateSize size;
        size.name = name_;
        size.unit = unit_;
        size.note = note_;
        size.kind = kind;
        size.entries = entries;
        size.bits_per_entry = bits;
        return size;
    }
    // Tells the sanitizer about a write; it panics on the second one in a tick.
    void note_write(const char* tag, bool already_written) const;
    // The same for one of several write points of a composite (a pipeline's
    // entry, a stage's edit, its hold): each point is one net with one driver.
    [[nodiscard]] uint32_t new_write_point();
    void note_write(uint32_t point, const std::string& point_name, const char* tag, bool already_written) const;
    [[nodiscard]] Sanitizer& sanitizer() const;
    [[nodiscard]] uint64_t cycle() const;

private:
    AllClocked& all_;
    uint32_t id_ = 0;
    Unit unit_ = Unit::none;
    std::string name_;
    std::string note_;
};

template <typename T>
class Reg final : public Clocked {
public:
    Reg(AllClocked& all, Unit unit, std::string name, T initial = T{})
        : Clocked(all,unit, std::move(name)), q_(std::move(initial)) {
        static_assert(FixedPayload<T>, "a register's payload must be a fixed bundle of bits (machine/state.hpp); what is not there yet is listed at the end of machine.hpp");
    }
    Reg(AllClocked& all, std::string name, T initial = T{}) : Reg(all,Unit::none, std::move(name), std::move(initial)) {}

    // The declared hardware bits of its payload (set when the machine is
    // built, from the profile).
    void declare_bits(uint64_t bits) { bits_ = bits; }
    [[nodiscard]] StateSize size() const override { return sized("register", 1, bits_); }

    // Q: the value at the start of this tick, for every reader, all tick.
    [[nodiscard]] const T& cur() const { return q_; }

    // D: the value at the clock edge. At most once per tick.
    void set_next(T value, const char* tag) {
        note_write(tag, d_.has_value());
        d_ = std::move(value);
    }

    void commit() override {
        if (!d_.has_value()) return; // nobody wrote it: hold
        q_ = std::move(*d_);
        d_.reset();
    }

    // Host/test seam, only between ticks (e.g. a DMA complete_tx the host
    // places into the TMA's event register): Q changes at once.
    void poke(T value) { q_ = std::move(value); }

private:
    T q_;
    std::optional<T> d_;
    std::optional<uint64_t> bits_;
};

// The roll call of everything clocked that the machine owns: it commits them
// all at the end of a cycle, and the state report walks it. The machine must
// not move after its state is built (Machine deletes its copy and move
// operations).
class AllClocked {
public:
    AllClocked(Sanitizer& sanitizer, const uint64_t& cycle) : sanitizer_(sanitizer), cycle_(cycle) {}
    AllClocked(const AllClocked&) = delete;
    AllClocked& operator=(const AllClocked&) = delete;

    uint32_t add(Clocked& reg) {
        clocked_.push_back(&reg);
        return next_id_++;
    }

    // An id for a second write point of a composite register (no storage of
    // its own: the sanitizer's write tags are kept per id).
    uint32_t new_write_point() { return next_id_++; }

    // The clock edge: everything takes its D (or holds).
    void commit() {
        for (auto* reg : clocked_) reg->commit();
    }

    // Every register's declared size, in declaration order.
    [[nodiscard]] std::vector<StateSize> sizes() const {
        std::vector<StateSize> all;
        all.reserve(clocked_.size());
        for (const auto* reg : clocked_) all.push_back(reg->size());
        return all;
    }

    [[nodiscard]] Sanitizer& sanitizer() const { return sanitizer_; }
    [[nodiscard]] uint64_t cycle() const { return cycle_; }

private:
    Sanitizer& sanitizer_;
    const uint64_t& cycle_;
    std::vector<Clocked*> clocked_;
    uint32_t next_id_ = 0;
};

inline Clocked::Clocked(AllClocked& all, Unit unit, std::string name)
    : all_(all), unit_(unit), name_(std::move(name)) {
    id_ = all_.add(*this);
}

inline void Clocked::note_write(const char* tag, bool already_written) const {
    if (already_written) {
        all_.sanitizer().check_register_rewrite(id_, name_, all_.cycle(), tag);
    } else {
        all_.sanitizer().record_register_write(id_, tag);
    }
}

inline uint32_t Clocked::new_write_point() { return all_.new_write_point(); }
inline Sanitizer& Clocked::sanitizer() const { return all_.sanitizer(); }
inline uint64_t Clocked::cycle() const { return all_.cycle(); }

inline void Clocked::note_write(
    uint32_t point, const std::string& point_name, const char* tag, bool already_written) const {
    if (already_written) {
        all_.sanitizer().check_register_rewrite(point, point_name, all_.cycle(), tag);
    } else {
        all_.sanitizer().record_register_write(point, tag);
    }
}

} // namespace nanosm

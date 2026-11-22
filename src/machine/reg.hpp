#pragma once

// Clocked registers: the storage boundaries of the declared hardware.
//
// A Reg<T> holds Q, the value every reader sees for the whole tick (cur()),
// and D, the value it takes at the clock edge. set_next() supplies D. A
// register nobody writes holds Q. A second set_next() in one tick is a bug,
// even from the same writer: compute the complete next value in one place
// (ordinary locals may be edited freely) and write it once.
//
// Every Reg registers itself with its RegisterFile at construction, and
// RegisterFile::commit() clocks all of them at the end of Machine::tick(), so
// a forgotten commit cannot happen.
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

#include <cstddef>
#include <optional>
#include <stdint.h>
#include <string>
#include <utility>
#include <vector>

namespace nanosm {

class RegisterFile;

// What one piece of clocked state declares about its size, for a report of the
// machine's state by unit: how many entries it has (a register: 1; a pipeline:
// its stages) and the bits of one entry, when its declaration states them.
// Declared hardware bits, not what C++ stores; none when not declared yet.
struct StateSize {
    std::string name;
    const char* kind = "register";
    uint64_t entries = 1;
    std::optional<uint64_t> bits_per_entry;
};

class RegisterBase {
public:
    RegisterBase(RegisterFile& file, std::string name);
    RegisterBase(const RegisterBase&) = delete;
    RegisterBase& operator=(const RegisterBase&) = delete;
    virtual ~RegisterBase() = default;

    [[nodiscard]] const std::string& name() const { return name_; }
    virtual void commit() = 0;
    [[nodiscard]] virtual StateSize size() const { return StateSize{name_, "register", 1, std::nullopt}; }

protected:
    // Tells the sanitizer about a write; it panics on the second one in a tick.
    void note_write(const char* tag, bool already_written) const;
    // The same for one of several write points of a composite (a pipeline's
    // entry, a stage's edit, its hold): each point is one net with one driver.
    [[nodiscard]] uint32_t new_write_point();
    void note_write(uint32_t point, const std::string& point_name, const char* tag, bool already_written) const;

private:
    RegisterFile& file_;
    uint32_t id_ = 0;
    std::string name_;
};

template <typename T>
class Reg final : public RegisterBase {
public:
    Reg(RegisterFile& file, std::string name, T initial = T{})
        : RegisterBase(file, std::move(name)), q_(std::move(initial)) {}

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
};

// Every register of one machine. The machine must not move after its
// registers are built (Machine deletes its copy and move operations).
class RegisterFile {
public:
    RegisterFile(Sanitizer& sanitizer, const uint64_t& cycle) : sanitizer_(sanitizer), cycle_(cycle) {}
    RegisterFile(const RegisterFile&) = delete;
    RegisterFile& operator=(const RegisterFile&) = delete;

    uint32_t add(RegisterBase& reg) {
        registers_.push_back(&reg);
        return next_id_++;
    }

    // An id for a second write point of a composite register (no storage of
    // its own: the sanitizer's write tags are kept per id).
    uint32_t new_write_point() { return next_id_++; }

    // The clock edge: every register takes its D (or holds).
    void commit() {
        for (auto* reg : registers_) reg->commit();
    }

    // Every register's declared size, in declaration order.
    [[nodiscard]] std::vector<StateSize> sizes() const {
        std::vector<StateSize> all;
        all.reserve(registers_.size());
        for (const auto* reg : registers_) all.push_back(reg->size());
        return all;
    }

    [[nodiscard]] Sanitizer& sanitizer() const { return sanitizer_; }
    [[nodiscard]] uint64_t cycle() const { return cycle_; }

private:
    Sanitizer& sanitizer_;
    const uint64_t& cycle_;
    std::vector<RegisterBase*> registers_;
    uint32_t next_id_ = 0;
};

inline RegisterBase::RegisterBase(RegisterFile& file, std::string name)
    : file_(file), name_(std::move(name)) {
    id_ = file_.add(*this);
}

inline void RegisterBase::note_write(const char* tag, bool already_written) const {
    if (already_written) {
        file_.sanitizer().check_register_rewrite(id_, name_, file_.cycle(), tag);
    } else {
        file_.sanitizer().record_register_write(id_, tag);
    }
}

inline uint32_t RegisterBase::new_write_point() { return file_.new_write_point(); }

inline void RegisterBase::note_write(
    uint32_t point, const std::string& point_name, const char* tag, bool already_written) const {
    if (already_written) {
        file_.sanitizer().check_register_rewrite(point, point_name, file_.cycle(), tag);
    } else {
        file_.sanitizer().record_register_write(point, tag);
    }
}

} // namespace nanosm

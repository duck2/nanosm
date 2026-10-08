#pragma once

// Clocked registers: the storage boundaries of the declared hardware
// (trash/rfc09.md 3a; trash/stepping-discipline-2026-10-08.md).
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
// "D was written this tick" (the optional's engaged bit) is mechanism: commit
// needs it, so it exists with the sanitizer compiled out. The writer tags and
// the duplicate-write PANIC are sanitizer bookkeeping
// (Sanitizer::record_register_write / check_register_rewrite); with
// NANOSM_SANITIZER=0 they compile to nothing and a second write would simply
// replace D. Correct code never writes twice, so both builds compute the same.

#include "sanitizer.hpp"

#include <cstddef>
#include <optional>
#include <stdint.h>
#include <string>
#include <utility>
#include <vector>

namespace nanosm {

class RegisterFile;

class RegisterBase {
public:
    RegisterBase(RegisterFile& file, std::string name);
    RegisterBase(const RegisterBase&) = delete;
    RegisterBase& operator=(const RegisterBase&) = delete;
    virtual ~RegisterBase() = default;

    [[nodiscard]] const std::string& name() const { return name_; }
    virtual void commit() = 0;

protected:
    // Tells the sanitizer about a write; it panics on the second one in a tick.
    void note_write(const char* tag, bool already_written) const;

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
        return static_cast<uint32_t>(registers_.size() - 1);
    }

    // The clock edge: every register takes its D (or holds).
    void commit() {
        for (auto* reg : registers_) reg->commit();
    }

    [[nodiscard]] Sanitizer& sanitizer() const { return sanitizer_; }
    [[nodiscard]] uint64_t cycle() const { return cycle_; }

private:
    Sanitizer& sanitizer_;
    const uint64_t& cycle_;
    std::vector<RegisterBase*> registers_;
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

} // namespace nanosm

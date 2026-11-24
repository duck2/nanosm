#pragma once

// A memory: words in banks, read and written only through its ports, clocked
// with the registers (reg.hpp). The register file's banks, the uniform file,
// shared memory, tensor memory, the instruction memory and the divergence
// stack are memories; flip-flops are registers, groups and pipelines.
//
// Its shape is fixed when the machine is built (build()), from the profile:
//   banks         independent RAMs of one shape (the RF's three, shared
//                 memory's sixteen); a bank is one memory with its own ports.
//   rows          words per bank.
//   elements      the parts of a word a write may enable one by one (a lane
//                 of a register row, a byte of a shared-memory word); a word
//                 is `elements` values of T.
//   word bits     the word's declared width in hardware (what C++ stores
//                 does not count).
//   read ports    per bank: each reads one word per cycle (a second read of
//                 the same word through it is the same read, broadcast).
//   write ports   per bank: each writes one word per cycle, its elements
//                 enabled one by one (several pieces of the same word through
//                 one named port merge into one write). None: the memory is
//                 read only, and a write into it is an internal error.
//   read delay    from a read's address to its word at the output. The memory
//                 does not move the word: the pipeline or register around it
//                 does (an RF read's stages, an A-read pipeline, an address
//                 register), so the delay is declared here and modelled there.
//   same cycle    what a read and a write of one word in one cycle see.
//
// A read returns the word as it is at the start of the cycle (Q). A write is
// held until the clock edge and lands there, after every read of the cycle:
// a read and a write of one word in one cycle see the old word (read first).
//
// A port is named by its number (shared memory's port A is 0, port B 1), or
// left to the memory, which takes the lowest one free in that bank. A read or
// write past the bank's ports in one cycle -- a port asked for a second word,
// or no port left -- is the sanitizer's PANIC, as a second write of a Reg:
// one port, one word per cycle. With NANOSM_SANITIZER=0 the access simply
// happens; correct code never does it, so both builds compute the same.
//
// peek() and contents() are the observers' view (the host, a test, the
// sanitizer's look at a value, the trace): they take no port. poke() is the
// host's seam between ticks (a launch image), and takes effect at once.
//
// Storage order is row, then bank, then element: word w = row x banks + bank,
// so a memory banked by word address (shared memory by 4-byte word, tmem by
// word) keeps its elements in address order.

#include "machine/reg.hpp"

#include <cstddef>
#include <span>
#include <stdint.h>
#include <string>
#include <utility>
#include <vector>

namespace nanosm {

// A read and a write of one word in one cycle.
enum class SameCycle : uint8_t {
    read_first, // the read sees the old word; the write lands at the clock edge
    read_only,  // nothing writes it while the machine runs
};

inline const char* same_cycle_text(SameCycle rule) {
    return rule == SameCycle::read_first ? "read first" : "read only";
}

struct MemoryShape {
    std::size_t banks = 1;
    std::size_t rows = 0;
    std::size_t elements = 1;
    uint64_t word_bits = 0;
    uint32_t read_ports = 1;
    uint32_t write_ports = 1;
    uint32_t read_delay = 1;
    SameCycle same_cycle = SameCycle::read_first;
};

template <typename T>
class Memory final : public Clocked {
public:
    static constexpr uint32_t any_port = ~uint32_t{0};

    Memory(AllClocked& all, Unit unit, std::string name) : Clocked(all,unit, std::move(name)) {
        static_assert(FixedPayload<T>, "a memory's word must be a fixed bundle of bits (machine/state.hpp)");
    }
    Memory(AllClocked& all, std::string name) : Memory(all,Unit::none, std::move(name)) {}

    // The memory's shape, once, when the machine is built.
    void build(const MemoryShape& shape) {
        if (built_) throw SimError("internal: memory '" + name() + "' built twice");
        // (A write enables at most 64 elements one by one; a memory with no
        // write port is read only, and a read-only one has none.)
        if (shape.banks == 0 || shape.elements == 0 || shape.elements > 64 ||
            (shape.write_ports == 0) != (shape.same_cycle == SameCycle::read_only)) {
            throw SimError("internal: memory '" + name() + "' built with an impossible shape");
        }
        built_ = true;
        shape_ = shape;
        storage_.assign(shape.banks * shape.rows * shape.elements, T{});
        read_rows_.assign(shape.banks * shape.read_ports, none);
        write_rows_.assign(shape.banks * shape.write_ports, none);
    }

    [[nodiscard]] const MemoryShape& shape() const { return shape_; }
    [[nodiscard]] std::size_t rows() const { return shape_.rows; }

    // A read through a port of `bank`: the word in `row` as it is now (Q).
    [[nodiscard]] std::span<const T> read(std::size_t bank, std::size_t row, const char* tag,
        uint32_t port = any_port) {
        check(bank, row);
        take(read_rows_, shape_.read_ports, bank, row, port, true, "read", tag);
        return word(bank, row);
    }

    // A write through a port of `bank`: element e of `values` enters the
    // word in `row` at the clock edge when bit e of `enables` is set.
    void write(std::size_t bank, std::size_t row, std::span<const T> values, uint64_t enables, const char* tag,
        uint32_t port = any_port) {
        check(bank, row);
        if (shape_.write_ports == 0) throw SimError("internal: a write into read-only memory '" + name() + "'");
        if (values.size() != shape_.elements) {
            throw SimError("internal: memory '" + name() + "': a write of " + std::to_string(values.size()) +
                " elements into a word of " + std::to_string(shape_.elements));
        }
        take(write_rows_, shape_.write_ports, bank, row, port, port != any_port, "write", tag);
        const std::size_t base = (row * shape_.banks + bank) * shape_.elements;
        for (std::size_t element = 0; element < shape_.elements; ++element) {
            if ((enables >> element & 1) != 0) staged_.emplace_back(base + element, values[element]);
        }
    }
    static constexpr uint64_t every_element = ~uint64_t{0};

    // The observers' view: no port.
    [[nodiscard]] std::span<const T> peek(std::size_t bank, std::size_t row) const {
        check(bank, row);
        return word(bank, row);
    }
    [[nodiscard]] const std::vector<T>& contents() const { return storage_; }
    // The host's seam, only between ticks: element `element` of the word in
    // `row` of `bank` changes at once.
    void poke(std::size_t bank, std::size_t row, std::size_t element, T value) {
        check(bank, row);
        if (element >= shape_.elements) {
            throw SimError("internal: memory '" + name() + "': element " + std::to_string(element) + " of a word of " +
                std::to_string(shape_.elements));
        }
        storage_.at((row * shape_.banks + bank) * shape_.elements + element) = std::move(value);
    }

    void commit() override {
        for (auto& [index, value] : staged_) storage_[index] = std::move(value);
        staged_.clear();
        for (const auto index : used_reads_) read_rows_[index] = none;
        for (const auto index : used_writes_) write_rows_[index] = none;
        used_reads_.clear();
        used_writes_.clear();
    }

    [[nodiscard]] StateSize size() const override {
        auto size = sized("memory", shape_.banks * shape_.rows, shape_.word_bits);
        size.banks = shape_.banks;
        size.read_ports = shape_.read_ports;
        size.write_ports = shape_.write_ports;
        size.read_delay = shape_.read_delay;
        size.same_cycle = same_cycle_text(shape_.same_cycle);
        return size;
    }

private:
    static constexpr std::size_t none = ~std::size_t{0};

    void check(std::size_t bank, std::size_t row) const {
        if (bank >= shape_.banks || row >= shape_.rows) {
            throw SimError("internal: memory '" + name() + "': bank " + std::to_string(bank) + " row " +
                std::to_string(row) + " is outside its " + std::to_string(shape_.banks) + " x " +
                std::to_string(shape_.rows) + " words");
        }
    }

    [[nodiscard]] std::span<const T> word(std::size_t bank, std::size_t row) const {
        return std::span<const T>(storage_.data() + (row * shape_.banks + bank) * shape_.elements, shape_.elements);
    }

    // One port of `bank` takes `row` this cycle. `same_row_shares`: a second
    // access of the same row through that port is the same access (a read's
    // broadcast; pieces of one word through a named write port).
    void take(std::vector<std::size_t>& rows, uint32_t ports, std::size_t bank, std::size_t row, uint32_t port,
        bool same_row_shares, const char* what, const char* tag) {
        auto& used = &rows == &read_rows_ ? used_reads_ : used_writes_;
        const std::size_t first = bank * ports;
        if (port != any_port) {
            if (port >= ports) {
                sanitizer().panic_memory_ports(name(), "bank " + std::to_string(bank) + ": a " + what +
                    " through port " + std::to_string(port) + " of " + std::to_string(ports), cycle(), tag);
                return;
            }
            auto& held = rows[first + port];
            if (held == row && same_row_shares) return;
            if (held != none) {
                sanitizer().panic_memory_ports(name(), "bank " + std::to_string(bank) + ": " + what + " port " +
                    std::to_string(port) + " asked for row " + std::to_string(row) + " after row " +
                    std::to_string(held) + " in one cycle", cycle(), tag);
                return;
            }
            held = row;
            used.push_back(first + port);
            return;
        }
        for (uint32_t candidate = 0; candidate < ports; ++candidate) {
            if (same_row_shares && rows[first + candidate] == row) return;
        }
        for (uint32_t candidate = 0; candidate < ports; ++candidate) {
            auto& held = rows[first + candidate];
            if (held != none) continue;
            held = row;
            used.push_back(first + candidate);
            return;
        }
        sanitizer().panic_memory_ports(name(), "bank " + std::to_string(bank) + ": a " + what + " of row " +
            std::to_string(row) + " with its " + std::to_string(ports) + " " + what + " port" +
            (ports == 1 ? "" : "s") + " taken this cycle", cycle(), tag);
    }

    MemoryShape shape_;
    std::vector<T> storage_;
    std::vector<std::pair<std::size_t, T>> staged_; // (element index, value), landing at the edge in order
    std::vector<std::size_t> read_rows_;  // per bank and read port: the row it reads this cycle
    std::vector<std::size_t> write_rows_; // per bank and write port: the row it writes this cycle
    std::vector<std::size_t> used_reads_;
    std::vector<std::size_t> used_writes_;
    bool built_ = false;
};

} // namespace nanosm

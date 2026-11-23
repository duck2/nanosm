#pragma once

// A group of registers: N registers of one kind under one name (one per warp,
// per bank, per slot), clocked with the others (reg.hpp).
//
// N comes from the profile and is fixed when the machine is built (build()).
// Each member is a register of its own: readers see its Q for the whole tick,
// set_next() supplies its D, and a member nobody writes holds. Different
// members may be written by different logic in one tick -- each has its own
// D input -- but one member takes at most one write per tick: that is one net
// with one driver, and a second write of the same member is the sanitizer's
// PANIC, as for Reg. With NANOSM_SANITIZER=0 the second write simply replaces
// the first; correct code never writes twice.
//
// The group reports its size as one unit: N members of the member's declared
// bits.

#include "machine/reg.hpp"

#include <cstddef>
#include <optional>
#include <stdint.h>
#include <string>
#include <utility>
#include <vector>

namespace nanosm {

template <typename T>
class Regs final : public RegisterBase {
public:
    // A member's Q as it is read: a reference to the value, or the value
    // itself for a group of single bits (they are packed; a bit has no
    // address to refer to).
    using Value = typename std::vector<T>::const_reference;

    // One member, for code written against a single register:
    // group[i].cur(), group[i].set_next(value, tag).
    class Member {
    public:
        Member(Regs& group, std::size_t index) : group_(group), index_(index) {}
        [[nodiscard]] Value cur() const { return std::as_const(group_.q_).at(index_); }
        void set_next(T value, const char* tag) { group_.set_next(index_, std::move(value), tag); }
        void poke(T value) { group_.poke(index_, std::move(value)); }

    private:
        Regs& group_;
        std::size_t index_;
    };
    class ConstMember {
    public:
        ConstMember(const Regs& group, std::size_t index) : group_(group), index_(index) {}
        [[nodiscard]] Value cur() const { return group_.q_.at(index_); }

    private:
        const Regs& group_;
        std::size_t index_;
    };

    // member_bits: the declared hardware bits of one member, or none where
    // not declared yet.
    Regs(RegisterFile& file, std::string name, std::optional<uint64_t> member_bits = std::nullopt)
        : RegisterBase(file, std::move(name)), member_bits_(member_bits) {
        static_assert(FixedPayload<T>, "a register's payload must be a fixed bundle of bits (machine/state.hpp); what is not there yet is listed at the end of machine.hpp");
    }

    // The number of members, once, when the machine is built.
    void build(std::size_t members, const T& initial = T{}) {
        if (built_) throw SimError("internal: register group '" + name() + "' built twice");
        built_ = true;
        q_.assign(members, initial);
        d_.assign(members, std::nullopt);
        points_.reserve(members);
        for (std::size_t member = 0; member < members; ++member) points_.push_back(new_write_point());
    }

    // The declared hardware bits of one member, where they follow from the
    // profile (set when the machine is built).
    void declare_member_bits(uint64_t bits) { member_bits_ = bits; }

    [[nodiscard]] std::size_t count() const { return q_.size(); }

    // Q of every member, by index: every reader's view for the whole tick.
    [[nodiscard]] const std::vector<T>& cur() const { return q_; }
    [[nodiscard]] Member operator[](std::size_t index) { return Member(*this, index); }
    [[nodiscard]] ConstMember operator[](std::size_t index) const { return ConstMember(*this, index); }
    [[nodiscard]] Member at(std::size_t index) {
        (void)q_.at(index);
        return Member(*this, index);
    }
    [[nodiscard]] ConstMember at(std::size_t index) const {
        (void)q_.at(index);
        return ConstMember(*this, index);
    }

    // One member's D. At most once per member per tick.
    void set_next(std::size_t index, T value, const char* tag) {
        auto& d = d_.at(index);
        // (The member's name is only built for the PANIC.)
        note_write(points_.at(index), d.has_value() ? name() + " [" + std::to_string(index) + "]" : std::string(), tag,
            d.has_value());
        if (!d.has_value()) written_.push_back(index);
        d = std::move(value);
    }

    void commit() override {
        for (const auto index : written_) {
            q_[index] = std::move(*d_[index]);
            d_[index].reset();
        }
        written_.clear();
    }

    // Host / test seam, only between ticks: Q changes at once.
    void poke(std::size_t index, T value) { q_.at(index) = std::move(value); }

    [[nodiscard]] StateSize size() const override { return StateSize{name(), "group", q_.size(), member_bits_}; }

private:
    std::vector<T> q_;
    std::vector<std::optional<T>> d_;
    std::vector<std::size_t> written_; // members with a D this tick
    std::vector<uint32_t> points_;     // one write point per member
    bool built_ = false;
    std::optional<uint64_t> member_bits_;
};

} // namespace nanosm

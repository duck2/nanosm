#pragma once

// A payload of up to N items: N places and a count. Hardware state has a
// fixed size, so a field whose length varies (the lanes of a round, the words
// of a beat, the packets of a buffer) is declared at its architectural
// maximum (limits.hpp) with a count of how many are meant. Going past N is an
// internal error: the profile is checked against the maxima when it is
// loaded, so a kernel cannot get there.
//
// It reads like the std::vector it replaces (size, [], push_back, a range)
// and is trivially copyable when T is, which is what a register's payload
// must be (state.hpp).

#include "qtxc/qtxc.hpp"

#include <array>
#include <cstddef>
#include <initializer_list>
#include <stdint.h>
#include <string>
#include <utility>
#include <vector>

namespace nanosm {

template <typename T, std::size_t N>
class Bounded {
public:
    using value_type = T;
    using iterator = T*;
    using const_iterator = const T*;

    constexpr Bounded() = default;
    Bounded(std::initializer_list<T> items) {
        for (const auto& item : items) push_back(item);
    }
    Bounded(std::size_t count, const T& value) { assign(count, value); }
    // To and from the std::vector a wire or a host-side view still is.
    Bounded(const std::vector<T>& items) {
        for (const auto& item : items) push_back(item);
    }
    Bounded& operator=(const std::vector<T>& items) {
        clear();
        for (const auto& item : items) push_back(item);
        return *this;
    }
    Bounded& operator=(std::initializer_list<T> items) {
        clear();
        for (const auto& item : items) push_back(item);
        return *this;
    }
    operator std::vector<T>() const { return std::vector<T>(begin(), end()); }

    [[nodiscard]] constexpr std::size_t size() const { return count_; }
    [[nodiscard]] constexpr bool empty() const { return count_ == 0; }
    [[nodiscard]] static constexpr std::size_t capacity() { return N; }

    [[nodiscard]] T* data() { return items_.data(); }
    [[nodiscard]] const T* data() const { return items_.data(); }
    [[nodiscard]] iterator begin() { return items_.data(); }
    [[nodiscard]] iterator end() { return items_.data() + count_; }
    [[nodiscard]] const_iterator begin() const { return items_.data(); }
    [[nodiscard]] const_iterator end() const { return items_.data() + count_; }

    [[nodiscard]] T& operator[](std::size_t index) { return items_[index]; }
    [[nodiscard]] const T& operator[](std::size_t index) const { return items_[index]; }
    [[nodiscard]] T& at(std::size_t index) {
        check(index);
        return items_[index];
    }
    [[nodiscard]] const T& at(std::size_t index) const {
        check(index);
        return items_[index];
    }
    [[nodiscard]] T& front() { return at(0); }
    [[nodiscard]] const T& front() const { return at(0); }
    [[nodiscard]] T& back() { return at(count_ - 1); }
    [[nodiscard]] const T& back() const { return at(count_ - 1); }

    void push_back(const T& item) {
        if (count_ == N) throw SimError("internal: a payload of at most " + std::to_string(N) + " items took one more");
        items_[count_++] = item;
    }
    template <typename... Fields>
    void emplace_back(Fields&&... fields) {
        push_back(T{std::forward<Fields>(fields)...});
    }
    void pop_back() {
        check(0);
        items_[--count_] = T{};
    }
    // The first item leaves and the others move up: an ordered buffer's head.
    void pop_front() {
        check(0);
        for (std::size_t index = 1; index < count_; ++index) items_[index - 1] = items_[index];
        items_[--count_] = T{};
    }
    void erase_at(std::size_t position) {
        check(position);
        for (std::size_t index = position + 1; index < count_; ++index) items_[index - 1] = items_[index];
        items_[--count_] = T{};
    }
    void clear() {
        for (std::size_t index = 0; index < count_; ++index) items_[index] = T{};
        count_ = 0;
    }
    void assign(std::size_t count, const T& value) {
        if (count > N) throw SimError("internal: a payload of at most " + std::to_string(N) + " items was given " +
            std::to_string(count));
        clear();
        for (std::size_t index = 0; index < count; ++index) items_[index] = value;
        count_ = static_cast<uint32_t>(count);
    }
    template <typename Iterator>
    void assign(Iterator first, Iterator last) {
        clear();
        for (; first != last; ++first) push_back(*first);
    }
    // More items behind the last (the only place a payload grows).
    template <typename Iterator>
    void insert(const_iterator position, Iterator first, Iterator last) {
        if (position != end()) throw SimError("internal: a payload grows only at its end");
        for (; first != last; ++first) push_back(*first);
    }
    void resize(std::size_t count) {
        if (count > N) throw SimError("internal: a payload of at most " + std::to_string(N) + " items was resized to " +
            std::to_string(count));
        for (std::size_t index = count; index < count_; ++index) items_[index] = T{};
        count_ = static_cast<uint32_t>(count);
    }

    [[nodiscard]] std::vector<T> to_vector() const { return std::vector<T>(begin(), end()); }

    // Places past the count are kept at T{}, so two equal payloads are equal
    // bit for bit; the comparison still looks only at the counted items.
    [[nodiscard]] bool operator==(const Bounded& other) const {
        if (count_ != other.count_) return false;
        for (std::size_t index = 0; index < count_; ++index) {
            if (!(items_[index] == other.items_[index])) return false;
        }
        return true;
    }

private:
    void check(std::size_t index) const {
        if (index >= count_) {
            throw SimError("internal: item " + std::to_string(index) + " of a payload holding " + std::to_string(count_));
        }
    }

    std::array<T, N> items_{};
    uint32_t count_ = 0;
};

} // namespace nanosm

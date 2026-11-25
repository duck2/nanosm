// The kernel's own memory by name (src/qtxc/; docs/qtx.md "Memory"): the
// `.shmem` and `.tmem` lines, where qtxc puts each object, and the same lines
// as assembly carries them.
//
//     .shmem QBUF[2],   rows=32, cols=64, elem=2          a tile: rows back to back, a row is cols x elem bytes
//     .shmem STATS[2],  bytes=1024, align=4               a plain buffer
//     .shmem OSTAGE[2], rows=32, cols=64, elem=4, over=QBUF, KBUF[0]
//     .tmem  S[2],      rows=32, cols=64                  tensor memory, in words
//
// An object is the author's bookkeeping: the machine never sees one, only the
// addresses instructions carry. So a line says nothing about a layout (a
// tile's layout is its TMA descriptor's and the tensor core's, both keyed by
// the address: src/layout.hpp), and its alignment is its own statement:
//
//   - `align=` is in the memory's unit (bytes; words for tmem) and replaces
//     the default. The defaults are only what the machine needs: a shared
//     tile on a multiple of its row and of 64 bytes (swizzle64's base rule;
//     64 is also what a TMA buffer needs), a plain buffer on 64, a tmem tile
//     on four of its rows (a matmul result's and an acc.ld's pointer rule).
//     A kernel that needs more says so (FlashAttention's software shuffle
//     wants its tiles on 16-row multiples).
//   - An array's elements are contiguous: element i is at NAME + i x stride,
//     the stride being one element rounded up to the alignment.
//
// Placement: objects go one after another from address 0 in the order they
// are declared, each on its alignment; the launch block (.param / .mbar /
// .data) stays at the top of shared memory; an object that does not fit is
// named. Two objects never overlap unless one says `over=`: it is then not
// placed but laid on the objects it lists (a bare array name is all its
// elements), which must follow one another in that order and be large enough
// (up to the last covered element's own size: the padding after an object is
// not its own). The first of them takes the overlay's alignment before
// anything is placed.
//
// Assembly carries each line with the address qtxc chose (`at=ADDRESS`,
// assembly only), so reading it back needs no placement. A hand-written
// assembly may put an object elsewhere, under the same rules: on its
// alignment, inside its memory, apart from the others unless it says over=
// (and then inside what it names).

#include "qtxc/internal.hpp"
#include "layout.hpp"

#include <algorithm>
#include <numeric>
#include <regex>
#include <sstream>

namespace nanosm {
namespace qtxc_internal {

namespace {

std::string at(uint32_t line) {
    return "line " + std::to_string(line) + ": ";
}

const char* directive_of(const MemoryObject& object) {
    return object.tmem ? ".tmem" : ".shmem";
}

const char* unit_of(const MemoryObject& object) {
    return object.tmem ? "words" : "bytes";
}

uint64_t round_up(uint64_t value, uint64_t multiple) {
    return (value + multiple - 1) / multiple * multiple;
}

// An argument's value: a constant expression, 1..maximum.
uint32_t argument(
    const std::string& text, const std::unordered_map<std::string, int32_t>& constants, uint64_t maximum,
    uint32_t line, const std::string& what) {
    std::string problem;
    const auto value = evaluate_expression(text, constants, problem);
    if (!value.has_value()) throw SimError(at(line) + what + " '" + text + "': " + problem);
    if (*value < 1 || static_cast<uint64_t>(*value) > maximum) {
        throw SimError(at(line) + what + " " + std::to_string(*value) + " must be 1.." + std::to_string(maximum));
    }
    return static_cast<uint32_t>(*value);
}

// "NAME" or "NAME[i]" of an `over=` list -> (name, element or none).
std::pair<std::string, std::optional<uint32_t>> target_of(const std::string& text) {
    static const std::regex pattern(R"(^([A-Za-z_][A-Za-z0-9_]*)(?:\[([0-9]+)\])?$)");
    std::smatch match;
    if (!std::regex_match(text, match, pattern)) return {text, std::nullopt};
    if (!match[2].matched) return {match[1].str(), std::nullopt};
    return {match[1].str(), static_cast<uint32_t>(std::stoul(match[2].str()))};
}

} // namespace

// `.shmem` / `.tmem` (`tail`: the line after the directive's name). The
// object is appended to `objects` -- before its `align=` is read, so that an
// alignment may be written with the object's own row (align=16 *
// NAME.row_bytes). It has no address yet unless the line gives one (`at=`,
// assembly: allow_at).
void parse_object_directive(
    const std::string& tail, bool tmem, const std::unordered_map<std::string, int32_t>& constants, uint32_t line,
    bool allow_at, std::vector<MemoryObject>& objects) {
    const std::string directive = tmem ? ".tmem" : ".shmem";
    const std::string form = tmem
        ? ".tmem NAME[COUNT], rows=R, cols=C [, align=WORDS] [, over=OBJECT, ...]"
        : ".shmem NAME[COUNT], rows=R, cols=C, elem=BYTES [, align=BYTES] [, over=OBJECT, ...] "
          "(a tile) or .shmem NAME[COUNT], bytes=N [, align=BYTES] (a plain buffer)";
    const auto args = operands(tail);
    static const std::regex name_pattern(R"(^([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[(.+)\])?$)");
    std::smatch match;
    if (args.empty() || !std::regex_match(args[0], match, name_pattern)) {
        throw SimError(at(line) + "expected " + form);
    }
    MemoryObject object;
    object.tmem = tmem;
    object.line = line;
    object.name = section_name(match[1].str(), line, tmem ? "tmem object" : "shared-memory object");
    if (match[2].matched) {
        object.count = argument(trim(match[2].str()), constants, 4096, line, directive + " " + object.name + ": the element count");
    }
    for (const auto& earlier : objects) {
        if (earlier.name == object.name) throw SimError(at(line) + "duplicate symbol '" + object.name + "'");
    }
    // key=value items; what follows `over=` without a key of its own belongs to it.
    std::vector<std::pair<std::string, std::string>> items;
    for (std::size_t index = 1; index < args.size(); ++index) {
        const auto equals = args[index].find('=');
        const auto key = equals == std::string::npos ? std::string() : trim(args[index].substr(0, equals));
        static const std::regex key_pattern(R"(^[a-z_]+$)");
        if (equals != std::string::npos && std::regex_match(key, key_pattern)) {
            items.emplace_back(key, trim(args[index].substr(equals + 1)));
        } else if (!items.empty() && items.back().first == "over") {
            items.emplace_back("over", args[index]);
        } else {
            throw SimError(at(line) + directive + " " + object.name + ": '" + args[index] + "' is not key=value; expected " + form);
        }
    }
    const auto value_of = [&](const std::string& key) -> std::optional<std::string> {
        std::optional<std::string> found;
        for (const auto& [name, value] : items) {
            if (name != key) continue;
            if (found.has_value()) throw SimError(at(line) + directive + " " + object.name + ": " + key + "= is given twice");
            found = value;
        }
        return found;
    };
    for (const auto& [key, value] : items) {
        const bool known = key == "rows" || key == "cols" || key == "align" || key == "over" ||
            (!tmem && (key == "elem" || key == "bytes")) || (allow_at && key == "at");
        if (!known) {
            const std::string hint = key == "layout"
                ? " (an object has no layout: a tile's layout is its TMA descriptor's and the tensor core's; say align= "
                  "if the kernel needs more alignment)"
                : key == "at" ? " (qtxc places objects; over= lays one on others)" : "";
            throw SimError(at(line) + directive + " " + object.name + ": unknown key '" + key + "='" + hint +
                "; expected " + form);
        }
        if (value.empty()) throw SimError(at(line) + directive + " " + object.name + ": " + key + "= has no value");
    }
    const auto what = directive + " " + object.name + ": ";
    const auto rows = value_of("rows");
    const auto cols = value_of("cols");
    const auto elem = value_of("elem");
    const auto bytes = value_of("bytes");
    if (bytes.has_value()) {
        if (rows.has_value() || cols.has_value() || elem.has_value()) {
            throw SimError(at(line) + what + "bytes= is a plain buffer; a tile says rows=, cols=, elem= and no bytes=");
        }
        object.size = argument(*bytes, constants, 1u << 24, line, what + "bytes");
    } else {
        if (!rows.has_value() || !cols.has_value() || (!tmem && !elem.has_value())) {
            throw SimError(at(line) + what + "expected " + form);
        }
        object.rows = argument(*rows, constants, 1u << 16, line, what + "rows");
        object.cols = argument(*cols, constants, 1u << 16, line, what + "cols");
        object.elem = tmem ? 1 : argument(*elem, constants, 64, line, what + "elem (an element's bytes)");
        if (tmem && !valid_tmem_row_words(object.cols)) {
            throw SimError(at(line) + what + "cols=" + std::to_string(object.cols) +
                ": a tmem row is a power of two of words, 4 or more (src/layout.hpp: the row length an acc.ld / "
                "acc.st names, a matmul result's n)");
        }
        const auto size = uint64_t{object.rows} * object_row(object);
        if (size > (1u << 24)) throw SimError(at(line) + what + "the tile is too large");
        object.size = static_cast<uint32_t>(size);
    }
    // The defaults: only what the machine needs.
    const uint32_t row = object.rows == 0 ? 0 : object_row(object);
    object.alignment = tmem ? 4 * object.cols : object.rows == 0 ? 64 : std::lcm(row, 64u);
    objects.push_back(object);
    auto& placed = objects.back();
    if (const auto align = value_of("align"); align.has_value()) {
        placed.alignment = argument(*align, constants, 1u << 24, line, what + "align");
        if (tmem && placed.alignment % (4 * placed.cols) != 0) {
            throw SimError(at(line) + what + "align=" + std::to_string(placed.alignment) +
                " words: a tmem object's base is a multiple of four of its rows (" + std::to_string(4 * placed.cols) +
                " words): a matmul's result and an acc.ld's sixteen rows need whole bands of four rows (src/layout.hpp)");
        }
    }
    placed.stride = static_cast<uint32_t>(round_up(placed.size, placed.alignment));
    for (const auto& [key, value] : items) {
        if (key != "over") continue;
        // An element's index may be a constant expression: keep its value.
        static const std::regex element(R"(^([A-Za-z_][A-Za-z0-9_]*)\s*\[(.+)\]$)");
        std::smatch target;
        if (std::regex_match(value, target, element)) {
            std::string problem;
            const auto index = evaluate_expression(trim(target[2].str()), constants, problem);
            if (!index.has_value() || *index < 0) {
                throw SimError(at(line) + what + "over=" + value + ": bad element index" +
                    (problem.empty() ? std::string() : " (" + problem + ")"));
            }
            placed.over.push_back(target[1].str() + "[" + std::to_string(*index) + "]");
        } else {
            placed.over.push_back(value);
        }
    }
    if (const auto address = value_of("at"); address.has_value()) {
        std::string problem;
        const auto value = evaluate_expression(*address, constants, problem);
        if (!value.has_value() || *value < 0 || *value > 0xffffffffll) {
            throw SimError(at(line) + what + "bad address at=" + *address);
        }
        placed.address = static_cast<uint32_t>(*value);
        placed.placed = true;
    }
}

// Every object that has no address yet gets one (the rule at the top of this
// file). `shmem_limit`: where the launch block starts, or the end of shared
// memory; `tmem_words`: the profile's tmem.
void place_objects(std::vector<MemoryObject>& objects, uint32_t shmem_limit, uint64_t tmem_words) {
    struct Target {
        std::size_t object = 0;
        uint32_t first = 0;
        uint32_t count = 0;
    };
    const auto find = [&](const std::string& name) -> std::optional<std::size_t> {
        for (std::size_t index = 0; index < objects.size(); ++index) {
            if (objects[index].name == name) return index;
        }
        return std::nullopt;
    };
    // What each overlay lies on, and the alignment its first target must take.
    std::vector<std::vector<Target>> targets(objects.size());
    std::vector<uint32_t> base_alignment(objects.size());
    for (std::size_t index = 0; index < objects.size(); ++index) base_alignment[index] = objects[index].alignment;
    for (std::size_t index = 0; index < objects.size(); ++index) {
        const auto& object = objects[index];
        if (object.over.empty()) continue;
        const auto what = at(object.line) + directive_of(object) + " " + object.name + ": ";
        for (const auto& written : object.over) {
            const auto [name, element] = target_of(written);
            const auto found = find(name);
            if (!found.has_value()) {
                throw SimError(what + "over=" + written + ": there is no " + directive_of(object) + " object '" + name + "'");
            }
            const auto& under = objects[*found];
            if (*found == index) throw SimError(what + "it cannot lie over itself");
            if (under.tmem != object.tmem) {
                throw SimError(what + "over=" + written + ": " + name + " is in " +
                    (under.tmem ? "tensor memory" : "shared memory") + ", not in " +
                    (object.tmem ? "tensor memory" : "shared memory"));
            }
            if (!under.over.empty()) {
                throw SimError(what + "over=" + written + ": " + name + " itself lies over other objects; name those");
            }
            if (element.has_value() && *element >= under.count) {
                throw SimError(what + "over=" + written + " is past the end of " + directive_of(under) + " " + name +
                    ", which has " + std::to_string(under.count) + (under.count == 1 ? " element" : " elements"));
            }
            targets[index].push_back({*found, element.value_or(0), element.has_value() ? 1 : under.count});
        }
        const auto& first = targets[index].front();
        if (!object.placed && first.first == 0) {
            base_alignment[first.object] = std::lcm(base_alignment[first.object], object.alignment);
        }
    }
    // One after another, in the order declared.
    uint64_t cursor[2] = {0, 0};
    for (std::size_t index = 0; index < objects.size(); ++index) {
        auto& object = objects[index];
        if (object.placed || !object.over.empty()) continue;
        auto& next = cursor[object.tmem ? 1 : 0];
        const auto address = round_up(next, base_alignment[index]);
        const uint64_t limit = object.tmem ? tmem_words : shmem_limit;
        if (address + object_extent(object) > limit) {
            throw SimError(at(object.line) + directive_of(object) + " " + object.name + " does not fit: it needs " +
                std::to_string(object_extent(object)) + " " + unit_of(object) + " at " + std::to_string(address) +
                ", and " + (object.tmem
                    ? "this profile's tensor memory has " + std::to_string(limit) + " words"
                    : "shared memory below the launch block (.param / .mbar / .data, at the top) ends at " +
                      std::to_string(limit)));
        }
        object.address = static_cast<uint32_t>(address);
        object.placed = true;
        next = address + object_extent(object);
    }
    // The overlays, on what they named. What an overlay may use is the run of
    // its targets: from the first one's base to the LAST covered element's
    // base + size. The stride (an element rounded up to the alignment) counts
    // only between two elements of one array: the padding after an object's
    // last element is not the object's, and the next declaration may sit in it.
    for (std::size_t index = 0; index < objects.size(); ++index) {
        auto& object = objects[index];
        if (object.over.empty()) continue;
        const auto what = at(object.line) + directive_of(object) + " " + object.name + ": ";
        uint64_t start = 0;
        uint64_t end = 0;
        for (std::size_t item = 0; item < targets[index].size(); ++item) {
            const auto& target = targets[index][item];
            const auto& under = objects[target.object];
            const uint64_t begin = uint64_t{under.address} + uint64_t{target.first} * under.stride;
            if (item == 0) {
                start = begin;
            } else {
                // The next element of the same array follows by the array's own stride.
                const auto& before = targets[index][item - 1];
                const bool next_element =
                    before.object == target.object && before.first + before.count == target.first;
                if (!next_element && begin != end) {
                    throw SimError(what + "over=: " + object.over[item] + " (at " + std::to_string(begin) +
                        ") does not follow " + object.over[item - 1] + " (which ends at " + std::to_string(end) +
                        "); the objects an overlay lists must be declared one after another, in that order" +
                        (begin > end ? ", and alignment left a gap here" : ""));
                }
            }
            end = begin + uint64_t{target.count - 1} * under.stride + under.size;
        }
        if (object.placed) {
            // An address given in assembly (at=): inside what it says it lies on.
            if (object.address < start || object.address + object_extent(object) > end) {
                throw SimError(what + "at=" + std::to_string(object.address) + ": it needs " +
                    std::to_string(object_extent(object)) + " " + unit_of(object) + " there, and the objects it lies over (" +
                    "over=) are at " + std::to_string(start) + " .. " + std::to_string(end));
            }
            continue;
        }
        if (start % object.alignment != 0) {
            throw SimError(what + "it would start at " + std::to_string(start) + " (" + object.over.front() +
                "), which is not a multiple of its alignment, " + std::to_string(object.alignment) + " " +
                unit_of(object));
        }
        if (start + object_extent(object) > end) {
            throw SimError(what + "it needs " + std::to_string(object_extent(object)) + " " + unit_of(object) +
                " and the objects it lies over have " + std::to_string(end - start));
        }
        object.address = static_cast<uint32_t>(start);
        object.placed = true;
    }
    // Every object, wherever its address came from (qtxc's placement above, or
    // an at= in hand-written assembly): on its alignment, inside its memory,
    // and no two overlapping unless one says so. An overlay was held inside
    // its targets above, so what is left is that the objects that lie over
    // nothing are apart.
    for (std::size_t index = 0; index < objects.size(); ++index) {
        const auto& object = objects[index];
        const auto what = at(object.line) + directive_of(object) + " " + object.name + ": ";
        const uint64_t limit = object.tmem ? tmem_words : shmem_limit;
        if (object.address % object.alignment != 0) {
            throw SimError(what + "it is at " + std::to_string(object.address) +
                ", which is not a multiple of its alignment, " + std::to_string(object.alignment) + " " + unit_of(object));
        }
        if (uint64_t{object.address} + object_extent(object) > limit) {
            throw SimError(what + "it does not fit: it needs " + std::to_string(object_extent(object)) + " " +
                unit_of(object) + " at " + std::to_string(object.address) + ", and " + (object.tmem
                    ? "this profile's tensor memory has " + std::to_string(limit) + " words"
                    : "shared memory below the launch block (.param / .mbar / .data, at the top) ends at " +
                      std::to_string(limit)));
        }
        if (!object.over.empty()) continue;
        for (std::size_t other = 0; other < index; ++other) {
            const auto& earlier = objects[other];
            if (!earlier.over.empty() || earlier.tmem != object.tmem) continue;
            const uint64_t begin = object.address;
            const uint64_t end = begin + object_extent(object);
            const uint64_t earlier_begin = earlier.address;
            const uint64_t earlier_end = earlier_begin + object_extent(earlier);
            if (begin < earlier_end && earlier_begin < end) {
                throw SimError(what + "it (" + std::to_string(begin) + " .. " + std::to_string(end) + ") overlaps " +
                    directive_of(earlier) + " " + earlier.name + " (" + std::to_string(earlier_begin) + " .. " +
                    std::to_string(earlier_end) + "); two objects share " + unit_of(object) +
                    " only when one says over= the other");
            }
        }
    }
}

// The line as assembly carries it: everything resolved, with the address.
std::string format_object_directive(const MemoryObject& object) {
    std::ostringstream out;
    out << directive_of(object) << ' ' << object.name;
    if (object.count != 1) out << '[' << object.count << ']';
    if (object.rows == 0) {
        out << ", bytes=" << object.size;
    } else {
        out << ", rows=" << object.rows << ", cols=" << object.cols;
        if (!object.tmem) out << ", elem=" << object.elem;
    }
    out << ", align=" << object.alignment;
    for (std::size_t index = 0; index < object.over.size(); ++index) {
        out << (index == 0 ? ", over=" : ", ") << object.over[index];
    }
    out << ", at=" << object.address;
    return out.str();
}

} // namespace qtxc_internal
} // namespace nanosm

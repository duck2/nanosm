// Constant expressions (src/qtxc/; docs/qtx.md "Memory"). Wherever qtx or assembly
// takes a constant -- an immediate, an address offset, an `.equ`, a
// directive's argument -- it may be an expression of numbers and names:
//
//     + - * / << >>   unary minus   ( )   log2(x)
//
// with C's precedence (* / over + - over << >>). log2 refuses an argument
// that is not a power of two; / refuses a zero divisor; nothing wraps (a
// value past +-2^40 anywhere in an expression is refused, before the step
// that would make it; whoever asked checks the range of the result).
// A name is an `.equ` constant, a launch section (its shared address), or a
// memory object (src/isa.hpp MemoryObject):
//
//     NAME          the object's address (its element 0)
//     NAME[i]       element i's address (an .mbar array's barrier i too);
//                   an index past the end is refused
//     NAME.bytes    the whole object's bytes; NAME[i].bytes one element's
//                   (.words for a tmem object)
//     NAME.stride   from one element to the next
//     NAME.row_bytes   a shared tile's row (cols x elem)
//     NAME.row_words   a tmem tile's row (cols)
//
// The names beyond the `.equ` constants come from whoever is parsing
// (ExpressionScope); with none set, an expression is numbers and constants.

#include "qtxc/internal.hpp"

#include <cctype>
#include <cstdlib>

namespace nanosm {
namespace qtxc_internal {

namespace {

thread_local const ExpressionNames* scope_names = nullptr;

struct Parser {
    const std::string& text;
    const std::unordered_map<std::string, int32_t>& constants;
    std::size_t at = 0;
    std::string problem;

    void skip() {
        while (at < text.size() && std::isspace(static_cast<unsigned char>(text[at])) != 0) ++at;
    }
    bool take(char c) {
        skip();
        if (at < text.size() && text[at] == c) {
            ++at;
            return true;
        }
        return false;
    }
    bool take(const char* two) {
        skip();
        if (at + 1 < text.size() && text[at] == two[0] && text[at + 1] == two[1]) {
            at += 2;
            return true;
        }
        return false;
    }
    bool fail(const std::string& why) {
        if (problem.empty()) problem = why;
        return false;
    }

    bool shift(int64_t& value) {
        if (!sum(value)) return false;
        for (;;) {
            const bool left = take("<<");
            if (!left && !take(">>")) return true;
            int64_t amount = 0;
            if (!sum(amount)) return false;
            if (amount < 0 || amount > 62) return fail("a shift count must be 0..62");
            if (left) {
                if (!multiply(value, amount > 40 ? limit + 1 : int64_t{1} << amount)) return false;
            } else {
                value >>= amount;
            }
        }
    }
    bool sum(int64_t& value) {
        if (!term(value)) return false;
        for (;;) {
            skip();
            if (at + 1 < text.size() && (text[at] == '<' || text[at] == '>')) return true;
            const bool plus = take('+');
            if (!plus && !take('-')) return true;
            int64_t right = 0;
            if (!term(right)) return false;
            value = plus ? value + right : value - right;
            if (!in_range(value)) return false;
        }
    }
    bool term(int64_t& value) {
        if (!unary(value)) return false;
        for (;;) {
            const bool times = take('*');
            if (!times && !take('/')) return true;
            int64_t right = 0;
            if (!unary(right)) return false;
            if (!times && right == 0) return fail("division by zero");
            if (times) {
                if (!multiply(value, right)) return false;
            } else {
                value /= right;
            }
        }
    }
    bool unary(int64_t& value) {
        if (take('-')) {
            if (!unary(value)) return false;
            value = -value;
            return true;
        }
        return primary(value);
    }
    // Every value stays within +-2^40, far inside 64 bits. A sum of two such
    // values cannot overflow and is checked after it is made; a product can,
    // so it is checked BEFORE the multiplication (a shift left is a product).
    static constexpr int64_t limit = int64_t{1} << 40;
    bool in_range(int64_t value) {
        if (value > limit || value < -limit) return fail("the value is out of range");
        return true;
    }
    bool multiply(int64_t& value, int64_t right) {
        const int64_t a = value < 0 ? -value : value;
        const int64_t b = right < 0 ? -right : right;
        if (b != 0 && a > limit / b) return fail("the value is out of range");
        value *= right;
        return true;
    }

    bool primary(int64_t& value) {
        skip();
        if (at >= text.size()) return fail("something is missing");
        if (take('(')) {
            if (!shift(value)) return false;
            if (!take(')')) return fail("a ')' is missing");
            return true;
        }
        const auto c = static_cast<unsigned char>(text[at]);
        if (std::isdigit(c) != 0) {
            std::size_t end = at;
            while (end < text.size() && std::isalnum(static_cast<unsigned char>(text[end])) != 0) ++end;
            const auto digits = text.substr(at, end - at);
            char* stop = nullptr;
            const auto number = std::strtoull(digits.c_str(), &stop, 0);
            if (stop != digits.c_str() + digits.size() || number > (uint64_t{1} << 40)) {
                return fail("'" + digits + "' is not a number");
            }
            at = end;
            value = static_cast<int64_t>(number);
            return true;
        }
        if (std::isalpha(c) == 0 && c != '_') return fail(std::string("unexpected '") + text[at] + "'");
        std::size_t end = at;
        while (end < text.size() &&
               (std::isalnum(static_cast<unsigned char>(text[end])) != 0 || text[end] == '_')) ++end;
        const auto name = text.substr(at, end - at);
        at = end;
        if (name == "log2") {
            skip();
            if (at < text.size() && text[at] == '(') {
                ++at;
                int64_t argument = 0;
                if (!shift(argument)) return false;
                if (!take(')')) return fail("a ')' is missing");
                if (argument <= 0 || (argument & (argument - 1)) != 0) {
                    return fail("log2(" + std::to_string(argument) + "): not a power of two");
                }
                value = 0;
                while ((int64_t{1} << value) < argument) ++value;
                return true;
            }
        }
        return named(name, value);
    }

    // NAME, NAME[i], and a property after either.
    bool named(const std::string& name, int64_t& value) {
        std::optional<int64_t> index;
        skip();
        if (at < text.size() && text[at] == '[') {
            ++at;
            int64_t element = 0;
            if (!shift(element)) return false;
            if (!take(']')) return fail("a ']' is missing");
            index = element;
        }
        std::string property;
        if (at < text.size() && text[at] == '.') {
            std::size_t end = at + 1;
            while (end < text.size() &&
                   (std::isalnum(static_cast<unsigned char>(text[end])) != 0 || text[end] == '_')) ++end;
            property = text.substr(at + 1, end - at - 1);
            at = end;
        }
        const auto written = name + (index.has_value() ? "[" + std::to_string(*index) + "]" : std::string());
        const MemoryObject* object = nullptr;
        const KernelSection* section = nullptr;
        if (scope_names != nullptr) {
            if (scope_names->objects != nullptr) {
                for (const auto& candidate : *scope_names->objects) {
                    if (candidate.name == name) object = &candidate;
                }
            }
            if (object == nullptr && scope_names->sections != nullptr) {
                for (const auto& candidate : *scope_names->sections) {
                    if (candidate.name == name) section = &candidate;
                }
            }
        }
        if (object != nullptr) {
            const std::string directive = object->tmem ? ".tmem " : ".shmem ";
            if (index.has_value() && (*index < 0 || *index >= object->count)) {
                return fail(written + " is past the end of " + directive + name + ", which has " +
                    std::to_string(object->count) + (object->count == 1 ? " element" : " elements"));
            }
            if (property.empty()) {
                if (!object->placed) {
                    return fail("the address of " + name + " is not known yet (an object's size and alignment "
                        "cannot depend on where objects are placed)");
                }
                value = int64_t{object->address} + (index.has_value() ? *index * object->stride : 0);
                return true;
            }
            const bool tile = object->rows != 0;
            const std::string size_word = object->tmem ? "words" : "bytes";
            // An object's own align= is read before its stride exists.
            if (object->stride == 0 && (property == "stride" || (property == size_word && !index.has_value()))) {
                return fail(name + "." + property + " is not known yet: it depends on the alignment being read "
                    "(use " + name + "[0]." + size_word + (tile ? " or its row" : "") + ")");
            }
            if (property == size_word) {
                value = index.has_value() ? object->size : object_extent(*object);
            } else if (property == "stride" && !index.has_value()) {
                value = object->stride;
            } else if (property == "row_bytes" && tile && !object->tmem) {
                value = object_row(*object);
            } else if (property == "row_words" && tile && object->tmem) {
                value = object_row(*object);
            } else {
                return fail("'" + written + "." + property + "': " + directive + name + " has ." + size_word +
                    " (" + name + "[i]." + size_word + " for one element), .stride" +
                    (tile ? std::string(object->tmem ? " and .row_words" : " and .row_bytes") : std::string()));
            }
            return true;
        }
        if (!property.empty()) {
            return fail("'" + written + "." + property + "': only a .shmem or .tmem object has properties");
        }
        if (index.has_value()) {
            if (section == nullptr || section->kind != SectionKind::mbar) {
                return fail("'" + written + "': " + name + " is not an array (.mbar NAME[C], .shmem NAME[C] and "
                    ".tmem NAME[C] are)");
            }
            if (*index < 0 || *index >= section->count) {
                return fail(written + " is past the end of .mbar " + name + ", which has " +
                    std::to_string(section->count) + (section->count == 1 ? " barrier" : " barriers"));
            }
            if (!scope_names->addresses_known) return fail("the address of " + name + " is not known yet");
            value = int64_t{section->address} + 8 * *index;
            return true;
        }
        if (const auto found = constants.find(name); found != constants.end()) {
            value = found->second;
            return true;
        }
        return fail("unknown name '" + name + "'");
    }
};

} // namespace

ExpressionScope::ExpressionScope(const ExpressionNames* names) : previous_(scope_names) {
    scope_names = names;
}

ExpressionScope::~ExpressionScope() {
    scope_names = previous_;
}

std::optional<int64_t> evaluate_expression(
    const std::string& text, const std::unordered_map<std::string, int32_t>& constants, std::string& problem) {
    Parser parser{text, constants, 0, {}};
    int64_t value = 0;
    if (parser.shift(value)) {
        parser.skip();
        if (parser.at == text.size()) return value;
        parser.fail(std::string("unexpected '") + text[parser.at] + "'");
    }
    problem = parser.problem;
    return std::nullopt;
}

} // namespace qtxc_internal
} // namespace nanosm

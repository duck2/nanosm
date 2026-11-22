#include "image.hpp"

#include "compiler.hpp"
#include "mbar.hpp"
#include "profile.hpp"
#include "tma.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <new>
#include <optional>
#include <sstream>

namespace nanosm {
namespace {

using json = nlohmann::json;

std::string hex_u64(uint64_t value) {
    std::ostringstream out;
    out << "0x" << std::hex << std::setw(16) << std::setfill('0') << value;
    return out.str();
}

uint8_t parse_hex_digit(char value, const std::string& field);

uint64_t parse_hex_u64(const std::string& text, const std::string& field) {
    if (text.size() != 18 || text[0] != '0' || text[1] != 'x') {
        throw SimError("launch image: " + field + " must be 0x followed by 16 hex digits");
    }
    uint64_t result = 0;
    for (std::size_t index = 2; index < text.size(); ++index) {
        result = (result << 4) | parse_hex_digit(text[index], field);
    }
    return result;
}

char hex_digit(uint8_t value) {
    return value < 10 ? static_cast<char>('0' + value) : static_cast<char>('a' + value - 10);
}

uint8_t parse_hex_digit(char value, const std::string& field) {
    if (value >= '0' && value <= '9') return static_cast<uint8_t>(value - '0');
    if (value >= 'a' && value <= 'f') return static_cast<uint8_t>(value - 'a' + 10);
    if (value >= 'A' && value <= 'F') return static_cast<uint8_t>(value - 'A' + 10);
    throw SimError("launch image: " + field + " contains a non-hex character");
}

std::string encode_bytes(const std::vector<uint8_t>& bytes) {
    if (bytes.size() > std::numeric_limits<std::size_t>::max() / 2) {
        throw SimError("launch image: memory segment is too large to encode");
    }
    std::string result(bytes.size() * 2, '0');
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        result[i * 2] = hex_digit(bytes[i] >> 4);
        result[i * 2 + 1] = hex_digit(bytes[i] & 0x0f);
    }
    return result;
}

std::vector<uint8_t> decode_bytes(const std::string& text, const std::string& field) {
    if (text.empty() || text.size() % 2 != 0) {
        throw SimError("launch image: " + field + " must contain a nonempty even number of hex digits");
    }
    std::vector<uint8_t> result(text.size() / 2);
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = static_cast<uint8_t>(
            (parse_hex_digit(text[i * 2], field) << 4) |
            parse_hex_digit(text[i * 2 + 1], field));
    }
    return result;
}

json emit_segments(const std::vector<ImageSegment>& segments) {
    auto result = json::array();
    for (const auto& segment : segments) {
        if (segment.bytes.empty()) throw SimError("launch image: empty memory segment");
        result.push_back({
            {"address", segment.address},
            {"bytes", encode_bytes(segment.bytes)},
        });
    }
    return result;
}

std::vector<ImageSegment> load_segments(const json& value, const std::string& name) {
    if (!value.is_array()) throw SimError("launch image: " + name + " is not an array");
    std::vector<ImageSegment> result;
    result.reserve(value.size());
    for (std::size_t index = 0; index < value.size(); ++index) {
        const auto field = name + "[" + std::to_string(index) + "]";
        const auto& item = value.at(index);
        ImageSegment segment;
        try {
            // Checked before it is read: get<uint64_t> is a bare cast (-8 or
            // 4160.9 would become some address).
            if (!item.at("address").is_number_unsigned()) {
                throw SimError("launch image: " + field + ".address is not an unsigned integer");
            }
            segment.address = item.at("address").get<uint64_t>();
            segment.bytes = decode_bytes(item.at("bytes").get<std::string>(), field + ".bytes");
        } catch (const json::exception& error) {
            throw SimError("launch image: " + field + ": " + error.what());
        }
        if (segment.address + segment.bytes.size() < segment.address) {
            throw SimError("launch image: " + field + " address range overflows");
        }
        result.push_back(std::move(segment));
    }
    return result;
}

// An unsigned 32-bit integer field, checked BEFORE it is narrowed (nlohmann's
// get<uint32_t> is a bare cast: 4294967296 would become 0, 4160.9 4160 and
// -8 0xfffffff8). Backlog C11 (mbars) and C2 (active_warps).
std::optional<uint32_t> unsigned_32(const json& value) {
    if (!value.is_number_unsigned()) return std::nullopt;
    const auto wide = value.get<uint64_t>();
    if (wide > std::numeric_limits<uint32_t>::max()) return std::nullopt;
    return static_cast<uint32_t>(wide);
}

// Optional: a launch that does not say takes the kernel's `.warps`
// (bind_launch); 0 is that absence, never written.
uint32_t load_active_warps(const json& document, const std::string& format) {
    if (!document.contains("active_warps")) return 0;
    const auto warps = unsigned_32(document.at("active_warps"));
    if (!warps.has_value()) throw SimError(format + ": active warp count is not an unsigned 32-bit integer");
    if (*warps == 0) throw SimError(format + ": active warp count is zero");
    return *warps;
}

void emit_active_warps(json& document, uint32_t warps) {
    if (warps != 0) document["active_warps"] = warps;
}

// "args": {"NAME": value}: an unsigned integer, a number, or the hex text of
// bytes (a descriptor). Emitted only when there are arguments.
void emit_args(json& document, const std::vector<LaunchArgument>& args) {
    if (args.empty()) return;
    auto object = json::object();
    for (const auto& arg : args) {
        if (object.contains(arg.name)) throw SimError("launch: argument '" + arg.name + "' is given twice");
        if (const auto* integer = std::get_if<uint64_t>(&arg.value)) object[arg.name] = *integer;
        else if (const auto* number = std::get_if<double>(&arg.value)) {
            if (!std::isfinite(*number)) {
                throw SimError("launch: argument '" + arg.name + "' is not a finite number");
            }
            object[arg.name] = *number;
        } else object[arg.name] = encode_bytes(std::get<std::vector<uint8_t>>(arg.value));
    }
    document["args"] = std::move(object);
}

std::vector<LaunchArgument> load_args(const json& document) {
    if (!document.contains("args")) return {};
    const auto& value = document.at("args");
    if (!value.is_object()) throw SimError("launch: args is not an object of name: value");
    std::vector<LaunchArgument> args;
    for (const auto& [name, item] : value.items()) {
        LaunchArgument arg;
        arg.name = name;
        if (item.is_number_unsigned()) arg.value = item.get<uint64_t>();
        else if (item.is_number()) arg.value = item.get<double>();
        else if (item.is_string()) arg.value = decode_bytes(item.get<std::string>(), "args." + name);
        else {
            throw SimError("launch: argument '" + name +
                "' is not an unsigned integer, a number or the hex text of bytes");
        }
        args.push_back(std::move(arg));
    }
    return args;
}

// What the kernel declares beside its words (src/isa.hpp Kernel), in a saved
// launch image; each key is written only when the kernel has it.
void emit_kernel_declarations(json& kernel_json, const Kernel& kernel) {
    if (kernel.warps != 0) kernel_json["warps"] = kernel.warps;
    if (!kernel.sections.empty()) {
        auto sections = json::array();
        for (const auto& section : kernel.sections) {
            json item = {{"name", section.name}, {"address", section.address}};
            switch (section.kind) {
            case SectionKind::param:
                item["kind"] = "param";
                item["type"] = param_type_name(section.type);
                break;
            case SectionKind::mbar:
                item["kind"] = "mbar";
                item["expect"] = section.expected;
                if (section.count != 1) item["count"] = section.count; // an array: NAME[count]
                break;
            case SectionKind::data:
                item["kind"] = "data";
                item["bytes"] = encode_bytes(section.bytes);
                break;
            }
            sections.push_back(std::move(item));
        }
        kernel_json["sections"] = std::move(sections);
    }
    if (!kernel.flags.empty()) {
        auto flags = json::array();
        for (const auto& flag : kernel.flags) flags.push_back({{"bar", flag.bar}, {"bytes", flag.bytes}});
        kernel_json["flags"] = std::move(flags);
    }
}

uint32_t declared_u32(const json& item, const char* key, const std::string& field) {
    const auto value = unsigned_32(item.at(key));
    if (!value.has_value()) throw SimError("launch image: " + field + "." + key + " is not an unsigned 32-bit integer");
    return *value;
}

void load_kernel_declarations(const json& kernel_json, Kernel& kernel) {
    if (kernel_json.contains("warps")) kernel.warps = declared_u32(kernel_json, "warps", "kernel");
    if (kernel_json.contains("params") || kernel_json.contains("data") || kernel_json.contains("mbars")) {
        throw SimError("launch image: kernel.params / data / mbars are kernel.sections since 2026-11-21 "
            "(the launch block); rebuild the image");
    }
    if (kernel_json.contains("sections")) {
        for (const auto& item : kernel_json.at("sections")) {
            KernelSection section;
            section.name = item.at("name").get<std::string>();
            const auto field = "kernel.sections." + section.name;
            section.address = declared_u32(item, "address", field);
            const auto kind = item.at("kind").get<std::string>();
            if (kind == "param") {
                section.kind = SectionKind::param;
                const auto type = item.at("type").get<std::string>();
                if (type == "ptr") section.type = ParamType::ptr;
                else if (type == "u32") section.type = ParamType::u32;
                else if (type == "f32") section.type = ParamType::f32;
                else if (type == "desc") section.type = ParamType::desc;
                else throw SimError("launch image: kernel parameter '" + section.name + "' has unknown type '" + type + "'");
            } else if (kind == "mbar") {
                section.kind = SectionKind::mbar;
                const auto expected = declared_u32(item, "expect", field);
                if (expected == 0 || expected > mbar_encoding::max_count) {
                    throw SimError("launch image: " + field + ": expected count must be 1..32767");
                }
                section.expected = static_cast<uint16_t>(expected);
                if (item.contains("count")) {
                    section.count = declared_u32(item, "count", field);
                    if (section.count == 0 || section.count > max_mbar_array) {
                        throw SimError("launch image: " + field + ": an mbar array has 1.." +
                            std::to_string(max_mbar_array) + " barriers");
                    }
                }
            } else if (kind == "data") {
                section.kind = SectionKind::data;
                section.bytes = decode_bytes(item.at("bytes").get<std::string>(), field + ".bytes");
            } else {
                throw SimError("launch image: " + field + " has unknown kind '" + kind + "'");
            }
            kernel.sections.push_back(std::move(section));
        }
    }
    if (kernel_json.contains("flags")) {
        for (const auto& item : kernel_json.at("flags")) {
            LaunchFlag flag;
            const auto bar = declared_u32(item, "bar", "kernel.flags");
            if (bar >= 6) throw SimError("launch image: kernel.flags: bar must be 0..5");
            flag.bar = static_cast<uint8_t>(bar);
            flag.bytes = declared_u32(item, "bytes", "kernel.flags");
            kernel.flags.push_back(flag);
        }
    }
}

void append_le(std::vector<uint8_t>& bytes, uint64_t value, unsigned count) {
    for (unsigned index = 0; index < count; ++index) bytes.push_back(static_cast<uint8_t>(value >> (8 * index)));
}

// The optional declared-mbar list (Launch::mbars): emitted only when there is
// one, so launches without declared mbars keep their exact text.
void emit_mbars(json& document, const std::vector<uint32_t>& mbars) {
    if (!mbars.empty()) document["mbars"] = mbars;
}

std::vector<uint32_t> load_mbars(const json& document) {
    if (!document.contains("mbars")) return {};
    const auto& value = document.at("mbars");
    if (!value.is_array()) throw SimError("launch: mbars is not an array of shmem addresses");
    std::vector<uint32_t> mbars;
    for (std::size_t index = 0; index < value.size(); ++index) {
        const auto address = unsigned_32(value.at(index));
        if (!address.has_value()) {
            throw SimError("launch: mbars[" + std::to_string(index) + "] is not a shmem address");
        }
        mbars.push_back(*address);
    }
    return mbars;
}

// Where the host puts the launch block in global memory (Launch::
// block_address); written only when the launch chose one.
void emit_block_address(json& document, const std::optional<uint64_t>& address) {
    if (address.has_value()) document["block_address"] = *address;
}

std::optional<uint64_t> load_block_address(const json& document) {
    if (!document.contains("block_address")) return std::nullopt;
    const auto& value = document.at("block_address");
    if (!value.is_number_unsigned()) throw SimError("launch: block_address is not an unsigned integer");
    return value.get<uint64_t>();
}

} // namespace

Launch bind_launch(const Kernel& kernel, Launch launch, const DeviceProfile& profile) {
    check_kernel_image(kernel, profile);
    if (launch.active_warps == 0) {
        if (kernel.warps == 0) {
            throw SimError("the launch gives no warp count and the kernel declares none (.warps)");
        }
        launch.active_warps = kernel.warps;
    } else if (kernel.warps != 0 && launch.active_warps != kernel.warps) {
        throw SimError("the launch asks for " + std::to_string(launch.active_warps) +
            " warps; the kernel is written for " + std::to_string(kernel.warps) + " (.warps)");
    }

    std::string signature;
    for (const auto& section : kernel.sections) {
        if (section.kind != SectionKind::param) continue;
        signature += (signature.empty() ? "" : ", ") + std::string(param_type_name(section.type)) + ' ' + section.name;
    }
    for (std::size_t index = 0; index < launch.args.size(); ++index) {
        const auto& name = launch.args[index].name;
        for (std::size_t other = 0; other < index; ++other) {
            if (launch.args[other].name == name) throw SimError("launch argument '" + name + "' is given twice");
        }
        const bool known = std::any_of(kernel.sections.begin(), kernel.sections.end(),
            [&](const KernelSection& section) { return section.kind == SectionKind::param && section.name == name; });
        if (!known) {
            throw SimError("launch argument '" + name + "' names no parameter of the kernel (" +
                (signature.empty() ? "it declares none" : "its signature: " + signature) + ")");
        }
    }
    launch.record.reset();
    if (kernel.sections.empty()) {
        launch.args.clear();
        return launch;
    }

    // The launch block's bytes, section by section (check_kernel_image has
    // settled that it fits and where each section sits).
    const auto block = *launch_block(kernel.sections, profile.shmem_bytes);
    std::vector<uint8_t> bytes(block.bytes, 0);
    std::vector<uint32_t> mbars;
    for (const auto& section : kernel.sections) {
        std::vector<uint8_t> piece;
        switch (section.kind) {
        case SectionKind::data:
            piece = section.bytes;
            break;
        case SectionKind::mbar: {
            MbarState state;
            state.valid = true;
            state.expected = section.expected;
            state.remaining = section.expected;
            for (uint32_t element = 0; element < section.count; ++element) {
                append_le(piece, encode_mbar_state(state), 8);
                mbars.push_back(section.address + 8 * element);
            }
            break;
        }
        case SectionKind::param: {
            const auto found = std::find_if(launch.args.begin(), launch.args.end(),
                [&](const LaunchArgument& arg) { return arg.name == section.name; });
            const auto what = "launch argument '" + section.name + "' (" + param_type_name(section.type) + ")";
            if (found == launch.args.end()) throw SimError(what + " is missing");
            const auto* integer = std::get_if<uint64_t>(&found->value);
            const auto* number = std::get_if<double>(&found->value);
            const auto* given = std::get_if<std::vector<uint8_t>>(&found->value);
            switch (section.type) {
            case ParamType::ptr:
                if (integer == nullptr) throw SimError(what + " must be an unsigned integer, a global byte address");
                // The parameter is 8 bytes; ldg and stg take their whole address
                // from one 32-bit register until they take a second one.
                if (*integer > 0xffffffffull) {
                    throw SimError(what + " = " + std::to_string(*integer) +
                        " does not fit 32 bits: ldg / stg take their address from one 32-bit register for now");
                }
                append_le(piece, *integer, 8);
                break;
            case ParamType::u32:
                if (integer == nullptr || *integer > 0xffffffffull) {
                    throw SimError(what + " must be an unsigned integer that fits 32 bits");
                }
                append_le(piece, *integer, 4);
                break;
            case ParamType::f32: {
                if (given != nullptr) throw SimError(what + " must be a number");
                const float value = integer != nullptr ? static_cast<float>(*integer) : static_cast<float>(*number);
                uint32_t bits = 0;
                std::memcpy(&bits, &value, sizeof(bits));
                append_le(piece, bits, 4);
                break;
            }
            case ParamType::desc:
                if (given == nullptr || given->size() != tma_descriptor_bytes) {
                    throw SimError(what + " must be a TMA descriptor, 64 bytes");
                }
                piece = *given;
                break;
            }
            break;
        }
        }
        std::copy(piece.begin(), piece.end(), bytes.begin() + (section.address - block.start));
    }
    launch.args.clear();
    mbars.insert(mbars.end(), launch.mbars.begin(), launch.mbars.end());
    launch.mbars = std::move(mbars);

    // A test's preload (Launch::test_shmem) is untimed; it may not stand in
    // for the launch block, whose bytes arrive by the copy.
    for (const auto& segment : launch.test_shmem) {
        if (segment.address < uint64_t{block.start} + block.bytes &&
            block.start < segment.address + segment.bytes.size()) {
            throw SimError("a test's shared-memory preload at " + std::to_string(segment.address) + " (" +
                std::to_string(segment.bytes.size()) + " bytes) touches the kernel's launch block (" +
                std::to_string(block.start) + ".." + std::to_string(block.start + block.bytes) +
                "): those bytes arrive by the launch copy");
        }
    }

    // The host writes the block into global memory and says where in the
    // launch record; the machine copies it from the first cycle on.
    const uint64_t sector = profile.gmem_sector_bytes;
    if (profile.gmem_bytes < block.bytes) {
        throw SimError("the kernel's launch block (" + std::to_string(block.bytes) + " bytes) does not fit global memory");
    }
    const uint64_t address = launch.block_address.has_value()
        ? *launch.block_address : (profile.gmem_bytes - block.bytes) / sector * sector;
    if (address % 4 != 0 || address > profile.gmem_bytes - block.bytes) {
        throw SimError("the launch block's global address " + std::to_string(address) + " must be a multiple of 4 with " +
            std::to_string(block.bytes) + " bytes inside global memory (" + std::to_string(profile.gmem_bytes) + ")");
    }
    for (const auto& segment : launch.gmem_image) {
        if (segment.address < address + block.bytes && address < segment.address + segment.bytes.size()) {
            throw SimError("the launch block (global " + std::to_string(address) + ".." +
                std::to_string(address + block.bytes) + ") overlaps the launch's own global memory at " +
                std::to_string(segment.address) + "; give the launch another block address");
        }
    }
    launch.gmem_image.push_back({address, std::move(bytes)});
    LaunchRecord record;
    record.global_address = address;
    record.shared_address = block.start;
    record.bytes = block.bytes;
    record.flags = kernel.flags;
    launch.record = std::move(record);
    return launch;
}

std::string emit_kernel_info(const Kernel& kernel) {
    auto params = json::array();
    auto sections = json::array();
    // The block starts at its first section and ends after its last.
    const uint32_t start = kernel.sections.empty() ? 0 : kernel.sections.front().address;
    const uint32_t end = kernel.sections.empty() ? 0
        : kernel.sections.back().address + section_bytes(kernel.sections.back());
    for (const auto& section : kernel.sections) {
        if (section.kind == SectionKind::param) {
            params.push_back({
                {"name", section.name},
                {"type", param_type_name(section.type)},
                {"address", section.address},
                {"bytes", param_bytes(section.type)},
            });
        }
        json item = {
            {"name", section.name},
            {"kind", section.kind == SectionKind::param ? "param" : section.kind == SectionKind::mbar ? "mbar" : "data"},
            {"address", section.address},
            {"bytes", section_bytes(section)},
        };
        if (section.kind == SectionKind::mbar) item["count"] = section.count;
        if (const auto bar = launch_flag_covering(kernel.flags, section.address - start + section_bytes(section));
            bar.has_value()) {
            item["flag"] = *bar;
        }
        sections.push_back(std::move(item));
    }
    auto flags = json::array();
    for (const auto& flag : kernel.flags) flags.push_back({{"bar", flag.bar}, {"bytes", flag.bytes}});
    auto symbols = json::object();
    for (const auto& [name, value] : kernel.symbols) symbols[name] = value;
    // The qtx's register names: "%x" -> "r5" / "ur6".
    auto registers = json::object();
    for (const auto& entry : kernel.register_names) {
        registers["%" + entry.name] =
            (entry.predicate ? "p" : entry.uniform ? "ur" : "r") + std::to_string(entry.number);
    }
    json document = {
        {"warps", kernel.warps},
        {"params", std::move(params)},
        {"sections", std::move(sections)},
        {"block", {{"address", start}, {"bytes", (end - start + 3) / 4 * 4}}},
        {"flags", std::move(flags)},
        {"symbols", std::move(symbols)},
        {"registers", std::move(registers)},
        {"warnings", kernel.warnings},
    };
    return document.dump(2) + '\n';
}

std::string emit_launch_state(const Launch& launch) {
    json document = {
        {"format", "nanosm-launch-state-v0"},
        {"test_shmem", emit_segments(launch.test_shmem)},
        {"gmem", emit_segments(launch.gmem_image)},
    };
    emit_active_warps(document, launch.active_warps);
    emit_args(document, launch.args);
    emit_mbars(document, launch.mbars);
    emit_block_address(document, launch.block_address);
    return document.dump(2) + '\n';
}

Launch load_launch_state(std::string_view source) {
    try {
        const auto document = json::parse(source);
        if (document.at("format").get<std::string>() != "nanosm-launch-state-v0") {
            throw SimError("launch state: unsupported format");
        }
        Launch launch;
        launch.active_warps = load_active_warps(document, "launch state");
        launch.test_shmem = load_segments(document.at("test_shmem"), "test_shmem");
        launch.gmem_image = load_segments(document.at("gmem"), "gmem");
        launch.mbars = load_mbars(document);
        launch.args = load_args(document);
        launch.block_address = load_block_address(document);
        return launch;
    } catch (const SimError&) {
        throw;
    } catch (const json::exception& error) {
        throw SimError(std::string("launch state: ") + error.what());
    } catch (const std::bad_alloc&) {
        throw SimError("launch state does not fit in host memory");
    } catch (const std::length_error&) {
        throw SimError("launch state exceeds host container limits");
    }
}

std::string emit_launch_image(const LaunchImage& image) {
    validate_kernel(image.kernel);
    auto words = json::array();
    for (const auto word : image.kernel.words) words.push_back(hex_u64(word));
    json document = {
        {"format", "nanosm-launch-v0"},
        {"profile", {
            {"name", image.kernel.profile_name},
            {"fingerprint", hex_u64(image.kernel.profile_fingerprint)},
        }},
        {"kernel", {
            {"register_count", image.kernel.register_count},
            {"words", std::move(words)},
        }},
        {"test_shmem", emit_segments(image.launch.test_shmem)},
        {"gmem", emit_segments(image.launch.gmem_image)},
    };
    emit_kernel_declarations(document["kernel"], image.kernel);
    emit_active_warps(document, image.launch.active_warps);
    emit_args(document, image.launch.args);
    emit_mbars(document, image.launch.mbars);
    emit_block_address(document, image.launch.block_address);
    return document.dump(2) + '\n';
}

LaunchImage load_launch_image(std::string_view source) {
    try {
        const auto document = json::parse(source);
        if (document.at("format").get<std::string>() != "nanosm-launch-v0") {
            throw SimError("launch image: unsupported format");
        }
        LaunchImage image;
        const auto& profile = document.at("profile");
        image.kernel.profile_name = profile.at("name").get<std::string>();
        image.kernel.profile_fingerprint = parse_hex_u64(
            profile.at("fingerprint").get<std::string>(), "profile.fingerprint");
        const auto& kernel = document.at("kernel");
        const auto register_count = unsigned_32(kernel.at("register_count"));
        if (!register_count.has_value()) {
            throw SimError("launch image: kernel.register_count is not an unsigned 32-bit integer");
        }
        if (*register_count > 256) throw SimError("launch image: register count exceeds 256");
        image.kernel.register_count = static_cast<uint16_t>(*register_count);
        // An image of a kernel built before 2026-11-19 carried an accumulator
        // table; its mma.async / acc.ld / acc.st words mean something else now.
        if (kernel.contains("accumulators")) {
            throw SimError("launch image: kernel.accumulators is no longer part of a kernel (an accumulator "
                "is a tmem pointer in a uniform register since 2026-11-19); rebuild the image");
        }
        const auto& words = kernel.at("words");
        if (!words.is_array()) throw SimError("launch image: kernel.words is not an array");
        image.kernel.words.reserve(words.size());
        for (std::size_t index = 0; index < words.size(); ++index) {
            image.kernel.words.push_back(parse_hex_u64(
                words.at(index).get<std::string>(),
                "kernel.words[" + std::to_string(index) + "]"));
        }
        load_kernel_declarations(kernel, image.kernel);
        image.launch.active_warps = load_active_warps(document, "launch image");
        image.launch.args = load_args(document);
        image.launch.test_shmem = load_segments(document.at("test_shmem"), "test_shmem");
        image.launch.gmem_image = load_segments(document.at("gmem"), "gmem");
        image.launch.mbars = load_mbars(document);
        image.launch.block_address = load_block_address(document);
        validate_kernel(image.kernel);
        return image;
    } catch (const SimError&) {
        throw;
    } catch (const json::exception& error) {
        throw SimError(std::string("launch image: ") + error.what());
    } catch (const std::bad_alloc&) {
        throw SimError("launch image does not fit in host memory");
    } catch (const std::length_error&) {
        throw SimError("launch image exceeds host container limits");
    }
}

LaunchImage load_launch_image_file(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw SimError("cannot open launch image '" + path + "'");
    std::ostringstream contents;
    contents << input.rdbuf();
    return load_launch_image(contents.str());
}

} // namespace nanosm

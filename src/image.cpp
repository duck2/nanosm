#include "image.hpp"

#include "compiler.hpp"
#include "profile.hpp"

#include <nlohmann/json.hpp>

#include <fstream>
#include <iomanip>
#include <limits>
#include <new>
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

} // namespace

std::string emit_launch_state(const Launch& launch) {
    if (launch.active_warps == 0) throw SimError("launch state: active warp count is zero");
    const json document = {
        {"format", "nanosm-launch-state-v0"},
        {"active_warps", launch.active_warps},
        {"shmem", emit_segments(launch.shmem_image)},
        {"gmem", emit_segments(launch.gmem_image)},
    };
    return document.dump(2) + '\n';
}

Launch load_launch_state(std::string_view source) {
    try {
        const auto document = json::parse(source);
        if (document.at("format").get<std::string>() != "nanosm-launch-state-v0") {
            throw SimError("launch state: unsupported format");
        }
        Launch launch;
        launch.active_warps = document.at("active_warps").get<uint32_t>();
        if (launch.active_warps == 0) throw SimError("launch state: active warp count is zero");
        launch.shmem_image = load_segments(document.at("shmem"), "shmem");
        launch.gmem_image = load_segments(document.at("gmem"), "gmem");
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
    if (image.launch.active_warps == 0) throw SimError("launch image: active warp count is zero");
    auto words = json::array();
    for (const auto word : image.kernel.words) words.push_back(hex_u64(word));
    auto accumulators = json::array();
    for (const auto& acc : image.kernel.accumulators) {
        accumulators.push_back({{"name", acc.name}, {"bank", acc.bank},
            {"base_word", acc.base_word}, {"rows", acc.rows}, {"cols", acc.cols}});
    }
    const json document = {
        {"format", "nanosm-launch-v0"},
        {"profile", {
            {"name", image.kernel.profile_name},
            {"fingerprint", hex_u64(image.kernel.profile_fingerprint)},
        }},
        {"kernel", {
            {"register_count", image.kernel.register_count},
            {"accumulators", std::move(accumulators)},
            {"words", std::move(words)},
        }},
        {"active_warps", image.launch.active_warps},
        {"shmem", emit_segments(image.launch.shmem_image)},
        {"gmem", emit_segments(image.launch.gmem_image)},
    };
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
        const auto register_count = kernel.at("register_count").get<uint32_t>();
        if (register_count > 256) throw SimError("launch image: register count exceeds 256");
        image.kernel.register_count = static_cast<uint16_t>(register_count);
        if (kernel.contains("accumulators")) {
            const auto& accumulators = kernel.at("accumulators");
            if (!accumulators.is_array() || accumulators.size() > 256)
                throw SimError("launch image: bad accumulator metadata array");
            for (const auto& item : accumulators) {
                Kernel::Accumulator acc;
                acc.name = item.at("name").get<std::string>();
                acc.bank = item.at("bank").get<uint8_t>();
                acc.base_word = item.at("base_word").get<uint32_t>();
                acc.rows = item.at("rows").get<uint16_t>();
                acc.cols = item.at("cols").get<uint16_t>();
                image.kernel.accumulators.push_back(std::move(acc));
            }
        }
        const auto& words = kernel.at("words");
        if (!words.is_array()) throw SimError("launch image: kernel.words is not an array");
        image.kernel.words.reserve(words.size());
        for (std::size_t index = 0; index < words.size(); ++index) {
            image.kernel.words.push_back(parse_hex_u64(
                words.at(index).get<std::string>(),
                "kernel.words[" + std::to_string(index) + "]"));
        }
        image.launch.active_warps = document.at("active_warps").get<uint32_t>();
        if (image.launch.active_warps == 0) {
            throw SimError("launch image: active warp count is zero");
        }
        image.launch.shmem_image = load_segments(document.at("shmem"), "shmem");
        image.launch.gmem_image = load_segments(document.at("gmem"), "gmem");
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

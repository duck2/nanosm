#include "capi.h"

#include "qtxc/qtxc.hpp"
#include "image.hpp"
#include "mbar.hpp"
#include "profile.hpp"
#include "machine/machine.hpp"
#include "tma_descriptor.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace {

struct Session {
    std::string launch_image;
    std::unique_ptr<nanosm::Machine> machine;
    bool ran = false;
};

thread_local std::string last_error;
std::unordered_map<uint32_t, std::unique_ptr<Session>> sessions;
uint32_t next_handle = 1;

void fail(std::string message) {
    last_error = std::move(message);
}

std::string_view input(const char* data, uint64_t size, const char* name) {
    if (data == nullptr && size != 0) throw nanosm::SimError(std::string(name) + " pointer is null");
    if (size > std::numeric_limits<std::size_t>::max()) {
        throw nanosm::SimError(std::string(name) + " is too large for this host");
    }
    return {data == nullptr ? "" : data, static_cast<std::size_t>(size)};
}

nanosm::TraceOptions trace_options(uint32_t flags, uint64_t max_events) {
    if ((flags & ~(NANOSM_TRACE_TIMELINE | NANOSM_TRACE_DEBUG)) != 0) {
        throw nanosm::SimError("unknown trace flag");
    }
    if (max_events > std::numeric_limits<std::size_t>::max()) {
        throw nanosm::SimError("trace event limit is too large for this host");
    }
    nanosm::TraceOptions options;
    options.timeline = (flags & (NANOSM_TRACE_TIMELINE | NANOSM_TRACE_DEBUG)) != 0;
    options.debug = (flags & NANOSM_TRACE_DEBUG) != 0;
    if (max_events != 0) options.max_events = static_cast<std::size_t>(max_events);
    return options;
}

// Test seam (scripts/driver_shuffle_check.py): with NANOSM_DRIVER_SHUFFLE_SEED
// set, every session runs its register drivers in a random order each cycle.
// The results must not change; see Machine::set_driver_shuffle_seed.
void apply_test_environment(nanosm::Machine& machine) {
    const char* seed = std::getenv("NANOSM_DRIVER_SHUFFLE_SEED");
    if (seed == nullptr || *seed == '\0') return;
    machine.set_driver_shuffle_seed(std::stoull(seed, nullptr, 0));
}

Session& session(uint32_t handle) {
    const auto found = sessions.find(handle);
    if (handle == NANOSM_INVALID_HANDLE || found == sessions.end()) {
        throw nanosm::SimError("invalid nanosm session handle");
    }
    return *found->second;
}

uint32_t insert(std::unique_ptr<Session> value) {
    while (next_handle == NANOSM_INVALID_HANDLE || sessions.contains(next_handle)) ++next_handle;
    const auto handle = next_handle++;
    sessions.emplace(handle, std::move(value));
    return handle;
}

uint64_t copy_string(const std::string& value, char* output, uint64_t capacity) {
    const auto required = static_cast<uint64_t>(value.size());
    if (output != nullptr && capacity > required) {
        std::memcpy(output, value.data(), value.size());
        output[value.size()] = '\0';
    }
    return required;
}

int32_t read_memory(
    uint32_t handle,
    uint64_t address,
    uint8_t* output,
    uint64_t size,
    bool global) {
    try {
        last_error.clear();
        auto& value = session(handle);
        const auto& memory = global ? value.machine->gmem() : value.machine->shmem();
        if (output == nullptr && size != 0) throw nanosm::SimError("memory output pointer is null");
        if (address > memory.size() || size > memory.size() - address) {
            throw nanosm::SimError(global ? "gmem read is out of bounds" : "shmem read is out of bounds");
        }
        std::memcpy(output, memory.data() + static_cast<std::size_t>(address), static_cast<std::size_t>(size));
        return NANOSM_OK;
    } catch (const std::exception& error) {
        fail(error.what());
        return NANOSM_ERROR;
    }
}

} // namespace

int32_t nanosm_mbar_encode(
    uint32_t phase,
    uint32_t expected,
    uint32_t remaining,
    int32_t tx_bytes,
    uint64_t* output) {
    try {
        last_error.clear();
        if (output == nullptr) throw nanosm::SimError("mbar output pointer is null");
        if (phase > 1) throw nanosm::SimError("mbar phase must be 0 or 1");
        if (expected == 0 || expected > nanosm::mbar_encoding::max_count) {
            throw nanosm::SimError("mbar expected count is out of range");
        }
        if (remaining > expected) {
            throw nanosm::SimError("mbar remaining count exceeds expected count");
        }
        if (tx_bytes > nanosm::mbar_encoding::tx_limit || tx_bytes < -nanosm::mbar_encoding::tx_limit) {
            throw nanosm::SimError("mbar transaction count is outside -(2^20 - 1)..2^20 - 1");
        }
        nanosm::MbarState state;
        state.phase = phase != 0;
        state.valid = true;
        state.expected = static_cast<uint16_t>(expected);
        state.remaining = static_cast<uint16_t>(remaining);
        state.tx_bytes = tx_bytes;
        *output = nanosm::encode_mbar_state(state);
        return NANOSM_OK;
    } catch (const std::exception& error) {
        fail(error.what());
        return NANOSM_ERROR;
    }
}

int32_t nanosm_tma_descriptor_encode(
    uint64_t base,
    uint32_t element_bytes,
    uint32_t d0,
    uint32_t d1,
    uint32_t d2,
    uint32_t d3,
    uint32_t s1,
    uint32_t s2,
    uint32_t s3,
    uint32_t tile_length,
    uint32_t tile_rows,
    uint32_t row_dimension,
    uint32_t layout,
    uint32_t transposed,
    uint8_t* output,
    uint64_t output_size) {
    try {
        last_error.clear();
        if (output == nullptr || output_size != nanosm::tma_descriptor_bytes) {
            throw nanosm::SimError("tma descriptor output must contain exactly 64 bytes");
        }
        if (element_bytes > UINT8_MAX) throw nanosm::SimError("tma element size is out of range");
        if (tile_length > UINT16_MAX || tile_rows > UINT16_MAX) {
            throw nanosm::SimError("tma tile length and rows must fit 16 bits");
        }
        if (row_dimension > UINT8_MAX) throw nanosm::SimError("tma row dimension must be 1, 2 or 3");
        if (layout > NANOSM_TMA_LAYOUT_SWIZZLE64) {
            throw nanosm::SimError("tma layout must be 0 (dense) or 1 (swizzle64)");
        }
        if (transposed > 1) throw nanosm::SimError("tma transposed must be zero or one");
        nanosm::TmaDescriptor descriptor;
        descriptor.base = base;
        descriptor.element_bytes = static_cast<uint8_t>(element_bytes);
        descriptor.dimensions = {d0, d1, d2, d3};
        descriptor.strides = {s1, s2, s3};
        descriptor.tile_length = static_cast<uint16_t>(tile_length);
        descriptor.tile_rows = static_cast<uint16_t>(tile_rows);
        descriptor.row_dimension = static_cast<uint8_t>(row_dimension);
        descriptor.layout = static_cast<nanosm::TileLayout>(layout);
        descriptor.transposed = transposed != 0;
        const auto bytes = nanosm::encode_tma_descriptor(descriptor);
        std::memcpy(output, bytes.data(), bytes.size());
        return NANOSM_OK;
    } catch (const std::exception& error) {
        fail(error.what());
        return NANOSM_ERROR;
    }
}

uint32_t nanosm_session_create_qtx(
    const char* profile_json,
    uint64_t profile_size,
    const char* qtx_source,
    uint64_t qtx_size,
    const char* launch_state_json,
    uint64_t launch_state_size,
    uint32_t trace_flags,
    uint64_t max_trace_events) {
    try {
        last_error.clear();
        auto profile = nanosm::parse_device_profile(input(profile_json, profile_size, "profile"));
        auto kernel = nanosm::compile_and_assemble(
            input(qtx_source, qtx_size, "qtx source"), profile);
        auto launch = nanosm::load_launch_state(
            input(launch_state_json, launch_state_size, "launch state"));
        auto value = std::make_unique<Session>();
        value->launch_image = nanosm::emit_launch_image({kernel, launch});
        value->machine = std::make_unique<nanosm::Machine>(
            profile, std::move(kernel), std::move(launch),
            trace_options(trace_flags, max_trace_events));
        apply_test_environment(*value->machine);
        return insert(std::move(value));
    } catch (const std::exception& error) {
        fail(error.what());
        return NANOSM_INVALID_HANDLE;
    }
}

uint64_t nanosm_qtx_kernel_info(
    const char* profile_json,
    uint64_t profile_size,
    const char* qtx_source,
    uint64_t qtx_size,
    char* output,
    uint64_t capacity) {
    try {
        last_error.clear();
        const auto profile = nanosm::parse_device_profile(input(profile_json, profile_size, "profile"));
        const auto kernel = nanosm::compile_and_assemble(input(qtx_source, qtx_size, "qtx source"), profile);
        return copy_string(nanosm::emit_kernel_info(kernel), output, capacity);
    } catch (const std::exception& error) {
        fail(error.what());
        return NANOSM_SIZE_ERROR;
    }
}

uint32_t nanosm_session_create_image(
    const char* profile_json,
    uint64_t profile_size,
    const char* launch_image_json,
    uint64_t launch_image_size,
    uint32_t trace_flags,
    uint64_t max_trace_events) {
    try {
        last_error.clear();
        auto profile = nanosm::parse_device_profile(input(profile_json, profile_size, "profile"));
        auto image = nanosm::load_launch_image(
            input(launch_image_json, launch_image_size, "launch image"));
        auto value = std::make_unique<Session>();
        value->launch_image = nanosm::emit_launch_image(image);
        value->machine = std::make_unique<nanosm::Machine>(
            profile, std::move(image.kernel), std::move(image.launch),
            trace_options(trace_flags, max_trace_events));
        apply_test_environment(*value->machine);
        return insert(std::move(value));
    } catch (const std::exception& error) {
        fail(error.what());
        return NANOSM_INVALID_HANDLE;
    }
}

int32_t nanosm_session_run(uint32_t handle, uint64_t cycle_limit) {
    try {
        last_error.clear();
        auto& value = session(handle);
        if (value.ran) throw nanosm::SimError("nanosm session has already run");
        if (cycle_limit == 0) throw nanosm::SimError("cycle limit is zero");
        value.machine->run(cycle_limit);
        value.ran = true;
        return NANOSM_OK;
    } catch (const std::exception& error) {
        fail(error.what());
        return NANOSM_ERROR;
    }
}

int32_t nanosm_session_destroy(uint32_t handle) {
    last_error.clear();
    if (handle == NANOSM_INVALID_HANDLE || sessions.erase(handle) != 1) {
        fail("invalid nanosm session handle");
        return NANOSM_ERROR;
    }
    return NANOSM_OK;
}

uint64_t nanosm_session_cycle(uint32_t handle) {
    try {
        last_error.clear();
        return session(handle).machine->cycle();
    } catch (const std::exception& error) {
        fail(error.what());
        return NANOSM_SIZE_ERROR;
    }
}

int32_t nanosm_session_read_gmem(
    uint32_t handle, uint64_t address, uint8_t* output, uint64_t size) {
    return read_memory(handle, address, output, size, true);
}

int32_t nanosm_session_read_shmem(
    uint32_t handle, uint64_t address, uint8_t* output, uint64_t size) {
    return read_memory(handle, address, output, size, false);
}

uint64_t nanosm_session_summary(uint32_t handle, char* output, uint64_t capacity) {
    try {
        last_error.clear();
        return copy_string(session(handle).machine->trace().summary_text(), output, capacity);
    } catch (const std::exception& error) {
        fail(error.what());
        return NANOSM_SIZE_ERROR;
    }
}

uint64_t nanosm_session_trace_json(uint32_t handle, char* output, uint64_t capacity) {
    try {
        last_error.clear();
        return copy_string(session(handle).machine->trace().perfetto_json(), output, capacity);
    } catch (const std::exception& error) {
        fail(error.what());
        return NANOSM_SIZE_ERROR;
    }
}

uint64_t nanosm_session_launch_image(uint32_t handle, char* output, uint64_t capacity) {
    try {
        last_error.clear();
        return copy_string(session(handle).launch_image, output, capacity);
    } catch (const std::exception& error) {
        fail(error.what());
        return NANOSM_SIZE_ERROR;
    }
}

uint64_t nanosm_last_error(char* output, uint64_t capacity) {
    return copy_string(last_error, output, capacity);
}

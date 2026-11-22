#pragma once

#include <stdint.h>

#if defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#define NANOSM_CAPI EMSCRIPTEN_KEEPALIVE
#elif defined(_WIN32) && defined(NANOSM_CAPI_BUILD)
#define NANOSM_CAPI __declspec(dllexport)
#elif defined(__GNUC__) && defined(NANOSM_CAPI_BUILD)
#define NANOSM_CAPI __attribute__((visibility("default")))
#else
#define NANOSM_CAPI
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum {
    NANOSM_OK = 0,
    NANOSM_ERROR = 1,
};

enum {
    NANOSM_TRACE_TIMELINE = 1u << 0,
    NANOSM_TRACE_DEBUG = 1u << 1,
};

#define NANOSM_INVALID_HANDLE UINT32_C(0)
#define NANOSM_SIZE_ERROR UINT64_MAX

// Encode an initialized mbar without duplicating its implementation-defined
// bit layout in language bindings or launch-image authors. tx_bytes is the
// signed transaction count, -(2^20 - 1)..2^20 - 1 (negative: complete_tx ran
// ahead of expect_tx).
NANOSM_CAPI int32_t nanosm_mbar_encode(
    uint32_t phase,
    uint32_t expected,
    uint32_t remaining,
    int32_t tx_bytes,
    uint64_t* output);

#define NANOSM_TMA_DESCRIPTOR_BYTES UINT32_C(64)
#define NANOSM_TMA_LAYOUT_DENSE UINT32_C(0)
#define NANOSM_TMA_LAYOUT_SWIZZLE64 UINT32_C(1)
// The descriptor (src/tma_descriptor.hpp): the tensor -- its base, element size,
// dimensions d0..d3 and strides s1..s3 (elements; dimension 0 is the
// contiguous one) -- and the tile: `tile_rows` runs of `tile_length`
// elements along dimension 0, the rows stepping along `row_dimension` (1..3),
// its shared-memory layout (src/layout.hpp: dense or swizzle64, the names of
// the profile's mma.layout) and, for swizzle64, whether it is transposed (0
// or 1).
// The direction, buffer, barrier and coordinates are the instruction's.
NANOSM_CAPI int32_t nanosm_tma_descriptor_encode(
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
    uint64_t output_size);

// All input text is a pointer plus byte count; it need not be NUL terminated.
// Handles are process-local integers rather than C++ pointers so the same ABI maps
// directly onto an Emscripten linear-memory call boundary.
NANOSM_CAPI uint32_t nanosm_session_create_qtx(
    const char* profile_json,
    uint64_t profile_size,
    const char* qtx_source,
    uint64_t qtx_size,
    const char* launch_state_json,
    uint64_t launch_state_size,
    uint32_t trace_flags,
    uint64_t max_trace_events);

// What a qtx kernel declares, as JSON (src/image.hpp emit_kernel_info): how
// many warps it is written for, its signature (the `.param` lines: name,
// type, shared-memory address, size) and its symbols (.equ constants and
// parameter addresses). A host needs only the signature to launch; the
// symbols are for a debugger or a test that looks inside shared memory.
// Follows the string getters' size convention below.
NANOSM_CAPI uint64_t nanosm_qtx_kernel_info(
    const char* profile_json,
    uint64_t profile_size,
    const char* qtx_source,
    uint64_t qtx_size,
    char* output,
    uint64_t capacity);

// Test seam: if the environment variable NANOSM_DRIVER_SHUFFLE_SEED is set
// when a session is created, the machine runs its register drivers in a
// random order every cycle (seeded). Results must be byte-identical; the
// ctest target nanosm_driver_shuffle_identical checks every launch.
NANOSM_CAPI uint32_t nanosm_session_create_image(
    const char* profile_json,
    uint64_t profile_size,
    const char* launch_image_json,
    uint64_t launch_image_size,
    uint32_t trace_flags,
    uint64_t max_trace_events);

NANOSM_CAPI int32_t nanosm_session_run(uint32_t handle, uint64_t cycle_limit);
NANOSM_CAPI int32_t nanosm_session_destroy(uint32_t handle);
NANOSM_CAPI uint64_t nanosm_session_cycle(uint32_t handle);

NANOSM_CAPI int32_t nanosm_session_read_gmem(
    uint32_t handle, uint64_t address, uint8_t* output, uint64_t size);
NANOSM_CAPI int32_t nanosm_session_read_shmem(
    uint32_t handle, uint64_t address, uint8_t* output, uint64_t size);

// String getters return the required byte count excluding the NUL terminator.
// Pass output=NULL/capacity=0 to query it. If capacity is greater than the returned
// count, the complete NUL-terminated string is copied. NANOSM_SIZE_ERROR means failure.
NANOSM_CAPI uint64_t nanosm_session_summary(
    uint32_t handle, char* output, uint64_t capacity);
NANOSM_CAPI uint64_t nanosm_session_trace_json(
    uint32_t handle, char* output, uint64_t capacity);
NANOSM_CAPI uint64_t nanosm_session_launch_image(
    uint32_t handle, char* output, uint64_t capacity);

// Last error is thread-local and follows the same size-query convention, but never fails.
NANOSM_CAPI uint64_t nanosm_last_error(char* output, uint64_t capacity);

#ifdef __cplusplus
}
#endif

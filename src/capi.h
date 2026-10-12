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
#define NANOSM_TMA_SWIZZLE_LINEAR UINT32_C(0)
#define NANOSM_TMA_SWIZZLE_XOR64_A UINT32_C(1)
#define NANOSM_TMA_SWIZZLE_XOR64_B UINT32_C(2)
NANOSM_CAPI int32_t nanosm_tma_descriptor_encode(
    uint32_t rank,
    uint32_t direction,
    uint32_t transpose,
    uint64_t gmem_base,
    uint32_t dim_x,
    uint32_t dim_y,
    uint32_t dim_z,
    uint32_t row_stride,
    uint32_t plane_stride,
    uint32_t box_x,
    uint32_t box_y,
    uint32_t box_z,
    uint32_t element_bytes,
    uint32_t swizzle,
    uint32_t shmem_base,
    uint32_t completion_mbar,
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

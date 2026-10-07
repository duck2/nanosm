#pragma once

#include <cstdint>
#include <span>

namespace nanosm {

// One model, not a runtime hardware mode. Included in the device fingerprint.
inline constexpr char mma_numeric_model[] = "fp16_raw_window25_rz_v1";

// Raw FP16 products plus raw FP32 C, fused over these spans (one core_k step).
// SF Tensor's tcgen05 unscaled FP16 arithmetic, adapted to our fusion extent.
// Not a claim of bit-equivalence to a larger B200 instruction.
uint32_t mma_dot_f16(std::span<const uint16_t> a, std::span<const uint16_t> b, uint32_t c);

} // namespace nanosm

"""Golden pipeline model for fpu_DSP48E1 - Clean separation of MUL and ADD paths."""

from math import isnan
import random
import math
import sys
from typing import Dict, Tuple

import numpy as np

QUIET_NAN = 0x7FC00000
INF = 0x7F800000

def bits_to_f32(x: int) -> np.float32:
    return np.uint32(x).view(np.float32)

def f32_to_bits(x: float) -> int:
    return np.float32(x).view(np.uint32).item()

def is_subnormal(bits: int) -> bool:
    exp = (bits >> 23) & 0xFF
    frac = bits & 0x7FFFFF
    return exp == 0 and frac != 0

def golden(op: str, in1_bits: int, in2_bits: int, in3_bits: int = 0) -> int:
    """Reference implementation using numpy float32"""
    f1 = bits_to_f32(in1_bits)
    f2 = bits_to_f32(in2_bits)
    f3 = bits_to_f32(in3_bits)

    # DAZ
    if is_subnormal(in1_bits):
        f1 = np.float32(0)
    if is_subnormal(in2_bits):
        f2 = np.float32(0)
    if is_subnormal(in3_bits):
        f3 = np.float32(0)

    if op == "mul":
        result = np.multiply(f1, f2, dtype=np.float32)
    elif op == "add":
        result = np.add(f1, f2, dtype=np.float32)
    elif op == "sub":
        result = np.subtract(f1, f2, dtype=np.float32)
    elif op == "fma":
        # Two rounds here is too inaccurate. To emulate true FMA
        # behavior, cast to float64, do math and then cast back
        f1, f2, f3 = np.float64([f1, f2, f3])
        result64 = np.add(np.multiply(f1, f2), f3)
        result = np.float32(result64)
    else:
        raise ValueError(f"unknown op: {op}")

    # FTZ
    if is_subnormal(f32_to_bits(result)):
        return 0

    # Canonical NaN
    if np.isnan(result):
        return QUIET_NAN

    return f32_to_bits(result)


def simulate_fmul(in1_bits: int, in2_bits: int) -> Tuple[Dict, int]:
    """Simulate FP multiply: result = in1 * in2."""
    print("\n=== FMUL PATH ===")

    frac1 = in1_bits & 0x7FFFFF
    frac2 = in2_bits & 0x7FFFFF
    exp1 = (in1_bits >> 23) & 0xFF
    exp2 = (in2_bits >> 23) & 0xFF
    sign1 = (in1_bits >> 31) & 1
    sign2 = (in2_bits >> 31) & 1
    sign = sign1 ^ sign2

    # DAZ: Denormals-As-Zero
    mant1 = 0 if exp1 == 0 else (1 << 23) | frac1
    mant2 = 0 if exp2 == 0 else (1 << 23) | frac2

    # NaN / Inf handling
    is_nan1 = exp1 == 0xFF and frac1 != 0
    is_nan2 = exp2 == 0xFF and frac2 != 0
    is_inf1 = exp1 == 0xFF and frac1 == 0
    is_inf2 = exp2 == 0xFF and frac2 == 0
    is_zero1 = exp1 == 0 and mant1 == 0
    is_zero2 = exp2 == 0 and mant2 == 0

    if is_nan1 or is_nan2:  # Propagate NaN
        return QUIET_NAN
    if (is_inf1 and is_zero2) or (is_inf2 and is_zero1):
        return QUIET_NAN  # 0 x Inf
    if is_inf1 or is_inf2:
        return (sign << 31) | INF
    if is_zero1 or is_zero2:
        return sign << 31
    
    print(f"Inputs: {bits_to_f32(in1_bits)} * {bits_to_f32(in2_bits)}")
    print(f"  in1_bits=0x{in1_bits:06X} in2_bits=0x{in2_bits:06X}")
    print(f"  frac1=0x{frac1:06X} mant1=0x{mant1:06X}, exp1={exp1}, sign1={sign1}")
    print(f"  frac2=0x{frac2:06X} mant2=0x{mant2:06X}, exp2={exp2}, sign2={sign2}")
    
    # Handle zero inputs (DAZ means exp=0 -> mant=0)
    if mant1 == 0 or mant2 == 0:
        return (sign1 ^ sign2) << 31

    product48 = mant1 * mant2
    print(f"  product_48=0x{product48:012X} (48-bit)")

    carry = (product48 >> 47) & 1
    if carry:
        # mantissa window = product[46:24] (23 bits)
        mant_pre = (product48 >> 24) & 0x7FFFFF
        guard = (product48 >> 23) & 1
        round = (product48 >> 22) & 1
        sticky = (product48 & ((1 << 22) - 1)) != 0   # bits [21:0]
        exp_result = exp1 + exp2 - 127 + 1
    else:
        # mantissa window = product[45:23] (23 bits)
        mant_pre = (product48 >> 23) & 0x7FFFFF
        guard = (product48 >> 22) & 1
        round = (product48 >> 21) & 1
        sticky = (product48 & ((1 << 21) - 1)) != 0   # bits [20:0]
        exp_result = exp1 + exp2 - 127

    lsb = mant_pre & 1

    print(f"  mant_pre=0x{mant_pre:06X}, G={guard}, R={round}, S={sticky}, LSB={lsb}")
    
    inc = guard and (round or sticky or lsb)
    mant_final = mant_pre + inc
    
    if mant_final & (1 << 23):
        mant_final = 0
        exp_result += 1

    print(f"  exp_result={exp_result}, mant_final=0x{mant_final:06X}, sign={sign}")

    if exp_result >= 255:
        out = (sign << 31) | INF
    elif exp_result <= 0:
        out = sign << 31
    else:
        out = (sign << 31) | (exp_result << 23) | mant_final

    print(f"  result=0x{out:08X} ({bits_to_f32(out)})")

    return out


def simulate_fadd(in1_bits: int, in2_bits: int, is_sub: bool) -> int:
    """Simulate FP add/sub: result = in1 +/- in2."""
    print("\n=== FADD/FSUB PATH ===")
    
    frac1 = in1_bits & 0x7FFFFF
    frac2 = in2_bits & 0x7FFFFF
    exp1 = (in1_bits >> 23) & 0xFF
    exp2 = (in2_bits >> 23) & 0xFF
    sign1 = (in1_bits >> 31) & 1
    sign2_raw = (in2_bits >> 31) & 1
    sign2 = sign2_raw ^ (1 if is_sub else 0)

    # NaN / Inf handling
    is_nan1 = exp1 == 0xFF and frac1 != 0
    is_inf1 = exp1 == 0xFF and frac1 == 0
    is_nan2 = exp2 == 0xFF and frac2 != 0
    is_inf2 = exp2 == 0xFF and frac2 == 0

    if is_nan1 or is_nan2: # Propagate NaN
        return QUIET_NAN
    if is_inf1 and is_inf2:
        if sign1 == sign2:  # Inf - Inf
            return QUIET_NAN
        return (sign1 << 31) | INF
    if is_inf1:  # Inf + finite
        return (sign1 << 31) | INF
    if is_inf2:  # Inf + finite
        return (sign2 << 31) | INF

    # DAZ
    mant1 = 0 if exp1 == 0 else (1 << 23) | frac1
    mant2 = 0 if exp2 == 0 else (1 << 23) | frac2

    print(f"Inputs: {bits_to_f32(in1_bits)} {'−' if is_sub else '+'} {bits_to_f32(in2_bits)}")
    print(f"  in1_bits=0x{in1_bits:06X} in2_bits=0x{in2_bits:06X}")
    print(f"  frac1=0x{frac1:06X} mant1=0x{mant1:06X}, exp1={exp1}, sign1={sign1}")
    print(f"  frac2=0x{frac2:06X} mant2=0x{mant2:06X}, exp2={exp2}, sign2={sign2}")

    # Decide HI/LO by exponent only (no mant compare here)
    if exp1 >= exp2:
        exp_hi, mant_hi, sign_hi = exp1, mant1, sign1
        exp_lo, mant_lo, sign_lo = exp2, mant2, sign2
    else:
        exp_hi, mant_hi, sign_hi = exp2, mant2, sign2
        exp_lo, mant_lo, sign_lo = exp1, mant1, sign1

    exp_diff = exp_hi - exp_lo
    print(f"  exp_diff={exp_diff}, shift_amt={min(exp_diff, 31)}")

    # Align Lo. Keep 2 of the shifted-out bits (G and R) and one sticky bit (S)
    mant_lo27 = mant_lo << 3
    mant_lo27_aligned = mant_lo27 >> exp_diff
    if exp_diff < 3:
        sticky = 0
    else:
        lower_mask = (1 << (exp_diff - 2)) - 1
        sticky = (mant_lo & lower_mask) != 0
    mant_lo27_aligned |= sticky

    # actually, shift up mantissa by a bit to include guard bit in the arithmetic
    mant_hi27 = mant_hi << 3

    print(f"  mant_hi27=0x{mant_hi27:06X}, mant_lo27=0x{mant_lo27:06X}, mant_lo27_aligned=0x{mant_lo27_aligned:06X}, sticky={int(sticky)}")

    signs_differ = sign_hi != sign_lo

    if signs_differ:
        mant_sum27 = mant_hi27 - mant_lo27_aligned
        if mant_sum27 < 0:  # This happens when exp_diff == 0 and lo was actually larger than hi
            mant_sum27 = -mant_sum27
            result_sign = sign_lo
        else:
            result_sign = sign_hi
    else:
        mant_sum27 = mant_hi27 + mant_lo27_aligned
        result_sign = sign_hi

    print(f"  signs_differ={signs_differ}, mant_sum26=0x{mant_sum27:06X}")
    
    if mant_sum27 == 0:
        return 0

    # LZC "module"
    lzc = 0
    for i in range(27):
        if mant_sum27 & (1 << (26 - i)):
            lzc = i
            break

    # Mantissa sum has 28th bit set -> there's carry -> shift right and increment exp
    if mant_sum27 >> 27:
        print("carry case")
        normalized = mant_sum27 >> 1
        exp_result = exp_hi + 1
        guard = (normalized >> 2) & 1
        round = (normalized >> 1) & 1
        sticky = (normalized & 1) | (mant_sum27 & 1)
    else:  # No carry. Shift mantissa sum until hidden 1 lands on bit 24
        print("no-carry case")
        normalized = mant_sum27 << lzc
        exp_result = exp_hi - lzc
        guard = (normalized >> 2) & 1
        round = (normalized >> 1) & 1
        sticky = normalized & 1

    print(f"  normalized=0x{normalized:06X}")

    mant_pre = (normalized >> 3) & 0x7FFFFF
    lsb = mant_pre & 1

    print(f"  mant_pre=0x{mant_pre:06X}, G={guard}, R={round}, S={sticky}, LSB={lsb}")
    
    inc = guard and (round or sticky or lsb)
    mant_final = mant_pre + inc

    print(f"  exp_result={exp_result}, mant_final=0x{mant_final:06X}, sign={result_sign}")

    if exp_result >= 255:
        out = (result_sign << 31) | 0x7F800000
    elif exp_result <= 0:
        out = result_sign << 31
    else:
        out = (result_sign << 31) | (exp_result << 23) | mant_final
    
    print(f"  result=0x{out:08X} ({bits_to_f32(out)})")
    
    return out


def simulate_fma_core(in1_bits: int, in2_bits: int, in3_bits: int, verbose=False) -> int:
    """Core FMA implementation: (in1 * in2) + in3 using 27-bit cheap FMA."""
    if verbose:
        print("\n=== FMA CORE (27-bit) ===")

    frac1 = in1_bits & 0x7FFFFF
    frac2 = in2_bits & 0x7FFFFF
    frac3 = in3_bits & 0x7FFFFF
    exp1 = (in1_bits >> 23) & 0xFF
    exp2 = (in2_bits >> 23) & 0xFF
    exp3 = (in3_bits >> 23) & 0xFF
    sign1 = (in1_bits >> 31) & 1
    sign2 = (in2_bits >> 31) & 1
    sign3 = (in3_bits >> 31) & 1
    
    # DAZ
    mant1 = 0 if exp1 == 0 else (1 << 23) | frac1
    mant2 = 0 if exp2 == 0 else (1 << 23) | frac2
    mant3 = 0 if exp3 == 0 else (1 << 23) | frac3
    
    # NaN / Inf handling
    is_nan1 = exp1 == 0xFF and frac1 != 0
    is_nan2 = exp2 == 0xFF and frac2 != 0
    is_nan3 = exp3 == 0xFF and frac3 != 0
    is_inf1 = exp1 == 0xFF and frac1 == 0
    is_inf2 = exp2 == 0xFF and frac2 == 0
    is_inf3 = exp3 == 0xFF and frac3 == 0
    is_zero1 = exp1 == 0 and mant1 == 0
    is_zero2 = exp2 == 0 and mant2 == 0
    is_zero3 = exp3 == 0 and mant3 == 0
    
    if is_nan1 or is_nan2 or is_nan3:
        return QUIET_NAN
    if (is_inf1 and is_zero2) or (is_inf2 and is_zero1):
        return QUIET_NAN
    
    sign_mul = sign1 ^ sign2
    
    if is_inf1 or is_inf2:
        mul_inf = (sign_mul << 31) | INF
        if is_inf3:
            if sign_mul == sign3:
                return mul_inf
            else:
                return QUIET_NAN
        return mul_inf
    
    if is_inf3:
        return (sign3 << 31) | INF

    # Compute product (48-bit mantissa product) - no special casing
    product48 = mant1 * mant2

    if verbose:
        print(f"Inputs: ({bits_to_f32(in1_bits)} * {bits_to_f32(in2_bits)}) + {bits_to_f32(in3_bits)}")
        print(f"  in1=0x{in1_bits:08X} in2=0x{in2_bits:08X} in3=0x{in3_bits:08X}")
        print(f"  product48=0x{product48:012X}")
    
    # Reduce to 27 bits (24 mantissa + 3 GRS bits)
    carry = (product48 >> 47) & 1
    if carry:
        product27 = (product48 >> 21) & 0x7FFFFFF
        sticky_mul = (product48 & ((1 << 21) - 1)) != 0
        product27 |= (1 if sticky_mul else 0)
        product_exp = exp1 + exp2 - 127 + 1
    else:
        product27 = (product48 >> 20) & 0x7FFFFFF
        sticky_mul = (product48 & ((1 << 20) - 1)) != 0
        product27 |= (1 if sticky_mul else 0)
        product_exp = exp1 + exp2 - 127

    # Kill product_exp if product48 = 0 (means mult result was killed bc of DAZ.)
    if product48 == 0:
        product_exp = 0

    # Align and add with mant3 (27-bit arithmetic)
    exp_diff = product_exp - exp3
    mant3_27 = mant3 << 3

    if verbose:
        print(f"  product27=0x{product27:07X}, product_exp={product_exp}")
        print(f"  exp_diff={exp_diff}")

    # Alignment
    if exp_diff >= 0:
        mant3_27_aligned = mant3_27 >> exp_diff
        lower_mask = (1 << exp_diff) - 1
        sticky_add = (mant3_27 & lower_mask) != 0
        mant3_27_aligned |= (1 if sticky_add else 0)
        product27_aligned = product27
        aligned_exp = product_exp
    else:
        product27_aligned = product27 >> (-exp_diff)
        lower_mask = (1 << (-exp_diff)) - 1
        sticky_add = (product27 & lower_mask) != 0
        product27_aligned |= (1 if sticky_add else 0)
        mant3_27_aligned = mant3_27
        aligned_exp = exp3

    if verbose:
        print(f"  product27_aligned=0x{product27_aligned:07X}, mant3_27_aligned=0x{mant3_27_aligned:07X}")
    
    # Add or subtract based on signs
    signs_differ = sign_mul != sign3
    
    if signs_differ:
        if product27_aligned >= mant3_27_aligned:
            mant_sum27 = product27_aligned - mant3_27_aligned
            result_sign = sign_mul
        else:
            mant_sum27 = mant3_27_aligned - product27_aligned
            result_sign = sign3
    else:
        mant_sum27 = product27_aligned + mant3_27_aligned
        result_sign = sign_mul
    
    if verbose:
        print(f"  mant_sum27=0x{mant_sum27:07X}")
    
    if mant_sum27 == 0:
        return 0
    
    # Normalize
    lzc = 0
    for i in range(27):
        if mant_sum27 & (1 << (26 - i)):
            lzc = i
            break
    
    # Check for carry (bit 27 set)
    if mant_sum27 >> 27:
        normalized = mant_sum27 >> 1
        exp_result = aligned_exp + 1
        guard = (normalized >> 2) & 1
        round_bit = (normalized >> 1) & 1
        sticky = (normalized & 1) | (mant_sum27 & 1)
    else:
        normalized = mant_sum27 << lzc
        exp_result = aligned_exp - lzc
        guard = (normalized >> 2) & 1
        round_bit = (normalized >> 1) & 1
        sticky = normalized & 1

    mant_pre = (normalized >> 3) & 0x7FFFFF
    lsb = mant_pre & 1
    
    if verbose:
        print(f"  lzc={lzc} carry={mant_sum27 >> 27} normalized=0x{normalized:07X}")
        print(f"  mant_pre=0x{mant_pre:06X}, G={guard}, R={round_bit}, S={sticky}, LSB={lsb}")
    
    # Round to nearest, ties to even
    inc = guard and (round_bit or sticky or lsb)
    mant_final = mant_pre + inc

    if mant_final & (1 << 23):
        mant_final = 0
        exp_result += 1

    if exp_result >= 255:
        out = (result_sign << 31) | INF
    elif exp_result <= 0:
        out = result_sign << 31
    else:
        out = (result_sign << 31) | (exp_result << 23) | mant_final
    
    if verbose:
        print(f"  exp_result={exp_result}, mant_final=0x{mant_final:06X}, sign={result_sign}")
        print(f"  result=0x{out:08X} ({bits_to_f32(out)})")

    return out


def simulate_fma_unified(op: str, in1_bits: int, in2_bits: int, in3_bits: int = 0) -> int:
    """Unified FMA-based implementation: all ops degeneracy through FMA core."""
    if op == "mul":
        return simulate_fma_core(in1_bits, in2_bits, 0)
    elif op == "add":
        return simulate_fma_core(f32_to_bits(1.0), in1_bits, in2_bits)
    elif op == "sub":
        return simulate_fma_core(f32_to_bits(1.0), in1_bits, in2_bits ^ 0x80000000)
    elif op == "fma":
        return simulate_fma_core(in1_bits, in2_bits, in3_bits)
    else:
        raise ValueError(f"unknown op: {op}")


def simulate_fpu(op: str, in1_bits: int, in2_bits: int, in3_bits: int = 0) -> int:
    """Dispatch to MUL, ADD/SUB, or FMA path (original separate paths)."""
    if op == "mul":
        return simulate_fmul(in1_bits, in2_bits)
    elif op == "add":
        return simulate_fadd(in1_bits, in2_bits, is_sub=False)
    elif op == "sub":
        return simulate_fadd(in1_bits, in2_bits, is_sub=True)
    elif op == "fma":
        return simulate_fma_core(in1_bits, in2_bits, in3_bits)
    else:
        raise ValueError(f"unknown op: {op}")


SMOKE_VECTORS_ADD = [
    ("add_simple", "add", f32_to_bits(1.0), f32_to_bits(2.0)),
    ("add_equal", "add", f32_to_bits(1.5), f32_to_bits(1.5)),
    ("swap_mag1", "add", f32_to_bits(1.0), f32_to_bits(-1.5)),
    ("swap_mag2", "add", f32_to_bits(1.0), f32_to_bits(-0.5)),
    ("plus_zero", "add", f32_to_bits(1.0), f32_to_bits(-1.0)),
    ("no_round1", "add", 0x3F800000, 0x30800000),
    ("no_round2", "add", 0x3F800000, 0x33800000),
    ("round_up1", "add", 0x3F800001, 0x33800000),
    ("round_up2", "add", 0x3F800000, 0x34000000),
    ("round_up3", "add", 0x3F800000, 0x33A00000),
    ("exp_up3", "add", 0x3FC00000, 0x3FC00000),
    ("left_norm1", "sub", 0x3F800001, 0x3F800000),
    ("left_norm2", "sub", 0x3F800001, 0x3F7FFFFF),
    ("to_inf", "add", 0x7F7FFFFF, 0x7F7FFFFF),
    ("daz", "add", 0x3F800000, 0x00000001),
    ("daz2", "add", 0x00000001, 0x00000001),
    ("ftz", "sub", 0x00800001, 0x00800000),
]

SMOKE_VECTORS_MUL = [
    ("mul_simple", "mul", 0x3F800000, 0x40000000),  # 1.0 * 2.0
    ("mul_equal", "mul", 0x3FC00000, 0x3FC00000),  # 1.5 * 1.5 -> carry (>=2)
    ("mul_passthru_1", "mul", 0xBF800000, 0x3F800000),  # -1.0 * 1.0 -> sign passthrough
    ("mul_passthru_2", "mul", 0x3F800000, 0xBF000000),  # 1.0 * -0.5
    ("zero_pos", "mul", 0x00000000, 0x3F800000),  # +0 * 1.0 -> +0
    ("zero_neg", "mul", 0x80000000, 0x3F800000),  # -0 * 1.0 -> -0 (if preserving sign)
    ("zero_sign_xor1", "mul", 0x00000000, 0xBF800000),  # +0 * -1.0 -> -0
    ("zero_sign_xor2", "mul", 0x80000000, 0xBF800000),  # -0 * -1.0 -> +0
    ("inf_times_finite", "mul", 0x7F800000, 0x3F800000),  # +Inf * 1.0 -> +Inf
    ("neg_inf_times_neg", "mul", 0xFF800000, 0xBF800000),  # -Inf * -1.0 -> +Inf
    ("inf_times_zero", "mul", 0x7F800000, 0x00000000),  # Inf * 0 -> NaN
    ("nan_payload", "mul", 0x7FC12345, 0x3F800000),  # qNaN * 1.0 -> qNaN (payload)
    ("snan_quiet", "mul", 0x7FA00001, 0x40000000),  # sNaN * 2.0 -> qNaN
    ("overflow_max2", "mul", 0x7F7FFFFF, 0x40000000),  # max finite * 2.0 -> +Inf
    ("overflow_near", "mul", 0x7F7FFFFF, 0x3F7FFFFE),  # max finite * ~0.999... (big rounding check)
    ("overflow_sign", "mul", 0xFF7FFFFF, 0x40000000),  # -max * 2.0 -> -Inf
    ("underflow_minmin", "mul", 0x00800000, 0x00800000),  # min normal * min normal -> subnormal (FTZ -> 0)
    ("underflow_tiny", "mul", 0x00800000, 0x3F000000),  # min normal * 0.5 -> subnormal (FTZ path)
    ("subnorm_x_norm", "mul", 0x00000001, 0x3F800000),  # tiniest subnorm * 1.0 (DAZ zero-in)
    ("subnorm_x_large", "mul", 0x00000010, 0x7F7FFFFF),  # small subnorm * max finite (DAZ/FTZ mix)
    ("round_tie_up", "mul", 0x3F800001, 0x3F000000),  # (1+ulp) * 0.5 → mantissa tie region
    ("round_tie_even", "mul", 0x3F7FFFFF, 0x3F7FFFFF),  # just-below 1.0 * just-below 1.0
    ("round_guard_only", "mul", 0x3F800000, 0x33800000),  # 1.0 * 2^-24 (GRS boundary)
    ("round_sticky_only", "mul", 0x3F800000, 0x33000001),  # 1.0 * value with far tail set
    ("no_carry_region", "mul", 0x3F800000, 0x3F800000),  # 1.0 * 1.0 → no carry
    ("carry_region", "mul", 0x3FC00000, 0x3FC00000),  # 1.5 * 1.5 → carry, exponent +1
    ("carry_edge", "mul", 0x3F800000, 0x3F7FFFFF),  # 1.0 * (1-ulp) → product just under 1
    ("sign_mix1", "mul", 0xBF7FFFFF, 0x3F7FFFFF),  # (-1+ulp) * (1-ulp) -> negative
    ("sign_mix2", "mul", 0x3FC00000, 0xBFC00000),  # 1.5 * -1.5
    ("hi_exp_ok", "mul", 0x7F000000, 0x3F000000),  # big normal * 0.5 → stays normal
    ("lo_exp_ok", "mul", 0x01000000, 0x40000000),  # small normal * 2.0 → normalize up from edge
    ("zero_times_pos", "mul", 0x00000000, 0x7F7FFFFF),  # +0 * max finite
    ("negzero_times_neg", "mul", 0x80000000, 0xFF7FFFFF),  # -0 * -max finite -> +0 if preserving sign rules
]

SMOKE_VECTORS_FMA = [
    ("fma_simple", "fma", 0x3F800000, 0x40000000, 0x40400000),  # 1.0 * 2.0 + 3.0 = 5.0
    ("fma_zero_prod", "fma", 0x00000000, 0x40000000, 0x40400000),  # 0.0 * 2.0 + 3.0 = 3.0
    ("fma_zero_add", "fma", 0x3F800000, 0x40000000, 0x00000000),  # 1.0 * 2.0 + 0.0 = 2.0
    ("fma_cancel", "fma", 0x3F800000, 0x40000000, 0xC0000000),  # 1.0 * 2.0 + (-2.0) = 0.0
    ("fma_neg_prod", "fma", 0xBF800000, 0x40000000, 0x40400000),  # -1.0 * 2.0 + 3.0 = 1.0
    ("fma_neg_add", "fma", 0x3F800000, 0x40000000, 0xC0400000),  # 1.0 * 2.0 + (-3.0) = -1.0
    ("fma_all_neg", "fma", 0xBF800000, 0xC0000000, 0xC0400000),  # -1.0 * -2.0 + (-3.0) = -1.0
    ("fma_precision1", "fma", 0x3F800000, 0x3F800001, 0x33800000),  # Tests FMA precision advantage
    ("fma_precision2", "fma", 0x4B800000, 0x34000000, 0x3F800000),  # Large * small + normal
    ("fma_overflow", "fma", 0x7F000000, 0x40000000, 0x7F000000),  # Huge * 2 + huge -> Inf
    ("fma_underflow", "fma", 0x00800000, 0x3F000000, 0x33800000),  # Min normal * 0.5 + tiny
    ("fma_inf_prod", "fma", 0x7F800000, 0x3F800000, 0x40400000),  # Inf * 1.0 + 3.0 = Inf
    ("fma_inf_add", "fma", 0x3F800000, 0x40000000, 0x7F800000),  # 1.0 * 2.0 + Inf = Inf
    ("fma_inf_cancel", "fma", 0x7F800000, 0x3F800000, 0xFF800000),  # Inf * 1.0 + (-Inf) = NaN
    ("fma_nan_in1", "fma", 0x7FC00000, 0x3F800000, 0x40400000),  # NaN * 1.0 + 3.0 = NaN
    ("fma_nan_in2", "fma", 0x3F800000, 0x7FC00000, 0x40400000),  # 1.0 * NaN + 3.0 = NaN
    ("fma_nan_in3", "fma", 0x3F800000, 0x40000000, 0x7FC00000),  # 1.0 * 2.0 + NaN = NaN
    ("fma_inf_zero", "fma", 0x7F800000, 0x00000000, 0x40400000),  # Inf * 0 + 3.0 = NaN
    ("fma_daz", "fma", 0x00000001, 0x3F800000, 0x40400000),  # subnorm * 1.0 + 3.0 = 3.0 (DAZ)
    ("fma_ftz", "fma", 0x00800000, 0x3F000000, 0x00000000),  # min normal * 0.5 + 0 -> subnorm (FTZ)
    ("fma_round_up", "fma", 0x3F800000, 0x3F800001, 0x33800000),  # Rounding test
    ("fma_round_down", "fma", 0x3F800000, 0x3F7FFFFF, 0xB3800000),  # Rounding test
    ("fma_tiny_contrib", "fma", 0x4B800000, 0x3F800000, 0x33800000),  # Large product + tiny addend
    ("fma_equal_mag", "fma", 0x3FC00000, 0x3FC00000, 0x40100000),  # 1.5 * 1.5 + 2.25 = 4.5
    ("fma_weird", "fma", 0x6FA17735, 0x006ED6E3, 0x85197FF4),
    ("fma_weird2", "fma", 0x0DE051A6, 0x300568D2, 0x84B871BB),
    ("fma_weird3", "fma", 0xF0A9B97F, 0xAD968213, 0xDEC56DC3),
    ("fma_weird4", "fma", 0x9CA5B74F, 0x401FB5A6, 0x1D4E2772),
    ("fma_weird5", "fma", 0x6D39DA75, 0x2C3D17E2, 0x5A9C0679)
]

def ulp_distance(a: int, b: int) -> int:
    """Calculate ULP distance between two float32 bit patterns."""
    if a == b:
        return 0
    # Handle sign differences
    if (a >> 31) != (b >> 31):
        # Different signs - check if both are zero
        if (a & 0x7FFFFFFF) == 0 and (b & 0x7FFFFFFF) == 0:
            return 0
        return abs(int(a) - int(b))
    # Same sign - treat as signed integers
    return abs(int(a) - int(b))

def run_smoke_tests_unified(smoke_vectors) -> None:
    """Test unified FMA implementation with exact match for mul/add/sub, 1 ULP for fma."""
    passed = 0
    for test_case in smoke_vectors:
        if len(test_case) == 4:
            name, op, in1, in2 = test_case
            in3 = 0
        elif len(test_case) == 5:
            name, op, in1, in2, in3 = test_case
        else:
            raise ValueError(f"Invalid test case format: {test_case}")
        
        print(f"\n{'='*60}")
        print(f"Test: {name}")
        out_bits = simulate_fma_unified(op, in1, in2, in3)
        expected = golden(op, in1, in2, in3)
        
        # Normalize -0 to +0 for comparison
        if expected == 0x80000000:
            expected = 0x0
        if out_bits == 0x80000000:
            out_bits = 0x0
        
        if op == "fma":
            # Allow 1 ULP error for FMA
            ulp_dist = ulp_distance(out_bits, expected)
            if ulp_dist <= 1:
                print(f"✓ PASS (ULP distance: {ulp_dist})")
                passed += 1
            else:
                print(f"✗ FAIL: expected 0x{expected:08X} ({bits_to_f32(expected)}), got 0x{out_bits:08X} ({bits_to_f32(out_bits)}), ULP distance: {ulp_dist}")
                sys.exit(-1)
        else:
            # Exact match for mul/add/sub
            if out_bits == expected:
                print(f"✓ PASS (exact)")
                passed += 1
            else:
                print(f"✗ FAIL: expected 0x{expected:08X} ({bits_to_f32(expected)}), got 0x{out_bits:08X} ({bits_to_f32(out_bits)})")
                sys.exit(-1)
    
    print(f"\n{'='*60}")
    print(f"All {passed}/{len(smoke_vectors)} tests passed! ✓")

def run_smoke_tests(smoke_vectors) -> None:
    """Test original separate paths (for reference)."""
    passed = 0
    for test_case in smoke_vectors:
        if len(test_case) == 4:
            name, op, in1, in2 = test_case
            in3 = 0
        elif len(test_case) == 5:
            name, op, in1, in2, in3 = test_case
        else:
            raise ValueError(f"Invalid test case format: {test_case}")
        
        print(f"\n{'='*60}")
        print(f"Test: {name}")
        out_bits = simulate_fpu(op, in1, in2, in3)
        expected = golden(op, in1, in2, in3)
        if out_bits == expected:
            print(f"✓ PASS")
            passed += 1
        else:
            print(f"✗ FAIL: expected 0x{expected:08X} ({bits_to_f32(expected)}), got 0x{out_bits:08X} ({bits_to_f32(out_bits)})")
            sys.exit(-1)
    
    print(f"\n{'='*60}")
    print(f"All {passed}/{len(smoke_vectors)} tests passed! ✓")

def run_fuzz_tests(n: int, ops) -> None:
    rng = random.Random(42)
    for _ in range(n):
        op = rng.choice(ops)
        in1_bits = rng.getrandbits(32)
        in2_bits = rng.getrandbits(32)
        in3_bits = rng.getrandbits(32) if op == "fma" else 0
        out_bits = simulate_fpu(op, in1_bits, in2_bits, in3_bits)
        expected = golden(op, in1_bits, in2_bits, in3_bits)
        if expected == 0x80000000:
            expected = 0x0
        if out_bits == 0x80000000:
            out_bits = 0x0
        if out_bits == expected:
            print(f"✓ PASS")
        else:
            print(f"✗ FAIL: expected 0x{expected:08X} ({bits_to_f32(expected)}), got 0x{out_bits:08X} ({bits_to_f32(out_bits)})")
            sys.exit(-1)

if __name__ == "__main__":
    # Test original separate paths
    #run_smoke_tests(SMOKE_VECTORS_ADD)
    #run_smoke_tests(SMOKE_VECTORS_MUL)
    #run_fuzz_tests(10000, ["mul"])
    
    # Test unified FMA-based implementation
    print("\n" + "="*60)
    print("Testing MUL through FMA (expect exact match)")
    print("="*60)
    run_smoke_tests_unified(SMOKE_VECTORS_MUL)
    
    print("\n" + "="*60)
    print("Testing ADD/SUB through FMA (expect exact match)")
    print("="*60)
    run_smoke_tests_unified(SMOKE_VECTORS_ADD)
    
    print("\n" + "="*60)
    print("Testing FMA (allow 1 ULP error)")
    print("="*60)
    run_smoke_tests_unified(SMOKE_VECTORS_FMA)

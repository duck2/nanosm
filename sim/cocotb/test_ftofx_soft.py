"""Soft model for ftofx (float to fixed-point converter)."""

import struct
import sys
from enum import IntEnum

class Mode(IntEnum):
    CLAMP = 0
    REPEAT = 1
    MIRROR = 2
    BORDER = 3

def float_to_bits(f):
    return struct.unpack('<I', struct.pack('<f', f))[0]

def bits_to_float(b):
    return struct.unpack('<f', struct.pack('<I', b & 0xFFFFFFFF))[0]

def ftofx(f_bits: int, m: int, mode: Mode, verbose: bool = False) -> tuple:
    """
    Convert F32 to fixed-point. Output is 16 bits (sign + 15 magnitude).
    n + m = 15 always, where n = integer bits, m = fractional bits.
    
    Inputs:
        f_bits: 32-bit IEEE 754 float
        m: fractional bits (0-15), implies n = 15 - m integer bits
        mode: clamp / repeat / mirror / border
    
    Core operation (same for all modes):
        shifted = mant24 * 2^(exp - 150 + m)  (position mantissa for fixed-point)
    
    Mode-specific:
        overflow = (exp > n + 126) i.e. value doesn't fit in n integer bits
        clamp:  if overflow, saturate; else output shifted (signed 16-bit)
        repeat: output lower m bits of shifted; ignore overflow
        mirror: output lower m bits; predicate = shifted[m] (parity bit)
        border: output lower m bits; predicate = overflow
    
    Returns:
        (result, predicate) tuple
    """
    n = 15 - m  # integer bits (including the sign bit position for clamp)
    
    sign = (f_bits >> 31) & 1
    exp = (f_bits >> 23) & 0xFF
    frac = f_bits & 0x7FFFFF

    # DAZ: denormals as zero
    if exp == 0:
        mant24 = 0
    else:
        mant24 = (1 << 23) | frac

    if verbose:
        print(f"Input: 0x{f_bits:08X} = {bits_to_float(f_bits)}")
        print(f"  sign={sign}, exp={exp}, n={n}, m={m}, mant24=0x{mant24:06X}")

    # Core operation: compute shifted magnitude
    # shifted = mant24 * 2^(exp - 150 + m) = trunc(|float| * 2^m)
    power = exp - 150 + m

    if verbose:
        print(f"  power = {exp} - 150 + {m} = {power}")

    if mant24 == 0:
        shifted = 0
    elif power >= 0:
        shifted = mant24 << power
    else:
        shifted = mant24 >> (-power)

    if verbose:
        print(f"  shifted = 0x{shifted:X} ({shifted})")

    # Overflow: magnitude exceeds 15 bits (32767)
    overflow = shifted > 32767

    if verbose:
        print(f"  overflow = {overflow}")

    # Mode-specific output
    if mode == Mode.CLAMP:
        # Signed 16-bit output, max magnitude = 32767
        if overflow:
            magnitude = 32767
        else:
            magnitude = shifted

        # Apply sign (2's complement, 16-bit)
        if sign:
            result = (-magnitude) & 0xFFFF
        else:
            result = magnitude

        return (result, False)

    elif mode == Mode.REPEAT:
        # Output lower m bits of shifted magnitude (abs value behavior)
        frac_mask = (1 << m) - 1
        frac_part = shifted & frac_mask
        return (frac_part, False)

    elif mode == Mode.MIRROR:
        # Output lower m bits, predicate = parity (bit m of shifted)
        # Uses abs value behavior
        frac_mask = (1 << m) - 1
        frac_part = shifted & frac_mask
        parity = (shifted >> m) & 1
        return (frac_part, bool(parity))

    elif mode == Mode.BORDER:
        # Output lower m bits, predicate = out of [0, 2^n) range
        frac_mask = (1 << m) - 1
        frac_part = shifted & frac_mask

        # Out of bounds if: overflow OR negative (for unsigned texture coords)
        oob = overflow or (sign == 1 and mant24 != 0)

        return (frac_part, oob)

    else:
        raise ValueError(f"Unknown mode: {mode}")


# =============================================================================
# Test vectors - all use n + m = 15
# =============================================================================

SMOKE_VECTORS_CLAMP = [
    # (name, float_bits, m, expected_result)
    # m=14: 1 integer bit + 14 frac bits, range [0, 2) for positive
    ("zero", float_to_bits(0.0), 14, 0),
    ("neg_zero", float_to_bits(-0.0), 14, 0),
    ("one", float_to_bits(1.0), 14, 16384),             # 1.0 * 2^14 = 16384
    ("half", float_to_bits(0.5), 14, 8192),             # 0.5 * 2^14 = 8192
    ("quarter", float_to_bits(0.25), 14, 4096),         # 0.25 * 2^14 = 4096
    ("almost_two", float_to_bits(1.999), 14, 32751),    # 1.999 * 16384 ≈ 32751
    ("two_clamp", float_to_bits(2.0), 14, 32767),       # 2.0 * 16384 = 32768 > 32767, clamp
    ("neg_one", float_to_bits(-1.0), 14, 0xC000),       # -16384 in 2's comp
    ("neg_half", float_to_bits(-0.5), 14, 0xE000),      # -8192 in 2's comp
    ("neg_two_clamp", float_to_bits(-2.0), 14, 0x8001), # -32767 (clamped)
    # m=8: 7 integer bits + 8 frac bits, range [0, 128) for positive
    ("one_m8", float_to_bits(1.0), 8, 256),             # 1.0 * 256 = 256
    ("two_m8", float_to_bits(2.0), 8, 512),             # 2.0 * 256 = 512
    ("hundred_m8", float_to_bits(100.0), 8, 25600),     # 100.0 * 256 = 25600
    ("overflow_m8", float_to_bits(200.0), 8, 32767),    # 200 * 256 = 51200 > 32767, clamp
    ("neg_one_m8", float_to_bits(-1.0), 8, 0xFF00),     # -256 in 2's comp
    # Edge cases
    ("tiny", float_to_bits(0.0001), 14, 1),             # 0.0001 * 16384 ≈ 1.6 -> 1
    ("very_tiny", float_to_bits(1e-10), 14, 0),         # underflow to 0
]

SMOKE_VECTORS_REPEAT = [
    # (name, float_bits, m, expected_frac)
    # m=8: 256-texel texture
    ("zero", float_to_bits(0.0), 8, 0),
    ("half", float_to_bits(0.5), 8, 128),               # 0.5 * 256 = 128
    ("one", float_to_bits(1.0), 8, 0),                  # 1.0 * 256 = 256 & 0xFF = 0
    ("one_point_five", float_to_bits(1.5), 8, 128),     # 1.5 * 256 = 384 & 0xFF = 128
    ("two", float_to_bits(2.0), 8, 0),                  # 2.0 * 256 = 512 & 0xFF = 0
    ("two_point_25", float_to_bits(2.25), 8, 64),       # 2.25 * 256 = 576 & 0xFF = 64
    ("neg_half", float_to_bits(-0.5), 8, 128),          # trunc(-128) & 0xFF -> 256-128 = 128
    ("neg_one", float_to_bits(-1.0), 8, 0),             # trunc(-256) & 0xFF = 0
]

SMOKE_VECTORS_MIRROR = [
    # (name, float_bits, m, expected_frac, expected_parity)
    # m=8: parity = (shifted >> 8) & 1
    ("zero", float_to_bits(0.0), 8, 0, False),
    ("half", float_to_bits(0.5), 8, 128, False),        # 128, parity = 0
    ("one", float_to_bits(1.0), 8, 0, True),            # 256 >> 8 = 1, parity = 1
    ("one_point_five", float_to_bits(1.5), 8, 128, True), # 384 >> 8 = 1
    ("two", float_to_bits(2.0), 8, 0, False),           # 512 >> 8 = 2, parity = 0
    ("two_point_five", float_to_bits(2.5), 8, 128, False), # 640 >> 8 = 2
    ("three", float_to_bits(3.0), 8, 0, True),          # 768 >> 8 = 3, parity = 1
]

SMOKE_VECTORS_BORDER = [
    # (name, float_bits, m, expected_frac, expected_oob)
    # m=8: n=7, valid unsigned range [0, 128)
    ("zero", float_to_bits(0.0), 8, 0, False),
    ("half", float_to_bits(0.5), 8, 128, False),
    ("one", float_to_bits(1.0), 8, 0, False),
    ("hundred", float_to_bits(100.0), 8, 0, False),     # 100 < 128, in range
    ("overflow", float_to_bits(200.0), 8, 0, True),     # 200 * 256 > 32767
    ("neg_half", float_to_bits(-0.5), 8, 128, True),    # negative -> OOB
    ("neg_one", float_to_bits(-1.0), 8, 0, True),       # negative -> OOB
]


def run_smoke_tests_clamp():
    print("\n" + "=" * 60)
    print("CLAMP MODE TESTS")
    print("=" * 60)
    passed = 0
    for name, f_bits, m, expected in SMOKE_VECTORS_CLAMP:
        result, pred = ftofx(f_bits, m, Mode.CLAMP, verbose=False)
        if result == expected:
            print(f"✓ {name}: 0x{result:04X} ({result})")
            passed += 1
        else:
            print(f"✗ {name}: expected 0x{expected:04X}, got 0x{result:04X}")
            ftofx(f_bits, m, Mode.CLAMP, verbose=True)
            sys.exit(-1)
    print(f"\nAll {passed}/{len(SMOKE_VECTORS_CLAMP)} clamp tests passed!")


def run_smoke_tests_repeat():
    print("\n" + "=" * 60)
    print("REPEAT MODE TESTS")
    print("=" * 60)
    passed = 0
    for name, f_bits, m, expected in SMOKE_VECTORS_REPEAT:
        result, pred = ftofx(f_bits, m, Mode.REPEAT, verbose=False)
        if result == expected:
            print(f"✓ {name}: {result}")
            passed += 1
        else:
            print(f"✗ {name}: expected {expected}, got {result}")
            ftofx(f_bits, m, Mode.REPEAT, verbose=True)
            sys.exit(-1)
    print(f"\nAll {passed}/{len(SMOKE_VECTORS_REPEAT)} repeat tests passed!")


def run_smoke_tests_mirror():
    print("\n" + "=" * 60)
    print("MIRROR MODE TESTS")
    print("=" * 60)
    passed = 0
    for name, f_bits, m, expected_frac, expected_par in SMOKE_VECTORS_MIRROR:
        result, pred = ftofx(f_bits, m, Mode.MIRROR, verbose=False)
        if result == expected_frac and pred == expected_par:
            print(f"✓ {name}: frac={result}, parity={pred}")
            passed += 1
        else:
            print(f"✗ {name}: expected ({expected_frac}, {expected_par}), got ({result}, {pred})")
            ftofx(f_bits, m, Mode.MIRROR, verbose=True)
            sys.exit(-1)
    print(f"\nAll {passed}/{len(SMOKE_VECTORS_MIRROR)} mirror tests passed!")


def run_smoke_tests_border():
    print("\n" + "=" * 60)
    print("BORDER MODE TESTS")
    print("=" * 60)
    passed = 0
    for name, f_bits, m, expected_frac, expected_oob in SMOKE_VECTORS_BORDER:
        result, pred = ftofx(f_bits, m, Mode.BORDER, verbose=False)
        if result == expected_frac and pred == expected_oob:
            print(f"✓ {name}: frac={result}, oob={pred}")
            passed += 1
        else:
            print(f"✗ {name}: expected ({expected_frac}, {expected_oob}), got ({result}, {pred})")
            ftofx(f_bits, m, Mode.BORDER, verbose=True)
            sys.exit(-1)
    print(f"\nAll {passed}/{len(SMOKE_VECTORS_BORDER)} border tests passed!")


# =============================================================================
# Golden models
# =============================================================================

def golden_clamp(f: float, m: int) -> int:
    """Golden: int(f * 2^m) clamped to [-32767, 32767], as 16-bit 2's complement."""
    scaled = f * (1 << m)
    truncated = int(scaled) if scaled >= 0 else -int(-scaled)
    clamped = max(-32767, min(32767, truncated))
    return clamped & 0xFFFF

def golden_repeat(f: float, m: int) -> int:
    """Golden: abs frac part = int(|f| * 2^m) & mask."""
    scaled = abs(f) * (1 << m)
    return int(scaled) & ((1 << m) - 1)

def golden_mirror(f: float, m: int) -> tuple:
    """Golden: (frac, parity). frac = abs frac part, parity = floor(|f|) & 1."""
    scaled = abs(f) * (1 << m)
    frac = int(scaled) & ((1 << m) - 1)
    parity = (int(scaled) >> m) & 1
    return (frac, bool(parity))

def golden_border_pred(f: float, m: int) -> bool:
    """Golden: oob predicate. True if f < 0 or f >= 2^n (where n = 15 - m)."""
    n = 15 - m
    return f < 0 or f >= (1 << n)


def run_golden_tests():
    """Test ftofx against golden models with directed vectors."""
    print("\n" + "=" * 60)
    print("GOLDEN MODEL TESTS")
    print("=" * 60)
    
    test_values = [
        0.0, -0.0, 0.5, -0.5, 0.25, -0.25, 0.125,
        1.0, -1.0, 1.5, -1.5, 1.25, 1.75,
        2.0, -2.0, 2.5, 3.0, 3.5,
        0.001, 0.0001, 0.00001,
        100.0, -100.0, 127.0, 128.0, 200.0,
        0.999, 1.001, 0.333, 0.666,
    ]
    m_values = [8, 10, 12, 14]
    
    errors = 0
    
    # CLAMP tests
    print("\nClamp mode:")
    for m in m_values:
        for fv in test_values:
            f_bits = float_to_bits(fv)
            result, _ = ftofx(f_bits, m, Mode.CLAMP)
            expected = golden_clamp(fv, m)
            if result != expected:
                print(f"  ✗ clamp({fv}, m={m}): expected 0x{expected:04X}, got 0x{result:04X}")
                errors += 1
    if errors == 0:
        print(f"  ✓ All {len(test_values) * len(m_values)} clamp tests passed")
    
    # REPEAT tests
    print("\nRepeat mode:")
    repeat_errors = 0
    for m in m_values:
        for fv in test_values:
            f_bits = float_to_bits(fv)
            result, _ = ftofx(f_bits, m, Mode.REPEAT)
            expected = golden_repeat(fv, m)
            if result != expected:
                print(f"  ✗ repeat({fv}, m={m}): expected {expected}, got {result}")
                repeat_errors += 1
    if repeat_errors == 0:
        print(f"  ✓ All {len(test_values) * len(m_values)} repeat tests passed")
    errors += repeat_errors
    
    # MIRROR tests
    print("\nMirror mode:")
    mirror_errors = 0
    for m in m_values:
        for fv in test_values:
            f_bits = float_to_bits(fv)
            result, pred = ftofx(f_bits, m, Mode.MIRROR)
            exp_frac, exp_par = golden_mirror(fv, m)
            if result != exp_frac or pred != exp_par:
                print(f"  ✗ mirror({fv}, m={m}): expected ({exp_frac}, {exp_par}), got ({result}, {pred})")
                mirror_errors += 1
    if mirror_errors == 0:
        print(f"  ✓ All {len(test_values) * len(m_values)} mirror tests passed")
    errors += mirror_errors
    
    # BORDER tests (only check predicate)
    print("\nBorder mode (predicate only):")
    border_errors = 0
    for m in m_values:
        for fv in test_values:
            f_bits = float_to_bits(fv)
            _, pred = ftofx(f_bits, m, Mode.BORDER)
            exp_pred = golden_border_pred(fv, m)
            if pred != exp_pred:
                print(f"  ✗ border({fv}, m={m}): expected oob={exp_pred}, got oob={pred}")
                border_errors += 1
    if border_errors == 0:
        print(f"  ✓ All {len(test_values) * len(m_values)} border predicate tests passed")
    errors += border_errors
    
    print("\n" + "=" * 60)
    if errors == 0:
        print("All golden tests passed!")
    else:
        print(f"FAILED: {errors} errors")
        sys.exit(-1)


def interactive_test():
    """Interactive mode for exploring behavior."""
    print("\nInteractive ftofx test. Enter float values or 'q' to quit.")
    print("Format: <float> [m] [mode]  (defaults: m=14, mode=clamp)")
    while True:
        try:
            line = input("> ").strip()
            if line.lower() == 'q':
                break
            parts = line.split()
            if not parts:
                continue
            f = float(parts[0])
            m = int(parts[1]) if len(parts) > 1 else 14
            mode_str = parts[2] if len(parts) > 2 else "clamp"
            mode = Mode[mode_str.upper()]
            
            f_bits = float_to_bits(f)
            result, pred = ftofx(f_bits, m, mode, verbose=True)
            print(f"  -> result=0x{result:04X} ({result}), predicate={pred}")
        except (ValueError, KeyError) as e:
            print(f"Error: {e}")
        except EOFError:
            break


if __name__ == "__main__":
    run_smoke_tests_clamp()
    run_smoke_tests_repeat()
    run_smoke_tests_mirror()
    run_smoke_tests_border()
    run_golden_tests()
    
    if len(sys.argv) > 1 and sys.argv[1] == "-i":
        interactive_test()

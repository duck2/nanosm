"""cocotb tests for ftofx (float to fixed-point converter, mag mode only)."""

from math import isinf, isnan
import struct
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, ReadOnly

LAT = 3

def float_to_bits(f):
    """Convert Python float to 32-bit IEEE 754 representation."""
    return struct.unpack('<I', struct.pack('<f', f))[0]

def bits_to_float(b):
    """Convert 32-bit IEEE 754 representation to Python float."""
    return struct.unpack('<f', struct.pack('<I', b & 0xFFFFFFFF))[0]

def golden(f_bits, frac_bits):
    """Software model of ftofx (mag mode)."""
    f = bits_to_float(f_bits)
    scaled = f * (2 ** frac_bits)
    if isinf(scaled) or isnan(scaled):
        truncated = 0
    else:
        truncated = int(scaled) if scaled >= 0 else -int(-scaled)
    clamped = max(-32767, min(32767, truncated))
    return clamped & 0xFFFF

async def reset(dut, cycles=2):
    dut.rst.value = 1
    for _ in range(cycles):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

async def run_stream(dut, vectors):
    """Drive (f_bits, frac_bits) vectors, collect outputs. Returns (expected, observed)."""
    expected = [golden(f, fb) for (f, fb) in vectors]
    observed = []

    await RisingEdge(dut.clk)

    for i, (f_bits, frac_bits) in enumerate(vectors):
        await FallingEdge(dut.clk)
        in_sig = getattr(dut, "in")
        in_sig.value = f_bits
        dut.frac_bits.value = frac_bits

        await RisingEdge(dut.clk)
        await ReadOnly()
        observed.append(int(dut.out.value))

    # Drain pipeline
    for _ in range(LAT):
        await RisingEdge(dut.clk)
        await ReadOnly()
        observed.append(int(dut.out.value))

    return expected, observed

def check_stream(dut, vectors, expected, observed, latency=LAT):
    assert len(observed) == len(expected) + latency
    mismatches = []
    for i, exp in enumerate(expected):
        got = observed[i + latency]
        if got != exp:
            f_bits, frac_bits = vectors[i]
            f_val = bits_to_float(f_bits)
            mismatches.append((i, f_bits, frac_bits, f_val, exp, got))

    if mismatches:
        lines = ["Stream mismatch(es):"]
        for i, f_bits, fb, f_val, exp, got in mismatches[:24]:
            lines.append(f"  idx={i:04d} float=0x{f_bits:08x} ({f_val}) frac={fb} expected=0x{exp:04x} got=0x{got:04x}")
        if len(mismatches) > 24:
            lines.append(f"  ... and {len(mismatches) - 24} more")
        raise AssertionError("\n".join(lines))

@cocotb.test()
async def test_ftofx_stream(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    directed = []

    # Zero
    directed.append((float_to_bits(0.0), 8))
    directed.append((float_to_bits(-0.0), 8))

    # Simple conversions with frac_bits=8 (multiply by 256)
    directed.append((float_to_bits(1.0), 8))    # -> 256
    directed.append((float_to_bits(0.5), 8))    # -> 128
    directed.append((float_to_bits(0.25), 8))   # -> 64
    directed.append((float_to_bits(2.0), 8))    # -> 512
    directed.append((float_to_bits(127.0), 8))  # -> 32512
    directed.append((float_to_bits(-1.0), 8))   # -> -256 = 0xFF00
    directed.append((float_to_bits(-0.5), 8))   # -> -128 = 0xFF80
    directed.append((float_to_bits(-128.0), 8)) # -> -32768 clamped to -32767

    # Boundary: exactly at int16 limits
    directed.append((float_to_bits(127.99609375), 8))  # 32767/256 -> 32767
    directed.append((float_to_bits(-127.99609375), 8)) # -> -32767

    # Overflow positive
    directed.append((float_to_bits(128.0), 8))    # -> 32768 > 32767, clamp
    directed.append((float_to_bits(1000.0), 8))   # definitely overflow

    # Overflow negative
    directed.append((float_to_bits(-128.001), 8)) # barely over -32768, clamp
    directed.append((float_to_bits(-1000.0), 8))  # definitely overflow

    # Different frac_bits values
    directed.append((float_to_bits(1.0), 0))   # -> 1
    directed.append((float_to_bits(1.0), 1))   # -> 2
    directed.append((float_to_bits(1.0), 4))   # -> 16
    directed.append((float_to_bits(0.5), 15))  # -> 16384
    directed.append((float_to_bits(0.25), 15)) # -> 8192

    # Small values (test right shifts)
    directed.append((float_to_bits(0.00390625), 8))  # 1/256 -> 1
    directed.append((float_to_bits(0.001953125), 8)) # 1/512 -> 0 (truncated)
    directed.append((float_to_bits(1e-10), 8))       # very small -> 0

    # Denormals (DAZ)
    directed.append((0x00000001, 8))  # Smallest denormal -> 0
    directed.append((0x007FFFFF, 8))  # Largest denormal -> 0

    # Random tests
    random.seed(0xDEAD)
    for _ in range(100):
        f = random.uniform(-100.0, 100.0)
        fb = random.randint(0, 12)
        directed.append((float_to_bits(f), fb))

    # Random floats in [0,1]
    for _ in range(50):
        f = random.uniform(0.0, 1.0)
        fb = random.randint(8, 12)
        directed.append((float_to_bits(f), fb))

    expected, observed = await run_stream(dut, directed)
    check_stream(dut, directed, expected, observed, latency=LAT)

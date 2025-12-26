"""cocotb tests for 32-bit barrel shifter using stream/array comparison (fixed latency)."""

import ctypes
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, ReadOnly

# ---- must match RTL encodings ----
OP_SHL = 0b00
OP_SHR = 0b01
OP_SRA = 0b11

LAT = 1  # fixed pipeline latency (cycles from input sample to output reg)

# ----------------- helpers -----------------
def to_s32(n: int) -> int:
    return ctypes.c_int32(n & 0xFFFFFFFF).value

def fmt_hex(n: int) -> str:
    return f"0x{(n & 0xFFFFFFFF):08x}"

def golden(op: int, in1: int, sh: int) -> int:
    in1_u = in1 & 0xFFFFFFFF
    sh &= 0x1F
    if op == OP_SHL:
        return (in1_u << sh) & 0xFFFFFFFF
    if op == OP_SHR:
        return (in1_u >> sh) & 0xFFFFFFFF
    if op == OP_SRA:
        return (to_s32(in1_u) >> sh) & 0xFFFFFFFF
    return 0

async def reset(dut, cycles=2):
    dut.rst.value = 1
    for _ in range(cycles):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

async def run_stream(dut, vectors):
    """
    Drive one vector per cycle, collect out each cycle.
    Returns (expected, observed) arrays. observed has len = len(vectors) + LAT.
    """
    expected = [golden(*v) for v in vectors]
    observed = []

    # Align phase: drive on falling, sample on rising
    await RisingEdge(dut.clk)

    # Issue all inputs, collecting an output each cycle
    for i, (op, in1, sh) in enumerate(vectors):
        await FallingEdge(dut.clk)
        dut.op.value = op
        dut.in1.value = in1 & 0xFFFFFFFF
        dut.shift_amt.value = sh & 0x1F

        await RisingEdge(dut.clk)
        await ReadOnly()
        observed.append(int(dut.out.value))

        dut._log.debug(f"[ISSUE {i:04d}] op={op:02b} in1={fmt_hex(in1)} sh={sh:2d} "
                       f"out_now={fmt_hex(observed[-1])}")

    # Drain the pipeline: LAT more cycles of outputs
    for d in range(LAT):
        await RisingEdge(dut.clk)
        await ReadOnly()
        observed.append(int(dut.out.value))
        dut._log.debug(f"[DRAIN {d}] out_now={fmt_hex(observed[-1])}")

    return expected, observed

def check_stream(dut, expected, observed, latency=LAT):
    """
    Check that observed stream equals expected stream shifted by 'latency'.
    observed should have len = len(expected) + latency.
    """
    assert len(observed) == len(expected) + latency, (
        f"Observed length {len(observed)} != expected {len(expected)} + latency {latency}"
    )

    # Compare per item
    mismatches = []
    for i, exp in enumerate(expected):
        got = observed[i + latency]
        if got != exp:
            mismatches.append((i, exp, got))

    if mismatches:
        lines = ["Stream mismatch(s):"]
        # Print up to a handful so logs stay readable
        for i, exp, got in mismatches[:16]:
            lines.append(f"  idx={i:04d} exp={fmt_hex(exp)} got={fmt_hex(got)}")
        if len(mismatches) > 16:
            lines.append(f"  ... and {len(mismatches) - 16} more")
        raise AssertionError("\n".join(lines))

# ----------------- test -----------------
@cocotb.test()
async def test_shifter_stream_fixed_latency(dut):
    """
    Feed an input array through the DUT, capture the output array (including drain),
    and verify observed[latency + i] == expected[i] for all i.
    """
    # clock & reset
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    # --- directed edge cases first (good for smoke) ---
    directed = [
        # zero shifts
        (OP_SHL, 0x12345678, 0),
        (OP_SHR, 0x89ABCDEF, 0),
        (OP_SRA, 0x80000000, 0),

        # max shifts
        (OP_SHL, 0x00000001, 31),
        (OP_SHR, 0x80000000, 31),
        (OP_SRA, 0x80000000, 31),

        # boundaries
        (OP_SHL, 0xFFFFFFFF, 1),
        (OP_SHR, 0xFFFFFFFF, 1),
        (OP_SRA, 0xFFFFFFFF, 1),

        (OP_SHL, 0x0000FFFF, 16),
        (OP_SHR, 0xFFFF0000, 16),
        (OP_SRA, 0xFFFF0000, 16),

        (OP_SHL, 0x80000000, 1),
        (OP_SHR, 0x00000001, 31),
        (OP_SRA, 0x80000001, 1),

        # mid values
        (OP_SHL, 0x12345678, 4),
        (OP_SHR, 0x12345678, 4),
        (OP_SRA, 0x12345678, 4),
    ]

    # --- random traffic (kept simple; no bubbles) ---
    N_RANDOM = 300
    rnd = [
        (random.choice([OP_SHL, OP_SHR, OP_SRA]),
         random.randrange(0, 1 << 32),
         random.randrange(0, 32))
        for _ in range(N_RANDOM)
    ]

    vectors = directed + rnd

    expected, observed = await run_stream(dut, vectors)
    check_stream(dut, expected, observed, latency=LAT)

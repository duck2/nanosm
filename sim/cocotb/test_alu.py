import ctypes
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, ReadOnly

# ---- Operation encoding (must match RTL) ------------------------------------
OP_ADD  = 0b00000  # y = in1 + in2 (32-bit, wrap)
OP_SUB  = 0b00001  # y = in1 - in2 (32-bit, wrap)
OP_MULS = 0b00010  # y = (int16)in1[15:0] * (int16)in2[15:0] -> 32-bit
OP_MULU = 0b00011  # y = (uint16)in1[15:0] * (uint16)in2[15:0] -> 32-bit
OP_XOR  = 0b00100  # y = in1 ^ in2
OP_OR   = 0b00101  # y = in1 | in2
OP_AND  = 0b00111  # y = in1 & in2

# ---- Fixed pipeline latency (cycles from input sample to out) ---------------
LAT = 1  # two registers on the DSP path, no others

# ---- Helpers ----------------------------------------------------------------
def to_s16(n: int) -> int:
    return ctypes.c_int16(n & 0xFFFF).value

def to_s32(n: int) -> int:
    return ctypes.c_int32(n & 0xFFFFFFFF).value

def mask32(n: int) -> int:
    return n & 0xFFFFFFFF

def fmt_hex(n: int) -> str:
    return f"0x{mask32(n):08x}"

def golden(op: int, in1: int, in2: int) -> int:
    """Software model of ALU ops, 32-bit wraparound."""
    a = mask32(in1)
    b = mask32(in2)
    if op == OP_ADD:
        return mask32(a + b)
    if op == OP_SUB:
        return mask32(a - b)
    if op == OP_MULS:
        return mask32(to_s16(a) * to_s16(b))
    if op == OP_MULU:
        return mask32((a & 0xFFFF) * (b & 0xFFFF))
    if op == OP_XOR:
        return mask32(a ^ b)
    if op == OP_OR:
        return mask32(a | b)
    if op == OP_AND:
        return mask32(a & b)
    return 0

async def reset_dut(dut, cycles: int = 2):
    """Synchronous reset aligned to the clock."""
    dut.rst.value = 1
    for _ in range(cycles):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

# ---- Single pipeline function -----------------------------------------------
async def run_stream(dut, vectors):
    """
    Drive one (op,in1,in2) per cycle; collect output every cycle.
    Returns (expected, observed) where observed has len == len(vectors) + LAT.
    """
    expected = [golden(op, a, b) for (op, a, b) in vectors]
    observed = []

    # Align phase: drive on falling, sample on rising
    await RisingEdge(dut.clk)

    for i, (op, a, b) in enumerate(vectors):
        # Put inputs for cycle i just after the falling edge
        await FallingEdge(dut.clk)
        dut.op.value  = op
        dut.in1.value = mask32(a)
        dut.in2.value = mask32(b)

        # Rising edge: DUT samples inputs into its first stage
        await RisingEdge(dut.clk)
        await ReadOnly()
        observed.append(int(dut.out.value))

        # Optional debug:
        # dut._log.debug(f"[ISSUE {i:04d}] op={op:02b} a={fmt_hex(a)} b={fmt_hex(b)} out={fmt_hex(observed[-1])}")

    # Drain the remaining LAT cycles
    for d in range(LAT):
        await RisingEdge(dut.clk)
        await ReadOnly()
        observed.append(int(dut.out.value))
        # dut._log.debug(f"[DRAIN {d}] out={fmt_hex(observed[-1])}")

    return expected, observed

def check_stream(expected, observed, latency: int = LAT):
    """Verify observed stream equals expected stream shifted by 'latency' cycles."""
    assert len(observed) == len(expected) + latency, (
        f"Observed length {len(observed)} != expected {len(expected)} + latency {latency}"
    )
    mismatches = []
    for i, exp in enumerate(expected):
        got = observed[i + latency]
        if got != exp:
            mismatches.append((i, exp, got))
    if mismatches:
        lines = ["Stream mismatch(es):"]
        for i, exp, got in mismatches[:24]:
            lines.append(f"  idx={i:04d} exp={fmt_hex(exp)} got={fmt_hex(got)}")
        if len(mismatches) > 24:
            lines.append(f"  ... and {len(mismatches) - 24} more")
        raise AssertionError("\n".join(lines))

# ---- One cohesive test ------------------------------------------------------
@cocotb.test()
async def test_alu_stream_fixed_latency(dut):    # Clock & reset
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Directed smoke (covers boundaries and mixing ops)
    directed = [
        # ADD/SUB wraparound and edges
        (OP_ADD, 0x00000005, 0x00000003),
        (OP_ADD, 0xFFFFFFFF, 0x00000001),
        (OP_ADD, 0x7FFFFFFF, 0x00000001),
        (OP_SUB, 0x000000FF, 0x00000010),
        (OP_SUB, 0x00000000, 0x00000001),
        (OP_SUB, 0x80000000, 0x00000001),

        # MULS / MULU (only lower 16 bits participate)
        (OP_MULS, 0x00000010, 0x00000100),   # 16 * 256
        (OP_MULS, 0x0000FFFF, 0x0000FFFF),  # (-1) * (-1) = 1
        (OP_MULS, 0x00007FFF, 0x00000002),  # 32767 * 2
        (OP_MULS, 0x0000FFFF, 0x00000002),  # (-1) * 2 = -2

        (OP_MULU, 0x00000003, 0x00000005),  # 3 * 5
        (OP_MULU, 0x00000100, 0x00000100),  # 256 * 256
        (OP_MULU, 0x0000FFFF, 0x0000FFFF),  # 65535 * 65535
        (OP_MULU, 0x00001000, 0x00001000),  # 4096 * 4096

        # Bitwise ops
        (OP_XOR, 0xAAAAAAAA, 0x55555555),
        (OP_XOR, 0x12345678, 0x12345678),
        (OP_OR,  0x12340000, 0x00005678),
        (OP_OR,  0x00000000, 0xFFFFFFFF),
        (OP_AND, 0xFF00FF00, 0x00FF00FF),
        (OP_AND, 0xFFFFFFFF, 0x12345678),
    ]

    # Randomized traffic for stress
    random.seed(42)
    OPS = [OP_ADD, OP_SUB, OP_MULS, OP_MULU, OP_XOR, OP_OR, OP_AND]
    N_RANDOM = 500
    rnd = []
    for _ in range(N_RANDOM):
        op = random.choice(OPS)
        a = random.randrange(0, 1 << 32)
        b = random.randrange(0, 1 << 32)
        rnd.append((op, a, b))

    vectors = directed + rnd

    expected, observed = await run_stream(dut, vectors)
    check_stream(expected, observed, latency=LAT)

import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, ReadOnly

# Latency
LAT = 2

# ---- Helpers ----------------------------------------------------------------
def mask24(n: int) -> int:
    return n & 0xFFFFFF

def mask48(n: int) -> int:
    return n & 0xFFFFFFFFFFFF

def fmt_hex24(n: int) -> str:
    return f"0x{mask24(n):06x}"

def fmt_hex48(n: int) -> str:
    return f"0x{mask48(n):012x}"

def golden(a: int, b: int) -> int:
    """Software model of 24x24 unsigned multiply."""
    return mask48(mask24(a) * mask24(b))

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
    Drive one (a,b) per cycle; collect output every cycle.
    Returns (expected, observed) where observed has len == len(vectors) + LAT.
    """
    expected = [golden(a, b) for (a, b) in vectors]
    observed = []

    # Align phase: drive on falling, sample on rising
    await RisingEdge(dut.clk)

    for i, (a, b) in enumerate(vectors):
        # Put inputs for cycle i just after the falling edge
        await FallingEdge(dut.clk)
        dut.a.value = mask24(a)
        dut.b.value = mask24(b)

        # Rising edge: DUT samples inputs into its first stage
        await RisingEdge(dut.clk)
        await ReadOnly()
        observed.append(int(dut.p.value))

    # Drain the remaining LAT cycles
    for d in range(LAT):
        await RisingEdge(dut.clk)
        await ReadOnly()
        observed.append(int(dut.p.value))

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
            lines.append(f"  idx={i:04d} exp={fmt_hex48(exp)} got={fmt_hex48(got)}")
        if len(mismatches) > 24:
            lines.append(f"  ... and {len(mismatches) - 24} more")
        raise AssertionError("\n".join(lines))

# ---- One cohesive test ------------------------------------------------------
@cocotb.test()
async def test_mult_stream_fixed_latency(dut):
    # Clock & reset
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Directed smoke (covers boundaries and edge cases)
    directed = [
        # Basic cases
        (0x000000, 0x000000),  # 0 * 0
        (0x000001, 0x000001),  # 1 * 1
        (0x000002, 0x000003),  # 2 * 3
        (0x000010, 0x000100),  # 16 * 256
        (0x000100, 0x000100),  # 256 * 256
        (0x001000, 0x001000),  # 4096 * 4096
        
        # Maximum values
        (0xFFFFFF, 0x000001),  # max * 1
        (0x000001, 0xFFFFFF),  # 1 * max
        (0xFFFFFF, 0xFFFFFF),  # max * max = 0xFFFFFE000001
        
        # Powers of 2
        (0x000001, 0x000001),  # 2^0 * 2^0
        (0x000002, 0x000002),  # 2^1 * 2^1
        (0x000004, 0x000004),  # 2^2 * 2^2
        (0x000008, 0x000008),  # 2^3 * 2^3
        (0x000100, 0x000200),  # 2^8 * 2^9
        (0x001000, 0x002000),  # 2^12 * 2^13
        (0x010000, 0x020000),  # 2^16 * 2^17
        (0x100000, 0x200000),  # 2^20 * 2^21
        (0x800000, 0x800000),  # 2^23 * 2^23
        
        # Test upper/lower bit interaction
        (0xFFFF00, 0x0000FF),  # upper bits * lower bits
        (0x0000FF, 0xFFFF00),  # lower bits * upper bits
        (0xFF0000, 0x0000FF),  # highest byte * lowest byte
        (0x0000FF, 0xFF0000),  # lowest byte * highest byte
        
        # Test 17-bit boundary (where split happens)
        (0x01FFFF, 0x000001),  # max 17-bit value * 1
        (0x000001, 0x01FFFF),  # 1 * max 17-bit value
        (0x01FFFF, 0x01FFFF),  # max 17-bit * max 17-bit
        (0x020000, 0x000001),  # first value above 17 bits
        (0x020000, 0x020000),  # 2^17 * 2^17
        
        # Alternating bit patterns
        (0xAAAAAA, 0x555555),  # alternating bits
        (0x555555, 0xAAAAAA),  # alternating bits (swapped)
        (0xAAAAAA, 0xAAAAAA),  # all alternating
        (0x555555, 0x555555),  # all alternating
        
        # Walking ones
        (0x000001, 0xFFFFFF),
        (0x000002, 0xFFFFFF),
        (0x000004, 0xFFFFFF),
        (0x000008, 0xFFFFFF),
        (0x000010, 0xFFFFFF),
        (0x000020, 0xFFFFFF),
        (0x000040, 0xFFFFFF),
        (0x000080, 0xFFFFFF),
    ]

    # Randomized traffic for stress
    random.seed(42)
    N_RANDOM = 500
    rnd = []
    for _ in range(N_RANDOM):
        a = random.randrange(0, 1 << 24)
        b = random.randrange(0, 1 << 24)
        rnd.append((a, b))

    vectors = directed + rnd

    expected, observed = await run_stream(dut, vectors)
    check_stream(expected, observed, latency=LAT)

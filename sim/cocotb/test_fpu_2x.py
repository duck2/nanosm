"""Smoke test for fpu_2x wrapper with dual clocks."""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

from test_fpu_soft import simulate_fma_core, bits_to_f32

FPU_LATENCY = 15

async def start_clocks(dut):
    """Start both clocks, aligned (clk_2x is 2x frequency of clk)."""
    clk_period = 8  # 125 MHz
    clk_2x_period = 4  # 250 MHz
    cocotb.start_soon(Clock(dut.clk, clk_period, units="ns").start())
    cocotb.start_soon(Clock(dut.clk_2x, clk_2x_period, units="ns").start())
    await Timer(1, units="ns")  # let clocks settle

async def reset_dut(dut, cycles=4):
    """Synchronous reset."""
    dut.rst.value = 1
    for _ in range(cycles):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

@cocotb.test()
async def test_fpu_2x_smoke(dut):
    """Smoke test: drive a few 2-wide operations and check outputs."""
    await start_clocks(dut)
    await reset_dut(dut)

    # Test vectors: (in1, in2, in3) as bits - simple MUL cases (in3=0)
    test_cases = [
        (0x3F800000, 0x40000000, 0x00000000),  # 1.0 * 2.0 + 0 = 2.0
        (0x40400000, 0x40800000, 0x00000000),  # 3.0 * 4.0 + 0 = 12.0
        (0x3F000000, 0x3F000000, 0x3F800000),  # 0.5 * 0.5 + 1.0 = 1.25
        (0x40A00000, 0x40C00000, 0x41200000),  # 5.0 * 6.0 + 10.0 = 40.0
    ]

    # Compute expected results
    expected = [simulate_fma_core(t[0], t[1], t[2]) for t in test_cases]

    # Drive 2 operations per clk cycle
    results = []
    num_pairs = len(test_cases) // 2

    for i in range(num_pairs):
        t0 = test_cases[i * 2]
        t1 = test_cases[i * 2 + 1]

        dut.in1_0.value = t0[0]
        dut.in2_0.value = t0[1]
        dut.in3_0.value = t0[2]
        dut.valid_in_0.value = 1

        dut.in1_1.value = t1[0]
        dut.in2_1.value = t1[1]
        dut.in3_1.value = t1[2]
        dut.valid_in_1.value = 1

        await RisingEdge(dut.clk)

    # Stop driving
    dut.valid_in_0.value = 0
    dut.valid_in_1.value = 0

    # Wait for pipeline to flush and collect results
    for _ in range(FPU_LATENCY + 5):
        await RisingEdge(dut.clk)
        if dut.valid_out_0.value == 1:
            results.append(int(dut.out_0.value))
        if dut.valid_out_1.value == 1:
            results.append(int(dut.out_1.value))

    # Check results
    print(f"\nExpected {len(expected)} results, got {len(results)}")
    for i, (exp, got) in enumerate(zip(expected, results)):
        exp_f = bits_to_f32(exp)
        got_f = bits_to_f32(got)
        status = "PASS" if exp == got else "FAIL"
        print(f"  [{status}] op{i}: exp=0x{exp:08X} ({exp_f}), got=0x{got:08X} ({got_f})")

    assert len(results) >= len(expected), f"Missing outputs: got {len(results)}, expected {len(expected)}"
    for i, (exp, got) in enumerate(zip(expected, results)):
        assert exp == got, f"Mismatch at op{i}: exp=0x{exp:08X}, got=0x{got:08X}"

    print("\n[PASS] fpu_2x smoke test passed!")

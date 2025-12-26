"""Cocotb testbench for the GPU register file (rf.sv)"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import random

async def write_reg(dut, reg, values, mask=0xFFFF):
    """Write values to a register with optional mask."""
    dut.wr_en.value = mask
    dut.wr_addr.value = reg
    for t in range(16):
        getattr(dut, f"wr_data[{t}]").value = values[t]
    await RisingEdge(dut.clk)
    dut.wr_en.value = 0

async def read_reg(dut, reg, port='a'):
    """Read all 16 lanes from a register on specified port (1 cycle latency)."""
    getattr(dut, f"rd_addr_{port}").value = reg
    await RisingEdge(dut.clk)
    await Timer(1, units='ps')
    return [getattr(dut, f"rd_data_{port}[{t}]").value.integer for t in range(16)]

@cocotb.test()
async def test_rf(dut):
    """Test register file: basic r/w, write mask, R0 hardwired to zero."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    dut.wr_en.value = 0
    await RisingEdge(dut.clk)

    # Basic read/write
    test_vals = [0x1000 + t for t in range(16)]
    await write_reg(dut, 1, test_vals)
    await RisingEdge(dut.clk)
    
    for port in ['a', 'b', 'c']:
        result = await read_reg(dut, 1, port)
        for t in range(16):
            assert result[t] == test_vals[t], f"Port {port} thread {t}: got {hex(result[t])}, expected {hex(test_vals[t])}"
    dut._log.info("Basic read/write: PASS")

    # Write mask test
    initial_vals = [0x5000 + t for t in range(16)]
    new_vals = [0x6000 + t for t in range(16)]
    await write_reg(dut, 5, initial_vals)
    await write_reg(dut, 5, new_vals, mask=0xAAAA)  # Only odd threads
    await RisingEdge(dut.clk)
    
    result = await read_reg(dut, 5)
    for t in range(16):
        expected = new_vals[t] if t % 2 == 1 else initial_vals[t]
        assert result[t] == expected, f"Thread {t}: got {hex(result[t])}, expected {hex(expected)}"
    dut._log.info("Write mask: PASS")

    # R0 always zero
    await write_reg(dut, 0, [0xDEADBEEF] * 16)
    await RisingEdge(dut.clk)
    
    for port in ['a', 'b', 'c']:
        result = await read_reg(dut, 0, port)
        for t in range(16):
            assert result[t] == 0, f"R0 port {port} thread {t} should be 0, got {hex(result[t])}"
    dut._log.info("R0 hardwired zero: PASS")

    # Quick random sanity check
    for reg in range(1, 32):
        vals = [random.randint(0, 0xFFFFFFFF) for _ in range(16)]
        await write_reg(dut, reg, vals)
        await RisingEdge(dut.clk)
        result = await read_reg(dut, reg)
        for t in range(16):
            assert result[t] == vals[t], f"R{reg}[{t}]: got {hex(result[t])}, expected {hex(vals[t])}"
    dut._log.info("All registers sanity check: PASS")

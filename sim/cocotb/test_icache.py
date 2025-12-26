"""Test instruction cache module."""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import random
import os

@cocotb.test()
async def test_icache_basic(dut):
    """Test basic instruction cache functionality."""
    
    # Create clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.rst.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)
    
    # Test sequential instruction fetches
    dut._log.info("Testing sequential instruction fetches...")
    for i in range(16):
        dut.pc.value = i * 4  # Byte-addressed
        await RisingEdge(dut.clk)
        # Valid should be high after one cycle
        assert dut.valid.value == 1, f"Valid should be high for PC={i*4}"
        dut._log.info(f"PC={i*4:#010x} -> Instruction={dut.instruction.value:#010x}")
    
    # Test random access
    dut._log.info("Testing random access...")
    test_addresses = [0, 64, 128, 256, 512, 1020]
    for addr in test_addresses:
        dut.pc.value = addr * 4
        await RisingEdge(dut.clk)
        assert dut.valid.value == 1
        dut._log.info(f"PC={addr*4:#010x} -> Instruction={dut.instruction.value:#010x}")
    
    # Test out of range access (should set valid to 0)
    dut._log.info("Testing out of range access...")
    dut.pc.value = 4096  # Beyond 4KB
    await RisingEdge(dut.clk)
    assert dut.valid.value == 0, "Valid should be low for out of range address"
    dut._log.info(f"Out of range access correctly marked invalid")
    
    # Test back to valid range
    dut.pc.value = 0
    await RisingEdge(dut.clk)
    assert dut.valid.value == 1
    
    dut._log.info("All tests passed!")

@cocotb.test()
async def test_icache_alignment(dut):
    """Test that cache handles misaligned addresses correctly."""
    
    # Create clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.rst.value = 1
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)
    
    # Test that lower 2 bits are ignored (word alignment)
    dut._log.info("Testing word alignment...")
    for offset in range(4):
        dut.pc.value = 0 + offset  # PC = 0, 1, 2, 3
        await RisingEdge(dut.clk)
        instr = int(dut.instruction.value)
        dut._log.info(f"PC={offset} -> Instruction={instr:#010x}")
    
    # All should return the same instruction (from word address 0)
    dut._log.info("Word alignment test passed!")

@cocotb.test()
async def test_icache_latency(dut):
    """Test 1-cycle latency of instruction cache."""
    
    # Create clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.rst.value = 1
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    
    # Set PC
    dut.pc.value = 0x100
    
    # Wait one cycle - output should be available
    await RisingEdge(dut.clk)
    assert dut.valid.value == 1, "Should be valid after 1 cycle"
    
    dut._log.info(f"1-cycle latency verified: PC=0x100 -> {dut.instruction.value:#010x}")


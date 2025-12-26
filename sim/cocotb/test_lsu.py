"""Test load-store unit (LSU) module."""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import random

class MemoryBank:
    """Simulates a 16-bank BRAM memory system."""
    
    def __init__(self, depth=1024):
        self.depth = depth
        self.banks = [[0 for _ in range(depth)] for _ in range(16)]
    
    def write(self, bank, row, data):
        """Write data to a specific bank and row."""
        if 0 <= bank < 16 and 0 <= row < self.depth:
            self.banks[bank][row] = data & 0xFFFFFFFF
    
    def read(self, bank, row):
        """Read data from a specific bank and row."""
        if 0 <= bank < 16 and 0 <= row < self.depth:
            return self.banks[bank][row]
        return 0
    
    def write_vector(self, base_addr, data):
        """Write 16 words starting at base_addr (word address)."""
        for i in range(16):
            word_addr = base_addr + i
            bank = word_addr & 0xF
            row = word_addr >> 4
            self.write(bank, row, data[i])
    
    def read_vector(self, base_addr):
        """Read 16 words starting at base_addr (word address)."""
        result = []
        for i in range(16):
            word_addr = base_addr + i
            bank = word_addr & 0xF
            row = word_addr >> 4
            result.append(self.read(bank, row))
        return result

async def memory_driver(dut, memory):
    """Drive memory read data based on memory bank model."""
    while True:
        await RisingEdge(dut.clk)
        for i in range(16):
            en = int(dut.mem_en[i].value)
            we = int(dut.mem_we[i].value)
            addr = int(dut.mem_addr[i].value)
            
            if en:
                if we:
                    wdata = int(dut.mem_wdata[i].value)
                    memory.write(i, addr, wdata)
                else:
                    rdata = memory.read(i, addr)
                    dut.mem_rdata[i].value = rdata

@cocotb.test()
async def test_lsu_aligned_load(dut):
    """Test aligned load operation (1-cycle)."""
    
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    memory = MemoryBank()
    cocotb.start_soon(memory_driver(dut, memory))
    
    dut.rst.value = 1
    dut.req_valid.value = 0
    dut.req_is_store.value = 0
    dut.base_addr.value = 0
    for i in range(16):
        dut.store_data[i].value = 0
    
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)
    
    dut._log.info("Testing aligned load (base_addr = 0x0)...")
    
    expected_data = [0x1000 + i for i in range(16)]
    memory.write_vector(0, expected_data)
    
    dut.req_valid.value = 1
    dut.req_is_store.value = 0
    dut.base_addr.value = 0
    
    await RisingEdge(dut.clk)
    assert int(dut.req_ready.value) == 0, "req_ready should be low during operation"
    
    await RisingEdge(dut.clk)
    assert int(dut.resp_valid.value) == 1, "resp_valid should be high after 2 cycles (aligned)"
    
    for i in range(16):
        result = int(dut.load_data[i].value)
        assert result == expected_data[i], f"Thread {i}: expected {expected_data[i]:#x}, got {result:#x}"
        dut._log.info(f"Thread {i}: loaded {result:#010x}")
    
    dut.req_valid.value = 0
    await RisingEdge(dut.clk)
    
    dut._log.info("Aligned load test passed!")

@cocotb.test()
async def test_lsu_aligned_store(dut):
    """Test aligned store operation (1-cycle)."""
    
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    memory = MemoryBank()
    cocotb.start_soon(memory_driver(dut, memory))
    
    dut.rst.value = 1
    dut.req_valid.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)
    
    dut._log.info("Testing aligned store (base_addr = 0x0)...")
    
    store_data = [0x2000 + i for i in range(16)]
    
    dut.req_valid.value = 1
    dut.req_is_store.value = 1
    dut.base_addr.value = 0
    for i in range(16):
        dut.store_data[i].value = store_data[i]
    
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    assert int(dut.resp_valid.value) == 1, "resp_valid should be high after 2 cycles"
    
    result = memory.read_vector(0)
    for i in range(16):
        assert result[i] == store_data[i], f"Thread {i}: expected {store_data[i]:#x}, got {result[i]:#x}"
        dut._log.info(f"Thread {i}: stored {result[i]:#010x}")
    
    dut.req_valid.value = 0
    await RisingEdge(dut.clk)
    
    dut._log.info("Aligned store test passed!")

@cocotb.test()
async def test_lsu_unaligned_load(dut):
    """Test unaligned load operation (2-cycle)."""
    
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    memory = MemoryBank()
    cocotb.start_soon(memory_driver(dut, memory))
    
    dut.rst.value = 1
    dut.req_valid.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)
    
    dut._log.info("Testing unaligned load (base_addr = 0x5)...")
    
    test_data = [0x3000 + i for i in range(32)]
    for i in range(32):
        word_addr = i
        bank = word_addr & 0xF
        row = word_addr >> 4
        memory.write(bank, row, test_data[i])
    
    dut.req_valid.value = 1
    dut.req_is_store.value = 0
    dut.base_addr.value = 5
    
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    assert int(dut.resp_valid.value) == 1, "resp_valid should be high after 3 cycles (unaligned)"
    
    for i in range(16):
        result = int(dut.load_data[i].value)
        expected = test_data[5 + i]
        assert result == expected, f"Thread {i}: expected {expected:#x}, got {result:#x}"
        dut._log.info(f"Thread {i}: loaded {result:#010x} (expected {expected:#010x})")
    
    dut.req_valid.value = 0
    await RisingEdge(dut.clk)
    
    dut._log.info("Unaligned load test passed!")

@cocotb.test()
async def test_lsu_unaligned_store(dut):
    """Test unaligned store operation (2-cycle)."""
    
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    memory = MemoryBank()
    cocotb.start_soon(memory_driver(dut, memory))
    
    dut.rst.value = 1
    dut.req_valid.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)
    
    dut._log.info("Testing unaligned store (base_addr = 0x7)...")
    
    store_data = [0x4000 + i for i in range(16)]
    
    dut.req_valid.value = 1
    dut.req_is_store.value = 1
    dut.base_addr.value = 7
    for i in range(16):
        dut.store_data[i].value = store_data[i]
    
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    assert int(dut.resp_valid.value) == 1, "resp_valid should be high after 3 cycles"
    
    for i in range(16):
        word_addr = 7 + i
        bank = word_addr & 0xF
        row = word_addr >> 4
        result = memory.read(bank, row)
        expected = store_data[i]
        assert result == expected, f"Thread {i}: expected {expected:#x}, got {result:#x}"
        dut._log.info(f"Thread {i}: stored {result:#010x} at addr {word_addr:#x} (bank={bank}, row={row})")
    
    dut.req_valid.value = 0
    await RisingEdge(dut.clk)
    
    dut._log.info("Unaligned store test passed!")

@cocotb.test()
async def test_lsu_multiple_operations(dut):
    """Test multiple back-to-back operations."""
    
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    memory = MemoryBank()
    cocotb.start_soon(memory_driver(dut, memory))
    
    dut.rst.value = 1
    dut.req_valid.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)
    
    dut._log.info("Testing multiple operations...")
    
    test_cases = [
        (0x00, [0x5000 + i for i in range(16)], True),
        (0x10, [0x5100 + i for i in range(16)], True),
        (0x05, [0x5200 + i for i in range(16)], True),
        (0x00, None, False),
        (0x10, None, False),
    ]
    
    for base_addr, data, is_store in test_cases:
        if is_store:
            dut._log.info(f"Store to base_addr={base_addr:#x}")
            dut.req_valid.value = 1
            dut.req_is_store.value = 1
            dut.base_addr.value = base_addr
            for i in range(16):
                dut.store_data[i].value = data[i]
            
            await RisingEdge(dut.clk)
            
            while int(dut.resp_valid.value) == 0:
                await RisingEdge(dut.clk)
            
            dut.req_valid.value = 0
            await RisingEdge(dut.clk)
            
        else:
            dut._log.info(f"Load from base_addr={base_addr:#x}")
            dut.req_valid.value = 1
            dut.req_is_store.value = 0
            dut.base_addr.value = base_addr
            
            await RisingEdge(dut.clk)
            
            while int(dut.resp_valid.value) == 0:
                await RisingEdge(dut.clk)
            
            result = [int(dut.load_data[i].value) for i in range(16)]
            expected = memory.read_vector(base_addr)
            
            for i in range(16):
                assert result[i] == expected[i], f"Mismatch at thread {i}"
                dut._log.info(f"Thread {i}: {result[i]:#010x}")
            
            dut.req_valid.value = 0
            await RisingEdge(dut.clk)
    
    dut._log.info("Multiple operations test passed!")

@cocotb.test()
async def test_lsu_boundary_cases(dut):
    """Test edge cases like bank boundaries."""
    
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    memory = MemoryBank()
    cocotb.start_soon(memory_driver(dut, memory))
    
    dut.rst.value = 1
    dut.req_valid.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)
    
    dut._log.info("Testing boundary cases...")
    
    boundary_addrs = [0x00, 0x10, 0x0F, 0x1F, 0x100, 0x101]
    
    for base_addr in boundary_addrs:
        store_data = [0x6000 + base_addr * 16 + i for i in range(16)]
        
        dut._log.info(f"Testing base_addr={base_addr:#x}")
        
        dut.req_valid.value = 1
        dut.req_is_store.value = 1
        dut.base_addr.value = base_addr
        for i in range(16):
            dut.store_data[i].value = store_data[i]
        
        await RisingEdge(dut.clk)
        
        while int(dut.resp_valid.value) == 0:
            await RisingEdge(dut.clk)
        
        dut.req_valid.value = 0
        await RisingEdge(dut.clk)
        
        dut.req_valid.value = 1
        dut.req_is_store.value = 0
        dut.base_addr.value = base_addr
        
        await RisingEdge(dut.clk)
        
        while int(dut.resp_valid.value) == 0:
            await RisingEdge(dut.clk)
        
        for i in range(16):
            result = int(dut.load_data[i].value)
            expected = store_data[i]
            assert result == expected, f"Addr {base_addr:#x}, Thread {i}: expected {expected:#x}, got {result:#x}"
        
        dut.req_valid.value = 0
        await RisingEdge(dut.clk)
    
    dut._log.info("Boundary cases test passed!")


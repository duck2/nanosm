"""Cocotb testbench for shmem.sv (shared memory with bank conflict resolution)."""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

LANES = 8
ADDR_W = 12
BANK_BITS = 3


def make_addr(bank: int, row: int) -> int:
    """Construct address from bank and row."""
    return (row << BANK_BITS) | bank


async def reset(dut):
    """Reset the DUT."""
    dut.rst.value = 1
    dut.w_valid.value = 0
    dut.r_valid.value = 0
    dut.w_lane_mask.value = 0
    dut.r_lane_mask.value = 0
    for i in range(LANES):
        getattr(dut, f"waddr[{i}]").value = 0
        getattr(dut, f"wdata[{i}]").value = 0
        getattr(dut, f"raddr[{i}]").value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)


async def write_lanes(dut, addrs: list[int], data: list[int], mask: int):
    """Perform a write operation. Wait until complete."""
    dut.w_valid.value = 1
    dut.w_lane_mask.value = mask
    for i in range(LANES):
        getattr(dut, f"waddr[{i}]").value = addrs[i]
        getattr(dut, f"wdata[{i}]").value = data[i]
    await RisingEdge(dut.clk)
    dut.w_valid.value = 0

    # Wait until arbiter finishes (busy goes low, w_ready goes high)
    for _ in range(LANES + 2):
        await RisingEdge(dut.clk)
        await Timer(1, unit='ps')
        if dut.w_ready.value == 1:
            return
    raise TimeoutError("write_lanes timed out waiting for w_ready")


async def read_lanes(dut, addrs: list[int], mask: int) -> list[int]:
    """Perform a read operation. Returns read data for all lanes."""
    dut.r_valid.value = 1
    dut.r_lane_mask.value = mask
    for i in range(LANES):
        getattr(dut, f"raddr[{i}]").value = addrs[i]
    await RisingEdge(dut.clk)
    dut.r_valid.value = 0

    # Wait for r_done
    for _ in range(LANES + 4):
        await RisingEdge(dut.clk)
        await Timer(1, unit='ps')
        if dut.r_done.value == 1:
            return [getattr(dut, f"rdata[{i}]").value.to_unsigned() for i in range(LANES)]
    raise TimeoutError("read_lanes timed out waiting for r_done")


@cocotb.test()
async def test_basic_write_read(dut):
    """Write to all lanes, read back - no conflicts."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    # Each lane writes to its own bank
    addrs = [make_addr(bank=i, row=0) for i in range(LANES)]
    data = [0xDEAD0000 | i for i in range(LANES)]

    await write_lanes(dut, addrs, data, mask=0xFF)
    result = await read_lanes(dut, addrs, mask=0xFF)

    for i in range(LANES):
        assert result[i] == data[i], f"Lane {i}: got 0x{result[i]:08x}, expected 0x{data[i]:08x}"
    dut._log.info("Basic write/read (no conflict): PASS")


@cocotb.test()
async def test_write_conflict(dut):
    """Write with bank conflict - all lanes to same bank, different rows."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    # All lanes hit bank 0, different rows
    addrs = [make_addr(bank=0, row=i) for i in range(LANES)]
    data = [0xCAFE0000 | i for i in range(LANES)]

    await write_lanes(dut, addrs, data, mask=0xFF)
    result = await read_lanes(dut, addrs, mask=0xFF)

    for i in range(LANES):
        assert result[i] == data[i], f"Lane {i}: got 0x{result[i]:08x}, expected 0x{data[i]:08x}"
    dut._log.info("Write with bank conflict: PASS")


@cocotb.test()
async def test_read_conflict(dut):
    """Read with bank conflict - multiple lanes reading same bank, different rows."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    # Write initial data: all lanes to bank 0
    addrs = [make_addr(bank=0, row=i) for i in range(LANES)]
    data = [0xBEEF0000 | i for i in range(LANES)]
    await write_lanes(dut, addrs, data, mask=0xFF)

    # Read back with conflict
    result = await read_lanes(dut, addrs, mask=0xFF)

    for i in range(LANES):
        assert result[i] == data[i], f"Lane {i}: got 0x{result[i]:08x}, expected 0x{data[i]:08x}"
    dut._log.info("Read with bank conflict: PASS")


@cocotb.test()
async def test_broadcast_read(dut):
    """Broadcast read - all lanes read same address."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    # Write a value to one location
    target_addr = make_addr(bank=3, row=42)
    addrs = [target_addr] + [0] * (LANES - 1)
    data = [0x12345678] + [0] * (LANES - 1)
    await write_lanes(dut, addrs, data, mask=0x01)

    # Broadcast read: all lanes read same address
    bc_addrs = [target_addr] * LANES
    result = await read_lanes(dut, bc_addrs, mask=0xFF)

    for i in range(LANES):
        assert result[i] == 0x12345678, f"Lane {i}: got 0x{result[i]:08x}, expected 0x12345678"
    dut._log.info("Broadcast read: PASS")


@cocotb.test()
async def test_partial_mask(dut):
    """Write and read with partial lane mask."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    # Only lanes 0, 2, 4, 6 active
    addrs = [make_addr(bank=i, row=10) for i in range(LANES)]
    data = [0xABCD0000 | i for i in range(LANES)]
    mask = 0x55  # lanes 0, 2, 4, 6

    await write_lanes(dut, addrs, data, mask=mask)
    result = await read_lanes(dut, addrs, mask=mask)

    for i in [0, 2, 4, 6]:
        assert result[i] == data[i], f"Lane {i}: got 0x{result[i]:08x}, expected 0x{data[i]:08x}"
    dut._log.info("Partial mask: PASS")


@cocotb.test()
async def test_overwrite(dut):
    """Overwrite existing data."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    addrs = [make_addr(bank=i, row=0) for i in range(LANES)]

    # First write
    data1 = [0x11111111] * LANES
    await write_lanes(dut, addrs, data1, mask=0xFF)

    # Second write (overwrite)
    data2 = [0x22222222] * LANES
    await write_lanes(dut, addrs, data2, mask=0xFF)

    result = await read_lanes(dut, addrs, mask=0xFF)

    for i in range(LANES):
        assert result[i] == data2[i], f"Lane {i}: got 0x{result[i]:08x}, expected 0x{data2[i]:08x}"
    dut._log.info("Overwrite: PASS")


@cocotb.test()
async def test_mixed_pattern(dut):
    """Test with mixed bank access pattern - 2 lanes per bank."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    # Lanes 0,4 -> bank 0, lanes 1,5 -> bank 1, etc.
    addrs = [make_addr(bank=i % 4, row=i) for i in range(LANES)]
    data = [0xF0F00000 | i for i in range(LANES)]

    await write_lanes(dut, addrs, data, mask=0xFF)
    result = await read_lanes(dut, addrs, mask=0xFF)

    for i in range(LANES):
        assert result[i] == data[i], f"Lane {i}: got 0x{result[i]:08x}, expected 0x{data[i]:08x}"
    dut._log.info("Mixed pattern (2 per bank): PASS")


@cocotb.test()
async def test_different_rows(dut):
    """Write to different rows in same bank sequentially."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    bank = 5
    rows = [10, 20, 30, 40]
    expected = {}

    # Write to different rows (one lane at a time)
    for row in rows:
        addr = make_addr(bank=bank, row=row)
        data_val = 0x99000000 | (row << 8)
        expected[row] = data_val
        addrs = [addr] + [0] * (LANES - 1)
        data = [data_val] + [0] * (LANES - 1)
        await write_lanes(dut, addrs, data, mask=0x01)

    # Read back each row
    for row in rows:
        addr = make_addr(bank=bank, row=row)
        addrs = [addr] + [0] * (LANES - 1)
        result = await read_lanes(dut, addrs, mask=0x01)
        assert result[0] == expected[row], f"Row {row}: got 0x{result[0]:08x}, expected 0x{expected[row]:08x}"

    dut._log.info("Different rows: PASS")


"""Cocotb testbench for shmem_arbiter.sv"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

LANES = 8
ADDR_W = 12
BANK_BITS = 3
ROW_BITS = ADDR_W - BANK_BITS


def make_addr(bank: int, row: int) -> int:
    """Construct address from bank and row."""
    return (row << BANK_BITS) | bank


def addr_bank(addr: int) -> int:
    return addr & ((1 << BANK_BITS) - 1)


def addr_row(addr: int) -> int:
    return addr >> BANK_BITS


async def reset(dut):
    """Reset the DUT."""
    dut.rst.value = 1
    dut.start.value = 0
    dut.lane_mask.value = 0
    for i in range(LANES):
        getattr(dut, f"addr[{i}]").value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)


async def start_request(dut, addrs: list[int], mask: int):
    """Start a new arbiter request."""
    dut.start.value = 1
    dut.lane_mask.value = mask
    for i in range(LANES):
        getattr(dut, f"addr[{i}]").value = addrs[i]
    await RisingEdge(dut.clk)
    dut.start.value = 0


def get_grant(dut) -> int:
    return dut.grant.value.to_unsigned()


def get_bank_en(dut) -> int:
    return dut.bank_en.value.to_unsigned()


def get_bank_addr(dut, bank: int) -> int:
    return getattr(dut, f"bank_addr[{bank}]").value.to_unsigned()


def check_outputs(dut, addrs: list[int], expected_grant: int):
    """Verify grant, bank_en, and bank_addr are consistent with expected grant."""
    grant = get_grant(dut)
    bank_en = get_bank_en(dut)
    assert grant == expected_grant, f"grant={bin(grant)}, expected {bin(expected_grant)}"

    # Compute expected bank_en and bank_addr from granted lanes
    expected_bank_en = 0
    expected_bank_addr = {}
    for lane in range(LANES):
        if grant & (1 << lane):
            b = addr_bank(addrs[lane])
            expected_bank_en |= (1 << b)
            if b not in expected_bank_addr:
                expected_bank_addr[b] = addr_row(addrs[lane])

    assert bank_en == expected_bank_en, f"bank_en={bin(bank_en)}, expected {bin(expected_bank_en)}"

    for b, exp_row in expected_bank_addr.items():
        actual = get_bank_addr(dut, b)
        assert actual == exp_row, f"bank_addr[{b}]={actual}, expected {exp_row}"


@cocotb.test()
async def test_no_conflict(dut):
    """All lanes hit different banks - should complete in 1 cycle."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    # Each lane hits its own bank (lane i -> bank i)
    addrs = [make_addr(bank=i, row=100+i) for i in range(LANES)]
    await start_request(dut, addrs, mask=0xFF)

    await Timer(1, unit='ps')
    check_outputs(dut, addrs, expected_grant=0xFF)
    assert dut.done.value == 1, "Should complete in 1 cycle"
    dut._log.info("No conflict (all different banks): PASS")


@cocotb.test()
async def test_full_conflict(dut):
    """All lanes hit same bank, different rows - takes LANES cycles."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    # All lanes hit bank 0, different rows
    addrs = [make_addr(bank=0, row=i) for i in range(LANES)]
    await start_request(dut, addrs, mask=0xFF)

    # Each cycle grants exactly one lane in priority order (lane 0 first)
    for cycle in range(LANES):
        await Timer(1, unit='ps')
        expected_grant = 1 << cycle
        check_outputs(dut, addrs, expected_grant=expected_grant)
        is_last = cycle == LANES - 1
        assert dut.done.value == is_last, f"done mismatch at cycle {cycle}"
        if not is_last:
            await RisingEdge(dut.clk)

    dut._log.info(f"Full conflict (all same bank, diff rows): PASS ({LANES} cycles)")


@cocotb.test()
async def test_broadcast_all(dut):
    """All lanes read same address - broadcast, 1 cycle, grants everyone."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    # All lanes read exact same address
    same_addr = make_addr(bank=3, row=42)
    addrs = [same_addr] * LANES
    await start_request(dut, addrs, mask=0xFF)

    await Timer(1, unit='ps')
    check_outputs(dut, addrs, expected_grant=0xFF)
    assert dut.done.value == 1, "Broadcast should complete in 1 cycle"
    dut._log.info("Broadcast all (same address): PASS")


@cocotb.test()
async def test_no_coalesce_same_bank(dut):
    """4 lanes hit bank 0 with different addresses - 4 cycles."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    # Lanes 0-3 hit bank 0 with different rows, lanes 4-7 inactive
    addrs = [make_addr(bank=0, row=i*10) for i in range(4)] + [0, 0, 0, 0]
    await start_request(dut, addrs, mask=0x0F)

    # Priority order: lane 0, 1, 2, 3
    for cycle in range(4):
        await Timer(1, unit='ps')
        expected_grant = 1 << cycle
        check_outputs(dut, addrs, expected_grant=expected_grant)
        is_last = cycle == 3
        assert dut.done.value == is_last, f"done mismatch at cycle {cycle}"
        if not is_last:
            await RisingEdge(dut.clk)

    dut._log.info("No coalesce (4 lanes same bank): PASS")


@cocotb.test()
async def test_mixed_banks(dut):
    """2 lanes per bank, different addresses - 2 cycles."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    # Lanes 0,4 -> bank 0, lanes 1,5 -> bank 1, etc.
    addrs = [make_addr(bank=i % 4, row=i) for i in range(LANES)]
    await start_request(dut, addrs, mask=0xFF)

    # Cycle 1: lanes 0-3 win (first per bank)
    await Timer(1, unit='ps')
    check_outputs(dut, addrs, expected_grant=0x0F)
    assert dut.done.value == 0

    # Cycle 2: lanes 4-7 win
    await RisingEdge(dut.clk)
    await Timer(1, unit='ps')
    check_outputs(dut, addrs, expected_grant=0xF0)
    assert dut.done.value == 1

    dut._log.info("Mixed banks (2 per bank): PASS")


@cocotb.test()
async def test_partial_mask(dut):
    """Only some lanes active, no conflict."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    # Only lanes 0 and 7 active, different banks
    addrs = [make_addr(bank=0, row=1)] + [0]*6 + [make_addr(bank=7, row=2)]
    await start_request(dut, addrs, mask=0x81)

    await Timer(1, unit='ps')
    # Both lanes granted, banks 0 and 7 enabled
    check_outputs(dut, addrs, expected_grant=0x81)
    assert dut.done.value == 1
    dut._log.info("Partial mask (2 lanes, no conflict): PASS")


@cocotb.test()
async def test_partial_broadcast(dut):
    """All addresses same - broadcast grants ALL lanes (even masked out)."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    same_addr = make_addr(bank=5, row=99)
    addrs = [same_addr] * LANES
    await start_request(dut, addrs, mask=0x0E)

    await Timer(1, unit='ps')
    # Broadcast grants ALL lanes per design (RF handles mask)
    check_outputs(dut, addrs, expected_grant=0xFF)
    assert dut.done.value == 1
    dut._log.info("Partial broadcast (grants all): PASS")


@cocotb.test()
async def test_grant_all_served(dut):
    """Verify all lanes eventually get granted in conflict case with correct outputs."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    # All lanes hit bank 0 with different rows
    addrs = [make_addr(bank=0, row=i) for i in range(LANES)]
    await start_request(dut, addrs, mask=0xFF)

    total_granted = 0
    for cycle in range(LANES):
        await Timer(1, unit='ps')
        expected_grant = 1 << cycle
        check_outputs(dut, addrs, expected_grant=expected_grant)
        total_granted |= expected_grant
        dut._log.info(f"Cycle {cycle}: grant={bin(expected_grant)}, bank_addr[0]={cycle}")
        if cycle < LANES - 1:
            await RisingEdge(dut.clk)

    assert dut.done.value == 1
    assert total_granted == 0xFF
    dut._log.info("All lanes served: PASS")


@cocotb.test()
async def test_bank_addr_correct(dut):
    """Verify bank_addr correctly routes winning lane's row."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    # Each lane hits its own bank with unique row
    addrs = [make_addr(bank=i, row=100+i) for i in range(LANES)]
    await start_request(dut, addrs, mask=0xFF)

    await Timer(1, unit='ps')
    check_outputs(dut, addrs, expected_grant=0xFF)
    assert dut.done.value == 1
    dut._log.info("bank_addr correct: PASS")

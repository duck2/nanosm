import sys
from pathlib import Path

import cocotb
from cocotb.triggers import Timer

# Add emu to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "emu"))
from encoding import encode, decode, Fmt, Op, ENCODE_TABLE


async def settle():
    """Let combinational logic settle."""
    await Timer(1, unit="ns")


def check_field(dut, name: str, expected: int):
    """Check a DUT signal matches expected value."""
    val = getattr(dut, name).value
    got = int(val) if hasattr(val, '__int__') else val.to_unsigned()
    assert got == expected, f"{name}: expected {expected}, got {got}"


@cocotb.test()
async def test_decode_r4_alu(dut):
    """Test R4 format ALU instructions."""
    for mnemonic in ['add', 'sub', 'and', 'or', 'xor', 'muls', 'mulu']:
        args = {'rd': 5, 'rs1': 10, 'rs2': 15, 'rs3': 0}
        instr = encode(mnemonic, args)
        dut.instr.value = instr
        await settle()

        check_field(dut, 'fmt', Fmt.R4)
        check_field(dut, 'rd', 5)
        check_field(dut, 'rs1', 10)
        check_field(dut, 'rs2', 15)
        assert dut.is_alu.value == 1, f"{mnemonic} should set is_alu"
        assert dut.writes_rd.value == 1, f"{mnemonic} should set writes_rd"


@cocotb.test()
async def test_decode_r4_shift(dut):
    """Test R4 format shift instructions."""
    for mnemonic in ['shl', 'shr', 'sra']:
        args = {'rd': 1, 'rs1': 2, 'rs2': 3}
        instr = encode(mnemonic, args)
        dut.instr.value = instr
        await settle()

        check_field(dut, 'fmt', Fmt.R4)
        assert dut.is_shift.value == 1, f"{mnemonic} should set is_shift"
        assert dut.writes_rd.value == 1


@cocotb.test()
async def test_decode_r4_fpu(dut):
    """Test R4 format FPU instructions."""
    for mnemonic in ['fadd', 'fsub', 'fmul']:
        args = {'rd': 1, 'rs1': 2, 'rs2': 3}
        instr = encode(mnemonic, args)
        dut.instr.value = instr
        await settle()

        check_field(dut, 'fmt', Fmt.R4)
        assert dut.is_fpu.value == 1, f"{mnemonic} should set is_fpu"
        assert dut.writes_rd.value == 1

    # fma has rs3
    instr = encode('fma', {'rd': 1, 'rs1': 2, 'rs2': 3, 'rs3': 4})
    dut.instr.value = instr
    await settle()
    check_field(dut, 'rs3', 4)
    assert dut.is_fpu.value == 1


@cocotb.test()
async def test_decode_i_format(dut):
    """Test I format (immediate) instructions."""
    for mnemonic in ['addi', 'subi', 'andi', 'ori', 'xori']:
        args = {'rd': 7, 'rs1': 8, 'imm': 123}
        instr = encode(mnemonic, args)
        dut.instr.value = instr
        await settle()

        check_field(dut, 'fmt', Fmt.I)
        check_field(dut, 'rd', 7)
        check_field(dut, 'rs1', 8)
        check_field(dut, 'imm12', 123)
        assert dut.is_alu_imm.value == 1, f"{mnemonic} should set is_alu_imm"

    for mnemonic in ['shli', 'shri', 'srai']:
        args = {'rd': 1, 'rs1': 2, 'imm': 5}
        instr = encode(mnemonic, args)
        dut.instr.value = instr
        await settle()

        check_field(dut, 'fmt', Fmt.I)
        assert dut.is_shift_imm.value == 1, f"{mnemonic} should set is_shift_imm"


@cocotb.test()
async def test_decode_m_format_shmem(dut):
    """Test M format scratchpad memory instructions."""
    # lds
    instr = encode('lds', {'rd': 5, 'rs1': 10, 'imm': 64})
    dut.instr.value = instr
    await settle()

    check_field(dut, 'fmt', Fmt.M)
    check_field(dut, 'rd', 5)
    check_field(dut, 'rs1', 10)
    assert dut.is_load.value == 1
    assert dut.is_shmem.value == 1
    assert dut.writes_rd.value == 1

    # sts
    instr = encode('sts', {'rs1': 10, 'rs2': 5, 'imm': 64})
    dut.instr.value = instr
    await settle()

    assert dut.is_store.value == 1
    assert dut.is_shmem.value == 1
    assert dut.writes_rd.value == 0


@cocotb.test()
async def test_decode_u_format(dut):
    """Test U format (lui)."""
    instr = encode('lui', {'rd': 3, 'imm': 0x12345})
    dut.instr.value = instr
    await settle()

    check_field(dut, 'fmt', Fmt.U)
    check_field(dut, 'rd', 3)
    check_field(dut, 'imm20', 0x12345)
    assert dut.is_lui.value == 1
    assert dut.writes_rd.value == 1


@cocotb.test()
async def test_decode_p_format(dut):
    """Test P format (predicates)."""
    # setp
    instr = encode('setp.lt.i32', {'pd': 1, 'rs1': 5, 'rs2': 6})
    dut.instr.value = instr
    await settle()

    check_field(dut, 'fmt', Fmt.P)
    check_field(dut, 'pd', 1)
    assert dut.is_setp.value == 1

    # pand
    instr = encode('pand', {'pd': 2, 'ps1': 0, 'ps2': 1})
    dut.instr.value = instr
    await settle()

    assert dut.is_pred_logic.value == 1


@cocotb.test()
async def test_decode_j_format(dut):
    """Test J format (branches)."""
    instr = encode('bra', {'target': 0x100})
    dut.instr.value = instr
    await settle()

    check_field(dut, 'fmt', Fmt.J)
    check_field(dut, 'imm22', 0x100)
    assert dut.is_branch.value == 1

    instr = encode('ssy', {'target': 0x200})
    dut.instr.value = instr
    await settle()
    assert dut.is_ssy.value == 1

    instr = encode('sync', {})
    dut.instr.value = instr
    await settle()
    assert dut.is_sync.value == 1


@cocotb.test()
async def test_decode_x_format(dut):
    """Test X format (misc)."""
    instr = encode('sread', {'rd': 5, 'sreg': 0})
    dut.instr.value = instr
    await settle()

    check_field(dut, 'fmt', Fmt.X)
    assert dut.is_sread.value == 1
    assert dut.writes_rd.value == 1

    instr = encode('nop', {})
    dut.instr.value = instr
    await settle()
    assert dut.is_nop.value == 1

    instr = encode('halt', {})
    dut.instr.value = instr
    await settle()
    assert dut.is_halt.value == 1


@cocotb.test()
async def test_decode_predication(dut):
    """Test predicate field decoding."""
    # Unpredicated (pred_reg = 7 = always)
    instr = encode('add', {'rd': 1, 'rs1': 2, 'rs2': 3})
    dut.instr.value = instr
    await settle()
    check_field(dut, 'pred_reg', 7)
    check_field(dut, 'pred_neg', 0)

    # Predicated on p3
    instr = encode('add', {'rd': 1, 'rs1': 2, 'rs2': 3, 'pred': 3})
    dut.instr.value = instr
    await settle()
    check_field(dut, 'pred_reg', 3)
    check_field(dut, 'pred_neg', 0)

    # Negated predicate @!p2
    instr = encode('add', {'rd': 1, 'rs1': 2, 'rs2': 3, 'pred': 2, 'pred_neg': True})
    dut.instr.value = instr
    await settle()
    check_field(dut, 'pred_reg', 2)
    check_field(dut, 'pred_neg', 1)


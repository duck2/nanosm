"""Tests for instruction encoding. Validates formats and round-trip."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from encoding import (
    Field, Format, encode, decode, assemble,
    FMT_R4, FMT_I, FMT_M, FMT_M2, FMT_U, FMT_P, FMT_J, FMT_X,
    Fmt, Op
)


class TestFormat:
    def test_no_overlap(self):
        """All formats should have no overlapping bits."""
        for fmt in [FMT_R4, FMT_I, FMT_M, FMT_M2, FMT_U, FMT_P, FMT_J, FMT_X]:
            assert fmt is not None  # _validate() runs in __init__

    def test_r4_bits_add_up(self):
        """R4 format should use all 32 bits."""
        f = FMT_R4
        total = sum(f.fields[n].width for n in f.fields)
        assert total == 32

    def test_encode_decode_r4(self):
        word = FMT_R4.encode(fmt=0, pred=0, rd=5, rs1=10, rs2=15, rs3=3, op=7)
        fields = FMT_R4.decode(word)
        assert fields['fmt'] == 0
        assert fields['rd'] == 5
        assert fields['rs1'] == 10
        assert fields['rs2'] == 15
        assert fields['rs3'] == 3
        assert fields['op'] == 7


class TestEncodeDecode:
    """Round-trip tests."""

    def test_alu_ops(self):
        for op in ['add', 'sub', 'and', 'or', 'xor', 'muls', 'mulu']:
            args = {'rd': 5, 'rs1': 10, 'rs2': 15}
            word = encode(op, args)
            dec_op, dec_args = decode(word)
            assert dec_op == op, f"{op}: got {dec_op}"
            assert dec_args['rd'] == 5
            assert dec_args['rs1'] == 10
            assert dec_args['rs2'] == 15

    def test_shift_ops(self):
        for op in ['shl', 'shr', 'sra']:
            args = {'rd': 3, 'rs1': 7, 'rs2': 2}
            word = encode(op, args)
            dec_op, dec_args = decode(word)
            assert dec_op == op

    def test_shift_imm_ops(self):
        for op in ['shli', 'shri', 'srai']:
            args = {'rd': 3, 'rs1': 7, 'imm': 5}
            word = encode(op, args)
            dec_op, dec_args = decode(word)
            assert dec_op == op
            assert dec_args['imm'] == 5

    def test_fpu_ops(self):
        for op in ['fadd', 'fsub', 'fmul']:
            args = {'rd': 1, 'rs1': 2, 'rs2': 3}
            word = encode(op, args)
            dec_op, dec_args = decode(word)
            assert dec_op == op

    def test_fma(self):
        args = {'rd': 1, 'rs1': 2, 'rs2': 3, 'rs3': 4}
        word = encode('fma', args)
        dec_op, dec_args = decode(word)
        assert dec_op == 'fma'
        assert dec_args['rs3'] == 4

    def test_fpu_unary(self):
        for op in ['frcp', 'itof']:
            args = {'rd': 5, 'rs1': 10}
            word = encode(op, args)
            dec_op, dec_args = decode(word)
            assert dec_op == op

    def test_ftofx(self):
        args = {'rd': 5, 'pd': 2, 'rs1': 10}
        word = encode('ftofx.8.clamp', args)
        dec_op, dec_args = decode(word)
        assert dec_op == 'ftofx.8.clamp'
        assert dec_args['rd'] == 5

    def test_memory_load(self):
        for op in ['ldg', 'lds']:
            args = {'rd': 5, 'rs1': 10, 'imm': 100}
            word = encode(op, args)
            dec_op, dec_args = decode(word)
            assert dec_op == op
            assert dec_args['imm'] == 100

    def test_memory_store(self):
        for op in ['stg', 'sts']:
            args = {'rs1': 10, 'rs2': 5, 'imm': 100}
            word = encode(op, args)
            dec_op, dec_args = decode(word)
            assert dec_op == op
            assert dec_args['rs1'] == 10
            assert dec_args['rs2'] == 5
            assert dec_args['imm'] == 100

    def test_memory_negative_offset(self):
        args = {'rd': 5, 'rs1': 10, 'imm': -4}
        word = encode('ldg', args)
        dec_op, dec_args = decode(word)
        assert dec_args['imm'] == -4

    def test_lui(self):
        args = {'rd': 5, 'imm': 0x12345}
        word = encode('lui', args)
        dec_op, dec_args = decode(word)
        assert dec_op == 'lui'
        # imm is 20 bits in U format, 0x12345 fits
        assert dec_args['imm'] == 0x12345

    def test_sread(self):
        from encoding import SReg
        args = {'rd': 5, 'sreg': SReg.LANE_ID}
        word = encode('sread', args)
        dec_op, dec_args = decode(word)
        assert dec_op == 'sread'
        assert dec_args['rd'] == 5
        assert dec_args['sreg'] == SReg.LANE_ID

    def test_mov(self):
        args = {'rd': 5, 'rs1': 10}
        word = encode('mov', args)
        dec_op, dec_args = decode(word)
        assert dec_op == 'add'  # mov is pseudo for add rd, rs1, r0
        assert dec_args['rd'] == 5
        assert dec_args['rs1'] == 10

    def test_setp(self):
        for cmp in ['lt', 'le', 'eq', 'ne', 'ge', 'gt']:
            args = {'pd': 1, 'rs1': 5, 'rs2': 10}
            word = encode(f'setp.{cmp}.i32', args)
            dec_op, dec_args = decode(word)
            assert f'.{cmp}.' in dec_op
            assert dec_args['pd'] == 1

    def test_pred_logic(self):
        for op in ['pand', 'por', 'pxor']:
            args = {'pd': 1, 'ps1': 2, 'ps2': 3}
            word = encode(op, args)
            dec_op, dec_args = decode(word)
            assert dec_op == op
            assert dec_args['pd'] == 1

    def test_bra(self):
        args = {'pred': 2, 'target': 0x100}
        word = encode('bra', args)
        dec_op, dec_args = decode(word)
        assert dec_op == 'bra'
        assert dec_args['target'] == 0x100
        assert dec_args['pred'] == 2

    def test_ssy(self):
        args = {'target': 0x50}
        word = encode('ssy', args)
        dec_op, dec_args = decode(word)
        assert dec_op == 'ssy'
        assert dec_args['target'] == 0x50

    def test_halt_nop(self):
        for op in ['halt', 'nop']:
            word = encode(op, {})
            dec_op, _ = decode(word)
            assert dec_op == op

    def test_sync_instruction(self):
        word = encode('sync', {})
        dec_op, dec_args = decode(word)
        assert dec_op == 'sync'

    def test_predicated_instruction(self):
        args = {'rd': 5, 'rs1': 10, 'rs2': 15, 'pred': 3}
        word = encode('add', args)
        dec_op, dec_args = decode(word)
        assert dec_op == 'add'
        assert dec_args['pred'] == 3

    def test_predicated_p0(self):
        """pred=0 (p0) should not be confused with 'always execute'."""
        args = {'rd': 1, 'rs1': 2, 'rs2': 3, 'pred': 0}
        word = encode('add', args)
        dec_op, dec_args = decode(word)
        assert dec_args['pred'] == 0

    def test_negated_predicate(self):
        args = {'rd': 1, 'rs1': 2, 'imm': 42, 'pred': 0, 'pred_neg': True}
        word = encode('addi', args)
        dec_op, dec_args = decode(word)
        assert dec_args['pred'] == 0
        assert dec_args['pred_neg'] == True

    def test_unpredicated_no_pred_in_args(self):
        """Unpredicated instruction should not have 'pred' in decoded args."""
        args = {'rd': 1, 'rs1': 2, 'rs2': 3}
        word = encode('add', args)
        dec_op, dec_args = decode(word)
        assert 'pred' not in dec_args
        assert 'pred_neg' not in dec_args


class TestFmtCodes:
    def test_all_unique(self):
        fmts = [Fmt.R4, Fmt.I, Fmt.M, Fmt.M2, Fmt.U, Fmt.P, Fmt.J, Fmt.X]
        assert len(fmts) == len(set(fmts))

    def test_fit_in_3_bits(self):
        for f in [Fmt.R4, Fmt.I, Fmt.M, Fmt.M2, Fmt.U, Fmt.P, Fmt.J, Fmt.X]:
            assert 0 <= f <= 7


class TestAssemble:
    def test_simple_program(self):
        instrs = [
            ('add', {'rd': 1, 'rs1': 2, 'rs2': 3}),
            ('addi', {'rd': 4, 'rs1': 1, 'imm': 10}),
            ('halt', {}),
        ]
        words = assemble(instrs, {})
        assert len(words) == 3

    def test_branch_with_label(self):
        instrs = [
            ('ssy', {'label': 'end'}),
            ('add', {'rd': 1, 'rs1': 2, 'rs2': 3}),
            ('halt', {}),
        ]
        labels = {'end': 2}
        words = assemble(instrs, labels)
        op, args = decode(words[0])
        assert op == 'ssy'
        assert args['target'] == 2


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])

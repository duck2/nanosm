# Instruction encoding -- definitions and encode/decode
#
# Predicate field (4 bits): bits[2:0] = register, bit[3] = negate.
#   pred[2:0] = 0b111 means always execute (no predicate).
#   pred[2:0] = 0-6 maps to p0-p6.
#   pred[3] = 1 negates the predicate (execute where pN is false).
# Branch targets (J-format imm) are absolute addresses, not PC-relative.
# I-format imm is unsigned (12-bit). Use subi for subtraction (gives extra bit).
# FMT_P has 5-bit rs1/rs2 fields even though pred logic only needs 3 bits.
#   This wastes bits but simplifies decode (same field positions as setp).
# ftofx encodes pd in rs2 field and frac bits in rs3.
# Decoded instructions use 'target' for resolved addresses (not 'label').

from dataclasses import dataclass

# =============================================================================
# Bitfield Infrastructure
# =============================================================================

@dataclass(frozen=True)
class Field:
    """A bitfield within an instruction word."""
    name: str
    hi: int
    lo: int

    @property
    def width(self) -> int:
        return self.hi - self.lo + 1

    @property
    def mask(self) -> int:
        return (1 << self.width) - 1


class Format:
    """Instruction format - a collection of fields."""
    def __init__(self, name: str, *fields: Field):
        self.name = name
        self.fields = {f.name: f for f in fields}
        self._validate()

    def _validate(self):
        used = [False] * 32
        for f in self.fields.values():
            if f.hi > 31 or f.lo < 0 or f.hi < f.lo:
                raise ValueError(f"Invalid field range: {f}")
            for b in range(f.lo, f.hi + 1):
                if used[b]:
                    raise ValueError(f"Overlapping bit {b} in format {self.name}")
                used[b] = True

    def encode(self, **vals) -> int:
        word = 0
        for name, val in vals.items():
            if name not in self.fields:
                raise KeyError(f"Unknown field '{name}' in format {self.name}")
            f = self.fields[name]
            if val < 0:
                val = val & f.mask
            word |= (val & f.mask) << f.lo
        return word

    def decode(self, word: int) -> dict:
        return {name: (word >> f.lo) & f.mask for name, f in self.fields.items()}


# =============================================================================
# Instruction Formats (8 total)
# Common: fmt[31:29] + pred[28:25] = 7 bits, payload = 25 bits, op at bottom
# pred: bit 3 = negate, bits 2:0 = register (7=always, 0-6=p0-p6)
# =============================================================================

# R4: compute ops (ALU, shift, FPU) - 5-bit op
# | fmt[31:29] | pred[28:25] | rd[24:20] | rs1[19:15] | rs2[14:10] | rs3[9:5] | op[4:0] |
FMT_R4 = Format('R4',
    Field('fmt', 31, 29), Field('pred', 28, 25),
    Field('rd', 24, 20), Field('rs1', 19, 15), Field('rs2', 14, 10),
    Field('rs3', 9, 5), Field('op', 4, 0),
)

# I: immediate ALU/shift - 3-bit op, 12-bit imm (unsigned)
# | fmt[31:29] | pred[28:25] | rd[24:20] | rs1[19:15] | imm[14:3] | op[2:0] |
FMT_I = Format('I',
    Field('fmt', 31, 29), Field('pred', 28, 25),
    Field('rd', 24, 20), Field('rs1', 19, 15), Field('imm', 14, 3), Field('op', 2, 0),
)

# M: memory 1D (loads/stores) - 3-bit op, 12-bit signed imm
# | fmt[31:29] | pred[28:25] | rd_rs2[24:20] | rs1[19:15] | imm[14:3] | op[2:0] |
FMT_M = Format('M',
    Field('fmt', 31, 29), Field('pred', 28, 25),
    Field('rd_rs2', 24, 20), Field('rs1', 19, 15), Field('imm', 14, 3), Field('op', 2, 0),
)

# M2: memory 2D (ld2d/st2d) - 5-bit op, 4 register fields
# | fmt[31:29] | pred[28:25] | rd[24:20] | rx[19:15] | ry[14:10] | rdesc[9:5] | op[4:0] |
FMT_M2 = Format('M2',
    Field('fmt', 31, 29), Field('pred', 28, 25),
    Field('rd', 24, 20), Field('rx', 19, 15), Field('ry', 14, 10),
    Field('rdesc', 9, 5), Field('op', 4, 0),
)

# U: upper immediate (lui) - no op, just rd <- imm20 << 12
# | fmt[31:29] | pred[28:25] | rd[24:20] | imm[19:0] |
FMT_U = Format('U',
    Field('fmt', 31, 29), Field('pred', 28, 25),
    Field('rd', 24, 20), Field('imm', 19, 0),
)

# P: predicates (setp, pred logic) - 4-bit op
# | fmt[31:29] | pred[28:25] | pd[24:22] | rs1[21:17] | rs2[16:12] | cmp[11:9] | typ[8] | x[7:4] | op[3:0] |
FMT_P = Format('P',
    Field('fmt', 31, 29), Field('pred', 28, 25),
    Field('pd', 24, 22), Field('rs1', 21, 17), Field('rs2', 16, 12),
    Field('cmp', 11, 9), Field('typ', 8, 8), Field('x', 7, 4), Field('op', 3, 0),
)

# J: jumps/branches - 3-bit op, 22-bit imm
# | fmt[31:29] | pred[28:25] | imm[24:3] | op[2:0] |
FMT_J = Format('J',
    Field('fmt', 31, 29), Field('pred', 28, 25),
    Field('imm', 24, 3), Field('op', 2, 0),
)

# X: misc (sread, nop, halt, tile_commit, tile_wait) - 4-bit op
# | fmt[31:29] | pred[28:25] | r1[24:20] | r2[19:15] | r3[14:10] | x[9:4] | op[3:0] |
# Field usage varies by op:
#   sread:       rd=r1, sreg=r2 (only bottom 3 bits used)
#   tile_commit: rdesc=r1, rx=r2, ry=r3
#   nop/halt/tile_wait: unused
FMT_X = Format('X',
    Field('fmt', 31, 29), Field('pred', 28, 25),
    Field('r1', 24, 20), Field('r2', 19, 15), Field('r3', 14, 10),
    Field('x', 9, 4), Field('op', 3, 0),
)

# =============================================================================
# Format IDs (3 bits)
# =============================================================================

class Fmt:
    R4 = 0  # compute: ALU, shift, FPU
    I  = 1  # immediate ALU/shift
    M  = 2  # memory 1D
    M2 = 3  # memory 2D
    U  = 4  # upper immediate
    P  = 5  # predicates
    J  = 6  # jumps
    X  = 7  # misc: sread, nop, halt


# =============================================================================
# Op codes (per format, at bottom of instruction)
# =============================================================================

class Op:
    # R4 format (5 bits = 32 values)
    # ALU: 0-7
    ADD, SUB, AND, OR, XOR, MULS, MULU = 0, 1, 2, 3, 4, 5, 6
    # Shift: 8-15
    SHL, SHR, SRA = 8, 9, 10
    # FPU: 16-31
    FADD, FSUB, FMUL, FMA, FRCP, ITOF, FTOFX_CLAMP, FTOFX_REPEAT = 16, 17, 18, 19, 20, 21, 22, 23

    # I format (3 bits = 8 values)
    ADDI, SUBI, ANDI, ORI, XORI = 0, 1, 2, 3, 4
    SHLI, SHRI, SRAI = 5, 6, 7

    # M format (3 bits = 8 values)
    LDG, LDS, STG, STS = 0, 1, 2, 3

    # M2 format (5 bits = 32 values)
    LD2D, ST2D = 0, 1

    # P format (4 bits = 16 values)
    SETP = 0
    PAND, POR, PXOR = 1, 2, 3

    # J format (3 bits = 8 values)
    BRA, SSY, SYNC = 0, 1, 2

    # X format (4 bits = 16 values)
    SREAD = 0
    NOP, HALT = 1, 2
    TILE_COMMIT, TILE_WAIT = 3, 4


class SReg:
    """Special register IDs for sread instruction."""
    LANE_ID   = 0  # Lane ID within warp (0-7)
    WARP_ID   = 1  # Warp ID within block (future)
    BLOCK_X   = 2  # Block ID X (future)
    BLOCK_Y   = 3  # Block ID Y (future)
    BLOCK_Z   = 4  # Block ID Z (future)
    GRID_X    = 5  # Grid size X (future)
    GRID_Y    = 6  # Grid size Y (future)
    NUM_LANES = 7  # Lanes per warp (constant 8)


class Cmp:
    LT, LE, EQ, NE, GE, GT = 0, 1, 2, 3, 4, 5
    _TO_STR = {0: 'lt', 1: 'le', 2: 'eq', 3: 'ne', 4: 'ge', 5: 'gt'}
    _FROM_STR = {'lt': 0, 'le': 1, 'eq': 2, 'ne': 3, 'ge': 4, 'gt': 5}


class SType:
    I32, U32 = 0, 1
    _TO_STR = {0: 'i32', 1: 'u32'}
    _FROM_STR = {'i32': 0, 's32': 0, 'u32': 1}


# =============================================================================
# Instruction Table
# =============================================================================

ENCODE_TABLE = {
    # R4: ALU
    'add':  (FMT_R4, {'fmt': Fmt.R4, 'op': Op.ADD}),
    'sub':  (FMT_R4, {'fmt': Fmt.R4, 'op': Op.SUB}),
    'and':  (FMT_R4, {'fmt': Fmt.R4, 'op': Op.AND}),
    'or':   (FMT_R4, {'fmt': Fmt.R4, 'op': Op.OR}),
    'xor':  (FMT_R4, {'fmt': Fmt.R4, 'op': Op.XOR}),
    'muls': (FMT_R4, {'fmt': Fmt.R4, 'op': Op.MULS}),
    'mulu': (FMT_R4, {'fmt': Fmt.R4, 'op': Op.MULU}),
    # R4: Shift
    'shl':  (FMT_R4, {'fmt': Fmt.R4, 'op': Op.SHL}),
    'shr':  (FMT_R4, {'fmt': Fmt.R4, 'op': Op.SHR}),
    'sra':  (FMT_R4, {'fmt': Fmt.R4, 'op': Op.SRA}),
    # R4: FPU
    'fadd': (FMT_R4, {'fmt': Fmt.R4, 'op': Op.FADD}),
    'fsub': (FMT_R4, {'fmt': Fmt.R4, 'op': Op.FSUB}),
    'fmul': (FMT_R4, {'fmt': Fmt.R4, 'op': Op.FMUL}),
    'fma':  (FMT_R4, {'fmt': Fmt.R4, 'op': Op.FMA}),
    'frcp': (FMT_R4, {'fmt': Fmt.R4, 'op': Op.FRCP, 'rs2': 0, 'rs3': 0}),
    'itof': (FMT_R4, {'fmt': Fmt.R4, 'op': Op.ITOF, 'rs2': 0, 'rs3': 0}),
    # R4: MOV pseudo (add rd, rs1, r0)
    'mov':  (FMT_R4, {'fmt': Fmt.R4, 'op': Op.ADD, 'rs2': 0, 'rs3': 0}),

    # I: ALU immediate
    'addi': (FMT_I, {'fmt': Fmt.I, 'op': Op.ADDI}),
    'subi': (FMT_I, {'fmt': Fmt.I, 'op': Op.SUBI}),
    'andi': (FMT_I, {'fmt': Fmt.I, 'op': Op.ANDI}),
    'ori':  (FMT_I, {'fmt': Fmt.I, 'op': Op.ORI}),
    'xori': (FMT_I, {'fmt': Fmt.I, 'op': Op.XORI}),
    # I: Shift immediate
    'shli': (FMT_I, {'fmt': Fmt.I, 'op': Op.SHLI}),
    'shri': (FMT_I, {'fmt': Fmt.I, 'op': Op.SHRI}),
    'srai': (FMT_I, {'fmt': Fmt.I, 'op': Op.SRAI}),

    # M: Memory 1D
    'ldg': (FMT_M, {'fmt': Fmt.M, 'op': Op.LDG}),
    'lds': (FMT_M, {'fmt': Fmt.M, 'op': Op.LDS}),
    'stg': (FMT_M, {'fmt': Fmt.M, 'op': Op.STG}),
    'sts': (FMT_M, {'fmt': Fmt.M, 'op': Op.STS}),

    # M2: Memory 2D
    'ld2d': (FMT_M2, {'fmt': Fmt.M2, 'op': Op.LD2D}),
    'st2d': (FMT_M2, {'fmt': Fmt.M2, 'op': Op.ST2D}),

    # U: Upper immediate
    'lui': (FMT_U, {'fmt': Fmt.U}),

    # P: Pred logic (setp handled specially)
    'pand': (FMT_P, {'fmt': Fmt.P, 'op': Op.PAND, 'cmp': 0, 'typ': 0, 'x': 0}),
    'por':  (FMT_P, {'fmt': Fmt.P, 'op': Op.POR, 'cmp': 0, 'typ': 0, 'x': 0}),
    'pxor': (FMT_P, {'fmt': Fmt.P, 'op': Op.PXOR, 'cmp': 0, 'typ': 0, 'x': 0}),

    # J: Jumps
    'bra':     (FMT_J, {'fmt': Fmt.J, 'op': Op.BRA}),
    'bra.uni': (FMT_J, {'fmt': Fmt.J, 'op': Op.BRA}),
    'ssy':     (FMT_J, {'fmt': Fmt.J, 'op': Op.SSY}),
    'sync':    (FMT_J, {'fmt': Fmt.J, 'op': Op.SYNC, 'imm': 0}),

    # X: Misc
    'sread':       (FMT_X, {'fmt': Fmt.X, 'op': Op.SREAD, 'r3': 0, 'x': 0}),
    'nop':         (FMT_X, {'fmt': Fmt.X, 'op': Op.NOP, 'r1': 0, 'r2': 0, 'r3': 0, 'x': 0}),
    'halt':        (FMT_X, {'fmt': Fmt.X, 'op': Op.HALT, 'r1': 0, 'r2': 0, 'r3': 0, 'x': 0}),
    'tile_commit': (FMT_X, {'fmt': Fmt.X, 'op': Op.TILE_COMMIT, 'x': 0}),
    'tile_wait':   (FMT_X, {'fmt': Fmt.X, 'op': Op.TILE_WAIT, 'r1': 0, 'r2': 0, 'r3': 0, 'x': 0}),
}

# =============================================================================
# Encode
# =============================================================================

def _validate_reg(name: str, val: int, max_val: int = 31):
    """Validate register number is in range."""
    if val < 0 or val > max_val:
        raise ValueError(f"{name}={val} out of range (0-{max_val})")


def encode(op: str, args: dict) -> int:
    """Encode instruction to 32-bit word."""
    # Validate registers
    for k in ('rd', 'rs1', 'rs2', 'rs3'):
        if k in args:
            _validate_reg(k, args[k], 31)
    for k in ('pd', 'ps1', 'ps2'):
        if k in args:
            _validate_reg(k, args[k], 6)
    if 'pred' in args:
        _validate_reg('pred', args['pred'], 6)

    # Validate shift immediate
    if op in ('shli', 'shri', 'srai') and 'imm' in args:
        if args['imm'] < 0 or args['imm'] > 31:
            raise ValueError(f"Shift amount {args['imm']} out of range (0-31)")

    # Validate I-format immediate is unsigned
    if op in ('addi', 'subi', 'andi', 'ori', 'xori') and 'imm' in args:
        if args['imm'] < 0 or args['imm'] > 0xFFF:
            raise ValueError(f"I-format immediate {args['imm']} out of range (0-4095)")

    # Validate lui immediate fits in 20 bits
    if op == 'lui' and 'imm' in args:
        if args['imm'] < 0 or args['imm'] > 0xFFFFF:
            raise ValueError(f"lui immediate {args['imm']} out of range (0-0xFFFFF)")

    # Handle predication: 4-bit field
    # bit 3 = negate, bits 2:0 = register (0b111=always, 0-6=p0-p6)
    pred_val = 0b111  # default: always execute
    if 'pred' in args:
        pred_val = args['pred']  # p0=0, p1=1, ..., p6=6
    if args.get('pred_neg', False):
        pred_val |= 0b1000  # set negate bit

    return _encode_op(op, args, pred_val)


def _encode_op(op: str, args: dict, pred: int) -> int:
    # ftofx special handling
    if op.startswith('ftofx.'):
        return _encode_ftofx(op, args, pred)
    # setp special handling
    if op.startswith('setp.'):
        return _encode_setp(op, args, pred)

    if op not in ENCODE_TABLE:
        raise ValueError(f"Unknown instruction: {op}")

    fmt, defaults = ENCODE_TABLE[op]
    vals = {}

    # Add pred
    if 'pred' in fmt.fields:
        vals['pred'] = pred

    # Add defaults
    for k, v in defaults.items():
        if k in fmt.fields:
            vals[k] = v

    # Map args to fields
    for k, v in args.items():
        if k in ('pred', 'pred_neg'):
            continue  # already handled
        elif k == 'imm':
            if 'imm' in fmt.fields:
                vals['imm'] = v & fmt.fields['imm'].mask
        elif k == 'target':
            if 'imm' in fmt.fields:
                vals['imm'] = v & fmt.fields['imm'].mask
        elif k == 'rd':
            if fmt == FMT_M:
                vals['rd_rs2'] = v
            elif fmt == FMT_X:
                vals['r1'] = v  # X format: rd -> r1 (for sread, tile_commit)
            else:
                vals['rd'] = v
        elif k == 'rs2':
            if fmt == FMT_M:
                vals['rd_rs2'] = v
            else:
                vals['rs2'] = v
        elif k == 'rs':  # M2 store source
            vals['rd'] = v
        elif k == 'sreg':  # X format: sreg -> r2
            vals['r2'] = v
        elif k == 'rx':  # X format tile_commit: rx -> r2
            if fmt == FMT_X:
                vals['r2'] = v
            else:
                vals['rx'] = v
        elif k == 'ry':  # X format tile_commit: ry -> r3
            if fmt == FMT_X:
                vals['r3'] = v
            else:
                vals['ry'] = v
        elif k == 'ps1':
            vals['rs1'] = v
        elif k == 'ps2':
            vals['rs2'] = v
        elif k in fmt.fields:
            vals[k] = v

    return fmt.encode(**vals)


def _encode_ftofx(op: str, args: dict, pred: int) -> int:
    parts = op.split('.')
    if parts[2] == 'repeat':
        opcode = Op.FTOFX_REPEAT
        frac = 15
    else:
        opcode = Op.FTOFX_CLAMP
        frac = int(parts[1])
    return FMT_R4.encode(
        fmt=Fmt.R4, op=opcode, pred=pred,
        rd=args['rd'], rs1=args['rs1'], rs2=args['pd'], rs3=frac
    )


def _encode_setp(op: str, args: dict, pred: int) -> int:
    parts = op.split('.')
    cmp_val = Cmp._FROM_STR[parts[1]]
    typ_val = SType._FROM_STR.get(parts[2], 0)
    return FMT_P.encode(
        fmt=Fmt.P, op=Op.SETP, pred=pred,
        pd=args['pd'], rs1=args['rs1'], rs2=args['rs2'],
        cmp=cmp_val, typ=typ_val, x=0
    )


# =============================================================================
# Decode
# =============================================================================

def decode(word: int) -> tuple[str, dict]:
    """Decode 32-bit word to (op, args)."""
    fmt = (word >> 29) & 0x7
    pred_enc = (word >> 25) & 0xF  # 4 bits

    op_str, args = _decode_op(word, fmt)

    # Decode predicate: bit 3 = negate, bits 2:0 = register (0b111=always, 0-6=p0-p6)
    pred_reg = pred_enc & 0b111
    pred_neg = bool(pred_enc & 0b1000)
    if pred_reg != 0b111:
        args['pred'] = pred_reg
    if pred_neg:
        args['pred_neg'] = True

    return op_str, args


def _decode_op(word: int, fmt: int) -> tuple[str, dict]:
    if fmt == Fmt.R4:
        return _decode_r4(word)

    if fmt == Fmt.I:
        return _decode_i(word)

    if fmt == Fmt.M:
        return _decode_m(word)

    if fmt == Fmt.M2:
        return _decode_m2(word)

    if fmt == Fmt.U:
        f = FMT_U.decode(word)
        return 'lui', {'rd': f['rd'], 'imm': f['imm']}

    if fmt == Fmt.P:
        return _decode_p(word)

    if fmt == Fmt.J:
        return _decode_j(word)

    if fmt == Fmt.X:
        return _decode_x(word)

    raise ValueError(f"Unknown format: {fmt}")


def _decode_r4(word: int) -> tuple[str, dict]:
    f = FMT_R4.decode(word)
    op = f['op']

    # ALU
    alu_ops = {Op.ADD: 'add', Op.SUB: 'sub', Op.AND: 'and', Op.OR: 'or',
               Op.XOR: 'xor', Op.MULS: 'muls', Op.MULU: 'mulu'}
    if op in alu_ops:
        return alu_ops[op], {'rd': f['rd'], 'rs1': f['rs1'], 'rs2': f['rs2']}

    # Shift
    shift_ops = {Op.SHL: 'shl', Op.SHR: 'shr', Op.SRA: 'sra'}
    if op in shift_ops:
        return shift_ops[op], {'rd': f['rd'], 'rs1': f['rs1'], 'rs2': f['rs2']}

    # FPU
    if op == Op.FADD:
        return 'fadd', {'rd': f['rd'], 'rs1': f['rs1'], 'rs2': f['rs2']}
    if op == Op.FSUB:
        return 'fsub', {'rd': f['rd'], 'rs1': f['rs1'], 'rs2': f['rs2']}
    if op == Op.FMUL:
        return 'fmul', {'rd': f['rd'], 'rs1': f['rs1'], 'rs2': f['rs2']}
    if op == Op.FMA:
        return 'fma', {'rd': f['rd'], 'rs1': f['rs1'], 'rs2': f['rs2'], 'rs3': f['rs3']}
    if op == Op.FRCP:
        return 'frcp', {'rd': f['rd'], 'rs1': f['rs1']}
    if op == Op.ITOF:
        return 'itof', {'rd': f['rd'], 'rs1': f['rs1']}
    if op == Op.FTOFX_CLAMP:
        frac = f['rs3']
        return f'ftofx.{frac}.clamp', {'rd': f['rd'], 'pd': f['rs2'], 'rs1': f['rs1']}
    if op == Op.FTOFX_REPEAT:
        return 'ftofx.15.repeat', {'rd': f['rd'], 'pd': f['rs2'], 'rs1': f['rs1']}

    raise ValueError(f"Unknown R4 op: {op}")


def _decode_i(word: int) -> tuple[str, dict]:
    f = FMT_I.decode(word)
    op = f['op']

    alu_ops = {Op.ADDI: 'addi', Op.SUBI: 'subi', Op.ANDI: 'andi', Op.ORI: 'ori', Op.XORI: 'xori'}
    if op in alu_ops:
        return alu_ops[op], {'rd': f['rd'], 'rs1': f['rs1'], 'imm': f['imm']}

    shift_ops = {Op.SHLI: 'shli', Op.SHRI: 'shri', Op.SRAI: 'srai'}
    if op in shift_ops:
        return shift_ops[op], {'rd': f['rd'], 'rs1': f['rs1'], 'imm': f['imm'] & 0x1F}

    raise ValueError(f"Unknown I op: {op}")


def _decode_m(word: int) -> tuple[str, dict]:
    f = FMT_M.decode(word)
    op = f['op']
    ops = {Op.LDG: 'ldg', Op.LDS: 'lds', Op.STG: 'stg', Op.STS: 'sts'}
    op_str = ops[op]
    if op_str in ('ldg', 'lds'):
        return op_str, {'rd': f['rd_rs2'], 'rs1': f['rs1'], 'imm': _sext12(f['imm'])}
    return op_str, {'rs1': f['rs1'], 'rs2': f['rd_rs2'], 'imm': _sext12(f['imm'])}


def _decode_m2(word: int) -> tuple[str, dict]:
    f = FMT_M2.decode(word)
    op = f['op']
    if op == Op.LD2D:
        return 'ld2d', {'rd': f['rd'], 'rx': f['rx'], 'ry': f['ry'], 'rdesc': f['rdesc']}
    if op == Op.ST2D:
        return 'st2d', {'rx': f['rx'], 'ry': f['ry'], 'rdesc': f['rdesc'], 'rs': f['rd']}
    raise ValueError(f"Unknown M2 op: {op}")


def _decode_j(word: int) -> tuple[str, dict]:
    f = FMT_J.decode(word)
    op = f['op']
    ops = {Op.BRA: 'bra', Op.SSY: 'ssy', Op.SYNC: 'sync'}
    op_str = ops[op]
    if op_str == 'sync':
        return 'sync', {}
    return op_str, {'target': f['imm']}


def _decode_p(word: int) -> tuple[str, dict]:
    f = FMT_P.decode(word)
    op = f['op']

    if op == Op.SETP:
        cmp_val = f['cmp']
        typ_val = f['typ']
        cmp_str = Cmp._TO_STR[cmp_val]
        typ_str = SType._TO_STR[typ_val]
        return f"setp.{cmp_str}.{typ_str}", {
            'pd': f['pd'], 'rs1': f['rs1'], 'rs2': f['rs2'],
            'cmp': cmp_val, 'type': typ_val
        }

    pred_ops = {Op.PAND: 'pand', Op.POR: 'por', Op.PXOR: 'pxor'}
    if op in pred_ops:
        if f['rs1'] > 6 or f['rs2'] > 6:
            raise ValueError(f"Invalid predicate register in {pred_ops[op]}")
        return pred_ops[op], {'pd': f['pd'], 'ps1': f['rs1'], 'ps2': f['rs2']}

    raise ValueError(f"Unknown P op: {op}")


def _decode_x(word: int) -> tuple[str, dict]:
    f = FMT_X.decode(word)
    op = f['op']
    if op == Op.SREAD:
        return 'sread', {'rd': f['r1'], 'sreg': f['r2']}
    if op == Op.NOP:
        return 'nop', {}
    if op == Op.HALT:
        return 'halt', {}
    if op == Op.TILE_COMMIT:
        return 'tile_commit', {'rd': f['r1'], 'rx': f['r2'], 'ry': f['r3']}
    if op == Op.TILE_WAIT:
        return 'tile_wait', {}
    raise ValueError(f"Unknown X op: {op}")


def _sext12(val: int) -> int:
    """Sign-extend 12-bit value."""
    return val - 0x1000 if val & 0x800 else val


# =============================================================================
# Assembly Helpers
# =============================================================================

def assemble(instrs: list, labels: dict) -> list[int]:
    """Assemble (op, args) list with label resolution."""
    words = []
    for op, args in instrs:
        resolved = args.copy()
        if 'label' in resolved:
            resolved['target'] = labels[resolved['label']]
            del resolved['label']
        words.append(encode(op, resolved))
    return words


def assemble_to_hex(instrs: list, labels: dict) -> str:
    """Assemble to hex string for $readmemh."""
    return '\n'.join(f'{w:08x}' for w in assemble(instrs, labels))

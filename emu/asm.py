"""Text ASM parser for the GPU emulator."""

import re
from typing import List, Tuple, Dict, Optional

from encoding import Cmp, SType

# Token patterns
REG = re.compile(r'^[rx](\d+)$', re.I)
PRED = re.compile(r'^p(\d+)$', re.I)
IMM = re.compile(r'^(-?0x[0-9a-f]+|-?\d+)$', re.I)
MEM = re.compile(r'^\[([rx]\d+)(?:([+-])(\d+|0x[0-9a-f]+))?\]$', re.I)
LABEL_DEF = re.compile(r'^(\w+):(.*)$')
PRED_PREFIX = re.compile(r'^@(!)?p(\d+)$', re.I)

def parse_reg(s: str) -> int:
    """Parse register name (r0-r31 or x0-x31)."""
    m = REG.match(s.strip())
    if not m:
        raise ValueError(f"Invalid register: {s}")
    return int(m.group(1))

def parse_pred(s: str) -> int:
    """Parse predicate register name (p0-p6)."""
    m = PRED.match(s.strip())
    if not m:
        raise ValueError(f"Invalid predicate register: {s}")
    val = int(m.group(1))
    assert 0 <= val <= 6, f"Predicate register {s} out of range (p0-p6)"
    return val

def parse_imm(s: str) -> int:
    """Parse immediate value (decimal or hex)."""
    s = s.strip()
    m = IMM.match(s)
    if not m:
        raise ValueError(f"Invalid immediate: {s}")
    val = m.group(1)
    if '0x' in val.lower():
        return int(val, 16)
    return int(val)

def parse_mem_operand(s: str) -> Tuple[int, int]:
    """Parse [rs1+imm] or [rs1] memory operand. Returns (rs1, imm)."""
    m = MEM.match(s.strip())
    if not m:
        raise ValueError(f"Invalid memory operand: {s}")
    rs1 = parse_reg(m.group(1))
    if m.group(2) is None:
        return rs1, 0
    imm = parse_imm(m.group(3))
    return rs1, imm if m.group(2) == '+' else -imm

def parse_line(line: str) -> Tuple[Optional[str], Optional[dict]]:
    """Parse a single assembly line. Returns (opcode, args_dict) or (None, None) for empty/comment."""
    line = re.split(r'[;#]', line)[0].strip()
    if not line:
        return None, None

    parts = re.split(r'[,\s]+', line)
    parts = [p for p in parts if p]

    # Handle predication prefix @pN or @!pN
    pred, pred_neg = None, False
    m = PRED_PREFIX.match(parts[0])
    if m:
        pred_neg = m.group(1) is not None
        pred = int(m.group(2))
        assert 0 <= pred <= 6, f"Predicate p{pred} out of range"
        parts = parts[1:]
        if not parts:
            return None, None

    op = parts[0].lower()
    args = {}
    if pred is not None:
        args['pred'] = pred
        if pred_neg:
            args['pred_neg'] = True

    # Instruction patterns
    r_type = {'add', 'sub', 'muls', 'mulu', 'and', 'or', 'xor', 'shl', 'shr', 'sra', 'fadd', 'fsub', 'fmul'}
    i_type = {'addi', 'subi', 'andi', 'ori', 'xori', 'shli', 'shri', 'srai'}
    unary_fpu = {'frcp', 'itof'}
    pred_logic = {'pand', 'por', 'pxor'}
    nullary = {'halt', 'nop', 'sync'}

    if op in r_type:
        args['rd'], args['rs1'], args['rs2'] = parse_reg(parts[1]), parse_reg(parts[2]), parse_reg(parts[3])

    elif op == 'fma':
        args['rd'], args['rs1'] = parse_reg(parts[1]), parse_reg(parts[2])
        args['rs2'], args['rs3'] = parse_reg(parts[3]), parse_reg(parts[4])

    elif op in unary_fpu:
        args['rd'], args['rs1'] = parse_reg(parts[1]), parse_reg(parts[2])

    elif op in i_type:
        args['rd'], args['rs1'], args['imm'] = parse_reg(parts[1]), parse_reg(parts[2]), parse_imm(parts[3])

    elif op == 'ldg':
        args['rd'] = parse_reg(parts[1])
        args['rs1'], args['imm'] = parse_mem_operand(parts[2])

    elif op == 'stg':
        args['rs1'], args['imm'] = parse_mem_operand(parts[1])
        args['rs2'] = parse_reg(parts[2])

    elif op == 'lds':
        args['rd'] = parse_reg(parts[1])
        args['rs1'], args['imm'] = parse_mem_operand(parts[2])

    elif op == 'sts':
        args['rs1'], args['imm'] = parse_mem_operand(parts[1])
        args['rs2'] = parse_reg(parts[2])

    elif op == 'lui':
        args['rd'], args['imm'] = parse_reg(parts[1]), parse_imm(parts[2])

    elif op == 'mov':
        args['rd'], args['rs1'] = parse_reg(parts[1]), parse_reg(parts[2])

    elif op == 'sread':
        args['rd'] = parse_reg(parts[1])
        sreg_map = {'lane_id': 0, 'warp_id': 1, 'block_x': 2, 'block_y': 3,
                    'block_z': 4, 'grid_x': 5, 'grid_y': 6, 'num_lanes': 7}
        sreg_name = parts[2].lower()
        args['sreg'] = sreg_map[sreg_name] if sreg_name in sreg_map else parse_imm(parts[2])

    elif op.startswith('setp.'):
        setp_parts = op.split('.')
        assert len(setp_parts) >= 3, f"Invalid setp format: {op}, expected setp.cmp.type"
        args['cmp'] = Cmp._FROM_STR[setp_parts[1]]
        args['type'] = SType._FROM_STR[setp_parts[2]]
        args['pd'], args['rs1'], args['rs2'] = parse_pred(parts[1]), parse_reg(parts[2]), parse_reg(parts[3])

    elif op == 'bra.uni':
        args['label'] = parts[1]

    elif op == 'bra':
        assert 'pred' in args, "bra requires predicate prefix @pN"
        args['label'] = parts[1]

    elif op == 'ssy':
        args['label'] = parts[1]

    elif op in pred_logic:
        args['pd'], args['ps1'], args['ps2'] = parse_pred(parts[1]), parse_pred(parts[2]), parse_pred(parts[3])

    elif op in nullary:
        pass

    else:
        raise ValueError(f"Unknown instruction: {op}")

    return op, args

def parse_program(text: str) -> Tuple[List[Tuple[str, dict]], Dict[str, int]]:
    """Parse complete assembly program. Returns (instructions, labels)."""
    instructions = []
    labels = {}

    for line in text.strip().split('\n'):
        line = re.split(r'[;#]', line)[0].strip()
        if not line:
            continue
        # Check for label definition
        m = LABEL_DEF.match(line)
        if m and ' ' not in m.group(1):
            labels[m.group(1)] = len(instructions)
            line = m.group(2).strip()
            if not line:
                continue
        op, args = parse_line(line)
        if op:
            instructions.append((op, args))

    return instructions, labels

def load_file(path: str) -> Tuple[List[Tuple[str, dict]], Dict[str, int]]:
    """Load and parse assembly file."""
    with open(path, 'r') as f:
        return parse_program(f.read())

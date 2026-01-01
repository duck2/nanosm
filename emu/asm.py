"""Text ASM parser for the GPU emulator."""

import re
from typing import List, Tuple, Dict

def parse_reg(s: str) -> int:
    """Parse register name (r0-r31 or x0-x31)."""
    s = s.strip().lower()
    if s.startswith('r') or s.startswith('x'):
        return int(s[1:])
    raise ValueError(f"Invalid register: {s}")

def parse_pred(s: str) -> int:
    """Parse predicate register name (p0-p7)."""
    s = s.strip().lower()
    if s.startswith('p'):
        return int(s[1:])
    raise ValueError(f"Invalid predicate register: {s}")

def parse_imm(s: str) -> int:
    """Parse immediate value (decimal or hex)."""
    s = s.strip()
    if s.startswith('0x') or s.startswith('0X'):
        return int(s, 16)
    if s.startswith('-0x') or s.startswith('-0X'):
        return -int(s[1:], 16)
    return int(s)

def parse_mem_operand(s: str) -> Tuple[int, int]:
    """Parse [rs1+imm] or [rs1] memory operand. Returns (rs1, imm)."""
    s = s.strip()
    if not s.startswith('[') or not s.endswith(']'):
        raise ValueError(f"Invalid memory operand: {s}")
    inner = s[1:-1].strip()
    if '+' in inner:
        parts = inner.split('+')
        return parse_reg(parts[0]), parse_imm(parts[1])
    elif '-' in inner and not inner.startswith('-'):
        parts = inner.split('-')
        return parse_reg(parts[0]), -parse_imm(parts[1])
    else:
        return parse_reg(inner), 0

def parse_line(line: str) -> Tuple[str, dict]:
    """Parse a single assembly line. Returns (opcode, args_dict) or (None, None) for empty/comment."""
    line = line.split(';')[0].split('#')[0].strip()
    if not line:
        return None, None

    parts = re.split(r'[,\s]+', line)
    parts = [p for p in parts if p]

    # Handle predication prefix @pN
    pred = None
    if parts[0].startswith('@'):
        pred = parse_pred(parts[0][1:])
        parts = parts[1:]
        if not parts:
            return None, None

    op = parts[0].lower()
    args = {}
    if pred is not None:
        args['pred'] = pred

    # R-type: op rd, rs1, rs2
    if op in ['add', 'sub', 'muls', 'mulu', 'and', 'or', 'xor', 'shl', 'shr', 'sra', 'fadd', 'fsub', 'fmul']:
        args['rd'] = parse_reg(parts[1])
        args['rs1'] = parse_reg(parts[2])
        args['rs2'] = parse_reg(parts[3])

    # R4-type: op rd, rs1, rs2, rs3 (fma)
    elif op == 'fma':
        args['rd'] = parse_reg(parts[1])
        args['rs1'] = parse_reg(parts[2])
        args['rs2'] = parse_reg(parts[3])
        args['rs3'] = parse_reg(parts[4])

    # Unary FPU: frcp, itof
    elif op in ['frcp', 'itof']:
        args['rd'] = parse_reg(parts[1])
        args['rs1'] = parse_reg(parts[2])

    # ftofx.N.clamp rd, pd, rs1 OR ftofx.15.repeat rd, pd, rs1
    elif op.startswith('ftofx.'):
        args['rd'] = parse_reg(parts[1])
        args['pd'] = parse_pred(parts[2])
        args['rs1'] = parse_reg(parts[3])

    # I-type: op rd, rs1, imm
    elif op in ['addi', 'andi', 'ori', 'xori', 'shli', 'shri', 'srai']:
        args['rd'] = parse_reg(parts[1])
        args['rs1'] = parse_reg(parts[2])
        args['imm'] = parse_imm(parts[3])

    # Global load: ldg rd, [rs1+imm]
    elif op == 'ldg':
        args['rd'] = parse_reg(parts[1])
        args['rs1'], args['imm'] = parse_mem_operand(parts[2])

    # Global store: stg [rs1+imm], rs2
    elif op == 'stg':
        args['rs1'], args['imm'] = parse_mem_operand(parts[1])
        args['rs2'] = parse_reg(parts[2])

    # 2D load: ld2d rd, rx, ry, rdesc
    elif op == 'ld2d':
        args['rd'] = parse_reg(parts[1])
        args['rx'] = parse_reg(parts[2])
        args['ry'] = parse_reg(parts[3])
        args['rdesc'] = parse_reg(parts[4])

    # 2D store: st2d rx, ry, rdesc, rs
    elif op == 'st2d':
        args['rx'] = parse_reg(parts[1])
        args['ry'] = parse_reg(parts[2])
        args['rdesc'] = parse_reg(parts[3])
        args['rs'] = parse_reg(parts[4])

    # Scratchpad load: lds rd, [rs1+imm]
    elif op == 'lds':
        args['rd'] = parse_reg(parts[1])
        args['rs1'], args['imm'] = parse_mem_operand(parts[2])

    # Scratchpad store: sts [rs1+imm], rs2
    elif op == 'sts':
        args['rs1'], args['imm'] = parse_mem_operand(parts[1])
        args['rs2'] = parse_reg(parts[2])

    # LUI: lui rd, imm
    elif op == 'lui':
        args['rd'] = parse_reg(parts[1])
        args['imm'] = parse_imm(parts[2])

    # MOV: mov rd, rs1
    elif op == 'mov':
        args['rd'] = parse_reg(parts[1])
        args['rs1'] = parse_reg(parts[2])

    # LID: lid rd
    elif op == 'lid':
        args['rd'] = parse_reg(parts[1])

    # Set predicate: setp.cmp.type pd, rs1, rs2 (e.g. setp.lt.i32, setp.eq.u32)
    elif op.startswith('setp.'):
        setp_parts = op.split('.')
        if len(setp_parts) < 3:
            raise ValueError(f"Invalid setp format: {op}, expected setp.cmp.type")
        args['cmp'] = setp_parts[1]
        args['type'] = setp_parts[2]
        args['pd'] = parse_pred(parts[1])
        args['rs1'] = parse_reg(parts[2])
        args['rs2'] = parse_reg(parts[3])

    # Unconditional branch: bra.uni label (no predicate)
    # Predicated uniform branch: @pN bra.uni label
    elif op == 'bra.uni':
        args['label'] = parts[1]

    # Predicated divergent branch: @pN bra label
    elif op == 'bra':
        if 'pred' not in args:
            raise ValueError(f"bra requires predicate prefix @pN")
        args['label'] = parts[1]

    # SSY: ssy label (set synchronization point)
    elif op == 'ssy':
        args['label'] = parts[1]

    # Tile commit: tile_commit rx, ry, rd
    elif op == 'tile_commit':
        args['rx'] = parse_reg(parts[1])
        args['ry'] = parse_reg(parts[2])
        args['rd'] = parse_reg(parts[3])

    # Tile wait: tile_wait
    elif op == 'tile_wait':
        pass

    # Predicate logic: pand/por/pxor pd, ps1, ps2
    elif op in ['pand', 'por', 'pxor']:
        args['pd'] = parse_pred(parts[1])
        args['ps1'] = parse_pred(parts[2])
        args['ps2'] = parse_pred(parts[3])

    # halt, nop (possibly with .s suffix)
    elif op in ['halt', 'nop'] or op in ['halt.s', 'nop.s']:
        pass

    else:
        # Check if it's an instruction with .s suffix we haven't handled
        base_op = op[:-2] if op.endswith('.s') else op
        if base_op in ['add', 'sub', 'muls', 'mulu', 'and', 'or', 'xor', 'shl', 'shr', 'sra', 'fadd', 'fsub', 'fmul']:
            args['rd'] = parse_reg(parts[1])
            args['rs1'] = parse_reg(parts[2])
            args['rs2'] = parse_reg(parts[3])
        elif base_op == 'fma':
            args['rd'] = parse_reg(parts[1])
            args['rs1'] = parse_reg(parts[2])
            args['rs2'] = parse_reg(parts[3])
            args['rs3'] = parse_reg(parts[4])
        elif base_op in ['addi', 'andi', 'ori', 'xori', 'shli', 'shri', 'srai']:
            args['rd'] = parse_reg(parts[1])
            args['rs1'] = parse_reg(parts[2])
            args['imm'] = parse_imm(parts[3])
        elif base_op == 'mov':
            args['rd'] = parse_reg(parts[1])
            args['rs1'] = parse_reg(parts[2])
        elif base_op == 'lid':
            args['rd'] = parse_reg(parts[1])
        elif base_op == 'lui':
            args['rd'] = parse_reg(parts[1])
            args['imm'] = parse_imm(parts[2])
        else:
            raise ValueError(f"Unknown instruction: {op}")

    return op, args

def parse_program(text: str) -> Tuple[List[Tuple[str, dict]], Dict[str, int]]:
    """Parse complete assembly program. Returns (instructions, labels)."""
    lines = text.strip().split('\n')
    instructions = []
    labels = {}

    for line in lines:
        # Remove comments first
        line = line.split(';')[0].split('#')[0].strip()
        if not line:
            continue
        # Check for label (must be at start of line, before any instruction)
        if ':' in line:
            label_part, rest = line.split(':', 1)
            # Only treat as label if label_part looks like an identifier (no spaces)
            if ' ' not in label_part.strip() and label_part.strip():
                labels[label_part.strip()] = len(instructions)
                line = rest.strip()
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

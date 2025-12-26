"""Text ASM parser for the GPU emulator."""

import re
from typing import List, Tuple, Dict

def parse_reg(s: str) -> int:
    """Parse register name (r0-r31 or x0-x31)."""
    s = s.strip().lower()
    if s.startswith('r') or s.startswith('x'):
        return int(s[1:])
    raise ValueError(f"Invalid register: {s}")

def parse_imm(s: str) -> int:
    """Parse immediate value (decimal or hex)."""
    s = s.strip()
    if s.startswith('0x') or s.startswith('0X'):
        return int(s, 16)
    if s.startswith('-0x') or s.startswith('-0X'):
        return -int(s[1:], 16)
    return int(s)

def parse_line(line: str) -> Tuple[str, dict]:
    """Parse a single assembly line. Returns (opcode, args_dict) or (None, None) for empty/comment."""
    line = line.split(';')[0].split('#')[0].strip()
    if not line:
        return None, None

    parts = re.split(r'[,\s]+', line)
    parts = [p for p in parts if p]
    op = parts[0].lower()
    args = {}

    # R-type: op rd, rs1, rs2
    if op in ['add', 'sub', 'muls', 'mulu', 'and', 'or', 'xor', 'slt', 'sltu', 'min', 'max', 'shl', 'shr', 'sra', 'fadd', 'fsub', 'fmul']:
        args['rd'] = parse_reg(parts[1])
        args['rs1'] = parse_reg(parts[2])
        args['rs2'] = parse_reg(parts[3])

    # R4-type: op rd, rs1, rs2, rs3 (fma)
    elif op == 'fma':
        args['rd'] = parse_reg(parts[1])
        args['rs1'] = parse_reg(parts[2])
        args['rs2'] = parse_reg(parts[3])
        args['rs3'] = parse_reg(parts[4])

    # I-type: op rd, rs1, imm
    elif op in ['addi', 'andi', 'ori', 'xori', 'shli', 'shri', 'srai']:
        args['rd'] = parse_reg(parts[1])
        args['rs1'] = parse_reg(parts[2])
        args['imm'] = parse_imm(parts[3])

    # Load: ld rd, imm(rs1) or ld rd, rs1, imm
    elif op == 'ld':
        args['rd'] = parse_reg(parts[1])
        if '(' in parts[2]:
            m = re.match(r'(-?\d+|0x[0-9a-fA-F]+)?\((\w+)\)', parts[2])
            if m:
                args['imm'] = parse_imm(m.group(1)) if m.group(1) else 0
                args['rs1'] = parse_reg(m.group(2))
            else:
                raise ValueError(f"Invalid load format: {parts[2]}")
        else:
            args['rs1'] = parse_reg(parts[2])
            args['imm'] = parse_imm(parts[3]) if len(parts) > 3 else 0

    # Store: st rs2, imm(rs1) or st rs1, rs2, imm (base, data, offset)
    elif op == 'st':
        if '(' in parts[2]:
            args['rs2'] = parse_reg(parts[1])  # data
            m = re.match(r'(-?\d+|0x[0-9a-fA-F]+)?\((\w+)\)', parts[2])
            if m:
                args['imm'] = parse_imm(m.group(1)) if m.group(1) else 0
                args['rs1'] = parse_reg(m.group(2))  # base
            else:
                raise ValueError(f"Invalid store format: {parts[2]}")
        else:
            args['rs1'] = parse_reg(parts[1])  # base
            args['rs2'] = parse_reg(parts[2])  # data
            args['imm'] = parse_imm(parts[3]) if len(parts) > 3 else 0

    # Scratchpad load: lds rd, addr
    elif op == 'lds':
        args['rd'] = parse_reg(parts[1])
        args['addr'] = parse_imm(parts[2])

    # Scratchpad store: sts rs1, addr
    elif op == 'sts':
        args['rs1'] = parse_reg(parts[1])
        args['addr'] = parse_imm(parts[2])

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

    # Branch: beq rs1, rs2, label
    elif op in ['beq', 'bne', 'blt', 'bge']:
        args['rs1'] = parse_reg(parts[1])
        args['rs2'] = parse_reg(parts[2])
        args['label'] = parts[3]

    # Jump: jmp label
    elif op == 'jmp':
        args['label'] = parts[1]

    # SSY: ssy label (set synchronization point)
    elif op == 'ssy':
        args['label'] = parts[1]

    # halt, nop (possibly with .s suffix)
    elif op in ['halt', 'nop'] or op in ['halt.s', 'nop.s']:
        pass

    else:
        # Check if it's an instruction with .s suffix we haven't handled
        base_op = op[:-2] if op.endswith('.s') else op
        if base_op in ['add', 'sub', 'muls', 'mulu', 'and', 'or', 'xor', 'slt', 'sltu', 'min', 'max', 'shl', 'shr', 'sra', 'fadd', 'fsub', 'fmul']:
            # Re-parse as R-type with .s suffix
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

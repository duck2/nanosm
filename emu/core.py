"""GPU functional emulator core. 8-lane SIMD with reconvergence stack."""

import struct
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

def f32_to_bits(f: float) -> int:
    return struct.unpack('<I', struct.pack('<f', f))[0]

def bits_to_f32(b: int) -> float:
    return struct.unpack('<f', struct.pack('<I', b & 0xFFFFFFFF))[0]

@dataclass
class ResourceDescriptor:
    """Resource descriptor for texture/buffer addressing."""
    base_addr: int = 0
    stride: int = 0
    type_: int = 0  # 0=buffer, 1=tex1d, 2=tex2d

@dataclass
class ReconvStackEntry:
    """Entry for the reconvergence stack. Holds mask and PC to switch to."""
    mask: int
    pc: int

@dataclass
class RenderTargetDesc:
    """Render target descriptor for tile commit."""
    base_addr: int = 0
    stride: int = 0      # bytes per row
    tile_width: int = 32
    tile_height: int = 32

@dataclass
class GPUState:
    """Complete GPU state for one warp."""
    num_lanes: int = 8
    rf: np.ndarray = field(default_factory=lambda: None)  # [32, lanes] registers
    pf: np.ndarray = field(default_factory=lambda: None)  # [8, lanes] predicate registers
    pc: int = 0
    active_mask: int = 0xFF  # all lanes active by default
    scratchpad: np.ndarray = field(default_factory=lambda: None)  # [4096] lane-interleaved
    reconv_stack: List[ReconvStackEntry] = field(default_factory=list)
    descriptors: List[ResourceDescriptor] = field(default_factory=list)
    global_mem: bytearray = field(default_factory=lambda: bytearray(4 * 1024 * 1024))  # 4MB
    halted: bool = False

    def __post_init__(self):
        if self.rf is None:
            self.rf = np.zeros((32, self.num_lanes), dtype=np.uint32)
        if self.pf is None:
            self.pf = np.zeros((8, self.num_lanes), dtype=np.bool_)
        if self.scratchpad is None:
            self.scratchpad = np.zeros(4096, dtype=np.uint32)
        if not self.descriptors:
            self.descriptors = [ResourceDescriptor() for _ in range(16)]

class Emulator:
    """Functional GPU emulator."""

    def __init__(self, num_lanes: int = 8):
        self.state = GPUState(num_lanes=num_lanes, active_mask=(1 << num_lanes) - 1)
        self.instructions: List[tuple] = []
        self.labels: dict = {}

    def load_program(self, instrs: List[tuple], labels: dict):
        """Load parsed program into emulator."""
        self.instructions = instrs
        self.labels = labels
        self.state.pc = 0
        self.state.halted = False

    def is_lane_active(self, lane: int) -> bool:
        return bool(self.state.active_mask & (1 << lane))

    def read_reg(self, reg: int) -> np.ndarray:
        """Read register, R0 always returns 0."""
        if reg == 0:
            return np.zeros(self.state.num_lanes, dtype=np.uint32)
        return self.state.rf[reg].copy()

    def write_reg(self, reg: int, values: np.ndarray):
        """Write register respecting active mask, R0 ignored."""
        if reg == 0:
            return
        for lane in range(self.state.num_lanes):
            if self.is_lane_active(lane):
                self.state.rf[reg, lane] = values[lane]

    def read_reg_f32(self, reg: int) -> np.ndarray:
        """Read register as float32."""
        bits = self.read_reg(reg)
        return np.array([bits_to_f32(int(b)) for b in bits], dtype=np.float32)

    def write_reg_f32(self, dst: int, values: np.ndarray):
        """Write float32 values to register."""
        bits = np.array([f32_to_bits(float(v)) for v in values], dtype=np.uint32)
        self.write_reg(dst, bits)

    def read_pred(self, pred: int) -> np.ndarray:
        """Read predicate register."""
        return self.state.pf[pred].copy()

    def write_pred(self, pred: int, values: np.ndarray):
        """Write predicate register respecting active mask."""
        for lane in range(self.state.num_lanes):
            if self.is_lane_active(lane):
                self.state.pf[pred, lane] = bool(values[lane])

    # =========================================================================
    # ALU Operations (16x16 mul only)
    # =========================================================================
    def alu_add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a.astype(np.int64) + b.astype(np.int64)).astype(np.uint32)

    def alu_sub(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a.astype(np.int64) - b.astype(np.int64)).astype(np.uint32)

    def alu_muls(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Signed 16x16 -> 32 multiply."""
        a16 = (a & 0xFFFF).astype(np.int16).astype(np.int32)
        b16 = (b & 0xFFFF).astype(np.int16).astype(np.int32)
        return (a16 * b16).astype(np.uint32)

    def alu_mulu(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Unsigned 16x16 -> 32 multiply."""
        a16 = (a & 0xFFFF).astype(np.uint32)
        b16 = (b & 0xFFFF).astype(np.uint32)
        return (a16 * b16).astype(np.uint32)

    def alu_and(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a & b

    def alu_or(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a | b

    def alu_xor(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a ^ b

    # =========================================================================
    # Shifter Operations
    # =========================================================================
    def shift_shl(self, a: np.ndarray, amt: np.ndarray) -> np.ndarray:
        return (a << (amt & 31)).astype(np.uint32)

    def shift_shr(self, a: np.ndarray, amt: np.ndarray) -> np.ndarray:
        return (a >> (amt & 31)).astype(np.uint32)

    def shift_sra(self, a: np.ndarray, amt: np.ndarray) -> np.ndarray:
        return (a.astype(np.int32) >> (amt & 31)).astype(np.uint32)

    # =========================================================================
    # FPU Operations (FMA based)
    # =========================================================================
    def fpu_fma(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        """FMA: a * b + c."""
        af = np.array([bits_to_f32(int(x)) for x in a], dtype=np.float32)
        bf = np.array([bits_to_f32(int(x)) for x in b], dtype=np.float32)
        cf = np.array([bits_to_f32(int(x)) for x in c], dtype=np.float32)
        result = af * bf + cf
        return np.array([f32_to_bits(float(x)) for x in result], dtype=np.uint32)

    def fpu_add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """ADD via FMA: 1.0 * a + b."""
        one = np.full(self.state.num_lanes, f32_to_bits(1.0), dtype=np.uint32)
        return self.fpu_fma(one, a, b)

    def fpu_sub(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """SUB: a - b = 1.0 * a + (-b)."""
        af = np.array([bits_to_f32(int(x)) for x in a], dtype=np.float32)
        bf = np.array([bits_to_f32(int(x)) for x in b], dtype=np.float32)
        result = af - bf
        return np.array([f32_to_bits(float(x)) for x in result], dtype=np.uint32)

    def fpu_mul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """MUL via FMA: a * b + 0."""
        zero = np.zeros(self.state.num_lanes, dtype=np.uint32)
        return self.fpu_fma(a, b, zero)

    def fpu_rcp(self, a: np.ndarray) -> np.ndarray:
        """Reciprocal: 1.0 / a."""
        af = np.array([bits_to_f32(int(x)) for x in a], dtype=np.float32)
        result = np.where(af != 0, 1.0 / af, np.float32(np.inf))
        return np.array([f32_to_bits(float(x)) for x in result], dtype=np.uint32)

    def fpu_itof(self, a: np.ndarray) -> np.ndarray:
        """Signed int32 to float32."""
        af = a.astype(np.int32).astype(np.float32)
        return np.array([f32_to_bits(float(x)) for x in af], dtype=np.uint32)

    def fpu_ftofx_clamp(self, a: np.ndarray, frac_bits: int) -> tuple[np.ndarray, np.ndarray]:
        """Float32 to signed 16-bit fixed-point with clamping. Returns (result, clamped)."""
        af = np.array([bits_to_f32(int(x)) for x in a], dtype=np.float32)
        scaled = af * (1 << frac_bits)
        clamped_vals = np.clip(scaled, -32767, 32767)
        was_clamped = (scaled < -32767) | (scaled > 32767)
        result = np.trunc(clamped_vals).astype(np.int16).astype(np.uint32) & 0xFFFF
        return result, was_clamped

    def fpu_ftofx_repeat(self, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Float32 to u0.15 with repeat mode. Returns (result, sign)."""
        af = np.array([bits_to_f32(int(x)) for x in a], dtype=np.float32)
        sign = af < 0
        frac = np.abs(af) - np.floor(np.abs(af))  # Get fractional part
        result = (frac * (1 << 15)).astype(np.uint32) & 0x7FFF
        return result, sign

    # =========================================================================
    # Memory Operations
    # =========================================================================
    def mem_load_global(self, addr: np.ndarray) -> np.ndarray:
        """Load 32-bit from global memory per lane."""
        result = np.zeros(self.state.num_lanes, dtype=np.uint32)
        for lane in range(self.state.num_lanes):
            if self.is_lane_active(lane):
                a = int(addr[lane]) & 0xFFFFFFFC  # align to 4
                if a + 4 <= len(self.state.global_mem):
                    result[lane] = struct.unpack('<I', self.state.global_mem[a:a+4])[0]
        return result

    def mem_store_global(self, addr: np.ndarray, data: np.ndarray):
        """Store 32-bit to global memory per lane."""
        for lane in range(self.state.num_lanes):
            if self.is_lane_active(lane):
                a = int(addr[lane]) & 0xFFFFFFFC
                if a + 4 <= len(self.state.global_mem):
                    self.state.global_mem[a:a+4] = struct.pack('<I', int(data[lane]))

    def mem_load_2d(self, rx: np.ndarray, ry: np.ndarray, desc_ptr: int) -> np.ndarray:
        """Load from 2D texture using descriptor."""
        desc = self.read_rt_desc(desc_ptr)
        result = np.zeros(self.state.num_lanes, dtype=np.uint32)
        for lane in range(self.state.num_lanes):
            if self.is_lane_active(lane):
                x, y = int(rx[lane]), int(ry[lane])
                addr = desc.base_addr + y * desc.stride + x * 4
                if addr + 4 <= len(self.state.global_mem):
                    result[lane] = struct.unpack('<I', self.state.global_mem[addr:addr+4])[0]
        return result

    def mem_store_2d(self, rx: np.ndarray, ry: np.ndarray, desc_ptr: int, data: np.ndarray):
        """Store to 2D texture using descriptor."""
        desc = self.read_rt_desc(desc_ptr)
        for lane in range(self.state.num_lanes):
            if self.is_lane_active(lane):
                x, y = int(rx[lane]), int(ry[lane])
                addr = desc.base_addr + y * desc.stride + x * 4
                if addr + 4 <= len(self.state.global_mem):
                    self.state.global_mem[addr:addr+4] = struct.pack('<I', int(data[lane]))

    def mem_load_scratch(self, addr: np.ndarray) -> np.ndarray:
        """Load from lane-interleaved scratchpad."""
        result = np.zeros(self.state.num_lanes, dtype=np.uint32)
        for lane in range(self.state.num_lanes):
            if self.is_lane_active(lane):
                a = int(addr[lane]) & 0xFFF  # 4KB = 4096 words
                result[lane] = self.state.scratchpad[a]
        return result

    def mem_store_scratch(self, addr: np.ndarray, data: np.ndarray):
        """Store to lane-interleaved scratchpad."""
        for lane in range(self.state.num_lanes):
            if self.is_lane_active(lane):
                a = int(addr[lane]) & 0xFFF
                self.state.scratchpad[a] = data[lane]

    # =========================================================================
    # Tile Operations
    # =========================================================================
    def read_rt_desc(self, ptr: int) -> RenderTargetDesc:
        """Read render target descriptor from global memory."""
        mem = self.state.global_mem
        return RenderTargetDesc(
            base_addr=struct.unpack('<I', mem[ptr:ptr+4])[0],
            stride=struct.unpack('<I', mem[ptr+4:ptr+8])[0],
            tile_width=struct.unpack('<I', mem[ptr+8:ptr+12])[0],
            tile_height=struct.unpack('<I', mem[ptr+12:ptr+16])[0],
        )

    def tile_commit(self, tile_x: int, tile_y: int, desc: RenderTargetDesc):
        """Commit scratchpad tile buffer to framebuffer (lane-interleaved layout)."""
        tw, th = desc.tile_width, desc.tile_height
        for row in range(th):
            for px in range(tw):
                scratch_addr = row * tw + px  # linear pixel order in scratchpad
                py = row
                fb_addr = (desc.base_addr
                           + (tile_y * th + py) * desc.stride
                           + (tile_x * tw + px) * 4)
                rgba = int(self.state.scratchpad[scratch_addr & 0xFFF])
                if fb_addr + 4 <= len(self.state.global_mem):
                    struct.pack_into('<I', self.state.global_mem, fb_addr, rgba)

    # =========================================================================
    # Control Flow (SSY/Branch/.S model)
    # =========================================================================
    def do_ssy(self, target_pc: int):
        """SSY: Push current active mask and target PC for later reconvergence."""
        self.state.reconv_stack.append(ReconvStackEntry(self.state.active_mask, target_pc))

    def do_bra(self, pred: int, target_pc: int):
        """Predicated branch with divergence handling."""
        cond = self.read_pred(pred)
        taken_mask = 0
        for lane in range(self.state.num_lanes):
            if self.is_lane_active(lane) and cond[lane]:
                taken_mask |= (1 << lane)
        not_taken_mask = self.state.active_mask & ~taken_mask

        if taken_mask == 0:
            self.state.pc += 1
        elif not_taken_mask == 0:
            self.state.pc = target_pc
        else:
            # Divergence: push taken path, continue with not-taken
            self.state.reconv_stack.append(ReconvStackEntry(taken_mask, target_pc))
            self.state.active_mask = not_taken_mask
            self.state.pc += 1

    def do_bra_uni(self, pred: Optional[int], target_pc: int):
        """Uniform branch. If pred is None, unconditional jump."""
        if pred is None:
            self.state.pc = target_pc
            return
        cond = self.read_pred(pred)
        for lane in range(self.state.num_lanes):
            if self.is_lane_active(lane):
                if cond[lane]:
                    self.state.pc = target_pc
                else:
                    self.state.pc += 1
                return
        self.state.pc += 1

    def do_sync(self):
        """Pop stack entry and switch to that mask/PC."""
        if self.state.reconv_stack:
            entry = self.state.reconv_stack.pop()
            self.state.active_mask = entry.mask
            self.state.pc = entry.pc
        else:
            self.state.pc += 1

    # =========================================================================
    # Instruction Execution
    # =========================================================================
    def exec_instr(self, op: str, args: dict):
        """Execute a single instruction."""
        sync_after = False
        if op.endswith('.s'):
            sync_after = True
            op = op[:-2]

        # Handle predication: narrow active mask to lanes where predicate is true
        saved_mask = None
        if 'pred' in args and op not in ('bra', 'bra.uni'):
            saved_mask = self.state.active_mask
            pred_vals = self.read_pred(args['pred'])
            pred_mask = sum((1 << i) for i in range(self.state.num_lanes) if pred_vals[i])
            self.state.active_mask &= pred_mask

        # ALU R-type: op rd, rs1, rs2
        if op == 'add':
            self.write_reg(args['rd'], self.alu_add(self.read_reg(args['rs1']), self.read_reg(args['rs2'])))
        elif op == 'sub':
            self.write_reg(args['rd'], self.alu_sub(self.read_reg(args['rs1']), self.read_reg(args['rs2'])))
        elif op == 'muls':
            self.write_reg(args['rd'], self.alu_muls(self.read_reg(args['rs1']), self.read_reg(args['rs2'])))
        elif op == 'mulu':
            self.write_reg(args['rd'], self.alu_mulu(self.read_reg(args['rs1']), self.read_reg(args['rs2'])))
        elif op == 'and':
            self.write_reg(args['rd'], self.alu_and(self.read_reg(args['rs1']), self.read_reg(args['rs2'])))
        elif op == 'or':
            self.write_reg(args['rd'], self.alu_or(self.read_reg(args['rs1']), self.read_reg(args['rs2'])))
        elif op == 'xor':
            self.write_reg(args['rd'], self.alu_xor(self.read_reg(args['rs1']), self.read_reg(args['rs2'])))

        # ALU I-type: op rd, rs1, imm
        elif op == 'addi':
            imm = np.full(self.state.num_lanes, args['imm'] & 0xFFFF, dtype=np.uint32)
            if args['imm'] & 0x8000:  # sign extend 16-bit
                imm = (imm | 0xFFFF0000).astype(np.uint32)
            self.write_reg(args['rd'], self.alu_add(self.read_reg(args['rs1']), imm))
        elif op == 'andi':
            imm = np.full(self.state.num_lanes, args['imm'] & 0xFFFF, dtype=np.uint32)
            self.write_reg(args['rd'], self.alu_and(self.read_reg(args['rs1']), imm))
        elif op == 'ori':
            imm = np.full(self.state.num_lanes, args['imm'] & 0xFFFF, dtype=np.uint32)
            self.write_reg(args['rd'], self.alu_or(self.read_reg(args['rs1']), imm))
        elif op == 'xori':
            imm = np.full(self.state.num_lanes, args['imm'] & 0xFFFF, dtype=np.uint32)
            self.write_reg(args['rd'], self.alu_xor(self.read_reg(args['rs1']), imm))

        # Shift ops
        elif op == 'shl':
            self.write_reg(args['rd'], self.shift_shl(self.read_reg(args['rs1']), self.read_reg(args['rs2'])))
        elif op == 'shr':
            self.write_reg(args['rd'], self.shift_shr(self.read_reg(args['rs1']), self.read_reg(args['rs2'])))
        elif op == 'sra':
            self.write_reg(args['rd'], self.shift_sra(self.read_reg(args['rs1']), self.read_reg(args['rs2'])))
        elif op == 'shli':
            amt = np.full(self.state.num_lanes, args['imm'] & 31, dtype=np.uint32)
            self.write_reg(args['rd'], self.shift_shl(self.read_reg(args['rs1']), amt))
        elif op == 'shri':
            amt = np.full(self.state.num_lanes, args['imm'] & 31, dtype=np.uint32)
            self.write_reg(args['rd'], self.shift_shr(self.read_reg(args['rs1']), amt))
        elif op == 'srai':
            amt = np.full(self.state.num_lanes, args['imm'] & 31, dtype=np.uint32)
            self.write_reg(args['rd'], self.shift_sra(self.read_reg(args['rs1']), amt))

        # FPU ops
        elif op == 'fma':
            self.write_reg(args['rd'], self.fpu_fma(self.read_reg(args['rs1']), self.read_reg(args['rs2']), self.read_reg(args['rs3'])))
        elif op == 'fadd':
            self.write_reg(args['rd'], self.fpu_add(self.read_reg(args['rs1']), self.read_reg(args['rs2'])))
        elif op == 'fsub':
            self.write_reg(args['rd'], self.fpu_sub(self.read_reg(args['rs1']), self.read_reg(args['rs2'])))
        elif op == 'fmul':
            self.write_reg(args['rd'], self.fpu_mul(self.read_reg(args['rs1']), self.read_reg(args['rs2'])))
        elif op == 'frcp':
            self.write_reg(args['rd'], self.fpu_rcp(self.read_reg(args['rs1'])))
        elif op == 'itof':
            self.write_reg(args['rd'], self.fpu_itof(self.read_reg(args['rs1'])))
        elif op.startswith('ftofx.'):
            parts = op.split('.')
            if len(parts) >= 3 and parts[2] == 'repeat':
                result, sign = self.fpu_ftofx_repeat(self.read_reg(args['rs1']))
                self.write_reg(args['rd'], result)
                self.write_pred(args['pd'], sign)
            else:
                frac_bits = int(parts[1])
                result, clamped = self.fpu_ftofx_clamp(self.read_reg(args['rs1']), frac_bits)
                self.write_reg(args['rd'], result)
                self.write_pred(args['pd'], clamped)

        # Load/Store global: ldg rd, [rs1+imm] / stg [rs1+imm], rs2
        elif op == 'ldg':
            base = self.read_reg(args['rs1'])
            addr = (base.astype(np.int64) + args.get('imm', 0)).astype(np.uint32)
            self.write_reg(args['rd'], self.mem_load_global(addr))
        elif op == 'stg':
            base = self.read_reg(args['rs1'])
            addr = (base.astype(np.int64) + args.get('imm', 0)).astype(np.uint32)
            self.mem_store_global(addr, self.read_reg(args['rs2']))

        # 2D texture load/store
        elif op == 'ld2d':
            desc_ptr = int(self.read_reg(args['rdesc'])[0])
            self.write_reg(args['rd'], self.mem_load_2d(self.read_reg(args['rx']), self.read_reg(args['ry']), desc_ptr))
        elif op == 'st2d':
            desc_ptr = int(self.read_reg(args['rdesc'])[0])
            self.mem_store_2d(self.read_reg(args['rx']), self.read_reg(args['ry']), desc_ptr, self.read_reg(args['rs']))

        # Scratchpad: lds rd, [rs1+imm] / sts [rs1+imm], rs2
        elif op == 'lds':
            base = self.read_reg(args['rs1'])
            addr = (base.astype(np.int64) + args.get('imm', 0)).astype(np.uint32)
            self.write_reg(args['rd'], self.mem_load_scratch(addr))
        elif op == 'sts':
            base = self.read_reg(args['rs1'])
            addr = (base.astype(np.int64) + args.get('imm', 0)).astype(np.uint32)
            self.mem_store_scratch(addr, self.read_reg(args['rs2']))

        # Move imm (upper 16 bits): lui rd, imm16
        elif op == 'lui':
            val = np.full(self.state.num_lanes, (args['imm'] & 0xFFFF) << 16, dtype=np.uint32)
            self.write_reg(args['rd'], val)

        # Move: mov rd, rs1
        elif op == 'mov':
            self.write_reg(args['rd'], self.read_reg(args['rs1']))

        # Lane ID: lid rd (writes 0,1,2,...,lanes-1 to each lane)
        elif op == 'lid':
            vals = np.arange(self.state.num_lanes, dtype=np.uint32)
            self.write_reg(args['rd'], vals)

        # SSY: set synchronization point
        elif op == 'ssy':
            self.do_ssy(self.labels[args['label']])
            self.state.pc += 1
            return

        # Set predicate: setp.cmp.type pd, rs1, rs2
        elif op.startswith('setp.'):
            rs1 = self.read_reg(args['rs1'])
            rs2 = self.read_reg(args['rs2'])
            cmp_op = args['cmp']
            typ = args.get('type', 'i32')
            signed = typ in ('i32', 's32')
            if cmp_op == 'lt':
                cond = rs1.astype(np.int32) < rs2.astype(np.int32) if signed else rs1 < rs2
            elif cmp_op == 'ge':
                cond = rs1.astype(np.int32) >= rs2.astype(np.int32) if signed else rs1 >= rs2
            elif cmp_op == 'eq':
                cond = rs1 == rs2
            elif cmp_op == 'ne':
                cond = rs1 != rs2
            elif cmp_op == 'le':
                cond = rs1.astype(np.int32) <= rs2.astype(np.int32) if signed else rs1 <= rs2
            elif cmp_op == 'gt':
                cond = rs1.astype(np.int32) > rs2.astype(np.int32) if signed else rs1 > rs2
            else:
                raise ValueError(f"Unknown comparison: {cmp_op}")
            self.write_pred(args['pd'], cond)

        # Predicated branch (divergent): @pN bra label
        elif op == 'bra':
            self.do_bra(args['pred'], self.labels[args['label']])
            return

        # Uniform branch: bra.uni label (unconditional) or @pN bra.uni label (predicated)
        elif op == 'bra.uni':
            self.do_bra_uni(args.get('pred'), self.labels[args['label']])
            return

        # Halt
        elif op == 'halt':
            self.state.halted = True
            return

        # Nop
        elif op == 'nop':
            pass

        # Tile commit: tile_commit rx, ry, rd (tile_x, tile_y, rt_desc_ptr)
        elif op == 'tile_commit':
            tile_x = int(self.read_reg(args['rx'])[0])
            tile_y = int(self.read_reg(args['ry'])[0])
            desc_ptr = int(self.read_reg(args['rd'])[0])
            desc = self.read_rt_desc(desc_ptr)
            self.tile_commit(tile_x, tile_y, desc)

        # Tile wait: nop in emulator (no async DMA)
        elif op == 'tile_wait':
            pass

        # Predicate logic: pand/por/pxor pd, ps1, ps2
        elif op == 'pand':
            p1 = self.read_pred(args['ps1'])
            p2 = self.read_pred(args['ps2'])
            self.write_pred(args['pd'], p1 & p2)
        elif op == 'por':
            p1 = self.read_pred(args['ps1'])
            p2 = self.read_pred(args['ps2'])
            self.write_pred(args['pd'], p1 | p2)
        elif op == 'pxor':
            p1 = self.read_pred(args['ps1'])
            p2 = self.read_pred(args['ps2'])
            self.write_pred(args['pd'], p1 ^ p2)

        else:
            raise ValueError(f"Unknown instruction: {op}")

        # Restore active mask if we narrowed it for predication
        if saved_mask is not None:
            self.state.active_mask = saved_mask

        # Handle .s suffix - sync after instruction
        if sync_after:
            self.do_sync()
        else:
            self.state.pc += 1

    def step(self) -> bool:
        """Execute one instruction. Returns True if still running."""
        if self.state.halted or self.state.pc >= len(self.instructions):
            return False
        op, args = self.instructions[self.state.pc]
        self.exec_instr(op, args)
        return not self.state.halted and self.state.pc < len(self.instructions)

    def run(self, max_steps: int = 100000) -> int:
        """Run until halt or max steps. Returns steps executed."""
        steps = 0
        while steps < max_steps and self.step():
            steps += 1
        return steps

    def dump_regs(self, regs: List[int] = None):
        """Print register contents."""
        if regs is None:
            regs = range(32)
        for r in regs:
            vals = self.state.rf[r]
            print(f"R{r:2d}: " + " ".join(f"{v:08x}" for v in vals))

    def dump_regs_f32(self, regs: List[int]):
        """Print register contents as float32."""
        for r in regs:
            vals = [bits_to_f32(int(v)) for v in self.state.rf[r]]
            print(f"R{r:2d}: " + " ".join(f"{v:12.6f}" for v in vals))

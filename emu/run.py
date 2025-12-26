"""Test runner for GPU emulator."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emu.core import Emulator, f32_to_bits
from emu.asm import parse_program, load_file

def run_program(text_or_path: str, verbose: bool = True) -> Emulator:
    """Run a program from text or file path."""
    if os.path.exists(text_or_path):
        instrs, labels = load_file(text_or_path)
    else:
        instrs, labels = parse_program(text_or_path)

    emu = Emulator(num_lanes=8)
    emu.load_program(instrs, labels)

    if verbose:
        print(f"Loaded {len(instrs)} instructions, {len(labels)} labels")

    steps = emu.run()

    if verbose:
        print(f"Executed {steps} steps")

    return emu

def test_basic_alu():
    """Test basic ALU operations."""
    prog = """
    ; Test basic ALU ops
    lid r1              ; r1 = lane_id (0,1,2,3,4,5,6,7)
    addi r2, r0, 10     ; r2 = 10
    add r3, r1, r2      ; r3 = lane_id + 10
    sub r4, r2, r1      ; r4 = 10 - lane_id
    muls r5, r1, r1     ; r5 = lane_id * lane_id (signed 16x16)
    halt
    """
    emu = run_program(prog, verbose=False)
    print("=== Test Basic ALU ===")
    emu.dump_regs([1, 2, 3, 4, 5])
    print()

def test_shifts():
    """Test shift operations."""
    prog = """
    lui r1, 0xFF        ; r1 = 0x00FF0000
    addi r1, r1, 0xFF   ; r1 = 0x00FF00FF
    lid r2              ; r2 = lane_id
    shl r3, r1, r2      ; r3 = r1 << lane_id
    shri r4, r1, 8      ; r4 = r1 >> 8
    halt
    """
    emu = run_program(prog, verbose=False)
    print("=== Test Shifts ===")
    emu.dump_regs([1, 2, 3, 4])
    print()

def test_memory():
    """Test memory operations."""
    prog = """
    ; Store lane_id * 4 at address 0x100 + lane_id*4
    lid r1              ; lane_id
    shli r2, r1, 2      ; r2 = lane_id * 4
    lui r3, 0           ; base = 0x100
    addi r3, r3, 0x100
    add r4, r3, r2      ; addr = 0x100 + lane_id*4
    st r4, r2, 0        ; store lane_id*4 at addr
    ; Load back
    ld r5, r4, 0
    halt
    """
    emu = run_program(prog, verbose=False)
    print("=== Test Memory ===")
    emu.dump_regs([1, 2, 4, 5])
    print()

def test_fpu():
    """Test FPU operations."""
    prog = """
    ; Load some floats using lui+addi to set bit patterns
    ; 1.0f = 0x3f800000, 2.0f = 0x40000000
    lui r1, 0x3f80      ; upper bits of 1.0
    lui r2, 0x4000      ; upper bits of 2.0
    fadd r3, r1, r2     ; r3 = 1.0 + 2.0 = 3.0 (0x40400000)
    fmul r4, r1, r2     ; r4 = 1.0 * 2.0 = 2.0
    fma r5, r2, r2, r1  ; r5 = 2.0 * 2.0 + 1.0 = 5.0 (0x40a00000)
    halt
    """
    emu = run_program(prog, verbose=False)
    print("=== Test FPU ===")
    print("As hex:")
    emu.dump_regs([1, 2, 3, 4, 5])
    print("As float:")
    emu.dump_regs_f32([1, 2, 3, 4, 5])
    print()

def test_branch():
    """Test branching with divergence using SSY/Branch/.S model."""
    prog = """
    ; if (lane_id < 4) r5 = 100 else r5 = 200
    ;
    ; Control flow:
    ;   SSY pushes (all_lanes, reconv_pc)
    ;   BLT pushes (taken_lanes, taken_pc), falls through with not_taken
    ;   NOP.S pops and switches

    lid r1              ; lane_id
    addi r2, r0, 4      ; threshold = 4
    addi r3, r0, 100    ; value for taken path
    addi r4, r0, 200    ; value for not-taken path

    ssy reconv          ; push (0xFF, reconv) - join point
    blt r1, r2, taken   ; push (0x0F, taken), mask = 0xF0, fall through

not_taken:
    ; Executed by lanes 4-7 (not_taken_mask = 0xF0)
    mov r5, r4          ; r5 = 200
    nop.s               ; pop -> switch to (0x0F, taken)

taken:
    ; Executed by lanes 0-3 (taken_mask = 0x0F)
    mov r5, r3          ; r5 = 100
    nop.s               ; pop -> switch to (0xFF, reconv)

reconv:
    ; All lanes active again
    halt
    """
    emu = run_program(prog, verbose=False)
    print("=== Test Branch (divergent) ===")
    print("Lane IDs 0-3 should have 100, 4-7 should have 200")
    emu.dump_regs([1, 5])
    print()

def test_scratchpad():
    """Test scratchpad memory."""
    prog = """
    lid r1              ; lane_id
    shli r2, r1, 1      ; r2 = lane_id * 2
    sts r2, 0           ; scratchpad[0] = lane_id * 2
    lds r3, 0           ; r3 = scratchpad[0]
    halt
    """
    emu = run_program(prog, verbose=False)
    print("=== Test Scratchpad ===")
    emu.dump_regs([1, 2, 3])
    print()

def test_saxpy():
    """Test SAXPY: y = a*x + y."""
    # Setup: preload x and y vectors in global memory
    prog = """
    ; SAXPY: y[i] = a * x[i] + y[i]
    ; x at 0x000, y at 0x100, a in r10
    ; Result in y

    lid r1              ; lane_id
    shli r2, r1, 2      ; offset = lane_id * 4

    ; Load x[lane_id]
    ld r3, r2, 0        ; r3 = x[lane_id] (from addr 0 + offset)

    ; Load y[lane_id]
    addi r4, r2, 0x100  ; addr of y
    ld r5, r4, 0        ; r5 = y[lane_id]

    ; a = 2.0 (0x40000000)
    lui r10, 0x4000

    ; y = a * x + y
    fma r6, r10, r3, r5

    ; Store result
    st r4, r6, 0

    halt
    """
    emu = Emulator(num_lanes=8)

    # Preload x = [1.0, 2.0, 3.0, ...] and y = [10.0, 20.0, 30.0, ...]
    import struct
    for i in range(8):
        x_val = float(i + 1)
        y_val = float((i + 1) * 10)
        x_addr = i * 4
        y_addr = 0x100 + i * 4
        emu.state.global_mem[x_addr:x_addr+4] = struct.pack('<f', x_val)
        emu.state.global_mem[y_addr:y_addr+4] = struct.pack('<f', y_val)

    from emu.asm import parse_program
    instrs, labels = parse_program(prog)
    emu.load_program(instrs, labels)
    emu.run()

    print("=== Test SAXPY ===")
    print("x = [1, 2, 3, 4, 5, 6, 7, 8]")
    print("y = [10, 20, 30, 40, 50, 60, 70, 80]")
    print("a = 2.0")
    print("Result y = a*x + y:")
    results = []
    for i in range(8):
        y_addr = 0x100 + i * 4
        val = struct.unpack('<f', emu.state.global_mem[y_addr:y_addr+4])[0]
        results.append(val)
    print(results)
    print("Expected: [12, 24, 36, 48, 60, 72, 84, 96]")
    print()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        emu = run_program(sys.argv[1])
        print("\nFinal register state:")
        emu.dump_regs(range(16))
    else:
        print("GPU Emulator Tests\n")
        test_basic_alu()
        test_shifts()
        test_memory()
        test_fpu()
        test_branch()
        test_scratchpad()
        test_saxpy()
        print("All tests completed!")


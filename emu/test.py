"""Run all basic emulator tests and verify results."""

import sys
import os
import struct

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emu.core import Emulator, f32_to_bits, bits_to_f32
from emu.asm import load_file

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), 'examples', 'basic')


def run_asm(path: str, num_lanes: int = 8) -> Emulator:
    """Load and run an ASM file, return the emulator."""
    instrs, labels = load_file(path)
    emu = Emulator(num_lanes=num_lanes)
    emu.load_program(instrs, labels)
    emu.run()
    return emu


def check_reg(emu: Emulator, reg: int, expected: list, name: str = "") -> bool:
    """Check register values match expected. Returns True if passed."""
    actual = list(emu.state.rf[reg])
    if actual == expected:
        return True
    print(f"  FAIL {name}: r{reg}")
    print(f"    Expected: {[hex(v) for v in expected]}")
    print(f"    Actual:   {[hex(v) for v in actual]}")
    return False


def check_reg_f32(emu: Emulator, reg: int, expected: list, tol: float = 1e-5, name: str = "") -> bool:
    """Check register values as floats. Returns True if passed."""
    actual = [bits_to_f32(int(v)) for v in emu.state.rf[reg]]
    for i, (a, e) in enumerate(zip(actual, expected)):
        if abs(a - e) > tol:
            print(f"  FAIL {name}: r{reg} lane {i}")
            print(f"    Expected: {expected}")
            print(f"    Actual:   {actual}")
            return False
    return True


def test_basic_alu() -> bool:
    """Test basic ALU operations."""
    emu = run_asm(os.path.join(EXAMPLES_DIR, 'basic_alu.asm'))
    passed = True
    # r1 = lane_id
    passed &= check_reg(emu, 1, list(range(8)), "basic_alu r1")
    # r2 = 10
    passed &= check_reg(emu, 2, [10]*8, "basic_alu r2")
    # r3 = lane_id + 10
    passed &= check_reg(emu, 3, [10+i for i in range(8)], "basic_alu r3")
    # r4 = 10 - lane_id
    passed &= check_reg(emu, 4, [10-i for i in range(8)], "basic_alu r4")
    # r5 = lane_id^2
    passed &= check_reg(emu, 5, [i*i for i in range(8)], "basic_alu r5")
    return passed


def test_fpu() -> bool:
    """Test FPU operations."""
    emu = run_asm(os.path.join(EXAMPLES_DIR, 'fpu.asm'))
    passed = True
    # r3 = 3.0
    passed &= check_reg_f32(emu, 3, [3.0]*8, name="fpu r3")
    # r4 = 2.0
    passed &= check_reg_f32(emu, 4, [2.0]*8, name="fpu r4")
    # r5 = 5.0
    passed &= check_reg_f32(emu, 5, [5.0]*8, name="fpu r5")
    return passed


def test_shifts() -> bool:
    """Test shift operations."""
    emu = run_asm(os.path.join(EXAMPLES_DIR, 'shifts.asm'))
    passed = True
    # r1 = 0x00FF00FF
    passed &= check_reg(emu, 1, [0x00FF00FF]*8, "shifts r1")
    # r2 = lane_id
    passed &= check_reg(emu, 2, list(range(8)), "shifts r2")
    # r3 = r1 << lane_id
    passed &= check_reg(emu, 3, [(0x00FF00FF << i) & 0xFFFFFFFF for i in range(8)], "shifts r3")
    # r4 = r1 >> 8
    passed &= check_reg(emu, 4, [0x0000FF00]*8, "shifts r4")
    return passed


def test_memory() -> bool:
    """Test memory operations."""
    emu = run_asm(os.path.join(EXAMPLES_DIR, 'memory.asm'))
    passed = True
    # r1 = lane_id
    passed &= check_reg(emu, 1, list(range(8)), "memory r1")
    # r2 = lane_id * 4
    passed &= check_reg(emu, 2, [i*4 for i in range(8)], "memory r2")
    # r4 = 0x100 + lane_id*4
    passed &= check_reg(emu, 4, [0x100 + i*4 for i in range(8)], "memory r4")
    # r5 = lane_id * 4 (loaded back)
    passed &= check_reg(emu, 5, [i*4 for i in range(8)], "memory r5")
    return passed


def test_scratchpad() -> bool:
    """Test scratchpad memory."""
    emu = run_asm(os.path.join(EXAMPLES_DIR, 'scratchpad.asm'))
    passed = True
    # r1 = lane_id
    passed &= check_reg(emu, 1, list(range(8)), "scratchpad r1")
    # r2 = lane_id * 2
    passed &= check_reg(emu, 2, [i*2 for i in range(8)], "scratchpad r2")
    # r3 = lane_id * 2 (loaded from scratchpad)
    passed &= check_reg(emu, 3, [i*2 for i in range(8)], "scratchpad r3")
    return passed


def test_loop() -> bool:
    """Test uniform loop."""
    emu = run_asm(os.path.join(EXAMPLES_DIR, 'loop.asm'))
    passed = True
    # r3 = 10+9+8+7+6+5+4+3+2+1 = 55
    passed &= check_reg(emu, 3, [55]*8, "loop r3")
    return passed


def test_divergent() -> bool:
    """Test divergent control flow."""
    emu = run_asm(os.path.join(EXAMPLES_DIR, 'divergent.asm'))
    passed = True
    # r5 = 100 for lanes 0-3, 200 for lanes 4-7
    expected_r5 = [100]*4 + [200]*4
    passed &= check_reg(emu, 5, expected_r5, "divergent r5")
    # r6 = r5 * 2
    expected_r6 = [200]*4 + [400]*4
    passed &= check_reg(emu, 6, expected_r6, "divergent r6")
    return passed


def test_nested_if() -> bool:
    """Test nested if with divergence."""
    emu = run_asm(os.path.join(EXAMPLES_DIR, 'nested_if.asm'))
    passed = True
    # r5: lanes 0-3 = 10, lanes 4-7 = 20
    expected_r5 = [10]*4 + [20]*4
    passed &= check_reg(emu, 5, expected_r5, "nested_if r5")
    # r6: lanes 0-1 = 100, lanes 2-3 = 200, lanes 4-7 = 300
    expected_r6 = [100, 100, 200, 200, 300, 300, 300, 300]
    passed &= check_reg(emu, 6, expected_r6, "nested_if r6")
    return passed


def test_saxpy() -> bool:
    """Test SAXPY with FMA."""
    # Pre-load memory with x and y vectors
    instrs, labels = load_file(os.path.join(EXAMPLES_DIR, 'saxpy.asm'))
    emu = Emulator(num_lanes=8)
    emu.load_program(instrs, labels)
    
    # Initialize x[0:8] at address 0x0000 with floats 1.0, 2.0, ..., 8.0
    x_vals = [float(i+1) for i in range(8)]
    for i, val in enumerate(x_vals):
        addr = i * 4
        struct.pack_into('<f', emu.state.global_mem, addr, val)
    
    # Initialize y[0:8] at address 0x0100 with floats 0.5, 0.5, ...
    y_vals = [0.5] * 8
    for i, val in enumerate(y_vals):
        addr = 0x100 + i * 4
        struct.pack_into('<f', emu.state.global_mem, addr, val)
    
    emu.run()
    
    # a = 2.0, result should be 2.0 * x[i] + 0.5
    # = 2.5, 4.5, 6.5, 8.5, 10.5, 12.5, 14.5, 16.5
    passed = True
    expected = [2.0 * x + 0.5 for x in x_vals]
    
    # Check result stored back in memory at y addresses
    for i in range(8):
        addr = 0x100 + i * 4
        actual = struct.unpack('<f', emu.state.global_mem[addr:addr+4])[0]
        if abs(actual - expected[i]) > 1e-5:
            print(f"  FAIL saxpy: y[{i}]")
            print(f"    Expected: {expected[i]}")
            print(f"    Actual:   {actual}")
            passed = False
    return passed


def test_predicated_execution() -> bool:
    """Test predicated execution (operations guarded by predicate)."""
    emu = run_asm(os.path.join(EXAMPLES_DIR, 'predicated.asm'))
    passed = True
    
    # r2: lanes 0-3 = 150, lanes 4-7 = 100
    expected_r2 = [150]*4 + [100]*4
    passed &= check_reg(emu, 2, expected_r2, "predicated r2")
    
    # r4: lanes 0-3 = lane_id, lanes 4-7 = 999
    expected_r4 = [0, 1, 2, 3, 999, 999, 999, 999]
    passed &= check_reg(emu, 4, expected_r4, "predicated r4")
    
    # r7: lanes 0-3 = 3.0f (0x40400000), lanes 4-7 = 0
    f32_3 = f32_to_bits(3.0)
    expected_r7 = [f32_3]*4 + [0]*4
    passed &= check_reg(emu, 7, expected_r7, "predicated r7")
    
    return passed


def main():
    tests = [
        ("basic_alu", test_basic_alu),
        ("fpu", test_fpu),
        ("shifts", test_shifts),
        ("memory", test_memory),
        ("scratchpad", test_scratchpad),
        ("loop", test_loop),
        ("divergent", test_divergent),
        ("nested_if", test_nested_if),
        ("saxpy", test_saxpy),
        ("predicated_execution", test_predicated_execution),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        print(f"Testing {name}... ", end="")
        try:
            if test_fn():
                print("PASS")
                passed += 1
            else:
                print("FAIL")
                failed += 1
        except Exception as e:
            print(f"ERROR: {e}")
            failed += 1
    
    print(f"\n{passed}/{passed+failed} tests passed")
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())


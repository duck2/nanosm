"""Hardware test for fpu_DSP48E1 against software golden model."""
import sys
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, ReadOnly

from test_fpu_soft import (
    golden,
    simulate_fma_core,
    SMOKE_VECTORS_MUL,
    SMOKE_VECTORS_ADD,
    SMOKE_VECTORS_FMA,
    bits_to_f32,
    f32_to_bits,
)

FPU_LATENCY = 15
VERBOSE = False

def debug_print(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

async def reset_dut(dut, cycles: int = 4):
    """Synchronous reset aligned to the clock."""
    dut.rst.value = 1
    for _ in range(cycles):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

def log_stage(stage_name: str, stage_data: dict):
    """Log stage intermediate values."""
    debug_print(f"\n--- {stage_name} ---")
    for key, val in stage_data.items():
        if isinstance(val, int) and val < 16:
            debug_print(f"  {key}: {val}")
        elif isinstance(val, int):
            debug_print(f"  {key}: 0x{val:X}")
        else:
            debug_print(f"  {key}: {val}")

def dump_hw_intermediates(dut):
    """Dump hardware intermediate values in same format as software model."""
    debug_print("\n*** HARDWARE ***")
    debug_print("\n=== FMA CORE (27-bit) ===")
    
    try:
        product48 = int(dut.s3_mult_p.value)
        debug_print(f"  product48=0x{product48:012X}")
    except:
        debug_print(f"  product48=<unavailable>")
    
    try:
        product27 = int(dut.s4_mul27.value)
        exp_hi = int(dut.s3_exp_hi.value)
        debug_print(f"  product27=0x{product27:07X}, exp_hi={exp_hi}")
    except:
        debug_print(f"  product27=<unavailable>")
    
    try:
        shift_amt = int(dut.s4_shift_amt.value)
        exp_hi = int(dut.s3_exp_hi.value)
        exp_lo = int(dut.s3_exp_lo.value)
        debug_print(f"  shift_amt={shift_amt}, exp_hi={exp_hi}, exp_lo={exp_lo}")
    except:
        debug_print(f"  shift_amt=<unavailable>")
    
    try:
        mant_hi27 = int(dut.s5_mant_hi27.value)
        mant_lo27 = int(dut.s5_mant_lo27.value)
        debug_print(f"  mant_hi27_aligned=0x{mant_hi27:07X}, mant_lo27_aligned=0x{mant_lo27:07X}")
    except:
        debug_print(f"  mant_aligned=<unavailable>")
    
    try:
        mant_sum28 = int(dut.s6_mant_pre28.value)
        debug_print(f"  mant_sum28=0x{mant_sum28:07X}")
    except:
        debug_print(f"  mant_sum28=<unavailable>")

async def run_single_fma(dut, in1_bits: int, in2_bits: int, in3_bits: int, test_name: str, use_soft_model: bool = False):
    """Run a single FMA operation through the pipeline and log all stages."""
    debug_print(f"\n{'='*80}")
    debug_print(f"TEST: {test_name}")
    debug_print(f"  in1=0x{in1_bits:08X} ({bits_to_f32(in1_bits)})")
    debug_print(f"  in2=0x{in2_bits:08X} ({bits_to_f32(in2_bits)})")
    debug_print(f"  in3=0x{in3_bits:08X} ({bits_to_f32(in3_bits)})")
    debug_print(f"{'='*80}")
    
    # Run golden model (numpy for mul/add/sub, software model for FMA)
    debug_print("\n*** GOLDEN MODEL ***")
    if use_soft_model:
        sw_result = simulate_fma_core(in1_bits, in2_bits, in3_bits, verbose=VERBOSE)
    else:
        sw_result = golden("fma", in1_bits, in2_bits, in3_bits)

    # Drive inputs and wait for pipeline to flush
    debug_print("\n*** HARDWARE DUT ***")
    await FallingEdge(dut.clk)
    dut.in1.value = in1_bits
    dut.in2.value = in2_bits
    dut.in3.value = in3_bits
    dut.valid_in.value = 1
    
    await RisingEdge(dut.clk)
    dut.valid_in.value = 0
    
    # Wait for pipeline latency (8 cycles) + some margin
    for i in range(18):
        await RisingEdge(dut.clk)
        await ReadOnly()
        
        # Check if valid_out is high
        if dut.valid_out.value == 1:
            hw_result = int(dut.out.value)
            
            # Dump hardware intermediates
            dump_hw_intermediates(dut)
            
            debug_print(f"\nHW result ready at cycle {i}")
            debug_print(f"  HW result: 0x{hw_result:08X} ({bits_to_f32(hw_result)})")
            debug_print(f"  SW result: 0x{sw_result:08X} ({bits_to_f32(sw_result)})")
            
            # Normalize -0 to +0
            if hw_result == 0x80000000:
                hw_result = 0x00000000
            if sw_result == 0x80000000:
                sw_result = 0x00000000
            
            # Compare - expect bit-exact match
            if hw_result == sw_result:
                debug_print(f"\n[PASS]: Bit-exact match!")
                return True
            else:
                print(f"\n[FAIL]: {test_name}")
                print(f"  Expected: 0x{sw_result:08X} ({bits_to_f32(sw_result)})")
                print(f"  Got:      0x{hw_result:08X} ({bits_to_f32(hw_result)})")
                return False
    
    print(f"\n[FAIL]: {test_name} - valid_out never asserted")
    return False

@cocotb.test()
async def test_fpu_mul_ops(dut):
    """Test MUL operations using numpy golden (expect bit-exact)."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Test first few MUL vectors
    passed = 0
    failed = 0
    for test_case in SMOKE_VECTORS_MUL:
        name, op, in1, in2 = test_case
        # Convert MUL to FMA: in1 * in2 + 0
        # Use numpy golden for MUL
        success = await run_single_fma(dut, in1, in2, 0, f"mul_{name}", use_soft_model=False)
        if success:
            passed += 1
        else:
            failed += 1
            break  # Stop on first failure

    print(f"\n{'='*80}")
    print(f"MUL Tests: {passed} passed, {failed} failed")
    print(f"{'='*80}")

    assert failed == 0, f"{failed} test(s) failed"

@cocotb.test()
async def test_fpu_add_ops(dut):
    """Test ADD operations using numpy golden (expect bit-exact)."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)
    
    # Test first few ADD vectors
    passed = 0
    failed = 0
    one_bits = f32_to_bits(1.0)
    
    for test_case in SMOKE_VECTORS_ADD:
        name, op, in1, in2 = test_case
        # Convert ADD to FMA: 1.0 * in1 + in2
        # Convert SUB to FMA: 1.0 * in1 + (-in2)
        # Use numpy golden for ADD/SUB
        if op == "sub":
            in2 = in2 ^ 0x80000000  # Flip sign bit
        success = await run_single_fma(dut, one_bits, in1, in2, f"{op}_{name}", use_soft_model=False)
        if success:
            passed += 1
        else:
            failed += 1
            break  # Stop on first failure
    
    print(f"\n{'='*80}")
    print(f"ADD/SUB Tests: {passed} passed, {failed} failed")
    print(f"{'='*80}")
    
    assert failed == 0, f"{failed} test(s) failed"

async def run_fpu_stream(dut, test_vectors):
    """Drive FPU with one operation per cycle and collect outputs."""
    expected = []
    for vec in test_vectors:
        if len(vec) == 4:
            name, op, in1, in2 = vec
            in3 = 0
            # MUL/ADD/SUB use numpy golden
            expected.append(golden("fma", in1, in2, in3))
        elif len(vec) == 5:
            name, op, in1, in2, in3 = vec
            # FMA uses software model
            expected.append(simulate_fma_core(in1, in2, in3))
        else:
            continue
    
    observed = []
    await RisingEdge(dut.clk)
    
    for vec in test_vectors:
        if len(vec) == 4:
            _, _, in1, in2 = vec
            in3 = 0
        else:
            _, _, in1, in2, in3 = vec
        
        await FallingEdge(dut.clk)
        dut.in1.value = in1
        dut.in2.value = in2
        dut.in3.value = in3
        dut.valid_in.value = 1
        
        await RisingEdge(dut.clk)
        await ReadOnly()
        if dut.valid_out.value == 1:
            observed.append(int(dut.out.value))
        else:
            observed.append(None)
    
    for _ in range(FPU_LATENCY):
        await FallingEdge(dut.clk)
        dut.valid_in.value = 0
        
        await RisingEdge(dut.clk)
        await ReadOnly()
        if dut.valid_out.value == 1:
            observed.append(int(dut.out.value))
        else:
            observed.append(None)
    
    return expected, observed

def check_fpu_stream(expected, observed, test_vectors, latency=FPU_LATENCY):
    """Verify observed stream matches expected with proper latency (bit-exact)."""
    assert len(observed) == len(expected) + latency
    
    mismatches = []
    for i, exp in enumerate(expected):
        got = observed[i + latency]
        
        if got is None:
            mismatches.append((i, exp, None, "no valid_out"))
            continue
        
        exp_norm = 0x00000000 if exp == 0x80000000 else exp
        got_norm = 0x00000000 if got == 0x80000000 else got
        
        if exp_norm != got_norm:
            mismatches.append((i, exp, got, "mismatch"))
    
    if mismatches:
        lines = ["FPU stream mismatch(es):"]
        for i, exp, got, reason in mismatches[:10]:
            # Extract inputs from test vector
            vec = test_vectors[i]
            if len(vec) == 4:
                name, op, in1, in2 = vec
                in3 = 0
            elif len(vec) == 5:
                name, op, in1, in2, in3 = vec
            else:
                in1 = in2 = in3 = 0
            
            if got is None:
                lines.append(f"  idx={i:04d} exp=0x{exp:08X} got=None ({reason})")
            else:
                lines.append(f"  idx={i:04d} exp=0x{exp:08X} got=0x{got:08X} ({reason})")
            lines.append(f"    in1=0x{in1:08X} in2=0x{in2:08X} in3=0x{in3:08X}")
        if len(mismatches) > 10:
            lines.append(f"  ... and {len(mismatches) - 10} more")
        raise AssertionError("\n".join(lines))

@cocotb.test()
async def test_fpu_fma_ops(dut):
    """Test FMA operations using software model (expect bit-exact)."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)
    
    # Test first few FMA vectors
    passed = 0
    failed = 0

    for test_case in SMOKE_VECTORS_FMA:
        name, op, in1, in2, in3 = test_case
        # Use software model for FMA
        success = await run_single_fma(dut, in1, in2, in3, f"fma_{name}", use_soft_model=True)
        if success:
            passed += 1
        else:
            failed += 1
            break  # Stop on first failure
    
    print(f"\n{'='*80}")
    print(f"FMA Tests: {passed} passed, {failed} failed")
    print(f"{'='*80}")
    
    assert failed == 0, f"{failed} test(s) failed"

@cocotb.test()
async def test_fpu_stream_pipeline(dut):
    """Test FPU with back-to-back operations every cycle."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)
    
    test_vectors = SMOKE_VECTORS_MUL + SMOKE_VECTORS_ADD + SMOKE_VECTORS_FMA
    
    print(f"\nRunning {len(test_vectors)} operations in pipelined mode...")
    expected, observed = await run_fpu_stream(dut, test_vectors)
    check_fpu_stream(expected, observed, test_vectors, latency=FPU_LATENCY)
    
    print(f"[PASS] All {len(expected)} pipelined operations completed successfully!")

@cocotb.test()
async def test_fpu_random_stream(dut):
    """Test FPU with 10000 random FMA operations in stream mode."""
    import random
    
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)
    
    # Generate 10000 random test vectors
    rng = random.Random(42)
    num_tests = 10000
    random_vectors = []
    
    print(f"\nGenerating {num_tests} random FMA test vectors...")
    for i in range(num_tests):
        in1 = rng.getrandbits(32)
        in2 = rng.getrandbits(32)
        in3 = rng.getrandbits(32)
        random_vectors.append((f"random_{i}", "fma", in1, in2, in3))
    
    print(f"Running {num_tests} random operations in pipelined mode...")
    expected, observed = await run_fpu_stream(dut, random_vectors)
    check_fpu_stream(expected, observed, random_vectors, latency=FPU_LATENCY)
    
    print(f"[PASS] All {len(expected)} random pipelined operations completed successfully!")

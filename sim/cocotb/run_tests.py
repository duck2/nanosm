"""Test runner for DSP ALU, FPU, multiplier, and shifter tests."""
import os
from cocotb_tools.runner import get_runner

os.environ["COCOTB_RESOLVE_X"] = "ZEROS"

DSP_BUILD_ARGS = [
    "-g2012", "-Y.sv",
    "-y", "rtl/sim_models", "-I", "rtl/sim_models",
    "-DUSE_XST_MODELS", "-DSIMULATION", "-m", "glbl"
]

SIMPLE_BUILD_ARGS = ["-g2012", "-Y.sv", "-DSIMULATION"]

def run_test(toplevel, test_module, verilog_sources, build_dir="sim_build", use_dsp=True):
    """Generic test runner."""
    runner = get_runner("icarus")
    runner.build(
        verilog_sources=verilog_sources,
        hdl_toplevel=toplevel,
        build_dir=build_dir,
        always=True,
        build_args=DSP_BUILD_ARGS if use_dsp else SIMPLE_BUILD_ARGS
    )
    runner.test(hdl_toplevel=toplevel, test_module=test_module, seed=1234)

def run_tests_alu():
    run_test("alu_DSP48E1", "test_alu", [
        "rtl/sim_models/glbl.v",
        "rtl/sim_models/DSP48E1.v",
        "rtl/gpu/alu_DSP48E1.sv"
    ])

def run_tests_fpu():
    run_test("fpu_DSP48E1", "test_fpu", [
        "rtl/sim_models/glbl.v",
        "rtl/sim_models/DSP48E1.v",
        "rtl/gpu/mult_24x24_DSP48E1.sv",
        "rtl/gpu/fpu_DSP48E1.sv",
    ])

def run_tests_fpu_2x():
    run_test("fpu_2x", "test_fpu_2x", [
        "rtl/sim_models/glbl.v",
        "rtl/sim_models/DSP48E1.v",
        "rtl/gpu/mult_24x24_DSP48E1.sv",
        "rtl/gpu/fpu_DSP48E1.sv",
        "rtl/gpu/fpu_2x.sv",
    ])

def run_tests_mult():
    run_test("mult_24x24_DSP48E1", "test_mult", [
        "rtl/sim_models/glbl.v",
        "rtl/sim_models/DSP48E1.v",
        "rtl/gpu/mult_24x24_DSP48E1.sv"
    ])

def run_tests_shifter():
    run_test("shifter", "test_shifter", ["rtl/gpu/shifter.sv"], use_dsp=False)

def run_tests_rf():
    run_test("rf", "test_rf", ["rtl/gpu/rf.sv", "rtl/sim_models/RAM32M.v"], use_dsp=False)

def run_tests_icache():
    run_test("icache", "test_icache", ["rtl/gpu/icache.sv"], use_dsp=False)

def run_tests_lsu():
    run_test("lsu", "test_lsu", ["rtl/gpu/lsu.sv"], use_dsp=False)

def run_tests_sreg():
    run_test("sreg", "test_sreg", ["rtl/gpu/sreg.sv"], use_dsp=False)

def run_tests_ftofx():
    run_test("ftofx", "test_ftofx", ["rtl/gpu/ftofx.sv"], use_dsp=False)

def run_tests_shmem_arbiter():
    run_test("shmem_arbiter", "test_shmem_arbiter", ["rtl/gpu/shmem_arbiter.sv"], use_dsp=False)

if __name__ == "__main__":
    #run_tests_alu()
    #run_tests_fpu()
    #run_tests_fpu_2x()
    #run_tests_mult()
    #run_tests_shifter()
    #run_tests_rf()
    #run_tests_icache()
    #run_tests_lsu()
    #run_tests_ftofx()
    run_tests_shmem_arbiter()

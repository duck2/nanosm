# Cocotb test runner

import os
import sys
from cocotb_tools.runner import get_runner

os.environ["COCOTB_RESOLVE_X"] = "ZEROS"

DSP_BUILD_ARGS = [
    "-g2012", "-Y.sv",
    "-y", "rtl/sim_models", "-I", "rtl/sim_models",
    "-DUSE_XST_MODELS", "-DSIMULATION", "-m", "glbl"
]
SIMPLE_BUILD_ARGS = ["-g2012", "-Y.sv", "-DSIMULATION"]
DSP_SOURCES = ["rtl/sim_models/glbl.v", "rtl/sim_models/DSP48E1.v"]

def run_test(toplevel: str, test_module: str, sources: list[str], use_dsp: bool):
    """Generic test runner."""
    runner = get_runner("icarus")
    runner.build(
        verilog_sources=sources,
        hdl_toplevel=toplevel,
        build_dir="sim_build",
        always=True,
        build_args=DSP_BUILD_ARGS if use_dsp else SIMPLE_BUILD_ARGS
    )
    runner.test(hdl_toplevel=toplevel, test_module=test_module, seed=1234)

# {name: (toplevel, sources, use_dsp)}  test_module is always "test_{name}"
TESTS = {
    "alu": ("alu_DSP48E1", DSP_SOURCES + ["rtl/gpu/alu_DSP48E1.sv"], True),
    "fpu": ("fpu_DSP48E1", DSP_SOURCES + ["rtl/gpu/mult_24x24_DSP48E1.sv", "rtl/gpu/fpu_DSP48E1.sv"], True),
    "fpu_2x": ("fpu_2x", DSP_SOURCES + ["rtl/gpu/mult_24x24_DSP48E1.sv", "rtl/gpu/fpu_DSP48E1.sv", "rtl/gpu/fpu_2x.sv"], True),
    "mult": ("mult_24x24_DSP48E1", DSP_SOURCES + ["rtl/gpu/mult_24x24_DSP48E1.sv"], True),
    "shifter": ("shifter", ["rtl/gpu/shifter.sv"], False),
    "rf": ("rf", ["rtl/gpu/rf.sv", "rtl/sim_models/RAM32M.v"], False),
    #"icache": ("icache", ["rtl/gpu/icache.sv"], False),
    #"ftofx": ("ftofx", ["rtl/gpu/ftofx.sv"], False),
    "shmem_arbiter": ("shmem_arbiter", ["rtl/gpu/shmem_arbiter.sv"], False),
    "shmem": ("shmem", ["rtl/gpu/shmem.sv", "rtl/gpu/shmem_arbiter.sv"], False),
    "decode": ("decode", ["rtl/gpu/decode.sv"], False),
}

if __name__ == "__main__":
    filter_str = sys.argv[1] if len(sys.argv) > 1 else ""
    matched = {k: v for k, v in TESTS.items() if filter_str in k}

    if not matched:
        print(f"No tests matching '{filter_str}'. Available: {list(TESTS.keys())}")
        sys.exit(1)

    print(f"Running {len(matched)} test(s): {list(matched.keys())}")
    for name, (toplevel, sources, use_dsp) in matched.items():
        print(f"\n{'='*60}\nRunning: {name}\n{'='*60}")
        run_test(toplevel, f"test_{name}", sources, use_dsp)

#!/usr/bin/env python3
"""Unified synthesis runner for GPU modules using Vivado."""

import sys
import subprocess
from pathlib import Path

VIVADO_PATH = r"/opt/vivado/2025.2/Vivado/bin/vivado"
WORKSPACE = Path(__file__).parent.parent
BUILD_DIR = WORKSPACE / "build"

MODULES = {
    "alu": {
        "top": "alu_DSP48E1",
        "files": ["rtl/gpu/alu_DSP48E1.sv"],
        "clock_period": 10.0,
    },
    "fpu": {
        "top": "fpu_wrapper",
        "files": ["rtl/gpu/fpu_DSP48E1.sv", "rtl/gpu/mult_24x24_DSP48E1.sv", "scripts/fpu_wrapper.sv"],
        "clock_period": 4.0,
    },
    "mult": {
        "top": "mult_24x24_DSP48E1",
        "files": ["rtl/gpu/mult_24x24_DSP48E1.sv"],
        "clock_period": 10.0,
    },
    "shifter": {
        "top": "shifter_wrapper",
        "files": ["scripts/shifter_wrapper.sv", "rtl/gpu/shifter.sv"],
        "clock_period": 4.0,
    },
    "rf": {
        "top": "rf",
        "files": ["rtl/gpu/rf.sv"],
        "clock_period": 10.0,
    },
    "icache": {
        "top": "icache",
        "files": ["rtl/gpu/icache.sv"],
        "clock_period": 10.0,
    },
    "lsu": {
        "top": "lsu",
        "files": ["rtl/gpu/lsu.sv"],
        "clock_period": 10.0,
    },
    "shmem": {
        "top": "shmem",
        "files": [
            "rtl/gpu/shmem.sv",
            "rtl/gpu/shmem_arbiter.sv"
        ],
        "clock_period": 8.0,
    },
    "shmem_arbiter": {
        "top": "shmem_arbiter",
        "files": ["rtl/gpu/shmem_arbiter.sv"],
        "clock_period": 8.0,
    },
    "cluster": {
        "top": "cluster",
        "files": [
            "rtl/gpu/cluster.sv",
            "rtl/gpu/alu_DSP48E1.sv",
            "rtl/gpu/shifter_2x.sv",
            "rtl/gpu/shifter.sv",
            "rtl/gpu/fpu_2x.sv",
            "rtl/gpu/fpu_DSP48E1.sv",
            "rtl/gpu/mult_24x24_DSP48E1.sv",
            "rtl/gpu/shmem.sv",
            "rtl/gpu/rf.sv",
        ],
        "clock_period": 8.0,
        "clocks": {
            "clk": {"period": 8.0},
            "clk_2x": {"period": 4.0, "source": "clk", "multiply_by": 2},
        },
    },
    "fpu_2x": {
        "top": "fpu_2x_wrapper",
        "files": [
            "scripts/fpu_2x_wrapper.sv",
            "rtl/gpu/fpu_2x.sv",
            "rtl/gpu/fpu_DSP48E1.sv",
            "rtl/gpu/mult_24x24_DSP48E1.sv",
        ],
        "clock_period": 8.0,  # clk period (125 MHz)
        "clocks": {
            "clk": {"period": 8.0},
            "clk_2x": {"period": 4.0, "source": "clk", "multiply_by": 2},
        },
    },
    "top": {
        "top": "top",
        "files": [
            "rtl/top.sv",
            "rtl/gpu/cluster.sv",
            "rtl/gpu/alu_DSP48E1.sv",
            "rtl/gpu/shifter_2x.sv",
            "rtl/gpu/shifter.sv",
            "rtl/gpu/fpu_2x.sv",
            "rtl/gpu/fpu_DSP48E1.sv",
            "rtl/gpu/mult_24x24_DSP48E1.sv",
            "rtl/gpu/shmem.sv",
            "rtl/gpu/rf.sv",
        ],
        "clock_period": 10.0,  # clk100 input period (100 MHz)
        "is_top": True,
        "clocks": {
            "clk100": {"period": 10.0, "port": "clk100"},
        },
        "reset_port": "rst_n",
    },
}

def create_tcl_script(module_cfg: dict, report_prefix: str, clock_period: float, run_pnr: bool) -> Path:
    """Create TCL script for Vivado synthesis."""
    rtl_files = [str((WORKSPACE / f).resolve()).replace('\\', '/') for f in module_cfg["files"]]
    build_dir = str(BUILD_DIR.resolve()).replace('\\', '/')
    
    xdc_path = BUILD_DIR / "constraints.xdc"
    
    # Build clock constraints
    clocks_cfg = module_cfg.get("clocks")
    if clocks_cfg:
        clock_lines = []
        for clk_name, clk_cfg in clocks_cfg.items():
            if "source" in clk_cfg:
                # Generated/derived clock
                src = clk_cfg["source"]
                mult = clk_cfg.get("multiply_by", 1)
                div = clk_cfg.get("divide_by", 1)
                clock_lines.append(
                    f"create_generated_clock -name {clk_name} "
                    f"-source [get_ports {src}] "
                    f"-multiply_by {mult} -divide_by {div} "
                    f"[get_ports {clk_name}]"
                )
            else:
                # Primary clock
                period = clk_cfg["period"]
                clock_lines.append(f"create_clock -period {period} -name {clk_name} [get_ports {clk_name}]")
        clock_constraints = "\n".join(clock_lines)
        primary_clk = next(k for k, v in clocks_cfg.items() if "source" not in v)
    else:
        clock_constraints = f"create_clock -period {clock_period} -name clk [get_ports clk]"
        primary_clk = "clk"
    
    reset_port = module_cfg.get("reset_port", "rst")
    is_top = module_cfg.get("is_top", False)
    
    if is_top:
        xdc_path.write_text(f'''
# Clock constraint for board input clock
{clock_constraints}

# MMCM generates clk125 and clk250 internally - Vivado derives these automatically

# Ignore reset in timing analysis
set_false_path -from [get_ports {reset_port}]
''')
    else:
        xdc_path.write_text(f'''
# Clock constraints
{clock_constraints}

# Ignore resets in timing analysis
set_false_path -from [get_ports {reset_port}]

# Input/output delays
set_input_delay -clock {primary_clk} 0.000 [all_inputs]
''')

    file_type = "Verilog" if module_cfg.get("is_verilog") else "SystemVerilog"
    file_ext = "*.v" if module_cfg.get("is_verilog") else "*.sv"
    
    pnr_commands = ""
    if run_pnr:
        pnr_commands = f"""
# Run placement and routing
place_design
route_design

# Generate post-route reports
report_timing_summary -file {report_prefix}_timing_routed.rpt -delay_type min_max -report_unconstrained -check_timing_verbose -max_paths 10 -input_pins -routable_nets
report_utilization -hierarchical -file {report_prefix}_utilization_routed.rpt
"""
    
    top = module_cfg["top"]
    is_top = module_cfg.get("is_top", False)
    synth_mode = "" if is_top else "-mode out_of_context"
    
    tcl_content = f"""
create_project -force {top} "{build_dir}" -part xc7a35ticsg324-1L
add_files {{{" ".join(rtl_files)}}}
add_files -fileset constrs_1 "{str(xdc_path).replace('\\', '/')}"
set_property file_type {file_type} [get_files {file_ext}]
set_property top {top} [get_filesets sources_1]

# Run synthesis
synth_design -top {top} -flatten_hierarchy rebuilt {synth_mode}

# Run optimization passes
opt_design -directive Explore

# Generate detailed reports
report_timing_summary -file {report_prefix}_timing.rpt -delay_type min_max -report_unconstrained -check_timing_verbose -max_paths 10 -input_pins -routable_nets
report_utilization -hierarchical -file {report_prefix}_utilization.rpt
report_drc -file {report_prefix}_drc.rpt

# Print summary to console
puts "\\nResource Usage Summary:"
puts "--------------------------------"
puts [report_utilization -cells [get_cells] -return_string]

{pnr_commands}

quit
"""
    script_path = BUILD_DIR / "synth.tcl"
    script_path.write_text(tcl_content)
    return script_path

def run_synth(module_name: str, run_pnr: bool = False):
    """Run Vivado synthesis for a module."""
    if module_name not in MODULES:
        print(f"Unknown module: {module_name}")
        print(f"Available modules: {', '.join(MODULES.keys())}")
        return
    
    BUILD_DIR.mkdir(exist_ok=True)
    
    module_cfg = MODULES[module_name]
    report_prefix = module_cfg["top"]
    clock_period = module_cfg["clock_period"]
    tcl_script = create_tcl_script(module_cfg, report_prefix, clock_period, run_pnr)
    
    print(f"\nSynthesizing {module_cfg['top']}...")
    if run_pnr:
        print("Also running placement and routing...")
    
    subprocess.run([VIVADO_PATH, "-mode", "batch", "-source", str(tcl_script)], cwd=BUILD_DIR)
    
    # Copy reports to workspace root
    suffixes = ['timing.rpt', 'utilization.rpt', 'drc.rpt']
    if run_pnr:
        suffixes.extend(['timing_routed.rpt', 'utilization_routed.rpt'])
    
    for suffix in suffixes:
        src = BUILD_DIR / f"{report_prefix}_{suffix}"
        dst = WORKSPACE / f"{report_prefix}_{suffix}"
        if src.exists():
            dst.write_text(src.read_text())
            print(f"Generated report: {dst}")

if __name__ == "__main__":
    run_pnr = "--pnr" in sys.argv or "--route" in sys.argv or "--place" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    
    if not args:
        print("Usage: python synth.py <module> [--pnr]")
        print(f"Available modules: {', '.join(MODULES.keys())}")
        sys.exit(1)
    
    run_synth(args[0], run_pnr)

#!/usr/bin/env python3
"""Unified synthesis runner for GPU modules using Vivado."""

import sys
import os
import glob
import shutil
import subprocess
import json
import re
import copy
from pathlib import Path


def _find_vivado() -> str:
    """Locate the Vivado launcher robustly across environments."""
    # 1. Explicit override
    env = os.environ.get("VIVADO_PATH")
    if env and Path(env).exists():
        return env
    # 2. Known-good Windows install (has Artix-7 part data)
    preferred = r"C:\Xilinx\Vivado\2024.2\bin\vivado.bat"
    if Path(preferred).exists():
        return preferred
    # 3. Any Windows install, newest first
    for cand in sorted(glob.glob(r"C:\Xilinx\Vivado\*\bin\vivado.bat"), reverse=True):
        return cand
    # 4. Common Linux install, newest first
    for cand in sorted(glob.glob("/opt/[Xx]ilinx/Vivado/*/bin/vivado"), reverse=True):
        if Path(cand).exists():
            return cand
    # 5. On PATH
    found = shutil.which("vivado") or shutil.which("vivado.bat")
    if found:
        return found
    raise FileNotFoundError(
        "Could not locate Vivado. Set the VIVADO_PATH environment variable to the "
        "vivado(.bat) launcher."
    )


VIVADO_PATH = _find_vivado()
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

STAGES_TCL = r'''
# Per-pipeline-stage critical path (stage registers must be named <tag>_*)
set _stages {%STAGES%}
set _fh [open "%PREFIX%_stages.rpt" w]
puts $_fh "from to slack_ns levels datapath_ns"
for {set i 0} {$i < [expr {[llength $_stages]-1}]} {incr i} {
    set _f [lindex $_stages $i]
    set _t [lindex $_stages [expr {$i+1}]]
    set _src [get_cells -quiet -hier -regexp ".*${_f}_.*"]
    set _dst [get_cells -quiet -hier -regexp ".*${_t}_.*"]
    if {[llength $_src]==0 || [llength $_dst]==0} { puts $_fh "$_f $_t NA NA NA"; continue }
    set _p [get_timing_paths -quiet -from $_src -to $_dst -max_paths 1]
    if {[llength $_p]==0} { puts $_fh "$_f $_t NA NA NA"; continue }
    set _p [lindex $_p 0]
    puts $_fh "$_f $_t [get_property SLACK $_p] [get_property LOGIC_LEVELS $_p] [get_property DATAPATH_DELAY $_p]"
}
close $_fh
'''


def create_tcl_script(module_cfg: dict, report_prefix: str, clock_period: float, run_pnr: bool, stages=None) -> Path:
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
    
    stages_block = ""
    if run_pnr and stages:
        stages_block = STAGES_TCL.replace("%STAGES%", " ".join(stages)).replace("%PREFIX%", report_prefix)

    pnr_commands = ""
    if run_pnr:
        pnr_commands = f"""
# Run placement and routing
place_design
route_design

# Generate post-route reports
report_timing_summary -file {report_prefix}_timing_routed.rpt -delay_type min_max -report_unconstrained -check_timing_verbose -max_paths 10 -input_pins -routable_nets
report_utilization -hierarchical -file {report_prefix}_utilization_routed.rpt
report_utilization -file {report_prefix}_util_flat.rpt
{stages_block}
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

def _util_int(text, label):
    """First integer in a flat report_utilization row labeled `label`."""
    m = re.search(r"\|\s*" + re.escape(label) + r"\s*\*?\s*\|\s*(\d+)\s*\|", text)
    return int(m.group(1)) if m else None


def extract_summary(report_prefix, clock_period):
    """Parse post-route reports into a compact dict so an agent reads numbers, not raw .rpt text."""
    timing = WORKSPACE / f"{report_prefix}_timing_routed.rpt"
    flat = WORKSPACE / f"{report_prefix}_util_flat.rpt"
    s = {"top": report_prefix, "clock_period_ns": clock_period}

    if timing.exists():
        t = timing.read_text(errors="ignore")
        idx = t.find("Design Timing Summary")
        seg = t[idx:idx + 2000] if idx >= 0 else ""
        m = re.search(r"\n\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+\d+", seg)
        if m:
            s["wns_ns"] = float(m.group(1))
            if clock_period:
                s["fmax_mhz"] = round(1000.0 / (clock_period - s["wns_ns"]), 1)
        j = t.find("Slack (")
        blk = t[j:j + 1500] if j >= 0 else ""
        cp = re.search(
            r"Source:\s*(\S+).*?Destination:\s*(\S+).*?"
            r"Data Path Delay:\s*([\d.]+)ns\s*\(logic[^()]*\(([\d.]+)%\)\s*route[^()]*\(([\d.]+)%\)\).*?"
            r"Logic Levels:\s*(\d+)",
            blk, re.S)
        if cp:
            s["critical_path"] = {
                "source": cp.group(1), "dest": cp.group(2),
                "datapath_ns": float(cp.group(3)),
                "pct_logic": float(cp.group(4)), "pct_route": float(cp.group(5)),
                "logic_levels": int(cp.group(6)),
            }

    if flat.exists():
        u = flat.read_text(errors="ignore")
        util = {
            "slice_luts": _util_int(u, "Slice LUTs"),
            "lut_as_logic": _util_int(u, "LUT as Logic"),
            "lut_as_memory": _util_int(u, "LUT as Memory"),
            "slice_registers": _util_int(u, "Slice Registers"),
            "slices": _util_int(u, "Slice"),
            "dsps": _util_int(u, "DSPs"),
            "block_ram_tile": _util_int(u, "Block RAM Tile"),
        }
        if util["slice_luts"] and util["slice_registers"] is not None:
            util["ff_lut_ratio"] = round(util["slice_registers"] / util["slice_luts"], 2)
        s["util"] = util

    sfile = WORKSPACE / f"{report_prefix}_stages.rpt"
    if sfile.exists():
        rows = []
        for ln in sfile.read_text(errors="ignore").splitlines()[1:]:
            p = ln.split()
            if len(p) >= 5:
                rows.append({"from": p[0], "to": p[1],
                             "slack_ns": None if p[2] == "NA" else float(p[2]),
                             "levels": None if p[3] == "NA" else int(p[3]),
                             "datapath_ns": None if p[4] == "NA" else float(p[4])})
        if rows:
            s["stages"] = rows
    return s


def run_synth(module_name: str, run_pnr: bool = False, emit_json: bool = False, clk_override: float = None, stages=None):
    """Run Vivado synthesis for a module."""
    if module_name not in MODULES:
        print(f"Unknown module: {module_name}")
        print(f"Available modules: {', '.join(MODULES.keys())}")
        return

    BUILD_DIR.mkdir(exist_ok=True)

    module_cfg = MODULES[module_name]
    if clk_override is not None:
        # Overconstrain to characterize fmax: tighten the clock and read where it lands.
        module_cfg = copy.deepcopy(module_cfg)
        module_cfg["clock_period"] = clk_override
        for ccfg in module_cfg.get("clocks", {}).values():
            if "source" not in ccfg:  # primary clock(s); derived clocks track via multiply/divide
                ccfg["period"] = clk_override
    report_prefix = module_cfg["top"]
    clock_period = module_cfg["clock_period"]
    tcl_script = create_tcl_script(module_cfg, report_prefix, clock_period, run_pnr, stages)
    
    print(f"\nSynthesizing {module_cfg['top']}...")
    if run_pnr:
        print("Also running placement and routing...")
    
    subprocess.run([VIVADO_PATH, "-mode", "batch", "-source", str(tcl_script)], cwd=BUILD_DIR)
    
    # Copy reports to workspace root
    suffixes = ['timing.rpt', 'utilization.rpt', 'drc.rpt']
    if run_pnr:
        suffixes.extend(['timing_routed.rpt', 'utilization_routed.rpt', 'util_flat.rpt', 'stages.rpt'])

    for suffix in suffixes:
        src = BUILD_DIR / f"{report_prefix}_{suffix}"
        dst = WORKSPACE / f"{report_prefix}_{suffix}"
        if src.exists():
            dst.write_text(src.read_text())
            print(f"Generated report: {dst}")

    if emit_json and run_pnr:
        summary = extract_summary(report_prefix, clock_period)
        (WORKSPACE / f"{report_prefix}_summary.json").write_text(json.dumps(summary, indent=2))
        print("\n=== SUMMARY_JSON ===")
        print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    run_pnr = "--pnr" in sys.argv or "--route" in sys.argv or "--place" in sys.argv
    emit_json = "--json" in sys.argv
    if emit_json:
        run_pnr = True  # summary is post-route only
    clk_override = None
    stages = None
    for a in sys.argv[1:]:
        if a.startswith("--clk="):
            clk_override = float(a.split("=", 1)[1])
        elif a.startswith("--stages="):
            stages = a.split("=", 1)[1].split(",")
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if not args:
        print("Usage: python synth.py <module> [--pnr] [--json] [--clk=<ns>] [--stages=s1,s2,...]")
        print("  --clk=<ns>        constrain the clock; set tight (e.g. 1) -> fmax = 1/(period-WNS) in ONE run")
        print("  --stages=s1,s2,.. per-pipeline-stage critical paths (stage regs named <tag>_*)")
        print(f"Available modules: {', '.join(MODULES.keys())}")
        sys.exit(1)

    run_synth(args[0], run_pnr, emit_json, clk_override, stages)

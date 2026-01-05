#!/usr/bin/env python3
"""Pre-packaged smoke verifier for an fp32 FMA core (in1*in2 + in3).

Usage:  python scripts/verify_fma.py <dut.sv> [extra.sv ...]

Checks a DUT named `fpu_DSP48E1` against a correctly-rounded fp32 FMA reference
over a fixed ~30-vector smoke suite, allowing up to TOL_ULP of error (the design
is ~27-bit internal, best-effort rounding — NOT IEEE-exact). Prints a compact
PASS/FAIL line. No testbench authoring, no sim logs to read.

DUT contract: module `fpu_DSP48E1 (input clk, rst, input [31:0] in1,in2,in3,
input valid_in, output [31:0] out, output valid_out)`, any fixed latency < 30.

Requires: iverilog/vvp on PATH, numpy.
"""

import sys
import subprocess
import tempfile
import shutil
import re
from pathlib import Path

import numpy as np

TOP = "fpu_DSP48E1"
TOL_ULP = 4  # 27-bit internal, best-effort rounding: allow a few fp32 ULPs

# Fixed smoke suite as (a, b, c) fp32 floats for a*b + c, chosen to avoid
# catastrophic cancellation so a clean PASS means "basically works".
VECTORS = [
    (1, 1, 0), (2, 3, 1), (1.5, 2, 0.5), (1, 0, 5), (0, 7, 3),
    (2, 2, 2), (-1, 1, 1), (3, -2, 6), (10, 0.5, 1), (1.25, 4, -1),
    (100, 0.01, 0), (-2, -3, -1), (0.5, 0.5, 0.5), (8, 8, 0), (1, 1, 1),
    (16, 16, 16), (3, 3, 3), (-4, 2, 10), (2.5, 4, 5), (7, 0, 0),
    (1, -1, 2), (6, 6, 4), (0.25, 8, 1), (-5, 2, 20), (1.5, 1.5, 1.5),
    (2, -0.5, 3), (9, 9, -1), (4, 0.25, 0.5), (12, 2, 1), (123.5, 2, 0.5),
]


def bits(x) -> int:
    return int(np.float32(x).view(np.uint32))


def fma_ref(a_bits, b_bits, c_bits) -> int:
    a = np.uint32(a_bits).view(np.float32)
    b = np.uint32(b_bits).view(np.float32)
    c = np.uint32(c_bits).view(np.float32)
    # product of two float32 is exact in float64 (48 < 52 mantissa bits)
    with np.errstate(over="ignore", invalid="ignore"):
        r = np.float32(np.float64(a) * np.float64(b) + np.float64(c))
    return int(np.float32(r).view(np.uint32))


def _mono(u: int) -> int:
    """Map fp32 bits to a monotonic ordering so adjacent floats differ by 1."""
    return (~u & 0xFFFFFFFF) if (u & 0x80000000) else (u | 0x80000000)


def ulp_dist(x: int, y: int) -> int:
    return abs(_mono(x) - _mono(y))


def is_zero(b: int) -> bool:
    return (b & 0x7FFFFFFF) == 0


def cancellation_vbits(n=12, seed=7):
    """Adversarial FMA vectors: c = -round(a*b). A TRUE fused FMA outputs the exact
    sub-ULP product residual; a truncated-intermediate design is off by thousands of
    ULP; a non-fused mul-then-add outputs 0. Only a true fused FMA matches fma_ref."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        a = np.float32(rng.uniform(1.0, 2.0)) * np.float32(2.0) ** int(rng.integers(-3, 4))
        b = np.float32(rng.uniform(1.0, 2.0)) * np.float32(2.0) ** int(rng.integers(-3, 4))
        a = np.float32(a); b = np.float32(b)
        rp = np.float32(np.float64(a) * np.float64(b))
        c = np.float32(-rp)
        with np.errstate(over="ignore", invalid="ignore"):
            true = np.float32(np.float64(a) * np.float64(b) + np.float64(c))
        if (int(np.float32(true).view(np.uint32)) & 0x7FFFFFFF) == 0:
            continue  # product exact -> not distinguishing
        out.append((bits(a), bits(b), bits(c)))
    return out


def build_tb(vecs, top=TOP):
    n = len(vecs)
    init = "\n".join(
        f"    a1[{i}]=32'h{a:08x}; a2[{i}]=32'h{b:08x}; a3[{i}]=32'h{c:08x};"
        for i, (a, b, c) in enumerate(vecs)
    )
    return f"""`timescale 1ns/1ps
module tb;
  reg clk=0, rst=1, valid_in=0;
  reg [31:0] in1=0, in2=0, in3=0;
  wire [31:0] out;
  wire valid_out;
  {top} dut(.clk(clk), .rst(rst), .in1(in1), .in2(in2), .in3(in3),
            .valid_in(valid_in), .out(out), .valid_out(valid_out));
  always #5 clk = ~clk;
  integer i;
  reg [31:0] a1 [0:{n-1}];
  reg [31:0] a2 [0:{n-1}];
  reg [31:0] a3 [0:{n-1}];
  initial begin
{init}
    rst=1; valid_in=0; repeat(4) @(posedge clk); rst=0;
    valid_in=1;
    for (i=0;i<{n};i=i+1) begin
      in1=a1[i]; in2=a2[i]; in3=a3[i];
      repeat(30) @(posedge clk);   // hold; pipeline fills with this op
      $display("R %0d %08x", i, out);
    end
    $finish;
  end
endmodule
"""


def main():
    if len(sys.argv) < 2:
        print("usage: verify_fma.py <dut.sv> [extra.sv ...]")
        sys.exit(2)
    top = TOP
    fused = "--fused" in sys.argv
    tol = 1 if fused else TOL_ULP   # true fused = correctly rounded (<=1 ULP everywhere)
    for a in sys.argv[1:]:
        if a.startswith("--top="):
            top = a.split("=", 1)[1]
        elif a.startswith("--tol="):
            tol = int(a.split("=", 1)[1])   # explicit override wins
    files = [str(Path(p).resolve()) for p in sys.argv[1:] if not p.startswith("--")]
    for f in files:
        if not Path(f).exists():
            print(f"FAIL: source not found: {f}")
            sys.exit(2)

    vb = [(bits(a), bits(b), bits(c)) for a, b, c in VECTORS]
    if fused:
        vb += cancellation_vbits()   # only a true fused FMA survives these
    expected = [fma_ref(*v) for v in vb]

    work = Path(tempfile.mkdtemp(prefix="vfma_"))
    try:
        tb = work / "tb.v"
        tb.write_text(build_tb(vb, top))
        vvp = work / "sim.vvp"
        c = subprocess.run(["iverilog", "-g2012", "-o", str(vvp), str(tb), *files],
                           capture_output=True, text=True)
        if c.returncode != 0:
            print("FAIL: iverilog compile error:")
            print((c.stderr or c.stdout).strip()[:1500])
            sys.exit(1)
        r = subprocess.run(["vvp", str(vvp)], capture_output=True, text=True)
        got = {}
        for m in re.finditer(r"^R (\d+) ([0-9a-fA-F]{8})", r.stdout, re.M):
            got[int(m.group(1))] = int(m.group(2), 16)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    fails = []
    worst = 0
    for i, (v, exp) in enumerate(zip(vb, expected)):
        g = got.get(i)
        if g is None:
            fails.append((i, None, exp, "no output"))
        elif is_zero(g) and is_zero(exp):
            continue
        else:
            d = ulp_dist(g, exp)
            worst = max(worst, d)
            if d > tol:
                fails.append((i, g, exp, f"{d} ulp"))

    n = len(vb)
    mode = "fused-strict" if fused else "smoke"
    if not fails:
        print(f"PASS {n}/{n} (worst {worst} ulp, tol {tol}, {mode})")
        sys.exit(0)
    print(f"FAIL {n - len(fails)}/{n} (tol {tol}, {mode})")
    for i, g, exp, why in fails[:8]:
        a, b, cc = vb[i]
        gs = "--------" if g is None else f"{g:08x}"
        print(f"  [{i}] a={a:08x} b={b:08x} c={cc:08x}: got={gs} want={exp:08x} ({why})")
    sys.exit(1)


if __name__ == "__main__":
    main()

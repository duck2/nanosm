#!/usr/bin/env python3
"""Pre-packaged smoke verifier for an fp16 (IEEE binary16) add/sub core.

Usage:  python scripts/verify_fp16.py <path/to/fp16_add.sv>

Compares the DUT against a numpy float16 FTZ/DAZ reference over a fixed ~25-vector
smoke suite and prints a compact PASS/FAIL line (no testbench authoring, no sim
logs to read). DUT contract: module `fp16_add (input clk, rst, input [15:0] a, b,
output [15:0] result)`, any fixed pipeline latency (< 20 cycles), no handshake.

Requires: iverilog/vvp on PATH, numpy.
"""

import sys
import subprocess
import tempfile
import shutil
import re
from pathlib import Path

import numpy as np

# Fixed smoke suite as (a, b) Python floats — all chosen so the exact fp16 result
# is unambiguous (no ULP-edge rounding), so a clean PASS means "basically works".
VECTORS = [
    (1.0, 1.0), (1.0, 2.0), (2.5, -1.5), (0.5, 0.5), (100.0, 0.5),
    (3.0, 0.0), (0.0, 7.0), (5.0, -5.0), (-0.0, 0.0), (-3.0, -4.0),
    (65504.0, 65504.0), (1.0, -1.0), (8.0, 8.0), (0.25, 0.75), (10.0, -2.5),
    (1024.0, 1.0), (-0.5, -0.5), (6.0, -6.0), (2.0, 2.0), (0.125, 0.125),
    (12.5, 3.25), (7.5, -0.25), (33.0, -1.0), (-100.0, 50.0), (0.75, 0.75),
]

SUBNORMAL_MIN = 2.0 ** -14  # smallest normal fp16


def _daz(x: np.float16) -> np.float16:
    """Flush subnormals to a signed zero (denormals-are-zero / flush-to-zero)."""
    f = float(x)
    if f != 0.0 and abs(f) < SUBNORMAL_MIN and np.isfinite(f):
        return np.float16(np.copysign(0.0, f))
    return x


def bits(x) -> int:
    return int(np.float16(x).view(np.uint16))


def fp16_ref(a_bits: int, b_bits: int) -> int:
    a = _daz(np.uint16(a_bits).view(np.float16))
    b = _daz(np.uint16(b_bits).view(np.float16))
    with np.errstate(over="ignore", invalid="ignore"):  # overflow→Inf is intended
        r = np.float16(np.float32(a) + np.float32(b))    # add wide, round-nearest-even
    r = _daz(r)                                          # FTZ the result
    return int(np.float16(r).view(np.uint16))


def is_zero(b: int) -> bool:
    return (b & 0x7FFF) == 0


def build_tb(vecs):
    n = len(vecs)
    av = "\n".join(f"    av[{i}]=16'h{a:04x}; bv[{i}]=16'h{b:04x};" for i, (a, b) in enumerate(vecs))
    return f"""`timescale 1ns/1ps
module tb;
  reg clk=0, rst=1;
  reg [15:0] a=0, b=0;
  wire [15:0] result;
  fp16_add dut(.clk(clk), .rst(rst), .a(a), .b(b), .result(result));
  always #5 clk = ~clk;
  integer i;
  reg [15:0] av [0:{n-1}];
  reg [15:0] bv [0:{n-1}];
  initial begin
{av}
    rst=1; repeat(4) @(posedge clk); rst=0;
    for (i=0;i<{n};i=i+1) begin
      a=av[i]; b=bv[i];
      repeat(20) @(posedge clk);
      $display("R %0d %04x", i, result);
    end
    $finish;
  end
endmodule
"""


def main():
    if len(sys.argv) < 2:
        print("usage: verify_fp16.py <fp16_add.sv>")
        sys.exit(2)
    dut = Path(sys.argv[1]).resolve()
    if not dut.exists():
        print(f"FAIL: DUT not found: {dut}")
        sys.exit(2)

    vecs_bits = [(bits(a), bits(b)) for a, b in VECTORS]
    expected = [fp16_ref(ab, bb) for ab, bb in vecs_bits]

    work = Path(tempfile.mkdtemp(prefix="vfp16_"))
    try:
        tb = work / "tb.v"
        tb.write_text(build_tb(vecs_bits))
        vvp = work / "sim.vvp"
        c = subprocess.run(["iverilog", "-g2012", "-o", str(vvp), str(tb), str(dut)],
                           capture_output=True, text=True)
        if c.returncode != 0:
            print("FAIL: iverilog compile error:")
            print((c.stderr or c.stdout).strip()[:1500])
            sys.exit(1)
        r = subprocess.run(["vvp", str(vvp)], capture_output=True, text=True)
        got = {}
        for m in re.finditer(r"^R (\d+) ([0-9a-fA-F]{4})", r.stdout, re.M):
            got[int(m.group(1))] = int(m.group(2), 16)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    fails = []
    for i, ((ab, bb), exp) in enumerate(zip(vecs_bits, expected)):
        g = got.get(i)
        if g is None:
            fails.append((i, ab, bb, None, exp))
        elif g != exp and not (is_zero(g) and is_zero(exp)):  # +0/-0 treated equal
            fails.append((i, ab, bb, g, exp))

    n = len(vecs_bits)
    if not fails:
        print(f"PASS {n}/{n}")
        sys.exit(0)
    print(f"FAIL {n - len(fails)}/{n}")
    for i, ab, bb, g, exp in fails[:6]:
        af, bf = VECTORS[i]
        gs = "----" if g is None else f"{g:04x}"
        print(f"  [{i}] {af}+{bf}: a={ab:04x} b={bb:04x} got={gs} want={exp:04x}")
    sys.exit(1)


if __name__ == "__main__":
    main()

"""Double-buffered staged vector add without TMA: warp 0 stages A/B through the RF
into shmem, warp 1 adds and stores C. Checks C = A + B.

Run from emu2: python -B -m nanosm kernels/pipeline.launch.py [--trace PATH]
No trace is written unless --trace names one: a fixed output path would be
shared by every overlapping run (ctest -j, the sanitizer ON/OFF check,
build and build_nosan at once).
"""

import argparse
from pathlib import Path

from nanosm import Launch


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--trace", type=Path, help="Perfetto output (default: no trace)")
args = parser.parse_args()

ROOT = Path(__file__).resolve().parents[1]
A = 0
B = 4096
C = 8192
ELEMENTS = 8 * 16

# The kernel's signature is ptr A, ptr B, ptr C; its two warps and its four
# barriers are declared in the qtx.
launch = Launch(
    kernel=Path(__file__).with_suffix("").with_suffix(".qtx"),
    profile=ROOT / "profiles" / "nanosm-v0.json",
    args={"A": A, "B": B, "C": C},
)
launch.gmem.f32(A, (float(i) for i in range(ELEMENTS)))
launch.gmem.f32(B, (float(2 * i) for i in range(ELEMENTS)))

with launch.run(trace=args.trace) as result:
    expected = [float(3 * i) for i in range(ELEMENTS)]
    actual = result.gmem_f32(C, ELEMENTS)
    if actual != expected:
        raise RuntimeError("pipeline output mismatch")
    print("pipeline ok")
    print(result.summary, end="")

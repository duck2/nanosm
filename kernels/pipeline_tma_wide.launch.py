"""The TMA pipeline with 64-element (8-sector) tiles: the 16-sector credit pool
binds. Checks C = A + B.

Run from emu2: python -B -m nanosm kernels/pipeline_tma_wide.launch.py [--trace PATH]
No trace is written unless --trace names one: a fixed output path would be
shared by every overlapping run (ctest -j, the sanitizer ON/OFF check,
build and build_nosan at once).
"""

import argparse
from pathlib import Path

from nanosm import Launch, tma_desc


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--trace", type=Path, help="Perfetto output (default: no trace)")
args = parser.parse_args()

ROOT = Path(__file__).resolve().parents[1]
A = 0
B = 4096
C = 8192
ELEMENTS = 8 * 64


def tensor(base: int) -> bytes:
    # One descriptor per input tensor (a dense one-dimensional tile of 64
    # elements); the stage's buffer and full barrier are the instruction's.
    return tma_desc(base=base, dimensions=(ELEMENTS,), tile=(64,), layout="dense")


# The kernel's signature is desc ADESC, desc BDESC, ptr C; its two warps and
# its four barriers are declared in the qtx.
launch = Launch(
    kernel=Path(__file__).with_suffix("").with_suffix(".qtx"),
    profile=ROOT / "profiles" / "nanosm-v0.json",
    args={"ADESC": tensor(A), "BDESC": tensor(B), "C": C},
)
launch.gmem.f32(A, (float(i) for i in range(ELEMENTS)))
launch.gmem.f32(B, (float(2 * i) for i in range(ELEMENTS)))

with launch.run(trace=args.trace) as result:
    expected = [float(3 * i) for i in range(ELEMENTS)]
    actual = result.gmem_f32(C, ELEMENTS)
    if actual != expected:
        raise RuntimeError("wide tma pipeline output mismatch")
    print("wide tma pipeline ok")
    print(result.summary, end="")

"""The TMA pipeline with 64-element (8-sector) tiles: the 16-sector credit pool
binds. Checks C = A + B.

Run from emu2: python -B -m nanosm kernels/pipeline_tma_wide.launch.py [--trace PATH]
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
FULL0 = 1024
FULL1 = 1032
EMPTY0 = 1040
EMPTY1 = 1048
ADESC = 1152
BDESC = 1216
ELEMENTS = 8 * 64

launch = Launch(
    kernel=Path(__file__).with_suffix("").with_suffix(".qtx"),
    profile=ROOT / "profiles" / "nanosm-v0.json",
    warps=2,
)
launch.gmem.f32(A, (float(i) for i in range(ELEMENTS)))
launch.gmem.f32(B, (float(2 * i) for i in range(ELEMENTS)))
for address in [FULL0, FULL1, EMPTY0, EMPTY1]:
    launch.shmem.mbar(address, expected=1)
# One descriptor per tensor; the stage's destination and full barrier are
# the instruction's address operand (built in the kernel).
for address, base in [(ADESC, A), (BDESC, B)]:
    launch.shmem.tma_desc(
        address,
        gmem_base=base,
        dimensions=(ELEMENTS,),
        box=(64,),
    )

with launch.run(trace=args.trace) as result:
    expected = [float(3 * i) for i in range(ELEMENTS)]
    actual = result.gmem_f32(C, ELEMENTS)
    if actual != expected:
        raise RuntimeError("wide tma pipeline output mismatch")
    print("wide tma pipeline ok")
    print(result.summary, end="")

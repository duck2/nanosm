"""Per-lane Collatz step counts with masked steps (@p add, @!p shri), four warps.

Run from emu2: python -B -m nanosm kernels/diverge_masked.launch.py
Options: --profile PATH, --trace PATH (Perfetto output).

The same inputs and check as kernels/diverge.launch.py; only the kernel
differs: each step's odd / even choice is two masked arms in straight-line
code (docs/qtx.md, "Masks") instead of ssy / @p bra / sync / sync. The
loop still sheds lanes with the divergence stack. Reports the cycle count
(not pinned).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import struct

from nanosm import Launch


ROOT = Path(__file__).resolve().parents[1]
IN, OUT = 0, 1024
WARPS = 4
N = 16 * WARPS
INPUTS = [1 + (37 * i) % 64 for i in range(N)]


def collatz_steps(n: int) -> int:
    steps = 0
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=ROOT / "profiles" / "nanosm-v0.json")
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()

    kernel = Path(__file__).with_suffix("").with_suffix(".qtx")
    launch = Launch(kernel=kernel, profile=args.profile, warps=WARPS, args={"IN": IN, "OUT": OUT})
    launch.gmem.u32(IN, INPUTS)
    launch.gmem.write(OUT, bytes(4 * N))

    with launch.run(trace=args.trace, cycle_limit=1_000_000) as result:
        counts = struct.unpack(f"<{N}I", result.gmem(OUT, 4 * N))
        for index, (n, got) in enumerate(zip(INPUTS, counts)):
            want = collatz_steps(n)
            if got != want:
                raise AssertionError(f"warp {index // 16} lane {index % 16}: n={n} took {got} steps, expected {want}")
        print(f"diverge_masked ok: {N} lanes, Collatz steps 0..{max(counts)} per lane, all exact, "
              f"in {result.cycle} cycles")
        print(result.summary, end="")


if __name__ == "__main__":
    main()

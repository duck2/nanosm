"""Per-lane Collatz step counts: divergent control flow (ssy, @p bra, sync), four warps.

Run from emu2: python -B -m nanosm kernels/diverge.launch.py
Options: --profile PATH, --trace PATH (Perfetto output).

Lane l of warp w counts the Collatz steps of n = 1 + (37 (16 w + l)) mod 64,
so every n in 1..64 appears once and each lane loops its own number of times
(0 for n = 1, up to 112 for n = 54 and 55), with a per-lane if / else in every
step. Checks every count against Python and reports the cycle count (not
pinned: divergent code is correct and slow by design).
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
    launch = Launch(kernel=kernel, profile=args.profile, warps=WARPS)
    launch.gmem.u32(IN, INPUTS)
    launch.gmem.write(OUT, bytes(4 * N))

    with launch.run(trace=args.trace, cycle_limit=1_000_000) as result:
        counts = struct.unpack(f"<{N}I", result.gmem(OUT, 4 * N))
        for index, (n, got) in enumerate(zip(INPUTS, counts)):
            want = collatz_steps(n)
            if got != want:
                raise AssertionError(f"warp {index // 16} lane {index % 16}: n={n} took {got} steps, expected {want}")
        print(f"diverge ok: {N} lanes, Collatz steps 0..{max(counts)} per lane, all exact, "
              f"in {result.cycle} cycles")
        print(result.summary, end="")


if __name__ == "__main__":
    main()

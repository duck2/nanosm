"""Vector add C = A + B over 32 KiB per vector (8192 fp32), four warps, ldg.v8/stg.v8.

Run from emu2: python -B -m nanosm kernels/vadd.launch.py
Options: --profile PATH (e.g. a scratch profile with lsu.staging_vectors = 1),
         --trace PATH (Perfetto output),
         --warps N (default 4; rewrites the kernel's BLOCKS_PER_WARP / WARP_STRIDE).

Checks every output bit-exactly against a Python fp32 reference, checks the
guard bytes after C, pins the algorithmic traffic counters (3 x 32 KiB, zero
waste), and reports achieved bytes per core cycle against the declared bus.
The cycle count is reported, not pinned: gmem-phase cycles are placeholder-
controller numbers until B4 (rfc09 section 4).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import struct
import tempfile

from nanosm import Launch


ROOT = Path(__file__).resolve().parents[1]
A, B, C = 0, 32768, 65536
N = 8192
WARPS = 4
GUARD = 512


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


# Coprime periods (29, 31) and inexact fractions: an element landing in the
# wrong lane, register or block cannot be hidden by a repeating pattern.
INPUT_A = [f32(((i * 13) % 29 - 14) / 7) for i in range(N)]
INPUT_B = [f32(((i * 17) % 31 - 15) / 3) for i in range(N)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=ROOT / "profiles" / "nanosm-v0.json")
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--warps", type=int, default=WARPS, choices=(1, 2, 4, 8),
                        help="active warps; the kernel's two stride constants are rewritten")
    args = parser.parse_args()

    kernel = Path(__file__).with_suffix("").with_suffix(".qtx")
    scratch = tempfile.TemporaryDirectory()
    if args.warps != WARPS:
        # Same kernel, same blocks: warp w streams blocks w, w+warps, ...
        source = kernel.read_text(encoding="utf-8")
        source = re.sub(r"\.equ BLOCKS_PER_WARP, \d+", f".equ BLOCKS_PER_WARP, {N // 128 // args.warps}", source)
        source = re.sub(r"\.equ WARP_STRIDE, \d+", f".equ WARP_STRIDE, {512 * args.warps}", source)
        kernel = Path(scratch.name) / "vadd.qtx"
        kernel.write_text(source, encoding="utf-8")

    launch = Launch(kernel=kernel, profile=args.profile, warps=args.warps)
    launch.gmem.f32(A, INPUT_A)
    launch.gmem.f32(B, INPUT_B)
    launch.gmem.write(C + 4 * N, bytes([0xA5]) * GUARD)

    with launch.run(trace=args.trace, max_trace_events=1_000_000, cycle_limit=1_000_000) as result:
        # fp32 add of two floats is exact in double, so one f32 rounding of the
        # double sum is the correctly rounded fp32 sum.
        expected = [f32(a + b) for a, b in zip(INPUT_A, INPUT_B)]
        actual = result.gmem_f32(C, N)
        for index, (got, want) in enumerate(zip(actual, expected)):
            if struct.pack("<f", got) != struct.pack("<f", want):
                raise AssertionError(f"C[{index}] = {got!r}, expected {want!r}")
        if result.gmem(C + 4 * N, GUARD) != bytes([0xA5]) * GUARD:
            raise AssertionError("vadd wrote past the end of C")

        traffic = 3 * 4 * N  # A and B read, C written: 98,304 bytes
        sectors = traffic // 32
        # Algorithmic counts, not a pinned performance result: 64 blocks x
        # (2 ldg.v8 + 1 stg.v8), each command exactly 16 whole sectors.
        expected_counters = {
            "lsu": {"vector_ld": 128, "vector_st": 64, "scalar_ld": 0, "scalar_st": 0,
                    "uni": 0, "sectors": sectors, "bytes": traffic, "useful_bytes": traffic},
            "gmem": {"commands": 192, "sectors": sectors, "beats": 4 * sectors, "bytes": traffic},
            "tma": {"commands": 0},
        }
        for engine, wanted in expected_counters.items():
            line = next(line for line in result.summary.splitlines() if line.startswith(engine + ":"))
            values = {key: int(value) for key, value in re.findall(r"(\w+)=(\d+)", line)}
            for key, value in wanted.items():
                if values.get(key) != value:
                    raise AssertionError(f"{engine}.{key}: {values.get(key)} != {value}")

        profile = json.loads(args.profile.read_bytes())
        gmem = profile["gmem"]
        bus = gmem["bus_bytes_per_dram_tick"] * gmem["dram_clock_hz"] / profile["core"]["clock_hz"]
        achieved = traffic / result.cycle
        lsu = profile["lsu"]
        print(f"vadd ok: {N} outputs bit-exact, {traffic} bytes in {result.cycle} cycles = "
              f"{achieved:.3f} B/cycle of a declared {bus:g} B/cycle bus "
              f"({100 * achieved / bus:.1f}%; warps={args.warps}, staging_vectors={lsu['staging_vectors']}, "
              f"active_commands={lsu['active_commands']})")
        print(result.summary, end="")


if __name__ == "__main__":
    main()

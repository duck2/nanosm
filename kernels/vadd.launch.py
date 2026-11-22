"""Vector add C = A + B over 32 KiB per vector (8192 fp32), four warps, ldg.v8/stg.v8.

Run from emu2: python -B -m nanosm kernels/vadd.launch.py
Options: --profile PATH (e.g. a scratch profile with lsu.active_commands = 1),
         --trace PATH (Perfetto output),
         --warps N (default 4; the kernel reads N_WARPS and strides by it).

The launch passes the three vectors' global addresses (the kernel's
signature: ptr A, ptr B, ptr C) and writes the inputs; nothing else.

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

from nanosm import Launch, launch_copy_counters


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
    parser.add_argument("--warps", type=int, default=WARPS, choices=(1, 2, 4, 8, 16),
                        help="active warps: warp w streams blocks w, w + warps, ...")
    args = parser.parse_args()

    kernel = Path(__file__).with_suffix("").with_suffix(".qtx")
    launch = Launch(kernel=kernel, profile=args.profile, warps=args.warps, args={"A": A, "B": B, "C": C})
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
        # And the launch copy's part: the three pointers (24 bytes) come from
        # global memory as one TMA load, one sector.
        expected_counters = launch_copy_counters(
            expected_counters, launch.copied_bytes, block_address=launch.block_address)
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
              f"({100 * achieved / bus:.1f}%; warps={args.warps}, "
              f"active_commands={lsu['active_commands']})")
        print(result.summary, end="")


if __name__ == "__main__":
    main()

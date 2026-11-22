"""FlashAttention on 32-row query tiles: 64 queries, 128 keys, width 64; six warps.

Run from emu2: python -B -m nanosm kernels/fa32.launch.py
Optional trace: --case rising --trace trash/fa32.perfetto.json

The same problem, inputs, outputs and checks as the 64-row kernel
(kernels/fa.launch.py at a5d0652), computed by kernels/fa32.qtx on two 32-row
query tiles with m32n64k8 commands on four 32 x 64 accumulators (S0/O0 for
query rows 0..31, S1/O1 for rows 32..63). Plain row-major fp16 inputs enter
through TMA (Q as two 32-row tiles, K and V as two 64-key blocks); the fp32
output leaves through two TMA stores of 32 rows each.

The reference and checking functions (inputs, references, check_close and
the problem constants) and the four descriptor arguments are imported from
kernels/fa.launch.py, not copied: the FA problem is defined once. This file
adds the checks of the new tile arrangement -- the tiles' addresses come from
the kernel's symbols, the shuffle inside a tile is restated from the README's
swizzle64 rule (not the simulator's layout code) -- and its own algorithmic
counter census.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path
import re
import struct

from nanosm import Launch, launch_copy_counters


ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("fa_reference", ROOT / "kernels/fa.launch.py")
fa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fa)
ROWS, KEYS, WIDTH, BLOCK, SCALE = fa.ROWS, fa.KEYS, fa.WIDTH, fa.BLOCK, fa.SCALE

QUERY_TILE = 32
Q_GLOBAL, K_GLOBAL, V_GLOBAL, O_GLOBAL = fa.Q_GLOBAL, fa.K_GLOBAL, fa.V_GLOBAL, fa.O_GLOBAL


def make_launch(q: list[list[float]], k: list[list[float]], v: list[list[float]]) -> Launch:
    # The kernel's signature is the four descriptors, one per tensor, cut at
    # the instruction's coordinates (c0 = 0, c1 = the first row). All
    # swizzle64 (the profile's mma.layout): Q a 32-row tile (one per query
    # tile), K a 64-row tile (one per key block), V the same tile transposed,
    # so its stored rows are the 64 channels, 64 keys each (the tensor core
    # reads B's K along a row); O 32 fp32 rows per store, at row 0 and row 32.
    launch = Launch(kernel=ROOT / "kernels/fa32.qtx", profile=ROOT / "profiles/nanosm-v0.json",
                    args=fa.descriptors(QUERY_TILE))
    for base, matrix in [(Q_GLOBAL, q), (K_GLOBAL, k), (V_GLOBAL, v)]:
        values = [value for row in matrix for value in row]
        launch.gmem.write(base, struct.pack(f"<{len(values)}e", *values))
    # Poison the output and P so unwritten entries cannot pass as launch zeros
    # (P's addresses are the kernel's; a test's look inside, not part of
    # launching). Block b's P is the two tiles PBUF[2 b] and PBUF[2 b + 1],
    # side by side: 64 rows of 128 bytes.
    launch.gmem.f32(O_GLOBAL, [math.nan] * (ROWS * WIDTH))
    for block in range(2):
        launch.test_shmem.write(launch.symbol(f"PBUF[{2 * block}]"),
                           struct.pack("<H", 0x7e00) * (ROWS * BLOCK))
    return launch


def swizzle64(base: int, spacing: int, row: int, byte: int) -> int:
    """Where byte `byte` of row `row` of a swizzle64 tile at `base` is stored.

    The README's rule, restated for rows of 64 bytes or more: the unshuffled
    address a = base + row x spacing + byte has the row number R = a // spacing
    (the address's own, not counted from the tile); its 16-byte piece's slot is
    XORed with R % 4 and its 4-byte word in the piece with (R // 4) % 4.
    """
    assert spacing >= 64 and spacing & (spacing - 1) == 0 and base % spacing == 0
    natural = base + row * spacing + byte
    key = natural // spacing
    return natural ^ ((key % 4) << 4) ^ (((key // 4) % 4) << 2)


def check_tile(result, base: int, spacing: int, rows: int, columns: int, element: str,
               expected, label: str) -> None:
    """Every element of a stored tile against expected(row, column), exactly."""
    size = struct.calcsize("<" + element)
    seen = set()
    for row in range(rows):
        for col in range(columns):
            address = swizzle64(base, spacing, row, size * col)
            seen.add(address)
            actual = struct.unpack("<" + element, result.shmem(address, size))[0]
            if actual != expected(row, col):
                raise AssertionError(f"{label}: row={row}, col={col}: {actual} != {expected(row, col)}")
    assert len(seen) == rows * columns, f"{label}: addresses are not a bijection"


def check_counters(summary: str, expected_counters: dict) -> None:
    for engine, expected in expected_counters.items():
        line = next(line for line in summary.splitlines() if line.startswith(engine + ":"))
        actual = {key: int(value) for key, value in re.findall(r"(\w+)=(\d+)", line)}
        for key, value in expected.items():
            if actual.get(key) != value:
                raise AssertionError(f"{engine}.{key}: {actual.get(key)} != {value}")


# Algorithmic counts, not a pinned performance result.
EXPECTED_COUNTERS = {
    # Six loads (two 4 KiB Q tiles, four 8 KiB K/V blocks) and two 8 KiB
    # stores: the 64-row kernel's bytes in eight commands instead of six.
    # 1792 sectors, each one bank round.
    "tma": {"commands": 8, "completed": 8, "logical_bytes": 57344,
            "useful_bytes": 57344, "sector_bytes": 57344,
            "port_requests": 1792, "port_rounds": 1792},
    "gmem": {"commands": 0, "sectors": 1792, "beats": 7168, "bytes": 57344},
    # 1792 TMA sectors + 8 descriptor reads + 24 statistics stores + 256 P
    # stores + 256 staging stores, all one round each. (The two float
    # constants are umov immediates: no shared-memory load.)
    "shmem": {"requests": 2336, "rounds": 2336},
    # 64 m32n64k8 commands (2 query tiles x 2 key blocks x (8 for S + 8 for
    # O)) of 128 steps on the 4x4x8 core: the 64-row kernel's 8192 steps.
    # Each command reads its 8 A blocks once, with the first B block (two
    # rounds), and 15 more B blocks (one round each): 8 x 17 rounds.
    "mma": {"commands": 64, "input_steps": 8192, "result_steps": 8192,
            "shmem_rounds": 8704, "a_block_reads": 512, "recurrence_stalls": 0},
    # Per worker: 2 x 16 score loads, 8 for the correction, 8 for the
    # normalization, of 8 beats; 8 correction stores of 8 beats. Four workers.
    "tmem": {"load_beats": 1536, "store_beats": 256},
    # Per worker: 2 x (64 + 1) ex2 and one rcp. Four workers.
    "sfu": {"ops": 524, "busy_cycles": 2096},
}


def run_case(case: str, trace: Path | None) -> None:
    q, k, v = fa.inputs(case)
    ref = fa.references(q, k, v)
    second_alpha = ref["stats"][1][2]
    if case == "rising":
        assert all(0 < alpha < 1 for alpha in second_alpha), "test must exercise every row's correction"
        sensitivity = max(abs(a - b) for a, b in zip(ref["model"], ref["broken"]))
        assert sensitivity > 1e-3, "data would not catch omitted old-O correction"
    else:
        assert all(alpha == 1 for alpha in second_alpha), "stable-max control is not stable"
    launch = make_launch(q, k, v)
    # Where the kernel keeps what the checks look at (its .shmem objects; qtxc
    # placed them).
    STATS = [launch.symbol(f"STATS[{block}]") for block in range(2)]
    K_SHARED = [launch.symbol("KBUF[0]"), launch.symbol("KBUF[1]")]     # key block b: 64 rows of 128 bytes
    V_SHARED = [launch.symbol("VBUF[0]"), launch.symbol("VBUF[1]")]     # transposed: 64 channel rows of 64 keys
    # [key block][query tile]: 32 rows of 128 bytes
    P_SHARED = [[launch.symbol(f"PBUF[{2 * block + t}]") for t in range(2)] for block in range(2)]
    O_STAGED = [launch.symbol(f"OSTAGE[{t}]") for t in range(2)]        # 32 rows of 256 bytes
    with launch.run(trace=trace, max_trace_events=1_000_000, cycle_limit=1_000_000) as result:
        output = result.gmem_f32(O_GLOBAL, ROWS * WIDTH)
        model_error = fa.check_close(output, ref["model"], 2e-6, "quantized online reference")
        attention_error = fa.check_close(output, ref["direct"], 2e-4, "direct attention")
        mismatches = [i for i, (a, b) in enumerate(zip(output, ref["model"])) if a != b]
        if mismatches:
            raise AssertionError(f"output differs from the quantized reference at {len(mismatches)} "
                                 f"places, first index {mismatches[0]}")
        for block in range(2):
            for field, values in enumerate(ref["stats"][block]):
                actual = list(struct.unpack("<64f", result.shmem(STATS[block] + field * 256, 256)))
                fa.check_close(actual, values, 1e-5, f"block {block} row statistic {field}")
        # The new arrangement, tile by tile. Query tile t's P rows (block 1's
        # buffer, the one the check reads) and staged output rows are its own
        # 32-row tiles; their bases are multiples of 16 rows, so the per-row
        # constants the workers count from query row 0 agree with the
        # address's own row number.
        for t in range(2):
            first = QUERY_TILE * t
            check_tile(result, P_SHARED[1][t], 128, QUERY_TILE, BLOCK, "e",
                       lambda row, col: ref["p"][first + row][col], f"P of query tile {t} (block 1)")
            check_tile(result, O_STAGED[t], 256, QUERY_TILE, WIDTH, "f",
                       lambda row, col: ref["model"][(first + row) * WIDTH + col], f"staged output, tile {t}")
        # The stage buffers that nothing overwrote (the staged output lies
        # over Q and K block 0): K block 1, V blocks 0 and 1, where the TMA
        # put them. V's stored row is a channel, its positions the keys.
        check_tile(result, K_SHARED[1], 128, BLOCK, WIDTH, "e",
                   lambda row, col: k[BLOCK + row][col], "K block 1")
        for b in range(2):
            check_tile(result, V_SHARED[b], 128, WIDTH, BLOCK, "e",
                       lambda channel, key: v[BLOCK * b + key][channel], f"V block {b}")
        # The kernel's own counts, and the launch copy's part: its launch
        # block (four descriptors, fifteen barrier states: 384
        # bytes) comes from global memory as one more TMA load, 12 sectors.
        check_counters(result.summary, launch_copy_counters(
            EXPECTED_COUNTERS, launch.copied_bytes, block_address=launch.block_address))
        print(f"FA32 {case}: PASS, 4096 outputs, cycles={result.cycle}, "
              f"online_error={model_error:.3g}, attention_error={attention_error:.3g}, "
              f"second_alpha=[{min(second_alpha):.5g},{max(second_alpha):.5g}]")
        print(result.summary, end="")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["all", "rising", "stable"], default="all")
    parser.add_argument("--trace", type=Path, help="Perfetto output; select a single --case")
    args = parser.parse_args()
    if args.trace is not None and args.case == "all":
        parser.error("--trace requires --case rising or --case stable")
    for case in (["rising", "stable"] if args.case == "all" else [args.case]):
        run_case(case, args.trace)


if __name__ == "__main__":
    main()

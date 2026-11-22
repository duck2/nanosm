"""FlashAttention over 64 queries and 128 keys with two 32x64 query tiles.

Run from emu2: python -B -m nanosm kernels/fa.launch.py
Optional trace: --case rising --trace fa.perfetto.json

Plain row-major FP16 inputs enter through TMA, not prepacked shared memory, and
the FP32 output leaves through TMA: workers stage normalized O in shmem as
fp32 swizzle64 tiles over the dead Q/K0 stages. Each worker pair publishes
32 output rows for its own TMA store.
The host supplies only the four descriptors (the kernel's signature); the
kernel's qtx declares its warps and barriers. The checks that look
inside shared memory (statistics, P, the staged output) take their addresses
from the kernel's symbols.
Workers compute P's and O's swizzled addresses and do all softmax, P
conversion/stores, O correction, normalization and O staging stores.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
import struct

from nanosm import Launch, launch_copy_counters, tma_desc


ROOT = Path(__file__).resolve().parents[1]
Q_GLOBAL, K_GLOBAL, V_GLOBAL, O_GLOBAL = 0, 8192, 24576, 65536
# O staging reuses the Q + K stage-0 region: 64 fp32 swizzle64 rows of 256 bytes, 16 KiB.
O_STAGE_BYTES = 16384
ROWS, KEYS, WIDTH, BLOCK = 64, 128, 64, 64
QUERY_TILE = 32


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def f16(value: float) -> float:
    return struct.unpack("<e", struct.pack("<e", value))[0]


SCALE = f32(math.log2(math.e) / math.sqrt(WIDTH))


def inputs(case: str) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    # Coprime periods avoid hiding tile/stride offsets behind power-of-two data.
    q = [[((row * 17 + d * 11) % 31 + 1) / 32 for d in range(WIDTH)] for row in range(ROWS)]
    k = []
    for key in range(KEYS):
        bias = 0 if key < BLOCK else (0.75 if case == "rising" else -1.0)
        k.append([((key * 13 + d * 7) % 29 - 14) / 32 + bias for d in range(WIDTH)])
    v = [[((key * 19 + d * 23) % 37 - 18) / 16 for d in range(WIDTH)] for key in range(KEYS)]
    return q, k, v


def descriptors(query_tile: int) -> dict[str, bytes]:
    """The four arguments of an FA kernel: one descriptor per tensor.

    Q and O move `query_tile` rows at a time. Both query tiles share the full
    K / V blocks at 0 and 64. V is transposed so a stored row holds one output
    channel across 64 keys.
    """
    def fp16(base: int, rows: int, tile_rows: int, transposed: bool = False) -> bytes:
        return tma_desc(base=base, dimensions=(WIDTH, rows), tile=(WIDTH, tile_rows),
                        element_bytes=2, layout="swizzle64", transposed=transposed)
    return {
        "QDESC": fp16(Q_GLOBAL, ROWS, query_tile),
        "KDESC": fp16(K_GLOBAL, KEYS, BLOCK),
        "VDESC": fp16(V_GLOBAL, KEYS, BLOCK, transposed=True),
        "ODESC": tma_desc(base=O_GLOBAL, dimensions=(WIDTH, ROWS), tile=(WIDTH, query_tile),
                          element_bytes=4, layout="swizzle64"),
    }


def make_launch(q: list[list[float]], k: list[list[float]], v: list[list[float]]) -> Launch:
    launch = Launch(kernel=ROOT / "kernels/fa.qtx", profile=ROOT / "profiles/nanosm-v0.json",
                    args=descriptors(QUERY_TILE))
    for base, matrix in [(Q_GLOBAL, q), (K_GLOBAL, k), (V_GLOBAL, v)]:
        values = [value for row in matrix for value in row]
        launch.gmem.write(base, struct.pack(f"<{len(values)}e", *values))
    # Poison outputs / P so unwritten entries cannot pass as launch-zero values
    # (P's address is the kernel's; a test's look inside, not part of launching).
    launch.gmem.f32(O_GLOBAL, [math.nan] * (ROWS * WIDTH))
    launch.test_shmem.write(launch.symbol("PBUF"), struct.pack("<H", 0x7e00) * (ROWS * BLOCK))
    return launch


def sum_eight(values: list[float]) -> float:
    pairs = [f32(values[i] + values[i + 1]) for i in range(0, 8, 2)]
    return f32(f32(pairs[0] + pairs[1]) + f32(pairs[2] + pairs[3]))


# The MMA core's K (profile mma.core_k): one fused group per core step.
CORE_K = 8


def mma_group(a: list[float], b: list[float], c: float) -> float:
    """Finite-input reference for fp16_raw_window25_rz_v1, independent of C++.

    Inputs here are already exact FP16 values. Products are exact in double;
    as_integer_ratio then aligns them using Python integers, not float sums.
    SF Tensor tcgen05's raw-exponent rule, applied per core step: one fused
    group of CORE_K products plus C (dot8 + C at 4x4x8).
    See third_party/tcgen05-LICENSE and src/machine/mma_numeric.cpp for provenance.
    """
    assert len(a) == len(b) == CORE_K and all(math.isfinite(x) for x in (*a, *b, c))
    exponents = [max(math.frexp(abs(x))[1] - 1, -14)
                 + max(math.frexp(abs(y))[1] - 1, -14)
                 for x, y in zip(a, b) if x != 0 and y != 0]
    if c != 0:
        exponents.append(max(math.frexp(abs(c))[1] - 1, -126))
    if not exponents:
        return 0.0
    quantum = max(exponents) - 25
    units = 0
    for value in [x * y for x, y in zip(a, b)] + [c]:
        numerator, denominator = value.as_integer_ratio()
        if quantum < 0:
            numerator <<= -quantum
        else:
            denominator <<= quantum
        units += (1 if numerator >= 0 else -1) * (abs(numerator) // denominator)
    if units == 0:
        return 0.0
    sign, magnitude = (0x80000000 if units < 0 else 0), abs(units)
    top = magnitude.bit_length() - 1
    exponent = top + quantum
    if exponent > 127:
        bits = sign | 0x7f800000
    elif exponent >= -126:
        mantissa = magnitude >> (top - 23) if top >= 23 else magnitude << (23 - top)
        bits = sign | ((exponent + 127) << 23) | (mantissa & 0x7fffff)
    else:
        shift = quantum + 149
        fraction = magnitude << shift if shift >= 0 else magnitude >> -shift
        bits = sign | fraction if fraction else 0
    return struct.unpack('<f', struct.pack('<I', bits))[0]


def references(q: list[list[float]], k: list[list[float]], v: list[list[float]]) -> dict:
    """Independent logical-matrix reference, with no simulator layout helpers.

    CORE_K-product groups use the shared raw-exponent window and FP32 truncation.
    Each group writes FP32; no hidden precision survives to the next group.
    Also compute direct double-precision attention as a separate accuracy check.
    """
    model, direct, no_correction = [], [], []
    stats = [[[] for _ in range(3)] for _ in range(2)]
    final_p = []
    for row in range(ROWS):
        scores = []
        for key in range(KEYS):
            acc = 0.0
            for d in range(0, WIDTH, CORE_K):
                acc = mma_group(q[row][d:d + CORE_K], k[key][d:d + CORE_K], acc)
            scores.append(acc)
        maximum, denominator = -math.inf, 0.0
        output = [0.0] * WIDTH
        broken = [0.0] * WIDTH
        for block in range(2):
            raw = scores[block * BLOCK:(block + 1) * BLOCK]
            new_max = max(maximum, f32(max(raw) * SCALE))
            alpha = f32(math.exp2(f32(maximum - new_max)))
            probabilities = [f32(math.exp2(f32(f32(score * SCALE) - new_max))) for score in raw]
            block_sum = 0.0
            for chunk in range(0, BLOCK, 8):
                block_sum = f32(block_sum + sum_eight(probabilities[chunk:chunk + 8]))
            denominator = f32(f32(denominator * alpha) + block_sum)
            maximum = new_max
            for dest, value in zip(stats[block], [maximum, denominator, alpha]):
                dest.append(value)
            packed = [f16(value) for value in probabilities]
            if block == 1:
                final_p.append(packed)
                output = [f32(value * alpha) for value in output]
            for key in range(0, BLOCK, CORE_K):
                for d in range(WIDTH):
                    a = packed[key:key + CORE_K]
                    b = [v[block * BLOCK + key + j][d] for j in range(CORE_K)]
                    output[d] = mma_group(a, b, output[d])
                    broken[d] = mma_group(a, b, broken[d])
            # broken deliberately omits the second block's old-O correction.
        inverse = f32(1.0 / denominator)
        model.extend(f32(value * inverse) for value in output)
        no_correction.extend(f32(value * inverse) for value in broken)
        natural_scores = [sum(q[row][d] * k[key][d] for d in range(WIDTH)) / math.sqrt(WIDTH)
                          for key in range(KEYS)]
        peak = max(natural_scores)
        weights = [math.exp(score - peak) for score in natural_scores]
        total = sum(weights)
        direct.extend(sum(weights[key] * v[key][d] for key in range(KEYS)) / total for d in range(WIDTH))
    return {"model": model, "direct": direct, "broken": no_correction, "stats": stats, "p": final_p}


def check_close(actual: list[float], expected: list[float], limit: float, label: str) -> float:
    if len(actual) != len(expected) or not all(math.isfinite(value) for value in actual):
        raise AssertionError(f"{label}: incomplete or nonfinite result")
    errors = [abs(a - b) for a, b in zip(actual, expected)]
    worst = max(range(len(errors)), key=errors.__getitem__)
    if errors[worst] > limit:
        raise AssertionError(f"{label}: index {worst}, actual={actual[worst]}, "
                             f"expected={expected[worst]}, error={errors[worst]} > {limit}")
    return errors[worst]


def run_case(case: str, trace: Path | None) -> None:
    q, k, v = inputs(case)
    ref = references(q, k, v)
    second_alpha = ref["stats"][1][2]
    if case == "rising":
        assert all(0 < alpha < 1 for alpha in second_alpha), "test must exercise every row's correction"
        sensitivity = max(abs(a - b) for a, b in zip(ref["model"], ref["broken"]))
        assert sensitivity > 1e-3, "data would not catch omitted old-O correction"
    else:
        assert all(alpha == 1 for alpha in second_alpha), "stable-max control is not stable"
    launch = make_launch(q, k, v)
    STATS, P_SHARED, O_SHARED = launch.symbol("STATS"), launch.symbol("PBUF"), launch.symbol("OSTAGE")
    with launch.run(trace=trace, max_trace_events=1_000_000, cycle_limit=1_000_000) as result:
        output = result.gmem_f32(O_GLOBAL, ROWS * WIDTH)
        model_error = check_close(output, ref["model"], 2e-6, "quantized online reference")
        attention_error = check_close(output, ref["direct"], 2e-4, "direct attention")
        for block in range(2):
            for field, values in enumerate(ref["stats"][block]):
                actual = list(struct.unpack("<64f", result.shmem(STATS + block * 1024 + field * 256, 256)))
                check_close(actual, values, 1e-5, f"block {block} row statistic {field}")
        p_bytes = result.shmem(P_SHARED, 8192)
        for row in range(ROWS):
            # The shuffle's row number is the address's own (src/layout.hpp):
            # P_SHARED // 128 + row. The kernel's per-row constants count from
            # the tile's row 0, which agrees because P_SHARED and O_SHARED are
            # multiples of 16 rows (2048 and 4096 bytes).
            key = P_SHARED // 128 + row
            for col in range(BLOCK):
                # Independent swizzle64 address (128-byte rows): the row whole,
                # the 16-byte piece's slot XOR key % 4, the word XOR (key // 4) % 4.
                offset = 128 * row + ((2 * col) ^ ((key % 4) << 4) ^ (((key // 4) % 4) << 2))
                actual = struct.unpack_from("<e", p_bytes, offset)[0]
                if actual != ref["p"][row][col]:
                    raise AssertionError(f"P feedback mismatch at row={row}, col={col}")
        o_bytes = result.shmem(O_SHARED, O_STAGE_BYTES)
        staged = set()
        for row in range(ROWS):
            key = O_SHARED // 256 + row
            for col in range(WIDTH):
                # Independent fp32 swizzle64 address (256-byte rows), the same rule.
                offset = 256 * row + ((4 * col) ^ ((key % 4) << 4) ^ (((key // 4) % 4) << 2))
                staged.add(offset)
                actual = struct.unpack_from("<f", o_bytes, offset)[0]
                if actual != ref["model"][row * WIDTH + col]:
                    raise AssertionError(f"staged O mismatch at row={row}, col={col}")
        assert len(staged) == ROWS * WIDTH, "staged O addresses are not a bijection"
        # These are algorithmic counts, not a pinned performance result.
        expected_counters = {
            # Two Q loads, four K/V loads, two O stores. All 1792 sectors
            # carry 32 useful bytes and each takes one shared-memory round.
            "tma": {"commands": 8, "completed": 8, "logical_bytes": 57344,
                    "useful_bytes": 57344, "sector_bytes": 57344,
                    "port_requests": 1792, "port_rounds": 1792},
            # All DRAM traffic is now TMA: no SIMT gmem commands remain. Total
            # traffic unchanged: 1792 sectors = 57344 bytes = useful (1.00x).
            "gmem": {"commands": 0, "sectors": 1792, "beats": 7168, "bytes": 57344},
            # 1792 TMA sector transfers + 8 descriptor reads + 24 statistics
            # stores + 256 P stores + 256 O stores. (The two float constants
            # are umov immediates: no shared-memory load.)
            "shmem": {"requests": 2336, "rounds": 2336},
            # Each 32x64x8 command walks 8 A blocks x 16 B blocks. Loading
            # A once per row group gives 8 x 17 shared-memory rounds.
            "mma": {"commands": 64, "input_steps": 8192, "result_steps": 8192,
                    "shmem_rounds": 8704, "a_block_reads": 512, "recurrence_stalls": 0},
            "tmem": {"load_beats": 1536, "store_beats": 256},
            "sfu": {"ops": 524, "busy_cycles": 2096},
        }
        # And the launch copy's part: the kernel's launch block (four
        # descriptors, ten barrier states: 384 bytes with the alignment) comes
        # from global memory as one more TMA load, 12 sectors.
        expected_counters = launch_copy_counters(
            expected_counters, launch.copied_bytes, block_address=launch.block_address)
        for engine, expected in expected_counters.items():
            line = next(line for line in result.summary.splitlines() if line.startswith(engine + ":"))
            actual = {key: int(value) for key, value in re.findall(r"(\w+)=(\d+)", line)}
            for key, value in expected.items():
                if actual.get(key) != value:
                    raise AssertionError(f"{engine}.{key}: {actual.get(key)} != {value}")
        print(f"FA {case}: PASS, 4096 outputs, cycles={result.cycle}, "
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

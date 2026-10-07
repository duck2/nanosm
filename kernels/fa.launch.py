"""First FA proof: 64 queries, 128 keys, head/value width 64; five warps.

Run from emu2: python -B -m nanosm kernels/fa.launch.py
Optional trace: --case rising --trace trash/fa-tma-out.perfetto.json

Plain row-major FP16 inputs enter through TMA, not prepacked shared memory, and
the FP32 output leaves through TMA: workers stage normalized O in shmem in the
fp32 xor64_a layout (over the dead Q/K0 stages) and one TMA store writes it.
The host supplies only descriptors, synchronization state and constants.
Workers compute P's and O's swizzled addresses and do all softmax, P
conversion/stores, O correction, normalization and O staging stores.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
import struct

from nanosm import Launch


ROOT = Path(__file__).resolve().parents[1]
Q_GLOBAL, K_GLOBAL, V_GLOBAL, O_GLOBAL = 0, 8192, 24576, 65536
Q_SHARED, K_SHARED, V_SHARED, P_SHARED = 0, 8192, 16384, 40960
K_SHARED1, V_SHARED1 = 24576, 32768
SCORE_DONE, OUTPUT_DONE, WORK_DONE, INPUT_DONE = 49160, 49168, 49176, 49184
OSTAGE_DONE, OSTORE_DONE = 49192, 49200
CONSTANTS, QDESC, KDESC, VDESC, STATS = 49216, 49664, 49728, 49792, 50176
KDESC1, VDESC1, ODESC = 49856, 49920, 49984
# O staging reuses the Q + K stage-0 region: 16 fp32 xor64_a objects, 16 KiB.
O_SHARED, O_STAGE_BYTES = 0, 16384
ROWS, KEYS, WIDTH, BLOCK = 64, 128, 64, 64


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


def make_launch(q: list[list[float]], k: list[list[float]], v: list[list[float]]) -> Launch:
    launch = Launch(kernel=ROOT / "kernels/fa.qtx", profile=ROOT / "profiles/nanosm-v0.json", warps=5)
    for base, matrix in [(Q_GLOBAL, q), (K_GLOBAL, k), (V_GLOBAL, v)]:
        values = [value for row in matrix for value in row]
        launch.gmem.write(base, struct.pack(f"<{len(values)}e", *values))
    # Poison outputs / P so unwritten entries cannot pass as launch-zero values.
    launch.gmem.f32(O_GLOBAL, [math.nan] * (ROWS * WIDTH))
    launch.shmem.write(P_SHARED, struct.pack("<H", 0x7e00) * (ROWS * BLOCK))
    launch.shmem.f32(CONSTANTS, [SCALE, -math.inf])
    # P addresses are computed in-kernel (shri/andi); no host-side layout table.
    # OUTPUT_DONE is the PV commit barrier (mma.commit, one arrival), so the
    # staged-output handoff gets its own four-worker barrier.
    for address, count in [(SCORE_DONE, 1), (OUTPUT_DONE, 1), (WORK_DONE, 4),
                           (INPUT_DONE, 1), (OSTAGE_DONE, 4), (OSTORE_DONE, 1)]:
        launch.shmem.mbar(address, expected=count)
    # Stage-granular boxes (roofline E2): one command per operand stage reads
    # each source row's full 128 bytes — full sectors at the NATURAL layout.
    for desc, gbase, dims, shape, sbase, layout, transpose in [
        (QDESC, Q_GLOBAL, (WIDTH, ROWS), (WIDTH, ROWS), Q_SHARED, "xor64_a", False),
        (KDESC, K_GLOBAL, (WIDTH, KEYS), (BLOCK, WIDTH), K_SHARED, "xor64_b", True),
        (VDESC, V_GLOBAL, (WIDTH, KEYS), (WIDTH, BLOCK), V_SHARED, "xor64_b", False),
        (KDESC1, K_GLOBAL, (WIDTH, KEYS), (BLOCK, WIDTH), K_SHARED1, "xor64_b", True),
        (VDESC1, V_GLOBAL, (WIDTH, KEYS), (WIDTH, BLOCK), V_SHARED1, "xor64_b", False),
    ]:
        launch.shmem.tma_desc(
            desc, direction="load", gmem_base=gbase,
            dimensions=dims, strides=(WIDTH,), box=shape,
            shmem_base=sbase, completion_mbar=INPUT_DONE,
            element_bytes=2, swizzle=layout, transpose=transpose,
        )
    # One store of the whole FP32 tile: the same xor64 byte permutation with
    # four floats per 16-byte row, sixteen consecutive 1024-byte objects.
    launch.shmem.tma_desc(
        ODESC, direction="store", gmem_base=O_GLOBAL,
        dimensions=(WIDTH, ROWS), strides=(WIDTH,), box=(WIDTH, ROWS),
        shmem_base=O_SHARED, completion_mbar=OSTORE_DONE,
        element_bytes=4, swizzle="xor64_a",
    )
    return launch


def sum_eight(values: list[float]) -> float:
    pairs = [f32(values[i] + values[i + 1]) for i in range(0, 8, 2)]
    return f32(f32(pairs[0] + pairs[1]) + f32(pairs[2] + pairs[3]))


def dot4(a: list[float], b: list[float], c: float) -> float:
    """Finite-input reference for fp16_raw_window25_rz_v1, independent of C++.

    Inputs here are already exact FP16 values. Products are exact in double;
    as_integer_ratio then aligns them using Python integers, not float sums.
    SF Tensor tcgen05's raw-exponent rule, applied per four-product core step.
    See third_party/tcgen05-LICENSE and src/mma_numeric.cpp for provenance.
    """
    assert len(a) == len(b) == 4 and all(math.isfinite(x) for x in (*a, *b, c))
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

    Four-product groups use the shared raw-exponent window and FP32 truncation.
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
            for d in range(0, WIDTH, 4):
                acc = dot4(q[row][d:d + 4], k[key][d:d + 4], acc)
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
            for key in range(0, BLOCK, 4):
                for d in range(WIDTH):
                    a = packed[key:key + 4]
                    b = [v[block * BLOCK + key + j][d] for j in range(4)]
                    output[d] = dot4(a, b, output[d])
                    broken[d] = dot4(a, b, broken[d])
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
            for col in range(BLOCK):
                # Inverse/reference address expression independent of runtime code.
                chunk, local_k = divmod(col, 8)
                slot = (row * 8 + local_k) ^ (2 * ((row // 4) % 2))
                actual = struct.unpack_from("<e", p_bytes, chunk * 1024 + slot * 2)[0]
                if actual != ref["p"][row][col]:
                    raise AssertionError(f"P feedback mismatch at row={row}, col={col}")
        o_bytes = result.shmem(O_SHARED, O_STAGE_BYTES)
        staged = set()
        for row in range(ROWS):
            for col in range(WIDTH):
                # Independent fp32 xor64_a address: object col//4, 16-byte row,
                # word (col % 4) XOR bit 2 of the row.
                offset = 1024 * (col // 4) + 16 * row + 4 * ((col % 4) ^ ((row // 4) % 2))
                staged.add(offset)
                actual = struct.unpack_from("<f", o_bytes, offset)[0]
                if actual != ref["model"][row * WIDTH + col]:
                    raise AssertionError(f"staged O mismatch at row={row}, col={col}")
        assert len(staged) == ROWS * WIDTH, "staged O addresses are not a bijection"
        # These are algorithmic counts, not a pinned performance result.
        expected_counters = {
            # Stage-granular loads plus ONE fp32 xor64_a store of the 16 KiB O
            # tile (was 32 per-lane stg.v8 commands): tma commands 5 -> 6 and
            # bytes 40960 -> 57344 (+16384 = O). Port-A data requests: 1280
            # input sectors + 512 output sectors; every one is two bank rounds
            # (both xor64 halves of a sector share four banks), 3584 rounds.
            "tma": {"commands": 6, "completed": 6, "logical_bytes": 57344,
                    "useful_bytes": 57344, "sector_bytes": 57344,
                    "port_requests": 1792, "port_rounds": 3584},
            # All DRAM traffic is now TMA: no SIMT gmem commands remain. Total
            # traffic unchanged: 1792 sectors = 57344 bytes = useful (1.00x).
            "gmem": {"commands": 0, "sectors": 1792, "beats": 7168, "bytes": 57344},
            # Was 1586 requests / 3122 rounds. +256 staging sts at two rounds
            # each (rows r and r+8 share a bank) = +512; +512 TMA store reads at
            # two rounds = +1024; + the store's descriptor and mbar-phase reads.
            "shmem": {"requests": 2356, "rounds": 4660},
            "mma": {"commands": 32, "input_steps": 16384, "result_steps": 16384,
                    "shmem_rounds": 16384, "recurrence_stalls": 0},
            "tmem": {"load_beats": 1536, "store_beats": 256},
            "sfu": {"ops": 524, "busy_cycles": 2096},
        }
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

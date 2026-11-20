"""Register drivers run in any order: every launch must come out byte-identical.

Machine::tick() (src/sim.cpp) computes each cycle's decisions -- the wires --
once, in a fixed dependency order, then runs every register's driver. A driver
reads only register Q values and the wires and writes only its own register,
so the drivers' order cannot matter. With NANOSM_DRIVER_SHUFFLE_SEED set, the
C API runs them (and the two independent request generators) in a random
order every cycle. Every launch's stdout and Perfetto trace must then be
byte-identical to the unshuffled run.

    python -B scripts/driver_shuffle_check.py --lib LIB [--seed N] [--jobs N]

A regression guard, not a proof: it catches a driver that reads another
register's D or state another driver writes, but only where these launches
exercise it.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]

# (name, launch script, arguments): the ledger launches. Traces go into this
# run's private temporary directory (never a shared fixed path).
LAUNCHES = [
    ("fa_rising", "kernels/fa.launch.py", ["--case", "rising"]),
    ("fa_stable", "kernels/fa.launch.py", ["--case", "stable"]),
    ("pipeline", "kernels/pipeline.launch.py", []),
    ("pipeline_tma", "kernels/pipeline_tma.launch.py", []),
    ("pipeline_tma_wide", "kernels/pipeline_tma_wide.launch.py", []),
    ("vadd", "kernels/vadd.launch.py", []),
    ("vadd_warps2", "kernels/vadd.launch.py", ["--warps", "2"]),
    ("vadd_warps16", "kernels/vadd.launch.py", ["--warps", "16"]),
    ("diverge", "kernels/diverge.launch.py", []),
]


def run_launch(library: Path, script: str, arguments: list[str], trace_path: Path, seed: int | None):
    env = dict(os.environ)
    env["NANOSM_LIBRARY"] = str(library)
    env.pop("NANOSM_DRIVER_SHUFFLE_SEED", None)
    if seed is not None:
        env["NANOSM_DRIVER_SHUFFLE_SEED"] = str(seed)
    if trace_path.exists():
        trace_path.unlink()
    command = [sys.executable, "-B", "-m", "nanosm", script, *arguments, "--trace", str(trace_path)]
    try:
        done = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, timeout=600)
    except subprocess.TimeoutExpired:
        return -1, b"", b"timed out after 600 s", b""
    trace = trace_path.read_bytes() if trace_path.exists() else b""
    return done.returncode, done.stdout, done.stderr, trace


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261008,
                        help="base seed; launch i uses seed + i")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1,
                        help="launches run at once (each has its own trace file)")
    args = parser.parse_args()

    failures: list[str] = []
    with tempfile.TemporaryDirectory() as scratch_name, ThreadPoolExecutor(max_workers=args.jobs) as pool:
        scratch = Path(scratch_name)
        runs = []
        for index, (name, script, arguments) in enumerate(LAUNCHES):
            seed = args.seed + index
            reference = pool.submit(run_launch, args.lib, script, arguments, scratch / f"{name}.reference.json", None)
            shuffled = pool.submit(run_launch, args.lib, script, arguments, scratch / f"{name}.shuffled.json", seed)
            runs.append((name, seed, reference, shuffled))
        for name, seed, reference_future, shuffled_future in runs:
            reference = reference_future.result()
            shuffled = shuffled_future.result()
            for label, (code, _, stderr, trace) in (("unshuffled", reference), ("shuffled", shuffled)):
                if code != 0:
                    failures.append(f"{name} ({label}) exited {code}: {stderr.decode('utf-8', 'replace')[-2000:]}")
                if not trace:
                    failures.append(f"{name} ({label}) wrote no trace")
            same_stdout = reference[1] == shuffled[1]
            same_trace = reference[3] == shuffled[3]
            if not same_stdout:
                failures.append(f"{name}: stdout differs under driver shuffle seed {seed}")
            if not same_trace:
                failures.append(f"{name}: Perfetto trace differs under driver shuffle seed {seed}")
            print(f"{name}: seed={seed} stdout {'same' if same_stdout else 'DIFFERS'}, "
                  f"trace {'same' if same_trace else 'DIFFERS'} ({len(reference[3])} bytes)")
    for failure in failures:
        print("FAIL", failure)
    if not failures:
        print("driver-shuffle check: OK")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

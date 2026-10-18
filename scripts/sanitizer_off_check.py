"""The sanitizer-OFF build must compute exactly what the sanitizer-ON build computes.

src/sanitizer.hpp states the rule: the machine may tell the Sanitizer what
happened and may ask it to check, but nothing the machine computes may depend
on sanitizer state. Compiling the Sanitizer to no-ops (NANOSM_SANITIZER=0)
is the mechanical proof: every launch must print byte-identical stdout and
write a byte-identical Perfetto trace, and every unit test that does not
deliberately expect a sanitizer panic must pass with identical output.

    python -B scripts/sanitizer_off_check.py --on-lib ON.dll --off-lib OFF.dll
        [--on-tests ON.exe --off-tests OFF.exe] [--save DIR]

Run from anywhere; launches run from the repository root. --save writes every
compared stdout plus a sha256 manifest (traces are hashed, not copied).
With only --on-lib (or only --on-tests) it records one variant for a later
manual comparison (e.g. a pre-refactor baseline).
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]

# (name, launch script, arguments). Every script takes --trace PATH; the trace
# goes into this run's private temporary directory, never a shared fixed path
# (overlapping runs -- ctest -j, build and build_nosan at once -- would read
# each other's files).
LAUNCHES = [
    ("fa_rising", "kernels/fa.launch.py", ["--case", "rising"]),
    ("fa_stable", "kernels/fa.launch.py", ["--case", "stable"]),
    ("pipeline", "kernels/pipeline.launch.py", []),
    ("pipeline_tma", "kernels/pipeline_tma.launch.py", []),
    ("pipeline_tma_wide", "kernels/pipeline_tma_wide.launch.py", []),
    ("vadd", "kernels/vadd.launch.py", []),
    ("vadd_warps2", "kernels/vadd.launch.py", ["--warps", "2"]),
    ("vadd_warps16", "kernels/vadd.launch.py", ["--warps", "16"]),
]

SKIP_PREFIX = "SKIP "
SKIP_REASON = " (expects a sanitizer panic"  # tests/tests.cpp main()


def run_launch(library: Path, script: str, arguments: list[str], scratch: Path):
    env = dict(os.environ)
    env["NANOSM_LIBRARY"] = str(library)
    trace_path = scratch / "trace.json"
    command = [sys.executable, "-B", "-m", "nanosm", script, *arguments, "--trace", str(trace_path)]
    if trace_path.exists():
        trace_path.unlink()
    done = subprocess.run(command, cwd=ROOT, env=env, capture_output=True)
    trace_bytes = trace_path.read_bytes() if trace_path.exists() else b""
    return done.returncode, done.stdout, done.stderr, trace_bytes


def run_tests(executable: Path):
    done = subprocess.run([str(executable)], cwd=ROOT, capture_output=True)
    return done.returncode, done.stdout, done.stderr


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def comparable_test_lines(stdout: bytes, skipped: set[str]) -> list[str]:
    """PASS lines of tests that ran in both builds; the count line is dropped
    (it differs by the skip note)."""
    lines = []
    for line in stdout.decode("utf-8", "replace").splitlines():
        if line.startswith(SKIP_PREFIX) or "tests passed" in line:
            continue
        if line.startswith("PASS ") and line[5:] in skipped:
            continue
        lines.append(line)
    return lines


def skipped_names(stdout: bytes) -> set[str]:
    names = set()
    for line in stdout.decode("utf-8", "replace").splitlines():
        if line.startswith(SKIP_PREFIX):
            names.add(line[len(SKIP_PREFIX):].rsplit(SKIP_REASON, 1)[0])
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--on-lib", type=Path)
    parser.add_argument("--off-lib", type=Path)
    parser.add_argument("--on-tests", type=Path)
    parser.add_argument("--off-tests", type=Path)
    parser.add_argument("--save", type=Path)
    args = parser.parse_args()

    variants = [(label, lib) for label, lib in (("on", args.on_lib), ("off", args.off_lib)) if lib]
    test_variants = [(label, exe) for label, exe in (("on", args.on_tests), ("off", args.off_tests)) if exe]
    if not variants and not test_variants:
        parser.error("nothing to run")
    manifest: list[str] = []
    failures: list[str] = []

    def save(name: str, data: bytes) -> None:
        if args.save is None:
            return
        args.save.mkdir(parents=True, exist_ok=True)
        (args.save / name).write_bytes(data)

    with tempfile.TemporaryDirectory() as scratch_name:
        scratch = Path(scratch_name)
        for name, script, arguments in LAUNCHES:
            results = {}
            for label, library in variants:
                code, stdout, stderr, trace_bytes = run_launch(library, script, arguments, scratch)
                results[label] = (code, stdout, trace_bytes)
                save(f"{name}.{label}.stdout.txt", stdout)
                manifest.append(
                    f"{name}.{label}: exit={code} stdout_sha256={sha(stdout)} stdout_bytes={len(stdout)} "
                    f"trace_sha256={sha(trace_bytes)} trace_bytes={len(trace_bytes)}")
                if code != 0:
                    failures.append(f"{name} ({label}) exited {code}: {stderr.decode('utf-8', 'replace')[-2000:]}")
                if not trace_bytes:
                    failures.append(f"{name} ({label}) wrote no trace")
            if len(results) == 2:
                if results["on"][1] != results["off"][1]:
                    failures.append(f"{name}: stdout differs between sanitizer ON and OFF")
                if results["on"][2] != results["off"][2]:
                    failures.append(f"{name}: Perfetto trace differs between sanitizer ON and OFF")

    test_results = {}
    for label, executable in test_variants:
        code, stdout, stderr = run_tests(executable)
        test_results[label] = (code, stdout)
        save(f"tests.{label}.stdout.txt", stdout)
        manifest.append(f"tests.{label}: exit={code} stdout_sha256={sha(stdout)} stdout_bytes={len(stdout)}")
        if code != 0:
            failures.append(f"tests ({label}) exited {code}: {stderr.decode('utf-8', 'replace')[-2000:]}")
    if len(test_results) == 2:
        skipped = skipped_names(test_results["off"][1])
        if skipped_names(test_results["on"][1]):
            failures.append("the sanitizer-ON test run skipped tests")
        on_lines = comparable_test_lines(test_results["on"][1], skipped)
        off_lines = comparable_test_lines(test_results["off"][1], skipped)
        manifest.append(f"tests: {len(on_lines)} comparable lines, {len(skipped)} skipped in OFF")
        if on_lines != off_lines:
            failures.append("unit-test output differs between sanitizer ON and OFF (skipped tests excluded)")

    report = "\n".join(manifest) + "\n"
    save("manifest.txt", report.encode())
    sys.stdout.write(report)
    for failure in failures:
        print("FAIL", failure)
    if not failures:
        print("sanitizer-off check: OK" if len(variants) == 2 or len(test_results) == 2 else "recorded")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

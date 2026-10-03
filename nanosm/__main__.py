from __future__ import annotations

import runpy
import sys


if len(sys.argv) < 2:
    raise SystemExit("usage: python -m nanosm LAUNCH.py [arguments...]")

script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")

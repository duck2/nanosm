from __future__ import annotations

import runpy
import sys

from . import launch

usage = "usage: python -m nanosm [--state-report] LAUNCH.py [arguments...]"
arguments = sys.argv[1:]
if arguments and arguments[0] == "--state-report":
    # The first machine the launch builds prints its state report first.
    launch.print_state_report = True
    arguments = arguments[1:]
if not arguments:
    raise SystemExit(usage)

script = arguments[0]
sys.argv = arguments
runpy.run_path(script, run_name="__main__")

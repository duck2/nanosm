# For agents working in this repository

nanosm is a cycle-accurate simulator of one GPU streaming multiprocessor, with a small
language (qtx) for its kernels. The kernels are meant to be written by models and read by
people.

- Start with `README.md`: how to build, the machine's parts, how a kernel is written.
- To write or change a kernel, then read the kernel in `kernels/` nearest to what you need.
  A kernel is `NAME.qtx` with `NAME.launch.py` beside it.
- To change the simulator, read `design.md`, then the unit's file in `src/machine/` (each
  opens with a description; the list is at the top of `src/machine/machine.cpp`).

## Build and run

    powershell -File scripts/build.ps1 -Fast     # build, fast tests
    python -B -m nanosm kernels/vadd.launch.py   # run one kernel and check its answer

`scripts/build.ps1` without `-Fast` runs every test; `scripts/build_nosan.ps1` is the same
with the sanitizer compiled out. Both must pass before a change to `src/` is done. A change
to a kernel needs only its launch script to pass.

## Rules

- In the simulator, state that takes the clock edge is a register, a group of registers, a
  pipeline, a FIFO, a memory or a black box. Never keep it in a standard-library container.
- Say plainly when you add a black box (a unit modelled opaquely) or change what one takes
  in, gives out or how long it takes.
- Comments describe the machine as it is, not how it got there.
- A kernel whose cycle count moved is a finding: say which and why.
- If the language or the machine gets in your way, or a design looks wrong, say so in your
  report. That is wanted.

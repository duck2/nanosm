# nanosm

Configure and build normally:

```sh
cmake -S . -B build
cmake --build build
cmake --build build --target test
cmake --build build --target run
```

The top-level Makefile is a convenience wrapper around the same CMake build:

```sh
make
make test
make run
```

The kernel path is deliberately visible and editable:

```sh
build/qtxc profiles/nanosm-v0.json kernels/vadd.qtx build/vadd.asm
build/qtxas profiles/nanosm-v0.json build/vadd.asm build/vadd.hex
build/nanosm_demo profiles/nanosm-v0.json build/vadd.hex
```

CMake performs those steps automatically for `build/generated/vadd.{asm,hex}`. On this Windows
checkout, `scripts/build.ps1` locates the installed CMake, Ninja and MinGW tools and runs the same
configure/build/test flow.

`qtx` has semantic instructions and no control bits. `asm` is scheduled qtx with explicit control
fields and is the intended human-editable layer. `hex` is one annotated 64-bit instruction word per
line and is what the simulator fetches. The same human-editable device profile is required at all
three stages. Assembly and hex carry its fingerprint, so a schedule cannot silently run with
different machine timings. Hex also carries the register count derived by qtxas; launch residency
is checked against the physical RF banks in the profile.

The current control-flow slice supports labels, unconditional `bra`, per-lane predicate registers,
`setp.{eq,ne,lt,le,gt,ge}` with register or immediate operands, and `@p0 bra.uni label`. A
`bra.uni` predicate must agree across every lane in its warp or the simulator PANICs. General
predicated execution, active masks, divergent `bra`, and reconvergence are not implemented yet.
Qtx conservatively drains live depbars and fixed-latency writes at every branch.

The JSON parser is the pinned nlohmann/json v3.12.0 single header under `third_party/nlohmann` so
configuration and builds remain offline and reproducible.

The demo launches sparse global-memory input state and runs scalar `ldg`/`fadd`/`stg` vector add.
The first controller coalesces warp accesses into 32-byte sectors, uses a finite FIFO with fixed
latency, and transfers sectors over one fixed-width bus. This is the replaceable timing policy in
front of the eventual bank/row scheduler, not a fixed-latency whole-load shortcut.

The core cycle is simulation time. `core_clock_hz` and `dram_clock_hz` derive deterministic DRAM
ticks with an integer phase accumulator. Controller latency and bus bytes are in DRAM ticks;
instruction latency and encoded stalls are in core cycles.

The simulator always collects a constant-size performance summary. Generate an optional Perfetto
timeline with:

```powershell
.\build\nanosm_demo.exe profiles\nanosm-v0.json build\generated\vadd.hex `
  --trace build\vadd.perfetto.json
```

Open `build/vadd.perfetto.json` in https://ui.perfetto.dev using **Open trace file**. The default
timeline contains interval-compressed warp states, issued instructions, shmem requests, gmem command
lifetimes, sector bus occupancy, depbar changes, and queue-depth transitions. Individual bus beats
are omitted; add `--trace-debug` when that detail is useful. Timeline recording is otherwise off,
and `--trace-max-events N` has a default of 200000. Hitting the limit leaves summary counters intact
and writes an explicit truncation marker rather than silently growing without bound.

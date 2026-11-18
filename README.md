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
make trace-pipeline
```

The kernel path is deliberately visible and editable. qtxc and qtxas produce the kernel image;
make_demo_image supplies example launch state and packs the complete launch image:

```sh
build/qtxc profiles/nanosm-v0.json kernels/vadd.qtx build/vadd.asm
build/qtxas profiles/nanosm-v0.json build/vadd.asm build/vadd.hex
build/make_demo_image profiles/nanosm-v0.json vadd build/vadd.hex build/vadd.launch.json
build/nanosm_run profiles/nanosm-v0.json build/vadd.launch.json
```

CMake performs those steps automatically for both vadd and pipeline under `build/generated`.
On this Windows checkout, `scripts/build.ps1` locates the installed CMake, Ninja and MinGW tools
and runs the same configure/build/test flow.

Running tests. The unit tests are GoogleTest (vendored in `third_party/googletest`), one file
per area in `tests/` with shared helpers in `tests/test_support.hpp`; every test is its own
ctest entry, so ctest runs them in parallel. Suites ending in `Slow`, the FA launch and the two
identity checks (driver shuffle, sanitizer ON/OFF) carry the ctest label `slow`.

```sh
powershell -File scripts/build.ps1          # build and run every test (the verification gate)
powershell -File scripts/build.ps1 -Fast    # build and run the fast set only (ctest -LE slow)
build/nanosm_tests --gtest_filter=Rf.*      # one suite (or Suite.name for one test), from the repo root
```

`scripts/build_nosan.ps1` takes the same `-Fast` switch for the sanitizer-OFF build.

Launch demos are ordinary Python files next to their kernels. They build sparse initial memory,
invoke qtx compilation and simulation in-process through `libnanosm_capi`, inspect the result, and
can optionally write a Perfetto trace:

```sh
python -m nanosm kernels/pipeline.launch.py
python -m nanosm kernels/pipeline_tma.launch.py
```

No Python package or third-party binding generator is required; the wrapper uses the standard
library's `ctypes`. Set `NANOSM_LIBRARY` when the shared library is outside `build`. A launch may
also call `write_image(path)` to produce a self-contained `.launch.json` replay artifact for
`nanosm_run`. The Python file is the authoring interface; JSON is transport, not another source of
truth.

The public boundary is [src/capi.h](src/capi.h): fixed-width C values, integer session handles,
pointer-plus-length inputs, and caller-owned output buffers. It performs no file I/O and exposes no
C++ objects or exceptions. The same functions are marked `EMSCRIPTEN_KEEPALIVE` under Emscripten,
so a later browser shell can put JSON/QTX in linear memory and use this ABI directly.

`qtx` has semantic instructions and no control bits. `asm` is scheduled qtx with explicit control
fields and is the intended human-editable layer. `hex` is one annotated 64-bit instruction word per
line and is what the simulator fetches. The same human-editable device profile is required at all
three stages. Assembly and hex carry its fingerprint, so a schedule cannot silently run with
different machine timings. Hex also carries the register count derived by qtxas; launch residency
is checked against the profile's warp slots (`core.warps`, 16) and its physical RF banks. An instruction reads the source registers
it names at issue in one RF round (the data of `acc.st` and `stg.v8` is read later, beat by beat,
and is not part of that round). An instruction whose distinct issue-time sources need more reads of
one bank (register mod `rf.banks`) than `rf.read_ports_per_bank` allows -- at the canonical one port,
two distinct sources in one bank -- is malformed: qtxc, qtxas and kernel load reject it with the
registers named.

Each warp also has a uniform register file (the profile's `urf` section; trash/rfc16.md and
trash/urf-slice-2026-11-08.md): sixteen 32-bit registers `ur0`..`ur15`, two read ports, one write
port. Only uniform arithmetic (`umov`, `uadd`, `uaddi`, `usub`, `ushl`/`ushr`/`ushli`/`ushri`,
`uand`/`uor`/`uxor` and their immediates, `usread urD, WARP_ID | N_WARPS`) and `r2ur urD, rS` (lane 0
of a vector register) write it, each presenting its result at the URF write port the same fixed
`urf.issue_to_port_cycles` after issue, so one issue per cycle is one URF write per cycle: no
calendar. `ur2r rD, urS` copies a uniform register into every lane of a vector register. An
instruction reads its uniform registers (at most two) in its one read round; they are not vector
registers and do not count toward the bank rule. The uniform consumers take them as operands:
`lds.uni`/`ldg.uni` bases, `atom.*.uni` address and operand, `acc.ld`/`acc.st` coordinates (row in
`urC`, column in `urC + 1`), mbar addresses and values, MMA operand addresses and `mma.commit`'s mbar.
The TMA's operands are uniform registers too (below). `bra.uni`'s predicate and the
`ldg.v8`/`stg.v8` base still come from lane 0 of a vector register. `sread` reads only the
per-lane `LANE_ID` / `THREAD_ID`.

The current control-flow slice supports labels, unconditional `bra`, per-lane predicate registers,
`setp.{eq,ne,lt,le,gt,ge}` with register or immediate operands, and `@p0 bra.uni label`. A
`bra.uni` predicate must agree across every lane in its warp or the simulator PANICs. General
predicated execution, active masks, divergent `bra`, and reconvergence are not implemented yet.
Qtx conservatively drains live depbars and fixed-latency writes at every branch.

Each warp issues from its own instruction FIFO of fetch packets (`core.instruction_buffer_packets`,
2; a packet is the two instruction words at an aligned 16-byte address), filled by one
instruction-memory read of one packet per cycle (a fixed one-cycle read), round-robin over the warps
with room. A branch into a packet's second half starts there.
A branch resolves one cycle after it issues -- a conditional `bra.uni` reads its predicate then --
and nothing after it issues before that: a not-taken branch continues with the words already
fetched, a taken one clears its warp's FIFO and fetches the target (`src/rf.hpp` has the cycles).

Mbar instructions currently take an address in a uniform register with no encoded offset:
`mbar.init`, `mbar.arrive`, `mbar.arrive.expect_tx`, `mbar.test_wait`, and bounded
`mbar.try_wait`. A failed try-wait parks in one of the profile-defined exact-wake slots, then
returns false on timeout; software loops when it needs a long wait.
Counts, byte counts, and parity accept either a register or an immediate. Mbar objects are
8-byte-aligned 64-bit shmem words; the exact field layout lives in `src/mbar.hpp`. The serialized
RMW device consumes one read and one write cycle on port A. DMA/MMA completion uses the simulator's
engine-facing `complete_mbar_tx` entry point rather than a SIMT instruction.

The JSON parser is the pinned nlohmann/json v3.12.0 single header under `third_party/nlohmann` so
configuration and builds remain offline and reproducible.

MMA arithmetic is `mma.numeric_model = "fp16_raw_window25_rz_v1"`, included in the profile
fingerprint. Each physical `core_k` step fuses its FP16 products with the FP32 accumulator:
select the maximum nonzero term's raw exponent (operand exponent sums for products), align
to a quantum of `2^(maximum - 25)` with magnitude truncation, sum signed integer units, and
normalize/truncate once to FP32. Subnormal operands are preserved; exact zero and underflowed
zero become +0, generated NaNs are `0x7fffffff`, and overflow produces signed infinity.
Only FP32 survives between steps. The canonical core is 4x4x8 (trash/rfc11.md section 0): each
step fuses eight products and C, which with K = 8 is one group per command, not the whole MMA
instruction across commands. The walker holds one 4x8 A block and streams the 8x4 B blocks across
N, reading each A block once per command; port B's reads have port A's depth and read latency.
There are no numerical tuning knobs or host floating-point operations in this path.

Each MMA shape is its own instruction, `mma.async.mXnYkZ` and `mma.async.init.mXnYkZ` (`init`
starts the accumulator at zero), and the shapes come from the profile: `mma.shapes` lists them
(canonical: `m64n64k8`, `m32n64k8`), and shape i of the list is opcode `0x70 + 2i` (and `+ 1` for
init) in the MMA opcode block 0x70..0x7f, so the list's order is part of the profile's fingerprint.
A shape's m, n and k are powers of two and multiples of the core's; its accumulator has at least one row per lane
and fits in tmem; and a shape with k above the core's must revisit each accumulator block late
enough to hide the accumulator latency: (m / core_m) x (n / core_n) >= port-A depth + read
latency + MMA result latency + tmem port depth (14 on the canonical profile). A kernel using a
shape the profile lacks, or an accumulator whose rows x columns differ from its MMA's shape, is
refused by qtxc, the assembler and kernel load. In the xor64 layout each 8-deep K block is a
whole (rows x 16 B) object, so a 32-row A object is the top half of a 64-row one when k is 8 and
the two halves of one 64-row tile can feed two m32 commands. A shape whose k is not a multiple of 8
(k = 4 on a 4x4x4 core) leaves its last K block partly empty: legal, but the profile prints a
warning when it loads, and the operand's footprint still counts the whole block.

`mma.layout` selects the fixed layout the tensor core reads both operands in and is included in the
profile fingerprint. The only supported swizzle is `"xor64"`; `"dense"` is the unswizzled diagnostic
baseline. A layout's address rule lives in one place, `src/layout.hpp`, used by the tensor core and
the TMA alike. `xor64` stores a tile as slices of 16-byte rows: the position along a row is cut into
16-byte pieces, piece s of every row forming slice s, and word w of row r sits at w XOR bit 2 of r --
one rule for every tile. The tensor core stores A as m rows and B as n rows, K sliced; `dense` is
A's m rows of k and B's k rows of n, row after row. The swizzle does not fix the operand's row count
or the spacing between K blocks. Future swizzles belong here as profile choices for different
physical core/bank arrangements, with matching TMA placement and SIMT address generation. This does
not imply hardware that switches between swizzles per instruction.

The arithmetic is adapted from [SF Tensor's tcgen05 model](https://github.com/sf-tensor/tcgen05)
at commit `d96b82180ac9041c09295f3a6a3c2c4e592ddce6`; attribution is in
[third_party/tcgen05-LICENSE](third_party/tcgen05-LICENSE). This borrows the measured arithmetic
rules, not B200's fusion extent, and does not claim whole-instruction B200 equivalence.
Tests include frozen B200 FP16 vectors, an upstream-derived 20,000-case regression digest,
all FP16 encodings, and physical fusion-boundary checks. FA separately checks a logical
eight-product-group reference (the same arithmetic, so it checks the flow, not the numerics) and
double-precision attention. Older profiles must name the new model and regenerate
their fingerprinted artifacts; serial scalar-FMA MMA arithmetic is no longer supported.

A `.launch.json` file is the complete cycle-0 input: profile-stamped kernel words, register count,
active warp count, and ordered sparse shmem/gmem byte segments. Bytes and 64-bit words are written
as hexadecimal strings, while addresses are JSON integers. Segments are applied in array order and
later overlapping bytes win. The simulator runner does not know which workload authored the image.

The demo launches sparse global-memory input state and runs a four-warp `ldg.v8`/`fadd`/`stg.v8`
vector add (`kernels/vadd.launch.py` checks it bit-exactly). There is no coalescer: a scalar
`ldg`/`stg` is one sector request per lane, `ldg.v8`/`stg.v8` move 512 contiguous bytes from a
warp-uniform 64 B-aligned base, and `ldg.uni` is one request fanned out to every lane. The placeholder
controller is a finite FIFO with fixed latency and one fixed-width bus behind a small LSU declared
in the profile's `lsu` section. This is the replaceable timing policy in front of the eventual
bank/row scheduler, not a fixed-latency whole-load shortcut.

**The TMA interface below is provisional and will change.** It is a first iteration of
trash/rfc17.md, to be looked at through kernels; do not treat its instruction form, operand read
or descriptor fields as settled design.

The TMA moves tiles through that same controller and through contended shmem port A. A 64-byte
descriptor in shmem describes one tensor and its tile: the global base, the element size (1, 2 or
4 bytes; 8 and 16 are reserved), four dimensions d0..d3 (an unused one is 1) with strides s1..s3 in
elements (dimension 0 is the contiguous one), and the tile -- `rows` runs of `length` elements
along dimension 0, the rows stepping along a named row dimension (1..3) -- with its shared-memory
layout: `dense` (row after row) or `xor64` (the rules above), and for xor64 whether it is
transposed (not transposed: the tile's rows are the stored rows and dimension 0 is sliced;
transposed: dimension 0's positions are the stored rows and the tile's row index is sliced). The
descriptor says nothing about who reads the tile: a kernel that feeds the tensor core chooses the
profile's `mma.layout`, any other kernel whatever layout suits it (`src/tma.hpp` has the byte
layout). The instruction says which tile, where it lands and who to tell:
`tma.load.Nd [urS], [DESC, urC], [urM]` and `tma.store.Nd [DESC, urC], [urS], [urM]`, N = 1..4:
the buffer's shared address in urS (64-byte aligned), the tile's signed coordinates c0..c(N-1) in
urC.. (the rest are zero), the completion barrier in urM (8-byte aligned; the low bits of both
addresses are not wired, and the sanitizer PANICs if they are set), the descriptor's address a
constant. So one descriptor serves both directions, every stage buffer and every tile of a tensor;
the other two coordinates choose which batch / head the tile is cut from. Element (x, y) comes from
base + (c0 + x + c1 s1 + c2 s2 + c3 s3 + y s_row) x element bytes; an element outside the tensor in
any dimension is zero-filled (load) or skipped (store). The URF's two read ports serve the whole SM,
so a tma reads its N + 2 registers two per round: urS and urM in its issue's read, then the
coordinates, one round per cycle (two rounds for N = 1, 2, three for N = 3, 4); while rounds remain,
issue refuses every instruction that reads a uniform register. The command reaches the TMA queue
when its last words arrive (canonically three cycles after issue for N <= 2, four for N > 2) and its
descriptor is fetched from shmem after that. Completion subtracts the tile's byte count from the
barrier after load writes commit or store source bytes are captured. An xor64 tile is compact: its
slices are 16 bytes x its stored-row count apart (a power of two 4 .. 128), the same spacing the
tensor core reads an xor64 operand of that many rows with; the sliced extent is a multiple of one
16-byte row's elements, at most 64, and elements are 2 or 4 bytes.

`kernels/pipeline_tma.launch.py` authors one descriptor per tensor with `launch.shmem.tma_desc`
(its keywords: `base`, `dimensions`, `strides`, `tile=(length, rows)`, `element_bytes`,
`row_dimension`, `layout="dense" | "xor64"`, `transposed`), runs the double-buffered A/B
pipeline, validates output, and with `--trace PATH` writes a Perfetto trace (no launch writes a trace unless asked, so
overlapping runs never share an output file).
The TMA trace tracks command lifetimes, command/sector-credit/return depth, port-A traffic, shared
bus ownership, and logical/useful/physical byte totals.

The core cycle is simulation time. `core_clock_hz` and `dram_clock_hz` derive deterministic DRAM
ticks with an integer phase accumulator. Controller latency and bus bytes are in DRAM ticks;
instruction latency and encoded stalls are in core cycles.

The simulator always collects a constant-size performance summary. Generate an optional Perfetto
timeline with:

```powershell
.\build\nanosm_run.exe profiles\nanosm-v0.json build\generated\pipeline.launch.json `
  --trace build\pipeline.perfetto.json
```

Or generate the same pipeline trace through CMake:

```powershell
cmake --build build --target trace_pipeline
```

Open `build/pipeline.perfetto.json` in https://ui.perfetto.dev using **Open trace file**. The default
timeline contains interval-compressed warp states, issued instructions, shmem requests, gmem command
lifetimes, sector bus occupancy, depbar changes, and queue-depth transitions. Individual bus beats
are omitted; add `--trace-debug` when that detail is useful. Timeline recording is otherwise off,
and `--trace-max-events N` has a default of 200000. Hitting the limit leaves summary counters intact
and writes an explicit truncation marker rather than silently growing without bound.

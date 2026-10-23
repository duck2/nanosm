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

The first TMA slice implements rank-1/2/3 load and store through that same controller and through
contended shmem port A. `tma.async [descriptor], rcoord` captures three signed coordinates and
continues; its 64-byte live descriptor is fetched from shmem later. Completion subtracts the full
logical box debt from the descriptor's mbar only after load writes commit or store source bytes are
captured. The v0 engine accepts element sizes 1, 2 and 4, dense shmem tiles, and zero-fills/drops
out-of-bounds elements. Swizzle and transpose are deliberately rejected in this slice.

`kernels/pipeline_tma.launch.py` authors four descriptors with `launch.shmem.tma_desc`, runs the
double-buffered A/B pipeline, validates output, and with `--trace PATH` writes a Perfetto trace
(no launch writes a trace unless asked, so overlapping runs never share an output file).
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

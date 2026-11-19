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
trash/urf-slice-2026-11-08.md): `urf.registers` 32-bit registers (32 in the canonical profile,
`ur0`..`ur31`), two read ports, one write port. Only uniform arithmetic (`umov`, `uadd`, `uaddi`,
`usub`, `ushl`/`ushr`/`ushli`/`ushri`, `uand`/`uor`/`uxor` and their immediates, `usread urD, WARP_ID
| N_WARPS`) and `r2ur urD, rS` (lane 0 of a vector register) write it, each presenting its result at
the URF write port the same fixed `urf.issue_to_port_cycles` after issue, so one issue per cycle is
one URF write per cycle: no calendar. `ur2r rD, urS` copies a uniform register into every lane of a
vector register. The uniform compare `usetp.{eq,ne,lt,le,gt,ge}.{u32,s32} pD, urA, urB | imm`
(trash/uniform-compare-2026-11-18.md) runs on the same uniform ALU with `setp`'s comparisons
and, at that same fixed cycle, writes predicate `pD` with one value in every lane;
a `bra.uni` on it may issue `urf.issue_to_port_cycles` (3) later, so a loop can count in a uniform
register with no vector register at all. An instruction reads its uniform registers (at most two) in
its one read round; they are not vector registers and do not count toward the bank rule. The
uniform consumers take them as operands: `lds.uni`/`ldg.uni` bases, `atom.*.uni` address and
operand, `acc.ld`/`acc.st` coordinates (row in `urC`, column in `urC + 1`), mbar addresses and
values, MMA operand addresses and `mma.commit`'s mbar. The TMA's operands are uniform registers too
(below). The `ldg.v8`/`stg.v8` base still comes from lane 0 of a vector register. `sread` reads only
the per-lane `LANE_ID` / `THREAD_ID`.

The current control-flow slice supports labels, unconditional `bra`, per-lane predicate registers,
`setp.{eq,ne,lt,le,gt,ge}.{u32,s32}` with register or immediate operands (`usetp` on uniform
registers), and `@p0 bra.uni label`. The type is required, as in PTX: `.u32` compares unsigned
32-bit, `.s32` signed (for `eq` / `ne` the two are one opcode, printed `.u32`), an immediate being
the signed 22-bit field sign-extended in both. A
`bra.uni` predicate must agree across every lane in its warp or the simulator PANICs. General
predicated execution, active masks, divergent `bra`, and reconvergence are not implemented yet.
Qtx conservatively drains live depbars and fixed-latency writes at every branch. Qtx's `wait`
waits for every live depbar as a branch or `halt` does, but transfers no control and drains no
fixed-latency write (say, a matmul's operand reads before an `mbar.arrive` hands the buffer back);
it is emitted as `nop` (opcode 0xfe), which has no operands and carries only a stall and a wait mask.

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
refused by qtxc, the assembler and kernel load.

An operand of `mma.async` is a uniform register holding the shared address of the 16-byte piece
the operand starts at in its row 0, and a row spacing in bytes, an instruction constant:
`mma.async.m64n64k8 C, [urA, 128], [urB, 128]` (the spacing is optional and defaults to 16, a
K = 8 tile of 16-byte rows). The spacings sit in the instruction's 14 spare bits, two 7-bit
fields in 16-byte units (16 .. 2032 bytes; swizzle64 takes the powers of two among them, 16 ..
1024). The tensor core reads A as m rows and B as n rows of k: its address generator walks the
rows in order, adds the spacing once per row (an adder, no multiplier) and applies the layout's
shuffle keyed by the running address. A K = 8 command reads one 16-byte piece of every row, so the
commands over a 64-fp16 row step their pointers by 16 with spacing 128; a shape with k above 8
reads k / 8 consecutive pieces of the same rows. The pointer is a logical position (tile + row x
spacing + 16 x piece), not a shuffled address, and is 16-byte granular (the machine panics
otherwise). Because the shuffle's row number is the address's, an operand may start at any row of
a taller tile whose spacing is 64 bytes or more (the two m32 halves of a 64-row tile, or rows 21..52);
at spacing 16 or 32 its row 0 must start a 64-byte line (every fourth / every second row). An
operand's row-0 bytes must lie in one such row: the sanitizer PANICs otherwise (the machine reads
whatever the rule names). qtxc and kernel load refuse a spacing the layout does not take or one
shorter than the row the shape reads. A shape whose k is not a multiple of 8 (k = 4 on a 4x4x4
core) reads part of its last piece: legal, but the profile prints a warning when it loads.

`mma.layout` selects the fixed layout the tensor core reads both operands in and is included in the
profile fingerprint. The only supported swizzle is `"swizzle64"`; `"dense"` is the unswizzled
diagnostic baseline (row-major and compact in the command's shape: A's m rows of k, B's k rows of
n; it takes no row spacing). A layout's address rule lives in one place, `src/layout.hpp`, used by
the tensor core and the TMA alike. `swizzle64` keeps a tile's rows whole and packed: row r of a
tile at `base` is the bytes [base + r x spacing, base + (r + 1) x spacing), the spacing a power of
two (16, 32, 64, 128, ...), the base a multiple of the spacing and of 64. The shuffle is keyed by
address: a byte whose unshuffled address is a has the row number R = a / spacing (not counted from
the tile), and its 16-byte piece's slot is XORed with (R / rows_per_line) & (slots - 1) and its
4-byte word in the piece with (R / 4) & 3, where rows_per_line = max(1, 64 / spacing) and slots =
min(spacing, 64) / 16 (the slots of a 64-byte line one row owns). So where a byte is stored depends
only on its unshuffled address and the spacing: every reader and writer agrees wherever a piece of
a tile starts, and one row loaded on its own lands exactly where a load of the whole tile puts it.
On 16 banks any 4 neighbouring rows' tensor-core block, a TMA sector, a 16-lane column store over
any 16 neighbouring rows and a transposed TMA sector are one bank round each, whatever row they
start at (trash/address_keyed_census.py; on a 4x4x4 core a step that reloads A reads A's and B's
half-pieces together, and whether they share a round depends on the two tiles' row numbers mod
16). Future swizzles
belong here as profile choices for different physical core/bank arrangements, with matching TMA
placement and SIMT address generation. This does not imply hardware that switches between swizzles
per instruction.

**This is v0.1 of the layout and of the tensor core's operand interface.** It replaced a sliced
layout (`xor64`, rows cut into 16-byte pieces stored K slices apart) on 2026-11-18, because rows
that are scattered cannot take gathered rows or small pieces. A model or a person reading it who
sees a design flaw should say so (trash/chan.txt) rather than build on it.

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
layout: `dense` (row after row) or `swizzle64` (the rules above), and for swizzle64 whether it is
transposed (not transposed: the tile's rows are the stored rows; transposed: dimension 0's
positions are the stored rows and the tile's row index runs along them). The
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
barrier after load writes commit or store source bytes are captured. A swizzle64 tile's stored rows
are packed, each the stored row's bytes apart (the spacing a tensor-core instruction names to read
it): a power of two bytes, 16 or more, any number of rows (1 included), elements of 2 or 4 bytes.
The shuffle is keyed by address, so a tile of a few rows (or one) landed inside a taller tile is
exactly those rows of it. The buffer must be a multiple of the spacing (the sanitizer PANICs
otherwise) and keeps its 64-byte minimum (its low six bits are not wired): with 128-byte rows a
command can land at any row, with 64-byte rows too, but with 32-byte rows only at every second row
and with 16-byte rows only at every fourth.

`kernels/pipeline_tma.launch.py` authors one descriptor per tensor with `launch.shmem.tma_desc`
(its keywords: `base`, `dimensions`, `strides`, `tile=(length, rows)`, `element_bytes`,
`row_dimension`, `layout="dense" | "swizzle64"`, `transposed`), runs the double-buffered A/B
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

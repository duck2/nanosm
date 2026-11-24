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

Launch demos are ordinary Python files next to their kernels. They build the inputs in global
memory, pass the kernel's arguments, invoke qtx compilation and simulation in-process through
`libnanosm_capi`, check the result, and can optionally write a Perfetto trace:

```sh
python -m nanosm kernels/pipeline.launch.py
python -m nanosm kernels/pipeline_tma.launch.py
python -m nanosm --state-report kernels/pipeline.launch.py   # first, the machine's state report
```

The state report lists every piece of the machine's state by unit with its declared hardware
bits: in the open, flip-flops (registers, groups, pipelines) and FIFO storage, counted; behind
walls, the black boxes (the arithmetic units, the tensor core's arithmetic, the memories), each
with its contract, its declared cost and the sign-off its definition carries (src/machine/blackbox.hpp);
and the totals of each, kept apart (src/machine/state_report.cpp; the C API's
`nanosm_session_state_report`).

**Black boxes and their sign-offs.** A black box is a unit modelled opaquely: registered inputs
and outputs, a fixed latency, a cost declared as a function of the profile, and anything inside.
There are ten: the ALU, the FPU, the special-function unit, the tensor core's MMA core and six
memories. Everything that defines one is between `// BLACK BOX: <name>` and
`// END BLACK BOX: <name>`, with one sign-off line, and only the human signs it.
`python scripts/blackboxes.py` lists each black box with its sign-off line, its state (not
signed, signed, or changed since signing) and its fingerprint, a hash of the text between its
markers; to sign one, write that fingerprint into its sign-off line,
`SignOff::signed_by_human("<fingerprint>")` in place of `SignOff::not_signed`. Changing a signed
black box is fine: the change gives it another fingerprint, so it shows as changed since signing
until the human signs it again. The simulator runs either way; the commit hook
(`scripts/hooks/pre-commit`) refuses a commit while a black box is not signed or has changed
since it was signed. Setup:

```sh
git config core.hooksPath scripts/hooks               # once per clone: enables the commit hook
git update-index --chmod=+x scripts/hooks/pre-commit  # once, after the hook is first added on Windows
```

A `git add` on Windows records the hook as mode 100644; the second line makes it 100755, which
Git on Unix needs to run it.

No Python package or third-party binding generator is required; the wrapper uses the standard
library's `ctypes`. Set `NANOSM_LIBRARY` when the shared library is outside `build`. A launch may
also call `write_image(path)` to produce a self-contained `.launch.json` replay artifact for
`nanosm_run`. The Python file is the authoring interface; JSON is transport, not another source of
truth.

**Registers.** In qtx a register is a name, `%x`. It exists from its first use and qtxc picks its
number (src/qtxc/registers.cpp). The instruction decides the
file: an operand position takes a vector register (sixteen lanes) or a uniform one (one value per
warp), and a name used as both is refused (`ur2r` and `r2ur` carry a value from one file to the
other). Predicates are `p0`..`p6` and a mask is `@p0`; a predicate may have a name too: `%odd` in a
predicate position (`setp.ne.u32 %odd, %bit, 0`, then `@%odd add ...`) gets one of the predicates
the kernel does not name by number, `%p0` there is `p0` itself, and a name is a register or a
predicate, not both. Numbers (`r5`, `ur5`) are assembly's: the
scheduled assembly qtxc writes holds them, says which name got which (`.name %row, r4`) and stays
hand-writable.

```
        sread   %lane, LANE_ID
        shli    %offset, %lane, 2
        ld.param.u32 %base, [IN]       ; IN: a .param (Launch, below)
        add     %ptr, %offset, %base
        ldg     %n, [%ptr]
```

A kernel declares registers only where the machine needs neighbours: `.reg %a<8>` is `%a0`..`%a7`,
eight adjacent vector registers, for `ldg.v8` / `stg.v8` / `acc.ld.v8` / `acc.st.v8`, which name the
first of eight; `.ureg %c<2>` is adjacent uniform registers, for a TMA's coordinates. The line may
stand anywhere in the file, and the count may be a constant expression (`.reg %a<4 * STAGES>`, with
the `.equ` above it). An instruction that needs a group and is given a lone name is refused with the
line to add. Names beginning `%qtxc_` are the compiler's own (the register an `ld.param` reads
through, the copies it inserts) and are refused where a kernel writes one.

- **Banks are qtxc's business.** An instruction's issue reads at most `rf.read_ports_per_bank`
  different registers from one bank (register number mod `rf.banks`: three banks, one port each, at
  the canonical profile). qtxc gives the names numbers that keep every instruction's sources apart; a
  group has one choice, its base, for all its members. A register named twice by one instruction is
  one read.
- **A copy where no number fits.** A value read beside registers of every bank has no bank left.
  qtxc then copies one source into a scratch register immediately before the instruction that needs
  it (a label on the instruction moves to the copy) and warns: `line 12: 'add %r, %v, %x' reads %x
  and %v from one register-file bank; qtxc inserted a copy of %x before it (one more instruction each
  time it runs)`. The copy is in the assembly, marked. Inside a loop the warning says so; a second
  name made before the loop (`mov %v2, %v`) and read there removes it (fa32's `%scale_b` and
  `%max_b` are such names). No kernel in `kernels/` needs a copy.
- **Sharing.** A name holds its value from a write to the last read that needs it. Every name gets
  a number of its own while its file has room -- the uniform file's `urf.registers`; for the vector
  file what each warp can have when the kernel's warps (`.warps`, else every warp slot) share
  `rf.rows_per_bank` -- and only beyond that do two names that never hold a value at the same time
  (two warps' roles, say) get one number. A write under a mask keeps the old value alive in the
  lanes it skips, and in a kernel that uses the divergence stack every vector write counts as such
  a write. A kernel that holds more values at once than a file has registers is refused.

`launch.register("%row")` (Python) and the C API's kernel info (`registers`, `warnings`) give a
test or a debugger the numbers; the Python launch and the `qtxc` tool print the warnings. Most unit
tests' kernels are still written in numbers, because they pin cycles that depend on a register's
bank: `tests/test_support.hpp` renames them on the way in (`r5` becomes `%r5`, a member of one group,
which keeps its number).

**Memory.** A kernel names its own memory and qtxc places it (src/qtxc/objects.cpp,
expressions.cpp). Names are bare (`%` is for registers):

```
.shmem QBUF[2],   rows=32, cols=64, elem=2        ; a tile: rows back to back, a row is cols x elem bytes
.shmem STATS[2],  bytes=1024, align=4             ; a plain buffer
.shmem PBUF[4],   rows=32, cols=64, elem=2, align=16 * PBUF.row_bytes
.shmem OSTAGE[2], rows=32, cols=64, elem=4, over=QBUF, KBUF[0]
.tmem  S[2],      rows=32, cols=64                ; tensor memory, in words

        umov  %kbuf, KBUF[0]
        uaddi %kbuf, %kbuf, KBUF.stride           ; the next element
        shli  %p_row, %row, log2(PBUF.row_bytes)  ; a row's address
        addi  %p_row, %p_row, PBUF
        lds   %a, [%byte + A_TILE[1]]             ; an object read where it is used
        acc.ld.v8 %col0, [%walk, S.row_words]
```

An object is the author's bookkeeping: the machine never sees one, only the addresses the
instructions carry. So a line says nothing about a layout (a tile's layout is its TMA descriptor's
and the tensor core's, both keyed by the address), and nothing ties a tile to a descriptor; a
`.shmem` tile that the host's descriptor does not match is still wrong.

- **Placement.** Objects go one after another from address 0 in the order they are declared, each
  on its alignment; shared memory in bytes, tmem in words from word 0. The launch block (`.param`,
  `.mbar`, `.data`: below) stays at the top of shared memory. An object that does not fit is named.
  An array `NAME[N]` is N elements side by side: element i is at `NAME + i x stride`, the stride
  being one element rounded up to the alignment.
- **Alignment is its own statement.** `align=` (bytes; words for tmem) replaces the default, and
  the defaults are only what the machine needs: a shared tile on a multiple of its row and of 64
  bytes (the swizzle's base rule, and what a TMA buffer needs), a plain buffer on 64, a tmem tile
  on four of its rows (the pointer rule of a matmul result and of `acc.ld`; its row, `cols`, is a
  power of two from 4, and its stride is padded to whole bands of four rows). A kernel that needs
  more says so: fa / fa32 shuffle a row's bytes themselves with one constant per row, which needs a
  base on a multiple of 16 rows, hence `align=16 * PBUF.row_bytes`.
- **`over=` is the only overlap.** An object with `over=A, B[0]` is not placed: it lies on the
  objects it lists (a bare array name is all its elements), which must follow one another in that
  order and be large enough; the first of them takes the overlay's alignment before anything is
  placed. What it may use ends at the last listed element's own size: the padding after an object
  (its stride minus its size) is not the object's, and the next declaration may sit in it. Objects
  never overlap otherwise. Whether the memory is free when the overlay is written
  is the kernel's argument (its barriers), which the sanitizer's ownership checks test at run time.
- **Names, elements, properties.** `NAME` is the object's address (its element 0) and `NAME[i]`
  element i's; an index past the end is refused. `NAME.bytes` is the whole object and
  `NAME[i].bytes` one element (`.words` for a tmem object); `NAME.stride` the step between
  elements; `NAME.row_bytes` a shared tile's row and `NAME.row_words` a tmem tile's.
- **Constant expressions.** Wherever qtx or assembly takes a constant (an immediate, an address
  offset `[%r + ...]`, an `.equ`, a directive's argument) it may be an expression: `+ - * / << >>`,
  unary minus, parentheses, and `log2(x)`, which refuses an argument that is not a power of two.
  Nothing wraps: a value past 2^40 anywhere in an expression is refused. Whoever uses the value
  checks that it fits its field. An expression that names a launch section
  (`.param`, `.mbar`, `.data`) keeps that section's launch-flag wait (Launch, below), through any
  `.equ`; an object's name never has one.
- **`ld.param.u32 %x, [NAME]`** (and `.f32`) reads a parameter into every lane of a vector
  register. It is written for the reader and stands for the two instructions a parameter read is:
  `umov` of the parameter's shared address into a uniform register qtxc names, then `lds.uni`. Of a
  `ptr` parameter it reads the low 32 bits. No hardware and no opcode comes with it.
- **Looking inside.** `launch.symbol("PBUF")` and `launch.symbol("PBUF[1]")` give an object's and
  an element's address (an `.mbar` array's barrier too); `launch.memory_object("PBUF")` and the C
  API's kernel info (`"objects"`) give its size, stride and shape. Assembly carries each object's
  line with the address qtxc chose (`at=ADDRESS`, assembly only); hex and a saved image carry none.
  An address written by hand gets the checks placement gives: on its alignment, inside its memory,
  apart from the other objects unless it says `over=`, and then inside what it names.

**Launch.** A qtx file is self-contained: it declares everything a launch needs to know beside its
code, and the host knows only the kernel's signature. (v0.1, a first working version: the copy and
the flags below are expected to change.)

```
.warps  6                          ; how many warps the kernel is written for (optional)
.param  ptr   OUT                  ; the signature, in order:
.param  u32   COUNT                ;   ptr  8 bytes, a 64-bit global byte address
.param  desc  QDESC                ;   u32, f32  4 bytes;  desc  64 bytes, a TMA descriptor
.mbar   O_STORED, expect 1         ; a barrier's initial state: phase 0, 1 arrival expected (8 bytes)
.mbar   S_READY[2], expect 1       ; an array: two such barriers side by side, ONE section
.data   TABLE, f32 0.5, 0.25, -inf ; initialized words (f32 or u32)
```

- **The `.param` lines are the signature.** The launch passes one argument per parameter, by name
  (`Launch(kernel=..., profile=..., args={"OUT": 65536, "QDESC": tma_desc(...)})`) or in the
  signature's order; a missing, unknown or wrongly typed argument is refused by name. Arguments and
  global memory are all the host gives: it needs no address inside shared memory.
- **The launch block.** Every `.param`, `.mbar` and `.data` line is a named section, and together,
  in the order they are declared, they are the launch block. Its place in shared memory is one
  fixed rule from the sections and the profile: the sections sit in declaration order, each aligned
  to its own size (`ptr` 8, `u32` / `f32` 4, `desc` 64, an mbar 8, data 4); the block starts at the
  top of shared memory (`shmem.bytes`) minus its bytes, rounded down to a multiple of 64. A section's
  name is its address, an immediate: `umov %arg, OUT` then `lds.uni %out, [%arg]` (which
  `ld.param.u32 %out, [OUT]` writes for you), or the name as the
  descriptor operand of `tma.load` / `tma.store`. An `.equ` whose expression names a section is
  an address in that section, and its users wait for it as the section's do. A `ptr` is 8 bytes, but `ldg` / `stg` still take their whole address from one 32-bit
  register, so a kernel reads the low word and the launch refuses a pointer that does not fit 32
  bits (a second address register is planned).
- **Barrier arrays.** `.mbar NAME[C], expect N` (C a constant expression, so a producer's buffers
  and their barriers can share one `.equ`) is C barriers side by side in ONE section: element
  i is at `NAME + 8 i` and is written `NAME[i]` (`umov %ready, S_READY[1]`; `NAME` alone is element
  0, and code may reach element t as `NAME + 8 t` in a register). An index past the array's end is
  refused. Because an array is one section it arrives, and is waited for, whole: that is how a
  kernel says "these belong together", so the ORDER of the declarations is free to mean only
  arrival order. Two separate `.mbar` lines promise nothing about each other's address.
- **The signature's order and the block's order.** The signature is the `.param` lines in the order
  written among themselves; the block is all the section lines in the order written. So a kernel
  can keep its public parameter order and still put a barrier it needs first between two
  parameters (`kernels/fa32.qtx`: `QDESC`, then `Q_LANDED[2]` and the other barriers, then `KDESC`,
  `VDESC`, `ODESC`).
- **Nothing of it is in shared memory at the first cycle.** The host writes the block's bytes (the
  arguments, the barrier states, the data) into global memory, by default at its top
  (`Launch(..., block_address=...)` chooses), and writes the launch record: host-written registers
  that are in place before the first cycle, like the warp count. The record holds the block's global
  address, its shared address, its size and the launch flags' cuts (174 bits at the canonical
  profile: src/image.hpp).
- **The launch copy.** From the first cycle the SM copies the block from global memory into shared
  memory, and the warps start at the first cycle too. The copy is the TMA's engine doing one dense
  load with no instruction behind it: its walker is loaded from the launch record, so it fetches no
  descriptor and completes into no barrier; its sectors pay global memory's latency (100 cycles),
  cross the bus at its rate (8 cycles per 32-byte sector) and share the bus, the TMA's queue (one
  of its four commands) and the port-B write with the kernel's own work. The kernel ends only after
  the copy has. A profile with no TMA cannot run a kernel that has sections.
- **Launch flags** (`core.launch_flags`, 4; 1..6). A launch flag is one of a warp's six depbars, taken
  from sb5 downward. It starts counted once in every warp, and is decremented in every warp when the
  copy has landed the first N bytes of the block, N being that flag's cut. An instruction waits on a
  flag through its ordinary wait mask, so nothing is new at issue and no warp signals another.
  **Declared first arrives first:** the sections land in declaration order. qtxc chooses the cuts
  (the first policy: one flag for each of the first `launch_flags` - 1 sections, the last flag for
  all the rest; assembly and hex carry them as `.flag sbN, BYTES`) and makes every instruction
  whose text names a section wait on that section's flag, unless the straight-line code before it
  already has; code that names no section runs meanwhile. A branch, a `halt` or qtx's `wait` do not
  wait for launch flags. A wait is on the NAMING instruction, not on the access: naming a section
  (by its own name, an element `NAME[i]`, or an `.equ` name for either) covers the whole section, an
  array included; a kernel that computes its way from one section into ANOTHER is not covered by
  qtxc, and the sanitizer PANICs on any read or write of block bytes that have not landed (an
  `lds`, an `sts`, an mbar operation, a TMA's descriptor or buffer). The summary's `launch:` line says when each flag cleared (`sb5:64@118`: depbar, block
  bytes, cycle).
- **There is no untimed way in.** The launch block reaches shared memory only by the copy
  (`core.launch_flags` 0 is refused: a kernel with sections needs a flag to wait on). The kernel's
  code is the one thing still in place before the first cycle without being timed (the instruction
  memory is loaded by the launch).
- **Float constants need no memory.** `umov urD, imm` takes any 32-bit word, so a constant is one
  instruction (`umov %bits, 0.180336878`, then `ur2r %scale, %bits` if every lane needs it). `.data` is for
  tables.
- **A barrier is initialized by its `.mbar` line** and not by `mbar.init` at run time: the warps
  start together and nothing would hold the others until one had initialized it.
- **`.warps N`** is optional; a kernel that works at any count leaves it out and the launch says
  (`warps=`). If both say, they must agree.
- **Looking inside is separate from launching.** `nanosm_qtx_kernel_info` (C API) returns the
  kernel's signature, its sections with their addresses and flags, and its symbols (`.equ` names,
  section and object addresses) and memory objects as JSON; Python has `launch.signature`,
  `launch.sections` and `launch.symbol("PBUF")` (`"PBUF[1]"` for an element). A test that checks a tile inside shared memory uses those names with
  `result.shmem(...)`. A test that poisons a region first uses `launch.test_shmem.write(...)` (C++:
  `Launch::test_shmem`): bytes placed before the first cycle, untimed, which is a TEST's access and
  not part of a launch. It may not touch the launch block, and a run that used it says so in its
  summary (`test_preload: shmem_bytes=N`, absent otherwise), so a harness that scores kernels can
  refuse it. Assembly carries the same directives; hex carries
  them with each section's address (kernel load checks it against the rule); neither hex nor a
  saved image carries symbols; all three keep the sections in their declared order, so a replay
  has the same timing. The launch copy counts in the `tma:` and `gmem:` summary lines like any
  load; `nanosm.launch_copy_counters` adds its part to a script's own expected counts (in global
  sectors, so it takes the block's global address when the launch chose one).

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
two distinct sources in one bank -- is malformed: qtxas and kernel load reject it with the
registers named. qtxc never writes one: it picks the numbers, and inserts a copy where none fit
(Registers, above).

Each warp also has a uniform register file (the profile's `urf` section): `urf.registers`
32-bit registers (32 in the canonical profile,
`ur0`..`ur31`), two read ports, one write port. `umov urD, imm` holds a whole 32-bit word (its
payload is the register in 6 bits and the word in 32; every other immediate is 22 bits signed): an
integer from -2^31 to 2^32 - 1, or a float literal (`1.5`, `1e-3`, `-inf`, `nan`), taken as its
IEEE bits; a word outside the 22-bit range prints as hex (`umov ur2, 0x3e38aa3b`). Only uniform
arithmetic (`umov`, `uadd`, `uaddi`,
`usub`, `ushl`/`ushr`/`ushli`/`ushri`, `uand`/`uor`/`uxor` and their immediates, `usread urD, WARP_ID
| N_WARPS`) and `r2ur urD, rS` (lane 0 of a vector register) write it, each presenting its result at
the URF write port the same fixed `urf.issue_to_port_cycles` after issue, so one issue per cycle is
one URF write per cycle: no calendar. `ur2r rD, urS` copies a uniform register into every lane of a
vector register. The uniform compare `usetp.{eq,ne,lt,le,gt,ge}.{u32,s32} pD, urA, urB | imm`
 runs on the same uniform ALU with `setp`'s comparisons
and, at that same fixed cycle, writes predicate `pD` with one value in every lane;
a `bra.uni` on it may issue `urf.issue_to_port_cycles` (3) later, so a loop can count in a uniform
register with no vector register at all. An instruction reads its uniform registers (at most two) in
its one read round; they are not vector registers and do not count toward the bank rule. The
uniform consumers take them as operands: `lds.uni`/`ldg.uni` bases, `atom.*.uni` address and
operand, `acc.ld`/`acc.st` tmem pointers, mbar addresses and values, MMA operand addresses and
result pointer, and `mma.commit`'s mbar. Two instructions read more than the two read ports give
one round: a TMA command (below) and `mma.async` (urA and urB at issue + 1, its result pointer urC at
issue + 2); while either has a round left, an instruction that reads a uniform register waits a
cycle. The TMA's operands are uniform registers too (below). `ldg.v8`/`stg.v8` take a per-lane address from a vector register, as `ldg`/`stg` do. `sread` reads only
the per-lane `LANE_ID` / `THREAD_ID`.

The current control-flow slice supports labels, unconditional `bra`, per-lane predicate registers,
`setp.{eq,ne,lt,le,gt,ge}.{u32,s32}` with register or immediate operands (`usetp` on uniform
registers), and `@p0 bra.uni label`. The type is required, as in PTX: `.u32` compares unsigned
32-bit, `.s32` signed (for `eq` / `ne` the two are one opcode, printed `.u32`), an immediate being
the signed 22-bit field sign-extended in both. A
`bra.uni` predicate must agree across every lane in its warp or the simulator PANICs.

Divergence is pre-Volta: one warp, one pc, a per-warp active mask
and a divergence stack (`core.divergence_stack_entries`, 8; 0 means none, and qtxc, the assembler
and kernel load then refuse the three forms below). The kernel author writes them; qtxc infers
nothing. Every stack entry is a (mask, pc) pair. `ssy L` pushes (the active mask, L); L may be
any instruction. `@p bra T` / `@!p bra T` is the divergent branch: the active lanes with the
predicate are taken, the others fall; if one side is empty the whole mask goes the other way,
otherwise (taken lanes, T) is pushed and the fall-through arm runs first. `sync` always pops an
entry and switches to it: the mask becomes the entry's and the warp goes to its pc. So an if / else
is `ssy L; @p bra T; <else arm>; sync; T: <then arm>; sync; L: ...`: each arm ends in its own
`sync` and L is an ordinary instruction. Every per-lane writer (arithmetic, moves, conversions,
special registers, the SFU, `ur2r`, predicates, `lds` / `sts`, `ldg` / `stg`, `ldg.v8` / `stg.v8`) writes only
the active lanes; fixed timing ignores the mask. Instructions that act for the whole warp are two
kinds. A warp-wide memory or engine operation under a partial mask is the sanitizer's PANIC:
`tma.*`, `mma.*`, `acc.*`, `mbar.*`, `atom.*.uni`, `ldg.uni` / `lds.uni`,
`r2ur` and `bra.uni`. Uniform integer arithmetic (`umov`, `uadd`, ...) is
not checked: it executes once in every arm that issues it, so a counter bumped in both arms is
bumped twice (`usetp` writes its predicate on the active lanes only). A halt inside an arm is
the sanitizer's PANIC too; stack overflow and a `sync` on an empty stack are the machine's.
`kernels/diverge.launch.py` counts Collatz steps per lane.

Masked instructions: `@p` / `@!p` on a per-lane instruction. The operation runs as usual and
its result is written with a mask. The per-lane instructions are exactly the writers above: every fixed
arithmetic instruction (ALU, FPU, the SFU, `cvt.pack`, `mov`, `sread`, `ur2r`, `setp`) and
`lds`, `sts`, `ldg`, `stg`, `ldg.v8`, `stg.v8`. The instruction's write mask is its predicate's lanes (inverted for
`@!p`); the lanes it writes are the active mask AND its write mask. The write mask is
read from the predicate file in the instruction's operand round, the cycle a branch reads
its predicate (issue + 1), so qtxc keeps the predicate's writer the distance it keeps a
`bra.uni`'s, and the sanitizer checks it the same way. There is no exception: a masked
`setp` leaves the predicate bit alone in the lanes its write mask turns off; a masked load
leaves those lanes of its register unchanged; a masked store sends nothing for them (and does
not check their addresses). Issue does not look at the write mask (it is not known yet) and
timing ignores it: the write booking, the SFU's cycles, the depbar and the memory slot are
taken as if every lane were on. A masked `lds` / `sts` / `ldg` / `stg` (or v8 form) whose write mask turns
every lane off makes no request and completes at its dispatch: its depbar is decremented then,
so a waiting instruction can issue 4 cycles after it at canonical depths. Every other
instruction refuses to be masked at qtxc, the assembler and kernel load (`uadd cannot be
masked (@p)`): uniform arithmetic, `usetp`, `r2ur`, the warp-wide memory and engine operations
listed above, `mbar.*`, `atom.*`, `fence.sh`, `nop`, `halt`, `ssy` and `sync`. `bra` and
`bra.uni` keep their own meaning of the predicate. In qtx `mov rD, rS` is `addi rD, rS, 0` (and
disassembles as that), so a register move can be masked: `@p0 mov %x, %y`.
`kernels/diverge_masked.launch.py` is the Collatz count with each step's odd / even choice
made by two masked arms instead of an if / else on the stack.

Qtx conservatively drains live depbars and fixed-latency writes at every branch and `sync`. Qtx's `wait`
waits for every live depbar as a branch or `halt` does, but transfers no control and drains no
fixed-latency write (say, a matmul's operand reads before an `mbar.arrive` hands the buffer back);
it is emitted as `nop` (opcode 0xfe), which has no operands and carries only a stall and a wait mask.

Each warp issues from its own instruction FIFO of fetch packets (`core.instruction_buffer_packets`,
2; a packet is the two instruction words at an aligned 16-byte address), filled by one
instruction-memory read of one packet per cycle (a fixed one-cycle read), round-robin over the warps
with room. A branch into a packet's second half starts there.
A branch resolves one cycle after it issues -- a conditional `bra.uni` reads its predicate then --
and nothing after it issues before that: a not-taken branch continues with the words already
fetched, a taken one clears its warp's FIFO and fetches the target (`src/timing.hpp` has the cycles).
A divergent `bra` and a `sync` resolve there too (a `sync` reads a registered copy of the stack's
top); a `sync` whose popped pc is the instruction after it continues like a not-taken branch (one
address compare), any other popped pc redirects like a taken one. An `ssy` pushes there without
holding its warp.

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
Only FP32 survives between steps. The canonical core is 4x4x8: each
step fuses eight products and C, which with K = 8 is one group per command, not the whole MMA
instruction across commands. The walker holds one 4x8 A block and streams the 8x4 B blocks across
N, reading each A block once per command; port B's reads have port A's depth and read latency.
There are no numerical tuning knobs or host floating-point operations in this path.

Each MMA shape is its own instruction, `mma.async.mXnYkZ` and `mma.async.init.mXnYkZ` (`init`
starts the accumulator at zero), and the shapes come from the profile: `mma.shapes` lists them
(canonical: `m64n64k8`, `m32n64k8`), and shape i of the list is opcode `0x70 + 2i` (and `+ 1` for
init) in the MMA opcode block 0x70..0x7f, so the list's order is part of the profile's fingerprint.
A shape's m, n and k are powers of two and multiples of the core's; its result has at least one row per lane
and fits in tmem; and a shape with k above the core's must revisit each accumulator block late
enough to hide the accumulator latency: (m / core_m) x (n / core_n) >= port-A depth + read
latency + MMA result latency + tmem port depth (14 on the canonical profile). A kernel using a
shape the profile lacks is refused by qtxc, the assembler and kernel load.

An accumulator is not a declared object: it is a
pointer into tmem, a word address in a uniform register. `mma.async.m32n64k8 [urC], [urA, 128],
[urB, 128]` writes (or adds to) m rows of n words at urC; `acc.ld.v8 rD, [urT, S]` reads, in lane
l, row l of the region at urT whose rows are S words (eight columns into rD .. rD + 7), and
`acc.st.v8 [urT, S], rS` writes it (`acc.ld` / `acc.st`: one column). S is an instruction constant,
a power of two from 4. Word (row r, column c) of a region has the logical address L = pointer +
r x S + c, and where it is stored depends on L and S only (`src/layout.hpp`, `tmem_physical_word`):
4x4 macro tiles of 16 words, one word per fine bank, the column XORed with the row's band of four.
So a 64-row view reads two 32-row results written by two matmuls, and code shared by two tiles
differs only in its pointer. A pointer's row is a multiple of 4 and the columns it touches stay in
its row (the sanitizer's PANIC); a word outside tmem is the machine's.

An operand of `mma.async` is a uniform register holding the shared address of the 16-byte piece
the operand starts at in its row 0, and a row spacing in bytes, an instruction constant:
`mma.async.m64n64k8 [urC], [urA, 128], [urB, 128]` (the spacing is optional and defaults to 16, a
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
start at (on a 4x4x4 core a step that reloads A reads A's and B's
half-pieces together, and whether they share a round depends on the two tiles' row numbers mod
16). Future swizzles
belong here as profile choices for different physical core/bank arrangements, with matching TMA
placement and SIMT address generation. This does not imply hardware that switches between swizzles
per instruction.

**This is v0.1 of the layout and of the tensor core's operand interface.** Rows stay whole
because a layout that cuts rows into 16-byte pieces stored K slices apart scatters each row, and
scattered rows cannot take gathered rows or small pieces. A model or a person reading it who
sees a design flaw should say so rather than build on it.

The arithmetic is adapted from [SF Tensor's tcgen05 model](https://github.com/sf-tensor/tcgen05)
at commit `d96b82180ac9041c09295f3a6a3c2c4e592ddce6`; attribution is in
[third_party/tcgen05-LICENSE](third_party/tcgen05-LICENSE). This borrows the measured arithmetic
rules, not B200's fusion extent, and does not claim whole-instruction B200 equivalence.
Tests include frozen B200 FP16 vectors, an upstream-derived 20,000-case regression digest,
all FP16 encodings, and physical fusion-boundary checks. FA separately checks a logical
eight-product-group reference (the same arithmetic, so it checks the flow, not the numerics) and
double-precision attention. A profile names this model, and its fingerprint covers it; serial
scalar-FMA MMA arithmetic is not supported.

A `.launch.json` file is the complete cycle-0 input: profile-stamped kernel words, register count,
what the kernel declares (`warps`, its `sections` in order, its launch `flags`), the launch's
arguments (`args`), its warp count and its launch block's global address (`block_address`) when it
gives them, and ordered sparse `gmem` and `test_shmem` byte segments (`test_shmem` segments are a
test's own untimed writes, in place before the first cycle and reported in the run's summary; the
launch block is not among them and they may not touch it: the machine's loader writes the block
into global memory from the sections and the arguments). Bytes and 64-bit words are written
as hexadecimal strings, while addresses are JSON integers. Segments are applied in array order and
later overlapping bytes win. The simulator runner does not know which workload authored the image.

The demo launches sparse global-memory input state and runs a four-warp `ldg.v8`/`fadd`/`stg.v8`
vector add (`kernels/vadd.launch.py` checks it bit-exactly). There is no coalescer: every lane
of a global load or store is its own 32-byte sector request, in lane order.

- `ldg rD, [rA + imm]` / `stg [rA + imm], rS`: each active lane names one word (4-byte aligned);
  a load keeps that word of its sector and drops the rest, a store writes it with byte enables.
- `ldg.v8 rD, [rA + imm]` / `stg.v8 [rA + imm], rS` (per lane): each active lane
  names one whole sector at its own address (32-byte aligned; the machine PANICs otherwise; on a
  profile whose sectors are not 32 bytes qtxc, the assembler and kernel load refuse both) and
  the sector's eight words are `rD .. rD+7` (`rS .. rS+7`) at that lane. A contiguous 512-byte
  block is the case "lane l's address is base + 32 l"; any other set of sectors is a gather. Two
  active lanes of one `stg.v8` naming the same sector is the sanitizer's PANIC, as two lanes on
  one word is for `stg`. Both forms can be masked (`@p`) and used under a partial active mask.
- `ldg.uni rD, [urA + imm]`: one request, the word fanned out to every lane.

The LSU (the profile's `lsu` section) has one buffer on each pipe and no register that serves
both: a **load buffer** between the incoming FIFO and the register file's write port, and a
**store buffer** between the register file's read port and the outgoing FIFO. A buffer is held
only while data is in it; the command slots (`lsu.active_commands`) hold the addresses and send
requests ahead, so one transfer's memory latency hides behind another's data. Every load returns
on one ordered read stream (request order), so the load buffer passes from one command to the
next and returning data always has somewhere to go.

A v8 instruction's lanes pass through the buffers in fixed groups: `lsu.load_group_lanes` and
`lsu.store_group_lanes` (1, 2, 4, 8 or 16; 4 and 4 in the canonical profile). A buffer is 256
bits per lane of a group. A load group is collected (its sectors arrive one lane after another)
and then written by eight register writes, `rD + j` with the group's lanes enabled, one a cycle,
taking write ports the way the other asynchronous writers do. A store group is read by eight
register read beats, `rS + j`, one a cycle, and then its sectors go to the outgoing FIFO; the
source registers stay protected until the last group has read them. Bigger groups cost buffer
bits and save register-file port time: 16 / group x 8 writes (or reads) per instruction. A
scalar `ldg` uses the load buffer for as many lanes as it has words (all sixteen from a group of
two up); a scalar `stg` brings all its lanes at dispatch, so the store group is at least two
lanes, and a scalar `stg` is accepted only when no store still has data to move.

The memory controller is a finite FIFO with fixed latency and ONE fixed-width data bus, shared by
reads and writes and by the LSU and the TMA. `gmem.turnaround_dram_ticks` (2) is the minimum idle
time between the last data beat in one direction and the first beat in the other (ticks the bus
was idle anyway count); the summary's `gmem bus:` line counts the changes of direction and the
ticks waited. This is the replaceable timing policy in front of the eventual bank/row scheduler,
not a fixed-latency whole-load shortcut.

**The TMA interface below is v0.1: a good start.** It has been
built on in two FlashAttention kernels, but its instruction form, operand
read and descriptor fields are still open to change. A model or a person reading it who sees a
design flaw should say so.

The TMA moves tiles through that same controller and through contended shmem port A. A 64-byte
descriptor in shmem describes one tensor and its tile: the global base, the element size (1, 2 or
4 bytes; 8 and 16 are reserved), four dimensions d0..d3 (an unused one is 1) with strides s1..s3 in
elements (dimension 0 is the contiguous one), and the tile -- `rows` runs of `length` elements
along dimension 0, the rows stepping along a named row dimension (1..3) -- with its shared-memory
layout: `dense` (row after row) or `swizzle64` (the rules above), and for swizzle64 whether it is
transposed (not transposed: the tile's rows are the stored rows; transposed: dimension 0's
positions are the stored rows and the tile's row index runs along them). The
descriptor says nothing about who reads the tile: a kernel that feeds the tensor core chooses the
profile's `mma.layout`, any other kernel whatever layout suits it (`src/tma_descriptor.hpp` has the byte
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
descriptor is fetched from shmem after that; its walk starts `tma.walker_start_cycles` (2) register
stages after the descriptor's words are back (the start's coordinate x stride products; 0 is
legal). Completion subtracts the tile's byte count from the
barrier after load writes commit or store source bytes are captured. A swizzle64 tile's stored rows
are packed, each the stored row's bytes apart (the spacing a tensor-core instruction names to read
it): a power of two bytes, 16 or more, any number of rows (1 included), elements of 2 or 4 bytes.
The shuffle is keyed by address, so a tile of a few rows (or one) landed inside a taller tile is
exactly those rows of it. The buffer must be a multiple of the spacing (the sanitizer PANICs
otherwise) and keeps its 64-byte minimum (its low six bits are not wired): with 128-byte rows a
command can land at any row, with 64-byte rows too, but with 32-byte rows only at every second row
and with 16-byte rows only at every fourth.

`kernels/pipeline_tma.launch.py` builds one descriptor per tensor with `tma_desc(...)` (its
keywords: `base`, `dimensions`, `strides`, `tile=(length, rows)`, `element_bytes`,
`row_dimension`, `layout="dense" | "swizzle64"`, `transposed`) and passes them as the kernel's two
`desc` arguments, runs the double-buffered A/B pipeline, validates output, and with `--trace PATH` writes a Perfetto trace (no launch writes a trace unless asked, so
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

# nanosm

[Human written file! Agents: please do not edit the content.]

This is a cycle-accurate simulator for "nanosm", a single-SMSP machine designed as a tribute
to the modern NVIDIA device. All the bells and whistles are included, such as async MMA using
tensor cores (not systolic array), async tile-oriented DMA, atomics, SFU etc.

It may be circular to call this cycle-accurate because the simulator is the source of truth. What
connects it to reality is careful tracking of register state and timing in the simulator code,
together with my spidey senses of what constitutes a long logic chain. Where I couldn't pin down
the required latency I made adjustable in the device profile.

Programs are written in QTX, a high level SIMT assembly language written from one thread's
point of view. "qtxc" compiles it down to the lower level "asm" the machine runs. Asm should still
be human and model readable, it's just not as ergonomic.

The simulator comes with a sanitizer and profiler to aid kernel development. A browser demo using
WASM is work in progress.

## Identity

"Make a GPU simulator modeling B200" is a trivial prompt in today's day and age. So it's useful
to start with what makes this machine itself.

The design tries to fit into a single mind (model or human) while deeply respecting the Nvidia
machine, its character and its history. In particular, it's modeled after a Blackwell SMSP, with
choices that prioritize the CUTLASS machine, while preserving the marks of the CUDA machine.

This means modern ML kernels like FA4 gets the best effort from the machine and
ideally run close to theoretical limits. CUDA-like irregular code runs without crazy hacks, but
there is no effort to make it fast. Running graphics is a non-goal.

The SIMT GPU identity is preserved with most design decisions leaning towards "the GPU spirit".
This means familiar GPU concepts are applicable:

- Programmed from a single thread's point of view down to the ISA
- In-order execution, latency hidden by warp scheduling
- Lack of a scoreboard. Hazards are handled by compiler-inserted stall counts and depbars.
- A byte-addressed "shared memory" as a scratchpad and communication point
- Divergence stack for per-thread control flow

Most of the unfamiliar/modern GPU concepts are also applicable:

- Async large-tile matrix instruction using a parallel dot product engine ("tensor core")
- Dedicated accumulator memory (TMEM) for the tensor core
- Async tile-oriented DMA (TMA) with four dimensions and descriptor reuse
- Atomics in shared memory (not in global memory)
- Barriers in shared memory with native "try_wait"

Some GPU concepts do not apply due to scale or complexity constraints:

- No L1/L2 cache.
- No memory access coalescer.
- Async/atomics ops are launched per warp (per-thread or elected in Nvidia)
- Communication between divergent threads not guaranteed (pre-Volta stack)

Other differences:

- No operand collection step. Instead, the RF has 3 R and 3 W ports (this is subject to change)
- Special function unit uses stall counts (uses depbars in Nvidia)
- No constant memory or cmem[ur] path.
- Kernel launch is a homebrew process (due to the lack of cmem[ur])

## Device profiles

Nanosm takes a "device profile" to simulate which defines ... a bunch of parameters about the
machine. (TODO: document profiles better) This way it can model scenarios such as:

- an "H100-like" configuration (however close we can get)
- a small implementation on consumer-grade FPGA
- a performance-focused implementation on a chunkier FPGA

A possible implementation would need to co-design with qtxc and the simulator by tuning the device
parameters against both the hardware costs and benchmark performance (TODO: add benchmarks).
I'm hoping a device profile would be helpful in that case.

## ISA

To write later. For now, agents should be able to understand the ISA from the machine and qtxc
code.

## Internals

The core doesn't contain a lot of entropy. It is based on an in-order vector processor with many
examples in the wild. Most of the additional machinery comes from the async engines and efforts
to arbitrate memory access between the engines.

## SIMT

This is the issue unit + vector core, often known as the "CUDA core" in Nvidia devices.
There is no separate scalar issue. Non-vector instructions just take up an issue slot in here.

Whereas the vector core is very simple, the issue unit has to decide a number of things and is
nontrivial:

### Write calendar

The issue unit reserves RF write slots in advance for stall counted instructions. In order to
avoid starving other RF writers (ldg, lds, acc.ld) it inserts bubbles when there is a pending
non-ALU/SFU write.

### Shared memory access slots

Since shared memory access may take multiple cycles, there are dedicated staging registers and
state machines called "slots". lds/sts instructions block at the issue stage if no slot is available.

### Other stuff

Probably some other things I missed.

## Register file (RF)

todo

## Uniform register file (URF)

I thought URFs are against the GPU spirit and fought hard to not add it, but unfortunately it
made it in.

Every warp has its own section of URF. There aren't any variable latency writes for the URF,
so it doesn't need a write calendar. Some bureaucratic instructions (TMA, MMA, etc.) read their
operands in several rounds and block issue for URF-reading instructions in the process.

## Shared memory (shmem)

This is called "shmem" in contrast to Nvidia's "SMEM" purely to confuse programmers, but it
serves the same purpose.

The shmem is hardcoded to have 2 read and 2 write ports. This is a very FPGA minded decision
but it makes the machine simple to model and this is probably not going to become an ASIC so I'm
OK with it.

Read port B is reserved for the tensor core while write port B is reserved for TMA loads.

Port A writes and reads are arbitrated between different units. (TODO: list the units)

## RMW and barriers

todo

## TMA

Just a TMA with 4 dimensions. Too bored to explain it in detail.

## MMA (tensor core)

goes biong biong with dot product engine and walks over the big tile wheee

## TMEM

probably requires more explanations

--- old material (ignore below this, probably outdated)

- Issue
- SIMT (vector pipeline)
- Shared memory
- TMA
- MMA
- TMEM

## identity
- Blackwell-like asynchrony on Fermi-like control. One SM. Small enough to hold in one head.
- No L1/L2 cache
- Pose nvidia-like difficulty (pipelining, warp specialization, banking); delete nvidia-like paperwork
- Configurable where possible: DRAM behavior, matmul sizes, instruction latency

## Engines
1. SIMT (warps, vector lanes, RF<->RF, gmem<->RF, shmem<->RF, tmem->RF)
2. DMA (async bulk-copying device, gmem<->shmem)
3. MMA (tensor core, tmem->tmem, shmem->tmem)

## Memory
1. RF
2. tmem, holds MMA accumulators
3. shmem, provides warp/engine communication and staging area
4. no L1/L2
5. gmem (dram)

## RF
3r 3w banked RF. The canonical profile says three separate 1r 1w banks. The reason to have this is to avoid
the need for an operand collector while having peace of mind about draining MMA results on time.
Should be OK if we have 2/3rd the RF capacity of an nvidia device? Maybe we could do with
fewer warps by skipping operand collection latency...

SIMT, global loads and MMA drain all want to write to RF so conflicts can occur. SIMT always
takes precedence in writes as it needs to satisfy a stall count. Global loads and MMA drain can
wait as they fire a depbar for completion anyway.

To avoid completely starving ldg and acc.ld stall SIMT instructions targeting a bank if the bank
has pending write from another source.

We probably need to allocate write slots for conflict resolution between SIMT warps. Stall
if unable to grab a write slot in the future.

No operand collector implies that an instruction can't read from the same bank twice. This is
an old-GPU behavior but eh what to do. I feel like the compiler can handle it?

There is no dynamic RF-ready scoreboard. The qtx compiler emits fixed stall counts and depbar
waits. Hand asm that reads or overwrites a register before its scheduled write commits PANICS;
only actual resource contention (for example a reserved write slot) may structurally stall issue.

Bank mapping is reg mod n_banks, so three consecutive registers cover the three banks in the
canonical profile. tma coordinate capture (rfc01) relies on this. Capacity is rows per physical
bank; one row holds one lane-wide 32 bit vector register. Allocation uses the kernel's exact
contiguous register count, then active_warps * allocated rows must fit every bank. There is no
independent magic max_warps field. Add scheduler warp slots only if we actually build that resource.
The canonical 176 rows per bank are provisional; with a 64-register allocation they derive the old
eight-warp result instead of declaring it independently.

## Shmem
<64KB software managed cache. Banked to provide high bw access. Typically has n_banks = n_lanes.
Simple bank conflict handler:
  Any conflict: serialized access (>n_banks latency)
  No conflict: full throughput (>1 latency)
  Broadcast access (every lane reads the same addr): >1 latency

There are many configurations this can take. This may be swept later, but by default, banks have
one RW port and one R port. Port A is shared between SIMT and DMA, and port B is dedicated to MMA.

(This comes from FPGA-brained thinking: muxes and routing are expensive, and both SIMT and DMA
may write to any bank, so this maintains some hope that they may share a crossbar. Maybe it's
not that expensive on an ASIC, in that case, feel free to ask your claude to change it.)

Write conflicts are UB. For R/W conflicts i think we should behave read-first, but dropping this
constraint may bring hardware performance, at least on an FPGA.

Provides atomics. An atomic request blocks others until it's served. Implemented by dedicated
RMW engine on port A. MMA can line up on port A to fire its mbar events.

No FIFO on port B. Port A currently has one active request slot and no input FIFO. lds/sts stall
structurally until that slot is free, but retire as soon as the request is accepted; they do not
block their warp while bank-conflict rounds execute. Both set a depbar. lds clears it after the
final RF write commits, sts after the final shmem write commits.

## Sync items
1. depbar (good old nvidia style depbar, encode set/wait in instruction. Warp-local: six counters
   next to the warp's PC in the scheduler. An accepted async instruction atomically increments its
   set counter and its architectural completion decrements it. A wait mask makes the instruction
   ineligible while any selected counter is nonzero. Engines clear them, but the namespace is the
   warp's; qtxc allocates indices statically, no cross-warp ownership protocol. Counter packing
   width is not architectural yet; underflow PANICS)
2. fence.sh (clear depbar when every port A write accepted before the fence has committed)
3. fence.g (clear depbar when no more global stores in queue)

Fences are machine-wide: they cover writes from every warp and engine, not just the issuing warp.
A warp's own ordering is already free (in-order issue plus depbar), so a fence exists only to make
other agents' writes visible; warp-scoping it would buy nothing.
4. shmem atomics
5. mbar (sync device provided by shmem atomics)

No hbar or hardware barrier file in v0. Mbar state stays in shmem. A small profile-sized exact
wake table holds only the warps currently parked in mbar.try_wait; it is not sized from RF
residency.

## Mbar
mbar is an opaque 8-byte-aligned 64 bit object in shmem. Physically it's two adjacent 32 bit
words in two banks. From low to high its fields are phase:1, valid:1, expected arrivals:15,
remaining arrivals:15, and remaining tx bytes:32. Thus one phase accepts at most 32767 arrival
events and 2^32-1 exact transaction bytes. The valid bit distinguishes zeroed shmem from an
initialized barrier.

mbar.* is warp-uniform: thread 0 makes one request on behalf of the warp, test_wait and try_wait
broadcast their predicate, and an incomplete try_wait sleeps the whole warp. Executing one under
a nonuniform mask PANICS. expected counts arrival events from warps or engines; waiting warps and
active lanes are not participants and don't affect the count.

The RMW engine has no input FIFO, matching port A's single-request discipline. It serves one
operation at a time; a SIMT mbar operation stalls structurally at issue while port A or the engine
is busy, exactly like lds/sts, and each engine (DMA completing tx, MMA committing) holds at most
one pending mbar event of its own and lines up for the free engine ahead of new SIMT work. Requests
live with their requesters, so there is no shared queue to size. One update takes two core cycles:
read both words, then validate/update/write both words and broadcast any phase change. A warp's own
sts before mbar.arrive needs no fence: both serialize through port A, so the arrive cannot start
until the store's rounds have committed. Since atomics block the banks anyway, making mbar 32 bit
saves four bytes per barrier but doesn't buy useful concurrency. 64 bit also avoids prematurely
squeezing the arrival and exact-byte tx counts.

mbar.init [x], n: phase=0, expected=n, remaining=n, tx=0.
mbar.arrive [x]: atomic remaining--.
mbar.arrive.expect_tx [x], bytes: atomic remaining-- and tx+=bytes. These have to happen in
one RMW bc doing arrive first could briefly complete the barrier before tx is added.

complete_tx [x], bytes, parity isn't a SIMT instruction. DMA does it internally after the
corresponding shmem writes have actually committed. The engine request carries the phase captured
for that transfer; it atomically does tx-=bytes and PANICS if the mbar has already changed phase.

MMA operand-consumed clears a depbar belonging to the issuing warp; it doesn't touch an mbar.
Software may wait for that depbar and do an ordinary mbar.arrive to hand a consumed shmem stage
back to another warp. mma.commit [x] snapshots the issuing warp's earlier MMA commands and makes
the engine line up an arrive RMW on port A after their results have committed to tmem. This exists
for cross-warp result-ready notification; a warp that only needs its own completion uses a depbar.

After every update, if remaining==0 and tx==0, phase flips and remaining reloads from expected.
mbar.test_wait p, [x], parity is a plain single-cycle port A read of the word holding phase; it
sets p=(phase != parity) and never blocks. The RMW engine writes phase last, so a test racing an
update observes either the old or the new phase, and both are legal outcomes.

mbar.try_wait p, [x], parity first reserves one of the profile's mbar_wait_slots and installs
{warp, address, parity, pdst}, then performs the same phase test. If complete it writes p=true and
immediately frees the slot. Otherwise the warp parks without further port A reads. When the RMW
engine flips a phase it broadcasts {address, new phase}; every exact matching slot writes p=true,
wakes its warp and frees itself. Install-before-probe closes the check/sleep race. If all slots are
occupied, a new try_wait stalls structurally before its probe; every warp may use the instruction,
but only the profile-defined number may be parked concurrently.

An incomplete wait that reaches mbar_try_wait_timeout_cycles writes p=false and frees its slot.
Software loops on false for a long wait, for example `W: mbar.try_wait p0, [x], parity; @!p0 bra.uni
W`. A phase flip on the deadline wins and returns true. Exact wake is correctness, not merely a
latency hint, so timeout needs no second shmem probe. The canonical profile starts with eight slots
and a 64-cycle timeout. Slot-full stalls, parked cycles, initial probes, exact wakes and timeouts
are separate trace counts.

One-bit parity names only the current incomplete phase or the immediately preceding phase.
Software must successfully observe a phase before arrivals for the phase after it begin; a retry
loop is not an ABA defense. Software keeps and flips parity only after a successful wait.

The RMW engine owns barrier state and phase-change events; the waiter table owns only useful sleep
state. Underflow, overflow, reinitializing or repurposing an mbar with a live waiter, completing
bytes against the wrong phase, etc PANIC. A wait over several mbar addresses is not an instruction
in v0; sequential try_wait loops do wait-all and unusual state machines can use mbar.test_wait.

## Memory ordering

Within one path, accesses complete in issue order. Rules exist only where paths cross. Even in
that case we try to stay close to the "natural law" ie whatever is easier to implement, unless
there is heavy performance cost

TMA complete_tx is issued only after its corresponding port A writes commit. Therefore a phase
completion observed by mbar.try_wait or mbar.test_wait is enough to gate MMA reads of that tile;
fence.sh is not needed on the TMA -> mbar -> MMA path. fence.sh is for explicitly ordering ordinary
port A writes before a later cross-path handoff.

Further rules get invented on the road (example kernels), not legislated here.

## SIMT core
Not much to describe. L lanes, W warps, pre-Volta reconv stack (we don't care about communication
between divergent branches as it's not optimal to begin with)

v0 branching is uniform-only: bra is unconditional, bra.uni is the only predicable opcode, and a
bra.uni whose predicate is not warp-uniform PANICS. Divergence and the reconv stack stay reserved.
setp.{eq,ne,lt,le,gt,ge} writes a per-lane predicate file p0..p6 at alu latency; predicate RAW/WAW
before the scheduled write commits PANICS like any RF hazard (no scoreboard). Branch displacements
are signed byte offsets from the following PC.

Every branch is a block boundary in the toolchain: qtxc makes it wait all live depbars and carry a
stall that covers outstanding fixed-latency writes, so any control transfer arrives with clean
timing state and the linear scheduling scan stays sound. This is software-visible: depbar-based
async (ldg prefetch) cannot stay in flight across a backedge. Deliberate v0 cost — mbar-based
async (tma) is untouched by it, so pipelined loops keep their engine overlap.

For now stick to shmem as the cross-lane/cross-warp channel. No shfl

SFU instructions added as needed. Instruction latency/throughput configurable

## MMA (tensor core)

`mma.async[.init] ACC, [rA], [rB]` captures two uniform shmem byte addresses and autonomously
walks the profile-static M,N,K shape in K/N/M order using the physical core shape. Both inputs
are fp16; each output uses ascending-kk fused accumulation into fp32 tmem. `.init` starts the
command from zero. The canonical 64x64x8, 4x4x4 profile uses RFC05's 64-byte-aligned XOR layout;
`dense` is retained as a profile-fingerprinted diagnostic layout and unsupported XOR shapes reject.

Physical input steps use shmem port B only. Distinct 32-bit words per bank determine conflict
rounds (same-word half accesses broadcast); the canonical XOR footprint is one round. Result
records are finite and retire to the targeted coarse tmem bank after `result_latency_cycles`.
A matching in-flight output block causes an explicit recurrence bubble. Adjacent commands may
feed while an older tail retires; there is no implicit command-boundary drain.

The attached depbar clears after the final input step, meaning operands consumed rather than
results ready. `mma.commit [rM]` is an ordered queue marker which presents one ordinary mbar
arrival only after all earlier tmem writes retire. MMA and acc.ld/st exclude one another on the
same coarse bank, including outstanding result writes; queued commands do not claim ownership.
Queue depth, initiation interval, result latency, logical/core shapes, and layout are in the
device profile and fingerprint.

## acc.ld (accumulator load)
- Load-shaped warp instruction, NONDESTRUCTIVE (pure read — two-pass softmax is legal): accumulator → RF
  ONLY, row-per-lane (makes softmax shuffle-free), variable latency, completes via dep-barrier. Epilogue
  applied in flight: none / scale / exp LUT / convert.
- Example: `acc.ld.exp r8, acc0, bias=r4, db=4` — lane m receives exp(S[m,:] - bias[m]) streamed over the
  tile; dep-barrier 4 clears at the last beat. A load that reads the accumulator and applies an op.
- accum→shmem = acc.ld + sts through the RF (tmem→smem doesn't exist as a datapath on blackwell either).

## memory
- Movement graph, 4 edges: (1) tma gmem⇄shmem (2) operands shmem→engine (3) acc.ld accum→RF (4) lds/sts.
- One writer type per structure: shmem{tma, sts} accum{mma} rf{alu, ldg, acc.ld} gmem{stg, tma-out}.
- shmem: banked, deterministic conflict serialization. No queues needed — FIFOs never create fence
  obligations, only second paths do. Access paths are the Shmem section's ports: A = SIMT + DMA + RMW,
  B = MMA.
- gmem: uncached, ldg/stg kept. Vocabulary: SECTOR = 32B aligned unit; REQUEST = one sector in the
  controller queue; BEAT = one bus cycle (a sector transfers in ceil(32/B) beats). Sector rule: a warp
  access becomes one request per unique sector touched; broadcast load = 1. Multiple lanes in one stg
  writing the same 32 bit word PANIC before the command enters the controller. No L1/L2: locality is
  software's job. 1 outstanding ldg per warp.
- Current first controller is the honest placeholder: finite sector capacity, fixed ready latency,
  FIFO selection and one shared fixed-width bus. A command admits all of its sectors atomically or
  structurally stalls; a command bigger than the whole controller PANICS. Replacing FIFO/fixed-ready
  with the bank/row policy below must not change coalescing, request shape, bus beats or depbar semantics.
- Core cycle is the simulator master time. Integer core_clock_hz and dram_clock_hz in the device
  profile drive a rational phase accumulator, which may produce zero, one or several DRAM ticks in
  one core cycle. Controller ready latency and bus bytes are per DRAM tick; instruction latencies,
  stall counts, issue, RF and on-chip engines remain in core cycles. No floating-point time.
- atomics: the CUDA menu (add/sub/min/max/and/or/xor/exch/cas u32/s32 + f32 add); old value returns via
  dep-barrier. shmem = globally blocking RMW engine on port A; gmem = atomic station at the controller
  (read beat + ALU + write beat, per-address ordered).
- gmem is a DRAM-controller-like model — structurally honest, numerically fictional: published
  addr→(bank,row,col) mapping; per-bank open row + busy-until (L_hit / L_miss declared); controller queue
  capacity Q; FR-FCFS with deterministic tie-breaks (hits > misses, reads > writes until write-queue
  watermark, oldest, request-id); shared bus = bandwidth bottleneck; read/write turnaround penalty;
  optional deterministic refresh. Wall: no bank groups / command timing / channels / ECC. Ordering:
  out-of-order service but PER-ADDRESS ordered (write-queue forwarding); on-chip paths remain true FIFOs.
  NO stochastic noise, ever. Trace: per-bank tracks, row hit/miss coloring, turnaround marks, queue depth.
- tma: blessed-tile bulk copy. Request form: trash/rfc01.md (closed, synced here). 64B shmem descriptors, rank 1-3; the
  instruction carries a descriptor address and one coordinate-register-group base. OOB =
  zero-fill reads / dropped writes; transpose-at-load for K.
  Engine queue depth: CONFIGURABLE, default 1 (ISA-invisible knob, same as the mma engine's).
  No cp.async analog: irregular gmem→shmem staging goes through the RF, deliberately.

## sizing
- Jurisdiction (subject to change): absolute sizes are FPGA (fit 7-series: DSP→tile, BRAM→srams, clock);
  ratios are nvidia-shaped (declared gmem ≈ 2-4 B/cyc, few hundred cycles latency).
- ONE canonical device profile, published as a datasheet. Configurable in emu2 (designer), fixed for users.
  The ground truth is profiles/nanosm-v0.json; unit-bearing names distinguish core cycles from
  DRAM ticks.

## implementation
- Launch is only initial state: encoded kernel words, active warp count, sparse shmem image and sparse
  gmem image. Memories are zeroed, image segments are applied in order before cycle 0, then warps start
  at pc 0. Tooling currently serializes this as self-contained nanosm-launch-v0 JSON with hex words
  and byte strings. The format is not architectural; loading it produces the same Launch struct.
- Demo launch state is authored in a Python file beside the qtx kernel. Python calls the simulator
  in-process through a memory-only C ABI; it does not spawn the command-line runner. A self-contained
  launch JSON may be emitted for replay, but it is a transport artifact, not competing ground truth.
- Bookkeeping (mbar words, dep-barriers, counters) is tracked as state, never as a busy resource.
  Mbar try-wait slots are the exception: they are a profile-sized device, and an instruction stalls
  structurally when every slot is occupied.
- Kernel completion: every warp halted, DMA/MMA/RMW engines idle, controller queues (including the
  write queue) drained. The harness reads gmem only after this, so a trailing fence.g is hygiene,
  not load-bearing.
- Counters: every resource accounts every cycle, mandatory stall reasons, attribution sums to total (a test).
- Trace is typed at the event source (`issue`, warp state, depbar, shmem request, gmem command/sector/bus,
  queue depth), never reconstructed from rendered debug strings. Constant-size summary counters are
  always on. Perfetto/Chrome JSON capture is opt-in, repeated warp states become duration intervals,
  bus beats are debug-only, queue counters emit only on change, and a hard event limit emits an
  explicit truncation marker. Scheduler attribution sums to cycles; warp attribution sums to
  cycles * active_warps.
- No traps: every error PANICS deterministically (pc, reason, cycle, full state) — hazard violation, mbar
  misuse, OOB (incl. shmem, no wrapping), misalign, stack overflow, cycle cap (the whole sandbox).
- Initial state = FF reset: all architectural state zeroed; gmem = the loaded image.

## open
- Per-row scale operand on mma accumulate (folds FA O-rescale). Paper kernel adjudicates.
- Remaining writing: cvt list, sread set, numbers (SFU, acc.ld bandwidth,
  latency table, clock ratio, DRAM preset).

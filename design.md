# nanosm identity v0 (actually, v3 but still dumb)

Cycle-accurate simulator + perf traces for a single-SM GPU.
The sim is the spec: no silicon truth behind it, but we still try to keep implementation in visible range.
Milestone: it's a thing when it runs flash attention and emits a trace.

## identity
- Blackwell-like asynchrony on Fermi-like control. One SM. Small enough to hold in one head.
- No L1/L2 cache
- Pose nvidia-like difficulty (pipelining, warp specialization, banking); delete nvidia-like paperwork
- Configurable where possible: DRAM behavior, matmul sizes, instruction latency

## isa
- 64-bit instructions, ifetch from single-cycle icache. Kernels have to fit in icache. The word
  format's spec is the constants and checks in src/isa.hpp; no separate format doc. Registers are
  8 bit. The 38 bit payload is opcode-shaped: RRR uses three registers and leaves 14 bits reserved;
  RRI/load/store uses two registers plus a signed 22 bit immediate. Do not spend bits on fields an
  opcode cannot use
- Families: mma.async, acc.ld (+epilogue), tma.async, sync (mbar.* / fence / atomics), simd core
  (alu/fpu/sfu/setp/bra/lds/sts/ldg/stg/cvt).
- Control bits: 5 bit stall count for fixed latency (room for FPGA-priced SFU), one set-depbar
  index and a 6 bit wait mask
  for variable latency. set increments on accepted issue; completion decrements; wait blocks while
  ANY selected counter is nonzero (so the instruction waits for ALL named dependencies)
- qtx has semantic instructions and no controls. qtxc schedules it into human-editable asm with
  explicit {pred, stall, set, wait}; qtxas encodes that into annotated 64-bit hex. The simulator
  fetches and decodes the hex, not the qtx IR. qtxc, qtxas and the simulator consume the same
  human-readable JSON device profile. asm and hex carry its name plus a fingerprint; qtxas and the
  simulator reject a mismatch because scheduled stalls are profile-specific. qtxas derives the
  kernel's contiguous register allocation as highest referenced register + 1 and hex carries it.
  The fingerprint is 64 bit FNV-1a over the canonical compact JSON serialization; it is stable
  across standard-library implementations
- Written from one thread's point of view. Convention of per-warp PC & masked execution
- PTX-like language, "qtx" for now. qtx -> asm -> hex, compiler takes gpu configuration
- .v8 ldg/stg exist

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

No hbar or hardware barrier file in v0. Mbar state stays in shmem; blocking mbar.wait keeps only
one sleep/wakeup record next to each warp's scheduler state.

## Mbar
mbar is an opaque 8-byte-aligned 64 bit object in shmem. Physically it's two adjacent 32 bit
words in two banks. Exact field widths later but conceptually it contains {phase, expected
arrivals, remaining arrivals, remaining tx bytes}.

mbar.* is warp-uniform: thread 0 makes one request on behalf of the warp, test_wait broadcasts its
predicate and wait sleeps the whole warp. Executing one under a nonuniform mask PANICS. expected
counts arrival events from warps or engines; waiting warps and active lanes are not participants
and don't affect the count.

The RMW engine blocks shmem service, reads both words, updates the decoded 64 bit state, writes
both words, and unblocks it. Since atomics block the banks anyway, making mbar 32 bit saves four
bytes per barrier but doesn't buy useful concurrency. 64 bit also avoids prematurely squeezing
the arrival and exact-byte tx counts.

mbar.init [x], n: phase=0, expected=n, remaining=n, tx=0.
mbar.arrive [x]: atomic remaining--.
mbar.arrive.expect_tx [x], bytes: atomic remaining-- and tx+=bytes. These have to happen in
one RMW bc doing arrive first could briefly complete the barrier before tx is added.

complete_tx [x], bytes isn't a SIMT instruction. DMA does it internally after the corresponding
shmem writes have actually committed. It atomically does tx-=bytes.

MMA operand-consumed clears a depbar belonging to the issuing warp; it doesn't touch an mbar.
Software may wait for that depbar and do an ordinary mbar.arrive to hand a consumed shmem stage
back to another warp. mma.commit [x] snapshots the issuing warp's earlier MMA commands and makes
the engine line up an arrive RMW on port A after their results have committed to tmem. This exists
for cross-warp result-ready notification; a warp that only needs its own completion uses a depbar.

After every update, if remaining==0 and tx==0, phase flips and remaining reloads from expected.
mbar.test_wait p, [x], parity is a plain single-cycle port A read of the word holding phase; it
sets p=(phase != parity) and never blocks. The RMW engine writes phase last, so a test racing an
update observes either the old or the new phase, and both are legal outcomes.

mbar.wait [x], parity performs the same test. If the phase is already complete it retires;
otherwise the warp sleeps with one {mbar address, parity} wait record in the scheduler and does
not keep reading port A. When the RMW engine flips a phase it broadcasts {address, new phase} and
every matching warp wakes. Installing the wait record before checking the phase visible after
that cycle's RMW step closes the check/sleep race. Software keeps and flips the parity bit after
either kind of successful wait.

There is no central waiter structure or hardware barrier file. The RMW engine owns barrier state
and phase-change events; the scheduler only owns one wait record per warp. Underflow, overflow,
reinitializing or repurposing an mbar with a live waiter, completing bytes against the wrong phase,
etc PANIC. A wait over several mbar addresses is not an instruction in v0; sequential mbar.wait
does wait-all and unusual state machines can use mbar.test_wait.

## Memory ordering

Within one path, accesses complete in issue order. Rules exist only where paths cross. Even in
that case we try to stay close to the "natural law" ie whatever is easier to implement, unless
there is heavy performance cost

TMA complete_tx is issued only after its corresponding port A writes commit. Therefore a phase
completion observed by mbar.wait or mbar.test_wait is enough to gate MMA reads of that tile;
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

Parallel fused-dot-product-add engine, M/N/K configurable. Physical MpxNpxKp core sweeps over
shmem/tmem inputs managed by the engine. Tentatively 8x8x4, instruction width <64x64x8

M input comes from shmem. Swizzle layout to prevent bank conflicts (need to specify this later).
DMA swizzles gmem layout into mma-friendly layout.

Only provides an async instruction issued by thread 0. If thread 0 is masked, skip (no elect)

The depbar attached to mma.async clears when that command's shmem operands have been consumed.
Commands execute in order, so consumption of the last command in a batch means the batch's shmem
stages may be reused. mma.commit can instead fire one mbar arrival when all earlier results have
committed to tmem.

Commands can be queued. Queue depth is configurable. Stalls via depbar when queue is full

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
  at pc 0. The container or transfer format is tooling and not architectural.
- Bookkeeping (mbar words, mbar wait records, dep-barriers, counters): tracked as state, never as busy resources.
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
- Remaining writing: cvt list, sread set, final mbar field widths, numbers (SFU, acc.ld bandwidth,
  latency table, clock ratio, DRAM preset).

# Writing a kernel in qtx

*Written by models from the code, and checked against it by other models. The pages a person wrote are `README.md` and `design.md`.*

## Running

A kernel `K.qtx` has a launch script `K.launch.py` beside it that gives the inputs, runs
the machine, checks the answer and prints the cycle count and where the cycles went. Add
`--trace out.json` for a timeline to open in https://ui.perfetto.dev.
`build/qtxc profiles/nanosm-v0.json K.qtx K.asm` compiles without running and writes the
scheduled assembly, which shows what the compiler decided for every instruction.

## The language

A kernel is one `.qtx` file: every warp starts at its first instruction and runs until
`halt`. You write instructions on named registers; the compiler, qtxc, numbers the
registers, makes each instruction wait for what it depends on, and places the memory you
declare. Limits are the profile's; qtxc refuses what does not fit and names the line. The
instruction set is `src/isa.hpp`.

```
; OUT[i] = 3 * IN[i] + 1, one unsigned word per lane of every warp.
.param ptr IN
.param ptr OUT

        usread %warp, WARP_ID
        ushli  %warp, %warp, 6          ; 64 bytes per warp: 16 lanes of 4 bytes
        ur2r   %offset, %warp           ; a uniform value into every lane
        sread  %lane, LANE_ID
        shli   %lane, %lane, 2          ; 4 bytes per lane
        add    %offset, %offset, %lane
        ld.param.u32 %in, [IN]
        ld.param.u32 %out, [OUT]
        add    %in, %in, %offset
        add    %out, %out, %offset
        ldg    %n, [%in]
        shli   %twice, %n, 1
        add    %n, %n, %twice
        addi   %n, %n, 1
        stg    [%out], %n
        halt
```

```python
import struct
from pathlib import Path
from nanosm import Launch

HERE = Path(__file__).resolve().parent
IN, OUT, WARPS = 0, 4096, 2
values = list(range(1, 16 * WARPS + 1))

launch = Launch(kernel=HERE / "triple.qtx", profile=HERE.parent / "profiles" / "nanosm-v0.json",
                warps=WARPS, args={"IN": IN, "OUT": OUT})
launch.gmem.u32(IN, values)
with launch.run() as result:
    got = list(struct.unpack(f"<{len(values)}I", result.gmem(OUT, 4 * len(values))))
    assert got == [3 * v + 1 for v in values], got
    print(f"ok in {result.cycle} cycles")
    print(result.summary, end="")
```

## Launch

A kernel declares what a launch must give it, beside its code:

```
.warps 2                     ; optional: written for exactly two warps
.param ptr  OUT              ; the signature, in order: ptr, u32, f32, or desc (a TMA descriptor)
.param u32  COUNT
.mbar  FULL[2], expect 1     ; two barriers, each completing after one arrival
.data  TABLE, f32 0.5, 0.25  ; constant words (f32 or u32)
```

- The host passes one argument per `.param` and writes global memory. It never touches
  shared memory or registers.
- These lines reach shared memory by a copy that runs while the kernel has already started,
  in the order written. An instruction that names one waits for it; code that names none
  runs meanwhile.
- `ld.param.u32 %x, [COUNT]` (or `.f32`) reads a parameter into every lane. Of a `ptr` it
  reads the low word: a global address is one 32-bit register.
- A name is its address in shared memory: `umov %m, FULL[1]`.
- One float constant needs no `.data`: `umov %bits, 0.5`, then `ur2r` for every lane.
- The Python side is `nanosm/launch.py`: `Launch(...)`, `launch.gmem`, `launch.run(...)`,
  `result.gmem(...)`, `result.cycle`, `result.summary`. A launch prints qtxc's warnings itself.
- Float results are the implementation's (rounding and denormals are not specified), so a
  check compares with a tolerance.

## Registers

| | written | holds |
|---|---|---|
| vector | `%x` | 32 bits in each lane |
| uniform | `%x` | one 32-bit value for the warp |
| predicate | `p0`, `p1`, ... or `%name` | one bit per lane |

(`%p0` .. `%p6` are the predicates themselves, so a register cannot have those names.)

- A name exists from its first use and qtxc gives it a number. The instruction decides its
  kind: `add` takes vector registers, `uadd` uniform ones. `ur2r %vec, %uni` copies a
  uniform value into every lane; `r2ur %uni, %vec` takes lane 0.
- One source of a per-lane ALU or FPU instruction may be a uniform register: the second
  (`fadd %d, %x, %u`, `setp.lt.u32 p0, %x, %u`), or either multiplicand of `fma`. A name is
  uniform when the rest of the kernel makes it so. A uniform first source or addend needs
  `ur2r` first. The uniform source takes no bank.
- Every warp has its own registers. Warps share only memory and barriers.
- The vector file is banked and an instruction's distinct sources must sit in different banks.
  qtxc arranges that; where it cannot, it inserts a copy and warns with the line.
- Some instructions take eight neighbouring registers and name the first. Declare the
  group: `.reg %a<8>` gives `%a0`..`%a7`; `.ureg %c<2>` is a uniform group.
- `sread %x, LANE_ID` or `THREAD_ID`; `usread %u, WARP_ID` or `N_WARPS`.

## Masks

`@p0` or `@!p0` before a per-lane instruction writes only the lanes where the predicate is
set (clear): `@p0 add %x, %x, %y`. It costs what the unmasked instruction costs. Uniform
instructions, barriers, the TMA and the tensor core refuse a mask.

## Memory

Shared memory (bytes) and tensor memory (words, the tensor core's accumulators) belong to
the kernel. Declare what you use and qtxc places it:

```
.shmem STAGE[2], bytes=64                  ; two plain buffers: STAGE[0], STAGE[1]
.shmem QBUF[2],  rows=32, cols=64, elem=2  ; two tiles: rows x cols elements of elem bytes
.tmem  S[2],     rows=32, cols=64          ; two accumulators
        lds   %x, [%byte + STAGE[1]]
        umov  %q, QBUF[0]
```

`NAME` is an address and `NAME[i]` an element's. `NAME.bytes`, `NAME.stride`,
`NAME.row_bytes` (tensor memory: `.words`, `.row_words`) are constants. `align=` replaces
the default alignment; `over=A, B` lays an object on others on purpose; nothing else
overlaps. An object is bookkeeping: the machine sees only addresses, and who may touch a
buffer when is the kernel's argument, made with barriers.

Wherever a constant goes it may be an expression (`+ - * / << >>`, parentheses,
`log2(x)`), and `.equ NAME, expression` names one.

## Instructions

`d` is the destination; `b|imm` a register or a constant. In the per-lane integer, float
and compare rows, the second source (for `fma`, either multiplicand) may be a uniform register.

| what | instructions |
|---|---|
| integers, per lane | `mov d, a\|imm`; `add and or xor shl shr d, a, b`; `addi andi ori xori shli shri d, a, imm`. Shifts are logical. There is no subtract, multiply or divide. |
| floats, per lane | `fadd fsub fmul fmax d, a, b`; `fma d, a, b, c` (a x b + c); `ex2 d, a` (2^a); `rcp d, a`; `cvt.pack.f16x2 d, lo, hi`. There is no float compare. |
| compare, per lane | `setp.{eq,ne,lt,le,gt,ge}.{u32,s32} p, a, b\|imm` |
| uniform | `umov d, a\|imm`; `uadd usub ushl ushr uand uor uxor d, a, b`; `uaddi ushli ushri uandi uori uxori d, a, imm`; `usetp.{...}.{u32,s32} p, a, b\|imm` |
| branch | `bra.uni L`; `@p bra.uni L` (every lane must agree); `halt` |
| divergence | `ssy L`; `@p bra L`; `sync` (written by hand today: `kernels/diverge.qtx`) |
| shared memory | `lds d, [a + imm]`; `sts [a + imm], s`; `lds.uni d, [u + imm]`; `atom.{add,and,xor,exch}.uni d, [u + imm], u2`; `fence.sh` |
| global memory | `ldg d, [a + imm]`; `stg [a + imm], s` (a word per lane); `ldg.v8 d0, [a + imm]`; `stg.v8 [a + imm], s0` (a sector of eight words per lane); `ldg.uni d, [u + imm]` |
| barriers | `mbar.init [u], count`; `mbar.arrive [u]`; `mbar.arrive.expect_tx [u], bytes`; `mbar.try_wait p, [u], parity`; `mbar.test_wait p, [u], parity` |
| TMA | `tma.load.Nd [buffer], [DESC, coord0], [barrier]`; `tma.store.Nd [DESC, coord0], [buffer], [barrier]` |
| tensor core | `mma.async.SHAPE [acc], [a, spacing], [b, spacing]`; `mma.async.init.SHAPE`; `mma.commit [barrier]`; `acc.ld`, `acc.st`, `acc.ld.v8`, `acc.st.v8` |
| other | `ld.param.u32 d, [NAME]`, `ld.param.f32`; `wait` |

## Barriers and the engines

Warps signal each other, and the TMA and the tensor core signal warps, through barriers. A
barrier's phase completes when its expected arrivals have happened and its outstanding
bytes are zero; it then flips to the next phase.

```
        mbar.arrive.expect_tx [%full], A_TILE[0].bytes   ; one arrival, and a tile still to come
        tma.load.1d [%buf], [ADESC, %tile], [%full]      ; the TMA takes the bytes off when done
        ...
WAIT:   mbar.try_wait p3, [%full], %parity               ; sleeps; can give up with p3 clear
@!p3    bra.uni WAIT
        uxori %parity, %parity, 1                        ; the parity is the kernel's to keep
```

A TMA descriptor (`.param desc D`, built in the launch script with `tma_desc(...)`,
`src/tma_descriptor.hpp`) says what a tensor is and how big a tile is; the instruction
says which tile, where in shared memory and which barrier to tell. `mma.commit [%m]` is
one arrival made by the tensor core when the matmuls queued before it are done. `wait`
holds a warp until its own loads and stores have finished; it does not wait for a TMA copy
or a matmul's results. `kernels/pipeline_tma.qtx` and `kernels/fa32.qtx` are the patterns
whole.

## Good to know before the first kernel

- **A scalar `ldg` or `stg` moves a whole sector for each lane's word.** Nothing merges
  requests. `ldg.v8` / `stg.v8` move the sector's eight words, so streaming code gives each
  lane eight neighbouring words. The run summary's byte counters show the difference.
- **The start of a kernel decides when its first load goes out.** A kernel's `.param`
  values arrive by a copy that takes a while after cycle 0; `ld.param` is a shared-memory
  load (shared memory serves few loads at once); every warp runs the same opening lines,
  one instruction a cycle among them; and qtxc issues instructions in the order written.
  Put what the first `ldg` needs first and the rest of the setup after it.
- **A branch waits for the warp's loads and stores.** A loop cannot keep a load in flight
  across its back edge; other warps fill the bus meanwhile.
- **A warp's own shared-memory accesses happen in the order written.** An `lds` after an
  `sts` of the same warp sees the store; no `fence.sh` is needed between them.
- **There is no shuffle between lanes, on purpose.** Lanes combine through shared memory:
  each lane stores its value in its own word, loads another lane's (the address differs in
  one bit: `xori`), combines, and repeats for each bit of the lane number.

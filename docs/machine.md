# The machine: an overview

*Written by models from the code, and checked against it by other models. The pages a person wrote are `README.md` and `design.md`.*

Every size and delay is a number in the device profile, `profiles/nanosm-v0.json`. Each
unit's source file in `src/machine/` opens with a comment that describes it; the list of
files is at the top of `src/machine/machine.cpp`.

```
 instruction memory
        |   fetch: one packet a cycle, round robin over the warps;
        |   each word is decoded once, as its packet lands
        v
 instruction buffers, one per warp (decoded instructions)
        |
        v
      ISSUE ------------------------------> branch
        |   at most one instruction a          resolves the next cycle; the
        |   cycle, from one warp               divergence stack is here
        v
 register read: the vector file, the uniform file, the predicates
        |
        +--> arithmetic: ALU, FPU, special-function unit --> register write
        |
        +--> a unit's queue (the vector registers are read there, later):
        |        shared memory and barriers | load / store unit
        |
        +--> dispatch to one unit: TMA | tensor core
```

```
                     one bus, one direction at a time
 global memory ==== memory controller ==+== load / store unit ==== register file
                                        |
                                        +== TMA ================== shared memory

 shared memory   port A: the warps' loads and stores, barriers and atomics, TMA store reads
                 port B: TMA load writes, the tensor core's operand reads

 tensor core     operands from shared memory -> accumulators in tensor memory,
                 which the warps read and write a row per lane
```

A warp waits in four ways: a stall count the compiler put after an instruction, a depbar
(a per-warp counter raised by a load or a store until it completes, and by a matmul until
it has read its operands; a per-lane load or store may also raise a second one, its read
depbar, until it has read its registers), a
barrier in shared memory (how warps wait for each other and for the engines), or a unit
with no room.

## Issue's loop

Every other long path in the machine runs one way and has a delay in the profile, so a
pipeline stage there moves a number. Issue is a loop: a warp that issues in one cycle must
be able to issue again in the next, so this has to close in one cycle:

```
 stored bits --> ready, per warp --> pick, round robin over 16 warps --> winner --> next stored bits
```

Everything issue asks is a register: the warp's own state (stall count, depbars, a probe
out, asleep, a branch pending), a credit count for the queue or engine it goes to (one
count per queue and engine, up at issue, down when the unit frees the entry), the write
calendar's entry for the instruction's kind and bank, and one inhibit bit per
register-file bank that arrives a cycle after the denial it answers. Issue looks at no other unit's decision of the same
cycle. A profile in which two of the ALU, FPU and special-function unit's write delays
differ by exactly one cycle is refused, so that a calendar booking is never looked at in
the cycle after it is made.

How deep the loop is has only been mapped, never placed or routed (7-series, LUT6 levels):
about 12 with the inhibits in the same cycle, 10 with them a cycle late, 8 with the three
stages written by hand. Each next-state bit depends on about 550 register bits, so no
mapping can be shallower than 4. The simulator models cycles and cannot see depth; what a
hardware build would add to get to 6 or 7 levels (a registered winner, one stored
"own ok" bit and one stored calendar bit per warp, the calendar written a cycle late)
changes no cycle count.

The simulator is written so that it could be hardware: state that takes the clock edge is
a register, a group of registers, a pipeline, a FIFO, a memory or a black box
(`src/machine/reg.hpp` and its neighbours), and each has one writer a cycle.
`python -B -m nanosm --state-report kernels/vadd.launch.py` lists all of it. A sanitizer
(on in the default build) stops a run with `PANIC` when a kernel breaks a rule the machine
does not check, such as reading a register before its write has landed.

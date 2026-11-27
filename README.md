<!-- Hand-written section. Agents: please do not edit. -->

# nanosm

A machine that attempts to capture the essence of a Blackwell-era GPU in a small core designed
from scratch.

It consists of a single SMSP (around 1/600th of an actual GPU), but it contains all the bells
and whistles expected from a CUTLASS device such as async MMA, TMA, atomics and barriers in
shared memory.

It can run kernels written in "QTX", a PTX clone, which is compiled down to .asm inserting stall
counts and depbars and then into an executable. You can find a tiny Flash Attention 4 kernel
in `kernels/fa32.qtx`.

The repo includes the QTX compiler (qtxc), a sanitizer, a profiler and the reference implementation
of the machine written in C++.

## Identity

"Make a GPU simulator modeling B200" is a trivial prompt today, so it's useful to start with what
makes this machine itself.

The design target is to be a simple and transparent version of the modern Nvidia device, in the spirit
of xv6 or nand2tetris. In the process I tried to stick close to the CUTLASS machine, but did not
completely give up on the CUDA machine.

This means modern ML kernels like FA4 get the best effort, ideally running close to theoretical limits.
One goal is to make at least a few tricks transfer between nanosm and Blackwell/Rubin and make this
useful as a kernel toy. CUDA-like irregular code runs without crazy hacks but there is no effort to
make it fast or close to the real thing. Running graphics is a non-goal.

Familiar GPU concepts are applicable:

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

- No float64 math.
- No L1/L2 cache.
- No memory access coalescer (corner-turn buffer for ldg.v8/stg.v8 instead)
- Async/atomic ops are launched per warp (per-thread or elected in Nvidia)
- Communication between divergent threads not guaranteed (pre-Volta stack)

Other differences:

- No operand collection step. Instead, the RF has 3 1R 1W banks (subject to change)
- Special function unit uses stall counts (uses depbars in Nvidia)
- No constant memory or cmem[ur] path.
- Kernel launch is a homebrew process (due to the lack of cmem[ur])
- No `shfl` (keeps the vector CPU spirits away)

## Realism

Right now this only exists as a C++ simulator. It is still tethered to reality by careful
tracking of register state and writers, together with my spidey senses of what constitutes
a long logic chain. When I couldn't tell logic or pipeline latency in a certain path I tried
to make it adjustable in the device profile.

I would give it a ~80% chance that this can be prompted into existence as synthesizable RTL
without major changes in the design. The machine shouldn't have any causality issues
and I paid special attention to reducing logic levels in the issue loop which is the only
place that would cause problems trying to add registers. The rest of the machine can be implemented
with flexible state and latency.

The code could be much shorter if written in behavioral RTL. I decided against that because
a) models are more comfortable in C++, and b) I wanted to eventually compile to WASM and was
afraid going through Verilator could mess with that. As a result, a lot of the lines in
the simulator are spent reinventing an event model and trying to ensure causality.

The microarchitecture is FPGA minded, making use of multi-port memory blocks and opting to use
state/memory instead of logic where possible. An ASIC implementation would need to take different
tradeoffs but honestly I would be surprised if anyone attempted that.

### Sanitizer

The simulator comes with a sanitizer to catch UB in kernels. (Right now, the only spec of what constitutes
UB is the sanitizer itself.) There is a `nosan` flow to make sure the simulation isn't secretly
depending on the sanitizer state.

## QTX and kernel toy

It is common knowledge that PTX is objectively the best language to write kernels in. However,
PTX is proprietary to NVIDIA, so we have to invent a completely unrelated language called QTX.

(Lowering from LLVM IR or anything that looks like CPU assembly is strictly prohibited)

QTX is a high-level GPU assembly language written from one thread's point of view.
It generally does what you expect it to do. See docs/qtx.md for a model-written overview of it.

Here is an excerpt from `kernels/pipeline_tma.qtx`:

```ptx
          mbar.arrive.expect_tx [%full0], 128            ; one arrival, 128 bytes still to come
          tma.load.1d [%buf_a0], [ADESC, %tile], [%full0] ; the tile mover fills a buffer on its own
          tma.load.1d [%buf_b0], [BDESC, %tile], [%full0]
          ...
  WAIT:   mbar.try_wait p3, [%full0], %parity0           ; the other warp sleeps until both landed
  @!p3    bra.uni WAIT
          lds   %x, [%lane + A_TILE[0]]
```

<!-- what else to say here. maybe after WASM demo and we have a bunch of fun kernel problems? -->

## Internals

See design.md for further discussion of the internals and the design decisions involved.
I try to keep it hand-written and up to date.

<!-- End of hand-written section. -->

## Build and run

Needs CMake 3.20 or later, Ninja, a C++20 compiler (built with GCC) and Python 3.

```sh
cmake -S . -B build -G Ninja
cmake --build build
python -B -m nanosm kernels/vadd.launch.py
```

Run `powershell -File scripts/build.ps1` for a Windows build.

### Compile, run and trace a kernel

A kernel `K.qtx` has a launch script `K.launch.py` beside it. The script gives the inputs,
compiles and runs the kernel, checks the answer, and prints the cycle count and where the
cycles went:

```sh
python -B -m nanosm kernels/vadd.launch.py
```

For a timeline of every warp, queue and bus, add `--trace` and open the file in
https://ui.perfetto.dev:

```sh
python -B -m nanosm kernels/vadd.launch.py --trace build/vadd.perfetto.json
```

To see what the compiler decided (register numbers, stall counts, depbars) without running:

```sh
build/qtxc  profiles/nanosm-v0.json kernels/vadd.qtx build/vadd.asm   # QTX -> scheduled assembly
build/qtxas profiles/nanosm-v0.json build/vadd.asm   build/vadd.hex   # assembly -> instruction words
```

`python -B -m nanosm --state-report kernels/vadd.launch.py` lists the machine's state and
its black boxes before the run.

## Where to read

- `design.md`: the machine's design.
- `docs/qtx.md`: writing a kernel. (Written by models.)
- `docs/machine.md`: the machine's parts in two pictures. (Written by models.)
- `kernels/`: the kernels, each with a launch script that checks its answer.
- `AGENTS.md`: for agents working in the repository.

<!-- human written! -->

## AI use

Thanks to all the language models for putting up with my "capricious architect" roleplay in this project.

In general, the implementation was done one step at a time by a single agent. Its outputs were
reviewed carefully by me and a bunch of design agents from different model lines, which were also
proposing design ideas and directions. Some of the ideas definitely made it into the architecture.

I hand-wrote the human facing parts of the project to avoid presenting a generic LLM texture.

The model names are mentioned in the "Assisted-by:" line of each commit.

<!-- human written end -->

## Credits and license

- The tensor core's arithmetic is adapted from SF Tensor's tcgen05 model
  (https://github.com/sf-tensor/tcgen05, MIT, `third_party/tcgen05-LICENSE`).
- JSON parsing is nlohmann/json 3.12.0
  (https://github.com/nlohmann/json, MIT, vendored in `third_party/nlohmann`).
- Tests use GoogleTest 1.17.0
  (https://github.com/google/googletest, BSD 3-Clause, vendored in `third_party/googletest`).

nanosm itself is under the MIT license (`LICENSE`).

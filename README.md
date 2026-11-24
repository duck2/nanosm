# nanosm

A cycle-accurate simulator of one GPU streaming multiprocessor, and a small language, qtx,
for writing its kernels. It is a tribute to NVIDIA's Blackwell: warps of sixteen lanes, a
banked register file, shared memory, barriers, a TMA that moves tiles on its own, a tensor
core with its own accumulator memory. It models none of them exactly.

The idea: a machine small enough to read whole, where a kernel's cycle count is the result
of its design and nothing in the hardware rescues a careless one. Kernels are written by
models and read by people.

```
        ldg.v8 %a0, [%a_ptr]        ; a 32-byte sector per lane, into %a0 .. %a7
        ldg.v8 %b0, [%b_ptr]
        fadd   %a0, %a0, %b0
        ...
        stg.v8 [%c_ptr], %a0
```

## Build and run

Needs CMake 3.20 or later, Ninja, a C++20 compiler (built with GCC) and Python 3. No
packages.

    cmake -S . -B build -G Ninja
    cmake --build build
    python -B -m nanosm kernels/vadd.launch.py

On Windows, `powershell -File scripts/build.ps1` finds the tools, builds and runs every
test (`-Fast` for the fast ones).

A launch prints its check and where the cycles went:

    vadd ok: 8192 outputs bit-exact, 98304 bytes in 26275 cycles = 3.741 B/cycle of a declared 4 B/cycle bus

Add `--trace out.json` to a launch for a timeline to open in https://ui.perfetto.dev.

## Read next

- `docs/qtx.md`: writing a kernel.
- `docs/machine.md`: what things cost, how to read a run, where the source is, how the
  simulator keeps its state.
- `kernels/`: vector add, two pipelines with barriers and the TMA, per-lane control flow,
  and attention (`fa32`).
- `profiles/nanosm-v0.json`: the device. Every size and delay is a number there.

## State of things

A first version. The launch, the TMA's instruction form and the tensor core's operand
interface are expected to change. If a design looks wrong to you, open an issue and say so.

The tensor core's arithmetic is adapted from SF Tensor's tcgen05 model
(https://github.com/sf-tensor/tcgen05, `third_party/tcgen05-LICENSE`). JSON parsing is
nlohmann/json; tests are GoogleTest. Both are vendored.

MIT license (`LICENSE`).

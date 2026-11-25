# nanosm

One GPU streaming multiprocessor as a small machine, and qtx, a language for writing its
kernels.

## Build and run

Needs CMake 3.20 or later, Ninja, a C++20 compiler (built with GCC) and Python 3. No
packages.

    cmake -S . -B build -G Ninja
    cmake --build build
    python -B -m nanosm kernels/vadd.launch.py

On Windows, `powershell -File scripts/build.ps1` finds the tools, builds and runs every
test (`-Fast` for the fast ones).

## Read next

- `design.md`: the machine's design.
- `docs/qtx.md`: writing a kernel.
- `docs/machine.md`: the machine's parts in two pictures.
- `kernels/`: the kernels, each with a launch script that checks its answer.
- `AGENTS.md`: for agents working in the repository.

The tensor core's arithmetic is adapted from SF Tensor's tcgen05 model
(https://github.com/sf-tensor/tcgen05, `third_party/tcgen05-LICENSE`). JSON parsing is
nlohmann/json; tests are GoogleTest. Both are vendored.

MIT license (`LICENSE`).

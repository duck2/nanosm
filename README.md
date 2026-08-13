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
```

The kernel path is deliberately visible and editable:

```sh
build/qtxc kernels/vadd.qtx build/vadd.asm
build/qtxas build/vadd.asm build/vadd.hex
build/nanosm_demo build/vadd.hex
```

CMake performs those steps automatically for `build/generated/vadd.{asm,hex}`. On this Windows
checkout, `scripts/build.ps1` locates the installed CMake, Ninja and MinGW tools and runs the same
configure/build/test flow.

`qtx` has semantic instructions and no control bits. `asm` is scheduled qtx with explicit control
fields and is the intended human-editable layer. `hex` is one annotated 64-bit instruction word per
line and is what the simulator fetches.

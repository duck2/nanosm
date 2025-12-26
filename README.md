# gpiu

Tiny and performance-serious GPU. It is heavily tailored to fit Xilinx 7-series FPGAs, but more
targets may be added in the future.

The architecture aims to look like a real GPU for compute purposes. Notable features are:

- FP32 math
- N-wide warps
- Warp scheduler
- Memory access coalescer
- 2D addressing support for global resources
- Warp divergence support

The most unique feature is tile-based rendering implemented as a compute kernel. gpiu is designed
to keep "soft rendering" reasonably fast. This is simply because as a solo developer I don't have
the time or interest to maintain hard graphics blocks, and the FPGA substrate constrains us in
terms of area anyway.

Future tasks include integration with VexRiscv, exposing the device to Linux,
writing a Gallium driver, shader compiler, Jax/PyTorch compiler... it's a tiny but ambitious
gpiu.

# nanosm

A tiny GPU-like soft accelerator which tries to be just complex enough to run FlashAttention while
still posing similar challenges as a production device.

The GPU-like features are:
* Warp scheduling
* Branch divergence stack
* Software-managed cache (shared memory)
* Tensor core (parallel fused dot-add)
* Async copy engine

Some features are intentionally left out to keep the hope that this can be implemented on a
hobbyist-grade FPGA. Such features include all graphics-specific stuff, float SFUs other than
strictly necessary for attention and many other bells and whistles such as proper bank conflict
management for the shared memory.

It should also be noted that this is designed as a single-partition, single-SM device. This is
to avoid the task scheduling / kernel launch shenanigans which are the unfun parts of a GPU.

## Programming model

It's the well known SIMT model with a PTX-like instruction set, i.e. no explicit vector
instructions. There aren't even warp-level shuffles (both to keep SIMT purity and save area).
Since there is only a single SM, the launcher only needs to set warp count.

Execution is GPU-style, in order with warp scheduling to hide latency. Stall cycles and barrier flags are inserted by the assembler.

## Status

Work in progress as I keep changing the goals and the design. The first version of this was
similar to [iDEA (2016)](https://warwick.ac.uk/fac/sci/eng/people/suhaib_fahmy/publications/cheah-phdthesis2016.pdf), just a minimal
vector processor which tries to use up leftover DSPs in a larger system while being somehow usable.

That did not go very well, so I switched
to a Larrabee-ish little GPU which can do ping-pong style soft rasterization, which made me realize that the thing is getting
dangerously close to being able to run a real compute kernel and it would probably help to drop graphics as graphics blocks in the design
are really not usable for anything else and it feels like a waste of area and development effort.

Currently, this is just starting v3 with the design getting adapted to the new goal of running attention. Rough progress:

* ✅ Shared memory
* 🟡 Functional emulator
* 🟡 ISA design (needs to be adapted for v3)
* 🟡 Float pipeline
* 🟡 Warp scheduler
* 🟡 Tensor core
* 🟡 Branch divergence stack
* ⬜ Async copy engine
* ⬜ Compiler

Legend:

* ⬜ : planned but not started
* 🟡 : I have a solidified design to implement / WIP in implementation
* ✅ : Design & impl done, not subject to change

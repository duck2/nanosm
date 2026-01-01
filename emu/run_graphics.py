"""Graphics test runner for GPU emulator with tile-based rendering."""

import sys
import os
import struct
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from emu.core import Emulator
from emu.asm import load_file

FB_WIDTH = 640
FB_HEIGHT = 480
TILE_WIDTH = 32
TILE_HEIGHT = 32
BPP = 4

FB_BASE = 0x100000
RT_DESC_ADDR = 0x1000
KERNEL_ARGS = 0x2000


def setup_render_target(emu: Emulator):
    """Write render target descriptor to memory."""
    mem = emu.state.global_mem
    stride = FB_WIDTH * BPP
    struct.pack_into('<I', mem, RT_DESC_ADDR, FB_BASE)
    struct.pack_into('<I', mem, RT_DESC_ADDR + 4, stride)
    struct.pack_into('<I', mem, RT_DESC_ADDR + 8, TILE_WIDTH)
    struct.pack_into('<I', mem, RT_DESC_ADDR + 12, TILE_HEIGHT)


def setup_kernel_args(emu: Emulator, **kwargs):
    """Write kernel arguments to memory and registers."""
    num_tiles_x = FB_WIDTH // TILE_WIDTH
    num_tiles_y = FB_HEIGHT // TILE_HEIGHT
    for lane in range(emu.state.num_lanes):
        emu.state.rf[20, lane] = RT_DESC_ADDR
        emu.state.rf[21, lane] = num_tiles_x
        emu.state.rf[22, lane] = num_tiles_y
    offset = 0
    for key, val in kwargs.items():
        struct.pack_into('<I', emu.state.global_mem, KERNEL_ARGS + offset, val)
        offset += 4


def extract_framebuffer(emu: Emulator) -> np.ndarray:
    """Extract framebuffer as numpy RGBA array."""
    fb = np.zeros((FB_HEIGHT, FB_WIDTH, 4), dtype=np.uint8)
    for y in range(FB_HEIGHT):
        for x in range(FB_WIDTH):
            addr = FB_BASE + y * FB_WIDTH * BPP + x * BPP
            rgba = struct.unpack('<I', emu.state.global_mem[addr:addr+4])[0]
            fb[y, x, 0] = (rgba >> 0) & 0xFF
            fb[y, x, 1] = (rgba >> 8) & 0xFF
            fb[y, x, 2] = (rgba >> 16) & 0xFF
            fb[y, x, 3] = (rgba >> 24) & 0xFF
    return fb


def run_graphics(path: str, show: bool = True, save: str = None, **kwargs):
    """Run a graphics kernel and display/save result."""
    instrs, labels = load_file(path)
    emu = Emulator(num_lanes=8)

    setup_render_target(emu)
    setup_kernel_args(emu, **kwargs)
    emu.load_program(instrs, labels)

    print(f"Loaded {len(instrs)} instructions from {path}")
    steps = emu.run(max_steps=10_000_000)
    print(f"Executed {steps} steps")

    fb = extract_framebuffer(emu)
    print(f"Framebuffer: {FB_WIDTH}x{FB_HEIGHT}")

    if save:
        Image.fromarray(fb[:, :, :3]).save(save)
        print(f"Saved to {save}")

    if show:
        plt.figure(figsize=(10, 7.5))
        plt.imshow(fb[:, :, :3])
        plt.title(Path(path).stem)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return emu, fb


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <kernel.asm> [--save output.png] [--no-show]")
        sys.exit(1)

    path = sys.argv[1]
    save = None
    show = True

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--save' and i + 1 < len(sys.argv):
            save = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--no-show':
            show = False
            i += 1
        else:
            i += 1

    run_graphics(path, show=show, save=save)


if __name__ == '__main__':
    main()

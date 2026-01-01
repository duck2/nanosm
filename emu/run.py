"""Run a single ASM file on the GPU emulator."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emu.core import Emulator
from emu.asm import load_file


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <file.asm> [--regs N] [--float]")
        sys.exit(1)

    path = sys.argv[1]
    show_regs = 16
    show_float = False

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--regs' and i + 1 < len(sys.argv):
            show_regs = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--float':
            show_float = True
            i += 1
        else:
            i += 1

    instrs, labels = load_file(path)
    emu = Emulator(num_lanes=8)
    emu.load_program(instrs, labels)

    print(f"Loaded {len(instrs)} instructions from {path}")
    steps = emu.run()
    print(f"Executed {steps} steps")

    print("\nFinal register state:")
    if show_float:
        emu.dump_regs_f32(range(show_regs))
    else:
        emu.dump_regs(range(show_regs))


if __name__ == '__main__':
    main()

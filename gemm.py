# gemm.py — one-output-tile GEMM in a cutedsl-like dsl for the device
# C[64,64] f32 = A[64,512] f16 @ B[512,64] f16 — K streamed as 8 tiles of 64, C resident in accumulator.
# Everything the dsl hides is shown compiled in gemm.asm.

from device import kernel, smem, accum, barrier, role, tma, mma

@kernel(warps=6)                    # launch ABI: warp count is a launch parameter; warps 6..31 never exist
def gemm(A, B, C, k_tiles=8):
    # static allocations — assembler resolves to addresses in the blessed (swizzled) layout, never visible here
    a_s = smem.tile("f16", (64, 64), stages=2)
    b_s = smem.tile("f16", (64, 64), stages=2)
    acc = accum.tile("f32", (64, 64))            # C lives here for the whole kernel

    full  = [barrier(expect=2) for _ in range(2)]   # a stage is full after 2 arrivals: A-tma + B-tma
    empty = [barrier(expect=1) for _ in range(2)]   # free after 1 arrival: the mma that consumed it
    cdone = barrier(expect=1)

    with role(warps=[0]):                        # ---- producer ----
        for s in range(2):                       # prologue: fill both stages WITHOUT waiting (cold start
            tma.load(a_s[s], A.tile(0, s), arrive=full[s])   # is solved by peeling, cutlass-style)
            tma.load(b_s[s], B.tile(s, 0), arrive=full[s])
        for k in range(2, k_tiles):
            s = k % 2                            # compiles to a 2-unrolled loop: barrier ids are immediates
            empty[s].wait()
            tma.load(a_s[s], A.tile(0, k), arrive=full[s])
            tma.load(b_s[s], B.tile(k, 0), arrive=full[s])

    with role(warps=[1]):                        # ---- mma issuer ----
        for k in range(k_tiles):
            s = k % 2
            full[s].wait()
            last = k == k_tiles - 1
            for ks in range(8):                  # one mma = one k8 step; software loops K (8 steps = k64)
                mma(acc, a_s[s].kstep(ks), b_s[s].kstep(ks),   # issue parks at the port while engine busy
                    arrive=(empty[s] | (cdone if last else 0)) if ks == 7 else None)

    with role(warps=[2, 3, 4, 5]):               # ---- epilogue: each warp owns 16 rows (lane m = row m) ----
        cdone.wait()
        for chunk in acc.drain_chunks(cols=8):   # RF is 32 regs → drain in 8-col chunks (see asm NOTE)
            C.store(chunk)                       # vector stg; store-drain counter orders it, no fence exists

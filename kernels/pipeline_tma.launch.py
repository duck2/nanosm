from pathlib import Path

from nanosm import Launch


ROOT = Path(__file__).resolve().parents[1]
A = 0
B = 4096
C = 8192
FULL0 = 1024
FULL1 = 1032
EMPTY0 = 1040
EMPTY1 = 1048
ADESC0 = 1152
BDESC0 = 1216
ADESC1 = 1280
BDESC1 = 1344
ELEMENTS = 8 * 16

launch = Launch(
    kernel=Path(__file__).with_suffix("").with_suffix(".qtx"),
    profile=ROOT / "profiles" / "nanosm-v0.json",
    warps=2,
)
launch.gmem.f32(A, (float(i) for i in range(ELEMENTS)))
launch.gmem.f32(B, (float(2 * i) for i in range(ELEMENTS)))
for address in [FULL0, FULL1, EMPTY0, EMPTY1]:
    launch.shmem.mbar(address, expected=1)
for address, base, tile in [
    (ADESC0, A, 0),
    (BDESC0, B, 64),
    (ADESC1, A, 128),
    (BDESC1, B, 192),
]:
    launch.shmem.tma_desc(
        address,
        direction="load",
        gmem_base=base,
        dimensions=(ELEMENTS,),
        box=(16,),
        shmem_base=tile,
        completion_mbar=FULL0 if tile < 128 else FULL1,
    )

with launch.run(trace=ROOT / "build" / "pipeline-tma.perfetto.json") as result:
    expected = [float(3 * i) for i in range(ELEMENTS)]
    actual = result.gmem_f32(C, ELEMENTS)
    if actual != expected:
        raise RuntimeError("tma pipeline output mismatch")
    print("tma pipeline ok")
    print(result.summary, end="")

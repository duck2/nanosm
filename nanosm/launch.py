from __future__ import annotations

import json
from pathlib import Path
import struct
from typing import Iterable

from ._native import Session, TRACE_DEBUG, TRACE_TIMELINE, encode_mbar, encode_tma_descriptor


class SparseMemory:
    def __init__(self) -> None:
        self._segments: list[dict[str, object]] = []
        # Addresses written by mbar(): in shmem, the launch DECLARES them as
        # mbars (live from launch for the sanitizer's C10 opacity, rfc12
        # section 11), like a kernel's mbar.init.
        self.mbars: list[int] = []

    def write(self, address: int, data: bytes | bytearray | memoryview) -> None:
        payload = bytes(data)
        if address < 0:
            raise ValueError("memory address must be nonnegative")
        if not payload:
            raise ValueError("memory segment must not be empty")
        self._segments.append({"address": address, "bytes": payload.hex()})

    def u32(self, address: int, values: Iterable[int]) -> None:
        items = list(values)
        self.write(address, struct.pack(f"<{len(items)}I", *items))

    def f32(self, address: int, values: Iterable[float]) -> None:
        items = list(values)
        self.write(address, struct.pack(f"<{len(items)}f", *items))

    def mbar(
        self,
        address: int,
        *,
        expected: int,
        phase: int = 0,
        remaining: int | None = None,
        tx_bytes: int = 0,
    ) -> None:
        if remaining is None:
            remaining = expected
        for name, value in {
            "phase": phase,
            "expected": expected,
            "remaining": remaining,
        }.items():
            if not 0 <= value <= 0xFFFFFFFF:
                raise ValueError(f"mbar {name} is outside the C ABI's uint32_t range")
        # The signed transaction count (rfc12 section 8): negative when
        # complete_tx ran ahead of expect_tx.
        if not -(2**20 - 1) <= tx_bytes <= 2**20 - 1:
            raise ValueError("mbar tx_bytes is outside -(2^20 - 1)..2^20 - 1")
        word = encode_mbar(phase, expected, remaining, tx_bytes)
        self.write(address, struct.pack("<Q", word))
        self.mbars.append(address)

    def tma_desc(
        self,
        address: int,
        *,
        base: int,
        dimensions: tuple[int, ...],
        tile: tuple[int, ...],
        element_bytes: int = 4,
        strides: tuple[int, ...] | None = None,
        row_dimension: int = 1,
        layout: str = "dense",
        transposed: bool = False,
    ) -> None:
        """A TMA descriptor (src/tma.hpp): one tensor, its tile size and layout.

        dimensions: d0 .. d3 in elements, dimension 0 the contiguous one (one
        to four values; the missing ones are 1). strides: s1 .. s(n-1) in
        elements, one index of that dimension to the next (default: dense).
        tile: (length,) or (length, rows): `rows` runs of `length` elements
        along dimension 0, the rows stepping along `row_dimension` (1..3).
        layout: the tile's shared-memory layout (src/layout.hpp), "dense"
        (row after row) or "xor64" (slices of 16-byte rows), the names of the
        profile's mma.layout; transposed (xor64 only): the positions along
        dimension 0 become the stored rows. The direction, buffer, barrier
        and coordinates are the instruction's (tma.load.Nd / tma.store.Nd).
        """
        if not 1 <= len(dimensions) <= 4:
            raise ValueError("tma dimensions must have one to four values")
        dims = tuple(dimensions) + (1,) * (4 - len(dimensions))
        if strides is None:
            strides = tuple(dims[0] * (dims[1] if k > 1 else 1) * (dims[2] if k > 2 else 1) for k in (1, 2, 3))
        elif len(strides) != len(dimensions) - 1:
            raise ValueError("tma strides must have one value per dimension after the first")
        else:
            strides = tuple(strides) + (1,) * (3 - len(strides))
        if len(tile) not in (1, 2):
            raise ValueError("tma tile must be (length,) or (length, rows)")
        length, rows = (tuple(tile) + (1,))[:2]
        layout_code = {"dense": 0, "xor64": 1}.get(layout) if isinstance(layout, str) else None
        if layout_code is None:
            raise ValueError("tma layout must be 'dense' or 'xor64'")
        for name, values, maximum in [
            ("base", (base,), 0xFFFFFFFFFFFFFFFF),
            ("dimensions", dims, 0xFFFFFFFF),
            ("strides", strides, 0xFFFFFFFF),
            ("element_bytes", (element_bytes,), 0xFFFFFFFF),
            ("tile", (length, rows), 0xFFFFFFFF),
            ("row_dimension", (row_dimension,), 0xFFFFFFFF),
        ]:
            if any(not isinstance(value, int) or value < 0 or value > maximum for value in values):
                raise ValueError(f"tma {name} is outside the C ABI range")
        if not isinstance(transposed, bool):
            raise ValueError("tma transposed must be a bool")
        if address % 64 != 0:
            raise ValueError("tma descriptor address must be 64-byte aligned")
        self.write(
            address,
            encode_tma_descriptor(
                base, element_bytes, dims, strides, length, rows, row_dimension, layout_code, int(transposed)
            ),
        )

    def _json(self) -> list[dict[str, object]]:
        return list(self._segments)


class Result:
    def __init__(self, session: Session):
        self._session = session

    @property
    def cycle(self) -> int:
        return self._session.cycle

    @property
    def summary(self) -> str:
        return self._session.summary

    def gmem(self, address: int, size: int) -> bytes:
        return self._session.read_gmem(address, size)

    def shmem(self, address: int, size: int) -> bytes:
        return self._session.read_shmem(address, size)

    def gmem_f32(self, address: int, count: int) -> list[float]:
        return list(struct.unpack(f"<{count}f", self.gmem(address, count * 4)))

    def write_trace(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self._session.trace_json, encoding="utf-8")

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "Result":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


class Launch:
    def __init__(
        self,
        *,
        kernel: str | Path,
        profile: str | Path,
        warps: int,
    ) -> None:
        if warps <= 0:
            raise ValueError("active warp count must be positive")
        self.kernel = Path(kernel)
        self.profile = Path(profile)
        self.warps = warps
        self.shmem = SparseMemory()
        self.gmem = SparseMemory()

    def _state_json(self) -> bytes:
        document = {
            "format": "nanosm-launch-state-v0",
            "active_warps": self.warps,
            "shmem": self.shmem._json(),
            "gmem": self.gmem._json(),
            "mbars": self.shmem.mbars,
        }
        return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")

    def _create_session(
        self,
        *,
        timeline: bool,
        trace_debug: bool,
        max_trace_events: int,
    ) -> Session:
        flags = 0
        if timeline:
            flags |= TRACE_TIMELINE
        if trace_debug:
            flags |= TRACE_DEBUG
        return Session.from_qtx(
            self.profile.read_bytes(),
            self.kernel.read_bytes(),
            self._state_json(),
            flags,
            max_trace_events,
        )

    def run(
        self,
        *,
        trace: str | Path | None = None,
        trace_debug: bool = False,
        max_trace_events: int = 200_000,
        cycle_limit: int = 100_000,
    ) -> Result:
        session = self._create_session(
            timeline=trace is not None or trace_debug,
            trace_debug=trace_debug,
            max_trace_events=max_trace_events,
        )
        try:
            session.run(cycle_limit)
            result = Result(session)
            if trace is not None:
                result.write_trace(trace)
            return result
        except BaseException:
            session.close()
            raise

    def write_image(self, path: str | Path) -> None:
        with self._create_session(
            timeline=False,
            trace_debug=False,
            max_trace_events=200_000,
        ) as session:
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(session.launch_image, encoding="utf-8")

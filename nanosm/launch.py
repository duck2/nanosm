from __future__ import annotations

import json
from pathlib import Path
import struct
from typing import Iterable

from ._native import Session, TRACE_DEBUG, TRACE_TIMELINE, encode_mbar, encode_tma_descriptor


class SparseMemory:
    def __init__(self) -> None:
        self._segments: list[dict[str, object]] = []

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

    def tma_desc(
        self,
        address: int,
        *,
        direction: str,
        gmem_base: int,
        dimensions: tuple[int, ...],
        box: tuple[int, ...],
        shmem_base: int,
        completion_mbar: int,
        element_bytes: int = 4,
        strides: tuple[int, ...] = (),
        swizzle: int | str = "linear",
        transpose: bool = False,
    ) -> None:
        rank = len(dimensions)
        if rank not in (1, 2, 3) or len(box) != rank or len(strides) != rank - 1:
            raise ValueError("tma dimensions/box must have rank 1..3 and rank-1 strides")
        direction_code = {"load": 0, "store": 1}.get(direction)
        if direction_code is None:
            raise ValueError("tma direction must be 'load' or 'store'")
        dimensions3 = tuple(dimensions) + (1,) * (3 - rank)
        box3 = tuple(box) + (1,) * (3 - rank)
        strides2 = tuple(strides) + (0,) * (2 - len(strides))
        for name, values in {
            "dimensions": dimensions3,
            "box": box3,
            "strides": strides2,
        }.items():
            if any(value < 0 or value > 0xFFFFFFFF for value in values):
                raise ValueError(f"tma {name} contains a value outside uint32_t")
        if isinstance(swizzle, str):
            swizzle_code = {"linear": 0, "xor64_a": 1, "xor64_b": 2}.get(swizzle)
            if swizzle_code is None:
                raise ValueError("tma swizzle must be 'linear', 'xor64_a', or 'xor64_b'")
        else:
            swizzle_code = swizzle
        for name, value, maximum in [
            ("gmem_base", gmem_base, 0xFFFFFFFFFFFFFFFF),
            ("element_bytes", element_bytes, 0xFFFFFFFF),
            ("swizzle", swizzle_code, 0xFFFFFFFF),
            ("shmem_base", shmem_base, 0xFFFFFFFF),
            ("completion_mbar", completion_mbar, 0xFFFFFFFF),
        ]:
            if not isinstance(value, int) or value < 0 or value > maximum:
                raise ValueError(f"tma {name} is outside the C ABI range")
        if not isinstance(transpose, bool):
            raise ValueError("tma transpose must be a bool")
        if address % 64 != 0:
            raise ValueError("tma descriptor address must be 64-byte aligned")
        self.write(
            address,
            encode_tma_descriptor(
                rank,
                direction_code,
                int(transpose),
                gmem_base,
                dimensions3,
                strides2,
                box3,
                element_bytes,
                swizzle_code,
                shmem_base,
                completion_mbar,
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

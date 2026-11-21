from __future__ import annotations

import json
import math
from pathlib import Path
import struct
from typing import Iterable, Mapping, Sequence

from ._native import Session, TRACE_DEBUG, TRACE_TIMELINE, encode_tma_descriptor, kernel_info


class SparseMemory:
    """Bytes the launch writes into a memory before the first cycle.

    Global memory is the host's: inputs go here. Shared memory is the
    kernel's; the launch needs none of it to start a kernel (arguments,
    barriers and constants are the kernel's own declarations). Writing shared
    memory here is a debugger's access, for a test that poisons a region; its
    address comes from Launch.symbol.
    """

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

    def _json(self) -> list[dict[str, object]]:
        return list(self._segments)


def tma_desc(
    *,
    base: int,
    dimensions: tuple[int, ...],
    tile: tuple[int, ...],
    element_bytes: int = 4,
    strides: tuple[int, ...] | None = None,
    row_dimension: int = 1,
    layout: str = "dense",
    transposed: bool = False,
) -> bytes:
    """A TMA descriptor (src/tma.hpp), the argument for a `.param desc`.

    One tensor, its tile size and layout, as its 64 bytes.
    dimensions: d0 .. d3 in elements, dimension 0 the contiguous one (one
    to four values; the missing ones are 1). strides: s1 .. s(n-1) in
    elements, one index of that dimension to the next (default: dense).
    tile: (length,) or (length, rows): `rows` runs of `length` elements
    along dimension 0, the rows stepping along `row_dimension` (1..3).
    layout: the tile's shared-memory layout (src/layout.hpp), "dense"
    (row after row) or "swizzle64" (whole rows, packed, shuffled inside
    each row by the row number its address gives; the stored row a power
    of two bytes, the buffer a multiple of it and of 64), the names of
    the profile's mma.layout;
    transposed (swizzle64 only): the positions along dimension 0 become
    the stored rows and the tile's row index runs along them. The
    direction, buffer, barrier and coordinates are the instruction's
    (tma.load.Nd / tma.store.Nd).
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
    layout_code = {"dense": 0, "swizzle64": 1}.get(layout) if isinstance(layout, str) else None
    if layout_code is None:
        raise ValueError("tma layout must be 'dense' or 'swizzle64'")
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
    return encode_tma_descriptor(
        base, element_bytes, dims, strides, length, rows, row_dimension, layout_code, int(transposed)
    )


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


def launch_copy_counters(expected: Mapping[str, Mapping[str, int]], copied_bytes: int,
                         sector_bytes: int = 32, bus_bytes: int = 8) -> dict[str, dict[str, int]]:
    """`expected` (summary line -> counter -> value: a kernel's own algorithmic
    counts) with the launch copy's part added: one TMA load of `copied_bytes`
    in whole sectors, each one global read and one shared-memory round. Keys
    a script does not pin are left alone."""
    sectors = -(-copied_bytes // sector_bytes)
    extra = {
        "tma": {"commands": 1 if copied_bytes else 0, "completed": 1 if copied_bytes else 0,
                "logical_bytes": copied_bytes, "useful_bytes": copied_bytes,
                "sector_bytes": sectors * sector_bytes, "port_requests": sectors, "port_rounds": sectors},
        "gmem": {"sectors": sectors, "beats": sectors * sector_bytes // bus_bytes, "bytes": sectors * sector_bytes},
        "shmem": {"requests": sectors, "rounds": sectors},
    }
    result = {line: dict(counters) for line, counters in expected.items()}
    for line, counters in extra.items():
        for key, value in counters.items():
            if key in result.get(line, {}):
                result[line][key] += value
    return result


class Launch:
    """One launch of a kernel: its arguments and the host's global memory.

    The kernel's qtx declares everything else (README "Launch"): its
    signature (`.param` lines), how many warps it is written for (`.warps`),
    its barriers (`.mbar`) and constants (`.data`).

    args: one value per parameter, by name ({"IN": 0, ...}) or in the
    signature's order ([0, ...]): an int for ptr and u32, a number for f32,
    the bytes of tma_desc(...) for desc. A missing, unknown or wrongly typed
    argument is refused by name when the launch runs.
    warps: needed only when the kernel does not say `.warps`; if both say,
    they must agree.
    """

    def __init__(
        self,
        *,
        kernel: str | Path,
        profile: str | Path,
        warps: int | None = None,
        args: Mapping[str, object] | Sequence[object] | None = None,
        block_address: int | None = None,
    ) -> None:
        if warps is not None and warps <= 0:
            raise ValueError("active warp count must be positive")
        if block_address is not None and block_address < 0:
            raise ValueError("the launch block's global address is negative")
        # Where in global memory the host puts the kernel's launch block (its
        # arguments, barrier states and data; README "Launch"). None: the top
        # of global memory.
        self.block_address = block_address
        self.kernel = Path(kernel)
        self.profile = Path(profile)
        self._info = json.loads(kernel_info(self.profile.read_bytes(), self.kernel.read_bytes()))
        self._warps = warps
        self.args = self._named(args)
        self.shmem = SparseMemory()
        self.gmem = SparseMemory()

    @property
    def signature(self) -> list[tuple[str, str]]:
        """The kernel's parameters in order: (type, name)."""
        return [(param["type"], param["name"]) for param in self._info["params"]]

    @property
    def warps(self) -> int:
        """The launch's warp count, or the kernel's `.warps` when it gives none."""
        return self._warps if self._warps is not None else int(self._info["warps"])

    @property
    def sections(self) -> list[dict[str, object]]:
        """The kernel's launch block in order: each section's name, kind
        (param, mbar, data), shared-memory address, bytes and launch flag (a
        depbar number; absent on a profile with no launch copy)."""
        return list(self._info["sections"])

    @property
    def copied_bytes(self) -> int:
        """The bytes the launch copy brings from global memory into shared
        memory: the launch block's size, or 0 when the kernel has no sections
        or the profile makes no copy (core.launch_flags 0). The copy is one
        more TMA command of whole sectors; a script that pins counters adds
        its part (launch_copy_counters)."""
        return int(self._info["block"]["bytes"]) if self._info["flags"] else 0

    def symbol(self, name: str) -> int:
        """The value of a name in the kernel's qtx: an .equ constant, or a
        section's (a parameter's, an mbar's, a .data's) address in shared
        memory. For a test or a debugger that looks inside the kernel's
        memory; a launch does not need it."""
        try:
            return int(self._info["symbols"][name])
        except KeyError:
            raise KeyError(f"{self.kernel.name} has no symbol '{name}'") from None

    def _named(self, args: Mapping[str, object] | Sequence[object] | None) -> dict[str, object]:
        if args is None:
            return {}
        if isinstance(args, Mapping):
            return dict(args)
        names = [name for _type, name in self.signature]
        values = list(args)
        if len(values) > len(names):
            raise ValueError(
                f"{len(values)} arguments for a kernel with {len(names)} parameters ({', '.join(names) or 'none'})")
        return dict(zip(names, values))

    @staticmethod
    def _argument_json(name: str, value: object) -> object:
        if isinstance(value, (bytes, bytearray, memoryview)):
            return bytes(value).hex()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"launch argument '{name}' must be an int, a number or bytes")
        if isinstance(value, int):
            if value < 0:
                raise ValueError(f"launch argument '{name}' is negative")
            return value
        if not math.isfinite(value):
            raise ValueError(f"launch argument '{name}' is not a finite number")
        return value

    def _state_json(self) -> bytes:
        document: dict[str, object] = {
            "format": "nanosm-launch-state-v0",
            "shmem": self.shmem._json(),
            "gmem": self.gmem._json(),
        }
        if self._warps is not None:
            document["active_warps"] = self._warps
        if self.args:
            document["args"] = {name: self._argument_json(name, value) for name, value in self.args.items()}
        if self.block_address is not None:
            document["block_address"] = self.block_address
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

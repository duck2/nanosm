from __future__ import annotations

import ctypes
import os
from pathlib import Path


OK = 0
INVALID_HANDLE = 0
SIZE_ERROR = (1 << 64) - 1
TRACE_TIMELINE = 1 << 0
TRACE_DEBUG = 1 << 1
_dll_directories: list[object] = []


def _load_library() -> ctypes.CDLL:
    root = Path(__file__).resolve().parents[1]
    candidates: list[Path] = []
    configured = os.environ.get("NANOSM_LIBRARY")
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            root / "build" / "libnanosm_capi.dll",
            root / "build" / "nanosm_capi.dll",
            root / "build" / "libnanosm_capi.so",
            root / "build" / "libnanosm_capi.dylib",
        ]
    )
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        for directory in [root / "build", Path(r"C:\msys64\mingw64\bin")]:
            if directory.is_dir():
                _dll_directories.append(os.add_dll_directory(str(directory)))
    failures: list[str] = []
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            return ctypes.CDLL(str(candidate))
        except OSError as error:
            failures.append(f"{candidate}: {error}")
    detail = "\n".join(failures) if failures else "no candidate library exists"
    raise RuntimeError(
        "could not load nanosm C API; build the project first or set NANOSM_LIBRARY\n" + detail
    )


lib = _load_library()

lib.nanosm_mbar_encode.argtypes = [
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_int32,  # tx_bytes: the signed transaction count
    ctypes.POINTER(ctypes.c_uint64),
]
lib.nanosm_mbar_encode.restype = ctypes.c_int32
lib.nanosm_tma_descriptor_encode.argtypes = [
    ctypes.c_uint64,  # base
    ctypes.c_uint32,  # element_bytes
    *[ctypes.c_uint32] * 4,  # dimensions d0..d3
    *[ctypes.c_uint32] * 3,  # strides s1..s3
    ctypes.c_uint32,  # tile_length
    ctypes.c_uint32,  # tile_rows
    ctypes.c_uint32,  # row_dimension
    ctypes.c_uint32,  # layout (0 dense, 1 xor64)
    ctypes.c_uint32,  # transposed
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_uint64,
]
lib.nanosm_tma_descriptor_encode.restype = ctypes.c_int32
lib.nanosm_session_create_qtx.argtypes = [
    ctypes.c_char_p,
    ctypes.c_uint64,
    ctypes.c_char_p,
    ctypes.c_uint64,
    ctypes.c_char_p,
    ctypes.c_uint64,
    ctypes.c_uint32,
    ctypes.c_uint64,
]
lib.nanosm_session_create_qtx.restype = ctypes.c_uint32
lib.nanosm_session_create_image.argtypes = [
    ctypes.c_char_p,
    ctypes.c_uint64,
    ctypes.c_char_p,
    ctypes.c_uint64,
    ctypes.c_uint32,
    ctypes.c_uint64,
]
lib.nanosm_session_create_image.restype = ctypes.c_uint32
lib.nanosm_session_run.argtypes = [ctypes.c_uint32, ctypes.c_uint64]
lib.nanosm_session_run.restype = ctypes.c_int32
lib.nanosm_session_destroy.argtypes = [ctypes.c_uint32]
lib.nanosm_session_destroy.restype = ctypes.c_int32
lib.nanosm_session_cycle.argtypes = [ctypes.c_uint32]
lib.nanosm_session_cycle.restype = ctypes.c_uint64
for name in ["nanosm_session_read_gmem", "nanosm_session_read_shmem"]:
    function = getattr(lib, name)
    function.argtypes = [
        ctypes.c_uint32,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint64,
    ]
    function.restype = ctypes.c_int32
for name in [
    "nanosm_session_summary",
    "nanosm_session_trace_json",
    "nanosm_session_launch_image",
]:
    function = getattr(lib, name)
    function.argtypes = [ctypes.c_uint32, ctypes.c_char_p, ctypes.c_uint64]
    function.restype = ctypes.c_uint64
lib.nanosm_last_error.argtypes = [ctypes.c_char_p, ctypes.c_uint64]
lib.nanosm_last_error.restype = ctypes.c_uint64


def last_error() -> str:
    size = int(lib.nanosm_last_error(None, 0))
    buffer = ctypes.create_string_buffer(size + 1)
    lib.nanosm_last_error(buffer, len(buffer))
    return buffer.value.decode("utf-8", errors="replace")


def _check(status: int) -> None:
    if status != OK:
        raise RuntimeError(last_error())


def encode_mbar(phase: int, expected: int, remaining: int, tx_bytes: int) -> int:
    output = ctypes.c_uint64()
    _check(lib.nanosm_mbar_encode(phase, expected, remaining, tx_bytes, ctypes.byref(output)))
    return int(output.value)


def encode_tma_descriptor(
    base: int,
    element_bytes: int,
    dimensions: tuple[int, int, int, int],
    strides: tuple[int, int, int],
    tile_length: int,
    tile_rows: int,
    row_dimension: int,
    layout: int,
    transposed: int,
) -> bytes:
    output = (ctypes.c_uint8 * 64)()
    _check(
        lib.nanosm_tma_descriptor_encode(
            base,
            element_bytes,
            *dimensions,
            *strides,
            tile_length,
            tile_rows,
            row_dimension,
            layout,
            transposed,
            output,
            len(output),
        )
    )
    return bytes(output)


def _string(function, handle: int) -> str:
    size = int(function(handle, None, 0))
    if size == SIZE_ERROR:
        raise RuntimeError(last_error())
    buffer = ctypes.create_string_buffer(size + 1)
    actual = int(function(handle, buffer, len(buffer)))
    if actual == SIZE_ERROR:
        raise RuntimeError(last_error())
    if actual != size:
        raise RuntimeError("nanosm string result changed between size query and copy")
    return buffer.value.decode("utf-8")


class Session:
    def __init__(self, handle: int):
        if handle == INVALID_HANDLE:
            raise RuntimeError(last_error())
        self._handle = handle

    @classmethod
    def from_qtx(
        cls,
        profile_json: bytes,
        qtx_source: bytes,
        launch_state_json: bytes,
        trace_flags: int,
        max_trace_events: int,
    ) -> "Session":
        return cls(
            int(
                lib.nanosm_session_create_qtx(
                    profile_json,
                    len(profile_json),
                    qtx_source,
                    len(qtx_source),
                    launch_state_json,
                    len(launch_state_json),
                    trace_flags,
                    max_trace_events,
                )
            )
        )

    @classmethod
    def from_image(
        cls,
        profile_json: bytes,
        launch_image_json: bytes,
        trace_flags: int,
        max_trace_events: int,
    ) -> "Session":
        return cls(
            int(
                lib.nanosm_session_create_image(
                    profile_json,
                    len(profile_json),
                    launch_image_json,
                    len(launch_image_json),
                    trace_flags,
                    max_trace_events,
                )
            )
        )

    def _live_handle(self) -> int:
        if self._handle == INVALID_HANDLE:
            raise RuntimeError("nanosm session is closed")
        return self._handle

    def run(self, cycle_limit: int) -> None:
        _check(lib.nanosm_session_run(self._live_handle(), cycle_limit))

    @property
    def cycle(self) -> int:
        value = int(lib.nanosm_session_cycle(self._live_handle()))
        if value == SIZE_ERROR:
            raise RuntimeError(last_error())
        return value

    @property
    def summary(self) -> str:
        return _string(lib.nanosm_session_summary, self._live_handle())

    @property
    def trace_json(self) -> str:
        return _string(lib.nanosm_session_trace_json, self._live_handle())

    @property
    def launch_image(self) -> str:
        return _string(lib.nanosm_session_launch_image, self._live_handle())

    def _read(self, function, address: int, size: int) -> bytes:
        if address < 0 or size < 0:
            raise ValueError("memory address and size must be nonnegative")
        buffer = (ctypes.c_uint8 * size)()
        _check(function(self._live_handle(), address, buffer, size))
        return bytes(buffer)

    def read_gmem(self, address: int, size: int) -> bytes:
        return self._read(lib.nanosm_session_read_gmem, address, size)

    def read_shmem(self, address: int, size: int) -> bytes:
        return self._read(lib.nanosm_session_read_shmem, address, size)

    def close(self) -> None:
        if self._handle != INVALID_HANDLE:
            handle = self._handle
            self._handle = INVALID_HANDLE
            _check(lib.nanosm_session_destroy(handle))

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def __del__(self):
        if getattr(self, "_handle", INVALID_HANDLE) != INVALID_HANDLE:
            lib.nanosm_session_destroy(self._handle)
            self._handle = INVALID_HANDLE

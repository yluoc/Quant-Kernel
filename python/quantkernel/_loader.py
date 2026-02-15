"""Library finder and ABI version check."""

import ctypes as ct
import os
import sys
from pathlib import Path

from ._abi import (
    ABI_MAJOR,
    ABI_MINOR,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _search_dirs() -> list[Path]:
    search_dirs: list[Path] = []

    env_path = os.environ.get("QK_LIB_PATH")
    if env_path:
        search_dirs.append(Path(env_path))

    root = _project_root()
    search_dirs.append(root / "build" / "cpp")
    search_dirs.append(root / "build")
    search_dirs.append(root)
    return search_dirs


def _kernel_names() -> list[str]:
    if sys.platform == "win32":
        return ["quantkernel.dll", "libquantkernel.dll"]
    if sys.platform == "darwin":
        return ["libquantkernel.dylib"]
    return ["libquantkernel.so"]


def _find_library(names: list[str], search_dirs: list[Path]) -> Path | None:
    for d in search_dirs:
        for name in names:
            candidate = d / name
            if candidate.exists():
                return candidate
    return None


def _find_library_or_raise(names: list[str], search_dirs: list[Path], purpose: str) -> Path:
    found = _find_library(names, search_dirs)
    if found is not None:
        return found

    raise OSError(
        f"Cannot find {purpose} shared library. "
        f"Searched: {[str(d) for d in search_dirs]}. "
        f"Set QK_LIB_PATH and build first."
    )


def _configure_function_signatures(lib: ct.CDLL) -> None:
    lib.qk_abi_version.restype = None
    lib.qk_abi_version.argtypes = [ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32)]


def _verify_abi(lib: ct.CDLL) -> None:
    major = ct.c_int32(-1)
    minor = ct.c_int32(-1)
    lib.qk_abi_version(ct.byref(major), ct.byref(minor))

    if major.value != ABI_MAJOR:
        raise RuntimeError(
            f"ABI major version mismatch: library={major.value}, expected={ABI_MAJOR}"
        )
    if minor.value < ABI_MINOR:
        raise RuntimeError(
            f"ABI minor version too old: library={minor.value}, expected>={ABI_MINOR}"
        )


def load_library() -> ct.CDLL:
    """Load the shared library and verify ABI version."""
    search_dirs = _search_dirs()
    path = _find_library_or_raise(_kernel_names(), search_dirs, "kernel")

    lib = ct.CDLL(str(path))
    _configure_function_signatures(lib)

    _verify_abi(lib)
    return lib

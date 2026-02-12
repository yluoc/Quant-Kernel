"""ABI version checks."""

import ctypes as ct
from quantkernel import ABI_MAJOR, ABI_MINOR


def test_abi_version_matches(qk):
    """Library ABI version matches Python expectations."""
    major = ct.c_int32(-1)
    minor = ct.c_int32(-1)
    qk._lib.qk_abi_version(ct.byref(major), ct.byref(minor))
    assert major.value == ABI_MAJOR
    assert minor.value >= ABI_MINOR

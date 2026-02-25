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


def test_public_abi_wrappers(qk):
    """QuantKernel exposes low-level ABI/plugin/error C APIs."""
    major, minor = qk.abi_version()
    assert major == ABI_MAJOR
    assert minor >= ABI_MINOR

    p_major, p_minor, p_name = qk.plugin_get_api()
    assert p_major == ABI_MAJOR
    assert p_minor >= ABI_MINOR
    assert isinstance(p_name, str)
    assert p_name != ""

    qk.clear_last_error()
    assert qk.get_last_error() == ""

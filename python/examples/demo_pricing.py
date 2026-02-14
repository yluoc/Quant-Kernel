#!/usr/bin/env python3
"""QuantKernel demo — library load and ABI check."""

import ctypes as ct
from quantkernel import QuantKernel


def main():
    qk = QuantKernel()
    major = ct.c_int32(-1)
    minor = ct.c_int32(-1)
    qk._lib.qk_abi_version(ct.byref(major), ct.byref(minor))
    print(f"QuantKernel ABI: {major.value}.{minor.value}")


if __name__ == "__main__":
    main()

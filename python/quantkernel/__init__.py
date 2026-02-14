"""QuantKernel — high-performance derivative pricing engine."""

from .engine import QuantKernel
from ._abi import (
    QK_CALL,
    QK_PUT,
    QK_OK,
    QK_ERR_NULL_PTR,
    QK_ERR_BAD_SIZE,
    QK_ERR_ABI_MISMATCH,
    QK_ERR_RUNTIME_INIT,
    ABI_MAJOR,
    ABI_MINOR,
)

__all__ = [
    "QuantKernel",
    "QK_CALL",
    "QK_PUT",
    "QK_OK",
    "QK_ERR_NULL_PTR",
    "QK_ERR_BAD_SIZE",
    "QK_ERR_ABI_MISMATCH",
    "QK_ERR_RUNTIME_INIT",
    "ABI_MAJOR",
    "ABI_MINOR",
]

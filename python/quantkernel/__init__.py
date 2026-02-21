"""QuantKernel — high-performance derivative pricing engine."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("quant-kernel")
except PackageNotFoundError:
    __version__ = "0.0.0"

from .engine import QuantKernel, QKError, QKNullPointerError, QKBadSizeError, QKInvalidInputError
from .accelerator import QuantAccelerator
from ._abi import (
    QK_CALL,
    QK_PUT,
    QK_OK,
    QK_ERR_NULL_PTR,
    QK_ERR_BAD_SIZE,
    QK_ERR_ABI_MISMATCH,
    QK_ERR_RUNTIME_INIT,
    QK_ERR_INVALID_INPUT,
    ABI_MAJOR,
    ABI_MINOR,
)

__all__ = [
    "QuantKernel",
    "QuantAccelerator",
    "QKError",
    "QKNullPointerError",
    "QKBadSizeError",
    "QKInvalidInputError",
    "QK_CALL",
    "QK_PUT",
    "QK_OK",
    "QK_ERR_NULL_PTR",
    "QK_ERR_BAD_SIZE",
    "QK_ERR_ABI_MISMATCH",
    "QK_ERR_RUNTIME_INIT",
    "QK_ERR_INVALID_INPUT",
    "ABI_MAJOR",
    "ABI_MINOR",
]

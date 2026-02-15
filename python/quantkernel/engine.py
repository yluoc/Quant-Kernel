"""QuantKernel high-level Python API."""

from ._abi import QK_CALL, QK_PUT
from ._loader import load_library


class QuantKernel:
    """Thin wrapper for the QuantKernel shared library."""

    CALL = QK_CALL
    PUT = QK_PUT

    def __init__(self):
        self._lib = load_library()

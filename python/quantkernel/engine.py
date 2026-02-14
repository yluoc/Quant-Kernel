"""QuantKernel high-level Python API."""

from ._abi import QK_CALL, QK_PUT, QK_OK
from ._loader import load_library


class QuantKernel:
    """Thin runtime/control wrapper for the QuantKernel shared library."""

    CALL = QK_CALL
    PUT = QK_PUT

    def __init__(self):
        self._lib = load_library()

    def _require_runtime_symbol(self, symbol_name: str):
        fn = getattr(self._lib, symbol_name, None)
        if fn is None:
            raise RuntimeError(
                "Runtime control API unavailable on this library. "
                "Set QK_USE_RUNTIME=1 to load the Rust runtime shell."
            )
        return fn

    def runtime_load_plugin(self, plugin_path: str) -> None:
        """Load or replace the active runtime plugin by filesystem path."""
        fn = self._require_runtime_symbol("qk_runtime_load_plugin")
        encoded = str(plugin_path).encode("utf-8")
        rc = fn(encoded)
        if rc != QK_OK:
            raise RuntimeError(f"qk_runtime_load_plugin failed with return code {rc}")

    def runtime_unload_plugin(self) -> None:
        """Unload the active runtime plugin."""
        fn = self._require_runtime_symbol("qk_runtime_unload_plugin")
        rc = fn()
        if rc != QK_OK:
            raise RuntimeError(f"qk_runtime_unload_plugin failed with return code {rc}")

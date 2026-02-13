"""Runtime shell loading tests."""

from pathlib import Path

import numpy as np
import pytest

from quantkernel import QK_CALL, QK_ROW_OK, QuantKernel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_CANDIDATES = [
    PROJECT_ROOT / "target" / "release" / "libquantkernel_runtime.so",
    PROJECT_ROOT / "rust" / "runtime" / "target" / "release" / "libquantkernel_runtime.so",
]
RUNTIME_LIB = next((p for p in RUNTIME_CANDIDATES if p.exists()), RUNTIME_CANDIDATES[0])
CPP_PLUGIN = PROJECT_ROOT / "build" / "cpp" / "libquantkernel.so"


@pytest.mark.skipif(not RUNTIME_LIB.exists(), reason="Rust runtime library is not built")
@pytest.mark.skipif(not CPP_PLUGIN.exists(), reason="C++ plugin library is not built")
def test_python_can_route_through_runtime(monkeypatch):
    monkeypatch.setenv("QK_USE_RUNTIME", "1")
    monkeypatch.setenv("QK_PLUGIN_PATH", str(CPP_PLUGIN))

    qk = QuantKernel()
    result = qk.bs_price(
        spot=np.array([100.0]),
        strike=np.array([100.0]),
        time_to_expiry=np.array([1.0]),
        volatility=np.array([0.20]),
        risk_free_rate=np.array([0.05]),
        dividend_yield=np.array([0.0]),
        option_type=np.array([QK_CALL], dtype=np.int32),
    )

    assert result["error_codes"][0] == QK_ROW_OK
    np.testing.assert_allclose(result["price"][0], 10.4506, atol=0.001)


def test_runtime_controls_not_available_without_runtime(monkeypatch):
    monkeypatch.delenv("QK_USE_RUNTIME", raising=False)
    monkeypatch.delenv("QK_PLUGIN_PATH", raising=False)

    qk = QuantKernel()
    with pytest.raises(RuntimeError, match="Runtime control API unavailable"):
        qk.runtime_unload_plugin()


@pytest.mark.skipif(not RUNTIME_LIB.exists(), reason="Rust runtime library is not built")
def test_runtime_bad_plugin_path_raises(monkeypatch):
    monkeypatch.setenv("QK_USE_RUNTIME", "1")
    monkeypatch.setenv("QK_PLUGIN_PATH", str(PROJECT_ROOT / "does_not_exist.so"))

    with pytest.raises(RuntimeError, match="qk_runtime_load_plugin failed"):
        QuantKernel()


@pytest.mark.skipif(not RUNTIME_LIB.exists(), reason="Rust runtime library is not built")
@pytest.mark.skipif(not CPP_PLUGIN.exists(), reason="C++ plugin library is not built")
def test_runtime_unload_and_reload(monkeypatch):
    monkeypatch.setenv("QK_USE_RUNTIME", "1")
    monkeypatch.setenv("QK_PLUGIN_PATH", str(CPP_PLUGIN))

    qk = QuantKernel()
    qk.runtime_unload_plugin()

    with pytest.raises(RuntimeError, match="qk_bs_price failed with return code -4"):
        qk.bs_price(
            spot=np.array([100.0]),
            strike=np.array([100.0]),
            time_to_expiry=np.array([1.0]),
            volatility=np.array([0.20]),
            risk_free_rate=np.array([0.05]),
            dividend_yield=np.array([0.0]),
            option_type=np.array([QK_CALL], dtype=np.int32),
        )

    qk.runtime_load_plugin(str(CPP_PLUGIN))
    result = qk.bs_price(
        spot=np.array([100.0]),
        strike=np.array([100.0]),
        time_to_expiry=np.array([1.0]),
        volatility=np.array([0.20]),
        risk_free_rate=np.array([0.05]),
        dividend_yield=np.array([0.0]),
        option_type=np.array([QK_CALL], dtype=np.int32),
    )
    assert result["error_codes"][0] == QK_ROW_OK

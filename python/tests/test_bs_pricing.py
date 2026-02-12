"""Black-Scholes pricing tests."""

import numpy as np
import pytest
from quantkernel import QK_CALL, QK_PUT, QK_ROW_OK


def test_atm_call(qk):
    """ATM call: S=100, K=100, T=1, vol=0.20, r=0.05, q=0."""
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
    np.testing.assert_allclose(result["delta"][0], 0.6368, atol=0.001)
    np.testing.assert_allclose(result["gamma"][0], 0.01876, atol=0.0005)
    np.testing.assert_allclose(result["vega"][0], 0.3752, atol=0.001)


def test_atm_put(qk):
    """ATM put — verify put-call parity."""
    params = dict(
        spot=np.array([100.0]),
        strike=np.array([100.0]),
        time_to_expiry=np.array([1.0]),
        volatility=np.array([0.20]),
        risk_free_rate=np.array([0.05]),
        dividend_yield=np.array([0.0]),
    )
    call = qk.bs_price(option_type=np.array([QK_CALL], dtype=np.int32), **params)
    put = qk.bs_price(option_type=np.array([QK_PUT], dtype=np.int32), **params)

    # Put-call parity: C - P = S - K*e^(-rT)
    S, K, r, T = 100.0, 100.0, 0.05, 1.0
    parity = S - K * np.exp(-r * T)
    np.testing.assert_allclose(
        call["price"][0] - put["price"][0], parity, atol=0.001
    )


def test_batch_pricing(qk):
    """Batch with 3 options."""
    n = 3
    result = qk.bs_price(
        spot=np.array([100.0, 110.0, 90.0]),
        strike=np.array([100.0, 105.0, 95.0]),
        time_to_expiry=np.array([1.0, 0.5, 0.25]),
        volatility=np.array([0.20, 0.25, 0.30]),
        risk_free_rate=np.array([0.05, 0.03, 0.04]),
        dividend_yield=np.array([0.0, 0.01, 0.02]),
        option_type=np.array([QK_CALL, QK_PUT, QK_CALL], dtype=np.int32),
    )
    assert len(result["price"]) == n
    assert all(ec == QK_ROW_OK for ec in result["error_codes"])
    # All prices should be positive for valid inputs
    assert all(p > 0 for p in result["price"])


def test_deep_itm_call(qk):
    """Deep ITM call: delta close to 1, price close to intrinsic."""
    result = qk.bs_price(
        spot=np.array([200.0]),
        strike=np.array([100.0]),
        time_to_expiry=np.array([0.01]),
        volatility=np.array([0.20]),
        risk_free_rate=np.array([0.0]),
        dividend_yield=np.array([0.0]),
        option_type=np.array([QK_CALL], dtype=np.int32),
    )
    assert result["error_codes"][0] == QK_ROW_OK
    np.testing.assert_allclose(result["price"][0], 100.0, atol=1.0)
    np.testing.assert_allclose(result["delta"][0], 1.0, atol=0.01)

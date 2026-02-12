"""Implied volatility solver tests."""

import numpy as np
import pytest
from quantkernel import QK_CALL, QK_PUT, QK_ROW_OK, QK_ROW_ERR_IV_NO_CONV


def test_round_trip_call(qk):
    """Price with known vol, then recover it via IV solver."""
    vol_true = 0.25
    result = qk.bs_price(
        spot=np.array([100.0]),
        strike=np.array([100.0]),
        time_to_expiry=np.array([1.0]),
        volatility=np.array([vol_true]),
        risk_free_rate=np.array([0.05]),
        dividend_yield=np.array([0.0]),
        option_type=np.array([QK_CALL], dtype=np.int32),
    )
    market_price = result["price"]

    iv_result = qk.iv_solve(
        spot=np.array([100.0]),
        strike=np.array([100.0]),
        time_to_expiry=np.array([1.0]),
        risk_free_rate=np.array([0.05]),
        dividend_yield=np.array([0.0]),
        option_type=np.array([QK_CALL], dtype=np.int32),
        market_price=market_price,
        tol=1e-10,
    )
    assert iv_result["error_codes"][0] == QK_ROW_OK
    np.testing.assert_allclose(iv_result["implied_vol"][0], vol_true, atol=1e-6)


def test_round_trip_put(qk):
    """Round-trip for put option."""
    vol_true = 0.30
    result = qk.bs_price(
        spot=np.array([110.0]),
        strike=np.array([100.0]),
        time_to_expiry=np.array([0.5]),
        volatility=np.array([vol_true]),
        risk_free_rate=np.array([0.03]),
        dividend_yield=np.array([0.02]),
        option_type=np.array([QK_PUT], dtype=np.int32),
    )

    iv_result = qk.iv_solve(
        spot=np.array([110.0]),
        strike=np.array([100.0]),
        time_to_expiry=np.array([0.5]),
        risk_free_rate=np.array([0.03]),
        dividend_yield=np.array([0.02]),
        option_type=np.array([QK_PUT], dtype=np.int32),
        market_price=result["price"],
        tol=1e-10,
    )
    assert iv_result["error_codes"][0] == QK_ROW_OK
    np.testing.assert_allclose(iv_result["implied_vol"][0], vol_true, atol=1e-6)


def test_batch_round_trip(qk):
    """Batch round-trip with multiple vols."""
    n = 5
    vols_true = np.array([0.10, 0.20, 0.30, 0.40, 0.50])
    spots = np.full(n, 100.0)
    strikes = np.full(n, 100.0)
    times = np.full(n, 1.0)
    rates = np.full(n, 0.05)
    divs = np.full(n, 0.0)
    types = np.full(n, QK_CALL, dtype=np.int32)

    result = qk.bs_price(
        spot=spots, strike=strikes, time_to_expiry=times,
        volatility=vols_true, risk_free_rate=rates,
        dividend_yield=divs, option_type=types,
    )

    iv_result = qk.iv_solve(
        spot=spots, strike=strikes, time_to_expiry=times,
        risk_free_rate=rates, dividend_yield=divs,
        option_type=types, market_price=result["price"],
    )
    assert all(ec == QK_ROW_OK for ec in iv_result["error_codes"])
    np.testing.assert_allclose(iv_result["implied_vol"], vols_true, atol=1e-6)

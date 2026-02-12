"""Error handling tests."""

import numpy as np
import pytest
from quantkernel import (
    QK_CALL,
    QK_ROW_OK,
    QK_ROW_ERR_NEGATIVE_S,
    QK_ROW_ERR_NEGATIVE_K,
    QK_ROW_ERR_NEGATIVE_T,
    QK_ROW_ERR_NEGATIVE_V,
    QK_ROW_ERR_BAD_TYPE,
    QK_ROW_ERR_BAD_PRICE,
    QK_ROW_ERR_NON_FINITE,
)


def test_negative_spot(qk):
    result = qk.bs_price(
        spot=np.array([-100.0]),
        strike=np.array([100.0]),
        time_to_expiry=np.array([1.0]),
        volatility=np.array([0.20]),
        risk_free_rate=np.array([0.05]),
        dividend_yield=np.array([0.0]),
        option_type=np.array([QK_CALL], dtype=np.int32),
    )
    assert result["error_codes"][0] == QK_ROW_ERR_NEGATIVE_S
    assert np.isnan(result["price"][0])


def test_negative_strike(qk):
    result = qk.bs_price(
        spot=np.array([100.0]),
        strike=np.array([-100.0]),
        time_to_expiry=np.array([1.0]),
        volatility=np.array([0.20]),
        risk_free_rate=np.array([0.05]),
        dividend_yield=np.array([0.0]),
        option_type=np.array([QK_CALL], dtype=np.int32),
    )
    assert result["error_codes"][0] == QK_ROW_ERR_NEGATIVE_K
    assert np.isnan(result["price"][0])


def test_negative_time(qk):
    result = qk.bs_price(
        spot=np.array([100.0]),
        strike=np.array([100.0]),
        time_to_expiry=np.array([-1.0]),
        volatility=np.array([0.20]),
        risk_free_rate=np.array([0.05]),
        dividend_yield=np.array([0.0]),
        option_type=np.array([QK_CALL], dtype=np.int32),
    )
    assert result["error_codes"][0] == QK_ROW_ERR_NEGATIVE_T
    assert np.isnan(result["price"][0])


def test_negative_vol(qk):
    result = qk.bs_price(
        spot=np.array([100.0]),
        strike=np.array([100.0]),
        time_to_expiry=np.array([1.0]),
        volatility=np.array([-0.20]),
        risk_free_rate=np.array([0.05]),
        dividend_yield=np.array([0.0]),
        option_type=np.array([QK_CALL], dtype=np.int32),
    )
    assert result["error_codes"][0] == QK_ROW_ERR_NEGATIVE_V
    assert np.isnan(result["price"][0])


def test_bad_option_type(qk):
    result = qk.bs_price(
        spot=np.array([100.0]),
        strike=np.array([100.0]),
        time_to_expiry=np.array([1.0]),
        volatility=np.array([0.20]),
        risk_free_rate=np.array([0.05]),
        dividend_yield=np.array([0.0]),
        option_type=np.array([99], dtype=np.int32),
    )
    assert result["error_codes"][0] == QK_ROW_ERR_BAD_TYPE
    assert np.isnan(result["price"][0])


def test_nan_input(qk):
    result = qk.bs_price(
        spot=np.array([np.nan]),
        strike=np.array([100.0]),
        time_to_expiry=np.array([1.0]),
        volatility=np.array([0.20]),
        risk_free_rate=np.array([0.05]),
        dividend_yield=np.array([0.0]),
        option_type=np.array([QK_CALL], dtype=np.int32),
    )
    assert result["error_codes"][0] == QK_ROW_ERR_NON_FINITE
    assert np.isnan(result["price"][0])


def test_bad_market_price_iv(qk):
    result = qk.iv_solve(
        spot=np.array([100.0]),
        strike=np.array([100.0]),
        time_to_expiry=np.array([1.0]),
        risk_free_rate=np.array([0.05]),
        dividend_yield=np.array([0.0]),
        option_type=np.array([QK_CALL], dtype=np.int32),
        market_price=np.array([-5.0]),
    )
    assert result["error_codes"][0] == QK_ROW_ERR_BAD_PRICE
    assert np.isnan(result["implied_vol"][0])


def test_mixed_valid_invalid(qk):
    """Batch where some rows are valid and some invalid."""
    result = qk.bs_price(
        spot=np.array([100.0, -50.0, 100.0]),
        strike=np.array([100.0, 100.0, 100.0]),
        time_to_expiry=np.array([1.0, 1.0, 1.0]),
        volatility=np.array([0.20, 0.20, 0.20]),
        risk_free_rate=np.array([0.05, 0.05, 0.05]),
        dividend_yield=np.array([0.0, 0.0, 0.0]),
        option_type=np.array([QK_CALL, QK_CALL, QK_CALL], dtype=np.int32),
    )
    assert result["error_codes"][0] == QK_ROW_OK
    assert result["error_codes"][1] == QK_ROW_ERR_NEGATIVE_S
    assert result["error_codes"][2] == QK_ROW_OK
    assert result["price"][0] > 0
    assert np.isnan(result["price"][1])
    assert result["price"][2] > 0


def test_bs_mismatched_lengths_raise(qk):
    with pytest.raises(ValueError, match="identical length"):
        qk.bs_price(
            spot=np.array([100.0, 101.0]),
            strike=np.array([100.0]),
            time_to_expiry=np.array([1.0, 1.0]),
            volatility=np.array([0.20, 0.20]),
            risk_free_rate=np.array([0.05, 0.05]),
            dividend_yield=np.array([0.0, 0.0]),
            option_type=np.array([QK_CALL, QK_CALL], dtype=np.int32),
        )


def test_iv_mismatched_lengths_raise(qk):
    with pytest.raises(ValueError, match="identical length"):
        qk.iv_solve(
            spot=np.array([100.0]),
            strike=np.array([100.0]),
            time_to_expiry=np.array([1.0]),
            risk_free_rate=np.array([0.05]),
            dividend_yield=np.array([0.0]),
            option_type=np.array([QK_CALL], dtype=np.int32),
            market_price=np.array([10.0, 11.0]),
        )


def test_bs_non_1d_raise(qk):
    with pytest.raises(ValueError, match="1-D array"):
        qk.bs_price(
            spot=np.array([[100.0]]),
            strike=np.array([100.0]),
            time_to_expiry=np.array([1.0]),
            volatility=np.array([0.20]),
            risk_free_rate=np.array([0.05]),
            dividend_yield=np.array([0.0]),
            option_type=np.array([QK_CALL], dtype=np.int32),
        )

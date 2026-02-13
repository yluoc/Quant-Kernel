"""Monte Carlo pricing tests."""

import numpy as np
from quantkernel import QK_CALL, QK_PUT, QK_ROW_OK, QK_ROW_ERR_BAD_PATHS


def test_mc_call_matches_bs_within_confidence(qk):
    bs = qk.bs_price(
        spot=np.array([100.0]),
        strike=np.array([100.0]),
        time_to_expiry=np.array([1.0]),
        volatility=np.array([0.20]),
        risk_free_rate=np.array([0.05]),
        dividend_yield=np.array([0.0]),
        option_type=np.array([QK_CALL], dtype=np.int32),
    )
    mc = qk.mc_price(
        spot=np.array([100.0]),
        strike=np.array([100.0]),
        time_to_expiry=np.array([1.0]),
        volatility=np.array([0.20]),
        risk_free_rate=np.array([0.05]),
        dividend_yield=np.array([0.0]),
        option_type=np.array([QK_CALL], dtype=np.int32),
        num_paths=np.array([200000], dtype=np.int32),
        rng_seed=np.array([42], dtype=np.uint64),
    )
    assert mc["error_codes"][0] == QK_ROW_OK
    assert mc["paths_used"][0] == 200000
    assert mc["std_error"][0] > 0.0
    tolerance = 4.0 * mc["std_error"][0] + 0.05
    np.testing.assert_allclose(mc["price"][0], bs["price"][0], atol=tolerance)


def test_mc_seed_reproducibility(qk):
    kwargs = dict(
        spot=np.array([100.0]),
        strike=np.array([100.0]),
        time_to_expiry=np.array([1.0]),
        volatility=np.array([0.20]),
        risk_free_rate=np.array([0.05]),
        dividend_yield=np.array([0.0]),
        option_type=np.array([QK_PUT], dtype=np.int32),
        num_paths=np.array([50000], dtype=np.int32),
        rng_seed=np.array([7], dtype=np.uint64),
    )
    a = qk.mc_price(**kwargs)
    b = qk.mc_price(**kwargs)
    assert a["error_codes"][0] == QK_ROW_OK
    assert b["error_codes"][0] == QK_ROW_OK
    assert a["price"][0] == b["price"][0]
    assert a["std_error"][0] == b["std_error"][0]


def test_mc_bad_paths_row_error(qk):
    mc = qk.mc_price(
        spot=np.array([100.0]),
        strike=np.array([100.0]),
        time_to_expiry=np.array([1.0]),
        volatility=np.array([0.20]),
        risk_free_rate=np.array([0.05]),
        dividend_yield=np.array([0.0]),
        option_type=np.array([QK_CALL], dtype=np.int32),
        num_paths=np.array([0], dtype=np.int32),
        rng_seed=np.array([123], dtype=np.uint64),
    )
    assert mc["error_codes"][0] == QK_ROW_ERR_BAD_PATHS
    assert mc["paths_used"][0] == 0
    assert np.isnan(mc["price"][0])
    assert np.isnan(mc["std_error"][0])

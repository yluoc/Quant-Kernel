"""Machine-learning API checks."""

import math

from quantkernel import QK_CALL, QK_PUT


def test_machine_learning_methods_are_callable_and_close_to_bsm(qk):
    spot = 100.0
    strike = 100.0
    t = 1.0
    vol = 0.2
    r = 0.03
    q = 0.01
    bsm = qk.black_scholes_merton_price(spot, strike, t, vol, r, q, QK_CALL)

    deep_bsde = qk.deep_bsde_price(spot, strike, t, vol, r, q, QK_CALL)
    pinns = qk.pinns_price(spot, strike, t, vol, r, q, QK_CALL)
    deep_hedging = qk.deep_hedging_price(spot, strike, t, vol, r, q, QK_CALL)
    neural_sde = qk.neural_sde_calibration_price(
        spot, strike, t, vol, r, q, QK_CALL, target_implied_vol=0.2
    )

    vals = [deep_bsde, pinns, deep_hedging, neural_sde]
    assert all(math.isfinite(v) and v > 0.0 for v in vals)
    assert all(abs(v - bsm) < 3.0 for v in vals)


def test_deep_bsde_put_and_call_are_reasonable(qk):
    s = 105.0
    k = 100.0
    t = 0.8
    vol = 0.24
    r = 0.02
    q = 0.01

    call = qk.deep_bsde_price(s, k, t, vol, r, q, QK_CALL)
    put = qk.deep_bsde_price(s, k, t, vol, r, q, QK_PUT)

    bsm_call = qk.black_scholes_merton_price(s, k, t, vol, r, q, QK_CALL)
    bsm_put = qk.black_scholes_merton_price(s, k, t, vol, r, q, QK_PUT)

    assert call > 0.0 and put > 0.0
    assert abs(call - bsm_call) < 3.0
    assert abs(put - bsm_put) < 3.0


def test_deep_hedging_respects_option_type(qk):
    common = dict(spot=100.0, strike=110.0, t=1.0, vol=0.25, r=0.02, q=0.01)
    call = qk.deep_hedging_price(**common, option_type=QK_CALL)
    put = qk.deep_hedging_price(**common, option_type=QK_PUT)
    assert call != put


def test_neural_sde_calibration_price_increases_with_target_vol(qk):
    common = dict(spot=100.0, strike=100.0, t=1.0, vol=0.2, r=0.03, q=0.01, option_type=QK_CALL)
    low = qk.neural_sde_calibration_price(**common, target_implied_vol=0.15, calibration_steps=300)
    high = qk.neural_sde_calibration_price(**common, target_implied_vol=0.30, calibration_steps=300)
    assert high > low

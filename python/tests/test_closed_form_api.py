"""Closed-form API checks."""

import math

from quantkernel import QK_CALL, QK_PUT


def test_closed_form_basics_are_callable(qk):
    bsm = qk.black_scholes_merton_price(100.0, 100.0, 1.0, 0.2, 0.03, 0.01, QK_CALL)
    b76 = qk.black76_price(102.0, 100.0, 1.0, 0.2, 0.03, QK_CALL)
    bach = qk.bachelier_price(100.0, 100.0, 1.0, 8.0, 0.03, QK_CALL)
    assert bsm > 0.0
    assert b76 > 0.0
    assert bach > 0.0


def test_heston_and_variance_gamma_are_callable(qk):
    heston = qk.heston_price_cf(
        100.0, 100.0, 1.0, 0.02, 0.01,
        0.04, 2.0, 0.04, 0.5, -0.5, QK_CALL, 1024, 120.0
    )
    vg = qk.variance_gamma_price_cf(
        100.0, 95.0, 1.2, 0.03, 0.01,
        0.2, -0.1, 0.2, QK_CALL, 1024, 120.0
    )
    assert heston > 0.0
    assert vg > 0.0


def test_merton_sabr_and_dupire_are_callable(qk):
    mjd = qk.merton_jump_diffusion_price(
        95.0, 100.0, 1.2, 0.22, 0.015, 0.005,
        0.2, -0.1, 0.2, 80, QK_CALL
    )
    iv = qk.sabr_hagan_lognormal_iv(100.0, 105.0, 1.0, 0.2, 0.5, -0.2, 0.4)
    sabr_price = qk.sabr_hagan_black76_price(100.0, 105.0, 1.0, 0.03, 0.2, 0.5, -0.2, 0.4, QK_CALL)
    local_vol = qk.dupire_local_vol(100.0, 1.0, 8.0, 2.0, -0.5, 0.03, 0.02, 0.01)
    assert mjd > 0.0
    assert iv > 0.0
    assert sabr_price > 0.0
    assert local_vol > 0.0 and math.isfinite(local_vol)


def test_put_call_parity_for_bsm(qk):
    s = 100.0
    k = 100.0
    t = 1.0
    r = 0.03
    q = 0.01
    call = qk.black_scholes_merton_price(s, k, t, 0.2, r, q, QK_CALL)
    put = qk.black_scholes_merton_price(s, k, t, 0.2, r, q, QK_PUT)
    rhs = s * math.exp(-q * t) - k * math.exp(-r * t)
    assert abs((call - put) - rhs) < 1e-10

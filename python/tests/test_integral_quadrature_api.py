"""Integral quadrature pricing API checks."""

import math

from quantkernel import QK_CALL, QK_PUT


def test_integral_quadrature_methods_are_callable_and_close_to_bsm(qk):
    spot = 100.0
    strike = 100.0
    t = 1.0
    vol = 0.2
    r = 0.03
    q = 0.01

    bsm = qk.black_scholes_merton_price(spot, strike, t, vol, r, q, QK_CALL)
    gh = qk.gauss_hermite_price(spot, strike, t, vol, r, q, QK_CALL, n_points=128)
    gl = qk.gauss_laguerre_price(spot, strike, t, vol, r, q, QK_CALL, n_points=64)
    gleg = qk.gauss_legendre_price(spot, strike, t, vol, r, q, QK_CALL, n_points=128, integration_limit=200.0)
    aq = qk.adaptive_quadrature_price(
        spot, strike, t, vol, r, q, QK_CALL,
        abs_tol=1e-9, rel_tol=1e-8, max_depth=14, integration_limit=200.0,
    )

    vals = [gh, gl, gleg, aq]
    assert all(math.isfinite(v) and v > 0.0 for v in vals)
    assert abs(gh - bsm) < 3e-2
    assert abs(gl - bsm) < 2e-2
    assert abs(gleg - bsm) < 2e-3
    assert abs(aq - bsm) < 2e-3


def test_gauss_hermite_put_call_parity(qk):
    s = 105.0
    k = 100.0
    t = 0.8
    vol = 0.25
    r = 0.02
    q = 0.01

    call = qk.gauss_hermite_price(s, k, t, vol, r, q, QK_CALL, n_points=40)
    put = qk.gauss_hermite_price(s, k, t, vol, r, q, QK_PUT, n_points=40)
    rhs = s * math.exp(-q * t) - k * math.exp(-r * t)
    assert abs((call - put) - rhs) < 2e-2

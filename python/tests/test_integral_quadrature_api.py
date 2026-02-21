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


def test_gauss_laguerre_is_not_legendre_proxy(qk):
    args = dict(spot=100.0, strike=100.0, t=1.0, vol=0.2, r=0.03, q=0.01, option_type=QK_CALL)

    laguerre = qk.gauss_laguerre_price(**args, n_points=16)
    mapped_legendre = qk.gauss_legendre_price(**args, n_points=64, integration_limit=200.0)

    assert abs(laguerre - mapped_legendre) > 1e-5


def test_adaptive_quadrature_controls_accuracy(qk):
    args = dict(spot=100.0, strike=100.0, t=1.0, vol=0.2, r=0.03, q=0.01, option_type=QK_CALL)
    bsm = qk.black_scholes_merton_price(**args)

    loose = qk.adaptive_quadrature_price(
        **args,
        abs_tol=1e-3,
        rel_tol=1e-3,
        max_depth=6,
        integration_limit=200.0,
    )
    tight = qk.adaptive_quadrature_price(
        **args,
        abs_tol=1e-10,
        rel_tol=1e-10,
        max_depth=20,
        integration_limit=200.0,
    )

    assert abs(tight - bsm) < 5e-3
    assert abs(loose - tight) > 1e-4

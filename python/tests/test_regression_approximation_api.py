"""Regression-approximation API checks."""

import math

from quantkernel import QK_CALL, QK_PUT


def test_regression_approximation_methods_are_callable_and_close_to_bsm(qk):
    spot = 100.0
    strike = 100.0
    t = 1.0
    vol = 0.2
    r = 0.03
    q = 0.01
    bsm = qk.black_scholes_merton_price(spot, strike, t, vol, r, q, QK_CALL)

    pce = qk.polynomial_chaos_expansion_price(spot, strike, t, vol, r, q, QK_CALL)
    rbf = qk.radial_basis_function_price(spot, strike, t, vol, r, q, QK_CALL)
    sgc = qk.sparse_grid_collocation_price(spot, strike, t, vol, r, q, QK_CALL)
    pod = qk.proper_orthogonal_decomposition_price(spot, strike, t, vol, r, q, QK_CALL)

    vals = [pce, rbf, sgc, pod]
    assert all(math.isfinite(v) and v > 0.0 for v in vals)
    assert all(abs(v - bsm) < 3.0 for v in vals)


def test_pce_accuracy_improves_with_more_basis_terms(qk):
    common = dict(spot=100.0, strike=100.0, t=1.0, vol=0.2, r=0.03, q=0.01, option_type=QK_CALL)
    bsm = qk.black_scholes_merton_price(**common)

    coarse = qk.polynomial_chaos_expansion_price(**common, polynomial_order=2, quadrature_points=8)
    fine = qk.polynomial_chaos_expansion_price(**common, polynomial_order=8, quadrature_points=64)

    assert abs(fine - bsm) <= abs(coarse - bsm)


def test_rbf_put_call_parity(qk):
    s = 105.0
    k = 100.0
    t = 0.8
    vol = 0.24
    r = 0.02
    q = 0.01

    call = qk.radial_basis_function_price(s, k, t, vol, r, q, QK_CALL)
    put = qk.radial_basis_function_price(s, k, t, vol, r, q, QK_PUT)
    rhs = s * math.exp(-q * t) - k * math.exp(-r * t)
    assert abs((call - put) - rhs) < 1e-9

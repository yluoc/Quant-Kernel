"""Adjoint-greeks API checks."""

import math

from quantkernel import QK_CALL, QK_PUT


def _fd_delta(qk, spot, strike, t, vol, r, q, option_type, bump=1e-3):
    up = qk.black_scholes_merton_price(spot + bump, strike, t, vol, r, q, option_type)
    dn = qk.black_scholes_merton_price(spot - bump, strike, t, vol, r, q, option_type)
    return (up - dn) / (2.0 * bump)


def test_adjoint_greeks_methods_return_finite_and_reasonable_deltas(qk):
    common = dict(spot=100.0, strike=100.0, t=1.0, vol=0.2, r=0.03, q=0.01, option_type=QK_CALL)
    fd = _fd_delta(qk, **common)

    pathwise = qk.pathwise_derivative_delta(**common, paths=30000, seed=11)
    lr = qk.likelihood_ratio_delta(**common, paths=30000, seed=11, weight_clip=6.0)
    aad = qk.aad_delta(**common, tape_steps=96, regularization=1e-6)

    for delta in (pathwise, lr, aad):
        assert math.isfinite(delta)
        assert 0.0 <= delta <= 1.0
        assert abs(delta - fd) < 0.1


def test_put_delta_sign_is_negative(qk):
    common_put = dict(spot=100.0, strike=110.0, t=1.0, vol=0.25, r=0.02, q=0.01, option_type=QK_PUT)

    pathwise = qk.pathwise_derivative_delta(**common_put, paths=25000, seed=7)
    lr = qk.likelihood_ratio_delta(**common_put, paths=25000, seed=7, weight_clip=5.0)
    aad = qk.aad_delta(**common_put, tape_steps=64, regularization=1e-6)

    assert pathwise < 0.0
    assert lr < 0.0
    assert aad < 0.0

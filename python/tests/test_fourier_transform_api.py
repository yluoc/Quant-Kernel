"""Fourier-transform pricing API checks."""

import math

from quantkernel import QK_CALL, QK_PUT


def test_fourier_methods_are_callable_and_close_to_bsm(qk):
    spot = 100.0
    strike = 100.0
    t = 1.0
    vol = 0.2
    r = 0.03
    q = 0.01

    bsm = qk.black_scholes_merton_price(spot, strike, t, vol, r, q, QK_CALL)
    cm = qk.carr_madan_fft_price(spot, strike, t, vol, r, q, QK_CALL, grid_size=4096, eta=0.25, alpha=1.5)
    cos = qk.cos_method_fang_oosterlee_price(spot, strike, t, vol, r, q, QK_CALL, n_terms=256, truncation_width=10.0)
    frft = qk.fractional_fft_price(spot, strike, t, vol, r, q, QK_CALL, grid_size=256, eta=0.25, lambda_=0.05, alpha=1.5)
    lewis = qk.lewis_fourier_inversion_price(spot, strike, t, vol, r, q, QK_CALL, integration_steps=4096, integration_limit=300.0)
    hilbert = qk.hilbert_transform_price(spot, strike, t, vol, r, q, QK_CALL, integration_steps=4096, integration_limit=300.0)

    vals = [cm, cos, frft, lewis, hilbert]
    assert all(math.isfinite(v) and v > 0.0 for v in vals)
    assert abs(cm - bsm) < 1e-2
    assert abs(cos - bsm) < 5e-6
    assert abs(frft - bsm) < 2e-2
    assert abs(lewis - bsm) < 1e-3
    assert abs(hilbert - bsm) < 1e-3


def test_hilbert_put_call_parity(qk):
    s = 105.0
    k = 100.0
    t = 0.8
    vol = 0.25
    r = 0.02
    q = 0.01

    call = qk.hilbert_transform_price(s, k, t, vol, r, q, QK_CALL)
    put = qk.hilbert_transform_price(s, k, t, vol, r, q, QK_PUT)
    rhs = s * math.exp(-q * t) - k * math.exp(-r * t)
    assert abs((call - put) - rhs) < 5e-3

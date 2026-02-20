"""Finite-difference scalar API checks."""

from __future__ import annotations

import math

from quantkernel import QK_CALL, QK_PUT


def test_fdm_scalar_methods_are_callable_and_finite(qk):
    spot = 100.0
    strike = 100.0
    t = 1.0
    vol = 0.2
    r = 0.03
    q = 0.01

    explicit = qk.explicit_fd_price(spot, strike, t, vol, r, q, QK_CALL, 120, 120, False)
    implicit = qk.implicit_fd_price(spot, strike, t, vol, r, q, QK_CALL, 120, 120, False)
    crank = qk.crank_nicolson_price(spot, strike, t, vol, r, q, QK_CALL, 120, 120, False)
    psor = qk.psor_price(spot, strike, t, vol, r, q, QK_PUT, 120, 120, 1.2, 1e-8, 10000)

    vals = [explicit, implicit, crank, psor]
    assert all(math.isfinite(v) and v >= 0.0 for v in vals)


def test_fdm_adi_methods_are_callable_and_finite(qk):
    args = dict(
        spot=100.0,
        strike=100.0,
        t=1.0,
        r=0.02,
        q=0.01,
        v0=0.04,
        kappa=2.0,
        theta_v=0.04,
        sigma=0.5,
        rho=-0.5,
        option_type=QK_CALL,
        s_steps=30,
        v_steps=20,
        time_steps=30,
    )

    douglas = qk.adi_douglas_price(**args)
    craig_sneyd = qk.adi_craig_sneyd_price(**args)
    hundsdorfer_verwer = qk.adi_hundsdorfer_verwer_price(**args)

    vals = [douglas, craig_sneyd, hundsdorfer_verwer]
    assert all(math.isfinite(v) for v in vals)

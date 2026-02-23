"""Tests for Heston MC likelihood-ratio delta estimator.

Validates:
1. Finite-difference bump comparison — LR delta should match central FD delta.
2. Seed reproducibility — same seed produces bit-identical results.
3. Batch consistency — batch API matches scalar calls.
4. Boundedness — delta is in [-1, 1] range.
"""

import math

import pytest

from quantkernel import QK_CALL, QK_PUT


_HESTON = dict(
    strike=100.0, t=1.0, r=0.05, q=0.02,
    v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7,
    paths=40000, steps=80, seed=42, weight_clip=6.0,
)


class TestFiniteDifferenceBump:
    """LR delta should approximate the central finite-difference bump.

    FD delta = (price(spot+eps) - price(spot-eps)) / (2*eps)

    We use generous tolerances because both LR and FD estimates carry MC noise.
    """

    def test_call_delta_matches_fd(self, qk):
        spot = 100.0
        eps = 0.5
        lr = qk.heston_lr_delta(spot=spot, option_type=QK_CALL, **_HESTON)
        p_up = qk.heston_monte_carlo_price(
            spot=spot + eps, option_type=QK_CALL,
            strike=_HESTON["strike"], t=_HESTON["t"], r=_HESTON["r"], q=_HESTON["q"],
            v0=_HESTON["v0"], kappa=_HESTON["kappa"], theta=_HESTON["theta"],
            sigma=_HESTON["sigma"], rho=_HESTON["rho"],
            paths=_HESTON["paths"], steps=_HESTON["steps"], seed=_HESTON["seed"],
        )
        p_dn = qk.heston_monte_carlo_price(
            spot=spot - eps, option_type=QK_CALL,
            strike=_HESTON["strike"], t=_HESTON["t"], r=_HESTON["r"], q=_HESTON["q"],
            v0=_HESTON["v0"], kappa=_HESTON["kappa"], theta=_HESTON["theta"],
            sigma=_HESTON["sigma"], rho=_HESTON["rho"],
            paths=_HESTON["paths"], steps=_HESTON["steps"], seed=_HESTON["seed"],
        )
        fd = (p_up - p_dn) / (2 * eps)
        assert math.isfinite(lr) and math.isfinite(fd)
        assert abs(lr - fd) < 0.15, (
            f"Call LR delta={lr:.6f} vs FD delta={fd:.6f}, diff={abs(lr-fd):.6f}"
        )

    def test_put_delta_matches_fd(self, qk):
        spot = 100.0
        eps = 0.5
        lr = qk.heston_lr_delta(spot=spot, option_type=QK_PUT, **_HESTON)
        p_up = qk.heston_monte_carlo_price(
            spot=spot + eps, option_type=QK_PUT,
            strike=_HESTON["strike"], t=_HESTON["t"], r=_HESTON["r"], q=_HESTON["q"],
            v0=_HESTON["v0"], kappa=_HESTON["kappa"], theta=_HESTON["theta"],
            sigma=_HESTON["sigma"], rho=_HESTON["rho"],
            paths=_HESTON["paths"], steps=_HESTON["steps"], seed=_HESTON["seed"],
        )
        p_dn = qk.heston_monte_carlo_price(
            spot=spot - eps, option_type=QK_PUT,
            strike=_HESTON["strike"], t=_HESTON["t"], r=_HESTON["r"], q=_HESTON["q"],
            v0=_HESTON["v0"], kappa=_HESTON["kappa"], theta=_HESTON["theta"],
            sigma=_HESTON["sigma"], rho=_HESTON["rho"],
            paths=_HESTON["paths"], steps=_HESTON["steps"], seed=_HESTON["seed"],
        )
        fd = (p_up - p_dn) / (2 * eps)
        assert math.isfinite(lr) and math.isfinite(fd)
        assert abs(lr - fd) < 0.15, (
            f"Put LR delta={lr:.6f} vs FD delta={fd:.6f}, diff={abs(lr-fd):.6f}"
        )


class TestSeedReproducibility:
    """Same seed must produce bit-identical results."""

    def test_deterministic_seed(self, qk):
        params = dict(spot=100.0, option_type=QK_CALL, **{**_HESTON, "seed": 777})
        d1 = qk.heston_lr_delta(**params)
        d2 = qk.heston_lr_delta(**params)
        assert d1 == d2, f"Non-deterministic: {d1} != {d2}"

    def test_different_seeds_differ(self, qk):
        base = dict(spot=100.0, option_type=QK_CALL, **{k: v for k, v in _HESTON.items() if k != "seed"})
        d1 = qk.heston_lr_delta(**base, seed=42)
        d2 = qk.heston_lr_delta(**base, seed=999)
        assert d1 != d2


class TestBatchMatchesScalar:
    """Batch API must produce identical results to scalar calls."""

    def test_batch_consistency(self, qk):
        import numpy as np
        params_list = [
            dict(spot=100.0, option_type=QK_CALL, **_HESTON),
            dict(spot=110.0, option_type=QK_PUT, **{**_HESTON, "seed": 99}),
        ]
        scalars = [qk.heston_lr_delta(**p) for p in params_list]
        batch = qk.heston_lr_delta_batch(
            **{k: [p[k] for p in params_list] for k in params_list[0]}
        )
        for i, s in enumerate(scalars):
            assert abs(batch[i] - s) < 1e-12, f"batch[{i}]={batch[i]}, scalar={s}"


class TestDeltaBoundedness:
    """Delta should be bounded within reasonable range."""

    @pytest.mark.parametrize("spot", [80.0, 100.0, 120.0])
    def test_call_delta_in_range(self, qk, spot):
        d = qk.heston_lr_delta(spot=spot, option_type=QK_CALL, **_HESTON)
        assert math.isfinite(d)
        # Call delta should be in [0, 1] (allow small MC noise)
        assert -0.1 <= d <= 1.1, f"Call delta out of range: {d:.6f}"

    @pytest.mark.parametrize("spot", [80.0, 100.0, 120.0])
    def test_put_delta_in_range(self, qk, spot):
        d = qk.heston_lr_delta(spot=spot, option_type=QK_PUT, **_HESTON)
        assert math.isfinite(d)
        # Put delta should be in [-1, 0] (allow small MC noise)
        assert -1.1 <= d <= 0.1, f"Put delta out of range: {d:.6f}"

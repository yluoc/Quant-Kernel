"""Tests for local volatility Monte Carlo pricer.

Validates:
1. Constant vol matches BSM Euler MC (within tolerance).
2. Boundedness — call and put prices within arbitrage bounds.
3. Monotonicity — call increasing in spot, put decreasing in spot.
4. Deterministic seed reproducibility.
5. Batch matches scalar.
"""

import math

import pytest

from quantkernel import QK_CALL, QK_PUT


_COMMON = dict(
    spot=100.0, strike=100.0, t=1.0, vol=0.2, r=0.03, q=0.01,
    option_type=QK_CALL, paths=20000, steps=64, seed=42,
)


class TestConstantVolMatchesBSMEuler:
    """With constant sigma_fn, local vol MC must match BSM Euler MC.

    Both use Euler discretization with constant drift and diffusion.
    With the same seed, paths, and steps, results should be very close.
    """

    def test_call_matches_euler(self, qk):
        lv = qk.local_vol_monte_carlo_price(**_COMMON)
        em = qk.euler_maruyama_price(**_COMMON)
        assert math.isfinite(lv) and math.isfinite(em)
        assert abs(lv - em) < 0.5, (
            f"Local vol call ({lv:.4f}) too far from Euler ({em:.4f})"
        )

    def test_put_matches_euler(self, qk):
        params = {**_COMMON, "option_type": QK_PUT}
        lv = qk.local_vol_monte_carlo_price(**params)
        em = qk.euler_maruyama_price(**params)
        assert math.isfinite(lv) and math.isfinite(em)
        assert abs(lv - em) < 0.5, (
            f"Local vol put ({lv:.4f}) too far from Euler ({em:.4f})"
        )

    def test_consistent_with_bsm_closed_form(self, qk):
        """Local vol MC should be close to BSM analytical price."""
        bsm = qk.black_scholes_merton_price(
            spot=100.0, strike=100.0, t=1.0, vol=0.2,
            r=0.03, q=0.01, option_type=QK_CALL
        )
        lv = qk.local_vol_monte_carlo_price(
            **{**_COMMON, "paths": 50000}
        )
        assert abs(lv - bsm) < 1.0, (
            f"Local vol ({lv:.4f}) too far from BSM ({bsm:.4f})"
        )


class TestBoundedness:
    """European option prices must satisfy no-arbitrage bounds."""

    @pytest.mark.parametrize("spot", [80.0, 100.0, 120.0])
    def test_call_bounded(self, qk, spot):
        price = qk.local_vol_monte_carlo_price(**{**_COMMON, "spot": spot})
        assert 0.0 <= price <= spot

    @pytest.mark.parametrize("spot", [80.0, 100.0, 120.0])
    def test_put_bounded(self, qk, spot):
        upper = _COMMON["strike"] * math.exp(-_COMMON["r"] * _COMMON["t"])
        price = qk.local_vol_monte_carlo_price(
            **{**_COMMON, "spot": spot, "option_type": QK_PUT}
        )
        assert 0.0 <= price <= upper


class TestMonotonicity:
    """Calls non-decreasing in spot; puts non-increasing in spot."""

    _spots = [80.0, 90.0, 100.0, 110.0, 120.0]

    def test_call_monotone(self, qk):
        prices = [
            qk.local_vol_monte_carlo_price(**{**_COMMON, "spot": s})
            for s in self._spots
        ]
        for i in range(len(prices) - 1):
            assert prices[i + 1] >= prices[i] - 0.5

    def test_put_monotone(self, qk):
        prices = [
            qk.local_vol_monte_carlo_price(
                **{**_COMMON, "spot": s, "option_type": QK_PUT}
            )
            for s in self._spots
        ]
        for i in range(len(prices) - 1):
            assert prices[i + 1] <= prices[i] + 0.5


class TestSeedDeterminism:
    """Same seed must produce bit-identical results."""

    def test_deterministic(self, qk):
        p1 = qk.local_vol_monte_carlo_price(**{**_COMMON, "seed": 777})
        p2 = qk.local_vol_monte_carlo_price(**{**_COMMON, "seed": 777})
        assert p1 == p2


class TestBatchMatchesScalar:
    """Batch API must produce identical results to scalar calls."""

    def test_batch_consistency(self, qk):
        params_list = [
            _COMMON,
            {**_COMMON, "spot": 110.0, "option_type": QK_PUT, "seed": 99},
        ]
        scalars = [qk.local_vol_monte_carlo_price(**p) for p in params_list]
        batch = qk.local_vol_monte_carlo_price_batch(
            **{k: [p[k] for p in params_list] for k in params_list[0]}
        )
        for i, s in enumerate(scalars):
            assert abs(batch[i] - s) < 1e-12, f"batch[{i}]={batch[i]}, scalar={s}"

"""Heston MC invariant tests — model-free correctness checks.

These tests verify structural properties of the Heston Monte Carlo pricer
that must hold regardless of model parameters:

1. Boundedness  — option prices are clamped within arbitrage-free bounds.
2. Monotonicity — call price increases in spot; put price decreases in spot.
3. Volatility   — higher vol-of-vol (sigma) should not decrease option value.
4. Finiteness   — output is always a finite real number (no NaN/inf).
"""

import math

import pytest

from quantkernel import QK_CALL, QK_PUT


# ---------------------------------------------------------------------------
# Shared Heston parameters (moderate, numerically stable regime)
# ---------------------------------------------------------------------------
_BASE = dict(
    strike=100.0, t=1.0, r=0.05, q=0.02,
    v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7,
    paths=20000, steps=80, seed=314,
)


# ===================================================================
# 1. Boundedness
# ===================================================================

class TestBoundedness:
    """European option prices must satisfy no-arbitrage bounds.

    Call: 0 <= C <= spot
    Put:  0 <= P <= strike * exp(-r * T)
    """

    @pytest.mark.parametrize("spot", [80.0, 100.0, 120.0])
    def test_call_bounded(self, qk, spot):
        price = qk.heston_monte_carlo_price(spot=spot, option_type=QK_CALL, **_BASE)
        assert 0.0 <= price <= spot, (
            f"Call price {price:.6f} out of bounds [0, {spot}]"
        )

    @pytest.mark.parametrize("spot", [80.0, 100.0, 120.0])
    def test_put_bounded(self, qk, spot):
        upper = _BASE["strike"] * math.exp(-_BASE["r"] * _BASE["t"])
        price = qk.heston_monte_carlo_price(spot=spot, option_type=QK_PUT, **_BASE)
        assert 0.0 <= price <= upper, (
            f"Put price {price:.6f} out of bounds [0, {upper:.6f}]"
        )


# ===================================================================
# 2. Monotonicity in Spot
# ===================================================================

class TestMonotonicity:
    """Calls are non-decreasing in spot; puts are non-increasing in spot.

    We use a fixed seed and generous margin to absorb MC noise.
    """

    _spots = [80.0, 90.0, 100.0, 110.0, 120.0]

    def test_call_monotone_increasing_in_spot(self, qk):
        prices = [
            qk.heston_monte_carlo_price(spot=s, option_type=QK_CALL, **_BASE)
            for s in self._spots
        ]
        for i in range(len(prices) - 1):
            assert prices[i + 1] >= prices[i] - 0.5, (
                f"Call monotonicity violated: C(spot={self._spots[i+1]})={prices[i+1]:.4f}"
                f" < C(spot={self._spots[i]})={prices[i]:.4f}"
            )

    def test_put_monotone_decreasing_in_spot(self, qk):
        prices = [
            qk.heston_monte_carlo_price(spot=s, option_type=QK_PUT, **_BASE)
            for s in self._spots
        ]
        for i in range(len(prices) - 1):
            assert prices[i + 1] <= prices[i] + 0.5, (
                f"Put monotonicity violated: P(spot={self._spots[i+1]})={prices[i+1]:.4f}"
                f" > P(spot={self._spots[i]})={prices[i]:.4f}"
            )


# ===================================================================
# 3. Volatility Effect (vol-of-vol)
# ===================================================================

class TestVolatilityEffect:
    """Increasing vol-of-vol (sigma) should generally not decrease ATM option value.

    Under Heston, higher sigma introduces more variance in the variance process,
    which tends to fatten tails and increase option prices (especially ATM).
    We allow a small MC-noise tolerance.
    """

    def test_call_price_nondecreasing_in_sigma(self, qk):
        base = dict(_BASE, spot=100.0, option_type=QK_CALL)
        price_low = qk.heston_monte_carlo_price(**{**base, "sigma": 0.2})
        price_high = qk.heston_monte_carlo_price(**{**base, "sigma": 0.5})
        assert price_high >= price_low - 0.5, (
            f"Call price decreased when sigma rose: "
            f"sigma=0.2 -> {price_low:.4f}, sigma=0.5 -> {price_high:.4f}"
        )

    def test_put_price_nondecreasing_in_sigma(self, qk):
        base = dict(_BASE, spot=100.0, option_type=QK_PUT)
        price_low = qk.heston_monte_carlo_price(**{**base, "sigma": 0.2})
        price_high = qk.heston_monte_carlo_price(**{**base, "sigma": 0.5})
        assert price_high >= price_low - 0.5, (
            f"Put price decreased when sigma rose: "
            f"sigma=0.2 -> {price_low:.4f}, sigma=0.5 -> {price_high:.4f}"
        )


# ===================================================================
# 4. Finite Output
# ===================================================================

class TestFiniteOutput:
    """All outputs must be finite (not NaN, not inf) for valid inputs."""

    @pytest.mark.parametrize("spot", [50.0, 100.0, 200.0])
    @pytest.mark.parametrize("option_type", [QK_CALL, QK_PUT])
    def test_finite(self, qk, spot, option_type):
        price = qk.heston_monte_carlo_price(
            spot=spot, option_type=option_type, **_BASE
        )
        assert math.isfinite(price), (
            f"Non-finite output: spot={spot}, type={option_type}, price={price}"
        )

    def test_extreme_itm_call(self, qk):
        """Deep in-the-money call (spot >> strike)."""
        price = qk.heston_monte_carlo_price(
            spot=300.0, option_type=QK_CALL, **_BASE
        )
        assert math.isfinite(price) and price > 0.0

    def test_extreme_otm_put(self, qk):
        """Deep out-of-the-money put (spot >> strike)."""
        price = qk.heston_monte_carlo_price(
            spot=300.0, option_type=QK_PUT, **_BASE
        )
        assert math.isfinite(price) and price >= 0.0

    def test_near_zero_maturity(self, qk):
        """Near-zero maturity should return intrinsic value."""
        call = qk.heston_monte_carlo_price(
            spot=110.0, option_type=QK_CALL,
            **{**_BASE, "t": 1e-14}
        )
        assert math.isfinite(call)
        # Intrinsic = max(110 - 100, 0) = 10
        assert abs(call - 10.0) < 0.01

    @pytest.mark.parametrize("rho", [-0.99, 0.0, 0.99])
    def test_extreme_correlation(self, qk, rho):
        """Extreme correlation values should still produce finite output."""
        price = qk.heston_monte_carlo_price(
            spot=100.0, option_type=QK_CALL,
            **{**_BASE, "rho": rho}
        )
        assert math.isfinite(price) and price > 0.0

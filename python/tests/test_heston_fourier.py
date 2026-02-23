"""Tests for Heston Fourier pricing methods.

Validates:
1. Consistency with Heston CF semi-analytical pricer (within tolerance).
2. Boundedness — prices within no-arbitrage bounds.
3. Call–put parity consistency across Fourier methods.
4. Batch matches scalar.
5. Finite output for reasonable inputs.
"""

import math

import pytest

from quantkernel import QK_CALL, QK_PUT


_HESTON = dict(
    spot=100.0, strike=100.0, t=1.0, r=0.03, q=0.01,
    v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7,
)


class TestConsistencyWithHestonCF:
    """Heston Fourier methods should agree with Heston CF (semi-analytical) within tolerance."""

    _tol = 1.5  # Fourier methods may differ slightly due to grid/truncation

    def _cf_price(self, qk, option_type):
        return qk.heston_price_cf(
            **_HESTON, option_type=option_type,
            integration_steps=8192, integration_limit=500.0,
        )

    @pytest.mark.parametrize("option_type", [QK_CALL, QK_PUT])
    def test_carr_madan_fft_vs_cf(self, qk, option_type):
        cf = self._cf_price(qk, option_type)
        ftm = qk.carr_madan_fft_heston_price(**_HESTON, option_type=option_type)
        assert math.isfinite(ftm), f"Carr-Madan FFT Heston returned {ftm}"
        assert abs(ftm - cf) < self._tol, f"CM-FFT={ftm:.4f} vs CF={cf:.4f}"

    @pytest.mark.parametrize("option_type", [QK_CALL, QK_PUT])
    def test_cos_method_vs_cf(self, qk, option_type):
        cf = self._cf_price(qk, option_type)
        ftm = qk.cos_method_fang_oosterlee_heston_price(**_HESTON, option_type=option_type)
        assert math.isfinite(ftm), f"COS Heston returned {ftm}"
        assert abs(ftm - cf) < self._tol, f"COS={ftm:.4f} vs CF={cf:.4f}"

    @pytest.mark.parametrize("option_type", [QK_CALL, QK_PUT])
    def test_fractional_fft_vs_cf(self, qk, option_type):
        cf = self._cf_price(qk, option_type)
        ftm = qk.fractional_fft_heston_price(**_HESTON, option_type=option_type)
        assert math.isfinite(ftm), f"Fractional FFT Heston returned {ftm}"
        assert abs(ftm - cf) < self._tol, f"FracFFT={ftm:.4f} vs CF={cf:.4f}"

    @pytest.mark.parametrize("option_type", [QK_CALL, QK_PUT])
    def test_lewis_vs_cf(self, qk, option_type):
        cf = self._cf_price(qk, option_type)
        ftm = qk.lewis_fourier_inversion_heston_price(**_HESTON, option_type=option_type)
        assert math.isfinite(ftm), f"Lewis Heston returned {ftm}"
        assert abs(ftm - cf) < self._tol, f"Lewis={ftm:.4f} vs CF={cf:.4f}"

    @pytest.mark.parametrize("option_type", [QK_CALL, QK_PUT])
    def test_hilbert_vs_cf(self, qk, option_type):
        cf = self._cf_price(qk, option_type)
        ftm = qk.hilbert_transform_heston_price(**_HESTON, option_type=option_type)
        assert math.isfinite(ftm), f"Hilbert Heston returned {ftm}"
        assert abs(ftm - cf) < self._tol, f"Hilbert={ftm:.4f} vs CF={cf:.4f}"


class TestBoundedness:
    """European option prices must satisfy no-arbitrage bounds."""

    _methods = [
        "carr_madan_fft_heston_price",
        "cos_method_fang_oosterlee_heston_price",
        "fractional_fft_heston_price",
        "lewis_fourier_inversion_heston_price",
        "hilbert_transform_heston_price",
    ]

    @pytest.mark.parametrize("method", _methods)
    def test_call_bounded(self, qk, method):
        price = getattr(qk, method)(**_HESTON, option_type=QK_CALL)
        assert 0.0 <= price <= _HESTON["spot"]

    @pytest.mark.parametrize("method", _methods)
    def test_put_bounded(self, qk, method):
        upper = _HESTON["strike"] * math.exp(-_HESTON["r"] * _HESTON["t"])
        price = getattr(qk, method)(**_HESTON, option_type=QK_PUT)
        assert 0.0 <= price <= upper


class TestPutCallParity:
    """All Fourier methods should satisfy C - P = S*exp(-q*T) - K*exp(-r*T)."""

    _tol = 1.5

    _methods = [
        "carr_madan_fft_heston_price",
        "cos_method_fang_oosterlee_heston_price",
        "fractional_fft_heston_price",
        "lewis_fourier_inversion_heston_price",
        "hilbert_transform_heston_price",
    ]

    @pytest.mark.parametrize("method", _methods)
    def test_put_call_parity(self, qk, method):
        fn = getattr(qk, method)
        call = fn(**_HESTON, option_type=QK_CALL)
        put = fn(**_HESTON, option_type=QK_PUT)
        forward_diff = (
            _HESTON["spot"] * math.exp(-_HESTON["q"] * _HESTON["t"])
            - _HESTON["strike"] * math.exp(-_HESTON["r"] * _HESTON["t"])
        )
        assert abs((call - put) - forward_diff) < self._tol, (
            f"{method}: C-P={call - put:.4f}, expected={forward_diff:.4f}"
        )


class TestBatchMatchesScalar:
    """Batch API must produce identical results to scalar calls."""

    def test_carr_madan_fft_batch(self, qk):
        params_list = [
            {**_HESTON, "option_type": QK_CALL},
            {**_HESTON, "option_type": QK_PUT, "spot": 110.0},
        ]
        scalars = [qk.carr_madan_fft_heston_price(**p, grid_size=4096, eta=0.25, alpha=1.5)
                   for p in params_list]
        batch = qk.carr_madan_fft_heston_price_batch(
            **{k: [p[k] for p in params_list] for k in params_list[0]},
            grid_size=[4096, 4096], eta=[0.25, 0.25], alpha=[1.5, 1.5]
        )
        for i, s in enumerate(scalars):
            assert abs(batch[i] - s) < 1e-12, f"batch[{i}]={batch[i]}, scalar={s}"

    def test_cos_method_batch(self, qk):
        params_list = [
            {**_HESTON, "option_type": QK_CALL},
            {**_HESTON, "option_type": QK_PUT, "spot": 110.0},
        ]
        scalars = [qk.cos_method_fang_oosterlee_heston_price(**p, n_terms=256, truncation_width=10.0)
                   for p in params_list]
        batch = qk.cos_method_fang_oosterlee_heston_price_batch(
            **{k: [p[k] for p in params_list] for k in params_list[0]},
            n_terms=[256, 256], truncation_width=[10.0, 10.0]
        )
        for i, s in enumerate(scalars):
            assert abs(batch[i] - s) < 1e-12, f"batch[{i}]={batch[i]}, scalar={s}"

    def test_lewis_batch(self, qk):
        params_list = [
            {**_HESTON, "option_type": QK_CALL},
            {**_HESTON, "option_type": QK_PUT, "spot": 110.0},
        ]
        scalars = [qk.lewis_fourier_inversion_heston_price(**p, integration_steps=4096, integration_limit=300.0)
                   for p in params_list]
        batch = qk.lewis_fourier_inversion_heston_price_batch(
            **{k: [p[k] for p in params_list] for k in params_list[0]},
            integration_steps=[4096, 4096], integration_limit=[300.0, 300.0]
        )
        for i, s in enumerate(scalars):
            assert abs(batch[i] - s) < 1e-12, f"batch[{i}]={batch[i]}, scalar={s}"

    def test_hilbert_batch(self, qk):
        params_list = [
            {**_HESTON, "option_type": QK_CALL},
            {**_HESTON, "option_type": QK_PUT, "spot": 110.0},
        ]
        scalars = [qk.hilbert_transform_heston_price(**p, integration_steps=4096, integration_limit=300.0)
                   for p in params_list]
        batch = qk.hilbert_transform_heston_price_batch(
            **{k: [p[k] for p in params_list] for k in params_list[0]},
            integration_steps=[4096, 4096], integration_limit=[300.0, 300.0]
        )
        for i, s in enumerate(scalars):
            assert abs(batch[i] - s) < 1e-12, f"batch[{i}]={batch[i]}, scalar={s}"

    def test_fractional_fft_batch(self, qk):
        params_list = [
            {**_HESTON, "option_type": QK_CALL},
            {**_HESTON, "option_type": QK_PUT, "spot": 110.0},
        ]
        scalars = [qk.fractional_fft_heston_price(**p, grid_size=256, eta=0.25, lambda_=0.05, alpha=1.5)
                   for p in params_list]
        batch = qk.fractional_fft_heston_price_batch(
            **{k: [p[k] for p in params_list] for k in params_list[0]},
            grid_size=[256, 256], eta=[0.25, 0.25], lambda_=[0.05, 0.05], alpha=[1.5, 1.5]
        )
        for i, s in enumerate(scalars):
            assert abs(batch[i] - s) < 1e-12, f"batch[{i}]={batch[i]}, scalar={s}"


class TestFiniteOutput:
    """All methods return finite prices for reasonable Heston inputs."""

    _methods = [
        "carr_madan_fft_heston_price",
        "cos_method_fang_oosterlee_heston_price",
        "fractional_fft_heston_price",
        "lewis_fourier_inversion_heston_price",
        "hilbert_transform_heston_price",
    ]

    @pytest.mark.parametrize("method", _methods)
    @pytest.mark.parametrize("option_type", [QK_CALL, QK_PUT])
    def test_finite(self, qk, method, option_type):
        price = getattr(qk, method)(**_HESTON, option_type=option_type)
        assert math.isfinite(price), f"{method} returned {price}"

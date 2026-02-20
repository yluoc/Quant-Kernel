"""Rule-based accelerator behavior checks."""

import numpy as np
import pytest

from quantkernel import QK_CALL, QK_PUT, QuantAccelerator


def test_bsm_batch_matches_scalar_qk(qk):
    accel = QuantAccelerator(qk=qk, backend="cpu")
    jobs = [
        {"spot": 95.0, "strike": 100.0, "t": 0.5, "vol": 0.2, "r": 0.02, "q": 0.01, "option_type": QK_CALL},
        {"spot": 100.0, "strike": 100.0, "t": 1.0, "vol": 0.25, "r": 0.03, "q": 0.00, "option_type": QK_CALL},
        {"spot": 120.0, "strike": 110.0, "t": 2.0, "vol": 0.18, "r": 0.01, "q": 0.02, "option_type": QK_CALL},
    ]

    batch = accel.price_batch("black_scholes_merton_price", jobs)
    scalar = np.array([qk.black_scholes_merton_price(**j) for j in jobs], dtype=np.float64)

    assert np.allclose(batch, scalar, atol=1e-6, rtol=1e-6)


def test_native_batch_strategy_preferred_on_cpu(qk):
    """On CPU, suggest_strategy should return native_batch for methods with C++ batch support."""
    accel = QuantAccelerator(qk=qk, backend="cpu")
    assert accel.suggest_strategy("heston_price_cf", 16) == "native_batch"
    assert accel.suggest_strategy("heston_price_cf", 128) == "native_batch"
    assert accel.suggest_strategy("black_scholes_merton_price", 1) == "native_batch"
    assert accel.suggest_strategy("crr_price", 1000) == "native_batch"


def test_heston_batch_matches_scalar_qk(qk):
    accel = QuantAccelerator(qk=qk, backend="cpu", max_workers=4)
    jobs = [
        {
            "spot": 100.0,
            "strike": 100.0,
            "t": 1.0,
            "r": 0.02,
            "q": 0.01,
            "v0": 0.04,
            "kappa": 2.0,
            "theta": 0.04,
            "sigma": 0.5,
            "rho": -0.5,
            "option_type": QK_CALL,
            "integration_steps": 512,
            "integration_limit": 100.0,
        },
        {
            "spot": 95.0,
            "strike": 100.0,
            "t": 0.75,
            "r": 0.015,
            "q": 0.005,
            "v0": 0.05,
            "kappa": 1.8,
            "theta": 0.04,
            "sigma": 0.45,
            "rho": -0.3,
            "option_type": QK_CALL,
            "integration_steps": 512,
            "integration_limit": 100.0,
        },
    ]

    batch = accel.price_batch("heston_price_cf", jobs)
    scalar = np.array([qk.heston_price_cf(**j) for j in jobs], dtype=np.float64)

    assert np.allclose(batch, scalar, atol=1e-10, rtol=1e-10)


def test_heston_cpu_vectorized_matches_scalar(qk):
    """Heston CuPy/NumPy vectorized path should closely match C++ scalar."""
    accel = QuantAccelerator(qk=qk, backend="cpu")
    jobs = [
        {
            "spot": 100.0, "strike": 100.0, "t": 1.0, "r": 0.02, "q": 0.01,
            "v0": 0.04, "kappa": 2.0, "theta": 0.04, "sigma": 0.5, "rho": -0.5,
            "option_type": QK_CALL, "integration_steps": 2048, "integration_limit": 120.0,
        },
        {
            "spot": 95.0, "strike": 100.0, "t": 0.75, "r": 0.015, "q": 0.005,
            "v0": 0.05, "kappa": 1.8, "theta": 0.04, "sigma": 0.45, "rho": -0.3,
            "option_type": QK_PUT, "integration_steps": 2048, "integration_limit": 120.0,
        },
    ]
    out = accel._vectorized_price("heston_price_cf", jobs, use_gpu=False)
    scalar = np.array([qk.heston_price_cf(**j) for j in jobs], dtype=np.float64)
    assert np.allclose(out, scalar, atol=0.01, rtol=0.01)


def test_variance_gamma_cpu_vectorized_matches_scalar(qk):
    """Variance Gamma CuPy/NumPy vectorized path should closely match C++ scalar."""
    accel = QuantAccelerator(qk=qk, backend="cpu")
    jobs = [
        {
            "spot": 100.0, "strike": 100.0, "t": 1.0, "r": 0.03, "q": 0.01,
            "sigma": 0.2, "theta": -0.1, "nu": 0.2, "option_type": QK_CALL,
            "integration_steps": 2048, "integration_limit": 120.0,
        },
        {
            "spot": 95.0, "strike": 100.0, "t": 0.5, "r": 0.02, "q": 0.0,
            "sigma": 0.25, "theta": -0.15, "nu": 0.3, "option_type": QK_PUT,
            "integration_steps": 2048, "integration_limit": 120.0,
        },
    ]
    out = accel._vectorized_price("variance_gamma_price_cf", jobs, use_gpu=False)
    scalar = np.array([qk.variance_gamma_price_cf(**j) for j in jobs], dtype=np.float64)
    assert np.allclose(out, scalar, atol=0.05, rtol=0.05)


def test_unknown_method_raises(qk):
    accel = QuantAccelerator(qk=qk)
    try:
        accel.price_batch("nope", [{"x": 1.0}])
        assert False, "Expected AttributeError"
    except AttributeError:
        assert True


def test_quantkernel_price_batch_convenience_matches_accelerator(qk):
    jobs = [
        {"spot": 100.0, "strike": 100.0, "t": 1.0, "vol": 0.2, "r": 0.03, "q": 0.01, "option_type": QK_CALL},
        {"spot": 101.0, "strike": 100.0, "t": 1.0, "vol": 0.21, "r": 0.03, "q": 0.01, "option_type": QK_CALL},
    ]

    out_qk = qk.price_batch("black_scholes_merton_price", jobs, backend="cpu")
    out_acc = qk.get_accelerator(backend="cpu").price_batch("black_scholes_merton_price", jobs)
    assert np.allclose(out_qk, out_acc, atol=1e-12, rtol=1e-12)


def test_quantkernel_accelerator_cache_keyed_by_backend_and_workers(qk):
    a1 = qk.get_accelerator(backend="cpu", max_workers=2)
    a2 = qk.get_accelerator(backend="cpu", max_workers=2)
    a3 = qk.get_accelerator(backend="auto", max_workers=2)
    assert a1 is a2
    assert a1 is not a3


def test_fourier_batch_matches_scalar_qk(qk):
    accel = QuantAccelerator(qk=qk, backend="cpu")
    jobs = [
        {
            "spot": 95.0,
            "strike": 100.0,
            "t": 0.5,
            "vol": 0.22,
            "r": 0.02,
            "q": 0.01,
            "option_type": QK_CALL,
            "n_terms": 256,
            "truncation_width": 10.0,
        },
        {
            "spot": 100.0,
            "strike": 100.0,
            "t": 1.0,
            "vol": 0.25,
            "r": 0.03,
            "q": 0.00,
            "option_type": QK_CALL,
            "n_terms": 256,
            "truncation_width": 10.0,
        },
        {
            "spot": 120.0,
            "strike": 110.0,
            "t": 2.0,
            "vol": 0.18,
            "r": 0.01,
            "q": 0.02,
            "option_type": QK_CALL,
            "n_terms": 256,
            "truncation_width": 10.0,
        },
    ]

    batch = accel.price_batch("cos_method_fang_oosterlee_price", jobs)
    scalar = np.array([qk.cos_method_fang_oosterlee_price(**j) for j in jobs], dtype=np.float64)
    assert np.allclose(batch, scalar, atol=1e-6, rtol=1e-6)



def test_gpu_backend_raises_without_cupy_for_vectorized_method(qk):
    """backend='gpu' must raise for vectorized methods when CuPy is unavailable."""
    accel = QuantAccelerator(qk=qk, backend="gpu")
    accel._cp = None  # force CuPy unavailable
    jobs = [{"spot": 100.0, "strike": 100.0, "t": 1.0, "vol": 0.2,
             "r": 0.03, "q": 0.01, "option_type": QK_CALL}]
    with pytest.raises(RuntimeError, match="CuPy"):
        accel.price_batch("black_scholes_merton_price", jobs)


def test_gpu_backend_raises_without_cupy_for_native_batch_method(qk):
    """backend='gpu' must raise for native-batch methods too — not silently run CPU."""
    accel = QuantAccelerator(qk=qk, backend="gpu")
    accel._cp = None
    jobs = [{"spot": 100.0, "strike": 100.0, "t": 1.0, "vol": 0.2,
             "r": 0.03, "q": 0.01, "option_type": QK_CALL,
             "time_steps": 50, "spot_steps": 50, "american_style": 0}]
    with pytest.raises(RuntimeError, match="CuPy"):
        accel.price_batch("explicit_fd_price", jobs)


def test_gpu_backend_raises_without_cupy_for_threaded_method(qk):
    """backend='gpu' must raise for heavy threaded methods too."""
    accel = QuantAccelerator(qk=qk, backend="gpu")
    accel._cp = None
    jobs = [{"spot": 100.0, "strike": 100.0, "t": 1.0, "r": 0.02, "q": 0.01,
             "v0": 0.04, "kappa": 2.0, "theta": 0.04, "sigma": 0.5, "rho": -0.5,
             "option_type": QK_CALL, "integration_steps": 256,
             "integration_limit": 80.0}]
    with pytest.raises(RuntimeError, match="CuPy"):
        accel.price_batch("heston_price_cf", jobs)


def test_gpu_backend_raises_without_cupy_for_mlm_method(qk):
    """backend='gpu' must raise for ML methods — no silent CPU fallback."""
    accel = QuantAccelerator(qk=qk, backend="gpu")
    accel._cp = None
    jobs = [{"spot": 100.0, "strike": 100.0, "t": 1.0, "vol": 0.2,
             "r": 0.03, "q": 0.01, "option_type": QK_CALL,
             "time_steps": 10, "hidden_width": 16,
             "training_epochs": 10, "learning_rate": 5e-3}]
    with pytest.raises(RuntimeError, match="CuPy"):
        accel.price_batch("deep_bsde_price", jobs)


def test_gpu_backend_raises_for_non_vectorizable_method(qk):
    """backend='gpu' must raise for non-vectorizable methods even with CuPy available."""
    accel = QuantAccelerator(qk=qk, backend="gpu")
    accel._cp = np
    with pytest.raises(RuntimeError, match="not supported"):
        accel.suggest_strategy("explicit_fd_price", 100)
    with pytest.raises(RuntimeError, match="not supported"):
        accel.suggest_strategy("deep_bsde_price", 100)



def test_native_batch_routing_covers_all_families(qk):
    """Every method with a *_batch counterpart on QuantKernel should be in _NATIVE_BATCH_METHODS."""
    accel = QuantAccelerator(qk=qk, backend="cpu")
    missing = []
    for method_name in dir(qk):
        if method_name.startswith("_") or not method_name.endswith("_batch"):
            continue
        scalar_name = method_name[:-6]
        if not hasattr(qk, scalar_name):
            continue
        if scalar_name not in accel._NATIVE_BATCH_METHODS:
            missing.append(scalar_name)
    assert missing == [], f"Methods with batch but not in _NATIVE_BATCH_METHODS: {missing}"


def test_native_batch_preference_for_closed_form(qk):
    """Closed-form methods should route through native batch on CPU, matching scalar exactly."""
    accel = QuantAccelerator(qk=qk, backend="cpu")
    jobs = [
        {"spot": 100.0, "strike": 100.0, "t": 1.0, "vol": 0.2,
         "r": 0.03, "q": 0.01, "option_type": QK_CALL},
        {"spot": 95.0, "strike": 100.0, "t": 0.5, "vol": 0.25,
         "r": 0.02, "q": 0.0, "option_type": QK_PUT},
    ]
    batch = accel.price_batch("black_scholes_merton_price", jobs)
    scalar = np.array([qk.black_scholes_merton_price(**j) for j in jobs], dtype=np.float64)
    # Native batch should be bit-identical to scalar (no vectorized approximation)
    assert np.allclose(batch, scalar, atol=1e-12, rtol=1e-12)


def test_native_batch_preference_for_tree(qk):
    """Tree methods should route through native batch on CPU."""
    accel = QuantAccelerator(qk=qk, backend="cpu")
    jobs = [
        {"spot": 100.0, "strike": 100.0, "t": 1.0, "vol": 0.2,
         "r": 0.03, "q": 0.01, "option_type": QK_CALL,
         "steps": 128, "american_style": 0},
        {"spot": 95.0, "strike": 100.0, "t": 0.5, "vol": 0.25,
         "r": 0.02, "q": 0.0, "option_type": QK_PUT,
         "steps": 128, "american_style": 1},
    ]
    batch = accel.price_batch("crr_price", jobs)
    scalar = np.array([qk.crr_price(**j) for j in jobs], dtype=np.float64)
    assert np.allclose(batch, scalar, atol=1e-12, rtol=1e-12)


def test_native_batch_preference_for_mc(qk):
    """Monte Carlo methods should route through native batch on CPU."""
    accel = QuantAccelerator(qk=qk, backend="cpu")
    jobs = [
        {"spot": 100.0, "strike": 100.0, "t": 1.0, "vol": 0.2,
         "r": 0.03, "q": 0.01, "option_type": QK_CALL,
         "paths": 1024, "seed": 42},
        {"spot": 95.0, "strike": 100.0, "t": 0.5, "vol": 0.25,
         "r": 0.02, "q": 0.0, "option_type": QK_PUT,
         "paths": 1024, "seed": 99},
    ]
    batch = accel.price_batch("standard_monte_carlo_price", jobs)
    scalar = np.array([qk.standard_monte_carlo_price(**j) for j in jobs], dtype=np.float64)
    assert np.allclose(batch, scalar, atol=1e-12, rtol=1e-12)


def test_native_batch_preference_for_fourier(qk):
    """Fourier methods should route through native batch on CPU."""
    accel = QuantAccelerator(qk=qk, backend="cpu")
    jobs = [
        {"spot": 100.0, "strike": 100.0, "t": 1.0, "vol": 0.2,
         "r": 0.03, "q": 0.01, "option_type": QK_CALL,
         "integration_steps": 1024, "integration_limit": 200.0},
        {"spot": 95.0, "strike": 100.0, "t": 0.5, "vol": 0.25,
         "r": 0.02, "q": 0.0, "option_type": QK_PUT,
         "integration_steps": 1024, "integration_limit": 200.0},
    ]
    batch = accel.price_batch("lewis_fourier_inversion_price", jobs)
    scalar = np.array([qk.lewis_fourier_inversion_price(**j) for j in jobs], dtype=np.float64)
    assert np.allclose(batch, scalar, atol=1e-12, rtol=1e-12)


def test_empty_batch_returns_empty(qk):
    accel = QuantAccelerator(qk=qk, backend="cpu")
    result = accel.price_batch("black_scholes_merton_price", [])
    assert result.shape == (0,)
    assert result.dtype == np.float64



def test_all_vectorized_methods_have_implementation(qk):
    """Every method in _VECTORIZED_METHODS must be handled in _vectorized_price."""
    accel = QuantAccelerator(qk=qk, backend="cpu")
    common = {
        "spot": 100.0, "strike": 100.0, "t": 1.0, "vol": 0.2,
        "r": 0.03, "q": 0.01, "option_type": QK_CALL,
    }
    extra_params = {
        "black_scholes_merton_price": {},
        "black76_price": {"forward": 100.0},
        "bachelier_price": {"forward": 100.0, "normal_vol": 20.0},
        "sabr_hagan_lognormal_iv": {"forward": 100.0, "alpha": 0.2, "beta": 0.5, "rho": -0.3, "nu": 0.4},
        "sabr_hagan_black76_price": {"forward": 100.0, "alpha": 0.2, "beta": 0.5, "rho": -0.3, "nu": 0.4},
        "dupire_local_vol": {"call_price": 10.0, "dC_dT": 5.0, "dC_dK": -0.03, "d2C_dK2": 0.02},
        "merton_jump_diffusion_price": {"jump_intensity": 1.0, "jump_mean": 0.0, "jump_vol": 0.1, "max_terms": 10},
        "heston_price_cf": {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "sigma": 0.5, "rho": -0.5,
                            "integration_steps": 256, "integration_limit": 80.0},
        "variance_gamma_price_cf": {"sigma": 0.2, "theta": -0.1, "nu": 0.2,
                                     "integration_steps": 256, "integration_limit": 80.0},
        "carr_madan_fft_price": {"grid_size": 64, "eta": 0.25, "alpha": 1.5},
        "cos_method_fang_oosterlee_price": {"n_terms": 64, "truncation_width": 10.0},
        "fractional_fft_price": {"grid_size": 64, "eta": 0.25, "lambda_": 0.05, "alpha": 1.5},
        "lewis_fourier_inversion_price": {"integration_steps": 256, "integration_limit": 200.0},
        "hilbert_transform_price": {"integration_steps": 256, "integration_limit": 200.0},
        "standard_monte_carlo_price": {"paths": 1024, "seed": 42},
        "euler_maruyama_price": {"paths": 1024, "steps": 10, "seed": 42},
        "milstein_price": {"paths": 1024, "steps": 10, "seed": 42},
        "importance_sampling_price": {"paths": 1024, "shift": 0.3, "seed": 42},
        "control_variates_price": {"paths": 1024, "seed": 42},
        "antithetic_variates_price": {"paths": 1024, "seed": 42},
        "stratified_sampling_price": {"paths": 1024, "seed": 42},
        "polynomial_chaos_expansion_price": {"polynomial_order": 4, "quadrature_points": 16},
        "radial_basis_function_price": {"centers": 8, "rbf_shape": 1.0, "ridge": 1e-4},
        "sparse_grid_collocation_price": {"level": 3, "nodes_per_dim": 5},
        "proper_orthogonal_decomposition_price": {"modes": 4, "snapshots": 16},
        "pathwise_derivative_delta": {"paths": 1024, "seed": 42},
        "likelihood_ratio_delta": {"paths": 1024, "seed": 42, "weight_clip": 6.0},
        "aad_delta": {"regularization": 1e-6},
        "neural_sde_calibration_price": {"target_implied_vol": 0.2},
    }
    missing = []
    for method in accel._VECTORIZED_METHODS:
        params = dict(common)
        if method in ("black76_price",):
            params.pop("q", None)
        if method in ("bachelier_price", "sabr_hagan_lognormal_iv",
                       "sabr_hagan_black76_price"):
            params.pop("vol", None)
            params.pop("q", None)
        if method == "dupire_local_vol":
            params.pop("spot", None)
            params.pop("vol", None)
            params.pop("option_type", None)
        if method == "sabr_hagan_lognormal_iv":
            params.pop("option_type", None)
            params.pop("r", None)
        if method in extra_params:
            params.update(extra_params[method])
        try:
            accel._vectorized_price(method, [params], use_gpu=False)
        except RuntimeError as e:
            if "not implemented" in str(e).lower():
                missing.append(method)
    assert not missing, f"_VECTORIZED_METHODS without _vectorized_price implementation: {missing}"

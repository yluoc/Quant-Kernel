"""Rule-based accelerator behavior checks."""

import numpy as np

from quantkernel import QK_CALL, QuantAccelerator


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


def test_heston_uses_threaded_strategy_for_large_batch(qk):
    accel = QuantAccelerator(qk=qk, backend="cpu")
    assert accel.suggest_strategy("heston_price_cf", 16) == "sequential"
    assert accel.suggest_strategy("heston_price_cf", 128) == "threaded"


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

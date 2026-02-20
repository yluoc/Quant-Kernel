"""Deterministic performance regression checks for batch paths."""

from __future__ import annotations

import statistics
import time

import numpy as np

from quantkernel import QK_CALL, QK_PUT


def _median_ms(fn, repeats: int = 3) -> float:
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1000.0)
    return statistics.median(samples)


def test_bsm_batch_speed_and_accuracy_regression(qk):
    rng = np.random.default_rng(7)
    n = 40000
    spot = rng.uniform(80.0, 120.0, n).astype(np.float64)
    strike = rng.uniform(80.0, 120.0, n).astype(np.float64)
    tau = rng.uniform(0.25, 2.0, n).astype(np.float64)
    vol = rng.uniform(0.1, 0.6, n).astype(np.float64)
    r = rng.uniform(0.0, 0.08, n).astype(np.float64)
    q = rng.uniform(0.0, 0.04, n).astype(np.float64)
    option_type = np.where((np.arange(n) & 1) == 0, QK_CALL, QK_PUT).astype(np.int32)

    jobs = [
        {
            "spot": float(spot[i]),
            "strike": float(strike[i]),
            "t": float(tau[i]),
            "vol": float(vol[i]),
            "r": float(r[i]),
            "q": float(q[i]),
            "option_type": int(option_type[i]),
        }
        for i in range(n)
    ]

    # Warmup JIT/caches
    qk.black_scholes_merton_price_batch(spot, strike, tau, vol, r, q, option_type)
    qk.price_batch("black_scholes_merton_price", jobs[:2000], backend="cpu")

    scalar_ms = _median_ms(
        lambda: [qk.black_scholes_merton_price(
            float(spot[i]), float(strike[i]), float(tau[i]), float(vol[i]),
            float(r[i]), float(q[i]), int(option_type[i])
        ) for i in range(n)],
    )
    py_batch_ms = _median_ms(
        lambda: qk.price_batch("black_scholes_merton_price", jobs, backend="cpu"),
    )
    native_batch_ms = _median_ms(
        lambda: qk.black_scholes_merton_price_batch(spot, strike, tau, vol, r, q, option_type),
    )

    bsm_batch = qk.black_scholes_merton_price_batch(spot, strike, tau, vol, r, q, option_type)
    bsm_scalar = np.array(
        [qk.black_scholes_merton_price(float(spot[i]), float(strike[i]), float(tau[i]), float(vol[i]),
                                       float(r[i]), float(q[i]), int(option_type[i])) for i in range(n)],
        dtype=np.float64,
    )
    assert np.allclose(bsm_batch, bsm_scalar, atol=1e-12, rtol=1e-12)

    # Deterministic perf guardrails:
    # Fail if batch paths regress to less than 5% faster than scalar.
    assert py_batch_ms < scalar_ms * 0.95
    assert native_batch_ms < scalar_ms * 0.95


def test_fourier_batch_speed_regression(qk):
    """COS method batch must beat scalar loop by >5%."""
    rng = np.random.default_rng(11)
    n = 2000
    spot = rng.uniform(80.0, 120.0, n).astype(np.float64)
    strike = rng.uniform(80.0, 120.0, n).astype(np.float64)
    tau = rng.uniform(0.25, 2.0, n).astype(np.float64)
    vol = rng.uniform(0.1, 0.5, n).astype(np.float64)
    r = rng.uniform(0.0, 0.06, n).astype(np.float64)
    q = rng.uniform(0.0, 0.03, n).astype(np.float64)
    ot = np.where((np.arange(n) & 1) == 0, QK_CALL, QK_PUT).astype(np.int32)
    n_terms = np.full(n, 128, dtype=np.int32)
    tw = np.full(n, 8.0)

    # Warmup
    qk.cos_method_fang_oosterlee_price_batch(spot[:100], strike[:100], tau[:100],
                                              vol[:100], r[:100], q[:100], ot[:100],
                                              n_terms[:100], tw[:100])

    scalar_ms = _median_ms(
        lambda: [qk.cos_method_fang_oosterlee_price(
            float(spot[i]), float(strike[i]), float(tau[i]), float(vol[i]),
            float(r[i]), float(q[i]), int(ot[i]), int(n_terms[i]), float(tw[i])
        ) for i in range(n)],
    )
    batch_ms = _median_ms(
        lambda: qk.cos_method_fang_oosterlee_price_batch(spot, strike, tau, vol, r, q, ot, n_terms, tw),
    )

    assert batch_ms < scalar_ms * 0.95


def test_tree_batch_speed_regression(qk):
    """CRR tree batch must beat scalar loop by >5%."""
    rng = np.random.default_rng(13)
    n = 2000
    spot = rng.uniform(80.0, 120.0, n).astype(np.float64)
    strike = rng.uniform(80.0, 120.0, n).astype(np.float64)
    tau = rng.uniform(0.25, 1.0, n).astype(np.float64)
    vol = rng.uniform(0.15, 0.4, n).astype(np.float64)
    r = rng.uniform(0.01, 0.05, n).astype(np.float64)
    q = rng.uniform(0.0, 0.02, n).astype(np.float64)
    ot = np.where((np.arange(n) & 1) == 0, QK_CALL, QK_PUT).astype(np.int32)
    steps = np.full(n, 64, dtype=np.int32)
    am = (np.arange(n) & 1).astype(np.int32)

    # Warmup
    qk.crr_price_batch(spot[:100], strike[:100], tau[:100], vol[:100],
                        r[:100], q[:100], ot[:100], steps[:100], am[:100])

    scalar_ms = _median_ms(
        lambda: [qk.crr_price(float(spot[i]), float(strike[i]), float(tau[i]), float(vol[i]),
                               float(r[i]), float(q[i]), int(ot[i]), int(steps[i]), bool(am[i]))
                 for i in range(n)],
    )
    batch_ms = _median_ms(
        lambda: qk.crr_price_batch(spot, strike, tau, vol, r, q, ot, steps, am),
    )

    assert batch_ms < scalar_ms * 0.95


def test_heston_batch_speed_regression(qk):
    """Heston batch must beat scalar loop by >5%."""
    rng = np.random.default_rng(17)
    n = 200
    spot = rng.uniform(90.0, 110.0, n).astype(np.float64)
    strike = rng.uniform(90.0, 110.0, n).astype(np.float64)
    tau = rng.uniform(0.25, 1.0, n).astype(np.float64)
    r = rng.uniform(0.01, 0.05, n).astype(np.float64)
    q = rng.uniform(0.0, 0.02, n).astype(np.float64)
    v0 = rng.uniform(0.02, 0.08, n).astype(np.float64)
    kappa = rng.uniform(1.0, 3.0, n).astype(np.float64)
    theta = rng.uniform(0.02, 0.08, n).astype(np.float64)
    sigma = rng.uniform(0.2, 0.5, n).astype(np.float64)
    rho = rng.uniform(-0.8, -0.3, n).astype(np.float64)
    ot = np.where((np.arange(n) & 1) == 0, QK_CALL, QK_PUT).astype(np.int32)
    isteps = np.full(n, 512, dtype=np.int32)
    ilimit = np.full(n, 80.0)

    # Warmup
    qk.heston_price_cf_batch(spot[:10], strike[:10], tau[:10], r[:10], q[:10],
                              v0[:10], kappa[:10], theta[:10], sigma[:10], rho[:10],
                              ot[:10], isteps[:10], ilimit[:10])

    scalar_ms = _median_ms(
        lambda: [qk.heston_price_cf(
            float(spot[i]), float(strike[i]), float(tau[i]),
            float(r[i]), float(q[i]), float(v0[i]), float(kappa[i]), float(theta[i]),
            float(sigma[i]), float(rho[i]), int(ot[i]), int(isteps[i]), float(ilimit[i])
        ) for i in range(n)],
    )
    batch_ms = _median_ms(
        lambda: qk.heston_price_cf_batch(spot, strike, tau, r, q, v0, kappa, theta, sigma, rho, ot, isteps, ilimit),
    )

    assert batch_ms < scalar_ms * 0.95

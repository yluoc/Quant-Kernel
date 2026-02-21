"""Native batch API accuracy checks."""

import numpy as np

from quantkernel import QK_CALL, QK_PUT


def test_closed_form_batch_apis_match_scalar(qk):
    rng = np.random.default_rng(42)
    n = 256

    spot = rng.uniform(80.0, 120.0, n).astype(np.float64)
    strike = rng.uniform(80.0, 120.0, n).astype(np.float64)
    tau = rng.uniform(0.25, 2.0, n).astype(np.float64)
    vol = rng.uniform(0.1, 0.6, n).astype(np.float64)
    r = rng.uniform(0.0, 0.08, n).astype(np.float64)
    q = rng.uniform(0.0, 0.04, n).astype(np.float64)
    option_type = np.where((np.arange(n) & 1) == 0, QK_CALL, QK_PUT).astype(np.int32)

    bsm_batch = qk.black_scholes_merton_price_batch(spot, strike, tau, vol, r, q, option_type)
    bsm_scalar = np.array(
        [qk.black_scholes_merton_price(float(spot[i]), float(strike[i]), float(tau[i]), float(vol[i]),
                                       float(r[i]), float(q[i]), int(option_type[i])) for i in range(n)],
        dtype=np.float64,
    )
    assert np.allclose(bsm_batch, bsm_scalar, atol=1e-12, rtol=1e-12)

    forward = spot * np.exp((r - q) * tau)
    black_batch = qk.black76_price_batch(forward, strike, tau, vol, r, option_type)
    black_scalar = np.array(
        [qk.black76_price(float(forward[i]), float(strike[i]), float(tau[i]), float(vol[i]),
                          float(r[i]), int(option_type[i])) for i in range(n)],
        dtype=np.float64,
    )
    assert np.allclose(black_batch, black_scalar, atol=1e-12, rtol=1e-12)

    normal_vol = spot * vol
    bach_batch = qk.bachelier_price_batch(forward, strike, tau, normal_vol, r, option_type)
    bach_scalar = np.array(
        [qk.bachelier_price(float(forward[i]), float(strike[i]), float(tau[i]), float(normal_vol[i]),
                            float(r[i]), int(option_type[i])) for i in range(n)],
        dtype=np.float64,
    )
    assert np.allclose(bach_batch, bach_scalar, atol=1e-12, rtol=1e-12)


def test_tree_mc_fourier_batch_apis_match_scalar(qk):
    rng = np.random.default_rng(7)
    n = 32

    spot = rng.uniform(80.0, 120.0, n).astype(np.float64)
    strike = rng.uniform(80.0, 120.0, n).astype(np.float64)
    tau = rng.uniform(0.25, 2.0, n).astype(np.float64)
    vol = rng.uniform(0.1, 0.5, n).astype(np.float64)
    r = rng.uniform(0.0, 0.06, n).astype(np.float64)
    q = rng.uniform(0.0, 0.03, n).astype(np.float64)
    option_type = np.where((np.arange(n) & 1) == 0, QK_CALL, QK_PUT).astype(np.int32)

    steps = rng.integers(64, 256, size=n, dtype=np.int32)
    american_style = (np.arange(n) & 1).astype(np.int32)
    crr_batch = qk.crr_price_batch(spot, strike, tau, vol, r, q, option_type, steps, american_style)
    crr_scalar = np.array(
        [qk.crr_price(float(spot[i]), float(strike[i]), float(tau[i]), float(vol[i]),
                      float(r[i]), float(q[i]), int(option_type[i]), int(steps[i]), bool(american_style[i]))
         for i in range(n)],
        dtype=np.float64,
    )
    assert np.allclose(crr_batch, crr_scalar, atol=1e-12, rtol=1e-12)

    grid_size = np.full(n, 1024, dtype=np.int32)
    eta = np.full(n, 0.25, dtype=np.float64)
    alpha = np.full(n, 1.5, dtype=np.float64)
    fft_batch = qk.carr_madan_fft_price_batch(spot, strike, tau, vol, r, q, option_type, grid_size, eta, alpha)
    fft_scalar = np.array(
        [qk.carr_madan_fft_price(float(spot[i]), float(strike[i]), float(tau[i]), float(vol[i]),
                                 float(r[i]), float(q[i]), int(option_type[i]),
                                 int(grid_size[i]), float(eta[i]), float(alpha[i]))
         for i in range(n)],
        dtype=np.float64,
    )
    assert np.allclose(fft_batch, fft_scalar, atol=1e-12, rtol=1e-12)

    m = 16
    paths = np.full(m, 2048, dtype=np.int32)
    seed = (np.arange(m, dtype=np.uint64) + np.uint64(12345))
    mc_batch = qk.standard_monte_carlo_price_batch(
        spot[:m], strike[:m], tau[:m], vol[:m], r[:m], q[:m], option_type[:m], paths, seed
    )
    mc_scalar = np.array(
        [qk.standard_monte_carlo_price(float(spot[i]), float(strike[i]), float(tau[i]), float(vol[i]),
                                       float(r[i]), float(q[i]), int(option_type[i]),
                                       int(paths[i]), int(seed[i]))
         for i in range(m)],
        dtype=np.float64,
    )
    assert np.allclose(mc_batch, mc_scalar, atol=1e-12, rtol=1e-12)


def test_heston_batch(qk):
    rng = np.random.default_rng(100)
    n = 16
    spot = rng.uniform(90.0, 110.0, n)
    strike = rng.uniform(90.0, 110.0, n)
    tau = rng.uniform(0.25, 1.0, n)
    r = rng.uniform(0.01, 0.05, n)
    q = rng.uniform(0.0, 0.02, n)
    v0 = rng.uniform(0.02, 0.08, n)
    kappa = rng.uniform(1.0, 3.0, n)
    theta = rng.uniform(0.02, 0.08, n)
    sigma = rng.uniform(0.2, 0.5, n)
    rho = rng.uniform(-0.8, -0.3, n)
    ot = np.where((np.arange(n) & 1) == 0, QK_CALL, QK_PUT).astype(np.int32)
    isteps = np.full(n, 512, dtype=np.int32)
    ilimit = np.full(n, 80.0)
    batch = qk.heston_price_cf_batch(spot, strike, tau, r, q, v0, kappa, theta, sigma, rho, ot, isteps, ilimit)
    scalar = np.array([qk.heston_price_cf(float(spot[i]), float(strike[i]), float(tau[i]),
                       float(r[i]), float(q[i]), float(v0[i]), float(kappa[i]), float(theta[i]),
                       float(sigma[i]), float(rho[i]), int(ot[i]), int(isteps[i]), float(ilimit[i]))
                       for i in range(n)])
    assert np.allclose(batch, scalar, atol=1e-12, rtol=1e-12)


def test_merton_jump_diffusion_batch(qk):
    rng = np.random.default_rng(101)
    n = 16
    spot = rng.uniform(90.0, 110.0, n)
    strike = rng.uniform(90.0, 110.0, n)
    tau = rng.uniform(0.25, 1.0, n)
    vol = rng.uniform(0.15, 0.4, n)
    r = rng.uniform(0.01, 0.05, n)
    q = rng.uniform(0.0, 0.02, n)
    ji = rng.uniform(0.5, 2.0, n)
    jm = rng.uniform(-0.1, 0.1, n)
    jv = rng.uniform(0.05, 0.2, n)
    mt = np.full(n, 20, dtype=np.int32)
    ot = np.where((np.arange(n) & 1) == 0, QK_CALL, QK_PUT).astype(np.int32)
    batch = qk.merton_jump_diffusion_price_batch(spot, strike, tau, vol, r, q, ji, jm, jv, mt, ot)
    scalar = np.array([qk.merton_jump_diffusion_price(float(spot[i]), float(strike[i]), float(tau[i]),
                       float(vol[i]), float(r[i]), float(q[i]), float(ji[i]), float(jm[i]),
                       float(jv[i]), int(mt[i]), int(ot[i])) for i in range(n)])
    assert np.allclose(batch, scalar, atol=1e-12, rtol=1e-12)


def test_variance_gamma_batch(qk):
    rng = np.random.default_rng(102)
    n = 16
    spot = rng.uniform(90.0, 110.0, n)
    strike = rng.uniform(90.0, 110.0, n)
    tau = rng.uniform(0.25, 1.0, n)
    r = rng.uniform(0.01, 0.05, n)
    q = rng.uniform(0.0, 0.02, n)
    sigma = rng.uniform(0.15, 0.4, n)
    theta = rng.uniform(-0.2, 0.0, n)
    nu = rng.uniform(0.1, 0.5, n)
    ot = np.where((np.arange(n) & 1) == 0, QK_CALL, QK_PUT).astype(np.int32)
    isteps = np.full(n, 512, dtype=np.int32)
    ilimit = np.full(n, 80.0)
    batch = qk.variance_gamma_price_cf_batch(spot, strike, tau, r, q, sigma, theta, nu, ot, isteps, ilimit)
    scalar = np.array([qk.variance_gamma_price_cf(float(spot[i]), float(strike[i]), float(tau[i]),
                       float(r[i]), float(q[i]), float(sigma[i]), float(theta[i]), float(nu[i]),
                       int(ot[i]), int(isteps[i]), float(ilimit[i])) for i in range(n)])
    assert np.allclose(batch, scalar, atol=1e-12, rtol=1e-12)


def test_sabr_hagan_lognormal_iv_batch(qk):
    rng = np.random.default_rng(103)
    n = 16
    forward = rng.uniform(90.0, 110.0, n)
    strike = rng.uniform(85.0, 115.0, n)
    tau = rng.uniform(0.25, 2.0, n)
    alpha = rng.uniform(0.1, 0.4, n)
    beta = rng.uniform(0.3, 0.9, n)
    rho = rng.uniform(-0.7, 0.0, n)
    nu = rng.uniform(0.2, 0.8, n)
    batch = qk.sabr_hagan_lognormal_iv_batch(forward, strike, tau, alpha, beta, rho, nu)
    scalar = np.array([qk.sabr_hagan_lognormal_iv(float(forward[i]), float(strike[i]), float(tau[i]),
                       float(alpha[i]), float(beta[i]), float(rho[i]), float(nu[i])) for i in range(n)])
    assert np.allclose(batch, scalar, atol=1e-12, rtol=1e-12)


def test_sabr_hagan_black76_price_batch(qk):
    rng = np.random.default_rng(104)
    n = 16
    forward = rng.uniform(90.0, 110.0, n)
    strike = rng.uniform(85.0, 115.0, n)
    tau = rng.uniform(0.25, 2.0, n)
    r = rng.uniform(0.01, 0.05, n)
    alpha = rng.uniform(0.1, 0.4, n)
    beta = rng.uniform(0.3, 0.9, n)
    rho = rng.uniform(-0.7, 0.0, n)
    nu = rng.uniform(0.2, 0.8, n)
    ot = np.where((np.arange(n) & 1) == 0, QK_CALL, QK_PUT).astype(np.int32)
    batch = qk.sabr_hagan_black76_price_batch(forward, strike, tau, r, alpha, beta, rho, nu, ot)
    scalar = np.array([qk.sabr_hagan_black76_price(float(forward[i]), float(strike[i]), float(tau[i]),
                       float(r[i]), float(alpha[i]), float(beta[i]), float(rho[i]), float(nu[i]),
                       int(ot[i])) for i in range(n)])
    assert np.allclose(batch, scalar, atol=1e-12, rtol=1e-12)


def test_dupire_local_vol_batch(qk):
    rng = np.random.default_rng(105)
    n = 16
    strike = rng.uniform(90.0, 110.0, n)
    tau = rng.uniform(0.25, 2.0, n)
    call_price = rng.uniform(5.0, 20.0, n)
    r = rng.uniform(0.01, 0.03, n)
    q = rng.uniform(0.0, 0.01, n)
    d2C_dK2 = rng.uniform(0.01, 0.05, n)
    dC_dK = rng.uniform(-0.05, -0.01, n)
    dC_dT = rng.uniform(3.0, 8.0, n)
    batch = qk.dupire_local_vol_batch(strike, tau, call_price, dC_dT, dC_dK, d2C_dK2, r, q)
    scalar = np.array([qk.dupire_local_vol(float(strike[i]), float(tau[i]), float(call_price[i]),
                       float(dC_dT[i]), float(dC_dK[i]), float(d2C_dK2[i]),
                       float(r[i]), float(q[i])) for i in range(n)])
    assert np.allclose(batch, scalar, atol=1e-12, rtol=1e-12)


def test_tree_lattice_batch_apis(qk):
    rng = np.random.default_rng(200)
    n = 16
    spot = rng.uniform(90.0, 110.0, n)
    strike = rng.uniform(90.0, 110.0, n)
    tau = rng.uniform(0.25, 1.0, n)
    vol = rng.uniform(0.15, 0.4, n)
    r = rng.uniform(0.01, 0.05, n)
    q = rng.uniform(0.0, 0.02, n)
    ot = np.where((np.arange(n) & 1) == 0, QK_CALL, QK_PUT).astype(np.int32)
    steps = rng.integers(64, 128, size=n, dtype=np.int32)
    am = (np.arange(n) & 1).astype(np.int32)

    for method in ("jarrow_rudd", "tian", "leisen_reimer", "trinomial_tree"):
        batch_fn = getattr(qk, f"{method}_price_batch")
        scalar_fn = getattr(qk, f"{method}_price")
        batch = batch_fn(spot, strike, tau, vol, r, q, ot, steps, am)
        scalar = np.array([scalar_fn(float(spot[i]), float(strike[i]), float(tau[i]), float(vol[i]),
                           float(r[i]), float(q[i]), int(ot[i]), int(steps[i]), bool(am[i]))
                           for i in range(n)])
        assert np.allclose(batch, scalar, atol=1e-12, rtol=1e-12), f"{method}_price_batch mismatch"

    dk_batch = qk.derman_kani_const_local_vol_price_batch(spot, strike, tau, vol, r, q, ot, steps, am)
    dk_scalar = np.array([qk.derman_kani_const_local_vol_price(float(spot[i]), float(strike[i]), float(tau[i]),
                          float(vol[i]), float(r[i]), float(q[i]), int(ot[i]), int(steps[i]), bool(am[i]))
                          for i in range(n)])
    assert np.allclose(dk_batch, dk_scalar, atol=1e-12, rtol=1e-12)


def test_mc_euler_milstein_longstaff_batch(qk):
    rng = np.random.default_rng(300)
    n = 8
    spot = rng.uniform(90.0, 110.0, n)
    strike = rng.uniform(90.0, 110.0, n)
    tau = rng.uniform(0.25, 1.0, n)
    vol = rng.uniform(0.15, 0.4, n)
    r = rng.uniform(0.01, 0.05, n)
    q = rng.uniform(0.0, 0.02, n)
    ot = np.where((np.arange(n) & 1) == 0, QK_CALL, QK_PUT).astype(np.int32)
    paths = np.full(n, 1024, dtype=np.int32)
    mc_steps = np.full(n, 50, dtype=np.int32)
    seed = (np.arange(n, dtype=np.uint64) + np.uint64(42))

    for method in ("euler_maruyama", "milstein", "longstaff_schwartz"):
        batch_fn = getattr(qk, f"{method}_price_batch")
        scalar_fn = getattr(qk, f"{method}_price")
        batch = batch_fn(spot, strike, tau, vol, r, q, ot, paths, mc_steps, seed)
        scalar = np.array([scalar_fn(float(spot[i]), float(strike[i]), float(tau[i]), float(vol[i]),
                           float(r[i]), float(q[i]), int(ot[i]), int(paths[i]), int(mc_steps[i]), int(seed[i]))
                           for i in range(n)])
        assert np.allclose(batch, scalar, atol=1e-12, rtol=1e-12), f"{method}_price_batch mismatch"


def test_quasi_mc_sobol_halton_batch(qk):
    rng = np.random.default_rng(301)
    n = 8
    spot = rng.uniform(90.0, 110.0, n)
    strike = rng.uniform(90.0, 110.0, n)
    tau = rng.uniform(0.25, 1.0, n)
    vol = rng.uniform(0.15, 0.4, n)
    r = rng.uniform(0.01, 0.05, n)
    q = rng.uniform(0.0, 0.02, n)
    ot = np.where((np.arange(n) & 1) == 0, QK_CALL, QK_PUT).astype(np.int32)
    paths = np.full(n, 1024, dtype=np.int32)

    for method in ("quasi_monte_carlo_sobol", "quasi_monte_carlo_halton"):
        batch_fn = getattr(qk, f"{method}_price_batch")
        scalar_fn = getattr(qk, f"{method}_price")
        batch = batch_fn(spot, strike, tau, vol, r, q, ot, paths)
        scalar = np.array([scalar_fn(float(spot[i]), float(strike[i]), float(tau[i]), float(vol[i]),
                           float(r[i]), float(q[i]), int(ot[i]), int(paths[i]))
                           for i in range(n)])
        assert np.allclose(batch, scalar, atol=1e-12, rtol=1e-12), f"{method}_price_batch mismatch"


def test_multilevel_mc_batch(qk):
    rng = np.random.default_rng(302)
    n = 8
    spot = rng.uniform(90.0, 110.0, n)
    strike = rng.uniform(90.0, 110.0, n)
    tau = rng.uniform(0.25, 1.0, n)
    vol = rng.uniform(0.15, 0.4, n)
    r = rng.uniform(0.01, 0.05, n)
    q = rng.uniform(0.0, 0.02, n)
    ot = np.where((np.arange(n) & 1) == 0, QK_CALL, QK_PUT).astype(np.int32)
    base_paths = np.full(n, 512, dtype=np.int32)
    levels = np.full(n, 3, dtype=np.int32)
    base_steps = np.full(n, 16, dtype=np.int32)
    seed = (np.arange(n, dtype=np.uint64) + np.uint64(42))
    batch = qk.multilevel_monte_carlo_price_batch(spot, strike, tau, vol, r, q, ot, base_paths, levels, base_steps, seed)
    scalar = np.array([qk.multilevel_monte_carlo_price(float(spot[i]), float(strike[i]), float(tau[i]),
                       float(vol[i]), float(r[i]), float(q[i]), int(ot[i]),
                       int(base_paths[i]), int(levels[i]), int(base_steps[i]), int(seed[i]))
                       for i in range(n)])
    assert np.allclose(batch, scalar, atol=1e-12, rtol=1e-12)


def test_importance_sampling_batch(qk):
    rng = np.random.default_rng(303)
    n = 8
    spot = rng.uniform(90.0, 110.0, n)
    strike = rng.uniform(90.0, 110.0, n)
    tau = rng.uniform(0.25, 1.0, n)
    vol = rng.uniform(0.15, 0.4, n)
    r = rng.uniform(0.01, 0.05, n)
    q = rng.uniform(0.0, 0.02, n)
    ot = np.where((np.arange(n) & 1) == 0, QK_CALL, QK_PUT).astype(np.int32)
    paths = np.full(n, 1024, dtype=np.int32)
    shift = rng.uniform(0.1, 0.5, n)
    seed = (np.arange(n, dtype=np.uint64) + np.uint64(42))
    batch = qk.importance_sampling_price_batch(spot, strike, tau, vol, r, q, ot, paths, shift, seed)
    scalar = np.array([qk.importance_sampling_price(float(spot[i]), float(strike[i]), float(tau[i]),
                       float(vol[i]), float(r[i]), float(q[i]), int(ot[i]),
                       int(paths[i]), float(shift[i]), int(seed[i]))
                       for i in range(n)])
    assert np.allclose(batch, scalar, atol=1e-12, rtol=1e-12)


def test_control_antithetic_stratified_batch(qk):
    rng = np.random.default_rng(304)
    n = 8
    spot = rng.uniform(90.0, 110.0, n)
    strike = rng.uniform(90.0, 110.0, n)
    tau = rng.uniform(0.25, 1.0, n)
    vol = rng.uniform(0.15, 0.4, n)
    r = rng.uniform(0.01, 0.05, n)
    q = rng.uniform(0.0, 0.02, n)
    ot = np.where((np.arange(n) & 1) == 0, QK_CALL, QK_PUT).astype(np.int32)
    paths = np.full(n, 1024, dtype=np.int32)
    seed = (np.arange(n, dtype=np.uint64) + np.uint64(42))

    for method in ("control_variates", "antithetic_variates", "stratified_sampling"):
        batch_fn = getattr(qk, f"{method}_price_batch")
        scalar_fn = getattr(qk, f"{method}_price")
        batch = batch_fn(spot, strike, tau, vol, r, q, ot, paths, seed)
        scalar = np.array([scalar_fn(float(spot[i]), float(strike[i]), float(tau[i]), float(vol[i]),
                           float(r[i]), float(q[i]), int(ot[i]), int(paths[i]), int(seed[i]))
                           for i in range(n)])
        assert np.allclose(batch, scalar, atol=1e-12, rtol=1e-12), f"{method}_price_batch mismatch"


def test_cos_fang_oosterlee_batch(qk):
    rng = np.random.default_rng(400)
    n = 16
    spot = rng.uniform(90.0, 110.0, n)
    strike = rng.uniform(90.0, 110.0, n)
    tau = rng.uniform(0.25, 1.0, n)
    vol = rng.uniform(0.15, 0.4, n)
    r = rng.uniform(0.01, 0.05, n)
    q = rng.uniform(0.0, 0.02, n)
    ot = np.where((np.arange(n) & 1) == 0, QK_CALL, QK_PUT).astype(np.int32)
    n_terms = np.full(n, 128, dtype=np.int32)
    tw = np.full(n, 8.0)
    batch = qk.cos_method_fang_oosterlee_price_batch(spot, strike, tau, vol, r, q, ot, n_terms, tw)
    scalar = np.array([qk.cos_method_fang_oosterlee_price(float(spot[i]), float(strike[i]), float(tau[i]),
                       float(vol[i]), float(r[i]), float(q[i]), int(ot[i]),
                       int(n_terms[i]), float(tw[i])) for i in range(n)])
    assert np.allclose(batch, scalar, atol=1e-12, rtol=1e-12)


def test_fractional_fft_batch(qk):
    rng = np.random.default_rng(401)
    n = 16
    spot = rng.uniform(90.0, 110.0, n)
    strike = rng.uniform(90.0, 110.0, n)
    tau = rng.uniform(0.25, 1.0, n)
    vol = rng.uniform(0.15, 0.4, n)
    r = rng.uniform(0.01, 0.05, n)
    q = rng.uniform(0.0, 0.02, n)
    ot = np.where((np.arange(n) & 1) == 0, QK_CALL, QK_PUT).astype(np.int32)
    gs = np.full(n, 256, dtype=np.int32)
    eta = np.full(n, 0.25)
    lam = np.full(n, 0.05)
    alpha = np.full(n, 1.5)
    batch = qk.fractional_fft_price_batch(spot, strike, tau, vol, r, q, ot, gs, eta, lam, alpha)
    scalar = np.array([qk.fractional_fft_price(float(spot[i]), float(strike[i]), float(tau[i]),
                       float(vol[i]), float(r[i]), float(q[i]), int(ot[i]),
                       int(gs[i]), float(eta[i]), float(lam[i]), float(alpha[i])) for i in range(n)])
    assert np.allclose(batch, scalar, atol=1e-12, rtol=1e-12)


def test_lewis_hilbert_batch(qk):
    rng = np.random.default_rng(402)
    n = 16
    spot = rng.uniform(90.0, 110.0, n)
    strike = rng.uniform(90.0, 110.0, n)
    tau = rng.uniform(0.25, 1.0, n)
    vol = rng.uniform(0.15, 0.4, n)
    r = rng.uniform(0.01, 0.05, n)
    q = rng.uniform(0.0, 0.02, n)
    ot = np.where((np.arange(n) & 1) == 0, QK_CALL, QK_PUT).astype(np.int32)
    isteps = np.full(n, 2048, dtype=np.int32)
    ilimit = np.full(n, 200.0)

    for method, batch_name in [("lewis_fourier_inversion", "lewis_fourier_inversion"),
                                ("hilbert_transform", "hilbert_transform")]:
        batch_fn = getattr(qk, f"{method}_price_batch")
        scalar_fn = getattr(qk, f"{method}_price")
        batch = batch_fn(spot, strike, tau, vol, r, q, ot, isteps, ilimit)
        scalar = np.array([scalar_fn(float(spot[i]), float(strike[i]), float(tau[i]),
                           float(vol[i]), float(r[i]), float(q[i]), int(ot[i]),
                           int(isteps[i]), float(ilimit[i])) for i in range(n)])
        assert np.allclose(batch, scalar, atol=1e-12, rtol=1e-12), f"{method}_price_batch mismatch"


def test_fdm_batch_apis(qk):
    rng = np.random.default_rng(500)
    n = 16
    spot = rng.uniform(90.0, 110.0, n)
    strike = rng.uniform(90.0, 110.0, n)
    tau = rng.uniform(0.25, 1.0, n)
    vol = rng.uniform(0.15, 0.4, n)
    r = rng.uniform(0.01, 0.05, n)
    q = rng.uniform(0.0, 0.02, n)
    ot = np.where((np.arange(n) & 1) == 0, QK_CALL, QK_PUT).astype(np.int32)
    time_steps = np.full(n, 50, dtype=np.int32)
    spot_steps = np.full(n, 50, dtype=np.int32)
    american_style = (np.arange(n) & 1).astype(np.int32)

    for method in ("explicit_fd", "implicit_fd", "crank_nicolson"):
        batch_fn = getattr(qk, f"{method}_price_batch")
        scalar_fn = getattr(qk, f"{method}_price")
        batch = batch_fn(spot, strike, tau, vol, r, q, ot, time_steps, spot_steps, american_style)
        scalar = np.array([scalar_fn(float(spot[i]), float(strike[i]), float(tau[i]), float(vol[i]),
                           float(r[i]), float(q[i]), int(ot[i]), int(time_steps[i]), int(spot_steps[i]),
                           bool(american_style[i])) for i in range(n)])
        assert np.allclose(batch, scalar, atol=1e-12, rtol=1e-12), f"{method}_price_batch mismatch"

    v0 = rng.uniform(0.02, 0.08, n)
    kappa = rng.uniform(1.0, 3.0, n)
    theta_v = rng.uniform(0.02, 0.08, n)
    sigma = rng.uniform(0.2, 0.5, n)
    rho = rng.uniform(-0.8, -0.3, n)
    s_steps = np.full(n, 20, dtype=np.int32)
    v_steps = np.full(n, 10, dtype=np.int32)
    time_steps_adi = np.full(n, 20, dtype=np.int32)

    for method in ("adi_douglas", "adi_craig_sneyd", "adi_hundsdorfer_verwer"):
        batch_fn = getattr(qk, f"{method}_price_batch")
        scalar_fn = getattr(qk, f"{method}_price")
        batch = batch_fn(spot, strike, tau, r, q, v0, kappa, theta_v, sigma, rho, ot, s_steps, v_steps, time_steps_adi)
        scalar = np.array([scalar_fn(float(spot[i]), float(strike[i]), float(tau[i]),
                           float(r[i]), float(q[i]), float(v0[i]), float(kappa[i]), float(theta_v[i]),
                           float(sigma[i]), float(rho[i]), int(ot[i]), int(s_steps[i]), int(v_steps[i]),
                           int(time_steps_adi[i])) for i in range(n)])
        assert np.allclose(batch, scalar, atol=1e-12, rtol=1e-12), f"{method}_price_batch mismatch"

    omega = np.full(n, 1.2)
    tol = np.full(n, 1e-8)
    max_iter = np.full(n, 10000, dtype=np.int32)
    batch = qk.psor_price_batch(spot, strike, tau, vol, r, q, ot, time_steps, spot_steps, omega, tol, max_iter)
    scalar = np.array([qk.psor_price(float(spot[i]), float(strike[i]), float(tau[i]), float(vol[i]),
                       float(r[i]), float(q[i]), int(ot[i]), int(time_steps[i]), int(spot_steps[i]),
                       float(omega[i]), float(tol[i]), int(max_iter[i])) for i in range(n)])
    assert np.allclose(batch, scalar, atol=1e-12, rtol=1e-12)


def test_iqm_batch_apis(qk):
    rng = np.random.default_rng(600)
    n = 16
    spot = rng.uniform(90.0, 110.0, n)
    strike = rng.uniform(90.0, 110.0, n)
    tau = rng.uniform(0.25, 1.0, n)
    vol = rng.uniform(0.15, 0.4, n)
    r = rng.uniform(0.01, 0.05, n)
    q = rng.uniform(0.0, 0.02, n)
    ot = np.where((np.arange(n) & 1) == 0, QK_CALL, QK_PUT).astype(np.int32)

    n_points = np.full(n, 64, dtype=np.int32)
    batch = qk.gauss_hermite_price_batch(spot, strike, tau, vol, r, q, ot, n_points)
    scalar = np.array([qk.gauss_hermite_price(float(spot[i]), float(strike[i]), float(tau[i]), float(vol[i]),
                       float(r[i]), float(q[i]), int(ot[i]), int(n_points[i])) for i in range(n)])
    assert np.allclose(batch, scalar, atol=1e-12, rtol=1e-12)

    n_points = np.full(n, 32, dtype=np.int32)
    batch = qk.gauss_laguerre_price_batch(spot, strike, tau, vol, r, q, ot, n_points)
    scalar = np.array([qk.gauss_laguerre_price(float(spot[i]), float(strike[i]), float(tau[i]), float(vol[i]),
                       float(r[i]), float(q[i]), int(ot[i]), int(n_points[i])) for i in range(n)])
    assert np.allclose(batch, scalar, atol=1e-12, rtol=1e-12)

    n_points = np.full(n, 64, dtype=np.int32)
    integration_limit = np.full(n, 200.0)
    batch = qk.gauss_legendre_price_batch(spot, strike, tau, vol, r, q, ot, n_points, integration_limit)
    scalar = np.array([qk.gauss_legendre_price(float(spot[i]), float(strike[i]), float(tau[i]), float(vol[i]),
                       float(r[i]), float(q[i]), int(ot[i]), int(n_points[i]),
                       float(integration_limit[i])) for i in range(n)])
    assert np.allclose(batch, scalar, atol=1e-12, rtol=1e-12)

    abs_tol = np.full(n, 1e-9)
    rel_tol = np.full(n, 1e-8)
    max_depth = np.full(n, 14, dtype=np.int32)
    integration_limit = np.full(n, 200.0)
    batch = qk.adaptive_quadrature_price_batch(spot, strike, tau, vol, r, q, ot, abs_tol, rel_tol, max_depth, integration_limit)
    scalar = np.array([qk.adaptive_quadrature_price(float(spot[i]), float(strike[i]), float(tau[i]), float(vol[i]),
                       float(r[i]), float(q[i]), int(ot[i]), float(abs_tol[i]), float(rel_tol[i]),
                       int(max_depth[i]), float(integration_limit[i])) for i in range(n)])
    assert np.allclose(batch, scalar, atol=1e-12, rtol=1e-12)


def test_ram_batch_apis(qk):
    rng = np.random.default_rng(700)
    n = 16
    spot = rng.uniform(90.0, 110.0, n)
    strike = rng.uniform(90.0, 110.0, n)
    tau = rng.uniform(0.25, 1.0, n)
    vol = rng.uniform(0.15, 0.4, n)
    r = rng.uniform(0.01, 0.05, n)
    q = rng.uniform(0.0, 0.02, n)
    ot = np.where((np.arange(n) & 1) == 0, QK_CALL, QK_PUT).astype(np.int32)

    polynomial_order = np.full(n, 4, dtype=np.int32)
    quadrature_points = np.full(n, 32, dtype=np.int32)
    batch = qk.polynomial_chaos_expansion_price_batch(spot, strike, tau, vol, r, q, ot, polynomial_order, quadrature_points)
    scalar = np.array([qk.polynomial_chaos_expansion_price(float(spot[i]), float(strike[i]), float(tau[i]),
                       float(vol[i]), float(r[i]), float(q[i]), int(ot[i]),
                       int(polynomial_order[i]), int(quadrature_points[i])) for i in range(n)])
    assert np.allclose(batch, scalar, atol=1e-12, rtol=1e-12)

    centers = np.full(n, 24, dtype=np.int32)
    rbf_shape = np.full(n, 1.0)
    ridge = np.full(n, 1e-4)
    batch = qk.radial_basis_function_price_batch(spot, strike, tau, vol, r, q, ot, centers, rbf_shape, ridge)
    scalar = np.array([qk.radial_basis_function_price(float(spot[i]), float(strike[i]), float(tau[i]),
                       float(vol[i]), float(r[i]), float(q[i]), int(ot[i]),
                       int(centers[i]), float(rbf_shape[i]), float(ridge[i])) for i in range(n)])
    assert np.allclose(batch, scalar, atol=1e-12, rtol=1e-12)

    level = np.full(n, 3, dtype=np.int32)
    nodes_per_dim = np.full(n, 9, dtype=np.int32)
    batch = qk.sparse_grid_collocation_price_batch(spot, strike, tau, vol, r, q, ot, level, nodes_per_dim)
    scalar = np.array([qk.sparse_grid_collocation_price(float(spot[i]), float(strike[i]), float(tau[i]),
                       float(vol[i]), float(r[i]), float(q[i]), int(ot[i]),
                       int(level[i]), int(nodes_per_dim[i])) for i in range(n)])
    assert np.allclose(batch, scalar, atol=1e-12, rtol=1e-12)

    modes = np.full(n, 8, dtype=np.int32)
    snapshots = np.full(n, 64, dtype=np.int32)
    batch = qk.proper_orthogonal_decomposition_price_batch(spot, strike, tau, vol, r, q, ot, modes, snapshots)
    scalar = np.array([qk.proper_orthogonal_decomposition_price(float(spot[i]), float(strike[i]), float(tau[i]),
                       float(vol[i]), float(r[i]), float(q[i]), int(ot[i]),
                       int(modes[i]), int(snapshots[i])) for i in range(n)])
    assert np.allclose(batch, scalar, atol=1e-12, rtol=1e-12)


def test_agm_batch_apis(qk):
    rng = np.random.default_rng(800)
    n = 8
    spot = rng.uniform(90.0, 110.0, n)
    strike = rng.uniform(90.0, 110.0, n)
    tau = rng.uniform(0.25, 1.0, n)
    vol = rng.uniform(0.15, 0.4, n)
    r = rng.uniform(0.01, 0.05, n)
    q = rng.uniform(0.0, 0.02, n)
    ot = np.where((np.arange(n) & 1) == 0, QK_CALL, QK_PUT).astype(np.int32)

    paths = np.full(n, 2048, dtype=np.int32)
    seed_arr = (np.arange(n, dtype=np.uint64) + np.uint64(42))
    batch = qk.pathwise_derivative_delta_batch(spot, strike, tau, vol, r, q, ot, paths, seed_arr)
    scalar = np.array([qk.pathwise_derivative_delta(float(spot[i]), float(strike[i]), float(tau[i]),
                       float(vol[i]), float(r[i]), float(q[i]), int(ot[i]),
                       int(paths[i]), int(seed_arr[i])) for i in range(n)])
    assert np.allclose(batch, scalar, atol=1e-12, rtol=1e-12)

    paths = np.full(n, 2048, dtype=np.int32)
    seed_arr = (np.arange(n, dtype=np.uint64) + np.uint64(42))
    weight_clip = np.full(n, 6.0)
    batch = qk.likelihood_ratio_delta_batch(spot, strike, tau, vol, r, q, ot, paths, seed_arr, weight_clip)
    scalar = np.array([qk.likelihood_ratio_delta(float(spot[i]), float(strike[i]), float(tau[i]),
                       float(vol[i]), float(r[i]), float(q[i]), int(ot[i]),
                       int(paths[i]), int(seed_arr[i]), float(weight_clip[i])) for i in range(n)])
    assert np.allclose(batch, scalar, atol=1e-12, rtol=1e-12)

    tape_steps = np.full(n, 64, dtype=np.int32)
    regularization = np.full(n, 1e-6)
    batch = qk.aad_delta_batch(spot, strike, tau, vol, r, q, ot, tape_steps, regularization)
    scalar = np.array([qk.aad_delta(float(spot[i]), float(strike[i]), float(tau[i]),
                       float(vol[i]), float(r[i]), float(q[i]), int(ot[i]),
                       int(tape_steps[i]), float(regularization[i])) for i in range(n)])
    assert np.allclose(batch, scalar, atol=1e-12, rtol=1e-12)


def test_mlm_batch_apis(qk):
    rng = np.random.default_rng(900)
    n = 8
    spot = rng.uniform(90.0, 110.0, n)
    strike = rng.uniform(90.0, 110.0, n)
    tau = rng.uniform(0.25, 1.0, n)
    vol = rng.uniform(0.15, 0.4, n)
    r = rng.uniform(0.01, 0.05, n)
    q = rng.uniform(0.0, 0.02, n)
    ot = np.where((np.arange(n) & 1) == 0, QK_CALL, QK_PUT).astype(np.int32)

    time_steps = np.full(n, 10, dtype=np.int32)
    hidden_width = np.full(n, 16, dtype=np.int32)
    training_epochs = np.full(n, 20, dtype=np.int32)
    learning_rate = np.full(n, 5e-3)
    batch = qk.deep_bsde_price_batch(spot, strike, tau, vol, r, q, ot, time_steps, hidden_width, training_epochs, learning_rate)
    scalar = np.array([qk.deep_bsde_price(float(spot[i]), float(strike[i]), float(tau[i]),
                       float(vol[i]), float(r[i]), float(q[i]), int(ot[i]),
                       int(time_steps[i]), int(hidden_width[i]), int(training_epochs[i]),
                       float(learning_rate[i])) for i in range(n)])
    assert np.allclose(batch, scalar, atol=1e-12, rtol=1e-12)

    collocation_points = np.full(n, 500, dtype=np.int32)
    boundary_points = np.full(n, 50, dtype=np.int32)
    epochs = np.full(n, 30, dtype=np.int32)
    loss_balance = np.full(n, 1.0)
    batch = qk.pinns_price_batch(spot, strike, tau, vol, r, q, ot, collocation_points, boundary_points, epochs, loss_balance)
    scalar = np.array([qk.pinns_price(float(spot[i]), float(strike[i]), float(tau[i]),
                       float(vol[i]), float(r[i]), float(q[i]), int(ot[i]),
                       int(collocation_points[i]), int(boundary_points[i]), int(epochs[i]),
                       float(loss_balance[i])) for i in range(n)])
    assert np.allclose(batch, scalar, atol=1e-12, rtol=1e-12)

    rehedge_steps = np.full(n, 10, dtype=np.int32)
    risk_aversion = np.full(n, 0.5)
    scenarios = np.full(n, 1000, dtype=np.int32)
    seed_arr = (np.arange(n, dtype=np.uint64) + np.uint64(42))
    batch = qk.deep_hedging_price_batch(spot, strike, tau, vol, r, q, ot, rehedge_steps, risk_aversion, scenarios, seed_arr)
    scalar = np.array([qk.deep_hedging_price(float(spot[i]), float(strike[i]), float(tau[i]),
                       float(vol[i]), float(r[i]), float(q[i]), int(ot[i]),
                       int(rehedge_steps[i]), float(risk_aversion[i]), int(scenarios[i]),
                       int(seed_arr[i])) for i in range(n)])
    assert np.allclose(batch, scalar, atol=1e-12, rtol=1e-12)

    target_implied_vol = np.full(n, 0.2)
    calibration_steps = np.full(n, 50, dtype=np.int32)
    regularization = np.full(n, 1e-3)
    batch = qk.neural_sde_calibration_price_batch(spot, strike, tau, vol, r, q, ot, target_implied_vol, calibration_steps, regularization)
    scalar = np.array([qk.neural_sde_calibration_price(float(spot[i]), float(strike[i]), float(tau[i]),
                       float(vol[i]), float(r[i]), float(q[i]), int(ot[i]),
                       float(target_implied_vol[i]), int(calibration_steps[i]),
                       float(regularization[i])) for i in range(n)])
    assert np.allclose(batch, scalar, atol=1e-12, rtol=1e-12)

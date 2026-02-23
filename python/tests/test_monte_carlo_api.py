"""Monte Carlo API checks."""

import math

from quantkernel import QK_CALL, QK_PUT


def test_monte_carlo_methods_return_finite_prices(qk):
    common = dict(spot=100.0, strike=100.0, t=1.0, vol=0.2, r=0.03, q=0.01, option_type=QK_CALL)

    prices = [
        qk.standard_monte_carlo_price(**common, paths=30000, seed=7),
        qk.euler_maruyama_price(**common, paths=12000, steps=64, seed=7),
        qk.milstein_price(**common, paths=12000, steps=64, seed=7),
        qk.longstaff_schwartz_price(**common, paths=10000, steps=40, seed=7),
        qk.quasi_monte_carlo_sobol_price(**common, paths=16384),
        qk.quasi_monte_carlo_halton_price(**common, paths=16384),
        qk.multilevel_monte_carlo_price(**common, base_paths=8192, levels=4, base_steps=8, seed=7),
        qk.importance_sampling_price(**common, paths=30000, shift=0.4, seed=7),
        qk.control_variates_price(**common, paths=30000, seed=7),
        qk.antithetic_variates_price(**common, paths=30000, seed=7),
        qk.stratified_sampling_price(**common, paths=30000, seed=7),
    ]

    assert all(math.isfinite(p) and p > 0.0 for p in prices)


def test_monte_carlo_prices_are_consistent_with_bsm(qk):
    common = dict(spot=100.0, strike=100.0, t=1.0, vol=0.2, r=0.03, q=0.01, option_type=QK_CALL)
    bsm = qk.black_scholes_merton_price(**common)

    mc = qk.standard_monte_carlo_price(**common, paths=60000, seed=11)
    anti = qk.antithetic_variates_price(**common, paths=60000, seed=11)
    ctrl = qk.control_variates_price(**common, paths=60000, seed=11)
    strat = qk.stratified_sampling_price(**common, paths=60000, seed=11)

    assert abs(mc - bsm) < 0.8
    assert abs(anti - bsm) < 0.6
    assert abs(ctrl - bsm) < 0.4
    assert abs(strat - bsm) < 0.4


def test_lsmc_american_put_not_below_european_mc_put(qk):
    common_put = dict(spot=100.0, strike=110.0, t=1.0, vol=0.25, r=0.04, q=0.0, option_type=QK_PUT)

    european_est = qk.standard_monte_carlo_price(**common_put, paths=50000, seed=19)
    american_est = qk.longstaff_schwartz_price(**common_put, paths=20000, steps=50, seed=19)

    assert american_est >= european_est - 0.25


def test_quasi_monte_carlo_variants_are_close(qk):
    common = dict(spot=100.0, strike=100.0, t=1.0, vol=0.2, r=0.03, q=0.01, option_type=QK_CALL)

    sobol = qk.quasi_monte_carlo_sobol_price(**common, paths=16384)
    halton = qk.quasi_monte_carlo_halton_price(**common, paths=16384)

    assert abs(sobol - halton) < 0.75


def test_heston_monte_carlo_price_is_finite_and_close_to_cf(qk):
    """Heston MC should produce a finite price reasonably close to Heston CF."""
    heston_params = dict(
        spot=100.0, strike=100.0, t=1.0, r=0.05, q=0.02,
        v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7,
        option_type=QK_CALL,
    )
    mc_price = qk.heston_monte_carlo_price(**heston_params, paths=50000, steps=100, seed=42)
    cf_price = qk.heston_price_cf(**heston_params)

    assert math.isfinite(mc_price) and mc_price > 0.0
    assert math.isfinite(cf_price) and cf_price > 0.0
    # MC with 50k paths, 100 steps should be within ~1.0 of CF
    assert abs(mc_price - cf_price) < 1.5, (
        f"Heston MC ({mc_price:.4f}) too far from CF ({cf_price:.4f})"
    )


def test_heston_monte_carlo_price_batch_matches_scalar(qk):
    """Heston MC batch should produce identical results to scalar calls."""
    import numpy as np
    params = [
        dict(spot=100.0, strike=100.0, t=1.0, r=0.05, q=0.02,
             v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7,
             option_type=QK_CALL, paths=5000, steps=50, seed=42),
        dict(spot=110.0, strike=100.0, t=0.5, r=0.03, q=0.01,
             v0=0.09, kappa=1.5, theta=0.06, sigma=0.4, rho=-0.5,
             option_type=QK_PUT, paths=5000, steps=50, seed=99),
    ]
    scalars = [qk.heston_monte_carlo_price(**p) for p in params]
    batch = qk.heston_monte_carlo_price_batch(
        **{k: [p[k] for p in params] for k in params[0]}
    )
    for i, s in enumerate(scalars):
        assert abs(batch[i] - s) < 1e-12, f"batch[{i}]={batch[i]}, scalar={s}"


def test_heston_monte_carlo_deterministic_seed(qk):
    """Same seed must produce bit-identical results."""
    common = dict(
        spot=100.0, strike=100.0, t=1.0, r=0.05, q=0.02,
        v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7,
        option_type=QK_CALL, paths=10000, steps=50, seed=777,
    )
    p1 = qk.heston_monte_carlo_price(**common)
    p2 = qk.heston_monte_carlo_price(**common)
    assert p1 == p2


def test_antithetic_uses_all_requested_paths(qk):
    common = dict(
        spot=100.0,
        strike=100.0,
        t=1.0,
        vol=0.2,
        r=0.03,
        q=0.01,
        option_type=QK_CALL,
        seed=123,
    )

    even = qk.antithetic_variates_price(**common, paths=2)
    odd = qk.antithetic_variates_price(**common, paths=3)

    assert abs(odd - even) > 1e-12

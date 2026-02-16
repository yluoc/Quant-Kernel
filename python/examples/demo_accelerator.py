"""QuantAccelerator demo: rule-based batch pricing.

Run after building the shared library:
    PYTHONPATH=python python3 python/examples/demo_accelerator.py
"""

from quantkernel import QK_CALL, QuantAccelerator, QuantKernel


def main() -> None:
    qk = QuantKernel()
    accel = QuantAccelerator(qk=qk, backend="auto")

    bsm_jobs = [
        {"spot": 100.0, "strike": 100.0, "t": 1.0, "vol": 0.2, "r": 0.03, "q": 0.01, "option_type": QK_CALL},
        {"spot": 95.0, "strike": 100.0, "t": 0.5, "vol": 0.25, "r": 0.02, "q": 0.00, "option_type": QK_CALL},
        {"spot": 105.0, "strike": 100.0, "t": 2.0, "vol": 0.18, "r": 0.01, "q": 0.02, "option_type": QK_CALL},
    ]

    strat = accel.suggest_strategy("black_scholes_merton_price", len(bsm_jobs))
    prices = accel.price_batch("black_scholes_merton_price", bsm_jobs)
    print("BSM strategy:", strat)
    print("BSM prices:", prices)

    heston_jobs = [
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
            "integration_steps": 1024,
            "integration_limit": 120.0,
        }
        for _ in range(64)
    ]

    strat = accel.suggest_strategy("heston_price_cf", len(heston_jobs))
    prices = accel.price_batch("heston_price_cf", heston_jobs)
    print("Heston strategy:", strat)
    print("Heston first 3 prices:", prices[:3])


if __name__ == "__main__":
    main()

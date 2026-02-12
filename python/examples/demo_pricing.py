#!/usr/bin/env python3
"""QuantKernel demo — Black-Scholes pricing and implied vol recovery."""

import time
import numpy as np
from quantkernel import QuantKernel

def main():
    qk = QuantKernel()

    # --- Single option pricing ---
    print("=" * 60)
    print("QuantKernel Demo — Black-Scholes Pricing Engine")
    print("=" * 60)

    print("\n1. Single ATM call option:")
    print("   S=100, K=100, T=1y, vol=20%, r=5%, q=0%\n")

    result = qk.bs_price(
        spot=np.array([100.0]),
        strike=np.array([100.0]),
        time_to_expiry=np.array([1.0]),
        volatility=np.array([0.20]),
        risk_free_rate=np.array([0.05]),
        dividend_yield=np.array([0.0]),
        option_type=np.array([QuantKernel.CALL], dtype=np.int32),
    )
    print(f"   Price:  {result['price'][0]:.4f}")
    print(f"   Delta:  {result['delta'][0]:.4f}")
    print(f"   Gamma:  {result['gamma'][0]:.6f}")
    print(f"   Vega:   {result['vega'][0]:.4f}")
    print(f"   Theta:  {result['theta'][0]:.6f}")
    print(f"   Rho:    {result['rho'][0]:.4f}")

    # --- Implied vol round-trip ---
    print("\n2. Implied vol round-trip:")
    print("   Pricing with vol=25%, then recovering vol from the price\n")

    bs_result = qk.bs_price(
        spot=np.array([100.0]),
        strike=np.array([100.0]),
        time_to_expiry=np.array([1.0]),
        volatility=np.array([0.25]),
        risk_free_rate=np.array([0.05]),
        dividend_yield=np.array([0.0]),
        option_type=np.array([QuantKernel.CALL], dtype=np.int32),
    )
    mkt_price = bs_result["price"]
    print(f"   BS price (vol=25%): {mkt_price[0]:.4f}")

    iv_result = qk.iv_solve(
        spot=np.array([100.0]),
        strike=np.array([100.0]),
        time_to_expiry=np.array([1.0]),
        risk_free_rate=np.array([0.05]),
        dividend_yield=np.array([0.0]),
        option_type=np.array([QuantKernel.CALL], dtype=np.int32),
        market_price=mkt_price,
    )
    print(f"   Recovered IV:       {iv_result['implied_vol'][0]:.6f}")
    print(f"   Iterations:         {iv_result['iterations'][0]}")

    # --- Batch performance benchmark ---
    print("\n3. Batch pricing benchmark:")
    for n in [1_000, 10_000, 100_000, 1_000_000]:
        spots = np.full(n, 100.0)
        strikes = np.linspace(80, 120, n)
        times = np.full(n, 1.0)
        vols = np.full(n, 0.20)
        rates = np.full(n, 0.05)
        divs = np.full(n, 0.0)
        types = np.zeros(n, dtype=np.int32)

        t0 = time.perf_counter()
        result = qk.bs_price(
            spot=spots, strike=strikes, time_to_expiry=times,
            volatility=vols, risk_free_rate=rates,
            dividend_yield=divs, option_type=types,
        )
        elapsed = time.perf_counter() - t0
        rate = n / elapsed
        print(f"   n={n:>10,}  time={elapsed*1000:8.2f} ms  ({rate/1e6:.1f}M opts/sec)")

    # --- IV solver benchmark ---
    print("\n4. IV solver benchmark:")
    for n in [1_000, 10_000, 100_000]:
        spots = np.full(n, 100.0)
        strikes = np.linspace(80, 120, n)
        times = np.full(n, 1.0)
        rates = np.full(n, 0.05)
        divs = np.full(n, 0.0)
        types = np.zeros(n, dtype=np.int32)
        vols = np.full(n, 0.25)

        bs = qk.bs_price(
            spot=spots, strike=strikes, time_to_expiry=times,
            volatility=vols, risk_free_rate=rates,
            dividend_yield=divs, option_type=types,
        )

        t0 = time.perf_counter()
        iv = qk.iv_solve(
            spot=spots, strike=strikes, time_to_expiry=times,
            risk_free_rate=rates, dividend_yield=divs,
            option_type=types, market_price=bs["price"],
        )
        elapsed = time.perf_counter() - t0
        rate = n / elapsed
        max_err = np.max(np.abs(iv["implied_vol"] - 0.25))
        print(f"   n={n:>10,}  time={elapsed*1000:8.2f} ms  "
              f"({rate/1e6:.2f}M/sec)  max_err={max_err:.1e}")

    print("\nDone.")


if __name__ == "__main__":
    main()

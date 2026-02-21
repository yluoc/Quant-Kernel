# QuantKernel

QuantKernel is a C++17 quantitative pricing kernel with Python bindings.
It focuses on fast scalar and batch option analytics across closed-form, lattice,
finite-difference, Monte Carlo, Fourier, quadrature, regression-approximation,
Greek-estimation, and ML-inspired methods.

Linux, macOS, and Windows are supported.

## Scope

QuantKernel provides:
- C++ shared library (`libquantkernel.so` / `libquantkernel.dylib` / `libquantkernel.dll`) with C ABI exports.
- Python package (`quantkernel`) with scalar and batch methods.
- Optional Python-level accelerator (`QuantAccelerator`) for backend selection (`auto`, `cpu`, `gpu`).

## Install

End users (recommended):

```bash
python -m pip install --upgrade pip
python -m pip install quant-kernel
```

This installs a prebuilt wheel on supported platforms and does not require local C++ compilation.

## Implemented Algorithm Families

### Closed-form / Semi-analytical
- Black-Scholes-Merton
- Black-76
- Bachelier
- Heston characteristic-function pricing
- Merton jump-diffusion
- Variance-Gamma characteristic-function pricing
- SABR (Hagan lognormal IV + Black-76 pricing)
- Dupire local volatility inversion

### Tree / Lattice
- CRR
- Jarrow-Rudd
- Tian
- Leisen-Reimer
- Trinomial tree
- Derman-Kani style local-vol tree entrypoints:
  - Constant local vol surface (`derman_kani_const_local_vol_price`)
  - Vanilla call-surface driven entrypoint (`derman_kani_call_surface_price`)

### Finite Difference
- Explicit FD
- Implicit FD
- Crank-Nicolson
- ADI (Douglas, Craig-Sneyd, Hundsdorfer-Verwer)
- PSOR

### Monte Carlo
- Standard Monte Carlo
- Euler-Maruyama
- Milstein
- Longstaff-Schwartz
- Quasi Monte Carlo (Sobol, Halton)
- Multilevel Monte Carlo
- Importance Sampling
- Control Variates
- Antithetic Variates
- Stratified Sampling

### Fourier Transform Methods
- Carr-Madan FFT
- COS (Fang-Oosterlee)
- Fractional FFT
- Lewis Fourier inversion
- Hilbert transform pricing

### Integral Quadrature
- Gauss-Hermite
- Gauss-Laguerre
- Gauss-Legendre
- Adaptive quadrature

### Regression Approximation
- Polynomial Chaos Expansion
- Radial Basis Functions
- Sparse Grid Collocation
- Proper Orthogonal Decomposition

### Greeks / Adjoint Methods
- Pathwise derivative delta
- Likelihood ratio delta
- AAD delta

### Machine-learning Inspired Pricing
- Deep BSDE
- PINNs
- Deep Hedging
- Neural SDE calibration

## Repository Layout

- `cpp/`
  - `include/quantkernel/qk_api.h`: C API declarations
  - `src/`: implementations and API bridge (`qk_api.cpp`)
  - `tests/`: C++ test executables
- `python/`
  - `quantkernel/`: Python API (`QuantKernel`, `QuantAccelerator`)
  - `tests/`: pytest suite
  - `examples/`: usage and benchmark scripts
- `Makefile`: common build/test commands

## Requirements

- CMake >= 3.14
- C++17 compiler
- Python >= 3.11
- NumPy

Optional:
- CuPy (for GPU backend in accelerator paths)

## Build

From project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

cmake -S . -B build
cmake --build build -j
```

## Python Setup (from source checkout)

Point Python to the package and shared library:

```bash
export PYTHONPATH=$PWD/python
export QK_LIB_PATH=$PWD/build/cpp
```

Then use:

```python
from quantkernel import QuantKernel, QK_CALL

qk = QuantKernel()
price = qk.black_scholes_merton_price(
    100.0, 100.0, 1.0, 0.2, 0.03, 0.01, QK_CALL
)
print(price)
```

## Batch Usage

```python
import numpy as np
from quantkernel import QuantKernel, QK_CALL, QK_PUT

qk = QuantKernel()
n = 100_000
rng = np.random.default_rng(42)

spot = rng.uniform(80.0, 120.0, n)
strike = rng.uniform(80.0, 120.0, n)
t = rng.uniform(0.25, 2.0, n)
vol = rng.uniform(0.1, 0.6, n)
r = rng.uniform(0.0, 0.08, n)
q = rng.uniform(0.0, 0.04, n)
option_type = np.where((np.arange(n) & 1) == 0, QK_CALL, QK_PUT).astype(np.int32)

prices = qk.black_scholes_merton_price_batch(spot, strike, t, vol, r, q, option_type)
print(prices[:3])
```

## Derman-Kani Call-Surface API (Python)

`derman_kani_call_surface_price` accepts:
- `surface_strikes`: 1D strikes
- `surface_maturities`: 1D maturities
- `surface_call_prices`:
  - 2D array with shape `(len(surface_maturities), len(surface_strikes))`, or
  - flattened 1D array of that size

Example:

```python
from quantkernel import QuantKernel, QK_CALL

qk = QuantKernel()
spot, r, q = 100.0, 0.03, 0.01
surface_strikes = [80, 90, 100, 110, 120]
surface_maturities = [0.5, 1.0, 1.5]

# Synthetic surface here; in production use observed call prices.
surface_call_prices = [
    [qk.black_scholes_merton_price(spot, k, tau, 0.2, r, q, QK_CALL) for k in surface_strikes]
    for tau in surface_maturities
]

price = qk.derman_kani_call_surface_price(
    spot=spot,
    strike=100.0,
    t=1.0,
    r=r,
    q=q,
    option_type=QK_CALL,
    surface_strikes=surface_strikes,
    surface_maturities=surface_maturities,
    surface_call_prices=surface_call_prices,
    steps=20,
)
print(price)
```

If `QK_LIB_PATH` is unset, the package also searches for a bundled shared library from an installed wheel.

## Testing

From project root:

```bash
make test-cpp
make test-py
# or
make quick
```

Direct commands:

```bash
ctest --test-dir build --output-on-failure
PYTHONPATH=python QK_LIB_PATH=build/cpp pytest -q python/tests
```

## Benchmark

```bash
PYTHONPATH=python QK_LIB_PATH=build/cpp \
python3 python/examples/benchmark_scalar_batch_cpp.py --n 50000 --repeats 3
```

## Error Handling

### C API
- Batch functions return ABI error codes (`QK_OK`, `QK_ERR_NULL_PTR`, `QK_ERR_BAD_SIZE`, `QK_ERR_INVALID_INPUT`, etc.).
- Use `qk_get_last_error()` for thread-local error detail.

### Python API
- Raises typed exceptions:
  - `QKError`
  - `QKNullPointerError`
  - `QKBadSizeError`
  - `QKInvalidInputError`

## License

`LICENSE` (WTFPL).

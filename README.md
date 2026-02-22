# QuantKernel

QuantKernel is a C++17 quantitative option pricing kernel with Python bindings. It provides scalar and batch evaluation of European option prices and Greeks across 40+ algorithms spanning closed-form, lattice, finite difference, Monte Carlo, Fourier, quadrature, regression, and machine learning methods.

The core is a single shared library (`libquantkernel.so` / `.dylib`) with a flat C ABI. The Python package (`quantkernel`) loads this library via ctypes. There are no external C++ dependencies.

QuantKernel is not a full term-structure framework, not a risk management system, and not a replacement for QuantLib. It is a focused, low-overhead pricing kernel.

## Architecture

```
Python (quantkernel)
  |
  | ctypes FFI
  v
C ABI boundary  (qk_api.h — flat extern "C" functions)
  |
  v
C++17 internals
  |-- Algorithm families  (closed-form, tree, FD, MC, Fourier, ...)
  |-- Model layer         (model_concepts.h — callable factories)
  |-- MC engine layer     (mc_engine.h — templated simulation loops)
  |-- Common utilities    (math, payoff, validation)
```

**Model layer.** Stochastic dynamics are expressed as callable factories that return lightweight lambdas. `make_bsm_terminal(vol, r, q)` returns a `(spot, t, z) -> S_T` functor implementing the GBM log-normal terminal distribution. Step models (`make_bsm_euler_step`, `make_bsm_milstein_step`) return `(s, dt, dw) -> s_next` functors. Sensitivity callables (`make_bsm_pathwise_dST_dSpot`, `make_bsm_lr_score`) are similarly factored. All callables are validated at compile time via `static_assert` traits.

**Engine layer.** The shared Monte Carlo engine (`mc_engine.h`) provides templated simulation loops — `estimate_terminal`, `estimate_terminal_antithetic`, and `estimate_stepwise` — parameterized over a normal-variate generator, a model callable, and an accumulator. The engine owns no RNG state; callers inject a generator via `mc::make_mt19937_normal(seed)` or any `() -> double` callable. This permits future substitution of Sobol, Philox, or other RNG strategies without modifying the engine.

**C ABI boundary.** All public functions are `extern "C"` with `QK_EXPORT` visibility. Scalar functions return `double` (NaN on error). Batch functions return `int32_t` error codes (`QK_OK`, `QK_ERR_NULL_PTR`, `QK_ERR_BAD_SIZE`, `QK_ERR_INVALID_INPUT`). Thread-local error detail is available via `qk_get_last_error()`. ABI versioning is enforced at load time.

**Python wrapper.** The `QuantKernel` class exposes every C function as a Python method. Batch methods accept NumPy arrays. An optional `QuantAccelerator` class provides CuPy-based GPU vectorization for large batches.

## Supported Models

The stochastic dynamics layer currently implements **Black-Scholes-Merton (geometric Brownian motion)** only. The model-factory pattern is designed so that additional models (Heston, local volatility, rough volatility) can be added as new callable factories without modifying the engine or algorithm code.

## Supported Payoffs

All Monte Carlo and adjoint Greek estimators evaluate **vanilla European payoffs** only: `max(S_T - K, 0)` for calls, `max(K - S_T, 0)` for puts. Tree and FD methods support American exercise. Barrier, Asian, and other exotic payoffs are not implemented.

## Algorithm Families

### Closed-Form / Semi-Analytical
Black-Scholes-Merton, Black-76, Bachelier, Heston (characteristic function), Merton jump-diffusion, Variance-Gamma (characteristic function), SABR (Hagan lognormal IV + Black-76), Dupire local volatility inversion.

### Tree / Lattice
CRR, Jarrow-Rudd, Tian, Leisen-Reimer, trinomial tree, Derman-Kani implied tree (constant local vol and call-surface-driven variants).

### Finite Difference
Explicit FD, implicit FD, Crank-Nicolson, ADI (Douglas, Craig-Sneyd, Hundsdorfer-Verwer), PSOR (American exercise).

### Monte Carlo
Standard MC (antithetic), Euler-Maruyama, Milstein, Longstaff-Schwartz (American), quasi-MC (Sobol, Halton), multilevel MC, importance sampling, control variates, antithetic variates, stratified sampling.

### Fourier Transform
Carr-Madan FFT, COS (Fang-Oosterlee), fractional FFT, Lewis Fourier inversion, Hilbert transform.

### Integral Quadrature
Gauss-Hermite, Gauss-Laguerre, Gauss-Legendre, adaptive quadrature.

### Regression Approximation
Polynomial chaos expansion, radial basis functions, sparse grid collocation, proper orthogonal decomposition.

### Greeks (Adjoint Methods)
Pathwise derivative delta, likelihood-ratio delta, BSM adjoint analytic delta (regularized).

### Machine Learning
Deep BSDE, PINNs, deep hedging, neural SDE calibration.

## Monte Carlo Design

The MC subsystem is structured as three independent layers:

1. **RNG policy.** The caller constructs a generator — typically `mc::make_mt19937_normal(seed)` — and passes it into the engine. The engine calls `gen()` to draw standard-normal variates. This decouples the RNG from the simulation loop.

2. **Model callable.** A functor mapping variates to asset prices. Terminal models map `(spot, t, z) -> S_T`. Step models map `(s, dt, dw) -> s_next`.

3. **Engine loop.** `estimate_terminal` runs a simple forward loop. `estimate_terminal_antithetic` pairs `+z` and `-z` draws for variance reduction. `estimate_stepwise` runs multi-step Euler/Milstein paths. All are header-only templates that inline through the callable indirection under any reasonable optimization level.

The accumulator receives `(S_T, z, path_index)` for terminal engines and `(S_T, path_index)` for stepwise engines, providing the metadata needed for variance reduction and sensitivity estimation.

## Greeks

**Pathwise derivative.** Differentiates the payoff indicator directly. For GBM, `dS_T / dSpot = S_T / spot`. Requires the payoff to be almost-everywhere differentiable (digital options are excluded).

**Likelihood ratio.** Differentiates the log-density instead of the payoff. Uses the score function `z / (vol * sqrt(t) * spot)` with configurable weight clipping. Works with discontinuous payoffs but has higher variance. Uses antithetic pairing.

**BSM adjoint delta.** Despite the `aad_delta` function name, this is a closed-form BSM delta computed via hand-written reverse-mode differentiation of the Black-Scholes formula, with Tikhonov regularization toward the ATM delta prior. It is not a general-purpose tape-based automatic differentiation engine. The function name is preserved for ABI compatibility. Internal C++ code can use the alias `bsm_adjoint_delta()`.

## Performance

- No virtual dispatch in hot paths. Model callables are monomorphized templates.
- No heap allocation in simulation loops. RNG state is stack-local.
- Header-only engine and model layers. Under LTO (enabled in Release builds), all lambda indirection is eliminated.
- Batch C API functions iterate over input arrays with no per-call overhead beyond the computation itself.
- OpenMP support is linked when available.

## Reproducibility

All Monte Carlo estimators produce deterministic, bit-reproducible results for a given seed on the same platform. The RNG is `std::mt19937_64` seeded by the caller. Draw order is fixed by the engine loop structure. Golden-seed regression tests pin absolute numerical outputs at 1e-12 tolerance to detect any changes in draw order, antithetic pairing, or accumulation behavior.

## Installation

From PyPI (recommended):

```bash
pip install quant-kernel
```

This installs a prebuilt wheel on supported platforms (Linux, macOS). No local C++ compilation required.

Python >= 3.11 and NumPy >= 1.24 are required. CuPy >= 12.0 is optional (GPU acceleration).

## Build from Source

```bash
cmake -S . -B build
cmake --build build -j
```

Requires CMake >= 3.20 and a C++17 compiler.

```bash
make quick        # configure + build + C++ tests + Python tests
make test-cpp     # C++ tests only
make test-py      # Python tests only
make bench        # benchmark (50k samples)
```

For development use, point Python to the local build:

```bash
export PYTHONPATH=$PWD/python
export QK_LIB_PATH=$PWD/build/cpp
```

## Python Usage

```python
from quantkernel import QuantKernel, QK_CALL, QK_PUT

qk = QuantKernel()

# Scalar pricing
price = qk.black_scholes_merton_price(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, QK_CALL)

# Batch pricing
import numpy as np
spot = np.array([100.0, 105.0, 95.0])
strike = np.full(3, 100.0)
t = np.full(3, 1.0)
vol = np.full(3, 0.2)
r = np.full(3, 0.05)
q = np.full(3, 0.0)
ot = np.full(3, QK_CALL, dtype=np.int32)

prices = qk.black_scholes_merton_price_batch(spot, strike, t, vol, r, q, ot)

# Monte Carlo
mc_price = qk.standard_monte_carlo_price(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, QK_CALL, 100000, 42)

# Greeks
delta = qk.pathwise_derivative_delta(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, QK_CALL, 50000, 42)
```

## C Usage

```c
#include <quantkernel/qk_api.h>
#include <stdio.h>

int main() {
    double price = qk_cf_black_scholes_merton_price(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, QK_CALL);
    printf("BSM call price: %.6f\n", price);

    double mc = qk_mcm_standard_monte_carlo_price(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, QK_CALL, 100000, 42);
    printf("MC call price:  %.6f\n", mc);

    double delta = qk_agm_pathwise_derivative_delta(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, QK_CALL, 50000, 42);
    printf("Pathwise delta: %.6f\n", delta);

    return 0;
}
```

Link against `-lquantkernel`.

## Cautions

- **BSM dynamics only.** The Monte Carlo model layer currently implements geometric Brownian motion. Heston, local vol, and other stochastic volatility models are not yet available through the shared engine.

- **`aad_delta` is not general AAD.** The function named `qk_agm_aad_delta` computes BSM delta via hand-differentiated closed-form Black-Scholes with regularization. It does not implement a computational tape, operator overloading, or any general-purpose automatic differentiation framework.

- **Vanilla European payoffs only** in the MC and adjoint Greek subsystems. Barrier, Asian, lookback, and other path-dependent payoffs are not supported. Longstaff-Schwartz supports American exercise as a special case.

- **Not a term-structure framework.** There is no yield curve construction, no vol surface interpolation infrastructure, no calendar/day-count conventions. Input rates and volatilities are flat scalars.

- **Platform-dependent reproducibility.** Bit-exact MC outputs are guaranteed for a given seed on the same platform and compiler. Cross-platform reproducibility (e.g., Linux vs macOS, GCC vs Clang) is not guaranteed due to differences in `std::normal_distribution` implementations.

- **Sample Option data source download** To run `examples/validate_.../py` files, download sample options pricing data source from this url: [optionsDX](https://www.optionsdx.com/option-chain-field-definitions/)

## Error Handling

**C API.** Scalar functions return NaN on invalid input. Batch functions return error codes: `QK_OK` (0), `QK_ERR_NULL_PTR` (-1), `QK_ERR_BAD_SIZE` (-2), `QK_ERR_INVALID_INPUT` (-5). Call `qk_get_last_error()` for a human-readable error string.

**Python API.** Raises typed exceptions: `QKError`, `QKNullPointerError`, `QKBadSizeError`, `QKInvalidInputError`.

## Goals

- Fast, correct scalar and batch European option pricing.
- Minimal-dependency C++ core suitable for embedding.
- Stable C ABI for language-agnostic integration.
- Extensible internal architecture for future multi-model support.

## Non-Goals

- General-purpose risk engine or portfolio management.
- Exotic payoff coverage.
- Term-structure or vol surface construction.
- Real-time market data integration.

## License

WTFPL v2. See `LICENSE`.

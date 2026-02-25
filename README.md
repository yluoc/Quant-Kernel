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
  |-- Model layer         (model_concepts.h — callable factories for MC)
  |-- CharFn layer        (charfn_concepts.h — characteristic function factories for Fourier)
  |-- MC engine layer     (mc_engine.h — templated simulation loops)
  |-- Common utilities    (math, payoff, validation)
```

**Model layer (MC).** Stochastic dynamics for Monte Carlo are expressed as callable factories that return lightweight lambdas. `make_bsm_terminal(vol, r, q)` returns a `(spot, t, z) -> S_T` functor implementing the GBM log-normal terminal distribution. Step models (`make_bsm_euler_step`, `make_bsm_milstein_step`) return `(s, dt, dw) -> s_next` functors. Two-dimensional step models (`make_heston_euler_step_2d`) return `(s, v, dt, dw1, dw2) -> (s_next, v_next)` for correlated multi-factor dynamics. Local volatility step models (`make_local_vol_euler_step`) accept a callable `sigma(s, t)` and produce `(s, t, dt, dw) -> s_next` functors with time-dependent diffusion. Sensitivity callables (`make_bsm_pathwise_dST_dSpot`, `make_bsm_lr_score`) are similarly factored. All callables are validated at compile time via `static_assert` traits (`is_terminal_model_v`, `is_step_model_v`, `is_step_model_2d_v`).

**CharFn layer (Fourier).** Characteristic functions for Fourier pricing are expressed as callable factories returning `(complex<double> u, double t) -> complex<double>` lambdas, validated by the `is_charfn_v<F>` trait. Each Fourier method has a template `_impl` function parameterized on a `CharFn` type. BSM and Heston factories (`make_bsm_log_charfn`, `make_heston_log_charfn`, and their log-return variants) capture model parameters at construction, keeping the hot-loop call to `phi(u, t)` only. Since `CharFn` is a template parameter, the compiler inlines the characteristic function into the integration loop — no `std::function`, no vtable, no heap allocation.

**Engine layer.** The shared Monte Carlo engine (`mc_engine.h`) provides templated simulation loops — `estimate_terminal`, `estimate_terminal_antithetic`, `estimate_stepwise`, and `estimate_stepwise_2d` — parameterized over a normal-variate generator, a model callable, and an accumulator. The engine owns no RNG state; callers inject a generator via `mc::make_mt19937_normal(seed)` or any `() -> double` callable.

**C ABI boundary.** All public functions are `extern "C"` with `QK_EXPORT` visibility. Scalar functions return `double` (NaN on error). Batch functions return `int32_t` error codes (`QK_OK`, `QK_ERR_NULL_PTR`, `QK_ERR_BAD_SIZE`, `QK_ERR_INVALID_INPUT`). Thread-local error detail is available via `qk_get_last_error()`. ABI versioning is enforced at load time. The current ABI version is 2.12. Minor bumps are additive only; existing function signatures are never changed or removed within a major version.

**Python wrapper.** The `QuantKernel` class exposes every C function as a Python method. Batch methods accept NumPy arrays. An optional `QuantAccelerator` class provides CuPy-based GPU vectorization for large batches.

## Supported Models

The pricing kernel supports multiple stochastic dynamics models, with availability varying by algorithm family:

- **Black-Scholes-Merton (GBM).** Available across all algorithm families: closed-form, tree, FD, MC, Fourier, quadrature, regression, ML, and adjoint Greeks.
- **Heston stochastic volatility.** Available via closed-form characteristic function integration, Monte Carlo (2D correlated Euler), all five Fourier transform methods (via CharFn), and ADI finite difference.
- **Local volatility.** Available via Monte Carlo (Euler discretization with callable `sigma(S, t)`) and Dupire local vol inversion (closed-form). The MC local vol pricer currently accepts a constant-vol callback at the C ABI level; user-defined `sigma(S, t)` surfaces require C++ template instantiation.
- **Merton jump-diffusion.** Closed-form only.
- **Variance-Gamma.** Closed-form characteristic function integration only.
- **SABR.** Hagan lognormal IV approximation + Black-76 pricing only.

Not all models are available in all algorithm families. See the capability matrix below.

## Model Interface by Algorithm Family

Each algorithm family uses a different abstraction for model dynamics:

- **Monte Carlo** uses **path simulation model factories** — callable objects that advance asset prices one step or to terminal time. Models are expressed as lambdas satisfying compile-time traits (`is_terminal_model_v`, `is_step_model_v`, `is_step_model_2d_v`). Adding a new model requires writing a factory function that returns a conforming lambda; the engine and accumulator code remain unchanged.

- **Fourier transform methods** use the **CharFn concept** — callable objects mapping `(complex<double> u, double t) -> complex<double>` representing the characteristic function of the log-price or log-return process. Each Fourier method has a template `_impl` that accepts any `CharFn`. Adding a new model (e.g., Variance-Gamma, CGMY) requires only writing a `make_MODEL_charfn(...)` factory.

- **Finite difference methods** use **PDE operators** that define the drift, diffusion, and mixed derivative coefficients on a discretized grid. BSM methods use a 1D grid; Heston ADI methods use a 2D (spot, variance) grid with operator splitting.

- **Tree / lattice methods** use **lattice parameterizations** — up/down multipliers and transition probabilities derived from the model's first and second moments. Each tree model (CRR, Jarrow-Rudd, Tian, etc.) hardcodes its parameterization. The Derman-Kani implied tree is driven by an external call price surface.

- **Regression and ML methods** are currently BSM-only. Their internal model coupling is not factored into a pluggable interface.

Model extensibility differs by family. MC and Fourier have explicit, documented extension points. FD requires new PDE operator implementations. Trees require new lattice parameterizations. Regression and ML methods would require internal refactoring to support non-BSM dynamics.

## Capability Matrix

| Algorithm Family | Model Interface | Supported Models | Payoff Scope | Notes |
|---|---|---|---|---|
| Closed-Form / Semi-Analytical | Direct formula | BSM, Black-76, Bachelier, Heston CF, Merton JD, Variance-Gamma CF, SABR, Dupire | European | Each model has its own implementation |
| Tree / Lattice | Lattice parameterization | BSM (all trees), Derman-Kani (implied tree) | European + American | American via backward induction |
| Finite Difference | PDE operators | BSM (explicit, implicit, CN, PSOR), Heston (ADI) | European + American | PSOR for American exercise |
| Monte Carlo | Path simulation factories | BSM, Heston (2D Euler), Local Vol (Euler) | European; American via LSM | Extensible via callable factories |
| Fourier Transform | CharFn concept | BSM, Heston | European | Extensible via `make_*_charfn` factories |
| Integral Quadrature | Direct formula | BSM | European | — |
| Regression Approximation | Direct formula | BSM | European | — |
| Adjoint Greeks | Sensitivity callables | BSM (pathwise, LR, adjoint), Heston (LR delta) | European | See Greeks section |
| Machine Learning | Direct formula | BSM | European | — |

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
Standard MC (antithetic), Euler-Maruyama, Milstein, Longstaff-Schwartz (American), quasi-MC (Sobol, Halton), multilevel MC, importance sampling, control variates, antithetic variates, stratified sampling, Heston MC (2D correlated full-truncation Euler), local volatility MC (Euler with callable diffusion).

### Fourier Transform
Carr-Madan FFT, COS (Fang-Oosterlee), fractional FFT, Lewis Fourier inversion, Hilbert transform. All five methods support both BSM and Heston dynamics via the pluggable CharFn concept.

### Integral Quadrature
Gauss-Hermite, Gauss-Laguerre, Gauss-Legendre, adaptive quadrature.

### Regression Approximation
Polynomial chaos expansion, radial basis functions, sparse grid collocation, proper orthogonal decomposition.

### Greeks (Adjoint Methods)
Pathwise derivative delta (BSM), likelihood-ratio delta (BSM), BSM adjoint analytic delta (regularized), Heston likelihood-ratio delta (antithetic, with weight clipping).

### Machine Learning
Deep BSDE, PINNs, deep hedging, neural SDE calibration.

## Monte Carlo Design

The MC subsystem is structured as three independent layers:

1. **RNG policy.** The caller constructs a generator — typically `mc::make_mt19937_normal(seed)` — and passes it into the engine. The engine calls `gen()` to draw standard-normal variates. This decouples the RNG from the simulation loop.

2. **Model callable.** A functor mapping variates to asset prices. Terminal models map `(spot, t, z) -> S_T`. Step models map `(s, dt, dw) -> s_next`. Two-dimensional step models map `(s, v, dt, dw1, dw2) -> (s_next, v_next)` for correlated multi-factor processes (e.g., Heston). Local volatility step models map `(s, t, dt, dw) -> s_next` with time-dependent diffusion.

3. **Engine loop.** `estimate_terminal` runs a simple forward loop. `estimate_terminal_antithetic` pairs `+z` and `-z` draws for variance reduction. `estimate_stepwise` runs multi-step Euler/Milstein paths. `estimate_stepwise_2d` runs correlated 2D paths (Heston). All are header-only templates that inline through the callable indirection under any reasonable optimization level.

The accumulator receives `(S_T, z, path_index)` for terminal engines and `(S_T, path_index)` for stepwise engines, providing the metadata needed for variance reduction and sensitivity estimation.

## Greeks

**Pathwise derivative (BSM).** Differentiates the payoff indicator directly. For GBM, `dS_T / dSpot = S_T / spot`. Requires the payoff to be almost-everywhere differentiable (digital options are excluded).

**Likelihood ratio — BSM.** Differentiates the log-density instead of the payoff. Uses the score function `z / (vol * sqrt(t) * spot)` with configurable weight clipping. Works with discontinuous payoffs but has higher variance. Uses antithetic pairing.

**Likelihood ratio — Heston.** Uses the score function derived from the first time step of the Heston discretization: `z1_0 / (sqrt(v0) * sqrt(dt) * S_0)`. Only the initial variance contributes to the score because subsequent transitions are conditionally independent of the spot. Uses antithetic variance reduction with RNG state saved on the stack (no heap allocation).

**BSM adjoint delta.** Despite the `aad_delta` function name, this is a closed-form BSM delta computed via hand-written reverse-mode differentiation of the Black-Scholes formula, with Tikhonov regularization toward the ATM delta prior. It is not a general-purpose tape-based automatic differentiation engine. The function name is preserved for ABI compatibility. Internal C++ code can use the alias `bsm_adjoint_delta()`.

## Performance

- No virtual dispatch in hot paths. Model callables and characteristic functions are monomorphized templates.
- No heap allocation in simulation loops or Fourier integration loops. RNG state is stack-local; CharFn lambdas capture parameters by value.
- Header-only engine, model, and CharFn layers. Under LTO (enabled in Release builds), all lambda indirection is eliminated.
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

- **Model coverage is not uniform across algorithm families.** Monte Carlo supports BSM, Heston, and local volatility dynamics. Fourier methods support BSM and Heston. Finite difference supports BSM and Heston (ADI). Trees support BSM only (plus Derman-Kani implied tree). Quadrature, regression, and ML methods are BSM-only. Not all models are available in all families.

- **Greeks coverage is model-dependent.** Pathwise derivative delta and likelihood-ratio delta are available for BSM. Likelihood-ratio delta is available for Heston. The adjoint delta function is BSM closed-form only. No Greeks are currently available for local volatility MC.

- **`aad_delta` is not general AAD.** The function named `qk_agm_aad_delta` computes BSM delta via hand-differentiated closed-form Black-Scholes with regularization. It does not implement a computational tape, operator overloading, or any general-purpose automatic differentiation framework.

- **Vanilla European payoffs only** in the MC and adjoint Greek subsystems. Barrier, Asian, lookback, and other path-dependent payoffs are not supported. Longstaff-Schwartz supports American exercise as a special case.

- **Not a term-structure framework.** There is no yield curve construction, no vol surface interpolation infrastructure, no calendar/day-count conventions. Input rates and volatilities are flat scalars.

- **Platform-dependent reproducibility.** Bit-exact MC outputs are guaranteed for a given seed on the same platform and compiler. Cross-platform reproducibility (e.g., Linux vs macOS, GCC vs Clang) is not guaranteed due to differences in `std::normal_distribution` implementations.

- **Local vol MC at the C ABI level uses constant volatility.** The template-based local vol pricer accepts arbitrary `sigma(S, t)` callables in C++, but the C ABI and Python wrappers currently expose only the constant-vol specialization.

- **Sample Option data source download** To run `examples/validate_.../py` files, download sample options pricing data source from this url: [optionsDX](https://www.optionsdx.com/option-chain-field-definitions/)

## ABI Versioning

The C ABI version is defined in `qk_abi.h` as `QK_ABI_MAJOR.QK_ABI_MINOR` (currently 2.12). The Python wrapper checks the ABI version at load time and refuses to proceed on major version mismatch or if the library minor version is older than expected.

**Backward compatibility policy:** Minor version bumps are additive only — new functions are added, but existing function signatures are never changed or removed. A major version bump (which has not yet occurred) would indicate a breaking change.

## Error Handling

**C API.** Scalar functions return NaN on invalid input. Batch functions return error codes: `QK_OK` (0), `QK_ERR_NULL_PTR` (-1), `QK_ERR_BAD_SIZE` (-2), `QK_ERR_INVALID_INPUT` (-5). Call `qk_get_last_error()` for a human-readable error string.

**Python API.** Raises typed exceptions: `QKError`, `QKNullPointerError`, `QKBadSizeError`, `QKInvalidInputError`.

## Goals

- Fast, correct scalar and batch European option pricing.
- Minimal-dependency C++ core suitable for embedding.
- Stable C ABI for language-agnostic integration.
- Extensible internal architecture with documented model extension points for MC and Fourier families.

## Non-Goals

- General-purpose risk engine or portfolio management.
- Exotic payoff coverage.
- Term-structure or vol surface construction.
- Real-time market data integration.

## License

WTFPL v2. See `LICENSE`.

## Changelog (Unreleased)

- Added Heston Monte Carlo pricing via 2D correlated Euler step engine (`estimate_stepwise_2d`).
- Added Heston likelihood-ratio delta with antithetic variance reduction.
- Added local volatility Monte Carlo pricing with callable diffusion (`make_local_vol_euler_step`).
- Introduced CharFn concept (`charfn_concepts.h`) — pluggable characteristic function factories for Fourier methods.
- Added Heston Fourier pricing for all five transform methods (Carr-Madan FFT, COS, fractional FFT, Lewis, Hilbert) via CharFn.
- Refactored all five Fourier method implementations to template `_impl` functions parameterized on CharFn.
- ABI minor bump (additive, backward-compatible).

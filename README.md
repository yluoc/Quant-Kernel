# QuantKernel

QuantKernel is a C++17 quant pricing kernel with a Python wrapper.  
Linux/macOS is recommended.

**v2.10** highlights:
- SIMD-vectorized closed-form batch loops (`#pragma omp simd`)
- OpenMP thread-parallel heavy batch APIs (tree, MC, Fourier, Heston, Merton, VG)
- Thread-local error messages via `qk_get_last_error()` / `qk_clear_last_error()`
- Typed Python exceptions (`QKNullPointerError`, `QKBadSizeError`, `QKInvalidInputError`)
- Full API parity: 53 scalar + 53 native batch APIs across all algorithm families
- C++ unit test suite (6 executables) integrated with `ctest`
- CI parity guard to prevent scalar/batch/accelerator/test wiring drift
- Deterministic perf regression guard (fails if batch paths regress past 5% vs scalar baseline)
- PEP 561 type stubs for IDE autocompletion
- `pyproject.toml` for modern packaging


## Models
- **Closed-form / semi-analytical**: Black-Scholes-Merton, Black-76, Bachelier, Heston CF, Merton jump-diffusion, Variance Gamma CF, SABR (Hagan), Dupire local vol.
- **Fourier-transform methods**: Carr-Madan FFT, COS (Fang-Oosterlee), Fractional FFT, Lewis Fourier inversion, Hilbert transform pricing.
- **Integral quadrature methods**: Gauss-Hermite, Gauss-Laguerre, Gauss-Legendre, Adaptive quadrature.
- **Tree/lattice**: CRR, Jarrow-Rudd, Tian, Leisen-Reimer, Trinomial, Derman-Kani (const local vol entrypoint).
- **Finite-difference**: Explicit FD, Implicit FD, Crank-Nicolson, ADI (Douglas/Craig-Sneyd/Hundsdorfer-Verwer), PSOR.
- **Monte Carlo methods**: Standard Monte Carlo, Euler-Maruyama, Milstein, Longstaff-Schwartz (LSMC), Quasi-Monte Carlo (Sobol/Halton), MLMC, Importance Sampling, Control Variates, Antithetic Variates, Stratified Sampling.
- **Regression approximation**: Polynomial Chaos Expansion, Radial Basis Functions, Sparse Grid Collocation, Proper Orthogonal Decomposition.
- **Adjoint Greeks**: Pathwise Derivative Method, Likelihood Ratio Method, Adjoint Algorithmic Differentiation (AAD).
- **Machine-learning methods**: Deep BSDE, Physics-Informed Neural Networks (PINNs), Deep Hedging, Neural SDE Calibration.

## Quick Start (Use From Another Project)
This quick start assumes you clone/build QuantKernel once, then use it from any working directory.

1. Clone and build QuantKernel (one-time setup).
```bash
git clone <your-repo-url> /opt/quantkernel
cd /opt/quantkernel

# Create isolated Python environment
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# Build C++ shared library
cmake -S . -B build
cmake --build build -j
```

2. In your own project directory, point Python at QuantKernel and its shared library.
```bash
export QK_ROOT=/opt/quantkernel
export PYTHONPATH=$QK_ROOT/python
export QK_LIB_PATH=$QK_ROOT/build/cpp
```

Make it permanent (bash):
```bash
nano ~/.bashrc # under Linux/Unix
# QuantKernel (local), add the following 3 lines to ~/.bashrc
export QK_ROOT=/opt/quantkernel
export PYTHONPATH="$QK_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export QK_LIB_PATH="$QK_ROOT/build/cpp"
# save it
source ~/.bashrc
```

3. Call it from anywhere.
```bash
# run the following python code.
from quantkernel import QuantKernel, QK_CALL
qk = QuantKernel()
print(qk.black_scholes_merton_price(100, 100, 1.0, 0.2, 0.03, 0.01, QK_CALL))
# if everything is configured correctly, you should see a numeric price printed.
```

Optional demos (run from the QuantKernel repo):
```bash
cd $QK_ROOT
PYTHONPATH=python python3 python/examples/demo_accelerator.py
PYTHONPATH=python python3 python/examples/run_all_algos.py --profile quick
```

## Native Batch APIs (Recommended)
- Scalar methods remain available for convenience.
- For throughput-sensitive workloads, prefer native batch entrypoints backed by C++ array-in/array-out kernels.
- Batch APIs currently available:
  - **Closed-form / semi-analytical**: `black_scholes_merton_price_batch`, `black76_price_batch`, `bachelier_price_batch`, `heston_price_cf_batch`, `merton_jump_diffusion_price_batch`, `variance_gamma_price_cf_batch`, `sabr_hagan_lognormal_iv_batch`, `sabr_hagan_black76_price_batch`, `dupire_local_vol_batch`
  - **Fourier-transform**: `carr_madan_fft_price_batch`, `cos_method_fang_oosterlee_price_batch`, `fractional_fft_price_batch`, `lewis_fourier_inversion_price_batch`, `hilbert_transform_price_batch`
  - **Tree/lattice**: `crr_price_batch`, `jarrow_rudd_price_batch`, `tian_price_batch`, `leisen_reimer_price_batch`, `trinomial_tree_price_batch`, `derman_kani_const_local_vol_price_batch`
  - **Monte Carlo**: `standard_monte_carlo_price_batch`, `euler_maruyama_price_batch`, `milstein_price_batch`, `longstaff_schwartz_price_batch`, `quasi_monte_carlo_sobol_price_batch`, `quasi_monte_carlo_halton_price_batch`, `multilevel_monte_carlo_price_batch`, `importance_sampling_price_batch`, `control_variates_price_batch`, `antithetic_variates_price_batch`, `stratified_sampling_price_batch`
  - **Finite-difference**: `explicit_fd_price_batch`, `implicit_fd_price_batch`, `crank_nicolson_price_batch`, `adi_douglas_price_batch`, `adi_craig_sneyd_price_batch`, `adi_hundsdorfer_verwer_price_batch`, `psor_price_batch`
  - **Integral quadrature**: `gauss_hermite_price_batch`, `gauss_laguerre_price_batch`, `gauss_legendre_price_batch`, `adaptive_quadrature_price_batch`
  - **Regression approximation**: `polynomial_chaos_expansion_price_batch`, `radial_basis_function_price_batch`, `sparse_grid_collocation_price_batch`, `proper_orthogonal_decomposition_price_batch`
  - **Adjoint Greeks**: `pathwise_derivative_delta_batch`, `likelihood_ratio_delta_batch`, `aad_delta_batch`
  - **Machine-learning**: `deep_bsde_price_batch`, `pinns_price_batch`, `deep_hedging_price_batch`, `neural_sde_calibration_price_batch`

Build optional Cython native module (releases GIL during long batch calls):
```bash
cd $QK_ROOT/python
QK_LIB_PATH=$QK_ROOT/build/cpp python3 setup.py build_ext --inplace
```
If the native extension is not built, QuantKernel automatically falls back to
the ctypes batch path (same API, slightly higher overhead).

Example:
```python
import numpy as np
from quantkernel import QuantKernel, QK_CALL, QK_PUT

qk = QuantKernel()
n = 100000
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

## Performance Snapshot
Use the reproducible benchmark script:
```bash
PYTHONPATH=python QK_LIB_PATH=$QK_ROOT/build/cpp \
python3 python/examples/benchmark_scalar_batch_cpp.py --n 50000 --repeats 3
```

Representative sample output (Ubuntu, `n=50000`, `repeats=3`; hardware-dependent):

| Mode | Median ms | Throughput (prices/s) | Speedup vs Python scalar |
|---|---:|---:|---:|
| Python scalar | 93.611 | 534,125 | 1.00x |
| Python batch (price_batch) | 35.282 | 1,417,151 | 2.65x |
| C++ batch API via Python | 2.214 | 22,583,202 | 42.28x |
| C++ direct executable (scalar) | 2.810 | 17,793,746 | 33.31x |
| C++ direct executable (batch) | 2.565 | 19,495,594 | 36.50x |

## Error Handling

### C++ API
Batch functions return error codes defined in `qk_abi.h`:

| Code | Constant | Meaning |
|---:|---|---|
| 0 | `QK_OK` | Success |
| -1 | `QK_ERR_NULL_PTR` | A required pointer argument was `NULL` |
| -2 | `QK_ERR_BAD_SIZE` | Batch size `n` was zero or negative |
| -5 | `QK_ERR_INVALID_INPUT` | Invalid parameter value |

After any error, call `qk_get_last_error()` for a human-readable message:

```c
int32_t rc = qk_cf_black_scholes_merton_price_batch(NULL, ...);
if (rc != QK_OK) {
    printf("Error %d: %s\n", rc, qk_get_last_error());
    // prints: "Error -1: null pointer: spot"
}
```

### Python API
Batch methods raise typed exceptions on failure:

```python
from quantkernel import QuantKernel, QKNullPointerError, QKBadSizeError

qk = QuantKernel()
try:
    qk.black_scholes_merton_price_batch(None, ...)
except QKNullPointerError as e:
    print(e)  # "null pointer: spot"
except QKBadSizeError as e:
    print(e)  # "bad batch size: -3"
```

All exception classes inherit from `QKError` (which inherits from `RuntimeError`).

## Developer Commands
Useful local commands from the repo root:

```bash
make build         # build shared library and C++ targets
make build-native  # build optional Cython native batch extension
make bench         # run scalar/batch benchmark table script
make quick         # C++ + Python test suite
make test-cpp      # run C++ unit tests only (via ctest)
make test-py       # run Python tests only (via pytest)
```

## Batch + GPU Acceleration (Python layer)
- No CUDA integration in C++ is required.
- Use `QuantKernel.price_batch(...)` or `QuantAccelerator`.
- Backend options: `auto`, `cpu`, `gpu`.
- GPU path is available when CuPy is installed.

### CuPy-Accelerated Algorithms

The following `method` names have explicit NumPy/CuPy vectorized implementations in `QuantAccelerator`:

| Method (`price_batch` / `QuantAccelerator`) | GPU Threshold | Acceleration Approach |
|-----------|--------------|----------------------|
| `black_scholes_merton_price` | 20,000 | Vectorized closed-form across batch |
| `black76_price` | 20,000 | Vectorized closed-form across batch |
| `bachelier_price` | 20,000 | Vectorized closed-form across batch |
| `sabr_hagan_lognormal_iv` | 15,000 | Vectorized SABR Hagan IV across batch |
| `sabr_hagan_black76_price` | 15,000 | Vectorized SABR IV + Black-76 pricing |
| `dupire_local_vol` | 30,000 | Vectorized Dupire local-vol inversion |
| `merton_jump_diffusion_price` | 10,000 | Vectorized jump-series summation |
| `carr_madan_fft_price` | 512 | Batched damped-CF FFT across frequency grid |
| `cos_method_fang_oosterlee_price` | 1,024 | Batched COS coefficient assembly and reduction |
| `fractional_fft_price` | 256 | Batched fractional-phase transform on shared grid |
| `lewis_fourier_inversion_price` | 1,024 | Batched Fourier integral with vectorized quadrature |
| `hilbert_transform_price` | 1,024 | Batched Fourier-probability inversion (P1/P2) |
| `standard_monte_carlo_price` | 1 | `(B, P)` random normals and vectorized payoff |
| `euler_maruyama_price` | 1 | Time-step loop with batched path evolution |
| `milstein_price` | 1 | Milstein path update with batched correction term |
| `importance_sampling_price` | 1 | Shifted-drift MC with likelihood-ratio weighting |
| `control_variates_price` | 1 | MC with batched control-variate adjustment |
| `antithetic_variates_price` | 1 | Paired `+Z/-Z` paths for variance reduction |
| `stratified_sampling_price` | 1 | Stratified uniforms + inverse-normal mapping |
| `polynomial_chaos_expansion_price` | 20,000 | Vectorized surrogate correction over BSM baseline |
| `radial_basis_function_price` | 20,000 | Vectorized surrogate correction over BSM baseline |
| `sparse_grid_collocation_price` | 20,000 | Vectorized surrogate correction over BSM baseline |
| `proper_orthogonal_decomposition_price` | 20,000 | Vectorized surrogate correction over BSM baseline |
| `pathwise_derivative_delta` | 1 | Batched pathwise Greek estimator |
| `likelihood_ratio_delta` | 1 | Batched likelihood-ratio Greek estimator |
| `aad_delta` | 20,000 | Vectorized closed-form delta with regularization |
| `neural_sde_calibration_price` | 20,000 | Vectorized BSM pricing at calibrated effective vol |

- **GPU Threshold**: minimum batch size before `gpu_vectorized` is selected in the current strategy logic (when `backend` is `auto` or `gpu` and CuPy is available).
- Methods not listed above use the C++ scalar path, dispatched as `threaded` or `sequential` depending on method class and batch size.

Install CuPy (optional, for GPU backend):
```bash
python3 -m pip install cupy-cuda12x
```

```python
from quantkernel import QuantKernel, QK_CALL

qk = QuantKernel()
jobs = [
    {"spot": 100, "strike": 100, "t": 1.0, "vol": 0.2, "r": 0.03, "q": 0.01, "option_type": QK_CALL},
    {"spot": 105, "strike": 100, "t": 0.5, "vol": 0.25, "r": 0.03, "q": 0.01, "option_type": QK_CALL},
]
prices = qk.price_batch("black_scholes_merton_price", jobs, backend="auto")
print(prices)
```

Run CPU vs GPU runtime simulation:

```bash
PYTHONPATH=python python3 python/examples/run_all_algos.py --profile quick
```

## IDE IntelliSense (`compile_commands.json`)

If your editor (VS Code, CLion, etc.) shows `#include` errors or missing symbols for C++ files, you need to generate a `compile_commands.json` at the project root. This is needed when:

- You have freshly cloned the repo and haven't built yet.
- New C++ source files or fuzz tests have been added since your last build.
- IntelliSense squiggles appear on valid `#include` directives.

Generate it by running from the project root:

```bash
# Build both targets with compile_commands.json generation
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build -j

cmake -S fuzztest -B fuzztest/build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build fuzztest/build -j

# Merge into a single root-level compile_commands.json
python3 -c "
import json, pathlib
root = pathlib.Path('.')
merged = []
seen = set()
for f in [root/'build'/'compile_commands.json', root/'fuzztest'/'build'/'compile_commands.json']:
    if f.exists():
        for e in json.loads(f.read_text()):
            key = (e.get('file'), e.get('directory'), e.get('command', e.get('arguments')))
            if key not in seen:
                seen.add(key)
                merged.append(e)
pathlib.Path('compile_commands.json').write_text(json.dumps(merged, indent=2) + '\n')
print(f'Wrote {len(merged)} entries to compile_commands.json')
"
```

If you're using VS Code, create a local `.vscode/c_cpp_properties.json` and set `compileCommands` to `${workspaceFolder}/compile_commands.json` (the `.vscode/` directory is ignored by default since it contains machine-specific paths).

For tools expecting a singular file name, keep `compile_command.json` in sync:
```bash
cp build/compile_commands.json compile_command.json
```

## Tests

### C++ Unit Tests
Six C++ test executables covering closed-form, tree/lattice, Monte Carlo, Fourier, error handling, and FDM/IQM/RAM/AGM/MLM parity:

```bash
make test-cpp
# or directly via ctest:
cd build && ctest --output-on-failure
```

### Python Tests
```bash
make quick       # C++ + Python tests
make test-py     # Python tests only
```

Deterministic native batch/perf checks:
```bash
PYTHONPATH=python QK_LIB_PATH=$QK_ROOT/build/cpp \
pytest -q python/tests/test_parity_guard.py python/tests/test_batch_api.py python/tests/test_perf_regression.py
```

CI runs deterministic fuzz/property checks (`FUZZTEST_PRNG_SEED` fixed), batch
accuracy checks, parity guards, and a deterministic performance guard that fails
if batch paths regress to less than 5% faster than scalar baseline.

## Build Optimization

The CMake build includes several performance features:
- **Precompiled headers** for `<cmath>`, `<vector>`, `<algorithm>`, etc. — reduces rebuild times
- **`-O3 -ffast-math -march=native`** — aggressive optimization for the host CPU
- **`-fopenmp-simd`** — enables SIMD pragmas without requiring the OpenMP threading runtime
- **OpenMP threading** (when available) — parallelizes heavy batch loops across cores
- **Link-time optimization** (`INTERPROCEDURAL_OPTIMIZATION TRUE`) — whole-program optimization

## Type Stubs

QuantKernel ships PEP 561 type stubs (`python/quantkernel/__init__.pyi` + `py.typed`).
IDEs like VS Code, PyCharm, and mypy will automatically pick up type information for
all public methods, constants, and exception classes.

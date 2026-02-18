# QuantKernel

QuantKernel is a C++17 quant pricing kernel with a Python wrapper.  
Linux/macOS is recommended.


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
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build -j

cmake -S fuzztest -B fuzztest/build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build fuzztest/build -j

# Merge into a single root-level compile_commands.json
python3 -c "
import json, pathlib
root = pathlib.Path('.')
merged, seen = [], set()
for f in [root/'build'/'compile_commands.json', root/'fuzztest'/'build'/'compile_commands.json']:
    if f.exists():
        for e in json.loads(f.read_text()):
            if e['file'] not in seen:
                seen.add(e['file'])
                merged.append(e)
pathlib.Path('compile_commands.json').write_text(json.dumps(merged, indent=2) + '\n')
print(f'Wrote {len(merged)} entries to compile_commands.json')
"
```

If you're using VS Code, create a local `.vscode/c_cpp_properties.json` and set `compileCommands` to `${workspaceFolder}/compile_commands.json` (the `.vscode/` directory is ignored by default since it contains machine-specific paths).

## Tests
```bash
make quick
make test-py
```

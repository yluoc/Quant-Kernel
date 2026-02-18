# QuantKernel

QuantKernel is a C++17 quant pricing kernel with a Python wrapper.  
Linux/macOS is recommended.


## Models
- Closed-form / semi-analytical: Black-Scholes-Merton, Black-76, Bachelier, Heston CF, Merton jump-diffusion, Variance Gamma CF, SABR (Hagan), Dupire local vol.
- Fourier-transform methods: Carr-Madan FFT, COS (Fang-Oosterlee), Fractional FFT, Lewis Fourier inversion, Hilbert transform pricing.
- Integral quadrature methods: Gauss-Hermite, Gauss-Laguerre, Gauss-Legendre, Adaptive quadrature.
- Tree/lattice: CRR, Jarrow-Rudd, Tian, Leisen-Reimer, Trinomial, Derman-Kani (const local vol entrypoint).
- Finite-difference: Explicit FD, Implicit FD, Crank-Nicolson, ADI (Douglas/Craig-Sneyd/Hundsdorfer-Verwer), PSOR.
- Monte Carlo methods: Standard Monte Carlo, Euler-Maruyama, Milstein, Longstaff-Schwartz (LSMC), Quasi-Monte Carlo (Sobol/Halton), MLMC, Importance Sampling, Control Variates, Antithetic Variates, Stratified Sampling.

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
nano ~/.bashrc
# QuantKernel (local), add the following 3 lines to ~/.bashrc
export QK_ROOT=/opt/quantkernel
export PYTHONPATH="$QK_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export QK_LIB_PATH="$QK_ROOT/build/cpp"
# save it
source ~/.bashrc
```

3. Call it from anywhere.
```bash
python3
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

The following algorithms have fully vectorized NumPy/CuPy implementations in `QuantAccelerator`, enabling GPU acceleration for large batches:

| Algorithm | GPU Threshold | Acceleration Approach |
|-----------|--------------|----------------------|
| Black-Scholes-Merton | 20,000 | Vectorized closed-form across batch |
| Black-76 | 20,000 | Vectorized closed-form across batch |
| Bachelier | 20,000 | Vectorized closed-form across batch |
| SABR Hagan (lognormal IV) | 15,000 | Vectorized closed-form across batch |
| SABR Hagan (Black-76 price) | 15,000 | Vectorized closed-form across batch |
| Dupire Local Vol | 30,000 | Vectorized closed-form across batch |
| Merton Jump Diffusion | 10,000 | Series expansion with vectorized BSM per term |
| Carr-Madan FFT | 512 | Batched damped-CF FFT across frequency grid |
| COS (Fang-Oosterlee) | 1,024 | Batched cosine-expansion coefficients and reductions |
| Fractional FFT | 256 | Batched fractional-phase transform on shared grid |
| Lewis Fourier Inversion | 1,024 | Batched Fourier integral with vectorized quadrature |
| Hilbert Transform Method | 1,024 | Batched Fourier-probability inversion (P1/P2) |
| Standard Monte Carlo | 1 | (B, P) random normals on GPU, vectorized payoff |
| Euler-Maruyama | 1 | Time-step loop, all (B, P) paths advance per step |
| Milstein | 1 | Euler + higher-order correction, same path structure |
| Importance Sampling | 1 | Shifted-drift MC with likelihood ratio weighting |
| Control Variates | 1 | MC with analytical control variate (terminal stock) |
| Antithetic Variates | 1 | Paired +Z/-Z paths for variance reduction |
| Stratified Sampling | 1 | Stratified uniforms via inverse CDF, then MC |

- **GPU Threshold**: minimum batch size before the GPU backend is selected (under `backend="auto"`). Monte Carlo methods use threshold 1 because path-level parallelism benefits from GPU even for a single job.
- Algorithms not listed (Heston CF, integral quadrature, tree/lattice, finite-difference, Longstaff-Schwartz, QMC Sobol/Halton, MLMC) run via threaded C++ scalar execution.

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

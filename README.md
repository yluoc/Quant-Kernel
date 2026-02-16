# QuantKernel

QuantKernel is a C++17 quant pricing kernel with a Python wrapper.

## Models
- Closed-form / semi-analytical: Black-Scholes-Merton, Black-76, Bachelier, Heston CF, Merton jump-diffusion, Variance Gamma CF, SABR (Hagan), Dupire local vol.
- Tree/lattice: CRR, Jarrow-Rudd, Tian, Leisen-Reimer, Trinomial, Derman-Kani (const local vol entrypoint).
- Finite-difference: Explicit FD, Implicit FD, Crank-Nicolson, ADI (Douglas/Craig-Sneyd/Hundsdorfer-Verwer), PSOR.

## Quick Start (Clone to First Price)
1. Clone and enter the repo.
```bash
git clone <your-repo-url>
cd quant_algo_cpp_py
```

2. Create and activate a Python environment.
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
```

3. Install Python dependencies.
```bash
python3 -m pip install -r requirements.txt
```

4. Build the C++ shared library.
```bash
cmake -S . -B build
cmake --build build -j
```

5. Run a first pricing call from Python.
```bash
PYTHONPATH=python python3 - <<'PY'
from quantkernel import QuantKernel, QK_CALL
qk = QuantKernel()
print(qk.black_scholes_merton_price(100, 100, 1.0, 0.2, 0.03, 0.01, QK_CALL))
PY
```

6. Optional: run demos.
```bash
PYTHONPATH=python python3 python/examples/demo_accelerator.py
PYTHONPATH=python python3 python/examples/run_all_algos.py --profile quick
```

## Batch + GPU Acceleration (Python layer)
- No CUDA integration in C++ is required.
- Use `QuantKernel.price_batch(...)` or `QuantAccelerator`.
- Backend options: `auto`, `cpu`, `gpu`.
- GPU path is available when CuPy is installed.

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

## Tests
```bash
make quick
make test-py
```

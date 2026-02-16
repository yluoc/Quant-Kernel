# QuantKernel

QuantKernel is a C++17 quant pricing kernel with a Python wrapper.

## Models
- Closed-form / semi-analytical: Black-Scholes-Merton, Black-76, Bachelier, Heston CF, Merton jump-diffusion, Variance Gamma CF, SABR (Hagan), Dupire local vol.
- Tree/lattice: CRR, Jarrow-Rudd, Tian, Leisen-Reimer, Trinomial, Derman-Kani (const local vol entrypoint).
- Finite-difference: Explicit FD, Implicit FD, Crank-Nicolson, ADI (Douglas/Craig-Sneyd/Hundsdorfer-Verwer), PSOR.

## Quick Start (Use From Another Project)
This quick start assumes you clone/build QuantKernel once, then use it from any working directory.

1. Clone and build QuantKernel (one-time setup).
```bash
git clone <your-repo-url> /opt/quantkernel
cd /opt/quantkernel
python3 -m pip install -r requirements.txt
cmake -S . -B build
cmake --build build -j
```

2. In your own project directory, point Python at QuantKernel and its shared library.
```bash
export QK_ROOT=/opt/quantkernel
export PYTHONPATH=$QK_ROOT/python
export QK_LIB_PATH=$QK_ROOT/build/cpp
```

3. Call it from anywhere.
```bash
python3 - <<'PY'
from quantkernel import QuantKernel, QK_CALL
qk = QuantKernel()
print(qk.black_scholes_merton_price(100, 100, 1.0, 0.2, 0.03, 0.01, QK_CALL))
PY
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

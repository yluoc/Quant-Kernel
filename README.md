# QuantKernel

QuantKernel is a C++17 quantitative pricing library with:
- A stable C ABI (`cpp/include/quantkernel/qk_api.h`, `cpp/include/quantkernel/qk_abi.h`)
- A Python wrapper (`python/quantkernel`)
- Model families covering closed-form, tree/lattice, and finite-difference methods

## Project Layout

```text
.
├── CMakeLists.txt
├── Makefile
├── README.md
├── requirements.txt
├── cpp/
│   ├── CMakeLists.txt
│   ├── include/quantkernel/
│   │   ├── qk_abi.h
│   │   └── qk_api.h
│   └── src/
│       ├── qk_api.cpp
│       ├── common/
│       └── algorithms/
│           ├── closed_form_semi_analytical/
│           ├── tree_lattice_methods/
│           └── finite_difference_methods/
├── python/
│   ├── quantkernel/
│   │   ├── __init__.py
│   │   ├── _abi.py
│   │   ├── _loader.py
│   │   └── engine.py
│   ├── examples/
│   └── tests/
└── fuzztest/
    ├── CMakeLists.txt
    ├── fuzz_closed_form_models.cpp
    ├── fuzz_tree_lattice_models.cpp
    ├── fuzz_finite_difference_models.cpp
    └── fuzztest/   # upstream fuzztest dependency
```

Usage guide:
- `CMakeLists.txt` (root): top-level build entry point; includes the `cpp/` library build.
- `Makefile`: convenience wrapper for configure/build/test/demo workflows (`make build`, `make quick`, `make demo`).
- `cpp/include/quantkernel/qk_abi.h`: ABI/version and return-code contract used by all clients.
- `cpp/include/quantkernel/qk_api.h`: exported C ABI pricing functions used by C/C++ and Python ctypes.
- `cpp/src/qk_api.cpp`: ABI bridge from C API calls to internal C++ model implementations.
- `cpp/src/algorithms/closed_form_semi_analytical/`: direct and characteristic-function pricers.
- `cpp/src/algorithms/tree_lattice_methods/`: binomial/trinomial/implied-tree model implementations.
- `cpp/src/algorithms/finite_difference_methods/`: PDE/grid-based pricing implementations.
- `python/quantkernel/_loader.py`: shared-library discovery, function-signature setup, and ABI verification.
- `python/quantkernel/engine.py`: high-level `QuantKernel` Python API mapped to C ABI functions.
- `python/examples/`: runnable examples for verifying install/load and basic usage.
- `python/tests/`: pytest-based API/behavior checks for the Python wrapper.
- `fuzztest/CMakeLists.txt`: separate fuzz test build that links `quantkernel` with fuzztest/gtest.
- `fuzztest/fuzz_*_models.cpp`: randomized robustness/property checks for closed-form, tree/lattice, and FDM models.

## Implemented Models

Closed-form / semi-analytical (`cpp/src/algorithms/closed_form_semi_analytical/`):
- Black-Scholes-Merton
- Black (1976)
- Bachelier
- Heston (characteristic-function pricing)
- Merton jump-diffusion
- Variance Gamma (characteristic-function pricing)
- SABR (Hagan lognormal asymptotics)
- Dupire local volatility

Tree/lattice (`cpp/src/algorithms/tree_lattice_methods/`):
- Cox-Ross-Rubinstein (CRR)
- Jarrow-Rudd
- Tian
- Leisen-Reimer
- Trinomial tree
- Derman-Kani implied tree (constant local-vol ABI entrypoint)

Finite-difference (`cpp/src/algorithms/finite_difference_methods/`):
- Explicit FD
- Implicit FD
- Crank-Nicolson
- ADI (Douglas, Craig-Sneyd, Hundsdorfer-Verwer)
- PSOR

## Build

### CMake

```bash
cmake -S . -B build
cmake --build build -j
```

### Makefile shortcut

```bash
make build
```

Shared library output:
- Linux: `build/cpp/libquantkernel.so`
- macOS: `build/cpp/libquantkernel.dylib`
- Windows: `build/cpp/quantkernel.dll` (or `libquantkernel.dll`)

## C/C++ Usage

Public headers:
- `cpp/include/quantkernel/qk_abi.h`
- `cpp/include/quantkernel/qk_api.h`

CMake consumer example:

```cmake
add_library(quantkernel SHARED IMPORTED)
set_target_properties(quantkernel PROPERTIES
    IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/build/cpp/libquantkernel.so"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/cpp/include"
)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE quantkernel)
```

Internal aggregate headers (useful for native C++ integration):

```cpp
#include "algorithms/closed_form_semi_analytical/closed_form_models.h"
#include "algorithms/tree_lattice_methods/tree_lattice_models.h"
#include "algorithms/finite_difference_methods/finite_difference_models.h"
```

## Python Usage

Install Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Build first, then run with the package path:

```bash
PYTHONPATH=python python3 -c "from quantkernel import QuantKernel; print(QuantKernel())"
```

Quick demo:

```bash
make demo
```

Example:

```python
from quantkernel import QuantKernel, QK_CALL

qk = QuantKernel()
bsm = qk.black_scholes_merton_price(100.0, 100.0, 1.0, 0.2, 0.03, 0.01, QK_CALL)
crr = qk.crr_price(100.0, 100.0, 1.0, 0.2, 0.03, 0.01, QK_CALL, 300)
fdm = qk.crank_nicolson_price(100.0, 100.0, 1.0, 0.2, 0.03, 0.01, QK_CALL, 400, 400)
print(bsm, crr, fdm)
```

Loader behavior:
- Searches `QK_LIB_PATH` (if set), then `build/cpp`, `build`, and project root
- Verifies ABI compatibility at load time via `qk_abi_version`

Python `QuantKernel` exposes:
- Closed-form methods: `black_scholes_merton_price`, `black76_price`, `bachelier_price`, `heston_price_cf`, `merton_jump_diffusion_price`, `variance_gamma_price_cf`, `sabr_hagan_lognormal_iv`, `sabr_hagan_black76_price`, `dupire_local_vol`
- Tree/lattice methods: `crr_price`, `jarrow_rudd_price`, `tian_price`, `leisen_reimer_price`, `trinomial_tree_price`, `derman_kani_const_local_vol_price`
- Finite-difference methods: `explicit_fd_price`, `implicit_fd_price`, `crank_nicolson_price`, `adi_douglas_price`, `adi_craig_sneyd_price`, `adi_hundsdorfer_verwer_price`, `psor_price`

## Testing

Project quick test path:

```bash
make quick
```

Python tests only:

```bash
make test-py
```

Fuzz tests (separate CMake project under `fuzztest/`):

```bash
cmake -S fuzztest -B build-fuzz
cmake --build build-fuzz -j
ctest --test-dir build-fuzz --output-on-failure
```

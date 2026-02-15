# QuantKernel

QuantKernel is a high-performance options analytics library with a stable C ABI and a Python wrapper.

Implemented C++ quantitative models are under:
- `cpp/src/algorithms/closed_form_semi_analytical/`
Includes:
- Black-Scholes-Merton
- Black (1976)
- Bachelier
- Heston (characteristic-function pricing)
- Merton Jump-Diffusion
- Variance Gamma (characteristic-function pricing)
- SABR (Hagan lognormal asymptotics)
- Dupire Local Volatility (direct and finite-difference forms)

Implemented C++ tree/lattice models are under:
- `cpp/src/algorithms/tree_lattice_methods/`
Includes:
- Cox-Ross-Rubinstein (CRR)
- Jarrow-Rudd
- Tian Tree
- Leisen-Reimer
- Trinomial Tree
- Derman-Kani implied tree (constant local-vol ABI entrypoint)

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

Main shared library output:
- Linux: `build/cpp/libquantkernel.so`
- macOS: `build/cpp/libquantkernel.dylib`
- Windows: `build/cpp/quantkernel.dll` (or `libquantkernel.dll`)

## Use As A C/C++ Library

Public headers:
- `cpp/include/quantkernel/qk_abi.h`
- `cpp/include/quantkernel/qk_api.h`

The API is C-compatible (`extern "C"`), so it can be used from C, C++, or FFI bindings.

### CMake consumer example

```cmake
add_library(quantkernel SHARED IMPORTED)
set_target_properties(quantkernel PROPERTIES
    IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/build/cpp/libquantkernel.so"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/cpp/include"
)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE quantkernel)
```

### C++ model include example

```cpp
#include "algorithms/closed_form_semi_analytical/closed_form_models.h"
#include "algorithms/tree_lattice_methods/tree_lattice_models.h"
```

## Use As A Python Library

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Build native libraries first, then run Python with package path:

```bash
PYTHONPATH=python python3 -c "from quantkernel import QuantKernel; print(QuantKernel())"
```

### Python usage example

```python
from quantkernel import QuantKernel, QK_CALL

qk = QuantKernel()
bsm = qk.black_scholes_merton_price(100.0, 100.0, 1.0, 0.2, 0.03, 0.01, QK_CALL)
crr = qk.crr_price(100.0, 100.0, 1.0, 0.2, 0.03, 0.01, QK_CALL, 300)
print(bsm, crr)
```

Python loader notes:
- Default search paths include `build/cpp`, `build`, and project root.
- Set `QK_LIB_PATH` to override where shared libraries are searched.

Python `QuantKernel` exposes:
- Closed-form methods: `black_scholes_merton_price`, `black76_price`, `bachelier_price`,
  `heston_price_cf`, `merton_jump_diffusion_price`, `variance_gamma_price_cf`,
  `sabr_hagan_lognormal_iv`, `sabr_hagan_black76_price`, `dupire_local_vol`
- Tree/lattice methods: `crr_price`, `jarrow_rudd_price`, `tian_price`,
  `leisen_reimer_price`, `trinomial_tree_price`, `derman_kani_const_local_vol_price`

## Run Tests

```bash
make test
```

This runs:
- C++ tests via `ctest`
- Python tests via `pytest`

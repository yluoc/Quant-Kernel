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

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

Main shared library output:
- Linux: `build/cpp/libquantkernel.so`
- macOS: `build/cpp/libquantkernel.dylib`
- Windows: `build/cpp/quantkernel.dll` (or `libquantkernel.dll`)

Optional Rust runtime shell:

```bash
cmake --build build --target quantkernel_runtime
```

Runtime output is under `rust/runtime/target/release/` (and may also appear in `target/release/` depending on build flow).

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
import ctypes as ct
from quantkernel import QuantKernel

qk = QuantKernel()
major = ct.c_int32(-1)
minor = ct.c_int32(-1)
qk._lib.qk_abi_version(ct.byref(major), ct.byref(minor))
print(major.value, minor.value)
```

Python loader notes:
- Default search paths include `build/cpp`, `build`, `target/release`, `rust/runtime/target/release`, and project root.
- Set `QK_LIB_PATH` to override where shared libraries are searched.

## Runtime Shell Mode (Optional)

To enable the Rust runtime shell for plugin lifecycle and ABI validation:

```bash
QK_USE_RUNTIME=1 \
QK_PLUGIN_PATH=$PWD/build/cpp/libquantkernel.so \
PYTHONPATH=python python3 python/examples/demo_pricing.py
```

If `QK_PLUGIN_PATH` is unset, the loader tries to discover a plugin in its default search directories.
The runtime plugin loader is currently implemented for Unix (`dlopen` path).
The runtime shell currently manages plugin load/unload and ABI checks; closed-form model execution remains in the C++ plugin.

## Run Tests

```bash
make test
```

This runs:
- C++ tests via `ctest`
- Python tests via `pytest`

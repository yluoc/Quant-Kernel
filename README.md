# QuantKernel

QuantKernel is a high-performance options analytics library with a stable C ABI and a Python wrapper.

Implemented engines:
- Black-Scholes pricing with Greeks (`qk_bs_price`)
- Implied volatility solver (`qk_iv_solve`)
- Monte Carlo pricer (`qk_mc_price`)

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

### Minimal C++ call example

```cpp
#include <quantkernel/qk_api.h>
#include <vector>
#include <cstdint>

int main() {
    std::vector<double> spot{100.0}, strike{100.0}, tte{1.0}, vol{0.2}, r{0.05}, q{0.0};
    std::vector<int32_t> opt{QK_CALL};
    std::vector<double> price(1), delta(1), gamma(1), vega(1), theta(1), rho(1);
    std::vector<int32_t> err(1);

    QKBSInput in{1, spot.data(), strike.data(), tte.data(), vol.data(), r.data(), q.data(), opt.data()};
    QKBSOutput out{price.data(), delta.data(), gamma.data(), vega.data(), theta.data(), rho.data(), err.data()};

    int32_t rc = qk_bs_price(&in, &out);
    return (rc == QK_OK && err[0] == QK_ROW_OK) ? 0 : 1;
}
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

### Python pricing example

```python
import numpy as np
from quantkernel import QuantKernel

qk = QuantKernel()

res = qk.bs_price(
    spot=np.array([100.0]),
    strike=np.array([100.0]),
    time_to_expiry=np.array([1.0]),
    volatility=np.array([0.20]),
    risk_free_rate=np.array([0.05]),
    dividend_yield=np.array([0.0]),
    option_type=np.array([QuantKernel.CALL], dtype=np.int32),
)

print(res["price"][0], res["error_codes"][0])
```

Python loader notes:
- Default search paths include `build/cpp`, `build`, `target/release`, `rust/runtime/target/release`, and project root.
- Set `QK_LIB_PATH` to override where shared libraries are searched.

## Runtime Shell Mode (Optional)

To route calls through the Rust runtime safety shell:

```bash
QK_USE_RUNTIME=1 \
QK_PLUGIN_PATH=$PWD/build/cpp/libquantkernel.so \
PYTHONPATH=python python3 python/examples/demo_pricing.py
```

If `QK_PLUGIN_PATH` is unset, the loader tries to discover a plugin in its default search directories.
The runtime plugin loader is currently implemented for Unix (`dlopen` path).

## Run Tests

```bash
make test
```

This runs:
- C++ tests via `ctest`
- Python tests via `pytest`

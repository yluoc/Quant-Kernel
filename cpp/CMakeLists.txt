# QuantKernel shared library
add_library(quantkernel SHARED
    src/qk_api.cpp
    src/algorithms/black_scholes/bs_pricer.cpp
    src/algorithms/implied_vol/iv_solver.cpp
)

target_include_directories(quantkernel PUBLIC include)

target_compile_options(quantkernel PRIVATE
    -Wall -Wextra -Wpedantic
    -O2 -ffast-math -march=native
)

set_target_properties(quantkernel PROPERTIES
    OUTPUT_NAME "quantkernel"
    PREFIX "lib"
    SOVERSION 1
)

# C++ tests
add_subdirectory(tests)

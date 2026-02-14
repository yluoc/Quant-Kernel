# QuantKernel shared library
add_library(quantkernel SHARED
    src/qk_api.cpp
    src/algorithms/closed_form_semi_analytical/black_scholes_merton/black_scholes_merton.cpp
    src/algorithms/closed_form_semi_analytical/black_1976/black_1976.cpp
    src/algorithms/closed_form_semi_analytical/bachelier/bachelier.cpp
    src/algorithms/closed_form_semi_analytical/heston/heston.cpp
    src/algorithms/closed_form_semi_analytical/merton_jump_diffusion/merton_jump_diffusion.cpp
    src/algorithms/closed_form_semi_analytical/variance_gamma/variance_gamma.cpp
    src/algorithms/closed_form_semi_analytical/sabr/sabr.cpp
    src/algorithms/closed_form_semi_analytical/dupire/dupire.cpp
)

target_include_directories(quantkernel
    PUBLIC include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

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

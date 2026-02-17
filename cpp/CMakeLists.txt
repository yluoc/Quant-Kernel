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
    src/algorithms/tree_lattice_methods/crr/crr.cpp
    src/algorithms/tree_lattice_methods/jarrow_rudd/jarrow_rudd.cpp
    src/algorithms/tree_lattice_methods/tian/tian.cpp
    src/algorithms/tree_lattice_methods/leisen_reimer/leisen_reimer.cpp
    src/algorithms/tree_lattice_methods/trinomial_tree/trinomial_tree.cpp
    src/algorithms/tree_lattice_methods/implied_tree/derman_kani.cpp
    src/algorithms/finite_difference_methods/explicit_fd/explicit_fd.cpp
    src/algorithms/finite_difference_methods/implicit_fd/implicit_fd.cpp
    src/algorithms/finite_difference_methods/crank_nicolson/crank_nicolson.cpp
    src/algorithms/finite_difference_methods/adi/adi.cpp
    src/algorithms/finite_difference_methods/psor/psor.cpp
    src/algorithms/monte_carlo_methods/standard_monte_carlo/standard_monte_carlo.cpp
    src/algorithms/monte_carlo_methods/euler_maruyama/euler_maruyama.cpp
    src/algorithms/monte_carlo_methods/milstein/milstein.cpp
    src/algorithms/monte_carlo_methods/longstaff_schwartz/longstaff_schwartz.cpp
    src/algorithms/monte_carlo_methods/quasi_monte_carlo/quasi_monte_carlo.cpp
    src/algorithms/monte_carlo_methods/multilevel_monte_carlo/multilevel_monte_carlo.cpp
    src/algorithms/monte_carlo_methods/importance_sampling/importance_sampling.cpp
    src/algorithms/monte_carlo_methods/control_variates/control_variates.cpp
    src/algorithms/monte_carlo_methods/antithetic_variates/antithetic_variates.cpp
    src/algorithms/monte_carlo_methods/stratified_sampling/stratified_sampling.cpp
)

target_include_directories(quantkernel
    PUBLIC include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_options(quantkernel PRIVATE
    -Wall -Wextra -Wpedantic
    -O3 -ffast-math -march=native -mtune=native
    -funroll-loops -ftree-vectorize
)

set_target_properties(quantkernel PROPERTIES
    OUTPUT_NAME "quantkernel"
    PREFIX "lib"
    SOVERSION 1
    INTERPROCEDURAL_OPTIMIZATION TRUE
)

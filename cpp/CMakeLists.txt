find_package(OpenMP)

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
    src/algorithms/monte_carlo_methods/heston_monte_carlo/heston_monte_carlo.cpp
    src/algorithms/monte_carlo_methods/heston_monte_carlo/heston_lr_delta.cpp
    src/algorithms/monte_carlo_methods/local_vol_monte_carlo/local_vol_monte_carlo.cpp
    src/algorithms/fourier_transform_methods/carr_madan_fft/carr_madan_fft.cpp
    src/algorithms/fourier_transform_methods/cos_method/cos_method.cpp
    src/algorithms/fourier_transform_methods/fractional_fft/fractional_fft.cpp
    src/algorithms/fourier_transform_methods/lewis_fourier_inversion/lewis_fourier_inversion.cpp
    src/algorithms/fourier_transform_methods/hilbert_transform/hilbert_transform.cpp
    src/algorithms/integral_quadrature/gauss_hermite/gauss_hermite.cpp
    src/algorithms/integral_quadrature/gauss_laguerre/gauss_laguerre.cpp
    src/algorithms/integral_quadrature/gauss_legendre/gauss_legendre.cpp
    src/algorithms/integral_quadrature/adaptive_quadrature/adaptive_quadrature.cpp
    src/algorithms/regression_approximation/polynomial_chaos_expansion/polynomial_chaos_expansion.cpp
    src/algorithms/regression_approximation/radial_basis_functions/radial_basis_functions.cpp
    src/algorithms/regression_approximation/sparse_grid_collocation/sparse_grid_collocation.cpp
    src/algorithms/regression_approximation/proper_orthogonal_decomposition/proper_orthogonal_decomposition.cpp
    src/algorithms/adjoint_greeks/pathwise_derivative/pathwise_derivative.cpp
    src/algorithms/adjoint_greeks/likelihood_ratio/likelihood_ratio.cpp
    src/algorithms/adjoint_greeks/aad/aad.cpp
    src/algorithms/machine_learning/deep_bsde/deep_bsde.cpp
    src/algorithms/machine_learning/pinns/pinns.cpp
    src/algorithms/machine_learning/deep_hedging/deep_hedging.cpp
    src/algorithms/machine_learning/neural_sde_calibration/neural_sde_calibration.cpp
)

target_include_directories(quantkernel
    PUBLIC include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(MSVC)
    target_compile_options(quantkernel PRIVATE /W4 /permissive-)
else()
    target_compile_options(quantkernel PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(quantkernel PRIVATE OpenMP::OpenMP_CXX)
endif()

target_precompile_headers(quantkernel PRIVATE
    <cstdint> <cmath> <cstring> <algorithm> <memory> <vector>
)

set_target_properties(quantkernel PROPERTIES
    OUTPUT_NAME "quantkernel"
    PREFIX "lib"
    INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
)

if(QK_BUILD_BENCHMARKS)
    add_executable(qk_bench_bsm_batch
        bench/benchmark_bsm_batch.cpp
    )
    target_link_libraries(qk_bench_bsm_batch PRIVATE quantkernel)
    target_include_directories(qk_bench_bsm_batch PRIVATE include)

    add_executable(qk_bench_all
        bench/benchmark_all.cpp
    )
    target_link_libraries(qk_bench_all PRIVATE quantkernel)
    target_include_directories(qk_bench_all PRIVATE include)
endif()

if(QK_BUILD_TESTS)
    enable_testing()

    set(QK_TEST_SOURCES
        tests/test_closed_form.cpp
        tests/test_tree_lattice.cpp
        tests/test_monte_carlo.cpp
        tests/test_fourier.cpp
        tests/test_error_handling.cpp
        tests/test_fdm_iqm_ram_agm_mlm.cpp
    )

    foreach(test_src ${QK_TEST_SOURCES})
        get_filename_component(test_name ${test_src} NAME_WE)
        add_executable(${test_name} ${test_src})
        target_link_libraries(${test_name} PRIVATE quantkernel)
        target_include_directories(${test_name} PRIVATE include ${CMAKE_CURRENT_SOURCE_DIR}/tests)
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
endif()

install(
    TARGETS quantkernel
    LIBRARY DESTINATION quantkernel
    RUNTIME DESTINATION quantkernel
    ARCHIVE DESTINATION quantkernel
)

file(GLOB QK_PY_MODULES
    "${CMAKE_SOURCE_DIR}/python/quantkernel/*.py"
    "${CMAKE_SOURCE_DIR}/python/quantkernel/*.pyi"
    "${CMAKE_SOURCE_DIR}/python/quantkernel/py.typed"
)

install(
    FILES ${QK_PY_MODULES}
    DESTINATION quantkernel
)

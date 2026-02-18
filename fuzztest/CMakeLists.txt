cmake_minimum_required(VERSION 3.19)
project(algorithm_fuzztest LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Bring in fuzztest (bundles googletest + googlemock)
set(FUZZTEST_FUZZING_MODE OFF CACHE BOOL "" FORCE)
include(FetchContent)
FetchContent_Declare(
    fuzztest
    GIT_REPOSITORY https://github.com/google/fuzztest.git
    GIT_TAG        d8e979e  # matches local clone
)
FetchContent_MakeAvailable(fuzztest)

# Bring in the quantkernel library from sibling directory.
# Override -ffast-math because ASAN (used by fuzztest) conflicts with it.
add_subdirectory(../cpp cpp_build)
get_target_property(_opts quantkernel COMPILE_OPTIONS)
if(_opts)
    list(REMOVE_ITEM _opts "-ffast-math")
    set_target_properties(quantkernel PROPERTIES COMPILE_OPTIONS "${_opts}")
endif()

enable_testing()
include(GoogleTest)

# --- Fuzz test executables ---

add_executable(fuzz_closed_form_models fuzz_closed_form_models.cpp)
target_link_libraries(fuzz_closed_form_models PRIVATE quantkernel gtest gmock)
target_include_directories(fuzz_closed_form_models PRIVATE ../cpp/src)
link_fuzztest(fuzz_closed_form_models)
gtest_discover_tests(fuzz_closed_form_models)

add_executable(fuzz_tree_lattice_models fuzz_tree_lattice_models.cpp)
target_link_libraries(fuzz_tree_lattice_models PRIVATE quantkernel gtest gmock)
target_include_directories(fuzz_tree_lattice_models PRIVATE ../cpp/src)
link_fuzztest(fuzz_tree_lattice_models)
gtest_discover_tests(fuzz_tree_lattice_models)

add_executable(fuzz_finite_difference_models fuzz_finite_difference_models.cpp)
target_link_libraries(fuzz_finite_difference_models PRIVATE quantkernel gtest gmock)
target_include_directories(fuzz_finite_difference_models PRIVATE ../cpp/src)
link_fuzztest(fuzz_finite_difference_models)
gtest_discover_tests(fuzz_finite_difference_models)

add_executable(fuzz_monte_carlo_models fuzz_monte_carlo_models.cpp)
target_link_libraries(fuzz_monte_carlo_models PRIVATE quantkernel gtest gmock)
target_include_directories(fuzz_monte_carlo_models PRIVATE ../cpp/src)
link_fuzztest(fuzz_monte_carlo_models)
gtest_discover_tests(fuzz_monte_carlo_models)

add_executable(fuzz_fourier_transform_models fuzz_fourier_transform_models.cpp)
target_link_libraries(fuzz_fourier_transform_models PRIVATE quantkernel gtest gmock)
target_include_directories(fuzz_fourier_transform_models PRIVATE ../cpp/src)
link_fuzztest(fuzz_fourier_transform_models)
gtest_discover_tests(fuzz_fourier_transform_models)

add_executable(fuzz_integral_quadrature_models fuzz_integral_quadrature_models.cpp)
target_link_libraries(fuzz_integral_quadrature_models PRIVATE quantkernel gtest gmock)
target_include_directories(fuzz_integral_quadrature_models PRIVATE ../cpp/src)
link_fuzztest(fuzz_integral_quadrature_models)
gtest_discover_tests(fuzz_integral_quadrature_models)

add_executable(fuzz_regression_approximation_models fuzz_regression_approximation_models.cpp)
target_link_libraries(fuzz_regression_approximation_models PRIVATE quantkernel gtest gmock)
target_include_directories(fuzz_regression_approximation_models PRIVATE ../cpp/src)
link_fuzztest(fuzz_regression_approximation_models)
gtest_discover_tests(fuzz_regression_approximation_models)

add_executable(fuzz_adjoint_greeks_models fuzz_adjoint_greeks_models.cpp)
target_link_libraries(fuzz_adjoint_greeks_models PRIVATE quantkernel gtest gmock)
target_include_directories(fuzz_adjoint_greeks_models PRIVATE ../cpp/src)
link_fuzztest(fuzz_adjoint_greeks_models)
gtest_discover_tests(fuzz_adjoint_greeks_models)

add_executable(fuzz_machine_learning_models fuzz_machine_learning_models.cpp)
target_link_libraries(fuzz_machine_learning_models PRIVATE quantkernel gtest gmock)
target_include_directories(fuzz_machine_learning_models PRIVATE ../cpp/src)
link_fuzztest(fuzz_machine_learning_models)
gtest_discover_tests(fuzz_machine_learning_models)

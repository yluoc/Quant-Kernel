cmake_minimum_required(VERSION 3.19)
project(algorithm_fuzztest LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Bring in fuzztest (bundles googletest + googlemock)
set(FUZZTEST_FUZZING_MODE OFF CACHE BOOL "" FORCE)
add_subdirectory(fuzztest)

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

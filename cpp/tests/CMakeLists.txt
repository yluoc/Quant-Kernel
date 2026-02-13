add_executable(qk_tests
    test_main.cpp
    test_bs_pricer.cpp
    test_iv_solver.cpp
    test_mc_pricer.cpp
)

target_link_libraries(qk_tests PRIVATE quantkernel)
target_include_directories(qk_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
add_test(NAME qk_tests COMMAND qk_tests)

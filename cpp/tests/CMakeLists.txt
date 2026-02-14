add_executable(qk_tests
    test_main.cpp
    test_closed_form_models.cpp
)

target_link_libraries(qk_tests PRIVATE quantkernel)
target_include_directories(qk_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
)
add_test(NAME qk_tests COMMAND qk_tests)

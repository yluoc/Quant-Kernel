cmake_minimum_required(VERSION 3.14)
project(QuantKernel LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
enable_testing()

add_subdirectory(cpp)

option(QK_BUILD_RUST_RUNTIME "Build Rust runtime safety shell" ON)
if(QK_BUILD_RUST_RUNTIME)
    find_program(CARGO_EXECUTABLE cargo)
    if(NOT CARGO_EXECUTABLE)
        message(FATAL_ERROR "QK_BUILD_RUST_RUNTIME=ON but cargo was not found.")
    endif()

    add_custom_target(quantkernel_runtime
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/rust/runtime/.tmp
        COMMAND ${CMAKE_COMMAND} -E env TMPDIR=${CMAKE_CURRENT_SOURCE_DIR}/rust/runtime/.tmp ${CARGO_EXECUTABLE} build --release
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/rust/runtime
        COMMENT "Building Rust runtime (cdylib)"
    )
endif()

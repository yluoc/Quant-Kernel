cmake_minimum_required(VERSION 3.20)
project(QuantKernel LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(QK_BUILD_TESTS "Build C++ tests" ON)
option(QK_BUILD_BENCHMARKS "Build C++ benchmarks" ON)

if(QK_BUILD_TESTS)
  enable_testing()
endif()

add_subdirectory(cpp)

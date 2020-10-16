cmake_minimum_required(VERSION 3.10)

project(DeepLibrary)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_library(OPENCL_LIB NAME OpenCL)

file(GLOB sources "./core/*.cpp")

add_executable(test test.cpp ${sources})
target_include_directories(test PRIVATE .)
target_link_libraries(test OpenCL)

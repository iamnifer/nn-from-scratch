cmake_minimum_required(VERSION 3.25)
project(nn_from_scratch)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()
add_subdirectory(tests)
add_subdirectory(extern/googletest)

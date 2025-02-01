cmake_minimum_required(VERSION 3.29)
project(nn_from_scratch)

set(CMAKE_CXX_STANDARD 14)

add_executable(nn_from_scratch
        types.h
        activation-function.h
        linear-layer.h
        nonlinear-layer.h
        loss-function.h
    main.cpp)

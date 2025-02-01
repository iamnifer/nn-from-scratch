#pragma once

#include <Eigen/Dense>

template<int N, int M>
using matrix = Eigen::Matrix<double, N, M>;

template<int N>
using col = Eigen::Matrix<double, N, 1>;

template<int M>
using row = Eigen::Matrix<double, 1, M>;

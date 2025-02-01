#pragma once

#include <random>

#include "types.h"

template<int M, int N>
class LinearLayer {
public:
    // x -> Ax + b
    matrix<M, N> A;
    col<M> b;
    LinearLayer() {
        std::mt19937 rng;
        std::normal_distribution<double> dist(0.0, std::sqrt(2.0 / N));

        A = Eigen::MatrixXd::NullaryExpr(M, N, [&]() { return dist(rng); });
        b = Eigen::VectorXd::Zero(M);
    }

    col<N> forward(const col<N>& x) const {
        return A * x + b;
    }

    void update(const col<N>& x, const row<N>& u, double learning_rate) {
        matrix<M, N> dA = (x * u).transpose();
        col<M> db = u.transpose();
        A -= dA * learning_rate;
        b -= db * learning_rate;
    }

    row<N> backward(const col<N>& x, const row<N>& u) const {
        row<N> dx = u * A;
        return dx;
    }
};

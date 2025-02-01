#pragma once

#include "types.h"
#include <functional>

using f_RR = std::function<double(double)>;

template<int M, int N>
class ActivationFunction {
    f_RR sigma;
    f_RR dsigma;

public:
    ActivationFunction(const f_RR& sigma_, const f_RR& dsigma_) : sigma(sigma_), dsigma(dsigma_) {};

    double operator()(double x) const {
        return sigma(x);
    }
    col<N> apply(const col<N>& x) const {
        col<N> res = x;
        for (int i = 0; i < N; ++i) {
            res(i, 0) = sigma(res(i, 0));
        }
        return res;
    }
    col<N> apply_der(const col<N>& x) const {
        col<N> res = x;
        for (int i = 0; i < N; ++i) {
            res(i, 0) = dsigma(res(i, 0));
        }
        return res;
    }
};

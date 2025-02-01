#pragma once

#include "types.h"

class LossFunction {
    public:
    template<int N>
    double dist(const col<N>& z, const col<N>& y) const {
        auto diff = (z - y).array();
        return (diff * diff).sum() / N;
    }
    template<int N>
    row<N> gradient(const col<N>& x_pred, const col<N>& y) const {
        return (2.0 / N) * (x_pred - y).transpose();
    }
};

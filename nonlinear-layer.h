#pragma once

#include "types.h"
#include "activation-function.h"

template<int M, int N>
class NonlinearLayer {
    // x -> sigma(x)
    ActivationFunction<M, N> sigma;
public:
    NonlinearLayer(const ActivationFunction<M, N>& sigma_) : sigma(sigma_) {}
    col<N> forward(const col<N>& x) const {
        return sigma.apply(x);
    }

    void update(const col<N>& x, const row<N>& u, double learning_rate) {}

    row<N> backward(const col<N>& x, const row<N>& u) {
        return u.array() * sigma.apply_der(x).transpose().array();
    }

};

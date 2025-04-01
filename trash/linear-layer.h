// #pragma once
//
// #include <random>
// #include "types.h"
// #include "random.h"
// #include "optimizer.h"
//
// namespace NNFS {
//
//
//
// class LinearLayer {
// public:
//     // x -> Ax + b
//
//     LinearLayer(int m, int n);
//
//     Vector forward(const Vector& x);
//
//     Vector backward(const Vector& grad);
//
//     void update(size_t lyr_index, Optimizer& opt);
//
//     void reset_grads();
//
//     const Matrix& get_A() const;
//
//     const Vector& get_b() const;
//
// private:
//     Matrix A_, grad_A_;
//     Vector b_, grad_b_;
//     Vector x_cached_;
//     // Vector grad_x_;
//     static Random<>& rng();
//
//     friend Optimizer;
// };
// }
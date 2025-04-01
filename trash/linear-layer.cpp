// #pragma once
//
// #include "linear-layer.h"
// #include <iostream>
//
// namespace NNFS {
//
//
//         // x -> Ax + b
//
//     LinearLayer::LinearLayer(int m, int n) {
//         // std::mt19937 rng;
//         // std::normal_distribution<double> dist(0.0, std::sqrt(2.0 / N));
//         //
//         // A = Eigen::MatrixXd::NullaryExpr(M, N, [&]() { return dist(rng); });
//         // b = Eigen::VectorXd::Zero(M);
//         A_ = rng().GenMatrix(m, n);
//         b_ = Vector::Zero(m);
//         std::cout << "init linear lyr " << m << "x" << n << std::endl;
//         std::cout << A_ << std::endl;
//         std::cout << b_ << std::endl;
//         std::cout << "---" << std::endl;
//     }
//
//     Vector LinearLayer::forward(const Vector& x) {
//         x_cached_ = x;
//         return A_ * x + b_;
//     }
//
//     // void LinearLayer::update(size_t lyr_index, const Vector& x, const Vector& u, Optimizer& opt) {
//     //
//     //     auto dA = u * x.transpose();
//     //     // std::cout << dA << std::endl;
//     //     auto db = u;
//     //     // std::cout << db << std::endl;
//     //     A_ -= dA * learning_rate;
//     //     b_ -= db * learning_rate;
//     // }
//     void LinearLayer::update(size_t lyr_index, Optimizer& opt) {
//         opt.update(lyr_index, *this);
//     }
//
//     Vector LinearLayer::backward(const Vector& grad) {
//         grad_A_ += grad * x_cached_.transpose();
//         grad_b_ += grad;
//         // Vector new_grad = A_.transpose() * grad;
//         // update(x, u, learning_rate);
//         // opt.update(lyr_index, A_, b_, x, grad);
//         // grad_x_ = A_.transpose() * grad;
//         return A_.transpose() * grad;
//     }
//
//     void LinearLayer::reset_grads() {
//         grad_A_ = Matrix::Zero(A_.rows(), A_.cols());
//         grad_b_ = Vector::Zero(b_.rows());
//     }
//
//     const Matrix& LinearLayer::get_A() const {
//         return A_;
//     }
//     const Vector& LinearLayer::get_b() const {
//         return b_;
//     }
//
//     Random<>& LinearLayer::rng() {
//         static Random rng;
//         return rng;
//     }
//
// }
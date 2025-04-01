// #include "nonlinear-layer.h"
//
// #include <iostream>
//
// namespace NNFS {
//
//     NonlinearLayer::NonlinearLayer(ActivationFunction sigma) : sigma_(std::move(sigma)) {}
//     Vector NonlinearLayer::forward(const Vector& x) {
//         x_cached_ = x;
//         return sigma_.apply(x);
//     }
//
//     void NonlinearLayer::update(size_t lyr_index, Optimizer& opt)  {
//
//     }
//
//     Vector NonlinearLayer::backward(const Vector& grad) {
//         // std::cout << "x " << x.transpose() << std::endl;
//         // auto jac_t = Matrix(sigma_.get_jac(x).transpose());
//
//         // std::cout << "jac_t " << Matrix(jac_t) << std::endl;
//         // jac_t = jac_t.transpose();
//         // std::cout << "jac_t " << jac_t << std::endl;
//         // if (abs(jac_t(0, 0)) > 1e-5 || abs(jac_t(1, 1)) > 1e-5 || abs(jac_t(2, 2)) > 1e-5) {
//         //     std::cout << "jac_t: " << jac_t.transpose() << std::endl;
//         // }
//         // Vector sus = sigma_.get_jac(x) * u;
//         // std::cout << "sus " << sus << std::endl;
//         grad_x_ = sigma_.get_jac(x_cached_) * grad;
//         return grad_x_;
//     }
//     void NonlinearLayer::reset_grads() {
//
//     }
// }

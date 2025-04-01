#include "layer.h"

namespace NNFS {

    LinearLayer::LinearLayer(int m, int n) {
        A_ = rng().GenMatrix(m, n);
        A_ *= std::sqrt(2.0 / n); // kaiming initialization
        b_ = Vector::Zero(m);
    }

    Vector LinearLayer::forward(const Vector& x) {
        x_cached_ = x;
        return A_ * x + b_;
    }

    void LinearLayer::update(size_t lyr_index, Optimizer& opt) {
        opt.update(lyr_index, *this);
    }

    Vector LinearLayer::backward(const Vector& grad) {
        grad_A_ += grad * x_cached_.transpose();
        grad_b_ += grad;
        return A_.transpose() * grad;
    }

    void LinearLayer::reset_grads() {
        grad_A_ = Matrix::Zero(A_.rows(), A_.cols());
        grad_b_ = Vector::Zero(b_.rows());
    }


    const Matrix& LinearLayer::get_A() const {
        return A_;
    }
    const Vector& LinearLayer::get_b() const {
        return b_;
    }

    Random<>& LinearLayer::rng() {
        static Random<> rng;
        return rng;
    }



    NonlinearLayer::NonlinearLayer(ActivationFunction sigma) : sigma_(std::move(sigma)) {}
    Vector NonlinearLayer::forward(const Vector& x) {
        x_cached_ = x;
        return sigma_.apply(x);
    }

    void NonlinearLayer::update(size_t lyr_index, Optimizer& opt)  {

    }

    Vector NonlinearLayer::backward(const Vector& grad) {
        return sigma_.get_jac(x_cached_) * grad;
    }
    void NonlinearLayer::reset_grads() {

    }
}
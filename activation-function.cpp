#include "activation-function.h"


namespace NNFS {
    Vector ReLU::apply(const Vector& x) const {
        return x.cwiseMax(0);
    }
    Matrix ReLU::get_jac(const Vector& x) const {
        return (x.array() > 0).cast<FType>().matrix().asDiagonal();
    }


    Vector Sigmoid::apply(const Vector& x) const {
        return 1.0 / (1.0 + (-x.array()).exp());
    }
    Matrix Sigmoid::get_jac(const Vector& x) const {
        Vector sig = apply(x);
        return (sig.array() * (1.0 - sig.array())).matrix().asDiagonal();
    }


    Vector Softmax::apply(const Vector& x) const {
        Vector exp_x = (x.array() - x.maxCoeff()).exp();
        return exp_x / exp_x.sum();
    }
    Matrix Softmax::get_jac(const Vector& x) const {
        Vector s = apply(x);
        Matrix jac = -s * s.transpose();
        jac.diagonal() = s.array() * (1.0 - s.array());
        return jac;
    }
}

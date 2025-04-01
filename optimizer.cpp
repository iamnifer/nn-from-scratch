#include "optimizer.h"
#include "layer.h"

namespace NNFS {

    SGD::SGD(FType learning_rate) : learning_rate_(learning_rate) {

    }

    void SGD::update(size_t, LinearLayer& layer) {
        layer.A_ -= layer.grad_A_ * learning_rate_;
        layer.b_ -= layer.grad_b_ * learning_rate_;
    }

    void SGD::step() {}


    Adam::Adam(size_t n_layers, FType lr, FType beta1, FType beta2, FType eps) :
    lr_(lr), beta1_(beta1), beta2_(beta2), beta1_pw_(beta1), beta2_pw_(beta2), eps_(eps),
    m_A_(n_layers), m_b_(n_layers), v_A_(n_layers), v_b_(n_layers) {
    }

    void Adam::update(size_t lyr_index, LinearLayer& layer) {
        assert(lyr_index < m_A_.size());
        auto& m_A = m_A_[lyr_index];
        auto& m_b = m_b_[lyr_index];
        auto& v_A = v_A_[lyr_index];
        auto& v_b = v_b_[lyr_index];
        if (m_A.rows() == 0 && m_A.cols() == 0) {
            m_A = Matrix::Zero(layer.A_.rows(), layer.A_.cols());
            m_b = Matrix::Zero(layer.b_.rows(), layer.b_.cols());
            v_A = Matrix::Zero(layer.A_.rows(), layer.A_.cols());
            v_b = Matrix::Zero(layer.b_.rows(), layer.b_.cols());
        }

        m_A = beta1_ * m_A + (1.0 - beta1_) * layer.grad_A_;
        m_b = beta1_ * m_b + (1.0 - beta1_) * layer.grad_b_;

        v_A = beta2_ * v_A + (1.0 - beta2_) * (layer.grad_A_.array().square().matrix());
        v_b = beta2_ * v_b + (1.0 - beta2_) * (layer.grad_b_.array().square().matrix());

        auto m_A_hat = m_A / (1.0 - beta1_pw_);
        auto m_b_hat = m_b / (1.0 - beta1_pw_);
        auto v_A_hat = v_A / (1.0 - beta2_pw_);
        auto v_b_hat = v_b / (1.0 - beta2_pw_);

        layer.A_ -= lr_ * Matrix(m_A_hat.array() / (v_A_hat.array().sqrt() + eps_));
        layer.b_ -= lr_ * Matrix(m_b_hat.array() / (v_b_hat.array().sqrt() + eps_));
    }

    void Adam::step() {
        beta1_pw_ *= beta1_;
        beta2_pw_ *= beta2_;
    }
}
#include <iostream>
#include "linear-layer.h"
#include "nonlinear-layer.h"
#include "loss-function.h"
#include "types.h"
#include <vector>

constexpr int M = 2, N = 2;
template<int N>
void print_col(const col<N>& x) {
    std::cout << '(';
    for (int i = 0; i < N; ++i) {
        std::cout << x[i];
        if (i < N - 1) {
            std::cout << " ";
        }
    }
    std::cout << ")^T";
}

int main() {
    matrix<M, N> f_A({{1, 2}, {3, 4}});
    col<N> f_b({{5}, {6}});

    int train_size = 2;
    std::vector<col<N>> xs, ys;
    xs.push_back(col<N>{{-1}, {1}});
    xs.push_back(col<N>{{2}, {3}});

    std::cout << "training data:" << std::endl;
    for (auto& x : xs) {
        ys.push_back(f_A * x + f_b);
        print_col(x);
        std::cout << " -> ";
        print_col(ys.back());
        std::cout << std::endl;
    }

    LinearLayer<M, N> l1;
    std::cout << "linear layer initialization:" << std::endl;
    std::cout << l1.A << std::endl;
    std::cout << l1.b << std::endl;
    std::cout << std::endl;
    LossFunction loss_f;

    double learning_rate = 0.2;
    std::cout.precision(4);
    for (int epoch = 0; epoch < 40; ++epoch) {
        std::cout << "epoch: " << epoch << std::endl;

        for (int i = 0; i < train_size; ++i) {
            auto x = xs[i];
            auto x_ = l1.forward(x);
            double loss = loss_f.dist(x_, ys[i]);

            std::cout << loss << ' ';
            auto begin_grad = loss_f.gradient(x_, ys[i]);
            l1.update(x, begin_grad, learning_rate);
        }

        std::cout << std::endl;
        learning_rate *= 0.95;
    }
}

#include "gtest/gtest.h"
#include "../neural-network.h"

using namespace NNFS;

TEST(BasicTests, Ax_plus_b) {
    const size_t x_dim = 2;
    const size_t dataset_size = 100;

    Random rng;
    Matrix f_A({{1, 2}, {3, 4}});
    Vector f_b({{5}, {6}});
    auto f = [&](Vector x) -> Vector {
        return f_A * x + f_b;
    };
    auto dataset = rng.GenDataset(dataset_size, x_dim, f);
    auto [train, test] = TrainTestSplit(dataset);

    NeuralNetwork nn = std::vector<Layer>{LinearLayer(2, 2)};
    LossFunction loss_f = MSE();

    TrainConfig basic_config = {
        train,
        test,
        train.size() / 4,
        400,
        100,
        SGD(0.01),
        loss_f
    };
    nn.train(basic_config);
    for (auto& [x, y] : dataset) {
        auto y_pred = nn.forward(x);
        EXPECT_LE(loss_f.dist(y_pred, y), 1e-8);
    }
}

TEST(BasicTests, sin) {
    Random<> rng;
    size_t dataset_size = 10000;
    size_t x_dim = 1;
    auto f = [&](Vector x) -> Vector {
        return x.array().sin();
    };
    auto dataset = rng.GenDataset(dataset_size, x_dim, f);
    auto [train, test] = TrainTestSplit(dataset);

    NeuralNetwork nn = std::vector<Layer>{
        LinearLayer(32, x_dim),
        NonlinearLayer(ReLU()),
        LinearLayer(32, 32),
        NonlinearLayer(Sigmoid()),
        LinearLayer(x_dim, 32),
    };
    LossFunction loss_f = MSE();

    TrainConfig basic_config = {
        train,
        test,
        1,
        20,
        1,
        Adam(5, 1e-3, 0.9, 0.999, 1e-8),
        loss_f
    };
    nn.train(basic_config);
    VectorSet pred_test = nn.forward(GetXs(test));
    FType avg_loss = loss_f.avg_dist(test, pred_test);
    std::cout << "test avg_loss: " << avg_loss << std::endl;
    EXPECT_LE(avg_loss, 1e-4);
}

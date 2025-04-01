#include "gtest/gtest.h"
#include "../neural-network.h"

using namespace NNFS;

TEST(MNIST, small) {
    Dataset mnist_train = LoadMNISTFromCSV("../../data/mnist_train_small.csv", true);
    Dataset mnist_test = LoadMNISTFromCSV("../../data/mnist_test_small.csv", true);

    NeuralNetwork nn = std::vector<Layer>{
        LinearLayer(128, 784),
        NonlinearLayer(ReLU()),
        LinearLayer(10, 128),
        NonlinearLayer(Softmax()),
    };
    LossFunction loss_f = CrossEntropy();
    TrainConfig mnist_config = {
        mnist_train,
        mnist_test,
        64,
        3,
        1,
        Adam(4, 1e-3, 0.9, 0.999, 1e-8),
        loss_f
    };
    std::cout << "training size: " << mnist_train.size() << std::endl;
    std::cout << "test size: " << mnist_test.size() << std::endl;
    nn.train(mnist_config);
    VectorSet xs_test = GetXs(mnist_test);
    VectorSet ys_test = GetYs(mnist_test);
    std::cout << ys_test[0].transpose() << std::endl;
    std::cout << ys_test[1].transpose() << std::endl;
    auto classes_test = GetClasses(ys_test);
    auto classes_pred = GetClasses(nn.forward(xs_test));
    std::cout << "classes pred: ";
    for (auto& i : classes_pred) std::cout << i << " "; std::cout << std::endl;
    std::cout << "classes test: ";
    for (auto& i : classes_test) std::cout << i << " "; std::cout << std::endl;
    FType acc = GetAccuracy(classes_pred, classes_test);
    std::cout << "acc: " << acc << std::endl;
    EXPECT_GE(acc, 0.90);
}

TEST(MNIST, full) {
    Dataset mnist_train = LoadMNISTFromCSV("../../data/mnist_train.csv", true);
    Dataset mnist_test = LoadMNISTFromCSV("../../data/mnist_test.csv", true);
    NeuralNetwork nn = std::vector<Layer>{
        LinearLayer(128, 784),
        NonlinearLayer(ReLU()),

        LinearLayer(10, 128),
        NonlinearLayer(Softmax()),
    };
    LossFunction loss_f = CrossEntropy();
    TrainConfig mnist_config = {
        mnist_train,
        mnist_test,
        64,
        3,
        1,
        Adam(4, 1e-3, 0.9, 0.999, 1e-8),
        loss_f
    };
    std::cout << "training size: " << mnist_train.size() << std::endl;
    std::cout << "test size: " << mnist_test.size() << std::endl;
    nn.train(mnist_config);
    VectorSet xs_test = GetXs(mnist_test);
    VectorSet ys_test = GetYs(mnist_test);
    std::cout << ys_test[0].transpose() << std::endl;
    std::cout << ys_test[1].transpose() << std::endl;
    auto classes_test = GetClasses(ys_test);
    auto classes_pred = GetClasses(nn.forward(xs_test));
    FType acc = GetAccuracy(classes_pred, classes_test);
    std::cout << "acc: " << acc << std::endl;
    EXPECT_GE(acc, 0.95);
}
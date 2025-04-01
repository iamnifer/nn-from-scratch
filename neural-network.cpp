#include "neural-network.h"
#include <iostream>

namespace NNFS {

    NeuralNetwork::NeuralNetwork(std::vector<Layer> layers) : layers_(std::move(layers)) {
    }

    Vector NeuralNetwork::forward(const Vector &x) {
        Vector res = x;
        for (auto& layer : layers_) {
            res = std::move(layer.forward(res));
        }
        return res;
    }
    VectorSet NeuralNetwork::forward(const VectorSet &xs) {
        VectorSet pred;
        for (auto& x : xs) {
            pred.push_back(forward(x));
        }
        return pred;
    }

    void NeuralNetwork::backward(Vector grad) {
        for (int i = int(layers_.size()) - 1; i >= 0; --i) {
            grad = std::move(layers_[i].backward(grad));
        }
    }
    void NeuralNetwork::train_epoch(const TrainConfig &config) {
        auto& dataset = config.train;
        size_t n = dataset.size();
        assert(n != 0);
        for (size_t batch_begin = 0; batch_begin < n; batch_begin += config.batch_size) {
            for (auto& layer : layers_) layer.reset_grads();

            size_t batch_end = std::min(batch_begin + config.batch_size, n);
            size_t this_batch_size = batch_end - batch_begin;
            for (size_t i = batch_begin; i < batch_end; ++i) {
                auto& [x, y] = dataset[i];
                Vector y_pred = forward(x);
                Vector grad = config.loss_f.gradient(y_pred, y);
                backward(grad / this_batch_size);
            }

            for (size_t lyr_ind = 0; lyr_ind < layers_.size(); ++lyr_ind) {
                layers_[lyr_ind].update(lyr_ind, config.optimizer);
            }
            config.optimizer.step();
        }
    }

    void NeuralNetwork::train(const TrainConfig& config) {
        for (size_t cur_epoch = 0; cur_epoch < config.num_epochs; ++cur_epoch) {
            train_epoch(config);
            if (config.info_per_epochs && cur_epoch % config.info_per_epochs == 0) {
                auto xs = GetXs(config.test);
                auto pred = forward(xs);
                FType loss = config.loss_f.avg_dist(config.test, pred);
                std::cout << "epoch #" << cur_epoch << "; avg_loss: " << loss << std::endl;
            }
        }
    }
    const Layer& NeuralNetwork::get_layer(int index) const {
        return layers_.at(index);
    }
}
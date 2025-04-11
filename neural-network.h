#pragma once

#include <vector>

#include "layer.h"
#include "loss-function.h"
#include "types.h"
#include "utils.h"

namespace NNFS
{

class NeuralNetwork
{
   public:
    NeuralNetwork(std::vector<Layer> layers);
    Vector forward(const Vector& x);
    VectorSet forward(const VectorSet& xs);
    void backward(Vector grad);
    void train_epoch(const TrainConfig& config);
    void train(const TrainConfig& config);
    const Layer& get_layer(int index) const;

   private:
    std::vector<Layer> layers_;
};
}  // namespace NNFS

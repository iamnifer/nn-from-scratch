#pragma once

#include "loss-function.h"
#include "optimizer.h"
#include "types.h"

namespace NNFS
{
struct TrainConfig
{
    Dataset train;
    Dataset test;
    size_t batch_size;
    size_t num_epochs;
    size_t info_per_epochs;
    mutable Optimizer optimizer;
    LossFunction loss_f;
};

std::pair<Dataset, Dataset> TrainTestSplit(const Dataset& dataset,
                                           double ratio = 0.8);
VectorSet GetXs(const Dataset& dataset);
VectorSet GetYs(const Dataset& dataset);
Dataset LoadMNISTFromCSV(const std::string& filename, bool labels);
std::vector<size_t> GetClasses(const VectorSet& ys_pred);
FType GetAccuracy(const std::vector<size_t>& pred,
                  const std::vector<size_t>& real);

}  // namespace NNFS
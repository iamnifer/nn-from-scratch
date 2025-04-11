#pragma once
#include <eigen/Eigen/Dense>

namespace NNFS
{
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using RowVector = Eigen::RowVectorXd;
using VectorSet = std::vector<Vector>;
using FType = double;
struct Sample
{
    Vector x, y;
};
using Dataset = std::vector<Sample>;
}  // namespace NNFS
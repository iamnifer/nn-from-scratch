#pragma once

#include <eigenrand/EigenRand/EigenRand>
#include <functional>
#include "types.h"


namespace NNFS {
    template <typename Dist = Eigen::Rand::NormalGen<FType>>
    class Random {
    public:
        template <typename... DistArgs>
        Random(uint64_t seed = kDefaultSeed, DistArgs... args) : gen_(seed), dist_(std::forward<DistArgs>(args)...) {}
        Matrix GenMatrix(size_t rows, size_t cols) {
            return dist_.template generate<Matrix>(rows, cols, gen_);
        }
        Vector GenVector(size_t n) {
            return dist_.template generate<Matrix>(n, 1, gen_);
        }
        std::vector<Sample> GenDataset(size_t size, size_t x_dim, std::function<Vector(Vector)> f) {
            std::vector<Sample> dataset;
            for (size_t i = 0; i < size; ++i) {
                Vector x = GenVector(x_dim);
                dataset.push_back({x, f(x)});
            }
            return dataset;
        }
        static constexpr uint64_t kDefaultSeed = 123;
    private:
        Dist dist_;
        Eigen::Rand::Vmt19937_64 gen_;
    };
}
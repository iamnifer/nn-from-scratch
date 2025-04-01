#pragma once
#include "extern/boost/te.h"
#include "types.h"

namespace te = boost::te;

namespace NNFS
{

class LinearLayer;

struct IOpt
{
    void update(size_t lyr_index, LinearLayer& lyr)
    {
        te::call(update_impl, *this, lyr_index, lyr);
    }
    void step()
    {
        te::call([](auto& self) { self.step(); }, *this);
    }

   private:
    static constexpr auto update_impl =
        [](auto& self, auto lyr_index, auto& lyr)
    { return self.update(lyr_index, lyr); };
};
using Optimizer = te::poly<IOpt>;

class SGD
{
   public:
    SGD(FType learning_rate);
    void update(size_t lyr_index, LinearLayer& layer);
    void step();

   private:
    FType learning_rate_;
};

class Adam
{
   public:
    Adam(size_t n_layers, FType lr, FType beta1, FType beta2, FType eps);
    void update(size_t lyr_index, LinearLayer& layer);
    void step();

   private:
    FType lr_;
    FType beta1_;
    FType beta2_;
    FType beta1_pw_;
    FType beta2_pw_;
    FType eps_;

    std::vector<Matrix> m_A_;
    std::vector<Vector> m_b_;
    std::vector<Matrix> v_A_;
    std::vector<Vector> v_b_;
};
}  // namespace NNFS

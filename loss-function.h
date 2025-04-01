#pragma once

#include <boost/te.h>

#include "types.h"

namespace te = boost::te;

namespace NNFS
{

struct ILossFun
{
    FType dist(const Vector& z, const Vector& y) const
    {
        return te::call<FType>(dist_impl, *this, z, y);
    }
    FType avg_dist(const Dataset& d, const VectorSet& pred) const
    {
        return te::call<FType>(avg_dist_impl, *this, d, pred);
    }
    Vector gradient(const Vector& y_pred, const Vector& y) const
    {
        return te::call<Vector>(gradient_impl, *this, y_pred, y);
    }

   private:
    static constexpr auto dist_impl =
        [](auto& self, const auto& z, const auto& y)
    { return self.dist(z, y); };
    static constexpr auto avg_dist_impl =
        [](auto& self, const auto& d, const auto& pred)
    { return self.avg_dist(d, pred); };

    static constexpr auto gradient_impl =
        [](auto& self, const auto& y_pred, const auto& y)
    { return self.gradient(y_pred, y); };
};
using LossFunction = te::poly<ILossFun>;

class MSE
{
   public:
    FType dist(const Vector& z, const Vector& y) const;
    FType avg_dist(const Dataset& d, const VectorSet& pred) const;
    Vector gradient(const Vector& y_pred, const Vector& y) const;
};

class CrossEntropy
{
   public:
    FType dist(const Vector& z, const Vector& y) const;
    FType avg_dist(const Dataset& d, const VectorSet& pred) const;
    Vector gradient(const Vector& y_pred, const Vector& y) const;
};
}  // namespace NNFS

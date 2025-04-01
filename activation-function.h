#pragma once

#include "boost/te.h"
#include "types.h"

namespace te = boost::te;

namespace NNFS
{
struct IActFun
{
    Vector apply(const Vector& x) const
    {
        return te::call<Vector>(apply_impl, *this, x);
    }
    Matrix get_jac(const Vector& x) const
    {
        return te::call<Matrix>(get_jac_impl, *this, x);
    }

   private:
    static constexpr auto apply_impl = [](auto& self, const auto& x)
    { return self.apply(x); };
    static constexpr auto get_jac_impl = [](auto& self, const auto& x)
    { return self.get_jac(x); };
};
using ActivationFunction = te::poly<IActFun>;

class ReLU
{
   public:
    Vector apply(const Vector& x) const;
    Matrix get_jac(const Vector& x) const;
};
class Sigmoid
{
   public:
    Vector apply(const Vector& x) const;
    Matrix get_jac(const Vector& x) const;
};
class Softmax
{
   public:
    Vector apply(const Vector& x) const;
    Matrix get_jac(const Vector& x) const;
};
}  // namespace NNFS

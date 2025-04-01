#pragma once

#include "activation-function.h"
#include "boost/te.h"
#include "optimizer.h"
#include "random.h"
#include "types.h"

namespace te = boost::te;

namespace NNFS
{
struct ILayer
{
    Vector forward(const Vector& x)
    {
        return te::call<Vector>(forward_impl, *this, x);
    }

    Vector backward(const Vector& grad)
    {
        return te::call<Vector>(backward_impl, *this, grad);
    }

    void update(size_t lyr_index, Optimizer& opt)
    {
        te::call(update_impl, *this, lyr_index, opt);
    }

    void reset_grads()
    {
        te::call([](auto& self) { self.reset_grads(); }, *this);
    }

   private:
    static constexpr auto forward_impl = [](auto& self, const auto& x)
    { return self.forward(x); };
    static constexpr auto backward_impl = [](auto& self, const auto& grad)
    { return self.backward(grad); };
    static constexpr auto update_impl =
        [](auto& self, auto lyr_index, auto& opt)
    { return self.update(lyr_index, opt); };
};
using Layer = te::poly<ILayer>;

class LinearLayer
{
   public:
    // x -> Ax + b

    LinearLayer(int m, int n);

    Vector forward(const Vector& x);

    Vector backward(const Vector& grad);

    void update(size_t lyr_index, Optimizer& opt);

    void reset_grads();

    const Matrix& get_A() const;

    const Vector& get_b() const;

   private:
    Matrix A_, grad_A_;
    Vector b_, grad_b_;
    Vector x_cached_;
    static Random<>& rng();

    friend SGD;
    friend Adam;
};

class NonlinearLayer
{
    // x -> sigma(x)
   public:
    NonlinearLayer(ActivationFunction sigma);
    Vector forward(const Vector& x);

    Vector backward(const Vector& grad);

    void update(size_t lyr_index, Optimizer& opt);

    void reset_grads();

   private:
    ActivationFunction sigma_;
    Vector x_cached_;
};
}  // namespace NNFS

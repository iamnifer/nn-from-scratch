// #pragma once
//
// #include "types.h"
// #include "activation-function.h"
// #include "optimizer.h"
//
// namespace NNFS {
//     class NonlinearLayer {
//         // x -> sigma(x)
//     public:
//         NonlinearLayer(ActivationFunction sigma);
//         Vector forward(const Vector& x);
//
//         Vector backward(const Vector& grad);
//
//         void update(size_t lyr_index, Optimizer& opt);
//
//         void reset_grads();
//     private:
//         ActivationFunction sigma_;
//         Vector x_cached_;
//         Vector grad_x_;
//     };
// }

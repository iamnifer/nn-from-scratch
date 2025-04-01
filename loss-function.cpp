#include "loss-function.h"

#include <iostream>

namespace NNFS {
    FType MSE::dist(const Vector& y_pred, const Vector& y) const {
        assert(y_pred.size() == y.size());
        if (y_pred.size() == 0) return 0;
        return ((y_pred - y).squaredNorm()) / y_pred.size();
    }
    FType MSE::avg_dist(const Dataset& d, const VectorSet& pred) const {
        assert(d.size() == pred.size());
        FType res = 0;
        for (size_t i = 0; i < d.size(); i++) {
            auto& [x, y] = d[i];
            auto& y_pred = pred[i];
            res += dist(y_pred, y);
        }
        return res / d.size();
    }
    Vector MSE::gradient(const Vector& y_pred, const Vector& y) const {
        assert(y.size() == y_pred.size());
        if (y.size() == 0) return Vector::Zero(0);
        return (2.0 / y.size()) * (y_pred - y);
    }

    FType CrossEntropy::dist(const Vector& y_pred, const Vector& y) const {
        assert(y_pred.size() == y.size());
        if (y_pred.size() == 0) return 0;
        return -(y.array() * y_pred.array().log()).sum();
    }

    FType CrossEntropy::avg_dist(const Dataset& d, const VectorSet& pred) const {
        assert(d.size() == pred.size());
        FType res = 0;
        for (size_t i = 0; i < d.size(); i++) {
            auto& [x, y] = d[i];
            auto& y_pred = pred[i];
            res += dist(y_pred, y);
        }
        return res / d.size();
    }

    Vector CrossEntropy::gradient(const Vector& y_pred, const Vector& y) const {
        assert(y.size() == y_pred.size());
        if (y.size() == 0) return Vector::Zero(0);
        return -(y.array() / y_pred.array());
    }
}
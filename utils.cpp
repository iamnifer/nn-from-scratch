#include "utils.h"
#include <fstream>
#include <ranges>

namespace NNFS {

    std::pair<Dataset, Dataset> TrainTestSplit(const Dataset& dataset, double ratio) {
        Dataset train, test;
        size_t train_sz = dataset.size() * ratio;
        train.insert(train.begin(), dataset.begin(), dataset.begin() + train_sz);
        test.insert(test.begin(), dataset.begin() + train_sz, dataset.end());
        return std::make_pair(train, test);
    }
    VectorSet GetXs(const Dataset& dataset) {
        VectorSet xs;
        for (auto& [x, y] : dataset) {
            xs.push_back(x);
        }
        return xs;
    }
    VectorSet GetYs(const Dataset& dataset) {
        VectorSet ys;
        for (auto& [x, y] : dataset) {
            ys.push_back(y);
        }
        return ys;
    }
    static constexpr size_t mnist_h = 28;
    static constexpr size_t mnist_pix = mnist_h * mnist_h;
    Dataset LoadMNISTFromCSV(const std::string& filename, bool labels) {
        Dataset dataset;
        std::ifstream in(filename);
        std::string line;
        if (labels) {
            std::getline(in, line);
        }
        while (std::getline(in, line)) {
            auto split_view = line | std::views::split(',');
            std::vector<int> pic;
            pic.reserve(1 + mnist_pix);
            for (const auto& num_view : split_view) {
                std::string num_str(num_view.begin(), num_view.end());
                int num = std::stoi(num_str);
                pic.push_back(num);
            }
            int label = pic[0];
            pic.erase(pic.begin());
            Vector x(pic.size());
            for (size_t i = 0; i < pic.size(); i++) x(i) = pic[i];
            x /= 256;
            Vector y = Vector::Zero(10);
            y(label) = 1;
            dataset.push_back({x, y});
        }
        return dataset;
    }
    std::vector<size_t> GetClasses(const VectorSet& ys_pred) {
        std::vector<size_t> classes;
        classes.reserve(ys_pred.size());
        for (auto& y_pred : ys_pred) {
            size_t max_prob = 0;
            for (size_t i = 0; i < y_pred.rows(); ++i) {
                if (y_pred(i) > y_pred(max_prob)) {
                    max_prob = i;
                }
            }
            classes.push_back(max_prob);
        }
        return classes;
    }
    FType GetAccuracy(const std::vector<size_t>& pred, const std::vector<size_t>& real) {
        assert(pred.size() == real.size());
        size_t cnt = 0;
        for (size_t i = 0; i < pred.size(); ++i) {
            cnt += pred[i] == real[i];
        }
        return static_cast<FType>(cnt) / pred.size();
    }
}
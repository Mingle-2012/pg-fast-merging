//
// Created by XiaoWu on 2024/11/23.
//

#include <cmath>
#include <cstdlib>
#include <valarray>

#ifndef MYANNS_METRICS_H
#define MYANNS_METRICS_H

namespace graph::metric {
struct l2 {
    template <typename T>
    static float
    apply(const T* x, const T* y, unsigned dim) {
        float r = 0;
        for (unsigned i = 0; i < dim; ++i) {
            float v = float(x[i]) - float(y[i]);
            v *= v;
            r += v;
        }
        return std::sqrt(r);
    }
};

struct angular {
    template <typename T>
    static float
    apply(const T* x, const T* y, unsigned dim) {
        float dot = 0;
        float norm_x = 0;
        float norm_y = 0;
        for (size_t i = 0; i < dim; ++i) {
            dot += float(x[i]) * float(y[i]);
            norm_x += float(x[i]) * float(x[i]);
            norm_y += float(y[i]) * float(y[i]);
        }
        return 1 - dot / (std::sqrt(norm_x) * std::sqrt(norm_y));
    }
};
}  // namespace graph::metric

#endif  // MYANNS_METRICS_H

//
// Created by XiaoWu on 2024/11/23.
//

#ifndef MERGE_VAMANA_H
#define MERGE_VAMANA_H

#include <random>
#include <omp.h>
#include "graph.h"
#include "dtype.h"
#include "metric.h"
#include "logger.h"
#include "timer.h"

using namespace merge;

namespace diskann {
class Vamana {
    private:
        /**
         * alpha
         */
        float alpha_;

        /*
         * search pool size
         */
        int L_;

        /**
         * maximum number of neighbors
         */
        int R_;

        void RobustPrune(Graph &graph,
                         IndexOracle &oracle,
                         float alpha,
                         int point,
                         std::vector<Neighbor> &candidates);

    public:
        Vamana(float alpha,
               int L,
               int R);

        void set_alpha(float alpha);

        void set_L(int L);

        void set_R(int R);

        Graph build(IndexOracle &oracle);
    };

}

#endif //MERGE_VAMANA_H

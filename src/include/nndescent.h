//
// Created by XiaoWu on 2024/11/23.
//

#ifndef MERGE_NNDESCENT_H
#define MERGE_NNDESCENT_H

#include <random>
#include <omp.h>
#include "graph.h"
#include "dtype.h"
#include "metric.h"
#include "logger.h"
#include "timer.h"

using namespace merge;

namespace nndescent {

class NNDescent {
    private:
        unsigned K{64};

        float rho{0.5};

        float delta{0.001};

        unsigned iteration{100};

        void initializeGraph(Graph &graph,
                             IndexOracle &oracle);

        void generateUpdate(Graph &graph);

        int applyUpdate(unsigned sample,
                        Graph &graph,
                        IndexOracle &oracle);

        void clearGraph(Graph &graph);

    public:
        NNDescent() = default;

        explicit NNDescent(int K, float rho=0.5, float delta=0.001, int iteration=20)
                : K(K), rho(rho), delta(delta), iteration(iteration) {}

        ~NNDescent() = default;

        Graph build(IndexOracle &oracle);
    };
}

#endif //MERGE_NNDESCENT_H

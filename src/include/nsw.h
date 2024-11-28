//
// Created by XiaoWu on 2024/11/23.
//

#ifndef MERGE_NSW_H
#define MERGE_NSW_H

#include <random>
#include <omp.h>
#include "graph.h"
#include "dtype.h"
#include "metric.h"
#include "logger.h"
#include "timer.h"

using namespace merge;

namespace nsw {
class NSW {
    private:
        int max_neighbors_;

        int ef_construction_;

        void addPoint(Graph &graph,
                      IndexOracle &oracle,
                      unsigned index);

        std::vector<Neighbor> multisearch(const Graph &graph,
                                          const IndexOracle &oracle,
                                          unsigned query,
                                          int attempts,
                                          int k);

    public:
        NSW(int max_neighbors,
            int ef_construction) : max_neighbors_(max_neighbors),
                                   ef_construction_(ef_construction) {}

        void set_max_neighbors(int max_neighbors) {
            this->max_neighbors_ = max_neighbors;
        }

        void set_ef_construction(int ef_construction) {
            this->ef_construction_ = ef_construction;
        }

        Graph build(IndexOracle &oracle);
    };
}

#endif //MERGE_NSW_H

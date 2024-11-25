//
// Created by XiaoWu on 2024/11/23.
//

#ifndef MERGE_TAUMNG_H
#define MERGE_TAUMNG_H

#include <random>
#include <omp.h>
#include "graph.h"
#include "dtype.h"
#include "metric.h"

using namespace merge;

namespace mng {
template<class DATA_TYPE>
class TauMNG {
private:
    /**
     * tau
     */
    float t_;

    /**
     * same as k in knn search
     */
    int h_;

    /**
     * search pool size
     */
    int b_;

public:
    TauMNG(float t,
           int h,
           int b) : t_(t), h_(h), b_(b) {}

    void set_b(int b) {
        this->b_ = b;
    }

    void set_h(int h) {
        this->h_ = h;
    }

    void build(Graph &graph,
               IndexOracle<DATA_TYPE> &oracle);

};

template<class DATA_TYPE>
void TauMNG<DATA_TYPE>::build(Graph &graph,
                              IndexOracle<DATA_TYPE> &oracle) {
    std::vector<int> final_graph, offsets;
    project(graph, final_graph, offsets);

#pragma omp parallel for schedule(dynamic, 256)
    for (int u = 0; u < graph.size(); ++u) {
        auto H_u_ = search(oracle, final_graph, offsets, oracle[u], h_, oracle.size(), b_);
        for (auto &v: H_u_) {
            if (u == v.id) {
                continue;
            }
            bool exist = false;
            for (auto &w: graph[u].candidates_) {
                if (w.id == v.id) {
                    exist = true;
                    break;
                }
            }
            if (exist) {
                continue;
            }
            if (v.distance <= 3 * t_) {
                graph[u].addNeighbor(v);
            } else {
                bool flag = false;
                for (auto &w: graph[u].candidates_) {
                    auto dist = oracle(w.id, v.id);
                    if (dist <= v.distance - 3 * t_) {
                        flag = true;
                        break;
                    }
                }
                if (!flag) {
                    graph[u].addNeighbor(v);
                }
            }
        }
    }
}
}

#endif //MERGE_TAUMNG_H

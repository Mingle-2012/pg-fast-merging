#include "include/taumng.h"

using namespace merge;


taumng::TauMNG::TauMNG(float t,
                          int h,
                          int b) {
    t_ = t;
    h_ = h;
    b_ = b;
}


void taumng::TauMNG::set_b(int b) {
    this->b_ = b;
}


void taumng::TauMNG::set_h(int h) {
    this->h_ = h;
}


void taumng::TauMNG::build(Graph &graph,
                           IndexOracle &oracle) {
    Timer timer;
    timer.start();

    std::vector<int> final_graph, offsets;
    project(graph, final_graph, offsets);

#pragma omp parallel for schedule(dynamic, 256)
    for (int u = 0; u < graph.size(); ++u) {
        if (u % 10000 == 0) {
            logger << "Processing " << u << " / " << graph.size() << std::endl;
        }
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

    timer.end();
    logger << "Construction time: " << timer.elapsed() << "s" << std::endl;
}
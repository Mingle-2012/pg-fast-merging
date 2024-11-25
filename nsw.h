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

using namespace merge;

namespace nsw {
template<class DATA_TYPE>
class NSW {
private:
    int max_neighbors_;

    int ef_construction_;

    void addPoint(Graph &graph,
                  IndexOracle<DATA_TYPE> &oracle,
                  unsigned index);

    std::vector<Neighbor> multisearch(const Graph &graph,
                                      const IndexOracle<DATA_TYPE> &oracle,
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

    Graph build(IndexOracle<DATA_TYPE> &oracle);
};

template<class DATA_TYPE>
Graph NSW<DATA_TYPE>::build(IndexOracle<DATA_TYPE> &oracle) {
    Graph graph;
    int total = oracle.size();
    graph.emplace_back(max_neighbors_);
    for (int i = 1; i < total; ++i) {
        addPoint(graph, oracle, i);
    }

    return graph;
}

template<class DATA_TYPE>
void NSW<DATA_TYPE>::addPoint(Graph &graph,
                              IndexOracle<DATA_TYPE> &oracle,
                              unsigned int index) {
    auto res = knn_search(oracle, graph, oracle[index], max_neighbors_, ef_construction_, -1, index);
    graph.emplace_back(max_neighbors_);
    for (int x = 0; x < max_neighbors_; ++x) {
        if (res[x].id == index || res[x].id == -1) {
            continue;
        }
        graph[index].addNeighbor(res[x]);
        graph[res[x].id].addNeighbor(Neighbor(index, res[x].distance, false));
    }
}

template<class DATA_TYPE>
std::vector<Neighbor> NSW<DATA_TYPE>::multisearch(const Graph &graph,
                                                  const IndexOracle<DATA_TYPE> &oracle,
                                                  unsigned int query,
                                                  int attempts,
                                                  int k) {
    std::priority_queue<Neighbor> candidates;
    std::vector<Neighbor> results;
    std::vector<bool> visited(graph.size(), false);
    std::mt19937 rng(2024);
    for (int it = 0; it < attempts; ++it) {
        std::vector<Neighbor> temp_results;

        int entry_point = rng() % graph.size();
        while (visited[entry_point]) {
            entry_point = rng() % graph.size();
        }
        auto dist = oracle(query, entry_point);
        candidates.push(Neighbor(entry_point, dist, false));
        while (!candidates.empty()) {
            auto &c = candidates.top();
            candidates.pop();

            if (results.size() >= k && c.distance > results[k - 1].distance) {
                break;
            }

            for (auto &v: graph[c.id].candidates_) {
                if (!visited[v.id]) {
                    visited[v.id] = true;
                    dist = oracle(query, v.id);
                    temp_results.emplace_back(v.id, dist, false);
                    candidates.push(Neighbor(v.id, dist, false));
                }
            }
        }
        results.insert(results.end(), temp_results.begin(), temp_results.end());
        std::sort(results.begin(), results.end());
        results.erase(std::unique(results.begin(), results.end(), [](const Neighbor &a,
                                                                     const Neighbor &b) {
            return a.id == b.id;
        }), results.end());
    }
    results.resize(k);
    return results;
}
}

#endif //MERGE_NSW_H

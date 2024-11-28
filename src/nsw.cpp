#include "include/nsw.h"

using namespace merge;

Graph nsw::NSW::build(IndexOracle &oracle) {
    Timer timer;
    timer.start();

    Graph graph;
    int total = oracle.size();
    graph.emplace_back(max_neighbors_);

    for (int i = 1; i < total; ++i) {
        if (i % 10000 == 0) {
            logger << "Processing " << i << " / " << graph.size() << std::endl;
        }
        addPoint(graph, oracle, i);
    }

    timer.end();
    logger << "Construction time: " << timer.elapsed() << "s" << std::endl;

    return graph;
}


void nsw::NSW::addPoint(Graph &graph,
                              IndexOracle &oracle,
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


std::vector<Neighbor> nsw::NSW::multisearch(const Graph &graph,
                                                  const IndexOracle &oracle,
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
        candidates.emplace(entry_point, dist, false);
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
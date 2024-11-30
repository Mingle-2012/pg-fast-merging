#include "vamana.h"

using namespace merge;


diskann::Vamana::Vamana(float alpha,
                          int L,
                          int R) {
    alpha_ = alpha;
    L_ = L;
    R_ = R;
}


void diskann::Vamana::set_alpha(float alpha) {
    this->alpha_ = alpha;
}


void diskann::Vamana::set_L(int L) {
    this->L_ = L;
}


void diskann::Vamana::set_R(int R) {
    this->R_ = R;
}


Graph diskann::Vamana::build(IndexOracle &oracle) {
    Timer timer;
    timer.start();

    int n = oracle.size();
    Graph graph(n);
    std::mt19937 rng(2024);
    for (int u = 0; u < n; ++u) {
        std::vector<int> init_(R_);
        gen_random(rng, init_.data(), R_, n);
        graph[u].M_ = R_;
        for (auto &v: init_) {
            if (u == v) {
                continue;
            }
            float dist = oracle(u, v);
            graph[u].candidates_.emplace_back(v, dist, false);
        }
        std::sort(graph[u].candidates_.begin(), graph[u].candidates_.end());
    }

    auto *center = new float[oracle.dim()];
    for (unsigned i = 0; i < oracle.size(); ++i) {
        auto pt = oracle[i];
        for (unsigned j = 0; j < oracle.dim(); ++j) {
            center[j] += pt[j];
        }
    }
    for (unsigned i = 0; i < oracle.dim(); ++i) {
        center[i] /= oracle.size();
    }

    unsigned root = 0;
    float minimum = std::numeric_limits<float>::max();
    for (int x = 0; x < oracle.size(); ++x) {
        auto dist = oracle(x, center);
        if (dist < minimum) {
            minimum = dist;
            root = x;
        }
    }
    delete[] center;

    std::vector<int> permutation(n);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::shuffle(permutation.begin(), permutation.end(), rng);
    for (int i = 0; i < n; ++i) {
        if (i % 10000 == 0) {
            logger << "Processing " << i << " / " << graph.size() << std::endl;
        }
        auto res = track_search(oracle, graph, oracle[permutation[i]], root, L_);
        RobustPrune(graph, oracle, 1.0f, permutation[i], res);
        for (auto &j: graph[permutation[i]].candidates_) {
            if (graph[j.id].candidates_.size() + 1 > R_) {
                graph[j.id].candidates_.emplace_back(permutation[i], j.distance, false);
                RobustPrune(graph, oracle, alpha_, j.id, graph[j.id].candidates_);
            } else {
                graph[j.id].addNeighbor(Neighbor(permutation[i], j.distance, false));
            }
        }
    }

    timer.end();
    logger << "Construction time: " << timer.elapsed() << "s" << std::endl;

    return graph;
}


void diskann::Vamana::RobustPrune(Graph &graph,
                                    IndexOracle &oracle,
                                    float alpha,
                                    int point,
                                    std::vector<Neighbor> &candidates) {
    candidates.insert(candidates.begin(), graph[point].candidates_.begin(), graph[point].candidates_.end());
    auto it = std::find(candidates.begin(), candidates.end(), Neighbor(point, 0, false));
    if (it != candidates.end()) {
        candidates.erase(it);
    }
    graph[point].candidates_.clear();
    while (!candidates.empty()) {
        auto min_it = std::min_element(candidates.begin(), candidates.end());
        auto p_star_ = *min_it;
        graph[point].candidates_.push_back(p_star_);
        if (graph[point].candidates_.size() >= R_) {
            break;
        }
        candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                                        [&](const Neighbor &p_prime_) {
                                            return alpha * oracle(p_star_.id, p_prime_.id) <= p_prime_.distance;
                                        }),
                         candidates.end());
    }
}
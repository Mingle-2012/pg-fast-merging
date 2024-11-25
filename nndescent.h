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

using namespace merge;

namespace nndescent {

struct NNDescentParams {
    unsigned K{64};

    float rho{0.5};

    float delta{0.001};

    unsigned iteration{100};

    friend std::ostream &operator<<(std::ostream &os,
                                    const NNDescentParams &params) {
        os << "M0: " << params.K << ", rho: " << params.rho << ", delta: " << params.delta
           << ", iteration: " << params.iteration;
        return os;
    }
};

template<class DATA_TYPE>
class NNDescent {
private:
    unsigned K{64};

    float rho{0.5};

    float delta{0.001};

    unsigned iteration{100};

    void initializeGraph(Graph &graph,
                         IndexOracle<DATA_TYPE> &oracle);

    void generateUpdate(Graph &graph);

    int applyUpdate(unsigned sample,
                    Graph &graph,
                    IndexOracle<DATA_TYPE> &oracle);

    void clearGraph(Graph &graph);

public:
    NNDescent() = default;

    explicit NNDescent(int K, float rho, float delta, int iteration)
            : K(K), rho(rho), delta(delta), iteration(iteration) {}

    ~NNDescent() = default;

    Graph build(IndexOracle<DATA_TYPE> &oracle);
};

template<class DATA_TYPE>
Graph NNDescent<DATA_TYPE>::build(IndexOracle<DATA_TYPE> &oracle) {
    int sample = static_cast<int>(static_cast<float>(K) * rho);
    Graph graph;
    initializeGraph(graph, oracle);
    for (size_t it = 0; it < iteration; ++it) {
        generateUpdate(graph);
        int cnt = applyUpdate(sample, graph, oracle);
        if (cnt <= delta * oracle.size() * K) {
            break;
        }
        clearGraph(graph);
    }
#pragma omp parallel for
    for (auto &u: graph) {
        std::sort(u.candidates_.begin(), u.candidates_.end());
    }
    return graph;
}

template<class DATA_TYPE>
void NNDescent<DATA_TYPE>::initializeGraph(Graph &graph,
                                           IndexOracle<DATA_TYPE> &oracle) {
    int total = oracle.size();
    graph.resize(total);
#pragma omp parallel
    {
        std::mt19937 rng(2024 + omp_get_thread_num());
#pragma omp for
        for (size_t i = 0; i < total; ++i) {
            graph[i].candidates_.reserve(K);
            std::vector<int> indices(K);

            gen_random(rng, indices.data(), K, total);
            for (int j = 0; j < K; ++j) {
                if (indices[j] == i) {
                    continue;
                }
                auto dist = oracle(i, indices[j]);
                graph[i].candidates_.emplace_back(indices[j], dist, true);
            }
            std::make_heap(graph[i].candidates_.begin(), graph[i].candidates_.end());
        }
    };
}

template<class DATA_TYPE>
void NNDescent<DATA_TYPE>::generateUpdate(Graph &graph) {
#pragma omp parallel
    {
        std::mt19937 rng(2024 + omp_get_thread_num());
#pragma omp for
        for (size_t v = 0; v < graph.size(); ++v) {
            auto &_old = graph[v].old_;
            auto &_new = graph[v].new_;
            unsigned size = graph[v].candidates_.size();
            for (size_t neighbor_index = 0; neighbor_index < size; ++neighbor_index) {
                auto &neighbor = graph[v].candidates_[neighbor_index];
                if (neighbor.flag) {
                    _new.emplace_back(neighbor.id);
                    {
                        std::lock_guard<std::mutex> guard(graph[neighbor.id].lock_);
                        graph[neighbor.id].reverse_new_.emplace_back(v);
                    }
                    neighbor.flag = false;
                } else {
                    _old.emplace_back(neighbor.id);
                    {
                        std::lock_guard<std::mutex> guard(graph[neighbor.id].lock_);
                        graph[neighbor.id].reverse_old_.emplace_back(v);
                    }
                }
            }
        }
    }
}

template<class DATA_TYPE>
int NNDescent<DATA_TYPE>::applyUpdate(unsigned sample,
                                      Graph &graph,
                                      IndexOracle<DATA_TYPE> &oracle) {
    int cnt = 0;
#pragma omp parallel
    {
        std::mt19937 rng(2024 + omp_get_thread_num());
#pragma omp for reduction(+:cnt) schedule(dynamic)
        for (auto &v: graph) {
            auto &_old = v.old_;
            auto &_new = v.new_;
            auto &_r_old = v.reverse_old_;
            auto &_r_new = v.reverse_new_;

            shuffle(_r_new.begin(), _r_new.end(), rng);
            if (_r_new.size() > sample) {
                _r_new.resize(sample);
            }
            shuffle(_r_old.begin(), _r_old.end(), rng);
            if (_r_old.size() > sample) {
                _r_old.resize(sample);
            }
            _old.insert(_old.end(), _r_old.begin(), _r_old.end());
            _new.insert(_new.end(), _r_new.begin(), _r_new.end());

            auto _new_size = _new.size();
            auto _old_size = _old.size();
            for (size_t i = 0; i < _new_size; ++i) {
                for (size_t j = i + 1; j < _new_size; ++j) {
                    if (_new[i] == _new[j]) {
                        continue;
                    }
                    auto dist = oracle(_new[i], _new[j]);
                    if (dist < graph[_new[i]].candidates_.front().distance ||
                        dist < graph[_new[j]].candidates_.front().distance) {
                        cnt += graph[_new[i]].insert(_new[j], dist);
                        cnt += graph[_new[j]].insert(_new[i], dist);
                    }
                }
                for (size_t j = 0; j < _old_size; ++j) {
                    if (_new[i] == _old[j]) {
                        continue;
                    }
                    auto dist = oracle(_new[i], _old[j]);
                    if (dist < graph[_new[i]].candidates_.front().distance ||
                        dist < graph[_old[j]].candidates_.front().distance) {
                        cnt += graph[_new[i]].insert(_old[j], dist);
                        cnt += graph[_old[j]].insert(_new[i], dist);
                    }
                }
            }
        }
    };
    return cnt;
}

template<class DATA_TYPE>
void NNDescent<DATA_TYPE>::clearGraph(Graph &graph) {
#pragma omp parallel for
    for (auto &v: graph) {
        v.old_.clear();
        v.new_.clear();
        v.reverse_old_.clear();
        v.reverse_new_.clear();
    }
}
}

#endif //MERGE_NNDESCENT_H

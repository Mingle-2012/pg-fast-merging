#include "include/fgim.h"

using namespace merge;

Merge::Merge(unsigned int M0,
             unsigned int M,
             unsigned int L) {
    if (M0 < 1) {
        throw std::invalid_argument("M0 must be greater than 0");
    }
    if (M < 1) {
        throw std::invalid_argument("M must be greater than 0");
    }
    if (L < 1) {
        throw std::invalid_argument("L must be greater than 0");
    }
    M0_ = M0;
    M_ = M;
    L_ = L;
}

Graph Merge::merge(const Graph &g1,
                              IndexOracle &oracle1,
                              const Graph &g2,
                              IndexOracle &oracle2,
                              IndexOracle &oracle) {
    Timer timer;
    timer.start();

    Graph graph(g1.size() + g2.size());

    Sampling(graph, g1, g2, oracle1, oracle2, oracle);

    Refinement(graph, oracle);

    timer.end();
    logger << "Merging time: " << timer.elapsed() << "s" << std::endl;
    return graph;
}


void Merge::Sampling(Graph &graph,
                     const Graph &g1,
                     const Graph &g2,
                     IndexOracle &oracle1,
                     IndexOracle &oracle2,
                     IndexOracle &oracle) {
    auto total = graph.size();
    auto _gs1 = g1.size();
    auto _gs2 = g2.size();
    std::vector<int> fg1, of1, fg2, of2;
    project(g1, fg1, of1);
    project(g2, fg2, of2);

#pragma omp parallel for schedule(dynamic, 256)
    for (size_t u = 0; u < total; ++u) {
        if (u % 10000 == 0) {
            logger << "Processing " << u << "/" << total << std::endl;
        }
        auto data = oracle[u];
        graph[u].M_ = M0_;
        graph[u].candidates_.reserve(M0_);
        if (u < _gs1) {
            auto &neighbors = g1[u].candidates_;
            std::copy(neighbors.begin(), neighbors.end(), std::back_inserter(graph[u].candidates_));
            if (graph[u].candidates_.size() > M0_) {
                graph[u].candidates_.resize(M0_);
            }
            std::make_heap(graph[u].candidates_.begin(), graph[u].candidates_.end());
            auto result = search(oracle2, fg2,
                                           of2, data, L_, _gs2, L_);
            for (auto &&res: result) {
                graph[u].insert(res.id + _gs1, res.distance);
            }
        } else {
            auto &neighbors = g2[u - _gs1].candidates_;
            for (const auto &neighbor: neighbors) {
                graph[u].candidates_.emplace_back(neighbor.id + _gs1, neighbor.distance, false);
            }
            if (graph[u].candidates_.size() > M0_) {
                graph[u].candidates_.resize(M0_);
            }
            std::make_heap(graph[u].candidates_.begin(), graph[u].candidates_.end());
            auto result = search(oracle1, fg1,
                                           of1, data, L_, _gs1, L_);
            for (auto &&res: result) {
                graph[u].insert(res.id, res.distance);
            }
        }
    }
}

void iterativeUpdate(Graph &graph, IndexOracle &oracle, int M0, int ITER_MAX, int SAMPLES, float THRESHOLD) {
    size_t it = 0;
#pragma omp parallel for
    for (auto &u: graph) {
        for (int v = (int) (u.candidates_.size() / 2); v < u.candidates_.size(); ++v) {
            u.new_.emplace_back(u.candidates_[v].id);
            u.candidates_[v].flag = false;
        }
    }
    while (++it && it <= ITER_MAX) {
        int cnt = 0;
#pragma omp parallel
        {
            std::mt19937 rng(2024 + omp_get_thread_num());
#pragma omp for reduction(+:cnt) schedule(dynamic)
            for (int vv = 0; vv < graph.size(); ++vv) {
                auto &v = graph[vv];
                auto &_old = v.old_;
                auto &_new = v.new_;

                {
                    std::lock_guard<std::mutex> guard(v.lock_);
                    auto &_r_old = v.reverse_old_;
                    auto &_r_new = v.reverse_new_;
                    shuffle(_r_new.begin(), _r_new.end(), rng);
                    if (_r_new.size() > SAMPLES) {
                        _r_new.resize(SAMPLES);
                    }
                    shuffle(_r_old.begin(), _r_old.end(), rng);
                    if (_r_old.size() > SAMPLES) {
                        _r_old.resize(SAMPLES);
                    }
                    if (!_r_old.empty() || !_r_new.empty()) {
                        _old.insert(_old.end(), _r_old.begin(), _r_old.end());
                        _r_old.clear();
                        _new.insert(_new.end(), _r_new.begin(), _r_new.end());
                        _r_new.clear();
                    }
                }

                std::sort(_old.begin(), _old.end());
                std::sort(_new.begin(), _new.end());
                _old.erase(std::unique(_old.begin(), _old.end()), _old.end());
                _new.erase(std::unique(_new.begin(), _new.end()), _new.end());

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
                _old.clear();
                _new.clear();

                for (auto &u: v.candidates_) {
                    if (u.flag) {
                        _new.emplace_back(u.id);
                        {
                            std::lock_guard<std::mutex> guard(graph[u.id].lock_);
                            graph[u.id].reverse_new_.emplace_back(vv);
                        }
                        u.flag = false;
                    } else {
                        _old.emplace_back(u.id);
                        {
                            std::lock_guard<std::mutex> guard(graph[u.id].lock_);
                            graph[u.id].reverse_old_.emplace_back(vv);
                        }
                    }
                }
            }
        }
        unsigned convergence = std::lround(THRESHOLD * static_cast<float>(graph.size()) * static_cast<float>(M0));
        if (cnt <= convergence) {
            break;
        }
    }
}


void prune(Graph &graph, IndexOracle& oracle, int M) {
#pragma omp parallel for schedule(dynamic)
    for (auto &u: graph) {
        auto &neighbors = u.candidates_;
        std::vector<Neighbor> candidates, _new_neighbors;
        {
            std::lock_guard<std::mutex> guard(u.lock_);
            candidates = neighbors;
            neighbors.clear();
        }
        std::sort(candidates.begin(), candidates.end());
        candidates.erase(std::unique(candidates.begin(), candidates.end(),
                                     [](const Neighbor &a,
                                        const Neighbor &b) {
                                         return a.id == b.id;
                                     }),
                         candidates.end());
        for (auto &&v: candidates) {
            bool reserve = true;
            for (auto &nn: _new_neighbors) {
                if (nn.distance <= std::numeric_limits<float>::epsilon()) {
                    continue;
                }
                auto dist = oracle(v.id, nn.id);
                if (dist < v.distance) {
                    {
                        std::lock_guard<std::mutex> guard(graph[nn.id].lock_);
                        graph[nn.id].candidates_.emplace_back(v.id, dist, true);
                    }
                    {
                        std::lock_guard<std::mutex> guard(graph[v.id].lock_);
                        graph[v.id].candidates_.emplace_back(nn.id, dist, true);
                    }
                    reserve = false;
                    break;
                }
            }
            if (reserve) {
                _new_neighbors.emplace_back(v);
            }
        }
        {
            std::lock_guard<std::mutex> guard(u.lock_);
            neighbors.insert(neighbors.end(), _new_neighbors.begin(), _new_neighbors.end());
        }
    }
#pragma omp parallel for
    for (auto &u: graph) {
        auto &candidates = u.candidates_;
        std::sort(candidates.begin(), candidates.end());
        candidates.erase(std::unique(candidates.begin(), candidates.end(),
                                     [](const Neighbor &a,
                                        const Neighbor &b) {
                                         return a.id == b.id;
                                     }),
                         candidates.end());
        if (candidates.size() > M) {
            candidates.resize(M);
        }
    }
}

void addReverseEdge(Graph &graph, int M) {
    Graph reverse_graph(graph.size());
#pragma omp parallel for
    for (int u = 0; u < graph.size(); ++u) {
        for (auto &v: graph[u].candidates_) {
            std::lock_guard<std::mutex> guard(reverse_graph[v.id].lock_);
            reverse_graph[v.id].candidates_.emplace_back(u, v.distance, true);
        }
    }
#pragma omp parallel for
    for (int u = 0; u < graph.size(); ++u) {
        graph[u].candidates_.insert(graph[u].candidates_.end(), reverse_graph[u].candidates_.begin(),
                                    reverse_graph[u].candidates_.end());
        std::sort(graph[u].candidates_.begin(), graph[u].candidates_.end());
        graph[u].candidates_.erase(std::unique(graph[u].candidates_.begin(), graph[u].candidates_.end(),
                                               [](const Neighbor &a,
                                                  const Neighbor &b) {
                                                   return a.id == b.id;
                                               }),
                                   graph[u].candidates_.end());
        if (graph[u].candidates_.size() > M) {
            graph[u].candidates_.resize(M);
        }
    }
}

void connect(Graph &graph){
    std::vector<int> degree(graph.size(), 0);
#pragma omp parallel for
    for (auto &u: graph) {
        for (auto &v: u.candidates_) {
            degree[v.id]++;
        }
    }

#pragma omp parallel for schedule(dynamic)
    for (int x = 0; x < graph.size(); ++x) {
        if (degree[x]) continue;
        auto &u = graph[x];
        auto nearest = u.candidates_.front();
        graph[nearest.id].candidates_.emplace_back(x, nearest.distance, true);
    }
}

void Merge::Refinement(Graph &graph,
                       IndexOracle &oracle) {
    iterativeUpdate(graph, oracle, M0_, ITER_MAX, SAMPLES, THRESHOLD);

    prune(graph, oracle, M_);

    addReverseEdge(graph, M_);

    prune(graph, oracle, M_);

    connect(graph);
}

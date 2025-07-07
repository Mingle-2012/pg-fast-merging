#include "nnmerge.h"

nnmerge::NNMerge::NNMerge(
    DatasetPtr& dataset, int K_, float rho, float delta, int iteration, float alpha)
    : NNDescent(dataset, K_, rho, delta, iteration), alpha(alpha) {
}

void
nnmerge::NNMerge::Combine(const IndexPtr& index1, const IndexPtr& index2) {
    print_info();
    auto& graph1 = index1->extractGraph();
    auto& graph2 = index2->extractGraph();

    graph_size_1 = graph1.size();
    graph_size_2 = graph2.size();
    size_t total = graph_size_1 + graph_size_2;
    Graph G_v(total);

    Timer timer;
    timer.start();

    splitGraph(G_v, graph1, graph2);
    addSamples();
    nndescent();
    mergeGraph(G_v);

    timer.end();
    logger << "Merging time: " << timer.elapsed() << "s" << std::endl;

    flatten_graph_ = FlattenGraph(graph_);
    built_ = true;
}

void
nnmerge::NNMerge::print_info() const {
    NNDescent::print_info();
    logger << "NNMerge with alpha: " << alpha << std::endl;
}

void
nnmerge::NNMerge::splitGraph(Graph& G_v, const Graph& graph1, const Graph& graph2) {
#pragma omp parallel for
    for (size_t u = 0; u < graph_size_1; ++u) {
        auto& neighbors = graph1[u].candidates_;
        unsigned truncation = std::min(neighbors.size(), (size_t)K_) * alpha;
        graph_[u].candidates_.reserve(K_);
        G_v[u].candidates_.reserve(K_ - truncation);
        std::copy(neighbors.begin(),
                  neighbors.begin() + truncation,
                  std::back_inserter(graph_[u].candidates_));
        std::copy(neighbors.begin() + truncation,
                  neighbors.end(),
                  std::back_inserter(G_v[u].candidates_));
    }

#pragma omp parallel for
    for (size_t u = graph_size_1; u < graph_size_1 + graph_size_2; ++u) {
        auto& neighbors = graph2[u - graph_size_1].candidates_;
        unsigned truncation = std::min(neighbors.size(), (size_t)K_) * alpha;
        graph_[u].candidates_.reserve(K_);
        G_v[u].candidates_.reserve(K_ - truncation);
        for (auto it = neighbors.begin(); it != neighbors.begin() + truncation; ++it) {
            graph_[u].candidates_.emplace_back(it->id + graph_size_1, it->distance, true);
        }
        for (auto it = neighbors.begin() + truncation; it != neighbors.end(); ++it) {
            G_v[u].candidates_.emplace_back(it->id + graph_size_1, it->distance, true);
        }
    }
}

void
nnmerge::NNMerge::addSamples() {
    std::mt19937 rng(2024);
    std::uniform_int_distribution<unsigned> _graph_1_samples(0, graph_size_1 - 1);
    std::uniform_int_distribution<unsigned> _graph_2_samples(graph_size_1,
                                                             graph_size_1 + graph_size_2 - 1);
#pragma omp parallel for
    for (size_t u = 0; u < graph_size_1 + graph_size_2; ++u) {
        unsigned current_size = graph_[u].candidates_.size();
        if (u < graph_size_1) {
            for (size_t v = 0; v < K_ - current_size; ++v) {
                unsigned _graph_2_sample = _graph_2_samples(rng);
                auto dist = (*oracle_)(u, _graph_2_sample);
                graph_[u].candidates_.emplace_back(_graph_2_sample, dist, true);
            }
        } else {
            for (size_t v = 0; v < K_ - current_size; ++v) {
                unsigned _graph_1_sample = _graph_1_samples(rng);
                auto dist = (*oracle_)(u, _graph_1_sample);
                graph_[u].candidates_.emplace_back(_graph_1_sample, dist, true);
            }
        }
        std::make_heap(graph_[u].candidates_.begin(), graph_[u].candidates_.end());
    }
}

void
nnmerge::NNMerge::nndescent() {
    size_t it = 0;
    unsigned sample = 100;
    while (++it && it <= iteration_) {
        logger << "Iteration " << it << ":" << std::flush;
        generateUpdate();
        int cnt = applyUpdate(sample);
        logger << cnt << " updates" << std::endl;
        unsigned convergence =
            std::lround(delta_ * static_cast<float>(graph_.size()) * static_cast<float>(K_));
        if (cnt <= convergence) {
            break;
        }
        clearGraph();
    }
}

void
nnmerge::NNMerge::mergeGraph(Graph& G_v) {
    int cnt = 0;
#pragma omp parallel for reduction(+ : cnt)
    for (size_t u = 0; u < graph_size_1 + graph_size_2; ++u) {
        for (auto& v : G_v[u].candidates_) {
            cnt += graph_[u].pushHeap(v.id, v.distance);
        }
    }
    logger << cnt << " updates" << std::endl;
}

void
nnmerge::NNMerge::build_internal() {
    throw std::runtime_error("Not implemented, please use Combine instead");
}

int
nnmerge::NNMerge::applyUpdate(unsigned int sample) {
    int cnt = 0;
#pragma omp parallel
    {
        std::mt19937 rng(2024 + omp_get_thread_num());
#pragma omp for reduction(+ : cnt) schedule(dynamic)
        for (auto& v : graph_) {
            auto& _old = v.old_;
            auto& _new = v.new_;
            auto& _r_old = v.reverse_old_;
            auto& _r_new = v.reverse_new_;

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
                    auto gid1 = _new[i] < graph_size_1 ? 0 : 1;
                    auto gid2 = _new[j] < graph_size_1 ? 0 : 1;
                    if (gid1 == gid2) {
                        continue;
                    }
                    auto dist = (*oracle_)(_new[i], _new[j]);
                    if (dist < graph_[_new[i]].candidates_.front().distance ||
                        dist < graph_[_new[j]].candidates_.front().distance) {
                        cnt += graph_[_new[i]].pushHeap(_new[j], dist);
                        cnt += graph_[_new[j]].pushHeap(_new[i], dist);
                    }
                }
                for (size_t j = 0; j < _old_size; ++j) {
                    if (_new[i] == _old[j]) {
                        continue;
                    }
                    auto gid1 = _new[i] < graph_size_1 ? 0 : 1;
                    auto gid2 = _old[j] < graph_size_1 ? 0 : 1;
                    if (gid1 == gid2) {
                        continue;
                    }
                    auto dist = (*oracle_)(_new[i], _old[j]);
                    if (dist < graph_[_new[i]].candidates_.front().distance ||
                        dist < graph_[_old[j]].candidates_.front().distance) {
                        cnt += graph_[_new[i]].pushHeap(_old[j], dist);
                        cnt += graph_[_old[j]].pushHeap(_new[i], dist);
                    }
                }
            }
        }
    }
    return cnt;
}

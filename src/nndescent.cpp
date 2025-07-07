#include "nndescent.h"

nndescent::NNDescent::NNDescent(DatasetPtr& dataset, int K, float rho, float delta, int iteration)
    : Index(dataset), K_(K), rho_(rho), delta_(delta), iteration_(iteration) {
}

void
nndescent::NNDescent::print_info() const {
    Index::print_info();
    logger << "NNDescent with parameters: " << std::endl;
    logger << "K: " << K_ << std::endl;
    logger << "rho: " << rho_ << std::endl;
    logger << "delta: " << delta_ << std::endl;
    logger << "iteration: " << iteration_ << std::endl;
}

void
nndescent::NNDescent::initializeGraph() {
    int total = oracle_->size();
    graph_.resize(total);
#pragma omp parallel
    {
        std::mt19937 rng(2024 + omp_get_thread_num());
#pragma omp for
        for (size_t i = 0; i < total; ++i) {
            graph_[i].candidates_.reserve(K_);
            std::vector<int> indices(K_);

            gen_random(rng, indices.data(), K_, total);
            for (int j = 0; j < K_; ++j) {
                if (indices[j] == i) {
                    continue;
                }
                auto dist = (*oracle_)(i, indices[j]);
                graph_[i].candidates_.emplace_back(indices[j], dist, true);
            }
            std::make_heap(graph_[i].candidates_.begin(), graph_[i].candidates_.end());
        }
    }
}

void
nndescent::NNDescent::generateUpdate() {
#pragma omp parallel
    {
        std::mt19937 rng(2024 + omp_get_thread_num());
#pragma omp for
        for (size_t v = 0; v < graph_.size(); ++v) {
            auto& _old = graph_[v].old_;
            auto& _new = graph_[v].new_;
            unsigned size = graph_[v].candidates_.size();
            for (size_t neighbor_index = 0; neighbor_index < size; ++neighbor_index) {
                auto& neighbor = graph_[v].candidates_[neighbor_index];
                if (neighbor.flag) {
                    _new.emplace_back(neighbor.id);
                    {
                        std::lock_guard<std::mutex> guard(graph_[neighbor.id].lock_);
                        graph_[neighbor.id].reverse_new_.emplace_back(v);
                    }
                    neighbor.flag = false;
                } else {
                    _old.emplace_back(neighbor.id);
                    {
                        std::lock_guard<std::mutex> guard(graph_[neighbor.id].lock_);
                        graph_[neighbor.id].reverse_old_.emplace_back(v);
                    }
                }
            }
        }
    }
}

int
nndescent::NNDescent::applyUpdate(unsigned sample) {
    int cnt = 0;
#pragma omp parallel
    {
        std::mt19937 rng(2024 + omp_get_thread_num());
#pragma omp for reduction(+ : cnt) schedule(dynamic, 256)
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

void
nndescent::NNDescent::clearGraph() {
#pragma omp parallel for
    for (auto& v : graph_) {
        v.old_.clear();
        v.new_.clear();
        v.reverse_old_.clear();
        v.reverse_new_.clear();
    }
}

void
nndescent::NNDescent::build_internal() {
    int sample = static_cast<int>(static_cast<float>(K_) * rho_);
    initializeGraph();
    for (size_t it = 0; it < iteration_; ++it) {
        generateUpdate();
        int cnt = applyUpdate(sample);
        logger << "Iteration " << it << " update " << cnt << " edges" << std::endl;
        if (cnt <= delta_ * oracle_->size() * K_) {
            break;
        }
        clearGraph();
    }
#pragma omp parallel for
    for (auto& u : graph_) {
        std::sort(u.candidates_.begin(), u.candidates_.end());
    }
}

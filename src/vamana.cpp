#include "vamana.h"

diskann::Vamana::Vamana(DatasetPtr& dataset, float alpha, int L, int R)
    : Index(dataset), alpha_(alpha), L_(L), R_(R) {
}

void
diskann::Vamana::set_alpha(float alpha) {
    this->alpha_ = alpha;
}

void
diskann::Vamana::set_L(int L) {
    this->L_ = L;
}

void
diskann::Vamana::set_R(int R) {
    this->R_ = R;
}

void
diskann::Vamana::RobustPrune(float alpha, int point, Neighbors& candidates) {
    candidates.insert(
        candidates.begin(), graph_[point].candidates_.begin(), graph_[point].candidates_.end());
    auto it = std::find(candidates.begin(), candidates.end(), Neighbor(point, 0, false));
    if (it != candidates.end()) {
        candidates.erase(it);
    }
    graph_[point].candidates_.clear();
    while (!candidates.empty()) {
        auto min_it = std::min_element(candidates.begin(), candidates.end());
        auto p_star_ = *min_it;
        graph_[point].candidates_.push_back(p_star_);
        if (graph_[point].candidates_.size() >= R_) {
            break;
        }
        candidates.erase(std::remove_if(candidates.begin(),
                                        candidates.end(),
                                        [&](const Neighbor& p_prime_) {
                                            return alpha * (*oracle_)(p_star_.id, p_prime_.id) <=
                                                   p_prime_.distance;
                                        }),
                         candidates.end());
    }
}

void
diskann::Vamana::build_internal() {
    int n = oracle_->size();
    std::mt19937 rng(2024);
    for (int u = 0; u < n; ++u) {
        std::vector<int> init_(R_);
        gen_random(rng, init_.data(), R_, n);
        for (auto& v : init_) {
            if (u == v) {
                continue;
            }
            float dist = (*oracle_)(u, v);
            graph_[u].candidates_.emplace_back(v, dist, false);
        }
        std::sort(graph_[u].candidates_.begin(), graph_[u].candidates_.end());
    }

    auto* center = new float[oracle_->dim()];
    for (unsigned i = 0; i < oracle_->size(); ++i) {
        auto pt = (*oracle_)[i];
        for (unsigned j = 0; j < oracle_->dim(); ++j) {
            center[j] += pt[j];
        }
    }
    for (unsigned i = 0; i < oracle_->dim(); ++i) {
        center[i] /= oracle_->size();
    }

    unsigned root = 0;
    float minimum = std::numeric_limits<float>::max();
    for (int x = 0; x < oracle_->size(); ++x) {
        auto dist = (*oracle_)(x, center);
        if (dist < minimum) {
            minimum = dist;
            root = x;
        }
    }
    delete[] center;

    std::vector<int> permutation(n);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::shuffle(permutation.begin(), permutation.end(), rng);

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i) {
        if (i % 10000 == 0) {
            logger << "Processing " << i << " / " << graph_.size() << std::endl;
        }
        auto res = track_search(
            oracle_.get(), visited_list_pool_.get(), graph_, (*oracle_)[permutation[i]], L_, root);
        res.erase(
            std::remove_if(
                res.begin(), res.end(), [&](const Neighbor& n) { return n.id == permutation[i]; }),
            res.end());
        {
            std::lock_guard<std::mutex> guard(graph_[permutation[i]].lock_);
            RobustPrune(1.0f, permutation[i], res);
        }
        for (auto& j : graph_[permutation[i]].candidates_) {
            std::lock_guard<std::mutex> neighbor_guard(graph_[j.id].lock_);
            if (graph_[j.id].candidates_.size() + 1 > R_) {
                //                graph_[j.id].candidates_.emplace_back(permutation[i], j.distance, false);
                Neighbors rev = {Neighbor(permutation[i], j.distance, false)};
                RobustPrune(alpha_, j.id, rev);
            } else {
                graph_[j.id].addNeighbor(Neighbor(permutation[i], j.distance, false));
            }
        }
    }
}

void
diskann::Vamana::partial_build(std::vector<uint32_t>& permutation) {
    int n = permutation.size();
    std::mt19937 rng(2024);
    for (int u = 0; u < n; ++u) {
        for (int i = 0; i < R_; ++i) {
            auto v = permutation[rng() % n];
            if (permutation[u] == v) {
                continue;
            }
            float dist = (*oracle_)(permutation[u], v);
            graph_[permutation[u]].candidates_.emplace_back(v, dist, false);
        }
        std::sort(graph_[permutation[u]].candidates_.begin(),
                  graph_[permutation[u]].candidates_.end());
    }
    logger << "2" << std::endl;

    auto* center = new float[oracle_->dim()];
    for (unsigned i = 0; i < n; ++i) {
        auto pt = (*oracle_)[permutation[i]];
        for (unsigned j = 0; j < oracle_->dim(); ++j) {
            center[j] += pt[j];
        }
    }
    for (unsigned i = 0; i < oracle_->dim(); ++i) {
        center[i] /= n;
    }
    unsigned root = 0;
    float minimum = std::numeric_limits<float>::max();
    for (int x = 0; x < n; ++x) {
        auto dist = (*oracle_)(permutation[x], center);
        if (dist < minimum) {
            minimum = dist;
            root = permutation[x];
        }
    }
    delete[] center;

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i) {
        if (i % 10000 == 0) {
            logger << "Processing " << i << " / " << n << std::endl;
        }
        auto res = track_search(
            oracle_.get(), visited_list_pool_.get(), graph_, (*oracle_)[permutation[i]], L_, root);
        {
            std::lock_guard<std::mutex> guard(graph_[permutation[i]].lock_);
            RobustPrune(1.0f, permutation[i], res);
        }
        for (auto& j : graph_[permutation[i]].candidates_) {
            std::lock_guard<std::mutex> neighbor_guard(graph_[j.id].lock_);
            if (graph_[j.id].candidates_.size() + 1 > R_) {
                //                graph_[j.id].candidates_.emplace_back(permutation[i], j.distance, false);
                Neighbors rev = {Neighbor(permutation[i], j.distance, false)};
                RobustPrune(alpha_, j.id, rev);
            } else {
                graph_[j.id].addNeighbor(Neighbor(permutation[i], j.distance, false));
            }
        }
    }

    built_ = true;
}

void
diskann::Vamana::print_info() const {
    Index::print_info();
    logger << "Vamana Index Info:" << std::endl;
    logger << "Alpha: " << alpha_ << std::endl;
    logger << "L: " << L_ << std::endl;
    logger << "R: " << R_ << std::endl;
}

diskann::DiskANN::DiskANN(DatasetPtr& dataset, float alpha, int L, int R, int k, int ell)
    : Index(dataset), alpha_(alpha), L_(L), R_(R), k_(k), ell_(ell) {
}

void
diskann::DiskANN::build_internal() {
    std::mt19937 rng(2024);
    auto kmeans = std::make_shared<Kmeans>(dataset_, k_);
    kmeans->Run();

    std::vector<IndexPtr> indexes;
    for (int k = 0; k < k_; ++k) {
        logger << "Processing " << k << " / " << k_ << std::endl;
        std::vector<uint32_t> permutation;
        permutation.reserve(oracle_->size() * ell_ / k_);
        for (int i = 0; i < oracle_->size(); ++i) {
            auto centers = kmeans->NearestCenter(i, ell_);
            if (std::find(centers.begin(), centers.end(), k) != centers.end()) {
                permutation.push_back(i);
            }
        }
        std::shuffle(permutation.begin(), permutation.end(), rng);
        auto vamana = std::make_shared<Vamana>(dataset_, alpha_, L_, R_);
        vamana->partial_build(permutation);
        indexes.emplace_back(vamana);
    }

    for (auto& index : indexes) {
        auto& graph = index->extractGraph();
        for (int i = 0; i < oracle_->size(); ++i) {
            graph_[i].candidates_.insert(graph_[i].candidates_.end(),
                                         graph[i].candidates_.begin(),
                                         graph[i].candidates_.end());
        }
    }

    for (int i = 0; i < oracle_->size(); ++i) {
        std::sort(graph_[i].candidates_.begin(), graph_[i].candidates_.end());
    }
}

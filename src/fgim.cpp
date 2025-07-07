#include "include/fgim.h"

using namespace graph;

FGIM::FGIM() : max_degree_(20), max_base_degree_(40), sample_rate_(0.3) {
}

FGIM::FGIM(unsigned int max_degree, float sample_rate)
    : max_degree_(max_degree), max_base_degree_(max_degree * 2), sample_rate_(sample_rate) {
}

FGIM::FGIM(DatasetPtr& dataset, unsigned int max_degree, float sample_rate, bool allocate)
    : Index(dataset, allocate),
      max_degree_(max_degree),
      max_base_degree_(max_degree * 2),
      sample_rate_(sample_rate) {
}

// TODO In the future, we will support loading from a file for each index type.
void
FGIM::load_latest(Graph& graph, const std::filesystem::path& directoryPath) {
    std::filesystem::path latestFilePath;
    int maxIteration = -1;

    if (!std::filesystem::exists(directoryPath) || !std::filesystem::is_directory(directoryPath)) {
        std::filesystem::create_directories(directoryPath);
        return;
    }

    std::string fileBasePrefix = "fgim_";
    std::string fileMiddlePart =
        dataset_->getName() + "_" + serial_ + "_k_" + std::to_string(max_degree_);
    std::string iterationSuffixPrefix = "_iter_";
    std::string fileExtension = ".bin";

    for (const auto& entry : std::filesystem::directory_iterator(directoryPath)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            if (filename.rfind(fileBasePrefix, 0) == 0 &&
                filename.find(fileMiddlePart) != std::string::npos &&
                filename.rfind(iterationSuffixPrefix) != std::string::npos &&
                filename.rfind(fileExtension) == filename.length() - fileExtension.length()) {
                size_t iterPrefixPos = filename.rfind(iterationSuffixPrefix);
                if (iterPrefixPos == std::string::npos)
                    continue;
                size_t iterStart = iterPrefixPos + iterationSuffixPrefix.length();
                size_t iterEnd = filename.rfind(fileExtension);
                if (iterEnd != std::string::npos && iterStart < iterEnd) {
                    std::string iterStr = filename.substr(iterStart, iterEnd - iterStart);
                    int currentIteration = std::stoi(iterStr);
                    if (currentIteration > maxIteration) {
                        maxIteration = currentIteration;
                        latestFilePath = entry.path();
                    }
                }
            }
        }
    }

    if (maxIteration != -1) {
        loadGraph(graph, latestFilePath, dataset_->getOracle());
        start_iter_ = maxIteration + 1;
    }
}

void
FGIM::update_neighbors(Graph& graph) {
    // FIXME Segmentation Fault???
    size_t it = start_iter_;
    unsigned samples = sample_rate_ * max_base_degree_;
    while (++it && it <= ITER_MAX) {
        int cnt = 0;
#pragma omp parallel
        {
            std::mt19937 rng(2024 + omp_get_thread_num());
            std::vector<int> _old, _new;
            _old.reserve(max_base_degree_ * 2);
            _new.reserve(max_base_degree_ * 2);
#pragma omp for reduction(+ : cnt) schedule(dynamic, 256)
            for (int vv = 0; vv < graph.size(); ++vv) {
                auto& v = graph[vv];
                _old.clear();
                _new.clear();
                for (auto& u : v.candidates_) {
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

                {
                    std::lock_guard<std::mutex> guard(v.lock_);
                    if (!v.reverse_old_.empty()) {
                        _old.insert(_old.end(),
                                    std::make_move_iterator(v.reverse_old_.begin()),
                                    std::make_move_iterator(v.reverse_old_.end()));
                        v.reverse_old_.clear();
                    }
                    if (!v.reverse_new_.empty()) {
                        _new.insert(_new.end(),
                                    std::make_move_iterator(v.reverse_new_.begin()),
                                    std::make_move_iterator(v.reverse_new_.end()));
                        v.reverse_new_.clear();
                    }
                }

                std::shuffle(_old.begin(), _old.end(), rng);
                if (_old.size() > samples) {
                    _old.resize(samples);
                }
                std::shuffle(_new.begin(), _new.end(), rng);
                if (_new.size() > samples) {
                    _new.resize(samples);
                }

                auto _new_size = _new.size();
                auto _old_size = _old.size();
                for (size_t i = 0; i < _new_size; ++i) {
                    for (size_t j = i + 1; j < _new_size; ++j) {
                        if (_new[i] == _new[j]) {
                            continue;
                        }
                        if (are_in_same_index(_new[i], _new[j])) {
                            continue;
                        }
                        auto dist = (*oracle_)(_new[i], _new[j]);
                        if (dist < graph[_new[i]].candidates_.front().distance ||
                            dist < graph[_new[j]].candidates_.front().distance) {
                            cnt += graph[_new[i]].pushHeap(_new[j], dist);
                            cnt += graph[_new[j]].pushHeap(_new[i], dist);
                        }
                    }
                    for (size_t j = 0; j < _old_size; ++j) {
                        if (_new[i] == _old[j]) {
                            continue;
                        }
                        if (are_in_same_index(_new[i], _old[j])) {
                            continue;
                        }
                        auto dist = (*oracle_)(_new[i], _old[j]);
                        if (dist < graph[_new[i]].candidates_.front().distance ||
                            dist < graph[_old[j]].candidates_.front().distance) {
                            cnt += graph[_new[i]].pushHeap(_old[j], dist);
                            cnt += graph[_old[j]].pushHeap(_new[i], dist);
                        }
                    }
                }
            }
        }

        logger << "Iteration " << it << " with " << cnt << " new edges" << std::endl;
        unsigned convergence = std::lround(THRESHOLD * static_cast<float>(graph.size()) *
                                           static_cast<float>(max_base_degree_));
        //        if (it % 5 == 0) {
        //            saveGraph(graph,
        //                      "fgim_" + dataset_->getName() + "_" + serial_ + "_k_" + std::to_string(max_degree_) +
        //                          "_iter_" + std::to_string(it) + ".bin");
        //        }
        //        connect_no_indegree(graph);
        if (cnt <= convergence) {
            break;
        }
    }
}

// No streamlined
//void
//FGIM::update_neighbors(Graph &graph) {
//    size_t it = 0;
//    unsigned samples = SAMPLES;
//
//    while (++it && it <= ITER_MAX) {
//        int cnt = 0;
//        long long dist_calc = 0;
//#pragma omp parallel
//        {
//            std::mt19937 rng(2024 + omp_get_thread_num());
//#pragma omp for
//            for (int vv = 0; vv < graph.size(); ++vv) {
//                auto &v = graph[vv];
//                for (auto &u: v.candidates_) {
//                    if (u.flag) {
//                        v.new_.emplace_back(u.id);
//                        {
//                            std::lock_guard<std::mutex> guard(graph[u.id].lock_);
//                            graph[u.id].reverse_new_.emplace_back(vv);
//                        }
//                        u.flag = false;
//                    } else {
//                        v.old_.emplace_back(u.id);
//                        {
//                            std::lock_guard<std::mutex> guard(graph[u.id].lock_);
//                            graph[u.id].reverse_old_.emplace_back(vv);
//                        }
//                    }
//                }
//            }
//
//#pragma omp for reduction(+ : cnt, dist_calc) schedule(dynamic)
//            for (int vv = 0; vv < graph.size(); ++vv) {
//                auto &v = graph[vv];
//                auto &_old = v.old_;
//                auto &_new = v.new_;
//
//                {
//                    std::lock_guard<std::mutex> guard(v.lock_);
//                    auto &_r_old = v.reverse_old_;
//                    auto &_r_new = v.reverse_new_;
//                    if (!_r_old.empty()) {
//                        _old.insert(_old.end(), _r_old.begin(), _r_old.end());
//                        _r_old.clear();
//                    }
//                    if (!_r_new.empty()) {
//                        _new.insert(_new.end(), _r_new.begin(), _r_new.end());
//                        _r_new.clear();
//                    }
//                }
//
//                std::shuffle(_old.begin(), _old.end(), rng);
//                if (_old.size() > samples) {
//                    _old.resize(samples);
//                }
//                std::shuffle(_new.begin(), _new.end(), rng);
//                if (_new.size() > samples) {
//                    _new.resize(samples);
//                }
//
//                auto _new_size = _new.size();
//                auto _old_size = _old.size();
//                for (size_t i = 0; i < _new_size; ++i) {
//                    for (size_t j = i + 1; j < _new_size; ++j) {
//                        if (_new[i] == _new[j]) {
//                            continue;
//                        }
//                        auto dist = (*oracle_)(_new[i], _new[j]);
//                        ++dist_calc;
//                        if (dist < graph[_new[i]].candidates_.front().distance ||
//                            dist < graph[_new[j]].candidates_.front().distance) {
//                            cnt += graph[_new[i]].pushHeap(_new[j], dist);
//                            cnt += graph[_new[j]].pushHeap(_new[i], dist);
//                        }
//                    }
//                    for (size_t j = 0; j < _old_size; ++j) {
//                        if (_new[i] == _old[j]) {
//                            continue;
//                        }
//                        auto dist = (*oracle_)(_new[i], _old[j]);
//                        ++dist_calc;
//                        if (dist < graph[_new[i]].candidates_.front().distance ||
//                            dist < graph[_old[j]].candidates_.front().distance) {
//                            cnt += graph[_new[i]].pushHeap(_old[j], dist);
//                            cnt += graph[_old[j]].pushHeap(_new[i], dist);
//                        }
//                    }
//                }
//                _old.clear();
//                _new.clear();
//            }
//        }
//
//        logger << std::fixed;
//        logger << "Iteration " << it << " with " << cnt << " new edges" << std::endl;
//        logger << "Dist calc: " << dist_calc << std::endl;
//        double sum_dist = 0;
//        for (auto &u: graph) {
//            for (auto &v: u.candidates_) {
//                sum_dist += v.distance;
//            }
//        }
//        logger << "Sum distance: " << sum_dist << std::endl;
//        logger << "Average distance: " << sum_dist / (graph.size() * max_base_degree_) << std::endl;
//
//        unsigned convergence = std::lround(THRESHOLD * static_cast<float>(graph.size()) *
//                                           static_cast<float>(max_base_degree_));
//        if (cnt <= convergence) {
//            break;
//        }
//    }
//}

void
FGIM::prune(Graph& graph, bool add) {
    std::vector<int> indegree(graph.size(), 0);
    for (auto& u : graph) {
        for (auto& v : u.candidates_) {
            indegree[v.id]++;
        }
    }

#pragma omp parallel for schedule(dynamic)
    for (auto& u : graph) {
        auto& neighbors = u.candidates_;
        Neighbors candidates, _new_neighbors;
        {
            std::lock_guard<std::mutex> guard(u.lock_);
            candidates = neighbors;
        }
        candidates.erase(
            std::unique(candidates.begin(),
                        candidates.end(),
                        [](const Neighbor& a, const Neighbor& b) { return a.id == b.id; }),
            candidates.end());
        for (auto&& v : candidates) {
            bool reserve = true;
            int inedge = 0;
            {
                std::lock_guard<std::mutex> guard(graph[v.id].lock_);
                inedge = indegree[v.id];
            }
            if (inedge > 1) {
                for (auto& nn : _new_neighbors) {
                    auto dist = (*oracle_)(v.id, nn.id);
                    if (dist < v.distance) {
                        {
                            std::lock_guard<std::mutex> lock(graph[v.id].lock_);
                            indegree[v.id]--;
                        }
                        if (add) {
                            {
                                std::lock_guard<std::mutex> guard(graph[nn.id].lock_);
                                graph[nn.id].candidates_.emplace_back(v.id, dist, true);
                            }
                            {
                                std::lock_guard<std::mutex> guard(graph[v.id].lock_);
                                graph[v.id].candidates_.emplace_back(nn.id, dist, true);
                            }
                        }
                        reserve = false;
                        break;
                    }
                }
            }
            if (reserve) {
                _new_neighbors.emplace_back(v);
            }
            if (_new_neighbors.size() >= max_base_degree_) {
                break;
            }
        }
        {
            std::lock_guard<std::mutex> guard(u.lock_);
            neighbors.swap(_new_neighbors);
        }
    }
    //#pragma omp parallel for
    //    for (auto &u: graph) {
    //        auto &candidates = u.candidates_;
    //        std::sort(candidates.begin(), candidates.end());
    //        candidates.erase(
    //                std::unique(candidates.begin(),
    //                            candidates.end(),
    //                            [](const Neighbor &a,
    //                               const Neighbor &b) { return a.id == b.id; }),
    //                candidates.end());
    //        if (candidates.size() > max_base_degree_) {
    //            candidates.resize(max_base_degree_);
    //        }
    //    }
}

void
FGIM::add_reverse_edge(Graph& graph) {
    Graph reverse_graph(graph.size());
#pragma omp parallel for
    for (int u = 0; u < graph.size(); ++u) {
        for (auto& v : graph[u].candidates_) {
            std::lock_guard<std::mutex> guard(reverse_graph[v.id].lock_);
            reverse_graph[v.id].candidates_.emplace_back(u, v.distance, true);
        }
    }
#pragma omp parallel for
    for (int u = 0; u < graph.size(); ++u) {
        graph[u].candidates_.insert(graph[u].candidates_.end(),
                                    reverse_graph[u].candidates_.begin(),
                                    reverse_graph[u].candidates_.end());
        std::sort(graph[u].candidates_.begin(), graph[u].candidates_.end());
        graph[u].candidates_.erase(
            std::unique(graph[u].candidates_.begin(), graph[u].candidates_.end()),
            graph[u].candidates_.end());
        if (graph[u].candidates_.size() > max_base_degree_) {
            graph[u].candidates_.resize(max_base_degree_);
        }
    }
}

void
FGIM::connect_no_indegree(Graph& graph) {
    int total = oracle_->size();
    std::vector<int> indegree(total, 0);

#pragma omp parallel for
    for (auto& u : graph) {
        std::sort(u.candidates_.begin(), u.candidates_.end());
        for (auto& v : u.candidates_) {
            indegree[v.id]++;
        }
    }

    int cnt = 0;

    std::vector<int> replace_pos(total, std::min((int)graph.size(), (int)max_base_degree_) - 1);
    for (int u = 0; u < total; ++u) {
        auto& neighbors = graph[u].candidates_;
        int need_replace_loc = neighbors.size() - 1;
        while (indegree[u] < 1 && need_replace_loc >= 0) {
            int need_replace_id = neighbors[need_replace_loc].id;
            bool has_connect = false;
            for (auto& neighbor : graph[need_replace_id].candidates_) {
                if (neighbor.id == u) {
                    has_connect = true;
                    break;
                }
            }
            if (replace_pos[need_replace_id] > 0 && !has_connect) {
                auto& replace_node =
                    graph[need_replace_id].candidates_[replace_pos[need_replace_id]];
                auto replace_id = replace_node.id;
                if (replace_id >= total || replace_id < 0) {
                    replace_pos[need_replace_id]--;
                    continue;
                }
                if (indegree[replace_id] > 1) {
                    indegree[replace_id]--;
                    replace_node.id = u;
                    replace_node.distance = neighbors[need_replace_loc].distance;
                    indegree[u]++;
                    cnt++;
                }
                replace_pos[need_replace_id]--;
            }
            need_replace_loc--;
        }
    }

    logger << "Connect " << cnt << " nodes" << std::endl;
}

bool
FGIM::are_in_same_index(size_t id1, size_t id2) const {
    if (offsets_.empty() || id1 >= offsets_.back() || id2 >= offsets_.back()) {
        logger << "Error: Vector index out of bounds or empty offsets." << std::endl;
        return false;
    }

    size_t diff = (id1 > id2) ? (id1 - id2) : (id2 - id1);
    if (diff >= max_index_size_) {
        return false;
    }

    auto it1 = std::upper_bound(offsets_.begin(), offsets_.end(), id1);
    size_t bucket_id = std::distance(offsets_.begin(), it1);

    size_t start = (bucket_id == 0) ? 0 : offsets_[bucket_id - 1];
    size_t end = offsets_[bucket_id];
    return id2 >= start && id2 < end;
}

void
FGIM::CrossQuery(std::vector<IndexPtr>& indexes) {
    Timer timer;
    timer.start();

    std::vector<std::reference_wrapper<Graph> > graphs;
    std::vector<std::reference_wrapper<HGraph> > hgraphs;
    bool isHGraph = true;
    for (auto& index : indexes) {
        auto hnsw_index = std::dynamic_pointer_cast<hnsw::HNSW>(index);
        if (hnsw_index == nullptr) {
            isHGraph = false;
            graphs.emplace_back(index->extractGraph());
        } else {
            hgraphs.emplace_back(hnsw_index->extractHGraph());
        }
    }

    size_t offset = 0;
    offsets_.clear();
    if (isHGraph) {
        for (auto& g : hgraphs) {
            auto& graph_ref = g.get();
            auto graph_size = graph_ref[0].size();
#pragma omp parallel for schedule(dynamic)
            for (size_t i = 0; i < graph_size; ++i) {
                auto& neighbors = graph_ref[0][i].candidates_;
                for (size_t j = 0; j < neighbors.size() && j < max_base_degree_; ++j) {
                    auto& neighbor = neighbors[j];
                    graph_[i + offset].candidates_.emplace_back(
                        neighbor.id + offset, neighbor.distance, true);
                }
                std::make_heap(graph_[i + offset].candidates_.begin(),
                               graph_[i + offset].candidates_.end());
            }
            offset += graph_size;
            max_index_size_ = std::max(max_index_size_, graph_size);
            offsets_.emplace_back(offset);
        }
    } else {
        for (auto& g : graphs) {
            auto& graph_ref = g.get();
            auto graph_size = graph_ref.size();
#pragma omp parallel for schedule(dynamic)
            for (size_t i = 0; i < graph_size; ++i) {
                auto& neighbors = graph_ref[i].candidates_;
                for (size_t j = 0; j < neighbors.size() && j < max_base_degree_; ++j) {
                    auto& neighbor = neighbors[j];
                    graph_[i + offset].candidates_.emplace_back(
                        neighbor.id + offset, neighbor.distance, true);
                }
                std::make_heap(graph_[i + offset].candidates_.begin(),
                               graph_[i + offset].candidates_.end());
            }
            offset += graph_size;
            max_index_size_ = std::max(max_index_size_, graph_size);
            offsets_.emplace_back(offset);
        }
    }

    unsigned L = max_base_degree_ / (indexes.size() - 1);
#pragma omp parallel for schedule(dynamic)
    for (int u = 0; u < oracle_->size(); ++u) {
        auto cur_graph_idx =
            std::upper_bound(offsets_.begin(), offsets_.end(), u) - offsets_.begin();
        auto data = (*oracle_)[u];

        for (size_t graph_idx = 0; graph_idx < indexes.size(); graph_idx++) {
            if (graph_idx == cur_graph_idx) {
                continue;
            }
            auto _offset = graph_idx == 0 ? 0 : offsets_[graph_idx - 1];
            auto& index = indexes[graph_idx];
            auto result = index->search(data, L, L);
            for (auto&& res : result) {
                graph_[u].pushHeap(res.id + _offset, res.distance);
            }
        }
    }

    timer.end();
    logger << "Cross query time: " << timer.elapsed() << "s" << std::endl;
}

void
FGIM::Refinement() {
    Timer timer;

    timer.start();
    update_neighbors(graph_);
    timer.end();
    logger << "Iterative update time: " << timer.elapsed() << "s" << std::endl;

    timer.start();
    prune(graph_);
    timer.end();
    logger << "Pruning time: " << timer.elapsed() << "s" << std::endl;

    timer.start();
    add_reverse_edge(graph_);
    timer.end();
    logger << "Adding reverse edge time: " << timer.elapsed() << "s" << std::endl;

    timer.start();
    connect_no_indegree(graph_);
    timer.end();
    logger << "Connecting no indegree time: " << timer.elapsed() << "s" << std::endl;
}

void
FGIM::Combine(std::vector<IndexPtr>& indexes) {
    if (dataset_ == nullptr) {
        std::vector<DatasetPtr> datasets;
        for (auto& index : indexes) {
            datasets.emplace_back(index->extractDataset());
        }
        dataset_ = Dataset::aggregate(datasets);
        oracle_ = dataset_->getOracle();
        visited_list_pool_ = dataset_->getVisitedListPool();
        base_ = dataset_->getBasePtr();
        Graph(oracle_->size()).swap(graph_);
    }
    print_info();

    for (auto& u : graph_) {
        u.candidates_.reserve(max_base_degree_);
    }

    load_latest(graph_);

    Timer timer;
    timer.start();

    if (start_iter_ == 0) {
        logger << "Start merging index from scratch." << std::endl;
        CrossQuery(indexes);
    } else {
        logger << "Start merging index from iteration " << start_iter_ << std::endl;
    }

    Refinement();

    timer.end();
    logger << "Merging time: " << timer.elapsed() << "s" << std::endl;

    flatten_graph_ = FlattenGraph(graph_);
    built_ = true;
}

void
FGIM::set_serial(const std::string& serial) {
    serial_ = serial;
}

const std::string&
FGIM::get_serial() const {
    return serial_;
}
void
FGIM::print_info() const {
    Index::print_info();
    logger << "Max degree: " << max_degree_ << std::endl;
    logger << "Max base degree: " << max_base_degree_ << std::endl;
    logger << "Sample rate: " << sample_rate_ << std::endl;
}

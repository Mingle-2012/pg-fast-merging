#include "hnsw.h"

using namespace graph;

hnsw::HNSW::HNSW(DatasetPtr& dataset, int max_neighbors, int ef_construction)
    : max_neighbors_(max_neighbors),
      max_base_neighbors_(max_neighbors * 2),
      ef_construction_(ef_construction),
      Index(dataset, false) {
    visited_table_ = std::unordered_set<int>();
    random_engine_.seed(2024);
    reverse_ = 1 / log(1.0 * max_neighbors_);

    levels_.reserve(oracle_->size());
    levels_.resize(oracle_->size(), 0);

    logger << "Building HNSW index with parameters:" << std::endl;
    logger << "max_neighbors: " << max_neighbors_ << std::endl;
    logger << "ef_construction: " << ef_construction_ << std::endl;
    logger << "dataset size: " << oracle_->size() << std::endl;

    int total = oracle_->size();

    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    for (int i = 0; i < total; i++) {
        levels_[i] = (int)(-log(distribution(random_engine_)) * reverse_);
        max_level_ = std::max(max_level_, levels_[i]);
    }

    graph_.reserve(max_level_ + 1);
    for (int i = 0; i <= max_level_; ++i) {
        graph_.emplace_back(total);
    }
}

hnsw::HNSW::HNSW(
    DatasetPtr& dataset, HGraph& graph, bool partial, int max_neighbors, int ef_construction)
    : Index(dataset, false),
      graph_(std::move(graph)),
      max_neighbors_(max_neighbors),
      max_base_neighbors_(max_neighbors * 2),
      ef_construction_(ef_construction) {
    random_engine_.seed(2024);
    reverse_ = 1 / log(1.0 * max_neighbors_);

    for (int i = graph.size() - 1; i >= 0; --i) {
        bool found = false;
        for (int j = 0; j < graph[i].size(); ++j) {
            if (!graph[i][j].candidates_.empty()) {
                enter_point_ = j;
                found = true;
                break;
            }
        }
        if (found) {
            cur_max_level_ = max_level_ = i;
            break;
        }
    }

    if (!partial) {
        flatten_graph_ = FlattenHGraph(graph_);
        built_ = true;
    } else {
        levels_.reserve(oracle_->size());
        levels_.resize(oracle_->size(), 0);
        cur_size_ = graph_[0].size();

        int total = oracle_->size();

        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        for (int i = cur_size_; i < total; i++) {
            levels_[i] = (int)(-log(distribution(random_engine_)) * reverse_);
            max_level_ = std::max(max_level_, levels_[i]);
        }

        graph_.reserve(max_level_ + 1);
        graph_.resize(max_level_ + 1);
        for (int i = 0; i <= max_level_; ++i) {
            graph_[i].resize(total);
        }
    }
}

int
hnsw::HNSW::seekPos(const Neighbors& vec) {
    int left = 0, right = vec.size() - 1;
    if (vec[right].id > 0) {
        return right;
    }
    int result = right;
    while (left <= right) {
        int mid = (left + right) / 2;
        if (vec[mid].id == -1) {
            result = mid;
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    return result;
}

void
hnsw::HNSW::addPoint(unsigned int index) {
    std::lock_guard<std::mutex> guard(graph_[0][index].lock_);

    int level = levels_[index];
    std::unique_lock<std::mutex> graph_lock(graph_lock_);
    int max_level_copy = cur_max_level_;
    if (level <= max_level_copy) {
        graph_lock.unlock();
    }

    uint32_t cur_node_ = enter_point_;
    for (auto i = max_level_copy; i > level; --i) {
        auto res = search_layer(
            oracle_.get(), visited_list_pool_.get(), graph_, i, (*oracle_)[index], 1, 1, cur_node_);
        cur_node_ = res[0].id;
    }

    for (auto i = std::min(level, max_level_copy); i >= 0; --i) {
        auto res = search_layer(oracle_.get(),
                                visited_list_pool_.get(),
                                graph_,
                                i,
                                (*oracle_)[index],
                                ef_construction_,
                                ef_construction_,
                                cur_node_);

        res.erase(std::remove_if(
                      res.begin(), res.end(), [index](const Neighbor& n) { return n.id == index; }),
                  res.end());
        res.erase(std::unique(res.begin(), res.end()), res.end());

        auto cur_max_cnt = level ? max_neighbors_ : max_base_neighbors_;
        prune(res, cur_max_cnt);

        auto& graph = graph_[i];
        auto& candidates = graph[index].candidates_;
        candidates.swap(res);
        for (auto& e : candidates) {
            std::lock_guard<std::mutex> lock(graph_[0][e.id].lock_);
            graph[e.id].addNeighbor(Neighbor(index, e.distance, false));
            prune(graph[e.id].candidates_, cur_max_cnt);
        }
        cur_node_ = candidates[0].id;
    }

    if (level > max_level_copy) {
        enter_point_ = index;
        cur_max_level_ = level;
    }
}

struct CompareByCloser {
    bool
    operator()(const Node& a, const Node& b) {
        return a.distance > b.distance;
    }
};

Neighbors
hnsw::HNSW::searchLayer(
    const Graph& graph, const float* query, size_t topk, size_t L, size_t entry_id) const {
    auto graph_sz = graph.size();
    std::vector<bool> visited(graph_sz, false);
    Neighbors retset(L + 1, Neighbor(-1, std::numeric_limits<float>::max(), false));
    auto dist = (*oracle_)(entry_id, query);
    retset[0] = Neighbor(entry_id, dist, true);
    int k = 0;
    while (k < L) {
        int nk = L;
        if (retset[k].flag) {
            retset[k].flag = false;
            auto n = retset[k].id;
            for (const auto& candidate : graph[n].candidates_) {
                auto id = candidate.id;
                if (visited[id])
                    continue;
                visited[id] = true;
                dist = (*oracle_)(id, query);
                if (dist >= retset[L - 1].distance)
                    continue;
                Neighbor nn(id, dist, true);
                int r = insert_into_pool(retset.data(), L, nn);
                if (r < nk)
                    nk = r;
            }
        }
        if (nk <= k)
            k = nk;
        else
            ++k;
    }
    retset.resize(topk);
    return retset;
}

Neighbors
hnsw::HNSW::searchLayer(
    Graph& graph, IndexOracle<float>& oracle, float* query, int enter_point, int ef) {
    visited_table_.clear();
    visited_table_.insert(enter_point);
    std::priority_queue<Node, std::vector<Node>, CompareByCloser> candidates;
    std::priority_queue<Node> result;
    auto dist = oracle(enter_point, query);
    candidates.emplace(enter_point, dist);
    result.emplace(enter_point, dist);

    auto farthest = result.top().distance;

    while (!candidates.empty()) {
        auto c = candidates.top();
        if (c.distance > farthest && result.size() == ef) {
            break;
        }
        candidates.pop();
        for (auto& n : graph[c.id].candidates_) {
            if (visited_table_.find(n.id) == visited_table_.end()) {
                visited_table_.insert(n.id);
                auto d = oracle(n.id, query);
                if (result.size() < ef || d < farthest) {
                    candidates.emplace(n.id, d);
                    result.emplace(n.id, d);
                    if (result.size() > ef) {
                        result.pop();
                    }
                    if (!result.empty()) {
                        farthest = result.top().distance;
                    }
                }
            }
        }
    }
    Neighbors ret;
    while (!result.empty()) {
        auto r = result.top();
        ret.emplace_back(r.id, r.distance, false);
        result.pop();
    }
    std::reverse(ret.begin(), ret.end());
    return ret;
}

void
hnsw::HNSW::prune(Neighbors& candidates, int max_neighbors) {
    if (candidates.size() <= max_neighbors) {
        return;
    }
    Neighbors ret_set;
    for (auto& v : candidates) {
        bool prune = false;
        for (auto& w : ret_set) {
            if ((*oracle_)(v.id, w.id) < v.distance) {
                prune = true;
                break;
            }
        }
        if (!prune) {
            ret_set.emplace_back(v);
        }
        if (ret_set.size() >= max_neighbors) {
            break;
        }
    }
    candidates.swap(ret_set);
}

Neighbors
hnsw::HNSW::search(const float* query, unsigned int topk, unsigned int L) const {
    unsigned cur_node_ = enter_point_;
    for (int i = flatten_graph_.size() - 1; i > 0; --i) {
        auto res = graph::search(
            oracle_.get(), visited_list_pool_.get(), flatten_graph_[i], query, 1, 1, cur_node_);
        cur_node_ = res[0].id;
    }
    auto res = graph::search(
        oracle_.get(), visited_list_pool_.get(), flatten_graph_[0], query, topk, L, cur_node_);
    return res;
}

void
hnsw::HNSW::print_info() const {
    Index::print_info();
    logger << "HNSW Index Info:" << std::endl;
    logger << "Max Neighbors: " << max_neighbors_ << std::endl;
    logger << "Max Base Neighbors: " << max_base_neighbors_ << std::endl;
    logger << "EF Construction: " << ef_construction_ << std::endl;
    logger << "Max Level: " << max_level_ << std::endl;

    if (max_neighbors_ == 0 || max_base_neighbors_ == 0 || ef_construction_ == 0) {
        logger << "Warning: max_neighbors, max_base_neighbors, and ef_construction should be "
                  "greater than 0."
               << std::endl;
    }
}

Graph&
hnsw::HNSW::extractGraph() {
    throw std::runtime_error(
        "HNSW does not support extractGraph, please use extractHGraph instead");
}

HGraph&
hnsw::HNSW::extractHGraph() {
    if (!built_) {
        throw std::runtime_error("Index is not built yet");
    }
    return graph_;
}

void
hnsw::HNSW::build_internal() {
    int total = (int)oracle_->size();
#pragma omp parallel for schedule(dynamic)
    for (int i = 1; i < total; ++i) {
        if (i % 100000 == 0) {
            logger << "Adding " << i << " / " << total << std::endl;
        }
        addPoint(i);
    }
}

void
hnsw::HNSW::partial_build(uint64_t start, uint64_t end) {
    if (end > oracle_->size() || start >= end) {
        throw std::invalid_argument("Invalid range for partial build");
    }
    print_info();
    logger << "Adding from " << start << " to " << end << std::endl;
    Timer timer;
    timer.start();
#pragma omp parallel for schedule(dynamic)
    for (auto i = start; i < end; ++i) {
        if (i % 10000 == 0) {
            logger << "Adding " << i << " / " << end << std::endl;
        }
        addPoint(i);
    }
    timer.end();
    logger << "Indexing time: " << timer.elapsed() << "s" << std::endl;
    cur_size_ += end - start;
    if (cur_size_ == oracle_->size()) {
        flatten_graph_ = FlattenHGraph(graph_);
        built_ = true;
    } else {
        built_ = false;
    }
}

void
hnsw::HNSW::partial_build(uint64_t num) {
    print_info();
    auto start = cur_size_, end = cur_size_ == 1 ? num : cur_size_ + num;
    if (num == 0) {
        end = oracle_->size();
        num = end - start;
        if (num == 0) {
            logger << "No points to add, skipping build." << std::endl;
            return;
        }
    }

    if (end > oracle_->size()) {
        throw std::invalid_argument("Invalid range for partial build");
    }
    logger << "Adding from " << start << " to " << end << std::endl;
    Timer timer;
    timer.start();
    {
#pragma omp parallel for schedule(dynamic)
        for (auto i = start; i < end; ++i) {
            if (i % 100000 == 0) {
                logger << "Adding " << i << " / " << end << std::endl;
            }
            addPoint(i);
        }
    }
    timer.end();
    logger << "Adding time: " << timer.elapsed() << "s" << std::endl;
    if (cur_size_ == 1) {
        cur_size_ = num;
    } else {
        cur_size_ += num;
    }
    if (cur_size_ == oracle_->size()) {
        flatten_graph_ = FlattenHGraph(graph_);
        built_ = true;
    } else {
        built_ = false;
    }
}

void
hnsw::HNSW::build() {
    print_info();
    if (built_) {
        logger << "Index is already built, skipping build." << std::endl;
        return;
    }

    Timer timer;
    timer.start();

    build_internal();

    timer.end();
    logger << "Indexing time: " << timer.elapsed() << "s" << std::endl;

    flatten_graph_ = FlattenHGraph(graph_);
    built_ = true;
}

void
hnsw::HNSW::add(DatasetPtr& dataset) {
    if (!built_) {
        throw std::runtime_error("Index is not built yet");
    }
    built_ = false;

    Timer timer;
    timer.start();

    auto cur_size = oracle_->size();
    auto total = dataset->getOracle()->size() + cur_size;
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    levels_.reserve(total);
    levels_.resize(total);
    for (auto i = cur_size; i < total; i++) {
        levels_[i] = (int)(-log(distribution(random_engine_)) * reverse_);
        max_level_ = std::max(max_level_, levels_[i]);
    }
    graph_.resize(max_level_ + 1);
    for (auto& level : graph_) {
        level.resize(total);
    }

    {
        std::vector<DatasetPtr> datasets = {dataset};
        dataset_->merge(datasets);
    }

#pragma omp parallel for schedule(dynamic)
    for (int i = cur_size; i < total; ++i) {
        if (i % 10000 == 0) {
            logger << "Adding " << i << " / " << total << std::endl;
        }
        addPoint(i);
    }

    timer.end();
    logger << "Adding time: " << timer.elapsed() << "s" << std::endl;

    flatten_graph_ = FlattenHGraph(graph_);
    built_ = true;
}

void
hnsw::HNSW::set_max_neighbors(int max_neighbors) {
    max_neighbors_ = max_neighbors;
    max_base_neighbors_ = max_neighbors * 2;
    reverse_ = 1 / log(1.0 * max_neighbors_);
}

void
hnsw::HNSW::set_ef_construction(int ef_construction) {
    this->ef_construction_ = ef_construction;
}

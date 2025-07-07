#include "nsw.h"

nsw::NSW::NSW(DatasetPtr& dataset, int max_neighbors, int ef_construction)
    : Index(dataset), max_neighbors_(max_neighbors), ef_construction_(ef_construction) {
}

//void
//nsw::NSW::build() {
//    Timer timer;
//    timer.start();
//
//    int total = oracle_->size();
//    graph_.emplace_back(max_neighbors_);
//
//    for (int i = 1; i < total; ++i) {
//        if (i % 10000 == 0) {
//            logger << "Processing " << i << " / " << graph_.size() << std::endl;
//        }
//        addPoint(i);
//    }
//
//    timer.end();
//    logger << "Construction time: " << timer.elapsed() << "s" << std::endl;
//}

void
nsw::NSW::addPoint(unsigned int index) {
    std::lock_guard<std::mutex> cur(graph_[index].lock_);
    auto res = knn_search(oracle_.get(),
                          visited_list_pool_.get(),
                          graph_,
                          (*oracle_)[index],
                          max_neighbors_,
                          ef_construction_,
                          -1,
                          index);
    for (auto& re : res) {
        graph_[index].addNeighbor(re, max_neighbors_);
        {
            std::lock_guard<std::mutex> guard(graph_[re.id].lock_);
            graph_[re.id].addNeighbor(Neighbor(index, re.distance, false), max_neighbors_);
        }
    }
}

Neighbors
nsw::NSW::multisearch(const Graph& graph_,
                      const IndexOracle<float>& oracle,
                      unsigned int query,
                      int attempts,
                      int k) {
    std::priority_queue<Neighbor> candidates;
    Neighbors results;
    std::vector<bool> visited(graph_.size(), false);
    std::mt19937 rng(2024);
    for (int it = 0; it < attempts; ++it) {
        Neighbors temp_results;

        int entry_point = rng() % graph_.size();
        while (visited[entry_point]) {
            entry_point = rng() % graph_.size();
        }
        auto dist = oracle(query, entry_point);
        candidates.emplace(entry_point, dist, false);
        while (!candidates.empty()) {
            auto& c = candidates.top();
            candidates.pop();

            if (results.size() >= k && c.distance > results[k - 1].distance) {
                break;
            }

            for (auto& v : graph_[c.id].candidates_) {
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
        results.erase(
            std::unique(results.begin(),
                        results.end(),
                        [](const Neighbor& a, const Neighbor& b) { return a.id == b.id; }),
            results.end());
    }
    results.resize(k);
    return results;
}

void
nsw::NSW::build_internal() {
    int total = oracle_->size();
#pragma omp parallel for schedule(dynamic)
    for (int i = 1; i < total; ++i) {
        if (i % 10000 == 0) {
            logger << "Processing " << i << " / " << graph_.size() << std::endl;
        }
        addPoint(i);
    }

    flatten_graph_ = FlattenGraph(graph_);
}

void
nsw::NSW::add(DatasetPtr& dataset) {
    if (!built_) {
        throw std::runtime_error("Index is not built yet");
    }
    built_ = false;

    auto cur_size = oracle_->size();
    auto total = dataset->getOracle()->size() + cur_size;
    graph_.reserve(total);
    graph_.resize(total);
    {
        std::vector<DatasetPtr> datasets = {dataset};
        dataset_->merge(datasets);
    }
    print_info();

    Timer timer;
    timer.start();
#pragma omp parallel for schedule(dynamic)
    for (int i = cur_size; i < total; ++i) {
        if (i % 10000 == 0) {
            logger << "Adding " << i << " / " << total << std::endl;
        }
        addPoint(i);
    }

    timer.end();
    logger << "Adding time: " << timer.elapsed() << "s" << std::endl;

    flatten_graph_ = FlattenGraph(graph_);
    built_ = true;
}
void
nsw::NSW::print_info() const {
    Index::print_info();
    logger << "NSW Index Info:" << std::endl;
    logger << "Max neighbors: " << max_neighbors_ << std::endl;
    logger << "EF Construction: " << ef_construction_ << std::endl;
}

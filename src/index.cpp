#include "index.h"

Index::Index()
    : dataset_(nullptr),
      base_(nullptr),
      oracle_(nullptr),
      built_(false),
      visited_list_pool_(nullptr) {
}

Index::Index(DatasetPtr& dataset, bool allocate)
    : dataset_(dataset),
      oracle_(dataset->getOracle()),
      base_(dataset->getBasePtr()),
      visited_list_pool_(dataset->getVisitedListPool()),
      built_(false) {
    if (allocate) {
        graph_.reserve(oracle_->size());
        graph_.resize(oracle_->size());
    }
}

void
Index::build_internal() {
    throw std::runtime_error("Index does not support build");
}

void
Index::build() {
    Timer timer;
    timer.start();

    build_internal();

    timer.end();
    logger << "Indexing time: " << timer.elapsed() << "s" << std::endl;

    flatten_graph_ = FlattenGraph(graph_);
    built_ = true;
}

void
Index::reset(DatasetPtr& dataset) {
    dataset_ = dataset;
    oracle_ = dataset->getOracle();
    base_ = dataset->getBasePtr();
    visited_list_pool_ = dataset->getVisitedListPool();
    graph_.clear();
    graph_.reserve(oracle_->size());
    graph_.resize(oracle_->size());
    built_ = false;
}

Graph&
Index::extractGraph() {
    if (!built_) {
        throw std::runtime_error("Index is not built");
    }
    return graph_;
}

DatasetPtr&
Index::extractDataset() {
    return dataset_;
}

void
Index::add(DatasetPtr& dataset) {
    throw std::runtime_error("Index does not support add");
}

Neighbors
Index::search(const float* query, unsigned int topk, unsigned int L) const {
    if (!built_) {
        throw std::runtime_error("Index is not built");
    }
    return graph::search(oracle_.get(), visited_list_pool_.get(), flatten_graph_, query, topk, L);
}

FlattenGraph&
Index::extractFlattenGraph() {
    if (!built_) {
        throw std::runtime_error("Index is not built");
    }
    return flatten_graph_;
}

IndexWrapper::IndexWrapper(DatasetPtr& dataset, Graph& graph) {
    dataset_ = dataset;
    oracle_ = dataset->getOracle();
    base_ = dataset->getBasePtr();
    visited_list_pool_ = dataset->getVisitedListPool();
    graph_ = std::move(graph);
    flatten_graph_ = FlattenGraph(graph_);
    built_ = true;
}

IndexWrapper::IndexWrapper(IndexPtr& index) {
    dataset_ = index->extractDataset();
    oracle_ = dataset_->getOracle();
    base_ = dataset_->getBasePtr();
    visited_list_pool_ = dataset_->getVisitedListPool();

    graph_.reserve(oracle_->size());
    graph_.resize(oracle_->size());
    auto& graph = index->extractGraph();
    for (size_t i = 0; i < oracle_->size(); ++i) {
        auto& neighbors = graph[i].candidates_;
        graph_[i].candidates_.reserve(neighbors.size());
        for (auto& neighbor : neighbors) {
            graph_[i].candidates_.emplace_back(neighbor.id, neighbor.distance, false);
        }
    }

    flatten_graph_ = FlattenGraph(graph_);
    built_ = true;
}

void
IndexWrapper::append(std::vector<IndexPtr>& indexes) {
    built_ = false;

    std::vector<DatasetPtr> datasets = {dataset_};
    for (auto& index : indexes) {
        datasets.emplace_back(index->extractDataset());
    }

    dataset_ = Dataset::aggregate(datasets);
    oracle_ = dataset_->getOracle();
    visited_list_pool_ = dataset_->getVisitedListPool();
    base_ = dataset_->getBasePtr();
    graph_.reserve(oracle_->size());

    for (auto& index : indexes) {
        auto& graph = index->extractGraph();
        for (auto& neighborhood : graph) {
            graph_.emplace_back(neighborhood);
        }
    }

    flatten_graph_ = FlattenGraph(graph_);
    built_ = true;
}

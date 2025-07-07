//
// Created by XiaoWu on 2025/2/14.
//

#ifndef MYANNS_INDEX_H
#define MYANNS_INDEX_H

#include <omp.h>

#include <random>
#include <unordered_set>

#include "dataset.h"
#include "dtype.h"
#include "graph.h"
#include "logger.h"
#include "metric.h"
#include "timer.h"
#include "visittable.h"

using namespace graph;

class Index {
protected:
    Graph graph_;

    DatasetPtr dataset_;

    OraclePtr oracle_;

    MatrixPtr<float> base_;

    VisitedListPoolPtr visited_list_pool_;

    FlattenGraph flatten_graph_;

    bool built_;

    virtual void
    build_internal();

public:
    Index();

    explicit Index(DatasetPtr& dataset, bool allocate = true);

    explicit Index(DatasetPtr& dataset, Graph& graph);

    virtual ~Index() = default;

    virtual void
    reset(DatasetPtr& dataset);

    virtual void
    build();

    /**
     * Add a dataset to the existing index. Note that data from the dataset will be appended to the existing data.
     * @param dataset
     */
    virtual void
    add(DatasetPtr& dataset);

    virtual Graph&
    extractGraph();

    virtual FlattenGraph&
    extractFlattenGraph();

    virtual DatasetPtr&
    extractDataset();

    /**
     * @brief The basic search function. It initializes with L random nodes and greedily expands the candidates. The results are pruned by the topk.
     * @param query
     * @param topk
     * @param L
     * @return
     */
    virtual Neighbors
    search(const float* query, unsigned int topk, unsigned int L) const;

    virtual void
    print_info() const;
};

using IndexPtr = std::shared_ptr<Index>;

class IndexWrapper : public Index {
public:
    explicit IndexWrapper(DatasetPtr& dataset, Graph& graph);

    explicit IndexWrapper(IndexPtr& index);

    IndexWrapper() = default;

    ~IndexWrapper() override = default;

    void
    append(std::vector<IndexPtr>& indexes);
};

// TODO Support IndexFactory

#endif  //MYANNS_INDEX_H

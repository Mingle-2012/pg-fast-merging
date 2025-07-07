//
// Created by XiaoWu on 2024/12/10.
//

#ifndef MYANNS_HNSW_H
#define MYANNS_HNSW_H

#include <omp.h>

#include <random>
#include <unordered_set>

#include "index.h"

namespace hnsw {

class HNSW : public Index {
protected:
    HGraph graph_;

    FlattenHGraph flatten_graph_;

    uint32_t max_neighbors_{};

    uint32_t max_base_neighbors_{};

    uint32_t max_level_{};

    uint32_t cur_max_level_{};

    uint32_t ef_construction_{};

    uint64_t cur_size_{1};

    std::vector<uint32_t> levels_;

    double reverse_{};

    std::unordered_set<int> visited_table_;

    std::default_random_engine random_engine_;

    std::mutex graph_lock_;

    virtual void
    addPoint(unsigned index);

    Neighbors
    searchLayer(
        const Graph& graph, const float* query, size_t topk, size_t L, size_t entry_id) const;

    /**
   * This implementation follows the original paper.
   * @param graph
   * @param oracle
   * @param query
   * @param enter_point
   * @param ef
   * @return
   */
    Neighbors
    searchLayer(Graph& graph, IndexOracle<float>& oracle, float* query, int enter_point, int ef);

    static int
    seekPos(const Neighbors& vec);

    void
    prune(Neighbors& candidates, int max_neighbors);

    void
    build_internal() override;

public:
    uint32_t enter_point_{};

    HNSW(DatasetPtr& dataset, int max_neighbors, int ef_construction);

    HNSW(DatasetPtr& dataset,
         HGraph& graph,
         bool partial = false,
         int max_neighbors = 32,
         int ef_construction = 200);

    ~HNSW() override = default;

    void
    set_max_neighbors(int max_neighbors);

    void
    set_ef_construction(int ef_construction);

    void
    build() override;

    void
    partial_build(uint64_t start, uint64_t end);

    void
    partial_build(uint64_t num = 0);

    Graph&
    extractGraph() override;

    HGraph&
    extractHGraph();

    void
    add(DatasetPtr& dataset) override;

    Neighbors
    search(const float* query, unsigned int topk, unsigned int L) const override;

    void
    print_info() const override;
};
}  // namespace hnsw

#endif  // MYANNS_HNSW_H

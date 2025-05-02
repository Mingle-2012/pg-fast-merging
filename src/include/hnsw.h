//
// Created by XiaoWu on 2024/12/10.
//

#ifndef MERGE_HNSW_H
#define MERGE_HNSW_H

#include <omp.h>

#include <random>
#include <unordered_set>

#include "index.h"

namespace hnsw {

class HNSW : public Index {
protected:
    HGraph graph_;

    FlattenHGraph flatten_graph_;

    uint32_t max_neighbors_;

    uint32_t max_base_neighbors_;

    uint32_t max_level_;

    uint32_t cur_max_level_;

    uint32_t ef_construction_;

    std::vector<uint32_t> levels;

    double reverse_;

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
    uint32_t enter_point_;

    HNSW(DatasetPtr& dataset, int max_neighbors, int ef_construction);

    ~HNSW() override = default;

    void
    set_max_neighbors(int max_neighbors);

    void
    set_ef_construction(int ef_construction);

    void
    build() override;

    Graph&
    extractGraph() override;

    HGraph&
    extractHGraph();

    void
    add(DatasetPtr& dataset) override;

    Neighbors
    search(const float* query, unsigned int topk, unsigned int L) const override;

    //        virtual Neighbors
    //        HNSW_search(HGraph &hnsw_graph,
    //                    IndexOracle<float> &oracle,
    //                    float *query,
    //                    int topk,
    //                    int ef_search) const;
};
}  // namespace hnsw

#endif  // MERGE_HNSW_H

//
// Created by XiaoWu on 2025/3/2.
//

#ifndef MYANNS_MGRAPH_H
#define MYANNS_MGRAPH_H

#include "fgim.h"

using namespace graph;

// MGraph refers to Merged hierarchical Graph-based anns index
class MGraph : public FGIM {
private:
    HGraph graph_;

    FlattenHGraph flatten_graph_;

    std::default_random_engine random_engine_;

    double reverse_;

    uint32_t ef_construction_;

    uint32_t max_level_;

    uint32_t cur_max_level_;

    std::vector<uint32_t> levels_;

    std::mutex graph_lock_;

    void
    CrossQuery(std::vector<IndexPtr>& indexes) override;

    void
    Refinement() override;

    void
    ReconstructHGraph();

    void
    heuristic(Neighbors& candidates, unsigned max_degree);

public:
    uint32_t enter_point_;

    MGraph();

    explicit MGraph(unsigned int max_degree, unsigned int ef_construction, float sample_rate = 0.3);

    explicit MGraph(DatasetPtr& dataset,
                    unsigned int max_degree,
                    unsigned int ef_construction,
                    float sample_rate = 0.3);

    Graph&
    extractGraph() override;

    HGraph&
    extractHGraph();

    void
    Combine(std::vector<IndexPtr>& indexes) override;

    Neighbors
    search(const float* query, unsigned int topk, unsigned int L) const override;

    void
    print_info() const override;
};

#endif  //MYANNS_MGRAPH_H

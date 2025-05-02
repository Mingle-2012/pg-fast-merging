//
// Created by XiaoWu on 2024/11/24.
//

#ifndef MERGE_MERGE_H
#define MERGE_MERGE_H

#include "dataset.h"
#include "dtype.h"
#include "graph.h"
#include "hnsw.h"
#include "index.h"
#include "logger.h"
#include "timer.h"

class FGIM : public Index {
protected:
    uint32_t max_degree_;

    uint32_t max_base_degree_;

    float sample_rate_;

    //        void
    //        Sampling(Graph &graph,
    //                 const Graph &g1,
    //                 const Graph &g2,
    //                 OraclePtr &oracle1,
    //                 OraclePtr &oracle2,
    //                 OraclePtr &oracle);
    virtual void
    CrossQuery(std::vector<IndexPtr>& indexes);

    virtual void
    Refinement();

    void
    update_neighbors(Graph& graph);

    void
    prune(Graph& graph, bool add = false);

    void
    add_reverse_edge(Graph& graph);

    void
    connect_no_indegree(Graph& graph);

public:
    static constexpr unsigned ITER_MAX = 30;

    static constexpr unsigned SAMPLES = 100;

    static constexpr float THRESHOLD = 0.002;

    FGIM();

    explicit FGIM(unsigned max_degree, float sample_rate = 0.3);

    FGIM(DatasetPtr& dataset, unsigned max_degree, float sample_rate = 0.3, bool allocate = true);

    ~FGIM() override = default;

    //        /**
    //               * @brief FGIM two PGs
    //               * @param g1  The first graph
    //               * @param oracle1 The distance oracle of the first graph
    //               * @param g2  The second nearest neighbor graph
    //               * @param oracle2 The distance oracle of the second graph
    //               * @param oracle The distance oracle
    //               * @return The merged graph
    //               */
    //        Graph
    //        merge(const Graph &g1,
    //              OraclePtr &oracle1,
    //              const Graph &g2,
    //              OraclePtr &oracle2,
    //              OraclePtr &oracle);

    virtual void
    Combine(std::vector<IndexPtr>& indexes);
};

#endif  // MERGE_MERGE_H

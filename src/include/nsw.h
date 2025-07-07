//
// Created by XiaoWu on 2024/11/23.
//

#ifndef MYANNS_NSW_H
#define MYANNS_NSW_H

#include <omp.h>

#include <random>

#include "index.h"

namespace nsw {
class NSW : public Index {
private:
    int max_neighbors_;

    int ef_construction_;

    void
    addPoint(unsigned index);

    Neighbors
    multisearch(
        const Graph& graph, const IndexOracle<float>& oracle, unsigned query, int attempts, int k);

    void
    build_internal() override;

public:
    NSW(DatasetPtr& dataset, int max_neighbors, int ef_construction);

    void
    add(graph::DatasetPtr& dataset) override;

    ~NSW() override = default;

    void
    set_max_neighbors(int max_neighbors) {
        this->max_neighbors_ = max_neighbors;
    }

    void
    set_ef_construction(int ef_construction) {
        this->ef_construction_ = ef_construction;
    }

    void
    print_info() const override;
};
}  // namespace nsw

#endif  // MYANNS_NSW_H

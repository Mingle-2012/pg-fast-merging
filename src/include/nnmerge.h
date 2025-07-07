//
// Created by XiaoWu on 2025/3/29.
//

#ifndef MYANNS_NNMERGE_H
#define MYANNS_NNMERGE_H

#include <omp.h>

#include <random>

#include "nndescent.h"

using namespace nndescent;

namespace nnmerge {
class NNMerge : public NNDescent {
private:
    float alpha{0.5};

    uint32_t graph_size_1;

    uint32_t graph_size_2;

    void
    splitGraph(Graph& G_v, const Graph& knng1, const Graph& knng2);

    void
    addSamples();

    void
    nndescent();

    void
    mergeGraph(Graph& G_v);

    void
    build_internal() override;

    int
    applyUpdate(unsigned int sample) override;

public:
    explicit NNMerge(DatasetPtr& dataset,
                     int K,
                     float rho = 0.5,
                     float delta = 0.001,
                     int iteration = 20,
                     float alpha = 0.5);

    ~NNMerge() override = default;

    void
    Combine(const IndexPtr& index1, const IndexPtr& index2);

    void
    print_info() const override;
};
}  // namespace nnmerge
#endif  //MYANNS_NNMERGE_H

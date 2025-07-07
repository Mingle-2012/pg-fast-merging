//
// Created by XiaoWu on 2024/11/23.
//

#ifndef MYANNS_NNDESCENT_H
#define MYANNS_NNDESCENT_H

#include <omp.h>

#include <random>

#include "index.h"

namespace nndescent {

class NNDescent : public Index {
protected:
    unsigned K_{64};

    float rho_{0.5};

    float delta_{0.001};

    unsigned iteration_{100};

    void
    initializeGraph();

    void
    generateUpdate();

    virtual int
    applyUpdate(unsigned sample);

    void
    clearGraph();

    void
    build_internal() override;

public:
    NNDescent(DatasetPtr& dataset, int K, float rho = 0.5, float delta = 0.001, int iteration = 20);

    ~NNDescent() override = default;

    void
    print_info() const override;
};
}  // namespace nndescent

#endif  // MYANNS_NNDESCENT_H

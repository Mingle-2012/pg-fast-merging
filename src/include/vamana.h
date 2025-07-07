//
// Created by XiaoWu on 2024/11/23.
//

#ifndef MYANNS_VAMANA_H
#define MYANNS_VAMANA_H

#include <omp.h>

#include <random>

#include "index.h"
#include "kmeans.h"

namespace diskann {
class Vamana : public Index {
private:
    /**
       * alpha
       */
    float alpha_;

    /*
       * search pool size
       */
    int L_;

    /**
       * maximum number of neighbors
       */
    int R_;

    void
    RobustPrune(float alpha, int point, Neighbors& candidates);

    void
    build_internal() override;

public:
    /**
             *
             * @param dataset
             * @param alpha
             * @param L
             * @param R
             */
    Vamana(DatasetPtr& dataset, float alpha, int L, int R);

    ~Vamana() override = default;

    void
    set_alpha(float alpha);

    void
    set_L(int L);

    void
    set_R(int R);

    void
    partial_build(std::vector<uint32_t>& permutation);

    void
    print_info() const override;
};

class DiskANN : public Index {
private:
    float alpha_;

    int L_;

    int R_;

    int k_;

    int ell_;

    void
    build_internal() override;

public:
    DiskANN(DatasetPtr& dataset, float alpha, int L, int R, int k, int ell);

    ~DiskANN() override = default;
};

}  // namespace diskann

#endif  // MYANNS_VAMANA_H

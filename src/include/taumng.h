//
// Created by XiaoWu on 2024/11/23.
//

#ifndef MYANNS_TAUMNG_H
#define MYANNS_TAUMNG_H

#include <omp.h>

#include <random>

#include "index.h"

namespace taumng {

class TauMNG : public Index {
private:
    /**
                       * tau
                       */
    float t_;

    /**
                       * same as k in knn HNSW_search
                       */
    int h_;

    /**
                       * search pool size
                       */
    int b_;

    Graph& base_;

    void
    build_internal() override;

public:
    /**
         * @brief Build a TauMNG graph.
         * @param oracle
         * @param graph
         * @param t
         * @param h
         * @param b
         */
    TauMNG(DatasetPtr& dataset, Graph& base, float t, int h, int b);

    void
    set_b(int b);

    void
    set_h(int h);

    void
    print_info() const override;
};
}  // namespace taumng

#endif  // MYANNS_TAUMNG_H

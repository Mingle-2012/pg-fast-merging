//
// Created by XiaoWu on 2024/11/23.
//

#ifndef MERGE_TAUMNG_H
#define MERGE_TAUMNG_H

#include <random>
#include <omp.h>
#include "graph.h"
#include "dtype.h"
#include "metric.h"
#include "logger.h"
#include "timer.h"

using namespace merge;

namespace taumng {

class TauMNG {
    private:
        /**
         * tau
         */
        float t_;

        /**
         * same as k in knn search
         */
        int h_;

        /**
         * search pool size
         */
        int b_;

    public:
        TauMNG(float t,
               int h,
               int b);

        void set_b(int b);

        void set_h(int h);

        void build(Graph &graph,
                   IndexOracle &oracle);

    };
}

#endif //MERGE_TAUMNG_H

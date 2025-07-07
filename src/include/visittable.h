//
// Created by XiaoWu on 2025/3/1.
//

#ifndef MYANNS_VISITTABLE_H
#define MYANNS_VISITTABLE_H

#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>

#include "logger.h"

using namespace graph;

namespace graph {

class VisitedList {
public:
    unsigned int version_;
    unsigned int* block_;
    size_t size_;

    explicit VisitedList(size_t num);

    void
    reset();

    ~VisitedList();
};

using VisitedListPtr = std::shared_ptr<VisitedList>;

class VisitedListPool {
private:
    size_t num_;

    std::deque<VisitedListPtr> pool_;

    std::mutex guard_;

public:
    VisitedListPool();

    static std::shared_ptr<VisitedListPool>
    getInstance(size_t num);

    VisitedListPtr
    getFreeVisitedList();

    void
    releaseVisitedList(const VisitedListPtr& ptr);
};

using VisitedListPoolPtr = std::shared_ptr<VisitedListPool>;

}  // namespace graph

#endif  //MYANNS_VISITTABLE_H

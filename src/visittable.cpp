//
// Created by XiaoWu on 2025/3/1.
//

#include "visittable.h"

graph::VisitedList::VisitedList(size_t num)
    : size_(num), version_(-1), block_(new unsigned int[num]) {
}

void
graph::VisitedList::reset() {
    if (not block_) {
        block_ = (unsigned int*)malloc(size_ * sizeof(unsigned int));
    }
    version_++;
    if (version_ == 0 || version_ >= std::numeric_limits<unsigned int>::max() - 1) {
        memset(block_, 0, sizeof(unsigned int) * size_);
        version_++;
    }
}

graph::VisitedList::~VisitedList() {
    delete[] block_;
}

graph::VisitedListPool::VisitedListPool() : num_(0) {
}

std::shared_ptr<VisitedListPool>
graph::VisitedListPool::getInstance(size_t num) {
    auto ptr = std::make_shared<VisitedListPool>();
    ptr->num_ = num;
    return ptr;
}

graph::VisitedListPtr
graph::VisitedListPool::getFreeVisitedList() {
    VisitedListPtr ptr;
    {
        std::unique_lock<std::mutex> lock(guard_);
        if (pool_.empty()) {
            ptr = std::make_shared<VisitedList>(num_);
        } else {
            ptr = pool_.front();
            pool_.pop_front();
        }
    }
    ptr->reset();
    return ptr;
}

void
graph::VisitedListPool::releaseVisitedList(const VisitedListPtr& ptr) {
    std::unique_lock<std::mutex> lock(guard_);
    pool_.push_back(ptr);
}

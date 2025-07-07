//
// Created by XiaoWu on 2025/3/29.
//

#ifndef MYANNS_KMEANS_H
#define MYANNS_KMEANS_H

#include <omp.h>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <random>

#include "dataset.h"

// TODO Only center is needed. Point can be replaced by short*.
struct Point {
    Point() = default;

    Point(int id, int group) : id_(id), group_(group) {
    }

    ~Point() {
        delete[] data_;
    }

    int id_{0};
    int group_{0};
    float* data_{nullptr};
};

class Kmeans {
private:
    std::vector<Point> points_;
    std::vector<Point> centers_;
    int maxIteration_{100};
    int k_;
    int pointNumber_;

    OraclePtr& oracle_;

    void
    Init();

    void
    Cluster();

    void
    Center();

public:
    Kmeans(DatasetPtr& dataset, int k);

    /**
     * Get the ell nearest centers of the p
     * @param p
     * @param ell
     * @return
     */
    std::vector<int>
    NearestCenter(int p, int ell);

    void
    Run();
};

#endif  //MYANNS_KMEANS_H

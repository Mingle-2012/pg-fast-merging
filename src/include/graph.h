//
// Created by XiaoWu on 2024/11/23.
//

/**
 * This implementation is based on the following references:
 * See https://github.com/facebookresearch/faiss and https://github.com/JieFengWang/mini_rnn for more details.
 */

#ifndef MERGE_GRAPH_H
#define MERGE_GRAPH_H

#include <cstdlib>
#include <mutex>
#include <random>
#include <algorithm>
#include <iostream>
#include <queue>
#include <functional>
#include <stack>
#include <fstream>
#include "dtype.h"

namespace merge {

    struct Node {
        int id;
        float distance;

        Node() = default;

        Node(int i,
             float d);

        inline bool operator<(const Node &n) const;

        Node &operator=(const Node &other);
    };

    struct Neighbor {
        int id;
        float distance;
        bool flag;

        Neighbor() = default;

        Neighbor(int i,
                 float d,
                 bool f);

        inline bool operator<(const Neighbor &n) const{
            return distance < n.distance;
        }

        inline bool operator==(const Neighbor &n) const{
            return id == n.id;
        }

        Neighbor &operator=(const Neighbor &other);
    };

    struct Neighborhood {
        std::mutex lock_;
        std::vector<Neighbor> candidates_;
        std::vector<int> old_;
        std::vector<int> new_;
        std::vector<int> reverse_old_;
        std::vector<int> reverse_new_;

        int M_{std::numeric_limits<int>::max()};

        Neighborhood() = default;

        explicit Neighborhood(int M);

        Neighborhood(int s,
                     std::mt19937 &rng,
                     int N);

        Neighborhood &operator=(const Neighborhood &other);

        Neighborhood(const Neighborhood &other);

        unsigned insert(int id,
                        float dist);

        void addNeighbor(Neighbor nn);
    };

    using Graph = std::vector<Neighborhood>;

    inline void gen_random(std::mt19937 &rng,
                           int *addr,
                           int size,
                           int N){
        for (int i = 0; i < size; ++i) {
            addr[i] = rng() % (N - size);
        }
        std::sort(addr, addr + size);
        for (int i = 1; i < size; ++i) {
            if (addr[i] <= addr[i - 1]) {
                addr[i] = addr[i - 1] + 1;
            }
        }
        int off = rng() % N;
        for (int i = 0; i < size; ++i) {
            addr[i] = (addr[i] + off) % N;
        }
    }

    void project(const Graph &graph,
                 std::vector<int> &final_graph,
                 std::vector<int> &offsets);

    std::vector<Neighbor> knn_search(IndexOracle &oracle,
                                     Graph &graph,
                                     const float *query,
                                     int topk,
                                     int L,
                                     int entry_id = -1,
                                     int graph_sz = -1);

    std::vector<Neighbor> track_search(IndexOracle &oracle,
                                       const Graph &graph,
                                       const float *query,
                                       int entry_id,
                                       int L);

    std::vector<Neighbor> search(IndexOracle &oracle,
                                 const std::vector<int> &final_graph,
                                 const std::vector<int> &offsets,
                                 const float *query,
                                 int topk,
                                 int total,
                                 int search_L,
                                 int K0 = 128);

    void saveGraph(Graph &graph,
                   const std::string &filename);

    void loadGraph(Graph &graph,
                   const std::string &filename);
}

#endif //MERGE_GRAPH_H

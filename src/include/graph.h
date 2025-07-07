//
// Created by XiaoWu on 2024/11/23.
//

/**
 * This implementation is based on the following references:
 * See https://github.com/facebookresearch/faiss and
 * https://github.com/JieFengWang/mini_rnn for more details.
 */

#ifndef MYANNS_GRAPH_H
#define MYANNS_GRAPH_H

#include <algorithm>
#include <bitset>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>
#include <random>
#include <stack>

#include "dataset.h"
#include "dtype.h"
#include "visittable.h"

namespace graph {

struct Node {
    int id;
    float distance;

    Node() = default;

    Node(int i, float d);

    inline bool
    operator<(const Node& n) const {
        return distance < n.distance;
    }

    Node&
    operator=(const Node& other);
};

using Nodes = std::vector<Node>;

struct Neighbor {
    int id;
    float distance;
    bool flag;

    Neighbor() = default;

    Neighbor(int i, float d, bool f);

    inline bool
    operator<(const Neighbor& n) const {
        return distance < n.distance;
    }

    inline bool
    operator==(const Neighbor& n) const {
        return id == n.id;
    }

    Neighbor(const Neighbor& other);

    Neighbor&
    operator=(const Neighbor& other);
};

using Neighbors = std::vector<Neighbor>;

struct Neighborhood {
    std::mutex lock_;
    Neighbors candidates_;
    std::vector<int> old_;
    std::vector<int> new_;
    std::vector<int> reverse_old_;
    std::vector<int> reverse_new_;

    float greatest_distance = std::numeric_limits<float>::max();

    Neighborhood() = default;

    Neighborhood(int s, std::mt19937& rng, int N);

    Neighborhood&
    operator=(const Neighborhood& other);

    Neighborhood(const Neighborhood& other);

    /**
             * Assure that candidates_ is already a heap with manually reserved capacity.
             * Otherwise, this operation is invalid.
             * @param id
             * @param dist
             * @return
             */
    unsigned
    pushHeap(int id, float dist);

    /**
             * This function add nn into sorted candidates_ within a capacity limit.
             * A capacity of negative numbers refers to no limit.
             * @param nn
             * @param capacity
             */
    void
    addNeighbor(Neighbor nn, int capacity = -1);

    /**
             * @brief Move the content of the other neighborhood to this neighborhood. Note that only the candidates are moved.
             * @param other
             */
    void
    move(Neighborhood& other);
};

using Graph = std::vector<Neighborhood>;
using HGraph = std::vector<Graph>;

struct FlattenGraph {
    std::vector<int> final_graph;
    std::vector<int> offsets;

    FlattenGraph() = default;

    explicit FlattenGraph(const Graph& graph);

    virtual std::vector<int> operator[](int i) const;

    virtual ~FlattenGraph() = default;
};

struct FlattenHGraph {
    std::vector<FlattenGraph> graphs_;

    FlattenHGraph() = default;

    explicit FlattenHGraph(const HGraph& graph);

    FlattenGraph& operator[](int i) const;

    [[nodiscard]] int
    size() const;
};

inline void
gen_random(std::mt19937& rng, int* addr, int size, int N) {
    if (N - size <= 0) {
        for (int i = 0; i < size; ++i) {
            addr[i] = i;
        }
        return;
    }
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

inline int
insert_into_pool(Neighbor* addr, int size, Neighbor nn) {
    int left = 0, right = size - 1;
    if (addr[left].distance > nn.distance) {
        memmove((char*)&addr[left + 1], &addr[left], size * sizeof(Neighbor));
        addr[left] = nn;
        return left;
    }
    if (addr[right].distance < nn.distance) {
        addr[size] = nn;
        return size;
    }
    while (left < right - 1) {
        int mid = (left + right) / 2;
        if (addr[mid].distance > nn.distance)
            right = mid;
        else
            left = mid;
    }
    while (left > 0) {
        if (addr[left].distance < nn.distance)
            break;
        if (addr[left].id == nn.id)
            return size + 1;
        left--;
    }
    if (addr[left].id == nn.id || addr[right].id == nn.id)
        return size + 1;
    memmove((char*)&addr[right + 1], &addr[right], (size - right) * sizeof(Neighbor));
    addr[right] = nn;
    return right;
}

Neighbors
knn_search(IndexOracle<float>* oracle,
           VisitedListPool* visited_list_pool,
           Graph& graph,
           const float* query,
           int topk,
           int L,
           int entry_id = -1,
           int graph_sz = -1);

Neighbors
search_layer(IndexOracle<float>* oracle,
             VisitedListPool* visited_list_pool,
             HGraph& hgraph,
             int layer,
             const float* query,
             int topk,
             int L,
             int entry_id = -1);

Neighbors
track_search(IndexOracle<float>* oracle,
             VisitedListPool* visited_list_pool,
             Graph& graph,
             const float* query,
             int L,
             int entry_id);

/**
     * @brief Search the graph with the given query.
     * @param oracle
     * @param fg
     * @param query
     * @param topk
     * @param search_L
     * @param entry_id
     * @param K0
     * @return
     */
Neighbors
search(IndexOracle<float>* oracle,
       VisitedListPool* visited_list_pool,
       const FlattenGraph& fg,
       const float* query,
       int topk,
       int search_L,
       int entry_id = -1,
       int K0 = 128);

int
checkConnectivity(const Graph& graph);

std::string
saveGraph(Graph& graph, const std::string& filename);

std::string
saveHGraph(HGraph& hgraph, const std::string& filename);

void
loadGraph(Graph& graph, const std::string& index_path, const OraclePtr& oracle = nullptr);

void
loadHGraph(HGraph& hgraph, const std::string& index_path, const OraclePtr& oracle = nullptr);
}  // namespace graph

#endif  // MYANNS_GRAPH_H

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

    int seed = 2024;

    /**
     * @brief Generate random numbers
     * @param rng
     * @param addr
     * @param size Size of the array
     * @param N Maximum value
     */
    inline void gen_random(std::mt19937 &rng,
                           int *addr,
                           int size,
                           int N) {
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

    struct Node {
        int id;
        float distance;

        Node() = default;

        Node(int i,
             float d) : id(i), distance(d) {}

        inline bool operator<(const Node &n) const {
            return distance < n.distance;
        }

        Node &operator=(const Node &other) {
            if (this == &other)
                return *this;
            id = other.id;
            distance = other.distance;
            return *this;
        }
    };

    struct Neighbor {
        int id;
        float distance;
        bool flag;
//        bool visited;

        Neighbor() = default;

        Neighbor(int i,
                 float d,
                 bool flag) : id(i), distance(d), flag(flag) {}

        inline bool operator<(const Neighbor &n) const {
            return distance < n.distance;
        }

        inline bool operator==(const Neighbor &n) const {
            return id == n.id;
        }

        Neighbor &operator=(const Neighbor &other) {
            if (this == &other)
                return *this;
            id = other.id;
            distance = other.distance;
            flag = other.flag;
            return *this;
        }
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

        explicit Neighborhood(int M) : M_(M) {
            M_ = M;
            candidates_.reserve(M_);
        }

        Neighborhood(int s,
                     std::mt19937 &rng,
                     int N) {
            new_.resize(s * 2);
            gen_random(rng, new_.data(), static_cast<int>(new_.size()), N);
        }

        Neighborhood &operator=(const Neighborhood &other) {
            if (this == &other)
                return *this;
            candidates_.clear();
            new_.clear();
            old_.clear();
            reverse_old_.clear();
            reverse_new_.clear();
            candidates_.reserve(other.candidates_.capacity());
            new_.reserve(other.new_.capacity());
            old_.reserve(other.old_.capacity());
            reverse_old_.reserve(other.reverse_old_.capacity());
            reverse_new_.reserve(other.reverse_new_.capacity());
            std::copy(
                    other.candidates_.begin(),
                    other.candidates_.end(),
                    std::back_inserter(candidates_));
            std::copy(
                    other.new_.begin(),
                    other.new_.end(),
                    std::back_inserter(new_));
            std::copy(
                    other.old_.begin(),
                    other.old_.end(),
                    std::back_inserter(old_));
            std::copy(
                    other.reverse_old_.begin(),
                    other.reverse_old_.end(),
                    std::back_inserter(reverse_old_));
            std::copy(
                    other.reverse_new_.begin(),
                    other.reverse_new_.end(),
                    std::back_inserter(reverse_new_));
            return *this;
        }

        Neighborhood(const Neighborhood &other) {
            new_.clear();
            candidates_.clear();
            old_.clear();
            reverse_old_.clear();
            reverse_new_.clear();
            new_.reserve(other.new_.capacity());
            candidates_.reserve(other.candidates_.capacity());
            old_.reserve(other.old_.capacity());
            reverse_old_.reserve(other.reverse_old_.capacity());
            reverse_new_.reserve(other.reverse_new_.capacity());
            std::copy(
                    other.new_.begin(),
                    other.new_.end(),
                    std::back_inserter(new_));
            std::copy(
                    other.candidates_.begin(),
                    other.candidates_.end(),
                    std::back_inserter(candidates_));
            std::copy(
                    other.old_.begin(),
                    other.old_.end(),
                    std::back_inserter(old_));
            std::copy(
                    other.reverse_old_.begin(),
                    other.reverse_old_.end(),
                    std::back_inserter(reverse_old_));
            std::copy(
                    other.reverse_new_.begin(),
                    other.reverse_new_.end(),
                    std::back_inserter(reverse_new_));
        }

        unsigned insert(int id,
                        float dist) {
            std::lock_guard<std::mutex> guard(lock_);
            if (!candidates_.empty() && dist >= candidates_.front().distance)
                return 0;
            for (auto &candidate: candidates_) {
                if (id == candidate.id)
                    return 0;
            }
            if (candidates_.size() < candidates_.capacity()) {
                candidates_.emplace_back(id, dist, true);
                std::push_heap(candidates_.begin(), candidates_.end());
            } else {
                std::pop_heap(candidates_.begin(), candidates_.end());
                candidates_[candidates_.size() - 1] = Neighbor(id, dist, true);
                std::push_heap(candidates_.begin(), candidates_.end());
            }
            return 1;
        }

        void addNeighbor(Neighbor nn) {
            std::lock_guard<std::mutex> guard(lock_);
            auto it = std::lower_bound(candidates_.begin(), candidates_.end(), nn);
            if (it == candidates_.end() || it->id != nn.id) {
                candidates_.insert(it, nn);
            }
            if (candidates_.size() > M_) {
                candidates_.pop_back();
            }
        }
    };

    using Graph = std::vector<Neighborhood>;

    void resetFlag(Graph &graph) {
        for (auto &v: graph) {
            for (auto &n: v.candidates_) {
                n.flag = true;
            }
        }
    }

    /**
     * @brief Project the graph to 1d
     * @param graph
     * @param final_graph
     * @param offsets
     */
    void project(const Graph &graph,
                 std::vector<int> &final_graph,
                 std::vector<int> &offsets) {
        auto total = graph.size();
        offsets.resize(total + 1);
        offsets[0] = 0;
        for (int u = 0; u < total; ++u) {
            offsets[u + 1] = offsets[u] + (int) graph[u].candidates_.size();
        }

        final_graph.resize(offsets.back(), -1);
#pragma omp parallel for
        for (int u = 0; u < total; ++u) {
            auto &pool = graph[u].candidates_;
            int offset = offsets[u];
            for (int i = 0; i < pool.size(); ++i) {
                final_graph[offset + i] = pool[i].id;
            }
        }
    }

    inline int insert_into_pool(Neighbor *addr,
                                int size,
                                Neighbor nn) {
        int left = 0, right = size - 1;
        if (addr[left].distance > nn.distance) {
            memmove((char *) &addr[left + 1], &addr[left],
                    size * sizeof(Neighbor));
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
            if (addr[left].distance < nn.distance) break;
            if (addr[left].id == nn.id) return size + 1;
            left--;
        }
        if (addr[left].id == nn.id || addr[right].id == nn.id) return size + 1;
        memmove((char *) &addr[right + 1], &addr[right],
                (size - right) * sizeof(Neighbor));
        addr[right] = nn;
        return right;
    }

    std::vector<Neighbor> search(IndexOracle<float> &oracle,
                                 const std::vector<int> &final_graph,
                                 const std::vector<int> &offsets,
                                 const float *query,
                                 int topk,
                                 int total,
                                 int search_L,
                                 int K0 = 128) {
        int L = std::max(search_L, topk);
        std::vector<bool> visited(total, false);
        std::vector<Neighbor> retset(L + 1);
        std::vector<int> init_ids(L);
        std::mt19937 rng(seed);

        gen_random(rng, init_ids.data(), L, total);
        for (int i = 0; i < L; i++) {
            int id = init_ids[i];
            float dist = oracle(id, query);
            retset[i] = Neighbor(id, dist, true);
        }
        std::sort(retset.begin(), retset.begin() + L);

        int k = 0;
        while (k < L) {
            int nk = L;
            if (retset[k].flag) {
                retset[k].flag = false;
                int n = retset[k].id;
                int offset = offsets[n];
                int K = std::min(K0, offsets[n + 1] - offset);
//                int M0 = offsets[n + 1] - offset;
                for (int m = 0; m < K; ++m) {
                    int id = final_graph[offset + m];
                    if (visited[id]) continue;

                    visited[id] = true;
                    float dist = oracle(id, query);
                    if (dist >= retset[L - 1].distance) continue;

                    Neighbor nn(id, dist, true);
                    int r = insert_into_pool(retset.data(), L, nn);

                    if (r < nk) nk = r;
                }
            }
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
        retset.resize(topk);
        return retset;
    };

    // search from entry_id
    std::vector<Neighbor> knn_search(IndexOracle<float> &oracle,
                                     Graph &graph,
                                     const float *query,
                                     int topk,
                                     int L,
                                     int entry_id = -1,
                                     int graph_sz = -1) {
        if (graph_sz == -1) {
            graph_sz = graph.size();
        }
        std::vector<bool> visited(graph_sz, false);
        std::vector<Neighbor> retset(L + 1, Neighbor(-1, std::numeric_limits<float>::max(), false));
        if (entry_id == -1) {
            std::mt19937 rng(seed);
            entry_id = rng() % graph_sz;
        }
        auto dist = oracle(entry_id, query);
        retset[0] = Neighbor(entry_id, dist, true);
        int k = 0;
        while (k < L) {
            int nk = L;
            if (retset[k].flag) {
                retset[k].flag = false;
                int n = retset[k].id;
                for (const auto &candidate: graph[n].candidates_) {
                    int id = candidate.id;
                    if (visited[id]) continue;
                    visited[id] = true;
                    dist = oracle(id, query);
                    if (dist >= retset[L - 1].distance) continue;
                    Neighbor nn(id, dist, true);
                    int r = insert_into_pool(retset.data(), L, nn);
                    if (r < nk) nk = r;
                }
            }
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
        retset.resize(topk);
        return retset;
    };

    std::vector<Neighbor> track_search(IndexOracle<float> &oracle,
                                       const Graph &graph,
                                       const float *query,
                                       int entry_id,
                                       int L) {
        std::vector<bool> visited(graph.size(), false);
        std::vector<Neighbor> retset(L + 1, Neighbor(-1, std::numeric_limits<float>::max(), false));
        auto dist = oracle(entry_id, query);
        retset[0] = Neighbor(entry_id, dist, true);
        std::vector<Neighbor> track_pool;
        track_pool.emplace_back(entry_id, dist, true);
        int k = 0;
        while (k < L) {
            int nk = L;
            if (retset[k].flag) {
                retset[k].flag = false;
                int n = retset[k].id;
                for (const auto &candidate: graph[n].candidates_) {
                    int id = candidate.id;
                    if (visited[id]) continue;
                    visited[id] = true;
                    dist = oracle(id, query);
                    if (dist >= retset[L - 1].distance) continue;
                    Neighbor nn(id, dist, true);
                    int r = insert_into_pool(retset.data(), L, nn);
                    track_pool.emplace_back(id, dist, true);
                    if (r < nk) nk = r;
                }
            }
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
        std::sort(track_pool.begin(), track_pool.end());
        return track_pool;
    };

    void saveGraph(Graph &graph,
                   const std::string &filename) {
        std::ofstream file(filename);
        file << std::fixed;
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open the file");
        }
        file << graph.size() << std::endl;
        for (size_t i = 0; i < graph.size(); ++i) {
            std::sort(graph[i].candidates_.begin(), graph[i].candidates_.end());
            file << i << " " << graph[i].candidates_.size() << std::endl;
            for (auto &neighbor: graph[i].candidates_) {
                file << neighbor.id << " " << neighbor.distance << " ";
            }
            file << std::endl;
        }

        file.close();
    }

    void loadGraph(Graph &graph,
                   const std::string &filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            return;
        }
        unsigned rows;
        file >> rows;
        graph.clear();
        graph.reserve(rows);
        for (size_t i = 0; i < rows; ++i) {
            unsigned id, num;
            file >> id >> num;
            Neighborhood neighborhood;
            neighborhood.candidates_.reserve(num);
            for (size_t j = 0; j < num; ++j) {
                unsigned tmp;
                float dist;
                file >> tmp >> dist;
                neighborhood.candidates_.emplace_back(tmp, dist, false);
            }
            graph.push_back(neighborhood);
        }

        file.close();
    }
}

#endif //MERGE_GRAPH_H

#include "evaluator.h"

#include "hnsw.h"
#include "mgraph.h"

#ifndef MULTITHREAD
#define MULTITHREAD 0
#endif

void
calRecall(
    Index& index, DatasetPtr& dataset, unsigned qsize, unsigned L, unsigned K, unsigned runs) {
    auto& query = dataset->getQuery();
    auto& groundTruth = dataset->getGroundTruth();
    float recall = 0;
    double qps = 0;
    for (int x = 0; x < runs; ++x) {
        Timer timer;
        timer.start();
        float local_recall = 0;
#if MULTITHREAD
#pragma omp parallel for reduction(+ : local_recall)
#endif
        for (size_t i = 0; i < qsize; ++i) {
            auto result = index.search(query[i], K, L);
            std::unordered_set<unsigned> gt(groundTruth[i], groundTruth[i] + K);
            size_t correct = 0;
            for (const auto& res : result) {
                if (gt.find(res.id) != gt.end()) {
                    correct++;
                }
            }
            local_recall += static_cast<float>(correct);
        }
        timer.end();
        qps = std::max(qps, (double)qsize / timer.elapsed());
        recall = std::max(local_recall / (static_cast<float>(qsize * K)), recall);
    }
    std::cout << "L: " << L << " recall: " << recall << " qps: " << qps << std::endl;
}

void
calRecall(const IndexPtr& index,
          DatasetPtr& dataset,
          unsigned qsize,
          unsigned L,
          unsigned K,
          unsigned runs) {
    auto& query = dataset->getQuery();
    auto& groundTruth = dataset->getGroundTruth();
    float recall = 0;
    double qps = 0;
    for (int x = 0; x < runs; ++x) {
        Timer timer;
        timer.start();
        float local_recall = 0;
#if MULTITHREAD
#pragma omp parallel for reduction(+ : local_recall)
#endif
        for (size_t i = 0; i < qsize; ++i) {
            auto result = index->search(query[i], K, L);
            std::unordered_set<unsigned> gt(groundTruth[i], groundTruth[i] + K);
            size_t correct = 0;
            for (const auto& res : result) {
                if (gt.find(res.id) != gt.end()) {
                    correct++;
                }
            }
            local_recall += static_cast<float>(correct);
        }
        timer.end();
        qps = std::max(qps, (double)qsize / timer.elapsed());
        recall = std::max(local_recall / (static_cast<float>(qsize * K)), recall);
    }
    std::cout << "L: " << L << " recall: " << recall << " qps: " << qps << std::endl;
}

void
graph::recall(std::variant<std::reference_wrapper<Index>, IndexPtr> index,
              DatasetPtr& dataset,
              int search_L,
              unsigned K,
              unsigned runs) {
    std::vector<int> search_Ls;
    if (search_L < 0) {
        if (K == 100) {
            search_Ls = {
                100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600};
        } else {
            for (int i = 20; i <= 100; i += 5) {
                search_Ls.push_back(i);
            }
            for (int i = 100; i <= 800; i += 10) {
                search_Ls.push_back(i);
            }
        }
    } else {
        search_Ls = {search_L};
    }
    size_t qsize = dataset->getQuery().size();

    for (auto L : search_Ls) {
        if (std::holds_alternative<IndexPtr>(index)) {
            calRecall(std::get<IndexPtr>(index), dataset, qsize, L, K, runs);
        } else {
            calRecall(std::get<std::reference_wrapper<Index> >(index), dataset, qsize, L, K, runs);
        }
    }
}

int
seek(const Neighbors& vec) {
    int left = 0, right = vec.size() - 1;
    if (vec.back().id > 0) {
        return right;
    }
    int result = left;
    while (left <= right) {
        int mid = (left + right) / 2;
        if (vec[mid].id == -1) {
            result = mid;
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    return result;
}

std::pair<Neighbors, int>
searchWithDist(IndexOracle<float>* oracle,
               VisitedListPool* visited_list_pool,
               Graph& graph,
               const float* query,
               int topk,
               int L,
               int entry_id = -1) {
    auto visit_pool_ptr = visited_list_pool->getFreeVisitedList();
    auto visit_list = visit_pool_ptr.get();
    auto* visit_array = visit_list->block_;
    auto visit_tag = visit_list->version_;
    Neighbors retset(L + 1, Neighbor(-1, std::numeric_limits<float>::max(), false));
    if (entry_id == -1) {
        std::vector<int> init_ids(L);
        std::mt19937 rng(2024);
        gen_random(rng, init_ids.data(), L, graph.size());
        for (int i = 0; i < L; ++i) {
            auto dist = (*oracle)(init_ids[i], query);
            retset[i] = Neighbor(init_ids[i], dist, true);
        }
    } else {
        auto dis = (*oracle)(entry_id, query);
        retset[0] = Neighbor(entry_id, dis, true);
    }
    int dist_calc = 0;

    int k = 0;
    while (k < L) {
        int nk = L;
        if (retset[k].flag) {
            retset[k].flag = false;
            auto n = retset[k].id;
            for (const auto& candidate : graph[n].candidates_) {
                auto id = candidate.id;
#ifdef USE_SSE
                _mm_prefetch(visit_array + id, _MM_HINT_T0);
//                _mm_prefetch(&oracle[id], _MM_HINT_T0);
#endif
                if (visit_array[id] == visit_tag)
                    continue;
                visit_array[id] = visit_tag;
                float dis = (*oracle)(id, query);
                ++dist_calc;
                if (dis >= retset[L - 1].distance)
                    continue;
                Neighbor nn(id, dis, true);
                int r = insert_into_pool(retset.data(), L, nn);
                if (r < nk)
                    nk = r;
            }
        }
        if (nk <= k) {
            k = nk;
        } else {
            ++k;
        }
    }
    int real_end = seek(retset);
    retset.resize(std::min(topk, real_end));

    visited_list_pool->releaseVisitedList(visit_pool_ptr);
    return {retset, dist_calc};
}

std::pair<Neighbors, int>
search(HGraph& hgraph, DatasetPtr& dataset, const float* query, int topk, int L, size_t entry_id) {
    unsigned cur_node_ = entry_id;
    int dist_calc = 0;
    for (int i = hgraph.size() - 1; i > 0; --i) {
        auto res = searchWithDist(dataset->getOracle().get(),
                                  dataset->getVisitedListPool().get(),
                                  hgraph[i],
                                  query,
                                  1,
                                  1,
                                  cur_node_);
        cur_node_ = res.first[0].id;
        dist_calc += res.second;
    }
    auto res = searchWithDist(dataset->getOracle().get(),
                              dataset->getVisitedListPool().get(),
                              hgraph[0],
                              query,
                              topk,
                              L,
                              cur_node_);
    dist_calc += res.second;
    return {res.first, dist_calc};
}

void
calDist(const IndexPtr& index,
        DatasetPtr& dataset,
        unsigned qsize,
        unsigned L,
        unsigned K,
        unsigned runs) {
    auto& query = dataset->getQuery();
    auto& groundTruth = dataset->getGroundTruth();

    HGraph hgraph;
    size_t enter_point;
    if (dynamic_cast<hnsw::HNSW*>(index.get())) {
        auto hnsw_index = std::static_pointer_cast<hnsw::HNSW>(index);
        hgraph = hnsw_index->extractHGraph();
        enter_point = hnsw_index->enter_point_;
    } else if (dynamic_cast<MGraph*>(index.get())) {
        auto mgraph_index = std::static_pointer_cast<MGraph>(index);
        hgraph = mgraph_index->extractHGraph();
        enter_point = mgraph_index->enter_point_;
    } else {
        float recall = 0;
        double dist_calc = 0;
        for (int x = 0; x < runs; ++x) {
            float local_recall = 0;
#if MULTITHREAD
#pragma omp parallel for reduction(+ : local_recall)
#endif
            for (size_t i = 0; i < qsize; ++i) {
                auto result = searchWithDist(dataset->getOracle().get(),
                                             dataset->getVisitedListPool().get(),
                                             index->extractGraph(),
                                             query[i],
                                             K,
                                             L);
                std::unordered_set<unsigned> gt(groundTruth[i], groundTruth[i] + K);
                size_t correct = 0;
                for (const auto& res : result.first) {
                    if (gt.find(res.id) != gt.end()) {
                        correct++;
                    }
                }
                local_recall += static_cast<float>(correct);
                dist_calc += result.second;
            }

            recall = std::max(local_recall / (static_cast<float>(qsize * K)), recall);
        }
        dist_calc /= runs * qsize;
        std::cout << std::fixed << std::setprecision(5) << "L: " << L << " recall: " << recall
                  << " dist_calc: " << dist_calc << std::endl;
        return;
    }

    float recall = 0;
    double dist_calc = 0;
    for (int x = 0; x < runs; ++x) {
        float local_recall = 0;
#if MULTITHREAD
#pragma omp parallel for reduction(+ : local_recall)
#endif
        for (size_t i = 0; i < qsize; ++i) {
            auto result = search(hgraph, dataset, query[i], K, L, enter_point);
            std::unordered_set<unsigned> gt(groundTruth[i], groundTruth[i] + K);
            size_t correct = 0;
            for (const auto& res : result.first) {
                if (gt.find(res.id) != gt.end()) {
                    correct++;
                }
            }
            local_recall += static_cast<float>(correct);
            dist_calc += result.second;
        }

        recall = std::max(local_recall / (static_cast<float>(qsize * K)), recall);
    }
    dist_calc /= runs * qsize;
    std::cout << std::fixed << std::setprecision(5) << "L: " << L << " recall: " << recall
              << " dist_calc: " << dist_calc << std::endl;
}

void
graph::dist(std::variant<std::reference_wrapper<Index>, IndexPtr> index,
            DatasetPtr& dataset,
            unsigned search_L,
            unsigned K,
            unsigned runs) {
    std::vector<unsigned> search_Ls;
    if (search_L == -1) {
        search_Ls = {
            20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000};
    } else {
        search_Ls = {search_L};
    }
    size_t qsize = dataset->getQuery().size();

    for (auto L : search_Ls) {
        if (std::holds_alternative<IndexPtr>(index)) {
            calDist(std::get<IndexPtr>(index), dataset, qsize, L, K, runs);
        } else {
            throw std::runtime_error("Not implemented");
        }
    }
}

//void
//graph::calRecall(const Graph& graph,
//                unsigned int K,
//                const Matrix<float>& query,
//                const std::vector<std::vector<unsigned int>>& groundTruth,
//                IndexOracle<float>* oracle,
//                unsigned search_L) {
//    std::vector<unsigned> search_Ls;
//    if (search_L == -1) {
//        search_Ls = {
//            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000};
//    } else {
//        search_Ls = {search_L};
//    }
//    size_t qsize = query.size();
//    size_t total = graph.size();
//
//    FlattenGraph fg(graph);
//
//    for (auto L : search_Ls) {
//        float recall = 0;
//        double qps = 0;
//        auto runs = 5;
//        for (int x = 0; x < runs; ++x) {
//            Timer timer;
//            timer.start();
//            float local_recall = 0;
//            //#pragma omp parallel for reduction(+:local_recall)
//            for (size_t i = 0; i < qsize; ++i) {
//                auto result = search(oracle, fg, query[i], K, L);
//                std::unordered_set<unsigned> gt(groundTruth[i].begin(), groundTruth[i].begin() + K);
//                size_t correct = 0;
//                for (const auto& res : result) {
//                    if (gt.find(res.id) != gt.end()) {
//                        correct++;
//                    }
//                }
//                local_recall += static_cast<float>(correct);
//            }
//            timer.end();
//            qps = std::max(qps, (double)qsize / timer.elapsed());
//            recall = std::max(local_recall / (static_cast<float>(qsize * K)), recall);
//        }
//        std::cout << "L: " << L << " recall: " << recall << " qps: " << qps << std::endl;
//    }
//}
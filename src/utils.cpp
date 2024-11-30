#include "utils.h"

using namespace merge;

std::vector<std::vector<unsigned int>> merge::loadGroundTruth(const std::string &filename,
                                                       unsigned int qsize,
                                                       unsigned int K){
    std::ifstream input(filename, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("Cannot open file " + filename);
    }
    std::vector<std::vector<unsigned>> groundTruth(qsize, std::vector<unsigned>(K));
    int t ;
    input.read((char *) &t, 4);
    std::vector<int> temp(t);
    input.seekg(0, std::ios::beg);
    for (int i = 0; i < qsize; i++) {
        int t;
        input.read((char *) &t, 4);
        input.read((char *) temp.data(), K * 4);
        for (int j = 0; j < K; j++) {
            groundTruth[i][j] = temp[j];
        }
    }
    input.close();
    return groundTruth;
}

void merge::evaluate(const Graph &graph,
              unsigned int K,
              const Matrix &query,
              const std::vector<std::vector<unsigned int>> &groundTruth,
              IndexOracle &oracle,
              unsigned search_L){
    std::vector<unsigned> search_Ls;
    if (search_L == -1) {
        search_Ls = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                     200, 300, 400, 500, 600, 700, 800, 900, 1000};
    } else {
        search_Ls = {search_L};
    }
    size_t qsize = query.size();
    size_t total = graph.size();

    std::vector<int> final_graph;
    std::vector<int> offsets;
    project(graph, final_graph, offsets);

    for (auto L: search_Ls) {
        float recall = 0;
        double qps = 0;
        auto runs = 5;
        for (int x = 0; x < runs; ++x) {
            Timer timer;
            timer.start();
            float local_recall = 0;
//#pragma omp parallel for reduction(+:local_recall)
            for (size_t i = 0; i < qsize; ++i) {
                auto result = search(oracle, final_graph, offsets, query[i], K, total, L);
                std::unordered_set<unsigned> gt(groundTruth[i].begin(), groundTruth[i].begin() + K);
                size_t correct = 0;
                for (const auto &res: result) {
                    if (gt.find(res.id) != gt.end()) {
                        correct++;
                    }
                }
                local_recall += static_cast<float>(correct);
            }
            timer.end();
            qps = std::max(qps, (double) qsize / timer.elapsed());
            recall = std::max(local_recall / (static_cast<float>(qsize * K)), recall);
        }
        std::cout << "L: " << L << " recall: " << recall << " qps: " << qps << std::endl;
    }
}
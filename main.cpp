#include <unordered_set>
#include "src/include/nndescent.h"
#include "src/include/merge.h"

using namespace merge;

std::vector<std::vector<unsigned int>> loadGroundTruth(const std::string &filename,
                       unsigned int qsize,
                       unsigned int K = 100) {
    std::ifstream input(filename, std::ios::binary);
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

void evaluate(const Graph &graph,
        unsigned int K,
        const Matrix &query,
        const std::vector<std::vector<unsigned int>> &groundTruth,
        IndexOracle &oracle,
        unsigned search_L = -1) {
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

void test_nndescent(IndexOracle &oracle,
                    const Matrix &query,
                    const std::vector<std::vector<unsigned int>> &groundTruth,
                    unsigned int K) {
    nndescent::NNDescent nndescent(20);

    auto graph = nndescent.build(oracle);

    evaluate(graph, K, query, groundTruth, oracle);
}

void fgim_nndescent(Matrix &base,
                    const Matrix &query,
                    const std::vector<std::vector<unsigned int>> &groundTruth,
                    unsigned int K) {
    Matrix base_half;
    base.halve(base_half);
    MatrixOracle<metric::l2> oracle_half1(base);
    MatrixOracle<metric::l2> oracle_half2(base_half);

    std::cout << "Preparing subgraphs" << std::endl;
    nndescent::NNDescent nndescent(20);
    auto graph_half_1 = nndescent.build(oracle_half1);
    auto graph_half_2 = nndescent.build(oracle_half2);

    Matrix merged;
    mergeMatrix(base, base_half, merged);

    MatrixOracle<metric::l2> oracle_merged(merged);

    std::cout << "Merging subgraphs" << std::endl;

    Merge merge;
    auto graph = merge.merge(graph_half_1, oracle_half1, graph_half_2, oracle_half2, oracle_merged);

    evaluate(graph, K, query, groundTruth, oracle_merged);
}

int main(){
    // set verbose to true if you want to see more information
    Log::setVerbose(true);
    // Recall@K
    int K = 10;

    std::string base_path = "/root/datasets/sift/1m/sift_base.fvecs";
    std::string query_path = "/root/datasets/sift/1m/sift_query.fvecs";
    std::string groundtruth_path = "/root/datasets/sift/1m/sift_groundtruth.ivecs";

    Matrix base;
    base.load(base_path);

    Matrix query;
    query.load(query_path);

    MatrixOracle<metric::l2> oracle(base);

    auto groundTruth = loadGroundTruth(groundtruth_path, query.size());

//    test_nndescent(oracle, query, groundTruth, K);

    fgim_nndescent(base, query, groundTruth, K);

    return 0;
}


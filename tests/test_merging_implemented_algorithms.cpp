#include "nndescent.h"
#include "vamana.h"
#include "taumng.h"
#include "nsw.h"
#include "fgim.h"
#include "utils.h"

using namespace merge;

std::pair<Graph, Graph> subgraph_nnd(IndexOracle &oracle_half1,
                  IndexOracle &oracle_half2) {
    nndescent::NNDescent nndescent(20);
    auto graph_half_1 = nndescent.build(oracle_half1);
    auto graph_half_2 = nndescent.build(oracle_half2);
    return std::make_pair(graph_half_1, graph_half_2);
}

std::pair<Graph, Graph> subgraph_vamana(IndexOracle &oracle_half1,
                                     IndexOracle &oracle_half2) {
    diskann::Vamana vamana(1.2, 100, 80);
    auto graph_half_1 = vamana.build(oracle_half1);
    auto graph_half_2 = vamana.build(oracle_half2);
    return std::make_pair(graph_half_1, graph_half_2);
}

std::pair<Graph, Graph> subgraph_taumng(IndexOracle &oracle_half1,
                                     IndexOracle &oracle_half2) {
    nndescent::NNDescent nndescent(20);
    auto graph_half_1 = nndescent.build(oracle_half1);
    auto graph_half_2 = nndescent.build(oracle_half2);

    taumng::TauMNG taumng(10, 80, 100);
    taumng.build(graph_half_1, oracle_half1);
    taumng.build(graph_half_2, oracle_half2);

    return std::make_pair(graph_half_1, graph_half_2);
}

std::pair<Graph, Graph> subgraph_nsw(IndexOracle &oracle_half1,
                                     IndexOracle &oracle_half2) {
    nsw::NSW nsw(32, 100);
    auto graph_half_1 = nsw.build(oracle_half1);
    auto graph_half_2 = nsw.build(oracle_half2);
    return std::make_pair(graph_half_1, graph_half_2);
}

void test_fgim(Matrix &base,
                    const Matrix &query,
                    const std::vector<std::vector<unsigned int>> &groundTruth,
                    unsigned int K,
                    unsigned int M0 = 20,
                    unsigned int L = 20,
                    unsigned int M = 80) {
    Matrix base_half;
    base.halve(base_half);
    MatrixOracle<metric::l2> oracle_half1(base);
    MatrixOracle<metric::l2> oracle_half2(base_half);

    std::cout << "Preparing subgraphs (2 parts)" << std::endl;
    /**
     * Change the algorithm to the one you want to test
     * nndescent: subgraph_nnd
     * vamana: subgraph_vamana
     * taumng: subgraph_taumng
     * nsw: subgraph_nsw
     */
    auto [graph_half_1, graph_half_2] = subgraph_nnd(oracle_half1, oracle_half2);

    Matrix merged;
    mergeMatrix(base, base_half, merged);

    MatrixOracle<metric::l2> oracle_merged(merged);

    std::cout << "Merging subgraphs" << std::endl;

    Merge merge(M0, M, L);
    auto graph = merge.merge(graph_half_1, oracle_half1, graph_half_2, oracle_half2, oracle_merged);

    evaluate(graph, K, query, groundTruth, oracle_merged);
}

int main(int argc, char **argv) {
    // set verbose to true if you want to see more information
    Log::setVerbose(false);
    // Recall@K
    int K = 10;

    // Change the path to your dataset
    std::string base_path = "../../datasets/sift/sift_base.fvecs";
    std::string query_path = "../../datasets/sift/sift_query.fvecs";
    std::string groundtruth_path = "../../datasets/sift/sift_groundtruth.ivecs";

    // Load the dataset and groundtruth
    Matrix base;
    base.load(base_path);
    Matrix query;
    query.load(query_path);
    MatrixOracle<metric::l2> oracle(base);
    auto groundTruth = loadGroundTruth(groundtruth_path, query.size());

    // Test FGIM
    if(argc == 1){
        test_fgim(base, query, groundTruth, K);
    }else{
        auto M0 = (unsigned)strtol(argv[1], nullptr, 10);
        auto L = (unsigned)strtol(argv[2], nullptr, 10);
        auto M = (unsigned)strtol(argv[3], nullptr, 10);
        test_fgim(base, query, groundTruth, K, M0, L, M);
    }

    return 0;
}

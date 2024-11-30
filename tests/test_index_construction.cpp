#include "nndescent.h"
#include "vamana.h"
#include "taumng.h"
#include "nsw.h"
#include "utils.h"

void test_nndescent(IndexOracle &oracle,
                    const Matrix &query,
                    const std::vector<std::vector<unsigned int>> &groundTruth,
                    unsigned int K) {
    nndescent::NNDescent nndescent(20);

    auto graph = nndescent.build(oracle);

    evaluate(graph, K, query, groundTruth, oracle);
}

void test_vamana(IndexOracle &oracle,
                 const Matrix &query,
                 const std::vector<std::vector<unsigned int>> &groundTruth,
                 unsigned int K) {
    diskann::Vamana vamana(1.2, 100, 80);

    auto graph = vamana.build(oracle);

    evaluate(graph, K, query, groundTruth, oracle);
}

void test_taumng(IndexOracle &oracle,
                 const Matrix &query,
                 const std::vector<std::vector<unsigned int>> &groundTruth,
                 unsigned int K) {
    nndescent::NNDescent nndescent(20);

    auto graph = nndescent.build(oracle);

    taumng::TauMNG taumng(10, 80, 100);

    taumng.build(graph, oracle);

    evaluate(graph, K, query, groundTruth, oracle);
}

void test_nsw(IndexOracle &oracle,
              const Matrix &query,
              const std::vector<std::vector<unsigned int>> &groundTruth,
              unsigned int K) {
    nsw::NSW nsw(32, 100);

    auto graph = nsw.build(oracle);

    evaluate(graph, K, query, groundTruth, oracle);
}

int main(int argc, char **argv) {
    // set verbose to true if you want to see more information
    Log::setVerbose(false);

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <algorithm>" << std::endl;
        std::cerr << "Algorithm: nndescent, vamana, taumng, nsw" << std::endl;
        return 1;
    }
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

    if (std::string(argv[1]) == "nndescent") {
        test_nndescent(oracle, query, groundTruth, K);
    } else if (std::string(argv[1]) == "vamana") {
        test_vamana(oracle, query, groundTruth, K);
    } else if (std::string(argv[1]) == "taumng") {
        test_taumng(oracle, query, groundTruth, K);
    } else if (std::string(argv[1]) == "nsw") {
        test_nsw(oracle, query, groundTruth, K);
    } else {
        std::cerr << "Unimplemented algorithm: " << argv[1] << std::endl;
        return 1;
    }

    return 0;
}

#include "annslib.h"

void
test_hnsw(DatasetPtr& dataset, int M, int ef_construction, int K = 10) {
    auto index = std::make_shared<hnsw::HNSW>(dataset, M, ef_construction);
    index->build();
    recall(index, dataset, -1, K);
}

void
test_vamana(DatasetPtr& dataset, float alpha, int L, int R, int K = 10) {
    auto index = std::make_shared<diskann::Vamana>(dataset, alpha, L, R);
    index->build();
    recall(index, dataset, -1, K);
}

int
main(int argc, char** argv) {
    // set verbose to true if you want to see more information
    Log::setVerbose(false);

    if (argc < 7) {
        std::cerr << "Usage: " << argv[0]
                  << " <base path> <metric> <query path> <groundtruth path> <topk> "
                     "<algorithm> <algorithm parameters>"
                  << std::endl;
        std::cerr << "Algorithm: HNSW, Vamana" << std::endl;
        std::cerr << "HNSW: M, ef_construction" << std::endl;
        std::cerr << "Vamana: alpha, L, R" << std::endl;
        return 1;
    }

    std::string base_path = argv[1];
    std::string query_path = argv[2];
    std::string groundtruth_path = argv[3];
    std::string metric = argv[4];
    DISTANCE distance;
    if (metric == "l2") {
        distance = DISTANCE::L2;
    } else if (metric == "angular") {
        distance = DISTANCE::COSINE;
    } else if (metric == "jaccard") {
        distance = DISTANCE::JACCARD;
    } else if (metric == "hamming") {
        distance = DISTANCE::HAMMING;
    } else {
        std::cerr << "Unknown metric: " << metric << std::endl;
        return 1;
    }

    auto dataset = Dataset::getInstance(base_path, query_path, groundtruth_path, distance);

    int K = std::stoi(argv[5]);

    std::string algorithm = argv[6];
    std::transform(algorithm.begin(), algorithm.end(), algorithm.begin(), ::tolower);

    if (algorithm == "hnsw") {
        if (argc < 9) {
            std::cerr << "Usage: HNSW <M> <ef_construction>" << std::endl;
            return 1;
        }
        int M = std::stoi(argv[7]);
        int ef_construction = std::stoi(argv[8]);
        test_hnsw(dataset, M, ef_construction, K);
    } else if (algorithm == "vamana") {
        if (argc < 10) {
            std::cerr << "Usage: Vamana <alpha> <L> <R>" << std::endl;
            return 1;
        }
        float alpha = std::stof(argv[7]);
        int L = std::stoi(argv[8]);
        int R = std::stoi(argv[9]);
        test_vamana(dataset, alpha, L, R, K);
    } else {
        std::cerr << "Unimplemented algorithm: " << algorithm << std::endl;
        std::cerr << "TauMNG, NSW, NNDescent are provided in the source code, "
                     "please modify the code to test them"
                  << std::endl;
        return 1;
    }

    return 0;
}

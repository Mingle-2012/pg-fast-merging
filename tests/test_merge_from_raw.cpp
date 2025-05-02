#include "annslib.h"

void
subgraph_hnsw(DatasetPtr& dataset1, DatasetPtr& dataset2, std::vector<IndexPtr>& indexes) {
    auto ind1 = std::make_shared<hnsw::HNSW>(dataset1, 32, 200);
    auto ind2 = std::make_shared<hnsw::HNSW>(dataset2, 32, 200);
    ind1->build();
    ind2->build();

    indexes.push_back(ind1);
    indexes.push_back(ind2);
}

void
subgraph_nnd(DatasetPtr& dataset1, DatasetPtr& dataset2, std::vector<IndexPtr>& indexes) {
    auto ind1 = std::make_shared<nndescent::NNDescent>(dataset1, 20);
    auto ind2 = std::make_shared<nndescent::NNDescent>(dataset2, 20);
    ind1->build();
    ind2->build();

    indexes.push_back(ind1);
    indexes.push_back(ind2);
}

void
subgraph_vamana(DatasetPtr& dataset1, DatasetPtr& dataset2, std::vector<IndexPtr>& indexes) {
    auto ind1 = std::make_shared<diskann::Vamana>(dataset1, 1.2, 100, 80);
    auto ind2 = std::make_shared<diskann::Vamana>(dataset2, 1.2, 100, 80);
    ind1->build();
    ind2->build();

    indexes.push_back(ind1);
    indexes.push_back(ind2);
}

void
subgraph_taumng(DatasetPtr& dataset1, DatasetPtr& dataset2, std::vector<IndexPtr>& indexes) {
    auto ind1 = std::make_shared<nndescent::NNDescent>(dataset1, 20);
    auto ind2 = std::make_shared<nndescent::NNDescent>(dataset2, 20);

    ind1->build();
    ind2->build();

    auto graph_half_1 = ind1->extractGraph();
    auto graph_half_2 = ind2->extractGraph();

    auto ind_1 = std::make_shared<taumng::TauMNG>(dataset1, graph_half_1, 10, 40, 200);
    auto ind_2 = std::make_shared<taumng::TauMNG>(dataset2, graph_half_2, 10, 40, 200);

    ind_1->build();
    ind_2->build();

    indexes.push_back(ind_1);
    indexes.push_back(ind_2);
}

void
subgraph_nsw(DatasetPtr& dataset1, DatasetPtr& dataset2, std::vector<IndexPtr>& indexes) {
    auto ind1 = std::make_shared<nsw::NSW>(dataset1, 32, 200);
    auto ind2 = std::make_shared<nsw::NSW>(dataset2, 32, 200);
    ind1->build();
    ind2->build();

    indexes.push_back(ind1);
    indexes.push_back(ind2);
}

void
test_fgim(DatasetPtr& dataset, unsigned int K) {
    auto datasets = std::vector<DatasetPtr>();
    int num_splits = 2;
    std::cout << "Number of splits: " << num_splits << std::endl;
    dataset->split(datasets, num_splits);
    datasets.insert(datasets.begin(), dataset);

    std::vector<IndexPtr> indexes;
    std::cout << "Preparing subgraphs (2 parts)" << std::endl;
    /**
   * Change the algorithm to the one you want to test
   * hnsw: subgraph_hnsw
   * nndescent: subgraph_nnd
   * vamana: subgraph_vamana
   * taumng: subgraph_taumng
   * nsw: subgraph_nsw
   */
    subgraph_hnsw(datasets[0], datasets[1], indexes);
    // subgraph_nnd(datasets[0], datasets[1], indexes);
    // subgraph_vamana(datasets[0], datasets[1], indexes);
    // subgraph_taumng(datasets[0], datasets[1], indexes);
    // subgraph_nsw(datasets[0], datasets[1], indexes);

    auto mgraph = std::make_shared<MGraph>(K, 200);
    mgraph->Combine(indexes);

    recall(mgraph, mgraph->extractDataset());
}

int
main(int argc, char** argv) {
    // set verbose to true if you want to see more information
    Log::setVerbose(false);

    if (argc < 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <base path> <metric> <query path> <groundtruth path> <M>" << std::endl;
        std::cerr << "Base path: path to the base dataset" << std::endl;
        std::cerr << "Metric: l2, angular" << std::endl;
        std::cerr << "Query path: path to the query dataset" << std::endl;
        std::cerr << "Groundtruth path: path to the groundtruth dataset" << std::endl;
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

    int K = std::stoi(argv[5]);
    auto dataset = Dataset::getInstance(base_path, query_path, groundtruth_path, distance);

    test_fgim(dataset, K);

    return 0;
}

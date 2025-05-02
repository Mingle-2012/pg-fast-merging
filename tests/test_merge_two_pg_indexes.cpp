#include "annslib.h"

int
main(int argc, char** argv) {
    // set verbose to true if you want to see more information
    Log::setVerbose(false);

    if (argc < 8) {
        std::cerr << "Usage: " << argv[0]
                  << " <graph 1> <base path 1> <graph 2> <bath path 2> <M> "
                     "<metric> <output>"
                  << std::endl;
        std::cerr << "graph 1: path to the first graph file" << std::endl;
        std::cerr << "base path 1: path to the first dataset file" << std::endl;
        std::cerr << "graph 2: path to the second graph file" << std::endl;
        std::cerr << "base path 2: path to the second dataset file" << std::endl;
        std::cerr << "k: number of neighbors" << std::endl;
        std::cerr << "metric: l2, angular" << std::endl;
        std::cerr << "output: path to the output graph file" << std::endl;
        return 1;
    }

    std::string graph1 = argv[1];
    std::string dataset1 = argv[2];
    std::string graph2 = argv[3];
    std::string dataset2 = argv[4];
    int k = std::stoi(argv[5]);
    std::string metric = argv[6];
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

    std::string output = argv[7];

    Graph g1, g2;
    loadGraph(g1, graph1);
    loadGraph(g2, graph2);

    auto d1 = Dataset::getInstance(dataset1, distance);
    auto d2 = Dataset::getInstance(dataset2, distance);

    if (d1->getOracle()->dim() != d2->getOracle()->dim()) {
        std::cerr << "Dataset dimension mismatch" << std::endl;
        return 1;
    }

    if (g1.size() != d1->getOracle()->size() || g2.size() != d2->getOracle()->size()) {
        std::cerr << "Graph and dataset size mismatch" << std::endl;
        return 1;
    }

    auto index1 = std::make_shared<IndexWrapper>(d1, g1);
    auto index2 = std::make_shared<IndexWrapper>(d2, g2);

    std::vector<IndexPtr> vec = {index1, index2};
    auto mgraph = std::make_shared<FGIM>(k);
    mgraph->Combine(vec);

    saveGraph(mgraph->extractGraph(), output);

    return 0;
}
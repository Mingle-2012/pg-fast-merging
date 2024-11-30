#include "fgim.h"
#include "utils.h"

using namespace merge;

int main(int argc, char **argv) {
    // set verbose to true if you want to see more information
    Log::setVerbose(false);

    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <graph 1> <dataset 1> <graph 2> <dataset 2> <metric> <output>" << std::endl;
        std::cerr << "Metric: l2, cosine" << std::endl;
        return 1;
    }

    std::string graph1 = argv[1];
    std::string dataset1 = argv[2];
    std::string graph2 = argv[3];
    std::string dataset2 = argv[4];
    std::string output = argv[5];

    Graph g1, g2;
    loadGraph(g1, graph1);
    loadGraph(g2, graph2);

    Matrix d1, d2;
    d1.load(dataset1);
    d2.load(dataset2);
    MatrixOracle<metric::l2> oracle1(d1), oracle2(d2);

    if (d1.dim() != d2.dim()) {
        std::cerr << "Dataset dimension mismatch" << std::endl;
        return 1;
    }

    if (g1.size() != d1.size() || g2.size() != d2.size()) {
        std::cerr << "Graph and dataset size mismatch" << std::endl;
        return 1;
    }

    Matrix merged;
    mergeMatrix(d1, d2, merged);

    MatrixOracle<metric::l2> oracle_merged(merged);

    Merge merge;
    auto graph = merge.merge(g1, oracle1, g2, oracle2, oracle_merged);

    saveGraph(graph, output);

    return 0;
}
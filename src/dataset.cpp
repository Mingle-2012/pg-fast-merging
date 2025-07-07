#include "dataset.h"

std::unordered_set<std::string> angular_datasets = {"deep", "glove", "crawl"};

std::shared_ptr<Dataset>
Dataset::getInstance(const std::string& name, const std::string& size) {
    auto dataset = std::make_shared<Dataset>();
    dataset->name_ = name;
    dataset->size_ = size;

    dataset->load();
    return dataset;
}

std::shared_ptr<Dataset>
Dataset::getInstance(const std::string& base_path, DISTANCE metric) {
    auto dataset = std::make_shared<Dataset>();
    dataset->name_ =
        base_path.substr(base_path.find_last_of("/\\") + 1,
                         base_path.find_last_of('.') - base_path.find_last_of("/\\") - 1);
    dataset->size_ = "custom";

    dataset->base_->load(base_path);

    switch (metric) {
        case DISTANCE::L2:
            dataset->oracle_ = MatrixOracle<float, metric::l2>::getInstance(*dataset->base_);
            break;
        case DISTANCE::COSINE:
            dataset->oracle_ = MatrixOracle<float, metric::angular>::getInstance(*dataset->base_);
            break;
        case DISTANCE::JACCARD:
            throw std::runtime_error("Jaccard distance is currently not supported");
            break;
        case DISTANCE::HAMMING:
            throw std::runtime_error("Hamming distance is currently not supported");
            break;
    }

    dataset->full_dataset_ = true;
    dataset->visited_list_pool_ = VisitedListPool::getInstance(dataset->base_->size());

    return dataset;
}

std::shared_ptr<Dataset>
Dataset::getInstance(const std::string& base_path,
                     const std::string& query_path,
                     const std::string& groundtruth_path,
                     DISTANCE metric) {
    auto dataset = std::make_shared<Dataset>();
    dataset->name_ =
        base_path.substr(base_path.find_last_of("/\\") + 1,
                         base_path.find_last_of('.') - base_path.find_last_of("/\\") - 1);
    dataset->size_ = "custom";

    dataset->base_->load(base_path);
    dataset->query_->load(query_path);
    dataset->groundTruth_->load(groundtruth_path);

    switch (metric) {
        case DISTANCE::L2:
            dataset->oracle_ = MatrixOracle<float, metric::l2>::getInstance(*dataset->base_);
            break;
        case DISTANCE::COSINE:
            dataset->oracle_ = MatrixOracle<float, metric::angular>::getInstance(*dataset->base_);
            break;
        case DISTANCE::JACCARD:
            throw std::runtime_error("Jaccard distance is currently not supported");
            break;
        case DISTANCE::HAMMING:
            throw std::runtime_error("Hamming distance is currently not supported");
            break;
    }

    dataset->full_dataset_ = true;
    dataset->visited_list_pool_ = VisitedListPool::getInstance(dataset->base_->size());

    return dataset;
}

Dataset::Dataset() {
    base_ = std::make_shared<Matrix<float> >();
    query_ = std::make_shared<Matrix<float> >();
    groundTruth_ = std::make_shared<Matrix<int> >();

    distance_ = DISTANCE::L2;
    oracle_ = nullptr;
    full_dataset_ = false;
    visited_list_pool_ = nullptr;
}

void
Dataset::load() {
    std::string base_path, query_path, groundtruth_path;
    if (name_ == "sift") {
        base_path = "/root/datasets/sift/" + size_ + "/sift_base.fvecs";
        if (size_ == "10k") {
            query_path = "/root/datasets/sift/10k/sift_query.fvecs";
        } else {
            query_path = "/root/datasets/sift/1m/sift_query.fvecs";
        }
        groundtruth_path = "/root/datasets/sift/" + size_ + "/sift_groundtruth.ivecs";
    } else if (name_ == "gist") {
        base_path = "/root/datasets/gist/gist_base.fvecs";
        query_path = "/root/datasets/gist/gist_query.fvecs";
        groundtruth_path = "/root/datasets/gist/gist_groundtruth.ivecs";
    } else if (name_ == "deep") {
        base_path = "/root/datasets/deep/1m/deep_base.fvecs";
        query_path = "/root/datasets/deep/1m/deep_query.fvecs";
        groundtruth_path = "/root/datasets/deep/1m/deep_groundtruth.ivecs";
    } else if (name_ == "msong") {
        base_path = "/root/datasets/msong/msong_base.fvecs";
        query_path = "/root/datasets/msong/msong_query.fvecs";
        groundtruth_path = "/root/datasets/msong/msong_groundtruth.ivecs";
    } else if (name_ == "glove") {
        base_path = "/root/datasets/glove/twitter/glove_base_100d.fvecs";
        query_path = "/root/datasets/glove/twitter/glove_query_100d.fvecs";
        groundtruth_path = "/root/datasets/glove/twitter/glove_groundtruth_100d.ivecs";
    } else if (name_ == "crawl") {
        base_path = "/root/datasets/crawl/crawl_base.fvecs";
        query_path = "/root/datasets/crawl/crawl_query.fvecs";
        groundtruth_path = "/root/datasets/crawl/crawl_groundtruth.ivecs";
    } else {
        throw std::runtime_error("Unknown dataset");
    }

    base_->load(base_path);
    query_->load(query_path);
    groundTruth_->load(groundtruth_path);

    if (angular_datasets.find(name_) != angular_datasets.end()) {
        oracle_ = MatrixOracle<float, metric::angular>::getInstance(*base_);
        distance_ = DISTANCE::COSINE;
    } else {
        oracle_ = MatrixOracle<float, metric::l2>::getInstance(*base_);
    }

    full_dataset_ = true;
    visited_list_pool_ = VisitedListPool::getInstance(base_->size());
}

void
Dataset::createOracle() {
    // TODO create oracle based on the distance metric or dataset name
}

Matrix<float>&
Dataset::getBase() const {
    if (base_->empty()) {
        throw std::runtime_error("Base matrix is empty");
    }
    return *base_;
}

Matrix<float>&
Dataset::getQuery() const {
    if (!full_dataset_) {
        throw std::runtime_error("Dataset is not fully loaded");
    }
    if (query_->empty()) {
        throw std::runtime_error("Query matrix is empty");
    }
    return *query_;
}

Matrix<int>&
Dataset::getGroundTruth() const {
    if (!full_dataset_) {
        throw std::runtime_error("Dataset is not fully loaded");
    }
    if (groundTruth_->empty()) {
        throw std::runtime_error("Ground truth matrix is empty");
    }
    return *groundTruth_;
}

OraclePtr&
Dataset::getOracle() {
    return oracle_;
}

VisitedListPoolPtr&
Dataset::getVisitedListPool() {
    return visited_list_pool_;
}

void
Dataset::split(std::vector<DatasetPtr>& datasets, unsigned int num) {
    if (!full_dataset_) {
        throw std::runtime_error("Dataset is not fully loaded");
    }

    bool angular = false;
    if (angular_datasets.find(name_) != angular_datasets.end()) {
        angular = true;
    }

    datasets.reserve(num - 1);
    datasets.resize(num - 1);
    auto matrices = base_->split(num);
    if (angular) {
        oracle_ = MatrixOracle<float, metric::angular>::getInstance(*base_);
    } else {
        oracle_ = MatrixOracle<float, metric::l2>::getInstance(*base_);
    }
    for (unsigned int i = 0; i < num - 1; i++) {
        datasets[i] = std::make_shared<Dataset>();
        datasets[i]->base_ = std::make_shared<Matrix<float> >(matrices[i]);
        if (angular) {
            datasets[i]->oracle_ =
                MatrixOracle<float, metric::angular>::getInstance(*datasets[i]->base_);
        } else {
            datasets[i]->oracle_ =
                MatrixOracle<float, metric::l2>::getInstance(*datasets[i]->base_);
        }
        datasets[i]->full_dataset_ = false;
        datasets[i]->name_ = name_;
        datasets[i]->distance_ = distance_;
        // Is it ok to use the same visited list pool?
        // NOTE: we assume that the datasets are not used concurrently
        datasets[i]->visited_list_pool_ = visited_list_pool_;
    }
}

std::vector<std::shared_ptr<Dataset> >
Dataset::subsets(const unsigned int num) const {
    if (!full_dataset_) {
        throw std::runtime_error("Dataset is not fully loaded");
    }

    bool angular = false;
    if (angular_datasets.find(name_) != angular_datasets.end()) {
        angular = true;
    }
    std::vector<std::shared_ptr<Dataset> > datasets;
    size_t size = oracle_->size() / num;
    size_t remainder = oracle_->size() % num;
    size_t offset = 0;
    for (unsigned int i = 0; i < num; i++) {
        if (i == num - 1) {
            size += remainder;
        }
        auto dataset = std::make_shared<Dataset>();
        dataset->base_ = std::make_shared<Matrix<float> >(*base_, offset, size);
        logger << "Creating subset of " << name_ << " " << i + 1 << "/" << num << " with size "
               << dataset->base_->size() << std::endl;
        if (angular) {
            dataset->oracle_ = MatrixOracle<float, metric::angular>::getInstance(*dataset->base_);
        } else {
            dataset->oracle_ = MatrixOracle<float, metric::l2>::getInstance(*dataset->base_);
        }
        dataset->name_ = name_;
        dataset->size_ = size_;
        dataset->distance_ = distance_;
        dataset->visited_list_pool_ = visited_list_pool_;
        dataset->full_dataset_ = false;
        offset += size;
        datasets.emplace_back(dataset);
    }
    return datasets;
}

void
Dataset::merge(std::vector<DatasetPtr>& datasets) {
    if (!full_dataset_) {
        throw std::runtime_error("Dataset is not the full dataset");
    }
    std::vector<MatrixPtr<float> > matrices;
    for (auto& dataset : datasets) {
        if (distance_ != dataset->distance_) {
            throw std::runtime_error("Cannot merge datasets with different distance metrics");
        }
        if (base_->dim() != dataset->getBase().dim()) {
            throw std::runtime_error("Cannot merge datasets with different dimensions");
        }
        matrices.emplace_back(dataset->getBasePtr());
    }
    base_->append(matrices);
    oracle_->reset(*base_);
}

DatasetPtr
Dataset::aggregate(std::vector<DatasetPtr>& datasets) {
    // TODO support merging datasets without query and ground truth
    if (datasets.empty()) {
        throw std::runtime_error("No dataset to merge");
    }
    DatasetPtr base = datasets[0];
    for (auto& dataset : datasets) {
        if (dataset->getBasePtr()->belong(base->getBase())) {
            throw std::runtime_error("Cannot merge datasets that are subsets of each other");
        }
        if (dataset->full_dataset_) {
            base = dataset;
            break;
        }
    }
    auto distance = base->getDistance();
    std::vector<MatrixPtr<float> > matrices;
    for (const auto& dataset : datasets) {
        matrices.emplace_back(dataset->getBasePtr());
        if (dataset->full_dataset_) {
            continue;
        }
        if (distance != dataset->getDistance()) {
            throw std::runtime_error("Cannot merge datasets with different distance metrics");
        }
        if (base->getBase().dim() != dataset->getBase().dim()) {
            throw std::runtime_error("Cannot merge datasets with different dimensions");
        }
    }

    auto matrixPtr = std::make_shared<Matrix<float> >(matrices);
    auto dataset = std::make_shared<Dataset>();
    dataset->base_ = matrixPtr;
    dataset->distance_ = distance;
    if (distance == DISTANCE::COSINE) {
        dataset->oracle_ = MatrixOracle<float, metric::angular>::getInstance(*matrixPtr);
    } else {
        dataset->oracle_ = MatrixOracle<float, metric::l2>::getInstance(*matrixPtr);
    }
    dataset->full_dataset_ = true;
    dataset->visited_list_pool_ = VisitedListPool::getInstance(matrixPtr->size());
    dataset->query_ = base->query_;
    dataset->groundTruth_ = base->groundTruth_;

    return dataset;
}

DISTANCE&
Dataset::getDistance() {
    return distance_;
}

MatrixPtr<float>&
Dataset::getBasePtr() {
    return base_;
}

std::string&
Dataset::getName() {
    return name_;
}

std::string&
Dataset::getSize() {
    return size_;
}

std::vector<std::vector<unsigned int> >
graph::loadGroundTruth(const std::string& filename, unsigned int qsize, unsigned int K) {
    std::ifstream input(filename, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("Cannot open file " + filename);
    }
    std::vector<std::vector<unsigned> > groundTruth(qsize, std::vector<unsigned>(K));
    int t;
    input.read((char*)&t, 4);
    std::vector<int> temp(t);
    input.seekg(0, std::ios::beg);
    for (int i = 0; i < qsize; i++) {
        int t;
        input.read((char*)&t, 4);
        input.read((char*)temp.data(), K * 4);
        for (int j = 0; j < K; j++) {
            groundTruth[i][j] = temp[j];
        }
    }
    input.close();
    return groundTruth;
}
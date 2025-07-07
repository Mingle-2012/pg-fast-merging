
#include "kmeans.h"

Kmeans::Kmeans(DatasetPtr& dataset, int k)
    : oracle_(dataset->getOracle()), k_(k), pointNumber_((int)dataset->getOracle()->size()) {
    centers_.reserve(k_);
    centers_.resize(k_);
    points_.reserve(pointNumber_);
    points_.resize(pointNumber_);
}

void
Kmeans::Init() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distribution(0, pointNumber_ - 1);
    std::map<int, int> uniqueMap;
    while ((int)uniqueMap.size() < k_) {
        int id = distribution(gen);
        uniqueMap.insert(std::pair<int, int>(id, id));
    }
    auto itMap = uniqueMap.begin();
    for (int i = 0; i < k_; i++) {
        int id = itMap->first;
        centers_[i].id_ = id;
        centers_[i].data_ = new float[oracle_->dim()];
        memcpy(centers_[i].data_, (*oracle_)[id], oracle_->dim() * sizeof(float));
        itMap++;
    }

    for (int i = 0; i < pointNumber_; i++) {
        points_[i].id_ = i;
        points_[i].group_ = NearestCenter(i, 1)[0];
    }
}

std::vector<int>
Kmeans::NearestCenter(int p, int ell) {
    std::vector<std::pair<int, float> > distance;
    for (int k = 0; k < k_; k++) {
        distance.emplace_back(k, (*oracle_)(p, centers_[k].id_));
    }
    std::sort(distance.begin(),
              distance.end(),
              [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                  return a.second < b.second;
              });
    std::vector<int> result;
    for (int i = 0; i < ell; i++) {
        result.push_back(distance[i].first);
    }
    return result;
}

void
Kmeans::Cluster() {
    for (int p = 0; p < pointNumber_; p++) {
        points_[p].group_ = NearestCenter(points_[p].id_, 1)[0];
    }
}

void
Kmeans::Center() {
    std::vector<int> count(k_, 0);

    for (auto& c : centers_) {
        memset(c.data_, 0, oracle_->dim() * sizeof(float));
    }
    for (auto& point : points_) {
        for (int d = 0; d < oracle_->dim(); ++d) {
            centers_[point.group_].data_[d] += (*oracle_)[point.id_][d];
        }
        count[point.group_]++;
    }
    for (int c = 0; c < k_; ++c) {
        for (int d = 0; d < oracle_->dim(); ++d) {
            centers_[c].data_[d] /= (float)std::max(1, count[c]);
        }
    }
}

void
Kmeans::Run() {
    Init();
    std::vector<Point> oldCenter(k_);
    for (auto& c : oldCenter) {
        c.data_ = new float[oracle_->dim()];
    }
    maxIteration_ = 100;
    for (int iteration = 0; iteration < maxIteration_; iteration++) {
        for (int x = 0; x < k_; ++x) {
            memcpy(oldCenter[x].data_, centers_[x].data_, oracle_->dim() * sizeof(float));
        }

        Cluster();
        Center();

        for (int k = 0; k < k_; k++) {
            std::cout << "Center " << k << ": ";
            for (int d = 0; d < 10; d++) {
                std::cout << centers_[k].data_[d] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "Old Center: " << std::endl;
        for (int k = 0; k < k_; k++) {
            std::cout << "Center " << k << ": ";
            for (int d = 0; d < 10; d++) {
                std::cout << oldCenter[k].data_[d] << " ";
            }
            std::cout << std::endl;
        }

        float sum = 0;
        for (int k = 0; k < k_; k++) {
            sum += (*oracle_)(centers_[k].data_, oldCenter[k].data_);
        }
        logger << "iteration " << iteration << " sum " << sum << std::endl;
        if (sum < 0.0001) {
            logger << "Converged after " << iteration << " iterations" << std::endl;
            break;
        }
    }
}

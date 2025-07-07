#include "taumng.h"

taumng::TauMNG::TauMNG(DatasetPtr& dataset, Graph& base, float t, int h, int b)
    : Index(dataset), t_(t), h_(h), b_(b), base_(base) {
}

void
taumng::TauMNG::set_b(int b) {
    this->b_ = b;
}

void
taumng::TauMNG::set_h(int h) {
    this->h_ = h;
}

void
taumng::TauMNG::print_info() const {
    Index::print_info();
    logger << "TauMNG parameters:" << std::endl;
    logger << "  t: " << t_ << std::endl;
    logger << "  h: " << h_ << std::endl;
    logger << "  b: " << b_ << std::endl;
}

//void
//taumng::TauMNG::build() {
//    Timer timer;
//    timer.start();
//
//    //    std::vector<int> final_graph_, offsets;
//    //    project(graph_, final_graph_, offsets);
//
//#pragma omp parallel for schedule(dynamic, 256)
//    for (int u = 0; u < graph_.size(); ++u) {
//        if (u % 10000 == 0) {
//            logger << "Processing " << u << " / " << graph_.size() << std::endl;
//        }
//        auto H_u_ = knn_search(oracle_.get(), graph_, (*oracle_)[u], h_, b_);
//        for (auto& v : H_u_) {
//            if (u == v.id) {
//                continue;
//            }
//            bool exist = false;
//            for (auto& w : graph_[u].candidates_) {
//                if (w.id == v.id) {
//                    exist = true;
//                    break;
//                }
//            }
//            if (exist) {
//                continue;
//            }
//            if (v.distance <= 3 * t_) {
//                graph_[u].addNeighbor(v);
//            } else {
//                bool flag = false;
//                for (auto& w : graph_[u].candidates_) {
//                    auto dist = (*oracle_)(w.id, v.id);
//                    if (dist <= v.distance - 3 * t_) {
//                        flag = true;
//                        break;
//                    }
//                }
//                if (!flag) {
//                    graph_[u].addNeighbor(v);
//                }
//            }
//        }
//    }
//
//    timer.end();
//    logger << "Construction time: " << timer.elapsed() << "s" << std::endl;
//}

void
taumng::TauMNG::build_internal() {
#pragma omp parallel for schedule(dynamic, 256)
    for (int u = 0; u < graph_.size(); ++u) {
        if (u % 10000 == 0) {
            logger << "Processing " << u << " / " << graph_.size() << std::endl;
        }
        auto H_u_ =
            knn_search(oracle_.get(), visited_list_pool_.get(), base_, (*oracle_)[u], h_, b_);
        Neighbors candidates;
        for (auto& v : H_u_) {
            if (u == v.id) {
                continue;
            }
            bool exist = false;
            for (auto& w : graph_[u].candidates_) {
                if (w.id == v.id) {
                    exist = true;
                    break;
                }
            }
            if (exist) {
                continue;
            }
            if (v.distance <= 3 * t_) {
                candidates.emplace_back(v);
            } else {
                bool flag = false;
                for (auto& w : graph_[u].candidates_) {
                    auto dist = (*oracle_)(w.id, v.id);
                    if (dist <= v.distance - 3 * t_) {
                        flag = true;
                        break;
                    }
                }
                if (!flag) {
                    candidates.emplace_back(v);
                }
            }
        }
        graph_[u].candidates_.swap(candidates);
    }
}

//
// Created by XiaoWu on 2024/11/23.
//

#ifndef MYANNS_TIMER_H
#define MYANNS_TIMER_H

#include <chrono>
#include <stdexcept>
#include <string>

namespace graph {
class Timer {
private:
    std::chrono::time_point<std::chrono::steady_clock> _start, _end;
    bool started;

public:
    Timer();

    ~Timer() = default;

    void
    start();

    void
    end();

    [[nodiscard]] double
    elapsed() const;
};
}  // namespace graph

#endif  // MYANNS_TIMER_H

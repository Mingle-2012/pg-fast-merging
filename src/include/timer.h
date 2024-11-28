//
// Created by XiaoWu on 2024/11/23.
//

#ifndef MERGE_TIMER_H
#define MERGE_TIMER_H

#include <chrono>
#include <string>
#include <stdexcept>

namespace merge{
    class Timer {
    private:
        std::chrono::time_point<std::chrono::steady_clock> _start, _end;
        bool started;
    public:
        Timer();

        ~Timer() = default;

        void start();

        void end();

        [[nodiscard]] double elapsed() const;
    };
}

#endif //MERGE_TIMER_H

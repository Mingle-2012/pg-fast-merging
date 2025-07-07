#include "timer.h"

using namespace graph;

Timer::Timer() {
    started = false;
}

void
Timer::start() {
    if (started) {
        throw std::runtime_error("Timer already started");
    }
    started = true;
    _start = std::chrono::steady_clock::now();
}

void
Timer::end() {
    if (!started) {
        throw std::runtime_error("Timer not started");
    }
    started = false;
    _end = std::chrono::steady_clock::now();
}

double
Timer::elapsed() const {
    return std::chrono::duration_cast<std::chrono::duration<double> >(_end - _start).count();
}

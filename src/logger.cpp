#include "logger.h"

using namespace merge;

bool Log::verbose = false;
bool Log::newLine = true;
std::ofstream Log::logFile;
std::mutex Log::mutex;

Log &Log::operator<<(std::ostream &(*func)(std::ostream &)) {
    std::lock_guard<std::mutex> guard(mutex);
    if (verbose) {
        int thread_id = omp_get_thread_num();
        if (newLine) {
            std::cout << "[Thread " << thread_id << "] ";
            newLine = false;
        }
        func(std::cout);
        newLine = true;
    }

    return *this;
}

std::string Log::getTimestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm *tm = std::localtime(&now_time);

    std::ostringstream oss;
    oss << std::put_time(tm, "%Y%m_%d_%H%M%S");
    return oss.str();
}

void Log::redirect(std::string filename) {
    std::string timestamp = getTimestamp();
    if (filename.empty()) {
        filename = "log_" + timestamp + ".txt";
    }
    logFile.open(filename);
    if (!logFile) {
        throw std::runtime_error("Cannot open log file");
        return;
    }
    std::cout.rdbuf(logFile.rdbuf());
}

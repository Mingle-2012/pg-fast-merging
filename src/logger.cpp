#include "logger.h"

using namespace graph;

bool Log::verbose = false;
bool Log::newLine = true;
std::string Log::dir = "/root/code/algotests/myanns/logs";
std::ofstream Log::logFile;
std::mutex Log::mutex;

Log&
Log::operator<<(std::ostream& (*func)(std::ostream&)) {
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

std::string
Log::getTimestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm* tm = std::localtime(&now_time);

    std::ostringstream oss;
    oss << std::put_time(tm, "%Y%m_%d_%H%M%S");
    return oss.str();
}

void
Log::redirect(std::string filename) {
    if (logFile.is_open()) {
        logFile.close();
    }
    if (!std::filesystem::exists(dir)) {
        std::filesystem::create_directory(dir);
    }
    std::string timestamp = getTimestamp();
    if (filename.empty()) {
        filename = dir + "/" + timestamp + ".txt";
    } else {
        filename = dir + "/" + filename + ".log";
    }
    logFile.open(filename, std::ios::app);
    if (!logFile) {
        throw std::runtime_error("Cannot open log file");
    }
    std::cout.rdbuf(logFile.rdbuf());
}

Log::~Log() {
    if (logFile.is_open()) {
        logFile.close();
    }
}

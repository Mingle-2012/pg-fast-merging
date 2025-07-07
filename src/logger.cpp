#include "logger.h"

using namespace graph;

bool Log::verbose = false;
bool Log::newLine = true;
std::string Log::dir = "./logs";
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
Log::redirect(const std::string& filename) {
    if (logFile.is_open()) {
        logFile.close();
    }

    std::filesystem::path p(filename);
    std::filesystem::path finalPath;
    std::string timestamp = getTimestamp();

    if (p.is_absolute()) {
        if (std::filesystem::is_directory(p)) {
            if (!std::filesystem::exists(p)) {
                std::filesystem::create_directories(p);
            }
            finalPath = p / (timestamp + ".log");
        } else if (p.has_filename()) {
            finalPath = p;
            if (finalPath.has_parent_path() && !std::filesystem::exists(finalPath.parent_path())) {
                std::filesystem::create_directories(finalPath.parent_path());
            }
        } else {
            if (!std::filesystem::exists(p)) {
                std::filesystem::create_directories(p);
            }
            finalPath = p / (timestamp + ".log");
        }
    } else {
        std::filesystem::path relativeBaseDir = dir;
        if (p.has_filename()) {
            finalPath = relativeBaseDir / p;
            if (!finalPath.has_extension()) {
                finalPath += ".log";
            }
        } else {
            finalPath = relativeBaseDir / p / (timestamp + ".log");
        }
        if (finalPath.has_parent_path() && !std::filesystem::exists(finalPath.parent_path())) {
            std::filesystem::create_directories(finalPath.parent_path());
        }
    }
    if (std::filesystem::exists(finalPath)) {
        std::string stem = finalPath.stem().string();
        std::string extension = finalPath.extension().string();
        finalPath = finalPath.parent_path() / (stem + "_" + timestamp + extension);
    }
    logFile.open(finalPath, std::ios::app);
    if (!logFile) {
        throw std::runtime_error("Cannot open log file: " + finalPath.string());
    }
    std::cout.rdbuf(logFile.rdbuf());
    logger << "Logging to file: " << finalPath << std::endl;
}

Log::~Log() {
    if (logFile.is_open()) {
        logFile.close();
    }
}

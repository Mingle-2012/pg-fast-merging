//
// Created by XiaoWu on 2024/11/23.
//

#ifndef MYANNS_LOGGER_H
#define MYANNS_LOGGER_H

#include <omp.h>

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>

namespace graph {
class Log {
private:
    static bool verbose;
    static std::string dir;
    static std::ofstream logFile;
    static std::mutex mutex;
    static bool newLine;

    template <typename T>
    bool
    containsNewline(const T& msg) {
        std::ostringstream oss;
        oss << msg;
        return oss.str().find('\n') != std::string::npos;
    }

public:
    Log() = default;

    template <typename T>
    Log&
    operator<<(const T& msg) {
        std::lock_guard<std::mutex> guard(mutex);
        if (verbose) {
            int thread_id = omp_get_thread_num();
            if (newLine) {
                std::cout << "[Thread " << thread_id << "] ";
                newLine = false;
            }
            std::cout << msg;
            if (containsNewline(msg)) {
                newLine = true;
            }
        }

        return *this;
    }

    Log&
    operator<<(std::ostream& (*func)(std::ostream&));

    /**
                   * Set the verbose flag
                   * @param v A boolean value to set whether the output should be printed
                   */
    static void
    setVerbose(bool v) {
        verbose = v;
    }

    /**
                   * Set the directory to store the log files
                   * @param d A string to specify the directory
                   */
    static void
    setDir(std::string& d) {
        dir = d;
    }

    /**
   * Redirect the output to a file
   * @param filename A string to specify the filename, if empty, the default
   * filename will be used. Note that the extension is default to be added as
   * .txt
   */
    static void
    redirect(const std::string& filename = "");

    ~Log();

    static std::string
    getTimestamp();
};

static Log logger;

}  // namespace graph

#endif  // MYANNS_LOGGER_H

//
// Created by XiaoWu on 2024/11/23.
//

#ifndef MERGE_LOGGER_H
#define MERGE_LOGGER_H

#include <string>
#include <fstream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <mutex>
#include <omp.h>

namespace merge {
    class Log {
    private:
        static bool verbose;
        static std::ofstream logFile;
        static std::mutex mutex;
        static bool newLine;

        static std::string getTimestamp();

        template<typename T>
        bool containsNewline(const T &msg){
            std::ostringstream oss;
            oss << msg;
            return oss.str().find('\n') != std::string::npos;
        }

    public:
        Log() = default;

        template<typename T>
        Log &operator<<(const T &msg){
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

        Log &operator<<(std::ostream &(*func)(std::ostream &));

        static void setVerbose(bool v) {
            verbose = v;
        }

        static void redirect(std::string filename = "");

        ~Log() = default;

    };

    static Log logger;
}

#endif //MERGE_LOGGER_H

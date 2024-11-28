cmake_minimum_required(VERSION 3.29)
project(Merge)

SET(CMAKE_CXX_FLAGS "-fpic -w -DHAVE_CXX0X -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0 -lrt -DNDEBUG -march=native -Ofast")

include_directories(src)

add_executable(Merge main.cpp
        src/include/metric.h
        src/include/dtype.h
        src/include/graph.h
        src/include/nndescent.h
        src/include/vamana.h
        src/include/taumng.h
        src/include/nsw.h
        src/include/merge.h
        src/dtype.cpp
        src/graph.cpp
        src/metric.cpp
        src/nndescent.cpp
        src/nsw.cpp
        src/taumng.cpp
        src/vamana.cpp
        src/merge.cpp
        src/include/timer.h
        src/timer.cpp
        src/include/logger.h
        src/logger.cpp)

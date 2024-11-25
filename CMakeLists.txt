cmake_minimum_required(VERSION 3.29)
project(Merge)

set(CMAKE_CXX_STANDARD 11)

add_executable(Merge main.cpp
        metric.h
        dtype.h
        graph.h
        nndescent.h
        vamana.h
        taumng.h
        nsw.h
        merge.h)

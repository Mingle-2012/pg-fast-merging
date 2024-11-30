file(GLOB_RECURSE CPP_SOURCES *.cpp)

add_library(${PROJECT_NAME} ${CPP_SOURCES}
        utils.cpp)
add_library(${PROJECT_NAME}_s STATIC ${CPP_SOURCES})
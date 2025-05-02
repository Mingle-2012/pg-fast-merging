file(GLOB_RECURSE CPP_SOURCES *.cpp)

add_library(${PROJECT_NAME} ${CPP_SOURCES})
add_library(${PROJECT_NAME}_s STATIC ${CPP_SOURCES})

file(GLOB TEST_SOURCES "${CMAKE_SOURCE_DIR}/src/*_test.cpp")

foreach (TEST_SRC ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SRC})
    target_link_libraries(${TEST_NAME} ${PROJECT_NAME})
    target_compile_options(${TEST_NAME} PRIVATE -Wall -Wextra -Werror -Wreturn-type)
endforeach ()

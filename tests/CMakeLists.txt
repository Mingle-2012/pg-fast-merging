file(GLOB TESTS_TEST_SOURCES "${CMAKE_SOURCE_DIR}/tests/test_*.cpp")

foreach (TEST_SRC ${TESTS_TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SRC})
    target_link_libraries(${TEST_NAME} ${PROJECT_NAME})
endforeach ()
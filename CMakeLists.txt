cmake_minimum_required(VERSION 3.5.1)
project(vslam)

include(FetchContent)

set(CMAKE_BUILD_TYPE Release CACHE STRING "Set build type to release")
# Download external libraries
FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG        v1.8.3
)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark tests" FORCE)
FetchContent_MakeAvailable(benchmark)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g")

# External Libraries
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# /include folder
include_directories(${PROJECT_SOURCE_DIR}/include)

add_library(${PROJECT_NAME} SHARED
    src/camera.cpp
    src/cornerdetector.cpp
    src/convolution.cpp
    src/edgedetector.cpp
    src/orb.cpp
)

add_executable(main src/main.cpp)
add_executable(benchmark_test benchmarking/corner_benchmark.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(main ${PROJECT_NAME})
target_link_libraries(benchmark_test ${PROJECT_NAME})
target_link_libraries(benchmark_test benchmark::benchmark)

# enable_testing()
# add_test(NAME benchmark_test COMMAND benchmark_test --benchmark_time_unit=ms)

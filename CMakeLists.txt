cmake_minimum_required(VERSION 3.10)
project(vslam LANGUAGES CUDA CXX)

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

FetchContent_Declare(
  pangolin
  GIT_REPOSITORY https://github.com/stevenlovegrove/Pangolin.git
  GIT_TAG master
)
FetchContent_MakeAvailable(pangolin)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g")

# Set CUDA Standard
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true")

add_definitions(-DUSE_CUDA)

# External Libraries
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
find_package(vpi REQUIRED)
include_directories(${VPI_INCLUDE_DIRS})
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

# TO REMOVE IF NOT USING ZED
find_package(ZED 3 REQUIRED)
include_directories(${ZED_INCLUDE_DIRS})
link_directories(${ZED_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_DIRS})
option(LINK_SHARED_ZED "Link with the ZED SDK shared executable" ON)
if (LINK_SHARED_ZED)
    SET(ZED_LIBS ${ZED_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})
else()
    SET(ZED_LIBS ${ZED_STATIC_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_LIBRARY})
endif()




# /include folder
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include/cuda)

add_library(${PROJECT_NAME} SHARED
    src/bfmatcher.cpp
    src/cuda/processing.cu
    src/calibration.cpp
    src/calibrationsettings.cpp
    src/camera.cpp
    src/convolution.cpp
    src/cornerdetector.cpp
    src/depthtopointcloud.cpp
    src/disparity.cpp
    src/disparitytodepth.cpp
    src/edgedetector.cpp
    src/imageformatconverter.cpp
    src/imagerectifier.cpp
    src/imageresizer.cpp
    src/orb.cpp
    src/save_depth.cpp
    src/stereodisparityparams.cpp
    src/zed_utils.cpp
)

add_executable(main src/bin/main.cpp)
add_executable(sanity_check_camera src/bin/sanity_check_camera.cpp)
add_executable(orb src/bin/run_orb.cpp)
add_executable(disparity_jetson src/bin/run_disparity_jetson_zed.cpp)
add_executable(disparity_image src/bin/run_disparity_image.cpp)
add_executable(run_calibration src/bin/run_calibration.cpp)
add_executable(view_calibration src/bin/view_calibration.cpp)
add_executable(trajectory_viz src/bin/plottrajectory.cpp)
add_executable(pointcloud src/bin/plotpointcloud.cpp)
add_executable(vpi_demo src/bin/vpi_demo.cpp)
add_executable(zed_main src/bin/zed_main.cpp)
add_executable(zed_main_threading src/bin/zed_main_threading.cpp)
add_executable(calculator src/bin/calculator.cpp)
# add_executable(benchmark_test benchmarking/corner_benchmark.cpp)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} vpi ${Pangolin_LIBRARIES} ${CUDA_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${ZED_LIBS})
target_link_libraries(main ${PROJECT_NAME})
target_link_libraries(trajectory_viz ${PROJECT_NAME})
target_link_libraries(pointcloud ${PROJECT_NAME})
target_link_libraries(vpi_demo ${PROJECT_NAME})
target_link_libraries(orb ${PROJECT_NAME})
target_link_libraries(disparity_jetson ${PROJECT_NAME})
target_link_libraries(disparity_image ${PROJECT_NAME})
target_link_libraries(sanity_check_camera ${PROJECT_NAME})
target_link_libraries(run_calibration ${PROJECT_NAME})
target_link_libraries(view_calibration ${PROJECT_NAME})
target_link_libraries(zed_main ${PROJECT_NAME})
target_link_libraries(zed_main_threading ${PROJECT_NAME})
target_link_libraries(calculator ${PROJECT_NAME})
# target_link_libraries(benchmark_test ${PROJECT_NAME})
# target_link_libraries(benchmark_test benchmark::benchmark)

# enable_testing()
# add_test(NAME benchmark_test COMMAND benchmark_test --benchmark_time_unit=ms)

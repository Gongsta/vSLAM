#include <benchmark/benchmark.h>

#include <opencv2/opencv.hpp>

#include "cornerdetector.hpp"
#include "edgedetector.hpp"
#include "convolution.hpp"
#include "types.hpp"


static void BM_OpenCVGaussianBlur(benchmark::State& state) {
  cv::Mat img;
  img = cv::imread("../livingroom.jpg", cv::IMREAD_COLOR);
  cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
  for (auto _ : state) {
    cv::GaussianBlur(img, img, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
  }
}
// Register the function as a benchmark
BENCHMARK(BM_OpenCVGaussianBlur);

static void BM_OpenCVSobelEdgeConvolution(benchmark::State& state) {
  cv::Mat img;
  img = cv::imread("../livingroom.jpg", cv::IMREAD_COLOR);
  cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
  int ddepth = -1;
  int ksize = 3;
  int scale = 1;
  int delta = 0;
  for (auto _ : state) {
    cv::Sobel(img, img, ddepth, 1, 0, ksize, scale, delta, cv::BORDER_DEFAULT);
  }
}
// Register the function as a benchmark
BENCHMARK(BM_OpenCVSobelEdgeConvolution);

static void BM_Convolution(benchmark::State& state) {
  cv::Mat img;
  img = cv::imread("../livingroom.jpg", cv::IMREAD_COLOR);
  cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
  EdgeDetector edge_detector{EdgeDetectorType::kSobelEdgeX};
  for (auto _ : state) {
    convolution::Convolve3x3(kScharrDx, img);
  }
}
// Register the function as a benchmark
BENCHMARK(BM_Convolution);



static void BM_SobelEdgeConvolution(benchmark::State& state) {
  cv::Mat img;
  img = cv::imread("../livingroom.jpg", cv::IMREAD_COLOR);
  cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
  EdgeDetector edge_detector{EdgeDetectorType::kSobelEdgeX};
  for (auto _ : state) {
    img = edge_detector.ConvolveImage(img);
  }
}
// Register the function as a benchmark
BENCHMARK(BM_SobelEdgeConvolution);

static void BM_OpenCVHarrisCornerConvolution(benchmark::State& state) {
  cv::Mat img;
  img = cv::imread("../livingroom.jpg", cv::IMREAD_COLOR);
  cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
  for (auto _ : state) {
    cv::cornerHarris(img, img, 2, 3, 0.04, cv::BORDER_DEFAULT);
  }
}
// Register the function as a benchmark
BENCHMARK(BM_OpenCVHarrisCornerConvolution);


static void BM_HarrisCornerConvolution(benchmark::State& state) {
  cv::Mat img;
  img = cv::imread("../livingroom.jpg", cv::IMREAD_COLOR);
  cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
  CornerDetector corner_detector{CornerDetectorType::kHarrisCorner};
  for (auto _ : state) {
    img = corner_detector.ConvolveImage(img);
  }
}
BENCHMARK(BM_HarrisCornerConvolution);

BENCHMARK_MAIN();

/**
 * @file edgedetector.cpp
 * @author Steven Gong (gong.steven@hotmail.com)
 * @brief EdgeDetector class implementation
 *
 * @copyright MIT License (c) 2023 Steven Gong
 *
 */
#include "edgedetector.hpp"

#include <exception>
#include <iostream>

#include "convolution.hpp"
#include "types.hpp"

Kernel3x3 EdgeDetector::SelectKernel(EdgeDetectorType detector_type) {
  switch (detector_type) {
    case EdgeDetectorType::kScharrEdgeX:
      return kScharrDx;
    case EdgeDetectorType::kScharrEdgeY:
      return kScharrDy;
    case EdgeDetectorType::kSobelEdgeX:
      return kSobelDx;
    case EdgeDetectorType::kSobelEdgeY:
      return kSobelDy;
  }
}

EdgeDetector::EdgeDetector(EdgeDetectorType detector_type)
    : detector_type_{detector_type}, kernel_{SelectKernel(detector_type)} {}

cv::Mat EdgeDetector::ConvolveImage(const cv::Mat& img) const {
  if (img.type() == CV_8UC1) {
    cv::Mat double_img;
    img.convertTo(double_img, CV_64FC1, 1.0 / 255, 0);
    return convolution::Convolve3x3(kernel_, double_img);
  } else if (img.type() != CV_64FC1) {
    throw std::invalid_argument{"Unknown image type, expecting either CV_8UC1 or CV_64FC1"};
  }
  return convolution::Convolve3x3(kernel_, img);
}

std::vector<Position> EdgeDetector::DetectEdges(const cv::Mat& img) const {
  std::vector<Position> positions;
  cv::Mat convolved_img = ConvolveImage(img);
  double* p;
  for (int i = 0; i < convolved_img.rows; i++) {
    p = convolved_img.ptr<double>(i);
    for (int j = 0; j < convolved_img.cols; j++) {
      if (p[j] > 0.5) {  // TODO: Play with threshold
        positions.push_back({i, j});
      }
    }
  }
  return positions;
}

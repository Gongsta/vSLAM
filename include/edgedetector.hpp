/**
 * @file edgedetector.hpp
 * @author Steven Gong (gong.steven@hotmail.com)
 * @brief
 *
 * MIT License
 *
 * Copyright (c) 2023 Steven Gong
 *
 */
#ifndef EDGEDETECTOR_HPP_
#define EDGEDETECTOR_HPP_

#include <opencv2/opencv.hpp>
#include <vector>

#include "types.hpp"

inline const Kernel3x3 kSobelDx((Kernel3x3() << -1, 0, 1, -2, 0, 2, -1, 0, 1).finished());
inline const Kernel3x3 kSobelDy((Kernel3x3() << 1, 2, 1, 0, 0, 0, -1, -2, -1).finished());
inline const Kernel3x3 kScharrDx((Kernel3x3() << -3, 0, 3, -10, 0, 10, -3, 0, 3).finished());
inline const Kernel3x3 kScharrDy((Kernel3x3() << -3, -10, -3, 0, 0, 0, 3, 10, 3).finished());

enum class EdgeDetectorType { kSobelEdgeX, kSobelEdgeY, kScharrEdgeX, kScharrEdgeY };

class EdgeDetector {
 public:
  EdgeDetector(EdgeDetectorType detector_type);
  std::vector<Position> DetectEdges(const cv::Mat& img) const;

  /**
   * @brief Convolve image with edge operators
   *
   * @param img Image in CV_8UC1 or CV_64FC1 format
   * @return cv::Mat
   */
  cv::Mat ConvolveImage(const cv::Mat& img) const;

 private:
  Kernel3x3 SelectKernel(EdgeDetectorType detector_type);

  const EdgeDetectorType detector_type_;
  const Kernel3x3 kernel_;
};

#endif

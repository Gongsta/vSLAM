/**
 * @file cornerdetector.hpp
 * @author Steven Gong (gong.steven@hotmail.com)
 * @brief CornerDetector class
 *
 * @copyright MIT License (c) 2023 Steven Gong
 *
 */
#ifndef CORNERDETECTOR_HPP_
#define CORNERDETECTOR_HPP_

#include <array>
#include <opencv2/core/mat.hpp>

#include "edgedetector.hpp"

enum class CornerDetectorType { kHarrisCorner };

class CornerDetector {
  // Reference: https://www.youtube.com/watch?v=nGya59Je4Bs&ab_channel=CyrillStachniss
 public:
  CornerDetector(CornerDetectorType detector_type);
  cv::Mat ConvolveImage(const cv::Mat& img);
  std::vector<Position> DetectCorners(const cv::Mat& img);

 private:
  CornerDetectorType detector_type_;

  const EdgeDetector x_edge_detector_;
  const EdgeDetector y_edge_detector_;
};

#endif

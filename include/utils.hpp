/**
 * @file utils.hpp
 * @author Steven Gong (gong.steven@hotmail.com)
 * @brief Utility functions
 *
 * @copyright MIT License (c) 2023 Steven Gong
 *
 */
#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <cmath>
#include <opencv2/opencv.hpp>

namespace utils {
inline cv::Mat Square(const cv::Mat& img) {
  double* new_p;
  cv::Mat squared_img(img.rows, img.cols, CV_64FC1);
  for (int i = 0; i < img.rows; i++) {
    new_p = squared_img.ptr<double>(i);
    for (int j = 0; j < img.cols; j++) {
      double p = img.at<double>(i, j);
      new_p[j] = p*p;
    }
  }
  return squared_img;
}

inline std::vector<double> SolveQuadratic(double a, double b, double c) {
  // std::cout << "quadratic "  << a << " " << b << " " << c << std::endl;
  std::vector<double> roots;
  double discriminant = b * b - 4.0 * a * c;
  if (discriminant > 0) {
    roots.push_back((-b + std::sqrt(discriminant)) / (2.0 * a));
    roots.push_back((-b - std::sqrt(discriminant)) / (2.0 * a));
  } else {
    return roots;
  }
  return roots;
}

}  // namespace utils

#endif

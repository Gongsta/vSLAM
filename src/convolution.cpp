/**
 * @file convolution.cpp
 * @author Steven Gong (gong.steven@hotmail.com)
 * @brief
 * @version 0.1
 * @date 2023-10-30
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "convolution.hpp"

#include <cmath>
#include <iostream>

namespace convolution {

cv::Mat Convolve3x3(const Kernel3x3& kernel, const cv::Mat& img) {
  cv::Mat convolved_img(img.rows, img.cols, CV_64FC1);
  for (int i = 3 / 2; i < img.rows - 3 / 2; i++) {
    for (int j = 3 / 2; j < img.cols - 3 / 2; j++) {
      convolved_img.at<double>(i, j) = convolution::Convolve3x3Patch(kernel, img, i, j);
    }
  }
  return convolved_img;
}

double Convolve3x3Patch(const Kernel3x3& kernel, const cv::Mat& img, const int row, const int col) {
  double val = 0;
  for (int i = -1; i <= 1; i++) {
    for (int j = -1; j <= 1; j++) {
      val += kernel(i + 1, j + 1) * img.at<double>(row + i, col + j);
    }
  }
  return val;
}

}  // namespace convolution

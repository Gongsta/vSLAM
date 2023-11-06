/**
 * @file convolution.hpp
 * @author Steven Gong (gong.steven@hotmail.com)
 * @brief
 *
 * @copyright MIT License (c) 2023 Steven Gong
 *
 */
#ifndef CONVOLUTION_HPP_
#define CONVOLUTION_HPP_

#include <opencv2/opencv.hpp>

#include "types.hpp"
#include "utils.hpp"

namespace convolution {

/**
 * @brief 3x3 Convolution on an entire image
 *
 * @param kernel 3x3 kernel
 * @param img grasyscale image
 * @return convolved grayscale image
 */
cv::Mat Convolve3x3(const Kernel3x3& kernel, const cv::Mat& img);

/**
 * @brief 3x3 convolution of image area centered at (row, col)
 *
 * @param kernel 3x3 kernel
 * @param img grayscale image
 * @param row row index
 * @param col column index
 * @return convolved value
 */
double Convolve3x3Patch(const Kernel3x3& kernel, const cv::Mat& img, int row, int col);
}  // namespace convolution

#endif

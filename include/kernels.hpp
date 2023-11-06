/**
 * @file kernels.hpp
 * @author Steven Gong (gong.steven@hotmail.com)
 * @brief Kernels
 * @version 0.1
 * @date 2023-10-30
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef IMAGEFILTER_HPP_
#define IMAGEFILTER_HPP_
#include <opencv2/opencv.hpp>

#include "types.hpp"

namespace convolution {
inline const Kernel3x3 kBoxKernel(
    (Kernel3x3() << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0).finished());
inline const Kernel3x3 kBoxKernelNormalized((Kernel3x3() << 1.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9,
                                             1.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9)
                                                .finished());

}  // namespace convolution

#endif

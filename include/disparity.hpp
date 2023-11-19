#ifndef DISPARITY_HPP_
#define DISPARITY_HPP_

#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/Rescale.h>
#include <vpi/algo/StereoDisparity.h>

#include <cstring>  // for memset
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <vpi/OpenCVInterop.hpp>

#include "vpi_utils.hpp"

class DisparityEstimator {
  // Disparity is optimized to run on multiple backends
 public:
  VPIStream& stream;

  // VPI
  VPIImage in_left = NULL;
  VPIImage in_right = NULL;
  VPIImage tmp_left = NULL;
  VPIImage tmp_right = NULL;
  VPIImage stereo_left = NULL;
  VPIImage stereo_right = NULL;
  VPIImage disparity = NULL;
  VPIImage confidence_map = NULL;
  VPIPayload stereo = NULL;

  VPIConvertImageFormatParams conv_params;
  VPIStereoDisparityEstimatorCreationParams stereo_params;

  uint64_t backends;

  DisparityEstimator(cv::Mat& cv_img_in, VPIStream& stream, uint64_t backends = VPI_BACKEND_CUDA);
  ~DisparityEstimator();
  void ProcessFrame(cv::Mat& cv_img_left, cv::Mat& cv_img_right, cv::Mat& cv_disparity_color,
                    cv::Mat& cv_confidence);
};

#endif

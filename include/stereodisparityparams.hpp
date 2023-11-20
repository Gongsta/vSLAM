#ifndef STEREODISPARITYPARAMS_HPP_
#define STEREODISPARITYPARAMS_HPP_

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

class StereoDisparityParams {
 public:
  VPIImageFormat stereo_format = VPI_IMAGE_FORMAT_Y16_ER;
  VPIImageFormat disparity_format = VPI_IMAGE_FORMAT_S16;
  int input_width = 480;
  int input_height = 270;
  int output_width = 480;
  int output_height = 270;
  bool use_confidence_map = false;

  VPIConvertImageFormatParams conv_params;
  VPIStereoDisparityEstimatorCreationParams stereo_params;
  uint64_t backends;

  StereoDisparityParams(const uint64_t backends);
};

#endif
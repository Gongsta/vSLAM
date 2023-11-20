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

#include "stereodisparityparams.hpp"
#include "vpi_utils.hpp"

class DisparityEstimator {
  // Disparity is optimized to run on multiple backends
 public:
  // VPI
  VPIImage disparity = NULL;
  VPIImage confidence_map = NULL;
  VPIPayload stereo = NULL;

  StereoDisparityParams params;

  DisparityEstimator(StereoDisparityParams params);
  ~DisparityEstimator();
  void Apply(VPIStream& stream, VPIImage& stereo_left, VPIImage& stereo_right,
             cv::Mat& cv_disparity_color, cv::Mat& cv_confidence);
};

#endif

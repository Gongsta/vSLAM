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
#include <utility>
#include <vpi/OpenCVInterop.hpp>

#include "imageformatconverter.hpp"
#include "imageresizer.hpp"
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

  ImageFormatConverter* left_converter = nullptr;
  ImageFormatConverter* right_converter = nullptr;
  ImageResizer* left_resizer = nullptr;
  ImageResizer* right_resizer = nullptr;

  // Whether to threshold the disparity
  bool threshold = false;

  DisparityEstimator(StereoDisparityParams params, bool threshold = false);
  ~DisparityEstimator();
  // Feed in preprocesssed VPIImage (grayscale and resized)
  std::pair<VPIImage&, VPIImage&> Apply(VPIStream& stream, VPIImage& left_img_rect_gray_resize,
                                        VPIImage& right_img_rect_gray_resize,
                                        cv::Mat& cv_disparity_color, cv::Mat& cv_confidence);
  // Feed in a cv::Mat. Will convert to VPIImage and apply preprocessing (grayscale and resize)
  std::pair<VPIImage&, VPIImage&> Apply(VPIStream& stream, cv::Mat& cv_img_left,
                                        cv::Mat& cv_img_right, cv::Mat& cv_disparity_color,
                                        cv::Mat& cv_confidence);
};

#endif

#ifndef DISPARITYTODEPTH_HPP_
#define DISPARITYTODEPTH_HPP_

#include <cuda_runtime.h>
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

#include "cudaimage.hpp"
#include "vpi_utils.hpp"

class DisparityToDepthConverter {
  // Disparity is optimized to run on multiple backends
 public:
  // VPI
  VPIImage depth_map;

  DisparityToDepthConverter(int width, int height, VPIImageFormat format);
  ~DisparityToDepthConverter();
  void ComputeDepth(cudaStream_t& stream, VPIImage& disparity_map);
  VPIImage& Apply(cudaStream_t& stream, VPIImage& disparity_map, cv::Mat& cv_depth_map);
};

#endif

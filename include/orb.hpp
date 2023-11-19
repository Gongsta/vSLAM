#ifndef ORB_HPP_
#define ORB_HPP_

#include <vpi/Array.h>
#include <vpi/Image.h>
#include <vpi/Pyramid.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/GaussianPyramid.h>
#include <vpi/algo/ImageFlip.h>
#include <vpi/algo/ORB.h>

#include <opencv2/opencv.hpp>
#include <vpi/OpenCVInterop.hpp>

#include "vpi_utils.hpp"

class ORB {
 public:
  // OpenCV
  std::vector<cv::KeyPoint> keypoints_one;
  std::vector<cv::KeyPoint> keypoints_two;
  cv::Mat descriptors_one;
  cv::Mat descriptors_two;

  // VPI
  VPIImage img_in, img_gray;
  VPIPyramid pyr_input;
  VPIArray keypoints, descriptors;
  VPIPayload orb_payload;
  VPIStream stream;
  VPIORBParams orb_params;
  bool create_stream = false;
  VPIBackend backend = VPI_BACKEND_CUDA;
  
  // ORB Params
  int32_t pyramid_levels;
  float pyramid_scales;



  ORB(cv::Mat cv_img_in, VPIStream stream);
  ~ORB();
  void ProcessFrame(cv::Mat& cv_img_in, cv::Mat& cv_img_out);
};

#endif

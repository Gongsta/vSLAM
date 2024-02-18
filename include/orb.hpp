/**
 * @file orb.hpp
 * @author Steven Gong (gong.steven@hotmail.com)
 * @brief ORB Feature Detector class
 *
 * @copyright MIT License (c) 2023 Steven Gong
 *
 */
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

class ORBFeatureDetector {
 public:
  VPIStream& stream;
  uint64_t backends = VPI_BACKEND_CUDA;

  // OpenCV
  std::vector<cv::KeyPoint> keypoints_one;
  std::vector<cv::KeyPoint> keypoints_two;
  cv::Mat descriptors_one;
  cv::Mat descriptors_two;

  // VPI
  VPIImage img_in = NULL;
  VPIImage img_gray = NULL;
  VPIPyramid pyr_input = NULL;
  VPIArray keypoints = NULL;
  VPIArray descriptors = NULL;
  VPIPayload orb_payload = NULL;
  VPIArrayData out_keypoints_data;
  VPIArrayData out_descriptors_data;

  // ORB Params
  VPIORBParams orb_params;
  int32_t pyramid_levels;
  float pyramid_scales;
  int out_capacity;
  int buf_capacity;

  ORBFeatureDetector(cv::Mat& cv_img_in, VPIStream& stream, uint64_t backends);
  ~ORBFeatureDetector();
  /**
   * @brief Takes in two regular images and outputs the keypoints and descriptors. Good for quick
   * demo purposes, but not for actual use.
   *
   * @param cv_img_in
   * @param cv_img_out
   */

  std::pair<VPIArray&, VPIArray&> Apply(cv::Mat& cv_img_in, cv::Mat& cv_img_out);
  std::pair<VPIArray&, VPIArray&> Apply(cv::Mat& cv_img_in, std::vector<cv::KeyPoint>& cvkeypoints);
  std::pair<VPIArray&, VPIArray&> Apply(cv::Mat& cv_img_in, cv::Mat& cv_img_out,
                                        std::vector<cv::KeyPoint>& cvkeypoints);
};

#endif

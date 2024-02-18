#ifndef BFMATCHER_HPP_
#define BFMATCHER_HPP_

#include <vpi/Array.h>
#include <vpi/Image.h>
#include <vpi/Pyramid.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/BruteForceMatcher.h>

#include <opencv2/opencv.hpp>
#include <vpi/OpenCVInterop.hpp>

#include "vpi_utils.hpp"

class BruteForceMatcher {
 public:
  VPIStream& stream;
  uint64_t backends = VPI_BACKEND_CUDA;
  VPIArray matches;
  int capacity;  // From ORBFeatureDetector, see orb.cpp

  BruteForceMatcher(VPIStream& stream, int capacity);
  ~BruteForceMatcher();

  VPIArray& Apply(VPIArray& query_descriptors, cv::Mat& queryImage,
                  std::vector<cv::KeyPoint>& queryKeypoints, VPIArray& reference_descriptors,
                  cv::Mat& referenceImage, std::vector<cv::KeyPoint>& referenceKeypoints,
                  cv::Mat& cv_img_out, std::vector<cv::DMatch>& cvMatches);
};
#endif

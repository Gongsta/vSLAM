#include "bfmatcher.hpp"

#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>

static std::vector<cv::DMatch> ConvertMatches(VPIMatches* outMatches,
                                              std::vector<cv::KeyPoint>& queryKeypoints) {
  std::vector<cv::DMatch> cvMatches;
  for (int i = 0; i < queryKeypoints.size(); i++) {
    if (outMatches[i].refIndex[0] == -1) {  // Match not found
      continue;
    }
    if (outMatches[i].distance[0] > 20) {  // Distance threshold
      continue;
    }
    cvMatches.push_back(cv::DMatch(outMatches[i].refIndex[0], i, outMatches[i].distance[0]));
  }
  return cvMatches;
}

static cv::Mat DrawMatches(std::vector<cv::DMatch>& cvMatches, int capacity, cv::Mat& queryImage,
                           std::vector<cv::KeyPoint>& queryKeypoints, cv::Mat& referenceImage,
                           std::vector<cv::KeyPoint>& referenceKeypoints) {
  // Draw matches
  cv::Mat outputImage;
  cv::drawMatches(referenceImage, referenceKeypoints, queryImage, queryKeypoints, cvMatches,
                  outputImage);

  // Display the matched image
  return outputImage;
}

BruteForceMatcher::BruteForceMatcher(VPIStream& stream, int capacity)
    : stream{stream}, capacity{capacity} {
  vpiArrayCreate(capacity, VPI_ARRAY_TYPE_MATCHES, 0, &matches);
}

BruteForceMatcher::~BruteForceMatcher() { vpiArrayDestroy(matches); }

VPIArray& BruteForceMatcher::Apply(VPIArray& query_descriptors, cv::Mat& queryImage,
                                   std::vector<cv::KeyPoint>& queryKeypoints,
                                   VPIArray& reference_descriptors, cv::Mat& referenceImage,
                                   std::vector<cv::KeyPoint>& referenceKeypoints,
                                   cv::Mat& cv_img_out, std::vector<cv::DMatch>& cvMatches) {
  CHECK_STATUS(vpiSubmitBruteForceMatcher(stream, VPI_BACKEND_CUDA, query_descriptors,
                                          reference_descriptors, VPI_NORM_HAMMING, 1, matches,
                                          VPI_ENABLE_CROSS_CHECK));
  CHECK_STATUS(vpiStreamSync(stream));  // Sync to ensure the matches are ready
  VPIArrayData out_matches_data;
  CHECK_STATUS(
      vpiArrayLockData(matches, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &out_matches_data));
  VPIMatches* outMatches = (VPIMatches*)out_matches_data.buffer.aos.data;
  cvMatches = ConvertMatches(outMatches, queryKeypoints);
  cv_img_out = DrawMatches(cvMatches, capacity, queryImage, queryKeypoints, referenceImage,
                           referenceKeypoints);
  CHECK_STATUS(vpiArrayUnlock(matches));

  return matches;
}

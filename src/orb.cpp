#include "orb.hpp"

#include <bitset>
#include <numeric>

static std::vector<cv::KeyPoint> GenerateCVKeypoints(VPIKeypointF32* kpts, int numKeypoints) {
  std::vector<cv::KeyPoint> keypoints;
  for (int i = 0; i < numKeypoints; i++) {
    keypoints.push_back(cv::KeyPoint(kpts[i].x, kpts[i].y, 3));
  }
  return keypoints;
}

static cv::Mat DrawKeypoints(cv::Mat img, std::vector<cv::KeyPoint> keypoints,
                             VPIBriefDescriptor* descs, int numKeypoints) {
  cv::Mat out;
  img.convertTo(out, CV_8UC1);
  cvtColor(out, out, cv::COLOR_GRAY2BGR);

  std::vector<int> distances(numKeypoints, 0);

  float maxDist = 0.f;

  for (int i = 0; i < numKeypoints; i++) {
    for (int j = 0; j < VPI_BRIEF_DESCRIPTOR_ARRAY_LENGTH; j++) {
      distances[i] += std::bitset<8 * sizeof(uint8_t)>(descs[i].data[j] ^ descs[0].data[j]).count();
    }
    if (distances[i] > maxDist) {
      maxDist = distances[i];
    }
  }

  uint8_t ids[256];
  std::iota(&ids[0], &ids[0] + 256, 0);
  cv::Mat idsMat(256, 1, CV_8UC1, ids);

  cv::Mat cmap;
  applyColorMap(idsMat, cmap, cv::COLORMAP_JET);

  // cv::drawKeypoints(out, keypoints, out, cv::Scalar::all(-1),
  //                   cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
  for (int i = 0; i < keypoints.size(); i++) {
    int cmapIdx = static_cast<int>(std::round((distances[i] / maxDist) * 255));

    circle(out, keypoints[i].pt, 3, cmap.at<cv::Vec3b>(cmapIdx, 0), -1);
  }

  return out;
}

// Old function, not used
static cv::Mat DrawKeypoints(cv::Mat img, VPIKeypointF32* kpts, VPIBriefDescriptor* descs,
                             int numKeypoints) {
  cv::Mat out;
  img.convertTo(out, CV_8UC1);
  cvtColor(out, out, cv::COLOR_GRAY2BGR);

  if (numKeypoints == 0) {
    return out;
  }

  std::vector<int> distances(numKeypoints, 0);
  float maxDist = 0.f;

  for (int i = 0; i < numKeypoints; i++) {
    for (int j = 0; j < VPI_BRIEF_DESCRIPTOR_ARRAY_LENGTH; j++) {
      distances[i] += std::bitset<8 * sizeof(uint8_t)>(descs[i].data[j] ^ descs[0].data[j]).count();
    }
    if (distances[i] > maxDist) {
      maxDist = distances[i];
    }
  }

  uint8_t ids[256];
  std::iota(&ids[0], &ids[0] + 256, 0);
  cv::Mat idsMat(256, 1, CV_8UC1, ids);

  cv::Mat cmap;
  applyColorMap(idsMat, cmap, cv::COLORMAP_JET);

  for (int i = 0; i < numKeypoints; i++) {
    int cmapIdx = static_cast<int>(std::round((distances[i] / maxDist) * 255));

    circle(out, cv::Point(kpts[i].x, kpts[i].y), 3, cmap.at<cv::Vec3b>(cmapIdx, 0), -1);
  }

  return out;
}

ORBFeatureDetector::ORBFeatureDetector(cv::Mat& cv_img_in, VPIStream& stream, uint64_t backends)
    : stream{stream}, backends{backends} {
  // Needs an image to know the dimension to allocate

  // Configure ORB parameters
  CHECK_STATUS(vpiInitORBParams(&orb_params));
  orb_params.fastParams.intensityThreshold = 142;
  orb_params.maxFeaturesPerLevel = 88;
  orb_params.maxPyramidLevels = 3;

  // For the output arrays capacity we can use the maximum number of features per level multiplied
  // by the maximum number of pyramid levels, this will be the de factor maximum for all levels of
  // the input.
  out_capacity = orb_params.maxFeaturesPerLevel * orb_params.maxPyramidLevels;
  // For the internal buffers capacity we can use the maximum number of features per level
  // multiplied by 20.
  // This will make FAST find a large number of corners so then ORB can select the top N corners in
  // accordance to Harris score of each corner, where N = maximum number of features per level.
  buf_capacity = orb_params.maxFeaturesPerLevel * 20;
  const uint64_t backendsWithCPU = backends | VPI_BACKEND_CPU;

  // ================
  // Allocate Memory for ORB
  CHECK_STATUS(vpiImageCreate(cv_img_in.cols, cv_img_in.rows, VPI_IMAGE_FORMAT_U8, 0, &img_gray));
  CHECK_STATUS(
      vpiArrayCreate(out_capacity, VPI_ARRAY_TYPE_KEYPOINT_F32, backendsWithCPU, &keypoints));
  CHECK_STATUS(
      vpiArrayCreate(out_capacity, VPI_ARRAY_TYPE_BRIEF_DESCRIPTOR, backendsWithCPU, &descriptors));
  CHECK_STATUS(vpiCreateORBFeatureDetector(backends, buf_capacity, &orb_payload));
  CHECK_STATUS(vpiPyramidCreate(cv_img_in.cols, cv_img_in.rows, VPI_IMAGE_FORMAT_U8,
                                orb_params.maxPyramidLevels, 0.5, backends, &pyr_input));
}

ORBFeatureDetector::~ORBFeatureDetector() {
  vpiImageDestroy(img_gray);
  vpiArrayDestroy(keypoints);
  vpiArrayDestroy(descriptors);
  vpiPayloadDestroy(orb_payload);
}

std::pair<VPIArray&, VPIArray&> ORBFeatureDetector::Apply(cv::Mat& cv_img_in, cv::Mat& cv_img_out) {
  std::vector<cv::KeyPoint> cvkeypoints;
  return Apply(cv_img_in, cv_img_out, cvkeypoints);
}

std::pair<VPIArray&, VPIArray&> ORBFeatureDetector::Apply(cv::Mat& cv_img_in,
                                                          std::vector<cv::KeyPoint>& cvkeypoints) {
  cv::Mat cv_img_out;
  return Apply(cv_img_in, cv_img_out, cvkeypoints);
}

std::pair<VPIArray&, VPIArray&> ORBFeatureDetector::Apply(cv::Mat& cv_img_in, cv::Mat& cv_img_out,
                                                          std::vector<cv::KeyPoint>& cvkeypoints) {
  CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(cv_img_in, 0, &img_in));

  // ================
  // Processing stage

  // Convert img to grayscale
  CHECK_STATUS(vpiSubmitConvertImageFormat(stream, backends, img_in, img_gray, NULL));

  // Create the Gaussian Pyramid for the image and wait for the execution
  CHECK_STATUS(
      vpiSubmitGaussianPyramidGenerator(stream, backends, img_gray, pyr_input, VPI_BORDER_CLAMP));

  // Get ORB features and wait for the execution to finish
  CHECK_STATUS(vpiSubmitORBFeatureDetector(stream, backends, orb_payload, pyr_input, keypoints,
                                           descriptors, &orb_params, VPI_BORDER_LIMITED));

  CHECK_STATUS(vpiStreamSync(stream));

  // =======================================
  // Output processing

  // Lock output keypoints and scores to retrieve its data on cpu memory
  VPIImageData img_data;
  CHECK_STATUS(
      vpiArrayLockData(keypoints, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &out_keypoints_data));
  CHECK_STATUS(vpiArrayLockData(descriptors, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS,
                                &out_descriptors_data));
  CHECK_STATUS(
      vpiImageLockData(img_gray, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &img_data));

  VPIKeypointF32* outKeypoints = (VPIKeypointF32*)out_keypoints_data.buffer.aos.data;
  VPIBriefDescriptor* outDescriptors = (VPIBriefDescriptor*)out_descriptors_data.buffer.aos.data;

  cv::Mat img;
  CHECK_STATUS(vpiImageDataExportOpenCVMat(img_data, &img));

  cvkeypoints = GenerateCVKeypoints(outKeypoints, *out_keypoints_data.buffer.aos.sizePointer);
  cv_img_out =
      DrawKeypoints(img, cvkeypoints, outDescriptors, *out_keypoints_data.buffer.aos.sizePointer);

  // Done handling outputs, don't forget to unlock them.
  CHECK_STATUS(vpiImageUnlock(img_gray));
  CHECK_STATUS(vpiArrayUnlock(keypoints));
  CHECK_STATUS(vpiArrayUnlock(descriptors));

  // Destroy image to avoid memory leaks
  vpiImageDestroy(img_in);
  return std::pair<VPIArray&, VPIArray&>(keypoints, descriptors);
}

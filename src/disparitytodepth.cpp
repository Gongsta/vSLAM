#include "disparitytodepth.hpp"

#include "cuda/processing.hpp"

DisparityToDepthConverter::DisparityToDepthConverter(int width, int height, float fx,
                                                     float baseline, VPIImageFormat format)
    : fx{fx}, baseline{baseline} {
  CHECK_STATUS(vpiImageCreate(width, height, format, 0, &depth_map));
}

DisparityToDepthConverter::~DisparityToDepthConverter() { vpiImageDestroy(depth_map); }

void DisparityToDepthConverter::ComputeDepth(cudaStream_t& stream, VPIImage& disparity_map) {
  CUDAImage<float> cuda_disparity_map{disparity_map};
  CUDAImage<float> cuda_depth_map{depth_map};

  ComputeDisparityToDepth(stream, cuda_disparity_map.data, cuda_depth_map.data,
                          cuda_disparity_map.width, cuda_disparity_map.height, 1.0, 0.1);
}

VPIImage& DisparityToDepthConverter::Apply(cudaStream_t& stream, VPIImage& disparity_map,
                                           cv::Mat& cv_depth_map) {
  ComputeDepth(stream, disparity_map);

  VPIImageData data;
  CHECK_STATUS(
      vpiImageLockData(depth_map, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &data));

  // Make an OpenCV matrix out of this image
  cv::Mat cv_depth;
  CHECK_STATUS(vpiImageDataExportOpenCVMat(data, &cv_depth));

  // Scale result
  cv_depth.convertTo(cv_depth, CV_8UC1, 255.0, 0);

  // Apply TURBO colormap to turn the disparities into color, reddish hues
  // represent objects closer to the camera, blueish are farther away.
  cv::applyColorMap(cv_depth, cv_depth_map, cv::COLORMAP_TURBO);

  // Done handling output, don't forget to unlock it.
  CHECK_STATUS(vpiImageUnlock(depth_map));
  return depth_map;
}

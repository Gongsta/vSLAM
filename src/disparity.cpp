#include "disparity.hpp"

DisparityEstimator::DisparityEstimator(StereoDisparityParams params): params{params} {
  // =================================
  // Allocate all VPI resources needed
  // Override some backend-dependent parameters, see
  // https://docs.nvidia.com/vpi/algo_stereo_disparity.html

  // Create the payload for Stereo Disparity algorithm.
  // Payload is created before the image objects so that non-supported backends can be trapped
  // with an error.
  CHECK_STATUS(vpiCreateStereoDisparityEstimator(params.backends, params.input_width, params.input_height,
                                                 params.stereo_format, &params.stereo_params,
                                                 &stereo));

  // Create the image where the disparity map will be stored.
  CHECK_STATUS(vpiImageCreate(params.output_width, params.output_height, params.disparity_format, 0,
                              &disparity));

  // Create the confidence image if the backend can support it
  if (params.use_confidence_map) {
    CHECK_STATUS(
        vpiImageCreate(params.output_width, params.output_height, VPI_IMAGE_FORMAT_U16, 0, &confidence_map));
  }
}

void DisparityEstimator::Apply(VPIStream& stream, VPIImage& stereo_left,
                                    VPIImage& stereo_right, cv::Mat& cv_disparity_color,
                                    cv::Mat& cv_confidence) {
  // Stereo Left and stereo right should be already in the correct format
  CHECK_STATUS(vpiSubmitStereoDisparityEstimator(stream, params.backends, stereo, stereo_left,
                                                 stereo_right, disparity, confidence_map, NULL));

  // Wait until the algorithm finishes processing
  CHECK_STATUS(vpiStreamSync(stream));

  // ========================================
  // Output pre-processing and saving to disk
  // Lock output to retrieve its data on cpu memory
  VPIImageData data;
  CHECK_STATUS(
      vpiImageLockData(disparity, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &data));

  // Make an OpenCV matrix out of this image
  cv::Mat cv_disparity;
  CHECK_STATUS(vpiImageDataExportOpenCVMat(data, &cv_disparity));

  // Scale result and write it to disk. Disparities are in Q10.5 format,
  // so to map it to float, it gets divided by 32. Then the resulting disparity range,
  // from 0 to stereo.maxDisparity gets mapped to 0-255 for proper output.
  cv_disparity.convertTo(cv_disparity, CV_8UC1, 255.0 / (32 * params.stereo_params.maxDisparity), 0);

  // Apply TURBO colormap to turn the disparities into color, reddish hues
  // represent objects closer to the camera, blueish are farther away.
  cv::applyColorMap(cv_disparity, cv_disparity_color, cv::COLORMAP_TURBO);

  // Done handling output, don't forget to unlock it.
  CHECK_STATUS(vpiImageUnlock(disparity));

  if (confidence_map) {
    VPIImageData data;
    CHECK_STATUS(
        vpiImageLockData(confidence_map, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &data));

    CHECK_STATUS(vpiImageDataExportOpenCVMat(data, &cv_confidence));

    // Confidence map varies from 0 to 65535, we scale it to
    // [0-255].
    cv_confidence.convertTo(cv_confidence, CV_8UC1, 255.0 / 65535, 0);

    CHECK_STATUS(vpiImageUnlock(confidence_map));

    // When pixel confidence is 0, its color in the disparity
    // output is black.
    cv::Mat cv_mask;
    cv::threshold(cv_confidence, cv_mask, 1, 255, cv::THRESH_BINARY);
    cv::cvtColor(cv_mask, cv_mask, cv::COLOR_GRAY2BGR);
    cv::bitwise_and(cv_disparity_color, cv_mask, cv_disparity_color);
  }

  return std::pair<VPIImage&, VPIImage&>(disparity, confidence_map);
}

DisparityEstimator::~DisparityEstimator() {
  // stream should be destroyed first in case images are still being used
  vpiImageDestroy(confidence_map);
  vpiImageDestroy(disparity);
  vpiPayloadDestroy(stereo);
}

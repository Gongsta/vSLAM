#include "disparity.hpp"

DisparityEstimator::DisparityEstimator(cv::Mat& cv_img_in, VPIStream& stream, uint64_t backends)
    : stream{stream}, backends{backends} {
  int32_t input_width = cv_img_in.cols;
  int32_t input_height = cv_img_in.rows;

  // =================================
  // Allocate all VPI resources needed
  // Format conversion parameters needed for input pre-processing
  CHECK_STATUS(vpiInitConvertImageFormatParams(&conv_params));
  CHECK_STATUS(vpiInitStereoDisparityEstimatorCreationParams(&stereo_params));

  // Default format and size for inputs and outputs
  VPIImageFormat stereoFormat = VPI_IMAGE_FORMAT_Y16_ER;
  VPIImageFormat disparityFormat = VPI_IMAGE_FORMAT_S16;

  int stereo_width = input_width;
  int stereo_height = input_height;
  int output_width = input_width;
  int output_height = input_height;

  // Override some backend-dependent parameters, see
  // https://docs.nvidia.com/vpi/algo_stereo_disparity.html

  if (backends == (VPI_BACKEND_PVA | VPI_BACKEND_NVENC | VPI_BACKEND_VIC)) {
    // Input and output width and height has to be 1920x1080 in block-linear format for
    // pva-nvenc-vic pipeline
    stereoFormat = VPI_IMAGE_FORMAT_Y16_ER_BL;
    stereo_width = 1920;
    stereo_height = 1080;

    // For PVA+NVENC+VIC mode, 16bpp input must be MSB-aligned, which
    // is equivalent to say that it is Q8.8 (fixed-point, 8 decimals).
    conv_params.scale = 256;

    // Maximum disparity is fixed to 256.
    stereo_params.maxDisparity = 256;

    // pva-nvenc-vic pipeline only supports downscaleFactor = 4
    stereo_params.downscaleFactor = 4;
    output_width = stereo_width / stereo_params.downscaleFactor;
    output_height = stereo_height / stereo_params.downscaleFactor;
  } else if (backends & VPI_BACKEND_OFA) {
    // Implementations using OFA require BL input
    stereoFormat = VPI_IMAGE_FORMAT_Y16_ER_BL;

    if (backends == VPI_BACKEND_OFA) {
      disparityFormat = VPI_IMAGE_FORMAT_S16_BL;
    }

    // Output width including downscaleFactor must be at least max(64,
    // maxDisparity/downscaleFactor) when OFA+PVA+VIC are used
    if (backends & VPI_BACKEND_PVA) {
      int downscaledWidth =
          (input_width + stereo_params.downscaleFactor - 1) / stereo_params.downscaleFactor;
      int minWidth =
          std::max(stereo_params.maxDisparity / stereo_params.downscaleFactor, downscaledWidth);
      output_width = std::max(64, minWidth);
      output_height = (input_height * stereo_width) / input_width;
      stereo_width = output_width * stereo_params.downscaleFactor;
      stereo_height = output_height * stereo_params.downscaleFactor;
    }

    // Maximum disparity can be either 128 or 256
    stereo_params.maxDisparity = 128;
  } else if (backends == VPI_BACKEND_PVA) {
    // PVA requires that input and output resolution is 480x270
    stereo_width = output_width = 480;
    stereo_height = output_height = 270;

    // maxDisparity must be 64
    stereo_params.maxDisparity = 64;
  }

  // Create the payload for Stereo Disparity algorithm.
  // Payload is created before the image objects so that non-supported backends can be trapped
  // with an error.
  CHECK_STATUS(vpiCreateStereoDisparityEstimator(backends, stereo_width, stereo_height,
                                                 stereoFormat, &stereo_params, &stereo));

  // Create the image where the disparity map will be stored.
  CHECK_STATUS(vpiImageCreate(output_width, output_height, disparityFormat, 0, &disparity));

  // Create the input stereo images
  CHECK_STATUS(vpiImageCreate(stereo_width, stereo_height, stereoFormat, 0, &stereo_left));
  CHECK_STATUS(vpiImageCreate(stereo_width, stereo_height, stereoFormat, 0, &stereo_right));

  // Create some temporary images, and the confidence image if the backend can support it
  if (backends == (VPI_BACKEND_PVA | VPI_BACKEND_NVENC | VPI_BACKEND_VIC)) {
    // Need an temporary image to convert BGR8 input from OpenCV into pixel-linear 16bpp
    // grayscale. We can't convert it directly to block-linear since CUDA backend doesn't
    // support it, and VIC backend doesn't support BGR8 inputs.
    CHECK_STATUS(vpiImageCreate(input_width, input_height, VPI_IMAGE_FORMAT_Y16_ER, 0, &tmp_left));
    CHECK_STATUS(vpiImageCreate(input_width, input_height, VPI_IMAGE_FORMAT_Y16_ER, 0, &tmp_right));

    // confidence map is needed for pva-nvenc-vic pipeline
    CHECK_STATUS(
        vpiImageCreate(output_width, output_height, VPI_IMAGE_FORMAT_U16, 0, &confidence_map));
  } else if (backends & VPI_BACKEND_OFA) {
    // OFA also needs a temporary buffer for format conversion
    CHECK_STATUS(vpiImageCreate(input_width, input_height, VPI_IMAGE_FORMAT_Y16_ER, 0, &tmp_left));
    CHECK_STATUS(vpiImageCreate(input_width, input_height, VPI_IMAGE_FORMAT_Y16_ER, 0, &tmp_right));

    if (backends & VPI_BACKEND_PVA) {
      // confidence map is supported by OFA+PVA
      CHECK_STATUS(
          vpiImageCreate(output_width, output_height, VPI_IMAGE_FORMAT_U16, 0, &confidence_map));
    }
  } else if (backends == VPI_BACKEND_PVA) {
    // PVA also needs a temporary buffer for format conversion and rescaling
    CHECK_STATUS(vpiImageCreate(input_width, input_height, stereoFormat, 0, &tmp_left));
    CHECK_STATUS(vpiImageCreate(input_width, input_height, stereoFormat, 0, &tmp_right));
  } else if (backends == VPI_BACKEND_CUDA) {
    CHECK_STATUS(
        vpiImageCreate(input_width, input_height, VPI_IMAGE_FORMAT_U16, 0, &confidence_map));
  }
}


void DisparityEstimator::ProcessFrame(cv::Mat& cv_img_left, cv::Mat& cv_img_right,
                                      cv::Mat& cv_disparity_color, cv::Mat& cv_confidence) {
                                    

  // We now wrap the loaded images into a VPIImage object to be used by VPI.
  // VPI won't make a copy of it, so the original image must be in scope at all times.
  CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(cv_img_left, 0, &in_left));
  CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(cv_img_right, 0, &in_right));

  // ================
  // Processing stage

  // -----------------
  // Pre-process input
  if (backends == (VPI_BACKEND_PVA | VPI_BACKEND_NVENC | VPI_BACKEND_VIC) ||
      backends == VPI_BACKEND_PVA || backends == VPI_BACKEND_OFA ||
      backends == (VPI_BACKEND_OFA | VPI_BACKEND_PVA | VPI_BACKEND_VIC)) {
    // Convert opencv input to temporary grayscale format using CUDA
    CHECK_STATUS(
        vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, in_left, tmp_left, &conv_params));
    CHECK_STATUS(
        vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, in_right, tmp_right, &conv_params));

    // Do both scale and final image format conversion on VIC.
    CHECK_STATUS(vpiSubmitRescale(stream, VPI_BACKEND_VIC, tmp_left, stereo_left, VPI_INTERP_LINEAR,
                                  VPI_BORDER_CLAMP, 0));
    CHECK_STATUS(vpiSubmitRescale(stream, VPI_BACKEND_VIC, tmp_right, stereo_right,
                                  VPI_INTERP_LINEAR, VPI_BORDER_CLAMP, 0));
  } else {
    // Convert opencv input to grayscale format using CUDA
    CHECK_STATUS(
        vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, in_left, stereo_left, &conv_params));
    CHECK_STATUS(
        vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, in_right, stereo_right, &conv_params));
  }

  // ------------------------------
  // Do stereo disparity estimation

  // Submit it with the input and output images
  CHECK_STATUS(vpiSubmitStereoDisparityEstimator(stream, backends, stereo, stereo_left,
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
  cv_disparity.convertTo(cv_disparity, CV_8UC1, 255.0 / (32 * stereo_params.maxDisparity), 0);

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

  vpiImageDestroy(in_left);
  vpiImageDestroy(in_right);
}

DisparityEstimator::~DisparityEstimator() {
  // stream should be destroyed first in case images are still being used 
  vpiImageDestroy(tmp_left);
  vpiImageDestroy(tmp_right);
  vpiImageDestroy(stereo_left);
  vpiImageDestroy(stereo_right);
  vpiImageDestroy(confidence_map);
  vpiImageDestroy(disparity);
  vpiPayloadDestroy(stereo);
}

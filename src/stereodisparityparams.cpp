#include "stereodisparityparams.hpp"

StereoDisparityParams::StereoDisparityParams(const uint64_t backends) : backends{backends} {
  CHECK_STATUS(vpiInitConvertImageFormatParams(&conv_params));
  CHECK_STATUS(vpiInitStereoDisparityEstimatorCreationParams(&stereo_params));

  switch (backends) {
    case VPI_BACKEND_CUDA:
      use_confidence_map = true;
      break;
    case VPI_BACKEND_PVA:
      input_width = output_width = 480;
      input_height = output_height = 270;
      // maxDisparity must be 64
      stereo_params.maxDisparity = 64;
      break;
    case VPI_BACKEND_OFA:
      // Implementations using OFA require BL input
      stereo_format = VPI_IMAGE_FORMAT_Y16_ER_BL;
      disparity_format = VPI_IMAGE_FORMAT_S16_BL;
      use_confidence_map = true;
      break;
    case VPI_BACKEND_TEGRA:
      stereo_format = VPI_IMAGE_FORMAT_Y16_ER_BL;
      disparity_format = VPI_IMAGE_FORMAT_S16;
      output_width = 1920;
      output_height = 1080;
      break;
    case (VPI_BACKEND_PVA | VPI_BACKEND_NVENC | VPI_BACKEND_VIC):
      stereo_format = VPI_IMAGE_FORMAT_Y16_ER_BL;
      input_width = 1920;
      input_height = 1080;
      use_confidence_map = true;
      // For PVA+NVENC+VIC mode, 16bpp input must be MSB-aligned, which
      // is equivalent to say that it is Q8.8 (fixed-point, 8 decimals).
      conv_params.scale = 256;

      // Maximum disparity is fixed to 256.
      stereo_params.maxDisparity = 256;

      // pva-nvenc-vic pipeline only supports downscaleFactor = 4
      stereo_params.downscaleFactor = 4;
      output_width = input_width / stereo_params.downscaleFactor;
      output_height = input_height / stereo_params.downscaleFactor;
      break;
    case (VPI_BACKEND_OFA | VPI_BACKEND_PVA | VPI_BACKEND_VIC):
      stereo_format = VPI_IMAGE_FORMAT_Y16_ER_BL;
      stereo_params.maxDisparity = 128;
      int downscaledWidth =
          (input_width + stereo_params.downscaleFactor - 1) / stereo_params.downscaleFactor;
      int minWidth =
          std::max(stereo_params.maxDisparity / stereo_params.downscaleFactor, downscaledWidth);
      output_width = std::max(64, minWidth);
      output_height = (input_height * input_width) / input_width;
      input_width = output_width * stereo_params.downscaleFactor;
      input_height = output_height * stereo_params.downscaleFactor;
      use_confidence_map = true;
      break;
  }
};

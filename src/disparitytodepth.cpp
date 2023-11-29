#include "cuda_processing.hpp"
#include "disparitytodepth.hpp"

DisparityToDepthConverter::DisparityToDepthConverter(int width, int height, VPIImageFormat format) {
  CHECK_STATUS(vpiImageCreate(width, height, format, 0, &depth_map));
}

DisparityToDepthConverter::~DisparityToDepthConverter() { vpiImageDestroy(depth_map); }

VPIImage& DisparityToDepthConverter::Apply(VPIImage& disparity_map) {
    CHECK_STATUS(
        vpiImageLockData(disparity_map, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &data));

    CHECK_STATUS(vpiImageDataExportOpenCVMat(data, &cv_confidence));

    CHECK_STATUS(vpiImageUnlock(disparity_map));
  disparity_map

}

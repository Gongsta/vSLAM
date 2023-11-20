#include "imagerectifier.hpp"

ImageRectifier::ImageRectifier(int width, int height, VPIImageFormat format) {
  CHECK_STATUS(vpiImageCreate(width, height, format, 0, &img_out));
}

VPIImage& ImageRectifier::Apply(VPIStream& stream, VPIImage& img_in) {
  // CHECK_STATUS(vpiSubmitRescale(stream, VPI_BACKEND_VIC, img_in, img_out, VPI_INTERP_LINEAR,
  //                               VPI_BORDER_CLAMP, 0));
  return img_out;
}
ImageRectifier::~ImageRectifier() { vpiImageDestroy(img_out); }

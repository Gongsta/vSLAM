#include "imagecopy.hpp"

ImageCopy::ImageCopy(int width, int height, VPIImageFormat format) {
  CHECK_STATUS(vpiImageCreate(width, height, format, 0, &img_out));
}

VPIImage& ImageCopy::Apply(VPIStream& stream, VPIImage& img_in) {
  img_out = img_in;
  // CHECK_STATUS(vpiSubmitRescale(stream, VPI_BACKEND_VIC, img_in, img_out, VPI_INTERP_LINEAR,
  //                               VPI_BORDER_CLAMP, 0));
  return img_out;
}
ImageCopy::~ImageCopy() { vpiImageDestroy(img_out); }

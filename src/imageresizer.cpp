#include "imageresizer.hpp"

ImageResizer::ImageResizer(int resize_width, int resize_height, VPIImageFormat image_format,
                           uint64_t backends)
    : backends{backends} {
  CHECK_STATUS(vpiImageCreate(resize_width, resize_height, image_format, 0, &img_out));
}

ImageResizer::~ImageResizer() { vpiImageDestroy(img_out); }

VPIImage& ImageResizer::Apply(VPIStream& stream, VPIImage& img_in) {
  CHECK_STATUS(
      vpiSubmitRescale(stream, backends, img_in, img_out, VPI_INTERP_LINEAR, VPI_BORDER_CLAMP, 0));
  return img_out;
}

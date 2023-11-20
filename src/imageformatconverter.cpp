#include "imageformatconverter.hpp"

ImageFormatConverter::ImageFormatConverter(int out_width, int out_height,
                                           VPIConvertImageFormatParams conv_params,
                                           VPIImageFormat image_format,
                                           uint64_t backends)
    : conv_params{conv_params}, img_out_format{image_format}, backends{backends} {
  CHECK_STATUS(vpiImageCreate(out_width, out_height, image_format, 0, &img_out));
}

ImageFormatConverter::~ImageFormatConverter() { vpiImageDestroy(img_out); }

VPIImage& ImageFormatConverter::Apply(VPIStream& stream, VPIImage img_in) {
  CHECK_STATUS(vpiSubmitConvertImageFormat(stream, backends, img_in, img_out, &conv_params));
  return img_out;
}

#ifndef IMAGERESIZER_HPP_
#define IMAGERESIZER_HPP_

#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/Rescale.h>
#include <vpi/algo/StereoDisparity.h>

#include "vpi_utils.hpp"

class ImageResizer {
 private:
  uint64_t backends;
  VPIImage img_out;

 public:
  ImageResizer(int resize_width, int resize_height, VPIImageFormat image_format,
               uint64_t backends = VPI_BACKEND_CUDA);
  ~ImageResizer();
  VPIImage& Apply(VPIStream& stream, VPIImage& img_in);
};

#endif

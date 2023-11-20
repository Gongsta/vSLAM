#ifndef RECTIFIER_HPP_
#define RECTIFIER_HPP_

#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/Rescale.h>
#include <vpi/algo/StereoDisparity.h>

#include "vpi_utils.hpp"

class ImageRectifier {
 private:
  VPIImage img_out;

 public:
  ImageRectifier(int width, int height, VPIImageFormat format);
  ~ImageRectifier();

  VPIImage& Apply(VPIStream& stream, VPIImage& img_in);
};

#endif

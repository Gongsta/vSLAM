#ifndef IMAGEFORMATCONVERTER_HPP_
#define IMAGEFORMATCONVERTER_HPP_

#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/Rescale.h>
#include <vpi/algo/StereoDisparity.h>

#include <cstring>  // for memset
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <vpi/OpenCVInterop.hpp>

#include "vpi_utils.hpp"

class ImageFormatConverter {
 private:
  VPIConvertImageFormatParams conv_params;
  VPIImageFormat img_out_format;
  uint64_t backends;

  VPIImage img_out;

 public:
  ImageFormatConverter(int out_width, int out_height, VPIConvertImageFormatParams conv_params,
                       VPIImageFormat image_format, uint64_t backends = VPI_BACKEND_CUDA);
  ~ImageFormatConverter();
  VPIImage& Apply(VPIStream& stream, VPIImage& img_in);
};

#endif

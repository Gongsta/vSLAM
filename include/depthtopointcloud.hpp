#ifndef DEPTHTOPOINTCLOUD_HPP_
#define DEPTHTOPOINTCLOUD_HPP_

#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/Rescale.h>
#include <vpi/algo/StereoDisparity.h>

#include <Eigen/Core>
#include <cstring>  // for memset
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <vpi/OpenCVInterop.hpp>

#include "vpi_utils.hpp"

class DepthToPointCloudConverter {
 public:
  DepthToPointCloudConverter(int height, int width);
  ~DepthToPointCloudConverter();
  Eigen::Vector3f* Apply(VPIStream& stream, float* depth_map);
};

#endif

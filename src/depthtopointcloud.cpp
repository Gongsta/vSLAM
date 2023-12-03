#include "cuda/processing.hpp"
#include "depthtopointcloud.hpp"

DepthToPointCloudConverter::DepthToPointCloudConverter(int width, int height) {
  // CUDAImage(width, height);

  // REMOVE THIS AND CREATE A CUDA IMAGE?
//   CHECK_STATUS(vpiImageCreate(width, height, format, 0, &depth_map));
// cudaMalloc()
}

DepthToPointCloudConverter::~DepthToPointCloudConverter() {}

Eigen::Vector3f* DepthToPointCloudConverter::Apply(VPIStream& stream, float* depth_map) {
    return nullptr;
}

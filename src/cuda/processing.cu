
#include "cuda/processing.hpp"

__global__ void ComputeDisparityToDepthCUDA(float* disparity_map, float* depth_map, int width,
                                            int height, float fx, float baseline) {
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  if (col < width && row < height) {
    int idx = row * width + col;
    depth_map[idx] = fx * baseline / disparity_map[idx];
  }
}

// __global__ void ComputeDepthToPointcloudCUDA(float* depth_map, Eigen::Vector3f* pointcloud_map, int width, int height, float fx, float fy, float cx, float cy) {
//   int col = blockDim.x * blockIdx.x + threadIdx.x;
//   int row = blockDim.y * blockIdx.y + threadIdx.y;
  
//   int idx = row * kernel_width + col;
//   if (idx < size) {
//     pointcloud_map[idx] = 
//   }


//   if (col < width && row < height) {
//     // Calculate the index in the 1D depth map
//     int idx = row * width + col;

//     // Calculate the 3D coordinates of the point in the camera coordinate system
//     float depth = depth_map[idx];
//     float x = (col - cx) * depth / fx;
//     float y = (row - cy) * depth / fy;
//     float z = depth;

//     // Calculate the index in the 1D point cloud map (assuming a 3-channel output)
//     int cloud_idx = idx * 3;

//     // Store the 3D coordinates in the point cloud map
//     pointcloud_map[cloud_idx] = x;
//     pointcloud_map[cloud_idx + 1] = y;
//     pointcloud_map[cloud_idx + 2] = z;
//   }
// }

void ComputeDisparityToDepth(cudaStream_t& stream, float* disparity_map, float* depth_map,
                             int width, int height, float fx, float baseline) {
  ComputeDisparityToDepthCUDA<<<dim3(ceil(width / 32.0), ceil(height / 32.0), 1), dim3(32, 32, 1),
                                0, stream>>>(disparity_map, depth_map, width, height, fx, baseline);
}



// TODO: Make fx,fy shared variables
__global__
void ComputeDisparityToDepth(float* disparity_map ,float* depth_map, int width, int height, float fx, float baseline) {
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  if ( col < width && row < height) {
      // get 1D coordinate for the grayscale image
      int idx = row * width + col;

      depth_map[idx] = fx * baseline / disparity_map[idx];
  }
}

__global__
void ComputeDepthToPointcloud(float* depth_map, float* pointcloud_map, int width, int height, float fx, float fy, float cx, float cy) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col < width && row < height) {
        // Calculate the index in the 1D depth map
        int idx = row * width + col;

        // Calculate the 3D coordinates of the point in the camera coordinate system
        float depth = depth_map[idx];
        float x = (col - cx) * depth / fx;
        float y = (row - cy) * depth / fy;
        float z = depth;

        // Calculate the index in the 1D point cloud map (assuming a 3-channel output)
        int cloud_idx = idx * 3;

        // Store the 3D coordinates in the point cloud map
        pointcloud_map[cloud_idx] = x;
        pointcloud_map[cloud_idx + 1] = y;
        pointcloud_map[cloud_idx + 2] = z;
    }
}

int main() {
  
 


}
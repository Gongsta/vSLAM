#include <cuda_runtime.h>

void ComputeDisparityToDepth(cudaStream_t& stream, float* disparity_map, float* depth_map,
                             int width, int height, float fx, float baseline);

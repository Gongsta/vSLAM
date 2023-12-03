#ifndef CUDAIMAGE_HPP_
#define CUDAIMAGE_HPP_

#include <cuda_runtime.h>
#include <vpi/Image.h>
#include <vpi/Stream.h>

#include <iostream>

#include "vpi_utils.hpp"

/**
 * @brief Wrapper around VPIImage, doesn't actually allocate / free the underlying memory. Follows
 * RAII pattern. CUDAImage MUST be destructed before underlying VPIImage can be used again.
 *
 */
template <typename T>
struct CUDAImage {
  T* data;
  int width;
  int height;
  VPIImage vpi_img;

  CUDAImage(VPIImage vpi_img) : vpi_img{vpi_img} { this->Lock(); }
  ~CUDAImage() { this->Unlock(); }

  void Lock() {
    VPIImageData vpi_img_data;
    CHECK_STATUS(vpiImageLockData(vpi_img, VPI_LOCK_READ_WRITE, VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR,
                                  &vpi_img_data));
    if (vpi_img_data.bufferType == VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR) {
      // Image dimensions
      width = vpi_img_data.buffer.pitch.planes[0].width;
      height = vpi_img_data.buffer.pitch.planes[0].height;
      data = reinterpret_cast<T*>(vpi_img_data.buffer.pitch.planes[0].data);
    } else {
      throw std::runtime_error(
          "Invalid Image Format, must be VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, got " +
          std::to_string(vpi_img_data.bufferType));
    }
  }

  void Unlock() { CHECK_STATUS(vpiImageUnlock(vpi_img)); }
};

#endif

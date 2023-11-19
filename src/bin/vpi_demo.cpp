#include <opencv2/core/version.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#if CV_MAJOR_VERSION >= 3
#include <opencv2/imgcodecs.hpp>
#else
#include <opencv2/highgui/highgui.hpp>
#endif

#include <vpi/Image.h>
#include <vpi/Stream.h>
#include <vpi/algo/BoxFilter.h>
#include <vpi/algo/ConvertImageFormat.h>

#include <iostream>
#include <vpi/OpenCVInterop.hpp>

// Function declarations
void processFrame(VPIStream stream, VPIImage image, VPIImage imageGray, VPIImage blurred,
                  cv::Mat& cvImage, cv::Mat& cvOut);

int main() {
  cv::VideoCapture capture;
  if (!capture.open(0, cv::CAP_V4L2)) {
    std::cerr << "Error opening video stream" << std::endl;
    return -1;
  }

  VPIStream stream;
  if (vpiStreamCreate(0, &stream) != VPI_SUCCESS) {
    std::cerr << "Error creating VPI stream" << std::endl;
    return -1;
  }

  VPIImage image, imageGray, blurred;
  cv::Mat cvImage, cvOut;

  capture.read(cvImage);

  // Allocate memory
  vpiImageCreate(cvImage.cols, cvImage.rows, VPI_IMAGE_FORMAT_U8, 0, &imageGray);
  vpiImageCreate(cvImage.cols, cvImage.rows, VPI_IMAGE_FORMAT_U8, 0, &blurred);

  while (true) {
    capture.read(cvImage);
    if (cvImage.empty()) {
      continue;
    }

    // Process the frame
    processFrame(stream, image, imageGray, blurred, cvImage, cvOut);

    cv::imshow("Image", cvOut);
    if (cv::waitKey(5) >= 0) {
      break;
    }
  }

  // Clean up
  vpiStreamDestroy(stream);
  vpiImageDestroy(image);
  vpiImageDestroy(imageGray);
  vpiImageDestroy(blurred);
  capture.release();

  return 0;
}

void processFrame(VPIStream stream, VPIImage image, VPIImage imageGray, VPIImage blurred,
                  cv::Mat& cvImage, cv::Mat& cvOut) {
  // Ensure previous image is destroyed before reassignment
  vpiImageCreateWrapperOpenCVMat(cvImage, 0, &image);

  // Convert BGR8 to Grayscale
  vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, image, imageGray, NULL);

  // Apply box filter
  vpiSubmitBoxFilter(stream, VPI_BACKEND_CUDA, imageGray, blurred, 5, 5, VPI_BORDER_ZERO);

  // Sync stream
  vpiStreamSync(stream);

  // Retrieve output image
  VPIImageData outData;
  vpiImageLockData(blurred, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &outData);
  vpiImageDataExportOpenCVMat(outData, &cvOut);
  vpiImageUnlock(blurred);
  vpiImageDestroy(image);
}

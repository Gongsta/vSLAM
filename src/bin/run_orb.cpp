#include <vpi/Array.h>
#include <vpi/Image.h>
#include <vpi/Pyramid.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/GaussianPyramid.h>
#include <vpi/algo/ImageFlip.h>
#include <vpi/algo/ORB.h>

#include <bitset>
#include <cstdio>
#include <cstring>  // for memset
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <vpi/OpenCVInterop.hpp>

#include "orb.hpp"
#include "vpi_utils.hpp"

int main() {
  VPIBackend backend = VPI_BACKEND_CUDA;
  cv::VideoCapture cap(0, cv::CAP_V4L2);
  cv::Mat cv_img_in, cv_img_out;

  try {
    cap.read(cv_img_in);

    VPIStream stream;
    CHECK_STATUS(vpiStreamCreate(0, &stream));
    ORB orb{cv_img_in, stream};

    while (true) {
      cap.read(cv_img_in);
      if (cv_img_in.empty()) {
        continue;
      }

      orb.ProcessFrame(cv_img_in, cv_img_out);
      cv::imshow("Image", cv_img_out);
      if (cv::waitKey(5) >= 0) {
        break;
      }
    }

    vpiStreamDestroy(stream);

  } catch (std::exception &e) {
    std::cerr << e.what() << std::endl;
  }

  return 0;
}

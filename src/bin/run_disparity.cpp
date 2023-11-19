#include "disparity.hpp"
#include "vpi_utils.hpp"

int main() {
  int retval = 0;
  cv::VideoCapture cap(0, cv::CAP_V4L2);
  cv::Mat raw_stereo_img;

  cap.read(raw_stereo_img);

  int width = raw_stereo_img.cols;
  int height = raw_stereo_img.rows;
  cv::Rect left_img_index = cv::Rect(0, 0, width / 2, height);
  cv::Rect right_img_index = cv::Rect(width / 2, 0, width / 2, height);

  cv::Mat cv_img_left = raw_stereo_img(left_img_index);
  cv::Mat cv_img_right = raw_stereo_img(right_img_index);

  cv::Mat cv_disparity_color, cv_confidence;

  // uint64_t backends = VPI_BACKEND_OFA | VPI_BACKEND_PVA | VPI_BACKEND_VIC;
  uint64_t backends = VPI_BACKEND_CUDA;
  VPIStream stream;

  try {
    CHECK_STATUS(vpiStreamCreate(0, &stream));

    DisparityEstimator disparity{cv_img_left, stream, backends};

    while (true) {
      cap.read(raw_stereo_img);
      cv_img_left = raw_stereo_img(left_img_index);
      cv_img_right = raw_stereo_img(right_img_index);

      disparity.ProcessFrame(cv_img_left, cv_img_right, cv_disparity_color, cv_confidence);

      imshow("disparity", cv_disparity_color);
      // imshow("confidence", cv_confidence);
      if (cv::waitKey(5) >= 0) {
        break;
      }
    }
  } catch (std::exception &e) {
    std::cerr << e.what() << std::endl;
    retval = 1;
  }

  // ========
  // Clean up
  vpiStreamDestroy(stream);

  return retval;
}

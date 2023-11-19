#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
  cv::VideoCapture cap(0, cv::CAP_V4L2);

  if (!cap.isOpened()) {
    std::cerr << "ERROR! Unable to open camera\n";
    return -1;
  }

  cv::Mat raw_stereo_img;

  while (cap.grab()) {
    cap.retrieve(raw_stereo_img);

    if (raw_stereo_img.empty()) {
      std::cerr << "ERROR! blank frame grabbed\n";
      continue;
    }

    cv::imshow("Stereo Image", raw_stereo_img);
    if (cv::waitKey(5) >= 0) {
      break;
    }
  }

  return 0;
}

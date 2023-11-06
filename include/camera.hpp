#ifndef CAMERA_HPP_
#define CAMERA_HPP_

#include <opencv2/videoio.hpp>

class Camera final : public cv::VideoCapture {
 public:
  Camera(int deviceID = 0, int apiID = cv::CAP_ANY);
};

#endif

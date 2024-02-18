#include <chrono>
#include <opencv2/opencv.hpp>
#include <sl/Camera.hpp>

#include "save_depth.hpp"
#include "zed_utils.hpp"

int main() {
  sl::Camera zed;

  sl::InitParameters init_params;
  init_params.camera_resolution = sl::RESOLUTION::HD1080;
  init_params.camera_fps = 30;
  init_params.depth_mode = sl::DEPTH_MODE::NEURAL;
  init_params.coordinate_units = sl::UNIT::METER;

  sl::ERROR_CODE err = zed.open(init_params);
  if (err != sl::ERROR_CODE::SUCCESS) {
    printf("%s\n", toString(err).c_str());
    zed.close();
    return 1;
  }

  sl::RuntimeParameters runtime_parameters;

  sl::Resolution image_size = zed.getCameraInformation().camera_configuration.resolution;
  sl::Mat image_zed(image_size.width, image_size.height, sl::MAT_TYPE::U8_C4);
  cv::Mat image_ocv = sl::slMat2cvMat(image_zed);

  auto start = std::chrono::high_resolution_clock::now();
  char key = ' ';
  // Loop until 'q' is pressed
  while (key != 'q') {
    if (zed.grab(runtime_parameters) == sl::ERROR_CODE::SUCCESS) {
      zed.retrieveImage(image_zed, sl::VIEW::DEPTH, sl::MEM::CPU);

      auto curr = std::chrono::high_resolution_clock::now();
      auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(curr - start).count();
      // std::cout << "Frequency: " << 1000.0 / latency << " Hz" << std::endl;
      // std::cout << "Zed Frequency: " << zed.getCurrentFPS() << std::endl;
      start = curr;
      cv::imshow("Image", image_ocv);
      cv::waitKey(1);
    }
  }

  zed.close();
  return 0;
}

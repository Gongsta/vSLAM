#include <sl/Camera.hpp>

#include "orb.hpp"
#include "vpi_utils.hpp"
int main() {
  sl::Camera zed;
  sl::InitParameters init_params;
  init_params.camera_resolution = sl::RESOLUTION::SVGA;
  init_params.camera_fps = 120;
  init_params.depth_mode = sl::DEPTH_MODE::NONE;
  init_params.coordinate_units = sl::UNIT::METER;

  sl::ERROR_CODE err = zed.open(init_params);
  if (!zed.getDeviceList().empty()) {
    std::cout << "Loading Zed with serial number" << zed.getDeviceList()[0].serial_number
              << std::endl;
  }
  if (err != sl::ERROR_CODE::SUCCESS) {
    printf("%s\n", toString(err).c_str());
    zed.close();
    return 1;
  }

  // Get Zed Camera Information
  sl::Resolution image_size = zed.getCameraInformation().camera_configuration.resolution;
  int img_width = image_size.width;
  int img_height = image_size.height;

  float fx = zed.getCameraInformation().camera_configuration.calibration_parameters.left_cam.fx;
  float baseline =
      zed.getCameraInformation().camera_configuration.calibration_parameters.getCameraBaseline();
  std::cout << "fx: " << fx << " baseline: " << baseline << std::endl;

  // Mapping sl::Mat to cv::Mat
  sl::Mat zed_img_left(image_size.width, image_size.height, sl::MAT_TYPE::U8_C4);
  cv::Mat cv_img_left = sl::slMat2cvMat(zed_img_left);
  sl::Mat zed_img_right(image_size.width, image_size.height, sl::MAT_TYPE::U8_C4);
  cv::Mat cv_img_right = sl::slMat2cvMat(zed_img_right);

  // cv::Mat cv_img_in, cv_img_out;

  uint64_t backends = VPI_BACKEND_CUDA;
  try {
    cap.read(cv_img_in);

    VPIStream stream;
    CHECK_STATUS(vpiStreamCreate(0, &stream));
    ORBFeatureDetector orb{cv_img_left, stream, backends};

    while (true) {
      if (zed.grab() != sl::ERROR_CODE::SUCCESS) {
        continue;
      }

      zed.retrieveImage(zed_img_left, sl::VIEW::LEFT, sl::MEM::CPU);

      orb.Apply(cv_img_left, cv_img_out);
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

#include "calibrationsettings.hpp"

void CalibrationSettings::read(const cv::FileNode& node) {
  node["board_size_width"] >> board_size.width;
  node["board_size_height"] >> board_size.height;
  node["square_size"] >> square_size;
  node["num_stereo_pairs"] >> num_stereo_pairs;
  node["calibration_export_path"] >> calibration_export_path;

  validate();
}

void CalibrationSettings::validate() {
  good_input = true;
  if (board_size.width <= 0 || board_size.height <= 0) {
    std::cerr << "Invalid Board size: " << board_size.width << " " << board_size.height
              << std::endl;
    good_input = false;
  }
  if (square_size <= 10e-6) {
    std::cerr << "Square size too small" << square_size << std::endl;
    good_input = false;
  }

  if (calibration_export_path.empty()) {  // Check for valid output
    std::cerr << "Calibration export path is not specified" << std::endl;
  }
}

#ifndef CAMERAPARAMETERS_HPP_
#define CAMERAPARAMETERS_HPP_

void CalibrationSettings::read(const cv::FileNode& node) {
  node["board_size_width"] >> board_size.width;
  node["board_size_height"] >> board_size.height;
  node["square_size"] >> square_size;
  node["num_stereo_pairs"] >> num_stereo_pairs;
  node["calibration_export_path"] >> calibration_export_path;

  validate();
}

// Configure the calibration settings
class CameraParameters {
 public:
  cv::Mat K;             // Calibration Matrix (intrinsic parameters)
  cv::Mat D;             // Distortion Parameters

  void read(const cv::FileNode& node);
  void validate();
};

class StereoCameraParameters {
 public:
  CameraParameters left_params;
  CameraParameters right_params;

}

static inline void
read(const cv::FileNode& node, CalibrationSettings& x,
     const CalibrationSettings& default_value = CalibrationSettings()) {
  if (node.empty()) {
    x = default_value;
  } else {
    x.read(node);
  }
}

#endif

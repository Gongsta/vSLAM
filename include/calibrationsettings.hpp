#ifndef CALIBRATIONSETTINGS_HPP
#define CALIBRATIONSETTINGS_HPP

#include <opencv2/opencv.hpp>
#include <string>

class CalibrationSettings {
 public:
  cv::Size board_size;  // The size of the board -> Number of items by width and height
  float square_size;    // The size of a square in your defined unit (point, millimeter,etc).
  int num_stereo_pairs;  // Number of stereo image pairs used for calibration
  std::string calibration_export_path;  // The path to export the calibration results
  bool good_input;

  void read(const cv::FileNode& node);
  void validate();
};

static inline void read(const cv::FileNode& node, CalibrationSettings& x, const CalibrationSettings& default_value = CalibrationSettings())
{
    if(node.empty())
        x = default_value;
    else
        x.read(node);
}

#endif

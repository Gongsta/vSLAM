/**
 * @file calibration.hpp
 * @author Steven Gong (gong.steven@hotmail.com)
 * @brief Runs calibration. Currently only works on a synchronized stereo image pair. Inspired from
 * https://docs.opencv.org/4.x/d4/d94/tutorial_camera_calibration.html
 *
 * @copyright MIT License (c) 2023 Steven Gong
 *
 */
#ifndef CALIBRATION_HPP_
#define CALIBRATION_HPP_

#include <opencv2/opencv.hpp>
#include <string>

#include "calibrationsettings.hpp"

enum VideoInputType { CAMERA, VIDEO_FILE, IMAGE_FOLDER };

class Calibration {
 private:
  const CalibrationSettings settings;

  std::string input;
  VideoInputType video_input_type;      // The input type, determined from input string
  int cameraID;                         // The camera ID if the input is a camera
  cv::VideoCapture cap;                 // The capture device if the input is a camera or a video
  std::vector<std::string> image_list;  // The list of images if the input is an image folder

  cv::Mat K_left;   // Left Camera Matrix (3x3)
  cv::Mat K_right;  // Right Camera Matrix (3x3)
  cv::Mat D_left;   // Left Distortion Matrix
  cv::Mat D_right;  // Right Distortion Matrix
  cv::Mat R;        // Rotation Matrix
  cv::Mat T;        // Translation Matrix
  cv::Mat E;        // Essential Matrix
  cv::Mat F;        // Fundamental Matrix

  // Stereo rectification
  cv::Mat R1, R2, P1, P2, Q;

  // Compute the undistort and rectify transformation map
  cv::Mat map1Left, map2Left, map1Right, map2Right;

  bool calibration_success;

 public:
  // To use an input camera -> give the ID of the camera, like "1"
  // To use a video  -> give the path of the input video, like "/tmp/x.avi"
  // To use a folder of images   -> give the path to folder, like "/tmp/calibration"
  Calibration(std::string source_path_or_camera_id, CalibrationSettings settings);

  void runCalibration();
  void runCalibrationAndExport();
  void exportCalibrationXML();
};

#endif

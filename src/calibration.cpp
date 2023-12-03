#include "calibration.hpp"

#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

Calibration::Calibration(std::string source_path_or_camera_id, CalibrationSettings settings)
    : settings{settings}, input{source_path_or_camera_id} {
  if (!settings.good_input) {
    std::cerr << "Invalid input detected. Exiting..." << std::endl;
    return;
  }

  if (input[0] >= '0' && input[0] <= '9' && input.length() == 1) {
    cameraID = input[0] - '0';
    video_input_type = VideoInputType::CAMERA;
  } else {
    if ((input.substr(input.length() - 4, 4) == ".mp4" || input.substr(input.length() -4, 4) == ".avi")) {
      video_input_type = VideoInputType::VIDEO_FILE;
    } else {
      video_input_type = VideoInputType::IMAGE_FOLDER;
    }
  }

  if (video_input_type == VideoInputType::CAMERA) {
    cap.open(cameraID);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 2);
  }
  if (video_input_type == VideoInputType::VIDEO_FILE) {
    cap.open(input);
  }
  if (video_input_type != VideoInputType::IMAGE_FOLDER && !cap.isOpened()) {
    std::cerr << "Could not open capture " << input << std::endl;
  }
}

void Calibration::runCalibration() {
  const cv::Size board_size = settings.board_size;
  const float square_size = settings.square_size;
  const int num_stereo_pairs = settings.num_stereo_pairs;

  std::vector<cv::Point3f> objp;
  for (int i = 0; i < board_size.height; ++i) {
    for (int j = 0; j < board_size.width; ++j) {
      objp.push_back(cv::Point3f(j * square_size, i * square_size, 0));
    }
  }

  // Arrays to store object points and image points from all images
  std::vector<std::vector<cv::Point3f>> object_points;
  std::vector<std::vector<cv::Point2f>> img_points_left, img_points_right;
  std::vector<cv::Point2f> corners_left, corners_right;

  cv::Mat raw_stereo_img;

  cap.read(raw_stereo_img);

  int width = raw_stereo_img.cols;
  int height = raw_stereo_img.rows;
  cv::Rect left_img_index = cv::Rect(0, 0, width / 2, height);
  cv::Rect right_img_index = cv::Rect(width / 2, 0, width / 2, height);

  cv::Mat raw_left_img = raw_stereo_img(left_img_index);
  cv::Mat raw_right_img = raw_stereo_img(right_img_index);

  while (object_points.size() < num_stereo_pairs) {
    cap.read(raw_stereo_img);
    if (raw_stereo_img.empty()) {
      std::cerr << "ERROR! blank frame grabbed" << std::endl;
      continue;
    }

    cv::Mat raw_stereo_img_gray;
    cv::cvtColor(raw_stereo_img, raw_stereo_img_gray, cv::COLOR_BGR2GRAY);

    cv::Mat raw_left_img_gray = raw_stereo_img_gray(left_img_index);
    cv::Mat raw_right_img_gray = raw_stereo_img_gray(right_img_index);
    raw_left_img = raw_stereo_img(left_img_index);
    raw_right_img = raw_stereo_img(right_img_index);

    // Find the chess board corners in both images
    bool found_left = cv::findChessboardCorners(raw_left_img_gray, board_size, corners_left);
    bool found_right = cv::findChessboardCorners(raw_right_img_gray, board_size, corners_right);

    if (found_left && found_right) {
      // Refine the corner positions
      cv::cornerSubPix(
          raw_left_img_gray, corners_left, cv::Size(11, 11), cv::Size(-1, -1),
          cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
      cv::cornerSubPix(
          raw_right_img_gray, corners_right, cv::Size(11, 11), cv::Size(-1, -1),
          cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));

      img_points_left.push_back(corners_left);
      img_points_right.push_back(corners_right);
      object_points.push_back(objp);

      cv::drawChessboardCorners(raw_left_img, board_size, corners_left, found_left);
      cv::putText(raw_left_img,
                  "Chessboard Found " + std::to_string(object_points.size()) + " / " +
                      std::to_string(num_stereo_pairs),
                  cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
      cv::putText(raw_left_img, "Press \"Space\" to continue", cv::Point(10, 60),
                  cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
      cv::imshow("Camera Calibration", raw_left_img);
      if (object_points.size() < num_stereo_pairs) {
        cv::waitKey(0);
      }

    } else {
      // cv::imshow("Chessboard not found, please point camera to a chessboard", raw_left_img);
      cv::putText(raw_left_img, "Chessboard Not Found", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                  1, cv::Scalar(0, 0, 255), 2);
      cv::imshow("Camera Calibration", raw_left_img);
      cv::waitKey(100);
    }
  }

  cv::calibrateCamera(object_points, img_points_left, raw_left_img.size(), K_left, D_left,
                      cv::noArray(), cv::noArray(), cv::CALIB_ZERO_TANGENT_DIST);

  cv::calibrateCamera(object_points, img_points_right, raw_right_img.size(), K_right, D_right,
                      cv::noArray(), cv::noArray(), cv::CALIB_ZERO_TANGENT_DIST);

  cv::stereoCalibrate(object_points, img_points_left, img_points_right, K_left, D_left, K_right,
                      D_right, raw_left_img.size(), R, T, E, F);

  cv::stereoRectify(K_left, D_left, K_right, D_right, raw_left_img.size(), R, T, R1, R2, P1, P2, Q);

  cv::initUndistortRectifyMap(K_left, D_left, R1, P1, raw_left_img.size(), CV_16SC2, map1Left,
                              map2Left);
  cv::initUndistortRectifyMap(K_right, D_right, R2, P2, raw_right_img.size(), CV_16SC2, map1Right,
                              map2Right);

  calibration_success = true;
  std::cout << "Stereo calibration and rectification completed!" << std::endl;
}

void Calibration::exportCalibrationXML() {
  if (!calibration_success) {
    std::cerr << "ERROR! Calibration wasn't successful, unable to export\n";
    return;
  }
  // Save the maps for future use
  cv::FileStorage fs(settings.calibration_export_path, cv::FileStorage::WRITE);
  fs << "K_left" << K_left << "K_right" << K_right << "D_left" << D_left << "D_right" << D_right
     << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q << "E" << E << "F" << F
     << "map1Left" << map1Left << "map2Left" << map2Left << "map1Right" << map1Right << "map2Right"
     << map2Right;
  fs.release();
  std::cout << "Exported file to " << settings.calibration_export_path << std::endl;
}

void Calibration::runCalibrationAndExport() {
  runCalibration();
  exportCalibrationXML();
}

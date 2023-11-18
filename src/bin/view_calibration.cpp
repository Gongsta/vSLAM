#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
  // Load the maps for undistortion and rectification
  cv::FileStorage fs("stereo_rectify_maps.xml", cv::FileStorage::READ);

  cv::Mat K_left, D_left, K_right, D_right;
  cv::Mat map1Left, map2Left, map1Right, map2Right;

  fs["K_left"] >> K_left;
  fs["K_right"] >> K_right;
  fs["D_left"] >> D_left;
  fs["D_right"] >> D_right;
  fs["map1Left"] >> map1Left;
  fs["map2Left"] >> map2Left;
  fs["map1Right"] >> map1Right;
  fs["map2Right"] >> map2Right;

  if (map1Left.empty() || map2Left.empty() || map1Right.empty() || map2Right.empty()) {
    std::cerr << "Error: Unable to load maps from XML file." << std::endl;
    return -1;
  }

  // Load new stereo images
  cv::VideoCapture cap(0);

  if (!cap.isOpened()) {
    std::cerr << "ERROR! Unable to open camera\n";
    return -1;
  }

  cv::Mat raw_stereo_img;

  int minDisparity = 0;
  int numDisparities = 16;  // Must be divisible by 16
  int blockSize = 15;        // Odd number
  // cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create(minDisparity, numDisparities, blockSize);
  cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create(numDisparities, blockSize);

  while (true) {
    cap.read(raw_stereo_img);
    int width = raw_stereo_img.cols;
    int height = raw_stereo_img.rows;
    cv::Rect left_img_index = cv::Rect(0, 0, width / 2, height);
    cv::Rect right_img_index = cv::Rect(width / 2, 0, width / 2, height);

    cv::Mat raw_left_img = raw_stereo_img(left_img_index);
    cv::Mat raw_right_img = raw_stereo_img(right_img_index);

    // Undistort and rectify images
    cv::Mat rect_left_img, rect_right_img;
    cv::remap(raw_left_img, rect_left_img, map1Left, map2Left, cv::INTER_LINEAR);
    cv::remap(raw_right_img, rect_right_img, map1Right, map2Right, cv::INTER_LINEAR);

    // Undistort the image
    cv::Mat undist_left_img, undist_right_img;
    cv::undistort(raw_left_img, undist_left_img, K_left, D_left);
    cv::undistort(raw_right_img, undist_right_img, K_right, D_right);
    // Display or save rectified images
    // cv::imshow("Rectified Left Image", undistortedImage);
    // cv::imshow("Rectified Right Image", raw_right_img);
    cv::Mat rect_stereo_img;
    cv::Mat comparison;
    // cv::hconcat(rectifiedLeft, rectifiedRight, rect_stereo_img);
    cv::hconcat(raw_left_img, rect_left_img, comparison);

    cv::Mat disparity;
    // stereo->compute(rect_left_img, rect_right_img, disparity);
    cv::cvtColor(undist_left_img, undist_left_img, cv::COLOR_BGR2GRAY);
    cv::cvtColor(undist_right_img, undist_right_img, cv::COLOR_BGR2GRAY);
    stereo->compute(undist_left_img, undist_right_img, disparity);
    // computeDisparity(frame, outputImg)
    cv::normalize(disparity, disparity, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::imshow("Disparity", disparity);

    // cv::imshow("Rectified Image", comparison);

    if (cv::waitKey(5) >= 0) {
      break;
    }
  }
  return 0;
}

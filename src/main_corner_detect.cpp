#include <iostream>
#include <opencv2/opencv.hpp>

// #include "camera.hpp"
// #include "orb.hpp"

#include "cornerdetector.hpp"

int main() {
  cv::Mat img;
  img = cv::imread("../livingroom.jpg", cv::IMREAD_COLOR);
  cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

  // EdgeDetector edge_detector{EdgeDetectorType::kSobelEdgeX};
  // cv::Mat edge_img = edge_detector.ConvolveImage(img);
  // cv::imwrite("../edge.jpg", edge_img);
  // cv::imshow("display image", edge_img);
  // cv::waitKey(0); // Wait for a keystroke in the window

  CornerDetector corner_detector{CornerDetectorType::kHarrisCorner};
  cv::Mat corner_img = corner_detector.ConvolveImage(img);
  cv::imwrite("../corner.jpg", corner_img);
  cv::imshow("display image", corner_img);
  cv::waitKey(0); // Wait for a keystroke in the window

  return 0;

  // edge_detector.con

  // CornerDetector corner_detector{CornerDetectorType::kSobelOperator};

  // corner_detector.DetectCorners(img);


  // process(image);

  // Extract ORB features
  // returns image with features
  return 0;
}

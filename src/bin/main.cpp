#include <iostream>
#include <opencv2/opencv.hpp>

void computeORB(cv::Ptr<cv::ORB> orb, const cv::Mat& frame, std::vector<cv::KeyPoint>& keypoints,
                cv::Mat& descriptors, cv::Mat& orb_img) {
  orb->detectAndCompute(frame, cv::Mat(), keypoints, descriptors);
  cv::drawKeypoints(frame, keypoints, orb_img);
}

void computeDisparity() {}

// void rectifyImage()

//     void computeDisparity() {
//   if (img_left.empty() || img_right.empty()) {
//     std::cerr << "Error: Images not found!" << std::endl;
//     return -1;
//   }

//   // Create StereoBM object with desired number of disparities and block size
//   int numDisparities = 16;  // Must be divisible by 16
//   int blockSize = 15;       // Typically odd
//   cv::Ptr<cv::StereoBM> stereoBM = cv::StereoBM::create(numDisparities, blockSize);

//   // Compute the disparity map
//   cv::Mat disparity;
//   stereoBM->compute(imgLeft, imgRight, disparity);

//   // Normalize the disparity map for visualization
//   cv::Mat disparityVis;
//   cv::normalize(disparity, disparityVis, 0, 255, cv::NORM_MINMAX, CV_8U);
// }

int main() {
  cv::VideoCapture cap(0);

  if (!cap.isOpened()) {
    std::cerr << "ERROR! Unable to open camera\n";
    return -1;
  }

  cv::Mat raw_stereo_img;

  cv::Ptr<cv::ORB> orb = cv::ORB::create(1000);

  int minDisparity = 0;
  int numDisparities = 64;  // Must be divisible by 16
  int blockSize = 7;        // Odd number
  cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create(minDisparity, numDisparities, blockSize);

  while (true) {
    cap.read(raw_stereo_img);

    if (raw_stereo_img.empty()) {
      std::cerr << "ERROR! blank frame grabbed\n";
      continue;
    }

    int width = raw_stereo_img.cols;
    int height = raw_stereo_img.rows;

    cv::Rect left_img_index = cv::Rect(0, 0, width / 2, height);
    cv::Rect right_img_index = cv::Rect(width / 2, 0, width / 2, height);

    cv::Mat left_img = raw_stereo_img(left_img_index);
    cv::Mat right_img = raw_stereo_img(right_img_index);

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    cv::Mat orb_img;
    computeORB(orb, raw_stereo_img, keypoints, descriptors, orb_img);

    cv::Mat disparity;
    stereo->compute(left_img, right_img, disparity);
    // computeDisparity(frame, outputImg)
    cv::normalize(disparity, disparity, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::imshow("Disparity", disparity);
    // cv::waitKey(0);  // Wait for a key press to close the window

    // cv::imshow("Live", orb_img(right_img_index));
    // Print out the descriptor values
    // std::cout << "Descriptor values: " << descriptors << std::endl;

    if (cv::waitKey(5) >= 0) {
      break;
    }
  }

  return 0;
}

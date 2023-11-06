/**
 * @file cornerdetector.cpp
 * @author Steven Gong (gong.steven@hotmail.com)
 * @brief CornerDetector class implementation
 *
 * @copyright MIT License (c) 2023 Steven Gong
 *
 */
#include "cornerdetector.hpp"

#include <exception>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>

#include "convolution.hpp"
#include "kernels.hpp"
#include "utils.hpp"

EdgeDetector SelectXEdgeDetectorType(CornerDetectorType detector_type) {
  switch (detector_type) {
    case CornerDetectorType::kHarrisCorner:
      EdgeDetector x_edge_detector{EdgeDetectorType::kSobelEdgeX};
      return x_edge_detector;
  }
}

EdgeDetector SelectYEdgeDetectorType(CornerDetectorType detector_type) {
  switch (detector_type) {
    case CornerDetectorType::kHarrisCorner:
      EdgeDetector y_edge_detector{EdgeDetectorType::kSobelEdgeY};
      return y_edge_detector;
  }
}
CornerDetector::CornerDetector(CornerDetectorType detector_type)
    : detector_type_{detector_type},
      x_edge_detector_{SelectXEdgeDetectorType(detector_type)},
      y_edge_detector_{SelectYEdgeDetectorType(detector_type)} {}

cv::Mat CornerDetector::ConvolveImage(const cv::Mat& img) {
  if (img.type() == CV_8UC1) {
    cv::Mat double_img;
    img.convertTo(double_img, CV_64FC1, 1.0 / 255, 0);
    return ConvolveImage(double_img);
  } else if (img.type() != CV_64FC1) {
    throw std::invalid_argument{"Unknown image type, expecting either CV_8UC1 or CV_64FC1"};
  }

  // Convolve image
  cv::Mat Jx = x_edge_detector_.ConvolveImage(img);
  cv::Mat Jy = y_edge_detector_.ConvolveImage(img);

  // // Scale Convolutions
  // Jx *= 10.0;
  // Jy *= 10.0;

  // Calculate structure matrix
  cv::Mat Jx2 = utils::Square(Jx);
  cv::Mat Jy2 = utils::Square(Jy);
  cv::Mat JxJy = Jx.mul(Jy);

  // Box Filter
  cv::Mat sum_Jx = convolution::Convolve3x3(convolution::kBoxKernel, Jx);
  cv::Mat sum_Jy = convolution::Convolve3x3(convolution::kBoxKernel, Jy);
  cv::Mat sum_Jx2 = convolution::Convolve3x3(convolution::kBoxKernel, Jx2);
  cv::Mat sum_JxJy = convolution::Convolve3x3(convolution::kBoxKernel, JxJy);
  cv::Mat sum_Jy2 = convolution::Convolve3x3(convolution::kBoxKernel, Jy2);
  cv::Mat sum_JxJy2 = utils::Square(sum_JxJy);

  cv::Mat img_with_corners;
  img.convertTo(img_with_corners, CV_8UC1, 255, 0);
  // Solve for eigenvalues of structure matrix
  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      std::vector<double> eigenvalues = utils::SolveQuadratic(
          1, -(sum_Jx2.at<double>(i, j) + sum_Jy2.at<double>(i, j)),
          sum_Jx2.at<double>(i, j) * sum_Jy2.at<double>(i, j) - sum_JxJy2.at<double>(i, j));

      if (eigenvalues.size() == 2) {
        // std::cout << eigenvalues[0] << " " << eigenvalues[1] << std::endl;
        if (eigenvalues[0] > 1.0 && eigenvalues[1] > 1.0) {
          // std::cout << eigenvalues[0] << " " << eigenvalues[1] << std::endl;
          img_with_corners.at<uchar>(i, j) = 255;
        }
      }
    }
  }

  return img_with_corners;
}

// std::vector<Position> CornerDetector::DetectCorners(const cv::Mat& img) {
//   std::vector<Position> positions;
//   cv::Mat convolved_img = ConvolveImage(img);
//   uchar* p;
//   for (int i = 0; i < convolved_img.rows; i++) {
//     p = convolved_img.ptr<uchar>(i);
//     for (int j = 0; j < img.cols; j++) {
//       if (p[j] > 1.0) {
//         positions.push_back({i, j});
//       }
//     }
//   }
//   return positions;
// }

std::vector<Position> CornerDetector::DetectCorners(const cv::Mat& img) {
  std::vector<Position> positions;
  cv::Mat convolved_img = ConvolveImage(img);

  const size_t num_threads = std::thread::hardware_concurrency();
  std::vector<std::thread> threads(num_threads);
  std::vector<Position> thread_positions[num_threads];

  auto detect_corners_lambda = [&](const int start_row, const int end_row, std::vector<Position>& local_positions) {
    for (int i = start_row; i < end_row; i++) {
      const uchar* p = convolved_img.ptr<uchar>(i);
      for (int j = 0; j < img.cols; j++) {
        if (p[j] > 1.0) {
          local_positions.push_back({i, j});
        }
      }
    }
  };

  int rows_per_thread = convolved_img.rows / num_threads;
  for (size_t i = 0; i < num_threads; ++i) {
    threads[i] = std::thread(detect_corners_lambda, i * rows_per_thread, (i + 1) * rows_per_thread, std::ref(thread_positions[i]));
  }

  for (auto& t : threads) {
    t.join();
  }

  for (const auto& local_positions : thread_positions) {
    positions.insert(positions.end(), local_positions.begin(), local_positions.end());
  }

  return positions;
}

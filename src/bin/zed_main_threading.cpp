#include <vpi/CUDAInterop.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <sl/Camera.hpp>
#include <thread>

#include "bfmatcher.hpp"
#include "cudaimage.hpp"
#include "depthtopointcloud.hpp"
#include "disparity.hpp"
#include "disparitytodepth.hpp"
#include "imageformatconverter.hpp"
#include "imageresizer.hpp"
#include "orb.hpp"
#include "save_depth.hpp"
#include "stereodisparityparams.hpp"
#include "vpi_utils.hpp"
#include "zed_utils.hpp"

// VPI/CUDA
uint64_t backends = VPI_BACKEND_CUDA;
VPIImage left_img_raw;
VPIImage right_img_raw;

cudaStream_t left_stream_cuda;
cudaStream_t right_stream_cuda;
VPIStream left_stream;
VPIStream right_stream;

// Threading setup
std::mutex queueMutex;
sl::Camera zed;
std::deque<cv::Mat> stereo_image_queue;
std::deque<cv::Mat> cv_left_img_queue;
std::deque<cv::Mat> cv_depth_queue;
// std::deque<std::vector<cv::Keypoint>> cv_depth_queue;

std::atomic<bool> running(true);
std::condition_variable queueCondVar;
std::condition_variable solverCondVar;

// OpenCV indices
cv::Rect left_img_index;
cv::Rect right_img_index;

// Disparity
cv::Mat cv_disparity_color;
cv::Mat cv_confidence;
cv::Mat cv_depth;

// Camera Parameters
float cx;
float cy;
float fx;
float baseline;

auto display_start = std::chrono::high_resolution_clock::now();
auto orb_start = std::chrono::high_resolution_clock::now();
auto disparity_start = std::chrono::high_resolution_clock::now();

void captureThreadFunction(sl::Camera& zed) {
  while (running) {
    sl::Mat zed_stereo_img;
    if (zed.grab() == sl::ERROR_CODE::SUCCESS) {
      zed.retrieveImage(zed_stereo_img, sl::VIEW::SIDE_BY_SIDE, sl::MEM::CPU);
      cv::Mat cv_stereo_img =
          sl::slMat2cvMat(zed_stereo_img).clone();  // Ensure deep copy for safe threading
      {
        std::lock_guard<std::mutex> lock(queueMutex);
        stereo_image_queue.push_back(cv_stereo_img);
        std::cout << "Queue size: " << stereo_image_queue.size() << std::endl;
      }
      queueCondVar.notify_all();
    }
  }
}

void ORBThreadFunction(ORBFeatureDetector& orb_t_1, ORBFeatureDetector& orb_t,
                       BruteForceMatcher& bfmatcher) {
  cv::namedWindow("ORB", cv::WINDOW_NORMAL);
  cv::resizeWindow("ORB", 672, 376);
  while (running) {
    std::unique_lock<std::mutex> lock(queueMutex);
    queueCondVar.wait(lock, [] { return !stereo_image_queue.empty() || !running; });

    if (!running && stereo_image_queue.empty()) {
      break;
    }

    cv::Mat cv_stereo_img = stereo_image_queue.front();
    lock.unlock();  // Unlock as soon as the shared resource is no longer needed.
    cv::Mat cv_img_out;
    std::vector<cv::DMatch> cvMatches;
    cv::Mat cv_left_img = cv_stereo_img(left_img_index);
    cv_left_img_queue.push_back(cv_left_img);

    if (cv_left_img_queue.size() == 2) {
      // t and t-1 keypoints
      std::vector<cv::KeyPoint> cvkeypoints_t;
      std::vector<cv::KeyPoint> cvkeypoints_t_1;
      cv::Mat cv_img_t_1 = cv_left_img_queue.front();
      cv_left_img_queue.pop_front();
      cv::Mat cv_img_t = cv_left_img_queue.front();
      // cv_img_t.pop_front(); // Uncomment if you want to run ORB at a slower rate

      std::pair<VPIArray&, VPIArray&> orb_results_t_1 =
          orb_t_1.Apply(cv_img_t_1, cv_img_out, cvkeypoints_t_1);
      std::pair<VPIArray&, VPIArray&> orb_results_t =
          orb_t.Apply(cv_img_t, cv_img_out, cvkeypoints_t);
      VPIArray& descriptors_t_1 = orb_results_t_1.second;
      VPIArray& descriptors_t = orb_results_t.second;
      bfmatcher.Apply(descriptors_t, cv_img_t, cvkeypoints_t, descriptors_t_1, cv_img_t_1,
                      cvkeypoints_t_1, cv_img_out, cvMatches);
      auto curr = std::chrono::high_resolution_clock::now();
      auto latency =
          std::chrono::duration_cast<std::chrono::milliseconds>(curr - orb_start).count();
      std::cout << "ORB Frequency: " << 1000.0 / latency << " Hz" << std::endl;
      orb_start = curr;
      cv::imshow("ORB", cv_img_out);
    }
    if (cv::waitKey(1) == 'q') {
      running = false;
    }
  }
}

void solverThread() {
  // Input
  // cv::namedWindow("ORB", cv::WINDOW_NORMAL);
  // cv::resizeWindow("ORB", 672, 376);
  // while (running) {
  //   std::unique_lock<std::mutex> lock(solver_mutex);
  //   solverCondVar.wait(lock, [] { return !stereo_image_queue.empty() || !running; });
  // }

  // std::vector<cv::KeyPoint> cvkeypoints_t;
  // std::vector<cv::KeyPoint> cvkeypoints_t_1;
  // // For each point , get the
  // Eigen::Matrix3d P;
  // P << 1, 0, 0, 0, 1, 0, 0, 0, 1;

  // Eigen::MatrixXd F1, F2;  // 2D points
  // Eigen::MatrixXd W1, W2;  // 3D points

  // F1.conservativeResize(cvMatches.size(), 2);

  // std::vector<cv::KeyPoint> matched_cvkeypoints_t;
  // std::vector<cv::KeyPoint> matched_cvkeypoints_t_1;
  // for (cv::DMatch& match : cvMatches) {
  //   // create set of matched keypoints
  //   matched_cvkeypoints_t_1.push_back(cvkeypoints_t_1[match.queryIdx]);
  //   matched_cvkeypoints_t.push_back(cvkeypoints_t[match.trainIdx]);
  // }
  // // Get the 3D point from the disparity map
  // for (cv::KeyPoint& keypoint : matched_cvkeypoints_t_1) {
  //   // Get the depth value
  //   Eigen::Matrix4d Q;

  //   double z = cv_depth.at<float>(keypoint.pt);
  //   Q << 1, 0, 0, -cx,             // NOLINT
  //       0, 1, 0, -cy,              // NOLINT
  //       0, 0, 0, -fx,              // NOLINT
  //       0, 0, -1.0 / baseline, 0;  // NOLINT

  //   Eigen::Vector4d 2d_point;
  //   point << keypoint.pt.x, keypoint.pt.y, depth, 1;
  //   Eigen::Vector4d 3d_homogeneous = Q * point;
  // }
}

void DisparityThreadFunction(DisparityEstimator& disparity_estimator,
                             ImageFormatConverter& disparity_converter,
                             DisparityToDepthConverter& disparity_to_depth) {
  cv::namedWindow("Disparity", cv::WINDOW_AUTOSIZE);
  while (running) {
    std::unique_lock<std::mutex> lock(queueMutex);
    queueCondVar.wait(lock, [] { return !stereo_image_queue.empty() || !running; });

    if (!running && stereo_image_queue.empty()) {
      break;
    }
    cv::Mat raw_stereo_img = stereo_image_queue.front();
    stereo_image_queue.pop_front();

    cv::Mat cv_img_left = raw_stereo_img(left_img_index);
    cv::Mat cv_img_right = raw_stereo_img(right_img_index);
    lock.unlock();  // Unlock as soon as the shared resource is no longer needed.

    std::pair<VPIImage&, VPIImage&> disparity_output = disparity_estimator.Apply(
        left_stream, cv_img_left, cv_img_right, cv_disparity_color, cv_confidence);
    VPIImage& disparity_map = disparity_output.first;
    VPIImage& confidence_map = disparity_output.second;

    VPIImage& disparity_map_f32 = disparity_converter.Apply(left_stream, disparity_map);
    VPIImage& depth_map = disparity_to_depth.Apply(left_stream_cuda, disparity_map_f32, cv_depth);

    auto curr = std::chrono::high_resolution_clock::now();
    auto latency =
        std::chrono::duration_cast<std::chrono::milliseconds>(curr - disparity_start).count();
    std::cout << "Disparity Frequency: " << 1000.0 / latency << " Hz" << std::endl;
    cv::imshow("Disparity", cv_disparity_color);
    cv::imshow("depth", cv_depth);
    disparity_start = curr;
    if (cv::waitKey(1) == 'q') {
      running = false;
    }
    cv_depth_queue.push_back(cv_depth);
  }
}

void displayThreadFunction() {
  cv::namedWindow("Image", cv::WINDOW_NORMAL);
  cv::resizeWindow("Image", 672, 376);
  while (running) {
    std::unique_lock<std::mutex> lock(queueMutex);
    queueCondVar.wait(lock, [] { return !stereo_image_queue.empty() || !running; });

    if (!running && stereo_image_queue.empty()) {
      break;
    }

    cv::Mat imgToDisplay = stereo_image_queue.front();
    stereo_image_queue.pop_front();
    lock.unlock();  // Unlock as soon as the shared resource is no longer needed.

    auto curr = std::chrono::high_resolution_clock::now();
    auto latency =
        std::chrono::duration_cast<std::chrono::milliseconds>(curr - display_start).count();
    std::cout << "Frequency: " << 1000.0 / latency << " Hz" << std::endl;
    std::cout << "Zed Frequency: " << zed.getCurrentFPS() << std::endl;
    cv::imshow("Image", imgToDisplay);
    display_start = curr;
    if (cv::waitKey(1) == 'q') {
      running = false;
    }
  }
}

int main() {
  sl::InitParameters init_params;
  init_params.camera_resolution = sl::RESOLUTION::VGA;
  init_params.camera_fps = 100;
  // init_params.camera_resolution = sl::RESOLUTION::SVGA;
  // init_params.camera_fps = 120;
  init_params.depth_mode = sl::DEPTH_MODE::NONE;
  init_params.coordinate_units = sl::UNIT::METER;

  sl::ERROR_CODE err = zed.open(init_params);
  if (err != sl::ERROR_CODE::SUCCESS) {
    std::cout << toString(err) << std::endl;
    zed.close();
    return 1;
  }

  sl::Resolution image_size = zed.getCameraInformation().camera_configuration.resolution;
  cx = zed.getCameraInformation().camera_configuration.calibration_parameters.left_cam.cx;
  cy = zed.getCameraInformation().camera_configuration.calibration_parameters.left_cam.cy;
  fx = zed.getCameraInformation().camera_configuration.calibration_parameters.left_cam.fx;
  baseline =
      zed.getCameraInformation().camera_configuration.calibration_parameters.getCameraBaseline();
  std::cout << "fx: " << fx << " baseline: " << baseline << std::endl;
  sl::Mat zed_img_left(image_size.width, image_size.height, sl::MAT_TYPE::U8_C4);
  left_img_index = cv::Rect(0, 0, image_size.width, image_size.height);
  right_img_index = cv::Rect(image_size.width, 0, image_size.width, image_size.height);

  while (zed.grab() != sl::ERROR_CODE::SUCCESS) {
  }  // wait until first zed frame is ready
  zed.retrieveImage(zed_img_left, sl::VIEW::LEFT, sl::MEM::CPU);
  cv::Mat cv_img_left = sl::slMat2cvMat(zed_img_left);

  VPIStream stream;
  CHECK_STATUS(vpiStreamCreate(0, &stream));
  cudaStreamCreate(&left_stream_cuda);
  // cudaStreamCreate(&right_stream_cuda);
  CHECK_STATUS(vpiStreamCreateWrapperCUDA(left_stream_cuda, VPI_BACKEND_CUDA | VPI_BACKEND_VIC,
                                          &left_stream));
  // Params
  StereoDisparityParams params{backends};
  params.input_height = image_size.height;
  params.input_width = image_size.width;
  params.output_height = image_size.height;
  params.output_width = image_size.width;
  DisparityEstimator disparity_estimator{params};
  ImageFormatConverter disparity_converter{params.output_width, params.output_height,
                                           params.conv_params, VPI_IMAGE_FORMAT_F32,
                                           VPI_BACKEND_CUDA};
  DisparityToDepthConverter disparity_to_depth{params.output_width, params.output_height, fx,
                                               baseline, VPI_IMAGE_FORMAT_F32};
  ORBFeatureDetector orb_t{cv_img_left, stream, backends};
  ORBFeatureDetector orb_t_1{cv_img_left, stream, backends};
  BruteForceMatcher bfmatcher{stream, orb_t.out_capacity};

  // Threads
  std::thread captureThread(captureThreadFunction, std::ref(zed));
  std::thread displayThread(displayThreadFunction);
  std::thread ORBThread(ORBThreadFunction, std::ref(orb_t_1), std::ref(orb_t), std::ref(bfmatcher));
  std::thread depthThread(DisparityThreadFunction, std::ref(disparity_estimator),
                          std::ref(disparity_converter), std::ref(disparity_to_depth));

  // Wait for threads to finish
  captureThread.join();
  displayThread.join();
  ORBThread.join();
  depthThread.join();

  zed.close();
  vpiStreamDestroy(stream);
  return 0;
}

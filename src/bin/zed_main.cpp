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

std::atomic<bool> running(true);
std::atomic<bool> visualize(true);

// VPI/CUDA
uint64_t backends = VPI_BACKEND_CUDA;
VPIImage left_img_raw;
VPIImage right_img_raw;

VPIStream stream;
VPIStream left_stream;
VPIStream right_stream;
cudaStream_t left_stream_cuda;
cudaStream_t right_stream_cuda;

//  Images
sl::Mat zed_stereo_img;
cv::Mat cv_stereo_img;
cv::Rect left_img_index;
cv::Rect right_img_index;

// Disparity
cv::Mat cv_disparity_color;
cv::Mat cv_confidence;
cv::Mat cv_depth;

// Queues of at most 2 images for storing t-1 and t image
std::deque<cv::Mat> cv_left_img_queue;
std::deque<cv::Mat> cv_depth_queue;

// Zed and Camera Parameters
sl::Camera zed;
sl::Resolution image_size;
float cx;
float cy;
float fx;
float baseline;
Eigen::Matrix4d Q;

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

  // Setup Zed Camera Parameters
  image_size = zed.getCameraInformation().camera_configuration.resolution;
  cx = zed.getCameraInformation().camera_configuration.calibration_parameters.left_cam.cx;
  cy = zed.getCameraInformation().camera_configuration.calibration_parameters.left_cam.cy;
  fx = zed.getCameraInformation().camera_configuration.calibration_parameters.left_cam.fx;
  baseline =
      zed.getCameraInformation().camera_configuration.calibration_parameters.getCameraBaseline();
  std::cout << "fx: " << fx << " baseline: " << baseline << std::endl;
  left_img_index = cv::Rect(0, 0, image_size.width, image_size.height);
  right_img_index = cv::Rect(image_size.width, 0, image_size.width, image_size.height);

  zed_stereo_img = sl::Mat(image_size.width, image_size.height, sl::MAT_TYPE::U8_C4);
  cv_stereo_img = sl::slMat2cvMat(zed_stereo_img).clone();  // Ensure deep copy for safe threading
  cv::Mat cv_img_left = cv_stereo_img(left_img_index);

  Q << 1, 0, 0, -cx,             // NOLINT
      0, 1, 0, -cy,              // NOLINT
      0, 0, 0, -fx,              // NOLINT
      0, 0, -1.0 / baseline, 0;  // NOLINT
  // VPI Params
  StereoDisparityParams params{backends};
  params.input_height = image_size.height;
  params.input_width = image_size.width;
  params.output_height = image_size.height;
  params.output_width = image_size.width;

  // Streams
  CHECK_STATUS(vpiStreamCreate(0, &stream));
  cudaStreamCreate(&left_stream_cuda);
  // cudaStreamCreate(&right_stream_cuda);
  CHECK_STATUS(vpiStreamCreateWrapperCUDA(left_stream_cuda, VPI_BACKEND_CUDA | VPI_BACKEND_VIC,
                                          &left_stream));

  // Estimators
  DisparityEstimator disparity_estimator{params};
  ImageFormatConverter disparity_converter{params.output_width, params.output_height,
                                           params.conv_params, VPI_IMAGE_FORMAT_F32,
                                           VPI_BACKEND_CUDA};
  DisparityToDepthConverter disparity_to_depth{params.output_width, params.output_height, fx,
                                               baseline, VPI_IMAGE_FORMAT_F32};
  ORBFeatureDetector orb_t{cv_img_left, stream, backends};
  ORBFeatureDetector orb_t_1{cv_img_left, stream, backends};
  BruteForceMatcher bfmatcher{stream, orb_t.out_capacity};

  while (running) {
    if (zed.grab() == sl::ERROR_CODE::SUCCESS) {
      auto start = std::chrono::high_resolution_clock::now();
      /*-----------CAPTURE--------------*/
      zed.retrieveImage(zed_stereo_img, sl::VIEW::SIDE_BY_SIDE, sl::MEM::CPU);
      cv_stereo_img =
          sl::slMat2cvMat(zed_stereo_img).clone();  // Ensure deep copy for safe threading
      if (visualize) {
        cv::imshow("Image", cv_stereo_img);
      }

      /*-----------DISPARITY--------------*/
      cv::Mat cv_img_left = cv_stereo_img(left_img_index);
      cv::Mat cv_img_right = cv_stereo_img(right_img_index);

      std::pair<VPIImage&, VPIImage&> disparity_output = disparity_estimator.Apply(
          left_stream, cv_img_left, cv_img_right, cv_disparity_color, cv_confidence);
      VPIImage& disparity_map = disparity_output.first;
      VPIImage& confidence_map = disparity_output.second;

      VPIImage& disparity_map_f32 = disparity_converter.Apply(left_stream, disparity_map);
      VPIImage& depth_map = disparity_to_depth.Apply(left_stream_cuda, disparity_map_f32, cv_depth);

      cv_depth_queue.push_back(cv_depth);

      if (visualize) {
        cv::imshow("Disparity", cv_disparity_color);
        cv::imshow("depth", cv_depth);
      }

      /*-----------ORB--------------*/
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

        std::pair<VPIArray&, VPIArray&> orb_results_t_1 =
            orb_t_1.Apply(cv_img_t_1, cv_img_out, cvkeypoints_t_1);
        std::pair<VPIArray&, VPIArray&> orb_results_t =
            orb_t.Apply(cv_img_t, cv_img_out, cvkeypoints_t);
        VPIArray& descriptors_t_1 = orb_results_t_1.second;
        VPIArray& descriptors_t = orb_results_t.second;
        bfmatcher.Apply(descriptors_t, cv_img_t, cvkeypoints_t, descriptors_t_1, cv_img_t_1,
                        cvkeypoints_t_1, cv_img_out, cvMatches);
        if (visualize) {
          cv::imshow("ORB", cv_img_out);
        }

        /*-----------Visual Odometry Solver--------------*/
        Eigen::MatrixXd F1, F2;  // 2D points
        Eigen::MatrixXd W1, W2;  // 3D points

        F1.conservativeResize(cvMatches.size(), 2);

        std::vector<cv::KeyPoint> matched_cvkeypoints_t_1;
        std::vector<cv::KeyPoint> matched_cvkeypoints_t;
        std::cout << "Matches: " << cvMatches.size() << std::endl;
        for (cv::DMatch& match : cvMatches) {
          // create set of matched keypoints
          matched_cvkeypoints_t_1.push_back(cvkeypoints_t_1[match.queryIdx]);
          matched_cvkeypoints_t.push_back(cvkeypoints_t[match.trainIdx]);
        }

        cv::Mat cv_depth_t_1 = cv_depth_queue.front();
        cv_depth_queue.pop_front();
        cv::Mat cv_depth_t = cv_depth_queue.front();

        // Get the 3D point from the depth map
        for (cv::KeyPoint& keypoint : matched_cvkeypoints_t_1) {
          double z = cv_depth_t_1.at<float>(keypoint.pt);
          Eigen::Vector4d point_2d;
          point_2d << keypoint.pt.x, keypoint.pt.y, z, 1;
          Eigen::Vector4d homogeneous_3d = Q * point_2d;
        }
        for (cv::KeyPoint& keypoint : matched_cvkeypoints_t) {
          Eigen::Matrix4d Q;
          double z = cv_depth_t.at<float>(keypoint.pt);
          Eigen::Vector4d point_2d;
          point_2d << keypoint.pt.x, keypoint.pt.y, z, 1;
          Eigen::Vector4d homogeneous_3d = Q * point_2d;
        }
      }

      auto curr = std::chrono::high_resolution_clock::now();
      auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(curr - start).count();
      std::cout << "Frequency: " << 1000.0 / latency << " Hz" << std::endl;
      start = curr;
      if (visualize) {
        if (cv::waitKey(1) == 'q') {
          running = false;
        }
      }
    }
  }

  zed.close();
  return 0;
}

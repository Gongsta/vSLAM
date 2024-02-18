#include <vpi/CUDAInterop.h>

#include <chrono>
#include <sl/Camera.hpp>

#include "cudaimage.hpp"
#include "depthtopointcloud.hpp"
#include "disparity.hpp"
#include "disparitytodepth.hpp"
#include "imageformatconverter.hpp"
#include "imageresizer.hpp"
#include "stereodisparityparams.hpp"
#include "zed_utils.hpp"

int main() {
  int retval = 0;

  sl::Camera zed;
  sl::InitParameters init_params;
  init_params.camera_resolution = sl::RESOLUTION::SVGA;
  init_params.camera_fps = 120;
  init_params.depth_mode = sl::DEPTH_MODE::NONE;
  init_params.coordinate_units = sl::UNIT::METER;

  sl::ERROR_CODE err = zed.open(init_params);
  if (!zed.getDeviceList().empty()) {
    std::cout << "Loading Zed with serial number" << zed.getDeviceList()[0].serial_number
              << std::endl;
  }
  if (err != sl::ERROR_CODE::SUCCESS) {
    printf("%s\n", toString(err).c_str());
    zed.close();
    return 1;
  }

  // Get Zed Camera Information
  sl::Resolution image_size = zed.getCameraInformation().camera_configuration.resolution;
  int img_width = image_size.width;
  int img_height = image_size.height;
  float fx = zed.getCameraInformation().camera_configuration.calibration_parameters.left_cam.fx;
  float baseline =
      zed.getCameraInformation().camera_configuration.calibration_parameters.getCameraBaseline();
  std::cout << "fx: " << fx << " baseline: " << baseline << std::endl;

  // Mapping sl::Mat to cv::Mat
  sl::Mat zed_img_left(image_size.width, image_size.height, sl::MAT_TYPE::U8_C4);
  cv::Mat cv_img_left = sl::slMat2cvMat(zed_img_left);
  sl::Mat zed_img_right(image_size.width, image_size.height, sl::MAT_TYPE::U8_C4);
  cv::Mat cv_img_right = sl::slMat2cvMat(zed_img_right);

  cv::Mat cv_disparity_color, cv_confidence, cv_depth;

  // uint64_t backends = VPI_BACKEND_OFA | VPI_BACKEND_PVA | VPI_BACKEND_VIC;
  uint64_t backends = VPI_BACKEND_CUDA;

  cudaStream_t left_stream_cuda;
  cudaStream_t right_stream_cuda;
  VPIStream left_stream;
  VPIStream right_stream;

  StereoDisparityParams params{backends};

  try {
    cudaStreamCreate(&left_stream_cuda);
    // cudaStreamCreate(&right_stream_cuda);
    CHECK_STATUS(vpiStreamCreateWrapperCUDA(left_stream_cuda, VPI_BACKEND_CUDA | VPI_BACKEND_VIC,
                                            &left_stream));

    // Create classes that allocate the memory of the images
    // Rectification is done by ZED already
    DisparityEstimator disparity{params};  // format is given by params.disparity_format

    ImageFormatConverter disparity_converter{params.output_width, params.output_height,
                                             params.conv_params, VPI_IMAGE_FORMAT_F32,
                                             VPI_BACKEND_CUDA};
    DisparityToDepthConverter disparity_to_depth{params.output_width, params.output_height, fx,
                                                 baseline, VPI_IMAGE_FORMAT_F32};

    DepthToPointCloudConverter depth_to_pointcloud{params.output_width, params.output_height};

    auto start = std::chrono::system_clock::now();
    int counter = 0;
    while (true) {
      // Get Images
      if (zed.grab() == sl::ERROR_CODE::SUCCESS) {
        counter++;
        counter %= 10;
        zed.retrieveImage(zed_img_left, sl::VIEW::LEFT, sl::MEM::CPU);
        zed.retrieveImage(zed_img_right, sl::VIEW::RIGHT, sl::MEM::CPU);

        std::pair<VPIImage&, VPIImage&> disparity_output = disparity.Apply(
            left_stream, cv_img_left, cv_img_right, cv_disparity_color, cv_confidence);
        VPIImage& disparity_map = disparity_output.first;
        VPIImage& confidence_map = disparity_output.second;

        // VPIImage& disparity_map_f32 = disparity_converter.Apply(left_stream, disparity_map);

        CHECK_STATUS(vpiStreamSync(left_stream));

        // VPIImage& depth_map =
        //     disparity_to_depth.Apply(left_stream_cuda, disparity_map_f32, cv_depth);

        auto end = std::chrono::system_clock::now();
        auto frequency = 1000.0 / std::chrono::duration<double, std::milli>(end - start).count();
        cv::imshow("Image", cv_img_left);
        cv::imshow("disparity", cv_disparity_color);
        // cv::imshow("depth", cv_depth);
        std::cout << frequency << " hz" << std::endl;
        start = end;
        // cv::imwrite("depth.jpg", cv_depth);
        // std::cout << cv_depth << std::endl;
        // break;

        if (cv::waitKey(5) >= 0) {
          break;
        }
      }
    }
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    retval = 1;
  }

  // ========
  // Clean up
  vpiStreamDestroy(left_stream);
  // vpiStreamDestroy(right_stream);

  return retval;
}

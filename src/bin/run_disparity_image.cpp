#include <vpi/CUDAInterop.h>
#include <vpi/Stream.h>

#include "cudaimage.hpp"
#include "depthtopointcloud.hpp"
#include "disparity.hpp"
#include "disparitytodepth.hpp"
#include "imageformatconverter.hpp"
#include "imageresizer.hpp"
#include "stereodisparityparams.hpp"
#include "vpi_utils.hpp"

int main() {
  int retval = 0;
  cv::Mat raw_stereo_img = cv::imread("../small_stereo.jpeg");
  int raw_stereo_width = raw_stereo_img.cols;
  int raw_stereo_height = raw_stereo_img.rows;

  std::cout << "raw_stereo_width: " << raw_stereo_width
            << " raw_stereo_height: " << raw_stereo_height << std::endl;
  cv::Rect left_img_index = cv::Rect(0, 0, raw_stereo_width / 2, raw_stereo_height);
  cv::Rect right_img_index =
      cv::Rect(raw_stereo_width / 2, 0, raw_stereo_width / 2, raw_stereo_height);

  cv::Mat cv_img_left = raw_stereo_img(left_img_index);
  cv::Mat cv_img_right = raw_stereo_img(right_img_index);

  int img_width = cv_img_left.cols;
  int img_height = cv_img_left.rows;

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
    // CHECK_STATUS(vpiStreamCreateWrapperCUDA(right_stream_cuda, VPI_BACKEND_CUDA |
    // VPI_BACKEND_VIC,
    //                                         &right_stream));

    // Create classes that allocate the memory of the images
    // ImageRectifier left_rectifier{img_height, img_width, VPI_IMAGE_FORMAT_U8};
    // ImageRectifier right_rectifier{img_height, img_width, VPI_IMAGE_FORMAT_U8};
    ImageFormatConverter left_converter{img_width, img_height, params.conv_params,
                                        VPI_IMAGE_FORMAT_Y16_ER, VPI_BACKEND_CUDA};
    ImageFormatConverter right_converter{img_width, img_height, params.conv_params,
                                         VPI_IMAGE_FORMAT_Y16_ER, VPI_BACKEND_CUDA};
    ImageResizer left_resizer{params.input_width, params.input_height, params.stereo_format,
                              VPI_BACKEND_VIC};
    ImageResizer right_resizer{params.input_width, params.input_height, params.stereo_format,
                               VPI_BACKEND_VIC};
    DisparityEstimator disparity{params};

    ImageFormatConverter disparity_converter{params.output_width, params.output_height,
                                             params.conv_params, VPI_IMAGE_FORMAT_F32,
                                             VPI_BACKEND_CUDA};

    DisparityToDepthConverter disparity_to_depth{
        params.output_width, params.output_height, 300, 0.05,
        VPI_IMAGE_FORMAT_F32};  // Mock baseline and fx values

    DepthToPointCloudConverter depth_to_pointcloud{params.output_width, params.output_height};

    while (true) {
      // Get Images
      VPIImage left_img_raw;
      VPIImage right_img_raw;
      cv_img_left = raw_stereo_img(left_img_index);
      cv_img_right = raw_stereo_img(right_img_index);
      CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(cv_img_left, 0, &left_img_raw));
      CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(cv_img_right, 0, &right_img_raw));

      std::cout << "wrapper completed" << std::endl;

      // VPIImage& left_img_rect = left_rectifier.Apply(left_stream, left_img_raw);
      VPIImage& left_img_rect_gray = left_converter.Apply(left_stream, left_img_raw);
      VPIImage& left_img_rect_gray_resize = left_resizer.Apply(left_stream, left_img_rect_gray);

      // VPIImage& right_img_rect = right_rectifier.Apply(right_stream, right_img_raw);
      VPIImage& right_img_rect_gray = right_converter.Apply(left_stream, right_img_raw);
      VPIImage& right_img_rect_gray_resize = right_resizer.Apply(left_stream, right_img_rect_gray);

      std::pair<VPIImage&, VPIImage&> disparity_output =
          disparity.Apply(left_stream, left_img_rect_gray_resize, right_img_rect_gray_resize,
                          cv_disparity_color, cv_confidence);
      VPIImage& disparity_map = disparity_output.first;
      VPIImage& confidence_map = disparity_output.second;

      VPIImage& disparity_map_f32 = disparity_converter.Apply(left_stream, disparity_map);

      // Sync left and right stream
      CHECK_STATUS(vpiStreamSync(left_stream));
      // CHECK_STATUS(vpiStreamSync(right_stream));

      VPIImage& depth_map = disparity_to_depth.Apply(left_stream_cuda, disparity_map_f32, cv_depth);

      cv::imwrite("depth.png", cv_depth);

      // Pointcloud & depth_to_pointcloud.Apply(left_stream,depth_map);

      // disparity.disparity

      // imshow("disparity", cv_disparity_color);
      // imwrite("disparity.png", cv_disparity_color);
      // imshow("confidence", cv_confidence);
      vpiImageDestroy(left_img_raw);
      vpiImageDestroy(right_img_raw);

      std::cout << "loop completed" << std::endl;

      if (cv::waitKey(5) >= 0) {
        break;
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

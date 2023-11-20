#include "disparity.hpp"
#include "imageformatconverter.hpp"
#include "imageresizer.hpp"
#include "stereodisparityparams.hpp"
#include "vpi_utils.hpp"

int main() {
  int retval = 0;
  cv::VideoCapture cap(0, cv::CAP_V4L2);
  cv::Mat raw_stereo_img;

  cap.read(raw_stereo_img);

  int raw_stereo_width = raw_stereo_img.cols;
  int raw_stereo_height = raw_stereo_img.rows;
  cv::Rect left_img_index = cv::Rect(0, 0, raw_stereo_width / 2, raw_stereo_height);
  cv::Rect right_img_index =
      cv::Rect(raw_stereo_width / 2, 0, raw_stereo_width / 2, raw_stereo_height);

  cv::Mat cv_img_left = raw_stereo_img(left_img_index);
  cv::Mat cv_img_right = raw_stereo_img(right_img_index);

  int img_width = cv_img_left.cols;
  int img_height = cv_img_left.rows;

  cv::Mat cv_disparity_color, cv_confidence;

  uint64_t backends = VPI_BACKEND_OFA | VPI_BACKEND_PVA | VPI_BACKEND_VIC;
  // uint64_t backends = VPI_BACKEND_CUDA;
  VPIStream left_stream, right_stream;

  StereoDisparityParams params{backends};

  try {
    CHECK_STATUS(vpiStreamCreate(0, &left_stream));
    CHECK_STATUS(vpiStreamCreate(0, &right_stream));

    // Create classes that allocate the memory of the images
    // ImageRectifier left_rectifier{img_height, img_width, VPI_IMAGE_FORMAT_U8};
    // ImageRectifier right_rectifier{img_height, img_width, VPI_IMAGE_FORMAT_U8};
    ImageFormatConverter left_converter{img_width, img_height, params.conv_params,
                                        VPI_IMAGE_FORMAT_Y16_ER, VPI_BACKEND_CUDA};
    ImageFormatConverter right_converter{img_width, img_height, params.conv_params,
                                         VPI_IMAGE_FORMAT_Y16_ER, VPI_BACKEND_CUDA};
    ImageResizer left_resizer{params.input_width, params.input_height, params.stereo_format, VPI_BACKEND_VIC};
    ImageResizer right_resizer{params.input_width, params.input_height, params.stereo_format, VPI_BACKEND_VIC};
    DisparityEstimator disparity{params};

    // Get Images
    VPIImage left_img_raw;
    VPIImage right_img_raw;

    while (true) {
      cap.read(raw_stereo_img);
      cv_img_left = raw_stereo_img(left_img_index);
      cv_img_right = raw_stereo_img(right_img_index);
      CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(cv_img_left, 0, &left_img_raw));
      CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(cv_img_right, 0, &right_img_raw));

      // VPIImage& left_img_rect = left_rectifier.Apply(left_stream, left_img_raw);
      VPIImage& left_img_rect_gray = left_converter.Apply(left_stream, left_img_raw);
      VPIImage& left_img_rect_gray_resize = left_resizer.Apply(left_stream, left_img_rect_gray);

      // VPIImage& right_img_rect = right_rectifier.Apply(right_stream, right_img_raw);
      VPIImage& right_img_rect_gray = right_converter.Apply(left_stream, right_img_raw);
      VPIImage& right_img_rect_gray_resize = right_resizer.Apply(left_stream,
      right_img_rect_gray);

      // // Sync left and right stream
      // CHECK_STATUS(vpiStreamSync(left_stream));
      // CHECK_STATUS(vpiStreamSync(right_stream));

      disparity.Apply(left_stream, left_img_rect_gray_resize, right_img_rect_gray_resize, cv_disparity_color,
                      cv_confidence);

      imshow("disparity", cv_disparity_color);
      // imshow("confidence", cv_confidence);
      vpiImageDestroy(left_img_raw);
      vpiImageDestroy(right_img_raw);

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
  vpiStreamDestroy(right_stream);

  return retval;
}

#include "zed_utils.hpp"

namespace sl {

// Mapping between MAT_TYPE and CV_TYPE
int getOCVtype(sl::MAT_TYPE type) {
  int cv_type = -1;
  switch (type) {
    case MAT_TYPE::F32_C1:
      cv_type = CV_32FC1;
      break;
    case MAT_TYPE::F32_C2:
      cv_type = CV_32FC2;
      break;
    case MAT_TYPE::F32_C3:
      cv_type = CV_32FC3;
      break;
    case MAT_TYPE::F32_C4:
      cv_type = CV_32FC4;
      break;
    case MAT_TYPE::U8_C1:
      cv_type = CV_8UC1;
      break;
    case MAT_TYPE::U8_C2:
      cv_type = CV_8UC2;
      break;
    case MAT_TYPE::U8_C3:
      cv_type = CV_8UC3;
      break;
    case MAT_TYPE::U8_C4:
      cv_type = CV_8UC4;
      break;
    default:
      break;
  }
  return cv_type;
}

/**
 * Conversion function between sl::Mat and cv::Mat
 **/
cv::Mat slMat2cvMat(Mat& input) {
  // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat
  // (getPtr<T>()) cv::Mat and sl::Mat will share a single memory structure
  return cv::Mat(input.getHeight(), input.getWidth(), getOCVtype(input.getDataType()),
                 input.getPtr<sl::uchar1>(MEM::CPU), input.getStepBytes(sl::MEM::CPU));
}

#ifdef HAVE_CUDA
/**
 * Conversion function between sl::Mat and cv::Mat
 **/
cv::cuda::GpuMat slMat2cvMatGPU(Mat& input) {
  // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat
  // (getPtr<T>()) cv::Mat and sl::Mat will share a single memory structure
  return cv::cuda::GpuMat(input.getHeight(), input.getWidth(), getOCVtype(input.getDataType()),
                          input.getPtr<sl::uchar1>(MEM::GPU), input.getStepBytes(sl::MEM::GPU));
}
#endif

}  // namespace sl

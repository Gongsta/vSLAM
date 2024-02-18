#include <opencv2/opencv.hpp>
#include <sl/Camera.hpp>

namespace sl {

// Mapping between MAT_TYPE and CV_TYPE
int getOCVtype(sl::MAT_TYPE type);

/**
 * Conversion function between sl::Mat and cv::Mat
 **/
cv::Mat slMat2cvMat(Mat& input);

/**
 * Conversion function between sl::Mat and cv::Mat for GPU
 **/
// cv::cuda::GpuMat slMat2cvMatGPU(Mat& input);

}  // namespace sl

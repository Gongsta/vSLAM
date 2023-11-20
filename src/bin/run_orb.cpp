#include "orb.hpp"
#include "vpi_utils.hpp"

int main() {
  cv::VideoCapture cap(0, cv::CAP_V4L2);
  cv::Mat cv_img_in, cv_img_out;

  uint64_t backends = VPI_BACKEND_CUDA;
  try {
    cap.read(cv_img_in);

    VPIStream stream;
    CHECK_STATUS(vpiStreamCreate(0, &stream));
    ORBFeatureDetector orb{cv_img_in, stream, backends};


    while (true) {
      cap.read(cv_img_in);
      if (cv_img_in.empty()) {
        continue;
      }

      orb.Apply(cv_img_in, cv_img_out);
      cv::imshow("Image", cv_img_out);
      if (cv::waitKey(5) >= 0) {
        break;
      }
    }

    vpiStreamDestroy(stream);

  } catch (std::exception &e) {
    std::cerr << e.what() << std::endl;
  }

  return 0;
}

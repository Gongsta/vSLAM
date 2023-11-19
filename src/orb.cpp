#include "orb.hpp"

static cv::Mat DrawKeypoints(cv::Mat img, VPIKeypointF32* kpts, int numKeypoints) {
  cv::Mat out;
  img.convertTo(out, CV_8UC1);
  cv::cvtColor(out, out, cv::COLOR_GRAY2BGR);

  if (numKeypoints == 0) {
    return out;
  }

  // prepare our colormap
  cv::Mat cmap(1, 256, CV_8UC3);
  {
    cv::Mat gray(1, 256, CV_8UC1);
    for (int i = 0; i < 256; ++i) {
      gray.at<unsigned char>(0, i) = i;
    }
    applyColorMap(gray, cmap, cv::COLORMAP_TURBO);
  }

  for (int i = 0; i < numKeypoints; ++i) {
    cv::Vec3b color = cmap.at<cv::Vec3b>(rand() % 255);
    circle(out, cv::Point(kpts[i].x, kpts[i].y), 3, cv::Scalar(color[0], color[1], color[2]), -1);
  }

  return out;
}

ORB::ORB(cv::Mat cv_img_in, VPIStream stream)
    : descriptors_one{orb_params.maxFeatures, VPI_BRIEF_DESCRIPTOR_ARRAY_LENGTH, CV_8U},
      descriptors_two{orb_params.maxFeatures, VPI_BRIEF_DESCRIPTOR_ARRAY_LENGTH, CV_8U},
      stream{stream}
       {
  // Needs an image to know the dimension to allocate
  CHECK_STATUS(vpiInitORBParams(&orb_params));

  // ================
  // Allocate Memory
  CHECK_STATUS(vpiImageCreate(cv_img_in.cols, cv_img_in.rows, VPI_IMAGE_FORMAT_U8, 0, &img_gray));

  // Create the output keypoint array
  CHECK_STATUS(vpiArrayCreate(orb_params.maxFeatures, VPI_ARRAY_TYPE_KEYPOINT_F32,
                              backend | VPI_BACKEND_CPU, &keypoints));

  // Create the output descriptors array
  CHECK_STATUS(vpiArrayCreate(orb_params.maxFeatures, VPI_ARRAY_TYPE_BRIEF_DESCRIPTOR,
                              backend | VPI_BACKEND_CPU, &descriptors));

  // Create the payload for ORB Feature Detector algorithm
  CHECK_STATUS(vpiCreateORBFeatureDetector(backend, 20000, &orb_payload));

  // Create the Gaussian Pyramid for the image
  CHECK_STATUS(vpiPyramidCreate(cv_img_in.cols, cv_img_in.rows, VPI_IMAGE_FORMAT_U8,
                                orb_params.pyramidLevels, 0.5, backend, &pyr_input));
}

ORB::~ORB() {
  vpiImageDestroy(img_gray);
  vpiArrayDestroy(keypoints);
  vpiArrayDestroy(descriptors);
  vpiPayloadDestroy(orb_payload);
  // if (create_stream) {
  //   vpiStreamDestroy(stream);
  // }
}

void ORB::ProcessFrame(cv::Mat& cv_img_in, cv::Mat& cv_img_out) {
  CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(cv_img_in, 0, &img_in));

  // ================
  // Processing stage

  // Convert img to grayscale
  CHECK_STATUS(vpiSubmitConvertImageFormat(stream, backend, img_in, img_gray, NULL));

  // Create the Gaussian Pyramid for the image and wait for the execution
  CHECK_STATUS(
      vpiSubmitGaussianPyramidGenerator(stream, backend, img_gray, pyr_input, VPI_BORDER_ZERO));

  // Get ORB features and wait for the execution to finish
  CHECK_STATUS(vpiSubmitORBFeatureDetector(stream, backend, orb_payload, pyr_input, keypoints,
                                           descriptors, &orb_params, VPI_BORDER_ZERO));

  CHECK_STATUS(vpiStreamSync(stream));

  // =======================================
  // Output processing

  // Lock output keypoints and scores to retrieve its data on cpu memory
  VPIArrayData out_keypoints_data;
  VPIImageData img_data;
  CHECK_STATUS(
      vpiArrayLockData(keypoints, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS,
      &out_keypoints_data));
  CHECK_STATUS(
      vpiImageLockData(img_gray, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &img_data));

  VPIKeypointF32* outKeypoints = (VPIKeypointF32*)out_keypoints_data.buffer.aos.data;

  printf("%d keypoints found\n", *out_keypoints_data.buffer.aos.sizePointer);

  cv::Mat img;
  CHECK_STATUS(vpiImageDataExportOpenCVMat(img_data, &img));

  cv_img_out = DrawKeypoints(img, outKeypoints, *out_keypoints_data.buffer.aos.sizePointer);

  // Done handling outputs, don't forget to unlock them.
  CHECK_STATUS(vpiImageUnlock(img_gray));
  CHECK_STATUS(vpiArrayUnlock(keypoints));

  // Destroy image to avoid memory leaks
  vpiImageDestroy(img_in);
}

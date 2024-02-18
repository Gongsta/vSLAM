#include "save_depth.hpp"

using namespace sl;
using namespace std;

int count_save = 0;
int mode_PointCloud = 0;
int mode_Depth = 0;
int PointCloud_format;
int Depth_format;

std::string PointCloud_format_ext = ".ply";
std::string Depth_format_ext = ".png";

void setPointCloudFormatName(int format) {
  switch (format) {
    case 0:
      PointCloud_format_ext = ".xyz";
      break;
    case 1:
      PointCloud_format_ext = ".pcd";
      break;
    case 2:
      PointCloud_format_ext = ".ply";
      break;
    case 3:
      PointCloud_format_ext = ".vtk";
      break;
    default:
      break;
  }
}

void setDepthFormatName(int format) {
  switch (format) {
    case 0:
      Depth_format_ext = ".png";
      break;
    case 1:
      Depth_format_ext = ".pfm";
      break;
    case 2:
      Depth_format_ext = ".pgm";
      break;
    default:
      break;
  }
}

void savePointCloud(Camera& zed, std::string filename) {
  sl::Mat point_cloud;
  zed.retrieveMeasure(point_cloud, sl::MEASURE::XYZRGBA);
  auto state = point_cloud.write((filename + PointCloud_format_ext).c_str());

  if (state == ERROR_CODE::SUCCESS) {
    std::cout << "Point Cloud has been saved under " << filename << PointCloud_format_ext << endl;
  } else {
    std::cout << "Failed to save point cloud... Please check that you have permissions to write at "
                 "this location ("
              << filename << "). Re-run the sample with administrator rights under windows" << endl;
  }
}

void saveDepth(Camera& zed, std::string filename) {
  sl::Mat depth;
  zed.retrieveMeasure(depth, sl::MEASURE::DEPTH);

  convertUnit(depth, zed.getInitParameters().coordinate_units, UNIT::MILLIMETER);
  auto state = depth.write((filename + Depth_format_ext).c_str());

  if (state == ERROR_CODE::SUCCESS) {
    std::cout << "Depth Map has been save under " << filename << Depth_format_ext << endl;
  } else {
    std::cout << "Failed to save depth map... Please check that you have permissions to write at "
                 "this location ("
              << filename << "). Re-run the sample with administrator rights under windows" << endl;
  }
}

void saveSbSImage(Camera& zed, std::string filename) {
  sl::Mat image_sbs;
  zed.retrieveImage(image_sbs, sl::VIEW::SIDE_BY_SIDE);

  auto state = image_sbs.write(filename.c_str());

  if (state == sl::ERROR_CODE::SUCCESS) {
    std::cout << "Side by Side image has been save under " << filename << endl;
  } else {
    std::cout << "Failed to save image... Please check that you have permissions to write at this "
                 "location ("
              << filename << "). Re-run the sample with administrator rights under windows" << endl;
  }
}

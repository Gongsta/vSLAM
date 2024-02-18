#ifndef SAVE_DEPTH_HPP
#define SAVE_DEPTH_HPP

#define NOMINMAX

#include <signal.h>

#include <iomanip>
#include <iostream>
#include <limits>
#include <opencv2/opencv.hpp>
#include <sl/Camera.hpp>
#include <thread>

const std::string prefixPointCloud = "Cloud_";  // Default PointCloud output file prefix
const std::string prefixDepth = "Depth_";       // Default Depth image output file prefix
const std::string path = "./";

void savePointCloud(sl::Camera& zed, std::string filename);
void saveDepth(sl::Camera& zed, std::string filename);
void saveSbSImage(sl::Camera& zed, std::string filename);

#endif

/**
 * @file camera.cpp
 * @author Steven Gong (gong.steven@hotmail.com)
 * @brief
 *
 * @copyright MIT License (c) 2023 Steven Gong
 *
 */
#include "camera.hpp"

#include <iostream>
#include <stdexcept>

Camera::Camera(int deviceID, int apiID) {
  open(deviceID, apiID);
  if (!isOpened()) {
    throw std::runtime_error("Cannot open camera with deviceID: " + std::to_string(deviceID) +
                             "and apiID: " + std::to_string(apiID));
  }
}

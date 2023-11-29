#include <Eigen/Core>

struct Pointcloud {
  Eigen::Vector3f* data;
  size_t size;
}

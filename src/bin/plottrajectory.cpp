#include <pangolin/pangolin.h>
#include <unistd.h>

#include <Eigen/Core>
#include <iostream>

#include "visualization_utils.hpp"

int main() {
  std::string trajectory_file = "../src/samples/trajectory.txt";
  std::ifstream fin(trajectory_file);
  if (!fin) {
    std::cout << "cannot find trajectory file at " << trajectory_file << std::endl;
    return 1;
  }

  std::vector<Eigen::Isometry3d> poses;
  while (!fin.eof()) {
    double time, tx, ty, tz, qx, qy, qz, qw;
    fin >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.rotate(Eigen::Quaterniond(qw, qx, qy, qz));
    T.pretranslate(Eigen::Vector3d(tx, ty, tz));
    poses.push_back(T);
  }

  pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
      pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0));
  pangolin::View& d_cam = pangolin::CreateDisplay()
                              .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
                              .SetHandler(new pangolin::Handler3D(s_cam));

  while (pangolin::ShouldQuit() == false) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    d_cam.Activate(s_cam);
    DrawTrajectory(poses);

    pangolin::FinishFrame();
    usleep(5000);  // sleep 5 ms
  }

  return 0;
}
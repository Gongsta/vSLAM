#include <pangolin/pangolin.h>

#include <Eigen/Core>
#include <iostream>
#include <vector>


void DrawPointcloud(std::vector<Eigen::Vector3d>& pointcloud, float point_size = 3.0) {
  glPointSize(point_size);
  glBegin(GL_POINTS);
  for (size_t i = 0; i < pointcloud.size(); i++) {
    glColor3f(1.0, 0.0, 0.0);
    glVertex3d(pointcloud[i][0], pointcloud[i][1], pointcloud[i][2]);
  }
  glEnd();
}

void DrawTrajectory(
    std::vector<Eigen::Isometry3d>& poses) {
  glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
  glLineWidth(2);

  // Draw the 3d frame
  for (size_t i = 0; i < poses.size(); i++) {
    Eigen::Vector3d Ow = poses[i].translation();
    Eigen::Vector3d Xw = poses[i] * (0.1 * Eigen::Vector3d(1, 0, 0));
    Eigen::Vector3d Yw = poses[i] * (0.1 * Eigen::Vector3d(0, 1, 0));
    Eigen::Vector3d Zw = poses[i] * (0.1 * Eigen::Vector3d(0, 0, 1));
    glBegin(GL_LINES);
    glColor3f(1.0, 0.0, 0.0);
    glVertex3d(Ow[0], Ow[1], Ow[2]);
    glVertex3d(Xw[0], Xw[1], Xw[2]);
    glColor3f(0.0, 1.0, 0.0);
    glVertex3d(Ow[0], Ow[1], Ow[2]);
    glVertex3d(Yw[0], Yw[1], Yw[2]);
    glColor3f(0.0, 0.0, 1.0);
    glVertex3d(Ow[0], Ow[1], Ow[2]);
    glVertex3d(Zw[0], Zw[1], Zw[2]);
    glEnd();
  }

  // Vertex to connect the different poses
  for (size_t i = 0; i < poses.size(); i++) {
    glColor3f(0.0, 0.0, 0.0);
    glBegin(GL_LINES);
    auto p1 = poses[i], p2 = poses[i + 1];
    glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
    glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
    glEnd();
  }
}

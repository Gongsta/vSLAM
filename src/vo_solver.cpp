#include <Eigen/Core>
#include <vector>

// Assuming Eigen library is used for matrix and vector operations
Eigen::MatrixXd minimize(const Eigen::VectorXd& PAR, const Eigen::MatrixXd& F1,
                         const Eigen::MatrixXd& F2, const Eigen::MatrixXd& W1,
                         const Eigen::MatrixXd& W2, const Eigen::MatrixXd& P1) {
  Eigen::Vector3d r = PAR.segment(0, 3);
  Eigen::Vector3d t = PAR.segment(3, 3);

  // Compute the transformation matrix from r and t
  Eigen::Matrix4d tran;
  // For demonstration, assume a function createTransformationMatrix(r, t) exists that creates the
  // 4x4 matrix
  tran = createTransformationMatrix(
      r, t);  // This is a placeholder for the actual rotation matrix computation

  Eigen::MatrixXd reproj1(F1.rows(), 3);
  Eigen::MatrixXd reproj2(F1.rows(), 3);

  for (int k = 0; k < F1.rows(); ++k) {
    Eigen::Vector4d f1 = Eigen::Vector4d(F1(k, 0), F1(k, 1), 1, 0);
    Eigen::Vector4d w2 = Eigen::Vector4d(W2(k, 0), W2(k, 1), W2(k, 2), 1);

    Eigen::Vector4d f2 = Eigen::Vector4d(F2(k, 0), F2(k, 1), 1, 0);
    Eigen::Vector4d w1 = Eigen::Vector4d(W1(k, 0), W1(k, 1), W1(k, 2), 1);

    Eigen::Vector4d f1_repr = P1 * tran * w2;
    f1_repr /= f1_repr(2);
    Eigen::Vector4d f2_repr = P1 * tran.inverse() * w1;
    f2_repr /= f2_repr(2);

    reproj1.row(k) = (f1.head(3) - f1_repr.head(3)).transpose();
    reproj2.row(k) = (f2.head(3) - f2_repr.head(3)).transpose();
  }

  Eigen::MatrixXd F = Eigen::MatrixXd(2 * F1.rows(), 3);
  F << reproj1, reproj2;

  return F;
}

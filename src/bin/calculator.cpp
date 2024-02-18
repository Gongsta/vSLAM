#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>
#include <unsupported/Eigen/LevenbergMarquardt>

// Functor for optimization

// Inspired from
// https://github.com/libigl/eigen/blob/master/unsupported/test/levenberg_marquardt.cpp
struct ReprojectionErrorFunctor : Eigen::DenseFunctor<double> {
  // Data members to hold points and matrices
  Eigen::MatrixXd F1, F2;  // 2D points
  Eigen::MatrixXd W1, W2;  // 3D points
  Eigen::MatrixXd P1;      // Projection matrix for the first camera

  ReprojectionErrorFunctor(const Eigen::MatrixXd& F1, const Eigen::MatrixXd& F2,
                           const Eigen::MatrixXd& W1, const Eigen::MatrixXd& W2,
                           const Eigen::Matrix3d& P1)
      : F1(F1), F2(F2), W1(W1), W2(W2), P1(P1) {}

  // Calculate reprojection errors
  int operator()(const Eigen::VectorXd& x, Eigen::VectorXd& fvec) const {
    // Decompose x into r (rotation) and t (translation)
    Eigen::Vector3d r = x.segment<3>(0);
    Eigen::Vector3d t = x.segment<3>(3);

    // Convert r into a rotation matrix
    Eigen::Matrix3d R = Eigen::AngleAxisd(r[0], Eigen::Vector3d::UnitZ()) *
                        Eigen::AngleAxisd(r[1], Eigen::Vector3d::UnitX()) *
                        Eigen::AngleAxisd(r[2], Eigen::Vector3d::UnitZ()).toRotationMatrix();

    // Construct the transformation matrix
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = R;
    T.block<3, 1>(0, 3) = t;

    // Compute reprojection for each point and calculate error
    for (int i = 0; i < F1.rows(); ++i) {
      Eigen::Vector4d w1_homog = W1.row(i).homogeneous();
      Eigen::Vector4d w2_homog = W2.row(i).homogeneous();

      Eigen::Vector3d f1_proj = P1 * (T * w2_homog);
      Eigen::Vector3d f2_proj = P1 * (T.inverse() * w1_homog);

      f1_proj /= f1_proj(2);
      f2_proj /= f2_proj(2);

      fvec.segment<2>(2 * i) = F1.row(i).transpose() - f1_proj.head<2>();
      fvec.segment<2>(2 * i + 1) = F2.row(i).transpose() - f2_proj.head<2>();
    }
    std::cout << "fvec: " << fvec << std::endl;

    return 0;  // 0 indicates success
  }

  int df(const Eigen::VectorXd& x, Eigen::MatrixXd& fjac) const {
    // What if F1 and F2 are not the same size?

    // Small perturbation for numerical differentiation
    const double delta = 1e-6;
    Eigen::VectorXd x_plus_delta = x;
    Eigen::VectorXd fvec_original(x.size());
    Eigen::VectorXd fvec_perturbed(x.size());

    // Evaluate the original function to fill fvec_original
    operator()(x, fvec_original);

    for (int j = 0; j < x.size(); ++j) {
      // Perturb parameter j
      x_plus_delta[j] += delta;

      // Evaluate function with perturbed parameter
      operator()(x_plus_delta, fvec_perturbed);

      // Restore parameter j
      x_plus_delta[j] = x[j];

      // Compute derivative (approximation) for each parameter
      for (int i = 0; i < fvec_original.size(); ++i) {
        fjac(i, j) = (fvec_perturbed[i] - fvec_original[i]) / delta;
      }
    }
    return 0;  // 0 indicates success
  }

  // Number of parameters: 6 (3 for rotation, 3 for translation)
  int inputs() const { return 6; }
  // Number of observations: twice the number of points (2D reprojection error per point in each
  // image)
  int values() const { return 2 * F1.rows(); }
};

void optimizeParameters(const Eigen::MatrixXd& F1, const Eigen::MatrixXd& F2,
                        const Eigen::MatrixXd& W1, const Eigen::MatrixXd& W2,
                        const Eigen::Matrix3d& P1) {
  Eigen::VectorXd x(6);  // Starting guess for parameters, initially zero

  ReprojectionErrorFunctor functor(F1, F2, W1, W2, P1);
  // Eigen::NumericalDiff<ReprojectionErrorFunctor> numDiff(functor);
  Eigen::LevenbergMarquardt<ReprojectionErrorFunctor> lm(functor);
  // Eigen::VectorXd fvec;
  // std::cout << functor(x, fvec) << std::endl;

  auto ret = lm.minimize(x);

  std::cout << "Optimized Parameters: " << x.transpose() << std::endl;
}

int main() {
  // Example usage
  // Define F1, F2, W1, W2, P1 based on your data
  Eigen::MatrixX2d F1(10, 2), F2(10, 2);
  Eigen::MatrixX3d W1(10, 3), W2(10, 3);
  Eigen::MatrixXd P1(3, 4);
  // Initialize P1
  P1 << 130, 0, 0, 0.1, 0, 289, 0, 0, 0, 0, 1, 0;

  // Call optimizeParameters with your data
  optimizeParameters(F1, F2, W1, W2, P1);

  return 0;
}

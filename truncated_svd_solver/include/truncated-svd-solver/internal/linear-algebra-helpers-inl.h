#ifndef ASLAM_CALIBRATION_ALGORITHMS_LINALG_INL_H
#define ASLAM_CALIBRATION_ALGORITHMS_LINALG_INL_H

#include <cmath>
#include <cstddef>
#include <limits>

#include <Eigen/Core>
#include <glog/logging.h>

namespace truncated_svd_solver {

template <typename T>
Eigen::MatrixXd computeCovariance(const T& R, size_t colBegin,
    size_t colEnd) {
  checkColumnIndices(R, colBegin, colEnd);
  CHECK_LT(colBegin, colEnd);
  CHECK_LT(colEnd, static_cast<size_t>(R.cols()));

  // NOTE: What about checking the form of R? Upper triangular matrix
  const size_t numCols = R.cols();
  const size_t dim = numCols - colBegin;
  Eigen::MatrixXd covariance = Eigen::MatrixXd::Zero(dim, dim);
  for (std::ptrdiff_t l = numCols - 1, Sigma_l = dim - 1;
      l >= (std::ptrdiff_t)(numCols - dim); --l, --Sigma_l) {
    double temp1 = 0;
    for (std::ptrdiff_t j = l + 1, Sigma_j = Sigma_l + 1;
        j < (std::ptrdiff_t)numCols; ++j, ++Sigma_j)
      temp1 += R(l, j) * covariance(Sigma_j, Sigma_l);
    const double R_ll = R(l, l);
    covariance(Sigma_l, Sigma_l) = 1 / R_ll * (1 / R_ll - temp1);
    for (std::ptrdiff_t i = l - 1, Sigma_i = Sigma_l - 1;
        i >= std::ptrdiff_t(numCols - dim); --i, --Sigma_i) {
      temp1 = 0;
      for (std::ptrdiff_t j = i + 1, Sigma_j = Sigma_i + 1;
          j <= l; ++j, ++Sigma_j)
        temp1 += R(i, j) * covariance(Sigma_j, Sigma_l);
      double temp2 = 0;
      for (std::ptrdiff_t j = l + 1, Sigma_j = Sigma_l + 1;
          j < (std::ptrdiff_t)numCols; ++j, ++Sigma_j)
        temp2 += R(i, j) * covariance(Sigma_l, Sigma_j);
      covariance(Sigma_i, Sigma_l) = 1 / R(i, i) * (-temp1 - temp2);
      covariance(Sigma_l, Sigma_i) = covariance(Sigma_i, Sigma_l);
    }
  }
  const size_t block_dim = colEnd - colBegin + 1;
  return covariance.block(0, 0, block_dim, block_dim);
}

}  // namespace truncated_svd_solver
#endif // ASLAM_CALIBRATION_ALGORITHMS_LINALG_INL_H

#include "truncated-svd-solver/marginalization.h"

#include <cmath>

#include <cholmod.h>
#include <Eigen/Dense>
#include <glog/logging.h>
#include <spqr.hpp>
#include <SuiteSparseQR.hpp>

#include "truncated-svd-solver/linear-algebra-helpers.h"

namespace truncated_svd_solver {

Eigen::MatrixXd marginalJacobian(cholmod_sparse* J_x, cholmod_sparse*
    J_thetat, cholmod_common* cholmod) {
  // Compute the QR factorization of J_x.
  SuiteSparseQR_factorization<double>* QR = SuiteSparseQR_factorize<double>(
    SPQR_ORDERING_BEST, SPQR_DEFAULT_TOL, J_x, cholmod);
  CHECK(QR != nullptr) << "SuiteSparseQR_factorize failed.";

  // Compute the Jacobian of the reduced system.
  cholmod_sparse* J_thetatQFull = SuiteSparseQR_qmult<double>(SPQR_XQ, QR,
    J_thetat, cholmod);
  CHECK(QR != nullptr) << "SuiteSparseQR_qmult failed.";

  std::ptrdiff_t* col_indices = new std::ptrdiff_t[J_x->ncol];
  for (size_t i = 0; i < J_x->ncol; ++i)
   col_indices[i] = i;
  cholmod_sparse* J_thetatQ = cholmod_l_submatrix(J_thetatQFull, NULL, -1,
    col_indices, J_x->ncol, 1, 0, cholmod);
  delete [] col_indices;
  CHECK(J_thetatQ != nullptr) << "cholmod_l_submatrix failed.";

  cholmod_sparse* J_thetat2 = cholmod_l_aat(J_thetat, NULL, 0, 1, cholmod);
  CHECK(J_thetat2 != nullptr) << "cholmod_l_aat failed.";

  cholmod_sparse* J_thetatQ2 = cholmod_l_aat(J_thetatQ, NULL, 0, 1,
    cholmod);
  CHECK(J_thetatQ2 != nullptr) << "cholmod_l_aat failed.";

  double alpha[2];
  alpha[0] = 1;
  double beta[2];
  beta[0] = -1;
  cholmod_sparse* Omega = cholmod_l_add(J_thetat2, J_thetatQ2, alpha, beta,
    1, 0, cholmod);
  CHECK(Omega != nullptr) << "cholmod_l_add failed.";

  Eigen::MatrixXd OmegaDense;
  cholmodSparseToEigenDenseCopy(Omega, OmegaDense);

  // Clean allocated memory.
  SuiteSparseQR_free(&QR, cholmod);
  cholmod_l_free_sparse(&J_thetatQFull, cholmod);
  cholmod_l_free_sparse(&J_thetatQ, cholmod);
  cholmod_l_free_sparse(&J_thetat2, cholmod);
  cholmod_l_free_sparse(&J_thetatQ2, cholmod);
  cholmod_l_free_sparse(&Omega, cholmod);

  return OmegaDense;
}

double marginalize(cholmod_sparse* Jt, size_t j, Eigen::MatrixXd& NS,
                   Eigen::MatrixXd& CS, Eigen::MatrixXd& Sigma,
                   Eigen::MatrixXd& SigmaP, Eigen::MatrixXd& Omega,
                   double norm_tol, double eps_tol) {
  CHECK_NOTNULL(Jt);

  // Initialize cholmod.
  cholmod_common cholmod;
  cholmod_l_start(&cholmod);

  // Convert to cholmod_sparse.
  cholmod_sparse* J = cholmod_l_transpose(Jt, 1, &cholmod);
  CHECK(J != nullptr) << "cholmod_l_transpose failed.";

  // Extract the part corresponding to the state/landmarks/...
  std::ptrdiff_t* col_indices = new std::ptrdiff_t[j];
  for (size_t i = 0; i < j; ++i)
   col_indices[i] = i;
  cholmod_sparse* J_x = cholmod_l_submatrix(J, NULL, -1, col_indices, j, 1,
    0, &cholmod);
  delete [] col_indices;
  CHECK(J_x != nullptr) << "cholmod_l_submatrix failed.";

  // Extract the part corresponding to the calibration parameters.
  col_indices = new std::ptrdiff_t[J->ncol - j];
  for (size_t i = j; i < J->ncol; ++i)
   col_indices[i - j] = i;
  cholmod_sparse* J_theta = cholmod_l_submatrix(J, NULL, -1, col_indices,
    J->ncol - j, 1, 0, &cholmod);
  delete [] col_indices;
  CHECK(J_theta != nullptr) << "cholmod_l_submatrix failed.";

  cholmod_sparse* J_thetat = cholmod_l_transpose(J_theta, 1, &cholmod);
  CHECK(J_thetat != nullptr) << "cholmod_l_transpose failed.";

  // Compute the marginal Jacobian.
  Omega = marginalJacobian(J_x, J_thetat, &cholmod);

  // Scale J_x.
  cholmod_dense* G_x = cholmod_l_allocate_dense(J_x->ncol, 1, J_x->ncol,
    CHOLMOD_REAL, &cholmod);
  CHECK(G_x != nullptr) << "cholmod_l_allocate_dense failed.";
  double* values = reinterpret_cast<double*>(G_x->x);

  for (size_t j = 0; j < J_x->ncol; ++j) {
    const double norm_col = colNorm(J_x, j);
    if (norm_col < norm_tol)
      values[j] = 0.0;
    else
      values[j] = 1.0 / norm_col;
  }

  bool success = cholmod_l_scale(G_x, CHOLMOD_COL, J_x, &cholmod);
  CHECK(success) << "cholmod_l_scale failed.";
  cholmod_l_free_dense(&G_x, &cholmod) ;

  // Scale J_thetat.
  cholmod_dense* G_theta = cholmod_l_allocate_dense(J_theta->ncol, 1,
    J_theta->ncol, CHOLMOD_REAL, &cholmod);
  CHECK(G_theta != nullptr) << "cholmod_l_allocate_dense failed.";
  values = reinterpret_cast<double*>(G_theta->x);

  for (size_t j = 0; j < J_theta->ncol; ++j) {
    const double normCol = colNorm(J_theta, j);
    if (normCol < norm_tol)
      values[j] = 0.0;
    else
      values[j] = 1.0 / normCol;
  }

  success = cholmod_l_scale(G_theta, CHOLMOD_ROW, J_thetat, &cholmod);
  CHECK(success) << "cholmod_l_scale failed.";
  cholmod_l_free_dense(&G_theta, &cholmod);

  // Compute the scaled marginal Jacobian.
  Eigen::MatrixXd OmegaScaled;
  OmegaScaled = marginalJacobian(J_x, J_thetat, &cholmod);

  // Cleanup cholmod.
  cholmod_l_free_sparse(&J, &cholmod);
  cholmod_l_free_sparse(&J_x, &cholmod);
  cholmod_l_free_sparse(&J_theta, &cholmod);
  cholmod_l_free_sparse(&J_thetat, &cholmod);
  cholmod_l_finish(&cholmod);

  // Compute the thin SVD of OmegaScaled.
  const Eigen::JacobiSVD<Eigen::MatrixXd> svd_scaled(OmegaScaled,
    Eigen::ComputeThinU | Eigen::ComputeThinV);

  // Compute the numerical rank.
  size_t nrank = OmegaScaled.cols();
  const Eigen::VectorXd& SScaled = svd_scaled.singularValues();
  const double tol = OmegaScaled.rows() * SScaled(0) * eps_tol;
  for (std::ptrdiff_t i = OmegaScaled.cols() - 1; i > 0; --i) {
    if (SScaled(i) > tol) {
      break;
    } else {
      nrank--;
    }
  }

  // Compute the thin SVD of Omega.
  const Eigen::JacobiSVD<Eigen::MatrixXd> svd(Omega,
    Eigen::ComputeThinU | Eigen::ComputeThinV);
  const Eigen::MatrixXd& V = svd.matrixV();

  // compute the numerical column space
  CS = V.block(0, 0, V.rows(), nrank);

  // compute the numerical null space
  NS = V.block(0, nrank, V.rows(), Omega.cols() - nrank);

  // compute the projected covariance matrix
  Eigen::MatrixXd inv_S(Eigen::MatrixXd::Zero(Omega.cols(), Omega.cols()));
  SigmaP = Eigen::MatrixXd::Zero(nrank, nrank);
  double sv_log_sum = 0.0;
  const Eigen::VectorXd& S = svd.singularValues();
  for (size_t i = 0u; i < nrank; ++i) {
    SigmaP(i, i) = 1.0 / S(i);
    sv_log_sum = sv_log_sum + log2(S(i));
    inv_S(i, i) = SigmaP(i, i);
  }

  // compute the covariance matrix
  Sigma = V * inv_S * V.transpose();
  return sv_log_sum;
}

}  // namespace truncated_svd_solver

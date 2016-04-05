#include "truncated-svd-solver/linear-algebra-helpers.h"

#include <algorithm>
#include <cmath>

#include <cholmod.h>
#include <Eigen/Dense>
#include <glog/logging.h>
#include <spqr.hpp>
#include <SuiteSparseQR.hpp>

namespace truncated_svd_solver {

cholmod_sparse* columnSubmatrix(cholmod_sparse* A, std::ptrdiff_t
    col_start_idx, std::ptrdiff_t col_end_idx, cholmod_common* cholmod) {
  CHECK_NOTNULL(A);
  CHECK_NOTNULL(cholmod);

  CHECK_GE(col_end_idx, col_start_idx);
  CHECK_LT(col_end_idx, static_cast<std::ptrdiff_t>(A->ncol));
  CHECK_GE(col_start_idx, 0);

  const std::ptrdiff_t num_indices = col_end_idx - col_start_idx + 1;
  std::ptrdiff_t* col_indices = new std::ptrdiff_t[num_indices];
  for (std::ptrdiff_t j = col_start_idx; j <= col_end_idx; ++j) {
    col_indices[j - col_start_idx] = j;
  }

  cholmod_sparse* A_sub = cholmod_l_submatrix(A, nullptr, -1, col_indices,
    num_indices, 1, 1, cholmod);
  delete [] col_indices;
  CHECK(A_sub != nullptr) << "cholmod_l_submatrix failed.";

  return A_sub;
}

cholmod_sparse* rowSubmatrix(cholmod_sparse* A, std::ptrdiff_t row_start_idx,
    std::ptrdiff_t row_end_idx, cholmod_common* cholmod) {
  CHECK_NOTNULL(A);
  CHECK_NOTNULL(cholmod);

  CHECK_GE(row_end_idx, row_start_idx);
  CHECK_LT(row_end_idx, static_cast<std::ptrdiff_t>(A->nrow));
  CHECK_GE(row_start_idx, 0);

  const std::ptrdiff_t num_indices = row_end_idx - row_start_idx + 1;
  std::ptrdiff_t* row_indices = new std::ptrdiff_t[num_indices];
  for (std::ptrdiff_t i = row_start_idx; i <= row_end_idx; ++i){
    row_indices[i - row_start_idx] = i;
  }

  cholmod_sparse* A_sub = cholmod_l_submatrix(A, row_indices, num_indices,
    nullptr, -1, 1, 1, cholmod);
  delete [] row_indices;
  CHECK(A_sub != nullptr) << "cholmod_l_submatrix failed.";

  return A_sub;
}

double colNorm(const cholmod_sparse* A, std::ptrdiff_t col_idx) {
  CHECK_NOTNULL(A);
  CHECK_GE(col_idx, 0);
  CHECK_LT(col_idx, static_cast<std::ptrdiff_t>(A->ncol));

  const std::ptrdiff_t* col_ptr =
    reinterpret_cast<const std::ptrdiff_t*>(A->p);
  const double* values = reinterpret_cast<const double*>(A->x);
  const std::ptrdiff_t p = col_ptr[col_idx];
  const std::ptrdiff_t num_elements = col_ptr[col_idx + 1] - p;

  double norm = 0.0;
  for (std::ptrdiff_t i = 0; i < num_elements; ++i) {
    norm += values[p + i] * values[p + i];
  }
  return std::sqrt(norm);
}

cholmod_dense* columnScalingMatrix(cholmod_sparse* A, cholmod_common*
    cholmod, double eps) {
  CHECK_NOTNULL(A);
  CHECK_NOTNULL(cholmod);
  CHECK_GT(eps, 0.0);

  cholmod_dense* G = cholmod_l_allocate_dense(A->ncol, 1, A->ncol,
    CHOLMOD_REAL, cholmod);
  CHECK(G != nullptr) << "cholmod_l_allocate_dense failed.";

  const double norm_tolerance = std::sqrt(A->nrow * eps);
  double* values = reinterpret_cast<double*>(G->x);
  for (std::ptrdiff_t col_idx = 0;
      col_idx < static_cast<std::ptrdiff_t>(A->ncol);
      ++col_idx) {
    const double norm = colNorm(A, col_idx);
    if (norm < norm_tolerance) {
      values[col_idx] = 0.0;
    } else {
      values[col_idx] = 1.0 / norm;
    }
  }
  return G;
}

void cholmodSparseToEigenDenseCopy(const cholmod_sparse* in,
    Eigen::MatrixXd& out) {
  CHECK_NOTNULL(in);
  out.setZero(in->nrow, in->ncol);

  const std::ptrdiff_t* row_ind =
    reinterpret_cast<const std::ptrdiff_t*>(in->i);
  const std::ptrdiff_t* col_ptr =
    reinterpret_cast<const std::ptrdiff_t*>(in->p);
  const double* values = reinterpret_cast<const double*>(in->x);
  for (std::ptrdiff_t col_idx = 0;
       col_idx < static_cast<std::ptrdiff_t>(in->ncol); ++col_idx)
    for (std::ptrdiff_t val_idx = col_ptr[col_idx];
         val_idx < col_ptr[col_idx + 1]; ++val_idx) {
      out(row_ind[val_idx], col_idx) = values[val_idx];
      if (in->stype && col_idx != row_ind[val_idx]) {
        out(col_idx, row_ind[val_idx]) = values[val_idx];
      }
    }
}

void eigenDenseToCholmodDenseView(const Eigen::VectorXd& in, cholmod_dense*
    out) {
  CHECK_NOTNULL(out);

  out->nrow = in.size();
  out->ncol = 1;
  out->nzmax = in.size();
  out->d = in.size();
  out->x = reinterpret_cast<void*>(const_cast<double*>(in.data()));
  out->z = nullptr;
  out->xtype = CHOLMOD_REAL;
  out->dtype = CHOLMOD_DOUBLE;
}

cholmod_dense* eigenDenseToCholmodDenseCopy(const Eigen::VectorXd& in,
    cholmod_common* cholmod) {
  CHECK_NOTNULL(cholmod);
  cholmod_dense* out = cholmod_l_allocate_dense(in.size(), 1, in.size(),
    CHOLMOD_REAL, cholmod);
  CHECK(out != nullptr) << "cholmod_l_allocate_dense failed.";

  double* out_val = reinterpret_cast<double*>(out->x);
  const double* in_val = in.data();
  std::copy(in_val, in_val + in.size(), out_val);
  return out;
}

void cholmodDenseToEigenDenseCopy(const cholmod_dense* in, Eigen::VectorXd&
    out) {
  CHECK_NOTNULL(in);
  out.resize(in->nrow);
  const double* in_val = reinterpret_cast<const double*>(in->x);
  std::copy(in_val, in_val + in->nrow, out.data());
}

cholmod_sparse* eigenDenseToCholmodSparseCopy(const Eigen::MatrixXd& A,
    cholmod_common* cholmod, double eps) {
  CHECK_NOTNULL(cholmod);
  CHECK_GT(eps, 0.0);

  size_t nzmax = 0;
  for (std::ptrdiff_t i = 0; i < A.rows(); ++i) {
    for (std::ptrdiff_t j = 0; j < A.cols(); ++j) {
      if (std::fabs(A(i, j)) > eps) {
        nzmax++;
      }
    }
  }

  cholmod_sparse* A_cholmod = cholmod_l_allocate_sparse(A.rows(), A.cols(),
    nzmax, 1, 1, 0, CHOLMOD_REAL, cholmod);
  CHECK(A_cholmod != nullptr) << "cholmod_l_allocate_sparse failed.";

  std::ptrdiff_t* row_ind = reinterpret_cast<std::ptrdiff_t*>(A_cholmod->i);
  std::ptrdiff_t* col_ptr = reinterpret_cast<std::ptrdiff_t*>(A_cholmod->p);
  double* values = reinterpret_cast<double*>(A_cholmod->x);
  std::ptrdiff_t row_it = 0;
  std::ptrdiff_t col_it = 1;
  for (std::ptrdiff_t c = 0; c < A.cols(); ++c) {
    for (std::ptrdiff_t r = 0; r < A.rows(); ++r)
      if (std::fabs(A(r, c)) > eps) {
        values[row_it] = A(r, c);
        row_ind[row_it] = r;
        row_it++;
      }
    col_ptr[col_it] = row_it;
    col_it++;
  }
  return A_cholmod;
}

std::ptrdiff_t estimateNumericalRank(const Eigen::VectorXd& singular_values,
                                     double tolerance) {
  CHECK_GT(tolerance, 0.0);

  std::ptrdiff_t numerical_rank = singular_values.size();
  for (std::ptrdiff_t i = singular_values.size() - 1; i >= 0; --i) {
    if (singular_values(i) > tolerance) {
      // Assumes that the singular values are ordered.
      break;
    } else {
      --numerical_rank;
    }
  }
  return numerical_rank;
}

double rankTol(const Eigen::VectorXd& singular_values, double eps) {
  CHECK_GT(singular_values.rows(), 0) << "Empty singular values vector.";
  CHECK_GT(eps, 0.0);
  return singular_values(0) * eps * singular_values.size();
}

double qrTol(cholmod_sparse* A, cholmod_common* cholmod, double eps) {
  CHECK_NOTNULL(A);
  CHECK_NOTNULL(cholmod);
  CHECK_GT(eps, 0.0);
  return 20.0 * static_cast<double>(A->nrow + A->ncol) * eps *
    spqr_maxcolnorm<double>(A, cholmod);
}

double svGap(const Eigen::VectorXd& singular_values, std::ptrdiff_t rank) {
  CHECK_LE(rank, singular_values.rows());
  CHECK_GE(rank, 0);

  if (rank == 0) {
    return 0.0;
  } else if (rank < singular_values.size()) {
    return singular_values(rank - 1) / singular_values(rank);
  }
  return std::numeric_limits<double>::infinity();
}

void reduceLeftHandSide(SuiteSparseQR_factorization<double>* factor,
    cholmod_sparse* A_rt, cholmod_sparse** Omega, cholmod_sparse** A_rtQ,
    cholmod_common* cholmod) {
  CHECK_NOTNULL(A_rt);
  CHECK_NOTNULL(cholmod);
  CHECK_NOTNULL(Omega);

  // Handle the special case where the QR part is empty.
  if (factor == nullptr) {
    // NO QR part!
    *Omega = cholmod_l_aat(A_rt, nullptr, 0, 1, cholmod);
    return;
  } else {
    CHECK(factor->QRsym != nullptr) << "Run symbolic factorization first.";
    CHECK(factor->QRnum != nullptr) << "Run QR decomposition first.";
  }
  CHECK_NOTNULL(factor);
  CHECK_NOTNULL(A_rtQ);

  cholmod_sparse* A_rtQFull = SuiteSparseQR_qmult<double>(SPQR_XQ, factor,
    A_rt, cholmod);
  CHECK(A_rtQFull != nullptr) << "SuiteSparseQR_qmult failed.";


  *A_rtQ = columnSubmatrix(A_rtQFull, 0, factor->QRsym->n - 1, cholmod);
  cholmod_l_free_sparse(&A_rtQFull, cholmod);

  cholmod_sparse* A_rtQ2 = cholmod_l_aat(*A_rtQ, nullptr, 0, 1, cholmod);
  CHECK(A_rtQ2 != nullptr) << "cholmod_l_aat failed.";
  A_rtQ2->stype = 1;

  cholmod_sparse* A_rt2 = cholmod_l_aat(A_rt, nullptr, 0, 1, cholmod);
  CHECK(A_rt2 != nullptr) << "cholmod_l_aat failed.";
  A_rt2->stype = 1;

  double alpha[2];
  alpha[0] = 1.0;
  double beta[2];
  beta[0] = -1.0;

  *Omega = cholmod_l_add(A_rt2, A_rtQ2, alpha, beta, 1, 1, cholmod);
  CHECK(*Omega != nullptr) << "cholmod_l_add failed.";

  cholmod_l_free_sparse(&A_rt2, cholmod);
  cholmod_l_free_sparse(&A_rtQ2, cholmod);
}

cholmod_dense* reduceRightHandSide(SuiteSparseQR_factorization<double>*
    factor, cholmod_sparse* A_rt, cholmod_sparse* A_rtQ, cholmod_dense* b,
    cholmod_common* cholmod) {
  CHECK_NOTNULL(A_rt);
  CHECK_NOTNULL(b);
  CHECK_NOTNULL(cholmod);

  // Handle the special case where the QR part is empty.
  if (factor == nullptr) {
    // NO QR part!
    double one = 1, zero = 0;
    cholmod_dense* b_reduced = cholmod_l_allocate_dense(
        A_rt->nrow, 1, A_rt->nrow, CHOLMOD_REAL, cholmod);
    CHECK(b_reduced != nullptr) << "cholmod_l_allocate_dense failed.";

    const int status = cholmod_l_sdmult(A_rt, 0, &one, &zero, b, b_reduced,
                                        cholmod);
    CHECK(status) << "cholmod_l_sdmult failed";
    return b_reduced;
  } else {
    CHECK(factor->QRsym != nullptr) << "Run symbolic factorization first.";
    CHECK(factor->QRnum != nullptr) << "Run QR decomposition first.";
  }
  CHECK_NOTNULL(factor);
  CHECK_NOTNULL(A_rtQ);

  cholmod_sparse* b_sparse = cholmod_l_dense_to_sparse(b, 1, cholmod);
  CHECK(b_sparse != nullptr) << "cholmod_l_dense_to_sparse failed.";

  cholmod_sparse* A_rtb = cholmod_l_ssmult(A_rt, b_sparse, 0, 1, 1, cholmod);
  CHECK(A_rtb != nullptr) << "cholmod_l_ssmult failed.";

  cholmod_sparse* QtbFull = SuiteSparseQR_qmult<double>(SPQR_QTX, factor,
    b_sparse, cholmod);
  CHECK(QtbFull != nullptr) << "SuiteSparseQR_qmult failed.";
  cholmod_l_free_sparse(&b_sparse, cholmod);

  cholmod_sparse* Qtb;
  Qtb = rowSubmatrix(QtbFull, 0, factor->QRsym->n - 1, cholmod);
  cholmod_l_free_sparse(&QtbFull, cholmod);

  cholmod_sparse* A_rtQQtb = cholmod_l_ssmult(A_rtQ, Qtb, 0, 1, 1, cholmod);
  CHECK(A_rtQQtb != nullptr) << "cholmod_l_ssmult failed.";
  cholmod_l_free_sparse(&Qtb, cholmod);

  double alpha[2];
  alpha[0] = 1.0;
  double beta[2];
  beta[0] = -1.0;

  cholmod_sparse* b_reduced_sparse = cholmod_l_add(A_rtb, A_rtQQtb, alpha,
    beta, 1, 1, cholmod);
  CHECK(b_reduced_sparse != nullptr) << "cholmod_l_add failed.";
  cholmod_l_free_sparse(&A_rtb, cholmod);
  cholmod_l_free_sparse(&A_rtQQtb, cholmod);

  cholmod_dense* b_reduced = cholmod_l_sparse_to_dense(b_reduced_sparse,
    cholmod);
  CHECK(b_reduced != nullptr) << "cholmod_l_sparse_to_dense failed.";
  cholmod_l_free_sparse(&b_reduced_sparse, cholmod);
  return b_reduced;
}

void analyzeSVD(const cholmod_sparse* Omega, Eigen::VectorXd& sv,
    Eigen::MatrixXd& U, Eigen::MatrixXd& V) {
  CHECK_NOTNULL(Omega);

  Eigen::MatrixXd OmegaDense;
  cholmodSparseToEigenDenseCopy(Omega, OmegaDense);
  const Eigen::JacobiSVD<Eigen::MatrixXd> svd(OmegaDense,
    Eigen::ComputeThinU | Eigen::ComputeThinV);
  U = svd.matrixU();
  V = svd.matrixV();
  sv = svd.singularValues();
}

void solveSVD(const cholmod_dense* b, const Eigen::VectorXd& sv, const
    Eigen::MatrixXd& U, const Eigen::MatrixXd& V, std::ptrdiff_t rank,
    Eigen::VectorXd& x) {
  CHECK_NOTNULL(b);
  CHECK_LE(rank, V.cols());
  CHECK_EQ(V.cols(), sv.rows());
  CHECK_EQ(U.rows(), static_cast<std::ptrdiff_t>(b->nrow));

  Eigen::Map<const Eigen::VectorXd> b_eigen(
    reinterpret_cast<const double*>(b->x), b->nrow);
  x = V.leftCols(rank) * sv.head(rank).asDiagonal().inverse() *
    U.leftCols(rank).adjoint() * b_eigen;
}

cholmod_dense* solveQR(SuiteSparseQR_factorization<double>* factor,
    cholmod_dense* b, cholmod_sparse* A_r, const Eigen::VectorXd& x_r,
    cholmod_common* cholmod) {
  CHECK_NOTNULL(factor);
  CHECK(factor->QRsym != nullptr) << "Run symbolic factorization first.";
  CHECK(factor->QRnum != nullptr) << "Run QR decomposition first.";
  CHECK_NOTNULL(b);
  CHECK_NOTNULL(cholmod);

  cholmod_dense* QtbmA_rx_r = nullptr;
  if (A_r != nullptr) {
    CHECK_EQ(x_r.size(), static_cast<int>(A_r->ncol))
      << "Dimension mismatch between x_r and A_r.";

    cholmod_dense x_r_cholmod;
    eigenDenseToCholmodDenseView(x_r, &x_r_cholmod);
    cholmod_dense* A_rx_r = cholmod_l_allocate_dense(A_r->nrow, 1, A_r->nrow,
      CHOLMOD_REAL, cholmod);
    CHECK(A_rx_r != nullptr) << "cholmod_l_allocate_dense failed.";

    double alpha[2];
    alpha[0] = 1.0;
    double beta[2];
    beta[0] = 0.0;
    const bool success = cholmod_l_sdmult(A_r, 0, alpha, beta, &x_r_cholmod,
                                          A_rx_r, cholmod);
    CHECK(success) << "cholmod_l_sdmult failed.";

    Eigen::Map<const Eigen::VectorXd> bEigen(
      reinterpret_cast<const double*>(b->x), b->nrow);
    Eigen::Map<const Eigen::VectorXd> A_rx_rEigen(
      reinterpret_cast<const double*>(A_rx_r->x), A_rx_r->nrow);
    const Eigen::VectorXd bmA_rx_rEigen = bEigen - A_rx_rEigen;
    cholmod_l_free_dense(&A_rx_r, cholmod);
    cholmod_dense bmA_rx_r;
    eigenDenseToCholmodDenseView(bmA_rx_rEigen, &bmA_rx_r);
    QtbmA_rx_r = SuiteSparseQR_qmult<double>(SPQR_QTX, factor, &bmA_rx_r,
                                             cholmod);
  } else {
    QtbmA_rx_r = SuiteSparseQR_qmult<double>(SPQR_QTX, factor, b, cholmod);
  }
  CHECK(QtbmA_rx_r != nullptr) << "SuiteSparseQR_qmult failed.";

  cholmod_dense* x_l = SuiteSparseQR_solve<double>(SPQR_RETX_EQUALS_B,
    factor, QtbmA_rx_r, cholmod);
  CHECK(x_l != nullptr) << "SuiteSparseQR_solve failed.";
  cholmod_l_free_dense(&QtbmA_rx_r, cholmod);
  return x_l;
}

}  // namespace truncated_svd_solver

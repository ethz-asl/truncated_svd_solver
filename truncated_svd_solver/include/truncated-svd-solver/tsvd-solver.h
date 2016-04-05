#ifndef TRUNCATED_SVD_SOLVER_TSVD_SOLVER_H
#define TRUNCATED_SVD_SOLVER_TSVD_SOLVER_H

#include <cstddef>
#include <string>
#include <vector>

#include <cholmod.h>
#include <Eigen/Core>

#include "truncated-svd-solver/tsvd-solver-options.h"

template <typename Entry> struct SuiteSparseQR_factorization;

namespace truncated_svd_solver {

/** The class LinearSolver implements a specific linear solver as a
    combination of SPQR and SVD.The right part of the input matrix can be
    marginalized-out and solved by SVD, while the rest of the variables are
    solved with SPQR.
    \brief Linear solver for incremental calibration
  */
class TruncatedSvdSolver {
 public:
  typedef TruncatedSvdSolverOptions Options;

  TruncatedSvdSolver(const Options& options = Options());
  TruncatedSvdSolver(const TruncatedSvdSolver& other) = delete;
  TruncatedSvdSolver& operator= (const TruncatedSvdSolver& other) = delete;
  TruncatedSvdSolver(TruncatedSvdSolver&& other) = delete;
  TruncatedSvdSolver& operator= (TruncatedSvdSolver&& other) = delete;
  virtual ~TruncatedSvdSolver();

  /**
   * This function solves a system of equation using marginalization.
   * \brief Marginalization solver
   *
   * \return void
   * \param[in] A sparse matrix left-hand side
   * \param[in] b dense vector right-hand side
   * \param[in] j starting column index for the marginalization
   * \param[out] x solution
   */
  void solve(cholmod_sparse* A, cholmod_dense* b, size_t j,
    Eigen::VectorXd& x);

  /**
   * This function analyzes the marginalized matrix at index j. The goal
   * is to estimate the numerical null and column space, the covariance, the
   * covariance on the column space. All the results are cached in the
   * solver.
   * \brief Marginalization analyzer
   *
   * \return void
   * \param[in] A sparse matrix left-hand side
   * \param[in] j starting column index for the marginalization
   */
  void analyzeMarginal(cholmod_sparse* A, size_t j);

  /// Clear the cached variables
  void clear();

  /// Returns the options
  const Options& getOptions() const;
  /// Returns the options
  Options& getOptions();
  /// Returns the current SVD rank
  std::ptrdiff_t getSVDRank() const;
  /// Returns the current SVD rank deficiency
  std::ptrdiff_t getSVDRankDeficiency() const;
  /// Returns the current gap in the singular values at the rank
  double getSvGap() const;
  /// Returns the current QR rank
  std::ptrdiff_t getQRRank() const;
  /// Returns the current QR rank deficiency
  std::ptrdiff_t getQRRankDeficiency() const;
  /// Returns the marginalization start index
  std::ptrdiff_t getMargStartIndex() const;
  /// Sets the marginalization start index
  void setMargStartIndex(std::ptrdiff_t index);
  /// Returns the current tolerance used by SPQR
  double getQRTolerance() const;
  /// Returns the current tolerance used by SVD
  double getSVDTolerance() const;
  /// Returns the current singular values
  const Eigen::VectorXd& getSingularValues() const;
  /// Returns the current left-singular vectors
  const Eigen::MatrixXd& getMatrixU() const;
  /// Returns the current right-singular vectors
  const Eigen::MatrixXd& getMatrixV() const;
  /// Returns the current null space for SVD
  Eigen::MatrixXd getNullSpace() const;
  /// Returns the current row space for SVD
  Eigen::MatrixXd getRowSpace() const;
  /// Returns the current covariance matrix for SVD
  Eigen::MatrixXd getCovariance() const;
  /// Returns the current covariance matrix on the row space for SVD
  Eigen::MatrixXd getRowSpaceCovariance() const;
  /// Returns the current log2 sum of the singular values
  double getSingularValuesLog2Sum() const;
  /// Returns the peak memory usage in bytes
  size_t getPeakMemoryUsage() const;
  /// Returns the current memory usage in bytes
  size_t getMemoryUsage() const;
  /// Returns the current number of flops
  double getNumFlops() const;
  /// Returns the current time for linear solving
  double getLinearSolverTime() const;
  /// Returns the current time for analyzing marginal
  double getMarginalAnalysisTime() const;
  /// Returns the current time for symbolic factorization
  double getSymbolicFactorizationTime() const;
  /// Returns the current time for numeric factorization
  double getNumericFactorizationTime() const;
  // TODO(schneith): we are missing here the Frobenius norm, available in
  //                 SPQR 3.4

 protected:
  void clearSvdAnalysisResultMembers();
  void analyzeSVD(cholmod_sparse * Omega);

  /// Linear solver options
  Options tsvd_options_;
  /// Cholmod common structure
  cholmod_common cholmod_;
  /// Caching current factorization if needed
  SuiteSparseQR_factorization<double>* factor_;
  /// Caching current estimated numerical rank for SVD
  std::ptrdiff_t svdRank_;
  /// Caching current gap in estimated singular values at the rank
  double svGap_;
  /// Caching current estimated numerical rank deficiency for SVD
  std::ptrdiff_t svdRankDeficiency_;
  /// Marginalization start index
  std::ptrdiff_t margStartIndex_;
  /// Caching current SVD tolerance
  double svdTolerance_;
  /// Caching current singular values
  Eigen::VectorXd singularValues_;
  /// Caching left-singular vectors
  Eigen::MatrixXd matrixU_;
  /// Caching right-singular vectors
  Eigen::MatrixXd matrixV_;
  /// Linear solver time
  double linearSolverTime_;
  /// Marginal analysis time
  double marginalAnalysisTime_;
};

}  // namespace truncated_svd_solver

#endif // TRUNCATED_SVD_SOLVER_TSVD_SOLVER_H

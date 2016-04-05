#ifndef TRUNCATED_SVD_SOLVER_TSVD_SOLVER_OPTIONS_H
#define TRUNCATED_SVD_SOLVER_TSVD_SOLVER_OPTIONS_H

namespace truncated_svd_solver {

/** The structure LinearSolverOptions defines the options for the
    LinearSolver class.
    \brief Linear solver options
  */
struct TruncatedSvdSolverOptions {
  TruncatedSvdSolverOptions();

  /// Perform column scaling/normalization
  bool columnScaling;
  /// Epsilon for when to consider an element being zero in the norm
  double epsNorm;
  /// Epsilon for SVD numerical rank
  double epsSVD;
  /// Epsilon for QR tolerance computation
  double epsQR;
  /// Fixed tolerance for SVD numerical rank
  double svdTol;
  /// Fixed tolerance for QR
  double qrTol;
  /// Verbose mode
  bool verbose;
};

}  // namespace truncated_svd_solver

#endif // TRUNCATED_SVD_SOLVER_TSVD_SOLVER_OPTIONS_H

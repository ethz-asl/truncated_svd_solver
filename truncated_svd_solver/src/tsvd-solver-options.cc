#include "truncated-svd-solver/tsvd-solver-options.h"

#include <limits>

namespace truncated_svd_solver {

TruncatedSvdSolverOptions::TruncatedSvdSolverOptions()
    : columnScaling(false),
      epsNorm(std::numeric_limits<double>::epsilon()),
      epsSVD(std::numeric_limits<double>::epsilon()),
      epsQR(std::numeric_limits<double>::epsilon()),
      svdTol(-1.0),
      qrTol(-1.0),
      verbose(false) {}

}  // namespace truncated_svd_solver

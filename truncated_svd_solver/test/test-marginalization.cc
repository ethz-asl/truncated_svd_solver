#include <cmath>
#include <cstddef>
#include <vector>

#include <Eigen/Core>
#include <eigen-checks/gtest.h>
#include <gtest/gtest.h>
#include <SuiteSparseQR.hpp>

#include "truncated-svd-solver/linear-algebra-helpers.h"
#include "truncated-svd-solver/marginalization.h"

namespace truncated_svd_solver {

bool getR(cholmod_sparse* A, cholmod_sparse** R, cholmod_common* cholmod) {
  CHECK_NOTNULL(A);
  CHECK_NOTNULL(R);
  CHECK_NOTNULL(cholmod);

  cholmod_sparse* qr_A = cholmod_l_transpose(A, 1, cholmod);
  SuiteSparseQR<double>(SPQR_ORDERING_FIXED, SPQR_NO_TOL, qr_A->ncol, 0, qr_A,
                        nullptr, nullptr, nullptr, nullptr, R, nullptr, nullptr,
                        nullptr, nullptr, cholmod);
  cholmod_l_free_sparse(&qr_A, cholmod);
  return cholmod->status;
}

TEST(TruncatedSvdSolver, Marginalization) {
  cholmod_common cholmod;
  cholmod_l_start(&cholmod);

  // Create a random jacobian.
  Eigen::MatrixXd J = Eigen::MatrixXd::Random(5, 5);
  Eigen::MatrixXd J_cov_expected = (J.transpose() * J).inverse();

  // Convert to cholmod sparse format.
  cholmod_sparse* Jt_cholmod =
      eigenDenseToCholmodSparseCopy(J.transpose(), &cholmod, 1e-16);

  // Test the results.
  Eigen::MatrixXd NS, CS, Sigma, SigmaP, Omega;
  const double svLogSum = marginalize(Jt_cholmod, 0, NS, CS, Sigma,
    SigmaP, Omega);

  EXPECT_NEAR(std::fabs(svLogSum),
              std::fabs(std::log2(std::fabs(J_cov_expected.determinant()))),
              1e-8);
  EXPECT_TRUE(EIGEN_MATRIX_NEAR(Sigma, J_cov_expected, 1e-12));

  cholmod_l_finish(&cholmod);
}

}  // namespace truncated_svd_solver

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();\
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  FLAGS_alsologtostderr = true;
  FLAGS_colorlogtostderr = true;
  return RUN_ALL_TESTS();
}

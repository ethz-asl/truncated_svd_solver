#include <cstddef>
#include <iostream>
#include <iomanip>

#include <cholmod.h>
#include <Eigen/Core>
#include <eigen-checks/gtest.h>
#include <gtest/gtest.h>
#include <SuiteSparseQR.hpp>

#include "truncated-svd-solver/tsvd-solver.h"
#include "truncated-svd-solver/linear-algebra-helpers.h"
#include "truncated-svd-solver/timing.h"

void evaluateSVDSPQRSolver(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
    const Eigen::VectorXd& x, double tol = 1e-9) {
  cholmod_common cholmod;
  cholmod_l_start(&cholmod);
  cholmod_sparse* A_CS = truncated_svd_solver::eigenDenseToCholmodSparseCopy(A,
    &cholmod);
  cholmod_dense b_CD;
  truncated_svd_solver::eigenDenseToCholmodDenseView(b, &b_CD);
  Eigen::VectorXd x_est;
  truncated_svd_solver::TruncatedSvdSolver linearSolver;
  for (std::ptrdiff_t i = 1; i < A.cols(); ++i) {
    linearSolver.solve(A_CS, &b_CD, i, x_est);
    double error = (b - A * x_est).norm();
    ASSERT_NEAR(error, 0, tol);
    linearSolver.getOptions().columnScaling = true;
    linearSolver.solve(A_CS, &b_CD, i, x_est);
    error = (b - A * x_est).norm();
    linearSolver.getOptions().columnScaling = false;
    ASSERT_NEAR(error, 0, tol);
  }
  cholmod_l_free_sparse(&A_CS, &cholmod);
  cholmod_l_finish(&cholmod);

  EXPECT_TRUE(EIGEN_MATRIX_NEAR(x_est, x, 1e-8));
}

void evaluateSVDSolver(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
    const Eigen::VectorXd& x) {
  const Eigen::JacobiSVD<Eigen::MatrixXd> svd(A,
    Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::VectorXd x_est = svd.solve(b);

  EXPECT_TRUE(EIGEN_MATRIX_NEAR(x_est, x, 1e-8));
}

void evaluateSPQRSolver(const Eigen::MatrixXd& A,
                                   const Eigen::VectorXd& b,
                                   const Eigen::VectorXd& x) {
  cholmod_common cholmod;
  cholmod_l_start(&cholmod);
  cholmod_sparse* A_CS =
      truncated_svd_solver::eigenDenseToCholmodSparseCopy(A, &cholmod);
  cholmod_dense b_CD;
  truncated_svd_solver::eigenDenseToCholmodDenseView(b, &b_CD);
  Eigen::VectorXd x_est;
  SuiteSparseQR_factorization<double>* factor = SuiteSparseQR_factorize<double>(
    SPQR_ORDERING_BEST, SPQR_DEFAULT_TOL, A_CS, &cholmod);
  cholmod_dense* Qtb = SuiteSparseQR_qmult<double>(SPQR_QTX, factor, &b_CD,
    &cholmod);
  cholmod_dense* x_est_cd = SuiteSparseQR_solve<double>(SPQR_RETX_EQUALS_B,
    factor, Qtb, &cholmod);
  cholmod_l_free_dense(&Qtb, &cholmod);
  truncated_svd_solver::cholmodDenseToEigenDenseCopy(x_est_cd, x_est);
  cholmod_l_free_dense(&x_est_cd, &cholmod);
  SuiteSparseQR_free(&factor, &cholmod);
  cholmod_l_free_sparse(&A_CS, &cholmod);
  cholmod_l_finish(&cholmod);

  EXPECT_TRUE(EIGEN_MATRIX_NEAR(x_est, x, 1e-8));
}

void evaluateSPQRSolverDeterminedSystem(
    const truncated_svd_solver::TruncatedSvdSolverOptions& options) {
  // Create the system.
  constexpr size_t kNumVariables = 10u;
  constexpr double kXResult = 2.0;

  Eigen::VectorXd Adiag(kNumVariables);
  for (size_t i = 0u; i < kNumVariables; ++i) {
    Adiag(i) = kNumVariables - i;
  }

  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(kNumVariables, kNumVariables);
  A.diagonal() = Adiag;

  Eigen::VectorXd b = A * Eigen::VectorXd::Constant(kNumVariables, kXResult);

  // Convert to cholmod types.
  cholmod_common_struct cholmod;
  cholmod_l_start(&cholmod);
  cholmod_sparse* A_cm =
      truncated_svd_solver::eigenDenseToCholmodSparseCopy(A, &cholmod);

  cholmod_dense b_cm;
  truncated_svd_solver::eigenDenseToCholmodDenseView(b, &b_cm);

  // Solve this system and check the results.
  truncated_svd_solver::TruncatedSvdSolver solver(options);

  for (std::ptrdiff_t i = 0u; i <= A.cols(); ++i) {
    const int num_calib_vars = kNumVariables - i;
    ASSERT_GE(num_calib_vars, 0);

    Eigen::VectorXd x;
    solver.clear();
    EXPECT_EQ(solver.getSVDRank(), -1);
    EXPECT_EQ(solver.getSVDRankDeficiency(), -1);
    EXPECT_EQ(solver.getQRRank(), 0);
    EXPECT_EQ(solver.getQRRankDeficiency(), 0);
    EXPECT_EQ(solver.getNullSpace().size(), 0);

    solver.analyzeMarginal(A_cm, i);
    EXPECT_EQ(solver.getSVDRank(), num_calib_vars);
    EXPECT_EQ(solver.getQRRank(), i);
    EXPECT_EQ(solver.getQRRankDeficiency(), 0);
    EXPECT_EQ(solver.getSVDRankDeficiency(), 0);
    EXPECT_EQ(solver.getNullSpace().size(), 0);

    solver.clear();
    EXPECT_EQ(solver.getSVDRank(), -1);
    EXPECT_EQ(solver.getSVDRankDeficiency(), -1);
    EXPECT_EQ(solver.getQRRank(), 0);
    EXPECT_EQ(solver.getQRRankDeficiency(), 0);
    EXPECT_EQ(solver.getNullSpace().size(), 0);

    solver.solve(A_cm, &b_cm, i, x);
    EXPECT_EQ(solver.getSVDRank(), num_calib_vars);
    EXPECT_EQ(solver.getQRRank(), i);
    EXPECT_EQ(solver.getQRRankDeficiency(), 0);
    EXPECT_EQ(solver.getSVDRankDeficiency(), 0);
    EXPECT_EQ(solver.getNullSpace().size(), 0);

    Eigen::VectorXd expectedSingularValues;
    if (options.columnScaling) {
      expectedSingularValues = Eigen::VectorXd::Ones(num_calib_vars);
    }else{
      expectedSingularValues =
          Adiag.tail(num_calib_vars).array() * Adiag.tail(num_calib_vars).array();
    }

    EXPECT_TRUE(
        EIGEN_MATRIX_NEAR(solver.getSingularValues(),
                          expectedSingularValues, 1e-8));
    EXPECT_TRUE(
        EIGEN_MATRIX_NEAR(x, Eigen::VectorXd::Constant(kNumVariables,
                                                       kXResult),
                          1e-8));
  }

  cholmod_l_free_sparse(&A_cm, &cholmod);
}

void buildUnderdeterminedSystem(
    const truncated_svd_solver::TruncatedSvdSolverOptions& options) {
  // Create the system.
  constexpr size_t kNumVariables = 10u;
  constexpr size_t kNumEquations = 5u;
  constexpr double kXResult = 2.3;

  Eigen::MatrixXd A(kNumEquations, kNumVariables);
  A.setRandom();
  Eigen::VectorXd b = A * Eigen::VectorXd::Constant(kNumVariables, kXResult);

  // Convert to cholmod types.
  cholmod_common_struct cholmod;
  cholmod_l_start(&cholmod);
  cholmod_sparse* A_cm =
      truncated_svd_solver::eigenDenseToCholmodSparseCopy(A, &cholmod);
  cholmod_dense b_cm;
  truncated_svd_solver::eigenDenseToCholmodDenseView(b, &b_cm);

  // Solve this system and check the results.
  truncated_svd_solver::TruncatedSvdSolver solver(options);

  Eigen::VectorXd x;
  size_t i = 8;
  solver.solve(A_cm, &b_cm, i, x);

  // Only testing the problem building here. No comparisons required.
  cholmod_l_free_sparse(&A_cm, &cholmod);
}

TEST(TruncatedSvdSolver, DeterminedSystemWithoutColumnScaling) {
  truncated_svd_solver::TruncatedSvdSolverOptions options;
  options.columnScaling = false;
  evaluateSPQRSolverDeterminedSystem(options);
}

TEST(TruncatedSvdSolver, DeterminedSystemWithColumnScaling) {
  truncated_svd_solver::TruncatedSvdSolverOptions options;
  options.columnScaling = true;
  evaluateSPQRSolverDeterminedSystem(options);
}

TEST(TruncatedSvdSolver, BuildUnderdeterminedSystem) {
  truncated_svd_solver::TruncatedSvdSolverOptions options;
  options.columnScaling = false;
  buildUnderdeterminedSystem(options);
}

TEST(TruncatedSvdSolver, OverdeterminedSystem) {
  Eigen::MatrixXd A = Eigen::MatrixXd::Random(100, 30);
  const Eigen::VectorXd x = Eigen::VectorXd::Random(30);
  Eigen::VectorXd b = A * x;

  // Standard case
  evaluateSVDSPQRSolver(A, b, x);
  evaluateSVDSolver(A, b, x);
  evaluateSPQRSolver(A, b, x);

  // Badly scaled case
  A.col(2) = 1e6 * A.col(2);
  A.col(28) = 1e6 * A.col(28);
  b = A * x;
  evaluateSVDSPQRSolver(A, b, x, 1e-3);
  evaluateSVDSolver(A, b, x);
  evaluateSPQRSolver(A, b, x);
  // TODO(schneith): Add tests for the rank-deficient case.
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  google::InitGoogleLogging(argv[0]);
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  FLAGS_alsologtostderr = true;  // NOLINT
  FLAGS_colorlogtostderr = true;  // NOLINT
  return RUN_ALL_TESTS();
}

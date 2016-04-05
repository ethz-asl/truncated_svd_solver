#include <cstddef>
#include <iostream>
#include <iomanip>

#include <eigen-checks/gtest.h>
#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <cholmod.h>
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
  for(size_t i = 0; i < kNumVariables; ++i){
    Adiag(i) =  kNumVariables - i;
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

  for (std::ptrdiff_t i = 0; i <= A.cols(); ++i) {
    const size_t num_calib_vars = kNumVariables - i;

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
    if(options.columnScaling){
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

TEST(TruncatedSvdSolver, OverdeterminedSystem) {
  Eigen::MatrixXd A = Eigen::MatrixXd::Random(100, 30);
  const Eigen::VectorXd x = Eigen::VectorXd::Random(30);
  Eigen::VectorXd b = A * x;

  std::cout << "-------------------------------------------------" << std::endl;
  std::cout << "|                  Standard case                |" << std::endl;
  std::cout << "-------------------------------------------------" << std::endl;
  evaluateSVDSPQRSolver(A, b, x);
  evaluateSVDSolver(A, b, x);
  evaluateSPQRSolver(A, b, x);

  std::cout << "-------------------------------------------------" << std::endl;
  std::cout << "|                 Badly scaled case             |" << std::endl;
  std::cout << "-------------------------------------------------" << std::endl;
  A.col(2) = 1e6 * A.col(2);
  A.col(28) = 1e6 * A.col(28);
  b = A * x;
  evaluateSVDSPQRSolver(A, b, x, 1e-3);
  evaluateSVDSolver(A, b, x);
  evaluateSPQRSolver(A, b, x);

//  std::cout << "-------------------------------------------------" << std::endl;
//  std::cout << "|                 Rank-deficient case 1         |" << std::endl;
//  std::cout << "-------------------------------------------------" << std::endl;
//  A = Eigen::MatrixXd::Random(100, 30);
//  A.col(10) = Eigen::VectorXd::Zero(A.rows());
//  b = A * x;
//  evaluateSVDSPQRSolver(A, b, x, 1e-3);
//  evaluateSVDSolver(A, b, x);
//  evaluateSPQRSolver(A, b, x);

//  std::cout << "-------------------------------------------------" << std::endl;
//  std::cout << "|                 Rank-deficient case 2         |" << std::endl;
//  std::cout << "-------------------------------------------------" << std::endl;
//  A = Eigen::MatrixXd::Random(100, 30);
//  A.col(10) = 2 * A.col(1) + 5 * A.col(20);
//  b = A * x;
//  evaluateSVDSPQRSolver(A, b, x, 1e-3);
//  evaluateSVDSolver(A, b, x);
//  evaluateSPQRSolver(A, b, x);

//  std::cout << "-------------------------------------------------" << std::endl;
//  std::cout << "|                 Near rank-deficient case 1    |" << std::endl;
//  std::cout << "-------------------------------------------------" << std::endl;
//  A = Eigen::MatrixXd::Random(100, 30);
//  A.col(10) = Eigen::VectorXd::Zero(A.rows());
//  b = A * x;
//  A.col(10) = truncated_svd_solver::NormalDistribution<100>(
//    Eigen::VectorXd::Zero(A.rows()),
//    1e-6 * Eigen::MatrixXd::Identity(A.rows(), A.rows())).getSample();
//  evaluateSVDSPQRSolver(A, b, x, 1e-3);
//  evaluateSVDSolver(A, b, x);
//  evaluateSPQRSolver(A, b, x);

//  std::cout << "-------------------------------------------------" << std::endl;
//  std::cout << "|                 Near rank-deficient case 2    |" << std::endl;
//  std::cout << "-------------------------------------------------" << std::endl;
//  A = Eigen::MatrixXd::Random(100, 30);
//  A.col(10) = 2 * A.col(1) + 5 * A.col(20);
//  b = A * x;
//  A.col(10) = truncated_svd_solver::NormalDistribution<100>(A.col(10),
//    1e-20 * Eigen::MatrixXd::Identity(A.rows(), A.rows())).getSample();
//  evaluateSVDSPQRSolver(A, b, x, 1e-3);
//  evaluateSVDSolver(A, b, x);
//  evaluateSPQRSolver(A, b, x);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();\
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  FLAGS_alsologtostderr = true;
  FLAGS_colorlogtostderr = true;
  return RUN_ALL_TESTS();
}

#ifndef TRUNCATED_SVD_SOLVER_MARGINALIZATION_H
#define TRUNCATED_SVD_SOLVER_MARGINALIZATION_H

#include <cstdlib>
#include <cstddef>

#include <cholmod.h>
#include <Eigen/Core>

namespace truncated_svd_solver {
/**
 * This function marginalizes variables from a sparse Jacobian. The Jacobian
 * is assumed to be ordered in such a way that the variables to be
 * marginalized are located to the right and start with index j.
 * \brief Variables marginalization
 *
 * \param[in] Jt Jacobian transpose as outputted by linear solvers
 * \param[in] j index from where to marginalize
 * \param[in] normTol tolerance for a zero norm column
 * \param[in] epsTol tolerance for SVD tolerance computation
 * \param[out] NS null space of the marginalized system
 * \param[out] CS column space of the marginalized system
 * \param[out] Sigma covariance of the marginalized system
 * \param[out] SigmaP projected covariance of the marginalized system
 * \param[out] Omega marginalized Fisher information matrix
 * \return sum of the log of the singular values of the marginalized system
 */
double marginalize(cholmod_sparse* Jt, size_t j, Eigen::MatrixXd& NS,
                   Eigen::MatrixXd& CS, Eigen::MatrixXd& Sigma,
                   Eigen::MatrixXd& SigmaP, Eigen::MatrixXd& Omega,
                   double normTol = 1e-8, double epsTol = 1e-4);

/**
 * This function returns the marginal Jacobian from two submatrices.
 * \brief Marginal Jacobian recovery
 *
 * \return marginal Jacobian
 * \param[in] J_x is the Jacobian containing the state variables
 * \param[in] J_thetat is the tranposed Jacobian containing the calibration
 *            variables
 */
Eigen::MatrixXd marginalJacobian(cholmod_sparse* J_x, cholmod_sparse*
  J_thetat, cholmod_common* cholmod);

}  // namespace truncated_svd_solver

#endif // TRUNCATED_SVD_SOLVER_MARGINALIZATION_H

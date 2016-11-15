#ifndef TRUNCATED_SVD_SOLVER_CHOLMODASD_HELPERS_H
#define TRUNCATED_SVD_SOLVER_CHOLMODASD_HELPERS_H

#include <cholmod.h>

#if CHOLMOD_MAIN_VERSION < 3
#define USE_OLD_INTERFACE
#endif

namespace truncated_svd_solver {

double& Get_SPQR_AnalyzeTime(cholmod_common& choldmod) {
#ifdef USE_OLD_INTERFACE
  return choldmod.other1[1];
#else
  return choldmod.SPQR_analyze_time;
#endif
}

double Get_SPQR_AnalyzeTime(const cholmod_common& choldmod) {
#ifdef USE_OLD_INTERFACE
  return choldmod.other1[1];
#else
  return choldmod.SPQR_analyze_time;
#endif
}

double& Get_SPQR_FactorizeTime(cholmod_common& choldmod) {
#ifdef USE_OLD_INTERFACE
  return choldmod.other1[2];
#else
  return choldmod.SPQR_factorize_time;
#endif
}

double Get_SPQR_FactorizeTime(const cholmod_common& choldmod) {
#ifdef USE_OLD_INTERFACE
  return choldmod.other1[2];
#else
  return choldmod.SPQR_factorize_time;
#endif
}

double& Get_SPQR_FlopcountBound(cholmod_common& choldmod) {
#ifdef USE_OLD_INTERFACE
  return choldmod.SPQR_xstat[0];
#else
  return choldmod.SPQR_flopcount_bound;
#endif
}

double Get_SPQR_FlopcountBound(const cholmod_common& choldmod) {
#ifdef USE_OLD_INTERFACE
  return choldmod.SPQR_xstat[0];
#else
  return choldmod.SPQR_flopcount_bound;
#endif
}

}  // namespace truncated_svd_solver
#undef USE_OLD_INTERFACE
#endif  // TRUNCATED_SVD_SOLVER_CHOLMODASD_HELPERS_H

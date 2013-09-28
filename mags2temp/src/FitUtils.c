#include "FitUtils.h"

#include <math.h>

#include <gsl/gsl_pow_int.h>

double fitAmplitude(const double hc_kt, void* params) {
  fitParams* fParams = (fitParams*)params;
  const double jFit = planckIntegral(hc_kt, fParams->jParams);
  const double hFit = planckIntegral(hc_kt, fParams->hParams);
  const double kFit = planckIntegral(hc_kt, fParams->kParams);
  const double jVal = fParams->jVal;
  const double hVal = fParams->hVal;
  const double kVal = fParams->kVal;
  const double jSig = fParams->jSig;
  const double hSig = fParams->hSig;
  const double kSig = fParams->kSig;
  return (jFit*jVal/gsl_pow_2(jSig) + hFit*hVal/gsl_pow_2(hSig) +
      kFit*kVal/gsl_pow_2(kSig)) / (gsl_pow_2(jFit)/gsl_pow_2(jSig) +
      gsl_pow_2(hFit)/gsl_pow_2(hSig) + gsl_pow_2(kFit)/gsl_pow_2(kSig));
}

double fitResidual(double hc_kt, void* params) {
  fitParams* fParams = (fitParams*)params;

  const double jFit = planckIntegral(hc_kt, fParams->jParams);
  const double hFit = planckIntegral(hc_kt, fParams->hParams);
  const double kFit = planckIntegral(hc_kt, fParams->kParams);
  const double jVal = fParams->jVal;
  const double hVal = fParams->hVal;
  const double kVal = fParams->kVal;
  const double jSig = fParams->jSig;
  const double hSig = fParams->hSig;
  const double kSig = fParams->kSig;

  // Duplicate of fitAmplitude
  const double amp = (jFit*jVal/gsl_pow_2(jSig) + hFit*hVal/gsl_pow_2(hSig) +
      kFit*kVal/gsl_pow_2(kSig)) / (gsl_pow_2(jFit)/gsl_pow_2(jSig) +
      gsl_pow_2(hFit)/gsl_pow_2(hSig) + gsl_pow_2(kFit)/gsl_pow_2(kSig));

  return gsl_pow_2((amp*jFit - jVal)/jSig) + gsl_pow_2((amp*hFit - hVal)/hSig) +
      gsl_pow_2((amp*kFit - kVal)/kSig);
}

int myMinimizerSetup(gsl_min_fminimizer* minimizer, gsl_function* func,
    const double xmin, const double xlo, const double xhi) {
  double myXmin = xmin;
  double myXlo = xlo;
  double myXhi = xhi;
  double fmin = GSL_FN_EVAL(func, myXmin);
  double flo = GSL_FN_EVAL(func, myXlo);
  double fhi = GSL_FN_EVAL(func, myXhi);
  int iter = 0;
  const int maxIter = 16;
  while (((fmin > flo) || (fmin > fhi)) && (iter < maxIter)) {
    if (fmin > flo) {
      myXhi = myXmin;
      fhi = fmin;
      myXmin = myXlo + 0.5*(myXmin - myXlo);
      fmin = GSL_FN_EVAL(func, myXmin);
    } else {
      myXlo = myXmin;
      flo = fmin;
      myXmin = myXmin + 0.5*(myXhi - myXmin);
      fmin = GSL_FN_EVAL(func, myXmin);
    }
    ++iter;
  }

  return gsl_min_fminimizer_set_with_values(minimizer, func, myXmin, fmin,
      myXlo, flo, myXhi, fhi);
}

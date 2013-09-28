#include "PlanckUtils.h"

#include <math.h>

#include <gsl/gsl_const_mksa.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_pow_int.h>

double planck(const double x, void* params) {
  const double hc_kt = *(double *)params;
  return (1.0 / expm1(hc_kt / x)) / gsl_pow_5(x);
}

double planckIntegral(double hc_kt, void* params) {
  integralParams* iParams = (integralParams*)params;

  /*
  gsl_function func;
  func.function = &planck;
  func.params = &hc_kt;

  return gsl_integration_glfixed(&func, iParams->xmin, iParams->xmax,
      iParams->gltable);
  */

  return iParams->bw * evalNormPlanck(hc_kt, iParams->xiso);
}

double planckRatio(double hc_kt, void* params) {
  ratioParams* rParams = (ratioParams*)params;
  const double ans1 = planckIntegral(hc_kt, &rParams->params1);
  const double ans2 = planckIntegral(hc_kt, &rParams->params2);
  return ans1 / ans2 - rParams->ratio;
}

double tempFromHckt(const double hc_kt) {
  return GSL_CONST_MKSA_SPEED_OF_LIGHT * GSL_CONST_MKSA_PLANCKS_CONSTANT_H /
      (GSL_CONST_MKSA_BOLTZMANN * hc_kt);
}

double hcktFromTemp(const double temp) {
  return GSL_CONST_MKSA_SPEED_OF_LIGHT * GSL_CONST_MKSA_PLANCKS_CONSTANT_H /
      (GSL_CONST_MKSA_BOLTZMANN * temp);
}

double evalNormPlanck(const double hc_kt, const double lambda) {
  double hcktParam = hc_kt;
  gsl_function func;
  func.function = &planck;
  func.params = &hcktParam;
  return GSL_FN_EVAL(&func, lambda);
}

double evalPlanck(const double hc_kt, const double lambda, const double amp) {
  return amp*evalNormPlanck(hc_kt, lambda);
}

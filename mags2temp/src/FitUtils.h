#ifndef FITUTILS_H_
#define FITUTILS_H_

#include <gsl/gsl_math.h>
#include <gsl/gsl_min.h>

#include "PlanckUtils.h"

double fitAmplitude(double hc_kt, void* params);
double fitResidual(double hc_kt, void* params);
int myMinimizerSetup(gsl_min_fminimizer* minimizer, gsl_function* func,
    double xmin, double xlo, double xhi);

typedef struct {
  integralParams* jParams;
  integralParams* hParams;
  integralParams* kParams;
  double jVal;
  double jSig;
  double hVal;
  double hSig;
  double kVal;
  double kSig;
} fitParams;

#endif  // FITUTILS_H_

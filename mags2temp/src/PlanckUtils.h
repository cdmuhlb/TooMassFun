#ifndef PLANCKUTILS_H_
#define PLANCKUTILS_H_

#include <gsl/gsl_integration.h>

double planck(double x, void* params);
double planckIntegral(double hc_kt, void* params);
double planckRatio(double hc_kt, void* params);
double tempFromHckt(double hc_kt);
double hcktFromTemp(double temp);

typedef struct {
  double xmin;
  double xmax;
  double xiso;
  double bw;
  gsl_integration_glfixed_table* gltable;
} integralParams;

typedef struct {
  double ratio;
  integralParams params1;
  integralParams params2;
} ratioParams;

double evalNormPlanck(double hc_kt, double lambda);
double evalPlanck(double hc_kt, double lambda, double amp);

#endif  // PLANCKUTILS_H_

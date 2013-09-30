#include "CieXyz.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <gsl/gsl_math.h>

#include "PlanckUtils.h"

double cmf_integrate(const double* lambdas, const double* cmfs, int size,
    gsl_function* func);
double cmf_integratePlanck(const double* lambdas, const double* cmfs, int size,
    double hc_kt, double amp);

matchingTable* cmf_alloc() {
  // TODO: Avoid hard-coded values
  const char* filename = "share/lin2012xyz2e_5_7sf.txt";
  const int size = 89;

  matchingTable* cmf = (matchingTable*)malloc(sizeof(matchingTable));
  cmf->size = size;
  cmf->lambdas = (double*)malloc(size*sizeof(double));
  cmf->cmfX = (double*)malloc(size*sizeof(double));
  cmf->cmfY = (double*)malloc(size*sizeof(double));
  cmf->cmfZ = (double*)malloc(size*sizeof(double));

  FILE* datafile = fopen(filename, "r");
  for (int i=0; i<size; ++i) {
    double lambda;
    double x;
    double y;
    double z;
    const int nRead = fscanf(datafile, "%lf %lf %lf %lf", &lambda, &x, &y, &z);
    assert(nRead == 4);
    cmf->lambdas[i] = lambda*1.0e-9;
    cmf->cmfX[i] = x;
    cmf->cmfY[i] = y;
    cmf->cmfZ[i] = z;
  }
  fclose(datafile);

  return cmf;
}

double cmf_integrate(const double* lambdas, const double* cmfs, const int size,
    gsl_function* func) {
  double sum = 0.0;
  for (int i=0; i<(size - 1); ++i) {
    sum += (lambdas[i+1] - lambdas[i]) *
        (cmfs[i+1]*GSL_FN_EVAL(func, lambdas[i+1]) +
         cmfs[i]*GSL_FN_EVAL(func, lambdas[i]));
  }
  return 0.5*sum;
}

double cmf_integratePlanck(const double* lambdas, const double* cmfs,
    const int size, const double hc_kt, const double amp) {
  double sum = 0.0;
  for (int i=0; i<(size - 1); ++i) {
    sum += (lambdas[i+1] - lambdas[i]) *
        (cmfs[i+1]*evalPlanck(hc_kt, lambdas[i+1], amp) +
         cmfs[i]*evalPlanck(hc_kt, lambdas[i], amp));
  }
  return 0.5*sum;
}

double cmf_cieX(matchingTable* cmf, gsl_function* func) {
  return cmf_integrate(cmf->lambdas, cmf->cmfX, cmf->size, func);
}

double cmf_cieY(matchingTable* cmf, gsl_function* func) {
  return cmf_integrate(cmf->lambdas, cmf->cmfY, cmf->size, func);
}

double cmf_cieZ(matchingTable* cmf, gsl_function* func) {
  return cmf_integrate(cmf->lambdas, cmf->cmfZ, cmf->size, func);
}

double planckCieX(matchingTable* cmf, const double hc_kt, const double amp) {
  return cmf_integratePlanck(cmf->lambdas, cmf->cmfX, cmf->size, hc_kt, amp);
}

double planckCieY(matchingTable* cmf, const double hc_kt, const double amp) {
  return cmf_integratePlanck(cmf->lambdas, cmf->cmfY, cmf->size, hc_kt, amp);
}

double planckCieZ(matchingTable* cmf, const double hc_kt, const double amp) {
  return cmf_integratePlanck(cmf->lambdas, cmf->cmfZ, cmf->size, hc_kt, amp);
}

void cmf_free(matchingTable* cmf) {
  free(cmf->cmfZ);
  free(cmf->cmfY);
  free(cmf->cmfX);
  free(cmf->lambdas);
  free(cmf);
}

#include "CieXyz.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <gsl/gsl_math.h>

#include "PlanckUtils.h"

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

double cmf_cieY(matchingTable* cmf, gsl_function* func) {
  double sum = 0.0;
  for (int i=0; i<(cmf->size - 1); ++i) {
    sum += (cmf->lambdas[i+1] - cmf->lambdas[i]) *
        (cmf->cmfY[i+1]*GSL_FN_EVAL(func, cmf->lambdas[i+1]) +
         cmf->cmfY[i]*GSL_FN_EVAL(func, cmf->lambdas[i]));
  }
  return 0.5*sum;
}

double planckCieY(matchingTable* cmf, const double hc_kt, const double amp) {
  double sum = 0.0;
  for (int i=0; i<(cmf->size - 1); ++i) {
    sum += (cmf->lambdas[i+1] - cmf->lambdas[i]) *
        (cmf->cmfY[i+1]*evalPlanck(hc_kt, cmf->lambdas[i+1], amp) +
         cmf->cmfY[i]*evalPlanck(hc_kt, cmf->lambdas[i], amp));
  }
  return 0.5*sum;
}

void cmf_free(matchingTable* cmf) {
  free(cmf->cmfZ);
  free(cmf->cmfY);
  free(cmf->cmfX);
  free(cmf->lambdas);
  free(cmf);
}

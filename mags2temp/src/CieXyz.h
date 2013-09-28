#ifndef CIEXYZ_H_
#define CIEXYZ_H_

#include <gsl/gsl_math.h>

typedef struct {
  double* lambdas;
  double* cmfX;
  double* cmfY;
  double* cmfZ;
  int size;
} matchingTable;

matchingTable* cmf_alloc();
double cmf_cieY(matchingTable* cfm, gsl_function* func);
void cmf_free(matchingTable* cmf);

double planckCieY(matchingTable* cmf, double hc_kt, double amp);

#endif  // CIEXYZ_H_

#include "MagUtils.h"

#include <math.h>

#include <gsl/gsl_math.h>
#include <gsl/gsl_pow_int.h>

#include "CieXyz.h"
#include "Filters.h"
#include "PlanckUtils.h"

double magFromAmp(const double hc_kt, const double amp) {
  const double bv = evalPlanck(hc_kt, FILTER_V_LAMBDA, amp);
  return -2.5*log10(bv / FILTER_V_F0);
}

double ampFromMag(const double hc_kt, const double mag) {
  const double bv = FILTER_V_F0 * pow(10.0, -0.4*mag);
  return bv / evalNormPlanck(hc_kt, FILTER_V_LAMBDA);
}

double flowerTemp(const double jVal, const double hVal, const double kVal) {
  const double bv = 1.103*(jVal - hVal) + 0.486*(hVal - kVal) + 0.228;
  const double logTemp = 3.979145106714099 +
      bv*(-0.654992268598245 +
      bv*( 1.740690042385095 +
      bv*(-4.608815154057166 +
      bv*( 6.792599779944473 +
      bv*(-5.396909891322525 +
      bv*( 2.192970376522490 +
      bv*(-0.359495739295671)))))));
  return pow(10.0, logTemp);
}

double jMagFromFlux(const double jFlux) {
  return -2.5*log10((jFlux/FILTER_J_BANDWIDTH) / FILTER_J_F0);
}

double hMagFromFlux(const double hFlux) {
  return -2.5*log10((hFlux/FILTER_H_BANDWIDTH) / FILTER_H_F0);
}

double kMagFromFlux(const double kFlux) {
  return -2.5*log10((kFlux/FILTER_K_BANDWIDTH) / FILTER_K_F0);
}

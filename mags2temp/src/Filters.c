#include "Filters.h"

integralParams makeJFilter(gsl_integration_glfixed_table* gltable) {
  const double jLambdaIso = FILTER_J_LAMBDA;
  const double jBandwidth = FILTER_J_BANDWIDTH;
  integralParams jParams;
  jParams.xmin = jLambdaIso - 0.5*jBandwidth;
  jParams.xmax = jLambdaIso + 0.5*jBandwidth;
  jParams.xiso = jLambdaIso;
  jParams.bw = jBandwidth;
  jParams.gltable = gltable;
  return jParams;
}

integralParams makeHFilter(gsl_integration_glfixed_table* gltable) {
  const double hLambdaIso = FILTER_H_LAMBDA;
  const double hBandwidth = FILTER_H_BANDWIDTH;
  integralParams hParams;
  hParams.xmin = hLambdaIso - 0.5*hBandwidth;
  hParams.xmax = hLambdaIso + 0.5*hBandwidth;
  hParams.xiso = hLambdaIso;
  hParams.bw = hBandwidth;
  hParams.gltable = gltable;
  return hParams;
}

integralParams makeKFilter(gsl_integration_glfixed_table* gltable) {
  const double kLambdaIso = FILTER_K_LAMBDA;
  const double kBandwidth = FILTER_K_BANDWIDTH;
  integralParams kParams;
  kParams.xmin = kLambdaIso - 0.5*kBandwidth;
  kParams.xmax = kLambdaIso + 0.5*kBandwidth;
  kParams.xiso = kLambdaIso;
  kParams.bw = kBandwidth;
  kParams.gltable = gltable;
  return kParams;
}

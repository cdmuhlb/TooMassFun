#ifndef FILTERS_H_
#define FILTERS_H_

#include <gsl/gsl_integration.h>

#include "PlanckUtils.h"

#define FILTER_V_LAMBDA (550.0e-9)
#define FILTER_J_LAMBDA (1.235e-6)
#define FILTER_H_LAMBDA (1.662e-6)
#define FILTER_K_LAMBDA (2.159e-6)

#define FILTER_J_BANDWIDTH (0.162e-6)
#define FILTER_H_BANDWIDTH (0.251e-6)
#define FILTER_K_BANDWIDTH (0.262e-6)

#define FILTER_V_F0 (3.60742e-12)
#define FILTER_J_F0 (3.129e-13)
#define FILTER_H_F0 (1.133e-13)
#define FILTER_K_F0 (4.283e-14)

integralParams makeJFilter(gsl_integration_glfixed_table* gltable);
integralParams makeHFilter(gsl_integration_glfixed_table* gltable);
integralParams makeKFilter(gsl_integration_glfixed_table* gltable);

#endif  // FILTERS_H_

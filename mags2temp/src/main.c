#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_min.h>
#include <gsl/gsl_pow_int.h>

#include "BufferedWriter.h"
#include "CieXyz.h"
#include "Filters.h"
#include "FitUtils.h"
#include "MagUtils.h"
#include "PlanckUtils.h"

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "Usage: mags2temp <input_file> <output_file>\n");
    return EXIT_FAILURE;
  }

  gsl_integration_glfixed_table* gltable =
      gsl_integration_glfixed_table_alloc(6);

  integralParams jParams = makeJFilter(gltable);
  integralParams hParams = makeHFilter(gltable);
  integralParams kParams = makeKFilter(gltable);

  matchingTable* cmf = cmf_alloc();

  // Minimization setup
  gsl_min_fminimizer* minimizer =
      gsl_min_fminimizer_alloc(gsl_min_fminimizer_brent); 
  gsl_function minFunc;
  minFunc.function = &fitResidual;

  // Disable error handler - be sure to check return codes!
  gsl_set_error_handler_off();

  fitParams fParams;
  fParams.jParams = &jParams;
  fParams.hParams = &hParams;
  fParams.kParams = &kParams;

  // Read data
  const int nFields = 6;
  const char* inputFilename = argv[1];
  const char* outputFilename = argv[2];
  FILE* in = fopen(inputFilename, "r");
  BufferedFloatWriter* bfw = bfw_new(outputFilename);
  float buf[nFields];
  int nRead = fread(buf, sizeof(float), nFields, in);
  int nRows = 0;
  int nFailures = 0;
  while (nRead == nFields) {
    fParams.jVal = (double)buf[0];
    fParams.jSig = (double)buf[1];
    fParams.hVal = (double)buf[2];
    fParams.hSig = (double)buf[3];
    fParams.kVal = (double)buf[4];
    fParams.kSig = (double)buf[5];

    // Fit values
    minFunc.params = &fParams;
    double hcktLo = 1.4e-7;     // T = 100,000 K
    double hcktHi = 1.4e-5;     // T = 1,000 K
    double hcktAns = 4.0e-6;
    
    int status;
    status = myMinimizerSetup(minimizer, &minFunc, hcktAns,
        hcktLo, hcktHi);
    if (status == GSL_SUCCESS) {
      do {
        status = gsl_min_fminimizer_iterate(minimizer);
        hcktAns = gsl_min_fminimizer_x_minimum(minimizer);
        hcktLo = gsl_min_fminimizer_x_lower(minimizer);
        hcktHi = gsl_min_fminimizer_x_upper(minimizer);
        status = gsl_min_test_interval(hcktLo, hcktHi, 1.0e-12, 1.0e-6);
      } while (status == GSL_CONTINUE);

      const double temp_final = tempFromHckt(hcktAns);
      const double amp_final = fitAmplitude(hcktAns, &fParams);
      const double chisq_final = gsl_min_fminimizer_f_minimum(minimizer);
      const double cieY = planckCieY(cmf, hcktAns, amp_final);
      const double aMag = magFromAmp(hcktAns, amp_final);
      bfw_put(bfw, (float)temp_final);
      bfw_put(bfw, (float)cieY);
      printf("%g %g %g %g\n", temp_final, aMag, cieY, chisq_final);

    } else {
      ++nFailures;
    }

    
    // Flower values
    /*
    const double jMag = jMagFromFlux(fParams.jVal);
    const double hMag = hMagFromFlux(fParams.hVal);
    const double kMag = kMagFromFlux(fParams.kVal);
    const double bt = flowerTemp(jMag, hMag, kMag);
    if ((bt > 1000.0) && (bt < 100000.0)) {
      const double bhckt = hcktFromTemp(bt);
      //const double bm = jMag;
      //const double ba = ampFromMag(bhckt, bm);
      //const double by = planckCieY(cmf, bhckt, ba);
      //bfw_put(bfw, (float)bt);
      //bfw_put(bfw, (float)by);
      //printf("%g %g %g %g\n", bt, bm, by, 0.0);

      // Fit amplitude from Flower temperature
      const double ba2 = fitAmplitude(bhckt, &fParams);
      const double by2 = planckCieY(cmf, bhckt, ba2);
      const double bm2 = magFromAmp(bhckt, ba2);
      bfw_put(bfw, (float)bt);
      bfw_put(bfw, (float)by2);
      printf("%g %g %g %g\n", bt, bm2, by2, 0.0);
    } else {
      ++nFailures;
    }
    */

    nRead = fread(buf, sizeof(float), nFields, in);
    ++nRows;
  }
  bfw_close(bfw);
  fclose(in);
  fprintf(stderr, "Failures: %d (out of %d)\n", nFailures, nRows);

  gsl_min_fminimizer_free(minimizer);
  cmf_free(cmf);
  gsl_integration_glfixed_table_free(gltable);

  return EXIT_SUCCESS;
}

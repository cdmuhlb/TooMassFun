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
  if (argc != 2) {
    fprintf(stderr, "Usage: mags2temp <output_dir>\n");
    return EXIT_FAILURE;
  }
  const char* outputDir = argv[1];

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
  const int nFields = 8;
  FILE* in = stdin;
  BufferedFloatWriter* lonOut = bfw_newFile(outputDir, "tmfLon.dat");
  BufferedFloatWriter* latOut = bfw_newFile(outputDir, "tmfLat.dat");
  BufferedFloatWriter* cieXOut = bfw_newFile(outputDir, "tmfCieX.dat");
  BufferedFloatWriter* cieYOut = bfw_newFile(outputDir, "tmfCieY.dat");
  BufferedFloatWriter* cieZOut = bfw_newFile(outputDir, "tmfCieZ.dat");

  float buf[nFields];
  int nRead = fread(buf, sizeof(float), nFields, in);
  int nRows = 0;
  int nFailures = 0;
  while (nRead == nFields) {
    const float lon = buf[0];
    const float lat = buf[1];
    fParams.jVal = (double)buf[2];
    fParams.jSig = (double)buf[3];
    fParams.hVal = (double)buf[4];
    fParams.hSig = (double)buf[5];
    fParams.kVal = (double)buf[6];
    fParams.kSig = (double)buf[7];


    // Option 1: Fit for black-body temperature
    minFunc.params = &fParams;
    double hcktLo = 1.4e-7;     // T = 100,000 K
    double hcktHi = 1.4e-5;     // T = 1,000 K
    double hcktAns = 4.0e-6;    // T = 3,600 K

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

      //const double temp_final = tempFromHckt(hcktAns);
      const double amp_final = fitAmplitude(hcktAns, &fParams);
      //const double chisq_final = gsl_min_fminimizer_f_minimum(minimizer);
      //const double aMag = magFromAmp(hcktAns, amp_final);
      const double cieX = planckCieX(cmf, hcktAns, amp_final);
      const double cieY = planckCieY(cmf, hcktAns, amp_final);
      const double cieZ = planckCieZ(cmf, hcktAns, amp_final);

      bfw_put(lonOut, lon);
      bfw_put(latOut, lat);
      bfw_put(cieXOut, (float)cieX);
      bfw_put(cieYOut, (float)cieY);
      bfw_put(cieZOut, (float)cieZ);
    } else {
      ++nFailures;
    }

    /*
    // Option 2: Get T_eff (Flower 1996, Torres 2007) from B-V
    //   (Bilir et al. 2008)
    const double jMag = jMagFromFlux(fParams.jVal);
    const double hMag = hMagFromFlux(fParams.hVal);
    const double kMag = kMagFromFlux(fParams.kVal);
    const double bt = flowerTemp(jMag, hMag, kMag);
    if ((bt > 1000.0) && (bt < 100000.0)) {
      const double bhckt = hcktFromTemp(bt);

      // Take V-band magnitude from J-band
      //const double bm = jMag;
      //const double ba = ampFromMag(bhckt, bm);

      // Fit amplitude from Flower temperature
      const double ba2 = fitAmplitude(bhckt, &fParams);
      //const double bm2 = magFromAmp(bhckt, ba2);

      const double cieX = planckCieX(cmf, bhckt, ba2);
      const double cieY = planckCieY(cmf, bhckt, ba2);
      const double cieZ = planckCieZ(cmf, bhckt, ba2);
      bfw_put(lonOut, lon);
      bfw_put(latOut, lat);
      bfw_put(cieXOut, (float)cieX);
      bfw_put(cieYOut, (float)cieY);
      bfw_put(cieZOut, (float)cieZ);
    } else {
      ++nFailures;
    }
    */

    // Loop update
    ++nRows;
    nRead = fread(buf, sizeof(float), nFields, in);
  }
  fclose(in);
  bfw_close(lonOut);
  bfw_close(latOut);
  bfw_close(cieXOut);
  bfw_close(cieYOut);
  bfw_close(cieZOut);
  fprintf(stderr, "Failures: %d (out of %d)\n", nFailures, nRows);

  gsl_min_fminimizer_free(minimizer);
  cmf_free(cmf);
  gsl_integration_glfixed_table_free(gltable);

  return EXIT_SUCCESS;
}

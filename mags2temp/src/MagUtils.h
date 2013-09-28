#ifndef MAGUTILS_H_
#define MAGUTILS_H_

double magFromAmp(double hc_kt, double amp);
double ampFromMag(double hc_kt, double mag);
double flowerTemp(double jVal, double hVal, double kVal);

double jMagFromFlux(double jFlux);
double hMagFromFlux(double hFlux);
double kMagFromFlux(double kFlux);

#endif  // MAGUTILS_H_

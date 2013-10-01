TooMassFun
==========
TooMassFun is a collection of programs and scripts for processing and analyzing
data from the [Two Micron All Sky Survey
(2MASS)](http://www.ipac.caltech.edu/2mass/).  The focus is on inferring visible
colors and rendering images of the sky.

Disclaimer: The routines provided here are for exploratory purposes only and are
**not** suitable for scientific work.

Dependencies
------------
TooMassFun contains components written in [Scala](http://www.scala-lang.org/),
C, C++, [CUDA](https://developer.nvidia.com/cuda-toolkit), and
[R](http://www.r-project.org/).  These components depend on the following
libraries and build tools:

* [sbt](http://www.scala-sbt.org/)
* Make
* [GSL (GNU Scientific Library)](http://www.gnu.org/software/gsl/)
* [OpenCV](http://opencv.org/)

Components
----------
TooMassFun consists of the following utilities:

* `PscConvert` (Scala): Parse 2MASS Point Source Catalog (PSC) data files and
  provide input to `mags2temp`
* `mags2temp` (C): Perform a black-body fit on J,H,Ks magnitudes to estimate
  color and brightness
* `SkyRender` (C++, CUDA): Render images of the night sky using output of
  `mags2temp`

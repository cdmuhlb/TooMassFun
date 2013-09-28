#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>

#include <cuda_runtime.h>
#include <vector_types.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "utils.h"
#include "kernels.h"

int main(int argc, char** argv) {
  // Process arguments
  if (argc != 3) {
    fprintf(stderr, "Usage: colorarray <input_file> <output_file>\n");
    return EXIT_FAILURE;
  }
  const std::string input_file = std::string(argv[1]);
  const std::string output_file = std::string(argv[2]);

  // Read input data
  int nRecords;
  float2* d_tempAmp;
  {
    struct stat fileStats;
    stat(input_file.c_str(), &fileStats);
    assert(fileStats.st_size % (2*sizeof(float)) == 0);
    nRecords = fileStats.st_size / (2*sizeof(float));
    float* h_tempAmp = new float[2*nRecords];
    FILE* in = fopen(input_file.c_str(), "r");
    const int nRead = fread(h_tempAmp, sizeof(float), 2*nRecords, in);
    assert(nRead == 2*nRecords);
    fclose(in);
    checkCudaErrors(cudaMalloc(&d_tempAmp, nRecords*sizeof(float2)));
    checkCudaErrors(cudaMemcpy(d_tempAmp, h_tempAmp, 2*nRecords*sizeof(float),
        cudaMemcpyHostToDevice));
    delete[] h_tempAmp;
  }

  uchar3* d_bgr;
  checkCudaErrors(cudaMalloc(&d_bgr, nRecords*sizeof(uchar3)));
  ColorArray(d_tempAmp, d_bgr, nRecords);
  cudaFree(d_tempAmp);
  const int dim = static_cast<int>(sqrt(nRecords));
  const int nPixels = dim*dim;
  assert(nPixels <= nRecords);
  cv::Mat image;
  image.create(dim, dim, CV_8UC3);
  checkCudaErrors(cudaMemcpy((uchar3*)image.ptr<unsigned char>(0),
      d_bgr, nPixels*sizeof(uchar3), cudaMemcpyDeviceToHost));
  cudaFree(d_bgr);
  cv::imwrite(output_file, image);

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  return EXIT_SUCCESS;
}

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>

#include <cuda_runtime.h>
#include <vector_types.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "utils.h"
#include "kernels.h"

int main(int argc, char** argv) {
  // Process arguments
  if (argc != 3) {
    fprintf(stderr, "Usage: colorarray <input_dir> <output_file>\n");
    return EXIT_FAILURE;
  }
  const std::string input_dir = std::string(argv[1]);
  const std::string output_file = std::string(argv[2]);

  // Read input data
  int nRecords;
  float* d_lon;
  float* d_lat;
  float* d_cieX;
  float* d_cieY;
  float* d_cieZ;
  {
    struct stat fileStats;
    std::string lonFilename = input_dir + "/tmfLon.dat";
    stat(lonFilename.c_str(), &fileStats);
    assert(fileStats.st_size % sizeof(float) == 0);
    nRecords = fileStats.st_size / sizeof(float);

    float* h_buf = new float[nRecords];

    { FILE* in = fopen(lonFilename.c_str(), "r");
      const int nRead = fread(h_buf, sizeof(float), nRecords, in);
      assert(nRead == nRecords);
      fclose(in);
      checkCudaErrors(cudaMalloc(&d_lon, nRecords*sizeof(float)));
      checkCudaErrors(cudaMemcpy(d_lon, h_buf, nRecords*sizeof(float),
          cudaMemcpyHostToDevice)); }
    { FILE* in = fopen((input_dir + "/tmfLat.dat").c_str(), "r");
      const int nRead = fread(h_buf, sizeof(float), nRecords, in);
      assert(nRead == nRecords);
      fclose(in);
      checkCudaErrors(cudaMalloc(&d_lat, nRecords*sizeof(float)));
      checkCudaErrors(cudaMemcpy(d_lat, h_buf, nRecords*sizeof(float),
          cudaMemcpyHostToDevice)); }
    { FILE* in = fopen((input_dir + "/tmfCieX.dat").c_str(), "r");
      const int nRead = fread(h_buf, sizeof(float), nRecords, in);
      assert(nRead == nRecords);
      fclose(in);
      checkCudaErrors(cudaMalloc(&d_cieX, nRecords*sizeof(float)));
      checkCudaErrors(cudaMemcpy(d_cieX, h_buf, nRecords*sizeof(float),
          cudaMemcpyHostToDevice)); }
    { FILE* in = fopen((input_dir + "/tmfCieY.dat").c_str(), "r");
      const int nRead = fread(h_buf, sizeof(float), nRecords, in);
      assert(nRead == nRecords);
      fclose(in);
      checkCudaErrors(cudaMalloc(&d_cieY, nRecords*sizeof(float)));
      checkCudaErrors(cudaMemcpy(d_cieY, h_buf, nRecords*sizeof(float),
          cudaMemcpyHostToDevice)); }
    { FILE* in = fopen((input_dir + "/tmfCieZ.dat").c_str(), "r");
      const int nRead = fread(h_buf, sizeof(float), nRecords, in);
      assert(nRead == nRecords);
      fclose(in);
      checkCudaErrors(cudaMalloc(&d_cieZ, nRecords*sizeof(float)));
      checkCudaErrors(cudaMemcpy(d_cieZ, h_buf, nRecords*sizeof(float),
          cudaMemcpyHostToDevice)); }

    delete[] h_buf;
  }

  //const int dim = static_cast<int>(sqrt(nRecords));
  //assert(nPixels <= nRecords);
  const int dimx = 3200;
  const int dimy = 15;
  const int nPixels = dimx*dimy;
  //uchar3* d_bgr;
  //checkCudaErrors(cudaMalloc(&d_bgr, nRecords*sizeof(uchar3)));
  //ColorArray(d_tempAmp, d_bgr, nRecords);

  float q95;
  // Wasteful way to get 95th percentile
  { thrust::device_vector<float> dvec(nRecords);
    thrust::device_ptr<float> dev_ptr(d_cieY);
    thrust::copy(dev_ptr, dev_ptr + nRecords, dvec.begin());
    thrust::sort(dvec.begin(), dvec.end());
    q95 = dvec[static_cast<int>(0.9995*nRecords)];
  }
  printf("q = %g\n", q95);

  //const float minLat = -90.0f;  // aaa
  //const float maxLat = -74.5f;  // aaa
  const float minLat = -4.9f;  // acc
  const float maxLat = -3.2f;  // acc
  //const float minLat = -1.6f;  // ace
  //const float maxLat =  0.0f;  // ace
  float3 fblack; fblack.x = 0.0f; fblack.y = 0.0f; fblack.z = 0.0f;
  thrust::device_vector<float3> dv_xyz(nPixels, fblack);
  RenderStars(d_lon, d_lat, d_cieX, d_cieY, d_cieZ, nRecords,
      minLat, maxLat, q95,
      thrust::raw_pointer_cast(dv_xyz.data()), dimx, dimy);

  cudaFree(d_cieX);
  cudaFree(d_cieY);
  cudaFree(d_cieZ);
  cudaFree(d_lat);
  cudaFree(d_lon);

  //uchar3 black; black.x = 0; black.y = 0; black.z = 0;
  //uchar3 white; white.x = 255; white.y = 255; white.z = 255;
  thrust::device_vector<uchar3> dv_bgr(nPixels);
  CieXyzToSrgb(thrust::raw_pointer_cast(dv_xyz.data()),
      thrust::raw_pointer_cast(dv_bgr.data()), nPixels);
  dv_xyz.clear(); // beware: async?

  cv::Mat image;
  image.create(dimy, dimx, CV_8UC3);
  //checkCudaErrors(cudaMemcpy((uchar3*)image.ptr<unsigned char>(0),
  //    d_bgr, nPixels*sizeof(uchar3), cudaMemcpyDeviceToHost));
  thrust::copy(dv_bgr.begin(), dv_bgr.end(),
      (uchar3*)image.ptr<unsigned char>(0));
  dv_bgr.clear();
  //cudaFree(d_bgr);
  cv::imwrite(output_file, image);

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  return EXIT_SUCCESS;
}

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

std::string patchName(int patchNum);

int main(int argc, char** argv) {
  // Process arguments
  if (argc != 4) {
    fprintf(stderr, "Usage: skyrender <input_dir> <output_file> <Y_max>\n");
    return EXIT_FAILURE;
  }
  const std::string input_dir = std::string(argv[1]);
  const std::string output_file = std::string(argv[2]);
  float yMax;
  { const int ret = sscanf(argv[3], "%f", &yMax);
    assert(ret == 1); }
  // Flower: yMax / 10^(-0.4*2.29)
  // FlowerFit: yMax / 10^(-0.4*1.1)

  const int dimx = 3200;
  const int dimy = 1600;
  const int nPixels = dimx*dimy;
  float3 fblack; fblack.x = 0.0f; fblack.y = 0.0f; fblack.z = 0.0f;
  thrust::device_vector<float3> dv_xyz(nPixels, fblack);

  const float minLat = -90.0f;
  const float maxLat = 90.0f;

  const int nPatches = 92;
  for (int patchNum=0; patchNum < nPatches; ++patchNum) {
    const std::string patchDir = patchName(patchNum);

    // Read input data
    int nRecords;
    float* d_lon;
    float* d_lat;
    float* d_cieX;
    float* d_cieY;
    float* d_cieZ;
    {
      struct stat fileStats;
      std::string lonFilename = input_dir + "/" + patchDir + "/tmfLon.dat";
      const int ret = stat(lonFilename.c_str(), &fileStats);
      if (ret != 0) {
        printf("Skipping patch %s\n", patchDir.c_str());
        continue;
      } else printf("Processing patch %s\n", patchDir.c_str());
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
      { FILE* in = fopen((input_dir+"/"+patchDir+"/tmfLat.dat").c_str(), "r");
        const int nRead = fread(h_buf, sizeof(float), nRecords, in);
        assert(nRead == nRecords);
        fclose(in);
        checkCudaErrors(cudaMalloc(&d_lat, nRecords*sizeof(float)));
        checkCudaErrors(cudaMemcpy(d_lat, h_buf, nRecords*sizeof(float),
            cudaMemcpyHostToDevice)); }
      { FILE* in = fopen((input_dir+"/"+patchDir+"/tmfCieX.dat").c_str(), "r");
        const int nRead = fread(h_buf, sizeof(float), nRecords, in);
        assert(nRead == nRecords);
        fclose(in);
        checkCudaErrors(cudaMalloc(&d_cieX, nRecords*sizeof(float)));
        checkCudaErrors(cudaMemcpy(d_cieX, h_buf, nRecords*sizeof(float),
            cudaMemcpyHostToDevice)); }
      { FILE* in = fopen((input_dir+"/"+patchDir+"/tmfCieY.dat").c_str(), "r");
        const int nRead = fread(h_buf, sizeof(float), nRecords, in);
        assert(nRead == nRecords);
        fclose(in);
        checkCudaErrors(cudaMalloc(&d_cieY, nRecords*sizeof(float)));
        checkCudaErrors(cudaMemcpy(d_cieY, h_buf, nRecords*sizeof(float),
            cudaMemcpyHostToDevice)); }
      { FILE* in = fopen((input_dir+"/"+patchDir+"/tmfCieZ.dat").c_str(), "r");
        const int nRead = fread(h_buf, sizeof(float), nRecords, in);
        assert(nRead == nRecords);
        fclose(in);
        checkCudaErrors(cudaMalloc(&d_cieZ, nRecords*sizeof(float)));
        checkCudaErrors(cudaMemcpy(d_cieZ, h_buf, nRecords*sizeof(float),
            cudaMemcpyHostToDevice)); }

      delete[] h_buf;
    }

    RenderStars(d_lon, d_lat, d_cieX, d_cieY, d_cieZ, nRecords,
        minLat, maxLat, yMax,
        thrust::raw_pointer_cast(dv_xyz.data()), dimx, dimy);

    cudaFree(d_cieX);
    cudaFree(d_cieY);
    cudaFree(d_cieZ);
    cudaFree(d_lat);
    cudaFree(d_lon);
  }

  // Create sRGB image (BGR order)
  thrust::device_vector<uchar3> dv_bgr(nPixels);
  CieXyzToSrgb(thrust::raw_pointer_cast(dv_xyz.data()),
      thrust::raw_pointer_cast(dv_bgr.data()), nPixels);
  dv_xyz.clear(); // beware: async?

  // Write image
  cv::Mat image;
  image.create(dimy, dimx, CV_8UC3);
  thrust::copy(dv_bgr.begin(), dv_bgr.end(),
      (uchar3*)image.ptr<unsigned char>(0));
  dv_bgr.clear();
  cv::imwrite(output_file, image);

  // Shutdown
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  return EXIT_SUCCESS;
}

std::string patchName(const int patchNum) {
  std::string c1((patchNum < 57) ? "a" : "b");
  if (patchNum < 57) {
    const int c3 = patchNum % 26;
    const int c2 = patchNum / 26;
    return std::string("a") + static_cast<char>('a' + c2) +
        static_cast<char>('a' + c3);
  } else {
    const int c3 = (patchNum - 57) % 26;
    const int c2 = (patchNum - 57) / 26;
    return std::string("b") + static_cast<char>('a' + c2) +
        static_cast<char>('a' + c3);
  }
}

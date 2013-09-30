#include "kernels.h"

#include <math_functions.h>

__device__
float2 cie_xy_from_temp(const float inTemp) {
  const float temp = min(max(inTemp, 1667.0f), 25000.0f);
  const float invT = 1.0f / temp;
  const float invT2 = invT * invT;
  const float invT3 = invT2 * invT;
  const float x = (temp < 4000.0f) ?
      (-0.2661239e9f*invT3 - 0.2343580e6f*invT2 +
       0.8776956e3f*invT + 0.179910f) :
      (-3.0258469e9f*invT3 + 2.1070379e6f*invT2 +
       0.2226347e3f*invT + 0.240390f);
  const float x2 = x * x;
  const float x3 = x2 * x;
  const float y = (temp < 2222.0f) ?
      (-1.1063814f*x3 - 1.34811020f*x2 + 2.18555832f*x - 0.20219683f) :
      ((temp < 4000.0f) ?
       (-0.9549476f*x3 - 1.37418593f*x2 + 2.09137015f*x - 0.16748867f) :
       ( 3.0817580f*x3 - 5.87338670f*x2 + 3.75112997f*x - 0.37001483f));

  float2 ans;
  ans.x = x;
  ans.y = y;
  return ans;
}

__device__
float3 cie_xyz_from_tempamp(const float2 tempamp) {
  const float2 xy = cie_xy_from_temp(tempamp.x);
  // TODO: Remove hard-coded constant
  const float maxY = 8.69284e-25f;      // PointFit, 2k<T<9k, 95%, DR 414
  //const float maxY = 4.909988e-24f;     // FlowerJmag, 2k<T<9k, 95%, DR 57
  //const float maxY = 2.676281e-23f;     // FlowerJmag, 2k<T<9k, 98.8%, DR 396
  //const float maxY = 1.62476e-24f;      // FlowerAmp, 2k<T<9k, 95%, DR 93
  //const float maxY = 4.732184e-24f;     // FlowerAmp, 2k<T<9k, 98%, DR 387
  const float myAmp = tempamp.y / maxY;
  //const float myAmp = 0.5;

  float3 xyz;
  xyz.y = min(myAmp, 1.0f);
  xyz.x = xyz.y * xy.x / xy.y;
  xyz.z = xyz.y * (1.0f - xy.x - xy.y) / xy.y;
  return xyz;
}

__device__
float srgb_gamma(const float cLin) {
  const float a = 0.055f;
  return (cLin < 0.0031308f) ? (12.92f*cLin) :
      ((1.0f + a)*powf(cLin, 1.0f/2.4f) - a);
}

__device__
uchar3 srgb_from_cie_xyz(const float3 cie_xyz) {
  const float rLin =  3.2406f*cie_xyz.x - 1.5372f*cie_xyz.y - 0.4986f*cie_xyz.z;
  const float gLin = -0.9689f*cie_xyz.x + 1.8758f*cie_xyz.y + 0.0415f*cie_xyz.z;
  const float bLin =  0.0557f*cie_xyz.x - 0.2040f*cie_xyz.y + 1.0570f*cie_xyz.z;

  uchar3 gbr;
  gbr.z = static_cast<unsigned char>(min(255.0f * srgb_gamma(rLin), 255.0f));
  gbr.y = static_cast<unsigned char>(min(255.0f * srgb_gamma(gLin), 255.0f));
  gbr.x = static_cast<unsigned char>(min(255.0f * srgb_gamma(bLin), 255.0f));
  return gbr;
}

__device__
uchar3 srgb_from_cie_xy(const float2 cie_xy) {
  float3 xyz;
  xyz.y = 0.5f;
  xyz.x = xyz.y * cie_xy.x / cie_xy.y;
  xyz.z = xyz.y * (1.0f - cie_xy.x - cie_xy.y) / cie_xy.y;

  return srgb_from_cie_xyz(xyz);
}

__global__
void color_array(const float2* tempAmp, uchar3* image, int size) {
  const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  if (x >= size) return;
  image[x] = srgb_from_cie_xyz(cie_xyz_from_tempamp(tempAmp[x]));
}

__global__
void render_stars(const float* d_lon, const float* d_lat,
    const float* d_cieX, const float* d_cieY, const float* d_cieZ,
    const int nStars,
    const float minLat, const float maxLat, const float yNorm,
    float3* image, const int dimx, const int dimy) {
  const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i >= nStars) return;

  // Get coordinates
  const float lon = d_lon[i];
  const float lat = d_lat[i];
  const float dy = maxLat - minLat;
  const float pxf = dimx*(lon/360.0f);
  const float pyf = dimy*((lat - minLat)/dy);
  const int px = static_cast<int>(pxf + 0.5);
  const int py = static_cast<int>(pyf + 0.5);
  //const int px = min(max(static_cast<int>(dimx*(lon/360.0f) + 0.5f), 0),
  //    dimx - 1);
  //const int py = min(max(static_cast<int>(dimy*((lat - minLat)/dy) + 0.5f), 0),
  //    dimy - 1);

  // Get color
  float3 xyz;
  //xyz.x = min(d_cieX[i] / yNorm, 1.0f);
  //xyz.y = min(d_cieY[i] / yNorm, 1.0f);
  //xyz.z = min(d_cieZ[i] / yNorm, 1.0f);
  xyz.x = d_cieX[i] / yNorm;
  xyz.y = d_cieY[i] / yNorm;
  xyz.z = d_cieZ[i] / yNorm;

  // Increment global memory
  //atomicAdd(&(image[py*dimx + px].x), xyz.x);
  //atomicAdd(&(image[py*dimx + px].y), xyz.y);
  //atomicAdd(&(image[py*dimx + px].z), xyz.z);
  const int win = 7;
  for (int i=-win; i<=win; ++i) {
    const int ppx = px + i;
    for (int j=-win; j<=win; ++j) {
      const int ppy = py + j;
      const float r = hypotf(ppx - pxf, ppy - pyf);
      //const float a = max(1.0f - 0.5f*r, 0.0f);
      const float bInv = 36.0;
      const float a = expf(-sqrtf(r*bInv));
      if ((ppx >= 0) && (ppx < dimx) && (ppy >= 0) && (ppy < dimy)) {
        atomicAdd(&(image[ppy*dimx + ppx].x), a*xyz.x);
        atomicAdd(&(image[ppy*dimx + ppx].y), a*xyz.y);
        atomicAdd(&(image[ppy*dimx + ppx].z), a*xyz.z);
      }
    }
  }
}

__global__
void map_cie_xyz_to_srgb(const float3* xyz, uchar3* bgr, const int size) {
  const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i >= size) return;
  bgr[i] = srgb_from_cie_xyz(xyz[i]);
}

void ColorArray(const float2* tempAmp, uchar3* image, const int size) {
  const dim3 blockSize(256);
  const unsigned int gridx = (size - 1)/blockSize.x + 1;
  const dim3 gridSize(gridx);
  color_array<<<gridSize, blockSize>>>(tempAmp, image, size);
}

void RenderStars(const float* d_lon, const float* d_lat,
    const float* d_cieX, const float* d_cieY, const float* d_cieZ,
    const int nStars,
    const float minLat, const float maxLat, const float yNorm,
    float3* image, const int dimx, const int dimy) {
  const dim3 blockSize(256);
  const unsigned int gridx = (nStars - 1)/blockSize.x + 1;
  const dim3 gridSize(gridx);
  render_stars<<<gridSize, blockSize>>>(d_lon, d_lat, d_cieX, d_cieY, d_cieZ,
      nStars, minLat, maxLat, yNorm, image, dimx, dimy);
}

void CieXyzToSrgb(const float3* xyz, uchar3* bgr, const int size) {
  const dim3 blockSize(256);
  const unsigned int gridx = (size - 1)/blockSize.x + 1;
  const dim3 gridSize(gridx);
  map_cie_xyz_to_srgb<<<gridSize, blockSize>>>(xyz, bgr, size);
}

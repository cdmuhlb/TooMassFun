#ifndef COLORARRAY_KERNELS_H_
#define COLORARRAY_KERNELS_H_

#include <vector_types.h>

void ColorArray(const float2* tempAmp, uchar3* image, int size);
void RenderStars(const float* d_lon, const float* d_lat,
    const float* d_cieX, const float* d_cieY, const float* d_cieZ,
    const int nStars,
    const float minLat, const float maxLat, const float yNorm,
    float3* image, const int dimx, const int dimy);
void CieXyzToSrgb(const float3* xyz, uchar3* bgr, const int size);

#endif  // COLORARRAY_KERNELS_H_

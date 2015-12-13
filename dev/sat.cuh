#include "common.cuh"

__device__ void scan(uint32_t* share);

__device__ void scan2d(uint32_t* image);

void scan2dSerial(uint32_t* image, int dim);
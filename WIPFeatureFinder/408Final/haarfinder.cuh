#include "common.cuh"


__device__ void haarfinder(uint32_t* sat, uint64_t* features, int32_t threshold);

__device__ uint64_t packFeature(uint8_t x1, uint8_t y1, uint8_t x2, uint8_t y2, uint8_t type, uint32_t mag);
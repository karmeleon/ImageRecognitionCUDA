#include "common.cuh"

__device__ void haarfinder(uint32_t* sat, feature* features, int32_t threshold, uint32_t* featureIndex);

__device__ feature packFeature(uint8_t x1, uint8_t y1, uint8_t x2, uint8_t y2, uint8_t type, int32_t mag);
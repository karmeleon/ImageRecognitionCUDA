#include "common.cuh"
#include "sat.cuh"
#include "haarfinder.cuh"

void printFeature(feature feat);

//uint64_t packFeature(feature unpacked);
//inline feature unpackFeature(uint64_t packed);

feature* findFeatures(uint32_t* images, uint32_t count, uint32_t* numFeatures);
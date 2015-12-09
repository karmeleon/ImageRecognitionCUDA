#include "common.cuh"
#include "sat.cuh"
#include "haarfinder.cuh"

void printFeature(feature feat);

feature* findFeatures(uint32_t* images, uint32_t count, uint32_t* numFeatures);
#include "common.cuh"
#include "sat.cuh"
#include "haarfinder.cuh"

typedef struct _feature {
	uint8_t x1, y1, x2, y2, type;
	int32_t mag;
} feature;

void printFeature(feature feat);

uint64_t packFeature(feature unpacked);
inline feature unpackFeature(uint64_t packed);

feature* findFeatures(uint32_t* images, uint32_t count, uint32_t* numFeatures);
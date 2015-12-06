#include "haarfinder.cuh"

#define GRANULARITY 1

// packs a feature into a 64-bit datatype. if any of the coords are greater than 127, you're going to have a bad time
__device__ uint64_t packFeature(uint8_t x1, uint8_t y1, uint8_t x2, uint8_t y2, uint8_t type, int32_t mag) {
	// 7 bits each for x1, y1, x2, y2
	uint64_t out = 0;
	// bitwise operations are so satisfying
	out |= ((uint64_t)x1) << 57;
	out |= ((uint64_t)y1) << 50;
	out |= ((uint64_t)x2) << 43;
	out |= ((uint64_t)y2) << 36;
	out |= ((uint64_t)type) << 33;
	((int32_t*)&out)[0] = ((int32_t)mag);
	return out;
}

__device__ uint32_t regionMag(uint32_t* sat, uint8_t x1, uint8_t y1, uint8_t x2, uint8_t y2) {
	//return sat[x2 + IMAGE_SIZE * y2] + sat[x1 + IMAGE_SIZE * y1] - sat[x2 + IMAGE_SIZE * y1] - sat[x1 + IMAGE_SIZE * y2];
	uint32_t A, B, C, D;
	// oh man these branches hurt me :(
	if (x1 == 0 || y1 == 0)
		A = 0;
	else
		A = sat[(x1 - 1) + IMAGE_SIZE * (y1 - 1)];

	if (y1 == 0)
		B = 0;
	else
		B = sat[x2 + IMAGE_SIZE * (y1 - 1)];

	if (x1 == 0)
		C = 0;
	else
		C = sat[(x1 - 1) + IMAGE_SIZE * y2];

	D = sat[x2 + IMAGE_SIZE * y2];
	return D + A - B - C;
}

__device__ void saveFeature(uint32_t* featureIndex, uint64_t* features, uint64_t feature) {
	if (feature != 0) {
		uint32_t featIdx = atomicAdd(featureIndex, 1);
		features[featIdx] = feature;
	}
}

__device__ uint64_t evalHorizEdge(uint32_t* sat, uint8_t x1, uint8_t y1, uint8_t x2, uint8_t y2, int32_t threshold) {
	uint8_t y3 = (y1 + y2) / 2;

	uint32_t topWeight = y3 - (y1 - 1);
	uint32_t bottomWeight = y2 - (y3 - 1);

	uint32_t top = regionMag(sat, x1, y1, x2, y3) * bottomWeight / (topWeight + bottomWeight);
	uint32_t bottom = regionMag(sat, x1, y3, x2, y2) * topWeight / (topWeight + bottomWeight);
	int32_t mag = bottom - top;
	if (abs(mag) > threshold * (x2 - x1) * (y2 - y1))
		return packFeature(x1, y1, x2, y2, HEDGE, mag);
	else
		return 0;
}

__device__ uint64_t evalVertEdge(uint32_t* sat, uint8_t x1, uint8_t y1, uint8_t x2, uint8_t y2, int32_t threshold) {
	uint8_t x3 = (x1 + x2) / 2;

	uint32_t leftWeight = x3 - (x1 - 1);
	uint32_t rightWeight = x2 - (x3 - 1);

	uint32_t left = regionMag(sat, x1, y1, x3, y2) * rightWeight / (leftWeight + rightWeight);
	uint32_t right = regionMag(sat, x3, y1, x2, y2) * leftWeight / (leftWeight + rightWeight);
	int32_t mag = right - left;
	if (abs(mag) > threshold * (x2 - x1) * (y2 - y1))
		return packFeature(x1, y1, x2, y2, VEDGE, mag);
	else
		return 0;
}

__device__ uint64_t evalHorizLine(uint32_t* sat, uint8_t x1, uint8_t y1, uint8_t x2, uint8_t y2, int32_t threshold) {
	uint8_t y3 = (3 * y1 + y2) / 4;
	uint8_t y4 = (y1 + 3 * y2) / 4;

	uint32_t topWeight = y3 - (y1 - 1);
	uint32_t centerWeight = y4 - (y3 - 1);
	uint32_t bottomWeight = y2 - (y4 - 1);
	// x1 == 0  && y1 == 0 && x2 == 4 && y2 == 4
	uint32_t top = regionMag(sat, x1, y1, x2, y3) * (centerWeight + bottomWeight) / (topWeight + centerWeight + bottomWeight);
	uint32_t center = regionMag(sat, x1, y3, x2, y4) * (topWeight + bottomWeight) / (topWeight + centerWeight + bottomWeight);
	uint32_t bottom = regionMag(sat, x1, y4, x2, y2) * (centerWeight + topWeight) / (topWeight + centerWeight + bottomWeight);
	int32_t mag = center - (top + bottom);
	if (abs(mag) > threshold * (x2 - x1) * (y2 - y1))
		return packFeature(x1, y1, x2, y2, HLINE, mag);
	else
		return 0;
}

__device__ uint64_t evalVertLine(uint32_t* sat, uint8_t x1, uint8_t y1, uint8_t x2, uint8_t y2, int32_t threshold) {
	uint8_t x3 = (3 * x1 + x2) / 4;
	uint8_t x4 = (x1 + 3 * x2) / 4;

	uint32_t leftWeight = x3 - (x1 - 1);
	uint32_t centerWeight = x4 - (x3 - 1);
	uint32_t rightWeight = x2 - (x4 - 1);

	uint32_t left = regionMag(sat, x1, y1, x3, y2) * (centerWeight + rightWeight) / (leftWeight + centerWeight + rightWeight);
	uint32_t center = regionMag(sat, x3, y1, x4, y2) * (leftWeight + rightWeight) / (leftWeight + centerWeight + rightWeight);
	uint32_t right = regionMag(sat, x4, y1, x2, y2) * (rightWeight + rightWeight) / (leftWeight + centerWeight + rightWeight);
	int32_t mag = center - (left + right);
	if (abs(mag) > threshold * (x2 - x1) * (y2 - y1))
		return packFeature(x1, y1, x2, y2, VLINE, mag);
	else
		return 0;
}

__device__ uint64_t evalFourRectangle(uint32_t* sat, uint8_t x1, uint8_t y1, uint8_t x2, uint8_t y2, int32_t threshold) {
	uint8_t x3 = (x1 + x2) / 2;
	uint8_t y3 = (y1 + y2) / 2;
	uint32_t topLeft = regionMag(sat, x1, y1, x3, y3);
	uint32_t topRight = regionMag(sat, x3, y1, x2, y3);
	uint32_t bottomLeft = regionMag(sat, x1, y3, x3, y2);
	uint32_t bottomRight = regionMag(sat, x3, y3, x2, y2);
	int32_t mag = (bottomLeft + topRight) - (topLeft + bottomRight);
	if (abs(mag) > threshold * (x2 - x1) * (y2 - y1))
		return packFeature(x1, y1, x2, y2, RECT4, mag);
	else
		return 0;
}

// finds haar-like features on the given SAT image. for best results, make sure sat is in shared memory.
__device__ void haarfinder(uint32_t* sat, uint64_t* features, int32_t threshold) {
	// screw writing "unsigned long long int", use stdint.h instead
	// we assume here that features is large enough to hold all the features we'll find and is in global memory

	// the counter to keep track of where in the feature buffer we are
	// ONLY ACCESS THIS ATOMICALLY
	__shared__ uint32_t featureIndex[1];
	featureIndex[0] = 0;

	const uint32_t stride = blockDim.x;

	for (uint32_t xSize = 4; xSize < IMAGE_SIZE; xSize++) {
		for (uint32_t ySize = 4; ySize < IMAGE_SIZE; ySize++) {

			// the total number of regions to process
			uint32_t xRegions = IMAGE_SIZE - xSize;
			uint32_t yRegions = IMAGE_SIZE - ySize;
			uint32_t numRegions = xRegions * yRegions;
			// number of regions each thread processes
			uint32_t numIterations = numRegions / stride;

			for (uint32_t i = 0; i < numIterations; i++) {
				if (i <= numIterations) {
					uint32_t idx = i * stride + threadIdx.x;
					uint8_t x1 = idx % xRegions;
					uint8_t y1 = idx / xRegions;
					uint8_t x2 = x1 + xSize;
					uint8_t y2 = y1 + ySize;

					// evaluate Haar-like features
					uint64_t horizEdge = evalHorizEdge(sat, x1, y1, x2, y2, threshold);
					saveFeature(&featureIndex[0], features, horizEdge);

					uint64_t vertEdge = evalVertEdge(sat, x1, y1, x2, y2, threshold);
					saveFeature(&featureIndex[0], features, vertEdge);

					uint64_t horizLine = evalHorizLine(sat, x1, y1, x2, y2, threshold);
					saveFeature(&featureIndex[0], features, horizLine);

					uint64_t vertLine = evalVertLine(sat, x1, y1, x2, y2, threshold);
					saveFeature(&featureIndex[0], features, vertLine);

					uint64_t fourRectangle = evalFourRectangle(sat, x1, y1, x2, y2, threshold);
					saveFeature(&featureIndex[0], features, fourRectangle);
				}
			}
		}
	}
}
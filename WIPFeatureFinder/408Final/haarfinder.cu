#include "haarfinder.cuh"

#define GRANULARITY 1

// packs a feature into a 64-bit datatype. if any of the coords are greater than 127, you're going to have a bad time
__device__ feature packFeature(uint8_t x1, uint8_t y1, uint8_t x2, uint8_t y2, uint8_t type, int32_t mag) {
	//// 7 bits each for x1, y1, x2, y2
	//uint64_t out = 0;
	//// bitwise operations are so satisfying
	//out |= ((uint64_t)x1) << 57;
	//out |= ((uint64_t)y1) << 50;
	//out |= ((uint64_t)x2) << 43;
	//out |= ((uint64_t)y2) << 36;
	//out |= ((uint64_t)type) << 33;
	//((int32_t*)&out)[0] = ((int32_t)mag);
	//return out;
	feature f;
	f.x1 = x1;
	f.y1 = y1;
	f.x2 = x2;
	f.y2 = y2;
	f.mag = mag;
	f.type = type;
	return f;
}

__device__ uint32_t regionMag(uint32_t* sat, uint8_t x1, uint8_t y1, uint8_t x2, uint8_t y2) {
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

__device__ feature noFeature() {
	feature f;
	f.mag = 0;
	return f;
}

__device__ void saveFeature(uint32_t* featureIndex, feature* features, feature feature) {
	if (feature.mag != 0) {
		uint32_t featIdx = atomicAdd(featureIndex, 1);
		features[featIdx] = feature;
	}
}

__device__ feature evalHorizEdge(uint32_t* sat, uint8_t x1, uint8_t y1, uint8_t x2, uint8_t y2, int32_t threshold) {
	uint8_t y3 = (y1 + y2) / 2;
	
	uint32_t whiteWeight = y3 - (y1 - 1);
	uint32_t blackWeight = y2 - y3;

	uint32_t white = regionMag(sat, x1, y1, x2, y3);
	uint32_t black = regionMag(sat, x1, y3 + 1, x2, y2);

	float diffWhite = (float)white / whiteWeight;
	float diffBlack = (float)black / blackWeight;

	float diffMag = diffWhite - diffBlack;
	if (abs(diffMag) > threshold * (x2 - x1))
		return packFeature(x1, y1, x2, y2, HEDGE, (int32_t)white - (int32_t)black);
	else
		return noFeature();
}

__device__ feature evalVertEdge(uint32_t* sat, uint8_t x1, uint8_t y1, uint8_t x2, uint8_t y2, int32_t threshold) {
	uint8_t x3 = (x1 + x2) / 2;

	uint32_t whiteWeight = x3 - (x1 - 1);
	uint32_t blackWeight = x2 - x3;

	uint32_t white = regionMag(sat, x1, y1, x3, y2);
	uint32_t black = regionMag(sat, x3 + 1, y1, x2, y2);

	float diffWhite = (float)white / whiteWeight;
	float diffBlack = (float)black / blackWeight;

	float diffMag = diffWhite - diffBlack;
	if (abs(diffMag) > threshold * (y2 - y1))
		return packFeature(x1, y1, x2, y2, VEDGE, (int32_t)white - (int32_t)black);
	else
		return noFeature();
}

__device__ feature evalHorizLine(uint32_t* sat, uint8_t x1, uint8_t y1, uint8_t x2, uint8_t y2, int32_t threshold) {
	uint8_t y3 = (3 * y1 + y2) / 4;
	uint8_t y4 = (y1 + 3 * y2) / 4;

	uint32_t whiteWeight = y3 - y1 + y2 - (y4 - 1);
	uint32_t blackWeight = y4 - y3;

	uint32_t white = 0;
	uint32_t black = 0;

	// top
	white += regionMag(sat, x1, y1, x2, y3);
	// bottom
	white += regionMag(sat, x1, y4 + 1, x2, y2);

	// middle
	black += regionMag(sat, x1, y3 + 1, x2, y4);

	float diffWhite = (float)white / whiteWeight;
	float diffBlack = (float)black / blackWeight;

	float diffMag = diffWhite - diffBlack;
	if (abs(diffMag) > threshold * (x2 - x1))
		return packFeature(x1, y1, x2, y2, HLINE, (int32_t)white - (int32_t)black);
	else
		return noFeature();
}

__device__ feature evalVertLine(uint32_t* sat, uint8_t x1, uint8_t y1, uint8_t x2, uint8_t y2, int32_t threshold) {
	uint8_t x3 = (3 * x1 + x2) / 4;
	uint8_t x4 = (x1 + 3 * x2) / 4;
	
	uint32_t whiteWeight = x3 - x1 + x2 - (x4 - 1);
	uint32_t blackWeight = x4 - x3;

	uint32_t white = 0;
	uint32_t black = 0;

	// left
	white += regionMag(sat, x1, y1, x3, y2);
	// right
	white += regionMag(sat, x4 + 1, y1, x2, y2);

	// center
	black += regionMag(sat, x3 + 1, y1, x4, y2);
	
	float diffWhite = (float)white / whiteWeight;
	float diffBlack = (float)black / blackWeight;

	float diffMag = diffWhite - diffBlack;
	if (abs(diffMag) > threshold * (y2 - y1))
		return packFeature(x1, y1, x2, y2, VLINE, (int32_t)white - (int32_t)black);
	else
		return noFeature();
}

__device__ feature evalFourRectangle(uint32_t* sat, uint8_t x1, uint8_t y1, uint8_t x2, uint8_t y2, int32_t threshold) {
	uint8_t x3 = (x1 + x2) / 2;
	uint8_t y3 = (y1 + y2) / 2;

	uint32_t topWeight = y3 - (y1 - 1);
	uint32_t bottomWeight = y2 - y3;
	uint32_t leftWeight = x3 - (x1 - 1);
	uint32_t rightWeight = x2 - x3;

	uint32_t whiteWeight = topWeight * leftWeight + bottomWeight * rightWeight;
	uint32_t blackWeight = topWeight * rightWeight + bottomWeight * leftWeight;

	uint32_t white = 0;
	uint32_t black = 0;

	// top left
	white += regionMag(sat, x1, y1, x3, y3);
	// top right
	black += regionMag(sat, x3 + 1, y1, x2, y3);
	// bottom left
	black += regionMag(sat, x1, y3 + 1, x3, y2);
	// bottom right
	white += regionMag(sat, x3 + 1, y3 + 1, x2, y2);

	float diffWhite = (float)white / whiteWeight;
	float diffBlack = (float)black / blackWeight;

	float diffMag = diffWhite - diffBlack;
	if (abs(diffMag) > threshold)
		return packFeature(x1, y1, x2, y2, RECT4, (int32_t)white - (int32_t)black);
	else
		return noFeature();
}

// finds haar-like features on the given SAT image. for best results, make sure sat is in shared memory.
__device__ void haarfinder(uint32_t* sat, feature* features, int32_t threshold) {
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
				uint32_t idx = i * stride + threadIdx.x;
				if (idx < numRegions) {
					uint8_t x1 = idx % xRegions;
					uint8_t y1 = idx / xRegions;
					uint8_t x2 = x1 + xSize;
					uint8_t y2 = y1 + ySize;

					// evaluate Haar-like features
					feature horizEdge = evalHorizEdge(sat, x1, y1, x2, y2, threshold);
					saveFeature(&featureIndex[0], features, horizEdge);

					feature vertEdge = evalVertEdge(sat, x1, y1, x2, y2, threshold);
					saveFeature(&featureIndex[0], features, vertEdge);

					feature horizLine = evalHorizLine(sat, x1, y1, x2, y2, threshold);
					saveFeature(&featureIndex[0], features, horizLine);

					feature vertLine = evalVertLine(sat, x1, y1, x2, y2, threshold);
					saveFeature(&featureIndex[0], features, vertLine);

					feature fourRectangle = evalFourRectangle(sat, x1, y1, x2, y2, threshold);
					saveFeature(&featureIndex[0], features, fourRectangle);
				}
			}
		}
	}
}
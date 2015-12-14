#include "serial_features.cuh"

#define SERIAL_FEATURES_PER_IMAGE (16000 * 5)

feature s_packFeature(uint8_t x1, uint8_t y1, uint8_t x2, uint8_t y2, uint8_t type, int32_t mag) {
	feature f;
	f.x1 = x1;
	f.y1 = y1;
	f.x2 = x2;
	f.y2 = y2;
	f.mag = mag;
	f.type = type;
	return f;
}

uint32_t s_regionMag(uint32_t* sat, uint8_t x1, uint8_t y1, uint8_t x2, uint8_t y2) {
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

feature s_noFeature() {
	feature f;
	f.mag = 0;
	return f;
}

feature s_evalHorizEdge(uint32_t* sat, uint8_t x1, uint8_t y1, uint8_t x2, uint8_t y2, int32_t threshold) {
	uint8_t y3 = (y1 + y2) / 2;

	const uint32_t whiteWeight = y3 - (y1 - 1);
	const uint32_t blackWeight = y2 - y3;

	const uint32_t white = s_regionMag(sat, x1, y1, x2, y3);
	const uint32_t black = s_regionMag(sat, x1, y3 + 1, x2, y2);

	float diffWhite = (float)white / whiteWeight;
	float diffBlack = (float)black / blackWeight;

	float diffMag = diffWhite - diffBlack;
	if (abs(diffMag) > threshold * (x2 - x1))
		return s_packFeature(x1, y1, x2, y2, HEDGE, (int32_t)white - (int32_t)black);
	else
		return s_noFeature();
}

feature s_evalVertEdge(uint32_t* sat, uint8_t x1, uint8_t y1, uint8_t x2, uint8_t y2, int32_t threshold) {
	uint8_t x3 = (x1 + x2) / 2;

	const uint32_t whiteWeight = x3 - (x1 - 1);
	const uint32_t blackWeight = x2 - x3;

	const uint32_t white = s_regionMag(sat, x1, y1, x3, y2);
	const uint32_t black = s_regionMag(sat, x3 + 1, y1, x2, y2);

	float diffWhite = (float)white / whiteWeight;
	float diffBlack = (float)black / blackWeight;

	float diffMag = diffWhite - diffBlack;
	if (abs(diffMag) > threshold * (y2 - y1))
		return s_packFeature(x1, y1, x2, y2, VEDGE, (int32_t)white - (int32_t)black);
	else
		return s_noFeature();
}

feature s_evalHorizLine(uint32_t* sat, uint8_t x1, uint8_t y1, uint8_t x2, uint8_t y2, int32_t threshold) {
	uint8_t y3 = (3 * y1 + y2) / 4;
	uint8_t y4 = (y1 + 3 * y2) / 4;

	const uint32_t whiteWeight = y3 - y1 + y2 - (y4 - 1);
	const uint32_t blackWeight = y4 - y3;

	uint32_t white = 0;
	uint32_t black = 0;

	// top
	white += s_regionMag(sat, x1, y1, x2, y3);
	// bottom
	white += s_regionMag(sat, x1, y4 + 1, x2, y2);

	// middle
	black += s_regionMag(sat, x1, y3 + 1, x2, y4);

	float diffWhite = (float)white / whiteWeight;
	float diffBlack = (float)black / blackWeight;

	float diffMag = diffWhite - diffBlack;
	if (abs(diffMag) > threshold * (x2 - x1))
		return s_packFeature(x1, y1, x2, y2, HLINE, (int32_t)white - (int32_t)black);
	else
		return s_noFeature();
}

feature s_evalVertLine(uint32_t* sat, uint8_t x1, uint8_t y1, uint8_t x2, uint8_t y2, int32_t threshold) {
	uint8_t x3 = (3 * x1 + x2) / 4;
	uint8_t x4 = (x1 + 3 * x2) / 4;

	const uint32_t whiteWeight = x3 - x1 + x2 - (x4 - 1);
	const uint32_t blackWeight = x4 - x3;

	uint32_t white = 0;
	uint32_t black = 0;

	// left
	white += s_regionMag(sat, x1, y1, x3, y2);
	// right
	white += s_regionMag(sat, x4 + 1, y1, x2, y2);

	// center
	black += s_regionMag(sat, x3 + 1, y1, x4, y2);

	float diffWhite = (float)white / whiteWeight;
	float diffBlack = (float)black / blackWeight;

	float diffMag = diffWhite - diffBlack;
	if (abs(diffMag) > threshold * (y2 - y1))
		return s_packFeature(x1, y1, x2, y2, VLINE, (int32_t)white - (int32_t)black);
	else
		return s_noFeature();
}

feature s_evalFourRectangle(uint32_t* sat, uint8_t x1, uint8_t y1, uint8_t x2, uint8_t y2, int32_t threshold) {
	uint8_t x3 = (x1 + x2) / 2;
	uint8_t y3 = (y1 + y2) / 2;

	const uint32_t topWeight = y3 - (y1 - 1);
	const uint32_t bottomWeight = y2 - y3;
	const uint32_t leftWeight = x3 - (x1 - 1);
	const uint32_t rightWeight = x2 - x3;

	const uint32_t whiteWeight = topWeight * leftWeight + bottomWeight * rightWeight;
	const uint32_t blackWeight = topWeight * rightWeight + bottomWeight * leftWeight;

	uint32_t white = 0;
	uint32_t black = 0;

	// top left
	white += s_regionMag(sat, x1, y1, x3, y3);
	// top right
	black += s_regionMag(sat, x3 + 1, y1, x2, y3);
	// bottom left
	black += s_regionMag(sat, x1, y3 + 1, x3, y2);
	// bottom right
	white += s_regionMag(sat, x3 + 1, y3 + 1, x2, y2);

	float diffWhite = (float)white / whiteWeight;
	float diffBlack = (float)black / blackWeight;

	float diffMag = diffWhite - diffBlack;
	if (abs(diffMag) > threshold)
		return s_packFeature(x1, y1, x2, y2, RECT4, (int32_t)white - (int32_t)black);
	else
		return s_noFeature();
}

feature* serialFindFeatures(uint32_t* hostImageBuffer, uint32_t count, uint32_t* numFeatures) {
	feature* outFeatures = (feature*)malloc(SERIAL_FEATURES_PER_IMAGE * count * sizeof(feature));
	feature** threadFeatureBuffers = (feature**)malloc(std::thread::hardware_concurrency() * sizeof(feature*));
	for (int i = 0; i < std::thread::hardware_concurrency(); i++)
		threadFeatureBuffers[i] = (feature*)malloc(SERIAL_FEATURES_PER_IMAGE * sizeof(feature));

	uint32_t currentFeature = 0;

#pragma omp parallel for num_threads(1)
	for (int imageIndex = 0; imageIndex < count; imageIndex++) {
		int id = omp_get_thread_num();
		int featureNum = 0;

		printf("thread %d calcing img %i\n", id, imageIndex);

		uint32_t* imagePtr = &(hostImageBuffer[imageIndex * IMAGE_SIZE * IMAGE_SIZE]);

		// convert to SAT
		for (int i = 1; i < IMAGE_SIZE; i++)
			imagePtr[i] += imagePtr[i - 1];

		for (int i = 1; i < IMAGE_SIZE; i++)
			imagePtr[i * IMAGE_SIZE] += imagePtr[(i - 1) * IMAGE_SIZE];

		for (int i = 1; i < IMAGE_SIZE; i++) {
			for (int j = 1; j < IMAGE_SIZE; j++) {
				imagePtr[j + IMAGE_SIZE * i] += imagePtr[j + IMAGE_SIZE * (i - 1)] + imagePtr[(j - 1) + IMAGE_SIZE * i] - imagePtr[(j - 1) + IMAGE_SIZE * (i - 1)];
			}
		}

		// find features
		for (uint32_t xSize = 4; xSize < IMAGE_SIZE; xSize++) {
			for (uint32_t ySize = 4; ySize < IMAGE_SIZE; ySize++) {

				// the total number of regions to process
				uint32_t xRegions = IMAGE_SIZE - xSize;
				uint32_t yRegions = IMAGE_SIZE - ySize;
				uint32_t numRegions = xRegions * yRegions;

				// number of regions each thread processes
				//uint32_t numIterations = (uint32_t)ceil((float)numRegions / stride);

				for (uint32_t i = 0; i < numRegions; i++) {
					uint8_t x1 = i % xRegions;
					uint8_t y1 = i / xRegions;
					uint8_t x2 = x1 + xSize;
					uint8_t y2 = y1 + ySize;

					// evaluate Haar-like features
					feature horizEdge = s_evalHorizEdge(imagePtr, x1, y1, x2, y2, THRESHOLD);
					if (horizEdge.mag != 0)
						threadFeatureBuffers[id][featureNum++] = horizEdge;

					feature vertEdge = s_evalVertEdge(imagePtr, x1, y1, x2, y2, THRESHOLD);
					if (vertEdge.mag != 0)
						threadFeatureBuffers[id][featureNum++] = vertEdge;

					feature horizLine = s_evalHorizLine(imagePtr, x1, y1, x2, y2, THRESHOLD);
					if (horizLine.mag != 0)
						threadFeatureBuffers[id][featureNum++] = horizLine;

					feature vertLine = s_evalVertLine(imagePtr, x1, y1, x2, y2, THRESHOLD);
					if (vertLine.mag != 0)
						threadFeatureBuffers[id][featureNum++] = vertLine;

					feature fourRectangle = s_evalFourRectangle(imagePtr, x1, y1, x2, y2, THRESHOLD);
					if (fourRectangle.mag != 0)
						threadFeatureBuffers[id][featureNum++] = fourRectangle;
				}
			}
		}

		uint32_t toCopyIdx;

		#pragma omp critical
		{
			toCopyIdx = currentFeature;
			currentFeature += featureNum;
		}

		memcpy(&(outFeatures[toCopyIdx]), threadFeatureBuffers[id], featureNum);
	}

	*numFeatures = currentFeature;
	outFeatures = (feature*)realloc(outFeatures, currentFeature * sizeof(feature));
	return outFeatures;
}
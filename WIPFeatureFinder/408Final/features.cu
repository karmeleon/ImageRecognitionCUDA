#include "features.cuh"

// 6 MiB should be enough
#define FEATURE_BUFFER_SIZE 6291456
#define FEATURES_IN_BUFFER FEATURE_BUFFER_SIZE / sizeof(uint64_t)
// fraction of free VRAM to use
#define FREE_VRAM_USAGE .6
// the average pixel difference to trigger a feature
#define THRESHOLD 1

__global__ void findFeatures(uint32_t* imageBuffer, uint64_t* featureBuffer) {
	uint32_t imgId = blockIdx.x;
	uint32_t* img = &(imageBuffer[IMAGE_SIZE * IMAGE_SIZE * imgId]);
	uint64_t* features = &(featureBuffer[FEATURES_IN_BUFFER * imgId]);

	// build the SAT
	scan2d(img);

	// copy the SAT to shared memory
	__shared__ uint32_t SAT[IMAGE_SIZE * IMAGE_SIZE];
	memcpy(SAT, img, IMAGE_SIZE * IMAGE_SIZE * sizeof(uint32_t));

	// find haar-like features
	haarfinder(SAT, features, THRESHOLD);
}

void printFeature(feature feat) {
	switch (feat.type) {
	case HEDGE:
		printf("Horiz Edge\n");
		break;
	case VEDGE:
		printf("Vert Edge\n");
		break;
	case HLINE:
		printf("Horiz Line\n");
		break;
	case VLINE:
		printf("Vert Line\n");
		break;
	case RECT4:
		printf("4-Rect\n");
		break;
	default:
		printf("Something else: %d\n", feat.type);
		break;
	}

	printf("Mag: %d\n", feat.mag);
	printf("(%d, %d) -> (%d, %d)\n", feat.x1, feat.y1, feat.x2, feat.y2);
}

uint64_t packFeature(feature unpacked) {
	// 7 bits each for x1, y1, x2, y2
	uint64_t out = 0;
	// bitwise operations are so satisfying
	out |= ((uint64_t)unpacked.x1) << 57;
	out |= ((uint64_t)unpacked.y1) << 50;
	out |= ((uint64_t)unpacked.x2) << 43;
	out |= ((uint64_t)unpacked.y2) << 36;
	out |= ((uint64_t)unpacked.type) << 33;
	((int32_t*)&out)[0] = ((int32_t)unpacked.mag);
	return out;
}

// unpacks a bitpacked CUDA feature into a C feature struct
inline feature unpackFeature(uint64_t packed) {
	feature out;
	out.x1 = (packed & 0xfe00000000000000) >> 57;
	out.y1 = (packed & 0x1fc000000000000) >> 50;
	out.x2 = (packed & 0x003f80000000000) >> 43;
	out.y2 = (packed & 0x00007f000000000) >> 36;
	out.type = (packed & 0xe00000000) >> 33;
	out.mag = (packed & 0xffffffff);
	return out;
}

feature* findFeatures(uint32_t* hostImageBuffer, uint32_t count, uint32_t* numFeatures) {
	// get the amount of vram we can allocate for this step
	size_t freeMem, totalMem;
	cudaMemGetInfo(&freeMem, &totalMem);
	printf("CUDA memory: Total: %d MB, free: %d MB\n", totalMem/1024/1024, freeMem/1024/1024);

	// compute number of images we can process at once
	int32_t concImages = freeMem * FREE_VRAM_USAGE / (FEATURE_BUFFER_SIZE + IMAGE_SIZE * IMAGE_SIZE * sizeof(uint32_t));
	printf("Computing %d images at once using %lu MB of memory\n", concImages, concImages * (FEATURE_BUFFER_SIZE + IMAGE_SIZE * IMAGE_SIZE * sizeof(uint32_t))/1024/1024);

	uint64_t* hostFeatureBuffer = (uint64_t*)malloc(FEATURE_BUFFER_SIZE * concImages);
	uint64_t* deviceFeatureBuffer;
	uint32_t* deviceImageBuffer;

	wbCheck(cudaMalloc((void**)&deviceFeatureBuffer, FEATURE_BUFFER_SIZE * concImages));
	wbCheck(cudaMalloc((void**)&deviceImageBuffer, IMAGE_SIZE * IMAGE_SIZE * concImages * sizeof(uint32_t)));

	uint32_t handledImages = 0;

	bool kernelRunning = true;
	bool willTerminate = false;

	// this will be expanded if necessary
	uint32_t finishedFeatureBufferSize = concImages * FEATURES_IN_BUFFER / 2;
	uint32_t numFinishedFeatures = 0;
	feature* finishedFeatures = (feature*)malloc(finishedFeatureBufferSize * sizeof(feature));

	// yay c++11
	uint32_t numThreads = std::thread::hardware_concurrency();
	feature** threadFeatureBuffers = (feature**)malloc(numThreads * sizeof(feature*));

	for (uint8_t i = 0; i < numThreads; i++) {
		threadFeatureBuffers[i] = (feature*)malloc(FEATURES_IN_BUFFER * sizeof(feature));
		memset(threadFeatureBuffers[i], 0, FEATURES_IN_BUFFER * sizeof(feature));
	}

	// the CUDA part

	// clear feature buffer and copy first batch of images to device
	wbCheck(cudaMemset(deviceFeatureBuffer, 0, FEATURE_BUFFER_SIZE * concImages));
	wbCheck(cudaMemcpy(&(deviceImageBuffer[handledImages]), hostImageBuffer, IMAGE_SIZE * IMAGE_SIZE * min(concImages, count - handledImages) * sizeof(uint32_t), cudaMemcpyHostToDevice));

	dim3 dimGrid(min(concImages, count), 1, 1);
	// one warp per block, we can crank this later if it helps
	dim3 dimBlock(32, 1, 1);

	findFeatures <<<dimGrid, dimBlock>>>(deviceImageBuffer, deviceFeatureBuffer);

	handledImages += min(concImages, count - handledImages);

	do {
		// copy feature buffer from device
		// cudaMemcpy blocks until the previous kernel finishes
		wbCheck(cudaMemcpy(hostFeatureBuffer, deviceFeatureBuffer, FEATURE_BUFFER_SIZE * concImages, cudaMemcpyDeviceToHost));

		willTerminate = !kernelRunning;

		// if there are more images to analyze, start them doing so
		if (handledImages < count) {
			// clear feature buffer and copy next batch of images to device
			wbCheck(cudaMemset(deviceFeatureBuffer, 0, FEATURE_BUFFER_SIZE * concImages));
			wbCheck(cudaMemcpy(&(deviceImageBuffer[handledImages]), hostImageBuffer, IMAGE_SIZE * IMAGE_SIZE * min(concImages, count - handledImages) * sizeof(uint32_t), cudaMemcpyHostToDevice));

			dim3 dimGrid(min(concImages, count - handledImages), 1, 1);
			// one warp per block, we can crank this later if it helps
			dim3 dimBlock(32, 1, 1);

			findFeatures <<<dimGrid, dimBlock>>>(deviceImageBuffer, deviceFeatureBuffer);

			handledImages += min(concImages, count - handledImages);
		} else
			kernelRunning = false;

		// convert bitpacked features into feature structs
		#pragma omp parallel for
		for (int32_t i = 0; i < concImages; i++) {
			uint32_t id = omp_get_thread_num();
			for (int32_t j = 0; j < FEATURES_IN_BUFFER; j++) {
				uint64_t packed = hostFeatureBuffer[j + FEATURES_IN_BUFFER * i];
				if (packed == 0) {
					// there are no more features in this buffer, dump it into the combined buffer
					#pragma omp critical
					{
						int32_t spaceInBuffer = finishedFeatureBufferSize - numFinishedFeatures;
						// make the combined buffer bigger if necessary
						if (spaceInBuffer < j)
							finishedFeatures = (feature*)realloc(finishedFeatures, (finishedFeatureBufferSize + concImages * FEATURES_IN_BUFFER / 2) * sizeof(feature));
						memcpy(&(finishedFeatures[numFinishedFeatures]), threadFeatureBuffers[id], j * sizeof(feature));
						numFinishedFeatures += j;
					}
					break;
				}

				// unpack the feature and stick it in this thread's buffer
				threadFeatureBuffers[id][j] = unpackFeature(packed);
			}
			// clear this thread's feature buffer
			memset(threadFeatureBuffers[id], 0, FEATURES_IN_BUFFER * sizeof(feature));
		}

		// clear the host feature buffer
		memset(hostFeatureBuffer, 0, FEATURE_BUFFER_SIZE * concImages);

	} while (!willTerminate);

	// cleanup, cleanup, everybody everywhere

	// C cleanup
	finishedFeatures = (feature*)realloc(finishedFeatures, numFinishedFeatures * sizeof(feature));
	for (uint8_t i = 0; i < numThreads; i++)
		free(threadFeatureBuffers[i]);
	free(threadFeatureBuffers);

	free(hostFeatureBuffer);

	// CUDA cleanup
	wbCheck(cudaFree(deviceFeatureBuffer));
	wbCheck(cudaFree(deviceImageBuffer));

	*numFeatures = numFinishedFeatures;
	return finishedFeatures;
}